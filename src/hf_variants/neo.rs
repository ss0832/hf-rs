// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Nuclear-Electronic Orbital Hartree-Fock (NEO-HF).
//!
//! Selected protons are quantum particles described by their own Gaussian basis, simultaneously
//! with the electrons. The electrons form a closed-shell RHF determinant and the `N_p` quantum
//! protons a high-spin (all same spin) determinant. With spin-summed electronic density `P_e`
//! and protonic density `P_p` (occupation one), the energy is
//!
//! `E = Tr(P_e h_e) + 1/2 Tr(P_e G_e(P_e)) + Tr(P_p h_p) + 1/2 Tr(P_p G_p(P_p))
//!      - sum (mu nu|la si) P_e,mu nu P_p,la si + V_nn(classical)`
//!
//! with `G_e = J - K/2`, `G_p = J - K` (same spin), `h_e = T + V_classical` and
//! `h_p = T/M_p - V_classical` (protons are repelled by the classical nuclei). The two Fock
//! operators `F_e = h_e + G_e - J^{ep}(P_p)` and `F_p = h_p + G_p - J^{pe}(P_e)` are solved
//! simultaneously, with one DIIS extrapolation over both commutator residuals. The basis
//! centers of the quantum protons are fixed; no basis-center optimization is performed.

use crate::basis::{Atom, ContractedGaussian};
use crate::diis::CdiisEngine;
use crate::matrix::Matrix;
use super::stationary_rhf;
use crate::scf::{
    RhfConvergence, build_fock_matrix, compute_eri_tensor, compute_nuclear_repulsion,
    compute_one_electron_integrals, eri_index,
};
use rayon::prelude::*;

/// Proton mass in electron masses (CODATA 2018).
pub const PROTON_MASS_AU: f64 = 1836.15267343;

/// Controls for [`run_neo_hf_with_options`].
#[derive(Clone, Debug)]
pub struct NeoHfOptions {
    /// Quantum-particle mass in electron masses.
    pub proton_mass: f64,
    pub max_iterations: usize,
    /// Change of the total energy between iterations.
    pub energy_tol: f64,
    /// Frobenius norm of each orthonormal-basis commutator `X^T(FPS - SPF)X`.
    pub residual_tol: f64,
    /// Frobenius norm of the density change divided by the basis dimension.
    pub density_tol: f64,
}

impl Default for NeoHfOptions {
    fn default() -> Self {
        Self {
            proton_mass: PROTON_MASS_AU,
            max_iterations: 200,
            energy_tol: 1e-10,
            residual_tol: 1e-8,
            density_tol: 1e-10,
        }
    }
}

/// Results from a Nuclear-Electronic Orbital (NEO-HF) calculation.
#[derive(Clone, Debug)]
pub struct NEOHFResult {
    pub total_energy: f64,
    /// `Tr(P_e h_e) + 1/2 Tr(P_e G_e)`: electrons in the field of the classical nuclei.
    pub electronic_energy: f64,
    /// `Tr(P_p h_p) + 1/2 Tr(P_p G_p)`: protonic kinetic, classical-nucleus repulsion and
    /// proton-proton terms.
    pub protonic_energy: f64,
    /// Electron-proton Coulomb attraction `-Tr(P_e J^{ep})` (negative).
    pub electron_proton_coulomb: f64,
    pub classical_nuc_repulsion: f64,
    /// Protonic kinetic energy `Tr(P_p T)/M_p` (the Hellmann-Feynman derivative `dE/d(1/M_p)`
    /// times `1/M_p`).
    pub protonic_kinetic_energy: f64,
    pub electronic_density: Matrix,
    pub protonic_density: Matrix,
    pub electronic_mo_coefficients: Matrix,
    pub protonic_mo_coefficients: Matrix,
    pub electronic_orbital_energies: Vec<f64>,
    pub protonic_orbital_energies: Vec<f64>,
    pub n_electron_pairs: usize,
    pub n_quantum_protons: usize,
    pub residual_norm: f64,
    pub density_change: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// NEO-HF with the CODATA proton mass. `tol` is the energy-change criterion; the commutator and
/// density criteria are derived from it.
pub fn run_neo_hf(
    classical_atoms: &[Atom],
    electronic_basis: &[ContractedGaussian],
    protonic_basis: &[ContractedGaussian],
    charge: i32,
    n_quantum_protons: usize,
    max_iter: usize,
    tol: f64,
) -> Result<NEOHFResult, String> {
    run_neo_hf_with_options(
        classical_atoms,
        electronic_basis,
        protonic_basis,
        charge,
        n_quantum_protons,
        &NeoHfOptions {
            max_iterations: max_iter,
            energy_tol: tol,
            residual_tol: (tol.sqrt() * 1e-2).clamp(1e-11, 1e-6),
            density_tol: (tol.sqrt() * 1e-2).clamp(1e-11, 1e-6),
            ..Default::default()
        },
    )
}

/// Same-spin two-particle operator `G_ij = sum_kl P_kl [(ij|kl) - (ik|jl)]`.
fn same_spin_g(p: &Matrix, eri: &[f64]) -> Matrix {
    let n = p.rows;
    let mut g = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut v = 0.0;
            for k in 0..n {
                for l in 0..n {
                    v += p[(k, l)] * (eri[eri_index(i, j, k, l)] - eri[eri_index(i, k, j, l)]);
                }
            }
            g[(i, j)] = v;
            g[(j, i)] = v;
        }
    }
    g
}

fn trace_product(a: &Matrix, b: &Matrix) -> f64 {
    a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum()
}

fn norm(a: &Matrix) -> f64 {
    a.data.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Orthonormal-basis commutator residual `X^T (F P S - S P F) X`.
fn residual(x: &Matrix, f: &Matrix, p: &Matrix, s: &Matrix) -> Matrix {
    x.transpose()
        .dot(&f.dot(p).dot(s).sub(&s.dot(p).dot(f)))
        .dot(x)
}

/// Diagonalizes `F` separately in the occupied and virtual spaces of `c`; the density is unchanged.
fn canonicalize(c: &Matrix, f: &Matrix, n_occ: usize) -> (Matrix, Vec<f64>) {
    let rank = c.cols;
    let fm = c.transpose().dot(f).dot(c);
    let mut rotation = Matrix::identity(rank);
    let mut energies = vec![0.0; rank];
    for (start, end) in [(0, n_occ), (n_occ, rank)] {
        if start == end {
            continue;
        }
        let block = Matrix::from_vec(
            end - start,
            end - start,
            (start..end)
                .flat_map(|p| (start..end).map(move |q| (p, q)))
                .map(|(p, q)| fm[(p, q)])
                .collect(),
        );
        let (values, vectors) = block.eigh();
        for p in start..end {
            energies[p] = values[p - start];
            for q in start..end {
                rotation[(p, q)] = vectors[(p - start, q - start)];
            }
        }
    }
    (c.dot(&rotation), energies)
}

fn block_diagonal(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(a.rows + b.rows, a.cols + b.cols);
    for i in 0..a.rows {
        for j in 0..a.cols {
            out[(i, j)] = a[(i, j)];
        }
    }
    for i in 0..b.rows {
        for j in 0..b.cols {
            out[(a.rows + i, a.cols + j)] = b[(i, j)];
        }
    }
    out
}

fn split_block(m: &Matrix, offset: usize, size: usize) -> Matrix {
    Matrix::from_vec(
        size,
        size,
        (0..size)
            .flat_map(|i| (0..size).map(move |j| (i, j)))
            .map(|(i, j)| m[(offset + i, offset + j)])
            .collect(),
    )
}

/// High-spin protons in a fixed external field `h`, from the aufbau orbitals of `F(previous)`.
/// Returns the coefficients and the density.
fn solve_protons(
    h: &Matrix,
    s: &Matrix,
    x: &Matrix,
    eri: &[f64],
    n_occ: usize,
    previous: &Matrix,
    tol: f64,
) -> Result<(Matrix, Matrix), String> {
    let mut f = h.add(&same_spin_g(previous, eri));
    let mut diis = CdiisEngine::new(8);
    for _ in 0..500 {
        let c = x.dot(&x.transpose().dot(&f).dot(x).eigh().1);
        let p = c.occupied_density(n_occ, 1.0);
        let next = h.add(&same_spin_g(&p, eri));
        let r = residual(x, &next, &p, s);
        if norm(&r) <= tol {
            return Ok((c, p));
        }
        let ortho = x.transpose().dot(&next).dot(x);
        diis.push(ortho.clone(), r);
        let extrapolated = diis.extrapolate().unwrap_or(ortho);
        // Back to the AO basis: X^T F X = O  =>  F = S X O X^T S.
        f = s.dot(x).dot(&extrapolated).dot(&x.transpose()).dot(s);
    }
    Err("protonic SCF in a fixed electronic field did not converge".into())
}

/// Packed electron-proton Coulomb integrals `(mu nu|la si)`, `mu >= nu`, `la >= si`.
struct ElectronProtonCoulomb {
    ne: usize,
    np: usize,
    values: Vec<f64>,
}

impl ElectronProtonCoulomb {
    fn new(electronic: &[ContractedGaussian], protonic: &[ContractedGaussian]) -> Self {
        let (ne, np) = (electronic.len(), protonic.len());
        let pairs_p: Vec<(usize, usize)> =
            (0..np).flat_map(|l| (0..=l).map(move |s| (l, s))).collect();
        let pairs_e: Vec<(usize, usize)> =
            (0..ne).flat_map(|m| (0..=m).map(move |n| (m, n))).collect();
        let mut values = vec![0.0; pairs_e.len() * pairs_p.len()];
        values
            .par_chunks_mut(pairs_p.len())
            .zip(pairs_e.par_iter())
            .for_each(|(chunk, &(mu, nu))| {
                for (v, &(la, si)) in chunk.iter_mut().zip(&pairs_p) {
                    *v = crate::integrals::contracted_eri(
                        &electronic[mu],
                        &electronic[nu],
                        &protonic[la],
                        &protonic[si],
                    );
                }
            });
        Self { ne, np, values }
    }

    fn packed(p: &Matrix) -> Vec<f64> {
        (0..p.rows)
            .flat_map(|i| (0..=i).map(move |j| (i, j)))
            .map(|(i, j)| if i == j { p[(i, j)] } else { 2.0 * p[(i, j)] })
            .collect()
    }

    fn unpack(n: usize, packed: &[f64]) -> Matrix {
        let mut out = Matrix::zeros(n, n);
        let mut k = 0;
        for i in 0..n {
            for j in 0..=i {
                out[(i, j)] = packed[k];
                out[(j, i)] = packed[k];
                k += 1;
            }
        }
        out
    }

    /// `J^{ep}_{mu nu} = sum_{la si} (mu nu|la si) P_p,la si`.
    fn electronic_coulomb(&self, p_p: &Matrix) -> Matrix {
        let d = Self::packed(p_p);
        let packed: Vec<f64> = self
            .values
            .par_chunks(d.len())
            .map(|row| row.iter().zip(&d).map(|(a, b)| a * b).sum())
            .collect();
        Self::unpack(self.ne, &packed)
    }

    /// `J^{pe}_{la si} = sum_{mu nu} (mu nu|la si) P_e,mu nu`.
    fn protonic_coulomb(&self, p_e: &Matrix) -> Matrix {
        let d = Self::packed(p_e);
        let width = self.np * (self.np + 1) / 2;
        let mut packed = vec![0.0; width];
        for (row, w) in self.values.chunks(width).zip(&d) {
            for (out, v) in packed.iter_mut().zip(row) {
                *out += w * v;
            }
        }
        Self::unpack(self.np, &packed)
    }
}

/// Coupled closed-shell-electron / high-spin-proton NEO-HF.
///
/// `electronic_basis` may include functions centered on the quantum-proton positions;
/// only `classical_atoms` contribute nuclear potentials and classical repulsion. Atoms with
/// effective core potentials are rejected because the pseudopotential acts on electrons only.
pub fn run_neo_hf_with_options(
    classical_atoms: &[Atom],
    electronic_basis: &[ContractedGaussian],
    protonic_basis: &[ContractedGaussian],
    charge: i32,
    n_quantum_protons: usize,
    options: &NeoHfOptions,
) -> Result<NEOHFResult, String> {
    let (ne, np) = (electronic_basis.len(), protonic_basis.len());
    if ne == 0 || np == 0 || n_quantum_protons == 0 {
        return Err("NEO-HF requires electronic and protonic bases and at least one quantum proton".into());
    }
    if !options.proton_mass.is_finite() || options.proton_mass <= 0.0 {
        return Err("NEO-HF requires a finite positive quantum-particle mass".into());
    }
    if options.max_iterations == 0
        || [options.energy_tol, options.residual_tol, options.density_tol]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err("invalid NEO-HF convergence controls".into());
    }
    if crate::ecp::has_ecp(classical_atoms) {
        return Err("NEO-HF does not support effective core potentials".into());
    }
    let classical_charge: f64 = classical_atoms.iter().map(|a| a.charge).sum();
    let n_electrons =
        classical_charge.round() as i64 + n_quantum_protons as i64 - i64::from(charge);
    if n_electrons <= 0 || n_electrons % 2 != 0 {
        return Err(format!(
            "closed-shell NEO-HF requires a positive even electron count, got {n_electrons}"
        ));
    }
    solve_two_component(
        classical_atoms,
        electronic_basis,
        protonic_basis,
        n_electrons as usize / 2,
        n_quantum_protons,
        options,
    )
}

/// Closed-shell electrons (`n_pairs` pairs) and `n_quantum` high-spin positive particles of
/// mass `options.proton_mass`, shared by NEO-HF and positronic HF.
pub(super) fn solve_two_component(
    classical_atoms: &[Atom],
    electronic_basis: &[ContractedGaussian],
    protonic_basis: &[ContractedGaussian],
    n_pairs: usize,
    n_quantum_protons: usize,
    options: &NeoHfOptions,
) -> Result<NEOHFResult, String> {
    let (ne, np) = (electronic_basis.len(), protonic_basis.len());
    if n_pairs == 0 {
        return Err("two-component HF requires at least one electron pair".into());
    }
    let (s_e, _, _, h_e) = compute_one_electron_integrals(classical_atoms, electronic_basis);
    let (s_p, t_p, v_p, _) = compute_one_electron_integrals(classical_atoms, protonic_basis);
    let t_p = t_p.scale(1.0 / options.proton_mass);
    let h_p = t_p.sub(&v_p);
    let x_e = s_e.canonical_orthogonalizer(1e-8).map_err(|e| e.to_string())?;
    let x_p = s_p.canonical_orthogonalizer(1e-8).map_err(|e| e.to_string())?;
    let (rank_e, rank_p) = (x_e.cols, x_p.cols);
    if n_pairs > rank_e || n_quantum_protons > rank_p {
        return Err("NEO-HF occupations exceed the retained basis rank".into());
    }
    let eri_e = compute_eri_tensor(electronic_basis);
    let eri_p = compute_eri_tensor(protonic_basis);
    let ep = ElectronProtonCoulomb::new(electronic_basis, protonic_basis);
    let nuclear = compute_nuclear_repulsion(classical_atoms);

    // Block-coordinate descent: relax the electrons in the field of the frozen protons, then the
    // protons in the field of the frozen electrons. Each half step lowers the energy, so the
    // coupled DIIS below starts near a minimum instead of drifting to a coupled saddle point
    // (as it does from a core guess for HF with a quantum proton).
    let inner = RhfConvergence {
        energy: 1e-10,
        residual: 1e-7,
        density: 1e-8,
    };
    let mut p_p = Matrix::zeros(np, np);
    let (mut c_e, mut p_e, mut c_p) = (Matrix::zeros(ne, rank_e), Matrix::zeros(ne, ne), Matrix::zeros(np, rank_p));
    for _ in 0..100 {
        let field_e = h_e.sub(&ep.electronic_coulomb(&p_p));
        let electrons = stationary_rhf::solve_with_fock(&field_e, &s_e, 0.0, n_pairs, 200, inner, |p| {
            build_fock_matrix(&field_e, p, &eri_e, ne)
        })?;
        (c_e, p_e) = (electrons.coefficients, electrons.density);
        let field_p = h_p.sub(&ep.protonic_coulomb(&p_e));
        (c_p, p_p) =
            solve_protons(&field_p, &s_p, &x_p, &eri_p, n_quantum_protons, &p_p, inner.residual)?;
        let f_e = build_fock_matrix(&h_e, &p_e, &eri_e, ne).sub(&ep.electronic_coulomb(&p_p));
        if norm(&residual(&x_e, &f_e, &p_e, &s_e)) < 1e-4 {
            break;
        }
    }

    let mut diis = CdiisEngine::new(8);
    let mut previous: Option<f64> = None;
    let mut change = f64::INFINITY;
    let mut residual_norm = f64::INFINITY;
    for iteration in 1..=options.max_iterations {
        let f_e_self = build_fock_matrix(&h_e, &p_e, &eri_e, ne);
        let f_p_self = h_p.add(&same_spin_g(&p_p, &eri_p));
        let j_ep = ep.electronic_coulomb(&p_p);
        let j_pe = ep.protonic_coulomb(&p_e);
        let f_e = f_e_self.sub(&j_ep);
        let f_p = f_p_self.sub(&j_pe);
        let electronic_energy = 0.5 * trace_product(&p_e, &h_e.add(&f_e_self));
        let protonic_energy = 0.5 * trace_product(&p_p, &h_p.add(&f_p_self));
        let coupling = -trace_product(&p_e, &j_ep);
        let energy = electronic_energy + protonic_energy + coupling + nuclear;
        let r_e = residual(&x_e, &f_e, &p_e, &s_e);
        let r_p = residual(&x_p, &f_p, &p_p, &s_p);
        residual_norm = norm(&r_e).max(norm(&r_p));
        if previous.is_some_and(|e| (energy - e).abs() <= options.energy_tol)
            && residual_norm <= options.residual_tol
            && change <= options.density_tol
        {
            let (c_e, eps_e) = canonicalize(&c_e, &f_e, n_pairs);
            let (c_p, eps_p) = canonicalize(&c_p, &f_p, n_quantum_protons);
            return Ok(NEOHFResult {
                total_energy: energy,
                electronic_energy,
                protonic_energy,
                electron_proton_coulomb: coupling,
                classical_nuc_repulsion: nuclear,
                protonic_kinetic_energy: trace_product(&p_p, &t_p),
                electronic_density: p_e,
                protonic_density: p_p,
                electronic_mo_coefficients: c_e,
                protonic_mo_coefficients: c_p,
                electronic_orbital_energies: eps_e,
                protonic_orbital_energies: eps_p,
                n_electron_pairs: n_pairs,
                n_quantum_protons,
                residual_norm,
                density_change: change,
                iterations: iteration,
                converged: true,
            });
        }
        previous = Some(energy);
        let ortho = block_diagonal(
            &x_e.transpose().dot(&f_e).dot(&x_e),
            &x_p.transpose().dot(&f_p).dot(&x_p),
        );
        diis.push(ortho.clone(), block_diagonal(&r_e, &r_p));
        let next = diis.extrapolate().unwrap_or(ortho);
        c_e = x_e.dot(&split_block(&next, 0, rank_e).eigh().1);
        c_p = x_p.dot(&split_block(&next, rank_e, rank_p).eigh().1);
        let next_e = c_e.occupied_density(n_pairs, 2.0);
        let next_p = c_p.occupied_density(n_quantum_protons, 1.0);
        change = (norm(&next_e.sub(&p_e)) / ne as f64).max(norm(&next_p.sub(&p_p)) / np as f64);
        p_e = next_e;
        p_p = next_p;
    }
    Err(format!(
        "NEO-HF did not converge in {} iterations: residual {residual_norm:e}, density change {change:e}",
        options.max_iterations
    ))
}
