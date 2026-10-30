// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Divide-and-conquer Hartree-Fock (DC-HF).
//!
//! Yang and Lee, J. Chem. Phys. 103, 5674 (1995); DC-HF as in Akama, Kobayashi and Nakai,
//! J. Comput. Chem. 28, 2003 (2007). Each subsystem `a` has a central atom set `S(a)` and a
//! buffer `B(a)` (all other atoms within `buffer_radius_bohr` of a central atom). On its
//! localization region `L(a) = S(a) + B(a)` the AO submatrices of the global Fock operator
//! `F[P]` and overlap define `F^a C^a = S^a C^a eps^a`. With one Fermi level `mu`,
//!
//! `P = sum_a sum_k f(eps^a_k) (p^a o C^a_k C^a_k^T)`, `f(e) = 2 / (1 + exp(beta (e - mu)))`,
//!
//! where `o` is the elementwise product and the partition matrix is `p^a_mn = 1` if both AOs
//! belong to `S(a)`, `1/2` if one belongs to `S(a)` and the other to `B(a)`, and 0 otherwise.
//! `mu` solves `Tr(P S) = N`. The self-consistent density is the fixed point `P = P_out[P]`,
//! accelerated by Pulay mixing of the density residual, and the energy is
//! `E = Tr(P (h + F[P]))/2 + E_nuc`. Because `P` is assembled from subsystem orbitals and is
//! not idempotent, `E` is not variational and may lie below RHF; when every buffer contains the
//! whole molecule DC-HF reduces to finite-temperature RHF. The global Fock build uses the full
//! packed ERI tensor, so this implementation is not linear scaling. Analytic nuclear gradients
//! would require the response of the subsystem eigenproblems and Fermi level and are not
//! provided.

use crate::basis::{Atom, ContractedGaussian};
use crate::matrix::Matrix;
use crate::scf::{
    build_fock_matrix, compute_eri_tensor, compute_nuclear_repulsion,
    compute_one_electron_integrals,
};
use rayon::prelude::*;

/// A subsystem of the divide-and-conquer partition.
#[derive(Clone, Debug)]
pub struct DCSubsystem {
    /// Central atoms `S(a)`.
    pub core_atoms: Vec<usize>,
    /// Buffer atoms `B(a)`.
    pub buffer_atoms: Vec<usize>,
    /// `L(a) = S(a) + B(a)`, sorted.
    pub extended_atoms: Vec<usize>,
    /// Global AO indices of `L(a)`, increasing.
    pub extended_ao_indices: Vec<usize>,
    /// Whether each AO of `extended_ao_indices` belongs to a central atom.
    pub central_ao: Vec<bool>,
}

impl DCSubsystem {
    /// Partition-matrix element for local AOs `i`, `j`: 1, 1/2 or 0.
    pub fn partition(&self, i: usize, j: usize) -> f64 {
        0.5 * (f64::from(u8::from(self.central_ao[i])) + f64::from(u8::from(self.central_ao[j])))
    }
}

/// Calculation options for DC-HF.
#[derive(Clone, Debug)]
pub struct DCOptions {
    /// Central atom sets; they must cover every atom exactly once. `None` uses consecutive
    /// groups of `atoms_per_subsystem` atoms.
    pub central_regions: Option<Vec<Vec<usize>>>,
    pub atoms_per_subsystem: usize,
    /// Buffer radius in bohr (distance from any central atom).
    pub buffer_radius_bohr: f64,
    /// Fermi-function inverse temperature `beta` in 1/Hartree.
    pub inverse_temperature: f64,
    pub max_iterations: usize,
    /// Energy change between iterations.
    pub energy_threshold: f64,
    /// Largest density-residual element `|P_out - P_in|`.
    pub density_threshold: f64,
    /// Number of Pulay mixing vectors.
    pub diis_size: usize,
}

impl Default for DCOptions {
    fn default() -> Self {
        Self {
            central_regions: None,
            atoms_per_subsystem: 1,
            buffer_radius_bohr: 6.0,
            inverse_temperature: 200.0,
            max_iterations: 300,
            energy_threshold: 1e-10,
            density_threshold: 1e-8,
            diis_size: 8,
        }
    }
}

/// Results of a divide-and-conquer Hartree-Fock calculation.
#[derive(Clone, Debug)]
pub struct DCHFResult {
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_energy: f64,
    /// Fermi level `mu` in Hartree.
    pub chemical_potential: f64,
    /// Self-consistent DC density matrix (spin-summed).
    pub density_matrix: Matrix,
    /// `Tr(P S)`.
    pub electron_count: f64,
    /// Largest element of the final density residual.
    pub density_residual: f64,
    pub subsystems: Vec<DCSubsystem>,
    pub n_subsystems: usize,
    pub iterations: usize,
    pub converged: bool,
}

/// Builds subsystems from central atom sets and a buffer radius.
pub fn partition_molecular_system(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    central_regions: &[Vec<usize>],
    buffer_radius_bohr: f64,
) -> Result<Vec<DCSubsystem>, String> {
    if !buffer_radius_bohr.is_finite() || buffer_radius_bohr < 0.0 {
        return Err("DC buffer radius must be finite and nonnegative".into());
    }
    let mut seen = vec![false; atoms.len()];
    for region in central_regions {
        if region.is_empty() {
            return Err("DC central regions must be nonempty".into());
        }
        for &a in region {
            if a >= atoms.len() || std::mem::replace(&mut seen[a], true) {
                return Err("DC central regions must cover each atom exactly once".into());
            }
        }
    }
    if seen.iter().any(|s| !s) {
        return Err("DC central regions must cover each atom exactly once".into());
    }
    let distance = |a: usize, b: usize| -> f64 {
        (0..3).map(|k| (atoms[a].position[k] - atoms[b].position[k]).powi(2)).sum::<f64>().sqrt()
    };
    Ok(central_regions
        .iter()
        .map(|region| {
            let mut core_atoms = region.clone();
            core_atoms.sort_unstable();
            let buffer_atoms: Vec<usize> = (0..atoms.len())
                .filter(|a| {
                    !core_atoms.contains(a)
                        && core_atoms.iter().any(|&c| distance(*a, c) <= buffer_radius_bohr)
                })
                .collect();
            let mut extended_atoms: Vec<usize> =
                core_atoms.iter().chain(&buffer_atoms).copied().collect();
            extended_atoms.sort_unstable();
            let extended_ao_indices: Vec<usize> = (0..basis.len())
                .filter(|&m| extended_atoms.contains(&basis[m].atom_index))
                .collect();
            let central_ao = extended_ao_indices
                .iter()
                .map(|&m| core_atoms.contains(&basis[m].atom_index))
                .collect();
            DCSubsystem {
                core_atoms,
                buffer_atoms,
                extended_atoms,
                extended_ao_indices,
                central_ao,
            }
        })
        .collect())
}

fn fermi(beta: f64, energy: f64, mu: f64) -> f64 {
    // 2 / (1 + exp(x)) without overflow.
    let x = beta * (energy - mu);
    if x > 0.0 {
        2.0 * (-x).exp() / (1.0 + (-x).exp())
    } else {
        2.0 / (1.0 + x.exp())
    }
}

struct Local {
    energies: Vec<f64>,
    coefficients: Matrix,
    /// Partition-weighted norms `q_k = sum p_mn C_mk S_mn C_nk`.
    weights: Vec<f64>,
}

/// Output density `P_out[F]` and Fermi level for a given global Fock matrix.
fn output_density(
    fock: &Matrix,
    overlap: &Matrix,
    subsystems: &[DCSubsystem],
    beta: f64,
    n_electrons: f64,
) -> Result<(Matrix, f64), String> {
    let locals: Vec<Local> = subsystems
        .par_iter()
        .map(|sub| {
            let idx = &sub.extended_ao_indices;
            let m = idx.len();
            let sub_matrix = |a: &Matrix| {
                Matrix::from_vec(
                    m,
                    m,
                    idx.iter().flat_map(|&i| idx.iter().map(move |&j| a[(i, j)])).collect(),
                )
            };
            let s = sub_matrix(overlap);
            let x = s.canonical_orthogonalizer(1e-8).map_err(|e| e.to_string())?;
            let (energies, u) = x.transpose().dot(&sub_matrix(fock)).dot(&x).eigh();
            let coefficients = x.dot(&u);
            let weights = (0..coefficients.cols)
                .map(|k| {
                    let mut q = 0.0;
                    for i in 0..m {
                        for j in 0..m {
                            q += sub.partition(i, j) * coefficients[(i, k)] * s[(i, j)] * coefficients[(j, k)];
                        }
                    }
                    q
                })
                .collect();
            Ok(Local { energies, coefficients, weights })
        })
        .collect::<Result<_, String>>()?;
    let count = |mu: f64| -> f64 {
        locals
            .iter()
            .map(|l| l.energies.iter().zip(&l.weights).map(|(e, q)| fermi(beta, *e, mu) * q).sum::<f64>())
            .sum()
    };
    let all = locals.iter().flat_map(|l| l.energies.iter().copied());
    let (mut lo, mut hi) = all.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), e| (a.min(e), b.max(e)));
    lo -= 1.0;
    hi += 1.0;
    if !(count(lo) < n_electrons && count(hi) > n_electrons) {
        return Err("DC Fermi level cannot be bracketed: partition-weighted occupations do not span N".into());
    }
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if count(mid) < n_electrons {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mu = 0.5 * (lo + hi);
    let n = fock.rows;
    let mut density = Matrix::zeros(n, n);
    for (sub, local) in subsystems.iter().zip(&locals) {
        let idx = &sub.extended_ao_indices;
        for (k, &e) in local.energies.iter().enumerate() {
            let f = fermi(beta, e, mu);
            if f == 0.0 {
                continue;
            }
            for (i, &gi) in idx.iter().enumerate() {
                let ci = local.coefficients[(i, k)];
                for (j, &gj) in idx.iter().enumerate() {
                    let p = sub.partition(i, j);
                    if p != 0.0 {
                        density[(gi, gj)] += f * p * ci * local.coefficients[(j, k)];
                    }
                }
            }
        }
    }
    Ok((density, mu))
}

/// Runs divide-and-conquer Hartree-Fock.
pub fn run_dc_hf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    options: &DCOptions,
) -> Result<DCHFResult, String> {
    let n = basis.len();
    if n == 0 || atoms.is_empty() {
        return Err("DC-HF requires atoms and a basis".into());
    }
    if !(options.inverse_temperature.is_finite() && options.inverse_temperature > 0.0)
        || options.max_iterations == 0
        || options.diis_size == 0
        || options.atoms_per_subsystem == 0
        || [options.energy_threshold, options.density_threshold]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err("invalid DC-HF controls".into());
    }
    let n_electrons = atoms.iter().map(|a| a.charge).sum::<f64>().round() - f64::from(charge);
    if n_electrons <= 0.0 {
        return Err(format!("invalid electron count {n_electrons} for DC-HF"));
    }
    let regions = options.central_regions.clone().unwrap_or_else(|| {
        (0..atoms.len())
            .collect::<Vec<_>>()
            .chunks(options.atoms_per_subsystem)
            .map(|c| c.to_vec())
            .collect()
    });
    let subsystems = partition_molecular_system(atoms, basis, &regions, options.buffer_radius_bohr)?;
    let nuclear = compute_nuclear_repulsion(atoms);
    let (s, _, _, h) = compute_one_electron_integrals(atoms, basis);
    let eri = compute_eri_tensor(basis);
    let beta = options.inverse_temperature;
    let energy = |p: &Matrix| -> f64 {
        let f = build_fock_matrix(&h, p, &eri, n);
        0.5 * p.data.iter().zip(h.add(&f).data).map(|(p, x)| p * x).sum::<f64>()
    };
    let mut density = match crate::sad::compute_sad_density(atoms, basis, charge, &s) {
        Ok(p) => p,
        Err(_) => Matrix::zeros(n, n),
    };
    let (mut inputs, mut residuals): (Vec<Matrix>, Vec<Matrix>) = (Vec::new(), Vec::new());
    let mut previous: Option<f64> = None;
    let mut residual_max = f64::INFINITY;
    let mut mu = 0.0;
    for iteration in 1..=options.max_iterations {
        let fock = build_fock_matrix(&h, &density, &eri, n);
        let (out, fermi_level) = output_density(&fock, &s, &subsystems, beta, n_electrons)?;
        mu = fermi_level;
        let residual = out.sub(&density);
        residual_max = residual.data.iter().fold(0f64, |m, v| m.max(v.abs()));
        let e = energy(&density);
        if residual_max <= options.density_threshold
            && previous.is_some_and(|p: f64| (e - p).abs() <= options.energy_threshold)
        {
            let electron_count = density.data.iter().zip(&s.data).map(|(p, s)| p * s).sum();
            return Ok(DCHFResult {
                total_energy: e + nuclear,
                electronic_energy: e,
                nuclear_energy: nuclear,
                chemical_potential: mu,
                density_matrix: density,
                electron_count,
                density_residual: residual_max,
                n_subsystems: subsystems.len(),
                subsystems,
                iterations: iteration,
                converged: true,
            });
        }
        previous = Some(e);
        // Pulay mixing: P_next = sum c_i P_out_i with sum c_i = 1 minimizing |sum c_i R_i|.
        inputs.push(out);
        residuals.push(residual);
        if inputs.len() > options.diis_size {
            inputs.remove(0);
            residuals.remove(0);
        }
        let m = residuals.len();
        let mut b = Matrix::zeros(m + 1, m + 1);
        for i in 0..m {
            for j in 0..m {
                b[(i, j)] = residuals[i].data.iter().zip(&residuals[j].data).map(|(x, y)| x * y).sum();
            }
            b[(i, m)] = -1.0;
            b[(m, i)] = -1.0;
        }
        let mut rhs = vec![0.0; m + 1];
        rhs[m] = -1.0;
        density = match b.lu_decompose() {
            Some(lu) => {
                let c = lu.solve(&rhs);
                inputs.iter().zip(&c).fold(Matrix::zeros(n, n), |acc, (x, c)| acc.add(&x.scale(*c)))
            }
            None => {
                inputs.drain(..m - 1);
                residuals.drain(..m - 1);
                inputs[0].clone()
            }
        };
    }
    Err(format!(
        "DC-HF did not converge in {} iterations: density residual {residual_max:e} (mu = {mu})",
        options.max_iterations
    ))
}

/// DC-HF nuclear gradients are not available: the DC energy is not stationary with respect to
/// the subsystem orbitals or the Fermi level, so a Hellmann-Feynman/Pulay formula is not exact.
pub fn compute_dc_hf_gradient(
    _atoms: &[Atom],
    _basis: &[ContractedGaussian],
    _result: &DCHFResult,
) -> Result<Vec<[f64; 3]>, String> {
    Err("DC-HF analytic gradients require subsystem and Fermi-level response, which is not implemented".into())
}
