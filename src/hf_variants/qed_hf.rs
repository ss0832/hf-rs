// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Single-mode coherent-state Pauli-Fierz RHF, without the photon zero-point constant.
//! `H = H_e + omega a^+ a - sqrt(omega/2) (lambda.mu)(a+a^+)
//!      + (lambda.mu)^2/2`. Minimizing the photon displacement leaves half the
//! dipole variance: `E_cav = Tr(P q)/2 - Tr(P d P d)/4`, where
//! `d=<lambda.r>` and `q=<(lambda.r)^2>`. The quadratic integral is evaluated
//! directly, retaining contributions outside the finite AO orbital span.
//! `F = F_e + q/2 - d P d/2`. Frequency affects photon displacement/occupation,
//! while this single-product-state ground-state energy is frequency independent
//! at fixed lambda. Electron-photon correlation and polariton excitations are absent.

use super::{magnetic::moment, stationary_rhf};
use crate::{
    Matrix,
    basis::{Atom, ContractedGaussian},
    derivatives::diff_cg_1st,
    scf::{
        RhfConvergence, build_fock_matrix, compute_eri_tensor, compute_nuclear_repulsion,
        compute_one_electron_integrals,
    },
};

#[derive(Clone, Debug)]
pub struct QedHfOptions {
    /// Photon energy, strictly positive, in Hartree.
    pub cavity_frequency: f64,
    /// Length-gauge coupling in sqrt(Hartree)/Bohr.
    pub coupling: [f64; 3],
    pub max_iterations: usize,
    pub energy_tol: f64,
    pub residual_tol: f64,
    pub density_tol: f64,
}

impl Default for QedHfOptions {
    fn default() -> Self {
        Self {
            cavity_frequency: 0.1,
            coupling: [0., 0., 0.05],
            max_iterations: 200,
            energy_tol: 1e-12,
            residual_tol: 1e-10,
            density_tol: 1e-10,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QEDHFResult {
    pub total_energy: f64,
    /// Physical electronic expectation, excluding classical nuclear repulsion.
    pub electronic_energy: f64,
    pub nuclear_energy: f64,
    /// Expectation of the bare dipole self-energy, including the mean-dipole term.
    pub dipole_self_energy: f64,
    /// Net photon + bilinear + self-energy contribution: half the dipole variance.
    pub cavity_energy: f64,
    pub coherent_displacement: f64,
    pub mean_photon_number: f64,
    pub cavity_frequency_hartree: f64,
    pub coupling_vector: [f64; 3],
    /// Total molecular electric dipole (electrons plus effective nuclei), in e Bohr.
    pub expected_dipole_bohr: [f64; 3],
    pub mo_coefficients: Matrix,
    pub orbital_energies: Vec<f64>,
    pub density_matrix: Matrix,
    pub fock_matrix: Matrix,
    pub overlap_matrix: Matrix,
    pub n_occ: usize,
    pub residual_norm: f64,
    pub density_change: f64,
    pub iterations: usize,
    pub converged: bool,
}

fn pair_operators(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    coupling: [f64; 3],
) -> (f64, f64) {
    let (mut dipole, mut quadratic) = (0., 0.);
    for i in 0..3 {
        if coupling[i] != 0. {
            dipole += coupling[i] * moment(a, b, &[i], [0.; 3]);
            for j in 0..3 {
                if coupling[j] != 0. {
                    quadratic += coupling[i] * coupling[j] * moment(a, b, &[i, j], [0.; 3]);
                }
            }
        }
    }
    (dipole, quadratic)
}

/// Exact AO integrals of `lambda.r` and `(lambda.r)^2`, in that order.
pub fn qed_dipole_operators(
    basis: &[ContractedGaussian],
    coupling: [f64; 3],
) -> Result<(Matrix, Matrix), String> {
    if basis.is_empty() || coupling.iter().any(|v| !v.is_finite()) {
        return Err("QED dipole operators require a nonempty basis and finite coupling".into());
    }
    let n = basis.len();
    let mut dipole = Matrix::zeros(n, n);
    let mut quadratic = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let (d, q) = pair_operators(&basis[i], &basis[j], coupling);
            dipole[(i, j)] = d;
            dipole[(j, i)] = d;
            quadratic[(i, j)] = q;
            quadratic[(j, i)] = q;
        }
    }
    Ok((dipole, quadratic))
}

/// Model Fock matrix for a spin-summed RHF density and packed physical AO ERIs.
pub fn build_qed_hf_fock(
    h: &Matrix,
    p: &Matrix,
    eri: &[f64],
    dipole: &Matrix,
    quadratic: &Matrix,
) -> Matrix {
    build_fock_matrix(h, p, eri, h.rows)
        .add(&quadratic.scale(0.5))
        .sub(&dipole.dot(p).dot(dipole).scale(0.5))
}

pub fn run_qed_hf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    cavity_frequency_hartree: f64,
    coupling_vector: [f64; 3],
    max_iter: usize,
    tol: f64,
) -> Result<QEDHFResult, String> {
    run_qed_hf_with_options(
        atoms,
        basis,
        charge,
        &QedHfOptions {
            cavity_frequency: cavity_frequency_hartree,
            coupling: coupling_vector,
            max_iterations: max_iter,
            energy_tol: tol,
            residual_tol: (tol.sqrt() * 1e-2).clamp(1e-11, 1e-8),
            density_tol: (tol.sqrt() * 1e-2).clamp(1e-11, 1e-10),
        },
    )
}

pub fn run_qed_hf_with_options(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    options: &QedHfOptions,
) -> Result<QEDHFResult, String> {
    if !options.cavity_frequency.is_finite() || options.cavity_frequency <= 0. {
        return Err("QED-HF requires a finite positive cavity frequency".into());
    }
    let ne = atoms.iter().map(|a| a.charge).sum::<f64>().round() as i64 - i64::from(charge);
    if ne <= 0 || ne % 2 != 0 {
        return Err("coherent-state QED-RHF requires a positive even electron count".into());
    }
    let n_occ = ne as usize / 2;
    let (d, q) = qed_dipole_operators(basis, options.coupling)?;
    let (s, _, _, h) = compute_one_electron_integrals(atoms, basis);
    let eri = compute_eri_tensor(basis);
    let nuclear = compute_nuclear_repulsion(atoms);
    let effective_h = h.add(&q.scale(0.5));
    let solved = stationary_rhf::solve_with_fock(
        &effective_h,
        &s,
        nuclear,
        n_occ,
        options.max_iterations,
        RhfConvergence {
            energy: options.energy_tol,
            residual: options.residual_tol,
            density: options.density_tol,
        },
        |p| build_qed_hf_fock(&h, p, &eri, &d, &q),
    )?;
    let p = &solved.density;
    let physical_fock = build_fock_matrix(&h, p, &eri, basis.len());
    let electronic_energy = 0.5
        * p.data
            .iter()
            .zip(h.add(&physical_fock).data)
            .map(|(p, f)| p * f)
            .sum::<f64>();
    let cavity_energy = 0.5 * p.data.iter().zip(&q.data).map(|(p, q)| p * q).sum::<f64>()
        - 0.25 * p.dot(&d).dot(p).dot(&d).trace();
    let props = crate::properties::compute_molecular_properties(atoms, basis, p, &s);
    let weighted_dipole: f64 = options
        .coupling
        .iter()
        .zip(props.dipole_moment_au)
        .map(|(l, mu)| l * mu)
        .sum();
    let displacement = weighted_dipole / (2. * options.cavity_frequency).sqrt();
    Ok(QEDHFResult {
        total_energy: solved.total_energy,
        electronic_energy,
        nuclear_energy: nuclear,
        dipole_self_energy: cavity_energy + 0.5 * weighted_dipole.powi(2),
        cavity_energy,
        coherent_displacement: displacement,
        mean_photon_number: displacement.powi(2),
        cavity_frequency_hartree: options.cavity_frequency,
        coupling_vector: options.coupling,
        expected_dipole_bohr: props.dipole_moment_au,
        mo_coefficients: solved.coefficients,
        orbital_energies: solved.orbital_energies,
        density_matrix: solved.density,
        fock_matrix: solved.fock,
        overlap_matrix: s,
        n_occ,
        residual_norm: solved.residual_norm,
        density_change: solved.density_change,
        iterations: solved.iterations,
        converged: true,
    })
}

/// Variational analytic nuclear gradient with exact moment, ECP and full-model Pulay terms.
pub fn compute_qed_hf_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    result: &QEDHFResult,
) -> Result<Vec<[f64; 3]>, String> {
    let n = basis.len();
    if !result.converged || result.mo_coefficients.rows != n || result.density_matrix.rows != n {
        return Err("QED-HF gradient requires a matching converged reference".into());
    }
    if result.mo_coefficients.cols != n {
        return Err(
            "QED-HF gradients after overlap rank truncation require projector response".into(),
        );
    }
    let (d, _) = qed_dipole_operators(basis, result.coupling_vector)?;
    let p = &result.density_matrix;
    let pdp = p.dot(&d).dot(p);
    let c = &result.mo_coefficients;
    let mut w = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            w[(i, j)] = (0..result.n_occ)
                .map(|k| 2. * result.orbital_energies[k] * c[(i, k)] * c[(j, k)])
                .sum();
        }
    }
    let mut gradient = crate::gradient::compute_gradient_from_density(atoms, basis, p, &w);
    for (atom, g) in gradient.iter_mut().enumerate() {
        for (axis, value) in g.iter_mut().enumerate() {
            for i in 0..n {
                for j in 0..n {
                    let (mut dd, mut dq) = (0., 0.);
                    if basis[i].atom_index == atom {
                        for term in diff_cg_1st(&basis[i], axis).terms {
                            let (a, b) = pair_operators(&term, &basis[j], result.coupling_vector);
                            dd += a;
                            dq += b;
                        }
                    }
                    if basis[j].atom_index == atom {
                        for term in diff_cg_1st(&basis[j], axis).terms {
                            let (a, b) = pair_operators(&basis[i], &term, result.coupling_vector);
                            dd += a;
                            dq += b;
                        }
                    }
                    *value += 0.5 * p[(i, j)] * dq - 0.5 * pdp[(i, j)] * dd;
                }
            }
        }
    }
    Ok(gradient)
}
