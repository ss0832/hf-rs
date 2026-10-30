// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Exchange-only OEP by direct energy minimization in an explicit finite local-potential basis.
//! `v_s = v_ext + (1-1/N) J[P_ref] + sum_b c_b exp(-zeta_b |r-R_b|^2)`.
//! The fixed reference is RHF. The determinant of the lowest eigenvectors of this local
//! Hamiltonian minimizes the physical HF energy over its potential coefficients.
//! `dE/dc_b = -4 sum_ia F_HF,ai G_b,ai/(eps_a-eps_i)` includes exact orbital response.
//! No denominator is clipped. A closing occupied/virtual gap fails explicitly.
//! This is finite-potential-basis OEP, distinct from LHF/CEDA; it is not a unique
//! complete-basis potential or a nuclear-response implementation.

use super::{
    local_potential::gaussian_potential_matrix,
    orbital_optimizer::{Controls, Evaluation, max_abs, try_minimize},
};
use crate::{
    Atom, ContractedGaussian, ConvAlgo, HFResult, Matrix,
    scf::{build_fock_matrix, contract_jk, run_scf_general},
};

#[derive(Clone, Debug)]
pub struct OepOptions {
    /// One atom-centered s-type potential function per atom and exponent (Bohr^-2).
    pub exponents: Vec<f64>,
    pub max_iterations: usize,
    pub reference_max_iterations: usize,
    pub energy_tol: f64,
    pub gradient_tol: f64,
    /// Smallest supported occupied/virtual gap in Hartree; no replacement denominator is used.
    pub minimum_gap: f64,
    pub max_parameter_step: f64,
    /// Additional AO potential matrices only; excludes the borrowed RHF reference.
    pub potential_budget_bytes: usize,
}

impl Default for OepOptions {
    fn default() -> Self {
        Self {
            exponents: vec![0.5, 2.],
            max_iterations: 1000,
            reference_max_iterations: 300,
            energy_tol: 1e-12,
            gradient_tol: 1e-8,
            minimum_gap: 1e-8,
            max_parameter_step: 0.25,
            potential_budget_bytes: 256 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OepResult {
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_energy: f64,
    pub reference_hf_energy: f64,
    pub density_matrix: Matrix,
    pub reference_density_matrix: Matrix,
    pub mo_coefficients: Matrix,
    pub orbital_energies: Vec<f64>,
    pub ks_hamiltonian: Matrix,
    pub hf_fock_matrix: Matrix,
    /// Ordering is atom-major, then exponents in options.exponents.
    pub parameters: Vec<f64>,
    pub parameter_gradient: Vec<f64>,
    pub gradient_norm: f64,
    pub ks_residual_norm: f64,
    pub minimum_gap: Option<f64>,
    pub fermi_amaldi_factor: f64,
    pub iterations: usize,
    pub converged: bool,
}

struct Point {
    parameters: Vec<f64>,
    density: Matrix,
    coefficients: Matrix,
    energies: Vec<f64>,
    ks: Matrix,
    fock: Matrix,
    gap: Option<f64>,
}

struct Problem<'a> {
    hf: &'a HFResult,
    baseline: Matrix,
    operators: Vec<Matrix>,
    minimum_gap: f64,
}

impl Problem<'_> {
    fn evaluate(&self, parameters: Vec<f64>) -> Result<Evaluation<Point>, String> {
        if parameters.len() != self.operators.len() || parameters.iter().any(|v| !v.is_finite()) {
            return Err("OEP requires one finite parameter per potential function".into());
        }
        let hf = self.hf;
        let x = &hf.mo_coefficients;
        let mut ks = self.baseline.clone();
        for (value, operator) in parameters.iter().zip(&self.operators) {
            for (entry, g) in ks.data.iter_mut().zip(&operator.data) {
                *entry += value * g;
            }
        }
        let (energies, u) = x.transpose().dot(&ks).dot(x).eigh();
        let c = x.dot(&u);
        let no = hf.n_occ;
        let gap = (no < c.cols).then(|| energies[no] - energies[no - 1]);
        if gap.is_some_and(|gap| gap < self.minimum_gap) {
            return Err(format!(
                "OEP occupied/virtual gap {:?} is below the supported threshold {}; degenerate response is not implemented",
                gap, self.minimum_gap
            ));
        }
        let density = c.occupied_density(no, 2.);
        let fock = build_fock_matrix(&hf.core_hamiltonian, &density, &hf.eri, c.rows);
        let energy = hf.nuclear_repulsion_energy
            + 0.5
                * density
                    .data
                    .iter()
                    .zip(hf.core_hamiltonian.add(&fock).data)
                    .map(|(p, f)| p * f)
                    .sum::<f64>();
        let fm = c.transpose().dot(&fock).dot(&c);
        let mut gradient = Vec::with_capacity(parameters.len());
        let mut diagonal = Vec::with_capacity(parameters.len());
        for operator in &self.operators {
            let gm = c.transpose().dot(operator).dot(&c);
            let (mut g, mut d) = (0., 0.);
            for a in no..c.cols {
                for i in 0..no {
                    let inverse_gap = 1. / (energies[a] - energies[i]);
                    g -= 4. * fm[(a, i)] * gm[(a, i)] * inverse_gap;
                    d += 4. * gm[(a, i)].powi(2) * inverse_gap;
                }
            }
            gradient.push(g);
            diagonal.push(d);
        }
        if !energy.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite OEP energy or potential derivative".into());
        }
        Ok(Evaluation {
            state: Point {
                parameters,
                density,
                coefficients: c,
                energies,
                ks,
                fock,
                gap,
            },
            value: energy,
            gradient,
            diagonal,
        })
    }
}

fn validate(options: &OepOptions) -> Result<(), String> {
    if options.exponents.is_empty()
        || options.exponents.iter().any(|v| !v.is_finite() || *v <= 0.)
        || options.max_iterations == 0
        || options.reference_max_iterations == 0
        || [
            options.energy_tol,
            options.gradient_tol,
            options.minimum_gap,
            options.max_parameter_step,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err("OEP requires positive finite exponents/tolerances/step bounds and positive iteration budgets".into());
    }
    for i in 0..options.exponents.len() {
        for j in 0..i {
            if options.exponents[i] == options.exponents[j] {
                return Err("duplicate OEP potential exponents are not supported".into());
            }
        }
    }
    Ok(())
}

pub fn run_oep_hf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    options: &OepOptions,
) -> Result<OepResult, String> {
    validate(options)?;
    let ne = atoms.iter().map(|a| a.charge).sum::<f64>().round() as i64 - i64::from(charge);
    if ne <= 0 || ne % 2 != 0 {
        return Err("exchange-only OEP currently requires a positive even electron count".into());
    }
    let hf = run_scf_general(
        atoms,
        basis,
        charge,
        options.reference_max_iterations,
        options.energy_tol.min(1e-12),
        ConvAlgo::Adiissoscf,
        None,
    )?;
    run_oep_hf_from_reference(atoms, basis, &hf, options)
}

/// Borrows packed AO ERIs from a converged closed-shell reference.
pub fn run_oep_hf_from_reference(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    options: &OepOptions,
) -> Result<OepResult, String> {
    let problem = prepare(atoms, basis, hf, options)?;
    let start = problem.evaluate(vec![0.; problem.operators.len()])?;
    let minimum = try_minimize(
        start,
        &Controls {
            max_iterations: options.max_iterations,
            gradient_tol: options.gradient_tol,
            value_tol: options.energy_tol,
            max_step: options.max_parameter_step,
        },
        |point, change| {
            problem.evaluate(
                point
                    .parameters
                    .iter()
                    .zip(change)
                    .map(|(x, d)| x + d)
                    .collect(),
            )
        },
    )?;
    if !minimum.converged {
        return Err(format!(
            "OEP did not converge in {} iterations: parameter gradient {:e}",
            minimum.iterations,
            max_abs(&minimum.point.gradient)
        ));
    }
    Ok(finish(hf, minimum.point, minimum.iterations, true))
}

/// Evaluates the physical HF energy and every analytic potential-coefficient derivative.
/// The reference density in the Fermi-Amaldi baseline is held fixed. This does not
/// evaluate nuclear response. `converged` only reports potential stationarity here.
pub fn evaluate_oep_hf_from_reference(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    parameters: &[f64],
    options: &OepOptions,
) -> Result<OepResult, String> {
    let problem = prepare(atoms, basis, hf, options)?;
    let point = problem.evaluate(parameters.to_vec())?;
    let converged = max_abs(&point.gradient) <= options.gradient_tol;
    Ok(finish(hf, point, 0, converged))
}

fn prepare<'a>(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &'a HFResult,
    options: &OepOptions,
) -> Result<Problem<'a>, String> {
    validate(options)?;
    let n = basis.len();
    let pairs = n.checked_mul(n + 1).and_then(|v| v.checked_div(2));
    let eri_len = pairs
        .and_then(|p| p.checked_mul(p + 1))
        .and_then(|v| v.checked_div(2));
    if !hf.converged
        || hf.n_occ == 0
        || hf.n_occ > hf.mo_coefficients.cols
        || hf.multiplicity != 1
        || hf.mo_coefficients.rows != n
        || atoms.is_empty()
        || [&hf.overlap_matrix, &hf.core_hamiltonian, &hf.density_matrix]
            .iter()
            .any(|m| m.rows != n || m.cols != n)
        || eri_len != Some(hf.eri.len())
    {
        return Err("OEP requires a matching converged RHF reference".into());
    }
    let metric = hf
        .mo_coefficients
        .transpose()
        .dot(&hf.overlap_matrix)
        .dot(&hf.mo_coefficients);
    if metric.data.iter().enumerate().any(|(k, v)| {
        !v.is_finite()
            || (v - if k / metric.cols == k % metric.cols {
                1.
            } else {
                0.
            })
            .abs()
                > 1e-7
    }) {
        return Err("OEP reference orbitals must be orthonormal in the AO metric".into());
    }
    let factor = 1. - 1. / (2 * hf.n_occ) as f64;
    let bytes = n
        .checked_mul(n)
        .and_then(|v| v.checked_mul(atoms.len()))
        .and_then(|v| v.checked_mul(options.exponents.len()))
        .and_then(|v| v.checked_mul(8));
    if bytes.is_none_or(|v| v > options.potential_budget_bytes) {
        return Err(format!(
            "OEP AO potential matrices require {:?} bytes, exceeding the configured budget {}",
            bytes, options.potential_budget_bytes
        ));
    }
    let (j, _) = contract_jk(&hf.density_matrix, &[], &hf.eri, basis.len());
    let operators: Result<Vec<_>, _> = atoms
        .iter()
        .flat_map(|atom| {
            options
                .exponents
                .iter()
                .map(move |exponent| gaussian_potential_matrix(basis, atom.position, *exponent))
        })
        .collect();
    Ok(Problem {
        hf,
        baseline: hf.core_hamiltonian.add(&j.scale(factor)),
        operators: operators?,
        minimum_gap: options.minimum_gap,
    })
}

fn finish(
    hf: &HFResult,
    point: Evaluation<Point>,
    iterations: usize,
    converged: bool,
) -> OepResult {
    let gradient_norm = max_abs(&point.gradient);
    let p = point.state;
    let residual = hf
        .mo_coefficients
        .transpose()
        .dot(
            &p.ks
                .dot(&p.density)
                .dot(&hf.overlap_matrix)
                .sub(&hf.overlap_matrix.dot(&p.density).dot(&p.ks)),
        )
        .dot(&hf.mo_coefficients);
    OepResult {
        total_energy: point.value,
        electronic_energy: point.value - hf.nuclear_repulsion_energy,
        nuclear_energy: hf.nuclear_repulsion_energy,
        reference_hf_energy: hf.total_energy,
        density_matrix: p.density,
        reference_density_matrix: hf.density_matrix.clone(),
        mo_coefficients: p.coefficients,
        orbital_energies: p.energies,
        ks_hamiltonian: p.ks,
        hf_fock_matrix: p.fock,
        parameters: p.parameters,
        parameter_gradient: point.gradient,
        gradient_norm,
        ks_residual_norm: max_abs(&residual.data),
        minimum_gap: p.gap,
        fermi_amaldi_factor: 1. - 1. / (2 * hf.n_occ) as f64,
        iterations,
        converged,
    }
}
