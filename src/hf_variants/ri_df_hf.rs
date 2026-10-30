// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Coulomb-metric RI-JK Hartree-Fock with actual auxiliary Gaussian functions.
//! Packed three-center tensors are shared with post-HF DF. Exchange contracts
//! occupied coefficients, O(N_aux N_AO^2 N_occ); no AO fourth-power ERI tensor
//! or AO-pair supermatrix is allocated.

use crate::{
    ao::{AoSpace, BasisKind},
    basis::{Atom, ContractedGaussian},
    context::AuxBasis,
    df::DfTensor,
    diis::{CdiisEngine, compute_diis_error},
    matrix::Matrix,
    scf::{compute_nuclear_repulsion, compute_one_electron_integrals},
};

#[derive(Clone, Debug)]
pub struct DFHFOptions {
    pub basis_kind: BasisKind,
    pub max_iterations: usize,
    pub energy_tolerance: f64,
    pub residual_tolerance: f64,
    pub density_tolerance: f64,
    pub metric_cutoff: f64,
    /// Optional streamed exact-ERI comparison. Off by default: computing the
    /// full fitting error has fourth-power cost even without storing ERIs.
    pub estimate_fitting_error: bool,
}
impl Default for DFHFOptions {
    fn default() -> Self {
        Self {
            basis_kind: BasisKind::Cartesian,
            max_iterations: 100,
            energy_tolerance: 1e-12,
            residual_tolerance: 1e-10,
            density_tolerance: 1e-10,
            metric_cutoff: 1e-10,
            estimate_fitting_error: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DFHFResult {
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_energy: f64,
    /// Number of retained Coulomb-metric auxiliary modes.
    pub n_aux: usize,
    /// Maximum absolute AO-ERI fitting error, only when explicitly requested.
    pub fitting_error: Option<f64>,
    /// Cartesian rows, retained physical MO columns (possibly rectangular).
    pub mo_coefficients: Matrix,
    pub density_matrix: Matrix,
    pub fock_matrix: Matrix,
    pub orbital_energies: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub residual_norm: f64,
}

fn occupied(c: &Matrix, no: usize) -> Matrix {
    let mut co = Matrix::zeros(c.rows, no);
    for p in 0..c.rows {
        for i in 0..no {
            co[(p, i)] = c[(p, i)];
        }
    }
    co
}
fn diagonalize(f: &Matrix, x: &Matrix) -> (Vec<f64>, Matrix) {
    let (eps, c) = x.transpose().dot(f).dot(x).eigh();
    (eps, x.dot(&c))
}
fn fock(tensor: &DfTensor, h: &Matrix, co: &Matrix) -> Result<Matrix, String> {
    let (j, k) = tensor.coulomb_exchange_rhf(co).map_err(|e| e.to_string())?;
    Ok(h.add(&j).sub(&k.scale(0.5)))
}
fn energy(p: &Matrix, h: &Matrix, f: &Matrix) -> f64 {
    0.5 * p
        .data
        .iter()
        .zip(&h.data)
        .zip(&f.data)
        .map(|((p, h), f)| p * (h + f))
        .sum::<f64>()
}
fn norm(m: &Matrix) -> f64 {
    m.data.iter().map(|v| v.abs()).fold(0., f64::max)
}

/// Compatibility entry point, using embedded def2-universal-JKFIT functions.
/// For reproducible fitting choices use `run_df_hf_with_aux` explicitly.
pub fn run_df_hf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Result<DFHFResult, String> {
    let auxiliary =
        AuxBasis::from_builtin(atoms, "def2-universal-jkfit").map_err(|e| e.to_string())?;
    run_df_hf_with_aux(
        atoms,
        basis,
        &auxiliary.functions,
        charge,
        &DFHFOptions {
            max_iterations: max_iter,
            energy_tolerance: tol,
            residual_tolerance: (tol.sqrt() * 1e-2).max(1e-11),
            density_tolerance: (tol.sqrt() * 0.1).clamp(1e-12, 1e-10),
            ..Default::default()
        },
    )
}

pub fn run_df_hf_with_aux(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    aux: &[ContractedGaussian],
    charge: i32,
    options: &DFHFOptions,
) -> Result<DFHFResult, String> {
    if atoms.is_empty()
        || basis.is_empty()
        || options.max_iterations == 0
        || [
            options.energy_tolerance,
            options.residual_tolerance,
            options.density_tolerance,
            options.metric_cutoff,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
        || atoms.iter().any(|a| {
            !a.charge.is_finite() || a.charge < 0. || a.position.iter().any(|v| !v.is_finite())
        })
    {
        return Err("DF-HF requires finite geometry and positive convergence controls".into());
    }
    let electrons = atoms.iter().map(|a| a.charge).sum::<f64>() - f64::from(charge);
    if electrons <= 0.
        || (electrons - electrons.round()).abs() > 1e-8
        || !(electrons.round() as usize).is_multiple_of(2)
    {
        return Err("DF-RHF requires a positive even electron count".into());
    }
    let tensor = DfTensor::build(basis, aux, options.metric_cutoff).map_err(|e| e.to_string())?;
    run_factorized_rhf(atoms, basis, &tensor, charge, options)
}

/// Residual-certified RHF whose Coulomb and exchange matrices come from three-index factors
/// `(pq|rs) ~ sum_Q B_Q(pq) B_Q(rs)` (RI/DF or pivoted Cholesky). Energy, metric orbital
/// residual and density change must all converge.
pub fn run_factorized_rhf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    tensor: &DfTensor,
    charge: i32,
    options: &DFHFOptions,
) -> Result<DFHFResult, String> {
    if tensor.n_ao != basis.len() || tensor.rank() == 0 {
        return Err("factorized RHF tensor does not match the AO basis".into());
    }
    let electrons = atoms.iter().map(|a| a.charge).sum::<f64>() - f64::from(charge);
    if electrons <= 0.
        || (electrons - electrons.round()).abs() > 1e-8
        || !(electrons.round() as usize).is_multiple_of(2)
    {
        return Err("factorized RHF requires a positive even electron count".into());
    }
    let no = electrons.round() as usize / 2;
    let (s, _, _, h) = compute_one_electron_integrals(atoms, basis);
    let mut space = AoSpace::new(basis, options.basis_kind).map_err(|e| e.to_string())?;
    let x = space.orthogonalizer(&s).map_err(|e| e.to_string())?;
    if no > x.cols {
        return Err("factorized RHF occupied count exceeds overlap rank".into());
    }
    let enuc = compute_nuclear_repulsion(atoms);
    let (_, mut c) = diagonalize(&h, &x);
    let mut co = occupied(&c, no);
    let mut p = co.dot(&co.transpose()).scale(2.);
    let mut diis = CdiisEngine::new(8);
    let mut last_energy = f64::INFINITY;
    for iter in 1..=options.max_iterations {
        let f = fock(tensor, &h, &co)?;
        let current_energy = energy(&p, &h, &f) + enuc;
        let error = x.transpose().dot(&compute_diis_error(&f, &p, &s)).dot(&x);
        let residual = norm(&error);
        let (eps, next_c) = diagonalize(&f, &x);
        let next_co = occupied(&next_c, no);
        let next_p = next_co.dot(&next_co.transpose()).scale(2.);
        let density_change = norm(&next_p.sub(&p));
        if !current_energy.is_finite() || !residual.is_finite() {
            return Err("DF-HF energy/residual became nonfinite".into());
        }
        if (current_energy - last_energy).abs() <= options.energy_tolerance
            && residual <= options.residual_tolerance
            && density_change <= options.density_tolerance
        {
            let final_f = fock(tensor, &h, &next_co)?;
            let final_energy = energy(&next_p, &h, &final_f) + enuc;
            let fitting_error = if options.estimate_fitting_error {
                let n = basis.len();
                let mut maximum = 0f64;
                for i in 0..n {
                    for j in 0..=i {
                        for k in 0..n {
                            for l in 0..=k {
                                maximum = maximum.max(
                                    (crate::integrals::contracted_eri(
                                        &basis[i], &basis[j], &basis[k], &basis[l],
                                    ) - tensor.eri(i, j, k, l))
                                    .abs(),
                                );
                            }
                        }
                    }
                }
                Some(maximum)
            } else {
                None
            };
            return Ok(DFHFResult {
                total_energy: final_energy,
                electronic_energy: final_energy - enuc,
                nuclear_energy: enuc,
                n_aux: tensor.rank(),
                fitting_error,
                mo_coefficients: next_c,
                density_matrix: next_p.clone(),
                orbital_energies: eps,
                iterations: iter,
                converged: true,
                residual_norm: norm(
                    &x.transpose()
                        .dot(&compute_diis_error(&final_f, &next_p, &s))
                        .dot(&x),
                ),
                fock_matrix: final_f,
            });
        }
        last_energy = current_energy;
        diis.push(f.clone(), error);
        let extrapolated = if iter >= 2 {
            diis.extrapolate().unwrap_or(f)
        } else {
            f
        };
        (_, c) = diagonalize(&extrapolated, &x);
        co = occupied(&c, no);
        p = co.dot(&co.transpose()).scale(2.);
    }
    Err(format!(
        "factorized RHF did not converge in {} iterations",
        options.max_iterations
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn df_hf_uses_auxiliary_basis_and_reports_unrequested_error_as_absent() {
        let atoms = [
            Atom::new("H", [0., 0., -0.7]),
            Atom::new("H", [0., 0., 0.7]),
        ];
        let basis = crate::basis::build_basis(
            &atoms,
            &crate::basis::parse_gaussian94(crate::basis::STO_3G_GBS),
        );
        let result = run_df_hf(&atoms, &basis, 0, 100, 1e-12).unwrap();
        assert!(result.converged);
        assert_eq!(result.fitting_error, None);
        assert!((result.total_energy + 1.1167143251).abs() < 1e-3);
    }
}
