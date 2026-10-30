// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Cholesky-decomposition Hartree-Fock (CD-HF).
//!
//! The AO two-electron integrals are replaced by pivoted Cholesky factors
//! `(pq|rs) ~ sum_K L_K(pq) L_K(rs)` computed on the fly to the residual diagonal threshold
//! `delta` ([`crate::cholesky::pivoted_cholesky_eri`]). Coulomb and exchange matrices are then
//! built exactly like RI-JK, `J = sum_K L_K Tr(L_K P)`, `K = 2 sum_K (L_K C_occ)(L_K C_occ)^T`,
//! in the shared residual-certified factorized RHF driver. As `delta -> 0` CD-HF converges to
//! conventional RHF; every reconstructed integral error is bounded by the reported residual.
//! Analytic nuclear gradients are not available (the pivot set changes with geometry).

use crate::basis::{Atom, ContractedGaussian};
use crate::cholesky::pivoted_cholesky_eri;
use crate::hf_variants::ri_df_hf::{run_factorized_rhf, DFHFOptions};
use crate::matrix::Matrix;

/// Default bound on the stored Cholesky factor bytes.
pub const CD_HF_FACTOR_BUDGET: usize = 512 * 1024 * 1024;

/// Results from a Cholesky-decomposed Hartree-Fock calculation.
#[derive(Clone, Debug)]
pub struct CDHFResult {
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_repulsion_energy: f64,
    pub n_cholesky_vectors: usize,
    /// Largest residual ERI diagonal; bounds every reconstructed-integral error.
    pub cholesky_max_error: f64,
    /// Cartesian AO rows, retained MO columns.
    pub mo_coefficients: Matrix,
    pub density_matrix: Matrix,
    pub fock_matrix: Matrix,
    pub orbital_energies: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub residual_norm: f64,
}

/// CD-HF with residual diagonal threshold `cholesky_delta` and the given SCF controls.
pub fn run_cd_hf_with_options(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    cholesky_delta: f64,
    options: &DFHFOptions,
) -> Result<CDHFResult, String> {
    let (tensor, info) = pivoted_cholesky_eri(basis, cholesky_delta, CD_HF_FACTOR_BUDGET).map_err(|e| e.to_string())?;
    let r = run_factorized_rhf(atoms, basis, &tensor, charge, options)?;
    Ok(CDHFResult {
        total_energy: r.total_energy,
        electronic_energy: r.electronic_energy,
        nuclear_repulsion_energy: r.nuclear_energy,
        n_cholesky_vectors: tensor.rank(),
        cholesky_max_error: info.max_residual,
        mo_coefficients: r.mo_coefficients,
        density_matrix: r.density_matrix,
        fock_matrix: r.fock_matrix,
        orbital_energies: r.orbital_energies,
        iterations: r.iterations,
        converged: r.converged,
        residual_norm: r.residual_norm,
    })
}

/// CD-HF with energy tolerance `tol` (residual/density tolerances follow the RHF defaults).
pub fn run_cd_hf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    cholesky_delta: f64,
    max_iter: usize,
    tol: f64,
) -> Result<CDHFResult, String> {
    let options = DFHFOptions {
        max_iterations: max_iter,
        energy_tolerance: tol,
        residual_tolerance: (tol.sqrt() * 1e-2).max(1e-11),
        density_tolerance: (tol.sqrt() * 0.1).clamp(1e-12, 1e-10),
        ..Default::default()
    };
    run_cd_hf_with_options(atoms, basis, charge, cholesky_delta, &options)
}
