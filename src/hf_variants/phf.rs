// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Spin-projected UHF (PUHF) and Hartree-Fock-Bogoliubov (HFB).
//!
//! PUHF is the exact Lowdin projection after variation of the (broken-symmetry) UHF determinant onto
//! `S = |S_z|` ([`super::spin_projection`]). Yamaguchi's approximate projection, which assumes a single
//! higher-spin contaminant described by a separate high-spin UHF calculation, is reported alongside as
//! a clearly labeled estimate.

use super::spin_projection::project_uhf_spin;
use crate::basis::{Atom, ContractedGaussian};
use crate::scf::{UHFResult, run_uhf};

/// Results of a spin-projected UHF calculation.
#[derive(Clone, Debug)]
pub struct PUHFResult {
    /// The (broken-symmetry) UHF determinant that is projected.
    pub raw_uhf: UHFResult,
    /// Exact Lowdin-projected energy `<Phi|H P_S|Phi>/<Phi|P_S|Phi>`.
    pub projected_energy: f64,
    /// Weight of the target spin component, `<Phi|P_S|Phi>`.
    pub target_weight: f64,
    /// Total weight of all other spin components, `1 - target_weight`.
    pub spin_contamination_weight: f64,
    /// Target spin quantum number times two (`2S`).
    pub target_spin: usize,
    /// Energy of a converged next-higher-spin UHF determinant, when available.
    pub high_spin_reference_energy: Option<f64>,
    /// Yamaguchi approximate projection (single-contaminant estimate), if its denominator is nonzero.
    pub yamaguchi_energy: Option<f64>,
}

/// Runs a broken-symmetry (even electron count) or doublet (odd) UHF, projects it exactly onto
/// `S = |S_z|`, and also evaluates Yamaguchi's approximate projection with the next-higher-spin UHF.
pub fn run_puhf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Result<PUHFResult, String> {
    let total_nuclear_charge: f64 = atoms.iter().map(|a| a.charge).sum();
    let total_electrons = (total_nuclear_charge.round() as i32) - charge;
    if total_electrons <= 0 {
        return Err("Invalid electron count for PUHF".to_string());
    }
    let multiplicity = if total_electrons % 2 != 0 { 2 } else { 1 };
    run_puhf_with_multiplicity(atoms, basis, charge, multiplicity, max_iter, tol)
}

/// Projects UHF onto `S = (multiplicity - 1)/2`. Singlets start from a broken-symmetry guess.
/// The exact projection does not require a separate high-spin calculation. Yamaguchi's estimate
/// is absent if the next spin sector is unavailable, fails to converge, or has zero denominator.
pub fn run_puhf_with_multiplicity(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    multiplicity: usize,
    max_iter: usize,
    tol: f64,
) -> Result<PUHFResult, String> {
    let bs = run_uhf(
        atoms,
        basis,
        charge,
        multiplicity,
        multiplicity == 1,
        max_iter,
        tol,
    )?;
    let projection = project_uhf_spin(&bs)?;
    let s_target = 0.5 * (multiplicity as f64 - 1.0);
    let s2_target = s_target * (s_target + 1.0);
    let high = if bs.n_beta > 0 && bs.n_alpha < bs.mo_coefficients_alpha.cols {
        run_uhf(atoms, basis, charge, multiplicity + 2, false, max_iter, tol)
            .ok()
            .filter(|result| result.converged)
    } else {
        None
    };
    let yamaguchi_energy = high.as_ref().and_then(|high| {
        let denominator = high.s_squared - bs.s_squared;
        (denominator.abs() > 1e-10).then(|| {
            (bs.total_energy * (high.s_squared - s2_target)
                - high.total_energy * (bs.s_squared - s2_target))
                / denominator
        })
    });
    Ok(PUHFResult {
        projected_energy: projection.energy,
        target_weight: projection.target_weight,
        spin_contamination_weight: 1.0 - projection.target_weight,
        target_spin: multiplicity - 1,
        high_spin_reference_energy: high.as_ref().map(|r| r.total_energy),
        yamaguchi_energy,
        raw_uhf: bs,
    })
}

/// Compatibility exports for the consistent singlet HFB model.
pub use super::hfb::{HFBResult, run_hfb};
