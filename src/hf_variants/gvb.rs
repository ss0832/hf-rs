// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Two-configuration SCF (TCSCF), i.e. one-pair perfect-pairing GVB.
//!
//! `|Psi> = c1 |core phi_g^2> - c2 |core phi_u^2>` with all orbitals variationally optimized.
//! For two electrons in two active orbitals every singlet can be written in this form in its
//! natural orbitals, so fully optimized TCSCF equals singlet CASSCF(2,2) with the selected
//! bonding/antibonding pair as the active space. The calculation uses the verified CAS
//! engine (all inactive-active, active-external and inactive-external rotations); the
//! reported orbitals are rotated to active natural orbitals, where `c1 = sqrt(n_g/2)` and
//! `c2 = sqrt(n_u/2)`. Multi-pair GVB-PP is not implemented.

use crate::basis::{Atom, ContractedGaussian};
use crate::context::Reference;
use crate::diis::ConvAlgo;
use crate::matrix::Matrix;
use crate::options::{ActiveSpace, PostHFOptions};
use crate::post_hf::casscf::CASSCFResult;
use crate::post_hf::casscf_engine::{solve_cas, CasResult};
use crate::scf::{run_scf_general, HFResult};

/// Results of a TCSCF calculation.
#[derive(Clone, Debug)]
pub struct TCSCFResult {
    pub hf_reference_energy: f64,
    pub tcscf_energy: f64,
    pub correlation_energy: f64,
    /// Weight of the doubly occupied bonding natural orbital, `sqrt(n_g/2)`.
    pub c1: f64,
    /// Weight of the doubly occupied antibonding natural orbital, `sqrt(n_u/2)`.
    pub c2: f64,
    /// Active natural-orbital occupations `(n_g, n_u)`, `n_g + n_u = 2`.
    pub natural_occupations: [f64; 2],
    pub active_bonding_orbital: usize,
    pub active_antibonding_orbital: usize,
    /// Optimized orbitals (Cartesian AO rows) with the active pair in natural-orbital form,
    /// ordered inactive, bonding, antibonding, external.
    pub mo_coefficients: Matrix,
    /// The TCSCF state is a spin singlet by construction.
    pub s_squared: f64,
    pub orbital_gradient_max: f64,
    pub iterations: usize,
    pub converged: bool,
    /// Underlying CAS(2,2) solution, used for analytic gradients.
    pub casscf: CASSCFResult,
}

/// Runs RHF and then TCSCF for the selected bonding (occupied) and antibonding (virtual) orbitals.
pub fn run_tcscf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    bonding_orbital: usize,
    antibonding_orbital: usize,
    max_iter: usize,
    tol: f64,
) -> Result<TCSCFResult, String> {
    let hf = run_scf_general(atoms, basis, charge, max_iter.max(100), tol, ConvAlgo::Adiissoscf, None)?;
    if !hf.converged {
        return Err("TCSCF requires a converged RHF reference".into());
    }
    run_tcscf_from_reference(&hf, bonding_orbital, antibonding_orbital, max_iter, tol)
}

/// TCSCF starting from an existing RHF reference.
pub fn run_tcscf_from_reference(
    hf: &HFResult,
    bonding_orbital: usize,
    antibonding_orbital: usize,
    max_iter: usize,
    tol: f64,
) -> Result<TCSCFResult, String> {
    let n_mo = hf.mo_coefficients.cols;
    if bonding_orbital >= hf.n_occ || antibonding_orbital < hf.n_occ || antibonding_orbital >= n_mo {
        return Err(format!(
            "invalid TCSCF orbitals: bonding ({bonding_orbital}) must be occupied (< {}), antibonding ({antibonding_orbital}) virtual (< {n_mo})",
            hf.n_occ
        ));
    }
    if max_iter == 0 || !tol.is_finite() || tol <= 0.0 {
        return Err("TCSCF requires a positive iteration limit and tolerance".into());
    }
    let mut options = PostHFOptions {
        active: Some(ActiveSpace { n_alpha: 1, n_beta: 1, n_orb: 2, orbitals: Some(vec![bonding_orbital, antibonding_orbital]) }),
        energy_tol: tol,
        ..Default::default()
    };
    options.cas.max_orbital_iterations = max_iter;
    options.cas.orbital_gradient_tol = tol.sqrt().min(1e-6);
    options.cas.target_s_squared = Some(0.0);
    let cas: CasResult = solve_cas(Reference::Rhf(hf), &options, None, true).map_err(|e| e.to_string())?;
    let active = &cas.active_space.active;
    let full_one = &cas.state_one_rdms[0];
    // active 2x2 block of the spin-traced one-particle density
    let block = if full_one.rows == 2 {
        full_one.clone()
    } else {
        Matrix::from_vec(2, 2, vec![
            full_one[(active[0], active[0])],
            full_one[(active[0], active[1])],
            full_one[(active[1], active[0])],
            full_one[(active[1], active[1])],
        ])
    };
    let (occ, vecs) = block.eigh(); // ascending: (n_u, n_g)
    let (n_u, n_g) = (occ[0].max(0.0), occ[1].max(0.0));
    // rotate the active pair to natural orbitals (bonding first)
    let mut c = cas.mo_coefficients.clone();
    for r in 0..c.rows {
        let a0 = cas.mo_coefficients[(r, active[0])];
        let a1 = cas.mo_coefficients[(r, active[1])];
        c[(r, active[0])] = a0 * vecs[(0, 1)] + a1 * vecs[(1, 1)];
        c[(r, active[1])] = a0 * vecs[(0, 0)] + a1 * vecs[(1, 0)];
    }
    let one = cas.output.one_rdm.clone().ok_or("CAS one-particle density is missing")?;
    let casscf = CASSCFResult {
        reference_energy: hf.total_energy,
        total_energy: cas.output.e_total,
        correlation_energy: cas.output.e_corr,
        iterations: cas.iterations,
        converged: true,
        orbital_gradient_norm: cas.orbital_gradient_max,
        natural_orbital_occupations: one.eigh().0,
        one_rdm_mo: one,
        mo_coefficients: cas.mo_coefficients.clone(),
        generalized_fock: cas.generalized_fock.transpose(),
        two_rdm_mo: cas.state_two_rdms[0].clone(),
        n_core: cas.active_space.inactive.len(),
        n_active_orbitals: active.len(),
    };
    Ok(TCSCFResult {
        hf_reference_energy: hf.total_energy,
        tcscf_energy: cas.output.e_total,
        correlation_energy: cas.output.e_total - hf.total_energy,
        c1: (n_g / 2.0).sqrt(),
        c2: (n_u / 2.0).sqrt(),
        natural_occupations: [n_g, n_u],
        active_bonding_orbital: bonding_orbital,
        active_antibonding_orbital: antibonding_orbital,
        mo_coefficients: c,
        s_squared: 0.0,
        orbital_gradient_max: cas.orbital_gradient_max,
        iterations: cas.iterations,
        converged: true,
        casscf,
    })
}

/// Analytic nuclear gradient of the variational TCSCF energy (CASSCF(2,2) Hellmann-Feynman route).
pub fn compute_tcscf_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    result: &TCSCFResult,
) -> Vec<[f64; 3]> {
    crate::post_hf::casscf::compute_casscf_analytical_gradient(atoms, basis, hf, &result.casscf)
}
