// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Compatibility API for the charge-conserving MPS/MPO DMRG engine.

use crate::basis::{Atom, ContractedGaussian};
use crate::context::Reference;
use crate::matrix::Matrix;
use crate::options::{ActiveSpace, PostHFOptions};
use crate::post_hf::dmrg_orbitals::solve_dmrg_reference;
use crate::post_hf::grad_common::compute_rhf_total_rdm_analytical_gradient;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::scf::HFResult;

/// Options for Active-Space Density Matrix Renormalization Group (`ActiveSpaceDMRG` / `DMRGSCF`).
#[derive(Clone, Debug)]
pub struct DMRGOptions {
    pub n_active_electrons: usize,
    pub n_active_orbitals: usize,
    /// Maximum MPS bond dimension $M$ retained during 2-site renormalization sweeps.
    pub max_bond_dimension: usize,
    /// Maximum directional passes in each two-site / one-site phase.
    pub max_sweeps: usize,
    /// Energy convergence threshold in Hartree.
    pub tol: f64,
    /// Whether to couple DMRG to macro-iteration orbital optimization (`DMRGSCF`).
    pub optimize_orbitals: bool,
}

impl DMRGOptions {
    pub fn new(
        n_active_electrons: usize,
        n_active_orbitals: usize,
        max_bond_dimension: usize,
    ) -> Self {
        Self {
            n_active_electrons,
            n_active_orbitals,
            max_bond_dimension: max_bond_dimension.max(1),
            max_sweeps: 12,
            tol: 1e-9,
            optimize_orbitals: false,
        }
    }
}

/// Result of an `ActiveSpaceDMRG` or `DMRGSCF` calculation.
#[derive(Clone, Debug)]
pub struct DMRGResult {
    pub is_scf: bool,
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub total_energy: f64,
    pub max_bond_dimension: usize,
    pub sweeps: usize,
    pub max_discarded_weight: f64,
    pub converged: bool,
    pub one_rdm_mo: Matrix,
    pub two_rdm_mo: Vec<f64>,
    /// Frobenius norm of the active spin-traced diagonal Gamma3[u,u,v,v,w,w].
    /// This is not the norm of the full 3-RDM.
    pub three_rdm_norm: f64,
    pub mo_coefficients: Matrix,
}

impl PostHFResult for DMRGResult {
    fn converged(&self) -> bool {
        self.converged
    }
    fn method_name(&self) -> &'static str {
        if self.is_scf { "DMRG-SCF" } else { "DMRG-CI" }
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }

    fn one_rdm(&self) -> Option<&Matrix> {
        Some(&self.one_rdm_mo)
    }
}

/// Active-Space Density Matrix Renormalization Group (`ActiveSpaceDMRG` / `DMRGSCF`) solver.
pub struct ActiveSpaceDMRG<'a> {
    hf: &'a HFResult,
    options: DMRGOptions,
}

impl<'a> ActiveSpaceDMRG<'a> {
    pub fn new(hf: &'a HFResult, options: DMRGOptions) -> Self {
        Self { hf, options }
    }
}

impl<'a> PostHFMethod for ActiveSpaceDMRG<'a> {
    type Output = DMRGResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let options = PostHFOptions {
            active: Some(ActiveSpace::new(
                self.options.n_active_electrons.div_ceil(2),
                self.options.n_active_electrons / 2,
                self.options.n_active_orbitals,
            )),
            max_iter: 200,
            energy_tol: self.options.tol,
            dmrg: crate::options::DmrgOptions {
                bond_dimensions: vec![self.options.max_bond_dimension],
                sweeps: self.options.max_sweeps,
                one_site_sweeps: self.options.max_sweeps,
                noise: 0.0,
                truncation_tol: 0.0,
            },
            ..Default::default()
        };
        let result = solve_dmrg_reference(
            Reference::Rhf(self.hf),
            &options,
            self.options.optimize_orbitals,
            None,
        )
        .map_err(|e| e.to_string())?;
        let n = result.mo_coefficients.cols;
        if n.checked_pow(4)
            .and_then(|x| x.checked_mul(8))
            .is_none_or(|x| x > 256 * 1024 * 1024)
        {
            return Err(
                "legacy DMRG full 2-RDM exceeds 256 MiB; use the typed active-density API".into(),
            );
        }
        let nc = result.active_space.inactive.len();
        let na = result.active_space.active.len();
        let core = |p, q| if p == q && p < nc { 2.0 } else { 0.0 };
        let active = |p, q| {
            if p >= nc && p < nc + na && q >= nc && q < nc + na {
                result.active_one_rdm[(p - nc, q - nc)]
            } else {
                0.0
            }
        };
        let mut two = vec![0.0; n.pow(4)];
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        let ga = if [p, q, r, s].iter().all(|&x| x >= nc && x < nc + na) {
                            result.active_two_rdm
                                [(((p - nc) * na + q - nc) * na + r - nc) * na + s - nc]
                        } else {
                            0.0
                        };
                        two[((p * n + q) * n + r) * n + s] = ga + core(p, q) * core(r, s)
                            - 0.5 * core(p, s) * core(r, q)
                            + core(p, q) * active(r, s)
                            + active(p, q) * core(r, s)
                            - 0.5 * (core(p, s) * active(r, q) + active(p, s) * core(r, q));
                    }
                }
            }
        }
        let mut diag_three_sq = 0.0;
        for p in 0..na {
            for q in 0..na {
                for r in 0..na {
                    let mut element = 0.0;
                    for a in 0..2 {
                        for b in 0..2 {
                            for c in 0..2 {
                                element += result
                                    .mps
                                    .state
                                    .fermion_expectation(&[
                                        (2 * p + a, true),
                                        (2 * q + b, true),
                                        (2 * r + c, true),
                                        (2 * r + c, false),
                                        (2 * q + b, false),
                                        (2 * p + a, false),
                                    ])
                                    .map_err(|e| e.to_string())?;
                            }
                        }
                    }
                    diag_three_sq += element * element;
                }
            }
        }
        Ok(DMRGResult {
            is_scf: self.options.optimize_orbitals,
            reference_energy: self.hf.total_energy,
            correlation_energy: result.output.e_corr,
            total_energy: result.output.e_total,
            max_bond_dimension: self.options.max_bond_dimension,
            sweeps: result.mps.sweep_energies.len() + result.mps.one_site_energies.len(),
            max_discarded_weight: result
                .mps
                .discarded_weights
                .iter()
                .copied()
                .fold(0.0, f64::max),
            converged: result.mps.converged,
            one_rdm_mo: result.output.one_rdm.ok_or("missing DMRG 1-RDM")?,
            two_rdm_mo: two,
            three_rdm_norm: diag_three_sq.sqrt(),
            mo_coefficients: result.mo_coefficients,
        })
    }
}

/// Legacy frozen-density nuclear contraction. Finite-M tensor response, orbital
/// response for DMRG-CI and nonstationary orbital response are not included.
/// The registry advertises numerical verification gradients until P4 supplies
/// the complete relaxed analytic derivative.
pub fn compute_dmrg_analytical_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    dmrg: &DMRGResult,
) -> Vec<[f64; 3]> {
    let mut hf_rot = hf.clone();
    hf_rot.mo_coefficients = dmrg.mo_coefficients.clone();
    compute_rhf_total_rdm_analytical_gradient(
        atoms,
        basis,
        &hf_rot,
        &dmrg.one_rdm_mo,
        &dmrg.two_rdm_mo,
    )
}
