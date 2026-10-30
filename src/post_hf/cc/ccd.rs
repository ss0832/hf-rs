// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Coupled Cluster Doubles (CCD) Engine.
//!
//! Solves the coupled cluster equations with pure double excitation cluster operator T2 (T1 = 0):
//! |Psi_CCD> = exp(T2) |Phi_0>
//! Accelerated with Tensor DIIS (TDIIS).

use crate::context::{MoIntegralCache, PostHFContext};
use crate::error::HfError;
use crate::mo_integrals::transform_eri_ao_to_mo;
use crate::options::PostHFOptions;
use crate::post_hf::cc::CCSDOptions;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::{ContextMethod, MethodId, MethodMeta};
use crate::scf::HFResult;
use std::borrow::Cow;

/// Results of a Coupled Cluster Doubles calculation.
#[derive(Clone, Debug)]
pub struct CCDResult {
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub total_energy: f64,
    pub iterations: usize,
    pub converged: bool,
    pub t2_amplitudes: Vec<f64>,
    pub n_occ: usize,
    pub n_virt: usize,
}

impl PostHFResult for CCDResult {
    fn intermediates(&self, max_elements: usize) -> Result<crate::output::TensorMap, HfError> {
        use crate::output::{ObservableUnit, TensorExport};
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.tensor(
            "method.t2",
            &self.t2_amplitudes,
            &[self.n_occ, self.n_occ, self.n_virt, self.n_virt],
            &["occupied_i", "occupied_j", "virtual_a", "virtual_b"],
            ObservableUnit::One,
        )?;
        Ok(tensors.finish())
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn method_name(&self) -> &'static str {
        "CCD"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }

    fn total_energy(&self) -> f64 {
        self.total_energy
    }
}

/// Closed-shell RHF-CCD calculation engine.
pub struct CCD<'a> {
    pub integral_cache: Option<&'a MoIntegralCache<'a>>,
    pub hf: &'a HFResult,
    pub options: CCSDOptions,
}

impl<'a> CCD<'a> {
    pub fn new(hf: &'a HFResult) -> Self {
        Self {
            integral_cache: None,
            hf,
            options: CCSDOptions::default(),
        }
    }

    pub fn with_options(mut self, options: CCSDOptions) -> Self {
        self.options = options;
        self
    }
}

impl<'a> PostHFMethod for CCD<'a> {
    type Output = CCDResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let n_occ = self.hf.n_occ;
        let n_basis = self.hf.orbital_energies.len();
        let n_virt = n_basis - n_occ;

        if n_occ == 0 || n_virt == 0 {
            return Err("CCD requires at least one occupied and one virtual orbital".to_string());
        }

        let mo_eri = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.full().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(transform_eri_ao_to_mo(
                &self.hf.mo_coefficients,
                &self.hf.eri,
            ))
        };
        if !self.hf.converged {
            return Err("CCD requires a converged RHF reference".into());
        }
        let result = super::rhf_ccsd::solve(
            self.hf.total_energy,
            &self.hf.orbital_energies,
            &mo_eri,
            0,
            n_occ,
            n_virt,
            &self.options,
            None,
            true,
        )?;
        Ok(CCDResult {
            reference_energy: result.reference_energy,
            correlation_energy: result.correlation_energy,
            total_energy: result.total_energy,
            iterations: result.iterations,
            converged: result.converged,
            t2_amplitudes: result.t2_amplitudes,
            n_occ,
            n_virt,
        })
    }
}

impl MethodMeta for CCD<'_> {
    const METHOD: MethodId = MethodId::Ccd;
}
impl<'a> ContextMethod<'a> for CCD<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        if options.frozen_core != 0 || options.frozen_virtual != 0 {
            return Err(HfError::Unsupported {
                method: "CCD".into(),
                detail: "frozen orbital CC adapter is pending".into(),
            });
        }
        let mut engine = Self::new(ctx.rhf()?);
        engine.integral_cache = Some(&ctx.ints);
        engine.options.max_iterations = options.max_iter;
        engine.options.energy_tol = options.energy_tol;
        engine.options.residual_tol = options.residual_tol;
        Ok(engine)
    }
}
