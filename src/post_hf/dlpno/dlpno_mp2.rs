// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! Compatibility API for the checked RI local-MP2 implementation.
use crate::basis::{Atom, ContractedGaussian};
use crate::context::{AuxBasis, PostHFContext};
use crate::error::HfError;
use crate::options::PostHFOptions;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::ContextMethod;
use crate::scf::HFResult;

#[derive(Clone, Debug)]
pub struct DLPNOMP2Options {
    pub t_pno: f64,
    pub t_cut_pairs: f64,
    /// Removed neglected-pair approximation. Nonzero values are rejected.
    pub t_cut_weak: f64,
    /// Empirical smoothing is outside the implemented local-correlation theory.
    pub smoothing: bool,
    pub delta_switch: f64,
}
impl Default for DLPNOMP2Options {
    fn default() -> Self {
        Self {
            t_pno: 1e-7,
            t_cut_pairs: 1e-5,
            t_cut_weak: 0.,
            smoothing: false,
            delta_switch: 0.2,
        }
    }
}
impl DLPNOMP2Options {
    pub(crate) fn typed(&self) -> Result<PostHFOptions, HfError> {
        if self.smoothing || self.t_cut_weak != 0. {
            return Err(HfError::InvalidInput("DLPNO smoothing and neglected-pair tapering are removed; use smoothing=false and t_cut_weak=0".into()));
        }
        if !self.delta_switch.is_finite() || self.delta_switch < 0. {
            return Err(HfError::InvalidInput(
                "invalid legacy switching parameter".into(),
            ));
        }
        let mut options = PostHFOptions::default();
        options.local.t_cut_pno = self.t_pno;
        options.local.t_cut_pairs = self.t_cut_pairs;
        if self.t_pno == 0. && self.t_cut_pairs == 0. {
            options.local.t_cut_do = 0.;
            options.local.t_cut_mkn = 0.;
            options.local.full_domains = true;
            options.local.weak_pairs = false;
        }
        options.validate()?;
        Ok(options)
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct DLPNOMP2Result {
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub strong_pair_energy: f64,
    pub weak_pair_energy: f64,
    pub delta_pno_energy: f64,
    pub total_energy: f64,
    pub n_strong_pairs: usize,
    pub n_weak_pairs: usize,
    pub n_neglected_pairs: usize,
}
impl PostHFResult for DLPNOMP2Result {
    fn method_name(&self) -> &'static str {
        "DLPNO-MP2"
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
    fn to_output(&self) -> Result<crate::output::PostHFOutput, HfError> {
        let mut out = crate::output::PostHFOutput::new(
            self.method_name(),
            self.reference_energy,
            self.total_energy,
        )?;
        out.components
            .insert("E_strong_pairs".into(), self.strong_pair_energy);
        out.components
            .insert("E_weak_pairs".into(), self.weak_pair_energy);
        out.components
            .insert("E_delta_PNO".into(), self.delta_pno_energy);
        Ok(out)
    }
}
#[derive(Clone, Default)]
pub(crate) struct LegacySystem<'a> {
    pub atoms: Option<&'a [Atom]>,
    pub basis: Option<&'a [ContractedGaussian]>,
    pub aux: Option<&'a AuxBasis>,
    pub ctx: Option<&'a PostHFContext<'a>>,
}
impl<'a> LegacySystem<'a> {
    pub fn run<T>(
        &self,
        hf: &'a HFResult,
        action: impl for<'b> FnOnce(&'b PostHFContext<'b>) -> Result<T, HfError>,
    ) -> Result<T, HfError> {
        if let Some(ctx) = self.ctx {
            if !std::ptr::eq(ctx.rhf()?, hf) {
                return Err(HfError::InvalidInput(
                    "legacy DLPNO context/reference mismatch".into(),
                ));
            }
            return action(ctx);
        }
        let atoms=self.atoms.ok_or_else(||HfError::InvalidInput("DLPNO requires geometry, orbital and auxiliary bases; use with_system or with_context".into()))?;
        let basis = self
            .basis
            .ok_or_else(|| HfError::InvalidInput("DLPNO requires the orbital basis".into()))?;
        let aux = self.aux.ok_or_else(|| {
            HfError::InvalidInput("DLPNO requires an explicit auxiliary basis".into())
        })?;
        let ctx = PostHFContext::from_rhf(hf, atoms, basis, Some(aux))?;
        action(&ctx)
    }
}
pub struct DLPNOMP2<'a> {
    pub hf: &'a HFResult,
    pub options: DLPNOMP2Options,
    system: LegacySystem<'a>,
}
impl<'a> DLPNOMP2<'a> {
    pub fn new(hf: &'a HFResult) -> Self {
        Self {
            hf,
            options: Default::default(),
            system: Default::default(),
        }
    }
    pub fn with_options(mut self, options: DLPNOMP2Options) -> Self {
        self.options = options;
        self
    }
    pub fn with_basis(mut self, basis: &'a [ContractedGaussian]) -> Self {
        self.system.basis = Some(basis);
        self
    }
    pub fn with_system(
        mut self,
        atoms: &'a [Atom],
        basis: &'a [ContractedGaussian],
        aux: &'a AuxBasis,
    ) -> Self {
        self.system = LegacySystem {
            atoms: Some(atoms),
            basis: Some(basis),
            aux: Some(aux),
            ctx: None,
        };
        self
    }
    pub fn with_context(mut self, ctx: &'a PostHFContext<'a>) -> Self {
        self.system.ctx = Some(ctx);
        self
    }
}
impl PostHFMethod for DLPNOMP2<'_> {
    type Output = DLPNOMP2Result;
    fn compute(&self) -> Result<Self::Output, String> {
        let options = self.options.typed().map_err(|e| e.to_string())?;
        self.system
            .run(self.hf, |ctx| {
                let result = super::LocalMp2::from_context(ctx, &options)?.compute_checked()?;
                Ok(DLPNOMP2Result {
                    reference_energy: result.e_scf,
                    correlation_energy: result.correlation_energy(),
                    total_energy: result.total_energy(),
                    strong_pair_energy: result.solution.e_strong,
                    weak_pair_energy: result.solution.e_weak,
                    delta_pno_energy: result.solution.delta_pno,
                    n_strong_pairs: result.n_strong_pairs,
                    n_weak_pairs: result.n_weak_pairs,
                    n_neglected_pairs: 0,
                })
            })
            .map_err(|e| e.to_string())
    }
}
