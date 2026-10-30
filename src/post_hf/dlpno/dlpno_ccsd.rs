// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! Compatibility names delegate to checked pair-projected RI-CCSD and iterative
//! triples. A molecular context with an explicit auxiliary basis is required.
use super::dlpno_mp2::{DLPNOMP2Options, LegacySystem};
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::ContextMethod;
use crate::{
    Atom, ContractedGaussian,
    context::{AuxBasis, PostHFContext},
    scf::HFResult,
};

#[derive(Clone, Debug, PartialEq)]
pub struct DLPNOMCCSDResult {
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub total_energy: f64,
    pub dlpno_mp2_correlation: f64,
}
impl PostHFResult for DLPNOMCCSDResult {
    fn method_name(&self) -> &'static str {
        "DLPNO-CCSD"
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
#[derive(Clone, Debug, PartialEq)]
pub struct DLPNOMCCSDTResult {
    pub reference_energy: f64,
    pub ccsd_correlation_energy: f64,
    pub triples_correlation_energy: f64,
    pub total_energy: f64,
}
impl PostHFResult for DLPNOMCCSDTResult {
    fn method_name(&self) -> &'static str {
        "DLPNO-CCSD(T1)"
    }
    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }
    fn correlation_energy(&self) -> f64 {
        self.ccsd_correlation_energy + self.triples_correlation_energy
    }
    fn total_energy(&self) -> f64 {
        self.total_energy
    }
}
pub struct DLPNOMCCSD<'a> {
    pub hf: &'a HFResult,
    pub options: DLPNOMP2Options,
    system: LegacySystem<'a>,
}
pub struct DLPNOMCCSDT<'a> {
    pub hf: &'a HFResult,
    pub options: DLPNOMP2Options,
    system: LegacySystem<'a>,
}
macro_rules! constructors {
    ($name:ident) => {
        impl<'a> $name<'a> {
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
    };
}
constructors!(DLPNOMCCSD);
constructors!(DLPNOMCCSDT);
impl PostHFMethod for DLPNOMCCSD<'_> {
    type Output = DLPNOMCCSDResult;
    fn compute(&self) -> Result<Self::Output, String> {
        let options = self.options.typed().map_err(|e| e.to_string())?;
        self.system
            .run(self.hf, |ctx| {
                let cc = super::LocalCcsd::from_context(ctx, &options)?.compute_checked()?;
                Ok(DLPNOMCCSDResult {
                    reference_energy: cc.e_scf,
                    correlation_energy: cc.correlation_energy(),
                    total_energy: cc.total_energy(),
                    dlpno_mp2_correlation: cc.e_lmp2,
                })
            })
            .map_err(|e| e.to_string())
    }
}
impl PostHFMethod for DLPNOMCCSDT<'_> {
    type Output = DLPNOMCCSDTResult;
    fn compute(&self) -> Result<Self::Output, String> {
        let options = self.options.typed().map_err(|e| e.to_string())?;
        self.system
            .run(self.hf, |ctx| {
                let cc = super::LocalCcsdT1::from_context(ctx, &options)?
                    .compute()
                    .map_err(|detail| crate::error::HfError::PostHf {
                        method: "DLPNO-CCSD(T1)".into(),
                        detail,
                    })?;
                Ok(DLPNOMCCSDTResult {
                    reference_energy: cc.reference_energy(),
                    ccsd_correlation_energy: cc.ccsd.correlation_energy(),
                    triples_correlation_energy: cc.triples.energy,
                    total_energy: cc.total_energy(),
                })
            })
            .map_err(|e| e.to_string())
    }
}
