// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! SCS-MP3: c_SS E2_SS + c_OS E2_OS + c3 E3 (Grimme, 2003).
use crate::{
    context::{PostHFContext, RefKind},
    error::HfError,
    options::PostHFOptions,
    post_hf::{PostHFMethod, PostHFResult, RHFMP3, UHFMP3, scs_mp2::ScsMp2},
    registry::{ContextMethod, MethodId, MethodMeta},
};
enum ThirdOrder<'a> {
    Rhf(RHFMP3<'a>),
    Uhf(UHFMP3<'a>),
}
pub struct ScsMp3<'a> {
    second: ScsMp2<'a>,
    third: ThirdOrder<'a>,
    scale: f64,
}
#[derive(Clone, Debug)]
pub struct ScsMp3Result {
    pub e_scf: f64,
    pub e_second_scaled: f64,
    pub e_same: f64,
    pub e_opposite: f64,
    /// Pure third-order increment, excluding the second-order energy.
    pub e_third: f64,
    pub third_scale: f64,
}
impl PostHFResult for ScsMp3Result {
    fn method_name(&self) -> &'static str {
        "SCS-MP3"
    }
    fn reference_energy(&self) -> f64 {
        self.e_scf
    }
    fn correlation_energy(&self) -> f64 {
        self.e_second_scaled + self.third_scale * self.e_third
    }
    fn to_output(&self) -> Result<crate::output::PostHFOutput, HfError> {
        let mut out =
            crate::output::PostHFOutput::new(self.method_name(), self.e_scf, self.total_energy())?;
        for (key, value) in [
            ("E_MP2_same_spin", self.e_same),
            ("E_MP2_opposite_spin", self.e_opposite),
            ("E_SCS_MP2", self.e_second_scaled),
            ("E_MP3_increment", self.e_third),
            ("E_scaled_MP3_increment", self.third_scale * self.e_third),
        ] {
            out.components.insert(key.into(), value);
        }
        Ok(out)
    }
}
impl MethodMeta for ScsMp3<'_> {
    const METHOD: MethodId = MethodId::ScsMp3;
}
impl<'a> ContextMethod<'a> for ScsMp3<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        if options.active.is_some() || options.root != 0 || options.n_roots != 1 {
            return Err(HfError::InvalidInput(
                "SCS-MP3 needs one canonical single-reference ground state".into(),
            ));
        }
        let third = match ctx.reference.kind() {
            RefKind::Rhf => ThirdOrder::Rhf(RHFMP3::from_context(ctx, options)?),
            RefKind::Uhf => ThirdOrder::Uhf(UHFMP3::from_context(ctx, options)?),
        };
        Ok(Self {
            second: ScsMp2::from_context(ctx, options)?,
            third,
            scale: options.scs_third_order,
        })
    }
}
impl PostHFMethod for ScsMp3<'_> {
    type Output = ScsMp3Result;
    fn compute(&self) -> Result<Self::Output, String> {
        let second = self.second.compute()?;
        let third = match &self.third {
            ThirdOrder::Rhf(engine) => engine.compute()?,
            ThirdOrder::Uhf(engine) => engine.compute()?,
        };
        let out = ScsMp3Result {
            e_scf: second.e_scf,
            e_second_scaled: second.e_corr,
            e_same: second.e_same,
            e_opposite: second.e_opposite,
            e_third: third.mp3_correlation_energy,
            third_scale: self.scale,
        };
        if !out.correlation_energy().is_finite() {
            return Err("nonfinite SCS-MP3 correlation energy".into());
        }
        Ok(out)
    }
}
