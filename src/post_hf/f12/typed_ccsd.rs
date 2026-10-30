// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Typed registry adapters for the coupled fixed-cusp CCSD-F12 equations.
use super::{CCSDF12Result, CCSDF12Variant, RHFCCSDF12, RHFMP2F12};
use crate::{
    context::PostHFContext,
    error::HfError,
    options::{F12Approximation, PostHFOptions},
    post_hf::{PostHFMethod, cc::CCSDOptions},
    registry::{ContextMethod, MethodId, MethodMeta},
};

pub struct CcsdF12Method<'a, const B: bool, const TRIPLES: bool> {
    pub engine: RHFCCSDF12<'a>,
}
impl<const B: bool, const T: bool> MethodMeta for CcsdF12Method<'_, B, T> {
    const METHOD: MethodId = match (B, T) {
        (false, false) => MethodId::CcsdF12a,
        (true, false) => MethodId::CcsdF12b,
        (false, true) => MethodId::CcsdTF12a,
        (true, true) => MethodId::CcsdTF12b,
    };
}
impl<'a, const B: bool, const T: bool> ContextMethod<'a> for CcsdF12Method<'a, B, T> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        // Reuse the checked RHF/frozen/DF/CABS option mapping. Unsupported
        // combinations fail before any expensive integral construction.
        options.validate()?;
        if options.f12.approximation != F12Approximation::ApproximationC
            || (options.f12.triples_scaling && !T)
        {
            return Err(HfError::Unsupported {
                method: Self::METHOD.name().into(),
                detail: "CCSD-F12 requires 3C; triples scaling requires a (T*) method".into(),
            });
        }
        let mp2 = RHFMP2F12::from_context(ctx, options)?;
        let mut engine = RHFCCSDF12::new(
            ctx.rhf()?,
            ctx.atoms,
            ctx.basis,
            if B {
                CCSDF12Variant::F12b
            } else {
                CCSDF12Variant::F12a
            },
        )
        .with_triples(T)
        .with_f12_options(mp2.options)
        .with_ccsd_options(CCSDOptions {
            max_iterations: options.max_iter,
            energy_tol: options.energy_tol,
            residual_tol: options.residual_tol,
            ..Default::default()
        });
        engine.integral_cache = Some(&ctx.ints);
        Ok(Self { engine })
    }
}
impl<const B: bool, const T: bool> PostHFMethod for CcsdF12Method<'_, B, T> {
    type Output = CCSDF12Result;
    fn compute(&self) -> Result<Self::Output, String> {
        self.engine.compute()
    }
}
