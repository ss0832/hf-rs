// SPDX-License-Identifier: Apache-2.0
use crate::context::PostHFContext;
use crate::error::HfError;
use crate::options::PostHFOptions;
use crate::post_hf::{CCSD, PostHFMethod, cc::CCSDTResult};
use crate::registry::{ContextMethod, MethodId, MethodMeta};

pub struct CcsdTriples<'a> {
    ctx: &'a PostHFContext<'a>,
    ccsd: CCSD<'a>,
}
impl MethodMeta for CcsdTriples<'_> {
    const METHOD: MethodId = MethodId::CcsdT;
}
impl<'a> ContextMethod<'a> for CcsdTriples<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        Ok(Self {
            ctx,
            ccsd: CCSD::from_context(ctx, options)?,
        })
    }
}
impl PostHFMethod for CcsdTriples<'_> {
    type Output = CCSDTResult;
    fn compute(&self) -> Result<CCSDTResult, String> {
        let cc = self.ccsd.compute()?;
        super::ccsd_t::compute_ccsd_t_correction_with_integrals(
            self.ctx.rhf().map_err(|e| e.to_string())?,
            &cc,
            self.ctx.ints.full().map_err(|e| e.to_string())?,
        )
    }
}
