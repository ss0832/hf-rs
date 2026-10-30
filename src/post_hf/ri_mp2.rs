// SPDX-License-Identifier: Apache-2.0
//! Canonical RI-MP2 over the context's common Coulomb-metric DF tensor.
use super::{PostHFMethod, PostHFResult};
use crate::context::{PostHFContext, orbital_denominator};
use crate::error::HfError;
use crate::options::PostHFOptions;
use crate::registry::{ContextMethod, MethodId, MethodMeta};

#[derive(Clone, Debug)]
pub struct RiMp2Result {
    pub e_scf: f64,
    pub e_same_spin: f64,
    pub e_opposite_spin: f64,
}
impl PostHFResult for RiMp2Result {
    fn method_name(&self) -> &'static str {
        "RI-MP2"
    }
    fn reference_energy(&self) -> f64 {
        self.e_scf
    }
    fn correlation_energy(&self) -> f64 {
        self.e_same_spin + self.e_opposite_spin
    }
    fn to_output(&self) -> Result<crate::output::PostHFOutput, HfError> {
        let mut out = crate::output::PostHFOutput::new("RI-MP2", self.e_scf, self.total_energy())?;
        out.components
            .insert("E_same_spin".into(), self.e_same_spin);
        out.components
            .insert("E_opposite_spin".into(), self.e_opposite_spin);
        Ok(out)
    }
}
pub struct RiMp2<'a> {
    ctx: &'a PostHFContext<'a>,
    options: PostHFOptions,
}
impl MethodMeta for RiMp2<'_> {
    const METHOD: MethodId = MethodId::RiMp2;
}
impl<'a> ContextMethod<'a> for RiMp2<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        let hf = ctx.rhf()?;
        if ctx.aux.is_none() {
            return Err(HfError::InvalidInput(
                "RI-MP2 requires an auxiliary basis".into(),
            ));
        }
        if options.frozen_core > hf.n_occ
            || options.frozen_virtual > hf.mo_coefficients.cols - hf.n_occ
        {
            return Err(HfError::InvalidInput(
                "frozen RI-MP2 partition exceeds reference space".into(),
            ));
        }
        Ok(Self {
            ctx,
            options: options.clone(),
        })
    }
}
impl PostHFMethod for RiMp2<'_> {
    type Output = RiMp2Result;
    fn compute(&self) -> Result<RiMp2Result, String> {
        let hf = self.ctx.rhf().map_err(|e| e.to_string())?;
        let b = self.ctx.ints.df(0).map_err(|e| e.to_string())?;
        let eps = &hf.orbital_energies;
        let mut same = 0.0;
        let mut opposite = 0.0;
        for i in self.options.frozen_core..hf.n_occ {
            for j in self.options.frozen_core..hf.n_occ {
                for a in hf.n_occ..eps.len() - self.options.frozen_virtual {
                    for c in hf.n_occ..eps.len() - self.options.frozen_virtual {
                        let gap = orbital_denominator(eps[i], eps[j], eps[a], eps[c])
                            .map_err(|e| e.to_string())?;
                        let v = b.eri(i, a, j, c);
                        let exchange = b.eri(i, c, j, a);
                        same += 0.5 * (v - exchange).powi(2) / gap;
                        opposite += v * v / gap;
                    }
                }
            }
        }
        Ok(RiMp2Result {
            e_scf: hf.total_energy,
            e_same_spin: same,
            e_opposite_spin: opposite,
        })
    }
}
