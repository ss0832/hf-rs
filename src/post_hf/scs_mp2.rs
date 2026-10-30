// SPDX-License-Identifier: Apache-2.0
use super::mp2::{RHFMP2, UHFMP2};
use super::ri_mp2::RiMp2;
use super::{PostHFMethod, PostHFResult};
use crate::context::{PostHFContext, RefKind};
use crate::error::HfError;
use crate::options::PostHFOptions;
use crate::registry::{ContextMethod, MethodId, MethodMeta};

enum Engine<'a> {
    Rhf(RHFMP2<'a>),
    Uhf(UHFMP2<'a>),
    Ri(Box<RiMp2<'a>>),
}
pub struct ScsMp2<'a> {
    engine: Engine<'a>,
    same: f64,
    opposite: f64,
}
pub struct ScsMp2Result {
    pub e_scf: f64,
    pub e_same: f64,
    pub e_opposite: f64,
    pub e_corr: f64,
}
impl PostHFResult for ScsMp2Result {
    fn method_name(&self) -> &'static str {
        "SCS-MP2"
    }
    fn reference_energy(&self) -> f64 {
        self.e_scf
    }
    fn correlation_energy(&self) -> f64 {
        self.e_corr
    }
    fn to_output(&self) -> Result<crate::output::PostHFOutput, HfError> {
        let mut out = crate::output::PostHFOutput::new("SCS-MP2", self.e_scf, self.total_energy())?;
        out.components.insert("E_MP2_same_spin".into(), self.e_same);
        out.components
            .insert("E_MP2_opposite_spin".into(), self.e_opposite);
        Ok(out)
    }
}
impl MethodMeta for ScsMp2<'_> {
    const METHOD: MethodId = MethodId::ScsMp2;
}
impl<'a> ContextMethod<'a> for ScsMp2<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        let engine = if options.density_fitting {
            Engine::Ri(Box::new(RiMp2::from_context(ctx, options)?))
        } else {
            match ctx.reference.kind() {
                RefKind::Rhf => Engine::Rhf(RHFMP2::from_context(ctx, options)?),
                RefKind::Uhf => Engine::Uhf(UHFMP2::from_context(ctx, options)?),
            }
        };
        Ok(Self {
            engine,
            same: options.scs_same_spin,
            opposite: options.scs_opposite_spin,
        })
    }
}
impl PostHFMethod for ScsMp2<'_> {
    type Output = ScsMp2Result;
    fn compute(&self) -> Result<ScsMp2Result, String> {
        let (e_scf, e_same, e_opposite) = match &self.engine {
            Engine::Rhf(engine) => {
                let r = engine.compute()?;
                (r.reference_energy, r.e_same_spin, r.e_opposite_spin)
            }
            Engine::Uhf(engine) => {
                let r = engine.compute()?;
                (
                    r.reference_energy,
                    r.e_alpha_alpha + r.e_beta_beta,
                    r.e_alpha_beta,
                )
            }
            Engine::Ri(engine) => {
                let r = engine.compute()?;
                (r.e_scf, r.e_same_spin, r.e_opposite_spin)
            }
        };
        Ok(ScsMp2Result {
            e_scf,
            e_same,
            e_opposite,
            e_corr: self.same * e_same + self.opposite * e_opposite,
        })
    }
}
