// SPDX-License-Identifier: Apache-2.0
//! Excitation-generated MRCISD and Davidson +Q, with HF or CASSCF orbitals.
use crate::context::{MoIntegralCache, PostHFContext, Reference};
use crate::error::HfError;
use crate::fci::engine::{FciSolverOptions, FciState};
use crate::fci::excitations::{Determinant, cas_references, singles_doubles, spin_complete};
use crate::fci::subspace::SubspaceHamiltonian;
use crate::matrix::Matrix;
use crate::mo_integrals::{MoEriTensor, transform_1e_ao_to_mo, transform_eri_ao_to_mo};
use crate::options::{ActiveSpace, PostHFOptions};
use crate::output::{PostHFOutput, State};
use crate::post_hf::casscf_engine::solve_cas;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::{ContextMethod, MethodId, MethodMeta};
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct MrciComputed {
    pub output: PostHFOutput,
    pub states: Vec<FciState>,
    pub determinants: Vec<Determinant>,
    pub reference_weights: Vec<f64>,
    pub davidson_corrections: Vec<f64>,
    pub n_reference_determinants: usize,
    pub mo_coefficients: Matrix,
    pub h_mo: Matrix,
    pub mo_eri: Arc<MoEriTensor>,
    pub nuclear_energy: f64,
}
impl MrciComputed {
    /// Variational MRCISD density; +Q does not define a corrected wavefunction.
    pub fn two_rdm(&self, root: usize) -> Result<Vec<f64>, HfError> {
        let state = self
            .states
            .get(root)
            .ok_or_else(|| HfError::InvalidInput("MRCI root is unavailable".into()))?;
        let engine = SubspaceHamiltonian::new(
            &self.h_mo,
            &self.mo_eri,
            self.nuclear_energy,
            self.determinants.clone(),
        )?;
        engine.rdm(&state.vector, 2)
    }
}
impl PostHFResult for MrciComputed {
    fn method_name(&self) -> &'static str {
        if self.output.method == "MRCISD+Q" {
            "MRCISD+Q"
        } else if self.output.method == "CISD" {
            "CISD"
        } else {
            "MRCISD"
        }
    }
    fn reference_energy(&self) -> f64 {
        self.output.e_scf
    }
    fn correlation_energy(&self) -> f64 {
        self.output.e_corr
    }
    fn total_energy(&self) -> f64 {
        self.output.e_total
    }
    fn one_rdm(&self) -> Option<&Matrix> {
        self.output.one_rdm.as_ref()
    }
    fn to_output(&self) -> Result<PostHFOutput, HfError> {
        Ok(self.output.clone())
    }
}

pub fn compute_mrci(
    reference: Reference<'_>,
    options: &PostHFOptions,
    cache: Option<&MoIntegralCache<'_>>,
    plus_q: bool,
) -> Result<MrciComputed, HfError> {
    options.validate()?;
    let active = options
        .active
        .as_ref()
        .ok_or_else(|| HfError::InvalidInput("MRCI requires an explicit ActiveSpace".into()))?;
    let mut ref_options = options.clone();
    // A single HF determinant defines the common reference of every CISD root.
    if active.n_orb == 0 {
        ref_options.root = 0;
        ref_options.n_roots = 1;
        ref_options.cas.state_weights.clear();
    }
    let cas = solve_cas(
        reference,
        &ref_options,
        cache,
        options.cas.optimize_orbitals,
    )?;
    let c = cas.mo_coefficients;
    let n = c.cols;
    let (na, nb) = reference.occupations();
    let refs = cas_references(&cas.active_space, n, 2_000_000)?;
    let reference_set = refs.iter().copied().collect::<HashSet<_>>();
    let generated = singles_doubles(n, na, nb, &refs, 2_000_000)?;
    let mut dets = spin_complete(n, na, nb, &generated, 2_000_000)?;
    let core_mask = if options.frozen_core == 64 {
        u64::MAX
    } else {
        (1u64 << options.frozen_core) - 1
    };
    let virtual_mask = if options.frozen_virtual == 0 {
        0
    } else {
        let all = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        let allowed = (1u64 << (n - options.frozen_virtual)) - 1;
        all ^ allowed
    };
    dets.retain(|&(a, b)| {
        a.0 & core_mask == core_mask
            && b.0 & core_mask == core_mask
            && (a.0 | b.0) & virtual_mask == 0
    });
    let h = transform_1e_ao_to_mo(&c, reference.core());
    // Reuse the cached full tensor only when it is expressed in these same MOs.
    let eri = if c == *reference.coefficients(0) {
        cache
            .map(|v| v.full().cloned())
            .transpose()?
            .unwrap_or_else(|| Arc::new(transform_eri_ao_to_mo(&c, reference.eri())))
    } else {
        Arc::new(transform_eri_ao_to_mo(&c, reference.eri()))
    };
    let engine = SubspaceHamiltonian::new(&h, &eri, reference.nuclear_energy(), dets)?;
    let sz = (na as f64 - nb as f64).abs() * 0.5;
    let solution = engine.solve(&FciSolverOptions {
        n_roots: options.n_roots,
        max_iterations: options.max_iter,
        residual_tol: options.residual_tol,
        target_s_squared: Some(options.cas.target_s_squared.unwrap_or(sz * (sz + 1.0))),
        ..Default::default()
    })?;
    let mut weights = Vec::new();
    let mut corrections = Vec::new();
    let mut states = Vec::new();
    for (root, state) in solution.states.iter().enumerate() {
        let weight = engine
            .determinants
            .iter()
            .zip(&state.vector)
            .filter(|(d, _)| reference_set.contains(d))
            .map(|(_, &v)| v * v)
            .sum::<f64>();
        let e_ref = cas.states.get(root).unwrap_or(&cas.states[0]).energy;
        let correction = if plus_q {
            (1.0 - weight) * (state.energy - e_ref)
        } else {
            0.0
        };
        weights.push(weight);
        corrections.push(correction);
        states.push(State {
            energy: state.energy + correction,
            s_squared: Some(state.s_squared),
            weight: 0.0,
            converged: true,
        });
    }
    states[options.root].weight = 1.0;
    let method = if plus_q { "MRCISD+Q" } else { "MRCISD" };
    let mut output = PostHFOutput::new(method, reference.energy(), states[options.root].energy)?;
    output.states = states;
    output.select_root(options.root)?;
    output.one_rdm = Some(Matrix::from_vec(
        n,
        n,
        engine.rdm(&solution.states[options.root].vector, 1)?,
    ));
    output.components.insert(
        if options.cas.optimize_orbitals {
            "E_CASSCF"
        } else {
            "E_CASCI"
        }
        .into(),
        cas.states
            .get(options.root)
            .unwrap_or(&cas.states[0])
            .energy,
    );
    output
        .components
        .insert("E_MRCISD".into(), solution.states[options.root].energy);
    output
        .components
        .insert("Delta_E_Q".into(), corrections[options.root]);
    output
        .components
        .insert("reference_weight".into(), weights[options.root]);
    output.validate()?;
    Ok(MrciComputed {
        output,
        states: solution.states,
        determinants: engine.determinants,
        reference_weights: weights,
        davidson_corrections: corrections,
        n_reference_determinants: refs.len(),
        mo_coefficients: c,
        h_mo: h,
        mo_eri: eri,
        nuclear_energy: reference.nuclear_energy(),
    })
}

macro_rules! mrci_adapter {
    ($name:ident,$method:ident,$q:literal) => {
        pub struct $name<'a> {
            ctx: &'a PostHFContext<'a>,
            options: PostHFOptions,
        }
        impl MethodMeta for $name<'_> {
            const METHOD: MethodId = MethodId::$method;
        }
        impl<'a> ContextMethod<'a> for $name<'a> {
            fn from_context(
                ctx: &'a PostHFContext<'a>,
                options: &PostHFOptions,
            ) -> Result<Self, HfError> {
                options.validate()?;
                if options.active.is_none() {
                    return Err(HfError::InvalidInput(
                        "MRCI requires an explicit ActiveSpace".into(),
                    ));
                }
                Ok(Self {
                    ctx,
                    options: options.clone(),
                })
            }
        }
        impl PostHFMethod for $name<'_> {
            type Output = MrciComputed;
            fn compute(&self) -> Result<MrciComputed, String> {
                compute_mrci(self.ctx.reference, &self.options, Some(&self.ctx.ints), $q)
                    .map_err(|e| e.to_string())
            }
        }
    };
}
mrci_adapter!(Mrci, Mrci, false);
mrci_adapter!(MrciQ, MrciQ, true);

pub struct SingleReferenceCi<'a> {
    ctx: &'a PostHFContext<'a>,
    options: PostHFOptions,
}
impl MethodMeta for SingleReferenceCi<'_> {
    const METHOD: MethodId = MethodId::Cisd;
}
impl<'a> ContextMethod<'a> for SingleReferenceCi<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        ctx.rhf()?;
        options.validate()?;
        let mut options = options.clone();
        options.active = Some(ActiveSpace::new(0, 0, 0));
        options.cas.optimize_orbitals = false;
        Ok(Self { ctx, options })
    }
}
impl PostHFMethod for SingleReferenceCi<'_> {
    type Output = MrciComputed;
    fn compute(&self) -> Result<MrciComputed, String> {
        let mut result = compute_mrci(
            self.ctx.reference,
            &self.options,
            Some(&self.ctx.ints),
            false,
        )
        .map_err(|e| e.to_string())?;
        result.output.method = "CISD".into();
        Ok(result)
    }
}
