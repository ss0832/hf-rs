// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Shared CAS reference and typed internally contracted CASPT2 adapter.

use crate::context::{MoIntegralCache, PostHFContext, Reference};
use crate::error::HfError;
use crate::matrix::Matrix;
use crate::mo_integrals::{transform_1e_ao_to_mo, transform_eri_ao_to_mo};
use crate::options::PostHFOptions;
use crate::output::PostHFOutput;
use crate::post_hf::caspt2_engine::{
    IcCaspt2Options, IcCaspt2Result, generalized_fock, solve_ic_caspt2,
};
use crate::post_hf::casscf_engine::{solve_cas, solve_cas_in_orbitals};
use crate::registry::{ContextMethod, MethodId, MethodMeta};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct Caspt2Computed {
    pub output: PostHFOutput,
    pub reference_cas_energy: f64,
    pub mo_coefficients: Matrix,
    pub equations: IcCaspt2Result,
}

pub fn solve_caspt2(
    context: &PostHFContext<'_>,
    options: &PostHFOptions,
) -> Result<Caspt2Computed, HfError> {
    compute_caspt2(context.reference, options, Some(&context.ints), None)
}

pub(crate) fn compute_caspt2(
    reference: Reference<'_>,
    options: &PostHFOptions,
    cache: Option<&MoIntegralCache<'_>>,
    supplied_orbitals: Option<&Matrix>,
) -> Result<Caspt2Computed, HfError> {
    options.validate()?;
    if options.n_roots != 1
        || options.root != 0
        || !options.cas.state_weights.is_empty()
        || options.density_fitting
    {
        return Err(HfError::Unsupported {
            method: "CASPT2".into(),
            detail: "only a single-state conventional-integral CAS reference is implemented".into(),
        });
    }
    let mut cas = if let Some(c) = supplied_orbitals {
        solve_cas_in_orbitals(reference, c, options)?
    } else {
        solve_cas(reference, options, cache, options.cas.optimize_orbitals)?
    };
    let mut h = transform_1e_ao_to_mo(&cas.mo_coefficients, reference.core());
    let mut eri = if cas.mo_coefficients == *reference.coefficients(0) {
        cache
            .map(|v| v.full().cloned())
            .transpose()?
            .unwrap_or_else(|| {
                Arc::new(transform_eri_ao_to_mo(
                    &cas.mo_coefficients,
                    reference.eri(),
                ))
            })
    } else {
        Arc::new(transform_eri_ao_to_mo(
            &cas.mo_coefficients,
            reference.eri(),
        ))
    };
    // The IPEA convention is orbital dependent. Use pseudocanonical orbitals
    // in each unfrozen/frozen inactive, active and external block.
    if options.cas.ipea_shift != 0.0 {
        let n = h.rows;
        let nc = cas.active_space.inactive.len();
        let na = cas.active_space.active.len();
        let density = cas
            .output
            .one_rdm
            .as_ref()
            .ok_or_else(|| HfError::Response("CAS reference density is missing".into()))?;
        let f = generalized_fock(&h, &eri, density);
        let ranges = [
            (0, options.frozen_core),
            (options.frozen_core, nc),
            (nc, nc + na),
            (nc + na, n - options.frozen_virtual),
            (n - options.frozen_virtual, n),
        ];
        let mut u = Matrix::identity(n);
        for (first, last) in ranges {
            let size = last - first;
            if size == 0 {
                continue;
            }
            let mut block = Matrix::zeros(size, size);
            for p in 0..size {
                for q in 0..size {
                    block[(p, q)] = f[(first + p, first + q)];
                }
            }
            let (_, rotate) = block.eigh();
            for p in 0..size {
                for q in 0..size {
                    u[(first + p, first + q)] = rotate[(p, q)];
                }
            }
        }
        let c = cas.mo_coefficients.dot(&u);
        let next = solve_cas_in_orbitals(reference, &c, options)?;
        if (next.output.e_total - cas.output.e_total).abs() > 1e-9 {
            return Err(HfError::Response(
                "pseudocanonical CAS reference changed its energy".into(),
            ));
        }
        cas = next;
        h = transform_1e_ao_to_mo(&cas.mo_coefficients, reference.core());
        eri = Arc::new(transform_eri_ao_to_mo(
            &cas.mo_coefficients,
            reference.eri(),
        ));
    }
    let equations = solve_ic_caspt2(
        &h,
        &eri,
        &cas.active_space,
        &cas.states[0].vector,
        &cas.state_one_rdms[0],
        &IcCaspt2Options {
            frozen_core: options.frozen_core,
            frozen_virtual: options.frozen_virtual,
            metric_tol: options.cas.caspt2_metric_tol,
            residual_tol: options.residual_tol,
            max_iterations: options.max_iter,
            ipea_shift: options.cas.ipea_shift,
            imaginary_shift: options.cas.imaginary_shift,
            real_shift: options.cas.real_shift,
        },
    )?;
    let e_ref = cas.output.e_total;
    let mut output = PostHFOutput::new(
        "CASPT2",
        reference.energy(),
        e_ref + equations.correlation_energy,
    )?;
    output.states[0].s_squared = Some(cas.states[0].s_squared);
    output.components.insert(
        if options.cas.optimize_orbitals || supplied_orbitals.is_some() {
            "E_CASSCF"
        } else {
            "E_CASCI"
        }
        .into(),
        e_ref,
    );
    for (i, energy) in equations.class_energies.iter().enumerate() {
        output.components.insert(
            format!("E_CASPT2_{}", ["A", "B", "C", "D", "E", "F", "G", "H"][i]),
            *energy,
        );
    }
    output
        .components
        .insert("E_PT2".into(), equations.correlation_energy);
    output
        .components
        .insert("E_PT2_uncorrected".into(), equations.uncorrected_energy);
    output
        .components
        .insert("reference_weight".into(), equations.reference_weight);
    output
        .components
        .insert("caspt2_residual".into(), equations.residual_norm);
    output
        .components
        .insert("caspt2_iterations".into(), equations.iterations as f64);
    output.components.insert(
        "contracted_dimension".into(),
        equations.retained_dimensions.iter().sum::<usize>() as f64,
    );
    // The CAS density is not a relaxed CASPT2 density.
    output.one_rdm = None;
    output.validate()?;
    Ok(Caspt2Computed {
        output,
        reference_cas_energy: e_ref,
        mo_coefficients: cas.mo_coefficients,
        equations,
    })
}

impl super::PostHFResult for Caspt2Computed {
    fn method_name(&self) -> &'static str {
        "CASPT2"
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
    fn to_output(&self) -> Result<PostHFOutput, HfError> {
        Ok(self.output.clone())
    }
}
pub struct Caspt2<'a> {
    context: &'a PostHFContext<'a>,
    options: PostHFOptions,
}
impl MethodMeta for Caspt2<'_> {
    const METHOD: MethodId = MethodId::Caspt2;
}
impl<'a> ContextMethod<'a> for Caspt2<'a> {
    fn from_context(
        context: &'a PostHFContext<'a>,
        options: &PostHFOptions,
    ) -> Result<Self, HfError> {
        options.validate()?;
        Ok(Self {
            context,
            options: options.clone(),
        })
    }
}
impl super::PostHFMethod for Caspt2<'_> {
    type Output = Caspt2Computed;
    fn compute(&self) -> Result<Self::Output, String> {
        solve_caspt2(self.context, &self.options).map_err(|e| e.to_string())
    }
}

/// Independent-geometry numerical verification, not a production analytic gradient.
pub fn numerical_gradient(
    atoms: &[crate::basis::Atom],
    basis: &[crate::basis::ContractedGaussian],
    charge: i32,
    basis_kind: crate::ao::BasisKind,
    options: &PostHFOptions,
    numerical: &super::numerical::NumericalGradientOptions,
) -> Result<super::numerical::NumericalGradientResult, HfError> {
    super::numerical::compute_rhf_numerical_gradient(
        atoms,
        basis,
        charge,
        basis_kind,
        numerical,
        |hf, point, moved| {
            let context = PostHFContext::from_rhf(hf, point, moved, None)?;
            Ok(solve_caspt2(&context, options)?.output.e_total)
        },
    )
}
