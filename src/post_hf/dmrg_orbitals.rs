// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! DMRG-CI and DMRGSCF using the common CAS orbital optimizer.

use crate::context::{MoIntegralCache, PostHFContext, Reference};
use crate::error::HfError;
use crate::matrix::Matrix;
use crate::options::{PostHFOptions, ResolvedActiveSpace};
use crate::output::PostHFOutput;
use crate::post_hf::casscf_engine::{ActiveSolution, solve_cas_with_solver};
use crate::post_hf::dmrg_engine::{MpsSolution, MpsSolverOptions, solve_mps};
use crate::registry::{ContextMethod, MethodId, MethodMeta};

#[derive(Clone, Debug)]
pub struct DmrgDensityResult {
    pub output: PostHFOutput,
    pub mo_coefficients: Matrix,
    pub active_space: ResolvedActiveSpace,
    pub active_one_rdm: Matrix,
    pub active_two_rdm: Vec<f64>,
    pub mps: MpsSolution,
    pub orbital_gradient_max: f64,
    pub orbital_iterations: usize,
}

pub fn solve_dmrg(
    context: &PostHFContext<'_>,
    options: &PostHFOptions,
    optimize: bool,
) -> Result<DmrgDensityResult, HfError> {
    solve_dmrg_reference(context.reference, options, optimize, Some(&context.ints))
}

pub(crate) fn solve_dmrg_reference(
    reference: Reference<'_>,
    options: &PostHFOptions,
    optimize: bool,
    cache: Option<&MoIntegralCache<'_>>,
) -> Result<DmrgDensityResult, HfError> {
    options.validate()?;
    if options.n_roots != 1
        || options.root != 0
        || !options.cas.state_weights.is_empty()
        || options.cas.target_s_squared.is_some()
        || options.dmrg.noise != 0.0
    {
        return Err(HfError::Unsupported { method: "DMRG".into(), detail: "only a single fixed-Sz root, zero noise and no explicit S2 projection are implemented".into() });
    }
    let solver = MpsSolverOptions {
        bond_dimensions: options.dmrg.bond_dimensions.clone(),
        sweeps: options.dmrg.sweeps,
        one_site_sweeps: options.dmrg.one_site_sweeps,
        energy_tol: options.energy_tol,
        residual_tol: options.residual_tol,
        truncation_tol: options.dmrg.truncation_tol,
        max_davidson_iterations: options.max_iter,
    };
    let mut active_solver = |h: &Matrix,
                             g: &crate::mo_integrals::MoEriTensor,
                             space: &ResolvedActiveSpace,
                             core_energy: f64,
                             _: &PostHFOptions| {
        let mps = solve_mps(h, g, space.n_alpha, space.n_beta, core_energy, &solver)?;
        if !mps.converged {
            return Err(HfError::PostHf {
                method: "DMRG".into(),
                detail: format!(
                    "energy did not converge with {} two-site and {} one-site passes",
                    solver.sweeps, solver.one_site_sweeps
                ),
            });
        }
        let (one, two) = mps.state.rdms()?;
        Ok(ActiveSolution {
            energies: vec![(mps.energy, mps.state.spin_squared()?)],
            states: mps,
            one: vec![one],
            two: vec![two],
        })
    };
    let result = solve_cas_with_solver(reference, options, cache, optimize, &mut active_solver)?;
    let mut output = result.output;
    output.method = if optimize { "DMRGSCF" } else { "DMRG" }.into();
    output.components.remove("E_CASCI");
    output.components.remove("E_CASSCF");
    output.components.insert(
        if optimize { "E_DMRGSCF" } else { "E_DMRGCI" }.into(),
        output.e_total,
    );
    output.components.insert(
        "dmrg_sweeps".into(),
        (result.states.sweep_energies.len() + result.states.one_site_energies.len()) as f64,
    );
    output.components.insert(
        "dmrg_two_site_sweeps".into(),
        result.states.sweep_energies.len() as f64,
    );
    output.components.insert(
        "dmrg_one_site_sweeps".into(),
        result.states.one_site_energies.len() as f64,
    );
    output.components.insert(
        "dmrg_max_discarded_weight".into(),
        result
            .states
            .discarded_weights
            .iter()
            .copied()
            .fold(0.0, f64::max),
    );
    output
        .components
        .insert("orbital_gradient_max".into(), result.orbital_gradient_max);
    output
        .components
        .insert("orbital_iterations".into(), result.iterations as f64);
    Ok(DmrgDensityResult {
        output,
        mo_coefficients: result.mo_coefficients,
        active_space: result.active_space,
        active_one_rdm: result.averaged_one_rdm,
        active_two_rdm: result.averaged_two_rdm,
        mps: result.states,
        orbital_gradient_max: result.orbital_gradient_max,
        orbital_iterations: result.iterations,
    })
}

impl super::PostHFResult for DmrgDensityResult {
    fn method_name(&self) -> &'static str {
        if self.output.method == "DMRGSCF" {
            "DMRGSCF"
        } else {
            "DMRG"
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
    fn converged(&self) -> bool {
        self.mps.converged
    }
    fn one_rdm(&self) -> Option<&Matrix> {
        self.output.one_rdm.as_ref()
    }
    fn to_output(&self) -> Result<PostHFOutput, HfError> {
        Ok(self.output.clone())
    }
}

pub struct DmrgMethod<'a, const OPTIMIZE: bool> {
    context: &'a PostHFContext<'a>,
    options: PostHFOptions,
}
impl<const OPTIMIZE: bool> MethodMeta for DmrgMethod<'_, OPTIMIZE> {
    const METHOD: MethodId = if OPTIMIZE {
        MethodId::DmrgScf
    } else {
        MethodId::Dmrg
    };
}
impl<'a, const OPTIMIZE: bool> ContextMethod<'a> for DmrgMethod<'a, OPTIMIZE> {
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
impl<const OPTIMIZE: bool> super::PostHFMethod for DmrgMethod<'_, OPTIMIZE> {
    type Output = DmrgDensityResult;
    fn compute(&self) -> Result<Self::Output, String> {
        solve_dmrg(self.context, &self.options, OPTIMIZE).map_err(|e| e.to_string())
    }
}

/// Debugging/verification differences of independently reconverged total energies.
pub fn numerical_gradient(
    optimize: bool,
    atoms: &[crate::basis::Atom],
    basis: &[crate::basis::ContractedGaussian],
    charge: i32,
    basis_kind: crate::ao::BasisKind,
    options: &PostHFOptions,
    numerical: &crate::post_hf::numerical::NumericalGradientOptions,
) -> Result<crate::post_hf::numerical::NumericalGradientResult, HfError> {
    options.validate()?;
    crate::post_hf::numerical::compute_rhf_numerical_gradient(
        atoms,
        basis,
        charge,
        basis_kind,
        numerical,
        |hf, point, moved| {
            let context = PostHFContext::from_rhf(hf, point, moved, None)?;
            Ok(solve_dmrg(&context, options, optimize)?.output.e_total)
        },
    )
}
