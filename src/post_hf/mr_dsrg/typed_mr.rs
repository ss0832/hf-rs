// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Single-state singlet MR-DSRG adapters in semicanonical orbitals.

use super::{
    density_cumulants::DEFAULT_CUMULANT_MEMORY_BUDGET as BUDGET, gno::GnoAlgebra,
    relaxation::solve_reference,
};
use crate::{
    context::PostHFContext,
    error::HfError,
    options::{ActiveSpace, PostHFOptions},
    output::PostHFOutput,
    post_hf::{PostHFMethod, PostHFResult, casscf_engine::solve_cas},
    registry::{ContextMethod, MethodId, MethodMeta},
};

fn unsupported(detail: &str) -> HfError {
    HfError::Unsupported {
        method: "MR-DSRG".into(),
        detail: detail.into(),
    }
}
pub(crate) fn reference_output(
    solution: &super::relaxation::MrReferenceSolution,
    method: &str,
    order: u8,
    scf_energy: f64,
    optimized: bool,
) -> Result<PostHFOutput, HfError> {
    let total = solution.total_energy;
    let mut output = PostHFOutput::new(method, scf_energy, total)?;
    let reference_label = if optimized { "E_CASSCF" } else { "E_CASCI" };
    output
        .components
        .insert(reference_label.into(), solution.initial_reference_energy);
    output
        .components
        .insert("E_unrelaxed".into(), solution.unrelaxed_energy);
    output.components.insert(
        "E_reference_source".into(),
        solution.source_reference_energy,
    );
    output.components.insert(
        "E_reference_relaxation".into(),
        total - solution.unrelaxed_energy,
    );
    output.components.insert(
        "reference_iterations".into(),
        solution.reference_iterations as f64,
    );
    output.components.insert(
        "reference_ci_change_max".into(),
        solution.reference_ci_change,
    );
    output.components.insert(
        "reference_energy_change".into(),
        solution.reference_energy_change,
    );
    output.components.insert(
        "reference_ci_residual".into(),
        solution.reference_ci_residual,
    );
    output
        .components
        .insert("E_PT2".into(), solution.second_order);
    if order == 3 {
        output
            .components
            .insert("E_PT3_increment".into(), solution.third_order);
    }
    output
        .components
        .insert("lambda2_norm".into(), solution.lambda2_norm);
    output
        .components
        .insert("lambda3_norm".into(), solution.lambda3_norm);
    if order >= 4 {
        output
            .components
            .insert("source_residual_max".into(), solution.source_residual);
        output
            .components
            .insert("amplitude_residual_max".into(), solution.amplitude_residual);
        output
            .components
            .insert("bch_order".into(), solution.bch_order as f64);
        output.components.insert(
            "amplitude_iterations".into(),
            solution.amplitude_iterations as f64,
        );
    }
    output.validate()?;
    Ok(output)
}

pub struct MrDrivenResult {
    output: PostHFOutput,
    reference_energy: f64,
    method_label: &'static str,
}
impl PostHFResult for MrDrivenResult {
    fn method_name(&self) -> &'static str {
        self.method_label
    }
    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }
    fn correlation_energy(&self) -> f64 {
        self.output.e_total - self.reference_energy
    }
    fn total_energy(&self) -> f64 {
        self.output.e_total
    }
    fn to_output(&self) -> Result<PostHFOutput, HfError> {
        Ok(self.output.clone())
    }
}

pub struct MrDrivenMethod<'a, const ORDER: u8> {
    context: &'a PostHFContext<'a>,
    options: PostHFOptions,
}
impl<const ORDER: u8> MethodMeta for MrDrivenMethod<'_, ORDER> {
    const METHOD: MethodId = match ORDER {
        2 => MethodId::DsrgMrPt2,
        3 => MethodId::DsrgMrPt3,
        5 => MethodId::MrLdsrg2Qc,
        _ => MethodId::MrLdsrg2,
    };
}
impl<'a, const ORDER: u8> ContextMethod<'a> for MrDrivenMethod<'a, ORDER> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        let hf = ctx.rhf()?;
        if !matches!(ORDER, 2..=5)
            || options.density_fitting
            || options.frozen_core != 0
            || options.frozen_virtual != 0
            || options.root != 0
            || options.n_roots != 1
            || options
                .cas
                .target_s_squared
                .is_some_and(|x| x.abs() > 1e-10)
        {
            return Err(unsupported(
                "single RHF singlet only; DF/frozen/state variants are not connected",
            ));
        }
        let n = hf
            .mo_coefficients
            .cols
            .checked_mul(2)
            .ok_or_else(|| unsupported("spin dimension overflow"))?;
        if GnoAlgebra::estimated_workspace_bytes(n)? > BUDGET {
            return Err(unsupported(
                "operator/density workspace estimate exceeds memory budget",
            ));
        }
        let mut options = options.clone();
        if options.active.is_none() {
            options.active = Some(ActiveSpace::new(0, 0, 0));
            options.cas.optimize_orbitals = false;
        }
        let active = options
            .active
            .as_ref()
            .ok_or_else(|| HfError::InvalidInput("missing MR active space".into()))?;
        active.resolve(hf.mo_coefficients.cols, hf.n_occ, hf.n_occ)?;
        if active.n_alpha != active.n_beta {
            return Err(unsupported("requires a spin-balanced singlet CAS"));
        }
        options.cas.target_s_squared = Some(0.);
        Ok(Self {
            context: ctx,
            options,
        })
    }
}
impl<const ORDER: u8> MrDrivenMethod<'_, ORDER> {
    fn evaluate(&self) -> Result<MrDrivenResult, HfError> {
        let hf = self.context.rhf()?;
        let cas = solve_cas(
            self.context.reference,
            &self.options,
            Some(&self.context.ints),
            self.options.cas.optimize_orbitals,
        )?;
        let solution = solve_reference(hf, cas, ORDER, &self.options, BUDGET)?;
        let output = reference_output(
            &solution,
            Self::METHOD.name(),
            ORDER,
            hf.total_energy,
            self.options.cas.optimize_orbitals,
        )?;
        Ok(MrDrivenResult {
            output,
            reference_energy: solution.initial_reference_energy,
            method_label: Self::METHOD.name(),
        })
    }
}
impl<const ORDER: u8> PostHFMethod for MrDrivenMethod<'_, ORDER> {
    type Output = MrDrivenResult;
    fn compute(&self) -> Result<MrDrivenResult, String> {
        self.evaluate().map_err(|e| e.to_string())
    }
}

/// Cholesky or explicit Coulomb-metric DF THREE-MRPT2. Conventional CASSCF may
/// supply the initial orbitals; its full integral workspace is released before
/// factorized CASCI/amplitudes/effective-CAS updates. DF orbital optimization is
/// explicitly unsupported until the common optimizer supports fitted integrals.
pub struct ThreeIndexMrpt2<'a> {
    context: &'a PostHFContext<'a>,
    options: PostHFOptions,
}
impl MethodMeta for ThreeIndexMrpt2<'_> {
    const METHOD: MethodId = MethodId::ThreeDsrgMrPt2;
}
impl<'a> ContextMethod<'a> for ThreeIndexMrpt2<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        let hf = ctx.rhf()?;
        if options.frozen_core != 0
            || options.frozen_virtual != 0
            || options.root != 0
            || options.n_roots != 1
            || (options.density_fitting
                && options.cas.optimize_orbitals
                && options.active.is_some())
            || options
                .cas
                .target_s_squared
                .is_some_and(|s| s.abs() > 1e-10)
        {
            return Err(unsupported(
                "THREE requires a single singlet; frozen/state variants and DF orbital optimization are not connected",
            ));
        }
        let mut options = options.clone();
        if options.active.is_none() {
            options.active = Some(ActiveSpace::new(0, 0, 0));
            options.cas.optimize_orbitals = false;
        }
        let active = options.active.as_ref().unwrap().resolve(
            hf.mo_coefficients.cols,
            hf.n_occ,
            hf.n_occ,
        )?;
        if active.n_alpha != active.n_beta {
            return Err(unsupported("THREE requires a spin-balanced singlet CAS"));
        }
        let n = hf.mo_coefficients.cols as u128;
        let na = 2 * active.active.len();
        let active_bytes = if na == 0 {
            0
        } else {
            GnoAlgebra::estimated_workspace_bytes(na)?
        };
        if 8 * active_bytes as u128 + 64 * n * n + 320 * (na as u128 + 1).pow(4) > BUDGET as u128 {
            return Err(unsupported(
                "THREE active/sector workspace exceeds memory budget",
            ));
        }
        if options.cas.optimize_orbitals {
            let nao = hf.mo_coefficients.rows as u128;
            if 64 * (n.max(nao)).pow(4) > BUDGET as u128 {
                return Err(unsupported(
                    "conventional CASSCF starting-orbital workspace exceeds memory budget; use fixed orbitals for the factorized route",
                ));
            }
        }
        options.cas.target_s_squared = Some(0.);
        Ok(Self {
            context: ctx,
            options,
        })
    }
}
impl PostHFMethod for ThreeIndexMrpt2<'_> {
    type Output = MrDrivenResult;
    fn compute(&self) -> Result<MrDrivenResult, String> {
        self.evaluate().map_err(|e| e.to_string())
    }
}
impl ThreeIndexMrpt2<'_> {
    fn evaluate(&self) -> Result<MrDrivenResult, HfError> {
        use super::three_reference::{cholesky_factors, solve_three_reference};
        let hf = self.context.rhf()?;
        let fitted = if self.options.density_fitting {
            Some(self.context.ints.df_ao()?)
        } else {
            None
        };
        let cholesky = if fitted.is_none() {
            Some(cholesky_factors(
                hf,
                self.options.dsrg.cholesky_tolerance,
                BUDGET,
            )?)
        } else {
            None
        };
        let factors = if let Some(fitted) = fitted {
            fitted.as_ref()
        } else {
            &cholesky.as_ref().unwrap().0
        };
        let initial = if self.options.cas.optimize_orbitals {
            // Do not retain full MO ERIs in the context cache on this route.
            Some(solve_cas(
                self.context.reference,
                &self.options,
                None,
                true,
            )?)
        } else {
            None
        };
        let coefficients = initial.as_ref().map(|cas| &cas.mo_coefficients);
        let (solution, sectors, peak) =
            solve_three_reference(hf, factors, &self.options, coefficients, BUDGET)?;
        let mut output =
            reference_output(&solution, Self::METHOD.name(), 2, hf.total_energy, false)?;
        if let Some(cas) = initial {
            output
                .components
                .insert("E_conventional_CASSCF_orbitals".into(), cas.output.e_total);
        }
        output
            .components
            .insert("factor_rank".into(), factors.rank() as f64);
        output
            .components
            .insert("factorized_sectors".into(), sectors as f64);
        output
            .components
            .insert("peak_sector_terms".into(), peak as f64);
        if let Some((_, residual)) = cholesky {
            output
                .components
                .insert("cholesky_max_residual".into(), residual);
        }
        output.validate()?;
        Ok(MrDrivenResult {
            output,
            reference_energy: solution.initial_reference_energy,
            method_label: Self::METHOD.name(),
        })
    }
}

/// Central nuclear differences, reconverging SCF, CAS and driven amplitudes.
pub fn numerical_gradient(
    method: MethodId,
    atoms: &[crate::basis::Atom],
    basis: &[crate::basis::ContractedGaussian],
    charge: i32,
    basis_kind: crate::ao::BasisKind,
    options: &PostHFOptions,
    numerical: &crate::post_hf::numerical::NumericalGradientOptions,
) -> Result<crate::post_hf::numerical::NumericalGradientResult, HfError> {
    numerical_gradient_with_aux(
        method, atoms, basis, None, charge, basis_kind, options, numerical,
    )
}

/// Numerical validation/debug derivative with an explicit co-moving auxiliary
/// basis. SCF, the DF metric, CASCI and reference relaxation are rebuilt at
/// every displaced geometry. This is not an analytic response implementation.
pub fn numerical_gradient_with_aux(
    method: MethodId,
    atoms: &[crate::basis::Atom],
    basis: &[crate::basis::ContractedGaussian],
    auxiliary: Option<&crate::context::AuxBasis>,
    charge: i32,
    basis_kind: crate::ao::BasisKind,
    options: &PostHFOptions,
    numerical: &crate::post_hf::numerical::NumericalGradientOptions,
) -> Result<crate::post_hf::numerical::NumericalGradientResult, HfError> {
    if !matches!(
        method,
        MethodId::DsrgMrPt2
            | MethodId::DsrgMrPt3
            | MethodId::MrLdsrg2
            | MethodId::MrLdsrg2Qc
            | MethodId::ThreeDsrgMrPt2
            | MethodId::SaDsrgPt2 | MethodId::SaDsrgPt2C | MethodId::MsDsrgPt2 | MethodId::XmsDsrgPt2 | MethodId::DwmsDsrgPt2
    ) {
        return Err(unsupported(
            "numerical adapter requires a supported single-state MR driven method",
        ));
    }
    options.validate()?;
    if options.density_fitting && auxiliary.is_none() {
        return Err(HfError::InvalidInput(
            "DF numerical derivatives require an explicit auxiliary basis".into(),
        ));
    }
    if auxiliary.is_some_and(|aux| aux.functions.iter().any(|g| g.atom_index >= atoms.len())) {
        return Err(HfError::InvalidInput(
            "auxiliary function has an invalid atom index".into(),
        ));
    }
    crate::post_hf::numerical::compute_rhf_numerical_gradient(
        atoms,
        basis,
        charge,
        basis_kind,
        numerical,
        |hf, point, moved| {
            let moved_auxiliary = auxiliary.map(|aux| {
                let mut aux = aux.clone();
                for g in &mut aux.functions {
                    for k in 0..3 {
                        g.center[k] +=
                            point[g.atom_index].position[k] - atoms[g.atom_index].position[k];
                    }
                }
                aux
            });
            let ctx = PostHFContext::from_rhf(hf, point, moved, moved_auxiliary.as_ref())?;
            Ok(crate::registry::create_solver(method, &ctx, options)?
                .compute()?
                .e_total)
        },
    )
}
