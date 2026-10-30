// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Single-Reference Driven Similarity Renormalization Group (`SR-DSRG-PT2`, `SR-DSRG-PT3`,
//! `SR-LDSRG(2)`, `UHF-SR-DSRG-PT2`) and Continuous Flow Similarity Renormalization Group
//! IMSRG(2) ODE solvers with Wegner and White generators.

use crate::context::transform_mixed_eri;
use crate::matrix::Matrix;
use crate::mo_integrals::transform_eri_ao_to_mo;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::scf::{HFResult, UHFResult};
use std::borrow::Cow;

/// Regularized Gaussian denominator $\frac{1 - e^{-s \Delta^2}}{\Delta}$ evaluated smoothly
/// for all $\Delta \in \mathbb{R}$, eliminating intruder-state divergences as $\Delta \to 0$.
#[inline]
pub fn dsrg_regularized_inv_denom(delta: f64, s: f64) -> f64 {
    let x = s * delta * delta;
    if x < 1e-7 {
        // Taylor expansion: s * delta * (1 - x/2 + x^2/6 - x^3/24)
        s * delta * (1.0 - 0.5 * x + x * x / 6.0 - x * x * x / 24.0)
    } else {
        -(-x).exp_m1() / delta
    }
}

/// Renormalized DSRG residual coupling factor $e^{-s \Delta^2}$.
#[inline]
pub fn dsrg_residual_weight(delta: f64, s: f64) -> f64 {
    (-s * delta * delta).exp()
}

/// Result of a Single-Reference Driven Similarity Renormalization Group (`SR-DSRG`) calculation.
#[derive(Clone, Debug)]
pub struct SRDSRGResult {
    pub method_label: &'static str,
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub pt2_energy: f64,
    /// Pure PT3 increment for PT3; full higher-order remainder for LDSRG(2).
    pub pt3_correction: f64,
    pub flow_param_s: f64,
    pub iterations: usize,
    pub converged: bool,
    pub max_t1: f64,
    pub max_t2: f64,
    pub one_rdm_mo: Option<Matrix>,
    pub ldsrg: Option<super::sr_dsrg::LdsrgSolution>,
    pub perturbative: Option<super::sr_dsrg::DsrgPt3Solution>,
}

impl PostHFResult for SRDSRGResult {
    fn converged(&self) -> bool {
        self.converged
    }
    fn to_output(&self) -> Result<crate::output::PostHFOutput, crate::error::HfError> {
        let mut output = crate::output::PostHFOutput::new(
            self.method_label,
            self.reference_energy,
            self.total_energy(),
        )?;
        output.one_rdm = self.one_rdm_mo.clone();
        output.components.insert("E_PT2".into(), self.pt2_energy);
        if self.perturbative.is_some() {
            output
                .components
                .insert("E_PT3_increment".into(), self.pt3_correction);
        }
        if let Some(solution) = &self.ldsrg {
            output
                .components
                .insert("source_residual_max".into(), solution.residual_norm);
            output.components.insert(
                "amplitude_residual_max".into(),
                solution.amplitude_residual_norm,
            );
            output
                .components
                .insert("bch_order".into(), solution.bch_order as f64);
        }
        Ok(output)
    }

    fn method_name(&self) -> &'static str {
        self.method_label
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }

    fn one_rdm(&self) -> Option<&Matrix> {
        self.one_rdm_mo.as_ref()
    }
}

// ============================================================================
// 1. SR-DSRG-PT2 (Closed-Shell Second-Order Driven SRG Perturbation Theory)
// ============================================================================

/// Second-Order Single-Reference Driven Similarity Renormalization Group (`SR-DSRG-PT2`)
/// (J. Chem. Phys. 141, 054109, 2014).
pub struct SRDSRGPT2<'a> {
    hf: &'a HFResult,
    flow_param_s: f64,
    integral_cache: Option<&'a crate::context::MoIntegralCache<'a>>,
}

impl<'a> SRDSRGPT2<'a> {
    pub fn new(hf: &'a HFResult, flow_param_s: f64) -> Self {
        Self {
            hf,
            flow_param_s,
            integral_cache: None,
        }
    }
}

impl<'a> PostHFMethod for SRDSRGPT2<'a> {
    type Output = SRDSRGResult;

    fn compute(&self) -> Result<Self::Output, String> {
        if !self.flow_param_s.is_finite()
            || self.flow_param_s < 0.
            || !self.hf.converged
            || self.hf.n_occ > self.hf.mo_coefficients.cols
            || self.hf.mo_coefficients.cols != self.hf.orbital_energies.len()
        {
            return Err(
                "SR-DSRG-PT2 requires nonnegative finite flow and compatible converged RHF".into(),
            );
        }
        let n_occ = self.hf.n_occ;
        let n_basis = self.hf.orbital_energies.len();
        let n_virt = n_basis - n_occ;
        let s = self.flow_param_s;
        let eps = &self.hf.orbital_energies;

        let f_mo = self
            .hf
            .mo_coefficients
            .transpose()
            .dot(&self.hf.fock_matrix)
            .dot(&self.hf.mo_coefficients);
        let mo_eri = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.full().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(transform_eri_ao_to_mo(
                &self.hf.mo_coefficients,
                &self.hf.eri,
            ))
        };

        let mut e_pt2 = 0.0;
        let mut max_t1 = 0.0_f64;
        let mut max_t2 = 0.0_f64;

        // 1-body contribution (for non-canonical references):
        for i in 0..n_occ {
            for a in 0..n_virt {
                let a_mo = n_occ + a;
                let delta_ia = eps[i] - eps[a_mo];
                let f_ia = f_mo[(i, a_mo)];
                let t_ia = f_ia * dsrg_regularized_inv_denom(delta_ia, s);
                let f_tilde = f_ia * (1.0 + dsrg_residual_weight(delta_ia, s));
                e_pt2 += 2.0 * f_tilde * t_ia;
                max_t1 = max_t1.max(t_ia.abs());
            }
        }

        // 2-body contribution:
        // E_2^(2)(s) = sum_{ijab} (ia|jb) [2(ia|jb) - (ib|ja)] * [1 - exp(-2 s (Delta_{ab}^{ij})^2)] / Delta_{ab}^{ij}
        let mut rdm1 = Matrix::zeros(n_basis, n_basis);
        for i in 0..n_occ {
            rdm1[(i, i)] = 2.0;
        }

        let mut t2 = vec![0.0; n_occ * n_occ * n_virt * n_virt];
        let idx =
            |i: usize, j: usize, a: usize, b: usize| ((i * n_occ + j) * n_virt + a) * n_virt + b;

        for i in 0..n_occ {
            for j in 0..n_occ {
                for a in 0..n_virt {
                    let a_mo = n_occ + a;
                    for b in 0..n_virt {
                        let b_mo = n_occ + b;
                        let delta = eps[i] + eps[j] - eps[a_mo] - eps[b_mo];
                        let iajb = mo_eri.get(i, a_mo, j, b_mo);
                        let ibja = mo_eri.get(i, b_mo, j, a_mo);
                        let t_ijab = iajb * dsrg_regularized_inv_denom(delta, s);
                        t2[idx(i, j, a, b)] = t_ijab;
                        max_t2 = max_t2.max(t_ijab.abs());

                        let v_tilde = (2.0 * iajb - ibja) * (1.0 + dsrg_residual_weight(delta, s));
                        e_pt2 += v_tilde * t_ijab;
                    }
                }
            }
        }

        // Unrelaxed 1-RDM response
        for i in 0..n_occ {
            for j in 0..n_occ {
                let mut sum = 0.0;
                for k in 0..n_occ {
                    for a in 0..n_virt {
                        for b in 0..n_virt {
                            let t_ikab = t2[idx(i, k, a, b)];
                            let t_jkab = t2[idx(j, k, a, b)];
                            let t_jkba = t2[idx(j, k, b, a)];
                            sum -= 2.0 * t_ikab * (2.0 * t_jkab - t_jkba);
                        }
                    }
                }
                rdm1[(i, j)] += sum;
            }
        }
        for a in 0..n_virt {
            let a_mo = n_occ + a;
            for b in 0..n_virt {
                let b_mo = n_occ + b;
                let mut sum = 0.0;
                for i in 0..n_occ {
                    for j in 0..n_occ {
                        for c in 0..n_virt {
                            let t_ijac = t2[idx(i, j, a, c)];
                            let t_ijbc = t2[idx(i, j, b, c)];
                            let t_ijcb = t2[idx(i, j, c, b)];
                            sum += 2.0 * t_ijac * (2.0 * t_ijbc - t_ijcb);
                        }
                    }
                }
                rdm1[(a_mo, b_mo)] += sum;
            }
        }

        Ok(SRDSRGResult {
            method_label: "SR-DSRG-PT2",
            reference_energy: self.hf.total_energy,
            correlation_energy: e_pt2,
            pt2_energy: e_pt2,
            pt3_correction: 0.0,
            flow_param_s: s,
            iterations: 1,
            converged: true,
            max_t1,
            max_t2,
            one_rdm_mo: Some(rdm1),
            ldsrg: None,
            perturbative: None,
        })
    }
}

// ============================================================================
// 2. UHF-SR-DSRG-PT2 (Open-Shell Unrestricted Second-Order DSRG-PT2)
// ============================================================================

/// Unrestricted Open-Shell Second-Order Driven Similarity Renormalization Group (`UHF-SR-DSRG-PT2`).
pub struct UHFSRDSRGPT2<'a> {
    uhf: &'a UHFResult,
    flow_param_s: f64,
    integral_cache: Option<&'a crate::context::MoIntegralCache<'a>>,
}

impl<'a> UHFSRDSRGPT2<'a> {
    pub fn new(uhf: &'a UHFResult, flow_param_s: f64) -> Self {
        Self {
            uhf,
            flow_param_s,
            integral_cache: None,
        }
    }
}

impl<'a> PostHFMethod for UHFSRDSRGPT2<'a> {
    type Output = SRDSRGResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let s = self.flow_param_s;
        let n_basis = self.uhf.mo_coefficients_alpha.cols;
        if !s.is_finite()
            || s < 0.
            || !self.uhf.converged
            || self.uhf.mo_coefficients_beta.cols != n_basis
            || self.uhf.orbital_energies_alpha.len() != n_basis
            || self.uhf.orbital_energies_beta.len() != n_basis
            || self.uhf.n_alpha > n_basis
            || self.uhf.n_beta > n_basis
        {
            return Err(
                "UHF-SR-DSRG-PT2 requires finite nonnegative flow and compatible converged UHF"
                    .into(),
            );
        }
        let na = self.uhf.n_alpha;
        let nb = self.uhf.n_beta;
        let nva = n_basis - na;
        let nvb = n_basis - nb;
        let eps_a = &self.uhf.orbital_energies_alpha;
        let eps_b = &self.uhf.orbital_energies_beta;

        let eri_aa = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.full().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(transform_eri_ao_to_mo(
                &self.uhf.mo_coefficients_alpha,
                &self.uhf.eri,
            ))
        };
        let eri_bb = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.full_beta().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(transform_eri_ao_to_mo(
                &self.uhf.mo_coefficients_beta,
                &self.uhf.eri,
            ))
        };
        let eri_ab = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.cross().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(
                transform_mixed_eri(
                    &self.uhf.mo_coefficients_alpha,
                    &self.uhf.mo_coefficients_beta,
                    &self.uhf.eri,
                )
                .map_err(|e| e.to_string())?,
            )
        };

        let mut e_aa = 0.0;
        let mut e_bb = 0.0;
        let mut e_ab = 0.0;
        let mut max_t2 = 0.0_f64;

        // Alpha-Alpha
        for i in 0..na {
            for j in 0..i {
                for a in 0..nva {
                    let a_mo = na + a;
                    for b in 0..a {
                        let b_mo = na + b;
                        let delta = eps_a[i] + eps_a[j] - eps_a[a_mo] - eps_a[b_mo];
                        let v_anti = eri_aa.get(i, a_mo, j, b_mo) - eri_aa.get(i, b_mo, j, a_mo);
                        let t_ijab = v_anti * dsrg_regularized_inv_denom(delta, s);
                        let v_tilde = v_anti * (1.0 + dsrg_residual_weight(delta, s));
                        e_aa += v_tilde * t_ijab;
                        max_t2 = max_t2.max(t_ijab.abs());
                    }
                }
            }
        }

        // Beta-Beta
        for i in 0..nb {
            for j in 0..i {
                for a in 0..nvb {
                    let a_mo = nb + a;
                    for b in 0..a {
                        let b_mo = nb + b;
                        let delta = eps_b[i] + eps_b[j] - eps_b[a_mo] - eps_b[b_mo];
                        let v_anti = eri_bb.get(i, a_mo, j, b_mo) - eri_bb.get(i, b_mo, j, a_mo);
                        let t_ijab = v_anti * dsrg_regularized_inv_denom(delta, s);
                        let v_tilde = v_anti * (1.0 + dsrg_residual_weight(delta, s));
                        e_bb += v_tilde * t_ijab;
                        max_t2 = max_t2.max(t_ijab.abs());
                    }
                }
            }
        }

        // Alpha-Beta
        for i in 0..na {
            for j in 0..nb {
                for a in 0..nva {
                    let a_mo = na + a;
                    for b in 0..nvb {
                        let b_mo = nb + b;
                        let delta = eps_a[i] + eps_b[j] - eps_a[a_mo] - eps_b[b_mo];
                        let v_ab = eri_ab.get(i, a_mo, j, b_mo);
                        let t_ijab = v_ab * dsrg_regularized_inv_denom(delta, s);
                        let v_tilde = v_ab * (1.0 + dsrg_residual_weight(delta, s));
                        e_ab += v_tilde * t_ijab;
                        max_t2 = max_t2.max(t_ijab.abs());
                    }
                }
            }
        }

        let e_corr = e_aa + e_bb + e_ab;
        Ok(SRDSRGResult {
            method_label: "UHF-SR-DSRG-PT2",
            reference_energy: self.uhf.total_energy,
            correlation_energy: e_corr,
            pt2_energy: e_corr,
            pt3_correction: 0.0,
            flow_param_s: s,
            iterations: 1,
            converged: true,
            max_t1: 0.0,
            max_t2,
            one_rdm_mo: None,
            ldsrg: None,
            perturbative: None,
        })
    }
}

/// Third-order coefficients of the canonical single-reference driven BCH equations.
pub struct SRDSRGPT3<'a> {
    hf: &'a HFResult,
    flow_param_s: f64,
    integral_cache: Option<&'a crate::context::MoIntegralCache<'a>>,
}
impl<'a> SRDSRGPT3<'a> {
    pub fn new(hf: &'a HFResult, flow_param_s: f64) -> Self {
        Self {
            hf,
            flow_param_s,
            integral_cache: None,
        }
    }
}
impl PostHFMethod for SRDSRGPT3<'_> {
    type Output = SRDSRGResult;
    fn compute(&self) -> Result<Self::Output, String> {
        let (h, occupied) = super::imsrg2::rhf_normal_ordered(self.hf, self.integral_cache)
            .map_err(|e| e.to_string())?;
        let solution = super::sr_dsrg::dsrg_pt3(&h, &occupied, self.flow_param_s)
            .map_err(|e| e.to_string())?;
        let (max_t1, max_t2) = amplitude_maxima(&solution.first_order_generator);
        Ok(SRDSRGResult {
            method_label: "SR-DSRG-PT3",
            reference_energy: self.hf.total_energy,
            correlation_energy: solution.second_order + solution.third_order,
            pt2_energy: solution.second_order,
            pt3_correction: solution.third_order,
            flow_param_s: self.flow_param_s,
            iterations: 1,
            converged: true,
            max_t1,
            max_t2,
            one_rdm_mo: None,
            ldsrg: None,
            perturbative: Some(solution),
        })
    }
}

/// Linearized DSRG(2): recursively truncate every BCH commutator to NO2B.
pub struct SRLDSRG2<'a> {
    hf: &'a HFResult,
    flow_param_s: f64,
    options: super::sr_dsrg::LdsrgOptions,
    integral_cache: Option<&'a crate::context::MoIntegralCache<'a>>,
}
impl<'a> SRLDSRG2<'a> {
    pub fn new(hf: &'a HFResult, flow_param_s: f64) -> Self {
        Self {
            hf,
            flow_param_s,
            options: Default::default(),
            integral_cache: None,
        }
    }
    pub fn with_options(mut self, options: super::sr_dsrg::LdsrgOptions) -> Self {
        self.options = options;
        self
    }
}
fn amplitude_maxima(a: &super::imsrg2::No2bOperator) -> (f64, f64) {
    (
        a.one.data.iter().fold(0f64, |v, t| v.max(t.abs())),
        a.two.iter().fold(0f64, |v, t| v.max(t.abs())),
    )
}
impl PostHFMethod for SRLDSRG2<'_> {
    type Output = SRDSRGResult;
    fn compute(&self) -> Result<Self::Output, String> {
        let (h, occupied) = super::imsrg2::rhf_normal_ordered(self.hf, self.integral_cache)
            .map_err(|e| e.to_string())?;
        let solution = super::sr_dsrg::solve_ldsrg2(&h, &occupied, self.flow_param_s, self.options)
            .map_err(|e| e.to_string())?;
        let pt2 = super::sr_dsrg::dsrg_pt2_energy(&h, &occupied, self.flow_param_s)
            .map_err(|e| e.to_string())?;
        let (max_t1, max_t2) = amplitude_maxima(&solution.generator);
        Ok(SRDSRGResult {
            method_label: "SR-LDSRG(2)",
            reference_energy: self.hf.total_energy,
            correlation_energy: solution.transformed.zero,
            pt2_energy: pt2,
            pt3_correction: solution.transformed.zero - pt2,
            flow_param_s: self.flow_param_s,
            iterations: solution.iterations,
            converged: true,
            max_t1,
            max_t2,
            one_rdm_mo: None,
            ldsrg: Some(solution),
            perturbative: None,
        })
    }
}

fn validate_sr_context(
    options: &crate::options::PostHFOptions,
) -> Result<(), crate::error::HfError> {
    options.validate()?;
    if options.active.is_some()
        || options.frozen_core != 0
        || options.frozen_virtual != 0
        || options.density_fitting
        || options.root != 0
        || options.n_roots != 1
        || options.dsrg.relaxation() != crate::options::ReferenceRelaxation::None
    {
        return Err(crate::error::HfError::Unsupported {
            method: "SR-DSRG".into(),
            detail: "requires one full-space determinant without DF/reference relaxation".into(),
        });
    }
    Ok(())
}
/// Typed RHF/UHF PT2 dispatch, borrowing the common integral cache.
pub struct TypedDsrgPt2<'a> {
    context: &'a crate::context::PostHFContext<'a>,
    flow: f64,
}
impl PostHFMethod for TypedDsrgPt2<'_> {
    type Output = SRDSRGResult;
    fn compute(&self) -> Result<Self::Output, String> {
        match self.context.reference {
            crate::context::Reference::Rhf(hf) => {
                let mut solver = SRDSRGPT2::new(hf, self.flow);
                solver.integral_cache = Some(&self.context.ints);
                solver.compute()
            }
            crate::context::Reference::Uhf(hf) => {
                let mut solver = UHFSRDSRGPT2::new(hf, self.flow);
                solver.integral_cache = Some(&self.context.ints);
                solver.compute()
            }
        }
    }
}
impl crate::registry::MethodMeta for TypedDsrgPt2<'_> {
    const METHOD: crate::registry::MethodId = crate::registry::MethodId::SrDsrgPt2;
}
impl<'a> crate::registry::ContextMethod<'a> for TypedDsrgPt2<'a> {
    fn from_context(
        ctx: &'a crate::context::PostHFContext<'a>,
        options: &crate::options::PostHFOptions,
    ) -> Result<Self, crate::error::HfError> {
        validate_sr_context(options)?;
        Ok(Self {
            context: ctx,
            flow: options.dsrg.flow_parameter,
        })
    }
}
impl crate::registry::MethodMeta for SRDSRGPT3<'_> {
    const METHOD: crate::registry::MethodId = crate::registry::MethodId::SrDsrgPt3;
}
impl<'a> crate::registry::ContextMethod<'a> for SRDSRGPT3<'a> {
    fn from_context(
        ctx: &'a crate::context::PostHFContext<'a>,
        options: &crate::options::PostHFOptions,
    ) -> Result<Self, crate::error::HfError> {
        validate_sr_context(options)?;
        let mut engine = Self::new(ctx.rhf()?, options.dsrg.flow_parameter);
        engine.integral_cache = Some(&ctx.ints);
        Ok(engine)
    }
}
impl crate::registry::MethodMeta for SRLDSRG2<'_> {
    const METHOD: crate::registry::MethodId = crate::registry::MethodId::SrLdsrg2;
}
impl<'a> crate::registry::ContextMethod<'a> for SRLDSRG2<'a> {
    fn from_context(
        ctx: &'a crate::context::PostHFContext<'a>,
        options: &crate::options::PostHFOptions,
    ) -> Result<Self, crate::error::HfError> {
        validate_sr_context(options)?;
        let mut engine = Self::new(ctx.rhf()?, options.dsrg.flow_parameter).with_options(
            super::sr_dsrg::LdsrgOptions {
                max_iter: options.max_iter,
                residual_tolerance: options.residual_tol,
                energy_tolerance: options.energy_tol,
                bch_tolerance: options.dsrg.bch_tolerance,
                bch_max_order: options.dsrg.bch_max_order,
            },
        );
        engine.integral_cache = Some(&ctx.ints);
        Ok(engine)
    }
}

// ============================================================================
// 5. Continuous-flow IMSRG(2), Wegner / White
// ============================================================================

pub use super::imsrg2::SRGGenerator;

/// Result of a Continuous Flow Similarity Renormalization Group (`SRG-ODE`) integration.
#[derive(Clone, Debug)]
pub struct SRGODEResult {
    pub generator: SRGGenerator,
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub total_energy: f64,
    pub flow_param_s: f64,
    pub n_rk4_steps: usize,
    pub residual_off_diagonal_norm: f64,
    pub flowed: super::imsrg2::No2bOperator,
}

impl PostHFResult for SRGODEResult {
    fn method_name(&self) -> &'static str {
        match self.generator {
            SRGGenerator::Wegner => "SRG-ODE(Wegner)",
            SRGGenerator::White => "SRG-ODE(White)",
        }
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }
}

/// Continuous Flow Similarity Renormalization Group (`SRG-ODE`) solver integrating
/// $dH/ds = [\eta,H]_{0,1,2}$ via fixed or adaptive fourth-order Runge-Kutta.
pub struct SRGODE<'a> {
    hf: &'a HFResult,
    generator: SRGGenerator,
    flow_param_s: f64,
    n_steps: usize,
    ode_tolerance: Option<f64>,
    integral_cache: Option<&'a crate::context::MoIntegralCache<'a>>,
}

impl<'a> SRGODE<'a> {
    pub fn new(
        hf: &'a HFResult,
        generator: SRGGenerator,
        flow_param_s: f64,
        n_steps: usize,
    ) -> Self {
        Self {
            hf,
            generator,
            flow_param_s,
            n_steps,
            ode_tolerance: None,
            integral_cache: None,
        }
    }
}

impl<'a> PostHFMethod for SRGODE<'a> {
    type Output = SRGODEResult;
    fn compute(&self) -> Result<Self::Output, String> {
        use super::imsrg2::integrate_imsrg2;
        let hf = self.hf;
        if !self.flow_param_s.is_finite()
            || self.flow_param_s < 0.
            || self.n_steps == 0
            || self.n_steps > 100_000
            || self
                .ode_tolerance
                .is_some_and(|v| !v.is_finite() || v <= 0.)
        {
            return Err("invalid IMSRG flow/step/tolerance controls".into());
        }
        let (initial, occupied) = super::imsrg2::rhf_normal_ordered(hf, self.integral_cache)
            .map_err(|e| e.to_string())?;
        let (flowed, steps) = integrate_imsrg2(
            &initial,
            &occupied,
            self.generator,
            self.flow_param_s,
            self.n_steps,
            self.ode_tolerance,
        )
        .map_err(|e| e.to_string())?;
        Ok(SRGODEResult {
            generator: self.generator,
            reference_energy: hf.total_energy,
            correlation_energy: flowed.zero,
            total_energy: hf.total_energy + flowed.zero,
            flow_param_s: self.flow_param_s,
            n_rk4_steps: steps,
            residual_off_diagonal_norm: flowed.off_diagonal_norm(&occupied),
            flowed,
        })
    }
}

impl SRGODE<'_> {
    pub fn with_ode_tolerance(mut self, tolerance: f64) -> Self {
        self.ode_tolerance = Some(tolerance);
        self
    }
}
impl crate::registry::MethodMeta for SRGODE<'_> {
    const METHOD: crate::registry::MethodId = crate::registry::MethodId::SrGode;
}
impl<'a> crate::registry::ContextMethod<'a> for SRGODE<'a> {
    fn from_context(
        ctx: &'a crate::context::PostHFContext<'a>,
        options: &crate::options::PostHFOptions,
    ) -> Result<Self, crate::error::HfError> {
        use crate::error::HfError;
        options.validate()?;
        if options.active.is_some()
            || options.frozen_core != 0
            || options.frozen_virtual != 0
            || options.density_fitting
            || options.root != 0
            || options.n_roots != 1
            || options.dsrg.relaxation() != crate::options::ReferenceRelaxation::None
        {
            return Err(HfError::Unsupported {method:"IM-SRG(2)".into(),detail:"adapter requires one full-space RHF determinant without DF/reference relaxation".into()});
        }
        let steps = (options.dsrg.flow_parameter / options.dsrg.ode_step)
            .ceil()
            .max(1.);
        if steps > 100_000. {
            return Err(HfError::InvalidInput(
                "IMSRG initial step count exceeds 100000".into(),
            ));
        }
        let mut engine = Self::new(
            ctx.rhf()?,
            options.dsrg.ode_generator,
            options.dsrg.flow_parameter,
            steps as usize,
        )
        .with_ode_tolerance(options.dsrg.ode_tolerance);
        engine.integral_cache = Some(&ctx.ints);
        Ok(engine)
    }
}
