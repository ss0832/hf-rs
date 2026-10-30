// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Single-reference DSRG from a recursively truncated NO2B BCH transformation.
//! See docs/sr_dsrg.md for the source condition and perturbative expansion.

use super::dsrg::{dsrg_regularized_inv_denom, dsrg_residual_weight};
use super::imsrg2::{No2bOperator, no2b_commutator};
use crate::{Matrix, diis::CdiisEngine, error::HfError};

/// Central energy differences with reconverged SCF and driven amplitudes.
/// All SR method/flow/convergence options pass through the typed registry.
pub fn numerical_gradient(
    method: crate::registry::MethodId,
    atoms: &[crate::basis::Atom],
    basis: &[crate::basis::ContractedGaussian],
    charge: i32,
    basis_kind: crate::ao::BasisKind,
    options: &crate::options::PostHFOptions,
    numerical: &crate::post_hf::numerical::NumericalGradientOptions,
) -> Result<crate::post_hf::numerical::NumericalGradientResult, HfError> {
    use crate::registry::{MethodId, create_solver};
    if !matches!(
        method,
        MethodId::SrDsrgPt2 | MethodId::SrDsrgPt3 | MethodId::SrLdsrg2
    ) {
        return Err(HfError::Unsupported {
            method: method.name().into(),
            detail: "SR-DSRG numerical-gradient adapter requires an SR driven method".into(),
        });
    }
    options.validate()?;
    crate::post_hf::numerical::compute_rhf_numerical_gradient(
        atoms,
        basis,
        charge,
        basis_kind,
        numerical,
        |hf, point, moved| {
            let context = crate::context::PostHFContext::from_rhf(hf, point, moved, None)?;
            Ok(create_solver(method, &context, options)?.compute()?.e_total)
        },
    )
}

#[derive(Clone, Copy, Debug)]
pub struct LdsrgOptions {
    pub max_iter: usize,
    pub residual_tolerance: f64,
    pub energy_tolerance: f64,
    pub bch_tolerance: f64,
    pub bch_max_order: usize,
}
impl Default for LdsrgOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            residual_tolerance: 1e-10,
            energy_tolerance: 1e-12,
            bch_tolerance: 1e-13,
            bch_max_order: 80,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LdsrgSolution {
    /// A = T - T^dagger, including all single and double excitations.
    pub generator: No2bOperator,
    pub transformed: No2bOperator,
    pub iterations: usize,
    pub residual_norm: f64,
    pub amplitude_residual_norm: f64,
    pub bch_order: usize,
}

#[derive(Clone, Debug)]
pub struct DsrgPt3Solution {
    pub second_order: f64,
    pub third_order: f64,
    pub first_order_generator: No2bOperator,
    pub second_order_generator: No2bOperator,
}

fn failure(detail: impl Into<String>) -> HfError {
    HfError::PostHf {
        method: "SR-DSRG".into(),
        detail: detail.into(),
    }
}
pub(crate) fn zero(n: usize) -> No2bOperator {
    No2bOperator {
        zero: 0.,
        one: Matrix::zeros(n, n),
        two: vec![0.; n.pow(4)],
    }
}

/// exp(ad_A) H with projection to ranks 0,1,2 after EACH commutator.
/// A dense exponential of the physical Fock-space Hamiltonian is different.
pub fn bch_no2b(
    h: &No2bOperator,
    a: &No2bOperator,
    occupied: &[bool],
    tolerance: f64,
    max_order: usize,
) -> Result<(No2bOperator, usize), HfError> {
    h.validate(occupied)?;
    a.validate(occupied)?;
    if !tolerance.is_finite() || tolerance <= 0. || max_order == 0 || max_order > 1000 {
        return Err(HfError::InvalidInput("invalid BCH tolerance/order".into()));
    }
    let mut transformed = h.clone();
    let mut term = h.clone();
    for k in 1..=max_order {
        term = no2b_commutator(&term, a, occupied)?;
        term.zero /= k as f64;
        term.one.data.iter_mut().for_each(|v| *v /= k as f64);
        term.two.iter_mut().for_each(|v| *v /= k as f64);
        term.validate(occupied)?;
        transformed = transformed.add_scaled(&term, 1.);
        transformed.validate(occupied)?;
        if term.norm_max() <= tolerance {
            return Ok((transformed, k));
        }
    }
    Err(failure(format!(
        "BCH expansion did not converge in {max_order} commutators"
    )))
}

#[derive(Clone, Copy)]
pub(crate) enum Excitation {
    Single(usize, usize),
    Double(usize, usize, usize, usize),
}
impl Excitation {
    pub(crate) fn gap(self, h: &No2bOperator) -> f64 {
        let f = &h.one;
        match self {
            Self::Single(a, i) => f[(i, i)] - f[(a, a)],
            Self::Double(a, b, i, j) => f[(i, i)] + f[(j, j)] - f[(a, a)] - f[(b, b)],
        }
    }
    pub(crate) fn value(self, h: &No2bOperator) -> f64 {
        let n = h.one.rows;
        match self {
            Self::Single(a, i) => h.one[(a, i)],
            Self::Double(a, b, i, j) => h.two[((a * n + b) * n + i) * n + j],
        }
    }
    pub(crate) fn insert(self, h: &mut No2bOperator, t: f64) {
        let n = h.one.rows;
        match self {
            Self::Single(a, i) => {
                h.one[(a, i)] = t;
                h.one[(i, a)] = -t;
            }
            Self::Double(a, b, i, j) => {
                for (p, q, r, s, sign) in [
                    (a, b, i, j, 1.),
                    (b, a, i, j, -1.),
                    (a, b, j, i, -1.),
                    (b, a, j, i, 1.),
                    (i, j, a, b, -1.),
                    (j, i, a, b, 1.),
                    (i, j, b, a, 1.),
                    (j, i, b, a, -1.),
                ] {
                    h.two[((p * n + q) * n + r) * n + s] = sign * t;
                }
            }
        }
    }
}
fn excitations(occupied: &[bool]) -> Vec<Excitation> {
    let holes: Vec<_> = (0..occupied.len()).filter(|&p| occupied[p]).collect();
    let particles: Vec<_> = (0..occupied.len()).filter(|&p| !occupied[p]).collect();
    let mut result = Vec::new();
    for &a in &particles {
        for &i in &holes {
            result.push(Excitation::Single(a, i));
        }
    }
    for (aa, &a) in particles.iter().enumerate() {
        for &b in &particles[aa + 1..] {
            for (ii, &i) in holes.iter().enumerate() {
                for &j in &holes[ii + 1..] {
                    result.push(Excitation::Double(a, b, i, j));
                }
            }
        }
    }
    result
}
pub(crate) fn generator(n: usize, ex: &[Excitation], amplitudes: &[f64]) -> No2bOperator {
    let mut a = zero(n);
    for (&e, &t) in ex.iter().zip(amplitudes) {
        e.insert(&mut a, t);
    }
    a
}
fn validate(h: &No2bOperator, occupied: &[bool], flow: f64) -> Result<(), HfError> {
    h.validate(occupied)?;
    if !flow.is_finite() || flow < 0. {
        return Err(HfError::InvalidInput(
            "DSRG flow must be finite and nonnegative".into(),
        ));
    }
    let n = occupied.len();
    for p in 0..n {
        for q in 0..n {
            if (h.one[(p, q)] - h.one[(q, p)]).abs() > 1e-10 {
                return Err(HfError::InvalidInput(
                    "DSRG Hamiltonian must be Hermitian".into(),
                ));
            }
            for r in 0..n {
                for s in 0..n {
                    let v = h.two[((p * n + q) * n + r) * n + s];
                    if (v - h.two[((r * n + s) * n + p) * n + q]).abs() > 1e-10
                        || (v + h.two[((q * n + p) * n + r) * n + s]).abs() > 1e-10
                        || (v + h.two[((p * n + q) * n + s) * n + r]).abs() > 1e-10
                    {
                        return Err(HfError::InvalidInput(
                            "DSRG two-body tensor must be Hermitian and antisymmetric".into(),
                        ));
                    }
                }
            }
        }
    }
    if excitations(occupied).iter().any(|e| !e.gap(h).is_finite()) {
        return Err(HfError::InvalidInput(
            "nonfinite DSRG excitation gap".into(),
        ));
    }
    Ok(())
}

/// Solve Hbar_ex = (Hbar_ex + Delta*t_ex) exp(-s*Delta^2).
/// Anderson/Pulay acceleration acts on the regularized fixed-point residual.
pub fn solve_ldsrg2(
    h: &No2bOperator,
    occupied: &[bool],
    flow: f64,
    options: LdsrgOptions,
) -> Result<LdsrgSolution, HfError> {
    validate(h, occupied, flow)?;
    if options.max_iter == 0
        || options.max_iter > 100_000
        || [
            options.residual_tolerance,
            options.energy_tolerance,
            options.bch_tolerance,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err(HfError::InvalidInput(
            "invalid SR-LDSRG(2) convergence controls".into(),
        ));
    }
    let ex = excitations(occupied);
    let gaps: Vec<_> = ex.iter().map(|e| e.gap(h)).collect();
    let mut t: Vec<_> = ex
        .iter()
        .zip(&gaps)
        .map(|(e, &d)| e.value(h) * dsrg_regularized_inv_denom(d, flow))
        .collect();
    let mut history = CdiisEngine::new(8);
    let mut previous_energy = None::<f64>;
    for iteration in 1..=options.max_iter {
        let a = generator(occupied.len(), &ex, &t);
        let (bar, order) = bch_no2b(
            h,
            &a,
            occupied,
            options.bch_tolerance,
            options.bch_max_order,
        )?;
        let mut amplitude_norm = 0f64;
        let mut physical_norm = 0f64;
        let mut error = Vec::with_capacity(ex.len());
        let mut target = Vec::with_capacity(ex.len());
        for ((&e, &d), &amp) in ex.iter().zip(&gaps).zip(&t) {
            let v = e.value(&bar);
            let next = (v + d * amp) * dsrg_regularized_inv_denom(d, flow);
            let residual = next - amp;
            // exp_m1 avoids cancellation in the source condition near d=0.
            let physical = -(-flow * d * d).exp_m1() * v - d * amp * dsrg_residual_weight(d, flow);
            if !residual.is_finite() || !physical.is_finite() {
                return Err(failure("nonfinite DSRG source residual"));
            }
            amplitude_norm = amplitude_norm.max(residual.abs());
            physical_norm = physical_norm.max(physical.abs());
            error.push(residual);
            target.push(next);
        }
        let energy_ok =
            previous_energy.is_none_or(|old| (bar.zero - old).abs() <= options.energy_tolerance);
        if amplitude_norm <= options.residual_tolerance
            && physical_norm <= options.residual_tolerance
            && energy_ok
        {
            return Ok(LdsrgSolution {
                generator: a,
                transformed: bar,
                iterations: iteration,
                residual_norm: physical_norm,
                amplitude_residual_norm: amplitude_norm,
                bch_order: order,
            });
        }
        previous_energy = Some(bar.zero);
        history.push(
            Matrix::from_vec(1, t.len(), target.clone()),
            Matrix::from_vec(1, t.len(), error),
        );
        t = history.extrapolate().map(|v| v.data).unwrap_or(target);
        if t.iter().any(|v| !v.is_finite()) {
            return Err(failure("nonfinite LDSRG amplitudes"));
        }
    }
    Err(failure(format!(
        "SR-LDSRG(2) amplitudes did not converge in {} iterations",
        options.max_iter
    )))
}

/// Second-order energy without allocating higher-order commutator intermediates.
pub fn dsrg_pt2_energy(h: &No2bOperator, occupied: &[bool], flow: f64) -> Result<f64, HfError> {
    validate(h, occupied, flow)?;
    let energy = excitations(occupied)
        .iter()
        .map(|e| {
            let d = e.gap(h);
            let v = e.value(h);
            v * v * dsrg_regularized_inv_denom(d, flow) * (1. + dsrg_residual_weight(d, flow))
        })
        .sum::<f64>();
    if !energy.is_finite() {
        return Err(failure("nonfinite second-order energy"));
    }
    Ok(energy)
}

/// Perturbative expansion through third order of the driven BCH equations.
pub fn dsrg_pt3(
    h: &No2bOperator,
    occupied: &[bool],
    flow: f64,
) -> Result<DsrgPt3Solution, HfError> {
    validate(h, occupied, flow)?;
    let n = occupied.len();
    let ex = excitations(occupied);
    let mut h0 = zero(n);
    for p in 0..n {
        h0.one[(p, p)] = h.one[(p, p)];
    }
    let h1 = h.add_scaled(&h0, -1.);
    let t1: Vec<_> = ex
        .iter()
        .map(|e| e.value(&h1) * dsrg_regularized_inv_denom(e.gap(h), flow))
        .collect();
    let a1 = generator(n, &ex, &t1);
    let c01 = no2b_commutator(&h0, &a1, occupied)?;
    let c11 = no2b_commutator(&h1, &a1, occupied)?;
    let c011 = no2b_commutator(&c01, &a1, occupied)?;
    let k2 = c11.add_scaled(&c011, 0.5);
    let t2: Vec<_> = ex
        .iter()
        .map(|e| e.value(&k2) * dsrg_regularized_inv_denom(e.gap(h), flow))
        .collect();
    let a2 = generator(n, &ex, &t2);
    let c02 = no2b_commutator(&h0, &a2, occupied)?;
    let e3 = no2b_commutator(&h1, &a2, occupied)?.zero
        + 0.5
            * (no2b_commutator(&c01, &a2, occupied)?.zero
                + no2b_commutator(&c02, &a1, occupied)?.zero
                + no2b_commutator(&c11, &a1, occupied)?.zero)
        + no2b_commutator(&c011, &a1, occupied)?.zero / 6.;
    if !k2.zero.is_finite() || !e3.is_finite() {
        return Err(failure("nonfinite perturbative energy"));
    }
    Ok(DsrgPt3Solution {
        second_order: k2.zero,
        third_order: e3,
        first_order_generator: a1,
        second_order_generator: a2,
    })
}
