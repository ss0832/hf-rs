// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Unrelaxed MR driven equations with exact correlated-reference contractions.
//! The Fdiag partition requires a semicanonical generalized Fock matrix.

use super::{
    dsrg::{dsrg_regularized_inv_denom, dsrg_residual_weight},
    gno::{GnoAlgebra, GnoReference},
    imsrg2::No2bOperator,
    sr_dsrg::{DsrgPt3Solution, Excitation, LdsrgOptions, LdsrgSolution, generator, zero},
};
use crate::{Matrix, diis::CdiisEngine, error::HfError};

fn failure(detail: impl Into<String>) -> HfError {
    HfError::PostHf {
        method: "MR-DSRG".into(),
        detail: detail.into(),
    }
}

/// A fixed core/active/external partition and exact ensemble contractions.
/// Purely active amplitudes are absent. Semi-internal singles and doubles remain.
pub struct MrDsrgEquations {
    algebra: GnoAlgebra,
    classes: Vec<u8>,
    active: Vec<usize>,
    excitations: Vec<Excitation>,
}
impl MrDsrgEquations {
    pub fn normal_order(&self, bare: &No2bOperator) -> Result<No2bOperator, HfError> {
        self.algebra.normal_order(bare)
    }
    /// Bare active-space operator after exact de-normal ordering/core contraction.
    pub fn active_effective_hamiltonian(
        &self,
        bar: &No2bOperator,
    ) -> Result<No2bOperator, HfError> {
        let core: Vec<_> = self
            .classes
            .iter()
            .enumerate()
            .filter_map(|(p, &group)| (group == 0).then_some(p))
            .collect();
        crate::fci::general_operator::contract_core(
            &self.algebra.denormal_order(bar)?,
            &core,
            &self.active,
        )
    }
    /// Perturbative CAS effective operator through order two or three.
    /// Unknown higher-order [H0,A_n] terms have zero CAS projection.
    pub fn perturbative_active_hamiltonian(
        &self,
        h: &No2bOperator,
        flow: f64,
        order: usize,
        options: LdsrgOptions,
    ) -> Result<No2bOperator, HfError> {
        self.validate(h, flow, options)?;
        let bar = match order {
            2 => {
                let mut h0 = zero(self.classes.len());
                for p in 0..self.classes.len() {
                    h0.one[(p, p)] = h.one[(p, p)];
                }
                let mut h1 = h.add_scaled(&h0, -1.);
                h1.zero = 0.;
                let a1 = self.linear_generator(h, &h0, &h1, flow)?;
                let c01 = self.algebra.commutator(&h0, &a1)?;
                let k2 = self.algebra.commutator(&h1.add_scaled(&c01, 0.5), &a1)?;
                h.add_scaled(&c01, 1.).add_scaled(&k2, 1.)
            }
            3 => self.third_order_hamiltonian(h, flow, options)?.1,
            _ => {
                return Err(HfError::InvalidInput(
                    "MR CAS effective Hamiltonian order must be two or three".into(),
                ));
            }
        };
        self.active_effective_hamiltonian(&bar)
    }
    pub fn new(reference: &GnoReference<'_>, memory_budget: usize) -> Result<Self, HfError> {
        let algebra = GnoAlgebra::new(reference, memory_budget)?;
        let classes = reference.orbital_classes();
        let holes: Vec<_> = (0..classes.len()).filter(|&p| classes[p] != 2).collect();
        let particles: Vec<_> = (0..classes.len()).filter(|&p| classes[p] != 0).collect();
        let mut excitations = Vec::new();
        for &a in &particles {
            for &i in &holes {
                if classes[a] != 1 || classes[i] != 1 {
                    excitations.push(Excitation::Single(a, i));
                }
            }
        }
        for (aa, &a) in particles.iter().enumerate() {
            for &b in &particles[aa + 1..] {
                for (ii, &i) in holes.iter().enumerate() {
                    for &j in &holes[ii + 1..] {
                        if [a, b, i, j].iter().any(|&p| classes[p] != 1) {
                            excitations.push(Excitation::Double(a, b, i, j));
                        }
                    }
                }
            }
        }
        Ok(Self {
            algebra,
            classes,
            active: reference.active_spin_orbitals(),
            excitations,
        })
    }
    fn validate(&self, h: &No2bOperator, flow: f64, options: LdsrgOptions) -> Result<(), HfError> {
        self.algebra.validate(h)?;
        if !flow.is_finite()
            || flow < 0.
            || options.max_iter == 0
            || options.max_iter > 100_000
            || options.bch_max_order == 0
            || options.bch_max_order > 1000
            || [
                options.residual_tolerance,
                options.energy_tolerance,
                options.bch_tolerance,
            ]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.)
        {
            return Err(HfError::InvalidInput(
                "invalid MR-DSRG flow/convergence controls".into(),
            ));
        }
        let n = self.classes.len();
        for p in 0..n {
            for q in 0..n {
                if (h.one[(p, q)] - h.one[(q, p)]).abs() > 1e-10 {
                    return Err(HfError::InvalidInput(
                        "MR-DSRG requires a Hermitian Hamiltonian".into(),
                    ));
                }
                if p != q && self.classes[p] == self.classes[q] && h.one[(p, q)].abs() > 1e-10 {
                    return Err(HfError::Unsupported {
                        method: "MR-DSRG/Fdiag".into(),
                        detail:
                            "generalized Fock core/active/external blocks must be semicanonical"
                                .into(),
                    });
                }
                for r in 0..n {
                    for s in 0..n {
                        if (h.two[((p * n + q) * n + r) * n + s]
                            - h.two[((r * n + s) * n + p) * n + q])
                            .abs()
                            > 1e-10
                        {
                            return Err(HfError::InvalidInput(
                                "MR-DSRG requires Hermitian two-body coefficients".into(),
                            ));
                        }
                    }
                }
            }
        }
        if self.excitations.iter().any(|e| !e.gap(h).is_finite()) {
            return Err(HfError::InvalidInput(
                "nonfinite MR-DSRG excitation gap".into(),
            ));
        }
        Ok(())
    }
    /// Solve the published Gaussian source condition with recursive GNO(2) BCH.
    pub fn solve_ldsrg2(
        &self,
        h: &No2bOperator,
        flow: f64,
        options: LdsrgOptions,
    ) -> Result<LdsrgSolution, HfError> {
        self.validate(h, flow, options)?;
        self.iterate(h, flow, options, |a| {
            self.algebra
                .bch(h, a, options.bch_tolerance, options.bch_max_order)
        })
    }
    /// QC convention: retain exactly the first two GNO(2) nested BCH terms.
    /// The Gaussian source is still solved to convergence for all amplitudes.
    pub fn solve_ldsrg2_qc(
        &self,
        h: &No2bOperator,
        flow: f64,
        options: LdsrgOptions,
    ) -> Result<LdsrgSolution, HfError> {
        self.validate(h, flow, options)?;
        self.iterate(h, flow, options, |a| {
            let first = self.algebra.commutator(h, a)?;
            let second = self.algebra.commutator(&first, a)?;
            let bar = h.add_scaled(&first, 1.).add_scaled(&second, 0.5);
            self.algebra.validate(&bar)?;
            Ok((bar, 2))
        })
    }
    fn iterate<F>(
        &self,
        drive: &No2bOperator,
        flow: f64,
        options: LdsrgOptions,
        mut evaluate: F,
    ) -> Result<LdsrgSolution, HfError>
    where
        F: FnMut(&No2bOperator) -> Result<(No2bOperator, usize), HfError>,
    {
        let gaps: Vec<_> = self.excitations.iter().map(|e| e.gap(drive)).collect();
        let mut t: Vec<_> = self
            .excitations
            .iter()
            .zip(&gaps)
            .map(|(e, &gap)| e.value(drive) * dsrg_regularized_inv_denom(gap, flow))
            .collect();
        let mut history = CdiisEngine::new(8);
        let mut previous = None::<f64>;
        for iteration in 1..=options.max_iter {
            let a = generator(self.classes.len(), &self.excitations, &t);
            let (bar, order) = evaluate(&a)?;
            let mut target = Vec::with_capacity(t.len());
            let mut residual = Vec::with_capacity(t.len());
            let mut amplitude_norm = 0f64;
            let mut physical_norm = 0f64;
            for ((&e, &gap), &amp) in self.excitations.iter().zip(&gaps).zip(&t) {
                let v = e.value(&bar);
                let next = (v + gap * amp) * dsrg_regularized_inv_denom(gap, flow);
                let physical =
                    -(-flow * gap * gap).exp_m1() * v - gap * amp * dsrg_residual_weight(gap, flow);
                let error = next - amp;
                if !error.is_finite() || !physical.is_finite() {
                    return Err(failure("nonfinite MR driven residual"));
                }
                amplitude_norm = amplitude_norm.max(error.abs());
                physical_norm = physical_norm.max(physical.abs());
                target.push(next);
                residual.push(error);
            }
            if amplitude_norm <= options.residual_tolerance
                && physical_norm <= options.residual_tolerance
                && previous.is_none_or(|old| (bar.zero - old).abs() <= options.energy_tolerance)
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
            previous = Some(bar.zero);
            history.push(
                Matrix::from_vec(1, t.len(), target.clone()),
                Matrix::from_vec(1, t.len(), residual),
            );
            t = history.extrapolate().map(|x| x.data).unwrap_or(target);
            if t.iter().any(|x| !x.is_finite()) {
                return Err(failure("nonfinite MR amplitudes"));
            }
        }
        Err(failure(format!(
            "MR driven equations did not converge in {} iterations",
            options.max_iter
        )))
    }
    // The Fdiag linear system is triangular: doubles first, then singles.
    // [H0,A2] supplies the active-density coupling required in both orders.
    fn linear_generator(
        &self,
        h: &No2bOperator,
        h0: &No2bOperator,
        k: &No2bOperator,
        flow: f64,
    ) -> Result<No2bOperator, HfError> {
        let mut t = vec![0.; self.excitations.len()];
        for (amp, &e) in t.iter_mut().zip(&self.excitations) {
            if matches!(e, Excitation::Double(..)) {
                *amp = e.value(k) * dsrg_regularized_inv_denom(e.gap(h), flow);
            }
        }
        let doubles = generator(self.classes.len(), &self.excitations, &t);
        let coupling = self.algebra.commutator(h0, &doubles)?;
        for (amp, &e) in t.iter_mut().zip(&self.excitations) {
            if matches!(e, Excitation::Single(..)) {
                *amp =
                    (e.value(k) + e.value(&coupling)) * dsrg_regularized_inv_denom(e.gap(h), flow);
            }
        }
        let out = generator(self.classes.len(), &self.excitations, &t);
        self.algebra.validate(&out)?;
        Ok(out)
    }
    /// Unrelaxed second-order increment, avoiding all second-order amplitudes.
    pub fn second_order_energy(
        &self,
        h: &No2bOperator,
        flow: f64,
        options: LdsrgOptions,
    ) -> Result<f64, HfError> {
        self.validate(h, flow, options)?;
        let mut h0 = zero(self.classes.len());
        for p in 0..self.classes.len() {
            h0.one[(p, p)] = h.one[(p, p)];
        }
        let mut h1 = h.add_scaled(&h0, -1.);
        h1.zero = 0.;
        let a1 = self.linear_generator(h, &h0, &h1, flow)?;
        let effective = h1.add_scaled(&self.algebra.commutator(&h0, &a1)?, 0.5);
        Ok(self.algebra.commutator(&effective, &a1)?.zero)
    }
    /// Second/third-order expansion of these same driven equations, Fdiag H0.
    /// The returned energies are increments relative to the reference scalar.
    pub fn perturbation_through_third_order(
        &self,
        h: &No2bOperator,
        flow: f64,
        options: LdsrgOptions,
    ) -> Result<DsrgPt3Solution, HfError> {
        Ok(self.third_order_hamiltonian(h, flow, options)?.0)
    }
    fn third_order_hamiltonian(
        &self,
        h: &No2bOperator,
        flow: f64,
        options: LdsrgOptions,
    ) -> Result<(DsrgPt3Solution, No2bOperator), HfError> {
        self.validate(h, flow, options)?;
        let mut h0 = zero(self.classes.len());
        for p in 0..self.classes.len() {
            h0.one[(p, p)] = h.one[(p, p)];
        }
        let mut h1 = h.add_scaled(&h0, -1.);
        h1.zero = 0.;
        let a1 = self.linear_generator(h, &h0, &h1, flow)?;
        let c01 = self.algebra.commutator(&h0, &a1)?;
        let c11 = self.algebra.commutator(&h1, &a1)?;
        let c011 = self.algebra.commutator(&c01, &a1)?;
        let k2 = c11.add_scaled(&c011, 0.5);
        let a2 = self.linear_generator(h, &h0, &k2, flow)?;
        let c02 = self.algebra.commutator(&h0, &a2)?;
        let k3 = self
            .algebra
            .commutator(&h1, &a2)?
            .add_scaled(&self.algebra.commutator(&c01, &a2)?, 0.5)
            .add_scaled(&self.algebra.commutator(&c02, &a1)?, 0.5)
            .add_scaled(&self.algebra.commutator(&c11, &a1)?, 0.5)
            .add_scaled(&self.algebra.commutator(&c011, &a1)?, 1. / 6.);
        let e3 = k3.zero;
        let bar = h
            .add_scaled(&c01, 1.)
            .add_scaled(&c02, 1.)
            .add_scaled(&k2, 1.)
            .add_scaled(&k3, 1.);
        if !k2.zero.is_finite() || !e3.is_finite() {
            return Err(failure("nonfinite MR perturbative energy"));
        }
        Ok((
            DsrgPt3Solution {
                second_order: k2.zero,
                third_order: e3,
                first_order_generator: a1,
                second_order_generator: a2,
            },
            bar,
        ))
    }
}
