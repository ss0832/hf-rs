// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! Checked Gaussian-geminal operators. Coulomb-geminal auxiliaries use a positive
//! binomial sum of Boys moments; no numerical radial quadrature is used here.
use super::gaussian_quartet::GaussianQuartet;
use crate::{basis::ContractedGaussian, error::HfError};
#[derive(Clone, Copy, Debug)]
pub enum GaussianKernel {
    Gaussian(f64),
    GaussianCoulomb(f64),
    GaussianR2(f64),
    /// [f12,T1+T2], with the bra and ket Gaussian kinetic actions explicit.
    KineticCommutator(f64),
}
#[derive(Clone, Debug)]
pub struct GaussianGeminal {
    pub terms: Vec<(f64, f64)>,
}
#[derive(Clone, Copy, Debug)]
pub enum GeminalOperator {
    Factor,
    FactorOverR,
    FactorSquared,
    PositiveDoubleCommutator,
    KineticCommutator,
}
impl GaussianGeminal {
    pub fn new(terms: Vec<(f64, f64)>) -> Result<Self, HfError> {
        if terms.is_empty()
            || terms
                .iter()
                .any(|(a, c)| !a.is_finite() || *a < 0. || !c.is_finite())
        {
            return Err(HfError::InvalidInput(
                "Gaussian geminals require finite nonnegative exponents and finite coefficients"
                    .into(),
            ));
        }
        Ok(Self { terms })
    }
    pub fn value(&self, r: f64) -> f64 {
        self.terms.iter().map(|(a, c)| c * (-a * r * r).exp()).sum()
    }
}
pub fn primitive_operator(
    exponents: [f64; 4],
    powers: [[usize; 3]; 4],
    centers: [[f64; 3]; 4],
    kernel: GaussianKernel,
) -> Result<f64, HfError> {
    let gamma = match kernel {
        GaussianKernel::Gaussian(a)
        | GaussianKernel::GaussianCoulomb(a)
        | GaussianKernel::GaussianR2(a)
        | GaussianKernel::KineticCommutator(a) => a,
    };
    if exponents.iter().any(|x| !x.is_finite() || *x <= 0.)
        || centers.iter().flatten().any(|v| !v.is_finite())
        || powers.iter().any(|l| l.iter().sum::<usize>() > 7)
        || !gamma.is_finite()
        || gamma < 0.
    {
        return Err(HfError::InvalidInput(
            "invalid primitive geminal exponents, centers or angular momenta".into(),
        ));
    }
    let quartet = GaussianQuartet::new(exponents, powers, centers);
    let value = match kernel {
        GaussianKernel::Gaussian(a) => quartet.gaussian(a),
        GaussianKernel::GaussianCoulomb(a) => quartet.coulomb(a),
        GaussianKernel::GaussianR2(a) => quartet.r2(a),
        GaussianKernel::KineticCommutator(a) => {
            if a == 0.0 {
                0.0
            } else {
                let (base, raised) = kinetic_preparations(exponents, powers, centers);
                base * quartet.gaussian(a)
                    + raised
                        .iter()
                        .map(|(weight, q)| weight * q.gaussian(a))
                        .sum::<f64>()
            }
        }
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(HfError::Integral(
            "nonfinite primitive geminal integral".into(),
        ))
    }
}
fn kinetic_preparations(
    e: [f64; 4],
    l: [[usize; 3]; 4],
    c: [[f64; 3]; 4],
) -> (f64, Vec<(f64, GaussianQuartet)>) {
    let mut base = 0.0;
    let mut raised = Vec::with_capacity(24);
    for orbital in 0..4 {
        let sign = if orbital % 2 == 0 { -1.0 } else { 1.0 };
        base += sign * e[orbital] * (2 * l[orbital].iter().sum::<usize>() + 3) as f64;
        for axis in 0..3 {
            let mut powers = l;
            powers[orbital][axis] += 2;
            raised.push((
                -sign * 2.0 * e[orbital].powi(2),
                GaussianQuartet::new(e, powers, c),
            ));
            let degree = l[orbital][axis];
            if degree >= 2 {
                let mut powers = l;
                powers[orbital][axis] -= 2;
                raised.push((
                    -sign * 0.5 * (degree * (degree - 1)) as f64,
                    GaussianQuartet::new(e, powers, c),
                ));
            }
        }
    }
    (base, raised)
}

/// Checked fit/product coefficients prepared once and reused across AO quartets.
#[derive(Clone, Debug)]
pub struct PreparedGeminalOperator {
    terms: Vec<(GaussianKernel, f64)>,
    operator: GeminalOperator,
}

impl PreparedGeminalOperator {
    pub fn new(factor: &GaussianGeminal, operator: GeminalOperator) -> Result<Self, HfError> {
        GaussianGeminal::new(factor.terms.clone())?;
        let mut terms = Vec::new();
        match operator {
            GeminalOperator::Factor
            | GeminalOperator::FactorOverR
            | GeminalOperator::KineticCommutator => {
                for &(exponent, coefficient) in &factor.terms {
                    let kernel = match operator {
                        GeminalOperator::FactorOverR => GaussianKernel::GaussianCoulomb(exponent),
                        GeminalOperator::KineticCommutator => {
                            GaussianKernel::KineticCommutator(exponent)
                        }
                        _ => GaussianKernel::Gaussian(exponent),
                    };
                    terms.push((kernel, coefficient));
                }
            }
            GeminalOperator::FactorSquared | GeminalOperator::PositiveDoubleCommutator => {
                for (i, &(x, cx)) in factor.terms.iter().enumerate() {
                    for (j, &(y, cy)) in factor.terms[..=i].iter().enumerate() {
                        let coefficient = cx * cy * if i == j { 1. } else { 2. };
                        let (kernel, weight) =
                            if matches!(operator, GeminalOperator::PositiveDoubleCommutator) {
                                (GaussianKernel::GaussianR2(x + y), 8. * x * y)
                            } else {
                                (GaussianKernel::Gaussian(x + y), 1.)
                            };
                        terms.push((kernel, coefficient * weight));
                    }
                }
            }
        }
        if terms.iter().any(|&(kernel, weight)| {
            let exponent = match kernel {
                GaussianKernel::Gaussian(a)
                | GaussianKernel::GaussianCoulomb(a)
                | GaussianKernel::GaussianR2(a)
                | GaussianKernel::KineticCommutator(a) => a,
            };
            !exponent.is_finite() || !weight.is_finite()
        }) {
            return Err(HfError::InvalidInput(
                "nonfinite Gaussian geminal product exponent or coefficient".into(),
            ));
        }
        terms.retain(|(kernel, weight)| {
            *weight != 0.0 && !matches!(kernel,GaussianKernel::KineticCommutator(a) if *a==0.0)
        });
        Ok(Self { terms, operator })
    }

    pub fn contracted(
        &self,
        a: &ContractedGaussian,
        b: &ContractedGaussian,
        c: &ContractedGaussian,
        d: &ContractedGaussian,
    ) -> Result<f64, HfError> {
        self.contracted_with_units([a, b, c, d], [false; 4])
    }

    /// (ab|operator|P), with an auxiliary function on electron two.
    pub fn three_center(
        &self,
        a: &ContractedGaussian,
        b: &ContractedGaussian,
        auxiliary: &ContractedGaussian,
    ) -> Result<f64, HfError> {
        self.require_multiplicative()?;
        let unit = crate::df::constant(auxiliary.center);
        self.contracted_with_units([a, b, auxiliary, &unit], [false, false, false, true])
    }

    /// (P|operator|Q). Unit functions are internal and are never normalized.
    pub fn two_center(
        &self,
        p: &ContractedGaussian,
        q: &ContractedGaussian,
    ) -> Result<f64, HfError> {
        self.require_multiplicative()?;
        let (one_p, one_q) = (crate::df::constant(p.center), crate::df::constant(q.center));
        self.contracted_with_units([p, &one_p, q, &one_q], [false, true, false, true])
    }

    fn require_multiplicative(&self) -> Result<(), HfError> {
        if matches!(self.operator, GeminalOperator::KineticCommutator) {
            return Err(HfError::Unsupported {
                method: "lower-center F12 integral".into(),
                detail: "the kinetic single commutator is not a multiplicative density kernel"
                    .into(),
            });
        }
        Ok(())
    }

    fn contracted_with_units(
        &self,
        [a, b, c, d]: [&ContractedGaussian; 4],
        units: [bool; 4],
    ) -> Result<f64, HfError> {
        if [a, b, c, d].iter().enumerate().any(|(index, g)| {
            g.primitives.is_empty()
                || g.l.iter().sum::<usize>() > 7
                || g.center.iter().any(|v| !v.is_finite())
                || g.primitives.iter().any(|p| {
                    !p.exponent.is_finite()
                        || p.exponent < 0.0
                        || (p.exponent == 0.0 && !units[index])
                        || !p.coefficient.is_finite()
                })
        }) {
            return Err(HfError::InvalidInput(
                "invalid contracted Gaussian geminal basis".into(),
            ));
        }
        if self.terms.is_empty() {
            return Ok(0.0);
        }
        let powers = [a.l, b.l, c.l, d.l];
        let centers = [a.center, b.center, c.center, d.center];
        let mut result = 0.;
        for pa in &a.primitives {
            for pb in &b.primitives {
                for pc in &c.primitives {
                    for pd in &d.primitives {
                        let exponents = [pa.exponent, pb.exponent, pc.exponent, pd.exponent];
                        let coefficient =
                            pa.coefficient * pb.coefficient * pc.coefficient * pd.coefficient;
                        let quartet = GaussianQuartet::new(exponents, powers, centers);
                        let kinetic = if matches!(self.operator, GeminalOperator::KineticCommutator)
                        {
                            Some(kinetic_preparations(exponents, powers, centers))
                        } else {
                            None
                        };
                        for &(kernel, weight) in &self.terms {
                            let value = match kernel {
                                GaussianKernel::Gaussian(a) => quartet.gaussian(a),
                                GaussianKernel::GaussianCoulomb(a) => quartet.coulomb(a),
                                GaussianKernel::GaussianR2(a) => quartet.r2(a),
                                GaussianKernel::KineticCommutator(a) => {
                                    let (base, raised) = kinetic.as_ref().ok_or_else(|| {
                                        HfError::Integral(
                                            "missing kinetic geminal preparation".into(),
                                        )
                                    })?;
                                    base * quartet.gaussian(a)
                                        + raised
                                            .iter()
                                            .map(|(weight, q)| weight * q.gaussian(a))
                                            .sum::<f64>()
                                }
                            };
                            result += coefficient * weight * value;
                        }
                    }
                }
            }
        }
        if result.is_finite() {
            Ok(result)
        } else {
            Err(HfError::Integral(
                "nonfinite contracted geminal integral".into(),
            ))
        }
    }
}

pub fn contracted_operator(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    c: &ContractedGaussian,
    d: &ContractedGaussian,
    factor: &GaussianGeminal,
    operator: GeminalOperator,
) -> Result<f64, HfError> {
    PreparedGeminalOperator::new(factor, operator)?.contracted(a, b, c, d)
}
