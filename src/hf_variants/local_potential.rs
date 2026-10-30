// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Exact Gaussian multiplicative-potential matrix elements, without quadrature.

use crate::{Matrix, basis::ContractedGaussian, integrals::primitive_overlap};

fn binomial(n: usize, k: usize) -> f64 {
    (1..=k).fold(1., |value, j| value * (n + 1 - j) as f64 / j as f64)
}

fn pair(a: &ContractedGaussian, b: &ContractedGaussian, center: [f64; 3], exponent: f64) -> f64 {
    let distance: f64 = (0..3).map(|k| (a.center[k] - center[k]).powi(2)).sum();
    let mut value = 0.;
    for pa in &a.primitives {
        let combined = pa.exponent + exponent;
        let product: [f64; 3] =
            std::array::from_fn(|k| (pa.exponent * a.center[k] + exponent * center[k]) / combined);
        let prefactor = pa.coefficient * (-(pa.exponent * exponent / combined) * distance).exp();
        if prefactor == 0. {
            continue;
        }
        for pb in &b.primitives {
            for x in 0..=a.l[0] {
                for y in 0..=a.l[1] {
                    for z in 0..=a.l[2] {
                        let power = [x, y, z];
                        let translation: f64 = (0..3)
                            .map(|k| {
                                binomial(a.l[k], power[k])
                                    * (product[k] - a.center[k]).powi((a.l[k] - power[k]) as i32)
                            })
                            .product();
                        value += prefactor
                            * pb.coefficient
                            * translation
                            * primitive_overlap(
                                combined,
                                power,
                                product,
                                pb.exponent,
                                b.l,
                                b.center,
                            );
                    }
                }
            }
        }
    }
    value
}

/// `<chi_mu | exp(-exponent |r-center|^2) | chi_nu>` in the supplied AO frame.
pub fn gaussian_potential_matrix(
    basis: &[ContractedGaussian],
    center: [f64; 3],
    exponent: f64,
) -> Result<Matrix, String> {
    if basis.is_empty()
        || !exponent.is_finite()
        || exponent <= 0.
        || center.iter().any(|v| !v.is_finite())
    {
        return Err(
            "Gaussian local potentials require a basis, finite center and positive finite exponent"
                .into(),
        );
    }
    let n = basis.len();
    let mut result = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let value = pair(&basis[i], &basis[j], center, exponent);
            result[(i, j)] = value;
            result[(j, i)] = value;
        }
    }
    Ok(result)
}
