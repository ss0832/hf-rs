// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Ordinary real Gaussian AOs, common gauge A=B x (r-O)/2. These are orbital
//! Zeeman and diamagnetic integrals, without a London/GIAO basis transformation.

use crate::{
    Matrix,
    basis::ContractedGaussian,
    derivatives::diff_cg_1st,
    integrals::{contracted_overlap, primitive_overlap},
};

/// Cartesian moments with no renormalization of raised angular powers.
pub(super) fn moment(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    axes: &[usize],
    origin: [f64; 3],
) -> f64 {
    let mut terms = vec![(a.l, 1.)];
    for &axis in axes {
        let mut next = Vec::with_capacity(2 * terms.len());
        for (l, scale) in terms {
            let mut raised = l;
            raised[axis] += 1;
            next.push((raised, scale));
            next.push((l, scale * (a.center[axis] - origin[axis])));
        }
        terms = next;
    }
    let mut value = 0.;
    for pa in &a.primitives {
        for pb in &b.primitives {
            for &(l, scale) in &terms {
                value += pa.coefficient
                    * pb.coefficient
                    * scale
                    * primitive_overlap(pa.exponent, l, a.center, pb.exponent, b.l, b.center);
            }
        }
    }
    value
}

pub(super) fn magnetic_pair(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    field: [f64; 3],
    origin: [f64; 3],
) -> (f64, f64) {
    let square: f64 = field.iter().map(|b| b * b).sum();
    let mut real = 0.;
    let mut imag = 0.;
    for i in 0..3 {
        for j in 0..3 {
            let weight = (if i == j { square } else { 0. }) - field[i] * field[j];
            if weight != 0. {
                real += weight * moment(a, b, &[i, j], origin) / 8.;
            }
        }
        if field[i] != 0. {
            let (j, k) = ((i + 1) % 3, (i + 2) % 3);
            // Spatial differentiation is minus differentiation of the center.
            for t in diff_cg_1st(b, k).terms {
                imag += 0.5 * field[i] * moment(a, &t, &[j], origin);
            }
            for t in diff_cg_1st(b, j).terms {
                imag -= 0.5 * field[i] * moment(a, &t, &[k], origin);
            }
        }
    }
    (real, imag)
}

pub(super) fn origin_pair(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    field: [f64; 3],
    origin: [f64; 3],
    axis: usize,
) -> (f64, f64) {
    let square: f64 = field.iter().map(|b| b * b).sum();
    let mut real = 0.;
    let mut imag = 0.;
    for i in 0..3 {
        real -= ((if i == axis { square } else { 0. }) - field[i] * field[axis])
            * moment(a, b, &[i], origin)
            / 4.;
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        if j == axis {
            for t in diff_cg_1st(b, k).terms {
                imag -= 0.5 * field[i] * contracted_overlap(a, &t);
            }
        }
        if k == axis {
            for t in diff_cg_1st(b, j).terms {
                imag += 0.5 * field[i] * contracted_overlap(a, &t);
            }
        }
    }
    (real, imag)
}

/// Returns (diamagnetic real part, orbital-Zeeman imaginary part), in a.u.
pub fn magnetic_one_electron_integrals(
    basis: &[ContractedGaussian],
    field: [f64; 3],
    origin: [f64; 3],
) -> Result<(Matrix, Matrix), String> {
    if field.iter().chain(&origin).any(|v| !v.is_finite()) {
        return Err("invalid magnetic field or gauge origin".into());
    }
    let n = basis.len();
    let mut real = Matrix::zeros(n, n);
    let mut imag = real.clone();
    for p in 0..n {
        for q in 0..n {
            (real[(p, q)], imag[(p, q)]) = magnetic_pair(&basis[p], &basis[q], field, origin);
        }
    }
    Ok((real, imag))
}
