// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Exact CI coordinate changes under real spatial-orbital rotations.
//! Givens factors act directly on spin strings; no determinant-overlap matrix.

use super::{general_operator::checked_space_dimension, string::DeterminantSpace};
use crate::{Matrix, error::HfError};
use std::collections::HashMap;

/// C_new = C_old U. Return coefficients of the same state in the new orbitals.
/// CI order is alpha-major/beta-minor with alpha-before-beta creation order.
pub fn rotate_ci(
    space: &DeterminantSpace,
    coefficients: &[f64],
    u: &Matrix,
    memory_budget: usize,
) -> Result<Vec<f64>, HfError> {
    let dim = checked_space_dimension(space)?;
    let n = space.n_orbitals;
    if u.rows != n
        || u.cols != n
        || u.data.len() != n * n
        || coefficients.len() != dim
        || u.data.iter().chain(coefficients).any(|x| !x.is_finite())
    {
        return Err(HfError::InvalidInput(
            "invalid CI orbital rotation dimensions/values".into(),
        ));
    }
    let bytes = 8 * dim as u128
        + 64 * (space.alpha_strings.len() + space.beta_strings.len()) as u128
        + 24 * (n * n) as u128;
    if bytes > memory_budget as u128 {
        return Err(HfError::Unsupported {
            method: "CI orbital rotation".into(),
            detail: "vector/address workspace exceeds memory budget".into(),
        });
    }
    let metric = u.transpose().dot(u);
    for p in 0..n {
        for q in 0..n {
            if (metric[(p, q)] - if p == q { 1. } else { 0. }).abs() > 1e-10 {
                return Err(HfError::InvalidInput(
                    "CI orbital rotation must be orthogonal".into(),
                ));
            }
        }
    }
    let mut work = u.clone();
    let mut out = coefficients.to_vec();
    let alpha: HashMap<_, _> = space
        .alpha_strings
        .iter()
        .enumerate()
        .map(|(i, s)| (s.0, i))
        .collect();
    let beta: HashMap<_, _> = space
        .beta_strings
        .iter()
        .enumerate()
        .map(|(i, s)| (s.0, i))
        .collect();
    let nb = space.beta_strings.len();
    // Q_m ... Q_1 U = D, hence wedge(U)^T c = wedge(D) Q_m ... Q_1 c.
    for p in 0..n {
        for q in (p + 1..n).rev() {
            let a = work[(p, p)];
            let b = work[(q, p)];
            if b == 0. {
                continue;
            }
            let norm = a.hypot(b);
            let cosine = a / norm;
            let sine = b / norm;
            for k in 0..n {
                let (a, b) = (work[(p, k)], work[(q, k)]);
                work[(p, k)] = cosine * a + sine * b;
                work[(q, k)] = -sine * a + cosine * b;
            }
            for spin in 0..2 {
                let (strings, addresses) = if spin == 0 {
                    (&space.alpha_strings, &alpha)
                } else {
                    (&space.beta_strings, &beta)
                };
                for (source, string) in strings.iter().enumerate() {
                    if !string.is_occupied(p) || string.is_occupied(q) {
                        continue;
                    }
                    let (target, phase) = string
                        .apply_single_excitation(q, p)
                        .ok_or_else(|| HfError::Integral("missing Givens excitation".into()))?;
                    let target = *addresses
                        .get(&target.0)
                        .ok_or_else(|| HfError::Integral("missing Givens target string".into()))?;
                    let other = if spin == 0 {
                        nb
                    } else {
                        space.alpha_strings.len()
                    };
                    for k in 0..other {
                        let (i, j) = if spin == 0 {
                            (source * nb + k, target * nb + k)
                        } else {
                            (k * nb + source, k * nb + target)
                        };
                        let (a, b) = (out[i], out[j]);
                        out[i] = cosine * a + sine * phase * b;
                        out[j] = -sine * phase * a + cosine * b;
                    }
                }
            }
        }
    }
    let mut negative = 0u64;
    for p in 0..n {
        if work[(p, p)] < 0. {
            negative |= 1u64 << p;
        }
    }
    for (ia, a) in space.alpha_strings.iter().enumerate() {
        for (ib, b) in space.beta_strings.iter().enumerate() {
            if ((a.0 ^ b.0) & negative).count_ones() % 2 == 1 {
                out[ia * nb + ib] = -out[ia * nb + ib];
            }
        }
    }
    if out.iter().any(|x| !x.is_finite()) {
        return Err(HfError::Integral(
            "nonfinite rotated CI coefficients".into(),
        ));
    }
    Ok(out)
}
