// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! Real Jacobi localization of an orbital subspace. Boys dipoles and Mulliken
//! population matrices share the same exact pair maximization.
use crate::error::HfError;
use crate::matrix::Matrix;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalizationMethod {
    FosterBoys,
    PipekMezey,
}
#[derive(Clone, Debug)]
pub struct LocalizationResult {
    pub coefficients: Matrix,
    pub rotation: Matrix,
    pub objective_history: Vec<f64>,
    pub gradient_max: f64,
    pub sweeps: usize,
}
fn finite(m: &Matrix) -> bool {
    m.data.iter().all(|x| x.is_finite())
}
fn symmetric(m: &Matrix, n: usize) -> bool {
    m.rows == n
        && m.cols == n
        && finite(m)
        && (0..n).all(|i| (0..i).all(|j| (m[(i, j)] - m[(j, i)]).abs() < 1e-12))
}
fn objective(ops: &[Matrix]) -> f64 {
    ops.iter()
        .map(|m| (0..m.rows).map(|i| m[(i, i)].powi(2)).sum::<f64>())
        .sum()
}
fn pair(ops: &[Matrix], i: usize, j: usize) -> (f64, f64) {
    let mut a = 0.0;
    let mut b = 0.0;
    for m in ops {
        let d = m[(i, i)] - m[(j, j)];
        let x = m[(i, j)];
        a += d * d - 4.0 * x * x;
        b += 4.0 * d * x;
    }
    (a, b)
}
fn rotate_columns(m: &mut Matrix, i: usize, j: usize, c: f64, s: f64) {
    for k in 0..m.rows {
        let x = m[(k, i)];
        let y = m[(k, j)];
        m[(k, i)] = c * x + s * y;
        m[(k, j)] = -s * x + c * y;
    }
}
fn optimize(
    c: &Matrix,
    mut ops: Vec<Matrix>,
    max_sweeps: usize,
    tol: f64,
) -> Result<LocalizationResult, HfError> {
    let n = c.cols;
    if !finite(c)
        || max_sweeps == 0
        || !tol.is_finite()
        || tol <= 0.0
        || ops.is_empty()
        || ops.iter().any(|m| !symmetric(m, n))
    {
        return Err(HfError::InvalidInput(
            "invalid localization matrices or convergence controls".into(),
        ));
    }
    let mut rotation = Matrix::identity(n);
    let mut history = vec![objective(&ops)];
    for sweep in 0..=max_sweeps {
        let mut gradient = 0.0_f64;
        let mut gain = 0.0_f64;
        for i in 0..n {
            for j in i + 1..n {
                let (a, b) = pair(&ops, i, j);
                gradient = gradient.max(b.abs());
                gain = gain.max(0.25 * (a.hypot(b) - a));
            }
        }
        // A pair minimum also has zero gradient. Check the possible gain to
        // detect the theta=pi/4 escape before accepting a stationary point.
        if gradient < tol && gain < 1e-14 * history.last().copied().unwrap().abs().max(1.0) {
            return Ok(LocalizationResult {
                coefficients: c.dot(&rotation),
                rotation,
                objective_history: history,
                gradient_max: gradient,
                sweeps: sweep,
            });
        }
        if sweep == max_sweeps {
            break;
        }
        for i in 0..n {
            for j in i + 1..n {
                let (a, b) = pair(&ops, i, j);
                if a == 0.0 && b == 0.0 {
                    continue;
                }
                // C_i'=cos(theta) C_i+sin(theta) C_j. Pair objective:
                // const+[a cos(4 theta)+b sin(4 theta)]/4. This is its maximum.
                let angle = 0.25 * b.atan2(a);
                let co = angle.cos();
                let si = angle.sin();
                for m in &mut ops {
                    rotate_columns(m, i, j, co, si);
                    for k in 0..n {
                        let x = m[(i, k)];
                        let y = m[(j, k)];
                        m[(i, k)] = co * x + si * y;
                        m[(j, k)] = -si * x + co * y;
                    }
                }
                rotate_columns(&mut rotation, i, j, co, si);
            }
        }
        let next = objective(&ops);
        if !next.is_finite() || next + 1e-12 * next.abs().max(1.0) < *history.last().unwrap() {
            return Err(HfError::Response(
                "localization objective decreased or became non-finite".into(),
            ));
        }
        history.push(next);
    }
    Err(HfError::Response(format!(
        "orbital localization did not converge in {max_sweeps} sweeps"
    )))
}
pub fn foster_boys(
    c: &Matrix,
    dipole_ao: &[Matrix; 3],
    max_sweeps: usize,
    tol: f64,
) -> Result<LocalizationResult, HfError> {
    if dipole_ao.iter().any(|m| !symmetric(m, c.rows)) {
        return Err(HfError::InvalidInput(
            "Boys AO dipole dimensions/symmetry are invalid".into(),
        ));
    }
    let ops = dipole_ao
        .iter()
        .map(|m| c.transpose().dot(m).dot(c))
        .collect();
    optimize(c, ops, max_sweeps, tol)
}
pub fn localize_foster_boys(
    c: &Matrix,
    dipole_ao: &[Matrix; 3],
    max_sweeps: usize,
    tol: f64,
) -> Result<Matrix, HfError> {
    Ok(foster_boys(c, dipole_ao, max_sweeps, tol)?.coefficients)
}
pub fn pipek_mezey(
    c: &Matrix,
    overlap: &Matrix,
    ranges: &[(usize, usize)],
    max_sweeps: usize,
    tol: f64,
) -> Result<LocalizationResult, HfError> {
    if !symmetric(overlap, c.rows) {
        return Err(HfError::InvalidInput(
            "PM AO overlap dimensions/symmetry are invalid".into(),
        ));
    }
    let mut seen = vec![false; c.rows];
    let sc = overlap.dot(c);
    let mut ops = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        if start > end || end > c.rows || seen[start..end].iter().any(|x| *x) {
            return Err(HfError::InvalidInput(
                "PM atom ranges overlap or exceed AO rows".into(),
            ));
        }
        seen[start..end].fill(true);
        let mut m = Matrix::zeros(c.cols, c.cols);
        for i in 0..c.cols {
            for j in 0..c.cols {
                m[(i, j)] = (start..end)
                    .map(|mu| 0.5 * (c[(mu, i)] * sc[(mu, j)] + sc[(mu, i)] * c[(mu, j)]))
                    .sum();
            }
        }
        ops.push(m);
    }
    if seen.iter().any(|x| !*x) {
        return Err(HfError::InvalidInput(
            "PM atom ranges must cover the AO rows".into(),
        ));
    }
    optimize(c, ops, max_sweeps, tol)
}
pub fn localize_pipek_mezey(
    c: &Matrix,
    s: &Matrix,
    ranges: &[(usize, usize)],
    max_sweeps: usize,
    tol: f64,
) -> Result<Matrix, HfError> {
    Ok(pipek_mezey(c, s, ranges, max_sweeps, tol)?.coefficients)
}
