// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! LHF/CEDA occupied-space correction on a molecular quadrature.
//! C_ij=<i|v_x+K_spin|j>, v_x=v_S+sum_ij phi_i phi_j C_ij/rho_spin.
//! The HOMO diagonal is set to zero to fix the additive constant. Off-diagonal
//! symmetric coordinates have sqrt(2) normalization. No density floor is used.

use super::local_grid::LocalGrid;
use crate::Matrix;
use rayon::prelude::*;

pub(super) struct ExchangePoint {
    pub potential: Matrix,
    pub slater: Matrix,
    pub constants: Matrix,
    pub linear_residual: f64,
    pub smallest_linear_eigenvalue: f64,
}
struct Accumulator {
    slater: Vec<f64>,
    operators: Vec<f64>,
    metric: Vec<f64>,
}

pub(super) fn exchange_workspace(n: usize, n_occ: usize) -> Option<usize> {
    let m = n_occ.checked_mul(n_occ.checked_add(1)?)?.checked_div(2)?;
    let np = n.checked_mul(n.checked_add(1)?)?.checked_div(2)?;
    m.checked_mul(np)?
        .checked_add(m.checked_mul(m)?)?
        .checked_add(np)?
        .checked_mul(rayon::current_num_threads().checked_add(2)?)?
        .checked_add(n.checked_mul(n)?.checked_mul(32)?)?
        .checked_mul(8)
}

pub(super) fn build_exchange(
    grid: &LocalGrid,
    c: &Matrix,
    n_occ: usize,
    k_spin: &Matrix,
    linear_tolerance: f64,
    minimum_eigenvalue: f64,
    workspace_limit_bytes: usize,
) -> Result<ExchangePoint, String> {
    let n = c.rows;
    if n_occ == 0
        || n_occ > c.cols
        || k_spin.rows != n
        || k_spin.cols != n
        || grid.scaled_aos.cols != n
        || !linear_tolerance.is_finite()
        || linear_tolerance <= 0.
        || !minimum_eigenvalue.is_finite()
        || minimum_eigenvalue <= 0.
    {
        return Err("invalid local-exchange occupied frame or linear controls".into());
    }
    let no_pairs: Vec<_> = (0..n_occ)
        .flat_map(|i| (0..=i).map(move |j| (i, j)))
        .collect();
    let ao_pairs: Vec<_> = (0..n).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
    let m = no_pairs.len();
    let np = ao_pairs.len();
    let workers = rayon::current_num_threads().min(grid.points.len());
    let bytes = exchange_workspace(n, n_occ)
        .and_then(|v| v.checked_add(grid.estimated_workspace_bytes))
        .ok_or("local-exchange workspace overflow")?;
    if bytes > workspace_limit_bytes {
        return Err(format!(
            "local-exchange workspace estimate {bytes} exceeds budget {workspace_limit_bytes}"
        ));
    }
    let factors: Vec<f64> = no_pairs
        .iter()
        .map(|(i, j)| if i == j { 1. } else { 2f64.sqrt() })
        .collect();
    // One accumulator per explicit worker range bounds live reduction storage.
    let parts: Vec<_> = (0..workers).into_par_iter().map(|worker| -> Result<Accumulator, String> {
        let mut out = Accumulator { slater: vec![0.; np], operators: vec![0.; m*np], metric: vec![0.; m*m] };
        let mut kernel = Vec::new(); let mut scratch = Vec::new();
        let mut phi = vec![0.; n_occ]; let mut w = vec![0.; n]; let mut features = vec![0.; m];
        for point in worker * grid.points.len() / workers..(worker+1) * grid.points.len() / workers {
            let ao = &grid.scaled_aos.data[point*n..(point+1)*n];
            for i in 0..n_occ { phi[i] = (0..n).map(|mu| ao[mu]*c[(mu,i)]).sum(); }
            let norm = phi.iter().fold(0f64, |value, p| value.hypot(*p));
            if norm == 0. || !norm.is_finite() { return Err("occupied density ratio is not representable at a nonzero quadrature row; no density floor is applied".into()); }
            let rho_weight = grid.row_weights[point] * norm * norm;
            phi.iter_mut().for_each(|v| *v /= norm);
            for mu in 0..n { w[mu] = (0..n_occ).map(|i| c[(mu,i)]*phi[i]).sum(); }
            grid.kernel(point, &mut kernel, &mut scratch);
            let slater: f64 = -ao_pairs.iter().zip(&kernel).map(|(&(mu,nu), value)|
                w[mu]*w[nu]*value*(if mu==nu {1.} else {2.})).sum::<f64>();
            for (a, (&(i,j), &factor)) in no_pairs.iter().zip(&factors).enumerate() { features[a] = factor*phi[i]*phi[j]; }
            for a in 0..m { for b in 0..=a { out.metric[a*m+b] += rho_weight*features[a]*features[b]; } }
            for (p, &(mu,nu)) in ao_pairs.iter().enumerate() {
                let product = grid.row_weights[point]*ao[mu]*ao[nu];
                out.slater[p] += product*slater;
                for a in 0..m { out.operators[a*np+p] += product*features[a]; }
            }
        }
        Ok(out)
    }).collect::<Result<_,_>>()?;
    let mut sum = Accumulator {
        slater: vec![0.; np],
        operators: vec![0.; m * np],
        metric: vec![0.; m * m],
    };
    for part in parts {
        sum.slater
            .iter_mut()
            .zip(part.slater)
            .for_each(|(a, b)| *a += b);
        sum.operators
            .iter_mut()
            .zip(part.operators)
            .for_each(|(a, b)| *a += b);
        sum.metric
            .iter_mut()
            .zip(part.metric)
            .for_each(|(a, b)| *a += b);
    }
    let expand = |packed: &[f64]| {
        let mut matrix = Matrix::zeros(n, n);
        for (&(i, j), &value) in ao_pairs.iter().zip(packed) {
            matrix[(i, j)] = value;
            matrix[(j, i)] = value;
        }
        matrix
    };
    let slater = expand(&sum.slater);
    let co = c.leading_columns(n_occ);
    let rhs_matrix = co.transpose().dot(&slater.add(k_spin)).dot(&co);
    let rhs: Vec<f64> = no_pairs
        .iter()
        .zip(&factors)
        .map(|(&(i, j), factor)| factor * rhs_matrix[(i, j)])
        .collect();
    let mut system = Matrix::identity(m);
    for a in 0..m {
        for b in 0..=a {
            let v = sum.metric[a * m + b];
            system[(a, b)] -= v;
            if a != b {
                system[(b, a)] -= v;
            }
        }
    }
    let reduced = Matrix::from_vec(
        m - 1,
        m - 1,
        (0..m - 1)
            .flat_map(|a| (0..m - 1).map(move |b| (a, b)))
            .map(|(a, b)| system[(a, b)])
            .collect(),
    );
    let smallest_linear_eigenvalue = if m > 1 { reduced.eigh().0[0] } else { 1. };
    if smallest_linear_eigenvalue < minimum_eigenvalue {
        return Err(format!(
            "LHF gauge-fixed linear equation is ill-conditioned: minimum eigenvalue {smallest_linear_eigenvalue:e}"
        ));
    }
    let mut constants = vec![0.; m];
    if m > 1 {
        let lu = reduced
            .lu_decompose()
            .ok_or("singular LHF gauge-fixed linear equation")?;
        constants[..m - 1].copy_from_slice(&lu.solve(&rhs[..m - 1]));
    }
    let linear_residual = (0..m)
        .map(|a| ((0..m).map(|b| system[(a, b)] * constants[b]).sum::<f64>() - rhs[a]).abs())
        .fold(0f64, f64::max);
    if !linear_residual.is_finite() || linear_residual > linear_tolerance {
        return Err(format!(
            "local-grid LHF equation residual {linear_residual:e} exceeds {linear_tolerance:e}; refine quadrature"
        ));
    }
    let mut correction = vec![0.; np];
    for a in 0..m {
        for p in 0..np {
            correction[p] += constants[a] * sum.operators[a * np + p];
        }
    }
    let potential = slater.add(&expand(&correction));
    let mut matrix = Matrix::zeros(n_occ, n_occ);
    for (a, &(i, j)) in no_pairs.iter().enumerate() {
        let value = constants[a] / factors[a];
        matrix[(i, j)] = value;
        matrix[(j, i)] = value;
    }
    Ok(ExchangePoint {
        potential,
        slater,
        constants: matrix,
        linear_residual,
        smallest_linear_eigenvalue,
    })
}
