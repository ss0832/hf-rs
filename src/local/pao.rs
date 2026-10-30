// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Projected Atomic Orbitals (PAOs) for local correlation methods.
//!
//! Spans the virtual orbital space by projecting out occupied orbital components
//! from the atomic orbital basis:
//! |tilde{mu}> = (1 - sum_i |phi_i><phi_i|) |chi_mu>

use crate::error::HfError;
use crate::matrix::Matrix;

/// Projected Atomic Orbital domain containing orthonormal virtual basis for an atomic cluster.
#[derive(Clone, Debug)]
pub struct PAODomain {
    pub atom_indices: Vec<usize>,
    pub pao_coefficients: Matrix, // Dimensions: n_basis x n_pao
}

/// Constructs the non-redundant Projected Atomic Orbitals (PAOs) spanning the virtual space.
///
/// - `c_occ`: Localized occupied orbital coefficient matrix (n_basis x n_occ).
/// - `overlap`: Overlap matrix S (n_basis x n_basis).
/// - `threshold`: Redundancy screening threshold for eigenvalues of S_PAO (e.g. 1e-4).
pub fn construct_paos(c_occ: &Matrix, overlap: &Matrix, threshold: f64) -> Result<Matrix, HfError> {
    let n_basis = overlap.rows;
    if overlap.cols != n_basis || c_occ.rows != n_basis || !threshold.is_finite() || threshold < 0.0
    {
        return Err(HfError::InvalidInput(
            "invalid PAO dimensions or threshold".into(),
        ));
    }

    // Density matrix in AO basis: P = C_occ * C_occ^T
    let p_ao = c_occ.dot(&c_occ.transpose());

    // Projector Q = I - P * S
    let ps = p_ao.dot(overlap);
    let eye = Matrix::identity(n_basis);
    let q_proj = eye.sub(&ps);

    orthonormalize_paos(
        &q_proj,
        overlap,
        &(0..n_basis).collect::<Vec<_>>(),
        threshold,
    )
}

/// Orthonormalize selected projected AOs as Q U lambda^(-1/2).
pub fn orthonormalize_paos(
    projector: &Matrix,
    overlap: &Matrix,
    columns: &[usize],
    threshold: f64,
) -> Result<Matrix, HfError> {
    let n_basis = overlap.rows;
    if overlap.cols != n_basis
        || projector.rows != n_basis
        || projector.cols != n_basis
        || !threshold.is_finite()
        || threshold < 0.0
        || columns.iter().any(|&i| i >= n_basis)
        || overlap
            .data
            .iter()
            .chain(&projector.data)
            .any(|x| !x.is_finite())
    {
        return Err(HfError::InvalidInput(
            "invalid PAO projector/overlap/threshold".into(),
        ));
    }
    let mut q_proj = Matrix::zeros(n_basis, columns.len());
    for (j, &col) in columns.iter().enumerate() {
        for i in 0..n_basis {
            q_proj[(i, j)] = projector[(i, col)];
        }
    }
    if columns.is_empty() {
        return Ok(q_proj);
    }
    let s_pao = q_proj.transpose().dot(overlap).dot(&q_proj);

    // Diagonalize S_pao to eliminate redundancies and linear dependencies
    let (evals, evecs) = s_pao.eigh();
    if evals.iter().any(|v| *v < -1e-10) {
        return Err(HfError::Integral(
            "PAO overlap has a negative eigenvalue".into(),
        ));
    }

    let mut non_redundant_cols = Vec::new();
    for (col_idx, &val) in evals.iter().enumerate() {
        if val > threshold.max(1e-10) {
            let col: Vec<f64> = (0..columns.len())
                .map(|r| evecs[(r, col_idx)] / val.sqrt())
                .collect();
            non_redundant_cols.push(col);
        }
    }

    let n_pao = non_redundant_cols.len();
    let mut c_pao = Matrix::zeros(columns.len(), n_pao);
    for (col_idx, col) in non_redundant_cols.into_iter().enumerate() {
        for row_idx in 0..columns.len() {
            c_pao[(row_idx, col_idx)] = col[row_idx];
        }
    }

    Ok(q_proj.dot(&c_pao))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pao_construction() {
        let s = Matrix::identity(4);
        let mut c_occ = Matrix::zeros(4, 2);
        c_occ[(0, 0)] = 1.0;
        c_occ[(1, 1)] = 1.0;

        let paos = construct_paos(&c_occ, &s, 1e-4).unwrap();
        assert_eq!(paos.cols, 2); // 4 - 2 = 2 non-redundant virtual PAOs
    }
}
