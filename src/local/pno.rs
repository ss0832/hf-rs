// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Pair Natural Orbitals (PNOs) for compact representation of electron pair correlation.
//!
//! Diagonalizes the pair correlation density matrix D^(ij) for each localized occupied pair (i, j)
//! to obtain a compact virtual subspace spanning only the correlating orbitals of that specific pair.

use crate::matrix::Matrix;
use crate::mo_integrals::MoEriTensor;

/// C^2 smooth quintic polynomial switching function S(t) = 6 t^5 - 15 t^4 + 10 t^3 for t in [0, 1].
/// S(0) = 0, S(1) = 1, S'(0) = S'(1) = S''(0) = S''(1) = 0.
#[inline]
pub fn quintic_switch(t: f64) -> f64 {
    if t <= 0.0 {
        0.0
    } else if t >= 1.0 {
        1.0
    } else {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }
}

/// Continuous taper function spanning [x_low, x_high].
/// Returns 0 for x <= x_low, 1 for x >= x_high, and C^2 smooth quintic in between.
#[inline]
pub fn smooth_taper(x: f64, x_low: f64, x_high: f64) -> f64 {
    if x <= x_low {
        0.0
    } else if x >= x_high || (x_high - x_low).abs() < 1e-15 {
        1.0
    } else {
        let t = (x - x_low) / (x_high - x_low);
        quintic_switch(t)
    }
}

/// Configuration options for Pair Natural Orbital construction and screening.
#[derive(Clone, Debug)]
pub struct PNOOptions {
    /// PNO occupation number threshold T_PNO (e.g. 1e-7 or 1e-8).
    pub t_pno: f64,
    /// Whether to apply continuous smoothing to PNO coefficients near T_PNO.
    pub smoothing: bool,
    /// Relative transition window delta for smoothing (e.g. 0.2 for +/- 20%).
    pub delta_pno: f64,
}

impl Default for PNOOptions {
    fn default() -> Self {
        Self {
            t_pno: 1e-7,
            smoothing: false,
            delta_pno: 0.2,
        }
    }
}

/// Pair Natural Orbitals representation for an occupied orbital pair (i, j).
#[derive(Clone, Debug)]
pub struct PNOBasis {
    pub pair: (usize, usize),
    pub occupations: Vec<f64>,
    pub pno_coefficients: Matrix, // Dimensions: n_virt x n_pno (semicanonicalized)
    pub semicanonical_energies: Vec<f64>, // Eigenvalues of projected virtual Fock matrix
}

/// Constructs Pair Natural Orbitals for all occupied orbital pairs (i, j)
/// with exact semicanonicalization of the virtual Fock matrix.
pub fn construct_pnos(
    n_occ: usize,
    n_virt: usize,
    eps: &[f64],
    mo_eri: &MoEriTensor,
    options: &PNOOptions,
) -> Result<Vec<PNOBasis>, crate::error::HfError> {
    use crate::error::HfError;
    if n_occ + n_virt != eps.len()
        || mo_eri.n_basis != eps.len()
        || eps.iter().any(|v| !v.is_finite())
        || !options.delta_pno.is_finite()
        || options.delta_pno < 0.
    {
        return Err(HfError::InvalidInput(
            "invalid canonical PNO dimensions/options".into(),
        ));
    }
    if options.smoothing {
        return Err(HfError::Unsupported {
            method: "PNO construction".into(),
            detail: "empirical PNO coefficient smoothing is not implemented".into(),
        });
    }
    let mut f = Matrix::zeros(n_virt, n_virt);
    for a in 0..n_virt {
        f[(a, a)] = eps[n_occ + a];
    }
    let mut out = Vec::new();
    for i in 0..n_occ {
        for j in 0..=i {
            let mut k = Matrix::zeros(n_virt, n_virt);
            for a in 0..n_virt {
                for b in 0..n_virt {
                    k[(a, b)] = mo_eri.get(i, n_occ + a, j, n_occ + b);
                }
            }
            let p = super::pair_space::natural_pair_space(
                &k,
                &f,
                eps[i] + eps[j],
                i == j,
                options.t_pno,
            )?;
            out.push(PNOBasis {
                pair: (i, j),
                occupations: p.natural_occupations,
                pno_coefficients: p.coefficients,
                semicanonical_energies: p.energies,
            });
        }
    }
    Ok(out)
}
