// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Spin-frame-invariant Fukutome classification from metric Pauli densities.
//! Henderson, Jimenez-Hoyos and Scuseria, DOI 10.1021/acs.jctc.7b01016,
//! Eqs. (6), (23) and Table I. Real representability is tested after a global
//! SU(2) rotation, without changing the spatial AO gauge.

use super::{
    complex_matrix::{ComplexMatrix, spin_diagonal},
    ghf::FukutomeClass,
};
use crate::Matrix;

#[derive(Clone, Debug)]
pub struct HfSymmetryAnalysis {
    pub class: FukutomeClass,
    pub spin_gram_eigenvalues: [f64; 3],
    pub real_spin_gram_eigenvalues: [f64; 3],
    pub imaginary_spin_gram_eigenvalues: [f64; 3],
    pub charge_current_norm: f64,
    pub time_reversal_invariant: bool,
    pub real_spin_frame_exists: bool,
}

fn gram(matrices: &[Matrix]) -> Matrix {
    let mut g = Matrix::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            g[(i, j)] = matrices[i]
                .data
                .iter()
                .zip(&matrices[j].data)
                .map(|(a, b)| a * b)
                .sum();
        }
    }
    g
}
fn significant_rank(values: &[f64], threshold: f64) -> usize {
    // Eigenvalues of a Gram matrix have an absolute roundoff floor set by
    // its trace. This is a diagnostic precision floor, not a physical cutoff.
    let floor = threshold + 64. * f64::EPSILON * values.iter().sum::<f64>().abs();
    values.iter().filter(|&&v| v > floor).count()
}

/// Classifies a single determinant, including paired UHF/GHF, using the lowest
/// symmetry class reachable by global spin rotation. The real AO gauge is fixed.
pub fn classify_complex_hf_symmetry(
    real: &Matrix,
    imag: &Matrix,
    overlap: &Matrix,
    tolerance: f64,
) -> Result<HfSymmetryAnalysis, String> {
    let n = overlap.rows;
    let dim = 2 * n;
    if n == 0
        || !tolerance.is_finite()
        || tolerance <= 0.
        || overlap.cols != n
        || overlap.data.len() != n * n
        || [real, imag]
            .iter()
            .any(|m| m.rows != dim || m.cols != dim || m.data.len() != dim * dim)
        || real
            .data
            .iter()
            .chain(&imag.data)
            .chain(&overlap.data)
            .any(|v| !v.is_finite())
    {
        return Err("invalid spin-density symmetry inputs".into());
    }
    for p in 0..n {
        for q in 0..n {
            if (overlap[(p, q)] - overlap[(q, p)]).abs() > tolerance {
                return Err("overlap must be symmetric".into());
            }
        }
    }
    let (e, u) = overlap.eigh();
    if e[n - 1] <= 0. || e.iter().any(|&v| v < -tolerance * e[n - 1]) {
        return Err("symmetry analysis requires positive semidefinite overlap".into());
    }
    let mut square = u.clone();
    for p in 0..n {
        for q in 0..n {
            square[(p, q)] *= e[q].max(0.).sqrt();
        }
    }
    let square = ComplexMatrix::real(spin_diagonal(&square.dot(&u.transpose())));
    let original = ComplexMatrix {
        re: real.clone(),
        im: imag.clone(),
    };
    let p = square.dot(&original).dot(&square);
    let scale = p.norm().max(1.);
    let threshold = (tolerance * scale).powi(2);
    if p.add(&p.dagger(), -1.).norm() > tolerance * scale
        || p.dot(&p).add(&p, -1.).norm() > tolerance * scale
    {
        return Err("symmetry analysis requires a Hermitian metric-idempotent determinant".into());
    }
    let mut mr = vec![Matrix::zeros(n, n); 3];
    let mut mi = mr.clone();
    let mut charge_imag = 0.;
    for a in 0..n {
        for b in 0..n {
            mr[0][(a, b)] = 0.5 * (p.re[(a, n + b)] + p.re[(n + a, b)]);
            mi[0][(a, b)] = 0.5 * (p.im[(a, n + b)] + p.im[(n + a, b)]);
            mr[1][(a, b)] = 0.5 * (p.im[(n + a, b)] - p.im[(a, n + b)]);
            mi[1][(a, b)] = 0.5 * (p.re[(a, n + b)] - p.re[(n + a, b)]);
            mr[2][(a, b)] = 0.5 * (p.re[(a, b)] - p.re[(n + a, n + b)]);
            mi[2][(a, b)] = 0.5 * (p.im[(a, b)] - p.im[(n + a, n + b)]);
            charge_imag += (0.5 * (p.im[(a, b)] + p.im[(n + a, n + b)])).powi(2);
        }
    }
    let gr = gram(&mr);
    let gi = gram(&mi);
    let gt = gr.add(&gi);
    let (et, _) = gt.eigh();
    let (er, _) = gr.eigh();
    let (ei, vi) = gi.eigh();
    let rank = significant_rank(&et, threshold);
    let rank_r = significant_rank(&er, threshold);
    let rank_i = significant_rank(&ei, threshold);
    let real_charge = charge_imag <= threshold;
    let time_reversal = real_charge && gr.trace() <= threshold;
    // Complex conjugation in some spin frame requires all imaginary spin
    // matrices to share one axis, perpendicular to every real spin matrix.
    let real_frame = real_charge
        && if rank_i == 0 {
            rank_r <= 2
        } else if rank_i == 1 {
            let axis = 2;
            let mut perpendicular = 0.;
            for a in 0..3 {
                for b in 0..3 {
                    perpendicular += vi[(a, axis)] * gr[(a, b)] * vi[(b, axis)];
                }
            }
            perpendicular.abs() <= threshold + 64. * f64::EPSILON * gr.trace().abs()
        } else {
            false
        };
    let class = if rank == 0 {
        if real_charge {
            FukutomeClass::TICS
        } else {
            FukutomeClass::CCW
        }
    } else if time_reversal {
        if rank == 1 {
            FukutomeClass::ASCW
        } else {
            FukutomeClass::TSCW
        }
    } else if rank == 1 {
        if real_charge && gi.trace() <= threshold {
            FukutomeClass::ASDW
        } else {
            FukutomeClass::ASW
        }
    } else if real_frame {
        FukutomeClass::TSDW
    } else {
        FukutomeClass::TSW
    };
    Ok(HfSymmetryAnalysis {
        class,
        spin_gram_eigenvalues: std::array::from_fn(|k| et[k]),
        real_spin_gram_eigenvalues: std::array::from_fn(|k| er[k]),
        imaginary_spin_gram_eigenvalues: std::array::from_fn(|k| ei[k]),
        charge_current_norm: charge_imag.sqrt(),
        time_reversal_invariant: time_reversal,
        real_spin_frame_exists: real_frame,
    })
}
