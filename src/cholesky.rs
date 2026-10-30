// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Pivoted Cholesky decomposition of the AO two-electron supermatrix computed on the fly
//! (Beebe & Linderberg, Int. J. Quantum Chem. 12, 683 (1977); Koch, Sanchez de Meras &
//! Pedersen, J. Chem. Phys. 118, 9481 (2003)).
//!
//! `(pq|rs) ~ sum_K L_K(pq) L_K(rs)` with the pivot of largest residual diagonal chosen at each
//! step (diagonals within 1e-12 relative of the largest are ties, resolved by the smallest packed
//! pair index). Only the diagonal `(pq|pq)` and the
//! columns `(pq|rs)` of the chosen pivots are evaluated, from complete shell quartets; neither a
//! four-index ERI tensor nor the AO-pair supermatrix is allocated. Because the residual matrix is
//! positive semidefinite, every reconstructed integral satisfies
//! `|(pq|rs) - sum_K L_K(pq) L_K(rs)| <= sqrt(D_pq D_rs) <= max_residual`.

use rayon::prelude::*;

use crate::ao::shell_ranges;
use crate::basis::ContractedGaussian;
use crate::df::DfTensor;
use crate::error::HfError;
use crate::matrix::Matrix;
use crate::scf::compound_index;
use crate::shell_integrals::compute_shell_quartet;

/// Diagnostics of a pivoted Cholesky factorization.
#[derive(Clone, Debug)]
pub struct CholeskyInfo {
    /// Largest remaining residual diagonal (upper bound of every reconstructed-integral error).
    pub max_residual: f64,
    /// Packed AO-pair index of each pivot, in selection order.
    pub pivots: Vec<usize>,
}

fn failure(detail: impl Into<String>) -> HfError {
    HfError::Integral(format!("pivoted Cholesky ERI decomposition: {}", detail.into()))
}

/// Pivoted Cholesky factors of the AO ERIs with residual diagonal threshold `threshold`.
/// The factor matrix uses the packed pair layout of [`DfTensor`]. `budget` bounds the stored
/// factor bytes.
pub fn pivoted_cholesky_eri(
    basis: &[ContractedGaussian],
    threshold: f64,
    budget: usize,
) -> Result<(DfTensor, CholeskyInfo), HfError> {
    let n = basis.len();
    if n == 0 || !threshold.is_finite() || threshold < 0.0 {
        return Err(HfError::InvalidInput("Cholesky requires a basis and a finite, nonnegative threshold".into()));
    }
    let shells = shell_ranges(basis)?;
    let ns = shells.len();
    let np = n * (n + 1) / 2;
    let shell_of: Vec<usize> = (0..n).map(|p| shells.iter().position(|r| r.contains(&p)).unwrap()).collect();
    // shell pairs (A >= B) and the packed AO pairs that belong to them
    let shell_pairs: Vec<(usize, usize)> = (0..ns).flat_map(|a| (0..=a).map(move |b| (a, b))).collect();
    let pair_of_ao_pair: Vec<usize> = {
        let mut out = vec![0; np];
        for p in 0..n {
            for q in 0..=p {
                let (a, b) = (shell_of[p], shell_of[q]);
                out[compound_index(p, q)] = a * (a + 1) / 2 + b;
            }
        }
        out
    };
    let slice = |s: usize| &basis[shells[s].clone()];

    // Diagonal (pq|pq) from (AB|AB) quartets.
    let diag_blocks: Vec<Vec<(usize, f64)>> = shell_pairs
        .par_iter()
        .map(|&(a, b)| -> Result<Vec<(usize, f64)>, HfError> {
            let block = compute_shell_quartet([slice(a), slice(b), slice(a), slice(b)])?;
            let (na, nb) = (shells[a].len(), shells[b].len());
            let mut out = Vec::new();
            for i in 0..na {
                for j in 0..nb {
                    let (p, q) = (shells[a].start + i, shells[b].start + j);
                    if p < q {
                        continue;
                    }
                    let ij = i * nb + j;
                    out.push((compound_index(p, q), block[ij * na * nb + ij]));
                }
            }
            Ok(out)
        })
        .collect::<Result<_, _>>()?;
    let mut diagonal = vec![0.0; np];
    for (idx, v) in diag_blocks.into_iter().flatten() {
        diagonal[idx] = v;
    }
    let scale = diagonal.iter().copied().fold(0.0_f64, f64::max).max(1.0);
    if diagonal.iter().any(|v| *v < -1e-12 * scale) {
        return Err(failure("negative ERI diagonal; the Coulomb metric is not positive semidefinite"));
    }
    let schwarz: Vec<f64> = {
        let mut out = vec![0.0_f64; shell_pairs.len()];
        for (idx, &d) in diagonal.iter().enumerate() {
            let sp = pair_of_ao_pair[idx];
            out[sp] = out[sp].max(d.max(0.0).sqrt());
        }
        out
    };

    // Exact columns (pq|r's') for all AO pairs r's' of one shell pair.
    let columns_of = |sp: usize| -> Result<Vec<(usize, Vec<f64>)>, HfError> {
        let (r, s) = shell_pairs[sp];
        let (nr, nss) = (shells[r].len(), shells[s].len());
        let mut targets = Vec::new();
        for k in 0..nr {
            for l in 0..nss {
                let (rr, ss) = (shells[r].start + k, shells[s].start + l);
                if rr >= ss {
                    targets.push((k, l, compound_index(rr, ss)));
                }
            }
        }
        let blocks: Vec<(usize, Vec<f64>)> = shell_pairs
            .par_iter()
            .enumerate()
            .filter(|&(pq, _)| schwarz[pq] * schwarz[sp] > 1e-15)
            .map(|(pq, &(a, b))| compute_shell_quartet([slice(a), slice(b), slice(r), slice(s)]).map(|blk| (pq, blk)))
            .collect::<Result<_, _>>()?;
        let mut cols: Vec<(usize, Vec<f64>)> = targets.iter().map(|t| (t.2, vec![0.0; np])).collect();
        for (pq, blk) in blocks {
            let (a, b) = shell_pairs[pq];
            let (na, nb) = (shells[a].len(), shells[b].len());
            for i in 0..na {
                for j in 0..nb {
                    let (p, q) = (shells[a].start + i, shells[b].start + j);
                    if p < q {
                        continue;
                    }
                    let row = compound_index(p, q);
                    for (t, &(k, l, _)) in targets.iter().enumerate() {
                        cols[t].1[row] = blk[((i * nb + j) * nr + k) * nss + l];
                    }
                }
            }
        }
        Ok(cols)
    };

    let mut factors: Vec<Vec<f64>> = Vec::new();
    let mut pivots = Vec::new();
    type CachedColumns = (usize, Vec<(usize, Vec<f64>)>);
    let mut cached: Option<CachedColumns> = None;
    let max_residual = loop {
        // Largest residual diagonal; diagonals equal to it within 1e-12 (relative) are ties and
        // resolve to the smallest packed index, so the pivot order does not depend on roundoff
        // (one-center diagonals of equivalent Cartesian components are exactly degenerate).
        let largest = diagonal.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let pivot = diagonal.iter().position(|&v| v >= largest - 1e-12 * largest.abs()).unwrap();
        let maximum = diagonal[pivot];
        if maximum <= threshold || factors.len() == np {
            break maximum.max(0.0);
        }
        if 8 * (np as u128) * (factors.len() as u128 + 2) > budget as u128 {
            return Err(failure("factor storage exceeds the memory budget"));
        }
        let sp = pair_of_ao_pair[pivot];
        if cached.as_ref().is_none_or(|(c, _)| *c != sp) {
            cached = Some((sp, columns_of(sp)?));
        }
        let column = &cached.as_ref().unwrap().1.iter().find(|(idx, _)| *idx == pivot).unwrap().1;
        let root = maximum.sqrt();
        let mut row: Vec<f64> = column
            .par_iter()
            .enumerate()
            .map(|(k, &v)| (v - factors.iter().map(|l| l[k] * l[pivot]).sum::<f64>()) / root)
            .collect();
        row[pivot] = root;
        for (d, l) in diagonal.iter_mut().zip(&row) {
            *d -= l * l;
            if *d < -1e-10 * scale {
                return Err(failure("nonpositive residual Coulomb diagonal"));
            }
            *d = d.max(0.0);
        }
        factors.push(row);
        pivots.push(pivot);
    };
    let rank = factors.len();
    if rank == 0 {
        return Err(failure("threshold exceeds every ERI diagonal; no Cholesky vectors were generated"));
    }
    let tensor = DfTensor { n_ao: n, n_aux_original: rank, b: Matrix::from_vec(rank, np, factors.into_iter().flatten().collect()) };
    Ok((tensor, CholeskyInfo { max_residual, pivots }))
}
