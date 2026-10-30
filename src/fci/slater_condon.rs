// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Slater-Condon rules for evaluating Hamiltonian matrix elements and direct H*v action.

use rayon::prelude::*;
use crate::fci::string::{DeterminantSpace, SpinString};
use crate::matrix::Matrix;
use crate::mo_integrals::MoEriTensor;

/// Evaluates the Hamiltonian matrix element <Phi_I | H | Phi_J> between two determinants
/// using Slater-Condon rules.
pub fn slater_condon_element(
    alpha_i: &SpinString,
    beta_i: &SpinString,
    alpha_j: &SpinString,
    beta_j: &SpinString,
    h_mo: &Matrix,
    mo_eri: &MoEriTensor,
) -> f64 {
    let diff_a = (alpha_i.0 ^ alpha_j.0).count_ones();
    let diff_b = (beta_i.0 ^ beta_j.0).count_ones();

    // Total spin-orbital difference is (diff_a + diff_b) / 2
    match (diff_a, diff_b) {
        // Case 0: Diagonal element (identical determinants)
        (0, 0) => {
            let mut val = 0.0;
            let occ_a = alpha_i.occupied_orbitals();
            let occ_b = beta_i.occupied_orbitals();

            // 1-electron alpha
            for &p in &occ_a {
                val += h_mo[(p, p)];
            }
            // 1-electron beta
            for &p in &occ_b {
                val += h_mo[(p, p)];
            }

            // 2-electron alpha-alpha: 0.5 * sum [ (pp|qq) - (pq|qp) ]
            for (idx_p, &p) in occ_a.iter().enumerate() {
                for &q in occ_a.iter().skip(idx_p + 1) {
                    val += mo_eri.get(p, p, q, q) - mo_eri.get(p, q, q, p);
                }
            }

            // 2-electron beta-beta: 0.5 * sum [ (pp|qq) - (pq|qp) ]
            for (idx_p, &p) in occ_b.iter().enumerate() {
                for &q in occ_b.iter().skip(idx_p + 1) {
                    val += mo_eri.get(p, p, q, q) - mo_eri.get(p, q, q, p);
                }
            }

            // 2-electron alpha-beta: sum (pp|qq)
            for &p in &occ_a {
                for &q in &occ_b {
                    val += mo_eri.get(p, p, q, q);
                }
            }

            val
        }

        // Case 1: Single excitation in alpha (diff_a = 2, diff_b = 0)
        (2, 0) => {
            let p = (alpha_i.0 & !alpha_j.0).trailing_zeros() as usize; // in i but not j
            let q = (alpha_j.0 & !alpha_i.0).trailing_zeros() as usize; // in j but not i

            let min_idx = p.min(q);
            let max_idx = p.max(q);
            let mask = ((1u64 << max_idx) - 1) & !((1u64 << (min_idx + 1)) - 1);
            let phase = if (alpha_i.0 & mask).count_ones() % 2 == 1 { -1.0 } else { 1.0 };

            let mut me = h_mo[(p, q)];
            let occ_a = alpha_i.occupied_orbitals();
            let occ_b = beta_i.occupied_orbitals();

            // alpha-alpha Coulomb - Exchange
            for &k in &occ_a {
                if k != p {
                    me += mo_eri.get(p, q, k, k) - mo_eri.get(p, k, k, q);
                }
            }

            // alpha-beta Coulomb
            for &k in &occ_b {
                me += mo_eri.get(p, q, k, k);
            }

            phase * me
        }

        // Case 2: Single excitation in beta (diff_a = 0, diff_b = 2)
        (0, 2) => {
            let p = (beta_i.0 & !beta_j.0).trailing_zeros() as usize;
            let q = (beta_j.0 & !beta_i.0).trailing_zeros() as usize;

            let min_idx = p.min(q);
            let max_idx = p.max(q);
            let mask = ((1u64 << max_idx) - 1) & !((1u64 << (min_idx + 1)) - 1);
            let phase = if (beta_i.0 & mask).count_ones() % 2 == 1 { -1.0 } else { 1.0 };

            let mut me = h_mo[(p, q)];
            let occ_a = alpha_i.occupied_orbitals();
            let occ_b = beta_i.occupied_orbitals();

            // beta-beta Coulomb - Exchange
            for &k in &occ_b {
                if k != p {
                    me += mo_eri.get(p, q, k, k) - mo_eri.get(p, k, k, q);
                }
            }

            // beta-alpha Coulomb
            for &k in &occ_a {
                me += mo_eri.get(p, q, k, k);
            }

            phase * me
        }

        // Case 3: Double excitation in alpha (diff_a = 4, diff_b = 0)
        (4, 0) => {
            let mut diff_p = alpha_i.0 & !alpha_j.0;
            let p1 = diff_p.trailing_zeros() as usize;
            diff_p &= diff_p - 1;
            let p2 = diff_p.trailing_zeros() as usize;

            let mut diff_q = alpha_j.0 & !alpha_i.0;
            let q1 = diff_q.trailing_zeros() as usize;
            diff_q &= diff_q - 1;
            let q2 = diff_q.trailing_zeros() as usize;

            // Phase determination via permutation parity
            let s = *alpha_j;
            let (s1, ph1) = s.apply_single_excitation(p1, q1).unwrap();
            let (_, ph2) = s1.apply_single_excitation(p2, q2).unwrap();
            let phase = ph1 * ph2;

            phase * (mo_eri.get(p1, q1, p2, q2) - mo_eri.get(p1, q2, p2, q1))
        }

        // Case 4: Double excitation in beta (diff_a = 0, diff_b = 4)
        (0, 4) => {
            let mut diff_p = beta_i.0 & !beta_j.0;
            let p1 = diff_p.trailing_zeros() as usize;
            diff_p &= diff_p - 1;
            let p2 = diff_p.trailing_zeros() as usize;

            let mut diff_q = beta_j.0 & !beta_i.0;
            let q1 = diff_q.trailing_zeros() as usize;
            diff_q &= diff_q - 1;
            let q2 = diff_q.trailing_zeros() as usize;

            let s = *beta_j;
            let (s1, ph1) = s.apply_single_excitation(p1, q1).unwrap();
            let (_, ph2) = s1.apply_single_excitation(p2, q2).unwrap();
            let phase = ph1 * ph2;

            phase * (mo_eri.get(p1, q1, p2, q2) - mo_eri.get(p1, q2, p2, q1))
        }

        // Case 5: Mixed double excitation (1 alpha, 1 beta: diff_a = 2, diff_b = 2)
        (2, 2) => {
            let p_a = (alpha_i.0 & !alpha_j.0).trailing_zeros() as usize;
            let q_a = (alpha_j.0 & !alpha_i.0).trailing_zeros() as usize;
            let p_b = (beta_i.0 & !beta_j.0).trailing_zeros() as usize;
            let q_b = (beta_j.0 & !beta_i.0).trailing_zeros() as usize;

            let min_a = p_a.min(q_a);
            let max_a = p_a.max(q_a);
            let mask_a = ((1u64 << max_a) - 1) & !((1u64 << (min_a + 1)) - 1);
            let ph_a = if (alpha_i.0 & mask_a).count_ones() % 2 == 1 { -1.0 } else { 1.0 };

            let min_b = p_b.min(q_b);
            let max_b = p_b.max(q_b);
            let mask_b = ((1u64 << max_b) - 1) & !((1u64 << (min_b + 1)) - 1);
            let ph_b = if (beta_i.0 & mask_b).count_ones() % 2 == 1 { -1.0 } else { 1.0 };

            ph_a * ph_b * mo_eri.get(p_a, q_a, p_b, q_b)
        }

        // Differ by 3 or more electrons: 0.0 by Slater-Condon rules
        _ => 0.0,
    }
}

/// Computes the explicit full Hamiltonian matrix over the determinant space.
pub fn compute_hamiltonian_matrix(
    space: &DeterminantSpace,
    h_mo: &Matrix,
    mo_eri: &MoEriTensor,
) -> Matrix {
    let n_det = space.n_determinants();
    let mut h = Matrix::zeros(n_det, n_det);

    let rows: Vec<Vec<f64>> = (0..n_det)
        .into_par_iter()
        .map(|i| {
            let (a_i, b_i) = space.linear_to_pair(i);
            let str_ai = &space.alpha_strings[a_i];
            let str_bi = &space.beta_strings[b_i];

            let mut row = vec![0.0; n_det];
            for j in 0..=i {
                let (a_j, b_j) = space.linear_to_pair(j);
                let str_aj = &space.alpha_strings[a_j];
                let str_bj = &space.beta_strings[b_j];

                let val = slater_condon_element(str_ai, str_bi, str_aj, str_bj, h_mo, mo_eri);
                row[j] = val;
            }
            row
        })
        .collect();

    for i in 0..n_det {
        for j in 0..=i {
            let val = rows[i][j];
            h[(i, j)] = val;
            h[(j, i)] = val;
        }
    }

    h
}

/// Evaluates direct Hamiltonian-vector product sigma = H * v in parallel with Rayon.
pub fn direct_h_matvec(
    space: &DeterminantSpace,
    h_mo: &Matrix,
    mo_eri: &MoEriTensor,
    v: &[f64],
) -> Vec<f64> {
    let n_det = space.n_determinants();
    assert_eq!(v.len(), n_det);

    (0..n_det)
        .into_par_iter()
        .map(|i| {
            let (a_i, b_i) = space.linear_to_pair(i);
            let str_ai = &space.alpha_strings[a_i];
            let str_bi = &space.beta_strings[b_i];

            let mut sum = 0.0;
            for (j, &v_j) in v.iter().enumerate() {
                if v_j.abs() < 1e-15 {
                    continue;
                }
                let (a_j, b_j) = space.linear_to_pair(j);
                let str_aj = &space.alpha_strings[a_j];
                let str_bj = &space.beta_strings[b_j];

                let h_ij = slater_condon_element(str_ai, str_bi, str_aj, str_bj, h_mo, mo_eri);
                sum += h_ij * v_j;
            }
            sum
        })
        .collect()
}
