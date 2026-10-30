// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! 1-particle and 2-particle reduced density matrices from CI wavefunctions.

use crate::fci::string::DeterminantSpace;
use crate::matrix::Matrix;

/// Computes the 1-particle reduced density matrix in MO basis:
/// gamma_{pq} = <Psi | a_p^\dagger a_q | Psi>
pub fn compute_1rdm(space: &DeterminantSpace, ci_vec: &[f64]) -> Matrix {
    let n_orb = space.n_orbitals;
    let mut rdm = Matrix::zeros(n_orb, n_orb);

    for (i, &c_i) in ci_vec.iter().enumerate() {
        if c_i.abs() < 1e-15 {
            continue;
        }
        let (a_i, b_i) = space.linear_to_pair(i);
        let str_ai = space.alpha_strings[a_i];
        let str_bi = space.beta_strings[b_i];

        // Diagonal contributions: occupied orbitals in determinant i
        for p in str_ai.occupied_orbitals() {
            rdm[(p, p)] += c_i * c_i;
        }
        for p in str_bi.occupied_orbitals() {
            rdm[(p, p)] += c_i * c_i;
        }

        // Off-diagonal alpha excitations: a_p^\dagger a_q |str_ai>
        for q in 0..n_orb {
            for p in 0..n_orb {
                if p == q {
                    continue;
                }
                let Some((str_aj, phase_a)) = str_ai.apply_single_excitation(p, q) else {
                    continue;
                };
                if let Ok(a_j) = space.alpha_strings.binary_search_by_key(&str_aj.0, |s| s.0) {
                    let j = space.pair_to_linear(a_j, b_i);
                    let c_j = ci_vec[j];
                    rdm[(p, q)] += c_i * c_j * phase_a;
                }
            }
        }

        // Off-diagonal beta excitations: a_p^\dagger a_q |str_bi>
        for q in 0..n_orb {
            for p in 0..n_orb {
                if p == q {
                    continue;
                }
                let Some((str_bj, phase_b)) = str_bi.apply_single_excitation(p, q) else {
                    continue;
                };
                if let Ok(b_j) = space.beta_strings.binary_search_by_key(&str_bj.0, |s| s.0) {
                    let j = space.pair_to_linear(a_i, b_j);
                    let c_j = ci_vec[j];
                    rdm[(p, q)] += c_i * c_j * phase_b;
                }
            }
        }
    }

    rdm
}

/// Computes the spin-summed 2-particle reduced density matrix in MO basis:
/// Gamma_{pqrs} = sum_{sigma tau} <Psi | a_{p sigma}^\dagger a_{r tau}^\dagger a_{s tau} a_{q sigma} | Psi>
/// Using the exact generator resolution in complete active/FCI space:
/// E_{pq} E_{rs} - delta_{qr} E_{ps} = sum_{sigma tau} a_{p sigma}^\dagger a_{r tau}^\dagger a_{s tau} a_{q sigma}.
pub fn compute_2rdm(space: &DeterminantSpace, ci_vec: &[f64]) -> Vec<f64> {
    let n_orb = space.n_orbitals;
    let n_det = space.n_determinants();
    let dim = n_orb * n_orb * n_orb * n_orb;
    let mut rdm2 = vec![0.0; dim];

    let d1 = compute_1rdm(space, ci_vec);

    // Compute w_{pq} = E_{pq} |Psi> for all p, q
    let mut w_vecs = vec![vec![0.0; n_det]; n_orb * n_orb];

    for j in 0..n_det {
        let c_j = ci_vec[j];
        if c_j.abs() < 1e-15 {
            continue;
        }
        let (a_j, b_j) = space.linear_to_pair(j);
        let str_aj = space.alpha_strings[a_j];
        let str_bj = space.beta_strings[b_j];

        // Alpha excitations: a_{p alpha}^\dagger a_{q alpha} |j>
        for q in 0..n_orb {
            for p in 0..n_orb {
                let Some((str_ak, phase_a)) = str_aj.apply_single_excitation(p, q) else {
                    continue;
                };
                if let Ok(a_k) = space.alpha_strings.binary_search_by_key(&str_ak.0, |s| s.0) {
                    let k = space.pair_to_linear(a_k, b_j);
                    w_vecs[p * n_orb + q][k] += c_j * phase_a;
                }
            }
        }

        // Beta excitations: a_{p beta}^\dagger a_{q beta} |j>
        for q in 0..n_orb {
            for p in 0..n_orb {
                let Some((str_bk, phase_b)) = str_bj.apply_single_excitation(p, q) else {
                    continue;
                };
                if let Ok(b_k) = space.beta_strings.binary_search_by_key(&str_bk.0, |s| s.0) {
                    let k = space.pair_to_linear(a_j, b_k);
                    w_vecs[p * n_orb + q][k] += c_j * phase_b;
                }
            }
        }
    }

    let idx4 = |p: usize, q: usize, r: usize, s: usize| -> usize {
        ((p * n_orb + q) * n_orb + r) * n_orb + s
    };

    // Gamma_{pqrs} = <E_{qp} Psi | E_{rs} Psi> - delta_{qr} D_{ps}
    for p in 0..n_orb {
        for q in 0..n_orb {
            let w_qp = &w_vecs[q * n_orb + p];
            for r in 0..n_orb {
                for s in 0..n_orb {
                    let w_rs = &w_vecs[r * n_orb + s];
                    let mut dot = 0.0;
                    for k in 0..n_det {
                        dot += w_qp[k] * w_rs[k];
                    }
                    if q == r {
                        dot -= d1[(p, s)];
                    }
                    rdm2[idx4(p, q, r, s)] = dot;
                }
            }
        }
    }

    rdm2
}

