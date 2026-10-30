// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Coupled-Perturbed Hartree-Fock (CPHF) solver for orbital response U^x.

use rayon::prelude::*;
use faer::prelude::Solve;
use crate::matrix::Matrix;
use crate::scf::{eri_index, HFResult};

pub use crate::mo_integrals::{MoEriTensor, transform_eri_ao_to_mo};


/// Represents the first-order orbital response matrices U^x for all 3*N nuclear perturbations.
#[derive(Clone, Debug)]
pub struct OrbitalResponse {
    /// U[perturbation_index] has shape [n_basis, n_basis] in MO basis
    pub u_matrices: Vec<Matrix>,
}

/// Solves the Coupled-Perturbed Hartree-Fock (CPHF) linear equations for all perturbations x.
/// (epsilon_a - epsilon_i) * U_{ai}^x + \sum_{bj} A_{ai, bj} * U_{bj}^x = B_{ai}^x
pub fn solve_cphf(
    hf: &HFResult,
    n_occ: usize,
    mo_skeleton_fock_derivs: &[Matrix], // F_{pq}^{(x)} in MO basis for each perturbation
    mo_overlap_derivs: &[Matrix],       // S_{pq}^{(x)} in MO basis for each perturbation
) -> OrbitalResponse {
    let n_basis = hf.mo_coefficients.cols;
    let n_virt = n_basis - n_occ;
    let n_cphf = n_virt * n_occ;
    let n_perts = mo_skeleton_fock_derivs.len();

    if n_cphf == 0 {
        return OrbitalResponse {
            u_matrices: vec![Matrix::zeros(n_basis, n_basis); n_perts],
        };
    }

    // Transform 2-electron integrals to MO basis in O(N^5)
    let c = &hf.mo_coefficients;
    let eri_ao = &hf.eri;
    let mo_eri = transform_eri_ao_to_mo(c, eri_ao);

    // Build CPHF coupling matrix in parallel with Rayon: A_{ai, bj} = 4*(ai|bj) - (ab|ji) - (aj|bi)
    let row_data: Vec<(usize, Vec<f64>)> = (0..n_cphf)
        .into_par_iter()
        .map(|row| {
            let a = row / n_occ;
            let i = row % n_occ;
            let a_mo = n_occ + a;
            let eps_diff = hf.orbital_energies[a_mo] - hf.orbital_energies[i];
            let mut row_vals = vec![0.0; n_cphf];
            row_vals[row] += eps_diff;

            for b in 0..n_virt {
                let b_mo = n_occ + b;
                for j in 0..n_occ {
                    let col = b * n_occ + j;
                    let val_aibj = mo_eri.get(a_mo, i, b_mo, j);
                    let val_abji = mo_eri.get(a_mo, b_mo, j, i);
                    let val_ajbi = mo_eri.get(a_mo, j, b_mo, i);
                    row_vals[col] += 4.0 * val_aibj - val_abji - val_ajbi;
                }
            }
            (row, row_vals)
        })
        .collect();

    let mut a_mat = Matrix::zeros(n_cphf, n_cphf);
    for (row, vals) in row_data {
        for (col, val) in vals.into_iter().enumerate() {
            a_mat[(row, col)] = val;
        }
    }

    // Build RHS B-vectors in parallel across perturbations
    let all_b: Vec<Vec<f64>> = (0..n_perts)
        .into_par_iter()
        .map(|pert| {
            let f_mo = &mo_skeleton_fock_derivs[pert];
            let s_mo = &mo_overlap_derivs[pert];
            let mut b_vec = vec![0.0; n_cphf];
            for a in 0..n_virt {
                let a_mo = n_occ + a;
                for i in 0..n_occ {
                    let idx = a * n_occ + i;
                    let mut b_val = -(f_mo[(a_mo, i)] - hf.orbital_energies[i] * s_mo[(a_mo, i)]);
                    for k in 0..n_occ {
                        for l in 0..n_occ {
                            let val_aikl = mo_eri.get(a_mo, i, k, l);
                            let val_akil = mo_eri.get(a_mo, k, i, l);
                            b_val += s_mo[(k, l)] * (2.0 * val_aikl - val_akil);
                        }
                    }
                    b_vec[idx] = b_val;
                }
            }
            b_vec
        })
        .collect();

    // Solve A * U = B: use faer simultaneous multi-RHS TRSM for n_cphf >= 16, or LU fallback
    let u_sols: Vec<Vec<f64>> = if n_cphf >= 16 {
        let a_faer = faer::MatRef::from_row_major_slice(&a_mat.data, n_cphf, n_cphf);
        let lu = a_faer.partial_piv_lu();
        let b_faer = faer::Mat::<f64>::from_fn(n_cphf, n_perts, |row, col| all_b[col][row]);
        let u_faer = lu.solve(&b_faer);
        (0..n_perts)
            .map(|col| (0..n_cphf).map(|row| u_faer[(row, col)]).collect())
            .collect()
    } else {
        let lu = a_mat.lu_decompose();
        all_b
            .into_iter()
            .map(|b_vec| lu.as_ref().map(|l| l.solve(&b_vec)).unwrap_or_else(|| vec![0.0; n_cphf]))
            .collect()
    };

    // Construct full U^x matrices in MO basis:
    // U_{ai} = sol, U_{ia} = - U_{ai} - S_{ia}
    // U_{ij} = - 0.5 * S_{ij}
    // U_{ab} = - 0.5 * S_{ab}
    let mut u_matrices = Vec::with_capacity(n_perts);

    for pert in 0..n_perts {
        let s_mo = &mo_overlap_derivs[pert];
        let u_ai_sol = &u_sols[pert];
        let mut u_mat = Matrix::zeros(n_basis, n_basis);

        // Occupied-Occupied
        for i in 0..n_occ {
            for j in 0..n_occ {
                u_mat[(i, j)] = -0.5 * s_mo[(i, j)];
            }
        }

        // Virtual-Virtual
        for a in 0..n_virt {
            let a_mo = n_occ + a;
            for b in 0..n_virt {
                let b_mo = n_occ + b;
                u_mat[(a_mo, b_mo)] = -0.5 * s_mo[(a_mo, b_mo)];
            }
        }

        // Occupied-Virtual
        for a in 0..n_virt {
            let a_mo = n_occ + a;
            for i in 0..n_occ {
                let idx = a * n_occ + i;
                let u_val = u_ai_sol[idx];
                u_mat[(a_mo, i)] = u_val;
                u_mat[(i, a_mo)] = -u_val - s_mo[(i, a_mo)];
            }
        }

        u_matrices.push(u_mat);
    }

    OrbitalResponse { u_matrices }
}

/// Helper to compute a single 4-index transformed MO ERI (p q | r s)
pub fn compute_mo_eri(
    c: &Matrix,
    eri_ao: &[f64],
    p: usize,
    q: usize,
    r: usize,
    s: usize,
) -> f64 {
    let n = c.rows;
    let mut sum = 0.0;

    for mu in 0..n {
        let cp = c[(mu, p)];
        if cp.abs() < 1e-12 { continue; }
        for nu in 0..n {
            let cp_cq = cp * c[(nu, q)];
            if cp_cq.abs() < 1e-12 { continue; }
            for lambda in 0..n {
                let cp_cq_cr = cp_cq * c[(lambda, r)];
                if cp_cq_cr.abs() < 1e-12 { continue; }
                for sigma in 0..n {
                    let coeff = cp_cq_cr * c[(sigma, s)];
                    if coeff.abs() < 1e-12 { continue; }
                    let idx = eri_index(mu, nu, lambda, sigma);
                    sum += coeff * eri_ao[idx];
                }
            }
        }
    }

    sum
}

/// Pre-factored CPHF solver for 1st- and 2nd-order orbital response equations.
pub struct CphfSolver {
    pub n_basis: usize,
    pub n_occ: usize,
    pub n_virt: usize,
    pub n_cphf: usize,
    pub lu: Option<crate::matrix::LUFactorization>,
}

impl CphfSolver {
    pub fn new(hf: &HFResult, n_occ: usize) -> Self {
        let n_basis = hf.mo_coefficients.cols;
        let n_virt = n_basis - n_occ;
        let n_cphf = n_virt * n_occ;

        if n_cphf == 0 {
            return Self {
                n_basis,
                n_occ,
                n_virt,
                n_cphf,
                lu: None,
            };
        }

        let c = &hf.mo_coefficients;
        let mo_eri = transform_eri_ao_to_mo(c, &hf.eri);
        let mut a_mat = Matrix::zeros(n_cphf, n_cphf);

        for a in 0..n_virt {
            let a_mo = n_occ + a;
            for i in 0..n_occ {
                let row = a * n_occ + i;
                let eps_diff = hf.orbital_energies[a_mo] - hf.orbital_energies[i];
                a_mat[(row, row)] += eps_diff;

                for b in 0..n_virt {
                    let b_mo = n_occ + b;
                    for j in 0..n_occ {
                        let col = b * n_occ + j;
                        let val_aibj = mo_eri.get(a_mo, i, b_mo, j);
                        let val_abji = mo_eri.get(a_mo, b_mo, j, i);
                        let val_ajbi = mo_eri.get(a_mo, j, b_mo, i);
                        let a_elem = 4.0 * val_aibj - val_abji - val_ajbi;
                        a_mat[(row, col)] += a_elem;
                    }
                }
            }
        }

        let lu = a_mat.lu_decompose();
        Self {
            n_basis,
            n_occ,
            n_virt,
            n_cphf,
            lu,
        }
    }

    /// Solves (A + B) x = rhs
    pub fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        if let Some(ref lu) = self.lu {
            lu.solve(rhs)
        } else {
            vec![0.0; rhs.len()]
        }
    }

    /// Solves for 2nd-order orbital response matrix U^{(2)} in MO basis:
    ///   (A + B)_{ai, bj} U_{bj}^{(2)} = B_{ai}^{(2)}
    /// with exact 2nd-order orthonormality conditions:
    ///   U_{ij}^{(2)} = -0.5 * M_{ij}^{(2)}
    ///   U_{ab}^{(2)} = -0.5 * M_{ab}^{(2)}
    ///   U_{ia}^{(2)} = - U_{ai}^{(2)} - M_{ia}^{(2)}
    /// where M^{(2)} = S^{(2)} + 2 U^{(1)T} S^{(1)} + 2 U^{(1)T} U^{(1)}.
    pub fn solve_second_order(
        &self,
        b2_ai: &[f64],
        s2_mo: &Matrix,
        u1_mo: &Matrix,
        s1_mo: &Matrix,
    ) -> Matrix {
        let n_basis = self.n_basis;
        let n_occ = self.n_occ;
        let n_virt = self.n_virt;

        // 1. Solve for virtual-occupied block U_{ai}^{(2)}
        let u2_ai = self.solve(b2_ai);

        // 2. Compute M^{(2)} = S^{(2)} + 2 U^{(1)T} S^{(1)} + 2 U^{(1)T} U^{(1)}
        let u1_t = u1_mo.transpose();
        let u1t_s1 = u1_t.dot(s1_mo);
        let u1t_u1 = u1_t.dot(u1_mo);

        let mut m2 = s2_mo.clone();
        for p in 0..n_basis {
            for q in 0..n_basis {
                m2[(p, q)] += 2.0 * (u1t_s1[(p, q)] + u1t_u1[(p, q)]);
            }
        }

        // 3. Assemble full U^{(2)} matrix
        let mut u2 = Matrix::zeros(n_basis, n_basis);

        // Occupied-occupied: -0.5 * M_{ij}
        for i in 0..n_occ {
            for j in 0..n_occ {
                u2[(i, j)] = -0.5 * m2[(i, j)];
            }
        }

        // Virtual-virtual: -0.5 * M_{ab}
        for a in 0..n_virt {
            let a_mo = n_occ + a;
            for b in 0..n_virt {
                let b_mo = n_occ + b;
                u2[(a_mo, b_mo)] = -0.5 * m2[(a_mo, b_mo)];
            }
        }

        // Virtual-occupied: solved from CPHF
        for a in 0..n_virt {
            let a_mo = n_occ + a;
            for i in 0..n_occ {
                let val = u2_ai[a * n_occ + i];
                u2[(a_mo, i)] = val;
                u2[(i, a_mo)] = -val - m2[(i, a_mo)];
            }
        }

        u2
    }
}

