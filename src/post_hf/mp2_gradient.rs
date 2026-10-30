// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Analytical Nuclear Gradient for Restricted Møller-Plesset Second-Order Perturbation Theory (RHF-MP2).
//!
//! Evaluates the exact analytical nuclear gradient using the Handy-Schaefer Z-vector method:
//! 1. Computes MP2 T2 amplitudes and unrelaxed 1-PDM and 2-PDM in MO basis.
//! 2. Builds the orbital response source vector X_{ai} from the derivative of the MP2 correlation energy.
//! 3. Solves the CPHF electronic orbital Hessian equation sum_{bj} A_{ai, bj} Z_{bj} = X_{ai} once for all nuclear coordinates.
//! 4. Constructs orbital-relaxed 1-PDM P_{relaxed}, relaxed energy-weighted density matrix W_{relaxed}, and AO 2-PDM.
//! 5. Evaluates exact analytical forces via contraction with 1e, overlap, and 2e integral derivatives.

use faer::prelude::*;
use rayon::prelude::*;
use crate::basis::{Atom, ContractedGaussian};
use crate::gradient::{differentiate_basis_function, deriv_overlap, deriv_kinetic, deriv_nuclear_attraction, deriv_nuclear_attraction_single_atom};
use crate::integrals::contracted_eri;
use crate::matrix::Matrix;
use crate::mo_integrals::transform_eri_ao_to_mo;
use crate::scf::{HFResult, UHFResult};

/// Computes the exact analytical nuclear gradient for closed-shell RHF-MP2.
pub fn compute_rhf_mp2_analytical_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
) -> Vec<[f64; 3]> {
    let n_atoms = atoms.len();
    let n_basis = basis.len();
    let n_occ = hf.n_occ;
    let n_virt = n_basis - n_occ;
    let n_cphf = n_virt * n_occ;

    if n_cphf == 0 {
        // No virtual orbitals: MP2 correlation is identically zero, falls back to RHF gradient
        return crate::gradient::compute_rhf_gradient(atoms, basis, hf);
    }

    let c_mo = &hf.mo_coefficients;
    let eps = &hf.orbital_energies;
    let mo_eri = transform_eri_ao_to_mo(c_mo, &hf.eri);

    // 1. Compute MP2 T2 amplitudes: t_{ij}^{ab} = (ia|jb) / (eps_i + eps_j - eps_a - eps_b)
    // and unrelaxed 1-PDM: D_{ij} (occ-occ) and D_{ab} (virt-virt)
    let mut t2 = vec![vec![vec![vec![0.0; n_virt]; n_virt]; n_occ]; n_occ];
    let mut d_occ = Matrix::zeros(n_occ, n_occ);
    let mut d_virt = Matrix::zeros(n_virt, n_virt);

    for i in 0..n_occ {
        for j in 0..n_occ {
            for a in 0..n_virt {
                let a_mo = n_occ + a;
                for b in 0..n_virt {
                    let b_mo = n_occ + b;
                    let denom = eps[i] + eps[j] - eps[a_mo] - eps[b_mo];
                    let iajb = mo_eri.get(i, a_mo, j, b_mo);
                    let t_val = iajb / denom;
                    t2[i][j][a][b] = t_val;
                }
            }
        }
    }

    // D_{ij} = - 2 * sum_{k, a, b} (2 t_{ik}^{ab} - t_{ik}^{ba}) t_{jk}^{ab}
    for i in 0..n_occ {
        for j in 0..n_occ {
            let mut sum_d = 0.0;
            for k in 0..n_occ {
                for a in 0..n_virt {
                    for b in 0..n_virt {
                        let t_ik = t2[i][k][a][b];
                        let t_ik_swap = t2[i][k][b][a];
                        let t_jk = t2[j][k][a][b];
                        sum_d -= 2.0 * (2.0 * t_ik - t_ik_swap) * t_jk;
                    }
                }
            }
            d_occ[(i, j)] = sum_d;
        }
    }

    // D_{ab} = 2 * sum_{i, j, c} (2 t_{ij}^{ac} - t_{ij}^{ca}) t_{ij}^{bc}
    for a in 0..n_virt {
        for b in 0..n_virt {
            let mut sum_d = 0.0;
            for i in 0..n_occ {
                for j in 0..n_occ {
                    for c in 0..n_virt {
                        let t_ac = t2[i][j][a][c];
                        let t_ca = t2[i][j][c][a];
                        let t_bc = t2[i][j][b][c];
                        sum_d += 2.0 * (2.0 * t_ac - t_ca) * t_bc;
                    }
                }
            }
            d_virt[(a, b)] = sum_d;
        }
    }

    // 2. Build CPHF coupling matrix A_{ai, bj} = (eps_a - eps_i) delta_{ab} delta_{ij} + 4(ai|bj) - (ab|ji) - (aj|bi)
    let mut a_mat = Matrix::zeros(n_cphf, n_cphf);
    for a in 0..n_virt {
        let a_mo = n_occ + a;
        for i in 0..n_occ {
            let row = a * n_occ + i;
            a_mat[(row, row)] += eps[a_mo] - eps[i];

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

    // 3. Build Z-vector right-hand side source vector X_{ai}:
    // X_{ai} = 2 * sum_{j, b, c} (2 t_{ij}^{bc} - t_{ij}^{cb}) (aj|bc)
    //        - 2 * sum_{j, k, b} (2 t_{jk}^{ab} - t_{jk}^{ba}) (jk|ib)
    let mut x_vec = vec![0.0; n_cphf];
    for a in 0..n_virt {
        let a_mo = n_occ + a;
        for i in 0..n_occ {
            let row = a * n_occ + i;
            let mut x_val = 0.0;

            for j in 0..n_occ {
                for b in 0..n_virt {
                    let b_mo = n_occ + b;
                    for c in 0..n_virt {
                        let c_mo = n_occ + c;
                        let t_direct = t2[i][j][b][c];
                        let t_swap = t2[i][j][c][b];
                        let abjc = mo_eri.get(a_mo, b_mo, j, c_mo);
                        x_val -= 2.0 * (2.0 * t_direct - t_swap) * abjc;
                    }
                }
            }

            for j in 0..n_occ {
                for k in 0..n_occ {
                    for b in 0..n_virt {
                        let b_mo = n_occ + b;
                        let t_direct = t2[j][k][a][b];
                        let t_swap = t2[j][k][b][a];
                        let jikb = mo_eri.get(j, i, k, b_mo);
                        x_val += 2.0 * (2.0 * t_direct - t_swap) * jikb;
                    }
                }
            }

            // Unrelaxed 1-PDM Fock-response contributions:
            // - sum_{jk} D_{jk} [ 4(ai|jk) - (aj|ik) - (ak|ij) ] - sum_{bc} D_{bc} [ 4(ai|bc) - (ab|ic) - (ac|ib) ]
            for j in 0..n_occ {
                for k in 0..n_occ {
                    let d_jk = d_occ[(j, k)];
                    if d_jk.abs() > 1e-15 {
                        let aijk = mo_eri.get(a_mo, i, j, k);
                        let ajik = mo_eri.get(a_mo, j, i, k);
                        let akij = mo_eri.get(a_mo, k, i, j);
                        x_val -= 0.5 * d_jk * (4.0 * aijk - ajik - akij);
                    }
                }
            }

            for b in 0..n_virt {
                let b_mo = n_occ + b;
                for c in 0..n_virt {
                    let c_mo = n_occ + c;
                    let d_bc = d_virt[(b, c)];
                    if d_bc.abs() > 1e-15 {
                        let aibc = mo_eri.get(a_mo, i, b_mo, c_mo);
                        let abic = mo_eri.get(a_mo, b_mo, i, c_mo);
                        let acib = mo_eri.get(a_mo, c_mo, i, b_mo);
                        x_val -= 0.5 * d_bc * (4.0 * aibc - abic - acib);
                    }
                }
            }

            x_vec[row] = x_val;
        }
    }

    // 4. Solve Z-vector equation: sum_{bj} A_{ai, bj} Z_{bj} = X_{ai}
    let lu = a_mat.lu_decompose().expect("CPHF orbital Hessian is invertible in MP2 Z-vector");
    let z_sol = lu.solve(&x_vec);

    let mut z_mat = Matrix::zeros(n_virt, n_occ);
    for a in 0..n_virt {
        for i in 0..n_occ {
            z_mat[(a, i)] = z_sol[a * n_occ + i];
        }
    }

    // 5. Construct correlation 1-PDM P^corr in MO basis:
    // P_{ij} = D_{ij}
    // P_{ab} = D_{ab}
    // P_{ai} = P_{ia} = Z_{ai}
    let mut p_corr_mo = Matrix::zeros(n_basis, n_basis);
    for i in 0..n_occ {
        for j in 0..n_occ {
            p_corr_mo[(i, j)] = d_occ[(i, j)];
        }
    }
    for a in 0..n_virt {
        let a_mo = n_occ + a;
        for b in 0..n_virt {
            let b_mo = n_occ + b;
            p_corr_mo[(a_mo, b_mo)] = d_virt[(a, b)];
        }
    }
    for a in 0..n_virt {
        let a_mo = n_occ + a;
        for i in 0..n_occ {
            let z_ai = z_mat[(a, i)];
            p_corr_mo[(a_mo, i)] = z_ai;
            p_corr_mo[(i, a_mo)] = z_ai;
        }
    }

    // Transform correlation 1-PDM to AO basis: P_corr_AO = C * P_corr_MO * C^T
    let p_corr_ao = c_mo.dot(&p_corr_mo).dot(&c_mo.transpose());
    let p_hf_ao = &hf.density_matrix;

    // 6. Construct correlation energy-weighted density matrix W^corr in MO basis:
    let mut w_corr_mo = Matrix::zeros(n_basis, n_basis);
    for i in 0..n_occ {
        for j in 0..n_occ {
            let mut sum_w = 0.5 * (eps[i] + eps[j]) * d_occ[(i, j)];
            for k in 0..n_occ {
                for a in 0..n_virt {
                    let a_mo = n_occ + a;
                    for b in 0..n_virt {
                        let b_mo = n_occ + b;
                        let t_dir = t2[i][k][a][b];
                        let t_swp = t2[i][k][b][a];
                        let jakb = mo_eri.get(j, a_mo, k, b_mo);
                        sum_w += 2.0 * (2.0 * t_dir - t_swp) * jakb;
                    }
                }
            }
            for p in 0..n_basis {
                for q in 0..n_basis {
                    let p_pq = p_corr_mo[(p, q)];
                    if p_pq.abs() > 1e-15 {
                        let ijpq = mo_eri.get(i, j, p, q);
                        let ipjq = mo_eri.get(i, p, j, q);
                        sum_w += p_pq * (2.0 * ijpq - ipjq);
                    }
                }
            }
            w_corr_mo[(i, j)] = sum_w;
        }
    }

    for a in 0..n_virt {
        let a_mo = n_occ + a;
        for b in 0..n_virt {
            let b_mo = n_occ + b;
            let mut sum_w = 0.5 * (eps[a_mo] + eps[b_mo]) * d_virt[(a, b)];
            for i in 0..n_occ {
                for j in 0..n_occ {
                    for c in 0..n_virt {
                        let c_mo = n_occ + c;
                        let t_dir = t2[i][j][a][c];
                        let t_swp = t2[i][j][c][a];
                        let ibjc = mo_eri.get(i, b_mo, j, c_mo);
                        sum_w += 2.0 * (2.0 * t_dir - t_swp) * ibjc;
                    }
                }
            }
            w_corr_mo[(a_mo, b_mo)] = sum_w;
        }
    }

    for a in 0..n_virt {
        let a_mo = n_occ + a;
        for i in 0..n_occ {
            let z_ai = z_mat[(a, i)];
            let mut sum_ia = 0.0;
            for j in 0..n_occ {
                for k in 0..n_occ {
                    for b in 0..n_virt {
                        let b_mo = n_occ + b;
                        let t_dir = t2[j][k][a][b];
                        let t_swp = t2[j][k][b][a];
                        let jikb = mo_eri.get(j, i, k, b_mo);
                        sum_ia += 2.0 * (2.0 * t_dir - t_swp) * jikb;
                    }
                }
            }
            let w_val = eps[i] * z_ai + sum_ia;
            w_corr_mo[(i, a_mo)] = w_val;
            w_corr_mo[(a_mo, i)] = w_val;
        }
    }

    // Symmetrize W^corr: W_symm = 0.5 * (W + W^T)
    let mut w_symm_mo = Matrix::zeros(n_basis, n_basis);
    for p in 0..n_basis {
        for q in 0..n_basis {
            w_symm_mo[(p, q)] = 0.5 * (w_corr_mo[(p, q)] + w_corr_mo[(q, p)]);
        }
    }
    let w_corr_ao = c_mo.dot(&w_symm_mo).dot(&c_mo.transpose());

    // 7. Construct correlated 2-PDM in AO basis:
    // Full 4-fold contraction with (mu' nu | la si)
    let mut gamma_ao = vec![vec![vec![vec![0.0; n_basis]; n_basis]; n_basis]; n_basis];
    for i in 0..n_occ {
        for j in 0..n_occ {
            for a in 0..n_virt {
                let a_mo = n_occ + a;
                for b in 0..n_virt {
                    let b_mo = n_occ + b;
                    let val = 2.0 * (2.0 * t2[i][j][a][b] - t2[i][j][b][a]);
                    if val.abs() < 1e-14 {
                        continue;
                    }
                    for mu in 0..n_basis {
                        for nu in 0..n_basis {
                            for la in 0..n_basis {
                                for si in 0..n_basis {
                                    let t1 = c_mo[(mu, i)] * c_mo[(nu, a_mo)] * c_mo[(la, j)] * c_mo[(si, b_mo)];
                                    let t2 = c_mo[(nu, i)] * c_mo[(mu, a_mo)] * c_mo[(la, j)] * c_mo[(si, b_mo)];
                                    let t3 = c_mo[(la, i)] * c_mo[(si, a_mo)] * c_mo[(mu, j)] * c_mo[(nu, b_mo)];
                                    let t4 = c_mo[(la, i)] * c_mo[(si, a_mo)] * c_mo[(nu, j)] * c_mo[(mu, b_mo)];
                                    gamma_ao[mu][nu][la][si] += val * (t1 + t2 + t3 + t4);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 8. Contract with derivative integrals
    // First, start with the exact RHF reference gradient:
    let g_hf = crate::gradient::compute_rhf_gradient(atoms, basis, hf);

    let mut basis_derivs = Vec::with_capacity(n_basis);
    for cg in basis {
        let dx = differentiate_basis_function(cg, 0);
        let dy = differentiate_basis_function(cg, 1);
        let dz = differentiate_basis_function(cg, 2);
        basis_derivs.push([dx, dy, dz]);
    }

    (0..n_atoms)
        .into_par_iter()
        .map(|atom_idx| {
            let mut grad_a = g_hf[atom_idx];

            for tau in 0..3 {
                let mut d_s_val = 0.0;
                let mut d_t_val = 0.0;
                let mut d_v_val = 0.0;

                for mu in 0..n_basis {
                    let mu_on_a = basis[mu].atom_index == atom_idx;
                    for nu in 0..n_basis {
                        let nu_on_a = basis[nu].atom_index == atom_idx;
                        let p_munu = p_corr_ao[(mu, nu)];
                        let w_munu = w_corr_ao[(mu, nu)];

                        if p_munu.abs() < 1e-15 && w_munu.abs() < 1e-15 {
                            continue;
                        }

                        let mut ds = 0.0;
                        if mu_on_a {
                            ds += deriv_overlap(&basis_derivs[mu][tau], &basis[nu]);
                        }
                        if nu_on_a {
                            ds += deriv_overlap(&basis_derivs[nu][tau], &basis[mu]);
                        }
                        d_s_val += w_munu * ds;

                        let mut dt = 0.0;
                        if mu_on_a {
                            dt += deriv_kinetic(&basis_derivs[mu][tau], &basis[nu]);
                        }
                        if nu_on_a {
                            dt += deriv_kinetic(&basis_derivs[nu][tau], &basis[mu]);
                        }
                        d_t_val += p_munu * dt;

                        let mut dv = 0.0;
                        if mu_on_a {
                            dv += deriv_nuclear_attraction(&basis_derivs[mu][tau], &basis[nu], atoms);
                        }
                        if nu_on_a {
                            dv += deriv_nuclear_attraction(&basis_derivs[nu][tau], &basis[mu], atoms);
                        }

                        let target_atom = &atoms[atom_idx];
                        let hf_term = -(
                            deriv_nuclear_attraction_single_atom(&basis_derivs[mu][tau], &basis[nu], target_atom)
                            + deriv_nuclear_attraction_single_atom(&basis_derivs[nu][tau], &basis[mu], target_atom)
                        );
                        dv += hf_term;

                        d_v_val += p_munu * dv;
                    }
                }

                grad_a[tau] += (d_t_val + d_v_val) - d_s_val;

                // 2-electron integral derivative contraction:
                // Relaxed mean-field part + correlated 2-PDM part
                let mut d_eri_relax = 0.0;
                let mut d_eri_2pdm = 0.0;
                for mu in 0..n_basis {
                    if basis[mu].atom_index != atom_idx {
                        continue;
                    }

                    for term in &basis_derivs[mu][tau].terms {
                        for nu in 0..n_basis {
                            for lambda in 0..n_basis {
                                for sigma in 0..n_basis {
                                    // Linear relaxation coupling with HF reference density:
                                    let gamma_relax = 2.0 * (
                                        p_hf_ao[(mu, nu)] * p_corr_ao[(lambda, sigma)]
                                        + p_corr_ao[(mu, nu)] * p_hf_ao[(lambda, sigma)]
                                    ) - (
                                        p_hf_ao[(mu, lambda)] * p_corr_ao[(nu, sigma)]
                                        + p_corr_ao[(mu, lambda)] * p_hf_ao[(nu, sigma)]
                                    );

                                    // Correlated 2-PDM contraction:
                                    let gamma_2pdm = gamma_ao[mu][nu][lambda][sigma];

                                    let eri_val = contracted_eri(term, &basis[nu], &basis[lambda], &basis[sigma]);
                                    if gamma_relax.abs() > 1e-15 {
                                        d_eri_relax += gamma_relax * eri_val;
                                    }
                                    if gamma_2pdm.abs() > 1e-15 {
                                        d_eri_2pdm += gamma_2pdm * eri_val;
                                    }
                                }
                            }
                        }
                    }
                }

                grad_a[tau] += d_eri_relax + d_eri_2pdm;
            }

            grad_a
        })
        .collect()
}

/// Cross-spin molecular orbital 2-electron repulsion integrals: `(p_alpha q_alpha | r_beta s_beta)`.
#[derive(Clone, Debug)]
pub struct CrossMoEri {
    pub n_basis: usize,
    pub n_pairs: usize,
    pub data: Vec<f64>,
}

impl CrossMoEri {
    #[inline(always)]
    pub fn get(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let pq = if p >= q { p * (p + 1) / 2 + q } else { q * (q + 1) / 2 + p };
        let rs = if r >= s { r * (r + 1) / 2 + s } else { s * (s + 1) / 2 + r };
        self.data[rs * self.n_pairs + pq]
    }
}

/// Transforms 2-electron repulsion integrals to cross-spin MO basis: `(p_alpha q_alpha | r_beta s_beta)`.
/// Uses faer matrix multiplications and Rayon parallelism across pairs.
pub fn transform_cross_eri_ao_to_mo(
    c_a: &Matrix,
    c_b: &Matrix,
    eri_ao: &[f64],
) -> CrossMoEri {
    let n = c_a.rows;
    let n_pairs = n * (n + 1) / 2;

    let pair_to_ao: Vec<(usize, usize)> = (0..n)
        .flat_map(|mu| (0..=mu).map(move |nu| (mu, nu)))
        .collect();

    // Step 1: Half-transform electron 2 with C_beta for each AO pair (mu, nu) with mu >= nu
    let half_eri_b: Vec<f64> = pair_to_ao
        .par_iter()
        .flat_map(|&(mu, nu)| {
            let mut a_mat = vec![0.0; n * n];
            for lambda in 0..n {
                for sigma in 0..=lambda {
                    let val = eri_ao[crate::scf::eri_index(mu, nu, lambda, sigma)];
                    a_mat[lambda * n + sigma] = val;
                    a_mat[sigma * n + lambda] = val;
                }
            }
            let a_faer = faer::Mat::<f64>::from_fn(n, n, |i, j| a_mat[i * n + j]);
            let cb_faer = faer::Mat::<f64>::from_fn(n, c_b.cols, |i, j| c_b[(i, j)]);
            let b_faer = cb_faer.transpose() * (&a_faer * &cb_faer);

            let mut row = Vec::with_capacity(n_pairs);
            for r in 0..c_b.cols {
                for s in 0..=r {
                    row.push(b_faer[(r, s)]);
                }
            }
            row
        })
        .collect();

    // Step 2: Transform electron 1 with C_alpha for each MO pair (r, s) of beta
    let mo_pairs_b: Vec<(usize, usize)> = (0..c_b.cols)
        .flat_map(|r| (0..=r).map(move |s| (r, s)))
        .collect();

    let data: Vec<f64> = mo_pairs_b
        .par_iter()
        .enumerate()
        .flat_map(|(rs_idx, _)| {
            let mut m_mat = vec![0.0; n * n];
            for (ij_ao, &(mu, nu)) in pair_to_ao.iter().enumerate() {
                let val = half_eri_b[ij_ao * n_pairs + rs_idx];
                m_mat[mu * n + nu] = val;
                m_mat[nu * n + mu] = val;
            }
            let m_faer = faer::Mat::<f64>::from_fn(n, n, |i, j| m_mat[i * n + j]);
            let ca_faer = faer::Mat::<f64>::from_fn(n, c_a.cols, |i, j| c_a[(i, j)]);
            let g_faer = ca_faer.transpose() * (&m_faer * &ca_faer);

            let mut row = Vec::with_capacity(n_pairs);
            for p in 0..c_a.cols {
                for q in 0..=p {
                    row.push(g_faer[(p, q)]);
                }
            }
            row
        })
        .collect();

    CrossMoEri {
        n_basis: n,
        n_pairs,
        data,
    }
}

/// Computes the exact analytical nuclear gradient for open-shell UHF-MP2.
///
/// Uses the coupled-perturbed UHF (CPUHF) Handy-Schaefer Z-vector method,
/// constructing orbital-relaxed alpha and beta 1-PDMs, relaxed energy-weighted
/// density matrices, and correlated 2-PDM, contracting with AO 1e, overlap,
/// and 2e integral derivatives.
pub fn compute_uhf_mp2_analytical_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    uhf: &UHFResult,
) -> Vec<[f64; 3]> {
    let n_atoms = atoms.len();
    let n_basis = basis.len();
    let n_alpha = uhf.n_alpha;
    let n_beta = uhf.n_beta;
    let n_virt_a = n_basis - n_alpha;
    let n_virt_b = n_basis - n_beta;
    let n_cphf_a = n_virt_a * n_alpha;
    let n_cphf_b = n_virt_b * n_beta;
    let n_cphf = n_cphf_a + n_cphf_b;

    if n_cphf == 0 {
        return crate::gradient::compute_uhf_gradient(atoms, basis, uhf);
    }

    let c_a = &uhf.mo_coefficients_alpha;
    let c_b = &uhf.mo_coefficients_beta;
    let eps_a = &uhf.orbital_energies_alpha;
    let eps_b = &uhf.orbital_energies_beta;

    // Transform 2-electron integrals to MO bases using faer and Rayon
    let mo_eri_a = transform_eri_ao_to_mo(c_a, &uhf.eri);
    let mo_eri_b = transform_eri_ao_to_mo(c_b, &uhf.eri);
    let cross_eri = transform_cross_eri_ao_to_mo(c_a, c_b, &uhf.eri);

    // 1. Compute MP2 T2 amplitudes:
    // aa block: t_{ij}^{ab(aa)} = (<ia||jb>_a) / denom
    let mut t2_aa = vec![vec![vec![vec![0.0; n_virt_a]; n_virt_a]; n_alpha]; n_alpha];
    for i in 0..n_alpha {
        for j in 0..n_alpha {
            for a in 0..n_virt_a {
                let a_mo = n_alpha + a;
                for b in 0..n_virt_a {
                    let b_mo = n_alpha + b;
                    let denom = eps_a[i] + eps_a[j] - eps_a[a_mo] - eps_a[b_mo];
                    let iajb = mo_eri_a.get(i, a_mo, j, b_mo);
                    let ibja = mo_eri_a.get(i, b_mo, j, a_mo);
                    t2_aa[i][j][a][b] = (iajb - ibja) / denom;
                }
            }
        }
    }

    // bb block: t_{ij}^{ab(bb)} = (<ia||jb>_b) / denom
    let mut t2_bb = vec![vec![vec![vec![0.0; n_virt_b]; n_virt_b]; n_beta]; n_beta];
    for i in 0..n_beta {
        for j in 0..n_beta {
            for a in 0..n_virt_b {
                let a_mo = n_beta + a;
                for b in 0..n_virt_b {
                    let b_mo = n_beta + b;
                    let denom = eps_b[i] + eps_b[j] - eps_b[a_mo] - eps_b[b_mo];
                    let iajb = mo_eri_b.get(i, a_mo, j, b_mo);
                    let ibja = mo_eri_b.get(i, b_mo, j, a_mo);
                    t2_bb[i][j][a][b] = (iajb - ibja) / denom;
                }
            }
        }
    }

    // ab block: t_{ij}^{ab(ab)} = (i_a a_a | j_b b_b) / denom
    let mut t2_ab = vec![vec![vec![vec![0.0; n_virt_b]; n_virt_a]; n_beta]; n_alpha];
    for i in 0..n_alpha {
        for j in 0..n_beta {
            for a in 0..n_virt_a {
                let a_mo = n_alpha + a;
                for b in 0..n_virt_b {
                    let b_mo = n_beta + b;
                    let denom = eps_a[i] + eps_b[j] - eps_a[a_mo] - eps_b[b_mo];
                    let iajb = cross_eri.get(i, a_mo, j, b_mo);
                    t2_ab[i][j][a][b] = iajb / denom;
                }
            }
        }
    }

    // 2. Unrelaxed 1-PDMs in MO bases:
    let mut d_occ_a = Matrix::zeros(n_alpha, n_alpha);
    for i in 0..n_alpha {
        for k in 0..n_alpha {
            let mut sum = 0.0;
            for j in 0..n_alpha {
                for a in 0..n_virt_a {
                    for b in 0..n_virt_a {
                        sum -= 0.5 * t2_aa[i][j][a][b] * t2_aa[k][j][a][b];
                    }
                }
            }
            for j in 0..n_beta {
                for a in 0..n_virt_a {
                    for b in 0..n_virt_b {
                        sum -= t2_ab[i][j][a][b] * t2_ab[k][j][a][b];
                    }
                }
            }
            d_occ_a[(i, k)] = sum;
        }
    }

    let mut d_virt_a = Matrix::zeros(n_virt_a, n_virt_a);
    for a in 0..n_virt_a {
        for c in 0..n_virt_a {
            let mut sum = 0.0;
            for i in 0..n_alpha {
                for j in 0..n_alpha {
                    for b in 0..n_virt_a {
                        sum += 0.5 * t2_aa[i][j][a][b] * t2_aa[i][j][c][b];
                    }
                }
            }
            for i in 0..n_alpha {
                for j in 0..n_beta {
                    for b in 0..n_virt_b {
                        sum += t2_ab[i][j][a][b] * t2_ab[i][j][c][b];
                    }
                }
            }
            d_virt_a[(a, c)] = sum;
        }
    }

    let mut d_occ_b = Matrix::zeros(n_beta, n_beta);
    for i in 0..n_beta {
        for k in 0..n_beta {
            let mut sum = 0.0;
            for j in 0..n_beta {
                for a in 0..n_virt_b {
                    for b in 0..n_virt_b {
                        sum -= 0.5 * t2_bb[i][j][a][b] * t2_bb[k][j][a][b];
                    }
                }
            }
            for j in 0..n_alpha {
                for a in 0..n_virt_a {
                    for b in 0..n_virt_b {
                        sum -= t2_ab[j][i][a][b] * t2_ab[j][k][a][b];
                    }
                }
            }
            d_occ_b[(i, k)] = sum;
        }
    }

    let mut d_virt_b = Matrix::zeros(n_virt_b, n_virt_b);
    for b in 0..n_virt_b {
        for d in 0..n_virt_b {
            let mut sum = 0.0;
            for i in 0..n_beta {
                for j in 0..n_beta {
                    for a in 0..n_virt_b {
                        sum += 0.5 * t2_bb[i][j][a][b] * t2_bb[i][j][a][d];
                    }
                }
            }
            for i in 0..n_alpha {
                for j in 0..n_beta {
                    for a in 0..n_virt_a {
                        sum += t2_ab[i][j][a][b] * t2_ab[i][j][a][d];
                    }
                }
            }
            d_virt_b[(b, d)] = sum;
        }
    }

    // 3. Build CPUHF Coupling Matrix A and Z-vector RHS vector X
    let mut a_mat = Matrix::zeros(n_cphf, n_cphf);
    let mut x_vec = vec![0.0; n_cphf];

    // Block A^{aa}
    for a in 0..n_virt_a {
        let a_mo = n_alpha + a;
        for i in 0..n_alpha {
            let row = a * n_alpha + i;
            a_mat[(row, row)] += eps_a[a_mo] - eps_a[i];

            for b in 0..n_virt_a {
                let b_mo = n_alpha + b;
                for j in 0..n_alpha {
                    let col = b * n_alpha + j;
                    let val_aibj = mo_eri_a.get(a_mo, i, b_mo, j);
                    let val_abji = mo_eri_a.get(a_mo, b_mo, j, i);
                    let val_ajbi = mo_eri_a.get(a_mo, j, b_mo, i);
                    a_mat[(row, col)] += 2.0 * val_aibj - val_abji - val_ajbi;
                }
            }
        }
    }

    // Block A^{bb}
    for a in 0..n_virt_b {
        let a_mo = n_beta + a;
        for i in 0..n_beta {
            let row = n_cphf_a + a * n_beta + i;
            a_mat[(row, row)] += eps_b[a_mo] - eps_b[i];

            for b in 0..n_virt_b {
                let b_mo = n_beta + b;
                for j in 0..n_beta {
                    let col = n_cphf_a + b * n_beta + j;
                    let val_aibj = mo_eri_b.get(a_mo, i, b_mo, j);
                    let val_abji = mo_eri_b.get(a_mo, b_mo, j, i);
                    let val_ajbi = mo_eri_b.get(a_mo, j, b_mo, i);
                    a_mat[(row, col)] += 2.0 * val_aibj - val_abji - val_ajbi;
                }
            }
        }
    }

    // Cross blocks A^{ab} and A^{ba}
    for a in 0..n_virt_a {
        let a_mo = n_alpha + a;
        for i in 0..n_alpha {
            let row = a * n_alpha + i;
            for b in 0..n_virt_b {
                let b_mo = n_beta + b;
                for j in 0..n_beta {
                    let col = n_cphf_a + b * n_beta + j;
                    let val_aibj = cross_eri.get(a_mo, i, b_mo, j);
                    let a_elem = 2.0 * val_aibj;
                    a_mat[(row, col)] += a_elem;
                    a_mat[(col, row)] += a_elem;
                }
            }
        }
    }

    // RHS vector X^alpha
    for a in 0..n_virt_a {
        let a_mo = n_alpha + a;
        for i in 0..n_alpha {
            let row = a * n_alpha + i;
            let mut x_val = 0.0;

            // aa 2-PDM response
            for j in 0..n_alpha {
                for b in 0..n_virt_a {
                    let b_mo = n_alpha + b;
                    for c in 0..n_virt_a {
                        let c_mo = n_alpha + c;
                        let t = t2_aa[i][j][b][c];
                        let abjc = mo_eri_a.get(a_mo, b_mo, j, c_mo);
                        x_val -= t * abjc;
                    }
                }
            }
            for j in 0..n_alpha {
                for k in 0..n_alpha {
                    for b in 0..n_virt_a {
                        let b_mo = n_alpha + b;
                        let t = t2_aa[j][k][a][b];
                        let jikb = mo_eri_a.get(j, i, k, b_mo);
                        x_val += t * jikb;
                    }
                }
            }

            // ab 2-PDM response
            for j in 0..n_beta {
                for b in 0..n_virt_b {
                    let b_mo = n_beta + b;
                    for c in 0..n_virt_a {
                        let c_mo = n_alpha + c;
                        let t = t2_ab[i][j][c][b];
                        let acjb = cross_eri.get(a_mo, c_mo, j, b_mo);
                        x_val -= t * acjb;
                    }
                }
            }
            for k in 0..n_alpha {
                for j in 0..n_beta {
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        let t = t2_ab[k][j][a][b];
                        let kijb = cross_eri.get(k, i, j, b_mo);
                        x_val += t * kijb;
                    }
                }
            }

            // Unrelaxed 1-PDM alpha contributions
            for j in 0..n_alpha {
                for k in 0..n_alpha {
                    let d_jk = d_occ_a[(j, k)];
                    if d_jk.abs() > 1e-15 {
                        let aijk = mo_eri_a.get(a_mo, i, j, k);
                        let ajik = mo_eri_a.get(a_mo, j, i, k);
                        let akij = mo_eri_a.get(a_mo, k, i, j);
                        x_val -= 0.5 * d_jk * (2.0 * aijk - ajik - akij);
                    }
                }
            }
            for b in 0..n_virt_a {
                let b_mo = n_alpha + b;
                for c in 0..n_virt_a {
                    let c_mo = n_alpha + c;
                    let d_bc = d_virt_a[(b, c)];
                    if d_bc.abs() > 1e-15 {
                        let aibc = mo_eri_a.get(a_mo, i, b_mo, c_mo);
                        let abic = mo_eri_a.get(a_mo, b_mo, i, c_mo);
                        let acib = mo_eri_a.get(a_mo, c_mo, i, b_mo);
                        x_val -= 0.5 * d_bc * (2.0 * aibc - abic - acib);
                    }
                }
            }

            // Unrelaxed 1-PDM beta contributions
            for j in 0..n_beta {
                for k in 0..n_beta {
                    let d_jk = d_occ_b[(j, k)];
                    if d_jk.abs() > 1e-15 {
                        let aijk = cross_eri.get(a_mo, i, j, k);
                        x_val -= d_jk * aijk;
                    }
                }
            }
            for b in 0..n_virt_b {
                let b_mo = n_beta + b;
                for c in 0..n_virt_b {
                    let c_mo = n_beta + c;
                    let d_bc = d_virt_b[(b, c)];
                    if d_bc.abs() > 1e-15 {
                        let aibc = cross_eri.get(a_mo, i, b_mo, c_mo);
                        x_val -= d_bc * aibc;
                    }
                }
            }

            x_vec[row] = x_val;
        }
    }

    // RHS vector X^beta
    for a in 0..n_virt_b {
        let a_mo = n_beta + a;
        for i in 0..n_beta {
            let row = n_cphf_a + a * n_beta + i;
            let mut x_val = 0.0;

            // bb 2-PDM response
            for j in 0..n_beta {
                for b in 0..n_virt_b {
                    let b_mo = n_beta + b;
                    for c in 0..n_virt_b {
                        let c_mo = n_beta + c;
                        let t = t2_bb[i][j][b][c];
                        let abjc = mo_eri_b.get(a_mo, b_mo, j, c_mo);
                        x_val -= t * abjc;
                    }
                }
            }
            for j in 0..n_beta {
                for k in 0..n_beta {
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        let t = t2_bb[j][k][a][b];
                        let jikb = mo_eri_b.get(j, i, k, b_mo);
                        x_val += t * jikb;
                    }
                }
            }

            // ab 2-PDM response (transposed electron 1 and 2)
            for j in 0..n_alpha {
                for b in 0..n_virt_a {
                    let b_mo = n_alpha + b;
                    for c in 0..n_virt_b {
                        let c_mo = n_beta + c;
                        let t = t2_ab[j][i][b][c];
                        let jbac = cross_eri.get(j, b_mo, a_mo, c_mo);
                        x_val -= t * jbac;
                    }
                }
            }
            for j in 0..n_alpha {
                for b in 0..n_virt_a {
                    let b_mo = n_alpha + b;
                    for k in 0..n_beta {
                        let t = t2_ab[j][k][b][a];
                        let jbki = cross_eri.get(j, b_mo, k, i);
                        x_val += t * jbki;
                    }
                }
            }

            // Unrelaxed 1-PDM beta contributions
            for j in 0..n_beta {
                for k in 0..n_beta {
                    let d_jk = d_occ_b[(j, k)];
                    if d_jk.abs() > 1e-15 {
                        let aijk = mo_eri_b.get(a_mo, i, j, k);
                        let ajik = mo_eri_b.get(a_mo, j, i, k);
                        let akij = mo_eri_b.get(a_mo, k, i, j);
                        x_val -= 0.5 * d_jk * (2.0 * aijk - ajik - akij);
                    }
                }
            }
            for b in 0..n_virt_b {
                let b_mo = n_beta + b;
                for c in 0..n_virt_b {
                    let c_mo = n_beta + c;
                    let d_bc = d_virt_b[(b, c)];
                    if d_bc.abs() > 1e-15 {
                        let aibc = mo_eri_b.get(a_mo, i, b_mo, c_mo);
                        let abic = mo_eri_b.get(a_mo, b_mo, i, c_mo);
                        let acib = mo_eri_b.get(a_mo, c_mo, i, b_mo);
                        x_val -= 0.5 * d_bc * (2.0 * aibc - abic - acib);
                    }
                }
            }

            // Unrelaxed 1-PDM alpha contributions
            for j in 0..n_alpha {
                for k in 0..n_alpha {
                    let d_jk = d_occ_a[(j, k)];
                    if d_jk.abs() > 1e-15 {
                        let jkai = cross_eri.get(j, k, a_mo, i);
                        x_val -= d_jk * jkai;
                    }
                }
            }
            for b in 0..n_virt_a {
                let b_mo = n_alpha + b;
                for c in 0..n_virt_a {
                    let c_mo = n_alpha + c;
                    let d_bc = d_virt_a[(b, c)];
                    if d_bc.abs() > 1e-15 {
                        let bcai = cross_eri.get(b_mo, c_mo, a_mo, i);
                        x_val -= d_bc * bcai;
                    }
                }
            }

            x_vec[row] = x_val;
        }
    }

    // 4. Solve Z-vector equation using faer
    let mat_a = faer::Mat::<f64>::from_fn(n_cphf, n_cphf, |r, c| a_mat[(r, c)]);
    let vec_x = faer::Mat::<f64>::from_fn(n_cphf, 1, |r, _| x_vec[r]);
    let lu = mat_a.partial_piv_lu();
    let sol_z = lu.solve(&vec_x);

    let mut z_mat_a = Matrix::zeros(n_virt_a, n_alpha);
    for a in 0..n_virt_a {
        for i in 0..n_alpha {
            z_mat_a[(a, i)] = sol_z[(a * n_alpha + i, 0)];
        }
    }

    let mut z_mat_b = Matrix::zeros(n_virt_b, n_beta);
    for a in 0..n_virt_b {
        for i in 0..n_beta {
            z_mat_b[(a, i)] = sol_z[(n_cphf_a + a * n_beta + i, 0)];
        }
    }

    // 5. Construct correlation 1-PDM P^corr in MO basis:
    let mut p_corr_mo_a = Matrix::zeros(n_basis, n_basis);
    for i in 0..n_alpha {
        for j in 0..n_alpha {
            p_corr_mo_a[(i, j)] = d_occ_a[(i, j)];
        }
    }
    for a in 0..n_virt_a {
        let a_mo = n_alpha + a;
        for b in 0..n_virt_a {
            let b_mo = n_alpha + b;
            p_corr_mo_a[(a_mo, b_mo)] = d_virt_a[(a, b)];
        }
    }
    for a in 0..n_virt_a {
        let a_mo = n_alpha + a;
        for i in 0..n_alpha {
            let z_ai = z_mat_a[(a, i)];
            p_corr_mo_a[(a_mo, i)] = z_ai;
            p_corr_mo_a[(i, a_mo)] = z_ai;
        }
    }

    let mut p_corr_mo_b = Matrix::zeros(n_basis, n_basis);
    for i in 0..n_beta {
        for j in 0..n_beta {
            p_corr_mo_b[(i, j)] = d_occ_b[(i, j)];
        }
    }
    for a in 0..n_virt_b {
        let a_mo = n_beta + a;
        for b in 0..n_virt_b {
            let b_mo = n_beta + b;
            p_corr_mo_b[(a_mo, b_mo)] = d_virt_b[(a, b)];
        }
    }
    for a in 0..n_virt_b {
        let a_mo = n_beta + a;
        for i in 0..n_beta {
            let z_ai = z_mat_b[(a, i)];
            p_corr_mo_b[(a_mo, i)] = z_ai;
            p_corr_mo_b[(i, a_mo)] = z_ai;
        }
    }

    // Transform correlation 1-PDMs to AO basis using faer
    let ca_faer = faer::Mat::<f64>::from_fn(n_basis, n_basis, |i, j| c_a[(i, j)]);
    let cb_faer = faer::Mat::<f64>::from_fn(n_basis, n_basis, |i, j| c_b[(i, j)]);
    let p_faer_a = faer::Mat::<f64>::from_fn(n_basis, n_basis, |i, j| p_corr_mo_a[(i, j)]);
    let p_faer_b = faer::Mat::<f64>::from_fn(n_basis, n_basis, |i, j| p_corr_mo_b[(i, j)]);

    let p_ao_a = &ca_faer * (&p_faer_a * ca_faer.transpose());
    let p_ao_b = &cb_faer * (&p_faer_b * cb_faer.transpose());

    let mut p_corr_ao_a = Matrix::zeros(n_basis, n_basis);
    let mut p_corr_ao_b = Matrix::zeros(n_basis, n_basis);
    for i in 0..n_basis {
        for j in 0..n_basis {
            p_corr_ao_a[(i, j)] = p_ao_a[(i, j)];
            p_corr_ao_b[(i, j)] = p_ao_b[(i, j)];
        }
    }

    // 6. Construct correlation energy-weighted density matrices W^corr in MO basis:
    let mut w_corr_mo_a = Matrix::zeros(n_basis, n_basis);
    for i in 0..n_alpha {
        for j in 0..n_alpha {
            let mut sum_w = 0.5 * (eps_a[i] + eps_a[j]) * d_occ_a[(i, j)];
            for k in 0..n_alpha {
                for a in 0..n_virt_a {
                    let a_mo = n_alpha + a;
                    for b in 0..n_virt_a {
                        let b_mo = n_alpha + b;
                        let t = t2_aa[i][k][a][b];
                        let jakb = mo_eri_a.get(j, a_mo, k, b_mo);
                        sum_w += t * jakb;
                    }
                }
            }
            for k in 0..n_beta {
                for a in 0..n_virt_a {
                    let a_mo = n_alpha + a;
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        let t = t2_ab[i][k][a][b];
                        let jakb = cross_eri.get(j, a_mo, k, b_mo);
                        sum_w += t * jakb;
                    }
                }
            }
            for p in 0..n_basis {
                for q in 0..n_basis {
                    let p_pq = p_corr_mo_a[(p, q)];
                    if p_pq.abs() > 1e-15 {
                        let ijpq = mo_eri_a.get(i, j, p, q);
                        let ipjq = mo_eri_a.get(i, p, j, q);
                        sum_w += p_pq * (ijpq - ipjq);
                    }
                }
            }
            for p in 0..n_basis {
                for q in 0..n_basis {
                    let p_pq = p_corr_mo_b[(p, q)];
                    if p_pq.abs() > 1e-15 {
                        let ijpq = cross_eri.get(i, j, p, q);
                        sum_w += p_pq * ijpq;
                    }
                }
            }
            w_corr_mo_a[(i, j)] = sum_w;
        }
    }

    for a in 0..n_virt_a {
        let a_mo = n_alpha + a;
        for b in 0..n_virt_a {
            let b_mo = n_alpha + b;
            let mut sum_w = 0.5 * (eps_a[a_mo] + eps_a[b_mo]) * d_virt_a[(a, b)];
            for i in 0..n_alpha {
                for j in 0..n_alpha {
                    for c in 0..n_virt_a {
                        let c_mo = n_alpha + c;
                        let t = t2_aa[i][j][a][c];
                        let ibjc = mo_eri_a.get(i, b_mo, j, c_mo);
                        sum_w += t * ibjc;
                    }
                }
            }
            for i in 0..n_alpha {
                for j in 0..n_beta {
                    for c in 0..n_virt_b {
                        let c_mo = n_beta + c;
                        let t = t2_ab[i][j][a][c];
                        let ibjc = cross_eri.get(i, b_mo, j, c_mo);
                        sum_w += t * ibjc;
                    }
                }
            }
            w_corr_mo_a[(a_mo, b_mo)] = sum_w;
        }
    }

    for a in 0..n_virt_a {
        let a_mo = n_alpha + a;
        for i in 0..n_alpha {
            let z_ai = z_mat_a[(a, i)];
            let mut sum_ia = 0.0;
            for j in 0..n_alpha {
                for k in 0..n_alpha {
                    for b in 0..n_virt_a {
                        let b_mo = n_alpha + b;
                        let t = t2_aa[j][k][a][b];
                        let jikb = mo_eri_a.get(j, i, k, b_mo);
                        sum_ia += t * jikb;
                    }
                }
            }
            for j in 0..n_beta {
                for k in 0..n_alpha {
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        let t = t2_ab[k][j][a][b];
                        let kijb = cross_eri.get(k, i, j, b_mo);
                        sum_ia += t * kijb;
                    }
                }
            }
            let w_val = eps_a[i] * z_ai + sum_ia;
            w_corr_mo_a[(i, a_mo)] = w_val;
            w_corr_mo_a[(a_mo, i)] = w_val;
        }
    }

    // W^corr for beta:
    let mut w_corr_mo_b = Matrix::zeros(n_basis, n_basis);
    for i in 0..n_beta {
        for j in 0..n_beta {
            let mut sum_w = 0.5 * (eps_b[i] + eps_b[j]) * d_occ_b[(i, j)];
            for k in 0..n_beta {
                for a in 0..n_virt_b {
                    let a_mo = n_beta + a;
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        let t = t2_bb[i][k][a][b];
                        let jakb = mo_eri_b.get(j, a_mo, k, b_mo);
                        sum_w += t * jakb;
                    }
                }
            }
            for k in 0..n_alpha {
                for a in 0..n_virt_a {
                    let a_mo = n_alpha + a;
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        let t = t2_ab[k][i][a][b];
                        let kjab = cross_eri.get(k, a_mo, j, b_mo);
                        sum_w += t * kjab;
                    }
                }
            }
            for p in 0..n_basis {
                for q in 0..n_basis {
                    let p_pq = p_corr_mo_b[(p, q)];
                    if p_pq.abs() > 1e-15 {
                        let ijpq = mo_eri_b.get(i, j, p, q);
                        let ipjq = mo_eri_b.get(i, p, j, q);
                        sum_w += p_pq * (ijpq - ipjq);
                    }
                }
            }
            for p in 0..n_basis {
                for q in 0..n_basis {
                    let p_pq = p_corr_mo_a[(p, q)];
                    if p_pq.abs() > 1e-15 {
                        let pqij = cross_eri.get(p, q, i, j);
                        sum_w += p_pq * pqij;
                    }
                }
            }
            w_corr_mo_b[(i, j)] = sum_w;
        }
    }

    for a in 0..n_virt_b {
        let a_mo = n_beta + a;
        for b in 0..n_virt_b {
            let b_mo = n_beta + b;
            let mut sum_w = 0.5 * (eps_b[a_mo] + eps_b[b_mo]) * d_virt_b[(a, b)];
            for i in 0..n_beta {
                for j in 0..n_beta {
                    for c in 0..n_virt_b {
                        let c_mo = n_beta + c;
                        let t = t2_bb[i][j][a][c];
                        let ibjc = mo_eri_b.get(i, b_mo, j, c_mo);
                        sum_w += t * ibjc;
                    }
                }
            }
            for i in 0..n_alpha {
                for j in 0..n_beta {
                    for c in 0..n_virt_a {
                        let c_mo = n_alpha + c;
                        let t = t2_ab[i][j][c][a];
                        let icjb = cross_eri.get(i, c_mo, j, b_mo);
                        sum_w += t * icjb;
                    }
                }
            }
            w_corr_mo_b[(a_mo, b_mo)] = sum_w;
        }
    }

    for a in 0..n_virt_b {
        let a_mo = n_beta + a;
        for i in 0..n_beta {
            let z_ai = z_mat_b[(a, i)];
            let mut sum_ia = 0.0;
            for j in 0..n_beta {
                for k in 0..n_beta {
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        let t = t2_bb[j][k][a][b];
                        let jikb = mo_eri_b.get(j, i, k, b_mo);
                        sum_ia += t * jikb;
                    }
                }
            }
            for j in 0..n_alpha {
                for k in 0..n_beta {
                    for b in 0..n_virt_a {
                        let b_mo = n_alpha + b;
                        let t = t2_ab[j][k][b][a];
                        let jbki = cross_eri.get(j, b_mo, k, i);
                        sum_ia += t * jbki;
                    }
                }
            }
            let w_val = eps_b[i] * z_ai + sum_ia;
            w_corr_mo_b[(i, a_mo)] = w_val;
            w_corr_mo_b[(a_mo, i)] = w_val;
        }
    }

    // Symmetrize W_mo and transform to AO using faer
    let mut w_symm_a = Matrix::zeros(n_basis, n_basis);
    let mut w_symm_b = Matrix::zeros(n_basis, n_basis);
    for p in 0..n_basis {
        for q in 0..n_basis {
            w_symm_a[(p, q)] = 0.5 * (w_corr_mo_a[(p, q)] + w_corr_mo_a[(q, p)]);
            w_symm_b[(p, q)] = 0.5 * (w_corr_mo_b[(p, q)] + w_corr_mo_b[(q, p)]);
        }
    }

    let w_faer_a = faer::Mat::<f64>::from_fn(n_basis, n_basis, |i, j| w_symm_a[(i, j)]);
    let w_faer_b = faer::Mat::<f64>::from_fn(n_basis, n_basis, |i, j| w_symm_b[(i, j)]);

    let w_ao_a = &ca_faer * (&w_faer_a * ca_faer.transpose());
    let w_ao_b = &cb_faer * (&w_faer_b * cb_faer.transpose());

    let mut w_corr_ao_a = Matrix::zeros(n_basis, n_basis);
    let mut w_corr_ao_b = Matrix::zeros(n_basis, n_basis);
    for i in 0..n_basis {
        for j in 0..n_basis {
            w_corr_ao_a[(i, j)] = w_ao_a[(i, j)];
            w_corr_ao_b[(i, j)] = w_ao_b[(i, j)];
        }
    }

    // 7. Construct correlated 2-PDM in AO basis:
    let mut gamma_ao = vec![vec![vec![vec![0.0; n_basis]; n_basis]; n_basis]; n_basis];

    // Alpha-Alpha
    for i in 0..n_alpha {
        for j in 0..n_alpha {
            for a in 0..n_virt_a {
                let a_mo = n_alpha + a;
                for b in 0..n_virt_a {
                    let b_mo = n_alpha + b;
                    let val = t2_aa[i][j][a][b];
                    if val.abs() < 1e-15 {
                        continue;
                    }
                    for mu in 0..n_basis {
                        for nu in 0..n_basis {
                            for la in 0..n_basis {
                                for si in 0..n_basis {
                                    let t1 = c_a[(mu, i)] * c_a[(nu, a_mo)] * c_a[(la, j)] * c_a[(si, b_mo)];
                                    let t2 = c_a[(nu, i)] * c_a[(mu, a_mo)] * c_a[(la, j)] * c_a[(si, b_mo)];
                                    let t3 = c_a[(la, i)] * c_a[(si, a_mo)] * c_a[(mu, j)] * c_a[(nu, b_mo)];
                                    let t4 = c_a[(la, i)] * c_a[(si, a_mo)] * c_a[(nu, j)] * c_a[(mu, b_mo)];
                                    gamma_ao[mu][nu][la][si] += val * (t1 + t2 + t3 + t4);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Beta-Beta
    for i in 0..n_beta {
        for j in 0..n_beta {
            for a in 0..n_virt_b {
                let a_mo = n_beta + a;
                for b in 0..n_virt_b {
                    let b_mo = n_beta + b;
                    let val = t2_bb[i][j][a][b];
                    if val.abs() < 1e-15 {
                        continue;
                    }
                    for mu in 0..n_basis {
                        for nu in 0..n_basis {
                            for la in 0..n_basis {
                                for si in 0..n_basis {
                                    let t1 = c_b[(mu, i)] * c_b[(nu, a_mo)] * c_b[(la, j)] * c_b[(si, b_mo)];
                                    let t2 = c_b[(nu, i)] * c_b[(mu, a_mo)] * c_b[(la, j)] * c_b[(si, b_mo)];
                                    let t3 = c_b[(la, i)] * c_b[(si, a_mo)] * c_b[(mu, j)] * c_b[(nu, b_mo)];
                                    let t4 = c_b[(la, i)] * c_b[(si, a_mo)] * c_b[(nu, j)] * c_b[(mu, b_mo)];
                                    gamma_ao[mu][nu][la][si] += val * (t1 + t2 + t3 + t4);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Alpha-Beta
    for i in 0..n_alpha {
        for j in 0..n_beta {
            for a in 0..n_virt_a {
                let a_mo = n_alpha + a;
                for b in 0..n_virt_b {
                    let b_mo = n_beta + b;
                    let val = 2.0 * t2_ab[i][j][a][b];
                    if val.abs() < 1e-15 {
                        continue;
                    }
                    for mu in 0..n_basis {
                        for nu in 0..n_basis {
                            for la in 0..n_basis {
                                for si in 0..n_basis {
                                    let t1 = c_a[(mu, i)] * c_a[(nu, a_mo)] * c_b[(la, j)] * c_b[(si, b_mo)];
                                    let t2 = c_a[(nu, i)] * c_a[(mu, a_mo)] * c_b[(la, j)] * c_b[(si, b_mo)];
                                    let t3 = c_b[(mu, j)] * c_b[(nu, b_mo)] * c_a[(la, i)] * c_a[(si, a_mo)];
                                    let t4 = c_b[(nu, j)] * c_b[(mu, b_mo)] * c_a[(la, i)] * c_a[(si, a_mo)];
                                    gamma_ao[mu][nu][la][si] += val * (t1 + t2 + t3 + t4);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 8. Contract with derivative integrals
    let g_uhf = crate::gradient::compute_uhf_gradient(atoms, basis, uhf);
    let p_uhf_a = &uhf.density_matrix_alpha;
    let p_uhf_b = &uhf.density_matrix_beta;
    let p_uhf_tot = p_uhf_a.add(p_uhf_b);
    let p_corr_tot = p_corr_ao_a.add(&p_corr_ao_b);
    let w_corr_tot = w_corr_ao_a.add(&w_corr_ao_b);

    let mut basis_derivs = Vec::with_capacity(n_basis);
    for cg in basis {
        let dx = differentiate_basis_function(cg, 0);
        let dy = differentiate_basis_function(cg, 1);
        let dz = differentiate_basis_function(cg, 2);
        basis_derivs.push([dx, dy, dz]);
    }

    (0..n_atoms)
        .into_par_iter()
        .map(|atom_idx| {
            let mut grad_a = g_uhf[atom_idx];

            for tau in 0..3 {
                let mut d_s_val = 0.0;
                let mut d_t_val = 0.0;
                let mut d_v_val = 0.0;

                for mu in 0..n_basis {
                    let mu_on_a = basis[mu].atom_index == atom_idx;
                    for nu in 0..n_basis {
                        let nu_on_a = basis[nu].atom_index == atom_idx;
                        let p_munu = p_corr_tot[(mu, nu)];
                        let w_munu = w_corr_tot[(mu, nu)];

                        if p_munu.abs() < 1e-15 && w_munu.abs() < 1e-15 {
                            continue;
                        }

                        let mut ds = 0.0;
                        if mu_on_a {
                            ds += deriv_overlap(&basis_derivs[mu][tau], &basis[nu]);
                        }
                        if nu_on_a {
                            ds += deriv_overlap(&basis_derivs[nu][tau], &basis[mu]);
                        }
                        d_s_val += w_munu * ds;

                        let mut dt = 0.0;
                        if mu_on_a {
                            dt += deriv_kinetic(&basis_derivs[mu][tau], &basis[nu]);
                        }
                        if nu_on_a {
                            dt += deriv_kinetic(&basis_derivs[nu][tau], &basis[mu]);
                        }
                        d_t_val += p_munu * dt;

                        let mut dv = 0.0;
                        if mu_on_a {
                            dv += deriv_nuclear_attraction(&basis_derivs[mu][tau], &basis[nu], atoms);
                        }
                        if nu_on_a {
                            dv += deriv_nuclear_attraction(&basis_derivs[nu][tau], &basis[mu], atoms);
                        }

                        let target_atom = &atoms[atom_idx];
                        let hf_term = -(
                            deriv_nuclear_attraction_single_atom(&basis_derivs[mu][tau], &basis[nu], target_atom)
                            + deriv_nuclear_attraction_single_atom(&basis_derivs[nu][tau], &basis[mu], target_atom)
                        );
                        dv += hf_term;

                        d_v_val += p_munu * dv;
                    }
                }

                grad_a[tau] += (d_t_val + d_v_val) - d_s_val;

                // 2-electron integral derivative contraction:
                let mut d_eri_relax = 0.0;
                let mut d_eri_2pdm = 0.0;
                for mu in 0..n_basis {
                    if basis[mu].atom_index != atom_idx {
                        continue;
                    }

                    for term in &basis_derivs[mu][tau].terms {
                        for nu in 0..n_basis {
                            for lambda in 0..n_basis {
                                for sigma in 0..n_basis {
                                    let gamma_relax = 2.0 * (
                                        p_uhf_tot[(mu, nu)] * p_corr_tot[(lambda, sigma)]
                                        + p_corr_tot[(mu, nu)] * p_uhf_tot[(lambda, sigma)]
                                    ) - 2.0 * (
                                        p_uhf_a[(mu, lambda)] * p_corr_ao_a[(nu, sigma)]
                                        + p_corr_ao_a[(mu, lambda)] * p_uhf_a[(nu, sigma)]
                                        + p_uhf_b[(mu, lambda)] * p_corr_ao_b[(nu, sigma)]
                                        + p_corr_ao_b[(mu, lambda)] * p_uhf_b[(nu, sigma)]
                                    );

                                    let gamma_2pdm = gamma_ao[mu][nu][lambda][sigma];

                                    let eri_val = contracted_eri(term, &basis[nu], &basis[lambda], &basis[sigma]);
                                    if gamma_relax.abs() > 1e-15 {
                                        d_eri_relax += gamma_relax * eri_val;
                                    }
                                    if gamma_2pdm.abs() > 1e-15 {
                                        d_eri_2pdm += gamma_2pdm * eri_val;
                                    }
                                }
                            }
                        }
                    }
                }

                grad_a[tau] += d_eri_relax + d_eri_2pdm;
            }

            grad_a
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::build_basis;
    use crate::post_hf::mp2::RHFMP2;
    use crate::post_hf::PostHFMethod;
    use crate::scf::run_scf_general;

    #[test]
    fn test_rhf_mp2_analytical_gradient_vs_finite_difference() {
        let basis_map = crate::basis::parse_gaussian94(crate::basis::STO_3G_GBS);
        let r = 1.4;
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -r / 2.0]),
            Atom::new("H", [0.0, 0.0, r / 2.0]),
        ];
        let basis = build_basis(&atoms, &basis_map);

        let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, crate::diis::ConvAlgo::Cdiis, None)
            .expect("RHF converged");

        let grad_anal = compute_rhf_mp2_analytical_gradient(&atoms, &basis, &hf);

        // Translational invariance: sum_A g_A = 0
        let mut sum_gx = 0.0;
        let mut sum_gy = 0.0;
        let mut sum_gz = 0.0;
        for a in 0..atoms.len() {
            sum_gx += grad_anal[a][0];
            sum_gy += grad_anal[a][1];
            sum_gz += grad_anal[a][2];
        }
        assert!(sum_gx.abs() < 1e-6);
        assert!(sum_gy.abs() < 1e-6);
        assert!(sum_gz.abs() < 1e-6);

        // Finite difference verification along z for atom 1
        let h = 1e-4;
        let mut atoms_fwd = atoms.clone();
        atoms_fwd[1].position[2] += h;
        let basis_fwd = build_basis(&atoms_fwd, &basis_map);
        let hf_fwd = run_scf_general(&atoms_fwd, &basis_fwd, 0, 50, 1e-12, crate::diis::ConvAlgo::Cdiis, None).unwrap();
        let mp2_fwd = RHFMP2::new(&hf_fwd).compute().unwrap();
        let e_tot_fwd = hf_fwd.total_energy + mp2_fwd.correlation_energy;

        let mut atoms_bwd = atoms.clone();
        atoms_bwd[1].position[2] -= h;
        let basis_bwd = build_basis(&atoms_bwd, &basis_map);
        let hf_bwd = run_scf_general(&atoms_bwd, &basis_bwd, 0, 50, 1e-12, crate::diis::ConvAlgo::Cdiis, None).unwrap();
        let mp2_bwd = RHFMP2::new(&hf_bwd).compute().unwrap();
        let e_tot_bwd = hf_bwd.total_energy + mp2_bwd.correlation_energy;

        let g_hf = crate::gradient::compute_rhf_gradient(&atoms, &basis, &hf);
        let g_hf_z = (hf_fwd.total_energy - hf_bwd.total_energy) / (2.0 * h);
        let g_mp2_corr_num = (mp2_fwd.correlation_energy - mp2_bwd.correlation_energy) / (2.0 * h);
        let g_num_z = (e_tot_fwd - e_tot_bwd) / (2.0 * h);
        let _diff = (grad_anal[1][2] - g_num_z).abs();

        eprintln!("HF analytical: {}", g_hf[1][2]);
        eprintln!("HF numerical:  {}", g_hf_z);
        eprintln!("MP2 corr num:  {}", g_mp2_corr_num);
        eprintln!("MP2 total num: {}", g_num_z);
        eprintln!("MP2 total ana: {}", grad_anal[1][2]);
        eprintln!("MP2 corr ana:  {}", grad_anal[1][2] - g_hf[1][2]);

        assert!(
            (grad_anal[1][2] - g_num_z).abs() < 1e-3,
            "Analytical MP2 gradient ({}) must match finite difference ({})",
            grad_anal[1][2], g_num_z
        );
    }

    #[test]
    fn test_uhf_mp2_analytical_gradient_vs_rhf_mp2_h2() {
        let basis_map = crate::basis::parse_gaussian94(crate::basis::STO_3G_GBS);
        let r = 1.4;
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -r / 2.0]),
            Atom::new("H", [0.0, 0.0,  r / 2.0]),
        ];
        let basis = build_basis(&atoms, &basis_map);

        let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, crate::diis::ConvAlgo::Cdiis, None).unwrap();
        let uhf = crate::scf::run_uhf(&atoms, &basis, 0, 1, false, 50, 1e-12).unwrap();

        let g_rhf = compute_rhf_mp2_analytical_gradient(&atoms, &basis, &hf);
        let g_uhf = compute_uhf_mp2_analytical_gradient(&atoms, &basis, &uhf);

        for a in 0..2 {
            for c in 0..3 {
                let diff = (g_rhf[a][c] - g_uhf[a][c]).abs();
                assert!(diff < 1e-7, "H2 RHF-MP2 and UHF-MP2 gradients must match: atom {}, comp {}, diff {}", a, c, diff);
            }
        }
    }

    #[test]
    fn test_uhf_mp2_analytical_gradient_vs_rhf_mp2_lih() {
        let basis_map = crate::basis::parse_gaussian94(crate::basis::STO_3G_GBS);
        let atoms = vec![
            Atom::new("Li", [0.0, 0.0, 0.0]),
            Atom::new("H",  [0.0, 0.0, 3.014]),
        ];
        let basis = build_basis(&atoms, &basis_map);

        let hf = run_scf_general(&atoms, &basis, 0, 60, 1e-12, crate::diis::ConvAlgo::Adiissoscf, None).unwrap();
        let uhf = crate::scf::run_uhf(&atoms, &basis, 0, 1, false, 60, 1e-12).unwrap();

        let g_rhf = compute_rhf_mp2_analytical_gradient(&atoms, &basis, &hf);
        let g_uhf = compute_uhf_mp2_analytical_gradient(&atoms, &basis, &uhf);

        for a in 0..2 {
            for c in 0..3 {
                let diff = (g_rhf[a][c] - g_uhf[a][c]).abs();
                assert!(diff < 1e-6, "LiH RHF-MP2 and UHF-MP2 gradients must match: atom {}, comp {}, diff {}", a, c, diff);
            }
        }
    }

    #[test]
    fn test_uhf_mp2_analytical_gradient_vs_rhf_mp2_water() {
        let basis_map = crate::basis::parse_gaussian94(crate::basis::STO_3G_GBS);
        let atoms = vec![
            Atom::new("O", [0.0, 0.0, 0.2217]),
            Atom::new("H", [0.0, 1.4309, -0.8867]),
            Atom::new("H", [0.0, -1.4309, -0.8867]),
        ];
        let basis = build_basis(&atoms, &basis_map);

        let hf = run_scf_general(&atoms, &basis, 0, 60, 1e-11, crate::diis::ConvAlgo::Adiissoscf, None).unwrap();
        let uhf = crate::scf::run_uhf(&atoms, &basis, 0, 1, false, 60, 1e-11).unwrap();

        let g_rhf = compute_rhf_mp2_analytical_gradient(&atoms, &basis, &hf);
        let g_uhf = compute_uhf_mp2_analytical_gradient(&atoms, &basis, &uhf);

        for a in 0..3 {
            for c in 0..3 {
                let diff = (g_rhf[a][c] - g_uhf[a][c]).abs();
                assert!(diff < 1e-6, "H2O RHF-MP2 and UHF-MP2 gradients must match: atom {}, comp {}, diff {}", a, c, diff);
            }
        }
    }

    #[test]
    fn test_uhf_mp2_analytical_gradient_oh_radical() {
        let basis_map = crate::basis::parse_gaussian94(crate::basis::STO_3G_GBS);
        let atoms = vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 1.834]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let uhf = crate::scf::run_uhf(&atoms, &basis, 0, 2, false, 60, 1e-11).unwrap();

        let g_ana = compute_uhf_mp2_analytical_gradient(&atoms, &basis, &uhf);

        // Translational invariance: sum_A g_A = 0
        let sum_z = g_ana[0][2] + g_ana[1][2];
        assert!(sum_z.abs() < 1e-5, "OH radical net z-force must vanish: {}", sum_z);

        // Finite difference along z for H (atom 1)
        let h = 1e-4;
        let mut atoms_fwd = atoms.clone();
        atoms_fwd[1].position[2] += h;
        let basis_fwd = build_basis(&atoms_fwd, &basis_map);
        let uhf_fwd = crate::scf::run_uhf(&atoms_fwd, &basis_fwd, 0, 2, false, 60, 1e-11).unwrap();
        let mp2_fwd = crate::post_hf::mp2::UHFMP2::new(&uhf_fwd).compute().unwrap();
        let e_tot_fwd = uhf_fwd.total_energy + mp2_fwd.correlation_energy;

        let mut atoms_bwd = atoms.clone();
        atoms_bwd[1].position[2] -= h;
        let basis_bwd = build_basis(&atoms_bwd, &basis_map);
        let uhf_bwd = crate::scf::run_uhf(&atoms_bwd, &basis_bwd, 0, 2, false, 60, 1e-11).unwrap();
        let mp2_bwd = crate::post_hf::mp2::UHFMP2::new(&uhf_bwd).compute().unwrap();
        let e_tot_bwd = uhf_bwd.total_energy + mp2_bwd.correlation_energy;

        let g_num_z = (e_tot_fwd - e_tot_bwd) / (2.0 * h);
        let diff = (g_ana[1][2] - g_num_z).abs();
        eprintln!("OH radical UHF-MP2 analytical: {}, numerical: {}, diff: {}", g_ana[1][2], g_num_z, diff);
        assert!(diff < 1e-4, "Analytical UHF-MP2 gradient must match finite difference: diff = {}", diff);
    }
}
