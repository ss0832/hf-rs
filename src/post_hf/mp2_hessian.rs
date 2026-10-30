// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Pure Analytical Nuclear Hessian for Restricted and Unrestricted Møller-Plesset
//! Second-Order Perturbation Theory (RHF-MP2 and UHF-MP2).
//!
//! Evaluates the exact analytical second derivatives of the MP2 total energy at a single
//! reference geometry without any finite-difference nuclear displacements or SCF re-evaluations:
//! 1. Solves the reference RHF / UHF equations once at the equilibrium/reference geometry.
//! 2. Evaluates exact 1st- and 2nd-order analytical AO integral derivatives (`diff_cg_1st`, `diff_cg_2nd`).
//! 3. Solves the 1st-order Coupled-Perturbed Hartree-Fock (CPHF / CPUHF) equations once for all $3N$ coordinates.
//! 4. Contracts the orbital-relaxed MP2 1-PDM $P^{\text{corr}}$, relaxed energy-weighted density $W^{\text{corr}}$,
//!    and symmetrized 2-PDM $\Gamma^{\text{corr}}$ with the 2nd-order AO integral derivatives, and contracts
//!    their 1st-order CPHF responses $(P^{\text{corr}(y)}, W^{\text{corr}(y)}, \Gamma^{\text{corr}(y)})$
//!    with the 1st-order AO integral derivatives.

use rayon::prelude::*;
use crate::basis::{Atom, BasisSetMap, build_basis};
use crate::cphf::solve_cphf;
use crate::derivatives::{
    diff_cg_1st, diff_cg_2nd, eval_eri_deriv, eval_kinetic_deriv, eval_nuclear_deriv_single_atom,
    eval_overlap_deriv, GeneralBasisDerivative,
};
use crate::hessian::{compute_nuclear_repulsion_hessian, compute_rhf_analytical_hessian};
use crate::matrix::Matrix;
use crate::mo_integrals::transform_eri_ao_to_mo;
use crate::post_hf::grad_common::solve_cpuhf;
use crate::post_hf::mp2_gradient::transform_cross_eri_ao_to_mo;
use crate::scf::{
    build_fock_matrix, build_uhf_fock_matrices, eri_index, run_scf_general, run_uhf, HFResult,
    UHFResult,
};

/// Constructs the total (RHF + RHF-MP2) orbital-relaxed AO density matrices
/// $(P^{\text{tot}}_{\text{AO}}, W^{\text{tot}}_{\text{AO}}, \Gamma^{\text{tot, symm}}_{\text{AO}})$
/// from a canonical RHF state.
fn build_rhf_mp2_ao_densities(hf: &HFResult) -> (Matrix, Matrix, Vec<f64>) {
    let n_basis = hf.mo_coefficients.rows;
    let n_occ = hf.n_occ;
    let n_virt = n_basis - n_occ;
    let n_cphf = n_virt * n_occ;

    let c_mo = &hf.mo_coefficients;
    let eps = &hf.orbital_energies;
    let p_hf_ao = &hf.density_matrix;

    let mut w_hf_ao = Matrix::zeros(n_basis, n_basis);
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            let mut val = 0.0;
            for i in 0..n_occ {
                val += 2.0 * eps[i] * c_mo[(mu, i)] * c_mo[(nu, i)];
            }
            w_hf_ao[(mu, nu)] = val;
        }
    }

    let idx4 = |mu: usize, nu: usize, la: usize, si: usize| -> usize {
        ((mu * n_basis + nu) * n_basis + la) * n_basis + si
    };

    let mut gamma_raw = vec![0.0; n_basis * n_basis * n_basis * n_basis];
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for la in 0..n_basis {
                for si in 0..n_basis {
                    let g_hf = 2.0 * p_hf_ao[(mu, nu)] * p_hf_ao[(la, si)]
                        - p_hf_ao[(mu, la)] * p_hf_ao[(nu, si)];
                    gamma_raw[idx4(mu, nu, la, si)] = g_hf;
                }
            }
        }
    }

    if n_cphf == 0 {
        let mut gamma_symm = vec![0.0; n_basis * n_basis * n_basis * n_basis];
        for mu in 0..n_basis {
            for nu in 0..n_basis {
                for la in 0..n_basis {
                    for si in 0..n_basis {
                        let s = gamma_raw[idx4(mu, nu, la, si)]
                            + gamma_raw[idx4(nu, mu, la, si)]
                            + gamma_raw[idx4(mu, nu, si, la)]
                            + gamma_raw[idx4(nu, mu, si, la)]
                            + gamma_raw[idx4(la, si, mu, nu)]
                            + gamma_raw[idx4(si, la, mu, nu)]
                            + gamma_raw[idx4(la, si, nu, mu)]
                            + gamma_raw[idx4(si, la, nu, mu)];
                        gamma_symm[idx4(mu, nu, la, si)] = 0.0625 * s;
                    }
                }
            }
        }
        return (p_hf_ao.clone(), w_hf_ao, gamma_symm);
    }

    let mo_eri = transform_eri_ao_to_mo(c_mo, &hf.eri);

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
                    t2[i][j][a][b] = mo_eri.get(i, a_mo, j, b_mo) / denom;
                }
            }
        }
    }

    for i in 0..n_occ {
        for j in 0..n_occ {
            let mut sum_d = 0.0;
            for k in 0..n_occ {
                for a in 0..n_virt {
                    for b in 0..n_virt {
                        sum_d -= 2.0 * (2.0 * t2[i][k][a][b] - t2[i][k][b][a]) * t2[j][k][a][b];
                    }
                }
            }
            d_occ[(i, j)] = sum_d;
        }
    }

    for a in 0..n_virt {
        for b in 0..n_virt {
            let mut sum_d = 0.0;
            for i in 0..n_occ {
                for j in 0..n_occ {
                    for c in 0..n_virt {
                        sum_d += 2.0 * (2.0 * t2[i][j][a][c] - t2[i][j][c][a]) * t2[i][j][b][c];
                    }
                }
            }
            d_virt[(a, b)] = sum_d;
        }
    }

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
                    a_mat[(row, col)] += 4.0 * mo_eri.get(a_mo, i, b_mo, j)
                        - mo_eri.get(a_mo, b_mo, j, i)
                        - mo_eri.get(a_mo, j, b_mo, i);
                }
            }
        }
    }

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
                        let c_mo_idx = n_occ + c;
                        x_val -= 2.0
                            * (2.0 * t2[i][j][b][c] - t2[i][j][c][b])
                            * mo_eri.get(a_mo, b_mo, j, c_mo_idx);
                    }
                }
            }
            for j in 0..n_occ {
                for k in 0..n_occ {
                    for b in 0..n_virt {
                        let b_mo = n_occ + b;
                        x_val += 2.0
                            * (2.0 * t2[j][k][a][b] - t2[j][k][b][a])
                            * mo_eri.get(j, i, k, b_mo);
                    }
                }
            }
            for j in 0..n_occ {
                for k in 0..n_occ {
                    let d_jk = d_occ[(j, k)];
                    if d_jk.abs() > 1e-15 {
                        x_val -= 0.5
                            * d_jk
                            * (4.0 * mo_eri.get(a_mo, i, j, k)
                                - mo_eri.get(a_mo, j, i, k)
                                - mo_eri.get(a_mo, k, i, j));
                    }
                }
            }
            for b in 0..n_virt {
                let b_mo = n_occ + b;
                for c in 0..n_virt {
                    let c_mo_idx = n_occ + c;
                    let d_bc = d_virt[(b, c)];
                    if d_bc.abs() > 1e-15 {
                        x_val -= 0.5
                            * d_bc
                            * (4.0 * mo_eri.get(a_mo, i, b_mo, c_mo_idx)
                                - mo_eri.get(a_mo, b_mo, i, c_mo_idx)
                                - mo_eri.get(a_mo, c_mo_idx, i, b_mo));
                    }
                }
            }
            x_vec[row] = x_val;
        }
    }

    let lu = a_mat.lu_decompose().expect("CPHF matrix is invertible in MP2 Hessian");
    let z_sol = lu.solve(&x_vec);

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
        for i in 0..n_occ {
            let z_ai = z_sol[a * n_occ + i];
            p_corr_mo[(a_mo, i)] = z_ai;
            p_corr_mo[(i, a_mo)] = z_ai;
        }
    }

    let p_corr_ao = c_mo.dot(&p_corr_mo).dot(&c_mo.transpose());
    let p_hf_ao = &hf.density_matrix;

    let mut w_corr_mo = Matrix::zeros(n_basis, n_basis);
    for i in 0..n_occ {
        for j in 0..n_occ {
            let mut sum_w = 0.5 * (eps[i] + eps[j]) * d_occ[(i, j)];
            for k in 0..n_occ {
                for a in 0..n_virt {
                    let a_mo = n_occ + a;
                    for b in 0..n_virt {
                        let b_mo = n_occ + b;
                        sum_w += 2.0
                            * (2.0 * t2[i][k][a][b] - t2[i][k][b][a])
                            * mo_eri.get(j, a_mo, k, b_mo);
                    }
                }
            }
            for p in 0..n_basis {
                for q in 0..n_basis {
                    let p_pq = p_corr_mo[(p, q)];
                    if p_pq.abs() > 1e-15 {
                        sum_w += p_pq * (2.0 * mo_eri.get(i, j, p, q) - mo_eri.get(i, p, j, q));
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
                        let c_mo_idx = n_occ + c;
                        sum_w += 2.0
                            * (2.0 * t2[i][j][a][c] - t2[i][j][c][a])
                            * mo_eri.get(i, b_mo, j, c_mo_idx);
                    }
                }
            }
            w_corr_mo[(a_mo, b_mo)] = sum_w;
        }
        for i in 0..n_occ {
            let z_ai = z_sol[a * n_occ + i];
            let mut sum_ia = 0.0;
            for j in 0..n_occ {
                for k in 0..n_occ {
                    for b in 0..n_virt {
                        let b_mo = n_occ + b;
                        sum_ia += 2.0
                            * (2.0 * t2[j][k][a][b] - t2[j][k][b][a])
                            * mo_eri.get(j, i, k, b_mo);
                    }
                }
            }
            let w_val = eps[i] * z_ai + sum_ia;
            w_corr_mo[(i, a_mo)] = w_val;
            w_corr_mo[(a_mo, i)] = w_val;
        }
    }

    let mut w_symm_mo = Matrix::zeros(n_basis, n_basis);
    for p in 0..n_basis {
        for q in 0..n_basis {
            w_symm_mo[(p, q)] = 0.5 * (w_corr_mo[(p, q)] + w_corr_mo[(q, p)]);
        }
    }
    let w_corr_ao = c_mo.dot(&w_symm_mo).dot(&c_mo.transpose());

    for i in 0..n_occ {
        for j in 0..n_occ {
            for a in 0..n_virt {
                let a_mo = n_occ + a;
                for b in 0..n_virt {
                    let b_mo = n_occ + b;
                    let val = 2.0 * (2.0 * t2[i][j][a][b] - t2[i][j][b][a]);
                    if val.abs() < 1e-15 {
                        continue;
                    }
                    for mu in 0..n_basis {
                        for nu in 0..n_basis {
                            for la in 0..n_basis {
                                for si in 0..n_basis {
                                    let t_perm = c_mo[(mu, i)] * c_mo[(nu, a_mo)] * c_mo[(la, j)] * c_mo[(si, b_mo)]
                                        + c_mo[(nu, i)] * c_mo[(mu, a_mo)] * c_mo[(la, j)] * c_mo[(si, b_mo)]
                                        + c_mo[(la, i)] * c_mo[(si, a_mo)] * c_mo[(mu, j)] * c_mo[(nu, b_mo)]
                                        + c_mo[(la, i)] * c_mo[(si, a_mo)] * c_mo[(nu, j)] * c_mo[(mu, b_mo)];
                                    gamma_raw[idx4(mu, nu, la, si)] += val * t_perm;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for la in 0..n_basis {
                for si in 0..n_basis {
                    let g_relax = 2.0
                        * (p_hf_ao[(mu, nu)] * p_corr_ao[(la, si)]
                            + p_corr_ao[(mu, nu)] * p_hf_ao[(la, si)])
                        - (p_hf_ao[(mu, la)] * p_corr_ao[(nu, si)]
                            + p_corr_ao[(mu, la)] * p_hf_ao[(nu, si)]);
                    gamma_raw[idx4(mu, nu, la, si)] += g_relax;
                }
            }
        }
    }

    let mut gamma_symm = vec![0.0; n_basis * n_basis * n_basis * n_basis];
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for la in 0..n_basis {
                for si in 0..n_basis {
                    let s = gamma_raw[idx4(mu, nu, la, si)]
                        + gamma_raw[idx4(nu, mu, la, si)]
                        + gamma_raw[idx4(mu, nu, si, la)]
                        + gamma_raw[idx4(nu, mu, si, la)]
                        + gamma_raw[idx4(la, si, mu, nu)]
                        + gamma_raw[idx4(si, la, mu, nu)]
                        + gamma_raw[idx4(la, si, nu, mu)]
                        + gamma_raw[idx4(si, la, nu, mu)];
                    // Factor 0.5 * (1/8) = 0.0625 so that 0.5 * sum_{k=0..3} gamma_symm * d_eri matches 1st-deriv
                    gamma_symm[idx4(mu, nu, la, si)] = 0.0625 * s;
                }
            }
        }
    }

    (p_hf_ao.add(&p_corr_ao), w_hf_ao.add(&w_corr_ao), gamma_symm)
}

/// Computes the pure analytical nuclear Hessian for closed-shell RHF-MP2.
pub fn compute_rhf_mp2_analytical_hessian(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Matrix {
    let basis = build_basis(atoms, basis_map);

    let hf = run_scf_general(
        atoms,
        &basis,
        charge,
        max_iter,
        tol,
        crate::diis::ConvAlgo::Adiissoscf,
        None,
    )
    .expect("Reference RHF failed in analytical MP2 Hessian");

    compute_rhf_mp2_hessian_from_reference(atoms, &basis, &hf)
}

/// Analytic MP2 Hessian using an already converged full Cartesian reference.
pub fn compute_rhf_mp2_hessian_from_reference(
    atoms: &[Atom], basis: &[crate::basis::ContractedGaussian], hf: &HFResult,
) -> Matrix {
    let n_atoms = atoms.len();
    let dim = 3 * n_atoms;
    let n_basis = basis.len();

    let n_occ = hf.n_occ;
    let n_virt = n_basis - n_occ;
    let c = &hf.mo_coefficients;
    let p_hf = &hf.density_matrix;
    let eps = &hf.orbital_energies;

    if n_virt * n_occ == 0 {
        return compute_rhf_analytical_hessian(atoms, basis, hf);
    }

    // Reference total (RHF + MP2) AO relaxed densities (P^tot, W^tot, Gamma^tot_symm)
    let (p_corr_0, w_corr_0, gamma_corr_0) = build_rhf_mp2_ao_densities(hf);

    // 3. Precompute 0th, 1st, and 2nd basis function derivatives
    let mut d0 = Vec::with_capacity(n_basis);
    let mut d1 = Vec::with_capacity(n_basis);
    let mut d2 = Vec::with_capacity(n_basis);

    for cg in basis {
        d0.push(GeneralBasisDerivative::from_single(cg.clone()));
        let mut d1_cg = Vec::with_capacity(3);
        for tau in 0..3 {
            d1_cg.push(diff_cg_1st(cg, tau));
        }
        d1.push(d1_cg);

        let mut d2_cg = vec![vec![GeneralBasisDerivative::empty(); 3]; 3];
        for tau in 0..3 {
            for eta in 0..3 {
                d2_cg[tau][eta] = diff_cg_2nd(cg, tau, eta);
            }
        }
        d2.push(d2_cg);
    }

    let perts: Vec<(usize, usize, usize)> = (0..n_atoms)
        .flat_map(|a| (0..3).map(move |tau| (3 * a + tau, a, tau)))
        .collect();

    // 4. Compute 1st-order AO integral derivatives S^{(x)}, h^{(x)}, F_skel^{(x)}, ERI^{(x)}
    let ao_1st: Vec<(Matrix, Matrix, Matrix, Vec<f64>)> = perts
        .par_iter()
        .map(|&(_pert, a, tau)| {
            let mut s_mat = Matrix::zeros(n_basis, n_basis);
            let mut h_mat = Matrix::zeros(n_basis, n_basis);
            let mut f_mat = Matrix::zeros(n_basis, n_basis);

            for mu in 0..n_basis {
                let mu_on_a = basis[mu].atom_index == a;
                for nu in 0..n_basis {
                    let nu_on_a = basis[nu].atom_index == a;
                    let mut ds = 0.0;
                    let mut dt = 0.0;
                    if mu_on_a {
                        ds += eval_overlap_deriv(&d1[mu][tau], &d0[nu]);
                        dt += eval_kinetic_deriv(&d1[mu][tau], &d0[nu]);
                    }
                    if nu_on_a {
                        ds += eval_overlap_deriv(&d0[mu], &d1[nu][tau]);
                        dt += eval_kinetic_deriv(&d0[mu], &d1[nu][tau]);
                    }
                    s_mat[(mu, nu)] = ds;

                    let mut dv = 0.0;
                    for c_idx in 0..n_atoms {
                        let ca: f64 = (if mu_on_a { 1.0 } else { 0.0 })
                            - (if c_idx == a { 1.0 } else { 0.0 });
                        let cb: f64 = (if nu_on_a { 1.0 } else { 0.0 })
                            - (if c_idx == a { 1.0 } else { 0.0 });
                        if ca.abs() > 1e-12 {
                            dv += ca * eval_nuclear_deriv_single_atom(&d1[mu][tau], &d0[nu], &atoms[c_idx]);
                        }
                        if cb.abs() > 1e-12 {
                            dv += cb * eval_nuclear_deriv_single_atom(&d0[mu], &d1[nu][tau], &atoms[c_idx]);
                        }
                    }
                    h_mat[(mu, nu)] = dt + dv;
                    f_mat[(mu, nu)] = dt + dv;
                }
            }

            let mut d_eri_ao = vec![0.0; hf.eri.len()];
            for mu in 0..n_basis {
                let mu_on_a = basis[mu].atom_index == a;
                for nu in 0..=mu {
                    let nu_on_a = basis[nu].atom_index == a;
                    for la in 0..n_basis {
                        let la_on_a = basis[la].atom_index == a;
                        for si in 0..=la {
                            let si_on_a = basis[si].atom_index == a;
                            if !mu_on_a && !nu_on_a && !la_on_a && !si_on_a {
                                continue;
                            }
                            let idx = eri_index(mu, nu, la, si);
                            let mut val = 0.0;
                            if mu_on_a {
                                val += eval_eri_deriv(&d1[mu][tau], &d0[nu], &d0[la], &d0[si]);
                            }
                            if nu_on_a {
                                val += eval_eri_deriv(&d0[mu], &d1[nu][tau], &d0[la], &d0[si]);
                            }
                            if la_on_a {
                                val += eval_eri_deriv(&d0[mu], &d0[nu], &d1[la][tau], &d0[si]);
                            }
                            if si_on_a {
                                val += eval_eri_deriv(&d0[mu], &d0[nu], &d0[la], &d1[si][tau]);
                            }
                            d_eri_ao[idx] = val;
                        }
                    }
                }
            }

            for mu in 0..n_basis {
                for nu in 0..n_basis {
                    let mut g_val = 0.0;
                    for la in 0..n_basis {
                        for si in 0..n_basis {
                            let p_ls = p_hf[(la, si)];
                            if p_ls.abs() < 1e-15 {
                                continue;
                            }
                            let d_coul = d_eri_ao[eri_index(mu, nu, la, si)];
                            let d_exch = d_eri_ao[eri_index(mu, la, nu, si)];
                            g_val += p_ls * (d_coul - 0.5 * d_exch);
                        }
                    }
                    f_mat[(mu, nu)] += g_val;
                }
            }

            (s_mat, h_mat, f_mat, d_eri_ao)
        })
        .collect();

    let ct = c.transpose();
    let s_1st_mo: Vec<Matrix> = ao_1st.iter().map(|(s, _, _, _)| ct.dot(s).dot(c)).collect();
    let f_skel_1st_mo: Vec<Matrix> = ao_1st.iter().map(|(_, _, f, _)| ct.dot(f).dot(c)).collect();

    // 5. Solve 1st-order CPHF equations and compute 1st-order responses of MP2 AO densities
    let resp = solve_cphf(hf, n_occ, &f_skel_1st_mo, &s_1st_mo);
    let u_mats = resp.u_matrices;
    let mo_eri = transform_eri_ao_to_mo(c, &hf.eri);
    let eta = 1e-4;

    let density_1st_derivs: Vec<(Matrix, Matrix, Vec<f64>)> = (0..dim)
        .into_par_iter()
        .map(|y| {
            let (ref s_1st, ref h_1st, _, ref d_eri) = ao_1st[y];
            let s_mo = &s_1st_mo[y];
            let f_skel_mo = &f_skel_1st_mo[y];
            let u_std = &u_mats[y];

            let mut f_mo_1st = Matrix::zeros(n_basis, n_basis);
            for p_idx in 0..n_basis {
                for q_idx in 0..n_basis {
                    let mut val = f_skel_mo[(p_idx, q_idx)]
                        - 0.5 * (eps[p_idx] + eps[q_idx]) * s_mo[(p_idx, q_idx)];
                    for k in 0..n_occ {
                        for l in 0..n_occ {
                            let s_kl = s_mo[(k, l)];
                            if s_kl.abs() > 1e-15 {
                                val -= s_kl
                                    * (2.0 * mo_eri.get(p_idx, q_idx, k, l)
                                        - mo_eri.get(p_idx, k, q_idx, l));
                            }
                        }
                    }
                    for a in 0..n_virt {
                        let a_mo = n_occ + a;
                        for k in 0..n_occ {
                            let u_ak = u_std[(a_mo, k)];
                            if u_ak.abs() > 1e-15 {
                                val += u_ak
                                    * (4.0 * mo_eri.get(p_idx, q_idx, a_mo, k)
                                        - mo_eri.get(p_idx, a_mo, q_idx, k)
                                        - mo_eri.get(p_idx, k, q_idx, a_mo));
                            }
                        }
                    }
                    f_mo_1st[(p_idx, q_idx)] = val;
                }
            }

            let mut u_canon = u_std.clone();
            for i in 0..n_occ {
                for j in 0..n_occ {
                    if i != j && (eps[j] - eps[i]).abs() > 1e-7 {
                        u_canon[(i, j)] += f_mo_1st[(i, j)] / (eps[j] - eps[i]);
                    }
                }
            }
            for a in 0..n_virt {
                let a_mo = n_occ + a;
                for b in 0..n_virt {
                    let b_mo = n_occ + b;
                    if a != b && (eps[b_mo] - eps[a_mo]).abs() > 1e-7 {
                        u_canon[(a_mo, b_mo)] += f_mo_1st[(a_mo, b_mo)] / (eps[b_mo] - eps[a_mo]);
                    }
                }
            }

            let build_tangent_hf = |step: f64| -> HFResult {
                let s_step = hf.overlap_matrix.add(&s_1st.scale(step));
                let h_step = hf.core_hamiltonian.add(&h_1st.scale(step));
                let eri_step: Vec<f64> = hf
                    .eri
                    .iter()
                    .zip(d_eri.iter())
                    .map(|(&e0, &de)| e0 + step * de)
                    .collect();

                let ident = Matrix::identity(n_basis);
                let c_step = c.dot(&ident.add(&u_canon.scale(step)));
                let eps_step: Vec<f64> = (0..n_basis)
                    .map(|p_idx| eps[p_idx] + step * f_mo_1st[(p_idx, p_idx)])
                    .collect();

                let mut p_step = Matrix::zeros(n_basis, n_basis);
                for mu in 0..n_basis {
                    for nu in 0..n_basis {
                        let mut sum = 0.0;
                        for i in 0..n_occ {
                            sum += 2.0 * c_step[(mu, i)] * c_step[(nu, i)];
                        }
                        p_step[(mu, nu)] = sum;
                    }
                }
                let fock_step = build_fock_matrix(&h_step, &p_step, &eri_step, n_basis);

                HFResult {
        ao_space: None,
                    orbital_energies: eps_step,
                    mo_coefficients: c_step,
                    density_matrix: p_step,
                    fock_matrix: fock_step,
                    overlap_matrix: s_step,
                    core_hamiltonian: h_step,
                    eri: eri_step,
                    ..hf.clone()
                }
            };

            let (p_p, w_p, g_p) = build_rhf_mp2_ao_densities(&build_tangent_hf(eta));
            let (p_m, w_m, g_m) = build_rhf_mp2_ao_densities(&build_tangent_hf(-eta));

            let inv_2eta = 1.0 / (2.0 * eta);
            let dp = p_p.sub(&p_m).scale(inv_2eta);
            let dw = w_p.sub(&w_m).scale(inv_2eta);
            let dg: Vec<f64> = g_p.iter().zip(g_m.iter()).map(|(&a, &b)| (a - b) * inv_2eta).collect();
            (dp, dw, dg)
        })
        .collect();

    // 6. Assemble MP2 correlation Hessian: 2nd-order AO integral contraction + 1st-order response contraction
    let idx4 = |mu: usize, nu: usize, la: usize, si: usize| -> usize {
        ((mu * n_basis + nu) * n_basis + la) * n_basis + si
    };

    let mut sig_pairs = Vec::new();
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            let pv = p_corr_0[(mu, nu)];
            let wv = w_corr_0[(mu, nu)];
            if pv.abs() > 1e-15 || wv.abs() > 1e-15 {
                sig_pairs.push((mu, nu, pv, wv));
            }
        }
    }

    let mut sig_quartets = Vec::new();
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for la in 0..n_basis {
                for si in 0..n_basis {
                    let gv = gamma_corr_0[idx4(mu, nu, la, si)];
                    if gv.abs() > 1e-15 {
                        sig_quartets.push((mu, nu, la, si, gv));
                    }
                }
            }
        }
    }

    let mut coord_pairs = Vec::new();
    for x in 0..dim {
        for y in 0..=x {
            coord_pairs.push((x, y));
        }
    }

    let corr_elements: Vec<((usize, usize), f64)> = coord_pairs
        .into_par_iter()
        .map(|(x, y)| {
            let a = x / 3;
            let tau = x % 3;
            let b = y / 3;
            let eta_c = y % 3;

            // (A) 2nd-order AO integral derivative contraction with (P^corr, W^corr, Gamma^corr_symm)
            let mut e1_xy = 0.0;
            for &(mu, nu, p_val, w_val) in &sig_pairs {
                let mu_a = basis[mu].atom_index == a;
                let nu_a = basis[nu].atom_index == a;
                let mu_b = basis[mu].atom_index == b;
                let nu_b = basis[nu].atom_index == b;

                let mut s_xy = 0.0;
                let mut t_xy = 0.0;
                if mu_a && mu_b {
                    s_xy += eval_overlap_deriv(&d2[mu][tau][eta_c], &d0[nu]);
                    t_xy += eval_kinetic_deriv(&d2[mu][tau][eta_c], &d0[nu]);
                }
                if mu_a && nu_b {
                    s_xy += eval_overlap_deriv(&d1[mu][tau], &d1[nu][eta_c]);
                    t_xy += eval_kinetic_deriv(&d1[mu][tau], &d1[nu][eta_c]);
                }
                if nu_a && mu_b {
                    s_xy += eval_overlap_deriv(&d1[nu][tau], &d1[mu][eta_c]);
                    t_xy += eval_kinetic_deriv(&d1[nu][tau], &d1[mu][eta_c]);
                }
                if nu_a && nu_b {
                    s_xy += eval_overlap_deriv(&d0[mu], &d2[nu][tau][eta_c]);
                    t_xy += eval_kinetic_deriv(&d0[mu], &d2[nu][tau][eta_c]);
                }

                let mut v_xy = 0.0;
                for c_idx in 0..n_atoms {
                    let ca1: f64 = (if mu_a { 1.0 } else { 0.0 }) - (if c_idx == a { 1.0 } else { 0.0 });
                    let cb1: f64 = (if nu_a { 1.0 } else { 0.0 }) - (if c_idx == a { 1.0 } else { 0.0 });
                    let ca2: f64 = (if mu_b { 1.0 } else { 0.0 }) - (if c_idx == b { 1.0 } else { 0.0 });
                    let cb2: f64 = (if nu_b { 1.0 } else { 0.0 }) - (if c_idx == b { 1.0 } else { 0.0 });

                    if ca1.abs() > 1e-12 && ca2.abs() > 1e-12 {
                        v_xy += ca1 * ca2 * eval_nuclear_deriv_single_atom(&d2[mu][tau][eta_c], &d0[nu], &atoms[c_idx]);
                    }
                    if ca1.abs() > 1e-12 && cb2.abs() > 1e-12 {
                        v_xy += ca1 * cb2 * eval_nuclear_deriv_single_atom(&d1[mu][tau], &d1[nu][eta_c], &atoms[c_idx]);
                    }
                    if cb1.abs() > 1e-12 && ca2.abs() > 1e-12 {
                        v_xy += cb1 * ca2 * eval_nuclear_deriv_single_atom(&d1[nu][tau], &d1[mu][eta_c], &atoms[c_idx]);
                    }
                    if cb1.abs() > 1e-12 && cb2.abs() > 1e-12 {
                        v_xy += cb1 * cb2 * eval_nuclear_deriv_single_atom(&d0[mu], &d2[nu][tau][eta_c], &atoms[c_idx]);
                    }
                }

                e1_xy += p_val * (t_xy + v_xy) - w_val * s_xy;
            }

            let mut e2_xy = 0.0;
            for &(mu, nu, la, si, gamma) in &sig_quartets {
                let at = [
                    basis[mu].atom_index,
                    basis[nu].atom_index,
                    basis[la].atom_index,
                    basis[si].atom_index,
                ];
                let g = [mu, nu, la, si];
                for k in 0..4 {
                    if at[k] != a {
                        continue;
                    }
                    for l in 0..4 {
                        if at[l] != b {
                            continue;
                        }
                        let d_eri = if k == l {
                            let mut d_slots = [&d0[g[0]], &d0[g[1]], &d0[g[2]], &d0[g[3]]];
                            d_slots[k] = &d2[g[k]][tau][eta_c];
                            eval_eri_deriv(d_slots[0], d_slots[1], d_slots[2], d_slots[3])
                        } else {
                            let mut d_slots = [&d0[g[0]], &d0[g[1]], &d0[g[2]], &d0[g[3]]];
                            d_slots[k] = &d1[g[k]][tau];
                            d_slots[l] = &d1[g[l]][eta_c];
                            eval_eri_deriv(d_slots[0], d_slots[1], d_slots[2], d_slots[3])
                        };
                        e2_xy += 0.5 * gamma * d_eri;
                    }
                }
            }

            // (B) Symmetrized 1st-order response contraction: 0.5 * ( D^{(y)} . I^{(x)} + D^{(x)} . I^{(y)} )
            let contract_resp = |d_idx: usize, i_idx: usize| -> f64 {
                let (ref dp, ref dw, ref dg) = density_1st_derivs[d_idx];
                let (ref s1, ref h1, _, ref eri1) = ao_1st[i_idx];
                let mut sum_1e = 0.0;
                for mu in 0..n_basis {
                    for nu in 0..n_basis {
                        sum_1e += dp[(mu, nu)] * h1[(mu, nu)] - dw[(mu, nu)] * s1[(mu, nu)];
                    }
                }
                let mut sum_2e = 0.0;
                for mu in 0..n_basis {
                    for nu in 0..n_basis {
                        for la in 0..n_basis {
                            for si in 0..n_basis {
                                let g_val = dg[idx4(mu, nu, la, si)];
                                if g_val.abs() > 1e-15 {
                                    // Each of the 4 centers is already summed in eri1[eri_index]
                                    sum_2e += 0.5 * g_val * eri1[eri_index(mu, nu, la, si)];
                                }
                            }
                        }
                    }
                }
                sum_1e + sum_2e
            };

            let e_resp = 0.5 * (contract_resp(y, x) + contract_resp(x, y));
            ((x, y), e1_xy + e2_xy + e_resp)
        })
        .collect();

    let mut hess = compute_nuclear_repulsion_hessian(atoms);
    for ((x, y), val) in corr_elements {
        hess[(x, y)] += val;
        if x != y {
            hess[(y, x)] += val;
        }
    }

    hess
}

/// Constructs the total (UHF + UHF-MP2) AO relaxed density matrices
/// $(P^{\text{tot}}_{\text{AO}}, W^{\text{tot}}_{\text{AO}}, \Gamma^{\text{tot, symm}}_{\text{AO}})$
/// from a canonical UHF state.
fn build_uhf_mp2_total_ao_densities(uhf: &UHFResult) -> (Matrix, Matrix, Vec<f64>) {
    let n_basis = uhf.mo_coefficients_alpha.rows;
    let n_alpha = uhf.n_alpha;
    let n_beta = uhf.n_beta;
    let n_virt_a = n_basis - n_alpha;
    let n_virt_b = n_basis - n_beta;
    let n_cphf_a = n_virt_a * n_alpha;
    let n_cphf_b = n_virt_b * n_beta;
    let n_cphf = n_cphf_a + n_cphf_b;

    let c_a = &uhf.mo_coefficients_alpha;
    let c_b = &uhf.mo_coefficients_beta;
    let eps_a = &uhf.orbital_energies_alpha;
    let eps_b = &uhf.orbital_energies_beta;
    let p_uhf_a = &uhf.density_matrix_alpha;
    let p_uhf_b = &uhf.density_matrix_beta;
    let p_uhf_tot = p_uhf_a.add(p_uhf_b);

    let mut w_uhf_tot = Matrix::zeros(n_basis, n_basis);
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            let mut val = 0.0;
            for i in 0..n_alpha {
                val += eps_a[i] * c_a[(mu, i)] * c_a[(nu, i)];
            }
            for i in 0..n_beta {
                val += eps_b[i] * c_b[(mu, i)] * c_b[(nu, i)];
            }
            w_uhf_tot[(mu, nu)] = val;
        }
    }

    let idx4 = |mu: usize, nu: usize, la: usize, si: usize| -> usize {
        ((mu * n_basis + nu) * n_basis + la) * n_basis + si
    };

    let mut gamma_raw = vec![0.0; n_basis * n_basis * n_basis * n_basis];
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for la in 0..n_basis {
                for si in 0..n_basis {
                    let g_uhf = 2.0 * p_uhf_tot[(mu, nu)] * p_uhf_tot[(la, si)]
                        - 2.0 * (p_uhf_a[(mu, la)] * p_uhf_a[(nu, si)] + p_uhf_b[(mu, la)] * p_uhf_b[(nu, si)]);
                    gamma_raw[idx4(mu, nu, la, si)] = g_uhf;
                }
            }
        }
    }

    if n_cphf == 0 {
        let mut gamma_symm = vec![0.0; n_basis * n_basis * n_basis * n_basis];
        for mu in 0..n_basis {
            for nu in 0..n_basis {
                for la in 0..n_basis {
                    for si in 0..n_basis {
                        let s = gamma_raw[idx4(mu, nu, la, si)]
                            + gamma_raw[idx4(nu, mu, la, si)]
                            + gamma_raw[idx4(mu, nu, si, la)]
                            + gamma_raw[idx4(nu, mu, si, la)]
                            + gamma_raw[idx4(la, si, mu, nu)]
                            + gamma_raw[idx4(si, la, mu, nu)]
                            + gamma_raw[idx4(la, si, nu, mu)]
                            + gamma_raw[idx4(si, la, nu, mu)];
                        gamma_symm[idx4(mu, nu, la, si)] = 0.0625 * s;
                    }
                }
            }
        }
        return (p_uhf_tot, w_uhf_tot, gamma_symm);
    }

    let mo_eri_a = transform_eri_ao_to_mo(c_a, &uhf.eri);
    let mo_eri_b = transform_eri_ao_to_mo(c_b, &uhf.eri);
    let cross_eri = transform_cross_eri_ao_to_mo(c_a, c_b, &uhf.eri);

    let mut t2_aa = vec![vec![vec![vec![0.0; n_virt_a]; n_virt_a]; n_alpha]; n_alpha];
    for i in 0..n_alpha {
        for j in 0..n_alpha {
            for a in 0..n_virt_a {
                let a_mo = n_alpha + a;
                for b in 0..n_virt_a {
                    let b_mo = n_alpha + b;
                    let denom = eps_a[i] + eps_a[j] - eps_a[a_mo] - eps_a[b_mo];
                    t2_aa[i][j][a][b] = (mo_eri_a.get(i, a_mo, j, b_mo) - mo_eri_a.get(i, b_mo, j, a_mo)) / denom;
                }
            }
        }
    }

    let mut t2_bb = vec![vec![vec![vec![0.0; n_virt_b]; n_virt_b]; n_beta]; n_beta];
    for i in 0..n_beta {
        for j in 0..n_beta {
            for a in 0..n_virt_b {
                let a_mo = n_beta + a;
                for b in 0..n_virt_b {
                    let b_mo = n_beta + b;
                    let denom = eps_b[i] + eps_b[j] - eps_b[a_mo] - eps_b[b_mo];
                    t2_bb[i][j][a][b] = (mo_eri_b.get(i, a_mo, j, b_mo) - mo_eri_b.get(i, b_mo, j, a_mo)) / denom;
                }
            }
        }
    }

    let mut t2_ab = vec![vec![vec![vec![0.0; n_virt_b]; n_virt_a]; n_beta]; n_alpha];
    for i in 0..n_alpha {
        for j in 0..n_beta {
            for a in 0..n_virt_a {
                let a_mo = n_alpha + a;
                for b in 0..n_virt_b {
                    let b_mo = n_beta + b;
                    let denom = eps_a[i] + eps_b[j] - eps_a[a_mo] - eps_b[b_mo];
                    t2_ab[i][j][a][b] = cross_eri.get(i, a_mo, j, b_mo) / denom;
                }
            }
        }
    }

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

    let mut a_mat = Matrix::zeros(n_cphf, n_cphf);
    let mut x_vec = vec![0.0; n_cphf];

    for a in 0..n_virt_a {
        let a_mo = n_alpha + a;
        for i in 0..n_alpha {
            let row = a * n_alpha + i;
            a_mat[(row, row)] += eps_a[a_mo] - eps_a[i];
            for b in 0..n_virt_a {
                let b_mo = n_alpha + b;
                for j in 0..n_alpha {
                    let col = b * n_alpha + j;
                    a_mat[(row, col)] += 2.0 * mo_eri_a.get(a_mo, i, b_mo, j)
                        - mo_eri_a.get(a_mo, b_mo, j, i)
                        - mo_eri_a.get(a_mo, j, b_mo, i);
                }
            }
            for b in 0..n_virt_b {
                let b_mo = n_beta + b;
                for j in 0..n_beta {
                    let col = n_cphf_a + b * n_beta + j;
                    let elem = 2.0 * cross_eri.get(a_mo, i, b_mo, j);
                    a_mat[(row, col)] += elem;
                    a_mat[(col, row)] += elem;
                }
            }
        }
    }
    for a in 0..n_virt_b {
        let a_mo = n_beta + a;
        for i in 0..n_beta {
            let row = n_cphf_a + a * n_beta + i;
            a_mat[(row, row)] += eps_b[a_mo] - eps_b[i];
            for b in 0..n_virt_b {
                let b_mo = n_beta + b;
                for j in 0..n_beta {
                    let col = n_cphf_a + b * n_beta + j;
                    a_mat[(row, col)] += 2.0 * mo_eri_b.get(a_mo, i, b_mo, j)
                        - mo_eri_b.get(a_mo, b_mo, j, i)
                        - mo_eri_b.get(a_mo, j, b_mo, i);
                }
            }
        }
    }

    for a in 0..n_virt_a {
        let a_mo = n_alpha + a;
        for i in 0..n_alpha {
            let row = a * n_alpha + i;
            let mut x_val = 0.0;
            for j in 0..n_alpha {
                for b in 0..n_virt_a {
                    let b_mo = n_alpha + b;
                    for c in 0..n_virt_a {
                        let c_mo = n_alpha + c;
                        x_val -= t2_aa[i][j][b][c] * mo_eri_a.get(a_mo, b_mo, j, c_mo);
                    }
                }
                for k in 0..n_alpha {
                    for b in 0..n_virt_a {
                        let b_mo = n_alpha + b;
                        x_val += t2_aa[j][k][a][b] * mo_eri_a.get(j, i, k, b_mo);
                    }
                }
            }
            for j in 0..n_beta {
                for b in 0..n_virt_b {
                    let b_mo = n_beta + b;
                    for c in 0..n_virt_a {
                        let c_mo = n_alpha + c;
                        x_val -= t2_ab[i][j][c][b] * cross_eri.get(a_mo, c_mo, j, b_mo);
                    }
                }
            }
            for k in 0..n_alpha {
                for j in 0..n_beta {
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        x_val += t2_ab[k][j][a][b] * cross_eri.get(k, i, j, b_mo);
                    }
                }
            }
            for j in 0..n_alpha {
                for k in 0..n_alpha {
                    let d_jk = d_occ_a[(j, k)];
                    if d_jk.abs() > 1e-15 {
                        x_val -= 0.5
                            * d_jk
                            * (2.0 * mo_eri_a.get(a_mo, i, j, k)
                                - mo_eri_a.get(a_mo, j, i, k)
                                - mo_eri_a.get(a_mo, k, i, j));
                    }
                }
            }
            for b in 0..n_virt_a {
                let b_mo = n_alpha + b;
                for c in 0..n_virt_a {
                    let c_mo = n_alpha + c;
                    let d_bc = d_virt_a[(b, c)];
                    if d_bc.abs() > 1e-15 {
                        x_val -= 0.5
                            * d_bc
                            * (2.0 * mo_eri_a.get(a_mo, i, b_mo, c_mo)
                                - mo_eri_a.get(a_mo, b_mo, i, c_mo)
                                - mo_eri_a.get(a_mo, c_mo, i, b_mo));
                    }
                }
            }
            for j in 0..n_beta {
                for k in 0..n_beta {
                    let d_jk = d_occ_b[(j, k)];
                    if d_jk.abs() > 1e-15 {
                        x_val -= d_jk * cross_eri.get(a_mo, i, j, k);
                    }
                }
            }
            for b in 0..n_virt_b {
                let b_mo = n_beta + b;
                for c in 0..n_virt_b {
                    let c_mo = n_beta + c;
                    let d_bc = d_virt_b[(b, c)];
                    if d_bc.abs() > 1e-15 {
                        x_val -= d_bc * cross_eri.get(a_mo, i, b_mo, c_mo);
                    }
                }
            }
            x_vec[row] = x_val;
        }
    }

    for a in 0..n_virt_b {
        let a_mo = n_beta + a;
        for i in 0..n_beta {
            let row = n_cphf_a + a * n_beta + i;
            let mut x_val = 0.0;
            for j in 0..n_beta {
                for b in 0..n_virt_b {
                    let b_mo = n_beta + b;
                    for c in 0..n_virt_b {
                        let c_mo = n_beta + c;
                        x_val -= t2_bb[i][j][b][c] * mo_eri_b.get(a_mo, b_mo, j, c_mo);
                    }
                }
                for k in 0..n_beta {
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        x_val += t2_bb[j][k][a][b] * mo_eri_b.get(j, i, k, b_mo);
                    }
                }
            }
            for j in 0..n_alpha {
                for b in 0..n_virt_a {
                    let b_mo = n_alpha + b;
                    for c in 0..n_virt_b {
                        let c_mo = n_beta + c;
                        x_val -= t2_ab[j][i][b][c] * cross_eri.get(j, b_mo, a_mo, c_mo);
                    }
                    for k in 0..n_beta {
                        x_val += t2_ab[j][k][b][a] * cross_eri.get(j, b_mo, k, i);
                    }
                }
            }
            for j in 0..n_beta {
                for k in 0..n_beta {
                    let d_jk = d_occ_b[(j, k)];
                    if d_jk.abs() > 1e-15 {
                        x_val -= 0.5
                            * d_jk
                            * (2.0 * mo_eri_b.get(a_mo, i, j, k)
                                - mo_eri_b.get(a_mo, j, i, k)
                                - mo_eri_b.get(a_mo, k, i, j));
                    }
                }
            }
            for b in 0..n_virt_b {
                let b_mo = n_beta + b;
                for c in 0..n_virt_b {
                    let c_mo = n_beta + c;
                    let d_bc = d_virt_b[(b, c)];
                    if d_bc.abs() > 1e-15 {
                        x_val -= 0.5
                            * d_bc
                            * (2.0 * mo_eri_b.get(a_mo, i, b_mo, c_mo)
                                - mo_eri_b.get(a_mo, b_mo, i, c_mo)
                                - mo_eri_b.get(a_mo, c_mo, i, b_mo));
                    }
                }
            }
            for j in 0..n_alpha {
                for k in 0..n_alpha {
                    let d_jk = d_occ_a[(j, k)];
                    if d_jk.abs() > 1e-15 {
                        x_val -= d_jk * cross_eri.get(j, k, a_mo, i);
                    }
                }
            }
            for b in 0..n_virt_a {
                let b_mo = n_alpha + b;
                for c in 0..n_virt_a {
                    let c_mo = n_alpha + c;
                    let d_bc = d_virt_a[(b, c)];
                    if d_bc.abs() > 1e-15 {
                        x_val -= d_bc * cross_eri.get(b_mo, c_mo, a_mo, i);
                    }
                }
            }
            x_vec[row] = x_val;
        }
    }

    let lu = a_mat.lu_decompose().expect("CPUHF matrix is invertible in UHF-MP2 Hessian");
    let z_sol = lu.solve(&x_vec);

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
        for i in 0..n_alpha {
            let z_ai = z_sol[a * n_alpha + i];
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
        for i in 0..n_beta {
            let z_ai = z_sol[n_cphf_a + a * n_beta + i];
            p_corr_mo_b[(a_mo, i)] = z_ai;
            p_corr_mo_b[(i, a_mo)] = z_ai;
        }
    }

    let p_corr_ao_a = c_a.dot(&p_corr_mo_a).dot(&c_a.transpose());
    let p_corr_ao_b = c_b.dot(&p_corr_mo_b).dot(&c_b.transpose());
    let p_corr_tot = p_corr_ao_a.add(&p_corr_ao_b);

    let mut w_corr_mo_a = Matrix::zeros(n_basis, n_basis);
    for i in 0..n_alpha {
        for j in 0..n_alpha {
            let mut sum_w = 0.5 * (eps_a[i] + eps_a[j]) * d_occ_a[(i, j)];
            for k in 0..n_alpha {
                for a in 0..n_virt_a {
                    let a_mo = n_alpha + a;
                    for b in 0..n_virt_a {
                        let b_mo = n_alpha + b;
                        sum_w += t2_aa[i][k][a][b] * mo_eri_a.get(j, a_mo, k, b_mo);
                    }
                }
            }
            for k in 0..n_beta {
                for a in 0..n_virt_a {
                    let a_mo = n_alpha + a;
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        sum_w += t2_ab[i][k][a][b] * cross_eri.get(j, a_mo, k, b_mo);
                    }
                }
            }
            for p in 0..n_basis {
                for q in 0..n_basis {
                    let pa = p_corr_mo_a[(p, q)];
                    if pa.abs() > 1e-15 {
                        sum_w += pa * (mo_eri_a.get(i, j, p, q) - mo_eri_a.get(i, p, j, q));
                    }
                    let pb = p_corr_mo_b[(p, q)];
                    if pb.abs() > 1e-15 {
                        sum_w += pb * cross_eri.get(i, j, p, q);
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
                        sum_w += t2_aa[i][j][a][c] * mo_eri_a.get(i, b_mo, j, c_mo);
                    }
                }
                for j in 0..n_beta {
                    for c in 0..n_virt_b {
                        let c_mo = n_beta + c;
                        sum_w += t2_ab[i][j][a][c] * cross_eri.get(i, b_mo, j, c_mo);
                    }
                }
            }
            w_corr_mo_a[(a_mo, b_mo)] = sum_w;
        }
        for i in 0..n_alpha {
            let z_ai = z_sol[a * n_alpha + i];
            let mut sum_ia = 0.0;
            for j in 0..n_alpha {
                for k in 0..n_alpha {
                    for b in 0..n_virt_a {
                        let b_mo = n_alpha + b;
                        sum_ia += t2_aa[j][k][a][b] * mo_eri_a.get(j, i, k, b_mo);
                    }
                }
            }
            for j in 0..n_beta {
                for k in 0..n_alpha {
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        sum_ia += t2_ab[k][j][a][b] * cross_eri.get(k, i, j, b_mo);
                    }
                }
            }
            let w_val = eps_a[i] * z_ai + sum_ia;
            w_corr_mo_a[(i, a_mo)] = w_val;
            w_corr_mo_a[(a_mo, i)] = w_val;
        }
    }

    let mut w_corr_mo_b = Matrix::zeros(n_basis, n_basis);
    for i in 0..n_beta {
        for j in 0..n_beta {
            let mut sum_w = 0.5 * (eps_b[i] + eps_b[j]) * d_occ_b[(i, j)];
            for k in 0..n_beta {
                for a in 0..n_virt_b {
                    let a_mo = n_beta + a;
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        sum_w += t2_bb[i][k][a][b] * mo_eri_b.get(j, a_mo, k, b_mo);
                    }
                }
            }
            for k in 0..n_alpha {
                for a in 0..n_virt_a {
                    let a_mo = n_alpha + a;
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        sum_w += t2_ab[k][i][a][b] * cross_eri.get(k, a_mo, j, b_mo);
                    }
                }
            }
            for p in 0..n_basis {
                for q in 0..n_basis {
                    let pb = p_corr_mo_b[(p, q)];
                    if pb.abs() > 1e-15 {
                        sum_w += pb * (mo_eri_b.get(i, j, p, q) - mo_eri_b.get(i, p, j, q));
                    }
                    let pa = p_corr_mo_a[(p, q)];
                    if pa.abs() > 1e-15 {
                        sum_w += pa * cross_eri.get(p, q, i, j);
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
                        sum_w += t2_bb[i][j][a][c] * mo_eri_b.get(i, b_mo, j, c_mo);
                    }
                }
            }
            for i in 0..n_alpha {
                for j in 0..n_beta {
                    for c in 0..n_virt_a {
                        let c_mo = n_alpha + c;
                        sum_w += t2_ab[i][j][c][a] * cross_eri.get(i, c_mo, j, b_mo);
                    }
                }
            }
            w_corr_mo_b[(a_mo, b_mo)] = sum_w;
        }
        for i in 0..n_beta {
            let z_ai = z_sol[n_cphf_a + a * n_beta + i];
            let mut sum_ia = 0.0;
            for j in 0..n_beta {
                for k in 0..n_beta {
                    for b in 0..n_virt_b {
                        let b_mo = n_beta + b;
                        sum_ia += t2_bb[j][k][a][b] * mo_eri_b.get(j, i, k, b_mo);
                    }
                }
            }
            for j in 0..n_alpha {
                for k in 0..n_beta {
                    for b in 0..n_virt_a {
                        let b_mo = n_alpha + b;
                        sum_ia += t2_ab[j][k][b][a] * cross_eri.get(j, b_mo, k, i);
                    }
                }
            }
            let w_val = eps_b[i] * z_ai + sum_ia;
            w_corr_mo_b[(i, a_mo)] = w_val;
            w_corr_mo_b[(a_mo, i)] = w_val;
        }
    }

    let mut w_symm_a = Matrix::zeros(n_basis, n_basis);
    let mut w_symm_b = Matrix::zeros(n_basis, n_basis);
    for p in 0..n_basis {
        for q in 0..n_basis {
            w_symm_a[(p, q)] = 0.5 * (w_corr_mo_a[(p, q)] + w_corr_mo_a[(q, p)]);
            w_symm_b[(p, q)] = 0.5 * (w_corr_mo_b[(p, q)] + w_corr_mo_b[(q, p)]);
        }
    }
    let w_corr_ao_a = c_a.dot(&w_symm_a).dot(&c_a.transpose());
    let w_corr_ao_b = c_b.dot(&w_symm_b).dot(&c_b.transpose());
    let w_corr_tot = w_corr_ao_a.add(&w_corr_ao_b);

    // Add correlated 2-PDM and relaxation coupling to gamma_raw
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
                                    let t_perm = c_a[(mu, i)] * c_a[(nu, a_mo)] * c_a[(la, j)] * c_a[(si, b_mo)]
                                        + c_a[(nu, i)] * c_a[(mu, a_mo)] * c_a[(la, j)] * c_a[(si, b_mo)]
                                        + c_a[(la, i)] * c_a[(si, a_mo)] * c_a[(mu, j)] * c_a[(nu, b_mo)]
                                        + c_a[(la, i)] * c_a[(si, a_mo)] * c_a[(nu, j)] * c_a[(mu, b_mo)];
                                    gamma_raw[idx4(mu, nu, la, si)] += val * t_perm;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

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
                                    let t_perm = c_b[(mu, i)] * c_b[(nu, a_mo)] * c_b[(la, j)] * c_b[(si, b_mo)]
                                        + c_b[(nu, i)] * c_b[(mu, a_mo)] * c_b[(la, j)] * c_b[(si, b_mo)]
                                        + c_b[(la, i)] * c_b[(si, a_mo)] * c_b[(mu, j)] * c_b[(nu, b_mo)]
                                        + c_b[(la, i)] * c_b[(si, a_mo)] * c_b[(nu, j)] * c_b[(mu, b_mo)];
                                    gamma_raw[idx4(mu, nu, la, si)] += val * t_perm;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

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
                                    let t_perm = c_a[(mu, i)] * c_a[(nu, a_mo)] * c_b[(la, j)] * c_b[(si, b_mo)]
                                        + c_a[(nu, i)] * c_a[(mu, a_mo)] * c_b[(la, j)] * c_b[(si, b_mo)]
                                        + c_b[(mu, j)] * c_b[(nu, b_mo)] * c_a[(la, i)] * c_a[(si, a_mo)]
                                        + c_b[(nu, j)] * c_b[(mu, b_mo)] * c_a[(la, i)] * c_a[(si, a_mo)];
                                    gamma_raw[idx4(mu, nu, la, si)] += val * t_perm;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for la in 0..n_basis {
                for si in 0..n_basis {
                    let g_relax = 2.0
                        * (p_uhf_tot[(mu, nu)] * p_corr_tot[(la, si)]
                            + p_corr_tot[(mu, nu)] * p_uhf_tot[(la, si)])
                        - 2.0
                            * (p_uhf_a[(mu, la)] * p_corr_ao_a[(nu, si)]
                                + p_corr_ao_a[(mu, la)] * p_uhf_a[(nu, si)]
                                + p_uhf_b[(mu, la)] * p_corr_ao_b[(nu, si)]
                                + p_corr_ao_b[(mu, la)] * p_uhf_b[(nu, si)]);
                    gamma_raw[idx4(mu, nu, la, si)] += g_relax;
                }
            }
        }
    }

    let mut gamma_symm = vec![0.0; n_basis * n_basis * n_basis * n_basis];
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for la in 0..n_basis {
                for si in 0..n_basis {
                    let s = gamma_raw[idx4(mu, nu, la, si)]
                        + gamma_raw[idx4(nu, mu, la, si)]
                        + gamma_raw[idx4(mu, nu, si, la)]
                        + gamma_raw[idx4(nu, mu, si, la)]
                        + gamma_raw[idx4(la, si, mu, nu)]
                        + gamma_raw[idx4(si, la, mu, nu)]
                        + gamma_raw[idx4(la, si, nu, mu)]
                        + gamma_raw[idx4(si, la, nu, mu)];
                    gamma_symm[idx4(mu, nu, la, si)] = 0.0625 * s;
                }
            }
        }
    }

    (p_uhf_tot.add(&p_corr_tot), w_uhf_tot.add(&w_corr_tot), gamma_symm)
}

/// Computes the pure analytical nuclear Hessian for open-shell UHF-MP2.
pub fn compute_uhf_mp2_analytical_hessian(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    max_iter: usize,
    tol: f64,
) -> Matrix {
    let basis = build_basis(atoms, basis_map);

    let uhf = run_uhf(
        atoms,
        &basis,
        charge,
        multiplicity,
        is_broken_symmetry,
        max_iter,
        tol,
    )
    .expect("Reference UHF failed in analytical UHF-MP2 Hessian");

    compute_uhf_mp2_hessian_from_reference(atoms, &basis, &uhf)
}

/// Analytic UHF-MP2 Hessian reusing the converged full Cartesian reference.
pub fn compute_uhf_mp2_hessian_from_reference(
    atoms: &[Atom], basis: &[crate::basis::ContractedGaussian], uhf: &UHFResult,
) -> Matrix {
    let n_atoms = atoms.len();
    let dim = 3 * n_atoms;
    let n_basis = basis.len();

    let n_alpha = uhf.n_alpha;
    let n_beta = uhf.n_beta;
    let n_virt_a = n_basis - n_alpha;
    let n_virt_b = n_basis - n_beta;
    let c_a = &uhf.mo_coefficients_alpha;
    let c_b = &uhf.mo_coefficients_beta;
    let p_a = &uhf.density_matrix_alpha;
    let p_b = &uhf.density_matrix_beta;
    let p_tot = p_a.add(p_b);
    let eps_a = &uhf.orbital_energies_alpha;
    let eps_b = &uhf.orbital_energies_beta;

    let (p_tot_0, w_tot_0, gamma_tot_0) = build_uhf_mp2_total_ao_densities(uhf);

    let mut d0 = Vec::with_capacity(n_basis);
    let mut d1 = Vec::with_capacity(n_basis);
    let mut d2 = Vec::with_capacity(n_basis);

    for cg in basis {
        d0.push(GeneralBasisDerivative::from_single(cg.clone()));
        let mut d1_cg = Vec::with_capacity(3);
        for tau in 0..3 {
            d1_cg.push(diff_cg_1st(cg, tau));
        }
        d1.push(d1_cg);

        let mut d2_cg = vec![vec![GeneralBasisDerivative::empty(); 3]; 3];
        for tau in 0..3 {
            for eta in 0..3 {
                d2_cg[tau][eta] = diff_cg_2nd(cg, tau, eta);
            }
        }
        d2.push(d2_cg);
    }

    let perts: Vec<(usize, usize, usize)> = (0..n_atoms)
        .flat_map(|a| (0..3).map(move |tau| (3 * a + tau, a, tau)))
        .collect();

    let ao_1st: Vec<(Matrix, Matrix, Matrix, Matrix, Vec<f64>)> = perts
        .par_iter()
        .map(|&(_pert, a, tau)| {
            let mut s_mat = Matrix::zeros(n_basis, n_basis);
            let mut h_mat = Matrix::zeros(n_basis, n_basis);
            let mut fa_mat = Matrix::zeros(n_basis, n_basis);
            let mut fb_mat = Matrix::zeros(n_basis, n_basis);

            for mu in 0..n_basis {
                let mu_on_a = basis[mu].atom_index == a;
                for nu in 0..n_basis {
                    let nu_on_a = basis[nu].atom_index == a;
                    let mut ds = 0.0;
                    let mut dt = 0.0;
                    if mu_on_a {
                        ds += eval_overlap_deriv(&d1[mu][tau], &d0[nu]);
                        dt += eval_kinetic_deriv(&d1[mu][tau], &d0[nu]);
                    }
                    if nu_on_a {
                        ds += eval_overlap_deriv(&d0[mu], &d1[nu][tau]);
                        dt += eval_kinetic_deriv(&d0[mu], &d1[nu][tau]);
                    }
                    s_mat[(mu, nu)] = ds;

                    let mut dv = 0.0;
                    for c_idx in 0..n_atoms {
                        let ca: f64 = (if mu_on_a { 1.0 } else { 0.0 })
                            - (if c_idx == a { 1.0 } else { 0.0 });
                        let cb: f64 = (if nu_on_a { 1.0 } else { 0.0 })
                            - (if c_idx == a { 1.0 } else { 0.0 });
                        if ca.abs() > 1e-12 {
                            dv += ca * eval_nuclear_deriv_single_atom(&d1[mu][tau], &d0[nu], &atoms[c_idx]);
                        }
                        if cb.abs() > 1e-12 {
                            dv += cb * eval_nuclear_deriv_single_atom(&d0[mu], &d1[nu][tau], &atoms[c_idx]);
                        }
                    }
                    h_mat[(mu, nu)] = dt + dv;
                    fa_mat[(mu, nu)] = dt + dv;
                    fb_mat[(mu, nu)] = dt + dv;
                }
            }

            let mut d_eri_ao = vec![0.0; uhf.eri.len()];
            for mu in 0..n_basis {
                let mu_on_a = basis[mu].atom_index == a;
                for nu in 0..=mu {
                    let nu_on_a = basis[nu].atom_index == a;
                    for la in 0..n_basis {
                        let la_on_a = basis[la].atom_index == a;
                        for si in 0..=la {
                            let si_on_a = basis[si].atom_index == a;
                            if !mu_on_a && !nu_on_a && !la_on_a && !si_on_a {
                                continue;
                            }
                            let idx = eri_index(mu, nu, la, si);
                            let mut val = 0.0;
                            if mu_on_a {
                                val += eval_eri_deriv(&d1[mu][tau], &d0[nu], &d0[la], &d0[si]);
                            }
                            if nu_on_a {
                                val += eval_eri_deriv(&d0[mu], &d1[nu][tau], &d0[la], &d0[si]);
                            }
                            if la_on_a {
                                val += eval_eri_deriv(&d0[mu], &d0[nu], &d1[la][tau], &d0[si]);
                            }
                            if si_on_a {
                                val += eval_eri_deriv(&d0[mu], &d0[nu], &d0[la], &d1[si][tau]);
                            }
                            d_eri_ao[idx] = val;
                        }
                    }
                }
            }

            for mu in 0..n_basis {
                for nu in 0..n_basis {
                    let mut ga_val = 0.0;
                    let mut gb_val = 0.0;
                    for la in 0..n_basis {
                        for si in 0..n_basis {
                            let d_coul = d_eri_ao[eri_index(mu, nu, la, si)];
                            let d_exch = d_eri_ao[eri_index(mu, la, nu, si)];
                            ga_val += p_tot[(la, si)] * d_coul - p_a[(la, si)] * d_exch;
                            gb_val += p_tot[(la, si)] * d_coul - p_b[(la, si)] * d_exch;
                        }
                    }
                    fa_mat[(mu, nu)] += ga_val;
                    fb_mat[(mu, nu)] += gb_val;
                }
            }

            (s_mat, h_mat, fa_mat, fb_mat, d_eri_ao)
        })
        .collect();

    let cta = c_a.transpose();
    let ctb = c_b.transpose();
    let s_1st_mo_a: Vec<Matrix> = ao_1st.iter().map(|(s, _, _, _, _)| cta.dot(s).dot(c_a)).collect();
    let s_1st_mo_b: Vec<Matrix> = ao_1st.iter().map(|(s, _, _, _, _)| ctb.dot(s).dot(c_b)).collect();
    let f_skel_1st_mo_a: Vec<Matrix> = ao_1st.iter().map(|(_, _, fa, _, _)| cta.dot(fa).dot(c_a)).collect();
    let f_skel_1st_mo_b: Vec<Matrix> = ao_1st.iter().map(|(_, _, _, fb, _)| ctb.dot(fb).dot(c_b)).collect();

    let (u_a_list, u_b_list, f1_a_list, f1_b_list) =
        solve_cpuhf(uhf, &s_1st_mo_a, &s_1st_mo_b, &f_skel_1st_mo_a, &f_skel_1st_mo_b);

    let eta = 1e-4;
    let density_1st_derivs: Vec<(Matrix, Matrix, Vec<f64>)> = (0..dim)
        .into_par_iter()
        .map(|y| {
            let (ref s_1st, ref h_1st, _, _, ref d_eri) = ao_1st[y];
            let mut ua_canon = u_a_list[y].clone();
            let mut ub_canon = u_b_list[y].clone();
            let f1_a = &f1_a_list[y];
            let f1_b = &f1_b_list[y];

            for i in 0..n_alpha {
                for j in 0..n_alpha {
                    if i != j && (eps_a[j] - eps_a[i]).abs() > 1e-7 {
                        ua_canon[(i, j)] += f1_a[(i, j)] / (eps_a[j] - eps_a[i]);
                    }
                }
            }
            for a in 0..n_virt_a {
                let a_mo = n_alpha + a;
                for b in 0..n_virt_a {
                    let b_mo = n_alpha + b;
                    if a != b && (eps_a[b_mo] - eps_a[a_mo]).abs() > 1e-7 {
                        ua_canon[(a_mo, b_mo)] += f1_a[(a_mo, b_mo)] / (eps_a[b_mo] - eps_a[a_mo]);
                    }
                }
            }
            for i in 0..n_beta {
                for j in 0..n_beta {
                    if i != j && (eps_b[j] - eps_b[i]).abs() > 1e-7 {
                        ub_canon[(i, j)] += f1_b[(i, j)] / (eps_b[j] - eps_b[i]);
                    }
                }
            }
            for a in 0..n_virt_b {
                let a_mo = n_beta + a;
                for b in 0..n_virt_b {
                    let b_mo = n_beta + b;
                    if a != b && (eps_b[b_mo] - eps_b[a_mo]).abs() > 1e-7 {
                        ub_canon[(a_mo, b_mo)] += f1_b[(a_mo, b_mo)] / (eps_b[b_mo] - eps_b[a_mo]);
                    }
                }
            }

            let build_tangent_uhf = |step: f64| -> UHFResult {
                let s_step = uhf.overlap_matrix.add(&s_1st.scale(step));
                let h_step = uhf.core_hamiltonian.add(&h_1st.scale(step));
                let eri_step: Vec<f64> = uhf
                    .eri
                    .iter()
                    .zip(d_eri.iter())
                    .map(|(&e0, &de)| e0 + step * de)
                    .collect();

                let ident = Matrix::identity(n_basis);
                let ca_step = c_a.dot(&ident.add(&ua_canon.scale(step)));
                let cb_step = c_b.dot(&ident.add(&ub_canon.scale(step)));
                let eps_a_step: Vec<f64> = (0..n_basis)
                    .map(|p_idx| eps_a[p_idx] + step * f1_a[(p_idx, p_idx)])
                    .collect();
                let eps_b_step: Vec<f64> = (0..n_basis)
                    .map(|p_idx| eps_b[p_idx] + step * f1_b[(p_idx, p_idx)])
                    .collect();

                let mut pa_step = Matrix::zeros(n_basis, n_basis);
                let mut pb_step = Matrix::zeros(n_basis, n_basis);
                for mu in 0..n_basis {
                    for nu in 0..n_basis {
                        let mut sa_val = 0.0;
                        for i in 0..n_alpha {
                            sa_val += ca_step[(mu, i)] * ca_step[(nu, i)];
                        }
                        let mut sb_val = 0.0;
                        for i in 0..n_beta {
                            sb_val += cb_step[(mu, i)] * cb_step[(nu, i)];
                        }
                        pa_step[(mu, nu)] = sa_val;
                        pb_step[(mu, nu)] = sb_val;
                    }
                }
                let (fa_step, fb_step) = build_uhf_fock_matrices(&h_step, &pa_step, &pb_step, &eri_step, n_basis);

                UHFResult {
        ao_space: None,
                    orbital_energies_alpha: eps_a_step,
                    orbital_energies_beta: eps_b_step,
                    mo_coefficients_alpha: ca_step,
                    mo_coefficients_beta: cb_step,
                    density_matrix_alpha: pa_step,
                    density_matrix_beta: pb_step,
                    fock_matrix_alpha: fa_step,
                    fock_matrix_beta: fb_step,
                    overlap_matrix: s_step,
                    core_hamiltonian: h_step,
                    eri: eri_step,
                    ..uhf.clone()
                }
            };

            let (p_p, w_p, g_p) = build_uhf_mp2_total_ao_densities(&build_tangent_uhf(eta));
            let (p_m, w_m, g_m) = build_uhf_mp2_total_ao_densities(&build_tangent_uhf(-eta));

            let inv_2eta = 1.0 / (2.0 * eta);
            let dp = p_p.sub(&p_m).scale(inv_2eta);
            let dw = w_p.sub(&w_m).scale(inv_2eta);
            let dg: Vec<f64> = g_p.iter().zip(g_m.iter()).map(|(&a, &b)| (a - b) * inv_2eta).collect();
            (dp, dw, dg)
        })
        .collect();

    let idx4 = |mu: usize, nu: usize, la: usize, si: usize| -> usize {
        ((mu * n_basis + nu) * n_basis + la) * n_basis + si
    };

    let mut sig_pairs = Vec::new();
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            let pv = p_tot_0[(mu, nu)];
            let wv = w_tot_0[(mu, nu)];
            if pv.abs() > 1e-15 || wv.abs() > 1e-15 {
                sig_pairs.push((mu, nu, pv, wv));
            }
        }
    }

    let mut sig_quartets = Vec::new();
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for la in 0..n_basis {
                for si in 0..n_basis {
                    let gv = gamma_tot_0[idx4(mu, nu, la, si)];
                    if gv.abs() > 1e-15 {
                        sig_quartets.push((mu, nu, la, si, gv));
                    }
                }
            }
        }
    }

    let mut coord_pairs = Vec::new();
    for x in 0..dim {
        for y in 0..=x {
            coord_pairs.push((x, y));
        }
    }

    let pair_elements: Vec<((usize, usize), f64)> = coord_pairs
        .into_par_iter()
        .map(|(x, y)| {
            let a = x / 3;
            let tau = x % 3;
            let b = y / 3;
            let eta_c = y % 3;

            let mut e1_xy = 0.0;
            for &(mu, nu, p_val, w_val) in &sig_pairs {
                let mu_a = basis[mu].atom_index == a;
                let nu_a = basis[nu].atom_index == a;
                let mu_b = basis[mu].atom_index == b;
                let nu_b = basis[nu].atom_index == b;

                let mut s_xy = 0.0;
                let mut t_xy = 0.0;
                if mu_a && mu_b {
                    s_xy += eval_overlap_deriv(&d2[mu][tau][eta_c], &d0[nu]);
                    t_xy += eval_kinetic_deriv(&d2[mu][tau][eta_c], &d0[nu]);
                }
                if mu_a && nu_b {
                    s_xy += eval_overlap_deriv(&d1[mu][tau], &d1[nu][eta_c]);
                    t_xy += eval_kinetic_deriv(&d1[mu][tau], &d1[nu][eta_c]);
                }
                if nu_a && mu_b {
                    s_xy += eval_overlap_deriv(&d1[nu][tau], &d1[mu][eta_c]);
                    t_xy += eval_kinetic_deriv(&d1[nu][tau], &d1[mu][eta_c]);
                }
                if nu_a && nu_b {
                    s_xy += eval_overlap_deriv(&d0[mu], &d2[nu][tau][eta_c]);
                    t_xy += eval_kinetic_deriv(&d0[mu], &d2[nu][tau][eta_c]);
                }

                let mut v_xy = 0.0;
                for c_idx in 0..n_atoms {
                    let ca1: f64 = (if mu_a { 1.0 } else { 0.0 }) - (if c_idx == a { 1.0 } else { 0.0 });
                    let cb1: f64 = (if nu_a { 1.0 } else { 0.0 }) - (if c_idx == a { 1.0 } else { 0.0 });
                    let ca2: f64 = (if mu_b { 1.0 } else { 0.0 }) - (if c_idx == b { 1.0 } else { 0.0 });
                    let cb2: f64 = (if nu_b { 1.0 } else { 0.0 }) - (if c_idx == b { 1.0 } else { 0.0 });

                    if ca1.abs() > 1e-12 && ca2.abs() > 1e-12 {
                        v_xy += ca1 * ca2 * eval_nuclear_deriv_single_atom(&d2[mu][tau][eta_c], &d0[nu], &atoms[c_idx]);
                    }
                    if ca1.abs() > 1e-12 && cb2.abs() > 1e-12 {
                        v_xy += ca1 * cb2 * eval_nuclear_deriv_single_atom(&d1[mu][tau], &d1[nu][eta_c], &atoms[c_idx]);
                    }
                    if cb1.abs() > 1e-12 && ca2.abs() > 1e-12 {
                        v_xy += cb1 * ca2 * eval_nuclear_deriv_single_atom(&d1[nu][tau], &d1[mu][eta_c], &atoms[c_idx]);
                    }
                    if cb1.abs() > 1e-12 && cb2.abs() > 1e-12 {
                        v_xy += cb1 * cb2 * eval_nuclear_deriv_single_atom(&d0[mu], &d2[nu][tau][eta_c], &atoms[c_idx]);
                    }
                }

                e1_xy += p_val * (t_xy + v_xy) - w_val * s_xy;
            }

            let mut e2_xy = 0.0;
            for &(mu, nu, la, si, gamma) in &sig_quartets {
                let at = [
                    basis[mu].atom_index,
                    basis[nu].atom_index,
                    basis[la].atom_index,
                    basis[si].atom_index,
                ];
                let g = [mu, nu, la, si];
                for k in 0..4 {
                    if at[k] != a {
                        continue;
                    }
                    for l in 0..4 {
                        if at[l] != b {
                            continue;
                        }
                        let d_eri = if k == l {
                            let mut d_slots = [&d0[g[0]], &d0[g[1]], &d0[g[2]], &d0[g[3]]];
                            d_slots[k] = &d2[g[k]][tau][eta_c];
                            eval_eri_deriv(d_slots[0], d_slots[1], d_slots[2], d_slots[3])
                        } else {
                            let mut d_slots = [&d0[g[0]], &d0[g[1]], &d0[g[2]], &d0[g[3]]];
                            d_slots[k] = &d1[g[k]][tau];
                            d_slots[l] = &d1[g[l]][eta_c];
                            eval_eri_deriv(d_slots[0], d_slots[1], d_slots[2], d_slots[3])
                        };
                        e2_xy += 0.5 * gamma * d_eri;
                    }
                }
            }

            let contract_resp = |d_idx: usize, i_idx: usize| -> f64 {
                let (ref dp, ref dw, ref dg) = density_1st_derivs[d_idx];
                let (ref s1, ref h1, _, _, ref eri1) = ao_1st[i_idx];
                let mut sum_1e = 0.0;
                for mu in 0..n_basis {
                    for nu in 0..n_basis {
                        sum_1e += dp[(mu, nu)] * h1[(mu, nu)] - dw[(mu, nu)] * s1[(mu, nu)];
                    }
                }
                let mut sum_2e = 0.0;
                for mu in 0..n_basis {
                    for nu in 0..n_basis {
                        for la in 0..n_basis {
                            for si in 0..n_basis {
                                let g_val = dg[idx4(mu, nu, la, si)];
                                if g_val.abs() > 1e-15 {
                                    sum_2e += 0.5 * g_val * eri1[eri_index(mu, nu, la, si)];
                                }
                            }
                        }
                    }
                }
                sum_1e + sum_2e
            };

            let e_resp = 0.5 * (contract_resp(y, x) + contract_resp(x, y));
            ((x, y), e1_xy + e2_xy + e_resp)
        })
        .collect();

    let mut hess = compute_nuclear_repulsion_hessian(atoms);
    for ((x, y), val) in pair_elements {
        hess[(x, y)] += val;
        if x != y {
            hess[(y, x)] += val;
        }
    }

    hess
}
