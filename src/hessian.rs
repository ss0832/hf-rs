// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Analytical Nuclear Hessian (2nd derivatives) and Harmonic Vibrational Analysis.
//!
//! Evaluates exact analytical second derivatives using Coupled-Perturbed Hartree-Fock (CPHF)
//! theory (Pople et al. 1979, Handy et al. 1984) and McMurchie-Davidson analytical integral derivatives.

use rayon::prelude::*;
use crate::basis::{Atom, BasisSetMap, build_basis, ContractedGaussian};
use crate::cphf::solve_cphf;
use crate::derivatives::{
    diff_cg_1st, diff_cg_2nd, eval_eri_deriv, eval_kinetic_deriv, eval_nuclear_deriv_single_atom,
    eval_overlap_deriv, GeneralBasisDerivative,
};
use crate::gradient::{compute_energy_weighted_density, compute_rhf_gradient};
use crate::matrix::Matrix;
use crate::scf::HFResult;

/// Computes the exact 2nd derivative of the nuclear repulsion energy: d^2 E_nuc / dR_x dR_y
pub fn compute_nuclear_repulsion_hessian(atoms: &[Atom]) -> Matrix {
    let n_atoms = atoms.len();
    let dim = 3 * n_atoms;
    let mut hess = Matrix::zeros(dim, dim);

    for a in 0..n_atoms {
        for b in 0..n_atoms {
            if a == b {
                continue;
            }
            let za_zb = atoms[a].charge * atoms[b].charge;
            let dx = atoms[a].position[0] - atoms[b].position[0];
            let dy = atoms[a].position[1] - atoms[b].position[1];
            let dz = atoms[a].position[2] - atoms[b].position[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let dist3 = dist * dist * dist;
            let dist5 = dist3 * dist * dist;
            let delta = [dx, dy, dz];

            for tau in 0..3 {
                for eta in 0..3 {
                    let kron = if tau == eta { 1.0 } else { 0.0 };
                    let val = za_zb * (kron / dist3 - 3.0 * delta[tau] * delta[eta] / dist5);
                    let row = 3 * a + tau;
                    let col = 3 * b + eta;
                    hess[(row, col)] = val;
                }
            }
        }
    }

    // Diagonal atom blocks via translational invariance: \sum_B d^2 E / dR_A dR_B = 0
    for a in 0..n_atoms {
        for tau in 0..3 {
            for eta in 0..3 {
                let row_a = 3 * a + tau;
                let col_a = 3 * a + eta;
                let mut sum_off = 0.0;
                for b in 0..n_atoms {
                    if a != b {
                        sum_off += hess[(row_a, 3 * b + eta)];
                    }
                }
                hess[(row_a, col_a)] = -sum_off;
            }
        }
    }

    hess
}

/// Computes the genuine analytical nuclear Hessian matrix (3*N x 3*N) for Restricted Hartree-Fock (RHF)
/// in Hartree / Bohr^2 using first-order Coupled-Perturbed Hartree-Fock (CPHF) and analytical integral derivatives.
pub fn try_compute_rhf_analytical_hessian(
    atoms: &[Atom], basis: &[ContractedGaussian], hf: &HFResult,
) -> Result<Matrix,crate::error::HfError> {
    if let Some(space)=&hf.ao_space { space.ensure_differentiable()?; }
    if crate::ecp::has_ecp(atoms) || hf.mo_coefficients.cols != basis.len() {
        return crate::hf_derivatives::HfNuclearDerivativeContext::from_rhf(atoms,basis,hf)?.hessian();
    }
    if !hf.converged { return Err(crate::error::HfError::Scf("reference did not converge".into())); }
    Ok(compute_rhf_analytical_hessian(atoms,basis,hf))
}

pub fn compute_rhf_analytical_hessian(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
) -> Matrix {
    if crate::ecp::has_ecp(atoms) || hf.mo_coefficients.cols != basis.len() {
        return crate::hf_derivatives::HfNuclearDerivativeContext::from_rhf(atoms, basis, hf)
            .and_then(|ctx| ctx.hessian()).expect("analytical ECP Hessian response failed");
    }
    let n_atoms = atoms.len();
    let n_basis = basis.len();
    let dim = 3 * n_atoms;
    let n_occ = hf.n_occ;

    let c = &hf.mo_coefficients;
    let p = &hf.density_matrix;
    let w = compute_energy_weighted_density(hf, n_occ);

    // Precompute 0th, 1st, and 2nd derivatives of each basis function
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

    // 1st derivatives of overlap and skeleton Fock matrices: S^(x) and F^(x)skel
    let perts: Vec<(usize, usize, usize)> = (0..n_atoms)
        .flat_map(|a| (0..3).map(move |tau| (3 * a + tau, a, tau)))
        .collect();

    let (s_1st_ao, f_skel_1st_ao): (Vec<Matrix>, Vec<Matrix>) = perts
        .into_par_iter()
        .map(|(_pert, a, tau)| {
            let mut s_mat = Matrix::zeros(n_basis, n_basis);
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

                    f_mat[(mu, nu)] = dt + dv;
                }
            }

            // 2-electron skeleton derivative:
            // G_{\mu\nu}^{(x)} = \sum_{\lambda\sigma} P_{\lambda\sigma} [ (\mu\nu|\lambda\sigma)^{(x)} - 0.5 * (\mu\lambda|\nu\sigma)^{(x)} ]
            for mu in 0..n_basis {
                let mu_on_a = basis[mu].atom_index == a;
                for nu in 0..n_basis {
                    let nu_on_a = basis[nu].atom_index == a;
                    let mut g_val = 0.0;

                    for lambda in 0..n_basis {
                        let lam_on_a = basis[lambda].atom_index == a;
                        for sigma in 0..n_basis {
                            let p_ls = p[(lambda, sigma)];
                            if p_ls.abs() < 1e-15 {
                                continue;
                            }
                            let sig_on_a = basis[sigma].atom_index == a;

                            if !mu_on_a && !nu_on_a && !lam_on_a && !sig_on_a {
                                continue;
                            }

                            let mut d_coulomb = 0.0;
                            if mu_on_a {
                                d_coulomb += eval_eri_deriv(&d1[mu][tau], &d0[nu], &d0[lambda], &d0[sigma]);
                            }
                            if nu_on_a {
                                d_coulomb += eval_eri_deriv(&d0[mu], &d1[nu][tau], &d0[lambda], &d0[sigma]);
                            }
                            if lam_on_a {
                                d_coulomb += eval_eri_deriv(&d0[mu], &d0[nu], &d1[lambda][tau], &d0[sigma]);
                            }
                            if sig_on_a {
                                d_coulomb += eval_eri_deriv(&d0[mu], &d0[nu], &d0[lambda], &d1[sigma][tau]);
                            }

                            let mut d_exchange = 0.0;
                            if mu_on_a {
                                d_exchange += eval_eri_deriv(&d1[mu][tau], &d0[lambda], &d0[nu], &d0[sigma]);
                            }
                            if lam_on_a {
                                d_exchange += eval_eri_deriv(&d0[mu], &d1[lambda][tau], &d0[nu], &d0[sigma]);
                            }
                            if nu_on_a {
                                d_exchange += eval_eri_deriv(&d0[mu], &d0[lambda], &d1[nu][tau], &d0[sigma]);
                            }
                            if sig_on_a {
                                d_exchange += eval_eri_deriv(&d0[mu], &d0[lambda], &d0[nu], &d1[sigma][tau]);
                            }

                            g_val += p_ls * (d_coulomb - 0.5 * d_exchange);
                        }
                    }

                    f_mat[(mu, nu)] += g_val;
                }
            }

            (s_mat, f_mat)
        })
        .unzip();

    // Transform 1st derivatives to MO basis: S_mo^(x) = C^T S^(x) C, F_mo^(x) = C^T F^(x) C
    let ct = c.transpose();
    let s_1st_mo: Vec<Matrix> = s_1st_ao.iter().map(|s| ct.dot(s).dot(c)).collect();
    let f_skel_1st_mo: Vec<Matrix> = f_skel_1st_ao.iter().map(|f| ct.dot(f).dot(c)).collect();

    // Solve CPHF linear equations for orbital responses U^x
    let resp = solve_cphf(hf, n_occ, &f_skel_1st_mo, &s_1st_mo);
    let u_mats = resp.u_matrices;
    let mo_eri = crate::mo_integrals::transform_eri_ao_to_mo(c, &hf.eri);

    // Start with exact nuclear repulsion Hessian
    let nuc_hess = compute_nuclear_repulsion_hessian(atoms);

    // Pre-extract non-zero density pairs
    let mut sig_pairs = Vec::new();
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            let p_val = p[(mu, nu)];
            let w_val = w[(mu, nu)];
            if p_val.abs() > 1e-15 || w_val.abs() > 1e-15 {
                sig_pairs.push((mu, nu, p_val, w_val));
            }
        }
    }

    // Precompute 2-particle density matrix: \Gamma_{\mu\nu\lambda\sigma} = P_{\mu\nu} P_{\lambda\sigma} - 0.5 * P_{\mu\lambda} P_{\nu\sigma}
    let mut sig_quartets = Vec::new();
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for lambda in 0..n_basis {
                for sigma in 0..n_basis {
                    let gamma = p[(mu, nu)] * p[(lambda, sigma)] - 0.5 * p[(mu, lambda)] * p[(nu, sigma)];
                    if gamma.abs() > 1e-15 {
                        sig_quartets.push((mu, nu, lambda, sigma, gamma));
                    }
                }
            }
        }
    }

    // Collect upper-triangular perturbation pairs (x, y) with x >= y
    let mut coord_pairs = Vec::new();
    for x in 0..dim {
        for y in 0..=x {
            coord_pairs.push((x, y));
        }
    }

    // Evaluate electronic Hessian elements in parallel
    let pair_elements: Vec<((usize, usize), f64)> = coord_pairs
        .into_par_iter()
        .map(|(x, y)| {
            let a = x / 3;
            let tau = x % 3;
            let b = y / 3;
            let eta = y % 3;

            // 1. One-electron skeleton derivative: \sum_{\mu\nu} P_{\mu\nu} h_{\mu\nu}^{(xy)} - W_{\mu\nu} S_{\mu\nu}^{(xy)}
            let mut e1_xy = 0.0;
            for &(mu, nu, p_val, w_val) in &sig_pairs {
                let mu_a = basis[mu].atom_index == a;
                let nu_a = basis[nu].atom_index == a;
                let mu_b = basis[mu].atom_index == b;
                let nu_b = basis[nu].atom_index == b;

                let mut s_xy = 0.0;
                let mut t_xy = 0.0;
                if mu_a && mu_b {
                    s_xy += eval_overlap_deriv(&d2[mu][tau][eta], &d0[nu]);
                    t_xy += eval_kinetic_deriv(&d2[mu][tau][eta], &d0[nu]);
                }
                if mu_a && nu_b {
                    s_xy += eval_overlap_deriv(&d1[mu][tau], &d1[nu][eta]);
                    t_xy += eval_kinetic_deriv(&d1[mu][tau], &d1[nu][eta]);
                }
                if nu_a && mu_b {
                    s_xy += eval_overlap_deriv(&d1[nu][tau], &d1[mu][eta]);
                    t_xy += eval_kinetic_deriv(&d1[nu][tau], &d1[mu][eta]);
                }
                if nu_a && nu_b {
                    s_xy += eval_overlap_deriv(&d0[mu], &d2[nu][tau][eta]);
                    t_xy += eval_kinetic_deriv(&d0[mu], &d2[nu][tau][eta]);
                }

                let mut v_xy = 0.0;
                for c_idx in 0..n_atoms {
                    let ca1: f64 = (if mu_a { 1.0 } else { 0.0 }) - (if c_idx == a { 1.0 } else { 0.0 });
                    let cb1: f64 = (if nu_a { 1.0 } else { 0.0 }) - (if c_idx == a { 1.0 } else { 0.0 });
                    let ca2: f64 = (if mu_b { 1.0 } else { 0.0 }) - (if c_idx == b { 1.0 } else { 0.0 });
                    let cb2: f64 = (if nu_b { 1.0 } else { 0.0 }) - (if c_idx == b { 1.0 } else { 0.0 });

                    if ca1.abs() > 1e-12 && ca2.abs() > 1e-12 {
                        v_xy += ca1 * ca2 * eval_nuclear_deriv_single_atom(&d2[mu][tau][eta], &d0[nu], &atoms[c_idx]);
                    }
                    if ca1.abs() > 1e-12 && cb2.abs() > 1e-12 {
                        v_xy += ca1 * cb2 * eval_nuclear_deriv_single_atom(&d1[mu][tau], &d1[nu][eta], &atoms[c_idx]);
                    }
                    if cb1.abs() > 1e-12 && ca2.abs() > 1e-12 {
                        v_xy += cb1 * ca2 * eval_nuclear_deriv_single_atom(&d1[nu][tau], &d1[mu][eta], &atoms[c_idx]);
                    }
                    if cb1.abs() > 1e-12 && cb2.abs() > 1e-12 {
                        v_xy += cb1 * cb2 * eval_nuclear_deriv_single_atom(&d0[mu], &d2[nu][tau][eta], &atoms[c_idx]);
                    }
                }

                e1_xy += p_val * (t_xy + v_xy) - w_val * s_xy;
            }

            // 2. Two-electron skeleton derivative: 0.5 * \sum_{\mu\nu\lambda\sigma} \Gamma_{\mu\nu\lambda\sigma} (\mu\nu|\lambda\sigma)^{(xy)}
            let mut e2_xy = 0.0;
            for &(mu, nu, lambda, sigma, gamma) in &sig_quartets {
                let at_mu = basis[mu].atom_index;
                let at_nu = basis[nu].atom_index;
                let at_lam = basis[lambda].atom_index;
                let at_sig = basis[sigma].atom_index;

                let at = [at_mu, at_nu, at_lam, at_sig];
                let g = [mu, nu, lambda, sigma];

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
                            d_slots[k] = &d2[g[k]][tau][eta];
                            eval_eri_deriv(d_slots[0], d_slots[1], d_slots[2], d_slots[3])
                        } else {
                            let mut d_slots = [&d0[g[0]], &d0[g[1]], &d0[g[2]], &d0[g[3]]];
                            d_slots[k] = &d1[g[k]][tau];
                            d_slots[l] = &d1[g[l]][eta];
                            eval_eri_deriv(d_slots[0], d_slots[1], d_slots[2], d_slots[3])
                        };

                        e2_xy += 0.5 * gamma * d_eri;
                    }
                }
            }

            // 3. Orbital response contribution from CPHF U^x and U^y
            let ux = &u_mats[x];
            let uy = &u_mats[y];
            let sx = &s_1st_mo[x];
            let sy = &s_1st_mo[y];
            let fx = &f_skel_1st_mo[x];
            let fy = &f_skel_1st_mo[y];

            let mut e_resp = 0.0;

            // Virtual-Occupied: 2 * \sum_{ai} [ U_ai^x (F_ai^y - eps_i S_ai^y) + U_ai^y (F_ai^x - eps_i S_ai^x) ]
            let n_virt = n_basis - n_occ;
            for a_virt in 0..n_virt {
                let a_mo = n_occ + a_virt;
                for i in 0..n_occ {
                    let term_x = ux[(a_mo, i)] * (fy[(a_mo, i)] - hf.orbital_energies[i] * sy[(a_mo, i)]);
                    let term_y = uy[(a_mo, i)] * (fx[(a_mo, i)] - hf.orbital_energies[i] * sx[(a_mo, i)]);
                    e_resp += 2.0 * (term_x + term_y);

                    // Cross-coupling with occupied-occupied overlap
                    for k in 0..n_occ {
                        for l in 0..n_occ {
                            let val_aikl = mo_eri.get(a_mo, i, k, l);
                            let val_akil = mo_eri.get(a_mo, k, i, l);
                            let coupling = 2.0 * val_aikl - val_akil;
                            e_resp -= 2.0 * (ux[(a_mo, i)] * sy[(k, l)] + uy[(a_mo, i)] * sx[(k, l)]) * coupling;
                        }
                    }
                }
            }

            // Occupied-Occupied:
            // - 2 * \sum_{ij} ( S_ij^x F_ij^y + S_ij^y F_ij^x ) + 2 * \sum_{ij} (eps_i + eps_j) S_ij^x S_ij^y
            // + 2 * \sum_{ij, kl} S_ij^x S_kl^y [ 2(ij|kl) - (ik|jl) ]
            for i in 0..n_occ {
                for j in 0..n_occ {
                    let f_term = sx[(i, j)] * fy[(i, j)] + sy[(i, j)] * fx[(i, j)];
                    let s_term = (hf.orbital_energies[i] + hf.orbital_energies[j]) * sx[(i, j)] * sy[(i, j)];
                    e_resp += 2.0 * (-f_term + s_term);

                    for k in 0..n_occ {
                        for l in 0..n_occ {
                            let val_ijkl = mo_eri.get(i, j, k, l);
                            let val_ikjl = mo_eri.get(i, k, j, l);
                            let coupling = 2.0 * val_ijkl - val_ikjl;
                            e_resp += 2.0 * sx[(i, j)] * sy[(k, l)] * coupling;
                        }
                    }
                }
            }

            let val = e1_xy + e2_xy + e_resp;
            ((x, y), val)
        })
        .collect();

    let mut hess = nuc_hess;
    for ((x, y), val) in pair_elements {
        hess[(x, y)] += val;
        if x != y {
            hess[(y, x)] += val;
        }
    }

    hess
}

/// Analytical UHF Hessian from spin-coupled density response, including scalar ECPs.
pub fn compute_uhf_analytical_hessian(
    atoms: &[Atom], basis: &[ContractedGaussian], hf: &crate::scf::UHFResult,
) -> Result<Matrix, crate::error::HfError> {
    crate::hf_derivatives::HfNuclearDerivativeContext::from_uhf(atoms, basis, hf)?.hessian()
}

/// Evaluates directional nuclear Hessian along two displacement vectors u and v:
/// H(u, v) = \sum_{xy} H_{xy} u_x v_y
pub fn compute_hessian_directional(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    u: &[f64],
    v: &[f64],
) -> f64 {
    let dim = 3 * atoms.len();
    assert_eq!(u.len(), dim, "Vector u must have length 3N");
    assert_eq!(v.len(), dim, "Vector v must have length 3N");

    let hess = compute_rhf_analytical_hessian(atoms, basis, hf);
    let mut sum = 0.0;
    for x in 0..dim {
        for y in 0..dim {
            sum += hess[(x, y)] * u[x] * v[y];
        }
    }
    sum
}

/// Computes a sub-block of the analytical nuclear Hessian for a specified subset of active atoms.
pub fn compute_hessian_block(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    active_atoms: &[usize],
) -> Matrix {
    let full_hess = compute_rhf_analytical_hessian(atoms, basis, hf);
    let n_act = active_atoms.len();
    let dim_act = 3 * n_act;
    let mut block = Matrix::zeros(dim_act, dim_act);

    for (ia, &a) in active_atoms.iter().enumerate() {
        for tau in 0..3 {
            let row_block = 3 * ia + tau;
            let row_full = 3 * a + tau;
            for (ib, &b) in active_atoms.iter().enumerate() {
                for eta in 0..3 {
                    let col_block = 3 * ib + eta;
                    let col_full = 3 * b + eta;
                    block[(row_block, col_block)] = full_hess[(row_full, col_full)];
                }
            }
        }
    }

    block
}

/// Computes the genuine analytical nuclear Hessian matrix (3*N x 3*N) in Hartree / Bohr^2 with specified formal charge.
pub fn compute_hessian_with_charge(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Matrix {
    let basis = build_basis(atoms, basis_map);
    let hf = crate::scf::run_scf_with_charge(atoms, &basis, charge, max_iter, tol)
        .expect("Reference SCF failed in Hessian");
    compute_rhf_analytical_hessian(atoms, &basis, &hf)
}

/// Computes the genuine analytical nuclear Hessian matrix (3*N x 3*N) in Hartree / Bohr^2 with zero charge.
pub fn compute_hessian(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    max_iter: usize,
    tol: f64,
) -> Matrix {
    compute_hessian_with_charge(atoms, basis_map, 0, max_iter, tol)
}

/// Computes nuclear Hessian by high-precision central differences of analytical gradients (reference / debugging).
pub fn compute_hessian_finite_difference(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Matrix {
    let n_atoms = atoms.len();
    let dim = 3 * n_atoms;
    let h = 1e-4; // Step size in Bohr

    let basis_0 = build_basis(atoms, basis_map);
    let hf_0 = crate::scf::run_scf_with_charge(atoms, &basis_0, charge, max_iter, tol)
        .expect("Reference SCF failed in Hessian");
    let p_0 = hf_0.density_matrix;

    let perts: Vec<(usize, usize, usize)> = (0..n_atoms)
        .flat_map(|a| (0..3).map(move |tau| (3 * a + tau, a, tau)))
        .collect();

    let grad_diffs: Vec<Vec<f64>> = perts
        .into_par_iter()
        .map(|(_col, a, tau)| {
            let mut atoms_fwd = atoms.to_vec();
            atoms_fwd[a].position[tau] += h;
            let basis_fwd = build_basis(&atoms_fwd, basis_map);
            let hf_fwd = crate::scf::run_scf_with_initial_density(
                &atoms_fwd, &basis_fwd, charge, max_iter, tol, Some(&p_0),
            )
            .expect("SCF forward step failed in Hessian");
            let g_fwd = compute_rhf_gradient(&atoms_fwd, &basis_fwd, &hf_fwd);

            let mut atoms_bwd = atoms.to_vec();
            atoms_bwd[a].position[tau] -= h;
            let basis_bwd = build_basis(&atoms_bwd, basis_map);
            let hf_bwd = crate::scf::run_scf_with_initial_density(
                &atoms_bwd, &basis_bwd, charge, max_iter, tol, Some(&p_0),
            )
            .expect("SCF backward step failed in Hessian");
            let g_bwd = compute_rhf_gradient(&atoms_bwd, &basis_bwd, &hf_bwd);

            let mut col_data = Vec::with_capacity(dim);
            for atom_idx in 0..n_atoms {
                for comp in 0..3 {
                    let d = (g_fwd[atom_idx][comp] - g_bwd[atom_idx][comp]) / (2.0 * h);
                    col_data.push(d);
                }
            }
            col_data
        })
        .collect();

    let mut hess = Matrix::zeros(dim, dim);
    for col in 0..dim {
        for row in 0..dim {
            hess[(row, col)] = grad_diffs[col][row];
        }
    }

    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (hess[(i, j)] + hess[(j, i)]);
            hess[(i, j)] = avg;
            hess[(j, i)] = avg;
        }
    }

    hess
}

/// Relative atomic masses in atomic mass units (amu) using CODATA values from crate::masses
#[inline]
pub fn atomic_mass_amu(z: usize) -> f64 {
    crate::masses::atomic_mass(z)
}

/// Result of Hessian calculation and vibrational frequency analysis.
/// Contains both raw unprojected and projected (Eckart translation/rotation removed) Hessians and frequencies.
#[derive(Clone, Debug)]
pub struct HessianResult {
    /// Raw unprojected Cartesian Hessian matrix (3N x 3N) in Hartree / Bohr^2.
    pub raw_hessian: Matrix,
    /// Projected Cartesian Hessian matrix (3N x 3N) with translation and rotation projected out.
    pub projected_hessian: Matrix,
    /// Projected mass-weighted Hessian matrix (3N x 3N) with translation and rotation projected out.
    pub projected_mass_weighted_hessian: Matrix,
    /// Raw vibrational frequencies in cm^-1 (unprojected).
    pub raw_frequencies: Vec<f64>,
    /// Harmonic vibrational frequencies in cm^-1 with translation and rotation projected out (genuine vibrations).
    pub projected_frequencies: Vec<f64>,
}

/// Projects translational and rotational components out of the Cartesian Hessian.
/// Returns (projected_cartesian_hessian, projected_mass_weighted_hessian).
pub fn project_hessian(atoms: &[Atom], hessian: &Matrix) -> (Matrix, Matrix) {
    let n_atoms = atoms.len();
    let dim = 3 * n_atoms;

    let masses: Vec<f64> = atoms
        .iter()
        .flat_map(|a| {
            let m = atomic_mass_amu(a.atomic_number);
            vec![m, m, m]
        })
        .collect();

    let total_mass: f64 = atoms.iter().map(|a| atomic_mass_amu(a.atomic_number)).sum();
    let mut com = [0.0; 3];
    for a in atoms {
        let m = atomic_mass_amu(a.atomic_number);
        for tau in 0..3 {
            com[tau] += m * a.position[tau];
        }
    }
    for tau in 0..3 {
        com[tau] /= total_mass;
    }

    // Shift coordinates to center-of-mass frame
    let x_com: Vec<[f64; 3]> = atoms
        .iter()
        .map(|a| [
            a.position[0] - com[0],
            a.position[1] - com[1],
            a.position[2] - com[2],
        ])
        .collect();

    // Moment of inertia tensor (3x3)
    let mut inertia = Matrix::zeros(3, 3);
    for (i, a) in atoms.iter().enumerate() {
        let m = atomic_mass_amu(a.atomic_number);
        let x = x_com[i][0];
        let y = x_com[i][1];
        let z = x_com[i][2];
        let r2 = x * x + y * y + z * z;

        inertia[(0, 0)] += m * (r2 - x * x);
        inertia[(1, 1)] += m * (r2 - y * y);
        inertia[(2, 2)] += m * (r2 - z * z);

        inertia[(0, 1)] -= m * x * y;
        inertia[(1, 0)] -= m * x * y;

        inertia[(0, 2)] -= m * x * z;
        inertia[(2, 0)] -= m * x * z;

        inertia[(1, 2)] -= m * y * z;
        inertia[(2, 1)] -= m * y * z;
    }

    let (_evals, evecs) = inertia.eigh();

    // Construct mass-weighted translation and rotation basis vectors in R^(3N)
    let mut tr_vectors: Vec<Vec<f64>> = Vec::new();

    // 3 Translational vectors
    for tau in 0..3 {
        let mut t_vec = vec![0.0; dim];
        for a in 0..n_atoms {
            let m = atomic_mass_amu(atoms[a].atomic_number);
            t_vec[3 * a + tau] = m.sqrt();
        }
        let norm: f64 = t_vec.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for v in &mut t_vec {
                *v /= norm;
            }
            tr_vectors.push(t_vec);
        }
    }

    // 3 Rotational vectors around principal axes of inertia
    for k in 0..3 {
        let axis = [evecs[(0, k)], evecs[(1, k)], evecs[(2, k)]];
        let mut r_vec = vec![0.0; dim];
        for a in 0..n_atoms {
            let m = atomic_mass_amu(atoms[a].atomic_number);
            let x = x_com[a];
            let rx = axis[1] * x[2] - axis[2] * x[1];
            let ry = axis[2] * x[0] - axis[0] * x[2];
            let rz = axis[0] * x[1] - axis[1] * x[0];

            let m_sqrt = m.sqrt();
            r_vec[3 * a] = m_sqrt * rx;
            r_vec[3 * a + 1] = m_sqrt * ry;
            r_vec[3 * a + 2] = m_sqrt * rz;
        }
        let norm: f64 = r_vec.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for v in &mut r_vec {
                *v /= norm;
            }
            tr_vectors.push(r_vec);
        }
    }

    // Orthonormalize TR vectors using Gram-Schmidt
    let mut ortho_tr: Vec<Vec<f64>> = Vec::new();
    for v in tr_vectors {
        let mut cur = v;
        for u in &ortho_tr {
            let dot: f64 = cur.iter().zip(u.iter()).map(|(a, b)| a * b).sum();
            for i in 0..dim {
                cur[i] -= dot * u[i];
            }
        }
        let norm: f64 = cur.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-8 {
            for x in &mut cur {
                *x /= norm;
            }
            ortho_tr.push(cur);
        }
    }

    // Projector P = I - sum_k u_k u_k^T
    let mut p_mat = Matrix::zeros(dim, dim);
    for i in 0..dim {
        p_mat[(i, i)] = 1.0;
        for u in &ortho_tr {
            for j in 0..dim {
                p_mat[(i, j)] -= u[i] * u[j];
            }
        }
    }

    // Mass-weighted Hessian D_mw = M^(-1/2) H M^(-1/2)
    let mut mass_weighted = Matrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            mass_weighted[(i, j)] = hessian[(i, j)] / (masses[i] * masses[j]).sqrt();
        }
    }

    // Projected mass-weighted Hessian: D_proj = P * D_mw * P
    let d_proj = p_mat.dot(&mass_weighted).dot(&p_mat);
    let mut d_proj_symm = Matrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            d_proj_symm[(i, j)] = 0.5 * (d_proj[(i, j)] + d_proj[(j, i)]);
        }
    }

    // Projected Cartesian Hessian: H_proj = M^(1/2) D_proj M^(1/2)
    let mut h_proj = Matrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            h_proj[(i, j)] = d_proj_symm[(i, j)] * (masses[i] * masses[j]).sqrt();
        }
    }

    (h_proj, d_proj_symm)
}

/// Computes raw (unprojected) vibrational frequencies in cm^-1 from the Cartesian Hessian.
pub fn compute_raw_vibrational_frequencies(atoms: &[Atom], hessian: &Matrix) -> Vec<f64> {
    let dim = 3 * atoms.len();
    let masses: Vec<f64> = atoms
        .iter()
        .flat_map(|a| {
            let m = atomic_mass_amu(a.atomic_number);
            vec![m, m, m]
        })
        .collect();

    let mut mass_weighted = Matrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            mass_weighted[(i, j)] = hessian[(i, j)] / (masses[i] * masses[j]).sqrt();
        }
    }

    let (evals, _evecs) = mass_weighted.eigh();
    const AU_TO_CM1: f64 = 5140.48714376;

    evals
        .into_iter()
        .map(|val| {
            if val >= 0.0 {
                val.sqrt() * AU_TO_CM1
            } else {
                -(-val).sqrt() * AU_TO_CM1
            }
        })
        .collect()
}

/// Computes harmonic vibrational frequencies in cm^-1 with translational and rotational components projected out.
pub fn compute_projected_vibrational_frequencies(atoms: &[Atom], hessian: &Matrix) -> Vec<f64> {
    let (_h_proj, d_proj) = project_hessian(atoms, hessian);
    let (evals, _evecs) = d_proj.eigh();
    const AU_TO_CM1: f64 = 5140.48714376;

    evals
        .into_iter()
        .map(|val| {
            if val.abs() < 1e-10 {
                0.0
            } else if val > 0.0 {
                val.sqrt() * AU_TO_CM1
            } else {
                -(-val).sqrt() * AU_TO_CM1
            }
        })
        .collect()
}

/// Computes complete vibrational frequency analysis including raw and projected Hessians and frequencies.
pub fn compute_hessian_analysis(atoms: &[Atom], hessian: &Matrix) -> HessianResult {
    let (h_proj, d_proj) = project_hessian(atoms, hessian);
    let raw_frequencies = compute_raw_vibrational_frequencies(atoms, hessian);
    let (evals, _evecs) = d_proj.eigh();
    const AU_TO_CM1: f64 = 5140.48714376;

    let projected_frequencies = evals
        .into_iter()
        .map(|val| {
            if val.abs() < 1e-10 {
                0.0
            } else if val > 0.0 {
                val.sqrt() * AU_TO_CM1
            } else {
                -(-val).sqrt() * AU_TO_CM1
            }
        })
        .collect();

    HessianResult {
        raw_hessian: hessian.clone(),
        projected_hessian: h_proj,
        projected_mass_weighted_hessian: d_proj,
        raw_frequencies,
        projected_frequencies,
    }
}

/// Computes harmonic vibrational frequencies in cm^-1 from the Cartesian Hessian.
/// Projects out translation and rotation (Eckart conditions).
pub fn compute_vibrational_frequencies(atoms: &[Atom], hessian: &Matrix) -> Vec<f64> {
    compute_projected_vibrational_frequencies(atoms, hessian)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::parse_gaussian94;

    #[test]
    fn test_rhf_analytical_hessian_vs_finite_diff_h2() {
        let basis_map = parse_gaussian94(crate::basis::STO_3G_GBS);
        let r = 1.4;
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -r / 2.0]),
            Atom::new("H", [0.0, 0.0, r / 2.0]),
        ];

        let basis = build_basis(&atoms, &basis_map);
        let hf = crate::scf::run_scf_with_charge(&atoms, &basis, 0, 50, 1e-12)
            .expect("SCF failed");

        let h_anal = compute_rhf_analytical_hessian(&atoms, &basis, &hf);
        let h_num = compute_hessian_finite_difference(&atoms, &basis_map, 0, 50, 1e-12);

        eprintln!("Analytical Hessian:\n{}", h_anal);
        eprintln!("Numerical Hessian:\n{}", h_num);

        let dim = 3 * atoms.len();
        for i in 0..dim {
            for j in 0..dim {
                let diff = (h_anal[(i, j)] - h_num[(i, j)]).abs();
                assert!(
                    diff < 1e-4,
                    "Hessian mismatch at ({}, {}): anal = {}, num = {}, diff = {}",
                    i, j, h_anal[(i, j)], h_num[(i, j)], diff
                );
            }
        }
    }
}
