// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Restricted Open-Shell Hartree-Fock (ROHF) and specialized HF variants.
//!
//! Provides:
//! - Roothaan-Guest-Saunders canonical ROHF with exact spin adaptation (<S^2> = S(S+1)).
//! - Quasi-Restricted Hartree-Fock (QRHF) for natural-orbital spin adaptation.
//! - Maximum Overlap Method (MOM-SCF) for targeting excited and core-hole states without variational collapse.
//! - Finite-temperature Fermi-Dirac thermal smearing for fractional occupation HF.

use rayon::prelude::*;
use crate::basis::{Atom, ContractedGaussian};
use crate::matrix::Matrix;
use crate::properties::{compute_molecular_properties, MolecularProperties};
use crate::scf::{
    build_fock_matrix, build_uhf_fock_matrices, compute_eri_tensor,
    compute_nuclear_repulsion, compute_one_electron_integrals, HFResult,
};

/// Results from a Restricted Open-Shell Hartree-Fock (ROHF) calculation.
#[derive(Clone, Debug)]
pub struct ROHFResult {
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_repulsion_energy: f64,
    pub orbital_energies: Vec<f64>,
    pub mo_coefficients: Matrix,
    pub density_matrix_alpha: Matrix,
    pub density_matrix_beta: Matrix,
    pub fock_matrix_alpha: Matrix,
    pub fock_matrix_beta: Matrix,
    pub overlap_matrix: Matrix,
    pub core_hamiltonian: Matrix,
    pub eri: Vec<f64>,
    pub s_squared: f64, // Exactly S(S+1) with zero spin contamination
    pub iterations: usize,
    pub converged: bool,
    pub charge: i32,
    pub multiplicity: usize,
    pub n_closed: usize,
    pub n_open: usize,
    pub properties: MolecularProperties,
}

impl ROHFResult {
    /// Total number of occupied spatial orbitals (closed + open).
    pub fn n_occ(&self) -> usize {
        self.n_closed + self.n_open
    }

    /// Number of alpha electrons.
    pub fn n_alpha(&self) -> usize {
        self.n_closed + self.n_open
    }

    /// Number of beta electrons.
    pub fn n_beta(&self) -> usize {
        self.n_closed
    }

    /// Spin quantum number S = (multiplicity - 1) / 2.
    pub fn spin_s(&self) -> f64 {
        (self.multiplicity as f64 - 1.0) / 2.0
    }
}

/// Runs a Restricted Open-Shell Hartree-Fock (ROHF) calculation.
///
/// Uses the canonical Roothaan-Guest-Saunders unified Fock matrix in the MO basis
/// with level shifts to eliminate variational coupling between Closed, Open,
/// and Virtual spaces while strictly maintaining <S^2> = S(S+1).
pub fn run_rohf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    multiplicity: usize,
    max_iter: usize,
    tol: f64,
) -> Result<ROHFResult, String> {
    let n = basis.len();
    let total_nuclear_charge: f64 = atoms.iter().map(|a| a.charge).sum();
    let total_electrons = (total_nuclear_charge.round() as i32) - charge;

    if total_electrons <= 0 {
        return Err(format!("Invalid electron count {} for charge {}", total_electrons, charge));
    }
    let n_electrons = total_electrons as usize;

    if multiplicity == 0 {
        return Err("Multiplicity must be >= 1".to_string());
    }
    if multiplicity-1>n_electrons {return Err("Multiplicity exceeds electron count".into());}
    let two_s = multiplicity - 1;
    if !(n_electrons + two_s).is_multiple_of(2) {
        return Err(format!(
            "Incompatible electron count ({}) and multiplicity ({})",
            n_electrons, multiplicity
        ));
    }

    let n_alpha = (n_electrons + two_s) / 2;
    let n_beta = (n_electrons - two_s) / 2;
    let n_closed = n_beta;
    let n_open = n_alpha - n_beta;
    let e_nuc=compute_nuclear_repulsion(atoms);
    let (s,_,_,h)=compute_one_electron_integrals(atoms,basis);
    let x_ortho=s.canonical_orthogonalizer(1e-8).map_err(|e|e.to_string())?;
    let n_mo=x_ortho.cols;
    let n_virt=n_mo.checked_sub(n_closed+n_open).ok_or("Electrons exceed retained overlap rank")?;
    let eri = compute_eri_tensor(basis);

    // Initial guess from core Hamiltonian
    let f_prime = x_ortho.transpose().dot(&h).dot(&x_ortho);
    let (_evals, c_prime) = f_prime.eigh();
    let mut c_mo = x_ortho.dot(&c_prime);

    let mut prev_energy = 0.0;
    let mut converged = false;
    let mut fock_alpha = h.clone();
    let mut fock_beta = h.clone();
    let mut iter_count = 0;
    let mut cdiis = crate::diis::CdiisEngine::new(8);
    let s_sqrt = x_ortho.transpose().dot(&s);

    for iteration in 1..=max_iter {
        iter_count = iteration;

        // Build idempotent density matrices from current MO coefficients
        let mut p_alpha = Matrix::zeros(n, n);
        let mut p_beta = Matrix::zeros(n, n);

        for mu in 0..n {
            for nu in 0..n {
                let mut sum_closed = 0.0;
                for i in 0..n_closed {
                    sum_closed += c_mo[(mu, i)] * c_mo[(nu, i)];
                }
                let mut sum_open = 0.0;
                for m in n_closed..(n_closed + n_open) {
                    sum_open += c_mo[(mu, m)] * c_mo[(nu, m)];
                }
                p_beta[(mu, nu)] = sum_closed;
                p_alpha[(mu, nu)] = sum_closed + sum_open;
            }
        }

        let (fa, fb) = build_uhf_fock_matrices(&h, &p_alpha, &p_beta, &eri, n);
        fock_alpha = fa;
        fock_beta = fb;

        // Electronic energy: 0.5 * Tr((P_alpha + P_beta) * H + P_alpha * F_alpha + P_beta * F_beta)
        let mut e_elec = 0.0;
        let p_tot = p_alpha.add(&p_beta);
        for i in 0..n {
            for j in 0..n {
                e_elec += 0.5 * (p_tot[(i, j)] * h[(i, j)]
                    + p_alpha[(i, j)] * fock_alpha[(i, j)]
                    + p_beta[(i, j)] * fock_beta[(i, j)]);
            }
        }
        let total_e = e_elec + e_nuc;

        // Transform Fock matrices to current MO basis
        let fa_mo = c_mo.transpose().dot(&fock_alpha).dot(&c_mo);
        let fb_mo = c_mo.transpose().dot(&fock_beta).dot(&c_mo);

        // Check off-diagonal coupling gradients:
        // Closed-Open: F_beta
        // Closed-Virt: 0.5 * (F_alpha + F_beta)
        // Open-Virt:   F_alpha
        let mut max_grad: f64 = 0.0;
        let mut err_mo = Matrix::zeros(n_mo, n_mo);

        for i in 0..n_closed {
            for m in n_closed..(n_closed + n_open) {
                let v = fb_mo[(i, m)];
                max_grad = max_grad.max(v.abs());
                err_mo[(i, m)] = v;
                err_mo[(m, i)] = -v;
            }
            for a in (n_closed + n_open)..n_mo {
                let v = 0.5 * (fa_mo[(i, a)] + fb_mo[(i, a)]);
                max_grad = max_grad.max(v.abs());
                err_mo[(i, a)] = v;
                err_mo[(a, i)] = -v;
            }
        }
        for m in n_closed..(n_closed + n_open) {
            for a in (n_closed + n_open)..n_mo {
                let v = fa_mo[(m, a)];
                max_grad = max_grad.max(v.abs());
                err_mo[(m, a)] = v;
                err_mo[(a, m)] = -v;
            }
        }

        let delta_e = (total_e - prev_energy).abs();
        prev_energy = total_e;

        let grad_tol = tol.sqrt().min(1e-8).max(tol);
        if iteration > 1 && delta_e < tol && max_grad < grad_tol {
            converged = true;
            break;
        }

        // Construct Roothaan-Guest-Saunders unified coupling operator in MO basis
        let mut f_eff = Matrix::zeros(n_mo, n_mo);
        let shift_c = 0.5;
        let shift_v = 0.5;

        // Diagonal blocks with level shifts to preserve subspace partitioning
        for p in 0..n_closed {
            for q in 0..n_closed {
                f_eff[(p, q)] = 0.5 * (fa_mo[(p, q)] + fb_mo[(p, q)]) - if p == q { shift_c } else { 0.0 };
            }
        }
        for p in n_closed..(n_closed + n_open) {
            for q in n_closed..(n_closed + n_open) {
                f_eff[(p, q)] = 0.5 * (fa_mo[(p, q)] + fb_mo[(p, q)]);
            }
        }
        for p in (n_closed + n_open)..n_mo {
            for q in (n_closed + n_open)..n_mo {
                f_eff[(p, q)] = 0.5 * (fa_mo[(p, q)] + fb_mo[(p, q)]) + if p == q { shift_v } else { 0.0 };
            }
        }

        // Off-diagonal coupling blocks
        for i in 0..n_closed {
            for m in n_closed..(n_closed + n_open) {
                let v = fb_mo[(i, m)];
                f_eff[(i, m)] = v;
                f_eff[(m, i)] = v;
            }
            for a in (n_closed + n_open)..n_mo {
                let v = 0.5 * (fa_mo[(i, a)] + fb_mo[(i, a)]);
                f_eff[(i, a)] = v;
                f_eff[(a, i)] = v;
            }
        }
        for m in n_closed..(n_closed + n_open) {
            for a in (n_closed + n_open)..n_mo {
                let v = fa_mo[(m, a)];
                f_eff[(m, a)] = v;
                f_eff[(a, m)] = v;
            }
        }

        // Transform effective Fock and antisymmetric orbital gradient error to fixed S^{1/2} orthonormal basis
        let q_ortho = s_sqrt.dot(&c_mo);
        let f_eff_ortho = q_ortho.dot(&f_eff).dot(&q_ortho.transpose());
        let err_ortho = q_ortho.dot(&err_mo).dot(&q_ortho.transpose());
        cdiis.push(f_eff_ortho.clone(), err_ortho);

        let f_ortho_use = if iteration >= 2 {
            cdiis.extrapolate().unwrap_or(f_eff_ortho)
        } else {
            f_eff_ortho
        };

        // Diagonalize effective Fock matrix in fixed orthonormal basis
        let (_evals_eff, v_ortho) = f_ortho_use.eigh();
        c_mo = x_ortho.dot(&v_ortho);
    }

    // Final canonicalization: diagonalize within each unshifted subspace
    // Closed subspace
    let mut c_final = c_mo.clone();
    let mut orbital_energies = vec![0.0; n_mo];

    if n_closed > 0 {
        let mut fc = Matrix::zeros(n_closed, n_closed);
        let fa_mo = c_final.transpose().dot(&fock_alpha).dot(&c_final);
        let fb_mo = c_final.transpose().dot(&fock_beta).dot(&c_final);
        for p in 0..n_closed {
            for q in 0..n_closed {
                fc[(p, q)] = 0.5 * (fa_mo[(p, q)] + fb_mo[(p, q)]);
            }
        }
        let (evals_c, uc) = fc.eigh();
        orbital_energies[..n_closed].copy_from_slice(&evals_c[..n_closed]);
        // Rotate closed columns
        let mut c_closed_block = Matrix::zeros(n, n_closed);
        for r in 0..n {
            for c in 0..n_closed {
                c_closed_block[(r, c)] = c_final[(r, c)];
            }
        }
        let rotated_c = c_closed_block.dot(&uc);
        for r in 0..n {
            for c in 0..n_closed {
                c_final[(r, c)] = rotated_c[(r, c)];
            }
        }
    }

    // Open subspace
    if n_open > 0 {
        let mut fo = Matrix::zeros(n_open, n_open);
        let fa_mo = c_final.transpose().dot(&fock_alpha).dot(&c_final);
        for p in 0..n_open {
            for q in 0..n_open {
                fo[(p, q)] = fa_mo[(n_closed + p, n_closed + q)];
            }
        }
        let (evals_o, uo) = fo.eigh();
        orbital_energies[n_closed..(n_closed + n_open)].copy_from_slice(&evals_o[..n_open]);
        let mut c_open_block = Matrix::zeros(n, n_open);
        for r in 0..n {
            for c in 0..n_open {
                c_open_block[(r, c)] = c_final[(r, n_closed + c)];
            }
        }
        let rotated_o = c_open_block.dot(&uo);
        for r in 0..n {
            for c in 0..n_open {
                c_final[(r, n_closed + c)] = rotated_o[(r, c)];
            }
        }
    }

    // Virtual subspace
    if n_virt > 0 {
        let mut fv = Matrix::zeros(n_virt, n_virt);
        let fa_mo = c_final.transpose().dot(&fock_alpha).dot(&c_final);
        let fb_mo = c_final.transpose().dot(&fock_beta).dot(&c_final);
        for p in 0..n_virt {
            for q in 0..n_virt {
                fv[(p, q)] = 0.5 * (fa_mo[(n_closed + n_open + p, n_closed + n_open + q)]
                    + fb_mo[(n_closed + n_open + p, n_closed + n_open + q)]);
            }
        }
        let (evals_v, uv) = fv.eigh();
        orbital_energies[(n_closed + n_open)..(n_closed + n_open + n_virt)].copy_from_slice(&evals_v[..n_virt]);
        let mut c_virt_block = Matrix::zeros(n, n_virt);
        for r in 0..n {
            for c in 0..n_virt {
                c_virt_block[(r, c)] = c_final[(r, n_closed + n_open + c)];
            }
        }
        let rotated_v = c_virt_block.dot(&uv);
        for r in 0..n {
            for c in 0..n_virt {
                c_final[(r, n_closed + n_open + c)] = rotated_v[(r, c)];
            }
        }
    }

    // Recompute exact idempotent densities and Fock matrices from canonical c_final
    let mut p_alpha = Matrix::zeros(n, n);
    let mut p_beta = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut sum_closed = 0.0;
            for i in 0..n_closed {
                sum_closed += c_final[(mu, i)] * c_final[(nu, i)];
            }
            let mut sum_open = 0.0;
            for m in n_closed..(n_closed + n_open) {
                sum_open += c_final[(mu, m)] * c_final[(nu, m)];
            }
            p_beta[(mu, nu)] = sum_closed;
            p_alpha[(mu, nu)] = sum_closed + sum_open;
        }
    }
    let (fock_alpha, fock_beta) = build_uhf_fock_matrices(&h, &p_alpha, &p_beta, &eri, n);
    let mut final_e_elec = 0.0;
    let p_tot = p_alpha.add(&p_beta);
    for i in 0..n {
        for j in 0..n {
            final_e_elec += 0.5 * (p_tot[(i, j)] * h[(i, j)]
                + p_alpha[(i, j)] * fock_alpha[(i, j)]
                + p_beta[(i, j)] * fock_beta[(i, j)]);
        }
    }

    // Exact spin expectation value S(S+1)
    let s_exact = (two_s as f64 / 2.0) * (two_s as f64 / 2.0 + 1.0);
    let properties = compute_molecular_properties(atoms, basis, &p_tot, &s);

    Ok(ROHFResult {
        total_energy: final_e_elec + e_nuc,
        electronic_energy: final_e_elec,
        nuclear_repulsion_energy: e_nuc,
        orbital_energies,
        mo_coefficients: c_final,
        density_matrix_alpha: p_alpha,
        density_matrix_beta: p_beta,
        fock_matrix_alpha: fock_alpha,
        fock_matrix_beta: fock_beta,
        overlap_matrix: s,
        core_hamiltonian: h,
        eri,
        s_squared: s_exact,
        iterations: iter_count,
        converged,
        charge,
        multiplicity,
        n_closed,
        n_open,
        properties,
    })
}

/// Computes the complete analytical nuclear gradient for Restricted Open-Shell Hartree-Fock (ROHF)
/// using the Roothaan-Goddard energy-weighted density matrix and exact open-shell two-electron contractions.
pub fn compute_rohf_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    rohf: &ROHFResult,
) -> Vec<[f64; 3]> {
    let n_atoms = atoms.len();
    let n_basis = basis.len();
    let pa = &rohf.density_matrix_alpha;
    let pb = &rohf.density_matrix_beta;
    let fa = &rohf.fock_matrix_alpha;
    let fb = &rohf.fock_matrix_beta;

    // Exact open-shell energy-weighted density matrix: W_AO = P^alpha F^alpha P^alpha + P^beta F^beta P^beta
    let w_a = pa.dot(fa).dot(pa);
    let w_b = pb.dot(fb).dot(pb);
    let w_ao = w_a.add(&w_b);

    let p_tot = pa.add(pb);

    let nuc_grad = crate::gradient::compute_nuclear_repulsion_gradient(atoms);

    let mut basis_derivs = Vec::with_capacity(n_basis);
    for cg in basis {
        let dx = crate::gradient::differentiate_basis_function(cg, 0);
        let dy = crate::gradient::differentiate_basis_function(cg, 1);
        let dz = crate::gradient::differentiate_basis_function(cg, 2);
        basis_derivs.push([dx, dy, dz]);
    }

    (0..n_atoms)
        .into_par_iter()
        .map(|atom_idx| {
            let mut grad_a = nuc_grad[atom_idx];

            for tau in 0..3 {
                let mut d_s_val = 0.0;
                let mut d_t_val = 0.0;
                let mut d_v_val = 0.0;

                for mu in 0..n_basis {
                    let mu_on_a = basis[mu].atom_index == atom_idx;
                    for nu in 0..n_basis {
                        let nu_on_a = basis[nu].atom_index == atom_idx;
                        let p_munu = p_tot[(mu, nu)];
                        let w_munu = w_ao[(mu, nu)];

                        if p_munu.abs() < 1e-15 && w_munu.abs() < 1e-15 {
                            continue;
                        }

                        let mut ds = 0.0;
                        if mu_on_a {
                            ds += crate::gradient::deriv_overlap(&basis_derivs[mu][tau], &basis[nu]);
                        }
                        if nu_on_a {
                            ds += crate::gradient::deriv_overlap(&basis_derivs[nu][tau], &basis[mu]);
                        }
                        d_s_val += w_munu * ds;

                        let mut dt = 0.0;
                        if mu_on_a {
                            dt += crate::gradient::deriv_kinetic(&basis_derivs[mu][tau], &basis[nu]);
                        }
                        if nu_on_a {
                            dt += crate::gradient::deriv_kinetic(&basis_derivs[nu][tau], &basis[mu]);
                        }
                        d_t_val += p_munu * dt;

                        let mut dv = 0.0;
                        if mu_on_a {
                            dv += crate::gradient::deriv_nuclear_attraction(&basis_derivs[mu][tau], &basis[nu], atoms);
                        }
                        if nu_on_a {
                            dv += crate::gradient::deriv_nuclear_attraction(&basis_derivs[nu][tau], &basis[mu], atoms);
                        }

                        let target_atom = &atoms[atom_idx];
                        let hf_term = -(
                            crate::gradient::deriv_nuclear_attraction_single_atom(&basis_derivs[mu][tau], &basis[nu], target_atom)
                            + crate::gradient::deriv_nuclear_attraction_single_atom(&basis_derivs[nu][tau], &basis[mu], target_atom)
                        );
                        dv += hf_term;

                        d_v_val += p_munu * dv;
                    }
                }

                grad_a[tau] += (d_t_val + d_v_val) - d_s_val;

                // 2-electron integral derivative contraction for ROHF:
                let mut d_eri_val = 0.0;
                for mu in 0..n_basis {
                    if basis[mu].atom_index != atom_idx {
                        continue;
                    }

                    for term in &basis_derivs[mu][tau].terms {
                        for nu in 0..n_basis {
                            for lambda in 0..n_basis {
                                for sigma in 0..n_basis {
                                    let coulomb = 2.0 * p_tot[(mu, nu)] * p_tot[(lambda, sigma)];
                                    let exchange = 2.0 * (pa[(mu, lambda)] * pa[(nu, sigma)] + pb[(mu, lambda)] * pb[(nu, sigma)]);
                                    let gamma = coulomb - exchange;

                                    if gamma.abs() < 1e-15 {
                                        continue;
                                    }

                                    let eri_val = crate::integrals::contracted_eri(term, &basis[nu], &basis[lambda], &basis[sigma]);
                                    d_eri_val += gamma * eri_val;
                                }
                            }
                        }
                    }
                }

                grad_a[tau] += d_eri_val;
            }

            grad_a
        })
        .collect()
}

/// Quasi-Restricted Hartree-Fock (QRHF) solver.
///
/// Obtains spin-adapted spatial orbitals by diagonalizing the state-averaged
/// density matrix P_avg = 0.5 * (P_alpha + P_beta) from an open-shell reference,
/// canonicalizing each subspace under the average Fock operator.
pub fn run_qrhf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    multiplicity: usize,
    max_iter: usize,
    tol: f64,
) -> Result<ROHFResult, String> {
    // Obtain unrestricted reference (or ROHF fallback) for QRHF natural orbitals
    let (p_alpha_ref, p_beta_ref, e_nuc, eri, h_core, s, reference_iterations, reference_converged) = match crate::scf::run_uhf(atoms, basis, charge, multiplicity, false, max_iter, tol) {
        Ok(uhf) => (
            uhf.density_matrix_alpha,
            uhf.density_matrix_beta,
            uhf.nuclear_repulsion_energy,
            uhf.eri,
            uhf.core_hamiltonian,
            uhf.overlap_matrix,
            uhf.iterations, uhf.converged,
        ),
        Err(_) => {
            let rohf = run_rohf(atoms, basis, charge, multiplicity, max_iter, tol)?;
            (
                rohf.density_matrix_alpha,
                rohf.density_matrix_beta,
                rohf.nuclear_repulsion_energy,
                rohf.eri,
                rohf.core_hamiltonian,
                rohf.overlap_matrix,
                rohf.iterations, rohf.converged,
            )
        }
    };

    let n = basis.len();
    let total_nuclear_charge: f64 = atoms.iter().map(|a| a.charge).sum();
    let total_electrons = (total_nuclear_charge.round() as i32) - charge;
    if total_electrons <= 0 {
        return Err(format!("Invalid electron count {} for QRHF", total_electrons));
    }
    let n_electrons = total_electrons as usize;
    let n_unpaired = multiplicity.saturating_sub(1);
    if n_electrons < n_unpaired || !(n_electrons - n_unpaired).is_multiple_of(2) {
        return Err(format!(
            "Electron count {} incompatible with multiplicity {}",
            n_electrons, multiplicity
        ));
    }
    let n_open = n_unpaired;
    let n_closed = (n_electrons - n_open) / 2;
    let n_alpha = n_closed + n_open;
    let n_beta = n_closed;

    if !reference_converged {return Err("QRHF reference SCF did not converge".into());}
    let x_ortho=s.canonical_orthogonalizer(1e-8).map_err(|e|e.to_string())?;
    let n_mo=x_ortho.cols;
    if n_alpha>n_mo {return Err("QRHF electron count exceeds retained overlap rank".into());}
    let p_avg=p_alpha_ref.add(&p_beta_ref).scale(0.5);
    let p_ortho=x_ortho.transpose().dot(&s).dot(&p_avg).dot(&s).dot(&x_ortho);
    let (evals, evecs) = p_ortho.eigh();

    // Sort natural orbitals in descending occupation order: ~1.0 (closed) -> ~0.5 (open) -> ~0.0 (virtual)
    let mut order: Vec<usize> = (0..n_mo).collect();
    order.sort_by(|&a, &b| evals[b].partial_cmp(&evals[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut u_sorted = Matrix::zeros(n_mo, n_mo);
    for (new_idx, &old_idx) in order.iter().enumerate() {
        for r in 0..n_mo {
            u_sorted[(r, new_idx)] = evecs[(r, old_idx)];
        }
    }

    // Natural orbitals in AO basis satisfying C^T S C = I
    let c_qrhf = x_ortho.dot(&u_sorted);

    // Initial density and Fock matrices from natural orbital occupation
    let mut p_a_init = Matrix::zeros(n, n);
    let mut p_b_init = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut sum_a = 0.0;
            for i in 0..n_alpha {
                sum_a += c_qrhf[(mu, i)] * c_qrhf[(nu, i)];
            }
            p_a_init[(mu, nu)] = sum_a;

            let mut sum_b = 0.0;
            for i in 0..n_beta {
                sum_b += c_qrhf[(mu, i)] * c_qrhf[(nu, i)];
            }
            p_b_init[(mu, nu)] = sum_b;
        }
    }
    let (f_a_init, f_b_init) = crate::scf::build_uhf_fock_matrices(&h_core, &p_a_init, &p_b_init, &eri, n);
    let f_avg = f_a_init.add(&f_b_init).scale(0.5);
    let f_mo = c_qrhf.transpose().dot(&f_avg).dot(&c_qrhf);

    let mut c_final = c_qrhf.clone();
    let mut orbital_energies = vec![0.0; n_mo];

    // Canonicalize closed block (0..n_closed)
    if n_closed > 0 {
        let mut fc = Matrix::zeros(n_closed, n_closed);
        for i in 0..n_closed {
            for j in 0..n_closed {
                fc[(i, j)] = f_mo[(i, j)];
            }
        }
        let (evals_c, uc) = fc.eigh();
        orbital_energies[..n_closed].copy_from_slice(&evals_c[..n_closed]);
        let mut c_closed_block = Matrix::zeros(n, n_closed);
        for r in 0..n {
            for c in 0..n_closed {
                c_closed_block[(r, c)] = c_qrhf[(r, c)];
            }
        }
        let rot_c = c_closed_block.dot(&uc);
        for r in 0..n {
            for c in 0..n_closed {
                c_final[(r, c)] = rot_c[(r, c)];
            }
        }
    }

    // Canonicalize open block (n_closed..n_closed + n_open)
    if n_open > 0 {
        let mut fo = Matrix::zeros(n_open, n_open);
        for i in 0..n_open {
            for j in 0..n_open {
                fo[(i, j)] = f_mo[(n_closed + i, n_closed + j)];
            }
        }
        let (evals_o, uo) = fo.eigh();
        orbital_energies[n_closed..(n_closed + n_open)].copy_from_slice(&evals_o[..n_open]);
        let mut c_open_block = Matrix::zeros(n, n_open);
        for r in 0..n {
            for c in 0..n_open {
                c_open_block[(r, c)] = c_qrhf[(r, n_closed + c)];
            }
        }
        let rot_o = c_open_block.dot(&uo);
        for r in 0..n {
            for c in 0..n_open {
                c_final[(r, n_closed + c)] = rot_o[(r, c)];
            }
        }
    }

    // Canonicalize virtual block (n_closed + n_open..n)
    let n_virt = n_mo - n_closed - n_open;
    if n_virt > 0 {
        let mut fv = Matrix::zeros(n_virt, n_virt);
        for i in 0..n_virt {
            for j in 0..n_virt {
                fv[(i, j)] = f_mo[(n_closed + n_open + i, n_closed + n_open + j)];
            }
        }
        let (evals_v, uv) = fv.eigh();
        orbital_energies[(n_closed + n_open)..n_mo].copy_from_slice(&evals_v[..n_virt]);
        let mut c_virt_block = Matrix::zeros(n, n_virt);
        for r in 0..n {
            for c in 0..n_virt {
                c_virt_block[(r, c)] = c_qrhf[(r, n_closed + n_open + c)];
            }
        }
        let rot_v = c_virt_block.dot(&uv);
        for r in 0..n {
            for c in 0..n_virt {
                c_final[(r, n_closed + n_open + c)] = rot_v[(r, c)];
            }
        }
    }

    // Recompute exact density matrices and total energy from c_final
    let mut p_alpha = Matrix::zeros(n, n);
    let mut p_beta = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut sum_a = 0.0;
            for i in 0..n_alpha {
                sum_a += c_final[(mu, i)] * c_final[(nu, i)];
            }
            p_alpha[(mu, nu)] = sum_a;

            let mut sum_b = 0.0;
            for i in 0..n_beta {
                sum_b += c_final[(mu, i)] * c_final[(nu, i)];
            }
            p_beta[(mu, nu)] = sum_b;
        }
    }

    let (f_alpha, f_beta) = crate::scf::build_uhf_fock_matrices(&h_core, &p_alpha, &p_beta, &eri, n);
    let mut e_elec = 0.0;
    for mu in 0..n {
        for nu in 0..n {
            e_elec += 0.5 * (p_alpha[(mu, nu)] * (h_core[(mu, nu)] + f_alpha[(mu, nu)])
                + p_beta[(mu, nu)] * (h_core[(mu, nu)] + f_beta[(mu, nu)]));
        }
    }
    let total_energy = e_elec + e_nuc;

    // S^2 for open-shell state with identical spatial orbitals: S_z(S_z + 1)
    let s_z = 0.5 * (n_open as f64);
    let s_squared = s_z * (s_z + 1.0);

    let p_tot = p_alpha.add(&p_beta);
    let properties = crate::properties::compute_molecular_properties(atoms, basis, &p_tot, &s);

    Ok(ROHFResult {
        total_energy,
        electronic_energy: e_elec,
        nuclear_repulsion_energy: e_nuc,
        orbital_energies,
        mo_coefficients: c_final,
        density_matrix_alpha: p_alpha,
        density_matrix_beta: p_beta,
        fock_matrix_alpha: f_alpha,
        fock_matrix_beta: f_beta,
        overlap_matrix: s,
        core_hamiltonian: h_core,
        eri,
        s_squared,
        n_closed,
        n_open,
        multiplicity,
        charge,
        iterations: reference_iterations,
        converged: reference_converged,
        properties,
    })
}

/// Maximum Overlap Method (MOM-SCF) for targeting excited or core-hole states.
///
/// Prevents variational collapse down to the ground state by replacing the Aufbau
/// orbital occupation ordering with projection onto a target orbital subspace:
/// p_j = \sqrt{ \sum_{i \in \text{target}} (C_{\text{target}}^T S C_j)^2 }.
pub fn run_scf_mom(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    target_occ_indices: &[usize],
    max_iter: usize,
    tol: f64,
) -> Result<HFResult, String> {
    let n = basis.len();
    if max_iter==0 || !tol.is_finite() || tol<=0.0 {return Err("Invalid MOM SCF controls".into());}
    let total_nuclear_charge: f64 = atoms.iter().map(|a| a.charge).sum();
    let total_electrons = (total_nuclear_charge.round() as i32) - charge;

    if total_electrons <= 0 {
        return Err(format!("Invalid electron count {} for charge {}", total_electrons, charge));
    }
    let n_electrons = total_electrons as usize;
    if !n_electrons.is_multiple_of(2) {
        return Err("MOM-SCF closed-shell requires an even number of electrons".to_string());
    }
    let n_occ = n_electrons / 2;

    if target_occ_indices.len() != n_occ {
        return Err(format!(
            "Target occupation indices count ({}) does not match n_occ ({})",
            target_occ_indices.len(), n_occ
        ));
    }

    let e_nuc = compute_nuclear_repulsion(atoms);
    let (s, _t, _v, h) = compute_one_electron_integrals(atoms, basis);
    let mut space=crate::ao::AoSpace::new(basis,crate::ao::BasisKind::Cartesian).map_err(|e|e.to_string())?;
    let x_ortho=space.orthogonalizer(&s).map_err(|e|e.to_string())?;
    let n_mo=x_ortho.cols;
    let unique=target_occ_indices.iter().copied().collect::<std::collections::HashSet<_>>();
    if unique.len()!=n_occ || target_occ_indices.iter().any(|&i|i>=n_mo) {
        return Err("MOM occupation indices must be distinct and within retained MO space".into());
    }
    let eri = compute_eri_tensor(basis);

    // Initial reference orbitals from core Hamiltonian
    let f_prime = x_ortho.transpose().dot(&h).dot(&x_ortho);
    let (_evals, c_prime) = f_prime.eigh();
    let mut c_target = x_ortho.dot(&c_prime);

    // Build initial density from target occupied orbitals
    let mut p = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut sum = 0.0;
            for &idx in target_occ_indices {
                sum += 2.0 * c_target[(mu, idx)] * c_target[(nu, idx)];
            }
            p[(mu, nu)] = sum;
        }
    }

    let mut prev_energy = 0.0;
    let mut converged = false;
    let mut fock;
    let mut mo_energies = vec![0.0; n_mo];
    let mut c_current = c_target.clone();
    let mut iter_count = 0;

    let mut diis=crate::diis::CdiisEngine::new(8);
    let gradient_tol=(tol.sqrt()*1e-2).max(1e-11);
    for iteration in 1..=max_iter {
        iter_count = iteration;
        fock = build_fock_matrix(&h, &p, &eri, n);

        let mut e_elec = 0.0;
        for i in 0..n {
            for j in 0..n {
                e_elec += 0.5 * p[(i, j)] * (h[(i, j)] + fock[(i, j)]);
            }
        }
        let total_e = e_elec + e_nuc;

        let delta_e = (total_e - prev_energy).abs();
        prev_energy = total_e;

        let error=crate::diis::compute_diis_error(&fock,&p,&s);
        let error=x_ortho.transpose().dot(&error).dot(&x_ortho);
        let error_norm=crate::diis::max_abs_element(&error);
        diis.push(fock.clone(),error);
        let f_use=diis.extrapolate().unwrap_or_else(||fock.clone());
        let f_prime = x_ortho.transpose().dot(&f_use).dot(&x_ortho);
        let (evals, c_prime) = f_prime.eigh();
        let c_new = x_ortho.dot(&c_prime);

        // Maximum Overlap Method projection:
        // Overlap matrix in MO space: O = C_target^T * S * C_new
        let o_mat = c_target.transpose().dot(&s).dot(&c_new);

        // Compute projections p_j for each candidate orbital j
        let target_cols_vec: Vec<usize> = if iteration == 1 {
            target_occ_indices.to_vec()
        } else {
            (0..n_occ).collect()
        };
        let mut projections = Vec::with_capacity(n_mo);
        for j in 0..n_mo {
            let mut proj_sq = 0.0;
            for &i in &target_cols_vec {
                let overlap = o_mat[(i, j)];
                proj_sq += overlap * overlap;
            }
            projections.push((proj_sq.sqrt(), j));
        }

        // Sort candidate orbitals by descending projection
        projections.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let selected_indices: Vec<usize> = projections.iter().take(n_occ).map(|p| p.1).collect();

        // Update density matrix with MOM-selected orbitals
        let mut p_new = Matrix::zeros(n, n);
        for mu in 0..n {
            for nu in 0..n {
                let mut sum = 0.0;
                for &idx in &selected_indices {
                    sum += 2.0 * c_new[(mu, idx)] * c_new[(nu, idx)];
                }
                p_new[(mu, nu)] = sum;
            }
        }

        // Place MOM-selected occupied orbitals in columns 0..n_occ
        let mut ordered_c = Matrix::zeros(n, n_mo);
        let mut ordered_evals = vec![0.0; n_mo];
        for (new_col, &old_col) in selected_indices.iter().enumerate() {
            for r in 0..n {
                ordered_c[(r, new_col)] = c_new[(r, old_col)];
            }
            ordered_evals[new_col] = evals[old_col];
        }
        let mut virt_col = n_occ;
        for old_col in 0..n_mo {
            if !selected_indices.contains(&old_col) {
                for r in 0..n {
                    ordered_c[(r, virt_col)] = c_new[(r, old_col)];
                }
                ordered_evals[virt_col] = evals[old_col];
                virt_col += 1;
            }
        }

        let density_change=p.rmsd(&p_new);
        converged=iteration>1 && delta_e<tol && error_norm<gradient_tol
            && density_change<(tol.sqrt()*0.1).clamp(1e-12,1e-10);
        p = if iteration<=3 && !converged {p_new.scale(0.7).add(&p.scale(0.3))} else {p_new};
        c_current = ordered_c;
        c_target = c_current.clone();
        mo_energies = ordered_evals;
        if converged {break;}
    }

    // Return one coherent state even when the iteration limit was reached.
    p=c_current.occupied_density(n_occ,2.0);
    fock=build_fock_matrix(&h,&p,&eri,n);
    let final_e_elec=0.5*p.data.iter().zip(h.add(&fock).data).map(|(p,hf)|p*hf).sum::<f64>();
    let final_mo=c_current.transpose().dot(&fock).dot(&c_current);
    for i in 0..n_mo {mo_energies[i]=final_mo[(i,i)];}
    let properties = compute_molecular_properties(atoms, basis, &p, &s);

    Ok(HFResult {
        ao_space: Some(std::sync::Arc::new(space)),
        total_energy: final_e_elec + e_nuc,
        electronic_energy: final_e_elec,
        nuclear_repulsion_energy: e_nuc,
        orbital_energies: mo_energies,
        mo_coefficients: c_current,
        density_matrix: p,
        fock_matrix: fock,
        overlap_matrix: s,
        core_hamiltonian: h,
        eri,
        iterations: iter_count,
        converged,
        charge,
        multiplicity: 1,
        n_occ,
        properties,
    })
}

/// Results from a Fermi-Dirac thermal smearing Hartree-Fock calculation.
#[derive(Clone, Debug)]
pub struct FermiSmearingResult {
    pub hf: HFResult,
    pub chemical_potential: f64,
    pub occupations: Vec<f64>,
    pub electronic_entropy: f64,
    pub free_energy: f64,
    pub temperature_kelvin: f64,
}

/// Runs a closed-shell Hartree-Fock calculation with Fermi-Dirac thermal smearing.
///
/// Fractional occupations f_i = 2 / (1 + exp((eps_i - mu) / (k_B T))) are determined
/// by finding the chemical potential mu that satisfies sum_i f_i = N_elec.
/// Prevents convergence oscillations for systems with narrow HOMO-LUMO gaps.
fn fermi_population(eps:&[f64],electrons:f64,kt:f64)->(f64,Vec<f64>) {
    let value=|e:f64,mu:f64| {let z=(e-mu)/kt; if z>=0.0 {let t=(-z).exp();2.0*t/(1.0+t)} else {2.0/(1.0+z.exp())}};
    let width=(80.0*kt).max(1.0);
    let mut low=eps[0]-width;
    let mut high=eps[eps.len()-1]+width;
    for _ in 0..160 {
        let mu=0.5*(low+high);
        let sum=eps.iter().map(|&e|value(e,mu)).sum::<f64>();
        if sum<electrons {low=mu;} else {high=mu;}
    }
    let mu=0.5*(low+high);
    (mu,eps.iter().map(|&e|value(e,mu)).collect())
}

fn thermal_density(c:&Matrix,occupations:&[f64])->Matrix {
    let mut weighted=c.clone();
    for i in 0..c.rows {for j in 0..c.cols {weighted[(i,j)]*=occupations[j].sqrt();}}
    weighted.dot(&weighted.transpose())
}

fn thermal_entropy(occupations:&[f64],kb:f64)->f64 {
    -2.0*kb*occupations.iter().map(|&n| {
        let p=n/2.0;
        let a=if p>0.0 {p*p.ln()} else {0.0};
        let b=if p<1.0 {(1.0-p)*(1.0-p).ln()} else {0.0};
        a+b
    }).sum::<f64>()
}

pub fn run_scf_fermi_smearing(
    atoms:&[Atom],basis:&[ContractedGaussian],charge:i32,temperature_kelvin:f64,
    max_iter:usize,tol:f64,
) -> Result<FermiSmearingResult,String> {
    if !temperature_kelvin.is_finite() || temperature_kelvin<=0.0 || max_iter==0 || !tol.is_finite() || tol<=0.0 {
        return Err("Fermi SCF requires positive temperature, tolerance and iteration limit".into());
    }
    let n=basis.len();
    let electrons=atoms.iter().map(|a|a.charge).sum::<f64>()-charge as f64;
    let (s,_,_,h)=compute_one_electron_integrals(atoms,basis);
    let mut space=crate::ao::AoSpace::new(basis,crate::ao::BasisKind::Cartesian).map_err(|e|e.to_string())?;
    let x=space.orthogonalizer(&s).map_err(|e|e.to_string())?;
    let xt=x.transpose();
    if electrons<=0.0 || electrons>2.0*x.cols as f64 {return Err("Fermi electron count exceeds retained basis capacity".into());}
    let kb=1.0/315775.02480407;
    let kt=kb*temperature_kelvin;
    let eri=compute_eri_tensor(basis);
    let e_nuc=compute_nuclear_repulsion(atoms);
    let (mut eps,cp)=xt.dot(&h).dot(&x).eigh();
    let mut c=x.dot(&cp);
    let (mut mu,mut occ)=fermi_population(&eps,electrons,kt);
    let mut p=thermal_density(&c,&occ);
    let mut f=build_fock_matrix(&h,&p,&eri,n);
    let mut previous=f64::INFINITY;
    let mut diis=crate::diis::CdiisEngine::new(8);
    let mut converged=false;
    let mut iterations=0;
    let mut energy=0.0;
    let mut entropy=0.0;
    for iteration in 1..=max_iter {
        iterations=iteration;
        let error=crate::diis::compute_diis_error(&f,&p,&s);
        diis.push(f.clone(),xt.dot(&error).dot(&x));
        let f_use=diis.extrapolate().unwrap_or_else(||f.clone());
        let (values,cp)=xt.dot(&f_use).dot(&x).eigh();
        c=x.dot(&cp);eps=values;
        (mu,occ)=fermi_population(&eps,electrons,kt);
        let next=thermal_density(&c,&occ);
        let density_change=p.rmsd(&next);
        // All reported quantities use the same undamped density.
        f=build_fock_matrix(&h,&next,&eri,n);
        energy=0.5*next.data.iter().zip(h.add(&f).data).map(|(p,hf)|p*hf).sum::<f64>();
        entropy=thermal_entropy(&occ,kb);
        let free=energy+e_nuc-temperature_kelvin*entropy;
        // Fermi fixed-point residual is stronger than an energy-only or commutator check.
        let (true_eps,true_cp)=xt.dot(&f).dot(&x).eigh();
        let (_,true_occ)=fermi_population(&true_eps,electrons,kt);
        let fixed=thermal_density(&x.dot(&true_cp),&true_occ);
        converged=iteration>1 && (free-previous).abs()<tol
            && density_change<(tol.sqrt()*0.1).clamp(1e-12,1e-10)
            && fixed.rmsd(&next)<(tol.sqrt()*1e-2).max(1e-11);
        previous=free;
        if converged || iteration==max_iter {p=next;break;}
        p=if iteration<=3 {p.scale(0.3).add(&next.scale(0.7))} else {next};
        f=build_fock_matrix(&h,&p,&eri,n);
    }
    let properties=compute_molecular_properties(atoms,basis,&p,&s);
    let hf=HFResult {ao_space:Some(std::sync::Arc::new(space)),total_energy:energy+e_nuc,
        electronic_energy:energy,nuclear_repulsion_energy:e_nuc,orbital_energies:eps,
        mo_coefficients:c,density_matrix:p,fock_matrix:f,overlap_matrix:s,core_hamiltonian:h,eri,
        iterations,converged,charge,multiplicity:1,n_occ:(electrons as usize).div_ceil(2),properties};
    Ok(FermiSmearingResult {hf,chemical_potential:mu,occupations:occ,electronic_entropy:entropy,
        free_energy:energy+e_nuc-temperature_kelvin*entropy,temperature_kelvin})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{parse_gaussian94, STO_3G_GBS, Atom, build_basis};

    #[test]
    fn test_rohf_lithium_atom() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![Atom::new("Li", [0.0, 0.0, 0.0])];
        let basis = build_basis(&atoms, &basis_map);

        // Li doublet: charge = 0, multiplicity = 2
        let res = run_rohf(&atoms, &basis, 0, 2, 50, 1e-10).expect("ROHF Li failed");

        assert_eq!(res.n_closed, 1);
        assert_eq!(res.n_open, 1);
        assert_eq!(res.multiplicity, 2);
        // Exact <S^2> for doublet is 0.75
        assert!((res.s_squared - 0.75).abs() < 1e-12);
        assert!(res.total_energy < -7.0, "Li STO-3G energy should be around -7.3 Hartree, got {}", res.total_energy);
    }

    #[test]
    fn test_qrhf_lithium_atom() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![Atom::new("Li", [0.0, 0.0, 0.0])];
        let basis = build_basis(&atoms, &basis_map);

        let qrhf = run_qrhf(&atoms, &basis, 0, 2, 50, 1e-10).expect("QRHF Li failed");
        assert_eq!(qrhf.n_closed, 1);
        assert_eq!(qrhf.n_open, 1);
        assert!((qrhf.s_squared - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_fermi_smearing_h2() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0,  0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);

        let res = run_scf_fermi_smearing(&atoms, &basis, 0, 1000.0, 50, 1e-8)
            .expect("Fermi smearing failed");

        assert_eq!(res.occupations.len(), basis.len());
        let sum_occ: f64 = res.occupations.iter().sum();
        assert!((sum_occ - 2.0).abs() < 1e-6);
        assert!(res.free_energy <= res.hf.total_energy + 1e-10);
    }

    #[test]
    fn test_mom_scf_h2() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0,  0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);

        // Ground state has target_occ = [0]
        let res_ground = run_scf_mom(&atoms, &basis, 0, &[0], 50, 1e-8)
            .expect("MOM-SCF ground failed");
        assert!(res_ground.total_energy < -1.1);

        // Excited state targeting orbital 1: target_occ = [1]
        let res_excited = run_scf_mom(&atoms, &basis, 0, &[1], 50, 1e-8)
            .expect("MOM-SCF excited failed");
        assert!(res_excited.total_energy > res_ground.total_energy);
    }

    #[test]
    fn test_rohf_triplet_ch2() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        // Triplet Methylene CH2 (C at origin, H at bond angle ~130 deg)
        let atoms = vec![
            Atom::new("C", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0,  1.6, 1.2]),
            Atom::new("H", [0.0, -1.6, 1.2]),
        ];
        let basis = build_basis(&atoms, &basis_map);

        // CH2 triplet: charge = 0, multiplicity = 3
        let res = run_rohf(&atoms, &basis, 0, 3, 60, 1e-8).expect("ROHF CH2 triplet failed");
        assert_eq!(res.multiplicity, 3);
        assert_eq!(res.n_closed, 3);
        assert_eq!(res.n_open, 2);
        // Exact <S^2> for triplet is 1 * 2 = 2.0
        assert!((res.s_squared - 2.0).abs() < 1e-12);
        assert!(res.total_energy < -38.0);
    }

    #[test]
    fn test_rohf_doublet_oh() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        // OH radical: O at origin, H along z
        let atoms = vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 1.83]),
        ];
        let basis = build_basis(&atoms, &basis_map);

        // OH radical doublet: charge = 0, multiplicity = 2
        let res = run_rohf(&atoms, &basis, 0, 2, 60, 1e-8).expect("ROHF OH doublet failed");
        assert_eq!(res.multiplicity, 2);
        assert_eq!(res.n_closed, 4);
        assert_eq!(res.n_open, 1);
        // Exact <S^2> for doublet is 0.5 * 1.5 = 0.75
        assert!((res.s_squared - 0.75).abs() < 1e-12);
        assert!(res.total_energy < -74.0);
    }

    #[test]
    fn test_rohf_analytical_gradient_vs_finite_difference() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 1.83]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let rohf = run_rohf(&atoms, &basis, 0, 2, 60, 1e-11).expect("ROHF failed");
        let grad_anal = compute_rohf_gradient(&atoms, &basis, &rohf);

        // Translational invariance
        assert!((grad_anal[0][0] + grad_anal[1][0]).abs() < 1e-5);
        assert!((grad_anal[0][1] + grad_anal[1][1]).abs() < 1e-5);
        assert!((grad_anal[0][2] + grad_anal[1][2]).abs() < 1e-5);

        // Finite difference along z for H atom
        let h = 1e-4;
        let mut atoms_fwd = atoms.clone();
        atoms_fwd[1].position[2] += h;
        let basis_fwd = build_basis(&atoms_fwd, &basis_map);
        let rohf_fwd = run_rohf(&atoms_fwd, &basis_fwd, 0, 2, 60, 1e-11).unwrap();

        let mut atoms_bwd = atoms.clone();
        atoms_bwd[1].position[2] -= h;
        let basis_bwd = build_basis(&atoms_bwd, &basis_map);
        let rohf_bwd = run_rohf(&atoms_bwd, &basis_bwd, 0, 2, 60, 1e-11).unwrap();

        let g_num_z = (rohf_fwd.total_energy - rohf_bwd.total_energy) / (2.0 * h);
        assert!(
            (grad_anal[1][2] - g_num_z).abs() < 1e-3,
            "Analytical ROHF gradient ({}) must match finite difference ({})",
            grad_anal[1][2],
            g_num_z
        );
    }
}
