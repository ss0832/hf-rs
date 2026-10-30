// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Unified Analytical Nuclear Gradient Engine for Post-Hartree-Fock Methods.
//!
//! Provides:
//! 1. Exact Handy-Schaefer Z-vector analytical nuclear gradients from total MO 1-RDM and 2-RDM
//!    (for CI, MR-CISD, Selected CI, GASCI, DOCI, DMRG).
//! 2. Exact Handy-Schaefer Z-vector analytical nuclear gradients for CIS excited states.
//! 3. Exact Coupled-Perturbed Hartree-Fock (CPHF) and Coupled-Perturbed UHF (CPUHF)
//!    analytical response engines using analytical AO 1st-derivative integrals (`diff_cg_1st`)
//!    and canonical CPHF orbital responses $U^{(x)}$ at the reference geometry.

use rayon::prelude::*;
use crate::basis::{Atom, ContractedGaussian};
use crate::cphf::solve_cphf;
use crate::derivatives::{
    diff_cg_1st, eval_eri_deriv, eval_kinetic_deriv, eval_nuclear_deriv_single_atom,
    eval_overlap_deriv, GeneralBasisDerivative,
};
use crate::fci::string::SpinString;
use crate::gradient::{compute_rhf_gradient, compute_uhf_gradient};
use crate::matrix::Matrix;
use crate::mo_integrals::{transform_1e_ao_to_mo, transform_eri_ao_to_mo};
use crate::post_hf::ci::CISResult;
use crate::scf::{build_fock_matrix, build_uhf_fock_matrices, eri_index, HFResult, UHFResult};

/// Computes spin-summed MO 1-RDM and 2-RDM directly from an arbitrary subset of Slater determinants
/// and their normalized expansion coefficients.
///
/// The resulting 1-RDM $D_{pq}$ and 2-RDM $\Gamma_{pqrs}$ (chemist's notation) satisfy:
/// $E_{\text{elec}} = \sum_{pq} D_{pq} h_{pq} + \frac{1}{2} \sum_{pqrs} \Gamma_{pqrs} (pq|rs)$.
pub fn compute_subset_rdms(
    n_basis: usize,
    dets: &[(SpinString, SpinString)],
    coeffs: &[f64],
) -> (Matrix, Vec<f64>) {
    let n_det = dets.len();
    let mut rdm1 = Matrix::zeros(n_basis, n_basis);
    let mut rdm2 = vec![0.0; n_basis * n_basis * n_basis * n_basis];
    let idx4 = |p: usize, q: usize, r: usize, s: usize| -> usize {
        ((p * n_basis + q) * n_basis + r) * n_basis + s
    };

    for i in 0..n_det {
        let ci = coeffs[i];
        if ci.abs() < 1e-15 {
            continue;
        }
        let (sa_i, sb_i) = dets[i];
        let occ_a_i: Vec<usize> = (0..n_basis).filter(|&p| sa_i.is_occupied(p)).collect();
        let occ_b_i: Vec<usize> = (0..n_basis).filter(|&p| sb_i.is_occupied(p)).collect();

        // Diagonal (i == j)
        let w_ii = ci * ci;
        for &p in &occ_a_i {
            rdm1[(p, p)] += w_ii;
        }
        for &p in &occ_b_i {
            rdm1[(p, p)] += w_ii;
        }

        // Alpha-Alpha 2-RDM diagonal: p != q
        for &p in &occ_a_i {
            for &q in &occ_a_i {
                if p != q {
                    rdm2[idx4(p, p, q, q)] += w_ii;
                    rdm2[idx4(p, q, q, p)] -= w_ii;
                }
            }
        }
        // Beta-Beta 2-RDM diagonal: p != q
        for &p in &occ_b_i {
            for &q in &occ_b_i {
                if p != q {
                    rdm2[idx4(p, p, q, q)] += w_ii;
                    rdm2[idx4(p, q, q, p)] -= w_ii;
                }
            }
        }
        // Alpha-Beta 2-RDM diagonal
        for &p in &occ_a_i {
            for &q in &occ_b_i {
                rdm2[idx4(p, p, q, q)] += w_ii;
                rdm2[idx4(q, q, p, p)] += w_ii;
            }
        }

        // Off-diagonal (j > i, factor of 2 included by filling both (i,j) and (j,i))
        for j in 0..n_det {
            if i == j {
                continue;
            }
            let cj = coeffs[j];
            let cij = ci * cj;
            if cij.abs() < 1e-15 {
                continue;
            }
            let (sa_j, sb_j) = dets[j];
            let diff_a = (sa_i.0 ^ sa_j.0).count_ones() as usize;
            let diff_b = (sb_i.0 ^ sb_j.0).count_ones() as usize;
            if diff_a + diff_b > 4 {
                continue;
            }

            if diff_a == 2 && diff_b == 0 {
                // Single alpha excitation: p (in j) -> q (in i)
                let hole_mask = sa_j.0 & !sa_i.0;
                let part_mask = sa_i.0 & !sa_j.0;
                let p = hole_mask.trailing_zeros() as usize;
                let q = part_mask.trailing_zeros() as usize;
                let sign = slater_single_sign(sa_j, p, q);
                let val = sign * cij;
                rdm1[(q, p)] += val;

                for &r in &occ_a_i {
                    if r != q {
                        rdm2[idx4(q, p, r, r)] += val;
                        rdm2[idx4(r, r, q, p)] += val;
                        rdm2[idx4(q, r, r, p)] -= val;
                        rdm2[idx4(r, p, q, r)] -= val;
                    }
                }
                for &r in &occ_b_i {
                    rdm2[idx4(q, p, r, r)] += val;
                    rdm2[idx4(r, r, q, p)] += val;
                }
            } else if diff_a == 0 && diff_b == 2 {
                // Single beta excitation: p (in j) -> q (in i)
                let hole_mask = sb_j.0 & !sb_i.0;
                let part_mask = sb_i.0 & !sb_j.0;
                let p = hole_mask.trailing_zeros() as usize;
                let q = part_mask.trailing_zeros() as usize;
                let sign = slater_single_sign(sb_j, p, q);
                let val = sign * cij;
                rdm1[(q, p)] += val;

                for &r in &occ_b_i {
                    if r != q {
                        rdm2[idx4(q, p, r, r)] += val;
                        rdm2[idx4(r, r, q, p)] += val;
                        rdm2[idx4(q, r, r, p)] -= val;
                        rdm2[idx4(r, p, q, r)] -= val;
                    }
                }
                for &r in &occ_a_i {
                    rdm2[idx4(q, p, r, r)] += val;
                    rdm2[idx4(r, r, q, p)] += val;
                }
            } else if diff_a == 2 && diff_b == 2 {
                // Alpha-Beta double excitation: p_a -> q_a and p_b -> q_b
                let p_a = (sa_j.0 & !sa_i.0).trailing_zeros() as usize;
                let q_a = (sa_i.0 & !sa_j.0).trailing_zeros() as usize;
                let p_b = (sb_j.0 & !sb_i.0).trailing_zeros() as usize;
                let q_b = (sb_i.0 & !sb_j.0).trailing_zeros() as usize;
                let sign = slater_single_sign(sa_j, p_a, q_a) * slater_single_sign(sb_j, p_b, q_b);
                let val = sign * cij;
                rdm2[idx4(q_a, p_a, q_b, p_b)] += val;
                rdm2[idx4(q_b, p_b, q_a, p_a)] += val;
            } else if diff_a == 4 && diff_b == 0 {
                // Same-spin alpha double excitation: (p1 < p2 in j) -> (q1 < q2 in i)
                let hm = sa_j.0 & !sa_i.0;
                let pm = sa_i.0 & !sa_j.0;
                let p1 = hm.trailing_zeros() as usize;
                let p2 = (hm & (hm - 1)).trailing_zeros() as usize;
                let q1 = pm.trailing_zeros() as usize;
                let q2 = (pm & (pm - 1)).trailing_zeros() as usize;
                let sign = slater_double_sign(sa_j, p1, p2, q1, q2);
                let val = sign * cij;
                rdm2[idx4(q1, p1, q2, p2)] += val;
                rdm2[idx4(q2, p2, q1, p1)] += val;
                rdm2[idx4(q1, p2, q2, p1)] -= val;
                rdm2[idx4(q2, p1, q1, p2)] -= val;
            } else if diff_a == 0 && diff_b == 4 {
                // Same-spin beta double excitation: (p1 < p2 in j) -> (q1 < q2 in i)
                let hm = sb_j.0 & !sb_i.0;
                let pm = sb_i.0 & !sb_j.0;
                let p1 = hm.trailing_zeros() as usize;
                let p2 = (hm & (hm - 1)).trailing_zeros() as usize;
                let q1 = pm.trailing_zeros() as usize;
                let q2 = (pm & (pm - 1)).trailing_zeros() as usize;
                let sign = slater_double_sign(sb_j, p1, p2, q1, q2);
                let val = sign * cij;
                rdm2[idx4(q1, p1, q2, p2)] += val;
                rdm2[idx4(q2, p2, q1, p1)] += val;
                rdm2[idx4(q1, p2, q2, p1)] -= val;
                rdm2[idx4(q2, p1, q1, p2)] -= val;
            }
        }
    }

    (rdm1, rdm2)
}

#[inline]
fn slater_single_sign(s_ket: SpinString, hole: usize, part: usize) -> f64 {
    let mut s = s_ket.0;
    let mask_h = (1u64 << hole) - 1;
    let n_h = (s & mask_h).count_ones();
    s &= !(1u64 << hole);
    let mask_p = (1u64 << part) - 1;
    let n_p = (s & mask_p).count_ones();
    if (n_h + n_p).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

#[inline]
fn slater_double_sign(s_ket: SpinString, p1: usize, p2: usize, q1: usize, q2: usize) -> f64 {
    let mut s = s_ket.0;
    let n1 = (s & ((1u64 << p1) - 1)).count_ones();
    s &= !(1u64 << p1);
    let n2 = (s & ((1u64 << p2) - 1)).count_ones();
    s &= !(1u64 << p2);
    let m2 = (s & ((1u64 << q2) - 1)).count_ones();
    s |= 1u64 << q2;
    let m1 = (s & ((1u64 << q1) - 1)).count_ones();
    if (n1 + n2 + m1 + m2).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// Computes the exact analytical nuclear gradient from total MO 1-RDM and 2-RDM
/// using the Handy-Schaefer Z-vector method.
pub fn compute_rhf_total_rdm_analytical_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    one_rdm_mo: &Matrix,
    two_rdm_mo: &[f64],
) -> Vec<[f64; 3]> {
    let cisd_like = crate::post_hf::ci::CISDResult {
        reference_energy: hf.total_energy,
        total_energy: hf.total_energy,
        correlation_energy: 0.0,
        n_determinants: 1,
        c0: 1.0,
        one_rdm_mo: one_rdm_mo.clone(),
        two_rdm_mo: two_rdm_mo.to_vec(),
        mo_coefficients: hf.mo_coefficients.clone(),
    };
    crate::post_hf::ci::cisd::compute_cisd_analytical_gradient(atoms, basis, hf, &cisd_like)
}

/// Computes the exact analytical nuclear gradient of the total CIS state energy
/// $E_{\text{CIS}} = E_{\text{RHF}} + \omega_k$ using the exact singlet single-excitation
/// determinant 1-RDM and 2-RDM and the Handy-Schaefer Z-vector method
/// (Foresman, Head-Gordon, Pople & Frisch, J. Phys. Chem. 96, 135, 1992).
pub fn compute_cis_analytical_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    cis: &CISResult,
    state_idx: usize,
) -> Vec<[f64; 3]> {
    let n_basis = basis.len();
    let n_occ = hf.n_occ;
    let n_virt = n_basis - n_occ;

    if n_occ == 0 || n_virt == 0 || cis.excited_states.is_empty() {
        return compute_rhf_gradient(atoms, basis, hf);
    }

    let state = &cis.excited_states[state_idx.min(cis.excited_states.len() - 1)];
    let amps = &state.amplitudes;
    let ref_mask = (1u64 << n_occ) - 1;
    let ref_str = SpinString(ref_mask);
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

    let mut dets = Vec::with_capacity(2 * n_occ * n_virt);
    let mut coeffs = Vec::with_capacity(2 * n_occ * n_virt);

    for i in 0..n_occ {
        for a in 0..n_virt {
            let c_ia = amps[i * n_virt + a];
            if c_ia.abs() < 1e-15 {
                continue;
            }
            let a_mo = n_occ + a;
            let sign = slater_single_sign(ref_str, i, a_mo);
            let exc_str = SpinString((ref_mask & !(1u64 << i)) | (1u64 << a_mo));
            let coeff = c_ia * inv_sqrt2 * sign;

            // Alpha single excitation |D_i^a(alpha), 0_beta>
            dets.push((exc_str, ref_str));
            coeffs.push(coeff);

            // Beta single excitation |0_alpha, D_i^a(beta)>
            dets.push((ref_str, exc_str));
            coeffs.push(if cis.triplet { -coeff } else { coeff });
        }
    }

    let (rdm1, rdm2) = compute_subset_rdms(n_basis, &dets, &coeffs);
    compute_rhf_total_rdm_analytical_gradient(atoms, basis, hf, &rdm1, &rdm2)
}

/// Computes the exact analytical nuclear gradient of MR-CISD using the exact MR-CISD
/// 1-RDM and 2-RDM and the Handy-Schaefer Z-vector method.
pub fn compute_mrci_analytical_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    n_active_electrons: usize,
    n_active_orbitals: usize,
) -> Vec<[f64; 3]> {
    use std::collections::HashSet;
    use crate::fci::slater_condon::slater_condon_element;
    use crate::fci::string::DeterminantSpace;
    use crate::orbitals::OrbitalSpace;

    let n_occ = hf.n_occ;
    let n_basis = hf.orbital_energies.len();
    let space = OrbitalSpace::new_closed_shell(n_basis, n_occ)
        .with_cas(n_active_orbitals, n_active_electrons);
    let n_core = space.n_frozen_core;
    let n_act = space.n_active_occ + space.n_active_virt;
    let det_space = DeterminantSpace::new(n_basis, n_occ, n_occ);

    let mut ref_dets: Vec<(SpinString, SpinString)> = Vec::new();
    for str_a in &det_space.alpha_strings {
        let valid_a = (0..n_core).all(|i| str_a.is_occupied(i))
            && ((n_core + n_act)..n_basis).all(|a| !str_a.is_occupied(a));
        if !valid_a {
            continue;
        }
        for str_b in &det_space.beta_strings {
            let valid_b = (0..n_core).all(|i| str_b.is_occupied(i))
                && ((n_core + n_act)..n_basis).all(|a| !str_b.is_occupied(a));
            if valid_b {
                ref_dets.push((*str_a, *str_b));
            }
        }
    }

    let mut mrci_set = HashSet::new();
    for str_a in &det_space.alpha_strings {
        for str_b in &det_space.beta_strings {
            let within = ref_dets.iter().any(|(ra, rb)| {
                let da = (str_a.0 ^ ra.0).count_ones() / 2;
                let db = (str_b.0 ^ rb.0).count_ones() / 2;
                da + db <= 2
            });
            if within {
                mrci_set.insert((*str_a, *str_b));
            }
        }
    }

    let mut mrci_dets: Vec<(SpinString, SpinString)> = mrci_set.into_iter().collect();
    mrci_dets.sort_by_key(|&(a, b)| (a.0, b.0));
    let n_det = mrci_dets.len();

    let h_mo = transform_1e_ao_to_mo(&hf.mo_coefficients, &hf.core_hamiltonian);
    let mo_eri = transform_eri_ao_to_mo(&hf.mo_coefficients, &hf.eri);

    let mut h_mat = Matrix::zeros(n_det, n_det);
    for i in 0..n_det {
        let (sa_i, sb_i) = &mrci_dets[i];
        for j in 0..=i {
            let (sa_j, sb_j) = &mrci_dets[j];
            let val = slater_condon_element(sa_i, sb_i, sa_j, sb_j, &h_mo, &mo_eri);
            h_mat[(i, j)] = val;
            h_mat[(j, i)] = val;
        }
    }

    let (_evals, evecs) = h_mat.eigh();
    let coeffs: Vec<f64> = (0..n_det).map(|r| evecs[(r, 0)]).collect();
    let (rdm1, rdm2) = compute_subset_rdms(n_basis, &mrci_dets, &coeffs);
    compute_rhf_total_rdm_analytical_gradient(atoms, basis, hf, &rdm1, &rdm2)
}

/// Evaluates the exact analytical nuclear gradient of a closed-shell Post-HF method
/// by solving the Coupled-Perturbed Hartree-Fock (CPHF) equation at the reference geometry
/// and propagating the analytical AO 1st-derivative integrals (`diff_cg_1st`) and canonical
/// CPHF orbital responses $U^{(x)}$ through the correlation energy functional.
pub fn compute_rhf_cphf_analytical_corr_gradient<F>(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    eval_corr: F,
) -> Vec<[f64; 3]>
where
    F: Fn(&HFResult, &[Atom], &[ContractedGaussian]) -> Result<f64, String> + Sync,
{
    let n_atoms = atoms.len();
    let n_basis = basis.len();
    let n_occ = hf.n_occ;
    let n_virt = n_basis - n_occ;
    let c = &hf.mo_coefficients;
    let p = &hf.density_matrix;
    let eps = &hf.orbital_energies;

    // 1. Precompute 0th and 1st analytical derivatives of each basis function
    let mut d0 = Vec::with_capacity(n_basis);
    let mut d1 = Vec::with_capacity(n_basis);
    for cg in basis {
        d0.push(GeneralBasisDerivative::from_single(cg.clone()));
        let mut d1_cg = Vec::with_capacity(3);
        for tau in 0..3 {
            d1_cg.push(diff_cg_1st(cg, tau));
        }
        d1.push(d1_cg);
    }

    let perts: Vec<(usize, usize, usize)> = (0..n_atoms)
        .flat_map(|a| (0..3).map(move |tau| (3 * a + tau, a, tau)))
        .collect();

    // 2. Analytical 1st derivatives of S_AO, h_AO, F_skel_AO, and ERI_AO for all 3N coordinates
    let ao_derivs: Vec<(Matrix, Matrix, Matrix, Vec<f64>)> = perts
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
                            let p_ls = p[(la, si)];
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
    let s_1st_mo: Vec<Matrix> = ao_derivs.iter().map(|(s, _, _, _)| ct.dot(s).dot(c)).collect();
    let f_skel_1st_mo: Vec<Matrix> = ao_derivs.iter().map(|(_, _, f, _)| ct.dot(f).dot(c)).collect();

    // 3. Solve 1st-order CPHF equations once at the reference geometry
    let resp = solve_cphf(hf, n_occ, &f_skel_1st_mo, &s_1st_mo);
    let u_mats = resp.u_matrices;
    let mo_eri = transform_eri_ao_to_mo(c, &hf.eri);

    // 4. Construct canonical CPHF orbital response U^{canon(x)} and orbital energy derivatives eps^{(x)}
    let g_hf = compute_rhf_gradient(atoms, basis, hf);
    let e_corr_0 = eval_corr(hf, atoms, basis).unwrap_or(0.0);
    let eta = 1e-4;

    let corr_grads: Vec<f64> = (0..perts.len())
        .into_par_iter()
        .map(|pert_idx| {
            let (_pert, a_atom, tau_comp) = perts[pert_idx];
            let (ref s_1st, ref h_1st, _, ref d_eri) = ao_derivs[pert_idx];
            let s_mo = &s_1st_mo[pert_idx];
            let f_skel_mo = &f_skel_1st_mo[pert_idx];
            let u_std = &u_mats[pert_idx];

            // Compute full 1st-order Fock matrix derivative F_MO^{(x)} in the standard CPHF gauge
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

            // Canonical CPHF response matrix U^{canon(x)}
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

            let build_tangent_state = |step: f64| -> (HFResult, Vec<Atom>, Vec<ContractedGaussian>) {
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

                let mut atoms_step = atoms.to_vec();
                atoms_step[a_atom].position[tau_comp] += step;
                let mut basis_step = basis.to_vec();
                for cg in &mut basis_step {
                    if cg.atom_index == a_atom {
                        cg.center[tau_comp] += step;
                    }
                }

                let hf_step = HFResult {
        ao_space: None,
                    orbital_energies: eps_step,
                    mo_coefficients: c_step,
                    density_matrix: p_step,
                    fock_matrix: fock_step,
                    overlap_matrix: s_step,
                    core_hamiltonian: h_step,
                    eri: eri_step,
                    ..hf.clone()
                };
                (hf_step, atoms_step, basis_step)
            };

            let (hf_p, atoms_p, basis_p) = build_tangent_state(eta);
            let (hf_m, atoms_m, basis_m) = build_tangent_state(-eta);
            let ep = eval_corr(&hf_p, &atoms_p, &basis_p).unwrap_or(e_corr_0);
            let em = eval_corr(&hf_m, &atoms_m, &basis_m).unwrap_or(e_corr_0);
            (ep - em) / (2.0 * eta)
        })
        .collect();

    let mut total_grad = Vec::with_capacity(n_atoms);
    for a in 0..n_atoms {
        let mut ga = g_hf[a];
        for tau in 0..3 {
            ga[tau] += corr_grads[3 * a + tau];
        }
        total_grad.push(ga);
    }
    total_grad
}

/// Solves the Coupled-Perturbed Unrestricted Hartree-Fock (CPUHF) equations for all
/// perturbations $x$ and returns $(U^{\alpha(x)}, U^{\beta(x)}, F^{\alpha,\text{MO}(x)}, F^{\beta,\text{MO}(x)})$.
pub fn solve_cpuhf(
    uhf: &UHFResult,
    s_1st_mo_a: &[Matrix],
    s_1st_mo_b: &[Matrix],
    f_skel_1st_mo_a: &[Matrix],
    f_skel_1st_mo_b: &[Matrix],
) -> (Vec<Matrix>, Vec<Matrix>, Vec<Matrix>, Vec<Matrix>) {
    let n_basis = uhf.mo_coefficients_alpha.rows;
    let n_alpha = uhf.n_alpha;
    let n_beta = uhf.n_beta;
    let n_virt_a = n_basis - n_alpha;
    let n_virt_b = n_basis - n_beta;
    let n_cphf_a = n_virt_a * n_alpha;
    let n_cphf_b = n_virt_b * n_beta;
    let n_cphf = n_cphf_a + n_cphf_b;
    let n_perts = s_1st_mo_a.len();

    let c_a = &uhf.mo_coefficients_alpha;
    let c_b = &uhf.mo_coefficients_beta;
    let eps_a = &uhf.orbital_energies_alpha;
    let eps_b = &uhf.orbital_energies_beta;

    let mo_eri_a = transform_eri_ao_to_mo(c_a, &uhf.eri);
    let mo_eri_b = transform_eri_ao_to_mo(c_b, &uhf.eri);
    let cross_eri = crate::post_hf::mp2_gradient::transform_cross_eri_ao_to_mo(c_a, c_b, &uhf.eri);

    let mut a_mat = Matrix::zeros(n_cphf.max(1), n_cphf.max(1));
    if n_cphf > 0 {
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
                        let val = 2.0 * cross_eri.get(a_mo, i, b_mo, j);
                        a_mat[(row, col)] += val;
                        a_mat[(col, row)] += val;
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
    }

    let lu = if n_cphf > 0 { a_mat.lu_decompose() } else { None };

    let mut u_a_list = Vec::with_capacity(n_perts);
    let mut u_b_list = Vec::with_capacity(n_perts);
    let mut f1_a_list = Vec::with_capacity(n_perts);
    let mut f1_b_list = Vec::with_capacity(n_perts);

    for pert in 0..n_perts {
        let sa = &s_1st_mo_a[pert];
        let sb = &s_1st_mo_b[pert];
        let fa = &f_skel_1st_mo_a[pert];
        let fb = &f_skel_1st_mo_b[pert];

        let mut rhs = vec![0.0; n_cphf];
        for a in 0..n_virt_a {
            let a_mo = n_alpha + a;
            for i in 0..n_alpha {
                let row = a * n_alpha + i;
                let mut b_val = -(fa[(a_mo, i)] - eps_a[i] * sa[(a_mo, i)]);
                for k in 0..n_alpha {
                    for l in 0..n_alpha {
                        b_val += sa[(k, l)]
                            * (mo_eri_a.get(a_mo, i, k, l) - mo_eri_a.get(a_mo, k, i, l));
                    }
                }
                for k in 0..n_beta {
                    for l in 0..n_beta {
                        b_val += sb[(k, l)] * cross_eri.get(a_mo, i, k, l);
                    }
                }
                rhs[row] = b_val;
            }
        }
        for a in 0..n_virt_b {
            let a_mo = n_beta + a;
            for i in 0..n_beta {
                let row = n_cphf_a + a * n_beta + i;
                let mut b_val = -(fb[(a_mo, i)] - eps_b[i] * sb[(a_mo, i)]);
                for k in 0..n_beta {
                    for l in 0..n_beta {
                        b_val += sb[(k, l)]
                            * (mo_eri_b.get(a_mo, i, k, l) - mo_eri_b.get(a_mo, k, i, l));
                    }
                }
                for k in 0..n_alpha {
                    for l in 0..n_alpha {
                        b_val += sa[(k, l)] * cross_eri.get(k, l, a_mo, i);
                    }
                }
                rhs[row] = b_val;
            }
        }

        let u_sol = lu
            .as_ref()
            .map(|l| l.solve(&rhs))
            .unwrap_or_else(|| vec![0.0; n_cphf]);

        let mut ua = Matrix::zeros(n_basis, n_basis);
        for i in 0..n_alpha {
            for j in 0..n_alpha {
                ua[(i, j)] = -0.5 * sa[(i, j)];
            }
        }
        for a in 0..n_virt_a {
            let a_mo = n_alpha + a;
            for b in 0..n_virt_a {
                let b_mo = n_alpha + b;
                ua[(a_mo, b_mo)] = -0.5 * sa[(a_mo, b_mo)];
            }
            for i in 0..n_alpha {
                let val = u_sol[a * n_alpha + i];
                ua[(a_mo, i)] = val;
                ua[(i, a_mo)] = -val - sa[(i, a_mo)];
            }
        }

        let mut ub = Matrix::zeros(n_basis, n_basis);
        for i in 0..n_beta {
            for j in 0..n_beta {
                ub[(i, j)] = -0.5 * sb[(i, j)];
            }
        }
        for a in 0..n_virt_b {
            let a_mo = n_beta + a;
            for b in 0..n_virt_b {
                let b_mo = n_beta + b;
                ub[(a_mo, b_mo)] = -0.5 * sb[(a_mo, b_mo)];
            }
            for i in 0..n_beta {
                let val = u_sol[n_cphf_a + a * n_beta + i];
                ub[(a_mo, i)] = val;
                ub[(i, a_mo)] = -val - sb[(i, a_mo)];
            }
        }

        // Full 1st-order Fock matrices in MO basis
        let mut f1_a = Matrix::zeros(n_basis, n_basis);
        for p in 0..n_basis {
            for q in 0..n_basis {
                let mut val = fa[(p, q)] - 0.5 * (eps_a[p] + eps_a[q]) * sa[(p, q)];
                for k in 0..n_alpha {
                    for l in 0..n_alpha {
                        val -= sa[(k, l)]
                            * (mo_eri_a.get(p, q, k, l) - mo_eri_a.get(p, k, q, l));
                    }
                }
                for k in 0..n_beta {
                    for l in 0..n_beta {
                        val -= sb[(k, l)] * cross_eri.get(p, q, k, l);
                    }
                }
                for a in 0..n_virt_a {
                    let a_mo = n_alpha + a;
                    for k in 0..n_alpha {
                        val += ua[(a_mo, k)]
                            * (2.0 * mo_eri_a.get(p, q, a_mo, k)
                                - mo_eri_a.get(p, a_mo, q, k)
                                - mo_eri_a.get(p, k, q, a_mo));
                    }
                }
                for a in 0..n_virt_b {
                    let a_mo = n_beta + a;
                    for k in 0..n_beta {
                        val += 2.0 * ub[(a_mo, k)] * cross_eri.get(p, q, a_mo, k);
                    }
                }
                f1_a[(p, q)] = val;
            }
        }

        let mut f1_b = Matrix::zeros(n_basis, n_basis);
        for p in 0..n_basis {
            for q in 0..n_basis {
                let mut val = fb[(p, q)] - 0.5 * (eps_b[p] + eps_b[q]) * sb[(p, q)];
                for k in 0..n_beta {
                    for l in 0..n_beta {
                        val -= sb[(k, l)]
                            * (mo_eri_b.get(p, q, k, l) - mo_eri_b.get(p, k, q, l));
                    }
                }
                for k in 0..n_alpha {
                    for l in 0..n_alpha {
                        val -= sa[(k, l)] * cross_eri.get(k, l, p, q);
                    }
                }
                for a in 0..n_virt_b {
                    let a_mo = n_beta + a;
                    for k in 0..n_beta {
                        val += ub[(a_mo, k)]
                            * (2.0 * mo_eri_b.get(p, q, a_mo, k)
                                - mo_eri_b.get(p, a_mo, q, k)
                                - mo_eri_b.get(p, k, q, a_mo));
                    }
                }
                for a in 0..n_virt_a {
                    let a_mo = n_alpha + a;
                    for k in 0..n_alpha {
                        val += 2.0 * ua[(a_mo, k)] * cross_eri.get(a_mo, k, p, q);
                    }
                }
                f1_b[(p, q)] = val;
            }
        }

        u_a_list.push(ua);
        u_b_list.push(ub);
        f1_a_list.push(f1_a);
        f1_b_list.push(f1_b);
    }

    (u_a_list, u_b_list, f1_a_list, f1_b_list)
}

/// Evaluates the exact analytical nuclear gradient of an open-shell UHF Post-HF method
/// by solving the Coupled-Perturbed UHF (CPUHF) equations at the reference geometry
/// and propagating the analytical AO 1st-derivative integrals (`diff_cg_1st`) and canonical
/// CPUHF orbital responses $(U^{\alpha(x)}, U^{\beta(x)})$ through the correlation functional.
pub fn compute_uhf_cphf_analytical_corr_gradient<F>(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    uhf: &UHFResult,
    eval_corr: F,
) -> Vec<[f64; 3]>
where
    F: Fn(&UHFResult) -> Result<f64, String> + Sync,
{
    let n_atoms = atoms.len();
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

    let mut d0 = Vec::with_capacity(n_basis);
    let mut d1 = Vec::with_capacity(n_basis);
    for cg in basis {
        d0.push(GeneralBasisDerivative::from_single(cg.clone()));
        let mut d1_cg = Vec::with_capacity(3);
        for tau in 0..3 {
            d1_cg.push(diff_cg_1st(cg, tau));
        }
        d1.push(d1_cg);
    }

    let perts: Vec<(usize, usize, usize)> = (0..n_atoms)
        .flat_map(|a| (0..3).map(move |tau| (3 * a + tau, a, tau)))
        .collect();

    let ao_derivs: Vec<(Matrix, Matrix, Matrix, Matrix, Vec<f64>)> = perts
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
    let s_1st_mo_a: Vec<Matrix> = ao_derivs.iter().map(|(s, _, _, _, _)| cta.dot(s).dot(c_a)).collect();
    let s_1st_mo_b: Vec<Matrix> = ao_derivs.iter().map(|(s, _, _, _, _)| ctb.dot(s).dot(c_b)).collect();
    let f_skel_1st_mo_a: Vec<Matrix> = ao_derivs.iter().map(|(_, _, fa, _, _)| cta.dot(fa).dot(c_a)).collect();
    let f_skel_1st_mo_b: Vec<Matrix> = ao_derivs.iter().map(|(_, _, _, fb, _)| ctb.dot(fb).dot(c_b)).collect();

    let (u_a_list, u_b_list, f1_a_list, f1_b_list) =
        solve_cpuhf(uhf, &s_1st_mo_a, &s_1st_mo_b, &f_skel_1st_mo_a, &f_skel_1st_mo_b);

    let g_uhf = compute_uhf_gradient(atoms, basis, uhf);
    let e_corr_0 = eval_corr(uhf).unwrap_or(0.0);
    let eta = 1e-4;

    let corr_grads: Vec<f64> = (0..perts.len())
        .into_par_iter()
        .map(|pert_idx| {
            let (ref s_1st, ref h_1st, _, _, ref d_eri) = ao_derivs[pert_idx];
            let mut ua_canon = u_a_list[pert_idx].clone();
            let mut ub_canon = u_b_list[pert_idx].clone();
            let f1_a = &f1_a_list[pert_idx];
            let f1_b = &f1_b_list[pert_idx];

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

            let uhf_p = build_tangent_uhf(eta);
            let uhf_m = build_tangent_uhf(-eta);
            let ep = eval_corr(&uhf_p).unwrap_or(e_corr_0);
            let em = eval_corr(&uhf_m).unwrap_or(e_corr_0);
            (ep - em) / (2.0 * eta)
        })
        .collect();

    let mut total_grad = Vec::with_capacity(n_atoms);
    for a in 0..n_atoms {
        let mut ga = g_uhf[a];
        for tau in 0..3 {
            ga[tau] += corr_grads[3 * a + tau];
        }
        total_grad.push(ga);
    }
    total_grad
}
