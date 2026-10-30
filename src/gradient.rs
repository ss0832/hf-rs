// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Analytical nuclear gradient evaluation for Restricted Hartree-Fock (RHF) and Unrestricted Hartree-Fock (UHF).

use rayon::prelude::*;
use crate::basis::{Atom, ContractedGaussian, Primitive};
use crate::integrals::{
    compute_hermite_coeffs, coulomb_auxiliary_integrals_flat, HermiteBuffer,
    contracted_kinetic, contracted_nuclear_attraction, contracted_overlap,
    BasisPairData, evaluate_pair_eri,
};
use crate::matrix::Matrix;
use crate::scf::{HFResult, UHFResult};

/// A derivative of a ContractedGaussian along a Cartesian axis (x, y, or z).
#[derive(Clone, Debug)]
pub struct BasisDerivative {
    pub terms: Vec<ContractedGaussian>,
}

#[derive(Clone, Copy)]
pub struct ActivePairItem<'a> {
    pub gamma: f64,
    pub q_ls: f64,
    pub pair_cd: &'a BasisPairData,
}

/// Contract ERI derivatives with only one screened (lambda, sigma) list per worker.
/// The list is reused for all derivative terms of (mu, nu), rather than retaining O(n_AO^4) items.
fn contract_eri_derivatives(
    basis: &[ContractedGaussian],
    terms: &[(usize, usize, ContractedGaussian)],
    term_nu_pairs: &[Vec<BasisPairData>],
    q_term: &[Vec<f64>],
    active_basis_pairs: &[(usize, usize, f64, &BasisPairData)],
    gamma: impl Fn(usize, usize, usize, usize) -> f64 + Sync,
) -> Vec<(usize, [f64; 3])> {
    let mut terms_by_mu = vec![Vec::new(); basis.len()];
    for (idx, &(mu, _, _)) in terms.iter().enumerate() { terms_by_mu[mu].push(idx); }
    (0..basis.len()).into_par_iter().map(|mu| {
        let mut result = [0.0; 3];
        let mut items = Vec::with_capacity(active_basis_pairs.len());
        let mut r_buf = [0.0; 8192];
        for nu in 0..basis.len() {
            let max_q = terms_by_mu[mu].iter().map(|&idx| q_term[idx][nu]).fold(0.0_f64, f64::max);
            if max_q < 1e-11 { continue; }
            items.clear();
            for &(lambda, sigma, q_ls, pair_cd) in active_basis_pairs {
                let g = gamma(mu, nu, lambda, sigma);
                if g.abs() * max_q * q_ls >= 1e-11 {
                    items.push(ActivePairItem { gamma: g, q_ls, pair_cd });
                }
            }
            for &idx in &terms_by_mu[mu] {
                let q = q_term[idx][nu];
                if q < 1e-11 { continue; }
                let pair_ab = &term_nu_pairs[idx][nu];
                for item in &items {
                    if (item.gamma * q * item.q_ls).abs() < 1e-11 { continue; }
                    result[terms[idx].1] += item.gamma * evaluate_pair_eri(pair_ab, item.pair_cd, &mut r_buf);
                }
            }
        }
        (basis[mu].atom_index, result)
    }).collect()
}

pub fn differentiate_basis_function(
    cg: &ContractedGaussian,
    tau: usize, // 0 for x, 1 for y, 2 for z
) -> BasisDerivative {
    let mut terms = Vec::new();

    let mut l_plus = cg.l;
    l_plus[tau] += 1;
    let mut prims_plus = Vec::with_capacity(cg.primitives.len());
    for p in &cg.primitives {
        prims_plus.push(Primitive {
            exponent: p.exponent,
            coefficient: p.coefficient * (2.0 * p.exponent),
        });
    }
    terms.push(ContractedGaussian {
        atom_index: cg.atom_index,
        center: cg.center,
        l: l_plus,
        primitives: prims_plus,
    });

    if cg.l[tau] > 0 {
        let mut l_minus = cg.l;
        l_minus[tau] -= 1;
        let l_factor = cg.l[tau] as f64;
        let mut prims_minus = Vec::with_capacity(cg.primitives.len());
        for p in &cg.primitives {
            prims_minus.push(Primitive {
                exponent: p.exponent,
                coefficient: p.coefficient * (-l_factor),
            });
        }
        terms.push(ContractedGaussian {
            atom_index: cg.atom_index,
            center: cg.center,
            l: l_minus,
            primitives: prims_minus,
        });
    }

    BasisDerivative { terms }
}

pub fn deriv_overlap(d_cg: &BasisDerivative, cg: &ContractedGaussian) -> f64 {
    d_cg.terms.iter().map(|t| contracted_overlap(t, cg)).sum()
}

pub fn deriv_kinetic(d_cg: &BasisDerivative, cg: &ContractedGaussian) -> f64 {
    d_cg.terms.iter().map(|t| contracted_kinetic(t, cg)).sum()
}

pub fn deriv_nuclear_attraction(d_cg: &BasisDerivative, cg: &ContractedGaussian, atoms: &[Atom]) -> f64 {
    d_cg.terms.iter().map(|t| contracted_nuclear_attraction(t, cg, atoms)).sum()
}

pub fn deriv_nuclear_attraction_all_atoms(
    d_cg: &BasisDerivative,
    cg: &ContractedGaussian,
    atoms: &[Atom],
) -> Vec<f64> {
    let mut results = vec![0.0; atoms.len()];
    let mut ex = HermiteBuffer::new();
    let mut ey = HermiteBuffer::new();
    let mut ez = HermiteBuffer::new();
    let mut r_buf = [0.0; 8192];

    for t in &d_cg.terms {
        for pa in &t.primitives {
            for pb in &cg.primitives {
                let coeff = pa.coefficient * pb.coefficient;
                if coeff.abs() < 1e-15 { continue; }

                let p = pa.exponent + pb.exponent;
                let inv_p = 1.0 / p;
                let p_coord = [
                    (pa.exponent * t.center[0] + pb.exponent * cg.center[0]) * inv_p,
                    (pa.exponent * t.center[1] + pb.exponent * cg.center[1]) * inv_p,
                    (pa.exponent * t.center[2] + pb.exponent * cg.center[2]) * inv_p,
                ];

                compute_hermite_coeffs(t.l[0], cg.l[0], pa.exponent, pb.exponent, t.center[0], cg.center[0], &mut ex);
                compute_hermite_coeffs(t.l[1], cg.l[1], pa.exponent, pb.exponent, t.center[1], cg.center[1], &mut ey);
                compute_hermite_coeffs(t.l[2], cg.l[2], pa.exponent, pb.exponent, t.center[2], cg.center[2], &mut ez);

                let max_t = t.l[0] + cg.l[0];
                let max_u = t.l[1] + cg.l[1];
                let max_v = t.l[2] + cg.l[2];
                let max_n = max_t + max_u + max_v;

                let stride_v = max_n + 1;
                let stride_u = (max_v + 1) * stride_v;
                let stride_t = (max_u + 1) * stride_u;
                let total_r_size = (max_t + 1) * stride_t;

                let prefactor = 2.0 * std::f64::consts::PI * inv_p;

                for (atom_idx, atom) in atoms.iter().enumerate() {
                    let p_minus_c = [
                        p_coord[0] - atom.position[0],
                        p_coord[1] - atom.position[1],
                        p_coord[2] - atom.position[2],
                    ];

                    coulomb_auxiliary_integrals_flat(max_t, max_u, max_v, p, p_minus_c, &mut r_buf[..total_r_size]);

                    let mut sum = 0.0;
                    for it in 0..=max_t {
                        let e_t = ex.get(t.l[0], cg.l[0], it);
                        if e_t == 0.0 { continue; }
                        for iu in 0..=max_u {
                            let e_u = ey.get(t.l[1], cg.l[1], iu);
                            if e_u == 0.0 { continue; }
                            for iv in 0..=max_v {
                                let e_v = ez.get(t.l[2], cg.l[2], iv);
                                if e_v == 0.0 { continue; }
                                let idx = it * stride_t + iu * stride_u + iv * stride_v;
                                sum += e_t * e_u * e_v * r_buf[idx];
                            }
                        }
                    }
                    results[atom_idx] += coeff * (-atom.charge * prefactor * sum);
                }
            }
        }
    }
    results
}

pub fn deriv_nuclear_attraction_single_atom(
    d_cg: &BasisDerivative,
    cg: &ContractedGaussian,
    atom: &Atom,
) -> f64 {
    let mut total = 0.0;
    for t in &d_cg.terms {
        for pa in &t.primitives {
            for pb in &cg.primitives {
                total += pa.coefficient * pb.coefficient * crate::integrals::primitive_nuclear_attraction(
                    pa.exponent, t.l, t.center,
                    pb.exponent, cg.l, cg.center,
                    atom.position, atom.charge,
                );
            }
        }
    }
    total
}

pub fn compute_nuclear_repulsion_gradient(atoms: &[Atom]) -> Vec<[f64; 3]> {
    let n_atoms = atoms.len();
    let mut grad = vec![[0.0; 3]; n_atoms];

    for a in 0..n_atoms {
        for b in 0..n_atoms {
            if a == b {
                continue;
            }
            let dx = atoms[a].position[0] - atoms[b].position[0];
            let dy = atoms[a].position[1] - atoms[b].position[1];
            let dz = atoms[a].position[2] - atoms[b].position[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            let dist3 = dist * dist * dist;
            let factor = atoms[a].charge * atoms[b].charge / dist3;

            grad[a][0] -= factor * dx;
            grad[a][1] -= factor * dy;
            grad[a][2] -= factor * dz;
        }
    }

    grad
}

pub fn compute_energy_weighted_density(hf: &HFResult, n_occ: usize) -> Matrix {
    let n = hf.mo_coefficients.rows;
    let mut w = Matrix::zeros(n, n);
    for mu in 0..n {
        for nu in 0..n {
            let mut val = 0.0;
            for i in 0..n_occ {
                val += 2.0 * hf.orbital_energies[i] * hf.mo_coefficients[(mu, i)] * hf.mo_coefficients[(nu, i)];
            }
            w[(mu, nu)] = val;
        }
    }
    w
}

/// Computes the complete analytical gradient of the RHF energy.
pub fn try_compute_rhf_gradient(
    atoms: &[Atom], basis: &[ContractedGaussian], hf: &HFResult,
) -> Result<Vec<[f64;3]>,crate::error::HfError> {
    if !hf.converged { return Err(crate::error::HfError::Scf("reference did not converge".into())); }
    if let Some(space)=&hf.ao_space { space.ensure_differentiable()?; }
    else if hf.mo_coefficients.cols < basis.len() {
        return Err(crate::error::HfError::LinearDependence { n_ao:basis.len(), n_mo:hf.mo_coefficients.cols });
    }
    Ok(compute_rhf_gradient(atoms,basis,hf))
}

pub fn compute_rhf_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
) -> Vec<[f64; 3]> {
    let total_nuclear_charge: f64 = atoms.iter().map(|a| a.charge).sum();
    let total_electrons = (total_nuclear_charge.round() as i32) - hf.charge;
    let n_occ = (total_electrons as usize) / 2;

    let p = &hf.density_matrix;
    let w = compute_energy_weighted_density(hf, n_occ);
    compute_gradient_from_density(atoms, basis, p, &w)
}

/// Evaluates the analytical Pulay gradient for arbitrary one-particle density matrix P and energy-weighted density W.
pub fn compute_gradient_from_density(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    p: &Matrix,
    w: &Matrix,
) -> Vec<[f64; 3]> {
    compute_gradient_from_density_with_eri_contraction(atoms, basis, p, w,
        |mu, nu, lambda, sigma| {
            if lambda == sigma {
                2.0 * p[(mu, nu)] * p[(lambda, lambda)] - p[(mu, lambda)] * p[(nu, lambda)]
            } else {
                4.0 * p[(mu, nu)] * p[(lambda, sigma)] - p[(mu, lambda)] * p[(nu, sigma)] - p[(mu, sigma)] * p[(nu, lambda)]
            }
        })
}

/// Shared screened derivative contraction, including nuclear/ECP and Pulay
/// terms. `gamma` supplies the symmetry-weighted first-center ERI density.
/// Its pair lists are retained per worker rather than per AO quartet.
pub(crate) fn compute_gradient_from_density_with_eri_contraction(
    atoms: &[Atom], basis: &[ContractedGaussian], p: &Matrix, w: &Matrix,
    gamma: impl Fn(usize, usize, usize, usize) -> f64 + Sync,
) -> Vec<[f64; 3]> {
    let n_atoms = atoms.len();
    let n_basis = basis.len();
    let nuc_grad = compute_nuclear_repulsion_gradient(atoms);

    let mut basis_derivs = Vec::with_capacity(n_basis);
    for cg in basis {
        let dx = differentiate_basis_function(cg, 0);
        let dy = differentiate_basis_function(cg, 1);
        let dz = differentiate_basis_function(cg, 2);
        basis_derivs.push([dx, dy, dz]);
    }

    let t0 = std::time::Instant::now();
    // Flatten all derivative terms for fine-grained parallel dispatch
    let mut term_entries = Vec::new();
    for mu in 0..n_basis {
        for tau in 0..3 {
            for term in &basis_derivs[mu][tau].terms {
                term_entries.push((mu, tau, term.clone()));
            }
        }
    }

    let basis_pairs: Vec<Vec<BasisPairData>> = (0..n_basis)
        .into_par_iter()
        .map(|lambda| {
            (0..=lambda)
                .map(|sigma| BasisPairData::new(&basis[lambda], &basis[sigma]))
                .collect()
        })
        .collect();

    // Precompute Cauchy-Schwarz bounds Q_{lambda, sigma} = sqrt((lambda sigma | lambda sigma))
    let schwarz_q: Vec<Vec<f64>> = (0..n_basis)
        .into_par_iter()
        .map(|lambda| {
            let mut row = vec![0.0; n_basis];
            let mut r_buf = [0.0; 8192];
            for sigma in 0..=lambda {
                let pair = &basis_pairs[lambda][sigma];
                if pair.primitives.is_empty() {
                    continue;
                }
                let q = evaluate_pair_eri(pair, pair, &mut r_buf)
                    .max(0.0)
                    .sqrt();
                row[sigma] = q;
            }
            row
        })
        .collect();

    // Precompute max Schwarz bound across all basis pairs
    let mut max_q_ls = 0.0f64;
    for row in &schwarz_q {
        for &val in row {
            if val > max_q_ls { max_q_ls = val; }
        }
    }

    // Precompute max absolute density elements per basis function
    let mut p_max_row = vec![0.0f64; n_basis];
    let mut p_max_global = 0.0f64;
    for i in 0..n_basis {
        for j in 0..n_basis {
            let val = p[(i, j)].abs();
            if val > p_max_row[i] { p_max_row[i] = val; }
            if val > p_max_global { p_max_global = val; }
        }
    }

    let mut active_basis_pairs = Vec::new();
    for lambda in 0..n_basis {
        for sigma in 0..=lambda {
            let data = &basis_pairs[lambda][sigma];
            let q_ls = schwarz_q[lambda][sigma];
            if q_ls >= 1e-10 && !data.primitives.is_empty() {
                active_basis_pairs.push((lambda, sigma, q_ls, data));
            }
        }
    }

    // Precompute Q(term, nu) and BasisPairData in a single pass with distance screening
    let (term_nu_pairs, q_term): (Vec<Vec<BasisPairData>>, Vec<Vec<f64>>) = term_entries
        .par_iter()
        .map(|(_, _, term)| {
            let mut pairs = Vec::with_capacity(n_basis);
            let mut qs = vec![0.0; n_basis];
            let mut r_buf = [0.0; 8192];
            for nu in 0..n_basis {
                let pair = BasisPairData::new(term, &basis[nu]);
                if pair.primitives.is_empty() {
                    pairs.push(pair);
                    continue;
                }
                let q = evaluate_pair_eri(&pair, &pair, &mut r_buf).max(0.0).sqrt();
                qs[nu] = q;
                pairs.push(pair);
            }
            (pairs, qs)
        })
        .unzip();

    eprintln!("  [Grad] Precalc Schwarz & Pairs ({:?} terms, {:?} active pairs): {:?}", term_entries.len(), active_basis_pairs.len(), t0.elapsed());

    let mut grad = nuc_grad;
    // Effective core potential contribution (no-op without ECP atoms)
    if crate::ecp::has_ecp(atoms) {
        let ecp_grad = crate::ecp::ecp_gradient(atoms, basis, p);
        for (g, e) in grad.iter_mut().zip(&ecp_grad) {
            for k in 0..3 {
                g[k] += e[k];
            }
        }
    }
    let t1 = std::time::Instant::now();

    // 1-electron basis derivative terms (overlap, kinetic, full nuclear attraction)
    // Symmetrized: sum_{nu} W_{mu,nu} (dmu|nu) + W_{nu,mu} (mu|dnu) = 2 sum_{nu} W_{mu,nu} (dmu|nu)
    let one_electron_basis: Vec<(usize, usize, f64)> = (0..n_basis * 3)
        .into_par_iter()
        .map(|task_id| {
            let mu = task_id / 3;
            let tau = task_id % 3;
            let atom_idx = basis[mu].atom_index;
            let d_mu = &basis_derivs[mu][tau];

            let mut ds = 0.0;
            let mut dt = 0.0;
            let mut dv = 0.0;
            for nu in 0..n_basis {
                let p_munu = p[(mu, nu)];
                let w_munu = w[(mu, nu)];
                if p_munu.abs() < 1e-15 && w_munu.abs() < 1e-15 {
                    continue;
                }
                if w_munu.abs() >= 1e-15 {
                    ds += w_munu * deriv_overlap(d_mu, &basis[nu]);
                }
                if p_munu.abs() >= 1e-15 {
                    dt += p_munu * deriv_kinetic(d_mu, &basis[nu]);
                    dv += p_munu * deriv_nuclear_attraction(d_mu, &basis[nu], atoms);
                }
            }
            let val = 2.0 * (dt + dv - ds);
            (atom_idx, tau, val)
        })
        .collect();

    for (atom_idx, tau, val) in one_electron_basis {
        grad[atom_idx][tau] += val;
    }
    eprintln!("  [Grad] 1-electron basis: {:?}", t1.elapsed());

    let t2 = std::time::Instant::now();
    // Operator derivative of nuclear attraction: <mu | dV_A / dR_A | nu>
    // Vectorized across all nuclei in a single pass using deriv_nuclear_attraction_all_atoms
    let nuc_pot_threads: Vec<Vec<[f64; 3]>> = (0..n_basis)
        .into_par_iter()
        .map(|mu| {
            let mut thread_grad = vec![[0.0; 3]; n_atoms];
            for nu in 0..=mu {
                let p_munu = p[(mu, nu)];
                if p_munu.abs() < 1e-12 {
                    continue;
                }

                for tau in 0..3 {
                    let v_mu = deriv_nuclear_attraction_all_atoms(&basis_derivs[mu][tau], &basis[nu], atoms);
                    if mu == nu {
                        for atom_idx in 0..n_atoms {
                            thread_grad[atom_idx][tau] -= 2.0 * p_munu * v_mu[atom_idx];
                        }
                    } else {
                        let v_nu = deriv_nuclear_attraction_all_atoms(&basis_derivs[nu][tau], &basis[mu], atoms);
                        for atom_idx in 0..n_atoms {
                            thread_grad[atom_idx][tau] -= 2.0 * p_munu * (v_mu[atom_idx] + v_nu[atom_idx]);
                        }
                    }
                }
            }
            thread_grad
        })
        .collect();

    for thread_grad in nuc_pot_threads {
        for atom_idx in 0..n_atoms {
            for tau in 0..3 {
                grad[atom_idx][tau] += thread_grad[atom_idx][tau];
            }
        }
    }
    eprintln!("  [Grad] Nuclear attraction operator: {:?}", t2.elapsed());

    let t3 = std::time::Instant::now();
    let eri_contributions = contract_eri_derivatives(basis, &term_entries, &term_nu_pairs, &q_term, &active_basis_pairs, gamma);
    for (atom_idx, values) in eri_contributions {
        for tau in 0..3 { grad[atom_idx][tau] += values[tau]; }
    }
    eprintln!("  [Grad] 2-electron ERI: {:?}", t3.elapsed());

    grad
}

/// Computes the complete analytical nuclear gradient for Unrestricted Hartree-Fock (UHF).
/// Supports open-shell systems, formal charges, radicals, and broken-symmetry states.
pub fn try_compute_uhf_gradient(
    atoms: &[Atom], basis: &[ContractedGaussian], hf: &UHFResult,
) -> Result<Vec<[f64;3]>,crate::error::HfError> {
    if !hf.converged { return Err(crate::error::HfError::Scf("reference did not converge".into())); }
    if let Some(space)=&hf.ao_space { space.ensure_differentiable()?; }
    else if hf.mo_coefficients_alpha.cols < basis.len() {
        return Err(crate::error::HfError::LinearDependence { n_ao:basis.len(), n_mo:hf.mo_coefficients_alpha.cols });
    }
    Ok(compute_uhf_gradient(atoms,basis,hf))
}

pub fn compute_uhf_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    uhf: &UHFResult,
) -> Vec<[f64; 3]> {
    let n_atoms = atoms.len();
    let n_basis = basis.len();

    let p_a = &uhf.density_matrix_alpha;
    let p_b = &uhf.density_matrix_beta;
    let p_tot = p_a.add(p_b);

    // Energy-weighted density matrices: W_alpha and W_beta
    let mut w_tot = Matrix::zeros(n_basis, n_basis);
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            let mut val = 0.0;
            for i in 0..uhf.n_alpha {
                val += uhf.orbital_energies_alpha[i] * uhf.mo_coefficients_alpha[(mu, i)] * uhf.mo_coefficients_alpha[(nu, i)];
            }
            for i in 0..uhf.n_beta {
                val += uhf.orbital_energies_beta[i] * uhf.mo_coefficients_beta[(mu, i)] * uhf.mo_coefficients_beta[(nu, i)];
            }
            w_tot[(mu, nu)] = val;
        }
    }

    let nuc_grad = compute_nuclear_repulsion_gradient(atoms);

    let mut basis_derivs = Vec::with_capacity(n_basis);
    for cg in basis {
        let dx = differentiate_basis_function(cg, 0);
        let dy = differentiate_basis_function(cg, 1);
        let dz = differentiate_basis_function(cg, 2);
        basis_derivs.push([dx, dy, dz]);
    }

    // Flatten all derivative terms for fine-grained parallel dispatch
    let mut term_entries = Vec::new();
    for mu in 0..n_basis {
        for tau in 0..3 {
            for term in &basis_derivs[mu][tau].terms {
                term_entries.push((mu, tau, term.clone()));
            }
        }
    }

    let basis_pairs: Vec<Vec<BasisPairData>> = (0..n_basis)
        .into_par_iter()
        .map(|lambda| {
            (0..=lambda)
                .map(|sigma| BasisPairData::new(&basis[lambda], &basis[sigma]))
                .collect()
        })
        .collect();

    // Precompute Cauchy-Schwarz bounds Q_{lambda, sigma} = sqrt((lambda sigma | lambda sigma))
    let schwarz_q: Vec<Vec<f64>> = (0..n_basis)
        .into_par_iter()
        .map(|lambda| {
            let mut row = vec![0.0; n_basis];
            let mut r_buf = [0.0; 8192];
            for sigma in 0..=lambda {
                let pair = &basis_pairs[lambda][sigma];
                if pair.primitives.is_empty() {
                    continue;
                }
                let q = evaluate_pair_eri(pair, pair, &mut r_buf)
                    .max(0.0)
                    .sqrt();
                row[sigma] = q;
            }
            row
        })
        .collect();

    let mut max_q_ls = 0.0f64;
    for row in &schwarz_q {
        for &val in row {
            if val > max_q_ls { max_q_ls = val; }
        }
    }

    let mut p_tot_max_row = vec![0.0f64; n_basis];
    let mut p_a_max_row = vec![0.0f64; n_basis];
    let mut p_b_max_row = vec![0.0f64; n_basis];
    let mut p_tot_max_global = 0.0f64;
    for i in 0..n_basis {
        for j in 0..n_basis {
            let tot_val = p_tot[(i, j)].abs();
            let a_val = p_a[(i, j)].abs();
            let b_val = p_b[(i, j)].abs();
            if tot_val > p_tot_max_row[i] { p_tot_max_row[i] = tot_val; }
            if a_val > p_a_max_row[i] { p_a_max_row[i] = a_val; }
            if b_val > p_b_max_row[i] { p_b_max_row[i] = b_val; }
            if tot_val > p_tot_max_global { p_tot_max_global = tot_val; }
        }
    }

    let mut active_basis_pairs = Vec::new();
    for lambda in 0..n_basis {
        for sigma in 0..=lambda {
            let data = &basis_pairs[lambda][sigma];
            let q_ls = schwarz_q[lambda][sigma];
            if q_ls >= 1e-10 && !data.primitives.is_empty() {
                active_basis_pairs.push((lambda, sigma, q_ls, data));
            }
        }
    }

    // Precompute Q(term, nu) and BasisPairData in a single pass with distance screening
    let (term_nu_pairs, q_term): (Vec<Vec<BasisPairData>>, Vec<Vec<f64>>) = term_entries
        .par_iter()
        .map(|(_, _, term)| {
            let mut pairs = Vec::with_capacity(n_basis);
            let mut qs = vec![0.0; n_basis];
            let mut r_buf = [0.0; 8192];
            for nu in 0..n_basis {
                let pair = BasisPairData::new(term, &basis[nu]);
                if pair.primitives.is_empty() {
                    pairs.push(pair);
                    continue;
                }
                let q = evaluate_pair_eri(&pair, &pair, &mut r_buf).max(0.0).sqrt();
                qs[nu] = q;
                pairs.push(pair);
            }
            (pairs, qs)
        })
        .unzip();


    let mut grad = nuc_grad;
    // Effective core potential contribution (no-op without ECP atoms)
    if crate::ecp::has_ecp(atoms) {
        let ecp_grad = crate::ecp::ecp_gradient(atoms, basis, &p_tot);
        for (g, e) in grad.iter_mut().zip(&ecp_grad) {
            for k in 0..3 {
                g[k] += e[k];
            }
        }
    }

    // 1-electron basis derivative terms
    let one_electron_basis: Vec<(usize, usize, f64)> = (0..n_basis * 3)
        .into_par_iter()
        .map(|task_id| {
            let mu = task_id / 3;
            let tau = task_id % 3;
            let atom_idx = basis[mu].atom_index;
            let d_mu = &basis_derivs[mu][tau];

            let mut ds = 0.0;
            let mut dt = 0.0;
            let mut dv = 0.0;
            for nu in 0..n_basis {
                let p_val = p_tot[(mu, nu)];
                let w_val = w_tot[(mu, nu)];
                if p_val.abs() < 1e-15 && w_val.abs() < 1e-15 {
                    continue;
                }
                if w_val.abs() >= 1e-15 {
                    ds += w_val * deriv_overlap(d_mu, &basis[nu]);
                }
                if p_val.abs() >= 1e-15 {
                    dt += p_val * deriv_kinetic(d_mu, &basis[nu]);
                    dv += p_val * deriv_nuclear_attraction(d_mu, &basis[nu], atoms);
                }
            }
            let val = 2.0 * (dt + dv - ds);
            (atom_idx, tau, val)
        })
        .collect();

    for (atom_idx, tau, val) in one_electron_basis {
        grad[atom_idx][tau] += val;
    }

    // Operator derivative of nuclear attraction: <mu | dV_A / dR_A | nu>
    // Vectorized across all nuclei in a single pass using deriv_nuclear_attraction_all_atoms
    let nuc_pot_threads: Vec<Vec<[f64; 3]>> = (0..n_basis)
        .into_par_iter()
        .map(|mu| {
            let mut thread_grad = vec![[0.0; 3]; n_atoms];
            for nu in 0..=mu {
                let p_munu = p_tot[(mu, nu)];
                if p_munu.abs() < 1e-12 {
                    continue;
                }

                for tau in 0..3 {
                    let v_mu = deriv_nuclear_attraction_all_atoms(&basis_derivs[mu][tau], &basis[nu], atoms);
                    if mu == nu {
                        for atom_idx in 0..n_atoms {
                            thread_grad[atom_idx][tau] -= 2.0 * p_munu * v_mu[atom_idx];
                        }
                    } else {
                        let v_nu = deriv_nuclear_attraction_all_atoms(&basis_derivs[nu][tau], &basis[mu], atoms);
                        for atom_idx in 0..n_atoms {
                            thread_grad[atom_idx][tau] -= 2.0 * p_munu * (v_mu[atom_idx] + v_nu[atom_idx]);
                        }
                    }
                }
            }
            thread_grad
        })
        .collect();

    for thread_grad in nuc_pot_threads {
        for atom_idx in 0..n_atoms {
            for tau in 0..3 {
                grad[atom_idx][tau] += thread_grad[atom_idx][tau];
            }
        }
    }

    let eri_contributions = contract_eri_derivatives(basis, &term_entries, &term_nu_pairs, &q_term, &active_basis_pairs,
        |mu, nu, lambda, sigma| {
            let coulomb = 2.0 * p_tot[(mu, nu)] * p_tot[(lambda, sigma)];
            let exchange = 2.0 * (p_a[(mu, lambda)] * p_a[(nu, sigma)] + p_b[(mu, lambda)] * p_b[(nu, sigma)]);
            let mut gamma = coulomb - exchange;
            if lambda != sigma {
                gamma += 2.0 * p_tot[(mu, nu)] * p_tot[(sigma, lambda)]
                    - 2.0 * (p_a[(mu, sigma)] * p_a[(nu, lambda)] + p_b[(mu, sigma)] * p_b[(nu, lambda)]);
            }
            gamma
        });
    for (atom_idx, values) in eri_contributions {
        for tau in 0..3 { grad[atom_idx][tau] += values[tau]; }
    }

    grad
}
