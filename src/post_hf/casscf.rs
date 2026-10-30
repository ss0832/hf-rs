// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Complete Active Space Self-Consistent Field (CASSCF).
//!
//! Provides simultaneous optimization of CI expansion coefficients in the active space
//! and orbital rotation parameters (Super-CI / Newton-Raphson) to minimize the energy.

use rayon::prelude::*;
use crate::basis::{Atom, ContractedGaussian};
use crate::gradient::{
    compute_nuclear_repulsion_gradient, deriv_kinetic, deriv_nuclear_attraction,
    deriv_nuclear_attraction_single_atom, deriv_overlap, differentiate_basis_function,
};
use crate::integrals::contracted_eri;
use crate::matrix::Matrix;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::scf::HFResult;

/// Configuration options for CASSCF calculations.
#[derive(Clone, Debug)]
pub struct CASSCFOptions {
    pub n_active_electrons: usize,
    pub n_active_orbitals: usize,
    pub max_macro_iterations: usize,
    pub grad_tol: f64,
    pub energy_tol: f64,
    pub step_damping: f64,
}

impl CASSCFOptions {
    pub fn new(n_active_electrons: usize, n_active_orbitals: usize) -> Self {
        Self {
            n_active_electrons,
            n_active_orbitals,
            max_macro_iterations: 80,
            grad_tol: 1e-7,
            energy_tol: 1e-12,
            step_damping: 0.5,
        }
    }
}

/// Results of a CASSCF calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct CASSCFResult {
    pub reference_energy: f64,
    pub total_energy: f64,
    pub correlation_energy: f64,
    pub iterations: usize,
    pub converged: bool,
    pub orbital_gradient_norm: f64,
    pub natural_orbital_occupations: Vec<f64>,
    pub one_rdm_mo: Matrix,
    pub mo_coefficients: Matrix,
    pub generalized_fock: Matrix,
    pub two_rdm_mo: Vec<f64>,
    pub n_core: usize,
    pub n_active_orbitals: usize,
}

impl PostHFResult for CASSCFResult {
    fn converged(&self)->bool {self.converged}
    fn method_name(&self) -> &'static str {
        "CASSCF"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }

    fn total_energy(&self) -> f64 {
        self.total_energy
    }

    fn one_rdm(&self) -> Option<&Matrix> {
        Some(&self.one_rdm_mo)
    }
}

/// CASSCF calculation engine.
pub struct CASSCF<'a> {
    pub hf: &'a HFResult,
    pub options: CASSCFOptions,
}

impl<'a> CASSCF<'a> {
    pub fn new(hf: &'a HFResult, options: CASSCFOptions) -> Self {
        Self { hf, options }
    }
}

impl<'a> PostHFMethod for CASSCF<'a> {
    type Output=CASSCFResult;
    fn compute(&self)->Result<Self::Output,String> {
        if !self.options.step_damping.is_finite() || self.options.step_damping<=0.0 || self.options.step_damping>1.0 {
            return Err("CASSCF step_damping must lie in (0,1]".into());
        }
        let mut options=crate::options::PostHFOptions{active:Some(crate::options::ActiveSpace::new(
            self.options.n_active_electrons.div_ceil(2),self.options.n_active_electrons/2,self.options.n_active_orbitals)),
            energy_tol:self.options.energy_tol,..Default::default()};
        options.cas.max_orbital_iterations=self.options.max_macro_iterations;
        options.cas.orbital_gradient_tol=self.options.grad_tol;
        options.cas.max_rotation=0.2*self.options.step_damping;
        let result=super::casscf_engine::solve_cas(crate::context::Reference::Rhf(self.hf),&options,None,true).map_err(|e|e.to_string())?;
        let one=result.output.one_rdm.as_ref().ok_or_else(||"CASSCF one-RDM is missing".to_string())?.clone();
        let (occupations,_)=one.eigh();
        Ok(CASSCFResult{reference_energy:self.hf.total_energy,total_energy:result.output.e_total,
            correlation_energy:result.output.e_corr,iterations:result.iterations,converged:true,
            orbital_gradient_norm:result.orbital_gradient_max,natural_orbital_occupations:occupations,
            one_rdm_mo:one,mo_coefficients:result.mo_coefficients,
            generalized_fock:result.generalized_fock.transpose(),two_rdm_mo:result.state_two_rdms[0].clone(),
            n_core:result.active_space.inactive.len(),n_active_orbitals:result.active_space.active.len()})
    }
}

/// Computes the exact analytical nuclear gradient for CASSCF wavefunctions.
///
/// Implements the exact variational CASSCF gradient using the generalized Hellmann-Feynman
/// theorem, where orbital and CI responses vanish identically:
///
///   g_{A, \tau} = g_{A, \tau}^{nuc} + sum_{\mu\nu} D_{\mu\nu} h_{\mu\nu}^{(A,\tau)}
///                 - sum_{\mu\nu} W_{\mu\nu} S_{\mu\nu}^{(A,\tau)}
///                 + sum_{\mu \in A, \nu\lambda\sigma} Gamma_{\mu\nu\lambda\sigma}^{AO, symm} (\mu' \nu | \lambda \sigma)
pub fn compute_casscf_analytical_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    _hf: &HFResult,
    casscf: &CASSCFResult,
) -> Vec<[f64; 3]> {
    let n_atoms = atoms.len();
    let n_basis = basis.len();
    let n_core = casscf.n_core;
    let n_act = casscf.n_active_orbitals;
    let n_occ = n_core + n_act;

    // 1. Construct total 2-RDM in MO basis
    let mut gamma_mo = vec![vec![vec![vec![0.0; n_basis]; n_basis]; n_basis]; n_basis];

    // Core-core mean-field 2-RDM
    for c in 0..n_core {
        for d in 0..n_core {
            gamma_mo[c][c][d][d] += 4.0;
            gamma_mo[c][d][d][c] -= 2.0;
        }
    }

    // Core-active mean-field 2-RDM
    for c in 0..n_core {
        for t in 0..n_act {
            let t_mo = n_core + t;
            for u in 0..n_act {
                let u_mo = n_core + u;
                let d_tu = casscf.one_rdm_mo[(t_mo, u_mo)];
                if d_tu.abs() < 1e-15 {
                    continue;
                }
                gamma_mo[t_mo][u_mo][c][c] += 2.0 * d_tu;
                gamma_mo[c][c][t_mo][u_mo] += 2.0 * d_tu;
                gamma_mo[t_mo][c][c][u_mo] -= d_tu;
                gamma_mo[c][u_mo][t_mo][c] -= d_tu;
            }
        }
    }

    // Active space correlated 2-RDM
    let idx4 = |p: usize, q: usize, r: usize, s: usize| -> usize {
        ((p * n_act + q) * n_act + r) * n_act + s
    };
    for t in 0..n_act {
        let t_mo = n_core + t;
        for u in 0..n_act {
            let u_mo = n_core + u;
            for v in 0..n_act {
                let v_mo = n_core + v;
                for w in 0..n_act {
                    let w_mo = n_core + w;
                    let val = casscf.two_rdm_mo[idx4(t, u, v, w)];
                    if val.abs() > 1e-15 {
                        gamma_mo[t_mo][u_mo][v_mo][w_mo] += val;
                    }
                }
            }
        }
    }

    // 2. Transform 2-RDM to AO basis
    let c = &casscf.mo_coefficients;
    let mut gamma_ao = vec![vec![vec![vec![0.0; n_basis]; n_basis]; n_basis]; n_basis];
    for p in 0..n_occ {
        for q in 0..n_occ {
            for r in 0..n_occ {
                for s in 0..n_occ {
                    let g_val = gamma_mo[p][q][r][s];
                    if g_val.abs() < 1e-15 {
                        continue;
                    }
                    for mu in 0..n_basis {
                        let c_mu = c[(mu, p)];
                        if c_mu.abs() < 1e-15 {
                            continue;
                        }
                        for nu in 0..n_basis {
                            let c_nu = c[(nu, q)];
                            let c_munu = c_mu * c_nu;
                            if c_munu.abs() < 1e-15 {
                                continue;
                            }
                            for la in 0..n_basis {
                                let c_la = c[(la, r)];
                                let c_munula = c_munu * c_la;
                                if c_munula.abs() < 1e-15 {
                                    continue;
                                }
                                for si in 0..n_basis {
                                    let c_si = c[(si, s)];
                                    gamma_ao[mu][nu][la][si] += g_val * c_munula * c_si;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 3. 8-fold symmetrize AO 2-RDM
    let mut gamma_symm = vec![vec![vec![vec![0.0; n_basis]; n_basis]; n_basis]; n_basis];
    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for la in 0..n_basis {
                for si in 0..n_basis {
                    let sum = gamma_ao[mu][nu][la][si]
                        + gamma_ao[nu][mu][la][si]
                        + gamma_ao[mu][nu][si][la]
                        + gamma_ao[nu][mu][si][la]
                        + gamma_ao[la][si][mu][nu]
                        + gamma_ao[si][la][mu][nu]
                        + gamma_ao[la][si][nu][mu]
                        + gamma_ao[si][la][nu][mu];
                    gamma_symm[mu][nu][la][si] = sum * 0.125;
                }
            }
        }
    }

    // 4. Construct 1-RDM and energy-weighted density matrix in AO basis
    let d_ao = c.dot(&casscf.one_rdm_mo).dot(&c.transpose());
    let mut w_mo_symm = Matrix::zeros(n_basis, n_basis);
    for p in 0..n_basis {
        for q in 0..n_basis {
            w_mo_symm[(p, q)] = 0.5 * (casscf.generalized_fock[(p, q)] + casscf.generalized_fock[(q, p)]);
        }
    }
    let w_ao = c.dot(&w_mo_symm).dot(&c.transpose());

    // 5. Precompute basis function spatial derivatives
    let mut basis_derivs = Vec::with_capacity(n_basis);
    for cg in basis {
        let dx = differentiate_basis_function(cg, 0);
        let dy = differentiate_basis_function(cg, 1);
        let dz = differentiate_basis_function(cg, 2);
        basis_derivs.push([dx, dy, dz]);
    }

    let g_nuc = compute_nuclear_repulsion_gradient(atoms);

    // 6. Assemble analytical nuclear gradient
    (0..n_atoms)
        .into_par_iter()
        .map(|atom_idx| {
            let mut grad_a = g_nuc[atom_idx];

            for tau in 0..3 {
                let mut d_s_val = 0.0;
                let mut d_t_val = 0.0;
                let mut d_v_val = 0.0;

                for mu in 0..n_basis {
                    let mu_on_a = basis[mu].atom_index == atom_idx;
                    for nu in 0..n_basis {
                        let nu_on_a = basis[nu].atom_index == atom_idx;
                        let d_munu = d_ao[(mu, nu)];
                        let w_munu = w_ao[(mu, nu)];

                        if d_munu.abs() < 1e-15 && w_munu.abs() < 1e-15 {
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
                        d_t_val += d_munu * dt;

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

                        d_v_val += d_munu * dv;
                    }
                }

                grad_a[tau] += (d_t_val + d_v_val) - d_s_val;

                // 2-electron skeleton derivative contraction
                let mut d_eri_val = 0.0;
                for mu in 0..n_basis {
                    if basis[mu].atom_index != atom_idx {
                        continue;
                    }
                    for term in &basis_derivs[mu][tau].terms {
                        for nu in 0..n_basis {
                            for la in 0..n_basis {
                                for si in 0..n_basis {
                                    let gamma_val = 2.0 * gamma_symm[mu][nu][la][si];
                                    if gamma_val.abs() < 1e-15 {
                                        continue;
                                    }
                                    let eri_val = contracted_eri(term, &basis[nu], &basis[la], &basis[si]);
                                    d_eri_val += gamma_val * eri_val;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{parse_gaussian94, STO_3G_GBS, Atom, build_basis};
    use crate::scf::run_scf_general;
    use crate::diis::ConvAlgo;

    #[test]
    fn test_h2_casscf_2_2() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0,  0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, ConvAlgo::Adiissoscf, None)
            .expect("HF failed");

        let opts = CASSCFOptions::new(2, 2);
        let casscf = CASSCF::new(&hf, opts).compute().expect("CASSCF failed");

        assert!(casscf.total_energy <= casscf.reference_energy);
        assert!(casscf.correlation_energy <= 0.0);
    }
}
