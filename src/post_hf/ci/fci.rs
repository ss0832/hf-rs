// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Full Configuration Interaction (FCI) within the active or full orbital space.
//!
//! Solves the exact Schrödinger equation in the given orbital space using
//! the Knowles-Handy determinant string formulation and the Davidson-Liu subspace eigensolver.

use rayon::prelude::*;
use crate::basis::{Atom, ContractedGaussian};
use crate::gradient::{
    compute_nuclear_repulsion_gradient, deriv_kinetic, deriv_nuclear_attraction,
    deriv_nuclear_attraction_single_atom, deriv_overlap, differentiate_basis_function,
};
use crate::integrals::contracted_eri;
use crate::matrix::Matrix;
use crate::mo_integrals::{transform_1e_ao_to_mo, transform_eri_ao_to_mo};
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::scf::HFResult;

/// Results of a Full Configuration Interaction calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct FCIResult {
    pub reference_energy: f64,
    pub total_energy: f64,
    pub correlation_energy: f64,
    pub n_determinants: usize,
    pub ground_state_vector: Vec<f64>,
    pub one_rdm_mo: Matrix,
    pub mo_coefficients: Matrix,
    pub two_rdm_mo: Vec<f64>,
}

impl PostHFResult for FCIResult {
    fn method_name(&self) -> &'static str {
        "FCI"
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

/// Full Configuration Interaction solver.
pub struct FCI<'a> {
    pub hf: &'a HFResult,
    pub tol: f64,
    pub max_iterations: usize,
}

impl<'a> FCI<'a> {
    pub fn new(hf: &'a HFResult) -> Self {
        Self {
            hf,
            tol: 1e-10,
            max_iterations: 150,
        }
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
}

impl<'a> PostHFMethod for FCI<'a> {
    type Output = FCIResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let c=&self.hf.mo_coefficients;
        if !self.hf.converged {return Err("FCI requires a converged reference".into());}
        let h=transform_1e_ao_to_mo(c,&self.hf.core_hamiltonian);
        let g=transform_eri_ao_to_mo(c,&self.hf.eri);
        let engine=crate::fci::engine::FciHamiltonian::new(&h,&g,self.hf.n_occ,self.hf.n_occ,self.hf.nuclear_repulsion_energy).map_err(|e|e.to_string())?;
        let solution=engine.solve(&crate::fci::engine::FciSolverOptions{max_iterations:self.max_iterations,residual_tol:self.tol,..Default::default()}).map_err(|e|e.to_string())?;
        let state=&solution.states[0];
        Ok(FCIResult{reference_energy:self.hf.total_energy,total_energy:state.energy,
            correlation_energy:state.energy-self.hf.total_energy,n_determinants:engine.dimension(),ground_state_vector:state.vector.clone(),
            one_rdm_mo:Matrix::from_vec(c.cols,c.cols,engine.rdm(&state.vector,1).map_err(|e|e.to_string())?),
            mo_coefficients:c.clone(),two_rdm_mo:engine.rdm(&state.vector,2).map_err(|e|e.to_string())?})
    }
}

/// Computes the exact analytical nuclear gradient for Full CI (FCI) wavefunctions.
///
/// Within a complete determinant space, the FCI wavefunction is completely variational
/// with respect to all CI coefficients (dE/dc_I = 0) and completely invariant to any
/// unitary orbital rotations. The exact analytical gradient is determined by contracting
/// the exact 1-RDM, energy-weighted density W, and 2-RDM with bare integral derivatives:
///
///   g_{A, \tau} = g_{A, \tau}^{nuc} + sum_{\mu\nu} D_{\mu\nu} h_{\mu\nu}^{(A,\tau)}
///                 - sum_{\mu\nu} W_{\mu\nu} S_{\mu\nu}^{(A,\tau)}
///                 + sum_{\mu \in A, \nu\lambda\sigma} Gamma_{\mu\nu\lambda\sigma}^{AO, symm} (\mu' \nu | \lambda \sigma)
pub fn compute_fci_analytical_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    fci: &FCIResult,
) -> Vec<[f64; 3]> {
    let n_atoms = atoms.len();
    let n_basis = basis.len();
    let c = &fci.mo_coefficients;

    let h_mo = transform_1e_ao_to_mo(c, &hf.core_hamiltonian);
    let mo_eri = transform_eri_ao_to_mo(c, &hf.eri);

    let idx4 = |p: usize, q: usize, r: usize, s: usize| -> usize {
        ((p * n_basis + q) * n_basis + r) * n_basis + s
    };

    // 1. Generalized Fock matrix: F_{pq} = sum_r D_{pr} h_{rq} + sum_{rst} Gamma_{prst} (qr|st)
    let mut f_gen = Matrix::zeros(n_basis, n_basis);
    for p in 0..n_basis {
        for q in 0..n_basis {
            let mut sum = 0.0;
            for r in 0..n_basis {
                let d_pr = fci.one_rdm_mo[(p, r)];
                if d_pr.abs() > 1e-15 {
                    sum += d_pr * h_mo[(r, q)];
                }
            }
            for r in 0..n_basis {
                for s in 0..n_basis {
                    for t in 0..n_basis {
                        let g_prst = fci.two_rdm_mo[idx4(p, r, s, t)];
                        if g_prst.abs() > 1e-15 {
                            sum += g_prst * mo_eri.get(q, r, s, t);
                        }
                    }
                }
            }
            f_gen[(p, q)] = sum;
        }
    }

    // 2. Transform 2-RDM to AO basis
    let mut gamma_ao = vec![vec![vec![vec![0.0; n_basis]; n_basis]; n_basis]; n_basis];
    for p in 0..n_basis {
        for q in 0..n_basis {
            for r in 0..n_basis {
                for s in 0..n_basis {
                    let g_val = fci.two_rdm_mo[idx4(p, q, r, s)];
                    if g_val.abs() < 1e-15 {
                        continue;
                    }
                    for mu in 0..n_basis {
                        let c_mu = c[(mu, p)];
                        if c_mu.abs() < 1e-15 { continue; }
                        for nu in 0..n_basis {
                            let c_nu = c[(nu, q)];
                            let c_munu = c_mu * c_nu;
                            if c_munu.abs() < 1e-15 { continue; }
                            for la in 0..n_basis {
                                let c_la = c[(la, r)];
                                let c_munula = c_munu * c_la;
                                if c_munula.abs() < 1e-15 { continue; }
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
                    let s = gamma_ao[mu][nu][la][si]
                        + gamma_ao[nu][mu][la][si]
                        + gamma_ao[mu][nu][si][la]
                        + gamma_ao[nu][mu][si][la]
                        + gamma_ao[la][si][mu][nu]
                        + gamma_ao[si][la][mu][nu]
                        + gamma_ao[la][si][nu][mu]
                        + gamma_ao[si][la][nu][mu];
                    gamma_symm[mu][nu][la][si] = s * 0.125;
                }
            }
        }
    }

    // 4. Construct 1-RDM and energy-weighted density in AO basis
    let d_ao = c.dot(&fci.one_rdm_mo).dot(&c.transpose());
    let mut w_mo_symm = Matrix::zeros(n_basis, n_basis);
    for p in 0..n_basis {
        for q in 0..n_basis {
            w_mo_symm[(p, q)] = 0.5 * (f_gen[(p, q)] + f_gen[(q, p)]);
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
    fn test_h2_fci_exact() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0,  0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, ConvAlgo::Adiissoscf, None)
            .expect("HF failed");

        let fci = FCI::new(&hf).compute().expect("FCI failed");
        eprintln!("FCI total energy: {:.10} Eh, corr: {:.10} Eh", fci.total_energy, fci.correlation_energy);
        assert_eq!(fci.n_determinants, 4); // (2 choose 1) * (2 choose 1) = 4
        assert!(fci.correlation_energy < 0.0);
        assert!(fci.total_energy < fci.reference_energy);

        // Trace of 1-RDM must equal number of electrons (2)
        let n_elec = fci.one_rdm_mo.trace();
        assert!((n_elec - 2.0).abs() < 1e-10, "1-RDM trace = {}", n_elec);
    }
}
