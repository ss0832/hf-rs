// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Configuration Interaction Singles and Doubles (CISD).
//!
//! Restricts the determinant expansion to the reference determinant,
//! all single excitations (S), and all double excitations (D).

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

/// Results of a CISD calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct CISDResult {
    pub reference_energy: f64,
    pub total_energy: f64,
    pub correlation_energy: f64,
    pub n_determinants: usize,
    pub c0: f64, // Weight of reference determinant
    pub one_rdm_mo: Matrix,
    pub two_rdm_mo: Vec<f64>,
    pub mo_coefficients: Matrix,
}

impl PostHFResult for CISDResult {
    fn method_name(&self) -> &'static str {
        "CISD"
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
}

/// Configuration Interaction Singles and Doubles solver.
pub struct CISD<'a> {
    pub hf: &'a HFResult,
}

impl<'a> CISD<'a> {
    pub fn new(hf: &'a HFResult) -> Self {
        Self { hf }
    }
}

impl<'a> PostHFMethod for CISD<'a> {
    type Output = CISDResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let mut options=crate::options::PostHFOptions{active:Some(crate::options::ActiveSpace::new(0,0,0)),..Default::default()};
        options.cas.optimize_orbitals=false;
        let result=super::mrci_engine::compute_mrci(crate::context::Reference::Rhf(self.hf),&options,None,false).map_err(|e|e.to_string())?;
        let n=self.hf.n_occ;let occupied=if n==64 {u64::MAX}else{(1u64<<n)-1};
        let ref_index=result.determinants.iter().position(|&(a,b)|a.0==occupied && b.0==occupied).ok_or_else(||"CISD reference determinant is missing".to_string())?;
        let two=result.two_rdm(0).map_err(|e|e.to_string())?;
        Ok(CISDResult{reference_energy:self.hf.total_energy,total_energy:result.output.e_total,correlation_energy:result.output.e_corr,
            n_determinants:result.determinants.len(),c0:result.states[0].vector[ref_index],
            one_rdm_mo:result.output.one_rdm.ok_or_else(||"CISD one-RDM is missing".to_string())?,two_rdm_mo:two,
            mo_coefficients:result.mo_coefficients})
    }
}

/// Computes the exact analytical nuclear gradient for CISD wavefunctions using the Handy-Schaefer Z-vector method.
///
/// In CISD, the wavefunction is variational with respect to CI coefficients (dE/dc_I = 0)
/// but not with respect to orbital rotations between occupied and virtual subspaces.
/// The orbital response is accounted for by solving the Coupled-Perturbed Hartree-Fock (CPHF)
/// Z-vector equation:
///
///   sum_{bj} (A + B)_{ai, bj} Z_{bj} = - 2 (F_{ai}^gen - F_{ia}^gen)
///
/// where F_{pq}^gen is the generalized Fock matrix constructed from the exact CISD 1- and 2-RDMs.
pub fn compute_cisd_analytical_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    cisd: &CISDResult,
) -> Vec<[f64; 3]> {
    let n_atoms = atoms.len();
    let n_basis = basis.len();
    let n_occ = hf.n_occ;
    let n_virt = n_basis - n_occ;
    let c = &cisd.mo_coefficients;
    let eps = &hf.orbital_energies;

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
                let d_pr = cisd.one_rdm_mo[(p, r)];
                if d_pr.abs() > 1e-15 {
                    sum += d_pr * h_mo[(r, q)];
                }
            }
            for r in 0..n_basis {
                for s in 0..n_basis {
                    for t in 0..n_basis {
                        let g_prst = cisd.two_rdm_mo[idx4(p, r, s, t)];
                        if g_prst.abs() > 1e-15 {
                            sum += g_prst * mo_eri.get(q, r, s, t);
                        }
                    }
                }
            }
            f_gen[(p, q)] = sum;
        }
    }

    // 2. Orbital Lagrangian / CPHF Z-vector source:
    // L_{ai} = 2 (F_{ai}^gen - F_{ia}^gen)
    let n_cphf = n_virt * n_occ;
    let mut rhs = vec![0.0; n_cphf];
    for a in 0..n_virt {
        let a_mo = n_occ + a;
        for i in 0..n_occ {
            let row = a * n_occ + i;
            rhs[row] = f_gen[(a_mo, i)] - f_gen[(i, a_mo)];
        }
    }

    // 3. Build CPHF coupling matrix:
    // A_{ai, bj} = (eps_a - eps_i) delta_{ab} delta_{ij} + 4(ai|bj) - (ab|ji) - (aj|bi)
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

    // 4. Solve CPHF Z-vector equation: A z = rhs
    let z_sol = if n_cphf > 0 {
        let lu = a_mat.lu_decompose().expect("CPHF coupling matrix is invertible in CISD Z-vector");
        lu.solve(&rhs)
    } else {
        Vec::new()
    };

    // 5. Construct relaxed 1-RDM and relaxed energy-weighted density matrix in MO basis
    let mut d_mo_relaxed = cisd.one_rdm_mo.clone();
    let mut w_mo_relaxed = Matrix::zeros(n_basis, n_basis);

    for p in 0..n_basis {
        for q in 0..n_basis {
            w_mo_relaxed[(p, q)] = 0.5 * (f_gen[(p, q)] + f_gen[(q, p)]);
        }
    }

    for a in 0..n_virt {
        let a_mo = n_occ + a;
        for i in 0..n_occ {
            let z_ai = z_sol[a * n_occ + i];
            // Relaxed 1-RDM has off-diagonal virtual-occupied contributions:
            d_mo_relaxed[(a_mo, i)] += z_ai;
            d_mo_relaxed[(i, a_mo)] += z_ai;

            // Relaxed energy-weighted density has orbital eigenvalue responses:
            let w_val = f_gen[(a_mo, i)] + eps[i] * z_ai;
            w_mo_relaxed[(a_mo, i)] = w_val;
            w_mo_relaxed[(i, a_mo)] = w_val;
        }
    }

    // Add Handy-Schaefer Z-vector two-electron field to occupied-occupied block of W:
    // W_{ij} += 0.5 * sum_{a, k} z_{ak} [ 4(ij|ak) - (ik|aj) - (jk|ai) ]
    for i in 0..n_occ {
        for j in 0..n_occ {
            let mut sum_z_field = 0.0;
            for a in 0..n_virt {
                let a_mo = n_occ + a;
                for k in 0..n_occ {
                    let z_ak = z_sol[a * n_occ + k];
                    let ijak = mo_eri.get(i, j, a_mo, k);
                    let ikaj = mo_eri.get(i, k, a_mo, j);
                    let jkai = mo_eri.get(j, k, a_mo, i);
                    sum_z_field += 0.5 * z_ak * (4.0 * ijak - ikaj - jkai);
                }
            }
            w_mo_relaxed[(i, j)] += sum_z_field;
        }
    }

    // 6. Transform 2-RDM to AO basis and 8-fold symmetrize
    let mut gamma_ao = vec![vec![vec![vec![0.0; n_basis]; n_basis]; n_basis]; n_basis];
    for p in 0..n_basis {
        for q in 0..n_basis {
            for r in 0..n_basis {
                for s in 0..n_basis {
                    let g_val = cisd.two_rdm_mo[idx4(p, q, r, s)];
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

    // Add Z-vector 2-electron skeleton relaxation coupling with HF reference density:
    let mut z_mo = Matrix::zeros(n_basis, n_basis);
    for a in 0..n_virt {
        let a_mo = n_occ + a;
        for i in 0..n_occ {
            let z_val = z_sol[a * n_occ + i];
            z_mo[(a_mo, i)] = z_val;
            z_mo[(i, a_mo)] = z_val;
        }
    }
    let p_z_ao = c.dot(&z_mo).dot(&c.transpose());
    let p_hf_ao = &hf.density_matrix;

    for mu in 0..n_basis {
        for nu in 0..n_basis {
            for la in 0..n_basis {
                for si in 0..n_basis {
                    let g_relax = 2.0 * (
                        p_hf_ao[(mu, nu)] * p_z_ao[(la, si)] + p_z_ao[(mu, nu)] * p_hf_ao[(la, si)]
                    ) - (
                        p_hf_ao[(mu, la)] * p_z_ao[(nu, si)] + p_z_ao[(mu, la)] * p_hf_ao[(nu, si)]
                    );
                    gamma_symm[mu][nu][la][si] += 0.5 * g_relax;
                }
            }
        }
    }

    // 7. Transform relaxed 1-RDM and W to AO basis
    let d_ao = c.dot(&d_mo_relaxed).dot(&c.transpose());
    let w_ao = c.dot(&w_mo_relaxed).dot(&c.transpose());

    // 8. Precompute basis function spatial derivatives
    let mut basis_derivs = Vec::with_capacity(n_basis);
    for cg in basis {
        let dx = differentiate_basis_function(cg, 0);
        let dy = differentiate_basis_function(cg, 1);
        let dz = differentiate_basis_function(cg, 2);
        basis_derivs.push([dx, dy, dz]);
    }

    let g_nuc = compute_nuclear_repulsion_gradient(atoms);

    // 9. Assemble analytical nuclear gradient
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

                // Two-electron derivative term
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

                grad_a[tau] += d_t_val + d_v_val - d_s_val + d_eri_val;
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
    fn test_h2_cisd() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0,  0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, ConvAlgo::Adiissoscf, None)
            .expect("HF failed");

        let cisd = CISD::new(&hf).compute().expect("CISD failed");
        assert!(cisd.correlation_energy < 0.0);
        assert!(cisd.c0.abs() > 0.9); // Dominant HF reference
    }
}
