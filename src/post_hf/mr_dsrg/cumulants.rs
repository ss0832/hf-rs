// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Generalized Normal Ordering (GNO) and Reduced Density Cumulants (Mukherjee & Kutzelnigg, 1997).
//!
//! Provides spin-summed and spin-orbital 1-body reduced density matrices ($\gamma_1$),
//! 1-body hole density matrices ($\eta_1$), 2-body density cumulants ($\lambda_2$),
//! 3-body density cumulants ($\lambda_3$), generalized Fock matrices, and semi-canonical
//! orbital transformations for multireference electronic structure methods.

use crate::mo_integrals::{MoEriTensor, transform_1e_ao_to_mo, transform_eri_ao_to_mo};
use crate::matrix::Matrix;
use crate::scf::HFResult;

/// Computes the exact orthogonal matrix exponential $U = \exp(\kappa)$ of an antisymmetric
/// matrix $\kappa^T = -\kappa$ via the Cayley transform $U = (I - \frac{1}{2}\kappa)^{-1}(I + \frac{1}{2}\kappa)$.
pub fn exp_antisymmetric(kappa: &Matrix) -> Matrix {
    let n = kappa.rows;
    let ident = Matrix::identity(n);
    let half_k = kappa.scale(0.5);
    let lhs = ident.sub(&half_k);
    let rhs = ident.add(&half_k);
    if let Some(lu) = lhs.lu_decompose() {
        let mut u = Matrix::zeros(n, n);
        for col in 0..n {
            let b: Vec<f64> = (0..n).map(|r| rhs[(r, col)]).collect();
            let x = lu.solve(&b);
            for r in 0..n {
                u[(r, col)] = x[r];
            }
        }
        u
    } else {
        ident.add(kappa)
    }
}

/// Approximation scheme for the 3-particle density cumulant ($\lambda_3$).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreePDC {
    /// Legacy 1/2-RDM reconstruction. Exact spin-orbital lambda3 requires
    /// `density_cumulants::exact_density_cumulants` and a true three-RDM.
    MK,
    /// Approximate reconstruction / decomposition of 3-body RDM from 1- and 2-body cumulants.
    MKDecomp,
    /// Neglect the 3-body cumulant ($\lambda_3 = 0$).
    Zero,
}

/// Complete cumulant decomposition of a multireference state in the MO basis.
#[derive(Clone, Debug)]
pub struct CumulantDecomposition {
    /// Spin-summed 1-RDM $\gamma_q^p = \langle \Psi_0 | \hat{E}_p^q | \Psi_0 \rangle$ (`n_mo x n_mo`).
    pub gamma1: Matrix,
    /// Spin-summed 1-hole density $\eta_q^p = 2 \delta_q^p - \gamma_q^p$ (`n_mo x n_mo`).
    pub eta1: Matrix,
    /// Spin-summed 2-RDM $\gamma_{rs}^{pq}$ flattened as `[p * N^3 + q * N^2 + r * N + s]` in chemist's `(pq|rs)` or physicist's `<pq|rs>` indexing.
    /// Stored here in chemist's notation $\Gamma_{pqrs} = \langle \hat{e}_{pq,rs} \rangle$ where $E_{\text{2e}} = \frac{1}{2}\sum_{pqrs} \Gamma_{pqrs}(pq|rs)$.
    pub gamma2: Vec<f64>,
    /// Spin-free 2-body density cumulant $\lambda_{pqrs} = \Gamma_{pqrs} - \gamma_{pq}\gamma_{rs} + \frac{1}{2}\gamma_{ps}\gamma_{rq}$.
    pub lambda2: Vec<f64>,
    /// Norm of the 2-body cumulant $\|\lambda_2\|_F$ (diagnostic of multireference static correlation).
    pub lambda2_norm: f64,
    /// Effective 3-body cumulant norm / weight under the chosen `ThreePDC` approximation.
    pub lambda3_norm: f64,
    /// Active-space 3-body cumulant tensor \lambda_{uvw}^{xyz} (size n_act^6, or empty if n_act==0 or ThreePDC::Zero).
    pub lambda3_active: Vec<f64>,
    /// Generalized Fock matrix in MO basis: $f_{pq} = h_{pq} + \sum_{rs} \gamma_{rs} \left[(pq|rs) - \frac{1}{2}(pr|qs)\right]$.
    pub generalized_fock: Matrix,
    /// Diagonal elements of the generalized Fock matrix $\epsilon_p = f_{pp}$.
    pub orbital_energies: Vec<f64>,
    /// Semi-canonical MO coefficient matrix (`n_ao x n_mo`).
    pub semi_canonical_coefficients: Matrix,
}

impl CumulantDecomposition {
    /// Returns the 2-body cumulant element $\lambda_{pqrs}$ in chemist's notation.
    #[inline]
    pub fn get_lambda2(&self, n_mo: usize, p: usize, q: usize, r: usize, s: usize) -> f64 {
        self.lambda2[((p * n_mo + q) * n_mo + r) * n_mo + s]
    }

    /// Returns the active 3-body cumulant element \lambda_{uvw}^{xyz}.
    #[inline]
    pub fn get_lambda3(&self, n_act: usize, u: usize, v: usize, w: usize, x: usize, y: usize, z: usize) -> f64 {
        if self.lambda3_active.is_empty() || n_act == 0 {
            0.0
        } else {
            let idx = ((((u * n_act + v) * n_act + w) * n_act + x) * n_act + y) * n_act + z;
            self.lambda3_active.get(idx).copied().unwrap_or(0.0)
        }
    }

    /// Returns the antisymmetrized spin-adapted 2-body cumulant in physicist's `<pq||rs>` notation:
    /// $\Lambda_{rs}^{pq} = \lambda_{prqs} - \frac{1}{2} \lambda_{psqr}$.
    #[inline]
    pub fn get_phys_lambda2(&self, n_mo: usize, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let dir = self.get_lambda2(n_mo, p, r, q, s);
        let exch = self.get_lambda2(n_mo, p, s, q, r);
        dir - 0.5 * exch
    }
}

/// Computes the Generalized Fock matrix in the MO basis:
/// $$f_{pq} = h_{pq} + \sum_{rs} \gamma_{rs} \left[(pq|rs) - \frac{1}{2}(pr|qs)\right]$$
pub fn build_generalized_fock_mo(
    h_mo: &Matrix,
    mo_eri: &MoEriTensor,
    gamma1: &Matrix,
) -> Matrix {
    let n_mo = h_mo.rows;
    let mut f_gen = h_mo.clone();
    for p in 0..n_mo {
        for q in 0..=p {
            let mut two_e = 0.0;
            for r in 0..n_mo {
                for s in 0..n_mo {
                    let g_rs = gamma1[(r, s)];
                    if g_rs.abs() > 1e-15 {
                        two_e += g_rs * (mo_eri.get(p, q, r, s) - 0.5 * mo_eri.get(p, r, q, s));
                    }
                }
            }
            f_gen[(p, q)] += two_e;
            f_gen[(q, p)] = f_gen[(p, q)];
        }
    }
    f_gen
}

/// Performs a block-diagonal semi-canonical transformation of the core, active, and virtual
/// orbital subspaces so that the generalized Fock matrix is diagonal within each subspace.
///
/// Returns `(c_semicanon, eps_semicanon, u_rot)` where `c_semicanon = c_mo * u_rot`.
pub fn semi_canonicalize(
    hf: &HFResult,
    gamma1_mo: &Matrix,
    n_core: usize,
    n_act: usize,
) -> (Matrix, Vec<f64>, Matrix) {
    let n_mo = hf.mo_coefficients.cols;
    let h_mo = transform_1e_ao_to_mo(&hf.mo_coefficients, &hf.core_hamiltonian);
    let mo_eri = transform_eri_ao_to_mo(&hf.mo_coefficients, &hf.eri);
    let f_gen = build_generalized_fock_mo(&h_mo, &mo_eri, gamma1_mo);

    let mut u_rot = Matrix::identity(n_mo);
    let mut eps = vec![0.0; n_mo];

    let blocks = [
        (0, n_core),
        (n_core, n_core + n_act),
        (n_core + n_act, n_mo),
    ];

    for &(start, end) in &blocks {
        let dim = end.saturating_sub(start);
        if dim == 0 {
            continue;
        }
        let mut f_sub = Matrix::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                f_sub[(i, j)] = f_gen[(start + i, start + j)];
            }
        }
        let (evals, evecs) = f_sub.eigh();
        for i in 0..dim {
            eps[start + i] = evals[i];
            for j in 0..dim {
                u_rot[(start + i, start + j)] = evecs[(i, j)];
            }
        }
    }

    let c_semi = hf.mo_coefficients.dot(&u_rot);
    (c_semi, eps, u_rot)
}

/// Legacy spin-free 1/2-RDM decomposition with reconstructed three-body terms.
/// Use `density_cumulants` for exact spin-orbital three-body cumulants.
pub fn compute_cumulants(
    hf: &HFResult,
    rdm1_mo: &Matrix,
    rdm2_mo: &[f64],
    n_core: usize,
    n_act: usize,
    three_pdc: ThreePDC,
) -> CumulantDecomposition {
    let n_mo = rdm1_mo.rows;
    let (c_semi, _, u_rot) = semi_canonicalize(hf, rdm1_mo, n_core, n_act);
    let ut = u_rot.transpose();

    // Rotate 1-RDM into the semi-canonical basis
    let gamma1 = ut.dot(rdm1_mo).dot(&u_rot);
    let mut eta1 = Matrix::zeros(n_mo, n_mo);
    for p in 0..n_mo {
        for q in 0..n_mo {
            eta1[(p, q)] = (if p == q { 2.0 } else { 0.0 }) - gamma1[(p, q)];
        }
    }

    // Rotate 2-RDM into the semi-canonical basis via 4 quarter-transformations
    let idx = |p: usize, q: usize, r: usize, s: usize| ((p * n_mo + q) * n_mo + r) * n_mo + s;
    let full_len = n_mo * n_mo * n_mo * n_mo;
    let full_rdm2_owned: Vec<f64>;
    let rdm2_full: &[f64] = if rdm2_mo.len() == full_len {
        rdm2_mo
    } else {
        let mut full = vec![0.0; full_len];
        for p in 0..n_mo {
            for q in 0..n_mo {
                for r in 0..n_mo {
                    for s in 0..n_mo {
                        let in_act = p >= n_core
                            && p < n_core + n_act
                            && q >= n_core
                            && q < n_core + n_act
                            && r >= n_core
                            && r < n_core + n_act
                            && s >= n_core
                            && s < n_core + n_act;
                        if in_act && rdm2_mo.len() == n_act * n_act * n_act * n_act {
                            let u = p - n_core;
                            let v = q - n_core;
                            let x = r - n_core;
                            let y = s - n_core;
                            full[idx(p, q, r, s)] =
                                rdm2_mo[((u * n_act + v) * n_act + x) * n_act + y];
                        } else {
                            full[idx(p, q, r, s)] = rdm1_mo[(p, q)] * rdm1_mo[(r, s)]
                                - 0.5 * rdm1_mo[(p, s)] * rdm1_mo[(r, q)];
                        }
                    }
                }
            }
        }
        full_rdm2_owned = full;
        &full_rdm2_owned
    };

    let mut t1 = vec![0.0; full_len];
    for p in 0..n_mo {
        for b in 0..n_mo {
            let u_bp = u_rot[(b, p)];
            if u_bp.abs() < 1e-15 {
                continue;
            }
            for q in 0..n_mo {
                for r in 0..n_mo {
                    for s in 0..n_mo {
                        t1[idx(p, q, r, s)] += u_bp * rdm2_full[idx(b, q, r, s)];
                    }
                }
            }
        }
    }
    let mut t2 = vec![0.0; n_mo * n_mo * n_mo * n_mo];
    for p in 0..n_mo {
        for q in 0..n_mo {
            for b in 0..n_mo {
                let u_bq = u_rot[(b, q)];
                if u_bq.abs() < 1e-15 {
                    continue;
                }
                for r in 0..n_mo {
                    for s in 0..n_mo {
                        t2[idx(p, q, r, s)] += u_bq * t1[idx(p, b, r, s)];
                    }
                }
            }
        }
    }
    let mut t3 = vec![0.0; n_mo * n_mo * n_mo * n_mo];
    for p in 0..n_mo {
        for q in 0..n_mo {
            for r in 0..n_mo {
                for b in 0..n_mo {
                    let u_br = u_rot[(b, r)];
                    if u_br.abs() < 1e-15 {
                        continue;
                    }
                    for s in 0..n_mo {
                        t3[idx(p, q, r, s)] += u_br * t2[idx(p, q, b, s)];
                    }
                }
            }
        }
    }
    let mut gamma2 = vec![0.0; n_mo * n_mo * n_mo * n_mo];
    for p in 0..n_mo {
        for q in 0..n_mo {
            for r in 0..n_mo {
                for s in 0..n_mo {
                    let mut sum = 0.0;
                    for b in 0..n_mo {
                        sum += u_rot[(b, s)] * t3[idx(p, q, r, b)];
                    }
                    gamma2[idx(p, q, r, s)] = sum;
                }
            }
        }
    }

    // Compute spin-summed 2-body cumulant in chemist's notation:
    // lambda2(p,q,r,s) = Gamma(p,q,r,s) - gamma(p,q)*gamma(r,s) + 0.5 * gamma(p,s)*gamma(r,q)
    let mut lambda2 = vec![0.0; n_mo * n_mo * n_mo * n_mo];
    let mut l2_sq = 0.0;
    for p in 0..n_mo {
        for q in 0..n_mo {
            for r in 0..n_mo {
                for s in 0..n_mo {
                    let g2 = gamma2[idx(p, q, r, s)];
                    let disconnected = gamma1[(p, q)] * gamma1[(r, s)] - 0.5 * gamma1[(p, s)] * gamma1[(r, q)];
                    let l2 = g2 - disconnected;
                    lambda2[idx(p, q, r, s)] = l2;
                    l2_sq += l2 * l2;
                }
            }
        }
    }

    // Compute 3-body cumulant tensor and exact Frobenius norm in active space
    let n_act_6 = n_act * n_act * n_act * n_act * n_act * n_act;
    let mut lambda3_active = vec![0.0; n_act_6];
    let idx_act = |u: usize, v: usize, w: usize, x: usize, y: usize, z: usize| {
        ((((u * n_act + v) * n_act + w) * n_act + x) * n_act + y) * n_act + z
    };

    if three_pdc != ThreePDC::Zero && n_act > 0 {
        for u in 0..n_act {
            let u_mo = n_core + u;
            for v in 0..n_act {
                let v_mo = n_core + v;
                for w in 0..n_act {
                    let w_mo = n_core + w;
                    for x in 0..n_act {
                        let x_mo = n_core + x;
                        for y in 0..n_act {
                            let y_mo = n_core + y;
                            for z in 0..n_act {
                                let z_mo = n_core + z;
                                let mut val = 0.0;
                                match three_pdc {
                                    ThreePDC::MK => {
                                        for t in 0..n_act {
                                            let t_mo = n_core + t;
                                            let g_tt = gamma1[(t_mo, t_mo)];
                                            let h_tt = eta1[(t_mo, t_mo)];
                                            let weight = 0.5 * (g_tt * h_tt * 0.5);
                                            let l2_uvxt = lambda2[idx(u_mo, v_mo, x_mo, t_mo)];
                                            let l2_twyz = lambda2[idx(t_mo, w_mo, y_mo, z_mo)];
                                            let l2_uvyt = lambda2[idx(u_mo, v_mo, y_mo, t_mo)];
                                            let l2_twxz = lambda2[idx(t_mo, w_mo, x_mo, z_mo)];
                                            let l2_uvzt = lambda2[idx(u_mo, v_mo, z_mo, t_mo)];
                                            let l2_twxy = lambda2[idx(t_mo, w_mo, x_mo, y_mo)];
                                            val += weight * (l2_uvxt * l2_twyz - 0.5 * l2_uvyt * l2_twxz - 0.5 * l2_uvzt * l2_twxy);
                                        }
                                    }
                                    ThreePDC::MKDecomp => {
                                        for t in 0..n_act {
                                            let t_mo = n_core + t;
                                            let l2_uvxt = lambda2[idx(u_mo, v_mo, x_mo, t_mo)];
                                            let l2_twyz = lambda2[idx(t_mo, w_mo, y_mo, z_mo)];
                                            let l2_uvyt = lambda2[idx(u_mo, v_mo, y_mo, t_mo)];
                                            let l2_twxz = lambda2[idx(t_mo, w_mo, x_mo, z_mo)];
                                            let l2_uvzt = lambda2[idx(u_mo, v_mo, z_mo, t_mo)];
                                            let l2_twxy = lambda2[idx(t_mo, w_mo, x_mo, y_mo)];
                                            val += 0.25 * (l2_uvxt * l2_twyz - 0.5 * l2_uvyt * l2_twxz - 0.5 * l2_uvzt * l2_twxy);
                                        }
                                    }
                                    ThreePDC::Zero => {}
                                }
                                lambda3_active[idx_act(u, v, w, x, y, z)] = val;
                            }
                        }
                    }
                }
            }
        }
    }

    let l3_norm = lambda3_active.iter().map(|&x| x * x).sum::<f64>().sqrt();

    let h_semi = transform_1e_ao_to_mo(&c_semi, &hf.core_hamiltonian);
    let eri_semi = transform_eri_ao_to_mo(&c_semi, &hf.eri);
    let f_semi = build_generalized_fock_mo(&h_semi, &eri_semi, &gamma1);
    let eps_semi: Vec<f64> = (0..n_mo).map(|p| f_semi[(p, p)]).collect();

    CumulantDecomposition {
        gamma1,
        eta1,
        gamma2,
        lambda2,
        lambda2_norm: l2_sq.sqrt(),
        lambda3_norm: l3_norm,
        lambda3_active,
        generalized_fock: f_semi,
        orbital_energies: eps_semi,
        semi_canonical_coefficients: c_semi,
    }
}

/// Evaluates the exact GNO scalar reference energy $E_0 = \langle \Psi_0 | \hat{H} | \Psi_0 \rangle$
/// from the 1-RDM $\gamma_1$ and 2-body cumulant $\lambda_2$:
/// $$E_0 = \sum_{pq} \gamma_{pq} h_{pq} + \frac{1}{2} \sum_{pqrs} \left[\gamma_{pq}\gamma_{rs} - \frac{1}{2}\gamma_{ps}\gamma_{rq} + \lambda_{pqrs}\right] (pq|rs) + E_{\text{nuc}}$$
pub fn evaluate_gno_reference_energy(
    h_mo: &Matrix,
    mo_eri: &MoEriTensor,
    gamma1: &Matrix,
    lambda2: &[f64],
    nuclear_repulsion: f64,
) -> f64 {
    let n_mo = h_mo.rows;
    let mut e1 = 0.0;
    let mut e2 = 0.0;
    for p in 0..n_mo {
        for q in 0..n_mo {
            e1 += gamma1[(p, q)] * h_mo[(p, q)];
            for r in 0..n_mo {
                for s in 0..n_mo {
                    let l2 = lambda2[((p * n_mo + q) * n_mo + r) * n_mo + s];
                    let g2 = gamma1[(p, q)] * gamma1[(r, s)] - 0.5 * gamma1[(p, s)] * gamma1[(r, q)] + l2;
                    e2 += 0.5 * g2 * mo_eri.get(p, q, r, s);
                }
            }
        }
    }
    e1 + e2 + nuclear_repulsion
}
