// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Real UHF -> UHF internal stability analysis.
//!
//! For real occupied-virtual rotations `kappa^sigma_ai` the UHF energy is
//! `E(kappa) = E0 + 2 sum F_ai kappa_ai + kappa (A + B) kappa + O(kappa^3)` with
//!
//! `(A + B)_{ai sigma, bj tau} = delta_st (delta_ij F_ab - delta_ab F_ij)
//!     + 2 (ai|bj) - delta_st [(ab|ij) + (aj|bi)]`.
//!
//! Products `(A + B) kappa` are formed AO-direct from one UHF Fock build of the symmetrized
//! transition densities `D^s = C_v kappa^s C_o^T + transpose`:
//! `[(A + B) kappa]^s = F^s_vv kappa^s - kappa^s F^s_oo + C_v^T (J[D^a + D^b] - K[D^s]) C_o`.
//! The lowest eigenpair is found by Davidson iteration, so no dense `(ov)^2` matrix is built.

use crate::matrix::Matrix;
use crate::scf::{UHFResult, build_uhf_fock_matrices};

/// Lowest eigenpair of the real UHF orbital Hessian `A + B`.
#[derive(Clone, Debug)]
pub struct UhfStability {
    /// Lowest eigenvalue (Hartree). Negative values are internal instabilities.
    pub lowest_eigenvalue: f64,
    /// Normalized eigenvector, alpha block (`n_virt_alpha x n_alpha`).
    pub alpha: Matrix,
    /// Normalized eigenvector, beta block (`n_virt_beta x n_beta`).
    pub beta: Matrix,
}

/// Eigenvalues below this are treated as instabilities.
pub const INSTABILITY_THRESHOLD: f64 = -1e-5;

struct Spin {
    occ: Matrix,
    virt: Matrix,
    f_oo: Matrix,
    f_vv: Matrix,
}

impl Spin {
    fn new(c: &Matrix, fock: &Matrix, n_occ: usize) -> Self {
        let n_mo = c.cols;
        let columns = |range: std::ops::Range<usize>| {
            Matrix::from_vec(
                c.rows,
                range.len(),
                (0..c.rows)
                    .flat_map(|mu| range.clone().map(move |p| (mu, p)))
                    .map(|(mu, p)| c[(mu, p)])
                    .collect(),
            )
        };
        let (occ, virt) = (columns(0..n_occ), columns(n_occ..n_mo));
        Self {
            f_oo: occ.transpose().dot(fock).dot(&occ),
            f_vv: virt.transpose().dot(fock).dot(&virt),
            occ,
            virt,
        }
    }

    fn len(&self) -> usize {
        self.occ.cols * self.virt.cols
    }

    fn unpack(&self, x: &[f64]) -> Matrix {
        Matrix::from_vec(self.virt.cols, self.occ.cols, x.to_vec())
    }

    fn diagonal(&self) -> Vec<f64> {
        (0..self.virt.cols)
            .flat_map(|a| (0..self.occ.cols).map(move |i| (a, i)))
            .map(|(a, i)| self.f_vv[(a, a)] - self.f_oo[(i, i)])
            .collect()
    }

    fn transition_density(&self, kappa: &Matrix) -> Matrix {
        let d = self.virt.dot(kappa).dot(&self.occ.transpose());
        d.add(&d.transpose())
    }
}

struct Hessian<'a> {
    alpha: Spin,
    beta: Spin,
    eri: &'a [f64],
    n_ao: usize,
}

impl Hessian<'_> {
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        let na = self.alpha.len();
        let (ka, kb) = (self.alpha.unpack(&x[..na]), self.beta.unpack(&x[na..]));
        let zero = Matrix::zeros(self.n_ao, self.n_ao);
        let (ga, gb) = build_uhf_fock_matrices(
            &zero,
            &self.alpha.transition_density(&ka),
            &self.beta.transition_density(&kb),
            self.eri,
            self.n_ao,
        );
        let block = |spin: &Spin, k: &Matrix, g: &Matrix| {
            spin.f_vv
                .dot(k)
                .sub(&k.dot(&spin.f_oo))
                .add(&spin.virt.transpose().dot(g).dot(&spin.occ))
        };
        let mut out = block(&self.alpha, &ka, &ga).data;
        out.extend(block(&self.beta, &kb, &gb).data);
        out
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Lowest eigenpair of the real UHF -> UHF orbital Hessian of a converged UHF solution.
pub fn uhf_internal_stability(uhf: &UHFResult) -> Result<UhfStability, String> {
    if !uhf.converged {
        return Err("UHF stability analysis requires a converged solution".into());
    }
    let n_ao = uhf.overlap_matrix.rows;
    let hessian = Hessian {
        alpha: Spin::new(&uhf.mo_coefficients_alpha, &uhf.fock_matrix_alpha, uhf.n_alpha),
        beta: Spin::new(&uhf.mo_coefficients_beta, &uhf.fock_matrix_beta, uhf.n_beta),
        eri: &uhf.eri,
        n_ao,
    };
    let na = hessian.alpha.len();
    let dim = na + hessian.beta.len();
    let split = |x: &[f64]| (hessian.alpha.unpack(&x[..na]), hessian.beta.unpack(&x[na..]));
    if dim == 0 {
        let (alpha, beta) = split(&[]);
        return Ok(UhfStability { lowest_eigenvalue: f64::INFINITY, alpha, beta });
    }
    let mut diagonal = hessian.alpha.diagonal();
    diagonal.extend(hessian.beta.diagonal());
    // Start from the unit vectors of the smallest diagonal elements, which spans
    // spin-symmetric and spin-antisymmetric combinations of degenerate pairs.
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&i, &j| diagonal[i].total_cmp(&diagonal[j]).then(i.cmp(&j)));
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut images: Vec<Vec<f64>> = Vec::new();
    let add = |mut v: Vec<f64>, basis: &mut Vec<Vec<f64>>, images: &mut Vec<Vec<f64>>| {
        for _ in 0..2 {
            for b in basis.iter() {
                let c = dot(&v, b);
                v.iter_mut().zip(b).for_each(|(x, y)| *x -= c * y);
            }
        }
        let norm = dot(&v, &v).sqrt();
        if norm > 1e-8 {
            v.iter_mut().for_each(|x| *x /= norm);
            images.push(hessian.apply(&v));
            basis.push(v);
        }
    };
    for &i in order.iter().take(dim.min(6)) {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        add(v, &mut basis, &mut images);
    }
    for _ in 0..200 {
        let m = basis.len();
        let mut small = Matrix::zeros(m, m);
        for i in 0..m {
            for j in 0..=i {
                let value = 0.5 * (dot(&basis[i], &images[j]) + dot(&basis[j], &images[i]));
                small[(i, j)] = value;
                small[(j, i)] = value;
            }
        }
        let (values, vectors) = small.eigh();
        let theta = values[0];
        let mut x = vec![0.0; dim];
        let mut ax = vec![0.0; dim];
        for k in 0..m {
            let w = vectors[(k, 0)];
            x.iter_mut().zip(&basis[k]).for_each(|(a, b)| *a += w * b);
            ax.iter_mut().zip(&images[k]).for_each(|(a, b)| *a += w * b);
        }
        let residual: Vec<f64> = ax.iter().zip(&x).map(|(a, b)| a - theta * b).collect();
        if dot(&residual, &residual).sqrt() < 1e-7 || m == dim {
            let (alpha, beta) = split(&x);
            return Ok(UhfStability { lowest_eigenvalue: theta, alpha, beta });
        }
        if m >= 40 {
            basis.clear();
            images.clear();
            add(x, &mut basis, &mut images);
        }
        let correction: Vec<f64> = residual
            .iter()
            .zip(&diagonal)
            .map(|(r, d)| {
                let denominator = d - theta;
                -r / if denominator.abs() < 1e-4 { 1e-4f64.copysign(denominator) } else { denominator }
            })
            .collect();
        add(correction, &mut basis, &mut images);
    }
    Err("UHF stability Davidson iterations did not converge".into())
}

/// Rotated orbitals `C exp(X)` along the lowest Hessian eigenvector when the solution is
/// internally unstable; `None` if it is stable.
pub fn follow_internal_instability(uhf: &UHFResult) -> Result<Option<(Matrix, Matrix)>, String> {
    let stability = uhf_internal_stability(uhf)?;
    if stability.lowest_eigenvalue >= INSTABILITY_THRESHOLD {
        return Ok(None);
    }
    // A finite step along the descending mode; the SCF then relaxes from the rotated orbitals.
    const STEP: f64 = 0.5;
    let rotate = |c: &Matrix, n_occ: usize, kappa: &Matrix| {
        let n_mo = c.cols;
        let mut x = Matrix::zeros(n_mo, n_mo);
        for a in 0..kappa.rows {
            for i in 0..kappa.cols {
                x[(n_occ + a, i)] = STEP * kappa[(a, i)];
                x[(i, n_occ + a)] = -STEP * kappa[(a, i)];
            }
        }
        c.dot(&crate::post_hf::mr_dsrg::cumulants::exp_antisymmetric(&x))
    };
    Ok(Some((
        rotate(&uhf.mo_coefficients_alpha, uhf.n_alpha, &stability.alpha),
        rotate(&uhf.mo_coefficients_beta, uhf.n_beta, &stability.beta),
    )))
}
