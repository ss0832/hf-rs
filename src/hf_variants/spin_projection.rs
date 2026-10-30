// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Exact Lowdin spin projection of a UHF determinant (projection after variation, PUHF).
//!
//! For a UHF determinant `|Phi>` with `S_z = M = (N_a - N_b)/2 >= 0` the projector onto `S = M` is
//! `P = (2S+1)/2 int_0^pi d^S_{MM}(b) exp(-i b S_y) sin(b) db` with `d^S_{SS}(b) = cos^{2S}(b/2)`
//! (the alpha/gamma integrals are trivial because `|Phi>` is an `S_z` eigenstate). Since `P`
//! commutes with `H` and is idempotent,
//!
//! `E_PUHF = <Phi|H P|Phi> / <Phi|P|Phi> = int w(b) <Phi|H R(b)|Phi> / int w(b) <Phi|R(b)|Phi>`.
//!
//! `R(b)` rotates every spinor by the real matrix `[[cos b/2, -sin b/2], [sin b/2, cos b/2]]`.
//! The overlap `<Phi|R Phi> = det O`, `O = C^T C_R`, and the Hamiltonian kernel follows from the
//! generalized Wick theorem with the transition density `M = C_R O^{-1} C^T`
//! (`<Phi|a+_p a_q|R Phi>/<Phi|R Phi> = M_qp`). In the AO metric, `O = C^T S C_R`.
//! Only occupied spinors and quadratic intermediates are allocated; the packed AO integrals are
//! borrowed from UHF. With `x = cos(b)`, the weighted kernels are polynomials of degree at most
//! `N_alpha`. Gauss-Legendre quadrature integrates them exactly, with a refined-grid check.

use crate::matrix::Matrix;
use crate::scf::{UHFResult, eri_index};

/// Result of an exact spin projection.
#[derive(Clone, Debug)]
pub struct SpinProjection {
    /// Projected energy `<Phi|H P|Phi>/<Phi|P|Phi>` (Hartree, including nuclear repulsion).
    pub energy: f64,
    /// Target spin quantum number `S` (= `M`).
    pub spin: f64,
    /// Weight of the target spin component, `<Phi|P|Phi>`, in (0, 1].
    pub target_weight: f64,
    /// Number of Gauss-Legendre nodes used for the spin-rotation integral.
    pub quadrature_nodes: usize,
}

struct Kernel {
    overlap: f64,
    energy_times_overlap: f64,
}

fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n.div_ceil(2) {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp;
        loop {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        let wi = 2.0 / ((1.0 - z * z) * dp * dp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    (x, w)
}

struct Problem<'a> {
    n: usize,
    n_alpha: usize,
    n_beta: usize,
    h: &'a Matrix,
    eri: &'a [f64],
    alpha: &'a Matrix,
    beta: &'a Matrix,
    overlap: &'a Matrix,
    nuclear: f64,
}

impl Problem<'_> {
    /// Occupied AO spinors (2n x N), for `c = cos(b/2)` and `s = sin(b/2)`.
    fn spinors(&self, c: f64, s: f64) -> Matrix {
        let (n, na, nb) = (self.n, self.n_alpha, self.n_beta);
        let mut out = Matrix::zeros(2 * n, na + nb);
        for i in 0..na {
            for p in 0..n {
                out[(p, i)] = c * self.alpha[(p, i)];
                out[(n + p, i)] = s * self.alpha[(p, i)];
            }
        }
        for j in 0..nb {
            // beta spinor (0, u_j) -> (-s u_j, c u_j)
            for p in 0..n {
                out[(p, na + j)] = -s * self.beta[(p, j)];
                out[(n + p, na + j)] = c * self.beta[(p, j)];
            }
        }
        out
    }

    fn kernel(
        &self,
        reference: &Matrix,
        metric_reference: &Matrix,
        c: f64,
        s: f64,
    ) -> Result<Kernel, String> {
        let rotated = self.spinors(c, s);
        let o = metric_reference.transpose().dot(&rotated);
        let lu = o
            .lu_decompose()
            .ok_or("singular rotated-determinant overlap at a quadrature node")?;
        let det = lu.determinant();
        // Solve all right-hand sides with the same factorization, without forming an inverse.
        let ne = self.n_alpha + self.n_beta;
        let mut solved = Matrix::zeros(ne, 2 * self.n);
        for p in 0..2 * self.n {
            let rhs: Vec<f64> = (0..ne).map(|i| reference[(p, i)]).collect();
            let col = lu.solve(&rhs);
            for i in 0..ne {
                solved[(i, p)] = col[i];
            }
        }
        let m = rotated.dot(&solved); // M_qp = <a+_p a_q>
        let n = self.n;
        let block = |s1: usize, s2: usize| -> Matrix {
            let mut out = Matrix::zeros(n, n);
            for r in 0..n {
                for c in 0..n {
                    out[(r, c)] = m[(s1 * n + r, s2 * n + c)];
                }
            }
            out
        };
        let blocks = [[block(0, 0), block(0, 1)], [block(1, 0), block(1, 1)]];
        let d = blocks[0][0].add(&blocks[1][1]);
        let mut e1 = 0.0;
        for p in 0..n {
            for q in 0..n {
                e1 += self.h[(p, q)] * d[(q, p)];
            }
        }
        // Coulomb: 1/2 sum (pr|qs) D_rp D_sq
        let mut coulomb = 0.0;
        // Exchange: -1/2 sum_{sp,sq} sum (pr|qs) M^{sq sp}_{s p} M^{sp sq}_{r q}
        let mut exchange = 0.0;
        for p in 0..n {
            for r in 0..n {
                for q in 0..n {
                    for s in 0..n {
                        let v = self.eri[eri_index(p, r, q, s)];
                        if v == 0.0 {
                            continue;
                        }
                        coulomb += v * d[(r, p)] * d[(s, q)];
                        for sp in 0..2 {
                            for sq in 0..2 {
                                exchange += v * blocks[sq][sp][(s, p)] * blocks[sp][sq][(r, q)];
                            }
                        }
                    }
                }
            }
        }
        let energy = e1 + 0.5 * coulomb - 0.5 * exchange + self.nuclear;
        Ok(Kernel {
            overlap: det,
            energy_times_overlap: det * energy,
        })
    }

    fn integrate(&self, nodes: usize) -> Result<(f64, f64), String> {
        let reference = self.spinors(1.0, 0.0);
        let mut metric_reference = Matrix::zeros(2 * self.n, self.n_alpha + self.n_beta);
        for spin in 0..2 {
            for p in 0..self.n {
                for i in 0..reference.cols {
                    metric_reference[(spin * self.n + p, i)] = (0..self.n)
                        .map(|q| self.overlap[(p, q)] * reference[(spin * self.n + q, i)])
                        .sum();
                }
            }
        }
        let spin = 0.5 * (self.n_alpha as f64 - self.n_beta as f64);
        let (x, w) = gauss_legendre(nodes);
        let (mut num, mut den) = (0.0, 0.0);
        for (xi, wi) in x.iter().zip(&w) {
            let c = (0.5 * (1.0 + xi)).sqrt();
            let s = (0.5 * (1.0 - xi)).sqrt();
            let weight = wi * c.powi((self.n_alpha - self.n_beta) as i32);
            let k = self.kernel(&reference, &metric_reference, c, s)?;
            num += weight * k.energy_times_overlap;
            den += weight * k.overlap;
        }
        let target_weight = 0.5 * (2.0 * spin + 1.0) * den;
        if !target_weight.is_finite()
            || target_weight <= 0.0
            || target_weight > 1.0 + 1e-8
            || !num.is_finite()
        {
            return Err(format!(
                "invalid target-spin norm in projection: {target_weight}"
            ));
        }
        Ok((num / den, target_weight))
    }
}

/// Exact Lowdin projection of a converged UHF determinant onto `S = (N_a - N_b)/2`.
pub fn project_uhf_spin(uhf: &UHFResult) -> Result<SpinProjection, String> {
    if !uhf.converged {
        return Err("spin projection requires a converged UHF determinant".into());
    }
    if uhf.n_alpha < uhf.n_beta {
        return Err("spin projection expects N_alpha >= N_beta".into());
    }
    let ca = &uhf.mo_coefficients_alpha;
    let cb = &uhf.mo_coefficients_beta;
    let n = ca.rows;
    if ca.rows != cb.rows
        || ca.cols < uhf.n_alpha
        || cb.cols < uhf.n_beta
        || uhf.n_alpha + uhf.n_beta == 0
    {
        return Err("alpha/beta MO spaces differ".into());
    }
    for (coeff, occupied) in [(ca, uhf.n_alpha), (cb, uhf.n_beta)] {
        let c = coeff.leading_columns(occupied);
        let metric = c.transpose().dot(&uhf.overlap_matrix).dot(&c);
        for i in 0..occupied {
            for j in 0..occupied {
                if (metric[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs() > 1e-8 {
                    return Err(
                        "spin projection requires metric-orthonormal occupied orbitals".into(),
                    );
                }
            }
        }
    }
    let problem = Problem {
        n,
        n_alpha: uhf.n_alpha,
        n_beta: uhf.n_beta,
        h: &uhf.core_hamiltonian,
        eri: &uhf.eri,
        alpha: ca,
        beta: cb,
        overlap: &uhf.overlap_matrix,
        nuclear: uhf.nuclear_repulsion_energy,
    };
    let base = ((uhf.n_alpha + 2) / 2).max(2);
    let (e1, w1) = problem.integrate(base)?;
    let (e2, w2) = problem.integrate(base + 4)?;
    if (e1 - e2).abs() > 1e-10 * (1.0 + e2.abs()) || (w1 - w2).abs() > 1e-10 {
        return Err(format!(
            "spin-rotation quadrature not converged ({e1} vs {e2})"
        ));
    }
    Ok(SpinProjection {
        energy: e2,
        spin: 0.5 * (uhf.n_alpha as f64 - uhf.n_beta as f64),
        target_weight: w2,
        quadrature_nodes: base + 4,
    })
}
