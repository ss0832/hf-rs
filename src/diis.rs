// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! SCF Convergence Accelerators:
//! - CDIIS (Commutator Direct Inversion in the Iterative Subspace, Pulay)
//! - ADIIS (Augmented Roothaan-Hall DIIS, Hu & Yang)
//! - SOSCF (Second-Order SCF with orbital rotation Newton steps, Chaban & Bacskay)
//! - ADIIS -> SOSCF (Default: robust global energy minimization switching to quadratic SOSCF)
//! - ADIIS -> CDIIS (ADIIS early phase switching to commutator DIIS)
//! - GDM (Geometric Direct Minimization with line search on the Grassmann manifold)

use crate::matrix::Matrix;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvAlgo {
    /// ADIIS (far from convergence) switching to CDIIS (Default). `adiis_soscf` is kept as an alias.
    Adiissoscf,
    /// Energy DIIS (Kudin-Scuseria-Cances)
    Ediis,
    /// Commutator DIIS (Pulay DIIS)
    Cdiis,
    /// Augmented Roothaan-Hall DIIS
    Adiis,
    /// ADIIS initial phase switching to CDIIS
    AdiisCdiis,
    /// Second-Order SCF
    Soscf,
    /// Geometric Direct Minimization with line search
    Gdm,
    /// Trust-Region Augmented Hessian (TRAH) second-order SCF
    Trah,
}

impl ConvAlgo {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().replace(['-', '_'], "").as_str() {
            "adiissoscf" | "default" => ConvAlgo::Adiissoscf,
            "cdiis" | "diis" | "pulay" => ConvAlgo::Cdiis,
            "adiis" => ConvAlgo::Adiis,
            "ediis" => ConvAlgo::Ediis,
            "adiiscdiis" => ConvAlgo::AdiisCdiis,
            "soscf" => ConvAlgo::Soscf,
            "gdm" => ConvAlgo::Gdm,
            "trah" | "trustregion" | "trust_region" => ConvAlgo::Trah,
            _ => ConvAlgo::Adiissoscf,
        }
    }
}

/// Computes the commutator error vector matrix e = F * P * S - S * P * F
pub fn compute_diis_error(f: &Matrix, p: &Matrix, s: &Matrix) -> Matrix {
    let fps = f.dot(p).dot(s);
    let spf = s.dot(p).dot(f);
    fps.sub(&spf)
}

/// Evaluates maximum absolute element in a matrix
pub fn max_abs_element(m: &Matrix) -> f64 {
    let mut max_val: f64 = 0.0;
    for &x in &m.data {
        let abs_x = x.abs();
        if abs_x > max_val {
            max_val = abs_x;
        }
    }
    max_val
}

/// CDIIS (Pulay DIIS) accelerator
pub struct CdiisEngine {
    pub max_subspace: usize,
    pub fock_history: Vec<Matrix>,
    pub error_history: Vec<Matrix>,
}

impl CdiisEngine {
    pub fn new(max_subspace: usize) -> Self {
        Self {
            max_subspace: max_subspace.max(2),
            fock_history: Vec::new(),
            error_history: Vec::new(),
        }
    }

    pub fn push(&mut self, fock: Matrix, error: Matrix) {
        if self.fock_history.len() >= self.max_subspace {
            self.fock_history.remove(0);
            self.error_history.remove(0);
        }
        self.fock_history.push(fock);
        self.error_history.push(error);
    }

    /// Extrapolates Fock matrix using Pulay's linear system
    pub fn extrapolate(&self) -> Option<Matrix> {
        let m = self.error_history.len();
        if m < 2 {
            return None;
        }

        let dim = m + 1;
        let mut b = Matrix::zeros(dim, dim);

        for i in 0..m {
            for j in 0..=i {
                let mut dot = 0.0;
                let ei = &self.error_history[i];
                let ej = &self.error_history[j];
                for k in 0..ei.data.len() {
                    dot += ei.data[k] * ej.data[k];
                }
                b[(i, j)] = dot;
                b[(j, i)] = dot;
            }
            b[(i, m)] = -1.0;
            b[(m, i)] = -1.0;
        }
        b[(m, m)] = 0.0;

        let mut rhs = vec![0.0; dim];
        rhs[m] = -1.0;

        let coeffs = solve_linear_system(&b, &rhs)?;

        let mut f_extrap = Matrix::zeros(self.fock_history[0].rows, self.fock_history[0].cols);
        for i in 0..m {
            let c_i = coeffs[i];
            for k in 0..f_extrap.data.len() {
                f_extrap.data[k] += c_i * self.fock_history[i].data[k];
            }
        }

        Some(f_extrap)
    }
}

/// Joint UHF CDIIS accelerator (Pulay 1982)
pub struct UhfCdiisEngine {
    pub max_subspace: usize,
    pub fock_a_history: Vec<Matrix>,
    pub fock_b_history: Vec<Matrix>,
    pub error_a_history: Vec<Matrix>,
    pub error_b_history: Vec<Matrix>,
}

impl UhfCdiisEngine {
    pub fn new(max_subspace: usize) -> Self {
        Self {
            max_subspace: max_subspace.max(2),
            fock_a_history: Vec::new(),
            fock_b_history: Vec::new(),
            error_a_history: Vec::new(),
            error_b_history: Vec::new(),
        }
    }

    pub fn push(&mut self, fa: Matrix, fb: Matrix, ea: Matrix, eb: Matrix) {
        if self.fock_a_history.len() >= self.max_subspace {
            self.fock_a_history.remove(0);
            self.fock_b_history.remove(0);
            self.error_a_history.remove(0);
            self.error_b_history.remove(0);
        }
        self.fock_a_history.push(fa);
        self.fock_b_history.push(fb);
        self.error_a_history.push(ea);
        self.error_b_history.push(eb);
    }

    pub fn extrapolate(&self) -> Option<(Matrix, Matrix)> {
        let m = self.error_a_history.len();
        if m < 2 {
            return None;
        }

        let dim = m + 1;
        let mut b = Matrix::zeros(dim, dim);

        for i in 0..m {
            for j in 0..=i {
                let mut dot = 0.0;
                let eai = &self.error_a_history[i];
                let eaj = &self.error_a_history[j];
                let ebi = &self.error_b_history[i];
                let ebj = &self.error_b_history[j];
                for k in 0..eai.data.len() {
                    dot += eai.data[k] * eaj.data[k] + ebi.data[k] * ebj.data[k];
                }
                b[(i, j)] = dot;
                b[(j, i)] = dot;
            }
            b[(i, m)] = -1.0;
            b[(m, i)] = -1.0;
        }
        b[(m, m)] = 0.0;

        let mut rhs = vec![0.0; dim];
        rhs[m] = -1.0;

        let coeffs = solve_linear_system(&b, &rhs)?;

        let n_rows = self.fock_a_history[0].rows;
        let n_cols = self.fock_a_history[0].cols;
        let mut fa_extrap = Matrix::zeros(n_rows, n_cols);
        let mut fb_extrap = Matrix::zeros(n_rows, n_cols);

        for i in 0..m {
            let c_i = coeffs[i];
            for k in 0..fa_extrap.data.len() {
                fa_extrap.data[k] += c_i * self.fock_a_history[i].data[k];
                fb_extrap.data[k] += c_i * self.fock_b_history[i].data[k];
            }
        }

        Some((fa_extrap, fb_extrap))
    }
}

/// Euclidean projection of `v` onto the probability simplex {c_i >= 0, sum c_i = 1}.
fn project_onto_simplex(v: &mut [f64]) {
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut css = 0.0;
    let mut theta = 0.0;
    for (k, &uk) in u.iter().enumerate() {
        css += uk;
        let t = (css - 1.0) / (k + 1) as f64;
        if uk - t > 0.0 {
            theta = t;
        }
    }
    for x in v.iter_mut() {
        *x = (*x - theta).max(0.0);
    }
}

/// Minimizes f(c) = 1/2 c^T Q c + b^T c over the simplex with projected gradient descent
/// (fixed step 1/L, L from the Gershgorin bound of the symmetrized Q).
fn minimize_quadratic_on_simplex(q: &[Vec<f64>], b: &[f64], start: &[f64]) -> Vec<f64> {
    let m = b.len();
    let lipschitz = (0..m)
        .map(|i| (0..m).map(|j| 0.5 * (q[i][j] + q[j][i]).abs()).sum::<f64>())
        .fold(1e-12_f64, f64::max);
    let step = 1.0 / lipschitz;
    let mut c = start.to_vec();
    project_onto_simplex(&mut c);
    for _ in 0..5000 {
        let mut next = vec![0.0; m];
        for i in 0..m {
            let mut g = b[i];
            for j in 0..m {
                g += 0.5 * (q[i][j] + q[j][i]) * c[j];
            }
            next[i] = c[i] - step * g;
        }
        project_onto_simplex(&mut next);
        let diff: f64 = next.iter().zip(&c).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        c = next;
        if diff < 1e-14 {
            break;
        }
    }
    c
}

/// Frobenius inner product Tr(A^T B) of two equally sized matrices.
fn frob(a: &Matrix, b: &Matrix) -> f64 {
    a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum()
}

fn linear_combination(mats: &[Matrix], c: &[f64]) -> Matrix {
    let mut out = Matrix::zeros(mats[0].rows, mats[0].cols);
    for (m, &ci) in mats.iter().zip(c) {
        for (o, x) in out.data.iter_mut().zip(&m.data) {
            *o += ci * x;
        }
    }
    out
}

/// ADIIS (Augmented Roothaan-Hall DIIS, Hu & Yang 2010) with total densities `P = 2 D`.
///
/// Minimizes the exact second-order expansion of the HF energy of the interpolated density about the
/// most recent iterate n:
/// `f(c) = E_n + sum_i c_i Tr[(P_i - P_n) F_n] + 1/2 sum_ij c_i c_j Tr[(P_i - P_n)(F_j - F_n)]`
/// on the simplex. For HF this expansion is exact, so `model_energy(c) = E_HF(sum_i c_i P_i)`.
pub struct AdiisEngine {
    pub max_subspace: usize,
    pub fock_history: Vec<Matrix>,
    pub density_history: Vec<Matrix>,
    pub energy_history: Vec<f64>,
}

impl AdiisEngine {
    pub fn new(max_subspace: usize) -> Self {
        Self {
            max_subspace: max_subspace.max(2),
            fock_history: Vec::new(),
            density_history: Vec::new(),
            energy_history: Vec::new(),
        }
    }

    pub fn push(&mut self, fock: Matrix, density: Matrix, energy: f64) {
        if self.fock_history.len() >= self.max_subspace {
            self.fock_history.remove(0);
            self.density_history.remove(0);
            self.energy_history.remove(0);
        }
        self.fock_history.push(fock);
        self.density_history.push(density);
        self.energy_history.push(energy);
    }

    /// Quadratic model `(Q, b, E_n)` with `f(c) = E_n + b^T c + 1/2 c^T Q c`.
    fn model(&self) -> (Vec<Vec<f64>>, Vec<f64>, f64) {
        let m = self.fock_history.len();
        let n = m - 1;
        let (pn, fnn) = (&self.density_history[n], &self.fock_history[n]);
        let dp: Vec<Matrix> = self.density_history.iter().map(|p| p.sub(pn)).collect();
        let df: Vec<Matrix> = self.fock_history.iter().map(|f| f.sub(fnn)).collect();
        let b: Vec<f64> = (0..m).map(|i| frob(&dp[i], fnn)).collect();
        let q: Vec<Vec<f64>> = (0..m).map(|i| (0..m).map(|j| frob(&dp[i], &df[j])).collect()).collect();
        (q, b, self.energy_history[n])
    }

    /// Model energy `f(c)` (equals the HF energy of the interpolated density).
    pub fn model_energy(&self, c: &[f64]) -> f64 {
        let (q, b, e_n) = self.model();
        let m = b.len();
        let mut e = e_n;
        for i in 0..m {
            e += b[i] * c[i];
            for j in 0..m {
                e += 0.5 * c[i] * q[i][j] * c[j];
            }
        }
        e
    }

    /// Optimal simplex weights.
    pub fn weights(&self) -> Option<Vec<f64>> {
        let m = self.fock_history.len();
        if m < 2 {
            return None;
        }
        let (q, b, _) = self.model();
        let mut start = vec![0.0; m];
        start[m - 1] = 1.0;
        Some(minimize_quadratic_on_simplex(&q, &b, &start))
    }

    /// Extrapolated Fock matrix `F = sum_i c_i F_i`.
    pub fn extrapolate(&self) -> Option<Matrix> {
        self.weights().map(|c| linear_combination(&self.fock_history, &c))
    }
}

/// EDIIS (Kudin, Scuseria & Cances 2002): `E(c) = sum_i c_i E_i - 1/4 sum_ij c_i c_j Tr[(P_i - P_j)(F_i - F_j)]`
/// for total densities `P`. For HF this equals the energy of the interpolated density.
pub struct EdiisEngine {
    pub max_subspace: usize,
    pub fock_history: Vec<Matrix>,
    pub density_history: Vec<Matrix>,
    pub energy_history: Vec<f64>,
}

impl EdiisEngine {
    pub fn new(max_subspace: usize) -> Self {
        Self {
            max_subspace: max_subspace.max(2),
            fock_history: Vec::new(),
            density_history: Vec::new(),
            energy_history: Vec::new(),
        }
    }

    pub fn push(&mut self, fock: Matrix, density: Matrix, energy: f64) {
        if self.fock_history.len() >= self.max_subspace {
            self.fock_history.remove(0);
            self.density_history.remove(0);
            self.energy_history.remove(0);
        }
        self.fock_history.push(fock);
        self.density_history.push(density);
        self.energy_history.push(energy);
    }

    fn pair_matrix(&self) -> Vec<Vec<f64>> {
        let m = self.fock_history.len();
        (0..m)
            .map(|i| {
                (0..m)
                    .map(|j| {
                        frob(
                            &self.density_history[i].sub(&self.density_history[j]),
                            &self.fock_history[i].sub(&self.fock_history[j]),
                        )
                    })
                    .collect()
            })
            .collect()
    }

    pub fn model_energy(&self, c: &[f64]) -> f64 {
        let x = self.pair_matrix();
        let m = c.len();
        let mut e: f64 = (0..m).map(|i| c[i] * self.energy_history[i]).sum();
        for i in 0..m {
            for j in 0..m {
                e -= 0.25 * c[i] * c[j] * x[i][j];
            }
        }
        e
    }

    pub fn weights(&self) -> Option<Vec<f64>> {
        let m = self.fock_history.len();
        if m < 2 {
            return None;
        }
        let x = self.pair_matrix();
        let q: Vec<Vec<f64>> = x.iter().map(|row| row.iter().map(|v| -0.5 * v).collect()).collect();
        let mut start = vec![0.0; m];
        start[m - 1] = 1.0;
        Some(minimize_quadratic_on_simplex(&q, &self.energy_history, &start))
    }

    pub fn extrapolate(&self) -> Option<Matrix> {
        self.weights().map(|c| linear_combination(&self.fock_history, &c))
    }
}

/// Second-Order SCF (SOSCF) step generator.
/// Evaluates orbital gradient g_{ai} = 4 * F_{ai} in MO basis and performs
/// Newton-Raphson orbital rotation \theta_{ai} = -F_{ai} / (\epsilon_a - \epsilon_i).
pub fn soscf_step(
    c: &Matrix,
    f: &Matrix,
    s: &Matrix,
    mo_energies: &[f64],
    n_occ: usize,
) -> (Matrix, Matrix, f64) {
    let n_basis = c.cols;
    let n_virt = n_basis - n_occ;

    // Transform Fock matrix to MO basis: F_MO = C^T * F * C
    let f_mo = c.transpose().dot(f).dot(c);

    let mut theta = Matrix::zeros(n_basis, n_basis);
    let mut max_grad: f64 = 0.0;

    for a in 0..n_virt {
        let a_mo = n_occ + a;
        for i in 0..n_occ {
            let f_ai = f_mo[(a_mo, i)];
            let abs_f = f_ai.abs();
            if abs_f > max_grad {
                max_grad = abs_f;
            }

            let denom = mo_energies[a_mo] - mo_energies[i];
            let raw_delta = if denom.abs() > 1e-4 {
                -f_ai / denom
            } else {
                -f_ai.signum() * 0.2
            };
            let delta = raw_delta.clamp(-0.2, 0.2);

            // Anti-symmetric generator: K_{i, a} = -\theta, K_{a, i} = \theta
            theta[(i, a_mo)] = -delta;
            theta[(a_mo, i)] = delta;
        }
    }

    // Update orbitals: C_trial = C * (I + theta)
    let c_trial = c.dot(&theta.exp());

    // Orthonormalize via Lowdin: S_MO = C^T * S * C, X = S_MO^{-1/2}, C_new = C * X
    let s_mo = c_trial.transpose().dot(s).dot(&c_trial);
    let x_ortho = s_mo.inv_sqrt();
    let c_new = c_trial.dot(&x_ortho);

    // Form updated density matrix P = 2 * \sum_{i=0}^{n_occ-1} C_i * C_i^T
    let p_new = c_new.occupied_density(n_occ, 2.0);

    (c_new, p_new, max_grad)
}

/// Geometric Direct Minimization (GDM) line search step.
/// Directly minimizes the Hartree-Fock energy along the orbital gradient direction on the Grassmann manifold.
pub fn gdm_step(
    c: &Matrix,
    f: &Matrix,
    s: &Matrix,
    n_occ: usize,
    step_size: f64,
) -> (Matrix, Matrix, f64) {
    let n_basis = c.cols;
    let n_virt = n_basis - n_occ;
    let f_mo = c.transpose().dot(f).dot(c);

    let mut grad_k = Matrix::zeros(n_basis, n_basis);
    let mut max_grad: f64 = 0.0;

    for a in 0..n_virt {
        let a_mo = n_occ + a;
        for i in 0..n_occ {
            let g_ai = f_mo[(a_mo, i)];
            let abs_g = g_ai.abs();
            if abs_g > max_grad {
                max_grad = abs_g;
            }
            grad_k[(i, a_mo)] = step_size * g_ai;
            grad_k[(a_mo, i)] = -step_size * g_ai;
        }
    }

    let c_trial = c.dot(&grad_k.exp());
    let s_mo = c_trial.transpose().dot(s).dot(&c_trial);
    let x_ortho = s_mo.inv_sqrt();
    let c_new = c_trial.dot(&x_ortho);

    let p_new = c_new.occupied_density(n_occ, 2.0);

    (c_new, p_new, max_grad)
}

/// Solves Ax = b using LU decomposition with partial pivoting
fn solve_linear_system(a: &Matrix, b: &[f64]) -> Option<Vec<f64>> {
    let n = a.rows;
    if a.cols != n || b.len() != n {
        return None;
    }

    let mut mat = a.clone();
    let mut x = b.to_vec();
    let mut piv: Vec<usize> = (0..n).collect();

    // Gaussian elimination with partial pivoting
    for i in 0..n {
        let mut max_row = i;
        let mut max_val = mat[(i, i)].abs();
        for k in (i + 1)..n {
            let val = mat[(k, i)].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }

        if max_val < 1e-28 {
            return None;
        }

        if max_row != i {
            piv.swap(i, max_row);
            x.swap(i, max_row);
            for c in 0..n {
                let tmp = mat[(i, c)];
                mat[(i, c)] = mat[(max_row, c)];
                mat[(max_row, c)] = tmp;
            }
        }

        let diag = mat[(i, i)];
        for j in (i + 1)..n {
            let factor = mat[(j, i)] / diag;
            mat[(j, i)] = factor;
            for k in (i + 1)..n {
                mat[(j, k)] -= factor * mat[(i, k)];
            }
            x[j] -= factor * x[i];
        }
    }

    // Back-substitution
    for i in (0..n).rev() {
        let mut sum = x[i];
        for k in (i + 1)..n {
            sum -= mat[(i, k)] * x[k];
        }
        x[i] = sum / mat[(i, i)];
    }

    Some(x)
}

/// Trust-Region Augmented Hessian (TRAH) second-order SCF convergence engine.
///
/// Designed specifically for post-HF electronic structure methods (MP2, MP3, CC, CASSCF),
/// where tight and robust convergence to true local minima is required without divergence.
///
/// Combines:
/// - Orbital gradient evaluation in the MO basis: g_{ai} = 4 * F_{ai}
/// - Augmented Hessian eigenvalue problem: A * [1, x]^T = \lambda * [1, x]^T
///   guaranteeing a strictly positive-definite level-shifted Hessian (H - \lambda I)
/// - Adaptive trust-region radius scaling based on predicted vs actual energy change
/// - Exact unitary orbital rotation via Cayley transform: U = (I - 0.5 * \Theta)^{-1} * (I + 0.5 * \Theta)
/// - Subspace canonicalization separating occupied and virtual orbitals
#[derive(Clone, Debug)]
pub struct TrahEngine {
    pub trust_radius: f64,
    pub min_radius: f64,
    pub max_radius: f64,
    pub prev_energy: Option<f64>,
    pub prev_pred_drop: Option<f64>,
}

impl TrahEngine {
    pub fn new(initial_radius: f64) -> Self {
        Self {
            trust_radius: initial_radius.clamp(1e-4, 1.0),
            min_radius: 1e-4,
            max_radius: 1.0,
            prev_energy: None,
            prev_pred_drop: None,
        }
    }

    /// Executes one TRAH iteration step.
    /// Returns (C_new, P_new, mo_energies_new, max_grad_norm).
    pub fn step(
        &mut self,
        c: &Matrix,
        f: &Matrix,
        s: &Matrix,
        mo_energies: &[f64],
        n_occ: usize,
        current_energy: f64,
    ) -> (Matrix, Matrix, Vec<f64>, f64) {
        let n_basis = c.cols;
        let n_virt = n_basis - n_occ;
        let m = n_occ * n_virt;

        // 1. Adaptive trust-region update based on previous step
        if let (Some(prev_e), Some(pred_drop)) = (self.prev_energy, self.prev_pred_drop) {
            let actual_drop = prev_e - current_energy; // Positive if energy decreased
            let rho = if pred_drop.abs() > 1e-15 {
                actual_drop / pred_drop
            } else {
                1.0
            };

            if rho < 0.25 || actual_drop < 0.0 {
                self.trust_radius = (self.trust_radius * 0.5).max(self.min_radius);
            } else if rho > 0.75 {
                self.trust_radius = (self.trust_radius * 1.5).min(self.max_radius);
            }
        }

        if m == 0 {
            let mut p = Matrix::zeros(c.rows, c.rows);
            for mu in 0..c.rows {
                for nu in 0..c.rows {
                    let mut val = 0.0;
                    for i in 0..n_occ {
                        val += 2.0 * c[(mu, i)] * c[(nu, i)];
                    }
                    p[(mu, nu)] = val;
                }
            }
            return (c.clone(), p, mo_energies.to_vec(), 0.0);
        }

        // 2. Transform Fock matrix to MO basis: F_MO = C^T * F * C
        let f_mo = c.transpose().dot(f).dot(c);

        // 3. Orbital gradient g and diagonal Hessian H
        let mut g = vec![0.0; m];
        let mut h_diag = vec![0.0; m];
        let mut max_grad: f64 = 0.0;

        for a in 0..n_virt {
            let a_mo = n_occ + a;
            for i in 0..n_occ {
                let k = a * n_occ + i;
                let f_ai = f_mo[(a_mo, i)];
                let g_k = 4.0 * f_ai;
                g[k] = g_k;
                let abs_g = g_k.abs();
                if abs_g > max_grad {
                    max_grad = abs_g;
                }

                let gap = mo_energies[a_mo] - mo_energies[i];
                let h_k = 4.0 * gap.max(1e-4);
                h_diag[k] = h_k;
            }
        }

        // 4. Build Augmented Hessian matrix A of size (m+1) x (m+1):
        // [ 0    g^T   ]
        // [ g   diag(H) ]
        let dim = m + 1;
        let mut a_aug = Matrix::zeros(dim, dim);
        for k in 0..m {
            a_aug[(0, k + 1)] = g[k];
            a_aug[(k + 1, 0)] = g[k];
            a_aug[(k + 1, k + 1)] = h_diag[k];
        }

        // Diagonalize Augmented Hessian
        let (evals, evecs) = a_aug.eigh();
        let lambda_0 = evals[0];

        // Extract eigenvector for lowest eigenvalue
        let mut v0 = evecs[(0, 0)];
        let mut vx = vec![0.0; m];
        for k in 0..m {
            vx[k] = evecs[(k + 1, 0)];
        }

        if v0 < 0.0 {
            v0 = -v0;
            for x_val in &mut vx {
                *x_val = -*x_val;
            }
        }

        // Step: x_k = vx_k / v0
        let mut x = if v0 > 1e-12 {
            vx.iter().map(|&val| val / v0).collect::<Vec<f64>>()
        } else {
            // Fallback: standard level-shifted gradient step
            (0..m).map(|k| -g[k] / (h_diag[k] - lambda_0.min(-1e-4))).collect()
        };

        // 5. Check and enforce trust-region boundary: ||x|| <= trust_radius
        let step_norm = (x.iter().map(|&v| v * v).sum::<f64>()).sqrt();
        let delta = self.trust_radius;

        if step_norm > delta && step_norm > 1e-14 {
            // Solve 1D secular equation for level shift mu: sum_k (g_k / (H_k + mu))^2 = delta^2
            let mut mu = (-lambda_0).max(1e-4);
            for _ in 0..8 {
                let mut phi = -delta * delta;
                let mut dphi = 0.0;
                for k in 0..m {
                    let denom = h_diag[k] + mu;
                    let gk_sq = g[k] * g[k];
                    phi += gk_sq / (denom * denom);
                    dphi -= 2.0 * gk_sq / (denom * denom * denom);
                }
                if dphi.abs() < 1e-14 {
                    break;
                }
                let step_mu = phi / dphi;
                mu -= step_mu;
                if mu < 0.0 {
                    mu = 1e-4;
                }
                if step_mu.abs() < 1e-8 {
                    break;
                }
            }

            // Update step with optimal level shift
            for k in 0..m {
                x[k] = -g[k] / (h_diag[k] + mu);
            }

            // Fallback safety scaling if still slightly exceeding delta
            let final_norm = (x.iter().map(|&v| v * v).sum::<f64>()).sqrt();
            if final_norm > delta {
                let scale = delta / final_norm;
                for v in &mut x {
                    *v *= scale;
                }
            }
        }

        // 6. Compute predicted energy drop: \Delta E_pred = - \sum_k (g_k * x_k + 0.5 * H_k * x_k^2)
        let mut pred_drop = 0.0;
        for k in 0..m {
            pred_drop -= g[k] * x[k] + 0.5 * h_diag[k] * x[k] * x[k];
        }
        self.prev_energy = Some(current_energy);
        self.prev_pred_drop = Some(pred_drop.max(1e-15));

        // 7. Form anti-symmetric generator Theta: \Theta_{i, a} = -x_{ai}, \Theta_{a, i} = x_{ai}
        let mut theta = Matrix::zeros(n_basis, n_basis);
        for a in 0..n_virt {
            let a_mo = n_occ + a;
            for i in 0..n_occ {
                let k = a * n_occ + i;
                let x_k = x[k];
                theta[(i, a_mo)] = -x_k;
                theta[(a_mo, i)] = x_k;
            }
        }

        // 8. Unitary orbital rotation via Cayley transform: U = (I - 0.5 * Theta)^(-1) * (I + 0.5 * Theta)
        let ident = Matrix::identity(n_basis);
        let half_theta = theta.scale(0.5);
        let i_minus_k = ident.sub(&half_theta);
        let i_plus_k = ident.add(&half_theta);

        let u_mat = if let Some(inv) = i_minus_k.inverse() {
            inv.dot(&i_plus_k)
        } else {
            ident.add(&theta)
        };

        let c_rot = c.dot(&u_mat);

        // 9. Lowdin orthonormalization for numerical cleanliness
        let s_mo = c_rot.transpose().dot(s).dot(&c_rot);
        let x_ortho = s_mo.inv_sqrt();
        let c_ortho = c_rot.dot(&x_ortho);

        // 10. Subspace canonicalization: diagonalize occupied-occupied and virtual-virtual blocks of F_MO
        let f_trial = c_ortho.transpose().dot(f).dot(&c_ortho);

        let mut f_oo = Matrix::zeros(n_occ, n_occ);
        for i in 0..n_occ {
            for j in 0..n_occ {
                f_oo[(i, j)] = f_trial[(i, j)];
            }
        }
        let (evals_o, w_o) = f_oo.eigh();

        let mut f_vv = Matrix::zeros(n_virt, n_virt);
        for a in 0..n_virt {
            for b in 0..n_virt {
                f_vv[(a, b)] = f_trial[(n_occ + a, n_occ + b)];
            }
        }
        let (evals_v, w_v) = f_vv.eigh();

        let mut c_final = Matrix::zeros(n_basis, n_basis);
        // Occupied columns: C_occ * W_o
        for mu in 0..c.rows {
            for j in 0..n_occ {
                let mut sum = 0.0;
                for i in 0..n_occ {
                    sum += c_ortho[(mu, i)] * w_o[(i, j)];
                }
                c_final[(mu, j)] = sum;
            }
        }
        // Virtual columns: C_virt * W_v
        for mu in 0..c.rows {
            for b in 0..n_virt {
                let mut sum = 0.0;
                for a in 0..n_virt {
                    sum += c_ortho[(mu, n_occ + a)] * w_v[(a, b)];
                }
                c_final[(mu, n_occ + b)] = sum;
            }
        }

        let mut mo_energies_new = Vec::with_capacity(n_basis);
        mo_energies_new.extend(evals_o);
        mo_energies_new.extend(evals_v);

        // Form updated density matrix P = 2 * \sum_{i=0}^{n_occ-1} C_i * C_i^T
        let mut p_new = Matrix::zeros(n_basis, n_basis);
        for mu in 0..c.rows {
            for nu in 0..c.rows {
                let mut val = 0.0;
                for i in 0..n_occ {
                    val += 2.0 * c_final[(mu, i)] * c_final[(nu, i)];
                }
                p_new[(mu, nu)] = val;
            }
        }

        (c_final, p_new, mo_energies_new, max_grad)
    }
}

impl Default for TrahEngine {
    fn default() -> Self {
        Self::new(0.4)
    }
}

