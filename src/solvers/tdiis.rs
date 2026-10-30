// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Tensor DIIS (Direct Inversion in the Iterative Subspace) for coupled cluster amplitudes
//! and general multidimensional nonlinear equations.

use crate::matrix::Matrix;

/// Tensor DIIS accelerator for nonlinear amplitude equations (CCD, CCSD, etc.).
#[derive(Clone, Debug)]
pub struct TensorDIIS {
    pub max_subspace: usize,
    pub param_history: Vec<Vec<f64>>,
    pub error_history: Vec<Vec<f64>>,
}

impl TensorDIIS {
    pub fn new(max_subspace: usize) -> Self {
        Self {
            max_subspace: max_subspace.max(2),
            param_history: Vec::new(),
            error_history: Vec::new(),
        }
    }

    /// Appends new parameter vector and residual error vector into the subspace.
    pub fn push(&mut self, param: Vec<f64>, error: Vec<f64>) {
        assert_eq!(param.len(), error.len(), "Parameter and error dimensions must match");
        if self.param_history.len() >= self.max_subspace {
            self.param_history.remove(0);
            self.error_history.remove(0);
        }
        self.param_history.push(param);
        self.error_history.push(error);
    }

    /// Extrapolates optimal parameters by minimizing the norm of the linear combination of residuals.
    pub fn extrapolate(&self) -> Option<Vec<f64>> {
        let m = self.error_history.len();
        if m < 2 {
            return self.param_history.last().cloned();
        }

        let dim = m + 1;
        let mut b = Matrix::zeros(dim, dim);

        for i in 0..m {
            for j in 0..=i {
                let dot: f64 = self.error_history[i]
                    .iter()
                    .zip(&self.error_history[j])
                    .map(|(a, b)| a * b)
                    .sum();
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

        let n_params = self.param_history[0].len();
        let mut p_extrap = vec![0.0; n_params];

        for i in 0..m {
            let c_i = coeffs[i];
            for k in 0..n_params {
                p_extrap[k] += c_i * self.param_history[i][k];
            }
        }

        Some(p_extrap)
    }

    /// Resets the DIIS subspace history.
    pub fn reset(&mut self) {
        self.param_history.clear();
        self.error_history.clear();
    }
}

fn solve_linear_system(a: &Matrix, b: &[f64]) -> Option<Vec<f64>> {
    let n = a.rows;
    if a.cols != n || b.len() != n {
        return None;
    }

    let mut mat = a.clone();
    let mut x = b.to_vec();

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

        if max_val < 1e-14 {
            return None;
        }

        if max_row != i {
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

    for i in (0..n).rev() {
        let mut sum = x[i];
        for k in (i + 1)..n {
            sum -= mat[(i, k)] * x[k];
        }
        x[i] = sum / mat[(i, i)];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tdiis_linear_convergence() {
        let mut tdiis = TensorDIIS::new(5);
        let target = vec![1.0, 2.0, 3.0];

        // Fixed-point iteration with DIIS acceleration: x_{k+1} = x_k - alpha * (x_k - target)
        let alpha = 0.3;
        let mut p = vec![0.0, 0.0, 0.0];
        for _ in 0..10 {
            let err: Vec<f64> = p.iter().zip(&target).map(|(&x, &t)| x - t).collect();
            let p_next: Vec<f64> = p.iter().zip(&err).map(|(&x, &e)| x - alpha * e).collect();
            tdiis.push(p_next.clone(), err);
            if let Some(extrap) = tdiis.extrapolate() {
                p = extrap;
            } else {
                p = p_next;
            }
        }

        for i in 0..3 {
            assert!((p[i] - target[i]).abs() < 1e-4);
        }
    }
}
