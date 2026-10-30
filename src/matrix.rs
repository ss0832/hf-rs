// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Pure Rust Matrix library with Jacobi eigenvalue decomposition for symmetric matrices.

use std::fmt;
use faer::linalg::solvers::DenseSolveCore;

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len());
        Matrix { rows, cols, data }
    }

    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline(always)]
    pub fn set(&mut self, r: usize, c: usize, val: f64) {
        self.data[r * self.cols + c] = val;
    }

    pub fn transpose(&self) -> Self {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out[(c, r)] = self[(r, c)];
            }
        }
        out
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Matrix dimension mismatch in dot product");
        if self.rows >= 16 && self.cols >= 16 && other.cols >= 16 {
            return self.dot_faer(other);
        }
        let mut out = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let aik = self[(i, k)];
                if aik == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    out[(i, j)] += aik * other[(k, j)];
                }
            }
        }
        out
    }

    /// Parallel matrix multiplication using faer BLAS gemm and Rayon.
    pub fn dot_faer(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Matrix dimension mismatch in dot product");
        let a = faer::MatRef::from_row_major_slice(&self.data, self.rows, self.cols);
        let b = faer::MatRef::from_row_major_slice(&other.data, other.rows, other.cols);
        let c = a * b;

        let mut out = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                out[(i, j)] = c[(i, j)];
            }
        }
        out
    }

    /// Returns the first `k` columns as an `rows x k` matrix.
    pub fn leading_columns(&self, k: usize) -> Matrix {
        assert!(k <= self.cols);
        let mut out = Matrix::zeros(self.rows, k);
        for i in 0..self.rows {
            out.data[i * k..(i + 1) * k].copy_from_slice(&self.data[i * self.cols..i * self.cols + k]);
        }
        out
    }

    /// Density-like product `scale * C_occ C_occ^T` built from the first `n_occ` columns of `self`.
    pub fn occupied_density(&self, n_occ: usize, scale: f64) -> Matrix {
        let c_occ = self.leading_columns(n_occ);
        c_occ.dot(&c_occ.transpose()).scale(scale)
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut out = self.clone();
        for (a, b) in out.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
        out
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut out = self.clone();
        for (a, b) in out.data.iter_mut().zip(other.data.iter()) {
            *a -= *b;
        }
        out
    }

    pub fn scale(&self, s: f64) -> Matrix {
        let mut out = self.clone();
        for a in out.data.iter_mut() {
            *a *= s;
        }
        out
    }

    pub fn trace(&self) -> f64 {
        assert_eq!(self.rows, self.cols);
        let mut tr = 0.0;
        for i in 0..self.rows {
            tr += self[(i, i)];
        }
        tr
    }

    /// Computes the Frobenius norm of (self - other)
    pub fn rmsd(&self, other: &Matrix) -> f64 {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut sum = 0.0;
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            let diff = a - b;
            sum += diff * diff;
        }
        (sum / (self.data.len() as f64)).sqrt()
    }

    /// High-performance symmetric eigendecomposition using faer divide & conquer / QR.
    pub fn eigh_faer(&self) -> Option<(Vec<f64>, Matrix)> {
        assert_eq!(self.rows, self.cols, "Matrix must be square");
        let n = self.rows;
        let m_ref = faer::MatRef::from_row_major_slice(&self.data, n, n);
        let evd = m_ref.self_adjoint_eigen(faer::Side::Lower).ok()?;
        let s = evd.S();
        let u = evd.U();

        let mut eig_pairs: Vec<(f64, usize)> = (0..n).map(|i| (s[i], i)).collect();
        eig_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut sorted_vals = Vec::with_capacity(n);
        let mut sorted_v = Matrix::zeros(n, n);
        for (new_col, (val, old_col)) in eig_pairs.into_iter().enumerate() {
            sorted_vals.push(val);
            let mut max_abs = 0.0;
            let mut sign = 1.0;
            for row in 0..n {
                let component = u[(row, old_col)];
                if component.abs() > max_abs {
                    max_abs = component.abs();
                    sign = if component >= 0.0 { 1.0 } else { -1.0 };
                }
            }
            for row in 0..n {
                sorted_v[(row, new_col)] = sign * u[(row, old_col)];
            }
        }
        Some((sorted_vals, sorted_v))
    }

    /// Classical Jacobi rotation method for symmetric matrix diagonalization.
    pub fn eigh_jacobi(&self) -> (Vec<f64>, Matrix) {
        assert_eq!(self.rows, self.cols, "Matrix must be square");
        let n = self.rows;
        let mut a = self.clone();
        let mut v = Matrix::identity(n);

        let max_sweeps = 60;
        let scale = a.data.iter().map(|x| x.abs()).fold(1.0_f64, f64::max);
        let tol = 1e-15 * scale;

        for sweep in 0..max_sweeps {
            let mut off_diag_sum = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let val = a[(i, j)];
                    off_diag_sum += val * val;
                }
            }

            // off_diag_sum is a squared norm. Comparing it directly with tol
            // stopped at ~3e-8 and prevented tightly converged small-system SCF.
            if off_diag_sum < tol * tol {
                break;
            }

            let threshold = if sweep < 4 {
                0.2 * off_diag_sum.sqrt() / (n as f64)
            } else {
                0.0
            };

            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = a[(p, q)];
                    if apq.abs() <= threshold {
                        continue;
                    }

                    let app = a[(p, p)];
                    let aqq = a[(q, q)];

                    let theta = 0.5 * (aqq - app) / apq;
                    let t = if theta >= 0.0 {
                        1.0 / (theta + (theta * theta + 1.0).sqrt())
                    } else {
                        -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                    };
                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;

                    for i in 0..n {
                        if i != p && i != q {
                            let a_ip = a[(i, p)];
                            let a_iq = a[(i, q)];
                            a[(i, p)] = c * a_ip - s * a_iq;
                            a[(p, i)] = a[(i, p)];
                            a[(i, q)] = s * a_ip + c * a_iq;
                            a[(q, i)] = a[(i, q)];
                        }
                    }
                    a[(p, p)] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                    a[(q, q)] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                    a[(p, q)] = 0.0;
                    a[(q, p)] = 0.0;

                    for i in 0..n {
                        let v_ip = v[(i, p)];
                        let v_iq = v[(i, q)];
                        v[(i, p)] = c * v_ip - s * v_iq;
                        v[(i, q)] = s * v_ip + c * v_iq;
                    }
                }
            }
        }

        // Extract eigenvalues and sort in ascending order
        let mut eig_pairs: Vec<(f64, Vec<f64>)> = (0..n)
            .map(|col| {
                let val = a[(col, col)];
                let vec: Vec<f64> = (0..n).map(|row| v[(row, col)]).collect();
                (val, vec)
            })
            .collect();

        eig_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut sorted_vals = Vec::with_capacity(n);
        let mut sorted_v = Matrix::zeros(n, n);
        for (col, (val, vec)) in eig_pairs.into_iter().enumerate() {
            sorted_vals.push(val);
            // Conventional phase alignment: ensure largest component is positive
            let mut max_abs = 0.0;
            let mut sign = 1.0;
            for row in 0..n {
                let component = vec[row];
                if component.abs() > max_abs {
                    max_abs = component.abs();
                    sign = if component >= 0.0 { 1.0 } else { -1.0 };
                }
            }
            for row in 0..n {
                sorted_v[(row, col)] = sign * vec[row];
            }
        }

        (sorted_vals, sorted_v)
    }

    /// Symmetric matrix diagonalization.
    /// Uses faer divide & conquer / QR for N >= 8, with classical Jacobi rotation for small matrices or fallback.
    /// Returns (eigenvalues in ascending order, eigenvector matrix whose columns are eigenvectors).
    pub fn eigh(&self) -> (Vec<f64>, Matrix) {
        assert_eq!(self.rows, self.cols, "Matrix must be square");
        let n = self.rows;
        if n >= 8 {
            return self.eigh_faer().unwrap_or_else(|| self.eigh_jacobi());
        }
        self.eigh_jacobi()
    }

    /// Computes S^(-1/2) for symmetric positive-definite matrix S.
    pub fn inv_sqrt(&self) -> Matrix {
        let (vals, u) = self.eigh();
        let n = self.rows;
        let mut d_inv_sqrt = Matrix::zeros(n, n);
        for i in 0..n {
            assert!(vals[i] > 1e-12, "Matrix is near-singular in inv_sqrt: eigenvalue = {}", vals[i]);
            d_inv_sqrt[(i, i)] = 1.0 / vals[i].sqrt();
        }
        // X = U * d^(-1/2) * U^T
        u.dot(&d_inv_sqrt).dot(&u.transpose())
    }

    /// Rectangular canonical orthogonalizer, dropping eigenvalues below `cutoff`.
    /// The columns of X satisfy X^T S X = I. No inverse of discarded modes is formed.
    pub fn canonical_orthogonalizer(&self, cutoff: f64) -> Result<Matrix, crate::error::HfError> {
        use crate::error::HfError;
        if self.rows != self.cols || self.rows == 0 || !cutoff.is_finite() || cutoff <= 0.0
            || self.data.iter().any(|x| !x.is_finite()) {
            return Err(HfError::InvalidInput("invalid overlap matrix or cutoff".into()));
        }
        let (values, vectors) = self.eigh();
        if values[0] < -cutoff {
            return Err(HfError::Integral(format!("overlap has negative eigenvalue {}", values[0])));
        }
        let retained = values.iter().enumerate().filter(|&(_, &v)| v >= cutoff).collect::<Vec<_>>();
        if retained.is_empty() { return Err(HfError::Integral("overlap has zero retained rank".into())); }
        let mut x = Matrix::zeros(self.rows, retained.len());
        for (col, &(old, &value)) in retained.iter().enumerate() {
            for row in 0..self.rows { x[(row, col)] = vectors[(row, old)] / value.sqrt(); }
        }
        Ok(x)
    }

    /// Matrix exponential by scaling and squaring with a converged Taylor series.
    /// In particular, a real antisymmetric generator produces an orthogonal rotation.
    pub fn exp(&self) -> Matrix {
        assert_eq!(self.rows, self.cols);
        let norm = (0..self.cols).map(|j| (0..self.rows).map(|i| self[(i,j)].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max);
        assert!(norm.is_finite(), "matrix exponential requires finite entries");
        let squarings = if norm <= 0.5 { 0 } else { (norm / 0.5).log2().ceil() as u32 };
        let a = self.scale(2.0_f64.powi(-(squarings as i32)));
        let mut sum = Matrix::identity(self.rows);
        let mut term = sum.clone();
        for k in 1..=48 {
            term = term.dot(&a).scale(1.0 / k as f64);
            sum = sum.add(&term);
            if term.data.iter().all(|v| v.abs() <= 1e-17) { break; }
        }
        for _ in 0..squarings { sum = sum.dot(&sum); }
        sum
    }

    /// Performs LU factorization with partial row pivoting: P * A = L * U.
    /// Returns None if the matrix is singular or non-square.
    pub fn lu_decompose(&self) -> Option<LUFactorization> {
        if self.rows != self.cols || self.rows == 0 {
            return None;
        }
        let n = self.rows;
        let mut lu = self.clone();
        let mut pivots = vec![0; n];

        for i in 0..n {
            // Find pivot row
            let mut max_val = lu[(i, i)].abs();
            let mut max_row = i;
            for k in (i + 1)..n {
                let val = lu[(k, i)].abs();
                if val > max_val {
                    max_val = val;
                    max_row = k;
                }
            }

            if max_val < 1e-15 {
                return None;
            }

            pivots[i] = max_row;

            if max_row != i {
                for j in 0..n {
                    let tmp = lu[(i, j)];
                    lu[(i, j)] = lu[(max_row, j)];
                    lu[(max_row, j)] = tmp;
                }
            }

            let pivot = lu[(i, i)];
            for k in (i + 1)..n {
                let factor = lu[(k, i)] / pivot;
                lu[(k, i)] = factor;
                for j in (i + 1)..n {
                    let val = lu[(i, j)];
                    lu[(k, j)] -= factor * val;
                }
            }
        }

        Some(LUFactorization { n, lu, pivots })
    }

    /// Computes the exact matrix inverse A^(-1) via faer partial pivoting LU for N >= 16 or LU solve fallback.
    /// Returns None if the matrix is singular.
    pub fn inverse(&self) -> Option<Matrix> {
        if self.rows != self.cols || self.rows == 0 {
            return None;
        }
        let n = self.rows;
        if n >= 16 {
            let m_ref = faer::MatRef::from_row_major_slice(&self.data, n, n);
            let lu = m_ref.partial_piv_lu();
            let inv_mat = lu.inverse();
            let mut out = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    out[(i, j)] = inv_mat[(i, j)];
                }
            }
            return Some(out);
        }
        let lu = self.lu_decompose()?;
        let mut inv = Matrix::zeros(n, n);
        let mut e = vec![0.0; n];
        for j in 0..n {
            e[j] = 1.0;
            let col = lu.solve(&e);
            for i in 0..n {
                inv[(i, j)] = col[i];
            }
            e[j] = 0.0;
        }
        Some(inv)
    }
}

/// Stores LU decomposition of a square matrix with row permutation pivots.
#[derive(Clone, Debug)]
pub struct LUFactorization {
    pub n: usize,
    pub lu: Matrix,
    pub pivots: Vec<usize>,
}

impl LUFactorization {
    /// Determinant of the factorized matrix: product of the U diagonal times the pivot sign.
    pub fn determinant(&self) -> f64 {
        let mut det = 1.0;
        for i in 0..self.n {
            det *= self.lu[(i, i)];
            if self.pivots[i] != i {
                det = -det;
            }
        }
        det
    }

    /// Solves A * x = b in O(n^2) operations using forward and back substitution.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        assert_eq!(b.len(), n, "Vector length mismatch in LU solve");
        let mut x = b.to_vec();

        // Apply row permutation
        for i in 0..n {
            let p = self.pivots[i];
            if p != i {
                x.swap(i, p);
            }
        }

        // Forward substitution: L * y = b' (L has unit diagonal)
        for i in 0..n {
            for j in 0..i {
                x[i] -= self.lu[(i, j)] * x[j];
            }
        }

        // Back substitution: U * x = y
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= self.lu[(i, j)] * x[j];
            }
            x[i] /= self.lu[(i, i)];
        }

        x
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;
    #[inline(always)]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.cols + index.1]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    #[inline(always)]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.cols + index.1]
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:12.6} ", self[(i, j)])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobi_eigh() {
        // Simple 2x2 symmetric matrix
        // [ 2  1 ]
        // [ 1  2 ]
        // eigenvalues: 1, 3
        let m = Matrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (vals, v) = m.eigh();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);

        // V * diag(vals) * V^T should equal m
        let mut d = Matrix::zeros(2, 2);
        d[(0, 0)] = vals[0];
        d[(1, 1)] = vals[1];
        let reconstructed = v.dot(&d).dot(&v.transpose());
        assert!(m.rmsd(&reconstructed) < 1e-10);
    }

    #[test]
    fn test_lu_decomposition() {
        // [ 2  1 ] [ x0 ]   [ 4  ]
        // [ 4  5 ] [ x1 ] = [ 11 ]
        // Solution: x0 = 1.5, x1 = 1.0
        let a = Matrix::from_vec(2, 2, vec![2.0, 1.0, 4.0, 5.0]);
        let b = vec![4.0, 11.0];
        let lu = a.lu_decompose().expect("LU decomposition failed");
        let x = lu.solve(&b);
        assert!((x[0] - 1.5).abs() < 1e-12);
        assert!((x[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_dot_faer() {
        let a = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::from_vec(2, 3, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c_standard = a.dot(&b);
        let c_faer = a.dot_faer(&b);
        assert!(c_standard.rmsd(&c_faer) < 1e-14);

        // Test symmetric eigendecomposition
        let sym = Matrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (vals, u) = sym.eigh();
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
        let mut d = Matrix::zeros(2, 2);
        d[(0, 0)] = vals[0];
        d[(1, 1)] = vals[1];
        let reconstructed = u.dot(&d).dot(&u.transpose());
        assert!(sym.rmsd(&reconstructed) < 1e-10);

        // Test inverse
        let inv = sym.inverse().expect("Inverse failed");
        let ident = sym.dot(&inv);
        assert!(ident.rmsd(&Matrix::identity(2)) < 1e-12);
    }
}
