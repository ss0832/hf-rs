// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Small split-complex matrix operations for the complex HF stationary solver.

use crate::Matrix;

#[derive(Clone)]
pub(super) struct ComplexMatrix {
    pub re: Matrix,
    pub im: Matrix,
}
impl ComplexMatrix {
    pub fn real(re: Matrix) -> Self {
        let im = Matrix::zeros(re.rows, re.cols);
        Self { re, im }
    }
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::real(Matrix::zeros(rows, cols))
    }
    pub fn dagger(&self) -> Self {
        Self {
            re: self.re.transpose(),
            im: self.im.transpose().scale(-1.),
        }
    }
    pub fn dot(&self, b: &Self) -> Self {
        Self {
            re: self.re.dot(&b.re).sub(&self.im.dot(&b.im)),
            im: self.re.dot(&b.im).add(&self.im.dot(&b.re)),
        }
    }
    pub fn add(&self, b: &Self, scale: f64) -> Self {
        Self {
            re: self.re.add(&b.re.scale(scale)),
            im: self.im.add(&b.im.scale(scale)),
        }
    }
    pub fn norm(&self) -> f64 {
        self.re
            .data
            .iter()
            .chain(&self.im.data)
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt()
    }
    // An opaque real vector layout for one common complex DIIS coefficient set.
    pub fn pack(&self) -> Matrix {
        Matrix::from_vec(
            self.re.rows,
            2 * self.re.cols,
            self.re.data.iter().chain(&self.im.data).copied().collect(),
        )
    }
    pub fn unpack(p: Matrix) -> Self {
        let n = p.data.len() / 2;
        Self {
            re: Matrix::from_vec(p.rows, p.cols / 2, p.data[..n].to_vec()),
            im: Matrix::from_vec(p.rows, p.cols / 2, p.data[n..].to_vec()),
        }
    }
    pub fn eigh(&self) -> Result<(Vec<f64>, Self), String> {
        let n = self.re.rows;
        let a = faer::Mat::from_fn(n, n, |i, j| {
            faer::c64::new(self.re[(i, j)], self.im[(i, j)])
        });
        let evd = a
            .self_adjoint_eigen(faer::Side::Lower)
            .map_err(|e| format!("complex Hermitian eigensolver: {e:?}"))?;
        let mut pairs: Vec<_> = (0..n).map(|i| (evd.S()[i].re, i)).collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut c = Self::zeros(n, n);
        for (q, &(_, old)) in pairs.iter().enumerate() {
            for p in 0..n {
                c.re[(p, q)] = evd.U()[(p, old)].re;
                c.im[(p, q)] = evd.U()[(p, old)].im;
            }
        }
        Ok((pairs.iter().map(|p| p.0).collect(), c))
    }
    pub fn density(&self, occupied: usize) -> Self {
        let c = Self {
            re: self.re.leading_columns(occupied),
            im: self.im.leading_columns(occupied),
        };
        c.dot(&c.dagger())
    }
}

pub(super) fn spin_diagonal(a: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(2 * a.rows, 2 * a.cols);
    for s in 0..2 {
        for p in 0..a.rows {
            for q in 0..a.cols {
                out[(s * a.rows + p, s * a.cols + q)] = a[(p, q)];
            }
        }
    }
    out
}
