// SPDX-License-Identifier: Apache-2.0
//! Cartesian working integrals and their real spherical AO subspace.
//!
//! SCF coefficients have Cartesian rows and physical MO columns. Thus C^T h C
//! and the four-index ERI transformation implement the same AO transformation
//! without allocating a second fourth-power integral tensor. Nuclear derivatives
//! contract Cartesian densities with the existing Cartesian integral engine.

use crate::basis::{ContractedGaussian, double_factorial};
use crate::error::HfError;
use crate::matrix::Matrix;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BasisKind {
    Cartesian,
    Spherical,
}

impl BasisKind {
    pub fn default_for_name(name: &str) -> Self {
        let name = name.to_ascii_lowercase().replace(['-', '_', ' '], "");
        if name.starts_with("def2") || name.contains("ccpv") || name.contains("6311") {
            Self::Spherical
        } else {
            Self::Cartesian
        }
    }
}

#[derive(Clone, Debug)]
pub struct AoSpace {
    pub kind: BasisKind,
    /// Cartesian AO -> physical AO matrix, before overlap rank truncation.
    pub cart_to_ao: Matrix,
    pub n_retained: usize,
}

impl AoSpace {
    pub fn cartesian(n: usize) -> Self {
        Self {
            kind: BasisKind::Cartesian,
            cart_to_ao: Matrix::identity(n),
            n_retained: n,
        }
    }
    pub fn new(basis: &[ContractedGaussian], kind: BasisKind) -> Result<Self, HfError> {
        let transform = match kind {
            BasisKind::Cartesian => Matrix::identity(basis.len()),
            BasisKind::Spherical => spherical_transform(basis)?,
        };
        let n_retained = transform.cols;
        Ok(Self {
            kind,
            cart_to_ao: transform,
            n_retained,
        })
    }
    pub fn orthogonalizer(&mut self, overlap: &Matrix) -> Result<Matrix, HfError> {
        let s = self
            .cart_to_ao
            .transpose()
            .dot(overlap)
            .dot(&self.cart_to_ao);
        let x = s.canonical_orthogonalizer(1e-8)?;
        self.n_retained = x.cols;
        Ok(self.cart_to_ao.dot(&x))
    }
    pub fn ensure_differentiable(&self) -> Result<(), HfError> {
        if self.n_retained < self.cart_to_ao.cols {
            Err(HfError::LinearDependence {
                n_ao: self.cart_to_ao.cols,
                n_mo: self.n_retained,
            })
        } else {
            Ok(())
        }
    }
    /// Recover coefficients in the physical (Cartesian or spherical) AO basis.
    pub fn physical_coefficients(&self, cart_coefficients: &Matrix) -> Result<Matrix, HfError> {
        let t = &self.cart_to_ao;
        let tt = t.transpose();
        let gram = tt
            .dot(t)
            .inverse()
            .ok_or_else(|| HfError::Integral("singular AO transform".into()))?;
        Ok(gram.dot(&tt).dot(cart_coefficients))
    }
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

/// Homogeneous real solid harmonic, in individually normalized Cartesian GTOs.
/// m=-l..l uses sine(m<0), zonal(m=0), cosine(m>0), with positive p_x/p_y.
fn polynomial(l: usize, m: isize) -> std::collections::HashMap<[usize; 3], f64> {
    let abs_m = m.unsigned_abs();
    let mut out = std::collections::HashMap::new();
    let norm = (factorial(l - abs_m) / factorial(l + abs_m)).sqrt()
        * if abs_m == 0 { 1.0 } else { 2.0_f64.sqrt() };
    for k in 0..=(l - abs_m) / 2 {
        let coefficient = (-1.0_f64).powi(k as i32) * factorial(2 * l - 2 * k)
            / (2.0_f64.powi(l as i32)
                * factorial(k)
                * factorial(l - k)
                * factorial(l - abs_m - 2 * k));
        for y in 0..=abs_m {
            let phase = match (m < 0, y % 4) {
                (true, 1) | (false, 0) => 1.0,
                (true, 3) | (false, 2) => -1.0,
                _ => 0.0,
            };
            if phase == 0.0 {
                continue;
            }
            let binomial = factorial(abs_m) / (factorial(y) * factorial(abs_m - y));
            for xx in 0..=k {
                for yy in 0..=k - xx {
                    let zz = k - xx - yy;
                    let powers = [abs_m - y + 2 * xx, y + 2 * yy, l - abs_m - 2 * k + 2 * zz];
                    let multinomial =
                        factorial(k) / (factorial(xx) * factorial(yy) * factorial(zz));
                    *out.entry(powers).or_insert(0.0) +=
                        norm * coefficient * binomial * phase * multinomial;
                }
            }
        }
    }
    out
}

/// Consecutive complete Cartesian shells, in the ordering of `build_basis`.
pub fn shell_ranges(basis: &[ContractedGaussian]) -> Result<Vec<std::ops::Range<usize>>, HfError> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < basis.len() {
        let a = &basis[start];
        let l = a.total_angular_momentum();
        let count = (l + 1) * (l + 2) / 2;
        if l > 7 || start + count > basis.len() {
            return Err(HfError::InvalidInput("incomplete Cartesian shell".into()));
        }
        let mut seen = std::collections::HashSet::new();
        for b in &basis[start..start + count] {
            if b.atom_index != a.atom_index
                || b.center != a.center
                || b.total_angular_momentum() != l
                || b.primitives.len() != a.primitives.len()
                || b.primitives
                    .iter()
                    .zip(&a.primitives)
                    .any(|(x, y)| x.exponent != y.exponent)
                || !seen.insert(b.l)
            {
                return Err(HfError::InvalidInput(
                    "noncontiguous or incomplete Cartesian shell".into(),
                ));
            }
        }
        ranges.push(start..start + count);
        start += count;
    }
    Ok(ranges)
}

/// Real spherical AOs: s; p_x,p_y,p_z; then m=-l..l for l>=2 (PySCF order).
pub fn spherical_transform(basis: &[ContractedGaussian]) -> Result<Matrix, HfError> {
    let shells = shell_ranges(basis)?;
    let n_sph = shells
        .iter()
        .map(|r| 2 * basis[r.start].total_angular_momentum() + 1)
        .sum();
    let mut t = Matrix::zeros(basis.len(), n_sph);
    let mut col = 0;
    for range in shells {
        let l = basis[range.start].total_angular_momentum();
        if l <= 1 {
            for row in range {
                t[(row, col)] = 1.0;
                col += 1;
            }
            continue;
        }
        let mut overlap = Matrix::zeros(range.len(), range.len());
        for (i, a) in basis[range.clone()].iter().enumerate() {
            for (j, b) in basis[range.clone()].iter().enumerate() {
                overlap[(i, j)] = crate::integrals::contracted_overlap(a, b);
            }
        }
        for m in -(l as isize)..=l as isize {
            let p = polynomial(l, m);
            let mut coefficients = Vec::with_capacity(range.len());
            for cg in &basis[range.clone()] {
                let weight =
                    cg.l.iter()
                        .map(|&n| double_factorial(2 * n as i64 - 1))
                        .product::<f64>()
                        .sqrt();
                coefficients.push(p.get(&cg.l).copied().unwrap_or(0.0) * weight);
            }
            let mut norm = 0.0;
            for i in 0..range.len() {
                for j in 0..range.len() {
                    norm += coefficients[i] * overlap[(i, j)] * coefficients[j];
                }
            }
            if norm <= 0.0 || !norm.is_finite() {
                return Err(HfError::Integral("invalid spherical normalization".into()));
            }
            for (row, &v) in range.clone().zip(&coefficients) {
                t[(row, col)] = v / norm.sqrt();
            }
            col += 1;
        }
    }
    Ok(t)
}
