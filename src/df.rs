// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Common Coulomb-metric density fitting with rectangular metric rank truncation.
use crate::basis::{ContractedGaussian, Primitive};
use crate::error::HfError;
use crate::integrals::contracted_eri;
use crate::matrix::Matrix;
use crate::scf::compound_index;
use rayon::prelude::*;

/// Coulomb whitening in a specified physical auxiliary space. The returned
/// coefficients have Cartesian rows, including for a pure spherical fit.
pub(crate) fn metric_orthogonalizer(
    auxiliary: &[ContractedGaussian],
    kind: crate::ao::BasisKind,
    cutoff: f64,
) -> Result<Matrix, HfError> {
    let metric = coulomb_metric(auxiliary);
    if kind == crate::ao::BasisKind::Cartesian {
        return metric.canonical_orthogonalizer(cutoff);
    }
    let transform = crate::ao::AoSpace::new(auxiliary, kind)?.cart_to_ao;
    let physical = transform.transpose().dot(&metric).dot(&transform);
    Ok(transform.dot(&physical.canonical_orthogonalizer(cutoff)?))
}

pub(crate) fn constant(center: [f64; 3]) -> ContractedGaussian {
    ContractedGaussian {
        atom_index: 0,
        center,
        l: [0; 3],
        primitives: vec![Primitive {
            exponent: 0.0,
            coefficient: 1.0,
        }],
    }
}

/// Two-center Coulomb integrals (P|Q), with normalized auxiliary functions.
pub fn coulomb_metric(aux: &[ContractedGaussian]) -> Matrix {
    let n = aux.len();
    let mut metric = Matrix::zeros(n, n);
    metric
        .data
        .par_chunks_mut(n.max(1))
        .enumerate()
        .for_each(|(p, row)| {
            let one_p = constant(aux[p].center);
            for (q, value) in row.iter_mut().enumerate() {
                *value = contracted_eri(&aux[p], &one_p, &aux[q], &constant(aux[q].center));
            }
        });
    metric
}

/// Three-center Coulomb integrals (P|mu nu), auxiliary rows, packed AO-pair columns.
pub fn three_center_integrals(basis: &[ContractedGaussian], aux: &[ContractedGaussian]) -> Matrix {
    let n = basis.len();
    let np = n * (n + 1) / 2;
    let mut integrals = Matrix::zeros(aux.len(), np);
    integrals
        .data
        .par_chunks_mut(np.max(1))
        .enumerate()
        .for_each(|(p, row)| {
            let one = constant(aux[p].center);
            for mu in 0..n {
                for nu in 0..=mu {
                    row[compound_index(mu, nu)] =
                        contracted_eri(&basis[mu], &basis[nu], &aux[p], &one);
                }
            }
        });
    integrals
}

#[derive(Clone, Debug)]
pub struct DfTensor {
    pub n_ao: usize,
    pub n_aux_original: usize,
    /// B[retained auxiliary mode, packed pair]. (pq|rs)_RI = sum_Q B_Qpq B_Qrs.
    pub b: Matrix,
}

impl DfTensor {
    pub fn build(
        basis: &[ContractedGaussian],
        aux: &[ContractedGaussian],
        metric_cutoff: f64,
    ) -> Result<Self, HfError> {
        Self::build_in_space(basis, aux, crate::ao::BasisKind::Cartesian, metric_cutoff)
    }
    /// Restrict the fitting metric to the selected physical auxiliary functions.
    pub fn build_in_space(
        basis: &[ContractedGaussian],
        aux: &[ContractedGaussian],
        kind: crate::ao::BasisKind,
        metric_cutoff: f64,
    ) -> Result<Self, HfError> {
        if basis.is_empty() || aux.is_empty() {
            return Err(HfError::InvalidInput(
                "DF requires orbital and auxiliary bases".into(),
            ));
        }
        let np = (basis.len() as u128) * (basis.len() as u128 + 1) / 2;
        let na = aux.len() as u128;
        if !metric_cutoff.is_finite() || metric_cutoff <= 0. {
            return Err(HfError::InvalidInput(
                "DF metric cutoff must be finite and positive".into(),
            ));
        }
        if 8 * (2 * na * np + 10 * na * na) > 256 * 1024 * 1024 {
            return Err(HfError::Integral(
                "estimated DF metric/three-center workspace exceeds 256 MiB".into(),
            ));
        }
        let x = metric_orthogonalizer(aux, kind, metric_cutoff)?;
        let b = x.transpose().dot(&three_center_integrals(basis, aux));
        Ok(Self {
            n_ao: basis.len(),
            n_aux_original: aux.len(),
            b,
        })
    }
    pub fn rank(&self) -> usize {
        self.b.rows
    }
    /// J and K for the spin-summed RHF density P=2 C_occ C_occ^T.
    /// Exchange uses B_Q C_occ and never forms four-index fitted ERIs.
    pub fn coulomb_exchange_rhf(&self, occupied: &Matrix) -> Result<(Matrix, Matrix), HfError> {
        if occupied.rows != self.n_ao
            || occupied.cols == 0
            || occupied.data.iter().any(|x| !x.is_finite())
        {
            return Err(HfError::InvalidInput(
                "invalid occupied coefficients for DF J/K".into(),
            ));
        }
        let n = self.n_ao;
        Ok((0..self.rank())
            .into_par_iter()
            .map(|k| {
                let mut b = Matrix::zeros(n, n);
                for p in 0..n {
                    for q in 0..n {
                        b[(p, q)] = self.b[(k, compound_index(p, q))];
                    }
                }
                let action = b.dot(occupied);
                let charge: f64 = 2.
                    * action
                        .data
                        .iter()
                        .zip(&occupied.data)
                        .map(|(x, c)| x * c)
                        .sum::<f64>();
                (b.scale(charge), action.dot(&action.transpose()).scale(2.))
            })
            .reduce(
                || (Matrix::zeros(n, n), Matrix::zeros(n, n)),
                |(mut j, mut k), (dj, dk)| {
                    for (a, b) in j.data.iter_mut().zip(dj.data) {
                        *a += b;
                    }
                    for (a, b) in k.data.iter_mut().zip(dk.data) {
                        *a += b;
                    }
                    (j, k)
                },
            ))
    }
    pub fn eri(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let pq = compound_index(p, q);
        let rs = compound_index(r, s);
        (0..self.rank())
            .map(|k| self.b[(k, pq)] * self.b[(k, rs)])
            .sum()
    }
    pub fn transform(&self, c: &Matrix) -> Result<Self, HfError> {
        if c.rows != self.n_ao {
            return Err(HfError::InvalidInput(
                "DF MO coefficient dimension mismatch".into(),
            ));
        }
        let np = c.cols * (c.cols + 1) / 2;
        let mut b = Matrix::zeros(self.rank(), np);
        b.data
            .par_chunks_mut(np.max(1))
            .enumerate()
            .for_each(|(k, row)| {
                let mut matrix = Matrix::zeros(c.rows, c.rows);
                for i in 0..c.rows {
                    for j in 0..c.rows {
                        matrix[(i, j)] = self.b[(k, compound_index(i, j))];
                    }
                }
                let mo = c.transpose().dot(&matrix).dot(c);
                for p in 0..c.cols {
                    for q in 0..=p {
                        row[compound_index(p, q)] = mo[(p, q)];
                    }
                }
            });
        Ok(Self {
            n_ao: c.cols,
            n_aux_original: self.n_aux_original,
            b,
        })
    }
    /// Closed-shell RI-MP2 correlation energy in the supplied canonical MO basis.
    pub fn mp2_energy(&self, eps: &[f64], n_occ: usize, n_frozen: usize) -> Result<f64, HfError> {
        if eps.len() != self.n_ao || n_frozen > n_occ || n_occ > eps.len() {
            return Err(HfError::InvalidInput(
                "invalid RI-MP2 orbital partition".into(),
            ));
        }
        let mut energy = 0.0;
        for i in n_frozen..n_occ {
            for j in n_frozen..n_occ {
                for a in n_occ..eps.len() {
                    for b in n_occ..eps.len() {
                        let gap = eps[i] + eps[j] - eps[a] - eps[b];
                        if gap.abs() < 1e-12 {
                            return Err(HfError::Response("singular RI-MP2 denominator".into()));
                        }
                        let v = self.eri(i, a, j, b);
                        energy += v * (2.0 * v - self.eri(i, b, j, a)) / gap;
                    }
                }
            }
        }
        Ok(energy)
    }
}
