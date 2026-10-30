// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Robust Coulomb-metric fitting of multiplicative F12 kernels.
//! Knizia (2010), Eqs. 2.8--2.10: D^T W3 + W3^T D - D^T W2 D.
//! Whitened auxiliary modes avoid an explicit inverse and four-index AO storage.

use super::operators::{GaussianGeminal, GeminalOperator, PreparedGeminalOperator};
use crate::{
    basis::ContractedGaussian, df::DfTensor, error::HfError, matrix::Matrix, scf::compound_index,
};
use rayon::prelude::*;
use std::{ops::Range, sync::Arc};

#[derive(Clone, Copy, Debug)]
pub struct F12DensityFitOptions {
    pub metric_cutoff: f64,
    /// Physical fitting functions; integral storage always uses Cartesian rows.
    pub auxiliary_basis_kind: crate::ao::BasisKind,
    /// Bound on estimated live fitting or transformed-pair workspaces.
    pub workspace_budget_bytes: usize,
}
impl Default for F12DensityFitOptions {
    fn default() -> Self {
        Self {
            metric_cutoff: 1e-10,
            auxiliary_basis_kind: crate::ao::BasisKind::Cartesian,
            workspace_budget_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Coulomb and spin-summed exchange for P=2 C_occ C_occ^T. The retained
/// three-center modes have only an occupied right leg. J is streamed from
/// the fitted charge, avoiding an auxiliary-by-full-AO-pair array.
pub fn fitted_coulomb_exchange_rhf(
    basis: &[ContractedGaussian],
    auxiliary: &[ContractedGaussian],
    occupied: &Matrix,
    options: F12DensityFitOptions,
) -> Result<(Matrix, Matrix), HfError> {
    let n = basis.len();
    if occupied.cols == 0 {
        return Err(HfError::InvalidInput(
            "DF F12 Fock requires occupied orbitals".into(),
        ));
    }
    // Reserve the two output matrices and the identity before constructing the
    // occupied-leg fitter, which performs its own angular/parallel estimate.
    let matrix_bytes = 8 * 3 * (n as u128).pow(2);
    let remaining = (options.workspace_budget_bytes as u128)
        .checked_sub(matrix_bytes)
        .filter(|&bytes| bytes > 0)
        .ok_or_else(|| HfError::Integral("DF F12 Fock workspace exceeds budget".into()))?;
    let fitter = F12PairFitter::new(
        basis,
        auxiliary,
        &Matrix::identity(n),
        occupied,
        F12DensityFitOptions {
            workspace_budget_bytes: remaining as usize,
            ..options
        },
    )?;
    let no = occupied.cols;
    let mut charge = Matrix::zeros(fitter.rank(), 1);
    for a in 0..fitter.rank() {
        charge[(a, 0)] = 2.0
            * (0..n)
                .flat_map(|mu| (0..no).map(move |i| (mu, i)))
                .map(|(mu, i)| fitter.b[(a, mu * no + i)] * occupied[(mu, i)])
                .sum::<f64>();
    }
    let charge = fitter.x.dot(&charge);
    let units: Vec<_> = auxiliary
        .iter()
        .map(|g| crate::df::constant(g.center))
        .collect();
    let mut coulomb = Matrix::zeros(n, n);
    let mut exchange = Matrix::zeros(n, n);
    coulomb
        .data
        .par_chunks_mut(n)
        .zip(exchange.data.par_chunks_mut(n))
        .enumerate()
        .try_for_each(|(mu, (jrow, krow))| -> Result<(), HfError> {
            for nu in 0..=mu {
                jrow[nu] = auxiliary
                    .iter()
                    .enumerate()
                    .map(|(p, g)| {
                        charge[(p, 0)]
                            * crate::integrals::contracted_eri(&basis[mu], &basis[nu], g, &units[p])
                    })
                    .sum();
                krow[nu] = 2.0
                    * (0..fitter.rank())
                        .flat_map(|a| (0..no).map(move |i| (a, i)))
                        .map(|(a, i)| fitter.b[(a, mu * no + i)] * fitter.b[(a, nu * no + i)])
                        .sum::<f64>();
                if !jrow[nu].is_finite() || !krow[nu].is_finite() {
                    return Err(HfError::Integral("nonfinite DF F12 J/K matrix".into()));
                }
            }
            Ok(())
        })?;
    for mu in 0..n {
        for nu in 0..mu {
            coulomb[(nu, mu)] = coulomb[(mu, nu)];
            exchange[(nu, mu)] = exchange[(mu, nu)];
        }
    }
    Ok((coulomb, exchange))
}

#[derive(Clone, Debug)]
pub struct FittedF12Kernel {
    coulomb: DfTensor,
    /// X^T W3 - 1/2 (X^T W2 X) B, with B=X^T (P|mu nu).
    corrected: Matrix,
    workspace_budget_bytes: usize,
}
impl FittedF12Kernel {
    pub fn build(
        basis: &[ContractedGaussian],
        auxiliary: &[ContractedGaussian],
        factor: &GaussianGeminal,
        operator: GeminalOperator,
        options: F12DensityFitOptions,
    ) -> Result<Self, HfError> {
        if matches!(operator, GeminalOperator::KineticCommutator) {
            return Err(HfError::Unsupported {
                method: "DF-F12 integral".into(),
                detail: "the kinetic single commutator is not a multiplicative density kernel"
                    .into(),
            });
        }
        if basis.is_empty()
            || auxiliary.is_empty()
            || options.workspace_budget_bytes == 0
            || !options.metric_cutoff.is_finite()
            || options.metric_cutoff <= 0.0
            || basis.iter().chain(auxiliary).any(|g| {
                g.primitives.is_empty()
                    || g.l.iter().sum::<usize>() > 7
                    || g.center.iter().any(|x| !x.is_finite())
                    || g.primitives.iter().any(|p| {
                        !p.exponent.is_finite() || p.exponent <= 0.0 || !p.coefficient.is_finite()
                    })
            })
        {
            return Err(HfError::InvalidInput(
                "invalid F12 density-fitting bases or controls".into(),
            ));
        }
        let n = basis.len() as u128;
        let np = n * (n + 1) / 2;
        let na = auxiliary.len() as u128;
        if 8 * (10 * na * na + 4 * na * np + 4 * np) > options.workspace_budget_bytes as u128 {
            return Err(HfError::Integral(
                "F12 density-fitting workspace estimate exceeds budget".into(),
            ));
        }
        let prepared = PreparedGeminalOperator::new(factor, operator)?;
        let x = crate::df::metric_orthogonalizer(
            auxiliary,
            options.auxiliary_basis_kind,
            options.metric_cutoff,
        )?;
        let xt = x.transpose();
        let b = xt.dot(&crate::df::three_center_integrals(basis, auxiliary));
        let np = np as usize;
        let mut three = Matrix::zeros(auxiliary.len(), np);
        three.data.par_chunks_mut(np).enumerate().try_for_each(
            |(p, row)| -> Result<(), HfError> {
                for mu in 0..basis.len() {
                    for nu in 0..=mu {
                        row[compound_index(mu, nu)] =
                            prepared.three_center(&basis[mu], &basis[nu], &auxiliary[p])?;
                    }
                }
                Ok(())
            },
        )?;
        let whitened_three = xt.dot(&three);
        drop(three);
        let mut two = Matrix::zeros(auxiliary.len(), auxiliary.len());
        two.data
            .par_chunks_mut(auxiliary.len())
            .enumerate()
            .try_for_each(|(p, row)| -> Result<(), HfError> {
                for (q, value) in row.iter_mut().enumerate() {
                    *value = prepared.two_center(&auxiliary[p], &auxiliary[q])?;
                }
                Ok(())
            })?;
        let kernel = xt.dot(&two).dot(&x);
        let corrected = whitened_three.sub(&kernel.dot(&b).scale(0.5));
        if b.data.iter().chain(&corrected.data).any(|x| !x.is_finite()) {
            return Err(HfError::Integral("nonfinite F12 fitted kernel".into()));
        }
        Ok(Self {
            coulomb: DfTensor {
                n_ao: basis.len(),
                n_aux_original: auxiliary.len(),
                b,
            },
            corrected,
            workspace_budget_bytes: options.workspace_budget_bytes,
        })
    }

    pub fn rank(&self) -> usize {
        self.coulomb.rank()
    }

    pub fn coulomb_tensor(&self) -> &DfTensor {
        &self.coulomb
    }

    pub fn integral(&self, p: usize, q: usize, r: usize, s: usize) -> Result<f64, HfError> {
        if [p, q, r, s].iter().any(|&i| i >= self.coulomb.n_ao) {
            return Err(HfError::InvalidInput(
                "F12 fitted integral index is out of bounds".into(),
            ));
        }
        let (pq, rs) = (compound_index(p, q), compound_index(r, s));
        let value = (0..self.rank())
            .map(|a| {
                self.coulomb.b[(a, pq)] * self.corrected[(a, rs)]
                    + self.corrected[(a, pq)] * self.coulomb.b[(a, rs)]
            })
            .sum::<f64>();
        if !value.is_finite() {
            return Err(HfError::Integral("nonfinite F12 fitted integral".into()));
        }
        Ok(value)
    }

    /// Ordered occupied pairs <p q|W|i j>, without a full fitted AO tensor.
    /// Both coefficient matrices use the fitted AO basis; they may be rectangular.
    pub fn mixed_pairs(&self, target: &Matrix, occupied: &Matrix) -> Result<Vec<Matrix>, HfError> {
        let (n, nr, no) = (self.coulomb.n_ao, target.cols, occupied.cols);
        if target.rows != n
            || occupied.rows != n
            || nr == 0
            || target.cols > n
            || occupied.cols > n
            || target.rows.checked_mul(target.cols) != Some(target.data.len())
            || occupied.rows.checked_mul(occupied.cols) != Some(occupied.data.len())
            || target
                .data
                .iter()
                .chain(&occupied.data)
                .any(|x| !x.is_finite())
        {
            return Err(HfError::InvalidInput(
                "invalid F12 fitted transformation coefficients".into(),
            ));
        }
        if no == 0 {
            return Ok(Vec::new());
        }
        let (na, nr128, no128) = (self.rank() as u128, nr as u128, no as u128);
        let permanent = 2 * na * (n as u128) * (n as u128 + 1) / 2;
        if 8 * (permanent
            + 4 * na * nr128 * no128
            + no128.pow(2) * nr128.pow(2)
            + 4 * (n as u128).pow(2))
            > self.workspace_budget_bytes as u128
        {
            return Err(HfError::Integral(
                "F12 fitted mixed-pair workspace estimate exceeds budget".into(),
            ));
        }
        let mut bs = vec![Matrix::zeros(nr, self.rank()); no];
        let mut ws = vec![Matrix::zeros(nr, self.rank()); no];
        let transpose = target.transpose();
        for mode in 0..self.rank() {
            for (input, output) in [(&self.coulomb.b, &mut bs), (&self.corrected, &mut ws)] {
                let mut ao = Matrix::zeros(n, n);
                for p in 0..n {
                    for q in 0..n {
                        ao[(p, q)] = input[(mode, compound_index(p, q))];
                    }
                }
                let block = transpose.dot(&ao).dot(occupied);
                for i in 0..no {
                    for p in 0..nr {
                        output[i][(p, mode)] = block[(p, i)];
                    }
                }
            }
        }
        let mut pairs = Vec::with_capacity(no * no);
        for i in 0..no {
            for j in 0..no {
                pairs.push(
                    bs[i]
                        .dot(&ws[j].transpose())
                        .add(&ws[i].dot(&bs[j].transpose())),
                );
            }
        }
        if pairs.iter().any(|m| m.data.iter().any(|x| !x.is_finite())) {
            return Err(HfError::Integral("nonfinite F12 fitted mixed pairs".into()));
        }
        Ok(pairs)
    }
}

/// Coulomb fitting with two coefficient legs contracted before storage.
/// No auxiliary-by-union-AO-pair array is built. The Coulomb modes are shared
/// across all multiplicative kernels made from the same fitting space.
pub struct F12PairFitter<'a> {
    basis: &'a [ContractedGaussian],
    auxiliary: &'a [ContractedGaussian],
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
    left_transpose: Matrix,
    right: Matrix,
    x: Matrix,
    b: Arc<Matrix>,
    n_target: usize,
    n_occupied: usize,
    budget: usize,
}

#[derive(Clone, Debug)]
pub struct FittedF12Pairs {
    b: Arc<Matrix>,
    corrected: Arc<Matrix>,
    scale: f64,
    n_target: usize,
    n_occupied: usize,
    budget: usize,
}

impl<'a> F12PairFitter<'a> {
    pub fn new(
        basis: &'a [ContractedGaussian],
        auxiliary: &'a [ContractedGaussian],
        target: &Matrix,
        occupied: &Matrix,
        options: F12DensityFitOptions,
    ) -> Result<Self, HfError> {
        let n = basis.len();
        if n == 0
            || auxiliary.is_empty()
            || target.rows != n
            || occupied.rows != n
            || target.cols == 0
            || target.cols > n
            || occupied.cols > n
            || target.rows.checked_mul(target.cols) != Some(target.data.len())
            || occupied.rows.checked_mul(occupied.cols) != Some(occupied.data.len())
            || target
                .data
                .iter()
                .chain(&occupied.data)
                .any(|x| !x.is_finite())
            || options.workspace_budget_bytes == 0
            || !options.metric_cutoff.is_finite()
            || options.metric_cutoff <= 0.0
            || basis.iter().chain(auxiliary).any(|g| {
                g.primitives.is_empty()
                    || g.l.iter().sum::<usize>() > 7
                    || g.center.iter().any(|x| !x.is_finite())
                    || g.primitives.iter().any(|p| {
                        !p.exponent.is_finite() || p.exponent <= 0.0 || !p.coefficient.is_finite()
                    })
            })
        {
            return Err(HfError::InvalidInput(
                "invalid occupied-leg F12 fitting space".into(),
            ));
        }
        let left_indices: Vec<_> = (0..n)
            .filter(|&p| (0..target.cols).any(|i| target[(p, i)] != 0.0))
            .collect();
        let right_indices: Vec<_> = (0..n)
            .filter(|&p| (0..occupied.cols).any(|i| occupied[(p, i)] != 0.0))
            .collect();
        let (na, np) = (
            auxiliary.len() as u128,
            target.cols as u128 * occupied.cols as u128,
        );
        let workers = auxiliary.len().min(rayon::current_num_threads()) as u128;
        let (nl, nr) = (left_indices.len() as u128, right_indices.len() as u128);
        let lmax = basis
            .iter()
            .map(|g| g.l.iter().sum::<usize>())
            .max()
            .unwrap_or(0);
        let amax = auxiliary
            .iter()
            .map(|g| g.l.iter().sum::<usize>())
            .max()
            .unwrap_or(0);
        let angular = (2 * lmax + amax + 1) as u128;
        let scratch = 16 * angular.pow(4) + nl * nr + target.cols as u128 * nr + np;
        let coefficients = target.cols as u128 * nl + occupied.cols as u128 * nr;
        if 8 * (10 * na * na + 6 * na * np + coefficients + workers * scratch)
            > options.workspace_budget_bytes as u128
        {
            return Err(HfError::Integral(
                "occupied-leg F12 fitting workspace estimate exceeds budget".into(),
            ));
        }
        let mut left_transpose = Matrix::zeros(target.cols, left_indices.len());
        let mut right = Matrix::zeros(right_indices.len(), occupied.cols);
        for (p, &mu) in left_indices.iter().enumerate() {
            for i in 0..target.cols {
                left_transpose[(i, p)] = target[(mu, i)];
            }
        }
        for (p, &mu) in right_indices.iter().enumerate() {
            for i in 0..occupied.cols {
                right[(p, i)] = occupied[(mu, i)];
            }
        }
        let x = crate::df::metric_orthogonalizer(
            auxiliary,
            options.auxiliary_basis_kind,
            options.metric_cutoff,
        )?;
        let mut result = Self {
            basis,
            auxiliary,
            left_indices,
            right_indices,
            left_transpose,
            right,
            x,
            b: Arc::new(Matrix::zeros(0, 0)),
            n_target: target.cols,
            n_occupied: occupied.cols,
            budget: options.workspace_budget_bytes,
        };
        let units: Vec<_> = auxiliary
            .iter()
            .map(|g| crate::df::constant(g.center))
            .collect();
        let raw = result.contract_legs(|p, mu, nu| {
            let value =
                crate::integrals::contracted_eri(&basis[mu], &basis[nu], &auxiliary[p], &units[p]);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(HfError::Integral(
                    "nonfinite occupied-leg Coulomb fitting integral".into(),
                ))
            }
        })?;
        result.b = Arc::new(result.x.transpose().dot(&raw));
        if result.b.data.iter().any(|x| !x.is_finite()) {
            return Err(HfError::Integral(
                "nonfinite occupied-leg Coulomb modes".into(),
            ));
        }
        Ok(result)
    }

    fn contract_legs(
        &self,
        integral: impl Fn(usize, usize, usize) -> Result<f64, HfError> + Sync,
    ) -> Result<Matrix, HfError> {
        let count = self.n_target * self.n_occupied;
        let mut raw = Matrix::zeros(self.auxiliary.len(), count);
        if count == 0 || self.left_indices.is_empty() || self.right_indices.is_empty() {
            return Ok(raw);
        }
        raw.data.par_chunks_mut(count).enumerate().try_for_each(
            |(p, row)| -> Result<(), HfError> {
                let mut block = Matrix::zeros(self.left_indices.len(), self.right_indices.len());
                for (a, &mu) in self.left_indices.iter().enumerate() {
                    for (b, &nu) in self.right_indices.iter().enumerate() {
                        block[(a, b)] = integral(p, mu, nu)?;
                    }
                }
                row.copy_from_slice(&self.left_transpose.dot(&block).dot(&self.right).data);
                Ok(())
            },
        )?;
        Ok(raw)
    }

    pub fn rank(&self) -> usize {
        self.b.rows
    }

    /// Coulomb modes B[A, p * n_occupied + i] in the retained metric space.
    pub fn coulomb_modes(&self) -> &Matrix {
        &self.b
    }

    /// The ordinary Coulomb fit uses B^T B and shares the same mode storage.
    pub fn coulomb_kernel(&self) -> FittedF12Pairs {
        FittedF12Pairs {
            b: Arc::clone(&self.b),
            corrected: Arc::clone(&self.b),
            scale: 0.5,
            n_target: self.n_target,
            n_occupied: self.n_occupied,
            budget: self.budget,
        }
    }

    pub fn kernel(
        &self,
        factor: &GaussianGeminal,
        operator: GeminalOperator,
    ) -> Result<FittedF12Pairs, HfError> {
        if matches!(operator, GeminalOperator::KineticCommutator) {
            return Err(HfError::Unsupported {
                method: "occupied-leg DF-F12 kernel".into(),
                detail: "the kinetic single commutator is not a multiplicative density kernel"
                    .into(),
            });
        }
        let prepared = PreparedGeminalOperator::new(factor, operator)?;
        let raw = self.contract_legs(|p, mu, nu| {
            prepared.three_center(&self.basis[mu], &self.basis[nu], &self.auxiliary[p])
        })?;
        let three = self.x.transpose().dot(&raw);
        drop(raw);
        let na = self.auxiliary.len();
        let mut two = Matrix::zeros(na, na);
        two.data.par_chunks_mut(na).enumerate().try_for_each(
            |(p, row)| -> Result<(), HfError> {
                for (q, value) in row.iter_mut().enumerate() {
                    *value = prepared.two_center(&self.auxiliary[p], &self.auxiliary[q])?;
                }
                Ok(())
            },
        )?;
        let metric = self.x.transpose().dot(&two).dot(&self.x);
        let corrected = three.sub(&metric.dot(&self.b).scale(0.5));
        if corrected.data.iter().any(|x| !x.is_finite()) {
            return Err(HfError::Integral(
                "nonfinite occupied-leg F12 kernel".into(),
            ));
        }
        Ok(FittedF12Pairs {
            b: Arc::clone(&self.b),
            corrected: Arc::new(corrected),
            scale: 1.0,
            n_target: self.n_target,
            n_occupied: self.n_occupied,
            budget: self.budget,
        })
    }
}

impl FittedF12Pairs {
    /// Select bra target ranges, retaining all ordered occupied ket pairs.
    /// This also builds rectangular f-squared/occupied blocks without storing
    /// their unused virtual or CABS columns.
    pub fn pairs(&self, left: Range<usize>, right: Range<usize>) -> Result<Vec<Matrix>, HfError> {
        if left.start >= left.end
            || right.start >= right.end
            || left.end > self.n_target
            || right.end > self.n_target
        {
            return Err(HfError::InvalidInput(
                "invalid F12 fitted target ranges".into(),
            ));
        }
        let (nl, nr, no, rank) = (left.len(), right.len(), self.n_occupied, self.b.rows);
        if no == 0 {
            return Ok(Vec::new());
        }
        let cells = 2 * self.b.data.len() as u128
            + 2 * rank as u128 * no as u128 * (nl + nr) as u128
            + (no as u128).pow(2) * nl as u128 * nr as u128
            + 4 * rank as u128 * (nl + nr) as u128
            + 8 * nl as u128 * nr as u128;
        let headers = (no as u128).pow(2) * std::mem::size_of::<Matrix>() as u128;
        if 8 * cells + headers > self.budget as u128 {
            return Err(HfError::Integral(
                "F12 fitted pair-output workspace estimate exceeds budget".into(),
            ));
        }
        let extract = |values: &Matrix, range: Range<usize>, i: usize| {
            let mut block = Matrix::zeros(range.len(), rank);
            for (p, t) in range.enumerate() {
                for a in 0..rank {
                    block[(p, a)] = values[(a, t * no + i)];
                }
            }
            block
        };
        let bl: Vec<_> = (0..no).map(|i| extract(&self.b, left.clone(), i)).collect();
        let wl: Vec<_> = (0..no)
            .map(|i| extract(&self.corrected, left.clone(), i))
            .collect();
        let br: Vec<_> = (0..no)
            .map(|i| extract(&self.b, right.clone(), i).transpose())
            .collect();
        let wr: Vec<_> = (0..no)
            .map(|i| extract(&self.corrected, right.clone(), i).transpose())
            .collect();
        let mut result = Vec::with_capacity(no * no);
        for i in 0..no {
            for j in 0..no {
                let mut block = bl[i].dot(&wr[j]).add(&wl[i].dot(&br[j]));
                if self.scale != 1.0 {
                    for value in &mut block.data {
                        *value *= self.scale;
                    }
                }
                result.push(block);
            }
        }
        if result.iter().any(|m| m.data.iter().any(|x| !x.is_finite())) {
            return Err(HfError::Integral(
                "nonfinite occupied-leg fitted F12 pairs".into(),
            ));
        }
        Ok(result)
    }
}
