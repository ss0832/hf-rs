// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! Pair-independent triples virtual spaces and the noncanonical Fock resolvent.
//! Ordered occupied triples retain their full virtual cubes. Workspace is bounded;
//! this reference implementation makes no linear-scaling storage claim.
use crate::{Matrix, error::HfError};
use std::sync::Arc;

/// Apply S on all three virtual legs without constructing a Kronecker matrix.
pub fn transform_cube(t: &[f64], n: usize, s: &Matrix) -> Result<Vec<f64>, HfError> {
    if s.cols != n
        || t.len()
            != n.checked_pow(3)
                .ok_or_else(|| HfError::InvalidInput("triples dimension overflow".into()))?
    {
        return Err(HfError::InvalidInput(
            "triples overlap/cube dimensions mismatch".into(),
        ));
    }
    let m = s.rows;
    if n == 0 || m == 0 {
        return Ok(vec![0.; m * m * m]);
    }
    let first = s.dot(&Matrix::from_vec(n, n * n, t.to_vec()));
    let mut second = Matrix::zeros(m * m, n);
    for a in 0..m {
        let slice = Matrix::from_vec(n, n, first.data[a * n * n..(a + 1) * n * n].to_vec());
        let transformed = s.dot(&slice);
        second.data[a * m * n..(a + 1) * m * n].copy_from_slice(&transformed.data);
    }
    Ok(second.dot(&s.transpose()).data)
}
#[derive(Clone, Debug)]
pub struct TriplesVirtualSpace {
    pub basis: Matrix,
    pub energies: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct TriplesBlock {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub space: Arc<TriplesVirtualSpace>,
    pub connected: Vec<f64>,
    pub singles: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct LocalTriplesProblem {
    pub f_occ: Matrix,
    pub blocks: Vec<TriplesBlock>,
}
#[derive(Clone, Debug)]
pub struct LocalTriplesSolution {
    pub amplitudes: Vec<Vec<f64>>,
    pub energy: f64,
    pub residual_max: f64,
    pub iterations: usize,
}
impl LocalTriplesProblem {
    fn validate(&self) -> Result<(), HfError> {
        let no = self.f_occ.rows;
        if self.f_occ.cols != no
            || self.blocks.len() != no.checked_pow(3).unwrap_or(usize::MAX)
            || self.f_occ.data.iter().any(|x| !x.is_finite())
            || (0..no)
                .any(|i| (0..i).any(|j| (self.f_occ[(i, j)] - self.f_occ[(j, i)]).abs() > 1e-12))
        {
            return Err(HfError::InvalidInput(
                "invalid triples occupied Fock/block dimensions".into(),
            ));
        }
        let nv = self.blocks.first().map_or(0, |b| b.space.basis.rows);
        let mut values = 0usize;
        for (index, b) in self.blocks.iter().enumerate() {
            let n = b.space.basis.cols;
            let cube = n
                .checked_pow(3)
                .ok_or_else(|| HfError::InvalidInput("triples cube dimension overflow".into()))?;
            if b.i >= no
                || b.j >= no
                || b.k >= no
                || (b.i * no + b.j) * no + b.k != index
                || b.space.basis.rows != nv
                || b.space.energies.len() != n
                || b.connected.len() != cube
                || b.singles.len() != cube
                || b.space
                    .basis
                    .data
                    .iter()
                    .chain(&b.space.energies)
                    .chain(&b.connected)
                    .chain(&b.singles)
                    .any(|x| !x.is_finite())
            {
                return Err(HfError::InvalidInput(
                    "invalid triples virtual space/source dimensions".into(),
                ));
            }
            let gram = b.space.basis.transpose().dot(&b.space.basis);
            if (0..n).any(|a| (0..n).any(|c| (gram[(a, c)] - f64::from(a == c)).abs() > 1e-8)) {
                return Err(HfError::InvalidInput(
                    "triples virtuals are not orthonormal".into(),
                ));
            }
            values = values
                .checked_add(cube)
                .ok_or_else(|| HfError::InvalidInput("triples workspace overflow".into()))?;
        }
        if values as u128 * 16 * 8 > 256 * 1024 * 1024 {
            return Err(HfError::Unsupported {
                method: "local triples".into(),
                detail: "ordered-triples workspace exceeds 256 MiB".into(),
            });
        }
        Ok(())
    }
    fn diagonal(&self) -> Result<Vec<Vec<f64>>, HfError> {
        self.blocks
            .iter()
            .map(|b| {
                let n = b.space.energies.len();
                let e = &b.space.energies;
                let occupied =
                    self.f_occ[(b.i, b.i)] + self.f_occ[(b.j, b.j)] + self.f_occ[(b.k, b.k)];
                let mut out = Vec::with_capacity(n * n * n);
                for a in 0..n {
                    for c in 0..n {
                        for d in 0..n {
                            let gap = e[a] + e[c] + e[d] - occupied;
                            if !gap.is_finite() || gap <= 1e-12 {
                                return Err(HfError::Response(
                                    "local triples requires positive Fock gaps".into(),
                                ));
                            }
                            out.push(gap);
                        }
                    }
                }
                Ok(out)
            })
            .collect()
    }
    pub fn apply(&self, t: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, HfError> {
        if t.len() != self.blocks.len()
            || t.iter()
                .zip(&self.blocks)
                .any(|(v, b)| v.len() != b.connected.len())
        {
            return Err(HfError::InvalidInput(
                "triples amplitude dimensions mismatch".into(),
            ));
        }
        let diagonal = self.diagonal()?;
        let no = self.f_occ.rows;
        let mut out = Vec::with_capacity(t.len());
        for (index, b) in self.blocks.iter().enumerate() {
            let mut value = t[index]
                .iter()
                .zip(&diagonal[index])
                .map(|(v, d)| v * d)
                .collect::<Vec<_>>();
            for l in 0..no {
                for (original, source, coupling) in [
                    (b.i, (l * no + b.j) * no + b.k, self.f_occ[(b.i, l)]),
                    (b.j, (b.i * no + l) * no + b.k, self.f_occ[(l, b.j)]),
                    (b.k, (b.i * no + b.j) * no + l, self.f_occ[(l, b.k)]),
                ] {
                    if l == original || coupling == 0. {
                        continue;
                    }
                    let target = &self.blocks[source].space;
                    let transformed = if Arc::ptr_eq(&b.space, target) {
                        t[source].clone()
                    } else {
                        transform_cube(
                            &t[source],
                            target.basis.cols,
                            &b.space.basis.transpose().dot(&target.basis),
                        )?
                    };
                    for (v, source) in value.iter_mut().zip(transformed) {
                        *v -= coupling * source;
                    }
                }
            }
            out.push(value);
        }
        Ok(out)
    }
    fn energy(&self, t: &[Vec<f64>]) -> f64 {
        self.blocks
            .iter()
            .zip(t)
            .map(|(b, t)| {
                let n = b.space.energies.len();
                let at = |a, c, d| t[(a * n + c) * n + d];
                let mut energy = 0.;
                for a in 0..n {
                    for c in 0..n {
                        for d in 0..n {
                            let tilde = 4. * at(a, c, d)
                                - 2. * at(a, d, c)
                                - 2. * at(c, a, d)
                                - 2. * at(d, c, a)
                                + at(c, d, a)
                                + at(d, a, c);
                            energy += tilde
                                * (b.connected[(a * n + c) * n + d]
                                    + b.singles[(a * n + c) * n + d])
                                / 3.;
                        }
                    }
                }
                energy
            })
            .sum()
    }
    pub fn solve(
        &self,
        iterative: bool,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<LocalTriplesSolution, HfError> {
        self.validate()?;
        if max_iter == 0 || !tolerance.is_finite() || tolerance <= 0. {
            return Err(HfError::InvalidInput(
                "invalid local triples convergence controls".into(),
            ));
        }
        let diagonal = self.diagonal()?;
        let mut t = self
            .blocks
            .iter()
            .zip(&diagonal)
            .map(|(b, d)| {
                b.connected
                    .iter()
                    .zip(d)
                    .map(|(w, d)| -w / d)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if !iterative {
            return Ok(LocalTriplesSolution {
                energy: self.energy(&t),
                amplitudes: t,
                residual_max: 0.,
                iterations: 0,
            });
        }
        let dot = |a: &[Vec<f64>], b: &[Vec<f64>]| {
            a.iter()
                .zip(b)
                .map(|(a, b)| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>())
                .sum::<f64>()
        };
        let at = self.apply(&t)?;
        let mut residual = self
            .blocks
            .iter()
            .zip(at)
            .map(|(b, at)| {
                b.connected
                    .iter()
                    .zip(at)
                    .map(|(w, at)| -w - at)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut z = residual
            .iter()
            .zip(&diagonal)
            .map(|(r, d)| r.iter().zip(d).map(|(r, d)| r / d).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut direction = z.clone();
        let mut rz = dot(&residual, &z);
        for iteration in 0..=max_iter {
            let max_res = residual
                .iter()
                .flatten()
                .map(|v| v.abs())
                .fold(0., f64::max);
            if max_res <= tolerance {
                return Ok(LocalTriplesSolution {
                    energy: self.energy(&t),
                    amplitudes: t,
                    residual_max: max_res,
                    iterations: iteration,
                });
            }
            if iteration == max_iter {
                break;
            }
            let ad = self.apply(&direction)?;
            let curvature = dot(&direction, &ad);
            if !curvature.is_finite() || curvature <= 0. || !rz.is_finite() {
                return Err(HfError::Response(
                    "local triples PCG lost positive curvature".into(),
                ));
            }
            let alpha = rz / curvature;
            for index in 0..t.len() {
                for a in 0..t[index].len() {
                    t[index][a] += alpha * direction[index][a];
                    residual[index][a] -= alpha * ad[index][a];
                    z[index][a] = residual[index][a] / diagonal[index][a];
                }
            }
            let next = dot(&residual, &z);
            let beta = next / rz;
            for index in 0..direction.len() {
                for a in 0..direction[index].len() {
                    direction[index][a] = z[index][a] + beta * direction[index][a];
                }
            }
            rz = next;
        }
        Err(HfError::PostHf {
            method: "local triples (T1)".into(),
            detail: format!("triples residual did not converge in {max_iter} iterations"),
        })
    }
}
