// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! Semicanonical PNOs and the Galerkin local-MP2 equations in pair-specific
//! orthonormal virtual spaces. Occupied Fock couplings are retained.
use crate::error::HfError;
use crate::matrix::Matrix;

#[derive(Clone, Debug)]
pub struct NaturalPairSpace {
    pub coefficients: Matrix,
    pub energies: Vec<f64>,
    pub natural_occupations: Vec<f64>,
    pub initial_amplitudes: Matrix,
    pub density: Matrix,
    pub delta_pno: f64,
    pub untruncated_estimate: f64,
}
pub fn pair_energy(k: &Matrix, t: &Matrix) -> f64 {
    (0..k.rows)
        .map(|a| {
            (0..k.cols)
                .map(|b| k[(a, b)] * (2.0 * t[(a, b)] - t[(b, a)]))
                .sum::<f64>()
        })
        .sum()
}
/// The supplied K/F are expressed in one orthonormal PAO domain. First form
/// the semicanonical MP2 guess, then D=(tilde(T)^T T+tilde(T) T^T)/(1+delta_ij).
pub fn natural_pair_space(
    k: &Matrix,
    f: &Matrix,
    occupied_sum: f64,
    diagonal: bool,
    cutoff: f64,
) -> Result<NaturalPairSpace, HfError> {
    let n = k.rows;
    if k.cols != n
        || f.rows != n
        || f.cols != n
        || !occupied_sum.is_finite()
        || !cutoff.is_finite()
        || cutoff < 0.0
        || k.data.iter().chain(&f.data).any(|x| !x.is_finite())
        || (0..n).any(|a| (0..a).any(|b| (f[(a, b)] - f[(b, a)]).abs() > 1e-12))
        || (diagonal && (0..n).any(|a| (0..a).any(|b| (k[(a, b)] - k[(b, a)]).abs() > 1e-12)))
    {
        return Err(HfError::InvalidInput(
            "invalid pair K/F matrix or PNO cutoff".into(),
        ));
    }
    if n == 0 {
        return Ok(NaturalPairSpace {
            coefficients: Matrix::zeros(0, 0),
            energies: Vec::new(),
            natural_occupations: Vec::new(),
            initial_amplitudes: Matrix::zeros(0, 0),
            density: Matrix::zeros(0, 0),
            delta_pno: 0.0,
            untruncated_estimate: 0.0,
        });
    }
    let (eps, u) = f.eigh();
    let kc = u.transpose().dot(k).dot(&u);
    let mut t = Matrix::zeros(n, n);
    for a in 0..n {
        for b in 0..n {
            let gap = occupied_sum - eps[a] - eps[b];
            if gap >= -1e-14 {
                return Err(HfError::Response(
                    "local MP2 requires a positive occupied/virtual gap".into(),
                ));
            }
            t[(a, b)] = kc[(a, b)] / gap;
        }
    }
    let mut tilde = t.scale(2.0);
    tilde = tilde.sub(&t.transpose());
    let density_c = tilde
        .transpose()
        .dot(&t)
        .add(&tilde.dot(&t.transpose()))
        .scale(if diagonal { 0.5 } else { 1.0 });
    let density = u.dot(&density_c).dot(&u.transpose());
    let (occupations, pnos) = density_c.eigh();
    if occupations.iter().any(|x| *x < -1e-10) {
        return Err(HfError::Integral(
            "pair density has a negative eigenvalue".into(),
        ));
    }
    // Zero cutoff is the full-space limit, including zero natural occupations.
    let kept = (0..n)
        .rev()
        .filter(|&a| cutoff == 0.0 || occupations[a] > cutoff)
        .collect::<Vec<_>>();
    let mut p = Matrix::zeros(n, kept.len());
    for (j, &col) in kept.iter().enumerate() {
        for a in 0..n {
            p[(a, j)] = pnos[(a, col)];
        }
    }
    let mut fc = Matrix::zeros(n, n);
    for a in 0..n {
        fc[(a, a)] = eps[a];
    }
    let fp = p.transpose().dot(&fc).dot(&p);
    let (energies, v) = if kept.is_empty() {
        (Vec::new(), Matrix::zeros(0, 0))
    } else {
        fp.eigh()
    };
    let pc = p.dot(&v);
    let initial_amplitudes = pc.transpose().dot(&t).dot(&pc);
    let projected_k = pc.transpose().dot(&kc).dot(&pc);
    let untruncated_estimate = pair_energy(&kc, &t);
    let delta_pno = untruncated_estimate - pair_energy(&projected_k, &initial_amplitudes);
    Ok(NaturalPairSpace {
        coefficients: u.dot(&pc),
        energies,
        natural_occupations: kept.iter().map(|&i| occupations[i]).collect(),
        initial_amplitudes,
        density,
        delta_pno,
        untruncated_estimate,
    })
}

#[derive(Clone, Debug)]
pub struct PairBlock {
    pub i: usize,
    pub j: usize,
    /// Common virtual basis -> pair semicanonical PNO basis.
    pub basis: Matrix,
    pub energies: Vec<f64>,
    pub exchange: Matrix,
    pub initial: Matrix,
    pub strong: bool,
    /// Ordered-pair correction; the i!=j factor is applied by the solver.
    pub delta_pno: f64,
}
#[derive(Clone, Debug)]
pub struct LocalPairProblem {
    pub f_occ: Matrix,
    pub pairs: Vec<PairBlock>,
}
#[derive(Clone, Debug)]
pub struct LocalMp2Solution {
    pub amplitudes: Vec<Matrix>,
    pub residual_max: f64,
    pub iterations: usize,
    pub e_strong: f64,
    pub e_weak: f64,
    pub delta_pno: f64,
}
impl LocalMp2Solution {
    pub fn correlation_energy(&self) -> f64 {
        self.e_strong + self.e_weak + self.delta_pno
    }
}
fn pair_index(i: usize, j: usize) -> usize {
    let p = i.max(j);
    p * (p + 1) / 2 + i.min(j)
}
impl LocalPairProblem {
    pub fn validate(&self) -> Result<(), HfError> {
        let no = self.f_occ.rows;
        if self.f_occ.cols != no
            || self.pairs.len() != no * (no + 1) / 2
            || self.f_occ.data.iter().any(|x| !x.is_finite())
            || (0..no)
                .any(|i| (0..i).any(|j| (self.f_occ[(i, j)] - self.f_occ[(j, i)]).abs() > 1e-12))
        {
            return Err(HfError::InvalidInput(
                "invalid LMP2 occupied Fock/pair dimensions".into(),
            ));
        }
        let nv = self.pairs.first().map_or(0, |p| p.basis.rows);
        let mut bytes = 0usize;
        for (index, p) in self.pairs.iter().enumerate() {
            let m = p.basis.cols;
            if p.i >= no
                || p.j > p.i
                || pair_index(p.i, p.j) != index
                || p.basis.rows != nv
                || p.energies.len() != m
                || p.exchange.rows != m
                || p.exchange.cols != m
                || p.initial.rows != m
                || p.initial.cols != m
                || p.basis
                    .data
                    .iter()
                    .chain(&p.exchange.data)
                    .chain(&p.initial.data)
                    .chain(&p.energies)
                    .any(|x| !x.is_finite())
                || !p.delta_pno.is_finite()
            {
                return Err(HfError::InvalidInput(
                    "invalid LMP2 pair-space dimensions or values".into(),
                ));
            }
            let gram = p.basis.transpose().dot(&p.basis);
            if (0..m).any(|a| (0..m).any(|b| (gram[(a, b)] - f64::from(a == b)).abs() > 1e-8)) {
                return Err(HfError::InvalidInput(
                    "LMP2 pair virtuals are not orthonormal".into(),
                ));
            }
            bytes = bytes
                .checked_add((p.basis.data.len() + 16 * m * m) * 8)
                .ok_or_else(|| HfError::InvalidInput("LMP2 workspace size overflow".into()))?;
        }
        if bytes > 256 * 1024 * 1024 {
            return Err(HfError::Unsupported {
                method: "LMP2".into(),
                detail: "pair-space workspace exceeds 256 MiB".into(),
            });
        }
        Ok(())
    }
    /// Apply the positive zeroth-order operator to amplitudes in pair spaces.
    /// Cross-pair virtual overlaps U_ij^T U_kj appear on both virtual legs.
    pub fn apply(&self, t: &[Matrix]) -> Result<Vec<Matrix>, HfError> {
        if t.len() != self.pairs.len()
            || t.iter()
                .zip(&self.pairs)
                .any(|(a, p)| a.rows != p.basis.cols || a.cols != a.rows)
        {
            return Err(HfError::InvalidInput(
                "invalid LMP2 amplitude dimensions".into(),
            ));
        }
        let no = self.f_occ.rows;
        let transform = |target: &PairBlock, i: usize, j: usize| {
            let index = pair_index(i, j);
            let source = &self.pairs[index];
            let overlap = target.basis.transpose().dot(&source.basis);
            let matrix = if i >= j {
                t[index].clone()
            } else {
                t[index].transpose()
            };
            overlap.dot(&matrix).dot(&overlap.transpose())
        };
        let mut out = Vec::with_capacity(t.len());
        for (index, pair) in self.pairs.iter().enumerate() {
            let m = pair.basis.cols;
            let mut result = Matrix::zeros(m, m);
            for a in 0..m {
                for b in 0..m {
                    result[(a, b)] = (pair.energies[a] + pair.energies[b]
                        - self.f_occ[(pair.i, pair.i)]
                        - self.f_occ[(pair.j, pair.j)])
                        * t[index][(a, b)];
                }
            }
            for k in 0..no {
                if k != pair.i && self.f_occ[(pair.i, k)] != 0.0 {
                    result = result.sub(&transform(pair, k, pair.j).scale(self.f_occ[(pair.i, k)]));
                }
                if k != pair.j && self.f_occ[(k, pair.j)] != 0.0 {
                    result = result.sub(&transform(pair, pair.i, k).scale(self.f_occ[(k, pair.j)]));
                }
            }
            out.push(result);
        }
        Ok(out)
    }
    // Independent coordinates are orthonormal in the ordered-pair Frobenius
    // metric: sqrt(2) for i>j, and symmetric virtual coordinates for i==j.
    fn coordinates(&self, mut f: impl FnMut(usize, usize, usize, f64)) {
        for (index, p) in self.pairs.iter().enumerate() {
            if !p.strong {
                continue;
            }
            for a in 0..p.basis.cols {
                for b in 0..p.basis.cols {
                    if p.i == p.j && b > a {
                        continue;
                    }
                    f(
                        index,
                        a,
                        b,
                        if p.i != p.j || a != b {
                            std::f64::consts::SQRT_2
                        } else {
                            1.0
                        },
                    );
                }
            }
        }
    }
    fn pack(&self, m: &[Matrix]) -> Vec<f64> {
        let mut out = Vec::new();
        self.coordinates(|i, a, b, s| {
            out.push(if self.pairs[i].i == self.pairs[i].j && a != b {
                (m[i][(a, b)] + m[i][(b, a)]) / s
            } else {
                s * m[i][(a, b)]
            })
        });
        out
    }
    fn unpack(&self, vector: &[f64], fixed: bool) -> Vec<Matrix> {
        let mut out = self
            .pairs
            .iter()
            .map(|p| {
                if fixed && !p.strong {
                    p.initial.clone()
                } else {
                    Matrix::zeros(p.basis.cols, p.basis.cols)
                }
            })
            .collect::<Vec<_>>();
        let mut n = 0;
        self.coordinates(|i, a, b, s| {
            out[i][(a, b)] = vector[n] / s;
            if self.pairs[i].i == self.pairs[i].j {
                out[i][(b, a)] = out[i][(a, b)];
            }
            n += 1;
        });
        out
    }
    pub fn solve(&self, max_iter: usize, tolerance: f64) -> Result<LocalMp2Solution, HfError> {
        self.validate()?;
        if max_iter == 0 || !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(HfError::InvalidInput(
                "invalid LMP2 convergence controls".into(),
            ));
        }
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
        let apply = |x: &[f64]| self.apply(&self.unpack(x, false)).map(|m| self.pack(&m));
        let mut diagonal = Vec::new();
        self.coordinates(|index, a, b, _| {
            let p = &self.pairs[index];
            diagonal.push(
                p.energies[a] + p.energies[b] - self.f_occ[(p.i, p.i)] - self.f_occ[(p.j, p.j)],
            );
        });
        if diagonal.iter().any(|d| !d.is_finite() || *d <= 1e-14) {
            return Err(HfError::Response(
                "LMP2 preconditioner has a nonpositive gap".into(),
            ));
        }
        let zero = vec![0.0; diagonal.len()];
        let fixed = self.unpack(&zero, true);
        let fixed_action = self.apply(&fixed)?;
        let rhs = self.pack(
            &self
                .pairs
                .iter()
                .zip(fixed_action)
                .map(|(p, a)| p.exchange.add(&a).scale(-1.0))
                .collect::<Vec<_>>(),
        );
        let mut x = self.pack(
            &self
                .pairs
                .iter()
                .map(|p| p.initial.clone())
                .collect::<Vec<_>>(),
        );
        let ax = apply(&x)?;
        let mut residual = rhs.iter().zip(ax).map(|(b, a)| b - a).collect::<Vec<_>>();
        let mut z = residual
            .iter()
            .zip(&diagonal)
            .map(|(r, d)| r / d)
            .collect::<Vec<_>>();
        let mut direction = z.clone();
        let mut rz = dot(&residual, &z);
        for iteration in 0..=max_iter {
            let max_res = residual.iter().map(|v| v.abs()).fold(0.0, f64::max);
            if max_res <= tolerance {
                let amplitudes = self.unpack(&x, true);
                let a = self.apply(&amplitudes)?;
                let mut e_strong = 0.0;
                let mut e_weak = 0.0;
                let mut delta_pno = 0.0;
                for (index, p) in self.pairs.iter().enumerate() {
                    let factor = if p.i == p.j { 1.0 } else { 2.0 };
                    let energy = factor
                        * pair_energy(&p.exchange.scale(2.0).add(&a[index]), &amplitudes[index]);
                    if p.strong {
                        e_strong += energy;
                    } else {
                        e_weak += energy;
                    }
                    delta_pno += factor * p.delta_pno;
                }
                return Ok(LocalMp2Solution {
                    amplitudes,
                    residual_max: max_res,
                    iterations: iteration,
                    e_strong,
                    e_weak,
                    delta_pno,
                });
            }
            if iteration == max_iter {
                break;
            }
            let ad = apply(&direction)?;
            let curvature = dot(&direction, &ad);
            if !curvature.is_finite() || curvature <= 0.0 || !rz.is_finite() {
                return Err(HfError::Response("LMP2 PCG lost positive curvature".into()));
            }
            let alpha = rz / curvature;
            for i in 0..x.len() {
                x[i] += alpha * direction[i];
                residual[i] -= alpha * ad[i];
            }
            for i in 0..z.len() {
                z[i] = residual[i] / diagonal[i];
            }
            let next_rz = dot(&residual, &z);
            let beta = next_rz / rz;
            for i in 0..direction.len() {
                direction[i] = z[i] + beta * direction[i];
            }
            rz = next_rz;
        }
        Err(HfError::PostHf {
            method: "LMP2".into(),
            detail: format!("pair residual did not converge in {max_iter} PCG iterations"),
        })
    }
}
