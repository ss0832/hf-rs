// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Single-determinant normal-ordered zero/one/two-body commutators and IMSRG(2).
//! Hergert et al., arXiv:1612.08315, Eqs. 9.104--9.106 and Appendix.
//! Spin-orbital two-body storage is O(N^4); induced normal-ordered three-body
//! terms are discarded. No determinant space is constructed in production.

use crate::{Matrix, error::HfError};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct No2bOperator {
    pub zero: f64,
    pub one: Matrix,
    /// Antisymmetric pair indices: coefficient of a_p^+ a_q^+ a_s a_r / 4.
    pub two: Vec<f64>,
}

pub use crate::options::ImsrgGenerator as SRGGenerator;

#[inline]
fn idx(n: usize, p: usize, q: usize, r: usize, s: usize) -> usize {
    ((p * n + q) * n + r) * n + s
}
#[inline]
fn g(t: &[f64], n: usize, p: usize, q: usize, r: usize, s: usize) -> f64 {
    t[idx(n, p, q, r, s)]
}
impl No2bOperator {
    pub fn validate(&self, occupied: &[bool]) -> Result<(), HfError> {
        let n = self.one.rows;
        let entries = n.checked_pow(4).unwrap_or(usize::MAX);
        if n == 0
            || self.one.cols != n
            || occupied.len() != n
            || entries > 256 * 1024 * 1024 / (8 * 16)
            || self.two.len() != entries
            || self.one.data.len() != n.checked_pow(2).unwrap_or(usize::MAX)
        {
            return Err(HfError::InvalidInput(
                "invalid NO2B dimensions or workspace above 256 MiB estimate".into(),
            ));
        }
        if !self.zero.is_finite()
            || self
                .one
                .data
                .iter()
                .chain(&self.two)
                .any(|v| !v.is_finite())
        {
            return Err(HfError::PostHf {
                method: "IM-SRG(2)".into(),
                detail: "nonfinite NO2B operator".into(),
            });
        }
        Ok(())
    }
    pub(crate) fn add_scaled(&self, other: &Self, scale: f64) -> Self {
        Self {
            zero: self.zero + scale * other.zero,
            one: self.one.add(&other.one.scale(scale)),
            two: self
                .two
                .iter()
                .zip(&other.two)
                .map(|(a, b)| a + scale * b)
                .collect(),
        }
    }
    pub(crate) fn norm_max(&self) -> f64 {
        self.one
            .data
            .iter()
            .chain(&self.two)
            .fold(self.zero.abs(), |a, b| a.max(b.abs()))
    }
    pub fn off_diagonal_norm(&self, occupied: &[bool]) -> f64 {
        let od = off_diagonal(self, occupied);
        (od.one.data.iter().map(|x| x * x).sum::<f64>()
            + 0.25 * od.two.iter().map(|x| x * x).sum::<f64>())
        .sqrt()
    }
}

/// General real [A,B]_NO2B, valid for Hermitian or anti-Hermitian operands.
/// No symmetry projection is applied to the result.
pub fn no2b_commutator(
    a: &No2bOperator,
    b: &No2bOperator,
    occupied: &[bool],
) -> Result<No2bOperator, HfError> {
    a.validate(occupied)?;
    b.validate(occupied)?;
    let n = occupied.len();
    let occ = |i: usize| f64::from(occupied[i]);
    let mut zero = 0.;
    for p in 0..n {
        for q in 0..n {
            if occupied[p] && !occupied[q] {
                zero += a.one[(p, q)] * b.one[(q, p)] - b.one[(p, q)] * a.one[(q, p)];
            }
            if !occupied[p] || !occupied[q] {
                continue;
            }
            for r in 0..n {
                for s in 0..n {
                    if !occupied[r] && !occupied[s] {
                        zero += 0.25
                            * (g(&a.two, n, p, q, r, s) * g(&b.two, n, r, s, p, q)
                                - g(&b.two, n, p, q, r, s) * g(&a.two, n, r, s, p, q));
                    }
                }
            }
        }
    }
    let mut one = Matrix::zeros(n, n);
    one.data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for (j, value) in row.iter_mut().enumerate() {
            for p in 0..n {
                *value += a.one[(i, p)] * b.one[(p, j)] - b.one[(i, p)] * a.one[(p, j)];
                for q in 0..n {
                    let w = occ(p) - occ(q);
                    if w != 0. {
                        *value += w
                            * (a.one[(p, q)] * g(&b.two, n, q, i, p, j)
                                - b.one[(p, q)] * g(&a.two, n, q, i, p, j));
                    }
                    for r in 0..n {
                        let w = occ(p) * occ(q) * (1. - occ(r))
                            + (1. - occ(p)) * (1. - occ(q)) * occ(r);
                        if w != 0. {
                            *value += 0.5
                                * w
                                * (g(&a.two, n, r, i, p, q) * g(&b.two, n, p, q, r, j)
                                    - g(&b.two, n, r, i, p, q) * g(&a.two, n, p, q, r, j));
                        }
                    }
                }
            }
        }
    });
    let mut two = vec![0.; n.pow(4)];
    two.par_chunks_mut(n * n)
        .enumerate()
        .for_each(|(ij, block)| {
            let (i, j) = (ij / n, ij % n);
            for k in 0..n {
                for l in 0..n {
                    let mut value = 0.;
                    for p in 0..n {
                        value += a.one[(i, p)] * g(&b.two, n, p, j, k, l)
                            - b.one[(i, p)] * g(&a.two, n, p, j, k, l)
                            - a.one[(j, p)] * g(&b.two, n, p, i, k, l)
                            + b.one[(j, p)] * g(&a.two, n, p, i, k, l)
                            - a.one[(p, k)] * g(&b.two, n, i, j, p, l)
                            + b.one[(p, k)] * g(&a.two, n, i, j, p, l)
                            + a.one[(p, l)] * g(&b.two, n, i, j, p, k)
                            - b.one[(p, l)] * g(&a.two, n, i, j, p, k);
                        for q in 0..n {
                            let w = 1. - occ(p) - occ(q);
                            if w != 0. {
                                value += 0.5
                                    * w
                                    * (g(&a.two, n, i, j, p, q) * g(&b.two, n, p, q, k, l)
                                        - g(&b.two, n, i, j, p, q) * g(&a.two, n, p, q, k, l));
                            }
                            let w = occ(p) - occ(q);
                            if w != 0. {
                                value += w
                                    * (g(&a.two, n, p, i, q, k) * g(&b.two, n, q, j, p, l)
                                        - g(&a.two, n, p, j, q, k) * g(&b.two, n, q, i, p, l)
                                        - g(&a.two, n, p, i, q, l) * g(&b.two, n, q, j, p, k)
                                        + g(&a.two, n, p, j, q, l) * g(&b.two, n, q, i, p, k));
                            }
                        }
                    }
                    block[k * n + l] = value;
                }
            }
        });
    Ok(No2bOperator { zero, one, two })
}

fn off_diagonal(h: &No2bOperator, occupied: &[bool]) -> No2bOperator {
    let n = occupied.len();
    let mut od = No2bOperator {
        zero: 0.,
        one: Matrix::zeros(n, n),
        two: vec![0.; n.pow(4)],
    };
    for p in 0..n {
        for q in 0..n {
            if occupied[p] != occupied[q] {
                od.one[(p, q)] = h.one[(p, q)];
            }
            for r in 0..n {
                for s in 0..n {
                    if occupied[p] == occupied[q]
                        && occupied[r] == occupied[s]
                        && occupied[p] != occupied[r]
                    {
                        od.two[idx(n, p, q, r, s)] = h.two[idx(n, p, q, r, s)];
                    }
                }
            }
        }
    }
    od
}
fn generator(
    h: &No2bOperator,
    occupied: &[bool],
    kind: SRGGenerator,
) -> Result<No2bOperator, HfError> {
    let n = occupied.len();
    let mut od = off_diagonal(h, occupied);
    if kind == SRGGenerator::Wegner {
        return no2b_commutator(&h.add_scaled(&od, -1.), &od, occupied);
    }
    let divide = |v: f64, gap: f64| -> Result<f64, HfError> {
        if v == 0. {
            return Ok(0.);
        }
        if gap.abs() < 1e-12 {
            return Err(HfError::Response("singular White IMSRG denominator".into()));
        }
        Ok(v / gap)
    };
    for p in 0..n {
        for q in 0..n {
            od.one[(p, q)] = divide(od.one[(p, q)], h.one[(p, p)] - h.one[(q, q)])?;
            for r in 0..n {
                for s in 0..n {
                    let index = idx(n, p, q, r, s);
                    od.two[index] = divide(
                        od.two[index],
                        h.one[(p, p)] + h.one[(q, q)] - h.one[(r, r)] - h.one[(s, s)],
                    )?;
                }
            }
        }
    }
    Ok(od)
}
fn rhs(h: &No2bOperator, occupied: &[bool], kind: SRGGenerator) -> Result<No2bOperator, HfError> {
    no2b_commutator(&generator(h, occupied, kind)?, h, occupied)
}
fn rk4(
    h: &No2bOperator,
    occupied: &[bool],
    kind: SRGGenerator,
    dt: f64,
) -> Result<No2bOperator, HfError> {
    let k1 = rhs(h, occupied, kind)?;
    let k2 = rhs(&h.add_scaled(&k1, 0.5 * dt), occupied, kind)?;
    let k3 = rhs(&h.add_scaled(&k2, 0.5 * dt), occupied, kind)?;
    let k4 = rhs(&h.add_scaled(&k3, dt), occupied, kind)?;
    let result = h
        .add_scaled(&k1, dt / 6.)
        .add_scaled(&k2, dt / 3.)
        .add_scaled(&k3, dt / 3.)
        .add_scaled(&k4, dt / 6.);
    result.validate(occupied)?;
    Ok(result)
}

/// RK4, optionally with maximum-component step-doubling error control. `n_steps`
/// sets the initial step; finite flow time is not a ground-state convergence claim.
pub fn integrate_imsrg2(
    initial: &No2bOperator,
    occupied: &[bool],
    kind: SRGGenerator,
    end: f64,
    n_steps: usize,
    tolerance: Option<f64>,
) -> Result<(No2bOperator, usize), HfError> {
    initial.validate(occupied)?;
    if !end.is_finite()
        || end < 0.
        || n_steps == 0
        || n_steps > 100_000
        || tolerance.is_some_and(|t| !t.is_finite() || t <= 0.)
    {
        return Err(HfError::InvalidInput(
            "IMSRG needs finite nonnegative flow and positive step/tolerance controls".into(),
        ));
    }
    let mut state = initial.clone();
    if end == 0. {
        return Ok((state, 0));
    }
    let mut time = 0.;
    let mut step = end / n_steps as f64;
    let mut accepted = 0;
    let mut attempts = 0;
    while time < end {
        attempts += 1;
        if attempts > 100_000 || step <= 0. || time + step == time {
            return Err(HfError::Response("IMSRG integration step exhausted".into()));
        }
        let dt = step.min(end - time);
        let trial = (|| {
            let whole = rk4(&state, occupied, kind, dt)?;
            let fine = if tolerance.is_some() {
                let half = rk4(&state, occupied, kind, dt * 0.5)?;
                Some(rk4(&half, occupied, kind, dt * 0.5)?)
            } else {
                None
            };
            Ok::<_, HfError>((whole, fine))
        })();
        let (whole, fine) = match trial {
            Ok(result) => result,
            Err(HfError::PostHf { .. }) if tolerance.is_some() => {
                step = dt * 0.25;
                continue;
            }
            Err(error) => return Err(error),
        };
        if let (Some(tol), Some(fine)) = (tolerance, fine) {
            let error = fine.add_scaled(&whole, -1.).norm_max() / 15.;
            let target = tol * (1. + fine.norm_max());
            let factor = if error == 0. {
                2.
            } else {
                (0.9 * (target / error).powf(0.2)).clamp(0.1, 2.)
            };
            if error > target {
                step = dt * factor;
                continue;
            }
            state = fine;
            step = dt * factor;
        } else {
            state = whole;
        }
        time += dt;
        accepted += 1;
        if end - time < 1e-14 * end {
            time = end;
        }
    }
    Ok((state, accepted))
}

/// Build a correlation-relative RHF Hamiltonian in interleaved spin orbitals.
pub(crate) fn rhf_normal_ordered<'a>(
    hf: &'a crate::scf::HFResult,
    integral_cache: Option<&crate::context::MoIntegralCache<'a>>,
) -> Result<(No2bOperator, Vec<bool>), HfError> {
    use std::borrow::Cow;
    let nm = hf.mo_coefficients.cols;
    if !hf.converged || nm != hf.orbital_energies.len() || hf.n_occ == 0 || hf.n_occ > nm {
        return Err(HfError::Scf(
            "NO2B methods require a converged, compatible RHF reference".into(),
        ));
    }
    let n = 2 * nm;
    if (n as u128).pow(4) * 8 * 16 > 256 * 1024 * 1024 {
        return Err(HfError::InvalidInput(
            "NO2B spin-orbital workspace exceeds 256 MiB estimate".into(),
        ));
    }
    let mo = if let Some(cache) = integral_cache {
        Cow::Borrowed(cache.full()?.as_ref())
    } else {
        Cow::Owned(crate::mo_integrals::transform_eri_ao_to_mo(
            &hf.mo_coefficients,
            &hf.eri,
        ))
    };
    let f = hf
        .mo_coefficients
        .transpose()
        .dot(&hf.fock_matrix)
        .dot(&hf.mo_coefficients);
    let mut one = Matrix::zeros(n, n);
    let mut two = vec![0.; n.pow(4)];
    for p in 0..n {
        for q in 0..n {
            if p % 2 == q % 2 {
                one[(p, q)] = f[(p / 2, q / 2)];
            }
            for r in 0..n {
                for s in 0..n {
                    two[((p * n + q) * n + r) * n + s] =
                        crate::context::spin_antisymmetrized(&mo, &mo, None, p, q, r, s);
                }
            }
        }
    }
    let occupied = (0..n).map(|p| p / 2 < hf.n_occ).collect::<Vec<_>>();
    let initial = No2bOperator { zero: 0., one, two };
    initial.validate(&occupied)?;
    Ok((initial, occupied))
}
