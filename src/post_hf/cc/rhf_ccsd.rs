// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Exact closed-shell spin reduction of Stanton et al., JCP 94, 4334 (1991).
//! T2 stores the alpha-beta block; equal-spin blocks are its antisymmetrization.
//! No spin-orbital fourth-power intermediates are allocated.

use super::ccsd::{CCSDOptions, CCSDResult, SpatialResidualCorrection};
use crate::solvers::TensorDIIS;
use crate::{matrix::Matrix, mo_integrals::MoEriTensor};
use rayon::prelude::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn solve(
    reference_energy: f64,
    eps: &[f64],
    eri: &MoEriTensor,
    fc: usize,
    no: usize,
    nv: usize,
    options: &CCSDOptions,
    correction: Option<&SpatialResidualCorrection<'_>>,
    doubles_only: bool,
) -> Result<CCSDResult, String> {
    if eps.len() != eri.n_basis
        || eps.iter().any(|x| !x.is_finite())
        || options.max_iterations == 0
        || !options.energy_tol.is_finite()
        || options.energy_tol <= 0.0
        || !options.residual_tol.is_finite()
        || options.residual_tol <= 0.0
    {
        return Err("invalid spatial CCSD energies or convergence controls".into());
    }
    let kernel = RhfCcsdKernel::new(eri, fc, no, nv)?;
    let mut t1 = vec![0.0; no * nv];
    let d1: Vec<_> = (0..no * nv)
        .map(|ia| eps[fc + ia / nv] - eps[fc + no + ia % nv])
        .collect();
    let d2: Vec<_> = (0..no * no * nv * nv)
        .map(|k| {
            let (i, j, a, b) = (
                k / (no * nv * nv),
                (k / (nv * nv)) % no,
                (k / nv) % nv,
                k % nv,
            );
            d1[i * nv + a] + d1[j * nv + b]
        })
        .collect();
    if d1
        .iter()
        .chain(&d2)
        .any(|x| !x.is_finite() || x.abs() < 1e-14)
    {
        return Err("singular spatial CCSD denominator".into());
    }
    let mut t2: Vec<_> = kernel.ov.data.iter().zip(&d2).map(|(g, d)| g / d).collect();
    let mut diis = TensorDIIS::new(options.diis_subspace);
    let (mut last, mut converged, mut iterations) = (0.0, false, 0);
    let (a, b) = (1.0 + 3.0_f64.sqrt(), 1.0 - 3.0_f64.sqrt());
    let swapped = |k: usize| (k / (nv * nv)) * nv * nv + (k % nv) * nv + (k / nv) % nv;
    for iteration in 0..options.max_iterations {
        iterations = iteration + 1;
        let (energy, mut r1, mut r2) = kernel.numerator(&t1, &t2);
        for (k, r) in r1.iter_mut().enumerate() {
            *r -= d1[k] * t1[k];
        }
        for (k, r) in r2.iter_mut().enumerate() {
            *r -= d2[k] * t2[k];
        }
        if let Some(source) = correction {
            let (s1, s2) = source(&t1)?;
            if s1.len() != r1.len()
                || s2.len() != r2.len()
                || s1.iter().chain(&s2).any(|x| !x.is_finite())
            {
                return Err("invalid spatial CCSD residual correction".into());
            }
            for (r, s) in r1.iter_mut().zip(s1) {
                *r += s;
            }
            for (r, s) in r2.iter_mut().zip(s2) {
                *r += s;
            }
        }
        if doubles_only {
            r1.fill(0.0);
        }
        if !energy.is_finite() || r1.iter().chain(&r2).any(|x| !x.is_finite()) {
            return Err("spatial CCSD energy/residual became nonfinite".into());
        }
        let max_res = r1
            .iter()
            .chain(&r2)
            .map(|x| x.abs())
            .chain((0..r2.len()).map(|k| (r2[k] - r2[swapped(k)]).abs()))
            .fold(0.0, f64::max);
        let change = (energy - last).abs();
        last = energy;
        if change < options.energy_tol && max_res <= options.residual_tol {
            converged = true;
            break;
        }
        let n1 = t1.len();
        let mut next = Vec::with_capacity(n1 + t2.len());
        let mut error = Vec::with_capacity(n1 + t2.len());
        next.extend((0..n1).map(|k| 2.0_f64.sqrt() * (t1[k] + r1[k] / d1[k])));
        error.extend(r1.iter().map(|r| 2.0_f64.sqrt() * r));
        // This invertible representation gives exactly the full spin residual
        // metric: 2|R1|^2 + 8|R2|^2 - 4<R2,R2_ab-swapped>.
        next.extend((0..t2.len()).map(|k| {
            let l = swapped(k);
            a * (t2[k] + r2[k] / d2[k]) + b * (t2[l] + r2[l] / d2[l])
        }));
        error.extend((0..t2.len()).map(|k| a * r2[k] + b * r2[swapped(k)]));
        diis.push(next.clone(), error);
        let updated = diis.extrapolate().unwrap_or(next);
        for k in 0..n1 {
            t1[k] = updated[k] / 2.0_f64.sqrt();
        }
        for k in 0..t2.len() {
            t2[k] = (a * updated[n1 + k] - b * updated[n1 + swapped(k)]) / (a * a - b * b);
        }
    }
    Ok(CCSDResult {
        reference_energy,
        correlation_energy: last,
        total_energy: reference_energy + last,
        iterations,
        converged,
        t1_amplitudes: t1,
        t2_amplitudes: t2,
        n_occ: no,
        n_virt: nv,
    })
}

pub(super) struct RhfCcsdKernel<'a> {
    eri: &'a MoEriTensor,
    fc: usize,
    no: usize,
    nv: usize,
    pp: Matrix,
    hh: Matrix,
    ov: Matrix,
    am: Matrix,
}

fn build(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64 + Sync) -> Matrix {
    Matrix::from_vec(
        rows,
        cols,
        (0..rows * cols)
            .into_par_iter()
            .map(|k| f(k / cols, k % cols))
            .collect(),
    )
}

impl<'a> RhfCcsdKernel<'a> {
    pub(super) fn new(
        eri: &'a MoEriTensor,
        fc: usize,
        no: usize,
        nv: usize,
    ) -> Result<Self, String> {
        if no == 0 || nv == 0 || fc + no + nv != eri.n_basis {
            return Err("invalid spatial CCSD orbital dimensions".into());
        }
        let estimate = (nv as u128).pow(4)
            + no as u128 * (nv as u128).pow(3)
            + 48 * (no as u128).pow(2) * (nv as u128).pow(2)
            + 8 * (no as u128).pow(4);
        if estimate * 8 > 256 * 1024 * 1024 {
            return Err("CCSD spatial workspace exceeds 256 MiB".into());
        }
        let g = |p, q, r, s| eri.get(fc + p, fc + q, fc + r, fc + s);
        Ok(Self {
            eri,
            fc,
            no,
            nv,
            pp: build(nv * nv, nv * nv, |ef, ab| {
                g(no + ab / nv, no + ef / nv, no + ab % nv, no + ef % nv)
            }),
            hh: build(no * no, no * no, |ij, mn| {
                g(mn / no, ij / no, mn % no, ij % no)
            }),
            ov: build(no * no, nv * nv, |ij, ab| {
                g(ij / no, no + ab / nv, ij % no, no + ab % nv)
            }),
            am: build(nv * nv, no * nv, |ef, ma| {
                g(no + ma % nv, no + ef / nv, ma / nv, no + ef % nv)
            }),
        })
    }

    #[inline(always)]
    fn g(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        self.eri
            .get(self.fc + p, self.fc + q, self.fc + r, self.fc + s)
    }

    /// Nonlinear numerator before subtracting the diagonal denominators.
    pub(super) fn numerator(&self, t1: &[f64], t2: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        let (o, v) = (self.no, self.nv);
        let t = |i, j, a, b| t2[((i * o + j) * v + a) * v + b];
        let s = |i, a| t1[i * v + a];
        let tau = build(o * o, v * v, |ij, ab| {
            t(ij / o, ij % o, ab / v, ab % v) + s(ij / o, ab / v) * s(ij % o, ab % v)
        });
        let tt = |i, j, a, b| t(i, j, a, b) + 0.5 * s(i, a) * s(j, b);
        let mut energy = 0.0;
        for ij in 0..o * o {
            for a in 0..v {
                for b in 0..v {
                    energy += (2.0 * self.ov[(ij, a * v + b)] - self.ov[(ij, b * v + a)])
                        * tau[(ij, a * v + b)];
                }
            }
        }
        let mut fme = Matrix::zeros(o, v);
        let mut fae = Matrix::zeros(v, v);
        let mut fmi = Matrix::zeros(o, o);
        for m in 0..o {
            for e in 0..v {
                for n in 0..o {
                    for f in 0..v {
                        fme[(m, e)] += s(n, f)
                            * (2.0 * self.g(m, o + e, n, o + f) - self.g(m, o + f, n, o + e));
                    }
                }
            }
        }
        for a in 0..v {
            for e in 0..v {
                for m in 0..o {
                    for f in 0..v {
                        fae[(a, e)] += s(m, f)
                            * (2.0 * self.g(m, o + f, o + a, o + e)
                                - self.g(m, o + e, o + a, o + f));
                    }
                }
                for m in 0..o {
                    for n in 0..o {
                        for f in 0..v {
                            fae[(a, e)] -= (2.0 * tt(m, n, a, f) - tt(m, n, f, a))
                                * self.g(m, o + e, n, o + f);
                        }
                    }
                }
            }
        }
        for m in 0..o {
            for i in 0..o {
                for n in 0..o {
                    for e in 0..v {
                        fmi[(m, i)] +=
                            s(n, e) * (2.0 * self.g(m, i, n, o + e) - self.g(m, o + e, n, i));
                    }
                }
                for n in 0..o {
                    for e in 0..v {
                        for f in 0..v {
                            fmi[(m, i)] += (2.0 * tt(i, n, e, f) - tt(i, n, f, e))
                                * self.g(m, o + e, n, o + f);
                        }
                    }
                }
            }
        }
        let fa2 = fae.sub(&t1_matrix(t1, o, v).transpose().dot(&fme).scale(0.5));
        let fm2 = fmi.add(&fme.dot(&t1_matrix(t1, o, v).transpose()).scale(0.5));
        let pp = tau.dot(&self.pp);
        let hh = self.hh.dot(&tau);
        let quadratic = tau.dot(&self.ov.transpose()).dot(&tau);
        let am = tau.dot(&self.am);
        let h1 = build(o * o, o * o, |ij, mn| {
            let (i, j, m, n) = (ij / o, ij % o, mn / o, mn % o);
            (0..v)
                .map(|e| s(j, e) * self.g(m, i, n, o + e) + s(i, e) * self.g(m, o + e, n, j))
                .sum()
        })
        .dot(&tau);
        let index = |m, b, e, j| ((m * v + b) * v + e) * o + j;
        let w: Vec<[f64; 4]> = (0..o * v * v * o)
            .into_par_iter()
            .map(|k| {
                let (m, b, e, j) = (k / (v * v * o), (k / (v * o)) % v, (k / o) % v, k % o);
                let bare_d = self.g(m, o + e, o + b, j);
                let bare_x = -self.g(m, j, o + b, o + e);
                let (mut d, mut x) = (bare_d, bare_x);
                for f in 0..v {
                    d += s(j, f) * self.g(m, o + e, o + b, o + f);
                    x -= s(j, f) * self.g(m, o + f, o + b, o + e);
                }
                for n in 0..o {
                    d -= s(n, b) * self.g(m, o + e, n, j);
                    x += s(n, b) * self.g(m, j, n, o + e);
                    for f in 0..v {
                        d -= (0.5 * t(j, n, f, b) - t(j, n, b, f) + s(j, f) * s(n, b))
                            * self.g(m, o + e, n, o + f)
                            + 0.5 * t(j, n, b, f) * self.g(m, o + f, n, o + e);
                        x += (0.5 * t(j, n, f, b) + s(j, f) * s(n, b)) * self.g(m, o + f, n, o + e);
                    }
                }
                [d, x, bare_d, bare_x]
            })
            .collect();
        let r1: Vec<f64> = (0..o * v)
            .into_par_iter()
            .map(|ia| {
                let (i, a) = (ia / v, ia % v);
                let mut value = 0.0;
                for e in 0..v {
                    value += s(i, e) * fae[(a, e)];
                }
                for m in 0..o {
                    value -= s(m, a) * fmi[(m, i)];
                    for e in 0..v {
                        value += (2.0 * t(i, m, a, e) - t(i, m, e, a)) * fme[(m, e)]
                            + s(m, e)
                                * (2.0 * self.g(m, o + e, o + a, i) - self.g(m, i, o + a, o + e));
                        for f in 0..v {
                            value += (2.0 * t(i, m, e, f) - t(i, m, f, e))
                                * self.g(m, o + f, o + a, o + e);
                        }
                    }
                    for n in 0..o {
                        for e in 0..v {
                            value -= (2.0 * t(m, n, a, e) - t(m, n, e, a)) * self.g(n, o + e, m, i);
                        }
                    }
                }
                value
            })
            .collect();
        let r2 = (0..o * o * v * v)
            .into_par_iter()
            .map(|k| {
                let (i, j, a, b) = (k / (o * v * v), (k / (v * v)) % o, (k / v) % v, k % v);
                let (ij, ab) = (i * o + j, a * v + b);
                let mut value = self.ov[(ij, ab)]
                    + pp[(ij, ab)]
                    + hh[(ij, ab)]
                    + h1[(ij, ab)]
                    + quadratic[(ij, ab)];
                for m in 0..o {
                    value -= s(m, b) * am[(ij, m * v + a)] + s(m, a) * am[(j * o + i, m * v + b)];
                    value -= t(i, m, a, b) * fm2[(m, j)] + t(m, j, a, b) * fm2[(m, i)];
                    value -= s(m, a) * self.g(m, i, o + b, j) + s(m, b) * self.g(m, j, o + a, i);
                    for e in 0..v {
                        let mbj = w[index(m, b, e, j)];
                        let maj = w[index(m, a, e, j)];
                        let mbi = w[index(m, b, e, i)];
                        let mai = w[index(m, a, e, i)];
                        value += (2.0 * t(i, m, a, e) - t(i, m, e, a)) * mbj[0]
                            + t(i, m, a, e) * mbj[1]
                            + t(i, m, e, b) * maj[1]
                            + t(j, m, e, a) * mbi[1]
                            + (2.0 * t(j, m, b, e) - t(j, m, e, b)) * mai[0]
                            + t(j, m, b, e) * mai[1]
                            - s(i, e) * s(m, a) * mbj[2]
                            + s(i, e) * s(m, b) * maj[3]
                            + s(j, e) * s(m, a) * mbi[3]
                            - s(j, e) * s(m, b) * mai[2];
                    }
                }
                for e in 0..v {
                    value += t(i, j, a, e) * fa2[(b, e)]
                        + t(i, j, e, b) * fa2[(a, e)]
                        + s(i, e) * self.g(o + a, o + e, o + b, j)
                        + s(j, e) * self.g(o + a, i, o + b, o + e);
                }
                value
            })
            .collect();
        (energy, r1, r2)
    }
}

fn t1_matrix(t1: &[f64], no: usize, nv: usize) -> Matrix {
    Matrix::from_vec(no, nv, t1.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_spatial_trial_residual_matches_public_pyscf() {
        let data: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/data/rhf_ccsd_residual_reference.json"
        ))
        .unwrap();
        let values = |v: &serde_json::Value| {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        for case in data["cases"].as_array().unwrap() {
            let (no, nv) = (
                case["n_occ"].as_u64().unwrap() as usize,
                case["n_virt"].as_u64().unwrap() as usize,
            );
            let n = no + nv;
            let eri = MoEriTensor {
                n_basis: n,
                n_pairs: n * (n + 1) / 2,
                data: values(&case["integrals"]),
            };
            let kernel = RhfCcsdKernel::new(&eri, 0, no, nv).unwrap();
            let options = CCSDOptions {
                max_iterations: 200,
                energy_tol: 1e-13,
                residual_tol: 1e-11,
                ..Default::default()
            };
            let result = solve(
                0.0,
                &values(&case["orbital_energies"]),
                &eri,
                0,
                no,
                nv,
                &options,
                None,
                false,
            )
            .unwrap();
            assert!(result.converged, "{} did not converge", case["name"]);
            let doubles = solve(
                0.0,
                &values(&case["orbital_energies"]),
                &eri,
                0,
                no,
                nv,
                &options,
                None,
                true,
            )
            .unwrap();
            assert!(doubles.converged, "{} CCD did not converge", case["name"]);
            assert!(doubles.t1_amplitudes.iter().all(|x| *x == 0.0));
            assert!(
                (doubles.correlation_energy - case["ccd_energy"].as_f64().unwrap()).abs() < 1e-10
            );
            let expected_doubles = values(&case["ccd_t2"]);
            let doubles_error = doubles
                .t2_amplitudes
                .iter()
                .zip(&expected_doubles)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            eprintln!(
                "{} CCD: {} amplitudes, error {doubles_error:e}",
                case["name"],
                expected_doubles.len()
            );
            assert!(doubles_error < 1e-9);
            assert!(
                (result.correlation_energy - case["converged_energy"].as_f64().unwrap()).abs()
                    < 1e-10
            );
            for (actual, key) in [
                (&result.t1_amplitudes, "converged_t1"),
                (&result.t2_amplitudes, "converged_t2"),
            ] {
                let expected = values(&case[key]);
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                eprintln!(
                    "{} {key}: {} components, error {error:e}",
                    case["name"],
                    actual.len()
                );
                assert!(error < 1e-9, "{} {key}: {error:e}", case["name"]);
            }
            for trial in case["trials"].as_array().unwrap() {
                let (e, r1, r2) = kernel.numerator(&values(&trial["t1"]), &values(&trial["t2"]));
                assert!((e - trial["energy"].as_f64().unwrap()).abs() < 1e-12);
                for (actual, key) in [(&r1, "numerator1"), (&r2, "numerator2")] {
                    let expected = values(&trial[key]);
                    let error = actual
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max);
                    eprintln!(
                        "{} {key}: {} components, error {error:e}",
                        case["name"],
                        actual.len()
                    );
                    assert!(error < 1e-12, "{} {key}: {error:e}", case["name"]);
                }
            }
        }
    }
}
