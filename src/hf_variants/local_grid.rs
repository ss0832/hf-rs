// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Positive atom-centered product quadrature and exact point Coulomb kernels.
//! A common logarithmic AO scale cancels from occupied density ratios. The stored
//! row weight includes the square of that scale; no density floor is introduced.

use crate::integrals::{BasisPairData, coulomb_auxiliary_integrals_flat};
use crate::{Atom, ContractedGaussian, Matrix};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct LocalGridOptions {
    pub radial_points: usize,
    pub polar_points: usize,
    pub azimuthal_points: usize,
    pub radial_scale_bohr: f64,
    pub overlap_tolerance: f64,
    pub workspace_limit_bytes: usize,
    pub kernel_cache_limit_bytes: usize,
}
impl Default for LocalGridOptions {
    fn default() -> Self {
        Self {
            radial_points: 64,
            polar_points: 20,
            azimuthal_points: 40,
            radial_scale_bohr: 1.,
            overlap_tolerance: 1e-6,
            workspace_limit_bytes: 256 * 1024 * 1024,
            kernel_cache_limit_bytes: 256 * 1024 * 1024,
        }
    }
}

pub(super) struct LocalGrid {
    pub points: Vec<[f64; 3]>,
    pub scaled_aos: Matrix,
    pub row_weights: Vec<f64>,
    pub overlap_error: f64,
    pairs: Vec<BasisPairData>,
    kernels: Option<Vec<f64>>,
    pub cached_points: usize,
    r_size: usize,
    pub estimated_workspace_bytes: usize,
}
fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
}
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.; n];
    let mut weights = vec![0.; n];
    for i in 0..n.div_ceil(2) {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let evaluate = |z: f64| {
            let (mut p0, mut p1) = (1., z);
            for k in 2..=n {
                let next = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = next;
            }
            (p1, n as f64 * (z * p1 - p0) / (z * z - 1.))
        };
        for _ in 0..80 {
            let (value, derivative) = evaluate(z);
            let next = z - value / derivative;
            let done = (next - z).abs() <= 4. * f64::EPSILON;
            z = next;
            if done {
                break;
            }
        }
        let (_, derivative) = evaluate(z);
        let w = 2. / ((1. - z * z) * derivative * derivative);
        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }
    (nodes, weights)
}
fn primitive_log(
    b: &ContractedGaussian,
    exponent: f64,
    coefficient: f64,
    point: [f64; 3],
) -> (f64, f64) {
    let mut logarithm = coefficient.abs().ln();
    let mut sign = coefficient.signum();
    for k in 0..3 {
        let d = point[k] - b.center[k];
        logarithm -= exponent * d * d;
        if b.l[k] > 0 {
            logarithm += b.l[k] as f64 * d.abs().ln();
            if !b.l[k].is_multiple_of(2) {
                sign *= d.signum();
            }
        }
    }
    (logarithm, sign)
}
fn ao_values(basis: &[ContractedGaussian], point: [f64; 3]) -> (f64, Vec<f64>) {
    let maximum = basis
        .iter()
        .flat_map(|b| {
            b.primitives
                .iter()
                .map(move |p| primitive_log(b, p.exponent, p.coefficient, point).0)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return (maximum, vec![0.; basis.len()]);
    }
    let values = basis
        .iter()
        .map(|b| {
            b.primitives
                .iter()
                .filter(|p| p.coefficient != 0.)
                .map(|p| {
                    let (log, sign) = primitive_log(b, p.exponent, p.coefficient, point);
                    sign * (log - maximum).exp()
                })
                .sum()
        })
        .collect();
    (maximum, values)
}
fn partition(point: [f64; 3], owner: usize, atoms: &[Atom], separations: &Matrix) -> f64 {
    let distances: Vec<_> = atoms.iter().map(|a| distance(point, a.position)).collect();
    let mut logs = vec![0.; atoms.len()];
    for a in 0..atoms.len() {
        for b in 0..a {
            let mut mu = ((distances[a] - distances[b]) / separations[(a, b)]).clamp(-1., 1.);
            for _ in 0..3 {
                mu = 0.5 * mu * (3. - mu * mu);
            }
            mu = mu.clamp(-1., 1.);
            logs[a] += (0.5 * (1. - mu)).ln();
            logs[b] += (0.5 * (1. + mu)).ln();
        }
    }
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (logs[owner] - maximum).exp() / logs.iter().map(|v| (v - maximum).exp()).sum::<f64>()
}
fn point_kernel(pair: &BasisPairData, point: [f64; 3], r: &mut [f64]) -> f64 {
    let mut sum = 0.;
    for p in &pair.primitives {
        let n = p.t_max + p.u_max + p.v_max;
        let sv = n + 1;
        let su = (p.v_max + 1) * sv;
        let st = (p.u_max + 1) * su;
        let size = (p.t_max + 1) * st;
        coulomb_auxiliary_integrals_flat(
            p.t_max,
            p.u_max,
            p.v_max,
            p.p,
            std::array::from_fn(|k| p.p_coord[k] - point[k]),
            &mut r[..size],
        );
        let mut value = 0.;
        for t in 0..=p.t_max {
            for u in 0..=p.u_max {
                for v in 0..=p.v_max {
                    value += p.ex[t] * p.ey[u] * p.ez[v] * r[t * st + u * su + v * sv];
                }
            }
        }
        sum += p.coeff * 2. * std::f64::consts::PI * p.inv_p * value;
    }
    sum
}

impl LocalGrid {
    pub fn new(
        atoms: &[Atom],
        basis: &[ContractedGaussian],
        frame: &Matrix,
        overlap: &Matrix,
        o: &LocalGridOptions,
    ) -> Result<Self, String> {
        if atoms.is_empty()
            || atoms
                .iter()
                .any(|a| a.position.iter().any(|v| !v.is_finite()))
            || basis.is_empty()
            || frame.rows != basis.len()
            || frame.cols == 0
            || overlap.rows != basis.len()
            || overlap.cols != basis.len()
            || !(8..=512).contains(&o.radial_points)
            || !(4..=128).contains(&o.polar_points)
            || !(8..=256).contains(&o.azimuthal_points)
            || !o.radial_scale_bohr.is_finite()
            || o.radial_scale_bohr <= 0.
            || !o.overlap_tolerance.is_finite()
            || o.overlap_tolerance <= 0.
            || basis.iter().any(|b| {
                b.l.iter()
                    .try_fold(0usize, |s, &l| s.checked_add(l))
                    .is_none_or(|l| l > 8)
                    || b.center.iter().any(|v| !v.is_finite())
                    || b.primitives.is_empty()
                    || b.primitives.iter().any(|p| {
                        !p.exponent.is_finite() || p.exponent <= 0. || !p.coefficient.is_finite()
                    })
            })
        {
            return Err("invalid local-potential grid, AO frame or convergence controls".into());
        }
        let n = basis.len();
        let count = atoms
            .len()
            .checked_mul(o.radial_points)
            .and_then(|v| v.checked_mul(o.polar_points))
            .and_then(|v| v.checked_mul(o.azimuthal_points))
            .ok_or("local grid point count overflow")?;
        let base = count
            .checked_mul(n.checked_add(16).ok_or("local grid dimension overflow")?)
            .and_then(|v| v.checked_mul(8))
            .and_then(|v| v.checked_add(n.checked_mul(1024)?.checked_mul(8)?))
            .and_then(|v| v.checked_add(n.checked_mul(n)?.checked_mul(32 * 8)?))
            .ok_or("local grid workspace overflow")?;
        if base > o.workspace_limit_bytes {
            return Err(format!(
                "local grid row estimate {base} exceeds workspace budget {}",
                o.workspace_limit_bytes
            ));
        }
        // Bound primitive-pair/Hermite storage before constructing any pair data.
        // This also covers dynamically allocated high-angular-momentum axes.
        let pair_upper = || -> Option<usize> {
            let counts: Vec<_> = basis.iter().map(|b| b.primitives.len()).collect();
            let sum = counts.iter().try_fold(0usize, |s, &v| s.checked_add(v))?;
            let squares = counts
                .iter()
                .try_fold(0usize, |s, &v| s.checked_add(v.checked_mul(v)?))?;
            let primitive_pairs = sum.checked_mul(sum)?.checked_add(squares)?.checked_div(2)?;
            let max_l = basis.iter().map(|b| b.l.iter().sum::<usize>()).max()?;
            let axis_bytes = (2 * max_l + 1).checked_mul(3 * 8)?;
            let primitive_bytes = std::mem::size_of::<crate::integrals::PrimitivePairData>()
                .checked_add(axis_bytes)?;
            let pair_count = n.checked_mul(n.checked_add(1)?)?.checked_div(2)?;
            let scratch = (2 * max_l + 1)
                .checked_pow(4)?
                .checked_mul(8)?
                .checked_mul(rayon::current_num_threads())?;
            primitive_pairs
                .checked_mul(primitive_bytes)?
                .checked_add(pair_count.checked_mul(std::mem::size_of::<BasisPairData>())?)?
                .checked_add(scratch)
        }()
        .ok_or("local grid primitive-pair workspace overflow")?;
        let base = base
            .checked_add(pair_upper)
            .ok_or("local grid total workspace overflow")?;
        if base > o.workspace_limit_bytes {
            return Err(format!(
                "local grid primitive-pair estimate {base} exceeds workspace budget {}",
                o.workspace_limit_bytes
            ));
        }
        let mut separations = Matrix::zeros(atoms.len(), atoms.len());
        for a in 0..atoms.len() {
            for b in 0..a {
                let d = distance(atoms[a].position, atoms[b].position);
                if !d.is_finite() || d < 1e-10 {
                    return Err("local grid centers must be finite and distinct".into());
                }
                separations[(a, b)] = d;
                separations[(b, a)] = d;
            }
        }
        let (radial, rw) = gauss_legendre(o.radial_points);
        let (polar, pw) = gauss_legendre(o.polar_points);
        let mut points = Vec::with_capacity(count);
        let mut values = Vec::with_capacity(count * n);
        let mut row_weights = Vec::with_capacity(count);
        for (a, atom) in atoms.iter().enumerate() {
            for (i, &x) in radial.iter().enumerate() {
                let r = o.radial_scale_bohr * (1. + x) / (1. - x);
                let radial_weight = rw[i] * 2. * o.radial_scale_bohr / (1. - x).powi(2) * r * r;
                for (j, &z) in polar.iter().enumerate() {
                    for k in 0..o.azimuthal_points {
                        let phi = 2. * std::f64::consts::PI * k as f64 / o.azimuthal_points as f64;
                        let xy = (1. - z * z).sqrt();
                        let point = [
                            atom.position[0] + r * xy * phi.cos(),
                            atom.position[1] + r * xy * phi.sin(),
                            atom.position[2] + r * z,
                        ];
                        let weight = radial_weight * pw[j] * 2. * std::f64::consts::PI
                            / o.azimuthal_points as f64
                            * partition(point, a, atoms, &separations);
                        if weight == 0. {
                            continue;
                        }
                        let (scale, ao) = ao_values(basis, point);
                        let combined = (weight.ln() + 2. * scale).exp();
                        // Only floating-point zero matrix weight is omitted; no density cutoff is used.
                        if combined == 0. {
                            continue;
                        }
                        if !combined.is_finite() || ao.iter().any(|v| !v.is_finite()) {
                            return Err("nonfinite local-grid AO value/weight".into());
                        }
                        points.push(point);
                        row_weights.push(combined);
                        values.extend(ao);
                    }
                }
            }
        }
        let aos = Matrix::from_vec(points.len(), n, values);
        let mut quadrature_overlap = Matrix::zeros(n, n);
        for start in (0..points.len()).step_by(512) {
            let end = (start + 512).min(points.len());
            let block = Matrix::from_vec(end - start, n, aos.data[start * n..end * n].to_vec());
            let mut weighted = block.transpose();
            for i in 0..n {
                for j in start..end {
                    weighted[(i, j - start)] *= row_weights[j];
                }
            }
            quadrature_overlap = quadrature_overlap.add(&weighted.dot(&block));
        }
        let error = frame
            .transpose()
            .dot(&quadrature_overlap.sub(overlap))
            .dot(frame);
        let overlap_error = error.data.iter().fold(0f64, |m, v| m.max(v.abs()));
        if overlap_error > o.overlap_tolerance {
            return Err(format!(
                "local-grid overlap error {overlap_error:e} exceeds {}; refine the radial/angular grid",
                o.overlap_tolerance
            ));
        }
        let pairs: Vec<_> = (0..n)
            .flat_map(|i| (0..=i).map(move |j| (i, j)))
            .map(|(i, j)| BasisPairData::new(&basis[i], &basis[j]))
            .collect();
        let r_size = pairs
            .iter()
            .flat_map(|p| &p.primitives)
            .map(|p| {
                (p.t_max + 1) * (p.u_max + 1) * (p.v_max + 1) * (p.t_max + p.u_max + p.v_max + 1)
            })
            .max()
            .unwrap_or(1);
        let kernel_bytes = points
            .len()
            .checked_mul(pairs.len())
            .and_then(|v| v.checked_mul(8))
            .ok_or("local grid kernel size overflow")?;
        let mut grid = Self {
            points,
            scaled_aos: aos,
            row_weights,
            overlap_error,
            pairs,
            kernels: None,
            cached_points: 0,
            r_size,
            estimated_workspace_bytes: base,
        };
        let cache_bytes = kernel_bytes
            .min(o.kernel_cache_limit_bytes)
            .min(o.workspace_limit_bytes - base);
        let cached_points = (cache_bytes / (grid.pairs.len() * 8)).min(grid.points.len());
        if cached_points > 0 {
            let mut kernels = vec![0.; cached_points * grid.pairs.len()];
            kernels
                .par_chunks_mut(grid.pairs.len())
                .enumerate()
                .for_each_init(
                    || vec![0.; r_size],
                    |r, (i, row)| {
                        for (value, pair) in row.iter_mut().zip(&grid.pairs) {
                            *value = point_kernel(pair, grid.points[i], r);
                        }
                    },
                );
            grid.kernels = Some(kernels);
            grid.cached_points = cached_points;
            grid.estimated_workspace_bytes += cached_points * grid.pairs.len() * 8;
        }
        Ok(grid)
    }
    pub fn kernel(&self, point: usize, output: &mut Vec<f64>, scratch: &mut Vec<f64>) {
        output.clear();
        if let Some(kernels) = self.kernels.as_ref().filter(|_| point < self.cached_points) {
            output.extend_from_slice(
                &kernels[point * self.pairs.len()..(point + 1) * self.pairs.len()],
            );
        } else {
            scratch.resize(self.r_size, 0.);
            output.extend(
                self.pairs
                    .iter()
                    .map(|p| point_kernel(p, self.points[point], scratch)),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_basis, parse_gaussian94};

    #[test]
    fn gauss_rules_integrate_every_supported_polynomial_moment() {
        for n in [8, 13, 64] {
            let (nodes, weights) = gauss_legendre(n);
            assert!(weights.iter().all(|w| *w > 0. && w.is_finite()));
            for power in 0..2 * n {
                let actual: f64 = nodes
                    .iter()
                    .zip(&weights)
                    .map(|(x, w)| w * x.powi(power as i32))
                    .sum();
                let expected = if power % 2 == 0 {
                    2. / (power + 1) as f64
                } else {
                    0.
                };
                assert!(
                    (actual - expected).abs() < 2e-13,
                    "Gauss n={n}, moment {power}"
                );
            }
        }
    }

    #[test]
    fn every_point_coulomb_component_through_g_matches_independent_integrals() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/lhf_reference.json")).unwrap();
        let model = &data["angular_operators"];
        let atoms: Vec<_> = model["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                Atom::new(
                    a["symbol"].as_str().unwrap(),
                    std::array::from_fn(|k| a["position"][k].as_f64().unwrap()),
                )
            })
            .collect();
        let basis = build_basis(
            &atoms,
            &parse_gaussian94(model["gaussian94"].as_str().unwrap()),
        );
        let mut worst = 0f64;
        let mut entries = 0;
        for operator in model["operators"].as_array().unwrap() {
            let center = std::array::from_fn(|k| operator["center"][k].as_f64().unwrap());
            for i in 0..basis.len() {
                for j in 0..basis.len() {
                    let pair = BasisPairData::new(&basis[i], &basis[j]);
                    let size = pair
                        .primitives
                        .iter()
                        .map(|p| {
                            (p.t_max + 1)
                                * (p.u_max + 1)
                                * (p.v_max + 1)
                                * (p.t_max + p.u_max + p.v_max + 1)
                        })
                        .max()
                        .unwrap_or(1);
                    let value = point_kernel(&pair, center, &mut vec![0.; size]);
                    let error = (value - operator["matrix"][i][j].as_f64().unwrap()).abs();
                    worst = worst.max(error);
                    entries += 1;
                    assert!(error < 2e-12, "point-Coulomb entry {i},{j}: {error:e}");
                }
            }
        }
        eprintln!("point Coulomb: {entries} entries, max error {worst:e}");
    }
}
