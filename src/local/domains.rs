// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! Differential-overlap integrals on a streaming atom-centered product grid.
//! Equal-radius Becke partition weights sum to one; no molecular grid/AO product
//! is retained. Projected AO columns are normalized before the DOI evaluation.
use crate::basis::{Atom, ContractedGaussian};
use crate::ecp::gauss_legendre;
use crate::error::HfError;
use crate::matrix::Matrix;

#[derive(Clone, Debug)]
pub struct DifferentialOverlaps {
    pub occupied: Matrix,
    pub occupied_pao: Matrix,
}
fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
fn partition(
    point: [f64; 3],
    center: usize,
    atoms: &[Atom],
    separation: &Matrix,
    distances: &mut [f64],
    logs: &mut [f64],
) -> f64 {
    for (d, a) in distances.iter_mut().zip(atoms) {
        *d = distance(point, a.position);
    }
    logs.fill(0.0);
    for a in 0..atoms.len() {
        for b in 0..a {
            let mut mu = ((distances[a] - distances[b]) / separation[(a, b)]).clamp(-1.0, 1.0);
            for _ in 0..3 {
                mu = 0.5 * mu * (3.0 - mu * mu);
            }
            mu = mu.clamp(-1.0, 1.0);
            logs[a] += (0.5 * (1.0 - mu)).ln();
            logs[b] += (0.5 * (1.0 + mu)).ln();
        }
    }
    let maximum = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total = logs.iter().map(|x| (x - maximum).exp()).sum::<f64>();
    (logs[center] - maximum).exp() / total
}
pub fn differential_overlaps(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    c: &Matrix,
    q: &Matrix,
    s: &Matrix,
    radial: usize,
    angular: usize,
) -> Result<DifferentialOverlaps, HfError> {
    let n = basis.len();
    if atoms.is_empty()
        || n == 0
        || c.rows != n
        || q.rows != n
        || q.cols != n
        || s.rows != n
        || s.cols != n
        || radial < 8
        || angular < 8
        || radial > 256
        || angular > 128
        || c.data
            .iter()
            .chain(&q.data)
            .chain(&s.data)
            .any(|x| !x.is_finite())
    {
        return Err(HfError::InvalidInput(
            "invalid DOI grid/basis/coefficient dimensions".into(),
        ));
    }
    let mut separation = Matrix::zeros(atoms.len(), atoms.len());
    for a in 0..atoms.len() {
        for b in 0..a {
            let d = distance(atoms[a].position, atoms[b].position);
            if d < 1e-10 {
                return Err(HfError::InvalidInput("DOI grid centers coincide".into()));
            }
            separation[(a, b)] = d;
            separation[(b, a)] = d;
        }
    }
    let metric = q.transpose().dot(s).dot(q);
    let mut paos = q.clone();
    for u in 0..n {
        let norm = metric[(u, u)].max(0.0).sqrt();
        for mu in 0..n {
            paos[(mu, u)] = if norm > 1e-10 {
                paos[(mu, u)] / norm
            } else {
                0.0
            };
        }
    }
    let mut occupied = Matrix::zeros(c.cols, c.cols);
    let mut occupied_pao = Matrix::zeros(c.cols, n);
    let (xr, wr) = gauss_legendre(radial);
    let (xt, wt) = gauss_legendre(angular);
    let nphi = 2 * angular;
    let mut values = Vec::with_capacity(512 * n);
    let mut weights = Vec::with_capacity(512);
    let mut distances = vec![0.0; atoms.len()];
    let mut logs = vec![0.0; atoms.len()];
    let accumulate = |values: &mut Vec<f64>,
                      weights: &mut Vec<f64>,
                      occupied: &mut Matrix,
                      occupied_pao: &mut Matrix| {
        if weights.is_empty() {
            return;
        }
        let ao = Matrix::from_vec(weights.len(), n, std::mem::take(values));
        let mut lmo = ao.dot(c);
        let mut pao = ao.dot(&paos);
        let mut weighted = Matrix::zeros(c.cols, weights.len());
        for p in 0..weights.len() {
            for i in 0..c.cols {
                let square = lmo[(p, i)].powi(2);
                lmo[(p, i)] = square;
                weighted[(i, p)] = weights[p] * square;
            }
            for u in 0..n {
                pao[(p, u)] = pao[(p, u)].powi(2);
            }
        }
        *occupied = occupied.add(&weighted.dot(&lmo));
        *occupied_pao = occupied_pao.add(&weighted.dot(&pao));
        *values = ao.data;
        values.clear();
        weights.clear();
    };
    for (center, atom) in atoms.iter().enumerate() {
        for ir in 0..radial {
            let r = (1.0 + xr[ir]) / (1.0 - xr[ir]);
            let rw = wr[ir] * 2.0 / (1.0 - xr[ir]).powi(2) * r * r;
            for it in 0..angular {
                let z = xt[it];
                let xy = (1.0 - z * z).sqrt();
                let w = rw * wt[it] * 2.0 * std::f64::consts::PI / nphi as f64;
                for iphi in 0..nphi {
                    let phi = 2.0 * std::f64::consts::PI * iphi as f64 / nphi as f64;
                    let point = [
                        atom.position[0] + r * xy * phi.cos(),
                        atom.position[1] + r * xy * phi.sin(),
                        atom.position[2] + r * z,
                    ];
                    let weight =
                        w * partition(point, center, atoms, &separation, &mut distances, &mut logs);
                    weights.push(weight);
                    for cg in basis {
                        let d = [
                            point[0] - cg.center[0],
                            point[1] - cg.center[1],
                            point[2] - cg.center[2],
                        ];
                        let r2 = d.iter().map(|v| v * v).sum::<f64>();
                        let polynomial = d[0].powi(cg.l[0] as i32)
                            * d[1].powi(cg.l[1] as i32)
                            * d[2].powi(cg.l[2] as i32);
                        values.push(
                            polynomial
                                * cg.primitives
                                    .iter()
                                    .map(|p| p.coefficient * (-p.exponent * r2).exp())
                                    .sum::<f64>(),
                        );
                    }
                    if weights.len() == 512 {
                        accumulate(&mut values, &mut weights, &mut occupied, &mut occupied_pao);
                    }
                }
            }
        }
    }
    accumulate(&mut values, &mut weights, &mut occupied, &mut occupied_pao);
    for value in occupied.data.iter_mut().chain(&mut occupied_pao.data) {
        *value = value.max(0.0).sqrt();
    }
    Ok(DifferentialOverlaps {
        occupied,
        occupied_pao,
    })
}

/// Modified Mulliken populations: distribute the two-center overlap population
/// according to squared orbital coefficients, then select whole atomic blocks.
pub fn modified_mulliken_populations(
    c: &Matrix,
    s: &Matrix,
    basis: &[ContractedGaussian],
    n_atoms: usize,
) -> Result<Matrix, HfError> {
    if c.rows != basis.len()
        || s.rows != c.rows
        || s.cols != c.rows
        || basis.iter().any(|b| b.atom_index >= n_atoms)
    {
        return Err(HfError::InvalidInput(
            "invalid modified Mulliken dimensions/atom indices".into(),
        ));
    }
    let mut populations = Matrix::zeros(c.cols, n_atoms);
    for i in 0..c.cols {
        for mu in 0..c.rows {
            populations[(i, basis[mu].atom_index)] += c[(mu, i)].powi(2) * s[(mu, mu)];
            for nu in 0..mu {
                let a = c[(mu, i)].powi(2) * s[(mu, mu)];
                let b = c[(nu, i)].powi(2) * s[(nu, nu)];
                let overlap_population = 2.0 * c[(mu, i)] * c[(nu, i)] * s[(mu, nu)];
                if a + b > 0.0 {
                    populations[(i, basis[mu].atom_index)] += overlap_population * a / (a + b);
                    populations[(i, basis[nu].atom_index)] += overlap_population * b / (a + b);
                }
            }
        }
    }
    Ok(populations)
}
