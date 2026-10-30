// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Exact spin-orbital density cumulants from CI ensembles, through rank three.
//! Antisymmetric tuple storage avoids materializing (2*N_active)^6 entries.

use crate::{Matrix, error::HfError, fci::string::DeterminantSpace};
use std::collections::{BTreeMap, HashSet};

pub const DEFAULT_CUMULANT_MEMORY_BUDGET: usize = 256 * 1024 * 1024;

impl crate::post_hf::casscf_engine::CasResult {
    /// Exact ACTIVE spin-orbital cumulants of the saved state-average ensemble.
    /// Indices refer to active orbitals in the returned CAS MO coefficient order.
    pub fn exact_active_cumulants(
        &self,
        memory_budget: usize,
    ) -> Result<ExactDensityCumulants, HfError> {
        let weights: Vec<_> = self
            .output
            .states
            .iter()
            .map(|state| state.weight)
            .collect();
        cas_cumulants(self, &weights, memory_budget)
    }
    /// Exact ACTIVE cumulants of one saved CI root, independently of SA weights.
    pub fn exact_root_cumulants(
        &self,
        root: usize,
        memory_budget: usize,
    ) -> Result<ExactDensityCumulants, HfError> {
        if root >= self.states.len() {
            return Err(HfError::InvalidInput(
                "CAS cumulant root is unavailable".into(),
            ));
        }
        let mut weights = vec![0.; self.states.len()];
        weights[root] = 1.;
        cas_cumulants(self, &weights, memory_budget)
    }
}
fn cas_cumulants(
    cas: &crate::post_hf::casscf_engine::CasResult,
    weights: &[f64],
    budget: usize,
) -> Result<ExactDensityCumulants, HfError> {
    let n = cas.active_space.active.len();
    let (na, nb) = (cas.active_space.n_alpha, cas.active_space.n_beta);
    if n > 64 || na > n || nb > n || weights.len() != cas.states.len() || weights.is_empty() {
        return Err(HfError::InvalidInput(
            "invalid saved CAS state/active-space dimensions".into(),
        ));
    }
    let count =
        |k: usize| (0..k.min(n - k)).fold(1u128, |v, i| v * (n - i) as u128 / (i + 1) as u128);
    let (nsa, nsb) = (count(na), count(nb));
    let dimension = nsa * nsb;
    if cas
        .states
        .iter()
        .any(|s| s.vector.len() as u128 != dimension)
    {
        return Err(HfError::InvalidInput(
            "saved CAS CI dimensions do not match active space".into(),
        ));
    }
    let string_bytes = 8 * (nsa + nsb);
    if string_bytes >= budget as u128 {
        return Err(HfError::Unsupported {
            method: "CAS exact cumulants".into(),
            detail: "spin strings exceed memory budget".into(),
        });
    }
    let space = DeterminantSpace::new(n, na, nb);
    let members: Vec<_> = cas
        .states
        .iter()
        .zip(weights)
        .map(|(state, &weight)| CiEnsembleMember {
            space: &space,
            coefficients: &state.vector,
            weight,
        })
        .collect();
    exact_density_cumulants(&members, budget - string_bytes as usize)
}

pub struct CiEnsembleMember<'a> {
    pub space: &'a DeterminantSpace,
    pub coefficients: &'a [f64],
    pub weight: f64,
}

/// Gamma[I,J]=<a^+_i1 ... a^+_ik a_jk ... a_j1>, I,J strictly increasing.
/// Public spin-orbital indices are interleaved alpha/beta: 2*p+spin.
#[derive(Clone, Debug)]
pub struct AntisymmetricDensity {
    n_spin: usize,
    order: usize,
    coefficients: Matrix,
}
fn choose(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    (0..k).fold(1usize, |v, i| v * (n - i) / (i + 1))
}
fn address(n: usize, indices: &[usize]) -> usize {
    let mut offset = 0;
    let mut start = 0;
    for (position, &p) in indices.iter().enumerate() {
        for skipped in start..p {
            offset += choose(n - skipped - 1, indices.len() - position - 1);
        }
        start = p + 1;
    }
    offset
}
fn tuples(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn visit(n: usize, k: usize, start: usize, tuple: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if tuple.len() == k {
            out.push(tuple.clone());
            return;
        }
        for p in start..n {
            tuple.push(p);
            visit(n, k, p + 1, tuple, out);
            tuple.pop();
        }
    }
    let mut out = Vec::with_capacity(choose(n, k));
    visit(n, k, 0, &mut Vec::new(), &mut out);
    out
}
fn sorted(indices: &[usize]) -> Option<([usize; 3], f64)> {
    let mut out = [0usize; 3];
    out[..indices.len()].copy_from_slice(indices);
    let mut sign = 1.;
    for p in 0..indices.len() {
        for q in p + 1..indices.len() {
            if out[p] == out[q] {
                return None;
            }
            if out[p] > out[q] {
                sign = -sign;
            }
        }
    }
    out[..indices.len()].sort_unstable();
    Some((out, sign))
}
impl AntisymmetricDensity {
    pub fn n_spin(&self) -> usize {
        self.n_spin
    }
    pub fn order(&self) -> usize {
        self.order
    }
    pub fn coefficients(&self) -> &Matrix {
        &self.coefficients
    }
    pub fn element(&self, bra: &[usize], ket: &[usize]) -> Result<f64, HfError> {
        if bra.len() != self.order
            || ket.len() != self.order
            || bra.iter().chain(ket).any(|&p| p >= self.n_spin)
        {
            return Err(HfError::InvalidInput(
                "invalid antisymmetric density indices".into(),
            ));
        }
        Ok(self.value(bra, ket))
    }
    fn value(&self, bra: &[usize], ket: &[usize]) -> f64 {
        let Some((left, sl)) = sorted(bra) else {
            return 0.;
        };
        let Some((right, sr)) = sorted(ket) else {
            return 0.;
        };
        sl * sr
            * self.coefficients[(
                address(self.n_spin, &left[..self.order]),
                address(self.n_spin, &right[..self.order]),
            )]
    }
    /// Frobenius norm of the full tensor, including antisymmetric permutations.
    pub fn full_norm(&self) -> f64 {
        let multiplicity = match self.order {
            1 => 1.,
            2 => 2.,
            3 => 6.,
            _ => unreachable!(),
        };
        multiplicity
            * self
                .coefficients
                .data
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct ExactDensityCumulants {
    pub gamma1: Matrix,
    pub gamma2: AntisymmetricDensity,
    pub lambda2: AntisymmetricDensity,
    pub lambda3: AntisymmetricDensity,
    pub n_electrons: usize,
}
fn determinant3(g: &Matrix, a: &[usize], b: &[usize]) -> f64 {
    g[(a[0], b[0])] * (g[(a[1], b[1])] * g[(a[2], b[2])] - g[(a[1], b[2])] * g[(a[2], b[1])])
        - g[(a[0], b[1])] * (g[(a[1], b[0])] * g[(a[2], b[2])] - g[(a[1], b[2])] * g[(a[2], b[0])])
        + g[(a[0], b[2])] * (g[(a[1], b[0])] * g[(a[2], b[1])] - g[(a[1], b[1])] * g[(a[2], b[0])])
}
fn disconnected3(g: &Matrix, l2: &AntisymmetricDensity, a: &[usize], b: &[usize]) -> f64 {
    let mut out = determinant3(g, a, b);
    for r in 0..3 {
        for s in 0..3 {
            let ar: [usize; 2] = std::array::from_fn(|k| a[k + usize::from(k >= r)]);
            let bs: [usize; 2] = std::array::from_fn(|k| b[k + usize::from(k >= s)]);
            out += if (r + s) % 2 == 0 { 1. } else { -1. } * g[(a[r], b[s])] * l2.value(&ar, &bs);
        }
    }
    out
}
impl ExactDensityCumulants {
    /// Reconstruct the true three-RDM, without retaining a second rank-six tensor.
    pub fn gamma3(&self, bra: &[usize], ket: &[usize]) -> Result<f64, HfError> {
        let connected = self.lambda3.element(bra, ket)?;
        Ok(connected + disconnected3(&self.gamma1, &self.lambda2, bra, ket))
    }
}

fn validate_member(
    member: &CiEnsembleMember<'_>,
    n: usize,
    electrons: usize,
) -> Result<(f64, usize), HfError> {
    let space = member.space;
    let allowed = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    let valid_strings = |strings: &[crate::fci::string::SpinString], ne: usize| {
        let mut unique = HashSet::new();
        !strings.is_empty()
            && strings
                .iter()
                .all(|s| s.0 & !allowed == 0 && s.n_electrons() == ne && unique.insert(s.0))
    };
    if space.n_orbitals != n
        || space.n_alpha > n
        || space.n_beta > n
        || space.n_alpha + space.n_beta != electrons
        || space
            .alpha_strings
            .len()
            .checked_mul(space.beta_strings.len())
            != Some(member.coefficients.len())
        || !valid_strings(&space.alpha_strings, space.n_alpha)
        || !valid_strings(&space.beta_strings, space.n_beta)
        || !member.weight.is_finite()
        || member.weight < 0.
        || member.coefficients.iter().any(|v| !v.is_finite())
    {
        return Err(HfError::InvalidInput(
            "incompatible CI ensemble/strings/coefficients".into(),
        ));
    }
    let norm = member.coefficients.iter().fold(0f64, |v, &c| v.hypot(c));
    if !norm.is_finite() || norm <= 0. {
        return Err(HfError::InvalidInput(
            "CI vector has zero or nonfinite norm".into(),
        ));
    }
    Ok((
        norm,
        member.coefficients.iter().filter(|&&v| v != 0.).count(),
    ))
}

/// Exact ensemble RDMs are averaged BEFORE computing nonlinear cumulants.
/// Members may have different N_alpha/N_beta but must share N and spatial orbitals.
/// The budget bounds estimated new tensor/sparse workspace, excluding borrowed CI.
pub fn exact_density_cumulants(
    members: &[CiEnsembleMember<'_>],
    memory_budget: usize,
) -> Result<ExactDensityCumulants, HfError> {
    let first = members
        .first()
        .ok_or_else(|| HfError::InvalidInput("empty CI ensemble".into()))?;
    let n = first.space.n_orbitals;
    if n > 64 || first.space.n_alpha > n || first.space.n_beta > n {
        return Err(HfError::InvalidInput(
            "CI ensemble supports at most 64 spatial orbitals".into(),
        ));
    }
    let ns = 2 * n;
    let ne = first.space.n_alpha + first.space.n_beta;
    let sizes = [choose(ns, 1), choose(ns, 2), choose(ns, 3)];
    let tensor_bytes = (sizes[0] as u128).pow(2) * 8
        + (sizes[1] as u128).pow(2) * 16
        + (sizes[2] as u128).pow(2) * 8;
    if tensor_bytes > memory_budget as u128 {
        return Err(HfError::Unsupported {
            method: "exact density cumulants".into(),
            detail: "packed tensors exceed memory budget".into(),
        });
    }
    let mut norms = Vec::with_capacity(members.len());
    let mut maximum_entries = 0u128;
    let mut weights = 0.;
    for member in members {
        let validation_bytes = (member.space.alpha_strings.len() as u128
            + member.space.beta_strings.len() as u128)
            * 64;
        if tensor_bytes + validation_bytes > memory_budget as u128 {
            return Err(HfError::Unsupported {
                method: "exact density cumulants".into(),
                detail: "CI string validation exceeds memory budget".into(),
            });
        }
        let (norm, nonzero) = validate_member(member, n, ne)?;
        norms.push(norm);
        weights += member.weight;
        if member.weight > 0. {
            for rank in 1..=3 {
                maximum_entries = maximum_entries.max(nonzero as u128 * choose(ne, rank) as u128);
            }
        }
    }
    if !weights.is_finite() || weights <= 0. {
        return Err(HfError::InvalidInput(
            "CI ensemble weights must have a positive finite sum".into(),
        ));
    }
    let estimate =
        tensor_bytes + maximum_entries * 128 + sizes.iter().map(|&v| v as u128 * 64).sum::<u128>();
    if estimate > memory_budget as u128 {
        return Err(HfError::Unsupported {
            method: "exact density cumulants".into(),
            detail: "tensor and sparse removal workspace exceed memory budget".into(),
        });
    }
    let mut rdms = Vec::new();
    for rank in 1..=3 {
        let mut coefficients = Matrix::zeros(sizes[rank - 1], sizes[rank - 1]);
        for (member, &norm) in members.iter().zip(&norms) {
            if member.weight == 0. {
                continue;
            }
            let mut residuals: BTreeMap<u128, Vec<(usize, f64)>> = BTreeMap::new();
            for (source, &coefficient) in member.coefficients.iter().enumerate() {
                if coefficient == 0. {
                    continue;
                }
                let (ai, bi) = member.space.linear_to_pair(source);
                let bits = member.space.alpha_strings[ai].0 as u128
                    | ((member.space.beta_strings[bi].0 as u128) << n);
                let occupied: Vec<_> = (0..ns)
                    .filter(|&p| bits & (1u128 << (p / 2 + (p % 2) * n)) != 0)
                    .collect();
                for tuple in tuples(ne, rank) {
                    let indices: Vec<_> = tuple.iter().map(|&i| occupied[i]).collect();
                    let mut remaining = bits;
                    let mut amplitude = coefficient / norm;
                    for &p in &indices {
                        let bit = 1u128 << (p / 2 + (p % 2) * n);
                        if (remaining & (bit - 1)).count_ones() % 2 == 1 {
                            amplitude = -amplitude;
                        }
                        remaining ^= bit;
                    }
                    residuals
                        .entry(remaining)
                        .or_default()
                        .push((address(ns, &indices), amplitude));
                }
            }
            let weight = member.weight / weights;
            for row in residuals.values() {
                for &(i, x) in row {
                    for &(j, y) in row {
                        coefficients[(i, j)] += weight * x * y;
                    }
                }
            }
        }
        rdms.push(AntisymmetricDensity {
            n_spin: ns,
            order: rank,
            coefficients,
        });
    }
    let mut iter = rdms.into_iter();
    let gamma1 = iter
        .next()
        .ok_or_else(|| HfError::Integral("missing first density".into()))?
        .coefficients;
    let gamma2 = iter
        .next()
        .ok_or_else(|| HfError::Integral("missing second density".into()))?;
    let mut lambda3 = iter
        .next()
        .ok_or_else(|| HfError::Integral("missing third density".into()))?;
    let mut lambda2 = gamma2.clone();
    let pairs = tuples(ns, 2);
    let triples = tuples(ns, 3);
    for (i, a) in pairs.iter().enumerate() {
        for (j, b) in pairs.iter().enumerate() {
            lambda2.coefficients[(i, j)] -= gamma1[(a[0], b[0])] * gamma1[(a[1], b[1])]
                - gamma1[(a[0], b[1])] * gamma1[(a[1], b[0])];
        }
    }
    for (i, a) in triples.iter().enumerate() {
        for (j, b) in triples.iter().enumerate() {
            lambda3.coefficients[(i, j)] -= disconnected3(&gamma1, &lambda2, a, b);
        }
    }
    Ok(ExactDensityCumulants {
        gamma1,
        gamma2,
        lambda2,
        lambda3,
        n_electrons: ne,
    })
}
