// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! General real spin-orbital CI operators, including transformed Hamiltonians.
//! No Coulomb eightfold symmetry is imposed on the two-body coefficients.

use super::{
    engine::{FciHamiltonian, FciSolution, FciSolverOptions},
    string::DeterminantSpace,
};
use crate::{
    Matrix, error::HfError, mo_integrals::MoEriTensor, post_hf::mr_dsrg::imsrg2::No2bOperator,
};
use std::collections::HashMap;

pub(crate) fn checked_space_dimension(space: &DeterminantSpace) -> Result<usize, HfError> {
    let n = space.n_orbitals;
    if n > 64 || space.n_alpha > n || space.n_beta > n {
        return Err(HfError::InvalidInput("invalid CI spin populations".into()));
    }
    let count =
        |k: usize| (0..k.min(n - k)).fold(1u128, |v, i| v * (n - i) as u128 / (i + 1) as u128);
    let allowed = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
    for (strings, population) in [
        (&space.alpha_strings, space.n_alpha),
        (&space.beta_strings, space.n_beta),
    ] {
        if strings.len() as u128 != count(population)
            || strings.windows(2).any(|x| x[0].0 >= x[1].0)
            || strings
                .iter()
                .any(|s| s.0 & !allowed != 0 || s.n_electrons() != population)
        {
            return Err(HfError::InvalidInput(
                "CI spin strings must be complete, canonical and unique".into(),
            ));
        }
    }
    space
        .alpha_strings
        .len()
        .checked_mul(space.beta_strings.len())
        .ok_or_else(|| HfError::InvalidInput("CI dimension overflow".into()))
}

fn validate(op: &No2bOperator) -> Result<(), HfError> {
    let n = op.one.rows;
    if n == 0 {
        if op.one.cols != 0 || !op.one.data.is_empty() || !op.two.is_empty() || !op.zero.is_finite()
        {
            return Err(HfError::InvalidInput("invalid scalar CI operator".into()));
        }
        return Ok(());
    }
    op.validate(&vec![false; n])?;
    for p in 0..n {
        for q in 0..n {
            if (op.one[(p, q)] - op.one[(q, p)]).abs() > 1e-10 {
                return Err(HfError::InvalidInput(
                    "CI operator must be Hermitian".into(),
                ));
            }
            for r in 0..n {
                for s in 0..n {
                    let at = |p, q, r, s| op.two[((p * n + q) * n + r) * n + s];
                    if (at(p, q, r, s) - at(r, s, p, q)).abs() > 1e-10
                        || (at(p, q, r, s) + at(q, p, r, s)).abs() > 1e-10
                        || (at(p, q, r, s) + at(p, q, s, r)).abs() > 1e-10
                    {
                        return Err(HfError::InvalidInput("CI operator two-body coefficients must be Hermitian and pair-antisymmetric".into()));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Project a bare operator onto occupied core x active Fock space x empty external.
/// Active indices preserve caller order. Coefficients remain general spin tensors.
pub fn contract_core(
    bare: &No2bOperator,
    core: &[usize],
    active: &[usize],
) -> Result<No2bOperator, HfError> {
    validate(bare)?;
    let n = bare.one.rows;
    let mut used = vec![false; n];
    for &p in core.iter().chain(active) {
        if p >= n || used[p] {
            return Err(HfError::InvalidInput(
                "invalid or overlapping core/active map".into(),
            ));
        }
        used[p] = true;
    }
    let get = |p, q, r, s| bare.two[((p * n + q) * n + r) * n + s];
    let na = active.len();
    let mut zero = bare.zero;
    for (ii, &i) in core.iter().enumerate() {
        zero += bare.one[(i, i)];
        for &j in &core[..ii] {
            zero += get(i, j, i, j);
        }
    }
    let mut one = Matrix::zeros(na, na);
    let mut two = vec![0.; na.pow(4)];
    for (p, &a) in active.iter().enumerate() {
        for (q, &b) in active.iter().enumerate() {
            one[(p, q)] = bare.one[(a, b)] + core.iter().map(|&i| get(a, i, b, i)).sum::<f64>();
            for (r, &c) in active.iter().enumerate() {
                for (s, &d) in active.iter().enumerate() {
                    two[((p * na + q) * na + r) * na + s] = get(a, b, c, d);
                }
            }
        }
    }
    let out = No2bOperator { zero, one, two };
    validate(&out)?;
    Ok(out)
}

struct Term {
    rank: usize,
    create: [usize; 2],
    annihilate: [usize; 2],
    value: f64,
}
fn change(bits: u128, p: usize, create: bool) -> Option<(u128, f64)> {
    let bit = 1u128 << p;
    if (bits & bit == 0) != create {
        return None;
    }
    Some((
        bits ^ bit,
        if (bits & (bit - 1)).count_ones().is_multiple_of(2) {
            1.
        } else {
            -1.
        },
    ))
}

pub struct SpinOrbitalHamiltonian<'a> {
    operator: &'a No2bOperator,
    space: &'a DeterminantSpace,
    bits: Vec<u128>,
    addresses: HashMap<u128, usize>,
    terms: Vec<Term>,
    memory_budget: usize,
    workspace_estimate: usize,
}
impl<'a> SpinOrbitalHamiltonian<'a> {
    pub fn new(
        operator: &'a No2bOperator,
        space: &'a DeterminantSpace,
        memory_budget: usize,
    ) -> Result<Self, HfError> {
        validate(operator)?;
        let dim = checked_space_dimension(space)?;
        let n = operator.one.rows;
        if n != 2 * space.n_orbitals {
            return Err(HfError::InvalidInput(
                "spin operator/CI dimensions disagree".into(),
            ));
        }
        let np = n * n.saturating_sub(1) / 2;
        let bytes = 80u128 * dim as u128 + 96 * ((n * n + np * np) as u128);
        if bytes > memory_budget as u128 {
            return Err(HfError::Unsupported {
                method: "Spin-orbital CI".into(),
                detail: "term/address workspace exceeds memory budget".into(),
            });
        }
        let orbital = |p: usize| p / 2 + (p % 2) * space.n_orbitals;
        let mut terms = Vec::new();
        for p in 0..n {
            for q in 0..n {
                let value = operator.one[(p, q)];
                if value == 0. {
                    continue;
                }
                if p % 2 != q % 2 {
                    return Err(HfError::Unsupported {
                        method: "Spin-orbital CI".into(),
                        detail: "operator changes alpha/beta populations".into(),
                    });
                }
                terms.push(Term {
                    rank: 1,
                    create: [orbital(p), 0],
                    annihilate: [orbital(q), 0],
                    value,
                });
            }
        }
        for p in 0..n {
            for q in p + 1..n {
                for r in 0..n {
                    for s in r + 1..n {
                        let value = operator.two[((p * n + q) * n + r) * n + s];
                        if value == 0. {
                            continue;
                        }
                        if p % 2 + q % 2 != r % 2 + s % 2 {
                            return Err(HfError::Unsupported {
                                method: "Spin-orbital CI".into(),
                                detail: "operator changes alpha/beta populations".into(),
                            });
                        }
                        terms.push(Term {
                            rank: 2,
                            create: [orbital(p), orbital(q)],
                            annihilate: [orbital(r), orbital(s)],
                            value,
                        });
                    }
                }
            }
        }
        let bits: Vec<_> = space
            .alpha_strings
            .iter()
            .flat_map(|a| {
                space
                    .beta_strings
                    .iter()
                    .map(move |b| a.0 as u128 | ((b.0 as u128) << space.n_orbitals))
            })
            .collect();
        let addresses = bits.iter().enumerate().map(|(i, &b)| (b, i)).collect();
        Ok(Self {
            operator,
            space,
            bits,
            addresses,
            terms,
            memory_budget,
            workspace_estimate: bytes as usize,
        })
    }
    pub fn apply(&self, c: &[f64]) -> Result<Vec<f64>, HfError> {
        if c.len() != self.bits.len() || c.iter().any(|x| !x.is_finite()) {
            return Err(HfError::InvalidInput(
                "invalid spin-orbital CI vector".into(),
            ));
        }
        let mut out: Vec<_> = c.iter().map(|x| self.operator.zero * x).collect();
        for (&bits, &amplitude) in self.bits.iter().zip(c) {
            if amplitude == 0. {
                continue;
            }
            for term in &self.terms {
                let mut state = Some((bits, 1.));
                for &p in &term.annihilate[..term.rank] {
                    state = state.and_then(|(b, s)| change(b, p, false).map(|(b, t)| (b, s * t)));
                }
                for &p in term.create[..term.rank].iter().rev() {
                    state = state.and_then(|(b, s)| change(b, p, true).map(|(b, t)| (b, s * t)));
                }
                if let Some((b, sign)) = state {
                    let target = self.addresses.get(&b).ok_or_else(|| {
                        HfError::Integral("missing spin-orbital CI target".into())
                    })?;
                    out[*target] += sign * term.value * amplitude;
                }
            }
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(HfError::Integral("nonfinite spin-orbital CI action".into()));
        }
        Ok(out)
    }
    pub fn diagonal(&self) -> Vec<f64> {
        self.bits
            .iter()
            .map(|&bits| {
                self.operator.zero
                    + self
                        .terms
                        .iter()
                        .filter(|t| {
                            t.create[..t.rank] == t.annihilate[..t.rank]
                                && t.create[..t.rank].iter().all(|&p| bits & (1u128 << p) != 0)
                        })
                        .map(|t| t.value)
                        .sum::<f64>()
            })
            .collect()
    }
    pub fn solve(&self, options: &FciSolverOptions) -> Result<FciSolution, HfError> {
        let dim = self.bits.len();
        let n = self.space.n_orbitals;
        if options.n_roots == 0
            || options.n_roots > dim
            || options.max_iterations == 0
            || !options.residual_tol.is_finite()
            || options.residual_tol <= 0.
        {
            return Err(HfError::InvalidInput(
                "invalid spin-orbital CI solver controls".into(),
            ));
        }
        let subspace = options.max_subspace.max(4 * options.n_roots).min(dim);
        let ci_bytes = if dim <= options.dense_threshold {
            64u128 * (dim as u128).pow(2) + 64 * dim as u128
        } else {
            8 * dim as u128 * (2 * subspace + 8) as u128
        };
        let spin_bytes = 128
            * ((self.space.alpha_strings.len() + self.space.beta_strings.len()) * n * n) as u128;
        if ci_bytes + spin_bytes + self.workspace_estimate as u128
            > self.memory_budget.min(options.memory_budget) as u128
        {
            return Err(HfError::Unsupported {
                method: "Spin-orbital CI".into(),
                detail: "eigensolver/spin workspace exceeds memory budget".into(),
            });
        }
        let h = Matrix::zeros(n, n);
        let np = n * (n + 1) / 2;
        let eri = MoEriTensor {
            n_basis: n,
            n_pairs: np,
            data: vec![0.; np * np],
        };
        let spin = FciHamiltonian::new(&h, &eri, self.space.n_alpha, self.space.n_beta, 0.)?;
        let bound =
            self.terms.iter().map(|t| t.value.abs()).sum::<f64>() + self.operator.zero.abs();
        spin.solve_with_action(
            options,
            |c| self.apply(c),
            || self.diagonal(),
            1. + 2. * bound,
        )
    }
}
