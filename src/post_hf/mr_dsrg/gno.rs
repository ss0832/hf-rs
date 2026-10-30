// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Generalized normal ordering and complete rank-two commutators for CI ensembles.
//! Fermionic CAR products are evaluated in vacuum operator space, then normal
//! ordered with exact rank-one/two/three reference densities. No Fock-space
//! matrices or determinant-pair Hamiltonians are constructed in production.

use super::{density_cumulants::ExactDensityCumulants, imsrg2::No2bOperator};
use crate::{Matrix, error::HfError};

#[derive(Clone, Copy)]
enum Orbital {
    Core,
    Active(usize),
    External,
}

/// A fixed occupied core, a correlated active reference and empty external space.
/// Maps refer to individual spin orbitals; no spin averaging is inferred.
pub struct GnoReference<'a> {
    density: &'a ExactDensityCumulants,
    orbitals: Vec<Orbital>,
}
impl<'a> GnoReference<'a> {
    pub fn new(
        density: &'a ExactDensityCumulants,
        n_spin: usize,
        core: &[usize],
        active: &[usize],
    ) -> Result<Self, HfError> {
        let n = density.gamma1.rows;
        if n_spin == 0
            || n_spin > i16::MAX as usize
            || active.len() != n
            || density.gamma1.cols != n
            || density.gamma1.data.len() != n * n
            || density.gamma2.n_spin() != n
            || density.lambda2.n_spin() != n
            || density.lambda3.n_spin() != n
            || density.gamma1.data.iter().any(|v| !v.is_finite())
        {
            return Err(HfError::InvalidInput(
                "incompatible GNO reference dimensions/density".into(),
            ));
        }
        let mut orbitals = vec![Orbital::External; n_spin];
        for &p in core {
            if p >= n_spin || !matches!(orbitals[p], Orbital::External) {
                return Err(HfError::InvalidInput(
                    "duplicate/out-of-range GNO core orbital".into(),
                ));
            }
            orbitals[p] = Orbital::Core;
        }
        for (a, &p) in active.iter().enumerate() {
            if p >= n_spin || !matches!(orbitals[p], Orbital::External) {
                return Err(HfError::InvalidInput(
                    "overlapping/out-of-range GNO active orbital".into(),
                ));
            }
            orbitals[p] = Orbital::Active(a);
        }
        Ok(Self { density, orbitals })
    }
    pub fn active_only(density: &'a ExactDensityCumulants) -> Result<Self, HfError> {
        Self::new(
            density,
            density.gamma1.rows,
            &[],
            &(0..density.gamma1.rows).collect::<Vec<_>>(),
        )
    }
    /// CAS coefficients are already ordered inactive/active/external.
    pub fn from_cas(
        density: &'a ExactDensityCumulants,
        cas: &crate::post_hf::casscf_engine::CasResult,
    ) -> Result<Self, HfError> {
        let core = cas.active_space.inactive.len() * 2;
        let active = cas.active_space.active.len() * 2;
        Self::new(
            density,
            cas.mo_coefficients.cols * 2,
            &(0..core).collect::<Vec<_>>(),
            &(core..core + active).collect::<Vec<_>>(),
        )
    }
    pub(crate) fn active_spin_orbitals(&self) -> Vec<usize> {
        let mut active = vec![0; self.density.gamma1.rows];
        for (p, orbital) in self.orbitals.iter().enumerate() {
            if let Orbital::Active(a) = orbital {
                active[*a] = p;
            }
        }
        active
    }
    pub(crate) fn orbital_classes(&self) -> Vec<u8> {
        self.orbitals
            .iter()
            .map(|p| match p {
                Orbital::Core => 0,
                Orbital::Active(_) => 1,
                Orbital::External => 2,
            })
            .collect()
    }
    pub fn n_spin(&self) -> usize {
        self.orbitals.len()
    }
    pub(crate) fn active_reference(&self) -> Result<Self, HfError> {
        Self::active_only(self.density)
    }
    /// <a^+_bra[0]...a^+_bra[k-1] a_ket[k-1]...a_ket[0]>, k=0..3.
    pub fn density(&self, bra: &[usize], ket: &[usize]) -> Result<f64, HfError> {
        if bra.len() != ket.len()
            || bra.len() > 3
            || bra.iter().chain(ket).any(|&p| p >= self.n_spin())
        {
            return Err(HfError::InvalidInput(
                "invalid GNO reference density indices".into(),
            ));
        }
        let mut phase = 1.;
        let mut cores = [[0usize; 3]; 2];
        let mut acts = [[0usize; 3]; 2];
        let mut nc = [0usize; 2];
        let mut na = [0usize; 2];
        for (side, indices) in [bra, ket].into_iter().enumerate() {
            for (i, &p) in indices.iter().enumerate() {
                if indices[..i].contains(&p) {
                    return Ok(0.);
                }
                match self.orbitals[p] {
                    Orbital::External => return Ok(0.),
                    Orbital::Core => {
                        if na[side] % 2 == 1 {
                            phase = -phase;
                        }
                        cores[side][nc[side]] = p;
                        nc[side] += 1;
                    }
                    Orbital::Active(a) => {
                        acts[side][na[side]] = a;
                        na[side] += 1;
                    }
                }
            }
            for i in 0..nc[side] {
                for j in i + 1..nc[side] {
                    if cores[side][i] > cores[side][j] {
                        phase = -phase;
                    }
                }
            }
            cores[side][..nc[side]].sort_unstable();
        }
        if nc[0] != nc[1] || cores[0][..nc[0]] != cores[1][..nc[1]] {
            return Ok(0.);
        }
        let (a, b) = (&acts[0][..na[0]], &acts[1][..na[1]]);
        let value = match a.len() {
            0 => 1.,
            1 => self.density.gamma1[(a[0], b[0])],
            2 => self.density.gamma2.element(a, b)?,
            3 => self.density.gamma3(a, b)?,
            _ => unreachable!(),
        };
        Ok(phase * value)
    }
}

fn groups(n: usize, k: usize) -> Vec<[usize; 3]> {
    let mut out = Vec::new();
    for p in 0..n {
        if k == 1 {
            out.push([p, 0, 0]);
            continue;
        }
        for q in p + 1..n {
            if k == 2 {
                out.push([p, q, 0]);
                continue;
            }
            for r in q + 1..n {
                out.push([p, q, r]);
            }
        }
    }
    out
}
pub(crate) fn pair(n: usize, p: usize, q: usize) -> usize {
    p * (2 * n - p - 1) / 2 + q - p - 1
}
pub(crate) fn triple(n: usize, a: [usize; 3]) -> usize {
    let mut index = 0;
    for p in 0..a[0] {
        index += (n - p - 1) * (n - p - 2) / 2;
    }
    for q in a[0] + 1..a[1] {
        index += n - q - 1;
    }
    index + a[2] - a[1] - 1
}
fn error(detail: impl Into<String>) -> HfError {
    HfError::PostHf {
        method: "GNO(2)".into(),
        detail: detail.into(),
    }
}

/// Fixed reference contractions reused across commutators and the BCH series.
/// Workspace is polynomial operator space, including a packed rank-three matrix.
pub struct GnoAlgebra {
    n: usize,
    pairs: Vec<[usize; 3]>,
    triples: Vec<[usize; 3]>,
    gamma1: Matrix,
    gamma2: Matrix,
    gamma3: Matrix,
    validation_occupations: Vec<bool>,
}
impl GnoAlgebra {
    /// Conservative new operator/density workspace, excluding borrowed inputs.
    pub fn estimated_workspace_bytes(n: usize) -> Result<usize, HfError> {
        if n == 0 || n > 128 {
            return Err(HfError::InvalidInput(
                "GNO spin dimension must be 1..=128".into(),
            ));
        }
        let np = n * (n - 1) / 2;
        let nt = if n < 3 { 0 } else { n * (n - 1) * (n - 2) / 6 };
        let bytes = 8u128
            * (16 * (n as u128).pow(4)
                + 4 * (n as u128).pow(2)
                + 3 * (np as u128).pow(2)
                + 2 * (nt as u128).pow(2))
            + 64 * (np + nt) as u128
            + 128 * ((n * n) as u128 + (np as u128).pow(2));
        usize::try_from(bytes).map_err(|_| error("GNO workspace size overflow"))
    }
    pub fn new(reference: &GnoReference<'_>, memory_budget: usize) -> Result<Self, HfError> {
        let n = reference.n_spin();
        if Self::estimated_workspace_bytes(n)? > memory_budget {
            return Err(HfError::Unsupported {
                method: "GNO(2)".into(),
                detail: "operator/density workspace estimate exceeds memory budget".into(),
            });
        }
        let np = n * (n - 1) / 2;
        let nt = if n < 3 { 0 } else { n * (n - 1) * (n - 2) / 6 };
        let pairs = groups(n, 2);
        let triples = groups(n, 3);
        let mut gamma1 = Matrix::zeros(n, n);
        let mut gamma2 = Matrix::zeros(np, np);
        let mut gamma3 = Matrix::zeros(nt, nt);
        for p in 0..n {
            for q in 0..n {
                gamma1[(p, q)] = reference.density(&[p], &[q])?;
            }
        }
        for (i, a) in pairs.iter().enumerate() {
            for (j, b) in pairs.iter().enumerate() {
                gamma2[(i, j)] = reference.density(&a[..2], &b[..2])?;
            }
        }
        for (i, a) in triples.iter().enumerate() {
            for (j, b) in triples.iter().enumerate() {
                gamma3[(i, j)] = reference.density(a, b)?;
            }
        }
        if gamma1
            .data
            .iter()
            .chain(&gamma2.data)
            .chain(&gamma3.data)
            .any(|v| !v.is_finite())
        {
            return Err(error("nonfinite reference contraction"));
        }
        Ok(Self {
            n,
            pairs,
            triples,
            gamma1,
            gamma2,
            gamma3,
            validation_occupations: vec![false; n],
        })
    }
    pub(crate) fn validate(&self, a: &No2bOperator) -> Result<(), HfError> {
        a.validate(&self.validation_occupations)?;
        for p in 0..self.n {
            for q in 0..self.n {
                for r in 0..self.n {
                    for s in 0..self.n {
                        let at = |p, q, r, s| a.two[((p * self.n + q) * self.n + r) * self.n + s];
                        if (at(p, q, r, s) + at(q, p, r, s)).abs() > 1e-10
                            || (at(p, q, r, s) + at(p, q, s, r)).abs() > 1e-10
                        {
                            return Err(HfError::InvalidInput(
                                "GNO two-body coefficients must be antisymmetric in each pair"
                                    .into(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
    fn packed(&self, a: &No2bOperator) -> Matrix {
        let mut g = Matrix::zeros(self.pairs.len(), self.pairs.len());
        for (i, &[p, q, _]) in self.pairs.iter().enumerate() {
            for (j, &[r, s, _]) in self.pairs.iter().enumerate() {
                g[(i, j)] = a.two[((p * self.n + q) * self.n + r) * self.n + s];
            }
        }
        g
    }
    fn contract_two(&self, one: &mut Matrix, two: &Matrix, scale: f64) {
        for (i, a) in self.pairs.iter().enumerate() {
            for (j, b) in self.pairs.iter().enumerate() {
                let v = scale * two[(i, j)];
                if v == 0. {
                    continue;
                }
                for r in 0..2 {
                    for s in 0..2 {
                        let sign = if (r + s) % 2 == 0 { 1. } else { -1. };
                        one[(a[1 - r], b[1 - s])] += sign * v * self.gamma1[(a[r], b[s])];
                    }
                }
            }
        }
    }
    /// Convert a physical vacuum-normal-ordered scalar/one/two-body operator.
    pub fn normal_order(&self, bare: &No2bOperator) -> Result<No2bOperator, HfError> {
        self.validate(bare)?;
        self.project(bare.zero, bare.one.clone(), self.packed(bare), None)
    }
    pub fn denormal_order(&self, a: &No2bOperator) -> Result<No2bOperator, HfError> {
        self.validate(a)?;
        let g = self.packed(a);
        let mut one = a.one.clone();
        self.contract_two(&mut one, &g, -1.);
        let zero = a.zero
            - one
                .data
                .iter()
                .zip(&self.gamma1.data)
                .map(|(x, y)| x * y)
                .sum::<f64>()
            - g.data
                .iter()
                .zip(&self.gamma2.data)
                .map(|(x, y)| x * y)
                .sum::<f64>();
        let out = No2bOperator {
            zero,
            one,
            two: a.two.clone(),
        };
        out.validate(&self.validation_occupations)?;
        Ok(out)
    }
    pub(crate) fn project(
        &self,
        mut zero: f64,
        mut one: Matrix,
        mut two: Matrix,
        three: Option<&Matrix>,
    ) -> Result<No2bOperator, HfError> {
        zero += one
            .data
            .iter()
            .zip(&self.gamma1.data)
            .map(|(x, y)| x * y)
            .sum::<f64>()
            + two
                .data
                .iter()
                .zip(&self.gamma2.data)
                .map(|(x, y)| x * y)
                .sum::<f64>();
        self.contract_two(&mut one, &two, 1.);
        if let Some(three) = three {
            zero += three
                .data
                .iter()
                .zip(&self.gamma3.data)
                .map(|(x, y)| x * y)
                .sum::<f64>();
            for (i, a) in self.triples.iter().enumerate() {
                for (j, b) in self.triples.iter().enumerate() {
                    let v = three[(i, j)];
                    if v == 0. {
                        continue;
                    }
                    for r in 0..3 {
                        for s in 0..3 {
                            let sign = if (r + s) % 2 == 0 { 1. } else { -1. };
                            let ar: [usize; 2] =
                                std::array::from_fn(|k| a[k + usize::from(k >= r)]);
                            let bs: [usize; 2] =
                                std::array::from_fn(|k| b[k + usize::from(k >= s)]);
                            let (ar, bs) = (pair(self.n, ar[0], ar[1]), pair(self.n, bs[0], bs[1]));
                            two[(ar, bs)] += sign * v * self.gamma1[(a[r], b[s])];
                            one[(a[r], b[s])] += sign * v * self.gamma2[(ar, bs)];
                        }
                    }
                }
            }
        }
        let mut full = vec![0.; self.n.pow(4)];
        for (i, &[p, q, _]) in self.pairs.iter().enumerate() {
            for (j, &[r, s, _]) in self.pairs.iter().enumerate() {
                for (a, b, c, d, sign) in [
                    (p, q, r, s, 1.),
                    (q, p, r, s, -1.),
                    (p, q, s, r, -1.),
                    (q, p, s, r, 1.),
                ] {
                    full[((a * self.n + b) * self.n + c) * self.n + d] = sign * two[(i, j)];
                }
            }
        }
        let out = No2bOperator {
            zero,
            one,
            two: full,
        };
        out.validate(&self.validation_occupations)?;
        Ok(out)
    }
    fn terms(&self, a: &No2bOperator) -> Vec<Term> {
        let g = self.packed(a);
        let mut one = a.one.clone();
        self.contract_two(&mut one, &g, -1.);
        let mut out = Vec::new();
        for p in 0..self.n {
            for q in 0..self.n {
                if one[(p, q)] != 0. {
                    out.push(Term {
                        rank: 1,
                        bra: [p, 0],
                        ket: [q, 0],
                        value: one[(p, q)],
                    });
                }
            }
        }
        for (i, &[p, q, _]) in self.pairs.iter().enumerate() {
            for (j, &[r, s, _]) in self.pairs.iter().enumerate() {
                if g[(i, j)] != 0. {
                    out.push(Term {
                        rank: 2,
                        bra: [p, q],
                        ket: [r, s],
                        value: g[(i, j)],
                    });
                }
            }
        }
        out
    }
    /// Complete [A,B] projected to GNO ranks 0,1,2; induced rank-three
    /// contributions are contracted with Gamma1/Gamma2/Gamma3 before truncation.
    pub fn commutator(&self, a: &No2bOperator, b: &No2bOperator) -> Result<No2bOperator, HfError> {
        self.validate(a)?;
        self.validate(b)?;
        let (a, b) = (self.terms(a), self.terms(b));
        let mut cre = vec![Vec::new(); self.n];
        let mut ann = vec![Vec::new(); self.n];
        for (index, term) in b.iter().enumerate() {
            for k in 0..term.rank {
                cre[term.bra[k]].push(index);
                ann[term.ket[k]].push(index);
            }
        }
        let mut marks = vec![usize::MAX; b.len()];
        let mut accumulator = VacuumAccumulator {
            n: self.n,
            one: Matrix::zeros(self.n, self.n),
            two: Matrix::zeros(self.pairs.len(), self.pairs.len()),
            three: Matrix::zeros(self.triples.len(), self.triples.len()),
        };
        for (stamp, ta) in a.iter().enumerate() {
            for indices in (0..ta.rank)
                .map(|k| &cre[ta.ket[k]])
                .chain((0..ta.rank).map(|k| &ann[ta.bra[k]]))
            {
                for &index in indices {
                    if marks[index] == stamp {
                        continue;
                    }
                    marks[index] = stamp;
                    let tb = &b[index];
                    let value = ta.value * tb.value;
                    if !value.is_finite() {
                        return Err(error("overflowed vacuum operator product"));
                    }
                    accumulator.product(ta, tb, value);
                    accumulator.product(tb, ta, -value);
                }
            }
        }
        self.project(
            0.,
            accumulator.one,
            accumulator.two,
            Some(&accumulator.three),
        )
    }
    pub fn bch(
        &self,
        h: &No2bOperator,
        a: &No2bOperator,
        tolerance: f64,
        max_order: usize,
    ) -> Result<(No2bOperator, usize), HfError> {
        self.validate(h)?;
        self.validate(a)?;
        if !tolerance.is_finite() || tolerance <= 0. || max_order == 0 || max_order > 1000 {
            return Err(HfError::InvalidInput(
                "invalid GNO BCH tolerance/order".into(),
            ));
        }
        let mut bar = h.clone();
        let mut term = h.clone();
        for k in 1..=max_order {
            term = self.commutator(&term, a)?;
            term.zero /= k as f64;
            term.one.data.iter_mut().for_each(|v| *v /= k as f64);
            term.two.iter_mut().for_each(|v| *v /= k as f64);
            bar = bar.add_scaled(&term, 1.);
            bar.validate(&self.validation_occupations)?;
            if term.norm_max() <= tolerance {
                return Ok((bar, k));
            }
        }
        Err(error("GNO BCH expansion did not converge"))
    }
}

struct Term {
    rank: usize,
    bra: [usize; 2],
    ket: [usize; 2],
    value: f64,
}
struct VacuumAccumulator {
    n: usize,
    one: Matrix,
    two: Matrix,
    three: Matrix,
}
impl VacuumAccumulator {
    fn product(&mut self, a: &Term, b: &Term, value: f64) {
        let mut ops = [0i16; 8];
        let mut k = 0;
        for term in [a, b] {
            for &p in &term.bra[..term.rank] {
                ops[k] = p as i16 + 1;
                k += 1;
            }
            for &p in term.ket[..term.rank].iter().rev() {
                ops[k] = -(p as i16 + 1);
                k += 1;
            }
        }
        self.order(ops, k, value, 0);
    }
    fn order(&mut self, mut ops: [i16; 8], len: usize, value: f64, contractions: usize) {
        for i in 0..len - 1 {
            if ops[i] < 0 && ops[i + 1] > 0 {
                if ops[i] + ops[i + 1] == 0 {
                    let mut reduced = ops;
                    reduced.copy_within(i + 2..len, i);
                    self.order(reduced, len - 2, value, contractions + 1);
                }
                ops.swap(i, i + 1);
                self.order(ops, len, -value, contractions);
                return;
            }
        }
        // The uncontracted highest-rank products cancel pairwise in [A,B].
        if contractions == 0 {
            return;
        }
        let rank = len / 2;
        let mut bra = [0usize; 3];
        let mut ket = [0usize; 3];
        let mut sign = 1.;
        for p in 0..rank {
            bra[p] = (ops[p] - 1) as usize;
            ket[p] = (-ops[len - 1 - p] - 1) as usize;
        }
        for indices in [&mut bra, &mut ket] {
            for i in 0..rank {
                for j in i + 1..rank {
                    if indices[i] == indices[j] {
                        return;
                    }
                    if indices[i] > indices[j] {
                        sign = -sign;
                    }
                }
            }
            indices[..rank].sort_unstable();
        }
        match rank {
            1 => self.one[(bra[0], ket[0])] += sign * value,
            2 => {
                self.two[(pair(self.n, bra[0], bra[1]), pair(self.n, ket[0], ket[1]))] +=
                    sign * value
            }
            3 => self.three[(triple(self.n, bra), triple(self.n, ket))] += sign * value,
            _ => unreachable!(),
        }
    }
}
