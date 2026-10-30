// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Integral-factorized Fdiag MRPT2. Excitations are streamed in fixed core-hole /
//! external-particle sectors. Only one sector and active-space operators are
//! stored; no full-space four-index integral, amplitude or GNO tensor is built.

use super::{
    dsrg::{dsrg_regularized_inv_denom, dsrg_residual_weight},
    gno::{GnoAlgebra, GnoReference, pair, triple},
    imsrg2::No2bOperator,
};
use crate::{Matrix, df::DfTensor, error::HfError};

fn failure(detail: impl Into<String>) -> HfError {
    HfError::PostHf {
        method: "THREE-DSRG-MRPT2".into(),
        detail: detail.into(),
    }
}

/// The correction is a bare active-space operator, suitable for an effective
/// CAS diagonalization. It excludes the physical CAS Hamiltonian itself.
pub struct ThreeIndexCorrection {
    pub energy: f64,
    pub active_operator: No2bOperator,
    pub sectors: usize,
    pub peak_sector_terms: usize,
}

#[derive(Clone, Copy)]
struct Term {
    rank: usize,
    bra: [usize; 2],
    ket: [usize; 2],
    value: f64,
}

struct Kernel<'a, 'r> {
    factors: &'a DfTensor,
    fock: &'a Matrix,
    reference: &'a GnoReference<'r>,
    active: Vec<usize>,
    classes: Vec<u8>,
    flow: f64,
}
impl Kernel<'_, '_> {
    fn integral(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let mut v = 0.;
        if p % 2 == r % 2 && q % 2 == s % 2 {
            v += self.factors.eri(p / 2, r / 2, q / 2, s / 2);
        }
        if p % 2 == s % 2 && q % 2 == r % 2 {
            v -= self.factors.eri(p / 2, s / 2, q / 2, r / 2);
        }
        v
    }
    fn gap(&self, a: usize, b: usize, i: usize, j: usize) -> f64 {
        self.fock[(i, i)] + self.fock[(j, j)] - self.fock[(a, a)] - self.fock[(b, b)]
    }
    fn double_amplitude(&self, a: usize, b: usize, i: usize, j: usize) -> f64 {
        if a == b || i == j || [a, b, i, j].iter().all(|&p| self.classes[p] == 1) {
            return 0.;
        }
        self.integral(a, b, i, j) * dsrg_regularized_inv_denom(self.gap(a, b, i, j), self.flow)
    }
    fn single(&self, a: usize, i: usize) -> Result<(f64, f64), HfError> {
        let mut coupling = 0.;
        for &r in &self.active {
            for &s in &self.active {
                coupling += (self.fock[(r, r)] - self.fock[(s, s)])
                    * self.double_amplitude(a, r, i, s)
                    * self.reference.density(&[r], &[s])?;
            }
        }
        let gap = self.fock[(i, i)] - self.fock[(a, a)];
        let source = dsrg_residual_weight(gap, self.flow);
        Ok((
            (self.fock[(a, i)] + coupling) * dsrg_regularized_inv_denom(gap, self.flow),
            self.fock[(a, i)] * (1. + source) + coupling * source,
        ))
    }
    /// Expand a GNO monomial into vacuum terms. Its scalar is immaterial in a
    /// commutator. Rank-two contraction signs follow the fermionic CAR.
    fn expand(&self, term: Term, out: &mut Vec<Term>) -> Result<(), HfError> {
        if term.value == 0. {
            return Ok(());
        }
        out.push(term);
        if term.rank == 2 {
            for r in 0..2 {
                for s in 0..2 {
                    let sign = if (r + s) % 2 == 0 { -1. } else { 1. };
                    let value = sign
                        * term.value
                        * self.reference.density(&[term.bra[r]], &[term.ket[s]])?;
                    if value != 0. {
                        out.push(Term {
                            rank: 1,
                            bra: [term.bra[1 - r], 0],
                            ket: [term.ket[1 - s], 0],
                            value,
                        });
                    }
                }
            }
        }
        Ok(())
    }
    fn sector(
        &self,
        core: &[usize],
        external: &[usize],
    ) -> Result<(Vec<Term>, Vec<Term>), HfError> {
        let mut t = Vec::new();
        let mut h = Vec::new();
        for rank in 1..=2 {
            if core.len() > rank || external.len() > rank {
                continue;
            }
            let holes = complete(core, &self.active, rank);
            let particles = complete(external, &self.active, rank);
            for bra in &particles {
                for ket in &holes {
                    let (amplitude, dressed) = if rank == 1 {
                        self.single(bra[0], ket[0])?
                    } else {
                        let gap = self.gap(bra[0], bra[1], ket[0], ket[1]);
                        let v = self.integral(bra[0], bra[1], ket[0], ket[1]);
                        (
                            v * dsrg_regularized_inv_denom(gap, self.flow),
                            v * (1. + dsrg_residual_weight(gap, self.flow)),
                        )
                    };
                    self.expand(
                        Term {
                            rank,
                            bra: *bra,
                            ket: *ket,
                            value: amplitude,
                        },
                        &mut t,
                    )?;
                    self.expand(
                        Term {
                            rank,
                            bra: *ket,
                            ket: *bra,
                            value: dressed,
                        },
                        &mut h,
                    )?;
                }
            }
        }
        Ok((t, h))
    }
}

fn complete(fixed: &[usize], active: &[usize], rank: usize) -> Vec<[usize; 2]> {
    let mut out = Vec::new();
    let mut append = |indices: &[usize]| {
        let mut tuple = [0; 2];
        tuple[..fixed.len()].copy_from_slice(fixed);
        tuple[fixed.len()..rank].copy_from_slice(indices);
        tuple[..rank].sort_unstable();
        out.push(tuple);
    };
    match rank - fixed.len() {
        0 => append(&[]),
        1 => active.iter().for_each(|&p| append(&[p])),
        2 => {
            for (i, &p) in active.iter().enumerate() {
                for &q in &active[i + 1..] {
                    append(&[p, q]);
                }
            }
        }
        _ => unreachable!(),
    }
    out
}

/// Packed vacuum operator after tracing the occupied core and empty external
/// space. Rank three exists only within the active space, never the full MO set.
struct ActiveAccumulator {
    map: Vec<Option<usize>>,
    classes: Vec<u8>,
    n: usize,
    zero: f64,
    one: Matrix,
    two: Matrix,
    three: Matrix,
}
impl ActiveAccumulator {
    fn new(classes: Vec<u8>, active: &[usize]) -> Self {
        let n = active.len();
        let mut map = vec![None; classes.len()];
        for (a, &p) in active.iter().enumerate() {
            map[p] = Some(a);
        }
        Self {
            map,
            classes,
            n,
            zero: 0.,
            one: Matrix::zeros(n, n),
            two: Matrix::zeros(n * n.saturating_sub(1) / 2, n * n.saturating_sub(1) / 2),
            three: Matrix::zeros(
                n * n.saturating_sub(1) * n.saturating_sub(2) / 6,
                n * n.saturating_sub(1) * n.saturating_sub(2) / 6,
            ),
        }
    }
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
        for i in 0..len.saturating_sub(1) {
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
        if contractions == 0 {
            return;
        }
        let rank = len / 2;
        let mut bra = [0; 3];
        let mut ket = [0; 3];
        for p in 0..rank {
            bra[p] = (ops[p] - 1) as usize;
            ket[p] = (-ops[len - 1 - p] - 1) as usize;
        }
        let mut phase = value;
        let mut core = [[0; 3]; 2];
        let mut nc = [0; 2];
        let mut act = [[0; 3]; 2];
        let mut na = [0; 2];
        for (side, indices) in [&bra[..rank], &ket[..rank]].into_iter().enumerate() {
            for (i, &p) in indices.iter().enumerate() {
                if indices[..i].contains(&p) {
                    return;
                }
                match self.classes[p] {
                    2 => return,
                    0 => {
                        if na[side] % 2 == 1 {
                            phase = -phase;
                        }
                        core[side][nc[side]] = p;
                        nc[side] += 1;
                    }
                    _ => {
                        act[side][na[side]] = self.map[p].unwrap();
                        na[side] += 1;
                    }
                }
            }
            for i in 0..nc[side] {
                for j in i + 1..nc[side] {
                    if core[side][i] > core[side][j] {
                        phase = -phase;
                    }
                }
            }
            core[side][..nc[side]].sort_unstable();
            for i in 0..na[side] {
                for j in i + 1..na[side] {
                    if act[side][i] > act[side][j] {
                        phase = -phase;
                    }
                }
            }
            act[side][..na[side]].sort_unstable();
        }
        if nc[0] != nc[1] || core[0][..nc[0]] != core[1][..nc[1]] {
            return;
        }
        let [a, b] = act;
        match na[0] {
            0 => self.zero += phase,
            1 => self.one[(a[0], b[0])] += phase,
            2 => self.two[(pair(self.n, a[0], a[1]), pair(self.n, b[0], b[1]))] += phase,
            3 => self.three[(triple(self.n, a), triple(self.n, b))] += phase,
            _ => unreachable!(),
        }
    }
}

/// Evaluate Fdiag MRPT2 and its Hermitian effective-CAS correction with exact
/// active cumulants. `fock` is the semicanonical spin-orbital generalized Fock.
/// The caller retains the three-index factors; there is no reconstructed ERI.
pub fn factorized_mrpt2(
    factors: &DfTensor,
    fock: &Matrix,
    reference: &GnoReference<'_>,
    flow: f64,
    memory_budget: usize,
) -> Result<ThreeIndexCorrection, HfError> {
    let ns = reference.n_spin();
    let active = reference.active_spin_orbitals();
    let na = active.len();
    let np = factors
        .n_ao
        .checked_mul(factors.n_ao + 1)
        .and_then(|x| x.checked_div(2));
    if ns != 2 * factors.n_ao
        || fock.rows != ns
        || fock.cols != ns
        || fock.data.len() != ns * ns
        || fock.data.iter().any(|v| !v.is_finite())
        || np != Some(factors.b.cols)
        || factors.b.data.len() != factors.b.rows * factors.b.cols
        || factors.b.data.iter().any(|v| !v.is_finite())
        || !flow.is_finite()
        || flow < 0.
    {
        return Err(HfError::InvalidInput(
            "invalid factorized MRPT2 dimensions, factors, Fock or flow".into(),
        ));
    }
    // Active GNO work plus three packed active operators and the largest sector.
    // Factors are borrowed; their storage is nevertheless counted in the budget.
    let active_bytes = if na == 0 {
        0
    } else {
        GnoAlgebra::estimated_workspace_bytes(na)?
    };
    let bytes = (active_bytes as u128) * 4
        + (factors.b.data.len() as u128) * 8
        + 320 * (1 + na as u128).pow(4)
        + (ns as u128).pow(2) * 64;
    if bytes > memory_budget as u128 {
        return Err(failure(
            "three-index / active / sector workspace exceeds memory budget",
        ));
    }
    let classes = reference.orbital_classes();
    for p in 0..ns {
        for q in 0..p {
            if (fock[(p, q)] - fock[(q, p)]).abs() > 1e-10
                || (classes[p] == classes[q] && fock[(p, q)].abs() > 1e-9)
            {
                return Err(failure(
                    "Fdiag requires a Hermitian block-semicanonical generalized Fock",
                ));
            }
        }
    }
    let core: Vec<_> = (0..ns).filter(|&p| classes[p] == 0).collect();
    let external: Vec<_> = (0..ns).filter(|&p| classes[p] == 2).collect();
    let kernel = Kernel {
        factors,
        fock,
        reference,
        active: active.clone(),
        classes: classes.clone(),
        flow,
    };
    let mut acc = ActiveAccumulator::new(classes, &active);
    let mut sectors = 0;
    let mut peak_sector_terms = 0;
    for nc in 0..=2 {
        for nv in 0..=2 {
            if nc == 0 && nv == 0 {
                continue;
            }
            for cs in complete(&[], &core, nc) {
                for vs in complete(&[], &external, nv) {
                    let (t, h) = kernel.sector(&cs[..nc], &vs[..nv])?;
                    sectors += 1;
                    peak_sector_terms = peak_sector_terms.max(t.len() + h.len());
                    for ht in &h {
                        for tt in &t {
                            let value = ht.value * tt.value;
                            if !value.is_finite() {
                                return Err(failure("nonfinite amplitude product"));
                            }
                            acc.product(ht, tt, value);
                            acc.product(tt, ht, -value);
                        }
                    }
                }
            }
        }
    }
    // K = (C + C^dagger)/2, C = [H_tilde(deexcitation), T]. Core and
    // external Gaussian contractions commute with active GNO rank truncation.
    for matrix in [&mut acc.one, &mut acc.two, &mut acc.three] {
        for p in 0..matrix.rows {
            for q in 0..p {
                let v = 0.5 * (matrix[(p, q)] + matrix[(q, p)]);
                matrix[(p, q)] = v;
                matrix[(q, p)] = v;
            }
        }
    }
    let (energy, active_operator) = if na == 0 {
        (
            acc.zero,
            No2bOperator {
                zero: acc.zero,
                one: acc.one,
                two: Vec::new(),
            },
        )
    } else {
        let active_reference = reference.active_reference()?;
        let algebra = GnoAlgebra::new(&active_reference, memory_budget)?;
        let normal = algebra.project(acc.zero, acc.one, acc.two, Some(&acc.three))?;
        (normal.zero, algebra.denormal_order(&normal)?)
    };
    if !energy.is_finite() {
        return Err(failure("nonfinite factorized energy"));
    }
    Ok(ThreeIndexCorrection {
        energy,
        active_operator,
        sectors,
        peak_sector_terms,
    })
}
