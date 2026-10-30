// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Charge-conserving two-site MPS variational solver.
//!
//! A spatial site has states |0>, |alpha>, |beta>, |alpha beta>. The two
//! bond charges count alpha and beta electrons, equivalently (N, 2 Sz).
//! The Hamiltonian is an exact direct-sum MPO of Jordan-Wigner product
//! strings. This deliberately simple, uncompressed MPO is a bounded-memory
//! correctness backend, not a complementary-operator performance backend.
//! No determinant list, full CI vector or full Hamiltonian is constructed.

use crate::error::HfError;
use crate::matrix::Matrix;
use crate::mo_integrals::MoEriTensor;
use crate::solvers::davidson::{DavidsonOptions, try_davidson_solver};
use std::collections::BTreeMap;

const BUDGET: usize = 256 * 1024 * 1024;
type Charge = (usize, usize);
const PHYSICAL: [Charge; 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

/// A signed partial permutation: ket column -> signed (bra row + 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Local([i8; 4]);
const ID: Local = Local([1, 2, 3, 4]);
const PARITY: Local = Local([1, -2, -3, 4]);
impl Local {
    fn multiply(self, rhs: Self) -> Self {
        let mut out = [0; 4];
        for (j, &v) in rhs.0.iter().enumerate() {
            if v != 0 {
                out[j] = v.signum() * self.0[v.unsigned_abs() as usize - 1];
            }
        }
        Self(out)
    }
    fn diagonal(self, s: usize) -> f64 {
        let v = self.0[s];
        if v.unsigned_abs() as usize == s + 1 {
            v.signum() as f64
        } else {
            0.0
        }
    }
}

fn fermion(spin: usize, creation: bool) -> Local {
    let mut out = [0; 4];
    for (s, value) in out.iter_mut().enumerate() {
        if ((s >> spin) & 1 == 0) == creation {
            let t = s ^ (1 << spin);
            let sign = if spin == 1 && s & 1 != 0 { -1 } else { 1 };
            *value = sign * (t + 1) as i8;
        }
    }
    Local(out)
}

/// Operators are supplied in mathematical left-to-right order.
fn product(n: usize, operators: &[(usize, bool)]) -> (f64, Vec<Local>) {
    let mut locals = vec![ID; n];
    for &(orbital, creation) in operators {
        let site = orbital / 2;
        for local in &mut locals[..site] {
            *local = local.multiply(PARITY);
        }
        locals[site] = locals[site].multiply(fermion(orbital % 2, creation));
    }
    let mut sign = 1.0;
    for local in &mut locals {
        let Some(first) = local.0.iter().find(|&&v| v != 0) else {
            return (0.0, locals);
        };
        if *first < 0 {
            sign = -sign;
            for x in &mut local.0 {
                *x = -*x;
            }
        }
    }
    (sign, locals)
}

#[derive(Clone, Debug)]
struct Term {
    coefficient: f64,
    local: Vec<Local>,
}

#[derive(Clone, Debug)]
struct Site {
    left: Vec<Charge>,
    right: Vec<Charge>,
    data: Vec<f64>,
}
impl Site {
    fn new(left: Vec<Charge>, right: Vec<Charge>) -> Self {
        let data = vec![0.0; left.len() * 4 * right.len()];
        Self { left, right, data }
    }
    fn at(&self, a: usize, s: usize, b: usize) -> f64 {
        self.data[(a * 4 + s) * self.right.len() + b]
    }
    fn slice(&self, s: usize) -> Matrix {
        let mut m = Matrix::zeros(self.left.len(), self.right.len());
        for a in 0..m.rows {
            for b in 0..m.cols {
                m[(a, b)] = self.at(a, s, b);
            }
        }
        m
    }
}

fn add(a: Charge, b: Charge) -> Charge {
    (a.0 + b.0, a.1 + b.1)
}
fn memory(words: Option<usize>) -> Result<(), HfError> {
    if words
        .and_then(|n| n.checked_mul(8))
        .is_none_or(|n| n > BUDGET)
    {
        Err(HfError::Unsupported {
            method: "DMRG".into(),
            detail: "MPS/MPO working storage exceeds 256 MiB".into(),
        })
    } else {
        Ok(())
    }
}

fn transfer(site: &Site, env: &Matrix, local: Local, left_to_right: bool) -> Matrix {
    let dim = if left_to_right {
        site.right.len()
    } else {
        site.left.len()
    };
    let mut out = Matrix::zeros(dim, dim);
    for s in 0..4 {
        let v = local.0[s];
        if v == 0 {
            continue;
        }
        let bra = site.slice(v.unsigned_abs() as usize - 1);
        let ket = site.slice(s);
        let m = if left_to_right {
            bra.transpose().dot(env).dot(&ket)
        } else {
            bra.dot(env).dot(&ket.transpose())
        };
        for (x, y) in out.data.iter_mut().zip(m.data) {
            *x += v.signum() as f64 * y;
        }
    }
    out
}

/// An MPS with exact U(1) x U(1) bond sectors and spatial-site dimension four.
#[derive(Clone, Debug)]
pub struct MatrixProductState {
    sites: Vec<Site>,
    target: Charge,
}

fn choose_capped(n: usize, k: usize, cap: usize) -> usize {
    let mut x = 1_u128;
    for i in 0..k.min(n - k) {
        x = x * (n - i) as u128 / (i + 1) as u128;
        if x >= cap as u128 {
            return cap;
        }
    }
    x as usize
}
impl MatrixProductState {
    fn random(n: usize, target: Charge, m: usize) -> Result<Self, HfError> {
        let mut bonds = vec![vec![(0, 0)]];
        let mut tensor_words = 0usize;
        for cut in 1..n {
            let hf = (cut.min(target.0), cut.min(target.1));
            let mut sectors = Vec::new();
            for a in 0..=cut.min(target.0) {
                for b in 0..=cut.min(target.1) {
                    if target.0 - a <= n - cut && target.1 - b <= n - cut {
                        let left =
                            choose_capped(cut, a, m).saturating_mul(choose_capped(cut, b, m));
                        let right = choose_capped(n - cut, target.0 - a, m)
                            .saturating_mul(choose_capped(n - cut, target.1 - b, m));
                        sectors.push(((a, b), left.min(right).min(m)));
                    }
                }
            }
            sectors.sort_by_key(|&(q, _)| (q.0.abs_diff(hf.0) + q.1.abs_diff(hf.1), q));
            let dimension = sectors
                .iter()
                .fold(0usize, |sum, x| sum.saturating_add(x.1))
                .min(m);
            // Guard the proposed tensors before allocating even their charge
            // labels. An enormous requested M must never allocate a large
            // bond list before the tensor budget is checked.
            tensor_words = bonds[cut - 1]
                .len()
                .checked_mul(dimension)
                .and_then(|x| x.checked_mul(4))
                .and_then(|x| x.checked_add(tensor_words))
                .ok_or_else(|| HfError::InvalidInput("MPS dimension overflow".into()))?;
            memory(Some(tensor_words))?;
            let mut bond = Vec::new();
            for repeat in 0..m {
                for &(q, count) in &sectors {
                    if repeat < count && bond.len() < m {
                        bond.push(q);
                    }
                }
                if bond.len() == m || sectors.iter().all(|x| x.1 <= repeat + 1) {
                    break;
                }
            }
            bonds.push(bond);
        }
        bonds.push(vec![target]);
        memory((0..n).try_fold(0usize, |sum, i| {
            bonds[i]
                .len()
                .checked_mul(bonds[i + 1].len())?
                .checked_mul(4)?
                .checked_add(sum)
        }))?;
        let mut seed = 0x8cb92baa675d941d_u64;
        let mut sites = Vec::new();
        for i in 0..n {
            let mut site = Site::new(bonds[i].clone(), bonds[i + 1].clone());
            for a in 0..site.left.len() {
                for (s, &q) in PHYSICAL.iter().enumerate() {
                    for b in 0..site.right.len() {
                        if add(site.left[a], q) == site.right[b] {
                            seed ^= seed << 13;
                            seed ^= seed >> 7;
                            seed ^= seed << 17;
                            let x = 2.0 * (seed >> 11) as f64 / (1u64 << 53) as f64 - 1.0;
                            site.data[(a * 4 + s) * site.right.len() + b] = x;
                        }
                    }
                }
            }
            sites.push(site);
        }
        let mut mps = Self { sites, target };
        for i in (0..n.saturating_sub(1)).rev() {
            let theta = mps.pair(i)?;
            mps.split(i, &theta, m, 0.0, false)?;
        }
        let norm = mps.expectation(&vec![ID; n]);
        if !norm.is_finite() || norm < 1e-28 {
            return Err(HfError::Response("zero initial MPS norm".into()));
        }
        for x in &mut mps.sites[0].data {
            *x /= norm.sqrt();
        }
        Ok(mps)
    }
    pub fn bond_dimensions(&self) -> Vec<usize> {
        self.sites
            .iter()
            .take(self.sites.len().saturating_sub(1))
            .map(|s| s.right.len())
            .collect()
    }
    pub fn electron_counts(&self) -> (usize, usize) {
        self.target
    }
    pub fn storage_bytes(&self) -> usize {
        self.sites.iter().map(|s| s.data.len() * 8).sum()
    }
    fn expectation(&self, locals: &[Local]) -> f64 {
        let mut env = Matrix::from_vec(1, 1, vec![1.0]);
        for (site, &local) in self.sites.iter().zip(locals) {
            env = transfer(site, &env, local, true);
        }
        env[(0, 0)]
    }
    /// Normalized expectation of a fermion monomial in interleaved spin order.
    pub fn fermion_expectation(&self, operators: &[(usize, bool)]) -> Result<f64, HfError> {
        let n = self.sites.len();
        if operators.iter().any(|x| x.0 >= 2 * n) {
            return Err(HfError::InvalidInput(
                "MPS operator index out of range".into(),
            ));
        }
        let (sign, locals) = product(n, operators);
        if sign == 0.0 {
            return Ok(0.0);
        }
        Ok(sign * self.expectation(&locals) / self.expectation(&vec![ID; n]))
    }
    /// All spin-traced normal-ordered RDM elements in p,q,r,s order.
    pub fn rdms(&self) -> Result<(Matrix, Vec<f64>), HfError> {
        let n = self.sites.len();
        memory(n.checked_pow(4).and_then(|x| x.checked_add(n * n)))?;
        let norm = self.expectation(&vec![ID; n]);
        let expect = |ops: &[(usize, bool)]| {
            let (sign, locals) = product(n, ops);
            if sign == 0.0 {
                0.0
            } else {
                sign * self.expectation(&locals) / norm
            }
        };
        let mut one = Matrix::zeros(n, n);
        let mut two = vec![0.0; n.pow(4)];
        for p in 0..n {
            for q in 0..n {
                for spin in 0..2 {
                    one[(p, q)] += expect(&[(2 * p + spin, true), (2 * q + spin, false)]);
                }
                for r in 0..n {
                    for s in 0..n {
                        for sigma in 0..2 {
                            for tau in 0..2 {
                                two[((p * n + q) * n + r) * n + s] += expect(&[
                                    (2 * p + sigma, true),
                                    (2 * r + tau, true),
                                    (2 * s + tau, false),
                                    (2 * q + sigma, false),
                                ]);
                            }
                        }
                    }
                }
            }
        }
        Ok((one, two))
    }
    pub fn spin_squared(&self) -> Result<f64, HfError> {
        let sz = (self.target.0 as f64 - self.target.1 as f64) * 0.5;
        let mut value = sz * sz + sz;
        for p in 0..self.sites.len() {
            for q in 0..self.sites.len() {
                value += self.fermion_expectation(&[
                    (2 * p + 1, true),
                    (2 * p, false),
                    (2 * q, true),
                    (2 * q + 1, false),
                ])?;
            }
        }
        Ok(value)
    }
    fn pair(&self, i: usize) -> Result<Vec<f64>, HfError> {
        let (a, b) = (&self.sites[i], &self.sites[i + 1]);
        let (dl, dm, dr) = (a.left.len(), a.right.len(), b.right.len());
        memory(dl.checked_mul(dr).and_then(|x| x.checked_mul(16 * 8)))?;
        let mut theta = vec![0.0; dl * 16 * dr];
        for l in 0..dl {
            for s in 0..4 {
                for t in 0..4 {
                    for r in 0..dr {
                        theta[((l * 4 + s) * 4 + t) * dr + r] =
                            (0..dm).map(|k| a.at(l, s, k) * b.at(k, t, r)).sum();
                    }
                }
            }
        }
        Ok(theta)
    }
    fn split(
        &mut self,
        i: usize,
        theta: &[f64],
        m: usize,
        tol: f64,
        forward: bool,
    ) -> Result<f64, HfError> {
        let left = &self.sites[i].left;
        let right = &self.sites[i + 1].right;
        let (dl, dr) = (left.len(), right.len());
        let mut rows = BTreeMap::<Charge, Vec<usize>>::new();
        let mut cols = BTreeMap::<Charge, Vec<usize>>::new();
        for (l, &charge) in left.iter().enumerate() {
            for (s, &physical) in PHYSICAL.iter().enumerate() {
                rows.entry(add(charge, physical))
                    .or_default()
                    .push(l * 4 + s);
            }
        }
        for (t, &physical) in PHYSICAL.iter().enumerate() {
            for (r, &charge) in right.iter().enumerate() {
                if charge.0 >= physical.0 && charge.1 >= physical.1 {
                    cols.entry((charge.0 - physical.0, charge.1 - physical.1))
                        .or_default()
                        .push(t * dr + r);
                }
            }
        }
        struct Schmidt {
            q: Charge,
            rows: Vec<usize>,
            cols: Vec<usize>,
            u: Matrix,
            v: Matrix,
            s: Vec<f64>,
        }
        let mut blocks = Vec::new();
        let mut weights = Vec::new();
        for (q, rows) in rows {
            let Some(cols) = cols.get(&q) else {
                continue;
            };
            let mut block = Matrix::zeros(rows.len(), cols.len());
            for (a, &row) in rows.iter().enumerate() {
                for (b, &col) in cols.iter().enumerate() {
                    block[(a, b)] = theta[row * 4 * dr + col];
                }
            }
            let view = faer::MatRef::from_row_major_slice(&block.data, block.rows, block.cols);
            let svd = view
                .thin_svd()
                .map_err(|_| HfError::Response("DMRG Schmidt SVD failed".into()))?;
            let rank = block.rows.min(block.cols);
            let mut u = Matrix::zeros(block.rows, rank);
            let mut v = Matrix::zeros(block.cols, rank);
            let mut singular = Vec::new();
            for k in 0..rank {
                let value = svd.S()[k];
                singular.push(value);
                weights.push((value * value, blocks.len(), k));
                for a in 0..block.rows {
                    u[(a, k)] = svd.U()[(a, k)];
                }
                for b in 0..block.cols {
                    v[(b, k)] = svd.V()[(b, k)];
                }
            }
            blocks.push(Schmidt {
                q,
                rows,
                cols: cols.clone(),
                u,
                v,
                s: singular,
            });
        }
        weights.sort_by(|a, b| b.0.total_cmp(&a.0));
        let total = weights.iter().map(|x| x.0).sum::<f64>();
        if !total.is_finite() || total <= 1e-28 {
            return Err(HfError::Response("zero two-site MPS norm".into()));
        }
        // A spin multiplet can have equal Schmidt weights in different Sz
        // sectors. Make a cut through such a multiplet deterministic instead
        // of letting eigensolver roundoff decide the retained charge sectors.
        // Prefer the local spin balance closest to the target spin density.
        let mut first = 0;
        while first < weights.len() {
            let mut end = first + 1;
            while end < weights.len()
                && weights[first].0 - weights[end].0 <= 1e-8 * weights[first].0
            {
                end += 1;
            }
            weights[first..end].sort_by_key(|&(_, block, column)| {
                let q = blocks[block].q;
                let balance = (q.0 as i128 - q.1 as i128) * self.sites.len() as i128
                    - (self.target.0 as i128 - self.target.1 as i128) * (i + 1) as i128;
                (balance.abs(), q, column)
            });
            first = end;
        }
        let positive = weights
            .iter()
            .take_while(|x| x.0 > total * 1e-28)
            .count()
            .max(1);
        let mut keep = positive.min(m);
        let mut discarded = weights[keep..].iter().map(|x| x.0).sum::<f64>();
        while keep > 1 && discarded + weights[keep - 1].0 <= tol * total {
            keep -= 1;
            discarded += weights[keep].0;
        }
        let norm = (total - discarded).sqrt();
        let bond = weights[..keep]
            .iter()
            .map(|x| blocks[x.1].q)
            .collect::<Vec<_>>();
        let mut a = Site::new(left.clone(), bond.clone());
        let mut b = Site::new(bond, right.clone());
        for (k, &(_, block, column)) in weights[..keep].iter().enumerate() {
            let block = &blocks[block];
            for (r, &row) in block.rows.iter().enumerate() {
                a.data[row * keep + k] =
                    block.u[(r, column)] * if forward { 1.0 } else { block.s[column] / norm };
            }
            for (c, &col) in block.cols.iter().enumerate() {
                b.data[k * 4 * dr + col] =
                    block.v[(c, column)] * if forward { block.s[column] / norm } else { 1.0 };
            }
        }
        debug_assert_eq!(a.data.len(), dl * 4 * keep);
        self.sites[i] = a;
        self.sites[i + 1] = b;
        Ok(discarded / total)
    }
}

/// A sweep is one directional pass, alternating left-to-right and right-to-left.
#[derive(Clone, Debug)]
pub struct MpsSolverOptions {
    pub bond_dimensions: Vec<usize>,
    pub sweeps: usize,
    /// Final fixed-bond one-site variational polishing passes.
    pub one_site_sweeps: usize,
    pub energy_tol: f64,
    pub residual_tol: f64,
    pub truncation_tol: f64,
    pub max_davidson_iterations: usize,
}
impl Default for MpsSolverOptions {
    fn default() -> Self {
        Self {
            bond_dimensions: vec![64, 128, 256],
            sweeps: 12,
            one_site_sweeps: 12,
            energy_tol: 1e-10,
            residual_tol: 1e-10,
            truncation_tol: 1e-12,
            max_davidson_iterations: 200,
        }
    }
}
#[derive(Clone, Debug)]
pub struct MpsSolution {
    pub energy: f64,
    pub state: MatrixProductState,
    pub sweep_energies: Vec<f64>,
    pub one_site_energies: Vec<f64>,
    pub discarded_weights: Vec<f64>,
    pub converged: bool,
    /// Largest pre-truncation two-site eigenproblem residual.
    pub max_local_residual: f64,
}

fn hamiltonian(h: &Matrix, eri: &MoEriTensor, constant: f64) -> Result<Vec<Term>, HfError> {
    let n = h.rows;
    // Includes the temporary map nodes and product-string keys, not just values.
    memory(
        n.checked_pow(4)
            .and_then(|x| x.checked_mul(4))
            .and_then(|k| k.checked_mul(n + 16)),
    )?;
    let mut strings = BTreeMap::<Vec<Local>, f64>::new();
    strings.insert(vec![ID; n], constant);
    let mut accumulate = |coefficient: f64, ops: &[(usize, bool)]| {
        if coefficient == 0.0 {
            return;
        }
        let (sign, local) = product(n, ops);
        if sign != 0.0 {
            *strings.entry(local).or_default() += coefficient * sign;
        }
    };
    for p in 0..n {
        for q in 0..n {
            for spin in 0..2 {
                accumulate(h[(p, q)], &[(2 * p + spin, true), (2 * q + spin, false)]);
            }
            for r in 0..n {
                for s in 0..n {
                    for sigma in 0..2 {
                        for tau in 0..2 {
                            accumulate(
                                0.5 * eri.get(p, q, r, s),
                                &[
                                    (2 * p + sigma, true),
                                    (2 * r + tau, true),
                                    (2 * s + tau, false),
                                    (2 * q + sigma, false),
                                ],
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(strings
        .into_iter()
        .filter(|(_, c)| *c != 0.0)
        .map(|(local, coefficient)| Term { coefficient, local })
        .collect())
}

struct EffectiveTerm {
    coefficient: f64,
    left: Matrix,
    right: Matrix,
    first: Local,
    second: Local,
}
struct Effective {
    dl: usize,
    dr: usize,
    nt: usize,
    allowed: Vec<usize>,
    terms: Vec<EffectiveTerm>,
}
impl Effective {
    fn new(
        mps: &MatrixProductState,
        terms: &[Term],
        i: usize,
        width: usize,
    ) -> Result<Self, HfError> {
        let nt = if width == 2 { 4 } else { 1 };
        let (left, right) = (&mps.sites[i].left, &mps.sites[i + width - 1].right);
        let (dl, dr) = (left.len(), right.len());
        let dense = dl.checked_mul(dr).and_then(|x| x.checked_mul(4 * nt));
        memory(dense.and_then(|x| x.checked_mul(100)).and_then(|x| {
            terms
                .len()
                .checked_mul(dl.checked_mul(dl)?.checked_add(dr.checked_mul(dr)?)?)?
                .checked_add(x)
        }))?;
        let mut allowed = Vec::new();
        for (l, &q) in left.iter().enumerate() {
            for (s, &qs) in PHYSICAL.iter().enumerate() {
                for (t, &qt) in PHYSICAL[..nt].iter().enumerate() {
                    for (r, &qr) in right.iter().enumerate() {
                        if add(add(q, qs), qt) == qr {
                            allowed.push(((l * 4 + s) * nt + t) * dr + r);
                        }
                    }
                }
            }
        }
        let mut effective = Vec::with_capacity(terms.len());
        for term in terms {
            let mut left = Matrix::from_vec(1, 1, vec![1.0]);
            for k in 0..i {
                left = transfer(&mps.sites[k], &left, term.local[k], true);
            }
            let mut right = Matrix::from_vec(1, 1, vec![1.0]);
            for k in (i + width..mps.sites.len()).rev() {
                right = transfer(&mps.sites[k], &right, term.local[k], false);
            }
            if left.data.iter().all(|x| *x == 0.0) || right.data.iter().all(|x| *x == 0.0) {
                continue;
            }
            effective.push(EffectiveTerm {
                coefficient: term.coefficient,
                left,
                right,
                first: term.local[i],
                second: if width == 2 { term.local[i + 1] } else { ID },
            });
        }
        Ok(Self {
            dl,
            dr,
            nt,
            allowed,
            terms: effective,
        })
    }
    fn expand(&self, x: &[f64]) -> Vec<f64> {
        let mut dense = vec![0.0; self.dl * 4 * self.nt * self.dr];
        for (&i, &x) in self.allowed.iter().zip(x) {
            dense[i] = x;
        }
        dense
    }
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        let x = self.expand(x);
        let mut out = vec![0.0; x.len()];
        for term in &self.terms {
            for s in 0..4 {
                for t in 0..self.nt {
                    let (sp, tp) = (term.first.0[s], term.second.0[t]);
                    if sp == 0 || tp == 0 {
                        continue;
                    }
                    let mut slice = Matrix::zeros(self.dl, self.dr);
                    for l in 0..self.dl {
                        for r in 0..self.dr {
                            slice[(l, r)] = x[((l * 4 + s) * self.nt + t) * self.dr + r];
                        }
                    }
                    if slice.data.iter().all(|v| *v == 0.0) {
                        continue;
                    }
                    let result = term.left.dot(&slice).dot(&term.right.transpose());
                    let (s, t) = (
                        sp.unsigned_abs() as usize - 1,
                        tp.unsigned_abs() as usize - 1,
                    );
                    let factor = term.coefficient * (sp.signum() * tp.signum()) as f64;
                    for l in 0..self.dl {
                        for r in 0..self.dr {
                            out[((l * 4 + s) * self.nt + t) * self.dr + r] +=
                                factor * result[(l, r)];
                        }
                    }
                }
            }
        }
        self.allowed.iter().map(|&i| out[i]).collect()
    }
    fn diagonal(&self) -> Vec<f64> {
        self.allowed
            .iter()
            .map(|&index| {
                let r = index % self.dr;
                let t = (index / self.dr) % self.nt;
                let s = (index / self.dr / self.nt) % 4;
                let l = index / self.dr / (4 * self.nt);
                self.terms
                    .iter()
                    .map(|term| {
                        term.coefficient
                            * term.left[(l, l)]
                            * term.right[(r, r)]
                            * term.first.diagonal(s)
                            * term.second.diagonal(t)
                    })
                    .sum()
            })
            .collect()
    }
}

/// Ground state in a fixed (Nalpha,Nbeta) sector; no total-spin projection.
pub fn solve_mps(
    h: &Matrix,
    eri: &MoEriTensor,
    n_alpha: usize,
    n_beta: usize,
    constant: f64,
    options: &MpsSolverOptions,
) -> Result<MpsSolution, HfError> {
    let n = h.rows;
    if h.cols != n
        || h.data.len() != n * n
        || eri.n_basis != n
        || eri.n_pairs != n * (n + 1) / 2
        || eri.data.len() != eri.n_pairs.pow(2)
        || n_alpha > n
        || n_beta > n
        || h.data.iter().chain(&eri.data).any(|x| !x.is_finite())
        || !constant.is_finite()
        || options.sweeps == 0
        || options.bond_dimensions.is_empty()
        || options.bond_dimensions.contains(&0)
        || options.max_davidson_iterations == 0
        || !options.energy_tol.is_finite()
        || options.energy_tol <= 0.0
        || !options.residual_tol.is_finite()
        || options.residual_tol <= 0.0
        || !options.truncation_tol.is_finite()
        || options.truncation_tol < 0.0
        || options.truncation_tol >= 1.0
    {
        return Err(HfError::InvalidInput(
            "invalid MPS integrals, sector or controls".into(),
        ));
    }
    for p in 0..n {
        for q in 0..n {
            if (h[(p, q)] - h[(q, p)]).abs() > 1e-12 {
                return Err(HfError::InvalidInput(
                    "MPS requires Hermitian integrals".into(),
                ));
            }
        }
    }
    for p in 0..eri.n_pairs {
        for q in 0..p {
            if (eri.data[p * eri.n_pairs + q] - eri.data[q * eri.n_pairs + p]).abs() > 1e-12 {
                return Err(HfError::InvalidInput(
                    "MPS requires pair-symmetric ERIs".into(),
                ));
            }
        }
    }
    if n == 0 {
        return Ok(MpsSolution {
            energy: constant,
            state: MatrixProductState {
                sites: vec![],
                target: (0, 0),
            },
            sweep_energies: vec![constant],
            one_site_energies: vec![],
            discarded_weights: vec![0.0],
            converged: true,
            max_local_residual: 0.0,
        });
    }
    let terms = hamiltonian(h, eri, constant)?;
    let mut state = MatrixProductState::random(n, (n_alpha, n_beta), options.bond_dimensions[0])?;
    let energy = |state: &MatrixProductState| {
        let norm = state.expectation(&vec![ID; n]);
        terms
            .iter()
            .map(|t| t.coefficient * state.expectation(&t.local))
            .sum::<f64>()
            / norm
    };
    let mut energies = Vec::new();
    let mut discarded_weights = Vec::new();
    let mut converged = n == 1;
    let mut max_local_residual = 0.0_f64;
    if n == 1 {
        energies.push(energy(&state));
        discarded_weights.push(0.0);
    }
    for sweep in 0..if n > 1 { options.sweeps } else { 0 } {
        let m = options.bond_dimensions[sweep.min(options.bond_dimensions.len() - 1)];
        let forward = sweep % 2 == 0;
        let mut discarded = 0.0_f64;
        for step in 0..n - 1 {
            let i = if forward { step } else { n - 2 - step };
            let effective = Effective::new(&state, &terms, i, 2)?;
            let diagonal = effective.diagonal();
            let solution = try_davidson_solver(
                diagonal.len(),
                &diagonal,
                |x| Ok(effective.apply(x)),
                DavidsonOptions {
                    n_roots: 1,
                    max_subspace: 40,
                    max_iterations: options.max_davidson_iterations,
                    tol: options.residual_tol,
                },
            )?;
            if !solution.converged {
                return Err(HfError::PostHf {
                    method: "DMRG".into(),
                    detail: format!("two-site Davidson did not converge at bond {i}"),
                });
            }
            let vector = &solution.eigenvectors[0];
            let sigma = effective.apply(vector);
            let residual = sigma
                .iter()
                .zip(vector)
                .map(|(s, x)| (s - solution.eigenvalues[0] * x).powi(2))
                .sum::<f64>()
                .sqrt();
            max_local_residual = max_local_residual.max(residual);
            let theta = effective.expand(vector);
            discarded =
                discarded.max(state.split(i, &theta, m, options.truncation_tol, forward)?);
        }
        let e = energy(&state);
        let done = energies
            .last()
            .is_some_and(|old: &f64| (e - old).abs() < options.energy_tol)
            && sweep + 1 >= options.bond_dimensions.len().max(2);
        energies.push(e);
        discarded_weights.push(discarded);
        if done {
            converged = true;
            break;
        }
    }
    let mut one_site_energies = Vec::new();
    if n > 1 && options.one_site_sweeps > 0 {
        converged = false;
        let initial_polish_energy = energy(&state);
        let first_forward = energies.len() % 2 == 0;
        for sweep in 0..options.one_site_sweeps {
            let forward = first_forward == (sweep % 2 == 0);
            for step in 0..n {
                let i = if forward { step } else { n - 1 - step };
                let effective = Effective::new(&state, &terms, i, 1)?;
                let diagonal = effective.diagonal();
                let solution = try_davidson_solver(
                    diagonal.len(),
                    &diagonal,
                    |x| Ok(effective.apply(x)),
                    DavidsonOptions {
                        n_roots: 1,
                        max_subspace: 40,
                        max_iterations: options.max_davidson_iterations,
                        tol: options.residual_tol,
                    },
                )?;
                if !solution.converged {
                    return Err(HfError::PostHf {
                        method: "DMRG".into(),
                        detail: format!("one-site Davidson did not converge at site {i}"),
                    });
                }
                state.sites[i].data = effective.expand(&solution.eigenvectors[0]);
                let shift = if forward && i + 1 < n {
                    Some((i, true))
                } else if !forward && i > 0 {
                    Some((i - 1, false))
                } else {
                    None
                };
                if let Some((cut, direction)) = shift {
                    let m = state.sites[cut].right.len();
                    let theta = state.pair(cut)?;
                    state.split(cut, &theta, m, 0.0, direction)?;
                }
            }
            let e = energy(&state);
            let old = one_site_energies
                .last()
                .copied()
                .unwrap_or(initial_polish_energy);
            // A fixed-sector one-site pass is variational. Truncating a zero
            // Schmidt direction while moving the center changes only roundoff.
            if e > old + 1e-10 {
                return Err(HfError::Response("nonvariational one-site MPS pass".into()));
            }
            one_site_energies.push(e);
            if sweep > 0 && (e - old).abs() < options.energy_tol {
                converged = true;
                break;
            }
        }
    }
    let e = energy(&state);
    if !e.is_finite() {
        return Err(HfError::Response("non-finite MPS energy".into()));
    }
    Ok(MpsSolution {
        energy: e,
        state,
        sweep_energies: energies,
        one_site_energies,
        discarded_weights,
        converged,
        max_local_residual,
    })
}
