// SPDX-License-Identifier: Apache-2.0
//! Spin-string Hamiltonian action. The three two-electron sectors (alpha/alpha,
//! beta/beta, alpha/beta) share link tables and stream O(N_det) intermediates.
//!
//! H = (h_ps - 1/2 sum_q (pq|qs)) E_ps + 1/2 (pq|rs) E_pq E_rs,
//! E_pq = a†_pα a_qα + a†_pβ a_qβ. No determinant-pair Hamiltonian is stored.
use super::string::{DeterminantSpace, SpinString};
use crate::error::HfError;
use crate::matrix::Matrix;
use crate::mo_integrals::MoEriTensor;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
struct Link {
    source: usize,
    target: usize,
    phase: f64,
}
struct Links {
    pairs: Vec<Vec<Link>>,
    addresses: HashMap<u64, usize>,
}
impl Links {
    fn new(strings: &[SpinString], n: usize) -> Result<Self, HfError> {
        let addresses = strings
            .iter()
            .enumerate()
            .map(|(i, s)| (s.0, i))
            .collect::<HashMap<_, _>>();
        let mut pairs = vec![Vec::new(); n * n];
        for (source, string) in strings.iter().enumerate() {
            for q in string.occupied_orbitals() {
                for p in 0..n {
                    if let Some((target, phase)) = string.apply_single_excitation(p, q) {
                        let target = *addresses
                            .get(&target.0)
                            .ok_or_else(|| HfError::Integral("missing excitation string".into()))?;
                        pairs[p * n + q].push(Link {
                            source,
                            target,
                            phase,
                        });
                    }
                }
            }
        }
        Ok(Self { pairs, addresses })
    }
}

fn checked_choose(n: usize, k: usize) -> Option<usize> {
    if k > n {
        return None;
    }
    let k = k.min(n - k);
    let mut value = 1u128;
    for i in 0..k {
        value = value.checked_mul((n - i) as u128)? / (i + 1) as u128;
    }
    usize::try_from(value).ok()
}

pub struct FciHamiltonian<'a> {
    pub space: DeterminantSpace,
    pub h: &'a Matrix,
    pub eri: &'a MoEriTensor,
    pub constant: f64,
    alpha: Links,
    beta: Links,
    effective_h: Matrix,
}

#[derive(Clone, Debug)]
pub struct FciSolverOptions {
    pub n_roots: usize,
    pub max_iterations: usize,
    pub max_subspace: usize,
    pub residual_tol: f64,
    pub target_s_squared: Option<f64>,
    pub dense_threshold: usize,
    pub memory_budget: usize,
}
impl Default for FciSolverOptions {
    fn default() -> Self {
        Self {
            n_roots: 1,
            max_iterations: 150,
            max_subspace: 40,
            residual_tol: 1e-10,
            target_s_squared: None,
            dense_threshold: 128,
            memory_budget: 256 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Debug)]
pub struct FciState {
    pub energy: f64,
    pub vector: Vec<f64>,
    pub s_squared: f64,
    pub residual_norm: f64,
}
#[derive(Clone, Debug)]
pub struct FciSolution {
    pub states: Vec<FciState>,
    pub iterations: usize,
}
impl<'a> FciHamiltonian<'a> {
    pub fn solve(&self, options: &FciSolverOptions) -> Result<FciSolution, HfError> {
        let n = self.h.rows;
        let electrons = self.space.n_alpha + self.space.n_beta;
        let hnorm = self.h.data.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let gnorm = self
            .eri
            .data
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        // A conservative spectral-width bound makes non-target spin sectors higher
        // than every physical state; final energies/residuals use the physical H.
        let penalty = 1.0
            + 2.0 * (electrons * n) as f64 * hnorm
            + (electrons * electrons * n * n) as f64 * gnorm;
        self.solve_with_action(options, |c| self.apply(c), || self.diagonal(), penalty)
    }
    /// Shared spin targeting and eigensolver for general Hermitian CI operators.
    pub(crate) fn solve_with_action<F, D>(
        &self,
        options: &FciSolverOptions,
        physical_action: F,
        diagonal: D,
        penalty: f64,
    ) -> Result<FciSolution, HfError>
    where
        F: Fn(&[f64]) -> Result<Vec<f64>, HfError>,
        D: Fn() -> Vec<f64>,
    {
        let dim = self.dimension();
        if options.n_roots == 0
            || options.n_roots > dim
            || options.max_iterations == 0
            || !options.residual_tol.is_finite()
            || options.residual_tol <= 0.0
            || !penalty.is_finite()
            || penalty <= 0.
        {
            return Err(HfError::InvalidInput("invalid FCI solver options".into()));
        }
        if let Some(target) = options.target_s_squared {
            self.validate_spin_target(target, options.n_roots)?;
        }
        let action = |c: &[f64]| -> Result<Vec<f64>, HfError> {
            let mut sigma = physical_action(c)?;
            if sigma.len() != dim || sigma.iter().any(|x| !x.is_finite()) {
                return Err(HfError::Integral(
                    "invalid external CI operator action".into(),
                ));
            }
            if let Some(target) = options.target_s_squared {
                let mut delta = self.spin_squared_action(c)?;
                for (x, &v) in delta.iter_mut().zip(c) {
                    *x -= target * v;
                }
                let squared = self.spin_squared_action(&delta)?;
                for ((x, &s), &d) in sigma.iter_mut().zip(&squared).zip(&delta) {
                    *x += penalty * (s - target * d);
                }
            }
            Ok(sigma)
        };
        let (vectors, iterations) = if dim <= options.dense_threshold {
            let bytes = dim
                .checked_mul(dim)
                .and_then(|x| x.checked_mul(8))
                .ok_or_else(|| HfError::InvalidInput("FCI matrix size overflow".into()))?;
            if bytes > options.memory_budget {
                return Err(HfError::Unsupported {
                    method: "FCI".into(),
                    detail: "dense Hamiltonian exceeds memory budget".into(),
                });
            }
            let mut matrix = Matrix::zeros(dim, dim);
            for column in 0..dim {
                let mut c = vec![0.0; dim];
                c[column] = 1.0;
                let sigma = action(&c)?;
                for row in 0..dim {
                    matrix[(row, column)] = sigma[row];
                }
            }
            for row in 0..dim {
                for col in 0..row {
                    if (matrix[(row, col)] - matrix[(col, row)]).abs() > 1e-10 {
                        return Err(HfError::Integral("FCI Hamiltonian is not Hermitian".into()));
                    }
                }
            }
            let (_, c) = matrix.eigh();
            (
                (0..options.n_roots)
                    .map(|i| (0..dim).map(|j| c[(j, i)]).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
                1,
            )
        } else {
            let subspace = options.max_subspace.max(4 * options.n_roots).min(dim);
            let bytes = dim
                .checked_mul(2 * subspace + 8)
                .and_then(|x| x.checked_mul(8))
                .ok_or_else(|| HfError::InvalidInput("FCI subspace size overflow".into()))?;
            if bytes > options.memory_budget {
                return Err(HfError::Unsupported {
                    method: "FCI".into(),
                    detail:
                        "Davidson subspace exceeds memory budget; reduce max_subspace or use DMRG"
                            .into(),
                });
            }
            let diagonal = diagonal();
            let result = crate::solvers::davidson::try_davidson_solver_projected(
                dim,
                &diagonal,
                |c| physical_action(c),
                |c| {
                    if let Some(target) = options.target_s_squared {
                        self.project_spin(c, target)
                    } else {
                        Ok(c.to_vec())
                    }
                },
                crate::solvers::DavidsonOptions {
                    n_roots: options.n_roots,
                    max_subspace: options.max_subspace,
                    max_iterations: options.max_iterations,
                    tol: options.residual_tol * 0.25,
                },
            )?;
            if !result.converged {
                return Err(HfError::PostHf {
                    method: "FCI".into(),
                    detail: format!("Davidson failed in {} iterations", result.iterations),
                });
            }
            (result.eigenvectors, result.iterations)
        };
        let mut states = Vec::with_capacity(vectors.len());
        for mut vector in vectors {
            let norm = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm <= 0.0 {
                return Err(HfError::Response("zero FCI eigenvector".into()));
            }
            for value in &mut vector {
                *value /= norm;
            }
            let sigma = physical_action(&vector)?;
            if sigma.len() != dim || sigma.iter().any(|x| !x.is_finite()) {
                return Err(HfError::Integral(
                    "invalid external CI physical residual".into(),
                ));
            }
            let energy = vector.iter().zip(&sigma).map(|(a, b)| a * b).sum::<f64>();
            let residual_norm = sigma
                .iter()
                .zip(&vector)
                .map(|(s, c)| (s - energy * c).powi(2))
                .sum::<f64>()
                .sqrt();
            let s_squared = self.spin_squared(&vector)?;
            if residual_norm > options.residual_tol
                || options
                    .target_s_squared
                    .is_some_and(|target| (s_squared - target).abs() > 1e-8)
            {
                return Err(HfError::PostHf {
                    method: "FCI".into(),
                    detail: format!("physical residual {residual_norm:.3e}, S²={s_squared:.10}"),
                });
            }
            states.push(FciState {
                energy,
                vector,
                s_squared,
                residual_norm,
            });
        }
        states.sort_by(|a, b| a.energy.total_cmp(&b.energy));
        Ok(FciSolution { states, iterations })
    }
    pub fn new(
        h: &'a Matrix,
        eri: &'a MoEriTensor,
        n_alpha: usize,
        n_beta: usize,
        constant: f64,
    ) -> Result<Self, HfError> {
        let n = h.rows;
        if n > 64
            || h.cols != n
            || eri.n_basis != n
            || n_alpha > n
            || n_beta > n
            || !constant.is_finite()
            || h.data.iter().any(|v| !v.is_finite())
        {
            return Err(HfError::InvalidInput(
                "invalid FCI Hamiltonian or electron count".into(),
            ));
        }
        let np = n * (n + 1) / 2;
        if eri.data.len() != np * np || eri.data.iter().any(|v| !v.is_finite()) {
            return Err(HfError::InvalidInput("invalid FCI integral tensor".into()));
        }
        let na = checked_choose(n, n_alpha)
            .ok_or_else(|| HfError::InvalidInput("FCI alpha dimension overflow".into()))?;
        let nb = checked_choose(n, n_beta)
            .ok_or_else(|| HfError::InvalidInput("FCI beta dimension overflow".into()))?;
        let dimension = na
            .checked_mul(nb)
            .ok_or_else(|| HfError::InvalidInput("FCI dimension overflow".into()))?;
        // Reject an infeasible exact space before generating strings or scratch buffers.
        if dimension > 8_000_000
            || na
                .checked_add(nb)
                .and_then(|v| v.checked_mul(n * n))
                .is_none_or(|v| v > 32_000_000)
        {
            return Err(HfError::Unsupported {
                method: "FCI".into(),
                detail: "exact spin-string workspace exceeds the supported memory budget".into(),
            });
        }
        let space = DeterminantSpace::new(n, n_alpha, n_beta);
        let alpha = Links::new(&space.alpha_strings, n)?;
        let beta = Links::new(&space.beta_strings, n)?;
        let mut effective_h = h.clone();
        for p in 0..n {
            for s in 0..n {
                effective_h[(p, s)] -= 0.5 * (0..n).map(|q| eri.get(p, q, q, s)).sum::<f64>();
            }
        }
        Ok(Self {
            space,
            h,
            eri,
            constant,
            alpha,
            beta,
            effective_h,
        })
    }
    pub fn dimension(&self) -> usize {
        self.space.n_determinants()
    }
    fn validate_vector(&self, c: &[f64]) -> Result<(), HfError> {
        if c.len() != self.dimension() || c.iter().any(|v| !v.is_finite()) {
            return Err(HfError::InvalidInput("invalid FCI vector".into()));
        }
        Ok(())
    }
    fn apply_alpha(&self, links: &[Link], input: &[f64], output: &mut [f64], scale: f64) {
        let nb = self.space.beta_strings.len();
        for link in links {
            for b in 0..nb {
                output[link.target * nb + b] += scale * link.phase * input[link.source * nb + b];
            }
        }
    }
    fn apply_beta(&self, links: &[Link], input: &[f64], output: &mut [f64], scale: f64) {
        let nb = self.space.beta_strings.len();
        for a in 0..self.space.alpha_strings.len() {
            for link in links {
                output[a * nb + link.target] += scale * link.phase * input[a * nb + link.source];
            }
        }
    }
    pub(crate) fn apply_unchecked(&self, c: &[f64]) -> Vec<f64> {
        let n = self.h.rows;
        let mut out = c.iter().map(|x| self.constant * x).collect::<Vec<_>>();
        for p in 0..n {
            for q in 0..n {
                let h = self.effective_h[(p, q)];
                self.apply_alpha(&self.alpha.pairs[p * n + q], c, &mut out, h);
                self.apply_beta(&self.beta.pairs[p * n + q], c, &mut out, h);
            }
        }
        let mut da = vec![0.0; c.len()];
        let mut db = da.clone();
        for r in 0..n {
            for s in 0..n {
                da.fill(0.0);
                db.fill(0.0);
                self.apply_alpha(&self.alpha.pairs[r * n + s], c, &mut da, 1.0);
                self.apply_beta(&self.beta.pairs[r * n + s], c, &mut db, 1.0);
                for p in 0..n {
                    for q in 0..n {
                        let g = self.eri.get(p, q, r, s);
                        if g == 0.0 {
                            continue;
                        }
                        self.apply_alpha(&self.alpha.pairs[p * n + q], &da, &mut out, 0.5 * g);
                        self.apply_beta(&self.beta.pairs[p * n + q], &db, &mut out, 0.5 * g);
                        self.apply_alpha(&self.alpha.pairs[p * n + q], &db, &mut out, g);
                    }
                }
            }
        }
        out
    }
    pub fn apply(&self, c: &[f64]) -> Result<Vec<f64>, HfError> {
        self.validate_vector(c)?;
        Ok(self.apply_unchecked(c))
    }
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.dimension())
            .map(|index| {
                let (a, b) = self.space.linear_to_pair(index);
                self.constant
                    + super::slater_condon::slater_condon_element(
                        &self.space.alpha_strings[a],
                        &self.space.beta_strings[b],
                        &self.space.alpha_strings[a],
                        &self.space.beta_strings[b],
                        self.h,
                        self.eri,
                    )
            })
            .collect()
    }
    fn bits(&self, index: usize) -> u128 {
        let (a, b) = self.space.linear_to_pair(index);
        self.space.alpha_strings[a].0 as u128
            | ((self.space.beta_strings[b].0 as u128) << self.h.rows)
    }
    fn address(&self, bits: u128) -> Option<usize> {
        let mask = if self.h.rows == 64 {
            u64::MAX as u128
        } else {
            (1u128 << self.h.rows) - 1
        };
        let a = self.alpha.addresses.get(&((bits & mask) as u64))?;
        let b = self.beta.addresses.get(&((bits >> self.h.rows) as u64))?;
        Some(self.space.pair_to_linear(*a, *b))
    }
    pub fn spin_squared_action(&self, c: &[f64]) -> Result<Vec<f64>, HfError> {
        self.validate_vector(c)?;
        let n = self.h.rows;
        let sz = (self.space.n_alpha as f64 - self.space.n_beta as f64) / 2.0;
        let mut out = c.iter().map(|x| sz * (sz + 1.0) * x).collect::<Vec<_>>();
        for (source, &coefficient) in c.iter().enumerate() {
            if coefficient == 0.0 {
                continue;
            }
            let bits = self.bits(source);
            for p in 0..n {
                let Some((raised, phase1)) = annihilate(bits, n + p)
                    .and_then(|(b, s)| create(b, p).map(|(b, t)| (b, s * t)))
                else {
                    continue;
                };
                for q in 0..n {
                    let Some((lowered, phase2)) = annihilate(raised, q)
                        .and_then(|(b, s)| create(b, n + q).map(|(b, t)| (b, s * t)))
                    else {
                        continue;
                    };
                    let target = self.address(lowered).ok_or_else(|| {
                        HfError::Integral("spin ladder left the determinant space".into())
                    })?;
                    out[target] += coefficient * phase1 * phase2;
                }
            }
        }
        Ok(out)
    }
    fn validate_spin_target(&self, target: f64, n_roots: usize) -> Result<(), HfError> {
        let n = self.h.rows;
        let electrons = self.space.n_alpha + self.space.n_beta;
        let sz = (self.space.n_alpha as f64 - self.space.n_beta as f64).abs() * 0.5;
        let spin = (target + 0.25).sqrt() - 0.5;
        if !target.is_finite()
            || target < 0.0
            || spin < sz - 1e-9
            || spin > electrons.min(2 * n - electrons) as f64 * 0.5 + 1e-9
            || ((spin - sz) - (spin - sz).round()).abs() > 1e-9
        {
            return Err(HfError::InvalidInput(
                "target S(S+1) is incompatible with the FCI spin sector".into(),
            ));
        }
        let twice_spin = (2.0 * spin).round() as usize;
        let up = (electrons + twice_spin) / 2;
        let down = electrons - up;
        let sector = checked_choose(n, up)
            .and_then(|a| checked_choose(n, down).and_then(|b| a.checked_mul(b)));
        let higher = if down == 0 || up == n {
            Some(0)
        } else {
            checked_choose(n, up + 1)
                .and_then(|a| checked_choose(n, down - 1).and_then(|b| a.checked_mul(b)))
        };
        let count = sector
            .zip(higher)
            .and_then(|(a, b)| a.checked_sub(b))
            .ok_or_else(|| HfError::InvalidInput("spin-sector dimension overflow".into()))?;
        if n_roots > count {
            return Err(HfError::InvalidInput(format!(
                "requested {n_roots} roots but target S²={target} has only {count}"
            )));
        }
        Ok(())
    }
    /// Exact spectral polynomial projector onto a total-spin sector.
    pub fn project_spin(&self, c: &[f64], target: f64) -> Result<Vec<f64>, HfError> {
        self.validate_vector(c)?;
        self.validate_spin_target(target, 1)?;
        let electrons = self.space.n_alpha + self.space.n_beta;
        let minimum = (self.space.n_alpha as f64 - self.space.n_beta as f64).abs() * 0.5;
        let maximum = electrons.min(2 * self.h.rows - electrons) as f64 * 0.5;
        let mut out = c.to_vec();
        let mut spin = minimum;
        while spin <= maximum + 1e-9 {
            let eigenvalue = spin * (spin + 1.0);
            if (eigenvalue - target).abs() > 1e-9 {
                let ss = self.spin_squared_action(&out)?;
                for (v, s) in out.iter_mut().zip(ss) {
                    *v = (s - eigenvalue * (*v)) / (target - eigenvalue);
                }
            }
            spin += 1.0;
        }
        Ok(out)
    }
    pub fn spin_squared(&self, c: &[f64]) -> Result<f64, HfError> {
        let out = self.spin_squared_action(c)?;
        let norm = c.iter().map(|x| x * x).sum::<f64>();
        if norm <= 0.0 {
            return Err(HfError::InvalidInput("zero CI vector".into()));
        }
        Ok(out.iter().zip(c).map(|(x, y)| x * y).sum::<f64>() / norm)
    }

    /// Spin-traced normal-ordered k-RDM, stored as p1,q1,p2,q2,... .
    /// Gamma_k = sum_spin <p1† ... pk† qk ... q1>. No symmetry is imposed.
    pub fn rdm(&self, c: &[f64], order: usize) -> Result<Vec<f64>, HfError> {
        let mut tensor = self.transition_rdm(c, c, order)?;
        let norm = c.iter().map(|x| x * x).sum::<f64>();
        if norm <= 0.0 {
            return Err(HfError::InvalidInput("zero CI vector".into()));
        }
        for value in &mut tensor {
            *value /= norm;
        }
        Ok(tensor)
    }
    pub fn transition_rdm(
        &self,
        bra: &[f64],
        ket: &[f64],
        order: usize,
    ) -> Result<Vec<f64>, HfError> {
        if order > 3 {
            return Err(HfError::InvalidInput(
                "materialized RDM order must be zero through three".into(),
            ));
        }
        self.accumulate_rdm(bra, ket, order, None)
    }
    /// Sum_tu F_tu Gamma4[pq,rs,vw,tu], generated directly into a sixth-rank
    /// tensor. The eighth-rank 4-RDM is never allocated.
    pub fn contracted_four_rdm(&self, c: &[f64], f: &Matrix) -> Result<Vec<f64>, HfError> {
        let mut tensor = self.accumulate_rdm(c, c, 4, Some(f))?;
        let norm = c.iter().map(|x| x * x).sum::<f64>();
        if norm <= 0.0 {
            return Err(HfError::InvalidInput("zero CI vector".into()));
        }
        for value in &mut tensor {
            *value /= norm;
        }
        Ok(tensor)
    }
    fn accumulate_rdm(
        &self,
        bra: &[f64],
        ket: &[f64],
        order: usize,
        f: Option<&Matrix>,
    ) -> Result<Vec<f64>, HfError> {
        self.validate_vector(bra)?;
        self.validate_vector(ket)?;
        let n = self.h.rows;
        if f.is_some_and(|m| m.rows != n || m.cols != n || m.data.iter().any(|v| !v.is_finite())) {
            return Err(HfError::InvalidInput(
                "invalid fourth-RDM contraction matrix".into(),
            ));
        }
        let out_order = order - usize::from(f.is_some());
        let length = n
            .checked_pow((2 * out_order) as u32)
            .ok_or_else(|| HfError::InvalidInput("RDM dimension overflow".into()))?;
        if length > 32 * 1024 * 1024 {
            return Err(HfError::Unsupported {
                method: "FCI RDM".into(),
                detail: "materialized tensor exceeds 256 MiB".into(),
            });
        }
        let mut out = vec![0.0; length];
        if order == 0 {
            out[0] = bra.iter().zip(ket).map(|(a, b)| a * b).sum();
            return Ok(out);
        }
        struct Accumulator<'e, 'b, 'h> {
            engine: &'e FciHamiltonian<'h>,
            bra: &'b [f64],
            order: usize,
            out_order: usize,
            f: Option<&'b Matrix>,
            out: &'b mut [f64],
            ann: [usize; 4],
            cre: [usize; 4],
        }
        impl Accumulator<'_, '_, '_> {
            fn remove(
                &mut self,
                bits: u128,
                depth: usize,
                coefficient: f64,
            ) -> Result<(), HfError> {
                if depth == self.order {
                    return self.insert(bits, 0, coefficient);
                }
                let mut occupied = bits;
                while occupied != 0 {
                    let q = occupied.trailing_zeros() as usize;
                    occupied &= occupied - 1;
                    let (next, phase) = annihilate(bits, q)
                        .ok_or_else(|| HfError::Integral("invalid occupied RDM index".into()))?;
                    self.ann[depth] = q;
                    self.remove(next, depth + 1, coefficient * phase)?;
                }
                Ok(())
            }
            fn insert(
                &mut self,
                bits: u128,
                depth: usize,
                coefficient: f64,
            ) -> Result<(), HfError> {
                let n = self.engine.h.rows;
                if depth == self.order {
                    let target = self.engine.address(bits).ok_or_else(|| {
                        HfError::Integral("RDM creation left its string space".into())
                    })?;
                    let mut index = 0;
                    for i in 0..self.out_order {
                        index = (index * n + self.cre[i]) * n + self.ann[i] % n;
                    }
                    let weight = self.f.map_or(1.0, |f| {
                        f[(self.cre[self.order - 1], self.ann[self.order - 1] % n)]
                    });
                    self.out[index] += self.bra[target] * coefficient * weight;
                    return Ok(());
                }
                let slot = self.order - 1 - depth;
                let spin = self.ann[slot] / n;
                for p in 0..n {
                    if let Some((next, phase)) = create(bits, spin * n + p) {
                        self.cre[slot] = p;
                        self.insert(next, depth + 1, coefficient * phase)?;
                    }
                }
                Ok(())
            }
        }
        let mut accumulator = Accumulator {
            engine: self,
            bra,
            order,
            out_order,
            f,
            out: &mut out,
            ann: [0; 4],
            cre: [0; 4],
        };
        for (source, &coefficient) in ket.iter().enumerate() {
            if coefficient != 0.0 {
                accumulator.remove(self.bits(source), 0, coefficient)?;
            }
        }
        Ok(out)
    }
}

fn annihilate(bits: u128, index: usize) -> Option<(u128, f64)> {
    let bit = 1u128 << index;
    if bits & bit == 0 {
        return None;
    }
    let phase = if (bits & (bit - 1)).count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    Some((bits ^ bit, phase))
}
fn create(bits: u128, index: usize) -> Option<(u128, f64)> {
    let bit = 1u128 << index;
    if bits & bit != 0 {
        return None;
    }
    let phase = if (bits & (bit - 1)).count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    Some((bits | bit, phase))
}
