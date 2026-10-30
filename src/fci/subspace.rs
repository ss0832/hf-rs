// SPDX-License-Identifier: Apache-2.0
//! Matrix-free CI in an explicitly generated determinant space.
//! Fermion operations produce only connected matrix elements; no FCI vector or
//! determinant-pair Hamiltonian is required for the iterative solver.
use super::engine::{FciSolution, FciSolverOptions, FciState};
use super::excitations::Determinant;
use super::slater_condon::slater_condon_element;
use crate::error::HfError;
use crate::matrix::Matrix;
use crate::mo_integrals::MoEriTensor;
use std::collections::HashMap;

fn remove(bits: u128, index: usize) -> Option<(u128, f64)> {
    let bit = 1u128 << index;
    (bits & bit != 0).then(|| {
        (
            bits ^ bit,
            if (bits & (bit - 1)).count_ones().is_multiple_of(2) {
                1.0
            } else {
                -1.0
            },
        )
    })
}
fn insert(bits: u128, index: usize) -> Option<(u128, f64)> {
    let bit = 1u128 << index;
    (bits & bit == 0).then(|| {
        (
            bits | bit,
            if (bits & (bit - 1)).count_ones().is_multiple_of(2) {
                1.0
            } else {
                -1.0
            },
        )
    })
}
pub struct SubspaceHamiltonian<'a> {
    pub determinants: Vec<Determinant>,
    pub h: &'a Matrix,
    pub eri: &'a MoEriTensor,
    pub constant: f64,
    n_alpha: usize,
    n_beta: usize,
    bits: Vec<u128>,
    addresses: HashMap<u128, usize>,
}
impl<'a> SubspaceHamiltonian<'a> {
    pub fn new(
        h: &'a Matrix,
        eri: &'a MoEriTensor,
        constant: f64,
        determinants: Vec<Determinant>,
    ) -> Result<Self, HfError> {
        let n = h.rows;
        let np = n * (n + 1) / 2;
        if n > 64
            || h.cols != n
            || eri.n_basis != n
            || eri.n_pairs != np
            || eri.data.len() != np * np
            || !constant.is_finite()
            || h.data.iter().chain(&eri.data).any(|x| !x.is_finite())
            || determinants.is_empty()
            || determinants.len() > 2_000_000
        {
            return Err(HfError::InvalidInput(
                "invalid CI subspace Hamiltonian or storage size".into(),
            ));
        }
        let n_alpha = determinants[0].0.n_electrons();
        let n_beta = determinants[0].1.n_electrons();
        let allowed = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        if determinants.iter().any(|&(a, b)| {
            (a.0 | b.0) & !allowed != 0 || a.n_electrons() != n_alpha || b.n_electrons() != n_beta
        }) {
            return Err(HfError::InvalidInput(
                "CI determinants have inconsistent orbital/spin counts".into(),
            ));
        }
        let bits = determinants
            .iter()
            .map(|&(a, b)| a.0 as u128 | ((b.0 as u128) << n))
            .collect::<Vec<_>>();
        let addresses = bits
            .iter()
            .enumerate()
            .map(|(i, &b)| (b, i))
            .collect::<HashMap<_, _>>();
        if addresses.len() != bits.len() {
            return Err(HfError::InvalidInput("duplicate CI determinants".into()));
        }
        Ok(Self {
            determinants,
            h,
            eri,
            constant,
            n_alpha,
            n_beta,
            bits,
            addresses,
        })
    }
    pub fn dimension(&self) -> usize {
        self.bits.len()
    }
    fn check(&self, c: &[f64]) -> Result<(), HfError> {
        if c.len() != self.dimension() || c.iter().any(|x| !x.is_finite()) {
            return Err(HfError::InvalidInput("invalid subspace CI vector".into()));
        }
        Ok(())
    }
    fn transitions(
        &self,
        source: usize,
        order: usize,
        mut emit: impl FnMut(usize, [usize; 4], f64),
    ) {
        let n = self.h.rows;
        let original = self.bits[source];
        let mut occupied = original;
        while occupied != 0 {
            let q = occupied.trailing_zeros() as usize;
            occupied &= occupied - 1;
            let Some((first, s1)) = remove(original, q) else {
                continue;
            };
            let spin1 = q / n;
            if order == 1 {
                for p in 0..n {
                    if let Some((next, s2)) = insert(first, spin1 * n + p)
                        && let Some(&target) = self.addresses.get(&next) {
                            emit(target, [p, q % n, 0, 0], s1 * s2);
                        }
                }
            } else {
                let mut occupied2 = first;
                while occupied2 != 0 {
                    let s = occupied2.trailing_zeros() as usize;
                    occupied2 &= occupied2 - 1;
                    let Some((second, s2)) = remove(first, s) else {
                        continue;
                    };
                    let spin2 = s / n;
                    for r in 0..n {
                        if let Some((third, s3)) = insert(second, spin2 * n + r) {
                            for p in 0..n {
                                if let Some((last, s4)) = insert(third, spin1 * n + p)
                                    && let Some(&target) = self.addresses.get(&last) {
                                        emit(target, [p, q % n, r, s % n], s1 * s2 * s3 * s4);
                                    }
                            }
                        }
                    }
                }
            }
        }
    }
    pub fn apply(&self, c: &[f64]) -> Result<Vec<f64>, HfError> {
        self.check(c)?;
        let mut out = c.iter().map(|x| self.constant * x).collect::<Vec<_>>();
        for (source, &value) in c.iter().enumerate() {
            if value != 0.0 {
                self.transitions(source, 1, |target, [p, q, _, _], sign| {
                    out[target] += value * sign * self.h[(p, q)]
                });
                self.transitions(source, 2, |target, [p, q, r, s], sign| {
                    out[target] += 0.5 * value * sign * self.eri.get(p, q, r, s)
                });
            }
        }
        Ok(out)
    }
    pub fn diagonal(&self) -> Vec<f64> {
        self.determinants
            .iter()
            .map(|&(a, b)| self.constant + slater_condon_element(&a, &b, &a, &b, self.h, self.eri))
            .collect()
    }
    pub fn rdm(&self, c: &[f64], order: usize) -> Result<Vec<f64>, HfError> {
        self.check(c)?;
        if !(1..=2).contains(&order) {
            return Err(HfError::InvalidInput(
                "subspace RDM order must be one or two".into(),
            ));
        }
        let length = self
            .h
            .rows
            .checked_pow((2 * order) as u32)
            .ok_or_else(|| HfError::InvalidInput("subspace RDM size overflow".into()))?;
        if length > 32 * 1024 * 1024 {
            return Err(HfError::Unsupported {
                method: "CI RDM".into(),
                detail: "materialized RDM exceeds 256 MiB".into(),
            });
        }
        let norm = c.iter().map(|x| x * x).sum::<f64>();
        if norm <= 0.0 {
            return Err(HfError::InvalidInput("zero subspace CI vector".into()));
        }
        let n = self.h.rows;
        let mut out = vec![0.0; length];
        for (source, &ket) in c.iter().enumerate() {
            if ket != 0.0 {
                self.transitions(source, order, |target, [p, q, r, s], phase| {
                    let i = if order == 1 {
                        p * n + q
                    } else {
                        ((p * n + q) * n + r) * n + s
                    };
                    out[i] += phase * c[target] * ket / norm;
                });
            }
        }
        Ok(out)
    }
    pub fn spin_squared_action(&self, c: &[f64]) -> Result<Vec<f64>, HfError> {
        self.spin_action(c, true)
    }
    /// Expectation value with omitted bra determinants equal to zero; unlike
    /// spin projection this does not require a spin-complete selected space.
    pub fn spin_squared_expectation(&self, c: &[f64]) -> Result<f64, HfError> {
        let action = self.spin_action(c, false)?;
        let norm = c.iter().map(|x| x * x).sum::<f64>();
        if norm == 0.0 {
            return Err(HfError::InvalidInput(
                "zero CI spin expectation vector".into(),
            ));
        }
        Ok(c.iter().zip(action).map(|(a, b)| a * b).sum::<f64>() / norm)
    }
    fn spin_action(&self, c: &[f64], strict: bool) -> Result<Vec<f64>, HfError> {
        self.check(c)?;
        let n = self.h.rows;
        let sz = (self.n_alpha as f64 - self.n_beta as f64) * 0.5;
        let mut out = c.iter().map(|x| sz * (sz + 1.0) * x).collect::<Vec<_>>();
        for (source, &coefficient) in c.iter().enumerate() {
            if coefficient == 0.0 {
                continue;
            }
            for p in 0..n {
                let Some((raised, s1)) = remove(self.bits[source], n + p)
                    .and_then(|(b, s)| insert(b, p).map(|(b, t)| (b, s * t)))
                else {
                    continue;
                };
                for q in 0..n {
                    let Some((lowered, s2)) = remove(raised, q)
                        .and_then(|(b, s)| insert(b, n + q).map(|(b, t)| (b, s * t)))
                    else {
                        continue;
                    };
                    let target = self.addresses.get(&lowered);
                    if target.is_none() && !strict {
                        continue;
                    }
                    let &target = target.ok_or_else(|| HfError::Unsupported {
                        method: "spin-targeted CI".into(),
                        detail: "determinant subspace is not closed under S²".into(),
                    })?;
                    out[target] += coefficient * s1 * s2;
                }
            }
        }
        Ok(out)
    }
    fn validate_target(&self, target: f64) -> Result<(), HfError> {
        let sz = (self.n_alpha as f64 - self.n_beta as f64).abs() * 0.5;
        let electrons = self.n_alpha + self.n_beta;
        let maximum = electrons.min(2 * self.h.rows - electrons) as f64 * 0.5;
        let spin = (target + 0.25).sqrt() - 0.5;
        if !target.is_finite()
            || target < 0.0
            || spin < sz - 1e-9
            || spin > maximum + 1e-9
            || ((spin - sz) - (spin - sz).round()).abs() > 1e-9
        {
            return Err(HfError::InvalidInput(
                "invalid subspace CI target spin".into(),
            ));
        }
        Ok(())
    }
    /// Spin multiplicity within a spin-complete space, counted by spatial
    /// occupation patterns rather than constructing a dense S² matrix.
    pub fn target_spin_dimension(&self, target: f64) -> Result<usize, HfError> {
        self.validate_target(target)?;
        let choose = |n: usize, k: usize| -> usize {
            if k > n {
                return 0;
            }
            let mut value = 1u128;
            for i in 0..k.min(n - k) {
                value = value * (n - i) as u128 / (i + 1) as u128;
            }
            value as usize
        };
        let twice_spin = (2.0 * ((target + 0.25).sqrt() - 0.5)).round() as usize;
        let mut patterns = HashMap::new();
        for &(a, b) in &self.determinants {
            *patterns.entry((a.0 & b.0, a.0 ^ b.0)).or_insert(0usize) += 1;
        }
        let mut total = 0usize;
        for ((doubly, singly), count) in patterns {
            let k = singly.count_ones() as usize;
            let alpha = self.n_alpha - doubly.count_ones() as usize;
            if count != choose(k, alpha) {
                return Err(HfError::Unsupported {
                    method: "spin-targeted CI".into(),
                    detail: "determinant space is not spin complete".into(),
                });
            }
            if twice_spin > k {
                continue;
            }
            let up = (k + twice_spin) / 2;
            total += choose(k, up) - choose(k, up + 1);
        }
        Ok(total)
    }
    fn project_spin(&self, c: &[f64], target: f64) -> Result<Vec<f64>, HfError> {
        let electrons = self.n_alpha + self.n_beta;
        let maximum = electrons.min(2 * self.h.rows - electrons) as f64 * 0.5;
        let mut spin = (self.n_alpha as f64 - self.n_beta as f64).abs() * 0.5;
        let mut out = c.to_vec();
        while spin <= maximum + 1e-9 {
            let eigenvalue = spin * (spin + 1.0);
            if (eigenvalue - target).abs() > 1e-9 {
                let ss = self.spin_squared_action(&out)?;
                for (x, y) in out.iter_mut().zip(ss) {
                    *x = (y - eigenvalue * (*x)) / (target - eigenvalue);
                }
            }
            spin += 1.0;
        }
        Ok(out)
    }
    pub fn solve(&self, options: &FciSolverOptions) -> Result<FciSolution, HfError> {
        let dim = self.dimension();
        if options.n_roots == 0
            || options.n_roots > dim
            || options.max_iterations == 0
            || !options.residual_tol.is_finite()
            || options.residual_tol <= 0.0
        {
            return Err(HfError::InvalidInput(
                "invalid subspace CI solver options".into(),
            ));
        }
        if let Some(target) = options.target_s_squared {
            self.validate_target(target)?;
            if options.n_roots > self.target_spin_dimension(target)? {
                return Err(HfError::InvalidInput(
                    "requested more CI roots than the selected spin sector contains".into(),
                ));
            }
        }
        // Use projected Davidson even in small spaces: its trial vectors stay in
        // the requested spin sector without any artificial spin penalty.
        let subspace = options.max_subspace.max(4 * options.n_roots).min(dim);
        if dim
            .checked_mul(2 * subspace + 8)
            .and_then(|n| n.checked_mul(8))
            .is_none_or(|n| n > options.memory_budget)
        {
            return Err(HfError::Unsupported {
                method: "CI".into(),
                detail: "Davidson vectors exceed memory budget".into(),
            });
        }
        let result = crate::solvers::davidson::try_davidson_solver_projected(
            dim,
            &self.diagonal(),
            |c| self.apply(c),
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
                method: "CI".into(),
                detail: format!(
                    "subspace Davidson failed after {} iterations",
                    result.iterations
                ),
            });
        }
        let mut states = Vec::new();
        for mut vector in result.eigenvectors {
            let norm = vector.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm <= 0.0 {
                return Err(HfError::Response("zero subspace CI solution".into()));
            }
            for x in &mut vector {
                *x /= norm;
            }
            let sigma = self.apply(&vector)?;
            let energy = sigma.iter().zip(&vector).map(|(a, b)| a * b).sum::<f64>();
            let residual_norm = sigma
                .iter()
                .zip(&vector)
                .map(|(a, b)| (a - energy * b).powi(2))
                .sum::<f64>()
                .sqrt();
            let s_squared = self
                .spin_action(&vector, options.target_s_squared.is_some())?
                .iter()
                .zip(&vector)
                .map(|(a, b)| a * b)
                .sum::<f64>();
            if residual_norm > options.residual_tol
                || options
                    .target_s_squared
                    .is_some_and(|t| (t - s_squared).abs() > 1e-8)
            {
                return Err(HfError::PostHf {
                    method: "CI".into(),
                    detail: format!("CI residual {residual_norm:.3e}, S²={s_squared:.10}"),
                });
            }
            states.push(FciState {
                energy,
                vector,
                s_squared,
                residual_norm,
            });
        }
        Ok(FciSolution {
            states,
            iterations: result.iterations,
        })
    }
}
