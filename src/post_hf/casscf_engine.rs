// SPDX-License-Identifier: Apache-2.0
//! Spin-targeted CASCI and CASSCF with a common inactive-core contraction.
//! Orbital optimization uses a frozen-CI diagonal Hessian and Pulay extrapolation
//! of accumulated exponential-rotation parameters. No full-space 2-RDM is stored.
use crate::context::{MoIntegralCache, Reference, active_integrals_from_mo};
use crate::diis::CdiisEngine;
use crate::error::HfError;
use crate::fci::engine::{FciHamiltonian, FciSolverOptions, FciState};
use crate::matrix::Matrix;
use crate::mo_integrals::{MoEriTensor, transform_1e_ao_to_mo, transform_eri_ao_to_mo};
use crate::options::{ActiveSpace, PostHFOptions, ResolvedActiveSpace};
use crate::output::{PostHFOutput, State};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct CasResult<T = Vec<FciState>> {
    pub output: PostHFOutput,
    pub mo_coefficients: Matrix,
    pub active_space: ResolvedActiveSpace,
    pub states: T,
    pub state_one_rdms: Vec<Matrix>,
    pub state_two_rdms: Vec<Vec<f64>>,
    pub averaged_one_rdm: Matrix,
    pub averaged_two_rdm: Vec<f64>,
    /// F[p,q] has its density index in the second column.
    pub generalized_fock: Matrix,
    pub orbital_gradient_max: f64,
    pub iterations: usize,
}

/// Generalized Fock element, including the inactive-active exchange terms.
/// The active 2-RDM is spin traced, normal ordered, in (p,q,r,s) order.
pub fn generalized_fock_element(
    p: usize,
    q: usize,
    nc: usize,
    d: &Matrix,
    gamma: &[f64],
    h: impl Fn(usize, usize) -> f64,
    g: impl Fn(usize, usize, usize, usize) -> f64,
) -> f64 {
    let na = d.rows;
    if q < nc {
        let mut f = 2.0 * h(p, q);
        for i in 0..nc {
            f += 4.0 * g(p, q, i, i) - 2.0 * g(p, i, i, q);
        }
        for s in 0..na {
            for t in 0..na {
                f += d[(s, t)] * (2.0 * g(p, q, nc + s, nc + t) - g(p, nc + t, nc + s, q));
            }
        }
        f
    } else if q < nc + na {
        let v = q - nc;
        let mut f = 0.0;
        for r in 0..na {
            let rr = nc + r;
            f += d[(v, r)]
                * (h(p, rr)
                    + (0..nc)
                        .map(|i| 2.0 * g(p, rr, i, i) - g(p, i, i, rr))
                        .sum::<f64>());
            for s in 0..na {
                for t in 0..na {
                    f += g(p, rr, nc + s, nc + t) * gamma[((v * na + r) * na + s) * na + t];
                }
            }
        }
        f
    } else {
        0.0
    }
}

/// Frozen-CI curvature along C exp(theta (|p><q|-|q><p|)).
pub fn orbital_curvature(
    p: usize,
    q: usize,
    nc: usize,
    d: &Matrix,
    gamma: &[f64],
    h: &Matrix,
    get: impl Fn(usize, usize, usize, usize) -> f64,
) -> f64 {
    let dh = |a, b| {
        let mut x = 0.0;
        if a == q {
            x += h[(p, b)];
        }
        if a == p {
            x -= h[(q, b)];
        }
        if b == q {
            x += h[(a, p)];
        }
        if b == p {
            x -= h[(a, q)];
        }
        x
    };
    let dg = |a, b, c, e| {
        let indices = [a, b, c, e];
        let mut value = 0.0;
        for leg in 0..4 {
            let mut x = indices;
            let sign = if x[leg] == q {
                x[leg] = p;
                1.0
            } else if x[leg] == p {
                x[leg] = q;
                -1.0
            } else {
                continue;
            };
            value += sign * get(x[0], x[1], x[2], x[3]);
        }
        value
    };
    2.0 * (generalized_fock_element(p, q, nc, d, gamma, dh, dg)
        - generalized_fock_element(q, p, nc, d, gamma, dh, dg))
}

/// Active eigensolver payload and spin-traced densities for the shared orbital optimizer.
pub(crate) struct ActiveSolution<T> {
    pub states: T,
    pub energies: Vec<(f64, f64)>,
    pub one: Vec<Matrix>,
    pub two: Vec<Vec<f64>>,
}

struct Evaluation<T> {
    c: Matrix,
    h: Matrix,
    eri: Arc<MoEriTensor>,
    permutation: Vec<usize>,
    states: T,
    energies: Vec<(f64, f64)>,
    one: Vec<Matrix>,
    two: Vec<Vec<f64>>,
    avg_one: Matrix,
    avg_two: Vec<f64>,
    f: Matrix,
    objective: f64,
}
impl<T> Evaluation<T> {
    fn g(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        self.eri.get(
            self.permutation[p],
            self.permutation[q],
            self.permutation[r],
            self.permutation[s],
        )
    }
    fn gradient(&self) -> Matrix {
        self.f.sub(&self.f.transpose()).scale(2.0)
    }
}

fn ordered_space(original: &ResolvedActiveSpace, n: usize) -> ResolvedActiveSpace {
    let nc = original.inactive.len();
    let na = original.active.len();
    ResolvedActiveSpace {
        n_alpha: original.n_alpha,
        n_beta: original.n_beta,
        inactive: (0..nc).collect(),
        active: (nc..nc + na).collect(),
        external: (nc + na..n).collect(),
    }
}
fn weights(options: &PostHFOptions) -> Vec<f64> {
    if options.cas.state_weights.is_empty() {
        let mut w = vec![0.0; options.n_roots];
        w[options.root] = 1.0;
        w
    } else {
        let total = options.cas.state_weights.iter().sum::<f64>();
        options
            .cas
            .state_weights
            .iter()
            .map(|x| x / total)
            .collect()
    }
}
fn fci_solution(
    h: &Matrix,
    eri: &MoEriTensor,
    space: &ResolvedActiveSpace,
    core_energy: f64,
    options: &PostHFOptions,
) -> Result<ActiveSolution<Vec<FciState>>, HfError> {
    let engine = FciHamiltonian::new(h, eri, space.n_alpha, space.n_beta, core_energy)?;
    let sz = (space.n_alpha as f64 - space.n_beta as f64).abs() * 0.5;
    let solution = engine.solve(&FciSolverOptions {
        n_roots: options.n_roots,
        max_iterations: options.max_iter,
        residual_tol: options.residual_tol,
        target_s_squared: Some(options.cas.target_s_squared.unwrap_or(sz * (sz + 1.0))),
        ..Default::default()
    })?;
    let mut one = Vec::new();
    let mut two = Vec::new();
    for state in &solution.states {
        one.push(Matrix::from_vec(
            h.rows,
            h.rows,
            engine.rdm(&state.vector, 1)?,
        ));
        two.push(engine.rdm(&state.vector, 2)?);
    }
    Ok(ActiveSolution {
        energies: solution
            .states
            .iter()
            .map(|s| (s.energy, s.s_squared))
            .collect(),
        states: solution.states,
        one,
        two,
    })
}

fn evaluate<T>(
    reference: Reference<'_>,
    c: Matrix,
    space: &ResolvedActiveSpace,
    options: &PostHFOptions,
    weights: &[f64],
    initial: Option<(Arc<MoEriTensor>, Vec<usize>)>,
    solve: &mut impl FnMut(
        &Matrix,
        &MoEriTensor,
        &ResolvedActiveSpace,
        f64,
        &PostHFOptions,
    ) -> Result<ActiveSolution<T>, HfError>,
) -> Result<Evaluation<T>, HfError> {
    let h = transform_1e_ao_to_mo(&c, reference.core());
    let (eri, permutation) = initial.unwrap_or_else(|| {
        (
            Arc::new(transform_eri_ao_to_mo(&c, reference.eri())),
            (0..c.cols).collect(),
        )
    });
    let get = |p: usize, q: usize, r: usize, s: usize| {
        eri.get(
            permutation[p],
            permutation[q],
            permutation[r],
            permutation[s],
        )
    };
    let ints = active_integrals_from_mo(&h, get, space.clone(), reference.nuclear_energy())?;
    let na = space.active.len();
    let nc = space.inactive.len();
    let rdm_bytes = na
        .checked_pow(4)
        .and_then(|x| x.checked_mul(options.n_roots + 1))
        .and_then(|x| x.checked_mul(8));
    if rdm_bytes.is_none_or(|x| x > 256 * 1024 * 1024) {
        return Err(HfError::Unsupported {
            method: "CASSCF/CASCI".into(),
            detail: "state 2-RDM storage exceeds 256 MiB".into(),
        });
    }
    let mut avg_one = Matrix::zeros(na, na);
    let mut avg_two = vec![0.0; na.pow(4)];
    let mut objective = 0.0;
    let solution = solve(&ints.h, &ints.eri, space, ints.core_energy, options)?;
    if solution.energies.len() != options.n_roots
        || solution.one.len() != options.n_roots
        || solution.two.len() != options.n_roots
        || solution
            .energies
            .iter()
            .any(|&(e, s)| !e.is_finite() || !s.is_finite())
        || solution.one.iter().any(|d| {
            d.rows != na
                || d.cols != na
                || d.data.len() != na * na
                || d.data.iter().any(|x| !x.is_finite())
        })
        || solution
            .two
            .iter()
            .any(|d| d.len() != na.pow(4) || d.iter().any(|x| !x.is_finite()))
    {
        return Err(HfError::Response(
            "invalid active-solver energies or densities".into(),
        ));
    }
    for (root, &w) in weights.iter().enumerate() {
        for (x, y) in avg_one.data.iter_mut().zip(&solution.one[root].data) {
            *x += w * y;
        }
        for (x, y) in avg_two.iter_mut().zip(&solution.two[root]) {
            *x += w * y;
        }
        objective += w * solution.energies[root].0;
    }
    let mut f = Matrix::zeros(c.cols, c.cols);
    for p in 0..c.cols {
        for q in 0..nc + na {
            f[(p, q)] =
                generalized_fock_element(p, q, nc, &avg_one, &avg_two, |a, b| h[(a, b)], get);
        }
    }
    Ok(Evaluation {
        c,
        h,
        eri,
        permutation,
        states: solution.states,
        energies: solution.energies,
        one: solution.one,
        two: solution.two,
        avg_one,
        avg_two,
        f,
        objective,
    })
}
fn pack(matrix: &Matrix, pairs: &[(usize, usize)]) -> Vec<f64> {
    pairs.iter().map(|&(p, q)| matrix[(p, q)]).collect()
}
fn skew(values: &[f64], pairs: &[(usize, usize)], n: usize) -> Matrix {
    let mut k = Matrix::zeros(n, n);
    for (&x, &(p, q)) in values.iter().zip(pairs) {
        k[(p, q)] = x;
        k[(q, p)] = -x;
    }
    k
}
fn max_abs(x: &[f64]) -> f64 {
    x.iter().map(|x| x.abs()).fold(0.0, f64::max)
}
fn limit_step(step: &mut [f64], limit: f64) {
    let scale = (limit / max_abs(step).max(limit)).min(1.0);
    for x in step {
        *x *= scale;
    }
}
/// Pull back the physical orbital gradient through the exponential map.
fn parameter_gradient(k: &Matrix, g: &Matrix) -> Result<Matrix, HfError> {
    let mut result = g.clone();
    let mut term = g.clone();
    for order in 1..100 {
        term = k
            .dot(&term)
            .sub(&term.dot(k))
            .scale(1.0 / (order + 1) as f64);
        result = result.add(&term);
        if max_abs(&term.data) < 1e-15 {
            return Ok(result);
        }
    }
    Err(HfError::Response(
        "CASSCF exponential-map derivative failed to converge".into(),
    ))
}

pub fn solve_cas(
    reference: Reference<'_>,
    options: &PostHFOptions,
    cache: Option<&MoIntegralCache<'_>>,
    optimize: bool,
) -> Result<CasResult, HfError> {
    solve_cas_impl(
        reference,
        options,
        cache,
        optimize,
        false,
        None,
        &mut fci_solution,
    )
}

/// Reuse the CAS orbital optimizer with a non-CI active-space solver.
pub(crate) fn solve_cas_with_solver<T>(
    reference: Reference<'_>,
    options: &PostHFOptions,
    cache: Option<&MoIntegralCache<'_>>,
    optimize: bool,
    solve: &mut impl FnMut(
        &Matrix,
        &MoEriTensor,
        &ResolvedActiveSpace,
        f64,
        &PostHFOptions,
    ) -> Result<ActiveSolution<T>, HfError>,
) -> Result<CasResult<T>, HfError> {
    // Finite-M MPS manifolds are not invariant under active-active rotations.
    solve_cas_impl(reference, options, cache, optimize, true, None, solve)
}

/// Re-diagonalize CASCI in an explicitly supplied orthonormal MO basis.
/// The AO reference and ERIs remain borrowed; no HFResult/ERI clone is needed.
pub fn solve_cas_in_orbitals(
    reference: Reference<'_>,
    coefficients: &Matrix,
    options: &PostHFOptions,
) -> Result<CasResult, HfError> {
    let original = reference.coefficients(0);
    if coefficients.rows != original.rows
        || coefficients.cols != original.cols
        || coefficients.data.len() != original.rows * original.cols
        || coefficients.data.iter().any(|x| !x.is_finite())
    {
        return Err(HfError::InvalidInput(
            "incompatible CASCI orbital coefficients".into(),
        ));
    }
    let overlap = match reference {
        Reference::Rhf(hf) => &hf.overlap_matrix,
        Reference::Uhf(hf) => &hf.overlap_matrix,
    };
    let metric = coefficients.transpose().dot(overlap).dot(coefficients);
    for p in 0..metric.rows {
        for q in 0..metric.cols {
            if (metric[(p, q)] - if p == q { 1. } else { 0. }).abs() > 1e-8 {
                return Err(HfError::InvalidInput(
                    "CASCI orbitals must be orthonormal in the AO metric".into(),
                ));
            }
        }
    }
    solve_cas_impl(
        reference,
        options,
        None,
        false,
        false,
        Some(coefficients),
        &mut fci_solution,
    )
}

fn solve_cas_impl<T>(
    reference: Reference<'_>,
    options: &PostHFOptions,
    cache: Option<&MoIntegralCache<'_>>,
    optimize: bool,
    active_rotations: bool,
    coefficients: Option<&Matrix>,
    solve: &mut impl FnMut(
        &Matrix,
        &MoEriTensor,
        &ResolvedActiveSpace,
        f64,
        &PostHFOptions,
    ) -> Result<ActiveSolution<T>, HfError>,
) -> Result<CasResult<T>, HfError> {
    options.validate()?;
    if !reference.converged() {
        return Err(HfError::Scf(
            "CASSCF/CASCI requires a converged reference".into(),
        ));
    }
    if options.density_fitting {
        return Err(HfError::Unsupported {
            method: "CASSCF/CASCI".into(),
            detail: "DF orbital optimizer is not implemented".into(),
        });
    }
    let c_ref = coefficients.unwrap_or_else(|| reference.coefficients(0));
    let (n_alpha, n_beta) = reference.occupations();
    let default_full = ActiveSpace::new(n_alpha, n_beta, c_ref.cols);
    let active = if optimize {
        options.active.as_ref().ok_or_else(|| {
            HfError::InvalidInput("CASSCF requires an explicit ActiveSpace".into())
        })?
    } else {
        options.active.as_ref().unwrap_or(&default_full)
    };
    let resolved = active.resolve(c_ref.cols, n_alpha, n_beta)?;
    if options.frozen_core > resolved.inactive.len()
        || options.frozen_virtual > resolved.external.len()
    {
        return Err(HfError::InvalidInput(
            "frozen orbitals exceed inactive/external space".into(),
        ));
    }
    let permutation = resolved.permutation();
    let n = c_ref.cols;
    let mut c0 = Matrix::zeros(c_ref.rows, n);
    for p in 0..n {
        for mu in 0..c_ref.rows {
            c0[(mu, p)] = c_ref[(mu, permutation[p])];
        }
    }
    let space = ordered_space(&resolved, n);
    let nc = space.inactive.len();
    let na = space.active.len();
    let group = |p| {
        if p < nc {
            0
        } else if p < nc + na {
            1
        } else {
            2
        }
    };
    let mut pairs = Vec::new();
    for p in 0..n - options.frozen_virtual {
        for q in options.frozen_core..p {
            if group(p) != group(q) || (active_rotations && group(p) == 1 && group(q) == 1) {
                pairs.push((p, q));
            }
        }
    }
    let weights = weights(options);
    let initial = cache
        .map(|cache| cache.full().map(|g| (g.clone(), permutation.clone())))
        .transpose()?;
    let mut current = evaluate(
        reference,
        c0.clone(),
        &space,
        options,
        &weights,
        initial,
        solve,
    )?;
    let mut parameters = vec![0.0; pairs.len()];
    let mut diis = CdiisEngine::new(8);
    let mut iterations = 0;
    if optimize && !pairs.is_empty() {
        let mut converged = false;
        for iteration in 0..options.cas.max_orbital_iterations {
            iterations = iteration + 1;
            let physical = current.gradient();
            let physical_values = pack(&physical, &pairs);
            let maximum = max_abs(&physical_values);
            if maximum < options.cas.orbital_gradient_tol {
                converged = true;
                break;
            }
            let k = skew(&parameters, &pairs, n);
            let grad = pack(&parameter_gradient(&k, &physical)?, &pairs);
            diis.push(
                Matrix::from_vec(parameters.len(), 1, parameters.clone()),
                Matrix::from_vec(grad.len(), 1, grad.clone()),
            );
            let mut directions = Vec::new();
            if diis.fock_history.len() >= 3
                && let Some(extrap) = diis.extrapolate() {
                    let mut step = extrap
                        .data
                        .iter()
                        .zip(&parameters)
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>();
                    limit_step(&mut step, 2.0 * options.cas.max_rotation);
                    if step.iter().zip(&grad).map(|(a, b)| a * b).sum::<f64>() < 0.0 {
                        directions.push(step);
                    }
                }
            let mut step = pairs
                .iter()
                .zip(&grad)
                .map(|(&(p, q), &g)| {
                    let diag = orbital_curvature(
                        p,
                        q,
                        nc,
                        &current.avg_one,
                        &current.avg_two,
                        &current.h,
                        |p, q, r, s| current.g(p, q, r, s),
                    );
                    -g / diag.abs().max(0.05)
                })
                .collect::<Vec<_>>();
            limit_step(&mut step, options.cas.max_rotation);
            directions.push(step);
            let mut accepted = None;
            for direction in directions {
                let slope = grad.iter().zip(&direction).map(|(a, b)| a * b).sum::<f64>();
                for backtrack in 0..12 {
                    let fraction = 0.5_f64.powi(backtrack);
                    let candidate = parameters
                        .iter()
                        .zip(&direction)
                        .map(|(a, b)| a + fraction * b)
                        .collect::<Vec<_>>();
                    let c = c0.dot(&skew(&candidate, &pairs, n).exp());
                    let next = evaluate(reference, c, &space, options, &weights, None, solve)?;
                    let decreases = next.objective <= current.objective + 1e-4 * fraction * slope;
                    let roundoff = (next.objective - current.objective).abs() < 1e-13
                        && max_abs(&pack(&next.gradient(), &pairs)) < maximum;
                    if decreases || roundoff {
                        accepted = Some((candidate, next));
                        break;
                    }
                }
                if accepted.is_some() {
                    break;
                }
            }
            if let Some((candidate, next)) = accepted {
                parameters = candidate;
                current = next;
            } else {
                return Err(HfError::PostHf {
                    method: "CASSCF".into(),
                    detail: format!("orbital line search failed: max gradient {maximum:.3e}"),
                });
            }
        }
        if !converged
            && max_abs(&pack(&current.gradient(), &pairs)) >= options.cas.orbital_gradient_tol
        {
            return Err(HfError::PostHf {
                method: "CASSCF".into(),
                detail: format!(
                    "orbital optimization did not converge in {iterations} iterations; max gradient {:.3e}",
                    max_abs(&pack(&current.gradient(), &pairs))
                ),
            });
        }
    }
    let mut full_one = Matrix::zeros(n, n);
    for i in 0..nc {
        full_one[(i, i)] = 2.0;
    }
    for p in 0..na {
        for q in 0..na {
            full_one[(nc + p, nc + q)] = current.one[options.root][(p, q)];
        }
    }
    let method = if optimize { "CASSCF" } else { "FCI" };
    let mut output =
        PostHFOutput::new(method, reference.energy(), current.energies[options.root].0)?;
    output.states = current
        .energies
        .iter()
        .zip(&weights)
        .map(|(&(energy, s_squared), &weight)| State {
            energy,
            s_squared: Some(s_squared),
            weight,
            converged: true,
        })
        .collect();
    output.one_rdm = Some(full_one);
    output.select_root(options.root)?;
    output.components.insert(
        if optimize { "E_CASSCF" } else { "E_CASCI" }.into(),
        output.e_total,
    );
    output
        .components
        .insert("E_state_average".into(), current.objective);
    output.validate()?;
    let orbital_gradient_max = max_abs(&pack(&current.gradient(), &pairs));
    Ok(CasResult {
        output,
        mo_coefficients: current.c,
        active_space: space,
        states: current.states,
        state_one_rdms: current.one,
        state_two_rdms: current.two,
        averaged_one_rdm: current.avg_one,
        averaged_two_rdm: current.avg_two,
        generalized_fock: current.f,
        orbital_gradient_max,
        iterations,
    })
}

impl<T> super::PostHFResult for CasResult<T> {
    fn method_name(&self) -> &'static str {
        if self.output.method == "CASSCF" {
            "CASSCF"
        } else {
            "FCI"
        }
    }
    fn reference_energy(&self) -> f64 {
        self.output.e_scf
    }
    fn correlation_energy(&self) -> f64 {
        self.output.e_corr
    }
    fn total_energy(&self) -> f64 {
        self.output.e_total
    }
    fn one_rdm(&self) -> Option<&Matrix> {
        self.output.one_rdm.as_ref()
    }
    fn to_output(&self) -> Result<PostHFOutput, HfError> {
        Ok(self.output.clone())
    }
}

macro_rules! cas_adapter {
    ($name:ident,$method:ident,$optimize:literal) => {
        pub struct $name<'a> {
            context: &'a crate::context::PostHFContext<'a>,
            options: PostHFOptions,
        }
        impl crate::registry::MethodMeta for $name<'_> {
            const METHOD: crate::registry::MethodId = crate::registry::MethodId::$method;
        }
        impl<'a> crate::registry::ContextMethod<'a> for $name<'a> {
            fn from_context(
                context: &'a crate::context::PostHFContext<'a>,
                options: &PostHFOptions,
            ) -> Result<Self, HfError> {
                options.validate()?;
                if $optimize && options.active.is_none() {
                    return Err(HfError::InvalidInput(
                        "CASSCF requires an explicit ActiveSpace".into(),
                    ));
                }
                Ok(Self {
                    context,
                    options: options.clone(),
                })
            }
        }
        impl super::PostHFMethod for $name<'_> {
            type Output = CasResult;
            fn compute(&self) -> Result<CasResult, String> {
                solve_cas(
                    self.context.reference,
                    &self.options,
                    Some(&self.context.ints),
                    $optimize,
                )
                .map_err(|e| e.to_string())
            }
        }
    };
}
cas_adapter!(Casci, Fci, false);
cas_adapter!(OrbitalOptimizedCas, Casscf, true);
