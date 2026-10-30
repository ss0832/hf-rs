// SPDX-License-Identifier: Apache-2.0
//! Deterministic PCI. The default wall generator follows Zhang–Evangelista
//! (2016), Eqs. 25 and 45–48: filter individual off-diagonal paths and normalize
//! after every shifted-Hamiltonian factor. No coefficient truncation or final
//! variational rediagonalization is part of this propagation.
use super::sci::{PCIGenerator, PCIOptions};
use super::selected_ci::{SelectedModel, en_correction};
use crate::context::{ActiveIntegrals, PostHFContext};
use crate::error::HfError;
use crate::fci::engine::{FciSolverOptions, FciState};
use crate::fci::excitations::{Determinant, singles_doubles};
use crate::fci::slater_condon::slater_condon_element;
use crate::fci::string::SpinString;
use crate::fci::subspace::SubspaceHamiltonian;
use crate::matrix::Matrix;
use crate::mo_integrals::MoEriTensor;
use crate::options::PostHFOptions;
use crate::output::{PostHFOutput, State};
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::{ContextMethod, MethodId, MethodMeta};
use std::collections::BTreeMap;
use std::sync::Arc;

type Wavefunction = BTreeMap<Determinant, f64>;
pub struct ProjectorModel {
    pub model: SelectedModel,
    pub projective_energy: f64,
}

#[derive(Clone, Debug)]
pub struct ProjectorCiResult {
    pub output: PostHFOutput,
    pub model: SelectedModel,
    pub active_integrals: Arc<ActiveIntegrals>,
}
impl ProjectorCiResult {
    pub fn active_two_rdm(&self) -> Result<Vec<f64>, HfError> {
        SubspaceHamiltonian::new(
            &self.active_integrals.h,
            &self.active_integrals.eri,
            self.active_integrals.core_energy,
            self.model.determinants.clone(),
        )?
        .rdm(&self.model.states[0].vector, 2)
    }
}
impl PostHFResult for ProjectorCiResult {
    fn method_name(&self) -> &'static str {
        "PCI"
    }
    fn reference_energy(&self) -> f64 {
        self.output.e_scf
    }
    fn correlation_energy(&self) -> f64 {
        self.output.e_corr
    }
    fn one_rdm(&self) -> Option<&Matrix> {
        self.output.one_rdm.as_ref()
    }
    fn to_output(&self) -> Result<PostHFOutput, HfError> {
        Ok(self.output.clone())
    }
}
pub struct ProjectorCi<'a> {
    ctx: &'a PostHFContext<'a>,
    options: PostHFOptions,
}
impl MethodMeta for ProjectorCi<'_> {
    const METHOD: MethodId = MethodId::Pci;
}
impl<'a> ContextMethod<'a> for ProjectorCi<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        if options.n_roots != 1 || options.cas.target_s_squared.is_some() {
            return Err(HfError::Unsupported {
                method: "PCI".into(),
                detail: "PCI propagates one untargeted ground state".into(),
            });
        }
        Ok(Self {
            ctx,
            options: options.clone(),
        })
    }
}
impl PostHFMethod for ProjectorCi<'_> {
    type Output = ProjectorCiResult;
    fn compute(&self) -> Result<Self::Output, String> {
        self.compute_checked().map_err(|e| e.to_string())
    }
}
impl ProjectorCi<'_> {
    fn compute_checked(&self) -> Result<ProjectorCiResult, HfError> {
        let ints = super::selected_ci::selected_integrals(
            self.ctx.reference,
            &self.options,
            Some(&self.ctx.ints),
            "PCI",
        )?;
        let settings = PCIOptions {
            generator: self.options.sci.projector_generator,
            polynomial_order: self.options.sci.polynomial_order,
            spawning_threshold: self.options.sci.selection_threshold,
            max_steps: self.options.max_iter,
            compute_pt2: self.options.sci.compute_pt2,
        };
        let result = solve_projector(
            &ints.h,
            &ints.eri,
            ints.core_energy,
            ints.space.n_alpha,
            ints.space.n_beta,
            &settings,
            &self.options,
        )?;
        let model = result.model;
        let engine = SubspaceHamiltonian::new(
            &ints.h,
            &ints.eri,
            ints.core_energy,
            model.determinants.clone(),
        )?;
        let state = &model.states[0];
        let d = engine.rdm(&state.vector, 1)?;
        let n = self.ctx.reference.coefficients(0).cols;
        let mut one = Matrix::zeros(n, n);
        for &i in &ints.space.inactive {
            one[(i, i)] = 2.0;
        }
        for (p, &pp) in ints.space.active.iter().enumerate() {
            for (q, &qq) in ints.space.active.iter().enumerate() {
                one[(pp, qq)] = d[p * ints.space.active.len() + q];
            }
        }
        let mut output = PostHFOutput::new(
            "PCI",
            self.ctx.reference.energy(),
            state.energy + model.corrections[0],
        )?;
        output.states = vec![State {
            energy: output.e_total,
            s_squared: Some(state.s_squared),
            weight: 1.0,
            converged: true,
        }];
        output
            .components
            .insert("E_variational".into(), state.energy);
        output
            .components
            .insert("E_projective".into(), result.projective_energy);
        output
            .components
            .insert("Delta_E_PT2".into(), model.corrections[0]);
        output.one_rdm = Some(one);
        output.validate()?;
        Ok(ProjectorCiResult {
            output,
            model,
            active_integrals: ints,
        })
    }
}
fn dot(a: &Wavefunction, b: &Wavefunction) -> f64 {
    a.iter()
        .map(|(d, c)| c * b.get(d).copied().unwrap_or(0.0))
        .sum()
}
fn add(a: &mut Wavefunction, b: &Wavefunction, scale: f64) {
    for (&d, &c) in b {
        *a.entry(d).or_default() += scale * c;
    }
}
fn scale(a: &mut Wavefunction, factor: f64) {
    for c in a.values_mut() {
        *c *= factor;
    }
}
fn normalize(a: &mut Wavefunction) -> Result<(), HfError> {
    let norm = dot(a, a).sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(HfError::Response(
            "PCI propagation produced a zero/non-finite vector".into(),
        ));
    }
    scale(a, 1.0 / norm);
    a.retain(|_, c| *c != 0.0);
    Ok(())
}

struct Propagator<'a> {
    h: &'a Matrix,
    g: &'a MoEriTensor,
    constant: f64,
    na: usize,
    nb: usize,
    limit: usize,
}
impl Propagator<'_> {
    /// (H-shift) c, retaining the diagonal and each |H_IJ c_J| > eta path.
    /// Sources are normalized; no independent parent/coefficient threshold is used.
    fn action(&self, c: &Wavefunction, shift: f64, eta: f64) -> Result<Wavefunction, HfError> {
        let mut out = Wavefunction::new();
        for (&parent, &coefficient) in c {
            if coefficient == 0.0 {
                continue;
            }
            let connected = singles_doubles(self.h.rows, self.na, self.nb, &[parent], 2_000_000)?;
            for child in connected {
                let mut element =
                    slater_condon_element(&child.0, &child.1, &parent.0, &parent.1, self.h, self.g);
                if child == parent {
                    element += self.constant - shift;
                }
                let amplitude = element * coefficient;
                if child == parent || amplitude.abs() > eta {
                    *out.entry(child).or_default() += amplitude;
                    if out.len() > self.limit {
                        return Err(HfError::Unsupported {
                            method: "PCI".into(),
                            detail: "spawning space exceeds max_determinants".into(),
                        });
                    }
                }
            }
        }
        if out.values().any(|v| !v.is_finite()) {
            return Err(HfError::Integral(
                "non-finite PCI spawning amplitude".into(),
            ));
        }
        Ok(out)
    }
    // For recurrence extensions, apply the same threshold to a normalized input
    // and restore its norm. This leaves eta in units of a normalized spawning rate.
    fn homogeneous_action(
        &self,
        c: &Wavefunction,
        shift: f64,
        eta: f64,
    ) -> Result<Wavefunction, HfError> {
        let norm = dot(c, c).sqrt();
        if norm == 0.0 {
            return Ok(Wavefunction::new());
        }
        let mut normalized = c.clone();
        scale(&mut normalized, norm.recip());
        let mut out = self.action(&normalized, shift, eta)?;
        scale(&mut out, norm);
        Ok(out)
    }
    fn recurrence(
        &self,
        c: &Wavefunction,
        center: f64,
        radius: f64,
        eta: f64,
        weights: &[f64],
    ) -> Result<Wavefunction, HfError> {
        let mut previous = c.clone();
        let mut accumulator = Wavefunction::new();
        add(&mut accumulator, &previous, weights[0]);
        if weights.len() == 1 {
            return Ok(accumulator);
        }
        let mut current = self.homogeneous_action(c, center, eta)?;
        scale(&mut current, -1.0 / radius);
        add(&mut accumulator, &current, weights[1]);
        for &weight in &weights[2..] {
            let mut next = self.homogeneous_action(&current, center, eta)?;
            scale(&mut next, -2.0 / radius);
            add(&mut next, &previous, -1.0);
            add(&mut accumulator, &next, weight);
            previous = current;
            current = next;
        }
        Ok(accumulator)
    }
    /// Historical Krylov choices are selected-space accelerators: path selection
    /// determines a model, then the *unfiltered Hermitian H* is used for its
    /// Davidson or fully reorthogonalized Lanczos/Ritz step. Applying Lanczos to
    /// the nonlinear, nonsymmetric path-filtered action would be invalid.
    fn krylov_step(
        &self,
        c: &Wavefunction,
        eta: f64,
        generator: PCIGenerator,
        order: usize,
        options: &PostHFOptions,
    ) -> Result<Wavefunction, HfError> {
        let mut selected = self.action(c, 0.0, eta)?;
        for &d in c.keys() {
            selected.entry(d).or_default();
        }
        let dets = selected.keys().copied().collect::<Vec<_>>();
        let engine = SubspaceHamiltonian::new(self.h, self.g, self.constant, dets.clone())?;
        let vector = if generator == PCIGenerator::Davidson {
            engine
                .solve(&FciSolverOptions {
                    residual_tol: options.residual_tol,
                    max_iterations: options.sci.diagonalization_max_iterations,
                    ..Default::default()
                })?
                .states
                .remove(0)
                .vector
        } else {
            let mut v = dets
                .iter()
                .map(|d| c.get(d).copied().unwrap_or(0.0))
                .collect::<Vec<_>>();
            let mut basis: Vec<Vec<f64>> = Vec::new();
            let mut actions: Vec<Vec<f64>> = Vec::new();
            for _ in 0..order.min(dets.len()) {
                let action = engine.apply(&v)?;
                basis.push(v);
                actions.push(action.clone());
                let mut w = action;
                for _ in 0..2 {
                    for b in &basis {
                        let overlap = b.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>();
                        for (a, b) in w.iter_mut().zip(b) {
                            *a -= overlap * b;
                        }
                    }
                }
                let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-13 {
                    break;
                }
                for x in &mut w {
                    *x /= norm;
                }
                v = w;
            }
            let k = basis.len();
            let mut projected = Matrix::zeros(k, k);
            for i in 0..k {
                for j in 0..k {
                    projected[(i, j)] = basis[i].iter().zip(&actions[j]).map(|(a, b)| a * b).sum();
                }
            }
            let (_, eigenvectors) = projected.eigh();
            (0..dets.len())
                .map(|i| (0..k).map(|j| eigenvectors[(j, 0)] * basis[j][i]).sum())
                .collect()
        };
        Ok(dets.into_iter().zip(vector).collect())
    }
}
fn bessel_i(order: usize, z: f64) -> f64 {
    let mut term = 1.0;
    for i in 1..=order {
        term *= 0.5 * z / i as f64;
    }
    let mut sum = term;
    for j in 1..100 {
        term *= 0.25 * z * z / (j * (j + order)) as f64;
        sum += term;
        if term <= f64::EPSILON * sum {
            break;
        }
    }
    sum
}
fn mask(n: usize) -> u64 {
    if n == 64 { u64::MAX } else { (1u64 << n) - 1 }
}

pub fn solve_projector(
    h: &Matrix,
    g: &MoEriTensor,
    constant: f64,
    na: usize,
    nb: usize,
    settings: &PCIOptions,
    options: &PostHFOptions,
) -> Result<ProjectorModel, HfError> {
    options.validate()?;
    if h.rows > 64
        || na > h.rows
        || nb > h.rows
        || settings.max_steps == 0
        || settings.polynomial_order == 0
        || settings.polynomial_order > 64
        || !settings.spawning_threshold.is_finite()
        || settings.spawning_threshold < 0.0
        || options.n_roots != 1
        || options.cas.target_s_squared.is_some()
        || settings.generator == PCIGenerator::Lanczos && settings.polynomial_order < 2
    {
        return Err(HfError::InvalidInput("invalid PCI space, generator, threshold, or iteration count; PCI propagates one untargeted ground state".into()));
    }
    let seed = (SpinString(mask(na)), SpinString(mask(nb)));
    let initial = SubspaceHamiltonian::new(h, g, constant, vec![seed])?;
    let estimate = initial.diagonal()[0];
    let mut h_norm = 0.0_f64;
    let mut v_norm = 0.0_f64;
    for p in 0..h.rows {
        h_norm = h_norm.max((0..h.rows).map(|q| h[(p, q)].abs()).sum());
        for r in 0..h.rows {
            let row = (0..h.rows)
                .flat_map(|q| (0..h.rows).map(move |s| g.get(p, q, r, s).abs()))
                .sum::<f64>();
            v_norm = v_norm.max(row);
        }
    }
    let electrons = (na + nb) as f64;
    // A first-quantized operator norm bounds the entire spectrum without FCI
    // enumeration. E_est is a variational upper bound to the ground eigenvalue.
    let upper = constant + electrons * h_norm + 0.5 * electrons * (electrons - 1.0) * v_norm;
    let radius = 0.5 * (upper - estimate).max(1e-12);
    let center = estimate + radius;
    let propagator = Propagator {
        h,
        g,
        constant,
        na,
        nb,
        limit: options.sci.max_determinants,
    };
    let mut state = Wavefunction::from([(seed, 1.0)]);
    let mut previous_energy = estimate;
    for step in 0..settings.max_steps {
        let eta = settings.spawning_threshold;
        let m = settings.polynomial_order;
        let mut next = match settings.generator {
            PCIGenerator::WallChebyshev => {
                let mut next = state.clone();
                for k in 1..=m {
                    let shift = estimate
                        + radius
                            * (1.0 - (std::f64::consts::PI * k as f64 / (m as f64 + 0.5)).cos());
                    next = propagator.action(&next, shift, eta)?;
                    normalize(&mut next)?;
                }
                next
            }
            PCIGenerator::Linear | PCIGenerator::Power => propagator.action(&state, upper, eta)?,
            PCIGenerator::Chebyshev | PCIGenerator::ExpChebyshev => {
                let weights = if settings.generator == PCIGenerator::Chebyshev {
                    let mut w = vec![0.0; m + 1];
                    w[m] = 1.0;
                    w
                } else {
                    (0..=m)
                        .map(|k| {
                            if k == 0 {
                                bessel_i(k, 0.5)
                            } else {
                                2.0 * bessel_i(k, 0.5)
                            }
                        })
                        .collect()
                };
                propagator.recurrence(&state, center, radius, eta, &weights)?
            }
            PCIGenerator::Quadratic | PCIGenerator::Cubic | PCIGenerator::Quartic => {
                let order = match settings.generator {
                    PCIGenerator::Quadratic => 2,
                    PCIGenerator::Cubic => 3,
                    _ => 4,
                };
                let mut result = state.clone();
                let mut term = state.clone();
                for k in 1..=order {
                    term = propagator.homogeneous_action(&term, estimate, eta)?;
                    scale(&mut term, -0.5 / (radius * k as f64));
                    add(&mut result, &term, 1.0);
                }
                result
            }
            PCIGenerator::Davidson | PCIGenerator::Lanczos => {
                propagator.krylov_step(&state, eta, settings.generator, m, options)?
            }
        };
        normalize(&mut next)?;
        // Remove the arbitrary global phase before testing the propagated vector.
        if dot(&state, &next) < 0.0 {
            scale(&mut next, -1.0);
        }
        let mut difference = next.clone();
        add(&mut difference, &state, -1.0);
        let change = dot(&difference, &difference).sqrt();
        let sigma = propagator.action(&next, 0.0, 0.0)?;
        let energy = dot(&next, &sigma);
        let mut residual = sigma.clone();
        add(&mut residual, &next, -energy);
        let residual_norm = dot(&residual, &residual).sqrt();
        let converged = (energy - previous_energy).abs() < options.energy_tol
            && change < options.residual_tol
            && (eta > 0.0 || residual_norm < options.residual_tol);
        if converged {
            let (&anchor, &coefficient) = next
                .iter()
                .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
                .unwrap();
            let projective_energy = sigma[&anchor] / coefficient;
            let dets = next.keys().copied().collect::<Vec<_>>();
            let engine = SubspaceHamiltonian::new(h, g, constant, dets.clone())?;
            let vector = next.values().copied().collect::<Vec<_>>();
            let s_squared = engine.spin_squared_expectation(&vector)?;
            let states = vec![FciState {
                energy,
                vector,
                residual_norm,
                s_squared,
            }];
            let corrections = if settings.compute_pt2 {
                en_correction(h, g, constant, na, nb, &dets, &states, None)?
            } else {
                vec![0.0]
            };
            return Ok(ProjectorModel {
                model: SelectedModel {
                    determinants: dets,
                    states,
                    corrections,
                    iterations: step + 1,
                },
                projective_energy,
            });
        }
        state = next;
        previous_energy = energy;
    }
    Err(HfError::PostHf {
        method: "PCI".into(),
        detail: format!(
            "projection did not converge in {} steps",
            settings.max_steps
        ),
    })
}
