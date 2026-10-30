// SPDX-License-Identifier: Apache-2.0
//! ACI (Schriber–Evangelista 2016), ASCI (Tubman et al. 2016), and HCI
//! (Holmes–Tubman–Umrigar 2016), sharing a matrix-free variational solver.
use crate::context::{
    ActiveIntegrals, MoIntegralCache, PostHFContext, Reference, active_integrals_from_mo,
};
use crate::error::HfError;
use crate::fci::engine::{FciSolverOptions, FciState};
use crate::fci::excitations::{Determinant, singles_doubles, spin_complete};
use crate::fci::slater_condon::slater_condon_element;
use crate::fci::string::SpinString;
use crate::fci::subspace::SubspaceHamiltonian;
use crate::matrix::Matrix;
use crate::mo_integrals::{MoEriTensor, transform_1e_ao_to_mo, transform_eri_ao_to_mo};
use crate::options::{ActiveSpace, PostHFOptions};
use crate::output::{PostHFOutput, State};
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::{ContextMethod, MethodId, MethodMeta};
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionKind {
    Aci,
    Asci,
    HeatBath,
}
impl SelectionKind {
    fn name(self) -> &'static str {
        match self {
            Self::Aci => "ACI",
            Self::Asci => "ASCI",
            Self::HeatBath => "HBCI",
        }
    }
}
#[derive(Clone, Debug)]
pub struct SelectedCiResult {
    pub output: PostHFOutput,
    pub determinants: Vec<Determinant>,
    pub states: Vec<FciState>,
    pub corrections: Vec<f64>,
    pub iterations: usize,
    pub active_integrals: Arc<ActiveIntegrals>,
    pub kind: SelectionKind,
}
impl SelectedCiResult {
    pub fn active_two_rdm(&self, root: usize) -> Result<Vec<f64>, HfError> {
        let state = self
            .states
            .get(root)
            .ok_or_else(|| HfError::InvalidInput("SCI root is unavailable".into()))?;
        let engine = SubspaceHamiltonian::new(
            &self.active_integrals.h,
            &self.active_integrals.eri,
            self.active_integrals.core_energy,
            self.determinants.clone(),
        )?;
        engine.rdm(&state.vector, 2)
    }
}
impl PostHFResult for SelectedCiResult {
    fn method_name(&self) -> &'static str {
        self.kind.name()
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

/// Lower eigenvalue of the 2x2 P/I Hamiltonian minus E_P (ACI Eq. 5).
/// The rational form avoids cancellation for a positive large diagonal gap.
pub fn aci_importance(delta: f64, coupling: f64) -> Result<f64, HfError> {
    if !delta.is_finite() || !coupling.is_finite() {
        return Err(HfError::Integral("non-finite ACI importance input".into()));
    }
    let radius = (0.5 * delta).hypot(coupling);
    Ok(if delta >= 0.0 {
        if radius == 0.0 {
            0.0
        } else {
            -coupling * (coupling / (0.5 * delta + radius))
        }
    } else {
        0.5 * delta - radius
    })
}
fn low_mask(n: usize) -> u64 {
    if n == 64 { u64::MAX } else { (1u64 << n) - 1 }
}
fn initial_determinant(
    n: usize,
    na: usize,
    nb: usize,
    target: Option<f64>,
) -> Result<Determinant, HfError> {
    let Some(target) = target else {
        return Ok((SpinString(low_mask(na)), SpinString(low_mask(nb))));
    };
    let spin = (target + 0.25).sqrt() - 0.5;
    let sz = (na as f64 - nb as f64).abs() * 0.5;
    let ne = na + nb;
    if !target.is_finite()
        || target < 0.0
        || spin < sz - 1e-9
        || spin > ne.min(2 * n - ne) as f64 * 0.5 + 1e-9
        || ((spin - sz) - (spin - sz).round()).abs() > 1e-9
    {
        return Err(HfError::InvalidInput(
            "SCI target S(S+1) is incompatible with the active electrons".into(),
        ));
    }
    let doubly = ((ne as f64 * 0.5) - spin).round() as usize;
    let a = low_mask(na);
    let beta_singles = nb - doubly;
    let beta = low_mask(doubly) | (low_mask(na + beta_singles) ^ low_mask(na));
    Ok((SpinString(a), SpinString(beta)))
}
fn complete(
    dets: Vec<Determinant>,
    n: usize,
    na: usize,
    nb: usize,
    options: &PostHFOptions,
) -> Result<Vec<Determinant>, HfError> {
    let mut dets = if options.cas.target_s_squared.is_some() {
        spin_complete(n, na, nb, &dets, options.sci.max_determinants)?
    } else {
        dets
    };
    dets.sort_unstable();
    dets.dedup();
    if dets.len() > options.sci.max_determinants {
        return Err(HfError::Unsupported {
            method: "SCI".into(),
            detail: "selected model exceeds max_determinants".into(),
        });
    }
    Ok(dets)
}
fn solve_model(
    h: &Matrix,
    g: &MoEriTensor,
    constant: f64,
    dets: Vec<Determinant>,
    options: &PostHFOptions,
) -> Result<(Vec<Determinant>, Vec<FciState>), HfError> {
    let engine = SubspaceHamiltonian::new(h, g, constant, dets)?;
    let available = if let Some(s) = options.cas.target_s_squared {
        engine.target_spin_dimension(s)?
    } else {
        engine.dimension()
    };
    let solution = engine.solve(&FciSolverOptions {
        n_roots: options.n_roots.min(available),
        max_iterations: options.sci.diagonalization_max_iterations,
        residual_tol: options.residual_tol,
        target_s_squared: options.cas.target_s_squared,
        ..Default::default()
    })?;
    Ok((engine.determinants, solution.states))
}
struct Candidate {
    det: Determinant,
    diagonal: f64,
    couplings: Vec<f64>,
    max_product: f64,
}
fn candidates(
    h: &Matrix,
    g: &MoEriTensor,
    constant: f64,
    na: usize,
    nb: usize,
    model: &[Determinant],
    states: &[FciState],
    sources: &[usize],
    excluded: &HashSet<Determinant>,
    screen: Option<f64>,
) -> Result<Vec<Candidate>, HfError> {
    let refs = sources.iter().map(|&i| model[i]).collect::<Vec<_>>();
    let mut external = singles_doubles(h.rows, na, nb, &refs, 2_000_000)?;
    external.retain(|d| !excluded.contains(d));
    if external
        .len()
        .checked_mul(64 + 24 * states.len())
        .is_none_or(|n| n > 256 * 1024 * 1024)
    {
        return Err(HfError::Unsupported {
            method: "SCI external space".into(),
            detail: "selection workspace exceeds 256 MiB".into(),
        });
    }
    let mut out = Vec::with_capacity(external.len());
    for det in external {
        let diagonal = constant + slater_condon_element(&det.0, &det.1, &det.0, &det.1, h, g);
        let mut couplings = vec![0.0; states.len()];
        let mut max_product = 0.0_f64;
        for &j in sources {
            let d = model[j];
            let element = slater_condon_element(&det.0, &det.1, &d.0, &d.1, h, g);
            for (root, state) in states.iter().enumerate() {
                let product = element * state.vector[j];
                max_product = max_product.max(product.abs());
                if screen.is_none_or(|cutoff| product.abs() > cutoff) {
                    couplings[root] += product;
                }
            }
        }
        if !diagonal.is_finite() || couplings.iter().any(|x| !x.is_finite()) {
            return Err(HfError::Integral("non-finite SCI external coupling".into()));
        }
        out.push(Candidate {
            det,
            diagonal,
            couplings,
            max_product,
        });
    }
    Ok(out)
}
fn coefficient_weights(states: &[FciState], weights: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            states
                .iter()
                .enumerate()
                .map(|(r, s)| weights[r] * s.vector[i].powi(2))
                .sum()
        })
        .collect()
}
fn state_weights(options: &PostHFOptions, n: usize) -> Vec<f64> {
    if options.cas.state_weights.is_empty() {
        vec![1.0 / n as f64; n]
    } else {
        let total = options.cas.state_weights[..n].iter().sum::<f64>();
        if total == 0.0 {
            vec![1.0 / n as f64; n]
        } else {
            options.cas.state_weights[..n]
                .iter()
                .map(|x| x / total)
                .collect()
        }
    }
}
fn coarse_grain(
    model: &[Determinant],
    states: &[FciState],
    options: &PostHFOptions,
) -> Vec<Determinant> {
    let weights = coefficient_weights(states, &state_weights(options, states.len()), model.len());
    let mut rank = (0..model.len()).collect::<Vec<_>>();
    rank.sort_by(|&a, &b| {
        weights[a]
            .total_cmp(&weights[b])
            .then_with(|| model[a].cmp(&model[b]))
    });
    let mut removed = 0.0;
    let mut keep = vec![true; model.len()];
    let mut remaining = model.len();
    for i in rank {
        if remaining <= states.len() {
            break;
        }
        if removed + weights[i] <= options.sci.coarse_graining * options.sci.selection_threshold {
            keep[i] = false;
            remaining -= 1;
            removed += weights[i];
        } else {
            break;
        }
    }
    model
        .iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, d)| *d)
        .collect()
}
pub(crate) fn en_correction(
    h: &Matrix,
    g: &MoEriTensor,
    constant: f64,
    na: usize,
    nb: usize,
    model: &[Determinant],
    states: &[FciState],
    screen: Option<f64>,
) -> Result<Vec<f64>, HfError> {
    let source = (0..model.len()).collect::<Vec<_>>();
    let excluded = model.iter().copied().collect();
    let external = candidates(
        h, g, constant, na, nb, model, states, &source, &excluded, screen,
    )?;
    let mut correction = vec![0.0; states.len()];
    for d in external {
        for (r, s) in states.iter().enumerate() {
            let v = d.couplings[r];
            if v == 0.0 {
                continue;
            }
            let denominator = s.energy - d.diagonal;
            if denominator.abs() < 1e-14 {
                return Err(HfError::Response(
                    "singular SCI Epstein-Nesbet denominator; enlarge the variational space".into(),
                ));
            }
            correction[r] += v * v / denominator;
        }
    }
    Ok(correction)
}

#[derive(Clone, Debug)]
pub struct SelectedModel {
    pub determinants: Vec<Determinant>,
    pub states: Vec<FciState>,
    pub corrections: Vec<f64>,
    pub iterations: usize,
}
pub fn solve_selected(
    h: &Matrix,
    g: &MoEriTensor,
    constant: f64,
    na: usize,
    nb: usize,
    kind: SelectionKind,
    options: &PostHFOptions,
) -> Result<SelectedModel, HfError> {
    options.validate()?;
    if h.rows > 64 || na > h.rows || nb > h.rows || options.sci.max_determinants > 2_000_000 {
        return Err(HfError::InvalidInput(
            "SCI active space/storage is unsupported".into(),
        ));
    }
    if kind == SelectionKind::Aci
        && options.sci.coarse_graining * options.sci.selection_threshold >= 1.0
    {
        return Err(HfError::InvalidInput(
            "ACI coarse_graining * selection_threshold must be below one".into(),
        ));
    }
    if kind == SelectionKind::Asci
        && (options.sci.n_core_determinants == 0
            || options.sci.n_target_determinants == 0
            || options.sci.n_core_determinants > options.sci.n_target_determinants
            || options.sci.n_target_determinants > options.sci.max_determinants)
    {
        return Err(HfError::InvalidInput(
            "ASCI requires 0 < n_core_determinants <= n_target_determinants <= max_determinants"
                .into(),
        ));
    }
    let seed = initial_determinant(h.rows, na, nb, options.cas.target_s_squared)?;
    // A single HF determinant can have zero overlap with an excited root in a
    // different spatial symmetry sector. A multiroot search starts from the
    // connected SD space, including zero-coupling determinants, so those sectors
    // are represented before coefficient-based selection begins.
    let initial = if options.n_roots > 1 {
        singles_doubles(h.rows, na, nb, &[seed], options.sci.max_determinants)?
    } else {
        vec![seed]
    };
    let mut p = complete(initial, h.rows, na, nb, options)?;
    let mut previous_model = Vec::new();
    let mut previous_energy = None;
    for iteration in 0..options.max_iter {
        let (model, states) = solve_model(h, g, constant, p.clone(), options)?;
        let excluded = model.iter().copied().collect::<HashSet<_>>();
        let source = (0..model.len()).collect::<Vec<_>>();
        let mut corrections = vec![0.0; states.len()];
        let next = match kind {
            SelectionKind::Aci => {
                let external = candidates(
                    h, g, constant, na, nb, &model, &states, &source, &excluded, None,
                )?;
                let weights = state_weights(options, states.len());
                let mut rank = Vec::new();
                for d in external {
                    let importance = states
                        .iter()
                        .enumerate()
                        .map(|(r, s)| aci_importance(d.diagonal - s.energy, d.couplings[r]))
                        .collect::<Result<Vec<_>, _>>()?;
                    let score = importance
                        .iter()
                        .zip(&weights)
                        .map(|(a, w)| w * a.abs())
                        .sum::<f64>();
                    rank.push((d.det, score, importance));
                }
                rank.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                let mut discarded = 0.0;
                let mut next = model.clone();
                for (d, score, importance) in rank {
                    if discarded + score <= options.sci.selection_threshold {
                        discarded += score;
                        for (c, e) in corrections.iter_mut().zip(importance) {
                            *c += e;
                        }
                    } else {
                        next.push(d);
                    }
                }
                next
            }
            SelectionKind::HeatBath => {
                let external = candidates(
                    h, g, constant, na, nb, &model, &states, &source, &excluded, None,
                )?;
                let mut next = model.clone();
                for d in external {
                    if d.max_product > options.sci.selection_threshold {
                        next.push(d.det);
                    }
                }
                next
            }
            SelectionKind::Asci => {
                let mut rank = (0..model.len()).collect::<Vec<_>>();
                let coeff = |i: usize| states.iter().map(|s| s.vector[i].abs()).fold(0.0, f64::max);
                rank.sort_by(|&a, &b| {
                    coeff(b)
                        .total_cmp(&coeff(a))
                        .then_with(|| model[a].cmp(&model[b]))
                });
                rank.truncate(options.sci.n_core_determinants);
                let core = rank.iter().map(|&i| model[i]).collect::<HashSet<_>>();
                let external =
                    candidates(h, g, constant, na, nb, &model, &states, &rank, &core, None)?;
                let mut scored = rank
                    .iter()
                    .map(|&i| (model[i], coeff(i)))
                    .collect::<Vec<_>>();
                for d in external {
                    let score = states
                        .iter()
                        .enumerate()
                        .map(|(r, s)| {
                            let gap = (d.diagonal - s.energy).abs();
                            let v = d.couplings[r].abs();
                            if gap == 0.0 {
                                if v == 0.0 { 0.0 } else { f64::INFINITY }
                            } else {
                                v / gap
                            }
                        })
                        .fold(0.0, f64::max);
                    scored.push((d.det, score));
                }
                scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                scored.truncate(options.sci.n_target_determinants);
                scored.into_iter().map(|(d, _)| d).collect()
            }
        };
        let next = complete(next, h.rows, na, nb, options)?;
        let (next, solved) = solve_model(h, g, constant, next, options)?;
        let energy = solved.iter().map(|s| s.energy).collect::<Vec<_>>();
        let stable_energy = previous_energy.as_ref().is_some_and(|old: &Vec<f64>| {
            old.len() == energy.len()
                && old
                    .iter()
                    .zip(&energy)
                    .all(|(a, b)| (a - b).abs() < options.energy_tol)
        });
        let fixed = match kind {
            SelectionKind::HeatBath => next == model,
            SelectionKind::Asci => next == model || stable_energy,
            SelectionKind::Aci => {
                next == previous_model && stable_energy
                    || next == model && coarse_grain(&next, &solved, options) == model
            }
        };
        if fixed {
            if solved.len() != options.n_roots {
                return Err(HfError::PostHf{method:kind.name().into(),detail:"selection space has too few requested roots; decrease threshold or increase target size".into()});
            }
            if !options.sci.compute_pt2 {
                corrections = vec![0.0; solved.len()];
            } else if kind != SelectionKind::Aci {
                corrections = en_correction(
                    h,
                    g,
                    constant,
                    na,
                    nb,
                    &next,
                    &solved,
                    if kind == SelectionKind::HeatBath {
                        Some(options.sci.pt2_threshold)
                    } else {
                        None
                    },
                )?;
            }
            if corrections.len() != solved.len() {
                return Err(HfError::Response(
                    "SCI correction/state dimensions changed at convergence".into(),
                ));
            }
            return Ok(SelectedModel {
                determinants: next,
                states: solved,
                corrections,
                iterations: iteration + 1,
            });
        }
        previous_model = next.clone();
        previous_energy = Some(energy);
        p = if kind == SelectionKind::Aci {
            complete(
                coarse_grain(&next, &solved, options),
                h.rows,
                na,
                nb,
                options,
            )?
        } else {
            next
        };
    }
    Err(HfError::PostHf {
        method: kind.name().into(),
        detail: format!(
            "selection did not converge in {} iterations",
            options.max_iter
        ),
    })
}

pub(crate) fn selected_integrals(
    reference: Reference<'_>,
    options: &PostHFOptions,
    cache: Option<&MoIntegralCache<'_>>,
    method: &str,
) -> Result<Arc<ActiveIntegrals>, HfError> {
    options.validate()?;
    if !reference.converged() {
        return Err(HfError::Scf("SCI reference did not converge".into()));
    }
    if options.density_fitting {
        return Err(HfError::Unsupported {
            method: method.into(),
            detail: "DF selected-CI integrals are not implemented".into(),
        });
    }
    let c = reference.coefficients(0);
    let (na, nb) = reference.occupations();
    if options.frozen_core > na.min(nb) || options.frozen_core + options.frozen_virtual > c.cols {
        return Err(HfError::InvalidInput(
            "invalid SCI frozen orbital counts".into(),
        ));
    }
    let default = ActiveSpace {
        n_alpha: na - options.frozen_core,
        n_beta: nb - options.frozen_core,
        n_orb: c.cols - options.frozen_core - options.frozen_virtual,
        orbitals: Some((options.frozen_core..c.cols - options.frozen_virtual).collect()),
    };
    let active = options.active.as_ref().unwrap_or(&default);
    let resolved = active.resolve(c.cols, na, nb)?;
    if (0..options.frozen_core).any(|i| !resolved.inactive.contains(&i))
        || (c.cols - options.frozen_virtual..c.cols).any(|i| !resolved.external.contains(&i))
    {
        return Err(HfError::InvalidInput(
            "SCI active orbitals overlap the requested frozen orbitals".into(),
        ));
    }
    if active.n_orb > 64 {
        return Err(HfError::Unsupported {
            method: method.into(),
            detail: "spin strings support at most 64 active orbitals".into(),
        });
    }
    let ints = if let Some(cache) = cache {
        cache.active(active)?.clone()
    } else {
        let h = transform_1e_ao_to_mo(c, reference.core());
        let g = transform_eri_ao_to_mo(c, reference.eri());
        Arc::new(active_integrals_from_mo(
            &h,
            |p, q, r, s| g.get(p, q, r, s),
            resolved,
            reference.nuclear_energy(),
        )?)
    };
    Ok(ints)
}

pub fn compute_selected(
    reference: Reference<'_>,
    options: &PostHFOptions,
    cache: Option<&MoIntegralCache<'_>>,
    kind: SelectionKind,
) -> Result<SelectedCiResult, HfError> {
    let ints = selected_integrals(reference, options, cache, kind.name())?;
    let c = reference.coefficients(0);
    let solved = solve_selected(
        &ints.h,
        &ints.eri,
        ints.core_energy,
        ints.space.n_alpha,
        ints.space.n_beta,
        kind,
        options,
    )?;
    let engine = SubspaceHamiltonian::new(
        &ints.h,
        &ints.eri,
        ints.core_energy,
        solved.determinants.clone(),
    )?;
    let d = engine.rdm(&solved.states[options.root].vector, 1)?;
    let mut one = Matrix::zeros(c.cols, c.cols);
    for &i in &ints.space.inactive {
        one[(i, i)] = 2.0;
    }
    for (p, &pp) in ints.space.active.iter().enumerate() {
        for (q, &qq) in ints.space.active.iter().enumerate() {
            one[(pp, qq)] = d[p * ints.space.active.len() + q];
        }
    }
    let mut output = PostHFOutput::new(
        kind.name(),
        reference.energy(),
        solved.states[options.root].energy + solved.corrections[options.root],
    )?;
    output.states = solved
        .states
        .iter()
        .enumerate()
        .map(|(r, s)| State {
            energy: s.energy + solved.corrections[r],
            s_squared: Some(s.s_squared),
            weight: f64::from(r == options.root),
            converged: true,
        })
        .collect();
    output.select_root(options.root)?;
    output.one_rdm = Some(one);
    output
        .components
        .insert("E_variational".into(), solved.states[options.root].energy);
    output
        .components
        .insert("Delta_E_PT2".into(), solved.corrections[options.root]);
    output.validate()?;
    Ok(SelectedCiResult {
        output,
        determinants: solved.determinants,
        states: solved.states,
        corrections: solved.corrections,
        iterations: solved.iterations,
        active_integrals: ints,
        kind,
    })
}
macro_rules! selected_adapter {
    ($name:ident,$method:ident,$kind:ident) => {
        pub struct $name<'a> {
            ctx: &'a PostHFContext<'a>,
            options: PostHFOptions,
        }
        impl MethodMeta for $name<'_> {
            const METHOD: MethodId = MethodId::$method;
        }
        impl<'a> ContextMethod<'a> for $name<'a> {
            fn from_context(
                ctx: &'a PostHFContext<'a>,
                options: &PostHFOptions,
            ) -> Result<Self, HfError> {
                options.validate()?;
                Ok(Self {
                    ctx,
                    options: options.clone(),
                })
            }
        }
        impl PostHFMethod for $name<'_> {
            type Output = SelectedCiResult;
            fn compute(&self) -> Result<SelectedCiResult, String> {
                compute_selected(
                    self.ctx.reference,
                    &self.options,
                    Some(&self.ctx.ints),
                    SelectionKind::$kind,
                )
                .map_err(|e| e.to_string())
            }
        }
    };
}
selected_adapter!(AdaptiveCi, Aci, Aci);
selected_adapter!(AdaptiveSamplingCi, Asci, Asci);
selected_adapter!(HeatBathCi, Hbci, HeatBath);
