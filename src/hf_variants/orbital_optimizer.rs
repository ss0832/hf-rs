// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Preconditioned L-BFGS over rebased orbital rotations, with a backtracking (Armijo) line
//! search on the exact objective. Each accepted step rotates the orbitals, after which the
//! rotation parameters restart at zero; the curvature pairs are kept across rebasing, as in
//! standard quasi-Newton orbital optimizers. Used by objectives whose orbital gradient is
//! available in closed form but whose Hessian is not (SA-HF, sigma-SCF).

/// Objective value, gradient and positive diagonal preconditioner at the current orbitals.
pub(super) struct Evaluation<S> {
    pub state: S,
    pub value: f64,
    pub gradient: Vec<f64>,
    pub diagonal: Vec<f64>,
}

pub(super) struct Controls {
    pub max_iterations: usize,
    pub gradient_tol: f64,
    pub value_tol: f64,
    /// Largest rotation component of a trial step (radians).
    pub max_step: f64,
}

pub(super) struct Minimum<S> {
    pub point: Evaluation<S>,
    pub iterations: usize,
    pub converged: bool,
    pub value_change: f64,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(super) fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0f64, |m, x| m.max(x.abs()))
}

/// `step(state, kappa)` returns the evaluation at the orbitals rotated by `kappa`.
pub(super) fn minimize<S>(
    start: Evaluation<S>,
    controls: &Controls,
    step: impl Fn(&S, &[f64]) -> Evaluation<S>,
) -> Minimum<S> {
    match try_minimize::<S, std::convert::Infallible>(start, controls, |state, delta| {
        Ok(step(state, delta))
    }) {
        Ok(result) => result,
        Err(never) => match never {},
    }
}

/// Fallible objective evaluation, for model domains with an explicitly required orbital gap.
pub(super) fn try_minimize<S, E>(
    start: Evaluation<S>,
    controls: &Controls,
    step: impl Fn(&S, &[f64]) -> Result<Evaluation<S>, E>,
) -> Result<Minimum<S>, E> {
    let mut point = start;
    let mut history: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut change = f64::INFINITY;
    let mut iterations = 0;
    let precondition = |v: &[f64], d: &[f64]| -> Vec<f64> {
        v.iter()
            .zip(d)
            .map(|(x, d)| x / d.abs().max(0.05))
            .collect()
    };
    let mut converged = max_abs(&point.gradient) <= controls.gradient_tol;
    while !converged && iterations < controls.max_iterations {
        iterations += 1;
        let g = &point.gradient;
        let mut q = g.clone();
        let mut coefficients = Vec::with_capacity(history.len());
        for (s, y) in history.iter().rev() {
            let rho = 1.0 / dot(y, s);
            let a = rho * dot(s, &q);
            q.iter_mut().zip(y).for_each(|(qi, yi)| *qi -= a * yi);
            coefficients.push((a, rho));
        }
        let mut r = precondition(&q, &point.diagonal);
        for ((s, y), (a, rho)) in history.iter().zip(coefficients.iter().rev()) {
            let b = rho * dot(y, &r);
            r.iter_mut().zip(s).for_each(|(ri, si)| *ri += (a - b) * si);
        }
        let mut direction: Vec<f64> = r.iter().map(|v| -v).collect();
        if dot(&direction, g) >= 0.0 {
            history.clear();
            direction = precondition(g, &point.diagonal)
                .iter()
                .map(|v| -v)
                .collect();
        }
        let largest = max_abs(&direction);
        if largest > controls.max_step {
            direction
                .iter_mut()
                .for_each(|v| *v *= controls.max_step / largest);
        }
        let slope = dot(&direction, g);
        let mut t = 1.0;
        let trial = loop {
            let kappa: Vec<f64> = direction.iter().map(|v| t * v).collect();
            let trial = step(&point.state, &kappa)?;
            if trial.value <= point.value + 1e-4 * t * slope || t < 1e-8 {
                break trial;
            }
            t *= 0.5;
        };
        let s: Vec<f64> = direction.iter().map(|v| t * v).collect();
        let y: Vec<f64> = trial.gradient.iter().zip(g).map(|(a, b)| a - b).collect();
        if dot(&s, &y) > 1e-14 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() {
            history.push((s, y));
            if history.len() > 12 {
                history.remove(0);
            }
        }
        change = (point.value - trial.value).abs();
        point = trial;
        converged =
            max_abs(&point.gradient) <= controls.gradient_tol && change <= controls.value_tol;
    }
    Ok(Minimum {
        point,
        iterations,
        converged,
        value_change: change,
    })
}
