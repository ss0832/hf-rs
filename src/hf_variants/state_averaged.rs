// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! State-averaged Hartree-Fock (SA-HF): one set of restricted spatial orbitals for an
//! ensemble of single determinants.
//!
//! Each state `I` is a determinant given by its occupied alpha and beta spatial orbitals, with
//! the exact determinant energy `E_I = sum_s Tr(P^I_s h) + 1/2 sum_s Tr(P^I_s G^I_s)`,
//! `G^I_s = J[P^I_a + P^I_b] - K[P^I_s]`. The orbitals minimize `E_SA = sum_I w_I E_I`; the
//! stationary condition is the vanishing ensemble orbital gradient
//!
//! `g_pq = dE_SA / d kappa_pq = 2 sum_I w_I sum_s (n^{Is}_q - n^{Is}_p) F^{Is}_pq`
//!
//! for the rotation `phi_q -> phi_q + kappa phi_p`, `phi_p -> phi_p - kappa phi_q`, with
//! `F^{Is} = h + G^I_s` in the MO basis. Rotations between orbitals with identical
//! occupations in every state and spin are redundant. The gradient is minimized with
//! preconditioned L-BFGS ([`super::orbital_optimizer`]) and a line search on the exact energy, so open-shell
//! states are never evaluated with a closed-shell (spin-summed) energy expression.
//! Orbital indices in a configuration refer to the guess orbitals (canonical orbitals of the
//! Fock operator of the superposition-of-atoms density, ordered by energy) and follow them
//! continuously during the optimization.

use super::orbital_optimizer::{Controls, Evaluation, max_abs, minimize};
use crate::basis::{Atom, ContractedGaussian};
use crate::matrix::Matrix;
use crate::scf::{
    build_fock_matrix, build_uhf_fock_matrices, compute_eri_tensor, compute_nuclear_repulsion,
    compute_one_electron_integrals,
};

/// A determinant of the ensemble: occupied alpha and beta spatial orbitals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateConfiguration {
    pub alpha: Vec<usize>,
    pub beta: Vec<usize>,
}

impl StateConfiguration {
    pub fn new(alpha: Vec<usize>, beta: Vec<usize>) -> Self {
        Self { alpha, beta }
    }

    /// Spatial occupations: 2 is doubly occupied, 1 an alpha electron (high-spin coupling of
    /// the open shells), 0 empty. Other values are rejected.
    pub fn from_occupations(occupations: &[f64]) -> Result<Self, String> {
        let (mut alpha, mut beta) = (Vec::new(), Vec::new());
        for (p, &n) in occupations.iter().enumerate() {
            if n == 2.0 {
                alpha.push(p);
                beta.push(p);
            } else if n == 1.0 {
                alpha.push(p);
            } else if n != 0.0 {
                return Err(format!(
                    "SA-HF occupation {n} of orbital {p} is not 0, 1 or 2"
                ));
            }
        }
        Ok(Self { alpha, beta })
    }
}

/// Convergence controls for [`run_state_averaged_hf_with_options`].
#[derive(Clone, Debug)]
pub struct SaHfOptions {
    pub max_iterations: usize,
    /// Largest absolute orbital-gradient component.
    pub gradient_tol: f64,
    /// Energy change of the last accepted step.
    pub energy_tol: f64,
}

impl Default for SaHfOptions {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            gradient_tol: 1e-7,
            energy_tol: 1e-11,
        }
    }
}

/// Result of a state-averaged Hartree-Fock calculation.
#[derive(Clone, Debug)]
pub struct StateAveragedHFResult {
    /// `sum_I w_I E_I` in Hartree.
    pub sa_energy: f64,
    /// Exact determinant energies `E_I` in Hartree (including nuclear repulsion).
    pub state_energies: Vec<f64>,
    pub weights: Vec<f64>,
    /// Ensemble-averaged spin-summed density `sum_I w_I (P^I_a + P^I_b)`.
    pub density_matrix: Matrix,
    pub mo_coefficients: Matrix,
    /// Diagonal of the ensemble- and spin-averaged Fock matrix in the MO basis.
    pub orbital_energies: Vec<f64>,
    /// Largest absolute ensemble orbital-gradient component.
    pub gradient_norm: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// SA-HF with an energy-change criterion `tol`; the gradient criterion is `sqrt(tol)/10`.
pub fn run_state_averaged_hf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    configurations: &[StateConfiguration],
    weights: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<StateAveragedHFResult, String> {
    run_state_averaged_hf_with_options(
        atoms,
        basis,
        configurations,
        weights,
        &SaHfOptions {
            max_iterations: max_iter,
            gradient_tol: (tol.sqrt() * 0.1).clamp(1e-10, 1e-5),
            energy_tol: tol,
        },
    )
}

struct Ensemble<'a> {
    h: &'a Matrix,
    eri: &'a [f64],
    nuclear: f64,
    states: &'a [StateConfiguration],
    weights: &'a [f64],
    /// Occupations `n[I][s][p]`.
    occupations: Vec<[Vec<f64>; 2]>,
}

struct StateEvaluation {
    energies: Vec<f64>,
    /// Per state and spin: AO Fock matrices.
    focks: Vec<[Matrix; 2]>,
    densities: Vec<[Matrix; 2]>,
}

impl Ensemble<'_> {
    fn density(c: &Matrix, occupied: &[usize]) -> Matrix {
        let n = c.rows;
        let mut p = Matrix::zeros(n, n);
        for &k in occupied {
            for mu in 0..n {
                for nu in 0..n {
                    p[(mu, nu)] += c[(mu, k)] * c[(nu, k)];
                }
            }
        }
        p
    }

    fn evaluate(&self, c: &Matrix) -> StateEvaluation {
        let n = c.rows;
        let dot = |a: &Matrix, b: &Matrix| -> f64 {
            a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum()
        };
        let mut out = StateEvaluation {
            energies: Vec::new(),
            focks: Vec::new(),
            densities: Vec::new(),
        };
        for state in self.states {
            let pa = Self::density(c, &state.alpha);
            let pb = Self::density(c, &state.beta);
            let (fa, fb) = build_uhf_fock_matrices(self.h, &pa, &pb, self.eri, n);
            let e =
                0.5 * (dot(&pa.add(&pb), self.h) + dot(&pa, &fa) + dot(&pb, &fb)) + self.nuclear;
            out.energies.push(e);
            out.focks.push([fa, fb]);
            out.densities.push([pa, pb]);
        }
        out
    }

    fn average(&self, energies: &[f64]) -> f64 {
        energies.iter().zip(self.weights).map(|(e, w)| e * w).sum()
    }

    /// Ensemble gradient and diagonal Hessian estimate over the nonredundant pairs `p > q`.
    fn gradient(
        &self,
        c: &Matrix,
        ev: &StateEvaluation,
        pairs: &[(usize, usize)],
    ) -> (Vec<f64>, Vec<f64>) {
        let mo: Vec<[Matrix; 2]> = ev
            .focks
            .iter()
            .map(|[fa, fb]| [c.transpose().dot(fa).dot(c), c.transpose().dot(fb).dot(c)])
            .collect();
        let mut g = vec![0.0; pairs.len()];
        let mut d = vec![0.0; pairs.len()];
        for (k, &(p, q)) in pairs.iter().enumerate() {
            for (i, w) in self.weights.iter().enumerate() {
                for s in 0..2 {
                    let n = &self.occupations[i][s];
                    let f = &mo[i][s];
                    g[k] += 2.0 * w * (n[q] - n[p]) * f[(p, q)];
                    d[k] += 2.0 * w * (n[q] - n[p]) * (f[(p, p)] - f[(q, q)]);
                }
            }
        }
        (g, d)
    }
}

fn rotate(c: &Matrix, pairs: &[(usize, usize)], step: &[f64]) -> Matrix {
    let m = c.cols;
    let mut x = Matrix::zeros(m, m);
    for (&(p, q), &k) in pairs.iter().zip(step) {
        // phi_q -> phi_q + k phi_p, phi_p -> phi_p - k phi_q
        x[(p, q)] = k;
        x[(q, p)] = -k;
    }
    c.dot(&crate::post_hf::mr_dsrg::cumulants::exp_antisymmetric(&x))
}

/// State-averaged HF over explicit determinants with exact determinant energies.
pub fn run_state_averaged_hf_with_options(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    configurations: &[StateConfiguration],
    weights: &[f64],
    options: &SaHfOptions,
) -> Result<StateAveragedHFResult, String> {
    let n = basis.len();
    if n == 0 || configurations.is_empty() || configurations.len() != weights.len() {
        return Err("SA-HF requires a basis and one weight per configuration".into());
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0)
        || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-12
    {
        return Err("SA-HF weights must be nonnegative and sum to one".into());
    }
    if options.max_iterations == 0
        || [options.gradient_tol, options.energy_tol]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err("invalid SA-HF convergence controls".into());
    }
    let (s, _, _, h) = compute_one_electron_integrals(atoms, basis);
    let x = s
        .canonical_orthogonalizer(1e-8)
        .map_err(|e| e.to_string())?;
    let n_mo = x.cols;
    let nuclear_charge: f64 = atoms.iter().map(|a| a.charge).sum();
    let mut electrons = None;
    for state in configurations {
        for occupied in [&state.alpha, &state.beta] {
            let mut sorted = occupied.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != occupied.len() || occupied.iter().any(|&k| k >= n_mo) {
                return Err(
                    "SA-HF occupied orbitals must be distinct and within the basis rank".into(),
                );
            }
        }
        let count = state.alpha.len() + state.beta.len();
        if count == 0 || electrons.is_some_and(|e| e != count) {
            return Err("all SA-HF determinants must have the same positive electron count".into());
        }
        electrons = Some(count);
    }
    let charge = nuclear_charge.round() as i32 - electrons.unwrap() as i32;
    let eri = compute_eri_tensor(basis);
    // Guess orbitals: canonical orbitals of the superposition-of-atoms Fock operator.
    let guess = match crate::sad::compute_sad_density(atoms, basis, charge, &s) {
        Ok(p) => build_fock_matrix(&h, &p, &eri, n),
        Err(_) => h.clone(),
    };
    let c = x.dot(&x.transpose().dot(&guess).dot(&x).eigh().1);

    let occupations: Vec<[Vec<f64>; 2]> = configurations
        .iter()
        .map(|state| {
            let mut na = vec![0.0; n_mo];
            let mut nb = vec![0.0; n_mo];
            state.alpha.iter().for_each(|&k| na[k] = 1.0);
            state.beta.iter().for_each(|&k| nb[k] = 1.0);
            [na, nb]
        })
        .collect();
    let pairs: Vec<(usize, usize)> = (0..n_mo)
        .flat_map(|p| (0..p).map(move |q| (p, q)))
        .filter(|&(p, q)| {
            occupations
                .iter()
                .zip(weights)
                .any(|(o, &w)| w > 0.0 && (o[0][p] != o[0][q] || o[1][p] != o[1][q]))
        })
        .collect();
    let ensemble = Ensemble {
        h: &h,
        eri: &eri,
        nuclear: compute_nuclear_repulsion(atoms),
        states: configurations,
        weights,
        occupations,
    };

    let evaluate = |c: Matrix| {
        let ev = ensemble.evaluate(&c);
        let (gradient, diagonal) = ensemble.gradient(&c, &ev, &pairs);
        Evaluation {
            value: ensemble.average(&ev.energies),
            state: (c, ev),
            gradient,
            diagonal,
        }
    };
    let minimum = minimize(
        evaluate(c),
        &Controls {
            max_iterations: options.max_iterations,
            gradient_tol: options.gradient_tol,
            value_tol: options.energy_tol,
            max_step: 0.5,
        },
        |(c, _), kappa| evaluate(rotate(c, &pairs, kappa)),
    );
    let gradient_norm = max_abs(&minimum.point.gradient);
    if !minimum.converged {
        return Err(format!(
            "SA-HF did not converge in {} iterations: gradient {gradient_norm:e}, energy change {:e}",
            options.max_iterations, minimum.value_change
        ));
    }
    let energy = minimum.point.value;
    let (c, ev) = minimum.point.state;
    let mut density = Matrix::zeros(n, n);
    let mut average_fock = Matrix::zeros(n, n);
    for (i, w) in weights.iter().enumerate() {
        for s in 0..2 {
            density = density.add(&ev.densities[i][s].scale(*w));
            average_fock = average_fock.add(&ev.focks[i][s].scale(0.5 * w));
        }
    }
    let mo_fock = c.transpose().dot(&average_fock).dot(&c);
    Ok(StateAveragedHFResult {
        sa_energy: energy,
        state_energies: ev.energies,
        weights: weights.to_vec(),
        density_matrix: density,
        orbital_energies: (0..n_mo).map(|p| mo_fock[(p, p)]).collect(),
        mo_coefficients: c,
        gradient_norm,
        iterations: minimum.iterations,
        converged: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_ensemble_orbital_gradient_component_matches_energy_differences() {
        let atoms = vec![
            Atom::new("H", [0.1, -0.2, -0.7]),
            Atom::new("H", [-0.1, 0.3, 0.7]),
        ];
        let basis = crate::build_basis(&atoms, &crate::get_builtin_basis("6-31g").unwrap());
        let (s, _, _, h) = compute_one_electron_integrals(&atoms, &basis);
        let eri = compute_eri_tensor(&basis);
        let x = s.canonical_orthogonalizer(1e-8).unwrap();
        let n = x.cols;
        let states = [
            StateConfiguration::new(vec![0, 1], vec![0]),
            StateConfiguration::new(vec![0, 2], vec![0]),
            StateConfiguration::new(vec![0], vec![0, 3]),
        ];
        let weights = [0.4, 0.35, 0.25];
        let occupations = states
            .iter()
            .map(|state| {
                let mut occ = [vec![0.; n], vec![0.; n]];
                for &p in &state.alpha {
                    occ[0][p] = 1.;
                }
                for &p in &state.beta {
                    occ[1][p] = 1.;
                }
                occ
            })
            .collect();
        let ensemble = Ensemble {
            h: &h,
            eri: &eri,
            nuclear: compute_nuclear_repulsion(&atoms),
            states: &states,
            weights: &weights,
            occupations,
        };
        let pairs: Vec<_> = (0..n).flat_map(|p| (0..p).map(move |q| (p, q))).collect();
        let mut worst = 0f64;
        for seed in 1..=3 {
            let initial: Vec<_> = (0..pairs.len())
                .map(|k| 0.17 * ((k + 7 * seed) as f64).sin())
                .collect();
            let c = rotate(&x, &pairs, &initial);
            let evaluated = ensemble.evaluate(&c);
            let (gradient, _) = ensemble.gradient(&c, &evaluated, &pairs);
            for k in 0..pairs.len() {
                let step = 1e-4;
                let value = |multiple| {
                    let mut delta = vec![0.; pairs.len()];
                    delta[k] = multiple * step;
                    ensemble.average(&ensemble.evaluate(&rotate(&c, &pairs, &delta)).energies)
                };
                let reference =
                    (-value(2.) + 8. * value(1.) - 8. * value(-1.) + value(-2.)) / (12. * step);
                let error = (gradient[k] - reference).abs();
                worst = worst.max(error);
                assert!(
                    error < 3e-8,
                    "seed {seed}, ensemble orbital gradient {k}: {error:e}"
                );
            }
        }
        eprintln!(
            "SA-HF: all {} arbitrary-frame orbital gradient components, max error {worst:e}",
            3 * pairs.len()
        );
    }
}
