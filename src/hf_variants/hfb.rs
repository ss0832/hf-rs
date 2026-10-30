// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Real spin-singlet HFB for the electronic Hamiltonian plus `-G B^+ B`, where
//! `B^+ = sum_p a^+_{p alpha} a^+_{p beta}` in a real orthonormal spatial basis.
//! The reduced pairing interaction is an explicit additional model interaction, not an
//! approximation to the electronic Coulomb integrals. Its normal and anomalous contractions
//! are both retained. Only the expectation of particle number is fixed; the state is an
//! even-parity quasiparticle vacuum, without number projection or odd-particle blocking.
//!
//! For one-spin normal density `d` and symmetric pair tensor `k`,
//! `E = 2 Tr(h d) + 2 Tr(J[d] d) - Tr(K[d] d) + Tr(K[k] k)
//!      - G (Tr(d^2) + Tr(k)^2) + E_nuc`.
//! Thus `f = h + 2J[d] - K[d] - G d`, `D = K[k] - G Tr(k) I`.
//! The negative spectral projector of `[[f-mu,D],[D,-f+mu]]` is
//! `R = [[d,k],[k,I-d]]`. The same fields define the energy and stationarity residual.

use crate::basis::{Atom, ContractedGaussian};
use crate::diis::{CdiisEngine, ConvAlgo, max_abs_element};
use crate::matrix::Matrix;
use crate::scf::{HFResult, contract_jk, run_scf_general};

#[derive(Clone, Debug)]
pub struct HfbOptions {
    pub pairing_strength: f64,
    pub max_iterations: usize,
    pub energy_tol: f64,
    pub residual_tol: f64,
    pub density_tol: f64,
    pub number_tol: f64,
}

impl Default for HfbOptions {
    fn default() -> Self {
        Self {
            pairing_strength: 0.05,
            max_iterations: 200,
            energy_tol: 1e-11,
            residual_tol: 1e-9,
            density_tol: 1e-9,
            number_tol: 1e-12,
        }
    }
}

/// Densities are in the real orthonormal frame given by `orbital_coefficients`.
#[derive(Clone, Debug)]
pub struct HFBResult {
    /// Physical RHF reference energy, without the additional pairing interaction.
    pub normal_energy: f64,
    /// Expectation of the complete electronic plus reduced-pairing Hamiltonian.
    pub total_hfb_energy: f64,
    /// Expectation of `-G B^+ B`, including its normal-density contraction.
    pub pairing_energy: f64,
    /// Physical electronic HF functional at the final normal density, including nuclear repulsion.
    pub normal_component_energy: f64,
    /// Anomalous contraction of the physical Coulomb interaction, `Tr(K[k] k)`.
    pub coulomb_pairing_energy: f64,
    /// Largest absolute eigenvalue of the anomalous field, in Hartree.
    pub pairing_gap: f64,
    pub chemical_potential: f64,
    pub quasiparticle_energies: Vec<f64>,
    /// Spin-summed normal density `2d`; its trace is the expected electron count.
    pub normal_density: Matrix,
    /// Symmetric `k_pq = <a_{q beta} a_{p alpha}>`; the full spin-orbital tensor is antisymmetric.
    pub pairing_tensor: Matrix,
    pub orbital_coefficients: Matrix,
    pub particle_number_variance: f64,
    pub number_error: f64,
    pub residual_norm: f64,
    pub idempotence_error: f64,
    pub iterations: usize,
    pub converged: bool,
}

struct Fields {
    f: Matrix,
    gap: Matrix,
    normal_energy: f64,
    anomalous_energy: f64,
    pairing_energy: f64,
}

impl Fields {
    fn energy(&self) -> f64 {
        self.normal_energy + self.anomalous_energy + self.pairing_energy
    }
}

fn inner(a: &Matrix, b: &Matrix) -> f64 {
    a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum()
}

fn fields(hf: &HFResult, d: &Matrix, k: &Matrix, g: f64) -> Fields {
    let x = &hf.mo_coefficients;
    let xt = x.transpose();
    let da = x.dot(d).dot(&xt);
    let ka = x.dot(k).dot(&xt);
    let (j, exchange) = contract_jk(&da.scale(2.), &[&da, &ka], &hf.eri, x.rows);
    let physical = hf.core_hamiltonian.add(&j).sub(&exchange[0]);
    let f = xt.dot(&physical).dot(x).sub(&d.scale(g));
    let coulomb_gap = xt.dot(&exchange[1]).dot(x);
    let gap = coulomb_gap.sub(&Matrix::identity(d.rows).scale(g * k.trace()));
    let normal_energy =
        inner(&da, &hf.core_hamiltonian.add(&physical)) + hf.nuclear_repulsion_energy;
    Fields {
        f,
        gap,
        normal_energy,
        anomalous_energy: inner(k, &coulomb_gap),
        pairing_energy: -g * (inner(d, d) + k.trace().powi(2)),
    }
}

fn nambu(f: &Matrix, gap: &Matrix, mu: f64) -> Matrix {
    let n = f.rows;
    let mut out = Matrix::zeros(2 * n, 2 * n);
    for p in 0..n {
        for q in 0..n {
            let normal = f[(p, q)] - if p == q { mu } else { 0. };
            out[(p, q)] = normal;
            out[(n + p, n + q)] = -normal;
            out[(p, n + q)] = gap[(p, q)];
            out[(n + p, q)] = gap[(p, q)];
        }
    }
    out
}

fn generalized(d: &Matrix, k: &Matrix) -> Matrix {
    let mut r = nambu(d, k, 0.);
    for p in 0..d.rows {
        r[(d.rows + p, d.rows + p)] += 1.;
    }
    r
}

fn densities(r: &Matrix) -> (Matrix, Matrix) {
    let n = r.rows / 2;
    let mut d = Matrix::zeros(n, n);
    let mut k = Matrix::zeros(n, n);
    for p in 0..n {
        for q in 0..n {
            d[(p, q)] = 0.5 * (r[(p, q)] - r[(n + p, n + q)] + if p == q { 1. } else { 0. });
            k[(p, q)] = 0.5 * (r[(p, n + q)] + r[(q, n + p)]);
        }
    }
    (d, k)
}

fn spectral_density(f: &Matrix, gap: &Matrix, mu: f64) -> Matrix {
    let n = f.rows;
    let (_, c) = nambu(f, gap, mu).eigh();
    let occupied = c.leading_columns(n);
    occupied.dot(&occupied.transpose())
}

/// Chemical potential is a constraint multiplier, never a clipped physical denominator.
fn number_constrained_density(
    f: &Matrix,
    gap: &Matrix,
    target: f64,
    tol: f64,
) -> Result<(Matrix, f64), String> {
    let n = f.rows;
    if target == 2. * n as f64 {
        let mu = f.eigh().0[n - 1] + max_abs_element(gap) + 1.;
        return Ok((generalized(&Matrix::identity(n), &Matrix::zeros(n, n)), mu));
    }
    let count = |r: &Matrix| 2. * (0..n).map(|i| r[(i, i)]).sum::<f64>();
    let mut bound = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| f[(i, j)].abs() + gap[(i, j)].abs())
                .sum::<f64>()
        })
        .fold(1., f64::max)
        + 1.;
    for _ in 0..32 {
        if count(&spectral_density(f, gap, -bound)) < target
            && count(&spectral_density(f, gap, bound)) > target
        {
            break;
        }
        bound *= 2.;
    }
    let (mut lower, mut upper) = (-bound, bound);
    for _ in 0..128 {
        let mu = 0.5 * (lower + upper);
        let r = spectral_density(f, gap, mu);
        let number = count(&r);
        if (number - target).abs() <= tol {
            return Ok((r, mu));
        }
        if number < target {
            lower = mu;
        } else {
            upper = mu;
        }
    }
    Err("HFB chemical potential did not satisfy the particle-number constraint".into())
}

/// Convenience driver with a real, spin-singlet paired vacuum and nonnegative coupling G.
pub fn run_hfb(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    pairing_strength_g: f64,
    max_iter: usize,
    tol: f64,
) -> Result<HFBResult, String> {
    run_hfb_with_options(
        atoms,
        basis,
        charge,
        &HfbOptions {
            pairing_strength: pairing_strength_g,
            max_iterations: max_iter,
            energy_tol: tol,
            residual_tol: (tol.sqrt() * 1e-2).clamp(1e-11, 1e-8),
            density_tol: (tol.sqrt() * 1e-2).clamp(1e-11, 1e-9),
            ..Default::default()
        },
    )
}

pub fn run_hfb_with_options(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    options: &HfbOptions,
) -> Result<HFBResult, String> {
    validate_options(options)?;
    let ne = atoms.iter().map(|a| a.charge).sum::<f64>().round() as i64 - i64::from(charge);
    if ne <= 0 || ne % 2 != 0 {
        return Err("spin-singlet HFB requires a positive even electron count; odd-particle blocking is not implemented".into());
    }
    let hf = run_scf_general(
        atoms,
        basis,
        charge,
        options.max_iterations,
        options.energy_tol.min(1e-12),
        ConvAlgo::Adiissoscf,
        None,
    )?;
    run_hfb_from_reference(&hf, options)
}

fn validate_options(o: &HfbOptions) -> Result<(), String> {
    if !o.pairing_strength.is_finite()
        || o.pairing_strength < 0.
        || o.max_iterations == 0
        || [o.energy_tol, o.residual_tol, o.density_tol, o.number_tol]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err(
            "HFB requires finite nonnegative G, positive tolerances and a nonzero iteration budget"
                .into(),
        );
    }
    Ok(())
}

/// Borrows the reference's packed AO integrals; no four-index MO tensor is constructed.
pub fn run_hfb_from_reference(hf: &HFResult, options: &HfbOptions) -> Result<HFBResult, String> {
    validate_options(options)?;
    if !hf.converged || hf.multiplicity != 1 {
        return Err("HFB requires a converged closed-shell RHF reference".into());
    }
    let n = hf.mo_coefficients.cols;
    if hf.n_occ == 0 || hf.n_occ > n {
        return Err("invalid HFB occupation".into());
    }
    let g = options.pairing_strength;
    let target = (2 * hf.n_occ) as f64;
    let mut d = Matrix::zeros(n, n);
    for p in 0..hf.n_occ {
        d[(p, p)] = 1.;
    }
    let mut k = Matrix::zeros(n, n);
    if g > 0. {
        // A coherent singlet vacuum at the required mean number. This starts on
        // the paired manifold without an arbitrary small-gap seed that can be
        // extrapolated back to an unstable normal fixed point.
        let filling = hf.n_occ as f64 / n as f64;
        d = Matrix::identity(n).scale(filling);
        k = Matrix::identity(n).scale((filling * (1. - filling)).sqrt());
    }
    let initial = fields(hf, &d, &k, g);
    let (_, mut mu) =
        number_constrained_density(&initial.f, &initial.gap, target, options.number_tol)?;
    let r = generalized(&d, &k);
    let mut previous_r = r;
    let mut previous_energy = f64::INFINITY;
    let mut diis = CdiisEngine::new(8);
    let mut converged = false;
    let mut iterations = 0;
    for iteration in 1..=options.max_iterations {
        iterations = iteration;
        let current = fields(hf, &d, &k, g);
        let r = generalized(&d, &k);
        let h = nambu(&current.f, &current.gap, mu);
        let residual = h.dot(&r).sub(&r.dot(&h));
        let energy = current.energy();
        if !energy.is_finite() {
            return Err("nonfinite HFB energy".into());
        }
        if (energy - previous_energy).abs() < options.energy_tol
            && max_abs_element(&residual) < options.residual_tol
            && max_abs_element(&r.sub(&previous_r)) < options.density_tol
            && (2. * d.trace() - target).abs() <= options.number_tol
        {
            // Also certify the occupied quasiparticle branch of the actual final fields.
            let (certified, certified_mu) =
                number_constrained_density(&current.f, &current.gap, target, options.number_tol)?;
            if max_abs_element(&certified.sub(&r)) < options.density_tol {
                let (cd, ck) = densities(&certified);
                let cf = fields(hf, &cd, &ck, g);
                let ch = nambu(&cf.f, &cf.gap, certified_mu);
                if max_abs_element(&ch.dot(&certified).sub(&certified.dot(&ch)))
                    < options.residual_tol
                {
                    (d, k) = (cd, ck);
                    mu = certified_mu;
                    converged = true;
                    break;
                }
            }
        }
        previous_energy = energy;
        previous_r = r.clone();
        diis.push(nambu(&current.f, &current.gap, 0.), residual);
        let extrapolated = diis
            .extrapolate()
            .unwrap_or_else(|| nambu(&current.f, &current.gap, 0.));
        let mut f = Matrix::zeros(n, n);
        let mut gap = Matrix::zeros(n, n);
        for p in 0..n {
            for q in 0..n {
                f[(p, q)] = extrapolated[(p, q)];
                gap[(p, q)] = extrapolated[(p, n + q)];
            }
        }
        let (mut next, mut next_mu) =
            number_constrained_density(&f, &gap, target, options.number_tol)?;
        let (candidate_d, candidate_k) = densities(&next);
        let roundoff = 32. * f64::EPSILON * (1. + energy.abs());
        if fields(hf, &candidate_d, &candidate_k, g).energy() > energy + roundoff {
            // CDIIS solves stationarity equations and can select a higher-energy
            // branch. The actual-field spectral projector is a constrained
            // descent direction. Backtracking its convex covariance step keeps
            // both the mean number and 0 <= R <= 1. Final certification restores
            // a pure spectral projector; the functional is never modified.
            diis = CdiisEngine::new(8);
            (next, next_mu) =
                number_constrained_density(&current.f, &current.gap, target, options.number_tol)?;
            let direction = next.sub(&r);
            let mut step = 1.;
            let mut accepted = false;
            for _ in 0..32 {
                let trial = r.add(&direction.scale(step));
                let (td, tk) = densities(&trial);
                if fields(hf, &td, &tk, g).energy() <= energy + roundoff {
                    next = trial;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                return Err("HFB constrained energy line search failed".into());
            }
        }
        mu = next_mu;
        (d, k) = densities(&next);
    }
    let final_fields = fields(hf, &d, &k, g);
    let r = generalized(&d, &k);
    let h = nambu(&final_fields.f, &final_fields.gap, mu);
    let (eigenvalues, _) = h.eigh();
    Ok(HFBResult {
        normal_energy: hf.total_energy,
        total_hfb_energy: final_fields.normal_energy
            + final_fields.anomalous_energy
            + final_fields.pairing_energy,
        normal_component_energy: final_fields.normal_energy,
        coulomb_pairing_energy: final_fields.anomalous_energy,
        pairing_energy: final_fields.pairing_energy,
        pairing_gap: final_fields
            .gap
            .eigh()
            .0
            .into_iter()
            .map(f64::abs)
            .fold(0., f64::max),
        chemical_potential: mu,
        quasiparticle_energies: eigenvalues[n..].to_vec(),
        normal_density: d.scale(2.),
        particle_number_variance: 4. * inner(&k, &k),
        pairing_tensor: k,
        orbital_coefficients: hf.mo_coefficients.clone(),
        number_error: (2. * d.trace() - target).abs(),
        residual_norm: max_abs_element(&h.dot(&r).sub(&r.dot(&h))),
        idempotence_error: max_abs_element(&r.dot(&r).sub(&r)),
        iterations,
        converged,
    })
}
