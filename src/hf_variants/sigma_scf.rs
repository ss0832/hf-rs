// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! sigma-SCF: single determinants that minimize the energy variance about a target `omega`
//! (Ye, Welborn, Ricke and Van Voorhis, J. Chem. Phys. 147, 214104 (2017)).
//!
//! `H` connects a determinant only to itself and its single and double excitations, so the
//! variance of an unrestricted determinant is exact in closed form (spin-orbital notation):
//!
//! `sigma^2(omega) = <(H - omega)^2> = (E - omega)^2 + sum_ia f_ai^2 + 1/4 sum_ijab <ab||ij>^2`.
//!
//! With the real rotation `phi_i -> phi_i + kappa phi_a`, `phi_a -> phi_a - kappa phi_i` the
//! orbital gradient is
//!
//! `g_ai = 4 (E - omega) f_ai + 2 sum_b f_ab f_bi - 2 sum_j f_aj f_ji
//!        + 2 sum_jb f_bj (<ba||ji> + <bi||ja>)
//!        + sum_bcj <bc||ij> <bc||aj> - sum_bjk <ab||jk> <ib||jk>`.
//!
//! The variance is minimized over occupied-virtual rotations of each spin by preconditioned
//! L-BFGS ([`super::orbital_optimizer`]). Occupied-occupied and virtual-virtual rotations leave
//! the determinant unchanged. Integrals are dense MO tensors, so memory is `O(n^4)` and time
//! `O(o^2 v^3)` per iteration: the method is intended for small systems.

use super::orbital_optimizer::{Controls, Evaluation, max_abs, minimize};
use super::state_averaged::StateConfiguration;
use crate::basis::{Atom, ContractedGaussian};
use crate::context::{SpinEriBlocks, transform_mixed_eri};
use crate::matrix::Matrix;
use crate::mo_integrals::transform_eri_ao_to_mo;
use crate::scf::{
    build_fock_matrix, build_uhf_fock_matrices, compute_eri_tensor, compute_nuclear_repulsion,
    compute_one_electron_integrals,
};
use std::sync::Arc;

/// Convergence controls for [`run_sigma_scf_with_options`].
#[derive(Clone, Debug)]
pub struct SigmaScfOptions {
    pub max_iterations: usize,
    pub gradient_tol: f64,
    pub variance_tol: f64,
}

impl Default for SigmaScfOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            gradient_tol: 1e-7,
            variance_tol: 1e-12,
        }
    }
}

/// Result of a sigma-SCF calculation.
#[derive(Clone, Debug)]
pub struct SigmaSCFResult {
    /// Energy expectation `<H>` of the optimized determinant in Hartree.
    pub total_energy: f64,
    /// Variance `<(H - omega)^2>` in Hartree^2.
    pub energy_variance: f64,
    /// `<H^2> - <H>^2`, the omega-independent part, in Hartree^2.
    pub intrinsic_variance: f64,
    /// Target energy `omega` in Hartree.
    pub target_energy: f64,
    pub mo_coefficients_alpha: Matrix,
    pub mo_coefficients_beta: Matrix,
    /// Spin-summed density matrix.
    pub density_matrix: Matrix,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub gradient_norm: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// sigma-SCF for the determinant given by spatial `target_occupations` of the guess orbitals
/// (2 = doubly occupied, 1 = alpha, 0 = empty); see [`StateConfiguration::from_occupations`].
pub fn run_sigma_scf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    target_energy: f64,
    target_occupations: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<SigmaSCFResult, String> {
    let configuration = StateConfiguration::from_occupations(target_occupations)?;
    run_sigma_scf_with_options(
        atoms,
        basis,
        target_energy,
        &configuration,
        &SigmaScfOptions {
            max_iterations: max_iter,
            gradient_tol: (tol.sqrt() * 0.1).clamp(1e-10, 1e-5),
            variance_tol: tol,
        },
    )
}

struct Model<'a> {
    h: &'a Matrix,
    eri: &'a [f64],
    nuclear: f64,
    omega: f64,
    n_alpha: usize,
    n_beta: usize,
}

struct Point {
    ca: Matrix,
    cb: Matrix,
    energy: f64,
    variance: f64,
    intrinsic_variance: f64,
}

impl Model<'_> {
    /// Variance, gradient over `[alpha (v_a x o_a), beta (v_b x o_b)]` and preconditioner.
    fn evaluate(&self, ca: Matrix, cb: Matrix) -> Evaluation<Point> {
        let n = ca.rows;
        let (na, nb) = (self.n_alpha, self.n_beta);
        let pa = ca.occupied_density(na, 1.0);
        let pb = cb.occupied_density(nb, 1.0);
        let (fa, fb) = build_uhf_fock_matrices(self.h, &pa, &pb, self.eri, n);
        let tr = |a: &Matrix, b: &Matrix| -> f64 {
            a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum()
        };
        let energy = 0.5 * (tr(&pa.add(&pb), self.h) + tr(&pa, &fa) + tr(&pb, &fb)) + self.nuclear;
        let fm = [
            ca.transpose().dot(&fa).dot(&ca),
            cb.transpose().dot(&fb).dot(&cb),
        ];
        let blocks = SpinEriBlocks {
            alpha: Arc::new(transform_eri_ao_to_mo(&ca, self.eri)),
            beta: Arc::new(transform_eri_ao_to_mo(&cb, self.eri)),
            cross: Some(Arc::new(
                transform_mixed_eri(&ca, &cb, self.eri).expect("same AO space"),
            )),
        };
        // Interleaved spin orbitals: 2k = alpha orbital k, 2k + 1 = beta orbital k.
        let m = ca.cols;
        let occupied: Vec<usize> = (0..na)
            .map(|k| 2 * k)
            .chain((0..nb).map(|k| 2 * k + 1))
            .collect();
        let virtual_: Vec<usize> = (na..m)
            .map(|k| 2 * k)
            .chain((nb..m).map(|k| 2 * k + 1))
            .collect();
        let f = |p: usize, q: usize| {
            if p % 2 == q % 2 {
                fm[p % 2][(p / 2, q / 2)]
            } else {
                0.0
            }
        };
        let v = |p, q, r, s| blocks.get(p, q, r, s);
        let mut singles = 0.0;
        for &i in &occupied {
            for &a in &virtual_ {
                singles += f(a, i).powi(2);
            }
        }
        let mut doubles = 0.0;
        for (x, &i) in occupied.iter().enumerate() {
            for &j in &occupied[x + 1..] {
                for (y, &a) in virtual_.iter().enumerate() {
                    for &b in &virtual_[y + 1..] {
                        doubles += v(a, b, i, j).powi(2);
                    }
                }
            }
        }
        let variance = (energy - self.omega).powi(2) + singles + doubles;
        // Gradient in the order alpha (a, i) then beta (a, i), row-major over (a, i).
        let mut gradient = Vec::new();
        let mut diagonal = Vec::new();
        for spin in 0..2 {
            let (occ_s, vir_s): (Vec<usize>, Vec<usize>) = (
                occupied.iter().copied().filter(|p| p % 2 == spin).collect(),
                virtual_.iter().copied().filter(|p| p % 2 == spin).collect(),
            );
            for &a in &vir_s {
                for &i in &occ_s {
                    let mut g = 4.0 * (energy - self.omega) * f(a, i);
                    for &b in &virtual_ {
                        g += 2.0 * f(a, b) * f(b, i);
                    }
                    for &j in &occupied {
                        g -= 2.0 * f(a, j) * f(j, i);
                    }
                    for &j in &occupied {
                        for &b in &virtual_ {
                            let fbj = f(b, j);
                            if fbj != 0.0 {
                                g += 2.0 * fbj * (v(b, a, j, i) + v(b, i, j, a));
                            }
                        }
                    }
                    for &j in &occupied {
                        for &b in &virtual_ {
                            for &c in &virtual_ {
                                g += v(b, c, i, j) * v(b, c, a, j);
                            }
                        }
                    }
                    for &b in &virtual_ {
                        for &j in &occupied {
                            for &k in &occupied {
                                g -= v(a, b, j, k) * v(i, b, j, k);
                            }
                        }
                    }
                    let gap = f(a, a) - f(i, i);
                    gradient.push(g);
                    diagonal.push(2.0 * gap * gap + (4.0 * (energy - self.omega) * gap).abs());
                }
            }
        }
        Evaluation {
            state: Point {
                ca,
                cb,
                energy,
                variance,
                intrinsic_variance: singles + doubles,
            },
            value: variance,
            gradient,
            diagonal,
        }
    }

    fn rotate(&self, point: &Point, kappa: &[f64]) -> Evaluation<Point> {
        let rotate_spin = |c: &Matrix, n_occ: usize, kappa: &[f64]| {
            let m = c.cols;
            let mut x = Matrix::zeros(m, m);
            for (k, &value) in kappa.iter().enumerate() {
                let (a, i) = (n_occ + k / n_occ, k % n_occ);
                // phi_i -> phi_i + kappa phi_a, phi_a -> phi_a - kappa phi_i
                x[(a, i)] = value;
                x[(i, a)] = -value;
            }
            c.dot(&crate::post_hf::mr_dsrg::cumulants::exp_antisymmetric(&x))
        };
        let split = (point.ca.cols - self.n_alpha) * self.n_alpha;
        let ca = if self.n_alpha > 0 {
            rotate_spin(&point.ca, self.n_alpha, &kappa[..split])
        } else {
            point.ca.clone()
        };
        let cb = if self.n_beta > 0 {
            rotate_spin(&point.cb, self.n_beta, &kappa[split..])
        } else {
            point.cb.clone()
        };
        self.evaluate(ca, cb)
    }
}

/// sigma-SCF for an explicit determinant of the guess orbitals (canonical orbitals of the
/// superposition-of-atoms Fock operator, ordered by energy), reordered so that the occupied
/// orbitals of each spin come first.
pub fn run_sigma_scf_with_options(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    target_energy: f64,
    configuration: &StateConfiguration,
    options: &SigmaScfOptions,
) -> Result<SigmaSCFResult, String> {
    let n = basis.len();
    if n == 0 || !target_energy.is_finite() {
        return Err("sigma-SCF requires a basis and a finite target energy".into());
    }
    if options.max_iterations == 0
        || [options.gradient_tol, options.variance_tol]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err("invalid sigma-SCF convergence controls".into());
    }
    let (s, _, _, h) = compute_one_electron_integrals(atoms, basis);
    let x = s
        .canonical_orthogonalizer(1e-8)
        .map_err(|e| e.to_string())?;
    let m = x.cols;
    let (alpha, beta) = (&configuration.alpha, &configuration.beta);
    for occupied in [alpha, beta] {
        let mut sorted = occupied.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != occupied.len() || occupied.iter().any(|&k| k >= m) {
            return Err(
                "sigma-SCF occupied orbitals must be distinct and within the basis rank".into(),
            );
        }
    }
    let electrons = alpha.len() + beta.len();
    if electrons == 0 {
        return Err("sigma-SCF requires at least one electron".into());
    }
    let nuclear_charge: f64 = atoms.iter().map(|a| a.charge).sum();
    let charge = nuclear_charge.round() as i32 - electrons as i32;
    let eri = compute_eri_tensor(basis);
    let guess = match crate::sad::compute_sad_density(atoms, basis, charge, &s) {
        Ok(p) => build_fock_matrix(&h, &p, &eri, n),
        Err(_) => h.clone(),
    };
    let c0 = x.dot(&x.transpose().dot(&guess).dot(&x).eigh().1);
    // Occupied orbitals first, in the given order, then the remaining orbitals by energy.
    let ordered = |occupied: &[usize]| {
        let order: Vec<usize> = occupied
            .iter()
            .copied()
            .chain((0..m).filter(|k| !occupied.contains(k)))
            .collect();
        Matrix::from_vec(
            n,
            m,
            (0..n)
                .flat_map(|mu| order.iter().map(move |&k| (mu, k)))
                .map(|(mu, k)| c0[(mu, k)])
                .collect(),
        )
    };
    let model = Model {
        h: &h,
        eri: &eri,
        nuclear: compute_nuclear_repulsion(atoms),
        omega: target_energy,
        n_alpha: alpha.len(),
        n_beta: beta.len(),
    };
    let minimum = minimize(
        model.evaluate(ordered(alpha), ordered(beta)),
        &Controls {
            max_iterations: options.max_iterations,
            gradient_tol: options.gradient_tol,
            value_tol: options.variance_tol,
            max_step: 0.3,
        },
        |point, kappa| model.rotate(point, kappa),
    );
    let gradient_norm = max_abs(&minimum.point.gradient);
    if !minimum.converged {
        return Err(format!(
            "sigma-SCF did not converge in {} iterations: gradient {gradient_norm:e}",
            options.max_iterations
        ));
    }
    let Point {
        ca,
        cb,
        energy,
        variance,
        intrinsic_variance,
    } = minimum.point.state;
    let density = ca
        .occupied_density(model.n_alpha, 1.0)
        .add(&cb.occupied_density(model.n_beta, 1.0));
    Ok(SigmaSCFResult {
        total_energy: energy,
        energy_variance: variance,
        intrinsic_variance,
        target_energy,
        mo_coefficients_alpha: ca,
        mo_coefficients_beta: cb,
        density_matrix: density,
        n_alpha: model.n_alpha,
        n_beta: model.n_beta,
        gradient_norm,
        iterations: minimum.iterations,
        converged: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orbitals() -> (Matrix, Vec<f64>, Matrix, f64) {
        let atoms = vec![
            Atom::new("H", [0.1, -0.2, -0.7]),
            Atom::new("H", [-0.1, 0.3, 0.7]),
        ];
        let basis = crate::build_basis(&atoms, &crate::get_builtin_basis("6-31g").unwrap());
        let (s, _, _, h) = compute_one_electron_integrals(&atoms, &basis);
        (
            h,
            compute_eri_tensor(&basis),
            s.canonical_orthogonalizer(1e-8).unwrap(),
            compute_nuclear_repulsion(&atoms),
        )
    }

    fn frame(x: &Matrix, seed: usize) -> Matrix {
        let n = x.cols;
        let mut generator = Matrix::zeros(n, n);
        for p in 0..n {
            for q in 0..p {
                let angle = 0.17 * ((3 * p + 7 * q + seed) as f64).sin();
                generator[(p, q)] = angle;
                generator[(q, p)] = -angle;
            }
        }
        x.dot(&crate::post_hf::mr_dsrg::cumulants::exp_antisymmetric(
            &generator,
        ))
    }

    #[test]
    fn every_sigma_orbital_gradient_component_matches_variance_differences() {
        let (h, eri, x, nuclear) = orbitals();
        let mut worst = 0f64;
        let mut checked = 0;
        for seed in 1..=3 {
            for (na, nb) in [(1, 1), (2, 1), (1, 0), (0, 1), (2, 2)] {
                let model = Model {
                    h: &h,
                    eri: &eri,
                    nuclear,
                    omega: -0.6,
                    n_alpha: na,
                    n_beta: nb,
                };
                let point = model.evaluate(frame(&x, seed), frame(&x, seed + 11));
                for k in 0..point.gradient.len() {
                    let step = 1e-4;
                    let value = |multiple| {
                        let mut delta = vec![0.; point.gradient.len()];
                        delta[k] = multiple * step;
                        model.rotate(&point.state, &delta).value
                    };
                    let reference =
                        (-value(2.) + 8. * value(1.) - 8. * value(-1.) + value(-2.)) / (12. * step);
                    let error = (point.gradient[k] - reference).abs();
                    worst = worst.max(error);
                    assert!(
                        error < 3e-7,
                        "seed {seed}, ({na},{nb}), orbital gradient {k}: {error:e}"
                    );
                    checked += 1;
                }
            }
        }
        eprintln!(
            "sigma-SCF: all {checked} arbitrary-frame orbital gradient components, max error {worst:e}"
        );
    }

    #[test]
    fn intrinsic_variance_is_independent_of_large_target_offsets() {
        let (h, eri, x, nuclear) = orbitals();
        let ca = frame(&x, 2);
        let cb = frame(&x, 13);
        let physical = Model {
            h: &h,
            eri: &eri,
            nuclear,
            omega: 0.,
            n_alpha: 1,
            n_beta: 1,
        };
        let shifted = Model {
            omega: -1e9,
            ..physical
        };
        let reference = physical
            .evaluate(ca.clone(), cb.clone())
            .state
            .intrinsic_variance;
        let actual = shifted.evaluate(ca, cb).state;
        assert!(reference > 1e-4);
        assert!((actual.intrinsic_variance - reference).abs() < 1e-14);
        // Subtracting two target-dominated squared energies discards this information.
        assert!(
            ((actual.variance - (actual.energy - shifted.omega).powi(2)) - reference).abs() > 1e-4
        );
    }
}
