// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Time-Dependent Hartree-Fock (TDHF) / Random Phase Approximation (RPA) for a closed-shell RHF reference.
//!
//! Spin-adapted real orbital Hessian blocks (chemists' notation):
//!
//! * singlet: `A = D + 2 (ia|jb) - (ij|ab)`, `B = 2 (ia|jb) - (ib|ja)`,
//! * triplet: `A = D - (ij|ab)`,             `B = -(ib|ja)`,
//!
//! with `D_{ia,jb} = (e_a - e_i) d_ij d_ab`. Real excitation energies require
//! `A - B` and `A + B` to be positive definite (a stable reference); otherwise an
//! instability error is returned. The Hermitian reduction
//! `(A - B)^{1/2} (A + B) (A - B)^{1/2} Z = w^2 Z` gives `X + Y = w^{-1/2} (A - B)^{1/2} Z`
//! and `X - Y = w^{1/2} (A - B)^{-1/2} Z`, normalized to `(X+Y).(X-Y) = 1`.
//! Singlet transition dipoles are `mu = sqrt(2) sum_ia (X+Y)_ia <i|r|a>` (electron
//! charge sign omitted, as usual) and `f = 2/3 w |mu|^2`; triplet transitions are
//! spin forbidden.

use crate::basis::{Atom, ContractedGaussian};
use crate::diis::ConvAlgo;
use crate::matrix::Matrix;
use crate::mo_integrals::transform_eri_ao_to_mo;
use crate::scf::{run_scf_general, HFResult};

const HARTREE_TO_EV: f64 = 27.211_386_245_988;
const DENSE_LIMIT_BYTES: u128 = 256 * 1024 * 1024;

/// One TDHF/RPA excited state.
#[derive(Clone, Debug)]
pub struct TDHFState {
    /// 1-based index within its spin manifold, in ascending energy.
    pub state_index: usize,
    pub excitation_energy_hartree: f64,
    pub excitation_energy_ev: f64,
    pub oscillator_strength: f64,
    pub transition_dipole: [f64; 3],
    pub is_triplet: bool,
    /// Excitation amplitudes `X_ia` (row index `i * n_virt + a`), with `(X+Y).(X-Y) = 1`.
    pub x: Vec<f64>,
    /// De-excitation amplitudes `Y_ia`.
    pub y: Vec<f64>,
}

/// Results from a TDHF / RPA calculation: singlets (ascending) followed by triplets (ascending).
#[derive(Clone, Debug)]
pub struct TDHFResult {
    pub reference_energy: f64,
    pub excited_states: Vec<TDHFState>,
    pub n_singlet_states: usize,
    pub n_triplet_states: usize,
    pub n_occupied: usize,
    pub n_virtual: usize,
}

impl TDHFResult {
    pub fn singlets(&self) -> impl Iterator<Item = &TDHFState> {
        self.excited_states.iter().filter(|s| !s.is_triplet)
    }

    pub fn triplets(&self) -> impl Iterator<Item = &TDHFState> {
        self.excited_states.iter().filter(|s| s.is_triplet)
    }
}

/// Runs RHF and then TDHF for the lowest `n_states_request` singlets. Triplets (whose RPA
/// problem is often unstable for minimal bases) are available from [`run_tdhf_from_reference`].
pub fn run_tdhf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    n_states_request: usize,
    max_iter: usize,
    tol: f64,
) -> Result<TDHFResult, String> {
    let hf = run_scf_general(atoms, basis, charge, max_iter, tol, ConvAlgo::Adiissoscf, None)?;
    if !hf.converged {
        return Err("TDHF requires a converged RHF reference".into());
    }
    run_tdhf_from_reference(&hf, basis, n_states_request, 0)
}

/// TDHF on an existing converged closed-shell RHF reference.
pub fn run_tdhf_from_reference(
    hf: &HFResult,
    basis: &[ContractedGaussian],
    n_singlets: usize,
    n_triplets: usize,
) -> Result<TDHFResult, String> {
    let n_occ = hf.n_occ;
    let n_mo = hf.mo_coefficients.cols;
    if hf.orbital_energies.len() != n_mo || hf.mo_coefficients.rows != basis.len() {
        return Err("TDHF reference/basis dimension mismatch".into());
    }
    if n_occ == 0 || n_occ >= n_mo {
        return Err("TDHF requires at least one occupied and one virtual orbital".into());
    }
    let n_virt = n_mo - n_occ;
    let n_ov = n_occ * n_virt;
    if 4 * 8 * (n_ov as u128).pow(2) > DENSE_LIMIT_BYTES {
        return Err("dense TDHF orbital Hessian exceeds 256 MiB".into());
    }
    let eps = &hf.orbital_energies;
    let mo = transform_eri_ao_to_mo(&hf.mo_coefficients, &hf.eri);
    let dipoles: [Matrix; 3] = std::array::from_fn(|comp| {
        hf.mo_coefficients
            .transpose()
            .dot(&crate::properties::compute_dipole_matrix(basis, comp))
            .dot(&hf.mo_coefficients)
    });

    let mut states = Vec::new();
    for (triplet, wanted) in [(false, n_singlets), (true, n_triplets)] {
        if wanted == 0 {
            continue;
        }
        let mut a_mat = Matrix::zeros(n_ov, n_ov);
        let mut b_mat = Matrix::zeros(n_ov, n_ov);
        for i in 0..n_occ {
            for a in 0..n_virt {
                let row = i * n_virt + a;
                for j in 0..n_occ {
                    for b in 0..n_virt {
                        let col = j * n_virt + b;
                        let (am, bm) = (n_occ + a, n_occ + b);
                        let iajb = mo.get(i, am, j, bm);
                        let ijab = mo.get(i, j, am, bm);
                        let ibja = mo.get(i, bm, j, am);
                        let coulomb = if triplet { 0.0 } else { 2.0 };
                        let mut av = coulomb * iajb - ijab;
                        if row == col {
                            av += eps[am] - eps[i];
                        }
                        a_mat[(row, col)] = av;
                        b_mat[(row, col)] = coulomb * iajb - ibja;
                    }
                }
            }
        }
        let label = if triplet { "triplet" } else { "singlet" };
        let (lam, u) = a_mat.sub(&b_mat).eigh();
        if lam[0] <= 1e-12 * lam[n_ov - 1].abs().max(1.0) {
            return Err(format!(
                "{label} TDHF: A - B is not positive definite (lowest eigenvalue {:.3e}); the RHF reference is unstable",
                lam[0]
            ));
        }
        let sqrt_lam: Vec<f64> = lam.iter().map(|v| v.sqrt()).collect();
        // (A-B)^{1/2} and (A-B)^{-1/2} in the orbital-pair basis
        let pow = |power: f64| -> Matrix {
            let mut scaled = u.clone();
            for r in 0..n_ov {
                for c in 0..n_ov {
                    scaled[(r, c)] *= sqrt_lam[c].powf(power);
                }
            }
            scaled.dot(&u.transpose())
        };
        let root = pow(1.0);
        let inv_root = pow(-1.0);
        let m = root.dot(&a_mat.add(&b_mat)).dot(&root);
        let (omega2, z) = m.eigh();
        if omega2[0] <= 0.0 {
            return Err(format!(
                "{label} TDHF: imaginary excitation energy (omega^2 = {:.3e}); the RHF reference is unstable",
                omega2[0]
            ));
        }
        for k in 0..wanted.min(n_ov) {
            let omega = omega2[k].sqrt();
            let zk: Vec<f64> = (0..n_ov).map(|r| z[(r, k)]).collect();
            let apply = |mat: &Matrix, scale: f64| -> Vec<f64> {
                (0..n_ov).map(|r| scale * (0..n_ov).map(|c| mat[(r, c)] * zk[c]).sum::<f64>()).collect()
            };
            let plus = apply(&root, omega.powf(-0.5));
            let minus = apply(&inv_root, omega.sqrt());
            let x: Vec<f64> = plus.iter().zip(&minus).map(|(p, q)| 0.5 * (p + q)).collect();
            let y: Vec<f64> = plus.iter().zip(&minus).map(|(p, q)| 0.5 * (p - q)).collect();
            let mut td = [0.0; 3];
            if !triplet {
                for (comp, d) in dipoles.iter().enumerate() {
                    let mut v = 0.0;
                    for i in 0..n_occ {
                        for a in 0..n_virt {
                            v += plus[i * n_virt + a] * d[(i, n_occ + a)];
                        }
                    }
                    td[comp] = std::f64::consts::SQRT_2 * v;
                }
            }
            let f = 2.0 / 3.0 * omega * (td[0] * td[0] + td[1] * td[1] + td[2] * td[2]);
            states.push(TDHFState {
                state_index: k + 1,
                excitation_energy_hartree: omega,
                excitation_energy_ev: omega * HARTREE_TO_EV,
                oscillator_strength: f,
                transition_dipole: td,
                is_triplet: triplet,
                x,
                y,
            });
        }
    }
    let n_singlet_states = states.iter().filter(|s| !s.is_triplet).count();
    let n_triplet_states = states.len() - n_singlet_states;
    Ok(TDHFResult {
        reference_energy: hf.total_energy,
        excited_states: states,
        n_singlet_states,
        n_triplet_states,
        n_occupied: n_occ,
        n_virtual: n_virt,
    })
}
