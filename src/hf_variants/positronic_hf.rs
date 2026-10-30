// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Positronic Hartree-Fock: closed-shell electrons plus one positron.
//!
//! The positron has charge +1 and the electron mass. It is repelled by the nuclei, attracted by
//! the electronic density (no electron-positron exchange: the particles are distinguishable),
//! and has no self-interaction. This is the two-component NEO-HF model
//! ([`super::neo`]) with one positive particle of unit mass; the coupled equations are solved
//! by the same block-descent plus coupled-DIIS procedure. `charge` is the charge of the
//! electronic molecule without the positron (PsH is H with `charge = -1`). Positron
//! annihilation and electron-positron correlation are not included.

use super::neo::{NeoHfOptions, solve_two_component};
use crate::basis::{Atom, ContractedGaussian};
use crate::matrix::Matrix;

/// Convergence controls for [`run_positronic_hf_with_options`].
#[derive(Clone, Debug)]
pub struct PositronicHfOptions {
    pub max_iterations: usize,
    pub energy_tol: f64,
    pub residual_tol: f64,
    pub density_tol: f64,
}

impl Default for PositronicHfOptions {
    fn default() -> Self {
        let neo = NeoHfOptions::default();
        Self {
            max_iterations: neo.max_iterations,
            energy_tol: neo.energy_tol,
            residual_tol: neo.residual_tol,
            density_tol: neo.density_tol,
        }
    }
}

/// Result of a Positronic Hartree-Fock calculation.
#[derive(Clone, Debug)]
pub struct PositronicHFResult {
    /// Total energy of the electron-positron bound system in Hartree.
    pub total_energy: f64,
    /// `Tr(P_e h) + 1/2 Tr(P_e G_e)`: electrons in the field of the nuclei.
    pub electronic_energy: f64,
    /// `Tr(P_p (T - V_nuc))`: positron kinetic energy and nuclear repulsion.
    pub positronic_energy: f64,
    /// Electron-positron Coulomb attraction `-Tr(P_e J[P_p])` (negative).
    pub ep_attraction_energy: f64,
    /// Nuclear repulsion energy in Hartree.
    pub nuclear_energy: f64,
    /// Positron kinetic energy `Tr(P_p T)`.
    pub positron_kinetic_energy: f64,
    /// Occupied positron orbital energy in Hartree.
    pub positron_orbital_energy: f64,
    /// Electronic orbital energies in Hartree.
    pub electron_orbital_energies: Vec<f64>,
    /// Spin-summed electronic density matrix.
    pub electron_density: Matrix,
    /// Positronic density matrix (trace with the overlap = 1).
    pub positron_density: Matrix,
    pub electron_mo_coefficients: Matrix,
    pub positron_mo_coefficients: Matrix,
    pub residual_norm: f64,
    /// Number of coupled SCF iterations taken.
    pub iterations: usize,
    /// Whether the SCF converged.
    pub converged: bool,
}

/// Positronic HF with the electronic basis also used for the positron. `tol` is the
/// energy-change criterion; commutator and density criteria are derived from it.
pub fn run_positronic_hf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Result<PositronicHFResult, String> {
    run_positronic_hf_with_options(
        atoms,
        basis,
        basis,
        charge,
        &PositronicHfOptions {
            max_iterations: max_iter,
            energy_tol: tol,
            residual_tol: (tol.sqrt() * 1e-2).clamp(1e-11, 1e-6),
            density_tol: (tol.sqrt() * 1e-2).clamp(1e-11, 1e-6),
        },
    )
}

/// Positronic HF with separate electronic and positronic bases.
pub fn run_positronic_hf_with_options(
    atoms: &[Atom],
    electronic_basis: &[ContractedGaussian],
    positronic_basis: &[ContractedGaussian],
    charge: i32,
    options: &PositronicHfOptions,
) -> Result<PositronicHFResult, String> {
    if electronic_basis.is_empty() || positronic_basis.is_empty() {
        return Err("positronic HF requires electronic and positronic bases".into());
    }
    if crate::ecp::has_ecp(atoms) {
        return Err("positronic HF does not support effective core potentials".into());
    }
    let nuclear_charge: f64 = atoms.iter().map(|a| a.charge).sum();
    let n_electrons = nuclear_charge.round() as i64 - i64::from(charge);
    if n_electrons <= 0 || n_electrons % 2 != 0 {
        return Err(format!(
            "closed-shell positronic HF requires a positive even electron count, got {n_electrons}"
        ));
    }
    let r = solve_two_component(
        atoms,
        electronic_basis,
        positronic_basis,
        n_electrons as usize / 2,
        1,
        &NeoHfOptions {
            proton_mass: 1.0,
            max_iterations: options.max_iterations,
            energy_tol: options.energy_tol,
            residual_tol: options.residual_tol,
            density_tol: options.density_tol,
        },
    )?;
    Ok(PositronicHFResult {
        total_energy: r.total_energy,
        electronic_energy: r.electronic_energy,
        positronic_energy: r.protonic_energy,
        ep_attraction_energy: r.electron_proton_coulomb,
        nuclear_energy: r.classical_nuc_repulsion,
        positron_kinetic_energy: r.protonic_kinetic_energy,
        positron_orbital_energy: r.protonic_orbital_energies[0],
        electron_orbital_energies: r.electronic_orbital_energies,
        electron_density: r.electronic_density,
        positron_density: r.protonic_density,
        electron_mo_coefficients: r.electronic_mo_coefficients,
        positron_mo_coefficients: r.protonic_mo_coefficients,
        residual_norm: r.residual_norm,
        iterations: r.iterations,
        converged: r.converged,
    })
}
