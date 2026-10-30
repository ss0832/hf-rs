// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Numerical radial atomic HF. Supported configurations are H 1s and He 1s^2;
//! the explicit ion interface also supports one/two-electron positive ions.
//! Energies are finite-grid values with separately certified SCF convergence.

pub use super::radial_1s::{NumericalAtomicOptions, NumericalHFResult, run_numerical_atomic_1s_hf};

/// Neutral-atom compatibility driver. Multishell neutral atoms fail explicitly.
pub fn run_numerical_atomic_hf(
    atomic_number: usize,
    n_grid: usize,
    max_iter: usize,
    tol: f64,
) -> Result<NumericalHFResult, String> {
    if atomic_number > 2 {
        return Err("neutral numerical atomic HF currently supports H and He; use the explicit 1s ion interface for higher Z".into());
    }
    run_numerical_atomic_1s_hf(
        atomic_number,
        atomic_number,
        &NumericalAtomicOptions {
            n_grid_points: n_grid,
            max_iterations: max_iter,
            energy_tol: tol,
            residual_tol: (tol.sqrt() * 1e-2).clamp(1e-8, 1e-6),
            density_tol: (tol.sqrt() * 1e-2).clamp(1e-10, 1e-7),
            ..Default::default()
        },
    )
}
