// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Scalar one-electron relativistic Hamiltonians and certified HF solvers.

pub use super::relativistic_engine::{
    RelativisticHFResult, RelativisticHfOptions, RelativisticModel, SPEED_OF_LIGHT_AU,
    compute_relativistic_core_hamiltonian, compute_relativistic_hf_gradient, run_relativistic_hf,
    run_relativistic_hf_with_options,
};
