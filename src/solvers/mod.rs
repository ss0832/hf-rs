// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Numerical solvers for electronic structure: Davidson-Liu eigensolver and Tensor DIIS.

pub mod davidson;
pub mod tdiis;

pub use davidson::{davidson_solver, try_davidson_solver, DavidsonOptions, DavidsonResult};
pub use tdiis::TensorDIIS;
