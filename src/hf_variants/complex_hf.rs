// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Complex RHF/UHF/GHF in ordinary Gaussian AOs with a common magnetic gauge.
//! The checked solver and variational gradient share the same stationary density.

pub use super::complex_engine::{
    ComplexHFResult, ComplexHfOptions, ComplexHfReference, build_complex_hf_fock,
    compute_complex_hf_gradient, run_complex_hf, run_complex_hf_with_options,
};
