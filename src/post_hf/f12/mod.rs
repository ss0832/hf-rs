// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Explicitly correlated F12 methods (MP2-F12, CCSD-F12a, CCSD-F12b, CCSD(T)-F12).
//!
//! Provides:
//! - Slater-type geminal (STG) correlation factor F_{12} = -(1/\gamma) * exp(-\gamma * r_{12})
//!   expanded with the Tew--Klopper six-Gaussian fit.
//! - Analytical Gaussian geminal integrals for arbitrary angular momentum (s, p, d, f, g...).
//! - Complementary Auxiliary Basis Set (CABS) projector and CABS singles correction.
//! - MP2-F12/3C(FIX), complete CABS Fock coupling and fixed cusp amplitudes.
//! - Canonical MP2-F12/3*A(FIX), GBC/EBC and a projected kinetic commutator.
//! - Nonlocal ECPs in 3C and 3*A, with explicit nonlocal commutator products.
//! - CCSD-F12a/b rewrites remain in progress.

pub mod geminal;
pub mod cabs;
pub mod mp2_f12;
pub mod ccsd_f12;
pub mod ccsd_intermediates;

pub use geminal::{
    GeminalCorrelationFactor,
    primitive_geminal_eri,
    contracted_geminal_integral,
    contracted_stg_integral,
    compute_geminal_tensor,
};

pub use cabs::{
    CABSBasis,
    compute_cabs_singles,
};

pub use mp2_f12::{
    F12Approximation,
    F12Hybrid,
    MP2F12Options,
    MP2F12Result,
    RHFMP2F12,
};

pub use ccsd_f12::{
    CCSDF12Variant,
    CCSDF12Result,
    RHFCCSDF12,
};

pub mod operators;
pub mod density_fit;
mod gaussian_quartet;
pub mod intermediates;
pub mod molecular;
pub mod nuclear;
pub mod typed_ccsd;
