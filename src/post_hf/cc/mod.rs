// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Coupled Cluster Theory Engine (CCD, CCSD, CCSD(T)).

pub mod ccd;
pub mod ccsd;
pub mod ccsd_t;
pub mod typed_triples;
mod rhf_ccsd;

pub use ccd::{CCD, CCDResult};
pub use ccsd::{CCSD, CCSDOptions, CCSDResult, UHFCCSD};
pub use ccsd_t::{compute_ccsd_t_correction, CCSDTResult};
