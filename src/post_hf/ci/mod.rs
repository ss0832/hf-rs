// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Configuration Interaction Engine: CIS, CISD, and FCI.

pub mod cis;
pub mod cisd;
pub mod fci;
pub mod mrci;
pub mod mrci_engine;

pub use cis::{CIS, CISResult, CISEstate};
pub use cisd::{CISD, CISDResult, compute_cisd_analytical_gradient};
pub use fci::{FCI, FCIResult, compute_fci_analytical_gradient};
pub use mrci::{MRCISD, MRCIResult};
