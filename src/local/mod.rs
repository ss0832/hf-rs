// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Local correlation infrastructure: Orbital Localization (Foster-Boys / Pipek-Mezey),
//! Projected Atomic Orbitals (PAOs), and Pair Natural Orbitals (PNOs).

pub mod localization;
pub mod pao;
pub mod pno;
pub mod pair_space;
pub mod domains;

pub use localization::{localize_foster_boys, localize_pipek_mezey, LocalizationMethod};
pub use pao::{construct_paos, PAODomain};
pub use pno::{construct_pnos, PNOBasis, PNOOptions};
pub mod triples_space;
pub mod transport;
