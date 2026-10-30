// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Advanced Multireference, Selected CI, Active-Space, and Driven Similarity Renormalization
//! Group (`MR-DSRG`) Post-Hartree-Fock Electronic Structure Suite.
//!
//! Includes:
//! - [`cumulants`]: Generalized Normal Ordering (GNO), Mukherjee-Kutzelnigg 1-, 2-, and 3-body
//!   density cumulants ($\gamma_1, \eta_1, \lambda_2, \lambda_3$), generalized Fock matrices,
//!   and semi-canonical orbital transformations.
//! - [`sci`]: Selected and Projector Configuration Interaction (`ACI`, `ACI+PT2`, `SA-ACI`,
//!   `ASCI`, `ASCI+PT2`, `HBCI`, `HBCI+PT2`, and `PCI` with 10 polynomial / Krylov generators).
//! - [`gasci`]: Generalized Active Space CI & SCF (`GASCI`, `GASSCF` across `GAS1..GAS6`)
//!   and seniority-zero Doubly Occupied Configuration Interaction (`DOCI`).
//! - [`dmrg`]: 2-site sweep Active-Space Density Matrix Renormalization Group (`ActiveSpaceDMRG`, `DMRGSCF`).
//! - [`avas`]: Atomic Valence Active Space (`AVAS`), Active Space Embedding Theory (`ASET` with
//!   `ASET`, `PAO`, and `IAO` virtual spaces), and correlated Natural Orbital selectors (`MP2NO`, `CINO`, `MRCINO`).
//! - [`dsrg`]: Single-reference Driven Similarity Renormalization Group (`SR-DSRG-PT2`,
//!   `UHF-SR-DSRG-PT2`, `SR-DSRG-PT3`, `SR-LDSRG(2)`) and Continuous Flow `SRG-ODE` (`Wegner`, `White`).
//! - [`mrdsrg`]: Multireference Driven Similarity Renormalization Group (`DSRG-MRPT2`,
//!   Cholesky/DF `THREE-DSRG-MRPT2`, `DSRG-MRPT3`, `MR-LDSRG(2)`, `MR-LDSRG(2)-QC`, and
//!   multi-state `SA-DSRG`, `MS-DSRG-PT2`, `XMS-DSRG-PT2`, `DWMS-DSRG-PT2`).

pub mod cumulants;
pub mod density_cumulants;
pub mod gno;
pub mod mr_driven;
pub mod relaxation;
pub mod typed_mr;
pub mod three_index;
pub mod three_reference;
pub mod multistate;
pub mod sci;
pub mod selected_ci;
pub mod projector_ci;
pub mod gasci;
pub mod dmrg;
pub mod avas;
pub mod dsrg;
pub mod imsrg2;
pub mod sr_dsrg;
pub mod mrdsrg;

pub use cumulants::{
    ThreePDC, CumulantDecomposition,
    build_generalized_fock_mo, semi_canonicalize, compute_cumulants, evaluate_gno_reference_energy,
};
pub use sci::{
    ACI, ACIOptions, ACIResult, compute_aci_analytical_gradient,
    ASCI, ASCIOptions, ASCIResult,
    HBCI, HBCIOptions, HBCIResult,
    PCI, PCIOptions, PCIResult, PCIGenerator,
};
pub use gasci::{
    GASSubspace, GASOptions, GASResult, GASCI, compute_gas_analytical_gradient,
    DOCI, DOCIResult, compute_doci_analytical_gradient,
};
pub use dmrg::{
    DMRGOptions, DMRGResult, ActiveSpaceDMRG, compute_dmrg_analytical_gradient,
};
pub use avas::{
    AVAS, AVASOptions, AVASResult,
    ASET, ASETOptions, ASETResult, ASETVirtualScheme,
    NaturalOrbitalSelector, NaturalOrbitalKind, NaturalOrbitalResult,
};
pub use dsrg::{
    dsrg_regularized_inv_denom, dsrg_residual_weight,
    SRDSRGResult, SRDSRGPT2, UHFSRDSRGPT2, SRDSRGPT3, SRLDSRG2,
    SRGGenerator, SRGODEResult, SRGODE,
};
pub use mrdsrg::{
    ZerothOrderH, RelaxRef, MRDSRGOptions, MRDSRGResult,
    DSRGMRPT2, ThreeDSRGMRPT2, DSRGMRPT3, MRLDSRG2,
    MultiStateDSRGVariant, MultiStateDSRGResult, MultiStateDSRG,
};
