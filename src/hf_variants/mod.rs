// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Comprehensive Taxonomy of Hartree-Fock (HF) Variants.
//!
//! Encompasses:
//! 1. Symmetry-Breaking & Fukutome Classification:
//!    - GHF (Generalized Hartree-Fock with 2-component non-collinear spinors)
//!    - Fukutome 8-class symmetry analyzer (TICS, CCW, ASCW, ASDW, ASW, TSCW, TSDW, TSW)
//!    - Complex HF (cRHF, cUHF, cGHF with broken time-reversal symmetry)
//! 2. Symmetry Restoration & Superconductivity / Pairing:
//!    - PUHF (exact Lowdin projection after variation, with a separate Yamaguchi estimate)
//!    - HFB (Hartree-Fock-Bogoliubov pairing theory with anomalous density kappa)
//! 3. Multi-Configuration & Static Correlation:
//!    - GVB-PP / TCSCF (Generalized Valence Bond & Two-Configuration SCF)
//! 4. Linear Response & Excited States:
//!    - TDHF / RPA (Time-Dependent Hartree-Fock & Random Phase Approximation)
//!    - SA-HF / State-Averaged ROHF (State-averaged ensemble HF)
//!    - \sigma-SCF (Variance-minimizing targeted excited state HF)
//! 5. Extended Particles Beyond Born-Oppenheimer:
//!    - NEO-HF (Nuclear-Electronic Orbital HF with quantum protons)
//!    - Positronic HF (coupled electron-positron multi-component HF)
//! 6. Strong Light-Matter Coupling:
//!    - QED-HF (Cavity Quantum Electrodynamics HF / Polaritonic HF)
//! 7. Low-Rank 2-Electron Integral Approximations:
//!    - CD-HF (Cholesky Decomposition HF with O(N^3) scaling)
//!    - DF-HF / RI-HF (Density Fitting / Resolution of Identity HF)
//! 8. Relativistic Effects:
//!    - X2C-HF / DKH2 / ZORA-HF (Scalar relativistic mass-velocity & Darwin corrections)
//! 9. Periodic & Grid-Based Systems:
//!    - CO-HF (1D Crystal Orbital Periodic HF with k-point band structure)
//!    - Numerical HF (Basis-set-free atomic radial finite-difference HF limit)
//!    - LHF / OEP-EXX (Localized Hartree-Fock / Optimized Effective Potential exact exchange)
//! 10. Analytical Nuclear Hessians:
//!    - Comprehensive nuclear Hessian and vibrational analysis for all HF variants

pub mod ghf;
pub mod phf;
pub mod gvb;
pub mod tdhf;
pub mod neo;
pub mod qed_hf;
pub mod cholesky_df;
pub mod relativistic;
pub mod x2c_core;
pub mod x2c_integrals;
pub mod dkh2_core;
mod stationary_rhf;
pub mod spin_projection;
mod relativistic_engine;
pub mod crystal;
pub mod complex_hf;
mod complex_matrix;
mod complex_engine;
pub mod magnetic;
pub mod symmetry;
mod orbital_optimizer;
pub mod state_averaged;
pub mod sigma_scf;
pub mod positronic_hf;
pub mod ri_df_hf;
pub mod lhf_oep;
mod local_grid;
mod local_exchange;
pub mod local_potential;
pub mod oep;
pub mod numerical_hf;
mod radial_1s;
pub mod dc_hf;
pub mod fmo;
pub mod hessian;

pub use ghf::{
    FukutomeClass, GHFResult, classify_fukutome_symmetry, run_ghf, build_ghf_fock_matrix,
    GhfOptions, run_ghf_with_options, compute_ghf_gradient,
    ghf_spin_expectations,
};
pub use spin_projection::{SpinProjection, project_uhf_spin};
pub mod hfb;
pub use hfb::{HfbOptions, run_hfb_with_options, run_hfb_from_reference};
pub use phf::{
    PUHFResult, run_puhf, run_puhf_with_multiplicity, HFBResult, run_hfb,
};
pub use gvb::{
    TCSCFResult, run_tcscf, run_tcscf_from_reference, compute_tcscf_gradient,
};
pub use tdhf::{
    TDHFResult, TDHFState, run_tdhf, run_tdhf_from_reference,
};
pub use neo::{
    NEOHFResult, NeoHfOptions, run_neo_hf, run_neo_hf_with_options, PROTON_MASS_AU,
};
pub use qed_hf::{
    QEDHFResult, QedHfOptions, run_qed_hf, run_qed_hf_with_options, compute_qed_hf_gradient,
};
pub use cholesky_df::{
    CDHFResult, run_cd_hf, run_cd_hf_with_options, CD_HF_FACTOR_BUDGET,
};
pub use relativistic::{
    RelativisticHFResult, RelativisticHfOptions, RelativisticModel, run_relativistic_hf, SPEED_OF_LIGHT_AU,
    run_relativistic_hf_with_options, compute_relativistic_hf_gradient,
};
pub use crystal::{
    COHFResult, BandPoint, CoHfOptions, run_crystal_orbital_hf, run_crystal_orbital_hf_with_options,
};
pub use complex_hf::{
    ComplexHFResult, ComplexHfOptions, ComplexHfReference, run_complex_hf,
    run_complex_hf_with_options, compute_complex_hf_gradient, build_complex_hf_fock,
};
pub use state_averaged::{
    SaHfOptions, StateAveragedHFResult, StateConfiguration, run_state_averaged_hf, run_state_averaged_hf_with_options,
};
pub use sigma_scf::{
    SigmaSCFResult, SigmaScfOptions, run_sigma_scf, run_sigma_scf_with_options,
};
pub use positronic_hf::{
    PositronicHFResult, PositronicHfOptions, run_positronic_hf, run_positronic_hf_with_options,
};
pub use ri_df_hf::{
    DFHFOptions, DFHFResult, run_df_hf, run_df_hf_with_aux, run_factorized_rhf,
};
pub use lhf_oep::{
    LHFResult, LhfOptions, LhfExchangeResult, LocalGridOptions, run_lhf_scf,
    run_lhf_with_options, run_lhf_from_reference, evaluate_lhf_exchange,
};
pub use oep::{
    OepOptions, OepResult, run_oep_hf, run_oep_hf_from_reference,
    evaluate_oep_hf_from_reference,
};
pub use numerical_hf::{
    NumericalHFResult, NumericalAtomicOptions, run_numerical_atomic_hf, run_numerical_atomic_1s_hf,
};
pub use dc_hf::{
    DCSubsystem, DCOptions, DCHFResult, partition_molecular_system, run_dc_hf, compute_dc_hf_gradient,
};
pub use fmo::{
    FMOFragment, FMOOptions, FMOPairInteraction, FMOClusterResult, FMOResult, auto_partition_fragments, run_fmo, compute_fmo_gradient,
};
pub use hessian::{
    HFVariantType, HFVariantHessianResult, evaluate_hf_variant_energy,
    compute_hf_variant_gradient, compute_hf_variant_analytical_gradient, compute_hf_variant_numerical_gradient, compute_hf_variant_hessian,
    compute_hf_variant_frequencies, run_hf_variant_hessian_analysis,
};
