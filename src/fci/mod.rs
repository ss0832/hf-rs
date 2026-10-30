// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Configuration Interaction & Determinant String Engine.
//!
//! Provides:
//! - Knowles-Handy bitstring representation of alpha and beta determinant strings.
//! - Lexical addressing with binomial coefficient tables.
//! - O(1) excitation creation and annihilation with branchless bitwise parity.
//! - Slater-Condon matrix elements for 1- and 2-electron Hamiltonian operators.
//! - 1-particle and 2-particle reduced density matrices (1-RDM, 2-RDM).

pub mod engine;
pub mod excitations;
pub mod general_operator;
pub mod orbital_rotation;
pub mod rdm;
pub mod slater_condon;
pub mod string;
pub mod subspace;

pub use rdm::{compute_1rdm, compute_2rdm};
pub use slater_condon::{compute_hamiltonian_matrix, direct_h_matvec, slater_condon_element};
pub use string::{DeterminantSpace, SpinString, generate_strings};
