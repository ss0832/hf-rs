// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Multi-Reference Configuration Interaction Singles and Doubles (MR-CISD).
//!
//! Generates all single and double excitations out of a multi-determinant reference
//! (Complete Active Space model space M_0) into the external space:
//! |Psi_MRCI> = sum_{R in M_0} c_R |Phi_R> + sum_{S} c_S |Phi_S> + sum_{D} c_D |Phi_D>

use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::scf::HFResult;

/// Results of a Multi-Reference CISD calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct MRCIResult {
    pub reference_energy: f64,
    pub total_energy: f64,
    pub correlation_energy: f64,
    pub n_reference_determinants: usize,
    pub n_total_determinants: usize,
    pub root_energies: Vec<f64>,
}

impl PostHFResult for MRCIResult {
    fn method_name(&self) -> &'static str {
        "MR-CISD"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }

    fn total_energy(&self) -> f64 {
        self.total_energy
    }
}

/// Multi-Reference Configuration Interaction Singles and Doubles engine.
pub struct MRCISD<'a> {
    pub hf: &'a HFResult,
    pub n_active_electrons: usize,
    pub n_active_orbitals: usize,
    pub n_roots: usize,
}

impl<'a> MRCISD<'a> {
    pub fn new(hf: &'a HFResult, n_active_electrons: usize, n_active_orbitals: usize) -> Self {
        Self {
            hf,
            n_active_electrons,
            n_active_orbitals,
            n_roots: 1,
        }
    }

    pub fn with_roots(mut self, n_roots: usize) -> Self {
        self.n_roots = n_roots.max(1);
        self
    }
}

impl<'a> PostHFMethod for MRCISD<'a> {
    type Output = MRCIResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let mut options=crate::options::PostHFOptions{active:Some(crate::options::ActiveSpace::new(self.n_active_electrons.div_ceil(2),self.n_active_electrons/2,self.n_active_orbitals)),n_roots:self.n_roots,..Default::default()};
        options.cas.optimize_orbitals=false;
        let result=super::mrci_engine::compute_mrci(crate::context::Reference::Rhf(self.hf),&options,None,false).map_err(|e|e.to_string())?;
        Ok(MRCIResult{reference_energy:self.hf.total_energy,total_energy:result.output.e_total,
            correlation_energy:result.output.e_corr,n_reference_determinants:result.n_reference_determinants,
            n_total_determinants:result.determinants.len(),root_energies:result.output.states.iter().map(|s|s.energy).collect()})
    }
}
