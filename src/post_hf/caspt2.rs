// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Compatibility interface for the shared internally contracted A-H backend.
//! IPEA uses canonical metric orthogonalization (OpenMolcas CORT convention).

use crate::context::Reference;
use crate::options::{ActiveSpace, PostHFOptions};
use crate::post_hf::caspt2_typed::{Caspt2Computed, compute_caspt2};
use crate::post_hf::casscf::CASSCFResult;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::scf::HFResult;

#[derive(Clone, Debug)]
pub struct CASPT2Options {
    pub n_active_electrons: usize,
    pub n_active_orbitals: usize,
    pub ipea_shift: f64,
    pub imaginary_shift: f64,
}
impl CASPT2Options {
    pub fn new(n_active_electrons: usize, n_active_orbitals: usize) -> Self {
        Self {
            n_active_electrons,
            n_active_orbitals,
            ipea_shift: 0.25,
            imaginary_shift: 0.0,
        }
    }
    pub fn with_ipea(mut self, shift: f64) -> Self {
        self.ipea_shift = shift;
        self
    }
    pub fn with_imaginary_shift(mut self, shift: f64) -> Self {
        self.imaginary_shift = shift;
        self
    }
    fn typed(&self) -> PostHFOptions {
        let mut options = PostHFOptions {
            active: Some(ActiveSpace::new(
                self.n_active_electrons.div_ceil(2),
                self.n_active_electrons / 2,
                self.n_active_orbitals,
            )),
            max_iter: 500,
            ..Default::default()
        };
        options.cas.ipea_shift = self.ipea_shift;
        options.cas.imaginary_shift = self.imaginary_shift;
        options
    }
}
impl Default for CASPT2Options {
    fn default() -> Self {
        Self::new(2, 2)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CASPT2Result {
    pub reference_energy: f64,
    pub casscf_energy: f64,
    pub pt2_correlation_energy: f64,
    pub total_energy: f64,
    /// Class H: two inactive electrons excited into external orbitals.
    pub e_external: f64,
    /// Classes A-G, including their interclass Hylleraas contributions.
    pub e_semi_internal: f64,
    /// Singles lie in the same contracted classes and have no separate
    /// invariant energy partition. This compatibility field is zero.
    pub e_singles: f64,
}
impl PostHFResult for CASPT2Result {
    fn method_name(&self) -> &'static str {
        "CASPT2"
    }
    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }
    fn correlation_energy(&self) -> f64 {
        self.total_energy - self.reference_energy
    }
    fn total_energy(&self) -> f64 {
        self.total_energy
    }
}
impl From<Caspt2Computed> for CASPT2Result {
    fn from(result: Caspt2Computed) -> Self {
        Self {
            reference_energy: result.output.e_scf,
            casscf_energy: result.reference_cas_energy,
            pt2_correlation_energy: result.equations.correlation_energy,
            total_energy: result.output.e_total,
            e_external: result.equations.class_energies[7],
            e_semi_internal: result.equations.class_energies[..7].iter().sum(),
            e_singles: 0.0,
        }
    }
}
pub struct CASPT2<'a, 'c> {
    pub hf: &'a HFResult,
    pub casscf: &'c CASSCFResult,
    pub options: CASPT2Options,
}
impl<'a, 'c> CASPT2<'a, 'c> {
    pub fn new(hf: &'a HFResult, casscf: &'c CASSCFResult, options: CASPT2Options) -> Self {
        Self {
            hf,
            casscf,
            options,
        }
    }
}
impl CASPT2<'_, '_> {
    pub fn from_hf(hf: &HFResult, options: CASPT2Options) -> Result<CASPT2Result, String> {
        compute_caspt2(Reference::Rhf(hf), &options.typed(), None, None)
            .map(Into::into)
            .map_err(|e| e.to_string())
    }
}
impl PostHFMethod for CASPT2<'_, '_> {
    type Output = CASPT2Result;
    fn compute(&self) -> Result<Self::Output, String> {
        if !self.casscf.converged || self.casscf.n_active_orbitals != self.options.n_active_orbitals
        {
            return Err(
                "CASPT2 requires a converged CASSCF reference with the requested active space"
                    .into(),
            );
        }
        let result = compute_caspt2(
            Reference::Rhf(self.hf),
            &self.options.typed(),
            None,
            Some(&self.casscf.mo_coefficients),
        )
        .map_err(|e| e.to_string())?;
        if (result.reference_cas_energy - self.casscf.total_energy).abs() > 1e-8 {
            return Err(
                "supplied CASSCF energy does not match the reconstructed active CI reference"
                    .into(),
            );
        }
        Ok(result.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{Atom, STO_3G_GBS, build_basis, parse_gaussian94};
    use crate::scf::run_scf_with_charge;

    #[test]
    fn test_caspt2_h2() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0, 0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let hf_res = run_scf_with_charge(&atoms, &basis, 0, 50, 1e-11).unwrap();

        let pt2_res = CASPT2::from_hf(&hf_res, CASPT2Options::new(2, 2)).unwrap();
        assert!(pt2_res.total_energy <= hf_res.total_energy);
        assert!((pt2_res.reference_energy - hf_res.total_energy).abs() < 1e-8);
    }
}
