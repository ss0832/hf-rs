// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Numerical nuclear derivatives of the complete MP2/CCSD-F12 total energies.
use super::{CABSBasis, MP2F12Options, RHFCCSDF12, RHFMP2F12};
use crate::{
    ao::BasisKind,
    basis::{Atom, ContractedGaussian},
    error::HfError,
    post_hf::numerical::{
        NumericalGradientOptions, NumericalGradientResult, compute_rhf_numerical_gradient,
        displaced_basis,
    },
    scf::HFResult,
};

impl RHFMP2F12<'_> {
    /// Central energy differences in bohr, with fresh SCF, CABS, full Fock and
    /// occupied-pair intermediates at every displaced geometry. Atom-owned raw
    /// auxiliary functions move with their atoms; CABS coefficients are rebuilt.
    pub fn numerical_gradient(
        &self,
        options: &NumericalGradientOptions,
    ) -> Result<NumericalGradientResult, HfError> {
        f12_numerical_gradient(
            self.hf_res,
            self.atoms,
            self.basis,
            &self.options,
            options,
            |hf, atoms, basis, settings| {
                Ok(RHFMP2F12::new(hf, atoms, basis)
                    .with_options(settings)
                    .compute_checked()?
                    .total_energy)
            },
        )
    }
}

impl RHFCCSDF12<'_> {
    /// Reconverge the modified CC equations and rebuild CABS at every geometry.
    /// CC energy/residual tolerances are tightened to 1e-13/1e-12 respectively
    /// when the requested settings are looser; the iteration limit is preserved.
    pub fn numerical_gradient(
        &self,
        options: &NumericalGradientOptions,
    ) -> Result<NumericalGradientResult, HfError> {
        let mut cc = self.ccsd_options.clone();
        if cc.max_iterations == 0
            || !cc.energy_tol.is_finite()
            || cc.energy_tol <= 0.
            || !cc.residual_tol.is_finite()
            || cc.residual_tol <= 0.
        {
            return Err(HfError::InvalidInput(
                "invalid CCSD-F12 gradient convergence controls".into(),
            ));
        }
        cc.energy_tol = cc.energy_tol.min(1e-13);
        cc.residual_tol = cc.residual_tol.min(1e-12);
        f12_numerical_gradient(
            self.hf_res,
            self.atoms,
            self.basis,
            &self.f12_options,
            options,
            |hf, atoms, basis, settings| {
                Ok(RHFCCSDF12::new(hf, atoms, basis, self.variant)
                    .with_f12_options(settings)
                    .with_ccsd_options(cc.clone())
                    .with_triples(self.compute_triples)
                    .compute_checked()?
                    .total_energy)
            },
        )
    }
}

fn f12_numerical_gradient(
    reference: &HFResult,
    reference_atoms: &[Atom],
    basis: &[ContractedGaussian],
    f12_options: &MP2F12Options,
    options: &NumericalGradientOptions,
    mut energy: impl FnMut(
        &HFResult,
        &[Atom],
        &[ContractedGaussian],
        MP2F12Options,
    ) -> Result<f64, HfError>,
) -> Result<NumericalGradientResult, HfError> {
    if !reference.converged || reference.mo_coefficients.rows != basis.len() {
        return Err(HfError::InvalidInput(
            "F12 gradient needs a compatible converged RHF reference".into(),
        ));
    }
    if let Some(cabs) = &f12_options.custom_cabs {
        cabs.validate_reference(reference, basis.len())
            .map_err(HfError::InvalidInput)?;
    }
    if f12_options.fitting_basis.is_some() || f12_options.fock_fitting_basis.is_some() {
        return Err(HfError::Unsupported {
            method: "fitted F12 correction gradient".into(),
            detail: "the fitting-space response has not been implemented".into(),
        });
    }
    let kind = reference
        .ao_space
        .as_ref()
        .map_or(BasisKind::Cartesian, |space| space.kind);
    let mut cabs_rank = f12_options.custom_cabs.as_ref().map(|cabs| cabs.n_cabs);
    compute_rhf_numerical_gradient(
        reference_atoms,
        basis,
        reference.charge,
        kind,
        options,
        |hf, atoms, basis| {
            if hf.n_occ != reference.n_occ
                || hf.mo_coefficients.cols != reference.mo_coefficients.cols
            {
                return Err(HfError::Response(
                    "F12 reference occupied count/OBS rank changed on reconvergence".into(),
                ));
            }
            let cabs = if let Some(original) = &f12_options.custom_cabs {
                let aux = displaced_basis(&original.aux_functions, reference_atoms, atoms)?;
                CABSBasis::from_reference(hf, basis, &aux, original.metric_threshold)
            } else {
                CABSBasis::from_builtin_reference(hf, basis, atoms, "cc-pvdz-f12-optri", 1e-8)
            }
            .map_err(HfError::Integral)?;
            if cabs_rank.is_some_and(|expected| expected != cabs.n_cabs) {
                return Err(HfError::Response(
                    "CABS rank changes across numerical-gradient geometries".into(),
                ));
            }
            cabs_rank = Some(cabs.n_cabs);
            let mut f12_options = f12_options.clone();
            f12_options.custom_cabs = Some(cabs);
            energy(hf, atoms, basis, f12_options)
        },
    )
}
