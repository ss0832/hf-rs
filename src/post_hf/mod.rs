// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Post-Hartree-Fock electronic structure methods.
//!
//! Provides a unified trait hierarchy, single-reference perturbation theory (MPn),
//! coupled cluster theory (CC), configuration interaction (CI), multi-reference methods (CASSCF),
//! and domain-based local pair natural orbital methods (DLPNO).

pub mod caspt2;
pub mod caspt2_engine;
pub mod caspt2_typed;
pub mod casscf;
pub mod casscf_engine;
pub mod cc;
pub mod ci;
pub mod dlpno;
pub mod dmrg_engine;
pub mod dmrg_orbitals;
pub mod f12;
pub mod grad_common;
pub mod hessian;
pub mod mp2;
pub mod mp2_gradient;
pub mod mp2_hessian;
pub mod mpn;
pub mod mr_dsrg;
pub mod numerical;
pub mod ri_mp2;
pub mod scs_mp2;
pub mod scs_mp3;

pub use grad_common::{
    compute_cis_analytical_gradient, compute_mrci_analytical_gradient,
    compute_rhf_cphf_analytical_corr_gradient, compute_rhf_total_rdm_analytical_gradient,
    compute_subset_rdms, compute_uhf_cphf_analytical_corr_gradient, solve_cpuhf,
};
pub use mp2_gradient::{compute_rhf_mp2_analytical_gradient, compute_uhf_mp2_analytical_gradient};
pub use mp2_hessian::{compute_rhf_mp2_analytical_hessian, compute_uhf_mp2_analytical_hessian};

pub use caspt2::{CASPT2, CASPT2Options, CASPT2Result};
pub use casscf::{CASSCF, CASSCFOptions, CASSCFResult};
pub use cc::{
    CCD, CCDResult, CCSD, CCSDOptions, CCSDResult, CCSDTResult, UHFCCSD, compute_ccsd_t_correction,
};
pub use ci::{CIS, CISD, CISDResult, CISEstate, CISResult, FCI, FCIResult, MRCIResult, MRCISD};
pub use dlpno::{
    DLPNOMCCSD, DLPNOMCCSDResult, DLPNOMCCSDT, DLPNOMCCSDTResult, DLPNOMP2, DLPNOMP2Options,
    DLPNOMP2Result,
};
pub use f12::{
    CABSBasis, CCSDF12Result, CCSDF12Variant, F12Approximation, GeminalCorrelationFactor,
    MP2F12Options, MP2F12Result, RHFCCSDF12, RHFMP2F12, compute_cabs_singles,
};
pub use mpn::{
    MP3Result, MP4Result, MPnResult, RHFMP3, RHFMP4, RHFMP5, RHFMP6, RHFMP7, RHFMP8, RHFMPn, UHFMP3,
};
pub use mr_dsrg::{
    ACI, ACIOptions, ACIResult, ASCI, ASCIOptions, ASCIResult, ASET, ASETOptions, ASETResult,
    ASETVirtualScheme, AVAS, AVASOptions, AVASResult, ActiveSpaceDMRG, CumulantDecomposition,
    DMRGOptions, DMRGResult, DOCI, DOCIResult, DSRGMRPT2, DSRGMRPT3, GASCI, GASOptions, GASResult,
    GASSubspace, HBCI, HBCIOptions, HBCIResult, MRDSRGOptions, MRDSRGResult, MRLDSRG2,
    MultiStateDSRG, MultiStateDSRGResult, MultiStateDSRGVariant, NaturalOrbitalKind,
    NaturalOrbitalResult, NaturalOrbitalSelector, PCI, PCIGenerator, PCIOptions, PCIResult,
    RelaxRef, SRDSRGPT2, SRDSRGPT3, SRDSRGResult, SRGGenerator, SRGODE, SRGODEResult, SRLDSRG2,
    ThreeDSRGMRPT2, ThreePDC, UHFSRDSRGPT2, ZerothOrderH, build_generalized_fock_mo,
    compute_aci_analytical_gradient, compute_cumulants, compute_dmrg_analytical_gradient,
    compute_doci_analytical_gradient, compute_gas_analytical_gradient, dsrg_regularized_inv_denom,
    dsrg_residual_weight, evaluate_gno_reference_energy, semi_canonicalize,
};

pub use hessian::{
    PostHFType, compute_mp2_gradient, compute_mp2_hessian, compute_mp2_hessian_with_charge,
    compute_mp2_vibrational_frequencies, compute_post_hf_gradient, compute_post_hf_hessian,
    compute_post_hf_vibrational_frequencies, compute_uhf_mp2_gradient, compute_uhf_mp2_hessian,
    compute_uhf_post_hf_gradient, compute_uhf_post_hf_hessian,
};

use crate::matrix::Matrix;

/// Trait implemented by all post-Hartree-Fock calculation engines.
pub trait PostHFMethod {
    type Output: PostHFResult;

    /// Runs the post-HF method and produces the result containing energies and wavefunctions.
    fn compute(&self) -> Result<Self::Output, String>;
}

/// Common result interface for all post-Hartree-Fock methods.
pub trait PostHFResult {
    /// Retained method tensors. Energy-only calculations do not copy them.
    fn intermediates(
        &self,
        max_elements: usize,
    ) -> Result<crate::output::TensorMap, crate::error::HfError> {
        Ok(crate::output::TensorExport::new(max_elements)?.finish())
    }
    fn converged(&self) -> bool {
        true
    }
    /// Convert legacy result fields into a consistent selected-state output.
    /// Context adapters rebase MR correlation energies on the original SCF energy.
    fn to_output(&self) -> Result<crate::output::PostHFOutput, crate::error::HfError> {
        let mut output = crate::output::PostHFOutput::new(
            self.method_name(),
            self.reference_energy(),
            self.total_energy(),
        )?;
        output.one_rdm = self.one_rdm().cloned();
        Ok(output)
    }
    /// Canonical name of the method (e.g. "RHF-MP2", "UHF-MP2", "CCSD(T)").
    fn method_name(&self) -> &'static str;

    /// Reference Hartree-Fock energy E_0 in Hartree.
    fn reference_energy(&self) -> f64;

    /// Correlation energy E_corr in Hartree.
    fn correlation_energy(&self) -> f64;

    /// Total energy E_total = E_0 + E_corr in Hartree.
    fn total_energy(&self) -> f64 {
        self.reference_energy() + self.correlation_energy()
    }

    /// One-particle reduced density matrix in MO basis (gamma_{pq}), if available.
    fn one_rdm(&self) -> Option<&Matrix> {
        None
    }

    /// Natural orbital occupation numbers (eigenvalues of 1-RDM), if available.
    fn natural_orbital_occupations(&self) -> Option<Vec<f64>> {
        self.one_rdm().map(|rdm| {
            let (evals, _) = rdm.eigh();
            evals
        })
    }
}
