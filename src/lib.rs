// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

#![forbid(unsafe_code)]
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::excessive_precision
)]

pub mod matrix;
pub mod basis_data;
pub mod basis;
mod builtin_catalog;
mod molecular_basis;
pub mod ao;
pub mod df;
pub mod options;
pub mod context;
pub mod output;
pub mod registry;
pub mod api;
pub mod cli;
pub mod validation;
pub mod shell_integrals;
pub mod ecp;
pub mod cholesky;
pub mod error;
pub mod hf_derivatives;
pub mod integrals;
pub mod scf;
pub mod uhf_stability;
pub mod gradient;
pub mod derivatives;
pub mod cphf;
pub mod hessian;
pub mod third_deriv;
pub mod fourth_deriv;
pub mod molden;
pub mod properties;
pub mod sad;
pub mod diis;
pub mod masses;
pub mod orbitals;
pub mod mo_integrals;
pub mod solvers;
pub mod fci;
pub mod local;
pub mod post_hf;
pub mod rohf;
pub mod hf_variants;
pub mod py_bindings;
mod py_api;

pub use masses::*;
pub use orbitals::{OrbitalSpace, OrbitalSpaceType, SpinOrbitalSpace};
pub use mo_integrals::{
    MoEriTensor, OovvIntegrals, ActiveMoIntegrals,
    transform_eri_ao_to_mo, transform_oovv, transform_active, transform_1e_ao_to_mo,
};
pub use solvers::{davidson_solver, DavidsonOptions, DavidsonResult, TensorDIIS};
pub use fci::{SpinString, DeterminantSpace, generate_strings, compute_1rdm, compute_2rdm};
pub use local::{localize_foster_boys, localize_pipek_mezey, construct_paos, construct_pnos, LocalizationMethod, PAODomain, PNOBasis, PNOOptions};
pub use post_hf::{
    PostHFMethod, PostHFResult,
    mp2::{RHFMP2, MP2Result, UHFMP2, UHFMP2Result},
    mpn::{RHFMP3, UHFMP3, MP3Result, RHFMP4, MP4Result, MPnResult, RHFMPn, RHFMP5, RHFMP6, RHFMP7, RHFMP8},
    cc::{CCD, CCDResult, CCSD, CCSDOptions, CCSDResult, UHFCCSD, compute_ccsd_t_correction, CCSDTResult},
    ci::{CIS, CISResult, CISEstate, CISD, CISDResult, FCI, FCIResult, MRCISD, MRCIResult},
    casscf::{CASSCF, CASSCFOptions, CASSCFResult},
    caspt2::{CASPT2, CASPT2Options, CASPT2Result},
    dlpno::{
        DLPNOMP2, DLPNOMP2Options, DLPNOMP2Result,
        DLPNOMCCSD, DLPNOMCCSDResult,
        DLPNOMCCSDT, DLPNOMCCSDTResult,
    },
    f12::{
        GeminalCorrelationFactor, CABSBasis, compute_cabs_singles,
        F12Approximation, MP2F12Options, MP2F12Result, RHFMP2F12,
        CCSDF12Variant, CCSDF12Result, RHFCCSDF12,
    },
    mr_dsrg::{
        ThreePDC, CumulantDecomposition,
        build_generalized_fock_mo, semi_canonicalize, compute_cumulants, evaluate_gno_reference_energy,
        ACI, ACIOptions, ACIResult, compute_aci_analytical_gradient,
        ASCI, ASCIOptions, ASCIResult,
        HBCI, HBCIOptions, HBCIResult,
        PCI, PCIOptions, PCIResult, PCIGenerator,
        GASSubspace, GASOptions, GASResult, GASCI, compute_gas_analytical_gradient,
        DOCI, DOCIResult, compute_doci_analytical_gradient,
        DMRGOptions, DMRGResult, ActiveSpaceDMRG, compute_dmrg_analytical_gradient,
        AVAS, AVASOptions, AVASResult,
        ASET, ASETOptions, ASETResult, ASETVirtualScheme,
        NaturalOrbitalSelector, NaturalOrbitalKind, NaturalOrbitalResult,
        dsrg_regularized_inv_denom, dsrg_residual_weight,
        SRDSRGResult, SRDSRGPT2, UHFSRDSRGPT2, SRDSRGPT3, SRLDSRG2,
        SRGGenerator, SRGODEResult, SRGODE,
        ZerothOrderH, RelaxRef, MRDSRGOptions, MRDSRGResult,
        DSRGMRPT2, ThreeDSRGMRPT2, DSRGMRPT3, MRLDSRG2,
        MultiStateDSRGVariant, MultiStateDSRGResult, MultiStateDSRG,
    },
    hessian::{
        PostHFType,
        compute_post_hf_gradient, compute_uhf_post_hf_gradient,
        compute_post_hf_hessian, compute_uhf_post_hf_hessian,
        compute_post_hf_vibrational_frequencies,
        compute_post_hf_hessian_analysis,
        compute_mp2_gradient, compute_uhf_mp2_gradient,
        compute_mp2_hessian, compute_mp2_hessian_with_charge, compute_uhf_mp2_hessian,
        compute_mp2_vibrational_frequencies,
    },
};
pub use basis::{
    Atom, BasisSetMap, ContractedGaussian, build_basis, try_build_basis, load_basis_from_file,
    parse_gaussian94, try_parse_gaussian94, parse_bse_json, shell_type_to_l, l_to_shell_type, cartesian_components_for_l,
    get_builtin_basis, get_builtin_aux_basis, list_builtin_basis_sets, list_builtin_aux_basis_sets,
    STO_3G_GBS, BASIS_3_21G_GBS, BASIS_6_31G_GBS, BASIS_6_31G_STAR_GBS, BASIS_6_31G_STAR_STAR_GBS, BASIS_6_311G_SS_GBS,
    BASIS_DEF2_SVP_GBS, BASIS_DEF2_TZVP_GBS,
    BASIS_CC_PVDZ_GBS, BASIS_CC_PVTZ_GBS, BASIS_AUG_CC_PVDZ_GBS,
};
pub use ecp::{assign_builtin_ecps, assign_ecps, get_builtin_ecp, EcpDef, EcpMap};
pub use cholesky::{pivoted_cholesky_eri, CholeskyInfo};
pub use matrix::Matrix;
pub use ao::{AoSpace,BasisKind};
pub use error::HfError;
pub use hf_derivatives::{HfNuclearDerivativeContext, NuclearDirection};
pub use diis::ConvAlgo;
pub use properties::{MolecularProperties, compute_molecular_properties, compute_mulliken_charges, compute_lowdin_charges};
pub use sad::compute_sad_density;
pub use scf::{
    run_scf, run_scf_with_charge, run_scf_with_initial_density, run_scf_general, run_scf_in_space, run_uhf, run_uhf_in_space,
    HFResult, UHFResult, analyze_rhf_stability, StabilityAnalysisResult,
    rotate_rhf_orbitals, run_rhf_with_stability_rotation, analyze_rhf_internal_stability, rhf_stability_matrix,
    build_uhf_fock_matrices,
};
pub use uhf_stability::{UhfStability, uhf_internal_stability};
pub use rohf::{
    ROHFResult, run_rohf, run_qrhf, run_scf_mom, run_scf_fermi_smearing, FermiSmearingResult,
};
pub use hf_variants::{
    FukutomeClass, GHFResult, classify_fukutome_symmetry, run_ghf, build_ghf_fock_matrix,
    GhfOptions, run_ghf_with_options, compute_ghf_gradient, ghf_spin_expectations,
    PUHFResult, run_puhf, run_puhf_with_multiplicity, HFBResult, run_hfb, SpinProjection, project_uhf_spin,
    HfbOptions, run_hfb_with_options, run_hfb_from_reference,
    TCSCFResult, run_tcscf, run_tcscf_from_reference, compute_tcscf_gradient,
    TDHFResult, TDHFState, run_tdhf, run_tdhf_from_reference,
    NEOHFResult, NeoHfOptions, run_neo_hf, run_neo_hf_with_options, PROTON_MASS_AU,
    QEDHFResult, QedHfOptions, run_qed_hf, run_qed_hf_with_options, compute_qed_hf_gradient,
    CDHFResult, run_cd_hf, run_cd_hf_with_options, CD_HF_FACTOR_BUDGET,
    RelativisticHFResult, RelativisticHfOptions, RelativisticModel, run_relativistic_hf, SPEED_OF_LIGHT_AU,
    run_relativistic_hf_with_options, compute_relativistic_hf_gradient,
    COHFResult, BandPoint, CoHfOptions, run_crystal_orbital_hf, run_crystal_orbital_hf_with_options,
    ComplexHFResult, ComplexHfOptions, ComplexHfReference, run_complex_hf,
    run_complex_hf_with_options, compute_complex_hf_gradient, build_complex_hf_fock,
    SaHfOptions, StateAveragedHFResult, StateConfiguration, run_state_averaged_hf, run_state_averaged_hf_with_options,
    SigmaSCFResult, SigmaScfOptions, run_sigma_scf, run_sigma_scf_with_options,
    PositronicHFResult, PositronicHfOptions, run_positronic_hf, run_positronic_hf_with_options,
    DFHFOptions, DFHFResult, run_df_hf, run_df_hf_with_aux, run_factorized_rhf,
    LHFResult, LhfOptions, LhfExchangeResult, LocalGridOptions, run_lhf_scf,
    run_lhf_with_options, run_lhf_from_reference, evaluate_lhf_exchange,
    OepOptions, OepResult, run_oep_hf, run_oep_hf_from_reference, evaluate_oep_hf_from_reference,
    NumericalHFResult, NumericalAtomicOptions, run_numerical_atomic_hf, run_numerical_atomic_1s_hf,
    DCSubsystem, DCOptions, DCHFResult, partition_molecular_system, run_dc_hf, compute_dc_hf_gradient,
    FMOFragment, FMOOptions, FMOPairInteraction, FMOClusterResult, FMOResult, auto_partition_fragments, run_fmo, compute_fmo_gradient,
    HFVariantType, HFVariantHessianResult, evaluate_hf_variant_energy,
    compute_hf_variant_gradient, compute_hf_variant_analytical_gradient, compute_hf_variant_numerical_gradient, compute_hf_variant_hessian,
    compute_hf_variant_frequencies, run_hf_variant_hessian_analysis,
};
pub use gradient::{compute_rhf_gradient, compute_uhf_gradient, compute_gradient_from_density};
pub use hessian::{
    compute_hessian, compute_hessian_with_charge, compute_vibrational_frequencies,
    compute_hessian_finite_difference, compute_rhf_analytical_hessian, compute_uhf_analytical_hessian,
    HessianResult, project_hessian, compute_raw_vibrational_frequencies,
    compute_projected_vibrational_frequencies, compute_hessian_analysis,
};
pub use third_deriv::{
    try_compute_third_derivatives_general,
    compute_third_derivatives_general, compute_uhf_third_derivatives,
    compute_third_derivatives, compute_third_derivatives_with_charge,
    compute_third_derivatives_block, compute_third_directional_derivative,
    compute_third_derivatives_subspace, ThirdDerivativeTensor, ThirdDerivativeBlock,
};
pub use fourth_deriv::{
    try_compute_fourth_derivatives_general,
    compute_fourth_derivatives, compute_fourth_derivatives_with_charge,
    compute_uhf_fourth_derivatives, compute_fourth_derivatives_general,
    compute_fourth_derivatives_block, compute_fourth_directional_derivative,
    compute_fourth_derivatives_subspace, FourthDerivativeTensor, FourthDerivativeBlock,
};
pub use molden::{
    generate_molden_string_rhf, generate_molden_string_uhf,
    write_molden_file_rhf, write_molden_file_uhf,
};
