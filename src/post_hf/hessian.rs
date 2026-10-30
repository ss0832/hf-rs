// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Nuclear Gradient and Hessian Engine for Post-Hartree-Fock Methods.
//!
//! Evaluates nuclear gradients and Hessians for closed-shell and open-shell Post-HF methods:
//! - RHF-MP2: 100% exact analytical nuclear gradient via the Handy-Schaefer Z-vector method
//!   and orbital-relaxed density matrices (P_relaxed, W_relaxed, and correlated 2-PDM).
//! - Higher Post-HF (MP3, MP4, CC, CI, CASPT2, F12): high-precision nuclear gradients
//!   and force-response Hessians.

use rayon::prelude::*;
use crate::basis::{Atom, BasisSetMap, build_basis};
use crate::matrix::Matrix;
use crate::post_hf::mp2::{RHFMP2, UHFMP2};
use crate::post_hf::mpn::{RHFMP3, UHFMP3, RHFMP4};
use crate::post_hf::cc::{CCD, CCSD, compute_ccsd_t_correction};
use crate::post_hf::ci::{CIS, CISD, FCI, MRCISD};
use crate::post_hf::casscf::{CASSCF, CASSCFOptions};
use crate::post_hf::caspt2::{CASPT2, CASPT2Options};
use crate::post_hf::f12::{RHFMP2F12, RHFCCSDF12, CCSDF12Variant};
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::scf::{run_scf_general, run_uhf, HFResult, UHFResult};

/// Supported Post-Hartree-Fock methods for analytical gradients and Hessians.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostHFType {
    MP2,
    MP3,
    MP4,
    CCD,
    CCSD,
    CCSDT,
    CIS,
    CISD,
    FCI,
    CASSCF,
    CASPT2,
    MRCI,
    DLPNOMP2,
    DLPNOMCCSD,
    DLPNOMCCSDT,
    MP2F12,
    CCSDF12A,
    CCSDF12B,
    CCSDTF12A,
    CCSDTF12B,
    ACI,
    ASCI,
    HBCI,
    PCI,
    GASCI,
    GASSCF,
    DOCI,
    DMRG,
    DMRGSCF,
    AVAS,
    ASET,
    SRDSRGPT2,
    SRDSRGPT3,
    SRLDSRG2,
    SRGODE,
    DSRGMRPT2,
    ThreeDSRGMRPT2,
    DSRGMRPT3,
    MRLDSRG2,
    MRLDSRG2QC,
    MSDSRGPT2,
    XMSDSRGPT2,
    DWMSDSRGPT2,
}

impl PostHFType {
    pub fn parse(s:&str)->Result<Self,crate::error::HfError> {
        use std::str::FromStr;
        use crate::registry::MethodId;
        match MethodId::from_str(s)? {
            MethodId::Mp2=>Ok(Self::MP2),
            MethodId::Mp3=>Ok(Self::MP3),
            MethodId::Mp4=>Ok(Self::MP4),
            MethodId::Ccd=>Ok(Self::CCD),
            MethodId::Ccsd=>Ok(Self::CCSD),
            MethodId::CcsdT=>Ok(Self::CCSDT),
            MethodId::Cis=>Ok(Self::CIS),
            MethodId::Cisd=>Ok(Self::CISD),
            MethodId::Fci=>Ok(Self::FCI),
            MethodId::Casscf=>Ok(Self::CASSCF),
            MethodId::Caspt2=>Ok(Self::CASPT2),
            MethodId::Mrci=>Ok(Self::MRCI),
            MethodId::DlpnoMp2=>Ok(Self::DLPNOMP2),
            MethodId::DlpnoCcsd=>Ok(Self::DLPNOMCCSD),
            MethodId::DlpnoCcsdT1=>Ok(Self::DLPNOMCCSDT),
            MethodId::Mp2F12=>Ok(Self::MP2F12),
            MethodId::CcsdF12a=>Ok(Self::CCSDF12A),
            MethodId::CcsdF12b=>Ok(Self::CCSDF12B),
            MethodId::CcsdTF12a=>Ok(Self::CCSDTF12A),
            MethodId::CcsdTF12b=>Ok(Self::CCSDTF12B),
            MethodId::Aci=>Ok(Self::ACI),
            MethodId::Asci=>Ok(Self::ASCI),
            MethodId::Hbci=>Ok(Self::HBCI),
            MethodId::Pci=>Ok(Self::PCI),
            MethodId::Gasci=>Ok(Self::GASCI),
            MethodId::Gasscf=>Ok(Self::GASSCF),
            MethodId::Doci=>Ok(Self::DOCI),
            MethodId::Dmrg=>Ok(Self::DMRG),
            MethodId::DmrgScf=>Ok(Self::DMRGSCF),
            MethodId::Avas=>Ok(Self::AVAS),
            MethodId::Aset=>Ok(Self::ASET),
            MethodId::SrDsrgPt2=>Ok(Self::SRDSRGPT2),
            MethodId::SrDsrgPt3=>Ok(Self::SRDSRGPT3),
            MethodId::SrLdsrg2=>Ok(Self::SRLDSRG2),
            MethodId::SrGode=>Ok(Self::SRGODE),
            MethodId::DsrgMrPt2=>Ok(Self::DSRGMRPT2),
            MethodId::ThreeDsrgMrPt2=>Ok(Self::ThreeDSRGMRPT2),
            MethodId::DsrgMrPt3=>Ok(Self::DSRGMRPT3),
            MethodId::MrLdsrg2=>Ok(Self::MRLDSRG2),
            MethodId::MrLdsrg2Qc=>Ok(Self::MRLDSRG2QC),
            MethodId::MsDsrgPt2=>Ok(Self::MSDSRGPT2),
            MethodId::XmsDsrgPt2=>Ok(Self::XMSDSRGPT2),
            MethodId::DwmsDsrgPt2=>Ok(Self::DWMSDSRGPT2),
            method=>Err(crate::error::HfError::Unsupported{method:method.name().into(),detail:"legacy post-HF derivative route; use the typed registry".into()}),
        }
    }
}

fn evaluate_rhf_post_hf_corr(
    hf: &HFResult,
    method: PostHFType,
    atoms: Option<&[Atom]>,
    basis: Option<&[crate::basis::ContractedGaussian]>,
) -> Result<f64, String> {
    use crate::post_hf::mr_dsrg::*;
    match method {
        PostHFType::MP2 => RHFMP2::new(hf).compute().map(|r| r.correlation_energy),
        PostHFType::MP3 => RHFMP3::new(hf).compute().map(|r| r.correlation_energy()),
        PostHFType::MP4 => RHFMP4::new(hf).compute().map(|r| r.mp4_total_correlation_energy),
        PostHFType::CCD => CCD::new(hf).compute().map(|r| r.correlation_energy),
        PostHFType::CCSD => CCSD::new(hf).compute().map(|r| r.correlation_energy),
        PostHFType::CCSDT => {
            let ccsd = CCSD::new(hf).compute()?;
            compute_ccsd_t_correction(hf, &ccsd).map(|r| r.correlation_energy())
        }
        PostHFType::CIS => CIS::new(hf).compute().map(|r| r.excited_states[0].excitation_energy_hartree),
        PostHFType::CISD => CISD::new(hf).compute().map(|r| r.correlation_energy),
        PostHFType::FCI => FCI::new(hf).compute().map(|r| r.correlation_energy),
        PostHFType::CASSCF => CASSCF::new(hf, CASSCFOptions::new(2, 2)).compute().map(|r| r.correlation_energy),
        PostHFType::CASPT2 => CASPT2::from_hf(hf, CASPT2Options::new(2, 2)).map(|r| r.total_energy - hf.total_energy),
        PostHFType::MRCI => MRCISD::new(hf, 2, 2).compute().map(|r| r.correlation_energy),
        PostHFType::DLPNOMP2 | PostHFType::DLPNOMCCSD | PostHFType::DLPNOMCCSDT => Err("DLPNO nuclear derivatives are Unsupported because pair-space truncation is discontinuous".into()),
        PostHFType::MP2F12 => match (atoms, basis) {
            (Some(a), Some(b)) => RHFMP2F12::new(hf, a, b).compute().map(|r| r.correlation_energy()),
            _ => Err("MP2-F12 requires molecular geometry and basis set".to_string()),
        },
        PostHFType::CCSDF12A => match (atoms, basis) {
            (Some(a), Some(b)) => RHFCCSDF12::new(hf, a, b, CCSDF12Variant::F12a).compute().map(|r| r.correlation_energy()),
            _ => Err("CCSD-F12a requires molecular geometry and basis set".to_string()),
        },
        PostHFType::CCSDF12B => match (atoms, basis) {
            (Some(a), Some(b)) => RHFCCSDF12::new(hf, a, b, CCSDF12Variant::F12b).compute().map(|r| r.correlation_energy()),
            _ => Err("CCSD-F12b requires molecular geometry and basis set".to_string()),
        },
        PostHFType::CCSDTF12A => match (atoms, basis) {
            (Some(a), Some(b)) => RHFCCSDF12::new(hf, a, b, CCSDF12Variant::F12a).with_triples(true).compute().map(|r| r.correlation_energy()),
            _ => Err("CCSD(T)-F12a requires molecular geometry and basis set".to_string()),
        },
        PostHFType::CCSDTF12B => match (atoms, basis) {
            (Some(a), Some(b)) => RHFCCSDF12::new(hf, a, b, CCSDF12Variant::F12b).with_triples(true).compute().map(|r| r.correlation_energy()),
            _ => Err("CCSD(T)-F12b requires molecular geometry and basis set".to_string()),
        },
        PostHFType::ACI => ACI::new(hf, ACIOptions::default()).compute().map(|r| r.correlation_energy()),
        PostHFType::ASCI => ASCI::new(hf, ASCIOptions::default()).compute().map(|r| r.correlation_energy()),
        PostHFType::HBCI => HBCI::new(hf, HBCIOptions::default()).compute().map(|r| r.correlation_energy()),
        PostHFType::PCI => PCI::new(hf, PCIOptions::default()).compute().map(|r| r.correlation_energy()),
        PostHFType::GASCI => {
            let n_core = hf.n_occ.saturating_sub(1);
            let opts = GASOptions::new(n_core, vec![GASSubspace::new(1, 0, 2), GASSubspace::new(1, 0, 2)], false);
            GASCI::new(hf, opts).compute().map(|r| r.correlation_energy)
        }
        PostHFType::GASSCF => {
            let n_core = hf.n_occ.saturating_sub(1);
            let opts = GASOptions::new(n_core, vec![GASSubspace::new(1, 0, 2), GASSubspace::new(1, 0, 2)], true);
            GASCI::new_scf(hf, opts).compute().map(|r| r.correlation_energy)
        }
        PostHFType::DOCI => DOCI::new(hf).compute().map(|r| r.correlation_energy),
        PostHFType::DMRG => ActiveSpaceDMRG::new(hf, DMRGOptions::new(2, 2, 16)).compute().map(|r| r.correlation_energy),
        PostHFType::DMRGSCF => {
            let mut opts = DMRGOptions::new(2, 2, 16);
            opts.optimize_orbitals = true;
            ActiveSpaceDMRG::new(hf, opts).compute().map(|r| r.correlation_energy)
        }
        PostHFType::AVAS => match basis {
            Some(b) => AVAS::new(hf, b, AVASOptions::default()).compute().map(|r| r.correlation_energy),
            None => Err("AVAS requires basis set".to_string()),
        },
        PostHFType::ASET => match basis {
            Some(b) => ASET::new(hf, b, ASETOptions::default()).compute().map(|r| r.correlation_energy),
            None => Err("ASET requires basis set".to_string()),
        },
        PostHFType::SRDSRGPT2 => SRDSRGPT2::new(hf, 0.5).compute().map(|r| r.correlation_energy),
        PostHFType::SRDSRGPT3 => SRDSRGPT3::new(hf, 0.5).compute().map(|r| r.correlation_energy),
        PostHFType::SRLDSRG2 => SRLDSRG2::new(hf, 0.5).compute().map(|r| r.correlation_energy),
        PostHFType::SRGODE => SRGODE::new(hf, SRGGenerator::Wegner, 0.5, 30).compute().map(|r| r.correlation_energy),
        PostHFType::DSRGMRPT2 => DSRGMRPT2::new(hf, MRDSRGOptions::new(2, 2)).compute().map(|r| r.relaxed_total_energy - hf.total_energy),
        PostHFType::ThreeDSRGMRPT2 => ThreeDSRGMRPT2::new(hf, MRDSRGOptions::new(2, 2), 1e-10).compute().map(|r| r.relaxed_total_energy - hf.total_energy),
        PostHFType::DSRGMRPT3 => DSRGMRPT3::new(hf, MRDSRGOptions::new(2, 2)).compute().map(|r| r.relaxed_total_energy - hf.total_energy),
        PostHFType::MRLDSRG2 => MRLDSRG2::new(hf, MRDSRGOptions::new(2, 2)).compute().map(|r| r.relaxed_total_energy - hf.total_energy),
        PostHFType::MRLDSRG2QC => MRLDSRG2::new_qc(hf, MRDSRGOptions::new(2, 2)).compute().map(|r| r.relaxed_total_energy - hf.total_energy),
        PostHFType::MSDSRGPT2 => MultiStateDSRG::new(hf, MRDSRGOptions::new(2, 2), 2, MultiStateDSRGVariant::MS).compute().map(|r| r.state_energies[0] - hf.total_energy),
        PostHFType::XMSDSRGPT2 => MultiStateDSRG::new(hf, MRDSRGOptions::new(2, 2), 2, MultiStateDSRGVariant::XMS).compute().map(|r| r.state_energies[0] - hf.total_energy),
        PostHFType::DWMSDSRGPT2 => MultiStateDSRG::new(hf, MRDSRGOptions::new(2, 2), 2, MultiStateDSRGVariant::DWMS { zeta: 0.5 }).compute().map(|r| r.state_energies[0] - hf.total_energy),
    }
}

fn evaluate_uhf_post_hf_corr(uhf: &UHFResult, method: PostHFType) -> Result<f64, String> {
    match method {
        PostHFType::MP2 => UHFMP2::new(uhf).compute().map(|r| r.correlation_energy),
        PostHFType::MP3 => UHFMP3::new(uhf).compute().map(|r| r.correlation_energy()),
        PostHFType::CCSD => crate::post_hf::cc::UHFCCSD::new(uhf).compute().map(|r| r.correlation_energy),
        PostHFType::SRDSRGPT2 => crate::post_hf::mr_dsrg::UHFSRDSRGPT2::new(uhf, 0.5).compute().map(|r| r.correlation_energy),
        _ => Err(format!("UHF post-HF correlation for {method:?} is not implemented")),
    }
}

/// Computes the nuclear gradient of the total Post-HF energy: d E_{Post-HF} / d R_A.
pub fn compute_post_hf_gradient(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    method: PostHFType,
    max_iter: usize,
    tol: f64,
) -> Vec<[f64; 3]> {
    let basis_0 = build_basis(atoms, basis_map);

    // Reference RHF calculation
    let hf_0 = run_scf_general(
        atoms,
        &basis_0,
        charge,
        max_iter,
        tol,
        crate::diis::ConvAlgo::Adiissoscf,
        None,
    )
    .expect("Reference RHF failed in Post-HF gradient");

    // F12 derivatives are central differences of reconverged total energies.
    // In particular the CABS projector cannot be frozen in a CPHF tangent.
    if matches!(method, PostHFType::MP2F12 | PostHFType::CCSDF12A
        | PostHFType::CCSDF12B | PostHFType::CCSDTF12A | PostHFType::CCSDTF12B)
    {
        let settings = crate::post_hf::numerical::NumericalGradientOptions {
            scf_max_iter: max_iter,
            scf_tolerance: tol.min(1e-13),
            ..Default::default()
        };
        if method == PostHFType::MP2F12 {
            return RHFMP2F12::new(&hf_0, atoms, &basis_0)
                .numerical_gradient(&settings)
                .expect("MP2-F12 numerical gradient failed").gradient;
        }
        let variant = if matches!(method, PostHFType::CCSDF12A | PostHFType::CCSDTF12A) {
            CCSDF12Variant::F12a
        } else { CCSDF12Variant::F12b };
        return RHFCCSDF12::new(&hf_0, atoms, &basis_0, variant)
            .with_triples(matches!(method, PostHFType::CCSDTF12A | PostHFType::CCSDTF12B))
            .with_ccsd_options(crate::post_hf::cc::CCSDOptions {
                max_iterations: 200, energy_tol: 1e-13, residual_tol: 1e-12,
                ..Default::default()
            })
            .numerical_gradient(&settings)
            .expect("CCSD-F12 numerical gradient failed").gradient;
    }

    // Exact analytical gradient for RHF-MP2
    if method == PostHFType::MP2 {
        return crate::post_hf::mp2_gradient::compute_rhf_mp2_analytical_gradient(atoms, &basis_0, &hf_0);
    }

    // Exact analytical gradient for CASSCF
    if method == PostHFType::CASSCF {
        let opts = crate::post_hf::casscf::CASSCFOptions::new(2, 2);
        let casscf = crate::post_hf::casscf::CASSCF::new(&hf_0, opts).compute().expect("Reference CASSCF failed");
        return crate::post_hf::casscf::compute_casscf_analytical_gradient(atoms, &basis_0, &hf_0, &casscf);
    }

    // Exact analytical gradient for FCI
    if method == PostHFType::FCI {
        let fci = crate::post_hf::ci::fci::FCI::new(&hf_0).compute().expect("Reference FCI failed");
        return crate::post_hf::ci::fci::compute_fci_analytical_gradient(atoms, &basis_0, &hf_0, &fci);
    }

    // Exact analytical gradient for CISD
    if method == PostHFType::CISD {
        let cisd = crate::post_hf::ci::cisd::CISD::new(&hf_0).compute().expect("Reference CISD failed");
        return crate::post_hf::ci::cisd::compute_cisd_analytical_gradient(atoms, &basis_0, &hf_0, &cisd);
    }

    // Exact analytical gradient for CIS
    if method == PostHFType::CIS {
        let cis = crate::post_hf::ci::CIS::new(&hf_0).compute().expect("Reference CIS failed");
        return crate::post_hf::grad_common::compute_cis_analytical_gradient(atoms, &basis_0, &hf_0, &cis, 0);
    }

    // Exact analytical gradient for MR-CISD
    if method == PostHFType::MRCI {
        return crate::post_hf::grad_common::compute_mrci_analytical_gradient(atoms, &basis_0, &hf_0, 2, 2);
    }

    // Exact analytical gradient for DOCI
    if method == PostHFType::DOCI {
        let doci = crate::post_hf::mr_dsrg::DOCI::new(&hf_0).compute().expect("Reference DOCI failed");
        return crate::post_hf::mr_dsrg::compute_doci_analytical_gradient(atoms, &basis_0, &hf_0, &doci);
    }

    // Exact analytical gradient for GASCI / GASSCF
    if method == PostHFType::GASCI || method == PostHFType::GASSCF {
        let n_core = hf_0.n_occ.saturating_sub(1);
        let opts = crate::post_hf::mr_dsrg::GASOptions::new(
            n_core,
            vec![
                crate::post_hf::mr_dsrg::GASSubspace::new(1, 0, 2),
                crate::post_hf::mr_dsrg::GASSubspace::new(1, 0, 2),
            ],
            method == PostHFType::GASSCF,
        );
        let gas = crate::post_hf::mr_dsrg::GASCI::new(&hf_0, opts).compute().expect("Reference GAS failed");
        return crate::post_hf::mr_dsrg::compute_gas_analytical_gradient(atoms, &basis_0, &hf_0, &gas);
    }

    // Reconverged energy differences until P4 supplies finite-M/orbital response.
    if matches!(method, PostHFType::DMRG | PostHFType::DMRGSCF | PostHFType::CASPT2) {
        let settings = crate::post_hf::numerical::NumericalGradientOptions {
            scf_max_iter: max_iter, scf_tolerance: tol.min(1e-13), ..Default::default()
        };
        return crate::post_hf::numerical::compute_rhf_numerical_gradient(
            atoms, &basis_0, charge, hf_0.ao_space.as_ref().map_or(crate::ao::BasisKind::Cartesian, |space| space.kind), &settings,
            |hf, atoms_step, basis_step| evaluate_rhf_post_hf_corr(hf, method, Some(atoms_step), Some(basis_step))
                .map(|corr| hf.total_energy + corr)
                .map_err(|detail| crate::error::HfError::PostHf { method: format!("{method:?}"), detail }),
        ).expect("reconverged numerical verification gradient failed").gradient;
    }

    // Exact analytical CPHF response gradient at the reference geometry (zero SCF re-evaluations)
    crate::post_hf::grad_common::compute_rhf_cphf_analytical_corr_gradient(
        atoms,
        &basis_0,
        &hf_0,
        |hf_step, atoms_step, basis_step| {
            evaluate_rhf_post_hf_corr(hf_step, method, Some(atoms_step), Some(basis_step))
        },
    )
}

/// Computes the nuclear gradient of the total open-shell UHF Post-HF energy: d E_{UHF-Post-HF} / d R_A.
pub fn compute_uhf_post_hf_gradient(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    method: PostHFType,
    max_iter: usize,
    tol: f64,
) -> Vec<[f64; 3]> {
    let basis_0 = build_basis(atoms, basis_map);

    let uhf_0 = run_uhf(atoms, &basis_0, charge, multiplicity, is_broken_symmetry, max_iter, tol)
        .expect("Reference UHF failed in UHF Post-HF gradient");

    if method == PostHFType::MP2 {
        return crate::post_hf::mp2_gradient::compute_uhf_mp2_analytical_gradient(atoms, &basis_0, &uhf_0);
    }

    // Exact analytical CPUHF response gradient at the reference geometry (zero UHF re-evaluations)
    crate::post_hf::grad_common::compute_uhf_cphf_analytical_corr_gradient(
        atoms,
        &basis_0,
        &uhf_0,
        |uhf_step| evaluate_uhf_post_hf_corr(uhf_step, method),
    )
}

/// Computes the nuclear Hessian matrix (3*N x 3*N) for closed-shell Post-HF methods.
pub fn compute_post_hf_hessian(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    method: PostHFType,
    max_iter: usize,
    tol: f64,
) -> Matrix {
    if method == PostHFType::MP2 {
        return crate::post_hf::mp2_hessian::compute_rhf_mp2_analytical_hessian(atoms, basis_map, charge, max_iter, tol);
    }

    let n_atoms = atoms.len();
    let dim = 3 * n_atoms;
    let h = 1e-4; // Step size in Bohr

    let perts: Vec<(usize, usize, usize)> = (0..n_atoms)
        .flat_map(|a| (0..3).map(move |tau| (3 * a + tau, a, tau)))
        .collect();

    let grad_diffs: Vec<Vec<f64>> = perts
        .into_par_iter()
        .map(|(_col, a, tau)| {
            // Forward step
            let mut atoms_fwd = atoms.to_vec();
            atoms_fwd[a].position[tau] += h;
            let g_fwd = compute_post_hf_gradient(&atoms_fwd, basis_map, charge, method, max_iter, tol);

            // Backward step
            let mut atoms_bwd = atoms.to_vec();
            atoms_bwd[a].position[tau] -= h;
            let g_bwd = compute_post_hf_gradient(&atoms_bwd, basis_map, charge, method, max_iter, tol);

            let mut col_data = Vec::with_capacity(dim);
            for atom_idx in 0..n_atoms {
                for comp in 0..3 {
                    let d = (g_fwd[atom_idx][comp] - g_bwd[atom_idx][comp]) / (2.0 * h);
                    col_data.push(d);
                }
            }
            col_data
        })
        .collect();

    let mut hess = Matrix::zeros(dim, dim);
    for (col_idx, col) in grad_diffs.into_iter().enumerate() {
        for row_idx in 0..dim {
            hess[(row_idx, col_idx)] = col[row_idx];
        }
    }

    // Symmetrize Hessian: H = (H + H^T) / 2
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (hess[(i, j)] + hess[(j, i)]);
            hess[(i, j)] = avg;
            hess[(j, i)] = avg;
        }
    }

    hess
}

/// Computes the nuclear Hessian matrix (3*N x 3*N) for open-shell UHF Post-HF methods.
pub fn compute_uhf_post_hf_hessian(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    method: PostHFType,
    max_iter: usize,
    tol: f64,
) -> Matrix {
    let n_atoms = atoms.len();
    let dim = 3 * n_atoms;

    if method == PostHFType::MP2 {
        return crate::post_hf::mp2_hessian::compute_uhf_mp2_analytical_hessian(
            atoms,
            basis_map,
            charge,
            multiplicity,
            is_broken_symmetry,
            max_iter,
            tol,
        );
    }

    let h = 1e-4;

    let perts: Vec<(usize, usize, usize)> = (0..n_atoms)
        .flat_map(|a| (0..3).map(move |tau| (3 * a + tau, a, tau)))
        .collect();

    let grad_diffs: Vec<Vec<f64>> = perts
        .into_par_iter()
        .map(|(_col, a, tau)| {
            let mut atoms_fwd = atoms.to_vec();
            atoms_fwd[a].position[tau] += h;
            let g_fwd = compute_uhf_post_hf_gradient(
                &atoms_fwd,
                basis_map,
                charge,
                multiplicity,
                is_broken_symmetry,
                method,
                max_iter,
                tol,
            );

            let mut atoms_bwd = atoms.to_vec();
            atoms_bwd[a].position[tau] -= h;
            let g_bwd = compute_uhf_post_hf_gradient(
                &atoms_bwd,
                basis_map,
                charge,
                multiplicity,
                is_broken_symmetry,
                method,
                max_iter,
                tol,
            );

            let mut col_data = Vec::with_capacity(dim);
            for atom_idx in 0..n_atoms {
                for comp in 0..3 {
                    let d = (g_fwd[atom_idx][comp] - g_bwd[atom_idx][comp]) / (2.0 * h);
                    col_data.push(d);
                }
            }
            col_data
        })
        .collect();

    let mut hess = Matrix::zeros(dim, dim);
    for (col_idx, col) in grad_diffs.into_iter().enumerate() {
        for row_idx in 0..dim {
            hess[(row_idx, col_idx)] = col[row_idx];
        }
    }

    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (hess[(i, j)] + hess[(j, i)]);
            hess[(i, j)] = avg;
            hess[(j, i)] = avg;
        }
    }

    hess
}

/// Convenience wrapper for closed-shell RHF-MP2 gradient.
pub fn compute_mp2_gradient(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Vec<[f64; 3]> {
    compute_post_hf_gradient(atoms, basis_map, charge, PostHFType::MP2, max_iter, tol)
}

/// Convenience wrapper for open-shell UHF-MP2 gradient.
pub fn compute_uhf_mp2_gradient(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    max_iter: usize,
    tol: f64,
) -> Vec<[f64; 3]> {
    compute_uhf_post_hf_gradient(atoms, basis_map, charge, multiplicity, is_broken_symmetry, PostHFType::MP2, max_iter, tol)
}

/// Convenience wrapper for closed-shell RHF-MP2 Hessian.
pub fn compute_mp2_hessian(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    max_iter: usize,
    tol: f64,
) -> Matrix {
    compute_post_hf_hessian(atoms, basis_map, 0, PostHFType::MP2, max_iter, tol)
}

/// Convenience wrapper for closed-shell RHF-MP2 Hessian with molecular charge.
pub fn compute_mp2_hessian_with_charge(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Matrix {
    compute_post_hf_hessian(atoms, basis_map, charge, PostHFType::MP2, max_iter, tol)
}

/// Convenience wrapper for open-shell UHF-MP2 Hessian.
pub fn compute_uhf_mp2_hessian(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    max_iter: usize,
    tol: f64,
) -> Matrix {
    compute_uhf_post_hf_hessian(atoms, basis_map, charge, multiplicity, is_broken_symmetry, PostHFType::MP2, max_iter, tol)
}

/// Computes harmonic vibrational frequencies (cm^-1), normal modes, and Zero-Point Energy (ZPE)
/// from a mass-weighted post-HF Hessian with translation and rotation projected out.
pub fn compute_post_hf_vibrational_frequencies(
    atoms: &[Atom],
    hessian: &Matrix,
) -> (Vec<f64>, Vec<Vec<f64>>, f64) {
    let n_atoms = atoms.len();
    let dim = 3 * n_atoms;

    // Use Eckart projected mass-weighted Hessian
    let (_h_proj, d_proj) = crate::hessian::project_hessian(atoms, hessian);
    let (evals, evecs) = d_proj.eigh();
    let conv_factor = 5_140.487_143_4;

    let mut frequencies_cm = Vec::with_capacity(dim);
    let mut normal_modes = Vec::with_capacity(dim);
    let mut zpe_hartree = 0.0;

    for k in 0..dim {
        let lambda = evals[k];
        let freq = if lambda >= 0.0 {
            conv_factor * libm::sqrt(lambda)
        } else {
            -conv_factor * libm::sqrt(-lambda)
        };
        frequencies_cm.push(freq);

        if freq > 1e-4 {
            zpe_hartree += 0.5 * freq / 219_474.631_363;
        }

        let mode: Vec<f64> = (0..dim).map(|r| evecs[(r, k)]).collect();
        normal_modes.push(mode);
    }

    (frequencies_cm, normal_modes, zpe_hartree)
}

/// Backward-compatible alias for compute_post_hf_vibrational_frequencies.
pub fn compute_mp2_vibrational_frequencies(
    atoms: &[Atom],
    hessian: &Matrix,
) -> (Vec<f64>, Vec<Vec<f64>>, f64) {
    compute_post_hf_vibrational_frequencies(atoms, hessian)
}

/// Full post-HF Hessian analysis returning both raw unprojected and projected Hessians and frequencies.
pub fn compute_post_hf_hessian_analysis(
    atoms: &[Atom],
    hessian: &Matrix,
) -> crate::hessian::HessianResult {
    crate::hessian::compute_hessian_analysis(atoms, hessian)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h2_mp2_hessian_and_vibrations() {
        let basis_map = crate::basis::parse_gaussian94(crate::basis::STO_3G_GBS);
        let r = 1.4; // Bohr
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -r / 2.0]),
            Atom::new("H", [0.0, 0.0,  r / 2.0]),
        ];

        // 1. RHF-MP2 Hessian
        let hess_rhf = compute_mp2_hessian(&atoms, &basis_map, 50, 1e-12);
        assert_eq!(hess_rhf.rows, 6);
        assert_eq!(hess_rhf.cols, 6);

        // Symmetry: H = H^T
        for i in 0..6 {
            for j in 0..6 {
                assert!((hess_rhf[(i, j)] - hess_rhf[(j, i)]).abs() < 1e-10);
            }
        }

        // Translational invariance along z
        let k_zz = hess_rhf[(2, 2)];
        let k_zz_off = hess_rhf[(2, 5)];
        assert!(k_zz > 0.0);
        assert!((k_zz + k_zz_off).abs() < 1e-4);

        // Vibrational frequencies
        let (freqs_rhf, _, zpe_rhf) = compute_mp2_vibrational_frequencies(&atoms, &hess_rhf);
        let stretch_freq = freqs_rhf.last().unwrap();
        assert!(*stretch_freq > 4500.0 && *stretch_freq < 5100.0, "Stretch freq: {}", stretch_freq);
        assert!(zpe_rhf > 0.01 && zpe_rhf < 0.02);

        // 2. UHF-MP2 Hessian (open-shell engine on closed-shell singlet)
        let hess_uhf = compute_uhf_mp2_hessian(&atoms, &basis_map, 0, 1, false, 50, 1e-12);
        let diff = hess_rhf.rmsd(&hess_uhf);
        assert!(diff < 1e-5, "RHF-MP2 and UHF-MP2 Hessians must agree (RMSD = {})", diff);
    }
}
