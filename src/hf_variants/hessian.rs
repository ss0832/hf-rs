// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Derivative dispatch and numerical Hessians for Hartree-Fock variants.
//!
//! For variants with implemented analytic gradients, numerical differentiation computes the Hessian:
//!   H_{A\tau, B\eta} = \frac{\partial^2 E}{\partial R_{A\tau} \partial R_{B\eta}}
//!
//! The energy dispatcher recognizes the following variants; unsupported analytic gradients fail:
//! - ROHF (Restricted Open-Shell HF)
//! - GHF (Generalized HF with non-collinear spins)
//! - PUHF (Spin-Projected UHF)
//! - HFB (Hartree-Fock-Bogoliubov pairing theory)
//! - TCSCF / GVB (Two-Configuration SCF)
//! - NEO-HF (Nuclear-Electronic Orbital HF)
//! - QED-HF (Cavity Polaritonic HF)
//! - Relativistic HF (Scalar Relativistic X2C / DKH2 / ZORA)
//! - CD-HF (Cholesky Decomposition HF)
//! - DF-HF (Density Fitting / Resolution of Identity HF)
//! - Complex-HF (cHF with time-reversal breaking)
//! - LHF (Localized HF) and finite-potential-basis OEP exact exchange
//! - Positronic-HF (e+ e- multi-component HF)
//!
//! Includes mass-weighted vibrational frequency analysis, normal modes, and Zero-Point Energy (ZPE).

use rayon::prelude::*;
use crate::basis::{Atom, BasisSetMap, ContractedGaussian, build_basis};
use crate::masses::atomic_mass;
use crate::matrix::Matrix;
use crate::hf_variants::{
    run_ghf, run_puhf, run_hfb, run_tcscf, run_neo_hf, run_qed_hf,
    run_cd_hf, run_relativistic_hf, RelativisticModel,
};
use crate::hf_variants::complex_hf::{ComplexHFResult,ComplexHfOptions,ComplexHfReference,run_complex_hf_with_options};
use crate::hf_variants::ri_df_hf::run_df_hf;
use crate::hf_variants::lhf_oep::run_lhf_scf;
use crate::hf_variants::positronic_hf::run_positronic_hf;
use crate::rohf::run_rohf;

/// HF energy variants; analytic-gradient availability is checked by the dispatcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HFVariantType {
    ROHF,
    GHF,
    PUHF,
    HFB,
    TCSCF,
    NEOHF,
    QEDHF,
    RelativisticHF,
    Dkh2HF,
    ZoraHF,
    CDHF,
    DFHF,
    ComplexHF,
    ComplexUHF,
    ComplexGHF,
    LHF,
    OEP,
    PositronicHF,
    DCHF,
    FMO,
    FMO3,
}

impl HFVariantType {
    pub fn parse(input: &str) -> Result<Self, crate::error::HfError> {
        crate::registry::parse_hf_variant(input)
    }
}

/// Evaluates the total energy of a geometry for a specific HF variant.
pub fn evaluate_hf_variant_energy(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    variant: HFVariantType,
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Result<f64, String> {
    let basis = build_basis(atoms, basis_map);
    match variant {
        HFVariantType::ROHF => {
            // High-spin doublet or triplet depending on charge / electron parity
            let total_nuclear_charge: f64 = atoms.iter().map(|a| a.charge).sum();
            let total_electrons = (total_nuclear_charge.round() as i32) - charge;
            let n_unpaired = if total_electrons % 2 != 0 { 1 } else { 2 };
            let multiplicity = n_unpaired + 1;
            run_rohf(atoms, &basis, charge, multiplicity, max_iter, tol).map(|r| r.total_energy)
        }
        HFVariantType::GHF => run_ghf(atoms, &basis, charge, 0.0, max_iter, tol).map(|r| r.total_energy),
        HFVariantType::PUHF => run_puhf(atoms, &basis, charge, max_iter, tol).map(|r| r.projected_energy),
        HFVariantType::HFB => run_hfb(atoms, &basis, charge, 0.05, max_iter, tol).map(|r| r.total_hfb_energy),
        HFVariantType::TCSCF => {
            let total_nuclear_charge: f64 = atoms.iter().map(|a| a.charge).sum();
            let total_electrons = (total_nuclear_charge.round() as i32) - charge;
            let n_occ = (total_electrons.max(1) as usize).div_ceil(2);
            let homo = n_occ.saturating_sub(1);
            let lumo = n_occ.min(basis.len() - 1);
            run_tcscf(atoms, &basis, charge, homo, lumo, max_iter, tol).map(|r| r.tcscf_energy)
        }
        HFVariantType::NEOHF => {
            // First hydrogen atom as quantum proton, remaining as classical
            let mut quantum_h = Vec::new();
            let mut classical_atoms = Vec::new();
            let mut found_h = false;
            for a in atoms {
                if a.atomic_number == 1 && !found_h {
                    quantum_h.push(a.clone());
                    found_h = true;
                } else {
                    classical_atoms.push(a.clone());
                }
            }
            if quantum_h.is_empty() {
                return Err("NEO-HF requires at least one hydrogen atom".to_string());
            }
            let q_basis = build_basis(&quantum_h, basis_map);
            let n_quantum = 1;
            run_neo_hf(&classical_atoms, &basis, &q_basis, charge, n_quantum, max_iter, tol).map(|r| r.total_energy)
        }
        HFVariantType::QEDHF => {
            let lambda_coupling = [0.0, 0.0, 0.05];
            let cavity_freq = 0.1;
            run_qed_hf(atoms, &basis, charge, cavity_freq, lambda_coupling, max_iter, tol).map(|r| r.total_energy)
        }
        HFVariantType::RelativisticHF => {
            run_relativistic_hf(atoms, &basis, charge, RelativisticModel::X2C, max_iter, tol).map(|r| r.total_energy)
        }
        HFVariantType::Dkh2HF => run_relativistic_hf(atoms, &basis, charge, RelativisticModel::DKH2, max_iter, tol).map(|r| r.total_energy),
        HFVariantType::ZoraHF => run_relativistic_hf(atoms, &basis, charge, RelativisticModel::ZORA, max_iter, tol).map(|r| r.total_energy),
        HFVariantType::CDHF => run_cd_hf(atoms, &basis, charge, 1e-4, max_iter, tol).map(|r| r.total_energy),
        HFVariantType::DFHF => run_df_hf(atoms, &basis, charge, max_iter, tol).map(|r| r.total_energy),
        HFVariantType::ComplexHF | HFVariantType::ComplexUHF | HFVariantType::ComplexGHF =>
            complex_variant(atoms,&basis,variant,charge,max_iter,tol).map(|r|r.total_energy),
        HFVariantType::LHF => run_lhf_scf(atoms, &basis, charge, max_iter, tol).map(|r| r.total_energy),
        HFVariantType::OEP => crate::hf_variants::oep::run_oep_hf(atoms, &basis, charge,
            &crate::hf_variants::oep::OepOptions { max_iterations: max_iter,
                energy_tol: tol, ..Default::default() }).map(|r| r.total_energy),
        HFVariantType::PositronicHF => run_positronic_hf(atoms, &basis, charge, max_iter, tol).map(|r| r.total_energy),
        HFVariantType::DCHF => {
            let options = crate::hf_variants::dc_hf::DCOptions {
                max_iterations: max_iter,
                energy_threshold: tol,
                ..Default::default()
            };
            crate::hf_variants::dc_hf::run_dc_hf(atoms, &basis, charge, &options).map(|r| r.total_energy)
        }
        HFVariantType::FMO | HFVariantType::FMO3 => {
            let frags = crate::hf_variants::fmo::auto_partition_fragments(atoms, (atoms.len() / 2).max(1));
            let options = crate::hf_variants::fmo::FMOOptions {
                order: if variant == HFVariantType::FMO3 { 3 } else { 2 },
                max_iterations: max_iter,
                tolerance: tol,
                ..Default::default()
            };
            crate::hf_variants::fmo::run_fmo(atoms, &basis, &frags, &options).map(|r| r.total_energy)
        }
    }
}

/// Computes the exact analytical nuclear gradient for a given Hartree-Fock variant.
pub fn compute_hf_variant_analytical_gradient(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    variant: HFVariantType,
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<[f64; 3]>, String> {
    let basis = build_basis(atoms, basis_map);

    match variant {
        HFVariantType::QEDHF => {
            let result = run_qed_hf(atoms, &basis, charge, 0.1, [0., 0., 0.05], max_iter, tol)?;
            crate::hf_variants::qed_hf::compute_qed_hf_gradient(atoms, &basis, &result)
        }
        HFVariantType::RelativisticHF | HFVariantType::Dkh2HF => {
            let model = if variant == HFVariantType::Dkh2HF { RelativisticModel::DKH2 } else { RelativisticModel::X2C };
            let result = run_relativistic_hf(atoms, &basis, charge, model, max_iter, tol)?;
            crate::hf_variants::relativistic::compute_relativistic_hf_gradient(atoms, &basis, &result)
        }
        HFVariantType::ComplexHF | HFVariantType::ComplexUHF | HFVariantType::ComplexGHF => {
            let result = complex_variant(atoms,&basis,variant,charge,max_iter,tol)?;
            crate::hf_variants::complex_hf::compute_complex_hf_gradient(atoms, &basis, &result)
        }
        HFVariantType::TCSCF => {
            let hf = crate::scf::run_scf_general(atoms, &basis, charge, max_iter.max(100), tol, crate::diis::ConvAlgo::Adiissoscf, None)?;
            if hf.n_occ == 0 || hf.n_occ >= hf.mo_coefficients.cols {
                return Err("TCSCF needs an occupied HOMO and a virtual LUMO".into());
            }
            let result = crate::hf_variants::gvb::run_tcscf_from_reference(&hf, hf.n_occ - 1, hf.n_occ, max_iter, tol)?;
            Ok(crate::hf_variants::gvb::compute_tcscf_gradient(atoms, &basis, &hf, &result))
        }
        HFVariantType::GHF => {
            let result = run_ghf(atoms, &basis, charge, 0.0, max_iter, tol)?;
            crate::hf_variants::ghf::compute_ghf_gradient(atoms, &basis, &result)
        }
        HFVariantType::DCHF => {
            let options = crate::hf_variants::dc_hf::DCOptions {
                max_iterations: max_iter,
                energy_threshold: tol,
                ..Default::default()
            };
            let res = crate::hf_variants::dc_hf::run_dc_hf(atoms, &basis, charge, &options)?;
            crate::hf_variants::dc_hf::compute_dc_hf_gradient(atoms, &basis, &res)
        }
        HFVariantType::FMO | HFVariantType::FMO3 => {
            let frags = crate::hf_variants::fmo::auto_partition_fragments(atoms, (atoms.len() / 2).max(1));
            let options = crate::hf_variants::fmo::FMOOptions {
                order: if variant == HFVariantType::FMO3 { 3 } else { 2 },
                max_iterations: max_iter,
                tolerance: tol,
                ..Default::default()
            };
            let res = crate::hf_variants::fmo::run_fmo(atoms, &basis, &frags, &options)?;
            crate::hf_variants::fmo::compute_fmo_gradient(atoms, &basis, &frags, &res)
        }
        HFVariantType::ROHF => {
            let total_nuclear_charge: f64 = atoms.iter().map(|a| a.charge).sum();
            let total_electrons = (total_nuclear_charge.round() as i32) - charge;
            let n_unpaired = if total_electrons % 2 != 0 { 1 } else { 2 };
            let multiplicity = n_unpaired + 1;
            let res = crate::rohf::run_rohf(atoms, &basis, charge, multiplicity, max_iter, tol)?;
            Ok(crate::rohf::compute_rohf_gradient(atoms, &basis, &res))
        }
        _ => Err(format!("analytic nuclear gradients are not implemented for {variant:?}; use the explicitly numerical debug interface for finite differences")),
    }
}

/// Explicit central-difference debug gradient. This is not analytic orbital response.
pub fn compute_hf_variant_numerical_gradient(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    variant: HFVariantType,
    charge: i32,
    max_iter: usize,
    tol: f64,
    step: f64,
) -> Result<Vec<[f64; 3]>, String> {
    if !step.is_finite() || step <= 0. {
        return Err("numerical gradient requires a finite positive step in Bohr".into());
    }
    let mut gradient = vec![[0.; 3]; atoms.len()];
    for atom in 0..atoms.len() {
        for axis in 0..3 {
            let mut plus = atoms.to_vec();
            let mut minus = atoms.to_vec();
            plus[atom].position[axis] += step;
            minus[atom].position[axis] -= step;
            let ep = evaluate_hf_variant_energy(&plus, basis_map, variant, charge, max_iter, tol)?;
            let em = evaluate_hf_variant_energy(&minus, basis_map, variant, charge, max_iter, tol)?;
            gradient[atom][axis] = (ep - em) / (2. * step);
        }
    }
    Ok(gradient)
}

fn complex_variant(atoms: &[Atom],basis: &[ContractedGaussian],variant: HFVariantType,charge: i32,max_iter: usize,tol: f64) -> Result<ComplexHFResult,String> {
    let ne=atoms.iter().map(|a|a.charge).sum::<f64>().round() as i64-i64::from(charge);
    let reference=match variant {HFVariantType::ComplexUHF=>ComplexHfReference::Unrestricted,HFVariantType::ComplexGHF=>ComplexHfReference::Generalized,_=>ComplexHfReference::Restricted};
    let options=ComplexHfOptions {reference,multiplicity:if reference==ComplexHfReference::Restricted {1} else {(ne.rem_euclid(2)+1) as usize},
        magnetic_field:[0.,0.,0.02],max_iterations:max_iter,energy_tol:tol,residual_tol:tol.sqrt().min(1e-8),density_tol:tol.sqrt().min(1e-8),..Default::default()};
    run_complex_hf_with_options(atoms,basis,charge,&options)
}

/// Computes the nuclear gradient for a given HF variant via analytical gradient engine.
pub fn compute_hf_variant_gradient(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    variant: HFVariantType,
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<[f64; 3]>, String> {
    compute_hf_variant_analytical_gradient(atoms, basis_map, variant, charge, max_iter, tol)
}

/// Numerical central differences of the dispatched gradient. This is not
/// an implementation of analytic second-order orbital response.
pub fn compute_hf_variant_hessian(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    variant: HFVariantType,
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Result<Matrix, String> {
    let n_atoms = atoms.len();
    let dim = 3 * n_atoms;
    let h = 2e-4; // Step size in Bohr

    let perts: Vec<(usize, usize, usize)> = (0..n_atoms)
        .flat_map(|a| (0..3).map(move |tau| (3 * a + tau, a, tau)))
        .collect();

    // Parallel perturbation evaluation across coordinates
    let grad_diffs: Result<Vec<Vec<f64>>, String> = perts
        .into_par_iter()
        .map(|(_col, a, tau)| {
            let mut atoms_fwd = atoms.to_vec();
            atoms_fwd[a].position[tau] += h;
            let g_fwd = compute_hf_variant_gradient(&atoms_fwd, basis_map, variant, charge, max_iter, tol)?;

            let mut atoms_bwd = atoms.to_vec();
            atoms_bwd[a].position[tau] -= h;
            let g_bwd = compute_hf_variant_gradient(&atoms_bwd, basis_map, variant, charge, max_iter, tol)?;

            let mut col_data = Vec::with_capacity(dim);
            for atom_idx in 0..n_atoms {
                for comp in 0..3 {
                    let d = (g_fwd[atom_idx][comp] - g_bwd[atom_idx][comp]) / (2.0 * h);
                    col_data.push(d);
                }
            }
            Ok(col_data)
        })
        .collect();

    let grad_diffs = grad_diffs?;

    let mut hess = Matrix::zeros(dim, dim);
    for col in 0..dim {
        for row in 0..dim {
            hess[(row, col)] = grad_diffs[col][row];
        }
    }

    // Symmetrize H = 0.5 * (H + H^T)
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (hess[(i, j)] + hess[(j, i)]);
            hess[(i, j)] = avg;
            hess[(j, i)] = avg;
        }
    }

    Ok(hess)
}

/// Computes harmonic vibrational frequencies in cm^-1 from the Cartesian Hessian.
pub fn compute_hf_variant_frequencies(atoms: &[Atom], hessian: &Matrix) -> Vec<f64> {
    let n_atoms = atoms.len();
    let dim = 3 * n_atoms;

    let masses: Vec<f64> = atoms
        .iter()
        .flat_map(|a| {
            let m = atomic_mass(a.atomic_number);
            vec![m, m, m]
        })
        .collect();

    let mut mass_weighted = Matrix::zeros(dim, dim);
    for i in 0..dim {
        for j in 0..dim {
            mass_weighted[(i, j)] = hessian[(i, j)] / (masses[i] * masses[j]).sqrt();
        }
    }

    let (evals, _evecs) = mass_weighted.eigh();
    const AU_TO_CM1: f64 = 5140.48714376;

    let mut freqs = Vec::with_capacity(dim);
    for val in evals {
        if val >= 0.0 {
            freqs.push(val.sqrt() * AU_TO_CM1);
        } else {
            freqs.push(-(-val).sqrt() * AU_TO_CM1);
        }
    }
    freqs
}

/// Comprehensive result of a Hartree-Fock Variant Nuclear Hessian and Vibrational calculation.
#[derive(Clone, Debug)]
pub struct HFVariantHessianResult {
    pub variant: HFVariantType,
    pub total_energy: f64,
    pub gradient: Vec<[f64; 3]>,
    pub gradient_norm: f64,
    /// Raw unprojected Cartesian Hessian matrix (3N x 3N) in Hartree / Bohr^2.
    pub raw_hessian: Matrix,
    /// Projected Cartesian Hessian matrix (3N x 3N) with translation and rotation projected out.
    pub projected_hessian: Matrix,
    /// Alias to projected Cartesian Hessian matrix.
    pub hessian: Matrix,
    /// Raw vibrational frequencies in cm^-1 (unprojected).
    pub raw_frequencies_cm1: Vec<f64>,
    /// Harmonic vibrational frequencies in cm^-1 with translation and rotation projected out.
    pub projected_frequencies_cm1: Vec<f64>,
    /// Alias to projected vibrational frequencies.
    pub frequencies_cm1: Vec<f64>,
    pub zero_point_energy_hartree: f64,
    pub zero_point_energy_kcal: f64,
}

/// Runs full Nuclear Hessian and Harmonic Vibrational Analysis for a given HF variant.
pub fn run_hf_variant_hessian_analysis(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    variant: HFVariantType,
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Result<HFVariantHessianResult, String> {
    let total_energy = evaluate_hf_variant_energy(atoms, basis_map, variant, charge, max_iter, tol)?;
    let gradient = compute_hf_variant_gradient(atoms, basis_map, variant, charge, max_iter, tol)?;

    let mut grad_norm_sq = 0.0;
    for g_atom in &gradient {
        for comp in g_atom {
            grad_norm_sq += comp * comp;
        }
    }
    let gradient_norm = grad_norm_sq.sqrt();

    let raw_hessian = compute_hf_variant_hessian(atoms, basis_map, variant, charge, max_iter, tol)?;
    let analysis = crate::hessian::compute_hessian_analysis(atoms, &raw_hessian);
    let raw_frequencies_cm1 = analysis.raw_frequencies;
    let projected_frequencies_cm1 = analysis.projected_frequencies;
    let projected_hessian = analysis.projected_hessian;

    // ZPE = 0.5 * \sum_{\nu > 0} \nu over projected genuine vibrational modes
    const CM1_TO_HARTREE: f64 = 4.5563352529120e-6;
    const HARTREE_TO_KCAL: f64 = 627.509474;

    let mut zpe_cm1 = 0.0;
    for &freq in &projected_frequencies_cm1 {
        // Discard 5 or 6 translation/rotation zero modes
        if freq > 1e-4 {
            zpe_cm1 += 0.5 * freq;
        }
    }
    let zero_point_energy_hartree = zpe_cm1 * CM1_TO_HARTREE;
    let zero_point_energy_kcal = zero_point_energy_hartree * HARTREE_TO_KCAL;

    Ok(HFVariantHessianResult {
        variant,
        total_energy,
        gradient,
        gradient_norm,
        raw_hessian: raw_hessian.clone(),
        projected_hessian: projected_hessian.clone(),
        hessian: projected_hessian,
        raw_frequencies_cm1,
        projected_frequencies_cm1: projected_frequencies_cm1.clone(),
        frequencies_cm1: projected_frequencies_cm1,
        zero_point_energy_hartree,
        zero_point_energy_kcal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{parse_gaussian94, STO_3G_GBS};

    #[test]
    fn test_tcscf_hessian_and_frequencies_h2() {
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0,  0.7]),
        ];
        let basis_map = parse_gaussian94(STO_3G_GBS);

        let res = run_hf_variant_hessian_analysis(&atoms, &basis_map, HFVariantType::TCSCF, 0, 50, 1e-6).unwrap();
        assert_eq!(res.hessian.rows, 6);
        assert_eq!(res.hessian.cols, 6);
        assert_eq!(res.frequencies_cm1.len(), 6);
        // Harmonic stretch frequency of H2 should be ~ 4000 cm^-1
        let max_freq = res.frequencies_cm1.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_freq > 2000.0 && max_freq < 6000.0);
        assert!(res.zero_point_energy_hartree > 0.005);
    }

    #[test]
    fn test_rohf_hessian_and_frequencies_h2_plus() {
        // H2+ cation doublet
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -1.0]),
            Atom::new("H", [0.0, 0.0,  1.0]),
        ];
        let basis_map = parse_gaussian94(STO_3G_GBS);

        let res = run_hf_variant_hessian_analysis(&atoms, &basis_map, HFVariantType::ROHF, 1, 50, 1e-6).unwrap();
        assert_eq!(res.hessian.rows, 6);
        assert_eq!(res.frequencies_cm1.len(), 6);
        let max_freq = res.frequencies_cm1.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_freq > 1000.0 && max_freq < 5000.0);
    }
}
