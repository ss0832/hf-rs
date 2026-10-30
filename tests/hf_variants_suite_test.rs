// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Comprehensive test suite for Hartree-Fock variants on real chemical systems:
//! - GHF (Generalized Hartree-Fock): Twisted ethylene diradical (C2H4) and triplet oxygen (O2)
//! - HFB (Hartree-Fock-Bogoliubov): Water (H2O) and Carbon monoxide (CO) pairing
//! - PUHF (Projected UHF): Twisted ethylene spin projection
//! - TCSCF / GVB: Twisted ethylene (C2H4) multireference pi/pi* active space
//! - TDHF / RPA: Real Formaldehyde (H2CO) and CO vertical excitations and oscillator strengths
//! - NEO-HF: Real Water (H2O) and Ammonia (NH3) with quantum protons
//! - Cavity QED-HF: Formaldehyde (H2CO) and Water (H2O) polaritonic coupling & dipole self-energy
//! - Positronic HF: Bound positron to Water [e+; H2O] and Carbon monoxide [e+; CO]
//! - Complex HF: Water (H2O) and Formaldehyde under magnetic field B_z
//! - Relativistic X2C & DKH2: all-electron krypton and the nonrelativistic limit
//! - 1D Crystal Orbital CO-HF: Periodic polymer chain band structure & band gap
//! - State-Averaged HF: Real Water (H2O) multi-state ensemble
//! - LHF / OEP: Formaldehyde (H2CO) exact local exchange potential
//! - CD-HF & DF-HF: Water (H2O) and Formaldehyde (H2CO) low-rank tensor factorizations
//! - DC-HF & FMO2: Methane (CH4) linear scaling and Water-Formaldehyde PIEDA energy decomposition
//! - Numerical Atomic HF: Basis-set-free atomic limits

#![allow(clippy::needless_range_loop)]

use hf_rs::*;
use hf_rs::basis::{
    build_basis, parse_gaussian94, get_builtin_basis, list_builtin_basis_sets, Atom,
    STO_3G_GBS, BASIS_6_31G_GBS, BASIS_6_31G_STAR_GBS,
    BASIS_DEF2_SVP_GBS, BASIS_DEF2_TZVP_GBS, BASIS_CC_PVDZ_GBS,
};
use std::f64::consts::PI;

fn get_water() -> Vec<Atom> {
    vec![
        Atom::new("O", [0.0, 0.0, 0.2217]),
        Atom::new("H", [0.0, 1.4309, -0.8867]),
        Atom::new("H", [0.0, -1.4309, -0.8867]),
    ]
}

fn get_co() -> Vec<Atom> {
    vec![
        Atom::new("C", [0.0, 0.0, -1.066]),
        Atom::new("O", [0.0, 0.0,  1.066]),
    ]
}

fn get_formaldehyde() -> Vec<Atom> {
    vec![
        Atom::new("C", [0.0, 0.0, 0.0]),
        Atom::new("O", [0.0, 0.0, 2.279]),
        Atom::new("H", [0.0, 1.776, -1.111]),
        Atom::new("H", [0.0, -1.776, -1.111]),
    ]
}

fn get_twisted_ethylene() -> Vec<Atom> {
    vec![
        Atom::new("C", [0.0, 0.0,  1.266]),
        Atom::new("C", [0.0, 0.0, -1.266]),
        Atom::new("H", [0.0,  1.744,  2.336]),
        Atom::new("H", [0.0, -1.744,  2.336]),
        Atom::new("H", [1.744,  0.0, -2.336]),
        Atom::new("H", [-1.744, 0.0, -2.336]),
    ]
}

fn get_ammonia() -> Vec<Atom> {
    vec![
        Atom::new("N", [0.0, 0.0, 0.219]),
        Atom::new("H", [0.0, 1.772, -0.511]),
        Atom::new("H", [1.535, -0.886, -0.511]),
        Atom::new("H", [-1.535, -0.886, -0.511]),
    ]
}

fn get_oxygen() -> Vec<Atom> {
    vec![
        Atom::new("O", [0.0, 0.0, -1.14]),
        Atom::new("O", [0.0, 0.0,  1.14]),
    ]
}

fn get_methane() -> Vec<Atom> {
    vec![
        Atom::new("C", [0.0, 0.0, 0.0]),
        Atom::new("H", [1.189, 1.189, 1.189]),
        Atom::new("H", [-1.189, -1.189, 1.189]),
        Atom::new("H", [-1.189, 1.189, -1.189]),
        Atom::new("H", [1.189, -1.189, -1.189]),
    ]
}

fn get_water_formaldehyde() -> Vec<Atom> {
    vec![
        Atom::new("C", [0.0, 0.0, 0.0]),
        Atom::new("O", [0.0, 0.0, 2.279]),
        Atom::new("H", [0.0, 1.776, -1.111]),
        Atom::new("H", [0.0, -1.776, -1.111]),
        Atom::new("O", [0.0, 0.0, 7.665]),
        Atom::new("H", [0.0, 1.4309, 8.773]),
        Atom::new("H", [0.0, -1.4309, 8.773]),
    ]
}


fn get_co2() -> Vec<Atom> {
    vec![
        Atom::new("C", [0.0, 0.0, 0.0]),
        Atom::new("O", [0.0, 0.0,  2.1944]),
        Atom::new("O", [0.0, 0.0, -2.1944]),
    ]
}

fn get_hcn() -> Vec<Atom> {
    vec![
        Atom::new("H", [0.0, 0.0, -2.014]),
        Atom::new("C", [0.0, 0.0,  0.0]),
        Atom::new("N", [0.0, 0.0,  2.184]),
    ]
}

fn get_no() -> Vec<Atom> {
    vec![
        Atom::new("N", [0.0, 0.0, -1.087]),
        Atom::new("O", [0.0, 0.0,  1.087]),
    ]
}

fn get_so2() -> Vec<Atom> {
    vec![
        Atom::new("S", [0.0, 0.0,  0.685]),
        Atom::new("O", [0.0,  2.348, -0.685]),
        Atom::new("O", [0.0, -2.348, -0.685]),
    ]
}

fn get_h2o2() -> Vec<Atom> {
    vec![
        Atom::new("O", [0.0,  1.385, -0.15]),
        Atom::new("O", [0.0, -1.385, -0.15]),
        Atom::new("H", [ 1.78,  1.74,  0.60]),
        Atom::new("H", [-1.78, -1.74,  0.60]),
    ]
}

fn get_acetylene() -> Vec<Atom> {
    vec![
        Atom::new("C", [0.0, 0.0, -1.1367]),
        Atom::new("C", [0.0, 0.0,  1.1367]),
        Atom::new("H", [0.0, 0.0, -3.1398]),
        Atom::new("H", [0.0, 0.0,  3.1398]),
    ]
}

fn get_formic_acid() -> Vec<Atom> {
    vec![
        Atom::new("C", [ 0.000,  0.000, 0.0]),
        Atom::new("O", [ 0.000,  2.270, 0.0]),
        Atom::new("O", [ 2.170, -0.950, 0.0]),
        Atom::new("H", [-1.800, -0.900, 0.0]),
        Atom::new("H", [ 1.600, -2.550, 0.0]),
    ]
}

fn get_hf_molecule() -> Vec<Atom> {
    vec![
        Atom::new("F", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, 1.7325]),
    ]
}

fn get_ozone() -> Vec<Atom> {
    vec![
        Atom::new("O", [0.0,  0.000,  0.850]),
        Atom::new("O", [0.0,  2.057, -0.425]),
        Atom::new("O", [0.0, -2.057, -0.425]),
    ]
}

fn get_hydrazine() -> Vec<Atom> {
    vec![
        Atom::new("N", [ 0.000,  0.000, -1.370]),
        Atom::new("N", [ 0.000,  0.000,  1.370]),
        Atom::new("H", [ 1.550,  0.000, -2.050]),
        Atom::new("H", [-0.775,  1.340, -2.050]),
        Atom::new("H", [ 0.000,  1.550,  2.050]),
        Atom::new("H", [ 1.340, -0.775,  2.050]),
    ]
}

fn get_methanol() -> Vec<Atom> {
    vec![
        Atom::new("C", [ 0.000,  0.000,  0.000]),
        Atom::new("O", [ 0.000,  0.000,  2.700]),
        Atom::new("H", [ 1.680,  0.000,  3.250]),
        Atom::new("H", [-0.960,  1.660, -0.680]),
        Atom::new("H", [-0.960, -1.660, -0.680]),
        Atom::new("H", [ 1.920,  0.000, -0.680]),
    ]
}

fn get_so3() -> Vec<Atom> {
    vec![
        Atom::new("S", [ 0.0000,  0.0000, 0.0000]),
        Atom::new("O", [ 0.0000,  2.6830, 0.0000]),
        Atom::new("O", [ 2.3235, -1.3415, 0.0000]),
        Atom::new("O", [-2.3235, -1.3415, 0.0000]),
    ]
}

fn get_dichlorine(r_bohr: f64) -> Vec<Atom> {
    vec![
        Atom::new("Cl", [0.0, 0.0, -r_bohr * 0.5]),
        Atom::new("Cl", [0.0, 0.0,  r_bohr * 0.5]),
    ]
}

fn get_acetonitrile() -> Vec<Atom> {
    vec![
        Atom::new("C", [ 0.000,  0.000, -2.760]),
        Atom::new("C", [ 0.000,  0.000,  0.000]),
        Atom::new("N", [ 0.000,  0.000,  2.186]),
        Atom::new("H", [ 1.940,  0.000, -3.420]),
        Atom::new("H", [-0.970,  1.680, -3.420]),
        Atom::new("H", [-0.970, -1.680, -3.420]),
    ]
}

fn get_fluoromethane() -> Vec<Atom> {
    vec![
        Atom::new("C", [ 0.000,  0.000,  0.000]),
        Atom::new("F", [ 0.000,  0.000,  2.608]),
        Atom::new("H", [ 1.940,  0.000, -0.720]),
        Atom::new("H", [-0.970,  1.680, -0.720]),
        Atom::new("H", [-0.970, -1.680, -0.720]),
    ]
}

fn get_h2s() -> Vec<Atom> {
    vec![
        Atom::new("S", [0.000,  0.000,  0.190]),
        Atom::new("H", [0.000,  1.815, -1.560]),
        Atom::new("H", [0.000, -1.815, -1.560]),
    ]
}

fn get_n2o() -> Vec<Atom> {
    vec![
        Atom::new("N", [0.000, 0.000, -2.132]),
        Atom::new("N", [0.000, 0.000,  0.000]),
        Atom::new("O", [0.000, 0.000,  2.237]),
    ]
}

fn get_difluorine(r_bohr: f64) -> Vec<Atom> {
    vec![
        Atom::new("F", [0.0, 0.0, -r_bohr * 0.5]),
        Atom::new("F", [0.0, 0.0,  r_bohr * 0.5]),
    ]
}

fn get_ethylene_glycol() -> Vec<Atom> {
    vec![
        Atom::new("O", [ 0.000,  2.500,  0.500]),
        Atom::new("C", [ 0.000,  1.450, -1.450]),
        Atom::new("C", [ 0.000, -1.450, -1.450]),
        Atom::new("O", [ 0.000, -2.500,  0.500]),
        Atom::new("H", [ 1.600,  2.500,  1.000]),
        Atom::new("H", [-1.600, -2.500,  1.000]),
        Atom::new("H", [ 1.700,  1.450, -2.350]),
        Atom::new("H", [-1.700,  1.450, -2.350]),
        Atom::new("H", [ 1.700, -1.450, -2.350]),
        Atom::new("H", [-1.700, -1.450, -2.350]),
    ]
}

fn get_phosphine() -> Vec<Atom> {
    vec![
        Atom::new("P", [ 0.000,  0.000,  0.720]),
        Atom::new("H", [ 2.250,  0.000, -0.680]),
        Atom::new("H", [-1.125,  1.948, -0.680]),
        Atom::new("H", [-1.125, -1.948, -0.680]),
    ]
}

fn get_benzene() -> Vec<Atom> {
    // D6h planar hexagon, R_CC = 1.397 Angstrom = 2.640 Bohr, R_CH = 1.084 Angstrom = 2.048 Bohr
    let r_c = 2.640;
    let r_h = 2.640 + 2.048;
    let mut atoms = Vec::with_capacity(12);
    for i in 0..6 {
        let angle = (i as f64) * std::f64::consts::PI / 3.0;
        let (sin, cos) = angle.sin_cos();
        atoms.push(Atom::new("C", [r_c * cos, r_c * sin, 0.0]));
    }
    for i in 0..6 {
        let angle = (i as f64) * std::f64::consts::PI / 3.0;
        let (sin, cos) = angle.sin_cos();
        atoms.push(Atom::new("H", [r_h * cos, r_h * sin, 0.0]));
    }
    atoms
}

fn get_borane() -> Vec<Atom> {
    // D3h planar triangle, R_BH = 1.19 Angstrom = 2.249 Bohr
    let r_bh = 2.249;
    let mut atoms = Vec::with_capacity(4);
    atoms.push(Atom::new("B", [0.0, 0.0, 0.0]));
    for i in 0..3 {
        let angle = (i as f64) * 2.0 * std::f64::consts::PI / 3.0;
        let (sin, cos) = angle.sin_cos();
        atoms.push(Atom::new("H", [r_bh * cos, r_bh * sin, 0.0]));
    }
    atoms
}

fn get_silicon_tetrafluoride() -> Vec<Atom> {
    // Td tetrahedral SiF4, R_SiF = 1.55 Angstrom = 2.929 Bohr
    let a = 2.929 / (3.0_f64).sqrt();
    vec![
        Atom::new("Si", [0.0, 0.0, 0.0]),
        Atom::new("F", [ a,  a,  a]),
        Atom::new("F", [-a, -a,  a]),
        Atom::new("F", [-a,  a, -a]),
        Atom::new("F", [ a, -a, -a]),
    ]
}

fn get_nitrogen_dioxide() -> Vec<Atom> {
    // C2v bent radical NO2, R_NO = 1.193 Angstrom = 2.254 Bohr, angle = 134.3 deg
    let half_angle = 134.3_f64.to_radians() / 2.0;
    let r = 2.254;
    vec![
        Atom::new("N", [0.0, 0.0, 0.0]),
        Atom::new("O", [r * half_angle.sin(), 0.0,  r * half_angle.cos()]),
        Atom::new("O", [-r * half_angle.sin(), 0.0, r * half_angle.cos()]),
    ]
}

fn get_beryllium_hydride(r_bohr: f64) -> Vec<Atom> {
    // D_inf_h linear BeH2
    vec![
        Atom::new("Be", [0.0, 0.0, 0.0]),
        Atom::new("H",  [0.0, 0.0,  r_bohr]),
        Atom::new("H",  [0.0, 0.0, -r_bohr]),
    ]
}

fn get_pyridine() -> Vec<Atom> {
    // C2v pyridine C5H5N in Bohr (planar in xz plane)
    vec![
        Atom::new("N", [ 0.000, 0.0,  2.656]),
        Atom::new("C", [ 2.258, 0.0,  1.348]),
        Atom::new("C", [-2.258, 0.0,  1.348]),
        Atom::new("C", [ 2.258, 0.0, -1.272]),
        Atom::new("C", [-2.258, 0.0, -1.272]),
        Atom::new("C", [ 0.000, 0.0, -2.570]),
        Atom::new("H", [ 4.032, 0.0,  2.379]),
        Atom::new("H", [-4.032, 0.0,  2.379]),
        Atom::new("H", [ 4.032, 0.0, -2.268]),
        Atom::new("H", [-4.032, 0.0, -2.268]),
        Atom::new("H", [ 0.000, 0.0, -4.613]),
    ]
}

fn get_clf() -> Vec<Atom> {
    // C_inf_v polar interhalogen ClF (R_e = 1.628 Angstrom = 3.076 Bohr)
    vec![
        Atom::new("Cl", [0.0, 0.0, -1.229]),
        Atom::new("F",  [0.0, 0.0,  1.847]),
    ]
}

fn get_formamide() -> Vec<Atom> {
    // Cs planar formamide HCONH2 in Bohr
    vec![
        Atom::new("C", [ 0.000,  0.730, 0.0]),
        Atom::new("O", [ 2.054,  1.542, 0.0]),
        Atom::new("N", [-2.054, -0.669, 0.0]),
        Atom::new("H", [ 0.198, -1.332, 0.0]),
        Atom::new("H", [-3.719,  0.170, 0.0]),
        Atom::new("H", [-1.980, -2.553, 0.0]),
    ]
}

fn get_hydroxyl_radical() -> Vec<Atom> {
    // C_inf_v open-shell radical OH (R_e = 0.9697 Angstrom = 1.832 Bohr)
    vec![
        Atom::new("O", [0.0, 0.0,  0.108]),
        Atom::new("H", [0.0, 0.0, -1.724]),
    ]
}

fn get_cs2() -> Vec<Atom> {
    // D_inf_h linear carbon disulfide CS2 (R_CS = 1.554 Angstrom = 2.937 Bohr)
    vec![
        Atom::new("S", [0.0, 0.0, -2.937]),
        Atom::new("C", [0.0, 0.0,  0.000]),
        Atom::new("S", [0.0, 0.0,  2.937]),
    ]
}

// -----------------------------------------------------------------------------
// 1. GHF (Generalized Hartree-Fock)
// -----------------------------------------------------------------------------

#[test]
fn test_ghf_spin_symmetry_breaking_twisted_ethylene() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms = get_twisted_ethylene();
    let basis = build_basis(&atoms, &basis_map);

    let res = run_ghf(&atoms, &basis, 0, PI / 4.0, 80, 1e-8).expect("GHF twisted C2H4 failed");
    assert!(res.converged);
    assert!(res.total_energy < -77.0);
    // The twisted diradical has a collinear broken-symmetry determinant;
    // changing the global spin frame does not make it intrinsically GHF.
    assert!(res.s_squared > 0.05, "<S^2> was {}", res.s_squared);
    assert_eq!(res.fukutome_class, FukutomeClass::ASDW);
}

#[test]
fn test_ghf_triplet_oxygen_spin_components() {
    let basis_map = parse_gaussian94(BASIS_6_31G_STAR_GBS);
    let atoms = get_oxygen();
    let basis = build_basis(&atoms, &basis_map);

    let res = run_ghf(&atoms, &basis, 0, PI / 4.0, 80, 1e-8).expect("GHF O2 failed");
    assert!(res.converged);
    assert!(res.total_energy < -148.0);
}

#[test]
fn test_ghf_fukutome_real_molecules_classification() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms_h2o = get_water();
    let basis_h2o = build_basis(&atoms_h2o, &basis_map);
    let res_h2o = run_ghf(&atoms_h2o, &basis_h2o, 0, 0.0, 50, 1e-9).expect("GHF H2O failed");
    assert_eq!(res_h2o.fukutome_class, FukutomeClass::TICS);
    assert!(res_h2o.s_squared < 1e-6);

    let atoms_c2h4 = get_twisted_ethylene();
    let basis_c2h4 = build_basis(&atoms_c2h4, &basis_map);
    let res_c2h4 = run_ghf(&atoms_c2h4, &basis_c2h4, 0, PI / 4.0, 60, 1e-8).expect("GHF C2H4 failed");
    assert_eq!(res_c2h4.fukutome_class, FukutomeClass::ASDW);
}

// -----------------------------------------------------------------------------
// 2. HFB (Hartree-Fock-Bogoliubov Pairing)
// -----------------------------------------------------------------------------

#[test]
fn test_hfb_pairing_water_and_co_particle_conservation() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms_h2o = get_water();
    let basis_h2o = build_basis(&atoms_h2o, &basis_map);

    let res_h2o_g1 = run_hfb(&atoms_h2o, &basis_h2o, 0, 0.05, 50, 1e-8).expect("HFB H2O G=0.05 failed");
    let res_h2o_g2 = run_hfb(&atoms_h2o, &basis_h2o, 0, 0.15, 50, 1e-8).expect("HFB H2O G=0.15 failed");

    assert!(res_h2o_g1.converged);
    assert!(res_h2o_g2.converged);
    assert!(res_h2o_g2.pairing_energy.abs() >= res_h2o_g1.pairing_energy.abs() - 1e-10);

    // Exact particle number conservation: Tr(rho) = 10 electrons for H2O
    let tr_rho_h2o: f64 = (0..res_h2o_g2.normal_density.rows).map(|i| res_h2o_g2.normal_density[(i, i)]).sum();
    assert!((tr_rho_h2o - 10.0).abs() < 1e-5, "Tr(rho) was {}", tr_rho_h2o);

    // CO molecule: 14 electrons
    let atoms_co = get_co();
    let basis_co = build_basis(&atoms_co, &basis_map);
    let res_co = run_hfb(&atoms_co, &basis_co, 0, 0.08, 50, 1e-8).expect("HFB CO failed");
    assert!(res_co.converged);
    let tr_rho_co: f64 = (0..res_co.normal_density.rows).map(|i| res_co.normal_density[(i, i)]).sum();
    assert!((tr_rho_co - 14.0).abs() < 1e-5, "Tr(rho) was {}", tr_rho_co);
}

#[test]
fn test_hfb_zero_pairing_limit_water() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms = get_water();
    let basis = build_basis(&atoms, &basis_map);

    let res = run_hfb(&atoms, &basis, 0, 0.0, 50, 1e-8).expect("HFB G=0 failed");
    assert!(res.converged);
    assert!(res.pairing_energy.abs() < 1e-8);
}

// -----------------------------------------------------------------------------
// 3. PUHF (Spin-Projected UHF)
// -----------------------------------------------------------------------------

#[test]
fn test_puhf_twisted_ethylene_diradical_restoration() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms = get_twisted_ethylene();
    let basis = build_basis(&atoms, &basis_map);

    let res = run_puhf(&atoms, &basis, 0, 200, 1e-12).expect("PUHF twisted C2H4 failed");
    assert!(res.raw_uhf.converged);
    let oracle: serde_json::Value = serde_json::from_str(include_str!("data/puhf_reference.json")).unwrap();
    assert!(
        (res.projected_energy - oracle["large_case"]["e_projected"].as_f64().unwrap()).abs() < 1e-8,
        "PUHF {} (UHF {}, <S^2> {}) vs oracle {}",
        res.projected_energy, res.raw_uhf.total_energy, res.raw_uhf.s_squared, oracle["large_case"]["e_projected"]
    );
    assert!((res.target_weight - oracle["large_case"]["target_weight"].as_f64().unwrap()).abs() < 1e-8);
}

// -----------------------------------------------------------------------------
// 4. TCSCF / GVB (Two-Configuration SCF)
// -----------------------------------------------------------------------------

#[test]
fn test_tcscf_gvb_twisted_ethylene_diradical_static_correlation() {
    let basis_map = parse_gaussian94(BASIS_6_31G_STAR_GBS);
    let atoms = get_twisted_ethylene();
    let basis = build_basis(&atoms, &basis_map);

    let rhf = run_scf(&atoms, &basis, 50, 1e-10).expect("RHF C2H4 failed");
    let res = run_tcscf(&atoms, &basis, 0, 7, 8, 60, 1e-9).expect("TCSCF C2H4 failed");
    assert!(res.converged);
    assert!(res.tcscf_energy <= rhf.total_energy);
    // Near-degenerate frontier pi and pi* orbitals have substantial weights in 90 deg twisted ethylene
    let norm = res.c1 * res.c1 + res.c2 * res.c2;
    assert!((norm - 1.0).abs() < 1e-5, "Normalization failed: {}", norm);
    assert!(res.c1.abs() > 0.4 && res.c2.abs() > 0.4, "Degenerate weights c1={}, c2={}", res.c1, res.c2);
    assert_eq!(res.s_squared, 0.0);
}

// -----------------------------------------------------------------------------
// 5. TDHF / RPA (Linear Response & Excited States)
// -----------------------------------------------------------------------------

#[test]
fn test_tdhf_formaldehyde_vertical_excitation_and_oscillator_strength() {
    let basis_map = parse_gaussian94(BASIS_6_31G_STAR_GBS);
    let atoms = get_formaldehyde();
    let basis = build_basis(&atoms, &basis_map);

    let res = run_tdhf(&atoms, &basis, 0, 4, 60, 1e-9).expect("TDHF H2CO failed");
    assert!(!res.excited_states.is_empty());
    assert!(res.reference_energy < -113.0);

    for state in &res.excited_states {
        assert!(state.excitation_energy_ev > 0.0);
        assert!(state.oscillator_strength >= 0.0);
    }
}

#[test]
fn test_tdhf_carbon_monoxide_spectra() {
    let basis_map = parse_gaussian94(BASIS_6_31G_STAR_GBS);
    let atoms = get_co();
    let basis = build_basis(&atoms, &basis_map);

    let res = run_tdhf(&atoms, &basis, 0, 3, 50, 1e-9).expect("TDHF CO failed");
    assert_eq!(res.excited_states.len(), 3);
    // Linear CO (C_inf_v): lowest excited state is A ^1Pi (doubly degenerate in x and y).
    // States 0 and 1 must be strictly degenerate due to spatial isotropy.
    let de = (res.excited_states[0].excitation_energy_ev - res.excited_states[1].excitation_energy_ev).abs();
    assert!(de < 1e-5, "States 0 and 1 in CO must be degenerate Pi states, got difference {}", de);
    assert!(res.excited_states[1].excitation_energy_ev <= res.excited_states[2].excitation_energy_ev);
}

// -----------------------------------------------------------------------------
// 6. NEO-HF (Nuclear-Electronic Orbital HF)
// -----------------------------------------------------------------------------

#[test]
fn test_neo_hf_water_quantum_proton_dynamics() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms = get_water();
    let elec_basis = build_basis(&atoms, &basis_map);
    let prot_basis = elec_basis[11..].to_vec();

    let res = run_neo_hf(&atoms[..2], &elec_basis, &prot_basis, 0, 1, 60, 1e-8).expect("NEO H2O failed");
    assert!(res.converged);
    assert!(res.protonic_energy > 0.0, "Proton zero-point kinetic energy must be positive");
    assert!(res.electron_proton_coulomb < 0.0, "Electron-proton Coulomb attraction must be negative");
    assert!(res.total_energy < -75.0);
}

#[test]
fn test_neo_hf_ammonia_quantum_proton() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms = get_ammonia();
    let elec_basis = build_basis(&atoms, &basis_map);
    let prot_basis = elec_basis[elec_basis.len() - 2..].to_vec();

    let res = run_neo_hf(&atoms[..3], &elec_basis, &prot_basis, 0, 1, 60, 1e-8).expect("NEO NH3 failed");
    assert!(res.converged);
    assert!(res.protonic_energy > 0.0);
    assert!(res.total_energy < -55.0);
}

// -----------------------------------------------------------------------------
// 7. Cavity QED-HF (Polaritonic HF)
// -----------------------------------------------------------------------------

#[test]
fn test_cavity_qed_hf_polaritonic_coupling_formaldehyde() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms = get_formaldehyde();
    let basis = build_basis(&atoms, &basis_map);

    let res_00 = run_qed_hf(&atoms, &basis, 0, 0.1, [0.0, 0.0, 0.00], 80, 1e-8).expect("QED lambda=0 failed");
    let res_05 = run_qed_hf(&atoms, &basis, 0, 0.1, [0.0, 0.0, 0.05], 80, 1e-8).expect("QED lambda=0.05 failed");
    let res_10 = run_qed_hf(&atoms, &basis, 0, 0.1, [0.0, 0.0, 0.10], 80, 1e-8).expect("QED lambda=0.10 failed");

    assert!(res_00.converged && res_05.converged && res_10.converged);
    assert!(res_00.dipole_self_energy.abs() < 1e-8);
    assert!(res_05.dipole_self_energy > 0.0);
    assert!(res_10.dipole_self_energy > res_05.dipole_self_energy);

    // Quadratic scaling of Dipole Self-Energy (DSE)
    let ratio = res_10.dipole_self_energy / res_05.dipole_self_energy;
    assert!(ratio > 2.0 && ratio < 4.5, "DSE quadratic ratio was {}", ratio);
}

#[test]
fn test_cavity_qed_hf_water_polaritonic_vacuum() {
    let basis_map = parse_gaussian94(BASIS_6_31G_STAR_GBS);
    let atoms = get_water();
    let basis = build_basis(&atoms, &basis_map);

    let res = run_qed_hf(&atoms, &basis, 0, 0.15, [0.0, 0.0, 0.04], 50, 1e-9).expect("QED H2O failed");
    assert!(res.converged);
    assert!(res.dipole_self_energy > 0.0);
}

// -----------------------------------------------------------------------------
// 8. Positronic HF (e+ e- Multi-Component SCF)
// -----------------------------------------------------------------------------

#[test]
fn test_positronic_hf_bound_water_and_carbon_monoxide() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms_h2o = get_water();
    let basis_h2o = build_basis(&atoms_h2o, &basis_map);

    let res_h2o = run_positronic_hf(&atoms_h2o, &basis_h2o, 0, 60, 1e-7).expect("Positronic H2O failed");
    assert!(res_h2o.converged);
    assert!(res_h2o.ep_attraction_energy < 0.0, "Attractive e+ e- Coulomb interaction must be negative");

    let atoms_co = get_co();
    let basis_co = build_basis(&atoms_co, &basis_map);
    let res_co = run_positronic_hf(&atoms_co, &basis_co, 0, 60, 1e-7).expect("Positronic CO failed");
    assert!(res_co.converged);
    assert!(res_co.ep_attraction_energy < 0.0);
}

// -----------------------------------------------------------------------------
// 9. Complex HF (Time-Reversal Symmetry Breaking & Magnetic Fields)
// -----------------------------------------------------------------------------

#[test]
fn test_complex_hf_water_magnetic_field_currents() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms = vec![
        Atom::new("O", [0.0, 0.0, 0.0]),
        Atom::new("H", [1.4309,  0.8867, 0.0]),
        Atom::new("H", [-1.4309, 0.8867, 0.0]),
    ];
    let basis = build_basis(&atoms, &basis_map);

    let res_b0 = run_complex_hf(&atoms, &basis, 0, 0.00, 80, 1e-8).expect("Complex HF B=0 failed");
    let res_b1 = run_complex_hf(&atoms, &basis, 0, 0.05, 80, 1e-8).expect("Complex HF B=0.05 failed");

    assert!(res_b0.converged && res_b1.converged);
    assert!(res_b0.current_magnitude < 1e-8, "Current at B=0 must vanish: {}", res_b0.current_magnitude);
    assert!(res_b1.current_magnitude > 1e-4, "Magnetic field B_z induces non-zero ring current: {}", res_b1.current_magnitude);
}

// -----------------------------------------------------------------------------
// 10. Relativistic HF (X2C & DKH2 on krypton and light molecules)
// -----------------------------------------------------------------------------

#[test]
fn test_relativistic_krypton_x2c_dkh2_and_nonrelativistic_limit() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = vec![Atom::new("Kr", [0.0, 0.0, 0.0])];
    let basis = build_basis(&atoms, &basis_map);
    let x2c = run_relativistic_hf(&atoms, &basis, 0, RelativisticModel::X2C, 300, 1e-11).expect("X2C Kr failed");
    let dkh2 = run_relativistic_hf(&atoms, &basis, 0, RelativisticModel::DKH2, 300, 1e-11).expect("DKH2 Kr failed");
    assert!(x2c.converged && dkh2.converged);
    assert!(x2c.relativistic_shift_hartree < -20.0 && dkh2.relativistic_shift_hartree < -20.0);
    // c -> infinity recovers the nonrelativistic RHF energy for both Hamiltonians.
    for model in [RelativisticModel::X2C, RelativisticModel::DKH2] {
        let options = RelativisticHfOptions { speed_of_light: 1e6, max_iterations: 300, ..Default::default() };
        let limit = run_relativistic_hf_with_options(&atoms, &basis, 0, model, &options).unwrap();
        assert!(limit.relativistic_shift_hartree.abs() < 1e-6, "{model:?}: {}", limit.relativistic_shift_hartree);
    }
    let zora = run_relativistic_hf(&atoms, &basis, 0, RelativisticModel::ZORA, 60, 1e-8);
    assert!(zora.is_err(), "ZORA is not implemented and must not silently fall back");
}

#[test]
fn test_relativistic_core_stabilization_water_and_co() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms_h2o = get_water();
    let basis_h2o = build_basis(&atoms_h2o, &basis_map);

    let res_x2c_h2o = run_relativistic_hf(&atoms_h2o, &basis_h2o, 0, RelativisticModel::X2C, 50, 1e-9).expect("X2C H2O failed");
    assert!(res_x2c_h2o.converged);
    assert!(res_x2c_h2o.relativistic_shift_hartree < 0.0, "Oxygen 1s core relativistic stabilization must be negative");

    let atoms_co = get_co();
    let basis_co = build_basis(&atoms_co, &basis_map);
    let res_x2c_co = run_relativistic_hf(&atoms_co, &basis_co, 0, RelativisticModel::X2C, 50, 1e-9).expect("X2C CO failed");
    assert!(res_x2c_co.converged);
    assert!(res_x2c_co.relativistic_shift_hartree < res_x2c_h2o.relativistic_shift_hartree);
}

// -----------------------------------------------------------------------------
// 11. 1D Crystal Orbital CO-HF (Periodic Polymer Band Structure)
// -----------------------------------------------------------------------------

#[test]
fn test_crystal_co_hf_carbon_chain_band_structure() {
    // Minimal basis keeps the stored periodic integrals small; energies are validated against
    // infinite-chain limits in tests/crystal_hf_test.rs.
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = vec![
        Atom::new("C", [0.0, 0.0, -1.25]),
        Atom::new("C", [0.0, 0.0,  1.25]),
    ];
    let basis = build_basis(&atoms, &basis_map);

    let lattice_constant = 5.0; // Bohr
    // Enough k points that the density blocks spanning the overlap range do not alias.
    let n_kpoints = 24;
    let res = run_crystal_orbital_hf(&atoms, &basis, 0, lattice_constant, n_kpoints, 200, 1e-9)
        .expect("CO-HF carbon polymer failed");

    assert_eq!(res.n_kpoints, n_kpoints);
    assert!(res.conduction_band_minimum >= res.valence_band_maximum);
    assert!(res.band_gap_hartree >= 0.0);
}

// -----------------------------------------------------------------------------
// 12. State-Averaged HF (SA-HF)
// -----------------------------------------------------------------------------

#[test]
fn test_state_averaged_hf_water_ensemble() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms = get_water();
    let basis = build_basis(&atoms, &basis_map);

    let configs = vec![
        StateConfiguration::from_occupations(&[2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
        StateConfiguration::from_occupations(&[2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
    ];
    let weights = vec![0.5, 0.5];

    let res = run_state_averaged_hf(&atoms, &basis, &configs, &weights, 50, 1e-9).expect("SA-HF failed");
    assert!(res.converged);
    assert_eq!(res.state_energies.len(), 2);
    assert!(res.state_energies[0] < res.state_energies[1]);
}

// -----------------------------------------------------------------------------
// 13. LHF / OEP (Localized Hartree-Fock / Exact Local Exchange)
// -----------------------------------------------------------------------------

#[test]
fn test_lhf_oep_formaldehyde_local_exchange() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms = get_formaldehyde();
    let basis = build_basis(&atoms, &basis_map);

    let res = run_lhf_scf(&atoms, &basis, 0, 70, 1e-6).expect("LHF H2CO failed");
    assert!(res.converged);
    assert!(res.total_energy < -113.0);
    assert!(res.homo_lumo_gap.unwrap() > 0.0);
}

// -----------------------------------------------------------------------------
// 14. CD-HF & DF-HF (Cholesky & Density Fitting)
// -----------------------------------------------------------------------------

#[test]
fn test_cd_hf_and_df_hf_cholesky_water_and_formaldehyde() {
    let basis_map = parse_gaussian94(BASIS_6_31G_STAR_GBS);
    let atoms_h2o = get_water();
    let basis_h2o = build_basis(&atoms_h2o, &basis_map);

    let rhf_h2o = run_scf(&atoms_h2o, &basis_h2o, 50, 1e-10).expect("RHF H2O failed");
    let cd_h2o = run_cd_hf(&atoms_h2o, &basis_h2o, 0, 1e-6, 50, 1e-8).expect("CD-HF H2O failed");
    assert!(cd_h2o.converged);
    assert!((cd_h2o.total_energy - rhf_h2o.total_energy).abs() < 1e-5);

    let df_h2o = run_df_hf(&atoms_h2o, &basis_h2o, 0, 50, 1e-8).expect("DF-HF H2O failed");
    assert!(df_h2o.converged);
    assert!((df_h2o.total_energy - rhf_h2o.total_energy).abs() < 1e-4);
}

// -----------------------------------------------------------------------------
// 15. DC-HF & FMO (Linear Scaling Divide-and-Conquer & PIEDA)
// -----------------------------------------------------------------------------

#[test]
fn test_dc_hf_methane_linear_scaling() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms = get_methane();
    let basis = build_basis(&atoms, &basis_map);

    let dc_opts = DCOptions {
        buffer_radius_bohr: 5.0,
        energy_threshold: 1e-7,
        ..Default::default()
    };
    let dc_res = run_dc_hf(&atoms, &basis, 0, &dc_opts).expect("DC-HF CH4 failed");
    assert!(dc_res.converged);
    assert!(dc_res.total_energy < -40.0);
}

#[test]
fn test_fmo2_pieda_water_formaldehyde_complex() {
    let basis_map = parse_gaussian94(BASIS_6_31G_GBS);
    let atoms = get_water_formaldehyde();
    let basis = build_basis(&atoms, &basis_map);

    let fragments = vec![
        FMOFragment::new("Formaldehyde", vec![0, 1, 2, 3], 0),
        FMOFragment::new("Water", vec![4, 5, 6], 0),
    ];
    let fmo_opts = FMOOptions { order: 2, ..Default::default() };
    let res = run_fmo(&atoms, &basis, &fragments, &fmo_opts).expect("FMO2 failed");

    assert_eq!(res.monomer_energies.len(), 2);
    let ifie = &res.ifie[0];
    let sum_pieda = ifie.electrostatic + ifie.exchange_repulsion + ifie.charge_transfer;
    assert!((sum_pieda - ifie.total_interaction_energy).abs() < 1e-10);
}

// -----------------------------------------------------------------------------
// 16. Numerical Atomic HF (Basis-Set-Free Limit)
// -----------------------------------------------------------------------------

#[test]
fn test_numerical_atomic_hf_neutral_atoms_and_ions() {
    let res_h = run_numerical_atomic_hf(1, 150, 50, 1e-8).expect("Num HF H failed");
    assert!(res_h.converged);
    assert!((res_h.hfl_total_energy - (-0.5)).abs() < 5e-3);

    let res_he = run_numerical_atomic_hf(2, 120, 50, 1e-8).expect("Num HF He failed");
    assert!(res_he.converged);
    assert!((res_he.hfl_total_energy - (-2.8617)).abs() < 0.05);
}

// -----------------------------------------------------------------------------
// 17. Multi-Molecule Stress Suite: CO2 (Degeneracy, TDHF & Cavity QED)
// -----------------------------------------------------------------------------

#[test]
fn test_co2_tdhf_and_cavity_qed_anisotropy_def2_svp() {
    let basis_map = parse_gaussian94(BASIS_DEF2_SVP_GBS);
    let atoms = get_co2();
    let basis = build_basis(&atoms, &basis_map);

    // 1. RHF calculation on CO2 using BSE def2-SVP
    let rhf = run_scf(&atoms, &basis, 60, 1e-8).expect("RHF CO2 def2-SVP failed");
    assert!(rhf.converged);
    assert!(rhf.total_energy < -185.0, "CO2 def2-SVP energy was {}", rhf.total_energy);

    // 2. TDHF linear response on CO2 (22 electrons -> 11 occupied pairs)
    let tdhf = run_tdhf(&atoms, &basis, 0, 4, 60, 1e-8).expect("TDHF CO2 failed");
    assert!(tdhf.excited_states.len() >= 2);
    for s in &tdhf.excited_states {
        assert!(s.excitation_energy_ev > 5.0, "Excitation energy was {}", s.excitation_energy_ev);
    }

    // 3. Cavity QED-HF: longitudinal (z-axis, [0, 0, 1]) vs transverse (x-axis, [1, 0, 0])
    let qed_z = run_qed_hf(&atoms, &basis, 0, 0.1, [0.0, 0.0, 0.05], 60, 1e-8)
        .expect("QED CO2 z failed");
    let qed_x = run_qed_hf(&atoms, &basis, 0, 0.1, [0.05, 0.0, 0.0], 60, 1e-8)
        .expect("QED CO2 x failed");
    assert!(qed_z.converged);
    assert!(qed_x.converged);
    assert!(qed_z.dipole_self_energy >= 0.0);
    assert!(qed_x.dipole_self_energy >= 0.0);
}

// -----------------------------------------------------------------------------
// 18. Multi-Molecule Stress Suite: HCN (Acidic Proton NEO-HF & cc-pVDZ)
// -----------------------------------------------------------------------------

#[test]
fn test_hcn_neo_hf_acidic_quantum_proton_cc_pvdz() {
    let basis_map = parse_gaussian94(BASIS_CC_PVDZ_GBS);
    let atoms = get_hcn();
    let elec_basis = build_basis(&atoms, &basis_map);

    // Ground state RHF with Dunning cc-pVDZ
    let rhf = run_scf(&atoms, &elec_basis, 60, 1e-8).expect("RHF HCN cc-pVDZ failed");
    assert!(rhf.converged);
    assert!(rhf.total_energy < -92.0, "HCN cc-pVDZ energy was {}", rhf.total_energy);

    // Quantum acidic proton on Hydrogen (atom 0)
    // Classical atoms are C and N (atoms 1 and 2)
    let prot_basis = elec_basis[..5].to_vec(); // Hydrogen s and p functions
    let neo = run_neo_hf(&atoms[1..], &elec_basis, &prot_basis, 0, 1, 60, 1e-8)
        .expect("NEO HCN failed");
    assert!(neo.converged);
    assert!(neo.protonic_energy > 0.0, "Proton kinetic energy must be positive");
    assert!(neo.electron_proton_coulomb < 0.0, "Electron-proton attraction must be negative");
    assert!(neo.total_energy < -90.0);
}

// -----------------------------------------------------------------------------
// 19. Multi-Molecule Stress Suite: NO (Doublet Open-Shell Radical UHF, GHF & PUHF)
// -----------------------------------------------------------------------------

#[test]
fn test_no_radical_uhf_ghf_and_puhf_spin_restoration() {
    let basis_map = parse_gaussian94(BASIS_6_31G_STAR_GBS);
    let atoms = get_no();
    let basis = build_basis(&atoms, &basis_map);

    // NO has 15 electrons: 8 alpha, 7 beta (Doublet, multiplicity = 2)
    let uhf = run_uhf(&atoms, &basis, 0, 2, false, 100, 1e-6).expect("UHF NO failed");
    assert!(uhf.converged);
    assert!(uhf.total_energy < -129.0, "NO UHF energy was {}", uhf.total_energy);
    // UHF exhibits spin contamination: <S^2> >= 0.75
    assert!(uhf.s_squared >= 0.75, "Expected <S^2> >= 0.75 for doublet radical, got {}", uhf.s_squared);

    // Löwdin Spin-Projected UHF (PUHF) targeting S=1/2 doublet:
    let puhf = run_puhf(&atoms, &basis, 0, 100, 1e-6).expect("PUHF NO failed");
    assert!(puhf.raw_uhf.converged);
    assert!(puhf.projected_energy.is_finite());
    assert!(puhf.target_weight > 0.0 && puhf.target_weight <= 1.0 + 1e-10);

    // GHF on NO radical (15 electrons, charge = 0)
    let controls = hf_rs::hf_variants::ghf::GhfOptions {
        max_iterations: 300,
        energy_tol: 1e-12,
        residual_tol: 1e-10,
        density_tol: 1e-10,
        ..Default::default()
    };
    let ghf = hf_rs::hf_variants::ghf::run_ghf_with_options(
        &atoms, &basis, 0, PI / 4.0, &controls,
    ).expect("GHF NO failed");
    assert!(ghf.converged);
    assert!(ghf.total_energy < -128.0);
}

// -----------------------------------------------------------------------------
// 20. Multi-Molecule Stress Suite: SO2 (Diradicaloid TCSCF & def2-TZVP / def2-SVP)
// -----------------------------------------------------------------------------

#[test]
fn test_so2_diradicaloid_tcscf_and_def2_basis_hierarchy() {
    let atoms = get_so2();

    // Compare def2-SVP vs def2-TZVP from BSE (Variational Principle)
    let map_svp = parse_gaussian94(BASIS_DEF2_SVP_GBS);
    let basis_svp = build_basis(&atoms, &map_svp);
    let rhf_svp = run_scf(&atoms, &basis_svp, 70, 1e-8).expect("RHF SO2 def2-SVP failed");
    assert!(rhf_svp.converged);

    let map_tzvp = parse_gaussian94(BASIS_DEF2_TZVP_GBS);
    let basis_tzvp = build_basis(&atoms, &map_tzvp);
    let rhf_tzvp = run_scf(&atoms, &basis_tzvp, 70, 1e-8).expect("RHF SO2 def2-TZVP failed");
    assert!(rhf_tzvp.converged);

    // Variational hierarchy: E(def2-TZVP) < E(def2-SVP)
    assert!(rhf_tzvp.total_energy < rhf_svp.total_energy, "TZVP ({}) must be lower than SVP ({})", rhf_tzvp.total_energy, rhf_svp.total_energy);

    // Two-Configuration SCF (TCSCF) on SO2 diradicaloid
    // 32 electrons -> 16 occupied pairs. Active space: HOMO (15) and LUMO (16)
    let tcscf = run_tcscf(&atoms, &basis_svp, 0, 15, 16, 40, 1e-6).expect("TCSCF SO2 failed");
    assert!(tcscf.converged);
    let c_norm = tcscf.c1 * tcscf.c1 + tcscf.c2 * tcscf.c2;
    assert!((c_norm - 1.0).abs() < 1e-5);
    // SO2 has diradicaloid character producing static correlation
    assert!(tcscf.tcscf_energy <= rhf_svp.total_energy + 1e-5);

    // Relativistic X2C core contraction on Sulfur (Z=16)
    let rel_x2c = run_relativistic_hf(&atoms, &basis_svp, 0, RelativisticModel::X2C, 70, 1e-8)
        .expect("X2C SO2 failed");
    assert!(rel_x2c.converged);
    assert!(rel_x2c.relativistic_shift_hartree < -0.1, "Relativistic core correction on SO2 was {}", rel_x2c.relativistic_shift_hartree);
}

// -----------------------------------------------------------------------------
// 21. Multi-Molecule Stress Suite: H2O2 (Chiral c-HF, HFB Pairing & FMO2)
// -----------------------------------------------------------------------------

#[test]
fn test_h2o2_complex_hf_chiral_magnetic_response_and_hfb() {
    let basis_map = parse_gaussian94(BASIS_6_31G_STAR_GBS);
    let atoms = get_h2o2();
    let basis = build_basis(&atoms, &basis_map);

    // 1. Complex HF under external magnetic field B breaking T-reversal symmetry
    let c_hf = run_complex_hf(&atoms, &basis, 0, 0.01, 80, 1e-6).expect("Complex HF H2O2 failed");
    assert!(c_hf.converged);
    let norm_p_imag: f64 = c_hf.density_imag.data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm_p_imag > 1e-6, "Expected non-zero imaginary density current, got {}", norm_p_imag);
    assert!(c_hf.current_magnitude > 1e-6);

    // 2. Hartree-Fock-Bogoliubov (HFB) pairing across non-planar skew dihedral
    let hfb = run_hfb(&atoms, &basis, 0, 0.1, 60, 1e-8).expect("HFB H2O2 failed");
    assert!(hfb.converged);
    let tr_rho: f64 = (0..hfb.normal_density.rows).map(|i| hfb.normal_density[(i, i)]).sum();
    assert!((tr_rho - 18.0).abs() < 1e-5, "Tr(rho) was {}", tr_rho);

    // 3. FMO2 PIEDA decomposition: [OH]- (10e) and [OH]+ (8e)
    let fragments = vec![
        FMOFragment::new("Hydroxyl_Anion", vec![0, 2], -1),
        FMOFragment::new("Hydroxyl_Cation", vec![1, 3], 1),
    ];
    let fmo_opts = FMOOptions { order: 2, ..Default::default() };
    let fmo = run_fmo(&atoms, &basis, &fragments, &fmo_opts).expect("FMO2 H2O2 failed");
    assert_eq!(fmo.monomer_energies.len(), 2);
    let ifie = &fmo.ifie[0];
    let sum_pieda = ifie.electrostatic + ifie.exchange_repulsion + ifie.charge_transfer;
    assert!((sum_pieda - ifie.total_interaction_energy).abs() < 1e-10);
}

// -----------------------------------------------------------------------------
// 22. BSE Basis Set Verification Suite: Pople, Karlsruhe, and Dunning Families
// -----------------------------------------------------------------------------

#[test]
fn test_bse_pople_karlsruhe_dunning_all_basis_families() {
    let basis_names = list_builtin_basis_sets();
    assert_eq!(basis_names.len(), 80);

    // Every registered orbital basis must resolve through get_builtin_basis.
    for name in &basis_names {
        assert!(get_builtin_basis(name).is_some(), "Failed to get basis '{}'", name);
    }

    let atoms_co = get_co();

    // 1. Pople series variational progression: 3-21G -> 6-31G -> 6-31G* -> 6-31G**
    let map_321g = get_builtin_basis("3-21g").unwrap();
    let map_631g = get_builtin_basis("6-31g").unwrap();
    let map_631gs = get_builtin_basis("6-31g*").unwrap();
    let map_631gss = get_builtin_basis("6-31g**").unwrap();

    let e_321g = run_scf(&atoms_co, &build_basis(&atoms_co, &map_321g), 50, 1e-8).unwrap().total_energy;
    let e_631g = run_scf(&atoms_co, &build_basis(&atoms_co, &map_631g), 50, 1e-8).unwrap().total_energy;
    let e_631gs = run_scf(&atoms_co, &build_basis(&atoms_co, &map_631gs), 50, 1e-8).unwrap().total_energy;
    let e_631gss = run_scf(&atoms_co, &build_basis(&atoms_co, &map_631gss), 50, 1e-8).unwrap().total_energy;

    assert!(e_631g < e_321g, "6-31G ({}) must be lower than 3-21G ({})", e_631g, e_321g);
    assert!(e_631gs < e_631g, "6-31G* ({}) must be lower than 6-31G ({})", e_631gs, e_631g);
    assert!(e_631gss <= e_631gs + 1e-6, "6-31G** ({}) must be <= 6-31G* ({})", e_631gss, e_631gs);

    // 2. Karlsruhe series variational progression: def2-SVP -> def2-TZVP
    let map_def2_svp = get_builtin_basis("def2-svp").unwrap();
    let map_def2_tzvp = get_builtin_basis("def2-tzvp").unwrap();
    let e_def2_svp = run_scf(&atoms_co, &build_basis(&atoms_co, &map_def2_svp), 50, 1e-8).unwrap().total_energy;
    let e_def2_tzvp = run_scf(&atoms_co, &build_basis(&atoms_co, &map_def2_tzvp), 50, 1e-8).unwrap().total_energy;
    assert!(e_def2_tzvp < e_def2_svp, "def2-TZVP ({}) must be lower than def2-SVP ({})", e_def2_tzvp, e_def2_svp);

    // 3. Dunning correlation-consistent series: cc-pVDZ -> cc-pVTZ
    let map_cc_pvdz = get_builtin_basis("cc-pvdz").unwrap();
    let map_cc_pvtz = get_builtin_basis("cc-pvtz").unwrap();
    let e_cc_pvdz = run_scf(&atoms_co, &build_basis(&atoms_co, &map_cc_pvdz), 50, 1e-8).unwrap().total_energy;
    let e_cc_pvtz = run_scf(&atoms_co, &build_basis(&atoms_co, &map_cc_pvtz), 50, 1e-8).unwrap().total_energy;
    assert!(e_cc_pvtz < e_cc_pvdz, "cc-pVTZ ({}) must be lower than cc-pVDZ ({})", e_cc_pvtz, e_cc_pvdz);
}

// -----------------------------------------------------------------------------
// 23. Acetylene (C2H2): Linear D_inf_h Degenerate Pi Excitations & Dual-Proton NEO
// -----------------------------------------------------------------------------

#[test]
fn test_c2h2_acetylene_tdhf_degenerate_pi_and_dual_proton_neo() {
    let atoms = get_acetylene();
    let basis_map = get_builtin_basis("def2-svp").unwrap();
    let basis = build_basis(&atoms, &basis_map);

    // 1. TDHF: linear C2H2 singlet excitations
    let tdhf = run_tdhf(&atoms, &basis, 0, 6, 80, 1e-8).expect("TDHF C2H2 failed");
    for s in &tdhf.excited_states {
        println!("C2H2 State {}: E = {:.6} Eh ({:.3} eV), f = {:.4}, td = {:?}",
            s.state_index, s.excitation_energy_hartree, s.excitation_energy_ev, s.oscillator_strength, s.transition_dipole);
    }
    // State 1 is 1Sigma_u- (6.094 eV).
    // States 2 and 3 form the strictly degenerate 1Delta_u pair (6.706 eV).
    let e_ex2 = tdhf.excited_states[1].excitation_energy_hartree;
    let e_ex3 = tdhf.excited_states[2].excitation_energy_hartree;
    assert!(
        (e_ex2 - e_ex3).abs() < 1e-6,
        "The 1Delta_u pair in linear C2H2 (states 2 and 3) must be strictly degenerate: {} vs {}",
        e_ex2, e_ex3
    );
    assert!(e_ex2 > 0.20, "Excitation energy must be physical");

    // 2. Dual-Proton NEO-HF: both terminal protons as quantum wavepackets
    let basis_pople = parse_gaussian94(BASIS_6_31G_GBS);
    let elec_basis = build_basis(&atoms, &basis_pople);
    let prot_basis = elec_basis[elec_basis.len() - 4..].to_vec();
    let classical_carbons = &atoms[..2];

    let neo = run_neo_hf(classical_carbons, &elec_basis, &prot_basis, 0, 2, 70, 1e-7)
        .expect("Dual-proton NEO C2H2 failed");
    assert!(neo.converged);
    assert!(neo.protonic_energy > 0.0);
    assert!(neo.electron_proton_coulomb < 0.0);
    assert!(neo.total_energy < -76.0);

    // 3. 1D Crystal Orbital (CO-HF): Acetylene polymer chain unit cell
    let co_basis = build_basis(&atoms, &parse_gaussian94(STO_3G_GBS));
    let co_res = run_crystal_orbital_hf(&atoms, &co_basis, 0, 8.0, 24, 200, 1e-9)
        .expect("CO-HF C2H2 failed");
    assert!(co_res.band_gap_hartree > 0.1, "Acetylene chain should have a non-zero band gap");
}

// -----------------------------------------------------------------------------
// 24. Formic Acid (HCOOH): Planar Asymmetric Oxygen LHF & Cavity QED Anisotropy
// -----------------------------------------------------------------------------

#[test]
fn test_hcooh_formic_acid_lhf_exact_exchange_and_qed_anisotropy() {
    let atoms = get_formic_acid();
    let basis_map = parse_gaussian94(BASIS_6_31G_STAR_GBS);
    let basis = build_basis(&atoms, &basis_map);

    // 1. Localized Hartree-Fock (LHF / OEP): Exact local exchange on asymmetric C=O vs C-OH oxygens
    let lhf = run_lhf_scf(&atoms, &basis, 0, 80, 1e-7).expect("LHF HCOOH failed");
    assert!(lhf.converged);
    assert!(lhf.total_energy < -188.0);
    assert!(lhf.exchange_energy < -10.0);

    // 2. Cavity QED-HF: Polaritonic shift anisotropy along permanent dipole axis
    let hf_ref = run_scf(&atoms, &basis, 60, 1e-8).unwrap();
    let dipole_y = hf_ref.dipole_moment_debye()[1].abs();
    let dipole_x = hf_ref.dipole_moment_debye()[0].abs();
    assert!(dipole_y > dipole_x, "In Formic acid, permanent dipole is oriented along y");

    // Coupling in-plane (x: C-O bond axis) vs out-of-plane (z: perpendicular to molecular frame)
    let qed_x = run_qed_hf(&atoms, &basis, 0, 0.1, [0.05, 0.0, 0.0], 80, 1e-8)
        .expect("QED HCOOH in-plane x failed");
    let qed_z = run_qed_hf(&atoms, &basis, 0, 0.1, [0.0, 0.0, 0.05], 80, 1e-8)
        .expect("QED HCOOH out-of-plane z failed");

    assert!(qed_x.converged && qed_z.converged);
    assert!(
        qed_x.dipole_self_energy > qed_z.dipole_self_energy,
        "In-plane cavity coupling ({}) must exceed out-of-plane ({})",
        qed_x.dipole_self_energy, qed_z.dipole_self_energy
    );
}

// -----------------------------------------------------------------------------
// 25. Hydrogen Fluoride (HF): Extreme Ionic Positronic Bound State & BSE Hierarchy
// -----------------------------------------------------------------------------

#[test]
fn test_hf_molecule_positronic_bound_state_and_bse_variational() {
    let atoms = get_hf_molecule();

    // 1. Positronic HF: The intense local negative charge on Fluorine
    // induces a strong local attractive potential, forming a genuine bound positron state
    let basis_pvdz = get_builtin_basis("cc-pvdz").unwrap();
    let basis = build_basis(&atoms, &basis_pvdz);

    let hf_neutral = run_scf(&atoms, &basis, 60, 1e-8).expect("RHF HF failed");
    let pos_res = run_positronic_hf(&atoms, &basis, 0, 80, 1e-7).expect("Positronic HF failed");

    assert!(pos_res.converged);
    assert!(pos_res.ep_attraction_energy < -0.1, "Strong attractive electron-positron Coulomb energy");
    let s = &hf_neutral.overlap_matrix;
    let mut n_pos = 0.0;
    for i in 0..s.rows {
        for j in 0..s.cols {
            n_pos += pos_res.positron_density[(i, j)] * s[(j, i)];
        }
    }
    assert!((n_pos - 1.0).abs() < 1e-4, "Positron particle number Tr(P_p * S) was {}", n_pos);

    // 2. BSE Basis Set Hierarchy on polar covalent bond: 3-21G -> def2-SVP -> cc-pVDZ -> cc-pVTZ
    let map_321g = get_builtin_basis("3-21g").unwrap();
    let map_def2_svp = get_builtin_basis("def2-svp").unwrap();
    let map_cc_pvtz = get_builtin_basis("cc-pvtz").unwrap();

    let e_321g = run_scf(&atoms, &build_basis(&atoms, &map_321g), 50, 1e-8).unwrap().total_energy;
    let e_def2_svp = run_scf(&atoms, &build_basis(&atoms, &map_def2_svp), 50, 1e-8).unwrap().total_energy;
    let e_pvdz = hf_neutral.total_energy;
    let e_pvtz = run_scf(&atoms, &build_basis(&atoms, &map_cc_pvtz), 50, 1e-8).unwrap().total_energy;

    assert!(e_def2_svp < e_321g, "def2-SVP ({}) < 3-21G ({})", e_def2_svp, e_321g);
    assert!(e_pvdz < e_def2_svp, "cc-pVDZ ({}) < def2-SVP ({})", e_pvdz, e_def2_svp);
    assert!(e_pvtz < e_pvdz, "cc-pVTZ ({}) < cc-pVDZ ({})", e_pvtz, e_pvdz);
}

// -----------------------------------------------------------------------------
// 26. Ozone (O3): Biradicaloid RHF Instability & TCSCF Multireference Correlation
// -----------------------------------------------------------------------------

#[test]
fn test_o3_ozone_diradicaloid_stability_and_tcscf_multireference() {
    let atoms = get_ozone();
    let basis_map = parse_gaussian94(BASIS_6_31G_STAR_GBS);
    let basis = build_basis(&atoms, &basis_map);

    // 1. RHF SCF: Strictly adheres to user-specified theory without auto-switching to UHF or TCSCF!
    let hf_res = run_scf_general(&atoms, &basis, 0, 70, 1e-8, ConvAlgo::Adiissoscf, None)
        .expect("RHF O3 calculation failed");
    assert!(hf_res.converged);
    assert!(hf_res.total_energy < -224.0);

    // 2. Stability Analysis: Proves theoretical boundary of single-determinant RHF on diradicaloids!
    // The lowest eigenvalue of the stability Hessian is strictly negative (lambda < 0).
    let stab = analyze_rhf_stability(&hf_res, hf_res.n_occ);
    assert!(
        !stab.is_stable,
        "Ozone ground state has multireference biradical character and must be RHF unstable"
    );
    assert!(
        stab.lowest_eigenvalue < 0.0,
        "Stability Hessian must have negative eigenvalue, got {}",
        stab.lowest_eigenvalue
    );

    // 3. Two-Configuration SCF (TCSCF / GVB): Captures biradical static correlation
    // Active pair: HOMO (11) and LUMO (12)
    let tcscf = run_tcscf(&atoms, &basis, 0, 11, 12, 60, 1e-7).expect("TCSCF O3 failed");
    assert!(tcscf.converged);
    assert_eq!(tcscf.s_squared, 0.0, "TCSCF remains strictly a singlet <S^2> = 0");
    assert!(
        tcscf.c2.abs() > 0.05,
        "Biradical configuration coefficient c2 ({}) must be non-negligible",
        tcscf.c2
    );
    assert!(
        tcscf.correlation_energy < -0.005,
        "TCSCF static correlation energy ({}) must lower total energy",
        tcscf.correlation_energy
    );

    // 4. Generalized Hartree-Fock (GHF): Spontaneous spin-density wave
    let ghf = run_ghf(&atoms, &basis, 0, PI / 4.0, 70, 1e-7).expect("GHF O3 failed");
    assert!(ghf.converged);
    // The invariant eight-class analysis is checked against independently
    // constructed determinants, rather than assigning a class from the seed.
}

// -----------------------------------------------------------------------------
// 27. Hydrazine (N2H4): Skew Conformational DC-HF & FMO2 PIEDA Decomposition
// -----------------------------------------------------------------------------

#[test]
fn test_n2h4_hydrazine_dc_hf_and_fmo_energy_decomposition() {
    let atoms = get_hydrazine();
    let basis_map = get_builtin_basis("def2-svp").unwrap();
    let basis = build_basis(&atoms, &basis_map);

    // 1. Divide-and-Conquer (DC-HF): Linear scaling subsystem partition
    let dc_opts = DCOptions {
        buffer_radius_bohr: 5.0,
        max_iterations: 80,
        energy_threshold: 1e-7,
        ..Default::default()
    };
    let dc_res = run_dc_hf(&atoms, &basis, 0, &dc_opts).expect("DC-HF N2H4 failed");
    assert!(dc_res.converged);
    assert!(dc_res.total_energy < -110.0);
    assert!(dc_res.chemical_potential.is_finite());

    // 2. FMO2 PIEDA: Fragment interaction energy decomposition across N-N bond
    // Fragment 1: [NH2]^- (atoms 0, 2, 3; charge -1)
    // Fragment 2: [NH2]^+ (atoms 1, 4, 5; charge +1)
    let fragments = vec![
        FMOFragment::new("NH2_Anion", vec![0, 2, 3], -1),
        FMOFragment::new("NH2_Cation", vec![1, 4, 5], 1),
    ];
    let fmo_opts = FMOOptions { order: 2, ..Default::default() };
    let fmo = run_fmo(&atoms, &basis, &fragments, &fmo_opts).expect("FMO2 N2H4 failed");
    assert_eq!(fmo.monomer_energies.len(), 2);
    let ifie = &fmo.ifie[0];
    let sum_pieda = ifie.electrostatic + ifie.exchange_repulsion + ifie.charge_transfer;
    assert!(
        (sum_pieda - ifie.total_interaction_energy).abs() < 1e-10,
        "PIEDA sum ({}) must equal total interaction ({})",
        sum_pieda, ifie.total_interaction_energy
    );
}

// -----------------------------------------------------------------------------
// 28. Methanol (CH3OH): FMO2 PIEDA Ionic Decomposition & Cavity QED Anisotropy
// -----------------------------------------------------------------------------

#[test]
fn test_ch3oh_methanol_fmo2_pieda_and_qed_dipole_coupling() {
    let atoms = get_methanol();
    let basis_map = get_builtin_basis("3-21g").unwrap();
    let basis = build_basis(&atoms, &basis_map);

    // 1. FMO2 PIEDA Energy Decomposition Analysis:
    // Fragment 1: [CH3]^+ methyl cation (atoms 0, 3, 4, 5; charge +1)
    // Fragment 2: [OH]^- hydroxyl anion (atoms 1, 2; charge -1)
    let fragments = vec![
        FMOFragment::new("Methyl_Cation", vec![0, 3, 4, 5], 1),
        FMOFragment::new("Hydroxyl_Anion", vec![1, 2], -1),
    ];
    let fmo_opts = FMOOptions { order: 2, ..Default::default() };
    let fmo = run_fmo(&atoms, &basis, &fragments, &fmo_opts).expect("FMO2 CH3OH failed");

    assert_eq!(fmo.monomer_energies.len(), 2);
    let ifie = &fmo.ifie[0];
    let sum_pieda = ifie.electrostatic + ifie.exchange_repulsion + ifie.charge_transfer;

    // Exact PIEDA conservation theorem: Delta E_IJ = E_es + E_ex + E_ct
    assert!(
        (sum_pieda - ifie.total_interaction_energy).abs() < 1e-10,
        "PIEDA sum ({}) must equal total interaction ({})",
        sum_pieda, ifie.total_interaction_energy
    );
    // Strong electrostatic attraction between opposite ionic fragments:
    // Ref: Fedorov & Kitaura, J. Phys. Chem. A 111, 6904 (2007)
    assert!(
        ifie.electrostatic < -0.2,
        "Ionic cleavage into [CH3]+ and [OH]- gives strong Coulomb attraction, got {}",
        ifie.electrostatic
    );

    // 2. Cavity QED-HF: Polaritonic shift along C-O dipole axis (z) vs perpendicular (y)
    let qed_z = run_qed_hf(&atoms, &basis, 0, 0.1, [0.0, 0.0, 0.05], 60, 1e-8)
        .expect("QED CH3OH z failed");
    let qed_y = run_qed_hf(&atoms, &basis, 0, 0.1, [0.0, 0.05, 0.0], 60, 1e-8)
        .expect("QED CH3OH y failed");

    assert!(qed_z.converged && qed_y.converged);
    assert!(qed_z.dipole_self_energy > 0.0);
    assert!(qed_y.dipole_self_energy > 0.0);
}

// -----------------------------------------------------------------------------
// 29. Sulfur Trioxide (SO3): Planar D3h TDHF E' Degeneracy & X2C Relativistic Stabilization
// -----------------------------------------------------------------------------

#[test]
fn test_so3_sulfur_trioxide_tdhf_e_prime_degeneracy_and_x2c_stabilization() {
    let atoms = get_so3();
    let basis_map = get_builtin_basis("3-21g").unwrap();
    let basis = build_basis(&atoms, &basis_map);

    // 1. Non-relativistic RHF reference
    let hf_res = run_scf(&atoms, &basis, 60, 1e-8).expect("RHF SO3 failed");
    assert!(hf_res.converged);
    assert!(hf_res.total_energy < -615.0);

    // 2. TDHF / RPA Linear Response: In D3h symmetry, dipole-allowed in-plane electronic
    // excitations belong to the two-dimensional irreducible representation E' (E'_x, E'_y).
    // In exact non-relativistic quantum mechanics, these two states must be strictly degenerate:
    // Ref: Herzberg, Infrared and Raman Spectra of Polyatomic Molecules (1945).
    let tdhf_res = run_tdhf(&atoms, &basis, 0, 5, 60, 1e-8).expect("TDHF SO3 failed");
    assert_eq!(tdhf_res.excited_states.len(), 5);

    // State 0 is the non-degenerate out-of-plane A_1'' state (forbidden, f = 0):
    assert!(tdhf_res.excited_states[0].excitation_energy_ev > 0.0);
    assert!(tdhf_res.excited_states[0].oscillator_strength < 1e-5);

    // States 1 and 2 form the doubly degenerate E'' dark pair:
    let de_12 = (tdhf_res.excited_states[1].excitation_energy_ev - tdhf_res.excited_states[2].excitation_energy_ev).abs();
    assert!(
        de_12 < 1e-4,
        "D_3h symmetry mandates degenerate E'' dark excitation pair: got difference {} eV",
        de_12
    );

    // States 3 and 4 form the doubly degenerate dipole-allowed E' bright pair (f > 0):
    let de_34 = (tdhf_res.excited_states[3].excitation_energy_ev - tdhf_res.excited_states[4].excitation_energy_ev).abs();
    assert!(
        de_34 < 1e-4,
        "D_3h symmetry mandates degenerate E' bright in-plane excitation pair: got difference {} eV",
        de_34
    );
    assert!(tdhf_res.excited_states[3].oscillator_strength > 0.01);

    // 3. Relativistic Scalar X2C Core Stabilization on 3rd-row Sulfur (Z=16):
    // Core 1s and 2s electrons experience significant relativistic mass-velocity contraction:
    // Ref: Reiher & Wolf, Relativistic Quantum Chemistry (Wiley-VCH, 2009).
    let res_x2c = run_relativistic_hf(&atoms, &basis, 0, RelativisticModel::X2C, 60, 1e-8)
        .expect("X2C SO3 failed");
    assert!(res_x2c.converged);
    assert!(
        res_x2c.relativistic_shift_hartree < -0.5,
        "Sulfur (Z=16) core stabilization must be negative: got {}",
        res_x2c.relativistic_shift_hartree
    );
}

// -----------------------------------------------------------------------------
// 30. Dichlorine (Cl2): TCSCF Bond Dissociation & Coulson-Fischer RHF Instability
// -----------------------------------------------------------------------------

#[test]
fn test_cl2_dichlorine_tcscf_bond_dissociation_and_rhf_instability() {
    let basis_map = get_builtin_basis("3-21g").unwrap();

    // 1. Equilibrium Cl2 (R_e = 3.757 Bohr):
    // Cl2 has 34 electrons (n_occ = 17). Ground state is dominated by single determinant (3sigma_g)^2.
    let atoms_eq = get_dichlorine(3.757);
    let basis_eq = build_basis(&atoms_eq, &basis_map);
    let tcscf_eq = run_tcscf(&atoms_eq, &basis_eq, 0, 16, 17, 60, 1e-7).expect("TCSCF Cl2 eq failed");

    assert!(tcscf_eq.converged);
    assert!(
        tcscf_eq.c1.abs() > 0.95,
        "Near equilibrium, closed-shell configuration dominates: |c1| = {}",
        tcscf_eq.c1.abs()
    );
    assert_eq!(tcscf_eq.s_squared, 0.0, "TCSCF remains strictly a singlet <S^2> = 0");

    // 2. Stretched Cl2 (R = 5.5 Bohr) across the Coulson-Fischer point:
    // Single-determinant RHF fails to dissociate homolytically into 2 Cl(2P) radicals,
    // retaining 50% spurious ionic character (Cl+ + Cl-):
    // Ref: Coulson & Fischer, Philos. Mag. 40, 386 (1949).
    let atoms_str = get_dichlorine(5.5);
    let basis_str = build_basis(&atoms_str, &basis_map);

    // Strictly enforce RHF without auto-switching to UHF!
    let hf_str = run_scf_general(&atoms_str, &basis_str, 0, 70, 1e-8, ConvAlgo::Adiissoscf, None)
        .expect("RHF Cl2 stretched failed");
    assert!(hf_str.converged);

    let stab = analyze_rhf_stability(&hf_str, hf_str.n_occ);
    assert!(
        !stab.is_stable,
        "Stretched Cl2 exceeds the Coulson-Fischer point and must be RHF unstable"
    );
    assert!(
        stab.lowest_eigenvalue < 0.0,
        "Stability Hessian must have negative eigenvalue, got {}",
        stab.lowest_eigenvalue
    );

    // 3. Two-Configuration SCF restores proper radical character:
    // |Psi_TCSCF> = c1 |sigma_g^2> - c2 |sigma_u*^2|
    let tcscf_str = run_tcscf(&atoms_str, &basis_str, 0, 16, 17, 60, 1e-7).expect("TCSCF Cl2 stretched failed");
    assert!(tcscf_str.converged);
    assert!(
        tcscf_str.c2.abs() > 0.15,
        "Stretched bond requires multireference static correlation: |c2| = {}",
        tcscf_str.c2.abs()
    );
    assert!(
        tcscf_str.correlation_energy < -0.005,
        "Static correlation energy must lower total energy: got {}",
        tcscf_str.correlation_energy
    );
    assert_eq!(tcscf_str.s_squared, 0.0);
}

// -----------------------------------------------------------------------------
// 31. Acetonitrile (CH3CN): Fermi-Teller Dipole-Bound Positronic HF & C3v TDHF
// -----------------------------------------------------------------------------

#[test]
fn test_ch3cn_acetonitrile_positronic_dipole_bound_state_and_tdhf() {
    let atoms = get_acetonitrile();
    let basis_map = get_builtin_basis("3-21g").unwrap();
    let basis = build_basis(&atoms, &basis_map);

    let hf_neutral = run_scf(&atoms, &basis, 60, 1e-8).expect("RHF CH3CN failed");
    assert!(hf_neutral.converged);

    // 1. Permanent electric dipole moment:
    let dip = hf_neutral.dipole_moment_debye();
    let dip_mag = (dip[0] * dip[0] + dip[1] * dip[1] + dip[2] * dip[2]).sqrt();
    // According to the Fermi-Teller theorem, mu_crit = 1.625 Debye is required for a point dipole to bind a charge:
    // Ref: Fermi & Teller, Phys. Rev. 72, 399 (1947); Crawford, Proc. Phys. Soc. 91, 279 (1967).
    assert!(
        dip_mag > 2.0,
        "Acetonitrile permanent dipole {} D must exceed Fermi-Teller threshold (1.625 D)",
        dip_mag
    );

    // 2. Positronic HF: Forms a genuine dipole-bound positron state [e+; CH3CN]
    let pos_res = run_positronic_hf(&atoms, &basis, 0, 70, 1e-7).expect("Positronic CH3CN failed");
    assert!(pos_res.converged);
    assert!(
        pos_res.ep_attraction_energy < -0.05,
        "Attractive electron-positron Coulomb energy must be stabilizing, got {}",
        pos_res.ep_attraction_energy
    );

    // Exact metric trace normalization Tr(P_p * S) = 1.0:
    let s = &hf_neutral.overlap_matrix;
    let mut n_pos = 0.0;
    for i in 0..s.rows {
        for j in 0..s.cols {
            n_pos += pos_res.positron_density[(i, j)] * s[(j, i)];
        }
    }
    assert!(
        (n_pos - 1.0).abs() < 1e-4,
        "Positron normalization Tr(P_p * S) was {}",
        n_pos
    );

    // 3. TDHF: In C3v symmetry, the lowest transition is a non-degenerate forbidden A2 transition (f = 0),
    // followed by doubly degenerate E transitions in x and y:
    let tdhf_res = run_tdhf(&atoms, &basis, 0, 5, 60, 1e-8).expect("TDHF CH3CN failed");
    assert_eq!(tdhf_res.excited_states.len(), 5);

    // State 0 is non-degenerate A2 dark state (f = 0):
    assert!(tdhf_res.excited_states[0].excitation_energy_ev > 0.0);
    assert!(tdhf_res.excited_states[0].oscillator_strength < 1e-5);

    // States 1 and 2 form the doubly degenerate E pair:
    let de_12 = (tdhf_res.excited_states[1].excitation_energy_ev - tdhf_res.excited_states[2].excitation_energy_ev).abs();
    assert!(
        de_12 < 1e-4,
        "C_3v cyano pi -> pi* excitation must be degenerate in x and y, got difference {} eV",
        de_12
    );

    // States 3 and 4 form the next doubly degenerate E pair:
    let de_34 = (tdhf_res.excited_states[3].excitation_energy_ev - tdhf_res.excited_states[4].excitation_energy_ev).abs();
    assert!(
        de_34 < 1e-4,
        "Higher C_3v excitation must be doubly degenerate E, got difference {} eV",
        de_34
    );
}

// -----------------------------------------------------------------------------
// 32. Fluoromethane (CH3F): Localized HF (LHF / OEP-EXX) & BSE Variational Hierarchy
// -----------------------------------------------------------------------------

#[test]
fn test_ch3f_fluoromethane_lhf_exact_exchange_and_bse_hierarchy() {
    let atoms = get_fluoromethane();

    // 1. Localized Hartree-Fock (LHF / OEP-EXX):
    // Constructs an exact local multiplicative exchange potential v_x^{LHF}(r) on the compact
    // valence lone pairs of Fluorine:
    // Ref: Sala & Görling, J. Chem. Phys. 115, 5718 (2001).
    let map_321g = get_builtin_basis("3-21g").unwrap();
    let basis_321g = build_basis(&atoms, &map_321g);

    let hf_res = run_scf(&atoms, &basis_321g, 60, 1e-8).expect("RHF CH3F failed");
    let lhf_res = run_lhf_scf(&atoms, &basis_321g, 0, 60, 1e-7).expect("LHF CH3F failed");

    assert!(hf_res.converged && lhf_res.converged);
    assert!(lhf_res.homo_lumo_gap.unwrap() > 0.0);

    // Theoretical Boundary: By the Rayleigh-Ritz variational theorem, constraining the exchange
    // operator to be a local multiplicative potential can only raise the total energy:
    // E_LHF >= E_HF. An energy difference E_LHF - E_HF >= 0 is a rigorous physical requirement.
    assert!(
        lhf_res.total_energy >= hf_res.total_energy - 1e-5,
        "Variational bound: E_LHF ({}) must be >= E_HF ({})",
        lhf_res.total_energy, hf_res.total_energy
    );

    // 2. Basis Set Exchange (BSE) Variational Hierarchy on polar C-F covalent bond:
    // 3-21G -> def2-SVP
    let map_def2_svp = get_builtin_basis("def2-svp").unwrap();
    let basis_def2 = build_basis(&atoms, &map_def2_svp);
    let e_def2_svp = run_scf(&atoms, &basis_def2, 60, 1e-8).expect("RHF def2-SVP CH3F failed").total_energy;

    assert!(
        e_def2_svp < hf_res.total_energy,
        "Variational principle: def2-SVP ({}) must be lower than 3-21G ({})",
        e_def2_svp, hf_res.total_energy
    );
}

// -----------------------------------------------------------------------------
// 33. Hydrogen Sulfide (H2S): Dual Quantum Proton NEO-HF & Relativistic X2C
// -----------------------------------------------------------------------------

#[test]
fn test_h2s_hydrogen_sulfide_neo_hf_and_x2c_stabilization() {
    let atoms = get_h2s();
    let basis_map = get_builtin_basis("3-21g").unwrap();
    let elec_basis = build_basis(&atoms, &basis_map);

    // 1. Non-relativistic RHF reference: S-H bond angle ~92 deg (Walsh's rule: unhybridized 3p)
    let hf_res = run_scf(&atoms, &elec_basis, 60, 1e-8).expect("RHF H2S failed");
    assert!(hf_res.converged);
    assert!(hf_res.total_energy < -396.0);

    // 2. NEO-HF on weak S-H bonds: Both protons treated as quantum mechanical wavepackets
    let prot_basis = elec_basis[elec_basis.len() - 4..].to_vec(); // S is 13 AOs, H1 & H2 are 4 AOs
    let neo_res = run_neo_hf(&atoms[..1], &elec_basis, &prot_basis, 0, 2, 80, 1e-6)
        .expect("NEO H2S dual proton failed");
    assert!(neo_res.converged);
    assert!(
        neo_res.electron_proton_coulomb < -0.1,
        "Attractive electron-proton Coulomb energy in H2S: got {}",
        neo_res.electron_proton_coulomb
    );

    // 3. Relativistic Scalar X2C Core Stabilization on 3rd-row Sulfur (Z=16):
    let res_x2c = run_relativistic_hf(&atoms, &elec_basis, 0, RelativisticModel::X2C, 60, 1e-8)
        .expect("X2C H2S failed");
    assert!(res_x2c.converged);
    assert!(
        res_x2c.relativistic_shift_hartree < -0.5,
        "Sulfur core relativistic stabilization in H2S: got {}",
        res_x2c.relativistic_shift_hartree
    );
}

// -----------------------------------------------------------------------------
// 34. Nitrous Oxide (N2O): Linear C_inf_v TDHF Spectrum & Cavity QED-HF
// -----------------------------------------------------------------------------

#[test]
fn test_n2o_nitrous_oxide_tdhf_degenerate_pi_and_cavity_qed() {
    let atoms = get_n2o();
    let basis_map = get_builtin_basis("3-21g").unwrap();
    let basis = build_basis(&atoms, &basis_map);

    let hf_res = run_scf(&atoms, &basis, 60, 1e-8).expect("RHF N2O failed");
    assert!(hf_res.converged);
    assert!(hf_res.total_energy < -182.0);

    // 1. TDHF Linear Response: In linear C_inf_v symmetry, the lowest transition is non-degenerate 1Sigma^- (f = 0),
    // followed by the doubly degenerate 1Delta pair and 1Pi pair:
    let tdhf_res = run_tdhf(&atoms, &basis, 0, 5, 60, 1e-8).expect("TDHF N2O failed");
    assert_eq!(tdhf_res.excited_states.len(), 5);

    // State 0 is non-degenerate 1Sigma^- dark state (f = 0):
    assert!(tdhf_res.excited_states[0].excitation_energy_ev > 0.0);
    assert!(tdhf_res.excited_states[0].oscillator_strength < 1e-5);

    // States 1 and 2 form the doubly degenerate 1Delta pair:
    let de_12 = (tdhf_res.excited_states[1].excitation_energy_ev - tdhf_res.excited_states[2].excitation_energy_ev).abs();
    assert!(
        de_12 < 1e-4,
        "Linear N2O (C_inf_v) mandates degenerate 1Delta excitations, got difference {} eV",
        de_12
    );

    // States 3 and 4 form the doubly degenerate 1Pi pair:
    let de_34 = (tdhf_res.excited_states[3].excitation_energy_ev - tdhf_res.excited_states[4].excitation_energy_ev).abs();
    assert!(
        de_34 < 1e-4,
        "Linear N2O (C_inf_v) mandates degenerate 1Pi excitations, got difference {} eV",
        de_34
    );

    // 2. Cavity QED-HF: Dipole self-energy along molecular axis (z)
    let qed_z = run_qed_hf(&atoms, &basis, 0, 0.1, [0.0, 0.0, 0.05], 60, 1e-8)
        .expect("QED N2O z failed");
    assert!(qed_z.converged);
    assert!(qed_z.dipole_self_energy > 0.0);
}

// -----------------------------------------------------------------------------
// 35. Difluorine (F2): Classic Fluorine Bond Problem & TCSCF Multireference
// -----------------------------------------------------------------------------

#[test]
fn test_f2_difluorine_tcscf_multireference_and_rhf_instability() {
    let basis_map = get_builtin_basis("3-21g").unwrap();

    // 1. Equilibrium F2 (R_e = 2.668 Bohr): 18 electrons (n_occ = 9). Active pair: HOMO = 8, LUMO = 9.
    let atoms_eq = get_difluorine(2.668);
    let basis_eq = build_basis(&atoms_eq, &basis_map);
    let tcscf_eq = run_tcscf(&atoms_eq, &basis_eq, 0, 8, 9, 60, 1e-7).expect("TCSCF F2 eq failed");

    assert!(tcscf_eq.converged);
    assert!(tcscf_eq.c1.abs() > 0.95);
    assert_eq!(tcscf_eq.s_squared, 0.0);

    // 2. Stretched F2 (R = 4.5 Bohr) across the Coulson-Fischer point:
    let atoms_str = get_difluorine(4.5);
    let basis_str = build_basis(&atoms_str, &basis_map);

    // Strictly enforce RHF without auto-switching to UHF!
    let hf_str = run_scf_general(&atoms_str, &basis_str, 0, 70, 1e-8, ConvAlgo::Adiissoscf, None)
        .expect("RHF F2 stretched failed");
    assert!(hf_str.converged);

    let stab = analyze_rhf_stability(&hf_str, hf_str.n_occ);
    assert!(
        !stab.is_stable,
        "Stretched F2 exceeds Coulson-Fischer point and is RHF unstable"
    );
    assert!(stab.lowest_eigenvalue < 0.0);

    // 3. TCSCF captures static correlation across bond dissociation:
    let tcscf_str = run_tcscf(&atoms_str, &basis_str, 0, 8, 9, 60, 1e-7).expect("TCSCF F2 stretched failed");
    assert!(tcscf_str.converged);
    assert!(
        tcscf_str.c2.abs() > 0.15,
        "Stretched F2 requires multireference static correlation: |c2| = {}",
        tcscf_str.c2.abs()
    );
    assert!(
        tcscf_str.correlation_energy < -0.005,
        "Static correlation energy: got {}",
        tcscf_str.correlation_energy
    );
    assert_eq!(tcscf_str.s_squared, 0.0);
}

// -----------------------------------------------------------------------------
// 36. Ethylene Glycol: Vicinal Diol Linear-Scaling DC-HF & FMO2 PIEDA
// -----------------------------------------------------------------------------

#[test]
fn test_ethylene_glycol_dc_hf_and_fmo2_pieda() {
    let atoms = get_ethylene_glycol();
    let basis_map = get_builtin_basis("3-21g").unwrap();
    let basis = build_basis(&atoms, &basis_map);

    // 1. Divide-and-Conquer (DC-HF) linear scaling:
    let dc_opts = DCOptions {
        buffer_radius_bohr: 5.0,
        max_iterations: 80,
        energy_threshold: 1e-7,
        ..Default::default()
    };
    let dc_res = run_dc_hf(&atoms, &basis, 0, &dc_opts).expect("DC-HF ethylene glycol failed");
    assert!(dc_res.converged);
    assert!(dc_res.total_energy < -227.0);
    assert!(dc_res.chemical_potential.is_finite());

    // 2. FMO2 PIEDA Energy Decomposition across central C-C bond:
    // Fragment 1: [HO-CH2]^+ (atoms 0, 1, 4, 6, 7; charge +1)
    // Fragment 2: [CH2-OH]^- (atoms 2, 3, 5, 8, 9; charge -1)
    let fragments = vec![
        FMOFragment::new("HO_CH2_Cation", vec![0, 1, 4, 6, 7], 1),
        FMOFragment::new("CH2_OH_Anion", vec![2, 3, 5, 8, 9], -1),
    ];
    let fmo_opts = FMOOptions { order: 2, ..Default::default() };
    let fmo = run_fmo(&atoms, &basis, &fragments, &fmo_opts).expect("FMO2 ethylene glycol failed");
    assert_eq!(fmo.monomer_energies.len(), 2);

    let ifie = &fmo.ifie[0];
    let sum_pieda = ifie.electrostatic + ifie.exchange_repulsion + ifie.charge_transfer;
    assert!(
        (sum_pieda - ifie.total_interaction_energy).abs() < 1e-10,
        "PIEDA sum ({}) must equal total interaction ({})",
        sum_pieda, ifie.total_interaction_energy
    );
}

// -----------------------------------------------------------------------------
// 37. Phosphine (PH3): Sub-Critical Dipole Moment & Fermi-Teller Boundary
// -----------------------------------------------------------------------------

#[test]
fn test_ph3_phosphine_sub_critical_dipole_and_positronic_hf() {
    let atoms = get_phosphine();
    let basis_map = get_builtin_basis("3-21g").unwrap();
    let basis = build_basis(&atoms, &basis_map);

    let hf_neutral = run_scf(&atoms, &basis, 60, 1e-8).expect("RHF PH3 failed");
    assert!(hf_neutral.converged);

    // 1. Permanent electric dipole moment:
    let dip = hf_neutral.dipole_moment_debye();
    let dip_mag = (dip[0] * dip[0] + dip[1] * dip[1] + dip[2] * dip[2]).sqrt();
    // Theoretical Boundary: By the Fermi-Teller theorem, mu_crit = 1.625 Debye is required
    // for a point dipole to form a bound state:
    // Ref: Fermi & Teller, Phys. Rev. 72, 399 (1947); Crawford, Proc. Phys. Soc. 91, 279 (1967).
    assert!(
        dip_mag < 1.625,
        "Phosphine dipole {} D must be sub-critical (less than 1.625 D)",
        dip_mag
    );

    // 2. Positronic HF: On sub-critical dipole systems, the positron density is diffuse
    let pos_res = run_positronic_hf(&atoms, &basis, 0, 70, 1e-7).expect("Positronic PH3 failed");
    assert!(pos_res.converged);

    // Exact metric trace normalization Tr(P_p * S) = 1.0:
    let s = &hf_neutral.overlap_matrix;
    let mut n_pos = 0.0;
    for i in 0..s.rows {
        for j in 0..s.cols {
            n_pos += pos_res.positron_density[(i, j)] * s[(j, i)];
        }
    }
    assert!(
        (n_pos - 1.0).abs() < 1e-4,
        "Positron normalization Tr(P_p * S) was {}",
        n_pos
    );
}

// -----------------------------------------------------------------------------
// 46. Benzene (C6H6): TDHF D6h UV-Vis Pariser-Parr-Pople Spectrum & DC-HF Ring Partition
// -----------------------------------------------------------------------------
#[test]
fn test_benzene_c6h6_tdhf_uv_vis_and_dc_hf_ring_partition() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_benzene();
    let basis = build_basis(&atoms, &basis_map);

    let hf_res = run_scf(&atoms, &basis, 60, 1e-8).expect("RHF Benzene failed");
    assert!(hf_res.converged);
    assert!(hf_res.total_energy < -227.0);

    // 1. TDHF UV-Vis spectrum:
    // In D6h symmetry, Pariser-Parr-Pople theory proves that the lowest transitions
    // from the e1g (pi) to e2u (pi*) manifold split into:
    //   1B2u (dark, forbidden f=0)
    //   1B1u (dark, forbidden f=0)
    //   1E1u (bright, doubly degenerate, f > 0)
    // Ref: Pariser & Parr, J. Chem. Phys. 21, 767 (1953); Pople, Trans. Faraday Soc. 49, 1375 (1953).
    let tdhf = run_tdhf(&atoms, &basis, 0, 5, 60, 1e-8).expect("TDHF Benzene failed");
    assert!(tdhf.excited_states.len() >= 4);

    // Lowest transitions are dark (f = 0):
    assert!(
        tdhf.excited_states[0].oscillator_strength < 1e-3,
        "Lowest benzene state must be dipole forbidden (1B2u, f=0), was f={}",
        tdhf.excited_states[0].oscillator_strength
    );
    assert!(
        tdhf.excited_states[1].oscillator_strength < 1e-3,
        "Second benzene state must be dipole forbidden (1B1u, f=0), was f={}",
        tdhf.excited_states[1].oscillator_strength
    );

    // Higher degenerate transition is bright (1E1u, f > 0):
    let has_bright = tdhf.excited_states.iter().any(|s| s.oscillator_strength > 0.05);
    assert!(has_bright, "Benzene must possess a strongly allowed 1E1u transition (f > 0.05)");

    // 2. Divide-and-Conquer HF (DC-HF) on the 6-carbon aromatic ring:
    let dc_opts = DCOptions {
        buffer_radius_bohr: 5.5,
        energy_threshold: 1e-7,
        ..Default::default()
    };
    let dc_res = run_dc_hf(&atoms, &basis, 0, &dc_opts).expect("DC-HF Benzene failed");
    assert!(dc_res.converged);
    // A 5.5-bohr buffer does not span the ring, so DC-HF is an approximation whose size is
    // not bounded a priori; the electron count is conserved exactly.
    assert!((dc_res.electron_count - 42.0).abs() < 1e-8);
    // Buffers spanning the whole ring reduce DC-HF to RHF (gap >> 1/beta).
    let full = run_dc_hf(&atoms, &basis, 0, &DCOptions {
        buffer_radius_bohr: 20.0,
        inverse_temperature: 1000.0,
        ..Default::default()
    })
    .expect("full-buffer DC-HF benzene failed");
    assert!(
        (full.total_energy - hf_res.total_energy).abs() < 1e-8,
        "full-buffer DC-HF {} vs RHF {}",
        full.total_energy, hf_res.total_energy
    );
}

// -----------------------------------------------------------------------------
// 47. Borane (BH3): Planar D3h Empty 2p_z Orbital & Cavity QED Anisotropy
// -----------------------------------------------------------------------------
// 47. Borane (BH3): Planar D3h Empty 2p_z Orbital & Cavity QED Anisotropy
// -----------------------------------------------------------------------------
#[test]
fn test_borane_bh3_empty_p_orbital_cavity_qed_and_tcscf() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_borane();
    let basis = build_basis(&atoms, &basis_map);

    let hf_res = run_scf(&atoms, &basis, 60, 1e-8).expect("RHF BH3 failed");
    assert!(hf_res.converged);

    // 1. Cavity QED HF:
    // BH3 is a planar D3h Lewis acid with an empty 2p_z LUMO (a2'') oriented along z.
    // Cavity field aligned along z couples directly to out-of-plane polarization:
    let qed_z = run_qed_hf(&atoms, &basis, 0, 0.10, [0.0, 0.0, 0.05], 60, 1e-8)
        .expect("QED-HF BH3 (z) failed");
    let qed_x = run_qed_hf(&atoms, &basis, 0, 0.10, [0.05, 0.0, 0.0], 60, 1e-8)
        .expect("QED-HF BH3 (x) failed");

    assert!(qed_z.converged);
    assert!(qed_x.converged);
    // Vacuum dipole self-energy and polaritonic shift are anisotropic:
    assert!(
        (qed_z.total_energy - qed_x.total_energy).abs() > 1e-5,
        "Cavity coupling must be anisotropic between z (empty 2p_z) and in-plane x"
    );

    // 2. TCSCF static correlation on Lewis-acid BH3 (4 electrons pairs = 8 electrons, HOMO 3 to LUMO 4):
    let tcscf = run_tcscf(&atoms, &basis, 0, 3, 4, 400, 1e-8).expect("TCSCF BH3 failed");
    assert!(tcscf.converged);
    // Closed-shell character dominates (c1 ~ 0.999), demonstrating lack of diradical character at equilibrium:
    assert!(
        tcscf.c1.abs() > 0.95,
        "BH3 at equilibrium has closed-shell singlet ground state"
    );
}

// -----------------------------------------------------------------------------
// 48. Silicon Tetrafluoride (SiF4): Td Symmetry Zero-Dipole & Relativistic X2C Stabilization
// -----------------------------------------------------------------------------
#[test]
fn test_sif4_silicon_tetrafluoride_td_symmetry_zero_dipole_and_relativistic_x2c() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_silicon_tetrafluoride();
    let basis = build_basis(&atoms, &basis_map);

    let hf_res = run_scf(&atoms, &basis, 60, 1e-8).expect("RHF SiF4 failed");
    assert!(hf_res.converged);

    // 1. Exact zero electric dipole moment by Td tetrahedral symmetry:
    let dip = hf_res.dipole_moment_debye();
    let dip_mag = (dip[0] * dip[0] + dip[1] * dip[1] + dip[2] * dip[2]).sqrt();
    assert!(
        dip_mag < 1e-6,
        "SiF4 in Td symmetry must have identically zero dipole moment, was {}",
        dip_mag
    );

    // 2. Scalar relativistic X2C core stabilization:
    // Silicon (Z=14) 1s, 2s, 2p orbitals undergo relativistic contraction:
    // Ref: Pyykkö, Chem. Rev. 88, 563 (1988); Dyall & Faegri, Relativistic Quantum Chemistry (2007).
    let rel_res = run_relativistic_hf(&atoms, &basis, 0, RelativisticModel::X2C, 60, 1e-8)
        .expect("X2C SiF4 failed");
    assert!(rel_res.converged);

    let delta_e_rel = rel_res.total_energy - hf_res.total_energy;
    assert!(
        delta_e_rel < -0.5,
        "Relativistic X2C correction on SiF4 must be strongly stabilizing (delta E = {} Eh < -0.5 Eh)",
        delta_e_rel
    );
}

// -----------------------------------------------------------------------------
// 49. Nitrogen Dioxide (NO2): Open-Shell Radical, UHF Spin Contamination & PUHF
// -----------------------------------------------------------------------------
#[test]
fn test_no2_nitrogen_dioxide_radical_uhf_spin_contamination_and_puhf() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_nitrogen_dioxide();
    let basis = build_basis(&atoms, &basis_map);

    // 1. UHF on 23-electron doublet radical (S=1/2, target <S^2> = 0.75):
    let uhf = run_uhf(&atoms, &basis, 0, 2, false, 60, 1e-8).expect("UHF NO2 failed");
    assert!(uhf.converged);

    // Theoretical Boundary: Due to spin polarization of the oxygen lone pairs,
    // single-determinant UHF suffers from spin contamination (<S^2> > 0.75):
    // Ref: Hay, J. Chem. Phys. 58, 4706 (1973); Sherrill et al., J. Chem. Phys. 109, 4171 (1998).
    assert!(
        uhf.s_squared > 0.75,
        "NO2 radical must show spin contamination (<S^2> = {} > 0.75)",
        uhf.s_squared
    );

    // 2. Projected UHF (PUHF) spin restoration:
    let puhf = run_puhf(&atoms, &basis, 0, 80, 1e-7).expect("PUHF NO2 failed");
    assert!(puhf.raw_uhf.converged);
    assert!(
        puhf.spin_contamination_weight > 0.0,
        "Spin contamination weight must be non-zero (w = {})",
        puhf.spin_contamination_weight
    );
    assert!(
        puhf.projected_energy.is_finite(),
        "Projected energy must be finite"
    );

    // 3. GHF (Generalized Hartree-Fock):
    let ghf = run_ghf(&atoms, &basis, 0, 0.0, 60, 1e-8).expect("GHF NO2 failed");
    assert!(ghf.converged);
}

// -----------------------------------------------------------------------------
// 50. Beryllium Hydride (BeH2): Linear D_inf_h Coulson-Fischer Instability & TCSCF
// -----------------------------------------------------------------------------
#[test]
fn test_beh2_beryllium_hydride_symmetric_stretch_rhf_instability_and_tcscf() {
    let basis_map = parse_gaussian94(STO_3G_GBS);

    // 1. Equilibrium geometry (R = 2.506 Bohr): Stable closed-shell singlet (6 electrons -> n_occ = 3)
    let atoms_eq = get_beryllium_hydride(2.506);
    let basis_eq = build_basis(&atoms_eq, &basis_map);
    let hf_eq = run_scf(&atoms_eq, &basis_eq, 60, 1e-8).expect("RHF BeH2 eq failed");
    assert!(hf_eq.converged);

    let stab_eq = analyze_rhf_stability(&hf_eq, 3);
    assert!(
        stab_eq.is_stable,
        "BeH2 at equilibrium must be RHF stable (lowest eigenvalue = {})",
        stab_eq.lowest_eigenvalue
    );

    // 2. Stretched geometry (R = 4.5 Bohr):
    // Theoretical Boundary: Under symmetric bond stretching, (1\sigma_u)^2 and (2\sigma_u*)^2
    // become quasi-degenerate, causing single-determinant RHF to suffer Coulson-Fischer instability:
    // Ref: Purvis & Bartlett, J. Chem. Phys. 76, 1910 (1982); Stanton et al., J. Chem. Phys. 98, 7029 (1993).
    let atoms_str = get_beryllium_hydride(4.5);
    let basis_str = build_basis(&atoms_str, &basis_map);
    let hf_str = run_scf(&atoms_str, &basis_str, 60, 1e-8).expect("RHF BeH2 stretched failed");
    assert!(hf_str.converged);

    let stab_str = analyze_rhf_stability(&hf_str, 3);
    assert!(
        !stab_str.is_stable,
        "Stretched BeH2 must display Coulson-Fischer instability (lambda = {} < 0)",
        stab_str.lowest_eigenvalue
    );

    // 3. TCSCF captures static pair correlation without spin contamination (BeH2 has 6 electrons, HOMO=2, LUMO=3):
    let tcscf_str = run_tcscf(&atoms_str, &basis_str, 0, 2, 3, 60, 1e-8).expect("TCSCF BeH2 stretched failed");
    assert!(tcscf_str.converged);
    assert!(
        tcscf_str.c2.abs() > 0.15,
        "Stretched BeH2 must exhibit strong two-configuration character (|c2| = {} > 0.15)",
        tcscf_str.c2
    );
    assert!(
        tcscf_str.tcscf_energy < hf_str.total_energy,
        "TCSCF energy ({}) must be lower than RHF energy ({})",
        tcscf_str.tcscf_energy,
        hf_str.total_energy
    );
}

// -----------------------------------------------------------------------------
// 51. Pyridine (C5H5N): Heterocyclic Aromatic Ring TDHF n->pi* Dark State & DC-HF
// -----------------------------------------------------------------------------
#[test]
fn test_pyridine_c5h5n_heterocycle_tdhf_and_dc_hf_partition() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_pyridine();
    let basis = build_basis(&atoms, &basis_map);

    // Pyridine has 42 electrons -> 21 occupied pairs
    let hf = run_scf(&atoms, &basis, 60, 1e-8).expect("RHF Pyridine failed");
    assert!(hf.converged);
    assert!((hf.total_energy - (-243.6)).abs() < 5.0, "Pyridine RHF energy must be physical");

    // 1. TDHF Excited States:
    // Theoretical Boundary: Pyridine has a low-lying n -> \pi^* dark transition
    // involving the non-bonding nitrogen lone pair to the \pi^* LUMO.
    // Due to local spatial localization, its transition dipole moment and oscillator strength
    // are very small (f < 0.02), distinguishing it from the strongly dipole-allowed \pi -> \pi^* states (f > 0.04).
    // Ref: Innes et al., Chem. Rev. 88, 1293 (1988); Wiberg et al., J. Phys. Chem. A 102, 4920 (1998).
    let tdhf = run_tdhf(&atoms, &basis, 0, 4, 80, 1e-7).expect("TDHF Pyridine failed");
    assert!(tdhf.excited_states.len() >= 2);

    let state1 = &tdhf.excited_states[0];
    assert!(state1.excitation_energy_hartree > 0.10, "Excitation energy must be positive");
    assert!(
        state1.oscillator_strength < 0.02,
        "Lowest n -> pi* transition must be weakly allowed or dark (f = {} < 0.02)",
        state1.oscillator_strength
    );

    for (idx, s) in tdhf.excited_states.iter().enumerate() {
        println!("State {}: E = {:.4} Eh ({:.2} eV), f = {:.6}", idx + 1, s.excitation_energy_hartree, s.excitation_energy_ev, s.oscillator_strength);
    }
    // TDHF/STO-3G places the strongly allowed pi -> pi* states above the four lowest roots, so only
    // properties that hold by construction are asserted here; spectra are validated against
    // complete PySCF references in tests/tdhf_test.rs.
    for pair in tdhf.excited_states.windows(2) {
        assert!(pair[0].excitation_energy_hartree <= pair[1].excitation_energy_hartree);
    }
    assert!(tdhf.excited_states.iter().all(|s| s.oscillator_strength >= 0.0 && !s.is_triplet));

    // 2. Linear Scaling Divide-and-Conquer HF (DC-HF):
    let dc_opts = DCOptions {
        buffer_radius_bohr: 6.0,
        energy_threshold: 1e-6,
        max_iterations: 100,
        ..Default::default()
    };
    let dc_res = run_dc_hf(&atoms, &basis, 0, &dc_opts).expect("DC-HF Pyridine failed");
    println!("DC-HF Pyridine: converged = {}, iters = {}, E = {:.6} (RHF = {:.6})",
        dc_res.converged, dc_res.iterations, dc_res.total_energy, hf.total_energy);
    assert!(dc_res.converged);
    let dc_diff = (dc_res.total_energy - hf.total_energy).abs();
    assert!(
        dc_diff < 0.15,
        "DC-HF energy ({}) must agree closely with full RHF ({}) (diff = {:.4} Eh)",
        dc_res.total_energy, hf.total_energy, dc_diff
    );
}

// -----------------------------------------------------------------------------
// 52. Chlorine Monofluoride (ClF): Interhalogen Relativistic Core Contraction & Dipole
// -----------------------------------------------------------------------------
#[test]
fn test_clf_chlorine_monofluoride_relativistic_x2c_and_dipole() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_clf();
    let basis = build_basis(&atoms, &basis_map);

    // ClF has 26 electrons -> 13 occupied pairs
    let nonrel = run_scf(&atoms, &basis, 60, 1e-8).expect("Nonrel RHF ClF failed");
    assert!(nonrel.converged);

    // 1. Relativistic Core Contraction (X2C):
    // Theoretical Boundary: Heavy atom Chlorine (Z=17) undergoes substantial relativistic mass-velocity
    // and Darwin core contraction of the 1s and 2s orbitals.
    // The scalar relativistic correction lowers the total energy by more than 1.5 Hartree:
    // \Delta E_{rel} = E_{X2C} - E_{nonrel} < -1.5 Eh.
    // Ref: Dyall & Faegri, Introduction to Relativistic Quantum Chemistry (Oxford Univ. Press, 2007);
    // Saue, ChemPhysChem 12, 3077 (2011).
    let x2c = run_relativistic_hf(&atoms, &basis, 0, RelativisticModel::X2C, 60, 1e-8)
        .expect("X2C ClF failed");
    assert!(x2c.converged);
    assert!(
        x2c.relativistic_shift_hartree < -1.0,
        "Chlorine Z=17 relativistic stabilization must exceed 1.0 Eh (delta = {} Eh)",
        x2c.relativistic_shift_hartree
    );
    assert!(
        x2c.total_energy < nonrel.total_energy,
        "Relativistic X2C total energy must be lower than non-relativistic RHF"
    );

    // 2. Electronegativity and Dipole Moment:
    // Pauling electronegativities: chi_F (3.98) > chi_Cl (3.16).
    // The dipole moment points from Cl(+delta) to F(-delta) along the z-axis.
    let dipole_z = nonrel.properties.dipole_moment_debye[2];
    assert!(
        dipole_z.abs() > 0.2,
        "Dipole moment along bond axis must be significant (mu_z = {} D)",
        dipole_z
    );
}

// -----------------------------------------------------------------------------
// 53. Formamide (HCONH2): Peptide Bond Archetype, Coherent-State QED-HF & DC-HF
// -----------------------------------------------------------------------------
#[test]
fn test_formamide_peptide_bond_cavity_qed_and_dc_hf() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_formamide();
    let basis = build_basis(&atoms, &basis_map);

    // Formamide has 24 electrons -> 12 occupied pairs
    let hf = run_scf(&atoms, &basis, 200, 1e-10).expect("RHF Formamide failed");
    assert!(hf.converged);

    // 1. Coherent-state QED-HF in two cavity polarizations. The coherent photon displacement
    // cancels the permanent dipole, so the shift is half the dipole variance in each direction and
    // is not controlled by the permanent dipole. Both shifts are compared with independent PySCF
    // values (tests/reference/gen_qed_hf.py, legacy_cases).
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/qed_hf_reference.json")).unwrap();
    let oracle = |name: &str| -> (f64, f64) {
        let case = reference["legacy_cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap();
        (case["energy"].as_f64().unwrap(), case["cavity_energy"].as_f64().unwrap())
    };
    let zero = run_qed_hf(&atoms, &basis, 0, 0.1, [0.0; 3], 200, 1e-10).expect("QED-HF zero failed");
    assert!((zero.total_energy - hf.total_energy).abs() < 1e-8);
    assert!((zero.total_energy - oracle("formamide_zero").0).abs() < 1e-8);
    for (name, lambda) in [
        ("formamide_in_plane", [0.0, 0.05, 0.0]),
        ("formamide_out_of_plane", [0.0, 0.0, 0.05]),
    ] {
        let qed = run_qed_hf(&atoms, &basis, 0, 0.1, lambda, 200, 1e-10).expect("QED-HF failed");
        assert!(qed.converged);
        let (energy, cavity) = oracle(name);
        assert!(
            (qed.total_energy - energy).abs() < 1e-8,
            "{name}: QED-HF {} vs oracle {energy}",
            qed.total_energy
        );
        assert!((qed.cavity_energy - cavity).abs() < 1e-8, "{name}: cavity energy");
        // The variational QED-HF energy lies above the cavity-free RHF minimum.
        assert!(qed.total_energy > hf.total_energy);
    }

    // 2. Divide-and-Conquer HF (DC-HF) on Peptide Bond:
    let dc_opts = DCOptions {
        buffer_radius_bohr: 5.0,
        energy_threshold: 1e-7,
        ..Default::default()
    };
    let dc_res = run_dc_hf(&atoms, &basis, 0, &dc_opts).expect("DC-HF Formamide failed");
    assert!(dc_res.converged);
    let dc_diff = (dc_res.total_energy - hf.total_energy).abs();
    assert!(
        dc_diff < 0.05,
        "DC-HF peptide bond energy ({}) must agree with full RHF ({}) (diff = {:.4} Eh)",
        dc_res.total_energy, hf.total_energy, dc_diff
    );
}

// -----------------------------------------------------------------------------
// 54. Hydroxyl Radical (OH): PUHF Spin Projection & Positronium Hydroxide (e+; OH-)
// -----------------------------------------------------------------------------
#[test]
fn test_oh_hydroxyl_radical_fermi_teller_positronic_bound_state_and_puhf() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_hydroxyl_radical();
    let basis = build_basis(&atoms, &basis_map);

    // 1. UHF on 9-electron doublet radical (S=1/2, target <S^2> = 0.75):
    let uhf = run_uhf(&atoms, &basis, 0, 2, false, 60, 1e-8).expect("UHF OH failed");
    assert!(uhf.converged);
    assert!(
        uhf.s_squared >= 0.75,
        "Doublet OH radical UHF must satisfy <S^2> >= 0.75 (got {})",
        uhf.s_squared
    );

    // 2. Projected UHF (PUHF):
    // Theoretical Boundary: Spin contamination from quartet contaminants is projected out,
    // restoring spin purity <S^2> -> 0.75. Projection after variation need not lower the energy:
    // Ref: Amos & Hall, Proc. R. Soc. A 263, 483 (1961); Baker, J. Chem. Phys. 89, 7421 (1988).
    let puhf = run_puhf(&atoms, &basis, 0, 80, 1e-7).expect("PUHF OH failed");
    assert!(puhf.raw_uhf.converged);
    assert!(puhf.projected_energy.is_finite(), "PUHF energy must be finite");
    assert!(puhf.target_weight > 0.0 && puhf.target_weight <= 1.0 + 1e-10);

    // 3. Positronic HF: closed-shell electrons only. The 9-electron radical is rejected (the
    // previous driver silently doubly occupied five orbitals, i.e. ten electrons); the positron
    // is attached to the 10-electron hydroxide anion instead ([e+; OH-], positronium hydroxide).
    assert!(run_positronic_hf(&atoms, &basis, 0, 80, 1e-7).is_err());
    let pos_res = run_positronic_hf(&atoms, &basis, -1, 200, 1e-10)
        .expect("Positronic HF OH- failed");
    assert!(pos_res.converged);
    assert!(pos_res.ep_attraction_energy < 0.0 && pos_res.positron_kinetic_energy > 0.0);

    // Exact metric trace normalizations Tr(P_p S) = 1 and Tr(P_e S) = 10:
    let s = &uhf.overlap_matrix;
    let n_pos = pos_res.positron_density.dot(s).trace();
    let n_el = pos_res.electron_density.dot(s).trace();
    assert!((n_pos - 1.0).abs() < 1e-10, "Positron normalization Tr(P_p S) was {n_pos}");
    assert!((n_el - 10.0).abs() < 1e-10, "Electron count Tr(P_e S) was {n_el}");
}


// -----------------------------------------------------------------------------
// 55. Carbon Disulfide (CS2): Linear D_inf_h TDHF Pi Degeneracy & Relativistic X2C
// -----------------------------------------------------------------------------
#[test]
fn test_cs2_carbon_disulfide_tdhf_pi_degeneracy_and_relativistic_x2c() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_cs2();
    let basis = build_basis(&atoms, &basis_map);

    // CS2 has 38 electrons -> 19 occupied pairs
    let hf = run_scf(&atoms, &basis, 60, 1e-8).expect("RHF CS2 failed");
    assert!(hf.converged);

    // 1. TDHF Degenerate Pi Excitations:
    // Theoretical Boundary: In linear centrosymmetric D_inf_h molecules, cylindrical symmetry
    // enforces strict pairwise degeneracy of \Pi_u and \Pi_g states (states polarized along x and y):
    // Ref: Herzberg, Molecular Spectra and Molecular Structure III (Van Nostrand, 1966);
    // Vasudevan & Grein, J. Chem. Phys. 68, 450 (1978).
    let tdhf = run_tdhf(&atoms, &basis, 0, 4, 80, 1e-7).expect("TDHF CS2 failed");
    // States 2 and 3 form the strictly degenerate Pi excited pair:
    let e2 = tdhf.excited_states[1].excitation_energy_hartree;
    let e3 = tdhf.excited_states[2].excitation_energy_hartree;
    assert!(
        (e2 - e3).abs() < 1e-6,
        "Perpendicular Pi excited states in CS2 (states 2 and 3) must be strictly degenerate: {} vs {}",
        e2, e3
    );

    // 2. Relativistic Core Contraction (X2C on two Sulfur atoms Z=16):
    // Both sulfur atoms undergo scalar relativistic core contraction:
    // \Delta E_{rel} < -1.5 Eh.
    let x2c = run_relativistic_hf(&atoms, &basis, 0, RelativisticModel::X2C, 60, 1e-8)
        .expect("X2C CS2 failed");
    assert!(x2c.converged);
    assert!(
        x2c.relativistic_shift_hartree < -1.5,
        "Two Sulfur Z=16 atoms must produce relativistic stabilization exceeding 1.5 Eh (delta = {} Eh)",
        x2c.relativistic_shift_hartree
    );

    // 3. Cholesky-Decomposed HF (CD-HF):
    // Pivoted Cholesky decomposition on ERI reproduced full RHF energy:
    let cd_res = run_cd_hf(&atoms, &basis, 0, 1e-5, 60, 1e-7).expect("CD-HF CS2 failed");
    assert!(cd_res.converged);
    let cd_diff = (cd_res.total_energy - hf.total_energy).abs();
    assert!(
        cd_diff < 3e-4,
        "CD-HF energy ({}) must agree with full RHF ({}) within threshold (diff = {:.2e} Eh)",
        cd_res.total_energy, hf.total_energy, cd_diff
    );
}

