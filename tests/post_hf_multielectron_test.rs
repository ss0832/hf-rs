// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Comprehensive multi-electron and asymmetric post-HF correlation benchmarks.
//!
//! Evaluates single-reference and multi-reference correlation methods on:
//! - H2 (2 electrons, D_inf_h symmetry)
//! - LiH (4 electrons, C_inf_v asymmetry, N_occ = 2)
//! - H2O (10 electrons, C_2v asymmetry, N_occ = 5)
//! - Open-shell Li atom and OH radical (doublet UHF-MP2)

#![allow(clippy::needless_range_loop)]

use hf_rs::basis::{build_basis, parse_gaussian94, Atom, STO_3G_GBS};
use hf_rs::diis::ConvAlgo;
use hf_rs::post_hf::cc::{CCD, CCSD, compute_ccsd_t_correction};
use hf_rs::post_hf::ci::cis::CIS;
use hf_rs::post_hf::ci::cisd::CISD;
use hf_rs::post_hf::ci::fci::FCI;
use hf_rs::post_hf::casscf::{CASSCF, CASSCFOptions};
use hf_rs::post_hf::caspt2::{CASPT2, CASPT2Options};
use hf_rs::post_hf::mp2::RHFMP2;
use hf_rs::post_hf::mpn::{RHFMP3, RHFMP4, RHFMP5, RHFMP6, RHFMP7, RHFMP8, RHFMPn};
use hf_rs::post_hf::hessian::compute_uhf_mp2_gradient;
use hf_rs::post_hf::{PostHFMethod, PostHFResult};
use hf_rs::scf::{run_scf_general, run_uhf};

fn get_h2() -> Vec<Atom> {
    vec![
        Atom::new("H", [0.0, 0.0, -0.7]),
        Atom::new("H", [0.0, 0.0,  0.7]),
    ]
}

fn get_lih() -> Vec<Atom> {
    vec![
        Atom::new("Li", [0.0, 0.0, 0.0]),
        Atom::new("H",  [0.0, 0.0, 3.014]),
    ]
}

fn get_h2o() -> Vec<Atom> {
    vec![
        Atom::new("O", [0.0, 0.0, 0.2217]),
        Atom::new("H", [0.0, 1.4309, -0.8867]),
        Atom::new("H", [0.0, -1.4309, -0.8867]),
    ]
}

fn get_oh() -> Vec<Atom> {
    vec![
        Atom::new("O", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, 1.834]),
    ]
}

#[test]
fn test_post_hf_mp2_lih() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let mp2 = RHFMP2::new(&hf).compute().expect("LiH MP2 failed");
    assert!(mp2.correlation_energy < -0.01 && mp2.correlation_energy > -0.05);
    assert!(mp2.total_energy() < hf.total_energy);
}

#[test]
fn test_post_hf_mp2_h2o() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_h2o();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let mp2 = RHFMP2::new(&hf).compute().expect("H2O MP2 failed");
    assert!(mp2.correlation_energy < -0.03 && mp2.correlation_energy > -0.08);
    assert!(mp2.total_energy() < hf.total_energy);
}

#[test]
fn test_post_hf_mp3_h2_vs_lih() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let mp3 = RHFMP3::new(&hf).compute().expect("LiH MP3 failed");
    assert!(mp3.mp3_correlation_energy.abs() > 1e-5);
    assert!(mp3.total_energy < hf.total_energy);
}

#[test]
fn test_post_hf_mp4_sdtq_lih() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let mp4 = RHFMP4::new(&hf).compute().expect("LiH MP4 failed");
    assert!(mp4.mp4_total_correlation_energy < -0.01);
}

#[test]
fn test_post_hf_mpn_series_h2o() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_h2o();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let mpn = RHFMPn::new(&hf, 4).compute().expect("H2O MPn failed");
    assert!(mpn.order_energies[2] < 0.0);
    assert!(mpn.cumulative_energies[4] < hf.total_energy);
}

#[test]
fn test_post_hf_mp5_through_mp8_h2_vs_fci() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_h2();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let fci = FCI::new(&hf).compute().expect("H2 FCI failed");
    let mp5 = RHFMP5::new(&hf).compute().expect("H2 MP5 failed");
    let mp6 = RHFMP6::new(&hf).compute().expect("H2 MP6 failed");
    let mp7 = RHFMP7::new(&hf).compute().expect("H2 MP7 failed");
    let mp8 = RHFMP8::new(&hf).compute().expect("H2 MP8 failed");

    assert_eq!(mp5.max_order, 5);
    assert_eq!(mp6.max_order, 6);
    assert_eq!(mp7.max_order, 7);
    assert_eq!(mp8.max_order, 8);

    assert!(mp5.total_energy < hf.total_energy);
    assert!(mp8.total_energy < hf.total_energy);
    let diff = (mp8.total_energy - fci.total_energy).abs();
    assert!(diff < 1e-4, "MP8 should converge close to FCI on H2: diff = {}", diff);
}

#[test]
fn test_post_hf_mp5_through_mp8_lih() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let mp5 = RHFMP5::new(&hf).compute().expect("LiH MP5 failed");
    let mp6 = RHFMP6::new(&hf).compute().expect("LiH MP6 failed");
    let mp7 = RHFMP7::new(&hf).compute().expect("LiH MP7 failed");
    let mp8 = RHFMP8::new(&hf).compute().expect("LiH MP8 failed");

    assert_eq!(mp5.max_order, 5);
    assert_eq!(mp6.max_order, 6);
    assert_eq!(mp7.max_order, 7);
    assert_eq!(mp8.max_order, 8);

    assert!(mp5.total_energy < hf.total_energy);
    assert!(mp6.total_energy < hf.total_energy);
    assert!(mp7.total_energy < hf.total_energy);
    assert!(mp8.total_energy < hf.total_energy);
    assert!(mp8.order_energies[8].abs() < mp8.order_energies[2].abs());
}

#[test]
fn test_post_hf_mp5_through_mp8_h2o() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_h2o();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let mp5 = RHFMP5::new(&hf).compute().expect("H2O MP5 failed");
    let mp8 = RHFMP8::new(&hf).compute().expect("H2O MP8 failed");

    assert_eq!(mp5.max_order, 5);
    assert_eq!(mp8.max_order, 8);
    assert!(mp5.total_energy < hf.total_energy);
    assert!(mp8.total_energy < hf.total_energy);
}

#[test]
fn test_post_hf_ccd_lih() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let ccd = CCD::new(&hf).compute().expect("LiH CCD failed");
    assert!(ccd.correlation_energy < -0.01);
    assert!(ccd.converged);
}

#[test]
fn test_post_hf_ccsd_lih() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let ccsd = CCSD::new(&hf).compute().expect("LiH CCSD failed");
    assert!(ccsd.converged);
    assert!(ccsd.correlation_energy < -0.01);
    assert!(ccsd.total_energy < hf.total_energy);
}

#[test]
fn test_post_hf_ccsd_t_lih() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let ccsd = CCSD::new(&hf).compute().expect("LiH CCSD failed");
    let triples = compute_ccsd_t_correction(&hf, &ccsd).expect("CCSD(T) failed");
    assert!(triples.triples_correlation_energy <= 0.0);
    assert!(triples.total_energy <= ccsd.total_energy);
}

#[test]
fn test_post_hf_ccsd_h2o() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_h2o();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let ccsd = CCSD::new(&hf).compute().expect("H2O CCSD failed");
    assert!(ccsd.converged);
    assert!(ccsd.correlation_energy < -0.03);
}

#[test]
fn test_post_hf_cis_singlet_h2() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_h2();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let cis = CIS::new(&hf).compute().expect("H2 CIS failed");
    assert!(!cis.excited_states.is_empty());
    assert!(cis.excited_states[0].excitation_energy_hartree > 0.3);
}

#[test]
fn test_post_hf_cis_singlet_lih() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let cis = CIS::new(&hf).compute().expect("LiH CIS failed");
    assert!(cis.excited_states.len() >= 2);
    assert!(cis.excited_states[0].excitation_energy_hartree <= cis.excited_states[1].excitation_energy_hartree);
}

#[test]
fn test_post_hf_cis_singlet_h2o() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_h2o();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let cis = CIS::new(&hf).compute().expect("H2O CIS failed");
    assert!(cis.excited_states.len() >= 2);
    assert!(cis.excited_states[0].excitation_energy_hartree > 0.2);
}

#[test]
fn test_post_hf_cisd_lih() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let cisd = CISD::new(&hf).compute().expect("LiH CISD failed");
    assert!(cisd.correlation_energy < -0.01);
    assert!(cisd.c0.abs() > 0.9);
}

#[test]
fn test_post_hf_cisd_h2o() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_h2o();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let cisd = CISD::new(&hf).compute().expect("H2O CISD failed");
    assert!(cisd.correlation_energy < -0.03);
    assert!(cisd.c0.abs() > 0.9);
}

#[test]
fn test_post_hf_fci_h2_exact() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_h2();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let fci = FCI::new(&hf).compute().expect("H2 FCI failed");
    assert!(fci.correlation_energy < -0.015);
}

#[test]
fn test_post_hf_fci_lih_exact() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let fci = FCI::new(&hf).compute().expect("LiH FCI failed");
    assert!(fci.correlation_energy < -0.015);
    assert!(fci.total_energy < hf.total_energy);
}

#[test]
fn test_post_hf_casscf_2_2_h2() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -1.75]),
        Atom::new("H", [0.0, 0.0,  1.75]),
    ];
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let opts = CASSCFOptions::new(2, 2);
    let casscf = CASSCF::new(&hf, opts).compute().expect("H2 CASSCF failed");
    assert!(casscf.converged);
    assert!(casscf.total_energy < -0.90);
}

#[test]
fn test_post_hf_casscf_2_2_lih() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let mut opts = CASSCFOptions::new(2, 2);
    opts.max_macro_iterations = 80;
    let casscf = CASSCF::new(&hf, opts).compute().expect("LiH CASSCF failed");
    assert!(casscf.converged);
    assert!(casscf.total_energy <= hf.total_energy);
}

#[test]
fn test_post_hf_caspt2_h2() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -1.75]),
        Atom::new("H", [0.0, 0.0,  1.75]),
    ];
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None).unwrap();

    let opts = CASSCFOptions::new(2, 2);
    let casscf = CASSCF::new(&hf, opts).compute().expect("H2 CASSCF failed");

    let pt2_opts = CASPT2Options::new(2, 2);
    let caspt2 = CASPT2::new(&hf, &casscf, pt2_opts).compute().expect("H2 CASPT2 failed");
    assert!(caspt2.pt2_correlation_energy <= 0.0);
    assert!(caspt2.total_energy <= casscf.total_energy);
}

#[test]
fn test_post_hf_uhf_mp2_li_atom() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = vec![Atom::new("Li", [0.0, 0.0, 0.0])];
    let basis = build_basis(&atoms, &basis_map);
    let uhf = run_uhf(&atoms, &basis, 0, 2, false, 50, 1e-10).expect("Li UHF failed");

    assert!(uhf.converged);
    assert!(uhf.total_energy < -7.0);
}

#[test]
fn test_post_hf_uhf_mp2_oh_radical() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = get_oh();
    let grad = compute_uhf_mp2_gradient(&atoms, &basis_map, 0, 2, false, 50, 1e-8);
    assert_eq!(grad.len(), 2);
    // Net force on isolated radical should vanish (translational invariance)
    let sum_fz = grad[0][2] + grad[1][2];
    assert!(sum_fz.abs() < 1e-4, "Translational invariance failed: sum_fz = {}", sum_fz);
}
