// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Tests for Post-Hartree-Fock MP2 calculations (RHF-MP2, UHF-MP2, SCS-MP2, and 1-RDM).

use hf_rs::*;

#[test]
fn test_h2_sto3g_rhf_and_uhf_mp2() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, 1.4]),
    ];
    let basis = build_basis(&atoms, &basis_map);

    // 1. RHF calculation
    let hf_res = run_scf(&atoms, &basis, 50, 1e-12).expect("RHF SCF failed");
    let mp2_res = RHFMP2::new(&hf_res)
        .with_rdm(true)
        .compute()
        .expect("RHF-MP2 computation failed");

    println!("H2 RHF Energy: {:.10} Eh", mp2_res.reference_energy());
    println!("H2 MP2 Corr Energy: {:.10} Eh", mp2_res.correlation_energy());
    println!("H2 Total MP2 Energy: {:.10} Eh", mp2_res.total_energy());
    println!("H2 SCS-MP2 Energy: {:.10} Eh", mp2_res.e_scs);

    // Correlation energy is negative
    assert!(mp2_res.correlation_energy() < 0.0);
    assert!(mp2_res.correlation_energy() > -0.1);

    // Check 1-RDM trace equals 2 electrons
    let rdm = mp2_res.one_rdm().expect("Missing 1-RDM");
    let trace: f64 = (0..rdm.rows).map(|i| rdm[(i, i)]).sum();
    println!("H2 1-RDM Trace: {:.8} (expected 2.0)", trace);
    assert!((trace - 2.0).abs() < 1e-6);

    // Natural orbital occupations
    let occs = mp2_res.natural_orbital_occupations().expect("Missing natural occupations");
    println!("H2 Natural Orbital Occupations: {:?}", occs);
    let occ_sum: f64 = occs.iter().sum();
    assert!((occ_sum - 2.0).abs() < 1e-6);

    // 2. UHF calculation (closed-shell singlet with UHF)
    let uhf_res = run_uhf(&atoms, &basis, 0, 1, false, 50, 1e-12).expect("UHF SCF failed");
    let uhf_mp2_res = UHFMP2::new(&uhf_res)
        .compute()
        .expect("UHF-MP2 computation failed");

    println!("H2 UHF-MP2 Corr Energy: {:.10} Eh", uhf_mp2_res.correlation_energy());

    // For a closed-shell singlet, RHF-MP2 and UHF-MP2 must give the exact same correlation energy
    assert!(
        (mp2_res.correlation_energy() - uhf_mp2_res.correlation_energy()).abs() < 1e-8,
        "RHF-MP2 ({}) and UHF-MP2 ({}) correlation energies must match",
        mp2_res.correlation_energy(),
        uhf_mp2_res.correlation_energy()
    );
}

#[test]
fn test_water_sto3g_mp2_all_electron_and_frozen_core() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let r_oh = 0.957; // Angstrom approx 1.8087 Bohr
    let angle_deg = 104.5;
    let rad = angle_deg * std::f64::consts::PI / 180.0;
    let z_h = r_oh * (rad / 2.0).cos();
    let y_h = r_oh * (rad / 2.0).sin();

    let atoms = vec![
        Atom::new("O", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0,  y_h, z_h]),
        Atom::new("H", [0.0, -y_h, z_h]),
    ];
    let basis = build_basis(&atoms, &basis_map);

    let hf_res = run_scf(&atoms, &basis, 50, 1e-12).expect("Water SCF failed");

    // All-electron MP2
    let ae_mp2 = RHFMP2::new(&hf_res).compute().expect("AE-MP2 failed");
    println!("H2O STO-3G RHF Energy: {:.8} Eh", ae_mp2.reference_energy());
    println!("H2O STO-3G AE-MP2 Corr Energy: {:.8} Eh", ae_mp2.correlation_energy());
    println!("H2O STO-3G AE-MP2 Total Energy: {:.8} Eh", ae_mp2.total_energy());

    // Frozen-core MP2 (freeze 1s orbital of Oxygen, n_core = 1)
    let fc_mp2 = RHFMP2::new(&hf_res)
        .with_frozen_core(1)
        .compute()
        .expect("FC-MP2 failed");
    println!("H2O STO-3G FC-MP2 Corr Energy: {:.8} Eh", fc_mp2.correlation_energy());

    // Correlation energy is negative
    assert!(ae_mp2.correlation_energy() < 0.0);
    assert!(fc_mp2.correlation_energy() < 0.0);

    // Frozen-core correlation energy magnitude must be slightly smaller than all-electron (|E_fc| < |E_ae|)
    assert!(fc_mp2.correlation_energy() > ae_mp2.correlation_energy());
}

#[test]
fn test_h2_rhf_and_uhf_mp2_hessian() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let r = 1.4; // Bohr
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -r / 2.0]),
        Atom::new("H", [0.0, 0.0,  r / 2.0]),
    ];

    // 1. Closed-shell RHF-MP2 Hessian
    let hess_rhf = compute_mp2_hessian(&atoms, &basis_map, 50, 1e-12);
    assert_eq!(hess_rhf.rows, 6);
    assert_eq!(hess_rhf.cols, 6);

    // Verify symmetry: H = H^T
    for i in 0..6 {
        for j in 0..6 {
            assert!(
                (hess_rhf[(i, j)] - hess_rhf[(j, i)]).abs() < 1e-10,
                "Hessian not symmetric at ({}, {})",
                i, j
            );
        }
    }

    // Verify translational invariance along z axis: H[2, 2] + H[2, 5] == 0
    let k_zz = hess_rhf[(2, 2)];
    let k_zz_off = hess_rhf[(2, 5)];
    println!("H2 RHF-MP2 k_zz: {:.8} Hartree/Bohr^2", k_zz);
    assert!(k_zz > 0.0, "Stretching force constant must be positive");
    assert!(
        (k_zz + k_zz_off).abs() < 1e-4,
        "Translational invariance violation: k_zz + k_zz_off = {}",
        k_zz + k_zz_off
    );

    // Vibrational frequencies
    let (freqs_rhf, _, zpe_rhf) = compute_mp2_vibrational_frequencies(&atoms, &hess_rhf);
    println!("H2 RHF-MP2 Harmonic Frequencies (cm^-1): {:?}", freqs_rhf);
    println!("H2 RHF-MP2 ZPE: {:.6} Hartree ({:.2} kcal/mol)", zpe_rhf, zpe_rhf * 627.509);

    // The highest frequency is the H-H stretch
    let stretch_freq = freqs_rhf.last().unwrap();
    assert!(*stretch_freq > 4500.0 && *stretch_freq < 5100.0, "Unexpected H-H stretch frequency: {}", stretch_freq);

    // Verify complete post-HF Hessian analysis with both raw and projected Hessians
    let analysis = compute_post_hf_hessian_analysis(&atoms, &hess_rhf);
    assert_eq!(analysis.raw_hessian.rows, 6);
    assert_eq!(analysis.projected_hessian.rows, 6);
    // In raw frequencies, rotational modes are ~1000 cm^-1 (not removed in raw)
    assert!(analysis.raw_frequencies[3] > 500.0, "Raw rotation mode 4 should be non-zero in raw Hessian: {}", analysis.raw_frequencies[3]);
    assert!(analysis.raw_frequencies[4] > 500.0, "Raw rotation mode 5 should be non-zero in raw Hessian: {}", analysis.raw_frequencies[4]);
    // In projected frequencies, all 5 translational and rotational modes are strictly zero!
    for m in 0..5 {
        assert!(
            analysis.projected_frequencies[m].abs() < 1e-4,
            "Projected translation/rotation mode {} must be zero: {}",
            m + 1,
            analysis.projected_frequencies[m]
        );
    }
    assert!((analysis.projected_frequencies[5] - stretch_freq).abs() < 1e-4);

    // 2. Open-shell UHF-MP2 Hessian
    let hess_uhf = compute_uhf_mp2_hessian(&atoms, &basis_map, 0, 1, false, 50, 1e-12);
    assert_eq!(hess_uhf.rows, 6);
    assert_eq!(hess_uhf.cols, 6);

    // Compare RHF-MP2 and UHF-MP2 Hessians
    let diff = hess_rhf.rmsd(&hess_uhf);
    println!("RMSD between RHF-MP2 and UHF-MP2 Hessians: {:.10} Hartree/Bohr^2", diff);
    assert!(
        diff < 1e-5,
        "RHF-MP2 and UHF-MP2 Hessians must agree for closed-shell singlet (RMSD = {})",
        diff
    );

    let (freqs_uhf, _, _) = compute_mp2_vibrational_frequencies(&atoms, &hess_uhf);
    let stretch_freq_uhf = freqs_uhf.last().unwrap();
    assert!(
        (stretch_freq - stretch_freq_uhf).abs() < 1.0,
        "Vibrational frequency mismatch between RHF-MP2 ({}) and UHF-MP2 ({})",
        stretch_freq,
        stretch_freq_uhf
    );
}

#[test]
fn test_water_mp2_hessian_and_frequencies() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let r_oh = 1.8087; // Bohr
    let angle_rad = 104.5 * std::f64::consts::PI / 180.0;
    let z_h = r_oh * (angle_rad / 2.0).cos();
    let y_h = r_oh * (angle_rad / 2.0).sin();

    let atoms = vec![
        Atom::new("O", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0,  y_h, z_h]),
        Atom::new("H", [0.0, -y_h, z_h]),
    ];

    let hess = compute_mp2_hessian(&atoms, &basis_map, 50, 1e-12);
    assert_eq!(hess.rows, 9);
    assert_eq!(hess.cols, 9);

    // Check symmetry
    for i in 0..9 {
        for j in 0..9 {
            assert!((hess[(i, j)] - hess[(j, i)]).abs() < 1e-10);
        }
    }

    let (freqs, modes, zpe) = compute_mp2_vibrational_frequencies(&atoms, &hess);
    println!("H2O MP2 Harmonic Frequencies (cm^-1): {:?}", freqs);
    println!("H2O MP2 Normal Modes count: {}", modes.len());
    println!("H2O MP2 ZPE: {:.6} Hartree ({:.2} kcal/mol)", zpe, zpe * 627.509);

    assert_eq!(modes.len(), 9);
    assert!(zpe > 0.01 && zpe < 0.05, "ZPE out of expected range: {}", zpe);
}

