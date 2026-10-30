// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Comprehensive test suite for Organometallic Complexes.
//!
//! Covers:
//! 1. Methyllithium (CH3-Li): Classic organo-alkali complex with strong carbanionic C-Li bond.
//! 2. Methyllithium Dimer ((CH3-Li)2): Multi-center bridging 2e-3c Li-C-Li bonding.
//! 3. Dimethylberyllium ((CH3)2-Be): Group 2 linear organometallic compound (C-Be-C).
//! 4. Methylmagnesium Hydride (CH3-Mg-H): Grignard reagent organomagnesium model.
//! 5. Methyltitanium Trihydride (CH3-TiH3): Early transition-metal organometallic complex featuring Ti 3d-orbitals.

use hf_rs::basis::{Atom, STO_3G_GBS, build_basis, parse_gaussian94};
use hf_rs::diis::ConvAlgo;
use hf_rs::gradient::compute_rhf_gradient;
use hf_rs::post_hf::PostHFMethod;
use hf_rs::post_hf::mp2::RHFMP2;
use hf_rs::scf::run_scf_general;

#[test]
fn test_methyllithium_ch3li_organometallic() {
    let basis_map = parse_gaussian94(STO_3G_GBS);

    // Methyllithium CH3-Li: C at origin, Li along z-axis, 3 H atoms in C3v symmetry
    // C-Li distance: ~1.97 Angstrom = 3.73 Bohr
    // C-H distance: ~1.10 Angstrom = 2.08 Bohr
    let c_pos = [0.0, 0.0, 0.0];
    let li_pos = [0.0, 0.0, 3.73];
    let h1_pos = [1.95, 0.0, -0.73];
    let h2_pos = [-0.975, 1.6887, -0.73];
    let h3_pos = [-0.975, -1.6887, -0.73];

    let atoms = vec![
        Atom::new("C", c_pos),
        Atom::new("Li", li_pos),
        Atom::new("H", h1_pos),
        Atom::new("H", h2_pos),
        Atom::new("H", h3_pos),
    ];

    let basis = build_basis(&atoms, &basis_map);
    assert_eq!(basis.len(), 5 + 5 + 1 + 1 + 1); // C(5), Li(5), 3*H(1) = 15 basis functions

    // 1. RHF SCF calculation
    let hf_res = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None)
        .expect("Methyllithium RHF SCF converged");

    println!("CH3Li RHF Total Energy: {:.8} Hartree", hf_res.total_energy);
    // Typical CH3Li STO-3G RHF energy is around -46.5 to -47.0 Hartree
    assert!(hf_res.total_energy < -46.0 && hf_res.total_energy > -48.0);
    assert!(hf_res.converged);

    // 2. Dipole moment: Carbanionic C^{\delta-} - Li^{\delta+} bond has large permanent dipole along z
    let dipole_total_debye = hf_res.properties.dipole_magnitude_debye;
    let dipole_z = hf_res.properties.dipole_moment_debye[2];
    println!(
        "CH3Li Dipole Moment: {:.4} Debye (z-component: {:.4})",
        dipole_total_debye, dipole_z
    );
    assert!(
        dipole_total_debye > 3.0,
        "CH3Li must have a strong permanent dipole due to polar C-Li bond"
    );
    assert!(
        dipole_z.abs() > 3.0,
        "Dipole must lie predominantly along C-Li axis (z)"
    );

    // 3. Analytical nuclear gradient
    let grad = compute_rhf_gradient(&atoms, &basis, &hf_res);
    let mut net_force = [0.0; 3];
    for g_a in &grad {
        net_force[0] += g_a[0];
        net_force[1] += g_a[1];
        net_force[2] += g_a[2];
    }
    println!(
        "CH3Li Net Force (Translational Invariance): {:?}",
        net_force
    );
    assert!(net_force[0].abs() < 1e-6);
    assert!(net_force[1].abs() < 1e-6);
    assert!(net_force[2].abs() < 1e-6);

    // 4. Post-HF MP2 correlation
    let mp2_res = RHFMP2::new(&hf_res).compute().expect("CH3Li MP2 converged");
    println!(
        "CH3Li MP2 Correlation Energy: {:.8} Hartree",
        mp2_res.correlation_energy
    );
    assert!(
        mp2_res.correlation_energy < -0.05,
        "MP2 correlation energy must be negative"
    );
}

#[test]
fn test_methyllithium_dimer_bridging_organometallic() {
    let basis_map = parse_gaussian94(STO_3G_GBS);

    // (CH3Li)2 dimer: Planar Li2C2 rhombus with 2e-3c bridging C-Li-C bonds
    // Li atoms at (+- 3.7 Bohr, 0, 0), C atoms at (0, +- 2.8 Bohr, 0)
    let atoms = vec![
        Atom::new("Li", [3.7, 0.0, 0.0]),
        Atom::new("Li", [-3.7, 0.0, 0.0]),
        Atom::new("C", [0.0, 2.8, 0.0]),
        Atom::new("C", [0.0, -2.8, 0.0]),
        Atom::new("H", [0.0, 2.8, 2.05]),
        Atom::new("H", [0.0, 2.8, -2.05]),
        Atom::new("H", [1.7, 2.8, 0.0]),
        Atom::new("H", [0.0, -2.8, 2.05]),
        Atom::new("H", [0.0, -2.8, -2.05]),
        Atom::new("H", [-1.7, -2.8, 0.0]),
    ];

    let basis = build_basis(&atoms, &basis_map);
    let hf_res = run_scf_general(&atoms, &basis, 0, 60, 1e-9, ConvAlgo::Adiissoscf, None)
        .expect("(CH3Li)2 dimer RHF SCF converged");

    println!(
        "(CH3Li)2 Dimer Total Energy: {:.8} Hartree",
        hf_res.total_energy
    );
    // Dimer energy is approximately 2 * E(monomer) - E_binding (~ -92.37 Hartree)
    assert!(hf_res.total_energy < -92.0 && hf_res.total_energy > -94.0);
    assert!(hf_res.converged);
}

#[test]
fn test_dimethylberyllium_organoberyllium() {
    let basis_map = parse_gaussian94(STO_3G_GBS);

    // Dimethylberyllium (CH3)2Be: Linear C-Be-C framework (D3d / D3h)
    // Be at origin, C at (0, 0, +- 3.2 Bohr)
    let atoms = vec![
        Atom::new("Be", [0.0, 0.0, 0.0]),
        Atom::new("C", [0.0, 0.0, 3.2]),
        Atom::new("C", [0.0, 0.0, -3.2]),
        Atom::new("H", [1.95, 0.0, 3.9]),
        Atom::new("H", [-0.975, 1.6887, 3.9]),
        Atom::new("H", [-0.975, -1.6887, 3.9]),
        Atom::new("H", [1.95, 0.0, -3.9]),
        Atom::new("H", [-0.975, 1.6887, -3.9]),
        Atom::new("H", [-0.975, -1.6887, -3.9]),
    ];

    let basis = build_basis(&atoms, &basis_map);
    let hf_res = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None)
        .expect("(CH3)2Be RHF SCF converged");

    println!("(CH3)2Be Total Energy: {:.8} Hartree", hf_res.total_energy);
    assert!(hf_res.converged);
    assert!(hf_res.total_energy < -89.0 && hf_res.total_energy > -93.0);

    // By inversion/linear symmetry, dipole moment must be zero
    let dipole_total = hf_res.properties.dipole_magnitude_debye;
    println!("(CH3)2Be Dipole: {:.6} Debye", dipole_total);
    assert!(
        dipole_total < 1e-4,
        "Centrosymmetric (CH3)2Be must have zero dipole moment"
    );
}

#[test]
fn test_methylmagnesium_grignard_model() {
    let basis_map = parse_gaussian94(STO_3G_GBS);

    // CH3-Mg-H: Classic Grignard model complex with C-Mg organometallic bond
    // Mg at origin, C along +z (3.9 Bohr), H-hydride along -z (-3.2 Bohr)
    let atoms = vec![
        Atom::new("Mg", [0.0, 0.0, 0.0]),
        Atom::new("C", [0.0, 0.0, 3.9]),
        Atom::new("H", [0.0, 0.0, -3.2]),
        Atom::new("H", [1.95, 0.0, 4.6]),
        Atom::new("H", [-0.975, 1.6887, 4.6]),
        Atom::new("H", [-0.975, -1.6887, 4.6]),
    ];

    let basis = build_basis(&atoms, &basis_map);
    let hf_res = run_scf_general(&atoms, &basis, 0, 50, 1e-10, ConvAlgo::Adiissoscf, None)
        .expect("CH3MgH Grignard model RHF SCF converged");

    println!("CH3MgH Total Energy: {:.8} Hartree", hf_res.total_energy);
    assert!(hf_res.converged);
    assert!(hf_res.total_energy < -235.0 && hf_res.total_energy > -242.0);

    // Check dipole moment (opposing polarizations C-Mg and Mg-H yield net ~0.66 Debye)
    let dipole_total = hf_res.properties.dipole_magnitude_debye;
    println!("CH3MgH Dipole: {:.4} Debye", dipole_total);
    assert!(dipole_total > 0.5);
}

#[test]
fn test_methyltitanium_trihydride_transition_metal() {
    let basis_map = parse_gaussian94(STO_3G_GBS);

    // CH3-TiH3: Early transition-metal organometallic complex containing Titanium 3d-orbitals
    // Ti at origin, C along +z (3.9 Bohr)
    let atoms = vec![
        Atom::new("Ti", [0.0, 0.0, 0.0]),
        Atom::new("C", [0.0, 0.0, 3.9]),
        Atom::new("H", [2.5, 0.0, -1.5]),
        Atom::new("H", [-1.25, 2.165, -1.5]),
        Atom::new("H", [-1.25, -2.165, -1.5]),
        Atom::new("H", [1.95, 0.0, 4.6]),
        Atom::new("H", [-0.975, 1.6887, 4.6]),
        Atom::new("H", [-0.975, -1.6887, 4.6]),
    ];

    let basis = build_basis(&atoms, &basis_map);
    println!("CH3TiH3 basis functions count: {}", basis.len());

    let hf_res = run_scf_general(&atoms, &basis, 0, 60, 1e-8, ConvAlgo::Adiissoscf, None)
        .expect("CH3TiH3 Transition-metal organometallic SCF converged");

    println!("CH3TiH3 Total Energy: {:.8} Hartree", hf_res.total_energy);
    assert!(hf_res.converged);
    assert!(hf_res.total_energy < -870.0 && hf_res.total_energy > -920.0);
}

#[test]
fn test_cyclopentadienyl_lithium_cpli_half_sandwich() {
    let basis_map = parse_gaussian94(STO_3G_GBS);

    // Cyclopentadienyl lithium Cp-Li (C5H5-Li): classic half-sandwich organometallic complex with \eta^5 hapticity.
    // Li is positioned on the C5 rotational axis above the planar cyclopentadienyl ring.
    // Pentagon radius: ~1.20 Angstrom = 2.268 Bohr
    // C-H bond length: ~1.08 Angstrom = 2.041 Bohr (r_H = 4.309 Bohr)
    // Li height above ring: ~1.85 Angstrom = 3.496 Bohr
    use std::f64::consts::PI;
    let r_c = 2.268;
    let r_h = 4.309;
    let mut atoms = Vec::new();
    for k in 0..5 {
        let theta = 2.0 * PI * (k as f64) / 5.0;
        atoms.push(Atom::new("C", [r_c * theta.cos(), r_c * theta.sin(), 0.0]));
    }
    for k in 0..5 {
        let theta = 2.0 * PI * (k as f64) / 5.0;
        atoms.push(Atom::new("H", [r_h * theta.cos(), r_h * theta.sin(), 0.0]));
    }
    atoms.push(Atom::new("Li", [0.0, 0.0, 3.496]));

    let basis = build_basis(&atoms, &basis_map);
    assert_eq!(basis.len(), 5 * 5 + 5 + 5); // 35 basis functions

    let hf_res = run_scf_general(&atoms, &basis, 0, 50, 1e-8, ConvAlgo::Adiissoscf, None)
        .expect("CpLi Half-sandwich organometallic SCF converged");

    println!("CpLi Total Energy: {:.8} Hartree", hf_res.total_energy);
    assert!(hf_res.converged);
    assert!(hf_res.total_energy < -195.0 && hf_res.total_energy > -202.0);

    // Compare all components in atomic units, independent of Debye conversion conventions.
    let dipole_total = hf_res.properties.dipole_magnitude_debye;
    let dipole_z = hf_res.properties.dipole_moment_debye[2];
    println!(
        "CpLi Dipole Moment: {:.4} Debye (z-component: {:.4})",
        dipole_total, dipole_z
    );
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/organometallic_reference.json")).unwrap();
    let expected = &oracle["systems"]["cpli"];
    assert!((hf_res.total_energy - expected["energy"].as_f64().unwrap()).abs() < 1e-8);
    for axis in 0..3 {
        assert!(
            (hf_res.properties.dipole_moment_au[axis]
                - expected["dipole_au"][axis].as_f64().unwrap())
            .abs()
                < 1e-8
        );
    }

    let start_grad = std::time::Instant::now();
    let grad = compute_rhf_gradient(&atoms, &basis, &hf_res);
    println!("CpLi Gradient finished in {:?}", start_grad.elapsed());
    let mut sum_f = [0.0; 3];
    for g in &grad {
        sum_f[0] += g[0];
        sum_f[1] += g[1];
        sum_f[2] += g[2];
    }
    println!("CpLi Net Force: {:?}", sum_f);
    assert!(sum_f[0].abs() < 1e-4);
    assert!(sum_f[1].abs() < 1e-4);
    assert!(sum_f[2].abs() < 1e-4);
}

#[test]
fn test_ferrocene_sandwich_complex() {
    let basis_map = parse_gaussian94(STO_3G_GBS);

    // Ferrocene Fe(\eta^5-C5H5)2: Prototypical transition-metal sandwich complex.
    // Conformation: Staggered D5d symmetry with Fe at origin [0, 0, 0].
    // Top Cp ring at z = +3.137 Bohr (+1.66 Angstrom)
    // Bottom Cp ring at z = -3.137 Bohr (-1.66 Angstrom, staggered by 36 degrees = PI/5)
    // C-C ring radius: r_c = 2.268 Bohr (~1.20 Angstrom)
    // C-H ring radius: r_h = 4.309 Bohr (~2.28 Angstrom)
    use std::f64::consts::PI;
    let r_c = 2.268;
    let r_h = 4.309;
    let z_ring = 3.137;

    let mut atoms = Vec::new();
    // Central Iron atom
    atoms.push(Atom::new("Fe", [0.0, 0.0, 0.0]));

    // Top Cp ring (k = 0..5)
    for k in 0..5 {
        let theta = 2.0 * PI * (k as f64) / 5.0;
        atoms.push(Atom::new(
            "C",
            [r_c * theta.cos(), r_c * theta.sin(), z_ring],
        ));
        atoms.push(Atom::new(
            "H",
            [r_h * theta.cos(), r_h * theta.sin(), z_ring],
        ));
    }

    // Bottom Cp ring (staggered by PI / 5.0)
    for k in 0..5 {
        let theta = 2.0 * PI * (k as f64) / 5.0 + (PI / 5.0);
        atoms.push(Atom::new(
            "C",
            [r_c * theta.cos(), r_c * theta.sin(), -z_ring],
        ));
        atoms.push(Atom::new(
            "H",
            [r_h * theta.cos(), r_h * theta.sin(), -z_ring],
        ));
    }

    assert_eq!(atoms.len(), 21); // 1 Fe + 10 C + 10 H

    let basis = build_basis(&atoms, &basis_map);
    // Cartesian STO-3G Fe has 19 functions; the two Cp rings contribute 60.
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/organometallic_reference.json")).unwrap();
    assert_eq!(
        basis.len(),
        oracle["systems"]["ferrocene"]["n_ao"].as_u64().unwrap() as usize
    );

    let start_scf = std::time::Instant::now();
    let hf_res = run_scf_general(&atoms, &basis, 0, 80, 1e-6, ConvAlgo::Cdiis, None)
        .expect("Ferrocene sandwich complex SCF converged");
    println!("Ferrocene SCF finished in {:?}", start_scf.elapsed());

    println!("Ferrocene Total Energy: {:.8} Hartree", hf_res.total_energy);
    assert!(hf_res.converged);
    // STO-3G Ferrocene RHF ground state energy is approximately -1621.80 Hartree (Fe: -1245.7 Eh, 2*Cp: -376.1 Eh)
    assert!(hf_res.total_energy < -1600.0 && hf_res.total_energy > -1640.0);

    // Staggered D5d Ferrocene has an inversion center (Ci):
    // The dipole moment MUST vanish identically by point-group symmetry!
    let dipole_mag = hf_res.properties.dipole_magnitude_debye;
    println!("Ferrocene Dipole Moment: {:.6} Debye", dipole_mag);
    assert!(
        dipole_mag < 1e-3,
        "Centrosymmetric D5d Ferrocene must have zero dipole moment: {}",
        dipole_mag
    );

    // Analytical gradient translational invariance
    let start_grad = std::time::Instant::now();
    let grad = compute_rhf_gradient(&atoms, &basis, &hf_res);
    println!("Ferrocene Gradient finished in {:?}", start_grad.elapsed());
    let mut sum_f = [0.0; 3];
    for g in &grad {
        sum_f[0] += g[0];
        sum_f[1] += g[1];
        sum_f[2] += g[2];
    }
    println!("Ferrocene Net Force: {:?}", sum_f);
    assert!(sum_f[0].abs() < 1e-4);
    assert!(sum_f[1].abs() < 1e-4);
    assert!(sum_f[2].abs() < 1e-4);
}
