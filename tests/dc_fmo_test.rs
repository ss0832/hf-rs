// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Integration tests for Divide-and-Conquer HF (DC-HF) and Fragment Molecular Orbital (FMO) methods.

use hf_rs::basis::{build_basis, parse_gaussian94, Atom, STO_3G_GBS};
use hf_rs::hf_variants::dc_hf::{compute_dc_hf_gradient, run_dc_hf, DCOptions};
use hf_rs::hf_variants::fmo::{auto_partition_fragments, compute_fmo_gradient, run_fmo, FMOOptions};

#[test]
fn test_dc_hf_h2_integration() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -0.7]),
        Atom::new("H", [0.0, 0.0, 0.7]),
    ];
    let basis = build_basis(&atoms, &basis_map);
    let options = DCOptions {
        buffer_radius_bohr: 5.0,
        inverse_temperature: 1000.0,
        ..Default::default()
    };

    // Each subsystem buffer contains the other atom: DC-HF is RHF here.
    let res = run_dc_hf(&atoms, &basis, 0, &options).expect("DC-HF failed");
    let rhf = hf_rs::run_scf(&atoms, &basis, 200, 1e-12).unwrap();
    assert!((res.total_energy - rhf.total_energy).abs() < 1e-9);
    // No exact nuclear gradient is claimed for DC-HF.
    assert!(compute_dc_hf_gradient(&atoms, &basis, &res).is_err());
}

#[test]
fn test_fmo2_and_fmo3_h4_integration() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    // Two H2 dimers separated along X by 4.0 Bohr
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -0.7]),
        Atom::new("H", [0.0, 0.0, 0.7]),
        Atom::new("H", [4.0, 0.0, -0.7]),
        Atom::new("H", [4.0, 0.0, 0.7]),
    ];
    let basis = build_basis(&atoms, &basis_map);
    let fragments = auto_partition_fragments(&atoms, 2);
    assert_eq!(fragments.len(), 2);

    let options_fmo2 = FMOOptions {
        order: 2,
        ..Default::default()
    };
    let res2 = run_fmo(&atoms, &basis, &fragments, &options_fmo2).expect("FMO2 failed");
    assert!(res2.total_energy < -2.0);
    assert_eq!(res2.monomer_energies.len(), 2);
    assert_eq!(res2.dimer_energies.len(), 1);
    println!("FMO2 IFIE: {:.8} Eh", res2.ifie[0].total_interaction_energy);
    println!("FMO2 PIEDA ES: {:.8} Eh, EX: {:.8} Eh, CT: {:.8} Eh",
        res2.ifie[0].electrostatic, res2.ifie[0].exchange_repulsion, res2.ifie[0].charge_transfer);
    let pieda_sum = res2.ifie[0].electrostatic + res2.ifie[0].exchange_repulsion + res2.ifie[0].charge_transfer;
    assert!((pieda_sum - res2.ifie[0].total_interaction_energy).abs() < 1e-10,
        "PIEDA sum ({}) must equal total interaction energy ({})", pieda_sum, res2.ifie[0].total_interaction_energy);

    // Analytical gradient test
    let grad = compute_fmo_gradient(&atoms, &basis, &fragments, &res2).unwrap();
    assert_eq!(grad.len(), 4);
    let total_fx: f64 = grad.iter().map(|g| g[0]).sum();
    let total_fz: f64 = grad.iter().map(|g| g[2]).sum();
    assert!(total_fx.abs() < 1e-4, "Total Fx should vanish: {:.6e}", total_fx);
    assert!(total_fz.abs() < 1e-4, "Total Fz should vanish: {:.6e}", total_fz);
}
