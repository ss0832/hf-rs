// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

#![allow(clippy::needless_range_loop)]

use hf_rs::*;

#[test]
fn test_h2_sto3g_energy_and_gradient() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);

    // H2 molecule at R = 1.4 Bohr (approx 0.74 Angstrom)
    // Oriented along z axis
    let r = 1.4;
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -r / 2.0]),
        Atom::new("H", [0.0, 0.0,  r / 2.0]),
    ];

    let basis = build_basis(&atoms, &basis_map);
    assert_eq!(basis.len(), 2);

    let hf_res = run_scf(&atoms, &basis, 50, 1e-12).expect("SCF failed");

    println!("H2 STO-3G Total Energy: {:.10} Eh", hf_res.total_energy);
    // Standard STO-3G H2 energy at R=1.4 Bohr is approx -1.1167143 Eh
    assert!((hf_res.total_energy - (-1.1167143)).abs() < 1e-5);

    // Compute Analytical Gradient
    let ana_grad = compute_rhf_gradient(&atoms, &basis, &hf_res);

    println!("Analytical Gradient:");
    for (i, g) in ana_grad.iter().enumerate() {
        println!("Atom {}: [{:12.8}, {:12.8}, {:12.8}]", i, g[0], g[1], g[2]);
    }

    // Translational invariance test: sum of gradients must be zero
    let sum_gx: f64 = ana_grad.iter().map(|g| g[0]).sum();
    let sum_gy: f64 = ana_grad.iter().map(|g| g[1]).sum();
    let sum_gz: f64 = ana_grad.iter().map(|g| g[2]).sum();
    assert!(sum_gx.abs() < 1e-8, "Sum Gx != 0: {}", sum_gx);
    assert!(sum_gy.abs() < 1e-8, "Sum Gy != 0: {}", sum_gy);
    assert!(sum_gz.abs() < 1e-8, "Sum Gz != 0: {}", sum_gz);

    // Compute Numerical Gradient by Central Difference
    let h = 1e-4; // step size in Bohr
    for a in 0..atoms.len() {
        for tau in 0..3 {
            // +h
            let mut atoms_plus = atoms.clone();
            atoms_plus[a].position[tau] += h;
            let basis_plus = build_basis(&atoms_plus, &basis_map);
            let res_plus = run_scf(&atoms_plus, &basis_plus, 50, 1e-12).unwrap();

            // -h
            let mut atoms_minus = atoms.clone();
            atoms_minus[a].position[tau] -= h;
            let basis_minus = build_basis(&atoms_minus, &basis_map);
            let res_minus = run_scf(&atoms_minus, &basis_minus, 50, 1e-12).unwrap();

            let num_grad = (res_plus.total_energy - res_minus.total_energy) / (2.0 * h);
            let diff = (ana_grad[a][tau] - num_grad).abs();

            println!(
                "Atom {} axis {}: Analytical = {:12.8}, Numerical = {:12.8}, Diff = {:10.2e}",
                a, tau, ana_grad[a][tau], num_grad, diff
            );

            assert!(
                diff < 1e-6,
                "Gradient mismatch for atom {} axis {}: analytical={}, numerical={}, diff={}",
                a, tau, ana_grad[a][tau], num_grad, diff
            );
        }
    }
}

#[test]
fn test_water_sto3g_energy_and_gradient() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);

    // H2O geometry in Angstrom (experimental approx: r_OH = 0.9578 A, angle = 104.5 deg)
    let angle_rad = 104.5_f64.to_radians();
    let r_ang = 0.9578;
    let h1_x = r_ang * (angle_rad / 2.0).sin();
    let h1_z = r_ang * (angle_rad / 2.0).cos();

    let atoms = vec![
        Atom::from_angstrom("O", [0.0, 0.0, 0.0]),
        Atom::from_angstrom("H", [h1_x, 0.0, h1_z]),
        Atom::from_angstrom("H", [-h1_x, 0.0, h1_z]),
    ];

    let basis = build_basis(&atoms, &basis_map);
    // STO-3G for H2O: O has 1s, 2s, 2px, 2py, 2pz (5 basis functions), each H has 1s (1 basis function) -> Total 7
    assert_eq!(basis.len(), 7);

    let hf_res = run_scf(&atoms, &basis, 50, 1e-12).expect("SCF failed");

    println!("H2O STO-3G Total Energy: {:.10} Eh", hf_res.total_energy);
    // Literature STO-3G H2O near equilibrium: approx -74.96 Eh
    assert!((hf_res.total_energy - (-74.96)).abs() < 0.1);

    // Analytical Gradient
    let ana_grad = compute_rhf_gradient(&atoms, &basis, &hf_res);

    println!("H2O Analytical Gradient (Hartree/Bohr):");
    for (i, g) in ana_grad.iter().enumerate() {
        println!("Atom {} ({}): [{:12.8}, {:12.8}, {:12.8}]", i, atoms[i].symbol, g[0], g[1], g[2]);
    }

    // Translational invariance test: sum of gradients must be zero
    let sum_gx: f64 = ana_grad.iter().map(|g| g[0]).sum();
    let sum_gy: f64 = ana_grad.iter().map(|g| g[1]).sum();
    let sum_gz: f64 = ana_grad.iter().map(|g| g[2]).sum();
    assert!(sum_gx.abs() < 1e-6, "Sum Gx != 0: {}", sum_gx);
    assert!(sum_gy.abs() < 1e-6, "Sum Gy != 0: {}", sum_gy);
    assert!(sum_gz.abs() < 1e-6, "Sum Gz != 0: {}", sum_gz);

    // Numerical gradient comparison for all 9 coordinates
    let h = 1e-4; // step size in Bohr
    for a in 0..atoms.len() {
        for tau in 0..3 {
            let mut atoms_plus = atoms.clone();
            atoms_plus[a].position[tau] += h;
            let basis_plus = build_basis(&atoms_plus, &basis_map);
            let res_plus = run_scf(&atoms_plus, &basis_plus, 50, 1e-12).unwrap();

            let mut atoms_minus = atoms.clone();
            atoms_minus[a].position[tau] -= h;
            let basis_minus = build_basis(&atoms_minus, &basis_map);
            let res_minus = run_scf(&atoms_minus, &basis_minus, 50, 1e-12).unwrap();

            let num_grad = (res_plus.total_energy - res_minus.total_energy) / (2.0 * h);
            let diff = (ana_grad[a][tau] - num_grad).abs();

            println!(
                "H2O Atom {} ({}) axis {}: Analytical = {:12.8}, Numerical = {:12.8}, Diff = {:10.2e}",
                a, atoms[a].symbol, tau, ana_grad[a][tau], num_grad, diff
            );

            assert!(
                diff < 1e-5,
                "Gradient mismatch for atom {} axis {}: analytical={}, numerical={}, diff={}",
                a, tau, ana_grad[a][tau], num_grad, diff
            );
        }
    }
}

#[test]
fn test_water_6_31g_energy_and_gradient() {
    let basis_map = parse_gaussian94(basis::BASIS_6_31G_GBS);

    let angle_rad = 104.5_f64.to_radians();
    let r_ang = 0.9578;
    let h1_x = r_ang * (angle_rad / 2.0).sin();
    let h1_z = r_ang * (angle_rad / 2.0).cos();

    let atoms = vec![
        Atom::from_angstrom("O", [0.0, 0.0, 0.0]),
        Atom::from_angstrom("H", [h1_x, 0.0, h1_z]),
        Atom::from_angstrom("H", [-h1_x, 0.0, h1_z]),
    ];

    let basis = build_basis(&atoms, &basis_map);
    // 6-31G for H2O: O has 1s, 2s, 2p (inner), 3s, 3p (outer) -> 1 + 1 + 3 + 1 + 3 = 9
    // H has 1s (inner), 2s (outer) -> 2 * 2 = 4 -> Total 13 basis functions
    assert_eq!(basis.len(), 13);

    let hf_res = run_scf(&atoms, &basis, 60, 1e-11).expect("SCF failed");

    println!("H2O 6-31G Total Energy: {:.10} Eh", hf_res.total_energy);
    // Literature 6-31G H2O: approx -75.98 Eh
    assert!((hf_res.total_energy - (-75.98)).abs() < 0.1);

    // Analytical Gradient
    let ana_grad = compute_rhf_gradient(&atoms, &basis, &hf_res);

    println!("H2O 6-31G Analytical Gradient (Hartree/Bohr):");
    for (i, g) in ana_grad.iter().enumerate() {
        println!("Atom {} ({}): [{:12.8}, {:12.8}, {:12.8}]", i, atoms[i].symbol, g[0], g[1], g[2]);
    }

    // Translational invariance test
    let sum_gz: f64 = ana_grad.iter().map(|g| g[2]).sum();
    assert!(sum_gz.abs() < 1e-5, "Sum Gz != 0: {}", sum_gz);

    // Check O z-coordinate numerical vs analytical
    let h = 1e-4;
    let mut atoms_plus = atoms.clone();
    atoms_plus[0].position[2] += h;
    let basis_plus = build_basis(&atoms_plus, &basis_map);
    let res_plus = run_scf(&atoms_plus, &basis_plus, 60, 1e-11).unwrap();

    let mut atoms_minus = atoms.clone();
    atoms_minus[0].position[2] -= h;
    let basis_minus = build_basis(&atoms_minus, &basis_map);
    let res_minus = run_scf(&atoms_minus, &basis_minus, 60, 1e-11).unwrap();

    let num_grad = (res_plus.total_energy - res_minus.total_energy) / (2.0 * h);
    let diff = (ana_grad[0][2] - num_grad).abs();

    println!("H2O 6-31G Atom 0 z: Analytical = {:12.8}, Numerical = {:12.8}, Diff = {:10.2e}", ana_grad[0][2], num_grad, diff);
    assert!(diff < 1e-5);
}

#[test]
fn test_gradient_quadratic_error_convergence_with_step_size() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);

    // H2 molecule at R = 1.4 Bohr
    let r = 1.4;
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -r / 2.0]),
        Atom::new("H", [0.0, 0.0,  r / 2.0]),
    ];
    let basis = build_basis(&atoms, &basis_map);
    let hf_res = run_scf(&atoms, &basis, 50, 1e-12).expect("SCF failed");
    let ana_grad = compute_rhf_gradient(&atoms, &basis, &hf_res);

    // Numerical gradient along z-axis for Atom 0 (at -r/2)
    let compute_num_grad = |step: f64| -> f64 {
        let mut atoms_p = atoms.clone();
        atoms_p[0].position[2] += step;
        let basis_p = build_basis(&atoms_p, &basis_map);
        let res_p = run_scf(&atoms_p, &basis_p, 50, 1e-12).unwrap();

        let mut atoms_m = atoms.clone();
        atoms_m[0].position[2] -= step;
        let basis_m = build_basis(&atoms_m, &basis_map);
        let res_m = run_scf(&atoms_m, &basis_m, 50, 1e-12).unwrap();

        (res_p.total_energy - res_m.total_energy) / (2.0 * step)
    };

    let ana = ana_grad[0][2];
    let h1 = 0.01;
    let h2 = 0.005;  // halved: h1 / 2
    let h3 = 0.0025; // halved: h2 / 2

    let num1 = compute_num_grad(h1);
    let num2 = compute_num_grad(h2);
    let num3 = compute_num_grad(h3);

    let err1 = (num1 - ana).abs();
    let err2 = (num2 - ana).abs();
    let err3 = (num3 - ana).abs();

    println!("Step h1 = {}: error = {:10.4e}", h1, err1);
    println!("Step h2 = {}: error = {:10.4e}", h2, err2);
    println!("Step h3 = {}: error = {:10.4e}", h3, err3);

    // When step size doubles (h2 -> h1), error expands by factor of ~4 (2^2 = 4)
    let ratio_1_2 = err1 / err2;
    let ratio_2_3 = err2 / err3;

    println!("Error ratio err(h1)/err(h2) = {:.4} (theoretical 2^2 = 4.0)", ratio_1_2);
    println!("Error ratio err(h2)/err(h3) = {:.4} (theoretical 2^2 = 4.0)", ratio_2_3);

    // Verify quadratic scaling: ratio must be close to 4.0 (within 5%)
    assert!((ratio_1_2 - 4.0).abs() < 0.2, "Error ratio 1->2 was {}, expected ~4.0", ratio_1_2);
    assert!((ratio_2_3 - 4.0).abs() < 0.2, "Error ratio 2->3 was {}, expected ~4.0", ratio_2_3);
}

#[test]
fn test_rohf_analytical_gradient_radical_oh() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = vec![
        Atom::new("O", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, 1.83]),
    ];
    let basis = build_basis(&atoms, &basis_map);
    let rohf = rohf::run_rohf(&atoms, &basis, 0, 2, 60, 1e-11).expect("ROHF failed");
    let grad_anal = rohf::compute_rohf_gradient(&atoms, &basis, &rohf);

    // Translational invariance: sum_A g_A = 0
    let mut sum_g = [0.0; 3];
    for a in 0..atoms.len() {
        for tau in 0..3 {
            sum_g[tau] += grad_anal[a][tau];
        }
    }
    assert!(sum_g[0].abs() < 1e-6);
    assert!(sum_g[1].abs() < 1e-6);
    assert!(sum_g[2].abs() < 1e-6);

    // Numerical finite difference comparison along z
    let h = 1e-4;
    let mut atoms_fwd = atoms.clone();
    atoms_fwd[1].position[2] += h;
    let basis_fwd = build_basis(&atoms_fwd, &basis_map);
    let rohf_fwd = rohf::run_rohf(&atoms_fwd, &basis_fwd, 0, 2, 60, 1e-11).unwrap();

    let mut atoms_bwd = atoms.clone();
    atoms_bwd[1].position[2] -= h;
    let basis_bwd = build_basis(&atoms_bwd, &basis_map);
    let rohf_bwd = rohf::run_rohf(&atoms_bwd, &basis_bwd, 0, 2, 60, 1e-11).unwrap();

    let g_num_z = (rohf_fwd.total_energy - rohf_bwd.total_energy) / (2.0 * h);
    println!("ROHF Analytical gradient on H: {:12.8}", grad_anal[1][2]);
    println!("ROHF Numerical gradient on H:  {:12.8}", g_num_z);
    println!("Difference: {:10.2e}", (grad_anal[1][2] - g_num_z).abs());

    assert!(
        (grad_anal[1][2] - g_num_z).abs() < 1e-3,
        "Analytical ROHF gradient ({}) must match finite difference ({})",
        grad_anal[1][2],
        g_num_z
    );
}

#[test]
fn test_rhf_mp2_analytical_gradient_h2() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let r = 1.4;
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -r / 2.0]),
        Atom::new("H", [0.0, 0.0,  r / 2.0]),
    ];
    let basis = build_basis(&atoms, &basis_map);

    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None)
        .expect("RHF SCF converged");

    let grad_anal = post_hf::mp2_gradient::compute_rhf_mp2_analytical_gradient(&atoms, &basis, &hf);

    // Translational invariance: sum_A g_A = 0
    let mut sum_g = [0.0; 3];
    for a in 0..atoms.len() {
        for tau in 0..3 {
            sum_g[tau] += grad_anal[a][tau];
        }
    }
    assert!(sum_g[0].abs() < 1e-8);
    assert!(sum_g[1].abs() < 1e-8);
    assert!(sum_g[2].abs() < 1e-8);

    // Finite difference check along z
    let h = 1e-4;
    let mut atoms_fwd = atoms.clone();
    atoms_fwd[1].position[2] += h;
    let basis_fwd = build_basis(&atoms_fwd, &basis_map);
    let hf_fwd = run_scf_general(&atoms_fwd, &basis_fwd, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None).unwrap();
    let mp2_fwd = post_hf::mp2::RHFMP2::new(&hf_fwd).compute().unwrap();
    let e_fwd = hf_fwd.total_energy + mp2_fwd.correlation_energy;

    let mut atoms_bwd = atoms.clone();
    atoms_bwd[1].position[2] -= h;
    let basis_bwd = build_basis(&atoms_bwd, &basis_map);
    let hf_bwd = run_scf_general(&atoms_bwd, &basis_bwd, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None).unwrap();
    let mp2_bwd = post_hf::mp2::RHFMP2::new(&hf_bwd).compute().unwrap();
    let e_bwd = hf_bwd.total_energy + mp2_bwd.correlation_energy;

    let g_num_z = (e_fwd - e_bwd) / (2.0 * h);
    let diff = (grad_anal[1][2] - g_num_z).abs();
    println!("MP2 Analytical gradient on H: {:12.8}", grad_anal[1][2]);
    println!("MP2 Numerical gradient on H:  {:12.8}", g_num_z);
    println!("Difference: {:10.2e}", diff);

    assert!(
        diff < 1e-5,
        "Analytical MP2 gradient ({}) must match finite difference ({}) within 1e-5 Eh/Bohr",
        grad_anal[1][2],
        g_num_z
    );
}

#[test]
fn test_casscf_analytical_gradient_h2() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -0.7]),
        Atom::new("H", [0.0, 0.0,  0.7]),
    ];
    let basis = build_basis(&atoms, &basis_map);

    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None)
        .expect("RHF SCF converged");

    let opts = post_hf::casscf::CASSCFOptions::new(2, 2);
    let casscf = post_hf::casscf::CASSCF::new(&hf, opts.clone())
        .compute()
        .expect("CASSCF calculation succeeded");

    println!("HF energy:     {:14.10}", hf.total_energy);
    println!("CASSCF energy: {:14.10}", casscf.total_energy);
    println!("Converged: {}, iters: {}", casscf.converged, casscf.iterations);
    println!("1-RDM MO:\n{:?}", casscf.one_rdm_mo);
    println!("2-RDM MO:\n{:?}", casscf.two_rdm_mo);

    let grad_anal = post_hf::casscf::compute_casscf_analytical_gradient(&atoms, &basis, &hf, &casscf);

    // 1. Translational invariance: sum_A g_A = 0
    let mut sum_g = [0.0; 3];
    for a in 0..atoms.len() {
        for tau in 0..3 {
            sum_g[tau] += grad_anal[a][tau];
        }
    }
    assert!(sum_g[0].abs() < 1e-8, "CASSCF translational invariance x failed: {}", sum_g[0]);
    assert!(sum_g[1].abs() < 1e-8, "CASSCF translational invariance y failed: {}", sum_g[1]);
    assert!(sum_g[2].abs() < 1e-8, "CASSCF translational invariance z failed: {}", sum_g[2]);

    // 2. Finite difference comparison along z
    let h = 1e-4;
    let mut atoms_fwd = atoms.clone();
    atoms_fwd[1].position[2] += h;
    let basis_fwd = build_basis(&atoms_fwd, &basis_map);
    let hf_fwd = run_scf_general(&atoms_fwd, &basis_fwd, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None).unwrap();
    let casscf_fwd = post_hf::casscf::CASSCF::new(&hf_fwd, opts.clone()).compute().unwrap();
    let e_fwd = casscf_fwd.total_energy;

    let mut atoms_bwd = atoms.clone();
    atoms_bwd[1].position[2] -= h;
    let basis_bwd = build_basis(&atoms_bwd, &basis_map);
    let hf_bwd = run_scf_general(&atoms_bwd, &basis_bwd, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None).unwrap();
    let casscf_bwd = post_hf::casscf::CASSCF::new(&hf_bwd, opts).compute().unwrap();
    let e_bwd = casscf_bwd.total_energy;

    let g_num_z = (e_fwd - e_bwd) / (2.0 * h);
    let diff = (grad_anal[1][2] - g_num_z).abs();

    println!("CASSCF Analytical gradient on H: {:12.8}", grad_anal[1][2]);
    println!("CASSCF Numerical gradient on H:  {:12.8}", g_num_z);
    println!("Difference: {:10.2e}", diff);

    assert!(
        diff < 1e-4,
        "Analytical CASSCF gradient ({}) must match finite difference ({}) within 1e-4 Eh/Bohr",
        grad_anal[1][2],
        g_num_z
    );
}

#[test]
fn test_fci_analytical_gradient_h2() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -0.7]),
        Atom::new("H", [0.0, 0.0,  0.7]),
    ];
    let basis = build_basis(&atoms, &basis_map);

    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None)
        .expect("RHF SCF converged");

    let fci = post_hf::ci::fci::FCI::new(&hf)
        .compute()
        .expect("FCI calculation succeeded");

    let grad_anal = post_hf::ci::fci::compute_fci_analytical_gradient(&atoms, &basis, &hf, &fci);

    // 1. Translational invariance: sum_A g_A = 0
    let mut sum_g = [0.0; 3];
    for a in 0..atoms.len() {
        for tau in 0..3 {
            sum_g[tau] += grad_anal[a][tau];
        }
    }
    assert!(sum_g[0].abs() < 1e-8, "FCI translational invariance x failed: {}", sum_g[0]);
    assert!(sum_g[1].abs() < 1e-8, "FCI translational invariance y failed: {}", sum_g[1]);
    assert!(sum_g[2].abs() < 1e-8, "FCI translational invariance z failed: {}", sum_g[2]);

    // 2. Finite difference comparison along z
    let h = 1e-4;
    let mut atoms_fwd = atoms.clone();
    atoms_fwd[1].position[2] += h;
    let basis_fwd = build_basis(&atoms_fwd, &basis_map);
    let hf_fwd = run_scf_general(&atoms_fwd, &basis_fwd, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None).unwrap();
    let fci_fwd = post_hf::ci::fci::FCI::new(&hf_fwd).compute().unwrap();
    let e_fwd = fci_fwd.total_energy;

    let mut atoms_bwd = atoms.clone();
    atoms_bwd[1].position[2] -= h;
    let basis_bwd = build_basis(&atoms_bwd, &basis_map);
    let hf_bwd = run_scf_general(&atoms_bwd, &basis_bwd, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None).unwrap();
    let fci_bwd = post_hf::ci::fci::FCI::new(&hf_bwd).compute().unwrap();
    let e_bwd = fci_bwd.total_energy;

    let g_num_z = (e_fwd - e_bwd) / (2.0 * h);
    let diff = (grad_anal[1][2] - g_num_z).abs();

    println!("FCI Analytical gradient on H: {:12.8}", grad_anal[1][2]);
    println!("FCI Numerical gradient on H:  {:12.8}", g_num_z);
    println!("Difference: {:10.2e}", diff);

    assert!(
        diff < 1e-4,
        "Analytical FCI gradient ({}) must match finite difference ({}) within 1e-4 Eh/Bohr",
        grad_anal[1][2],
        g_num_z
    );
}

#[test]
fn test_cisd_analytical_gradient_h2() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let r = 1.4;
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -r / 2.0]),
        Atom::new("H", [0.0, 0.0,  r / 2.0]),
    ];
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None)
        .expect("RHF calculation succeeded");

    let cisd = post_hf::ci::cisd::CISD::new(&hf)
        .compute()
        .expect("CISD calculation succeeded");

    let grad_anal = post_hf::ci::cisd::compute_cisd_analytical_gradient(&atoms, &basis, &hf, &cisd);

    // 1. Translational invariance: sum_A g_A = 0
    let mut sum_g = [0.0; 3];
    for a in 0..atoms.len() {
        for tau in 0..3 {
            sum_g[tau] += grad_anal[a][tau];
        }
    }
    assert!(sum_g[0].abs() < 1e-8, "CISD translational invariance x failed: {}", sum_g[0]);
    assert!(sum_g[1].abs() < 1e-8, "CISD translational invariance y failed: {}", sum_g[1]);
    assert!(sum_g[2].abs() < 1e-8, "CISD translational invariance z failed: {}", sum_g[2]);

    // 2. Finite difference comparison along z
    let h = 1e-4;
    let mut atoms_fwd = atoms.clone();
    atoms_fwd[1].position[2] += h;
    let basis_fwd = build_basis(&atoms_fwd, &basis_map);
    let hf_fwd = run_scf_general(&atoms_fwd, &basis_fwd, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None).unwrap();
    let cisd_fwd = post_hf::ci::cisd::CISD::new(&hf_fwd).compute().unwrap();
    let e_fwd = cisd_fwd.total_energy;

    let mut atoms_bwd = atoms.clone();
    atoms_bwd[1].position[2] -= h;
    let basis_bwd = build_basis(&atoms_bwd, &basis_map);
    let hf_bwd = run_scf_general(&atoms_bwd, &basis_bwd, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None).unwrap();
    let cisd_bwd = post_hf::ci::cisd::CISD::new(&hf_bwd).compute().unwrap();
    let e_bwd = cisd_bwd.total_energy;

    let g_num_z = (e_fwd - e_bwd) / (2.0 * h);
    let diff = (grad_anal[1][2] - g_num_z).abs();

    println!("CISD Analytical gradient on H: {:12.8}", grad_anal[1][2]);
    println!("CISD Numerical gradient on H:  {:12.8}", g_num_z);
    println!("Difference: {:10.2e}", diff);

    assert!(
        diff < 1e-4,
        "Analytical CISD gradient ({}) must match finite difference ({}) within 1e-4 Eh/Bohr",
        grad_anal[1][2],
        g_num_z
    );
}

#[test]
fn test_rhf_mp2_analytical_gradient_lih() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let r = 3.014; // Li-H bond length in Bohr (~ 1.595 A)
    let atoms = vec![
        Atom::new("Li", [0.0, 0.0, 0.0]),
        Atom::new("H",  [0.0, 0.0, r]),
    ];
    let basis = build_basis(&atoms, &basis_map);
    assert_eq!(basis.len(), 6); // Li (5) + H (1) = 6 basis functions

    let hf = run_scf(&atoms, &basis, 50, 1e-12)
        .expect("RHF calculation succeeded");

    let grad_anal = post_hf::mp2_gradient::compute_rhf_mp2_analytical_gradient(&atoms, &basis, &hf);

    // 1. Translational invariance: sum_A g_A = 0
    let mut sum_g = [0.0; 3];
    for a in 0..atoms.len() {
        for tau in 0..3 {
            sum_g[tau] += grad_anal[a][tau];
        }
    }
    assert!(sum_g[0].abs() < 1e-7, "MP2 translational invariance x failed: {}", sum_g[0]);
    assert!(sum_g[1].abs() < 1e-7, "MP2 translational invariance y failed: {}", sum_g[1]);
    assert!(sum_g[2].abs() < 1e-7, "MP2 translational invariance z failed: {}", sum_g[2]);

    // 2. Finite difference comparison along z
    let h = 1e-4;
    let mut atoms_fwd = atoms.clone();
    atoms_fwd[1].position[2] += h;
    let basis_fwd = build_basis(&atoms_fwd, &basis_map);
    let hf_fwd = run_scf(&atoms_fwd, &basis_fwd, 50, 1e-12).unwrap();
    let mp2_fwd = post_hf::mp2::RHFMP2::new(&hf_fwd).compute().unwrap();
    let e_fwd = hf_fwd.total_energy + mp2_fwd.correlation_energy;

    let mut atoms_bwd = atoms.clone();
    atoms_bwd[1].position[2] -= h;
    let basis_bwd = build_basis(&atoms_bwd, &basis_map);
    let hf_bwd = run_scf(&atoms_bwd, &basis_bwd, 50, 1e-12).unwrap();
    let mp2_bwd = post_hf::mp2::RHFMP2::new(&hf_bwd).compute().unwrap();
    let e_bwd = hf_bwd.total_energy + mp2_bwd.correlation_energy;

    let g_num_z = (e_fwd - e_bwd) / (2.0 * h);
    let g_hf_num_z = (hf_fwd.total_energy - hf_bwd.total_energy) / (2.0 * h);
    let g_corr_num_z = (mp2_fwd.correlation_energy - mp2_bwd.correlation_energy) / (2.0 * h);
    let g_hf_anal = crate::gradient::compute_rhf_gradient(&atoms, &basis, &hf);

    println!("LiH HF Anal gradient on H:       {:12.8}", g_hf_anal[1][2]);
    println!("LiH HF Num gradient on H:        {:12.8}", g_hf_num_z);
    println!("LiH HF Diff:                     {:10.2e}", (g_hf_anal[1][2] - g_hf_num_z).abs());

    let g_corr_anal_z = grad_anal[1][2] - g_hf_anal[1][2];
    println!("LiH MP2 Corr Anal gradient on H: {:12.8}", g_corr_anal_z);
    println!("LiH MP2 Corr Num gradient on H:  {:12.8}", g_corr_num_z);
    println!("LiH MP2 Corr Diff:               {:10.2e}", (g_corr_anal_z - g_corr_num_z).abs());

    let diff = (grad_anal[1][2] - g_num_z).abs();
    println!("LiH Total MP2 Anal gradient on H: {:12.8}", grad_anal[1][2]);
    println!("LiH Total MP2 Num gradient on H:  {:12.8}", g_num_z);
    println!("Total Difference: {:10.2e}", diff);
    assert!(diff < 1e-7, "MP2 LiH gradient must match finite difference to < 1e-7: diff = {}", diff);
}

#[test]
fn test_cisd_analytical_gradient_lih() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let r = 3.014;
    let atoms = vec![
        Atom::new("Li", [0.0, 0.0, 0.0]),
        Atom::new("H",  [0.0, 0.0, r]),
    ];
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None)
        .expect("RHF calculation succeeded");

    let cisd = post_hf::ci::cisd::CISD::new(&hf)
        .compute()
        .expect("CISD calculation succeeded");

    let grad_anal = post_hf::ci::cisd::compute_cisd_analytical_gradient(&atoms, &basis, &hf, &cisd);

    // 1. Translational invariance
    let mut sum_g = [0.0; 3];
    for a in 0..atoms.len() {
        for tau in 0..3 {
            sum_g[tau] += grad_anal[a][tau];
        }
    }
    assert!(sum_g[0].abs() < 1e-7);
    assert!(sum_g[1].abs() < 1e-7);
    assert!(sum_g[2].abs() < 1e-7);

    // 2. Finite difference comparison along z
    let h = 1e-4;
    let mut atoms_fwd = atoms.clone();
    atoms_fwd[1].position[2] += h;
    let basis_fwd = build_basis(&atoms_fwd, &basis_map);
    let hf_fwd = run_scf_general(&atoms_fwd, &basis_fwd, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None).unwrap();
    let cisd_fwd = post_hf::ci::cisd::CISD::new(&hf_fwd).compute().unwrap();
    let e_fwd = cisd_fwd.total_energy;

    let mut atoms_bwd = atoms.clone();
    atoms_bwd[1].position[2] -= h;
    let basis_bwd = build_basis(&atoms_bwd, &basis_map);
    let hf_bwd = run_scf_general(&atoms_bwd, &basis_bwd, 0, 50, 1e-12, diis::ConvAlgo::Cdiis, None).unwrap();
    let cisd_bwd = post_hf::ci::cisd::CISD::new(&hf_bwd).compute().unwrap();
    let e_bwd = cisd_bwd.total_energy;

    let g_num_z = (e_fwd - e_bwd) / (2.0 * h);
    let diff = (grad_anal[1][2] - g_num_z).abs();

    println!("LiH CISD Analytical gradient on H: {:12.8}", grad_anal[1][2]);
    println!("LiH CISD Numerical gradient on H:  {:12.8}", g_num_z);
    println!("Difference: {:10.2e}", diff);

    assert!(
        diff < 5e-6,
        "Analytical CISD gradient ({}) must match finite difference ({}) within 5e-6 Eh/Bohr",
        grad_anal[1][2],
        g_num_z
    );
}




