// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

#![allow(clippy::needless_range_loop)]

use hf_rs::*;

#[test]
fn test_h2_plus_cation_uhf_gradient() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);

    // H2+ cation: 1 electron, charge = +1, multiplicity = 2 (doublet)
    let r = 2.0; // Bohr
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -r / 2.0]),
        Atom::new("H", [0.0, 0.0,  r / 2.0]),
    ];

    let basis = build_basis(&atoms, &basis_map);
    assert_eq!(basis.len(), 2);

    let uhf_res = run_uhf(&atoms, &basis, 1, 2, false, 50, 1e-12)
        .expect("UHF SCF failed for H2+");

    assert!(uhf_res.converged);
    assert_eq!(uhf_res.n_alpha, 1);
    assert_eq!(uhf_res.n_beta, 0);
    // <S^2> for pure doublet should be 0.5 * 1.5 = 0.75
    assert!((uhf_res.s_squared - 0.75).abs() < 1e-5, "<S^2> was {}", uhf_res.s_squared);

    println!("H2+ UHF Total Energy: {:.10} Eh, <S^2> = {:.4}", uhf_res.total_energy, uhf_res.s_squared);

    // Analytical gradient
    let ana_grad = compute_uhf_gradient(&atoms, &basis, &uhf_res);

    // Translational invariance test
    let sum_gx: f64 = ana_grad.iter().map(|g| g[0]).sum();
    let sum_gy: f64 = ana_grad.iter().map(|g| g[1]).sum();
    let sum_gz: f64 = ana_grad.iter().map(|g| g[2]).sum();
    assert!(sum_gx.abs() < 1e-8);
    assert!(sum_gy.abs() < 1e-8);
    assert!(sum_gz.abs() < 1e-8);

    // Central difference verification
    let h = 1e-4;
    for a in 0..atoms.len() {
        for tau in 0..3 {
            let mut atoms_plus = atoms.clone();
            atoms_plus[a].position[tau] += h;
            let basis_plus = build_basis(&atoms_plus, &basis_map);
            let res_plus = run_uhf(&atoms_plus, &basis_plus, 1, 2, false, 50, 1e-12).unwrap();

            let mut atoms_minus = atoms.clone();
            atoms_minus[a].position[tau] -= h;
            let basis_minus = build_basis(&atoms_minus, &basis_map);
            let res_minus = run_uhf(&atoms_minus, &basis_minus, 1, 2, false, 50, 1e-12).unwrap();

            let num_grad = (res_plus.total_energy - res_minus.total_energy) / (2.0 * h);
            let diff = (ana_grad[a][tau] - num_grad).abs();

            println!(
                "H2+ Atom {} tau {}: ana = {:12.8}, num = {:12.8}, diff = {:10.2e}",
                a, tau, ana_grad[a][tau], num_grad, diff
            );
            assert!(diff < 1e-6, "H2+ gradient mismatch: diff = {}", diff);
        }
    }
}

#[test]
fn test_hydroxide_anion_formal_charge() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);

    // OH- anion: 10 electrons (O=8, H=1, charge=-1), singlet
    let r_ang = 0.96;
    let atoms = vec![
        Atom::from_angstrom("O", [0.0, 0.0, 0.0]),
        Atom::from_angstrom("H", [0.0, 0.0, r_ang]),
    ];

    let basis = build_basis(&atoms, &basis_map);
    assert_eq!(basis.len(), 6); // O: 5, H: 1

    let rhf_res = run_scf_with_charge(&atoms, &basis, -1, 50, 1e-12)
        .expect("RHF SCF failed for OH-");
    assert!(rhf_res.converged);

    let uhf_res = run_uhf(&atoms, &basis, -1, 1, false, 50, 1e-12)
        .expect("UHF SCF failed for OH-");
    assert!(uhf_res.converged);

    println!("OH- RHF Energy: {:.10} Eh", rhf_res.total_energy);
    println!("OH- UHF Energy: {:.10} Eh", uhf_res.total_energy);

    // For closed shell singlet, RHF and UHF energies must be identical
    assert!((rhf_res.total_energy - uhf_res.total_energy).abs() < 1e-8);
    assert!((uhf_res.s_squared - 0.0).abs() < 1e-6);

    // Check analytical RHF gradient vs UHF gradient
    let rhf_grad = compute_rhf_gradient(&atoms, &basis, &rhf_res);
    let uhf_grad = compute_uhf_gradient(&atoms, &basis, &uhf_res);

    for a in 0..atoms.len() {
        for tau in 0..3 {
            let diff = (rhf_grad[a][tau] - uhf_grad[a][tau]).abs();
            assert!(diff < 1e-7, "RHF and UHF gradient mismatch for OH-: {}", diff);
        }
    }
}

#[test]
fn test_h2_stability_and_broken_symmetry_singlet() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);

    // 1. Equilibrium H2 (R = 1.4 Bohr) -> RHF is STABLE
    {
        let r = 1.4;
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -r / 2.0]),
            Atom::new("H", [0.0, 0.0,  r / 2.0]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let rhf_res = run_scf(&atoms, &basis, 50, 1e-12).unwrap();
        let stab = analyze_rhf_stability(&rhf_res, 1);

        println!("Equilibrium H2 (R=1.4 Bohr) Stability: is_stable={}, lowest_eig={:.6}", stab.is_stable, stab.lowest_eigenvalue);
        assert!(stab.is_stable, "Equilibrium H2 should be stable");
        assert!(stab.lowest_eigenvalue > 0.0);
    }

    // 2. Stretched H2 (R = 3.5 Bohr) -> RHF is UNSTABLE, broken-symmetry UHF has lower energy
    {
        let r = 3.5;
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -r / 2.0]),
            Atom::new("H", [0.0, 0.0,  r / 2.0]),
        ];
        let basis = build_basis(&atoms, &basis_map);

        let rhf_res = run_scf(&atoms, &basis, 50, 1e-12).unwrap();
        let stab = analyze_rhf_stability(&rhf_res, 1);

        println!("Stretched H2 (R=3.5 Bohr) Stability: is_stable={}, lowest_eig={:.6}", stab.is_stable, stab.lowest_eigenvalue);
        assert!(!stab.is_stable, "Stretched H2 RHF should be unstable");
        assert!(stab.lowest_eigenvalue < 0.0);

        // Run UHF with broken symmetry
        let bs_uhf_res = run_uhf(&atoms, &basis, 0, 1, true, 80, 1e-12)
            .expect("Broken-symmetry UHF failed");

        println!("Stretched H2 RHF Energy: {:.10} Eh", rhf_res.total_energy);
        println!("Stretched H2 BS-UHF Energy: {:.10} Eh, <S^2> = {:.4}", bs_uhf_res.total_energy, bs_uhf_res.s_squared);

        // UHF must have lower energy than RHF
        assert!(bs_uhf_res.total_energy < rhf_res.total_energy - 0.04);
        // Spin contamination for broken symmetry singlet diradical: <S^2> is between 0.5 and 1.0
        assert!(bs_uhf_res.s_squared > 0.5);

        // Verify analytical gradient for broken-symmetry singlet vs central difference
        let ana_grad = compute_uhf_gradient(&atoms, &basis, &bs_uhf_res);
        let h = 1e-4;
        for a in 0..atoms.len() {
            for tau in 0..3 {
                let mut atoms_plus = atoms.clone();
                atoms_plus[a].position[tau] += h;
                let basis_plus = build_basis(&atoms_plus, &basis_map);
                let res_plus = run_uhf(&atoms_plus, &basis_plus, 0, 1, true, 80, 1e-12).unwrap();

                let mut atoms_minus = atoms.clone();
                atoms_minus[a].position[tau] -= h;
                let basis_minus = build_basis(&atoms_minus, &basis_map);
                let res_minus = run_uhf(&atoms_minus, &basis_minus, 0, 1, true, 80, 1e-12).unwrap();

                let num_grad = (res_plus.total_energy - res_minus.total_energy) / (2.0 * h);
                let diff = (ana_grad[a][tau] - num_grad).abs();

                println!(
                    "BS-UHF Atom {} tau {}: ana = {:12.8}, num = {:12.8}, diff = {:10.2e}",
                    a, tau, ana_grad[a][tau], num_grad, diff
                );
                assert!(diff < 1e-6, "BS-UHF gradient mismatch: diff = {}", diff);
            }
        }
    }
}

#[test]
fn test_triplet_o2_uhf_gradient() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);

    // O2 triplet molecule (ground state 3Sigma_g-)
    // Experimental bond length approx 1.208 A (approx 2.28 Bohr)
    let r = 2.28;
    let atoms = vec![
        Atom::new("O", [0.0, 0.0, -r / 2.0]),
        Atom::new("O", [0.0, 0.0,  r / 2.0]),
    ];

    let basis = build_basis(&atoms, &basis_map);
    assert_eq!(basis.len(), 10); // 5 for each O

    // 16 electrons, charge = 0, multiplicity = 3 (triplet: N_alpha=9, N_beta=7)
    let uhf_res = run_uhf(&atoms, &basis, 0, 3, false, 80, 1e-11)
        .expect("UHF failed for triplet O2");

    assert!(uhf_res.converged);
    assert_eq!(uhf_res.n_alpha, 9);
    assert_eq!(uhf_res.n_beta, 7);

    println!("Triplet O2 UHF Energy: {:.10} Eh, <S^2> = {:.4}", uhf_res.total_energy, uhf_res.s_squared);
    // <S^2> for triplet (S=1) is S(S+1) = 2.0. With minimal spin contamination, it should be close to 2.0
    assert!((uhf_res.s_squared - 2.0).abs() < 0.2, "<S^2> was {}", uhf_res.s_squared);

    // Analytical gradient
    let ana_grad = compute_uhf_gradient(&atoms, &basis, &uhf_res);

    // Translational invariance test
    let sum_gx: f64 = ana_grad.iter().map(|g| g[0]).sum();
    let sum_gy: f64 = ana_grad.iter().map(|g| g[1]).sum();
    let sum_gz: f64 = ana_grad.iter().map(|g| g[2]).sum();
    assert!(sum_gx.abs() < 1e-7);
    assert!(sum_gy.abs() < 1e-7);
    assert!(sum_gz.abs() < 1e-7);

    // Numerical gradient comparison along z-axis for atom 0
    let h = 1e-4;
    let mut atoms_plus = atoms.clone();
    atoms_plus[0].position[2] += h;
    let basis_plus = build_basis(&atoms_plus, &basis_map);
    let res_plus = run_uhf(&atoms_plus, &basis_plus, 0, 3, false, 80, 1e-11).unwrap();

    let mut atoms_minus = atoms.clone();
    atoms_minus[0].position[2] -= h;
    let basis_minus = build_basis(&atoms_minus, &basis_map);
    let res_minus = run_uhf(&atoms_minus, &basis_minus, 0, 3, false, 80, 1e-11).unwrap();

    let num_grad = (res_plus.total_energy - res_minus.total_energy) / (2.0 * h);
    let diff = (ana_grad[0][2] - num_grad).abs();

    println!("Triplet O2 Atom 0 z-gradient: ana = {:12.8}, num = {:12.8}, diff = {:10.2e}", ana_grad[0][2], num_grad, diff);
    assert!(diff < 1e-6, "Triplet O2 gradient mismatch: diff = {}", diff);
}

#[test]
fn test_uhf_gradient_quadratic_error_convergence() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);

    // H2+ cation (open-shell doublet, charge=+1, mult=2)
    let r = 2.0;
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -r / 2.0]),
        Atom::new("H", [0.0, 0.0,  r / 2.0]),
    ];

    let basis = build_basis(&atoms, &basis_map);
    let uhf_res = run_uhf(&atoms, &basis, 1, 2, false, 50, 1e-12).unwrap();
    let ana_grad = compute_uhf_gradient(&atoms, &basis, &uhf_res);
    let ana = ana_grad[0][2];

    let compute_num_grad = |step: f64| -> f64 {
        let mut atoms_p = atoms.clone();
        atoms_p[0].position[2] += step;
        let basis_p = build_basis(&atoms_p, &basis_map);
        let res_p = run_uhf(&atoms_p, &basis_p, 1, 2, false, 50, 1e-12).unwrap();

        let mut atoms_m = atoms.clone();
        atoms_m[0].position[2] -= step;
        let basis_m = build_basis(&atoms_m, &basis_map);
        let res_m = run_uhf(&atoms_m, &basis_m, 1, 2, false, 50, 1e-12).unwrap();

        (res_p.total_energy - res_m.total_energy) / (2.0 * step)
    };

    let h1 = 0.01;
    let h2 = 0.005;  // halved
    let h3 = 0.0025; // halved again

    let num1 = compute_num_grad(h1);
    let num2 = compute_num_grad(h2);
    let num3 = compute_num_grad(h3);

    let err1 = (num1 - ana).abs();
    let err2 = (num2 - ana).abs();
    let err3 = (num3 - ana).abs();

    println!("UHF H2+ Step h1 = {}: error = {:10.4e}", h1, err1);
    println!("UHF H2+ Step h2 = {}: error = {:10.4e}", h2, err2);
    println!("UHF H2+ Step h3 = {}: error = {:10.4e}", h3, err3);

    let ratio_1_2 = err1 / err2;
    let ratio_2_3 = err2 / err3;

    println!("UHF Error ratio err(h1)/err(h2) = {:.4} (theoretical 2^2 = 4.0)", ratio_1_2);
    println!("UHF Error ratio err(h2)/err(h3) = {:.4} (theoretical 2^2 = 4.0)", ratio_2_3);

    // Verify quadratic scaling (error scales as h^2, ratio ~ 4.0)
    assert!((ratio_1_2 - 4.0).abs() < 0.2, "UHF Error ratio 1->2 was {}, expected ~4.0", ratio_1_2);
    assert!((ratio_2_3 - 4.0).abs() < 0.2, "UHF Error ratio 2->3 was {}, expected ~4.0", ratio_2_3);
}
