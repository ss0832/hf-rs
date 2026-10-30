// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::*;

// Five-point differences of the independently evaluated analytical gradient.
// A 1e-4 two-point step amplifies finite SCF residuals above the Hessian tolerance.
fn reference_hessian(atoms: &[Atom], basis_map: &BasisSetMap) -> Matrix {
    let dim = 3 * atoms.len();
    let step = 5e-3;
    let mut result = Matrix::zeros(dim, dim);
    for j in 0..dim {
        for (offset, weight) in [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)] {
            let mut moved = atoms.to_vec();
            moved[j / 3].position[j % 3] += offset * step;
            let basis = build_basis(&moved, basis_map);
            let hf = run_scf_general(&moved, &basis, 0, 150, 1e-13, ConvAlgo::Cdiis, None).unwrap();
            assert!(hf.converged);
            let g = compute_rhf_gradient(&moved, &basis, &hf);
            for i in 0..dim {
                result[(i, j)] += weight * g[i / 3][i % 3] / (12.0 * step);
            }
        }
    }
    result
}

#[test]
fn test_h2_hessian_and_frequencies() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);

    // H2 molecule at R = 1.4 Bohr along z-axis
    let r = 1.4;
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -r / 2.0]),
        Atom::new("H", [0.0, 0.0, r / 2.0]),
    ];

    println!("Calculating Hessian for H2...");
    let hess = compute_hessian(&atoms, &basis_map, 50, 1e-12);

    println!("H2 6x6 Hessian (Hartree / Bohr^2):");
    println!("{}", hess);

    // 2. Translational invariance check: sum of rows / columns must be 0
    for tau in 0..3 {
        for eta in 0..3 {
            let sum_val = hess[(tau, eta)] + hess[(3 + tau, eta)];
            assert!(
                sum_val.abs() < 1e-6,
                "Translational invariance broken for ({}, {}): sum = {}",
                tau,
                eta,
                sum_val
            );
        }
    }

    // 3. Vibrational frequencies via complete analysis
    let analysis = compute_hessian_analysis(&atoms, &hess);
    println!("Raw Vibrational Frequencies (cm^-1):");
    for (i, f) in analysis.raw_frequencies.iter().enumerate() {
        println!("  Mode {:2}: {:10.2} cm^-1", i + 1, f);
    }

    println!("Harmonic Vibrational Frequencies (Projected, cm^-1):");
    for (i, f) in analysis.projected_frequencies.iter().enumerate() {
        println!("  Mode {:2}: {:10.2} cm^-1", i + 1, f);
    }

    // For H2 (diatomic linear), Eckart projection removes 3 translational + 2 rotational modes = 5 modes with exactly 0.00 cm^-1.
    for m in 0..5 {
        assert!(
            analysis.projected_frequencies[m].abs() < 1e-4,
            "Projected mode {} is not zero: {}",
            m + 1,
            analysis.projected_frequencies[m]
        );
    }

    // Stretch frequency: approx 4871.88 cm^-1
    let stretch_mode = analysis.projected_frequencies[5];
    println!(
        "H2 Stretch Frequency (Projected): {:.2} cm^-1",
        stretch_mode
    );
    assert!(
        stretch_mode > 4500.0 && stretch_mode < 5500.0,
        "Unexpected stretch frequency: {}",
        stretch_mode
    );

    // Verify both raw and projected Hessians are populated
    assert_eq!(analysis.raw_hessian.rows, 6);
    assert_eq!(analysis.projected_hessian.rows, 6);
}

#[test]
fn test_h2_third_derivatives() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);

    let r = 1.4;
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -r / 2.0]),
        Atom::new("H", [0.0, 0.0, r / 2.0]),
    ];

    println!("Calculating 3rd nuclear derivatives for H2...");
    let tensor = compute_third_derivatives(&atoms, &basis_map, 50, 1e-12);

    // Translational invariance: \sum_A T_{A\tau, B\eta, C\xi} = 0
    for tau in 0..3 {
        for eta in 0..3 {
            for xi in 0..3 {
                let mut sum_a = 0.0;
                for a in 0..atoms.len() {
                    sum_a += tensor.get(3 * a + tau, eta, xi);
                }
                assert!(
                    sum_a.abs() < 1e-5,
                    "3rd deriv translational invariance failed: sum = {}",
                    sum_a
                );
            }
        }
    }

    // Print significant non-zero 3rd derivative components along z (stretch axis)
    // For H2 along z: z0 is index 2, z1 is index 5
    let t_z0_z0_z0 = tensor.get(2, 2, 2);
    let t_z0_z0_z1 = tensor.get(2, 2, 5);
    println!("d^3 E / dz0^3:       {:15.8} Hartree / Bohr^3", t_z0_z0_z0);
    println!("d^3 E / dz0^2 dz1:   {:15.8} Hartree / Bohr^3", t_z0_z0_z1);
    assert!((t_z0_z0_z0 + t_z0_z0_z1).abs() < 1e-5);
}

#[test]
fn test_h2_analytical_hessian_vs_finite_difference() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let r = 1.4;
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -r / 2.0]),
        Atom::new("H", [0.0, 0.0, r / 2.0]),
    ];

    let hess_anal = compute_hessian(&atoms, &basis_map, 50, 1e-12);
    let hess_num = reference_hessian(&atoms, &basis_map);

    let dim = 3 * atoms.len();
    for i in 0..dim {
        for j in 0..dim {
            let diff = (hess_anal[(i, j)] - hess_num[(i, j)]).abs();
            println!(
                "Hessian ({}, {}): anal = {:12.8}, num = {:12.8}, diff = {:10.2e}",
                i,
                j,
                hess_anal[(i, j)],
                hess_num[(i, j)],
                diff
            );
            assert!(
                diff < 1e-5,
                "Analytical Hessian ({}, {}) does not match finite difference: anal={}, num={}, diff={}",
                i,
                j,
                hess_anal[(i, j)],
                hess_num[(i, j)],
                diff
            );
        }
    }
}

#[test]
fn test_water_analytical_hessian_vs_finite_difference() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let angle_rad = 104.5_f64.to_radians();
    let r_ang = 0.9578;
    let h1_x = r_ang * (angle_rad / 2.0).sin();
    let h1_z = r_ang * (angle_rad / 2.0).cos();

    let atoms = vec![
        Atom::from_angstrom("O", [0.0, 0.0, 0.0]),
        Atom::from_angstrom("H", [h1_x, 0.0, h1_z]),
        Atom::from_angstrom("H", [-h1_x, 0.0, h1_z]),
    ];

    let hess_anal = compute_hessian(&atoms, &basis_map, 50, 1e-12);
    let hess_num = reference_hessian(&atoms, &basis_map);

    let dim = 3 * atoms.len();
    let mut max_diff = 0.0;
    let mut max_loc = (0, 0);
    for i in 0..dim {
        for j in 0..dim {
            let diff = (hess_anal[(i, j)] - hess_num[(i, j)]).abs();
            if diff > max_diff {
                max_diff = diff;
                max_loc = (i, j);
            }
            if diff > 1e-4 {
                println!(
                    "Mismatch ({}, {}): anal = {:12.8}, num = {:12.8}, diff = {:10.4e}",
                    i,
                    j,
                    hess_anal[(i, j)],
                    hess_num[(i, j)],
                    diff
                );
            }
        }
    }
    println!("Max difference at {:?}: {:10.4e}", max_loc, max_diff);
    assert!(
        max_diff < 1e-5,
        "Water analytical Hessian differs from finite difference by {}",
        max_diff
    );
}
