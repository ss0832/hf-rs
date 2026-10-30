// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

#![allow(clippy::needless_range_loop)]

use hf_rs::*;

#[test]
fn test_h2_fourth_derivatives_rhf() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);

    // H2 molecule along z-axis at R = 1.4 Bohr
    let r = 1.4;
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -r / 2.0]),
        Atom::new("H", [0.0, 0.0, r / 2.0]),
    ];

    let tensor = compute_fourth_derivatives(&atoms, &basis_map, 50, 1e-12);
    let dim = tensor.dim;
    assert_eq!(dim, 6);

    // 2. Non-zero elements along molecular bond axis (z0 = 2, z1 = 5)
    let q_2222 = tensor.get(2, 2, 2, 2);
    let q_2225 = tensor.get(2, 2, 2, 5);
    let q_2255 = tensor.get(2, 2, 5, 5);
    let q_5555 = tensor.get(5, 5, 5, 5);

    println!("H2 RHF 4th derivatives along z-axis:");
    println!("  Q[2, 2, 2, 2] = {:14.8} Eh / Bohr^4", q_2222);
    println!("  Q[2, 2, 2, 5] = {:14.8} Eh / Bohr^4", q_2225);
    println!("  Q[2, 2, 5, 5] = {:14.8} Eh / Bohr^4", q_2255);
    println!("  Q[5, 5, 5, 5] = {:14.8} Eh / Bohr^4", q_5555);

    assert!(
        q_2222.abs() > 0.01,
        "Bond-axis 4th derivative should be non-zero"
    );
    // Inversion symmetry between H(0) and H(1)
    assert!((q_2222 - q_5555).abs() < 1e-6);

    // 3. Compare against finite difference of 3rd derivative T_{2, 2, 2}
    let h = 1e-4;
    let mut atoms_p = atoms.clone();
    atoms_p[0].position[2] += h;
    let t_p = compute_third_derivatives(&atoms_p, &basis_map, 50, 1e-12);

    let mut atoms_m = atoms.clone();
    atoms_m[0].position[2] -= h;
    let t_m = compute_third_derivatives(&atoms_m, &basis_map, 50, 1e-12);

    let num_q_2222 = (t_p.get(2, 2, 2) - t_m.get(2, 2, 2)) / (2.0 * h);
    let diff = (q_2222 - num_q_2222).abs();

    println!(
        "Q[2,2,2,2]: Tensor = {:14.8}, Num from T = {:14.8}, Diff = {:10.2e}",
        q_2222, num_q_2222, diff
    );
    assert!(
        diff < 5e-4,
        "4th derivative mismatch against dT/dR: diff = {}",
        diff
    );
}

#[test]
fn test_h2_plus_fourth_derivatives_uhf() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);

    // H2+ cation doublet (open-shell, charge = +1, mult = 2)
    let r = 2.0;
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -r / 2.0]),
        Atom::new("H", [0.0, 0.0, r / 2.0]),
    ];

    let tensor = compute_uhf_fourth_derivatives(&atoms, &basis_map, 1, 2, false, 50, 1e-12);
    let dim = tensor.dim;
    assert_eq!(dim, 6);

    let q_2222 = tensor.get(2, 2, 2, 2);
    let q_5555 = tensor.get(5, 5, 5, 5);

    println!("H2+ UHF 4th derivatives:");
    println!("  Q[2, 2, 2, 2] = {:14.8} Eh / Bohr^4", q_2222);
    println!("  Q[5, 5, 5, 5] = {:14.8} Eh / Bohr^4", q_5555);

    assert!(q_2222.abs() > 0.01);
    assert!((q_2222 - q_5555).abs() < 1e-6);
}
