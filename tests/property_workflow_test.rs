// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Actual structure-to-property workflows, including a nonlocal ECP. All Cartesian derivatives
//! and multipole components are checked. Numerical differences appear only in the independent
//! ECP Hessian oracle; hf-rs uses analytic forces and Hessians throughout.

use hf_rs::{
    Atom, ContractedGaussian, ConvAlgo, HFResult, Matrix, build_basis, compute_hessian_analysis,
    compute_rhf_analytical_hessian, compute_rhf_gradient, get_builtin_basis, run_scf_general,
};

struct Point {
    atoms: Vec<Atom>,
    basis: Vec<ContractedGaussian>,
    hf: HFResult,
    gradient: Vec<[f64; 3]>,
}

fn evaluate(name: &str, x: &[f64]) -> Point {
    let (mut atoms, basis_name) = if name == "water" {
        let (s, c) = (0.5 * x[1]).sin_cos();
        (
            vec![
                Atom::new("O", [0.; 3]),
                Atom::new("H", [x[0] * s, 0., x[0] * c]),
                Atom::new("H", [-x[0] * s, 0., x[0] * c]),
            ],
            "6-31g*",
        )
    } else {
        (
            vec![Atom::new("Rb", [0.; 3]), Atom::new("H", [0., 0., x[0]])],
            "def2-svp",
        )
    };
    hf_rs::ecp::assign_builtin_ecps(&mut atoms, basis_name);
    let basis = build_basis(&atoms, &get_builtin_basis(basis_name).unwrap());
    let hf = run_scf_general(&atoms, &basis, 0, 300, 1e-13, ConvAlgo::Adiissoscf, None).unwrap();
    assert!(hf.converged, "{name}: unconverged RHF");
    let gradient = compute_rhf_gradient(&atoms, &basis, &hf);
    Point {
        atoms,
        basis,
        hf,
        gradient,
    }
}

fn jacobian(name: &str, x: &[f64]) -> Matrix {
    if name != "water" {
        let mut j = Matrix::zeros(6, 1);
        j[(5, 0)] = 1.;
        return j;
    }
    let (s, c) = (0.5 * x[1]).sin_cos();
    let mut j = Matrix::zeros(9, 2);
    for (atom, sign) in [(1, 1.), (2, -1.)] {
        j[(3 * atom, 0)] = sign * s;
        j[(3 * atom + 2, 0)] = c;
        j[(3 * atom, 1)] = sign * x[0] * c / 2.;
        j[(3 * atom + 2, 1)] = -x[0] * s / 2.;
    }
    j
}

fn force(point: &Point, j: &Matrix) -> Vec<f64> {
    (0..j.cols)
        .map(|k| {
            (0..j.rows)
                .map(|i| j[(i, k)] * point.gradient[i / 3][i % 3])
                .sum()
        })
        .collect()
}

fn optimize(name: &str, initial: &[f64]) -> (Vec<f64>, Point, usize) {
    let mut x = initial.to_vec();
    // RbH uses a force secant; water uses exact internal-coordinate Newton steps.
    let mut old_r = initial[0] - 0.1;
    let mut old_force = if name == "water" {
        0.
    } else {
        evaluate(name, &[old_r]).gradient[1][2]
    };
    for iteration in 0..16 {
        let point = evaluate(name, &x);
        let j = jacobian(name, &x);
        let g = force(&point, &j);
        if g.iter().all(|v| v.abs() < 2e-9) {
            return (x, point, iteration);
        }
        if name == "water" {
            let h = compute_rhf_analytical_hessian(&point.atoms, &point.basis, &point.hf);
            let mut internal = j.transpose().dot(&h).dot(&j);
            let cross = g[1] / x[0];
            internal[(0, 1)] += cross;
            internal[(1, 0)] += cross;
            internal[(1, 1)] -= point
                .atoms
                .iter()
                .zip(&point.gradient)
                .map(|(a, g)| (0..3).map(|k| a.position[k] * g[k]).sum::<f64>())
                .sum::<f64>()
                / 4.;
            let step = internal.lu_decompose().unwrap().solve(&g);
            for k in 0..x.len() {
                x[k] -= step[k];
            }
        } else {
            let step = g[0] * (x[0] - old_r) / (g[0] - old_force);
            old_r = x[0];
            old_force = g[0];
            x[0] -= step;
        }
    }
    panic!("{name}: structural optimization did not reach the force tolerance");
}

fn compare_properties(case: &serde_json::Value) {
    let name = case["name"].as_str().unwrap();
    let initial: Vec<f64> = case["initial"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let (x, point, iterations) = optimize(name, &initial);
    for (k, actual) in x.iter().enumerate() {
        assert!(
            (actual - case["optimized"][k].as_f64().unwrap()).abs() < 5e-8,
            "{name}: optimized coordinate {k}: {actual}"
        );
    }
    assert!(
        (point.hf.total_energy - case["energy"].as_f64().unwrap()).abs() < 1e-9,
        "{name}: energy"
    );
    let dim = point.atoms.len() * 3;
    for i in 0..dim {
        assert!(
            (point.gradient[i / 3][i % 3] - case["gradient"][i / 3][i % 3].as_f64().unwrap()).abs()
                < 5e-9,
            "{name}: gradient {i}"
        );
    }
    let p = &point.hf.properties;
    for i in 0..3 {
        assert!(
            (p.dipole_moment_au[i] - case["dipole_au"][i].as_f64().unwrap()).abs() < 1e-7,
            "{name}: dipole {i}"
        );
        for j in 0..3 {
            assert!(
                (p.quadrupole_moment_au[i][j] - case["quadrupole_au"][i][j].as_f64().unwrap())
                    .abs()
                    < 1e-6,
                "{name}: quadrupole {i},{j}"
            );
        }
    }
    assert!(
        p.mulliken_charges.iter().sum::<f64>().abs() < 1e-9,
        "{name}: Mulliken electron count"
    );
    assert!(
        p.lowdin_charges.iter().sum::<f64>().abs() < 1e-9,
        "{name}: Lowdin electron count"
    );
    let h = compute_rhf_analytical_hessian(&point.atoms, &point.basis, &point.hf);
    let mut hessian_error = 0f64;
    for i in 0..dim {
        for j in 0..dim {
            hessian_error =
                hessian_error.max((h[(i, j)] - case["hessian"][i][j].as_f64().unwrap()).abs());
        }
    }
    assert!(
        hessian_error < 3e-7,
        "{name}: all {dim}x{dim} Hessian entries: {hessian_error:e}"
    );
    let analysis = compute_hessian_analysis(&point.atoms, &h);
    let frequencies: Vec<f64> = analysis
        .projected_frequencies
        .iter()
        .copied()
        .filter(|v| v.abs() > 1.)
        .collect();
    let expected = case["frequencies_cm1"].as_array().unwrap();
    assert_eq!(
        frequencies.len(),
        expected.len(),
        "{name}: vibrational mode count"
    );
    for (i, value) in frequencies.iter().enumerate() {
        assert!(
            (value - expected[i].as_f64().unwrap()).abs() < 0.02,
            "{name}: frequency {i}: {value}"
        );
    }
    let zero_modes = dim - expected.len();
    let mut raw = analysis.raw_frequencies.clone();
    raw.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
    assert!(
        raw[..zero_modes].iter().all(|v| v.abs() < 1.),
        "{name}: raw rigid-body modes {raw:?}"
    );

    check_multipole_transport(name, &point.atoms, &point.hf);
    println!(
        "PROPERTY_WORKFLOW {}",
        serde_json::json!({"name": name, "optimized": x, "optimization_steps": iterations,
        "energy_hartree": point.hf.total_energy, "dipole_debye": p.dipole_moment_debye, "frequencies_cm1": frequencies,
        "hessian_entries": dim * dim, "max_hessian_error": hessian_error, "raw_rigid_frequencies_cm1": &raw[..zero_modes]})
    );
}

fn check_multipole_transport(name: &str, atoms: &[Atom], hf: &HFResult) {
    let p = &hf.properties;
    // A neutral molecule's dipole is independent of translation and transforms as a vector.
    let rotation = Matrix::from_vec(3, 3, vec![0.36, -0.48, 0.8, 0.8, 0.6, 0., -0.48, 0.64, 0.6]);
    let shift = [0.7, -0.4, 0.3];
    let mut moved = atoms.to_vec();
    for atom in &mut moved {
        atom.position = std::array::from_fn(|i| {
            shift[i]
                + (0..3)
                    .map(|j| rotation[(i, j)] * atom.position[j])
                    .sum::<f64>()
        });
    }
    let basis_name = if name == "water" {
        "6-31g*"
    } else {
        "def2-svp"
    };
    let basis = build_basis(&moved, &get_builtin_basis(basis_name).unwrap());
    let transported =
        run_scf_general(&moved, &basis, 0, 300, 1e-13, ConvAlgo::Adiissoscf, None).unwrap();
    assert!(transported.converged);
    assert!(
        (transported.total_energy - hf.total_energy).abs() < 1e-9,
        "{name}: rigid energy transport"
    );
    let dipole: [f64; 3] = std::array::from_fn(|i| {
        (0..3)
            .map(|j| rotation[(i, j)] * p.dipole_moment_au[j])
            .sum()
    });
    for i in 0..3 {
        assert!(
            (transported.properties.dipole_moment_au[i] - dipole[i]).abs() < 1e-7,
            "{name}: rigid dipole transport {i}"
        );
        for j in 0..3 {
            let rotated_quad: f64 = (0..3)
                .flat_map(|a| (0..3).map(move |b| (a, b)))
                .map(|(a, b)| rotation[(i, a)] * p.quadrupole_moment_au[a][b] * rotation[(j, b)])
                .sum();
            let expected_quad = rotated_quad + 1.5 * (shift[i] * dipole[j] + shift[j] * dipole[i])
                - if i == j {
                    (0..3).map(|a| shift[a] * dipole[a]).sum::<f64>()
                } else {
                    0.
                };
            assert!(
                (transported.properties.quadrupole_moment_au[i][j] - expected_quad).abs() < 1e-6,
                "{name}: all-component quadrupole transport ({i},{j})"
            );
        }
    }
}

#[test]
fn water_structure_dipole_and_harmonic_spectrum() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/property_workflow_reference.json")).unwrap();
    compare_properties(&data["cases"][0]);
}

#[test]
fn nonlocal_ecp_rbh_structure_dipole_and_harmonic_spectrum() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/property_workflow_reference.json")).unwrap();
    compare_properties(&data["cases"][1]);
}

#[test]
fn all_multipole_components_transform_under_rigid_motion() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/property_workflow_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut atoms: Vec<Atom> = case["symbols"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, s)| {
                Atom::new(
                    s.as_str().unwrap(),
                    std::array::from_fn(|k| case["positions_bohr"][i][k].as_f64().unwrap()),
                )
            })
            .collect();
        let basis_name = if name == "water" {
            "6-31g*"
        } else {
            "def2-svp"
        };
        hf_rs::ecp::assign_builtin_ecps(&mut atoms, basis_name);
        let basis = build_basis(&atoms, &get_builtin_basis(basis_name).unwrap());
        let hf =
            run_scf_general(&atoms, &basis, 0, 300, 1e-13, ConvAlgo::Adiissoscf, None).unwrap();
        assert!(hf.converged);
        check_multipole_transport(name, &atoms, &hf);
    }
}
