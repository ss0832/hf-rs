// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Coherent-state QED-RHF: complete independent matrix and nuclear-gradient comparisons,
//! exact quadratic moments, photon displacement, and coordinate/frequency invariances.

use hf_rs::hf_variants::qed_hf::{build_qed_hf_fock, qed_dipole_operators};
use hf_rs::{
    Atom, ContractedGaussian, Matrix, QedHfOptions, build_basis, compute_qed_hf_gradient,
    get_builtin_basis, parse_gaussian94, run_qed_hf_with_options,
};

fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/qed_hf_reference.json")).unwrap()
}

fn problem(case: &serde_json::Value) -> (Vec<Atom>, Vec<ContractedGaussian>, QedHfOptions) {
    let mut atoms: Vec<Atom> = case["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            Atom::new(
                a["symbol"].as_str().unwrap(),
                std::array::from_fn(|k| a["position"][k].as_f64().unwrap()),
            )
        })
        .collect();
    if !case["ecp"].is_null() {
        use hf_rs::ecp::{EcpDef, EcpTerm};
        atoms[0].ecp = Some(std::sync::Arc::new(EcpDef {
            symbol: "He".into(),
            n_core: 0,
            l_max: 2,
            local: vec![EcpTerm {
                power: 2,
                exponent: 0.7,
                coefficient: 0.17,
            }],
            semilocal: vec![
                vec![EcpTerm {
                    power: 2,
                    exponent: 0.53,
                    coefficient: 0.11,
                }],
                vec![EcpTerm {
                    power: 2,
                    exponent: 0.61,
                    coefficient: -0.07,
                }],
            ],
        }));
    }
    let map = if let Some(gbs) = case["gaussian94"].as_str() {
        parse_gaussian94(gbs)
    } else {
        get_builtin_basis(case["basis_name"].as_str().unwrap()).unwrap()
    };
    let basis = build_basis(&atoms, &map);
    let options = QedHfOptions {
        cavity_frequency: case["frequency"].as_f64().unwrap(),
        coupling: std::array::from_fn(|k| case["coupling"][k].as_f64().unwrap()),
        max_iterations: 300,
        energy_tol: 1e-13,
        residual_tol: 1e-10,
        density_tol: 1e-11,
    };
    (atoms, basis, options)
}

fn matrix(value: &serde_json::Value) -> Matrix {
    let rows = value.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect(),
    )
}

#[test]
fn coherent_qed_hf_matches_all_independent_matrices_and_energies() {
    let (mut energy_error, mut matrix_error) = (0f64, 0f64);
    let mut entries = 0;
    for case in data()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (atoms, basis, options) = problem(case);
        let result = run_qed_hf_with_options(
            &atoms,
            &basis,
            case["charge"].as_i64().unwrap() as i32,
            &options,
        )
        .unwrap();
        let de = (result.total_energy - case["energy"].as_f64().unwrap()).abs();
        energy_error = energy_error.max(de);
        assert!(
            de < 1e-9,
            "{name}: E {}, expected {}",
            result.total_energy,
            case["energy"]
        );
        assert!(
            result.residual_norm < options.residual_tol
                && result.density_change < options.density_tol
        );
        assert!(
            (result.total_energy
                - result.electronic_energy
                - result.nuclear_energy
                - result.cavity_energy)
                .abs()
                < 1e-11
        );
        assert!((result.cavity_energy - case["cavity_energy"].as_f64().unwrap()).abs() < 1e-9);
        assert!(
            (result.dipole_self_energy - case["bare_dipole_self_energy"].as_f64().unwrap()).abs()
                < 1e-9
        );
        assert!(
            (result.coherent_displacement - case["coherent_displacement"].as_f64().unwrap()).abs()
                < 1e-9
        );
        assert!((result.mean_photon_number - result.coherent_displacement.powi(2)).abs() < 1e-14);
        let (dipole, quadratic) = qed_dipole_operators(&basis, options.coupling).unwrap();
        let (_, _, _, h) = hf_rs::scf::compute_one_electron_integrals(&atoms, &basis);
        let eri = hf_rs::scf::compute_eri_tensor(&basis);
        let arbitrary = matrix(&case["matrices"]["arbitrary_density"]);
        let arbitrary_fock = build_qed_hf_fock(&h, &arbitrary, &eri, &dipole, &quadratic);
        for (label, actual) in [
            ("dipole", &dipole),
            ("quadratic", &quadratic),
            ("density", &result.density_matrix),
            ("fock", &result.fock_matrix),
            ("arbitrary_fock", &arbitrary_fock),
        ] {
            let expected = matrix(&case["matrices"][label]);
            for i in 0..actual.rows {
                for j in 0..actual.cols {
                    let error = (actual[(i, j)] - expected[(i, j)]).abs();
                    matrix_error = matrix_error.max(error);
                    assert!(error < 1e-8, "{name}: {label} ({i},{j}) {error:e}");
                    entries += 1;
                }
            }
        }
        if name == "He_complete_occupation" {
            // A single s AO has zero projected transition dipole at its own center,
            // but the exact position variance and cavity correction remain positive.
            let projected = dipole
                .dot(&result.overlap_matrix.inverse().unwrap())
                .dot(&dipole);
            assert!(quadratic[(0, 0)] - projected[(0, 0)] > 1e-3);
            assert!(result.cavity_energy > 1e-3);
        }
    }
    println!(
        "QED-HF: 7 energies max error {energy_error:e}; {entries} matrix entries max error {matrix_error:e}"
    );
}

#[test]
fn every_qed_hf_nuclear_component_matches_independent_refined_energies() {
    let (mut worst, mut translation) = (0f64, 0f64);
    let mut components = 0;
    for case in data()["cases"].as_array().unwrap() {
        let (atoms, basis, options) = problem(case);
        let result = run_qed_hf_with_options(
            &atoms,
            &basis,
            case["charge"].as_i64().unwrap() as i32,
            &options,
        )
        .unwrap();
        let gradient = compute_qed_hf_gradient(&atoms, &basis, &result).unwrap();
        for (a, g) in gradient.iter().enumerate() {
            for (k, actual) in g.iter().enumerate() {
                let error = (actual - case["gradient"][a][k].as_f64().unwrap()).abs();
                worst = worst.max(error);
                assert!(
                    error < 3e-7,
                    "{}: gradient ({a},{k}) {error:e}",
                    case["name"]
                );
                components += 1;
            }
        }
        for k in 0..3 {
            translation = translation.max(gradient.iter().map(|g| g[k]).sum::<f64>().abs());
        }
    }
    assert!(translation < 1e-9, "QED-HF translation: {translation:e}");
    println!(
        "QED-HF: all {components} analytic nuclear components max error {worst:e}; translation {translation:e}"
    );
}

#[test]
fn coherent_photon_frequency_and_rigid_coordinate_invariances() {
    for index in [2, 5, 6] {
        let cases = data();
        let case = &cases["cases"][index];
        let (atoms, basis, options) = problem(case);
        let charge = case["charge"].as_i64().unwrap() as i32;
        let first = run_qed_hf_with_options(&atoms, &basis, charge, &options).unwrap();
        let doubled = run_qed_hf_with_options(
            &atoms,
            &basis,
            charge,
            &QedHfOptions {
                cavity_frequency: 2. * options.cavity_frequency,
                ..options.clone()
            },
        )
        .unwrap();
        assert!((first.total_energy - doubled.total_energy).abs() < 1e-12);
        assert!((first.mean_photon_number - 2. * doubled.mean_photon_number).abs() < 1e-12);
        let opposite = run_qed_hf_with_options(
            &atoms,
            &basis,
            charge,
            &QedHfOptions {
                coupling: options.coupling.map(|v| -v),
                ..options.clone()
            },
        )
        .unwrap();
        assert!((first.total_energy - opposite.total_energy).abs() < 1e-12);
        assert!((first.coherent_displacement + opposite.coherent_displacement).abs() < 1e-12);
        let r = Matrix::from_vec(3, 3, vec![0.36, -0.48, 0.8, 0.8, 0.6, 0., -0.48, 0.64, 0.6]);
        let shift = [0.7, -0.4, 0.3];
        let mut moved = atoms.clone();
        for atom in &mut moved {
            atom.position = std::array::from_fn(|i| {
                shift[i] + (0..3).map(|j| r[(i, j)] * atom.position[j]).sum::<f64>()
            });
        }
        let map = if let Some(gbs) = case["gaussian94"].as_str() {
            parse_gaussian94(gbs)
        } else {
            get_builtin_basis(case["basis_name"].as_str().unwrap()).unwrap()
        };
        let moved_basis = build_basis(&moved, &map);
        let transported = run_qed_hf_with_options(
            &moved,
            &moved_basis,
            charge,
            &QedHfOptions {
                coupling: std::array::from_fn(|i| {
                    (0..3).map(|j| r[(i, j)] * options.coupling[j]).sum()
                }),
                ..options.clone()
            },
        )
        .unwrap();
        assert!(
            (first.total_energy - transported.total_energy).abs() < 1e-10,
            "{}: coherent origin/rotation invariance",
            case["name"]
        );
        for i in 0..3 {
            let expected: f64 = (0..3)
                .map(|j| r[(i, j)] * first.expected_dipole_bohr[j])
                .sum::<f64>()
                + f64::from(charge) * shift[i];
            assert!((transported.expected_dipole_bohr[i] - expected).abs() < 1e-8);
        }
    }
}

#[test]
fn qed_hf_dispatch_and_unsupported_controls_are_explicit() {
    let atoms = vec![
        Atom::new("H", [0., 0., -0.7]),
        Atom::new("H", [0., 0., 0.7]),
    ];
    let map = get_builtin_basis("sto-3g").unwrap();
    let basis = build_basis(&atoms, &map);
    let options = QedHfOptions::default();
    let result = run_qed_hf_with_options(&atoms, &basis, 0, &options).unwrap();
    let gradient = compute_qed_hf_gradient(&atoms, &basis, &result).unwrap();
    let dispatched = hf_rs::compute_hf_variant_analytical_gradient(
        &atoms,
        &map,
        hf_rs::HFVariantType::QEDHF,
        0,
        options.max_iterations,
        options.energy_tol,
    )
    .unwrap();
    for a in 0..2 {
        for k in 0..3 {
            assert!((gradient[a][k] - dispatched[a][k]).abs() < 1e-9);
        }
    }
    assert!(
        run_qed_hf_with_options(
            &atoms,
            &basis,
            0,
            &QedHfOptions {
                cavity_frequency: 0.,
                ..options.clone()
            }
        )
        .is_err()
    );
    assert!(
        run_qed_hf_with_options(
            &atoms,
            &basis,
            0,
            &QedHfOptions {
                coupling: [f64::NAN, 0., 0.],
                ..options.clone()
            }
        )
        .is_err()
    );
    assert!(run_qed_hf_with_options(&atoms, &basis, 1, &options).is_err());
    let mut redundant = basis.clone();
    redundant.push(basis[0].clone());
    let reduced = run_qed_hf_with_options(&atoms, &redundant, 0, &options).unwrap();
    assert!((result.total_energy - reduced.total_energy).abs() < 1e-11);
    assert!(
        compute_qed_hf_gradient(&atoms, &redundant, &reduced)
            .unwrap_err()
            .contains("projector response")
    );
}
