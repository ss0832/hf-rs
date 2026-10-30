// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    Atom, build_basis, get_builtin_basis,
    hf_variants::x2c_integrals::{spin_free_integrals, spin_free_nuclear_tangent},
};
use hf_rs::{Matrix, hf_variants::x2c_core::SpinFreeX2cCore};
const BUDGET: usize = 32 * 1024 * 1024;
const C: f64 = 137.035999084;
fn matrix(value: &serde_json::Value) -> Matrix {
    let rows = value.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect(),
    )
}

#[test]
fn every_x2c_matrix_entry_and_frechet_response_matches_independent_4c_oracles() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/x2c_core_reference.json")).unwrap();
    let mut count = [0; 2];
    let mut maximum = [0f64; 2];
    for case in data["cases"].as_array().unwrap() {
        let m: Vec<_> = case["matrices"]
            .as_array()
            .unwrap()
            .iter()
            .map(matrix)
            .collect();
        let core = SpinFreeX2cCore::build(&m[0], &m[1], &m[2], &m[3], C, BUDGET).unwrap();
        for (actual, expected) in core
            .hamiltonian
            .data
            .iter()
            .zip(matrix(&case["hamiltonian"]).data)
        {
            let error = (actual - expected).abs();
            maximum[0] = maximum[0].max(error);
            count[0] += 1;
            assert!(error < 1e-8, "{} core: {error:e}", case["name"]);
        }
        for point in case["directions"].as_array().unwrap() {
            let d: Vec<_> = point["tangents"]
                .as_array()
                .unwrap()
                .iter()
                .map(matrix)
                .collect();
            let actual = core.response(&d[0], &d[1], &d[2], &d[3]).unwrap();
            for (a, b) in actual.data.iter().zip(matrix(&point["response"]).data) {
                let error = (a - b).abs();
                maximum[1] = maximum[1].max(error);
                count[1] += 1;
                assert!(error < 1e-8, "{} response: {error:e}", case["name"]);
            }
        }
    }
    assert_eq!(count, [424, 1272]);
    println!("X2C all matrix/response counts {count:?}; maximum errors {maximum:?}");
}

#[test]
fn x2c_constant_potential_degenerate_response_and_nonrelativistic_limits_are_exact() {
    let s = Matrix::identity(3);
    let mut t = Matrix::zeros(3, 3);
    let tau = [0.4, 0.4, 1.1];
    for p in 0..3 {
        t[(p, p)] = tau[p];
    }
    let v = s.scale(-0.2);
    let w = t.scale(-0.4);
    let d = Matrix::from_vec(
        3,
        3,
        vec![0.1, -0.02, 0.03, -0.02, -0.07, 0.04, 0.03, 0.04, 0.02],
    );
    let zero = Matrix::zeros(3, 3);
    let core = SpinFreeX2cCore::build(&s, &t, &v, &w, C, BUDGET).unwrap();
    let derivative = core.response(&zero, &d, &zero, &d.scale(-0.4)).unwrap();
    for p in 0..3 {
        for q in 0..3 {
            let energy = 2. * tau[p] / ((1. + 2. * tau[p] / (C * C)).sqrt() + 1.) - 0.2;
            assert!((core.hamiltonian[(p, q)] - if p == q { energy } else { 0. }).abs() < 1e-10);
            let divided =
                2. / ((1. + 2. * tau[p] / (C * C)).sqrt() + (1. + 2. * tau[q] / (C * C)).sqrt());
            assert!((derivative[(p, q)] - d[(p, q)] * divided).abs() < 1e-10);
        }
    }
    let large = SpinFreeX2cCore::build(&s, &t, &v, &w, 1e6, BUDGET).unwrap();
    for (a, b) in large.hamiltonian.data.iter().zip(t.add(&v).data) {
        assert!((a - b).abs() < 1e-10);
    }
    assert!(SpinFreeX2cCore::build(&s, &t, &v, &w, 0., BUDGET).is_err());
    assert!(SpinFreeX2cCore::build(&s, &t, &v, &w, C, 1).is_err());
    assert!(SpinFreeX2cCore::build(&s.scale(-1.), &t, &v, &w, C, BUDGET).is_err());
}

#[test]
fn all_native_pvp_integrals_nuclear_tangents_and_x2c_core_derivatives_match_pyscf() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/x2c_core_reference.json")).unwrap();
    let mut base_error = [0f64; 4];
    let mut tangent_error = [0f64; 4];
    let mut core_error = 0f64;
    let mut counts = [0; 3];
    for case in data["cases"].as_array().unwrap() {
        let atoms: Vec<_> = case["atoms"]
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
        let basis = build_basis(
            &atoms,
            &get_builtin_basis(if case["basis_constant"] == "STO_3G_GBS" {
                "sto-3g"
            } else {
                "6-31g"
            })
            .unwrap(),
        );
        let ints = spin_free_integrals(&atoms, &basis, BUDGET).unwrap();
        for (k, actual) in [&ints.overlap, &ints.kinetic, &ints.potential, &ints.pvp]
            .into_iter()
            .enumerate()
        {
            let expected = matrix(&case["matrices"][k]);
            for (a, b) in actual.data.iter().zip(expected.data) {
                let e = (a - b).abs();
                base_error[k] = base_error[k].max(e);
                counts[0] += 1;
                assert!(
                    e < [1e-10, 1e-10, 1e-10, 1e-8][k],
                    "{} base {k}: {e:e}",
                    case["name"]
                );
            }
        }
        let core = SpinFreeX2cCore::build(
            &ints.overlap,
            &ints.kinetic,
            &ints.potential,
            &ints.pvp,
            C,
            BUDGET,
        )
        .unwrap();
        for point in case["nuclear"].as_array().unwrap() {
            let nucleus = point["nucleus"].as_u64().unwrap() as usize;
            let axis = point["axis"].as_u64().unwrap() as usize;
            let d = spin_free_nuclear_tangent(&atoms, &basis, nucleus, axis, BUDGET).unwrap();
            for (k, actual) in [&d.overlap, &d.kinetic, &d.potential, &d.pvp]
                .into_iter()
                .enumerate()
            {
                let expected = matrix(&point["tangents"][k]);
                for (a, b) in actual.data.iter().zip(expected.data) {
                    let e = (a - b).abs();
                    tangent_error[k] = tangent_error[k].max(e);
                    counts[1] += 1;
                    assert!(
                        e < [1e-10, 1e-10, 1e-9, 1e-7][k],
                        "{} tangent {nucleus}/{axis} operator {k}: {e:e}",
                        case["name"]
                    );
                }
            }
            let actual = core
                .response(&d.overlap, &d.kinetic, &d.potential, &d.pvp)
                .unwrap();
            for (a, b) in actual.data.iter().zip(matrix(&point["core_response"]).data) {
                let e = (a - b).abs();
                core_error = core_error.max(e);
                counts[2] += 1;
                assert!(
                    e < 1e-8,
                    "{} X2C nuclear {nucleus}/{axis}: {e:e}",
                    case["name"]
                );
            }
        }
    }
    assert_eq!(counts, [1696, 11892, 2973]);
    println!(
        "X2C native integral/tangent/core-response counts {counts:?}; errors {base_error:?}, {tangent_error:?}, {core_error:e}"
    );
}

#[test]
fn molecular_x2c_energies_and_every_analytic_nuclear_component_match_pyscf() {
    use hf_rs::{
        RelativisticHfOptions, RelativisticModel, compute_relativistic_hf_gradient,
        run_relativistic_hf_with_options,
    };
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/x2c_core_reference.json")).unwrap();
    let mut energy_error = 0f64;
    let mut analytic_error = 0f64;
    let mut stencil_error = 0f64;
    let mut translation = 0f64;
    let mut count = 0;
    for case in data["cases"].as_array().unwrap() {
        let atoms: Vec<_> = case["atoms"]
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
        let basis = build_basis(
            &atoms,
            &get_builtin_basis(if case["basis_constant"] == "STO_3G_GBS" {
                "sto-3g"
            } else {
                "6-31g"
            })
            .unwrap(),
        );
        let options = RelativisticHfOptions {
            max_iterations: 300,
            convergence: hf_rs::scf::RhfConvergence {
                energy: 1e-12,
                residual: 1e-12,
                density: 1e-12,
            },
            ..Default::default()
        };
        let result =
            run_relativistic_hf_with_options(&atoms, &basis, 0, RelativisticModel::X2C, &options)
                .unwrap();
        let error = (result.total_energy - case["e_total"].as_f64().unwrap()).abs();
        energy_error = energy_error.max(error);
        assert!(error < 1e-8, "{} X2C energy {error:e}", case["name"]);
        assert!(
            result.residual_norm <= options.convergence.residual
                && result.density_change <= options.convergence.density
        );
        let gradient = compute_relativistic_hf_gradient(&atoms, &basis, &result).unwrap();
        for point in case["nuclear"].as_array().unwrap() {
            let a = point["nucleus"].as_u64().unwrap() as usize;
            let axis = point["axis"].as_u64().unwrap() as usize;
            let analytic = (gradient[a][axis] - case["gradient"][a][axis].as_f64().unwrap()).abs();
            let stencil =
                (gradient[a][axis] - point["five_point_gradient"].as_f64().unwrap()).abs();
            analytic_error = analytic_error.max(analytic);
            stencil_error = stencil_error.max(stencil);
            count += 1;
            assert!(
                analytic < 1e-6 && stencil < 1e-6,
                "{} nuclear {a}/{axis}: analytic {analytic:e}, stencil {stencil:e}",
                case["name"]
            );
        }
        for axis in 0..3 {
            let value = gradient.iter().map(|g| g[axis]).sum::<f64>().abs();
            translation = translation.max(value);
            assert!(value < 1e-9);
        }
        let map = get_builtin_basis(if case["basis_constant"] == "STO_3G_GBS" {
            "sto-3g"
        } else {
            "6-31g"
        })
        .unwrap();
        let compatibility = hf_rs::evaluate_hf_variant_energy(
            &atoms,
            &map,
            hf_rs::HFVariantType::RelativisticHF,
            0,
            300,
            1e-12,
        )
        .unwrap();
        assert!((compatibility - case["e_total"].as_f64().unwrap()).abs() < 1e-8);
        let compatibility = hf_rs::compute_hf_variant_analytical_gradient(
            &atoms,
            &map,
            hf_rs::HFVariantType::RelativisticHF,
            0,
            300,
            1e-12,
        )
        .unwrap();
        for point in case["nuclear"].as_array().unwrap() {
            let a = point["nucleus"].as_u64().unwrap() as usize;
            let axis = point["axis"].as_u64().unwrap() as usize;
            assert!(
                (compatibility[a][axis] - point["five_point_gradient"].as_f64().unwrap()).abs()
                    < 1e-6
            );
        }
    }
    assert_eq!(count, 27);
    println!(
        "Molecular X2C: five energies {energy_error:e}; all {count} nuclear components analytic {analytic_error:e}, FD {stencil_error:e}, translation {translation:e}"
    );
}

#[test]
fn x2c_rank_reduction_preserves_energy_and_rejects_unimplemented_projector_response() {
    use hf_rs::{
        RelativisticHfOptions, RelativisticModel, compute_relativistic_hf_gradient,
        run_relativistic_hf_with_options,
    };
    let atoms = vec![
        Atom::new("H", [0., 0., -0.7]),
        Atom::new("H", [0., 0., 0.7]),
    ];
    let mut basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let mut options = RelativisticHfOptions::default();
    let reference =
        run_relativistic_hf_with_options(&atoms, &basis, 0, RelativisticModel::X2C, &options)
            .unwrap();
    basis.push(basis[0].clone());
    let reduced =
        run_relativistic_hf_with_options(&atoms, &basis, 0, RelativisticModel::X2C, &options)
            .unwrap();
    assert_eq!(
        (reduced.mo_coefficients.rows, reduced.mo_coefficients.cols),
        (3, 2)
    );
    assert!((reduced.total_energy - reference.total_energy).abs() < 1e-10);
    assert!(compute_relativistic_hf_gradient(&atoms, &basis, &reduced).is_err());
    assert!(
        run_relativistic_hf_with_options(&atoms, &basis, 1, RelativisticModel::X2C, &options)
            .is_err()
    );
    options.max_work_bytes = 1;
    assert!(
        run_relativistic_hf_with_options(&atoms, &basis, 0, RelativisticModel::X2C, &options)
            .is_err()
    );
    options = RelativisticHfOptions::default();
    options.max_iterations = 0;
    assert!(
        run_relativistic_hf_with_options(&atoms, &basis, 0, RelativisticModel::X2C, &options)
            .is_err()
    );
    assert_eq!(
        hf_rs::HFVariantType::parse("dkh2").unwrap(),
        hf_rs::HFVariantType::Dkh2HF
    );
    assert_eq!(
        hf_rs::HFVariantType::parse("zora").unwrap(),
        hf_rs::HFVariantType::ZoraHF
    );
}
