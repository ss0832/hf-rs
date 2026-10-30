// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Local multiplicative LHF/CEDA: independent grids, all matrix entries,
//! exact two-electron limits, cache/stream and occupied-space gauge invariance.

use hf_rs::{
    Atom, ContractedGaussian, ConvAlgo, LhfOptions, LocalGridOptions, Matrix, build_basis,
    evaluate_lhf_exchange, get_builtin_basis, parse_gaussian94, run_lhf_from_reference,
    run_scf_general,
};

fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/lhf_reference.json")).unwrap()
}
fn problem(case: &serde_json::Value) -> (Vec<Atom>, Vec<ContractedGaussian>, hf_rs::HFResult) {
    let mut atoms: Vec<_> = case["atoms"]
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
        let term = |exponent, coefficient| EcpTerm {
            power: 2,
            exponent,
            coefficient,
        };
        atoms[0].ecp = Some(std::sync::Arc::new(EcpDef {
            symbol: "He".into(),
            n_core: 0,
            l_max: 2,
            local: vec![term(0.7, 0.17)],
            semilocal: vec![vec![term(0.53, 0.11)], vec![term(0.61, -0.07)]],
        }));
    }
    let map = if let Some(gbs) = case["gaussian94"].as_str() {
        parse_gaussian94(gbs)
    } else {
        get_builtin_basis(case["basis_name"].as_str().unwrap()).unwrap()
    };
    let basis = build_basis(&atoms, &map);
    let hf = run_scf_general(
        &atoms,
        &basis,
        case["charge"].as_i64().unwrap() as i32,
        300,
        1e-13,
        ConvAlgo::Adiissoscf,
        None,
    )
    .unwrap();
    (atoms, basis, hf)
}
fn compare(actual: &Matrix, expected: &serde_json::Value) -> f64 {
    let mut error = 0f64;
    for i in 0..actual.rows {
        for j in 0..actual.cols {
            error = error.max((actual[(i, j)] - expected[i][j].as_f64().unwrap()).abs());
        }
    }
    error
}
fn fine() -> LhfOptions {
    LhfOptions {
        energy_tolerance: 1e-12,
        residual_tolerance: 1e-9,
        density_tolerance: 1e-10,
        grid: LocalGridOptions {
            radial_points: 80,
            polar_points: 28,
            azimuthal_points: 56,
            overlap_tolerance: 1e-7,
            workspace_limit_bytes: 512 * 1024 * 1024,
            ..Default::default()
        },
        max_grid_refinements: 1,
        ..Default::default()
    }
}

#[test]
fn lhf_matches_independent_grid_energies_and_every_local_potential_entry() {
    let (mut energy_error, mut matrix_error) = (0f64, 0f64);
    for case in data()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (atoms, basis, hf) = problem(case);
        let o = fine();
        let initial = evaluate_lhf_exchange(&atoms, &basis, &hf, &o)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        for (matrix, key) in [
            (&initial.potential, "reference_potential"),
            (&initial.slater, "reference_slater"),
        ] {
            let error = compare(matrix, &case[key]);
            matrix_error = matrix_error.max(error);
            assert!(error < 5e-6, "{name} {key}: {error:e}");
        }
        let r = run_lhf_from_reference(&atoms, &basis, &hf, &o)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let error = (r.total_energy - case["energy"].as_f64().unwrap()).abs();
        energy_error = energy_error.max(error);
        assert!(error < 2e-8, "{name}: energy {error:e}");
        for (matrix, key) in [
            (&r.density_matrix, "density"),
            (&r.ks_fock_matrix, "fock"),
            (&r.local_exchange.potential, "potential"),
            (&r.local_exchange.slater, "slater"),
        ] {
            let error = compare(matrix, &case[key]);
            matrix_error = matrix_error.max(error);
            assert!(error < 5e-6, "{name} {key}: {error:e}");
        }
        for (value, expected) in r
            .orbital_energies
            .iter()
            .zip(case["orbital_energies"].as_array().unwrap())
        {
            assert!(
                (value - expected.as_f64().unwrap()).abs() < 5e-6,
                "{name}: KS eigenvalue mismatch"
            );
        }
        assert!(r.residual_norm <= o.residual_tolerance && r.density_change <= o.density_tolerance);
        // A single determinant cannot lie below RHF.
        assert!(r.total_energy >= hf.total_energy - 1e-10, "{name}: below RHF");
        if hf.n_occ == 1 {
            assert!((r.total_energy - hf.total_energy).abs() < 1e-10);
            assert_eq!(r.local_exchange.grid_points, 0);
        } else {
            assert!(r.local_exchange.grid_points > 0);
        }
        assert_eq!(
            r.homo_lumo_gap.is_none(),
            hf.n_occ == hf.mo_coefficients.cols
        );
        println!(
            "{name}: E {}, grid {}, overlap {}, linear {}, cache {}",
            r.total_energy,
            r.local_exchange.grid_points,
            r.local_exchange.grid_overlap_error,
            r.local_exchange.linear_residual,
            r.local_exchange.cached_kernel_points
        );
    }
    println!("LHF maximum energy error {energy_error:e}, matrix error {matrix_error:e}");
}

#[test]
fn local_exchange_cache_stream_and_occupied_rotation_agree() {
    let data = data();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "water_sto3g")
        .unwrap();
    let (atoms, basis, hf) = problem(case);
    let mut o = LhfOptions::default();
    let reference = evaluate_lhf_exchange(&atoms, &basis, &hf, &o).unwrap();
    assert!(reference.cached_kernel_points > 0);
    o.grid.kernel_cache_limit_bytes = 0;
    let streamed = evaluate_lhf_exchange(&atoms, &basis, &hf, &o).unwrap();
    assert_eq!(streamed.cached_kernel_points, 0);
    assert!(
        reference
            .potential
            .data
            .iter()
            .zip(&streamed.potential.data)
            .all(|(a, b)| (a - b).abs() < 1e-12)
    );
    let mut rotated = hf.clone();
    let angle: f64 = 0.31;
    let mut rotation = Matrix::identity(hf.mo_coefficients.cols);
    rotation[(0, 0)] = angle.cos();
    rotation[(hf.n_occ - 1, hf.n_occ - 1)] = angle.cos();
    rotation[(0, hf.n_occ - 1)] = -angle.sin();
    rotation[(hf.n_occ - 1, 0)] = angle.sin();
    rotated.mo_coefficients = hf.mo_coefficients.dot(&rotation);
    let point = evaluate_lhf_exchange(&atoms, &basis, &rotated, &o).unwrap();
    let difference = point.potential.sub(&reference.potential);
    let occupied = hf.mo_coefficients.leading_columns(hf.n_occ);
    let projected = occupied.transpose().dot(&difference).dot(&occupied);
    let constant = (0..hf.n_occ).map(|i| projected[(i, i)]).sum::<f64>() / hf.n_occ as f64;
    let residual = difference.sub(&hf.overlap_matrix.scale(constant));
    let error = residual.data.iter().fold(0f64, |m, v| m.max(v.abs()));
    assert!(
        error < 1e-5,
        "occupied rotation must change only the potential gauge, residual {error:e}"
    );
    o.grid.workspace_limit_bytes = 1;
    assert!(
        evaluate_lhf_exchange(&atoms, &basis, &hf, &o)
            .unwrap_err()
            .contains("budget")
    );
}
