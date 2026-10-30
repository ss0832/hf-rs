// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! FMO energies, complete cluster matrices, decomposition, all supported analytic
//! nuclear components, noncontiguous fragments, cache/stream identity and limits.

use hf_rs::{
    Atom, ContractedGaussian, FMOClusterResult, FMOFragment, FMOOptions, Matrix, build_basis,
    compute_fmo_gradient, get_builtin_basis, parse_gaussian94, run_fmo, run_scf_with_charge,
};

fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/fmo_reference.json")).unwrap()
}
fn problem(
    v: &serde_json::Value,
) -> (
    Vec<Atom>,
    Vec<ContractedGaussian>,
    Vec<FMOFragment>,
    FMOOptions,
) {
    let mut atoms: Vec<_> = v["atoms"]
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
    if !v["ecp"].is_null() {
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
    let map = if let Some(gbs) = v["gaussian94"].as_str() {
        parse_gaussian94(gbs)
    } else {
        get_builtin_basis(v["basis_name"].as_str().unwrap()).unwrap()
    };
    let basis = build_basis(&atoms, &map);
    let fragments = v["fragments"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            FMOFragment::new(
                &format!("Fragment {i}"),
                f.as_array()
                    .unwrap()
                    .iter()
                    .map(|a| a.as_u64().unwrap() as usize)
                    .collect(),
                v["charges"][i].as_i64().unwrap() as i32,
            )
        })
        .collect();
    let options = FMOOptions {
        order: v["order"].as_u64().unwrap() as usize,
        electrostatic_embedding: v["embedding"].as_bool().unwrap(),
        tolerance: 1e-13,
        residual_tolerance: 1e-10,
        density_tolerance: 1e-11,
        embedding_density_tolerance: 1e-10,
        ..Default::default()
    };
    (atoms, basis, fragments, options)
}
fn compare_matrix(actual: &Matrix, expected: &serde_json::Value) -> f64 {
    assert_eq!(actual.rows, expected.as_array().unwrap().len());
    let mut worst = 0f64;
    for i in 0..actual.rows {
        for j in 0..actual.cols {
            worst = worst.max((actual[(i, j)] - expected[i][j].as_f64().unwrap()).abs());
        }
    }
    worst
}
fn cluster<'a>(members: &[usize], r: &'a hf_rs::FMOResult) -> &'a FMOClusterResult {
    match members {
        [i] => &r.monomer_results[*i],
        [i, j] => {
            &r.dimer_results
                .iter()
                .find(|(pair, _)| *pair == (*i, *j))
                .unwrap()
                .1
        }
        [i, j, k] => {
            &r.trimer_results
                .as_ref()
                .unwrap()
                .iter()
                .find(|(triple, _)| *triple == (*i, *j, *k))
                .unwrap()
                .1
        }
        _ => panic!("invalid cluster"),
    }
}

#[test]
fn fmo_matches_independent_energies_every_matrix_and_all_supported_gradients() {
    let (mut energy_error, mut matrix_error, mut gradient_error) = (0f64, 0f64, 0f64);
    let (mut matrix_entries, mut gradient_components) = (0, 0);
    for case in data()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (atoms, basis, fragments, options) = problem(case);
        let r =
            run_fmo(&atoms, &basis, &fragments, &options).unwrap_or_else(|e| panic!("{name}: {e}"));
        let de = (r.total_energy - case["energy"].as_f64().unwrap()).abs();
        energy_error = energy_error.max(de);
        assert!(de < 1e-9, "{name}: energy {de:e}");
        assert!(r.converged && r.estimated_workspace_bytes <= options.workspace_limit_bytes);
        for reference in case["clusters"].as_array().unwrap() {
            let members: Vec<_> = reference["members"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let actual = cluster(&members, &r);
            let de =
                (actual.internal_energy - reference["internal_energy"].as_f64().unwrap()).abs();
            energy_error = energy_error.max(de);
            assert!(de < 1e-9, "{name}: cluster {members:?} energy {de:e}");
            for (matrix, key) in [
                (&actual.density_matrix, "density"),
                (&actual.fock_matrix, "fock"),
                (&actual.embedding_potential, "field"),
            ] {
                let error = compare_matrix(matrix, &reference[key]);
                matrix_error = matrix_error.max(error);
                matrix_entries += matrix.data.len();
                assert!(error < 1e-7, "{name}: cluster {members:?} {key} {error:e}");
            }
            assert!(
                actual.residual_norm < 2e-10,
                "{name}: residual {}",
                actual.residual_norm
            );
        }
        for (pair, expected) in r.ifie.iter().zip(case["pairs"].as_array().unwrap()) {
            for (value, key) in [
                (pair.total_interaction_energy, "total"),
                (pair.electrostatic, "electrostatic"),
                (pair.exchange_repulsion, "exchange"),
                (pair.charge_transfer, "relaxation"),
                (pair.embedding_correction, "embedding"),
            ] {
                let error = (value - expected[key].as_f64().unwrap()).abs();
                energy_error = energy_error.max(error);
                assert!(error < 1e-8, "{name}: {key} {error:e}");
            }
            assert!(
                (pair.electrostatic + pair.exchange_repulsion + pair.charge_transfer
                    - pair.total_interaction_energy)
                    .abs()
                    < 1e-12
            );
        }
        for (actual, expected) in r
            .trimer_energies
            .iter()
            .flatten()
            .zip(case["triples"].as_array().unwrap())
        {
            assert!((actual.1 - expected["correction"].as_f64().unwrap()).abs() < 1e-9);
        }
        let gradient = compute_fmo_gradient(&atoms, &basis, &fragments, &r);
        if case["gradient"].is_null() {
            assert!(gradient.unwrap_err().contains("response"));
        } else {
            let gradient = gradient.unwrap();
            for a in 0..atoms.len() {
                for k in 0..3 {
                    let error = (gradient[a][k] - case["gradient"][a][k].as_f64().unwrap()).abs();
                    gradient_error = gradient_error.max(error);
                    gradient_components += 1;
                    assert!(error < 2e-8, "{name}: nuclear component {a},{k}: {error:e}");
                }
            }
        }
        println!(
            "{name}: E={:.13}, embedding cycles {}",
            r.total_energy, r.embedding_iterations
        );
    }
    println!(
        "FMO max errors: energy {energy_error:e}; {matrix_entries} matrix entries {matrix_error:e}; {gradient_components} nuclear components {gradient_error:e}"
    );
}

#[test]
fn fmo_cache_stream_and_fragment_ao_order_agree() {
    let data = data();
    let case = data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "H8_four_fragments_FMO3")
        .unwrap();
    let (atoms, basis, mut fragments, mut options) = problem(case);
    let cached = run_fmo(&atoms, &basis, &fragments, &options).unwrap();
    assert!(cached.esp_cache_bytes > 0);
    options.esp_cache_limit_bytes = 0;
    let streamed = run_fmo(&atoms, &basis, &fragments, &options).unwrap();
    assert_eq!(streamed.esp_cache_bytes, 0);
    assert!((streamed.total_energy - cached.total_energy).abs() < 1e-11);
    for (a, b) in cached.monomer_results.iter().zip(&streamed.monomer_results) {
        assert!(
            a.density_matrix
                .data
                .iter()
                .zip(&b.density_matrix.data)
                .all(|(a, b)| (a - b).abs() < 1e-10)
        );
    }
    fragments.reverse();
    for f in &mut fragments {
        f.atom_indices.reverse();
    }
    let mut permuted = basis.clone();
    permuted.rotate_left(3);
    permuted.reverse();
    let reordered = run_fmo(&atoms, &permuted, &fragments, &options).unwrap();
    assert!((reordered.total_energy - cached.total_energy).abs() < 1e-10);
    let theta: f64 = 0.37;
    let moved: Vec<_> = atoms
        .iter()
        .map(|a| {
            let mut a = a.clone();
            let [x, y, z] = a.position;
            a.position = [
                theta.cos() * x - theta.sin() * y + 0.2,
                theta.sin() * x + theta.cos() * y - 0.3,
                z + 0.4,
            ];
            a
        })
        .collect();
    let moved_basis = build_basis(&moved, &get_builtin_basis("sto-3g").unwrap());
    let transformed = run_fmo(&moved, &moved_basis, &fragments, &options).unwrap();
    assert!((transformed.total_energy - cached.total_energy).abs() < 1e-10);
}

#[test]
fn full_fragment_order_is_rhf_and_invalid_inputs_fail() {
    let data = data();
    let case = &data["cases"][6];
    let (atoms, basis, fragments, mut options) = problem(case);
    let r = run_fmo(&atoms, &basis, &fragments, &options).unwrap();
    let hf = run_scf_with_charge(&atoms, &basis, 0, 300, 1e-13).unwrap();
    assert!((r.total_energy - hf.total_energy).abs() < 1e-10);
    options.workspace_limit_bytes = 1;
    assert!(
        run_fmo(&atoms, &basis, &fragments, &options)
            .unwrap_err()
            .contains("budget")
    );
    options = FMOOptions {
        order: 0,
        ..Default::default()
    };
    assert!(run_fmo(&atoms, &basis, &fragments, &options).is_err());
    options.order = 4;
    assert!(run_fmo(&atoms, &basis, &fragments, &options).is_err());
    options.order = 2;
    let mut bad = fragments.clone();
    bad[1].atom_indices.push(0);
    assert!(run_fmo(&atoms, &basis, &bad, &options).is_err());
    bad = fragments.clone();
    bad[1].atom_indices.pop();
    assert!(run_fmo(&atoms, &basis, &bad, &options).is_err());
    bad = fragments.clone();
    bad[0].multiplicity = 3;
    assert!(run_fmo(&atoms, &basis, &bad, &options).is_err());
    bad = fragments.clone();
    bad[0].charge += 1;
    assert!(run_fmo(&atoms, &basis, &bad, &options).is_err());
    options.tolerance = f64::NAN;
    assert!(run_fmo(&atoms, &basis, &fragments, &options).is_err());
}
