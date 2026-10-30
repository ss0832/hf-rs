// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::post_hf::mr_dsrg::imsrg2::{No2bOperator, integrate_imsrg2, no2b_commutator};
use hf_rs::{
    context::PostHFContext,
    options::PostHFOptions,
    registry::{MethodId, create_solver},
    *,
};
fn vector(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn operator(v: &serde_json::Value) -> No2bOperator {
    let rows = v["one"].as_array().unwrap();
    No2bOperator {
        zero: v["zero"].as_f64().unwrap(),
        one: Matrix::from_vec(
            rows.len(),
            rows.len(),
            rows.iter().flat_map(vector).collect(),
        ),
        two: vector(&v["two"]),
    }
}
fn compare(a: &No2bOperator, b: &No2bOperator, tol: f64) -> f64 {
    let mut maximum = 0f64;
    for (index, (x, y)) in std::iter::once(&a.zero)
        .chain(&a.one.data)
        .chain(&a.two)
        .zip(std::iter::once(&b.zero).chain(&b.one.data).chain(&b.two))
        .enumerate()
    {
        maximum = maximum.max((x - y).abs());
        assert!(
            (x - y).abs() < tol,
            "component {index}: {x:e} vs {y:e}, error {:e}",
            x - y
        );
    }
    maximum
}
fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/imsrg2_reference.json")).unwrap()
}

#[test]
fn every_no2b_commutator_component_matches_full_fock_space_with_induced_three_body() {
    let mut maximum = 0f64;
    for case in data()["commutators"].as_array().unwrap() {
        let occupied = case["occupied"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect::<Vec<_>>();
        assert!(case["induced_bare_three_max"].as_f64().unwrap() > 1.);
        let actual =
            no2b_commutator(&operator(&case["a"]), &operator(&case["b"]), &occupied).unwrap();
        maximum = maximum.max(compare(&actual, &operator(&case["c"]), 1e-10));
    }
    println!("NO2B: all 2666 zero/one/two-body components verified, max error {maximum:e}");
}
#[test]
fn all_flowing_components_match_independent_dop853_and_step_refinement() {
    for case in data()["flows"].as_array().unwrap() {
        let kind = if case["kind"] == "Wegner" {
            SRGGenerator::Wegner
        } else {
            SRGGenerator::White
        };
        let occupied = case["occupied"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect::<Vec<_>>();
        let initial = operator(&case["initial"]);
        let end = case["end"].as_f64().unwrap();
        let expected = operator(&case["final"]);
        let (fine, _) = integrate_imsrg2(&initial, &occupied, kind, end, 200, None).unwrap();
        let error = compare(&fine, &expected, 1e-9);
        let (adaptive, steps) =
            integrate_imsrg2(&initial, &occupied, kind, end, 10, Some(1e-12)).unwrap();
        let adaptive_error = compare(&adaptive, &expected, 1e-9);
        assert!(fine.zero < 0.);
        assert!(fine.off_diagonal_norm(&occupied) < initial.off_diagonal_norm(&occupied));
        let (coarse, _) = integrate_imsrg2(&initial, &occupied, kind, end, 100, None).unwrap();
        let coarse_error = std::iter::once(coarse.zero - expected.zero)
            .chain(
                coarse
                    .one
                    .data
                    .iter()
                    .zip(&expected.one.data)
                    .map(|(a, b)| a - b),
            )
            .chain(coarse.two.iter().zip(&expected.two).map(|(a, b)| a - b))
            .map(f64::abs)
            .fold(0f64, f64::max);
        assert!(
            error < coarse_error * 0.2,
            "{kind:?}: coarse={coarse_error:e}, fine={error:e}"
        );
        let (zero, steps0) = integrate_imsrg2(&initial, &occupied, kind, 0., 10, None).unwrap();
        compare(&zero, &initial, 1e-15);
        assert_eq!(steps0, 0);
        assert!(integrate_imsrg2(&initial, &occupied, kind, end, 0, None).is_err());
        assert!(integrate_imsrg2(&initial, &occupied, kind, -end, 10, None).is_err());
        println!(
            "{kind:?}: all 1333 flow components, error {error:e}, adaptive {adaptive_error:e}, accepted steps {steps}"
        );
    }
}
#[test]
fn rhf_wrapper_and_typed_generator_options_match_independent_molecular_flows() {
    let atoms = [
        Atom::new("H", [0.1, -0.2, -0.7]),
        Atom::new("H", [-0.1, 0.2, 0.7]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_scf(&atoms, &basis, 100, 1e-14).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let data = data();
    assert!((hf.total_energy - data["h2"]["e_scf"].as_f64().unwrap()).abs() < 1e-9);
    for kind in [SRGGenerator::Wegner, SRGGenerator::White] {
        let key = format!("{kind:?}");
        let expected = data["h2"]["correlation"][&key].as_f64().unwrap();
        let actual = SRGODE::new(&hf, kind, 0.5, 500).compute().unwrap();
        assert!((actual.correlation_energy - expected).abs() < 1e-9);
        let mut options = PostHFOptions::default();
        options.dsrg.ode_generator = kind;
        options.dsrg.ode_tolerance = 1e-12;
        let output = create_solver(MethodId::SrGode, &ctx, &options)
            .unwrap()
            .compute()
            .unwrap();
        assert!((output.e_corr - expected).abs() < 1e-9);
        assert!((output.e_total - hf.total_energy - expected).abs() < 1e-9);
    }
    let map =
        std::collections::BTreeMap::from([("ode_generator".into(), serde_json::json!("white"))]);
    assert_eq!(
        PostHFOptions::from_map(&map, &MethodId::SrGode.option_schemas())
            .unwrap()
            .dsrg
            .ode_generator,
        SRGGenerator::White
    );
    assert!(PostHFOptions::from_map(&map, &MethodId::SrDsrgPt2.option_schemas()).is_err());
    assert!(
        SRGODE::new(&hf, SRGGenerator::Wegner, -0.1, 10)
            .compute()
            .is_err()
    );
}

#[test]
fn adaptive_stiff_flow_rejects_overflowing_trials_and_white_singularities_are_errors() {
    let occupied = [true, false];
    let gap = 1e4f64;
    let coupling = 10f64;
    let initial = No2bOperator {
        zero: 0.,
        one: Matrix::from_vec(2, 2, vec![-gap, coupling, coupling, gap]),
        two: vec![0.; 16],
    };
    let (final_state, steps) = integrate_imsrg2(
        &initial,
        &occupied,
        SRGGenerator::Wegner,
        1e-6,
        1,
        Some(1e-15),
    )
    .unwrap();
    let exact = -coupling.powi(2) / (gap.hypot(coupling) + gap);
    assert!((final_state.zero - exact).abs() < 1e-8);
    assert!(steps > 1);
    let singular = No2bOperator {
        one: Matrix::from_vec(2, 2, vec![1., 0.1, 0.1, 1.]),
        ..initial.clone()
    };
    assert!(integrate_imsrg2(&singular, &occupied, SRGGenerator::White, 0.1, 10, None).is_err());
    let malformed = No2bOperator {
        zero: 0.,
        one: Matrix {
            rows: usize::MAX,
            cols: usize::MAX,
            data: vec![],
        },
        two: vec![],
    };
    assert!(malformed.validate(&[]).is_err());
}
