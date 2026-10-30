// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::post_hf::f12::operators::GaussianGeminal;
use hf_rs::post_hf::f12::{
    CABSBasis, F12Approximation, F12Hybrid, GeminalCorrelationFactor, MP2F12Options, RHFMP2F12,
};
use hf_rs::post_hf::{PostHFMethod, PostHFResult};
use hf_rs::*;

fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect(),
    )
}
fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 1e-10,
        "{a} vs {b}, difference {}",
        (a - b).abs()
    );
}
fn helium(name: &str) -> (Vec<Atom>, Vec<ContractedGaussian>, HFResult, CABSBasis) {
    let mut atoms = vec![Atom::new("He", [0.13, -0.21, 0.31])];
    if name.contains("ecp") {
        atoms[0].ecp = Some(std::sync::Arc::new(ecp::EcpDef {
            symbol: "He".into(),
            n_core: 0,
            l_max: usize::from(name == "he_nonlocal_ecp"),
            local: vec![ecp::EcpTerm {
                power: 2,
                exponent: 0.7,
                coefficient: 0.17,
            }],
            semilocal: if name == "he_nonlocal_ecp" {
                vec![vec![ecp::EcpTerm {
                    power: 2,
                    exponent: 0.53,
                    coefficient: 0.11,
                }]]
            } else {
                vec![]
            },
        }));
    }
    let obs = build_basis(
        &atoms,
        &parse_gaussian94("He 0\nS 1 1\n1.4 1\nS 1 1\n0.45 1\n****"),
    );
    let aux = build_basis(
        &atoms,
        &parse_gaussian94("He 0\nS 1 1\n0.2 1\nS 1 1\n0.9 1\nS 1 1\n3.2 1\n****"),
    );
    let hf = run_scf(&atoms, &obs, 150, 1e-14).unwrap();
    let cabs = CABSBasis::from_reference(&hf, &obs, &aux, 1e-8).unwrap();
    (atoms, obs, hf, cabs)
}
#[test]
fn molecular_3c_energies_and_every_intermediate_match_radial_wavefunction_oracle_with_ecps() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/mp2_f12_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        println!("MP2-F12 case {name}");
        let (atoms, obs, hf, cabs) = helium(name);
        let factor = GaussianGeminal::new(
            case["terms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| (t[0].as_f64().unwrap(), t[1].as_f64().unwrap()))
                .collect(),
        )
        .unwrap();
        let options = MP2F12Options {
            custom_cabs: Some(cabs.clone()),
            custom_geminal: if name.starts_with("he_slater_fit") {
                None
            } else {
                Some(factor)
            },
            gamma: case["gamma"].as_f64().unwrap_or(1.0),
            ..Default::default()
        };
        let result = RHFMP2F12::new(&hf, &atoms, &obs)
            .with_options(options.clone())
            .compute()
            .unwrap();
        for (value, key) in [
            (result.hf_energy, "e_scf"),
            (result.mp2_correlation_energy, "e_mp2"),
            (result.f12_correlation_energy, "e_f12"),
            (result.cabs_singles_energy, "e_cabs"),
            (result.total_energy, "e_total"),
        ] {
            close(value, case[key].as_f64().unwrap());
        }
        assert_eq!(result.pairs.len(), 1);
        let pair = &result.pairs[0];
        for (values, key) in [(pair.v, "v"), (pair.x, "x"), (pair.b, "b")] {
            for value in values {
                close(value, case[key].as_f64().unwrap());
            }
        }
        for (values, key) in [(&pair.c, "c"), (&pair.g, "g")] {
            let expected = matrix(&case[key]);
            assert_eq!((values.rows, values.cols), (expected.rows, expected.cols));
            for (&a, &b) in values.data.iter().zip(&expected.data) {
                close(a, b);
            }
        }
        close(result.f12_triplet_energy, 0.0);
        let output = result.to_output().unwrap();
        close(output.e_scf, hf.total_energy);
        close(
            output.e_corr,
            result.mp2_correlation_energy
                + result.f12_correlation_energy
                + result.cabs_singles_energy,
        );
        close(
            output.components["E_CABS_singles"],
            result.cabs_singles_energy,
        );
        let a = RHFMP2F12::new(&hf, &atoms, &obs)
            .with_options(MP2F12Options {
                approximation: F12Approximation::ApproximationA,
                ..options
            })
            .compute_checked();
        let a = a.unwrap();
        close(
            a.f12_correlation_energy,
            case["three_star_a"]["e_f12"].as_f64().unwrap(),
        );
        for value in a.pairs[0].b_bar {
            close(value, case["three_star_a"]["b_bar"].as_f64().unwrap());
        }
        assert!(a.pairs[0].c.data.iter().all(|&value| value == 0.0));
        if name == "he_nonlocal_ecp" {
            assert!(
                case["three_star_a"]["nonlocal_b_bar_correction"]
                    .as_f64()
                    .unwrap()
                    .abs()
                    > 1e-7
            );
        }
        if name == "he_gaussian" {
            let frozen = RHFMP2F12::new(&hf, &atoms, &obs)
                .with_options(MP2F12Options {
                    custom_cabs: Some(cabs.clone()),
                    frozen_core: 1,
                    ..Default::default()
                })
                .compute()
                .unwrap();
            close(frozen.f12_correlation_energy, 0.0);
            close(frozen.mp2_correlation_energy, 0.0);
            close(frozen.total_correlation_energy, result.cabs_singles_energy);
            let without_core = RHFMP2F12::new(&hf, &atoms, &obs)
                .with_options(MP2F12Options {
                    custom_cabs: Some(cabs),
                    frozen_core: 1,
                    cabs_core_singles: false,
                    ..Default::default()
                })
                .compute_checked()
                .unwrap();
            close(without_core.cabs_singles_energy, 0.0);
            close(without_core.total_energy, hf.total_energy);
        }
    }
}
#[test]
fn nonlocal_sp_ecp_matches_every_off_axis_pair_and_frozen_core_reference() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/f12_nonlocal_reference.json")).unwrap();
    let mut atoms: Vec<_> = data["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| Atom::new("He", std::array::from_fn(|axis| p[axis].as_f64().unwrap())))
        .collect();
    for atom in &mut atoms {
        atom.ecp = Some(std::sync::Arc::new(ecp::EcpDef {
            symbol: "He".into(),
            n_core: 0,
            l_max: 2,
            local: vec![ecp::EcpTerm {
                power: 2,
                exponent: 0.7,
                coefficient: 0.17,
            }],
            semilocal: vec![
                vec![ecp::EcpTerm {
                    power: 2,
                    exponent: 0.53,
                    coefficient: 0.11,
                }],
                vec![ecp::EcpTerm {
                    power: 2,
                    exponent: 0.61,
                    coefficient: -0.07,
                }],
            ],
        }));
    }
    let obs = build_basis(
        &atoms,
        &parse_gaussian94("He 0\nS 1 1\n1.4 1\nS 1 1\n0.45 1\n****"),
    );
    let aux = build_basis(
        &atoms,
        &parse_gaussian94("He 0\nS 1 1\n0.2 1\nP 1 1\n0.8 1\n****"),
    );
    let hf = run_scf(&atoms, &obs, 150, 1e-14).unwrap();
    close(hf.total_energy, data["e_scf"].as_f64().unwrap());
    let cabs = CABSBasis::from_reference(&hf, &obs, &aux, 1e-8).unwrap();
    for frozen_core in [0, 1] {
        let result = RHFMP2F12::new(&hf, &atoms, &obs)
            .with_options(MP2F12Options {
                approximation: F12Approximation::ApproximationA,
                custom_cabs: Some(cabs.clone()),
                custom_geminal: Some(GaussianGeminal::new(vec![(0.3, -0.4), (1.2, -0.2)]).unwrap()),
                frozen_core,
                ..Default::default()
            })
            .compute_checked()
            .unwrap();
        let expected: Vec<_> = data["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["pair"][0].as_u64().unwrap() as usize >= frozen_core)
            .collect();
        assert_eq!(result.pairs.len(), expected.len());
        for (actual, expected) in result.pairs.iter().zip(expected) {
            assert_eq!(
                actual.pair,
                std::array::from_fn(|k| expected["pair"][k].as_u64().unwrap() as usize)
            );
            for k in 0..2 {
                close(actual.b_bar[k], expected["b_bar"][k].as_f64().unwrap());
            }
        }
    }
    for approximation in [
        F12Approximation::ApproximationA,
        F12Approximation::ApproximationC,
    ] {
        let constant = RHFMP2F12::new(&hf, &atoms, &obs)
            .with_options(MP2F12Options {
                approximation,
                custom_cabs: Some(cabs.clone()),
                custom_geminal: Some(GaussianGeminal::new(vec![(0.0, 0.8)]).unwrap()),
                ..Default::default()
            })
            .compute_checked()
            .unwrap();
        for pair in constant.pairs {
            for value in pair
                .v
                .into_iter()
                .chain(pair.x)
                .chain(pair.b)
                .chain(pair.b_bar)
            {
                close(value, 0.0);
            }
        }
    }
}

#[test]
fn constant_geminal_projects_to_zero_for_every_occupied_pair() {
    let atoms = vec![
        Atom::new("He", [-0.1, 0.2, -1.3]),
        Atom::new("He", [0.3, -0.2, 1.4]),
    ];
    let obs = build_basis(
        &atoms,
        &parse_gaussian94("He 0\nS 1 1\n1.4 1\nS 1 1\n0.45 1\n****"),
    );
    let aux = build_basis(
        &atoms,
        &parse_gaussian94("He 0\nS 1 1\n0.2 1\nP 1 1\n0.8 1\n****"),
    );
    let hf = run_scf(&atoms, &obs, 150, 1e-14).unwrap();
    let cabs = CABSBasis::from_reference(&hf, &obs, &aux, 1e-8).unwrap();
    let result = RHFMP2F12::new(&hf, &atoms, &obs)
        .with_options(MP2F12Options {
            custom_cabs: Some(cabs),
            custom_geminal: Some(GaussianGeminal::new(vec![(0.0, 0.8)]).unwrap()),
            cabs_singles: false,
            ..Default::default()
        })
        .compute()
        .unwrap();
    for pair in result.pairs {
        for value in pair
            .v
            .into_iter()
            .chain(pair.x)
            .chain(pair.b)
            .chain(pair.c.data.iter().copied())
        {
            close(value, 0.0);
        }
        let (s, t) = pair.fixed_energy();
        close(s, 0.0);
        close(t, 0.0);
    }
}
#[test]
fn typed_mp2_f12_uses_context_and_reports_original_scf_reference() {
    use hf_rs::{
        context::PostHFContext,
        options::PostHFOptions,
        registry::{MethodId, create_solver},
    };
    let (atoms, obs, hf, _) = helium("he_gaussian");
    let context = PostHFContext::from_rhf(&hf, &atoms, &obs, None).unwrap();
    let mut options = PostHFOptions::default();
    options.f12.cabs_basis = Some("def2-svp-ri".into());
    let output = create_solver(MethodId::Mp2F12, &context, &options)
        .unwrap()
        .compute()
        .unwrap();
    close(output.e_total - output.e_scf, output.e_corr);
    close(
        output.e_corr,
        output.components["E_MP2"]
            + output.components["E_F12"]
            + output.components["E_CABS_singles"],
    );
    let parsed = PostHFOptions::from_map(
        &std::collections::BTreeMap::from([
            ("f12_approximation".into(), serde_json::json!("3*A(FIX)")),
            ("cabs_basis".into(), serde_json::json!("def2-svp-ri")),
        ]),
        &[hf_rs::options::F12_OPTIONS],
    )
    .unwrap();
    let a = create_solver(MethodId::Mp2F12, &context, &parsed)
        .unwrap()
        .compute()
        .unwrap();
    let legacy_a = RHFMP2F12::new(&hf, &atoms, &obs)
        .with_options(MP2F12Options {
            approximation: F12Approximation::ApproximationA,
            custom_cabs: Some(
                CABSBasis::from_builtin_reference(&hf, &obs, &atoms, "def2-svp-ri", 1e-8).unwrap(),
            ),
            ..Default::default()
        })
        .compute_checked()
        .unwrap();
    close(a.e_total, legacy_a.total_energy);
    options.frozen_virtual = 1;
    assert!(create_solver(MethodId::Mp2F12, &context, &options).is_err());
    options.frozen_virtual = 0;
    options.density_fitting = true;
    assert!(create_solver(MethodId::Mp2F12, &context, &options).is_err());
}
#[test]
fn workspace_budget_reaches_all_typed_f12_engines_and_preserves_energy() {
    use hf_rs::{
        context::PostHFContext,
        options::{F12_OPTIONS, PostHFOptions},
        registry::{MethodId, create_solver},
    };
    let (atoms, obs, hf, cabs) = helium("he_gaussian");
    let make = |budget| {
        RHFMP2F12::new(&hf, &atoms, &obs).with_options(MP2F12Options {
            custom_cabs: Some(cabs.clone()),
            workspace_budget_bytes: budget,
            ..Default::default()
        })
    };
    let default = make(MP2F12Options::default().workspace_budget_bytes)
        .compute_checked()
        .unwrap();
    close(
        default.total_energy,
        make(1024 * 1024).compute_checked().unwrap().total_energy,
    );
    for budget in [0, 1] {
        let error = make(budget).compute_checked().unwrap_err().to_string();
        assert!(error.contains("workspace"), "{error}");
    }
    let parse = |budget| {
        PostHFOptions::from_map(
            &std::collections::BTreeMap::from([
                ("f12_workspace_bytes".into(), serde_json::json!(budget)),
                ("cabs_basis".into(), serde_json::json!("def2-svp-ri")),
            ]),
            &[F12_OPTIONS],
        )
    };
    assert!(parse(0).is_err());
    assert!(parse(-1).is_err());
    let parsed = parse(1).unwrap();
    assert_eq!(parsed.f12.workspace_budget_bytes, 1);
    let context = PostHFContext::from_rhf(&hf, &atoms, &obs, None).unwrap();
    for method in [MethodId::Mp2F12, MethodId::CcsdF12a, MethodId::CcsdTF12b] {
        let error = create_solver(method, &context, &parsed)
            .unwrap()
            .compute()
            .unwrap_err()
            .to_string();
        assert!(error.contains("workspace"), "{method:?}: {error}");
    }
}

#[test]
fn typed_hybrid_and_singles_controls_match_explicit_engines() {
    use hf_rs::{
        context::PostHFContext,
        options::{F12_OPTIONS, PostHFOptions},
        registry::{MethodId, create_solver},
    };
    let (atoms, obs, hf, _) = helium("he_gaussian");
    let context = PostHFContext::from_rhf(&hf, &atoms, &obs, None).unwrap();
    for (name, hybrid) in [("HY1", F12Hybrid::Hy1), ("HY2", F12Hybrid::Hy2)] {
        let mut map = std::collections::BTreeMap::from([
            ("f12_hybrid".into(), serde_json::json!(name)),
            ("cabs_basis".into(), serde_json::json!("def2-svp-ri")),
            ("cabs_singles".into(), serde_json::json!(false)),
            ("cabs_core_singles".into(), serde_json::json!(false)),
        ]);
        let parsed = PostHFOptions::from_map(&map, &[F12_OPTIONS]).unwrap();
        let output = create_solver(MethodId::Mp2F12, &context, &parsed)
            .unwrap()
            .compute()
            .unwrap();
        let explicit = RHFMP2F12::new(&hf, &atoms, &obs)
            .with_options(MP2F12Options {
                hybrid,
                cabs_singles: false,
                cabs_core_singles: false,
                custom_cabs: Some(
                    CABSBasis::from_builtin_reference(&hf, &obs, &atoms, "def2-svp-ri", 1e-8)
                        .unwrap(),
                ),
                ..Default::default()
            })
            .compute_checked()
            .unwrap();
        close(output.e_total, explicit.total_energy);
        close(output.components["E_CABS_singles"], 0.0);
        assert_eq!(output.method, MethodId::Mp2F12.name());
        assert!(explicit.method_name().contains(name));
        map.insert("f12_approximation".into(), serde_json::json!("3*A(FIX)"));
        let unsupported = PostHFOptions::from_map(&map, &[F12_OPTIONS]).unwrap();
        assert!(create_solver(MethodId::Mp2F12, &context, &unsupported).is_err());
        map.insert("f12_hybrid".into(), serde_json::json!("HY3"));
        assert!(PostHFOptions::from_map(&map, &[F12_OPTIONS]).is_err());
    }
}

#[test]
fn invalid_geminal_scales_and_unimplemented_approximations_return_errors() {
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-300, 1e300] {
        assert!(GeminalCorrelationFactor::new(scale).is_err());
    }
    let (atoms, obs, hf, _) = helium("he_gaussian");
    {
        let approximation = F12Approximation::ApproximationB;
        assert!(
            RHFMP2F12::new(&hf, &atoms, &obs)
                .with_approximation(approximation)
                .compute()
                .is_err()
        );
    }
}

#[test]
fn malformed_public_cabs_and_hf_inputs_return_errors() {
    use hf_rs::post_hf::f12::cabs::compute_cabs_singles_full;
    let (atoms, obs, mut hf, mut cabs) = helium("he_gaussian");
    cabs.c_obs_to_cabs.data.pop();
    assert!(compute_cabs_singles_full(&hf, &atoms, &obs, &cabs).is_err());
    let (_, _, _, cabs) = helium("he_gaussian");
    hf.n_occ = hf.mo_coefficients.cols + 1;
    assert!(compute_cabs_singles_full(&hf, &atoms, &obs, &cabs).is_err());
    assert!(
        RHFMP2F12::new(&hf, &atoms, &obs)
            .with_options(MP2F12Options {
                custom_cabs: Some(cabs.clone()),
                ..Default::default()
            })
            .compute_checked()
            .is_err()
    );
    hf.n_occ = 1;
    hf.density_matrix.data.pop();
    assert!(compute_cabs_singles_full(&hf, &atoms, &obs, &cabs).is_err());
}

#[test]
fn slater_expansion_matches_independent_psi4_values_at_every_scale() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/f12_slater_fit_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let factor = GeminalCorrelationFactor::new(case["beta"].as_f64().unwrap()).unwrap();
        for (index, term) in case["normalized_terms"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let exponent = term[0].as_f64().unwrap();
            let coefficient = term[1].as_f64().unwrap();
            assert!((factor.exponents[index] - exponent).abs() < 2e-15 * exponent.abs().max(1.0));
            assert!((factor.coefficients[index] - coefficient).abs() < 2e-15);
        }
    }
}
