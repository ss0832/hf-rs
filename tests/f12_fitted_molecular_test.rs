// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    Matrix,
    ao::BasisKind,
    basis::{Atom, ContractedGaussian, build_basis, parse_gaussian94},
    context::AuxBasis,
    ecp::{EcpDef, EcpTerm},
    post_hf::{
        PostHFResult,
        cc::CCSDOptions,
        f12::{
            CABSBasis, CCSDF12Variant, F12Approximation, F12Hybrid, MP2F12Options, RHFCCSDF12,
            RHFMP2F12, molecular::molecular_ccsd_projected_coulomb_fitted_in_space,
            molecular::molecular_ccsd_projected_coulomb_fitted_with_options,
            operators::GaussianGeminal,
        },
    },
    scf::{HFResult, RhfConvergence, run_scf_in_space_with_convergence},
};
use serde_json::Value;

fn data() -> Value {
    serde_json::from_str(include_str!("data/f12_fitted_molecular_reference.json")).unwrap()
}
fn datasets() -> Vec<Value> {
    let mixed: Value =
        serde_json::from_str(include_str!("data/f12_mixed_fock_reference.json")).unwrap();
    let exact_obs: Value =
        serde_json::from_str(include_str!("data/f12_exact_obs_coulomb_reference.json")).unwrap();
    std::iter::once(data())
        .chain(mixed["models"].as_array().unwrap().iter().cloned())
        .chain(exact_obs["models"].as_array().unwrap().iter().cloned())
        .collect()
}
fn matrix(v: &Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
            .collect(),
    )
}
fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() < 1e-10,
        "{a} vs {b}: error {:e}",
        (a - b).abs()
    );
}
fn setup(
    data: &Value,
    nonlocal: bool,
) -> (Vec<Atom>, Vec<ContractedGaussian>, HFResult, MP2F12Options) {
    let mut atoms: Vec<_> = data["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| Atom::new("He", std::array::from_fn(|axis| p[axis].as_f64().unwrap())))
        .collect();
    if nonlocal {
        for atom in &mut atoms {
            let term = |exponent, coefficient| EcpTerm {
                power: 2,
                exponent,
                coefficient,
            };
            atom.ecp = Some(std::sync::Arc::new(EcpDef {
                symbol: "He".into(),
                n_core: 0,
                l_max: 2,
                local: vec![term(0.7, 0.17)],
                semilocal: vec![vec![term(0.53, 0.11)], vec![term(0.61, -0.07)]],
            }));
        }
    }
    let basis = |key: &str| build_basis(&atoms, &parse_gaussian94(data[key].as_str().unwrap()));
    let obs = basis("obs_g94");
    let aux = basis("aux_g94");
    let fitting = AuxBasis {
        name: None,
        functions: basis("fit_g94"),
    };
    let fock_fitting = data["mixed_fock"]
        .as_bool()
        .unwrap_or(false)
        .then(|| AuxBasis {
            name: None,
            functions: basis("fock_fit_g94"),
        });
    let hf = run_scf_in_space_with_convergence(
        &atoms,
        &obs,
        0,
        200,
        RhfConvergence {
            energy: 1e-13,
            residual: 1e-12,
            density: 1e-12,
        },
        hf_rs::diis::ConvAlgo::Adiissoscf,
        None,
        BasisKind::Cartesian,
    )
    .unwrap();
    let cabs = CABSBasis::from_reference(&hf, &obs, &aux, 1e-8).unwrap();
    let factor = GaussianGeminal::new(
        data["terms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (t[0].as_f64().unwrap(), t[1].as_f64().unwrap()))
            .collect(),
    )
    .unwrap();
    (
        atoms,
        obs,
        hf,
        MP2F12Options {
            custom_cabs: Some(cabs),
            fitting_basis: (!data["exact_correction"].as_bool().unwrap_or(false))
                .then_some(fitting),
            fitting_basis_kind: if data["spherical_fitting"].as_bool().unwrap_or(false) {
                BasisKind::Spherical
            } else {
                BasisKind::Cartesian
            },
            fock_fitting_basis: fock_fitting,
            fock_fitting_basis_kind: if data["spherical_fock_fitting"].as_bool().unwrap_or(false) {
                BasisKind::Spherical
            } else {
                BasisKind::Cartesian
            },
            fit_orbital_mp2: data["fit_orbital_mp2"].as_bool().unwrap_or(false),
            exact_obs_coulomb: data["exact_obs_coulomb"].as_bool().unwrap_or(false),
            custom_geminal: Some(factor),
            ..Default::default()
        },
    )
}
fn phases(hf: &HFResult, case: &Value) -> Vec<f64> {
    let alignment = hf
        .mo_coefficients
        .transpose()
        .dot(&hf.overlap_matrix)
        .dot(&matrix(&case["mo_coefficients"]));
    (0..alignment.rows)
        .map(|i| {
            assert!(alignment[(i, i)].abs() > 1. - 1e-9);
            alignment[(i, i)].signum()
        })
        .collect()
}

#[test]
fn every_fitted_pair_projected_coulomb_and_hybrid_matches_independent_dense_oracle() {
    let mut components = 0;
    let mut maximum = 0f64;
    for data in datasets() {
        for nonlocal in [false, true] {
            let (atoms, obs, hf, base) = setup(&data, nonlocal);
            for case in data["cases"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| c["nonlocal_ecp"].as_bool().unwrap() == nonlocal)
            {
                let fc = case["frozen_core"].as_u64().unwrap() as usize;
                let phase = phases(&hf, case);
                let options = MP2F12Options {
                    frozen_core: fc,
                    ..base.clone()
                };
                let result = RHFMP2F12::new(&hf, &atoms, &obs)
                    .with_options(options.clone())
                    .compute_checked()
                    .unwrap();
                assert_eq!(result.fitted_correction, base.fitting_basis.is_some());
                assert_eq!(result.fitted_fock, base.fock_fitting_basis.is_some());
                assert_eq!(result.fitted_orbital_mp2, base.fit_orbital_mp2);
                assert_eq!(result.exact_obs_coulomb, base.exact_obs_coulomb);
                close(
                    result.to_output().unwrap().components["F12_exact_obs_coulomb"],
                    u8::from(base.exact_obs_coulomb) as f64,
                );
                assert_eq!(
                    result.to_output().unwrap().components["F12_fitted_correction"],
                    u8::from(base.fitting_basis.is_some()) as f64
                );
                for (actual, key) in [
                    (result.hf_energy, "e_hf"),
                    (result.mp2_correlation_energy, "e_mp2"),
                    (result.cabs_singles_energy, "e_cabs"),
                ] {
                    close(actual, case[key].as_f64().unwrap());
                }
                close(
                    result.mp2_correlation_energy + result.f12_correlation_energy,
                    case["e_mp2_f12"].as_f64().unwrap(),
                );
                assert_eq!(result.pairs.len(), case["pairs"].as_array().unwrap().len());
                for (pair, expected) in result.pairs.iter().zip(case["pairs"].as_array().unwrap()) {
                    assert_eq!(
                        pair.pair,
                        std::array::from_fn(|k| expected["pair"][k].as_u64().unwrap() as usize)
                    );
                    for (actual, key) in [
                        (pair.v, "v"),
                        (pair.x, "x"),
                        (pair.b, "b"),
                        (pair.v_bar, "v_bar"),
                        (pair.b_bar, "b_bar"),
                    ] {
                        for k in 0..2 {
                            let target = expected[key][k].as_f64().unwrap();
                            maximum = maximum.max((actual[k] - target).abs());
                            close(actual[k], target);
                            components += 1;
                        }
                    }
                    for (actual, key) in [(&pair.c, "c"), (&pair.g, "g")] {
                        let expected = matrix(&expected[key]);
                        for a in 0..actual.rows {
                            for b in 0..actual.cols {
                                let value = actual[(a, b)]
                                    * phase[hf.n_occ + a]
                                    * phase[hf.n_occ + b]
                                    * phase[pair.pair[0]]
                                    * phase[pair.pair[1]];
                                maximum = maximum.max((value - expected[(a, b)]).abs());
                                close(value, expected[(a, b)]);
                                components += 1;
                            }
                        }
                    }
                }
                let projected = if base.exact_obs_coulomb {
                    molecular_ccsd_projected_coulomb_fitted_with_options(
                        &hf,
                        &obs,
                        base.custom_cabs.as_ref().unwrap(),
                        base.custom_geminal.as_ref().unwrap(),
                        &options,
                    )
                    .unwrap()
                } else if let Some(fitting) = &base.fitting_basis {
                    molecular_ccsd_projected_coulomb_fitted_in_space(
                        &hf,
                        &obs,
                        base.custom_cabs.as_ref().unwrap(),
                        base.custom_geminal.as_ref().unwrap(),
                        fitting,
                        base.fitting_basis_kind,
                        fc,
                        base.workspace_budget_bytes,
                    )
                    .unwrap()
                } else {
                    hf_rs::post_hf::f12::molecular::molecular_ccsd_projected_coulomb(
                        &hf,
                        &obs,
                        base.custom_cabs.as_ref().unwrap(),
                        base.custom_geminal.as_ref().unwrap(),
                    )
                    .unwrap()
                    .into_iter()
                    .enumerate()
                    .filter(|(ij, _)| ij / hf.n_occ >= fc && ij % hf.n_occ >= fc)
                    .map(|(_, matrix)| matrix)
                    .collect()
                };
                let nc = hf.n_occ - fc;
                for (ij, actual) in projected.iter().enumerate() {
                    let expected = matrix(&case["projected_coulomb"][ij]);
                    for p in 0..actual.rows {
                        for q in 0..actual.cols {
                            let value = actual[(p, q)]
                                * phase[p]
                                * phase[q]
                                * phase[fc + ij / nc]
                                * phase[fc + ij % nc];
                            maximum = maximum.max((value - expected[(p, q)]).abs());
                            close(value, expected[(p, q)]);
                            components += 1;
                        }
                    }
                }
                for (hybrid, key) in [(F12Hybrid::Hy1, "hy1"), (F12Hybrid::Hy2, "hy2")] {
                    let value = RHFMP2F12::new(&hf, &atoms, &obs)
                        .with_options(MP2F12Options {
                            hybrid,
                            ..options.clone()
                        })
                        .compute_checked()
                        .unwrap();
                    let expected = &case["hybrids"][key];
                    close(
                        value.f12_correlation_energy - result.f12_correlation_energy,
                        expected["energy_shift"].as_f64().unwrap(),
                    );
                    for (ij, (actual, original)) in
                        value.pairs.iter().zip(&result.pairs).enumerate()
                    {
                        for k in 0..2 {
                            let shift = expected["b_shift"][ij][k].as_f64().unwrap();
                            maximum = maximum.max((actual.b[k] - original.b[k] - shift).abs());
                            maximum =
                                maximum.max((actual.b_bar[k] - original.b_bar[k] - shift).abs());
                            close(actual.b[k] - original.b[k], shift);
                            close(actual.b_bar[k] - original.b_bar[k], shift);
                            components += 2;
                        }
                    }
                }
                let error = RHFMP2F12::new(&hf, &atoms, &obs)
                    .with_options(MP2F12Options {
                        approximation: F12Approximation::ApproximationA,
                        ..options.clone()
                    })
                    .compute_checked()
                    .unwrap_err();
                assert!(error.to_string().contains("only for 3C"));
                assert!(
                    RHFMP2F12::new(&hf, &atoms, &obs)
                        .with_options(MP2F12Options {
                            workspace_budget_bytes: 1,
                            ..options
                        })
                        .compute_checked()
                        .is_err()
                );
            }
        }
    }
    println!(
        "Fitted molecular pairs/projected/hybrid components: {components}; maximum error {maximum:e}"
    );
}

#[test]
fn fitted_coupled_ccsd_f12a_b_all_amplitudes_and_energies_match_public_pyscf_iteration() {
    let mut components = 0;
    let mut maximum = 0f64;
    for data in datasets() {
        for nonlocal in [false, true] {
            let (atoms, obs, hf, base) = setup(&data, nonlocal);
            for case in data["cases"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| c["nonlocal_ecp"].as_bool().unwrap() == nonlocal)
            {
                let fc = case["frozen_core"].as_u64().unwrap() as usize;
                let phase = phases(&hf, case);
                for variant in [CCSDF12Variant::F12a, CCSDF12Variant::F12b] {
                    for (hybrid, key) in [
                        (F12Hybrid::Full, "full"),
                        (F12Hybrid::Hy1, "hy1"),
                        (F12Hybrid::Hy2, "hy2"),
                    ] {
                        let result = RHFCCSDF12::new(&hf, &atoms, &obs, variant)
                            .with_f12_options(MP2F12Options {
                                frozen_core: fc,
                                hybrid,
                                ..base.clone()
                            })
                            .with_ccsd_options(CCSDOptions {
                                max_iterations: 300,
                                energy_tol: 1e-13,
                                residual_tol: 1e-12,
                                ..Default::default()
                            })
                            .with_triples(true)
                            .compute_checked()
                            .unwrap();
                        assert!(result.ccsd_result.converged);
                        assert_eq!(result.fitted_correction, base.fitting_basis.is_some());
                        assert_eq!(result.exact_obs_coulomb, base.exact_obs_coulomb);
                        let output = result.to_output().unwrap();
                        close(
                            output.components["F12_exact_obs_coulomb"],
                            u8::from(base.exact_obs_coulomb) as f64,
                        );
                        close(
                            output.components["E_F12_C_tau"],
                            case["e_coupling_a"].as_f64().unwrap(),
                        );
                        close(
                            output.components["E_F12_additional_b"],
                            case["e_additional_b"].as_f64().unwrap(),
                        );
                        let shift = if hybrid == F12Hybrid::Full {
                            0.
                        } else {
                            case["hybrids"][key]["energy_shift"].as_f64().unwrap()
                        };
                        let t_shift = case["e_triples"].as_f64().unwrap() * shift
                            / case["e_mp2"].as_f64().unwrap();
                        let total = if variant == CCSDF12Variant::F12a {
                            "e_total_a"
                        } else {
                            "e_total_b"
                        };
                        for (actual, expected) in [
                            (
                                result.total_energy,
                                case[total].as_f64().unwrap() + shift + t_shift,
                            ),
                            (
                                result.ccsd_correlation_energy,
                                case["e_cc"].as_f64().unwrap(),
                            ),
                            (
                                result.f12_correction_energy,
                                case["e_fixed"].as_f64().unwrap() + shift,
                            ),
                            (
                                result.coupling_components.fock_amplitude,
                                case["e_coupling_a"].as_f64().unwrap(),
                            ),
                            (
                                result.coupling_components.additional_f12b,
                                case["e_additional_b"].as_f64().unwrap(),
                            ),
                            (
                                result.triples_energy.unwrap(),
                                case["e_triples"].as_f64().unwrap(),
                            ),
                            (
                                result.scaled_triples_energy.unwrap(),
                                case["e_scaled_triples"].as_f64().unwrap() + t_shift,
                            ),
                        ] {
                            maximum = maximum.max((actual - expected).abs());
                            close(actual, expected);
                            components += 1;
                        }
                        let cc = &result.ccsd_result;
                        for i in 0..cc.n_occ {
                            for a in 0..cc.n_virt {
                                let p = i * cc.n_virt + a;
                                let actual =
                                    cc.t1_amplitudes[p] * phase[fc + i] * phase[hf.n_occ + a];
                                let expected = case["t1"][p].as_f64().unwrap();
                                maximum = maximum.max((actual - expected).abs());
                                close(actual, expected);
                                components += 1;
                                for j in 0..cc.n_occ {
                                    for b in 0..cc.n_virt {
                                        let index =
                                            ((i * cc.n_occ + j) * cc.n_virt + a) * cc.n_virt + b;
                                        let actual = cc.t2_amplitudes[index]
                                            * phase[fc + i]
                                            * phase[fc + j]
                                            * phase[hf.n_occ + a]
                                            * phase[hf.n_occ + b];
                                        let expected = case["t2"][index].as_f64().unwrap();
                                        maximum = maximum.max((actual - expected).abs());
                                        close(actual, expected);
                                        components += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    println!("Fitted CCSD-F12a/b components: {components}; maximum error {maximum:e}");
}

#[test]
fn typed_fitting_basis_controls_reach_mp2_and_both_ccsd_variants() {
    use hf_rs::{
        context::PostHFContext,
        options::{F12_OPTIONS, PostHFOptions},
        registry::{ContextMethod, MethodId, create_solver},
    };
    let data = data();
    let (atoms, obs, hf, _) = setup(&data, false);
    let context = PostHFContext::from_rhf(&hf, &atoms, &obs, None).unwrap();
    let parsed = PostHFOptions::from_map(
        &std::collections::BTreeMap::from([
            ("f12_fitting_basis".into(), serde_json::json!("def2-svp-ri")),
            ("cabs_basis".into(), serde_json::json!("def2-svp-ri")),
        ]),
        &[F12_OPTIONS],
    )
    .unwrap();
    let mapped = RHFMP2F12::from_context(&context, &parsed).unwrap();
    assert_eq!(
        mapped
            .options
            .fitting_basis
            .as_ref()
            .unwrap()
            .name
            .as_deref(),
        Some("def2-svp-ri")
    );
    let mp2 = mapped.compute_checked().unwrap();
    for id in [
        MethodId::Mp2F12,
        MethodId::CcsdF12a,
        MethodId::CcsdF12b,
        MethodId::CcsdTF12a,
        MethodId::CcsdTF12b,
    ] {
        assert_eq!(
            id.gradient_kind(hf_rs::context::RefKind::Rhf, &parsed),
            hf_rs::registry::GradientKind::Unsupported
        );
    }
    for id in [MethodId::Mp2F12, MethodId::CcsdF12a, MethodId::CcsdF12b] {
        let actual = create_solver(id, &context, &parsed)
            .unwrap()
            .compute()
            .unwrap();
        assert_eq!(actual.components["F12_fitted_correction"], 1.);
        if id == MethodId::Mp2F12 {
            close(actual.e_total, mp2.total_energy);
        }
        let mut invalid = parsed.clone();
        invalid.f12.workspace_budget_bytes = 1;
        assert!(
            create_solver(id, &context, &invalid)
                .unwrap()
                .compute()
                .is_err()
        );
    }
    let mut invalid = parsed.clone();
    invalid.f12.approximation = F12Approximation::ApproximationA;
    assert!(RHFMP2F12::from_context(&context, &invalid).is_err());
    invalid = parsed.clone();
    invalid.f12.fitting_basis = Some("unknown-fitting-basis".into());
    assert!(RHFMP2F12::from_context(&context, &invalid).is_err());
    invalid = parsed;
    invalid.density_fitting = true;
    assert!(RHFMP2F12::from_context(&context, &invalid).is_err());
    let error = mapped.numerical_gradient(&Default::default()).unwrap_err();
    assert!(error.to_string().contains("fitting-space response"));
}

#[test]
fn separate_fock_and_physical_fit_controls_reach_typed_solvers_and_reject_invalid_references() {
    use hf_rs::{
        context::{PostHFContext, RefKind},
        options::{F12_OPTIONS, PostHFOptions},
        registry::{ContextMethod, GradientKind, MethodId, create_solver},
    };
    let data = data();
    let (atoms, obs, hf, _) = setup(&data, false);
    let context = PostHFContext::from_rhf(&hf, &atoms, &obs, None).unwrap();
    let mut controls = std::collections::BTreeMap::from([
        ("f12_fitting_basis".into(), serde_json::json!("def2-svp-ri")),
        (
            "f12_fock_fitting_basis".into(),
            serde_json::json!("def2-svp-ri"),
        ),
        ("cabs_basis".into(), serde_json::json!("def2-svp-ri")),
        (
            "f12_fitting_basis_kind".into(),
            serde_json::json!("spherical"),
        ),
        (
            "f12_fock_fitting_basis_kind".into(),
            serde_json::json!("cartesian"),
        ),
        ("f12_fit_orbital_mp2".into(), serde_json::json!(true)),
        ("f12_exact_obs_coulomb".into(), serde_json::json!(true)),
    ]);
    let options = PostHFOptions::from_map(&controls, &[F12_OPTIONS]).unwrap();
    let mapped = RHFMP2F12::from_context(&context, &options).unwrap();
    assert_eq!(mapped.options.fitting_basis_kind, BasisKind::Spherical);
    assert_eq!(mapped.options.fock_fitting_basis_kind, BasisKind::Cartesian);
    assert!(mapped.options.fit_orbital_mp2);
    assert!(mapped.options.exact_obs_coulomb);
    assert_eq!(
        mapped
            .options
            .fock_fitting_basis
            .as_ref()
            .unwrap()
            .name
            .as_deref(),
        Some("def2-svp-ri")
    );
    for id in [
        MethodId::Mp2F12,
        MethodId::CcsdF12a,
        MethodId::CcsdF12b,
        MethodId::CcsdTF12b,
    ] {
        let output = create_solver(id, &context, &options)
            .unwrap()
            .compute()
            .unwrap();
        for key in [
            "F12_fitted_fock",
            "F12_fitted_orbital_mp2",
            "F12_exact_obs_coulomb",
            "F12_spherical_fit",
        ] {
            assert_eq!(output.components[key], 1.);
        }
        assert_eq!(output.components["F12_spherical_fock_fit"], 0.);
    }
    let mut fock_only = options.clone();
    fock_only.f12.fitting_basis = None;
    fock_only.f12.fit_orbital_mp2 = false;
    fock_only.f12.exact_obs_coulomb = false;
    for id in [
        MethodId::Mp2F12,
        MethodId::CcsdF12a,
        MethodId::CcsdF12b,
        MethodId::CcsdTF12a,
        MethodId::CcsdTF12b,
    ] {
        assert_eq!(
            id.gradient_kind(RefKind::Rhf, &fock_only),
            GradientKind::Unsupported
        );
    }
    assert!(
        RHFMP2F12::from_context(&context, &fock_only)
            .unwrap()
            .numerical_gradient(&Default::default())
            .is_err()
    );
    let mut invalid_reference = hf.clone();
    invalid_reference.orbital_energies[0] += 0.01;
    let engine =
        RHFMP2F12::new(&invalid_reference, &atoms, &obs).with_options(mapped.options.clone());
    assert!(
        engine
            .compute_checked()
            .unwrap_err()
            .to_string()
            .contains("independently canonical")
    );
    let mut invalid = options.clone();
    invalid.f12.fitting_basis = None;
    assert!(RHFMP2F12::from_context(&context, &invalid).is_err());
    invalid.f12.fit_orbital_mp2 = false;
    assert!(
        RHFMP2F12::from_context(&context, &invalid)
            .err()
            .unwrap()
            .to_string()
            .contains("exact OBS Coulomb mixing")
    );
    let mut direct = mapped.options.clone();
    direct.fitting_basis = None;
    direct.fit_orbital_mp2 = false;
    assert!(
        RHFMP2F12::new(&hf, &atoms, &obs)
            .with_options(direct)
            .compute_checked()
            .unwrap_err()
            .to_string()
            .contains("exact OBS Coulomb mixing")
    );
    invalid = options.clone();
    invalid.f12.fock_fitting_basis = Some("unknown-fitting-basis".into());
    assert!(RHFMP2F12::from_context(&context, &invalid).is_err());
    for key in ["f12_fitting_basis_kind", "f12_fock_fitting_basis_kind"] {
        let old = controls.insert(key.into(), serde_json::json!("invalid"));
        assert!(PostHFOptions::from_map(&controls, &[F12_OPTIONS]).is_err());
        controls.insert(key.into(), old.unwrap());
    }
    controls.insert("f12_exact_obs_coulomb".into(), serde_json::json!("invalid"));
    assert!(PostHFOptions::from_map(&controls, &[F12_OPTIONS]).is_err());
}
