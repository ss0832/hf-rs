// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    Matrix,
    ao::BasisKind,
    basis::{Atom, ContractedGaussian, build_basis, parse_gaussian94},
    ecp::{EcpDef, EcpTerm},
    post_hf::{
        PostHFMethod, PostHFResult,
        cc::{CCSD, CCSDOptions},
        f12::{
            CABSBasis, CCSDF12Variant, F12Approximation, MP2F12Options, RHFCCSDF12, RHFMP2F12,
            operators::GaussianGeminal,
        },
    },
    scf::{HFResult, RhfConvergence, run_scf_in_space_with_convergence},
};

fn vector(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter().flat_map(vector).collect(),
    )
}
fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "{actual:.15} vs {expected:.15}: {:e}",
        actual - expected
    );
}
fn setup(
    data: &serde_json::Value,
    nonlocal: bool,
) -> (Vec<Atom>, Vec<ContractedGaussian>, HFResult, MP2F12Options) {
    let mut atoms: Vec<_> = data["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| Atom::new("He", std::array::from_fn(|a| p[a].as_f64().unwrap())))
        .collect();
    if nonlocal {
        for atom in &mut atoms {
            atom.ecp = Some(std::sync::Arc::new(EcpDef {
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
    }
    let obs = build_basis(&atoms, &parse_gaussian94(data["obs_g94"].as_str().unwrap()));
    let aux = build_basis(&atoms, &parse_gaussian94(data["aux_g94"].as_str().unwrap()));
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
    let options = MP2F12Options {
        custom_cabs: Some(cabs),
        custom_geminal: Some(
            GaussianGeminal::new(
                data["terms"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|t| (t[0].as_f64().unwrap(), t[1].as_f64().unwrap()))
                    .collect(),
            )
            .unwrap(),
        ),
        ..Default::default()
    };
    (atoms, obs, hf, options)
}
fn controls() -> CCSDOptions {
    CCSDOptions {
        max_iterations: 300,
        energy_tol: 1e-13,
        residual_tol: 1e-11,
        diis_subspace: 8,
    }
}

#[test]
fn coupled_ccsd_f12_every_amplitude_and_energy_match_independent_iteration_with_ecp_and_frozen_core()
 {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/ccsd_f12_reference.json")).unwrap();
    let mut components = 0;
    for nonlocal in [false, true] {
        let (atoms, obs, hf, base) = setup(&data, nonlocal);
        let ordinary = CCSD::new(&hf).with_options(controls()).compute().unwrap();
        for case in data["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["nonlocal_ecp"].as_bool().unwrap() == nonlocal)
        {
            let fc = case["frozen_core"].as_u64().unwrap() as usize;
            let options = MP2F12Options {
                frozen_core: fc,
                ..base.clone()
            };
            let mp2 = RHFMP2F12::new(&hf, &atoms, &obs)
                .with_options(options.clone())
                .compute()
                .unwrap();
            close(mp2.mp2_correlation_energy, case["e_mp2"].as_f64().unwrap());
            close(
                mp2.mp2_correlation_energy + mp2.f12_correlation_energy,
                case["e_mp2_f12"].as_f64().unwrap(),
            );
            let alignment = hf
                .mo_coefficients
                .transpose()
                .dot(&hf.overlap_matrix)
                .dot(&matrix(&case["mo_coefficients"]));
            let phase: Vec<_> = (0..alignment.rows)
                .map(|i| {
                    assert!(alignment[(i, i)].abs() > 1. - 1e-9);
                    alignment[(i, i)].signum()
                })
                .collect();
            for variant in [CCSDF12Variant::F12a, CCSDF12Variant::F12b] {
                let result = RHFCCSDF12::new(&hf, &atoms, &obs, variant)
                    .with_f12_options(options.clone())
                    .with_ccsd_options(controls())
                    .with_triples(true)
                    .compute_checked()
                    .unwrap();
                let cc = &result.ccsd_result;
                assert!(result.converged());
                assert_eq!(cc.n_occ, hf.n_occ - fc);
                close(result.reference_energy(), hf.total_energy);
                close(
                    result.total_energy() - result.reference_energy(),
                    result.correlation_energy(),
                );
                for (actual, key) in [
                    (result.hf_energy, "e_hf"),
                    (result.cabs_singles_energy, "e_cabs"),
                    (result.ccsd_correlation_energy, "e_cc"),
                    (result.f12_correction_energy, "e_fixed"),
                    (result.triples_energy.unwrap(), "e_triples"),
                    (result.scaled_triples_energy.unwrap(), "e_scaled_triples"),
                ] {
                    close(actual, case[key].as_f64().unwrap());
                }
                let mut coupling = case["e_coupling_a"].as_f64().unwrap();
                let total = match variant {
                    CCSDF12Variant::F12a => case["e_total_a"].as_f64().unwrap(),
                    CCSDF12Variant::F12b => {
                        coupling += case["e_additional_b"].as_f64().unwrap();
                        case["e_total_b"].as_f64().unwrap()
                    }
                };
                close(result.coupling_energy, coupling);
                close(result.total_energy, total);
                let (no, nv) = (cc.n_occ, cc.n_virt);
                let expected1 = vector(&case["t1"]);
                let expected2 = vector(&case["t2"]);
                let mut maximum = 0f64;
                for i in 0..no {
                    for a in 0..nv {
                        let index = i * nv + a;
                        let value = cc.t1_amplitudes[index] * phase[fc + i] * phase[hf.n_occ + a];
                        maximum = maximum.max((value - expected1[index]).abs());
                        assert!((value - expected1[index]).abs() < 1e-9);
                        components += 1;
                        for j in 0..no {
                            for b in 0..nv {
                                let index = ((i * no + j) * nv + a) * nv + b;
                                let value = cc.t2_amplitudes[index]
                                    * phase[fc + i]
                                    * phase[fc + j]
                                    * phase[hf.n_occ + a]
                                    * phase[hf.n_occ + b];
                                maximum = maximum.max((value - expected2[index]).abs());
                                assert!(
                                    (value - expected2[index]).abs() < 1e-9,
                                    "T2 {i},{j},{a},{b}: {value} vs {}",
                                    expected2[index]
                                );
                                components += 1;
                            }
                        }
                    }
                }
                if fc == 0 {
                    assert!((cc.correlation_energy - ordinary.correlation_energy).abs() > 1e-5);
                    assert!(result.triples_energy.unwrap().abs() > 1e-9);
                }
                println!(
                    "CCSD-F12 {variant:?} nonlocal={nonlocal} frozen={fc}: total={:.15} amplitude_max_error={maximum:e}",
                    result.total_energy
                );
            }
        }
    }
    println!("All {components} amplitude components checked");
}

#[test]
fn zero_and_constant_geminals_recover_ccsd_and_invalid_or_unconverged_inputs_fail() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/ccsd_f12_reference.json")).unwrap();
    let (atoms, obs, mut hf, base) = setup(&data, true);
    let ordinary = CCSD::new(&hf).with_options(controls()).compute().unwrap();
    for coefficient in [0., 0.8] {
        let options = MP2F12Options {
            custom_geminal: Some(GaussianGeminal::new(vec![(0., coefficient)]).unwrap()),
            cabs_singles: false,
            ..base.clone()
        };
        for variant in [CCSDF12Variant::F12a, CCSDF12Variant::F12b] {
            let result = RHFCCSDF12::new(&hf, &atoms, &obs, variant)
                .with_f12_options(options.clone())
                .with_ccsd_options(controls())
                .compute_checked()
                .unwrap();
            close(result.correlation_energy(), ordinary.correlation_energy);
            for (a, b) in result
                .ccsd_result
                .t1_amplitudes
                .iter()
                .chain(&result.ccsd_result.t2_amplitudes)
                .zip(ordinary.t1_amplitudes.iter().chain(&ordinary.t2_amplitudes))
            {
                close(*a, *b);
            }
        }
    }
    let engine =
        RHFCCSDF12::new(&hf, &atoms, &obs, CCSDF12Variant::F12a).with_f12_options(base.clone());
    assert!(
        engine
            .clone()
            .with_ccsd_options(CCSDOptions {
                max_iterations: 1,
                ..controls()
            })
            .compute_checked()
            .is_err()
    );
    assert!(
        engine
            .clone()
            .with_ccsd_options(CCSDOptions {
                energy_tol: f64::NAN,
                ..controls()
            })
            .numerical_gradient(&hf_rs::post_hf::numerical::NumericalGradientOptions::default())
            .is_err()
    );
    assert!(
        engine
            .clone()
            .with_f12_options(MP2F12Options {
                approximation: F12Approximation::ApproximationA,
                ..base.clone()
            })
            .compute_checked()
            .is_err()
    );
    assert!(
        engine
            .with_f12_options(MP2F12Options {
                frozen_core: hf.n_occ,
                ..base.clone()
            })
            .compute_checked()
            .is_err()
    );
    hf.converged = false;
    assert!(
        RHFCCSDF12::new(&hf, &atoms, &obs, CCSDF12Variant::F12a)
            .with_f12_options(base)
            .compute_checked()
            .is_err()
    );
}

#[test]
fn typed_ccsd_f12_variants_preserve_components_reference_and_checked_options() {
    use hf_rs::post_hf::f12::typed_ccsd::CcsdF12Method;
    use hf_rs::{
        context::PostHFContext,
        options::PostHFOptions,
        registry::{ContextMethod, MethodId, create_solver},
    };
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/ccsd_f12_reference.json")).unwrap();
    let (atoms, obs, hf, _) = setup(&data, false);
    let context = PostHFContext::from_rhf(&hf, &atoms, &obs, None).unwrap();
    let mut options = PostHFOptions {
        max_iter: 300,
        energy_tol: 1e-13,
        residual_tol: 1e-11,
        frozen_core: 1,
        ..Default::default()
    };
    options.f12.cabs_basis = Some("def2-svp-ri".into());
    let a = CcsdF12Method::<false, false>::from_context(&context, &options)
        .unwrap()
        .compute()
        .unwrap();
    let b = CcsdF12Method::<true, false>::from_context(&context, &options)
        .unwrap()
        .compute()
        .unwrap();
    for (method, expected) in [
        (MethodId::CcsdF12a, a.total_energy),
        (MethodId::CcsdF12b, b.total_energy),
        (MethodId::CcsdTF12a, a.total_energy),
        (MethodId::CcsdTF12b, b.total_energy),
    ] {
        let output = create_solver(method, &context, &options)
            .unwrap()
            .compute()
            .unwrap();
        assert_eq!(output.method, method.name());
        close(output.e_total, expected); // Two active electrons: triples vanish.
        close(output.e_scf, hf.total_energy);
        close(output.e_total - output.e_scf, output.e_corr);
        close(
            output.components["E_CCSD"]
                + output.components["E_F12_fixed"]
                + output.components["E_F12_coupling"]
                + output.components["E_CABS_singles"]
                + output
                    .components
                    .get("E_scaled_triples")
                    .copied()
                    .unwrap_or(0.),
            output.e_corr,
        );
        output.validate().unwrap();
    }
    for unsupported in [
        PostHFOptions {
            frozen_virtual: 1,
            ..options.clone()
        },
        PostHFOptions {
            density_fitting: true,
            ..options.clone()
        },
        PostHFOptions {
            root: 1,
            n_roots: 2,
            ..options.clone()
        },
    ] {
        assert!(create_solver(MethodId::CcsdF12a, &context, &unsupported).is_err());
    }
    options.f12.approximation = F12Approximation::ApproximationA;
    assert!(create_solver(MethodId::CcsdF12a, &context, &options).is_err());
}

#[test]
fn every_ccsd_f12_nuclear_gradient_component_matches_independent_energies() {
    use hf_rs::post_hf::numerical::NumericalGradientOptions;
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/ccsd_f12_reference.json")).unwrap();
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/ccsd_f12_gradient_reference.json")).unwrap();
    let mut components = 0;
    for nonlocal in [false, true] {
        let (atoms, obs, hf, base) = setup(&data, nonlocal);
        for case in reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["nonlocal_ecp"].as_bool().unwrap() == nonlocal)
        {
            let fc = case["frozen_core"].as_u64().unwrap() as usize;
            for variant in [CCSDF12Variant::F12a, CCSDF12Variant::F12b] {
                let engine = RHFCCSDF12::new(&hf, &atoms, &obs, variant)
                    .with_triples(true)
                    .with_f12_options(MP2F12Options {
                        frozen_core: fc,
                        ..base.clone()
                    })
                    .with_ccsd_options(controls());
                let coarse = engine
                    .numerical_gradient(&NumericalGradientOptions {
                        step_bohr: 0.001,
                        ..Default::default()
                    })
                    .unwrap();
                let fine = engine
                    .numerical_gradient(&NumericalGradientOptions {
                        step_bohr: 0.0005,
                        ..Default::default()
                    })
                    .unwrap();
                assert_eq!(fine.energy_evaluations, 13);
                let key = if variant == CCSDF12Variant::F12a {
                    "gradient_a"
                } else {
                    "gradient_b"
                };
                let expected = matrix(&case[key]);
                let mut coarse_error = 0f64;
                let mut fine_error = 0f64;
                for atom in 0..atoms.len() {
                    for axis in 0..3 {
                        let target = expected[(atom, axis)];
                        let c = (coarse.gradient[atom][axis] - target).abs();
                        let f = (fine.gradient[atom][axis] - target).abs();
                        assert!(
                            c < 1e-7 && f < 1e-7,
                            "{variant:?} ECP={nonlocal} fc={fc} component={atom},{axis}: coarse={c:e} fine={f:e}"
                        );
                        coarse_error = coarse_error.max(c);
                        fine_error = fine_error.max(f);
                        assert!((fine.gradient[0][axis] + fine.gradient[1][axis]).abs() < 1e-9);
                        components += 1;
                    }
                }
                assert!(
                    fine_error < 0.35 * coarse_error,
                    "step convergence: coarse={coarse_error:e}, fine={fine_error:e}"
                );
                println!(
                    "CCSD(T*)-F12 {variant:?} ECP={nonlocal} fc={fc}: all 6 gradient components, coarse_error={coarse_error:e}, fine_error={fine_error:e}"
                );
            }
        }
    }
    assert_eq!(components, 48);
}

#[test]
fn frozen_ccsd_f12_retains_nonzero_triples_and_all_active_amplitudes() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/ccsd_f12_frozen_reference.json")).unwrap();
    for nonlocal in [false, true] {
        let (atoms, obs, hf, base) = setup(&data, nonlocal);
        let case = data["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| {
                c["nonlocal_ecp"].as_bool().unwrap() == nonlocal
                    && c["frozen_core"].as_u64().unwrap() == 1
            })
            .unwrap();
        let result = RHFCCSDF12::new(&hf, &atoms, &obs, CCSDF12Variant::F12b)
            .with_f12_options(MP2F12Options {
                frozen_core: 1,
                ..base
            })
            .with_ccsd_options(controls())
            .with_triples(true)
            .compute_checked()
            .unwrap();
        for (actual, key) in [
            (result.hf_energy, "e_hf"),
            (result.cabs_singles_energy, "e_cabs"),
            (result.ccsd_correlation_energy, "e_cc"),
            (result.f12_correction_energy, "e_fixed"),
            (result.total_energy, "e_total_b"),
            (result.triples_energy.unwrap(), "e_triples"),
            (result.scaled_triples_energy.unwrap(), "e_scaled_triples"),
        ] {
            close(actual, case[key].as_f64().unwrap());
        }
        assert!(result.triples_energy.unwrap().abs() > 1e-8);
        let cc = &result.ccsd_result;
        assert_eq!(cc.n_occ, 2);
        let alignment = hf
            .mo_coefficients
            .transpose()
            .dot(&hf.overlap_matrix)
            .dot(&matrix(&case["mo_coefficients"]));
        let phase: Vec<_> = (0..alignment.rows)
            .map(|i| {
                assert!(alignment[(i, i)].abs() > 1. - 1e-9);
                alignment[(i, i)].signum()
            })
            .collect();
        let t1 = vector(&case["t1"]);
        let t2 = vector(&case["t2"]);
        let (no, nv) = (cc.n_occ, cc.n_virt);
        let mut maximum = 0f64;
        for i in 0..no {
            for a in 0..nv {
                let index = i * nv + a;
                maximum = maximum.max(
                    (cc.t1_amplitudes[index] * phase[1 + i] * phase[hf.n_occ + a] - t1[index])
                        .abs(),
                );
                for j in 0..no {
                    for b in 0..nv {
                        let index = ((i * no + j) * nv + a) * nv + b;
                        let value = cc.t2_amplitudes[index]
                            * phase[1 + i]
                            * phase[1 + j]
                            * phase[hf.n_occ + a]
                            * phase[hf.n_occ + b];
                        maximum = maximum.max((value - t2[index]).abs());
                    }
                }
            }
        }
        assert!(maximum < 1e-9);
        println!(
            "Frozen He3 ECP={nonlocal}: all 42 amplitudes, max_error={maximum:e}, (T*)={:e}",
            result.scaled_triples_energy.unwrap()
        );
    }
}
