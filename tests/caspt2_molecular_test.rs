// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    context::PostHFContext,
    options::{ActiveSpace, PostHFOptions},
    post_hf::caspt2_typed::solve_caspt2,
    *,
};
fn number(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap()
}
fn setup_model(model: &serde_json::Value) -> (Vec<Atom>, Vec<ContractedGaussian>) {
    let mut atoms: Vec<_> = model["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            Atom::new(
                a["symbol"].as_str().unwrap(),
                std::array::from_fn(|k| number(&a["position"][k])),
            )
        })
        .collect();
    if !model["ecp"].is_null() {
        atoms[0].ecp = Some(std::sync::Arc::new(ecp::EcpDef {
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
    let basis = if let Some(gbs) = model["gaussian94"].as_str() {
        build_basis(&atoms, &parse_gaussian94(gbs))
    } else {
        build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap())
    };
    (atoms, basis)
}

#[test]
fn molecular_energies_match_openmolcas_with_the_same_shifts_and_canonical_metric() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/caspt2_openmolcas_reference.json")).unwrap();
    let mut maximum = 0.0_f64;
    let mut count = 0;
    for model in data["cases"].as_array().unwrap() {
        let (atoms, basis) = setup_model(model);
        let hf = run_scf_with_charge(
            &atoms,
            &basis,
            model["charge"].as_i64().unwrap() as i32,
            250,
            1e-14,
        )
        .unwrap();
        let context = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
        let ne = model["active_electrons"].as_u64().unwrap() as usize;
        let na = model["n_active"].as_u64().unwrap() as usize;
        for expected in model["results"].as_array().unwrap() {
            let mut opts = PostHFOptions {
                active: Some(ActiveSpace::new(ne.div_ceil(2), ne / 2, na)),
                energy_tol: 1e-13,
                residual_tol: 1e-11,
                max_iter: 1000,
                ..Default::default()
            };
            opts.cas.ipea_shift = number(&expected["ipea_shift"]);
            opts.cas.real_shift = number(&expected["real_shift"]);
            opts.cas.imaginary_shift = number(&expected["imaginary_shift"]);
            opts.cas.orbital_gradient_tol = 1e-9;
            let result = solve_caspt2(&context, &opts).unwrap();
            let delta = (result.output.e_total - number(&expected["e_total"])).abs();
            println!(
                "CASPT2 {} IPEA={} real={} imag={}: ref={:.12} total={:.12} error={delta:e}",
                model["name"],
                opts.cas.ipea_shift,
                opts.cas.real_shift,
                opts.cas.imaginary_shift,
                result.reference_cas_energy,
                result.output.e_total
            );
            maximum = maximum.max(delta);
            count += 1;
            assert!(delta < 1e-6, "{}", model["name"]);
            assert!(
                (result.output.e_corr - (result.output.e_total - hf.total_energy)).abs() < 1e-12
            );
            assert!(result.output.one_rdm.is_none());
        }
    }
    println!("OpenMolcas: count={count}, maximum={maximum:e}");
}

fn options(ipea: f64, imaginary: f64, optimize: bool) -> PostHFOptions {
    let mut opts = PostHFOptions {
        active: Some(ActiveSpace::new(1, 1, 2)),
        energy_tol: 1e-13,
        residual_tol: 1e-11,
        max_iter: 1000,
        ..Default::default()
    };
    opts.cas.ipea_shift = ipea;
    opts.cas.imaginary_shift = imaginary;
    opts.cas.optimize_orbitals = optimize;
    opts.cas.orbital_gradient_tol = 1e-9;
    opts
}

#[test]
fn all_nuclear_components_match_independent_energy_stencils_including_nonlocal_ecp() {
    let models: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_molecular_reference.json")).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/caspt2_molecular_reference.json")).unwrap();
    let step = number(&oracle["step"]);
    let mut max_five = 0.0_f64;
    let mut max_api = 0.0_f64;
    let mut max_translation = 0.0_f64;
    let mut count = 0;
    for case in oracle["cases"].as_array().unwrap() {
        let index = case["model_index"].as_u64().unwrap() as usize;
        let model = &models["cases"][index];
        let (atoms, basis) = setup_model(model);
        let charge = model["charge"].as_i64().unwrap() as i32;
        for (mode, expected) in case["results"].as_array().unwrap().iter().enumerate() {
            let opts = options(
                number(&expected["ipea_shift"]),
                number(&expected["imaginary_shift"]),
                index == 3,
            );
            let settings = post_hf::numerical::NumericalGradientOptions {
                step_bohr: step,
                ..Default::default()
            };
            let api = post_hf::caspt2_typed::numerical_gradient(
                &atoms,
                &basis,
                charge,
                ao::BasisKind::Cartesian,
                &opts,
                &settings,
            )
            .unwrap();
            assert_eq!(api.kind, registry::GradientKind::Numerical);
            assert!(
                (api.reference_energy - number(&expected["e_total"])).abs() < 1e-10,
                "model={index}, mode={mode}, Rust={:e}, NumPy={:e}",
                api.reference_energy,
                number(&expected["e_total"])
            );
            let mut gradient = vec![[0.; 3]; atoms.len()];
            for (component, value) in case["gradient"][mode]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .enumerate()
            {
                let mut values = Vec::new();
                for offset in [-2., -1., 1., 2.] {
                    let mut point = atoms.clone();
                    point[component / 3].position[component % 3] += offset * step;
                    let moved =
                        post_hf::numerical::displaced_basis(&basis, &atoms, &point).unwrap();
                    let hf = scf::run_scf_in_space_with_convergence(
                        &point,
                        &moved,
                        charge,
                        200,
                        scf::RhfConvergence {
                            energy: 1e-13,
                            residual: 1e-12,
                            density: 1e-12,
                        },
                        diis::ConvAlgo::Adiissoscf,
                        None,
                        ao::BasisKind::Cartesian,
                    )
                    .unwrap();
                    let ctx = PostHFContext::from_rhf(&hf, &point, &moved, None).unwrap();
                    values.push(solve_caspt2(&ctx, &opts).unwrap().output.e_total);
                }
                let five = (values[0] - 8. * values[1] + 8. * values[2] - values[3]) / (12. * step);
                max_five = max_five.max((five - value).abs());
                max_api = max_api.max((api.gradient[component / 3][component % 3] - value).abs());
                gradient[component / 3][component % 3] = five;
                count += 1;
            }
            for axis in 0..3 {
                max_translation =
                    max_translation.max(gradient.iter().map(|g| g[axis]).sum::<f64>().abs());
            }
        }
    }
    println!(
        "CASPT2 gradients: count={count}, five_point={max_five:e}, API={max_api:e}, translation={max_translation:e}"
    );
    assert_eq!(count, 48);
    assert!(max_five < 1e-6);
    assert!(max_api < 1e-6);
    assert!(max_translation < 1e-9);
}

#[test]
fn typed_registry_and_legacy_interfaces_share_reference_and_correlation_energies() {
    use post_hf::caspt2::{CASPT2, CASPT2Options};
    use post_hf::casscf::{CASSCF, CASSCFOptions};
    let models: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_molecular_reference.json")).unwrap();
    let model = &models["cases"][3];
    let (atoms, basis) = setup_model(model);
    let hf = run_scf_with_charge(&atoms, &basis, 1, 200, 1e-14).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let opts = options(0.25, 0., true);
    let typed = solve_caspt2(&ctx, &opts).unwrap();
    let registry = registry::create_solver(registry::MethodId::Caspt2, &ctx, &opts)
        .unwrap()
        .compute()
        .unwrap();
    assert!((registry.e_total - typed.output.e_total).abs() < 1e-12);
    let legacy = CASPT2::from_hf(&hf, CASPT2Options::new(2, 2)).unwrap();
    assert!((legacy.total_energy - typed.output.e_total).abs() < 1e-8);
    let mut settings = CASSCFOptions::new(2, 2);
    settings.grad_tol = 1e-9;
    let reference = CASSCF::new(&hf, settings).compute().unwrap();
    let supplied = CASPT2::new(&hf, &reference, CASPT2Options::new(2, 2))
        .compute()
        .unwrap();
    assert!((supplied.total_energy - typed.output.e_total).abs() < 1e-9);
    assert!(
        (legacy.e_external + legacy.e_semi_internal + legacy.e_singles
            - legacy.pt2_correlation_energy)
            .abs()
            < 1e-12
    );
    assert!((registry.components["E_CASSCF"] - typed.reference_cas_energy).abs() < 1e-12);
    let mut invalid = opts.clone();
    invalid.n_roots = 2;
    assert!(solve_caspt2(&ctx, &invalid).is_err());
    invalid = opts.clone();
    invalid.cas.real_shift = 0.1;
    invalid.cas.imaginary_shift = 0.1;
    assert!(solve_caspt2(&ctx, &invalid).is_err());
    invalid = opts;
    invalid.max_iter = 1;
    invalid.cas.optimize_orbitals = false;
    assert!(solve_caspt2(&ctx, &invalid).is_err());
}

#[test]
fn rectangular_orbitals_and_legacy_gradient_dispatch_preserve_the_energy_surface() {
    let models: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_molecular_reference.json")).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/caspt2_molecular_reference.json")).unwrap();
    let model = &models["cases"][1];
    let (atoms, basis) = setup_model(model);
    let mut hf = run_scf(&atoms, &basis, 200, 1e-14).unwrap();
    let rows = model["initial_mo_coeff"].as_array().unwrap();
    hf.mo_coefficients = Matrix::from_vec(
        rows.len(),
        3,
        rows.iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(number))
            .collect(),
    );
    hf.orbital_energies.truncate(3);
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    for expected in oracle["extra_cases"][0]["results"].as_array().unwrap() {
        let opts = options(
            number(&expected["ipea_shift"]),
            number(&expected["imaginary_shift"]),
            false,
        );
        let result = solve_caspt2(&ctx, &opts).unwrap();
        assert_eq!(result.mo_coefficients.cols, 3);
        assert!((result.output.e_total - number(&expected["e_total"])).abs() < 1e-10);
    }
    let model = &models["cases"][3];
    let (atoms, basis) = setup_model(model);
    let opts = options(0.25, 0., true);
    let typed = post_hf::caspt2_typed::numerical_gradient(
        &atoms,
        &basis,
        1,
        ao::BasisKind::Cartesian,
        &opts,
        &Default::default(),
    )
    .unwrap();
    let legacy = post_hf::hessian::compute_post_hf_gradient(
        &atoms,
        &get_builtin_basis("sto-3g").unwrap(),
        1,
        post_hf::hessian::PostHFType::CASPT2,
        200,
        1e-13,
    );
    let mut maximum = 0.0_f64;
    for (a, b) in legacy.iter().flatten().zip(typed.gradient.iter().flatten()) {
        maximum = maximum.max((a - b).abs());
    }
    println!("CASPT2 legacy gradient: components=9, error={maximum:e}");
    assert!(maximum < 1e-6);
}
