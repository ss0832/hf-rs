// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    context::{PostHFContext, Reference},
    options::{ActiveSpace, PostHFOptions, ReferenceRelaxation},
    post_hf::casscf_engine::{solve_cas, solve_cas_in_orbitals},
    registry::{MethodId, create_solver},
    *,
};
fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/mr_molecular_reference.json")).unwrap()
}
fn number(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap()
}
fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(number))
            .collect(),
    )
}
fn strict() -> PostHFOptions {
    let mut o = PostHFOptions {
        max_iter: 150,
        energy_tol: 1e-14,
        residual_tol: 1e-12,
        ..Default::default()
    };
    o.dsrg.bch_tolerance = 1e-14;
    o.cas.optimize_orbitals = false;
    o.cas.target_s_squared = Some(0.);
    o
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

fn reference(model: &serde_json::Value) -> (Vec<Atom>, Vec<ContractedGaussian>, HFResult) {
    let (atoms, basis) = setup_model(model);
    let mut hf = run_scf_with_charge(
        &atoms,
        &basis,
        model["charge"].as_i64().unwrap() as i32,
        250,
        1e-14,
    )
    .unwrap();
    assert!((hf.total_energy - number(&model["e_scf"])).abs() < 1e-10);
    if !model["n_mo"].is_null() {
        hf.mo_coefficients = matrix(&model["initial_mo_coeff"]);
        let f = hf
            .mo_coefficients
            .transpose()
            .dot(&hf.fock_matrix)
            .dot(&hf.mo_coefficients);
        hf.orbital_energies = (0..f.rows).map(|p| f[(p, p)]).collect();
    }
    (atoms, basis, hf)
}
#[test]
fn typed_mr_casci_energies_match_independent_fock_roots_for_molecules_rectangular_space_and_ecp() {
    let data = data();
    let mut maximum = 0f64;
    for model in data["cases"].as_array().unwrap() {
        let (atoms, basis, hf) = reference(model);
        let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
        let mut opts = strict();
        opts.active = Some(ActiveSpace::new(1, 1, 2));
        opts.cas.optimize_orbitals = model["optimize"].as_bool().unwrap_or(false);
        opts.cas.orbital_gradient_tol = 1e-9;

        for (method, expected) in [
            (
                MethodId::DsrgMrPt2,
                number(&model["e_cas"]) + number(&model["pt2"]),
            ),
            (
                MethodId::DsrgMrPt3,
                number(&model["e_cas"]) + number(&model["pt2"]) + number(&model["pt3"]),
            ),
            (MethodId::MrLdsrg2, number(&model["transformed"]["zero"])),
        ] {
            let output = create_solver(method, &ctx, &opts)
                .unwrap()
                .compute()
                .unwrap();
            let error = (output.e_total - expected).abs();
            maximum = maximum.max(error);
            assert!(
                error < 1e-9,
                "{} {}: {error:e}",
                model["name"],
                method.name()
            );
            assert!(
                (output.components[if opts.cas.optimize_orbitals {
                    "E_CASSCF"
                } else {
                    "E_CASCI"
                }] - number(&model["e_cas"]))
                .abs()
                    < 1e-10
            );
            assert!(
                (output.components["E_PT2"] - number(&model["pt2"])).abs() < 1e-10,
                "{} {} PT2 {:e} vs {:e}",
                model["name"],
                method.name(),
                output.components["E_PT2"],
                number(&model["pt2"])
            );
            assert!((output.e_scf + output.e_corr - output.e_total).abs() < 1e-14);
            assert!(output.components["lambda2_norm"] > 1e-4);
            assert!(output.one_rdm.is_none());
            if method == MethodId::MrLdsrg2 {
                assert!(output.components["source_residual_max"] < 1e-12);
            }
            let mut permuted = opts.clone();
            let nc = hf.n_occ - 1;
            permuted.active.as_mut().unwrap().orbitals = Some(vec![nc + 1, nc]);
            let alternate = create_solver(method, &ctx, &permuted)
                .unwrap()
                .compute()
                .unwrap();
            assert!((alternate.e_total - output.e_total).abs() < 1e-10);
        }
    }
    println!("Molecular MR typed energies: 12 methods/models, max error={maximum:e}");
}
#[test]
fn cas_zero_and_full_limits_and_optimized_reference_are_preserved() {
    let data = data();
    let model = &data["cases"][0];
    let (atoms, basis, hf) = reference(model);
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let mut opts = strict();
    for (mr, sr) in [
        (MethodId::DsrgMrPt2, MethodId::SrDsrgPt2),
        (MethodId::DsrgMrPt3, MethodId::SrDsrgPt3),
        (MethodId::MrLdsrg2, MethodId::SrLdsrg2),
    ] {
        let a = create_solver(mr, &ctx, &opts).unwrap().compute().unwrap();
        let b = create_solver(sr, &ctx, &opts).unwrap().compute().unwrap();
        assert!(
            (a.e_total - b.e_total).abs() < 1e-10,
            "{} zero-CAS limit",
            mr.name()
        );
    }
    opts.active = Some(ActiveSpace::new(1, 1, 3));
    let cas = solve_cas(Reference::Rhf(&hf), &opts, None, false).unwrap();
    for method in [MethodId::DsrgMrPt2, MethodId::DsrgMrPt3, MethodId::MrLdsrg2] {
        let result = create_solver(method, &ctx, &opts)
            .unwrap()
            .compute()
            .unwrap();
        assert!((result.e_total - cas.output.e_total).abs() < 1e-10);
        assert_eq!(result.components["E_PT2"], 0.);
    }
    opts.active = Some(ActiveSpace::new(1, 1, 2));
    opts.cas.optimize_orbitals = true;
    opts.cas.orbital_gradient_tol = 1e-9;
    let cas = solve_cas(Reference::Rhf(&hf), &opts, None, true).unwrap();
    opts.dsrg.flow_parameter = 0.;
    let result = create_solver(MethodId::MrLdsrg2, &ctx, &opts)
        .unwrap()
        .compute()
        .unwrap();
    assert!((result.e_total - cas.output.e_total).abs() < 1e-10);
    assert!(result.components.contains_key("E_CASSCF"));
}
#[test]
fn fixed_orbital_cas_borrows_ao_reference_and_rejects_invalid_orbitals_and_mr_options() {
    let data = data();
    let model = &data["cases"][0];
    let (atoms, basis, hf) = reference(model);
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let mut opts = strict();
    opts.active = Some(ActiveSpace::new(1, 1, 2));
    let cas = solve_cas(Reference::Rhf(&hf), &opts, None, false).unwrap();
    let mut k = Matrix::zeros(3, 3);
    k[(0, 1)] = 0.23;
    k[(1, 0)] = -0.23;
    let rotated = hf.mo_coefficients.dot(&k.exp());
    let other = solve_cas_in_orbitals(Reference::Rhf(&hf), &rotated, &opts).unwrap();
    assert!((cas.output.e_total - other.output.e_total).abs() < 1e-10);
    let mut bad = rotated.clone();
    bad.data[0] += 0.1;
    assert!(solve_cas_in_orbitals(Reference::Rhf(&hf), &bad, &opts).is_err());
    assert!(solve_cas_in_orbitals(Reference::Rhf(&hf), &Matrix::zeros(2, 2), &opts).is_err());
    for variant in 0..5 {
        let mut bad = opts.clone();
        match variant {
            0 => bad.density_fitting = true,
            1 => bad.frozen_core = 1,
            2 => bad.frozen_virtual = 1,
            3 => bad.cas.target_s_squared = Some(2.),
            _ => bad.n_roots = 2,
        }
        assert!(create_solver(MethodId::DsrgMrPt2, &ctx, &bad).is_err());
    }
    let values = std::collections::BTreeMap::from([
        ("bch_tolerance".into(), serde_json::json!(1e-14)),
        ("bch_max_order".into(), serde_json::json!(60)),
    ]);
    let parsed = PostHFOptions::from_map(&values, &MethodId::MrLdsrg2.option_schemas());
    assert!(parsed.is_ok());
    let invalid = std::collections::BTreeMap::from([("ode_step".into(), serde_json::json!(0.1))]);
    assert!(PostHFOptions::from_map(&invalid, &MethodId::MrLdsrg2.option_schemas()).is_err());
    assert_eq!(
        MethodId::DsrgMrPt2.gradient_kind(context::RefKind::Uhf, &opts),
        registry::GradientKind::Unsupported
    );
    assert_eq!(
        MethodId::DsrgMrPt2.gradient_kind(context::RefKind::Rhf, &opts),
        registry::GradientKind::Numerical
    );
}

#[test]
fn every_mr_nuclear_gradient_component_matches_independent_all_electron_and_ecp_oracles() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_gradient_reference.json")).unwrap();
    let variants = [MethodId::DsrgMrPt2, MethodId::DsrgMrPt3, MethodId::MrLdsrg2]
        .map(|m| (m, ReferenceRelaxation::None));
    let _ = verify_every_gradient_component(&oracle, &variants);
}

#[test]
fn relaxed_molecular_energies_and_all_ci_coefficients_match_independent_fock_space_oracles() {
    use hf_rs::{
        fci::orbital_rotation::rotate_ci, options::ReferenceRelaxation,
        post_hf::mr_dsrg::relaxation::solve_reference,
    };
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_relaxed_molecular_reference.json")).unwrap();
    let models = data();
    let mut energy_error = 0f64;
    let mut ci_error = 0f64;
    let mut effective_error = 0f64;
    let mut effective_count = 0;
    let mut count = 0;
    for case in oracle["cases"].as_array().unwrap() {
        let model = &models["cases"][case["model_index"].as_u64().unwrap() as usize];
        let (atoms, basis, hf) = reference(model);
        let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
        for expected in case["results"].as_array().unwrap() {
            let mut options = strict();
            options.active = Some(ActiveSpace::new(1, 1, 2));
            options.cas.optimize_orbitals = model["optimize"].as_bool().unwrap_or(false);
            options.cas.orbital_gradient_tol = 1e-9;
            options.dsrg.reference_ci_tol = 1e-10;
            options.dsrg.reference_energy_tol = 1e-12;
            options.dsrg.reference_relaxation = match expected["mode"].as_str().unwrap() {
                "once" => ReferenceRelaxation::Once,
                "twice" => ReferenceRelaxation::Twice,
                "iterate" => ReferenceRelaxation::Iterate,
                _ => panic!("unexpected oracle mode"),
            };
            let order = expected["order"].as_u64().unwrap() as u8;
            let method = match order {
                2 => MethodId::DsrgMrPt2,
                3 => MethodId::DsrgMrPt3,
                _ => MethodId::MrLdsrg2,
            };
            let cas = solve_cas(
                Reference::Rhf(&hf),
                &options,
                None,
                options.cas.optimize_orbitals,
            )
            .unwrap();
            let space = DeterminantSpace::new(2, 1, 1);
            let nc = cas.active_space.inactive.len();
            let solution = solve_reference(&hf, cas, order, &options, 256 * 1024 * 1024).unwrap();
            let error = (solution.total_energy - number(&expected["energy"])).abs();
            energy_error = energy_error.max(error);
            assert!(
                error < 1e-9,
                "{} {order} {} energy error={error:e}",
                case["name"],
                expected["mode"]
            );
            let oracle_c = matrix(&expected["mo_coeff"]);
            if options.dsrg.reference_relaxation == ReferenceRelaxation::Once && order != 4 {
                use hf_rs::{
                    context::spin_antisymmetrized,
                    mo_integrals::{transform_1e_ao_to_mo, transform_eri_ao_to_mo},
                    post_hf::mr_dsrg::{
                        gno::GnoReference, imsrg2::No2bOperator, mr_driven::MrDsrgEquations,
                        sr_dsrg::LdsrgOptions,
                    },
                };
                let ordered =
                    solve_cas_in_orbitals(Reference::Rhf(&hf), &oracle_c, &options).unwrap();
                let density = ordered.exact_root_cumulants(0, 256 * 1024 * 1024).unwrap();
                let reference = GnoReference::from_cas(&density, &ordered).unwrap();
                let equations = MrDsrgEquations::new(&reference, 256 * 1024 * 1024).unwrap();
                let ns = reference.n_spin();
                let one = transform_1e_ao_to_mo(&oracle_c, &hf.core_hamiltonian);
                let g = transform_eri_ao_to_mo(&oracle_c, &hf.eri);
                let mut bare = No2bOperator {
                    zero: hf.nuclear_repulsion_energy,
                    one: Matrix::zeros(ns, ns),
                    two: vec![0.; ns.pow(4)],
                };
                for p in 0..ns {
                    for q in 0..ns {
                        if p % 2 == q % 2 {
                            bare.one[(p, q)] = one[(p / 2, q / 2)];
                        }
                        for r in 0..ns {
                            for t in 0..ns {
                                bare.two[((p * ns + q) * ns + r) * ns + t] =
                                    spin_antisymmetrized(&g, &g, None, p, q, r, t);
                            }
                        }
                    }
                }
                let h = equations.normal_order(&bare).unwrap();
                let effective = equations
                    .perturbative_active_hamiltonian(
                        &h,
                        0.5,
                        order as usize,
                        LdsrgOptions {
                            max_iter: 150,
                            residual_tolerance: 1e-12,
                            energy_tolerance: 1e-14,
                            bch_tolerance: 1e-14,
                            bch_max_order: 80,
                        },
                    )
                    .unwrap();
                let oracle = &expected["initial_effective"];
                let error = (effective.zero - number(&oracle["zero"])).abs();
                effective_error = effective_error.max(error);
                effective_count += 1;
                assert!(error < 1e-8);
                for (i, x) in effective.one.data.iter().enumerate() {
                    effective_error =
                        effective_error.max((x - number(&oracle["one"][i / 4][i % 4])).abs());
                    effective_count += 1;
                    assert!(
                        (x - number(&oracle["one"][i / 4][i % 4])).abs() < 1e-8,
                        "effective one-body {i}"
                    );
                }
                for (i, x) in effective.two.iter().enumerate() {
                    effective_error = effective_error.max((x - number(&oracle["two"][i])).abs());
                    effective_count += 1;
                    assert!(
                        (x - number(&oracle["two"][i])).abs() < 1e-8,
                        "effective two-body {i}"
                    );
                }
            }
            let u = solution
                .mo_coefficients
                .transpose()
                .dot(&hf.overlap_matrix)
                .dot(&oracle_c);
            let mut active_u = Matrix::zeros(2, 2);
            for p in 0..2 {
                for q in 0..2 {
                    active_u[(p, q)] = u[(nc + p, nc + q)];
                }
            }
            let rotated = rotate_ci(
                &space,
                &solution.ci_coefficients,
                &active_u,
                256 * 1024 * 1024,
            )
            .unwrap();
            let target: Vec<_> = expected["coefficients"]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect();
            let overlap: f64 = rotated.iter().zip(&target).map(|(a, b)| a * b).sum();
            let sign = overlap.signum();
            for (i, (a, b)) in rotated.iter().zip(&target).enumerate() {
                let error = (a - sign * b).abs();
                ci_error = ci_error.max(error);
                assert!(
                    error < 1e-8,
                    "{} {order} {} CI {i}: {error:e}",
                    case["name"],
                    expected["mode"]
                );
            }
            assert!(solution.reference_ci_residual < 1e-10);
            match options.dsrg.reference_relaxation {
                ReferenceRelaxation::Once => assert_eq!(solution.reference_iterations, 1),
                ReferenceRelaxation::Twice => assert_eq!(solution.reference_iterations, 2),
                _ => assert!(solution.reference_ci_change < 1e-10),
            }
            let output = create_solver(method, &ctx, &options)
                .unwrap()
                .compute()
                .unwrap();
            assert!((output.e_total - solution.total_energy).abs() < 1e-11);
            assert_eq!(
                output.components["reference_iterations"],
                solution.reference_iterations as f64
            );
            assert!(
                (output.components["E_unrelaxed"] + output.components["E_reference_relaxation"]
                    - output.e_total)
                    .abs()
                    < 1e-14
            );
            assert!(output.one_rdm.is_none());
            assert_eq!(
                method.gradient_kind(context::RefKind::Rhf, &options),
                registry::GradientKind::Numerical
            );
            count += 1;
        }
    }
    assert_eq!(count, 36);
    assert_eq!(effective_count, 2184);
    println!(
        "Relaxed MR: {count} molecular energies, max energy error={energy_error:e}, all CI components error={ci_error:e}, all {effective_count} effective coefficients error={effective_error:e}"
    );
}

#[test]
fn reference_relaxation_controls_limits_and_failed_macroconvergence_are_explicit() {
    use hf_rs::options::ReferenceRelaxation;
    let d = data();
    let (atoms, basis, hf) = reference(&d["cases"][0]);
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let mut options = strict();
    options.active = Some(ActiveSpace::new(1, 1, 2));
    options.dsrg.relax_reference = true;
    let legacy = create_solver(MethodId::DsrgMrPt2, &ctx, &options)
        .unwrap()
        .compute()
        .unwrap();
    options.dsrg.relax_reference = false;
    options.dsrg.reference_relaxation = ReferenceRelaxation::Once;
    let explicit = create_solver(MethodId::DsrgMrPt2, &ctx, &options)
        .unwrap()
        .compute()
        .unwrap();
    assert_eq!(legacy.e_total, explicit.e_total);
    options.dsrg.reference_relaxation = ReferenceRelaxation::Iterate;
    options.dsrg.max_reference_iterations = 1;
    assert!(
        create_solver(MethodId::DsrgMrPt2, &ctx, &options)
            .unwrap()
            .compute()
            .is_err()
    );
    options.dsrg.max_reference_iterations = 30;
    for mode in [
        ReferenceRelaxation::Once,
        ReferenceRelaxation::Twice,
        ReferenceRelaxation::Iterate,
    ] {
        options.dsrg.reference_relaxation = mode;
        options.active = Some(ActiveSpace::new(1, 1, 3));
        let cas = solve_cas(Reference::Rhf(&hf), &options, None, false).unwrap();
        for method in [MethodId::DsrgMrPt2, MethodId::DsrgMrPt3, MethodId::MrLdsrg2] {
            let output = create_solver(method, &ctx, &options)
                .unwrap()
                .compute()
                .unwrap();
            assert!((output.e_total - cas.output.e_total).abs() < 1e-10);
            assert!(output.components["E_reference_relaxation"].abs() < 1e-10);
        }
        options.active = None;
        for (mr, sr) in [
            (MethodId::DsrgMrPt2, MethodId::SrDsrgPt2),
            (MethodId::DsrgMrPt3, MethodId::SrDsrgPt3),
            (MethodId::MrLdsrg2, MethodId::SrLdsrg2),
        ] {
            let output = create_solver(mr, &ctx, &options)
                .unwrap()
                .compute()
                .unwrap();
            let mut unrelaxed = options.clone();
            unrelaxed.dsrg.reference_relaxation = ReferenceRelaxation::None;
            let target = create_solver(sr, &ctx, &unrelaxed)
                .unwrap()
                .compute()
                .unwrap();
            assert!((output.e_total - target.e_total).abs() < 1e-10);
            assert!(create_solver(sr, &ctx, &options).is_err());
        }
    }
    let values = std::collections::BTreeMap::from([
        ("reference_relaxation".into(), serde_json::json!("iterate")),
        ("max_reference_iterations".into(), serde_json::json!(40)),
        ("reference_energy_tol".into(), serde_json::json!(1e-12)),
        ("reference_ci_tol".into(), serde_json::json!(1e-10)),
    ]);
    let parsed = PostHFOptions::from_map(&values, &MethodId::DsrgMrPt2.option_schemas()).unwrap();
    assert_eq!(parsed.dsrg.relaxation(), ReferenceRelaxation::Iterate);
    assert_eq!(parsed.dsrg.max_reference_iterations, 40);
    for (key, value) in [
        ("reference_relaxation", serde_json::json!("invalid")),
        ("max_reference_iterations", serde_json::json!(0)),
        ("reference_energy_tol", serde_json::json!(0)),
        ("reference_ci_tol", serde_json::json!(-1)),
    ] {
        let bad = std::collections::BTreeMap::from([(key.into(), value)]);
        assert!(PostHFOptions::from_map(&bad, &MethodId::DsrgMrPt2.option_schemas()).is_err());
    }
    assert!(PostHFOptions::from_map(&values, &MethodId::SrDsrgPt2.option_schemas()).is_err());
}

fn verify_every_gradient_component(
    oracle: &serde_json::Value,
    variants: &[(MethodId, ReferenceRelaxation)],
) -> (f64, f64, f64, usize) {
    verify_every_gradient_with_fitting(oracle, variants, false)
}
fn verify_every_gradient_with_fitting(
    oracle: &serde_json::Value,
    variants: &[(MethodId, ReferenceRelaxation)],
    fitted: bool,
) -> (f64, f64, f64, usize) {
    let data = data();
    let methods: Vec<_> = variants.iter().map(|v| v.0).collect();
    let mut maximum = 0f64;
    let mut api_maximum = 0f64;
    let mut translation_maximum = 0f64;
    let mut count = 0;
    for case in oracle["cases"].as_array().unwrap() {
        let model = &data["cases"][case["model_index"].as_u64().unwrap() as usize];
        let (atoms, basis) = setup_model(model);
        let charge = model["charge"].as_i64().unwrap() as i32;
        let basis_map = if let Some(gbs) = model["gaussian94"].as_str() {
            parse_gaussian94(gbs)
        } else {
            get_builtin_basis("sto-3g").unwrap()
        };
        let step = number(&case["step"]);
        let n = atoms.len() * 3;
        let mut gradients = vec![vec![0.; n]; methods.len()];
        let mut options = strict();
        options.density_fitting = fitted;
        options.active = Some(ActiveSpace::new(1, 1, 2));
        options.dsrg.reference_ci_tol = 1e-10;
        options.dsrg.reference_energy_tol = 1e-12;
        for component in 0..n {
            let mut values = vec![vec![0.; methods.len()]; 4];
            for (j, offset) in [-2., -1., 1., 2.].iter().enumerate() {
                let mut moved = atoms.clone();
                moved[component / 3].position[component % 3] += offset * step;
                let moved_basis = build_basis(&moved, &basis_map);
                let hf = scf::run_scf_in_space_with_convergence(
                    &moved,
                    &moved_basis,
                    charge,
                    300,
                    scf::RhfConvergence {
                        energy: 1e-14,
                        residual: 1e-13,
                        density: 1e-13,
                    },
                    ConvAlgo::Cdiis,
                    None,
                    ao::BasisKind::Cartesian,
                )
                .unwrap();
                let aux = if fitted {
                    Some(context::AuxBasis::from_builtin(&moved, "def2-svp-ri").unwrap())
                } else {
                    None
                };
                let ctx = PostHFContext::from_rhf(&hf, &moved, &moved_basis, aux.as_ref()).unwrap();
                for (k, method) in methods.iter().enumerate() {
                    options.dsrg.reference_relaxation = variants[k].1;
                    values[j][k] = create_solver(*method, &ctx, &options)
                        .unwrap()
                        .compute()
                        .unwrap()
                        .e_total;
                }
            }
            for k in 0..methods.len() {
                gradients[k][component] = (values[0][k] - 8. * values[1][k] + 8. * values[2][k]
                    - values[3][k])
                    / (12. * step);
            }
        }
        for k in 0..methods.len() {
            for component in 0..n {
                let expected = number(&case["gradient"][k][component]);
                let error = (gradients[k][component] - expected).abs();
                maximum = maximum.max(error);
                count += 1;
                assert!(
                    error < 1e-6,
                    "{} {} component {component}, error={error:e}",
                    model["name"],
                    methods[k].name()
                );
            }
            for axis in 0..3 {
                let total = (0..atoms.len())
                    .map(|a| gradients[k][a * 3 + axis])
                    .sum::<f64>();
                translation_maximum = translation_maximum.max(total.abs());
                assert!(
                    total.abs() < 1e-9,
                    "{} {} axis {axis} translation={total:e}",
                    model["name"],
                    methods[k].name()
                );
            }
            options.dsrg.reference_relaxation = variants[k].1;
            let aux = if fitted {
                Some(context::AuxBasis::from_builtin(&atoms, "def2-svp-ri").unwrap())
            } else {
                None
            };
            let numerical = post_hf::mr_dsrg::typed_mr::numerical_gradient_with_aux(
                methods[k],
                &atoms,
                &basis,
                aux.as_ref(),
                charge,
                ao::BasisKind::Cartesian,
                &options,
                &post_hf::numerical::NumericalGradientOptions {
                    step_bohr: 5e-4,
                    scf_max_iter: 300,
                    scf_tolerance: 1e-14,
                    scf_residual_tolerance: 1e-13,
                    scf_density_tolerance: 1e-13,
                },
            )
            .unwrap();
            assert_eq!(numerical.kind, registry::GradientKind::Numerical);
            assert_eq!(numerical.energy_evaluations, 2 * n + 1);
            for component in 0..n {
                let expected = number(&case["gradient"][k][component]);
                let error = (numerical.gradient[component / 3][component % 3] - expected).abs();
                api_maximum = api_maximum.max(error);
                assert!(
                    error < 1e-6,
                    "{} API {} component {component}, error={error:e}",
                    model["name"],
                    methods[k].name()
                );
            }
        }
    }
    assert_eq!(count, 15 * variants.len());
    println!(
        "MR nuclear derivatives: all {count} components, five-point error={maximum:e}, API error={api_maximum:e}, translation={translation_maximum:e}"
    );
    (maximum, api_maximum, translation_maximum, count)
}

#[test]
fn every_relaxed_mr_nuclear_gradient_component_matches_independent_all_electron_and_ecp_oracles() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_relaxed_gradient_reference.json")).unwrap();
    let variants: Vec<_> = [MethodId::DsrgMrPt2, MethodId::DsrgMrPt3, MethodId::MrLdsrg2]
        .into_iter()
        .flat_map(|m| {
            [
                ReferenceRelaxation::Once,
                ReferenceRelaxation::Twice,
                ReferenceRelaxation::Iterate,
            ]
            .map(|r| (m, r))
        })
        .collect();
    let _ = verify_every_gradient_component(&oracle, &variants);
}

#[test]
fn compatibility_mr_methods_match_all_relaxed_energies_and_reference_density_components() {
    use hf_rs::post_hf::mr_dsrg::{DSRGMRPT2, DSRGMRPT3, MRDSRGOptions, MRLDSRG2, RelaxRef};
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_relaxed_molecular_reference.json")).unwrap();
    let models = data();
    let mut energy_error = 0f64;
    let mut density_error = 0f64;
    let mut energies = 0;
    let mut components = 0;
    for case in oracle["cases"].as_array().unwrap().iter().take(3) {
        let model = &models["cases"][case["model_index"].as_u64().unwrap() as usize];
        let (_, _, hf) = reference(model);
        for expected in case["results"].as_array().unwrap() {
            let mut options = MRDSRGOptions::new(2, 2);
            options.max_iter = 150;
            options.tol = 1e-12;
            options.relax_ref = match expected["mode"].as_str().unwrap() {
                "once" => RelaxRef::Once,
                "twice" => RelaxRef::Twice,
                _ => RelaxRef::Iterate,
            };
            let result = match expected["order"].as_u64().unwrap() {
                2 => DSRGMRPT2::new(&hf, options).compute().unwrap(),
                3 => DSRGMRPT3::new(&hf, options).compute().unwrap(),
                _ => MRLDSRG2::new(&hf, options).compute().unwrap(),
            };
            let error = (result.relaxed_total_energy - number(&expected["energy"])).abs();
            energy_error = energy_error.max(error);
            assert!(
                error < 1e-9,
                "{} {} {} energy {error:e}",
                case["name"],
                expected["order"],
                expected["mode"]
            );
            let nc = hf.n_occ - 1;
            let c = matrix(&expected["mo_coeff"]);
            let ci: Vec<_> = expected["coefficients"]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect();
            let mut d = Matrix::zeros(c.cols, c.cols);
            for i in 0..nc {
                d[(i, i)] = 2.;
            }
            // Exact alpha/beta partial traces in the CAS(1 alpha,1 beta) basis.
            for p in 0..2 {
                for q in 0..2 {
                    d[(nc + p, nc + q)] = (0..2)
                        .map(|r| ci[p * 2 + r] * ci[q * 2 + r] + ci[r * 2 + p] * ci[r * 2 + q])
                        .sum();
                }
            }
            let target = c.dot(&d).dot(&c.transpose());
            let c = result.reference_mo_coefficients.as_ref().unwrap();
            let actual = c.dot(&result.one_rdm_mo).dot(&c.transpose());
            for (i, (a, b)) in actual.data.iter().zip(&target.data).enumerate() {
                let error = (a - b).abs();
                density_error = density_error.max(error);
                assert!(
                    error < 1e-8,
                    "{} density component {i}: {error:e}",
                    case["name"]
                );
                components += 1;
            }
            assert!(result.one_rdm().is_none());
            assert!(result.natural_orbital_occupations().is_none());
            assert!(result.converged());
            assert_eq!(result.total_energy(), result.relaxed_total_energy);
            let output = result.to_output().unwrap();
            assert_eq!(output.e_scf, hf.total_energy);
            assert_eq!(output.e_total, result.relaxed_total_energy);
            assert!((output.e_scf + output.e_corr - output.e_total).abs() < 1e-14);
            assert!((output.components["E_CASCI"] - number(&expected["e_cas"])).abs() < 1e-10);
            assert!(output.one_rdm.is_none());
            if energies == 0 {
                let mut unconverged = result.clone();
                unconverged.converged = false;
                assert!(!unconverged.converged());
                assert!(unconverged.to_output().is_err());
            }
            energies += 1;
        }
        let mut options = MRDSRGOptions::new(2, 2);
        options.max_iter = 150;
        options.tol = 1e-12;
        options.relax_ref = RelaxRef::None;
        let results = [
            (
                DSRGMRPT2::new(&hf, options.clone()).compute().unwrap(),
                number(&model["e_cas"]) + number(&model["pt2"]),
            ),
            (
                DSRGMRPT3::new(&hf, options.clone()).compute().unwrap(),
                number(&model["e_cas"]) + number(&model["pt2"]) + number(&model["pt3"]),
            ),
            (
                MRLDSRG2::new(&hf, options).compute().unwrap(),
                number(&model["transformed"]["zero"]),
            ),
        ];
        for (result, target) in results {
            let error = (result.relaxed_total_energy - target).abs();
            energy_error = energy_error.max(error);
            assert!(error < 1e-9);
            assert_eq!(result.macro_iterations, 0);
            assert!(result.one_rdm().is_none());
            energies += 1;
        }
    }
    assert_eq!(energies, 36);
    assert_eq!(components, 486);
    println!(
        "Compatibility MR: {energies} oracle energies, max error={energy_error:e}; all {components} AO reference-density components, error={density_error:e}"
    );
}

#[test]
fn compatibility_mr_rejects_unimplemented_variants_and_invalid_controls_without_silent_fallbacks() {
    use hf_rs::post_hf::mr_dsrg::{
        DSRGMRPT2, DSRGMRPT3, MRDSRGOptions, MRLDSRG2, ThreePDC, ZerothOrderH,
    };
    let d = data();
    let (_, _, hf) = reference(&d["cases"][0]);
    for variant in 0..13 {
        let mut options = MRDSRGOptions::new(2, 2);
        match variant {
            0 => options.zeroth_order_h = ZerothOrderH::Ffull,
            1 => options.zeroth_order_h = ZerothOrderH::FdiagVactv,
            2 => options.zeroth_order_h = ZerothOrderH::FdiagVdiag,
            3 => options.three_pdc = ThreePDC::Zero,
            4 => options.three_pdc = ThreePDC::MKDecomp,
            5 => options.nivo = true,
            6 => options.sequential_transformation = true,
            7 => options.n_active_electrons = 1,
            8 => options.max_iter = 0,
            9 => options.tol = f64::NAN,
            10 => options.flow_param_s = -1.,
            11 => options.n_active_orbitals = 1000,
            _ => options.n_active_electrons = 4,
        }
        assert!(
            DSRGMRPT2::new(&hf, options.clone()).compute().is_err(),
            "variant {variant}"
        );
        assert!(
            DSRGMRPT3::new(&hf, options.clone()).compute().is_err(),
            "variant {variant}"
        );
        assert!(
            MRLDSRG2::new(&hf, options).compute().is_err(),
            "variant {variant}"
        );
    }
}

#[test]
fn factorized_cholesky_compatibility_matches_mr_for_ecp_and_rectangular_orbitals() {
    use hf_rs::post_hf::mr_dsrg::{DSRGMRPT2, MRDSRGOptions, RelaxRef, ThreeDSRGMRPT2};
    let d = data();
    let mut maximum = 0f64;
    let mut count = 0;
    for model in d["cases"].as_array().unwrap().iter().take(3) {
        let (_, _, hf) = reference(model);
        for mode in [RelaxRef::None, RelaxRef::Once, RelaxRef::Iterate] {
            let mut options = MRDSRGOptions::new(2, 2);
            options.max_iter = 150;
            options.tol = 1e-12;
            options.relax_ref = mode;
            let full = DSRGMRPT2::new(&hf, options.clone()).compute().unwrap();
            let factorized = ThreeDSRGMRPT2::new(&hf, options.clone(), 1e-12)
                .compute()
                .unwrap();
            let error = (full.relaxed_total_energy - factorized.relaxed_total_energy).abs();
            maximum = maximum.max(error);
            assert!(error < 1e-9, "{} {mode:?} {error:e}", model["name"]);
            assert_eq!(factorized.method_name(), "THREE-DSRG-MRPT2");
            assert!(factorized.one_rdm().is_none());
            assert_eq!(factorized.to_output().unwrap().e_scf, hf.total_energy);
            count += 1;
        }
    }
    let (_, _, hf) = reference(&d["cases"][0]);
    for cutoff in [f64::NAN, f64::INFINITY, -1.] {
        assert!(
            ThreeDSRGMRPT2::new(&hf, MRDSRGOptions::new(2, 2), cutoff)
                .compute()
                .is_err()
        );
    }
    let mut oversized = hf.clone();
    oversized.core_hamiltonian = Matrix::zeros(64, 64);
    assert!(
        ThreeDSRGMRPT2::new(&oversized, MRDSRGOptions::new(2, 2), 1e-12)
            .compute()
            .unwrap_err()
            .contains("dimensions/integrals")
    );
    assert_eq!(count, 9);
    println!(
        "Factorized Cholesky compatibility: {count} same-Hamiltonian energies, max error={maximum:e}"
    );
}

#[test]
fn three_index_typed_energies_and_all_relaxed_ci_components_match_independent_fock_oracles() {
    use hf_rs::post_hf::mr_dsrg::three_reference::{cholesky_factors, solve_three_reference};
    let molecular = data();
    let relaxed: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_relaxed_molecular_reference.json")).unwrap();
    let mut maximum = 0f64;
    let mut ci_maximum = 0f64;
    let mut count = 0;
    let mut ci_count = 0;
    for case in relaxed["cases"].as_array().unwrap() {
        let model = &molecular["cases"][case["model_index"].as_u64().unwrap() as usize];
        let (atoms, basis, hf) = reference(model);
        let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
        let mut options = strict();
        options.active = Some(ActiveSpace::new(1, 1, 2));
        options.cas.optimize_orbitals = model["optimize"].as_bool().unwrap_or(false);
        options.cas.orbital_gradient_tol = 1e-9;
        options.dsrg.cholesky_tolerance = 1e-12;
        options.dsrg.reference_ci_tol = 1e-10;
        options.dsrg.reference_energy_tol = 1e-12;
        let unrelaxed = create_solver(MethodId::ThreeDsrgMrPt2, &ctx, &options)
            .unwrap()
            .compute()
            .unwrap();
        let error = (unrelaxed.e_total - number(&model["e_cas"]) - number(&model["pt2"])).abs();
        assert!(error < 1e-9, "{} unrelaxed {error:e}", model["name"]);
        maximum = maximum.max(error);
        count += 1;
        let initial = if options.cas.optimize_orbitals {
            Some(solve_cas(Reference::Rhf(&hf), &options, None, true).unwrap())
        } else {
            None
        };
        let (factors, residual) = cholesky_factors(&hf, 1e-12, 256 * 1024 * 1024).unwrap();
        assert!(residual <= 1e-12);
        for expected in case["results"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| v["order"] == 2)
        {
            options.dsrg.reference_relaxation = match expected["mode"].as_str().unwrap() {
                "once" => ReferenceRelaxation::Once,
                "twice" => ReferenceRelaxation::Twice,
                _ => ReferenceRelaxation::Iterate,
            };
            let (solution, _, _) = solve_three_reference(
                &hf,
                &factors,
                &options,
                initial.as_ref().map(|cas| &cas.mo_coefficients),
                256 * 1024 * 1024,
            )
            .unwrap();
            let error = (solution.total_energy - number(&expected["energy"])).abs();
            assert!(
                error < 1e-9,
                "{} {} {error:e}",
                model["name"],
                expected["mode"]
            );
            maximum = maximum.max(error);
            let oracle_c = matrix(&expected["mo_coeff"]);
            let u = solution
                .mo_coefficients
                .transpose()
                .dot(&hf.overlap_matrix)
                .dot(&oracle_c);
            let nc = hf.n_occ - 1;
            let mut ua = Matrix::zeros(2, 2);
            for p in 0..2 {
                for q in 0..2 {
                    ua[(p, q)] = u[(nc + p, nc + q)];
                }
            }
            let rotated = fci::orbital_rotation::rotate_ci(
                &DeterminantSpace::new(2, 1, 1),
                &solution.ci_coefficients,
                &ua,
                256 * 1024 * 1024,
            )
            .unwrap();
            let target: Vec<_> = expected["coefficients"]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect();
            let phase = rotated
                .iter()
                .zip(&target)
                .map(|(x, y)| x * y)
                .sum::<f64>()
                .signum();
            for (x, y) in rotated.iter().zip(&target) {
                let error = (x - phase * y).abs();
                assert!(error < 1e-8);
                ci_maximum = ci_maximum.max(error);
                ci_count += 1;
            }
            let output = create_solver(MethodId::ThreeDsrgMrPt2, &ctx, &options)
                .unwrap()
                .compute()
                .unwrap();
            assert!((output.e_total - solution.total_energy).abs() < 1e-11);
            count += 1;
        }
    }
    assert_eq!(count, 16);
    assert_eq!(ci_count, 48);
    println!(
        "THREE independent molecular energies: {count}, max {maximum:e}; all {ci_count} CI components: {ci_maximum:e}"
    );
}

#[test]
fn three_index_every_nuclear_component_matches_independent_all_electron_and_nonlocal_ecp_oracles() {
    let mut unrelaxed: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_gradient_reference.json")).unwrap();
    for case in unrelaxed["cases"].as_array_mut().unwrap() {
        case["gradient"] = serde_json::json!([case["gradient"][0]]);
    }
    verify_every_gradient_component(
        &unrelaxed,
        &[(MethodId::ThreeDsrgMrPt2, ReferenceRelaxation::None)],
    );
    let mut relaxed: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_relaxed_gradient_reference.json")).unwrap();
    for case in relaxed["cases"].as_array_mut().unwrap() {
        case["gradient"] = serde_json::json!([
            case["gradient"][0],
            case["gradient"][1],
            case["gradient"][2]
        ]);
    }
    let variants = [
        ReferenceRelaxation::Once,
        ReferenceRelaxation::Twice,
        ReferenceRelaxation::Iterate,
    ]
    .map(|mode| (MethodId::ThreeDsrgMrPt2, mode));
    verify_every_gradient_component(&relaxed, &variants);
}

#[test]
fn three_index_density_fitted_energy_integrals_and_ci_match_independent_pyscf_fock_oracles() {
    use hf_rs::post_hf::mr_dsrg::three_reference::solve_three_reference;
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/three_df_reference.json")).unwrap();
    let molecular = data();
    let mut energy_error = 0f64;
    let mut integral_error = 0f64;
    let mut ci_error = 0f64;
    let mut counts = [0; 3];
    for case in oracle["cases"].as_array().unwrap() {
        let model = &molecular["cases"][case["model_index"].as_u64().unwrap() as usize];
        let (atoms, basis, hf) = reference(model);
        let auxiliary = context::AuxBasis::from_builtin(&atoms, "def2-svp-ri").unwrap();
        let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, Some(&auxiliary)).unwrap();
        let factors = ctx.ints.df_ao().unwrap();
        assert_eq!(factors.rank(), case["rank"].as_u64().unwrap() as usize);
        let n = basis.len();
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        let error = (factors.eri(p, q, r, s)
                            - number(&case["eri"][((p * n + q) * n + r) * n + s]))
                        .abs();
                        assert!(error < 1e-9, "DF integral {p}/{q}/{r}/{s} {error:e}");
                        integral_error = integral_error.max(error);
                        counts[1] += 1;
                    }
                }
            }
        }
        let mut options = strict();
        options.active = Some(ActiveSpace::new(1, 1, 2));
        options.density_fitting = true;
        options.dsrg.reference_ci_tol = 1e-10;
        options.dsrg.reference_energy_tol = 1e-12;
        let output = create_solver(MethodId::ThreeDsrgMrPt2, &ctx, &options)
            .unwrap()
            .compute()
            .unwrap();
        let error = (output.e_total - number(&case["e_cas"]) - number(&case["pt2"])).abs();
        assert!(error < 1e-9);
        energy_error = energy_error.max(error);
        counts[0] += 1;
        for (i, mode) in [
            ReferenceRelaxation::Once,
            ReferenceRelaxation::Twice,
            ReferenceRelaxation::Iterate,
        ]
        .into_iter()
        .enumerate()
        {
            options.dsrg.reference_relaxation = mode;
            let (solution, _, _) =
                solve_three_reference(&hf, factors, &options, None, 256 * 1024 * 1024).unwrap();
            let expected = &case["results"][i];
            let error = (solution.total_energy - number(&expected["energy"])).abs();
            assert!(error < 1e-9);
            energy_error = energy_error.max(error);
            counts[0] += 1;
            let u = solution
                .mo_coefficients
                .transpose()
                .dot(&hf.overlap_matrix)
                .dot(&matrix(&expected["mo_coeff"]));
            let nc = hf.n_occ - 1;
            let mut ua = Matrix::zeros(2, 2);
            for p in 0..2 {
                for q in 0..2 {
                    ua[(p, q)] = u[(nc + p, nc + q)];
                }
            }
            let rotated = fci::orbital_rotation::rotate_ci(
                &DeterminantSpace::new(2, 1, 1),
                &solution.ci_coefficients,
                &ua,
                256 * 1024 * 1024,
            )
            .unwrap();
            let target: Vec<_> = expected["coefficients"]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect();
            let phase = rotated
                .iter()
                .zip(&target)
                .map(|(x, y)| x * y)
                .sum::<f64>()
                .signum();
            for (x, y) in rotated.iter().zip(target) {
                let error = (x - phase * y).abs();
                assert!(error < 1e-8);
                ci_error = ci_error.max(error);
                counts[2] += 1;
            }
            let output = create_solver(MethodId::ThreeDsrgMrPt2, &ctx, &options)
                .unwrap()
                .compute()
                .unwrap();
            assert!((output.e_total - solution.total_energy).abs() < 1e-11);
        }
    }
    assert_eq!(counts, [8, 162, 24]);
    println!(
        "THREE DF: {} energies error {energy_error:e}, {} integrals error {integral_error:e}, {} CI components error {ci_error:e}",
        counts[0], counts[1], counts[2]
    );
}

#[test]
fn three_index_df_every_nuclear_component_includes_moving_auxiliary_metric_and_nonlocal_ecp() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/three_df_gradient_reference.json")).unwrap();
    let variants = [
        ReferenceRelaxation::None,
        ReferenceRelaxation::Once,
        ReferenceRelaxation::Twice,
        ReferenceRelaxation::Iterate,
    ]
    .map(|mode| (MethodId::ThreeDsrgMrPt2, mode));
    verify_every_gradient_with_fitting(&oracle, &variants, true);
}

#[test]
fn three_index_empty_and_full_cas_limits_and_invalid_controls_are_explicit() {
    let model = &data()["cases"][0];
    let (atoms, basis, hf) = reference(model);
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let mut options = strict();
    options.dsrg.cholesky_tolerance = 1e-12;
    let three = create_solver(MethodId::ThreeDsrgMrPt2, &ctx, &options)
        .unwrap()
        .compute()
        .unwrap();
    let sr = create_solver(MethodId::SrDsrgPt2, &ctx, &options)
        .unwrap()
        .compute()
        .unwrap();
    assert!((three.e_total - sr.e_total).abs() < 1e-10);
    options.active = Some(ActiveSpace::new(1, 1, 3));
    options.dsrg.reference_relaxation = ReferenceRelaxation::Iterate;
    let three = create_solver(MethodId::ThreeDsrgMrPt2, &ctx, &options)
        .unwrap()
        .compute()
        .unwrap();
    let cas = solve_cas(Reference::Rhf(&hf), &options, None, false).unwrap();
    assert!((three.e_total - cas.output.e_total).abs() < 1e-10);
    assert!(three.components["E_PT2"].abs() < 1e-12);
    options.active = Some(ActiveSpace::new(1, 1, 2));
    options.dsrg.max_reference_iterations = 1;
    assert!(
        create_solver(MethodId::ThreeDsrgMrPt2, &ctx, &options)
            .unwrap()
            .compute()
            .is_err()
    );
    for threshold in [f64::NAN, f64::INFINITY, -1.] {
        options.dsrg.cholesky_tolerance = threshold;
        assert!(create_solver(MethodId::ThreeDsrgMrPt2, &ctx, &options).is_err());
    }
    let values =
        std::collections::BTreeMap::from([("cholesky_tolerance".into(), serde_json::json!(1e-12))]);
    let parsed =
        PostHFOptions::from_map(&values, &MethodId::ThreeDsrgMrPt2.option_schemas()).unwrap();
    assert_eq!(parsed.dsrg.cholesky_tolerance, 1e-12);
    assert!(PostHFOptions::from_map(&values, &MethodId::DsrgMrPt2.option_schemas()).is_err());
    let mut large = hf.clone();
    large.mo_coefficients = Matrix::zeros(basis.len(), 80);
    large.orbital_energies = vec![0.; 80];
    let large_context = PostHFContext::from_rhf(&large, &atoms, &basis, None).unwrap();
    let mut optimized = strict();
    optimized.active = Some(ActiveSpace::new(1, 1, 2));
    optimized.cas.optimize_orbitals = true;
    let error = create_solver(MethodId::ThreeDsrgMrPt2, &large_context, &optimized)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("starting-orbital workspace exceeds memory budget"));
}

#[test]
fn three_index_cholesky_residual_bounds_every_ao_integral_and_budget_precedes_allocation() {
    use hf_rs::post_hf::mr_dsrg::three_reference::cholesky_factors;
    let (_, basis, hf) = reference(&data()["cases"][0]);
    let mut previous_rank = 0;
    for threshold in [1e-1, 1e-3, 1e-6, 1e-12, 0.] {
        let (b, residual) = cholesky_factors(&hf, threshold, 1024 * 1024).unwrap();
        assert!(b.rank() >= previous_rank);
        assert!(residual <= threshold);
        previous_rank = b.rank();
        let n = basis.len();
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        let error = (b.eri(p, q, r, s) - hf.eri[scf::eri_index(p, q, r, s)]).abs();
                        assert!(error <= residual + 1e-12);
                    }
                }
            }
        }
    }
    assert!(
        cholesky_factors(&hf, 1e-12, 1024)
            .err()
            .unwrap()
            .to_string()
            .contains("memory budget")
    );
}

#[test]
fn multistate_single_state_full_cas_limits_and_option_failures_are_explicit() {
    let model = &data()["cases"][0];
    let (atoms, basis, hf) = reference(model);
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let mut opts = strict();
    opts.dsrg.cholesky_tolerance = 1e-12;
    let sr = create_solver(MethodId::SrDsrgPt2, &ctx, &opts)
        .unwrap()
        .compute()
        .unwrap();
    for m in [
        MethodId::SaDsrgPt2,
        MethodId::SaDsrgPt2C,
        MethodId::MsDsrgPt2,
        MethodId::XmsDsrgPt2,
        MethodId::DwmsDsrgPt2,
    ] {
        let empty = create_solver(m, &ctx, &opts).unwrap().compute().unwrap();
        assert!((empty.e_total - sr.e_total).abs() < 1e-10);
    }
    opts.active = Some(ActiveSpace::new(1, 1, 2));
    let reference = create_solver(MethodId::DsrgMrPt2, &ctx, &opts)
        .unwrap()
        .compute()
        .unwrap();
    for m in [
        MethodId::SaDsrgPt2C,
        MethodId::MsDsrgPt2,
        MethodId::XmsDsrgPt2,
        MethodId::DwmsDsrgPt2,
    ] {
        let output = create_solver(m, &ctx, &opts).unwrap().compute().unwrap();
        assert!((output.e_total - reference.e_total).abs() < 1e-10);
    }
    opts.dsrg.reference_relaxation = ReferenceRelaxation::Once;
    let relaxed = create_solver(MethodId::DsrgMrPt2, &ctx, &opts)
        .unwrap()
        .compute()
        .unwrap();
    opts.dsrg.reference_relaxation = ReferenceRelaxation::None;
    assert!(
        (create_solver(MethodId::SaDsrgPt2, &ctx, &opts)
            .unwrap()
            .compute()
            .unwrap()
            .e_total
            - relaxed.e_total)
            .abs()
            < 1e-10
    );
    opts.n_roots = 2;
    opts.active = Some(ActiveSpace::new(1, 1, 3));
    let cas = solve_cas(Reference::Rhf(&hf), &opts, None, false).unwrap();
    for m in [
        MethodId::SaDsrgPt2,
        MethodId::SaDsrgPt2C,
        MethodId::MsDsrgPt2,
        MethodId::XmsDsrgPt2,
        MethodId::DwmsDsrgPt2,
    ] {
        let output = create_solver(m, &ctx, &opts).unwrap().compute().unwrap();
        for i in 0..2 {
            assert!((output.states[i].energy - cas.output.states[i].energy).abs() < 1e-10);
        }
    }
    opts.active = Some(ActiveSpace::new(1, 1, 2));
    opts.cas.state_weights = vec![0.3, 0.7];
    assert!(create_solver(MethodId::XmsDsrgPt2, &ctx, &opts).is_err());
    assert_eq!(
        MethodId::XmsDsrgPt2.gradient_kind(context::RefKind::Rhf, &opts),
        registry::GradientKind::Unsupported
    );
    let weighted = create_solver(MethodId::SaDsrgPt2C, &ctx, &opts)
        .unwrap()
        .compute()
        .unwrap();
    assert!(weighted.e_total.is_finite());
    opts.cas.state_weights.clear();
    opts.dsrg.reference_relaxation = ReferenceRelaxation::Iterate;
    assert!(create_solver(MethodId::MsDsrgPt2, &ctx, &opts).is_err());
    opts.dsrg.reference_relaxation = ReferenceRelaxation::None;
    opts.frozen_core = 1;
    assert!(create_solver(MethodId::MsDsrgPt2, &ctx, &opts).is_err());
    let values = std::collections::BTreeMap::from([(
        "dynamic_weight_parameter".into(),
        serde_json::json!(25.),
    )]);
    assert_eq!(
        PostHFOptions::from_map(&values, &MethodId::DwmsDsrgPt2.option_schemas())
            .unwrap()
            .dsrg
            .dynamic_weight_parameter,
        25.
    );
    assert!(PostHFOptions::from_map(&values, &MethodId::SaDsrgPt2.option_schemas()).is_err());
    assert_eq!(
        "SA-DSRG-PT2c".parse::<MethodId>().unwrap(),
        MethodId::SaDsrgPt2C
    );
}

fn multistate_id(name: &str) -> (MethodId, f64) {
    match name {
        "sa" => (MethodId::SaDsrgPt2, 50.),
        "sac" => (MethodId::SaDsrgPt2C, 50.),
        "ms" => (MethodId::MsDsrgPt2, 50.),
        "xms" => (MethodId::XmsDsrgPt2, 50.),
        "dw0" => (MethodId::DwmsDsrgPt2, 0.),
        "dwlarge" => (MethodId::DwmsDsrgPt2, 1e14),
        _ => (MethodId::DwmsDsrgPt2, 50.),
    }
}

#[test]
fn forte_executable_only_single_and_multistate_energies_match_all_roots() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/forte_dsrg_reference.json")).unwrap();
    let molecular = data();
    let mut count = 0;
    let mut maximum = 0f64;
    for case in oracle["cases"].as_array().unwrap() {
        let index = case["model_index"].as_u64().unwrap() as usize;
        let (atoms, basis, hf) = reference(&molecular["cases"][index]);
        let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
        let mut options = strict();
        options.active = Some(ActiveSpace::new(1, 1, 2));
        options.n_roots = case["n_roots"].as_u64().unwrap() as usize;
        options.root = options.n_roots - 1;
        options.dsrg.cholesky_tolerance = 1e-12;
        options.dsrg.reference_ci_tol = 1e-10;
        options.dsrg.reference_energy_tol = 1e-12;
        options.dsrg.reference_relaxation = match case["relaxation"].as_str().unwrap_or("none") {
            "none" => ReferenceRelaxation::None,
            "once" => ReferenceRelaxation::Once,
            "twice" => ReferenceRelaxation::Twice,
            "iterate" => ReferenceRelaxation::Iterate,
            _ => panic!("invalid oracle relaxation selector"),
        };
        let method = case["method"]
            .as_str()
            .unwrap()
            .parse::<MethodId>()
            .unwrap();
        let actual = create_solver(method, &ctx, &options)
            .unwrap()
            .compute()
            .unwrap();
        assert!((actual.e_scf - number(&case["e_scf"])).abs() < 1e-10);
        assert_eq!(actual.states.len(), options.n_roots);
        assert!(
            (actual.components["E_CASCI"] - number(&case["cas_energies"][options.root])).abs()
                < 1e-9
        );
        for (i, state) in actual.states.iter().enumerate() {
            let error = (state.energy - number(&case["energies"][i])).abs();
            maximum = maximum.max(error);
            assert!(error < 1e-9, "Forte {index} {method} root {i}: {error:e}");
            count += 1;
        }
        assert!((actual.e_total - number(&case["energies"][options.root])).abs() < 1e-9);
    }
    assert_eq!(count, 46);
    println!("Forte executable oracle: all {count} energies, maximum error {maximum:e}");
}

#[test]
fn multistate_typed_molecular_all_states_and_selected_roots_match_independent_fock_oracles() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/multistate_reference.json")).unwrap();
    assert_eq!(oracle["cases"].as_array().unwrap().len(), 7);
    let molecular = data();
    let mut count = 0;
    let mut maximum = 0f64;
    for case in oracle["cases"].as_array().unwrap() {
        let index = case["model_index"].as_u64().unwrap() as usize;
        let (atoms, basis, hf) = reference(&molecular["cases"][index]);
        let fitted = case["fitted"].as_bool().unwrap();
        let aux = if fitted {
            Some(context::AuxBasis::from_builtin(&atoms, "def2-svp-ri").unwrap())
        } else {
            None
        };
        let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, aux.as_ref()).unwrap();
        let mut options = strict();
        options.active = Some(ActiveSpace::new(1, 1, 2));
        options.n_roots = case["roots"].as_u64().unwrap() as usize;
        options.density_fitting = fitted;
        options.cas.optimize_orbitals = index == 3;
        options.cas.orbital_gradient_tol = 1e-9;
        options.cas.max_orbital_iterations = 150;
        options.dsrg.cholesky_tolerance = 1e-12;
        for expected in case["results"].as_array().unwrap() {
            let (method, zeta) = multistate_id(expected["variant"].as_str().unwrap());
            options.dsrg.dynamic_weight_parameter = zeta;
            options.dsrg.flow_parameter = number(&expected["flow"]);
            options.root = options.n_roots - 1;
            let output = create_solver(method, &ctx, &options)
                .unwrap()
                .compute()
                .unwrap();
            assert_eq!(output.root, options.root);
            assert_eq!(output.states.len(), options.n_roots);
            for (i, state) in output.states.iter().enumerate() {
                let error = (state.energy - number(&expected["energies"][i])).abs();
                maximum = maximum.max(error);
                assert!(
                    error < 1e-8,
                    "model {index}, fitted={fitted}, {}, root {i}, error={error:e}",
                    expected["variant"]
                );
                count += 1;
                assert!(state.s_squared.unwrap().abs() < 1e-12);
            }
            assert!((output.e_total - number(&expected["energies"][options.root])).abs() < 1e-8);
            assert!(
                (output.components["E_CASCI"] - number(&case["cas_energies"][options.root])).abs()
                    < 1e-9
            );
            assert!((output.e_total - output.e_corr - hf.total_energy).abs() < 1e-12);
            assert!(output.one_rdm.is_none());
            assert_eq!(
                method.gradient_kind(context::RefKind::Rhf, &options),
                registry::GradientKind::Numerical
            );
        }
    }
    assert_eq!(count, 210);
    println!("Multistate typed: all {count} energies/root selections, maximum error {maximum:e}");
}

#[test]
fn multistate_legacy_adapters_use_the_same_singlet_factorized_engine_and_scf_output() {
    use post_hf::mr_dsrg::{MRDSRGOptions, MultiStateDSRG, MultiStateDSRGVariant as V};
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/multistate_reference.json")).unwrap();
    let molecular = data();
    let mut count = 0;
    let mut maximum = 0f64;
    for case in oracle["cases"].as_array().unwrap().iter().take(3) {
        let index = case["model_index"].as_u64().unwrap() as usize;
        let (_, _, hf) = reference(&molecular["cases"][index]);
        let mut options = MRDSRGOptions::new(2, 2);
        options.tol = 1e-12;
        options.max_iter = 150;
        for expected in case["results"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| number(&r["flow"]) == 0.5)
        {
            let variant = match expected["variant"].as_str().unwrap() {
                "sa" => V::StateAverage,
                "sac" => V::StateAverageContracted,
                "ms" => V::MS,
                "xms" => V::XMS,
                "dw0" => V::DWMS { zeta: 0. },
                "dw50" => V::DWMS { zeta: 50. },
                _ => V::DWMS { zeta: 1e14 },
            };
            let result = MultiStateDSRG::new(&hf, options.clone(), 2, variant)
                .compute()
                .unwrap();
            for i in 0..2 {
                let error = (result.state_energies[i] - number(&expected["energies"][i])).abs();
                maximum = maximum.max(error);
                assert!(error < 1e-9);
                count += 1;
            }
            assert_eq!(result.to_output().unwrap().e_scf, hf.total_energy);
        }
    }
    assert_eq!(count, 42);
    println!("Multistate legacy: {count} energies, max {maximum:e}");
}

#[test]
fn multistate_every_method_state_and_nuclear_component_matches_independent_ecp_and_df_oracles() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/multistate_gradient_reference.json")).unwrap();
    assert_eq!(oracle["cases"].as_array().unwrap().len(), 5);
    let molecular = data();
    let mut count = 0;
    let mut maximum = 0f64;
    let mut api_error = 0f64;
    let mut translation = 0f64;
    let methods = [
        MethodId::SaDsrgPt2,
        MethodId::SaDsrgPt2C,
        MethodId::MsDsrgPt2,
        MethodId::XmsDsrgPt2,
        MethodId::DwmsDsrgPt2,
    ];
    for case in oracle["cases"].as_array().unwrap() {
        let index = case["model_index"].as_u64().unwrap() as usize;
        let model = &molecular["cases"][index];
        let (atoms, basis) = setup_model(model);
        let fitted = case["fitted"].as_bool().unwrap();
        let n = atoms.len() * 3;
        let step = number(&case["step"]);
        let basis_map = if let Some(gbs) = model["gaussian94"].as_str() {
            parse_gaussian94(gbs)
        } else {
            get_builtin_basis("sto-3g").unwrap()
        };
        let mut options = strict();
        options.active = Some(ActiveSpace::new(1, 1, 2));
        options.n_roots = 2;
        options.density_fitting = fitted;
        options.cas.optimize_orbitals = index == 3;
        options.cas.orbital_gradient_tol = 1e-13;
        options.cas.max_orbital_iterations = 400;
        if index == 3 {
            options.residual_tol = 1e-14;
        }
        options.dsrg.cholesky_tolerance = 1e-12;
        let mut gradients = vec![vec![vec![0.; n]; 2]; methods.len()];
        for component in 0..n {
            let mut values = vec![vec![vec![0.; 2]; methods.len()]; 4];
            for (j, offset) in [-2., -1., 1., 2.].iter().enumerate() {
                let mut moved = atoms.clone();
                moved[component / 3].position[component % 3] += offset * step;
                let moved_basis = build_basis(&moved, &basis_map);
                let hf = scf::run_scf_in_space_with_convergence(
                    &moved,
                    &moved_basis,
                    model["charge"].as_i64().unwrap() as i32,
                    300,
                    scf::RhfConvergence {
                        energy: 1e-14,
                        residual: 1e-13,
                        density: 1e-13,
                    },
                    ConvAlgo::Cdiis,
                    None,
                    ao::BasisKind::Cartesian,
                )
                .unwrap();
                let aux = if fitted {
                    Some(context::AuxBasis::from_builtin(&moved, "def2-svp-ri").unwrap())
                } else {
                    None
                };
                let ctx = PostHFContext::from_rhf(&hf, &moved, &moved_basis, aux.as_ref()).unwrap();
                for (k, &method) in methods.iter().enumerate() {
                    let output = create_solver(method, &ctx, &options)
                        .unwrap()
                        .compute()
                        .unwrap();
                    for (root, value) in values[j][k].iter_mut().enumerate() {
                        *value = output.states[root].energy;
                    }
                }
            }
            for k in 0..methods.len() {
                for root in 0..2 {
                    gradients[k][root][component] = (values[0][k][root] - 8. * values[1][k][root]
                        + 8. * values[2][k][root]
                        - values[3][k][root])
                        / (12. * step);
                }
            }
        }
        for (k, &method) in methods.iter().enumerate() {
            for root in 0..2 {
                for component in 0..n {
                    let error = (gradients[k][root][component]
                        - number(&case["gradient"][k][root][component]))
                    .abs();
                    maximum = maximum.max(error);
                    assert!(
                        error < 1e-6,
                        "{index} DF={fitted} {method} root {root} coordinate {component}: {error:e}"
                    );
                    count += 1;
                }
                for axis in 0..3 {
                    let sum = (0..atoms.len())
                        .map(|a| gradients[k][root][3 * a + axis])
                        .sum::<f64>()
                        .abs();
                    translation = translation.max(sum);
                    assert!(
                        sum < 1e-9,
                        "{index} DF={fitted} {method} root {root} axis {axis} translation={sum:e}"
                    );
                }
                options.root = root;
                let aux = if fitted {
                    Some(context::AuxBasis::from_builtin(&atoms, "def2-svp-ri").unwrap())
                } else {
                    None
                };
                let actual = post_hf::mr_dsrg::typed_mr::numerical_gradient_with_aux(
                    method,
                    &atoms,
                    &basis,
                    aux.as_ref(),
                    model["charge"].as_i64().unwrap() as i32,
                    ao::BasisKind::Cartesian,
                    &options,
                    &post_hf::numerical::NumericalGradientOptions {
                        step_bohr: 5e-4,
                        scf_max_iter: 300,
                        scf_tolerance: 1e-14,
                        scf_residual_tolerance: 1e-13,
                        scf_density_tolerance: 1e-13,
                    },
                )
                .unwrap();
                assert_eq!(actual.kind, registry::GradientKind::Numerical);
                for component in 0..n {
                    let error = (actual.gradient[component / 3][component % 3]
                        - number(&case["gradient"][k][root][component]))
                    .abs();
                    api_error = api_error.max(error);
                    assert!(
                        error < 1e-6,
                        "API {index} DF={fitted} {method} root {root} coordinate {component}: {error:e}"
                    );
                }
            }
        }
    }
    assert_eq!(count, 390);
    println!(
        "Multistate nuclear: all {count} components, oracle={maximum:e}, API={api_error:e}, translation={translation:e}"
    );
}

#[test]
fn qc_molecular_energies_all_reference_coefficients_and_compatibility_match_independent_oracles() {
    use hf_rs::{
        fci::orbital_rotation::rotate_ci,
        options::ReferenceRelaxation,
        post_hf::mr_dsrg::{MRDSRGOptions, MRLDSRG2, RelaxRef, relaxation::solve_reference},
    };
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_qc_reference.json")).unwrap();
    let models = data();
    let mut maximum = 0f64;
    let mut ci_maximum = 0f64;
    let mut count = 0;
    let mut compatibility_maximum = 0f64;
    let mut compatibility_count = 0;
    for case in oracle["molecules"].as_array().unwrap() {
        let model = &models["cases"][case["model_index"].as_u64().unwrap() as usize];
        let (atoms, basis, hf) = reference(model);
        let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
        for expected in case["results"].as_array().unwrap() {
            let mut options = strict();
            options.active = Some(ActiveSpace::new(1, 1, 2));
            options.cas.optimize_orbitals = model["optimize"].as_bool().unwrap_or(false);
            options.cas.orbital_gradient_tol = 1e-9;
            options.dsrg.reference_ci_tol = 1e-10;
            options.dsrg.reference_energy_tol = 1e-12;
            options.dsrg.reference_relaxation = match expected["mode"].as_str().unwrap() {
                "none" => ReferenceRelaxation::None,
                "once" => ReferenceRelaxation::Once,
                "twice" => ReferenceRelaxation::Twice,
                _ => ReferenceRelaxation::Iterate,
            };
            let output = create_solver(MethodId::MrLdsrg2Qc, &ctx, &options)
                .unwrap()
                .compute()
                .unwrap();
            let error = (output.e_total - number(&expected["energy"])).abs();
            maximum = maximum.max(error);
            assert!(
                error < 1e-9,
                "{} {}: {error:e}",
                case["name"],
                expected["mode"]
            );
            assert_eq!(output.method, MethodId::MrLdsrg2Qc.name());
            assert_eq!(output.components["bch_order"], 2.);
            assert!(output.components["source_residual_max"] < 1e-12);
            assert!(output.one_rdm.is_none());
            let cas = solve_cas(
                Reference::Rhf(&hf),
                &options,
                None,
                options.cas.optimize_orbitals,
            )
            .unwrap();
            let nc = cas.active_space.inactive.len();
            let solution = solve_reference(&hf, cas, 5, &options, 256 * 1024 * 1024).unwrap();
            let u = solution
                .mo_coefficients
                .transpose()
                .dot(&hf.overlap_matrix)
                .dot(&matrix(&expected["mo_coeff"]));
            let ua = Matrix::from_vec(
                2,
                2,
                (0..2)
                    .flat_map(|p| (0..2).map(move |q| (p, q)))
                    .map(|(p, q)| u[(nc + p, nc + q)])
                    .collect(),
            );
            let ci = rotate_ci(
                &DeterminantSpace::new(2, 1, 1),
                &solution.ci_coefficients,
                &ua,
                256 * 1024 * 1024,
            )
            .unwrap();
            let target: Vec<_> = expected["coefficients"]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .collect();
            let sign = ci
                .iter()
                .zip(&target)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .signum();
            for (a, b) in ci.iter().zip(target) {
                let e = (a - sign * b).abs();
                ci_maximum = ci_maximum.max(e);
                assert!(e < 1e-8);
            }
            if !options.cas.optimize_orbitals {
                let mut legacy = MRDSRGOptions::new(2, 2);
                legacy.max_iter = 150;
                legacy.tol = 1e-12;
                legacy.relax_ref = match options.dsrg.reference_relaxation {
                    ReferenceRelaxation::None => RelaxRef::None,
                    ReferenceRelaxation::Once => RelaxRef::Once,
                    ReferenceRelaxation::Twice => RelaxRef::Twice,
                    _ => RelaxRef::Iterate,
                };
                let legacy = MRLDSRG2::new_qc(&hf, legacy).compute().unwrap();
                let error = (legacy.total_energy() - number(&expected["energy"])).abs();
                compatibility_maximum = compatibility_maximum.max(error);
                assert!(error < 1e-9);
                assert_eq!(legacy.method_name(), MethodId::MrLdsrg2Qc.name());
                assert_eq!(legacy.to_output().unwrap().e_scf, hf.total_energy);
                assert!(legacy.one_rdm().is_none());
                compatibility_count += 1;
            }
            count += 1;
        }
    }
    assert_eq!(count, 16);
    assert_eq!(compatibility_count, 12);
    println!(
        "QC molecular: {count} energies error={maximum:e}, all 64 CI coefficients error={ci_maximum:e}; compatibility {compatibility_count} energies error={compatibility_maximum:e}"
    );
    let values = std::collections::BTreeMap::from([("bch_max_order".into(), serde_json::json!(2))]);
    assert!(PostHFOptions::from_map(&values, &MethodId::MrLdsrg2Qc.option_schemas()).is_err());
    let mut bad = strict();
    bad.density_fitting = true;
    assert_eq!(
        MethodId::MrLdsrg2Qc.gradient_kind(context::RefKind::Rhf, &bad),
        registry::GradientKind::Unsupported
    );
}

#[test]
fn every_qc_nuclear_coordinate_matches_independent_ecp_and_all_electron_energy_stencils() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_qc_gradient_reference.json")).unwrap();
    let variants = [
        ReferenceRelaxation::Once,
        ReferenceRelaxation::Twice,
        ReferenceRelaxation::Iterate,
    ]
    .map(|mode| (MethodId::MrLdsrg2Qc, mode));
    let (five_point, api, translation, count) = verify_every_gradient_component(&oracle, &variants);
    assert_eq!(count, 45);
    println!(
        "QC nuclear derivatives: all {count} components, five-point error={five_point:e}, API error={api:e}, translation={translation:e}"
    );
}
