// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    context::PostHFContext,
    options::PostHFOptions,
    post_hf::{
        PostHFMethod,
        mr_dsrg::{
            dsrg::{SRDSRGPT2, SRDSRGPT3, SRLDSRG2, UHFSRDSRGPT2},
            imsrg2::No2bOperator,
            sr_dsrg::{LdsrgOptions, bch_no2b, dsrg_pt3, solve_ldsrg2},
        },
    },
    registry::{MethodId, create_solver},
    *,
};
fn number(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap()
}
fn vector(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(number).collect()
}
fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter().flat_map(vector).collect(),
    )
}
fn operator(v: &serde_json::Value) -> No2bOperator {
    No2bOperator {
        zero: number(&v["zero"]),
        one: matrix(&v["one"]),
        two: vector(&v["two"]),
    }
}
fn occupied(v: &serde_json::Value) -> Vec<bool> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect()
}
fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/sr_dsrg_reference.json")).unwrap()
}
fn close(x: f64, y: f64, tol: f64) {
    assert!(
        (x - y).abs() < tol,
        "{x:.15e} vs {y:.15e}, difference {:e}",
        x - y
    );
}
fn compare(a: &No2bOperator, b: &No2bOperator, tol: f64) -> f64 {
    let mut max = 0f64;
    assert_eq!(a.one.rows, b.one.rows);
    assert_eq!(a.two.len(), b.two.len());
    for (i, (&x, &y)) in std::iter::once(&a.zero)
        .chain(&a.one.data)
        .chain(&a.two)
        .zip(std::iter::once(&b.zero).chain(&b.one.data).chain(&b.two))
        .enumerate()
    {
        assert!(
            (x - y).abs() < tol,
            "operator component {i}: {x:e} vs {y:e}, difference {:e}",
            x - y
        );
        max = max.max((x - y).abs());
    }
    max
}
fn strict() -> LdsrgOptions {
    LdsrgOptions {
        residual_tolerance: 1e-12,
        energy_tolerance: 1e-14,
        bch_tolerance: 1e-14,
        ..Default::default()
    }
}
fn rotate(a: &No2bOperator, spatial: &Matrix) -> No2bOperator {
    let n = a.one.rows;
    let mut u = Matrix::zeros(n, n);
    for p in 0..n {
        for q in 0..n {
            if p % 2 == q % 2 {
                u[(p, q)] = spatial[(p / 2, q / 2)];
            }
        }
    }
    let pairs: Vec<_> = (0..n)
        .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
        .collect();
    let mut w = Matrix::zeros(pairs.len(), pairs.len());
    let mut g = Matrix::zeros(pairs.len(), pairs.len());
    for (pq, &(p, q)) in pairs.iter().enumerate() {
        for (rs, &(r, s)) in pairs.iter().enumerate() {
            w[(pq, rs)] = u[(p, r)] * u[(q, s)] - u[(p, s)] * u[(q, r)];
            g[(pq, rs)] = a.two[((p * n + q) * n + r) * n + s];
        }
    }
    let g = w.transpose().dot(&g).dot(&w);
    let mut two = vec![0.; n.pow(4)];
    for (pq, &(p, q)) in pairs.iter().enumerate() {
        for (rs, &(r, s)) in pairs.iter().enumerate() {
            let v = g[(pq, rs)];
            for (a, b, c, d, sign) in [
                (p, q, r, s, 1.),
                (q, p, r, s, -1.),
                (p, q, s, r, -1.),
                (q, p, s, r, 1.),
            ] {
                two[((a * n + b) * n + c) * n + d] = sign * v;
            }
        }
    }
    No2bOperator {
        zero: a.zero,
        one: u.transpose().dot(&a.one).dot(&u),
        two,
    }
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
fn every_bch_component_matches_pade_exponential_of_fock_projection_superoperator() {
    let data = data();
    let case = &data["bch"];
    let h = operator(&case["h"]);
    let a = operator(&case["a"]);
    let occ = occupied(&case["occupied"]);
    let (actual, order) = bch_no2b(&h, &a, &occ, 1e-14, 80).unwrap();
    let error = compare(&actual, &operator(&case["bar"]), 1e-11);
    assert!(order > 10);
    assert!(bch_no2b(&h, &a, &occ, 1e-14, 1).is_err());
    assert!(bch_no2b(&h, &a, &occ, f64::NAN, 80).is_err());
    println!("BCH: all 1333 coefficients, max error {error:e}, {order} commutators");
}

#[test]
fn all_generic_driven_and_perturbative_amplitudes_match_independent_nonlinear_oracle() {
    let data = data();
    let case = &data["generic"];
    let h = operator(&case["initial"]);
    let occ = occupied(&case["occupied"]);
    let flow = number(&case["flow"]);
    let actual = solve_ldsrg2(&h, &occ, flow, strict()).unwrap();
    let error = compare(&actual.transformed, &operator(&case["transformed"]), 2e-11).max(compare(
        &actual.generator,
        &operator(&case["generator"]),
        2e-11,
    ));
    assert!(actual.residual_norm < 1e-12);
    let pt = dsrg_pt3(&h, &occ, flow).unwrap();
    close(pt.second_order, number(&case["second_order"]), 1e-10);
    close(pt.third_order, number(&case["third_order"]), 1e-10);
    let pt_error = compare(
        &pt.first_order_generator,
        &operator(&case["first_order_generator"]),
        2e-10,
    )
    .max(compare(
        &pt.second_order_generator,
        &operator(&case["second_order_generator"]),
        2e-10,
    ));
    assert!(
        pt.first_order_generator
            .one
            .data
            .iter()
            .any(|v| v.abs() > 1e-3)
    );
    assert!(actual.generator.one.data.iter().any(|v| v.abs() > 1e-3));
    let zero = solve_ldsrg2(&h, &occ, 0., strict()).unwrap();
    compare(&zero.transformed, &h, 1e-15);
    assert!(
        zero.generator
            .one
            .data
            .iter()
            .chain(&zero.generator.two)
            .all(|&v| v == 0.)
    );
    let zero_pt = dsrg_pt3(&h, &occ, 0.).unwrap();
    assert_eq!(zero_pt.second_order, 0.);
    assert_eq!(zero_pt.third_order, 0.);
    assert!(
        solve_ldsrg2(
            &h,
            &occ,
            flow,
            LdsrgOptions {
                max_iter: 1,
                ..strict()
            }
        )
        .is_err()
    );
    for bad in [-0.1, f64::NAN, f64::INFINITY] {
        assert!(solve_ldsrg2(&h, &occ, bad, strict()).is_err());
        assert!(dsrg_pt3(&h, &occ, bad).is_err());
    }
    println!(
        "Generic model: all 5332 amplitude/Hbar coefficients, error {error:e}, PT expansion {pt_error:e}, iterations {}",
        actual.iterations
    );
}

#[test]
fn molecular_energies_all_amplitudes_and_typed_solvers_match_pyscf_fock_oracle() {
    let data = data();
    for model in data["molecules"].as_array().unwrap() {
        let (atoms, basis) = setup_model(model);
        let hf = run_scf_with_charge(
            &atoms,
            &basis,
            model["charge"].as_i64().unwrap() as i32,
            200,
            1e-14,
        )
        .unwrap();
        close(hf.total_energy, number(&model["e_scf"]), 1e-10);
        let u = hf
            .mo_coefficients
            .transpose()
            .dot(&hf.overlap_matrix)
            .dot(&matrix(&model["mo_coeff"]));
        let context = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
        for case in model["cases"].as_array().unwrap() {
            let flow = number(&case["flow"]);
            let actual = SRLDSRG2::new(&hf, flow)
                .with_options(strict())
                .compute()
                .unwrap();
            let solution = actual.ldsrg.as_ref().unwrap();
            let error = compare(
                &rotate(&solution.transformed, &u),
                &operator(&case["transformed"]),
                2e-9,
            )
            .max(compare(
                &rotate(&solution.generator, &u),
                &operator(&case["generator"]),
                2e-9,
            ));
            close(
                actual.correlation_energy,
                number(&case["transformed"]["zero"]),
                1e-10,
            );
            if model["name"].as_str().unwrap().starts_with("He/3s") {
                assert!(actual.max_t1 > 1e-5);
            }
            let pt = SRDSRGPT3::new(&hf, flow).compute().unwrap();
            close(pt.pt2_energy, number(&case["second_order"]), 1e-10);
            close(pt.pt3_correction, number(&case["third_order"]), 1e-10);
            let expansion = pt.perturbative.as_ref().unwrap();
            let pt_error = compare(
                &rotate(&expansion.first_order_generator, &u),
                &operator(&case["first_order_generator"]),
                2e-9,
            )
            .max(compare(
                &rotate(&expansion.second_order_generator, &u),
                &operator(&case["second_order_generator"]),
                2e-9,
            ));
            close(
                SRDSRGPT2::new(&hf, flow)
                    .compute()
                    .unwrap()
                    .correlation_energy,
                pt.pt2_energy,
                1e-10,
            );
            for (method, expected) in [
                (MethodId::SrDsrgPt2, pt.pt2_energy),
                (MethodId::SrLdsrg2, actual.correlation_energy),
                (MethodId::SrDsrgPt3, pt.correlation_energy),
            ] {
                let mut opts = PostHFOptions::default();
                opts.dsrg.flow_parameter = flow;
                opts.residual_tol = 1e-12;
                opts.energy_tol = 1e-14;
                opts.dsrg.bch_tolerance = 1e-14;
                let typed = create_solver(method, &context, &opts)
                    .unwrap()
                    .compute()
                    .unwrap();
                close(typed.e_corr, expected, 1e-12);
                close(typed.e_total, hf.total_energy + expected, 1e-12);
            }
            println!(
                "{} s={flow}: all operator coefficients error {error:e}, perturbative {pt_error:e}, E={:.14}",
                model["name"], actual.correlation_energy
            );
        }
        let limit = SRDSRGPT3::new(&hf, 1e6).compute().unwrap();
        close(limit.pt2_energy, number(&model["mp2"]), 1e-10);
        close(limit.pt3_correction, number(&model["mp3"]), 1e-10);
        assert!(limit.one_rdm_mo.is_none());
        for method in [MethodId::SrDsrgPt2, MethodId::SrDsrgPt3, MethodId::SrLdsrg2] {
            let mut opts = PostHFOptions {
                frozen_core: 1,
                ..Default::default()
            };
            assert!(create_solver(method, &context, &opts).is_err());
            opts.frozen_core = 0;
            opts.density_fitting = true;
            assert!(create_solver(method, &context, &opts).is_err());
        }
        close(
            SRDSRGPT2::new(&hf, 0.)
                .compute()
                .unwrap()
                .correlation_energy,
            0.,
            1e-16,
        );
        let uhf = run_uhf(
            &atoms,
            &basis,
            model["charge"].as_i64().unwrap() as i32,
            1,
            false,
            200,
            1e-14,
        )
        .unwrap();
        close(
            UHFSRDSRGPT2::new(&uhf, 0.)
                .compute()
                .unwrap()
                .correlation_energy,
            0.,
            1e-16,
        );
        let uhf_ctx = PostHFContext::from_uhf(&uhf, &atoms, &basis, None).unwrap();
        let pt2 = UHFSRDSRGPT2::new(&uhf, 0.5).compute().unwrap();
        close(
            create_solver(MethodId::SrDsrgPt2, &uhf_ctx, &PostHFOptions::default())
                .unwrap()
                .compute()
                .unwrap()
                .e_corr,
            pt2.correlation_energy,
            1e-12,
        );
        for method in [MethodId::SrDsrgPt3, MethodId::SrLdsrg2] {
            assert!(create_solver(method, &uhf_ctx, &PostHFOptions::default()).is_err());
        }
        for bad in [-0.1, f64::NAN, f64::INFINITY] {
            assert!(SRDSRGPT2::new(&hf, bad).compute().is_err());
            assert!(UHFSRDSRGPT2::new(&uhf, bad).compute().is_err());
        }
    }
}

#[test]
fn degenerate_gaps_remain_smooth_and_options_do_not_silently_accept_ode_controls() {
    let h = No2bOperator {
        zero: 0.,
        one: Matrix::from_vec(2, 2, vec![0., 0.1, 0.1, 0.]),
        two: vec![0.; 16],
    };
    let occ = [true, false];
    let actual = solve_ldsrg2(&h, &occ, 0.5, strict()).unwrap();
    compare(&actual.transformed, &h, 1e-16);
    assert_eq!(actual.generator.one[(1, 0)], 0.);
    let mut overflow = h.clone();
    overflow.one[(0, 0)] = 1e308;
    overflow.one[(1, 1)] = -1e308;
    assert!(solve_ldsrg2(&overflow, &occ, 0.5, strict()).is_err());
    assert!(dsrg_pt3(&overflow, &occ, 0.5).is_err());
    for controls in [
        LdsrgOptions {
            residual_tolerance: 0.,
            ..strict()
        },
        LdsrgOptions {
            energy_tolerance: f64::NAN,
            ..strict()
        },
        LdsrgOptions {
            bch_tolerance: f64::INFINITY,
            ..strict()
        },
        LdsrgOptions {
            bch_max_order: 0,
            ..strict()
        },
        LdsrgOptions {
            max_iter: 0,
            ..strict()
        },
    ] {
        assert!(solve_ldsrg2(&h, &occ, 0.5, controls).is_err());
    }
    for d in [-1e-10, -1e-14, 0., 1e-14, 1e-10] {
        close(
            hf_rs::post_hf::mr_dsrg::dsrg_regularized_inv_denom(d, 0.5),
            0.5 * d,
            1e-20,
        );
    }
    let map = std::collections::BTreeMap::from([
        ("bch_tolerance".into(), serde_json::json!(1e-14)),
        ("bch_max_order".into(), serde_json::json!(90)),
    ]);
    let opts = PostHFOptions::from_map(&map, &MethodId::SrLdsrg2.option_schemas()).unwrap();
    assert_eq!(opts.dsrg.bch_max_order, 90);
    assert_eq!(opts.dsrg.bch_tolerance, 1e-14);
    assert!(PostHFOptions::from_map(&map, &MethodId::SrDsrgPt3.option_schemas()).is_err());
    let map = std::collections::BTreeMap::from([("ode_step".into(), serde_json::json!(0.01))]);
    for method in [MethodId::SrDsrgPt2, MethodId::SrDsrgPt3, MethodId::SrLdsrg2] {
        assert!(PostHFOptions::from_map(&map, &method.option_schemas()).is_err());
    }
}

#[test]
fn rectangular_mo_references_preserve_sr_dsrg_energies_and_cached_uhf_pt2() {
    let data = data();
    let model = &data["molecules"][1];
    let atoms = [Atom::new("He", [0., 0., 0.])];
    let gbs = model["gaussian94"]
        .as_str()
        .unwrap()
        .replace("****", "S 1 1.0\n0.9 1.0\n****");
    let basis = build_basis(&atoms, &parse_gaussian94(&gbs));
    let hf = run_scf(&atoms, &basis, 200, 1e-14).unwrap();
    let uhf = run_uhf(&atoms, &basis, 0, 1, false, 200, 1e-14).unwrap();
    assert_eq!((hf.mo_coefficients.rows, hf.mo_coefficients.cols), (4, 3));
    assert_eq!(
        (
            uhf.mo_coefficients_alpha.rows,
            uhf.mo_coefficients_alpha.cols
        ),
        (4, 3)
    );
    let rhf_ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let uhf_ctx = PostHFContext::from_uhf(&uhf, &atoms, &basis, None).unwrap();
    let case = &model["cases"][1];
    for (method, expected) in [
        (MethodId::SrDsrgPt2, number(&case["second_order"])),
        (
            MethodId::SrDsrgPt3,
            number(&case["second_order"]) + number(&case["third_order"]),
        ),
        (MethodId::SrLdsrg2, number(&case["transformed"]["zero"])),
    ] {
        let output = create_solver(method, &rhf_ctx, &PostHFOptions::default())
            .unwrap()
            .compute()
            .unwrap();
        close(output.e_corr, expected, 1e-9);
    }
    let output = create_solver(MethodId::SrDsrgPt2, &uhf_ctx, &PostHFOptions::default())
        .unwrap()
        .compute()
        .unwrap();
    close(output.e_corr, number(&case["second_order"]), 1e-9);
}

#[test]
fn every_numerical_nuclear_gradient_component_matches_independent_ecp_and_all_electron_oracles() {
    let data = data();
    let mut maximum = 0f64;
    let mut api_maximum = 0f64;
    let mut count = 0;
    for case in data["gradients"].as_array().unwrap() {
        let model = &data["molecules"][case["model_index"].as_u64().unwrap() as usize];
        let (atoms, basis) = setup_model(model);
        let basis_map = if let Some(gbs) = model["gaussian94"].as_str() {
            parse_gaussian94(gbs)
        } else {
            get_builtin_basis("sto-3g").unwrap()
        };
        assert_eq!(atoms.len(), 2);
        assert!(!basis.is_empty());
        let step = number(&case["step"]);
        let mut gradients = [[0f64; 6]; 2];
        for component in 0..6 {
            let mut values = [[0f64; 2]; 4];
            for (index, displacement) in [-2., -1., 1., 2.].iter().enumerate() {
                let mut shifted = atoms.clone();
                shifted[component / 3].position[component % 3] += displacement * step;
                let basis = build_basis(&shifted, &basis_map);
                let hf = hf_rs::scf::run_scf_in_space_with_convergence(
                    &shifted,
                    &basis,
                    model["charge"].as_i64().unwrap() as i32,
                    300,
                    hf_rs::scf::RhfConvergence {
                        energy: 1e-14,
                        residual: 1e-13,
                        density: 1e-13,
                    },
                    ConvAlgo::Cdiis,
                    None,
                    hf_rs::ao::BasisKind::Cartesian,
                )
                .unwrap();
                values[index][0] = hf.total_energy
                    + SRLDSRG2::new(&hf, 0.5)
                        .with_options(strict())
                        .compute()
                        .unwrap()
                        .correlation_energy;
                values[index][1] = hf.total_energy
                    + SRDSRGPT3::new(&hf, 0.5)
                        .compute()
                        .unwrap()
                        .correlation_energy;
            }
            for method in 0..2 {
                gradients[method][component] = (values[0][method] - 8. * values[1][method]
                    + 8. * values[2][method]
                    - values[3][method])
                    / (12. * step);
            }
        }
        for (method, key) in ["ldsrg", "pt3"].iter().enumerate() {
            for component in 0..6 {
                let error = (gradients[method][component] - number(&case[key][component])).abs();
                maximum = maximum.max(error);
                count += 1;
                assert!(error < 1e-6, "{key} component {component}: error {error:e}");
            }
            for axis in 0..3 {
                let translation = gradients[method][axis] + gradients[method][axis + 3];
                assert!(
                    translation.abs() < 1e-9,
                    "{} {key} axis {axis}: gradients {:?}, sum {translation:e}",
                    model["name"],
                    gradients[method]
                );
            }
            let mut options = PostHFOptions::default();
            options.dsrg.flow_parameter = 0.5;
            options.residual_tol = 1e-12;
            options.energy_tol = 1e-14;
            options.dsrg.bch_tolerance = 1e-14;
            let actual = hf_rs::post_hf::mr_dsrg::sr_dsrg::numerical_gradient(
                [MethodId::SrLdsrg2, MethodId::SrDsrgPt3][method],
                &atoms,
                &basis,
                model["charge"].as_i64().unwrap() as i32,
                hf_rs::ao::BasisKind::Cartesian,
                &options,
                &hf_rs::post_hf::numerical::NumericalGradientOptions {
                    step_bohr: 5e-4,
                    scf_tolerance: 1e-14,
                    scf_residual_tolerance: 1e-13,
                    scf_density_tolerance: 1e-13,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(actual.kind, hf_rs::registry::GradientKind::Numerical);
            assert_eq!(actual.energy_evaluations, 13);
            for component in 0..6 {
                let x = actual.gradient[component / 3][component % 3];
                let y = number(&case[key][component]);
                api_maximum = api_maximum.max((x - y).abs());
                close(x, y, 1e-6);
            }
        }
    }
    println!("SR-DSRG numerical gradient API: all {count} components, max error {api_maximum:e}");
    println!(
        "Numerical SR-DSRG gradients: all {count} Cartesian components, max error {maximum:e}"
    );
}

#[test]
fn open_shell_ecp_pt2_matches_pyscf_ump2_with_nonzero_same_spin_correlation() {
    let data = data();
    let model = &data["uhf_pt2"];
    let (atoms, basis) = setup_model(model);
    let hf = run_uhf(&atoms, &basis, 0, 2, false, 300, 1e-14).unwrap();
    close(hf.total_energy, number(&model["e_scf"]), 1e-10);
    let context = PostHFContext::from_uhf(&hf, &atoms, &basis, None).unwrap();
    for case in model["cases"].as_array().unwrap() {
        let flow = number(&case["flow"]);
        let expected = number(&case["energy"]);
        assert!(number(&case["same_spin"]).abs() > 1e-8);
        let legacy = UHFSRDSRGPT2::new(&hf, flow).compute().unwrap();
        close(legacy.correlation_energy, expected, 1e-10);
        let mut options = PostHFOptions::default();
        options.dsrg.flow_parameter = flow;
        let typed = create_solver(MethodId::SrDsrgPt2, &context, &options)
            .unwrap()
            .compute()
            .unwrap();
        close(typed.e_corr, expected, 1e-10);
    }
}
