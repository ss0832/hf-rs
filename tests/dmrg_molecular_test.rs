// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    context::{PostHFContext, Reference},
    options::{ActiveSpace, PostHFOptions},
    post_hf::{
        casscf_engine::solve_cas,
        dmrg_orbitals::{numerical_gradient, solve_dmrg},
    },
    registry::{GradientKind, MethodId, create_solver},
    *,
};
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
fn options() -> PostHFOptions {
    let mut o = PostHFOptions {
        active: Some(ActiveSpace::new(1, 1, 2)),
        max_iter: 200,
        energy_tol: 1e-13,
        residual_tol: 1e-11,
        ..Default::default()
    };
    o.dmrg.bond_dimensions = vec![16];
    o.dmrg.truncation_tol = 0.0;
    o.cas.orbital_gradient_tol = 1e-9;
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
fn molecular_casci_and_dmrgscf_match_pyscf_including_ecp_and_rectangular_orbitals() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_molecular_reference.json")).unwrap();
    let mut error = 0.0_f64;
    for model in data["cases"].as_array().unwrap() {
        let (atoms, basis, hf) = reference(model);
        let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
        let optimize = model["optimize"].as_bool().unwrap_or(false);
        let method = if optimize {
            MethodId::DmrgScf
        } else {
            MethodId::Dmrg
        };
        let opts = options();
        let result = solve_dmrg(&ctx, &opts, optimize).unwrap();
        let out = create_solver(method, &ctx, &opts)
            .unwrap()
            .compute()
            .unwrap();
        let delta = (out.e_total - number(&model["e_cas"])).abs();
        error = error.max(delta);
        assert!(delta < 1e-8, "{}: {delta:e}", model["name"]);
        assert!((out.e_total - result.output.e_total).abs() < 1e-12);
        assert!((out.e_corr - (out.e_total - hf.total_energy)).abs() < 1e-12);
        if optimize {
            assert!(result.orbital_gradient_max < 1e-9);
        }
        let cas = solve_cas(Reference::Rhf(&hf), &opts, None, optimize).unwrap();
        let d = result.output.one_rdm.as_ref().unwrap();
        let d_cas = cas.output.one_rdm.as_ref().unwrap();
        let ao = result
            .mo_coefficients
            .dot(d)
            .dot(&result.mo_coefficients.transpose());
        let ao_cas = cas
            .mo_coefficients
            .dot(d_cas)
            .dot(&cas.mo_coefficients.transpose());
        assert!(
            ao.data
                .iter()
                .zip(&ao_cas.data)
                .all(|(a, b)| (a - b).abs() < 1e-7)
        );
    }
    println!("DMRG molecular maximum energy error={error:e}");
}

#[test]
fn unsupported_controls_and_unconverged_states_are_rejected() {
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, 1.4]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_scf(&atoms, &basis, 100, 1e-14).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let mut o = options();
    o.dmrg.noise = 1e-4;
    assert!(solve_dmrg(&ctx, &o, false).is_err());
    let mut o = options();
    o.cas.target_s_squared = Some(0.0);
    assert!(solve_dmrg(&ctx, &o, false).is_err());
    let mut o = options();
    o.dmrg.sweeps = 1;
    o.dmrg.one_site_sweeps = 0;
    assert!(solve_dmrg(&ctx, &o, false).is_err());
    let mut o = options();
    o.active = Some(ActiveSpace::new(0, 0, 0));
    let out = solve_dmrg(&ctx, &o, false).unwrap();
    assert!((out.output.e_total - hf.total_energy).abs() < 1e-12);
    let values =
        std::collections::BTreeMap::from([("one_site_sweeps".into(), serde_json::json!(8))]);
    assert_eq!(
        PostHFOptions::from_map(&values, &MethodId::Dmrg.option_schemas())
            .unwrap()
            .dmrg
            .one_site_sweeps,
        8
    );
    assert_eq!(
        MethodId::Dmrg.gradient_kind(context::RefKind::Rhf, &options()),
        GradientKind::Numerical
    );
    assert_eq!(
        MethodId::Dmrg.gradient_kind(context::RefKind::Uhf, &options()),
        GradientKind::Unsupported
    );
}

#[test]
fn finite_m_scf_includes_nonredundant_active_active_rotations() {
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, 1.4]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let mut hf = run_scf(&atoms, &basis, 100, 1e-14).unwrap();
    let mut rotation = Matrix::zeros(2, 2);
    rotation[(1, 0)] = 0.2;
    rotation[(0, 1)] = -0.2;
    hf.mo_coefficients = hf.mo_coefficients.dot(&rotation.exp());
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let mut opts = options();
    opts.dmrg.bond_dimensions = vec![1];
    let fixed = solve_dmrg(&ctx, &opts, false).unwrap();
    let optimized = solve_dmrg(&ctx, &opts, true).unwrap();
    assert!(fixed.output.e_total - hf.total_energy > 1e-4);
    assert!((optimized.output.e_total - hf.total_energy).abs() < 1e-9);
    assert!(optimized.orbital_gradient_max < 1e-9);
    assert!(optimized.orbital_iterations > 0);
}

#[test]
fn open_shell_context_preserves_both_active_electron_counts() {
    let atoms = vec![
        Atom::new("H", [0.1, -0.2, 0.3]),
        Atom::new("H", [0.4, 0.2, 1.7]),
        Atom::new("H", [1.4, -0.3, -0.4]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let uhf = run_uhf(&atoms, &basis, 0, 2, false, 150, 1e-14).unwrap();
    let ctx = PostHFContext::from_uhf(&uhf, &atoms, &basis, None).unwrap();
    let mut opts = options();
    opts.active = Some(ActiveSpace::new(2, 1, 3));
    let result = solve_dmrg(&ctx, &opts, false).unwrap();
    let exact = solve_cas(Reference::Uhf(&uhf), &opts, None, false).unwrap();
    assert_eq!(result.mps.state.electron_counts(), (2, 1));
    assert!((result.output.e_total - exact.output.e_total).abs() < 1e-10);
    assert!(
        result
            .active_one_rdm
            .data
            .iter()
            .zip(&exact.averaged_one_rdm.data)
            .all(|(a, b)| (a - b).abs() < 1e-8)
    );
    assert!(
        result
            .active_two_rdm
            .iter()
            .zip(&exact.averaged_two_rdm)
            .all(|(a, b)| (a - b).abs() < 1e-8)
    );
}

#[test]
fn all_nuclear_components_match_pyscf_energy_stencils_including_nonlocal_ecp() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_molecular_reference.json")).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/dmrg_gradient_reference.json")).unwrap();
    let step = number(&oracle["step"]);
    let mut max_five = 0.0_f64;
    let mut max_api = 0.0_f64;
    let mut max_translation = 0.0_f64;
    let mut count = 0;
    for case in oracle["cases"].as_array().unwrap() {
        let model = &data["cases"][case["model_index"].as_u64().unwrap() as usize];
        let (atoms, basis) = setup_model(model);
        let charge = model["charge"].as_i64().unwrap() as i32;
        for (mode, optimize) in [false, true].into_iter().enumerate() {
            let opts = options();
            let settings = post_hf::numerical::NumericalGradientOptions {
                step_bohr: step,
                ..Default::default()
            };
            let api = numerical_gradient(
                optimize,
                &atoms,
                &basis,
                charge,
                ao::BasisKind::Cartesian,
                &opts,
                &settings,
            )
            .unwrap();
            assert_eq!(api.kind, GradientKind::Numerical);
            let mut gradient = vec![[0.0; 3]; atoms.len()];
            for (component, expected) in case["gradient"][mode]
                .as_array()
                .unwrap()
                .iter()
                .map(number)
                .enumerate()
            {
                let mut values = Vec::new();
                for offset in [-2.0, -1.0, 1.0, 2.0] {
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
                    values.push(solve_dmrg(&ctx, &opts, optimize).unwrap().output.e_total);
                }
                let value =
                    (values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (12.0 * step);
                max_five = max_five.max((value - expected).abs());
                max_api =
                    max_api.max((api.gradient[component / 3][component % 3] - expected).abs());
                gradient[component / 3][component % 3] = value;
                count += 1;
            }
            for axis in 0..3 {
                max_translation =
                    max_translation.max(gradient.iter().map(|g| g[axis]).sum::<f64>().abs());
            }
        }
    }
    println!(
        "DMRG gradients: count={count}, five_point={max_five:e}, API={max_api:e}, translation={max_translation:e}"
    );
    assert_eq!(count, 30);
    assert!(max_five < 1e-6);
    assert!(max_api < 1e-6);
    assert!(max_translation < 1e-9);
}
