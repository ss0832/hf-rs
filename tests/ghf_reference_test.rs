// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::hf_variants::ghf::{
    GhfOptions, build_ghf_fock_matrix, compute_ghf_gradient, run_ghf_with_options,
};
use hf_rs::*;
use serde_json::Value;
fn number(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn matrix(v: &Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(number))
            .collect(),
    )
}
fn block(a: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(2 * a.rows, 2 * a.cols);
    for p in 0..a.rows {
        for q in 0..a.cols {
            out[(p, q)] = a[(p, q)];
            out[(a.rows + p, a.cols + q)] = a[(p, q)];
        }
    }
    out
}
fn setup(model: &Value) -> (Vec<Atom>, Vec<ContractedGaussian>) {
    let mut atoms = model["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            Atom::new(
                a["symbol"].as_str().unwrap(),
                std::array::from_fn(|k| number(&a["position"][k])),
            )
        })
        .collect::<Vec<_>>();
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
        build_basis(
            &atoms,
            &get_builtin_basis(model["basis_name"].as_str().unwrap_or("sto-3g")).unwrap(),
        )
    };
    (atoms, basis)
}

#[test]
fn tilted_guess_follows_a_spin_instability_and_all_ethylene_gradient_components_match_pyscf() {
    let data: Value = serde_json::from_str(include_str!("data/ghf_reference.json")).unwrap();
    let model = &data["stress_cases"][0];
    let (atoms, basis) = setup(model);
    let result =
        hf_variants::ghf::run_ghf(&atoms, &basis, 0, std::f64::consts::FRAC_PI_4, 100, 1e-12)
            .unwrap();
    let error = (result.total_energy - number(&model["e_total"])).abs();
    println!(
        "GHF ethylene: Rust={:.14}, PySCF={:.14}, S2={:.12}",
        result.total_energy,
        number(&model["e_total"]),
        result.s_squared
    );
    assert!(error < 1e-8, "energy error {error:e}");
    assert!(result.total_energy < number(&model["e_rhf"]) - 1e-3);
    assert!((result.s_squared - number(&model["s_squared"])).abs() < 1e-7);
    let density_error = aligned_error(&result.density_matrix_2n, &matrix(&model["density"]));
    assert!(
        density_error < 1e-8,
        "spin-frame density error {density_error:e}"
    );
    let (_, _, _, h) = scf::compute_one_electron_integrals(&atoms, &basis);
    let eri = scf::compute_eri_tensor(&basis);
    let external_density = matrix(&model["density"]);
    let fock_error = maximum(
        &build_ghf_fock_matrix(&h, &external_density, &eri, basis.len()),
        &matrix(&model["fock"]),
    );
    assert!(fock_error < 1e-10, "Fock error {fock_error:e}");
    println!(
        "GHF ethylene full matrices: density={density_error:e}, fock={fock_error:e}, entries={}",
        external_density.data.len()
    );
    let gradient = compute_ghf_gradient(&atoms, &basis, &result).unwrap();
    let error_gradient = gradient
        .iter()
        .flatten()
        .zip(model["gradient"].as_array().unwrap().iter().map(number))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let translation = (0..3)
        .map(|axis| gradient.iter().map(|g| g[axis]).sum::<f64>().abs())
        .fold(0.0_f64, f64::max);
    println!(
        "GHF spin instability: energy={error:e}, gradient_components=18, gradient={error_gradient:e}, translation={translation:e}"
    );
    assert!(error_gradient < 1e-6);
    assert!(translation < 1e-9);
}
fn options(seed: Matrix) -> GhfOptions {
    GhfOptions {
        max_iterations: 500,
        energy_tol: 1e-13,
        residual_tol: 1e-10,
        density_tol: 1e-10,
        initial_density: Some(seed.occupied_density(seed.cols, 1.)),
        ..Default::default()
    }
}
fn maximum(a: &Matrix, b: &Matrix) -> f64 {
    a.data
        .iter()
        .zip(&b.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max)
}
fn spin_rotate(p: &Matrix, angle: f64, flip: f64) -> Matrix {
    let n = p.rows / 2;
    let mut u = Matrix::zeros(2 * n, 2 * n);
    let (c, s) = ((angle * 0.5).cos(), (angle * 0.5).sin());
    for a in 0..n {
        u[(a, a)] = c;
        u[(a, n + a)] = -s * flip;
        u[(n + a, a)] = s;
        u[(n + a, n + a)] = c * flip;
    }
    u.dot(p).dot(&u.transpose())
}
fn aligned_error(p: &Matrix, q: &Matrix) -> f64 {
    let n = p.rows / 2;
    [1., -1.]
        .into_iter()
        .map(|flip| {
            let v = spin_rotate(p, 0., flip);
            let mut a = 0.;
            let mut b = 0.;
            for i in 0..n {
                for j in 0..n {
                    let (x, z) = (
                        (v[(i, n + j)] + v[(n + i, j)]) * 0.5,
                        (v[(i, j)] - v[(n + i, n + j)]) * 0.5,
                    );
                    let (xx, zz) = (
                        (q[(i, n + j)] + q[(n + i, j)]) * 0.5,
                        (q[(i, j)] - q[(n + i, n + j)]) * 0.5,
                    );
                    a += x * xx + z * zz;
                    b += z * xx - x * zz;
                }
            }
            maximum(&spin_rotate(p, b.atan2(a), flip), q)
        })
        .fold(f64::INFINITY, f64::min)
}
#[test]
fn real_ghf_matches_pyscf_and_returns_a_consistent_stationary_determinant() {
    let data: Value = serde_json::from_str(include_str!("data/ghf_reference.json")).unwrap();
    let mut max_energy = 0.0_f64;
    let mut max_density = 0.0_f64;
    let mut max_fock = 0.0_f64;
    let mut components = 0;
    for model in data["cases"].as_array().unwrap() {
        let (atoms, basis) = setup(model);
        let n = basis.len();
        let charge = model["charge"].as_i64().unwrap() as i32;
        let opts = options(matrix(&model["initial_coeff"]));
        let result = run_ghf_with_options(&atoms, &basis, charge, 0., &opts).unwrap();
        let delta = (result.total_energy - number(&model["e_total"])).abs();
        max_energy = max_energy.max(delta);
        let density = aligned_error(&result.density_matrix_2n, &matrix(&model["density"]));
        max_density = max_density.max(density);
        println!(
            "GHF {}: E={:.14}, delta={delta:e}, density={density:e}, S2={:.12}, iter={}",
            model["name"], result.total_energy, result.s_squared, result.iterations
        );
        assert!(delta < 1e-8);
        assert!(density < 1e-8);
        assert!((result.s_squared - number(&model["s_squared"])).abs() < 1e-8);
        let (s, _, _, h) = scf::compute_one_electron_integrals(&atoms, &basis);
        let s2 = block(&s);
        let p = &result.density_matrix_2n;
        assert!((p.dot(&s2).trace() - result.n_electrons as f64).abs() < 1e-10);
        assert!(maximum(&p.dot(&s2).dot(p), p) < 1e-10);
        assert!(
            maximum(
                &result
                    .spinor_coefficients
                    .occupied_density(result.n_electrons, 1.),
                p
            ) < 1e-10
        );
        assert!(
            maximum(
                &result
                    .spinor_coefficients
                    .transpose()
                    .dot(&s2)
                    .dot(&result.spinor_coefficients),
                &Matrix::identity(2 * n)
            ) < 1e-10
        );
        let eri = scf::compute_eri_tensor(&basis);
        assert!(
            maximum(
                &build_ghf_fock_matrix(&h, p, &eri, n),
                &result.fock_matrix_2n
            ) < 1e-12
        );
        let arbitrary = matrix(&model["random_density"]);
        let (_, spin_square) = hf_variants::ghf::ghf_spin_expectations(&arbitrary, &s);
        assert!((spin_square - number(&model["random_s_squared"])).abs() < 1e-10);
        let fock = build_ghf_fock_matrix(&h, &arbitrary, &eri, n);
        max_fock = max_fock.max(maximum(&fock, &matrix(&model["random_fock"])));
        assert!(maximum(&fock, &matrix(&model["random_fock"])) < 1e-10);
        let rotated = spin_rotate(&arbitrary, 0.73, 1.);
        let rotated_fock = build_ghf_fock_matrix(&h, &rotated, &eri, n);
        assert!(maximum(&rotated_fock, &spin_rotate(&fock, 0.73, 1.)) < 1e-11);
        components += fock.data.len();
        let mut fc = result.fock_matrix_2n.dot(&result.spinor_coefficients);
        let mut sc = s2.dot(&result.spinor_coefficients);
        for i in 0..sc.rows {
            for j in 0..sc.cols {
                sc[(i, j)] *= result.spinor_energies[j];
            }
        }
        fc = fc.sub(&sc);
        assert!(fc.data.iter().all(|x| x.abs() < 1e-9));
    }
    println!(
        "GHF oracles: energy={max_energy:e}, density={max_density:e}, fock={max_fock:e}, components={components}"
    );
}
#[test]
fn all_analytic_nuclear_components_include_spin_coherence_and_nonlocal_ecp() {
    let data: Value = serde_json::from_str(include_str!("data/ghf_reference.json")).unwrap();
    let mut maximum = 0.0_f64;
    let mut translation = 0.0_f64;
    let mut count = 0;
    for model in data["cases"].as_array().unwrap() {
        let (atoms, basis) = setup(model);
        let charge = model["charge"].as_i64().unwrap() as i32;
        let result = run_ghf_with_options(
            &atoms,
            &basis,
            charge,
            0.,
            &options(matrix(&model["initial_coeff"])),
        )
        .unwrap();
        let gradient = compute_ghf_gradient(&atoms, &basis, &result).unwrap();
        if model["name"].as_str().unwrap().starts_with("H2+") {
            let dispatch = hf_variants::hessian::compute_hf_variant_analytical_gradient(
                &atoms,
                &get_builtin_basis("sto-3g").unwrap(),
                hf_variants::hessian::HFVariantType::GHF,
                charge,
                200,
                1e-13,
            )
            .unwrap();
            assert!(
                dispatch
                    .iter()
                    .flatten()
                    .zip(gradient.iter().flatten())
                    .all(|(a, b)| (a - b).abs() < 1e-9)
            );
        }
        for (a, b) in gradient
            .iter()
            .flatten()
            .zip(model["gradient"].as_array().unwrap().iter().map(number))
        {
            maximum = maximum.max((a - b).abs());
            count += 1;
        }
        for axis in 0..3 {
            translation = translation.max(gradient.iter().map(|g| g[axis]).sum::<f64>().abs());
        }
    }
    println!(
        "GHF analytic gradients: components={count}, error={maximum:e}, translation={translation:e}"
    );
    assert_eq!(count, 27);
    assert!(maximum < 1e-6);
    assert!(translation < 1e-9);
}
#[test]
fn overlap_rank_controls_invalid_inputs_and_nonconvergence_are_explicit() {
    let atoms = vec![
        Atom::new("H", [0., 0., -0.7]),
        Atom::new("H", [0., 0., 0.7]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let mut opts = GhfOptions {
        max_iterations: 1,
        ..Default::default()
    };
    assert!(run_ghf_with_options(&atoms, &basis, 0, 0., &opts).is_err());
    opts = GhfOptions::default();
    opts.max_work_bytes = 1;
    assert!(run_ghf_with_options(&atoms, &basis, 0, 0., &opts).is_err());
    opts = GhfOptions::default();
    opts.initial_density = Some(Matrix::identity(4));
    assert!(run_ghf_with_options(&atoms, &basis, 0, 0., &opts).is_err());
    let mut duplicate = basis.clone();
    duplicate.extend(basis.clone());
    let result = run_ghf_with_options(&atoms, &duplicate, 0, 0., &Default::default()).unwrap();
    assert_eq!(result.spinor_coefficients.cols, 4);
    assert!((result.total_energy + 1.116714325176).abs() < 1e-9);
    assert!(compute_ghf_gradient(&atoms, &duplicate, &result).is_err());
}
