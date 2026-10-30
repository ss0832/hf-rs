// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Spin-free DKH2: every AO matrix entry and Frechet response against an independent
//! element-wise NumPy construction (tests/reference/gen_dkh2.py), exact model limits,
//! molecular energies against PySCF/NumPy and the external Psi4 program
//! (tests/reference/gen_dkh2_psi4.py), and every analytic nuclear gradient component
//! against five-point energy differences.

use hf_rs::hf_variants::dkh2_core::SpinFreeDkh2Core;
use hf_rs::hf_variants::x2c_core::SpinFreeX2cCore;
use hf_rs::{
    build_basis, compute_relativistic_hf_gradient, get_builtin_basis, run_relativistic_hf_with_options, Atom, Matrix,
    RelativisticHfOptions, RelativisticModel,
};

const BUDGET: usize = 32 * 1024 * 1024;
const C: f64 = 137.035999084;

fn matrix(value: &serde_json::Value) -> Matrix {
    let rows = value.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter().flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap())).collect(),
    )
}

fn atoms_of(case: &serde_json::Value) -> Vec<Atom> {
    case["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| Atom::new(a["symbol"].as_str().unwrap(), std::array::from_fn(|k| a["position"][k].as_f64().unwrap())))
        .collect()
}

fn basis_name(constant: &str) -> &'static str {
    match constant {
        "STO_3G_GBS" => "sto-3g",
        "BASIS_6_31G_GBS" => "6-31g",
        other => panic!("unexpected basis constant {other}"),
    }
}

fn tight() -> RelativisticHfOptions {
    RelativisticHfOptions {
        max_iterations: 500,
        convergence: hf_rs::scf::RhfConvergence { energy: 1e-12, residual: 1e-10, density: 1e-10 },
        ..Default::default()
    }
}

#[test]
fn every_dkh2_matrix_entry_and_frechet_response_matches_the_independent_elementwise_oracle() {
    let data: serde_json::Value = serde_json::from_str(include_str!("data/dkh2_reference.json")).unwrap();
    let (mut count, mut worst) = ([0usize; 2], [0f64; 2]);
    for case in data["cases"].as_array().unwrap() {
        let m: Vec<_> = case["matrices"].as_array().unwrap().iter().map(matrix).collect();
        let core = SpinFreeDkh2Core::build(&m[0], &m[1], &m[2], &m[3], C, BUDGET).unwrap();
        for (a, b) in core.hamiltonian.data.iter().zip(matrix(&case["hamiltonian"]).data) {
            let error = (a - b).abs();
            worst[0] = worst[0].max(error);
            count[0] += 1;
            assert!(error < 1e-9 * (1. + b.abs()), "{} entry {a} vs {b}", case["name"]);
        }
        // Hermiticity is not imposed by symmetrization; it must hold by construction.
        let n = core.hamiltonian.rows;
        for p in 0..n {
            for q in 0..p {
                assert!((core.hamiltonian[(p, q)] - core.hamiltonian[(q, p)]).abs() < 1e-10 * (1. + core.hamiltonian[(p, q)].abs()));
            }
        }
        for point in case["directions"].as_array().unwrap() {
            let d: Vec<_> = point["tangents"].as_array().unwrap().iter().map(matrix).collect();
            let actual = core.response(&d[0], &d[1], &d[2], &d[3]).unwrap();
            for (a, b) in actual.data.iter().zip(matrix(&point["response"]).data) {
                let error = (a - b).abs();
                worst[1] = worst[1].max(error);
                count[1] += 1;
                assert!(error < 1e-7 * (1. + b.abs()), "{} response {a} vs {b}", case["name"]);
            }
        }
    }
    assert_eq!(count, [25 + 81 + 169 + 225, 3 * (25 + 81 + 169 + 225)]);
    println!("DKH2 matrix/response counts {count:?}; maximum errors {worst:?}");
}

#[test]
fn dkh2_constant_potential_and_nonrelativistic_limits_are_exact() {
    // For V = v0 S and W = <p chi|v0|p chi> = 2 v0 T the odd operator vanishes identically and
    // E1 = v0, so h = (E - c^2)(T) + v0 S exactly; X2C gives the same operator.
    let s = Matrix::from_vec(3, 3, vec![1.0, 0.2, 0.05, 0.2, 1.0, 0.1, 0.05, 0.1, 1.0]);
    let t = Matrix::from_vec(3, 3, vec![0.9, 0.1, 0.0, 0.1, 2.3, 0.4, 0.0, 0.4, 7.5]);
    let v0 = -3.7;
    let v = s.scale(v0);
    let w = t.scale(2. * v0);
    let zero = Matrix::zeros(3, 3);
    let free = SpinFreeDkh2Core::build(&s, &t, &zero, &zero, C, BUDGET).unwrap();
    let shifted = SpinFreeDkh2Core::build(&s, &t, &v, &w, C, BUDGET).unwrap();
    let x2c = SpinFreeX2cCore::build(&s, &t, &v, &w, C, BUDGET).unwrap();
    for p in 0..3 {
        for q in 0..3 {
            let expected = free.hamiltonian[(p, q)] + v0 * s[(p, q)];
            assert!((shifted.hamiltonian[(p, q)] - expected).abs() < 1e-11);
            assert!((x2c.hamiltonian[(p, q)] - expected).abs() < 1e-9);
        }
    }
    // c -> infinity: h -> T + V
    let v = Matrix::from_vec(3, 3, vec![-5.0, 0.3, 0.1, 0.3, -2.0, -0.2, 0.1, -0.2, -0.7]);
    let w = Matrix::from_vec(3, 3, vec![-9.0, 1.1, 0.2, 1.1, -14.0, 0.9, 0.2, 0.9, -40.0]);
    let large = SpinFreeDkh2Core::build(&s, &t, &v, &w, 1e6, BUDGET).unwrap();
    for (a, b) in large.hamiltonian.data.iter().zip(t.add(&v).data) {
        assert!((a - b).abs() < 1e-9);
    }
    // DKH2 is exact through second order in V: its deviation from the infinite-order X2C
    // operator is third order and scales as c^-4 at large c.
    let deviation = |c: f64| {
        let d = SpinFreeDkh2Core::build(&s, &t, &v, &w, c, BUDGET).unwrap().hamiltonian;
        let x = SpinFreeX2cCore::build(&s, &t, &v, &w, c, BUDGET).unwrap().hamiltonian;
        d.data.iter().zip(&x.data).map(|(a, b)| (a - b).abs()).fold(0., f64::max)
    };
    let ratio = deviation(2000.) / deviation(4000.);
    assert!((ratio - 16.).abs() < 0.2, "DKH2 - X2C deviation scales with ratio {ratio} (expected 16)");
    assert!(SpinFreeDkh2Core::build(&s, &t, &v, &w, 0., BUDGET).is_err());
    assert!(SpinFreeDkh2Core::build(&s, &t, &v, &w, C, 1).is_err());
    assert!(SpinFreeDkh2Core::build(&s, &t.scale(-1.), &v, &w, C, BUDGET).is_err());
}

#[test]
fn dkh2_response_is_continuous_at_degenerate_kinetic_eigenvalues() {
    let s = Matrix::identity(3);
    let t = Matrix::from_vec(3, 3, vec![0.4, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.1]);
    let v = Matrix::from_vec(3, 3, vec![-2.0, 0.3, 0.1, 0.3, -1.0, -0.2, 0.1, -0.2, -0.5]);
    let w = Matrix::from_vec(3, 3, vec![-3.0, 0.4, 0.2, 0.4, -2.5, 0.3, 0.2, 0.3, -6.0]);
    let dt = Matrix::from_vec(3, 3, vec![0.1, -0.02, 0.03, -0.02, -0.07, 0.04, 0.03, 0.04, 0.02]);
    let dv = Matrix::from_vec(3, 3, vec![0.05, 0.01, -0.02, 0.01, 0.03, 0.02, -0.02, 0.02, -0.04]);
    let dw = dv.scale(1.7);
    let ds = Matrix::from_vec(3, 3, vec![0.0, 0.01, 0.0, 0.01, 0.0, -0.02, 0.0, -0.02, 0.0]);
    let core = SpinFreeDkh2Core::build(&s, &t, &v, &w, C, BUDGET).unwrap();
    let analytic = core.response(&ds, &dt, &dv, &dw).unwrap();
    let h = 1e-4;
    let at = |x: f64| {
        SpinFreeDkh2Core::build(&s.add(&ds.scale(x)), &t.add(&dt.scale(x)), &v.add(&dv.scale(x)), &w.add(&dw.scale(x)), C, BUDGET)
            .unwrap()
            .hamiltonian
    };
    let (m2, m1, p1, p2) = (at(-2. * h), at(-h), at(h), at(2. * h));
    for k in 0..9 {
        let fd = (m2.data[k] - 8. * m1.data[k] + 8. * p1.data[k] - p2.data[k]) / (12. * h);
        assert!((fd - analytic.data[k]).abs() < 1e-8, "component {k}: {fd} vs {}", analytic.data[k]);
    }
}

#[test]
fn molecular_dkh2_energies_match_pyscf_numpy_and_external_psi4() {
    let data: serde_json::Value = serde_json::from_str(include_str!("data/dkh2_reference.json")).unwrap();
    let mut worst = 0f64;
    for case in data["cases"].as_array().unwrap() {
        let atoms = atoms_of(case);
        let basis = build_basis(&atoms, &get_builtin_basis(basis_name(case["basis_constant"].as_str().unwrap())).unwrap());
        let r = run_relativistic_hf_with_options(&atoms, &basis, 0, RelativisticModel::DKH2, &tight()).unwrap();
        let e = case["e_total"].as_f64().unwrap();
        let error = (r.total_energy - e).abs();
        worst = worst.max(error);
        assert!(error < 1e-8, "{} DKH2 energy {} vs {}", case["name"], r.total_energy, e);
        assert!((r.non_relativistic_energy - case["e_nonrelativistic"].as_f64().unwrap()).abs() < 1e-8);
        assert!(r.relativistic_shift_hartree < 0.0, "scalar relativity stabilizes core electrons");
    }
    let psi4: serde_json::Value = serde_json::from_str(include_str!("data/dkh2_psi4_reference.json")).unwrap();
    let c_psi4 = psi4["dkh_speed_of_light"].as_f64().unwrap();
    let mut worst_psi4 = [0f64; 2];
    for case in psi4["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (atoms, constant) = match name {
            "Ne" => (vec![Atom::new("Ne", [0.1, -0.2, 0.3])], "sto-3g"),
            "Ar" => (vec![Atom::new("Ar", [0.1, -0.2, 0.3])], "sto-3g"),
            "Kr" => (vec![Atom::new("Kr", [0.0, 0.0, 0.0])], "sto-3g"),
            "water" => (
                vec![Atom::new("O", [0.1, -0.2, 0.3]), Atom::new("H", [1.53, 0.68, 0.3]), Atom::new("H", [-1.33, 0.68, 0.3])],
                "6-31g",
            ),
            "HCl" => (vec![Atom::new("Cl", [0.1, -0.2, 0.3]), Atom::new("H", [0.1, -0.2, 2.7])], "6-31g"),
            other => panic!("unexpected case {other}"),
        };
        let basis = build_basis(&atoms, &get_builtin_basis(constant).unwrap());
        let mut options = tight();
        options.speed_of_light = c_psi4;
        let dkh = run_relativistic_hf_with_options(&atoms, &basis, 0, RelativisticModel::DKH2, &options).unwrap();
        // Total energies of a few thousand Eh carry ~1e-12 relative SCF noise in the external program.
        let tolerance = 1e-8_f64.max(4e-12 * dkh.total_energy.abs());
        let e1 = (dkh.total_energy - case["e_dkh2"].as_f64().unwrap()).abs();
        worst_psi4[0] = worst_psi4[0].max(e1);
        assert!(e1 < tolerance, "{name} DKH2 vs Psi4: {e1:e}");
        let mut x2c_options = tight();
        x2c_options.speed_of_light = psi4["x2c_speed_of_light"].as_f64().unwrap();
        let x2c = run_relativistic_hf_with_options(&atoms, &basis, 0, RelativisticModel::X2C, &x2c_options).unwrap();
        let e2 = (x2c.total_energy - case["e_x2c"].as_f64().unwrap()).abs();
        worst_psi4[1] = worst_psi4[1].max(e2);
        assert!(e2 < tolerance, "{name} X2C vs Psi4: {e2:e}");
    }
    println!("DKH2 energies: PySCF/NumPy {worst:e}; Psi4 DKH2/X2C {worst_psi4:?}");
}

#[test]
fn every_analytic_dkh2_nuclear_gradient_component_matches_five_point_energy_differences() {
    let data: serde_json::Value = serde_json::from_str(include_str!("data/dkh2_reference.json")).unwrap();
    let (mut count, mut worst, mut translation) = (0usize, 0f64, 0f64);
    for case in data["cases"].as_array().unwrap() {
        let atoms = atoms_of(case);
        let constant = basis_name(case["basis_constant"].as_str().unwrap());
        let map = get_builtin_basis(constant).unwrap();
        let basis = build_basis(&atoms, &map);
        let r = run_relativistic_hf_with_options(&atoms, &basis, 0, RelativisticModel::DKH2, &tight()).unwrap();
        let g = compute_relativistic_hf_gradient(&atoms, &basis, &r).unwrap();
        let reference = case["five_point_gradient"].as_array().unwrap();
        for (a, row) in reference.iter().enumerate() {
            for axis in 0..3 {
                let error = (g[a][axis] - row[axis].as_f64().unwrap()).abs();
                worst = worst.max(error);
                count += 1;
                assert!(error < 1e-6, "{} nucleus {a} axis {axis}: {error:e}", case["name"]);
            }
        }
        for axis in 0..3 {
            let sum = g.iter().map(|x| x[axis]).sum::<f64>().abs();
            translation = translation.max(sum);
            assert!(sum < 1e-9, "{} translation {sum:e}", case["name"]);
        }
        // The variant dispatcher uses the same backend.
        let dispatched =
            hf_rs::compute_hf_variant_analytical_gradient(&atoms, &map, hf_rs::HFVariantType::Dkh2HF, 0, 500, 1e-12).unwrap();
        for (x, y) in dispatched.iter().zip(&g) {
            for axis in 0..3 {
                assert!((x[axis] - y[axis]).abs() < 1e-8);
            }
        }
    }
    assert_eq!(count, 3 + 3 + 9 + 6);
    println!("DKH2 analytic gradient: {count} components, max error {worst:e}, translation {translation:e}");
}

#[test]
fn zora_is_explicitly_unsupported() {
    let atoms = vec![Atom::new("H", [0., 0., -0.7]), Atom::new("H", [0., 0., 0.7])];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let err = run_relativistic_hf_with_options(&atoms, &basis, 0, RelativisticModel::ZORA, &RelativisticHfOptions::default())
        .unwrap_err();
    assert!(err.contains("ZORA"), "{err}");
}
