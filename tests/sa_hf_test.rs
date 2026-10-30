// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! State-averaged HF over explicit determinants against an independent SciPy minimization
//! (tests/reference/gen_sa_hf.py), exact single-state limits (RHF, ROHF), ensemble symmetry
//! and stationarity of the exact ensemble energy.

use hf_rs::{
    Atom, Matrix, SaHfOptions, StateConfiguration, build_basis, get_builtin_basis, run_rohf,
    run_scf, run_state_averaged_hf, run_state_averaged_hf_with_options,
};

fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/sa_hf_reference.json")).unwrap()
}

fn tight() -> SaHfOptions {
    SaHfOptions {
        max_iterations: 2000,
        gradient_tol: 1e-8,
        energy_tol: 1e-12,
    }
}

fn atoms(case: &serde_json::Value) -> Vec<Atom> {
    case["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            Atom::new(
                a["symbol"].as_str().unwrap(),
                std::array::from_fn(|k| a["position"][k].as_f64().unwrap()),
            )
        })
        .collect()
}

fn indices(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

fn basis_name(case: &serde_json::Value) -> &'static str {
    match case["constant"].as_str().unwrap() {
        "STO_3G_GBS" => "sto-3g",
        "BASIS_6_31G_GBS" => "6-31g",
        other => panic!("unexpected basis {other}"),
    }
}

#[test]
fn state_averaged_hf_matches_independent_minimization() {
    let mut worst = 0f64;
    for case in data()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let atoms = atoms(case);
        let basis = build_basis(&atoms, &get_builtin_basis(basis_name(case)).unwrap());
        let states: Vec<StateConfiguration> = case["states"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| StateConfiguration::new(indices(&s[0]), indices(&s[1])))
            .collect();
        let weights: Vec<f64> = case["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| w.as_f64().unwrap())
            .collect();
        let r = run_state_averaged_hf_with_options(&atoms, &basis, &states, &weights, &tight())
            .unwrap();
        assert!(r.converged);
        let reference = case["sa_energy"].as_f64().unwrap();
        worst = worst.max((r.sa_energy - reference).abs());
        assert!(
            (r.sa_energy - reference).abs() < 1e-8,
            "{name}: {} vs {reference}",
            r.sa_energy
        );
        for (e, reference) in r
            .state_energies
            .iter()
            .zip(case["state_energies"].as_array().unwrap())
        {
            let reference = reference.as_f64().unwrap();
            worst = worst.max((e - reference).abs());
            assert!(
                (e - reference).abs() < 1e-7,
                "{name}: state {e} vs {reference}"
            );
        }
        let average: f64 = r
            .state_energies
            .iter()
            .zip(&weights)
            .map(|(e, w)| e * w)
            .sum();
        assert!((average - r.sa_energy).abs() < 1e-12);
        let density = &case["density"];
        let mut dp = 0f64;
        for i in 0..r.density_matrix.rows {
            for j in 0..r.density_matrix.cols {
                dp = dp.max((r.density_matrix[(i, j)] - density[i][j].as_f64().unwrap()).abs());
            }
        }
        assert!(dp < 1e-6, "{name}: ensemble density {dp:e}");
    }
    println!("SA-HF: max |dE| {worst:e}");
}

#[test]
fn single_state_limits_are_rhf_and_rohf() {
    let atoms = vec![
        Atom::new("O", [0.1, -0.2, 0.3]),
        Atom::new("H", [1.5, 0.7, 0.4]),
        Atom::new("H", [-0.4, 1.3, -0.2]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let rhf = run_scf(&atoms, &basis, 200, 1e-12).unwrap();
    let closed = StateConfiguration::from_occupations(&[2.0, 2.0, 2.0, 2.0, 2.0]).unwrap();
    let sa =
        run_state_averaged_hf_with_options(&atoms, &basis, &[closed], &[1.0], &tight()).unwrap();
    assert!(
        (sa.sa_energy - rhf.total_energy).abs() < 1e-9,
        "{} vs RHF {}",
        sa.sa_energy,
        rhf.total_energy
    );

    // One high-spin determinant is ROHF.
    let triplet = StateConfiguration::new((0..6).collect(), (0..4).collect());
    let rohf = run_rohf(&atoms, &basis, 0, 3, 300, 1e-12).unwrap();
    let sa =
        run_state_averaged_hf_with_options(&atoms, &basis, &[triplet], &[1.0], &tight()).unwrap();
    assert!(
        (sa.sa_energy - rohf.total_energy).abs() < 1e-8,
        "{} vs ROHF {}",
        sa.sa_energy,
        rohf.total_energy
    );
}

#[test]
fn ensemble_energy_is_stationary_and_symmetric() {
    let case = data()["cases"][0].clone();
    assert_eq!(case["name"], "B_atom_p_ensemble");
    let atoms = atoms(&case);
    let basis = build_basis(&atoms, &get_builtin_basis("6-31g").unwrap());
    let states: Vec<StateConfiguration> = (2..5)
        .map(|p| StateConfiguration::new(vec![0, 1, p], vec![0, 1]))
        .collect();
    let r = run_state_averaged_hf_with_options(&atoms, &basis, &states, &[1.0 / 3.0; 3], &tight())
        .unwrap();
    // The equally weighted p ensemble keeps the three 2p determinants degenerate.
    for e in &r.state_energies {
        assert!((e - r.state_energies[0]).abs() < 1e-8);
    }
    // Reordering the states does not change the ensemble.
    let reversed: Vec<StateConfiguration> = states.iter().rev().cloned().collect();
    let again =
        run_state_averaged_hf_with_options(&atoms, &basis, &reversed, &[1.0 / 3.0; 3], &tight())
            .unwrap();
    assert!((again.sa_energy - r.sa_energy).abs() < 1e-10);

    // Central differences of the exact ensemble energy along random orbital rotations vanish.
    let c = &r.mo_coefficients;
    let energy = |u: &Matrix| -> f64 { ensemble_energy(&atoms, &basis, &states, &c.dot(u)) };
    let n = c.cols;
    let mut x = Matrix::zeros(n, n);
    for p in 0..n {
        for q in 0..p {
            let v = ((p * 7 + q * 3) % 11) as f64 / 11.0 - 0.5;
            x[(p, q)] = v;
            x[(q, p)] = -v;
        }
    }
    let h = 1e-4;
    let exp = |t: f64| hf_rs::post_hf::mr_dsrg::cumulants::exp_antisymmetric(&x.scale(t));
    let derivative = (energy(&exp(h)) - energy(&exp(-h))) / (2.0 * h);
    assert!(
        derivative.abs() < 1e-6,
        "directional derivative {derivative}"
    );
    assert!((energy(&Matrix::identity(n)) - r.sa_energy).abs() < 1e-10);
}

/// Equal-weight ensemble energy of explicit determinants from orbitals `c` (test-side).
fn ensemble_energy(
    atoms: &[Atom],
    basis: &[hf_rs::ContractedGaussian],
    states: &[StateConfiguration],
    c: &Matrix,
) -> f64 {
    let (_, _, _, h) = hf_rs::scf::compute_one_electron_integrals(atoms, basis);
    let eri = hf_rs::scf::compute_eri_tensor(basis);
    let nuclear = hf_rs::scf::compute_nuclear_repulsion(atoms);
    let n = c.rows;
    let density = |occupied: &[usize]| {
        let mut p = Matrix::zeros(n, n);
        for &k in occupied {
            for i in 0..n {
                for j in 0..n {
                    p[(i, j)] += c[(i, k)] * c[(j, k)];
                }
            }
        }
        p
    };
    let dot =
        |a: &Matrix, b: &Matrix| -> f64 { a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum() };
    states
        .iter()
        .map(|s| {
            let (pa, pb) = (density(&s.alpha), density(&s.beta));
            let (fa, fb) = hf_rs::build_uhf_fock_matrices(&h, &pa, &pb, &eri, n);
            0.5 * (dot(&pa.add(&pb), &h) + dot(&pa, &fa) + dot(&pb, &fb)) + nuclear
        })
        .sum::<f64>()
        / states.len() as f64
}

#[test]
fn invalid_ensembles_are_rejected() {
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -0.7]),
        Atom::new("H", [0.0, 0.0, 0.7]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let a = StateConfiguration::new(vec![0], vec![0]);
    let b = StateConfiguration::new(vec![0, 1], vec![1]);
    assert!(
        run_state_averaged_hf(&atoms, &basis, &[a.clone(), b], &[0.5, 0.5], 100, 1e-10).is_err()
    );
    assert!(run_state_averaged_hf(&atoms, &basis, std::slice::from_ref(&a), &[0.9], 100, 1e-10).is_err());
    assert!(StateConfiguration::from_occupations(&[1.5, 0.5]).is_err());
    let repeated = StateConfiguration::new(vec![0, 0], vec![]);
    assert!(run_state_averaged_hf(&atoms, &basis, &[repeated], &[1.0], 100, 1e-10).is_err());
}
