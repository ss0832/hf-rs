// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! sigma-SCF against full-CI-space variance minimizations (tests/reference/gen_sigma_scf.py),
//! the exact one-electron limit (eigenstates have zero variance), the far-below-spectrum
//! limit (Hartree-Fock ground state) and stationarity of the closed-form variance.

use hf_rs::{
    Atom, SigmaScfOptions, StateConfiguration, build_basis, get_builtin_basis, run_scf,
    run_sigma_scf, run_sigma_scf_with_options,
};

fn tight() -> SigmaScfOptions {
    SigmaScfOptions {
        max_iterations: 3000,
        gradient_tol: 1e-8,
        variance_tol: 1e-14,
    }
}

fn h2() -> Vec<Atom> {
    vec![
        Atom::new("H", [0.0, 0.0, -0.7]),
        Atom::new("H", [0.1, 0.0, 0.7]),
    ]
}

#[test]
fn sigma_scf_matches_full_ci_space_variance_minimization() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/sigma_scf_reference.json")).unwrap();
    let mut worst = 0f64;
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let atoms: Vec<Atom> = case["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                Atom::new(
                    a["symbol"].as_str().unwrap(),
                    std::array::from_fn(|k| a["position"][k].as_f64().unwrap()),
                )
            })
            .collect();
        let basis_name = match case["constant"].as_str().unwrap() {
            "STO_3G_GBS" => "sto-3g",
            _ => "6-31g",
        };
        let basis = build_basis(&atoms, &get_builtin_basis(basis_name).unwrap());
        let list = |key: &str| -> Vec<usize> {
            case[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect()
        };
        let configuration = StateConfiguration::new(list("alpha"), list("beta"));
        let omega = case["omega"].as_f64().unwrap();
        let r =
            run_sigma_scf_with_options(&atoms, &basis, omega, &configuration, &tight()).unwrap();
        let (variance, energy) = (
            case["variance"].as_f64().unwrap(),
            case["energy"].as_f64().unwrap(),
        );
        worst = worst.max((r.energy_variance - variance).abs());
        assert!(
            (r.energy_variance - variance).abs() < 1e-9,
            "{name}: variance {} vs {variance}",
            r.energy_variance
        );
        assert!(
            (r.total_energy - energy).abs() < 1e-6,
            "{name}: energy {} vs {energy}",
            r.total_energy
        );
        assert!((r.intrinsic_variance - (variance - (energy - omega).powi(2))).abs() < 1e-8);
    }
    println!("sigma-SCF: max |d variance| {worst:e}");
}

#[test]
fn one_electron_eigenstates_have_zero_variance() {
    // H2+: the determinant is one orbital, and H is the core Hamiltonian.
    let atoms = h2();
    let basis = build_basis(&atoms, &get_builtin_basis("6-31g").unwrap());
    let (s, _, _, h) = hf_rs::scf::compute_one_electron_integrals(&atoms, &basis);
    let x = s.canonical_orthogonalizer(1e-8).unwrap();
    let (eigenvalues, _) = x.transpose().dot(&h).dot(&x).eigh();
    let nuclear = hf_rs::scf::compute_nuclear_repulsion(&atoms);
    for target in [1usize, 2] {
        let exact = eigenvalues[target] + nuclear;
        let configuration = StateConfiguration::new(vec![target], vec![]);
        let r = run_sigma_scf_with_options(&atoms, &basis, exact + 0.05, &configuration, &tight())
            .unwrap();
        assert!(
            r.intrinsic_variance.abs() < 1e-10,
            "intrinsic variance {}",
            r.intrinsic_variance
        );
        assert!(
            (r.total_energy - exact).abs() < 1e-9,
            "{} vs eigenvalue {exact}",
            r.total_energy
        );
        assert!((r.energy_variance - 0.05f64.powi(2)).abs() < 1e-10);
    }
}

#[test]
fn target_far_below_the_spectrum_approaches_hartree_fock() {
    // sigma^2 = <H^2> - 2 omega <H> + omega^2: for omega -> -infinity the minimizer tends to
    // the HF determinant, with an energy error O(1/omega^2).
    let atoms = h2();
    let basis = build_basis(&atoms, &get_builtin_basis("6-31g").unwrap());
    let hf = run_scf(&atoms, &basis, 200, 1e-12).unwrap();
    let closed = StateConfiguration::from_occupations(&[2.0]).unwrap();
    let mut previous = f64::INFINITY;
    for omega in [-1e2, -1e3, -1e4] {
        let r = run_sigma_scf_with_options(&atoms, &basis, omega, &closed, &tight()).unwrap();
        let error = r.total_energy - hf.total_energy;
        assert!(error >= -1e-12, "variational HF bound");
        assert!(
            error < previous / 50.0 || error < 1e-11,
            "omega {omega}: error {error}"
        );
        previous = error;
    }
    assert!(previous < 1e-9);
}

#[test]
fn legacy_occupation_interface_and_invalid_input() {
    let atoms = h2();
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    // [1, 1]: one alpha electron in each orbital, the M_s = 1 triplet determinant, which is an
    // exact eigenstate in the minimal basis (variance (E - omega)^2 only).
    let r = run_sigma_scf(&atoms, &basis, -0.5, &[1.0, 1.0], 500, 1e-12).unwrap();
    assert!(r.intrinsic_variance.abs() < 1e-10);
    assert!(run_sigma_scf(&atoms, &basis, -0.5, &[1.5, 0.5], 100, 1e-10).is_err());
    assert!(run_sigma_scf(&atoms, &basis, f64::NAN, &[2.0], 100, 1e-10).is_err());
}
