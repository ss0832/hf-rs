// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Finite-potential-basis exchange-only OEP against an independent PySCF/SciPy oracle
//! (tests/reference/gen_oep.py): exact Gaussian potential matrices through G functions,
//! arbitrary-coefficient energies and analytic derivatives, optimized energies and densities,
//! the two-electron and complete-occupation limits, the HF lower bound, and explicit failure
//! where every descent closes the occupied/virtual gap.

use hf_rs::hf_variants::local_potential::gaussian_potential_matrix;
use hf_rs::{
    Atom, ContractedGaussian, ConvAlgo, Matrix, OepOptions, build_basis,
    evaluate_oep_hf_from_reference, get_builtin_basis, parse_gaussian94, run_oep_hf,
    run_oep_hf_from_reference, run_scf_general,
};

fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/oep_reference.json")).unwrap()
}

fn atoms(case: &serde_json::Value) -> Vec<Atom> {
    let mut atoms: Vec<Atom> = case["atoms"]
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
    if !case["ecp"].is_null() {
        // The semilocal He model of tests/reference/gen_mr_molecular.py.
        use hf_rs::ecp::{EcpDef, EcpTerm};
        let term = |exponent, coefficient| EcpTerm { power: 2, exponent, coefficient };
        atoms[0].ecp = Some(std::sync::Arc::new(EcpDef {
            symbol: "He".into(),
            n_core: 0,
            l_max: 2,
            local: vec![term(0.7, 0.17)],
            semilocal: vec![vec![term(0.53, 0.11)], vec![term(0.61, -0.07)]],
        }));
    }
    atoms
}

fn basis(case: &serde_json::Value, atoms: &[Atom]) -> Vec<ContractedGaussian> {
    let map = if let Some(gbs) = case["gaussian94"].as_str() {
        parse_gaussian94(gbs)
    } else {
        get_builtin_basis(case["basis_name"].as_str().unwrap()).unwrap()
    };
    build_basis(atoms, &map)
}

fn options(case: &serde_json::Value) -> OepOptions {
    OepOptions {
        exponents: case["exponents"].as_array().unwrap().iter().map(|e| e.as_f64().unwrap()).collect(),
        max_iterations: 5000,
        energy_tol: 1e-13,
        gradient_tol: 1e-9,
        ..Default::default()
    }
}

fn vector(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn max_difference(a: &Matrix, b: &serde_json::Value) -> f64 {
    let mut worst = 0f64;
    for i in 0..a.rows {
        for j in 0..a.cols {
            worst = worst.max((a[(i, j)] - b[i][j].as_f64().unwrap()).abs());
        }
    }
    worst
}

fn reference(atoms: &[Atom], basis: &[ContractedGaussian], case: &serde_json::Value) -> hf_rs::HFResult {
    let charge = case["charge"].as_i64().unwrap() as i32;
    let hf = run_scf_general(atoms, basis, charge, 300, 1e-13, ConvAlgo::Adiissoscf, None).unwrap();
    assert!((hf.total_energy - case["reference_energy"].as_f64().unwrap()).abs() < 1e-9);
    hf
}

#[test]
fn every_gaussian_potential_matrix_entry_matches_three_center_overlaps() {
    let data = data();
    let mut worst = 0f64;
    let mut entries = 0;
    for case in data["cases"].as_array().unwrap() {
        let atoms = atoms(case);
        let basis = basis(case, &atoms);
        let exponents = vector(&case["exponents"]);
        let mut k = 0;
        for atom in &atoms {
            for &exponent in &exponents {
                let native = gaussian_potential_matrix(&basis, atom.position, exponent).unwrap();
                let error = max_difference(&native, &case["operators"][k]);
                assert!(error < 1e-12, "{}: potential {k} error {error:e}", case["name"]);
                worst = worst.max(error);
                entries += native.data.len();
                k += 1;
            }
        }
    }
    // Every Cartesian S/P/D/F/G component pair, in unit-normalized AOs.
    let high = &data["high_angular_momentum"];
    let atoms = atoms(high);
    let basis = build_basis(&atoms, &parse_gaussian94(high["gaussian94"].as_str().unwrap()));
    let n = basis.len();
    let mut overlap = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            overlap[(i, j)] = hf_rs::integrals::contracted_overlap(&basis[i], &basis[j]);
        }
    }
    for case in high["cases"].as_array().unwrap() {
        let center = std::array::from_fn(|k| case["center"][k].as_f64().unwrap());
        let native = gaussian_potential_matrix(&basis, center, case["exponent"].as_f64().unwrap()).unwrap();
        let mut normalized = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                normalized[(i, j)] = native[(i, j)] / (overlap[(i, i)] * overlap[(j, j)]).sqrt();
            }
        }
        let error = max_difference(&normalized, &case["matrix"]);
        assert!(error < 1e-12, "high angular momentum exponent {}: error {error:e}", case["exponent"]);
        worst = worst.max(error);
        entries += n * n;
    }
    assert!(worst < 1e-12, "max potential-matrix error {worst:e}");
    println!("OEP potentials: {entries} entries, max error {worst:e}");
}

#[test]
fn arbitrary_coefficient_energies_and_derivatives_match_oracle() {
    let data = data();
    let (mut worst_e, mut worst_g) = (0f64, 0f64);
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let atoms = atoms(case);
        let basis = basis(case, &atoms);
        let hf = reference(&atoms, &basis, case);
        let parameters = vector(&case["trial_parameters"]);
        let r = evaluate_oep_hf_from_reference(&atoms, &basis, &hf, &parameters, &options(case)).unwrap();
        let de = (r.total_energy - case["trial_energy"].as_f64().unwrap()).abs();
        worst_e = worst_e.max(de);
        assert!(de < 1e-10, "{name}: trial energy error {de:e}");
        // Analytic oracle derivatives and refined five-point energy differences.
        for (key, tolerance) in [("trial_gradient", 1e-9), ("trial_gradient_fd", 3e-8)] {
            for (native, reference) in r.parameter_gradient.iter().zip(vector(&case[key])) {
                worst_g = worst_g.max((native - reference).abs());
                assert!((native - reference).abs() < tolerance, "{name}: {key} {native} vs {reference}");
            }
        }
        assert!(max_difference(&r.density_matrix, &case["trial_density"]) < 1e-9, "{name}: density");
        assert!(max_difference(&r.ks_hamiltonian, &case["trial_ks"]) < 1e-10, "{name}: KS matrix");
        assert!(max_difference(&r.hf_fock_matrix, &case["trial_fock"]) < 1e-9, "{name}: HF Fock");
        for (e, reference) in r.orbital_energies.iter().zip(vector(&case["trial_orbital_energies"])) {
            assert!((e - reference).abs() < 1e-10, "{name}: KS orbital energy");
        }
    }
    println!("OEP trial points: max |dE| {worst_e:e}, max |d gradient| {worst_g:e}");
}

#[test]
fn optimized_energies_and_densities_match_independent_minimization() {
    let data = data();
    let mut worst = 0f64;
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let atoms = atoms(case);
        let basis = basis(case, &atoms);
        let hf = reference(&atoms, &basis, case);
        let r = run_oep_hf_from_reference(&atoms, &basis, &hf, &options(case)).unwrap();
        let _ = &run_oep_hf; // the full driver is exercised in the failure tests
        assert!(r.converged && r.gradient_norm <= 1e-9, "{name}: gradient {:e}", r.gradient_norm);
        let de = (r.total_energy - case["energy"].as_f64().unwrap()).abs();
        worst = worst.max(de);
        assert!(de < 1e-9, "{name}: {} vs {}", r.total_energy, case["energy"]);
        // Potential coefficients need not be unique; the occupied space and density are.
        assert!(max_difference(&r.density_matrix, &case["density"]) < 1e-6, "{name}: density");
        // A single determinant cannot lie below RHF.
        assert!(r.total_energy >= hf.total_energy - 1e-10, "{name}: below RHF");
        assert!(r.ks_residual_norm < 1e-9, "{name}: KS determinant is not self-consistent");
        let n_electrons = 2 * hf.n_occ;
        if n_electrons == 2 || hf.n_occ == hf.mo_coefficients.cols {
            // Two electrons: -J/2 is local and the Fermi-Amaldi baseline is exact;
            // complete occupation: one determinant. Both reproduce RHF.
            assert!((r.total_energy - hf.total_energy).abs() < 1e-10, "{name}: RHF limit");
        } else {
            assert!(r.total_energy > hf.total_energy + 1e-6, "{name}: OEP should not reach RHF");
        }
    }
    println!("OEP optimized energies: max |dE| {worst:e}");
}

#[test]
fn closing_gap_without_minimizer_fails_explicitly() {
    // LiH/STO-3G with s potentials (0.5, 2.0) on both atoms: every independent descent lowers
    // the energy while the Kohn-Sham gap collapses; no open-gap stationary point is reached.
    let data = data();
    let case = &data["ill_posed_cases"][0];
    assert_eq!(case["name"], "LiH_sto3g");
    for descent in case["descents"].as_array().unwrap() {
        assert!(descent["gradient"].as_f64().unwrap() > 1e-4);
    }
    let atoms = atoms(case);
    let basis = basis(case, &atoms);
    let error = run_oep_hf(&atoms, &basis, 0, &options(case)).unwrap_err();
    assert!(error.contains("gap") || error.contains("did not converge"), "{error}");
}

#[test]
fn invalid_controls_and_memory_budget_are_rejected() {
    let data = data();
    let case = &data["cases"][0];
    let atoms = atoms(case);
    let basis = basis(case, &atoms);
    for bad in [
        OepOptions { exponents: vec![], ..Default::default() },
        OepOptions { exponents: vec![1.0, 1.0], ..Default::default() },
        OepOptions { exponents: vec![-1.0], ..Default::default() },
        OepOptions { minimum_gap: 0.0, ..Default::default() },
        OepOptions { potential_budget_bytes: 8, ..Default::default() },
    ] {
        assert!(run_oep_hf(&atoms, &basis, 0, &bad).is_err());
    }
    // Odd electron counts are outside the closed-shell model.
    assert!(run_oep_hf(&atoms, &basis, 1, &OepOptions::default()).is_err());
    assert!(gaussian_potential_matrix(&basis, [0.0; 3], 0.0).is_err());
}
