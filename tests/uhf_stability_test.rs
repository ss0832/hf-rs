// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! AO-direct UHF internal stability against dense PySCF-integral orbital Hessians
//! (tests/reference/gen_uhf_stability.py), finite-difference curvature along the eigenvector,
//! the RHF singlet/triplet limit, and deterministic broken-symmetry instability following.

use hf_rs::uhf_stability::uhf_internal_stability;
use hf_rs::{
    Atom, Matrix, UHFResult, analyze_rhf_internal_stability, analyze_rhf_stability, build_basis,
    build_uhf_fock_matrices, get_builtin_basis, run_rhf_with_stability_rotation, run_scf, run_uhf,
};

fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/uhf_stability_reference.json")).unwrap()
}

fn solve(case: &serde_json::Value) -> UHFResult {
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
    let name = match case["basis_constant"].as_str().unwrap() {
        "STO_3G_GBS" => "sto-3g",
        "BASIS_6_31G_GBS" => "6-31g",
        other => panic!("unexpected basis {other}"),
    };
    let basis = build_basis(&atoms, &get_builtin_basis(name).unwrap());
    let two_s = case["spin_2s"].as_u64().unwrap() as usize;
    let broken = case["broken_symmetry"].as_bool().unwrap();
    let charge = case["charge"].as_i64().unwrap() as i32;
    let uhf = run_uhf(&atoms, &basis, charge, two_s + 1, broken, 500, 1e-12).unwrap();
    if case["name"].as_str().unwrap().ends_with("closed_shell") {
        // The default RHF guess ends at an internally unstable closed-shell point here
        // (singlet instability -0.069 Eh); follow it to the stable RHF of the oracle.
        let (rhf, stability, _) =
            run_rhf_with_stability_rotation(&atoms, &basis, charge, 200, 1e-12, 8, 0.3).unwrap();
        assert!(stability.is_stable);
        closed_shell_point(uhf, &rhf)
    } else {
        uhf
    }
}

/// The RHF determinant as a UHF stationary point (`F_alpha = F_beta = F_RHF`).
fn closed_shell_point(mut uhf: UHFResult, rhf: &hf_rs::HFResult) -> UHFResult {
    assert!(rhf.converged && uhf.n_alpha == rhf.n_occ && uhf.n_beta == rhf.n_occ);
    uhf.mo_coefficients_alpha = rhf.mo_coefficients.clone();
    uhf.mo_coefficients_beta = rhf.mo_coefficients.clone();
    uhf.fock_matrix_alpha = rhf.fock_matrix.clone();
    uhf.fock_matrix_beta = rhf.fock_matrix.clone();
    uhf.density_matrix_alpha = rhf.density_matrix.scale(0.5);
    uhf.density_matrix_beta = rhf.density_matrix.scale(0.5);
    uhf.total_energy = rhf.total_energy;
    uhf.s_squared = 0.0;
    uhf
}

fn energy(uhf: &UHFResult, ca: &Matrix, cb: &Matrix) -> f64 {
    let pa = ca.occupied_density(uhf.n_alpha, 1.0);
    let pb = cb.occupied_density(uhf.n_beta, 1.0);
    let n = pa.rows;
    let (fa, fb) = build_uhf_fock_matrices(&uhf.core_hamiltonian, &pa, &pb, &uhf.eri, n);
    let dot = |a: &Matrix, b: &Matrix| -> f64 { a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum() };
    0.5 * (dot(&pa.add(&pb), &uhf.core_hamiltonian) + dot(&pa, &fa) + dot(&pb, &fb))
        + uhf.nuclear_repulsion_energy
}

fn rotated(c: &Matrix, n_occ: usize, kappa: &Matrix, t: f64) -> Matrix {
    let n_mo = c.cols;
    let mut x = Matrix::zeros(n_mo, n_mo);
    for a in 0..kappa.rows {
        for i in 0..kappa.cols {
            x[(n_occ + a, i)] = t * kappa[(a, i)];
            x[(i, n_occ + a)] = -t * kappa[(a, i)];
        }
    }
    c.dot(&hf_rs::post_hf::mr_dsrg::cumulants::exp_antisymmetric(&x))
}

#[test]
fn lowest_uhf_hessian_eigenvalues_match_dense_oracle() {
    let mut worst = 0f64;
    for case in data()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let uhf = solve(case);
        assert!(
            (uhf.total_energy - case["e_uhf"].as_f64().unwrap()).abs() < 1e-8,
            "{name}: UHF {} vs {}",
            uhf.total_energy,
            case["e_uhf"]
        );
        let stability = uhf_internal_stability(&uhf).unwrap();
        let reference = case["lowest_eigenvalues"][0].as_f64().unwrap();
        worst = worst.max((stability.lowest_eigenvalue - reference).abs());
        assert!(
            (stability.lowest_eigenvalue - reference).abs() < 1e-8,
            "{name}: lowest eigenvalue {} vs {reference}",
            stability.lowest_eigenvalue
        );
        let norm: f64 = stability.alpha.data.iter().chain(&stability.beta.data).map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }
    println!("UHF stability: max |d lambda| {worst:e}");
}

#[test]
fn eigenvalue_equals_energy_curvature_along_its_eigenvector() {
    // d^2 E / dt^2 = 2 lambda along C exp(t X) at a stationary point.
    for case in data()["cases"].as_array().unwrap().iter().filter(|c| {
        c["name"].as_str().unwrap().starts_with("C2H4") || c["name"] == "CH2_triplet"
    }) {
        let uhf = solve(case);
        let st = uhf_internal_stability(&uhf).unwrap();
        let e = |t: f64| {
            energy(
                &uhf,
                &rotated(&uhf.mo_coefficients_alpha, uhf.n_alpha, &st.alpha, t),
                &rotated(&uhf.mo_coefficients_beta, uhf.n_beta, &st.beta, t),
            )
        };
        let h = 1e-3;
        let e0 = e(0.0);
        assert!((e0 - uhf.total_energy).abs() < 1e-10);
        let curvature = (e(h) + e(-h) - 2.0 * e0) / (h * h);
        let fine = (e(h / 2.0) + e(-h / 2.0) - 2.0 * e0) / (h * h / 4.0);
        assert!(
            (curvature - 2.0 * st.lowest_eigenvalue).abs() < 1e-5,
            "{}: curvature {curvature} (refined {fine}) vs 2 lambda {}",
            case["name"],
            2.0 * st.lowest_eigenvalue
        );
    }
}

#[test]
fn closed_shell_uhf_hessian_is_the_rhf_singlet_triplet_union() {
    let atoms = vec![
        Atom::new("C", [0.0, 0.0, 1.266]),
        Atom::new("C", [0.0, 0.0, -1.266]),
        Atom::new("H", [0.0, 1.744, 2.336]),
        Atom::new("H", [0.0, -1.744, 2.336]),
        Atom::new("H", [1.744, 0.0, -2.336]),
        Atom::new("H", [-1.744, 0.0, -2.336]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("6-31g").unwrap());
    let rhf = run_scf(&atoms, &basis, 200, 1e-12).unwrap();
    let closed = closed_shell_point(run_uhf(&atoms, &basis, 0, 1, false, 500, 1e-12).unwrap(), &rhf);
    let dot = |a: &Matrix, b: &Matrix| -> f64 { a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum() };
    let (pa, pb) = (&closed.density_matrix_alpha, &closed.density_matrix_beta);
    let (fa, _) = build_uhf_fock_matrices(&closed.core_hamiltonian, pa, pb, &closed.eri, pa.rows);
    assert!(dot(&fa.sub(&closed.fock_matrix_alpha), &fa.sub(&closed.fock_matrix_alpha)).sqrt() < 1e-10);
    let lowest = uhf_internal_stability(&closed).unwrap().lowest_eigenvalue;
    let triplet = analyze_rhf_stability(&rhf, rhf.n_occ).lowest_eigenvalue;
    let singlet = analyze_rhf_internal_stability(&rhf, rhf.n_occ).lowest_eigenvalue;
    assert!(triplet < 0.0, "twisted ethylene RHF has a triplet instability");
    assert!(
        (lowest - triplet.min(singlet)).abs() < 1e-8,
        "UHF {lowest} vs RHF singlet {singlet} / triplet {triplet}"
    );
}

#[test]
fn broken_symmetry_search_ends_at_the_stable_solution_deterministically() {
    // Degenerate frontier orbitals make the broken-symmetry start depend on rounding; without
    // instability following the SCF ends at either -77.9306 or the unstable -77.8305 Eh.
    let case = data()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "C2H4_twisted_broken_symmetry")
        .unwrap()
        .clone();
    for _ in 0..3 {
        let uhf = solve(&case);
        assert!((uhf.total_energy - case["e_uhf"].as_f64().unwrap()).abs() < 1e-8);
        assert!(uhf_internal_stability(&uhf).unwrap().lowest_eigenvalue > 0.1);
    }
}
