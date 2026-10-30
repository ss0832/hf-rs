// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Cholesky-decomposition HF: pivot sequences, factor counts and energies against an
//! independent dense NumPy factorization of PySCF integrals (tests/reference/gen_cd_hf.py),
//! the rigorous reconstructed-integral bound, and the delta -> 0 limit to conventional RHF.

use hf_rs::scf::{compute_eri_tensor, eri_index};
use hf_rs::{
    assign_builtin_ecps, build_basis, get_builtin_basis, pivoted_cholesky_eri, run_cd_hf, run_scf_general, Atom,
    ConvAlgo, CD_HF_FACTOR_BUDGET,
};

fn basis_name(constant: &str) -> &'static str {
    match constant {
        "STO_3G_GBS" => "sto-3g",
        "BASIS_6_31G_GBS" => "6-31g",
        other => panic!("{other}"),
    }
}

fn cases() -> Vec<(serde_json::Value, Vec<Atom>, &'static str)> {
    let data: serde_json::Value = serde_json::from_str(include_str!("data/cd_hf_reference.json")).unwrap();
    data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let atoms = c["atoms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| Atom::new(a["symbol"].as_str().unwrap(), std::array::from_fn(|k| a["position"][k].as_f64().unwrap())))
                .collect();
            (c.clone(), atoms, basis_name(c["basis_constant"].as_str().unwrap()))
        })
        .collect()
}

#[test]
fn cholesky_pivots_counts_and_cd_hf_energies_match_the_independent_dense_factorization() {
    let mut worst = 0f64;
    let mut n_checked = 0;
    for (case, atoms, bname) in cases() {
        let basis = build_basis(&atoms, &get_builtin_basis(bname).unwrap());
        for t in case["thresholds"].as_array().unwrap() {
            let delta = t["threshold"].as_f64().unwrap();
            let (tensor, info) = pivoted_cholesky_eri(&basis, delta, CD_HF_FACTOR_BUDGET).unwrap();
            let pivots: Vec<usize> = t["pivots"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            assert_eq!(tensor.rank(), t["n_vectors"].as_u64().unwrap() as usize, "{} delta {delta}", case["name"]);
            assert_eq!(info.pivots, pivots, "{} delta {delta}: pivot sequence", case["name"]);
            assert!(info.max_residual <= delta);
            assert!((info.max_residual - t["max_residual"].as_f64().unwrap()).abs() < 1e-12);
            let r = run_cd_hf(&atoms, &basis, 0, delta, 300, 1e-12).unwrap();
            assert!(r.converged);
            let error = (r.total_energy - t["e_cd_hf"].as_f64().unwrap()).abs();
            worst = worst.max(error);
            n_checked += 1;
            assert!(error < 1e-9, "{} delta {delta}: CD-HF {} vs {}", case["name"], r.total_energy, t["e_cd_hf"]);
            assert_eq!(r.n_cholesky_vectors, tensor.rank());
        }
    }
    assert_eq!(n_checked, 9);
    println!("CD-HF: {n_checked} factorizations, max energy error {worst:e}");
}

#[test]
fn every_reconstructed_integral_obeys_the_residual_bound() {
    // |(pq|rs) - sum_K L_K(pq) L_K(rs)| <= sqrt(D_pq D_rs) <= max residual diagonal.
    let (_, atoms, bname) = cases().into_iter().find(|c| c.0["name"] == "water").unwrap();
    let basis = build_basis(&atoms, &get_builtin_basis(bname).unwrap());
    let exact = compute_eri_tensor(&basis);
    let n = basis.len();
    for delta in [1e-2, 1e-4, 1e-7] {
        let (t, info) = pivoted_cholesky_eri(&basis, delta, CD_HF_FACTOR_BUDGET).unwrap();
        let mut worst = 0f64;
        for p in 0..n {
            for q in 0..=p {
                for r in 0..n {
                    for s in 0..=r {
                        worst = worst.max((t.eri(p, q, r, s) - exact[eri_index(p, q, r, s)]).abs());
                    }
                }
            }
        }
        assert!(worst <= info.max_residual + 1e-13, "delta {delta}: error {worst:e} > residual {:e}", info.max_residual);
    }
}

#[test]
fn cd_hf_converges_to_conventional_rhf_including_ecps() {
    let mut systems: Vec<(Vec<Atom>, &str)> = cases().into_iter().map(|(_, a, b)| (a, b)).collect();
    let mut hi = vec![Atom::new("H", [0.0, 0.0, 0.0]), Atom::new("I", [0.2, -0.1, 3.05])];
    assign_builtin_ecps(&mut hi, "def2-svp");
    systems.push((hi, "def2-svp"));
    for (atoms, bname) in systems {
        let basis = build_basis(&atoms, &get_builtin_basis(bname).unwrap());
        let conventional = run_scf_general(&atoms, &basis, 0, 300, 1e-12, ConvAlgo::Cdiis, None).unwrap();
        let cd = run_cd_hf(&atoms, &basis, 0, 1e-12, 300, 1e-12).unwrap();
        assert!((cd.total_energy - conventional.total_energy).abs() < 1e-9, "{bname}: {} vs {}", cd.total_energy, conventional.total_energy);
    }
}

#[test]
fn cd_hf_rejects_invalid_requests() {
    let atoms = vec![Atom::new("H", [0.0, 0.0, -0.7]), Atom::new("H", [0.0, 0.0, 0.7])];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    assert!(run_cd_hf(&atoms, &basis, 1, 1e-6, 50, 1e-10).is_err(), "odd electron count");
    assert!(run_cd_hf(&atoms, &basis, 0, 10.0, 50, 1e-10).is_err(), "threshold above every diagonal");
    assert!(pivoted_cholesky_eri(&basis, -1.0, CD_HF_FACTOR_BUDGET).is_err());
    assert!(pivoted_cholesky_eri(&basis, 1e-8, 16).is_err(), "budget");
}
