// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! TCSCF (one-pair GVB = singlet CASSCF(2,2)) against PySCF CASSCF(2,2) energies, active natural
//! occupations and analytic nuclear gradients (tests/reference/gen_tcscf.py), plus exact limits.

use hf_rs::post_hf::PostHFMethod;
use hf_rs::{build_basis, compute_tcscf_gradient, get_builtin_basis, run_scf_general, run_tcscf_from_reference, Atom, ConvAlgo, FCI};

fn basis_name(constant: &str) -> &'static str {
    match constant {
        "STO_3G_GBS" => "sto-3g",
        "BASIS_6_31G_GBS" => "6-31g",
        other => panic!("{other}"),
    }
}

#[test]
fn tcscf_energies_occupations_and_gradients_match_pyscf_casscf22() {
    let data: serde_json::Value = serde_json::from_str(include_str!("data/tcscf_reference.json")).unwrap();
    let (mut worst_e, mut worst_n, mut worst_g, mut count) = (0f64, 0f64, 0f64, 0usize);
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let atoms: Vec<Atom> = case["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| Atom::new(a["symbol"].as_str().unwrap(), std::array::from_fn(|k| a["position"][k].as_f64().unwrap())))
            .collect();
        let basis = build_basis(&atoms, &get_builtin_basis(basis_name(case["basis_constant"].as_str().unwrap())).unwrap());
        let hf = run_scf_general(&atoms, &basis, 0, 500, 1e-12, ConvAlgo::Adiissoscf, None).unwrap();
        assert!((hf.total_energy - case["e_rhf"].as_f64().unwrap()).abs() < 1e-8, "{name} RHF");
        let g = case["bonding"].as_u64().unwrap() as usize;
        let u = case["antibonding"].as_u64().unwrap() as usize;
        let r = run_tcscf_from_reference(&hf, g, u, 300, 1e-12).unwrap();
        let de = (r.tcscf_energy - case["e_tcscf"].as_f64().unwrap()).abs();
        worst_e = worst_e.max(de);
        assert!(de < 1e-8, "{name}: TCSCF {} vs PySCF CASSCF(2,2) {}", r.tcscf_energy, case["e_tcscf"]);
        assert!(r.tcscf_energy <= hf.total_energy + 1e-10, "TCSCF is variational relative to RHF");
        assert!((r.c1 * r.c1 + r.c2 * r.c2 - 1.0).abs() < 1e-10);
        assert!((r.natural_occupations[0] + r.natural_occupations[1] - 2.0).abs() < 1e-10);
        for k in 0..2 {
            let dn = (r.natural_occupations[k] - case["natural_occupations"][k].as_f64().unwrap()).abs();
            worst_n = worst_n.max(dn);
            assert!(dn < 1e-5, "{name} occupation {k}");
        }
        let grad = compute_tcscf_gradient(&atoms, &basis, &hf, &r);
        for (a, row) in case["gradient"].as_array().unwrap().iter().enumerate() {
            for axis in 0..3 {
                let dg = (grad[a][axis] - row[axis].as_f64().unwrap()).abs();
                worst_g = worst_g.max(dg);
                count += 1;
                assert!(dg < 1e-6, "{name} gradient {a}/{axis}: {} vs {}", grad[a][axis], row[axis]);
            }
        }
        for axis in 0..3 {
            assert!(grad.iter().map(|x| x[axis]).sum::<f64>().abs() < 1e-9, "{name} translation");
        }
    }
    println!("TCSCF: max |dE| {worst_e:e}, max |dn| {worst_n:e}, {count} gradient components max {worst_g:e}");
}

#[test]
fn tcscf_of_minimal_basis_h2_is_full_ci() {
    let atoms = vec![Atom::new("H", [0.0, 0.0, -1.5]), Atom::new("H", [0.0, 0.0, 1.5])];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_scf_general(&atoms, &basis, 0, 200, 1e-12, ConvAlgo::Cdiis, None).unwrap();
    let r = run_tcscf_from_reference(&hf, 0, 1, 200, 1e-12).unwrap();
    let fci = FCI::new(&hf).compute().unwrap();
    assert!((r.tcscf_energy - fci.total_energy).abs() < 1e-10, "{} vs FCI {}", r.tcscf_energy, fci.total_energy);
    assert!(run_tcscf_from_reference(&hf, 1, 1, 50, 1e-10).is_err(), "bonding orbital must be occupied");
}
