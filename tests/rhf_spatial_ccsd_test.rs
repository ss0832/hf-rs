// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::post_hf::cc::{CCSD, CCSDOptions};
use hf_rs::*;

#[test]
fn water_f12_orbital_basis_matches_pyscf_without_spin_orbital_workspace() {
    let input: serde_json::Value =
        serde_json::from_str(include_str!("data/psi4_mp2_f12_published_reference.json")).unwrap();
    let case = &input["cases"][0];
    let atoms: Vec<_> = case["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|atom| {
            Atom::new(
                atom[0].as_str().unwrap(),
                std::array::from_fn(|k| atom[1][k].as_f64().unwrap()),
            )
        })
        .collect();
    let basis = build_basis(&atoms, &parse_gaussian94(case["obs_g94"].as_str().unwrap()));
    let hf = scf::run_scf_in_space_with_convergence(
        &atoms,
        &basis,
        0,
        200,
        scf::RhfConvergence {
            energy: 1e-13,
            residual: 1e-11,
            density: 1e-11,
        },
        ConvAlgo::Cdiis,
        None,
        ao::BasisKind::Spherical,
    )
    .unwrap();
    let result = CCSD::new(&hf)
        .with_options(CCSDOptions {
            max_iterations: 200,
            energy_tol: 1e-13,
            residual_tol: 1e-11,
            ..Default::default()
        })
        .compute()
        .unwrap();
    assert!(result.converged);
    assert_eq!((result.n_occ, result.n_virt), (5, 43));
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("data/rhf_ccsd_water_reference.json")).unwrap();
    for (actual, key) in [
        (hf.total_energy, "e_scf"),
        (result.correlation_energy, "e_corr"),
        (result.total_energy, "e_total"),
    ] {
        let reference = expected[key].as_f64().unwrap();
        println!("water/cc-pVDZ-F12 {key}: error {:e}", actual - reference);
        assert!(
            (actual - reference).abs() < 1e-10,
            "{key}: {actual} vs {reference}"
        );
    }
}
