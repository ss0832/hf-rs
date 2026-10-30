// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::post_hf::PostHFMethod;
use hf_rs::post_hf::f12::operators::GaussianGeminal;
use hf_rs::post_hf::f12::{CABSBasis, MP2F12Options, RHFMP2F12};
use hf_rs::*;

fn verify(case: &serde_json::Value) {
    let name = case["name"].as_str().unwrap();
    let atoms = case["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|atom| {
            Atom::new(
                atom[0].as_str().unwrap(),
                std::array::from_fn(|a| atom[1][a].as_f64().unwrap()),
            )
        })
        .collect::<Vec<_>>();
    let obs = build_basis(&atoms, &parse_gaussian94(case["obs_g94"].as_str().unwrap()));
    let aux = build_basis(&atoms, &parse_gaussian94(case["aux_g94"].as_str().unwrap()));
    let hf = scf::run_scf_in_space(
        &atoms,
        &obs,
        0,
        150,
        1e-14,
        ConvAlgo::Cdiis,
        None,
        if case["spherical"].as_bool().unwrap() {
            ao::BasisKind::Spherical
        } else {
            ao::BasisKind::Cartesian
        },
    )
    .unwrap();
    assert_eq!(hf.n_occ, case["n_occ"].as_u64().unwrap() as usize);
    assert_eq!(
        hf.mo_coefficients.cols,
        case["n_mo"].as_u64().unwrap() as usize
    );
    let cabs = CABSBasis::from_reference(&hf, &obs, &aux, 1e-8).unwrap();
    println!(
        "{name}: OBS {} Cartesian / {} MO, CABS {}, raw aux {}",
        obs.len(),
        hf.mo_coefficients.cols,
        cabs.n_cabs,
        aux.len()
    );
    let beta = case["beta"].as_f64().unwrap();
    let factor = GaussianGeminal::new(
        case["terms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (t[0].as_f64().unwrap(), t[1].as_f64().unwrap()))
            .collect(),
    )
    .unwrap();
    let result = RHFMP2F12::new(&hf, &atoms, &obs)
        .with_options(MP2F12Options {
            gamma: beta,
            frozen_core: case["frozen_core"].as_u64().unwrap() as usize,
            custom_cabs: Some(cabs),
            custom_geminal: if beta == 1.0 { None } else { Some(factor) },
            ..Default::default()
        })
        .compute()
        .unwrap();
    for (a, key) in [
        (result.hf_energy, "e_scf"),
        (result.mp2_correlation_energy, "e_mp2"),
        (result.f12_correlation_energy, "e_f12"),
        (result.cabs_singles_energy, "e_cabs"),
        (result.f12_singlet_energy, "e_f12_singlet"),
        (result.f12_triplet_energy, "e_f12_triplet"),
        (result.total_energy, "e_total"),
    ] {
        let b = case[key].as_f64().unwrap();
        println!(
            "{name} {key}: Rust {a:.15}, Psi4 {b:.15}, error {:.3e}",
            (a - b).abs()
        );
        assert!((a - b).abs() < 1e-10, "{name} {key}: {a} vs {b}");
    }
}

#[test]
fn complete_3c_energies_match_psi4_with_triplets_frozen_core_and_different_factors() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/psi4_mp2_f12_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        verify(case);
    }
}

#[test]
#[ignore = "full cc-pVDZ-F12/OptRI benchmark; run explicitly with optimized integrals"]
fn published_water_cc_pvdz_f12_benchmark() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/psi4_mp2_f12_published_reference.json")).unwrap();
    verify(&data["cases"][0]);
}
