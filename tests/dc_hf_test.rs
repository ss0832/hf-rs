// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Divide-and-conquer HF against an independent NumPy fixed-point oracle
//! (tests/reference/gen_dc_hf.py), the full-buffer RHF limit, buffer convergence,
//! partition invariances and input validation.

use hf_rs::{Atom, ContractedGaussian, DCOptions, build_basis, get_builtin_basis, run_dc_hf, run_scf};

fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/dc_hf_reference.json")).unwrap()
}

fn system(case: &serde_json::Value) -> (Vec<Atom>, Vec<ContractedGaussian>, Vec<Vec<usize>>) {
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
    let name = match case["constant"].as_str().unwrap() {
        "STO_3G_GBS" => "sto-3g",
        _ => "6-31g",
    };
    let basis = build_basis(&atoms, &get_builtin_basis(name).unwrap());
    let regions = case["central_regions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect())
        .collect();
    (atoms, basis, regions)
}

fn options(regions: Vec<Vec<usize>>, radius: f64, beta: f64) -> DCOptions {
    DCOptions {
        central_regions: Some(regions),
        buffer_radius_bohr: radius,
        inverse_temperature: beta,
        energy_threshold: 1e-12,
        density_threshold: 1e-11,
        max_iterations: 500,
        ..Default::default()
    }
}

#[test]
fn dc_fixed_points_match_independent_oracle() {
    let (mut worst_e, mut worst_p) = (0f64, 0f64);
    for case in data()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (atoms, basis, regions) = system(case);
        let r = run_dc_hf(
            &atoms,
            &basis,
            0,
            &options(regions, case["buffer_radius"].as_f64().unwrap(), case["inverse_temperature"].as_f64().unwrap()),
        )
        .unwrap();
        assert!(r.converged);
        let de = (r.total_energy - case["energy"].as_f64().unwrap()).abs();
        worst_e = worst_e.max(de);
        assert!(de < 1e-9, "{name}: {} vs {}", r.total_energy, case["energy"]);
        assert!((r.chemical_potential - case["chemical_potential"].as_f64().unwrap()).abs() < 1e-7, "{name}: mu");
        let reference = &case["density"];
        let mut dp = 0f64;
        for i in 0..basis.len() {
            for j in 0..basis.len() {
                dp = dp.max((r.density_matrix[(i, j)] - reference[i][j].as_f64().unwrap()).abs());
            }
        }
        worst_p = worst_p.max(dp);
        assert!(dp < 1e-8, "{name}: density {dp:e}");
        let n_electrons: f64 = atoms.iter().map(|a| a.charge).sum();
        assert!((r.electron_count - n_electrons).abs() < 1e-9);
        // The small buffers make DC-HF a genuine approximation here (not RHF).
        assert!((r.total_energy - case["rhf_energy"].as_f64().unwrap()).abs() > 1e-3);
    }
    println!("DC-HF: max |dE| {worst_e:e}, max |dP| {worst_p:e}");
}

#[test]
fn full_buffer_limit_is_rhf_and_buffers_converge() {
    let case = &data()["cases"][0];
    assert_eq!(case["name"], "H8_chain");
    let (atoms, basis, regions) = system(case);
    let rhf = run_scf(&atoms, &basis, 300, 1e-13).unwrap();
    // Every buffer contains the whole chain; with a gap of ~0.5 Eh and beta = 2000 the
    // occupations are 2 and 0 to machine precision.
    let full = run_dc_hf(&atoms, &basis, 0, &options(regions.clone(), 100.0, 2000.0)).unwrap();
    assert!((full.total_energy - rhf.total_energy).abs() < 1e-9, "{} vs RHF {}", full.total_energy, rhf.total_energy);
    let mut dp = 0f64;
    for (a, b) in full.density_matrix.data.iter().zip(&rhf.density_matrix.data) {
        dp = dp.max((a - b).abs());
    }
    assert!(dp < 1e-7, "density {dp:e}");
    // Larger buffers approach RHF.
    let errors: Vec<f64> = [3.0, 6.0, 12.0]
        .iter()
        .map(|&radius| {
            let r = run_dc_hf(&atoms, &basis, 0, &options(regions.clone(), radius, 200.0)).unwrap();
            (r.total_energy - rhf.total_energy).abs()
        })
        .collect();
    assert!(errors[1] < errors[0] && errors[2] < errors[1], "{errors:?}");
}

#[test]
fn subsystem_order_does_not_matter() {
    let case = &data()["cases"][2];
    let (atoms, basis, regions) = system(case);
    let radius = case["buffer_radius"].as_f64().unwrap();
    let a = run_dc_hf(&atoms, &basis, 0, &options(regions.clone(), radius, 200.0)).unwrap();
    let mut reversed = regions;
    reversed.reverse();
    for r in &mut reversed {
        r.reverse();
    }
    let b = run_dc_hf(&atoms, &basis, 0, &options(reversed, radius, 200.0)).unwrap();
    assert!((a.total_energy - b.total_energy).abs() < 1e-10);
}

#[test]
fn invalid_partitions_are_rejected() {
    let case = &data()["cases"][0];
    let (atoms, basis, _) = system(case);
    for regions in [vec![vec![0, 1], vec![1, 2, 3, 4, 5, 6, 7]], vec![vec![0, 1, 2, 3]], vec![vec![], (0..8).collect()]] {
        assert!(run_dc_hf(&atoms, &basis, 0, &options(regions, 3.0, 200.0)).is_err());
    }
    let all: Vec<Vec<usize>> = vec![(0..8).collect()];
    assert!(run_dc_hf(&atoms, &basis, 0, &options(all.clone(), -1.0, 200.0)).is_err());
    assert!(run_dc_hf(&atoms, &basis, 0, &options(all, 3.0, 0.0)).is_err());
}
