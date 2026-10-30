// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::matrix::Matrix;
use hf_rs::mo_integrals::MoEriTensor;
use hf_rs::post_hf::dmrg_engine::{MpsSolverOptions, solve_mps};
use serde_json::Value;

fn vector(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn delta(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn all_density_components_and_energies_match_pyscf_and_same_m_block2_sz() {
    let source = std::env::var("HF_RS_DMRG_ORACLE_PATH")
        .ok()
        .map(|p| std::fs::read_to_string(p).unwrap())
        .unwrap_or_else(|| include_str!("data/dmrg_reference.json").into());
    let data: Value = serde_json::from_str(&source).unwrap();
    let mut energy_error = 0.0_f64;
    let mut density_error = 0.0_f64;
    let mut density_count = 0;
    for case in data["cases"].as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as usize;
        let h = Matrix::from_vec(
            n,
            n,
            case["h"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(vector)
                .collect(),
        );
        let g = &case["eri"];
        let pairs = n * (n + 1) / 2;
        let mut eri = MoEriTensor {
            n_basis: n,
            n_pairs: pairs,
            data: vec![0.0; pairs * pairs],
        };
        for p in 0..n {
            for q in 0..=p {
                for r in 0..n {
                    for s in 0..=r {
                        eri.data[(r * (r + 1) / 2 + s) * pairs + p * (p + 1) / 2 + q] =
                            g[p][q][r][s].as_f64().unwrap();
                    }
                }
            }
        }
        let na = case["n_alpha"].as_u64().unwrap() as usize;
        let nb = case["n_beta"].as_u64().unwrap() as usize;
        let sweeps = case["sweeps"].as_u64().unwrap() as usize;
        for expected in case["block2"].as_array().unwrap() {
            let m = expected["m"].as_u64().unwrap() as usize;
            let result = solve_mps(
                &h,
                &eri,
                na,
                nb,
                case["constant"].as_f64().unwrap(),
                &MpsSolverOptions {
                    bond_dimensions: vec![m; sweeps],
                    sweeps,
                    one_site_sweeps: case["one_site_sweeps"].as_u64().unwrap() as usize,
                    energy_tol: 1e-14,
                    residual_tol: 1e-12,
                    truncation_tol: 0.0,
                    ..Default::default()
                },
            )
            .unwrap();
            let (one, two) = result.state.rdms().unwrap();
            let electrons = (na + nb) as f64;
            assert!(((0..n).map(|p| one[(p, p)]).sum::<f64>() - electrons).abs() < 1e-10);
            let mut reconstructed = case["constant"].as_f64().unwrap();
            for p in 0..n {
                for q in 0..n {
                    assert!(
                        ((0..n)
                            .map(|r| two[((p * n + q) * n + r) * n + r])
                            .sum::<f64>()
                            - (electrons - 1.0) * one[(p, q)])
                            .abs()
                            < 1e-10
                    );
                    reconstructed += h[(p, q)] * one[(p, q)];
                    for r in 0..n {
                        for s in 0..n {
                            reconstructed +=
                                0.5 * eri.get(p, q, r, s) * two[((p * n + q) * n + r) * n + s];
                        }
                    }
                }
            }
            assert!((reconstructed - result.energy).abs() < 1e-10);
            let de = (result.energy - expected["energy"].as_f64().unwrap()).abs();
            let d1 = delta(&one.data, &vector(&expected["one_rdm"]));
            let d2 = delta(&two, &vector(&expected["two_rdm"]));
            println!(
                "{} M={m}: energy={:.15}, block2={:.15}, delta={de:e}, RDM1={d1:e}, RDM2={d2:e}",
                case["name"],
                result.energy,
                expected["energy"].as_f64().unwrap()
            );
            energy_error = energy_error.max(de);
            density_error = density_error.max(d1).max(d2);
            density_count += n * n + n.pow(4);
            if m == case["bond_dimensions"]
                .as_array()
                .unwrap()
                .last()
                .unwrap()
                .as_u64()
                .unwrap() as usize
            {
                assert!((result.energy - case["fci_energy"].as_f64().unwrap()).abs() < 1e-10);
                assert!(delta(&one.data, &vector(&case["fci_one_rdm"])) < 1e-8);
                assert!(delta(&two, &vector(&case["fci_two_rdm"])) < 1e-8);
            }
        }
    }
    println!(
        "DMRG oracle summary: energy_error={energy_error:e}, density_error={density_error:e}, density_count={density_count}"
    );
    assert!(energy_error <= 1e-6);
    // The plan fixes the energy criterion. For finite-M variational states,
    // separate convergence histories resolve densities less tightly than
    // energies; all elements are checked at 3e-5. The exact large-M densities
    // above retain their independent 1e-8 criterion.
    assert!(density_error <= 3e-5);
}
