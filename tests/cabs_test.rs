// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::post_hf::f12::cabs::{
    CABSBasis, compute_cabs_singles_full, compute_cabs_singles_in_space, extended_fock,
};
use hf_rs::*;
fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect(),
    )
}
fn close(a: &Matrix, b: &Matrix, tolerance: f64) {
    assert_eq!((a.rows, a.cols), (b.rows, b.cols));
    for (index, (&a, &b)) in a.data.iter().zip(&b.data).enumerate() {
        assert!((a - b).abs() < tolerance, "component {index}: {a} vs {b}");
    }
}
#[test]
fn every_cabs_projector_full_fock_and_singles_component_matches_pyscf_with_ecp() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/cabs_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        println!("CABS case {}", case["name"]);
        let mut atoms = case["symbols"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, s)| {
                Atom::new(
                    s.as_str().unwrap(),
                    std::array::from_fn(|a| case["positions"][i][a].as_f64().unwrap()),
                )
            })
            .collect::<Vec<_>>();
        if case["name"] == "he_local_ecp" {
            atoms[0].ecp = Some(std::sync::Arc::new(ecp::EcpDef {
                symbol: "He".into(),
                n_core: 0,
                l_max: 0,
                local: vec![ecp::EcpTerm {
                    power: 2,
                    exponent: 0.7,
                    coefficient: 0.17,
                }],
                semilocal: vec![],
            }));
        }
        let spherical = case["name"] == "heh_spherical";
        let definitions = if spherical {
            parse_gaussian94("He 0\nS 1 1\n1.4 1\nD 1 1\n0.6 1\n****\nH 0\nS 1 1\n1.1 1\n****")
        } else if case["name"] == "he_overlap_rank_drop" {
            parse_gaussian94("He 0\nS 1 1\n1.4 1\nS 1 1\n1.4 1\nS 1 1\n0.4 1\n****")
        } else {
            get_builtin_basis("sto-3g").unwrap()
        };
        let obs = build_basis(&atoms, &definitions);
        let hf = scf::run_scf_in_space(
            &atoms,
            &obs,
            i32::from(spherical),
            150,
            1e-14,
            ConvAlgo::Cdiis,
            None,
            if spherical {
                ao::BasisKind::Spherical
            } else {
                ao::BasisKind::Cartesian
            },
        )
        .unwrap();
        assert_eq!(obs.len(), case["n_cart"].as_u64().unwrap() as usize);
        assert_eq!(
            hf.mo_coefficients.cols,
            case["n_mo"].as_u64().unwrap() as usize
        );
        assert!((hf.total_energy - case["e_scf"].as_f64().unwrap()).abs() < 1e-10);
        let definitions = if spherical {
            parse_gaussian94(
                "H 0\nS 1 1\n0.41 1\nP 1 1\n0.65 1\nD 1 1\n0.52 1\n****\nHe 0\nS 1 1\n0.41 1\nP 1 1\n0.65 1\nD 1 1\n0.52 1\n****",
            )
        } else {
            parse_gaussian94(
                "H 0\nS 1 1\n0.41 1\nP 1 1\n0.65 1\n****\nHe 0\nS 1 1\n0.41 1\nP 1 1\n0.65 1\n****",
            )
        };
        let aux = build_basis(&atoms, &definitions);
        let cabs = CABSBasis::from_reference(&hf, &obs, &aux, 1e-8).unwrap();
        assert_eq!(cabs.n_cabs, case["cabs_rank"].as_u64().unwrap() as usize);
        let c = cabs.union_coefficients();
        let union = obs.iter().chain(aux.iter()).cloned().collect::<Vec<_>>();
        let mut overlap = Matrix::zeros(union.len(), union.len());
        for i in 0..union.len() {
            for j in 0..union.len() {
                overlap[(i, j)] = integrals::contracted_overlap(&union[i], &union[j]);
            }
        }
        let covectors = overlap.dot(&c);
        close(
            &covectors.dot(&covectors.transpose()),
            &matrix(&case["cabs_metric_projector"]),
            1e-9,
        );
        // With a redundant OBS, raw AO coefficients contain an arbitrary null
        // component. Compare all physical projector elements in the AO metric.
        if case["name"] != "he_overlap_rank_drop" {
            close(
                &c.dot(&c.transpose()),
                &matrix(&case["cabs_projector"]),
                1e-9,
            );
        }
        let f = extended_fock(&hf, &atoms, &obs, &aux).unwrap();
        close(&f.fock, &matrix(&case["full_fock"]), 1e-9);
        close(&f.exchange, &matrix(&case["exchange"]), 1e-10);
        let result = compute_cabs_singles_full(&hf, &atoms, &obs, &cabs).unwrap();
        assert!((result.energy - case["singles_energy"].as_f64().unwrap()).abs() < 1e-10);
        assert_eq!(
            result.virtual_energies.len(),
            case["virtual_energies"].as_array().unwrap().len()
        );
        for (a, b) in result
            .virtual_energies
            .iter()
            .zip(case["virtual_energies"].as_array().unwrap())
        {
            assert!((a - b.as_f64().unwrap()).abs() < 1e-9);
        }
        let mut occupied = Matrix::zeros(c.rows, hf.n_occ);
        for mu in 0..obs.len() {
            for i in 0..hf.n_occ {
                occupied[(mu, i)] = hf.mo_coefficients[(mu, i)];
            }
        }
        let singles = occupied
            .dot(&result.amplitudes)
            .dot(&result.virtual_coefficients.transpose());
        let singles_metric = overlap
            .dot(&occupied)
            .dot(&result.amplitudes)
            .dot(&overlap.dot(&result.virtual_coefficients).transpose());
        close(&singles_metric, &matrix(&case["singles_metric"]), 1e-9);
        for fc in 0..=hf.n_occ {
            let active = compute_cabs_singles_in_space(&hf, &atoms, &obs, &cabs, fc).unwrap();
            let expected = case["singles_energy_by_frozen_core"][fc].as_f64().unwrap();
            assert!((active.energy - expected).abs() < 1e-10);
            let active_metric = overlap
                .dot(&occupied)
                .dot(&active.amplitudes)
                .dot(&overlap.dot(&active.virtual_coefficients).transpose());
            close(
                &active_metric,
                &matrix(&case["singles_metric_by_frozen_core"][fc]),
                1e-9,
            );
            for i in 0..fc {
                for a in 0..active.amplitudes.cols {
                    assert_eq!(active.amplitudes[(i, a)], 0.0);
                }
            }
        }
        assert!(compute_cabs_singles_in_space(&hf, &atoms, &obs, &cabs, hf.n_occ + 1).is_err());
        if case["name"] != "he_overlap_rank_drop" {
            close(&singles, &matrix(&case["singles_ao"]), 1e-9);
        }
        let empty = CABSBasis::from_reference(&hf, &obs, &obs, 1e-8).unwrap();
        assert_eq!(empty.n_cabs, 0);
        assert!(
            compute_cabs_singles_full(&hf, &atoms, &obs, &empty)
                .unwrap()
                .energy
                .abs()
                < 1e-20
        );
    }
}
