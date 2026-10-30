// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    ao::BasisKind,
    df::DfTensor,
    hf_variants::ri_df_hf::{DFHFOptions, run_df_hf_with_aux},
    *,
};
fn close(x: f64, y: f64) {
    assert!((x - y).abs() < 1e-9, "{x:0.15} vs {y:0.15}: {:e}", x - y);
}
fn atoms(case: &serde_json::Value) -> Vec<Atom> {
    case["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            Atom::new(
                a[0].as_str().unwrap(),
                std::array::from_fn(|k| a[1][k].as_f64().unwrap()),
            )
        })
        .collect()
}
#[test]
fn ri_hf_every_density_fock_j_k_component_matches_pyscf_with_ecp_spherical_and_rank_drop() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/df_hf_reference.json")).unwrap();
    let mut components = 0;
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut atoms = atoms(case);
        if name == "he_ecp" {
            atoms[0].ecp = Some(std::sync::Arc::new(ecp::EcpDef {
                symbol: "He".into(),
                n_core: 0,
                l_max: 2,
                local: vec![ecp::EcpTerm {
                    power: 2,
                    exponent: 0.7,
                    coefficient: 0.17,
                }],
                semilocal: vec![
                    vec![ecp::EcpTerm {
                        power: 2,
                        exponent: 0.53,
                        coefficient: 0.11,
                    }],
                    vec![ecp::EcpTerm {
                        power: 2,
                        exponent: 0.61,
                        coefficient: -0.07,
                    }],
                ],
            }));
        }
        let (basis, aux) = if name == "water" {
            (
                build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap()),
                build_basis(&atoms, &get_builtin_aux_basis("def2-svp-ri").unwrap()),
            )
        } else {
            let extra = if name == "he_rank_drop" {
                "S 1 1\n1.4 1\n"
            } else if name == "he_spherical_d" {
                "D 1 1\n0.65 1\n"
            } else {
                ""
            };
            let obs = format!("He 0\nS 1 1\n1.4 1\nS 1 1\n0.45 1\n{extra}****");
            let duplicate = if name == "he_rank_drop" {
                "S 1 1\n0.2 1\n"
            } else {
                ""
            };
            let aux = format!(
                "He 0\nS 1 1\n0.2 1\nS 1 1\n0.9 1\nS 1 1\n3.2 1\nP 1 1\n0.8 1\nD 1 1\n0.8 1\n{duplicate}****"
            );
            (
                build_basis(&atoms, &parse_gaussian94(&obs)),
                build_basis(&atoms, &parse_gaussian94(&aux)),
            )
        };
        let options = DFHFOptions {
            basis_kind: if case["spherical"].as_bool().unwrap() {
                BasisKind::Spherical
            } else {
                BasisKind::Cartesian
            },
            residual_tolerance: 1e-11,
            density_tolerance: 1e-11,
            energy_tolerance: 1e-13,
            estimate_fitting_error: true,
            max_iterations: 300,
            ..Default::default()
        };
        let result = run_df_hf_with_aux(&atoms, &basis, &aux, 0, &options).unwrap();
        close(result.total_energy, case["e_scf"].as_f64().unwrap());
        close(
            result.fitting_error.unwrap(),
            case["maximum_fitting_error"].as_f64().unwrap(),
        );
        assert_eq!(result.n_aux, case["n_aux"].as_u64().unwrap() as usize);
        assert_eq!(
            result.mo_coefficients.cols,
            case["n_mo"].as_u64().unwrap() as usize
        );
        assert!(result.residual_norm < 1e-10);
        for (a, b) in result
            .orbital_energies
            .iter()
            .zip(case["orbital_energies"].as_array().unwrap())
        {
            close(*a, b.as_f64().unwrap());
        }
        let mut occupied = Matrix::zeros(
            basis.len(),
            atoms.iter().map(|a| a.charge).sum::<f64>() as usize / 2,
        );
        for p in 0..basis.len() {
            for i in 0..occupied.cols {
                occupied[(p, i)] = result.mo_coefficients[(p, i)];
            }
        }
        let tensor = DfTensor::build(&basis, &aux, options.metric_cutoff).unwrap();
        let (j, k) = tensor.coulomb_exchange_rhf(&occupied).unwrap();
        for (key, actual) in [
            ("density", &result.density_matrix),
            ("fock", &result.fock_matrix),
            ("j", &j),
            ("k", &k),
        ] {
            for p in 0..basis.len() {
                for q in 0..basis.len() {
                    close(actual[(p, q)], case[key][p][q].as_f64().unwrap());
                    components += 1;
                }
            }
        }
        assert!(
            run_df_hf_with_aux(
                &atoms,
                &basis,
                &aux,
                0,
                &DFHFOptions {
                    max_iterations: 1,
                    ..options.clone()
                }
            )
            .is_err()
        );
        assert!(run_df_hf_with_aux(&atoms, &basis, &aux, 1, &options).is_err());
        println!(
            "RI-HF {name}: E={:0.15}, rank={}, max_residual={:e}",
            result.total_energy, result.mo_coefficients.cols, result.residual_norm
        );
    }
    println!("RI-HF all {components} matrix components verified");
}
