// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    api::{Molecule, MoleculeOptions, OptionMap, ScfOptions},
    ecp,
};
use serde_json::json;
use std::path::PathBuf;

fn hi(basis: serde_json::Value, potential: serde_json::Value) -> Molecule {
    let mut options = MoleculeOptions::default();
    options.set_basis(&basis).unwrap();
    options.set_ecp(&potential).unwrap();
    Molecule::new(&[1, 53], &[[0.0; 3], [0.2, -0.1, 3.0]], options).unwrap()
}

#[test]
fn every_energy_and_nuclear_gradient_component_matches_independent_mixed_basis_oracle() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/basis_assignment_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let mut options = MoleculeOptions::default();
        options.set_basis(&case["basis"]).unwrap();
        options.set_ecp(&case["ecp"]).unwrap();
        let numbers: Vec<usize> = serde_json::from_value(case["atomic_numbers"].clone()).unwrap();
        let positions: Vec<[f64; 3]> = serde_json::from_value(case["positions"].clone()).unwrap();
        let molecule = Molecule::new(&numbers, &positions, options).unwrap();
        assert_eq!(
            molecule.basis().len(),
            case["n_ao"].as_u64().unwrap() as usize
        );
        assert_eq!(
            molecule
                .atoms()
                .iter()
                .map(|atom| atom.charge as usize)
                .sum::<usize>(),
            case["electrons"].as_u64().unwrap() as usize
        );
        let settings = [
            ("scf_max_iter".into(), json!(300)),
            ("scf_energy_tol".into(), json!(1e-13)),
            ("scf_residual_tol".into(), json!(1e-11)),
        ]
        .into();
        let result = molecule.run("hf", &settings, &["gradient".into()]).unwrap();
        assert!(
            (result.output.e_total - case["energy"].as_f64().unwrap()).abs() < 1e-8,
            "{}",
            case["name"]
        );
        let gradient = result.gradient.unwrap();
        for (atom, row) in gradient.iter().enumerate() {
            for (axis, value) in row.iter().enumerate() {
                assert!(
                    (value - case["gradient"][atom][axis].as_f64().unwrap()).abs() < 1e-6,
                    "{} {atom} {axis}",
                    case["name"]
                );
            }
        }
        for axis in 0..3 {
            assert!(gradient.iter().map(|g| g[axis]).sum::<f64>().abs() < 1e-9);
        }
    }
}

#[test]
fn element_orbitals_and_independent_named_ecps_have_explicit_precedence() {
    let basis = json!({"H":"sto-3g", "I":"def2-svp"});
    let automatic = hi(basis.clone(), json!(null));
    assert_eq!(automatic.atoms()[1].ecp.as_ref().unwrap().n_core, 28);
    assert!(automatic.atoms()[0].ecp.is_none());
    let overridden = hi(basis.clone(), json!({"I":"lanl2dz"}));
    assert_eq!(overridden.atoms()[1].ecp.as_ref().unwrap().n_core, 46);
    assert_eq!(automatic.basis().len(), overridden.basis().len());
    let independent = hi(basis.clone(), json!("def2-ecp"));
    assert_eq!(independent.atoms()[1].ecp, automatic.atoms()[1].ecp);
    let disabled = hi(basis, json!({"I":null}));
    assert!(disabled.atoms()[1].ecp.is_none());
    assert_eq!(disabled.atoms()[1].charge, 53.0);
}

#[test]
fn bse_json_and_gaussian94_ecp_exports_match_every_channel_and_radial_term() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("third_party/basis_set_exchange");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).unwrap())
            .unwrap();
    let mut elements = 0;
    for entry in manifest["basis_sets"].as_array().unwrap() {
        let data =
            std::fs::read_to_string(root.join("data").join(entry["json_file"].as_str().unwrap()))
                .unwrap();
        let g94 = std::fs::read_to_string(
            root.join("data")
                .join(entry["gaussian94_file"].as_str().unwrap()),
        )
        .unwrap();
        let json_map = ecp::parse_ecp_bse_json(&data).unwrap();
        let gaussian_map = ecp::parse_ecp_gaussian94(&g94).unwrap();
        assert_eq!(json_map, gaussian_map, "{}", entry["name"]);
        elements += json_map.len();
    }
    assert!(
        elements > 300,
        "all exported scalar potentials must be covered"
    );
}

#[test]
fn all_added_orbital_exponents_and_coefficients_match_independent_export_formats() {
    type Rows = Vec<(f64, f64)>;
    type Shells = Vec<(usize, Rows)>;
    fn signature(map: &hf_rs::basis::BasisSetMap) -> std::collections::BTreeMap<String, Shells> {
        map.iter()
            .map(|(symbol, shells)| {
                let mut result = Vec::new();
                for shell in shells {
                    let columns = if shell.shell_type == "SP" {
                        vec![
                            (0, &shell.coefficients),
                            (1, shell.p_coefficients.as_ref().unwrap()),
                        ]
                    } else {
                        vec![(
                            hf_rs::basis::shell_type_to_l(&shell.shell_type).unwrap(),
                            &shell.coefficients,
                        )]
                    };
                    for (l, column) in columns {
                        let mut rows: Rows = shell
                            .exponents
                            .iter()
                            .copied()
                            .zip(column.iter().copied())
                            .filter(|(_, c)| *c != 0.0)
                            .collect();
                        rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        result.push((l, rows));
                    }
                }
                result.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (symbol.clone(), result)
            })
            .collect()
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("third_party/basis_set_exchange");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).unwrap())
            .unwrap();
    for entry in manifest["basis_sets"].as_array().unwrap() {
        let data =
            std::fs::read_to_string(root.join("data").join(entry["json_file"].as_str().unwrap()))
                .unwrap();
        let g94 = std::fs::read_to_string(
            root.join("data")
                .join(entry["gaussian94_file"].as_str().unwrap()),
        )
        .unwrap();
        let g94 = if entry["ecp_elements"].as_array().unwrap().is_empty() {
            g94
        } else {
            let prefix = g94
                .lines()
                .take_while(|line| {
                    !line
                        .split_whitespace()
                        .next()
                        .is_some_and(|token| token.ends_with("-ECP"))
                })
                .collect::<Vec<_>>()
                .join("\n");
            prefix[..prefix.rfind("****").unwrap() + 4].to_string()
        };
        let json_map = hf_rs::basis::parse_bse_json(&data).unwrap();
        let gaussian_map = hf_rs::basis::try_parse_gaussian94(&g94).unwrap();
        assert_eq!(
            signature(&json_map),
            signature(&gaussian_map),
            "{}",
            entry["name"]
        );
    }
}

#[test]
fn combined_basis_files_keep_associated_ecps_and_allow_independent_ecp_files() {
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("third_party/basis_set_exchange/data");
    for extension in ["gbs", "json"] {
        let source = root.join(format!("orbital_15.{extension}"));
        let mut options = MoleculeOptions::default();
        options
            .set_basis(&json!({"H":"sto-3g", "Na":source}))
            .unwrap();
        let molecule = Molecule::new(&[1, 11], &[[0.0; 3], [0.2, 0.1, 3.0]], options).unwrap();
        assert_eq!(molecule.atoms()[1].ecp.as_ref().unwrap().n_core, 10);
        let mut independent = MoleculeOptions::default();
        independent
            .set_basis(&json!({"*":"sto-3g", "Na":"6-31g"}))
            .unwrap();
        independent.set_ecp(&json!({"11":source})).unwrap();
        let other = Molecule::new(&[1, 11], &[[0.0; 3], [0.2, 0.1, 3.0]], independent).unwrap();
        assert_eq!(other.atoms()[1].ecp, molecule.atoms()[1].ecp);
    }
}

#[test]
fn incomplete_assignments_and_malformed_potentials_are_checked_errors() {
    for value in [
        json!({"H":"sto-3g", "1":"6-31g"}),
        json!({"NotAnElement":"sto-3g"}),
        json!({}),
    ] {
        assert!(MoleculeOptions::default().set_basis(&value).is_err());
    }
    let mut options = MoleculeOptions::default();
    options.set_basis(&json!({"H":"sto-3g"})).unwrap();
    assert!(Molecule::new(&[1, 11], &[[0.0; 3], [0.0, 0.0, 3.0]], options).is_err());
    let mut options = MoleculeOptions::default();
    options.set_ecp(&json!({"H":"def2-ecp"})).unwrap();
    assert!(Molecule::new(&[1, 1], &[[0.0; 3], [0.0, 0.0, 1.4]], options).is_err());
    for text in [
        "H-ECP 1 0\np potential",
        "H-ECP 1 0\np potential\n1000000000",
        "H-ECP 1000000 0",
        "H-ECP 0 1\ns potential\n0",
        "H-ECP 0 0\ns potential\n1\n2 NaN 1",
    ] {
        assert!(ecp::parse_ecp_gaussian94(text).is_err(), "{text}");
    }
}

#[test]
fn independent_local_nonlocal_ecp_files_preserve_analytic_fourth_response() {
    let file =
        std::env::temp_dir().join(format!("hf-rs-independent-ecp-{}.gbs", std::process::id()));
    std::fs::write(
        &file,
        "H-ECP 1 0\np potential\n1\n2 0.7 -0.25\ns-p potential\n1\n2 0.4 0.15\n",
    )
    .unwrap();
    let mut options = MoleculeOptions::default();
    options.set_basis(&json!({"H":"sto-3g"})).unwrap();
    options.set_ecp(&json!({"H":file})).unwrap();
    let molecule = Molecule::new(&[1, 1], &[[0.1, 0.2, -0.7], [-0.1, 0.0, 0.7]], options).unwrap();
    std::fs::remove_file(&file).unwrap();
    let settings = ScfOptions {
        energy_tol: 1e-14,
        residual_tol: 1e-12,
        density_tol: 1e-12,
        ..Default::default()
    };
    let fourth = molecule
        .nuclear_derivative(&[0, 2, 2, 5], settings)
        .unwrap();
    let directions = [0, 2, 2, 5].map(|i| {
        let mut direction = vec![0.0; 6];
        direction[i] = 1.0;
        direction
    });
    let directional = molecule
        .directional_derivative(&directions, settings)
        .unwrap();
    assert!((fourth - directional).abs() < 1e-10);
    assert!(fourth.abs() > 1e-8);
    assert!(
        molecule
            .run("hf", &OptionMap::new(), &["gradient".into()])
            .unwrap()
            .gradient
            .is_some()
    );
}
