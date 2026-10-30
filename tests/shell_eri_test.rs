// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{Atom, build_basis, parse_gaussian94};

fn spdf_problem() -> (serde_json::Value, Vec<Atom>, Vec<hf_rs::ContractedGaussian>) {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/spherical_reference.json")).unwrap();
    let data = data["spdf_integrals"].clone();
    let mut atoms: Vec<_> = data["symbols"]
        .as_array()
        .unwrap()
        .iter()
        .zip(data["geometry_bohr"].as_array().unwrap())
        .map(|(s, r)| {
            Atom::new(
                s.as_str().unwrap(),
                std::array::from_fn(|k| r[k].as_f64().unwrap()),
            )
        })
        .collect();
    let term = |name: &str| {
        let a = &data["ecp_terms"][name];
        hf_rs::ecp::EcpTerm {
            power: a[0].as_i64().unwrap() as i32,
            exponent: a[1].as_f64().unwrap(),
            coefficient: a[2].as_f64().unwrap(),
        }
    };
    atoms[2].ecp = Some(std::sync::Arc::new(hf_rs::ecp::EcpDef {
        symbol: "Be".into(),
        n_core: 0,
        l_max: 4,
        local: vec![term("local")],
        semilocal: ["s", "p", "d", "f"].iter().map(|x| vec![term(x)]).collect(),
    }));
    let map = parse_gaussian94(data["basis_gaussian94"].as_str().unwrap());
    let basis = build_basis(&atoms, &map);
    assert_eq!(basis.len(), 80);
    (data, atoms, basis)
}

struct SpdfInputs {
    basis: std::path::PathBuf,
    ecp: std::path::PathBuf,
    positions: Vec<[f64; 3]>,
}
impl SpdfInputs {
    fn new() -> Self {
        use serde_json::json;
        let (data, atoms, _) = spdf_problem();
        let stem = format!("hf-rs-spdf-{}", std::process::id());
        let basis = std::env::temp_dir().join(format!("{stem}.gbs"));
        let ecp = std::env::temp_dir().join(format!("{stem}.json"));
        std::fs::write(&basis, data["basis_gaussian94"].as_str().unwrap()).unwrap();
        let potentials: Vec<_> = ["s", "p", "d", "f", "local"].iter().enumerate().map(|(l, key)| {
            let v = &data["ecp_terms"][key];
            json!({"angular_momentum":[l],"r_exponents":[v[0]],"gaussian_exponents":[v[1]],"coefficients":[[v[2]]]})
        }).collect();
        std::fs::write(
            &ecp,
            json!({"elements":{"4":{"ecp_electrons":0,"ecp_potentials":potentials}}}).to_string(),
        )
        .unwrap();
        Self {
            basis,
            ecp,
            positions: atoms.iter().map(|a| a.position).collect(),
        }
    }
    fn molecule(&self, moved: Option<(usize, f64)>) -> hf_rs::api::Molecule {
        let mut positions = self.positions.clone();
        if let Some((coordinate, delta)) = moved {
            positions[coordinate / 3][coordinate % 3] += delta;
        }
        hf_rs::api::Molecule::new(
            &[2, 3, 4, 5],
            &positions,
            hf_rs::api::MoleculeOptions {
                basis: self.basis.to_string_lossy().into_owned(),
                ecp: Some(self.ecp.to_string_lossy().into_owned()),
                charge: 10,
                ..Default::default()
            },
        )
        .unwrap()
    }
}
impl Drop for SpdfInputs {
    fn drop(&mut self) {
        std::fs::remove_file(&self.basis).unwrap();
        std::fs::remove_file(&self.ecp).unwrap();
    }
}

#[test]
fn mixed_spdf_bare_integral_derivatives_through_fourth() {
    use hf_rs::api::{IntegralRequest, ScfOptions};
    let inputs = SpdfInputs::new();
    let molecule = inputs.molecule(None);
    let pair_indices = vec![(0..20).collect(), (20..40).collect()];
    // Every pair component, and one nontrivial Cartesian member of every one of
    // the 256 angular quartets. Integral values separately cover every member.
    let quartet_indices: Vec<Vec<_>> = (0..4)
        .map(|center| {
            [
                0,
                1 + center % 3,
                4 + (2 * center + 1) % 6,
                10 + (3 * center + 2) % 10,
            ]
            .map(|i| 20 * center + i)
            .to_vec()
        })
        .collect();
    for operator in ["overlap", "kinetic", "nuclear_attraction", "ecp", "eri"] {
        let indices = if operator == "eri" {
            quartet_indices.clone()
        } else {
            pair_indices.clone()
        };
        let patterns = if operator == "ecp" {
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [6, 7, 8, 6]]
        } else {
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [0, 1, 2, 0]]
        };
        for pattern in patterns {
            let mut prefix = Vec::new();
            for coordinate in pattern {
                let mut derivative = prefix.clone();
                derivative.push(coordinate);
                let request = IntegralRequest {
                    operator: operator.into(),
                    indices: indices.clone(),
                    derivative_coordinates: derivative.clone(),
                    ..Default::default()
                };
                let actual = molecule
                    .integrals(&request, ScfOptions::default())
                    .unwrap()
                    .tensor
                    .values;
                let mut expected = vec![0.; actual.len()];
                let step = 1e-3;
                for (shift, weight) in [(-2., 1.), (-1., -8.), (1., 8.), (2., -1.)] {
                    let displaced = inputs.molecule(Some((coordinate, shift * step)));
                    let values = displaced
                        .integrals(
                            &IntegralRequest {
                                derivative_coordinates: prefix.clone(),
                                ..request.clone()
                            },
                            ScfOptions::default(),
                        )
                        .unwrap()
                        .tensor
                        .values;
                    for (sum, v) in expected.iter_mut().zip(values) {
                        *sum += weight * v / (12. * step);
                    }
                }
                let worst = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f64::max);
                eprintln!(
                    "SPDF {operator} derivative {derivative:?}: {} components, max error {worst:.8e}",
                    actual.len()
                );
                assert!(actual.iter().chain(&expected).all(|v| v.is_finite()));
                assert!(worst < 2e-7, "{operator} {derivative:?}: {worst}");
                prefix = derivative;
            }
        }
    }
}

#[test]
fn all_spdf_pair_blocks_match_pyscf_every_cartesian_component() {
    let (data, atoms, basis) = spdf_problem();
    let ecp = hf_rs::ecp::ecp_one_electron_matrix(&atoms, &basis[..40]);
    for operator in ["overlap", "kinetic", "nuclear_attraction", "ecp"] {
        let mut worst = 0f64;
        let mut counts = [0usize; 16];
        for i in 0..20 {
            for j in 0..20 {
                let a = &basis[i];
                let b = &basis[20 + j];
                let actual = match operator {
                    "overlap" => hf_rs::integrals::contracted_overlap(a, b),
                    "kinetic" => hf_rs::integrals::contracted_kinetic(a, b),
                    "nuclear_attraction" => {
                        hf_rs::integrals::contracted_nuclear_attraction(a, b, &atoms)
                    }
                    _ => ecp[(i, 20 + j)],
                };
                let error = (actual - data["pairs"][operator][i * 20 + j].as_f64().unwrap()).abs();
                assert!(error.is_finite());
                worst = worst.max(error);
                counts[a.l.iter().sum::<usize>() * 4 + b.l.iter().sum::<usize>()] += 1;
            }
        }
        assert!(counts.iter().all(|&n| n > 0));
        let limit = if operator == "ecp" { 1e-8 } else { 2e-10 };
        eprintln!(
            "SPDF {operator}: 16 angular pairs, 400 Cartesian components, max error {worst:.8e}"
        );
        assert!(worst < limit, "{operator}: {worst}");
    }
}

#[test]
fn all_spdf_quartets_match_pyscf_every_cartesian_component_both_engines() {
    use rayon::prelude::*;
    let (data, _atoms, basis) = spdf_problem();
    let reference: Vec<f64> = data["eri"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    assert_eq!(reference.len(), 160000);
    let ranges = [0..1, 1..4, 4..10, 10..20];
    let results: Vec<_> = (0..256usize)
        .into_par_iter()
        .map(|block| {
            let angular = [block / 64, block / 16 % 4, block / 4 % 4, block % 4];
            let shells: Vec<_> = (0..4)
                .map(|a| {
                    &basis[(20 * a + ranges[angular[a]].start)..(20 * a + ranges[angular[a]].end)]
                })
                .collect();
            let shell = hf_rs::shell_integrals::compute_shell_quartet([
                shells[0], shells[1], shells[2], shells[3],
            ])
            .unwrap();
            let mut count = 0;
            let mut scalar_worst = 0f64;
            let mut shell_worst = 0f64;
            for i in ranges[angular[0]].clone() {
                for j in ranges[angular[1]].clone() {
                    for k in ranges[angular[2]].clone() {
                        for l in ranges[angular[3]].clone() {
                            let expected = reference[((i * 20 + j) * 20 + k) * 20 + l];
                            let scalar = hf_rs::integrals::contracted_eri(
                                &basis[i],
                                &basis[20 + j],
                                &basis[40 + k],
                                &basis[60 + l],
                            );
                            scalar_worst = scalar_worst.max((scalar - expected).abs());
                            shell_worst = shell_worst.max((shell[count] - expected).abs());
                            count += 1;
                        }
                    }
                }
            }
            assert_eq!(count, shell.len());
            assert!(scalar_worst.is_finite() && shell_worst.is_finite());
            assert!(
                scalar_worst < 2e-10 && shell_worst < 2e-10,
                "L={angular:?}: scalar {scalar_worst:e}, shell {shell_worst:e}"
            );
            (count, scalar_worst, shell_worst)
        })
        .collect();
    assert_eq!(results.len(), 256);
    assert_eq!(results.iter().map(|x| x.0).sum::<usize>(), 160000);
    eprintln!(
        "SPDF ERIs: 256 angular quartets, 160000 components in each engine; scalar max {:.8e}, shell max {:.8e}",
        results.iter().map(|x| x.1).fold(0f64, f64::max),
        results.iter().map(|x| x.2).fold(0f64, f64::max)
    );
}

#[test]
fn shell_recurrences_match_independent_primitive_integrals_all_block_components() {
    let centers = [
        [0.1, -0.2, 0.3],
        [0.4, 0.6, -0.5],
        [-0.2, 0.7, 0.8],
        [0.9, -0.3, 0.2],
    ];
    for angular in [
        [0, 0, 0, 0],
        [2, 1, 2, 0],
        [3, 0, 1, 0],
        [4, 0, 1, 0],
        [7, 0, 1, 0],
    ] {
        let shells = (0..4)
            .map(|i| {
                let map = parse_gaussian94(&format!(
                    "H 0\n{} 2 1\n{} 0.7\n{} -0.2\n****",
                    hf_rs::basis::l_to_shell_type(angular[i]),
                    0.8 + 0.2 * i as f64,
                    0.3 + 0.07 * i as f64
                ));
                build_basis(&[Atom::new("H", centers[i])], &map)
            })
            .collect::<Vec<_>>();
        let actual = hf_rs::shell_integrals::compute_shell_quartet([
            &shells[0], &shells[1], &shells[2], &shells[3],
        ])
        .unwrap();
        let mut index = 0;
        for a in &shells[0] {
            for b in &shells[1] {
                for c in &shells[2] {
                    for d in &shells[3] {
                        let expected = hf_rs::integrals::contracted_eri(a, b, c, d);
                        assert!(
                            (actual[index] - expected).abs() < 2e-11,
                            "L={angular:?}, powers {:?} {:?} {:?} {:?}: {} vs {expected}",
                            a.l,
                            b.l,
                            c.l,
                            d.l,
                            actual[index]
                        );
                        index += 1;
                    }
                }
            }
        }
    }
}

#[test]
fn shell_packing_and_screening_match_all_unique_ao_quartets() {
    let atoms = [
        Atom::new("H", [0.2, 0.1, -0.5]),
        Atom::new("H", [-0.1, 0.3, 0.8]),
    ];
    let map = parse_gaussian94("H 0\nS 1 1\n0.8 1\nP 1 1\n0.7 1\nD 1 1\n0.6 1\n****");
    let basis = build_basis(&atoms, &map);
    let actual = hf_rs::shell_integrals::compute_shell_eri_tensor(&basis).unwrap();
    let expected = hf_rs::scf::compute_eri_tensor_cartesian_pairs(&basis);
    assert_eq!(actual.len(), expected.len());
    let worst = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(worst < 2e-11, "max shell ERI error {worst}");
}
