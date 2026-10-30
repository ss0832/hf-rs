// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{Matrix, post_hf::f12::ccsd_intermediates::CcsdF12Coupling};

fn vector(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn matrices(v: &serde_json::Value) -> Vec<Matrix> {
    v.as_array().unwrap().iter().map(matrix).collect()
}
fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter().flat_map(vector).collect(),
    )
}

#[test]
fn every_full_obs_projected_coulomb_component_matches_psi4_pyscf_with_nonlocal_ecp() {
    use hf_rs::{
        ao::BasisKind,
        basis::{Atom, build_basis, parse_gaussian94},
        ecp::{EcpDef, EcpTerm},
        post_hf::f12::{
            CABSBasis, molecular::molecular_ccsd_projected_coulomb, operators::GaussianGeminal,
        },
        scf::{RhfConvergence, run_scf_in_space_with_convergence},
    };
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/ccsd_f12_projected_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let mut atoms: Vec<_> = data["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| Atom::new("He", std::array::from_fn(|a| p[a].as_f64().unwrap())))
            .collect();
        if case["nonlocal_ecp"].as_bool().unwrap() {
            for atom in &mut atoms {
                atom.ecp = Some(std::sync::Arc::new(EcpDef {
                    symbol: "He".into(),
                    n_core: 0,
                    l_max: 2,
                    local: vec![EcpTerm {
                        power: 2,
                        exponent: 0.7,
                        coefficient: 0.17,
                    }],
                    semilocal: vec![
                        vec![EcpTerm {
                            power: 2,
                            exponent: 0.53,
                            coefficient: 0.11,
                        }],
                        vec![EcpTerm {
                            power: 2,
                            exponent: 0.61,
                            coefficient: -0.07,
                        }],
                    ],
                }));
            }
        }
        let obs = build_basis(&atoms, &parse_gaussian94(data["obs_g94"].as_str().unwrap()));
        let aux = build_basis(&atoms, &parse_gaussian94(data["aux_g94"].as_str().unwrap()));
        let hf = run_scf_in_space_with_convergence(
            &atoms,
            &obs,
            0,
            200,
            RhfConvergence {
                energy: 1e-13,
                residual: 1e-12,
                density: 1e-12,
            },
            hf_rs::diis::ConvAlgo::Adiissoscf,
            None,
            BasisKind::Cartesian,
        )
        .unwrap();
        let cabs = CABSBasis::from_reference(&hf, &obs, &aux, 1e-8).unwrap();
        let factor = GaussianGeminal::new(
            data["terms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| (t[0].as_f64().unwrap(), t[1].as_f64().unwrap()))
                .collect(),
        )
        .unwrap();
        let actual = molecular_ccsd_projected_coulomb(&hf, &obs, &cabs, &factor).unwrap();
        let expected = matrices(&case["v"]);
        let alignment = hf
            .mo_coefficients
            .transpose()
            .dot(&hf.overlap_matrix)
            .dot(&matrix(&case["mo_coefficients"]));
        let phase: Vec<_> = (0..alignment.rows)
            .map(|i| {
                assert!(alignment[(i, i)].abs() > 1. - 1e-9);
                alignment[(i, i)].signum()
            })
            .collect();
        let mut maximum = 0f64;
        assert_eq!(actual.len(), expected.len());
        for (ij, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            let i = ij / hf.n_occ;
            let j = ij % hf.n_occ;
            for p in 0..actual.rows {
                for q in 0..actual.cols {
                    let value = actual[(p, q)] * phase[p] * phase[q] * phase[i] * phase[j];
                    maximum = maximum.max((value - expected[(p, q)]).abs());
                    assert!(
                        (value - expected[(p, q)]).abs() < 1e-10,
                        "pair {ij} bra {p},{q}: {value} vs {}",
                        expected[(p, q)]
                    );
                }
            }
        }
        println!(
            "Projected Coulomb ECP={}: 64 components, max_error={maximum:e}",
            case["nonlocal_ecp"]
        );
        let constant = molecular_ccsd_projected_coulomb(
            &hf,
            &obs,
            &cabs,
            &GaussianGeminal::new(vec![(0., 0.8)]).unwrap(),
        )
        .unwrap();
        assert!(
            constant
                .iter()
                .flat_map(|m| &m.data)
                .all(|x| x.abs() < 1e-10)
        );
    }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-13, "{a} vs {b}");
}

#[test]
fn every_ccsd_f12_source_and_energy_term_matches_full_pair_tensor_projection() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/ccsd_f12_coupling_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let no = case["n_occ"].as_u64().unwrap() as usize;
        let nv = case["n_virt"].as_u64().unwrap() as usize;
        let v = matrices(&case["v"]);
        let c = matrices(&case["c"]);
        let t1 = vector(&case["t1"]);
        let t2 = vector(&case["t2"]);
        let coupling = CcsdF12Coupling::from_pair_integrals(no, nv, &v, &c).unwrap();
        let source = coupling.residual_corrections(&t1).unwrap();
        for (actual, key) in [(&source.singles, "singles"), (&source.doubles, "doubles")] {
            let expected = vector(&case[key]);
            assert_eq!(actual.len(), expected.len());
            for (&a, b) in actual.iter().zip(expected) {
                close(a, b);
            }
        }
        let energy = coupling.energy_corrections(&t1, &t2).unwrap();
        close(energy.fock_amplitude, case["fock_energy"].as_f64().unwrap());
        close(
            energy.additional_f12b,
            case["additional_f12b"].as_f64().unwrap(),
        );
        assert!(coupling.residual_corrections(&[]).is_err());
        assert!(coupling.energy_corrections(&t1, &[]).is_err());
        let zero_v = vec![Matrix::zeros(no + nv, no + nv); no * no];
        let zero_c = vec![Matrix::zeros(nv, nv); no * no];
        let zero = CcsdF12Coupling::from_pair_integrals(no, nv, &zero_v, &zero_c).unwrap();
        let source = zero.residual_corrections(&t1).unwrap();
        assert!(
            source
                .singles
                .iter()
                .chain(&source.doubles)
                .all(|&v| v == 0.)
        );
        let e = zero.energy_corrections(&t1, &t2).unwrap();
        close(e.fock_amplitude, 0.);
        close(e.additional_f12b, 0.);
        let mut bad = v;
        bad[0][(0, 1)] += 0.1;
        assert!(CcsdF12Coupling::from_pair_integrals(no, nv, &bad, &c).is_err());
    }
}
