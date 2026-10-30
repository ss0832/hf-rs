// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    Matrix,
    ao::BasisKind,
    basis::{Atom, ContractedGaussian, build_basis, parse_gaussian94},
    ecp::{EcpDef, EcpTerm},
    post_hf::f12::{
        CABSBasis,
        cabs::{compute_cabs_singles_with_fock, extended_fock, extended_fock_fitted},
        density_fit::F12DensityFitOptions,
    },
    scf::{RhfConvergence, run_scf_in_space_with_convergence},
};
use serde_json::Value;

fn matrix(v: &Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
            .collect(),
    )
}

#[test]
fn every_fitted_fock_and_singles_response_component_matches_pyscf_including_nonlocal_ecp() {
    let data: Value =
        serde_json::from_str(include_str!("data/f12_fitted_fock_reference.json")).unwrap();
    let mut count = 0;
    let mut maximum = 0f64;
    let mut spectrum_maximum = 0f64;
    let mut raw_response_maximum = 0f64;
    let mut check = |a: f64, b: f64| {
        count += 1;
        maximum = maximum.max((a - b).abs());
        assert!(
            (a - b).abs() < 1e-10,
            "{a} vs {b}; error {:e}",
            (a - b).abs()
        );
    };
    for nonlocal in [false, true] {
        let mut atoms: Vec<_> = data["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| Atom::new("He", std::array::from_fn(|axis| p[axis].as_f64().unwrap())))
            .collect();
        if nonlocal {
            for atom in &mut atoms {
                let term = |exponent, coefficient| EcpTerm {
                    power: 2,
                    exponent,
                    coefficient,
                };
                atom.ecp = Some(std::sync::Arc::new(EcpDef {
                    symbol: "He".into(),
                    n_core: 0,
                    l_max: 2,
                    local: vec![term(0.7, 0.17)],
                    semilocal: vec![vec![term(0.53, 0.11)], vec![term(0.61, -0.07)]],
                }));
            }
        }
        let basis = |key: &str| build_basis(&atoms, &parse_gaussian94(data[key].as_str().unwrap()));
        let obs = basis("obs_g94");
        let aux = basis("aux_g94");
        let fitting = basis("fit_g94");
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
        let union: Vec<_> = obs.iter().chain(&aux).cloned().collect();
        let mut overlap = Matrix::zeros(union.len(), union.len());
        for mu in 0..union.len() {
            for nu in 0..union.len() {
                overlap[(mu, nu)] = hf_rs::integrals::contracted_overlap(&union[mu], &union[nu]);
            }
        }
        let mut occupied = Matrix::zeros(obs.len() + aux.len(), hf.n_occ);
        for mu in 0..obs.len() {
            for i in 0..hf.n_occ {
                occupied[(mu, i)] = hf.mo_coefficients[(mu, i)];
            }
        }
        for case in data["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["nonlocal_ecp"].as_bool().unwrap() == nonlocal)
        {
            let mut fit = fitting.clone();
            if case["duplicated"].as_bool().unwrap() {
                // Duplicate complete S shells at both centers.
                for atom in 0..atoms.len() {
                    fit.push(
                        fitting
                            .iter()
                            .find(|g| g.atom_index == atom)
                            .unwrap()
                            .clone(),
                    );
                }
            }
            let options = F12DensityFitOptions {
                auxiliary_basis_kind: if case["spherical"].as_bool().unwrap() {
                    BasisKind::Spherical
                } else {
                    BasisKind::Cartesian
                },
                ..Default::default()
            };
            let fock = extended_fock_fitted(&hf, &atoms, &obs, &aux, &fit, options).unwrap();
            check(hf.total_energy, case["e_scf"].as_f64().unwrap());
            for (actual, expected) in [
                (&fock.fock, &case["fock"]),
                (&fock.exchange, &case["half_exchange"]),
            ] {
                let expected = matrix(expected);
                assert_eq!((actual.rows, actual.cols), (expected.rows, expected.cols));
                for (&a, b) in actual.data.iter().zip(expected.data) {
                    check(a, b);
                }
            }
            let singles = compute_cabs_singles_with_fock(&hf, &cabs, &fock, 0).unwrap();
            check(singles.energy, case["singles_energy"].as_f64().unwrap());
            let expected = case["virtual_energies"].as_array().unwrap();
            assert_eq!(singles.virtual_energies.len(), expected.len());
            for (&a, b) in singles.virtual_energies.iter().zip(expected) {
                // Match the existing CABS spectrum check: projected nearly
                // dependent auxiliary modes amplify overlap roundoff here.
                let error = (a - b.as_f64().unwrap()).abs();
                spectrum_maximum = spectrum_maximum.max(error);
                assert!(error < 1e-9);
            }
            let response = occupied
                .dot(&singles.amplitudes)
                .dot(&singles.virtual_coefficients.transpose());
            for (&a, b) in response
                .data
                .iter()
                .zip(matrix(&case["singles_response"]).data)
            {
                // Raw AO response coefficients inherit the same near-null
                // metric amplification as the existing CABS oracle tests.
                let error = (a - b).abs();
                raw_response_maximum = raw_response_maximum.max(error);
                assert!(error < 1e-9);
            }
            let metric_response = overlap.dot(&response).dot(&overlap);
            for (&a, b) in metric_response
                .data
                .iter()
                .zip(matrix(&case["singles_metric_response"]).data)
            {
                check(a, b);
            }
            let obs_error = (0..obs.len())
                .flat_map(|mu| (0..obs.len()).map(move |nu| (mu, nu)))
                .map(|(mu, nu)| (fock.fock[(mu, nu)] - hf.fock_matrix[(mu, nu)]).abs())
                .fold(0f64, f64::max);
            assert!(
                obs_error > 1e-7,
                "raw fitted OBS Fock must not be silently replaced"
            );
        }
        assert!(
            extended_fock_fitted(
                &hf,
                &atoms,
                &obs,
                &aux,
                &fitting,
                F12DensityFitOptions {
                    workspace_budget_bytes: 1,
                    ..Default::default()
                }
            )
            .is_err()
        );
        let mut invalid = hf.clone();
        invalid.converged = false;
        assert!(
            extended_fock_fitted(&invalid, &atoms, &obs, &aux, &fitting, Default::default())
                .is_err()
        );
    }
    println!("fitted Fock/CABS response components: {count}, maximum {maximum:e}");
    println!(
        "CABS spectrum maximum {spectrum_maximum:e}; raw AO response maximum {raw_response_maximum:e} (existing CABS 1e-9 convention)"
    );
}

#[test]
fn complete_density_product_gives_exact_extended_fock() {
    let atoms = [Atom::new("He", [0.1, -0.2, 0.3])];
    let build = |exponent| -> Vec<ContractedGaussian> {
        build_basis(
            &atoms,
            &parse_gaussian94(&format!("He 0\nS 1 1\n{exponent} 1\n****")),
        )
    };
    let obs = build(0.7);
    let fitting = build(1.4);
    let hf = hf_rs::scf::run_scf_in_space(
        &atoms,
        &obs,
        0,
        100,
        1e-13,
        hf_rs::diis::ConvAlgo::Cdiis,
        None,
        BasisKind::Cartesian,
    )
    .unwrap();
    let exact = extended_fock(&hf, &atoms, &obs, &[]).unwrap();
    let fitted =
        extended_fock_fitted(&hf, &atoms, &obs, &[], &fitting, Default::default()).unwrap();
    assert!((exact.fock[(0, 0)] - fitted.fock[(0, 0)]).abs() < 1e-12);
    assert!((exact.exchange[(0, 0)] - fitted.exchange[(0, 0)]).abs() < 1e-12);
}
