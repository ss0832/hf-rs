// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    Matrix,
    ao::BasisKind,
    basis::{Atom, ContractedGaussian, build_basis, parse_gaussian94},
    context::AuxBasis,
    post_hf::{
        cc::CCSDOptions,
        f12::{CABSBasis, CCSDF12Variant, MP2F12Options, RHFCCSDF12, RHFMP2F12},
    },
    scf::{HFResult, RhfConvergence, run_scf_in_space_with_convergence},
};
use serde_json::Value;
use std::collections::BTreeMap;

fn matrix(v: &Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
            .collect(),
    )
}
fn setup(
    mixed: bool,
) -> (
    Vec<Atom>,
    Vec<ContractedGaussian>,
    HFResult,
    MP2F12Options,
    Value,
    Vec<f64>,
) {
    let input: Value =
        serde_json::from_str(include_str!("data/psi4_mp2_f12_published_reference.json")).unwrap();
    let case = &input["cases"][0];
    let expected: Value = serde_json::from_str(if mixed {
        include_str!("data/f12_mixed_fock_water_reference.json")
    } else {
        include_str!("data/f12_fitted_water_reference.json")
    })
    .unwrap();
    let atoms: Vec<_> = case["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            Atom::new(
                a[0].as_str().unwrap(),
                std::array::from_fn(|k| a[1][k].as_f64().unwrap()),
            )
        })
        .collect();
    let obs = build_basis(&atoms, &parse_gaussian94(case["obs_g94"].as_str().unwrap()));
    let aux = build_basis(&atoms, &parse_gaussian94(case["aux_g94"].as_str().unwrap()));
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
        hf_rs::diis::ConvAlgo::Cdiis,
        None,
        BasisKind::Spherical,
    )
    .unwrap();
    let cabs = CABSBasis::from_reference(&hf, &obs, &aux, 1e-8).unwrap();
    assert_eq!(cabs.n_cabs, expected["n_cabs"].as_u64().unwrap() as usize);
    let alignment = hf
        .mo_coefficients
        .transpose()
        .dot(&hf.overlap_matrix)
        .dot(&matrix(&expected["mo_coefficients"]));
    let phases = (0..alignment.rows)
        .map(|i| {
            assert!(alignment[(i, i)].abs() > 1. - 1e-9);
            alignment[(i, i)].signum()
        })
        .collect();
    let fitting = AuxBasis::from_builtin(&atoms, "aug-cc-pvdz-rifit").unwrap();
    let options = MP2F12Options {
        custom_cabs: Some(cabs),
        fock_fitting_basis: mixed.then(|| fitting.clone()),
        fitting_basis: Some(fitting),
        fitting_basis_kind: if mixed {
            BasisKind::Spherical
        } else {
            BasisKind::Cartesian
        },
        fock_fitting_basis_kind: if mixed {
            BasisKind::Spherical
        } else {
            BasisKind::Cartesian
        },
        fit_orbital_mp2: mixed,
        frozen_core: 1,
        ..Default::default()
    };
    (atoms, obs, hf, options, expected, phases)
}

#[derive(Default)]
struct Errors {
    maximum: BTreeMap<&'static str, f64>,
    count: usize,
}
impl Errors {
    fn add(&mut self, key: &'static str, actual: f64, expected: f64) {
        assert!(actual.is_finite() && expected.is_finite());
        let error = (actual - expected).abs();
        let maximum = self.maximum.entry(key).or_default();
        *maximum = maximum.max(error);
        self.count += 1;
    }
    fn check(&self, mixed: bool) {
        println!(
            "Water {} F12: {} components; maxima {:?}",
            if mixed { "mixed" } else { "fitted" },
            self.count,
            self.maximum
        );
        assert!(
            self.maximum.values().all(|&x| x < 1e-10),
            "{:?}",
            self.maximum
        );
    }
}

#[test]
fn all_water_mp2_f12_pair_and_energy_components_match_independent_block_operators() {
    for mixed in [false, true] {
        let (atoms, obs, hf, options, expected, phases) = setup(mixed);
        let result = RHFMP2F12::new(&hf, &atoms, &obs)
            .with_options(options)
            .compute_checked()
            .unwrap();
        let mut errors = Errors::default();
        for (actual, key) in [
            (result.hf_energy, "e_hf"),
            (result.cabs_singles_energy, "e_cabs"),
            (result.mp2_correlation_energy, "e_mp2"),
            (result.f12_correlation_energy, "e_f12"),
            (result.total_energy, "e_total_mp2"),
        ] {
            errors.add(key, actual, expected[key].as_f64().unwrap());
        }
        assert_eq!(
            result.pairs.len(),
            expected["pairs"].as_array().unwrap().len()
        );
        for (pair, reference) in result
            .pairs
            .iter()
            .zip(expected["pairs"].as_array().unwrap())
        {
            assert_eq!(
                pair.pair,
                std::array::from_fn(|i| reference["pair"][i].as_u64().unwrap() as usize)
            );
            for (actual, key) in [
                (pair.v, "v"),
                (pair.x, "x"),
                (pair.b, "b"),
                (pair.v_bar, "v_bar"),
                (pair.b_bar, "b_bar"),
            ] {
                for k in 0..2 {
                    errors.add(key, actual[k], reference[key][k].as_f64().unwrap());
                }
            }
            for (actual, key) in [(&pair.c, "c"), (&pair.g, "g")] {
                let target = matrix(&reference[key]);
                assert_eq!((actual.rows, actual.cols), (target.rows, target.cols));
                for a in 0..actual.rows {
                    for b in 0..actual.cols {
                        let value = actual[(a, b)]
                            * phases[hf.n_occ + a]
                            * phases[hf.n_occ + b]
                            * phases[pair.pair[0]]
                            * phases[pair.pair[1]];
                        errors.add(key, value, target[(a, b)]);
                    }
                }
            }
        }
        assert_eq!(result.fitted_fock, mixed);
        assert_eq!(result.fitted_orbital_mp2, mixed);
        errors.check(mixed);
    }
}

#[test]
fn all_water_ccsd_f12b_amplitudes_and_nonzero_triples_match_public_pyscf_iteration() {
    for mixed in [false, true] {
        let (atoms, obs, hf, options, expected, phases) = setup(mixed);
        let projected =
            hf_rs::post_hf::f12::molecular::molecular_ccsd_projected_coulomb_fitted_in_space(
                &hf,
                &obs,
                options.custom_cabs.as_ref().unwrap(),
                &hf_rs::post_hf::f12::GeminalCorrelationFactor::new(1.)
                    .unwrap()
                    .gaussian()
                    .unwrap(),
                options.fitting_basis.as_ref().unwrap(),
                options.fitting_basis_kind,
                1,
                options.workspace_budget_bytes,
            )
            .unwrap();
        let mut errors = Errors::default();
        assert_eq!(projected.len(), 16);
        for (ij, actual) in projected.iter().enumerate() {
            let target = matrix(&expected["projected_coulomb"][ij]);
            for p in 0..actual.rows {
                for q in 0..actual.cols {
                    let value = actual[(p, q)]
                        * phases[p]
                        * phases[q]
                        * phases[1 + ij / 4]
                        * phases[1 + ij % 4];
                    errors.add("projected_coulomb", value, target[(p, q)]);
                }
            }
        }
        let result = RHFCCSDF12::new(&hf, &atoms, &obs, CCSDF12Variant::F12b)
            .with_f12_options(options)
            .with_triples(true)
            .with_ccsd_options(CCSDOptions {
                max_iterations: 200,
                energy_tol: 1e-13,
                residual_tol: 1e-12,
                ..Default::default()
            })
            .compute_checked()
            .unwrap();
        for (actual, target, key) in [
            (
                result.total_energy,
                expected["e_total_b"].as_f64().unwrap(),
                "e_total_b",
            ),
            (
                result.ccsd_correlation_energy,
                expected["e_cc"].as_f64().unwrap(),
                "e_cc",
            ),
            (
                result.f12_correction_energy,
                expected["e_fixed"].as_f64().unwrap(),
                "e_fixed",
            ),
            (
                result.coupling_energy,
                expected["e_coupling_a"].as_f64().unwrap()
                    + expected["e_additional_b"].as_f64().unwrap(),
                "coupling",
            ),
            (
                result.triples_energy.unwrap(),
                expected["e_triples"].as_f64().unwrap(),
                "triples",
            ),
            (
                result.scaled_triples_energy.unwrap(),
                expected["e_scaled_triples"].as_f64().unwrap(),
                "scaled_triples",
            ),
        ] {
            errors.add(key, actual, target);
        }
        assert!(result.ccsd_result.converged);
        assert!(result.triples_energy.unwrap().abs() > 1e-4);
        let cc = &result.ccsd_result;
        assert_eq!((cc.n_occ, cc.n_virt), (4, 43));
        for i in 0..cc.n_occ {
            for a in 0..cc.n_virt {
                let p = i * cc.n_virt + a;
                let actual = cc.t1_amplitudes[p] * phases[1 + i] * phases[hf.n_occ + a];
                errors.add("t1", actual, expected["t1"][p].as_f64().unwrap());
                for j in 0..cc.n_occ {
                    for b in 0..cc.n_virt {
                        let p = ((i * cc.n_occ + j) * cc.n_virt + a) * cc.n_virt + b;
                        let actual = cc.t2_amplitudes[p]
                            * phases[1 + i]
                            * phases[1 + j]
                            * phases[hf.n_occ + a]
                            * phases[hf.n_occ + b];
                        errors.add("t2", actual, expected["t2"][p].as_f64().unwrap());
                    }
                }
            }
        }
        assert_eq!(result.fitted_fock, mixed);
        assert_eq!(result.fitted_orbital_mp2, mixed);
        errors.add(
            "scaling_mp2",
            result.mp2_correlation_energy,
            expected["e_mp2"].as_f64().unwrap(),
        );
        errors.add(
            "scaling_mp2_f12",
            result.mp2_f12_correlation_energy,
            expected["e_mp2"].as_f64().unwrap() + expected["e_f12"].as_f64().unwrap(),
        );
        errors.check(mixed);
    }
}
