// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::post_hf::f12::{
    density_fit::{
        F12DensityFitOptions, F12PairFitter, FittedF12Kernel, fitted_coulomb_exchange_rhf,
    },
    operators::{GaussianGeminal, GeminalOperator, PreparedGeminalOperator},
};
use hf_rs::*;

fn matrix(v: &serde_json::Value) -> Matrix {
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
fn every_pure_auxiliary_fitted_kernel_and_streamed_fock_component_matches_public_oracle() {
    let data = data();
    let (obs, aux, factor) = setup(&data);
    let options = F12DensityFitOptions {
        auxiliary_basis_kind: ao::BasisKind::Spherical,
        ..Default::default()
    };
    let occupied = matrix(&data["occupied"]);
    let left = matrix(&data["left"]);
    let fitter = F12PairFitter::new(&obs, &aux, &left, &occupied, options).unwrap();
    assert_eq!(
        fitter.rank(),
        data["spherical_rank"].as_u64().unwrap() as usize
    );
    let mut count = 0;
    let mut maximum = 0.0_f64;
    let mut check = |a: f64, b: f64| {
        count += 1;
        maximum = maximum.max((a - b).abs());
        assert!((a - b).abs() < 1e-10, "{a} vs {b}");
    };
    for case in data["cases"].as_array().unwrap() {
        let op = operator(case["name"].as_str().unwrap());
        let kernel = FittedF12Kernel::build(&obs, &aux, &factor, op, options).unwrap();
        assert_eq!(kernel.rank(), fitter.rank());
        for p in 0..obs.len() {
            for q in 0..obs.len() {
                for r in 0..obs.len() {
                    for s in 0..obs.len() {
                        check(
                            kernel.integral(p, q, r, s).unwrap(),
                            case["spherical_fitted"][scf::compound_index(p, q)]
                                [scf::compound_index(r, s)]
                            .as_f64()
                            .unwrap(),
                        );
                    }
                }
            }
        }
        let dense = kernel.mixed_pairs(&left, &occupied).unwrap();
        let early = fitter
            .kernel(&factor, op)
            .unwrap()
            .pairs(0..4, 0..4)
            .unwrap();
        for (pair, block) in early.iter().enumerate() {
            let expected = matrix(&case["spherical_mixed"][pair / 2][pair % 2]);
            for ((&a, &b), &c) in block.data.iter().zip(&dense[pair].data).zip(&expected.data) {
                check(a, c);
                check(b, c);
            }
        }
    }
    for case in data["fock_components"].as_array().unwrap() {
        let options = F12DensityFitOptions {
            auxiliary_basis_kind: if case["kind"] == "spherical" {
                ao::BasisKind::Spherical
            } else {
                ao::BasisKind::Cartesian
            },
            ..Default::default()
        };
        let (j, k) = fitted_coulomb_exchange_rhf(&obs, &aux, &occupied, options).unwrap();
        for (actual, expected) in [(&j, &case["coulomb"]), (&k, &case["exchange"])] {
            for (&a, b) in actual.data.iter().zip(matrix(expected).data) {
                check(a, b);
            }
        }
    }
    println!("pure auxiliary/streamed J/K components: {count}, maximum {maximum:e}");
    for options in [
        F12DensityFitOptions {
            workspace_budget_bytes: 1,
            ..options
        },
        F12DensityFitOptions {
            metric_cutoff: f64::NAN,
            ..options
        },
    ] {
        assert!(fitted_coulomb_exchange_rhf(&obs, &aux, &occupied, options).is_err());
    }
    assert!(
        fitted_coulomb_exchange_rhf(&obs, &aux, &Matrix::zeros(obs.len(), 0), options).is_err()
    );
    assert!(
        fitted_coulomb_exchange_rhf(&obs, &aux, &Matrix::zeros(obs.len() + 1, 1), options).is_err()
    );
}
fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/f12_density_fit_reference.json")).unwrap()
}
fn setup(
    data: &serde_json::Value,
) -> (
    Vec<ContractedGaussian>,
    Vec<ContractedGaussian>,
    GaussianGeminal,
) {
    let at = |i| {
        vec![Atom::new(
            "He",
            std::array::from_fn(|k| data["positions"][i][k].as_f64().unwrap()),
        )]
    };
    let obs = build_basis(&at(0), &parse_gaussian94(data["obs_g94"].as_str().unwrap()));
    let aux = build_basis(&at(1), &parse_gaussian94(data["aux_g94"].as_str().unwrap()));
    let factor = GaussianGeminal::new(
        data["terms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (t[0].as_f64().unwrap(), t[1].as_f64().unwrap()))
            .collect(),
    )
    .unwrap();
    (obs, aux, factor)
}
fn operator(name: &str) -> GeminalOperator {
    match name {
        "factor" => GeminalOperator::Factor,
        "factor_over_r" => GeminalOperator::FactorOverR,
        "squared" => GeminalOperator::FactorSquared,
        "commutator" => GeminalOperator::PositiveDoubleCommutator,
        _ => panic!("unknown fixture operator"),
    }
}

#[test]
fn every_two_three_center_and_robust_fitted_component_matches_public_psi4() {
    let data = data();
    let (obs, aux, factor) = setup(&data);
    let mut count = 0;
    let mut max_error = 0.0_f64;
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let op = operator(name);
        let kernel = PreparedGeminalOperator::new(&factor, op).unwrap();
        for p in 0..aux.len() {
            for q in 0..aux.len() {
                let actual = kernel.two_center(&aux[p], &aux[q]).unwrap();
                let expected = case["two"][p][q].as_f64().unwrap();
                max_error = max_error.max((actual - expected).abs());
                count += 1;
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "{name} two {p} {q}: {actual} vs {expected}"
                );
            }
        }
        for p in 0..aux.len() {
            for mu in 0..obs.len() {
                for nu in 0..=mu {
                    let actual = kernel.three_center(&obs[mu], &obs[nu], &aux[p]).unwrap();
                    let expected = case["three"][p][scf::compound_index(mu, nu)]
                        .as_f64()
                        .unwrap();
                    max_error = max_error.max((actual - expected).abs());
                    count += 1;
                    assert!(
                        (actual - expected).abs() < 1e-10,
                        "{name} three {p} {mu} {nu}: {actual} vs {expected}"
                    );
                }
            }
        }
        let fitted = FittedF12Kernel::build(&obs, &aux, &factor, op, Default::default()).unwrap();
        let mut redundant = aux.clone();
        redundant.push(aux[0].clone());
        let rank_drop =
            FittedF12Kernel::build(&obs, &redundant, &factor, op, Default::default()).unwrap();
        assert_eq!(rank_drop.rank(), fitted.rank());
        for p in 0..obs.len() {
            for q in 0..obs.len() {
                for r in 0..obs.len() {
                    for s in 0..obs.len() {
                        let actual = fitted.integral(p, q, r, s).unwrap();
                        let expected = case["fitted"][scf::compound_index(p, q)]
                            [scf::compound_index(r, s)]
                        .as_f64()
                        .unwrap();
                        max_error = max_error.max((actual - expected).abs());
                        count += 1;
                        assert!(
                            (actual - expected).abs() < 1e-10,
                            "{name} fit {p} {q} {r} {s}: {actual} vs {expected}"
                        );
                        assert!((rank_drop.integral(p, q, r, s).unwrap() - expected).abs() < 1e-10);
                    }
                }
            }
        }
        let mixed = fitted
            .mixed_pairs(&matrix(&data["left"]), &matrix(&data["occupied"]))
            .unwrap();
        let fitter = F12PairFitter::new(
            &obs,
            &aux,
            &matrix(&data["left"]),
            &matrix(&data["occupied"]),
            Default::default(),
        )
        .unwrap();
        assert_eq!(fitter.rank(), fitted.rank());
        let occupied_first = fitter.kernel(&factor, op).unwrap();
        let early_pairs = occupied_first.pairs(0..4, 0..4).unwrap();
        let rectangular = occupied_first.pairs(1..4, 0..2).unwrap();
        for (pair, actual) in mixed.iter().enumerate() {
            let expected = matrix(&case["mixed"][pair / 2][pair % 2]);
            for (a, b) in actual.data.iter().zip(expected.data) {
                max_error = max_error.max((a - b).abs());
                count += 1;
                assert!(
                    (a - b).abs() < 1e-10,
                    "{name} mixed pair {pair}: {a} vs {b}"
                );
            }
            for (a, b) in early_pairs[pair].data.iter().zip(&actual.data) {
                assert!((a - b).abs() < 1e-10);
            }
            for p in 0..3 {
                for q in 0..2 {
                    assert!((rectangular[pair][(p, q)] - actual[(p + 1, q)]).abs() < 1e-10);
                }
            }
        }
        assert!(occupied_first.pairs(0..5, 0..4).is_err());
        assert!(
            occupied_first
                .pairs(std::ops::Range { start: 2, end: 1 }, 0..4)
                .is_err()
        );
        assert!(
            fitter
                .kernel(&factor, GeminalOperator::KineticCommutator)
                .is_err()
        );
        assert!(fitted.integral(obs.len(), 0, 0, 0).is_err());
        let mut malformed = matrix(&data["left"]);
        malformed.data.pop();
        assert!(
            fitted
                .mixed_pairs(&malformed, &matrix(&data["occupied"]))
                .is_err()
        );
        assert!(
            fitted
                .mixed_pairs(
                    &Matrix::zeros(obs.len(), 100),
                    &Matrix::zeros(obs.len(), 100)
                )
                .is_err()
        );
        assert!(
            fitted
                .mixed_pairs(&matrix(&data["left"]), &Matrix::zeros(obs.len(), 0))
                .unwrap()
                .is_empty()
        );
    }
    println!("two/three/robust/mixed components: {count}, maximum error {max_error:e}");
}

#[test]
fn coulomb_and_complete_product_limits_and_invalid_controls() {
    let data = data();
    let (obs, aux, factor) = setup(&data);
    let coulomb = GaussianGeminal::new(vec![(0.0, 1.0)]).unwrap();
    let fitted = FittedF12Kernel::build(
        &obs,
        &aux,
        &coulomb,
        GeminalOperator::FactorOverR,
        Default::default(),
    )
    .unwrap();
    let standard = hf_rs::df::DfTensor::build(&obs, &aux, 1e-10).unwrap();
    let fitter = F12PairFitter::new(
        &obs,
        &aux,
        &matrix(&data["left"]),
        &matrix(&data["occupied"]),
        Default::default(),
    )
    .unwrap();
    let direct = fitter.coulomb_kernel().pairs(0..4, 0..4).unwrap();
    let robust = fitted
        .mixed_pairs(&matrix(&data["left"]), &matrix(&data["occupied"]))
        .unwrap();
    assert_eq!(fitter.coulomb_modes().cols, 8);
    for (a, b) in direct.iter().zip(&robust) {
        for (x, y) in a.data.iter().zip(&b.data) {
            assert!((x - y).abs() < 1e-10);
        }
    }
    for p in 0..obs.len() {
        for q in 0..obs.len() {
            for r in 0..obs.len() {
                for s in 0..obs.len() {
                    assert!(
                        (fitted.integral(p, q, r, s).unwrap() - standard.eri(p, q, r, s)).abs()
                            < 1e-10
                    );
                }
            }
        }
    }
    let atoms = [Atom::new("He", [0.1, -0.2, 0.3])];
    let orbital = build_basis(&atoms, &parse_gaussian94("He 0\nS 1 1\n0.7 1\n****"));
    let product = build_basis(&atoms, &parse_gaussian94("He 0\nS 1 1\n1.4 1\n****"));
    for op in [
        GeminalOperator::Factor,
        GeminalOperator::FactorOverR,
        GeminalOperator::FactorSquared,
        GeminalOperator::PositiveDoubleCommutator,
    ] {
        let kernel = PreparedGeminalOperator::new(&factor, op).unwrap();
        let exact = kernel
            .contracted(&orbital[0], &orbital[0], &orbital[0], &orbital[0])
            .unwrap();
        let fit =
            FittedF12Kernel::build(&orbital, &product, &factor, op, Default::default()).unwrap();
        assert!((fit.integral(0, 0, 0, 0).unwrap() - exact).abs() < 1e-12);
    }
    for options in [
        F12DensityFitOptions {
            workspace_budget_bytes: 1,
            ..Default::default()
        },
        F12DensityFitOptions {
            metric_cutoff: 0.0,
            ..Default::default()
        },
    ] {
        assert!(
            FittedF12Kernel::build(&obs, &aux, &factor, GeminalOperator::Factor, options).is_err()
        );
    }
    let kinetic =
        PreparedGeminalOperator::new(&factor, GeminalOperator::KineticCommutator).unwrap();
    assert!(kinetic.two_center(&aux[0], &aux[1]).is_err());
    assert!(kinetic.three_center(&obs[0], &obs[1], &aux[0]).is_err());
    assert!(
        FittedF12Kernel::build(
            &obs,
            &aux,
            &factor,
            GeminalOperator::KineticCommutator,
            Default::default()
        )
        .is_err()
    );
    let physical = PreparedGeminalOperator::new(&factor, GeminalOperator::Factor).unwrap();
    let mut invalid = aux[0].clone();
    invalid.primitives[0].exponent = 0.0;
    assert!(physical.two_center(&invalid, &aux[0]).is_err());
    assert!(physical.three_center(&obs[0], &obs[1], &invalid).is_err());
    let left = matrix(&data["left"]);
    let occupied = matrix(&data["occupied"]);
    let sparse_left = left.clone();
    let mut sparse_occupied = occupied.clone();
    for mu in 2..obs.len() {
        for i in 0..sparse_occupied.cols {
            sparse_occupied[(mu, i)] = 0.0;
        }
    }
    let fitter = F12PairFitter::new(
        &obs,
        &aux,
        &sparse_left,
        &sparse_occupied,
        Default::default(),
    )
    .unwrap();
    let fast = fitter
        .kernel(&factor, GeminalOperator::Factor)
        .unwrap()
        .pairs(0..4, 0..4)
        .unwrap();
    let expected = FittedF12Kernel::build(
        &obs,
        &aux,
        &factor,
        GeminalOperator::Factor,
        Default::default(),
    )
    .unwrap()
    .mixed_pairs(&sparse_left, &sparse_occupied)
    .unwrap();
    for (a, b) in fast.iter().zip(expected) {
        for (x, y) in a.data.iter().zip(b.data) {
            assert!((x - y).abs() < 1e-10);
        }
    }
    assert!(
        F12PairFitter::new(
            &obs,
            &aux,
            &left,
            &occupied,
            F12DensityFitOptions {
                workspace_budget_bytes: 1,
                ..Default::default()
            }
        )
        .is_err()
    );
    let empty = F12PairFitter::new(
        &obs,
        &aux,
        &left,
        &Matrix::zeros(obs.len(), 0),
        Default::default(),
    )
    .unwrap();
    assert!(
        empty
            .kernel(&factor, GeminalOperator::Factor)
            .unwrap()
            .pairs(0..4, 0..4)
            .unwrap()
            .is_empty()
    );
}
