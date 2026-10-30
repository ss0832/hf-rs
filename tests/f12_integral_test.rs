// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::basis::{ContractedGaussian, Primitive};
use hf_rs::post_hf::f12::operators::*;
fn exponents(case: &serde_json::Value) -> [f64; 4] {
    std::array::from_fn(|i| case["exponents"][i].as_f64().unwrap())
}
fn centers(case: &serde_json::Value) -> [[f64; 3]; 4] {
    std::array::from_fn(|i| std::array::from_fn(|a| case["centers"][i][a].as_f64().unwrap()))
}
fn powers(case: &serde_json::Value) -> [[usize; 3]; 4] {
    std::array::from_fn(|i| {
        std::array::from_fn(|a| case["powers"][i][a].as_u64().unwrap() as usize)
    })
}
#[test]
fn all_cartesian_components_through_k_match_independent_quadrature_and_zero_exponent_limits() {
    let ordinary: serde_json::Value =
        serde_json::from_str(include_str!("data/f12_integral_reference.json")).unwrap();
    let high: serde_json::Value =
        serde_json::from_str(include_str!("data/f12_high_degree_reference.json")).unwrap();
    for (index, case) in ordinary["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(high["cases"].as_array().unwrap())
        .enumerate()
    {
        let exponent = case["gamma"].as_f64().unwrap();
        let e = exponents(case);
        let l = powers(case);
        let c = centers(case);
        for (kernel, name) in [
            (GaussianKernel::Gaussian(exponent), "gaussian"),
            (GaussianKernel::GaussianCoulomb(exponent), "coulomb"),
            (GaussianKernel::GaussianR2(exponent), "r2"),
            (
                GaussianKernel::KineticCommutator(exponent),
                "kinetic_commutator",
            ),
            (GaussianKernel::Gaussian(0.), "zero_gaussian"),
            (GaussianKernel::GaussianCoulomb(0.), "zero_coulomb"),
        ] {
            let actual = primitive_operator(e, l, c, kernel).unwrap();
            let expected = case[name].as_f64().unwrap();
            assert!(
                (actual - expected).abs() < 2e-11 * expected.abs().max(1.),
                "case {index} {name}: {actual} vs {expected}"
            );
        }
        let original =
            primitive_operator(e, l, c, GaussianKernel::KineticCommutator(exponent)).unwrap();
        let reversed = primitive_operator(
            [e[1], e[0], e[3], e[2]],
            [l[1], l[0], l[3], l[2]],
            [c[1], c[0], c[3], c[2]],
            GaussianKernel::KineticCommutator(exponent),
        )
        .unwrap();
        assert!((original + reversed).abs() < 2e-11);
        assert!(
            primitive_operator(e, l, c, GaussianKernel::KineticCommutator(0.0))
                .unwrap()
                .abs()
                < 2e-11
        );
    }
}
#[test]
fn every_geminal_operator_has_correct_contraction_products_and_commutator_factor() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/f12_integral_reference.json")).unwrap();
    let case = &data["contracted"];
    let e = exponents(case);
    let l = powers(case);
    let c = centers(case);
    let basis = std::array::from_fn::<_, 4, _>(|i| ContractedGaussian {
        atom_index: 0,
        center: c[i],
        l: l[i],
        primitives: vec![Primitive {
            exponent: e[i],
            coefficient: case["coefficients"][i].as_f64().unwrap(),
        }],
    });
    let factor = GaussianGeminal::new(
        case["terms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|term| (term[0].as_f64().unwrap(), term[1].as_f64().unwrap()))
            .collect(),
    )
    .unwrap();
    for (operator, name) in [
        (GeminalOperator::Factor, "factor"),
        (GeminalOperator::FactorOverR, "factor_over_r"),
        (GeminalOperator::FactorSquared, "squared"),
        (GeminalOperator::PositiveDoubleCommutator, "commutator"),
        (GeminalOperator::KineticCommutator, "kinetic_commutator"),
    ] {
        let actual = contracted_operator(
            &basis[0], &basis[1], &basis[2], &basis[3], &factor, operator,
        )
        .unwrap();
        let expected = case["expected"][name].as_f64().unwrap();
        assert!(
            (actual - expected).abs() < 1e-12,
            "{name}: {actual} vs {expected}"
        );
    }
    assert!(GaussianGeminal::new(vec![(-0.1, 1.)]).is_err());
    assert!(primitive_operator(e, l, c, GaussianKernel::Gaussian(f64::NAN)).is_err());
}
