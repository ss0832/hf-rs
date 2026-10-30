// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    DeterminantSpace, Matrix,
    post_hf::mr_dsrg::{
        density_cumulants::{CiEnsembleMember, ExactDensityCumulants, exact_density_cumulants},
        gno::{GnoAlgebra, GnoReference},
        imsrg2::{No2bOperator, no2b_commutator},
        sr_dsrg::bch_no2b,
    },
};
const BUDGET: usize = 256 * 1024 * 1024;
fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/gno_reference.json")).unwrap()
}
fn number(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap()
}
fn vector(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(number).collect()
}
fn compute(case: &serde_json::Value) -> ExactDensityCumulants {
    let n = case["n_spatial"].as_u64().unwrap() as usize;
    let records = case["members"].as_array().unwrap();
    let spaces: Vec<_> = records
        .iter()
        .map(|m| {
            DeterminantSpace::new(
                n,
                m["na"].as_u64().unwrap() as usize,
                m["nb"].as_u64().unwrap() as usize,
            )
        })
        .collect();
    let coefficients: Vec<_> = records.iter().map(|m| vector(&m["coefficients"])).collect();
    let members: Vec<_> = spaces
        .iter()
        .zip(&coefficients)
        .zip(records)
        .map(|((space, c), m)| CiEnsembleMember {
            space,
            coefficients: c,
            weight: number(&m["weight"]),
        })
        .collect();
    exact_density_cumulants(&members, BUDGET).unwrap()
}
fn pairs(n: usize) -> Vec<[usize; 2]> {
    (0..n)
        .flat_map(|p| (p + 1..n).map(move |q| [p, q]))
        .collect()
}
fn triples(n: usize) -> Vec<[usize; 3]> {
    (0..n)
        .flat_map(|p| (p + 1..n).flat_map(move |q| (q + 1..n).map(move |r| [p, q, r])))
        .collect()
}
const PERM3: [([usize; 3], f64); 6] = [
    ([0, 1, 2], 1.),
    ([0, 2, 1], -1.),
    ([1, 0, 2], -1.),
    ([1, 2, 0], 1.),
    ([2, 0, 1], 1.),
    ([2, 1, 0], -1.),
];
fn expand2(packed: &serde_json::Value, n: usize) -> Vec<f64> {
    let groups = pairs(n);
    let mut out = vec![0.; n.pow(4)];
    for (i, &[p, q]) in groups.iter().enumerate() {
        for (j, &[r, s]) in groups.iter().enumerate() {
            let value = number(&packed[i][j]);
            for (a, b, c, d, sign) in [
                (p, q, r, s, 1.),
                (q, p, r, s, -1.),
                (p, q, s, r, -1.),
                (q, p, s, r, 1.),
            ] {
                out[((a * n + b) * n + c) * n + d] = sign * value;
            }
        }
    }
    out
}
fn expand3(packed: &serde_json::Value, n: usize) -> Vec<f64> {
    let groups = triples(n);
    let mut out = vec![0.; n.pow(6)];
    for (i, a) in groups.iter().enumerate() {
        for (j, b) in groups.iter().enumerate() {
            let value = number(&packed[i][j]);
            for (pa, sa) in PERM3 {
                for (pb, sb) in PERM3 {
                    let indices = [a[pa[0]], a[pa[1]], a[pa[2]], b[pb[0]], b[pb[1]], b[pb[2]]];
                    let address = indices.iter().fold(0usize, |v, &p| v * n + p);
                    out[address] = value * sa * sb;
                }
            }
        }
    }
    out
}

fn operator(v: &serde_json::Value) -> No2bOperator {
    let rows = v["one"].as_array().unwrap();
    No2bOperator {
        zero: number(&v["zero"]),
        one: Matrix::from_vec(
            rows.len(),
            rows.len(),
            rows.iter().flat_map(vector).collect(),
        ),
        two: vector(&v["two"]),
    }
}
fn indices(v: &serde_json::Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}
fn reference<'a>(c: &serde_json::Value, d: &'a ExactDensityCumulants) -> GnoReference<'a> {
    GnoReference::new(
        d,
        c["n_spin"].as_u64().unwrap() as usize,
        &indices(&c["core"]),
        &indices(&c["active"]),
    )
    .unwrap()
}
fn compare(a: &No2bOperator, b: &No2bOperator, tol: f64) -> f64 {
    assert_eq!(a.one.rows, b.one.rows);
    assert_eq!(a.two.len(), b.two.len());
    let mut maximum = 0f64;
    for (i, (x, y)) in std::iter::once(&a.zero)
        .chain(&a.one.data)
        .chain(&a.two)
        .zip(std::iter::once(&b.zero).chain(&b.one.data).chain(&b.two))
        .enumerate()
    {
        let error = (x - y).abs();
        maximum = maximum.max(error);
        assert!(
            error < tol,
            "coefficient {i}: {x:e} vs {y:e}, error {error:e}"
        );
    }
    maximum
}
#[test]
fn complete_gno_commutator_and_bch_match_full_fock_operator_basis_oracle() {
    let data = data();
    let mut maximum = 0f64;
    let mut count = 0;
    for case in data["cases"].as_array().unwrap() {
        let d = compute(case);
        let r = reference(case, &d);
        let algebra = GnoAlgebra::new(&r, BUDGET).unwrap();
        let h = operator(&case["h"]);
        let a = operator(&case["a"]);
        assert!(number(&case["induced_three_max"]) > 1e-4);
        let comm = algebra.commutator(&h, &a).unwrap();
        maximum = maximum.max(compare(&comm, &operator(&case["commutator"]), 1e-10));
        let (bar, order) = algebra.bch(&h, &a, 1e-14, 80).unwrap();
        assert!(order > 2);
        maximum = maximum.max(compare(&bar, &operator(&case["bch"]), 1e-10));
        let bare = algebra.denormal_order(&h).unwrap();
        maximum = maximum.max(compare(&bare, &operator(&case["bare_h"]), 1e-10));
        maximum = maximum.max(compare(&algebra.normal_order(&bare).unwrap(), &h, 1e-10));
        count += 4 * (1 + h.one.data.len() + h.two.len());
        println!("{} BCH order {order}", case["name"]);
    }
    println!("GNO full operator coefficients={count}, maximum error={maximum:e}");
}
#[test]
fn every_embedded_density_component_matches_full_fock_oracle() {
    let data = data();
    let mut maximum = 0f64;
    let mut count = 0;
    for case in data["cases"].as_array().unwrap() {
        let d = compute(case);
        let r = reference(case, &d);
        let n = r.n_spin();
        for rank in 1..=3 {
            let values = match rank {
                1 => case["densities"]["1"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(vector)
                    .collect(),
                2 => expand2(&case["densities"]["2"], n),
                3 => expand3(&case["densities"]["3"], n),
                _ => unreachable!(),
            };
            for (address, expected) in values.into_iter().enumerate() {
                let mut remainder = address;
                let mut ix = vec![0; rank * 2];
                for p in ix.iter_mut().rev() {
                    *p = remainder % n;
                    remainder /= n;
                }
                let actual = r.density(&ix[..rank], &ix[rank..]).unwrap();
                let error = (actual - expected).abs();
                maximum = maximum.max(error);
                assert!(
                    error < 1e-11,
                    "{} rank {rank} indices {ix:?}: {actual:e} vs {expected:e}",
                    case["name"]
                );
                count += 1;
            }
        }
    }
    println!("Embedded density components={count}, maximum error={maximum:e}");
}
#[test]
fn slater_limit_matches_single_reference_no2b_and_algebra_identities() {
    let data = data();
    for case in data["cases"].as_array().unwrap() {
        let d = compute(case);
        let r = reference(case, &d);
        let algebra = GnoAlgebra::new(&r, BUDGET).unwrap();
        let h = operator(&case["h"]);
        let a = operator(&case["a"]);
        let zero = No2bOperator {
            zero: 0.,
            one: Matrix::zeros(6, 6),
            two: vec![0.; 6usize.pow(4)],
        };
        compare(&algebra.commutator(&h, &h).unwrap(), &zero, 1e-12);
        compare(&algebra.bch(&h, &zero, 1e-14, 1).unwrap().0, &h, 1e-14);
        let mut negative = algebra.commutator(&a, &h).unwrap();
        negative.zero *= -1.;
        negative.one.data.iter_mut().for_each(|x| *x *= -1.);
        negative.two.iter_mut().for_each(|x| *x *= -1.);
        compare(&algebra.commutator(&h, &a).unwrap(), &negative, 1e-12);
        if case["name"] == "slater_limit" || case["name"] == "vacuum_limit" {
            let mut occ = vec![false; 6];
            if case["name"] == "slater_limit" {
                occ[0] = true;
                occ[1] = true;
            }
            compare(
                &algebra.commutator(&h, &a).unwrap(),
                &no2b_commutator(&h, &a, &occ).unwrap(),
                1e-11,
            );
            compare(
                &algebra.bch(&h, &a, 1e-14, 80).unwrap().0,
                &bch_no2b(&h, &a, &occ, 1e-14, 80).unwrap().0,
                1e-11,
            );
        }
    }
}
#[test]
fn genuine_three_body_coherence_changes_scalar_with_identical_lower_densities() {
    let data = data();
    let mut results = Vec::new();
    for case in data["coherent_scalars"].as_array().unwrap() {
        let d = compute(case);
        let r = GnoReference::active_only(&d).unwrap();
        let algebra = GnoAlgebra::new(&r, BUDGET).unwrap();
        let c = algebra
            .commutator(&operator(&case["h"]), &operator(&case["a"]))
            .unwrap();
        assert!((c.zero - number(&case["scalar"])).abs() < 1e-11);
        results.push(c);
    }
    assert!((results[0].zero - results[1].zero).abs() > 1e-3);
    results[1].zero = results[0].zero;
    compare(&results[0], &results[1], 1e-12);
}
#[test]
fn invalid_maps_tensors_budgets_and_bch_controls_are_rejected() {
    let data = data();
    let case = &data["cases"][0];
    let d = compute(case);
    assert!(GnoReference::new(&d, 6, &[0], &[0, 1, 2, 3, 4, 5]).is_err());
    assert!(GnoReference::new(&d, 6, &[], &[0, 1, 2, 3, 4, 6]).is_err());
    assert!(GnoReference::new(&d, 6, &[], &[0, 1]).is_err());
    let r = reference(case, &d);
    assert!(r.density(&[0], &[]).is_err());
    assert!(r.density(&[6], &[0]).is_err());
    assert!(r.density(&[0; 4], &[0; 4]).is_err());
    assert!(GnoAlgebra::new(&r, 1).is_err());
    let algebra = GnoAlgebra::new(&r, BUDGET).unwrap();
    let h = operator(&case["h"]);
    let a = operator(&case["a"]);
    for (t, k) in [(0., 80), (f64::NAN, 80), (1e-14, 0), (1e-14, 1001)] {
        assert!(algebra.bch(&h, &a, t, k).is_err());
    }
    assert!(algebra.bch(&h, &a, 1e-30, 1).is_err());
    let mut bad = h.clone();
    bad.two[0] = 1.;
    assert!(algebra.commutator(&bad, &a).is_err());
    bad = h.clone();
    bad.one.data[0] = f64::INFINITY;
    assert!(algebra.normal_order(&bad).is_err());
    bad = h.clone();
    bad.two.pop();
    assert!(algebra.denormal_order(&bad).is_err());
}

fn mr_data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/mr_driven_reference.json")).unwrap()
}
fn strict() -> hf_rs::post_hf::mr_dsrg::sr_dsrg::LdsrgOptions {
    hf_rs::post_hf::mr_dsrg::sr_dsrg::LdsrgOptions {
        residual_tolerance: 1e-13,
        energy_tolerance: 1e-14,
        bch_tolerance: 1e-14,
        ..Default::default()
    }
}
#[test]
fn mr_driven_energies_and_all_amplitudes_match_independent_pade_roots_and_lambda_fits() {
    use hf_rs::post_hf::mr_dsrg::mr_driven::MrDsrgEquations;
    let data = mr_data();
    let mut maximum = 0f64;
    let mut count = 0;
    for case in data["cases"].as_array().unwrap() {
        let d = compute(case);
        let r = reference(case, &d);
        let equations = MrDsrgEquations::new(&r, BUDGET).unwrap();
        let h = operator(&case["h"]);
        let flow = number(&case["flow"]);
        let solution = equations.solve_ldsrg2(&h, flow, strict()).unwrap();
        maximum = maximum.max(compare(
            &solution.generator,
            &operator(&case["generator"]),
            1e-10,
        ));
        maximum = maximum.max(compare(
            &solution.transformed,
            &operator(&case["transformed"]),
            1e-10,
        ));
        assert!(solution.residual_norm < 1e-13);
        assert!(solution.amplitude_residual_norm < 1e-13);
        let pt = equations
            .perturbation_through_third_order(&h, flow, strict())
            .unwrap();
        assert!(
            (equations.second_order_energy(&h, flow, strict()).unwrap() - pt.second_order).abs()
                < 1e-12
        );
        assert!(
            (pt.second_order - number(&case["pt2"])).abs() < 1e-10,
            "{} PT2 {} vs {}",
            case["name"],
            pt.second_order,
            case["pt2"]
        );
        assert!(
            (pt.third_order - number(&case["pt3"])).abs() < 1e-10,
            "{} PT3 {} vs {}",
            case["name"],
            pt.third_order,
            case["pt3"]
        );
        maximum = maximum.max(compare(
            &pt.first_order_generator,
            &operator(&case["a1"]),
            1e-10,
        ));
        maximum = maximum.max(compare(
            &pt.second_order_generator,
            &operator(&case["a2"]),
            1e-10,
        ));
        if case["name"] == "correlated_embedded_pair" {
            let active = indices(&case["active"]);
            let core = indices(&case["core"]);
            let mut correction = 0f64;
            for a in 0..6 {
                for i in 0..6 {
                    if core.contains(&a)
                        || !(core.contains(&i) || active.contains(&i))
                        || (active.contains(&a) && active.contains(&i))
                    {
                        continue;
                    }
                    let gap = h.one[(i, i)] - h.one[(a, a)];
                    let naive = h.one[(a, i)]
                        * hf_rs::post_hf::mr_dsrg::dsrg::dsrg_regularized_inv_denom(gap, flow);
                    correction =
                        correction.max((pt.first_order_generator.one[(a, i)] - naive).abs());
                }
            }
            assert!(
                correction > 1e-4,
                "missing active-density doubles-to-singles coupling"
            );
        }
        count += 4 * (1 + h.one.data.len() + h.two.len());
        println!(
            "{} iterations={} PT2={:e} PT3={:e} oracle_refinement={}",
            case["name"],
            solution.iterations,
            pt.second_order,
            pt.third_order,
            case["polynomial_refinement"]
        );
    }
    println!("MR driven coefficient comparisons={count}, max error={maximum:e}");
}
#[test]
fn mr_driven_zero_flow_full_active_and_slater_limits() {
    use hf_rs::post_hf::mr_dsrg::{
        mr_driven::MrDsrgEquations,
        sr_dsrg::{dsrg_pt3, solve_ldsrg2},
    };
    let data = mr_data();
    for case in data["cases"].as_array().unwrap() {
        let d = compute(case);
        let r = reference(case, &d);
        let eq = MrDsrgEquations::new(&r, BUDGET).unwrap();
        let h = operator(&case["h"]);
        let flow = number(&case["flow"]);
        let solution = eq.solve_ldsrg2(&h, 0., strict()).unwrap();
        compare(&solution.transformed, &h, 1e-14);
        let pt = eq
            .perturbation_through_third_order(&h, 0., strict())
            .unwrap();
        assert_eq!(pt.second_order, 0.);
        assert_eq!(pt.third_order, 0.);
        if case["name"] == "slater_reference" {
            let occ = vec![true, true, false, false, false, false];
            let sr = solve_ldsrg2(&h, &occ, flow, strict()).unwrap();
            let mr = eq.solve_ldsrg2(&h, flow, strict()).unwrap();
            compare(&mr.transformed, &sr.transformed, 1e-11);
            compare(&mr.generator, &sr.generator, 1e-11);
            let srpt = dsrg_pt3(&h, &occ, flow).unwrap();
            let mrpt = eq
                .perturbation_through_third_order(&h, flow, strict())
                .unwrap();
            assert!((srpt.second_order - mrpt.second_order).abs() < 1e-11);
            assert!((srpt.third_order - mrpt.third_order).abs() < 1e-11);
        }
        if case["name"] == "correlated_three_electron" {
            compare(
                &eq.solve_ldsrg2(&h, flow, strict()).unwrap().transformed,
                &h,
                1e-14,
            );
            let pt = eq
                .perturbation_through_third_order(&h, flow, strict())
                .unwrap();
            assert_eq!(pt.second_order, 0.);
            assert_eq!(pt.third_order, 0.);
        }
    }
}
#[test]
fn mr_driven_rejects_noncanonical_hamiltonians_and_invalid_controls() {
    use hf_rs::post_hf::mr_dsrg::mr_driven::MrDsrgEquations;
    let data = mr_data();
    let case = &data["cases"][3];
    let d = compute(case);
    let r = reference(case, &d);
    let eq = MrDsrgEquations::new(&r, BUDGET).unwrap();
    let h = operator(&case["h"]);
    for flow in [-1., f64::NAN, f64::INFINITY] {
        assert!(eq.solve_ldsrg2(&h, flow, strict()).is_err());
    }
    let mut opts = strict();
    opts.max_iter = 0;
    assert!(eq.solve_ldsrg2(&h, 0.5, opts).is_err());
    opts = strict();
    opts.bch_max_order = 0;
    assert!(eq.perturbation_through_third_order(&h, 0.5, opts).is_err());
    opts = strict();
    opts.max_iter = 1;
    assert!(eq.solve_ldsrg2(&h, 0.5, opts).is_err());
    let mut noncanonical = h.clone();
    noncanonical.one[(1, 2)] = 0.1;
    noncanonical.one[(2, 1)] = 0.1;
    assert!(eq.solve_ldsrg2(&noncanonical, 0.5, strict()).is_err());
    let mut nonhermitian = h.clone();
    nonhermitian.one[(0, 1)] += 0.1;
    assert!(eq.solve_ldsrg2(&nonhermitian, 0.5, strict()).is_err());
}

#[test]
fn quadratic_bch_mr_source_matches_every_independent_fock_coefficient() {
    use hf_rs::post_hf::mr_dsrg::{imsrg2::no2b_commutator, mr_driven::MrDsrgEquations};
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/mr_qc_reference.json")).unwrap();
    let full = mr_data();
    let mut maximum = 0f64;
    let mut count = 0;
    let mut difference = 0f64;
    let mut most_iterations = 0;
    for (index, case) in data["generic"].as_array().unwrap().iter().enumerate() {
        let d = compute(case);
        let r = reference(case, &d);
        let equations = MrDsrgEquations::new(&r, BUDGET).unwrap();
        let h = operator(&case["h"]);
        let solution = equations
            .solve_ldsrg2_qc(&h, number(&case["flow"]), strict())
            .unwrap();
        maximum = maximum.max(compare(
            &solution.generator,
            &operator(&case["generator"]),
            1e-10,
        ));
        maximum = maximum.max(compare(
            &solution.transformed,
            &operator(&case["transformed"]),
            1e-10,
        ));
        assert_eq!(solution.bch_order, 2);
        assert!(solution.residual_norm < 1e-13);
        assert!(solution.amplitude_residual_norm < 1e-13);
        most_iterations = most_iterations.max(solution.iterations);
        difference = difference.max(
            (solution.transformed.zero - number(&full["cases"][index]["transformed"]["zero"]))
                .abs(),
        );
        count += 2 * (1 + h.one.data.len() + h.two.len());
        let zero = equations.solve_ldsrg2_qc(&h, 0., strict()).unwrap();
        compare(&zero.transformed, &h, 1e-14);
        assert_eq!(zero.generator.zero, 0.);
        assert!(
            zero.generator
                .one
                .data
                .iter()
                .chain(&zero.generator.two)
                .all(|&x| x == 0.)
        );
        if case["name"] == "slater_reference" {
            let occ = vec![true, true, false, false, false, false];
            let c1 = no2b_commutator(&h, &solution.generator, &occ).unwrap();
            let c2 = no2b_commutator(&c1, &solution.generator, &occ).unwrap();
            let mut sr_bar = h.clone();
            sr_bar.zero += c1.zero + 0.5 * c2.zero;
            for ((x, a), b) in sr_bar
                .one
                .data
                .iter_mut()
                .zip(&c1.one.data)
                .zip(&c2.one.data)
            {
                *x += a + 0.5 * b;
            }
            for ((x, a), b) in sr_bar.two.iter_mut().zip(&c1.two).zip(&c2.two) {
                *x += a + 0.5 * b;
            }
            compare(&solution.transformed, &sr_bar, 1e-12);
        }
        if case["name"] == "correlated_embedded_pair" {
            let mut short = strict();
            short.max_iter = 1;
            assert!(
                equations
                    .solve_ldsrg2_qc(&h, number(&case["flow"]), short)
                    .is_err()
            );
            assert!(equations.solve_ldsrg2_qc(&h, -1., strict()).is_err());
        }
    }
    assert_eq!(count, 13330);
    assert!(most_iterations > 2);
    assert!(difference > 1e-8);
    println!(
        "QC all {count} operator coefficients: error={maximum:e}, full-BCH energy difference={difference:e}, max source iterations={most_iterations}"
    );
}
