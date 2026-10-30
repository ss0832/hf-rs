// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    DeterminantSpace, Matrix,
    fci::{
        engine::FciSolverOptions,
        general_operator::{SpinOrbitalHamiltonian, contract_core},
        orbital_rotation::rotate_ci,
    },
    post_hf::mr_dsrg::imsrg2::No2bOperator,
};
const BUDGET: usize = 256 * 1024 * 1024;
fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/mr_relaxation_reference.json")).unwrap()
}
fn num(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap()
}
fn vector(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}
fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    let n = rows.len();
    Matrix::from_vec(
        n,
        if n == 0 {
            0
        } else {
            rows[0].as_array().unwrap().len()
        },
        rows.iter().flat_map(vector).collect(),
    )
}
fn op(v: &serde_json::Value) -> No2bOperator {
    No2bOperator {
        zero: num(&v["zero"]),
        one: matrix(&v["one"]),
        two: vector(&v["two"]),
    }
}
fn space(v: &serde_json::Value) -> DeterminantSpace {
    DeterminantSpace::new(
        v["n_spatial"].as_u64().unwrap() as usize,
        v["na"].as_u64().unwrap() as usize,
        v["nb"].as_u64().unwrap() as usize,
    )
}
fn compare(a: &[f64], b: &[f64], tol: f64) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut maximum = 0f64;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let e = (x - y).abs();
        maximum = maximum.max(e);
        assert!(e < tol, "component {i}: {x:e} vs {y:e}, difference {e:e}");
    }
    maximum
}
fn compare_op(a: &No2bOperator, b: &No2bOperator) -> f64 {
    let mut maximum = (a.zero - b.zero).abs();
    assert!(maximum < 1e-10);
    maximum = maximum.max(compare(&a.one.data, &b.one.data, 1e-10));
    maximum.max(compare(&a.two, &b.two, 1e-10))
}
#[test]
fn every_general_operator_ci_matrix_component_and_dense_davidson_root_matches_full_fock_space() {
    let d = data();
    let mut count = 0;
    let mut maximum = 0f64;
    for case in d["operators"].as_array().unwrap() {
        let operator = op(&case["operator"]);
        let space = space(case);
        let h = SpinOrbitalHamiltonian::new(&operator, &space, BUDGET).unwrap();
        let expected = matrix(&case["matrix"]);
        let dim = expected.rows;
        for column in 0..dim {
            let mut c = vec![0.; dim];
            c[column] = 1.;
            let actual = h.apply(&c).unwrap();
            let expected: Vec<_> = (0..dim).map(|row| expected[(row, column)]).collect();
            maximum = maximum.max(compare(&actual, &expected, 1e-11));
            count += dim;
        }
        let diagonal: Vec<_> = (0..dim).map(|i| expected[(i, i)]).collect();
        compare(&h.diagonal(), &diagonal, 1e-11);
        for dense in [128, 0] {
            let solution = h
                .solve(&FciSolverOptions {
                    n_roots: dim.min(3),
                    dense_threshold: dense,
                    residual_tol: 1e-11,
                    max_iterations: 200,
                    ..Default::default()
                })
                .unwrap();
            for (i, state) in solution.states.iter().enumerate() {
                assert!((state.energy - num(&case["energies"][i])).abs() < 1e-10);
                assert!(state.residual_norm < 1e-11);
            }
        }
    }
    println!("General spin CI matrix: all {count} components, maximum error={maximum:e}");
}
#[test]
fn spin_targeting_preserves_non_coulomb_effective_two_body_components() {
    let d = data();
    let case = &d["spin_scalar"];
    let operator = op(&case["operator"]);
    let space = space(case);
    let h = SpinOrbitalHamiltonian::new(&operator, &space, BUDGET).unwrap();
    let n = operator.one.rows;
    let mut non_coulomb = 0f64;
    let at = |p, q, r, s| operator.two[((p * n + q) * n + r) * n + s];
    for p in 0..3 {
        for q in 0..3 {
            for r in 0..3 {
                for s in 0..3 {
                    non_coulomb = non_coulomb.max(
                        (at(2 * p, 2 * q + 1, 2 * r, 2 * s + 1)
                            - at(2 * r, 2 * q + 1, 2 * p, 2 * s + 1))
                        .abs(),
                    );
                }
            }
        }
    }
    assert!(non_coulomb > 1e-3);
    for root in case["spin_roots"].as_array().unwrap() {
        for dense in [128, 0] {
            let solution = h
                .solve(&FciSolverOptions {
                    n_roots: 2,
                    target_s_squared: Some(num(&root["s_squared"])),
                    dense_threshold: dense,
                    residual_tol: 1e-11,
                    ..Default::default()
                })
                .unwrap();
            for (i, state) in solution.states.iter().enumerate() {
                assert!((state.energy - num(&root["energies"][i])).abs() < 1e-10);
                assert!((state.s_squared - num(&root["s_squared"])).abs() < 1e-10);
                let projector: Vec<_> = state
                    .vector
                    .iter()
                    .flat_map(|x| state.vector.iter().map(move |y| x * y))
                    .collect();
                compare(&projector, &matrix(&root["projectors"][i]).data, 1e-10);
            }
        }
    }
}
#[test]
fn every_core_contracted_coefficient_matches_full_fock_projection() {
    let d = data();
    let case = &d["core_projection"];
    let bare = op(&case["operator"]);
    let indices = |key: &str| {
        case[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect::<Vec<_>>()
    };
    let actual = contract_core(&bare, &indices("core"), &indices("active")).unwrap();
    let error = compare_op(&actual, &op(&case["projected"]));
    println!("Core contraction: all 273 coefficients, maximum error={error:e}");
}
#[test]
fn every_rotated_ci_coefficient_matches_explicit_fock_overlap_for_proper_and_improper_rotations() {
    let d = data();
    let mut maximum = 0f64;
    let mut count = 0;
    for case in d["rotations"].as_array().unwrap() {
        let space = space(case);
        let c = vector(&case["coefficients"]);
        let u = matrix(&case["u"]);
        let result = rotate_ci(&space, &c, &u, BUDGET).unwrap();
        maximum = maximum.max(compare(&result, &vector(&case["rotated"]), 1e-11));
        count += c.len();
        compare(
            &rotate_ci(&space, &result, &u.transpose(), BUDGET).unwrap(),
            &c,
            1e-11,
        );
        for scale in [1e-200, 1e200] {
            let scaled: Vec<_> = c.iter().map(|x| x * scale).collect();
            let normalized: Vec<_> = rotate_ci(&space, &scaled, &u, BUDGET)
                .unwrap()
                .iter()
                .map(|x| x / scale)
                .collect();
            compare(&normalized, &result, 1e-11);
        }
    }
    println!("CI rotations: all {count} coefficients, maximum error={maximum:e}");
}
#[test]
fn invalid_operators_spin_changes_orbitals_spaces_and_workspace_controls_are_errors() {
    let d = data();
    let case = &d["operators"][0];
    let operator = op(&case["operator"]);
    let space = space(case);
    assert!(SpinOrbitalHamiltonian::new(&operator, &space, 1).is_err());
    let h = SpinOrbitalHamiltonian::new(&operator, &space, BUDGET).unwrap();
    assert!(h.apply(&[]).is_err());
    for roots in [0, usize::MAX] {
        assert!(
            h.solve(&FciSolverOptions {
                n_roots: roots,
                ..Default::default()
            })
            .is_err()
        );
    }
    assert!(
        h.solve(&FciSolverOptions {
            memory_budget: 1,
            ..Default::default()
        })
        .is_err()
    );
    let mut bad = operator.clone();
    bad.one[(0, 1)] = 0.1;
    bad.one[(1, 0)] = 0.1;
    assert!(SpinOrbitalHamiltonian::new(&bad, &space, BUDGET).is_err());
    bad = operator.clone();
    bad.two[0] = 0.1;
    assert!(SpinOrbitalHamiltonian::new(&bad, &space, BUDGET).is_err());
    bad = operator.clone();
    bad.zero = f64::NAN;
    assert!(SpinOrbitalHamiltonian::new(&bad, &space, BUDGET).is_err());
    let mut malformed = space.clone();
    malformed.alpha_strings.swap(0, 1);
    assert!(SpinOrbitalHamiltonian::new(&operator, &malformed, BUDGET).is_err());
    assert!(contract_core(&operator, &[0], &[0, 1]).is_err());
    assert!(contract_core(&operator, &[], &[6]).is_err());
    let u = Matrix::identity(3);
    let c = vec![1.; space.n_determinants()];
    assert!(rotate_ci(&space, &c, &u, 1).is_err());
    let mut nonorthogonal = u.clone();
    nonorthogonal[(0, 0)] = 2.;
    assert!(rotate_ci(&space, &c, &nonorthogonal, BUDGET).is_err());
    assert!(rotate_ci(&space, &[], &u, BUDGET).is_err());
    assert!(rotate_ci(&malformed, &c, &u, BUDGET).is_err());
}
