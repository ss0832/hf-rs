// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    Matrix,
    df::DfTensor,
    fci::{engine::FciSolverOptions, orbital_rotation::rotate_ci, string::DeterminantSpace},
    options::ResolvedActiveSpace,
    post_hf::mr_dsrg::{
        MultiStateDSRGVariant as Variant,
        imsrg2::No2bOperator,
        multistate::{multistate_effective, source_operator},
    },
};
const BUDGET: usize = 256 * 1024 * 1024;
fn num(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap()
}
fn vector(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}
fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter().flat_map(vector).collect(),
    )
}
fn variant(name: &str) -> Variant {
    match name {
        "sa" => Variant::StateAverage,
        "sac" => Variant::StateAverageContracted,
        "ms" => Variant::MS,
        "xms" => Variant::XMS,
        "dw0" => Variant::DWMS { zeta: 0. },
        "dw50" => Variant::DWMS { zeta: 50. },
        _ => Variant::DWMS { zeta: 1e14 },
    }
}
fn compare(a: &[f64], b: &[f64], tol: f64) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut max = 0f64;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let err = (x - y).abs();
        max = max.max(err);
        assert!(err < tol, "component {i}: {x:e} vs {y:e}: {err:e}");
    }
    max
}
fn anchor_operator(source: &No2bOperator, spatial: &Matrix) -> No2bOperator {
    let n = source.one.rows;
    let mut u = Matrix::zeros(n, n);
    for p in 0..n {
        for q in 0..n {
            if p % 2 == q % 2 {
                u[(p, q)] = spatial[(p / 2, q / 2)];
            }
        }
    }
    let mut two = vec![0.; n.pow(4)];
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    for a in 0..n {
                        for b in 0..n {
                            for c in 0..n {
                                for d in 0..n {
                                    two[((p * n + q) * n + r) * n + s] += u[(p, a)]
                                        * u[(q, b)]
                                        * u[(r, c)]
                                        * u[(s, d)]
                                        * source.two[((a * n + b) * n + c) * n + d];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    No2bOperator {
        zero: source.zero,
        one: u.dot(&source.one).dot(&u.transpose()),
        two,
    }
}

#[test]
fn every_multistate_source_operator_model_element_energy_and_ci_coordinate_matches_full_fock_oracle()
 {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/multistate_reference.json")).unwrap();
    assert_eq!(data["cases"].as_array().unwrap().len(), 7);
    let mut counts = [0; 4];
    let mut errors = [0f64; 4];
    for case in data["cases"].as_array().unwrap() {
        let h = matrix(&case["h"]);
        let n = h.rows;
        let b = matrix(&case["factors"]);
        let factors = DfTensor {
            n_ao: n,
            n_aux_original: b.rows,
            b,
        };
        let nc = case["n_core"].as_u64().unwrap() as usize;
        let na = case["n_active"].as_u64().unwrap() as usize;
        let vectors: Vec<_> = case["ci"].as_array().unwrap().iter().map(vector).collect();
        let roots = vectors.len();
        let space = ResolvedActiveSpace {
            n_alpha: 1,
            n_beta: 1,
            inactive: (0..nc).collect(),
            active: (nc..nc + na).collect(),
            external: (nc + na..n).collect(),
        };
        let dets = DeterminantSpace::new(na, 1, 1);
        let options = FciSolverOptions {
            n_roots: roots,
            residual_tol: 1e-12,
            target_s_squared: Some(0.),
            ..Default::default()
        };
        let weights = vec![1. / roots as f64; roots];
        let nuclear = num(&case["nuclear"]);
        for expected in case["results"].as_array().unwrap() {
            let name = expected["variant"].as_str().unwrap();
            let flow = num(&expected["flow"]);
            let actual = multistate_effective(
                &h,
                &factors,
                nuclear,
                &space,
                &vectors,
                &weights,
                variant(name),
                flow,
                &options,
                BUDGET,
            )
            .unwrap();
            errors[0] = errors[0].max(compare(
                &actual.energies,
                &vector(&expected["energies"]),
                1e-9,
            ));
            counts[0] += roots;
            let mut ua = Matrix::zeros(na, na);
            for p in 0..na {
                for q in 0..na {
                    ua[(p, q)] = actual.orbital_rotation[(nc + p, nc + q)];
                }
            }
            for (i, c) in actual.ci_coefficients.iter().enumerate() {
                let anchor = rotate_ci(&dets, c, &ua.transpose(), BUDGET).unwrap();
                let target = vector(&expected["ci"][i]);
                let sign = anchor
                    .iter()
                    .zip(&target)
                    .map(|(x, y)| x * y)
                    .sum::<f64>()
                    .signum();
                errors[1] = errors[1].max(compare(
                    &anchor,
                    &target.iter().map(|x| sign * x).collect::<Vec<_>>(),
                    1e-8,
                ));
                counts[1] += anchor.len();
            }
            let rotation = matrix(&expected["model_rotation"]);
            let signs: Vec<_> = (0..roots)
                .map(|j| {
                    (0..roots)
                        .map(|i| rotation[(i, j)] * actual.model_rotation[(i, j)])
                        .sum::<f64>()
                        .signum()
                })
                .collect();
            for i in 0..roots {
                for j in 0..roots {
                    let x = actual.model_hamiltonian[(i, j)] * signs[i] * signs[j];
                    let y = num(&expected["model_hamiltonian"][i][j]);
                    errors[2] = errors[2].max(compare(&[x], &[y], 1e-9));
                    counts[2] += 1;
                    compare(
                        &[actual.weights[(i, j)]],
                        &[num(&expected["weights"][i][j])],
                        1e-10,
                    );
                }
            }
            let current: Vec<Vec<f64>> = (0..roots)
                .map(|j| {
                    (0..dets.n_determinants())
                        .map(|k| {
                            (0..roots)
                                .map(|i| vectors[i][k] * actual.model_rotation[(i, j)])
                                .sum()
                        })
                        .collect()
                })
                .collect();
            for (j, op) in expected["sources"].as_array().unwrap().iter().enumerate() {
                let weight: Vec<_> = (0..roots).map(|i| actual.weights[(j, i)]).collect();
                let source = source_operator(
                    &h, &factors, nuclear, &space, &current, &weight, flow, BUDGET,
                )
                .unwrap();
                let anchor = anchor_operator(&source.operator, &source.active_rotation);
                errors[3] = errors[3].max(compare(&[anchor.zero], &[num(&op["zero"])], 1e-9));
                errors[3] =
                    errors[3].max(compare(&anchor.one.data, &matrix(&op["one"]).data, 1e-9));
                errors[3] = errors[3].max(compare(&anchor.two, &vector(&op["two"]), 1e-9));
                counts[3] += 1 + anchor.one.data.len() + anchor.two.len();
            }
        }
    }
    println!("Multistate all-component counts {counts:?}; max errors {errors:?}");
}

#[test]
fn ensemble_rotation_covariance_dw_limits_and_degenerate_reference_weights_are_preserved() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/multistate_reference.json")).unwrap();
    let case = &data["cases"][0];
    let h = matrix(&case["h"]);
    let n = h.rows;
    let b = matrix(&case["factors"]);
    let factors = DfTensor {
        n_ao: n,
        n_aux_original: b.rows,
        b,
    };
    let space = ResolvedActiveSpace {
        n_alpha: 1,
        n_beta: 1,
        inactive: vec![],
        active: vec![0, 1],
        external: vec![2],
    };
    let vectors: Vec<_> = case["ci"].as_array().unwrap().iter().map(vector).collect();
    let theta = 0.37f64;
    let (s, c) = theta.sin_cos();
    let mixed = vec![
        vectors[0]
            .iter()
            .zip(&vectors[1])
            .map(|(a, b)| c * a + s * b)
            .collect(),
        vectors[0]
            .iter()
            .zip(&vectors[1])
            .map(|(a, b)| -s * a + c * b)
            .collect(),
    ];
    let options = FciSolverOptions {
        n_roots: 2,
        target_s_squared: Some(0.),
        residual_tol: 1e-12,
        ..Default::default()
    };
    let evaluate = |vs: &[Vec<f64>], v| {
        multistate_effective(
            &h,
            &factors,
            num(&case["nuclear"]),
            &space,
            vs,
            &[0.5, 0.5],
            v,
            0.5,
            &options,
            BUDGET,
        )
        .unwrap()
    };
    for v in [
        Variant::StateAverage,
        Variant::StateAverageContracted,
        Variant::XMS,
        Variant::DWMS { zeta: 0. },
    ] {
        let original = evaluate(&vectors, v);
        let rotated = evaluate(&mixed, v);
        compare(&original.energies, &rotated.energies, 1e-10);
        if v != Variant::XMS {
            let r = Matrix::from_vec(2, 2, vec![c, -s, s, c]);
            compare(
                &r.transpose().dot(&original.model_hamiltonian).dot(&r).data,
                &rotated.model_hamiltonian.data,
                1e-10,
            );
        }
    }
    compare(
        &evaluate(&vectors, Variant::StateAverageContracted).energies,
        &evaluate(&vectors, Variant::DWMS { zeta: 0. }).energies,
        1e-12,
    );
    compare(
        &evaluate(&vectors, Variant::MS).energies,
        &evaluate(&vectors, Variant::DWMS { zeta: 1e14 }).energies,
        1e-12,
    );
    let zero = DfTensor {
        n_ao: n,
        n_aux_original: 1,
        b: Matrix::zeros(1, n * (n + 1) / 2),
    };
    let degenerate = multistate_effective(
        &Matrix::zeros(n, n),
        &zero,
        0.,
        &space,
        &mixed,
        &[0.5, 0.5],
        Variant::DWMS { zeta: 1e14 },
        0.5,
        &options,
        BUDGET,
    )
    .unwrap();
    compare(&degenerate.energies, &[0., 0.], 1e-12);
    compare(&degenerate.weights.data, &[0.5; 4], 1e-12);
    for budget in [0, 1024] {
        assert!(
            multistate_effective(
                &h,
                &factors,
                0.,
                &space,
                &vectors,
                &[0.5, 0.5],
                Variant::MS,
                0.5,
                &options,
                budget
            )
            .err()
            .unwrap()
            .to_string()
            .contains("memory budget")
        );
    }
    for weights in [[0., 0.], [0.5, -0.5], [f64::NAN, 1.]] {
        assert!(
            multistate_effective(
                &h,
                &factors,
                0.,
                &space,
                &vectors,
                &weights,
                Variant::MS,
                0.5,
                &options,
                BUDGET
            )
            .is_err()
        );
    }
    for zeta in [-1., f64::NAN, f64::INFINITY] {
        assert!(
            multistate_effective(
                &h,
                &factors,
                0.,
                &space,
                &vectors,
                &[0.5, 0.5],
                Variant::DWMS { zeta },
                0.5,
                &options,
                BUDGET
            )
            .is_err()
        );
    }
    let mut invalid = vectors.clone();
    invalid[0] = vec![0., 1., 0., 0.];
    invalid[1] = vec![1., 0., 0., 0.];
    assert!(
        multistate_effective(
            &h,
            &factors,
            0.,
            &space,
            &invalid,
            &[0.5, 0.5],
            Variant::MS,
            0.5,
            &options,
            BUDGET
        )
        .err()
        .unwrap()
        .to_string()
        .contains("singlet")
    );
    let mut invalid = vectors.clone();
    invalid[1] = invalid[0].clone();
    assert!(
        multistate_effective(
            &h,
            &factors,
            0.,
            &space,
            &invalid,
            &[0.5, 0.5],
            Variant::MS,
            0.5,
            &options,
            BUDGET
        )
        .is_err()
    );
}

#[test]
fn large_mo_multistate_sources_keep_only_active_ci_and_bounded_sector_workspaces() {
    use hf_rs::{context::active_integrals_from_mo, fci::engine::FciHamiltonian};
    let n = 80;
    let np = n * (n + 1) / 2;
    let factors = DfTensor {
        n_ao: n,
        n_aux_original: 2,
        b: Matrix::from_vec(
            2,
            np,
            (0..2 * np)
                .map(|k| 0.013 * ((k + 1) as f64 * 0.73).cos())
                .collect(),
        ),
    };
    let mut h = Matrix::zeros(n, n);
    for p in 0..n {
        h[(p, p)] = -1.3 + 0.021 * p as f64;
    }
    let space = ResolvedActiveSpace {
        n_alpha: 1,
        n_beta: 1,
        inactive: vec![0],
        active: vec![1, 2],
        external: (3..n).collect(),
    };
    let active =
        active_integrals_from_mo(&h, |p, q, r, s| factors.eri(p, q, r, s), space.clone(), 0.)
            .unwrap();
    let options = FciSolverOptions {
        n_roots: 2,
        target_s_squared: Some(0.),
        residual_tol: 1e-12,
        ..Default::default()
    };
    let reference = FciHamiltonian::new(&active.h, &active.eri, 1, 1, active.core_energy)
        .unwrap()
        .solve(&options)
        .unwrap();
    let vectors: Vec<_> = reference.states.into_iter().map(|s| s.vector).collect();
    let result = multistate_effective(
        &h,
        &factors,
        0.,
        &space,
        &vectors,
        &[0.5, 0.5],
        Variant::MS,
        0.5,
        &options,
        8 * 1024 * 1024,
    )
    .unwrap();
    assert_eq!(result.source_transformations, 2);
    assert!(result.peak_sector_terms <= 100);
    assert_eq!(result.ci_coefficients.len(), 2);
    assert!(result.ci_coefficients.iter().all(|c| c.len() == 4));
    assert!(result.energies.iter().all(|e| e.is_finite()));
    println!(
        "160-spin multistate: two four-component CI states, {} peak sector terms, energies {:?}",
        result.peak_sector_terms, result.energies
    );
}
