// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    DeterminantSpace, Matrix,
    fci::engine::FciHamiltonian,
    mo_integrals::MoEriTensor,
    post_hf::mr_dsrg::density_cumulants::{
        CiEnsembleMember, ExactDensityCumulants, exact_density_cumulants,
    },
};
const BUDGET: usize = 256 * 1024 * 1024;
fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/density_cumulants_reference.json")).unwrap()
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
fn assert_close(x: f64, y: f64, maximum: &mut f64) {
    let error = (x - y).abs();
    *maximum = maximum.max(error);
    assert!(error < 1e-10, "{x:e} vs {y:e}: error {error:e}");
}

#[test]
fn every_spin_orbital_density_and_cumulant_component_matches_exterior_log_oracle() {
    let data = data();
    let mut maximum = 0f64;
    let mut count = 0usize;
    for case in data["cases"].as_array().unwrap() {
        let c = compute(case);
        let n = c.gamma1.rows;
        assert_eq!(c.lambda3.coefficients().rows, 56);
        for p in 0..n {
            for q in 0..n {
                assert_close(
                    c.gamma1[(p, q)],
                    number(&case["gamma1"][p][q]),
                    &mut maximum,
                );
                count += 1;
            }
        }
        for (key, actual) in [("gamma2", &c.gamma2), ("lambda2", &c.lambda2)] {
            let expected = expand2(&case[key], n);
            for p in 0..n {
                for q in 0..n {
                    for r in 0..n {
                        for s in 0..n {
                            assert_close(
                                actual.element(&[p, q], &[r, s]).unwrap(),
                                expected[((p * n + q) * n + r) * n + s],
                                &mut maximum,
                            );
                            count += 1;
                        }
                    }
                }
            }
        }
        let gamma = expand3(&case["gamma3"], n);
        let lambda = expand3(&case["lambda3"], n);
        for address in 0..n.pow(6) {
            let mut remainder = address;
            let mut indices = [0usize; 6];
            for slot in indices.iter_mut().rev() {
                *slot = remainder % n;
                remainder /= n;
            }
            assert_close(
                c.gamma3(&indices[..3], &indices[3..]).unwrap(),
                gamma[address],
                &mut maximum,
            );
            assert_close(
                c.lambda3.element(&indices[..3], &indices[3..]).unwrap(),
                lambda[address],
                &mut maximum,
            );
            count += 2;
        }
        if case["name"].as_str().unwrap().contains("slater") {
            assert!(c.lambda2.full_norm() < 1e-12);
            assert!(c.lambda3.full_norm() < 1e-12);
        }
    }
    println!(
        "Exact density cumulants: all {count} spin-orbital components, maximum error {maximum:e}"
    );
}

#[test]
fn identical_lower_rdms_do_not_determine_lambda3_and_ensembles_are_formed_before_cumulants() {
    let data = data();
    let cases = data["cases"].as_array().unwrap();
    let plus = compute(&cases[0]);
    let minus = compute(&cases[1]);
    assert_eq!(plus.gamma1.data, minus.gamma1.data);
    assert_eq!(
        plus.gamma2.coefficients().data,
        minus.gamma2.coefficients().data
    );
    assert_eq!(
        plus.lambda2.coefficients().data,
        minus.lambda2.coefficients().data
    );
    let difference = plus
        .lambda3
        .coefficients()
        .data
        .iter()
        .zip(&minus.lambda3.coefficients().data)
        .map(|(x, y)| (x - y).abs())
        .fold(0f64, f64::max);
    assert!(difference > 0.9);
    let mixed = compute(
        cases
            .iter()
            .find(|c| c["name"] == "spin_multiplet_average")
            .unwrap(),
    );
    assert!((mixed.gamma1[(0, 0)] - 0.5).abs() < 1e-14);
    assert!((mixed.gamma1[(1, 1)] - 0.5).abs() < 1e-14);
    assert!((mixed.lambda2.element(&[0, 1], &[0, 1]).unwrap() + 0.25).abs() < 1e-14);
}

#[test]
fn spin_trace_matches_independent_fci_rdms_and_all_particle_contractions() {
    let data = data();
    let mut maximum = 0f64;
    for case in data["cases"].as_array().unwrap() {
        let c = compute(case);
        let ns = c.gamma1.rows;
        let n = ns / 2;
        let h = Matrix::zeros(n, n);
        let np = n * (n + 1) / 2;
        let eri = MoEriTensor {
            n_basis: n,
            n_pairs: np,
            data: vec![0.; np * np],
        };
        for rank in 1..=3 {
            let mut expected = vec![0.; n.pow(2 * rank)];
            let weight_sum = case["members"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| number(&m["weight"]))
                .sum::<f64>();
            for member in case["members"].as_array().unwrap() {
                let engine = FciHamiltonian::new(
                    &h,
                    &eri,
                    member["na"].as_u64().unwrap() as usize,
                    member["nb"].as_u64().unwrap() as usize,
                    0.,
                )
                .unwrap();
                let density = engine
                    .rdm(&vector(&member["coefficients"]), rank as usize)
                    .unwrap();
                for (x, y) in expected.iter_mut().zip(density) {
                    *x += number(&member["weight"]) / weight_sum * y;
                }
            }
            for (address, &value) in expected.iter().enumerate() {
                let mut remainder = address;
                let mut spatial = vec![0usize; 2 * rank as usize];
                for index in spatial.iter_mut().rev() {
                    *index = remainder % n;
                    remainder /= n;
                }
                let mut actual = 0.;
                for spins in 0..1usize << rank {
                    let a: Vec<_> = (0..rank as usize)
                        .map(|k| 2 * spatial[2 * k] + ((spins >> k) & 1))
                        .collect();
                    let b: Vec<_> = (0..rank as usize)
                        .map(|k| 2 * spatial[2 * k + 1] + ((spins >> k) & 1))
                        .collect();
                    actual += match rank {
                        1 => c.gamma1[(a[0], b[0])],
                        2 => c.gamma2.element(&a, &b).unwrap(),
                        3 => c.gamma3(&a, &b).unwrap(),
                        _ => unreachable!(),
                    };
                }
                assert_close(actual, value, &mut maximum);
            }
        }
        for p in 0..ns {
            for q in 0..ns {
                let partial = (0..ns)
                    .map(|t| c.gamma2.element(&[p, t], &[q, t]).unwrap())
                    .sum::<f64>();
                assert_close(
                    partial,
                    (c.n_electrons as f64 - 1.) * c.gamma1[(p, q)],
                    &mut maximum,
                );
                for r in 0..ns {
                    for s in 0..ns {
                        let partial = (0..ns)
                            .map(|t| c.gamma3(&[p, q, t], &[r, s, t]).unwrap())
                            .sum::<f64>();
                        assert_close(
                            partial,
                            (c.n_electrons as f64 - 2.)
                                * c.gamma2.element(&[p, q], &[r, s]).unwrap(),
                            &mut maximum,
                        );
                    }
                }
            }
        }
    }
    println!("FCI spin traces and particle contractions: maximum error {maximum:e}");
}

#[test]
fn scale_invariance_memory_budgets_vacuum_and_invalid_ensembles_are_checked() {
    let space = DeterminantSpace::new(4, 2, 1);
    let dim = space.n_determinants();
    let c: Vec<_> = (0..dim).map(|i| (i as f64 + 0.3).sin()).collect();
    let base = exact_density_cumulants(
        &[CiEnsembleMember {
            space: &space,
            coefficients: &c,
            weight: 1.,
        }],
        BUDGET,
    )
    .unwrap();
    for scale in [1e-200, 1e200] {
        let scaled: Vec<_> = c.iter().map(|v| v * scale).collect();
        let actual = exact_density_cumulants(
            &[CiEnsembleMember {
                space: &space,
                coefficients: &scaled,
                weight: 3.,
            }],
            BUDGET,
        )
        .unwrap();
        for (x, y) in actual
            .lambda3
            .coefficients()
            .data
            .iter()
            .zip(&base.lambda3.coefficients().data)
        {
            assert!((x - y).abs() < 1e-12);
        }
    }
    assert!(exact_density_cumulants(&[], BUDGET).is_err());
    for weight in [-1., 0., f64::NAN] {
        assert!(
            exact_density_cumulants(
                &[CiEnsembleMember {
                    space: &space,
                    coefficients: &c,
                    weight
                }],
                BUDGET
            )
            .is_err()
        );
    }
    assert!(
        exact_density_cumulants(
            &[CiEnsembleMember {
                space: &space,
                coefficients: &c,
                weight: 1.
            }],
            64
        )
        .is_err()
    );
    let zeros = vec![0.; dim];
    assert!(
        exact_density_cumulants(
            &[CiEnsembleMember {
                space: &space,
                coefficients: &zeros,
                weight: 1.
            }],
            BUDGET
        )
        .is_err()
    );
    let mut malformed = space.clone();
    malformed.alpha_strings[0] = malformed.alpha_strings[1];
    assert!(
        exact_density_cumulants(
            &[CiEnsembleMember {
                space: &malformed,
                coefficients: &c,
                weight: 1.
            }],
            BUDGET
        )
        .is_err()
    );
    let vacuum = DeterminantSpace::new(0, 0, 0);
    let result = exact_density_cumulants(
        &[CiEnsembleMember {
            space: &vacuum,
            coefficients: &[1.],
            weight: 1.,
        }],
        BUDGET,
    )
    .unwrap();
    assert_eq!(result.n_electrons, 0);
    assert!(result.gamma1.data.is_empty());
    assert!(base.lambda3.element(&[0, 1], &[0, 1]).is_err());
    assert!(base.lambda3.element(&[0, 1, 8], &[0, 1, 2]).is_err());
    assert!(
        exact_density_cumulants(
            &[CiEnsembleMember {
                space: &space,
                coefficients: &c,
                weight: 1.,
            }],
            50_000
        )
        .is_err()
    );
    let large = DeterminantSpace::new(64, 0, 0);
    assert!(
        exact_density_cumulants(
            &[CiEnsembleMember {
                space: &large,
                coefficients: &[1.],
                weight: 1.,
            }],
            BUDGET
        )
        .is_err()
    );
    let mut invalid = space.clone();
    invalid.n_alpha = usize::MAX;
    assert!(
        exact_density_cumulants(
            &[
                CiEnsembleMember {
                    space: &space,
                    coefficients: &c,
                    weight: 1.
                },
                CiEnsembleMember {
                    space: &invalid,
                    coefficients: &c,
                    weight: 1.
                },
            ],
            BUDGET
        )
        .is_err()
    );
}

#[test]
fn saved_casci_roots_and_state_average_feed_exact_active_cumulants() {
    use hf_rs::{
        context::Reference,
        options::{ActiveSpace, PostHFOptions},
        post_hf::casscf_engine::solve_cas,
        *,
    };
    let atoms = [
        Atom::new("Li", [0.1, -0.2, 0.3]),
        Atom::new("H", [0.4, 0.1, 3.2]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_scf(&atoms, &basis, 150, 1e-14).unwrap();
    let mut options = PostHFOptions {
        active: Some(ActiveSpace::new(2, 2, 4)),
        n_roots: 2,
        ..Default::default()
    };
    options.cas.state_weights = vec![0.2, 0.8];
    options.cas.target_s_squared = Some(0.);
    let cas = solve_cas(Reference::Rhf(&hf), &options, None, false).unwrap();
    let average = cas.exact_active_cumulants(BUDGET).unwrap();
    let roots: Vec<_> = (0..2)
        .map(|root| cas.exact_root_cumulants(root, BUDGET).unwrap())
        .collect();
    let mut maximum = 0f64;
    for (index, c) in std::iter::once(&average).chain(roots.iter()).enumerate() {
        let one = if index == 0 {
            &cas.averaged_one_rdm
        } else {
            &cas.state_one_rdms[index - 1]
        };
        let two = if index == 0 {
            &cas.averaged_two_rdm
        } else {
            &cas.state_two_rdms[index - 1]
        };
        for p in 0..4 {
            for q in 0..4 {
                assert_close(
                    c.gamma1[(2 * p, 2 * q)] + c.gamma1[(2 * p + 1, 2 * q + 1)],
                    one[(p, q)],
                    &mut maximum,
                );
                for r in 0..4 {
                    for s in 0..4 {
                        let mut value = 0.;
                        for sigma in 0..2 {
                            for tau in 0..2 {
                                value += c
                                    .gamma2
                                    .element(
                                        &[2 * p + sigma, 2 * r + tau],
                                        &[2 * q + sigma, 2 * s + tau],
                                    )
                                    .unwrap();
                            }
                        }
                        assert_close(value, two[((p * 4 + q) * 4 + r) * 4 + s], &mut maximum);
                    }
                }
            }
        }
    }
    assert!(average.lambda3.full_norm() > 1e-6);
    assert!(cas.exact_root_cumulants(2, BUDGET).is_err());
    assert!(cas.exact_active_cumulants(1).is_err());
    println!(
        "CASCI root/state-average cumulants: all active 1/2-RDM spin traces, maximum error {maximum:e}"
    );
}
