// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::matrix::Matrix;
use hf_rs::mo_integrals::{MoEriTensor, transform_1e_ao_to_mo, transform_eri_ao_to_mo};
use hf_rs::options::ResolvedActiveSpace;
use hf_rs::post_hf::caspt2_engine::{IcCaspt2Options, solve_ic_caspt2};
use hf_rs::*;
use serde_json::Value;

fn number(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn vector(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(number).collect()
}
fn tensor(v: &Value, n: usize) -> MoEriTensor {
    let np = n * (n + 1) / 2;
    let mut out = MoEriTensor {
        n_basis: n,
        n_pairs: np,
        data: vec![0.0; np * np],
    };
    for p in 0..n {
        for q in 0..=p {
            for r in 0..n {
                for s in 0..=r {
                    out.data[(r * (r + 1) / 2 + s) * np + p * (p + 1) / 2 + q] =
                        number(&v[p][q][r][s]);
                }
            }
        }
    }
    out
}

#[test]
fn all_classes_metrics_and_first_order_coefficients_match_independent_matrix_oracles() {
    let data: Value = serde_json::from_str(include_str!("data/caspt2_reference.json")).unwrap();
    let mut energy_error = 0.0_f64;
    let mut coefficient_error = 0.0_f64;
    let mut components = 0;
    for case in data["cases"].as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as usize;
        let nc = case["n_core"].as_u64().unwrap() as usize;
        let na = case["n_active"].as_u64().unwrap() as usize;
        let space = ResolvedActiveSpace {
            n_alpha: case["n_alpha"].as_u64().unwrap() as usize,
            n_beta: case["n_beta"].as_u64().unwrap() as usize,
            inactive: (0..nc).collect(),
            active: (nc..nc + na).collect(),
            external: (nc + na..n).collect(),
        };
        let h = Matrix::from_vec(
            n,
            n,
            case["h"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(vector)
                .collect(),
        );
        let g = tensor(&case["eri"], n);
        let one = Matrix::from_vec(na, na, vector(&case["active_one"]));
        let ci = vector(&case["active_ci"]);
        let addresses = case["determinants"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, d)| ((d[0].as_u64().unwrap(), d[1].as_u64().unwrap()), i))
            .collect::<std::collections::HashMap<_, _>>();
        for expected in case["results"].as_array().unwrap() {
            let result = solve_ic_caspt2(
                &h,
                &g,
                &space,
                &ci,
                &one,
                &IcCaspt2Options {
                    ipea_shift: number(&expected["ipea_shift"]),
                    real_shift: number(&expected["real_shift"]),
                    imaginary_shift: number(&expected["imaginary_shift"]),
                    residual_tol: 1e-11,
                    ..Default::default()
                },
            )
            .unwrap();
            let delta = (result.correlation_energy - number(&expected["energy"])).abs();
            energy_error = energy_error.max(delta);
            assert!(delta < 1e-10, "{}: energy error {delta:e}", case["name"]);
            assert!(
                (result.fock_reference_energy - number(&case["reference_fock_energy"])).abs()
                    < 1e-11
            );
            let raw = case["raw_dimensions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            assert_eq!(result.raw_dimensions.as_slice(), raw.as_slice());
            let ranks = case["ranks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            assert_eq!(result.retained_dimensions.as_slice(), ranks.as_slice());
            for class in 0..8 {
                assert!(
                    (result.class_energies[class] - number(&expected["components"][class])).abs()
                        < 1e-10
                );
                for (&x, y) in result.metric_eigenvalues[class]
                    .iter()
                    .zip(vector(&case["metric_eigenvalues"][class]))
                {
                    assert!((x - y).abs() < 1e-8);
                }
            }
            let (spectrum, _) = result.zeroth_order.eigh();
            for (&x, y) in spectrum.iter().zip(vector(&expected["zeroth_spectrum"])) {
                assert!(
                    (x - y).abs() < 1e-8 + 1e-10 * y.abs(),
                    "{} IPEA={} spectrum {x:e} vs {y:e}",
                    case["name"],
                    number(&expected["ipea_shift"])
                );
            }
            for (&(a, b), &x) in result
                .first_order_determinants
                .iter()
                .zip(&result.first_order)
            {
                let y = number(&expected["first_order"][addresses[&(a.0, b.0)]]);
                coefficient_error = coefficient_error.max((x - y).abs());
                components += 1;
                assert!(
                    (x - y).abs() < 1e-8,
                    "{}: coefficient error={:e}",
                    case["name"],
                    x - y
                );
            }
            assert!(result.residual_norm < 1e-11);
        }
    }
    println!(
        "CASPT2 exact matrices: energy={energy_error:e}, first_order={coefficient_error:e}, components={components}"
    );
}

#[test]
fn empty_active_space_is_mp2_and_complete_active_space_has_zero_correction() {
    let atoms = vec![
        Atom::new("Li", [0.1, -0.2, 0.3]),
        Atom::new("H", [0.4, 0.1, 3.2]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_scf(&atoms, &basis, 150, 1e-14).unwrap();
    let n = hf.mo_coefficients.cols;
    let h = transform_1e_ao_to_mo(&hf.mo_coefficients, &hf.core_hamiltonian);
    let g = transform_eri_ao_to_mo(&hf.mo_coefficients, &hf.eri);
    let space = ResolvedActiveSpace {
        n_alpha: 0,
        n_beta: 0,
        inactive: (0..hf.n_occ).collect(),
        active: vec![],
        external: (hf.n_occ..n).collect(),
    };
    let result = solve_ic_caspt2(
        &h,
        &g,
        &space,
        &[1.0],
        &Matrix::zeros(0, 0),
        &Default::default(),
    )
    .unwrap();
    let mp2 = RHFMP2::new(&hf).compute().unwrap();
    assert!((result.correlation_energy - mp2.correlation_energy).abs() < 1e-10);
    for (frozen_core, frozen_virtual) in [(1, 0), (0, 1), (1, 1), (2, 0)] {
        let result = solve_ic_caspt2(
            &h,
            &g,
            &space,
            &[1.0],
            &Matrix::zeros(0, 0),
            &IcCaspt2Options {
                frozen_core,
                frozen_virtual,
                ..Default::default()
            },
        )
        .unwrap();
        let mut expected = 0.0;
        for i in frozen_core..hf.n_occ {
            for j in frozen_core..hf.n_occ {
                for a in hf.n_occ..n - frozen_virtual {
                    for b in hf.n_occ..n - frozen_virtual {
                        let v = g.get(i, a, j, b);
                        expected += v * (2.0 * v - g.get(i, b, j, a))
                            / (hf.orbital_energies[i] + hf.orbital_energies[j]
                                - hf.orbital_energies[a]
                                - hf.orbital_energies[b]);
                    }
                }
            }
        }
        assert!((result.correlation_energy - expected).abs() < 1e-10);
    }
    let engine = fci::engine::FciHamiltonian::new(&h, &g, hf.n_occ, hf.n_occ, 0.0).unwrap();
    let solution = engine.solve(&Default::default()).unwrap();
    let one = Matrix::from_vec(n, n, engine.rdm(&solution.states[0].vector, 1).unwrap());
    let space = ResolvedActiveSpace {
        n_alpha: hf.n_occ,
        n_beta: hf.n_occ,
        inactive: vec![],
        active: (0..n).collect(),
        external: vec![],
    };
    let result = solve_ic_caspt2(
        &h,
        &g,
        &space,
        &solution.states[0].vector,
        &one,
        &Default::default(),
    )
    .unwrap();
    assert_eq!(result.correlation_energy, 0.0);
    assert_eq!(result.retained_dimensions, [0; 8]);
}

#[test]
fn imaginary_shift_has_a_continuous_exact_zero_denominator_limit() {
    let space = ResolvedActiveSpace {
        n_alpha: 0,
        n_beta: 0,
        inactive: vec![0],
        active: vec![],
        external: vec![1],
    };
    let n = 2;
    let mut g = MoEriTensor {
        n_basis: n,
        n_pairs: 3,
        data: vec![0.; 9],
    };
    let pair = |p: usize, q: usize| p.max(q) * (p.max(q) + 1) / 2 + p.min(q);
    for (p, q, r, s, value) in [
        (0, 0, 0, 0, 0.5),
        (1, 1, 1, 1, 0.5),
        (0, 0, 1, 1, 0.25),
        (0, 1, 0, 1, 0.125),
    ] {
        g.data[pair(p, q) * 3 + pair(r, s)] = value;
        g.data[pair(r, s) * 3 + pair(p, q)] = value;
    }
    let mut previous = 0.0;
    for gap in [-1e-7, 0., 1e-7] {
        let h = Matrix::from_vec(2, 2, vec![-0.5, 0., 0., -0.375 + 0.5 * gap]);
        let result = solve_ic_caspt2(
            &h,
            &g,
            &space,
            &[1.],
            &Matrix::zeros(0, 0),
            &IcCaspt2Options {
                imaginary_shift: 0.2,
                residual_tol: 1e-12,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.first_order.iter().all(|x| x.is_finite()));
        assert!(result.correlation_energy.abs() < 1e-7);
        if gap == 0. {
            assert_eq!(result.correlation_energy, 0.);
            assert!(
                solve_ic_caspt2(
                    &h,
                    &g,
                    &space,
                    &[1.],
                    &Matrix::zeros(0, 0),
                    &Default::default()
                )
                .is_err()
            );
        } else {
            let expected = -2. * 0.125_f64.powi(2) * gap / (gap * gap + 0.04)
                + 0.125_f64.powi(2) * gap.powi(3) / (gap * gap + 0.04).powi(2);
            assert!((result.correlation_energy - expected).abs() < 1e-14);
            if gap < 0. {
                previous = result.correlation_energy;
            } else {
                assert!((result.correlation_energy + previous).abs() < 1e-14);
            }
        }
    }
}

#[test]
fn invalid_controls_and_unbounded_active_spaces_fail_before_large_allocations() {
    let n = 24;
    let np = n * (n + 1) / 2;
    let h = Matrix::zeros(n, n);
    let g = MoEriTensor {
        n_basis: n,
        n_pairs: np,
        data: vec![0.; np * np],
    };
    let space = ResolvedActiveSpace {
        n_alpha: 12,
        n_beta: 12,
        inactive: vec![],
        active: (0..n).collect(),
        external: vec![],
    };
    let error = solve_ic_caspt2(
        &h,
        &g,
        &space,
        &[1.],
        &Matrix::zeros(n, n),
        &Default::default(),
    )
    .unwrap_err();
    assert!(matches!(error, error::HfError::Unsupported { .. }));
    let invalid = IcCaspt2Options {
        real_shift: 0.1,
        imaginary_shift: 0.1,
        ..Default::default()
    };
    assert!(matches!(
        solve_ic_caspt2(&h, &g, &space, &[1.], &Matrix::zeros(n, n), &invalid),
        Err(error::HfError::InvalidInput(_))
    ));
}
