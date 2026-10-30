// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::fci::engine::{FciHamiltonian, FciSolverOptions};
use hf_rs::matrix::Matrix;
use hf_rs::mo_integrals::MoEriTensor;
use hf_rs::post_hf::dmrg_engine::{MpsSolverOptions, solve_mps};

fn model(n: usize) -> (Matrix, MoEriTensor) {
    let mut h = Matrix::zeros(n, n);
    for p in 0..n {
        for q in 0..n {
            h[(p, q)] = if p == q {
                -1.2 + 0.31 * p as f64
            } else {
                0.17 * ((p + q + 1) as f64).sin()
            };
        }
    }
    let pairs = n * (n + 1) / 2;
    let mut eri = MoEriTensor {
        n_basis: n,
        n_pairs: pairs,
        data: vec![0.0; pairs * pairs],
    };
    for p in 0..pairs {
        for q in 0..pairs {
            eri.data[p * pairs + q] = (0..3)
                .map(|k| 0.08 * ((p + 2 * k + 1) as f64).cos() * ((q + 2 * k + 1) as f64).cos())
                .sum();
        }
    }
    (h, eri)
}

#[test]
fn large_m_matches_fci_and_every_rdm_component_in_closed_and_open_shell_sectors() {
    for (n, a, b) in [
        (0, 0, 0),
        (1, 1, 0),
        (2, 1, 1),
        (3, 2, 1),
        (4, 2, 2),
        (4, 3, 1),
    ] {
        let (h, g) = model(n);
        let engine = FciHamiltonian::new(&h, &g, a, b, 0.37).unwrap();
        let exact = engine.solve(&FciSolverOptions::default()).unwrap();
        let dmrg = solve_mps(
            &h,
            &g,
            a,
            b,
            0.37,
            &MpsSolverOptions {
                bond_dimensions: vec![32],
                truncation_tol: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(dmrg.converged, "n={n}, sector=({a},{b})");
        assert!(
            (dmrg.energy - exact.states[0].energy).abs() < 1e-10,
            "n={n}, dmrg={}, exact={}",
            dmrg.energy,
            exact.states[0].energy
        );
        assert_eq!(dmrg.state.electron_counts(), (a, b));
        assert!(dmrg.state.bond_dimensions().iter().all(|&d| d <= 32));
        let (one, two) = dmrg.state.rdms().unwrap();
        for (order, actual) in [(1, one.data), (2, two)] {
            let expected = engine.rdm(&exact.states[0].vector, order).unwrap();
            let delta = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(
                delta < 1e-8,
                "n={n}, RDM order {order}, max error={delta:e}"
            );
        }
        assert!((dmrg.state.spin_squared().unwrap() - exact.states[0].s_squared).abs() < 1e-8);
    }
}

#[test]
fn finite_m_is_variational_and_has_measured_schmidt_discarding() {
    let (h, g) = model(4);
    let exact = FciHamiltonian::new(&h, &g, 2, 2, 0.0)
        .unwrap()
        .solve(&FciSolverOptions::default())
        .unwrap();
    let mut last = f64::INFINITY;
    for m in [2, 4, 16] {
        let result = solve_mps(
            &h,
            &g,
            2,
            2,
            0.0,
            &MpsSolverOptions {
                bond_dimensions: vec![m],
                sweeps: 40,
                truncation_tol: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.energy >= exact.states[0].energy - 1e-11);
        assert!(result.energy <= last + 1e-9);
        assert!(result.state.bond_dimensions().iter().all(|&d| d <= m));
        if m == 2 {
            assert!(result.discarded_weights.iter().any(|&x| x > 1e-8));
        }
        last = result.energy;
    }
    assert!((last - exact.states[0].energy).abs() < 1e-10);
}

#[test]
fn invalid_inputs_and_unconverged_sweeps_are_reported() {
    let (h, g) = model(4);
    assert!(solve_mps(&h, &g, 5, 0, 0.0, &Default::default()).is_err());
    assert!(
        solve_mps(
            &h,
            &g,
            2,
            2,
            0.0,
            &MpsSolverOptions {
                bond_dimensions: vec![0],
                ..Default::default()
            }
        )
        .is_err()
    );
    let result = solve_mps(
        &h,
        &g,
        2,
        2,
        0.0,
        &MpsSolverOptions {
            sweeps: 1,
            one_site_sweeps: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!result.converged);
}

#[test]
fn long_chain_storage_is_polynomial_and_budget_errors_are_explicit() {
    // C(20,10)^2 = 34,134,779,536 determinants; none are enumerated.
    let n = 20;
    let mut h = Matrix::zeros(n, n);
    for p in 0..n - 1 {
        h[(p, p + 1)] = -1.0;
        h[(p + 1, p)] = -1.0;
    }
    let np = n * (n + 1) / 2;
    let mut g = MoEriTensor {
        n_basis: n,
        n_pairs: np,
        data: vec![0.0; np * np],
    };
    for p in 0..n {
        let pq = p * (p + 1) / 2 + p;
        g.data[pq * np + pq] = 2.0;
    }
    let result = solve_mps(
        &h,
        &g,
        10,
        10,
        0.0,
        &MpsSolverOptions {
            bond_dimensions: vec![4],
            sweeps: 2,
            one_site_sweeps: 0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(result.energy.is_finite());
    assert_eq!(result.state.electron_counts(), (10, 10));
    assert!(result.state.bond_dimensions().iter().all(|&m| m <= 4));
    assert!(result.state.storage_bytes() < 20 * 4 * 4 * 4 * 8);
    println!(
        "20-site MPS tensor storage: {} bytes",
        result.state.storage_bytes()
    );
    assert!(
        solve_mps(
            &h,
            &g,
            10,
            10,
            0.0,
            &MpsSolverOptions {
                bond_dimensions: vec![usize::MAX],
                ..Default::default()
            }
        )
        .unwrap_err()
        .to_string()
        .contains("256 MiB")
    );
    let (h, g) = model(36);
    assert!(
        solve_mps(&h, &g, 18, 18, 0.0, &Default::default())
            .unwrap_err()
            .to_string()
            .contains("256 MiB")
    );
}
