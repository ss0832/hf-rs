// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Linear-memory 1s radial HF against independent finite-grid and continuum solvers.

use hf_rs::{NumericalAtomicOptions, run_numerical_atomic_1s_hf, run_numerical_atomic_hf};

fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/numerical_atomic_reference.json")).unwrap()
}

#[test]
fn radial_energies_and_every_grid_value_match_independent_tridiagonal_oracles() {
    let (mut energy_error, mut orbital_error) = (0f64, 0f64);
    let mut points = 0;
    for case in data()["grid_cases"].as_array().unwrap() {
        let z = case["z"].as_u64().unwrap() as usize;
        let ne = case["electrons"].as_u64().unwrap() as usize;
        let n = case["n_grid"].as_u64().unwrap() as usize;
        let options = NumericalAtomicOptions {
            n_grid_points: n,
            energy_tol: 1e-12,
            ..Default::default()
        };
        let result = run_numerical_atomic_1s_hf(z, ne, &options).unwrap();
        assert!(
            result.converged,
            "Z={z}, N={ne}, grid={n}: residual {}, density {}",
            result.residual_norm, result.density_change
        );
        let de = (result.total_energy - case["energy"].as_f64().unwrap()).abs();
        energy_error = energy_error.max(de);
        assert!(de < 1e-9, "Z={z}, N={ne}, grid={n}: energy {de:e}");
        assert!((result.orbital_energy_1s - case["orbital_energy"].as_f64().unwrap()).abs() < 1e-9);
        for (i, p) in result.radial_wavefunction.iter().enumerate() {
            let error = (p - case["radial_wavefunction"][i].as_f64().unwrap()).abs();
            orbital_error = orbital_error.max(error);
            assert!(
                error < 1e-8,
                "Z={z}, N={ne}, grid={n}, radial point {i}: {error:e}"
            );
            points += 1;
        }
        let norm = result
            .radial_wavefunction
            .iter()
            .map(|p| p * p * result.grid_spacing_bohr)
            .sum::<f64>();
        assert!((norm - 1.).abs() < 1e-12);
        assert_eq!(result.charge, (z - ne) as i32);
        assert!(
            (result.total_energy
                - result.kinetic_energy
                - result.nuclear_attraction_energy
                - result.electron_repulsion_energy)
                .abs()
                < 1e-12
        );
        assert!(
            (result.total_energy - ne as f64 * result.orbital_energy_1s
                + result.electron_repulsion_energy)
                .abs()
                < 1e-12
        );
    }
    println!(
        "Radial 1s HF: 7 energies max error {energy_error:e}, all {points} radial values max error {orbital_error:e}"
    );
}

#[test]
fn one_electron_radial_grid_matches_the_exact_discrete_coulomb_solution() {
    for z in [1, 2, 10] {
        let result = run_numerical_atomic_1s_hf(
            z,
            1,
            &NumericalAtomicOptions {
                n_grid_points: 1024,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.converged);
        let zh = z as f64 * result.grid_spacing_bohr;
        let exact = -((z * z) as f64) / (1. + (1. + zh * zh).sqrt());
        assert!((result.total_energy - exact).abs() < 2e-10 * (z * z) as f64);
        assert_eq!(result.electron_repulsion_energy, 0.);
    }
}

#[test]
fn radial_hf_converges_to_independent_continuum_boundary_value_energies() {
    for case in data()["continuum_cases"].as_array().unwrap() {
        let z = case["z"].as_u64().unwrap() as usize;
        let reference = case["energy"].as_f64().unwrap();
        let mut previous = f64::INFINITY;
        for n in [1024, 2048, 32768] {
            let result = run_numerical_atomic_1s_hf(
                z,
                2,
                &NumericalAtomicOptions {
                    n_grid_points: n,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(
                result.converged,
                "Z={z}, grid={n}, residual {}",
                result.residual_norm
            );
            let error = (result.total_energy - reference).abs();
            assert!(
                error < previous / 3.,
                "Z={z}, grid={n}: refinement {error:e} vs {previous:e}"
            );
            previous = error;
            if n == 32768 {
                assert!(error < 2e-6, "Z={z}: continuum {error:e}");
                assert!(result.virial_error.abs() < 1e-5);
                println!(
                    "RADIAL_CONTINUUM Z={z}, N=2, grid={n}, E={:.12}, continuum={reference:.12}, error={error:e}, residual={:e}, virial={:e}",
                    result.total_energy, result.residual_norm, result.virial_error
                );
            }
        }
    }
    // Published helium HF limit printed to five decimal places (not the correlated energy).
    let he = run_numerical_atomic_1s_hf(
        2,
        2,
        &NumericalAtomicOptions {
            n_grid_points: 32768,
            ..Default::default()
        },
    )
    .unwrap();
    assert!((he.total_energy - (-2.86168)).abs() < 1e-6);
}

#[test]
fn radial_boundary_error_is_separate_from_scf_and_grid_convergence() {
    let options = NumericalAtomicOptions {
        n_grid_points: 1023,
        r_max_bohr: Some(10.),
        ..Default::default()
    };
    let small = run_numerical_atomic_1s_hf(2, 2, &options).unwrap();
    let large = run_numerical_atomic_1s_hf(
        2,
        2,
        &NumericalAtomicOptions {
            n_grid_points: 2047,
            r_max_bohr: Some(20.),
            ..options
        },
    )
    .unwrap();
    assert!(small.converged && large.converged);
    assert_eq!(small.grid_spacing_bohr, large.grid_spacing_bohr);
    assert!((small.total_energy - large.total_energy).abs() < 1e-7);
}

#[test]
fn radial_configuration_memory_and_convergence_controls_are_explicit() {
    assert!(
        run_numerical_atomic_hf(3, 128, 100, 1e-10)
            .unwrap_err()
            .contains("neutral")
    );
    assert!(run_numerical_atomic_1s_hf(2, 3, &Default::default()).is_err());
    assert!(run_numerical_atomic_1s_hf(1, 2, &Default::default()).is_err());
    assert!(
        run_numerical_atomic_1s_hf(
            2,
            2,
            &NumericalAtomicOptions {
                memory_limit_bytes: 1024,
                ..Default::default()
            }
        )
        .unwrap_err()
        .contains("memory budget")
    );
    assert!(
        run_numerical_atomic_1s_hf(
            2,
            2,
            &NumericalAtomicOptions {
                r_max_bohr: Some(f64::NAN),
                ..Default::default()
            }
        )
        .is_err()
    );
    let unfinished = run_numerical_atomic_1s_hf(
        2,
        2,
        &NumericalAtomicOptions {
            n_grid_points: 128,
            max_iterations: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!unfinished.converged);
    for (z, n) in [(1, 150), (2, 120)] {
        let result = run_numerical_atomic_hf(z, n, 50, 1e-8).unwrap();
        assert!(result.converged);
    }
}
