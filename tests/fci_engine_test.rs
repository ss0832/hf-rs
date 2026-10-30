use hf_rs::fci::engine::FciHamiltonian;
use hf_rs::{Matrix, mo_integrals::MoEriTensor};

#[test]
fn davidson_finds_lower_roots_outside_the_lowest_diagonal_guess_sector() {
    use hf_rs::solvers::davidson::{DavidsonOptions, try_davidson_solver};
    let mut h = Matrix::zeros(8, 8);
    for i in 0..8 {
        h[(i, i)] = i as f64;
    }
    // The first two diagonal guesses are exact eigenvectors, but the block
    // (6,7) has a lower eigenvalue. A residual-only unit-guess solver misses it.
    h[(6, 7)] = 10.0;
    h[(7, 6)] = 10.0;
    let expected = h.eigh().0;
    let diagonal = (0..8).map(|i| h[(i, i)]).collect::<Vec<_>>();
    let result = try_davidson_solver(
        8,
        &diagonal,
        |c| {
            Ok((0..8)
                .map(|i| (0..8).map(|j| h[(i, j)] * c[j]).sum())
                .collect())
        },
        DavidsonOptions {
            n_roots: 2,
            tol: 1e-11,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(result.converged);
    for (a, b) in result.eigenvalues.iter().zip(expected) {
        assert!((a - b).abs() < 1e-10, "{a} vs {b}");
    }
}

fn model(n: usize) -> (Matrix, MoEriTensor) {
    let mut h = Matrix::zeros(n, n);
    let np = n * (n + 1) / 2;
    let mut eri = MoEriTensor {
        n_basis: n,
        n_pairs: np,
        data: vec![0.0; np * np],
    };
    for p in 0..n {
        for q in 0..n {
            h[(p, q)] = if p == q {
                -1.4 + 0.4 * p as f64
            } else {
                0.05 * ((p + q + 1) as f64).sin()
            };
        }
    }
    for pq in 0..np {
        for rs in 0..np {
            eri.data[rs * np + pq] = (0..3)
                .map(|k| {
                    0.1 * (((k + 1) * (pq + 1)) as f64 * 0.37).sin()
                        * (((k + 1) * (rs + 1)) as f64 * 0.37).sin()
                })
                .sum();
        }
    }
    (h, eri)
}

#[test]
fn all_sigma_components_match_independent_slater_condon_matrix_in_all_spin_sectors() {
    let (h, g) = model(4);
    for (na, nb) in [(2, 2), (2, 1), (3, 0), (0, 0)] {
        let engine = FciHamiltonian::new(&h, &g, na, nb, 0.321).unwrap();
        let dim = engine.dimension();
        let c = (0..dim)
            .map(|i| (i as f64 + 0.37).sin())
            .collect::<Vec<_>>();
        let sigma = engine.apply(&c).unwrap();
        let matrix = hf_rs::fci::compute_hamiltonian_matrix(&engine.space, &h, &g);
        for i in 0..dim {
            let expected = (0..dim).map(|j| matrix[(i, j)] * c[j]).sum::<f64>() + 0.321 * c[i];
            assert!(
                (sigma[i] - expected).abs() < 2e-12,
                "N=({na},{nb}) row {i}: {} vs {expected}",
                sigma[i]
            );
        }
    }
}

#[test]
fn exact_spin_ladder_distinguishes_closed_shell_and_open_shell_determinants() {
    let (h, g) = model(2);
    let engine = FciHamiltonian::new(&h, &g, 1, 1, 0.0).unwrap();
    for (index, s2) in [(0, 0.0), (1, 1.0), (2, 1.0), (3, 0.0)] {
        let mut c = vec![0.0; 4];
        c[index] = 1.0;
        assert!((engine.spin_squared(&c).unwrap() - s2).abs() < 1e-13);
    }
    let parallel = FciHamiltonian::new(&h, &g, 2, 0, 0.0).unwrap();
    assert!((parallel.spin_squared(&[1.0]).unwrap() - 2.0).abs() < 1e-13);
}

#[test]
fn every_sigma_spin_and_reduced_density_component_matches_pyscf() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/fci_engine_reference.json")).unwrap();
    let (h, g) = model(4);
    let f = Matrix::from_vec(
        4,
        4,
        oracle["f"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
            .collect(),
    );
    for case in oracle["cases"].as_array().unwrap() {
        let na = case["n_alpha"].as_u64().unwrap() as usize;
        let nb = case["n_beta"].as_u64().unwrap() as usize;
        let engine = FciHamiltonian::new(&h, &g, na, nb, 0.321).unwrap();
        let c = case["vector"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        let mut checks = vec![
            ("sigma", engine.apply(&c).unwrap()),
            ("s_squared_action", engine.spin_squared_action(&c).unwrap()),
        ];
        for (name, order) in [("one_rdm", 1), ("two_rdm", 2), ("three_rdm", 3)] {
            checks.push((name, engine.rdm(&c, order).unwrap()));
        }
        checks.push((
            "contracted_four_rdm",
            engine.contracted_four_rdm(&c, &f).unwrap(),
        ));
        for (name, actual) in checks {
            let expected = case[name].as_array().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (i, (&a, b)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (a - b.as_f64().unwrap()).abs() < 3e-12,
                    "N=({na},{nb}) {name}[{i}]: {a} vs {b}"
                );
            }
        }
        let one = engine.rdm(&c, 1).unwrap();
        let two = engine.rdm(&c, 2).unwrap();
        let three = engine.rdm(&c, 3).unwrap();
        assert!(((0..4).map(|p| one[p * 4 + p]).sum::<f64>() - (na + nb) as f64).abs() < 1e-12);
        for p in 0..4 {
            for q in 0..4 {
                for r in 0..4 {
                    for s in 0..4 {
                        let trace = (0..4)
                            .map(|t| three[(((((p * 4 + q) * 4 + r) * 4 + s) * 4 + t) * 4) + t])
                            .sum::<f64>();
                        assert!(
                            (trace - ((na + nb) as f64 - 2.0) * two[((p * 4 + q) * 4 + r) * 4 + s])
                                .abs()
                                < 2e-12
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn dense_and_davidson_roots_match_pyscf_with_physical_residual_checks() {
    use hf_rs::fci::engine::FciSolverOptions;
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/fci_engine_reference.json")).unwrap();
    let (h, g) = model(4);
    for case in oracle["cases"].as_array().unwrap() {
        let na = case["n_alpha"].as_u64().unwrap() as usize;
        let nb = case["n_beta"].as_u64().unwrap() as usize;
        let engine = FciHamiltonian::new(&h, &g, na, nb, 0.321).unwrap();
        for dense_threshold in [128, 0] {
            let options = FciSolverOptions {
                n_roots: 3,
                dense_threshold,
                residual_tol: 1e-10,
                ..Default::default()
            };
            let solution = engine.solve(&options).unwrap();
            for (i, state) in solution.states.iter().enumerate() {
                let expected = case["energies"][i].as_f64().unwrap();
                assert!(
                    (state.energy - expected).abs() < 1e-10,
                    "N=({na},{nb}) dense={dense_threshold} root {i}"
                );
                assert!(state.residual_norm < 1e-10);
            }
        }
    }
}

#[test]
fn spin_targeted_singlet_triplet_and_quintet_roots_match_pyscf() {
    use hf_rs::fci::engine::FciSolverOptions;
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/fci_engine_reference.json")).unwrap();
    let (h, g) = model(4);
    let engine = FciHamiltonian::new(&h, &g, 2, 2, 0.321).unwrap();
    for target in [0.0_f64, 2.0, 6.0] {
        let key = format!("{target:.1}");
        let expected = oracle["cases"][0]["spin_targets"][key].as_f64().unwrap();
        for dense_threshold in [128, 0] {
            let options = FciSolverOptions {
                target_s_squared: Some(target),
                dense_threshold,
                ..Default::default()
            };
            let result = engine.solve(&options).unwrap();
            assert!(
                (result.states[0].energy - expected).abs() < 1e-10,
                "target={target} dense={dense_threshold}"
            );
            assert!((result.states[0].s_squared - target).abs() < 1e-10);
        }
    }
}
