use hf_rs::fci::engine::FciSolverOptions;
use hf_rs::fci::excitations::singles_doubles;
use hf_rs::fci::subspace::SubspaceHamiltonian;
use hf_rs::fci::{DeterminantSpace, SpinString, slater_condon_element};
use hf_rs::{Matrix, mo_integrals::MoEriTensor};

fn model(n: usize) -> (Matrix, MoEriTensor) {
    let mut h = Matrix::zeros(n, n);
    let np = n * (n + 1) / 2;
    for p in 0..n {
        for q in 0..n {
            h[(p, q)] = if p == q {
                -1.4 + 0.4 * p as f64
            } else {
                0.05 * ((p + q + 1) as f64).sin()
            };
        }
    }
    let mut g = MoEriTensor {
        n_basis: n,
        n_pairs: np,
        data: vec![0.0; np * np],
    };
    for pq in 0..np {
        for rs in 0..np {
            g.data[rs * np + pq] = (0..3)
                .map(|k| {
                    0.1 * (((k + 1) * (pq + 1)) as f64 * 0.37).sin()
                        * (((k + 1) * (rs + 1)) as f64 * 0.37).sin()
                })
                .sum();
        }
    }
    (h, g)
}
#[test]
fn every_subspace_sigma_spin_and_density_component_matches_pyscf() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/fci_engine_reference.json")).unwrap();
    let (h, g) = model(4);
    for case in data["cases"].as_array().unwrap() {
        let na = case["n_alpha"].as_u64().unwrap() as usize;
        let nb = case["n_beta"].as_u64().unwrap() as usize;
        let full = DeterminantSpace::new(4, na, nb);
        let dets = full
            .alpha_strings
            .iter()
            .flat_map(|&a| full.beta_strings.iter().map(move |&b| (a, b)))
            .collect();
        let engine = SubspaceHamiltonian::new(&h, &g, 0.321, dets).unwrap();
        let c = case["vector"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>();
        for (name, values) in [
            ("sigma", engine.apply(&c).unwrap()),
            ("s_squared_action", engine.spin_squared_action(&c).unwrap()),
            ("one_rdm", engine.rdm(&c, 1).unwrap()),
            ("two_rdm", engine.rdm(&c, 2).unwrap()),
        ] {
            let expected = case[name].as_array().unwrap();
            assert_eq!(values.len(), expected.len());
            for (i, (&a, b)) in values.iter().zip(expected).enumerate() {
                assert!(
                    (a - b.as_f64().unwrap()).abs() < 3e-12,
                    "({na},{nb}) {name}[{i}]"
                );
            }
        }
        let result = engine
            .solve(&FciSolverOptions {
                n_roots: 3,
                ..Default::default()
            })
            .unwrap();
        for (i, state) in result.states.iter().enumerate() {
            assert!((state.energy - case["energies"][i].as_f64().unwrap()).abs() < 1e-10);
        }
        if na == 2 && nb == 2 {
            for target in [0.0, 2.0, 6.0] {
                let expected = case["spin_targets"][format!("{target:.1}")]
                    .as_f64()
                    .unwrap();
                let state = engine
                    .solve(&FciSolverOptions {
                        target_s_squared: Some(target),
                        ..Default::default()
                    })
                    .unwrap();
                assert!((state.states[0].energy - expected).abs() < 1e-10);
            }
        }
    }
}
#[test]
fn generated_sd_space_sigma_matches_independent_slater_condon_rows() {
    let (h, g) = model(6);
    let dets = singles_doubles(6, 3, 2, &[(SpinString(7), SpinString(3))], 100_000).unwrap();
    let engine = SubspaceHamiltonian::new(&h, &g, 0.17, dets).unwrap();
    let c = (0..engine.dimension())
        .map(|i| (i as f64 + 0.37).cos())
        .collect::<Vec<_>>();
    let actual = engine.apply(&c).unwrap();
    for (i, &(a, b)) in engine.determinants.iter().enumerate() {
        let expected = 0.17 * c[i]
            + engine
                .determinants
                .iter()
                .zip(&c)
                .map(|(&(aa, bb), &v)| v * slater_condon_element(&a, &b, &aa, &bb, &h, &g))
                .sum::<f64>();
        assert!(
            (actual[i] - expected).abs() < 2e-11,
            "row {i}: {} vs {expected}",
            actual[i]
        );
    }
}
