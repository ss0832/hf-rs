use hf_rs::Matrix;
use hf_rs::local::pair_space::{LocalPairProblem, PairBlock, natural_pair_space};
fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows.first().map_or(0, |r| r.as_array().unwrap().len()),
        rows.iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect(),
    )
}
fn close(a: &Matrix, b: &Matrix, tolerance: f64) {
    assert_eq!((a.rows, a.cols), (b.rows, b.cols));
    for (index, (&a, &b)) in a.data.iter().zip(&b.data).enumerate() {
        assert!((a - b).abs() < tolerance, "component {index}: {a} vs {b}");
    }
}
fn problem(case: &serde_json::Value) -> LocalPairProblem {
    LocalPairProblem {
        f_occ: matrix(&case["f_occ"]),
        pairs: case["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| PairBlock {
                i: p["i"].as_u64().unwrap() as usize,
                j: p["j"].as_u64().unwrap() as usize,
                basis: matrix(&p["basis"]),
                energies: p["energies"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
                exchange: matrix(&p["exchange"]),
                initial: matrix(&p["initial"]),
                strong: p["strong"].as_bool().unwrap(),
                delta_pno: 0.,
            })
            .collect(),
    }
}
#[test]
fn coupled_pair_amplitudes_and_hylleraas_energy_match_dense_galerkin_reference() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/local_pair_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let p = problem(case);
        let solution = p.solve(100, 1e-12).unwrap();
        for (actual, expected) in solution
            .amplitudes
            .iter()
            .zip(case["amplitudes"].as_array().unwrap())
        {
            close(actual, &matrix(expected), 1e-12);
        }
        assert!(
            (solution.correlation_energy() - case["energy"].as_f64().unwrap()).abs() < 1e-12,
            "{}",
            case["name"]
        );
        assert!((solution.e_strong - case["e_strong"].as_f64().unwrap()).abs() < 1e-12);
        assert!((solution.e_weak - case["e_weak"].as_f64().unwrap()).abs() < 1e-12);
        assert!(solution.residual_max <= 1e-12);
        let applied = p.apply(&solution.amplitudes).unwrap();
        for (index, pair) in p.pairs.iter().enumerate().filter(|(_, p)| p.strong) {
            for (&a, &k) in applied[index].data.iter().zip(&pair.exchange.data) {
                assert!((a + k).abs() < 1e-12);
            }
        }
    }
}
#[test]
fn every_pair_density_projector_and_truncation_correction_matches_numpy() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/local_pair_reference.json")).unwrap();
    for case in data["pnos"].as_array().unwrap() {
        let result = natural_pair_space(
            &matrix(&case["k"]),
            &matrix(&case["f"]),
            case["occupied_sum"].as_f64().unwrap(),
            case["diagonal"].as_bool().unwrap(),
            case["cutoff"].as_f64().unwrap(),
        )
        .unwrap();
        close(&result.density, &matrix(&case["density"]), 1e-13);
        close(
            &result.coefficients.dot(&result.coefficients.transpose()),
            &matrix(&case["projector"]),
            1e-11,
        );
        close(
            &result
                .coefficients
                .dot(&result.initial_amplitudes)
                .dot(&result.coefficients.transpose()),
            &matrix(&case["initial_ao"]),
            1e-12,
        );
        assert!((result.delta_pno - case["delta_pno"].as_f64().unwrap()).abs() < 1e-13);
        assert!((result.untruncated_estimate - case["estimate"].as_f64().unwrap()).abs() < 1e-13);
        for (a, b) in result
            .natural_occupations
            .iter()
            .zip(case["occupations"].as_array().unwrap())
        {
            assert!((a - b.as_f64().unwrap()).abs() < 1e-13);
        }
    }
}
#[test]
fn invalid_pair_data_singular_gaps_and_unconverged_pcg_return_errors() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/local_pair_reference.json")).unwrap();
    let mut p = problem(&data["cases"][1]);
    assert!(p.solve(1, 1e-14).is_err());
    p.f_occ[(0, 1)] += 0.1;
    assert!(p.solve(100, 1e-10).is_err());
    assert!(natural_pair_space(&Matrix::identity(2), &Matrix::identity(2), 3., false, 0.).is_err());
    let mut f = Matrix::identity(2);
    f[(1, 0)] = 0.1;
    assert!(natural_pair_space(&Matrix::identity(2), &f, -1., false, 0.).is_err());
}
