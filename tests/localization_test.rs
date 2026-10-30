use hf_rs::Matrix;
use hf_rs::local::localization::{foster_boys, pipek_mezey};
use hf_rs::local::pao::{construct_paos, orthonormalize_paos};

fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows.first().unwrap().as_array().unwrap().len(),
        rows.iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
            .collect(),
    )
}
fn objective(c: &Matrix, ops: &[Matrix]) -> f64 {
    ops.iter()
        .map(|m| {
            let d = c.transpose().dot(m).dot(c);
            (0..d.rows).map(|i| d[(i, i)].powi(2)).sum::<f64>()
        })
        .sum()
}
fn rotated(c: &Matrix, i: usize, j: usize, angle: f64) -> Matrix {
    let mut out = c.clone();
    for mu in 0..c.rows {
        out[(mu, i)] = angle.cos() * c[(mu, i)] + angle.sin() * c[(mu, j)];
        out[(mu, j)] = -angle.sin() * c[(mu, i)] + angle.cos() * c[(mu, j)];
    }
    out
}
#[test]
fn every_pair_rotation_maximizes_boys_and_mulliken_populations() {
    let m = Matrix::from_vec(2, 2, vec![1.0, 0.37, 0.37, -0.4]);
    let zero = Matrix::zeros(2, 2);
    let r = foster_boys(
        &Matrix::identity(2),
        &[m.clone(), zero.clone(), zero],
        100,
        1e-12,
    )
    .unwrap();
    let expected = m.data.iter().map(|x| x * x).sum::<f64>();
    assert!((objective(&r.coefficients, &[m]) - expected).abs() < 1e-12);
    let c = rotated(&Matrix::identity(2), 0, 1, 0.31);
    let pm = pipek_mezey(&c, &Matrix::identity(2), &[(0, 1), (1, 2)], 100, 1e-12).unwrap();
    assert!((pm.objective_history.last().unwrap() - 2.0).abs() < 1e-12);
    // At theta=pi/4 the PM gradient vanishes at a minimum, not a maximum.
    let saddle = rotated(&Matrix::identity(2), 0, 1, std::f64::consts::FRAC_PI_4);
    let escaped =
        pipek_mezey(&saddle, &Matrix::identity(2), &[(0, 1), (1, 2)], 100, 1e-12).unwrap();
    assert!((escaped.objective_history.last().unwrap() - 2.0).abs() < 1e-12);
}
#[test]
fn molecular_objectives_all_orbital_gradients_and_pao_projector_match_independent_reference() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/localization_reference.json")).unwrap();
    let c = matrix(&data["coefficients"]);
    let s = matrix(&data["overlap"]);
    let dipoles = [
        matrix(&data["dipoles"][0]),
        matrix(&data["dipoles"][1]),
        matrix(&data["dipoles"][2]),
    ];
    let ranges = data["atom_ao_ranges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r[0].as_u64().unwrap() as usize,
                r[1].as_u64().unwrap() as usize,
            )
        })
        .collect::<Vec<_>>();
    let mut pop = Vec::new();
    for &(start, end) in &ranges {
        let mut p = Matrix::zeros(s.rows, s.cols);
        for mu in 0..s.rows {
            for nu in 0..s.cols {
                p[(mu, nu)] = 0.5
                    * s[(mu, nu)]
                    * (f64::from((start..end).contains(&mu))
                        + f64::from((start..end).contains(&nu)));
            }
        }
        pop.push(p);
    }
    for (name, result, ops) in [
        (
            "boys",
            foster_boys(&c, &dipoles, 1000, 1e-10).unwrap(),
            dipoles.to_vec(),
        ),
        (
            "pipek_mezey",
            pipek_mezey(&c, &s, &ranges, 1000, 1e-10).unwrap(),
            pop,
        ),
    ] {
        assert!(
            (objective(&result.coefficients, &ops)
                - data["objectives"][name]["optimized"].as_f64().unwrap())
            .abs()
                < 1e-9,
            "{name}: actual={}, expected={}, gradient={}",
            objective(&result.coefficients, &ops),
            data["objectives"][name]["optimized"],
            result.gradient_max
        );
        for pair in result.objective_history.windows(2) {
            assert!(pair[1] >= pair[0] - 1e-12);
        }
        let gram = result
            .coefficients
            .transpose()
            .dot(&s)
            .dot(&result.coefficients);
        for i in 0..c.cols {
            for j in 0..c.cols {
                assert!((gram[(i, j)] - f64::from(i == j)).abs() < 1e-12);
            }
        }
        for i in 0..c.cols {
            for j in i + 1..c.cols {
                let step = 1e-3;
                let f = |x| objective(&rotated(&result.coefficients, i, j, x), &ops);
                let derivative = (f(-2.0 * step) - 8.0 * f(-step) + 8.0 * f(step) - f(2.0 * step))
                    / (12.0 * step);
                assert!(derivative.abs() < 1e-8, "{name} pair {i},{j}: {derivative}");
            }
        }
    }
    let pao = construct_paos(&c, &s, 0.0).unwrap();
    assert_eq!(pao.cols, data["n_pao"].as_u64().unwrap() as usize);
    let gram = pao.transpose().dot(&s).dot(&pao);
    let occupied = c.transpose().dot(&s).dot(&pao);
    let projector = pao.dot(&pao.transpose());
    let expected = matrix(&data["pao_projector"]);
    for i in 0..pao.cols {
        for j in 0..pao.cols {
            assert!((gram[(i, j)] - f64::from(i == j)).abs() < 1e-11);
        }
    }
    assert!(occupied.data.iter().all(|x| x.abs() < 1e-11));
    for (&a, &b) in projector.data.iter().zip(&expected.data) {
        assert!((a - b).abs() < 1e-10);
    }
    let q = Matrix::identity(s.rows).sub(&c.dot(&c.transpose()).dot(&s));
    let domain = orthonormalize_paos(&q, &s, &[0, 2, 4], 0.0).unwrap();
    assert!(
        c.transpose()
            .dot(&s)
            .dot(&domain)
            .data
            .iter()
            .all(|x| x.abs() < 1e-10)
    );
}
#[test]
fn invalid_partitions_and_unfinished_localization_return_errors() {
    let c = Matrix::identity(2);
    assert!(pipek_mezey(&c, &c, &[(0, 2), (1, 2)], 100, 1e-10).is_err());
    assert!(pipek_mezey(&c, &c, &[(0, 1)], 100, 1e-10).is_err());
    assert!(construct_paos(&c, &c, -1.0).is_err());
    let r = Matrix::from_vec(2, 2, vec![0.7, 0.3, 0.3, -0.4]);
    assert!(foster_boys(&c, &[r, Matrix::zeros(2, 2), Matrix::zeros(2, 2)], 0, 1e-10).is_err());
}
