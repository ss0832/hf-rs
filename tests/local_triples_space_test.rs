use hf_rs::{Matrix, local::triples_space::*};
use std::sync::Arc;
fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect(),
    )
}
fn vector(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn problem(case: &serde_json::Value) -> LocalTriplesProblem {
    LocalTriplesProblem {
        f_occ: matrix(&case["f_occ"]),
        blocks: case["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| TriplesBlock {
                i: b["i"].as_u64().unwrap() as usize,
                j: b["j"].as_u64().unwrap() as usize,
                k: b["k"].as_u64().unwrap() as usize,
                space: Arc::new(TriplesVirtualSpace {
                    basis: matrix(&b["basis"]),
                    energies: vector(&b["energies"]),
                }),
                connected: vector(&b["w"]),
                singles: vector(&b["v"]),
            })
            .collect(),
    }
}
#[test]
fn every_three_leg_overlap_transform_matches_dense_kronecker_reference() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/local_triples_space_reference.json")).unwrap();
    let case = &data["transform"];
    let actual = transform_cube(&vector(&case["cube"]), 3, &matrix(&case["overlap"])).unwrap();
    for (a, b) in actual.iter().zip(vector(&case["expected"])) {
        assert!((a - b).abs() < 1e-12);
    }
}
#[test]
fn full_and_finite_tno_triples_match_all_dense_t0_and_t1_amplitudes() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/local_triples_space_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let p = problem(case);
        for (iterative, name, energy) in [(false, "t0", "e_t0"), (true, "t1", "e_t1")] {
            let solution = p.solve(iterative, 100, 1e-13).unwrap();
            assert!((solution.energy - case[energy].as_f64().unwrap()).abs() < 1e-12);
            for (actual, expected) in solution
                .amplitudes
                .iter()
                .zip(case[name].as_array().unwrap())
            {
                for (a, b) in actual.iter().zip(vector(expected)) {
                    assert!((a - b).abs() < 1e-12, "{} {name}: {a} vs {b}", case["name"]);
                }
            }
            if iterative {
                let applied = p.apply(&solution.amplitudes).unwrap();
                for (a, b) in applied.iter().zip(&p.blocks) {
                    for (a, w) in a.iter().zip(&b.connected) {
                        assert!((a + w).abs() < 1e-13);
                    }
                }
            }
        }
        assert!(p.solve(true, 1, 1e-14).is_err());
    }
}
