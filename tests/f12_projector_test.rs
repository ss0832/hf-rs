// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    Matrix,
    options::F12Hybrid,
    post_hf::f12::intermediates::{F12FockSpace, F12NonlocalPairView, F12PairIntegralView},
};
fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
            .collect(),
    )
}
fn array(v: &serde_json::Value) -> [f64; 2] {
    std::array::from_fn(|i| v[i].as_f64().unwrap())
}
#[test]
fn all_projector_intermediates_and_fixed_pair_energies_match_full_product_space() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/f12_projector_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let space = F12FockSpace {
            n_occ: case["n_occ"].as_u64().unwrap() as usize,
            n_obs: case["n_obs"].as_u64().unwrap() as usize,
            fock: matrix(&case["fock"]),
            nonlocal: matrix(&case["nonlocal_part"]),
            orbital_energies: case["orbital_energies"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect(),
        };
        for pair in case["pairs"].as_array().unwrap() {
            let factor = matrix(&pair["factor"]);
            let coulomb = matrix(&pair["coulomb"]);
            let squared = matrix(&pair["squared"]);
            let squared_exchange = matrix(&pair["squared_exchange"]);
            let input = F12PairIntegralView {
                pair: std::array::from_fn(|i| pair["pair"][i].as_u64().unwrap() as usize),
                factor: &factor,
                coulomb: &coulomb,
                squared: &squared,
                squared_exchange: &squared_exchange,
                coulomb_factor: array(&pair["coulomb_factor"]),
                positive_commutator: array(&pair["commutator"]),
            };
            let result = space.pair_intermediates(input).unwrap();
            for (hybrid, label) in [(F12Hybrid::Hy1, "hy1"), (F12Hybrid::Hy2, "hy2")] {
                let result = space.pair_intermediates_with_hybrid(input, hybrid).unwrap();
                let expected = &pair["hybrids"][label];
                for (a, b) in result.b.into_iter().zip(array(&expected["b"])) {
                    assert!((a - b).abs() < 1e-12, "{label} B: {a} vs {b}");
                }
                let (singlet, triplet) = result.fixed_energy();
                assert!((singlet - expected["singlet"].as_f64().unwrap()).abs() < 1e-12);
                assert!((triplet - expected["triplet"].as_f64().unwrap()).abs() < 1e-12);
            }
            for (actual, key) in [(result.v, "v"), (result.x, "x"), (result.b, "b")] {
                for (a, b) in actual.into_iter().zip(array(&pair[key])) {
                    assert!((a - b).abs() < 1e-12, "{key}: {a} vs {b}");
                }
            }
            for (actual, key) in [(&result.c, "c"), (&result.g, "g")] {
                let expected = matrix(&pair[key]);
                assert_eq!((actual.rows, actual.cols), (expected.rows, expected.cols));
                for (&a, &b) in actual.data.iter().zip(&expected.data) {
                    assert!((a - b).abs() < 1e-12, "{key}: {a} vs {b}");
                }
            }
            let (s, t) = result.fixed_energy();
            assert!(
                (s - pair["singlet"].as_f64().unwrap()).abs() < 1e-12,
                "singlet {s} vs {}",
                pair["singlet"]
            );
            assert!(
                (t - pair["triplet"].as_f64().unwrap()).abs() < 1e-12,
                "triplet {t} vs {}",
                pair["triplet"]
            );
            let potential = matrix(&case["nonlocal_potential"]);
            let action = matrix(&pair["nonlocal_right_action"]);
            let kinetic = matrix(&pair["kinetic_commutator"]);
            let a = space
                .pair_intermediates_3a_with_nonlocal(
                    input,
                    &kinetic,
                    Some(F12NonlocalPairView {
                        potential: &potential,
                        factor_right_action: &action,
                    }),
                )
                .unwrap();
            assert!(a.c.data.iter().all(|&value| value == 0.0));
            for (actual, expected) in a.b_bar.into_iter().zip(array(&pair["a_b_bar"])) {
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "3*A B: {actual} vs {expected}"
                );
            }
            for (actual, expected) in a.v.into_iter().zip(result.v) {
                assert!((actual - expected).abs() < 1e-12);
            }
            let (s, t) = a.fixed_energy();
            assert!((s - pair["a_singlet"].as_f64().unwrap()).abs() < 1e-12);
            assert!((t - pair["a_triplet"].as_f64().unwrap()).abs() < 1e-12);
            let without = space.pair_intermediates_3a(input, &kinetic).unwrap();
            if potential.data.iter().all(|&value| value == 0.0) {
                assert_eq!(a.b_bar, without.b_bar);
            } else {
                assert!((a.b_bar[0] - without.b_bar[0]).abs() > 1e-7);
            }
            let mut malformed = potential.clone();
            malformed[(0, 1)] += 0.1;
            assert!(
                space
                    .pair_intermediates_3a_with_nonlocal(
                        input,
                        &kinetic,
                        Some(F12NonlocalPairView {
                            potential: &malformed,
                            factor_right_action: &action,
                        }),
                    )
                    .is_err()
            );
        }
    }
}
