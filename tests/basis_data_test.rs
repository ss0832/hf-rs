// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Embedded basis set data: oracle comparison with the BSE JSON manifest
//! (tests/reference/gen_basis_manifest.py) and physical requirements (normalization, positive-definite
//! overlap) for every element of every built-in orbital basis set.

use hf_rs::basis::{shell_type_to_l, PERIODIC_TABLE};
use hf_rs::integrals::contracted_overlap;
use hf_rs::{build_basis, get_builtin_aux_basis, get_builtin_basis, list_builtin_aux_basis_sets, list_builtin_basis_sets, Atom, BasisSetMap};

fn manifest() -> serde_json::Value {
    serde_json::from_str(include_str!("data/basis_manifest.json")).unwrap()
}

/// (l, number of primitives, sum of exponents, sum of |coefficients|) per shell, sorted for comparison.
fn shell_signature(map: &BasisSetMap, symbol: &str) -> Vec<(usize, usize, f64, f64)> {
    let mut out = Vec::new();
    for sh in &map[&symbol.to_uppercase()] {
        let nonzero = |coeffs: &[f64]| -> (usize, f64, f64) {
            let mut n = 0;
            let (mut se, mut sc) = (0.0, 0.0);
            for (e, c) in sh.exponents.iter().zip(coeffs) {
                if *c != 0.0 {
                    n += 1;
                    se += e;
                    sc += c.abs();
                }
            }
            (n, se, sc)
        };
        if sh.shell_type == "SP" {
            let (n, se, sc) = nonzero(&sh.coefficients);
            out.push((0, n, se, sc));
            let (n, se, sc) = nonzero(sh.p_coefficients.as_ref().unwrap());
            out.push((1, n, se, sc));
        } else {
            let (n, se, sc) = nonzero(&sh.coefficients);
            out.push((shell_type_to_l(&sh.shell_type).unwrap(), n, se, sc));
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

#[test]
fn embedded_basis_data_matches_bse_json_manifest() {
    let manifest = manifest();
    let sets = manifest["basis_sets"].as_object().unwrap();
    let mut n_checked = 0;
    for (name, elements) in sets {
        let map = get_builtin_basis(name)
            .or_else(|| get_builtin_aux_basis(name))
            .unwrap_or_else(|| panic!("basis '{}' is in the manifest but not built in", name));
        let elements = elements.as_object().unwrap();
        assert_eq!(map.len(), elements.len(), "{}: element coverage differs from BSE", name);
        for (z, shells) in elements {
            let symbol = PERIODIC_TABLE[z.parse::<usize>().unwrap() - 1];
            assert!(map.contains_key(&symbol.to_uppercase()), "{}: missing element {}", name, symbol);
            let mut expected: Vec<(usize, usize, f64, f64)> = shells
                .as_array()
                .unwrap()
                .iter()
                .map(|s| {
                    let s = s.as_array().unwrap();
                    (s[0].as_u64().unwrap() as usize, s[1].as_u64().unwrap() as usize, s[2].as_f64().unwrap(), s[3].as_f64().unwrap())
                })
                .collect();
            expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let got = shell_signature(&map, symbol);
            assert_eq!(got.len(), expected.len(), "{} {}: number of shells", name, symbol);
            for (g, e) in got.iter().zip(&expected) {
                assert_eq!((g.0, g.1), (e.0, e.1), "{} {}: shell structure", name, symbol);
                assert!((g.2 - e.2).abs() <= 1e-9 * e.2.abs().max(1.0), "{} {}: exponent sum {} vs {}", name, symbol, g.2, e.2);
                assert!((g.3 - e.3).abs() <= 1e-9 * e.3.abs().max(1.0), "{} {}: coefficient sum {} vs {}", name, symbol, g.3, e.3);
            }
            n_checked += 1;
        }
    }
    assert!(n_checked > 700, "manifest unexpectedly small ({})", n_checked);
}

#[test]
fn every_built_in_name_is_in_the_manifest() {
    let manifest = manifest();
    let sets = manifest["basis_sets"].as_object().unwrap();
    for name in list_builtin_basis_sets().into_iter().chain(list_builtin_aux_basis_sets()) {
        assert!(sets.contains_key(name), "{} missing from manifest", name);
    }
}

#[test]
fn contracted_functions_are_normalized_for_all_elements() {
    // <chi|chi> = 1 for every Cartesian component of every shell (up to g for the larger sets).
    for name in list_builtin_basis_sets() {
        let map = get_builtin_basis(name).unwrap();
        let mut symbols: Vec<&String> = map.keys().collect();
        symbols.sort();
        for sym in symbols {
            let atom = Atom::new(sym, [0.0, 0.0, 0.0]);
            for cg in build_basis(&[atom], &map) {
                let s = contracted_overlap(&cg, &cg);
                assert!((s - 1.0).abs() < 1e-10, "{} {} l={:?}: <chi|chi> = {:.14}", name, sym, cg.l, s);
            }
        }
    }
}

#[test]
fn overlap_matrices_are_positive_definite_diatomics() {
    // Physical requirement: S is symmetric positive definite for any non-degenerate geometry.
    for name in ["sto-3g", "6-31g*", "def2-svp", "cc-pvdz"] {
        let map = get_builtin_basis(name).unwrap();
        let atoms = [Atom::new("C", [0.0, 0.0, 0.0]), Atom::new("O", [0.0, 0.0, 2.13])];
        let basis = build_basis(&atoms, &map);
        let n = basis.len();
        let mut s = hf_rs::Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                s[(i, j)] = contracted_overlap(&basis[i], &basis[j]);
            }
        }
        for i in 0..n {
            for j in 0..i {
                assert!((s[(i, j)] - s[(j, i)]).abs() < 1e-13, "{}: S not symmetric", name);
            }
        }
        let (evals, _) = s.eigh();
        assert!(evals[0] > 1e-4, "{}: smallest overlap eigenvalue {}", name, evals[0]);
    }
}

#[test]
fn basis_name_aliases_resolve_to_the_same_data() {
    for (a, b) in [
        ("6-31G(d)", "6-31g*"),
        ("6-31G(d,p)", "6-31g**"),
        ("6-311G(d,p)", "6-311g**"),
        ("cc-pVDZ", "CC_PVDZ"),
        ("def2-SVP", "def2svp"),
        ("STO-3G", "sto3g"),
    ] {
        let (ma, mb) = (get_builtin_basis(a).unwrap(), get_builtin_basis(b).unwrap());
        assert_eq!(shell_signature(&ma, "C"), shell_signature(&mb, "C"), "{} vs {}", a, b);
    }
    assert!(get_builtin_basis("not-a-basis").is_none());
    assert!(get_builtin_aux_basis("cc-pvdz").is_none(), "orbital sets must not resolve as auxiliary sets");
}
