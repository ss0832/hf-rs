// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! TDHF/RPA against complete PySCF-based singlet and triplet spectra
//! (tests/reference/gen_tdhf.py): every excitation energy, oscillator strengths summed over
//! degenerate groups, gauge-invariant amplitude Gram spectra, metric orthonormality,
//! spin selection and explicit instability errors.

use hf_rs::{assign_builtin_ecps, build_basis, get_builtin_basis, run_scf_general, run_tdhf_from_reference, Atom, ConvAlgo, Matrix, TDHFState};

fn basis_name(constant: &str) -> &'static str {
    match constant {
        "STO_3G_GBS" => "sto-3g",
        "BASIS_6_31G_GBS" => "6-31g",
        "BASIS_DEF2_SVP_GBS" => "def2-svp",
        other => panic!("{other}"),
    }
}

/// Degenerate groups of ascending energies.
fn groups(e: &[f64]) -> Vec<std::ops::Range<usize>> {
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    for k in 0..e.len() {
        match out.last_mut() {
            Some(r) if (e[k] - e[r.end - 1]).abs() < 1e-6 => r.end = k + 1,
            _ => out.push(k..k + 1),
        }
    }
    out
}

fn gram_spectrum(vecs: &[Vec<f64>]) -> Vec<f64> {
    let m = vecs.len();
    let mut g = Matrix::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            g[(i, j)] = vecs[i].iter().zip(&vecs[j]).map(|(a, b)| a * b).sum();
        }
    }
    g.eigh().0
}

#[test]
fn tdhf_spectra_match_complete_pyscf_rpa_references() {
    let data: serde_json::Value = serde_json::from_str(include_str!("data/tdhf_reference.json")).unwrap();
    let (mut n_roots, mut worst_e, mut worst_f, mut worst_g) = (0usize, 0f64, 0f64, 0f64);
    let mut unstable_seen = 0;
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut atoms: Vec<Atom> = case["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| Atom::new(a["symbol"].as_str().unwrap(), std::array::from_fn(|k| a["position"][k].as_f64().unwrap())))
            .collect();
        let bname = basis_name(case["basis_constant"].as_str().unwrap());
        if case["ecp"].is_string() {
            assign_builtin_ecps(&mut atoms, bname);
        }
        let basis = build_basis(&atoms, &get_builtin_basis(bname).unwrap());
        let hf = run_scf_general(&atoms, &basis, 0, 300, 1e-12, ConvAlgo::Cdiis, None).unwrap();
        assert!(hf.converged);
        assert!((hf.total_energy - case["e_rhf"].as_f64().unwrap()).abs() < 1e-8, "{name} RHF");
        let n_ov = hf.n_occ * (hf.mo_coefficients.cols - hf.n_occ);
        for (label, triplet) in [("singlet", false), ("triplet", true)] {
            let reference = &case[label];
            let result = if triplet {
                run_tdhf_from_reference(&hf, &basis, 0, n_ov)
            } else {
                run_tdhf_from_reference(&hf, &basis, n_ov, 0)
            };
            if reference["unstable"].as_bool().unwrap() {
                let err = result.unwrap_err();
                assert!(err.contains("unstable"), "{name} {label}: {err}");
                unstable_seen += 1;
                continue;
            }
            let result = result.unwrap();
            let states: Vec<&TDHFState> = result.excited_states.iter().collect();
            assert_eq!(states.len(), n_ov);
            let e_ref: Vec<f64> = reference["energies"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let f_ref: Vec<f64> = reference["oscillator_strengths"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let p_ref: Vec<Vec<f64>> = reference["x_plus_y"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
                .collect();
            for (k, s) in states.iter().enumerate() {
                assert_eq!(s.is_triplet, triplet);
                let error = (s.excitation_energy_hartree - e_ref[k]).abs();
                worst_e = worst_e.max(error);
                n_roots += 1;
                assert!(error < 1e-8, "{name} {label} root {k}: {} vs {}", s.excitation_energy_hartree, e_ref[k]);
                if triplet {
                    assert_eq!(s.oscillator_strength, 0.0, "triplet transitions are spin forbidden");
                }
                // metric orthonormality X_k.X_l - Y_k.Y_l = delta_kl (not imposed by any post-processing)
                for (l, t) in states.iter().enumerate().take(k + 1) {
                    let dot: f64 = s.x.iter().zip(&t.x).map(|(a, b)| a * b).sum::<f64>()
                        - s.y.iter().zip(&t.y).map(|(a, b)| a * b).sum::<f64>();
                    let expected = if k == l { 1.0 } else { 0.0 };
                    assert!((dot - expected).abs() < 1e-8, "{name} {label} metric ({k},{l}) = {dot}");
                }
            }
            for g in groups(&e_ref) {
                let f_sum: f64 = states[g.clone()].iter().map(|s| s.oscillator_strength).sum();
                let f_sum_ref: f64 = f_ref[g.clone()].iter().sum();
                worst_f = worst_f.max((f_sum - f_sum_ref).abs());
                assert!((f_sum - f_sum_ref).abs() < 1e-8, "{name} {label} oscillator group {g:?}");
                let plus: Vec<Vec<f64>> = states[g.clone()]
                    .iter()
                    .map(|s| s.x.iter().zip(&s.y).map(|(x, y)| x + y).collect())
                    .collect();
                let a = gram_spectrum(&plus);
                let b = gram_spectrum(&p_ref[g.clone()]);
                for (x, y) in a.iter().zip(&b) {
                    worst_g = worst_g.max((x - y).abs());
                    assert!((x - y).abs() < 1e-6 * (1.0 + y.abs()), "{name} {label} amplitude Gram {x} vs {y}");
                }
            }
        }
    }
    assert_eq!(unstable_seen, 1, "the formaldehyde/STO-3G triplet instability must be reported");
    println!("TDHF: {n_roots} roots, max |dE| {worst_e:e}, max |df| {worst_f:e}, max amplitude Gram {worst_g:e}");
}

#[test]
fn tdhf_singlet_and_triplet_ordering_for_h2() {
    let atoms = vec![Atom::new("H", [0.0, 0.0, -0.7]), Atom::new("H", [0.0, 0.0, 0.7])];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let legacy = hf_rs::run_tdhf(&atoms, &basis, 0, 1, 100, 1e-12).unwrap();
    assert_eq!((legacy.n_singlet_states, legacy.n_triplet_states), (1, 0));
    let hf = run_scf_general(&atoms, &basis, 0, 100, 1e-12, ConvAlgo::Cdiis, None).unwrap();
    let r = run_tdhf_from_reference(&hf, &basis, 1, 1).unwrap();
    assert!(!r.excited_states[0].is_triplet && r.excited_states[1].is_triplet);
    assert!((r.excited_states[0].excitation_energy_hartree - legacy.excited_states[0].excitation_energy_hartree).abs() < 1e-10);
    // H2 minimal basis: one sigma_g -> sigma_u excitation; exchange places the singlet above the triplet.
    assert!(r.excited_states[0].excitation_energy_hartree > r.excited_states[1].excitation_energy_hartree);
    assert!(r.excited_states[0].oscillator_strength > 0.0 && r.excited_states[1].oscillator_strength == 0.0);
}
