// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Exact Lowdin spin projection of UHF determinants against full-CI-space projections of the same
//! PySCF determinants (tests/reference/gen_puhf.py), Yamaguchi's estimate, and exact limits.

use hf_rs::{
    Atom, build_basis, get_builtin_basis, project_uhf_spin, run_puhf, run_puhf_with_multiplicity,
    run_uhf,
};

#[test]
fn exact_spin_projection_matches_full_ci_space_projection() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/puhf_reference.json")).unwrap();
    let (mut worst_e, mut worst_w) = (0f64, 0f64);
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let atoms: Vec<Atom> = case["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                Atom::new(
                    a["symbol"].as_str().unwrap(),
                    std::array::from_fn(|k| a["position"][k].as_f64().unwrap()),
                )
            })
            .collect();
        let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
        let charge = case["charge"].as_i64().unwrap() as i32;
        let two_s = case["spin_2s"].as_u64().unwrap() as usize;
        let uhf = run_uhf(&atoms, &basis, charge, two_s + 1, two_s == 0, 500, 1e-12).unwrap();
        // the same stationary determinant is projected on both sides
        assert!(
            (uhf.total_energy - case["e_uhf"].as_f64().unwrap()).abs() < 1e-8,
            "{name}: UHF {} vs {}",
            uhf.total_energy,
            case["e_uhf"]
        );
        assert!(
            (uhf.s_squared - case["s2_uhf"].as_f64().unwrap()).abs() < 1e-6,
            "{name}: <S^2>"
        );
        let p = project_uhf_spin(&uhf).unwrap();
        let de = (p.energy - case["e_projected"].as_f64().unwrap()).abs();
        let dw = (p.target_weight - case["target_weight"].as_f64().unwrap()).abs();
        worst_e = worst_e.max(de);
        worst_w = worst_w.max(dw);
        assert!(
            de < 1e-8,
            "{name}: projected {} vs {}",
            p.energy,
            case["e_projected"]
        );
        assert!(
            dw < 1e-8,
            "{name}: target weight {} vs {}",
            p.target_weight,
            case["target_weight"]
        );
        assert!(p.target_weight > 0.0 && p.target_weight <= 1.0 + 1e-12);
        assert!((p.spin - 0.5 * two_s as f64).abs() < 1e-12);

        // Projection depends only on occupied orbitals, including in a truncated virtual space.
        let mut occupied_only = uhf.clone();
        occupied_only.mo_coefficients_alpha =
            uhf.mo_coefficients_alpha.leading_columns(uhf.n_alpha);
        occupied_only.mo_coefficients_beta = uhf.mo_coefficients_beta.leading_columns(uhf.n_beta);
        let occupied_projection = project_uhf_spin(&occupied_only).unwrap();
        assert!((occupied_projection.energy - p.energy).abs() < 1e-12);
        assert!((occupied_projection.target_weight - p.target_weight).abs() < 1e-12);

        let puhf =
            run_puhf_with_multiplicity(&atoms, &basis, charge, two_s + 1, 500, 1e-12).unwrap();
        assert!(
            (puhf.projected_energy - p.energy).abs() < 1e-8,
            "{name}: run_puhf projects the same determinant"
        );
        // Yamaguchi estimate from the same two determinants
        let st = 0.5 * two_s as f64;
        let t = st * (st + 1.0);
        let (e_bs, s_bs) = (
            case["e_uhf"].as_f64().unwrap(),
            case["s2_uhf"].as_f64().unwrap(),
        );
        let (e_hs, s_hs) = (
            case["e_high_spin"].as_f64().unwrap(),
            case["s2_high_spin"].as_f64().unwrap(),
        );
        let yamaguchi = (e_bs * (s_hs - t) - e_hs * (s_bs - t)) / (s_hs - s_bs);
        assert!(
            (puhf.yamaguchi_energy.unwrap() - yamaguchi).abs() < 1e-7,
            "{name}: Yamaguchi"
        );
        assert!((puhf.spin_contamination_weight + puhf.target_weight - 1.0).abs() < 1e-14);
    }
    println!("PUHF: max |dE| {worst_e:e}, max |d weight| {worst_w:e}");
}

#[test]
fn projecting_a_spin_pure_determinant_is_the_identity() {
    // Equilibrium H2: the UHF solution is the closed-shell RHF determinant (pure singlet).
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -0.7]),
        Atom::new("H", [0.0, 0.0, 0.7]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let uhf = run_uhf(&atoms, &basis, 0, 1, false, 200, 1e-12).unwrap();
    assert!(uhf.s_squared.abs() < 1e-10);
    let p = project_uhf_spin(&uhf).unwrap();
    assert!((p.target_weight - 1.0).abs() < 1e-10);
    assert!((p.energy - uhf.total_energy).abs() < 1e-10);
    // High-spin determinants (S_z = S_max) are already pure.
    let li = vec![Atom::new("Li", [0.0, 0.0, 0.0])];
    let lb = build_basis(&li, &get_builtin_basis("sto-3g").unwrap());
    let doublet = run_uhf(&li, &lb, 0, 2, false, 200, 1e-12).unwrap();
    let pd = project_uhf_spin(&doublet).unwrap();
    assert!(
        pd.target_weight > 0.99,
        "Li doublet UHF is nearly pure (weight {})",
        pd.target_weight
    );

    // A full one-orbital basis has no higher-spin sector. Exact projection still exists.
    let he = vec![Atom::new("He", [0.0; 3])];
    let hb = build_basis(&he, &get_builtin_basis("sto-3g").unwrap());
    let pure = run_puhf(&he, &hb, 0, 100, 1e-12).unwrap();
    assert!((pure.target_weight - 1.0).abs() < 1e-12);
    assert!((pure.projected_energy - pure.raw_uhf.total_energy).abs() < 1e-12);
    assert!(pure.high_spin_reference_energy.is_none());
    assert!(pure.yamaguchi_energy.is_none());
}
