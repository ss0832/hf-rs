// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Self-consistent HFB against independent canonical-BCS minimization and explicit Fock-space
//! Hamiltonian/RDM checks. Every normal and anomalous AO density entry is compared.

use hf_rs::{
    Atom, ConvAlgo, HfbOptions, Matrix, build_basis, get_builtin_basis, run_hfb,
    run_hfb_from_reference, run_scf_general,
};

#[test]
fn hfb_energies_and_complete_densities_match_full_fock_oracles() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/hfb_reference.json")).unwrap();
    let (mut worst_e, mut worst_d, mut worst_k) = (0f64, 0f64, 0f64);
    let mut entries = 0;
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
        let hf = run_scf_general(
            &atoms,
            &basis,
            case["charge"].as_i64().unwrap() as i32,
            300,
            1e-13,
            ConvAlgo::Adiissoscf,
            None,
        )
        .unwrap();
        let options = HfbOptions {
            pairing_strength: case["pairing_strength"].as_f64().unwrap(),
            energy_tol: 1e-12,
            residual_tol: 1e-10,
            density_tol: 1e-10,
            max_iterations: 300,
            ..Default::default()
        };
        let result = run_hfb_from_reference(&hf, &options).unwrap();
        assert!(
            result.converged,
            "{name}: HFB residual {}, iterations {}",
            result.residual_norm, result.iterations
        );
        assert!(result.residual_norm < 2e-10, "{name}: stationarity");
        assert!(result.number_error < 2e-12, "{name}: number");
        assert!(
            result.idempotence_error < 1e-12,
            "{name}: pure generalized density"
        );
        let de = (result.total_hfb_energy - case["energy"].as_f64().unwrap()).abs();
        worst_e = worst_e.max(de);
        assert!(
            de < 1e-9,
            "{name}: energy {} vs {}",
            result.total_hfb_energy,
            case["energy"]
        );
        assert!(
            (result.pairing_energy - case["pairing_energy"].as_f64().unwrap()).abs() < 1e-8,
            "{name}: complete pairing interaction"
        );
        assert!(
            (result.total_hfb_energy
                - result.normal_component_energy
                - result.coulomb_pairing_energy
                - result.pairing_energy)
                .abs()
                < 1e-12
        );
        assert!(
            (result.particle_number_variance - case["particle_number_variance"].as_f64().unwrap())
                .abs()
                < 1e-8,
            "{name}: number variance"
        );
        let c = &result.orbital_coefficients;
        let d = c.dot(&result.normal_density).dot(&c.transpose());
        let k = c.dot(&result.pairing_tensor).dot(&c.transpose());
        let gauge = if result.pairing_tensor.trace() < 0. {
            -1.
        } else {
            1.
        };
        for i in 0..d.rows {
            for j in 0..d.cols {
                let dd = (d[(i, j)] - case["normal_density_ao"][i][j].as_f64().unwrap()).abs();
                let dk =
                    (gauge * k[(i, j)] - case["pairing_tensor_ao"][i][j].as_f64().unwrap()).abs();
                worst_d = worst_d.max(dd);
                worst_k = worst_k.max(dk);
                assert!(
                    dd < 1e-8 && dk < 1e-8,
                    "{name}: AO density ({i},{j}): {dd:e}/{dk:e}"
                );
                entries += 2;
            }
        }
        if result.particle_number_variance > 1e-3 {
            assert!(
                (result.chemical_potential - case["chemical_potential"].as_f64().unwrap()).abs()
                    < 1e-8,
                "{name}: mu"
            );
            assert!(
                (result.pairing_gap - case["pairing_gap"].as_f64().unwrap()).abs() < 1e-8,
                "{name}: gap"
            );
            for (i, e) in result.quasiparticle_energies.iter().enumerate() {
                assert!(
                    (e - case["quasiparticle_energies"][i].as_f64().unwrap()).abs() < 1e-8,
                    "{name}: quasiparticle {i}"
                );
            }
        }
        if options.pairing_strength == 0. {
            assert!((result.total_hfb_energy - hf.total_energy).abs() < 1e-12);
            assert!(result.pairing_energy.abs() < 1e-14);
        }
        println!(
            "HFB {name}: E={:.14}, variance={:.10}, residual={:.2e}, iterations={}",
            result.total_hfb_energy,
            result.particle_number_variance,
            result.residual_norm,
            result.iterations
        );
    }
    println!(
        "HFB full comparison: energies={worst_e:e}, {entries} density entries: normal={worst_d:e}, anomalous={worst_k:e}"
    );
}

#[test]
fn hfb_is_invariant_to_real_orbital_frames_and_rejects_invalid_controls() {
    let atoms = vec![
        Atom::new("H", [0., 0., 0.]),
        Atom::new("H", [2.1, 0.2, 0.]),
        Atom::new("H", [0.4, 1.7, 0.3]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_scf_general(&atoms, &basis, 1, 300, 1e-13, ConvAlgo::Adiissoscf, None).unwrap();
    let options = HfbOptions {
        pairing_strength: 1.5,
        energy_tol: 1e-12,
        residual_tol: 1e-10,
        density_tol: 1e-10,
        ..Default::default()
    };
    let first = run_hfb_from_reference(&hf, &options).unwrap();
    let mut rotated = hf.clone();
    let (s, c) = 0.43f64.sin_cos();
    let q = Matrix::from_vec(3, 3, vec![1., 0., 0., 0., c, -s, 0., s, c]);
    rotated.mo_coefficients = hf.mo_coefficients.dot(&q);
    let second = run_hfb_from_reference(&rotated, &options).unwrap();
    assert!(first.converged && second.converged);
    assert!((first.total_hfb_energy - second.total_hfb_energy).abs() < 1e-10);
    let original = first
        .orbital_coefficients
        .dot(&first.normal_density)
        .dot(&first.orbital_coefficients.transpose());
    let transported = second
        .orbital_coefficients
        .dot(&second.normal_density)
        .dot(&second.orbital_coefficients.transpose());
    assert!(original.rmsd(&transported) < 1e-10);
    assert!(
        run_hfb_from_reference(
            &hf,
            &HfbOptions {
                pairing_strength: -0.1,
                ..options.clone()
            }
        )
        .is_err()
    );
    assert!(
        run_hfb_from_reference(
            &hf,
            &HfbOptions {
                number_tol: f64::NAN,
                ..options
            }
        )
        .is_err()
    );
    let h = vec![Atom::new("H", [0.; 3])];
    let hb = build_basis(&h, &get_builtin_basis("sto-3g").unwrap());
    assert!(
        run_hfb(&h, &hb, 0, 0.1, 100, 1e-10)
            .unwrap_err()
            .contains("blocking")
    );
}

#[test]
fn analytic_dispatch_never_silently_uses_energy_differences() {
    let atoms = vec![
        Atom::new("H", [0., 0., -0.7]),
        Atom::new("H", [0., 0., 0.7]),
    ];
    let map = get_builtin_basis("sto-3g").unwrap();
    use hf_rs::HFVariantType;
    for variant in [
        HFVariantType::PUHF,
        HFVariantType::HFB,
        HFVariantType::DFHF,
        HFVariantType::CDHF,
        HFVariantType::NEOHF,
        HFVariantType::LHF,
    ] {
        let error =
            hf_rs::compute_hf_variant_analytical_gradient(&atoms, &map, variant, 0, 100, 1e-10)
                .unwrap_err();
        assert!(error.contains("analytic nuclear gradients are not implemented"));
        assert!(hf_rs::compute_hf_variant_gradient(&atoms, &map, variant, 0, 100, 1e-10).is_err());
    }
    assert!(
        hf_rs::compute_hf_variant_numerical_gradient(
            &atoms,
            &map,
            HFVariantType::PUHF,
            0,
            100,
            1e-10,
            f64::NAN
        )
        .is_err()
    );
}
