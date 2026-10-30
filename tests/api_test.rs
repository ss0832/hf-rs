// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    api::{self, LengthUnit, Molecule, MoleculeOptions, OptionMap, ReferenceChoice, ScfOptions},
    context::RefKind,
};
use serde_json::{Value, json};
use std::sync::Arc;

fn h2() -> Molecule {
    Molecule::new(
        &[1, 1],
        &[[0., 0., -0.7], [0., 0., 0.7]],
        MoleculeOptions::default(),
    )
    .unwrap()
}
fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a} vs {b}");
}

#[test]
fn intermediates_are_opt_in_bounded_and_reconstruct_ccsd_energy() {
    let molecule = h2();
    assert!(
        molecule
            .run("hf", &OptionMap::new(), &[])
            .unwrap()
            .intermediates
            .is_empty()
    );
    let model = molecule.intermediates("hf", &OptionMap::new(), 20).unwrap();
    assert_eq!(model.values().map(|t| t.values.len()).sum::<usize>(), 20);
    assert!(molecule.intermediates("hf", &OptionMap::new(), 19).is_err());
    assert!(molecule.intermediates("hf", &OptionMap::new(), 0).is_err());
    let output = molecule
        .run("ccsd", &OptionMap::new(), &["intermediates".into()])
        .unwrap();
    let tensors = &output.intermediates;
    assert_eq!(tensors["method.t1"].shape, vec![1, 1]);
    assert_eq!(tensors["method.t2"].shape, vec![1, 1, 1, 1]);
    let integral = molecule
        .integrals(
            &api::IntegralRequest {
                operator: "eri".into(),
                representation: "mo_alpha".into(),
                indices: vec![vec![0], vec![1], vec![0], vec![1]],
                ..Default::default()
            },
            ScfOptions::default(),
        )
        .unwrap()
        .tensor
        .values[0];
    let t1 = tensors["method.t1"].values[0];
    let t2 = tensors["method.t2"].values[0];
    close(integral * (t2 + t1 * t1), output.output.e_corr, 1e-10);
    assert!(
        molecule
            .intermediates("ccsd", &OptionMap::new(), 1)
            .is_err()
    );
    let qed = molecule
        .intermediates("qed-hf", &OptionMap::new(), 100)
        .unwrap();
    assert!(qed.contains_key("model.fock_cartesian_ao"));
    assert_eq!(qed["model.fock_cartesian_ao"].unit, "hartree");
    // A result that does not retain amplitudes does not synthesize them.
    assert!(
        molecule
            .intermediates("mp2", &OptionMap::new(), 100)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn bounded_integral_blocks_reconstruct_hf_and_match_mo_context() {
    use hf_rs::api::IntegralRequest;
    let molecule = h2();
    let settings = ScfOptions::default();
    let reference = molecule.scf(settings).unwrap();
    let api::ScfReference::Rhf(hf) = &*reference else {
        panic!("RHF required")
    };
    for (operator, matrix) in [
        ("overlap", &hf.overlap_matrix),
        ("core_hamiltonian", &hf.core_hamiltonian),
        ("fock", &hf.fock_matrix),
    ] {
        let block = molecule
            .integrals(
                &IntegralRequest {
                    operator: operator.into(),
                    ..Default::default()
                },
                settings,
            )
            .unwrap();
        assert_eq!(block.tensor.shape, vec![2, 2]);
        assert_eq!(block.tensor.axes, vec!["p", "q"]);
        for (&v, &r) in block.tensor.values.iter().zip(&matrix.data) {
            close(v, r, 1e-12);
        }
    }
    let context =
        hf_rs::context::PostHFContext::from_rhf(hf, molecule.atoms(), molecule.basis(), None)
            .unwrap();
    let mo = context.ints.full().unwrap();
    let indices = vec![vec![1, 0], vec![0, 1], vec![1], vec![1, 0]];
    let block = molecule
        .integrals(
            &IntegralRequest {
                operator: "eri".into(),
                representation: "mo_alpha".into(),
                indices: indices.clone(),
                ..Default::default()
            },
            settings,
        )
        .unwrap();
    let mut expected = Vec::new();
    for &p in &indices[0] {
        for &q in &indices[1] {
            for &r in &indices[2] {
                for &s in &indices[3] {
                    expected.push(mo.get(p, q, r, s));
                }
            }
        }
    }
    for (&v, &r) in block.tensor.values.iter().zip(&expected) {
        close(v, r, 1e-12);
    }
    assert!(!block.orbital_response_included);
    let custom = molecule
        .integrals(
            &IntegralRequest {
                operator: "eri".into(),
                representation: "custom".into(),
                indices: block.indices.clone(),
                coefficients: vec![hf.mo_coefficients.clone()],
                ..Default::default()
            },
            settings,
        )
        .unwrap();
    for (&v, &r) in custom.tensor.values.iter().zip(&block.tensor.values) {
        close(v, r, 1e-12);
    }
    let fock = molecule
        .integrals(
            &IntegralRequest {
                operator: "fock".into(),
                representation: "mo_alpha".into(),
                ..Default::default()
            },
            settings,
        )
        .unwrap();
    for i in 0..2 {
        for j in 0..2 {
            close(
                fock.tensor.values[2 * i + j],
                if i == j { hf.orbital_energies[i] } else { 0. },
                1e-10,
            );
        }
    }
    assert!(Arc::ptr_eq(&reference, &molecule.scf(settings).unwrap()));
    for request in [
        IntegralRequest {
            operator: "eri".into(),
            max_elements: 15,
            ..Default::default()
        },
        IntegralRequest {
            workspace_budget_bytes: 1,
            ..Default::default()
        },
        IntegralRequest {
            indices: vec![vec![2], vec![0]],
            ..Default::default()
        },
        IntegralRequest {
            indices: vec![vec![], vec![0]],
            ..Default::default()
        },
        IntegralRequest {
            operator: "fock".into(),
            derivative_coordinates: vec![0],
            ..Default::default()
        },
    ] {
        assert!(
            molecule.integrals(&request, settings).is_err(),
            "{request:?}"
        );
    }
}

#[test]
fn physical_spherical_integrals_and_fixed_mo_derivatives_keep_their_frames() {
    let molecule = Molecule::new(
        &[8, 1, 1],
        &[[0.; 3], [0., -1.4, 1.1], [0., 1.4, 1.1]],
        MoleculeOptions {
            basis: "6-31g*".into(),
            cartesian: false,
            ..Default::default()
        },
    )
    .unwrap();
    let transform = hf_rs::ao::AoSpace::new(molecule.basis(), hf_rs::ao::BasisKind::Spherical)
        .unwrap()
        .cart_to_ao;
    assert!(transform.cols < transform.rows);
    for operator in ["overlap", "core_hamiltonian"] {
        let raw = molecule
            .integrals(
                &api::IntegralRequest {
                    operator: operator.into(),
                    ..Default::default()
                },
                ScfOptions::default(),
            )
            .unwrap()
            .tensor;
        let cartesian = hf_rs::Matrix::from_vec(raw.shape[0], raw.shape[1], raw.values);
        let expected = transform.transpose().dot(&cartesian).dot(&transform);
        let actual = molecule
            .integrals(
                &api::IntegralRequest {
                    operator: operator.into(),
                    representation: "physical_ao".into(),
                    ..Default::default()
                },
                ScfOptions::default(),
            )
            .unwrap()
            .tensor;
        assert_eq!(actual.shape, vec![transform.cols; 2]);
        for (&a, &b) in actual.values.iter().zip(&expected.data) {
            close(a, b, 1e-11);
        }
    }
    let molecule = h2();
    let reference = molecule.scf(ScfOptions::default()).unwrap();
    let api::ScfReference::Rhf(hf) = &*reference else {
        panic!("RHF required")
    };
    let fixed_coefficients = hf.mo_coefficients.clone();
    let request = api::IntegralRequest {
        operator: "eri".into(),
        representation: "mo_alpha".into(),
        indices: vec![vec![0], vec![1], vec![0], vec![1]],
        derivative_coordinates: vec![2, 5, 2, 5],
        ..Default::default()
    };
    let actual = molecule.integrals(&request, ScfOptions::default()).unwrap();
    assert!(!actual.orbital_response_included);
    let step = 1e-3;
    let mut expected = 0.;
    for (shift, weight) in [(-2., 1.), (-1., -8.), (1., 8.), (2., -1.)] {
        let displaced = Molecule::new(
            &[1, 1],
            &[[0., 0., -0.7], [0., 0., 0.7 + shift * step]],
            MoleculeOptions::default(),
        )
        .unwrap();
        let value = displaced
            .integrals(
                &api::IntegralRequest {
                    representation: "custom".into(),
                    coefficients: vec![fixed_coefficients.clone()],
                    derivative_coordinates: vec![2, 5, 2],
                    ..request.clone()
                },
                ScfOptions::default(),
            )
            .unwrap()
            .tensor
            .values[0];
        expected += weight * value / (12. * step);
    }
    close(actual.tensor.values[0], expected, 1e-8);
}

#[test]
fn mixed_spin_integral_blocks_and_ri_factors_preserve_their_frames() {
    use hf_rs::api::IntegralRequest;
    let molecule = Molecule::new(
        &[3, 1],
        &[[0.; 3], [0.2, 0.1, 3.]],
        MoleculeOptions {
            charge: 1,
            multiplicity: 2,
            reference: ReferenceChoice::Uhf,
            ..Default::default()
        },
    )
    .unwrap();
    let reference = molecule.scf(ScfOptions::default()).unwrap();
    let api::ScfReference::Uhf(hf) = &*reference else {
        panic!("UHF required")
    };
    let context =
        hf_rs::context::PostHFContext::from_uhf(hf, molecule.atoms(), molecule.basis(), None)
            .unwrap();
    let cross = context.ints.cross().unwrap();
    let indices = vec![vec![0, 2], vec![1], vec![0, 1], vec![2]];
    let block = molecule
        .integrals(
            &IntegralRequest {
                operator: "eri".into(),
                representation: "mo_alpha_beta".into(),
                indices,
                ..Default::default()
            },
            ScfOptions::default(),
        )
        .unwrap();
    let mut offset = 0;
    for &p in &block.indices[0] {
        for &q in &block.indices[1] {
            for &r in &block.indices[2] {
                for &s in &block.indices[3] {
                    close(block.tensor.values[offset], cross.get(p, q, r, s), 1e-11);
                    offset += 1;
                }
            }
        }
    }
    let molecule = h2();
    let request = IntegralRequest {
        operator: "df_factors".into(),
        aux_basis: Some("def2-svp-ri".into()),
        ..Default::default()
    };
    let block = molecule.integrals(&request, ScfOptions::default()).unwrap();
    let auxiliary =
        hf_rs::context::AuxBasis::from_builtin(molecule.atoms(), "def2-svp-ri").unwrap();
    let factors =
        hf_rs::df::DfTensor::build(molecule.basis(), &auxiliary.functions, 1e-10).unwrap();
    assert_eq!(block.tensor.unit, "hartree^0.5");
    for p in 0..2 {
        for q in 0..2 {
            for r in 0..2 {
                for s in 0..2 {
                    let value = (0..block.tensor.shape[0])
                        .map(|a| {
                            block.tensor.values[(a * 2 + p) * 2 + q]
                                * block.tensor.values[(a * 2 + r) * 2 + s]
                        })
                        .sum::<f64>();
                    close(value, factors.eri(p, q, r, s), 1e-11);
                }
            }
        }
    }
}

#[test]
fn bare_integral_derivatives_through_fourth_match_lower_order_differences() {
    use hf_rs::api::IntegralRequest;
    let positions = [[0.1, -0.2, -0.7], [-0.3, 0.4, 0.9]];
    let molecule = Molecule::new(&[1, 1], &positions, MoleculeOptions::default()).unwrap();
    let mut prefix = Vec::new();
    for coordinate in [0, 1, 2, 3] {
        let mut derivative = prefix.clone();
        derivative.push(coordinate);
        for operator in [
            "overlap",
            "kinetic",
            "nuclear_attraction",
            "core_hamiltonian",
            "eri",
            "df_metric",
            "df_three_center",
        ] {
            let arity = if operator == "eri" { 4 } else { 2 };
            let indices = if operator == "df_three_center" {
                vec![vec![0], vec![0], vec![1]]
            } else if arity == 4 {
                vec![vec![0], vec![1], vec![0], vec![1]]
            } else {
                vec![vec![0], vec![1]]
            };
            let value = molecule
                .integrals(
                    &IntegralRequest {
                        operator: operator.into(),
                        indices: indices.clone(),
                        derivative_coordinates: derivative.clone(),
                        aux_basis: operator.starts_with("df_").then(|| "def2-svp-ri".into()),
                        ..Default::default()
                    },
                    ScfOptions::default(),
                )
                .unwrap()
                .tensor
                .values[0];
            let mut expected = 0.;
            let step = 1e-3;
            for (shift, weight) in [(-2., 1.), (-1., -8.), (1., 8.), (2., -1.)] {
                let mut moved = positions;
                moved[coordinate / 3][coordinate % 3] += shift * step;
                let displaced = Molecule::new(&[1, 1], &moved, MoleculeOptions::default()).unwrap();
                expected += weight
                    * displaced
                        .integrals(
                            &IntegralRequest {
                                operator: operator.into(),
                                indices: indices.clone(),
                                derivative_coordinates: prefix.clone(),
                                aux_basis: operator
                                    .starts_with("df_")
                                    .then(|| "def2-svp-ri".into()),
                                ..Default::default()
                            },
                            ScfOptions::default(),
                        )
                        .unwrap()
                        .tensor
                        .values[0]
                    / (12. * step);
            }
            close(value, expected, 1e-7);
        }
        prefix = derivative;
    }
}

#[test]
fn hf_observables_are_explicit_typed_and_in_atomic_units() {
    use hf_rs::output::HfObservables;
    let molecule = h2();
    assert!(
        molecule
            .run("hf", &OptionMap::new(), &[])
            .unwrap()
            .observables
            .is_empty()
    );
    let values = molecule.hf_observables("hf", &OptionMap::new()).unwrap();
    let hf = molecule.scf(ScfOptions::default()).unwrap();
    let api::ScfReference::Rhf(hf) = &*hf else {
        panic!("RHF required")
    };
    assert_eq!(
        serde_json::to_value(&values).unwrap(),
        serde_json::to_value(hf.observables().unwrap()).unwrap()
    );
    close(
        values["homo_lumo_gap"].value.as_f64().unwrap(),
        hf.homo_lumo_gap().unwrap(),
        1e-14,
    );
    assert_eq!(values["dipole"].unit, "e*bohr");
    assert_eq!(values["dipole"].length_power, 1.);
    assert_eq!(values["n_electrons"].value, json!(2));
    let mut bad = hf.clone();
    bad.orbital_energies[0] = f64::NAN;
    assert!(bad.observables().is_err());
    let hydrogen = Molecule::new(
        &[1],
        &[[0.; 3]],
        MoleculeOptions {
            multiplicity: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let spin = hydrogen.hf_observables("uhf", &OptionMap::new()).unwrap();
    close(spin["s_squared"].value.as_f64().unwrap(), 0.75, 1e-12);
    assert!(spin["homo_lumo_gap_beta"].value.is_null());
    assert!(molecule.hf_observables("ccsd", &OptionMap::new()).is_err());
}

#[test]
fn molecular_variant_observables_preserve_default_model_energies() {
    use hf_rs::HFVariantType as V;
    let molecule = h2();
    for (name, variant, key) in [
        ("ghf", V::GHF, "s_squared"),
        ("puhf", V::PUHF, "target_spin_weight"),
        ("hfb", V::HFB, "pairing_gap"),
        ("tcscf", V::TCSCF, "natural_occupations"),
        ("qed-hf", V::QEDHF, "mean_photon_number"),
        ("cd-hf", V::CDHF, "cholesky_rank"),
        ("lhf", V::LHF, "exchange_energy"),
        ("oep", V::OEP, "ks_hamiltonian"),
        ("complex-hf", V::ComplexHF, "imaginary_density_norm"),
        ("x2c", V::RelativisticHF, "relativistic_shift"),
        ("dc-hf", V::DCHF, "chemical_potential"),
    ] {
        let r = molecule
            .run(name, &OptionMap::new(), &["observables".into()])
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(r.observables.contains_key(key), "{name}: {key}");
        let legacy = hf_rs::evaluate_hf_variant_energy(
            molecule.atoms(),
            molecule.basis_map(),
            variant,
            0,
            100,
            1e-12,
        )
        .unwrap();
        close(r.output.e_total, legacy, 1e-11);
    }
}

#[test]
fn specialized_hf_observables_cover_excitations_ensembles_and_radial_models() {
    let molecule = h2();
    let config = json!({"alpha":[0],"beta":[0]});
    for (name, options, key) in [
        ("tdhf", OptionMap::new(), "oscillator_strengths"),
        ("qrhf", OptionMap::new(), "s_squared"),
        (
            "mom",
            OptionMap::from([("occupied_orbitals".into(), json!([0]))]),
            "homo_energy",
        ),
        (
            "fermi",
            OptionMap::from([("thermal_energy".into(), json!(0.01))]),
            "entropy_over_kb",
        ),
        (
            "sa-hf",
            OptionMap::from([
                ("configurations".into(), json!([config.clone()])),
                ("weights".into(), json!([1.])),
            ]),
            "state_energies",
        ),
        (
            "sigma-scf",
            OptionMap::from([
                ("configuration".into(), config),
                ("target_energy".into(), json!(-1.1167143251757503)),
            ]),
            "energy_variance",
        ),
    ] {
        let r = molecule
            .run(name, &options, &["observables".into()])
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(r.observables.contains_key(key), "{name}");
        r.output.validate().unwrap();
        if name == "tdhf" {
            assert!(r.output.e_total > r.output.e_scf);
        }
    }
    let atom = Molecule::new(&[2], &[[0.; 3]], MoleculeOptions::default()).unwrap();
    let radial = atom
        .hf_observables(
            "radial-hf",
            &OptionMap::from([("n_grid_points".into(), json!(256))]),
        )
        .unwrap();
    assert_eq!(radial["radial_wavefunction"].unit, "bohr^-0.5");
    let grid = radial["radial_grid"].value.as_array().unwrap();
    let u = radial["radial_wavefunction"].value.as_array().unwrap();
    let dr = grid[1].as_f64().unwrap() - grid[0].as_f64().unwrap();
    close(
        u.iter().map(|v| v.as_f64().unwrap().powi(2) * dr).sum(),
        1.,
        1e-10,
    );
    let crystal = atom
        .hf_observables(
            "crystal-hf",
            &OptionMap::from([
                ("lattice_constant".into(), json!(8.)),
                ("n_kpoints".into(), json!(4)),
            ]),
        )
        .unwrap();
    assert_eq!(crystal["k_cartesian"].unit, "bohr^-1");
    assert!(crystal["conduction_band_minimum"].value.is_null());
    assert!(molecule.hf_observables("mom", &OptionMap::new()).is_err());
    assert!(
        molecule
            .run("tdhf", &OptionMap::new(), &["gradient".into()])
            .is_err()
    );
}

#[test]
fn particle_and_fragment_observables_keep_their_computed_decomposition() {
    let molecule = h2();
    for (name, key) in [
        ("neo-hf", "protonic_kinetic_energy"),
        ("positron", "positron_orbital_energy"),
        ("ri-hf", "n_auxiliary"),
    ] {
        let r = molecule
            .hf_observables(name, &OptionMap::new())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(r.contains_key(key), "{name}");
    }
    for name in ["fmo2", "fmo3"] {
        let molecule = Molecule::new(
            &[2, 2, 2],
            &[[0., 0., 0.], [0., 0., 5.], [4., 0., 0.]],
            MoleculeOptions::default(),
        )
        .unwrap();
        let calculation = molecule
            .run(name, &OptionMap::new(), &["observables".into()])
            .unwrap();
        let r = &calculation.observables;
        let total = r["pair_interaction_energies"].value[0].as_f64().unwrap();
        let parts = [
            "pair_electrostatic",
            "pair_exchange_repulsion",
            "pair_charge_transfer_and_mixing",
        ];
        close(
            total,
            parts
                .iter()
                .map(|key| r[*key].value[0].as_f64().unwrap())
                .sum(),
            1e-12,
        );
        assert_eq!(r["pair_interaction_energies"].unit, "hartree");
        if name == "fmo3" {
            assert_eq!(r["trimer_indices"].value, json!([[0, 1, 2]]));
            let sum = |key: &str| {
                r[key]
                    .value
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .sum::<f64>()
            };
            close(
                calculation.output.e_total,
                sum("monomer_energies")
                    + sum("pair_interaction_energies")
                    + sum("trimer_corrections"),
                1e-12,
            );
        }
    }
}

#[test]
fn rust_units_and_scf_cache_are_shared_across_methods() {
    let molecule = h2();
    let a = molecule.scf(ScfOptions::default()).unwrap();
    let result = molecule.run("MP2", &OptionMap::new(), &[]).unwrap();
    let b = molecule.scf(ScfOptions::default()).unwrap();
    assert!(Arc::ptr_eq(&a, &b));
    close(result.output.e_scf, -1.1167143251757503, 1e-9);
    let positions = [
        [0., 0., -0.7 * hf_rs::BOHR_TO_ANGSTROM],
        [0., 0., 0.7 * hf_rs::BOHR_TO_ANGSTROM],
    ];
    let converted = api::run(
        "mp2",
        &[1, 1],
        &positions,
        MoleculeOptions {
            unit: LengthUnit::Angstrom,
            ..Default::default()
        },
        &OptionMap::new(),
        &[],
    )
    .unwrap();
    close(converted.output.e_total, result.output.e_total, 1e-12);
    let strict = molecule
        .scf(ScfOptions {
            energy_tol: 1e-13,
            residual_tol: 1e-12,
            density_tol: 1e-12,
            ..Default::default()
        })
        .unwrap();
    assert!(!Arc::ptr_eq(&a, &strict));
}

#[test]
fn every_hf_gradient_and_hessian_component_matches_public_oracle() {
    let data: Value = serde_json::from_str(include_str!("data/scf_reference.json")).unwrap();
    let expected = &data["systems"]["h2"]["results"]["sto-3g"];
    let result = h2()
        .run(
            "hf",
            &OptionMap::new(),
            &["gradient".into(), "hessian".into()],
        )
        .unwrap();
    close(
        result.output.e_total,
        expected["energy"].as_f64().unwrap(),
        1e-9,
    );
    for (i, row) in result.gradient.unwrap().iter().enumerate() {
        for (k, &value) in row.iter().enumerate() {
            close(value, expected["gradient"][i][k].as_f64().unwrap(), 1e-6);
        }
    }
    let h = result.hessian.unwrap();
    for i in 0..6 {
        for j in 0..6 {
            close(h[(i, j)], expected["hessian"][i][j].as_f64().unwrap(), 1e-5);
        }
    }
}

#[test]
fn open_shell_reference_and_higher_hf_response_are_preserved() {
    let molecule = Molecule::new(
        &[1, 1],
        &[[0., 0., -0.7], [0., 0., 0.7]],
        MoleculeOptions {
            charge: 1,
            multiplicity: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let result = molecule
        .run("hf", &OptionMap::new(), &["gradient".into()])
        .unwrap();
    assert_eq!(result.reference, RefKind::Uhf);
    close(result.output.components["S_squared"], 0.75, 1e-12);
    let z = [0., 0., 1., 0., 0., 0.].to_vec();
    for order in 1..=4 {
        let value = molecule
            .nuclear_derivative(&vec![2; order], ScfOptions::default())
            .unwrap();
        let directional = molecule
            .directional_derivative(&vec![z.clone(); order], ScfOptions::default())
            .unwrap();
        close(value, directional, 1e-10);
    }
}

#[test]
fn selected_state_and_options_survive_the_common_boundary() {
    let values = OptionMap::from([
        ("active_n_alpha".into(), json!(1)),
        ("active_n_beta".into(), json!(1)),
        ("active_n_orb".into(), json!(2)),
        ("n_roots".into(), json!(2)),
        ("root".into(), json!(1)),
    ]);
    let result = h2().run("fci", &values, &[]).unwrap();
    assert_eq!(result.output.root, 1);
    close(result.output.e_total, result.output.states[1].energy, 1e-12);
    assert!(result.output.e_total > result.output.states[0].energy);
    assert!(result.output.components.contains_key("E_CASCI"));
    let json = result.to_json();
    assert_eq!(json["root"], 1);
    assert_eq!(json["states"].as_array().unwrap().len(), 2);
}

#[test]
fn invalid_inputs_and_unavailable_analytic_routes_return_errors() {
    let molecule = h2();
    for (method, values, property) in [
        ("unknown", OptionMap::new(), "energy"),
        (
            "mp2",
            OptionMap::from([("unknown".into(), json!(1))]),
            "energy",
        ),
        (
            "mp2",
            OptionMap::from([("frozen_core".into(), json!(1))]),
            "gradient",
        ),
        ("ccsd", OptionMap::new(), "gradient"),
        ("hf", OptionMap::new(), "unknown"),
        (
            "hf",
            OptionMap::from([("scf_max_iter".into(), json!(0))]),
            "energy",
        ),
    ] {
        assert!(
            molecule.run(method, &values, &[property.into()]).is_err(),
            "{method} {property}"
        );
    }
    assert!(Molecule::new(&[1, 1], &[[0.; 3]; 2], MoleculeOptions::default()).is_err());
    assert!(Molecule::new(&[0], &[[0.; 3]], MoleculeOptions::default()).is_err());
    assert!(
        Molecule::new(
            &[1],
            &[[0.; 3]],
            MoleculeOptions {
                multiplicity: 2,
                reference: ReferenceChoice::Rhf,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn zero_cavity_coupling_recovers_all_hf_energy_gradient_components() {
    let molecule = h2();
    let expected = molecule
        .run("hf", &OptionMap::new(), &["gradient".into()])
        .unwrap();
    let result = molecule
        .hf_variant(
            "qed",
            &OptionMap::from([("coupling".into(), json!([0., 0., 0.]))]),
            &["gradient".into()],
        )
        .unwrap();
    close(result.output.e_total, expected.output.e_total, 1e-10);
    for (a, b) in result
        .gradient
        .unwrap()
        .iter()
        .flatten()
        .zip(expected.gradient.unwrap().iter().flatten())
    {
        close(*a, *b, 1e-8);
    }
    assert!(
        molecule
            .hf_variant("missing", &OptionMap::new(), &[])
            .is_err()
    );
    assert!(
        molecule
            .hf_variant("qed", &OptionMap::from([("typo".into(), json!(1))]), &[])
            .is_err()
    );
}

#[test]
fn cli_uses_embedded_independent_data_and_rejects_misspelled_controls() {
    assert_eq!(
        hf_rs::HFVariantType::parse("fmo3").unwrap(),
        hf_rs::HFVariantType::FMO3
    );
    assert!(hf_rs::HFVariantType::parse("unknown").is_err());
    let value = hf_rs::cli::execute(&["--oracle-compare".into()]).unwrap();
    assert_eq!(value["passed"], true, "{value}");
    assert_eq!(value["cases"].as_array().unwrap().len(), 4);
    assert!(hf_rs::cli::execute(&["--opt".into(), "typo=1".into()]).is_err());
    assert!(
        hf_rs::cli::execute(&[
            "--opt".into(),
            "scf_max_iter=100".into(),
            "--opt".into(),
            "scf_max_iter=200".into()
        ])
        .is_err()
    );
}

mod exports {
    // SPDX-License-Identifier: Apache-2.0
    // Copyright (c) 2026 ss0832
    //
    // This code is part of the hf-rs library.

    //! Check exported orbital frames, spin occupations, ECPs and cache identity.
    use hf_rs::{
        Matrix,
        api::{Molecule, MoleculeOptions, ReferenceChoice, ScfOptions, ScfReference},
    };
    use serde_json::Value;
    use std::sync::Arc;

    fn matrix(value: &Value) -> Matrix {
        let data = value.as_array().unwrap();
        let mut matrix = Matrix::zeros(data.len(), data[0].as_array().unwrap().len());
        for (i, row) in data.iter().enumerate() {
            for (j, entry) in row.as_array().unwrap().iter().enumerate() {
                matrix[(i, j)] = entry.as_f64().unwrap();
            }
        }
        matrix
    }

    fn check(molecule: &Molecule) -> Value {
        let options = ScfOptions::default();
        let before = molecule.scf(options).unwrap();
        let snapshot = molecule.scf_wavefunction(options).unwrap();
        let overlap = matrix(&snapshot["overlap_matrix"]);
        let mut electrons = 0.;
        for channel in snapshot["channels"].as_array().unwrap() {
            let c = matrix(&channel["coefficients"]);
            let density = matrix(&channel["density_matrix"]);
            let metric = c.transpose().dot(&overlap).dot(&c);
            let occupations: Vec<_> = channel["occupations"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            for i in 0..c.cols {
                for j in 0..c.cols {
                    assert!((metric[(i, j)] - if i == j { 1. } else { 0. }).abs() < 1e-10);
                }
            }
            for i in 0..c.rows {
                for j in 0..c.rows {
                    let reconstructed: f64 = (0..c.cols)
                        .map(|k| c[(i, k)] * occupations[k] * c[(j, k)])
                        .sum();
                    assert!((reconstructed - density[(i, j)]).abs() < 1e-9);
                    electrons += density[(i, j)] * overlap[(j, i)];
                }
            }
        }
        let count = snapshot["n_alpha"].as_u64().unwrap() + snapshot["n_beta"].as_u64().unwrap();
        assert!((electrons - count as f64).abs() < 1e-9);
        assert_eq!(
            snapshot["basis_functions"].as_array().unwrap().len(),
            molecule.basis().len()
        );
        assert!(snapshot.get("eri").is_none());
        let text = molecule.scf_molden(options).unwrap();
        let expected = match &*before {
            ScfReference::Rhf(hf) => hf_rs::molden::try_generate_molden_string_rhf(
                molecule.atoms(),
                molecule.basis_map(),
                hf,
            )
            .unwrap(),
            ScfReference::Uhf(hf) => hf_rs::molden::try_generate_molden_string_uhf(
                molecule.atoms(),
                molecule.basis_map(),
                hf,
            )
            .unwrap(),
        };
        assert_eq!(text, expected);
        assert!(Arc::ptr_eq(&before, &molecule.scf(options).unwrap()));
        snapshot
    }

    #[test]
    fn cartesian_spherical_and_unrestricted_exports_preserve_orbital_frames() {
        for cartesian in [true, false] {
            let m = Molecule::new(
                &[8, 1, 1],
                &[[0., 0., 0.], [0., -1.4, 1.1], [0., 1.4, 1.1]],
                MoleculeOptions {
                    basis: "6-31g*".into(),
                    cartesian,
                    ..Default::default()
                },
            )
            .unwrap();
            let snapshot = check(&m);
            assert_eq!(
                snapshot["ao_basis_kind"],
                if cartesian { "cartesian" } else { "spherical" }
            );
        }
        let m = Molecule::new(
            &[1],
            &[[0., 0., 0.]],
            MoleculeOptions {
                multiplicity: 2,
                reference: ReferenceChoice::Uhf,
                ..Default::default()
            },
        )
        .unwrap();
        let snapshot = check(&m);
        assert_eq!(snapshot["channels"][0]["spin"], "alpha");
        assert_eq!(snapshot["channels"][1]["spin"], "beta");
        assert_eq!(snapshot["channels"][1]["occupations"][0], 0.);
    }

    #[test]
    fn mixed_basis_ecp_exports_retain_effective_charges_and_potential_terms() {
        let mut config = MoleculeOptions::default();
        config
            .set_basis(&serde_json::json!({"Na":"lanl2dz","H":"sto-3g"}))
            .unwrap();
        let m = Molecule::new(&[11, 1], &[[0., 0., 0.], [0.2, 0.1, 3.1]], config).unwrap();
        let snapshot = check(&m);
        assert_eq!(snapshot["atoms"][0]["effective_nuclear_charge"], 1.);
        assert_eq!(snapshot["atoms"][0]["ecp"]["n_core"], 10);
        assert_eq!(
            snapshot["atoms"][0]["ecp"]["local"]
                .as_array()
                .unwrap()
                .len(),
            m.atoms()[0].ecp.as_ref().unwrap().local.len()
        );
        assert!(
            m.scf_molden(ScfOptions::default())
                .unwrap()
                .contains("[core]\n1 : 10")
        );
    }
}
