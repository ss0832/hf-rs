// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    ao::BasisKind,
    basis::{Atom, build_basis, parse_gaussian94},
    ecp::{EcpDef, EcpTerm},
    error::HfError,
    post_hf::{
        f12::{CABSBasis, F12Approximation, MP2F12Options, RHFMP2F12, operators::GaussianGeminal},
        numerical::{NumericalGradientOptions, compute_rhf_numerical_gradient},
    },
    registry::GradientKind,
    scf::run_scf,
};

fn atoms(data: &serde_json::Value) -> Vec<Atom> {
    data["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| Atom::new("He", std::array::from_fn(|axis| p[axis].as_f64().unwrap())))
        .collect()
}

#[test]
fn all_numerical_3c_gradient_components_match_independent_psi4_with_frozen_core() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/psi4_f12_gradient_reference.json")).unwrap();
    let atoms = atoms(&data);
    let basis = build_basis(&atoms, &parse_gaussian94(data["obs_g94"].as_str().unwrap()));
    let aux = build_basis(&atoms, &parse_gaussian94(data["aux_g94"].as_str().unwrap()));
    let hf = run_scf(&atoms, &basis, 200, 1e-13).unwrap();
    let cabs = CABSBasis::from_reference(&hf, &basis, &aux, 1e-8).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let solver = RHFMP2F12::new(&hf, &atoms, &basis).with_options(MP2F12Options {
            frozen_core: case["frozen_core"].as_u64().unwrap() as usize,
            custom_cabs: Some(cabs.clone()),
            ..Default::default()
        });
        let mut coarse_error = 0.0;
        for step_bohr in [1e-3, 5e-4] {
            let mut max_error = 0.0_f64;
            let result = solver
                .numerical_gradient(&NumericalGradientOptions {
                    step_bohr,
                    ..Default::default()
                })
                .unwrap();
            assert_eq!(result.kind, GradientKind::Numerical);
            assert_eq!(result.step_bohr, step_bohr);
            assert_eq!(result.energy_evaluations, 13);
            for (a, g) in result.gradient.iter().enumerate() {
                for (axis, value) in g.iter().enumerate() {
                    let expected = case["gradient"][a][axis].as_f64().unwrap();
                    max_error = max_error.max((value - expected).abs());
                    assert!(
                        (value - expected).abs() < 1e-7,
                        "fc={} atom={a} axis={axis} h={step_bohr}: {value} vs {expected}",
                        case["frozen_core"]
                    );
                }
            }
            println!(
                "Psi4 all components: fc={} h={step_bohr} max_error={max_error:.6e}",
                case["frozen_core"]
            );
            if step_bohr == 1e-3 {
                coarse_error = max_error;
            } else {
                assert!(
                    max_error < 0.35 * coarse_error,
                    "central-difference error must fall with the squared step: {max_error} vs {coarse_error}"
                );
            }
        }
    }
}

#[test]
fn nonlocal_ecp_3a_all_gradient_components_converge_and_obey_rigid_motion_identities() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/f12_nonlocal_reference.json")).unwrap();
    let mut atoms = atoms(&data);
    for atom in &mut atoms {
        atom.ecp = Some(std::sync::Arc::new(EcpDef {
            symbol: "He".into(),
            n_core: 0,
            l_max: 2,
            local: vec![EcpTerm {
                power: 2,
                exponent: 0.7,
                coefficient: 0.17,
            }],
            semilocal: vec![
                vec![EcpTerm {
                    power: 2,
                    exponent: 0.53,
                    coefficient: 0.11,
                }],
                vec![EcpTerm {
                    power: 2,
                    exponent: 0.61,
                    coefficient: -0.07,
                }],
            ],
        }));
    }
    let basis = build_basis(
        &atoms,
        &parse_gaussian94("He 0\nS 1 1\n1.4 1\nS 1 1\n0.45 1\n****"),
    );
    let aux = build_basis(
        &atoms,
        &parse_gaussian94("He 0\nS 1 1\n0.2 1\nP 1 1\n0.8 1\n****"),
    );
    let hf = run_scf(&atoms, &basis, 200, 1e-13).unwrap();
    // A nondefault cutoff also checks preservation of the supplied CABS space.
    let cabs = CABSBasis::from_reference(&hf, &basis, &aux, 1e-6).unwrap();
    let solver = RHFMP2F12::new(&hf, &atoms, &basis).with_options(MP2F12Options {
        approximation: F12Approximation::ApproximationA,
        custom_cabs: Some(cabs),
        custom_geminal: Some(GaussianGeminal::new(vec![(0.3, -0.4), (1.2, -0.2)]).unwrap()),
        ..Default::default()
    });
    let coarse = solver
        .numerical_gradient(&NumericalGradientOptions::default())
        .unwrap();
    let fine = solver
        .numerical_gradient(&NumericalGradientOptions {
            step_bohr: 5e-4,
            ..Default::default()
        })
        .unwrap();
    let mut torque = [0.0; 3];
    let mut max_refinement = 0.0_f64;
    for (atom, nucleus) in atoms.iter().enumerate() {
        for (axis, torque_value) in torque.iter_mut().enumerate() {
            assert!((coarse.gradient[atom][axis] - fine.gradient[atom][axis]).abs() < 1e-7);
            max_refinement =
                max_refinement.max((coarse.gradient[atom][axis] - fine.gradient[atom][axis]).abs());
            assert!(fine.gradient[atom][axis].abs() > 1e-6);
            let b = (axis + 1) % 3;
            let c = (axis + 2) % 3;
            *torque_value += nucleus.position[b] * fine.gradient[atom][c]
                - nucleus.position[c] * fine.gradient[atom][b];
        }
    }
    for (axis, value) in torque.iter().enumerate() {
        assert!((fine.gradient[0][axis] + fine.gradient[1][axis]).abs() < 1e-8);
        assert!(value.abs() < 1e-7);
    }
    println!(
        "Nonlocal ECP A all components: max_refinement={max_refinement:.6e}, torque={torque:?}"
    );
}

#[test]
fn numerical_gradients_propagate_failed_energies_and_reject_invalid_steps() {
    let atoms = vec![Atom::new("He", [0.13, -0.21, 0.31])];
    let basis = build_basis(
        &atoms,
        &parse_gaussian94("He 0\nS 1 1\n1.4 1\nS 1 1\n0.45 1\n****"),
    );
    for step_bohr in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-300] {
        let result = compute_rhf_numerical_gradient(
            &atoms,
            &basis,
            0,
            BasisKind::Cartesian,
            &NumericalGradientOptions {
                step_bohr,
                ..Default::default()
            },
            |_, _, _| panic!("invalid step evaluated an energy"),
        );
        assert!(matches!(result, Err(HfError::InvalidInput(_))));
    }
    let mut calls = 0;
    let result = compute_rhf_numerical_gradient(
        &atoms,
        &basis,
        0,
        BasisKind::Cartesian,
        &NumericalGradientOptions::default(),
        |hf, _, _| {
            calls += 1;
            if calls == 2 {
                Err(HfError::PostHf {
                    method: "oracle".into(),
                    detail: "nonconverged amplitude".into(),
                })
            } else {
                Ok(hf.total_energy)
            }
        },
    );
    assert!(matches!(result, Err(HfError::PostHf { .. })));
    assert_eq!(calls, 2);
}
