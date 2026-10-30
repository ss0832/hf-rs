// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    context::PostHFContext,
    options::PostHFOptions,
    post_hf::{PostHFMethod, PostHFResult},
    registry::{MethodId, create_solver},
    *,
};
fn number(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap()
}
fn vector(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(number).collect()
}
fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter().flat_map(vector).collect(),
    )
}
fn atoms(case: &serde_json::Value) -> Vec<Atom> {
    case["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            Atom::new(
                a[0].as_str().unwrap(),
                std::array::from_fn(|k| number(&a[1][k])),
            )
        })
        .collect()
}
fn close(x: f64, y: f64) {
    assert!((x - y).abs() < 1e-9, "{x:0.15} vs {y:0.15}: {:e}", x - y);
}

#[test]
fn scs_mp3_rhf_uhf_and_scaling_options_match_psi4() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/scs_mp3_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let atoms = atoms(case);
        let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
        let rhf;
        let uhf;
        let context = if case["multiplicity"].as_u64().unwrap() == 1 {
            rhf = run_scf(&atoms, &basis, 200, 1e-14).unwrap();
            PostHFContext::from_rhf(&rhf, &atoms, &basis, None).unwrap()
        } else {
            uhf = run_uhf(
                &atoms,
                &basis,
                0,
                case["multiplicity"].as_u64().unwrap() as usize,
                false,
                200,
                1e-14,
            )
            .unwrap();
            PostHFContext::from_uhf(&uhf, &atoms, &basis, None).unwrap()
        };
        let output = create_solver(MethodId::ScsMp3, &context, &PostHFOptions::default())
            .unwrap()
            .compute()
            .unwrap();
        close(output.e_scf, number(&case["e_scf"]));
        close(output.e_total, number(&case["e_scs_total"]));
        close(
            output.components["E_MP2_same_spin"],
            number(&case["e_same"]),
        );
        close(
            output.components["E_MP2_opposite_spin"],
            number(&case["e_opposite"]),
        );
        close(output.components["E_MP3_increment"], number(&case["e_mp3"]));
        for scale in [0., 1.] {
            let custom = PostHFOptions {
                scs_same_spin: 1.,
                scs_opposite_spin: 1.,
                scs_third_order: scale,
                ..Default::default()
            };
            let actual = create_solver(MethodId::ScsMp3, &context, &custom)
                .unwrap()
                .compute()
                .unwrap();
            close(
                actual.e_corr,
                number(&case["e_mp2"]) + scale * number(&case["e_mp3"]),
            );
        }
        let unsupported = PostHFOptions {
            density_fitting: true,
            ..Default::default()
        };
        assert!(create_solver(MethodId::ScsMp3, &context, &unsupported).is_err());
    }
    let map =
        std::collections::BTreeMap::from([("scs_third_order".into(), serde_json::json!(0.5))]);
    let parsed = PostHFOptions::from_map(&map, &MethodId::ScsMp3.option_schemas()).unwrap();
    close(parsed.scs_third_order, 0.5);
    assert!(PostHFOptions::from_map(&map, &MethodId::ScsMp2.option_schemas()).is_err());
}

#[test]
fn cis_every_root_amplitude_and_transition_property_match_pyscf_tda() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/cis_reference.json")).unwrap();
    let mut components = 0;
    for case in data["cases"].as_array().unwrap() {
        let atoms = atoms(case);
        let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
        let hf = run_scf(&atoms, &basis, 200, 1e-14).unwrap();
        let triplet = case["triplet"].as_bool().unwrap();
        let result = CIS::new(&hf)
            .with_basis(&basis)
            .with_roots(4)
            .with_root(2)
            .with_triplet(triplet)
            .compute()
            .unwrap();
        let alignment = hf
            .mo_coefficients
            .transpose()
            .dot(&hf.overlap_matrix)
            .dot(&matrix(&case["mo_coefficients"]));
        // MO rotations and degenerate CIS eigenvectors have arbitrary gauges.
        // Align complete equal-energy subspaces, not just individual signs.
        let (no, nv) = (result.n_occ, result.n_virt);
        let mut uo = Matrix::zeros(no, no);
        let mut uv = Matrix::zeros(nv, nv);
        for i in 0..no {
            for j in 0..no {
                uo[(i, j)] = alignment[(i, j)];
            }
        }
        for a in 0..nv {
            for b in 0..nv {
                uv[(a, b)] = alignment[(no + a, no + b)];
            }
        }
        let rotated: Vec<_> = result
            .excited_states
            .iter()
            .map(|state| {
                uo.transpose()
                    .dot(&Matrix::from_vec(no, nv, state.amplitudes.clone()))
                    .dot(&uv)
                    .data
            })
            .collect();
        for (root, state) in result.excited_states.iter().enumerate() {
            let omega = number(&case["energies"][root]);
            close(state.excitation_energy_hartree, omega);
            let expected = vector(&case["amplitudes"][root]);
            let mut aligned = vec![0.; no * nv];
            let mut dipole = [0.; 3];
            let mut weight = 0.;
            let mut oscillator_sum = 0.;
            let mut reference_oscillator_sum = 0.;
            for (k, other) in result.excited_states.iter().enumerate() {
                if (other.excitation_energy_hartree - omega).abs() > 1e-8 {
                    continue;
                }
                let overlap: f64 = rotated[k].iter().zip(&expected).map(|(a, b)| a * b).sum();
                weight += overlap * overlap;
                for (target, value) in aligned.iter_mut().zip(&rotated[k]) {
                    *target += overlap * value;
                }
                for (axis, value) in dipole.iter_mut().enumerate() {
                    *value += overlap * other.transition_dipole[axis];
                }
                oscillator_sum += other.oscillator_strength;
                reference_oscillator_sum += number(&case["oscillator_strengths"][k]);
            }
            close(weight, 1.);
            close(oscillator_sum, reference_oscillator_sum);
            for (x, y) in aligned.iter().zip(&expected) {
                close(*x, *y);
                components += 1;
            }
            close(
                (2. / 3.) * omega * dipole.iter().map(|x| x * x).sum::<f64>(),
                number(&case["oscillator_strengths"][root]),
            );
            for axis in 0..3 {
                close(
                    dipole[axis],
                    number(&case["transition_dipoles"][root][axis]),
                );
                if triplet {
                    assert_eq!(state.transition_dipole[axis], 0.);
                }
                components += 1;
            }
        }
        let context = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
        let options = PostHFOptions {
            n_roots: 4,
            root: 2,
            cis_triplet: triplet,
            ..Default::default()
        };
        let output = create_solver(MethodId::Cis, &context, &options)
            .unwrap()
            .compute()
            .unwrap();
        close(
            output.e_total,
            hf.total_energy + result.excited_states[2].excitation_energy_hartree,
        );
        close(result.total_energy(), output.e_total);
        assert_eq!(output.root, 2);
        assert_eq!(
            output.states[2].s_squared,
            Some(if triplet { 2. } else { 0. })
        );
        close(
            output.components["excitation_energy_hartree"],
            output.e_corr,
        );
        output.validate().unwrap();
        assert!(CIS::new(&hf).with_roots(0).compute().is_err());
        assert!(CIS::new(&hf).with_roots(1).with_root(1).compute().is_err());
    }
    println!("CIS: {components} amplitude/dipole components, all 16 singlet/triplet roots");
}
