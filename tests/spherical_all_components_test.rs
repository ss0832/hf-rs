// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Every Cartesian nuclear component, including rectangular spherical-AO response.
use hf_rs::ao::BasisKind;
use hf_rs::*;
use std::collections::HashMap;

fn tensor(ctx: &HfNuclearDerivativeContext<'_>, order: usize, dim: usize) -> Vec<f64> {
    let mut cache = HashMap::new();
    (0..dim.pow(order as u32))
        .map(|mut index| {
            let mut coordinates = vec![0; order];
            for x in coordinates.iter_mut().rev() {
                *x = index % dim;
                index /= dim;
            }
            coordinates.sort_unstable();
            *cache
                .entry(coordinates.clone())
                .or_insert_with(|| ctx.coordinate_derivative(&coordinates).unwrap())
        })
        .collect()
}

fn snapshot(atoms: &[Atom], map: &BasisSetMap, base: bool, open_shell: bool) -> Vec<Vec<f64>> {
    let basis = build_basis(atoms, map);
    if open_shell {
        let hf =
            scf::run_uhf_in_space(atoms, &basis, 0, 2, false, 300, 1e-15, BasisKind::Spherical)
                .unwrap();
        assert!(hf.converged);
        let ctx = HfNuclearDerivativeContext::from_uhf(atoms, &basis, &hf).unwrap();
        collect(
            &ctx,
            hf.total_energy,
            gradient::try_compute_uhf_gradient(atoms, &basis, &hf).unwrap(),
            base,
        )
    } else {
        let hf = scf::run_scf_in_space(
            atoms,
            &basis,
            1,
            250,
            1e-15,
            ConvAlgo::Cdiis,
            None,
            BasisKind::Spherical,
        )
        .unwrap();
        assert!(hf.converged);
        let ctx = HfNuclearDerivativeContext::from_rhf(atoms, &basis, &hf).unwrap();
        collect(
            &ctx,
            hf.total_energy,
            gradient::try_compute_rhf_gradient(atoms, &basis, &hf).unwrap(),
            base,
        )
    }
}

fn collect(
    ctx: &HfNuclearDerivativeContext<'_>,
    energy: f64,
    g: Vec<[f64; 3]>,
    base: bool,
) -> Vec<Vec<f64>> {
    let mut values = vec![vec![energy], g.into_iter().flatten().collect()];
    for order in 2..=if base { 4 } else { 3 } {
        values.push(tensor(ctx, order, 6));
    }
    if base {
        for (i, value) in values[1].iter().enumerate() {
            assert!((value - ctx.coordinate_derivative(&[i]).unwrap()).abs() < 1e-8);
        }
    }
    values
}

fn verify_spherical(ecp: bool, open_shell: bool) {
    let mut atoms = [
        Atom::new("He", [-0.1, 0.2, 0.3]),
        Atom::new("H", [0.4, -0.3, 1.8]),
    ];
    if ecp {
        let term = |exponent, coefficient| ecp::EcpTerm {
            power: 2,
            exponent,
            coefficient,
        };
        atoms[0].ecp = Some(std::sync::Arc::new(EcpDef {
            symbol: "He".into(),
            n_core: 0,
            l_max: 2,
            local: vec![term(0.6, -0.05)],
            semilocal: vec![vec![term(0.9, 0.1)], vec![term(0.7, -0.04)]],
        }));
    }
    let map = parse_gaussian94("He 0\nS 1 1\n1.4 1\nD 1 1\n0.6 1\n****\nH 0\nS 1 1\n1.1 1\n****");
    let base = snapshot(&atoms, &map, true, open_shell);
    let h = if open_shell { 0.00125 } else { 0.0025 };
    let mut worst = [0.0_f64; 4];
    let mut counts = [0; 4];
    for coordinate in 0..6 {
        let mut differences = (0..4)
            .map(|order| vec![0.0; 6usize.pow(order)])
            .collect::<Vec<_>>();
        for (offset, weight) in [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)] {
            let mut moved = atoms.clone();
            moved[coordinate / 3].position[coordinate % 3] += offset * h;
            let values = snapshot(&moved, &map, false, open_shell);
            for order in 0..4 {
                for (sum, &v) in differences[order].iter_mut().zip(&values[order]) {
                    *sum += weight * v / (12.0 * h);
                }
            }
        }
        for order in 0..4 {
            for (prefix, &reference) in differences[order].iter().enumerate() {
                worst[order] =
                    worst[order].max((base[order + 1][prefix * 6 + coordinate] - reference).abs());
                counts[order] += 1;
            }
        }
        eprintln!(
            "spherical all-component coordinate {}/6 complete",
            coordinate + 1
        );
    }
    for order in 0..4 {
        assert_eq!(counts[order], 6usize.pow(order as u32 + 1));
        eprintln!(
            "spherical ECP={ecp} UHF={open_shell} order {}: {} components, max error {:.8e}",
            order + 1,
            counts[order],
            worst[order]
        );
        assert!(worst[order] < [1e-6, 1e-5, 2e-7, 3e-6][order]);
    }
}

#[test]
fn spherical_rhf_all_nuclear_components_through_fourth() {
    verify_spherical(false, false);
}
#[test]
fn spherical_ecp_rhf_all_nuclear_components_through_fourth() {
    verify_spherical(true, false);
}
#[test]
fn spherical_ecp_uhf_all_nuclear_components_through_fourth() {
    verify_spherical(true, true);
}

fn c1_d_model(name: &str) -> serde_json::Value {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/spherical_reference.json")).unwrap();
    let model = data["c1_d_systems"][name].clone();
    assert!(model.is_object(), "missing C1 d-shell oracle {name}");
    model
}

fn c1_d_atoms(model: &serde_json::Value) -> Vec<Atom> {
    let mut atoms: Vec<_> = model["geometry_bohr"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, p)| {
            Atom::new(
                if i == 0 { "He" } else { "H" },
                std::array::from_fn(|k| p[k].as_f64().unwrap()),
            )
        })
        .collect();
    if model["ecp"].as_bool().unwrap() {
        let term = |exponent, coefficient| ecp::EcpTerm {
            power: 2,
            exponent,
            coefficient,
        };
        atoms[0].ecp = Some(std::sync::Arc::new(EcpDef {
            symbol: "He".into(),
            n_core: 0,
            l_max: 3,
            local: vec![term(0.6, -0.05)],
            semilocal: vec![
                vec![term(0.9, 0.10)],
                vec![term(0.7, -0.04)],
                vec![term(0.8, 0.08)],
            ],
        }));
    }
    assert_eq!(atoms.len(), 4);
    let vectors: Vec<[f64; 3]> = atoms[1..]
        .iter()
        .map(|a| std::array::from_fn(|k| a.position[k] - atoms[0].position[k]))
        .collect();
    let volume = vectors[0][0] * (vectors[1][1] * vectors[2][2] - vectors[1][2] * vectors[2][1])
        - vectors[0][1] * (vectors[1][0] * vectors[2][2] - vectors[1][2] * vectors[2][0])
        + vectors[0][2] * (vectors[1][0] * vectors[2][1] - vectors[1][1] * vectors[2][0]);
    assert!(volume.abs() > 0.1);
    let mut distances = Vec::new();
    for i in 0..4 {
        for j in 0..i {
            distances.push(
                (0..3)
                    .map(|k| (atoms[i].position[k] - atoms[j].position[k]).powi(2))
                    .sum::<f64>(),
            );
        }
    }
    distances.sort_by(f64::total_cmp);
    assert!(distances.windows(2).all(|x| (x[1] - x[0]).abs() > 1e-3));
    atoms
}

fn c1_d_snapshot(
    model: &serde_json::Value,
    atoms: &[Atom],
    max_order: usize,
    anchor: bool,
) -> Vec<Vec<f64>> {
    let map = parse_gaussian94(model["basis_gaussian94"].as_str().unwrap());
    // The other two nuclei have no AOs; all four still enter the Hamiltonian,
    // nuclear repulsion and all twelve nuclear-coordinate derivatives.
    let basis = build_basis(&atoms[..2], &map);
    assert_eq!(basis.len(), 8);
    assert_eq!(
        basis
            .iter()
            .filter(|b| b.l.iter().sum::<usize>() == 2)
            .count(),
        6
    );
    let kind = if model["cartesian"].as_bool().unwrap() {
        BasisKind::Cartesian
    } else {
        BasisKind::Spherical
    };
    let charge = model["charge"].as_i64().unwrap() as i32;
    let convergence = scf::RhfConvergence {
        energy: 1e-15,
        residual: 1e-13,
        density: 1e-13,
    };
    if model["multiplicity"] == 2 {
        let hf = scf::run_uhf_in_space_with_convergence(
            atoms,
            &basis,
            charge,
            2,
            false,
            400,
            convergence,
            kind,
        )
        .unwrap();
        assert!(hf.converged);
        assert_eq!(
            hf.mo_coefficients_alpha.cols,
            model["n_ao"].as_u64().unwrap() as usize
        );
        let ctx = HfNuclearDerivativeContext::from_uhf(atoms, &basis, &hf).unwrap();
        c1_d_collect(
            &ctx,
            model,
            &basis,
            hf.total_energy,
            gradient::try_compute_uhf_gradient(atoms, &basis, &hf).unwrap(),
            &hf.density_matrix_alpha.add(&hf.density_matrix_beta),
            &hf.overlap_matrix,
            max_order,
            anchor,
        )
    } else {
        let hf = scf::run_scf_in_space_with_convergence(
            atoms,
            &basis,
            charge,
            400,
            convergence,
            ConvAlgo::Cdiis,
            None,
            kind,
        )
        .unwrap();
        assert!(hf.converged);
        assert_eq!(
            hf.mo_coefficients.cols,
            model["n_ao"].as_u64().unwrap() as usize
        );
        let ctx = HfNuclearDerivativeContext::from_rhf(atoms, &basis, &hf).unwrap();
        c1_d_collect(
            &ctx,
            model,
            &basis,
            hf.total_energy,
            gradient::try_compute_rhf_gradient(atoms, &basis, &hf).unwrap(),
            &hf.density_matrix,
            &hf.overlap_matrix,
            max_order,
            anchor,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn c1_d_collect(
    ctx: &HfNuclearDerivativeContext<'_>,
    model: &serde_json::Value,
    basis: &[ContractedGaussian],
    energy: f64,
    gradient: Vec<[f64; 3]>,
    density: &Matrix,
    overlap: &Matrix,
    max_order: usize,
    anchor: bool,
) -> Vec<Vec<f64>> {
    let mut values = vec![vec![energy], gradient.into_iter().flatten().collect()];
    values.push(ctx.hessian().unwrap().data);
    if anchor {
        assert!((energy - model["energy"].as_f64().unwrap()).abs() < 1e-8);
        for (order, key, tolerance) in [(1, "gradient", 1e-6), (2, "hessian", 1e-5)] {
            let expected: Vec<f64> = model[key]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap())
                .map(|x| x.as_f64().unwrap())
                .collect();
            assert_eq!(expected.len(), 12usize.pow(order as u32));
            for (i, (&v, &r)) in values[order].iter().zip(&expected).enumerate() {
                assert!(
                    (v - r).abs() < tolerance,
                    "C1 d PySCF {key} {i}: {v} vs {r}"
                );
            }
        }
        let t = ao::spherical_transform(basis).unwrap();
        let mut d_orbitals = Matrix::zeros(basis.len(), 5);
        for i in 0..basis.len() {
            for j in 0..5 {
                d_orbitals[(i, j)] = t[(i, j + 1)];
            }
        }
        let projected = d_orbitals
            .transpose()
            .dot(overlap)
            .dot(density)
            .dot(overlap)
            .dot(&d_orbitals);
        let weight = (0..5).map(|i| projected[(i, i)]).sum::<f64>();
        assert!(
            weight > 1e-3,
            "the d shell must have nontrivial occupied projection"
        );
        assert!(
            (weight - model["d_projection_weight"].as_f64().unwrap()).abs() < 1e-8,
            "C1 d projection {weight}"
        );
        eprintln!(
            "C1 d oracle anchors pass: Cartesian={}, ECP={}, multiplicity={}, d projection={weight:.8}",
            model["cartesian"], model["ecp"], model["multiplicity"]
        );
    }
    for order in 3..=max_order {
        values.push(tensor(ctx, order, 12));
    }
    values
}

fn verify_c1_d(name: &str) {
    let model = c1_d_model(name);
    let atoms = c1_d_atoms(&model);
    let base = c1_d_snapshot(&model, &atoms, 4, true);
    let step = 1.25e-3;
    let mut worst = [0.0_f64; 4];
    let mut locations = [0usize; 4];
    let mut counts = [0usize; 4];
    for coordinate in 0..12 {
        let mut differences = (0..4)
            .map(|order| vec![0.; 12usize.pow(order)])
            .collect::<Vec<_>>();
        for (offset, weight) in [(-2., 1.), (-1., -8.), (1., 8.), (2., -1.)] {
            let mut moved = atoms.clone();
            moved[coordinate / 3].position[coordinate % 3] += offset * step;
            let values = c1_d_snapshot(&model, &moved, 3, false);
            for order in 0..4 {
                for (sum, &v) in differences[order].iter_mut().zip(&values[order]) {
                    *sum += weight * v / (12. * step);
                }
            }
        }
        for order in 0..4 {
            for (prefix, &reference) in differences[order].iter().enumerate() {
                let index = prefix * 12 + coordinate;
                let error = (base[order + 1][index] - reference).abs();
                assert!(error.is_finite());
                if error > worst[order] {
                    worst[order] = error;
                    locations[order] = index;
                }
                counts[order] += 1;
            }
        }
        eprintln!(
            "C1 d {name}: reference coordinate {}/12 complete",
            coordinate + 1
        );
    }
    for order in 0..4 {
        assert_eq!(counts[order], 12usize.pow(order as u32 + 1));
        eprintln!(
            "C1 d {name} order {}: {} components, max error {:.8e} at flat index {}",
            order + 1,
            counts[order],
            worst[order],
            locations[order]
        );
        assert!(worst[order] < [1e-6, 1e-5, 2e-7, 3e-6][order]);
    }
}

#[test]
fn c1_d_oracle_energy_gradient_hessian_anchors() {
    for name in [
        "cartesian_ae_rhf",
        "cartesian_ecp_rhf",
        "cartesian_ecp_uhf",
        "spherical_ae_rhf",
        "spherical_ecp_rhf",
        "spherical_ecp_uhf",
    ] {
        let model = c1_d_model(name);
        let atoms = c1_d_atoms(&model);
        c1_d_snapshot(&model, &atoms, 2, true);
    }
}

#[test]
fn c1_d_cartesian_rhf_all_components_through_fourth() {
    verify_c1_d("cartesian_ae_rhf");
}
#[test]
fn c1_d_cartesian_ecp_rhf_all_components_through_fourth() {
    verify_c1_d("cartesian_ecp_rhf");
}
#[test]
fn c1_d_cartesian_ecp_uhf_all_components_through_fourth() {
    verify_c1_d("cartesian_ecp_uhf");
}
#[test]
fn c1_d_spherical_rhf_all_components_through_fourth() {
    verify_c1_d("spherical_ae_rhf");
}
#[test]
fn c1_d_spherical_ecp_rhf_all_components_through_fourth() {
    verify_c1_d("spherical_ecp_rhf");
}
#[test]
fn c1_d_spherical_ecp_uhf_all_components_through_fourth() {
    verify_c1_d("spherical_ecp_uhf");
}

fn mixed_spdf_snapshot(
    atoms: &[Atom],
    ecp: bool,
    open_shell: bool,
    max_order: usize,
) -> Vec<Vec<f64>> {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/spherical_reference.json")).unwrap();
    let mut map = parse_gaussian94(data["spdf_integrals"]["basis_gaussian94"].as_str().unwrap());
    // Contracted SPDF blocks are covered independently in shell_eri_test. A
    // one-primitive response model keeps all nuclear tensor entries affordable.
    for shells in map.values_mut() {
        for shell in shells {
            shell.exponents.truncate(1);
            shell.coefficients.truncate(1);
        }
    }
    let full = build_basis(atoms, &map);
    // One Cartesian function of each angular momentum, on four distinct C1
    // nuclei. The full-shell Cartesian/spherical d tests above are separate.
    let basis: Vec<_> = [0, 22, 49, 74].iter().map(|&i| full[i].clone()).collect();
    assert_eq!(
        basis
            .iter()
            .map(|g| g.l.iter().sum::<usize>())
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    let settings = scf::RhfConvergence {
        // This ionic model has a positive total near 10 Eh. Its energy
        // rounding floor exceeds 1e-15; state residual/density stay at 1e-13.
        energy: 1e-14,
        residual: 1e-13,
        density: 1e-13,
    };
    if open_shell {
        let hf = scf::run_uhf_in_space_with_convergence(
            atoms,
            &basis,
            11,
            2,
            false,
            400,
            settings,
            BasisKind::Cartesian,
        )
        .unwrap();
        assert!(hf.converged);
        let ctx = HfNuclearDerivativeContext::from_uhf(atoms, &basis, &hf).unwrap();
        let mut values = vec![
            vec![hf.total_energy],
            gradient::try_compute_uhf_gradient(atoms, &basis, &hf)
                .unwrap()
                .into_iter()
                .flatten()
                .collect(),
        ];
        for order in 2..=max_order {
            values.push(tensor(&ctx, order, 12));
        }
        values
    } else {
        let hf = scf::run_scf_in_space_with_convergence(
            atoms,
            &basis,
            10,
            400,
            settings,
            ConvAlgo::Cdiis,
            None,
            BasisKind::Cartesian,
        )
        .unwrap();
        assert!(hf.converged, "mixed SPDF ECP={ecp}");
        let ctx = HfNuclearDerivativeContext::from_rhf(atoms, &basis, &hf).unwrap();
        let mut values = vec![
            vec![hf.total_energy],
            gradient::try_compute_rhf_gradient(atoms, &basis, &hf)
                .unwrap()
                .into_iter()
                .flatten()
                .collect(),
        ];
        for order in 2..=max_order {
            values.push(tensor(&ctx, order, 12));
        }
        values
    }
}

fn mixed_spdf_atoms(ecp: bool) -> Vec<Atom> {
    let positions = [
        [-0.4, 0.1, -0.6],
        [0.7, -0.5, 1.0],
        [-1.1, 1.0, 0.4],
        [0.9, 1.3, -0.8],
    ];
    let mut atoms: Vec<_> = ["He", "Li", "Be", "B"]
        .iter()
        .zip(positions)
        .map(|(s, r)| Atom::new(s, r))
        .collect();
    if ecp {
        let term = |exponent, coefficient| ecp::EcpTerm {
            power: 2,
            exponent,
            coefficient,
        };
        atoms[2].ecp = Some(std::sync::Arc::new(EcpDef {
            symbol: "Be".into(),
            n_core: 0,
            l_max: 4,
            local: vec![term(0.6, -0.05)],
            semilocal: vec![
                vec![term(0.9, 0.10)],
                vec![term(0.7, -0.04)],
                vec![term(0.8, 0.08)],
                vec![term(1.0, -0.03)],
            ],
        }));
    }
    atoms
}

#[test]
fn c1_mixed_spdf_independent_energy_and_all_gradient_anchors() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/spherical_reference.json")).unwrap();
    for (name, ecp, open_shell) in [
        ("ae_rhf", false, false),
        ("ecp_rhf", true, false),
        ("ecp_uhf", true, true),
    ] {
        let actual = mixed_spdf_snapshot(&mixed_spdf_atoms(ecp), ecp, open_shell, 1);
        let reference = &data["c1_mixed_spdf"][name];
        let energy_error = (actual[0][0] - reference["energy"].as_f64().unwrap()).abs();
        assert!(energy_error < 1e-8, "{name}: energy {energy_error}");
        let expected: Vec<_> = reference["gradient"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap())
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert_eq!(actual[1].len(), 12);
        let gradient_error = actual[1]
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f64::max);
        assert!(gradient_error < 1e-6, "{name}: gradient {gradient_error}");
        eprintln!(
            "C1 mixed SPDF independent {name}: energy error {energy_error:.8e}, all 12 gradient components max error {gradient_error:.8e}"
        );
    }
}

fn verify_c1_mixed_spdf(ecp: bool, open_shell: bool) {
    let atoms = mixed_spdf_atoms(ecp);
    let base = mixed_spdf_snapshot(&atoms, ecp, open_shell, 4);
    let mut worst = [0f64; 4];
    let mut counts = [0usize; 4];
    let step = 1.25e-3;
    for coordinate in 0..12 {
        let mut differences: Vec<_> = (0..4).map(|order| vec![0.; 12usize.pow(order)]).collect();
        for (shift, weight) in [(-2., 1.), (-1., -8.), (1., 8.), (2., -1.)] {
            let mut moved = atoms.clone();
            moved[coordinate / 3].position[coordinate % 3] += shift * step;
            let values = mixed_spdf_snapshot(&moved, ecp, open_shell, 3);
            for order in 0..4 {
                for (sum, &v) in differences[order].iter_mut().zip(&values[order]) {
                    *sum += weight * v / (12. * step);
                }
            }
        }
        for order in 0..4 {
            for (prefix, &expected) in differences[order].iter().enumerate() {
                let error = (base[order + 1][prefix * 12 + coordinate] - expected).abs();
                assert!(error.is_finite());
                worst[order] = worst[order].max(error);
                counts[order] += 1;
            }
        }
        eprintln!(
            "C1 mixed SPDF ECP={ecp} UHF={open_shell}: coordinate {}/12",
            coordinate + 1
        );
    }
    for order in 0..4 {
        assert_eq!(counts[order], 12usize.pow(order as u32 + 1));
        eprintln!(
            "C1 mixed SPDF ECP={ecp} UHF={open_shell} order {}: {} components, max error {:.8e}",
            order + 1,
            counts[order],
            worst[order]
        );
        assert!(worst[order] < [1e-6, 1e-5, 2e-7, 3e-6][order]);
    }
}

#[test]
fn c1_mixed_spdf_rhf_all_nuclear_components_through_fourth() {
    verify_c1_mixed_spdf(false, false);
}
#[test]
fn c1_mixed_spdf_ecp_rhf_all_nuclear_components_through_fourth() {
    verify_c1_mixed_spdf(true, false);
}
#[test]
fn c1_mixed_spdf_ecp_uhf_all_nuclear_components_through_fourth() {
    verify_c1_mixed_spdf(true, true);
}
