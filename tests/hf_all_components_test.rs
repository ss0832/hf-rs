// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Every Cartesian component of orders 1..4 is compared with a five-point
//! difference of the next lower derivative. References are never symmetrized.
use hf_rs::*;
use std::sync::Arc;

const SCF_TOL: f64 = 1e-15;

struct Case {
    name: &'static str,
    atoms: Vec<Atom>,
    map: BasisSetMap,
    multiplicity: usize,
    charge: i32,
    c1: bool,
}

fn case(ecp: bool, open_shell: bool) -> Case {
    let mut atoms = vec![
        Atom::new("H", [0.1, -0.2, -0.7]),
        Atom::new("H", [-0.1, 0.1, 0.7]),
    ];
    if ecp {
        let term = |exponent, coefficient| hf_rs::ecp::EcpTerm {
            power: 2,
            exponent,
            coefficient,
        };
        atoms[0].ecp = Some(Arc::new(EcpDef {
            symbol: "H".into(),
            n_core: 0,
            l_max: 2,
            local: vec![term(0.6, -0.15)],
            semilocal: vec![vec![term(0.9, 0.25)], vec![term(0.7, -0.10)]],
        }));
    }
    if open_shell {
        atoms.push(Atom::new("H", [1.5, 0.3, 0.0]));
    }
    Case {
        name: if open_shell {
            "ECP UHF H3 doublet"
        } else if ecp {
            "ECP RHF H2"
        } else {
            "all-electron RHF H2/STO-3G"
        },
        atoms,
        map: if ecp {
            parse_gaussian94("H 0\nS 1 1.0\n0.8 1.0\n****\n")
        } else {
            get_builtin_basis("sto-3g").unwrap()
        },
        multiplicity: if open_shell { 2 } else { 1 },
        charge: 0,
        c1: false,
    }
}

// [energy, gradient, Hessian, third tensor], all in row-major Cartesian order.
fn snapshot(case: &Case, atoms: &[Atom], base: bool) -> Vec<Vec<f64>> {
    let basis = build_basis(atoms, &case.map);
    if case.multiplicity == 1 {
        let hf = run_scf_general(
            atoms,
            &basis,
            case.charge,
            250,
            SCF_TOL,
            ConvAlgo::Cdiis,
            None,
        )
        .unwrap();
        assert!(hf.converged);
        let ctx = HfNuclearDerivativeContext::from_rhf(atoms, &basis, &hf).unwrap();
        let grad = compute_rhf_gradient(atoms, &basis, &hf);
        let h = if !base && !hf_rs::ecp::has_ecp(atoms) {
            compute_rhf_analytical_hessian(atoms, &basis, &hf)
        } else {
            ctx.hessian().unwrap()
        };
        collect_snapshot(&ctx, hf.total_energy, &grad, h, base)
    } else {
        let hf = run_uhf(
            atoms,
            &basis,
            case.charge,
            case.multiplicity,
            false,
            250,
            SCF_TOL,
        )
        .unwrap();
        let electrons = atoms
            .iter()
            .map(|a| a.charge.round() as usize)
            .sum::<usize>() as i32
            - case.charge;
        assert!(hf.converged && hf.n_alpha + hf.n_beta == electrons as usize);
        let ctx = HfNuclearDerivativeContext::from_uhf(atoms, &basis, &hf).unwrap();
        let grad = compute_uhf_gradient(atoms, &basis, &hf);
        let h = ctx.hessian().unwrap();
        collect_snapshot(&ctx, hf.total_energy, &grad, h, base)
    }
}

fn collect_snapshot(
    ctx: &HfNuclearDerivativeContext<'_>,
    energy: f64,
    gradient: &[[f64; 3]],
    h: Matrix,
    base: bool,
) -> Vec<Vec<f64>> {
    let dim = gradient.len() * 3;
    let mut g: Vec<f64> = gradient.iter().flatten().copied().collect();
    if base {
        for (i, value) in g.iter_mut().enumerate() {
            let direct = ctx.coordinate_derivative(&[i]).unwrap();
            assert!(
                (direct - *value).abs() < 1e-8,
                "independent gradient component {i}"
            );
            *value = direct;
        }
    }
    let mut third = vec![0.0; dim.pow(3)];
    // Base tensors come from the public checked tensor APIs in verify_all.
    // Displaced third derivatives are evaluated separately through the context.
    if !base {
        for i in 0..dim {
            for j in 0..=i {
                for k in 0..=j {
                    let value = ctx.coordinate_derivative(&[i, j, k]).unwrap();
                    for [a, b, c] in [
                        [i, j, k],
                        [i, k, j],
                        [j, i, k],
                        [j, k, i],
                        [k, i, j],
                        [k, j, i],
                    ] {
                        third[(a * dim + b) * dim + c] = value;
                    }
                }
            }
        }
    }
    vec![vec![energy], g, h.data, third]
}

fn coordinates(mut index: usize, dim: usize, order: usize) -> Vec<usize> {
    let mut result = vec![0; order];
    for c in result.iter_mut().rev() {
        *c = index % dim;
        index /= dim;
    }
    result
}

fn verify_all(case: Case) {
    let dim = 3 * case.atoms.len();
    let mut base = snapshot(&case, &case.atoms, true);
    base[3] = try_compute_third_derivatives_general(
        &case.atoms,
        &case.map,
        case.charge,
        case.multiplicity,
        false,
        250,
        SCF_TOL,
    )
    .unwrap()
    .data;
    let fourth = try_compute_fourth_derivatives_general(
        &case.atoms,
        &case.map,
        case.charge,
        case.multiplicity,
        false,
        250,
        SCF_TOL,
    )
    .unwrap();
    base.push(fourth.data);
    let mut counts = [0usize; 4];
    let mut worst = [0.0_f64; 4];
    let mut locations = [0usize; 4];
    // The small-gap doublet needs a finer reference step: the convergence test
    // below demonstrates the expected 16-fold reduction of five-point bias.
    let step = if case.multiplicity > 1 || case.c1 {
        1.25e-3
    } else {
        5e-3
    };
    for coordinate in 0..dim {
        let mut differences: Vec<Vec<f64>> =
            (0..4).map(|order| vec![0.0; dim.pow(order)]).collect();
        for (offset, weight) in [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)] {
            let mut moved = case.atoms.clone();
            moved[coordinate / 3].position[coordinate % 3] += offset * step;
            let values = snapshot(&case, &moved, false);
            for order in 0..4 {
                for (sum, &value) in differences[order].iter_mut().zip(&values[order]) {
                    *sum += weight * value / (12.0 * step);
                }
            }
        }
        for order in 0..4 {
            for (prefix, &reference) in differences[order].iter().enumerate() {
                let index = prefix * dim + coordinate;
                let error = (base[order + 1][index] - reference).abs();
                assert!(error.is_finite());
                counts[order] += 1;
                if error > worst[order] {
                    worst[order] = error;
                    locations[order] = index;
                }
            }
        }
        eprintln!(
            "{}: all-component reference coordinate {}/{} complete",
            case.name,
            coordinate + 1,
            dim
        );
    }
    for order in 1..=4 {
        assert_eq!(counts[order - 1], dim.pow(order as u32));
        eprintln!(
            "{} order {order}: {} components, max error {:.6e} at {:?}",
            case.name,
            counts[order - 1],
            worst[order - 1],
            coordinates(locations[order - 1], dim, order)
        );
    }
    for (order, limit) in [1e-6, 1e-5, 2e-7, 3e-6].into_iter().enumerate() {
        assert!(
            worst[order] < limit,
            "{} order {} maximum error {} at {:?} exceeds {limit}",
            case.name,
            order + 1,
            worst[order],
            coordinates(locations[order], dim, order + 1)
        );
    }
}

#[test]
fn all_electron_rhf_every_cartesian_component_through_fourth() {
    verify_all(case(false, false));
}

#[test]
fn ecp_rhf_every_cartesian_component_through_fourth() {
    verify_all(case(true, false));
}

#[test]
fn ecp_open_shell_uhf_every_cartesian_component_through_fourth() {
    verify_all(case(true, true));
}

fn c1_case(ecp: bool, open_shell: bool) -> Case {
    let mut model = case(ecp, open_shell);
    model.name = match (ecp, open_shell) {
        (false, _) => "C1 noncoplanar HeH3+ RHF, s/p AOs",
        (true, false) => "C1 nonlocal-ECP HeH3+ RHF, s/p AOs",
        (true, true) => "C1 nonlocal-ECP HeH3 UHF doublet, s/p AOs",
    };
    let potential = model.atoms[0].ecp.clone();
    model.atoms = vec![
        Atom::new("He", [-0.4, 0.1, -0.6]),
        Atom::new("H", [0.7, -0.5, 1.0]),
        Atom::new("H", [-1.1, 1.0, 0.4]),
        Atom::new("H", [0.9, 1.3, -0.8]),
    ];
    if let Some(potential) = potential {
        let mut definition = (*potential).clone();
        definition.symbol = "He".into();
        model.atoms[0].ecp = Some(Arc::new(definition));
    }
    model.map = parse_gaussian94(
        "He 0\nS 1 1.0\n0.8 1.0\nP 1 1.0\n0.6 1.0\n****\nH 0\nS 1 1.0\n1.0 1.0\n****\n",
    );
    model.charge = if open_shell { 0 } else { 1 };
    model.c1 = true;
    // Four noncoplanar points with six distinct pair distances admit no
    // nonidentity distance-preserving atom permutation or mirror plane.
    let vectors: Vec<_> = model.atoms[1..]
        .iter()
        .map(|a| std::array::from_fn::<_, 3, _>(|k| a.position[k] - model.atoms[0].position[k]))
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
                    .map(|k| (model.atoms[i].position[k] - model.atoms[j].position[k]).powi(2))
                    .sum::<f64>(),
            );
        }
    }
    distances.sort_by(f64::total_cmp);
    assert!(distances.windows(2).all(|x| (x[1] - x[0]).abs() > 1e-3));
    assert_eq!(build_basis(&model.atoms, &model.map).len(), 7);
    model
}

#[test]
fn c1_all_electron_rhf_all_components_through_fourth() {
    verify_all(c1_case(false, false));
}
#[test]
fn c1_nonlocal_ecp_rhf_all_components_through_fourth() {
    verify_all(c1_case(true, false));
}
#[test]
fn c1_nonlocal_ecp_uhf_all_components_through_fourth() {
    verify_all(c1_case(true, true));
}

#[test]
fn open_shell_reference_shows_five_point_step_convergence() {
    let case = case(true, true);
    let selected = |atoms: &[Atom], tol: f64| {
        let basis = build_basis(atoms, &case.map);
        let hf = run_uhf(atoms, &basis, 0, 2, false, 500, tol).unwrap();
        assert!(hf.converged, "SCF tol={tol}");
        let ctx = HfNuclearDerivativeContext::from_uhf(atoms, &basis, &hf).unwrap();
        [
            ctx.coordinate_derivative(&[5, 5]).unwrap(),
            ctx.coordinate_derivative(&[5, 5, 5]).unwrap(),
            ctx.coordinate_derivative(&[2, 5, 5]).unwrap(),
            ctx.coordinate_derivative(&[2, 5, 5, 5]).unwrap(),
        ]
    };
    let mut previous: Option<[f64; 2]> = None;
    for step in [5e-3, 2.5e-3, 1.25e-3] {
        let tol = 1e-15;
        let base = selected(&case.atoms, tol);
        let mut reference = [0.0; 2];
        for (offset, weight) in [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)] {
            let mut moved = case.atoms.clone();
            moved[1].position[2] += offset * step;
            let values = selected(&moved, tol);
            reference[0] += weight * values[0] / (12.0 * step);
            reference[1] += weight * values[2] / (12.0 * step);
        }
        let errors = [
            (base[1] - reference[0]).abs(),
            (base[3] - reference[1]).abs(),
        ];
        eprintln!("FD step {step:e}: worst-component third/fourth FD errors {errors:?}");
        if let Some(old) = previous {
            for order in 0..2 {
                let ratio = old[order] / errors[order];
                assert!(
                    (10.0..22.0).contains(&ratio),
                    "five-point truncation error ratio {ratio}"
                );
            }
        }
        previous = Some(errors);
        if step == 1.25e-3 {
            assert!(errors[0] < 2e-7 && errors[1] < 3e-6);
        }
    }
}
