// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use hf_rs::ecp::{EcpTerm, ecp_nuclear_derivative_matrix};
use hf_rs::*;
use std::sync::Arc;

fn minimal_basis() -> BasisSetMap {
    parse_gaussian94("H 0\nS 1 1.0\n0.8 1.0\n****\n")
}

fn molecule(ecp: bool) -> Vec<Atom> {
    let mut atoms = vec![
        Atom::new("H", [0.1, -0.2, -0.7]),
        Atom::new("H", [-0.1, 0.1, 0.7]),
    ];
    if ecp {
        let term = |exponent, coefficient| EcpTerm {
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
    atoms
}

fn direction(n: usize, coordinate: usize) -> NuclearDirection {
    let mut v = vec![[0.0; 3]; n];
    v[coordinate / 3][coordinate % 3] = 1.0;
    v
}

fn shift(atoms: &[Atom], coordinate: usize, delta: f64) -> Vec<Atom> {
    let mut moved = atoms.to_vec();
    moved[coordinate / 3].position[coordinate % 3] += delta;
    moved
}

fn rhf(atoms: &[Atom], map: &BasisSetMap) -> (Vec<ContractedGaussian>, HFResult) {
    let basis = build_basis(atoms, map);
    let hf = run_scf_general(atoms, &basis, 0, 150, 1e-13, ConvAlgo::Cdiis, None).unwrap();
    assert!(hf.converged);
    (basis, hf)
}

#[test]
fn analytical_response_first_and_second_match_verified_rhf_derivatives() {
    let atoms = molecule(false);
    let map = get_builtin_basis("sto-3g").unwrap();
    let (basis, hf) = rhf(&atoms, &map);
    let context = HfNuclearDerivativeContext::from_rhf(&atoms, &basis, &hf).unwrap();
    let g = compute_rhf_gradient(&atoms, &basis, &hf);
    let h = compute_rhf_analytical_hessian(&atoms, &basis, &hf);
    for &i in &[0, 2, 5] {
        let dg = context.coordinate_derivative(&[i]).unwrap();
        println!(
            "gradient {i}: analytic response={dg:.12}, reference={:.12}",
            g[i / 3][i % 3]
        );
        assert!((dg - g[i / 3][i % 3]).abs() < 1e-8);
        for &j in &[0, 2, 5] {
            let dh = context.coordinate_derivative(&[i, j]).unwrap();
            println!(
                "Hessian {i},{j}: analytic response={dh:.12}, reference={:.12}",
                h[(i, j)]
            );
            assert!((dh - h[(i, j)]).abs() < 1e-7);
        }
    }
}

#[test]
fn rhf_third_and_fourth_match_independent_differences_of_analytical_hessians() {
    let atoms = molecule(false);
    let map = minimal_basis();
    let (basis, hf) = rhf(&atoms, &map);
    let context = HfNuclearDerivativeContext::from_rhf(&atoms, &basis, &hf).unwrap();
    let h0 = compute_rhf_analytical_hessian(&atoms, &basis, &hf);
    let step = 5e-3;
    let displaced_hessian = |d| {
        let moved = shift(&atoms, 2, d);
        let (b, f) = rhf(&moved, &map);
        compute_rhf_analytical_hessian(&moved, &b, &f)
    };
    let pp = displaced_hessian(2.0 * step);
    let p = displaced_hessian(step);
    let m = displaced_hessian(-step);
    let mm = displaced_hessian(-2.0 * step);
    for &(i, j) in &[(2, 2), (0, 5)] {
        let third = (-pp[(i, j)] + 8.0 * p[(i, j)] - 8.0 * m[(i, j)] + mm[(i, j)]) / (12.0 * step);
        let fourth = (-pp[(i, j)] + 16.0 * p[(i, j)] - 30.0 * h0[(i, j)] + 16.0 * m[(i, j)]
            - mm[(i, j)])
            / (12.0 * step * step);
        let t = context.coordinate_derivative(&[i, j, 2]).unwrap();
        let q = context.coordinate_derivative(&[i, j, 2, 2]).unwrap();
        println!("{i},{j}: third {t:.10} vs {third:.10}; fourth {q:.10} vs {fourth:.10}");
        assert!((t - third).abs() < 2e-7);
        assert!((q - fourth).abs() < 2e-6);
    }
}

#[test]
fn ecp_operator_center_derivatives_through_fourth_match_gaussian_closed_form() {
    let mut atoms = vec![
        Atom::new("H", [0.1, 0.2, -0.4]),
        Atom::new("H", [-0.2, 0.1, 0.7]),
        Atom::new("He", [0.3, -0.1, 0.2]),
    ];
    let zeta = 0.9;
    let coefficient = -1.7;
    atoms[2].ecp = Some(Arc::new(EcpDef {
        symbol: "He".into(),
        n_core: 0,
        l_max: 0,
        local: vec![EcpTerm {
            power: 2,
            exponent: zeta,
            coefficient,
        }],
        semilocal: vec![],
    }));
    let basis: Vec<_> = [0.7, 1.3]
        .iter()
        .enumerate()
        .map(|(i, &exponent)| ContractedGaussian {
            atom_index: i,
            center: atoms[i].position,
            l: [0; 3],
            primitives: vec![hf_rs::basis::Primitive {
                exponent,
                coefficient: 1.0,
            }],
        })
        .collect();
    let a = basis[0].center;
    let b = basis[1].center;
    let c = atoms[2].position;
    let alpha = 0.7;
    let beta = 1.3;
    let gamma = alpha + beta + zeta;
    let dist2 = |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>();
    let value = coefficient
        * (std::f64::consts::PI / gamma).powf(1.5)
        * (-(alpha * beta * dist2(a, b) + alpha * zeta * dist2(a, c) + beta * zeta * dist2(b, c))
            / gamma)
            .exp();
    let k = zeta * (alpha + beta) / gamma;
    let x = c[2] - (alpha * a[2] + beta * b[2]) / (alpha + beta);
    let expected = [
        -2.0 * k * x,
        4.0 * k * k * x * x - 2.0 * k,
        -8.0 * k.powi(3) * x.powi(3) + 12.0 * k * k * x,
        16.0 * k.powi(4) * x.powi(4) - 48.0 * k.powi(3) * x * x + 12.0 * k * k,
    ];
    for order in 1..=4 {
        let directions = vec![direction(atoms.len(), 8); order];
        let got = ecp_nuclear_derivative_matrix(&atoms, &basis, &directions).unwrap()[(0, 1)];
        println!(
            "ECP order {order}: {got:.12} vs {:.12}",
            value * expected[order - 1]
        );
        assert!((got - value * expected[order - 1]).abs() < 2e-10);
    }
}

#[test]
fn ecp_rhf_and_uhf_fourth_include_response_and_nonlocal_projector_motion() {
    let atoms = molecule(true);
    let map = minimal_basis();
    let (basis, hf) = rhf(&atoms, &map);
    let context = HfNuclearDerivativeContext::from_rhf(&atoms, &basis, &hf).unwrap();
    let g = compute_rhf_gradient(&atoms, &basis, &hf);
    assert!((context.coordinate_derivative(&[2]).unwrap() - g[0][2]).abs() < 1e-8);
    let step = 5e-3;
    // Independent energy differences validate the second-order response first.
    let energy = |d| rhf(&shift(&atoms, 2, d), &map).1.total_energy;
    let epp = energy(2.0 * step);
    let ep = energy(step);
    let em = energy(-step);
    let emm = energy(-2.0 * step);
    let h = (-epp + 16.0 * ep - 30.0 * hf.total_energy + 16.0 * em - emm) / (12.0 * step * step);
    assert!((context.coordinate_derivative(&[2, 2]).unwrap() - h).abs() < 2e-7);
    // Then compare fourth order to a five-point second derivative of that Hessian.
    let hessian = |d| {
        let moved = shift(&atoms, 2, d);
        let (b, f) = rhf(&moved, &map);
        HfNuclearDerivativeContext::from_rhf(&moved, &b, &f)
            .unwrap()
            .coordinate_derivative(&[0, 5])
            .unwrap()
    };
    let h0 = context.coordinate_derivative(&[0, 5]).unwrap();
    let fd = (-hessian(2.0 * step) + 16.0 * hessian(step) - 30.0 * h0 + 16.0 * hessian(-step)
        - hessian(-2.0 * step))
        / (12.0 * step * step);
    let fourth = context.coordinate_derivative(&[0, 5, 2, 2]).unwrap();
    println!("nonlocal ECP RHF fourth: {fourth:.10} vs {fd:.10}");
    assert!((fourth - fd).abs() < 3e-6);
    // Compute another variable ordering directly; this is not a symmetrized tensor lookup.
    let independent = HfNuclearDerivativeContext::from_rhf(&atoms, &basis, &hf).unwrap();
    let permuted = independent.coordinate_derivative(&[2, 0, 2, 5]).unwrap();
    assert!((fourth - permuted).abs() < 1e-9);

    let uhf = run_uhf(&atoms, &basis, 1, 2, false, 150, 1e-13).unwrap();
    assert!(uhf.converged);
    let u = HfNuclearDerivativeContext::from_uhf(&atoms, &basis, &uhf).unwrap();
    let g = compute_uhf_gradient(&atoms, &basis, &uhf);
    assert!((u.coordinate_derivative(&[2]).unwrap() - g[0][2]).abs() < 1e-8);
    let curvature = |d| {
        let moved = shift(&atoms, 2, d);
        let b = build_basis(&moved, &map);
        let f = run_uhf(&moved, &b, 1, 2, false, 150, 1e-13).unwrap();
        assert!(f.converged);
        HfNuclearDerivativeContext::from_uhf(&moved, &b, &f)
            .unwrap()
            .coordinate_derivative(&[2, 2])
            .unwrap()
    };
    let h0 = u.coordinate_derivative(&[2, 2]).unwrap();
    let fd = (-curvature(2.0 * step) + 16.0 * curvature(step) - 30.0 * h0
        + 16.0 * curvature(-step)
        - curvature(-2.0 * step))
        / (12.0 * step * step);
    let fourth = u.coordinate_derivative(&[2, 2, 2, 2]).unwrap();
    println!("nonlocal ECP UHF fourth: {fourth:.10} vs {fd:.10}");
    assert!((fourth - fd).abs() < 3e-6);
}

#[test]
fn open_shell_fourth_solves_both_spin_response_channels() {
    let mut atoms = molecule(true);
    atoms.push(Atom::new("H", [1.5, 0.3, 0.0]));
    let map = minimal_basis();
    let basis = build_basis(&atoms, &map);
    let hf = run_uhf(&atoms, &basis, 0, 2, false, 250, 1e-13).unwrap();
    assert!(hf.converged && hf.n_alpha == 2 && hf.n_beta == 1);
    let context = HfNuclearDerivativeContext::from_uhf(&atoms, &basis, &hf).unwrap();
    let grad = compute_uhf_gradient(&atoms, &basis, &hf);
    assert!((context.coordinate_derivative(&[2]).unwrap() - grad[0][2]).abs() < 1e-7);
    let step = 5e-3;
    let curvature = |d| {
        let moved = shift(&atoms, 2, d);
        let b = build_basis(&moved, &map);
        let f = run_uhf(&moved, &b, 0, 2, false, 250, 1e-13).unwrap();
        assert!(f.converged);
        HfNuclearDerivativeContext::from_uhf(&moved, &b, &f)
            .unwrap()
            .coordinate_derivative(&[2, 2])
            .unwrap()
    };
    let h0 = context.coordinate_derivative(&[2, 2]).unwrap();
    let fd = (-curvature(2.0 * step) + 16.0 * curvature(step) - 30.0 * h0
        + 16.0 * curvature(-step)
        - curvature(-2.0 * step))
        / (12.0 * step * step);
    let fourth = context.coordinate_derivative(&[2, 2, 2, 2]).unwrap();
    println!("three-electron UHF fourth: {fourth:.10} vs {fd:.10}");
    assert!((fourth - fd).abs() < 3e-6);
}

#[test]
fn high_angular_hermite_overflow_matches_moments_and_independent_eri_path() {
    use hf_rs::integrals::{contracted_eri, primitive_eri, primitive_overlap};
    let gamma = 2.3_f64;
    let moment = (1..=7).map(|i| (2 * i - 1) as f64).product::<f64>() / (2.0 * gamma).powi(7)
        * (std::f64::consts::PI / gamma).powf(1.5);
    let got = primitive_overlap(1.0, [7, 0, 0], [0.0; 3], 1.3, [7, 0, 0], [0.0; 3]);
    assert!((got - moment).abs() < 1e-12 * moment.abs());
    let gaussian = |l| ContractedGaussian {
        atom_index: 0,
        center: [0.0; 3],
        l,
        primitives: vec![hf_rs::basis::Primitive {
            exponent: 1.0,
            coefficient: 1.0,
        }],
    };
    let a = gaussian([7, 0, 0]);
    let b = gaussian([0; 3]);
    let direct = primitive_eri(
        1.0, a.l, a.center, 1.0, a.l, a.center, 1.0, b.l, b.center, 1.0, b.l, b.center,
    );
    let packed = contracted_eri(&a, &a, &b, &b);
    assert!((direct - packed).abs() < 1e-11 * direct.abs().max(1.0));
    // Balanced angular momenta require a Coulomb workspace larger than 8,192 doubles.
    let g = gaussian([3, 2, 2]);
    let direct = primitive_eri(
        1.0, g.l, g.center, 1.0, g.l, g.center, 1.0, g.l, g.center, 1.0, g.l, g.center,
    );
    let packed = contracted_eri(&g, &g, &g, &g);
    assert!((direct - packed).abs() < 1e-11 * direct.abs().max(1.0));
}

#[test]
fn small_matrix_diagonalization_resolves_tight_scf_residuals() {
    let a = Matrix::from_vec(
        3,
        3,
        vec![-1.0, 2e-9, -3e-9, 2e-9, -0.3, 5e-9, -3e-9, 5e-9, 0.6],
    );
    let (eps, c) = a.eigh_jacobi();
    let ac = a.dot(&c);
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((ac[(i, j)] - c[(i, j)] * eps[j]).abs());
        }
    }
    assert!(worst < 5e-14, "eigenpair residual {worst}");
}

#[test]
fn def2_ecp_fourth_derivative_is_quadrature_converged() {
    use hf_rs::ecp::{EcpQuadrature, ecp_nuclear_derivative_matrix_with};
    let mut atoms = vec![
        Atom::new("H", [0.1, -0.3, -0.8]),
        Atom::new("H", [-0.2, 0.2, 0.9]),
        Atom::new("I", [0.3, -0.1, 0.2]),
    ];
    assign_builtin_ecps(&mut atoms, "def2-svp");
    let basis: Vec<_> = [([1, 0, 0], 0.7), ([0, 0, 0], 1.3)]
        .into_iter()
        .enumerate()
        .map(|(i, (l, exponent))| ContractedGaussian {
            atom_index: i,
            center: atoms[i].position,
            l,
            primitives: vec![hf_rs::basis::Primitive {
                exponent,
                coefficient: 1.0,
            }],
        })
        .collect();
    let directions = vec![direction(atoms.len(), 8); 4];
    let q = EcpQuadrature::default();
    let fine = EcpQuadrature {
        panel_nodes: 24,
        angular_scale: 2.0,
        angular_margin: 28,
        ..q
    };
    let fourth = ecp_nuclear_derivative_matrix_with(&atoms, &basis, &directions, &q).unwrap();
    let refined = ecp_nuclear_derivative_matrix_with(&atoms, &basis, &directions, &fine).unwrap();
    let step = 5e-3;
    let second = |d| {
        let moved = shift(&atoms, 8, d);
        ecp_nuclear_derivative_matrix_with(&moved, &basis, &directions[..2], &fine).unwrap()
    };
    let pp = second(2.0 * step);
    let p = second(step);
    let z = second(0.0);
    let m = second(-step);
    let mm = second(-2.0 * step);
    for i in 0..2 {
        for j in 0..2 {
            let fd = (-pp[(i, j)] + 16.0 * p[(i, j)] - 30.0 * z[(i, j)] + 16.0 * m[(i, j)]
                - mm[(i, j)])
                / (12.0 * step * step);
            assert!((fourth[(i, j)] - refined[(i, j)]).abs() < 2e-8);
            assert!(
                (refined[(i, j)] - fd).abs() < 2e-6,
                "def2 ECP fourth ({i},{j}): {} vs {fd}",
                refined[(i, j)]
            );
        }
    }
}

#[test]
fn directional_block_and_mode_apis_use_the_same_analytical_response() {
    let atoms = molecule(true);
    let map = minimal_basis();
    let (basis, hf) = rhf(&atoms, &map);
    let ctx = HfNuclearDerivativeContext::from_rhf(&atoms, &basis, &hf).unwrap();
    let u = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let v = vec![0.3, -0.2, 0.1, 0.0, 0.4, -0.2];
    let dirs = |v: &[f64]| v.chunks_exact(3).map(|x| [x[0], x[1], x[2]]).collect();
    let expected3 = ctx.derivative(&[dirs(&u), dirs(&u), dirs(&v)]).unwrap();
    let expected4 = ctx
        .derivative(&[dirs(&u), dirs(&v), dirs(&u), dirs(&v)])
        .unwrap();
    assert!(
        (compute_third_directional_derivative(&atoms, &map, &u, &u, &v, 0, 150, 1e-13) - expected3)
            .abs()
            < 1e-10
    );
    assert!(
        (compute_fourth_directional_derivative(
            &atoms, &map, &u, &v, &u, &v, 0, 1, false, 150, 1e-13
        ) - expected4)
            .abs()
            < 1e-10
    );
    let modes = vec![u, v];
    let third = compute_third_derivatives_subspace(&atoms, &map, &modes, 0, 150, 1e-13);
    let fourth = compute_fourth_derivatives_subspace(&atoms, &map, &modes, 0, 1, false, 150, 1e-13);
    assert!((third[1] - expected3).abs() < 1e-10); // [0,0,1]
    assert!((fourth[5] - expected4).abs() < 1e-10); // [0,1,0,1]
    let block3 = compute_third_derivatives_block(&atoms, &map, &[1], 0, 150, 1e-13);
    let block4 = compute_fourth_derivatives_block(&atoms, &map, &[1], 0, 1, false, 150, 1e-13);
    assert!((block3.get(0, 1, 2) - ctx.coordinate_derivative(&[3, 4, 5]).unwrap()).abs() < 1e-10);
    assert!(
        (block4.get(0, 1, 2, 2) - ctx.coordinate_derivative(&[3, 4, 5, 5]).unwrap()).abs() < 1e-10
    );
}
