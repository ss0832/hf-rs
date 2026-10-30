// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Effective core potentials: parser, analytic limits of the semi-local integrals, quadrature convergence,
//! invariances, and PySCF oracle comparison (tests/data/ecp_reference.json, tests/reference/gen_ecp.py).

use std::f64::consts::PI;

use hf_rs::basis::{ContractedGaussian, Primitive};
use hf_rs::ecp::{
    EcpDef, EcpFunction, EcpQuadrature, EcpTerm, assign_builtin_ecps, ecp_gradient,
    ecp_matrix_with, ecp_one_electron_matrix, ecp_one_electron_matrix_with, get_builtin_ecp,
    parse_ecp_gaussian94,
};
use hf_rs::integrals::contracted_overlap;
use hf_rs::{
    Atom, ConvAlgo, Matrix, build_basis, compute_rhf_gradient, compute_uhf_gradient,
    get_builtin_basis, run_scf_general, run_uhf,
};

fn s_cg(center: [f64; 3], l: [usize; 3], exps: &[f64], coefs: &[f64]) -> ContractedGaussian {
    ContractedGaussian {
        atom_index: 0,
        center,
        l,
        primitives: exps
            .iter()
            .zip(coefs)
            .map(|(&e, &c)| Primitive {
                exponent: e,
                coefficient: c,
            })
            .collect(),
    }
}

fn term(power: i32, exponent: f64, coefficient: f64) -> EcpTerm {
    EcpTerm {
        power,
        exponent,
        coefficient,
    }
}

// ------------------------------------------------------------------------------------------------
// Parser and data
// ------------------------------------------------------------------------------------------------

#[test]
fn def2_ecp_data_matches_the_bse_json_manifest_for_all_50_elements() {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("data/basis_manifest.json")).unwrap();
    let expected = manifest["ecp_def2"].as_object().unwrap();
    let map = get_builtin_ecp("def2-svp").unwrap();
    assert_eq!(map.len(), 50, "def2 has ECPs for Z = 37..86");
    assert_eq!(expected.len(), 50);
    for (z, e) in expected {
        let sym = hf_rs::basis::PERIODIC_TABLE[z.parse::<usize>().unwrap() - 1].to_uppercase();
        let got = &map[&sym];
        assert_eq!(
            got.n_core,
            e["n_core"].as_u64().unwrap() as usize,
            "{} n_core",
            sym
        );
        assert_eq!(
            got.l_max,
            e["l_max"].as_u64().unwrap() as usize,
            "{} l_max",
            sym
        );
        for (l, ch) in e["channels"].as_object().unwrap() {
            let l: usize = l.parse().unwrap();
            let terms: &[EcpTerm] = if l == got.l_max {
                &got.local
            } else {
                &got.semilocal[l]
            };
            let ch = ch.as_array().unwrap();
            assert_eq!(
                terms.len(),
                ch[0].as_u64().unwrap() as usize,
                "{} channel {} term count",
                sym,
                l
            );
            let se: f64 = terms.iter().map(|t| t.exponent).sum();
            let sc: f64 = terms.iter().map(|t| t.coefficient).sum();
            let sp: i32 = terms.iter().map(|t| t.power).sum();
            assert!(
                (se - ch[1].as_f64().unwrap()).abs() < 1e-9 * se.abs().max(1.0),
                "{} {} exponents",
                sym,
                l
            );
            assert!(
                (sc - ch[2].as_f64().unwrap()).abs() < 1e-9 * sc.abs().max(1.0),
                "{} {} coefficients",
                sym,
                l
            );
            assert_eq!(sp as i64, ch[3].as_i64().unwrap(), "{} {} r powers", sym, l);
        }
    }
    // spot checks that do not depend on the manifest: core sizes of the def2 ECP28 / ECP46 / ECP60 families
    assert_eq!(
        (map["RB"].n_core, map["I"].n_core, map["XE"].n_core),
        (28, 28, 28)
    );
    assert_eq!(
        (map["CS"].n_core, map["AU"].n_core, map["PB"].n_core),
        (46, 60, 60)
    );
    for n in ["def2-tzvp", "def2-qzvp", "def2svp", "TZVP"] {
        assert_eq!(get_builtin_ecp(n).unwrap().len(), 50);
    }
    assert!(get_builtin_ecp("cc-pvdz").is_none() && get_builtin_ecp("sto-3g").is_none());
}

#[test]
fn ecp_parser_reads_all_terms_of_iodine() {
    let text = "I-ECP     3     28\nf potential\n  2\n2      1.5      -2.0\n2      0.5       3.0\ns-f potential\n  1\n2      4.0       5.0\np-f potential\n 1\n1 2.0 1.0\nd-f potential\n 1\n2 3.0 -1.0\n";
    let map = parse_ecp_gaussian94(text).unwrap();
    let e = &map["I"];
    assert_eq!((e.n_core, e.l_max), (28, 3));
    assert_eq!(e.local, vec![term(2, 1.5, -2.0), term(2, 0.5, 3.0)]);
    assert_eq!(e.semilocal[0], vec![term(2, 4.0, 5.0)]);
    assert_eq!(e.semilocal[1], vec![term(1, 2.0, 1.0)]);
    assert!(
        parse_ecp_gaussian94("I-ECP 3 28\nz potential\n").is_err(),
        "unknown channel letters are rejected"
    );
}

#[test]
fn assigning_ecps_reduces_the_effective_nuclear_charge() {
    let mut atoms = vec![
        Atom::new("I", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, 3.0]),
        Atom::new("Pb", [5.0, 0.0, 0.0]),
    ];
    assign_builtin_ecps(&mut atoms, "def2-svp");
    assert_eq!(atoms[0].charge, 25.0);
    assert_eq!(atoms[1].charge, 1.0);
    assert_eq!(atoms[2].charge, 22.0);
    assert_eq!(atoms[0].atomic_number, 53, "the atomic number is kept");
    assert!(atoms[0].ecp.is_some() && atoms[1].ecp.is_none());
    // all-electron basis sets do not attach ECPs
    let mut ae = vec![Atom::new("I", [0.0, 0.0, 0.0])];
    assign_builtin_ecps(&mut ae, "cc-pvdz");
    assert!(ae[0].ecp.is_none() && ae[0].charge == 53.0);
}

// ------------------------------------------------------------------------------------------------
// Analytic limits of the integral engine
// ------------------------------------------------------------------------------------------------

fn matrix_1x1(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    c: [f64; 3],
    def: &EcpDef,
    q: &EcpQuadrature,
) -> f64 {
    let fa = vec![EcpFunction::from_contracted(a, c)];
    let fb = vec![EcpFunction::from_contracted(b, c)];
    ecp_matrix_with(&fa, &fb, false, c, def, q)[(0, 0)]
}

#[test]
fn local_gaussian_potential_matches_the_three_center_gaussian_formula() {
    // <a| d exp(-zeta |r-C|^2) |b> for s-type primitives has a closed form (Gaussian product theorem).
    let (zeta, d) = (0.9, -1.7);
    let def = EcpDef {
        symbol: "X".into(),
        n_core: 0,
        l_max: 0,
        local: vec![term(2, zeta, d)],
        semilocal: vec![],
    };
    let q = EcpQuadrature::default();
    let c = [0.1, -0.2, 0.3];
    for &(alpha, beta, a, b) in &[
        (0.7, 1.3, [0.0, 0.0, 0.0], [0.4, 0.2, -0.6]),
        (3.1, 0.2, [1.5, -0.5, 0.2], [-0.8, 0.9, 1.1]),
        (12.0, 5.0, [0.1, -0.2, 0.3], [0.0, 0.0, 0.3]),
    ] {
        let ga = s_cg(a, [0, 0, 0], &[alpha], &[1.0]);
        let gb = s_cg(b, [0, 0, 0], &[beta], &[1.0]);
        let dist2 = |x: [f64; 3], y: [f64; 3]| (0..3).map(|k| (x[k] - y[k]).powi(2)).sum::<f64>();
        let g = alpha + beta + zeta;
        let exact = d
            * (PI / g).powf(1.5)
            * (-(alpha * beta * dist2(a, b)
                + alpha * zeta * dist2(a, c)
                + beta * zeta * dist2(b, c))
                / g)
                .exp();
        let got = matrix_1x1(&ga, &gb, c, &def, &q);
        assert!(
            (got - exact).abs() < 1e-11 * exact.abs().max(1e-3),
            "local: {} vs {}",
            got,
            exact
        );
    }
}

#[test]
fn semilocal_projection_of_s_and_p_functions_centered_on_the_ecp_matches_closed_forms() {
    let (alpha, beta, zeta, d) = (1.3, 0.45, 0.8, 2.3);
    let c = [0.2, 0.1, -0.3];
    let gamma = alpha + beta + zeta;
    let q = EcpQuadrature::default();
    // l = 0 channel: <s|P_0 dU|s> = d pi^{3/2} gamma^{-3/2}
    let def0 = EcpDef {
        symbol: "X".into(),
        n_core: 0,
        l_max: 1,
        local: vec![],
        semilocal: vec![vec![term(2, zeta, d)]],
    };
    let s_a = s_cg(c, [0, 0, 0], &[alpha], &[1.0]);
    let s_b = s_cg(c, [0, 0, 0], &[beta], &[1.0]);
    let got = matrix_1x1(&s_a, &s_b, c, &def0, &q);
    assert!(
        (got - d * PI.powf(1.5) * gamma.powf(-1.5)).abs() < 1e-12,
        "s-s {}",
        got
    );
    // p functions are invisible to the l = 0 projector
    for l in [[1, 0, 0], [0, 1, 0], [0, 0, 1]] {
        let p_a = s_cg(c, l, &[alpha], &[1.0]);
        let p_b = s_cg(c, l, &[beta], &[1.0]);
        assert!(
            matrix_1x1(&p_a, &p_b, c, &def0, &q).abs() < 1e-13,
            "p under l=0 projector"
        );
        assert!(
            matrix_1x1(&s_a, &p_b, c, &def0, &q).abs() < 1e-13,
            "s-p coupling vanishes by symmetry"
        );
    }
    // l = 1 channel: <p_x|P_1 dU|p_x> = d pi^{3/2} / 2 gamma^{-5/2}; s functions are invisible
    let def1 = EcpDef {
        symbol: "X".into(),
        n_core: 0,
        l_max: 2,
        local: vec![],
        semilocal: vec![vec![], vec![term(2, zeta, d)]],
    };
    for l in [[1, 0, 0], [0, 1, 0], [0, 0, 1]] {
        let p_a = s_cg(c, l, &[alpha], &[1.0]);
        let p_b = s_cg(c, l, &[beta], &[1.0]);
        let got = matrix_1x1(&p_a, &p_b, c, &def1, &q);
        assert!(
            (got - d * PI.powf(1.5) / 2.0 * gamma.powf(-2.5)).abs() < 1e-12,
            "p-p {}",
            got
        );
    }
    assert!(
        matrix_1x1(&s_a, &s_b, c, &def1, &q).abs() < 1e-13,
        "s functions under the l = 1 projector"
    );
    assert!(
        matrix_1x1(
            &s_cg(c, [1, 0, 0], &[alpha], &[1.0]),
            &s_cg(c, [0, 1, 0], &[beta], &[1.0]),
            c,
            &def1,
            &q
        )
        .abs()
            < 1e-13
    );
}

#[test]
fn complete_set_of_semilocal_projectors_reproduces_the_local_potential() {
    // Sum_{l=0}^{inf} P_l = 1. With the same radial function f in all channels l < L (L large) and
    // U_L = 0 the semi-local operator equals the local potential f for functions of limited angular content.
    let f = vec![term(2, 1.1, 3.0), term(3, 0.6, -0.8)];
    let l_big = 14;
    let semi = EcpDef {
        symbol: "X".into(),
        n_core: 0,
        l_max: l_big,
        local: vec![],
        semilocal: vec![f.clone(); l_big],
    };
    let loc = EcpDef {
        symbol: "X".into(),
        n_core: 0,
        l_max: 0,
        local: f,
        semilocal: vec![],
    };
    let q = EcpQuadrature::default();
    let c = [0.0, 0.0, 0.0];
    let funcs = [
        s_cg([0.0, 0.0, 0.0], [0, 0, 0], &[2.0, 0.4], &[0.6, 0.5]),
        s_cg([0.0, 0.0, 0.0], [1, 0, 0], &[0.9], &[1.0]),
        s_cg([0.3, 0.4, -0.5], [0, 0, 0], &[1.5], &[1.0]),
        s_cg([-0.6, 0.2, 0.7], [0, 1, 1], &[0.8], &[1.0]),
        s_cg([0.9, 0.9, 0.0], [2, 0, 0], &[0.5], &[1.0]),
    ];
    let ef: Vec<EcpFunction> = funcs
        .iter()
        .map(|g| EcpFunction::from_contracted(g, c))
        .collect();
    let m_semi = ecp_matrix_with(&ef, &ef, true, c, &semi, &q);
    let m_loc = ecp_matrix_with(&ef, &ef, true, c, &loc, &q);
    let mut worst = 0.0_f64;
    for i in 0..funcs.len() {
        for j in 0..funcs.len() {
            worst = worst.max((m_semi[(i, j)] - m_loc[(i, j)]).abs());
        }
    }
    assert!(
        worst < 1e-8,
        "projector completeness violated: {:.3e}",
        worst
    );
}

#[test]
fn quadrature_is_converged_and_matrices_are_symmetric() {
    let mut atoms = vec![
        Atom::new("I", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.3, -0.2, 3.0]),
    ];
    assign_builtin_ecps(&mut atoms, "def2-svp");
    let basis = build_basis(&atoms, &get_builtin_basis("def2-svp").unwrap());
    let coarse = ecp_one_electron_matrix(&atoms, &basis);
    let fine = ecp_one_electron_matrix_with(
        &atoms,
        &basis,
        &EcpQuadrature {
            panel_nodes: 24,
            panel_ratio: 1.2,
            first_edge: 0.001,
            angular_scale: 2.4,
            angular_margin: 40,
        },
    );
    let n = basis.len();
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            worst = worst.max((coarse[(i, j)] - fine[(i, j)]).abs());
            assert!((fine[(i, j)] - fine[(j, i)]).abs() < 1e-13);
        }
    }
    assert!(
        worst < 1e-10,
        "default quadrature differs from the refined one by {:.3e}",
        worst
    );
}

// ------------------------------------------------------------------------------------------------
// Oracle (PySCF) and physical requirements for molecules
// ------------------------------------------------------------------------------------------------

struct System {
    name: String,
    atoms: Vec<Atom>,
    charge: i32,
    spin: usize,
    basis: String,
    energy: f64,
    gradient: Vec<[f64; 3]>,
    eigs: Vec<f64>,
}

fn load_systems() -> Vec<System> {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/ecp_reference.json")).unwrap();
    let mut out = Vec::new();
    for (name, s) in data["systems"].as_object().unwrap() {
        let basis = s["basis"].as_str().unwrap().to_string();
        let mut atoms: Vec<Atom> = s["geometry_bohr"]
            .as_str()
            .unwrap()
            .split(';')
            .map(|tok| {
                let p: Vec<&str> = tok.split_whitespace().collect();
                Atom::new(
                    p[0],
                    [
                        p[1].parse().unwrap(),
                        p[2].parse().unwrap(),
                        p[3].parse().unwrap(),
                    ],
                )
            })
            .collect();
        assign_builtin_ecps(&mut atoms, &basis);
        let vec = |v: &serde_json::Value| -> Vec<f64> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        };
        out.push(System {
            name: name.clone(),
            atoms,
            charge: s["charge"].as_i64().unwrap() as i32,
            spin: s["spin_2s"].as_u64().unwrap() as usize,
            basis,
            energy: s["energy"].as_f64().unwrap(),
            gradient: s["gradient"]
                .as_array()
                .unwrap()
                .iter()
                .map(|g| {
                    let g = vec(g);
                    [g[0], g[1], g[2]]
                })
                .collect(),
            eigs: vec(&s["ecp_generalized_eigenvalues"]),
        });
    }
    out
}

#[test]
fn ecp_matrix_generalized_eigenvalues_match_pyscf() {
    for sys in load_systems() {
        let basis = build_basis(&sys.atoms, &get_builtin_basis(&sys.basis).unwrap());
        let n = basis.len();
        let mut s = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                s[(i, j)] = contracted_overlap(&basis[i], &basis[j]);
            }
        }
        let v = ecp_one_electron_matrix(&sys.atoms, &basis);
        let x = s.inv_sqrt();
        let a = x.transpose().dot(&v).dot(&x);
        let (evals, _) = a.eigh();
        assert_eq!(evals.len(), sys.eigs.len(), "{}", sys.name);
        let worst = evals
            .iter()
            .zip(&sys.eigs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!(
            "{:11} n_ao = {:3}  max |d eig| = {:.2e}",
            sys.name, n, worst
        );
        assert!(
            worst < 5e-8,
            "{}: ECP generalized eigenvalues differ from PySCF by {:.3e}",
            sys.name,
            worst
        );
    }
}

fn scf_energy_and_gradient(sys: &System, atoms: &[Atom]) -> (f64, Vec<[f64; 3]>) {
    let basis = build_basis(atoms, &get_builtin_basis(&sys.basis).unwrap());
    if sys.spin == 0 {
        let hf =
            run_scf_general(atoms, &basis, sys.charge, 300, 1e-12, ConvAlgo::Cdiis, None).unwrap();
        assert!(hf.converged, "{} did not converge", sys.name);
        (hf.total_energy, compute_rhf_gradient(atoms, &basis, &hf))
    } else {
        let uhf = run_uhf(atoms, &basis, sys.charge, sys.spin + 1, false, 300, 1e-12).unwrap();
        assert!(uhf.converged, "{} did not converge", sys.name);
        (uhf.total_energy, compute_uhf_gradient(atoms, &basis, &uhf))
    }
}

// Energy-only displacements must not recompute the analytical gradient being checked.
fn scf_energy(sys: &System, atoms: &[Atom]) -> f64 {
    let basis = build_basis(atoms, &get_builtin_basis(&sys.basis).unwrap());
    let (energy, converged) = if sys.spin == 0 {
        let hf =
            run_scf_general(atoms, &basis, sys.charge, 300, 1e-12, ConvAlgo::Cdiis, None).unwrap();
        (hf.total_energy, hf.converged)
    } else {
        let hf = run_uhf(atoms, &basis, sys.charge, sys.spin + 1, false, 300, 1e-12).unwrap();
        (hf.total_energy, hf.converged)
    };
    assert!(converged, "{} did not converge", sys.name);
    energy
}

#[test]
fn ecp_scf_energy_and_gradient_match_pyscf() {
    let mut failures = Vec::new();
    for sys in load_systems() {
        let (e, g) = scf_energy_and_gradient(&sys, &sys.atoms);
        let de = (e - sys.energy).abs();
        let dg = g
            .iter()
            .zip(&sys.gradient)
            .flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs()))
            .fold(0.0, f64::max);
        let gsum = (0..3)
            .map(|k| g.iter().map(|x| x[k]).sum::<f64>().abs())
            .fold(0.0, f64::max);
        println!(
            "{:11} dE = {:.2e}  dG = {:.2e}  |sum G| = {:.2e}",
            sys.name, de, dg, gsum
        );
        if de > 1e-8 || dg > 1e-6 || gsum > 1e-9 {
            failures.push(format!(
                "{}: dE={:.2e} dG={:.2e} sumG={:.2e}",
                sys.name, de, dg, gsum
            ));
        }
    }
    assert!(failures.is_empty(), "{:#?}", failures);
}

#[test]
fn ecp_energy_is_rotation_and_translation_invariant() {
    let sys = load_systems()
        .into_iter()
        .find(|s| s.name == "auh")
        .unwrap();
    let e0 = scf_energy(&sys, &sys.atoms);
    // rigid rotation about an off-origin axis plus translation
    let (ca, sa) = (0.7_f64.cos(), 0.7_f64.sin());
    let (cb, sb) = (1.9_f64.cos(), 1.9_f64.sin());
    let mut moved = sys.atoms.clone();
    for a in moved.iter_mut() {
        let p = a.position;
        let q1 = [ca * p[0] - sa * p[1], sa * p[0] + ca * p[1], p[2]];
        let q2 = [q1[0], cb * q1[1] - sb * q1[2], sb * q1[1] + cb * q1[2]];
        a.position = [q2[0] + 0.37, q2[1] - 1.1, q2[2] + 0.62];
    }
    let e1 = scf_energy(&sys, &moved);
    assert!(
        (e0 - e1).abs() < 1e-9,
        "energy changes under rigid motion by {:.3e}",
        (e0 - e1).abs()
    );
}

#[test]
fn ecp_gradient_matches_five_point_finite_differences_of_the_energy() {
    for name in ["hi", "agh_cation"] {
        let sys = load_systems().into_iter().find(|s| s.name == name).unwrap();
        let (_, g) = scf_energy_and_gradient(&sys, &sys.atoms);
        let h = 1e-3;
        let mut worst = 0.0_f64;
        for (atom, axis) in [(0usize, 2usize), (1, 2), (1, 0)] {
            let energy = |dx: f64| {
                let mut a = sys.atoms.clone();
                a[atom].position[axis] += dx;
                scf_energy(&sys, &a)
            };
            let fd = (-energy(2.0 * h) + 8.0 * energy(h) - 8.0 * energy(-h) + energy(-2.0 * h))
                / (12.0 * h);
            worst = worst.max((fd - g[atom][axis]).abs());
        }
        assert!(
            worst < 5e-7,
            "{}: analytic ECP gradient differs from FD by {:.3e}",
            name,
            worst
        );
    }
}

#[test]
fn ecp_gradient_function_obeys_translational_invariance_and_is_linear_in_the_density() {
    let sys = load_systems()
        .into_iter()
        .find(|s| s.name == "agh")
        .unwrap();
    let basis = build_basis(&sys.atoms, &get_builtin_basis(&sys.basis).unwrap());
    let n = basis.len();
    let mut p = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let v = ((i * 31 + j * 17) % 13) as f64 / 13.0 - 0.5;
            p[(i, j)] = v;
            p[(j, i)] = v;
        }
    }
    let g = ecp_gradient(&sys.atoms, &basis, &p);
    for k in 0..3 {
        let sum: f64 = g.iter().map(|x| x[k]).sum();
        assert!(
            sum.abs() < 1e-11,
            "sum of ECP forces along axis {}: {:.3e}",
            k,
            sum
        );
    }
    let g2 = ecp_gradient(&sys.atoms, &basis, &p.scale(2.5));
    for a in 0..g.len() {
        for k in 0..3 {
            assert!((g2[a][k] - 2.5 * g[a][k]).abs() < 1e-12 * (1.0 + g[a][k].abs()));
        }
    }
}

#[test]
fn ecp_electron_count_and_nuclear_repulsion_use_effective_charges() {
    let sys = load_systems().into_iter().find(|s| s.name == "hi").unwrap();
    let basis = build_basis(&sys.atoms, &get_builtin_basis(&sys.basis).unwrap());
    let hf = run_scf_general(&sys.atoms, &basis, 0, 200, 1e-11, ConvAlgo::Cdiis, None).unwrap();
    assert_eq!(
        hf.n_occ, 13,
        "HI/def2-SVP/ECP28: (25 + 1) / 2 valence electron pairs"
    );
    assert!((hf.nuclear_repulsion_energy - 25.0 / 3.0).abs() < 1e-12);
    assert!(
        (hf.density_matrix.dot(&hf.overlap_matrix).trace() - 26.0).abs() < 1e-9,
        "Tr(PS) = number of valence electrons"
    );
}
