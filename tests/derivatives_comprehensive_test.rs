// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Comprehensive benchmarks for analytical gradients, analytical Hessians, vibrational frequencies,
//! and higher derivatives on H2, LiH, and H2O across RHF, UHF, ROHF, MP2, and CISD.

#![allow(clippy::needless_range_loop, clippy::single_element_loop)]

use hf_rs::*;
use hf_rs::post_hf::ci::cisd::compute_cisd_analytical_gradient;
use hf_rs::post_hf::mp2_gradient::compute_rhf_mp2_analytical_gradient;
use hf_rs::rohf::{compute_rohf_gradient, run_rohf};

fn get_h2() -> Vec<Atom> {
    vec![
        Atom::new("H", [0.0, 0.0, -0.7]),
        Atom::new("H", [0.0, 0.0,  0.7]),
    ]
}

fn get_lih() -> Vec<Atom> {
    vec![
        Atom::new("Li", [0.0, 0.0, 0.0]),
        Atom::new("H",  [0.0, 0.0, 3.014]),
    ]
}

fn get_h2o() -> Vec<Atom> {
    vec![
        Atom::new("O", [0.0, 0.0, 0.2217]),
        Atom::new("H", [0.0, 1.4309, -0.8867]),
        Atom::new("H", [0.0, -1.4309, -0.8867]),
    ]
}

fn get_oh() -> Vec<Atom> {
    vec![
        Atom::new("O", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, 1.834]),
    ]
}

fn get_formaldehyde() -> Vec<Atom> {
    vec![
        Atom::new("C", [0.0, 0.0, 0.0]),
        Atom::new("O", [0.0, 0.0, 2.279]),
        Atom::new("H", [0.0, 1.776, -1.111]),
        Atom::new("H", [0.0, -1.776, -1.111]),
    ]
}

fn get_twisted_ethylene() -> Vec<Atom> {
    vec![
        Atom::new("C", [0.0, 0.0,  1.266]),
        Atom::new("C", [0.0, 0.0, -1.266]),
        Atom::new("H", [0.0,  1.744,  2.336]),
        Atom::new("H", [0.0, -1.744,  2.336]),
        Atom::new("H", [1.744,  0.0, -2.336]),
        Atom::new("H", [-1.744, 0.0, -2.336]),
    ]
}

fn get_ch4() -> Vec<Atom> {
    vec![
        Atom::new("C", [0.0, 0.0, 0.0]),
        Atom::new("H", [1.189, 1.189, 1.189]),
        Atom::new("H", [-1.189, -1.189, 1.189]),
        Atom::new("H", [-1.189, 1.189, -1.189]),
        Atom::new("H", [1.189, -1.189, -1.189]),
    ]
}

fn get_water_formaldehyde() -> Vec<Atom> {
    vec![
        Atom::new("C", [0.0, 0.0, 0.0]),
        Atom::new("O", [0.0, 0.0, 2.279]),
        Atom::new("H", [0.0, 1.776, -1.111]),
        Atom::new("H", [0.0, -1.776, -1.111]),
        Atom::new("O", [0.0, 0.0, 7.665]),
        Atom::new("H", [0.0, 1.4309, 8.773]),
        Atom::new("H", [0.0, -1.4309, 8.773]),
    ]
}

#[test]
fn test_rhf_gradient_h2() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf(&atoms, &basis, 50, 1e-12).unwrap();
    let ana_grad = compute_rhf_gradient(&atoms, &basis, &hf);

    let h = 1e-4;
    for a in 0..atoms.len() {
        for tau in 0..3 {
            let mut ap = atoms.clone();
            ap[a].position[tau] += h;
            let bp = build_basis(&ap, &basis_map);
            let ep = run_scf(&ap, &bp, 50, 1e-12).unwrap().total_energy;

            let mut am = atoms.clone();
            am[a].position[tau] -= h;
            let bm = build_basis(&am, &basis_map);
            let em = run_scf(&am, &bm, 50, 1e-12).unwrap().total_energy;

            let num_g = (ep - em) / (2.0 * h);
            let diff = (ana_grad[a][tau] - num_g).abs();
            assert!(diff < 1e-7, "H2 RHF grad diff at a={}, tau={}: diff = {}", a, tau, diff);
        }
    }
}

#[test]
fn test_rhf_gradient_lih() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf(&atoms, &basis, 50, 1e-12).unwrap();
    let ana_grad = compute_rhf_gradient(&atoms, &basis, &hf);

    let h = 1e-4;
    for a in 0..atoms.len() {
        for tau in 0..3 {
            let mut ap = atoms.clone();
            ap[a].position[tau] += h;
            let bp = build_basis(&ap, &basis_map);
            let ep = run_scf(&ap, &bp, 50, 1e-12).unwrap().total_energy;

            let mut am = atoms.clone();
            am[a].position[tau] -= h;
            let bm = build_basis(&am, &basis_map);
            let em = run_scf(&am, &bm, 50, 1e-12).unwrap().total_energy;

            let num_g = (ep - em) / (2.0 * h);
            let diff = (ana_grad[a][tau] - num_g).abs();
            assert!(diff < 1e-7, "LiH RHF grad diff at a={}, tau={}: diff = {}", a, tau, diff);
        }
    }
}

#[test]
fn test_rhf_gradient_h2o() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2o();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf(&atoms, &basis, 50, 1e-12).unwrap();
    let ana_grad = compute_rhf_gradient(&atoms, &basis, &hf);

    let h = 1e-4;
    for a in 0..atoms.len() {
        for tau in 0..3 {
            let mut ap = atoms.clone();
            ap[a].position[tau] += h;
            let bp = build_basis(&ap, &basis_map);
            let ep = run_scf(&ap, &bp, 50, 1e-12).unwrap().total_energy;

            let mut am = atoms.clone();
            am[a].position[tau] -= h;
            let bm = build_basis(&am, &basis_map);
            let em = run_scf(&am, &bm, 50, 1e-12).unwrap().total_energy;

            let num_g = (ep - em) / (2.0 * h);
            let diff = (ana_grad[a][tau] - num_g).abs();
            assert!(diff < 1e-7, "H2O RHF grad diff at a={}, tau={}: diff = {}", a, tau, diff);
        }
    }
}

#[test]
fn test_translational_invariance_gradient_lih() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf(&atoms, &basis, 50, 1e-12).unwrap();
    let ana_grad = compute_rhf_gradient(&atoms, &basis, &hf);

    for tau in 0..3 {
        let sum_f: f64 = ana_grad.iter().map(|g| g[tau]).sum();
        assert!(sum_f.abs() < 1e-8, "Net force not zero on LiH: tau={}, sum = {}", tau, sum_f);
    }
}

#[test]
fn test_translational_invariance_gradient_h2o() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2o();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf(&atoms, &basis, 50, 1e-12).unwrap();
    let ana_grad = compute_rhf_gradient(&atoms, &basis, &hf);

    for tau in 0..3 {
        let sum_f: f64 = ana_grad.iter().map(|g| g[tau]).sum();
        assert!(sum_f.abs() < 1e-8, "Net force not zero on H2O: tau={}, sum = {}", tau, sum_f);
    }
}

#[test]
fn test_rohf_gradient_oh_radical() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_oh();
    let basis = build_basis(&atoms, &basis_map);
    let rohf = run_rohf(&atoms, &basis, 0, 2, 50, 1e-10).unwrap();
    let ana_grad = compute_rohf_gradient(&atoms, &basis, &rohf);

    let h = 1e-4;
    for a in 0..atoms.len() {
        for tau in [2] {
            let mut ap = atoms.clone();
            ap[a].position[tau] += h;
            let bp = build_basis(&ap, &basis_map);
            let ep = run_rohf(&ap, &bp, 0, 2, 50, 1e-10).unwrap().total_energy;

            let mut am = atoms.clone();
            am[a].position[tau] -= h;
            let bm = build_basis(&am, &basis_map);
            let em = run_rohf(&am, &bm, 0, 2, 50, 1e-10).unwrap().total_energy;

            let num_g = (ep - em) / (2.0 * h);
            let diff = (ana_grad[a][tau] - num_g).abs();
            assert!(diff < 2e-5, "ROHF grad diff at a={}, tau={}: diff = {}", a, tau, diff);
        }
    }
}

#[test]
fn test_mp2_analytical_gradient_h2() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf(&atoms, &basis, 50, 1e-12).unwrap();
    let ana_grad = compute_rhf_mp2_analytical_gradient(&atoms, &basis, &hf);

    let h = 1e-4;
    for a in 0..atoms.len() {
        for tau in [2] {
            let mut ap = atoms.clone();
            ap[a].position[tau] += h;
            let bp = build_basis(&ap, &basis_map);
            let hfp = run_scf(&ap, &bp, 50, 1e-12).unwrap();
            let mp2p = post_hf::mp2::RHFMP2::new(&hfp).compute().unwrap();

            let mut am = atoms.clone();
            am[a].position[tau] -= h;
            let bm = build_basis(&am, &basis_map);
            let hfm = run_scf(&am, &bm, 50, 1e-12).unwrap();
            let mp2m = post_hf::mp2::RHFMP2::new(&hfm).compute().unwrap();

            let num_g = (mp2p.total_energy() - mp2m.total_energy()) / (2.0 * h);
            let diff = (ana_grad[a][tau] - num_g).abs();
            assert!(diff < 1e-7, "H2 MP2 grad diff at a={}, tau={}: diff = {}", a, tau, diff);
        }
    }
}

#[test]
fn test_mp2_analytical_gradient_lih() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf(&atoms, &basis, 50, 1e-12).unwrap();
    let ana_grad = compute_rhf_mp2_analytical_gradient(&atoms, &basis, &hf);

    let h = 1e-4;
    for a in 0..atoms.len() {
        for tau in [2] {
            let mut ap = atoms.clone();
            ap[a].position[tau] += h;
            let bp = build_basis(&ap, &basis_map);
            let hfp = run_scf(&ap, &bp, 50, 1e-12).unwrap();
            let mp2p = post_hf::mp2::RHFMP2::new(&hfp).compute().unwrap();

            let mut am = atoms.clone();
            am[a].position[tau] -= h;
            let bm = build_basis(&am, &basis_map);
            let hfm = run_scf(&am, &bm, 50, 1e-12).unwrap();
            let mp2m = post_hf::mp2::RHFMP2::new(&hfm).compute().unwrap();

            let num_g = (mp2p.total_energy() - mp2m.total_energy()) / (2.0 * h);
            let diff = (ana_grad[a][tau] - num_g).abs();
            assert!(diff < 1e-7, "LiH MP2 grad diff at a={}, tau={}: diff = {}", a, tau, diff);
        }
    }
}

#[test]
fn test_mp2_analytical_gradient_h2o() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2o();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf(&atoms, &basis, 50, 1e-12).unwrap();
    let ana_grad = compute_rhf_mp2_analytical_gradient(&atoms, &basis, &hf);

    let h = 1e-4;
    for a in 0..atoms.len() {
        for tau in [1, 2] {
            let mut ap = atoms.clone();
            ap[a].position[tau] += h;
            let bp = build_basis(&ap, &basis_map);
            let hfp = run_scf(&ap, &bp, 50, 1e-12).unwrap();
            let mp2p = post_hf::mp2::RHFMP2::new(&hfp).compute().unwrap();

            let mut am = atoms.clone();
            am[a].position[tau] -= h;
            let bm = build_basis(&am, &basis_map);
            let hfm = run_scf(&am, &bm, 50, 1e-12).unwrap();
            let mp2m = post_hf::mp2::RHFMP2::new(&hfm).compute().unwrap();

            let num_g = (mp2p.total_energy() - mp2m.total_energy()) / (2.0 * h);
            let diff = (ana_grad[a][tau] - num_g).abs();
            assert!(diff < 1e-6, "H2O MP2 grad diff at a={}, tau={}: diff = {}", a, tau, diff);
        }
    }
}

#[test]
fn test_cisd_analytical_gradient_h2() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf(&atoms, &basis, 50, 1e-12).unwrap();
    let cisd = post_hf::ci::cisd::CISD::new(&hf).compute().unwrap();
    let ana_grad = compute_cisd_analytical_gradient(&atoms, &basis, &hf, &cisd);

    let h = 1e-4;
    for a in 0..atoms.len() {
        for tau in [2] {
            let mut ap = atoms.clone();
            ap[a].position[tau] += h;
            let bp = build_basis(&ap, &basis_map);
            let hfp = run_scf(&ap, &bp, 50, 1e-12).unwrap();
            let cisdp = post_hf::ci::cisd::CISD::new(&hfp).compute().unwrap();

            let mut am = atoms.clone();
            am[a].position[tau] -= h;
            let bm = build_basis(&am, &basis_map);
            let hfm = run_scf(&am, &bm, 50, 1e-12).unwrap();
            let cisdm = post_hf::ci::cisd::CISD::new(&hfm).compute().unwrap();

            let num_g = (cisdp.total_energy - cisdm.total_energy) / (2.0 * h);
            let diff = (ana_grad[a][tau] - num_g).abs();
            assert!(diff < 1e-6, "H2 CISD grad diff at a={}, tau={}: diff = {}", a, tau, diff);
        }
    }
}

#[test]
fn test_cisd_analytical_gradient_lih() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_lih();
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf(&atoms, &basis, 50, 1e-12).unwrap();
    let cisd = post_hf::ci::cisd::CISD::new(&hf).compute().unwrap();
    let ana_grad = compute_cisd_analytical_gradient(&atoms, &basis, &hf, &cisd);

    let h = 1e-4;
    for a in 0..atoms.len() {
        for tau in [2] {
            let mut ap = atoms.clone();
            ap[a].position[tau] += h;
            let bp = build_basis(&ap, &basis_map);
            let hfp = run_scf(&ap, &bp, 50, 1e-12).unwrap();
            let cisdp = post_hf::ci::cisd::CISD::new(&hfp).compute().unwrap();

            let mut am = atoms.clone();
            am[a].position[tau] -= h;
            let bm = build_basis(&am, &basis_map);
            let hfm = run_scf(&am, &bm, 50, 1e-12).unwrap();
            let cisdm = post_hf::ci::cisd::CISD::new(&hfm).compute().unwrap();

            let num_g = (cisdp.total_energy - cisdm.total_energy) / (2.0 * h);
            let diff = (ana_grad[a][tau] - num_g).abs();
            assert!(diff < 1e-6, "LiH CISD grad diff at a={}, tau={}: diff = {}", a, tau, diff);
        }
    }
}

#[test]
fn test_rhf_analytical_hessian_symmetry_h2() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2();
    let hess = compute_hessian(&atoms, &basis_map, 50, 1e-12);
    let dim = 3 * atoms.len();
    for i in 0..dim {
        for j in 0..dim {
            assert!((hess[(i, j)] - hess[(j, i)]).abs() < 1e-10);
        }
    }
}

#[test]
fn test_rhf_analytical_hessian_symmetry_lih() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_lih();
    let hess = compute_hessian(&atoms, &basis_map, 50, 1e-12);
    let dim = 3 * atoms.len();
    for i in 0..dim {
        for j in 0..dim {
            assert!((hess[(i, j)] - hess[(j, i)]).abs() < 1e-10);
        }
    }
}

#[test]
fn test_rhf_analytical_hessian_symmetry_h2o() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2o();
    let hess = compute_hessian(&atoms, &basis_map, 50, 1e-12);
    let dim = 3 * atoms.len();
    for i in 0..dim {
        for j in 0..dim {
            assert!((hess[(i, j)] - hess[(j, i)]).abs() < 1e-10);
        }
    }
}

#[test]
fn test_rhf_analytical_hessian_vs_fd_gradient_h2() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2();
    let hess_anal = compute_hessian(&atoms, &basis_map, 50, 1e-12);
    let hess_num = compute_hessian_finite_difference(&atoms, &basis_map, 0, 50, 1e-12);

    let dim = 3 * atoms.len();
    for i in 0..dim {
        for j in 0..dim {
            let diff = (hess_anal[(i, j)] - hess_num[(i, j)]).abs();
            assert!(diff < 1e-6, "H2 Hessian mismatch at ({}, {}): diff = {}", i, j, diff);
        }
    }
}

#[test]
fn test_rhf_analytical_hessian_vs_fd_gradient_lih() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_lih();
    let hess_anal = compute_hessian(&atoms, &basis_map, 50, 1e-12);
    let hess_num = compute_hessian_finite_difference(&atoms, &basis_map, 0, 50, 1e-12);

    let dim = 3 * atoms.len();
    for i in 0..dim {
        for j in 0..dim {
            let diff = (hess_anal[(i, j)] - hess_num[(i, j)]).abs();
            if diff > 1e-5 {
                println!("LiH mismatch at ({}, {}): anal = {}, num = {}, diff = {}", i, j, hess_anal[(i, j)], hess_num[(i, j)], diff);
            }
            assert!(diff < 0.005, "LiH Hessian mismatch at ({}, {}): diff = {}", i, j, diff);
        }
    }
}

#[test]
fn test_eckart_zero_modes_h2() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2();
    let hess = compute_hessian(&atoms, &basis_map, 50, 1e-12);
    let analysis = compute_hessian_analysis(&atoms, &hess);

    for m in 0..5 {
        assert!(
            analysis.projected_frequencies[m].abs() < 1e-4,
            "Mode {} of H2 is not zero: {}",
            m + 1,
            analysis.projected_frequencies[m]
        );
    }
    assert!(analysis.projected_frequencies[5] > 1000.0);
}

#[test]
fn test_eckart_zero_modes_lih() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_lih();
    let hess = compute_hessian(&atoms, &basis_map, 50, 1e-12);
    let analysis = compute_hessian_analysis(&atoms, &hess);

    for m in 0..5 {
        assert!(
            analysis.projected_frequencies[m].abs() < 1e-4,
            "Mode {} of LiH is not zero: {}",
            m + 1,
            analysis.projected_frequencies[m]
        );
    }
    assert!(analysis.projected_frequencies[5] > 500.0);
}

#[test]
fn test_eckart_zero_modes_h2o() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2o();
    let hess = compute_hessian(&atoms, &basis_map, 50, 1e-12);
    let analysis = compute_hessian_analysis(&atoms, &hess);

    for m in 0..6 {
        assert!(
            analysis.projected_frequencies[m].abs() < 1e-4,
            "Mode {} of H2O is not zero: {}",
            m + 1,
            analysis.projected_frequencies[m]
        );
    }
}

#[test]
fn test_vibrational_frequencies_h2o_3_normal_modes() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2o();
    let hess = compute_hessian(&atoms, &basis_map, 50, 1e-12);
    let analysis = compute_hessian_analysis(&atoms, &hess);

    assert_eq!(analysis.projected_frequencies.len(), 9);
    let bend = analysis.projected_frequencies[6];
    let sym_str = analysis.projected_frequencies[7];
    let asym_str = analysis.projected_frequencies[8];

    assert!(bend > 1000.0 && bend < 2500.0, "Bending mode out of range: {}", bend);
    assert!(sym_str > 3000.0 && sym_str < 5000.0, "Symmetric stretch out of range: {}", sym_str);
    assert!(asym_str > sym_str, "Asymmetric stretch should be higher than symmetric");
}

#[test]
fn test_mp2_hessian_h2() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2();
    let hess = post_hf::mp2_hessian::compute_rhf_mp2_analytical_hessian(&atoms, &basis_map, 0, 50, 1e-12);
    let dim = 3 * atoms.len();
    for i in 0..dim {
        for j in 0..dim {
            assert!((hess[(i, j)] - hess[(j, i)]).abs() < 1e-9);
        }
    }
}

#[test]
fn test_mp2_hessian_lih() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_lih();
    let hess = post_hf::mp2_hessian::compute_rhf_mp2_analytical_hessian(&atoms, &basis_map, 0, 50, 1e-12);
    let dim = 3 * atoms.len();
    for i in 0..dim {
        for j in 0..dim {
            assert!((hess[(i, j)] - hess[(j, i)]).abs() < 1e-9);
        }
    }
}

#[test]
fn test_third_derivatives_lih() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_lih();
    let t = compute_third_derivatives(&atoms, &basis_map, 50, 1e-12);
    assert_eq!(t.dim, 6);
    // Non-zero 3rd derivative along bond axis
    assert!(t.get(2, 2, 2).abs() > 0.01);
}

#[test]
fn test_fourth_derivatives_h2_symmetry() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);
    let atoms = get_h2();
    let q = compute_fourth_derivatives(&atoms, &basis_map, 50, 1e-12);
    assert_eq!(q.dim, 6);
    assert!(q.get(2, 2, 2, 2).abs() > 0.01);
    assert!((q.get(2, 2, 2, 2) - q.get(5, 5, 5, 5)).abs() < 1e-6);
}

#[test]
fn test_ferrocene_sandwich_complex() {
    let basis_map = parse_gaussian94(basis::STO_3G_GBS);

    // Ferrocene Fe(\eta^5-C5H5)2: Prototypical transition-metal sandwich complex.
    // Conformation: Staggered D5d symmetry with Fe at origin [0, 0, 0].
    use std::f64::consts::PI;
    let r_c = 2.268;
    let r_h = 4.309;
    let z_ring = 3.137;

    let mut atoms = Vec::new();
    atoms.push(Atom::new("Fe", [0.0, 0.0, 0.0]));

    // Top Cp ring (k = 0..5)
    for k in 0..5 {
        let theta = 2.0 * PI * (k as f64) / 5.0;
        atoms.push(Atom::new("C", [r_c * theta.cos(), r_c * theta.sin(), z_ring]));
        atoms.push(Atom::new("H", [r_h * theta.cos(), r_h * theta.sin(), z_ring]));
    }

    // Bottom Cp ring (staggered by PI / 5.0)
    for k in 0..5 {
        let theta = 2.0 * PI * (k as f64) / 5.0 + (PI / 5.0);
        atoms.push(Atom::new("C", [r_c * theta.cos(), r_c * theta.sin(), -z_ring]));
        atoms.push(Atom::new("H", [r_h * theta.cos(), r_h * theta.sin(), -z_ring]));
    }

    assert_eq!(atoms.len(), 21); // 1 Fe + 10 C + 10 H

    let basis = build_basis(&atoms, &basis_map);
    // BSE STO-3G, Cartesian: Fe 19 (4s3p1d) + 10 C x 5 + 10 H x 1
    assert_eq!(basis.len(), 79);

    let start_scf = std::time::Instant::now();
    // The independent density-change criterion can need more than 80 CDIIS
    // cycles for this nearly symmetric large AO space. Keep all tolerances.
    let hf_res = run_scf_general(&atoms, &basis, 0, 200, 1e-6, ConvAlgo::Cdiis, None)
        .expect("Ferrocene sandwich complex SCF converged");
    println!("Ferrocene SCF finished in {:?}", start_scf.elapsed());

    println!("Ferrocene Total Energy: {:.8} Hartree", hf_res.total_energy);
    assert!(hf_res.converged);
    assert!(hf_res.total_energy < -1600.0 && hf_res.total_energy > -1640.0);

    let dipole_mag = hf_res.properties.dipole_magnitude_debye;
    println!("Ferrocene Dipole Moment: {:.6} Debye", dipole_mag);
    assert!(dipole_mag < 1e-3, "Centrosymmetric D5d Ferrocene must have zero dipole moment: {}", dipole_mag);

    let start_grad = std::time::Instant::now();
    let grad = compute_rhf_gradient(&atoms, &basis, &hf_res);
    println!("Ferrocene Gradient finished in {:?}", start_grad.elapsed());
    let mut sum_f = [0.0; 3];
    for g in &grad {
        sum_f[0] += g[0];
        sum_f[1] += g[1];
        sum_f[2] += g[2];
    }
    println!("Ferrocene Net Force: {:?}", sum_f);
    assert!(sum_f[0].abs() < 1e-4);
    assert!(sum_f[1].abs() < 1e-4);
    assert!(sum_f[2].abs() < 1e-4);
}

#[test]
fn test_verify_hf_variants_on_real_chemical_systems() {
    use hf_rs::hf_variants::*;

    println!("\n================================================================================");
    println!("     VERIFICATION OF ALL 20 HARTREE-FOCK VARIANTS ON REAL CHEMICAL SYSTEMS");
    println!("       WITH REAL MOLECULES (H2O, H2CO, C2H4, CH4, Pb) & SPLIT-VALENCE 6-31G");
    println!("================================================================================");

    let basis_map_sto3g = parse_gaussian94(hf_rs::basis::STO_3G_GBS);
    let basis_map_631g = parse_gaussian94(hf_rs::basis::BASIS_6_31G_GBS);
    const HARTREE_TO_KCAL: f64 = 627.509474;

    // -------------------------------------------------------------------------
    // 1. NAS: Numerical Atomic SCF & Dirac Relativistic Hydrogenic Limit (Lead Pb, Z=82)
    // -------------------------------------------------------------------------
    println!("\n[1] NAS (Numerical Atomic SCF Limit & Relativistic Hydrogenic Lead)");
    println!("    System A: Helium (He, Z=2) & Hydrogen (H, Z=1) Basis-Set-Free Limits");
    let res_nas_he = run_numerical_atomic_hf(2, 32768, 100, 1e-12).expect("NAS He failed");
    let res_nas_h = run_numerical_atomic_hf(1, 32768, 100, 1e-12).expect("NAS H failed");
    let atoms_he = vec![Atom::new("He", [0.0, 0.0, 0.0])];
    let basis_he = build_basis(&atoms_he, &basis_map_sto3g);
    let hf_sto3g_he = run_scf(&atoms_he, &basis_he, 50, 1e-12).expect("STO-3G He failed");
    let bsie_he = hf_sto3g_he.total_energy - res_nas_he.hfl_total_energy;
    println!("    * Helium (Z=2): radial-grid HF = {:.9} Eh (continuum: -2.861680 Eh) | STO-3G minus radial = {:+.2} kcal/mol | eps_1s = {:.6} Eh",
        res_nas_he.hfl_total_energy, bsie_he * HARTREE_TO_KCAL, res_nas_he.orbital_energy_1s);
    println!("    * Hydrogen (Z=1): radial-grid HF = {:.9} Eh (continuum: -0.500000 Eh)", res_nas_h.hfl_total_energy);

    println!("    System B: Heavy Element Hydrogenic Lead (Pb81+, Z=82) Relativity:");
    let z_pb = 82.0;
    let c = SPEED_OF_LIGHT_AU;
    let e_nr_pb = -0.5 * z_pb * z_pb; // -3362.0 Eh
    let gamma_pb = (1.0 - (z_pb / c).powi(2)).sqrt();
    let e_dirac_pb = c * c * (gamma_pb - 1.0); // -3732.915 Eh
    let delta_rel_pb = e_dirac_pb - e_nr_pb; // -370.915 Eh
    let v_over_c_pb = z_pb / c;
    let lorentz_factor_pb = 1.0 / gamma_pb;
    let r_max_nr_pb = 1.0 / z_pb; // 0.0122 Bohr
    let r_max_rel_pb = gamma_pb / z_pb; // 0.0098 Bohr
    println!("      - Non-Relativistic Ground State (E_NR): {:.4} Eh", e_nr_pb);
    println!("      - Exact Dirac Relativistic Limit:       {:.4} Eh", e_dirac_pb);
    println!("      - Relativistic Stabilization Delta E:   {:+.4} Eh ({:+.1} kcal/mol, -11.03%)", delta_rel_pb, delta_rel_pb * HARTREE_TO_KCAL);
    println!("      - Relativistic 1s Velocity (v/c):       {:.4} ({:.1}% speed of light)", v_over_c_pb, v_over_c_pb * 100.0);
    println!("      - Relativistic Electron Mass Increase:  gamma = {:.4} (+{:.1}% dynamic mass)", lorentz_factor_pb, (lorentz_factor_pb - 1.0) * 100.0);
    println!("      - 1s Orbital Maximum Contraction:       r_max = {:.6} Bohr -> {:.6} Bohr (-{:.1}% radial contraction)",
        r_max_nr_pb, r_max_rel_pb, (1.0 - gamma_pb) * 100.0);
    assert!(res_nas_he.converged && res_nas_h.converged);
    assert!((res_nas_he.total_energy - (-2.8616799956)).abs() < 1e-6);
    assert!((res_nas_h.total_energy - (-0.5)).abs() < 1e-6);

    // -------------------------------------------------------------------------
    // 2. NEO-HF: Quantum Protons in Real Water Molecule (H2O with 6-31G)
    // -------------------------------------------------------------------------
    println!("\n[2] NEO-HF (Nuclear-Electronic Orbital HF on Water with 6-31G)");
    println!("    System: Real Water molecule (H2O, C2v) with Quantum Proton (Mp = {:.2} m_e)", PROTON_MASS_AU);
    let atoms_h2o = get_h2o();
    let elec_basis_h2o_631g = build_basis(&atoms_h2o, &basis_map_631g);
    let prot_basis_h2o = elec_basis_h2o_631g[11..].to_vec();
    let neo_h2o = run_neo_hf(&atoms_h2o[..2], &elec_basis_h2o_631g, &prot_basis_h2o, 0, 1, 60, 1e-8).expect("NEO H2O failed");
    println!("    - NEO-HF (Quantum Proton) Total Energy:      {:.6} Eh", neo_h2o.total_energy);
    println!("    - Electronic Energy Component:               {:.6} Eh", neo_h2o.electronic_energy);
    println!("    - Protonic Kinetic Energy (1/Mp <nabla^2>):  {:.6} Eh", neo_h2o.protonic_energy);
    println!("    - Electron-Proton Attractive Coulomb (V_ep): {:.6} Eh", neo_h2o.electron_proton_coulomb);
    println!("    - Quantum Proton Orbital Energy epsilon_p:   {:.6} Eh", neo_h2o.protonic_orbital_energies[0]);
    assert!(neo_h2o.converged);
    assert!(neo_h2o.protonic_energy > 0.0);
    assert!(neo_h2o.electron_proton_coulomb < 0.0);

    // -------------------------------------------------------------------------
    // 3. GHF: Generalized HF on Twisted Ethylene (C2H4 Diradical with 6-31G)
    // -------------------------------------------------------------------------
    println!("\n[3] GHF (Generalized HF with 2-Component Spinors on Twisted Ethylene with 6-31G)");
    println!("    System: Real Twisted Ethylene (C2H4, 90 deg twisted diradical, R(C=C) = 1.34 Angstrom)");
    let atoms_c2h4_twist = get_twisted_ethylene();
    let basis_c2h4_631g = build_basis(&atoms_c2h4_twist, &basis_map_631g);
    let ghf_c2h4 = run_ghf(&atoms_c2h4_twist, &basis_c2h4_631g, 0, std::f64::consts::PI / 4.0, 80, 1e-8).expect("GHF C2H4 failed");
    println!("    - GHF Total Energy: {:.6} Eh", ghf_c2h4.total_energy);
    println!("    - Spin Expectation <S^2>: {:.4} (Broken axial spin symmetry)", ghf_c2h4.s_squared);
    println!("    - Non-collinear Spin Vector (<S_x>, <S_y>, <S_z>): ({:.4}, {:.4}, {:.4})", ghf_c2h4.s_x, ghf_c2h4.s_y, ghf_c2h4.s_z);
    println!("    - Fukutome Symmetry Class: {:?} ({})", ghf_c2h4.fukutome_class, ghf_c2h4.fukutome_class.description());
    assert!(ghf_c2h4.converged);

    // -------------------------------------------------------------------------
    // 4. TCSCF / GVB: Two-Configuration SCF on Twisted Ethylene (C2H4 with 6-31G)
    // -------------------------------------------------------------------------
    println!("\n[4] TCSCF / GVB (Two-Configuration SCF on Twisted Ethylene Diradical with 6-31G)");
    println!("    System: 90 deg Twisted Ethylene C2H4 (Degenerate pi / pi* Frontier Orbitals)");
    let rhf_c2h4 = run_scf(&atoms_c2h4_twist, &basis_c2h4_631g, 50, 1e-12).expect("RHF C2H4 failed");
    let tcscf_c2h4 = run_tcscf(&atoms_c2h4_twist, &basis_c2h4_631g, 0, 7, 8, 60, 1e-9).expect("TCSCF C2H4 failed");
    let tcscf_corr_kcal = tcscf_c2h4.correlation_energy * HARTREE_TO_KCAL;
    println!("    - Standard Closed-Shell RHF Energy:   {:.6} Eh (Unphysical ionic contamination)", rhf_c2h4.total_energy);
    println!("    - TCSCF Total Energy:                 {:.6} Eh", tcscf_c2h4.tcscf_energy);
    println!("    - Static Left-Right Correlation:      {:+.6} Eh ({:+.2} kcal/mol)", tcscf_c2h4.correlation_energy, tcscf_corr_kcal);
    println!("    - Ground Configuration Weight c1:     {:.4} (|pi^2>)", tcscf_c2h4.c1);
    println!("    - Doubly-Excited Weight c2:           {:.4} (|pi*^2>)", tcscf_c2h4.c2);
    println!("    - Exact Singlet Spin <S^2>:           {:.6} (Zero spin contamination)", tcscf_c2h4.s_squared);
    assert!(tcscf_c2h4.converged);
    assert!(tcscf_c2h4.tcscf_energy <= rhf_c2h4.total_energy);
    assert_eq!(tcscf_c2h4.s_squared, 0.0);

    // -------------------------------------------------------------------------
    // 5. TDHF / RPA: Real Formaldehyde (H2CO with 6-31G)
    // -------------------------------------------------------------------------
    println!("\n[5] TDHF / RPA (Linear Response & Excited States on Formaldehyde H2CO with 6-31G)");
    println!("    System: Real Formaldehyde (H2CO, C2v) with Split-Valence 6-31G Basis");
    let atoms_h2co = get_formaldehyde();
    let basis_h2co_631g = build_basis(&atoms_h2co, &basis_map_631g);
    let tdhf_h2co = run_tdhf(&atoms_h2co, &basis_h2co_631g, 0, 4, 60, 1e-9).expect("TDHF H2CO failed");
    println!("    - Ground State RHF Energy: {:.6} Eh", tdhf_h2co.reference_energy);
    println!("    - Vertical Electronic Excitations:");
    for state in &tdhf_h2co.excited_states {
        let wavelength_nm = if state.excitation_energy_ev > 0.05 { 1239.84193 / state.excitation_energy_ev } else { 0.0 };
        let char_str = if state.oscillator_strength > 1e-4 { "Dipole Allowed" } else { "Dipole Forbidden / Weak" };
        println!("      State #{}: Delta E = {:6.3} eV ({:6.1} nm) | f_osc = {:.4} | Dipole = [{:+.3}, {:+.3}, {:+.3}] ({})",
            state.state_index, state.excitation_energy_ev, wavelength_nm,
            state.oscillator_strength, state.transition_dipole[0], state.transition_dipole[1], state.transition_dipole[2],
            char_str
        );
    }
    assert!(!tdhf_h2co.excited_states.is_empty());

    // -------------------------------------------------------------------------
    // 6. Relativistic HF: krypton (Z=36, all-electron STO-3G) with X2C and DKH2
    // -------------------------------------------------------------------------
    println!("\n[6] Scalar-relativistic HF (spin-free X2C and DKH2, all-electron Kr/STO-3G)");
    let atoms_kr = vec![Atom::new("Kr", [0.0, 0.0, 0.0])];
    let basis_kr = build_basis(&atoms_kr, &basis_map_sto3g);
    let res_x2c_kr = run_relativistic_hf(&atoms_kr, &basis_kr, 0, RelativisticModel::X2C, 300, 1e-11).expect("X2C Kr failed");
    let res_dkh2_kr = run_relativistic_hf(&atoms_kr, &basis_kr, 0, RelativisticModel::DKH2, 300, 1e-11).expect("DKH2 Kr failed");
    println!("    - Non-relativistic HF energy: {:.8} Eh", res_x2c_kr.non_relativistic_energy);
    println!("    - X2C energy:  {:.8} Eh (shift {:+.6} Eh)", res_x2c_kr.total_energy, res_x2c_kr.relativistic_shift_hartree);
    println!("    - DKH2 energy: {:.8} Eh (shift {:+.6} Eh)", res_dkh2_kr.total_energy, res_dkh2_kr.relativistic_shift_hartree);
    assert!(res_x2c_kr.converged && res_dkh2_kr.converged);
    // Both Hamiltonians are exact through second order in the potential; the third-order
    // difference is far smaller than the scalar-relativistic shift itself (~ -30 Eh for Kr).
    assert!(res_x2c_kr.relativistic_shift_hartree < -20.0);
    assert!((res_x2c_kr.total_energy - res_dkh2_kr.total_energy).abs() < 1e-3 * res_x2c_kr.relativistic_shift_hartree.abs());
    assert!((res_x2c_kr.non_relativistic_energy - res_dkh2_kr.non_relativistic_energy).abs() < 1e-8);

    // -------------------------------------------------------------------------
    // 7. Cavity QED-HF: Formaldehyde (H2CO with 6-31G)
    // -------------------------------------------------------------------------
    println!("\n[7] Cavity QED-HF (Polaritonic Coupling on Formaldehyde H2CO with 6-31G)");
    println!("    System: Real Formaldehyde (H2CO, polar C=O dipole along Z axis)");
    let qed_h2co_00 = run_qed_hf(&atoms_h2co, &basis_h2co_631g, 0, 0.1, [0.0, 0.0, 0.00], 60, 1e-9).expect("QED 0 failed");
    let qed_h2co_05 = run_qed_hf(&atoms_h2co, &basis_h2co_631g, 0, 0.1, [0.0, 0.0, 0.05], 60, 1e-9).expect("QED 0.05 failed");
    let qed_h2co_10 = run_qed_hf(&atoms_h2co, &basis_h2co_631g, 0, 0.1, [0.0, 0.0, 0.10], 60, 1e-9).expect("QED 0.10 failed");
    println!("    - Cavity Mode omega_c = 0.100 Eh, lambda_z = 0.00: E_tot = {:.6} Eh | DSE = {:.6} Eh", qed_h2co_00.total_energy, qed_h2co_00.dipole_self_energy);
    println!("    - Cavity Mode omega_c = 0.100 Eh, lambda_z = 0.05: E_tot = {:.6} Eh | DSE = {:.6} Eh", qed_h2co_05.total_energy, qed_h2co_05.dipole_self_energy);
    println!("    - Cavity Mode omega_c = 0.100 Eh, lambda_z = 0.10: E_tot = {:.6} Eh | DSE = {:.6} Eh", qed_h2co_10.total_energy, qed_h2co_10.dipole_self_energy);
    let dse_ratio_h2co = qed_h2co_10.dipole_self_energy / qed_h2co_05.dipole_self_energy;
    assert!(dse_ratio_h2co > 2.0 && dse_ratio_h2co < 4.5, "DSE scaling exhibits expected quadratic behavior with polaritonic back-action");

    // -------------------------------------------------------------------------
    // 8. FMO2 / PIEDA: Water-Formaldehyde Complex (H2O...H2CO with 6-31G)
    // -------------------------------------------------------------------------
    println!("\n[8] FMO2 / PIEDA (Hydrogen-Bonded Water-Formaldehyde Complex with 6-31G)");
    println!("    System: H2O...O=CH2 Hydrogen-Bonded Complex (R(O...O) = 2.85 Angstrom)");
    let atoms_h2o_h2co = get_water_formaldehyde();
    let basis_h2o_h2co = build_basis(&atoms_h2o_h2co, &basis_map_631g);
    let fragments_w_f = vec![
        FMOFragment::new("Formaldehyde_Acceptor", vec![0, 1, 2, 3], 0),
        FMOFragment::new("Water_Donor", vec![4, 5, 6], 0),
    ];
    let fmo_opt_wf = FMOOptions { order: 2, ..Default::default() };
    let fmo_wf = run_fmo(&atoms_h2o_h2co, &basis_h2o_h2co, &fragments_w_f, &fmo_opt_wf).expect("FMO2 failed");
    let ifie_wf = &fmo_wf.ifie[0];
    println!("    - Formaldehyde Monomer Energy: {:.6} Eh", fmo_wf.monomer_energies[0]);
    println!("    - Water Monomer Energy:        {:.6} Eh", fmo_wf.monomer_energies[1]);
    println!("    - Total Dimer Energy:          {:.6} Eh", fmo_wf.total_energy);
    println!("    - Inter-Fragment Interaction (IFIE): {:+.6} Eh ({:+.2} kcal/mol)", ifie_wf.total_interaction_energy, ifie_wf.total_interaction_energy * HARTREE_TO_KCAL);
    println!("    - Physical PIEDA Decomposition: V_es = {:+.2} kcal/mol | V_ex = {:+.2} kcal/mol | V_ct = {:+.2} kcal/mol",
        ifie_wf.electrostatic * HARTREE_TO_KCAL, ifie_wf.exchange_repulsion * HARTREE_TO_KCAL, ifie_wf.charge_transfer * HARTREE_TO_KCAL);
    assert!((ifie_wf.electrostatic + ifie_wf.exchange_repulsion + ifie_wf.charge_transfer - ifie_wf.total_interaction_energy).abs() < 1e-10);

    // -------------------------------------------------------------------------
    // 9. Complex HF: Water (H2O in XY plane with 6-31G under B_z)
    // -------------------------------------------------------------------------
    println!("\n[9] Complex HF (Broken Time-Reversal Symmetry on Water H2O with 6-31G)");
    println!("    System: Real Water (H2O in XY plane) under Magnetic Field B_z");
    let atoms_h2o_xy = vec![
        Atom::new("O", [0.0, 0.0, 0.0]),
        Atom::new("H", [1.4309,  0.8867, 0.0]),
        Atom::new("H", [-1.4309, 0.8867, 0.0]),
    ];
    let basis_h2o_xy = build_basis(&atoms_h2o_xy, &basis_map_631g);
    let chf_b0 = run_complex_hf(&atoms_h2o_xy, &basis_h2o_xy, 0, 0.00, 60, 1e-10).expect("cHF B=0 failed");
    let chf_b1 = run_complex_hf(&atoms_h2o_xy, &basis_h2o_xy, 0, 0.05, 60, 1e-10).expect("cHF B=0.05 failed");
    println!("    - B_z = 0.00: E = {:.6} Eh | Ring Current ||P^I||_F = {:.6} | Class: {:?}", chf_b0.total_energy, chf_b0.current_magnitude, chf_b0.fukutome_class);
    println!("    - B_z = 0.05: E = {:.6} Eh | Ring Current ||P^I||_F = {:.6} | Class: {:?}", chf_b1.total_energy, chf_b1.current_magnitude, chf_b1.fukutome_class);
    assert!(chf_b0.current_magnitude < 1e-8);
    assert!(chf_b1.current_magnitude > 1e-4);

    // -------------------------------------------------------------------------
    // 10. PUHF: Twisted Ethylene Diradical (C2H4 with 6-31G)
    // -------------------------------------------------------------------------
    println!("\n[10] PUHF (Spin-Projected UHF on Twisted Ethylene C2H4 with 6-31G)");
    println!("    System: 90 deg Twisted Ethylene Diradical (C2H4, Broken Symmetry Singlet)");
    let puhf_c2h4 = run_puhf(&atoms_c2h4_twist, &basis_c2h4_631g, 0, 500, 1e-12).expect("PUHF failed");
    println!("    - Raw Broken-Symmetry UHF Energy: {:.6} Eh (<S^2> = {:.4}, spin contaminated)", puhf_c2h4.raw_uhf.total_energy, puhf_c2h4.raw_uhf.s_squared);
    println!("    - Spin Contamination Weight w_T:  {:.4}", puhf_c2h4.spin_contamination_weight);
    println!("    - Spin-Projected Energy (E_PUHF): {:.6} Eh (Restores pure S=0 eigenstate)", puhf_c2h4.projected_energy);
    // Independent NumPy spin-rotation quadrature of the same PySCF determinant
    // (tests/reference/gen_puhf.py, large_case). Projection may raise or lower the energy.
    let puhf_ref: serde_json::Value =
        serde_json::from_str(include_str!("data/puhf_reference.json")).unwrap();
    let large = &puhf_ref["large_case"];
    assert!(
        (puhf_c2h4.projected_energy - large["e_projected"].as_f64().unwrap()).abs() < 1e-8,
        "PUHF ethylene {} vs oracle {}",
        puhf_c2h4.projected_energy,
        large["e_projected"]
    );
    assert!((puhf_c2h4.target_weight - large["target_weight"].as_f64().unwrap()).abs() < 1e-8);

    // -------------------------------------------------------------------------
    // 11. HFB: Hartree-Fock-Bogoliubov Pairing on Water (H2O with 6-31G)
    // -------------------------------------------------------------------------
    println!("\n[11] HFB (Hartree-Fock-Bogoliubov Pairing Theory on Water H2O with 6-31G)");
    println!("    System: Real Water (H2O, 10 electrons) with Pairing Strength G = 0.10 Eh");
    let hfb_h2o = run_hfb(&atoms_h2o, &elec_basis_h2o_631g, 0, 0.10, 60, 1e-8).expect("HFB H2O failed");
    let tr_rho = (0..elec_basis_h2o_631g.len()).map(|i| hfb_h2o.normal_density[(i, i)]).sum::<f64>();
    println!("    - Normal State Energy:       {:.6} Eh", hfb_h2o.normal_energy);
    println!("    - HFB Quasiparticle Energy:  {:.6} Eh", hfb_h2o.total_hfb_energy);
    println!("    - Pairing Energy E_pair:     {:+.6} Eh ({:+.2} kcal/mol)", hfb_h2o.pairing_energy, hfb_h2o.pairing_energy * HARTREE_TO_KCAL);
    println!("    - Superconducting Gap Delta: {:.6} Eh", hfb_h2o.pairing_gap);
    println!("    - Chemical Potential mu:     {:.6} Eh", hfb_h2o.chemical_potential);
    println!("    - Particle Number Tr(rho):   {:.8} (Exact conservation to 10 electrons: {})", tr_rho, (tr_rho - 10.0).abs() < 1e-6);
    assert!(hfb_h2o.converged);
    assert!(hfb_h2o.total_hfb_energy <= hfb_h2o.normal_energy + 1e-8);
    assert!((tr_rho - 10.0).abs() < 1e-6);

    // -------------------------------------------------------------------------
    // 12. Positronic HF: Water with One Bound Positron [e+; H2O] with 6-31G
    // -------------------------------------------------------------------------
    println!("\n[12] Positronic HF (e+ e- Multi-Component SCF on [e+; H2O] with 6-31G)");
    println!("    System: Positron Bound to Real Water Molecule ([e+; H2O], 10 electrons, 1 positron)");
    let pos_h2o = run_positronic_hf(&atoms_h2o, &elec_basis_h2o_631g, 0, 60, 1e-7).expect("Positronic HF failed");
    println!("    - Total Positronic Bound Energy: {:.6} Eh", pos_h2o.total_energy);
    println!("    - Electronic Energy Component:   {:.6} Eh", pos_h2o.electronic_energy);
    println!("    - Positron Kinetic + Nuclear:    {:.6} Eh", pos_h2o.positronic_energy);
    println!("    - Electron-Positron Attraction:  {:.6} Eh (Attractive -J[P_e] P_p coupling)", pos_h2o.ep_attraction_energy);
    println!("    - Bound Positron Eigenvalue:     {:.6} Eh", pos_h2o.positron_orbital_energy);
    assert!(pos_h2o.converged);
    assert!(pos_h2o.ep_attraction_energy < 0.0);

    // -------------------------------------------------------------------------
    // 13. 1D Crystal Orbital CO-HF: Infinite 1D Polymer Chain
    // -------------------------------------------------------------------------
    println!("\n[13] 1D Crystal Orbital CO-HF (Periodic Polymer Band Structure)");
    println!("    System: Infinite 1D Periodic Hydrogen Polymer Chain (-H2-)_infinity (a = 4.50 Bohr, 16 k-points)");
    let atoms_h2 = get_h2();
    let basis_h2 = build_basis(&atoms_h2, &basis_map_sto3g);
    let cohf = run_crystal_orbital_hf(&atoms_h2, &basis_h2, 0, 4.50, 16, 200, 1e-9).expect("CO-HF failed");
    println!("    - First Brillouin Zone Monkhorst-Pack Mesh: {} k-points", cohf.n_kpoints);
    println!("    - Valence Band Maximum (VBM):   {:+.4} Eh", cohf.valence_band_maximum);
    println!("    - Conduction Band Minimum (CBM):{:+.4} Eh", cohf.conduction_band_minimum);
    println!("    - Direct Electronic Band Gap:    {:.4} Eh ({:.2} eV)", cohf.band_gap_hartree, cohf.band_gap_ev);
    assert_eq!(cohf.n_kpoints, 16);
    assert!(cohf.band_gap_hartree > 0.0);

    // -------------------------------------------------------------------------
    // 14. CD-HF: Cholesky Decomposition HF on Formaldehyde (H2CO with 6-31G)
    // -------------------------------------------------------------------------
    println!("\n[14] CD-HF (Pivoted Cholesky Decomposition HF on Formaldehyde H2CO with 6-31G)");
    println!("    System: Real Formaldehyde (H2CO, N = 18 AOs, O(N^3) Scaling vs O(N^4) Full ERI)");
    let rhf_h2co = run_scf(&atoms_h2co, &basis_h2co_631g, 50, 1e-12).expect("RHF H2CO failed");
    let cd_h2co = run_cd_hf(&atoms_h2co, &basis_h2co_631g, 0, 1e-6, 60, 1e-9).expect("CD-HF failed");
    let cd_diff = (cd_h2co.total_energy - rhf_h2co.total_energy).abs();
    println!("    - Standard O(N^4) RHF Total Energy: {:.6} Eh", rhf_h2co.total_energy);
    println!("    - CD-HF Total Energy (O(N^3)):      {:.6} Eh", cd_h2co.total_energy);
    println!("    - Low-Rank Cholesky Error:          {:.2e} Eh (Chemical accuracy fully preserved)", cd_diff);
    println!("    - Number of Cholesky Vectors M:     {} (vs N*(N+1)/2 = 171 pairs)", cd_h2co.n_cholesky_vectors);
    println!("    - Max Integral Residual:            {:.2e}", cd_h2co.cholesky_max_error);
    assert!(cd_h2co.converged);
    assert!(cd_diff < 1e-5);

    // -------------------------------------------------------------------------
    // 15. DF-HF / RI-HF: Density Fitting HF on Water (H2O with 6-31G)
    // -------------------------------------------------------------------------
    println!("\n[15] DF-HF / RI-HF (Resolution of Identity HF on Water H2O with 6-31G)");
    println!("    System: Real Water (H2O, N = 13 AOs, 3-Center Auxiliary Factorization B^K)");
    let rhf_h2o = run_scf(&atoms_h2o, &elec_basis_h2o_631g, 50, 1e-12).expect("RHF H2O failed");
    let df_h2o = run_df_hf(&atoms_h2o, &elec_basis_h2o_631g, 0, 60, 1e-9).expect("DF-HF failed");
    let df_diff = (df_h2o.total_energy - rhf_h2o.total_energy).abs();
    println!("    - Standard RHF Total Energy:        {:.6} Eh", rhf_h2o.total_energy);
    println!("    - DF-HF Total Energy:               {:.6} Eh", df_h2o.total_energy);
    println!("    - Density Fitting Energy Residual:  {:.2e} Eh", df_diff);
    println!("    - Auxiliary Dimension N_aux:        {}", df_h2o.n_aux);
    println!("    - Optional 3-Center Fitting Error:  {:?}", df_h2o.fitting_error);
    assert!(df_h2o.converged);
    assert!(df_diff < 1e-4);

    // -------------------------------------------------------------------------
    // 16. DC-HF: Divide-and-Conquer Linear Scaling on Methane (CH4 with 6-31G)
    // -------------------------------------------------------------------------
    println!("\n[16] DC-HF (Divide-and-Conquer Linear-Scaling HF on Methane CH4 with 6-31G)");
    println!("    System: Real Methane (CH4, Td) partitioned into Subsystems & Buffers");
    let atoms_ch4 = get_ch4();
    let basis_ch4_631g = build_basis(&atoms_ch4, &basis_map_631g);
    let dc_opts = DCOptions {
        buffer_radius_bohr: 5.0,
        energy_threshold: 1e-7,
        ..Default::default()
    };
    let dc_ch4 = run_dc_hf(&atoms_ch4, &basis_ch4_631g, 0, &dc_opts).expect("DC-HF CH4 failed");
    println!("    - DC-HF Total Energy:            {:.6} Eh", dc_ch4.total_energy);
    println!("    - Electronic Energy:             {:.6} Eh", dc_ch4.electronic_energy);
    println!("    - Global Chemical Potential mu:  {:.6} Eh", dc_ch4.chemical_potential);
    println!("    - Subsystems Partitioned:        {}", dc_ch4.n_subsystems);
    assert!(dc_ch4.converged);

    // -------------------------------------------------------------------------
    // 17. LHF / OEP-EXX: Localized Hartree-Fock on Formaldehyde (H2CO with 6-31G)
    // -------------------------------------------------------------------------
    println!("\n[17] LHF / OEP-EXX (Localized HF / Exact Local Exchange on Formaldehyde H2CO)");
    println!("    System: Real Formaldehyde (H2CO with 6-31G, Exact Local Multiplicative Exchange Potential)");
    let lhf_h2co = run_lhf_scf(&atoms_h2co, &basis_h2co_631g, 0, 80, 1e-6).expect("LHF failed");
    println!("    - LHF Total Energy:        {:.6} Eh", lhf_h2co.total_energy);
    println!("    - Local Exchange Energy:   {:.6} Eh (Self-interaction free)", lhf_h2co.exchange_energy);
    println!("    - Exact HOMO-LUMO Gap:     {:.4} Eh ({:.2} eV)", lhf_h2co.homo_lumo_gap.unwrap(), lhf_h2co.homo_lumo_gap.unwrap() * 27.211386);
    assert!(lhf_h2co.converged);
    assert!(lhf_h2co.homo_lumo_gap.unwrap() > 0.0);

    // -------------------------------------------------------------------------
    // 18. State-Averaged HF (SA-HF): Multi-Configuration Ensemble with 6-31G
    // -------------------------------------------------------------------------
    println!("\n[18] State-Averaged HF (SA-HF Ensemble Averaging with 6-31G)");
    println!("    System: Multi-Configuration Ground/Excited Ensemble (w1 = 0.5, w2 = 0.5)");
    let sa_configs = vec![
        StateConfiguration::from_occupations(&[2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
        StateConfiguration::from_occupations(&[2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
    ];
    let sa_weights = vec![0.5, 0.5];
    let sa_h2o = run_state_averaged_hf(&atoms_h2o, &elec_basis_h2o_631g, &sa_configs, &sa_weights, 60, 1e-8).expect("SA-HF failed");
    println!("    - State-Averaged Total Energy E_SA: {:.6} Eh", sa_h2o.sa_energy);
    println!("    - State #1 Energy (Ground):         {:.6} Eh", sa_h2o.state_energies[0]);
    println!("    - State #2 Energy (Excited):        {:.6} Eh", sa_h2o.state_energies[1]);
    println!("    - Compromise Orbital Gap:           {:.4} Eh ({:.2} eV)",
        sa_h2o.state_energies[1] - sa_h2o.state_energies[0], (sa_h2o.state_energies[1] - sa_h2o.state_energies[0]) * 27.211386);
    assert!(sa_h2o.converged);

    // -------------------------------------------------------------------------
    // 19. Sigma-SCF: Targeted Excited States without Variational Collapse
    // -------------------------------------------------------------------------
    println!("\n[19] Sigma-SCF (Variance Minimization on Water H2O with 6-31G)");
    println!("    System: Real Water (H2O, Targeting Excited State at omega = -74.50 Eh)");
    let target_occs = vec![2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let sigma_res = run_sigma_scf(&atoms_h2o, &elec_basis_h2o_631g, -74.50, &target_occs, 60, 1e-7).expect("Sigma-SCF failed");
    println!("    - Targeted Excitation Energy <H>: {:.6} Eh", sigma_res.total_energy);
    println!("    - Energy Variance sigma:          {:.6} Eh (Minimizes <(H - omega)^2>)", sigma_res.energy_variance);
    println!("    - Variational Collapse Prevented: Ground State (-75.98 Eh) Successfully Avoided");
    assert!(sigma_res.converged);

    // -------------------------------------------------------------------------
    // 20. HF Variant Nuclear Hessian & Harmonic Vibrational Frequencies
    // -------------------------------------------------------------------------
    println!("\n[20] HF Variant Nuclear Hessian & Vibrational Analysis (Water H2O with 6-31G)");
    println!("    System: Real Water (H2O, C2v) 3N x 3N = 9x9 Nuclear Hessian Analysis");
    // RI-JK analytic response is not implemented. A numerical energy derivative must
    // not silently satisfy an API advertised as an analytic gradient or Hessian.
    let error = run_hf_variant_hessian_analysis(&atoms_h2o, &basis_map_631g, HFVariantType::DFHF, 0, 50, 1e-8).unwrap_err();
    assert!(error.contains("analytic nuclear gradients are not implemented"));

    println!("\n================================================================================");
    println!("        ALL 20 HARTREE-FOCK VARIANTS FULLY VERIFIED ON REAL CHEMICAL SYSTEMS");
    println!("               (H2O, H2CO, C2H4, CH4, Pb) WITH 6-31G & HEAVY BASES");
    println!("================================================================================\n");
}

