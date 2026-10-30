// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Wall-clock benchmark harness (release build, Cartesian basis functions).
//! Usage: cargo run --release --example bench

use std::time::Instant;
use hf_rs::{
    Atom, build_basis, get_builtin_basis, run_scf_general, ConvAlgo, RHFMP2, CCSD,
    compute_rhf_gradient, compute_hessian,
};
use hf_rs::post_hf::PostHFMethod;

fn water() -> Vec<Atom> {
    let a = 104.5_f64.to_radians() / 2.0;
    let r = 0.9578;
    vec![
        Atom::from_angstrom("O", [0.0, 0.0, 0.0]),
        Atom::from_angstrom("H", [r * a.sin(), 0.0, r * a.cos()]),
        Atom::from_angstrom("H", [-r * a.sin(), 0.0, r * a.cos()]),
    ]
}

fn benzene() -> Vec<Atom> {
    let (rc, rh) = (1.39, 2.47);
    let mut atoms = Vec::new();
    for k in 0..6 {
        let t = std::f64::consts::PI / 3.0 * k as f64;
        atoms.push(Atom::from_angstrom("C", [rc * t.cos(), rc * t.sin(), 0.0]));
    }
    for k in 0..6 {
        let t = std::f64::consts::PI / 3.0 * k as f64;
        atoms.push(Atom::from_angstrom("H", [rh * t.cos(), rh * t.sin(), 0.0]));
    }
    atoms
}

fn time<T>(label: &str, n_bf: usize, f: impl FnOnce() -> T) -> T {
    let t0 = Instant::now();
    let out = f();
    println!("{{\"case\": \"{}\", \"n_bf\": {}, \"seconds\": {:.3}}}", label, n_bf, t0.elapsed().as_secs_f64());
    out
}

fn main() {
    let ccpvdz = get_builtin_basis("cc-pvdz").unwrap();
    let b631gs = get_builtin_basis("6-31g*").unwrap();

    let w = water();
    let wb = build_basis(&w, &ccpvdz);
    let hf = time("h2o_ccpvdz_rhf", wb.len(), || run_scf_general(&w, &wb, 0, 100, 1e-10, ConvAlgo::Cdiis, None).unwrap());
    println!("# E_RHF(H2O/cc-pVDZ cart) = {:.10}", hf.total_energy);
    time("h2o_ccpvdz_mp2", wb.len(), || RHFMP2::new(&hf).compute().unwrap());
    time("h2o_ccpvdz_ccsd", wb.len(), || CCSD::new(&hf).compute().unwrap());

    let w6 = build_basis(&w, &b631gs);
    let hf6 = run_scf_general(&w, &w6, 0, 100, 1e-10, ConvAlgo::Cdiis, None).unwrap();
    time("h2o_631gs_rhf_gradient", w6.len(), || compute_rhf_gradient(&w, &w6, &hf6));
    time("h2o_631gs_rhf_hessian", w6.len(), || compute_hessian(&w, &b631gs, 100, 1e-10));

    let bz = benzene();
    let bzb = build_basis(&bz, &b631gs);
    let hfb = time("c6h6_631gs_rhf", bzb.len(), || run_scf_general(&bz, &bzb, 0, 100, 1e-10, ConvAlgo::Cdiis, None).unwrap());
    println!("# E_RHF(C6H6/6-31G* cart) = {:.10}", hfb.total_energy);
}
