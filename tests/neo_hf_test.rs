// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! NEO-HF and positronic HF against an independent PySCF-integral oracle (tests/reference/gen_neo_hf.py), the
//! point-proton / infinite-mass limit, the Hellmann-Feynman mass derivative, invariances and
//! density constraints.

use hf_rs::{
    Atom, ContractedGaussian, Matrix, NEOHFResult, NeoHfOptions, PositronicHfOptions, build_basis,
    get_builtin_basis, parse_gaussian94, run_neo_hf, run_neo_hf_with_options, run_positronic_hf,
    run_positronic_hf_with_options, run_scf,
};

fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/neo_hf_reference.json")).unwrap()
}

fn position(v: &serde_json::Value) -> [f64; 3] {
    std::array::from_fn(|k| v[k].as_f64().unwrap())
}

struct System {
    classical: Vec<Atom>,
    electronic: Vec<ContractedGaussian>,
    protonic: Vec<ContractedGaussian>,
    charge: i32,
    n_protons: usize,
}

fn system(case: &serde_json::Value, protonic_gbs: &str, transform: impl Fn([f64; 3]) -> [f64; 3]) -> System {
    let classical: Vec<Atom> = case["classical"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| Atom::new(a["symbol"].as_str().unwrap(), transform(position(&a["position"]))))
        .collect();
    let quantum: Vec<Atom> = case["quantum"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| Atom::new("H", transform(position(r))))
        .collect();
    // Electronic functions on every nucleus, including the quantum-proton sites.
    let all: Vec<Atom> = classical.iter().chain(&quantum).cloned().collect();
    System {
        electronic: build_basis(&all, &get_builtin_basis("6-31g").unwrap()),
        protonic: build_basis(&quantum, &parse_gaussian94(protonic_gbs)),
        charge: case["charge"].as_i64().unwrap() as i32,
        n_protons: quantum.len(),
        classical,
    }
}

fn solve(s: &System, options: &NeoHfOptions) -> NEOHFResult {
    run_neo_hf_with_options(&s.classical, &s.electronic, &s.protonic, s.charge, s.n_protons, options)
        .unwrap()
}

fn tight() -> NeoHfOptions {
    NeoHfOptions {
        max_iterations: 500,
        energy_tol: 1e-12,
        residual_tol: 1e-9,
        density_tol: 1e-11,
        ..Default::default()
    }
}

fn overlap(basis: &[ContractedGaussian]) -> Matrix {
    let n = basis.len();
    let mut s = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            s[(i, j)] = hf_rs::integrals::contracted_overlap(&basis[i], &basis[j]);
        }
    }
    s
}

fn max_difference(a: &Matrix, b: &serde_json::Value) -> f64 {
    let mut worst = 0f64;
    for i in 0..a.rows {
        for j in 0..a.cols {
            worst = worst.max((a[(i, j)] - b[i][j].as_f64().unwrap()).abs());
        }
    }
    worst
}

#[test]
fn neo_hf_matches_independent_oracle() {
    let data = data();
    let gbs = data["protonic_basis"].as_str().unwrap();
    let (mut worst_e, mut worst_d) = (0f64, 0f64);
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let s = system(case, gbs, |r| r);
        let r = solve(&s, &tight());
        assert!(r.converged);
        for (value, key) in [
            (r.total_energy, "energy"),
            (r.electronic_energy, "electronic_energy"),
            (r.protonic_energy, "protonic_energy"),
            (r.electron_proton_coulomb, "electron_proton_coulomb"),
            (r.protonic_kinetic_energy, "protonic_kinetic_energy"),
            (r.classical_nuc_repulsion, "nuclear_repulsion"),
        ] {
            let reference = case[key].as_f64().unwrap();
            worst_e = worst_e.max((value - reference).abs());
            assert!((value - reference).abs() < 1e-8, "{name}: {key} {value} vs {reference}");
        }
        let de = max_difference(&r.electronic_density, &case["electronic_density"]);
        let dp = max_difference(&r.protonic_density, &case["protonic_density"]);
        worst_d = worst_d.max(de).max(dp);
        assert!(de < 1e-6 && dp < 1e-6, "{name}: densities {de:e} {dp:e}");

        // Idempotent determinants with the right particle numbers.
        let (se, sp) = (overlap(&s.electronic), overlap(&s.protonic));
        let pe = &r.electronic_density;
        let pp = &r.protonic_density;
        let n_e = case["n_electrons"].as_f64().unwrap();
        assert!((pe.dot(&se).trace() - n_e).abs() < 1e-10);
        assert!((pp.dot(&sp).trace() - s.n_protons as f64).abs() < 1e-10);
        let ie = pe.dot(&se).dot(pe).sub(&pe.scale(2.0));
        let ip = pp.dot(&sp).dot(pp).sub(pp);
        assert!(ie.data.iter().chain(&ip.data).all(|x| x.abs() < 1e-10), "{name}: idempotency");
        assert!(r.electron_proton_coulomb < 0.0 && r.protonic_kinetic_energy > 0.0);
        assert!(r.residual_norm <= 1e-9);
    }
    println!("NEO-HF: max |dE| {worst_e:e}, max |dP| {worst_d:e}");
}

#[test]
fn point_proton_infinite_mass_limit_recovers_rhf() {
    // A single extremely tight s function and an extremely heavy particle reduce the quantum
    // proton to a classical point nucleus: smearing ~ <r^2> rho_e ~ 1e-10, kinetic 1.5 a/M.
    let gbs = "H     0\nS   1   1.00\n  1.0e10   1.0\n****\n";
    let classical = vec![Atom::new("F", [0.0, 0.0, 0.0])];
    let site = [0.1, -0.05, 1.733];
    let all = vec![classical[0].clone(), Atom::new("H", site)];
    let electronic = build_basis(&all, &get_builtin_basis("6-31g").unwrap());
    let protonic = build_basis(&[Atom::new("H", site)], &parse_gaussian94(gbs));
    let options = NeoHfOptions { proton_mass: 1e20, ..tight() };
    let neo = run_neo_hf_with_options(&classical, &electronic, &protonic, 0, 1, &options).unwrap();
    let rhf = run_scf(&all, &electronic, 200, 1e-12).unwrap();
    assert!(
        (neo.total_energy - rhf.total_energy).abs() < 1e-8,
        "NEO point limit {} vs RHF {}",
        neo.total_energy,
        rhf.total_energy
    );
    assert!(neo.protonic_kinetic_energy < 1e-9);
}

#[test]
fn mass_derivative_obeys_hellmann_feynman() {
    // dE/d(1/M) = <T_p> for the variational energy; central difference in 1/M.
    let data = data();
    let gbs = data["protonic_basis"].as_str().unwrap();
    for case in data["cases"].as_array().unwrap() {
        let s = system(case, gbs, |r| r);
        let mass = hf_rs::PROTON_MASS_AU;
        let at = |m: f64| solve(&s, &NeoHfOptions { proton_mass: m, ..tight() });
        let centre = at(mass);
        let h = 1e-3 / mass;
        let (plus, minus) = (at(1.0 / (1.0 / mass + h)), at(1.0 / (1.0 / mass - h)));
        let derivative = (plus.total_energy - minus.total_energy) / (2.0 * h);
        let expectation = centre.protonic_kinetic_energy * mass;
        assert!(
            (derivative - expectation).abs() < 1e-6 * expectation,
            "{}: dE/d(1/M) {derivative} vs <T> {expectation}",
            case["name"]
        );
    }
}

#[test]
fn energy_is_invariant_under_rigid_motion() {
    let data = data();
    let gbs = data["protonic_basis"].as_str().unwrap();
    let (c, s) = (0.6f64.cos(), 0.6f64.sin());
    for case in data["cases"].as_array().unwrap() {
        let reference = solve(&system(case, gbs, |r| r), &tight()).total_energy;
        // Rotation about an oblique axis (z then x) and a translation.
        let moved = system(case, gbs, |r| {
            let (x, y, z) = (c * r[0] - s * r[1], s * r[0] + c * r[1], r[2]);
            [x + 0.7, c * y - s * z - 1.1, s * y + c * z + 0.4]
        });
        let e = solve(&moved, &tight()).total_energy;
        assert!((e - reference).abs() < 1e-9, "{}: {e} vs {reference}", case["name"]);
    }
}

#[test]
fn invalid_neo_inputs_are_rejected() {
    let data = data();
    let gbs = data["protonic_basis"].as_str().unwrap();
    let case = &data["cases"][0];
    let s = system(case, gbs, |r| r);
    // HF with one quantum proton and charge +1 has an odd electron count.
    assert!(run_neo_hf(&s.classical, &s.electronic, &s.protonic, 1, 1, 100, 1e-10).is_err());
    assert!(run_neo_hf(&s.classical, &s.electronic, &s.protonic, 0, 0, 100, 1e-10).is_err());
    let massless = NeoHfOptions { proton_mass: 0.0, ..tight() };
    assert!(
        run_neo_hf_with_options(&s.classical, &s.electronic, &s.protonic, 0, 1, &massless).is_err()
    );
    // More quantum protons than protonic functions.
    assert!(run_neo_hf(&s.classical, &s.electronic, &s.protonic[..1], -1, 2, 100, 1e-10).is_err());
}

#[test]
fn positronic_hf_matches_independent_oracle() {
    let data = data();
    let mut worst = 0f64;
    for case in data["positronic_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let atoms: Vec<Atom> = case["classical"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| Atom::new(a["symbol"].as_str().unwrap(), position(&a["position"])))
            .collect();
        let basis = build_basis(&atoms, &get_builtin_basis("6-31g").unwrap());
        let charge = case["charge"].as_i64().unwrap() as i32;
        let options = PositronicHfOptions {
            max_iterations: 500,
            energy_tol: 1e-12,
            residual_tol: 1e-9,
            density_tol: 1e-11,
        };
        let r = run_positronic_hf_with_options(&atoms, &basis, &basis, charge, &options).unwrap();
        assert!(r.converged);
        for (value, key) in [
            (r.total_energy, "energy"),
            (r.electronic_energy, "electronic_energy"),
            (r.positronic_energy, "protonic_energy"),
            (r.ep_attraction_energy, "electron_proton_coulomb"),
            (r.positron_kinetic_energy, "protonic_kinetic_energy"),
            (r.nuclear_energy, "nuclear_repulsion"),
        ] {
            let reference = case[key].as_f64().unwrap();
            worst = worst.max((value - reference).abs());
            assert!((value - reference).abs() < 1e-8, "{name}: {key} {value} vs {reference}");
        }
        let de = max_difference(&r.electron_density, &case["electronic_density"]);
        let dp = max_difference(&r.positron_density, &case["protonic_density"]);
        assert!(de < 1e-6 && dp < 1e-6, "{name}: densities {de:e} {dp:e}");
        let s = overlap(&basis);
        assert!((r.positron_density.dot(&s).trace() - 1.0).abs() < 1e-10);
        // Same-basis positronic HF is NEO-HF with one unit-mass particle; the legacy driver
        // reaches the same state.
        let legacy = run_positronic_hf(&atoms, &basis, charge, 500, 1e-12).unwrap();
        assert!((legacy.total_energy - r.total_energy).abs() < 1e-9);
    }
    println!("positronic HF: max |dE| {worst:e}");
    // Odd electron counts are rejected rather than doubly occupying the last orbital.
    let h = vec![Atom::new("H", [0.0; 3])];
    let basis = build_basis(&h, &get_builtin_basis("6-31g").unwrap());
    assert!(run_positronic_hf(&h, &basis, 0, 100, 1e-10).is_err());
}
