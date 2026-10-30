// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Crystal-orbital RHF for chains against infinite-chain limits of finite PySCF chains
//! (tests/reference/gen_crystal_hf.py), supercell equivalence, the isolated-molecule limit,
//! near-field/multipole invariance of the lattice sums, band symmetry and input validation.

use hf_rs::{
    Atom, CoHfOptions, ContractedGaussian, build_basis, get_builtin_basis,
    run_crystal_orbital_hf, run_crystal_orbital_hf_with_options, run_scf,
};

fn data() -> serde_json::Value {
    serde_json::from_str(include_str!("data/crystal_hf_reference.json")).unwrap()
}

fn cell(case: &serde_json::Value) -> (Vec<Atom>, Vec<ContractedGaussian>, f64) {
    let atoms: Vec<Atom> = case["unit"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            Atom::new(
                a["symbol"].as_str().unwrap(),
                std::array::from_fn(|k| a["position"][k].as_f64().unwrap()),
            )
        })
        .collect();
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    (atoms, basis, case["lattice_constant"].as_f64().unwrap())
}

fn options(n_kpoints: usize) -> CoHfOptions {
    CoHfOptions {
        n_kpoints,
        energy_tol: 1e-12,
        density_tol: 1e-10,
        ..Default::default()
    }
}

fn case(name: &str) -> serde_json::Value {
    data()["cases"].as_array().unwrap().iter().find(|c| c["name"] == name).unwrap().clone()
}

#[test]
fn energies_per_cell_match_infinite_chain_limits() {
    for (name, n_kpoints) in [("H2_chain", 24), ("trans_polyacetylene", 32), ("water_chain", 16)] {
        let case = case(name);
        let (atoms, basis, a) = cell(&case);
        let r = run_crystal_orbital_hf_with_options(&atoms, &basis, 0, a, &options(n_kpoints)).unwrap();
        let reference = case["energy_per_cell"].as_f64().unwrap();
        // The oracle's own uncertainty: the change between its two largest-N estimates.
        let uncertainty = case["extrapolation_change"].as_f64().unwrap().abs();
        let tolerance = (10.0 * uncertainty).max(1e-8);
        println!(
            "{name}: E/cell {:.12} vs {reference:.12} (diff {:.2e}, tolerance {tolerance:.1e}), gap {:.4}",
            r.total_energy_per_cell,
            r.total_energy_per_cell - reference,
            r.band_gap_hartree
        );
        assert!((r.total_energy_per_cell - reference).abs() < tolerance, "{name}");
        let electrons: f64 = atoms.iter().map(|a| a.charge).sum();
        assert!((r.electrons_per_cell - electrons).abs() < 1e-9);
        assert!(r.band_gap_hartree > 0.0);
    }
}

#[test]
fn supercell_doubling_is_exact() {
    let case = case("H2_chain");
    let (atoms, basis, a) = cell(&case);
    // Equal real-space reach: exchange/density blocks to 23 primitive cells (window of 47 k points)
    // and 11 doubled cells (23 k points), plus every overlap and near-field block.
    let single = run_crystal_orbital_hf_with_options(&atoms, &basis, 0, a, &CoHfOptions { max_exchange_cells: 22, ..options(46) }).unwrap();
    let mut doubled = atoms.clone();
    doubled.extend(atoms.iter().map(|x| Atom::new(&x.symbol, [x.position[0], x.position[1], x.position[2] + a])));
    let basis2 = build_basis(&doubled, &get_builtin_basis("sto-3g").unwrap());
    let double = run_crystal_orbital_hf_with_options(&doubled, &basis2, 0, 2.0 * a, &CoHfOptions { max_exchange_cells: 11, ..options(23) }).unwrap();
    // 46 k points of the primitive cell fold exactly onto 23 k points of the doubled cell.
    assert!(
        (double.total_energy_per_cell - 2.0 * single.total_energy_per_cell).abs() < 1e-9,
        "{} vs 2 x {}",
        double.total_energy_per_cell,
        single.total_energy_per_cell
    );
    assert!((double.valence_band_maximum - single.valence_band_maximum).abs() < 1e-8);
    assert!((double.conduction_band_minimum - single.conduction_band_minimum).abs() < 1e-8, "CBM {} vs {}", double.conduction_band_minimum, single.conduction_band_minimum);
}

#[test]
fn widely_spaced_cells_reduce_to_the_molecule() {
    let atoms = vec![Atom::new("H", [0.0, 0.1, 0.0]), Atom::new("H", [0.0, 0.1, 1.4])];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let molecule = run_scf(&atoms, &basis, 200, 1e-13).unwrap();
    let r = run_crystal_orbital_hf_with_options(&atoms, &basis, 0, 40.0, &options(1)).unwrap();
    assert_eq!(r.overlap_range, 0);
    // Remaining interaction: quadrupole-quadrupole lattice sum ~ Q^2 / a^5 ~ 1e-8 Eh.
    assert!((r.total_energy_per_cell - molecule.total_energy).abs() < 1e-7);
}

#[test]
fn far_field_lattice_sums_are_converged_and_bands_are_symmetric() {
    // Dipoles perpendicular to the chain: the dipole-dipole tail is carried by the multipole
    // lattice sums, so moving the near/far boundary must not change the energy.
    let case = case("water_chain");
    let (atoms, basis, a) = cell(&case);
    let reference = run_crystal_orbital_hf_with_options(&atoms, &basis, 0, a, &options(12)).unwrap();
    for (near, order) in [(reference.near_field_cells + 5, 8), (reference.near_field_cells, 10)] {
        let o = CoHfOptions { near_field_cells: Some(near), multipole_order: order, ..options(12) };
        let r = run_crystal_orbital_hf_with_options(&atoms, &basis, 0, a, &o).unwrap();
        assert!(
            (r.total_energy_per_cell - reference.total_energy_per_cell).abs() < 1e-9,
            "near {near}, order {order}: {} vs {}",
            r.total_energy_per_cell,
            reference.total_energy_per_cell
        );
    }
    // Time reversal: eps(k) = eps(-k).
    let bands = &reference.band_structure;
    for p in bands {
        if let Some(q) = bands.iter().find(|q| (q.k_fractional + p.k_fractional).abs() < 1e-12) {
            for (x, y) in p.band_energies.iter().zip(&q.band_energies) {
                assert!((x - y).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn invalid_chains_are_rejected() {
    let atoms = vec![Atom::new("H", [0.0, 0.0, 0.0]), Atom::new("H", [0.0, 0.0, 1.4])];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    // Charged cell, odd electrons, aliasing k mesh, bad lattice constant.
    assert!(run_crystal_orbital_hf(&atoms, &basis, 2, 4.0, 24, 100, 1e-10).is_err());
    let single = vec![Atom::new("H", [0.0; 3])];
    let basis1 = build_basis(&single, &get_builtin_basis("sto-3g").unwrap());
    assert!(run_crystal_orbital_hf(&single, &basis1, 0, 4.0, 24, 100, 1e-10).is_err());
    let error = run_crystal_orbital_hf(&atoms, &basis, 0, 4.0, 4, 100, 1e-10).unwrap_err();
    assert!(error.contains("alias"), "{error}");
    assert!(run_crystal_orbital_hf(&atoms, &basis, 0, -1.0, 24, 100, 1e-10).is_err());
}
