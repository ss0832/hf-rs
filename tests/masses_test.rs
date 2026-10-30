// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::masses::{
    atomic_mass, atomic_mass_au, standard_atomic_weight,
    element_symbol, element_name, z_from_symbol, element_info,
    ELEMENTS, AMU_TO_AU, BOHR_TO_ANGSTROM, HARTREE_TO_EV, HARTREE_TO_J, SPEED_OF_LIGHT_AU,
};

#[test]
fn test_all_118_elements_coverage() {
    assert_eq!(ELEMENTS.len(), 118, "Must have exactly 118 elements");

    for z in 1..=118 {
        let el = element_info(z).expect("Element info must exist for 1..=118");
        assert_eq!(el.atomic_number, z);
        assert_eq!(element_symbol(z), el.symbol);
        assert_eq!(element_name(z), el.name);
        assert!(el.primary_mass_number >= z, "Mass number must be >= Z for Z={}", z);
        assert!(el.mass_amu > (z as f64) * 0.9, "Mass in amu must be reasonable for Z={}", z);
        assert!(el.standard_atomic_weight > 0.0);

        // Check inverse symbol lookup
        let found_z = z_from_symbol(el.symbol);
        assert_eq!(found_z, Some(z), "Symbol lookup failed for {}", el.symbol);

        // Case-insensitive lookup
        let lower = el.symbol.to_ascii_lowercase();
        assert_eq!(z_from_symbol(&lower), Some(z));
        let upper = el.symbol.to_ascii_uppercase();
        assert_eq!(z_from_symbol(&upper), Some(z));

        // Atomic mass in au vs amu
        let m_amu = atomic_mass(z);
        let m_au = atomic_mass_au(z);
        assert!((m_au - m_amu * AMU_TO_AU).abs() < 1e-10);
    }
}

#[test]
fn test_codata_fundamental_constants() {
    // 2022 CODATA recommended values
    assert!((AMU_TO_AU - 1822.888486209).abs() < 1e-6);
    assert!((BOHR_TO_ANGSTROM - 0.529177210903).abs() < 1e-8);
    assert!((HARTREE_TO_EV - 27.211386245988).abs() < 1e-8);
    assert!((HARTREE_TO_J - 4.3597447222071e-18).abs() < 1e-25);
    assert!((SPEED_OF_LIGHT_AU - 137.035999084).abs() < 1e-6);
}

#[test]
fn test_specific_elements_codata_nist() {
    // Hydrogen H-1: 1.00782503223 amu
    assert!((atomic_mass(1) - 1.00782503223).abs() < 1e-8);
    assert_eq!(element_symbol(1), "H");
    assert_eq!(element_name(1), "Hydrogen");

    // Helium He-4: 4.00260325413 amu
    assert!((atomic_mass(2) - 4.00260325413).abs() < 1e-8);

    // Carbon C-12: exactly 12.000000000 amu by definition
    assert!((atomic_mass(6) - 12.0).abs() < 1e-12);
    assert!((standard_atomic_weight(6) - 12.011).abs() < 0.01);

    // Nitrogen N-14: 14.00307400443 amu
    assert!((atomic_mass(7) - 14.00307400443).abs() < 1e-8);

    // Oxygen O-16: 15.99491461957 amu
    assert!((atomic_mass(8) - 15.99491461957).abs() < 1e-8);

    // Fluorine F-19: 18.99840316273 amu
    assert!((atomic_mass(9) - 18.99840316273).abs() < 1e-8);

    // Iron Fe-56: 55.9349363 amu
    assert!((atomic_mass(26) - 55.9349363).abs() < 1e-4);

    // Gold Au-197: 196.9665687 amu
    assert!((atomic_mass(79) - 196.9665687).abs() < 1e-4);

    // Uranium U-238: 238.0507882 amu
    assert!((atomic_mass(92) - 238.0507882).abs() < 1e-4);

    // Oganesson Og-294: 294.21392 amu
    assert!((atomic_mass(118) - 294.21392).abs() < 1e-4);
    assert_eq!(element_symbol(118), "Og");
    assert_eq!(element_name(118), "Oganesson");
}

#[test]
fn test_out_of_bounds_fallback() {
    assert_eq!(atomic_mass(0), 0.0);
    assert_eq!(atomic_mass(119), 238.0); // 2.0 * 119
    assert_eq!(element_symbol(0), "X");
    assert_eq!(element_symbol(119), "X");
    assert_eq!(element_name(0), "Unknown");
    assert_eq!(element_name(119), "Unknown");
    assert_eq!(element_info(0), None);
    assert_eq!(element_info(119), None);
}
