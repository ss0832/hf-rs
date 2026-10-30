use hf_rs::{Atom, build_basis, parse_gaussian94};

#[test]
fn self_consistent_atomic_ensembles_match_pyscf_energy_and_entire_density() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/sad_reference.json")).unwrap();
    let map = parse_gaussian94(hf_rs::basis::STO_3G_GBS);
    for symbol in ["H", "C", "N", "O"] {
        let atom = Atom::new(symbol, [0.3, -0.2, 0.7]);
        let basis = build_basis(std::slice::from_ref(&atom), &map);
        let result = hf_rs::sad::atomic_fractional_hf(&atom, &basis).unwrap();
        let expected = &oracle["atoms"][symbol];
        assert!(
            (result.energy - expected["energy"].as_f64().unwrap()).abs() < 1e-9,
            "{symbol}: {} vs {}",
            result.energy,
            expected["energy"]
        );
        for i in 0..basis.len() {
            for j in 0..basis.len() {
                assert!(
                    (result.density[(i, j)] - expected["density"][i][j].as_f64().unwrap()).abs()
                        < 1e-8,
                    "{symbol} ({i},{j})"
                );
            }
        }
        assert!((result.occupations.iter().sum::<f64>() - atom.charge).abs() < 1e-12);
        assert!(result.iterations > 1);
    }
}

#[test]
fn molecular_sad_has_exact_charge_and_is_translation_invariant() {
    let atoms = [
        Atom::new("C", [0.1, 0.2, 0.3]),
        Atom::new("O", [0.3, -0.1, 2.3]),
    ];
    let map = parse_gaussian94(hf_rs::basis::STO_3G_GBS);
    let basis = build_basis(&atoms, &map);
    let (s, _, _, _) = hf_rs::scf::compute_one_electron_integrals(&atoms, &basis);
    let p = hf_rs::sad::compute_sad_density(&atoms, &basis, 1, &s).unwrap();
    assert!((p.dot(&s).trace() - 13.0).abs() < 1e-12);
    let mut shifted = atoms.clone();
    for a in &mut shifted {
        for k in 0..3 {
            a.position[k] += 1.3 * (k + 1) as f64;
        }
    }
    let basis = build_basis(&shifted, &map);
    let (s, _, _, _) = hf_rs::scf::compute_one_electron_integrals(&shifted, &basis);
    let again = hf_rs::sad::compute_sad_density(&shifted, &basis, 1, &s).unwrap();
    assert!(p.rmsd(&again) < 1e-14);
}
