use hf_rs::context::*;
use hf_rs::options::*;
use hf_rs::*;
use std::collections::BTreeMap;
use std::sync::Arc;

#[test]
fn every_registered_option_has_documented_defaults_and_default_roundtrips() {
    let catalog = hf_rs::api::method_catalog();
    let mut defaults = BTreeMap::new();
    let mut schemas = Vec::new();
    for id in hf_rs::registry::MethodId::ALL {
        schemas.extend(id.option_schemas());
    }
    for method in catalog.as_array().unwrap() {
        let keys = method["options"].as_array().unwrap();
        let details = method["option_details"].as_array().unwrap();
        assert_eq!(keys.len(), details.len());
        for entry in details {
            let key = entry["name"].as_str().unwrap();
            assert!(!entry["description"].as_str().unwrap().is_empty());
            assert!(!entry["type"].as_str().unwrap().is_empty());
            assert!(!entry["unit"].as_str().unwrap().is_empty());
            if !entry["default"].is_null() && !key.starts_with("scf_") && key != "broken_symmetry" {
                defaults.insert(key.to_string(), entry["default"].clone());
            }
        }
    }
    assert_eq!(
        PostHFOptions::from_map(&defaults, &schemas).unwrap(),
        PostHFOptions::default()
    );
}

#[test]
fn unequal_active_electron_and_orbital_counts_construct_the_correct_partition() {
    let partition =
        hf_rs::orbitals::OrbitalSpace::from_active_space(7, 3, 3, &ActiveSpace::new(1, 1, 3))
            .unwrap();
    assert_eq!(partition.n_frozen_core, 2);
    assert_eq!(partition.n_active_occ, 1);
    assert_eq!(partition.n_active_virt, 2);
    assert_eq!(partition.n_external_virt, 2);
}

#[test]
fn registry_names_roundtrip_and_unknown_methods_are_errors() {
    use hf_rs::registry::{GradientKind, MethodId};
    use std::str::FromStr;
    for &method in MethodId::ALL {
        assert_eq!(MethodId::from_str(method.name()).unwrap(), method);
    }
    assert_eq!(MethodId::from_str("uhf-mp2").unwrap(), MethodId::Mp2);
    assert!(matches!(
        MethodId::from_str("MP2_typo"),
        Err(HfError::UnknownMethod(_))
    ));
    assert!(PostHFType::parse("MP2_typo").is_err());
    let options = PostHFOptions::default();
    assert_eq!(
        MethodId::DlpnoCcsd.gradient_kind(RefKind::Rhf, &options),
        GradientKind::Unsupported
    );
    assert_eq!(
        MethodId::Mp2F12.gradient_kind(RefKind::Rhf, &options),
        GradientKind::Numerical
    );
}

#[test]
fn selected_state_energy_is_rebased_on_scf_and_invalid_roots_fail() {
    use hf_rs::output::*;
    let mut output = PostHFOutput::new("CASPT2", -75.0, -75.2).unwrap();
    output.components.insert("E_CASSCF".into(), -75.1);
    output.states.push(State {
        energy: -75.05,
        s_squared: Some(2.0),
        weight: 0.5,
        converged: true,
    });
    output.select_root(1).unwrap();
    assert!((output.e_corr + 0.05).abs() < 1e-12);
    output.validate().unwrap();
    assert!(output.select_root(2).is_err());
}

#[test]
fn open_shell_active_space_and_explicit_permutation_are_validated() {
    let active = ActiveSpace {
        n_alpha: 2,
        n_beta: 1,
        n_orb: 3,
        orbitals: Some(vec![4, 1, 3]),
    };
    let resolved = active.resolve(6, 3, 2).unwrap();
    assert_eq!(resolved.inactive, vec![0]);
    assert_eq!(resolved.active, vec![4, 1, 3]);
    assert_eq!(resolved.external, vec![2, 5]);
    assert_eq!(resolved.permutation(), vec![0, 4, 1, 3, 2, 5]);
    assert!(ActiveSpace::new(2, 2, 1).resolve(4, 2, 2).is_err());
    assert!(ActiveSpace::new(1, 1, 2).resolve(4, 3, 2).is_err());
    assert!(
        ActiveSpace {
            n_alpha: 1,
            n_beta: 1,
            n_orb: 2,
            orbitals: Some(vec![1, 1])
        }
        .resolve(4, 2, 2)
        .is_err()
    );
    assert_eq!(
        ActiveSpace::new(0, 0, 0).resolve(4, 2, 2).unwrap().inactive,
        vec![0, 1]
    );
}

#[test]
fn typed_options_reject_unknown_inapplicable_and_invalid_values() {
    let mut values = BTreeMap::new();
    values.insert("active_n_alpha".into(), serde_json::json!(2));
    values.insert("active_n_beta".into(), serde_json::json!(1));
    values.insert("active_n_orb".into(), serde_json::json!(3));
    values.insert("root".into(), serde_json::json!(1));
    values.insert("n_roots".into(), serde_json::json!(2));
    let options = PostHFOptions::from_map(&values, &[ACTIVE_OPTIONS]).unwrap();
    assert_eq!(options.active.unwrap().n_beta, 1);
    assert!(PostHFOptions::from_map(&values, &[]).is_err());
    values.insert("max_iter".into(), serde_json::json!(0));
    assert!(PostHFOptions::from_map(&values, &[ACTIVE_OPTIONS]).is_err());
    values.remove("max_iter");
    values.insert("max_itre".into(), serde_json::json!(100));
    assert!(PostHFOptions::from_map(&values, &[ACTIVE_OPTIONS]).is_err());
}

#[test]
fn lazy_integrals_share_storage_and_spin_blocks_reproduce_rhf_energy() {
    let atoms = [
        Atom::new("H", [0.1, -0.2, -0.7]),
        Atom::new("H", [-0.1, 0.1, 0.7]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_scf(&atoms, &basis, 100, 1e-13).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    assert!(Arc::ptr_eq(
        ctx.ints.full().unwrap(),
        ctx.ints.full().unwrap()
    ));
    assert!(Arc::ptr_eq(
        ctx.ints.full().unwrap(),
        ctx.ints.full_beta().unwrap()
    ));
    let spin = ctx.ints.spin_blocks().unwrap();
    let mo = ctx.ints.full().unwrap();
    for p in 0..4 {
        for q in 0..4 {
            for r in 0..4 {
                for s in 0..4 {
                    let expected = if p % 2 == r % 2 && q % 2 == s % 2 {
                        mo.get(p / 2, r / 2, q / 2, s / 2)
                    } else {
                        0.0
                    } - if p % 2 == s % 2 && q % 2 == r % 2 {
                        mo.get(p / 2, s / 2, q / 2, r / 2)
                    } else {
                        0.0
                    };
                    assert!((spin.get(p, q, r, s) - expected).abs() < 1e-13);
                }
            }
        }
    }
    let active = ctx.ints.active(&ActiveSpace::new(2, 2, 2));
    assert!(active.is_err());
    let empty = ctx.ints.active(&ActiveSpace::new(0, 0, 0)).unwrap();
    assert!((empty.core_energy - hf.total_energy).abs() < 1e-12);
    let all = ctx.ints.active(&ActiveSpace::new(1, 1, 2)).unwrap();
    assert!(Arc::ptr_eq(
        &all,
        &ctx.ints.active(&ActiveSpace::new(1, 1, 2)).unwrap()
    ));
    assert_eq!(all.eri.n_basis, 2);
}

#[test]
fn mixed_uhf_integrals_match_direct_ordered_ao_contraction_every_component() {
    let atoms = [
        Atom::new("H", [0.0, 0.0, -1.75]),
        Atom::new("H", [0.0, 0.0, 1.75]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_uhf(&atoms, &basis, 0, 1, true, 150, 1e-13).unwrap();
    let ctx = PostHFContext::from_uhf(&hf, &atoms, &basis, None).unwrap();
    let mixed = ctx.ints.cross().unwrap();
    let a = &hf.mo_coefficients_alpha;
    let b = &hf.mo_coefficients_beta;
    for p in 0..2 {
        for q in 0..2 {
            for r in 0..2 {
                for s in 0..2 {
                    let mut expected = 0.0;
                    for mu in 0..2 {
                        for nu in 0..2 {
                            for la in 0..2 {
                                for si in 0..2 {
                                    expected += a[(mu, p)]
                                        * a[(nu, q)]
                                        * b[(la, r)]
                                        * b[(si, s)]
                                        * hf.eri[scf::eri_index(mu, nu, la, si)];
                                }
                            }
                        }
                    }
                    assert!((mixed.get(p, q, r, s) - expected).abs() < 1e-12);
                }
            }
        }
    }
}
