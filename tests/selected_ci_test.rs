use hf_rs::Matrix;
use hf_rs::fci::engine::{FciHamiltonian, FciSolverOptions};
use hf_rs::fci::subspace::SubspaceHamiltonian;
use hf_rs::mo_integrals::MoEriTensor;
use hf_rs::options::PostHFOptions;
use hf_rs::post_hf::mr_dsrg::selected_ci::{SelectionKind, aci_importance, solve_selected};

fn model() -> (Matrix, MoEriTensor) {
    let n = 4;
    let mut h = Matrix::zeros(n, n);
    let np = n * (n + 1) / 2;
    let mut g = MoEriTensor {
        n_basis: n,
        n_pairs: np,
        data: vec![0.0; np * np],
    };
    for p in 0..n {
        for q in 0..n {
            h[(p, q)] = if p == q {
                -1.4 + 0.4 * p as f64
            } else {
                0.05 * ((p + q + 1) as f64).sin()
            };
        }
    }
    for pq in 0..np {
        for rs in 0..np {
            g.data[rs * np + pq] = (0..3)
                .map(|k| {
                    0.1 * (((k + 1) * (pq + 1)) as f64 * 0.37).sin()
                        * (((k + 1) * (rs + 1)) as f64 * 0.37).sin()
                })
                .sum();
        }
    }
    (h, g)
}

#[test]
fn finite_threshold_energies_corrections_and_every_rdm_component_match_independent_pyscf() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/selected_ci_reference.json")).unwrap();
    let (h, g) = model();
    for case in data["cases"].as_array().unwrap() {
        let kind = match case["kind"].as_str().unwrap() {
            "ACI" => SelectionKind::Aci,
            "ASCI" => SelectionKind::Asci,
            "HBCI" => SelectionKind::HeatBath,
            _ => unreachable!(),
        };
        let mut o = PostHFOptions::default();
        o.sci.selection_threshold = case["threshold"].as_f64().unwrap();
        o.sci.pt2_threshold = case["eps2"].as_f64().unwrap();
        o.sci.n_core_determinants = case["core"].as_u64().unwrap() as usize;
        o.sci.n_target_determinants = case["target"].as_u64().unwrap() as usize;
        o.sci.coarse_graining = case["gamma"].as_f64().unwrap();
        let result = solve_selected(&h, &g, 0.321, 2, 2, kind, &o).unwrap();
        let label = format!(
            "{kind:?}, sigma={}, target={}",
            o.sci.selection_threshold, o.sci.n_target_determinants
        );
        assert_eq!(
            result.determinants.len(),
            case["n_determinants"].as_u64().unwrap() as usize,
            "{label}"
        );
        assert!(
            (result.states[0].energy - case["energy_variational"].as_f64().unwrap()).abs() < 1e-10,
            "{label} Evar"
        );
        assert!(
            (result.corrections[0] - case["correction"].as_f64().unwrap()).abs() < 1e-10,
            "{label} correction: {} vs {}",
            result.corrections[0],
            case["correction"]
        );
        assert!(result.states[0].residual_norm < 1e-10);
        let engine = SubspaceHamiltonian::new(&h, &g, 0.321, result.determinants).unwrap();
        for (name, order) in [("one_rdm", 1), ("two_rdm", 2)] {
            let actual = engine.rdm(&result.states[0].vector, order).unwrap();
            let expected = case[name].as_array().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (i, (&a, b)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (a - b.as_f64().unwrap()).abs() < 1e-7,
                    "{label} {name}[{i}]: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn lower_two_state_root_remains_negative_for_intruders_and_avoids_cancellation() {
    for delta in [-3.0, -0.01, 0.0, 0.01, 3.0] {
        for v in [0.0, 0.003, -0.21] {
            let m = Matrix::from_vec(2, 2, vec![0.0, v, v, delta]);
            let expected = m.eigh().0[0];
            assert!((aci_importance(delta, v).unwrap() - expected).abs() < 1e-13);
        }
    }
    assert!((aci_importance(1e12, 1e-3).unwrap() + 1e-18).abs() < 1e-32);
    assert!(aci_importance(f64::NAN, 1.0).is_err());
}

#[test]
fn zero_threshold_spin_targeted_excited_roots_and_open_shell_limits_match_fci() {
    let (h, g) = model();
    for (na, nb, spin) in [(2, 2, 0.0), (2, 2, 2.0), (2, 1, 0.75)] {
        let fci = FciHamiltonian::new(&h, &g, na, nb, 0.321)
            .unwrap()
            .solve(&FciSolverOptions {
                n_roots: 2,
                target_s_squared: Some(spin),
                ..Default::default()
            })
            .unwrap();
        for kind in [
            SelectionKind::Aci,
            SelectionKind::Asci,
            SelectionKind::HeatBath,
        ] {
            let mut o = PostHFOptions {
                n_roots: 2,
                root: 1,
                ..Default::default()
            };
            o.cas.target_s_squared = Some(spin);
            o.sci.selection_threshold = 0.0;
            o.sci.n_core_determinants = 36;
            o.sci.n_target_determinants = 36;
            o.sci.pt2_threshold = 0.0;
            let r = solve_selected(&h, &g, 0.321, na, nb, kind, &o).unwrap();
            for (root, state) in r.states.iter().enumerate() {
                assert!(
                    (state.energy - fci.states[root].energy).abs() < 1e-10,
                    "{kind:?} ({na},{nb}) root {root}"
                );
                assert!((state.s_squared - spin).abs() < 1e-8);
                assert!(r.corrections[root].abs() < 1e-10);
            }
        }
    }
}

#[test]
fn unfinished_selection_and_invalid_storage_counts_return_errors() {
    let (h, g) = model();
    let mut o = PostHFOptions {
        max_iter: 1,
        ..Default::default()
    };
    o.sci.selection_threshold = 0.0;
    for kind in [
        SelectionKind::Aci,
        SelectionKind::Asci,
        SelectionKind::HeatBath,
    ] {
        assert!(solve_selected(&h, &g, 0.321, 2, 2, kind, &o).is_err());
    }
    o.max_iter = 100;
    o.sci.n_core_determinants = 20;
    o.sci.n_target_determinants = 10;
    assert!(solve_selected(&h, &g, 0.321, 2, 2, SelectionKind::Asci, &o).is_err());
    o.sci.max_determinants = 2;
    assert!(solve_selected(&h, &g, 0.321, 2, 2, SelectionKind::HeatBath, &o).is_err());
}

#[test]
fn typed_active_frozen_explicit_and_selected_root_outputs_match_casci() {
    use hf_rs::context::PostHFContext;
    use hf_rs::options::ActiveSpace;
    use hf_rs::registry::{MethodId, create_solver};
    use hf_rs::{Atom, build_basis, get_builtin_basis, run_scf};
    let atoms = vec![
        Atom::new("Li", [0.1, -0.2, 0.3]),
        Atom::new("H", [0.4, 0.1, 3.2]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_scf(&atoms, &basis, 150, 1e-14).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    for (active, frozen) in [
        (ActiveSpace::new(2, 2, 6), 0),
        (ActiveSpace::new(1, 1, 5), 1),
        (
            ActiveSpace {
                n_alpha: 1,
                n_beta: 1,
                n_orb: 2,
                orbitals: Some(vec![2, 4]),
            },
            1,
        ),
    ] {
        let mut o = PostHFOptions {
            active: Some(active),
            frozen_core: frozen,
            n_roots: 2,
            root: 1,
            ..Default::default()
        };
        // CASCI defaults to the minimum-spin sector; SCI can also return roots
        // of other spins unless this sector is explicitly requested.
        o.cas.target_s_squared = Some(0.0);
        o.sci.selection_threshold = 0.0;
        o.sci.n_core_determinants = 1000;
        o.sci.n_target_determinants = 1000;
        let cas =
            hf_rs::post_hf::casscf_engine::solve_cas(ctx.reference, &o, Some(&ctx.ints), false)
                .unwrap();
        let fci = &cas.output;
        let fci_ao = cas
            .mo_coefficients
            .dot(fci.one_rdm.as_ref().unwrap())
            .dot(&cas.mo_coefficients.transpose());
        for method in [MethodId::Aci, MethodId::Asci, MethodId::Hbci] {
            let out = create_solver(method, &ctx, &o).unwrap().compute().unwrap();
            assert_eq!(out.root, 1);
            assert_eq!(out.states.len(), 2);
            assert!(
                (out.e_total - fci.e_total).abs() < 1e-10,
                "{method} frozen={frozen}: {} vs {}, states {:?} vs {:?}",
                out.e_total,
                fci.e_total,
                out.states,
                fci.states
            );
            assert!((out.e_total - out.e_corr - hf.total_energy).abs() < 1e-12);
            let sci_ao = hf
                .mo_coefficients
                .dot(out.one_rdm.as_ref().unwrap())
                .dot(&hf.mo_coefficients.transpose());
            for p in 0..hf.mo_coefficients.cols {
                for q in 0..hf.mo_coefficients.cols {
                    assert!(
                        (sci_ao[(p, q)] - fci_ao[(p, q)]).abs() < 1e-7,
                        "{method} D[{p},{q}]"
                    );
                }
            }
        }
    }
    let o = PostHFOptions {
        active: Some(ActiveSpace::new(2, 2, 6)),
        frozen_core: 1,
        ..Default::default()
    };
    assert!(
        create_solver(MethodId::Hbci, &ctx, &o)
            .unwrap()
            .compute()
            .is_err()
    );
}

#[test]
fn pci_path_filtered_wavefunction_and_every_density_component_match_independent_propagation() {
    use hf_rs::post_hf::mr_dsrg::projector_ci::solve_projector;
    use hf_rs::post_hf::mr_dsrg::{PCIGenerator, PCIOptions};
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/projector_ci_reference.json")).unwrap();
    let (h, g) = model();
    let o = PostHFOptions {
        residual_tol: 1e-9,
        ..Default::default()
    };
    for case in data["cases"].as_array().unwrap() {
        let generator = match case["generator"].as_str().unwrap() {
            "WallChebyshev" => PCIGenerator::WallChebyshev,
            "Linear" => PCIGenerator::Linear,
            "Power" => PCIGenerator::Power,
            "Chebyshev" => PCIGenerator::Chebyshev,
            "ExpChebyshev" => PCIGenerator::ExpChebyshev,
            "Quadratic" => PCIGenerator::Quadratic,
            "Cubic" => PCIGenerator::Cubic,
            "Quartic" => PCIGenerator::Quartic,
            "Davidson" => PCIGenerator::Davidson,
            "Lanczos" => PCIGenerator::Lanczos,
            _ => unreachable!(),
        };
        let settings = PCIOptions {
            generator,
            spawning_threshold: case["eta"].as_f64().unwrap(),
            polynomial_order: case["order"].as_u64().unwrap() as usize,
            max_steps: 5000,
            compute_pt2: true,
        };
        let r = solve_projector(&h, &g, 0.321, 2, 2, &settings, &o).unwrap();
        let label = format!("{generator:?} eta={}", settings.spawning_threshold);
        assert!(
            (r.model.states[0].energy - case["energy_variational"].as_f64().unwrap()).abs() < 1e-10,
            "{label}"
        );
        assert!(
            (r.projective_energy - case["projective"].as_f64().unwrap()).abs() < 1e-8,
            "{label} projective"
        );
        assert!(
            (r.model.corrections[0] - case["correction"].as_f64().unwrap()).abs() < 1e-10,
            "{label} PT2"
        );
        let engine = SubspaceHamiltonian::new(&h, &g, 0.321, r.model.determinants).unwrap();
        for (name, order) in [("one_rdm", 1), ("two_rdm", 2)] {
            let actual = engine.rdm(&r.model.states[0].vector, order).unwrap();
            for (i, (&a, b)) in actual
                .iter()
                .zip(case[name].as_array().unwrap())
                .enumerate()
            {
                assert!(
                    (a - b.as_f64().unwrap()).abs() < 1e-7,
                    "{label} {name}[{i}]: {a} vs {b}"
                );
            }
        }
        if settings.spawning_threshold == 0.0 {
            assert!(
                r.model.states[0].residual_norm < o.residual_tol,
                "{label} physical residual"
            );
        } else {
            assert!(
                (r.model.states[0].energy - case["rediagonalized_energy"].as_f64().unwrap()).abs()
                    > 1e-7,
                "PCI's propagated vector must not be replaced by a final model eigenvector"
            );
        }
    }
}

#[test]
fn pci_returns_an_error_for_unconverged_propagation() {
    use hf_rs::post_hf::mr_dsrg::PCIOptions;
    use hf_rs::post_hf::mr_dsrg::projector_ci::solve_projector;
    let (h, g) = model();
    let settings = PCIOptions {
        max_steps: 1,
        spawning_threshold: 0.0,
        ..Default::default()
    };
    assert!(solve_projector(&h, &g, 0.321, 2, 2, &settings, &PostHFOptions::default()).is_err());
}

#[test]
fn typed_pci_frozen_explicit_active_and_generator_options_match_casci() {
    use hf_rs::context::PostHFContext;
    use hf_rs::options::{ActiveSpace, PCIGenerator};
    use hf_rs::registry::{MethodId, create_solver};
    use hf_rs::{Atom, build_basis, get_builtin_basis, run_scf};
    let atoms = vec![
        Atom::new("Li", [0.1, -0.2, 0.3]),
        Atom::new("H", [0.4, 0.1, 3.2]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_scf(&atoms, &basis, 150, 1e-14).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let mut o = PostHFOptions {
        active: Some(ActiveSpace {
            n_alpha: 1,
            n_beta: 1,
            n_orb: 2,
            // Reordered HOMO/virtual pair: its ground state is nondegenerate
            // and has overlap with the paired PCI trial. An all-virtual pi
            // pair can instead have a lower triplet outside that trial sector.
            orbitals: Some(vec![4, 1]),
        }),
        frozen_core: 1,
        max_iter: 3000,
        ..Default::default()
    };
    o.sci.selection_threshold = 0.0;
    let cas = hf_rs::post_hf::casscf_engine::solve_cas(ctx.reference, &o, Some(&ctx.ints), false)
        .unwrap();
    let expected = cas
        .mo_coefficients
        .dot(cas.output.one_rdm.as_ref().unwrap())
        .dot(&cas.mo_coefficients.transpose());
    for generator in [PCIGenerator::WallChebyshev, PCIGenerator::Davidson] {
        o.sci.projector_generator = generator;
        let out = create_solver(MethodId::Pci, &ctx, &o)
            .unwrap()
            .compute()
            .unwrap();
        assert!(
            (out.e_total - cas.output.e_total).abs() < 1e-10,
            "{generator:?}: {} vs {}, S2={:?}",
            out.e_total,
            cas.output.e_total,
            out.states[0].s_squared
        );
        let actual = hf
            .mo_coefficients
            .dot(out.one_rdm.as_ref().unwrap())
            .dot(&hf.mo_coefficients.transpose());
        for p in 0..actual.rows {
            for q in 0..actual.cols {
                assert!((actual[(p, q)] - expected[(p, q)]).abs() < 1e-7);
            }
        }
    }
    let mut map = std::collections::BTreeMap::new();
    map.insert(
        "projector_generator".into(),
        serde_json::json!("exp-chebyshev"),
    );
    map.insert("polynomial_order".into(), serde_json::json!(8));
    let parsed = PostHFOptions::from_map(&map, &MethodId::Pci.option_schemas()).unwrap();
    assert_eq!(parsed.sci.projector_generator, PCIGenerator::ExpChebyshev);
    assert_eq!(parsed.sci.polynomial_order, 8);
    map.insert("projector_generator".into(), serde_json::json!("typo"));
    assert!(PostHFOptions::from_map(&map, &MethodId::Pci.option_schemas()).is_err());
}
