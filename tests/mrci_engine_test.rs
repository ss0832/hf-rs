use hf_rs::context::{PostHFContext, Reference};
use hf_rs::options::{ActiveSpace, PostHFOptions};
use hf_rs::post_hf::ci::mrci_engine::compute_mrci;
use hf_rs::registry::{MethodId, create_solver};
use hf_rs::*;

fn setup() -> (Vec<Atom>, Vec<ContractedGaussian>, HFResult) {
    let atoms = vec![
        Atom::new("Li", [0.1, -0.2, 0.3]),
        Atom::new("H", [0.4, 0.1, 3.2]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_scf(&atoms, &basis, 150, 1e-14).unwrap();
    (atoms, basis, hf)
}
fn rotate_rdm2(gamma: &[f64], c: &Matrix) -> Vec<f64> {
    let n = c.cols;
    assert_eq!(c.rows, n);
    let mut result = gamma.to_vec();
    for leg in 0..4 {
        let mut next = vec![0.0; n.pow(4)];
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        let index = [p, q, r, s];
                        let mut sum = 0.0;
                        for a in 0..n {
                            let mut source = index;
                            source[leg] = a;
                            sum += c[(index[leg], a)]
                                * result
                                    [((source[0] * n + source[1]) * n + source[2]) * n + source[3]];
                        }
                        next[((p * n + q) * n + r) * n + s] = sum;
                    }
                }
            }
        }
        result = next;
    }
    result
}
#[test]
fn mrci_energies_reference_weights_plus_q_and_all_densities_match_independent_pyscf_projection() {
    let (atoms, basis, hf) = setup();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/mrci_reference.json")).unwrap();
    for (name, active, optimize, frozen) in [
        ("hf22", ActiveSpace::new(1, 1, 2), false, 0),
        ("casscf22", ActiveSpace::new(1, 1, 2), true, 0),
        ("cisd", ActiveSpace::new(0, 0, 0), false, 0),
        ("full", ActiveSpace::new(2, 2, 6), false, 0),
        ("frozen22", ActiveSpace::new(1, 1, 2), false, 1),
    ] {
        let mut options = PostHFOptions {
            active: Some(active),
            frozen_core: frozen,
            ..Default::default()
        };
        options.cas.optimize_orbitals = optimize;
        options.cas.orbital_gradient_tol = 1e-9;
        let result = compute_mrci(Reference::Rhf(&hf), &options, Some(&ctx.ints), false).unwrap();
        let expected = &data["cases"][name];
        assert_eq!(
            result.n_reference_determinants,
            expected["n_reference"].as_u64().unwrap() as usize
        );
        assert_eq!(
            result.determinants.len(),
            expected["n_determinants"].as_u64().unwrap() as usize
        );
        let state = &expected["states"][0];
        assert!(
            (result.output.e_total - state["energy"].as_f64().unwrap()).abs() < 1e-9,
            "{name} energy"
        );
        assert!(
            (result.reference_weights[0] - state["reference_weight"].as_f64().unwrap()).abs()
                < 1e-7,
            "{name} weight"
        );
        let c = &result.mo_coefficients;
        let ao = c
            .dot(result.output.one_rdm.as_ref().unwrap())
            .dot(&c.transpose());
        for p in 0..ao.rows {
            for q in 0..ao.cols {
                assert!(
                    (ao[(p, q)] - state["one_rdm_ao"][p][q].as_f64().unwrap()).abs() < 1e-7,
                    "{name} density {p},{q}"
                );
            }
        }
        let two = rotate_rdm2(&result.two_rdm(0).unwrap(), c);
        for (i, (&a, b)) in two
            .iter()
            .zip(state["two_rdm_ao"].as_array().unwrap())
            .enumerate()
        {
            assert!((a - b.as_f64().unwrap()).abs() < 1e-7, "{name} Gamma[{i}]");
        }
        let q = compute_mrci(Reference::Rhf(&hf), &options, Some(&ctx.ints), true).unwrap();
        assert!(
            (q.davidson_corrections[0] - state["delta_q"].as_f64().unwrap()).abs() < 1e-9,
            "{name} +Q"
        );
        assert!(
            (q.output.e_total - result.output.e_total - q.davidson_corrections[0]).abs() < 1e-12
        );
        let output = create_solver(MethodId::MrciQ, &ctx, &options)
            .unwrap()
            .compute()
            .unwrap();
        assert!((output.e_total - q.output.e_total).abs() < 1e-12);
        assert!((output.e_corr - output.e_total + hf.total_energy).abs() < 1e-12);
        if name == "full" {
            assert!(q.davidson_corrections[0].abs() < 1e-10);
        }
        if name == "cisd" {
            assert!(
                (CISD::new(&hf).compute().unwrap().total_energy - result.output.e_total).abs()
                    < 1e-10
            );
        }
    }
}
#[test]
fn selected_ci_roots_and_complete_active_limit_have_consistent_energies() {
    let (atoms, basis, hf) = setup();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/mrci_reference.json")).unwrap();
    let mut options = PostHFOptions {
        active: Some(ActiveSpace::new(1, 1, 2)),
        n_roots: 2,
        root: 1,
        ..Default::default()
    };
    options.cas.optimize_orbitals = false;
    let result = create_solver(MethodId::Mrci, &ctx, &options)
        .unwrap()
        .compute()
        .unwrap();
    assert_eq!(result.root, 1);
    assert!(
        (result.e_total
            - data["cases"]["hf22"]["states"][1]["energy"]
                .as_f64()
                .unwrap())
        .abs()
            < 1e-9
    );
    let cisd = create_solver(MethodId::Cisd, &ctx, &PostHFOptions::default())
        .unwrap()
        .compute()
        .unwrap();
    assert!(
        (cisd.e_total
            - data["cases"]["cisd"]["states"][0]["energy"]
                .as_f64()
                .unwrap())
        .abs()
            < 1e-10
    );
    options.active = Some(ActiveSpace::new(2, 2, 6));
    let mrci = create_solver(MethodId::Mrci, &ctx, &options)
        .unwrap()
        .compute()
        .unwrap();
    let fci = create_solver(MethodId::Fci, &ctx, &options)
        .unwrap()
        .compute()
        .unwrap();
    assert!((mrci.e_total - fci.e_total).abs() < 1e-10);
}

#[test]
fn open_shell_mrci_is_spin_complete_and_matches_pyscf_in_every_density_component() {
    let (atoms, basis, _) = setup();
    let hf = run_uhf(&atoms, &basis, 1, 2, false, 150, 1e-14).unwrap();
    let ctx = PostHFContext::from_uhf(&hf, &atoms, &basis, None).unwrap();
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/mrci_reference.json")).unwrap();
    let expected = &data["cases"]["open_hf11"];
    let state = &expected["states"][0];
    let mut options = PostHFOptions {
        active: Some(ActiveSpace::new(1, 0, 1)),
        ..Default::default()
    };
    options.cas.optimize_orbitals = false;
    let result = compute_mrci(Reference::Uhf(&hf), &options, Some(&ctx.ints), true).unwrap();
    assert_eq!(
        result.determinants.len(),
        expected["n_determinants"].as_u64().unwrap() as usize
    );
    assert!((result.states[0].energy - state["energy"].as_f64().unwrap()).abs() < 1e-9);
    assert!((result.states[0].s_squared - 0.75).abs() < 1e-10);
    assert!((result.davidson_corrections[0] - state["delta_q"].as_f64().unwrap()).abs() < 1e-9);
    let c = &result.mo_coefficients;
    let d = c
        .dot(result.output.one_rdm.as_ref().unwrap())
        .dot(&c.transpose());
    for p in 0..d.rows {
        for q in 0..d.cols {
            assert!(
                (d[(p, q)] - state["one_rdm_ao"][p][q].as_f64().unwrap()).abs() < 1e-7,
                "open density {p},{q}"
            );
        }
    }
    let gamma = rotate_rdm2(&result.two_rdm(0).unwrap(), c);
    for (i, (&a, b)) in gamma
        .iter()
        .zip(state["two_rdm_ao"].as_array().unwrap())
        .enumerate()
    {
        assert!((a - b.as_f64().unwrap()).abs() < 1e-7, "open Gamma[{i}]");
    }
    options.active = Some(ActiveSpace::new(2, 1, 6));
    let full = compute_mrci(Reference::Uhf(&hf), &options, Some(&ctx.ints), false).unwrap();
    let fci = create_solver(MethodId::Fci, &ctx, &options)
        .unwrap()
        .compute()
        .unwrap();
    assert!((full.output.e_total - fci.e_total).abs() < 1e-10);
}
