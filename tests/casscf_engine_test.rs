use hf_rs::context::{PostHFContext, Reference};
use hf_rs::mo_integrals::{transform_1e_ao_to_mo, transform_eri_ao_to_mo};
use hf_rs::options::{ActiveSpace, PostHFOptions};
use hf_rs::post_hf::casscf_engine::{generalized_fock_element, orbital_curvature, solve_cas};
use hf_rs::*;

fn lih() -> (Vec<Atom>, Vec<ContractedGaussian>, HFResult) {
    let atoms = vec![
        Atom::new("Li", [0.1, -0.2, 0.3]),
        Atom::new("H", [0.4, 0.1, 3.2]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_scf(&atoms, &basis, 150, 1e-14).unwrap();
    assert!(hf.converged);
    (atoms, basis, hf)
}
#[test]
fn all_orbital_gradients_and_diagonal_curvatures_match_frozen_ci_energy_differences() {
    let (_, _, hf) = lih();
    let options = PostHFOptions {
        active: Some(ActiveSpace::new(1, 1, 2)),
        ..Default::default()
    };
    let cas = solve_cas(Reference::Rhf(&hf), &options, None, false).unwrap();
    let c = &cas.mo_coefficients;
    let n = c.cols;
    let nc = 1;
    let na = 2;
    let d = &cas.state_one_rdms[0];
    let gamma = &cas.state_two_rdms[0];
    let h = transform_1e_ao_to_mo(c, &hf.core_hamiltonian);
    let g = transform_eri_ao_to_mo(c, &hf.eri);
    // Construct a total 2-RDM explicitly here; the production gradient does not.
    let core = |p, q| if p == q && p < nc { 2.0 } else { 0.0 };
    let active = |p, q| {
        if p >= nc && p < nc + na && q >= nc && q < nc + na {
            d[(p - nc, q - nc)]
        } else {
            0.0
        }
    };
    let total_gamma = |p, q, r, s| {
        let ga = if [p, q, r, s].iter().all(|&i| i >= nc && i < nc + na) {
            gamma[(((p - nc) * na + q - nc) * na + r - nc) * na + s - nc]
        } else {
            0.0
        };
        ga + core(p, q) * core(r, s) - 0.5 * core(p, s) * core(r, q)
            + core(p, q) * active(r, s)
            + active(p, q) * core(r, s)
            - 0.5 * (core(p, s) * active(r, q) + active(p, s) * core(r, q))
    };
    for p in 0..n {
        for q in 0..p {
            let energy = |theta: f64| {
                let mut k = Matrix::zeros(n, n);
                k[(p, q)] = theta;
                k[(q, p)] = -theta;
                let rotated = c.dot(&k.exp());
                let hh = transform_1e_ao_to_mo(&rotated, &hf.core_hamiltonian);
                let gg = transform_eri_ao_to_mo(&rotated, &hf.eri);
                let mut e = 0.0;
                for a in 0..n {
                    for b in 0..n {
                        e += hh[(a, b)] * (core(a, b) + active(a, b));
                        for r in 0..n {
                            for s in 0..n {
                                e += 0.5 * gg.get(a, b, r, s) * total_gamma(a, b, r, s);
                            }
                        }
                    }
                }
                e
            };
            let f = |p, q| {
                generalized_fock_element(
                    p,
                    q,
                    nc,
                    d,
                    gamma,
                    |a, b| h[(a, b)],
                    |a, b, r, s| g.get(a, b, r, s),
                )
            };
            let step = 0.001;
            let em2 = energy(-2.0 * step);
            let em = energy(-step);
            let ep = energy(step);
            let ep2 = energy(2.0 * step);
            let first = (em2 - 8.0 * em + 8.0 * ep - ep2) / (12.0 * step);
            let second =
                (-ep2 + 16.0 * ep - 30.0 * energy(0.0) + 16.0 * em - em2) / (12.0 * step * step);
            assert!(
                (2.0 * (f(p, q) - f(q, p)) - first).abs() < 1e-8,
                "gradient {p},{q}: {} vs {first}",
                2.0 * (f(p, q) - f(q, p))
            );
            let curvature =
                orbital_curvature(p, q, nc, d, gamma, &h, |a, b, r, s| g.get(a, b, r, s));
            assert!(
                (curvature - second).abs() < 2e-7,
                "curvature {p},{q}: {curvature} vs {second}"
            );
        }
    }
}
#[test]
fn casscf_lih_is_variational_converged_and_orthonormal() {
    let (atoms, basis, hf) = lih();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let mut options = PostHFOptions {
        active: Some(ActiveSpace::new(1, 1, 2)),
        ..Default::default()
    };
    options.cas.max_orbital_iterations = 150;
    options.cas.orbital_gradient_tol = 1e-9;
    let initial = solve_cas(Reference::Rhf(&hf), &options, Some(&ctx.ints), false).unwrap();
    let result = solve_cas(Reference::Rhf(&hf), &options, Some(&ctx.ints), true).unwrap();
    eprintln!(
        "LiH CASSCF E={} iterations={} maxg={}",
        result.output.e_total, result.iterations, result.orbital_gradient_max
    );
    assert!(result.output.e_total <= initial.output.e_total + 1e-12);
    assert!(result.orbital_gradient_max < 1e-7);
    assert!(result.states[0].s_squared.abs() < 1e-10);
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/casscf_reference.json")).unwrap();
    let expected = &oracle["results"]["lih_cas22"];
    assert!((result.output.e_total - expected["energy"].as_f64().unwrap()).abs() < 1e-9);
    let ao = result
        .mo_coefficients
        .dot(result.output.one_rdm.as_ref().unwrap())
        .dot(&result.mo_coefficients.transpose());
    for p in 0..ao.rows {
        for q in 0..ao.cols {
            assert!(
                (ao[(p, q)] - expected["one_rdm_ao"][p][q].as_f64().unwrap()).abs() < 1e-7,
                "AO density ({p},{q}): {} vs {}",
                ao[(p, q)],
                expected["one_rdm_ao"][p][q]
            );
        }
    }
    let metric = result
        .mo_coefficients
        .transpose()
        .dot(&hf.overlap_matrix)
        .dot(&result.mo_coefficients);
    for p in 0..metric.rows {
        for q in 0..metric.cols {
            assert!((metric[(p, q)] - f64::from(p == q)).abs() < 1e-11);
        }
    }
}

#[test]
fn full_active_and_closed_shell_limits_and_selected_root_semantics_match_pyscf() {
    let (atoms, basis, hf) = lih();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/casscf_reference.json")).unwrap();
    for (name, active) in [
        ("lih_full", ActiveSpace::new(2, 2, 6)),
        ("lih_closed", ActiveSpace::new(1, 1, 1)),
        ("lih_closed", ActiveSpace::new(0, 0, 0)),
    ] {
        let options = PostHFOptions {
            active: Some(active),
            ..Default::default()
        };
        let result = solve_cas(Reference::Rhf(&hf), &options, Some(&ctx.ints), true).unwrap();
        assert!(
            (result.output.e_total - oracle["results"][name]["energy"].as_f64().unwrap()).abs()
                < 1e-10,
            "{name}"
        );
    }
    let mut options = PostHFOptions {
        active: Some(ActiveSpace::new(1, 1, 2)),
        n_roots: 2,
        root: 1,
        ..Default::default()
    };
    options.cas.state_weights = vec![0.4, 0.6];
    options.cas.max_orbital_iterations = 150;
    options.cas.orbital_gradient_tol = 1e-8;
    let result = solve_cas(Reference::Rhf(&hf), &options, Some(&ctx.ints), true).unwrap();
    let expected = &oracle["results"]["lih_sa22"];
    eprintln!(
        "SA LiH E={:?} avg={} maxg={} iter={}",
        result.states.iter().map(|s| s.energy).collect::<Vec<_>>(),
        result.output.components["E_state_average"],
        result.orbital_gradient_max,
        result.iterations
    );
    assert!(
        (result.output.components["E_state_average"]
            - expected["energy_average"].as_f64().unwrap())
        .abs()
            < 1e-9
    );
    for (i, state) in result.states.iter().enumerate() {
        assert!((state.energy - expected["energies"][i].as_f64().unwrap()).abs() < 1e-9);
    }
    assert_eq!(result.output.root, 1);
    assert_eq!(result.output.e_total, result.states[1].energy);
    assert!((result.output.e_corr - result.output.e_total + hf.total_energy).abs() < 1e-12);
    let registry =
        hf_rs::registry::create_solver(hf_rs::registry::MethodId::Casscf, &ctx, &options)
            .unwrap()
            .compute()
            .unwrap();
    assert_eq!(registry.root, 1);
    assert!((registry.e_total - result.output.e_total).abs() < 1e-12);
}

#[test]
fn open_shell_full_space_explicit_selection_and_invalid_spin_are_checked() {
    let atoms = vec![
        Atom::new("He", [-0.1, 0.2, 0.0]),
        Atom::new("H", [0.3, -0.2, 1.7]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_uhf(&atoms, &basis, 0, 2, false, 150, 1e-14).unwrap();
    let ctx = PostHFContext::from_uhf(&hf, &atoms, &basis, None).unwrap();
    let mut active = ActiveSpace::new(2, 1, 2);
    active.orbitals = Some(vec![1, 0]);
    let mut options = PostHFOptions {
        active: Some(active),
        ..Default::default()
    };
    let optimized = solve_cas(Reference::Uhf(&hf), &options, Some(&ctx.ints), true).unwrap();
    options.active.as_mut().unwrap().orbitals = None;
    let canonical = solve_cas(Reference::Uhf(&hf), &options, Some(&ctx.ints), false).unwrap();
    assert!((optimized.output.e_total - canonical.output.e_total).abs() < 1e-11);
    assert!((optimized.states[0].s_squared - 0.75).abs() < 1e-11);
    let ao1 = optimized
        .mo_coefficients
        .dot(optimized.output.one_rdm.as_ref().unwrap())
        .dot(&optimized.mo_coefficients.transpose());
    let ao2 = canonical
        .mo_coefficients
        .dot(canonical.output.one_rdm.as_ref().unwrap())
        .dot(&canonical.mo_coefficients.transpose());
    for (a, b) in ao1.data.iter().zip(&ao2.data) {
        assert!((a - b).abs() < 1e-10);
    }
    options.cas.target_s_squared = Some(0.0);
    assert!(solve_cas(Reference::Uhf(&hf), &options, Some(&ctx.ints), true).is_err());
    let mut options = PostHFOptions::default();
    assert!(solve_cas(Reference::Uhf(&hf), &options, Some(&ctx.ints), true).is_err());
    options.active = Some(ActiveSpace::new(2, 1, 2));
    options.n_roots = 3;
    assert!(solve_cas(Reference::Uhf(&hf), &options, Some(&ctx.ints), false).is_err());
}
