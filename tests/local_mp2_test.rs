use hf_rs::context::{AuxBasis, PostHFContext};
use hf_rs::options::{LocalOptions, PostHFOptions};
use hf_rs::post_hf::dlpno::LocalMp2;
use hf_rs::registry::{ContextMethod, MethodId, create_solver};
use hf_rs::*;
fn full_options() -> PostHFOptions {
    PostHFOptions {
        max_iter: 300,
        local: LocalOptions {
            t_cut_pno: 0.,
            t_cut_pairs: 0.,
            t_cut_do: 0.,
            t_cut_mkn: 0.,
            full_domains: true,
            weak_pairs: false,
            ..Default::default()
        },
        ..Default::default()
    }
}
#[test]
fn zero_cutoffs_full_domains_match_same_auxiliary_ri_mp2_with_frozen_orbitals() {
    let atoms = [
        Atom::new("O", [0.13, -0.21, 0.31]),
        Atom::new("H", [0.23, 1.22, -0.77]),
        Atom::new("H", [-0.08, -1.64, -0.83]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let aux = AuxBasis {
        name: Some("6-31g test auxiliary".into()),
        functions: build_basis(&atoms, &get_builtin_basis("6-31g").unwrap()),
    };
    let hf = run_scf(&atoms, &basis, 150, 1e-14).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, Some(&aux)).unwrap();
    for (fc, fv) in [
        (0, 0),
        (1, 0),
        (0, 1),
        (hf.n_occ, 0),
        (0, hf.mo_coefficients.cols - hf.n_occ),
    ] {
        let options = PostHFOptions {
            frozen_core: fc,
            frozen_virtual: fv,
            ..full_options()
        };
        let local = LocalMp2::from_context(&ctx, &options)
            .unwrap()
            .compute_checked()
            .unwrap();
        let canonical = create_solver(MethodId::RiMp2, &ctx, &options)
            .unwrap()
            .compute()
            .unwrap();
        assert!(
            (local.correlation_energy() - canonical.e_corr).abs() < 1e-10,
            "frozen({fc},{fv}) local={} canonical={}",
            local.correlation_energy(),
            canonical.e_corr
        );
        let output = create_solver(MethodId::DlpnoMp2, &ctx, &options)
            .unwrap()
            .compute()
            .unwrap();
        assert!((output.e_corr - canonical.e_corr).abs() < 1e-10);
        assert!(local.solution.residual_max <= options.residual_tol);
    }
    let no_aux = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    assert!(LocalMp2::from_context(&no_aux, &full_options()).is_err());
}
#[test]
fn finite_pno_domains_and_weak_pairs_produce_checked_distinct_spaces() {
    let atoms = [
        Atom::new("He", [0., 0., 0.]),
        Atom::new("He", [0.2, 0.3, 5.]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("6-31g").unwrap());
    let aux = AuxBasis {
        name: None,
        functions: basis.clone(),
    };
    let hf = run_scf(&atoms, &basis, 150, 1e-14).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, Some(&aux)).unwrap();
    let options = PostHFOptions {
        local: LocalOptions {
            t_cut_pno: 1e-5,
            t_cut_pairs: 1e-3,
            t_cut_do: 0.1,
            t_cut_mkn: 0.1,
            ..Default::default()
        },
        ..Default::default()
    };
    let result = LocalMp2::from_context(&ctx, &options)
        .unwrap()
        .compute_checked()
        .unwrap();
    assert!(
        result
            .pair_domain_atoms
            .iter()
            .any(|a| a.len() < atoms.len())
    );
    assert!(result.pair_fit_atoms.iter().any(|a| a.len() < atoms.len()));
    assert!(result.n_strong_pairs > 0 && result.n_weak_pairs > 0);
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/local_mp2_reference.json")).unwrap();
    assert!((result.correlation_energy() - oracle["e_corr"].as_f64().unwrap()).abs() < 1e-10);
    assert!((result.solution.e_strong - oracle["e_strong"].as_f64().unwrap()).abs() < 1e-10);
    assert!((result.solution.e_weak - oracle["e_weak"].as_f64().unwrap()).abs() < 1e-10);
    assert!((result.solution.delta_pno - oracle["delta_pno"].as_f64().unwrap()).abs() < 1e-10);
    assert!(result.solution.residual_max <= options.residual_tol);
    // Entire-pair PNO removal remains explicit; no artificial retained mode.
    let empty = PostHFOptions {
        local: LocalOptions {
            t_cut_pno: 1.,
            ..full_options().local
        },
        ..full_options()
    };
    let result = LocalMp2::from_context(&ctx, &empty)
        .unwrap()
        .compute_checked()
        .unwrap();
    assert!(result.pair_ranks.iter().all(|&rank| rank == 0));
    assert!(result.solution.e_strong.abs() < 1e-14);
}

#[test]
fn full_local_spaces_stay_inside_rectangular_spherical_virtual_subspace() {
    let atoms = [
        Atom::new("He", [-0.1, 0.2, 0.3]),
        Atom::new("H", [0.4, -0.3, 1.8]),
    ];
    let map = parse_gaussian94("He 0\nS 1 1\n1.4 1\nD 1 1\n0.6 1\n****\nH 0\nS 1 1\n1.1 1\n****");
    let basis = build_basis(&atoms, &map);
    let hf = scf::run_scf_in_space(
        &atoms,
        &basis,
        1,
        150,
        1e-14,
        ConvAlgo::Cdiis,
        None,
        hf_rs::ao::BasisKind::Spherical,
    )
    .unwrap();
    assert!(hf.mo_coefficients.rows > hf.mo_coefficients.cols);
    let aux = AuxBasis {
        name: None,
        functions: basis.clone(),
    };
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, Some(&aux)).unwrap();
    let options = full_options();
    let local = LocalMp2::from_context(&ctx, &options)
        .unwrap()
        .compute_checked()
        .unwrap();
    let canonical = create_solver(MethodId::RiMp2, &ctx, &options)
        .unwrap()
        .compute()
        .unwrap();
    assert!((local.correlation_energy() - canonical.e_corr).abs() < 1e-10);
    assert!(
        local
            .pair_ranks
            .iter()
            .all(|&r| r == hf.mo_coefficients.cols - hf.n_occ)
    );
}
