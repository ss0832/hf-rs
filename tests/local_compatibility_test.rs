use hf_rs::context::{AuxBasis, PostHFContext};
use hf_rs::options::{LocalOptions, PostHFOptions};
use hf_rs::post_hf::dlpno::*;
use hf_rs::registry::{ContextMethod, MethodId, create_solver};
use hf_rs::*;
#[test]
fn compatibility_methods_require_real_systems_and_delegate_to_same_auxiliary_theory() {
    let atoms = [
        Atom::new("H", [0., 0., -0.7]),
        Atom::new("H", [0., 0., 0.7]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_scf(&atoms, &basis, 100, 1e-14).unwrap();
    let aux = AuxBasis::from_builtin(&atoms, "def2-svp-ri").unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, Some(&aux)).unwrap();
    let options = DLPNOMP2Options {
        t_pno: 0.,
        t_cut_pairs: 0.,
        ..Default::default()
    };
    assert!(DLPNOMP2::new(&hf).compute().is_err());
    assert!(DLPNOMCCSD::new(&hf).compute().is_err());
    assert!(DLPNOMCCSDT::new(&hf).compute().is_err());
    let typed = PostHFOptions {
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
    };
    let mp = DLPNOMP2::new(&hf)
        .with_system(&atoms, &basis, &aux)
        .with_options(options.clone())
        .compute()
        .unwrap();
    let canonical = create_solver(MethodId::RiMp2, &ctx, &typed)
        .unwrap()
        .compute()
        .unwrap();
    assert!((mp.correlation_energy - canonical.e_corr).abs() < 1e-12);
    assert!(
        (mp.correlation_energy - mp.strong_pair_energy - mp.weak_pair_energy - mp.delta_pno_energy)
            .abs()
            < 1e-14
    );
    let cc = DLPNOMCCSD::new(&hf)
        .with_context(&ctx)
        .with_options(options.clone())
        .compute()
        .unwrap();
    let reference = LocalCcsd::from_context(&ctx, &typed)
        .unwrap()
        .compute()
        .unwrap();
    assert!((cc.correlation_energy - reference.correlation_energy()).abs() < 1e-12);
    assert!((cc.dlpno_mp2_correlation - canonical.e_corr).abs() < 1e-12);
    let triples = DLPNOMCCSDT::new(&hf)
        .with_context(&ctx)
        .with_options(options.clone())
        .compute()
        .unwrap();
    assert!((triples.correlation_energy() - reference.correlation_energy()).abs() < 1e-12);
    assert_eq!(triples.triples_correlation_energy, 0.);
    let unsupported = DLPNOMP2Options {
        smoothing: true,
        ..options
    };
    assert!(
        DLPNOMP2::new(&hf)
            .with_context(&ctx)
            .with_options(unsupported)
            .compute()
            .is_err()
    );
    let unsupported = DLPNOMP2Options {
        t_cut_weak: 1e-8,
        ..Default::default()
    };
    assert!(
        DLPNOMP2::new(&hf)
            .with_context(&ctx)
            .with_options(unsupported)
            .compute()
            .is_err()
    );
}
