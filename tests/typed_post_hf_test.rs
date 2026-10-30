use hf_rs::context::*;
use hf_rs::options::PostHFOptions;
use hf_rs::registry::{ContextMethod, MethodId, create_solver};
use hf_rs::*;

#[test]
fn context_mp2_frozen_core_full_density_matches_pyscf() {
    let atoms = [
        Atom::new("Li", [0.1, -0.2, 0.3]),
        Atom::new("H", [0.4, 0.1, 3.2]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("6-31g").unwrap());
    let hf = run_scf(&atoms, &basis, 150, 1e-14).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/mp2_frozen_reference.json")).unwrap();
    for frozen in [0, 1] {
        let options = PostHFOptions {
            frozen_core: frozen,
            ..Default::default()
        };
        let result = RHFMP2::from_context(&ctx, &options)
            .unwrap()
            .with_rdm(true)
            .compute()
            .unwrap();
        let expected = &oracle["results"][frozen.to_string()];
        assert!((result.correlation_energy - expected["e_corr"].as_f64().unwrap()).abs() < 1e-9);
        let rdm = result.one_rdm_mo.as_ref().unwrap();
        assert!((rdm.trace() - 4.0).abs() < 1e-11);
        let ao = hf
            .mo_coefficients
            .dot(rdm)
            .dot(&hf.mo_coefficients.transpose());
        for i in 0..ao.rows {
            for j in 0..ao.cols {
                assert!(
                    (ao[(i, j)] - expected["one_rdm_ao"][i][j].as_f64().unwrap()).abs() < 1e-8,
                    "frozen={frozen} ({i},{j})"
                );
            }
        }
        let output = create_solver(MethodId::Mp2, &ctx, &options)
            .unwrap()
            .compute()
            .unwrap();
        assert!((output.e_corr - result.correlation_energy).abs() < 1e-12);
        output.validate().unwrap();
        let scaled=create_solver(MethodId::ScsMp2,&ctx,&options).unwrap().compute().unwrap();
        let expected_scs=result.e_same_spin/3.0+1.2*result.e_opposite_spin;
        assert!((scaled.e_corr-expected_scs).abs()<1e-12);
    }
}

#[test]
fn typed_ump2_and_ccsd_use_shared_spin_integrals_and_propagate_nonconvergence() {
    let atoms = [
        Atom::new("H", [0.0, 0.0, -0.7]),
        Atom::new("H", [0.0, 0.0, 0.7]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let hf = run_scf(&atoms, &basis, 100, 1e-14).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, None).unwrap();
    let output = create_solver(MethodId::Ccsd, &ctx, &PostHFOptions::default())
        .unwrap()
        .compute()
        .unwrap();
    let fci = FCI::new(&hf).compute().unwrap();
    assert!((output.e_total - fci.total_energy).abs() < 1e-9);
    let options = PostHFOptions {
        max_iter: 1,
        ..Default::default()
    };
    assert!(
        create_solver(MethodId::Ccsd, &ctx, &options)
            .unwrap()
            .compute()
            .is_err()
    );
    let uhf = run_uhf(&atoms, &basis, 0, 1, false, 100, 1e-14).unwrap();
    let uctx = PostHFContext::from_uhf(&uhf, &atoms, &basis, None).unwrap();
    let uout = create_solver(MethodId::Mp2, &uctx, &PostHFOptions::default())
        .unwrap()
        .compute()
        .unwrap();
    let rout = create_solver(MethodId::Mp2, &ctx, &PostHFOptions::default())
        .unwrap()
        .compute()
        .unwrap();
    assert!((uout.e_total - rout.e_total).abs() < 1e-11);
}
