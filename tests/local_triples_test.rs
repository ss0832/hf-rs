use hf_rs::context::{AuxBasis, PostHFContext};
use hf_rs::options::{LocalOptions, PostHFOptions};
use hf_rs::post_hf::dlpno::{LocalCcsdT0, LocalCcsdT1};
use hf_rs::registry::{ContextMethod, MethodId, create_solver};
use hf_rs::*;
#[test]
fn local_t0_and_iterative_t1_match_independent_determinant_resolvents() {
    let atoms = [
        Atom::new("O", [0.13, -0.21, 0.31]),
        Atom::new("H", [0.23, 1.22, -0.77]),
        Atom::new("H", [-0.08, -1.64, -0.83]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let aux = AuxBasis {
        name: None,
        functions: build_basis(&atoms, &get_builtin_basis("6-31g").unwrap()),
    };
    let hf = run_scf(&atoms, &basis, 150, 1e-14).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, Some(&aux)).unwrap();
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/local_triples_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let options = PostHFOptions {
            max_iter: 300,
            residual_tol: 1e-11,
            energy_tol: 1e-13,
            local: LocalOptions {
                t_cut_pno: case["cutoff"].as_f64().unwrap(),
                t_cut_tno: Some(case["tno_cutoff"].as_f64().unwrap()),
                t_cut_pairs: 0.,
                t_cut_do: 0.,
                t_cut_mkn: 0.,
                full_domains: true,
                weak_pairs: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let t0 = LocalCcsdT0::from_context(&ctx, &options)
            .unwrap()
            .compute()
            .unwrap();
        let t1 = LocalCcsdT1::from_context(&ctx, &options)
            .unwrap()
            .compute()
            .unwrap();
        assert!(
            (t0.triples.energy - case["e_t0"].as_f64().unwrap()).abs() < 1e-10,
            "{} T0: {} vs {}",
            case["name"],
            t0.triples.energy,
            case["e_t0"]
        );
        assert!(
            (t1.triples.energy - case["e_t1"].as_f64().unwrap()).abs() < 1e-10,
            "{} T1: {} vs {}",
            case["name"],
            t1.triples.energy,
            case["e_t1"]
        );
        assert!(t1.triples.residual_max <= options.residual_tol);
        assert!((t0.triples.energy - t1.triples.energy).abs() > 1e-7);
        let mut ranks = t1.tno_ranks.clone();
        ranks.sort_unstable();
        let mut expected = case["tno_ranks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        expected.sort_unstable();
        assert_eq!(ranks, expected);
        if case["cutoff"].as_f64().unwrap() == 0. {
            assert!(
                (t1.triples.energy - data["canonical_ri_triples"].as_f64().unwrap()).abs() < 1e-10
            );
            let mut inherited = options.clone();
            inherited.local.t_cut_tno = None;
            let output = create_solver(MethodId::DlpnoCcsdT1, &ctx, &inherited)
                .unwrap()
                .compute()
                .unwrap();
            assert!((output.e_corr - t1.correlation_energy()).abs() < 1e-10);
        }
    }
    let mut invalid = PostHFOptions::default();
    invalid.local.t_cut_tno = Some(-1.);
    assert!(invalid.validate().is_err());
    invalid.local.t_cut_tno = None;
    invalid.local.iterative_triples = true;
    assert!(LocalCcsdT0::from_context(&ctx, &invalid).is_err());
}
