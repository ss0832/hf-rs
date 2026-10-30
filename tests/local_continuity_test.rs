use hf_rs::context::{AuxBasis, PostHFContext};
use hf_rs::options::{LocalOptions, PostHFOptions};
use hf_rs::post_hf::dlpno::{LocalMp2, LocalSpaceAnchor};
use hf_rs::registry::ContextMethod;
use hf_rs::*;
fn options(cutoff: f64) -> PostHFOptions {
    PostHFOptions {
        residual_tol: 1e-12,
        max_iter: 300,
        local: LocalOptions {
            t_cut_pno: cutoff,
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
fn transported_spaces_match_independent_energies_for_every_nuclear_coordinate() {
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/local_continuity_reference.json")).unwrap();
    let snapshots = reference["snapshots"].as_array().unwrap();
    let atoms_at = |case: &serde_json::Value| {
        ["O", "H", "H"]
            .iter()
            .enumerate()
            .map(|(i, symbol)| {
                Atom::new(
                    symbol,
                    std::array::from_fn(|a| case["positions"][i][a].as_f64().unwrap()),
                )
            })
            .collect::<Vec<_>>()
    };
    let atoms = atoms_at(&snapshots[0]);
    let definitions = get_builtin_basis("sto-3g").unwrap();
    let basis = build_basis(&atoms, &definitions);
    let aux = AuxBasis {
        name: None,
        functions: build_basis(&atoms, &get_builtin_basis("6-31g").unwrap()),
    };
    let hf = run_scf(&atoms, &basis, 150, 1e-14).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &atoms, &basis, Some(&aux)).unwrap();
    let options = options(reference["cutoff"].as_f64().unwrap());
    let anchor = LocalSpaceAnchor::from_context(&ctx, &options).unwrap();
    let ranks = anchor.pair_ranks();
    let base = anchor.compute_mp2(&ctx).unwrap();
    let adaptive = LocalMp2::from_context(&ctx, &options)
        .unwrap()
        .compute_checked()
        .unwrap();
    assert!((base.correlation_energy() - adaptive.correlation_energy()).abs() < 1e-11);
    for case in snapshots {
        let moved = atoms_at(case);
        let basis = build_basis(&moved, &definitions);
        let aux = AuxBasis {
            name: None,
            functions: build_basis(&moved, &get_builtin_basis("6-31g").unwrap()),
        };
        let hf = run_scf(&moved, &basis, 150, 1e-14).unwrap();
        let ctx = PostHFContext::from_rhf(&hf, &moved, &basis, Some(&aux)).unwrap();
        let actual = anchor.compute_mp2(&ctx).unwrap();
        assert_eq!(actual.pair_ranks, ranks);
        assert!(
            (actual.correlation_energy() - case["e_corr"].as_f64().unwrap()).abs() < 1e-10,
            "coordinate {} displacement {}: {} vs {}",
            case["coordinate"],
            case["displacement"],
            actual.correlation_energy(),
            case["e_corr"]
        );
    }
}
