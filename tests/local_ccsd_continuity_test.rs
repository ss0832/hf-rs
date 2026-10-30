// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::context::{AuxBasis, PostHFContext};
use hf_rs::options::{LocalOptions, PostHFOptions};
use hf_rs::post_hf::dlpno::{
    LocalCcsd, LocalCcsdT0, LocalCcsdT1, LocalSpaceAnchor, LocalTriplesAnchor,
};
use hf_rs::registry::ContextMethod;
use hf_rs::*;

fn options(cutoff: f64) -> PostHFOptions {
    PostHFOptions {
        max_iter: 300,
        energy_tol: 1e-13,
        residual_tol: 1e-11,
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
fn atoms(case: &serde_json::Value) -> Vec<Atom> {
    ["O", "H", "H"]
        .iter()
        .enumerate()
        .map(|(i, symbol)| {
            Atom::new(
                symbol,
                std::array::from_fn(|a| case["positions"][i][a].as_f64().unwrap()),
            )
        })
        .collect()
}

#[test]
fn transported_ccsd_matches_all_ao_amplitudes_at_every_coordinate_and_rigid_motion() {
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/local_ccsd_continuity_reference.json")).unwrap();
    let snapshots = reference["snapshots"].as_array().unwrap();
    let original = atoms(&snapshots[0]);
    let definitions = get_builtin_basis("sto-3g").unwrap();
    let auxdefs = get_builtin_basis("6-31g").unwrap();
    let basis = build_basis(&original, &definitions);
    let aux = AuxBasis {
        name: None,
        functions: build_basis(&original, &auxdefs),
    };
    let hf = run_scf(&original, &basis, 150, 1e-14).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &original, &basis, Some(&aux)).unwrap();
    let opts = options(reference["cutoff"].as_f64().unwrap());
    let anchor = LocalSpaceAnchor::from_context(&ctx, &opts).unwrap();
    let ranks = anchor.pair_ranks();
    assert_eq!(
        ranks,
        reference["pair_ranks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>()
    );
    let base = anchor.compute_ccsd(&ctx).unwrap();
    let adaptive = LocalCcsd::from_context(&ctx, &opts)
        .unwrap()
        .compute_checked()
        .unwrap();
    assert!((base.correlation_energy() - adaptive.correlation_energy()).abs() < 1e-11);
    let full_options = options(0.);
    let full_anchor = LocalSpaceAnchor::from_context(&ctx, &full_options).unwrap();
    let full = full_anchor.compute_ccsd(&ctx).unwrap();
    let full_adaptive = LocalCcsd::from_context(&ctx, &full_options)
        .unwrap()
        .compute_checked()
        .unwrap();
    assert!((full.correlation_energy() - full_adaptive.correlation_energy()).abs() < 1e-10);
    let limited = PostHFOptions {
        max_iter: 1,
        ..opts.clone()
    };
    assert!(
        LocalSpaceAnchor::from_context(&ctx, &limited)
            .unwrap()
            .compute_ccsd(&ctx)
            .is_err()
    );
    let mut changed_basis = basis.clone();
    changed_basis[0].primitives[0].exponent *= 1.01;
    let changed = PostHFContext::from_rhf(&hf, &original, &changed_basis, Some(&aux)).unwrap();
    assert!(anchor.compute_ccsd(&changed).is_err());

    let mut components = 0;
    let mut max_energy = 0f64;
    let mut max_amplitude = 0f64;
    for (sample, case) in snapshots.iter().enumerate() {
        let moved = atoms(case);
        let basis = build_basis(&moved, &definitions);
        let aux = AuxBasis {
            name: None,
            functions: build_basis(&moved, &auxdefs),
        };
        let hf = run_scf(&moved, &basis, 150, 1e-14).unwrap();
        let ctx = PostHFContext::from_rhf(&hf, &moved, &basis, Some(&aux)).unwrap();
        let result = anchor.compute_ccsd(&ctx).unwrap();
        assert_eq!(result.pair_ranks, ranks);
        assert_eq!(result.pair_domain_atoms, base.pair_domain_atoms);
        let error = (result.e_ccsd - case["e_ccsd"].as_f64().unwrap()).abs();
        max_energy = max_energy.max(error);
        assert!(error < 1e-9, "snapshot {sample}: CC energy error {error:e}");
        assert!((result.delta_pno - case["delta_pno"].as_f64().unwrap()).abs() < 1e-10);
        let (no, nv, nao) = (result.occupied.cols, result.virtuals.cols, basis.len());
        for mu in 0..nao {
            for nu in 0..nao {
                let mut t1 = 0.;
                for i in 0..no {
                    for a in 0..nv {
                        t1 += result.occupied[(mu, i)]
                            * result.t1[i * nv + a]
                            * result.virtuals[(nu, a)];
                    }
                }
                let error = (t1 - case["t1_ao"][mu * nao + nu].as_f64().unwrap()).abs();
                max_amplitude = max_amplitude.max(error);
                assert!(error < 1e-8, "snapshot {sample}, T1 {mu},{nu}: {error:e}");
                components += 1;
                for rho in 0..nao {
                    for sigma in 0..nao {
                        let mut t2 = 0.;
                        for i in 0..no {
                            for j in 0..no {
                                for a in 0..nv {
                                    for b in 0..nv {
                                        t2 += result.occupied[(mu, i)]
                                            * result.occupied[(nu, j)]
                                            * result.virtuals[(rho, a)]
                                            * result.virtuals[(sigma, b)]
                                            * result.t2[((i * no + j) * nv + a) * nv + b];
                                    }
                                }
                            }
                        }
                        let index = ((mu * nao + nu) * nao + rho) * nao + sigma;
                        let error = (t2 - case["t2_ao"][index].as_f64().unwrap()).abs();
                        max_amplitude = max_amplitude.max(error);
                        assert!(
                            error < 1e-8,
                            "snapshot {sample}, T2 {mu},{nu},{rho},{sigma}: {error:e}"
                        );
                        components += 1;
                    }
                }
            }
        }
    }
    assert_eq!(components, 49_000);
    println!(
        "Fixed-space CCSD: {} snapshots, all {components} AO amplitudes; max energy {max_energy:e}, max amplitude {max_amplitude:e}",
        snapshots.len()
    );
}

#[test]
fn transported_tno_t0_t1_match_independent_determinant_resolvents_at_every_coordinate() {
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("data/local_triples_continuity_reference.json")).unwrap();
    let snapshots = reference["snapshots"].as_array().unwrap();
    let original = atoms(&snapshots[0]);
    let definitions = get_builtin_basis("sto-3g").unwrap();
    let auxdefs = get_builtin_basis("6-31g").unwrap();
    let basis = build_basis(&original, &definitions);
    let aux = AuxBasis {
        name: None,
        functions: build_basis(&original, &auxdefs),
    };
    let hf = run_scf(&original, &basis, 150, 1e-14).unwrap();
    let ctx = PostHFContext::from_rhf(&hf, &original, &basis, Some(&aux)).unwrap();
    let mut opts = options(reference["cutoff"].as_f64().unwrap());
    opts.local.t_cut_tno = Some(reference["tno_cutoff"].as_f64().unwrap());
    let anchor = LocalTriplesAnchor::from_context(&ctx, &opts).unwrap();
    let ranks: Vec<_> = reference["tno_ranks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert!(ranks.iter().any(|&r| r < 2));
    assert!(ranks.contains(&2));
    let mut maximum = 0f64;
    for (sample, case) in snapshots.iter().enumerate() {
        let moved = atoms(case);
        let basis = build_basis(&moved, &definitions);
        let aux = AuxBasis {
            name: None,
            functions: build_basis(&moved, &auxdefs),
        };
        let hf = run_scf(&moved, &basis, 150, 1e-14).unwrap();
        let ctx = PostHFContext::from_rhf(&hf, &moved, &basis, Some(&aux)).unwrap();
        for (key, result) in [
            ("e_t0", anchor.compute_t0(&ctx).unwrap()),
            ("e_t1", anchor.compute_t1(&ctx).unwrap()),
        ] {
            assert_eq!(result.tno_ranks, ranks);
            let error = (result.triples.energy - case[key].as_f64().unwrap()).abs();
            maximum = maximum.max(error);
            assert!(error < 1e-9, "snapshot {sample}, {key}: {error:e}");
            assert!(result.triples.residual_max < 1e-10);
            assert!((result.ccsd.e_ccsd - case["e_ccsd"].as_f64().unwrap()).abs() < 1e-9);
            assert!((result.ccsd.delta_pno - case["delta_pno"].as_f64().unwrap()).abs() < 1e-10);
            if sample == 0 {
                let adaptive = if key == "e_t0" {
                    LocalCcsdT0::from_context(&ctx, &opts)
                        .unwrap()
                        .compute()
                        .unwrap()
                } else {
                    LocalCcsdT1::from_context(&ctx, &opts)
                        .unwrap()
                        .compute()
                        .unwrap()
                };
                assert!((adaptive.triples.energy - result.triples.energy).abs() < 1e-10);
            }
        }
    }
    println!(
        "Fixed-space (T0)/(T1): all {} snapshots, maximum energy error {maximum:e}",
        snapshots.len()
    );
}
