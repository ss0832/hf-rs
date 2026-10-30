use hf_rs::context::{AuxBasis, PostHFContext};
use hf_rs::options::{LocalOptions, PostHFOptions};
use hf_rs::post_hf::dlpno::LocalCcsd;
use hf_rs::registry::{ContextMethod, MethodId, create_solver};
use hf_rs::*;
#[test]
fn localized_full_space_ri_ccsd_matches_independent_pyscf_energy_and_all_ao_amplitudes() {
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
        serde_json::from_str(include_str!("data/local_ccsd_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let options = PostHFOptions {
            max_iter: 300,
            residual_tol: 1e-11,
            energy_tol: 1e-13,
            local: LocalOptions {
                t_cut_pno: case["cutoff"].as_f64().unwrap(),
                t_cut_pairs: 0.,
                t_cut_do: 0.,
                t_cut_mkn: 0.,
                full_domains: true,
                weak_pairs: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = LocalCcsd::from_context(&ctx, &options)
            .unwrap()
            .compute_checked()
            .unwrap();
        assert!(
            (result.e_ccsd - case["e_ccsd"].as_f64().unwrap()).abs() < 1e-9,
            "{} Rust={}, reference={}",
            case["name"],
            result.e_ccsd,
            case["e_ccsd"]
        );
        assert!((result.delta_pno - case["delta_pno"].as_f64().unwrap()).abs() < 1e-10);
        let no = result.occupied.cols;
        let nv = result.virtuals.cols;
        let nao = basis.len();
        for mu in 0..nao {
            for nu in 0..nao {
                let t1 = (0..no)
                    .flat_map(|i| (0..nv).map(move |a| (i, a)))
                    .map(|(i, a)| {
                        result.occupied[(mu, i)] * result.t1[i * nv + a] * result.virtuals[(nu, a)]
                    })
                    .sum::<f64>();
                assert!(
                    (t1 - case["t1_ao"][mu][nu].as_f64().unwrap()).abs() < 1e-8,
                    "T1 {mu},{nu}"
                );
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
                        assert!(
                            (t2 - case["t2_ao"][mu][nu][rho][sigma].as_f64().unwrap()).abs() < 1e-8,
                            "T2 {mu},{nu},{rho},{sigma}: {t2} vs {}",
                            case["t2_ao"][mu][nu][rho][sigma]
                        );
                    }
                }
            }
        }
        let output = create_solver(MethodId::DlpnoCcsd, &ctx, &options)
            .unwrap()
            .compute()
            .unwrap();
        assert!((output.e_corr - result.correlation_energy()).abs() < 1e-11);
    }
}
