// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::{
    DeterminantSpace, Matrix,
    df::DfTensor,
    post_hf::mr_dsrg::{
        density_cumulants::{CiEnsembleMember, exact_density_cumulants},
        gno::GnoReference,
        imsrg2::No2bOperator,
        mr_driven::MrDsrgEquations,
        sr_dsrg::LdsrgOptions,
        three_index::factorized_mrpt2,
    },
};
const BUDGET: usize = 256 * 1024 * 1024;

fn integral(b: &DfTensor, p: usize, q: usize, r: usize, s: usize) -> f64 {
    let direct = if p % 2 == r % 2 && q % 2 == s % 2 {
        b.eri(p / 2, r / 2, q / 2, s / 2)
    } else {
        0.
    };
    let exchange = if p % 2 == s % 2 && q % 2 == r % 2 {
        b.eri(p / 2, s / 2, q / 2, r / 2)
    } else {
        0.
    };
    direct - exchange
}

#[test]
fn streamed_sectors_match_every_dense_gno_energy_and_effective_operator_coefficient() {
    let mut count = 0;
    let mut maximum = 0f64;
    for (nc, na, nv, alpha, beta) in [
        (1, 2, 1, 1, 1),
        (1, 3, 1, 2, 1),
        (0, 3, 1, 2, 1),
        (1, 3, 0, 2, 1),
        (0, 2, 0, 1, 1),
    ] {
        let n = nc + na + nv;
        let ns = 2 * n;
        let space = DeterminantSpace::new(na, alpha, beta);
        let mut ci: Vec<_> = (0..space.n_determinants())
            .map(|k| ((k + 2) as f64 * 1.134).sin())
            .collect();
        let norm = ci.iter().map(|v| v * v).sum::<f64>().sqrt();
        ci.iter_mut().for_each(|v| *v /= norm);
        let density = exact_density_cumulants(
            &[CiEnsembleMember {
                space: &space,
                coefficients: &ci,
                weight: 1.,
            }],
            BUDGET,
        )
        .unwrap();
        let core: Vec<_> = (0..2 * nc).collect();
        let active: Vec<_> = (2 * nc..2 * (nc + na)).collect();
        let reference = GnoReference::new(&density, ns, &core, &active).unwrap();
        let factors = DfTensor {
            n_ao: n,
            n_aux_original: 3,
            b: Matrix::from_vec(
                3,
                n * (n + 1) / 2,
                (0..3 * n * (n + 1) / 2)
                    .map(|k| 0.13 * ((k + 1) as f64 * 0.73).cos())
                    .collect(),
            ),
        };
        let mut h = No2bOperator {
            zero: -2.3,
            one: Matrix::zeros(ns, ns),
            two: vec![0.; ns.pow(4)],
        };
        let classes = |p| {
            if p < 2 * nc {
                0
            } else if p < 2 * (nc + na) {
                1
            } else {
                2
            }
        };
        for p in 0..ns {
            h.one[(p, p)] = -1.3 + 0.21 * (p / 2) as f64;
            for q in 0..ns {
                if classes(p) != classes(q) && p % 2 == q % 2 {
                    h.one[(p, q)] = 0.03 * ((p + q + 1) as f64).sin();
                }
                for r in 0..ns {
                    for s in 0..ns {
                        h.two[((p * ns + q) * ns + r) * ns + s] = integral(&factors, p, q, r, s);
                    }
                }
            }
        }
        let equations = MrDsrgEquations::new(&reference, BUDGET).unwrap();
        let base = equations.active_effective_hamiltonian(&h).unwrap();
        for flow in [0., 0.05, 0.5, 2.] {
            let actual = factorized_mrpt2(&factors, &h.one, &reference, flow, BUDGET).unwrap();
            let expected = equations
                .second_order_energy(&h, flow, LdsrgOptions::default())
                .unwrap();
            let mut effective = equations
                .perturbative_active_hamiltonian(&h, flow, 2, LdsrgOptions::default())
                .unwrap();
            effective.zero -= base.zero;
            for (x, y) in effective.one.data.iter_mut().zip(&base.one.data) {
                *x -= y;
            }
            for (x, y) in effective.two.iter_mut().zip(&base.two) {
                *x -= y;
            }
            assert!(
                (actual.energy - expected).abs() < 1e-11,
                "{nc}/{na}/{nv} s={flow} energy {} vs {expected}",
                actual.energy
            );
            for (i, (x, y)) in std::iter::once(&actual.active_operator.zero)
                .chain(&actual.active_operator.one.data)
                .chain(&actual.active_operator.two)
                .zip(
                    std::iter::once(&effective.zero)
                        .chain(&effective.one.data)
                        .chain(&effective.two),
                )
                .enumerate()
            {
                let error = (x - y).abs();
                maximum = maximum.max(error);
                assert!(
                    error < 1e-11,
                    "{nc}/{na}/{nv} s={flow} coefficient {i}: {x:e} vs {y:e} ({error:e})"
                );
                count += 1;
            }
        }
    }
    println!("Factorized sectors: {count} effective coefficients, maximum error {maximum:e}");
}

#[test]
fn large_external_space_streams_under_eight_mib_without_global_gno_tensors() {
    let n = 80;
    let ns = 2 * n;
    let space = DeterminantSpace::new(2, 1, 1);
    let ci = [0.8, 0.3, 0.3, 0.4242640687119285];
    let norm = ci.iter().map(|v| v * v).sum::<f64>().sqrt();
    let ci: Vec<_> = ci.iter().map(|v| v / norm).collect();
    let density = exact_density_cumulants(
        &[CiEnsembleMember {
            space: &space,
            coefficients: &ci,
            weight: 1.,
        }],
        BUDGET,
    )
    .unwrap();
    let reference = GnoReference::new(&density, ns, &[0, 1], &[2, 3, 4, 5]).unwrap();
    let factors = DfTensor {
        n_ao: n,
        n_aux_original: 2,
        b: Matrix::from_vec(
            2,
            n * (n + 1) / 2,
            (0..n * (n + 1))
                .map(|k| 0.013 * ((k + 1) as f64 * 0.73).cos())
                .collect(),
        ),
    };
    let mut f = Matrix::zeros(ns, ns);
    for p in 0..ns {
        f[(p, p)] = -1.3 + 0.021 * (p / 2) as f64;
    }
    let result = factorized_mrpt2(&factors, &f, &reference, 0.5, 8 * 1024 * 1024).unwrap();
    assert!(result.energy.is_finite() && result.energy < 0.);
    assert!(result.peak_sector_terms < 256);
    assert_eq!(result.active_operator.two.len(), 4usize.pow(4));
    assert!(
        factorized_mrpt2(&factors, &f, &reference, 0.5, 1024)
            .err()
            .unwrap()
            .to_string()
            .contains("memory budget")
    );
    println!(
        "160 spin orbitals, {} sectors, {} peak terms, E2={:e}",
        result.sectors, result.peak_sector_terms, result.energy
    );
}

#[test]
fn large_single_determinant_limit_matches_independent_spatial_regularized_mp2_sum() {
    let n = 80;
    let ns = 2 * n;
    let occupied = 2;
    let space = DeterminantSpace::new(0, 0, 0);
    let density = exact_density_cumulants(
        &[CiEnsembleMember {
            space: &space,
            coefficients: &[1.],
            weight: 1.,
        }],
        BUDGET,
    )
    .unwrap();
    let core: Vec<_> = (0..2 * occupied).collect();
    let reference = GnoReference::new(&density, ns, &core, &[]).unwrap();
    let factors = DfTensor {
        n_ao: n,
        n_aux_original: 2,
        b: Matrix::from_vec(
            2,
            n * (n + 1) / 2,
            (0..n * (n + 1))
                .map(|k| 0.013 * ((k + 1) as f64 * 0.73).cos())
                .collect(),
        ),
    };
    let mut f = Matrix::zeros(ns, ns);
    for p in 0..ns {
        f[(p, p)] = -1.3 + 0.021 * (p / 2) as f64;
    }
    let flow = 0.5;
    let actual = factorized_mrpt2(&factors, &f, &reference, flow, 8 * 1024 * 1024).unwrap();
    let mut expected = 0.;
    for i in 0..occupied {
        for j in 0..occupied {
            for a in occupied..n {
                for b in occupied..n {
                    let gap = f[(2 * i, 2 * i)] + f[(2 * j, 2 * j)]
                        - f[(2 * a, 2 * a)]
                        - f[(2 * b, 2 * b)];
                    let v = factors.eri(i, a, j, b);
                    expected += v
                        * (2. * v - factors.eri(i, b, j, a))
                        * (-(-2. * flow * gap * gap).exp_m1())
                        / gap;
                }
            }
        }
    }
    assert!((actual.energy - expected).abs() < 1e-12);
    println!(
        "160-spin MP2 limit: error={:e}, {} peak terms",
        (actual.energy - expected).abs(),
        actual.peak_sector_terms
    );
}
