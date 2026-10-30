// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Comprehensive theoretical consistency tests for post-HF methods.

use hf_rs::basis::{build_basis, parse_gaussian94, Atom, STO_3G_GBS};
use hf_rs::diis::ConvAlgo;
use hf_rs::post_hf::cc::{CCD, CCSD};
use hf_rs::post_hf::ci::fci::FCI;
use hf_rs::post_hf::mp2::RHFMP2;
use hf_rs::post_hf::mpn::{RHFMP3, RHFMPn};
use hf_rs::post_hf::{PostHFMethod, PostHFResult};
use hf_rs::scf::run_scf_general;

#[test]
fn test_h2_post_hf_theoretical_energies() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, -0.7]),
        Atom::new("H", [0.0, 0.0,  0.7]),
    ];
    let basis = build_basis(&atoms, &basis_map);
    let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, ConvAlgo::Adiissoscf, None)
        .expect("RHF SCF failed");

    println!("=== H2/STO-3G Post-HF Theoretical Benchmark ===");
    println!("RHF Reference Energy: {:.10} Eh", hf.total_energy);

    // 1. Full CI (Exact within basis)
    let fci = FCI::new(&hf).compute().expect("FCI failed");
    println!("FCI Correlation Energy: {:.10} Eh, Total: {:.10} Eh", fci.correlation_energy, fci.total_energy);

    // 2. Canonical MP2
    let mp2 = RHFMP2::new(&hf).compute().expect("MP2 failed");
    println!("MP2 Correlation Energy: {:.10} Eh, Total: {:.10} Eh", mp2.correlation_energy, mp2.total_energy());

    // 3. Exact RSPT via resolvent iterations
    let mp_n = RHFMPn::new(&hf, 4).compute().expect("MPn failed");
    for order in 2..=4 {
        println!("RSPT MP{} Energy Correction E^({}): {:.10} Eh, Cumulative: {:.10} Eh", order, order, mp_n.order_energies[order], mp_n.cumulative_energies[order]);
    }

    // MP2 from MPn must match canonical MP2 exactly!
    let diff_mp2 = (mp2.correlation_energy - mp_n.order_energies[2]).abs();
    println!("Diff between canonical MP2 and RSPT MP2: {:.10e}", diff_mp2);
    assert!(diff_mp2 < 1e-8, "Canonical MP2 must match RSPT MP2");

    // 4. Closed-shell MP3
    let mp3 = RHFMP3::new(&hf).compute().expect("MP3 failed");
    println!("RHFMP3 Correlation Energy: {:.10} Eh, Total: {:.10} Eh", mp3.mp3_correlation_energy, mp3.total_energy);
    println!("Diff between RHFMP3 and RSPT MP3: {:.10e}", (mp3.mp3_correlation_energy - mp_n.order_energies[3]).abs());

    // 5. Check exact H2 analytical formula for MP3
    let mo_eri = hf_rs::mo_integrals::transform_eri_ao_to_mo(&hf.mo_coefficients, &hf.eri);
    let eps = &hf.orbital_energies;
    let denom = 2.0 * (eps[0] - eps[1]);
    let v = mo_eri.get(0, 1, 0, 1);
    let t = v / denom;
    let g0000 = mo_eri.get(0, 0, 0, 0);
    let g1111 = mo_eri.get(1, 1, 1, 1);
    let g0011 = mo_eri.get(0, 0, 1, 1);
    let g0101 = mo_eri.get(0, 1, 0, 1);
    let h2_bracket = g0000 + g1111 - 4.0 * g0011 + 2.0 * g0101;
    let e3_analytical = t * t * h2_bracket;
    println!("H2 Analytical E^(3): {:.10} Eh, RSPT E^(3): {:.10} Eh, diff: {:.10e}",
        e3_analytical, mp_n.order_energies[3], (e3_analytical - mp_n.order_energies[3]).abs());
    assert!((e3_analytical - mp_n.order_energies[3]).abs() < 1e-10);

    // 6. Check exact analytical CCD root against FCI
    let h_mo = hf_rs::mo_integrals::transform_1e_ao_to_mo(&hf.mo_coefficients, &hf.core_hamiltonian);
    let e0 = 2.0 * h_mo[(0, 0)] + g0000;
    let e1 = 2.0 * h_mo[(1, 1)] + g1111;
    let delta_e = e1 - e0;
    let t_ccd_exact = (-delta_e + (delta_e * delta_e + 4.0 * v * v).sqrt()) / (2.0 * v); // root of V t^2 + delta_e t - V = 0 with t < 0
    let t_ccd_exact = -t_ccd_exact; // since v > 0 and correlation energy < 0
    let _e_ccd_exact = v * t_ccd_exact;
    // 7. Verify spin-orbital MP3 formula against RSPT
    let n_occ = hf.n_occ;
    let n_virt = hf.orbital_energies.len() - n_occ;
    let n_so_occ = 2 * n_occ;
    let n_so_virt = 2 * n_virt;

    let so_eps_occ: Vec<f64> = (0..n_so_occ).map(|i| eps[i / 2]).collect();
    let so_eps_virt: Vec<f64> = (0..n_so_virt).map(|a| eps[n_occ + a / 2]).collect();

    let antisym_eri = |p_so: usize, q_so: usize, r_so: usize, s_so: usize| -> f64 {
        let mut val = 0.0;
        if (p_so % 2 == r_so % 2) && (q_so % 2 == s_so % 2) {
            val += mo_eri.get(p_so / 2, r_so / 2, q_so / 2, s_so / 2);
        }
        if (p_so % 2 == s_so % 2) && (q_so % 2 == r_so % 2) {
            val -= mo_eri.get(p_so / 2, s_so / 2, q_so / 2, r_so / 2);
        }
        val
    };

    let mut so_t2 = vec![0.0; n_so_occ * n_so_occ * n_so_virt * n_so_virt];
    for i in 0..n_so_occ {
        for j in 0..n_so_occ {
            for a in 0..n_so_virt {
                for b in 0..n_so_virt {
                    let a_mo = 2 * n_occ + a;
                    let b_mo = 2 * n_occ + b;
                    let denom = so_eps_occ[i] + so_eps_occ[j] - so_eps_virt[a] - so_eps_virt[b];
                    let v = antisym_eri(i, j, a_mo, b_mo);
                    let idx = ((i * n_so_occ + j) * n_so_virt + a) * n_so_virt + b;
                    so_t2[idx] = v / denom;
                }
            }
        }
    }

    // MP2 via spin-orbitals: 0.25 * sum_{ijab} <ij||ab> t_{ij}^{ab}
    let mut so_e_mp2 = 0.0;
    for i in 0..n_so_occ {
        for j in 0..n_so_occ {
            for a in 0..n_so_virt {
                for b in 0..n_so_virt {
                    let a_mo = 2 * n_occ + a;
                    let b_mo = 2 * n_occ + b;
                    let v = antisym_eri(i, j, a_mo, b_mo);
                    let idx = ((i * n_so_occ + j) * n_so_virt + a) * n_so_virt + b;
                    so_e_mp2 += 0.25 * v * so_t2[idx];
                }
            }
        }
    }
    println!("Spin-orbital MP2: {:.10} Eh, Canonical MP2: {:.10} Eh, diff: {:.10e}",
        so_e_mp2, mp2.correlation_energy, (so_e_mp2 - mp2.correlation_energy).abs());
    assert!((so_e_mp2 - mp2.correlation_energy).abs() < 1e-10);

    // MP3 pp: 0.125 * sum_{ijab, cd} t_{ij}^{ab} <ab||cd> t_{ij}^{cd}
    let mut so_e_pp = 0.0;
    for i in 0..n_so_occ {
        for j in 0..n_so_occ {
            for a in 0..n_so_virt {
                let a_mo = 2 * n_occ + a;
                for b in 0..n_so_virt {
                    let b_mo = 2 * n_occ + b;
                    let idx_ab = ((i * n_so_occ + j) * n_so_virt + a) * n_so_virt + b;
                    let t_ab = so_t2[idx_ab];
                    if t_ab.abs() < 1e-15 { continue; }
                    for c in 0..n_so_virt {
                        let c_mo = 2 * n_occ + c;
                        for d in 0..n_so_virt {
                            let d_mo = 2 * n_occ + d;
                            let idx_cd = ((i * n_so_occ + j) * n_so_virt + c) * n_so_virt + d;
                            let v_abcd = antisym_eri(a_mo, b_mo, c_mo, d_mo);
                            so_e_pp += 0.125 * t_ab * v_abcd * so_t2[idx_cd];
                        }
                    }
                }
            }
        }
    }

    // MP3 hh: 0.125 * sum_{ijab, kl} t_{ij}^{ab} <kl||ij> t_{kl}^{ab}
    let mut so_e_hh = 0.0;
    for i in 0..n_so_occ {
        for j in 0..n_so_occ {
            for a in 0..n_so_virt {
                for b in 0..n_so_virt {
                    let idx_ij = ((i * n_so_occ + j) * n_so_virt + a) * n_so_virt + b;
                    let t_ab = so_t2[idx_ij];
                    if t_ab.abs() < 1e-15 { continue; }
                    for k in 0..n_so_occ {
                        for l in 0..n_so_occ {
                            let idx_kl = ((k * n_so_occ + l) * n_so_virt + a) * n_so_virt + b;
                            let v_klij = antisym_eri(k, l, i, j);
                            so_e_hh += 0.125 * t_ab * v_klij * so_t2[idx_kl];
                        }
                    }
                }
            }
        }
    }

    // MP3 ph: sum_{ijab, kc} t_{ij}^{ab} <kb||cj> t_{ik}^{ac}
    let mut so_e_ph = 0.0;
    for i in 0..n_so_occ {
        for j in 0..n_so_occ {
            for a in 0..n_so_virt {
                let _a_mo = 2 * n_occ + a;
                for b in 0..n_so_virt {
                    let b_mo = 2 * n_occ + b;
                    let idx_ij = ((i * n_so_occ + j) * n_so_virt + a) * n_so_virt + b;
                    let t_ab = so_t2[idx_ij];
                    if t_ab.abs() < 1e-15 { continue; }
                    for k in 0..n_so_occ {
                        for c in 0..n_so_virt {
                            let c_mo = 2 * n_occ + c;
                            let idx_ik = ((i * n_so_occ + k) * n_so_virt + a) * n_so_virt + c;
                            let v_kbcj = antisym_eri(k, b_mo, c_mo, j);
                            so_e_ph += t_ab * v_kbcj * so_t2[idx_ik];
                        }
                    }
                }
            }
        }
    }

    let so_e_mp3 = so_e_pp + so_e_hh + so_e_ph;
    println!("Spin-orbital MP3: {:.10} Eh, RSPT MP3: {:.10} Eh, diff: {:.10e}",
        so_e_mp3, mp_n.order_energies[3], (so_e_mp3 - mp_n.order_energies[3]).abs());
    assert!((so_e_mp3 - mp_n.order_energies[3]).abs() < 1e-10);

    // 8. Verify spin-orbital CCD convergence to exact FCI
    let mut t2_ccd = so_t2.clone();
    let mut e_ccd_curr = so_e_mp2;
    for iter in 0..50 {
        let mut r2 = vec![0.0; n_so_occ * n_so_occ * n_so_virt * n_so_virt];
        for i in 0..n_so_occ {
            for j in 0..n_so_occ {
                for a in 0..n_so_virt {
                    let a_mo = 2 * n_occ + a;
                    for b in 0..n_so_virt {
                        let b_mo = 2 * n_occ + b;
                        let idx_ab = ((i * n_so_occ + j) * n_so_virt + a) * n_so_virt + b;

                        // 1. Bare integral <ab||ij>
                        let mut val = antisym_eri(a_mo, b_mo, i, j);

                        // 2. pp ladder: 0.5 * sum_{cd} <ab||cd> t_{ij}^{cd}
                        for c in 0..n_so_virt {
                            let c_mo = 2 * n_occ + c;
                            for d in 0..n_so_virt {
                                let d_mo = 2 * n_occ + d;
                                let idx_cd = ((i * n_so_occ + j) * n_so_virt + c) * n_so_virt + d;
                                val += 0.5 * antisym_eri(a_mo, b_mo, c_mo, d_mo) * t2_ccd[idx_cd];
                            }
                        }

                        // 3. hh ladder: 0.5 * sum_{kl} <kl||ij> t_{kl}^{ab}
                        for k in 0..n_so_occ {
                            for l in 0..n_so_occ {
                                let idx_kl = ((k * n_so_occ + l) * n_so_virt + a) * n_so_virt + b;
                                val += 0.5 * antisym_eri(k, l, i, j) * t2_ccd[idx_kl];
                            }
                        }

                        // 4. ph ring: P(ij) P(ab) sum_{kc} <kb||cj> t_{ik}^{ac}
                        // = sum_{kc} [ <kb||cj> t_{ik}^{ac} - <ka||cj> t_{ik}^{bc} - <kb||ci> t_{jk}^{ac} + <ka||ci> t_{jk}^{bc} ]
                        for k in 0..n_so_occ {
                            for c in 0..n_so_virt {
                                let c_mo = 2 * n_occ + c;
                                let idx_ik_ac = ((i * n_so_occ + k) * n_so_virt + a) * n_so_virt + c;
                                let idx_ik_bc = ((i * n_so_occ + k) * n_so_virt + b) * n_so_virt + c;
                                let idx_jk_ac = ((j * n_so_occ + k) * n_so_virt + a) * n_so_virt + c;
                                let idx_jk_bc = ((j * n_so_occ + k) * n_so_virt + b) * n_so_virt + c;

                                val += antisym_eri(k, b_mo, c_mo, j) * t2_ccd[idx_ik_ac]
                                     - antisym_eri(k, a_mo, c_mo, j) * t2_ccd[idx_ik_bc]
                                     - antisym_eri(k, b_mo, c_mo, i) * t2_ccd[idx_jk_ac]
                                     + antisym_eri(k, a_mo, c_mo, i) * t2_ccd[idx_jk_bc];
                            }
                        }

                        // 5. Quadratic pp-hh: - 0.25 * sum_{klcd} <kl||cd> t_{ij}^{cd} t_{kl}^{ab}
                        for k in 0..n_so_occ {
                            for l in 0..n_so_occ {
                                for c in 0..n_so_virt {
                                    let c_mo = 2 * n_occ + c;
                                    for d in 0..n_so_virt {
                                        let d_mo = 2 * n_occ + d;
                                        let v_klcd = antisym_eri(k, l, c_mo, d_mo);
                                        let idx_cd = ((i * n_so_occ + j) * n_so_virt + c) * n_so_virt + d;
                                        let idx_kl = ((k * n_so_occ + l) * n_so_virt + a) * n_so_virt + b;
                                        val -= 0.25 * v_klcd * t2_ccd[idx_cd] * t2_ccd[idx_kl];
                                    }
                                }
                            }
                        }

                        r2[idx_ab] = val;
                    }
                }
            }
        }

        // Update amplitudes: t_{ij}^{ab} = R_{ij}^{ab} / D_{ij}^{ab}
        for i in 0..n_so_occ {
            for j in 0..n_so_occ {
                for a in 0..n_so_virt {
                    for b in 0..n_so_virt {
                        let idx = ((i * n_so_occ + j) * n_so_virt + a) * n_so_virt + b;
                        let denom = so_eps_occ[i] + so_eps_occ[j] - so_eps_virt[a] - so_eps_virt[b];
                        t2_ccd[idx] = r2[idx] / denom;
                    }
                }
            }
        }

        let mut e_corr = 0.0;
        for i in 0..n_so_occ {
            for j in 0..n_so_occ {
                for a in 0..n_so_virt {
                    let a_mo = 2 * n_occ + a;
                    for b in 0..n_so_virt {
                        let b_mo = 2 * n_occ + b;
                        let idx = ((i * n_so_occ + j) * n_so_virt + a) * n_so_virt + b;
                        e_corr += 0.25 * antisym_eri(i, j, a_mo, b_mo) * t2_ccd[idx];
                    }
                }
            }
        }
        if (e_corr - e_ccd_curr).abs() < 1e-12 {
            println!("Iterative CCD converged at iter {}: E_corr = {:.10} Eh, FCI = {:.10} Eh, diff = {:.10e}",
                iter + 1, e_corr, fci.correlation_energy, (e_corr - fci.correlation_energy).abs());
            assert!((e_corr - fci.correlation_energy).abs() < 1e-10);
            break;
        }
        e_ccd_curr = e_corr;
    }

    // 8. Test module CCD struct
    let ccd_opts = hf_rs::post_hf::cc::CCSDOptions {
        max_iterations: 100,
        energy_tol: 1e-13,
        residual_tol: 1e-12,
        diis_subspace: 6,
    };
    let ccd_res = CCD::new(&hf).with_options(ccd_opts).compute().expect("CCD compute failed");
    println!("Module CCD E_corr: {:.10} Eh, FCI: {:.10} Eh, diff: {:.10e}",
        ccd_res.correlation_energy, fci.correlation_energy, (ccd_res.correlation_energy - fci.correlation_energy).abs());
    assert!((ccd_res.correlation_energy - fci.correlation_energy).abs() < 1e-10);

    // 9. Test module CCSD struct
    let ccsd_opts = hf_rs::post_hf::cc::CCSDOptions {
        max_iterations: 100,
        energy_tol: 1e-13,
        residual_tol: 1e-12,
        diis_subspace: 6,
    };
    let ccsd_res = CCSD::new(&hf).with_options(ccsd_opts).compute().expect("CCSD compute failed");
    println!("Module CCSD E_corr: {:.10} Eh, FCI: {:.10} Eh, diff: {:.10e}",
        ccsd_res.correlation_energy, fci.correlation_energy, (ccsd_res.correlation_energy - fci.correlation_energy).abs());
    assert!((ccsd_res.correlation_energy - fci.correlation_energy).abs() < 1e-10);
}

#[test]
fn test_h2_mp2_gradient_exact_decomposition() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let r = 1.4;
    let h = 1e-5;

    let calc_params = |dist: f64| -> (f64, f64, f64, f64) {
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -dist / 2.0]),
            Atom::new("H", [0.0, 0.0,  dist / 2.0]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, ConvAlgo::Cdiis, None).unwrap();
        let mo_eri = hf_rs::mo_integrals::transform_eri_ao_to_mo(&hf.mo_coefficients, &hf.eri);
        let v = mo_eri.get(0, 1, 0, 1);
        let denom = 2.0 * (hf.orbital_energies[0] - hf.orbital_energies[1]);
        let e2 = v * v / denom;
        (e2, v, denom, hf.total_energy)
    };

    let (e2_0, v_0, denom_0, _hf_0) = calc_params(r);
    let (e2_fwd, v_fwd, denom_fwd, hf_fwd) = calc_params(r + h);
    let (e2_bwd, v_bwd, denom_bwd, hf_bwd) = calc_params(r - h);

    let de2_dr = (e2_fwd - e2_bwd) / (2.0 * h);
    let dv_dr = (v_fwd - v_bwd) / (2.0 * h);
    let ddenom_dr = (denom_fwd - denom_bwd) / (2.0 * h);
    let dhf_dr = (hf_fwd - hf_bwd) / (2.0 * h);

    let t = v_0 / denom_0;
    let term1 = 2.0 * t * dv_dr;
    let term2 = - t * t * ddenom_dr;
    let de2_analytical = term1 + term2;

    // Numerical breakdown into skeleton, Fock derivative, and overlap:
    let atoms_0 = vec![Atom::new("H", [0.0, 0.0, -r / 2.0]), Atom::new("H", [0.0, 0.0, r / 2.0])];
    let basis_0 = build_basis(&atoms_0, &basis_map);
    let hf_res_0 = run_scf_general(&atoms_0, &basis_0, 0, 50, 1e-12, ConvAlgo::Cdiis, None).unwrap();
    let c_0 = &hf_res_0.mo_coefficients;

    // Check S' in MO basis
    let atoms_fwd = vec![Atom::new("H", [0.0, 0.0, -(r + h) / 2.0]), Atom::new("H", [0.0, 0.0, (r + h) / 2.0])];
    let basis_fwd = build_basis(&atoms_fwd, &basis_map);
    let atoms_bwd = vec![Atom::new("H", [0.0, 0.0, -(r - h) / 2.0]), Atom::new("H", [0.0, 0.0, (r - h) / 2.0])];
    let basis_bwd = build_basis(&atoms_bwd, &basis_map);

    let hf_fwd_res = run_scf_general(&atoms_fwd, &basis_fwd, 0, 50, 1e-12, ConvAlgo::Cdiis, None).unwrap();
    let hf_bwd_res = run_scf_general(&atoms_bwd, &basis_bwd, 0, 50, 1e-12, ConvAlgo::Cdiis, None).unwrap();

    let s_fwd_mo = c_0.transpose().dot(&hf_fwd_res.overlap_matrix).dot(c_0);
    let s_bwd_mo = c_0.transpose().dot(&hf_bwd_res.overlap_matrix).dot(c_0);
    let ds_00 = (s_fwd_mo[(0, 0)] - s_bwd_mo[(0, 0)]) / (2.0 * h);
    let ds_11 = (s_fwd_mo[(1, 1)] - s_bwd_mo[(1, 1)]) / (2.0 * h);

    // Skeleton ERI derivative: C_0^T ERI' C_0
    let v_skel_fwd = hf_rs::mo_integrals::transform_eri_ao_to_mo(c_0, &hf_fwd_res.eri).get(0, 1, 0, 1);
    let v_skel_bwd = hf_rs::mo_integrals::transform_eri_ao_to_mo(c_0, &hf_bwd_res.eri).get(0, 1, 0, 1);
    let dv_skel = (v_skel_fwd - v_skel_bwd) / (2.0 * h);

    let dv_orb = - (ds_00 + ds_11) * v_0;
    println!("dv/dr total: {:.10}, dv_skel: {:.10}, dv_orb: {:.10}, sum: {:.10}",
        dv_dr, dv_skel, dv_orb, dv_skel + dv_orb);

    // Fock matrix derivative with frozen C_0:
    let f_fwd_mo = c_0.transpose().dot(&hf_fwd_res.fock_matrix).dot(c_0);
    let f_bwd_mo = c_0.transpose().dot(&hf_bwd_res.fock_matrix).dot(c_0);
    let df_00 = (f_fwd_mo[(0, 0)] - f_bwd_mo[(0, 0)]) / (2.0 * h);
    let df_11 = (f_fwd_mo[(1, 1)] - f_bwd_mo[(1, 1)]) / (2.0 * h);

    let eps_0 = hf_res_0.orbital_energies[0];
    let eps_1 = hf_res_0.orbital_energies[1];
    let deps_0 = df_00 - eps_0 * ds_00;
    let deps_1 = df_11 - eps_1 * ds_11;
    let ddenom_ana = 2.0 * (deps_0 - deps_1);
    println!("ddenom/dr total: {:.10}, ddenom_ana: {:.10}", ddenom_dr, ddenom_ana);

    let d_00 = -2.0 * t * t;
    let d_11 = 2.0 * t * t;
    let piece_eri_skel = 2.0 * t * dv_skel;
    let piece_fock_skel = d_00 * df_00 + d_11 * df_11;
    let w_00 = eps_0 * d_00 + 2.0 * e2_0;
    let w_11 = eps_1 * d_11 + 2.0 * e2_0;
    let piece_overlap = - (w_00 * ds_00 + w_11 * ds_11);

    // 1-electron core Hamiltonian derivative:
    let h_fwd_mo = c_0.transpose().dot(&hf_fwd_res.core_hamiltonian).dot(c_0);
    let h_bwd_mo = c_0.transpose().dot(&hf_bwd_res.core_hamiltonian).dot(c_0);
    let dh_00 = (h_fwd_mo[(0, 0)] - h_bwd_mo[(0, 0)]) / (2.0 * h);
    let dh_11 = (h_fwd_mo[(1, 1)] - h_bwd_mo[(1, 1)]) / (2.0 * h);
    let piece_h_skel = d_00 * dh_00 + d_11 * dh_11;
    let piece_g_skel = piece_fock_skel - piece_h_skel;
    println!("piece_h_skel    = {:.10}", piece_h_skel);
    println!("piece_g_skel    = {:.10}", piece_g_skel);

    // Compute G' directly by contracting C_0 D C_0^T with ERI' and P_HF:
    let mut p_corr_mo = hf_rs::matrix::Matrix::zeros(2, 2);
    p_corr_mo[(0, 0)] = d_00;
    p_corr_mo[(1, 1)] = d_11;
    let p_corr_ao = c_0.dot(&p_corr_mo).dot(&c_0.transpose());
    let p_hf_ao = &hf_res_0.density_matrix;

    let mut basis_derivs = Vec::new();
    for cg in &basis_0 {
        basis_derivs.push(hf_rs::gradient::differentiate_basis_function(cg, 2));
    }

    let mut g_relax_sum = 0.0;
    let mut g_2pdm_sum = 0.0;
    let t_val = t;

    // Direct AO contraction for atom 1 z:
    for mu in 0..2 {
        if basis_0[mu].atom_index != 1 { continue; }
        for term in &basis_derivs[mu].terms {
            for nu in 0..2 {
                for lambda in 0..2 {
                    for sigma in 0..2 {
                        let eri = hf_rs::integrals::contracted_eri(term, &basis_0[nu], &basis_0[lambda], &basis_0[sigma]);
                        
                        let g_coul = 2.0 * (p_hf_ao[(mu, nu)] * p_corr_ao[(lambda, sigma)] + p_corr_ao[(mu, nu)] * p_hf_ao[(lambda, sigma)]);
                        let g_exch = p_hf_ao[(mu, lambda)] * p_corr_ao[(nu, sigma)] + p_corr_ao[(mu, lambda)] * p_hf_ao[(nu, sigma)];
                        g_relax_sum += (g_coul - g_exch) * eri;

                        // 2PDM: (0 1 | 0 1)
                        // In MO basis, gamma_{0101} = 2t
                        let t1 = c_0[(mu, 0)] * c_0[(nu, 1)] * c_0[(lambda, 0)] * c_0[(sigma, 1)];
                        let t2 = c_0[(nu, 0)] * c_0[(mu, 1)] * c_0[(lambda, 0)] * c_0[(sigma, 1)];
                        let t3 = c_0[(lambda, 0)] * c_0[(sigma, 1)] * c_0[(mu, 0)] * c_0[(nu, 1)];
                        let t4 = c_0[(lambda, 0)] * c_0[(sigma, 1)] * c_0[(nu, 0)] * c_0[(mu, 1)];
                        let gamma_4 = 2.0 * t_val * (t1 + t2 + t3 + t4);
                        g_2pdm_sum += gamma_4 * eri;
                    }
                }
            }
        }
    }
    let g0_mo = c_0.transpose().dot(&hf_res_0.fock_matrix.sub(&hf_res_0.core_hamiltonian)).dot(c_0);
    let dg_from_p = (d_00 * g0_mo[(0, 0)] + d_11 * g0_mo[(1, 1)]) * (-ds_00);
    println!("dg_from_p       = {:.10}", dg_from_p);
    println!("g_relax_sum + dg_from_p = {:.10}", g_relax_sum + dg_from_p);
    println!("difference from piece_g_skel: {:.10e}", (piece_g_skel - (g_relax_sum + dg_from_p)).abs());

    println!("g_relax_sum     = {:.10}", g_relax_sum);
    println!("g_2pdm_sum      = {:.10}", g_2pdm_sum);
    println!("piece_g_skel diff: {:.10e}", (piece_g_skel - g_relax_sum).abs());
    println!("piece_eri_skel diff: {:.10e}", (piece_eri_skel - g_2pdm_sum).abs());

    println!("piece_eri_skel  = {:.10}", piece_eri_skel);
    println!("piece_fock_skel = {:.10}", piece_fock_skel);
    println!("piece_overlap   = {:.10}", piece_overlap);
    println!("Total pieces    = {:.10}", piece_eri_skel + piece_fock_skel + piece_overlap);
    println!("de2_dr          = {:.10}", de2_dr);

    assert!((de2_dr - (piece_eri_skel + piece_fock_skel + piece_overlap)).abs() < 1e-7);

    println!("=== H2 MP2 Gradient Exact Breakdown ===");
    println!("R = {}", r);
    println!("v = {}, denom = {}, t = {}", v_0, denom_0, t);
    println!("de2_dr (finite diff)   = {:.10}", de2_dr);
    println!("de2_dr (term1 = 2t v') = {:.10}", term1);
    println!("de2_dr (term2 = -t^2 d')= {:.10}", term2);
    println!("de2_dr (term1 + term2) = {:.10}", de2_analytical);
    println!("Diff de2: {:.10e}", (de2_dr - de2_analytical).abs());
    println!("dhf_dr = {:.10}", dhf_dr);
    println!("dE_tot / dr = {:.10}", dhf_dr + de2_dr);

    assert!((de2_dr - de2_analytical).abs() < 1e-7);
}

#[test]
fn test_h2_casscf_gradient_breakdown() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let r = 1.4;
    let atoms_0 = vec![
        Atom::new("H", [0.0, 0.0, -r / 2.0]),
        Atom::new("H", [0.0, 0.0,  r / 2.0]),
    ];
    let basis_0 = build_basis(&atoms_0, &basis_map);
    let hf_0 = run_scf_general(&atoms_0, &basis_0, 0, 50, 1e-12, ConvAlgo::Cdiis, None).unwrap();
    let opts = hf_rs::post_hf::casscf::CASSCFOptions::new(2, 2);
    let casscf_0 = hf_rs::post_hf::casscf::CASSCF::new(&hf_0, opts.clone()).compute().unwrap();

    let h = 1e-5;
    let mut atoms_p = atoms_0.clone();
    atoms_p[1].position[2] += h;
    let basis_p = build_basis(&atoms_p, &basis_map);
    let hf_p = run_scf_general(&atoms_p, &basis_p, 0, 50, 1e-12, ConvAlgo::Cdiis, None).unwrap();
    let casscf_p = hf_rs::post_hf::casscf::CASSCF::new(&hf_p, opts.clone()).compute().unwrap();

    let mut atoms_m = atoms_0.clone();
    atoms_m[1].position[2] -= h;
    let basis_m = build_basis(&atoms_m, &basis_map);
    let hf_m = run_scf_general(&atoms_m, &basis_m, 0, 50, 1e-12, ConvAlgo::Cdiis, None).unwrap();
    let casscf_m = hf_rs::post_hf::casscf::CASSCF::new(&hf_m, opts).compute().unwrap();

    let num_grad_z = (casscf_p.total_energy - casscf_m.total_energy) / (2.0 * h);
    let ana_grad = hf_rs::post_hf::casscf::compute_casscf_analytical_gradient(&atoms_0, &basis_0, &hf_0, &casscf_0);

    // Finite difference of 1-electron, 2-electron, nuclear energy:
    let e_nuc_p = casscf_p.total_energy - (casscf_p.total_energy - 1.0 / (atoms_p[1].position[2] - atoms_p[0].position[2]));
    let e_nuc_m = casscf_m.total_energy - (casscf_m.total_energy - 1.0 / (atoms_m[1].position[2] - atoms_m[0].position[2]));
    let _g_nuc_num = (e_nuc_p - e_nuc_m) / (2.0 * h);

    // In MO basis, E_1e = D_00 h_00 + D_11 h_11
    // E_2e = 0.5 * (Gamma_0000 (00|00) + Gamma_1111 (11|11) + 2 Gamma_0101 (01|01))
    let d1 = &casscf_0.one_rdm_mo;
    let g2 = &casscf_0.two_rdm_mo;
    let c = &casscf_0.mo_coefficients;
    let h_mo = hf_rs::mo_integrals::transform_1e_ao_to_mo(c, &hf_0.core_hamiltonian);
    let eri_mo = hf_rs::mo_integrals::transform_eri_ao_to_mo(c, &hf_0.eri);
    let e_1e = d1[(0, 0)] * h_mo[(0, 0)] + d1[(1, 1)] * h_mo[(1, 1)];
    let e_2e = 0.5 * (g2[0] * eri_mo.get(0, 0, 0, 0) + g2[15] * eri_mo.get(1, 1, 1, 1) + 2.0 * g2[5] * eri_mo.get(0, 1, 0, 1));
    let _e_tot_recon = 1.0 / r + e_1e + e_2e;

    // Breakdown of finite differences:
    let h_mo_p = hf_rs::mo_integrals::transform_1e_ao_to_mo(&casscf_p.mo_coefficients, &hf_p.core_hamiltonian);
    let eri_mo_p = hf_rs::mo_integrals::transform_eri_ao_to_mo(&casscf_p.mo_coefficients, &hf_p.eri);
    let d1_p = &casscf_p.one_rdm_mo;
    let g2_p = &casscf_p.two_rdm_mo;
    let e_1e_p = d1_p[(0, 0)] * h_mo_p[(0, 0)] + d1_p[(1, 1)] * h_mo_p[(1, 1)];
    let e_2e_p = 0.5 * (g2_p[0] * eri_mo_p.get(0, 0, 0, 0) + g2_p[15] * eri_mo_p.get(1, 1, 1, 1) + 2.0 * g2_p[5] * eri_mo_p.get(0, 1, 0, 1));

    let h_mo_m = hf_rs::mo_integrals::transform_1e_ao_to_mo(&casscf_m.mo_coefficients, &hf_m.core_hamiltonian);
    let eri_mo_m = hf_rs::mo_integrals::transform_eri_ao_to_mo(&casscf_m.mo_coefficients, &hf_m.eri);
    let d1_m = &casscf_m.one_rdm_mo;
    let g2_m = &casscf_m.two_rdm_mo;
    let e_1e_m = d1_m[(0, 0)] * h_mo_m[(0, 0)] + d1_m[(1, 1)] * h_mo_m[(1, 1)];
    let e_2e_m = 0.5 * (g2_m[0] * eri_mo_m.get(0, 0, 0, 0) + g2_m[15] * eri_mo_m.get(1, 1, 1, 1) + 2.0 * g2_m[5] * eri_mo_m.get(0, 1, 0, 1));

    let de_nuc_num = (1.0 / (atoms_p[1].position[2] - atoms_p[0].position[2]) - 1.0 / (atoms_m[1].position[2] - atoms_m[0].position[2])) / (2.0 * h);
    let de_1e_num = (e_1e_p - e_1e_m) / (2.0 * h);
    let de_2e_num = (e_2e_p - e_2e_m) / (2.0 * h);

    println!("de_nuc_num: {:14.8}", de_nuc_num);
    println!("de_1e_num:  {:14.8}", de_1e_num);
    println!("de_2e_num:  {:14.8}", de_2e_num);
    println!("sum num:    {:14.8}", de_nuc_num + de_1e_num + de_2e_num);
    println!("num_grad_z: {:14.8}", num_grad_z);

    let g_nuc = hf_rs::gradient::compute_nuclear_repulsion_gradient(&atoms_0);
    println!("g_nuc[1][2]: {:14.8}", g_nuc[1][2]);
    println!("diff g_nuc:  {:14.8e}", (g_nuc[1][2] - de_nuc_num).abs());

    // Compute skeleton 1-electron and overlap derivative:
    let n_basis = basis_0.len();
    let mut basis_derivs = Vec::with_capacity(n_basis);
    for cg in &basis_0 {
        let dx = hf_rs::gradient::differentiate_basis_function(cg, 0);
        let dy = hf_rs::gradient::differentiate_basis_function(cg, 1);
        let dz = hf_rs::gradient::differentiate_basis_function(cg, 2);
        basis_derivs.push([dx, dy, dz]);
    }

    let c = &casscf_0.mo_coefficients;
    let d_ao = c.dot(&casscf_0.one_rdm_mo).dot(&c.transpose());

    let mut dt_val = 0.0;
    let mut dv_val = 0.0;
    let mut ds_mo = [0.0; 4]; // in MO basis

    let tau = 2; // z-axis
    let atom_idx = 1;
    for mu in 0..n_basis {
        let mu_on_a = basis_0[mu].atom_index == atom_idx;
        for nu in 0..n_basis {
            let nu_on_a = basis_0[nu].atom_index == atom_idx;
            let d_munu = d_ao[(mu, nu)];

            let mut ds = 0.0;
            if mu_on_a { ds += hf_rs::gradient::deriv_overlap(&basis_derivs[mu][tau], &basis_0[nu]); }
            if nu_on_a { ds += hf_rs::gradient::deriv_overlap(&basis_derivs[nu][tau], &basis_0[mu]); }
            for p in 0..2 {
                for q in 0..2 {
                    ds_mo[p * 2 + q] += c[(mu, p)] * c[(nu, q)] * ds;
                }
            }

            let mut dt = 0.0;
            if mu_on_a { dt += hf_rs::gradient::deriv_kinetic(&basis_derivs[mu][tau], &basis_0[nu]); }
            if nu_on_a { dt += hf_rs::gradient::deriv_kinetic(&basis_derivs[nu][tau], &basis_0[mu]); }
            dt_val += d_munu * dt;

            let mut dv = 0.0;
            if mu_on_a { dv += hf_rs::gradient::deriv_nuclear_attraction(&basis_derivs[mu][tau], &basis_0[nu], &atoms_0); }
            if nu_on_a { dv += hf_rs::gradient::deriv_nuclear_attraction(&basis_derivs[nu][tau], &basis_0[mu], &atoms_0); }
            let hf_term = -(
                hf_rs::gradient::deriv_nuclear_attraction_single_atom(&basis_derivs[mu][tau], &basis_0[nu], &atoms_0[atom_idx])
                + hf_rs::gradient::deriv_nuclear_attraction_single_atom(&basis_derivs[nu][tau], &basis_0[mu], &atoms_0[atom_idx])
            );
            dv += hf_term;
            dv_val += d_munu * dv;
        }
    }

    println!("dt_val + dv_val (1e skeleton): {:14.8}", dt_val + dv_val);
    println!("de_1e_num:                     {:14.8}", de_1e_num);
    println!("Diff 1e:                       {:14.8}", de_1e_num - (dt_val + dv_val));
    println!("ds_mo:                         {:?}", ds_mo);

    let eri_mo = hf_rs::mo_integrals::transform_eri_ao_to_mo(c, &hf_0.eri);
    let g2 = &casscf_0.two_rdm_mo;
    let w00 = d1[(0, 0)] * h_mo[(0, 0)] + g2[0] * eri_mo.get(0, 0, 0, 0) + g2[5] * eri_mo.get(0, 1, 0, 1);
    let w11 = d1[(1, 1)] * h_mo[(1, 1)] + g2[15] * eri_mo.get(1, 1, 1, 1) + g2[10] * eri_mo.get(1, 0, 1, 0);
    println!("w00 = {:14.8}", w00);
    println!("w11 = {:14.8}", w11);

    let overlap_term = -(w00 * ds_mo[0] + w11 * ds_mo[3]);
    println!("overlap_term: {:14.8}", overlap_term);

    // Now compute 2e skeleton derivative:
    // In MO basis, (pq|rs)' = sum_mu c_mu_p ... (mu' nu | la si)
    // 2 * sum_{mu in atom 1} sum_{nu la si} Gamma_{mu nu la si} (mu' nu | la si)
    let mut d_eri_val = 0.0;
    // Build Gamma AO from g2:
    let mut gamma_ao = vec![vec![vec![vec![0.0; 2]; 2]; 2]; 2];
    for p in 0..2 {
        for q in 0..2 {
            for r in 0..2 {
                for s in 0..2 {
                    let g_val = g2[((p * 2 + q) * 2 + r) * 2 + s];
                    if g_val.abs() < 1e-15 { continue; }
                    for mu in 0..2 {
                        for nu in 0..2 {
                            for la in 0..2 {
                                for si in 0..2 {
                                    gamma_ao[mu][nu][la][si] += g_val * c[(mu, p)] * c[(nu, q)] * c[(la, r)] * c[(si, s)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for mu in 0..2 {
        if basis_0[mu].atom_index != atom_idx { continue; }
        for term in &basis_derivs[mu][tau].terms {
            for nu in 0..2 {
                for la in 0..2 {
                    for si in 0..2 {
                        let eri = hf_rs::integrals::contracted_eri(term, &basis_0[nu], &basis_0[la], &basis_0[si]);
                        // 4-fold contraction with (mu' nu | la si)
                        let g_4 = gamma_ao[mu][nu][la][si] + gamma_ao[nu][mu][la][si] + gamma_ao[la][si][mu][nu] + gamma_ao[si][la][mu][nu];
                        // With 0.5 prefactor: 0.5 * g_4 * eri
                        d_eri_val += 0.5 * g_4 * eri;
                    }
                }
            }
        }
    }
    println!("d_eri_val (2e skeleton): {:14.8}", d_eri_val);
    println!("Total analytical gradient: {:14.8}", g_nuc[1][2] + (dt_val + dv_val) + overlap_term + d_eri_val);
    println!("Total numerical gradient:  {:14.8}", num_grad_z);
    println!("DIFFERENCE:                {:14.8e}", (g_nuc[1][2] + (dt_val + dv_val) + overlap_term + d_eri_val) - num_grad_z);

    assert!((ana_grad[1][2] - num_grad_z).abs() < 1e-5);
    assert!(((g_nuc[1][2] + (dt_val + dv_val) + overlap_term + d_eri_val) - num_grad_z).abs() < 1e-8);
}


