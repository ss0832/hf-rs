// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Perturbative Triples (T) correction for Coupled Cluster: CCSD(T).
//!
//! Evaluates the non-iterative O(N^7) connected triples correlation energy Delta E_(T)
//! from converged CCSD T1 and T2 amplitudes.

use crate::mo_integrals::{MoEriTensor, transform_eri_ao_to_mo};
use crate::post_hf::PostHFResult;
use crate::post_hf::cc::CCSDResult;
use crate::scf::HFResult;
use rayon::prelude::*;

/// Results of a CCSD(T) calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct CCSDTResult {
    pub reference_energy: f64,
    pub ccsd_correlation_energy: f64,
    pub triples_correlation_energy: f64,
    pub total_energy: f64,
}

impl PostHFResult for CCSDTResult {
    fn method_name(&self) -> &'static str {
        "CCSD(T)"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.ccsd_correlation_energy + self.triples_correlation_energy
    }

    fn total_energy(&self) -> f64 {
        self.total_energy
    }
}

/// Computes the non-iterative (T) triples correction from a converged RHF-CCSD result.
pub fn compute_ccsd_t_correction(hf: &HFResult, ccsd: &CCSDResult) -> Result<CCSDTResult, String> {
    let mo = transform_eri_ao_to_mo(&hf.mo_coefficients, &hf.eri);
    compute_ccsd_t_correction_with_integrals(hf, ccsd, &mo)
}

/// Triples energy using the shared context's already transformed integrals.
pub fn compute_ccsd_t_correction_with_integrals(
    hf: &HFResult,
    ccsd: &CCSDResult,
    mo_eri: &MoEriTensor,
) -> Result<CCSDTResult, String> {
    compute_ccsd_t_correction_in_space(&hf.orbital_energies, ccsd, mo_eri)
}

/// The supplied MO integrals and energies contain only correlated occupied and
/// retained virtual orbitals; the reference energy still includes frozen cores.
pub(crate) fn compute_ccsd_t_correction_in_space(
    orbital_energies: &[f64],
    ccsd: &CCSDResult,
    mo_eri: &MoEriTensor,
) -> Result<CCSDTResult, String> {
    if !ccsd.converged {
        return Err("CCSD(T) requires converged CCSD amplitudes".into());
    }
    let n_occ = ccsd.n_occ;
    let n_virt = ccsd.n_virt;
    let nm = n_occ + n_virt;
    if orbital_energies.len() != nm
        || mo_eri.n_basis != nm
        || ccsd.t1_amplitudes.len() != n_occ * n_virt
        || ccsd.t2_amplitudes.len() != n_occ * n_occ * n_virt * n_virt
        || orbital_energies
            .iter()
            .chain(&ccsd.t1_amplitudes)
            .chain(&ccsd.t2_amplitudes)
            .any(|v| !v.is_finite())
    {
        return Err("CCSD(T) requires consistent finite energies, integrals and amplitudes".into());
    }

    if n_occ == 0 || n_virt == 0 {
        return Err("CCSD(T) requires at least one occupied and one virtual orbital".to_string());
    }

    if 2 * n_occ < 3 || 2 * n_virt < 3 {
        // By Pauli exclusion, connected triples vanish identically for systems with < 3 electrons or < 3 virtuals
        return Ok(CCSDTResult {
            reference_energy: ccsd.reference_energy,
            ccsd_correlation_energy: ccsd.correlation_energy,
            triples_correlation_energy: 0.0,
            total_energy: ccsd.total_energy,
        });
    }

    if ccsd.t2_amplitudes.is_empty() {
        return Err("CCSD(T) requires converged T2 amplitudes".to_string());
    }

    let eps = orbital_energies;
    let highest_occupied = eps[..n_occ]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let lowest_virtual = eps[n_occ..].iter().copied().fold(f64::INFINITY, f64::min);
    if lowest_virtual - highest_occupied <= 1e-12 {
        return Err("CCSD(T) requires a positive occupied/virtual gap".into());
    }

    // Parallel accumulation of (T) energy across occupied index i
    let e_triples: f64 = (0..n_occ)
        .into_par_iter()
        .map(|i| {
            let mut local_t = 0.0;
            for j in 0..n_occ {
                for k in 0..n_occ {
                    let (w, v) = triples_sources(
                        n_occ,
                        n_virt,
                        &ccsd.t1_amplitudes,
                        &ccsd.t2_amplitudes,
                        mo_eri,
                        i,
                        j,
                        k,
                    );
                    let at = |a, b, c| w[(a * n_virt + b) * n_virt + c];
                    for a in 0..n_virt {
                        let a_mo = n_occ + a;
                        for b in 0..n_virt {
                            let b_mo = n_occ + b;
                            for c in 0..n_virt {
                                let c_mo = n_occ + c;
                                let denom =
                                    eps[i] + eps[j] + eps[k] - eps[a_mo] - eps[b_mo] - eps[c_mo];

                                let w_abc = at(a, b, c);
                                let w_acb = at(a, c, b);
                                let w_bac = at(b, a, c);
                                let w_cba = at(c, b, a);
                                let w_bca = at(b, c, a);
                                let w_cab = at(c, a, b);

                                let w_tilde = 4.0 * w_abc - 2.0 * w_acb - 2.0 * w_bac - 2.0 * w_cba
                                    + w_bca
                                    + w_cab;
                                let v_abc = v[(a * n_virt + b) * n_virt + c];

                                // Ordered spatial triples: 1/3, with three singles/ERI
                                // terms. The prior 1/6 prefactor halved the connected term.
                                local_t += (1.0 / 3.0) * w_tilde * (w_abc + v_abc) / denom;
                            }
                        }
                    }
                }
            }
            local_t
        })
        .sum();

    let ref_e = ccsd.reference_energy;
    let ccsd_e = ccsd.correlation_energy;
    let total_e = ref_e + ccsd_e + e_triples;

    Ok(CCSDTResult {
        reference_energy: ref_e,
        ccsd_correlation_energy: ccsd_e,
        triples_correlation_energy: e_triples,
        total_energy: total_e,
    })
}

/// Connected and singles triples numerators in any orthonormal occupied/virtual
/// bases. Singles have three terms; all six simultaneous permutations enter W.
pub(crate) fn triples_sources(
    no: usize,
    nv: usize,
    t1: &[f64],
    t2: &[f64],
    mo_eri: &MoEriTensor,
    i: usize,
    j: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let calc_d = |i: usize, j: usize, k: usize, a: usize, b: usize, c: usize| -> f64 {
        let b_mo = no + b;
        let c_mo = no + c;
        let mut d_val = 0.0;
        for e in 0..nv {
            let e_mo = no + e;
            let ck_eb = mo_eri.get(c_mo, k, e_mo, b_mo);
            let idx_ij_ae = ((i * no + j) * nv + a) * nv + e;
            d_val += t2[idx_ij_ae] * ck_eb;
        }
        for m in 0..no {
            let ck_mj = mo_eri.get(c_mo, k, m, j);
            let idx_im_ab = ((i * no + m) * nv + a) * nv + b;
            d_val -= t2[idx_im_ab] * ck_mj;
        }
        d_val
    };

    let calc_w = |i: usize, j: usize, k: usize, a: usize, b: usize, c: usize| -> f64 {
        calc_d(i, j, k, a, b, c)
            + calc_d(j, k, i, b, c, a)
            + calc_d(k, i, j, c, a, b)
            + calc_d(i, k, j, a, c, b)
            + calc_d(k, j, i, c, b, a)
            + calc_d(j, i, k, b, a, c)
    };

    let calc_v = |i: usize, j: usize, k: usize, a: usize, b: usize, c: usize| -> f64 {
        if t1.is_empty() {
            return 0.0;
        }
        let a_mo = no + a;
        let b_mo = no + b;
        let c_mo = no + c;
        t1[i * nv + a] * mo_eri.get(j, b_mo, k, c_mo)
            + t1[j * nv + b] * mo_eri.get(k, c_mo, i, a_mo)
            + t1[k * nv + c] * mo_eri.get(i, a_mo, j, b_mo)
    };

    let mut connected = Vec::with_capacity(nv * nv * nv);
    let mut singles = Vec::with_capacity(nv * nv * nv);
    for a in 0..nv {
        for b in 0..nv {
            for c in 0..nv {
                connected.push(calc_w(i, j, k, a, b, c));
                singles.push(calc_v(i, j, k, a, b, c));
            }
        }
    }
    (connected, singles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{Atom, STO_3G_GBS, build_basis, parse_gaussian94};
    use crate::diis::ConvAlgo;
    use crate::post_hf::PostHFMethod;
    use crate::post_hf::cc::CCSD;
    use crate::scf::run_scf_general;

    #[test]
    fn test_h2_ccsd_t() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0, 0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, ConvAlgo::Adiissoscf, None)
            .expect("HF failed");

        let ccsd = CCSD::new(&hf).compute().expect("CCSD failed");
        let ccsd_t = compute_ccsd_t_correction(&hf, &ccsd).expect("CCSD(T) failed");

        assert!(ccsd_t.ccsd_correlation_energy < 0.0);
        assert!(ccsd_t.total_energy <= ccsd_t.reference_energy);
    }
}
