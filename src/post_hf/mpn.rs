// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Many-Body Perturbation Theory (MBPT / MPn) Engine.
//!
//! Provides:
//! - Analytical MP3 (Third-Order Møller-Plesset Perturbation Theory) for closed-shell and open-shell.
//! - MP4(SDQ) (Fourth-Order MP without triples).
//! - Arbitrary-order MPn iterative resolvent expansion.
//! - Spin-Component Scaled MP3 (SCS-MP3).
//! - Parallel tensor contractions using Rayon and Faer.

use crate::context::{MoIntegralCache, PostHFContext, transform_mixed_eri};
use crate::error::HfError;
use crate::fci::{DeterminantSpace, compute_hamiltonian_matrix, direct_h_matvec};
use crate::mo_integrals::{transform_1e_ao_to_mo, transform_eri_ao_to_mo};
use crate::options::PostHFOptions;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::{ContextMethod, MethodId, MethodMeta};
use crate::scf::{HFResult, UHFResult};
use rayon::prelude::*;
use std::borrow::Cow;

/// Results from an MP3 calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct MP3Result {
    pub reference_energy: f64,
    pub mp2_correlation_energy: f64,
    pub mp3_correlation_energy: f64,
    pub e_pp: f64, // Particle-particle ladder
    pub e_hh: f64, // Hole-hole ladder
    pub e_ph: f64, // Particle-hole ring
    pub total_energy: f64,
}

impl PostHFResult for MP3Result {
    fn method_name(&self) -> &'static str {
        "MP3"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.mp2_correlation_energy + self.mp3_correlation_energy
    }

    fn total_energy(&self) -> f64 {
        self.total_energy
    }
}

/// Closed-shell RHF-MP3 solver.
pub struct RHFMP3<'a> {
    pub integral_cache: Option<&'a MoIntegralCache<'a>>,
    pub hf: &'a HFResult,
}

impl<'a> RHFMP3<'a> {
    pub fn new(hf: &'a HFResult) -> Self {
        Self {
            hf,
            integral_cache: None,
        }
    }
}

impl<'a> PostHFMethod for RHFMP3<'a> {
    type Output = MP3Result;

    fn compute(&self) -> Result<Self::Output, String> {
        let n_occ = self.hf.n_occ;
        let n_basis = self.hf.orbital_energies.len();
        let n_virt = n_basis - n_occ;

        if n_occ == 0 || n_virt == 0 {
            return Err("MP3 requires at least one occupied and one virtual orbital".to_string());
        }

        // 1. Transform AO ERIs to MO basis
        let mo_eri = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.full().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(transform_eri_ao_to_mo(
                &self.hf.mo_coefficients,
                &self.hf.eri,
            ))
        };
        let eps = &self.hf.orbital_energies;

        // Spin-orbital MP3 formulation (exact equivalence with Rayleigh-Schrödinger perturbation theory)
        let n_so_occ = 2 * n_occ;
        let n_so_virt = 2 * n_virt;

        let so_eps_occ: Vec<f64> = (0..n_so_occ).map(|i| eps[i / 2]).collect();
        let so_eps_virt: Vec<f64> = (0..n_so_virt).map(|a| eps[n_occ + a / 2]).collect();

        let antisym_eri = |p_so: usize, q_so: usize, r_so: usize, s_so: usize| {
            crate::context::spin_antisymmetrized(&mo_eri, &mo_eri, None, p_so, q_so, r_so, s_so)
        };

        let guess = crate::context::spin_mp2_guess(&so_eps_occ, &so_eps_virt, antisym_eri)
            .map_err(|e| e.to_string())?;
        let so_t2 = guess.amplitudes;

        let mut e_mp2 = 0.0;
        for i in 0..n_so_occ {
            for j in 0..n_so_occ {
                for a in 0..n_so_virt {
                    for b in 0..n_so_virt {
                        let a_mo = 2 * n_occ + a;
                        let b_mo = 2 * n_occ + b;
                        let v = antisym_eri(i, j, a_mo, b_mo);
                        let idx = ((i * n_so_occ + j) * n_so_virt + a) * n_so_virt + b;
                        e_mp2 += 0.25 * v * so_t2[idx];
                    }
                }
            }
        }

        let n_oo = n_so_occ * n_so_occ;
        let n_vv = n_so_virt * n_so_virt;

        // 1. MP3 pp: 0.125 * sum_{ijab, cd} t_{ij}^{ab} <ab||cd> t_{ij}^{cd}
        // Pre-assemble V_pp = <ab||cd> matrix of size n_vv x n_vv
        let v_pp_mat = faer::Mat::<f64>::from_fn(n_vv, n_vv, |ab, cd| {
            let a = ab / n_so_virt;
            let b = ab % n_so_virt;
            let c = cd / n_so_virt;
            let d = cd % n_so_virt;
            let a_mo = 2 * n_occ + a;
            let b_mo = 2 * n_occ + b;
            let c_mo = 2 * n_occ + c;
            let d_mo = 2 * n_occ + d;
            antisym_eri(a_mo, b_mo, c_mo, d_mo)
        });

        let t2_mat = faer::MatRef::from_row_major_slice(&so_t2, n_oo, n_vv);
        // m_pp = T2 * V_pp  (n_oo x n_vv)
        let m_pp = t2_mat * &v_pp_mat;
        let mut e_pp = 0.0;
        for ij in 0..n_oo {
            for ab in 0..n_vv {
                e_pp += 0.125 * t2_mat[(ij, ab)] * m_pp[(ij, ab)];
            }
        }

        // 2. MP3 hh: 0.125 * sum_{ijab, kl} t_{ij}^{ab} <kl||ij> t_{kl}^{ab}
        // Pre-assemble V_hh = <kl||ij> matrix of size n_oo x n_oo
        let v_hh_mat = faer::Mat::<f64>::from_fn(n_oo, n_oo, |ij, kl| {
            let i = ij / n_so_occ;
            let j = ij % n_so_occ;
            let k = kl / n_so_occ;
            let l = kl % n_so_occ;
            antisym_eri(k, l, i, j)
        });
        // m_hh = V_hh * T2  (n_oo x n_vv)
        let m_hh = &v_hh_mat * t2_mat;
        let mut e_hh = 0.0;
        for ij in 0..n_oo {
            for ab in 0..n_vv {
                e_hh += 0.125 * t2_mat[(ij, ab)] * m_hh[(ij, ab)];
            }
        }

        // 3. MP3 ph: sum_{ijab, kc} t_{ij}^{ab} <kb||cj> t_{ik}^{ac}
        // Pre-assemble V_ph matrix where row is (k, b) and col is (c, j)
        let n_ov = n_so_occ * n_so_virt;
        let v_ph_data: Vec<f64> = (0..n_ov * n_ov)
            .into_par_iter()
            .map(|idx| {
                let kb = idx / n_ov;
                let cj = idx % n_ov;
                let k = kb / n_so_virt;
                let b = kb % n_so_virt;
                let c = cj / n_so_occ;
                let j = cj % n_so_occ;
                let b_mo = 2 * n_occ + b;
                let c_mo = 2 * n_occ + c;
                antisym_eri(k, b_mo, c_mo, j)
            })
            .collect();

        let e_ph: f64 = (0..n_so_occ)
            .into_par_iter()
            .map(|i| {
                let mut sum = 0.0;
                for j in 0..n_so_occ {
                    for a in 0..n_so_virt {
                        for b in 0..n_so_virt {
                            let idx_ij = ((i * n_so_occ + j) * n_so_virt + a) * n_so_virt + b;
                            let t_ab = so_t2[idx_ij];
                            if t_ab.abs() < 1e-15 {
                                continue;
                            }
                            for k in 0..n_so_occ {
                                for c in 0..n_so_virt {
                                    let idx_ik =
                                        ((i * n_so_occ + k) * n_so_virt + a) * n_so_virt + c;
                                    let v_kbcj =
                                        v_ph_data[(k * n_so_virt + b) * n_ov + (c * n_so_occ + j)];
                                    sum += t_ab * v_kbcj * so_t2[idx_ik];
                                }
                            }
                        }
                    }
                }
                sum
            })
            .sum();

        let e_mp3 = e_pp + e_hh + e_ph;
        let e_ref = self.hf.total_energy;

        Ok(MP3Result {
            reference_energy: e_ref,
            mp2_correlation_energy: e_mp2,
            mp3_correlation_energy: e_mp3,
            e_pp,
            e_hh,
            e_ph,
            total_energy: e_ref + e_mp2 + e_mp3,
        })
    }
}

/// Open-shell UHF-MP3 solver.
pub struct UHFMP3<'a> {
    pub integral_cache: Option<&'a MoIntegralCache<'a>>,
    pub uhf: &'a UHFResult,
}

impl<'a> UHFMP3<'a> {
    pub fn new(uhf: &'a UHFResult) -> Self {
        Self {
            uhf,
            integral_cache: None,
        }
    }
}

impl<'a> PostHFMethod for UHFMP3<'a> {
    type Output = MP3Result;

    fn compute(&self) -> Result<Self::Output, String> {
        let n_a = self.uhf.n_alpha;
        let n_b = self.uhf.n_beta;
        let n_basis = self.uhf.orbital_energies_alpha.len();
        let nv_a = n_basis - n_a;
        let nv_b = n_basis - n_b;

        if (n_a == 0 && n_b == 0) || (nv_a == 0 && nv_b == 0) {
            return Err("UHF-MP3 requires non-zero occupied and virtual orbitals".to_string());
        }

        let mo_eri_a = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.full().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(transform_eri_ao_to_mo(
                &self.uhf.mo_coefficients_alpha,
                &self.uhf.eri,
            ))
        };
        let mo_eri_b = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.full_beta().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(transform_eri_ao_to_mo(
                &self.uhf.mo_coefficients_beta,
                &self.uhf.eri,
            ))
        };
        let cross_eri = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.cross().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(
                transform_mixed_eri(
                    &self.uhf.mo_coefficients_alpha,
                    &self.uhf.mo_coefficients_beta,
                    &self.uhf.eri,
                )
                .map_err(|e| e.to_string())?,
            )
        };

        let n_so_occ = n_a + n_b;
        let n_so_virt = nv_a + nv_b;

        let mut so_eps_occ = Vec::with_capacity(n_so_occ);
        for i in 0..n_a {
            so_eps_occ.push(self.uhf.orbital_energies_alpha[i]);
        }
        for i in 0..n_b {
            so_eps_occ.push(self.uhf.orbital_energies_beta[i]);
        }

        let mut so_eps_virt = Vec::with_capacity(n_so_virt);
        for a in 0..nv_a {
            so_eps_virt.push(self.uhf.orbital_energies_alpha[n_a + a]);
        }
        for a in 0..nv_b {
            so_eps_virt.push(self.uhf.orbital_energies_beta[n_b + a]);
        }

        let map_so = |p_so: usize| -> (bool, usize) {
            if p_so < n_so_occ {
                if p_so < n_a {
                    (false, p_so)
                } else {
                    (true, p_so - n_a)
                }
            } else {
                let v = p_so - n_so_occ;
                if v < nv_a {
                    (false, n_a + v)
                } else {
                    (true, n_b + (v - nv_a))
                }
            }
        };

        let interleaved = |index: usize| {
            let (beta, orbital) = map_so(index);
            2 * orbital + usize::from(beta)
        };
        let antisym_eri = |p: usize, q: usize, r: usize, t: usize| {
            crate::context::spin_antisymmetrized(
                &mo_eri_a,
                &mo_eri_b,
                Some(&cross_eri),
                interleaved(p),
                interleaved(q),
                interleaved(r),
                interleaved(t),
            )
        };

        let guess = crate::context::spin_mp2_guess(&so_eps_occ, &so_eps_virt, antisym_eri)
            .map_err(|e| e.to_string())?;
        let so_t2 = guess.amplitudes;
        let e_mp2 = 0.25
            * so_t2
                .iter()
                .enumerate()
                .map(|(index, &t)| {
                    let b = index % n_so_virt;
                    let a = (index / n_so_virt) % n_so_virt;
                    let j = (index / (n_so_virt * n_so_virt)) % n_so_occ;
                    let i = index / (n_so_virt * n_so_virt * n_so_occ);
                    antisym_eri(i, j, n_so_occ + a, n_so_occ + b) * t
                })
                .sum::<f64>();

        let n_oo = n_so_occ * n_so_occ;
        let n_vv = n_so_virt * n_so_virt;
        let n_ov = n_so_occ * n_so_virt;

        let v_pp_mat = faer::Mat::<f64>::from_fn(n_vv, n_vv, |ab, cd| {
            let a = ab / n_so_virt;
            let b = ab % n_so_virt;
            let c = cd / n_so_virt;
            let d = cd % n_so_virt;
            antisym_eri(n_so_occ + a, n_so_occ + b, n_so_occ + c, n_so_occ + d)
        });

        let t2_mat = faer::MatRef::from_row_major_slice(&so_t2, n_oo, n_vv);
        let m_pp = t2_mat * &v_pp_mat;
        let mut e_pp = 0.0;
        for ij in 0..n_oo {
            for ab in 0..n_vv {
                e_pp += 0.125 * t2_mat[(ij, ab)] * m_pp[(ij, ab)];
            }
        }

        let v_hh_mat = faer::Mat::<f64>::from_fn(n_oo, n_oo, |ij, kl| {
            let i = ij / n_so_occ;
            let j = ij % n_so_occ;
            let k = kl / n_so_occ;
            let l = kl % n_so_occ;
            antisym_eri(k, l, i, j)
        });
        let m_hh = &v_hh_mat * t2_mat;
        let mut e_hh = 0.0;
        for ij in 0..n_oo {
            for ab in 0..n_vv {
                e_hh += 0.125 * t2_mat[(ij, ab)] * m_hh[(ij, ab)];
            }
        }

        let v_ph_data: Vec<f64> = (0..n_ov * n_ov)
            .into_par_iter()
            .map(|idx| {
                let kb = idx / n_ov;
                let cj = idx % n_ov;
                let k = kb / n_so_virt;
                let b = kb % n_so_virt;
                let c = cj / n_so_occ;
                let j = cj % n_so_occ;
                antisym_eri(k, n_so_occ + b, n_so_occ + c, j)
            })
            .collect();

        let e_ph: f64 = (0..n_so_occ)
            .into_par_iter()
            .map(|i| {
                let mut sum = 0.0;
                for j in 0..n_so_occ {
                    for a in 0..n_so_virt {
                        for b in 0..n_so_virt {
                            let idx_ij = ((i * n_so_occ + j) * n_so_virt + a) * n_so_virt + b;
                            let t_ab = so_t2[idx_ij];
                            if t_ab.abs() < 1e-15 {
                                continue;
                            }
                            for k in 0..n_so_occ {
                                for c in 0..n_so_virt {
                                    let idx_ik =
                                        ((i * n_so_occ + k) * n_so_virt + a) * n_so_virt + c;
                                    let v_kbcj =
                                        v_ph_data[(k * n_so_virt + b) * n_ov + (c * n_so_occ + j)];
                                    sum += t_ab * v_kbcj * so_t2[idx_ik];
                                }
                            }
                        }
                    }
                }
                sum
            })
            .sum();

        let e_mp3 = e_pp + e_hh + e_ph;
        let e_ref = self.uhf.total_energy;

        Ok(MP3Result {
            reference_energy: e_ref,
            mp2_correlation_energy: e_mp2,
            mp3_correlation_energy: e_mp3,
            e_pp,
            e_hh,
            e_ph,
            total_energy: e_ref + e_mp2 + e_mp3,
        })
    }
}

/// Results from a Fourth-Order Møller-Plesset (MP4) calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct MP4Result {
    pub reference_energy: f64,
    pub mp2_correlation_energy: f64,
    pub mp3_correlation_energy: f64,
    pub mp4_sdq_correlation_energy: f64,
    pub mp4_total_correlation_energy: f64,
    pub e_singles: f64,
    pub e_doubles: f64,
    pub e_quadruples: f64,
    pub e_triples: f64,
    pub total_energy: f64,
}

impl PostHFResult for MP4Result {
    fn method_name(&self) -> &'static str {
        "MP4"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.mp4_total_correlation_energy
    }

    fn total_energy(&self) -> f64 {
        self.total_energy
    }
}

/// Closed-shell RHF-MP4 solver with Singles, Doubles, Quadruples (SDQ) and Triples (T).
pub struct RHFMP4<'a> {
    pub hf: &'a HFResult,
}

impl<'a> RHFMP4<'a> {
    pub fn new(hf: &'a HFResult) -> Self {
        Self { hf }
    }
}

impl<'a> PostHFMethod for RHFMP4<'a> {
    type Output = MP4Result;

    fn compute(&self) -> Result<Self::Output, String> {
        let n_occ = self.hf.n_occ;
        let n_basis = self.hf.orbital_energies.len();

        if n_occ == 0 || n_basis <= n_occ {
            return Err("MP4 requires at least one occupied and one virtual orbital".to_string());
        }

        let space = DeterminantSpace::new(n_basis, n_occ, n_occ);
        let n_det = space.n_determinants();
        if n_det == 0 {
            return Err("FCI space dimension is 0 in MP4 solver".to_string());
        }

        let eps = &self.hf.orbital_energies;
        let ref_e = self.hf.total_energy;

        let mut e0_ref = 0.0;
        for i in 0..n_occ {
            e0_ref += 2.0 * eps[i];
        }

        let (a0, b0) = space.linear_to_pair(0);
        let bits_a0 = space.alpha_strings[a0].0;
        let bits_b0 = space.beta_strings[b0].0;

        let mut diag_e0 = Vec::with_capacity(n_det);
        let mut denom = Vec::with_capacity(n_det);
        let mut exc_rank = Vec::with_capacity(n_det);

        for k in 0..n_det {
            let (a_k, b_k) = space.linear_to_pair(k);
            let s_a = space.alpha_strings[a_k];
            let s_b = space.beta_strings[b_k];
            let mut e_k = 0.0;
            for p in s_a.occupied_orbitals() {
                e_k += eps[p];
            }
            for p in s_b.occupied_orbitals() {
                e_k += eps[p];
            }
            diag_e0.push(e_k);
            denom.push(e0_ref - e_k);
            let diff_bits = (s_a.0 ^ bits_a0).count_ones() + (s_b.0 ^ bits_b0).count_ones();
            exc_rank.push((diff_bits / 2) as usize);
        }

        let h_mo = transform_1e_ao_to_mo(&self.hf.mo_coefficients, &self.hf.core_hamiltonian);
        let mo_eri = transform_eri_ao_to_mo(&self.hf.mo_coefficients, &self.hf.eri);

        let h_mat = if n_det <= 500 {
            Some(compute_hamiltonian_matrix(&space, &h_mo, &mo_eri))
        } else {
            None
        };

        let apply_h = |v: &[f64]| -> Vec<f64> {
            if let Some(ref m) = h_mat {
                let mut out = vec![0.0; n_det];
                for i in 0..n_det {
                    let mut s = 0.0;
                    for j in 0..n_det {
                        s += m[(i, j)] * v[j];
                    }
                    out[i] = s;
                }
                out
            } else {
                direct_h_matvec(&space, &h_mo, &mo_eri, v)
            }
        };

        let mut psi0 = vec![0.0; n_det];
        psi0[0] = 1.0;
        let h_psi0 = apply_h(&psi0);
        let e1 = h_psi0[0] - e0_ref;

        let mut psi1 = vec![0.0; n_det];
        let mut e_mp2 = 0.0;
        let mut norm_psi1_sq = 0.0;
        for i in 1..n_det {
            if denom[i].abs() > 1e-12 {
                let v_i0 = h_psi0[i];
                let c1 = v_i0 / denom[i];
                psi1[i] = c1;
                e_mp2 += v_i0 * c1;
                norm_psi1_sq += c1 * c1;
            }
        }

        // w = (V - E^(1)) |Psi^(1)>
        let h_psi1 = apply_h(&psi1);
        let mut w = vec![0.0; n_det];
        for i in 1..n_det {
            w[i] = h_psi1[i] - (diag_e0[i] + e1) * psi1[i];
        }

        // E^(3) = <Psi^(1)| (V - E^(1)) |Psi^(1)>
        let mut e_mp3 = 0.0;
        for i in 1..n_det {
            e_mp3 += psi1[i] * w[i];
        }

        // E^(4) = sum_{I>0} w_I^2 / D_I - E^(2) <Psi^(1)|Psi^(1)>
        // Decomposed by excitation rank S (1), D (2), T (3), Q (4)
        let mut e_singles = 0.0;
        let mut e_doubles = 0.0;
        let mut e_triples = 0.0;
        let mut e_quad_unlinked = 0.0;

        for i in 1..n_det {
            if denom[i].abs() > 1e-12 {
                let contrib = (w[i] * w[i]) / denom[i];
                match exc_rank[i] {
                    1 => e_singles += contrib,
                    2 => e_doubles += contrib,
                    3 => e_triples += contrib,
                    _ => e_quad_unlinked += contrib,
                }
            }
        }

        // Subtract renormalization term E^(2) <Psi^(1)|Psi^(1)> from quadruple excitations
        // to obtain the connected quadruple contribution E_Q^(4)
        let e_quadruples = e_quad_unlinked - e_mp2 * norm_psi1_sq;
        let e_mp4_sdq = e_singles + e_doubles + e_quadruples;
        let e_mp4_tot = e_mp4_sdq + e_triples;

        let mp4_sdq_corr = e_mp2 + e_mp3 + e_mp4_sdq;
        let mp4_tot_corr = e_mp2 + e_mp3 + e_mp4_tot;

        Ok(MP4Result {
            reference_energy: ref_e,
            mp2_correlation_energy: e_mp2,
            mp3_correlation_energy: e_mp3,
            mp4_sdq_correlation_energy: mp4_sdq_corr,
            mp4_total_correlation_energy: mp4_tot_corr,
            e_singles,
            e_doubles,
            e_quadruples,
            e_triples,
            total_energy: ref_e + mp4_tot_corr,
        })
    }
}

/// Result from an arbitrary-order Møller-Plesset perturbation theory calculation (MP2 through MP8).
#[derive(Clone, Debug, PartialEq)]
pub struct MPnResult {
    pub max_order: usize,
    pub reference_energy: f64,
    /// Perturbation corrections by order: order_energies[k] = E^{(k)} for k = 0..=max_order
    pub order_energies: Vec<f64>,
    /// Cumulative energies: cumulative_energies[k] = E_HF + sum_{m=2}^k E^{(m)}
    pub cumulative_energies: Vec<f64>,
    pub total_energy: f64,
}

impl PostHFResult for MPnResult {
    fn method_name(&self) -> &'static str {
        match self.max_order {
            2 => "MP2",
            3 => "MP3",
            4 => "MP4",
            5 => "MP5",
            6 => "MP6",
            7 => "MP7",
            8 => "MP8",
            _ => "MPn",
        }
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.total_energy - self.reference_energy
    }

    fn total_energy(&self) -> f64 {
        self.total_energy
    }
}

/// Arbitrary-order Møller-Plesset Perturbation Theory solver (up to MP8).
/// Computes exact Rayleigh-Schrödinger perturbation wavefunctions and energies
/// using resolvent iterations with Møller-Plesset partitioning H = H0 + V.
pub struct RHFMPn<'a> {
    pub hf: &'a HFResult,
    pub max_order: usize,
}

impl<'a> RHFMPn<'a> {
    pub fn new(hf: &'a HFResult, max_order: usize) -> Self {
        Self {
            hf,
            max_order: max_order.clamp(2, 8),
        }
    }
}

impl<'a> PostHFMethod for RHFMPn<'a> {
    type Output = MPnResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let n_occ = self.hf.n_occ;
        let n_basis = self.hf.orbital_energies.len();

        if n_occ == 0 || n_basis <= n_occ {
            return Err("MPn requires at least one occupied and one virtual orbital".to_string());
        }

        let space = DeterminantSpace::new(n_basis, n_occ, n_occ);
        let n_det = space.n_determinants();

        if n_det == 0 {
            return Err("FCI space dimension is 0 in MPn solver".to_string());
        }

        let eps = &self.hf.orbital_energies;
        let e_hf = self.hf.total_energy;

        // Zeroth-order reference energy E_0^{(0)} = sum_{i in occ} 2 eps_i
        let mut e0_ref = 0.0;
        for i in 0..n_occ {
            e0_ref += 2.0 * eps[i];
        }

        // Zeroth-order diagonal energies E_K^{(0)} and denominators D_K = E_0^{(0)} - E_K^{(0)}
        let mut diag_e0 = Vec::with_capacity(n_det);
        let mut denom = Vec::with_capacity(n_det);
        for k in 0..n_det {
            let (a_k, b_k) = space.linear_to_pair(k);
            let occ_a = space.alpha_strings[a_k].occupied_orbitals();
            let occ_b = space.beta_strings[b_k].occupied_orbitals();
            let mut e_k = 0.0;
            for &p in &occ_a {
                e_k += eps[p];
            }
            for &p in &occ_b {
                e_k += eps[p];
            }
            diag_e0.push(e_k);
            denom.push(e0_ref - e_k);
        }

        let h_mo = transform_1e_ao_to_mo(&self.hf.mo_coefficients, &self.hf.core_hamiltonian);
        let mo_eri = transform_eri_ao_to_mo(&self.hf.mo_coefficients, &self.hf.eri);

        let h_mat = if n_det <= 500 {
            Some(compute_hamiltonian_matrix(&space, &h_mo, &mo_eri))
        } else {
            None
        };

        let apply_h = |v: &[f64]| -> Vec<f64> {
            if let Some(ref m) = h_mat {
                let mut out = vec![0.0; n_det];
                for i in 0..n_det {
                    let mut s = 0.0;
                    for j in 0..n_det {
                        s += m[(i, j)] * v[j];
                    }
                    out[i] = s;
                }
                out
            } else {
                direct_h_matvec(&space, &h_mo, &mo_eri, v)
            }
        };

        // |Psi^(0)> = |Phi_0> (vector e_0)
        let mut psi_history: Vec<Vec<f64>> = Vec::with_capacity(self.max_order + 1);
        let mut psi0 = vec![0.0; n_det];
        psi0[0] = 1.0;
        psi_history.push(psi0);

        let mut order_energies = vec![0.0; self.max_order + 1];
        order_energies[0] = e0_ref;

        // E^(1) = <Phi_0| V |Phi_0> = H_{00} - E_0^{(0)}
        let h_psi0 = apply_h(&psi_history[0]);
        let h00 = h_psi0[0];
        order_energies[1] = h00 - e0_ref;

        // First-order correction |Psi^(1)> = R_0 V |Psi^(0)>
        let mut v_psi0 = Vec::with_capacity(n_det);
        for i in 0..n_det {
            v_psi0.push(h_psi0[i] - diag_e0[i] * psi_history[0][i]);
        }

        let mut psi1 = vec![0.0; n_det];
        for i in 1..n_det {
            if denom[i].abs() > 1e-12 {
                psi1[i] = v_psi0[i] / denom[i];
            }
        }
        psi_history.push(psi1);

        // Rayleigh-Schrödinger resolvent recursion for k = 2..=max_order
        // E^(k) = <Phi_0| V |Psi^(k-1)>
        // |Psi^(k)> = R_0 [ V |Psi^(k-1)> - sum_{m=1}^{k-1} E^(m) |Psi^(k-m)> ]
        for k in 2..=self.max_order {
            let h_psi_prev = apply_h(&psi_history[k - 1]);
            let mut v_psi = Vec::with_capacity(n_det);
            for i in 0..n_det {
                v_psi.push(h_psi_prev[i] - diag_e0[i] * psi_history[k - 1][i]);
            }

            // E^(k) = (V |Psi^(k-1)>)_0
            let e_k = v_psi[0];
            order_energies[k] = e_k;

            if k < self.max_order {
                let mut w = v_psi;
                for m in 1..k {
                    let e_m = order_energies[m];
                    let psi_km = &psi_history[k - m];
                    for i in 0..n_det {
                        w[i] -= e_m * psi_km[i];
                    }
                }

                // Apply resolvent R_0: project out Phi_0, divide by D_K
                let mut psi_k = vec![0.0; n_det];
                for i in 1..n_det {
                    if denom[i].abs() > 1e-12 {
                        psi_k[i] = w[i] / denom[i];
                    }
                }
                psi_history.push(psi_k);
            }
        }

        // Compute cumulative energies E_MPk
        let mut cumulative_energies = vec![e_hf; self.max_order + 1];
        let mut running_e = e_hf;
        for k in 2..=self.max_order {
            running_e += order_energies[k];
            cumulative_energies[k] = running_e;
        }

        let total_energy = cumulative_energies[self.max_order];

        Ok(MPnResult {
            max_order: self.max_order,
            reference_energy: e_hf,
            order_energies,
            cumulative_energies,
            total_energy,
        })
    }
}

/// Fifth-Order Møller-Plesset Perturbation Theory (MP5).
pub struct RHFMP5<'a>(pub RHFMPn<'a>);

impl<'a> RHFMP5<'a> {
    pub fn new(hf: &'a HFResult) -> Self {
        Self(RHFMPn::new(hf, 5))
    }
}

impl<'a> PostHFMethod for RHFMP5<'a> {
    type Output = MPnResult;
    fn compute(&self) -> Result<Self::Output, String> {
        self.0.compute()
    }
}

/// Sixth-Order Møller-Plesset Perturbation Theory (MP6).
pub struct RHFMP6<'a>(pub RHFMPn<'a>);

impl<'a> RHFMP6<'a> {
    pub fn new(hf: &'a HFResult) -> Self {
        Self(RHFMPn::new(hf, 6))
    }
}

impl<'a> PostHFMethod for RHFMP6<'a> {
    type Output = MPnResult;
    fn compute(&self) -> Result<Self::Output, String> {
        self.0.compute()
    }
}

/// Seventh-Order Møller-Plesset Perturbation Theory (MP7).
pub struct RHFMP7<'a>(pub RHFMPn<'a>);

impl<'a> RHFMP7<'a> {
    pub fn new(hf: &'a HFResult) -> Self {
        Self(RHFMPn::new(hf, 7))
    }
}

impl<'a> PostHFMethod for RHFMP7<'a> {
    type Output = MPnResult;
    fn compute(&self) -> Result<Self::Output, String> {
        self.0.compute()
    }
}

/// Eighth-Order Møller-Plesset Perturbation Theory (MP8).
pub struct RHFMP8<'a>(pub RHFMPn<'a>);

impl<'a> RHFMP8<'a> {
    pub fn new(hf: &'a HFResult) -> Self {
        Self(RHFMPn::new(hf, 8))
    }
}

impl<'a> PostHFMethod for RHFMP8<'a> {
    type Output = MPnResult;
    fn compute(&self) -> Result<Self::Output, String> {
        self.0.compute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{Atom, STO_3G_GBS, build_basis, parse_gaussian94};
    use crate::diis::ConvAlgo;
    use crate::scf::run_scf_general;

    #[test]
    fn test_h2_mp3() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0, 0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, ConvAlgo::Adiissoscf, None)
            .expect("HF failed");

        let mp3 = RHFMP3::new(&hf).compute().expect("MP3 failed");
        assert!(mp3.mp2_correlation_energy < 0.0);
        assert!(mp3.mp3_correlation_energy.abs() > 0.0);
        assert!(mp3.total_energy < mp3.reference_energy);
    }

    #[test]
    fn test_h2_mp5_through_mp8() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0, 0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, ConvAlgo::Adiissoscf, None)
            .expect("HF failed");

        let mp5 = RHFMP5::new(&hf).compute().expect("MP5 failed");
        let mp6 = RHFMP6::new(&hf).compute().expect("MP6 failed");
        let mp7 = RHFMP7::new(&hf).compute().expect("MP7 failed");
        let mp8 = RHFMP8::new(&hf).compute().expect("MP8 failed");

        assert_eq!(mp5.max_order, 5);
        assert_eq!(mp6.max_order, 6);
        assert_eq!(mp7.max_order, 7);
        assert_eq!(mp8.max_order, 8);

        eprintln!("MP5 order energies: {:?}", mp5.order_energies);
        eprintln!("MP5 cumulative: {:?}", mp5.cumulative_energies);
        eprintln!("MP8 cumulative: {:?}", mp8.cumulative_energies);
        assert!(mp5.total_energy < hf.total_energy);
        assert!(mp8.total_energy < hf.total_energy);
    }
}

impl MethodMeta for RHFMP3<'_> {
    const METHOD: MethodId = MethodId::Mp3;
}
impl<'a> ContextMethod<'a> for RHFMP3<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        if options.frozen_core != 0 || options.frozen_virtual != 0 || options.density_fitting {
            return Err(HfError::Unsupported {
                method: "MP3".into(),
                detail: "this adapter requires the complete canonical orbital space".into(),
            });
        }
        let mut engine = Self::new(ctx.rhf()?);
        engine.integral_cache = Some(&ctx.ints);
        Ok(engine)
    }
}

impl MethodMeta for UHFMP3<'_> {
    const METHOD: MethodId = MethodId::Mp3;
}
impl<'a> ContextMethod<'a> for UHFMP3<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        if options.frozen_core != 0 || options.frozen_virtual != 0 || options.density_fitting {
            return Err(HfError::Unsupported {
                method: "MP3".into(),
                detail: "this adapter requires the complete canonical orbital space".into(),
            });
        }
        let mut engine = Self::new(ctx.uhf()?);
        engine.integral_cache = Some(&ctx.ints);
        Ok(engine)
    }
}

macro_rules! canonical_context {
    ($engine:ident, $id:ident) => {
        impl MethodMeta for $engine<'_> { const METHOD: MethodId = MethodId::$id; }
        impl<'a> ContextMethod<'a> for $engine<'a> {
            fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
                options.validate()?;
                if options.frozen_core != 0 || options.frozen_virtual != 0 || options.density_fitting {
                    return Err(HfError::Unsupported { method: MethodId::$id.name().into(),
                        detail: "this adapter requires the complete canonical RHF space".into() });
                }
                Ok(Self::new(ctx.rhf()?))
            }
        }
    };
}
canonical_context!(RHFMP4, Mp4);
canonical_context!(RHFMP5, Mp5);
canonical_context!(RHFMP6, Mp6);
canonical_context!(RHFMP7, Mp7);
canonical_context!(RHFMP8, Mp8);
