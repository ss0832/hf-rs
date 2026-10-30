// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Second-order Møller-Plesset perturbation theory (MP2).
//!
//! Provides:
//! - Canonical closed-shell RHF-MP2 with full or frozen-core orbital spaces.
//! - Spin-component scaled MP2 (SCS-MP2) separating same-spin and opposite-spin correlation.
//! - Open-shell UHF-MP2 (alpha-alpha, beta-beta, and alpha-beta pair energies).
//! - MP2 1-particle reduced density matrix (1-RDM) and natural orbital occupation numbers.

use crate::context::{MoIntegralCache, PostHFContext, transform_mixed_eri};
use crate::error::HfError;
use crate::matrix::Matrix;
use crate::mo_integrals::{transform_eri_ao_to_mo, transform_oovv};
use crate::options::PostHFOptions;
use crate::orbitals::{OrbitalSpace, SpinOrbitalSpace};
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::{ContextMethod, MethodId, MethodMeta};
use crate::scf::{HFResult, UHFResult};
use rayon::prelude::*;
use std::borrow::Cow;

/// Results of a closed-shell RHF-MP2 calculation.
#[derive(Clone, Debug)]
pub struct MP2Result {
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub e_same_spin: f64,
    pub e_opposite_spin: f64,
    pub e_scs: f64,
    pub orbital_space: OrbitalSpace,
    pub one_rdm_mo: Option<Matrix>,
}

impl PostHFResult for MP2Result {
    fn method_name(&self) -> &'static str {
        "RHF-MP2"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }

    fn one_rdm(&self) -> Option<&Matrix> {
        self.one_rdm_mo.as_ref()
    }
}

/// Closed-shell RHF-MP2 calculation engine.
#[derive(Clone, Debug)]
pub struct RHFMP2<'a> {
    pub integral_cache: Option<&'a MoIntegralCache<'a>>,
    pub hf_res: &'a HFResult,
    pub space: OrbitalSpace,
    pub c_same_spin: f64,
    pub c_opposite_spin: f64,
    pub compute_rdm: bool,
}

impl<'a> RHFMP2<'a> {
    /// Creates a new RHF-MP2 engine using all-electron correlation.
    pub fn new(hf_res: &'a HFResult) -> Self {
        let space = hf_res.orbital_space();
        Self {
            integral_cache: None,
            hf_res,
            space,
            // Standard Grimme SCS-MP2 scaling factors: c_ss = 1/3, c_os = 6/5 = 1.2
            c_same_spin: 1.0 / 3.0,
            c_opposite_spin: 1.2,
            compute_rdm: false,
        }
    }

    /// Sets a custom orbital space (e.g. frozen core or active window).
    pub fn with_orbital_space(mut self, space: OrbitalSpace) -> Self {
        self.space = space;
        self
    }

    /// Freezes the lowest `n_core` spatial orbitals.
    pub fn with_frozen_core(mut self, n_core: usize) -> Self {
        self.space = self.space.with_frozen_core(n_core);
        self
    }

    /// Configures custom Spin-Component-Scaled (SCS) factors.
    pub fn with_scs_factors(mut self, c_ss: f64, c_os: f64) -> Self {
        self.c_same_spin = c_ss;
        self.c_opposite_spin = c_os;
        self
    }

    /// Requests calculation of the MP2 unrelaxed 1-particle reduced density matrix.
    pub fn with_rdm(mut self, compute: bool) -> Self {
        self.compute_rdm = compute;
        self
    }
}

impl PostHFMethod for RHFMP2<'_> {
    type Output = MP2Result;

    fn compute(&self) -> Result<MP2Result, String> {
        let occ_range = self.space.correlated_occ_range();
        let virt_range = self.space.correlated_virt_range();

        if occ_range.is_empty() {
            return Err("Cannot compute MP2: no correlated occupied orbitals".to_string());
        }
        if virt_range.is_empty() {
            return Err("Cannot compute MP2: no correlated virtual orbitals".to_string());
        }

        let c = &self.hf_res.mo_coefficients;
        let eri_ao = &self.hf_res.eri;
        let eps = &self.hf_res.orbital_energies;

        let n_occ = occ_range.len();
        let n_virt = virt_range.len();
        let n_basis = self.space.n_basis;
        let n_basis_ao = c.rows;

        let mut c_occ = Matrix::zeros(n_basis_ao, n_occ);
        let mut c_virt = Matrix::zeros(n_basis_ao, n_virt);
        for (i_idx, i_mo) in occ_range.clone().enumerate() {
            for mu in 0..n_basis_ao {
                c_occ[(mu, i_idx)] = c[(mu, i_mo)];
            }
        }
        for (a_idx, a_mo) in virt_range.clone().enumerate() {
            for mu in 0..n_basis_ao {
                c_virt[(mu, a_idx)] = c[(mu, a_mo)];
            }
        }

        // Transform integrals block-selectively directly to OOVV space in O(N_occ^2 N^3)
        let oovv = if let Some(cache) = self.integral_cache {
            let all = cache.oovv().map_err(|e| e.to_string())?;
            if n_occ == all.n_occ && n_virt == all.n_virt {
                Cow::Borrowed(all.as_ref())
            } else {
                let mut subset = crate::mo_integrals::OovvIntegrals {
                    n_occ,
                    n_virt,
                    data: vec![0.0; n_occ * n_virt * n_occ * n_virt],
                };
                for (i, ii) in occ_range.clone().enumerate() {
                    for (a, aa) in virt_range.clone().enumerate() {
                        for (j, jj) in occ_range.clone().enumerate() {
                            for (b, bb) in virt_range.clone().enumerate() {
                                subset.data[(i * n_virt + a) * (n_occ * n_virt) + j * n_virt + b] =
                                    all.get(ii, aa - self.hf_res.n_occ, jj, bb - self.hf_res.n_occ);
                            }
                        }
                    }
                }
                Cow::Owned(subset)
            }
        } else {
            Cow::Owned(transform_oovv(&c_occ, &c_virt, eri_ao))
        };

        let mut e_corr = 0.0;
        let mut e_ss = 0.0;
        let mut e_os = 0.0;

        // Amplitude tensor t_{ij}^{ab} = (ia|jb) / (e_i + e_j - e_a - e_b) if RDM requested
        let mut t_amp = if self.compute_rdm {
            vec![0.0; n_occ * n_occ * n_virt * n_virt]
        } else {
            Vec::new()
        };

        for (i_idx, i) in occ_range.clone().enumerate() {
            for (j_idx, j) in occ_range.clone().enumerate() {
                let e_ij = eps[i] + eps[j];

                for (a_idx, a) in virt_range.clone().enumerate() {
                    for (b_idx, b) in virt_range.clone().enumerate() {
                        let denom = e_ij - eps[a] - eps[b];
                        if denom.abs() < 1e-14 {
                            return Err(format!(
                                "Near-zero orbital energy denominator in MP2: eps[{}] + eps[{}] - eps[{}] - eps[{}] = {}",
                                i, j, a, b, denom
                            ));
                        }

                        let g_iajb = oovv.get(i_idx, a_idx, j_idx, b_idx);
                        let g_ibja = oovv.get(i_idx, b_idx, j_idx, a_idx);

                        // Dirac anti-symmetrized: <ij||ab> = (ia|jb) - (ib|ja)
                        let asym = g_iajb - g_ibja;

                        // Same-spin (triplet pair) contribution: 1/2 * (g_iajb - g_ibja)^2 / denom
                        // Summed over all i,j,a,b gives:
                        let term_ss = 0.5 * asym * asym / denom;
                        // Opposite-spin (singlet pair) contribution: g_iajb^2 / denom
                        let term_os = g_iajb * g_iajb / denom;

                        // Total closed-shell MP2 pair energy: (ia|jb) [2(ia|jb) - (ib|ja)] / denom
                        let term_total = g_iajb * (2.0 * g_iajb - g_ibja) / denom;

                        e_ss += term_ss;
                        e_os += term_os;
                        e_corr += term_total;

                        if self.compute_rdm {
                            let idx = ((i_idx * n_occ + j_idx) * n_virt + a_idx) * n_virt + b_idx;
                            t_amp[idx] = g_iajb / denom;
                        }
                    }
                }
            }
        }

        let e_scs = self.c_same_spin * e_ss + self.c_opposite_spin * e_os;

        // Construct 1-RDM if requested
        let one_rdm_mo = if self.compute_rdm {
            let mut rdm = Matrix::zeros(n_basis, n_basis);
            // Frozen occupied orbitals retain their reference occupation.
            for i in 0..self.hf_res.n_occ {
                rdm[(i, i)] = 2.0;
            }

            // Occupied-occupied block: gamma_{ij} = 2 delta_{ij} - sum_{k, a, b} (2 t_{ik}^{ab} - t_{ik}^{ba}) t_{jk}^{ab}
            for (i_idx, i) in occ_range.clone().enumerate() {
                rdm[(i, i)] = 2.0;
                for (j_idx, j) in occ_range.clone().enumerate() {
                    let mut delta_ij = 0.0;
                    for k_idx in 0..n_occ {
                        for a_idx in 0..n_virt {
                            for b_idx in 0..n_virt {
                                let idx_ikab =
                                    ((i_idx * n_occ + k_idx) * n_virt + a_idx) * n_virt + b_idx;
                                let idx_ikba =
                                    ((i_idx * n_occ + k_idx) * n_virt + b_idx) * n_virt + a_idx;
                                let idx_jkab =
                                    ((j_idx * n_occ + k_idx) * n_virt + a_idx) * n_virt + b_idx;

                                let t_ikab = t_amp[idx_ikab];
                                let t_ikba = t_amp[idx_ikba];
                                let t_jkab = t_amp[idx_jkab];

                                delta_ij += 2.0 * (2.0 * t_ikab - t_ikba) * t_jkab;
                            }
                        }
                    }
                    rdm[(i, j)] -= delta_ij;
                }
            }

            // Virtual-virtual block: gamma_{ab} = 2 * sum_{i, j, c} (2 t_{ij}^{ac} - t_{ij}^{ca}) t_{ij}^{bc}
            for (a_idx, a) in virt_range.clone().enumerate() {
                for (b_idx, b) in virt_range.clone().enumerate() {
                    let mut sum_ab = 0.0;
                    for i_idx in 0..n_occ {
                        for j_idx in 0..n_occ {
                            for c_idx in 0..n_virt {
                                let idx_ijac =
                                    ((i_idx * n_occ + j_idx) * n_virt + a_idx) * n_virt + c_idx;
                                let idx_ijca =
                                    ((i_idx * n_occ + j_idx) * n_virt + c_idx) * n_virt + a_idx;
                                let idx_ijbc =
                                    ((i_idx * n_occ + j_idx) * n_virt + b_idx) * n_virt + c_idx;

                                let t_ijac = t_amp[idx_ijac];
                                let t_ijca = t_amp[idx_ijca];
                                let t_ijbc = t_amp[idx_ijbc];

                                sum_ab += 2.0 * (2.0 * t_ijac - t_ijca) * t_ijbc;
                            }
                        }
                    }
                    rdm[(a, b)] += sum_ab;
                }
            }

            Some(rdm)
        } else {
            None
        };

        Ok(MP2Result {
            reference_energy: self.hf_res.total_energy,
            correlation_energy: e_corr,
            e_same_spin: e_ss,
            e_opposite_spin: e_os,
            e_scs,
            orbital_space: self.space.clone(),
            one_rdm_mo,
        })
    }
}

/// Results of an open-shell UHF-MP2 calculation.
#[derive(Clone, Debug)]
pub struct UHFMP2Result {
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub e_alpha_alpha: f64,
    pub e_beta_beta: f64,
    pub e_alpha_beta: f64,
    pub spin_orbital_space: SpinOrbitalSpace,
}

impl PostHFResult for UHFMP2Result {
    fn method_name(&self) -> &'static str {
        "UHF-MP2"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }
}

/// Open-shell UHF-MP2 calculation engine.
#[derive(Clone, Debug)]
pub struct UHFMP2<'a> {
    pub integral_cache: Option<&'a MoIntegralCache<'a>>,
    pub uhf_res: &'a UHFResult,
    pub space: SpinOrbitalSpace,
}

impl<'a> UHFMP2<'a> {
    pub fn new(uhf_res: &'a UHFResult) -> Self {
        let space = uhf_res.spin_orbital_space();
        Self {
            uhf_res,
            space,
            integral_cache: None,
        }
    }
}

impl PostHFMethod for UHFMP2<'_> {
    type Output = UHFMP2Result;

    fn compute(&self) -> Result<UHFMP2Result, String> {
        let c_a = &self.uhf_res.mo_coefficients_alpha;
        let c_b = &self.uhf_res.mo_coefficients_beta;
        let eri_ao = &self.uhf_res.eri;
        let eps_a = &self.uhf_res.orbital_energies_alpha;
        let eps_b = &self.uhf_res.orbital_energies_beta;

        let mo_eri_a = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.full().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(transform_eri_ao_to_mo(c_a, eri_ao))
        };
        let mo_eri_b = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.full_beta().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(transform_eri_ao_to_mo(c_b, eri_ao))
        };

        let occ_a = self.space.alpha.correlated_occ_range();
        let virt_a = self.space.alpha.correlated_virt_range();
        let occ_b = self.space.beta.correlated_occ_range();
        let virt_b = self.space.beta.correlated_virt_range();

        // 1. Alpha-Alpha: 1/4 * sum_{ijab} <ij||ab>_a^2 / (eps_i^a + eps_j^a - eps_a^a - eps_b^a)
        let mut e_aa = 0.0;
        for i in occ_a.clone() {
            for j in occ_a.clone() {
                for a in virt_a.clone() {
                    for b in virt_a.clone() {
                        let denom = crate::context::orbital_denominator(
                            eps_a[i], eps_a[j], eps_a[a], eps_a[b],
                        )
                        .map_err(|e| e.to_string())?;
                        let asym = mo_eri_a.anti_symmetrized(i, j, a, b);
                        e_aa += 0.25 * asym * asym / denom;
                    }
                }
            }
        }

        // 2. Beta-Beta: 1/4 * sum_{ijab} <ij||ab>_b^2 / (eps_i^b + eps_j^b - eps_a^b - eps_b^b)
        let mut e_bb = 0.0;
        for i in occ_b.clone() {
            for j in occ_b.clone() {
                for a in virt_b.clone() {
                    for b in virt_b.clone() {
                        let denom = crate::context::orbital_denominator(
                            eps_b[i], eps_b[j], eps_b[a], eps_b[b],
                        )
                        .map_err(|e| e.to_string())?;
                        let asym = mo_eri_b.anti_symmetrized(i, j, a, b);
                        e_bb += 0.25 * asym * asym / denom;
                    }
                }
            }
        }

        let cross = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.cross().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(transform_mixed_eri(c_a, c_b, eri_ao).map_err(|e| e.to_string())?)
        };
        let e_ab = occ_a
            .into_par_iter()
            .map(|i| {
                let mut energy = 0.0;
                for a in virt_a.clone() {
                    for j in occ_b.clone() {
                        for b in virt_b.clone() {
                            let v = cross.get(i, a, j, b);
                            let denominator = crate::context::orbital_denominator(
                                eps_a[i], eps_b[j], eps_a[a], eps_b[b],
                            )
                            .map_err(|e| e.to_string())?;
                            energy += v * v / denominator;
                        }
                    }
                }
                Ok::<f64, String>(energy)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<f64>();

        let e_corr = e_aa + e_bb + e_ab;

        Ok(UHFMP2Result {
            reference_energy: self.uhf_res.total_energy,
            correlation_energy: e_corr,
            e_alpha_alpha: e_aa,
            e_beta_beta: e_bb,
            e_alpha_beta: e_ab,
            spin_orbital_space: self.space.clone(),
        })
    }
}

impl MethodMeta for RHFMP2<'_> {
    const METHOD: MethodId = MethodId::Mp2;
}
impl MethodMeta for UHFMP2<'_> {
    const METHOD: MethodId = MethodId::Mp2;
}
impl<'a> ContextMethod<'a> for RHFMP2<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        let hf = ctx.rhf()?;
        if options.frozen_core > hf.n_occ
            || options.frozen_virtual > hf.mo_coefficients.cols - hf.n_occ
        {
            return Err(HfError::InvalidInput(
                "frozen MP2 partition exceeds reference space".into(),
            ));
        }
        let mut engine = Self::new(hf);
        engine.integral_cache = Some(&ctx.ints);
        engine.space = engine
            .space
            .with_frozen_core(options.frozen_core)
            .with_frozen_virtual(options.frozen_virtual);
        engine.c_same_spin = options.scs_same_spin;
        engine.c_opposite_spin = options.scs_opposite_spin;
        Ok(engine)
    }
}
impl<'a> ContextMethod<'a> for UHFMP2<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        let hf = ctx.uhf()?;
        if options.frozen_core > hf.n_alpha.min(hf.n_beta)
            || options.frozen_virtual > hf.mo_coefficients_alpha.cols - hf.n_alpha.max(hf.n_beta)
        {
            return Err(HfError::InvalidInput(
                "frozen UMP2 partition exceeds reference space".into(),
            ));
        }
        let mut engine = Self::new(hf);
        engine.integral_cache = Some(&ctx.ints);
        engine.space.alpha = engine
            .space
            .alpha
            .with_frozen_core(options.frozen_core)
            .with_frozen_virtual(options.frozen_virtual);
        engine.space.beta = engine
            .space
            .beta
            .with_frozen_core(options.frozen_core)
            .with_frozen_virtual(options.frozen_virtual);
        Ok(engine)
    }
}
