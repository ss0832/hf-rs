// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Closed-Shell and Open-Shell Coupled Cluster Singles and Doubles (CCSD).
//!
//! Features:
//! - Canonical RHF-CCSD with spin-adapted T1 and T2 amplitudes.
//! - Tensor DIIS (TDIIS) convergence acceleration.
//! - Parallel tensor contractions using Rayon and Faer.
//! - Open-shell UHF-CCSD engine for radicals and high-spin configurations.

use crate::context::{MoIntegralCache, PostHFContext, transform_mixed_eri};
use crate::error::HfError;
use crate::mo_integrals::transform_eri_ao_to_mo;
use crate::options::PostHFOptions;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::{ContextMethod, MethodId, MethodMeta};
use crate::scf::{HFResult, UHFResult};
use crate::solvers::TensorDIIS;
use rayon::prelude::*;
use std::borrow::Cow;

/// Configuration options for Coupled Cluster calculations.
#[derive(Clone, Debug)]
pub struct CCSDOptions {
    pub max_iterations: usize,
    pub energy_tol: f64,
    pub residual_tol: f64,
    pub diis_subspace: usize,
}

impl Default for CCSDOptions {
    fn default() -> Self {
        Self {
            max_iterations: 80,
            energy_tol: 1e-8,
            residual_tol: 1e-7,
            diis_subspace: 6,
        }
    }
}

/// Results of a Coupled Cluster Singles and Doubles calculation.
#[derive(Clone, Debug)]
pub struct CCSDResult {
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub total_energy: f64,
    pub iterations: usize,
    pub converged: bool,
    pub t1_amplitudes: Vec<f64>, // [n_occ * n_virt]
    pub t2_amplitudes: Vec<f64>, // [n_occ * n_occ * n_virt * n_virt]
    pub n_occ: usize,
    pub n_virt: usize,
}

impl PostHFResult for CCSDResult {
    fn intermediates(&self, max_elements: usize) -> Result<crate::output::TensorMap, HfError> {
        use crate::output::{ObservableUnit, TensorExport};
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.tensor(
            "method.t1",
            &self.t1_amplitudes,
            &[self.n_occ, self.n_virt],
            &["occupied", "virtual"],
            ObservableUnit::One,
        )?;
        tensors.tensor(
            "method.t2",
            &self.t2_amplitudes,
            &[self.n_occ, self.n_occ, self.n_virt, self.n_virt],
            &["occupied_i", "occupied_j", "virtual_a", "virtual_b"],
            ObservableUnit::One,
        )?;
        Ok(tensors.finish())
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn method_name(&self) -> &'static str {
        "CCSD"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }

    fn total_energy(&self) -> f64 {
        self.total_energy
    }
}

/// Closed-shell RHF-CCSD calculation engine.
pub struct CCSD<'a> {
    pub integral_cache: Option<&'a MoIntegralCache<'a>>,
    pub hf: &'a HFResult,
    pub options: CCSDOptions,
}

impl<'a> CCSD<'a> {
    pub fn new(hf: &'a HFResult) -> Self {
        Self {
            integral_cache: None,
            hf,
            options: CCSDOptions::default(),
        }
    }

    pub fn with_options(mut self, options: CCSDOptions) -> Self {
        self.options = options;
        self
    }
}

/// Core general spin-orbital CCSD solver implementing Stanton, Gauss, Watts & Bartlett,
/// J. Chem. Phys. 94, 4334 (1991), Eqs. (1)-(10).
///
/// Spin-orbital indexing convention for `antisym_eri(p, q, r, s)`:
/// - `0..n_so_occ` are occupied spin-orbitals
/// - `n_so_occ..(n_so_occ + n_so_virt)` are virtual spin-orbitals
pub(crate) type SpinOrbitalSolution = (f64, usize, bool, Vec<f64>, Vec<f64>);

pub(crate) fn solve_spin_orbital_ccsd<F>(
    n_so_occ: usize,
    n_so_virt: usize,
    so_eps_occ: &[f64],
    so_eps_virt: &[f64],
    antisym_eri: F,
    options: &CCSDOptions,
) -> Result<SpinOrbitalSolution, String>
where
    F: Fn(usize, usize, usize, usize) -> f64 + Sync,
{
    solve_general_spin_orbital_ccsd(
        n_so_occ,
        n_so_virt,
        so_eps_occ,
        so_eps_virt,
        antisym_eri,
        options,
        None,
        None,
    )
}
/// Linear Galerkin projector and optional initial amplitudes. Components outside
/// the projected space retain their supplied initial value (fixed weak pairs).
pub(crate) struct CcsdProjection<'a> {
    pub project: &'a dyn Fn(&mut [f64], &mut [f64]),
    pub initial: Option<(&'a [f64], &'a [f64])>,
}
/// Optional additional true residual, evaluated on the current amplitudes.
type SpinResidualCorrection<'a> =
    dyn Fn(&[f64], &[f64], &mut [f64], &mut [f64]) -> Result<(), String> + 'a;
pub(crate) type SpatialResidualCorrection<'a> =
    dyn Fn(&[f64]) -> Result<(Vec<f64>, Vec<f64>), String> + 'a;
/// Noncanonical occupied/virtual Fock blocks are included in every residual.
/// Projection acts on true residuals before preconditioning and DIIS.
pub(crate) fn solve_general_spin_orbital_ccsd<F>(
    n_so_occ: usize,
    n_so_virt: usize,
    so_eps_occ: &[f64],
    so_eps_virt: &[f64],
    antisym_eri: F,
    options: &CCSDOptions,
    fock: Option<(&crate::Matrix, &crate::Matrix)>,
    projection: Option<CcsdProjection<'_>>,
) -> Result<SpinOrbitalSolution, String>
where
    F: Fn(usize, usize, usize, usize) -> f64 + Sync,
{
    solve_general_spin_orbital_ccsd_with_residual(
        n_so_occ,
        n_so_virt,
        so_eps_occ,
        so_eps_virt,
        antisym_eri,
        options,
        fock,
        projection,
        None,
    )
}

fn solve_general_spin_orbital_ccsd_with_residual<F>(
    n_so_occ: usize,
    n_so_virt: usize,
    so_eps_occ: &[f64],
    so_eps_virt: &[f64],
    antisym_eri: F,
    options: &CCSDOptions,
    fock: Option<(&crate::Matrix, &crate::Matrix)>,
    projection: Option<CcsdProjection<'_>>,
    correction: Option<&SpinResidualCorrection<'_>>,
) -> Result<SpinOrbitalSolution, String>
where
    F: Fn(usize, usize, usize, usize) -> f64 + Sync,
{
    if so_eps_occ.len() != n_so_occ
        || so_eps_virt.len() != n_so_virt
        || options.max_iterations == 0
        || !options.residual_tol.is_finite()
        || options.residual_tol <= 0.
        || !options.energy_tol.is_finite()
        || options.energy_tol <= 0.
    {
        return Err("invalid CCSD dimensions or convergence controls".into());
    }
    if let Some((fo, fv)) = fock {
        for (f, n, eps) in [(fo, n_so_occ, so_eps_occ), (fv, n_so_virt, so_eps_virt)] {
            if f.rows != n
                || f.cols != n
                || f.data.iter().any(|v| !v.is_finite())
                || (0..n).any(|i| {
                    (f[(i, i)] - eps[i]).abs() > 1e-10
                        || (0..i).any(|j| (f[(i, j)] - f[(j, i)]).abs() > 1e-12)
                })
            {
                return Err("noncanonical CCSD Fock blocks must be finite, symmetric and have the supplied diagonal".into());
            }
        }
    }
    let estimate = (n_so_virt as u128).pow(4)
        + 48 * (n_so_occ as u128).pow(2) * (n_so_virt as u128).pow(2)
        + 8 * (n_so_occ as u128).pow(4);
    if estimate * 8 > 256 * 1024 * 1024 {
        return Err(
            "CCSD spin-orbital workspace exceeds 256 MiB; a spatial solver is required".into(),
        );
    }
    let dim_so_t1 = n_so_occ * n_so_virt;
    let dim_so_t2 = n_so_occ * n_so_occ * n_so_virt * n_so_virt;
    let dim_total = dim_so_t1 + dim_so_t2;

    let mut so_t1 = vec![0.0; dim_so_t1];
    let mut so_d1 = vec![0.0; dim_so_t1];

    for i in 0..n_so_occ {
        for a in 0..n_so_virt {
            so_d1[i * n_so_virt + a] = so_eps_occ[i] - so_eps_virt[a];
        }
    }

    let guess = crate::context::spin_mp2_guess(so_eps_occ, so_eps_virt, &antisym_eri)
        .map_err(|e| e.to_string())?;
    let mut so_t2 = guess.amplitudes;
    let so_d2 = guess.denominators;
    if so_d1.iter().any(|d| !d.is_finite() || d.abs() < 1e-14) {
        return Err("singular CC singles denominator".into());
    }

    if let Some(space) = &projection {
        if let Some((t1, t2)) = space.initial {
            if t1.len() != so_t1.len() || t2.len() != so_t2.len() {
                return Err("CCSD initial amplitude dimensions mismatch".into());
            }
            so_t1.copy_from_slice(t1);
            so_t2.copy_from_slice(t2);
        } else {
            (space.project)(&mut so_t1, &mut so_t2);
        }
    }
    let fixed = projection.as_ref().map(|space| {
        let mut p1 = so_t1.clone();
        let mut p2 = so_t2.clone();
        (space.project)(&mut p1, &mut p2);
        (
            so_t1.iter().zip(p1).map(|(t, p)| t - p).collect::<Vec<_>>(),
            so_t2.iter().zip(p2).map(|(t, p)| t - p).collect::<Vec<_>>(),
        )
    });
    let n_oo = n_so_occ * n_so_occ;
    let n_vv = n_so_virt * n_so_virt;
    let n_ov = n_so_occ * n_so_virt;

    // Precompute V_pp matrix: <ab||ef> (n_vv x n_vv, rows = ef, cols = ab)
    let v_pp_mat = faer::Mat::<f64>::from_fn(n_vv, n_vv, |ef, ab| {
        let a = ab / n_so_virt;
        let b = ab % n_so_virt;
        let e = ef / n_so_virt;
        let f = ef % n_so_virt;
        antisym_eri(n_so_occ + a, n_so_occ + b, n_so_occ + e, n_so_occ + f)
    });

    // Precompute V_hh matrix: <mn||ij> (n_oo x n_oo, rows = ij, cols = mn)
    let v_hh_mat = faer::Mat::<f64>::from_fn(n_oo, n_oo, |ij, mn| {
        let i = ij / n_so_occ;
        let j = ij % n_so_occ;
        let m = mn / n_so_occ;
        let n = mn % n_so_occ;
        antisym_eri(m, n, i, j)
    });

    // Precompute V_oovv matrix: <ij||ab> (n_oo x n_vv)
    let v_oovv_mat = faer::Mat::<f64>::from_fn(n_oo, n_vv, |ij, ab| {
        let i = ij / n_so_occ;
        let j = ij % n_so_occ;
        let a = ab / n_so_virt;
        let b = ab % n_so_virt;
        antisym_eri(i, j, n_so_occ + a, n_so_occ + b)
    });

    // Precompute V_ph: <mb||ej> (n_ov x n_ov)
    let v_ph_data: Vec<f64> = (0..n_ov * n_ov)
        .into_par_iter()
        .map(|idx| {
            let mb = idx / n_ov;
            let ej = idx % n_ov;
            let m = mb / n_so_virt;
            let b = mb % n_so_virt;
            let e = ej / n_so_occ;
            let j = ej % n_so_occ;
            antisym_eri(m, n_so_occ + b, n_so_occ + e, j)
        })
        .collect();

    let mut tdiis = TensorDIIS::new(options.diis_subspace);
    let mut last_e_corr = 0.0;
    let mut converged = false;
    let mut iters = 0;

    for iter in 0..options.max_iterations {
        iters = iter + 1;

        // Build effective doubles tau_{ij}^{ab} and tau_tilde_{ij}^{ab} (Stanton Eqs. 9-10)
        let mut tau = vec![0.0; dim_so_t2];
        let mut tau_tilde = vec![0.0; dim_so_t2];
        for i in 0..n_so_occ {
            for j in 0..n_so_occ {
                let ij = i * n_so_occ + j;
                for a in 0..n_so_virt {
                    let t_ia = so_t1[i * n_so_virt + a];
                    for b in 0..n_so_virt {
                        let ab = a * n_so_virt + b;
                        let idx = ij * n_vv + ab;
                        let disc = t_ia * so_t1[j * n_so_virt + b]
                            - so_t1[i * n_so_virt + b] * so_t1[j * n_so_virt + a];
                        tau[idx] = so_t2[idx] + disc;
                        tau_tilde[idx] = so_t2[idx] + 0.5 * disc;
                    }
                }
            }
        }

        let tau_mat = faer::MatRef::from_row_major_slice(&tau, n_oo, n_vv);
        let tau_tilde_mat = faer::MatRef::from_row_major_slice(&tau_tilde, n_oo, n_vv);

        // 1. Current correlation energy: E_corr = 1/4 sum_{ijab} <ij||ab> tau_{ij}^{ab}
        let mut e_corr = 0.0;
        for ij in 0..n_oo {
            for ab in 0..n_vv {
                e_corr += 0.25 * v_oovv_mat[(ij, ab)] * tau_mat[(ij, ab)];
            }
        }

        let de = (e_corr - last_e_corr).abs();
        last_e_corr = e_corr;

        // 2. Build Stanton et al. one-body intermediates: F_{me}, F_{ae}, F_{mi} (Eqs. 3-5)
        let mut f_ov = vec![0.0; n_so_occ * n_so_virt];
        for m in 0..n_so_occ {
            for e in 0..n_so_virt {
                let mut sum = 0.0;
                for n in 0..n_so_occ {
                    let mn = m * n_so_occ + n;
                    for f in 0..n_so_virt {
                        let ef = e * n_so_virt + f;
                        sum += so_t1[n * n_so_virt + f] * v_oovv_mat[(mn, ef)];
                    }
                }
                f_ov[m * n_so_virt + e] = sum;
            }
        }

        let mut f_vv = vec![0.0; n_so_virt * n_so_virt];
        for a in 0..n_so_virt {
            let a_so = n_so_occ + a;
            for e in 0..n_so_virt {
                let e_so = n_so_occ + e;
                let mut val = fock.map_or(0., |(_, fv)| if a == e { 0. } else { fv[(a, e)] });
                for m in 0..n_so_occ {
                    for f in 0..n_so_virt {
                        let f_so = n_so_occ + f;
                        val += so_t1[m * n_so_virt + f] * antisym_eri(m, a_so, f_so, e_so);
                    }
                }
                for mn in 0..n_oo {
                    for f in 0..n_so_virt {
                        let af = a * n_so_virt + f;
                        let ef = e * n_so_virt + f;
                        val -= 0.5 * tau_tilde_mat[(mn, af)] * v_oovv_mat[(mn, ef)];
                    }
                }
                f_vv[a * n_so_virt + e] = val;
            }
        }

        let mut f_oo = vec![0.0; n_so_occ * n_so_occ];
        for m in 0..n_so_occ {
            for i in 0..n_so_occ {
                let mut val = fock.map_or(0., |(fo, _)| if m == i { 0. } else { fo[(m, i)] });
                for n in 0..n_so_occ {
                    for e in 0..n_so_virt {
                        let e_so = n_so_occ + e;
                        val += so_t1[n * n_so_virt + e] * antisym_eri(m, n, i, e_so);
                    }
                }
                for n in 0..n_so_occ {
                    let in_idx = i * n_so_occ + n;
                    let mn_idx = m * n_so_occ + n;
                    for ef in 0..n_vv {
                        val += 0.5 * tau_tilde_mat[(in_idx, ef)] * v_oovv_mat[(mn_idx, ef)];
                    }
                }
                f_oo[m * n_so_occ + i] = val;
            }
        }

        // Modified F_ae and F_mi for the T2 equation:
        // F_tilde_{be} = F_{be} - 0.5 * sum_m t_m^b F_{me}
        // F_tilde_{mj} = F_{mj} + 0.5 * sum_e t_j^e F_{me}
        let mut f_vv_t2 = f_vv.clone();
        for b in 0..n_so_virt {
            for e in 0..n_so_virt {
                let mut corr = 0.0;
                for m in 0..n_so_occ {
                    corr += so_t1[m * n_so_virt + b] * f_ov[m * n_so_virt + e];
                }
                f_vv_t2[b * n_so_virt + e] -= 0.5 * corr;
            }
        }
        let mut f_oo_t2 = f_oo.clone();
        for m in 0..n_so_occ {
            for j in 0..n_so_occ {
                let mut corr = 0.0;
                for e in 0..n_so_virt {
                    corr += so_t1[j * n_so_virt + e] * f_ov[m * n_so_virt + e];
                }
                f_oo_t2[m * n_so_occ + j] += 0.5 * corr;
            }
        }

        // 3. Build two-body intermediates W_mnij, W_abef, W_mbej (Stanton Eqs. 6-8)
        // Contraction of tau with V_pp and V_hh + quadratic tau * V_oovv * tau
        let w_pp_tau = tau_mat * &v_pp_mat;
        let w_hh_tau = &v_hh_mat * tau_mat;
        let v_oovv_tau_t = &v_oovv_mat * tau_mat.transpose();
        let q1_tau_mat = v_oovv_tau_t.transpose() * tau_mat;

        // T1-dressed part of W_mnij: P(ij) sum_e t_j^e <mn||ie>
        let w_mnij_t1 = faer::Mat::<f64>::from_fn(n_oo, n_oo, |ij, mn| {
            let i = ij / n_so_occ;
            let j = ij % n_so_occ;
            let m = mn / n_so_occ;
            let n = mn % n_so_occ;
            let mut sum = 0.0;
            for e in 0..n_so_virt {
                let e_so = n_so_occ + e;
                sum += so_t1[j * n_so_virt + e] * antisym_eri(m, n, i, e_so)
                    - so_t1[i * n_so_virt + e] * antisym_eri(m, n, j, e_so);
            }
            sum
        });
        let w_hh_t1_contrib = &w_mnij_t1 * tau_mat;

        // T1-dressed part of W_abef: -P(ab) sum_m t_m^b <am||ef>
        // Contracted with 0.5 * sum_{ef} tau_{ij}^{ef}:
        // -0.5 * P(ab) sum_m t_m^b sum_{ef} tau_{ij}^{ef} <am||ef>
        // Wait: <am||ef> = -<ma||ef>, so we can precompute G_{ma, ij} = sum_{ef} tau_{ij}^{ef} <am||ef>
        let v_am_ef = faer::Mat::<f64>::from_fn(n_vv, n_ov, |ef, ma| {
            let e = ef / n_so_virt;
            let f = ef % n_so_virt;
            let m = ma / n_so_virt;
            let a = ma % n_so_virt;
            antisym_eri(n_so_occ + a, m, n_so_occ + e, n_so_occ + f)
        });
        // tau_v_am: (n_oo x n_ov), entry (ij, ma) = sum_{ef} tau_{ij}^{ef} <am||ef>
        let tau_v_am = tau_mat * &v_am_ef;

        // Ring intermediate W_{mbej} (Eq. 8):
        // W_{mbej} = <mb||ej> + sum_f t_j^f <mb||ef> - sum_n t_n^b <mn||ej>
        //            - sum_{nf} (0.5 * t_{jn}^{fb} + t_j^f t_n^b) <mn||ef>
        let w_mbej: Vec<f64> = (0..n_ov * n_ov)
            .into_par_iter()
            .map(|idx| {
                let mb = idx / n_ov;
                let ej = idx % n_ov;
                let m = mb / n_so_virt;
                let b = mb % n_so_virt;
                let e = ej / n_so_occ;
                let j = ej % n_so_occ;
                let b_so = n_so_occ + b;
                let e_so = n_so_occ + e;

                let mut val = v_ph_data[idx];
                for f in 0..n_so_virt {
                    let f_so = n_so_occ + f;
                    val += so_t1[j * n_so_virt + f] * antisym_eri(m, b_so, e_so, f_so);
                }
                for n in 0..n_so_occ {
                    val -= so_t1[n * n_so_virt + b] * antisym_eri(m, n, e_so, j);
                }
                for n in 0..n_so_occ {
                    let mn = m * n_so_occ + n;
                    let jn = j * n_so_occ + n;
                    let t_nb = so_t1[n * n_so_virt + b];
                    for f in 0..n_so_virt {
                        let ef = e * n_so_virt + f;
                        let fb = f * n_so_virt + b;
                        let t_jf = so_t1[j * n_so_virt + f];
                        val -= (0.5 * so_t2[jn * n_vv + fb] + t_jf * t_nb) * v_oovv_mat[(mn, ef)];
                    }
                }
                val
            })
            .collect();

        // 4. Spin-orbital T1 residuals: R_i^a (Stanton Eq. 1)
        let mut r1 = vec![0.0; dim_so_t1];
        for i in 0..n_so_occ {
            for a in 0..n_so_virt {
                let a_so = n_so_occ + a;
                let mut val1 = 0.0;

                // + sum_e t_i^e F_{ae}
                for e in 0..n_so_virt {
                    val1 += so_t1[i * n_so_virt + e] * f_vv[a * n_so_virt + e];
                }
                // - sum_m t_m^a F_{mi}
                for m in 0..n_so_occ {
                    val1 -= so_t1[m * n_so_virt + a] * f_oo[m * n_so_occ + i];
                }
                // + sum_{me} t_{im}^{ae} F_{me} - sum_{nf} t_n^f <na||if>
                for m in 0..n_so_occ {
                    let im = i * n_so_occ + m;
                    for e in 0..n_so_virt {
                        let ae = a * n_so_virt + e;
                        let e_so = n_so_occ + e;
                        val1 += so_t2[im * n_vv + ae] * f_ov[m * n_so_virt + e];
                        val1 += antisym_eri(m, a_so, e_so, i) * so_t1[m * n_so_virt + e];
                    }
                }
                // - 0.5 sum_{mef} t_{im}^{ef} <ma||ef> (= +0.5 sum_{mef} <ma||fe> t_{im}^{ef})
                for m in 0..n_so_occ {
                    let im = i * n_so_occ + m;
                    for e in 0..n_so_virt {
                        let e_so = n_so_occ + e;
                        for f in 0..n_so_virt {
                            let f_so = n_so_occ + f;
                            let ef = e * n_so_virt + f;
                            val1 -= 0.5 * so_t2[im * n_vv + ef] * antisym_eri(m, a_so, e_so, f_so);
                        }
                    }
                }
                // - 0.5 sum_{mne} t_{mn}^{ae} <nm||ei> (= -0.5 sum_{mne} <mn||ei> t_{mn}^{ae})
                for m in 0..n_so_occ {
                    for n in 0..n_so_occ {
                        let mn = m * n_so_occ + n;
                        for e in 0..n_so_virt {
                            let e_so = n_so_occ + e;
                            let ae = a * n_so_virt + e;
                            val1 -= 0.5 * so_t2[mn * n_vv + ae] * antisym_eri(n, m, e_so, i);
                        }
                    }
                }
                r1[i * n_so_virt + a] = val1;
            }
        }

        // 5. Spin-orbital T2 residuals: R_{ij}^{ab} (Stanton Eq. 2)
        let mut r2: Vec<f64> = (0..n_so_occ)
            .into_par_iter()
            .flat_map(|i| {
                let mut slice_i = Vec::with_capacity(n_so_occ * n_so_virt * n_so_virt);
                for j in 0..n_so_occ {
                    let ij = i * n_so_occ + j;
                    for a in 0..n_so_virt {
                        let a_so = n_so_occ + a;
                        for b in 0..n_so_virt {
                            let b_so = n_so_occ + b;
                            let ab = a * n_so_virt + b;

                            // <ij||ab> + 0.5 * tau * W_abef (bare + Q1/2) + 0.5 * tau * W_mnij (bare + T1 + Q1/2)
                            let mut val = v_oovv_mat[(ij, ab)]
                                + 0.5 * w_pp_tau[(ij, ab)]
                                + 0.5 * w_hh_tau[(ij, ab)]
                                + 0.5 * w_hh_t1_contrib[(ij, ab)]
                                + 0.25 * q1_tau_mat[(ij, ab)];

                            // T1-dressed part of 0.5 * sum_{ef} tau_{ij}^{ef} W_{abef}:
                            // -0.5 * P(ab) sum_m t_m^b (sum_{ef} tau_{ij}^{ef} <am||ef>)
                            for m in 0..n_so_occ {
                                let ma = m * n_so_virt + a;
                                let mb = m * n_so_virt + b;
                                val -= 0.5
                                    * (so_t1[mb] * tau_v_am[(ij, ma)]
                                        - so_t1[ma] * tau_v_am[(ij, mb)]);
                            }

                            // + P(ab) sum_e t_{ij}^{ae} F_tilde_{be}
                            for e in 0..n_so_virt {
                                let ae = a * n_so_virt + e;
                                let be = b * n_so_virt + e;
                                val += so_t2[ij * n_vv + ae] * f_vv_t2[b * n_so_virt + e]
                                    - so_t2[ij * n_vv + be] * f_vv_t2[a * n_so_virt + e];
                            }

                            // - P(ij) sum_m t_{im}^{ab} F_tilde_{mj}
                            for m in 0..n_so_occ {
                                let im = i * n_so_occ + m;
                                let jm = j * n_so_occ + m;
                                val -= so_t2[im * n_vv + ab] * f_oo_t2[m * n_so_occ + j]
                                    - so_t2[jm * n_vv + ab] * f_oo_t2[m * n_so_occ + i];
                            }

                            // + P(ij)P(ab) sum_{me} [ t_{im}^{ae} W_{mbej} - t_i^e t_m^a <mb||ej> ]
                            for m in 0..n_so_occ {
                                let t_ma = so_t1[m * n_so_virt + a];
                                let t_mb = so_t1[m * n_so_virt + b];
                                for e in 0..n_so_virt {
                                    let t_ie = so_t1[i * n_so_virt + e];
                                    let t_je = so_t1[j * n_so_virt + e];

                                    let idx_im_ae =
                                        ((i * n_so_occ + m) * n_so_virt + a) * n_so_virt + e;
                                    let idx_im_be =
                                        ((i * n_so_occ + m) * n_so_virt + b) * n_so_virt + e;
                                    let idx_jm_ae =
                                        ((j * n_so_occ + m) * n_so_virt + a) * n_so_virt + e;
                                    let idx_jm_be =
                                        ((j * n_so_occ + m) * n_so_virt + b) * n_so_virt + e;

                                    let mb_ej = (m * n_so_virt + b) * n_ov + (e * n_so_occ + j);
                                    let ma_ej = (m * n_so_virt + a) * n_ov + (e * n_so_occ + j);
                                    let mb_ei = (m * n_so_virt + b) * n_ov + (e * n_so_occ + i);
                                    let ma_ei = (m * n_so_virt + a) * n_ov + (e * n_so_occ + i);

                                    val += (so_t2[idx_im_ae] * w_mbej[mb_ej]
                                        - t_ie * t_ma * v_ph_data[mb_ej])
                                        - (so_t2[idx_im_be] * w_mbej[ma_ej]
                                            - t_ie * t_mb * v_ph_data[ma_ej])
                                        - (so_t2[idx_jm_ae] * w_mbej[mb_ei]
                                            - t_je * t_ma * v_ph_data[mb_ei])
                                        + (so_t2[idx_jm_be] * w_mbej[ma_ei]
                                            - t_je * t_mb * v_ph_data[ma_ei]);
                                }
                            }

                            // + P(ij) sum_e t_i^e <ab||ej> - P(ab) sum_m t_m^a <mb||ij>
                            for e in 0..n_so_virt {
                                let e_so = n_so_occ + e;
                                val += antisym_eri(a_so, b_so, e_so, j) * so_t1[i * n_so_virt + e]
                                    - antisym_eri(a_so, b_so, e_so, i) * so_t1[j * n_so_virt + e];
                            }
                            for m in 0..n_so_occ {
                                val -= antisym_eri(m, b_so, i, j) * so_t1[m * n_so_virt + a]
                                    - antisym_eri(m, a_so, i, j) * so_t1[m * n_so_virt + b];
                            }

                            slice_i.push(val);
                        }
                    }
                }
                slice_i
            })
            .collect();

        // True amplitude residuals, followed by the Galerkin projection.
        for (index, &d) in so_d1.iter().enumerate() {
            r1[index] -= d * so_t1[index];
        }
        for (index, &d) in so_d2.iter().enumerate() {
            r2[index] -= d * so_t2[index];
        }
        if let Some(add_residual) = correction {
            add_residual(&so_t1, &so_t2, &mut r1, &mut r2)?;
        }
        if let Some(space) = &projection {
            (space.project)(&mut r1, &mut r2);
        }
        let max_res = r1.iter().chain(&r2).map(|v| v.abs()).fold(0., f64::max);
        if !e_corr.is_finite() || r1.iter().chain(&r2).any(|v| !v.is_finite()) {
            return Err("CCSD energy/residual became non-finite".into());
        }
        if de < options.energy_tol && max_res <= options.residual_tol {
            converged = true;
            break;
        }
        let mut err_vec = Vec::with_capacity(dim_total);
        err_vec.extend_from_slice(&r1);
        err_vec.extend_from_slice(&r2);
        for (r, d) in r1.iter_mut().zip(&so_d1) {
            *r /= d;
        }
        for (r, d) in r2.iter_mut().zip(&so_d2) {
            *r /= d;
        }
        if let Some(space) = &projection {
            (space.project)(&mut r1, &mut r2);
        }
        let mut next_t = Vec::with_capacity(dim_total);
        next_t.extend(so_t1.iter().zip(&r1).map(|(t, step)| t + step));
        next_t.extend(so_t2.iter().zip(&r2).map(|(t, step)| t + step));

        // DIIS extrapolation
        tdiis.push(next_t.clone(), err_vec);
        let p_extrap = tdiis.extrapolate().unwrap_or(next_t);

        so_t1.copy_from_slice(&p_extrap[..dim_so_t1]);
        so_t2.copy_from_slice(&p_extrap[dim_so_t1..]);
        if let (Some(space), Some((f1, f2))) = (&projection, &fixed) {
            (space.project)(&mut so_t1, &mut so_t2);
            for (t, f) in so_t1.iter_mut().zip(f1) {
                *t += f;
            }
            for (t, f) in so_t2.iter_mut().zip(f2) {
                *t += f;
            }
        }
    }

    Ok((last_e_corr, iters, converged, so_t1, so_t2))
}

impl<'a> PostHFMethod for CCSD<'a> {
    type Output = CCSDResult;

    fn compute(&self) -> Result<Self::Output, String> {
        self.compute_with_spatial_residual(0, None)
    }
}

impl CCSD<'_> {
    /// Correlate occupied orbitals after `frozen_core`; keep the full reference
    /// Fock operator and map all integrals back to the original MO indices.
    pub(crate) fn compute_with_spatial_residual(
        &self,
        frozen_core: usize,
        correction: Option<&SpatialResidualCorrection<'_>>,
    ) -> Result<CCSDResult, String> {
        if !self.hf.converged || frozen_core > self.hf.n_occ {
            return Err("CCSD requires a converged reference and a valid frozen core".into());
        }
        let n_occ = self.hf.n_occ - frozen_core;
        let n_basis = self.hf.orbital_energies.len();
        let n_virt = n_basis - self.hf.n_occ;

        if n_occ == 0 || n_virt == 0 {
            return Err("CCSD requires at least one occupied and one virtual orbital".to_string());
        }

        let mo_eri = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.full().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(transform_eri_ao_to_mo(
                &self.hf.mo_coefficients,
                &self.hf.eri,
            ))
        };
        super::rhf_ccsd::solve(
            self.hf.total_energy,
            &self.hf.orbital_energies,
            &mo_eri,
            frozen_core,
            n_occ,
            n_virt,
            &self.options,
            correction,
            false,
        )
    }
}

/// Open-shell UHF-CCSD calculation engine.
pub struct UHFCCSD<'a> {
    pub integral_cache: Option<&'a MoIntegralCache<'a>>,
    pub uhf: &'a UHFResult,
    pub options: CCSDOptions,
}

impl<'a> UHFCCSD<'a> {
    pub fn new(uhf: &'a UHFResult) -> Self {
        Self {
            integral_cache: None,
            uhf,
            options: CCSDOptions::default(),
        }
    }

    pub fn with_options(mut self, options: CCSDOptions) -> Self {
        self.options = options;
        self
    }
}

impl<'a> PostHFMethod for UHFCCSD<'a> {
    type Output = CCSDResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let n_a = self.uhf.n_alpha;
        let n_b = self.uhf.n_beta;
        let n_basis = self.uhf.orbital_energies_alpha.len();
        let nv_a = n_basis - n_a;
        let nv_b = n_basis - n_b;

        if (n_a == 0 && n_b == 0) || (nv_a == 0 && nv_b == 0) {
            return Err("UHF-CCSD requires non-zero occupied and virtual orbitals".to_string());
        }

        if !self.uhf.converged {
            return Err("UHF-CCSD requires a converged reference".into());
        }
        if 2 * n_basis > 150 {
            return Err("UHF-CCSD spin-orbital solver is limited to 150 spin orbitals".into());
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

        // Map spin-orbital index p_so in 0..(n_so_occ + n_so_virt) to (is_beta, spatial_mo_idx)
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

        let (last_e_corr, iters, converged, so_t1, so_t2) = solve_spin_orbital_ccsd(
            n_so_occ,
            n_so_virt,
            &so_eps_occ,
            &so_eps_virt,
            antisym_eri,
            &self.options,
        )?;

        let ref_e = self.uhf.total_energy;
        Ok(CCSDResult {
            reference_energy: ref_e,
            correlation_energy: last_e_corr,
            total_energy: ref_e + last_e_corr,
            iterations: iters,
            converged,
            t1_amplitudes: so_t1,
            t2_amplitudes: so_t2,
            n_occ: n_so_occ,
            n_virt: n_so_virt,
        })
    }
}

impl MethodMeta for CCSD<'_> {
    const METHOD: MethodId = MethodId::Ccsd;
}
impl<'a> ContextMethod<'a> for CCSD<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        if options.frozen_core != 0 || options.frozen_virtual != 0 {
            return Err(HfError::Unsupported {
                method: "CCSD".into(),
                detail: "frozen orbital CC adapter is pending".into(),
            });
        }
        let mut engine = Self::new(ctx.rhf()?);
        engine.integral_cache = Some(&ctx.ints);
        engine.options.max_iterations = options.max_iter;
        engine.options.energy_tol = options.energy_tol;
        engine.options.residual_tol = options.residual_tol;
        Ok(engine)
    }
}

impl MethodMeta for UHFCCSD<'_> {
    const METHOD: MethodId = MethodId::Ccsd;
}
impl<'a> ContextMethod<'a> for UHFCCSD<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        if options.frozen_core != 0 || options.frozen_virtual != 0 {
            return Err(HfError::Unsupported {
                method: "UHFCCSD".into(),
                detail: "frozen orbital CC adapter is pending".into(),
            });
        }
        let mut engine = Self::new(ctx.uhf()?);
        engine.integral_cache = Some(&ctx.ints);
        engine.options.max_iterations = options.max_iter;
        engine.options.energy_tol = options.energy_tol;
        engine.options.residual_tol = options.residual_tol;
        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{Atom, STO_3G_GBS, build_basis, parse_gaussian94};
    use crate::diis::ConvAlgo;
    use crate::scf::run_scf_general;

    #[test]
    fn test_h2_ccsd() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0, 0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, ConvAlgo::Adiissoscf, None)
            .expect("HF failed");

        let ccsd = CCSD::new(&hf).compute().expect("CCSD failed");
        eprintln!(
            "CCSD total energy: {:.10} Eh, corr: {:.10} Eh",
            ccsd.total_energy, ccsd.correlation_energy
        );
        assert!(ccsd.converged);
        assert!(ccsd.correlation_energy < 0.0);
        assert!(ccsd.total_energy < ccsd.reference_energy);
    }
}
