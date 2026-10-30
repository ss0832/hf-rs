// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Fixed-cusp CCSD-F12a/b with retained F12 sources in the CC iteration.
//!
//! The uneliminated V/B/X functional and current-amplitude C/tau terms are
//! evaluated after converging the modified singles and doubles equations.
//! F12b adds the projected-Coulomb/tau/singles functional. Both variants share
//! amplitudes. The 3C intermediates retain the full OBS+CABS Fock operator;
//! nonlocal ECP products use this finite RI resolution.
//! See Knizia (2010), Eqs. 4.123--4.127. Published hybrid/default-option
//! benchmarks remain a separate validation requirement.

use crate::basis::{Atom, ContractedGaussian};
use crate::context::{MoIntegralCache, PostHFContext};
use crate::error::HfError;
use crate::matrix::Matrix;
use crate::mo_integrals::MoEriTensor;
use crate::post_hf::cc::ccsd_t::compute_ccsd_t_correction_in_space;
use crate::post_hf::cc::{CCSD, CCSDOptions, CCSDResult};
use crate::post_hf::f12::ccsd_intermediates::{CcsdF12Coupling, CcsdF12EnergyCorrection};
use crate::post_hf::f12::mp2_f12::{F12Hybrid, MP2F12Options, RHFMP2F12};
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::scf::HFResult;

/// Variant of explicitly correlated coupled-cluster approximation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CCSDF12Variant {
    /// Fixed-cusp a energy functional and modified CCSD residuals.
    F12a,
    /// The same amplitude equations, with the additional b energy functional.
    F12b,
}

/// Result of a CCSD-F12 calculation.
#[derive(Clone, Debug)]
pub struct CCSDF12Result {
    /// Reference Hartree-Fock energy E_{HF} in Hartree.
    pub hf_energy: f64,
    /// CABS singles correction energy E_{CABS} in Hartree.
    pub cabs_singles_energy: f64,
    /// Conventional orbital CCSD correlation energy in Hartree.
    pub ccsd_correlation_energy: f64,
    /// Explicitly correlated F12 correction energy E_{F12} in Hartree.
    pub f12_correction_energy: f64,
    /// Current-amplitude C/tau coupling, plus projected-Coulomb terms for F12b.
    pub coupling_energy: f64,
    /// Both functionals evaluated from the same converged amplitudes. These
    /// components permit comparing a/b without rebuilding F12 intermediates.
    pub coupling_components: CcsdF12EnergyCorrection,
    /// Unscaled perturbative triples correction (T) if requested.
    pub triples_energy: Option<f64>,
    /// Scaled perturbative triples correction (T)* = (T) * (E_{MP2-F12} / E_{MP2}) if requested.
    pub scaled_triples_energy: Option<f64>,
    /// Total correlation energy in Hartree.
    pub total_correlation_energy: f64,
    /// Total energy in Hartree.
    pub total_energy: f64,
    /// Method variant (F12a or F12b).
    pub variant: CCSDF12Variant,
    pub hybrid: F12Hybrid,
    pub fitted_correction: bool,
    pub fitted_fock: bool,
    pub fitted_orbital_mp2: bool,
    pub exact_obs_coulomb: bool,
    /// Conventional MP2 denominator in (T*), using the selected orbital fit.
    pub mp2_correlation_energy: f64,
    /// Coupled MP2-F12 correlation for (T*), excluding CABS singles.
    pub mp2_f12_correlation_energy: f64,
    pub fitting_basis_kind: crate::ao::BasisKind,
    pub fock_fitting_basis_kind: crate::ao::BasisKind,
    /// Converged amplitudes of the modified CC equations. Occupied indices
    /// exclude frozen cores; virtual indices retain the original OBS virtuals.
    pub ccsd_result: CCSDResult,
}

impl PostHFResult for CCSDF12Result {
    fn method_name(&self) -> &'static str {
        match (self.variant, self.triples_energy.is_some()) {
            (CCSDF12Variant::F12a, false) => "CCSD-F12a",
            (CCSDF12Variant::F12b, false) => "CCSD-F12b",
            (CCSDF12Variant::F12a, true) => "CCSD(T*)-F12a",
            (CCSDF12Variant::F12b, true) => "CCSD(T*)-F12b",
        }
    }

    fn converged(&self) -> bool {
        self.ccsd_result.converged
    }

    fn reference_energy(&self) -> f64 {
        self.hf_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.total_correlation_energy
    }

    fn total_energy(&self) -> f64 {
        self.total_energy
    }

    fn to_output(&self) -> Result<crate::output::PostHFOutput, HfError> {
        let mut output = crate::output::PostHFOutput::new(
            self.method_name(),
            self.hf_energy,
            self.total_energy,
        )?;
        match self.hybrid {
            F12Hybrid::Full => {}
            F12Hybrid::Hy1 => output.method.push_str("/HY1"),
            F12Hybrid::Hy2 => output.method.push_str("/HY2"),
        }
        for (key, value) in [
            ("E_MP2", self.mp2_correlation_energy),
            ("E_MP2_F12", self.mp2_f12_correlation_energy),
            ("E_CCSD", self.ccsd_correlation_energy),
            ("E_F12_fixed", self.f12_correction_energy),
            ("E_F12_coupling", self.coupling_energy),
            ("E_F12_C_tau", self.coupling_components.fock_amplitude),
            (
                "E_F12_additional_b",
                self.coupling_components.additional_f12b,
            ),
            ("E_CABS_singles", self.cabs_singles_energy),
            (
                "F12_fitted_correction",
                u8::from(self.fitted_correction) as f64,
            ),
            ("F12_fitted_fock", u8::from(self.fitted_fock) as f64),
            (
                "F12_exact_obs_coulomb",
                u8::from(self.exact_obs_coulomb) as f64,
            ),
            (
                "F12_fitted_orbital_mp2",
                u8::from(self.fitted_orbital_mp2) as f64,
            ),
            (
                "F12_spherical_fit",
                u8::from(
                    self.fitted_correction
                        && self.fitting_basis_kind == crate::ao::BasisKind::Spherical,
                ) as f64,
            ),
            (
                "F12_spherical_fock_fit",
                u8::from(
                    self.fitted_fock
                        && self.fock_fitting_basis_kind == crate::ao::BasisKind::Spherical,
                ) as f64,
            ),
        ] {
            output.components.insert(key.into(), value);
        }
        if let (Some(t), Some(scaled)) = (self.triples_energy, self.scaled_triples_energy) {
            output.components.insert("E_triples".into(), t);
            output.components.insert("E_scaled_triples".into(), scaled);
        }
        Ok(output)
    }
}

/// Explicitly correlated CCSD-F12 calculation engine.
#[derive(Clone, Debug)]
pub struct RHFCCSDF12<'a> {
    pub integral_cache: Option<&'a MoIntegralCache<'a>>,
    pub hf_res: &'a HFResult,
    pub atoms: &'a [Atom],
    pub basis: &'a [ContractedGaussian],
    pub variant: CCSDF12Variant,
    pub compute_triples: bool,
    pub f12_options: MP2F12Options,
    pub ccsd_options: CCSDOptions,
}

impl<'a> RHFCCSDF12<'a> {
    /// Creates a new CCSD-F12 calculation engine.
    pub fn new(
        hf_res: &'a HFResult,
        atoms: &'a [Atom],
        basis: &'a [ContractedGaussian],
        variant: CCSDF12Variant,
    ) -> Self {
        Self {
            integral_cache: None,
            hf_res,
            atoms,
            basis,
            variant,
            compute_triples: false,
            f12_options: MP2F12Options::default(),
            ccsd_options: CCSDOptions::default(),
        }
    }

    /// Requests calculation of scaled perturbative triples (CCSD(T)-F12).
    pub fn with_triples(mut self, compute: bool) -> Self {
        self.compute_triples = compute;
        self
    }

    /// Sets custom F12 options (e.g. \gamma or CABS).
    pub fn with_f12_options(mut self, options: MP2F12Options) -> Self {
        self.f12_options = options;
        self
    }

    /// Sets custom conventional CCSD options (convergence thresholds, DIIS).
    pub fn with_ccsd_options(mut self, options: CCSDOptions) -> Self {
        self.ccsd_options = options;
        self
    }
}

impl RHFCCSDF12<'_> {
    pub fn compute_checked(&self) -> Result<CCSDF12Result, HfError> {
        let failure = |detail| HfError::PostHf {
            method: "CCSD-F12".into(),
            detail,
        };
        if self.f12_options.approximation != super::mp2_f12::F12Approximation::ApproximationC {
            return Err(HfError::Unsupported {
                method: "CCSD-F12".into(),
                detail: "the coupled a/b driver requires 3C intermediates".into(),
            });
        }
        let hf = self.hf_res;
        let fc = self.f12_options.frozen_core;
        if fc >= hf.n_occ || hf.mo_coefficients.cols <= hf.n_occ {
            return Err(HfError::InvalidInput(
                "CCSD-F12 needs correlated occupied and virtual orbitals".into(),
            ));
        }
        let context;
        let cache = if let Some(cache) = self.integral_cache {
            cache
        } else {
            context = PostHFContext::from_rhf(hf, self.atoms, self.basis, None)?;
            &context.ints
        };
        let mut mp2_engine =
            RHFMP2F12::new(hf, self.atoms, self.basis).with_options(self.f12_options.clone());
        mp2_engine.integral_cache = Some(cache);
        let (mut mp2, projected) = mp2_engine.compute_prepared(true)?;
        let projected =
            projected.ok_or_else(|| failure("missing projected F12 coupling".into()))?;
        let (no, nv, nm) = (
            hf.n_occ - fc,
            hf.mo_coefficients.cols - hf.n_occ,
            hf.mo_coefficients.cols,
        );
        // Remove frozen occupied bra indices only. The strong-orthogonality
        // projector and Fock construction above still contain every occupied MO.
        let v: Vec<_> = projected
            .into_iter()
            .map(|m| {
                if fc == 0 {
                    return m;
                }
                let mut active = Matrix::zeros(nm - fc, nm - fc);
                for p in 0..active.rows {
                    for q in 0..active.cols {
                        active[(p, q)] = m[(fc + p, fc + q)];
                    }
                }
                active
            })
            .collect();
        let mut c = vec![Matrix::zeros(nv, nv); no * no];
        let mut fixed_energy = 0.;
        for pair in &mp2.pairs {
            let [i, j] = pair.pair;
            c[(i - fc) * no + j - fc] = pair.c.clone();
            c[(j - fc) * no + i - fc] = pair.c.transpose();
            // Elimination of conventional amplitudes belongs only to MP2.
            // CCSD uses its actual converged amplitudes and the raw functional.
            let eps_sum = hf.orbital_energies[i] + hf.orbital_energies[j];
            let (singlet, triplet) = pair.fixed_uneliminated_energy(eps_sum);
            fixed_energy += singlet + triplet;
        }
        let coupling = CcsdF12Coupling::from_pair_integrals(no, nv, &v, &c)?;
        drop(v);
        drop(c);
        mp2.pairs.clear();
        let source = |t1: &[f64]| {
            let r = coupling
                .residual_corrections(t1)
                .map_err(|e| e.to_string())?;
            Ok((r.singles, r.doubles))
        };
        let mut engine = CCSD::new(hf).with_options(self.ccsd_options.clone());
        engine.integral_cache = Some(cache);
        let cc = engine
            .compute_with_spatial_residual(fc, Some(&source))
            .map_err(failure)?;
        if !cc.converged {
            return Err(failure(
                "modified CCSD-F12 amplitude equations did not converge".into(),
            ));
        }
        let correction = coupling.energy_corrections(&cc.t1_amplitudes, &cc.t2_amplitudes)?;
        let coupling_energy = correction.fock_amplitude
            + match self.variant {
                CCSDF12Variant::F12a => 0.,
                CCSDF12Variant::F12b => correction.additional_f12b,
            };
        let (triples_energy, scaled_triples_energy) = if self.compute_triples {
            let denom = mp2.mp2_correlation_energy;
            if !denom.is_finite() || denom.abs() <= 1e-12 {
                return Err(failure(
                    "(T*) scaling is undefined for zero MP2 correlation".into(),
                ));
            }
            let full = cache.full()?;
            let active;
            let integrals = if fc == 0 {
                full.as_ref()
            } else {
                let n = nm - fc;
                let np = n * (n + 1) / 2;
                let pairs: Vec<_> = (0..n).flat_map(|p| (0..=p).map(move |q| (p, q))).collect();
                active = MoEriTensor {
                    n_basis: n,
                    n_pairs: np,
                    data: (0..np * np)
                        .map(|index| {
                            let (p, q) = pairs[index % np];
                            let (r, s) = pairs[index / np];
                            full.get(fc + p, fc + q, fc + r, fc + s)
                        })
                        .collect(),
                };
                &active
            };
            let t = compute_ccsd_t_correction_in_space(&hf.orbital_energies[fc..], &cc, integrals)
                .map_err(failure)?
                .triples_correlation_energy;
            (
                Some(t),
                Some(t * (denom + mp2.f12_correlation_energy) / denom),
            )
        } else {
            (None, None)
        };
        let correlation = cc.correlation_energy
            + fixed_energy
            + coupling_energy
            + mp2.cabs_singles_energy
            + scaled_triples_energy.unwrap_or(0.);
        if !correlation.is_finite() {
            return Err(failure("nonfinite coupled CCSD-F12 energy".into()));
        }
        Ok(CCSDF12Result {
            hf_energy: hf.total_energy,
            cabs_singles_energy: mp2.cabs_singles_energy,
            ccsd_correlation_energy: cc.correlation_energy,
            f12_correction_energy: fixed_energy,
            coupling_energy,
            coupling_components: correction,
            triples_energy,
            scaled_triples_energy,
            total_correlation_energy: correlation,
            total_energy: hf.total_energy + correlation,
            variant: self.variant,
            hybrid: self.f12_options.hybrid,
            fitted_correction: self.f12_options.fitting_basis.is_some(),
            fitted_fock: self.f12_options.fock_fitting_basis.is_some(),
            fitted_orbital_mp2: self.f12_options.fit_orbital_mp2,
            exact_obs_coulomb: self.f12_options.exact_obs_coulomb,
            mp2_correlation_energy: mp2.mp2_correlation_energy,
            mp2_f12_correlation_energy: mp2.mp2_correlation_energy + mp2.f12_correlation_energy,
            fitting_basis_kind: self.f12_options.fitting_basis_kind,
            fock_fitting_basis_kind: self.f12_options.fock_fitting_basis_kind,
            ccsd_result: cc,
        })
    }
}
impl PostHFMethod for RHFCCSDF12<'_> {
    type Output = CCSDF12Result;
    fn compute(&self) -> Result<CCSDF12Result, String> {
        self.compute_checked().map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{Atom, STO_3G_GBS, build_basis, parse_gaussian94};
    use crate::diis::ConvAlgo;
    use crate::scf::run_scf_general;

    #[test]
    fn test_h2_ccsd_f12() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0, 0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, ConvAlgo::Adiissoscf, None)
            .expect("HF failed");

        let ccsd_f12a = RHFCCSDF12::new(&hf, &atoms, &basis, CCSDF12Variant::F12a);
        let res_a = ccsd_f12a.compute().expect("CCSD-F12a failed");
        assert!(res_a.total_energy < res_a.reference_energy());

        let ccsd_f12b = RHFCCSDF12::new(&hf, &atoms, &basis, CCSDF12Variant::F12b);
        let res_b = ccsd_f12b.compute().expect("CCSD-F12b failed");
        assert!(res_b.total_energy < res_b.reference_energy());
    }
}
