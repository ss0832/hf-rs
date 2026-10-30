// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! RHF MP2-F12/3C(FIX), with strong-orthogonality Ansatz 3, complete CABS
//! Fock coupling, Gaussian-product/kinetic intermediates and fixed cusp amplitudes.
use crate::basis::{Atom, ContractedGaussian};
use crate::context::{AuxBasis, MoIntegralCache, PostHFContext};
use crate::error::HfError;
use crate::options::PostHFOptions;
pub use crate::options::{F12Approximation, F12Hybrid};
use crate::post_hf::f12::cabs::{
    CABSBasis, compute_cabs_singles_with_fock, compute_cabs_singles_with_mixed_fock, extended_fock,
    extended_fock_fitted,
};
use crate::post_hf::f12::geminal::GeminalCorrelationFactor;
use crate::post_hf::f12::intermediates::F12PairIntermediates;
use crate::post_hf::f12::molecular::{MolecularF12Options, molecular_intermediates_and_coupling};
use crate::post_hf::f12::operators::GaussianGeminal;
use crate::post_hf::mp2::RHFMP2;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::{ContextMethod, MethodId, MethodMeta};
use crate::scf::HFResult;

#[derive(Clone, Debug)]
pub struct MP2F12Options {
    pub gamma: f64,
    pub approximation: F12Approximation,
    pub hybrid: F12Hybrid,
    pub cabs_singles: bool,
    /// Include frozen occupied orbitals in the CABS singles correction.
    pub cabs_core_singles: bool,
    pub custom_cabs: Option<CABSBasis>,
    /// Fit F12 correction kernels; conventional MP2 is controlled separately.
    pub fitting_basis: Option<AuxBasis>,
    pub fitting_basis_kind: crate::ao::BasisKind,
    /// Fit the raw extended Fock, then restore its exact orthogonal OBS block.
    pub fock_fitting_basis: Option<AuxBasis>,
    pub fock_fitting_basis_kind: crate::ao::BasisKind,
    /// Use the correction fitting basis for conventional MP2 and its Schur G.
    /// Conventional CCSD/(T) retain exact orbital integrals.
    pub fit_orbital_mp2: bool,
    /// Use exact Coulomb projector integrals with both resolution indices in
    /// the orthogonal OBS. Raw CABS projection retains the consistent DF model.
    /// This does not select the conventional MP2/Schur G approximation.
    pub exact_obs_coulomb: bool,
    /// An explicit Gaussian factor replaces the scaled Slater fit when supplied.
    pub custom_geminal: Option<GaussianGeminal>,
    pub frozen_core: usize,
    /// Limit for the live mixed-tensor workspace estimate of the pair intermediates.
    pub workspace_budget_bytes: usize,
}
impl Default for MP2F12Options {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            approximation: F12Approximation::ApproximationC,
            hybrid: F12Hybrid::Full,
            cabs_singles: true,
            cabs_core_singles: true,
            custom_cabs: None,
            fitting_basis: None,
            fitting_basis_kind: crate::ao::BasisKind::Cartesian,
            fock_fitting_basis: None,
            fock_fitting_basis_kind: crate::ao::BasisKind::Cartesian,
            fit_orbital_mp2: false,
            exact_obs_coulomb: false,
            custom_geminal: None,
            frozen_core: 0,
            workspace_budget_bytes: crate::post_hf::f12::molecular::DEFAULT_F12_WORKSPACE_BYTES,
        }
    }
}
#[derive(Clone, Debug)]
pub struct MP2F12Result {
    pub approximation: F12Approximation,
    pub hybrid: F12Hybrid,
    /// Whether the F12 correction kernels used robust Coulomb-metric fitting.
    pub fitted_correction: bool,
    pub fitted_fock: bool,
    pub fitted_orbital_mp2: bool,
    pub exact_obs_coulomb: bool,
    pub fitting_basis_kind: crate::ao::BasisKind,
    pub fock_fitting_basis_kind: crate::ao::BasisKind,
    pub hf_energy: f64,
    pub cabs_singles_energy: f64,
    pub mp2_correlation_energy: f64,
    pub f12_correlation_energy: f64,
    pub f12_singlet_energy: f64,
    pub f12_triplet_energy: f64,
    /// Relative to the original SCF energy, including CABS singles.
    pub total_correlation_energy: f64,
    pub total_energy: f64,
    pub gamma: f64,
    pub pairs: Vec<F12PairIntermediates>,
}
impl PostHFResult for MP2F12Result {
    fn method_name(&self) -> &'static str {
        match (self.approximation, self.hybrid) {
            (F12Approximation::ApproximationA, _) => "MP2-F12/3*A(FIX)",
            (F12Approximation::ApproximationB, _) => "MP2-F12/3B",
            (F12Approximation::ApproximationC, F12Hybrid::Full) => "MP2-F12/3C(FIX)",
            (F12Approximation::ApproximationC, F12Hybrid::Hy1) => "MP2-F12/3C(FIX,HY1)",
            (F12Approximation::ApproximationC, F12Hybrid::Hy2) => "MP2-F12/3C(FIX,HY2)",
        }
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
        let mut out = crate::output::PostHFOutput::new(
            self.method_name(),
            self.hf_energy,
            self.total_energy,
        )?;
        for (key, value) in [
            ("E_MP2", self.mp2_correlation_energy),
            ("E_F12", self.f12_correlation_energy),
            ("E_CABS_singles", self.cabs_singles_energy),
            ("E_F12_singlet", self.f12_singlet_energy),
            ("E_F12_triplet", self.f12_triplet_energy),
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
            out.components.insert(key.into(), value);
        }
        Ok(out)
    }
}
#[derive(Clone, Debug)]
pub struct RHFMP2F12<'a> {
    pub hf_res: &'a HFResult,
    pub atoms: &'a [Atom],
    pub basis: &'a [ContractedGaussian],
    pub options: MP2F12Options,
    pub integral_cache: Option<&'a MoIntegralCache<'a>>,
}
impl<'a> RHFMP2F12<'a> {
    pub fn new(hf_res: &'a HFResult, atoms: &'a [Atom], basis: &'a [ContractedGaussian]) -> Self {
        Self {
            hf_res,
            atoms,
            basis,
            options: MP2F12Options::default(),
            integral_cache: None,
        }
    }
    pub fn with_options(mut self, options: MP2F12Options) -> Self {
        self.options = options;
        self
    }
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.options.gamma = gamma;
        self
    }
    pub fn with_approximation(mut self, approximation: F12Approximation) -> Self {
        self.options.approximation = approximation;
        self
    }
    pub fn with_frozen_core(mut self, frozen_core: usize) -> Self {
        self.options.frozen_core = frozen_core;
        self
    }
    pub fn compute_checked(&self) -> Result<MP2F12Result, HfError> {
        Ok(self.compute_prepared(false)?.0)
    }
    pub(crate) fn compute_prepared(
        &self,
        ccsd_coupling: bool,
    ) -> Result<(MP2F12Result, Option<Vec<crate::Matrix>>), HfError> {
        if self.options.hybrid != F12Hybrid::Full
            && self.options.approximation != F12Approximation::ApproximationC
        {
            return Err(HfError::Unsupported {
                method: "MP2-F12".into(),
                detail: "HY1/HY2 are defined here only for 3C(FIX)".into(),
            });
        }
        if self.options.approximation == F12Approximation::ApproximationB {
            return Err(HfError::Unsupported {
                method: "MP2-F12".into(),
                detail: "3B requires its own commutator/exchange intermediates".into(),
            });
        }
        if (self.options.fitting_basis.is_some() || self.options.fock_fitting_basis.is_some())
            && self.options.approximation != F12Approximation::ApproximationC
        {
            return Err(HfError::Unsupported {
                method: "fitted MP2-F12 correction".into(),
                detail: "robust fitting is implemented only for 3C(FIX)".into(),
            });
        }
        if self.options.fit_orbital_mp2 && self.options.fitting_basis.is_none() {
            return Err(HfError::InvalidInput(
                "fitted orbital MP2 requires an explicit F12 fitting basis".into(),
            ));
        }
        if self.options.exact_obs_coulomb && self.options.fitting_basis.is_none() {
            return Err(HfError::InvalidInput(
                "exact OBS Coulomb mixing requires an explicit F12 fitting basis".into(),
            ));
        }
        let slater = GeminalCorrelationFactor::new(self.options.gamma)?;
        let factor = match &self.options.custom_geminal {
            Some(f) => GaussianGeminal::new(f.terms.clone())?,
            None => slater.gaussian()?,
        };
        if self.options.frozen_core > self.hf_res.n_occ {
            return Err(HfError::InvalidInput(
                "F12 frozen core exceeds occupied space".into(),
            ));
        }
        // The CABS projector is required even when the singles energy is disabled.
        let cabs = match &self.options.custom_cabs {
            Some(c) => c.clone(),
            None => CABSBasis::from_builtin_reference(
                self.hf_res,
                self.basis,
                self.atoms,
                "cc-pvdz-f12-optri",
                1e-8,
            )
            .map_err(HfError::Integral)?,
        };
        cabs.validate_reference(self.hf_res, self.basis.len())
            .map_err(HfError::InvalidInput)?;
        super::molecular::check_workspace(
            self.hf_res,
            self.basis,
            &cabs,
            self.options.frozen_core,
            self.options.workspace_budget_bytes,
            self.options.fitting_basis.as_ref(),
            ccsd_coupling,
            self.options.exact_obs_coulomb,
        )?;
        let mut conventional = RHFMP2::new(self.hf_res).with_frozen_core(self.options.frozen_core);
        conventional.integral_cache = self.integral_cache;
        let e_mp2 = if self.options.frozen_core == self.hf_res.n_occ
            || self.hf_res.mo_coefficients.cols == self.hf_res.n_occ
        {
            0.0
        } else if self.options.fit_orbital_mp2 {
            let fitting = self.options.fitting_basis.as_ref().unwrap();
            crate::df::DfTensor::build_in_space(
                self.basis,
                &fitting.functions,
                self.options.fitting_basis_kind,
                1e-10,
            )?
            .transform(&self.hf_res.mo_coefficients)?
            .mp2_energy(
                &self.hf_res.orbital_energies,
                self.hf_res.n_occ,
                self.options.frozen_core,
            )?
        } else {
            conventional
                .compute()
                .map_err(|detail| HfError::PostHf {
                    method: "MP2-F12".into(),
                    detail,
                })?
                .correlation_energy
        };
        let extended = if self.options.cabs_singles || self.options.frozen_core < self.hf_res.n_occ
        {
            Some(if let Some(fitting) = &self.options.fock_fitting_basis {
                extended_fock_fitted(
                    self.hf_res,
                    self.atoms,
                    self.basis,
                    &cabs.aux_functions,
                    &fitting.functions,
                    super::density_fit::F12DensityFitOptions {
                        auxiliary_basis_kind: self.options.fock_fitting_basis_kind,
                        workspace_budget_bytes: self.options.workspace_budget_bytes,
                        ..Default::default()
                    },
                )?
            } else {
                extended_fock(self.hf_res, self.atoms, self.basis, &cabs.aux_functions)
                    .map_err(HfError::Integral)?
            })
        } else {
            None
        };
        let e_cabs = match (self.options.cabs_singles, &extended) {
            (true, Some(fock)) => {
                let singles = if self.options.fock_fitting_basis.is_some() {
                    compute_cabs_singles_with_mixed_fock
                } else {
                    compute_cabs_singles_with_fock
                };
                singles(
                    self.hf_res,
                    &cabs,
                    fock,
                    if self.options.cabs_core_singles {
                        0
                    } else {
                        self.options.frozen_core
                    },
                )
                .map_err(HfError::Integral)?
                .energy
            }
            _ => 0.0,
        };
        let (pairs, projected) = match &extended {
            Some(fock) if self.options.frozen_core < self.hf_res.n_occ => {
                molecular_intermediates_and_coupling(
                    self.hf_res,
                    self.atoms,
                    self.basis,
                    &cabs,
                    &factor,
                    fock,
                    MolecularF12Options {
                        frozen_core: self.options.frozen_core,
                        approximation: self.options.approximation,
                        hybrid: self.options.hybrid,
                        workspace_budget_bytes: self.options.workspace_budget_bytes,
                        fitting_basis: self.options.fitting_basis.as_ref(),
                        fitting_basis_kind: self.options.fitting_basis_kind,
                        fitted_fock: self.options.fock_fitting_basis.is_some(),
                        fit_orbital_mp2: self.options.fit_orbital_mp2,
                        exact_obs_coulomb: self.options.exact_obs_coulomb,
                    },
                    ccsd_coupling,
                )?
            }
            _ => (Vec::new(), None),
        };
        let (mut singlet, mut triplet) = (0.0, 0.0);
        for pair in &pairs {
            let (s, t) = pair.fixed_energy();
            singlet += s;
            triplet += t;
        }
        let f12 = singlet + triplet;
        let correlation = e_mp2 + f12 + e_cabs;
        if !correlation.is_finite() {
            return Err(HfError::Integral("nonfinite F12 correlation energy".into()));
        }
        Ok((
            MP2F12Result {
                approximation: self.options.approximation,
                hybrid: self.options.hybrid,
                fitted_correction: self.options.fitting_basis.is_some(),
                fitted_fock: self.options.fock_fitting_basis.is_some(),
                fitted_orbital_mp2: self.options.fit_orbital_mp2,
                exact_obs_coulomb: self.options.exact_obs_coulomb,
                fitting_basis_kind: self.options.fitting_basis_kind,
                fock_fitting_basis_kind: self.options.fock_fitting_basis_kind,
                hf_energy: self.hf_res.total_energy,
                cabs_singles_energy: e_cabs,
                mp2_correlation_energy: e_mp2,
                f12_correlation_energy: f12,
                f12_singlet_energy: singlet,
                f12_triplet_energy: triplet,
                total_correlation_energy: correlation,
                total_energy: self.hf_res.total_energy + correlation,
                gamma: self.options.gamma,
                pairs,
            },
            projected,
        ))
    }
}
impl PostHFMethod for RHFMP2F12<'_> {
    type Output = MP2F12Result;
    fn compute(&self) -> Result<MP2F12Result, String> {
        self.compute_checked().map_err(|e| e.to_string())
    }
}
impl MethodMeta for RHFMP2F12<'_> {
    const METHOD: MethodId = MethodId::Mp2F12;
}
impl<'a> ContextMethod<'a> for RHFMP2F12<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        if options.active.is_some() || options.root != 0 || options.n_roots != 1 {
            return Err(HfError::InvalidInput(
                "MP2-F12 needs a single RHF reference without a CAS selection".into(),
            ));
        }
        if options.frozen_virtual != 0 || options.density_fitting {
            return Err(HfError::Unsupported {
                method: "MP2-F12".into(),
                detail: "frozen virtuals and fully density-fitted F12 have not been implemented; f12_fitting_basis selects fitted correction kernels with an exact orbital baseline".into(),
            });
        }
        if options.f12.hybrid != F12Hybrid::Full
            && options.f12.approximation != F12Approximation::ApproximationC
        {
            return Err(HfError::Unsupported {
                method: "MP2-F12".into(),
                detail: "HY1/HY2 are defined here only for 3C(FIX)".into(),
            });
        }
        if (options.f12.fitting_basis.is_some() || options.f12.fock_fitting_basis.is_some())
            && options.f12.approximation != F12Approximation::ApproximationC
        {
            return Err(HfError::Unsupported {
                method: "fitted MP2-F12 correction".into(),
                detail: "robust fitting is implemented only for 3C(FIX)".into(),
            });
        }
        let hf = ctx.rhf()?;
        let name = options
            .f12
            .cabs_basis
            .as_deref()
            .unwrap_or("cc-pvdz-f12-optri");
        let cabs = CABSBasis::from_builtin_reference(hf, ctx.basis, ctx.atoms, name, 1e-8)
            .map_err(HfError::Integral)?;
        let fitting_basis = options
            .f12
            .fitting_basis
            .as_deref()
            .map(|name| AuxBasis::from_builtin(ctx.atoms, name))
            .transpose()?;
        let fock_fitting_basis = options
            .f12
            .fock_fitting_basis
            .as_deref()
            .map(|name| AuxBasis::from_builtin(ctx.atoms, name))
            .transpose()?;
        if options.f12.fit_orbital_mp2 && fitting_basis.is_none() {
            return Err(HfError::InvalidInput(
                "fitted orbital MP2 requires an explicit F12 fitting basis".into(),
            ));
        }
        if options.f12.exact_obs_coulomb && fitting_basis.is_none() {
            return Err(HfError::InvalidInput(
                "exact OBS Coulomb mixing requires an explicit F12 fitting basis".into(),
            ));
        }
        Ok(Self {
            hf_res: hf,
            atoms: ctx.atoms,
            basis: ctx.basis,
            integral_cache: Some(&ctx.ints),
            options: MP2F12Options {
                gamma: options.f12.beta,
                approximation: options.f12.approximation,
                hybrid: options.f12.hybrid,
                cabs_singles: options.f12.cabs_singles,
                cabs_core_singles: options.f12.cabs_core_singles,
                custom_cabs: Some(cabs),
                fitting_basis,
                fitting_basis_kind: options.f12.fitting_basis_kind,
                fock_fitting_basis,
                fock_fitting_basis_kind: options.f12.fock_fitting_basis_kind,
                fit_orbital_mp2: options.f12.fit_orbital_mp2,
                exact_obs_coulomb: options.f12.exact_obs_coulomb,
                frozen_core: options.frozen_core,
                workspace_budget_bytes: options.f12.workspace_budget_bytes,
                ..Default::default()
            },
        })
    }
}
