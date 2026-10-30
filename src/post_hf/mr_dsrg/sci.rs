// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Selected and Projector Configuration Interaction Solvers:
//! - Adaptive Configuration Interaction (`ACI`, `ACI+PT2`, `SA-ACI`)
//! - Adaptive Sampling Configuration Interaction (`ASCI`, `ASCI+PT2`)
//! - Heat-Bath Configuration Interaction (`HBCI`, `HBCI+PT2`)
//! - Projector Configuration Interaction (`PCI`) with 10 polynomial / Krylov generators

use crate::basis::{Atom, ContractedGaussian};
use crate::fci::slater_condon::slater_condon_element;
use crate::fci::string::SpinString;
use crate::matrix::Matrix;
use crate::mo_integrals::{MoEriTensor, transform_1e_ao_to_mo, transform_eri_ao_to_mo};
use crate::post_hf::grad_common::compute_rhf_total_rdm_analytical_gradient;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::scf::HFResult;
use std::collections::HashSet;

/// Generates all single and double excitations out of a set of reference determinants.
pub fn generate_sd_connected_space(
    n_basis: usize,
    n_alpha: usize,
    n_beta: usize,
    model_dets: &[(SpinString, SpinString)],
) -> Result<Vec<(SpinString, SpinString)>, String> {
    let model_set: HashSet<_> = model_dets.iter().copied().collect();
    let mut connected =
        crate::fci::excitations::singles_doubles(n_basis, n_alpha, n_beta, model_dets, 2_000_000)
            .map_err(|e| e.to_string())?;
    connected.retain(|d| !model_set.contains(d));
    Ok(connected)
}

/// Diagonalizes the Hamiltonian in a determinant subspace and returns `(evals, evecs)`
/// where `evals` includes `nuclear_repulsion`.
pub fn diagonalize_subspace(
    dets: &[(SpinString, SpinString)],
    h_mo: &Matrix,
    mo_eri: &MoEriTensor,
    nuclear_repulsion: f64,
) -> Result<(Vec<f64>, Matrix), String> {
    let n_det = dets.len();
    if n_det == 0
        || n_det
            .checked_mul(n_det)
            .and_then(|n| n.checked_mul(8))
            .is_none_or(|n| n > 128 * 1024 * 1024)
    {
        return Err("selected-CI dense subspace exceeds the 128 MiB limit or is empty".into());
    }
    let mut h_mat = Matrix::zeros(n_det, n_det);
    for i in 0..n_det {
        let (sa_i, sb_i) = &dets[i];
        for j in 0..=i {
            let (sa_j, sb_j) = &dets[j];
            let mut val = slater_condon_element(sa_i, sb_i, sa_j, sb_j, h_mo, mo_eri);
            if i == j {
                val += nuclear_repulsion;
            }
            h_mat[(i, j)] = val;
            h_mat[(j, i)] = val;
        }
    }
    Ok(h_mat.eigh())
}

// ============================================================================
// 1. Adaptive Configuration Interaction (ACI / ACI+PT2 / SA-ACI)
// ============================================================================

/// Options for Adaptive Configuration Interaction (`ACI`, `ACI+PT2`, `SA-ACI`).
#[derive(Clone, Debug)]
pub struct ACIOptions {
    /// Energy importance threshold $\sigma$ in Hartree (default `1e-3`).
    pub sigma: f64,
    /// Coarse-graining squared-coefficient tail parameter (default `0.5` / Hartree).
    pub gamma: f64,
    /// Number of roots for State-Averaged ACI (`SA-ACI`, default `1`).
    pub n_roots: usize,
    /// Whether to include the discarded two-state energy-importance sum (ACI Eq. 9).
    pub compute_pt2: bool,
    /// Maximum number of iterative model-space refinement cycles.
    pub max_cycles: usize,
}

impl Default for ACIOptions {
    fn default() -> Self {
        Self {
            sigma: 1e-3,
            gamma: 0.5,
            n_roots: 1,
            compute_pt2: true,
            max_cycles: 100,
        }
    }
}

/// Result of an Adaptive Configuration Interaction (`ACI`, `ACI+PT2`, `SA-ACI`) calculation.
#[derive(Clone, Debug)]
pub struct ACIResult {
    pub reference_energy: f64,
    pub variational_energy: f64,
    pub pt2_correction: f64,
    pub total_energy: f64,
    pub n_determinants: usize,
    pub root_energies: Vec<f64>,
    pub determinants: Vec<(SpinString, SpinString)>,
    pub coefficients: Vec<f64>,
    pub one_rdm_mo: Matrix,
    pub two_rdm_mo: Vec<f64>,
}

impl PostHFResult for ACIResult {
    fn method_name(&self) -> &'static str {
        "ACI+PT2"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.total_energy - self.reference_energy
    }

    fn one_rdm(&self) -> Option<&Matrix> {
        Some(&self.one_rdm_mo)
    }
}

/// Adaptive Configuration Interaction (`ACI`, `ACI+PT2`, `SA-ACI`) solver
/// (J. Chem. Phys. 144, 161106, 2016; 148, 074114, 2018).
pub struct ACI<'a> {
    hf: &'a HFResult,
    options: ACIOptions,
}

impl<'a> ACI<'a> {
    pub fn new(hf: &'a HFResult, options: ACIOptions) -> Self {
        Self { hf, options }
    }
}

impl<'a> PostHFMethod for ACI<'a> {
    type Output = ACIResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let mut options = crate::options::PostHFOptions::default();
        options.n_roots = self.options.n_roots;
        options.sci.selection_threshold = self.options.sigma;
        options.sci.coarse_graining = self.options.gamma;
        options.max_iter = self.options.max_cycles;
        options.sci.compute_pt2 = self.options.compute_pt2;
        let result = super::selected_ci::compute_selected(
            crate::context::Reference::Rhf(self.hf),
            &options,
            None,
            super::selected_ci::SelectionKind::Aci,
        )
        .map_err(|e| e.to_string())?;
        let two_rdm_mo = result.active_two_rdm(0).map_err(|e| e.to_string())?;
        Ok(ACIResult {
            reference_energy: self.hf.total_energy,
            variational_energy: result.states[0].energy,
            pt2_correction: result.corrections[0],
            total_energy: result.output.e_total,
            n_determinants: result.determinants.len(),
            root_energies: result.states.iter().map(|state| state.energy).collect(),
            coefficients: result.states[0].vector.clone(),
            determinants: result.determinants.clone(),
            one_rdm_mo: result.output.one_rdm.ok_or("SCI density is unavailable")?,
            two_rdm_mo,
        })
    }
}

/// Computes the exact analytical nuclear gradient of the variational ACI wavefunction.
pub fn compute_aci_analytical_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    aci: &ACIResult,
) -> Vec<[f64; 3]> {
    compute_rhf_total_rdm_analytical_gradient(atoms, basis, hf, &aci.one_rdm_mo, &aci.two_rdm_mo)
}

// ============================================================================
// 2. Adaptive Sampling Configuration Interaction (ASCI / ASCI+PT2)
// ============================================================================

/// Options for Adaptive Sampling Configuration Interaction (`ASCI`, `ASCI+PT2`).
#[derive(Clone, Debug)]
pub struct ASCIOptions {
    /// Maximum number of core determinants $N_c$ used to generate excitations.
    pub n_core_dets: usize,
    /// Maximum number of target variational determinants $N_t$.
    pub n_target_dets: usize,
    /// Whether to compute Epstein-Nesbet PT2 correction.
    pub compute_pt2: bool,
    /// Maximum number of refinement iterations.
    pub max_iter: usize,
}

impl Default for ASCIOptions {
    fn default() -> Self {
        Self {
            n_core_dets: 50,
            n_target_dets: 200,
            compute_pt2: true,
            max_iter: 100,
        }
    }
}

/// Result of an Adaptive Sampling Configuration Interaction (`ASCI`, `ASCI+PT2`) calculation.
#[derive(Clone, Debug)]
pub struct ASCIResult {
    pub reference_energy: f64,
    pub variational_energy: f64,
    pub pt2_correction: f64,
    pub total_energy: f64,
    pub n_determinants: usize,
    pub one_rdm_mo: Matrix,
    pub two_rdm_mo: Vec<f64>,
}

impl PostHFResult for ASCIResult {
    fn method_name(&self) -> &'static str {
        "ASCI+PT2"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.total_energy - self.reference_energy
    }

    fn one_rdm(&self) -> Option<&Matrix> {
        Some(&self.one_rdm_mo)
    }
}

/// Adaptive Sampling Configuration Interaction (`ASCI`, `ASCI+PT2`) solver
/// (Tubman, Lee, Takeshita, Head-Gordon & Whaley, J. Chem. Phys. 145, 044112, 2016).
pub struct ASCI<'a> {
    hf: &'a HFResult,
    options: ASCIOptions,
}

impl<'a> ASCI<'a> {
    pub fn new(hf: &'a HFResult, options: ASCIOptions) -> Self {
        Self { hf, options }
    }
}

impl<'a> PostHFMethod for ASCI<'a> {
    type Output = ASCIResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let mut options = crate::options::PostHFOptions::default();
        options.sci.n_core_determinants = self.options.n_core_dets;
        options.sci.n_target_determinants = self.options.n_target_dets;
        options.max_iter = self.options.max_iter;
        options.sci.compute_pt2 = self.options.compute_pt2;
        let result = super::selected_ci::compute_selected(
            crate::context::Reference::Rhf(self.hf),
            &options,
            None,
            super::selected_ci::SelectionKind::Asci,
        )
        .map_err(|e| e.to_string())?;
        let two_rdm_mo = result.active_two_rdm(0).map_err(|e| e.to_string())?;
        Ok(ASCIResult {
            reference_energy: self.hf.total_energy,
            variational_energy: result.states[0].energy,
            pt2_correction: result.corrections[0],
            total_energy: result.output.e_total,
            n_determinants: result.determinants.len(),
            one_rdm_mo: result.output.one_rdm.ok_or("SCI density is unavailable")?,
            two_rdm_mo,
        })
    }
}

// ============================================================================
// 3. Heat-Bath Configuration Interaction (HBCI / HBCI+PT2)
// ============================================================================

/// Options for Heat-Bath Configuration Interaction (`HBCI`, `HBCI+PT2`).
#[derive(Clone, Debug)]
pub struct HBCIOptions {
    /// Variational Heat-Bath importance threshold $\epsilon_1$ in Hartree (default `1e-3`).
    pub eps1: f64,
    /// Perturbative PT2 screening threshold $\epsilon_2$ in Hartree (default `1e-5`).
    pub eps2: f64,
    /// Whether to compute the screened Epstein-Nesbet PT2 correction.
    pub compute_pt2: bool,
    /// Maximum number of variational growth cycles.
    pub max_cycles: usize,
}

impl Default for HBCIOptions {
    fn default() -> Self {
        Self {
            eps1: 1e-3,
            eps2: 1e-5,
            compute_pt2: true,
            max_cycles: 100,
        }
    }
}

/// Result of a Heat-Bath Configuration Interaction (`HBCI`, `HBCI+PT2`) calculation.
#[derive(Clone, Debug)]
pub struct HBCIResult {
    pub reference_energy: f64,
    pub variational_energy: f64,
    pub pt2_correction: f64,
    pub total_energy: f64,
    pub n_determinants: usize,
    pub one_rdm_mo: Matrix,
    pub two_rdm_mo: Vec<f64>,
}

impl PostHFResult for HBCIResult {
    fn method_name(&self) -> &'static str {
        "HBCI+PT2"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.total_energy - self.reference_energy
    }

    fn one_rdm(&self) -> Option<&Matrix> {
        Some(&self.one_rdm_mo)
    }
}

/// Heat-Bath Configuration Interaction (`HBCI`, `HBCI+PT2`) solver
/// (Holmes, Tubman & Umrigar, J. Chem. Theory Comput. 12, 3674, 2016).
pub struct HBCI<'a> {
    hf: &'a HFResult,
    options: HBCIOptions,
}

impl<'a> HBCI<'a> {
    pub fn new(hf: &'a HFResult, options: HBCIOptions) -> Self {
        Self { hf, options }
    }
}

impl<'a> PostHFMethod for HBCI<'a> {
    type Output = HBCIResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let mut options = crate::options::PostHFOptions::default();
        options.sci.selection_threshold = self.options.eps1;
        options.sci.pt2_threshold = self.options.eps2;
        options.max_iter = self.options.max_cycles;
        options.sci.compute_pt2 = self.options.compute_pt2;
        let result = super::selected_ci::compute_selected(
            crate::context::Reference::Rhf(self.hf),
            &options,
            None,
            super::selected_ci::SelectionKind::HeatBath,
        )
        .map_err(|e| e.to_string())?;
        let two_rdm_mo = result.active_two_rdm(0).map_err(|e| e.to_string())?;
        Ok(HBCIResult {
            reference_energy: self.hf.total_energy,
            variational_energy: result.states[0].energy,
            pt2_correction: result.corrections[0],
            total_energy: result.output.e_total,
            n_determinants: result.determinants.len(),
            one_rdm_mo: result.output.one_rdm.ok_or("SCI density is unavailable")?,
            two_rdm_mo,
        })
    }
}

// ============================================================================
// 4. Projector Configuration Interaction (PCI)
// ============================================================================

pub use crate::options::PCIGenerator;

/// Options for Projector Configuration Interaction (`PCI`).
#[derive(Clone, Debug)]
pub struct PCIOptions {
    pub generator: PCIGenerator,
    /// Deterministic spawning threshold $\tau_{\text{spawn}}$ (default `1e-4`).
    pub spawning_threshold: f64,
    /// Polynomial order $m$ per projection step (default `4`).
    pub polynomial_order: usize,
    /// Maximum number of projection steps.
    pub max_steps: usize,
    /// Whether to add Epstein-Nesbet PT2 correction.
    pub compute_pt2: bool,
}

impl Default for PCIOptions {
    fn default() -> Self {
        Self {
            generator: PCIGenerator::WallChebyshev,
            spawning_threshold: 1e-4,
            polynomial_order: 4,
            max_steps: 2000,
            compute_pt2: true,
        }
    }
}

/// Result of a Projector Configuration Interaction (`PCI`) calculation.
#[derive(Clone, Debug)]
pub struct PCIResult {
    pub reference_energy: f64,
    pub projective_energy: f64,
    pub variational_energy: f64,
    pub pt2_correction: f64,
    pub total_energy: f64,
    pub n_determinants: usize,
    pub generator: PCIGenerator,
    pub one_rdm_mo: Matrix,
    pub two_rdm_mo: Vec<f64>,
}

impl PostHFResult for PCIResult {
    fn method_name(&self) -> &'static str {
        "PCI"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.total_energy - self.reference_energy
    }

    fn one_rdm(&self) -> Option<&Matrix> {
        Some(&self.one_rdm_mo)
    }
}

/// Projector Configuration Interaction (`PCI`) solver
/// (J. Chem. Theory Comput. 12, 4326, 2016).
pub struct PCI<'a> {
    hf: &'a HFResult,
    options: PCIOptions,
}

impl<'a> PCI<'a> {
    pub fn new(hf: &'a HFResult, options: PCIOptions) -> Self {
        Self { hf, options }
    }
}

impl<'a> PostHFMethod for PCI<'a> {
    type Output = PCIResult;
    fn compute(&self) -> Result<Self::Output, String> {
        if !self.hf.converged {
            return Err("PCI reference did not converge".into());
        }
        let h = transform_1e_ao_to_mo(&self.hf.mo_coefficients, &self.hf.core_hamiltonian);
        let g = transform_eri_ao_to_mo(&self.hf.mo_coefficients, &self.hf.eri);
        let mut options = crate::options::PostHFOptions {
            max_iter: self.options.max_steps,
            ..Default::default()
        };
        options.residual_tol = 1e-9;
        let result = super::projector_ci::solve_projector(
            &h,
            &g,
            self.hf.nuclear_repulsion_energy,
            self.hf.n_occ,
            self.hf.n_occ,
            &self.options,
            &options,
        )
        .map_err(|e| e.to_string())?;
        let model = result.model;
        let engine = crate::fci::subspace::SubspaceHamiltonian::new(
            &h,
            &g,
            self.hf.nuclear_repulsion_energy,
            model.determinants.clone(),
        )
        .map_err(|e| e.to_string())?;
        let state = &model.states[0];
        let one_rdm_mo = Matrix::from_vec(
            h.rows,
            h.rows,
            engine.rdm(&state.vector, 1).map_err(|e| e.to_string())?,
        );
        let two_rdm_mo = engine.rdm(&state.vector, 2).map_err(|e| e.to_string())?;
        Ok(PCIResult {
            reference_energy: self.hf.total_energy,
            projective_energy: result.projective_energy,
            variational_energy: state.energy,
            pt2_correction: model.corrections[0],
            total_energy: state.energy + model.corrections[0],
            n_determinants: model.determinants.len(),
            generator: self.options.generator,
            one_rdm_mo,
            two_rdm_mo,
        })
    }
}
