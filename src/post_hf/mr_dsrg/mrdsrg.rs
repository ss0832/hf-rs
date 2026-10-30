// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Multireference Driven Similarity Renormalization Group (`MR-DSRG`) Suite:
//! - Second-Order Multireference DSRG Perturbation Theory (`DSRG-MRPT2`)
//! - Density-Fitted / Cholesky 3-Index `THREE-DSRG-MRPT2`
//! - Third-Order Multireference DSRG Perturbation Theory (`DSRG-MRPT3`)
//! - Iterative Linearized Multireference DSRG (`MR-LDSRG(2)` and `MR-LDSRG(2)-QC`)
//! - Multi-State DSRG (`SA-DSRG`, `MS-DSRG-PT2`, `XMS-DSRG-PT2`, `DWMS-DSRG-PT2`)

use crate::matrix::Matrix;
use crate::post_hf::mr_dsrg::cumulants::ThreePDC;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::scf::HFResult;

/// Zeroth-order Hamiltonian $\hat{H}^{(0)}$ partitioning for `DSRG-MRPT2` and `DSRG-MRPT3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZerothOrderH {
    /// Diagonal generalized Fock operator in the semi-canonical basis (`Fdiag`).
    Fdiag,
    /// Full block-diagonal generalized Fock operator (`Ffull`: core-core, active-active, virtual-virtual).
    Ffull,
    /// Dyall-type Hamiltonian: diagonal core/virtual Fock + full two-body active Hamiltonian (`FdiagVactv`).
    FdiagVactv,
    /// Fink-type retaining diagonal two-body blocks (`FdiagVdiag`).
    FdiagVdiag,
}

/// Reference relaxation protocol for Multireference DSRG (`DSRG-MRPT2`, `DSRG-MRPT3`, `MR-LDSRG(2)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelaxRef {
    /// Unrelaxed reference wavefunction $|\Psi_0\rangle$.
    None,
    /// Single-step diagonalization of the dressed active-space Hamiltonian $\bar{H}_{\text{act}}(s)$ (`Once`).
    Once,
    /// Two-step reference relaxation (`Twice`: solve DSRG -> relax $|\Psi_0\rangle$ -> re-solve DSRG -> relax).
    Twice,
    /// Self-consistent Tick-Tock macro-iteration until reference and DSRG amplitudes converge (`Iterate`).
    Iterate,
}

/// Options for Multireference Driven Similarity Renormalization Group (`MR-DSRG`) methods.
#[derive(Clone, Debug)]
pub struct MRDSRGOptions {
    pub n_active_electrons: usize,
    pub n_active_orbitals: usize,
    /// Flow parameter $s$ in $E_h^{-2}$ (default `0.5`).
    pub flow_param_s: f64,
    /// Choice of zeroth-order Hamiltonian $\hat{H}^{(0)}$.
    pub zeroth_order_h: ZerothOrderH,
    /// Reference relaxation strategy.
    pub relax_ref: RelaxRef,
    /// Compatibility selector: MK uses exact CI cumulants in the verified
    /// single-state adapters; approximate variants are not connected there.
    pub three_pdc: ThreePDC,
    /// Non-Interactive Virtual Orbital (`NIVO`) filtering: neglect 2-body tensors with $\ge 3$ virtual indices.
    pub nivo: bool,
    /// Sequential unitary transformation $\bar{H}(s) = e^{-\hat{A}_2} e^{-\hat{A}_1} \hat{H} e^{\hat{A}_1} e^{\hat{A}_2}$.
    pub sequential_transformation: bool,
    /// Maximum macro-iterations for `MR-LDSRG(2)` or `RelaxRef::Iterate`.
    pub max_iter: usize,
    /// Convergence threshold in Hartree.
    pub tol: f64,
}

impl MRDSRGOptions {
    pub fn new(n_active_electrons: usize, n_active_orbitals: usize) -> Self {
        Self {
            n_active_electrons,
            n_active_orbitals,
            flow_param_s: 0.5,
            zeroth_order_h: ZerothOrderH::Fdiag,
            relax_ref: RelaxRef::Once,
            three_pdc: ThreePDC::MK,
            nivo: false,
            sequential_transformation: false,
            max_iter: 20,
            tol: 1e-8,
        }
    }
}

/// Result of a Multireference Driven Similarity Renormalization Group (`MR-DSRG`) calculation.
#[derive(Clone, Debug)]
pub struct MRDSRGResult {
    pub method_label: &'static str,
    pub reference_energy: f64,
    pub unrelaxed_dsrg_energy: f64,
    pub relaxed_total_energy: f64,
    pub correlation_energy: f64,
    pub flow_param_s: f64,
    pub lambda2_norm: f64,
    pub lambda3_norm: f64,
    pub macro_iterations: usize,
    pub converged: bool,
    /// Reference density, not a relaxed correlated density. Its orbital frame is
    /// `reference_mo_coefficients` when available.
    pub one_rdm_mo: Matrix,
    pub reference_mo_coefficients: Option<Matrix>,
    scf_energy: f64,
    components: std::collections::BTreeMap<String, f64>,
}

impl PostHFResult for MRDSRGResult {
    fn method_name(&self) -> &'static str {
        self.method_label
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }

    fn total_energy(&self) -> f64 {
        self.relaxed_total_energy
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn to_output(&self) -> Result<crate::output::PostHFOutput, crate::error::HfError> {
        if !self.converged {
            return Err(crate::error::HfError::PostHf {
                method: self.method_label.into(),
                detail: "reference or amplitude iterations did not converge".into(),
            });
        }
        let mut out = crate::output::PostHFOutput::new(
            self.method_label,
            self.scf_energy,
            self.relaxed_total_energy,
        )?;
        out.components = self.components.clone();
        out.components
            .insert("E_CASCI".into(), self.reference_energy);
        out.components
            .insert("E_unrelaxed".into(), self.unrelaxed_dsrg_energy);
        out.components.insert(
            "E_reference_relaxation".into(),
            self.relaxed_total_energy - self.unrelaxed_dsrg_energy,
        );
        out.validate()?;
        Ok(out)
    }
}

// ============================================================================
// 1. DSRG-MRPT2 (Second-Order Multireference Driven SRG Perturbation Theory)
// ============================================================================

/// Second-Order Multireference Driven Similarity Renormalization Group (`DSRG-MRPT2`)
/// (J. Chem. Theory Comput. 11, 2097, 2015).
pub struct DSRGMRPT2<'a> {
    hf: &'a HFResult,
    options: MRDSRGOptions,
}
impl<'a> DSRGMRPT2<'a> {
    pub fn new(hf: &'a HFResult, options: MRDSRGOptions) -> Self {
        Self { hf, options }
    }
}
impl PostHFMethod for DSRGMRPT2<'_> {
    type Output = MRDSRGResult;
    fn compute(&self) -> Result<MRDSRGResult, String> {
        exact_compatibility(self.hf, &self.options, 2)
    }
}

/// Map the supported compatibility options before any active-space allocation.
fn exact_compatibility(
    hf: &HFResult,
    legacy: &MRDSRGOptions,
    order: u8,
) -> Result<MRDSRGResult, String> {
    use crate::{
        context::Reference,
        post_hf::{
            casscf_engine::solve_cas,
            mr_dsrg::{
                density_cumulants::DEFAULT_CUMULANT_MEMORY_BUDGET as BUDGET, gno::GnoAlgebra,
                relaxation::solve_reference,
            },
        },
    };
    let options = compatibility_options(legacy)?;
    let ns = hf.mo_coefficients.cols.checked_mul(2).ok_or("MR spin dimension overflow")?;
    if GnoAlgebra::estimated_workspace_bytes(ns).map_err(|e| e.to_string())? > BUDGET {
        return Err("MR-DSRG operator/density workspace exceeds memory budget".into());
    }
    let cas = solve_cas(Reference::Rhf(hf), &options, None, false).map_err(|e| e.to_string())?;
    let solution = solve_reference(hf, cas, order, &options, BUDGET).map_err(|e| e.to_string())?;
    let label = match order {
        2 => "DSRG-MRPT2",
        3 => "DSRG-MRPT3",
        5 => "MR-LDSRG(2)+QC",
        _ => "MR-LDSRG(2)",
    };
    compatibility_result(hf, legacy, solution, order, label)
}

fn compatibility_options(legacy: &MRDSRGOptions) -> Result<crate::options::PostHFOptions, String> {
    use crate::options::{ActiveSpace, PostHFOptions, ReferenceRelaxation};
    if legacy.zeroth_order_h != ZerothOrderH::Fdiag
        || legacy.three_pdc != ThreePDC::MK
        || legacy.nivo
        || legacy.sequential_transformation
        || !legacy.n_active_electrons.is_multiple_of(2)
    {
        return Err("MR-DSRG compatibility adapter supports Fdiag, exact CI cumulants and one spin-balanced singlet; alternative partitions, cumulant approximations, NIVO and sequential transformations are not connected".into());
    }
    let mut options = PostHFOptions { active: Some(ActiveSpace::new(
        legacy.n_active_electrons / 2,
        legacy.n_active_electrons / 2,
        legacy.n_active_orbitals,
    )), max_iter: legacy.max_iter, energy_tol: legacy.tol, residual_tol: legacy.tol, ..Default::default() };
    options.cas.optimize_orbitals = false;
    options.cas.target_s_squared = Some(0.);
    options.dsrg.flow_parameter = legacy.flow_param_s;
    options.dsrg.reference_energy_tol = legacy.tol;
    options.dsrg.reference_ci_tol = legacy.tol;
    options.dsrg.max_reference_iterations = legacy.max_iter;
    options.dsrg.reference_relaxation = match legacy.relax_ref {
        RelaxRef::None => ReferenceRelaxation::None,
        RelaxRef::Once => ReferenceRelaxation::Once,
        RelaxRef::Twice => ReferenceRelaxation::Twice,
        RelaxRef::Iterate => ReferenceRelaxation::Iterate,
    };
    options.validate().map_err(|e| e.to_string())?;
    Ok(options)
}

fn compatibility_result(
    hf: &HFResult, legacy: &MRDSRGOptions,
    solution: super::relaxation::MrReferenceSolution, order: u8, label: &'static str,
) -> Result<MRDSRGResult, String> {
    use crate::{fci::engine::FciHamiltonian, mo_integrals::MoEriTensor,
        post_hf::mr_dsrg::typed_mr::reference_output};
    let na = legacy.n_active_orbitals;
    let populations = (legacy.n_active_electrons / 2, legacy.n_active_electrons / 2);
    let nc = hf.n_occ.checked_sub(populations.0).ok_or("invalid active electron count")?;
    let output = reference_output(&solution, label, order, hf.total_energy, false)
        .map_err(|e| e.to_string())?;
    let h = Matrix::zeros(na, na);
    let np = na * (na + 1) / 2;
    let g = MoEriTensor {
        n_basis: na,
        n_pairs: np,
        data: vec![0.; np * np],
    };
    let spin =
        FciHamiltonian::new(&h, &g, populations.0, populations.1, 0.).map_err(|e| e.to_string())?;
    let active_density = spin
        .rdm(&solution.ci_coefficients, 1)
        .map_err(|e| e.to_string())?;
    let n = solution.mo_coefficients.cols;
    let mut reference_density = Matrix::zeros(n, n);
    for i in 0..nc {
        reference_density[(i, i)] = 2.;
    }
    for p in 0..na {
        for q in 0..na {
            reference_density[(nc + p, nc + q)] = active_density[p * na + q];
        }
    }
    Ok(MRDSRGResult {
        method_label: label,
        reference_energy: solution.initial_reference_energy,
        unrelaxed_dsrg_energy: solution.unrelaxed_energy,
        relaxed_total_energy: solution.total_energy,
        correlation_energy: solution.total_energy - solution.initial_reference_energy,
        flow_param_s: legacy.flow_param_s,
        lambda2_norm: solution.lambda2_norm,
        lambda3_norm: solution.lambda3_norm,
        macro_iterations: solution.reference_iterations,
        converged: true,
        one_rdm_mo: reference_density,
        reference_mo_coefficients: Some(solution.mo_coefficients),
        scf_energy: hf.total_energy,
        components: output.components,
    })
}

// ============================================================================
// 2. THREE-DSRG-MRPT2 (Cholesky / Density-Fitted 3-Index DSRG-MRPT2)
// ============================================================================

/// Density-Fitted / Cholesky-Decomposed 3-Index `THREE-DSRG-MRPT2`
///
/// Factorizes the MO electron repulsion tensor $(pq|rs) = \sum_{P=1}^{N_{\text{aux}}} B_{pq}^P B_{rs}^P$
/// via streamed pivoted Cholesky columns up to threshold $\tau_{\text{Chol}}$.
pub struct ThreeDSRGMRPT2<'a> {
    hf: &'a HFResult,
    options: MRDSRGOptions,
    cholesky_tol: f64,
}

impl<'a> ThreeDSRGMRPT2<'a> {
    pub fn new(hf: &'a HFResult, options: MRDSRGOptions, cholesky_tol: f64) -> Self {
        Self {
            hf,
            options,
            cholesky_tol,
        }
    }
}

impl<'a> PostHFMethod for ThreeDSRGMRPT2<'a> {
    type Output = MRDSRGResult;

    fn compute(&self) -> Result<Self::Output, String> {
        use super::{density_cumulants::DEFAULT_CUMULANT_MEMORY_BUDGET as BUDGET,
            three_reference::{cholesky_factors, solve_three_reference}};
        let options = compatibility_options(&self.options)?;
        let (factors, residual) = cholesky_factors(self.hf, self.cholesky_tol, BUDGET)
            .map_err(|e| e.to_string())?;
        let (solution, sectors, peak) = solve_three_reference(self.hf, &factors, &options, None, BUDGET)
            .map_err(|e| e.to_string())?;
        let mut result = compatibility_result(self.hf, &self.options, solution, 2, "THREE-DSRG-MRPT2")?;
        result.components.insert("cholesky_rank".into(), factors.rank() as f64);
        result.components.insert("cholesky_max_residual".into(), residual);
        result.components.insert("factorized_sectors".into(), sectors as f64);
        result.components.insert("peak_sector_terms".into(), peak as f64);
        Ok(result)
    }
}

// ============================================================================
// 3. DSRG-MRPT3 (Third-Order Multireference Driven SRG Perturbation Theory)
// ============================================================================

/// Third-Order Multireference Driven Similarity Renormalization Group (`DSRG-MRPT3`)
/// (J. Chem. Phys. 146, 124132, 2017).
pub struct DSRGMRPT3<'a> {
    hf: &'a HFResult,
    options: MRDSRGOptions,
}

impl<'a> DSRGMRPT3<'a> {
    pub fn new(hf: &'a HFResult, options: MRDSRGOptions) -> Self {
        Self { hf, options }
    }
}

impl PostHFMethod for DSRGMRPT3<'_> {
    type Output = MRDSRGResult;
    fn compute(&self) -> Result<MRDSRGResult, String> {
        exact_compatibility(self.hf, &self.options, 3)
    }
}

// ============================================================================
// 4. MR-LDSRG(2) and MR-LDSRG(2)-QC (Iterative Linearized Multireference DSRG)
// ============================================================================

/// Iterative Linearized Multireference Driven Similarity Renormalization Group
/// (`MR-LDSRG(2)` and quadratic commutator `MR-LDSRG(2)-QC`, J. Chem. Phys. 144, 164114, 2016).
pub struct MRLDSRG2<'a> {
    hf: &'a HFResult,
    options: MRDSRGOptions,
    quadratic_commutator_only: bool,
}

impl<'a> MRLDSRG2<'a> {
    pub fn new(hf: &'a HFResult, options: MRDSRGOptions) -> Self {
        Self {
            hf,
            options,
            quadratic_commutator_only: false,
        }
    }

    /// Creates an `MR-LDSRG(2)-QC` solver that terminates the BCH expansion at the quadratic
    /// nested commutator $\frac{1}{2}[[\hat{H}, \hat{A}]_{1,2}, \hat{A}]_{1,2}$.
    pub fn new_qc(hf: &'a HFResult, options: MRDSRGOptions) -> Self {
        Self {
            hf,
            options,
            quadratic_commutator_only: true,
        }
    }
}

impl PostHFMethod for MRLDSRG2<'_> {
    type Output = MRDSRGResult;
    fn compute(&self) -> Result<MRDSRGResult, String> {
        exact_compatibility(
            self.hf,
            &self.options,
            if self.quadratic_commutator_only { 5 } else { 4 },
        )
    }
}

// ============================================================================
// 5. Multi-State Driven Similarity Renormalization Group (SA / MS / XMS / DWMS)
// ============================================================================

/// Multi-state formalism for excited states and avoided crossings in `MR-DSRG`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MultiStateDSRGVariant {
    /// State-Averaged DSRG (`SA-DSRG`): single unitary transformation built from state-averaged cumulants.
    StateAverage,
    /// One common ensemble transformation, projected onto the selected CAS states.
    StateAverageContracted,
    /// Multi-State DSRG-PT2 (`MS-DSRG-PT2`): state-specific DSRG transformations + off-diagonal $H_{\alpha\beta}^{\text{eff}}$.
    MS,
    /// Extended Multi-State DSRG-PT2 (`XMS-DSRG-PT2`): rotates model states to diagonalize $\bar{f}^{\text{SA}}$ prior to DSRG.
    XMS,
    /// Dynamically Weighted Multi-State DSRG-PT2 (`DWMS-DSRG-PT2`) with Gaussian weight width $\zeta$ ($E_h^{-2}$).
    DWMS { zeta: f64 },
}

/// Result of a Multi-State DSRG (`SA-DSRG`, `MS-DSRG-PT2`, `XMS-DSRG-PT2`, `DWMS-DSRG-PT2`) calculation.
#[derive(Clone, Debug)]
pub struct MultiStateDSRGResult {
    pub variant: MultiStateDSRGVariant,
    pub state_energies: Vec<f64>,
    pub casci_energies: Vec<f64>,
    pub effective_hamiltonian: Matrix,
    pub flow_param_s: f64,
    /// Final correlated CI coordinates, in the exported MO frame.
    pub mo_coefficients: Matrix,
    pub ci_coefficients: Vec<Vec<f64>>,
    pub model_rotation: Matrix,
    pub ensemble_weights: Matrix,
    pub(crate) output: crate::output::PostHFOutput,
}

impl PostHFResult for MultiStateDSRGResult {
    fn method_name(&self) -> &'static str { super::multistate::method_name(self.variant) }
    fn reference_energy(&self) -> f64 { self.casci_energies[self.output.root] }
    fn correlation_energy(&self) -> f64 { self.output.e_total - self.reference_energy() }
    fn total_energy(&self) -> f64 { self.output.e_total }
    fn to_output(&self) -> Result<crate::output::PostHFOutput, crate::error::HfError> {
        self.output.validate()?;
        Ok(self.output.clone())
    }
}

/// Multi-State Driven Similarity Renormalization Group (`SA-DSRG`, `MS-DSRG-PT2`, `XMS-DSRG-PT2`, `DWMS-DSRG-PT2`) solver.
pub struct MultiStateDSRG<'a> {
    hf: &'a HFResult,
    options: MRDSRGOptions,
    n_states: usize,
    variant: MultiStateDSRGVariant,
}

impl<'a> MultiStateDSRG<'a> {
    pub fn new(
        hf: &'a HFResult,
        options: MRDSRGOptions,
        n_states: usize,
        variant: MultiStateDSRGVariant,
    ) -> Self {
        Self {
            hf,
            options,
            n_states,
            variant,
        }
    }
}

impl PostHFMethod for MultiStateDSRG<'_> {
    type Output = MultiStateDSRGResult;
    fn compute(&self) -> Result<Self::Output, String> {
        use super::{density_cumulants::DEFAULT_CUMULANT_MEMORY_BUDGET as BUDGET,
            three_reference::cholesky_factors, multistate::solve_multistate};
        if !matches!(self.options.relax_ref, RelaxRef::None | RelaxRef::Once) {
            return Err("Multistate DSRG defines one effective-Hamiltonian diagonalization; iterative state-reference updates are unsupported".into());
        }
        let mut options = compatibility_options(&self.options)?;
        options.n_roots = self.n_states;
        options.dsrg.reference_relaxation = crate::options::ReferenceRelaxation::None;
        options.validate().map_err(|e| e.to_string())?;
        let (factors, residual) = cholesky_factors(self.hf, 1e-12, BUDGET).map_err(|e| e.to_string())?;
        let mut result = solve_multistate(self.hf, &factors, &options, self.variant, None).map_err(|e| e.to_string())?;
        result.output.components.insert("cholesky_max_residual".into(), residual);
        Ok(result)
    }
}
