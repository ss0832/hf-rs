// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Validated electronic-structure options. Method names are resolved by the registry;
//! this module parses option values without a second method dispatch table.
use crate::error::HfError;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ActiveSpace {
    pub n_alpha: usize,
    pub n_beta: usize,
    pub n_orb: usize,
    pub orbitals: Option<Vec<usize>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedActiveSpace {
    pub n_alpha: usize,
    pub n_beta: usize,
    pub inactive: Vec<usize>,
    pub active: Vec<usize>,
    pub external: Vec<usize>,
}
impl ResolvedActiveSpace {
    pub fn permutation(&self) -> Vec<usize> {
        self.inactive
            .iter()
            .chain(&self.active)
            .chain(&self.external)
            .copied()
            .collect()
    }
}
impl ActiveSpace {
    pub fn new(n_alpha: usize, n_beta: usize, n_orb: usize) -> Self {
        Self {
            n_alpha,
            n_beta,
            n_orb,
            orbitals: None,
        }
    }
    pub fn resolve(
        &self,
        n_mo: usize,
        total_alpha: usize,
        total_beta: usize,
    ) -> Result<ResolvedActiveSpace, HfError> {
        let invalid = |text: &str| HfError::InvalidInput(format!("active space: {text}"));
        if total_alpha > n_mo
            || total_beta > n_mo
            || self.n_alpha > self.n_orb
            || self.n_beta > self.n_orb
            || self.n_alpha > total_alpha
            || self.n_beta > total_beta
        {
            return Err(invalid("electron count exceeds orbital capacity"));
        }
        let n_inactive = total_alpha - self.n_alpha;
        if total_beta - self.n_beta != n_inactive {
            return Err(invalid("inactive orbitals must be doubly occupied"));
        }
        if n_inactive + self.n_orb > n_mo {
            return Err(invalid("inactive plus active space exceeds available MOs"));
        }
        let active = self
            .orbitals
            .clone()
            .unwrap_or_else(|| (n_inactive..n_inactive + self.n_orb).collect());
        let selected = active.iter().copied().collect::<HashSet<_>>();
        if active.len() != self.n_orb
            || selected.len() != self.n_orb
            || active.iter().any(|&i| i >= n_mo)
        {
            return Err(invalid(
                "orbital list must contain n_orb distinct valid MO indices",
            ));
        }
        let complement = (0..n_mo)
            .filter(|i| !selected.contains(i))
            .collect::<Vec<_>>();
        Ok(ResolvedActiveSpace {
            n_alpha: self.n_alpha,
            n_beta: self.n_beta,
            inactive: complement[..n_inactive].to_vec(),
            active,
            external: complement[n_inactive..].to_vec(),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LocalOptions {
    pub t_cut_pno: f64,
    /// Defaults to t_cut_pno when no separate triples threshold is supplied.
    pub t_cut_tno: Option<f64>,
    pub t_cut_pairs: f64,
    pub t_cut_do: f64,
    pub t_cut_mkn: f64,
    pub full_domains: bool,
    pub weak_pairs: bool,
    pub iterative_triples: bool,
}
impl Default for LocalOptions {
    fn default() -> Self {
        Self {
            t_cut_pno: 1e-7,
            t_cut_tno: None,
            t_cut_pairs: 1e-5,
            t_cut_do: 1e-2,
            t_cut_mkn: 1e-3,
            full_domains: false,
            weak_pairs: true,
            iterative_triples: false,
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct DmrgOptions {
    pub bond_dimensions: Vec<usize>,
    pub sweeps: usize,
    pub one_site_sweeps: usize,
    pub noise: f64,
    pub truncation_tol: f64,
}
impl Default for DmrgOptions {
    fn default() -> Self {
        Self {
            bond_dimensions: vec![64, 128, 256],
            sweeps: 12,
            one_site_sweeps: 12,
            noise: 0.0,
            truncation_tol: 1e-10,
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct CasOptions {
    pub caspt2_metric_tol: f64,
    pub max_orbital_iterations: usize,
    pub orbital_gradient_tol: f64,
    pub max_rotation: f64,
    pub optimize_orbitals: bool,
    pub target_s_squared: Option<f64>,
    pub state_weights: Vec<f64>,
    pub ipea_shift: f64,
    pub imaginary_shift: f64,
    pub real_shift: f64,
}
impl Default for CasOptions {
    fn default() -> Self {
        Self {
            max_orbital_iterations: 80,
            caspt2_metric_tol: 1e-10,
            orbital_gradient_tol: 1e-7,
            max_rotation: 0.1,
            optimize_orbitals: true,
            target_s_squared: None,
            state_weights: Vec::new(),
            ipea_shift: 0.25,
            imaginary_shift: 0.0,
            real_shift: 0.0,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImsrgGenerator {
    Wegner,
    White,
}
/// Number of effective-CAS rediagonalizations in a driven MR calculation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReferenceRelaxation {
    #[default]
    None,
    Once,
    Twice,
    Iterate,
}
#[derive(Clone, Debug, PartialEq)]
pub struct DsrgOptions {
    pub flow_parameter: f64,
    /// Maximum residual Coulomb diagonal in the THREE pivoted Cholesky factors.
    pub cholesky_tolerance: f64,
    /// Gaussian energy-separation weight in DWMS, in inverse squared Hartree.
    pub dynamic_weight_parameter: f64,
    /// Compatibility switch: true selects Once unless an explicit mode is set.
    pub relax_reference: bool,
    pub reference_relaxation: ReferenceRelaxation,
    pub max_reference_iterations: usize,
    pub reference_energy_tol: f64,
    pub reference_ci_tol: f64,
    pub ode_step: f64,
    pub ode_tolerance: f64,
    pub ode_generator: ImsrgGenerator,
    pub bch_tolerance: f64,
    pub bch_max_order: usize,
}
impl Default for DsrgOptions {
    fn default() -> Self {
        Self {
            flow_parameter: 0.5,
            cholesky_tolerance: 1e-10,
            dynamic_weight_parameter: 50.0,
            relax_reference: false,
            reference_relaxation: ReferenceRelaxation::None,
            max_reference_iterations: 30,
            reference_energy_tol: 1e-10,
            reference_ci_tol: 1e-8,
            ode_step: 0.01,
            ode_tolerance: 1e-9,
            ode_generator: ImsrgGenerator::Wegner,
            bch_tolerance: 1e-13,
            bch_max_order: 80,
        }
    }
}
impl DsrgOptions {
    pub fn relaxation(&self) -> ReferenceRelaxation {
        if self.reference_relaxation == ReferenceRelaxation::None && self.relax_reference {
            ReferenceRelaxation::Once
        } else {
            self.reference_relaxation
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F12Approximation {
    /// Canonical 3*A(FIX), with GBC/EBC and the exchange commutator neglected in A.
    ApproximationA,
    /// Reserved for the distinct 3B theory; no substitute is reported as 3B.
    ApproximationB,
    ApproximationC,
}
/// Resolution used for the residual nonlocal term in the 3C B matrix.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum F12Hybrid {
    #[default]
    Full,
    /// Omit the complete residual Y term (Knizia 2010, Eq. 4.52).
    Hy1,
    /// Omit only its CABS-CABS right resolution (Eq. 4.53, last term).
    Hy2,
}
#[derive(Clone, Debug, PartialEq)]
pub struct F12Options {
    pub beta: f64,
    pub approximation: F12Approximation,
    pub hybrid: F12Hybrid,
    pub cabs_basis: Option<String>,
    /// Robust fitting of F12 correction kernels, with separate orbital/Fock controls.
    pub fitting_basis: Option<String>,
    pub fitting_basis_kind: crate::ao::BasisKind,
    pub fock_fitting_basis: Option<String>,
    pub fock_fitting_basis_kind: crate::ao::BasisKind,
    pub fit_orbital_mp2: bool,
    /// Restore exact Coulomb integrals only within the orthogonal OBS projector.
    pub exact_obs_coulomb: bool,
    pub cabs_singles: bool,
    /// Include frozen occupied orbitals in CABS singles (the Psi4 convention).
    pub cabs_core_singles: bool,
    pub triples_scaling: bool,
    /// Limit for the live mixed-tensor workspace estimate, in bytes.
    pub workspace_budget_bytes: usize,
}
impl Default for F12Options {
    fn default() -> Self {
        Self {
            beta: 1.0,
            approximation: F12Approximation::ApproximationC,
            hybrid: F12Hybrid::Full,
            cabs_basis: None,
            fitting_basis: None,
            fitting_basis_kind: crate::ao::BasisKind::Cartesian,
            fock_fitting_basis: None,
            fock_fitting_basis_kind: crate::ao::BasisKind::Cartesian,
            fit_orbital_mp2: false,
            exact_obs_coulomb: false,
            cabs_singles: true,
            cabs_core_singles: true,
            triples_scaling: false,
            workspace_budget_bytes: crate::post_hf::f12::molecular::DEFAULT_F12_WORKSPACE_BYTES,
        }
    }
}
/// Polynomial / Krylov generator type for Projector Configuration Interaction (`PCI`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PCIGenerator {
    /// Wall-Chebyshev polynomial projector $\prod_{k=1}^m (\hat{H} - x_k)$.
    WallChebyshev,
    /// Pure Chebyshev polynomial of the negatively scaled Hamiltonian.
    Chebyshev,
    /// Exponential Chebyshev generator.
    ExpChebyshev,
    /// Linear Taylor projector $1 - \Delta\tau (\hat{H} - E_S)$.
    Linear,
    /// Quadratic Taylor projector $1 - \Delta\tau \tilde{H} + \frac{1}{2}\Delta\tau^2 \tilde{H}^2$.
    Quadratic,
    /// Cubic Taylor projector $\sum_{k=0}^3 \frac{(-\Delta\tau \tilde{H})^k}{k!}$.
    Cubic,
    /// Quartic Taylor projector $\sum_{k=0}^4 \frac{(-\Delta\tau \tilde{H})^k}{k!}$.
    Quartic,
    /// Power method projector $(E_{\max} - \hat{H}) / (E_{\max} - E_0)$.
    Power,
    /// Davidson selected-space accelerator (unfiltered Hermitian model).
    Davidson,
    /// Reorthogonalized Lanczos selected-space accelerator.
    Lanczos,
}

impl std::str::FromStr for PCIGenerator {
    type Err = HfError;
    fn from_str(value: &str) -> Result<Self, HfError> {
        match value
            .to_ascii_lowercase()
            .replace(['-', '_', ' '], "")
            .as_str()
        {
            "wallchebyshev" | "wall" => Ok(Self::WallChebyshev),
            "chebyshev" => Ok(Self::Chebyshev),
            "expchebyshev" => Ok(Self::ExpChebyshev),
            "linear" => Ok(Self::Linear),
            "quadratic" => Ok(Self::Quadratic),
            "cubic" => Ok(Self::Cubic),
            "quartic" => Ok(Self::Quartic),
            "power" => Ok(Self::Power),
            "davidson" => Ok(Self::Davidson),
            "lanczos" => Ok(Self::Lanczos),
            _ => Err(HfError::InvalidInput(format!(
                "unknown PCI generator: {value}"
            ))),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SciOptions {
    pub diagonalization_max_iterations: usize,
    pub projector_generator: PCIGenerator,
    pub polynomial_order: usize,
    pub selection_threshold: f64,
    pub pt2_threshold: f64,
    pub max_determinants: usize,
    pub n_core_determinants: usize,
    pub n_target_determinants: usize,
    pub coarse_graining: f64,
    pub compute_pt2: bool,
}
impl Default for SciOptions {
    fn default() -> Self {
        Self {
            projector_generator: PCIGenerator::WallChebyshev,
            diagonalization_max_iterations: 100,
            polynomial_order: 4,
            selection_threshold: 1e-5,
            pt2_threshold: 1e-8,
            max_determinants: 100_000,
            n_core_determinants: 50,
            n_target_determinants: 200,
            coarse_graining: 0.5,
            compute_pt2: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PostHFOptions {
    pub active: Option<ActiveSpace>,
    pub frozen_core: usize,
    pub frozen_virtual: usize,
    pub root: usize,
    pub n_roots: usize,
    pub max_iter: usize,
    pub residual_tol: f64,
    pub energy_tol: f64,
    pub density_fitting: bool,
    pub aux_basis: Option<String>,
    pub numerical_gradient_step: f64,
    pub local: LocalOptions,
    pub dmrg: DmrgOptions,
    pub cas: CasOptions,
    pub dsrg: DsrgOptions,
    pub f12: F12Options,
    pub sci: SciOptions,
    pub scs_same_spin: f64,
    pub scs_opposite_spin: f64,
    pub scs_third_order: f64,
    pub cis_triplet: bool,
}
impl Default for PostHFOptions {
    fn default() -> Self {
        Self {
            active: None,
            frozen_core: 0,
            frozen_virtual: 0,
            root: 0,
            n_roots: 1,
            max_iter: 100,
            residual_tol: 1e-10,
            energy_tol: 1e-12,
            density_fitting: false,
            aux_basis: None,
            numerical_gradient_step: 1e-3,
            local: LocalOptions::default(),
            dmrg: DmrgOptions::default(),
            cas: CasOptions::default(),
            dsrg: DsrgOptions::default(),
            f12: F12Options::default(),
            sci: SciOptions::default(),
            scs_same_spin: 1.0 / 3.0,
            scs_opposite_spin: 1.2,
            scs_third_order: 0.25,
            cis_triplet: false,
        }
    }
}

pub const COMMON_OPTIONS: &[&str] = &[
    "frozen_core",
    "frozen_virtual",
    "root",
    "n_roots",
    "max_iter",
    "residual_tol",
    "energy_tol",
    "density_fitting",
    "aux_basis",
    "numerical_gradient_step",
];
pub const ACTIVE_OPTIONS: &[&str] = &[
    "active_n_alpha",
    "active_n_beta",
    "active_n_orb",
    "active_orbitals",
];
pub const CAS_OPTIONS: &[&str] = &[
    "max_orbital_iterations",
    "orbital_gradient_tol",
    "max_rotation",
    "target_s_squared",
    "state_weights",
];
pub const MRCI_OPTIONS: &[&str] = &["optimize_orbitals"];
pub const CASPT2_OPTIONS: &[&str] = &[
    "ipea_shift",
    "imaginary_shift",
    "real_shift",
    "caspt2_metric_tol",
];
pub const LOCAL_OPTIONS: &[&str] = &[
    "t_cut_pno",
    "t_cut_tno",
    "t_cut_pairs",
    "t_cut_do",
    "t_cut_mkn",
    "full_domains",
    "weak_pairs",
    "iterative_triples",
];
pub const DMRG_OPTIONS: &[&str] = &[
    "bond_dimensions",
    "sweeps",
    "one_site_sweeps",
    "noise",
    "truncation_tol",
];
pub const DSRG_OPTIONS: &[&str] = &[
    "flow_parameter",
    "relax_reference",
    "ode_step",
    "ode_tolerance",
];
pub const MR_DSRG_OPTIONS: &[&str] = &[
    "flow_parameter",
    "relax_reference",
    "reference_relaxation",
    "max_reference_iterations",
    "reference_energy_tol",
    "reference_ci_tol",
    "bch_tolerance",
    "bch_max_order",
    "optimize_orbitals",
];
pub const THREE_DSRG_OPTIONS: &[&str] = &[
    "flow_parameter",
    "cholesky_tolerance",
    "relax_reference",
    "reference_relaxation",
    "max_reference_iterations",
    "reference_energy_tol",
    "reference_ci_tol",
    "optimize_orbitals",
];
pub const MULTISTATE_DSRG_OPTIONS: &[&str] =
    &["flow_parameter", "cholesky_tolerance", "optimize_orbitals"];
pub const DWMS_DSRG_OPTIONS: &[&str] = &["dynamic_weight_parameter"];
/// Fixed quadratic BCH has no convergence threshold or maximum BCH order.
pub const MR_QC_OPTIONS: &[&str] = &[
    "flow_parameter",
    "relax_reference",
    "reference_relaxation",
    "max_reference_iterations",
    "reference_energy_tol",
    "reference_ci_tol",
    "optimize_orbitals",
];
pub const SR_DSRG_OPTIONS: &[&str] = &["flow_parameter"];
pub const LDSRG_OPTIONS: &[&str] = &["flow_parameter", "bch_tolerance", "bch_max_order"];
pub const IMSRG_OPTIONS: &[&str] = &[
    "flow_parameter",
    "ode_step",
    "ode_tolerance",
    "ode_generator",
];
pub const F12_OPTIONS: &[&str] = &[
    "beta",
    "f12_approximation",
    "f12_hybrid",
    "cabs_basis",
    "f12_fitting_basis",
    "f12_fitting_basis_kind",
    "f12_fock_fitting_basis",
    "f12_fock_fitting_basis_kind",
    "f12_fit_orbital_mp2",
    "f12_exact_obs_coulomb",
    "cabs_singles",
    "cabs_core_singles",
    "triples_scaling",
    "f12_workspace_bytes",
];
pub const SCI_OPTIONS: &[&str] = &[
    "selection_threshold",
    "pt2_threshold",
    "max_determinants",
    "ci_max_iter",
];
pub const ACI_OPTIONS: &[&str] = &["coarse_graining", "compute_pt2"];
pub const ASCI_OPTIONS: &[&str] = &[
    "n_core_determinants",
    "n_target_determinants",
    "compute_pt2",
];
pub const HCI_OPTIONS: &[&str] = &["compute_pt2"];
pub const PCI_OPTIONS: &[&str] = &[
    "ci_max_iter",
    "projector_generator",
    "polynomial_order",
    "selection_threshold",
    "max_determinants",
    "compute_pt2",
];
pub const SCS_OPTIONS: &[&str] = &["scs_same_spin", "scs_opposite_spin"];
pub const SCS_MP3_OPTIONS: &[&str] = &["scs_same_spin", "scs_opposite_spin", "scs_third_order"];
pub const CIS_OPTIONS: &[&str] = &["cis_triplet"];

/// Public input metadata. Defaults are read from the same typed objects as parsing.
/// A null default means omission, not acceptance of an explicit JSON null.
pub fn option_definition(name: &str) -> Option<Value> {
    let d = PostHFOptions::default();
    macro_rules! row {
        ($value:expr,$kind:literal,$unit:literal,$help:literal) => {
            (json!($value), $kind, $unit, $help)
        };
    }
    let (default, kind, unit, description) = match name {
        "frozen_core" => row!(
            d.frozen_core,
            "integer",
            "1",
            "Number of lowest doubly occupied spatial orbitals excluded from correlation."
        ),
        "frozen_virtual" => row!(
            d.frozen_virtual,
            "integer",
            "1",
            "Number of highest virtual spatial orbitals excluded from correlation."
        ),
        "root" => row!(
            d.root,
            "integer",
            "1",
            "Zero-based selected state; must be smaller than the computed state count."
        ),
        "n_roots" => row!(
            d.n_roots,
            "integer",
            "1",
            "Positive number of requested states; support depends on the selected method."
        ),
        "max_iter" => row!(
            d.max_iter,
            "integer",
            "1",
            "Positive post-HF amplitude, CI selection or flow iteration limit; separate from scf_max_iter."
        ),
        "residual_tol" => row!(
            d.residual_tol,
            "real",
            "method-native residual",
            "Positive convergence threshold for the selected solver's documented residual."
        ),
        "energy_tol" => row!(
            d.energy_tol,
            "real",
            "hartree",
            "Positive post-HF energy-change convergence threshold."
        ),
        "density_fitting" => row!(
            d.density_fitting,
            "boolean",
            "1",
            "Request a density-fitted branch where implemented; fitted method names select their own branch."
        ),
        "aux_basis" => row!(
            d.aux_basis,
            "string",
            "1",
            "Name of the physical auxiliary Coulomb-fitting basis; required by fitted adapters."
        ),
        "numerical_gradient_step" => row!(
            d.numerical_gradient_step,
            "real",
            "bohr",
            "Positive displacement for explicitly numerical low-level debug routes; does not enable analytic forces."
        ),
        "active_n_alpha" => row!(
            None::<usize>,
            "integer",
            "1",
            "Active alpha-electron count; supply together with active_n_beta and active_n_orb."
        ),
        "active_n_beta" => row!(
            None::<usize>,
            "integer",
            "1",
            "Active beta-electron count; inactive orbitals must be doubly occupied."
        ),
        "active_n_orb" => row!(
            None::<usize>,
            "integer",
            "1",
            "Active spatial-orbital count; positive and large enough for both spin populations."
        ),
        "active_orbitals" => row!(
            None::<Vec<usize>>,
            "integer list",
            "1",
            "Distinct zero-based original MO indices; omission selects consecutive active orbitals after inactive occupied orbitals."
        ),
        "max_orbital_iterations" => row!(
            d.cas.max_orbital_iterations,
            "integer",
            "1",
            "Positive CASSCF/DMRGSCF orbital-optimization iteration limit."
        ),
        "orbital_gradient_tol" => row!(
            d.cas.orbital_gradient_tol,
            "real",
            "hartree",
            "Positive maximum orbital-gradient convergence threshold."
        ),
        "max_rotation" => row!(
            d.cas.max_rotation,
            "real",
            "radian",
            "Positive orbital-rotation trust bound."
        ),
        "optimize_orbitals" => row!(
            d.cas.optimize_orbitals,
            "boolean",
            "1",
            "Optimize the multireference orbitals where the adapter implements that option."
        ),
        "target_s_squared" => row!(
            d.cas.target_s_squared,
            "real",
            "1",
            "Requested S(S+1) for supported exact spin projection; omission uses the adapter's spin convention."
        ),
        "state_weights" => row!(
            d.cas.state_weights,
            "real list",
            "1",
            "Nonnegative state-average weights; an empty list requests the adapter's default equal weighting."
        ),
        "ipea_shift" => row!(
            d.cas.ipea_shift,
            "real",
            "hartree",
            "CASPT2 IPEA shift in the documented contracted zeroth-order Hamiltonian."
        ),
        "real_shift" => row!(
            d.cas.real_shift,
            "real",
            "hartree",
            "CASPT2 real level shift; its energy correction is reported separately."
        ),
        "imaginary_shift" => row!(
            d.cas.imaginary_shift,
            "real",
            "hartree",
            "Nonnegative CASPT2 imaginary level shift."
        ),
        "caspt2_metric_tol" => row!(
            d.cas.caspt2_metric_tol,
            "real",
            "1",
            "Positive relative overlap cutoff in the internally contracted CASPT2 space."
        ),
        "t_cut_pno" => row!(
            d.local.t_cut_pno,
            "real",
            "1",
            "Nonnegative pair-natural-orbital occupation cutoff."
        ),
        "t_cut_tno" => row!(
            d.local.t_cut_tno,
            "real",
            "1",
            "Triples-natural-orbital occupation cutoff; omission inherits t_cut_pno."
        ),
        "t_cut_pairs" => row!(
            d.local.t_cut_pairs,
            "real",
            "hartree",
            "Nonnegative pair-energy threshold separating strong and weak pairs."
        ),
        "t_cut_do" => row!(
            d.local.t_cut_do,
            "real",
            "native basis metric",
            "Nonnegative differential-overlap domain threshold in the documented local model."
        ),
        "t_cut_mkn" => row!(
            d.local.t_cut_mkn,
            "real",
            "native basis metric",
            "Nonnegative local fitting-domain threshold."
        ),
        "full_domains" => row!(
            d.local.full_domains,
            "boolean",
            "1",
            "Use the full molecular domain instead of adaptive local domains."
        ),
        "weak_pairs" => row!(
            d.local.weak_pairs,
            "boolean",
            "1",
            "Retain the documented weak-pair correction."
        ),
        "iterative_triples" => row!(
            d.local.iterative_triples,
            "boolean",
            "1",
            "Select the supported iterative local triples correction rather than the T0 approximation."
        ),
        "bond_dimensions" => row!(
            d.dmrg.bond_dimensions,
            "integer list",
            "1",
            "Positive MPS bond-dimension schedule, extended using its last value."
        ),
        "sweeps" => row!(
            d.dmrg.sweeps,
            "integer",
            "1",
            "Positive number of two-site DMRG sweeps."
        ),
        "one_site_sweeps" => row!(
            d.dmrg.one_site_sweeps,
            "integer",
            "1",
            "Number of final one-site DMRG sweeps."
        ),
        "noise" => row!(
            d.dmrg.noise,
            "real",
            "method-native noise",
            "Only zero noise is implemented by the current DMRG adapter."
        ),
        "truncation_tol" => row!(
            d.dmrg.truncation_tol,
            "real",
            "1",
            "Positive discarded Schmidt-weight threshold."
        ),
        "flow_parameter" => row!(
            d.dsrg.flow_parameter,
            "real",
            "hartree^-2",
            "Nonnegative DSRG/IM-SRG flow endpoint."
        ),
        "cholesky_tolerance" => row!(
            d.dsrg.cholesky_tolerance,
            "real",
            "hartree",
            "Positive maximum residual Coulomb diagonal for factorized THREE kernels."
        ),
        "dynamic_weight_parameter" => row!(
            d.dsrg.dynamic_weight_parameter,
            "real",
            "hartree^-2",
            "Nonnegative DWMS Gaussian energy-separation weight parameter."
        ),
        "relax_reference" => row!(
            d.dsrg.relax_reference,
            "boolean",
            "1",
            "Compatibility flag selecting one reference rediagonalization unless an explicit relaxation mode is supplied."
        ),
        "reference_relaxation" => row!(
            format!("{:?}", d.dsrg.reference_relaxation).to_ascii_lowercase(),
            "string",
            "1",
            "Number or convergence policy of effective-CAS reference rediagonalizations."
        ),
        "max_reference_iterations" => row!(
            d.dsrg.max_reference_iterations,
            "integer",
            "1",
            "Positive iteration limit for iterative reference relaxation."
        ),
        "reference_energy_tol" => row!(
            d.dsrg.reference_energy_tol,
            "real",
            "hartree",
            "Positive relaxed-reference energy-change threshold."
        ),
        "reference_ci_tol" => row!(
            d.dsrg.reference_ci_tol,
            "real",
            "1",
            "Positive relaxed-reference CI-vector change threshold."
        ),
        "ode_step" => row!(
            d.dsrg.ode_step,
            "real",
            "hartree^-2",
            "Positive initial flow-integration step."
        ),
        "ode_tolerance" => row!(
            d.dsrg.ode_tolerance,
            "real",
            "method-native residual",
            "Positive adaptive ODE local-error tolerance."
        ),
        "ode_generator" => row!(
            format!("{:?}", d.dsrg.ode_generator).to_ascii_lowercase(),
            "string",
            "1",
            "IM-SRG generator: Wegner commutator or regularized White form."
        ),
        "bch_tolerance" => row!(
            d.dsrg.bch_tolerance,
            "real",
            "hartree",
            "Positive operator-norm cutoff for the converged recursive BCH series."
        ),
        "bch_max_order" => row!(
            d.dsrg.bch_max_order,
            "integer",
            "1",
            "Positive maximum recursive BCH order; exhaustion is an error."
        ),
        "beta" => row!(
            d.f12.beta,
            "real",
            "bohr^-1",
            "Positive Slater-geminal exponent; physical CABS/fitting spaces must match the chosen model."
        ),
        "f12_approximation" => row!(
            match d.f12.approximation {
                F12Approximation::ApproximationA => "3*A(FIX)",
                F12Approximation::ApproximationB => "3B(FIX)",
                F12Approximation::ApproximationC => "3C(FIX)",
            },
            "string",
            "1",
            "Implemented fixed-amplitude F12 approximation; the distinct 3B theory is unsupported."
        ),
        "f12_hybrid" => row!(
            format!("{:?}", d.f12.hybrid).to_ascii_lowercase(),
            "string",
            "1",
            "Full residual nonlocal resolution, HY1 omission or HY2 CABS-CABS omission."
        ),
        "cabs_basis" => row!(
            d.f12.cabs_basis,
            "string",
            "1",
            "Physical complementary auxiliary basis name; required by the F12 molecular adapter."
        ),
        "f12_fitting_basis" => row!(
            d.f12.fitting_basis,
            "string",
            "1",
            "Optional robust-fitting basis for F12 correction kernels."
        ),
        "f12_fock_fitting_basis" => row!(
            d.f12.fock_fitting_basis,
            "string",
            "1",
            "Optional independent Fock fitting basis; omission uses the implemented unfitted/F12 selection rules."
        ),
        "f12_fitting_basis_kind" => row!(
            format!("{:?}", d.f12.fitting_basis_kind).to_ascii_lowercase(),
            "string",
            "1",
            "Cartesian or real spherical physical correction-fitting space."
        ),
        "f12_fock_fitting_basis_kind" => row!(
            format!("{:?}", d.f12.fock_fitting_basis_kind).to_ascii_lowercase(),
            "string",
            "1",
            "Cartesian or real spherical physical Fock-fitting space."
        ),
        "f12_fit_orbital_mp2" => row!(
            d.f12.fit_orbital_mp2,
            "boolean",
            "1",
            "Use the selected robust fitting space for the orbital MP2 contribution where supported."
        ),
        "f12_exact_obs_coulomb" => row!(
            d.f12.exact_obs_coulomb,
            "boolean",
            "1",
            "Restore exact Coulomb integrals inside the OBS projector; fitted remaining spaces stay fitted."
        ),
        "cabs_singles" => row!(
            d.f12.cabs_singles,
            "boolean",
            "1",
            "Include the implemented CABS-singles correction."
        ),
        "cabs_core_singles" => row!(
            d.f12.cabs_core_singles,
            "boolean",
            "1",
            "Include frozen occupied orbitals in CABS singles."
        ),
        "triples_scaling" => row!(
            d.f12.triples_scaling,
            "boolean",
            "1",
            "Use the selected F12 triples-scaling convention where implemented."
        ),
        "f12_workspace_bytes" => row!(
            d.f12.workspace_budget_bytes,
            "integer",
            "byte",
            "Positive limit on the live mixed-tensor workspace estimate; excludes reference storage and process overhead."
        ),
        "ci_max_iter" => row!(
            d.sci.diagonalization_max_iterations,
            "integer",
            "1",
            "Positive inner variational Davidson iteration limit, separate from outer selection/propagation max_iter."
        ),
        "projector_generator" => row!(
            format!("{:?}", d.sci.projector_generator).to_ascii_lowercase(),
            "string",
            "1",
            "PCI polynomial projector or selected-space Davidson/Lanczos accelerator."
        ),
        "polynomial_order" => row!(
            d.sci.polynomial_order,
            "integer",
            "1",
            "Positive polynomial order for supported PCI generators."
        ),
        "selection_threshold" => row!(
            d.sci.selection_threshold,
            "real",
            "method-native selection",
            "Nonnegative ACI energy importance, HCI coupling threshold or PCI spawning threshold; definitions differ by method."
        ),
        "pt2_threshold" => row!(
            d.sci.pt2_threshold,
            "real",
            "hartree",
            "Nonnegative HCI individual coupling screen before summation in the PT2 search."
        ),
        "max_determinants" => row!(
            d.sci.max_determinants,
            "integer",
            "1",
            "Positive determinant-space limit; reaching the limit does not certify convergence."
        ),
        "n_core_determinants" => row!(
            d.sci.n_core_determinants,
            "integer",
            "1",
            "Positive ASCI core-search determinant count."
        ),
        "n_target_determinants" => row!(
            d.sci.n_target_determinants,
            "integer",
            "1",
            "Positive ASCI target variational determinant count."
        ),
        "coarse_graining" => row!(
            d.sci.coarse_graining,
            "real",
            "1",
            "Nonnegative ACI pruning factor applied to the numerical Hartree selection threshold; the cumulative squared-weight bound must remain below one."
        ),
        "compute_pt2" => row!(
            d.sci.compute_pt2,
            "boolean",
            "1",
            "Compute the adapter's deterministic selected-space correction; returned RDMs remain variational."
        ),
        "scs_same_spin" => row!(
            d.scs_same_spin,
            "real",
            "1",
            "Finite same-spin second-order energy scale."
        ),
        "scs_opposite_spin" => row!(
            d.scs_opposite_spin,
            "real",
            "1",
            "Finite opposite-spin second-order energy scale."
        ),
        "scs_third_order" => row!(
            d.scs_third_order,
            "real",
            "1",
            "Finite third-order increment scale in SCS-MP3."
        ),
        "cis_triplet" => row!(
            d.cis_triplet,
            "boolean",
            "1",
            "Select triplet rather than singlet spin-adapted CIS excitations."
        ),
        _ => return None,
    };
    let choices: &[&str] = match name {
        "reference_relaxation" => &["none", "once", "twice", "iterate"],
        "ode_generator" => &["wegner", "white"],
        "f12_approximation" => &["3C(FIX)", "3*A(FIX)"],
        "f12_hybrid" => &["full", "hy1", "hy2"],
        "f12_fitting_basis_kind" | "f12_fock_fitting_basis_kind" => &["cartesian", "spherical"],
        "projector_generator" => &[
            "wallchebyshev",
            "chebyshev",
            "expchebyshev",
            "linear",
            "quadratic",
            "cubic",
            "quartic",
            "power",
            "davidson",
            "lanczos",
        ],
        _ => &[],
    };
    Some(
        json!({"name":name,"type":kind,"default":default,"unit":unit,
        "description":description,"choices":choices,"null_input_allowed":false}),
    )
}

fn number(key: &str, value: &Value) -> Result<f64, HfError> {
    value
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| HfError::InvalidInput(format!("option {key} requires a finite number")))
}
fn integer(key: &str, value: &Value) -> Result<usize, HfError> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| {
            HfError::InvalidInput(format!("option {key} requires a nonnegative integer"))
        })
}
fn boolean(key: &str, value: &Value) -> Result<bool, HfError> {
    value
        .as_bool()
        .ok_or_else(|| HfError::InvalidInput(format!("option {key} requires a boolean")))
}
fn string(key: &str, value: &Value) -> Result<String, HfError> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| HfError::InvalidInput(format!("option {key} requires a string")))
}
fn integers(key: &str, value: &Value) -> Result<Vec<usize>, HfError> {
    value
        .as_array()
        .ok_or_else(|| HfError::InvalidInput(format!("option {key} requires an integer list")))?
        .iter()
        .map(|v| integer(key, v))
        .collect()
}
impl PostHFOptions {
    /// `schemas` comes from the selected typed registry entry. Inapplicable and
    /// misspelled options are errors instead of being silently ignored.
    pub fn from_map(
        values: &BTreeMap<String, Value>,
        schemas: &[&[&str]],
    ) -> Result<Self, HfError> {
        let mut result = Self::default();
        let mut na = None;
        let mut nb = None;
        let mut norb = None;
        let mut orbitals = None;
        for (key, value) in values {
            if !COMMON_OPTIONS.contains(&key.as_str())
                && !schemas.iter().any(|schema| schema.contains(&key.as_str()))
            {
                return Err(HfError::InvalidInput(format!(
                    "unknown or inapplicable option: {key}"
                )));
            }
            match key.as_str() {
                "frozen_core" => result.frozen_core = integer(key, value)?,
                "frozen_virtual" => result.frozen_virtual = integer(key, value)?,
                "root" => result.root = integer(key, value)?,
                "n_roots" => result.n_roots = integer(key, value)?,
                "max_iter" => result.max_iter = integer(key, value)?,
                "residual_tol" => result.residual_tol = number(key, value)?,
                "energy_tol" => result.energy_tol = number(key, value)?,
                "density_fitting" => result.density_fitting = boolean(key, value)?,
                "aux_basis" => result.aux_basis = Some(string(key, value)?),
                "numerical_gradient_step" => result.numerical_gradient_step = number(key, value)?,
                "active_n_alpha" => na = Some(integer(key, value)?),
                "active_n_beta" => nb = Some(integer(key, value)?),
                "active_n_orb" => norb = Some(integer(key, value)?),
                "active_orbitals" => orbitals = Some(integers(key, value)?),
                "max_orbital_iterations" => {
                    result.cas.max_orbital_iterations = integer(key, value)?
                }
                "orbital_gradient_tol" => result.cas.orbital_gradient_tol = number(key, value)?,
                "max_rotation" => result.cas.max_rotation = number(key, value)?,
                "optimize_orbitals" => result.cas.optimize_orbitals = boolean(key, value)?,
                "target_s_squared" => result.cas.target_s_squared = Some(number(key, value)?),
                "state_weights" => {
                    result.cas.state_weights = value
                        .as_array()
                        .ok_or_else(|| {
                            HfError::InvalidInput("state_weights requires a list".into())
                        })?
                        .iter()
                        .map(|v| number(key, v))
                        .collect::<Result<_, _>>()?
                }
                "ipea_shift" => result.cas.ipea_shift = number(key, value)?,
                "imaginary_shift" => result.cas.imaginary_shift = number(key, value)?,
                "caspt2_metric_tol" => result.cas.caspt2_metric_tol = number(key, value)?,
                "real_shift" => result.cas.real_shift = number(key, value)?,
                "t_cut_pno" => result.local.t_cut_pno = number(key, value)?,
                "t_cut_tno" => result.local.t_cut_tno = Some(number(key, value)?),
                "t_cut_pairs" => result.local.t_cut_pairs = number(key, value)?,
                "t_cut_do" => result.local.t_cut_do = number(key, value)?,
                "t_cut_mkn" => result.local.t_cut_mkn = number(key, value)?,
                "full_domains" => result.local.full_domains = boolean(key, value)?,
                "weak_pairs" => result.local.weak_pairs = boolean(key, value)?,
                "iterative_triples" => result.local.iterative_triples = boolean(key, value)?,
                "bond_dimensions" => result.dmrg.bond_dimensions = integers(key, value)?,
                "sweeps" => result.dmrg.sweeps = integer(key, value)?,
                "one_site_sweeps" => result.dmrg.one_site_sweeps = integer(key, value)?,
                "noise" => result.dmrg.noise = number(key, value)?,
                "truncation_tol" => result.dmrg.truncation_tol = number(key, value)?,
                "flow_parameter" => result.dsrg.flow_parameter = number(key, value)?,
                "cholesky_tolerance" => result.dsrg.cholesky_tolerance = number(key, value)?,
                "dynamic_weight_parameter" => {
                    result.dsrg.dynamic_weight_parameter = number(key, value)?
                }
                "relax_reference" => result.dsrg.relax_reference = boolean(key, value)?,
                "reference_relaxation" => {
                    result.dsrg.reference_relaxation =
                        match string(key, value)?.to_ascii_lowercase().as_str() {
                            "none" => ReferenceRelaxation::None,
                            "once" => ReferenceRelaxation::Once,
                            "twice" => ReferenceRelaxation::Twice,
                            "iterate" => ReferenceRelaxation::Iterate,
                            _ => {
                                return Err(HfError::InvalidInput(
                                    "reference_relaxation must be none, once, twice or iterate"
                                        .into(),
                                ));
                            }
                        };
                }
                "max_reference_iterations" => {
                    result.dsrg.max_reference_iterations = integer(key, value)?
                }
                "reference_energy_tol" => result.dsrg.reference_energy_tol = number(key, value)?,
                "reference_ci_tol" => result.dsrg.reference_ci_tol = number(key, value)?,
                "bch_tolerance" => result.dsrg.bch_tolerance = number(key, value)?,
                "bch_max_order" => result.dsrg.bch_max_order = integer(key, value)?,
                "ode_step" => result.dsrg.ode_step = number(key, value)?,
                "ode_tolerance" => result.dsrg.ode_tolerance = number(key, value)?,
                "ode_generator" => {
                    result.dsrg.ode_generator =
                        match string(key, value)?.to_ascii_lowercase().as_str() {
                            "wegner" => ImsrgGenerator::Wegner,
                            "white" => ImsrgGenerator::White,
                            _ => {
                                return Err(HfError::InvalidInput(
                                    "ode_generator must be wegner or white".into(),
                                ));
                            }
                        }
                }
                "beta" => result.f12.beta = number(key, value)?,
                "f12_hybrid" => {
                    result.f12.hybrid = match string(key, value)?.to_ascii_lowercase().as_str() {
                        "full" | "none" => F12Hybrid::Full,
                        "hy1" => F12Hybrid::Hy1,
                        "hy2" => F12Hybrid::Hy2,
                        _ => {
                            return Err(HfError::InvalidInput(
                                "f12_hybrid must be full, HY1 or HY2".into(),
                            ));
                        }
                    };
                }
                "f12_approximation" => {
                    result.f12.approximation =
                        match string(key, value)?.to_ascii_lowercase().as_str() {
                            "3c" | "3c(fix)" => F12Approximation::ApproximationC,
                            "3*a" | "3*a(fix)" => F12Approximation::ApproximationA,
                            _ => {
                                return Err(HfError::InvalidInput(
                                    "f12_approximation must be 3C(FIX) or 3*A(FIX)".into(),
                                ));
                            }
                        }
                }
                "cabs_basis" => result.f12.cabs_basis = Some(string(key, value)?),
                "f12_fitting_basis" => result.f12.fitting_basis = Some(string(key, value)?),
                "f12_fock_fitting_basis" => {
                    result.f12.fock_fitting_basis = Some(string(key, value)?);
                }
                "f12_fit_orbital_mp2" => result.f12.fit_orbital_mp2 = boolean(key, value)?,
                "f12_exact_obs_coulomb" => result.f12.exact_obs_coulomb = boolean(key, value)?,
                "f12_fitting_basis_kind" | "f12_fock_fitting_basis_kind" => {
                    let kind = match string(key, value)?.to_ascii_lowercase().as_str() {
                        "cartesian" => crate::ao::BasisKind::Cartesian,
                        "spherical" => crate::ao::BasisKind::Spherical,
                        _ => {
                            return Err(HfError::InvalidInput(format!(
                                "{key} must be cartesian or spherical"
                            )));
                        }
                    };
                    if key == "f12_fitting_basis_kind" {
                        result.f12.fitting_basis_kind = kind;
                    } else {
                        result.f12.fock_fitting_basis_kind = kind;
                    }
                }
                "cabs_singles" => result.f12.cabs_singles = boolean(key, value)?,
                "cabs_core_singles" => result.f12.cabs_core_singles = boolean(key, value)?,
                "triples_scaling" => result.f12.triples_scaling = boolean(key, value)?,
                "f12_workspace_bytes" => result.f12.workspace_budget_bytes = integer(key, value)?,
                "n_core_determinants" => result.sci.n_core_determinants = integer(key, value)?,
                "ci_max_iter" => result.sci.diagonalization_max_iterations = integer(key, value)?,
                "n_target_determinants" => result.sci.n_target_determinants = integer(key, value)?,
                "coarse_graining" => result.sci.coarse_graining = number(key, value)?,
                "projector_generator" => {
                    result.sci.projector_generator = string(key, value)?.parse()?
                }
                "polynomial_order" => result.sci.polynomial_order = integer(key, value)?,
                "compute_pt2" => result.sci.compute_pt2 = boolean(key, value)?,
                "selection_threshold" => result.sci.selection_threshold = number(key, value)?,
                "pt2_threshold" => result.sci.pt2_threshold = number(key, value)?,
                "max_determinants" => result.sci.max_determinants = integer(key, value)?,
                "scs_same_spin" => result.scs_same_spin = number(key, value)?,
                "scs_opposite_spin" => result.scs_opposite_spin = number(key, value)?,
                "scs_third_order" => result.scs_third_order = number(key, value)?,
                "cis_triplet" => result.cis_triplet = boolean(key, value)?,
                _ => {
                    return Err(HfError::InvalidInput(format!(
                        "option parser missing schema entry: {key}"
                    )));
                }
            }
        }
        if na.is_some() || nb.is_some() || norb.is_some() || orbitals.is_some() {
            result.active = Some(ActiveSpace {
                n_alpha: na
                    .ok_or_else(|| HfError::InvalidInput("active_n_alpha is required".into()))?,
                n_beta: nb
                    .ok_or_else(|| HfError::InvalidInput("active_n_beta is required".into()))?,
                n_orb: norb
                    .ok_or_else(|| HfError::InvalidInput("active_n_orb is required".into()))?,
                orbitals,
            });
        }
        result.validate()?;
        Ok(result)
    }
    pub fn validate(&self) -> Result<(), HfError> {
        if self.f12.workspace_budget_bytes == 0 {
            return Err(HfError::InvalidInput(
                "F12 workspace budget must be positive".into(),
            ));
        }
        if self
            .local
            .t_cut_tno
            .is_some_and(|v| !v.is_finite() || v < 0.)
        {
            return Err(HfError::InvalidInput("invalid TNO cutoff".into()));
        }
        if [
            self.cas.ipea_shift,
            self.cas.real_shift,
            self.scs_same_spin,
            self.scs_opposite_spin,
            self.scs_third_order,
        ]
        .iter()
        .any(|v| !v.is_finite())
            || self
                .cas
                .target_s_squared
                .is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            return Err(HfError::InvalidInput(
                "non-finite method parameter or invalid target spin".into(),
            ));
        }
        if self.dsrg.max_reference_iterations == 0
            || self.dsrg.max_reference_iterations > 100_000
            || self.dsrg.bch_max_order == 0
            || self.dsrg.bch_max_order > 1000
            || self.max_iter == 0
            || self.n_roots == 0
            || self.root >= self.n_roots
            || self.cas.max_orbital_iterations == 0
            || self.dmrg.sweeps == 0
            || self.dmrg.bond_dimensions.is_empty()
            || self.dmrg.bond_dimensions.contains(&0)
            || self.sci.max_determinants == 0
            || self.sci.diagonalization_max_iterations == 0
        {
            return Err(HfError::InvalidInput(
                "invalid root, iteration count or DMRG bond dimension".into(),
            ));
        }
        for (key, v, zero_allowed) in [
            ("residual_tol", self.residual_tol, false),
            ("energy_tol", self.energy_tol, false),
            (
                "numerical_gradient_step",
                self.numerical_gradient_step,
                false,
            ),
            ("orbital_gradient_tol", self.cas.orbital_gradient_tol, false),
            ("max_rotation", self.cas.max_rotation, false),
            ("imaginary_shift", self.cas.imaginary_shift, true),
            ("caspt2_metric_tol", self.cas.caspt2_metric_tol, false),
            ("t_cut_pno", self.local.t_cut_pno, true),
            ("t_cut_pairs", self.local.t_cut_pairs, true),
            ("t_cut_do", self.local.t_cut_do, true),
            ("t_cut_mkn", self.local.t_cut_mkn, true),
            ("noise", self.dmrg.noise, true),
            ("truncation_tol", self.dmrg.truncation_tol, true),
            ("flow_parameter", self.dsrg.flow_parameter, true),
            ("cholesky_tolerance", self.dsrg.cholesky_tolerance, true),
            (
                "dynamic_weight_parameter",
                self.dsrg.dynamic_weight_parameter,
                true,
            ),
            ("ode_step", self.dsrg.ode_step, false),
            ("ode_tolerance", self.dsrg.ode_tolerance, false),
            ("bch_tolerance", self.dsrg.bch_tolerance, false),
            (
                "reference_energy_tol",
                self.dsrg.reference_energy_tol,
                false,
            ),
            ("reference_ci_tol", self.dsrg.reference_ci_tol, false),
            ("beta", self.f12.beta, false),
            ("selection_threshold", self.sci.selection_threshold, true),
            ("pt2_threshold", self.sci.pt2_threshold, true),
            ("coarse_graining", self.sci.coarse_graining, true),
        ] {
            if !v.is_finite() || v < 0.0 || (!zero_allowed && v == 0.0) {
                return Err(HfError::InvalidInput(format!("invalid option {key}: {v}")));
            }
        }
        if self
            .cas
            .state_weights
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
            || (!self.cas.state_weights.is_empty()
                && (self.cas.state_weights.len() != self.n_roots
                    || self.cas.state_weights.iter().sum::<f64>() <= 0.0))
        {
            return Err(HfError::InvalidInput(
                "state_weights must be nonnegative, nonzero and match n_roots".into(),
            ));
        }
        Ok(())
    }
}
