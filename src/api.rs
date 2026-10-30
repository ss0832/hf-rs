// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Shared calculation boundary for Rust, Python, ASE and the command line.
//! Coordinates are converted here; all returned energies and derivatives use
//! atomic units. A molecule owns its geometry and caches its converged SCF.

use crate::{
    ao::BasisKind,
    basis::{Atom, BasisSetMap, ContractedGaussian, atomic_number_to_symbol, try_build_basis},
    context::{AuxBasis, PostHFContext, RefKind, Reference},
    diis::ConvAlgo,
    error::HfError,
    hf_derivatives::HfNuclearDerivativeContext,
    matrix::Matrix,
    options::PostHFOptions,
    output::{HfObservables, ObservableMap, PostHFOutput, TensorMap, component_unit},
    registry::{GradientKind, MethodId, create_solver},
    scf::{HFResult, RhfConvergence, UHFResult},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    str::FromStr,
    sync::{Arc, Mutex},
};

pub type OptionMap = BTreeMap<String, Value>;

/// Explicitly bounded bare integral inspection, with fixed orbital coefficients.
#[derive(Clone, Debug)]
pub struct IntegralRequest {
    pub operator: String,
    pub representation: String,
    /// One index list per tensor axis; an empty outer vector selects complete axes.
    pub indices: Vec<Vec<usize>>,
    pub derivative_coordinates: Vec<usize>,
    /// One shared matrix or one matrix per axis for representation=custom.
    pub coefficients: Vec<Matrix>,
    pub aux_basis: Option<String>,
    pub metric_cutoff: f64,
    pub max_elements: usize,
    pub workspace_budget_bytes: usize,
}
impl Default for IntegralRequest {
    fn default() -> Self {
        Self {
            operator: "overlap".into(),
            representation: "cartesian_ao".into(),
            indices: Vec::new(),
            derivative_coordinates: Vec::new(),
            coefficients: Vec::new(),
            aux_basis: None,
            metric_cutoff: 1e-10,
            max_elements: 1_000_000,
            workspace_budget_bytes: 128 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct IntegralBlock {
    pub operator: String,
    pub representation: String,
    pub indices: Vec<Vec<usize>>,
    pub derivative_coordinates: Vec<usize>,
    pub orbital_response_included: bool,
    pub tensor: crate::output::TensorData,
}

fn tensor_indices(
    request: &[Vec<usize>],
    dimensions: &[usize],
    limit: usize,
) -> Result<Vec<Vec<usize>>, HfError> {
    if !request.is_empty() && request.len() != dimensions.len() {
        return Err(HfError::InvalidInput(
            "supply one index list per integral axis".into(),
        ));
    }
    let shape: Vec<_> = dimensions
        .iter()
        .enumerate()
        .map(|(axis, &n)| {
            if request.is_empty() {
                n
            } else {
                request[axis].len()
            }
        })
        .collect();
    if shape.contains(&0) {
        return Err(HfError::InvalidInput(
            "integral index lists must be nonempty".into(),
        ));
    }
    crate::output::TensorData::checked_len(&shape, limit)?;
    let selected: Vec<_> = dimensions
        .iter()
        .enumerate()
        .map(|(axis, &n)| {
            if request.is_empty() {
                (0..n).collect()
            } else {
                request[axis].clone()
            }
        })
        .collect();
    for (axis, indices) in selected.iter().enumerate() {
        if indices.iter().any(|&i| i >= dimensions[axis]) {
            return Err(HfError::InvalidInput(format!(
                "integral index out of range on axis {axis}"
            )));
        }
    }
    Ok(selected)
}

fn workspace_limit(elements: u128, budget: usize) -> Result<(), HfError> {
    if budget == 0 || elements.saturating_mul(8) > budget as u128 {
        Err(HfError::InvalidInput("integral/reference workspace estimate exceeds workspace_budget_bytes; request a smaller block or increase the explicit budget".into()))
    } else {
        Ok(())
    }
}

fn selected_columns(c: &Matrix, indices: &[usize]) -> Matrix {
    let mut result = Matrix::zeros(c.rows, indices.len());
    for i in 0..c.rows {
        for (j, &column) in indices.iter().enumerate() {
            result[(i, j)] = c[(i, column)];
        }
    }
    result
}

fn bare_pair_derivative(
    atoms: &[Atom],
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    directions: &[crate::NuclearDirection],
    operator: &str,
) -> f64 {
    use crate::derivatives::{
        GeneralBasisDerivative, diff_cg_directions, eval_kinetic_deriv,
        eval_nuclear_deriv_single_atom, eval_overlap_deriv,
    };
    if operator == "overlap" || operator == "kinetic" {
        let relative: Vec<_> = directions
            .iter()
            .map(|v| std::array::from_fn(|k| v[a.atom_index][k] - v[b.atom_index][k]))
            .collect();
        let left = diff_cg_directions(a, &relative);
        let right = GeneralBasisDerivative::from_single(b.clone());
        return if operator == "overlap" {
            eval_overlap_deriv(&left, &right)
        } else {
            eval_kinetic_deriv(&left, &right)
        };
    }
    let mut result = 0.;
    for (center, atom) in atoms.iter().enumerate() {
        for mask in 0..1usize << directions.len() {
            let mut left = Vec::new();
            let mut right = Vec::new();
            for (k, v) in directions.iter().enumerate() {
                if mask & (1 << k) != 0 {
                    left.push(std::array::from_fn(|a0| {
                        v[a.atom_index][a0] - v[center][a0]
                    }));
                } else {
                    right.push(std::array::from_fn(|a0| {
                        v[b.atom_index][a0] - v[center][a0]
                    }));
                }
            }
            result += eval_nuclear_deriv_single_atom(
                &diff_cg_directions(a, &left),
                &diff_cg_directions(b, &right),
                atom,
            );
        }
    }
    result
}

/// Quarter transformations of a selected four-leg block; never form full MO ERIs.
fn transform_eri_block(n: usize, packed: &[f64], c: &[Matrix]) -> Vec<f64> {
    use rayon::prelude::*;
    let [np, nq, nr, ns] = std::array::from_fn(|i| c[i].cols);
    let mut first = vec![0.; np * n * n * n];
    first.par_iter_mut().enumerate().for_each(|(index, value)| {
        let p = index / (n * n * n);
        let tail = index % (n * n * n);
        let j = tail / (n * n);
        let k = tail / n % n;
        let l = tail % n;
        *value = (0..n)
            .map(|i| c[0][(i, p)] * packed[crate::scf::eri_index(i, j, k, l)])
            .sum();
    });
    let mut second = vec![0.; np * nq * n * n];
    second
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, value)| {
            let p = index / (nq * n * n);
            let q = index / (n * n) % nq;
            let tail = index % (n * n);
            *value = (0..n)
                .map(|j| c[1][(j, q)] * first[(p * n + j) * n * n + tail])
                .sum();
        });
    drop(first);
    let mut third = vec![0.; np * nq * nr * n];
    third.par_iter_mut().enumerate().for_each(|(index, value)| {
        let pair = index / (nr * n);
        let r = index / n % nr;
        let l = index % n;
        *value = (0..n)
            .map(|k| c[2][(k, r)] * second[(pair * n + k) * n + l])
            .sum();
    });
    drop(second);
    let mut result = vec![0.; np * nq * nr * ns];
    result
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, value)| {
            let triple = index / ns;
            let s = index % ns;
            *value = (0..n).map(|l| c[3][(l, s)] * third[triple * n + l]).sum();
        });
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LengthUnit {
    Bohr,
    Angstrom,
}
impl FromStr for LengthUnit {
    type Err = HfError;
    fn from_str(value: &str) -> Result<Self, HfError> {
        match value.to_ascii_lowercase().as_str() {
            "bohr" => Ok(Self::Bohr),
            "angstrom" => Ok(Self::Angstrom),
            _ => Err(HfError::InvalidInput(format!(
                "unknown coordinate unit {value}; use bohr or angstrom"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReferenceChoice {
    Auto,
    Rhf,
    Uhf,
}
impl FromStr for ReferenceChoice {
    type Err = HfError;
    fn from_str(value: &str) -> Result<Self, HfError> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "rhf" => Ok(Self::Rhf),
            "uhf" => Ok(Self::Uhf),
            _ => Err(HfError::InvalidInput(format!(
                "unknown reference {value}; use auto, rhf or uhf"
            ))),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MoleculeOptions {
    pub unit: LengthUnit,
    pub basis: String,
    pub basis_by_element: BTreeMap<String, String>,
    pub ecp: Option<String>,
    pub ecp_by_element: BTreeMap<String, String>,
    pub cartesian: bool,
    pub charge: i32,
    pub multiplicity: usize,
    pub reference: ReferenceChoice,
}
impl Default for MoleculeOptions {
    fn default() -> Self {
        Self {
            unit: LengthUnit::Bohr,
            basis: "sto-3g".into(),
            basis_by_element: BTreeMap::new(),
            ecp: None,
            ecp_by_element: BTreeMap::new(),
            cartesian: true,
            charge: 0,
            multiplicity: 1,
            reference: ReferenceChoice::Auto,
        }
    }
}

impl MoleculeOptions {
    /// Set a common orbital basis or element map; `*` supplies an optional fallback.
    pub fn set_basis(&mut self, selection: &Value) -> Result<(), HfError> {
        let (default, elements) = crate::molecular_basis::assignment(selection, false)?;
        self.basis = default.unwrap_or_default();
        self.basis_by_element = elements;
        Ok(())
    }

    /// Omission uses associated ECPs; explicit names/files or `none` override them.
    pub fn set_ecp(&mut self, selection: &Value) -> Result<(), HfError> {
        let (default, elements) = crate::molecular_basis::assignment(selection, true)?;
        self.ecp = default;
        self.ecp_by_element = elements;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScfOptions {
    pub max_iter: usize,
    pub energy_tol: f64,
    pub residual_tol: f64,
    pub density_tol: f64,
    pub convergence: Option<ConvAlgo>,
    pub broken_symmetry: bool,
}
impl Default for ScfOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            energy_tol: 1e-12,
            residual_tol: 1e-10,
            density_tol: 1e-10,
            convergence: None,
            broken_symmetry: false,
        }
    }
}
impl ScfOptions {
    /// Remove SCF-specific keys before validating the method's option schema.
    pub fn extract(values: &mut OptionMap) -> Result<Self, HfError> {
        let mut options = Self::default();
        if let Some(value) = values.remove("scf_max_iter") {
            options.max_iter = value
                .as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&v| v > 0)
                .ok_or_else(|| {
                    HfError::InvalidInput("scf_max_iter must be a positive integer".into())
                })?;
        }
        for (key, target) in [
            ("scf_energy_tol", &mut options.energy_tol),
            ("scf_residual_tol", &mut options.residual_tol),
            ("scf_density_tol", &mut options.density_tol),
        ] {
            if let Some(value) = values.remove(key) {
                *target = value
                    .as_f64()
                    .filter(|v| v.is_finite() && *v > 0.)
                    .ok_or_else(|| {
                        HfError::InvalidInput(format!("{key} must be finite and positive"))
                    })?;
            }
        }
        if let Some(value) = values.remove("broken_symmetry") {
            options.broken_symmetry = value
                .as_bool()
                .ok_or_else(|| HfError::InvalidInput("broken_symmetry must be boolean".into()))?;
        }
        if let Some(value) = values.remove("scf_conv_algo") {
            options.convergence = Some(
                match value
                    .as_str()
                    .unwrap_or("")
                    .to_ascii_lowercase()
                    .replace('-', "_")
                    .as_str()
                {
                    "adiis_soscf" => ConvAlgo::Adiissoscf,
                    "cdiis" => ConvAlgo::Cdiis,
                    "adiis" => ConvAlgo::Adiis,
                    "adiis_cdiis" => ConvAlgo::AdiisCdiis,
                    "soscf" => ConvAlgo::Soscf,
                    "gdm" => ConvAlgo::Gdm,
                    "trah" => ConvAlgo::Trah,
                    _ => return Err(HfError::InvalidInput("unknown scf_conv_algo".into())),
                },
            );
        }
        Ok(options)
    }
    fn validate(self) -> Result<(), HfError> {
        if self.max_iter == 0
            || [self.energy_tol, self.residual_tol, self.density_tol]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.)
        {
            return Err(HfError::InvalidInput(
                "invalid SCF iteration limit or tolerance".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ScfReference {
    Rhf(HFResult),
    Uhf(UHFResult),
}
impl ScfReference {
    pub fn reference(&self) -> Reference<'_> {
        match self {
            Self::Rhf(hf) => Reference::Rhf(hf),
            Self::Uhf(hf) => Reference::Uhf(hf),
        }
    }
    pub fn iterations(&self) -> usize {
        match self {
            Self::Rhf(hf) => hf.iterations,
            Self::Uhf(hf) => hf.iterations,
        }
    }
}
struct ScfCache {
    options: ScfOptions,
    kind: RefKind,
    result: Arc<ScfReference>,
}

pub struct Molecule {
    atoms: Vec<Atom>,
    basis: Vec<ContractedGaussian>,
    basis_map: BasisSetMap,
    options: MoleculeOptions,
    cache: Mutex<Option<ScfCache>>,
}
impl Molecule {
    pub fn new(
        numbers: &[usize],
        positions: &[[f64; 3]],
        options: MoleculeOptions,
    ) -> Result<Self, HfError> {
        if numbers.is_empty()
            || numbers.len() != positions.len()
            || numbers.iter().any(|z| !(1..=118).contains(z))
            || positions.iter().flatten().any(|v| !v.is_finite())
            || options.multiplicity == 0
        {
            return Err(HfError::InvalidInput(
                "invalid atom counts, atomic numbers, coordinates or multiplicity".into(),
            ));
        }
        let scale = match options.unit {
            LengthUnit::Bohr => 1.,
            LengthUnit::Angstrom => crate::masses::ANGSTROM_TO_BOHR,
        };
        let mut atoms: Vec<_> = numbers
            .iter()
            .zip(positions)
            .map(|(&z, p)| Atom::new(atomic_number_to_symbol(z), p.map(|x| x * scale)))
            .collect();
        if atoms
            .iter()
            .flat_map(|a| a.position)
            .any(|v| !v.is_finite())
        {
            return Err(HfError::InvalidInput(
                "coordinate conversion overflow".into(),
            ));
        }
        for i in 0..atoms.len() {
            if atoms[..i].iter().any(|a| a.position == atoms[i].position) {
                return Err(HfError::InvalidInput("coincident nuclei".into()));
            }
        }
        let basis_map = crate::molecular_basis::resolve(&mut atoms, &options)?;
        let electrons =
            atoms.iter().map(|a| a.charge.round() as i64).sum::<i64>() - i64::from(options.charge);
        let spin = options.multiplicity - 1;
        if electrons <= 0
            || spin as u128 > electrons as u128
            || !(electrons as usize - spin).is_multiple_of(2)
        {
            return Err(HfError::InvalidInput(
                "electron count is incompatible with charge/multiplicity".into(),
            ));
        }
        if options.reference == ReferenceChoice::Rhf && options.multiplicity != 1 {
            return Err(HfError::InvalidInput("RHF requires multiplicity 1".into()));
        }
        let basis = try_build_basis(&atoms, &basis_map)?;
        Ok(Self {
            atoms,
            basis,
            basis_map,
            options,
            cache: Mutex::new(None),
        })
    }
    pub fn atoms(&self) -> &[Atom] {
        &self.atoms
    }
    pub fn basis(&self) -> &[ContractedGaussian] {
        &self.basis
    }
    /// Actual per-element definitions used to construct this molecule's basis.
    pub fn basis_map(&self) -> &BasisSetMap {
        &self.basis_map
    }
    /// Export the configured RHF/UHF reference, with Cartesian AO coefficient rows.
    /// This does not export a correlated or specialized HF-variant wavefunction.
    pub fn scf_wavefunction(&self, options: ScfOptions) -> Result<Value, HfError> {
        let reference = self.scf(options)?;
        crate::output::scf_snapshot(&self.atoms, &self.basis, &reference)
    }
    /// Generate Molden orbitals for the configured RHF/UHF SCF reference.
    pub fn scf_molden(&self, options: ScfOptions) -> Result<String, HfError> {
        match &*self.scf(options)? {
            ScfReference::Rhf(hf) => {
                crate::molden::try_generate_molden_string_rhf(&self.atoms, &self.basis_map, hf)
            }
            ScfReference::Uhf(hf) => {
                crate::molden::try_generate_molden_string_uhf(&self.atoms, &self.basis_map, hf)
            }
        }
    }
    fn kind(&self) -> RefKind {
        match self.options.reference {
            ReferenceChoice::Rhf => RefKind::Rhf,
            ReferenceChoice::Uhf => RefKind::Uhf,
            ReferenceChoice::Auto if self.options.multiplicity == 1 => RefKind::Rhf,
            ReferenceChoice::Auto => RefKind::Uhf,
        }
    }
    pub fn scf(&self, options: ScfOptions) -> Result<Arc<ScfReference>, HfError> {
        self.scf_in_reference(options, self.kind())
    }
    fn scf_in_reference(
        &self,
        options: ScfOptions,
        kind: RefKind,
    ) -> Result<Arc<ScfReference>, HfError> {
        options.validate()?;
        if kind == RefKind::Rhf && (self.options.multiplicity != 1 || options.broken_symmetry) {
            return Err(HfError::InvalidInput(
                "RHF requires a singlet without broken_symmetry".into(),
            ));
        }
        // UHF uses CDIIS. Reject an explicit accelerator it cannot honor.
        if kind == RefKind::Uhf
            && options
                .convergence
                .is_some_and(|algo| algo != ConvAlgo::Cdiis)
        {
            return Err(HfError::Unsupported {
                method: "UHF".into(),
                detail: "scf_conv_algo is currently fixed to cdiis".into(),
            });
        }
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| HfError::Scf("SCF cache lock poisoned".into()))?;
        if let Some(cached) = cache
            .as_ref()
            .filter(|c| c.options == options && c.kind == kind)
        {
            return Ok(Arc::clone(&cached.result));
        }
        let basis_kind = if self.options.cartesian {
            BasisKind::Cartesian
        } else {
            BasisKind::Spherical
        };
        let result = match kind {
            RefKind::Rhf => ScfReference::Rhf(
                crate::scf::run_scf_in_space_with_convergence(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    options.max_iter,
                    RhfConvergence {
                        energy: options.energy_tol,
                        residual: options.residual_tol,
                        density: options.density_tol,
                    },
                    options.convergence.unwrap_or(ConvAlgo::Adiissoscf),
                    None,
                    basis_kind,
                )
                .map_err(HfError::Scf)?,
            ),
            RefKind::Uhf => ScfReference::Uhf(
                crate::scf::run_uhf_in_space_with_convergence(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    self.options.multiplicity,
                    options.broken_symmetry,
                    options.max_iter,
                    RhfConvergence {
                        energy: options.energy_tol,
                        residual: options.residual_tol,
                        density: options.density_tol,
                    },
                    basis_kind,
                )
                .map_err(HfError::Scf)?,
            ),
        };
        if !result.reference().converged() || !result.reference().energy().is_finite() {
            return Err(HfError::Scf(
                "reference did not converge to a finite energy".into(),
            ));
        }
        let result = Arc::new(result);
        *cache = Some(ScfCache {
            options,
            kind,
            result: Arc::clone(&result),
        });
        Ok(result)
    }
    pub fn run(
        &self,
        method: &str,
        options: &OptionMap,
        properties: &[String],
    ) -> Result<CalculationResult, HfError> {
        self.run_with_limit(method, options, properties, 1_000_000)
    }
    fn run_with_limit(
        &self,
        method: &str,
        options: &OptionMap,
        properties: &[String],
        max_elements: usize,
    ) -> Result<CalculationResult, HfError> {
        let id = match MethodId::from_str(method) {
            Ok(
                MethodId::Rohf | MethodId::Ghf | MethodId::Qrhf | MethodId::Mom | MethodId::Fermi,
            ) => {
                return self.hf_variant_with_limit(method, options, properties, max_elements);
            }
            Ok(id) => id,
            Err(_) => return self.hf_variant_with_limit(method, options, properties, max_elements),
        };
        let mut values = options.clone();
        let scf_options = ScfOptions::extract(&mut values)?;
        let parsed = PostHFOptions::from_map(&values, &id.option_schemas())?;
        for property in properties {
            if !matches!(
                property.as_str(),
                "energy" | "gradient" | "hessian" | "observables" | "intermediates"
            ) {
                return Err(HfError::InvalidInput(format!(
                    "unknown property {property}"
                )));
            }
        }
        let gradient = properties.iter().any(|p| p == "gradient");
        let hessian = properties.iter().any(|p| p == "hessian");
        let kind = if id == MethodId::Uhf {
            if self.options.reference == ReferenceChoice::Rhf {
                return Err(HfError::InvalidInput(
                    "UHF method conflicts with reference=rhf".into(),
                ));
            }
            RefKind::Uhf
        } else {
            self.kind()
        };
        let is_hf = matches!(id, MethodId::Rhf | MethodId::Uhf);
        let want_observables = properties.iter().any(|p| p == "observables");
        if want_observables && !is_hf {
            return Err(HfError::Unsupported { method: id.name().into(), detail: "method-specific observables are available for HF variants; use components and states for post-HF results".into() });
        }
        if !is_hf
            && (gradient || hessian)
            && id.gradient_kind(kind, &parsed) != GradientKind::Analytic
        {
            return Err(HfError::Unsupported { method: id.name().into(), detail: "analytic response is unavailable for these options; numerical debug routes are separate".into() });
        }
        if is_hf && !values.is_empty() {
            return Err(HfError::InvalidInput(
                "HF accepts only scf_* and broken_symmetry options".into(),
            ));
        }
        if !is_hf && (gradient || hessian) && id != MethodId::Mp2 {
            return Err(HfError::Unsupported {
                method: id.name().into(),
                detail: "analytic response is not connected to the common API".into(),
            });
        }
        let reference = self.scf_in_reference(scf_options, kind)?;
        let borrowed = reference.reference();
        if !is_hf
            && (gradient || hessian)
            && (borrowed.coefficients(0).cols != self.basis.len()
                || self.atoms.iter().any(|a| a.ecp.is_some()))
        {
            return Err(HfError::Unsupported {
                method: id.name().into(),
                detail:
                    "this analytic MP2 response requires a full-rank all-electron Cartesian space"
                        .into(),
            });
        }
        let auxiliary = parsed
            .aux_basis
            .as_deref()
            .map(|name| AuxBasis::from_builtin(&self.atoms, name))
            .transpose()?;
        let want_intermediates = properties.iter().any(|p| p == "intermediates");
        let mut intermediates = TensorMap::new();
        let mut output = if is_hf {
            PostHFOutput::new(
                if kind == RefKind::Rhf { "RHF" } else { "UHF" },
                borrowed.energy(),
                borrowed.energy(),
            )?
        } else {
            let context =
                PostHFContext::new(borrowed, &self.atoms, &self.basis, auxiliary.as_ref())?;
            let solver = create_solver(id, &context, &parsed)?;
            if want_intermediates {
                let (output, tensors) = solver.compute_with_intermediates(max_elements)?;
                intermediates = tensors;
                output
            } else {
                solver.compute()?
            }
        };
        if is_hf && let ScfReference::Uhf(hf) = &*reference {
            output.components.insert("S_squared".into(), hf.s_squared);
        }
        if want_intermediates && is_hf {
            intermediates = match &*reference {
                ScfReference::Rhf(hf) => hf.intermediates(max_elements)?,
                ScfReference::Uhf(hf) => hf.intermediates(max_elements)?,
            };
        }
        let mut result = CalculationResult {
            output,
            intermediates,
            observables: if want_observables {
                match &*reference {
                    ScfReference::Rhf(hf) => hf.observables()?,
                    ScfReference::Uhf(hf) => hf.observables()?,
                }
            } else {
                ObservableMap::new()
            },
            reference: kind,
            gradient: None,
            hessian: None,
            scf_iterations: reference.iterations(),
            n_cartesian_ao: self.basis.len(),
            n_orbitals: borrowed.coefficients(0).cols,
        };
        if gradient {
            result.gradient = Some(if is_hf {
                match &*reference {
                    ScfReference::Rhf(hf) => {
                        crate::gradient::try_compute_rhf_gradient(&self.atoms, &self.basis, hf)?
                    }
                    ScfReference::Uhf(hf) => {
                        crate::gradient::try_compute_uhf_gradient(&self.atoms, &self.basis, hf)?
                    }
                }
            } else {
                match &*reference {
                    ScfReference::Rhf(hf) => {
                        crate::post_hf::mp2_gradient::compute_rhf_mp2_analytical_gradient(
                            &self.atoms,
                            &self.basis,
                            hf,
                        )
                    }
                    ScfReference::Uhf(hf) => {
                        crate::post_hf::mp2_gradient::compute_uhf_mp2_analytical_gradient(
                            &self.atoms,
                            &self.basis,
                            hf,
                        )
                    }
                }
            });
        }
        if hessian {
            result.hessian = Some(if is_hf {
                self.hf_response(&reference)?.hessian()?
            } else {
                match &*reference {
                    ScfReference::Rhf(hf) => {
                        crate::post_hf::mp2_hessian::compute_rhf_mp2_hessian_from_reference(
                            &self.atoms,
                            &self.basis,
                            hf,
                        )
                    }
                    ScfReference::Uhf(hf) => {
                        crate::post_hf::mp2_hessian::compute_uhf_mp2_hessian_from_reference(
                            &self.atoms,
                            &self.basis,
                            hf,
                        )
                    }
                }
            });
        }
        if result
            .gradient
            .iter()
            .flatten()
            .flatten()
            .any(|v| !v.is_finite())
            || result
                .hessian
                .as_ref()
                .is_some_and(|h| h.data.iter().any(|v| !v.is_finite()))
        {
            return Err(HfError::Response("nonfinite calculation derivative".into()));
        }
        result.output.validate()?;
        Ok(result)
    }
    fn hf_response<'a>(
        &'a self,
        reference: &'a ScfReference,
    ) -> Result<HfNuclearDerivativeContext<'a>, HfError> {
        match reference {
            ScfReference::Rhf(hf) => {
                HfNuclearDerivativeContext::from_rhf(&self.atoms, &self.basis, hf)
            }
            ScfReference::Uhf(hf) => {
                HfNuclearDerivativeContext::from_uhf(&self.atoms, &self.basis, hf)
            }
        }
    }
    /// Molecular HF variants use their documented default model unless an
    /// explicitly supported option is supplied. Unknown controls are errors.
    pub fn hf_variant(
        &self,
        method: &str,
        options: &OptionMap,
        properties: &[String],
    ) -> Result<CalculationResult, HfError> {
        self.hf_variant_with_limit(method, options, properties, 1_000_000)
    }
    fn hf_variant_with_limit(
        &self,
        method: &str,
        options: &OptionMap,
        properties: &[String],
        max_elements: usize,
    ) -> Result<CalculationResult, HfError> {
        use crate::hf_variants::HFVariantType as V;
        let name = method.to_ascii_lowercase().replace(['-', '_', ' '], "");
        if matches!(
            name.as_str(),
            "tdhf"
                | "rpa"
                | "sahf"
                | "sigmascf"
                | "crystalhf"
                | "cohf"
                | "atomichf"
                | "radialhf"
                | "qrhf"
                | "mom"
                | "momscf"
                | "fermi"
                | "fermihf"
                | "fermismearing"
        ) {
            return self.specialized_hf(&name, options, properties, max_elements);
        }
        let variant = crate::registry::parse_hf_variant(method)?;
        if !self.options.cartesian {
            return Err(HfError::Unsupported {
                method: method.into(),
                detail: "this HF-variant adapter currently requires cartesian=true".into(),
            });
        }
        if self.options.multiplicity != 1 && variant != V::ROHF {
            return Err(HfError::Unsupported { method: method.into(), detail: "this default-model adapter requires multiplicity=1; use its typed Rust options for other spin states".into() });
        }
        let mut values = options.clone();
        let model_keys: &[&str] = match variant {
            V::QEDHF => &["cavity_frequency", "coupling"],
            V::DFHF => &["aux_basis", "metric_tol"],
            _ => &[],
        };
        for key in options.keys() {
            if !matches!(key.as_str(), "scf_max_iter" | "scf_energy_tol")
                && !model_keys.contains(&key.as_str())
            {
                return Err(HfError::InvalidInput(format!(
                    "unknown or inapplicable {method} option: {key}"
                )));
            }
        }
        let settings = ScfOptions::extract(&mut values)?;
        let gradient = properties.iter().any(|p| p == "gradient");
        if properties.iter().any(|p| {
            !matches!(
                p.as_str(),
                "energy" | "gradient" | "observables" | "intermediates"
            )
        }) || (gradient && !matches!(variant, V::QEDHF | V::ROHF | V::GHF))
        {
            return Err(HfError::Unsupported {
                method: method.into(),
                detail:
                    "requested analytic property is not connected to this default-model adapter"
                        .into(),
            });
        }
        let number = |key: &str, default: f64| -> Result<f64, HfError> {
            values.get(key).map_or(Ok(default), |v| {
                v.as_f64()
                    .filter(|x| x.is_finite())
                    .ok_or_else(|| HfError::InvalidInput(format!("{key} must be finite")))
            })
        };
        let mut grad = None;
        let mut iterations = 0;
        let mut orbitals = self.basis.len();
        let want_observables = properties.iter().any(|p| p == "observables");
        let mut observables = ObservableMap::new();
        let want_intermediates = properties.iter().any(|p| p == "intermediates");
        let mut intermediates = TensorMap::new();
        macro_rules! capture {
            ($result:expr, $energy:ident) => {{
                let result = $result.map_err(HfError::Scf)?;
                if want_intermediates {
                    intermediates = result.intermediates(max_elements)?;
                }
                if want_observables {
                    observables = result.observables()?;
                }
                result.$energy
            }};
        }
        let energy = match variant {
            V::ROHF => {
                let result = crate::rohf::run_rohf(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    self.options.multiplicity,
                    settings.max_iter,
                    settings.energy_tol,
                )
                .map_err(HfError::Scf)?;
                if !result.converged {
                    return Err(HfError::Scf("ROHF did not converge".into()));
                }
                iterations = result.iterations;
                if want_intermediates {
                    intermediates = result.intermediates(max_elements)?;
                }
                if want_observables {
                    observables = result.observables()?;
                }
                if gradient {
                    grad = Some(crate::rohf::compute_rohf_gradient(
                        &self.atoms,
                        &self.basis,
                        &result,
                    ));
                }
                result.total_energy
            }
            V::GHF => {
                let result = crate::hf_variants::run_ghf_with_options(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    0.,
                    &crate::hf_variants::GhfOptions {
                        max_iterations: settings.max_iter,
                        energy_tol: settings.energy_tol,
                        ..Default::default()
                    },
                )
                .map_err(HfError::Scf)?;
                if !result.converged {
                    return Err(HfError::Scf("GHF did not converge".into()));
                }
                iterations = result.iterations;
                if want_intermediates {
                    intermediates = result.intermediates(max_elements)?;
                }
                if want_observables {
                    observables = result.observables()?;
                }
                if gradient {
                    grad = Some(
                        crate::hf_variants::compute_ghf_gradient(&self.atoms, &self.basis, &result)
                            .map_err(HfError::Response)?,
                    );
                }
                result.total_energy
            }
            V::QEDHF => {
                let coupling = if let Some(value) = values.get("coupling") {
                    let input = value.as_array().filter(|a| a.len() == 3).ok_or_else(|| {
                        HfError::InvalidInput("coupling must have three finite components".into())
                    })?;
                    let mut vector = [0.; 3];
                    for k in 0..3 {
                        vector[k] = input[k]
                            .as_f64()
                            .filter(|x| x.is_finite())
                            .ok_or_else(|| HfError::InvalidInput("nonfinite coupling".into()))?;
                    }
                    vector
                } else {
                    [0., 0., 0.05]
                };
                let result = crate::hf_variants::run_qed_hf_with_options(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    &crate::hf_variants::QedHfOptions {
                        cavity_frequency: number("cavity_frequency", 0.1)?,
                        coupling,
                        max_iterations: settings.max_iter,
                        energy_tol: settings.energy_tol,
                        ..Default::default()
                    },
                )
                .map_err(HfError::Scf)?;
                iterations = result.iterations;
                if want_intermediates {
                    intermediates = result.intermediates(max_elements)?;
                }
                if want_observables {
                    observables = result.observables()?;
                }
                if gradient {
                    grad = Some(
                        crate::hf_variants::compute_qed_hf_gradient(
                            &self.atoms,
                            &self.basis,
                            &result,
                        )
                        .map_err(HfError::Response)?,
                    );
                }
                result.total_energy
            }
            V::DFHF => {
                let name = values.get("aux_basis").map_or(Ok("def2-svp-ri"), |v| {
                    v.as_str()
                        .ok_or_else(|| HfError::InvalidInput("aux_basis must be a name".into()))
                })?;
                let aux = AuxBasis::from_builtin(&self.atoms, name)?;
                let result = crate::hf_variants::run_df_hf_with_aux(
                    &self.atoms,
                    &self.basis,
                    &aux.functions,
                    self.options.charge,
                    &crate::hf_variants::DFHFOptions {
                        max_iterations: settings.max_iter,
                        energy_tolerance: settings.energy_tol,
                        metric_cutoff: number("metric_tol", 1e-10)?,
                        ..Default::default()
                    },
                )
                .map_err(HfError::Scf)?;
                if !result.converged {
                    return Err(HfError::Scf("DF-HF did not converge".into()));
                }
                iterations = result.iterations;
                if want_intermediates {
                    intermediates = result.intermediates(max_elements)?;
                }
                if want_observables {
                    observables = result.observables()?;
                }
                orbitals = result.mo_coefficients.cols;
                result.total_energy
            }
            V::PUHF => capture!(
                crate::run_puhf(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    settings.max_iter,
                    settings.energy_tol
                ),
                projected_energy
            ),
            V::HFB => capture!(
                crate::run_hfb(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    0.05,
                    settings.max_iter,
                    settings.energy_tol
                ),
                total_hfb_energy
            ),
            V::TCSCF => {
                let ne = self.atoms.iter().map(|a| a.charge).sum::<f64>().round() as i64
                    - i64::from(self.options.charge);
                let occ = (ne.max(1) as usize).div_ceil(2);
                capture!(
                    crate::run_tcscf(
                        &self.atoms,
                        &self.basis,
                        self.options.charge,
                        occ.saturating_sub(1),
                        occ.min(self.basis.len() - 1),
                        settings.max_iter,
                        settings.energy_tol
                    ),
                    tcscf_energy
                )
            }
            V::NEOHF => {
                let index = self
                    .atoms
                    .iter()
                    .position(|a| a.atomic_number == 1)
                    .ok_or_else(|| {
                        HfError::InvalidInput("NEO-HF requires at least one hydrogen atom".into())
                    })?;
                let quantum = vec![self.atoms[index].clone()];
                let classical: Vec<_> = self
                    .atoms
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != index)
                    .map(|(_, a)| a.clone())
                    .collect();
                let qb = try_build_basis(&quantum, &self.basis_map)?;
                capture!(
                    crate::run_neo_hf(
                        &classical,
                        &self.basis,
                        &qb,
                        self.options.charge,
                        1,
                        settings.max_iter,
                        settings.energy_tol
                    ),
                    total_energy
                )
            }
            V::RelativisticHF | V::Dkh2HF | V::ZoraHF => {
                let model = match variant {
                    V::Dkh2HF => crate::RelativisticModel::DKH2,
                    V::ZoraHF => crate::RelativisticModel::ZORA,
                    _ => crate::RelativisticModel::X2C,
                };
                capture!(
                    crate::run_relativistic_hf(
                        &self.atoms,
                        &self.basis,
                        self.options.charge,
                        model,
                        settings.max_iter,
                        settings.energy_tol
                    ),
                    total_energy
                )
            }
            V::CDHF => capture!(
                crate::run_cd_hf(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    1e-4,
                    settings.max_iter,
                    settings.energy_tol
                ),
                total_energy
            ),
            V::ComplexHF | V::ComplexUHF | V::ComplexGHF => {
                let ne = self.atoms.iter().map(|a| a.charge).sum::<f64>().round() as i64
                    - i64::from(self.options.charge);
                let reference = match variant {
                    V::ComplexUHF => crate::ComplexHfReference::Unrestricted,
                    V::ComplexGHF => crate::ComplexHfReference::Generalized,
                    _ => crate::ComplexHfReference::Restricted,
                };
                capture!(
                    crate::run_complex_hf_with_options(
                        &self.atoms,
                        &self.basis,
                        self.options.charge,
                        &crate::ComplexHfOptions {
                            reference,
                            multiplicity: if reference == crate::ComplexHfReference::Restricted {
                                1
                            } else {
                                (ne.rem_euclid(2) + 1) as usize
                            },
                            magnetic_field: [0., 0., 0.02],
                            max_iterations: settings.max_iter,
                            energy_tol: settings.energy_tol,
                            residual_tol: settings.energy_tol.sqrt().min(1e-8),
                            density_tol: settings.energy_tol.sqrt().min(1e-8),
                            ..Default::default()
                        }
                    ),
                    total_energy
                )
            }
            V::LHF => capture!(
                crate::run_lhf_scf(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    settings.max_iter,
                    settings.energy_tol
                ),
                total_energy
            ),
            V::OEP => capture!(
                crate::run_oep_hf(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    &crate::OepOptions {
                        max_iterations: settings.max_iter,
                        energy_tol: settings.energy_tol,
                        ..Default::default()
                    }
                ),
                total_energy
            ),
            V::PositronicHF => capture!(
                crate::run_positronic_hf(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    settings.max_iter,
                    settings.energy_tol
                ),
                total_energy
            ),
            V::DCHF => capture!(
                crate::run_dc_hf(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    &crate::DCOptions {
                        max_iterations: settings.max_iter,
                        energy_threshold: settings.energy_tol,
                        ..Default::default()
                    }
                ),
                total_energy
            ),
            V::FMO | V::FMO3 => {
                if matches!(variant, V::FMO | V::FMO3) && self.options.charge != 0 {
                    return Err(HfError::Unsupported {
                        method: method.into(),
                        detail: "charged FMO requires explicit typed fragment charges".into(),
                    });
                }
                let fragments =
                    crate::auto_partition_fragments(&self.atoms, (self.atoms.len() / 2).max(1));
                capture!(
                    crate::run_fmo(
                        &self.atoms,
                        &self.basis,
                        &fragments,
                        &crate::FMOOptions {
                            order: if variant == V::FMO3 { 3 } else { 2 },
                            max_iterations: settings.max_iter,
                            tolerance: settings.energy_tol,
                            ..Default::default()
                        }
                    ),
                    total_energy
                )
            }
        };
        if grad.iter().flatten().flatten().any(|v| !v.is_finite()) {
            return Err(HfError::Response("nonfinite HF-variant gradient".into()));
        }
        let output = PostHFOutput::new(format!("{variant:?}"), energy, energy)?;
        Ok(CalculationResult {
            output,
            observables,
            intermediates,
            reference: RefKind::Rhf,
            gradient: grad,
            hessian: None,
            scf_iterations: iterations,
            n_cartesian_ao: self.basis.len(),
            n_orbitals: orbitals,
        })
    }
    fn specialized_hf(
        &self,
        name: &str,
        options: &OptionMap,
        properties: &[String],
        max_elements: usize,
    ) -> Result<CalculationResult, HfError> {
        if !self.options.cartesian
            || properties
                .iter()
                .any(|p| !matches!(p.as_str(), "energy" | "observables" | "intermediates"))
        {
            return Err(HfError::Unsupported {
                method: name.into(),
                detail:
                    "this specialized adapter accepts Cartesian energy and observables requests"
                        .into(),
            });
        }
        let keys: &[&str] = match name {
            "tdhf" | "rpa" => &["n_singlets", "n_triplets", "root"],
            "sahf" => &["configurations", "weights"],
            "sigmascf" => &["configuration", "target_energy"],
            "crystalhf" | "cohf" => &["lattice_constant", "n_kpoints"],
            "atomichf" | "radialhf" => &["n_grid_points", "r_max"],
            "mom" | "momscf" => &["occupied_orbitals"],
            "fermi" | "fermihf" | "fermismearing" => &["thermal_energy"],
            _ => &[],
        };
        for key in options.keys() {
            if !matches!(key.as_str(), "scf_max_iter" | "scf_energy_tol")
                && !keys.contains(&key.as_str())
            {
                return Err(HfError::InvalidInput(format!(
                    "unknown or inapplicable {name} option: {key}"
                )));
            }
        }
        let mut values = options.clone();
        let scf = ScfOptions::extract(&mut values)?;
        let integer = |key: &str, default: usize| -> Result<usize, HfError> {
            values.get(key).map_or(Ok(default), |v| {
                v.as_u64()
                    .and_then(|x| usize::try_from(x).ok())
                    .ok_or_else(|| {
                        HfError::InvalidInput(format!("{key} must be a nonnegative integer"))
                    })
            })
        };
        let number = |key: &str, default: Option<f64>| -> Result<f64, HfError> {
            values
                .get(key)
                .and_then(Value::as_f64)
                .or(default.filter(|_| !values.contains_key(key)))
                .filter(|x| x.is_finite())
                .ok_or_else(|| {
                    HfError::InvalidInput(format!(
                        "{key} must be finite and supplied when required"
                    ))
                })
        };
        let ne = self.atoms.iter().map(|a| a.charge).sum::<f64>().round() as i64
            - i64::from(self.options.charge);
        let configuration = |value: &Value| -> Result<crate::StateConfiguration, HfError> {
            let object = value.as_object().ok_or_else(|| {
                HfError::InvalidInput(
                    "configuration must contain alpha and beta orbital-index lists".into(),
                )
            })?;
            if object.keys().any(|k| k != "alpha" && k != "beta") {
                return Err(HfError::InvalidInput("unknown configuration key".into()));
            }
            let indices =
                |key: &str| -> Result<Vec<usize>, HfError> {
                    serde_json::from_value(object.get(key).cloned().ok_or_else(|| {
                        HfError::InvalidInput(format!("missing configuration {key}"))
                    })?)
                    .map_err(|e| HfError::InvalidInput(e.to_string()))
                };
            let c = crate::StateConfiguration::new(indices("alpha")?, indices("beta")?);
            if (c.alpha.len() + c.beta.len()) as i64 != ne {
                return Err(HfError::InvalidInput(
                    "configuration electron count differs from molecular charge".into(),
                ));
            }
            Ok(c)
        };
        let want = properties.iter().any(|p| p == "observables");
        let mut observables = ObservableMap::new();
        let want_intermediates = properties.iter().any(|p| p == "intermediates");
        let mut intermediates = TensorMap::new();
        let iterations;
        let mut n_orbitals = self.basis.len();
        let mut reference = self.kind();
        macro_rules! collect {
            ($r:ident) => {
                if want_intermediates {
                    intermediates = $r.intermediates(max_elements)?;
                }
                if want {
                    observables = $r.observables()?;
                }
            };
        }
        let output = match name {
            "tdhf" | "rpa" => {
                if self.options.multiplicity != 1 || self.options.reference == ReferenceChoice::Uhf
                {
                    return Err(HfError::InvalidInput(
                        "TDHF requires an RHF reference".into(),
                    ));
                }
                let hf = self.scf_in_reference(scf, RefKind::Rhf)?;
                let ScfReference::Rhf(hf) = &*hf else {
                    unreachable!()
                };
                let r = crate::run_tdhf_from_reference(
                    hf,
                    &self.basis,
                    integer("n_singlets", 1)?,
                    integer("n_triplets", 0)?,
                )
                .map_err(HfError::Scf)?;
                collect!(r);
                reference = RefKind::Rhf;
                iterations = hf.iterations;
                n_orbitals = hf.mo_coefficients.cols;
                let mut output = PostHFOutput::new("TDHF", hf.total_energy, hf.total_energy)?;
                output.states = r
                    .excited_states
                    .iter()
                    .map(|s| crate::output::State {
                        energy: hf.total_energy + s.excitation_energy_hartree,
                        s_squared: Some(if s.is_triplet { 2. } else { 0. }),
                        weight: 0.,
                        converged: true,
                    })
                    .collect();
                output.select_root(integer("root", 0)?)?;
                output.states[output.root].weight = 1.;
                output
            }
            "sahf" => {
                let configs = values
                    .get("configurations")
                    .and_then(Value::as_array)
                    .ok_or_else(|| HfError::InvalidInput("sahf requires configurations".into()))?
                    .iter()
                    .map(configuration)
                    .collect::<Result<Vec<_>, _>>()?;
                let weights: Vec<f64> = serde_json::from_value(
                    values
                        .get("weights")
                        .cloned()
                        .ok_or_else(|| HfError::InvalidInput("sahf requires weights".into()))?,
                )
                .map_err(|e| HfError::InvalidInput(e.to_string()))?;
                let r = crate::run_state_averaged_hf_with_options(
                    &self.atoms,
                    &self.basis,
                    &configs,
                    &weights,
                    &crate::SaHfOptions {
                        max_iterations: scf.max_iter,
                        energy_tol: scf.energy_tol,
                        ..Default::default()
                    },
                )
                .map_err(HfError::Scf)?;
                collect!(r);
                iterations = r.iterations;
                reference = RefKind::Rhf;
                PostHFOutput::new("SA-HF", r.sa_energy, r.sa_energy)?
            }
            "sigmascf" => {
                let config = configuration(values.get("configuration").ok_or_else(|| {
                    HfError::InvalidInput("sigma-scf requires configuration".into())
                })?)?;
                let r = crate::run_sigma_scf_with_options(
                    &self.atoms,
                    &self.basis,
                    number("target_energy", None)?,
                    &config,
                    &crate::SigmaScfOptions {
                        max_iterations: scf.max_iter,
                        variance_tol: scf.energy_tol,
                        ..Default::default()
                    },
                )
                .map_err(HfError::Scf)?;
                collect!(r);
                iterations = r.iterations;
                reference = RefKind::Uhf;
                PostHFOutput::new("sigma-SCF", r.total_energy, r.total_energy)?
            }
            "crystalhf" | "cohf" => {
                if self.options.multiplicity != 1 {
                    return Err(HfError::InvalidInput(
                        "crystal HF requires a closed-shell cell".into(),
                    ));
                }
                let r = crate::run_crystal_orbital_hf(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    number("lattice_constant", None)?,
                    integer("n_kpoints", 16)?,
                    scf.max_iter,
                    scf.energy_tol,
                )
                .map_err(HfError::Scf)?;
                collect!(r);
                iterations = r.iterations;
                reference = RefKind::Rhf;
                PostHFOutput::new(
                    "Crystal-HF",
                    r.total_energy_per_cell,
                    r.total_energy_per_cell,
                )?
            }
            "atomichf" | "radialhf" => {
                if self.atoms.len() != 1 || self.atoms[0].ecp.is_some() || !(1..=2).contains(&ne) {
                    return Err(HfError::InvalidInput(
                        "radial HF requires one all-electron atom/ion with one or two electrons"
                            .into(),
                    ));
                }
                let r = crate::run_numerical_atomic_1s_hf(
                    self.atoms[0].atomic_number,
                    ne as usize,
                    &crate::NumericalAtomicOptions {
                        n_grid_points: integer("n_grid_points", 4096)?,
                        max_iterations: scf.max_iter,
                        energy_tol: scf.energy_tol,
                        r_max_bohr: values
                            .get("r_max")
                            .map(|_| number("r_max", None))
                            .transpose()?,
                        ..Default::default()
                    },
                )
                .map_err(HfError::Scf)?;
                collect!(r);
                iterations = r.iterations;
                n_orbitals = 1;
                PostHFOutput::new("Radial-HF", r.total_energy, r.total_energy)?
            }
            "qrhf" => {
                let r = crate::run_qrhf(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    self.options.multiplicity,
                    scf.max_iter,
                    scf.energy_tol,
                )
                .map_err(HfError::Scf)?;
                if !r.converged {
                    return Err(HfError::Scf("QRHF did not converge".into()));
                }
                collect!(r);
                iterations = r.iterations;
                reference = RefKind::Rhf;
                PostHFOutput::new("QRHF", r.total_energy, r.total_energy)?
            }
            "mom" | "momscf" => {
                if self.options.multiplicity != 1 {
                    return Err(HfError::InvalidInput(
                        "MOM adapter requires a closed-shell configuration".into(),
                    ));
                }
                let occ: Vec<usize> =
                    serde_json::from_value(values.get("occupied_orbitals").cloned().ok_or_else(
                        || HfError::InvalidInput("MOM requires occupied_orbitals".into()),
                    )?)
                    .map_err(|e| HfError::InvalidInput(e.to_string()))?;
                let r = crate::run_scf_mom(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    &occ,
                    scf.max_iter,
                    scf.energy_tol,
                )
                .map_err(HfError::Scf)?;
                if !r.converged {
                    return Err(HfError::Scf("MOM did not converge".into()));
                }
                collect!(r);
                iterations = r.iterations;
                reference = RefKind::Rhf;
                PostHFOutput::new("MOM", r.total_energy, r.total_energy)?
            }
            _ => {
                if self.options.multiplicity != 1 {
                    return Err(HfError::InvalidInput(
                        "Fermi-HF adapter requires a closed-shell ensemble".into(),
                    ));
                }
                let r = crate::run_scf_fermi_smearing(
                    &self.atoms,
                    &self.basis,
                    self.options.charge,
                    number("thermal_energy", None)? * 315775.02480407,
                    scf.max_iter,
                    scf.energy_tol,
                )
                .map_err(HfError::Scf)?;
                if !r.hf.converged {
                    return Err(HfError::Scf("Fermi-HF did not converge".into()));
                }
                collect!(r);
                iterations = r.hf.iterations;
                reference = RefKind::Rhf;
                PostHFOutput::new("Fermi-HF", r.hf.total_energy, r.hf.total_energy)?
            }
        };
        output.validate()?;
        Ok(CalculationResult {
            output,
            observables,
            intermediates,
            reference,
            gradient: None,
            hessian: None,
            scf_iterations: iterations,
            n_cartesian_ao: self.basis.len(),
            n_orbitals,
        })
    }
    /// Copy retained method tensors with a total numerical-element bound.
    /// Empty output means this result does not retain exportable tensors.
    pub fn intermediates(
        &self,
        method: &str,
        options: &OptionMap,
        max_elements: usize,
    ) -> Result<TensorMap, HfError> {
        crate::output::TensorExport::new(max_elements)?;
        Ok(self
            .run_with_limit(method, options, &["intermediates".into()], max_elements)?
            .intermediates)
    }
    /// Return method-specific physical quantities in atomic units on request.
    pub fn hf_observables(
        &self,
        method: &str,
        options: &OptionMap,
    ) -> Result<ObservableMap, HfError> {
        Ok(self
            .run(method, options, &["observables".into()])?
            .observables)
    }
    pub fn nuclear_derivative(
        &self,
        coordinates: &[usize],
        options: ScfOptions,
    ) -> Result<f64, HfError> {
        if coordinates.is_empty() || coordinates.len() > 4 {
            return Err(HfError::InvalidInput(
                "HF nuclear derivative order must be 1 through 4".into(),
            ));
        }
        let reference = self.scf(options)?;
        self.hf_response(&reference)?
            .coordinate_derivative(coordinates)
    }
    pub fn directional_derivative(
        &self,
        directions: &[Vec<f64>],
        options: ScfOptions,
    ) -> Result<f64, HfError> {
        if directions.is_empty() || directions.len() > 4 {
            return Err(HfError::InvalidInput(
                "expected 1..4 nuclear directions".into(),
            ));
        }
        let directions = directions
            .iter()
            .map(|d| crate::hf_derivatives::direction_from_flat(d, self.atoms.len()))
            .collect::<Result<Vec<_>, _>>()?;
        let reference = self.scf(options)?;
        self.hf_response(&reference)?.derivative(&directions)
    }
}

pub struct CalculationResult {
    pub output: PostHFOutput,
    pub observables: ObservableMap,
    pub intermediates: TensorMap,
    pub reference: RefKind,
    pub gradient: Option<Vec<[f64; 3]>>,
    pub hessian: Option<Matrix>,
    pub scf_iterations: usize,
    pub n_cartesian_ao: usize,
    pub n_orbitals: usize,
}
fn matrix_rows(matrix: &Matrix) -> Vec<Vec<f64>> {
    matrix
        .data
        .chunks(matrix.cols.max(1))
        .map(|row| row.to_vec())
        .collect()
}
impl CalculationResult {
    pub fn to_json(&self) -> Value {
        let o = &self.output;
        json!({ "method": o.method, "reference": if self.reference == RefKind::Rhf { "rhf" } else { "uhf" },
            "e_scf": o.e_scf, "e_corr": o.e_corr, "e_total": o.e_total, "components": o.components,
            "component_units": o.components.keys().map(|k| (k.clone(), component_unit(k).map(|u| u.to_json()))).collect::<BTreeMap<_,_>>(),
            "observables": self.observables, "intermediates": self.intermediates,
            "states": o.states.iter().map(|s| json!({"energy": s.energy, "s_squared": s.s_squared, "weight": s.weight, "converged": s.converged})).collect::<Vec<_>>(),
            "root": o.root, "one_rdm": o.one_rdm.as_ref().map(matrix_rows), "gradient": self.gradient,
            "hessian": self.hessian.as_ref().map(matrix_rows), "gradient_kind": self.gradient.as_ref().map(|_| "analytic"),
            "hessian_kind": self.hessian.as_ref().map(|_| "analytic"), "scf_iterations": self.scf_iterations,
            "n_cartesian_ao": self.n_cartesian_ao, "n_orbitals": self.n_orbitals,
            "units": {"energy": "hartree", "gradient": "hartree/bohr", "hessian": "hartree/bohr^2", "one_rdm": "MO"} })
    }
}

pub fn run(
    method: &str,
    numbers: &[usize],
    positions: &[[f64; 3]],
    molecule: MoleculeOptions,
    options: &OptionMap,
    properties: &[String],
) -> Result<CalculationResult, HfError> {
    Molecule::new(numbers, positions, molecule)?.run(method, options, properties)
}

pub fn method_catalog() -> Value {
    let defaults = PostHFOptions::default();
    json!(
        MethodId::ALL
            .iter()
            .map(|id| {
                let mut keys = if matches!(id, MethodId::Rhf | MethodId::Uhf) {
                    vec![
                        "scf_max_iter",
                        "scf_energy_tol",
                        "scf_residual_tol",
                        "scf_density_tol",
                        "scf_conv_algo",
                        "broken_symmetry",
                    ]
                } else if matches!(
                    id,
                    MethodId::Rohf
                        | MethodId::Ghf
                        | MethodId::Qrhf
                        | MethodId::Mom
                        | MethodId::Fermi
                ) {
                    let mut hf = vec!["scf_max_iter", "scf_energy_tol"];
                    if *id == MethodId::Mom {
                        hf.push("occupied_orbitals");
                    }
                    if *id == MethodId::Fermi {
                        hf.push("thermal_energy");
                    }
                    hf
                } else {
                    let mut post = crate::options::COMMON_OPTIONS.to_vec();
                    for schema in id.option_schemas() {
                        post.extend_from_slice(schema);
                    }
                    post
                };
                if !matches!(id,MethodId::Rhf|MethodId::Uhf|MethodId::Rohf|MethodId::Ghf|MethodId::Qrhf|MethodId::Mom|MethodId::Fermi) {
                    keys.extend_from_slice(&["scf_max_iter","scf_energy_tol","scf_residual_tol","scf_density_tol","scf_conv_algo","broken_symmetry"]);
                }
                keys.sort_unstable();
                keys.dedup();
                json!({"name": id.name(), "family": format!("{:?}", id.family()), "options": keys,
            "option_details": keys.iter().map(|k| api_option_definition(k).expect("registered option metadata")).collect::<Vec<_>>(),
            "rhf_gradient": format!("{:?}", id.gradient_kind(RefKind::Rhf, &defaults)),
            "uhf_gradient": format!("{:?}", id.gradient_kind(RefKind::Uhf, &defaults))})
            })
            .collect::<Vec<_>>()
    )
}

/// Describe every common input option, including its actual typed default.
pub fn option_catalog() -> Value {
    let catalog = method_catalog();
    let mut entries = BTreeMap::new();
    for method in catalog.as_array().expect("method catalog is an array") {
        for entry in method["option_details"]
            .as_array()
            .expect("option details are an array")
        {
            entries.insert(
                entry["name"].as_str().expect("option name").to_string(),
                entry.clone(),
            );
        }
    }
    json!(entries.into_values().collect::<Vec<_>>())
}

/// The common HF-model boundary, including adapter defaults and input limits.
pub fn hf_variant_catalog() -> Value {
    let rows: &[(&str, &[&str], &str)] = &[
        (
            "rohf",
            &[],
            "The supplied multiplicity defines the open-shell populations.",
        ),
        (
            "ghf",
            &[],
            "Real spinor determinant; the adapter supplies multiplicity=1 and zero spin bias.",
        ),
        (
            "qrhf",
            &[],
            "Quasi-restricted natural orbitals with the supplied multiplicity.",
        ),
        (
            "mom",
            &["occupied_orbitals"],
            "Closed-shell maximum-overlap determinant with explicit occupied guess indices.",
        ),
        (
            "fermi-hf",
            &["thermal_energy"],
            "Closed-shell finite-temperature ensemble; thermal_energy is kBT in Hartree.",
        ),
        (
            "puhf",
            &[],
            "Default singlet projection; exported raw-UHF tensors are not a projected density.",
        ),
        (
            "hfb",
            &[],
            "Default pairing strength 0.05 Hartree; typed HfbOptions exposes the pairing model.",
        ),
        (
            "tcscf",
            &[],
            "Active pair is the highest occupied and lowest unoccupied guess orbital.",
        ),
        (
            "qed-hf",
            &["cavity_frequency", "coupling"],
            "Single coherent photon product state; no photon zero-point constant or electron-photon correlation.",
        ),
        (
            "df-hf",
            &["aux_basis", "metric_tol"],
            "Coulomb-metric RI-JK with an explicit physical auxiliary basis.",
        ),
        (
            "cd-hf",
            &[],
            "Default Cholesky tolerance 1e-4 Hartree; typed API changes the threshold.",
        ),
        (
            "x2c",
            &[],
            "Spin-free one-electron X2C with untransformed two-electron integrals.",
        ),
        (
            "dkh2",
            &[],
            "Spin-free one-electron DKH2 with untransformed two-electron integrals.",
        ),
        (
            "zora",
            &[],
            "Recognized name; the required grid Hamiltonian is not implemented and returns an error.",
        ),
        (
            "complex-hf",
            &[],
            "Restricted complex determinant in magnetic field [0,0,0.02] atomic units.",
        ),
        (
            "cuhf",
            &[],
            "Unrestricted complex determinant in magnetic field [0,0,0.02] atomic units.",
        ),
        (
            "cghf",
            &[],
            "Generalized complex determinant in magnetic field [0,0,0.02] atomic units.",
        ),
        (
            "lhf",
            &[],
            "Self-consistent grid LHF/CEDA; typed LhfOptions exposes grid/convergence controls.",
        ),
        (
            "oep",
            &[],
            "Finite Gaussian-potential OEP; typed OepOptions controls the potential space.",
        ),
        (
            "neo-hf",
            &[],
            "First hydrogen is quantized in the selected electronic hydrogen basis; use typed API for a proton basis.",
        ),
        (
            "positronic-hf",
            &[],
            "Default one-positron product mean field in the supplied Gaussian working space.",
        ),
        (
            "dc-hf",
            &[],
            "Default consecutive subsystems; typed API supplies central/buffer partitions.",
        ),
        (
            "fmo2",
            &[],
            "Neutral consecutive fragments of max(N_atoms/2,1) atoms; typed FMOFragment specifies chemical partitions.",
        ),
        (
            "fmo3",
            &[],
            "Same default fragments as FMO2; connected trimers require at least three fragments.",
        ),
        (
            "tdhf",
            &["n_singlets", "n_triplets", "root"],
            "RHF linear-response RPA; e_total selects a vertical excited state.",
        ),
        (
            "sahf",
            &["configurations", "weights"],
            "Common-orbital determinant ensemble with explicit populations and nonnegative weights summing to one.",
        ),
        (
            "sigma-scf",
            &["configuration", "target_energy"],
            "Determinant variance optimization about an explicit target energy.",
        ),
        (
            "crystal-hf",
            &["lattice_constant", "n_kpoints"],
            "Neutral closed-shell one-dimensional chain along z; energy is per cell.",
        ),
        (
            "radial-hf",
            &["n_grid_points", "r_max"],
            "Single all-electron atom/ion with one or two 1s electrons; Gaussian basis is unused by the radial solver.",
        ),
    ];
    let qed = crate::hf_variants::QedHfOptions::default();
    json!(rows.iter().map(|(name, keys, model)| {
        let mut definitions = vec![api_option_definition("scf_max_iter").unwrap(),api_option_definition("scf_energy_tol").unwrap()];
        if *name == "sigma-scf" {
            definitions[1]["unit"] = json!("hartree^2");
            definitions[1]["description"] = json!("Legacy common-adapter name for the positive variance-change threshold; this sigma-SCF model has no ordinary SCF energy-change criterion. Typed SigmaScfOptions exposes variance_tol and gradient_tol independently.");
        }
        for key in *keys {
            let (kind, default, unit, description) = match *key {
                "cavity_frequency" => ("real",json!(qed.cavity_frequency),"hartree","Strictly positive photon energy."),
                "coupling" => ("real[3]",json!(qed.coupling),"hartree^0.5/bohr","Finite Cartesian length-gauge coupling vector."),
                "aux_basis" => ("string",json!("def2-svp-ri"),"1","Built-in physical Coulomb-fitting basis name."),
                "metric_tol" => ("real",json!(1e-10),"hartree*bohr^3","Positive absolute auxiliary Coulomb-metric eigenvalue cutoff."),
                "n_singlets" => ("integer",json!(1),"1","Nonnegative number of singlet response roots; total requested roots must be positive."),
                "n_triplets" => ("integer",json!(0),"1","Nonnegative number of triplet response roots."),
                "configurations" => ("configuration list",Value::Null,"1","Required determinants, each with distinct zero-based alpha and beta occupied indices and the molecular electron count."),
                "weights" => ("real list",Value::Null,"1","Required finite nonnegative weights, one per configuration, summing to one within 1e-12."),
                "configuration" => ("configuration",Value::Null,"1","Required determinant with alpha/beta occupied indices and the molecular electron count."),
                "target_energy" => ("real",Value::Null,"hartree","Required finite target energy for variance optimization."),
                "lattice_constant" => ("real",Value::Null,"bohr","Required positive z lattice period; the adapter validates its real-space/k-point model."),
                "n_kpoints" => ("integer",json!(16),"1","Positive nonaliasing one-dimensional k mesh."),
                "n_grid_points" => ("integer",json!(4096),"1","Radial grid size, at least 16 points."),
                "r_max" => ("real",Value::Null,"bohr","Omission uses the solver's charge-scaled radial boundary; a supplied boundary must be positive."),
                _ => { definitions.push(api_option_definition(key).unwrap()); continue; }
            };
            definitions.push(json!({"name":key,"type":kind,"default":default,"unit":unit,"description":description,"null_input_allowed":false}));
        }
        json!({"name":name,"options":definitions.iter().map(|x|x["name"].clone()).collect::<Vec<_>>(),"option_details":definitions,"cartesian_required":true,"model":model})
    }).collect::<Vec<_>>())
}

fn api_option_definition(name: &str) -> Option<Value> {
    let scf = ScfOptions::default();
    let (kind, default, unit, description) = match name {
        "scf_max_iter" => (
            "integer",
            json!(scf.max_iter),
            "1",
            "Positive SCF iteration limit.",
        ),
        "scf_energy_tol" => (
            "real",
            json!(scf.energy_tol),
            "hartree",
            "Positive SCF energy-change threshold.",
        ),
        "scf_residual_tol" => (
            "real",
            json!(scf.residual_tol),
            "hartree",
            "Positive SCF FDS-SDF residual threshold.",
        ),
        "scf_density_tol" => (
            "real",
            json!(scf.density_tol),
            "1",
            "Positive SCF density-change threshold.",
        ),
        "scf_conv_algo" => (
            "string",
            Value::Null,
            "1",
            "Omission selects RHF ADIIS/SOSCF or UHF CDIIS; explicit unsupported accelerators are errors.",
        ),
        "broken_symmetry" => (
            "boolean",
            json!(scf.broken_symmetry),
            "1",
            "Request an unrestricted broken-symmetry initial determinant; requires reference=uhf.",
        ),
        "occupied_orbitals" => (
            "integer list",
            Value::Null,
            "1",
            "Required distinct zero-based MOM occupied guess-orbital indices, with the molecular closed-shell electron count.",
        ),
        "thermal_energy" => (
            "real",
            Value::Null,
            "hartree",
            "Required positive Fermi-HF kBT; the legacy typed Rust entry point instead labels its temperature argument in kelvin.",
        ),
        _ => return crate::options::option_definition(name),
    };
    let choices: &[&str] = if name == "scf_conv_algo" {
        &[
            "adiis_soscf",
            "cdiis",
            "adiis",
            "adiis_cdiis",
            "soscf",
            "gdm",
            "trah",
        ]
    } else {
        &[]
    };
    Some(
        json!({"name":name,"type":kind,"default":default,"unit":unit,"description":description,"choices":choices,"null_input_allowed":false}),
    )
}

impl Molecule {
    /// Inspect bare scalar integrals. MO derivatives hold coefficients fixed;
    /// nuclear energy response is provided by the separate response APIs.
    pub fn integrals(
        &self,
        request: &IntegralRequest,
        settings: ScfOptions,
    ) -> Result<IntegralBlock, HfError> {
        use crate::output::TensorData;
        settings.validate()?;
        if request.derivative_coordinates.len() > 4
            || request
                .derivative_coordinates
                .iter()
                .any(|&i| i >= 3 * self.atoms.len())
        {
            return Err(HfError::InvalidInput(
                "integral derivatives require zero through four valid nuclear coordinates".into(),
            ));
        }
        if !request.metric_cutoff.is_finite() || request.metric_cutoff <= 0. {
            return Err(HfError::InvalidInput(
                "metric_cutoff must be finite and positive".into(),
            ));
        }
        if request.operator.starts_with("df_") {
            return self.ri_integrals(request);
        }
        let operator = request.operator.as_str();
        let arity = if operator == "eri" {
            4
        } else if matches!(
            operator,
            "overlap"
                | "kinetic"
                | "nuclear_attraction"
                | "ecp"
                | "core_hamiltonian"
                | "fock"
                | "fock_alpha"
                | "fock_beta"
        ) {
            2
        } else {
            return Err(HfError::InvalidInput(format!(
                "unknown integral operator {operator}"
            )));
        };
        if request.aux_basis.is_some() {
            return Err(HfError::InvalidInput(
                "aux_basis applies only to df_* integral operators".into(),
            ));
        }
        let representation = request.representation.as_str();
        if !matches!(
            representation,
            "cartesian_ao"
                | "physical_ao"
                | "mo_alpha"
                | "mo_beta"
                | "mo_alpha_beta"
                | "mo_beta_alpha"
                | "custom"
        ) {
            return Err(HfError::InvalidInput(
                "unknown integral representation".into(),
            ));
        }
        if arity != 4 && matches!(representation, "mo_alpha_beta" | "mo_beta_alpha") {
            return Err(HfError::InvalidInput(
                "mixed-spin spatial MO representations require four-index Coulomb integrals".into(),
            ));
        }
        if representation != "custom" && !request.coefficients.is_empty() {
            return Err(HfError::InvalidInput(
                "explicit coefficients require representation=custom".into(),
            ));
        }
        let n = self.basis.len();
        if representation == "custom"
            && (request.coefficients.is_empty()
                || (request.coefficients.len() != 1 && request.coefficients.len() != arity))
        {
            return Err(HfError::InvalidInput("custom representation requires one shared coefficient matrix or one per integral axis".into()));
        }
        for c in &request.coefficients {
            if c.rows != n
                || c.cols == 0
                || c.rows.checked_mul(c.cols) != Some(c.data.len())
                || c.data.iter().any(|x| !x.is_finite())
            {
                return Err(HfError::InvalidInput("custom coefficient matrices require finite Cartesian-AO rows and nonempty columns".into()));
            }
        }
        let fock = operator.starts_with("fock");
        if fock && !request.derivative_coordinates.is_empty() {
            return Err(HfError::Unsupported {method:operator.into(),detail:"a self-consistent Fock derivative requires density response; inspect bare core/ERI derivatives or use the HF energy-response API".into()});
        }
        let physical = if representation == "physical_ao" {
            workspace_limit(16 * (n as u128).pow(2), request.workspace_budget_bytes)?;
            Some(
                crate::ao::AoSpace::new(
                    &self.basis,
                    if self.options.cartesian {
                        BasisKind::Cartesian
                    } else {
                        BasisKind::Spherical
                    },
                )?
                .cart_to_ao,
            )
        } else {
            None
        };
        let estimated_dimensions = if representation == "custom" {
            (0..arity)
                .map(|i| {
                    request.coefficients[if request.coefficients.len() == 1 {
                        0
                    } else {
                        i
                    }]
                    .cols
                })
                .collect::<Vec<_>>()
        } else {
            vec![physical.as_ref().map_or(n, |c| c.cols); arity]
        };
        tensor_indices(
            &request.indices,
            &estimated_dimensions,
            request.max_elements,
        )?;
        let np = (n as u128) * (n as u128 + 1) / 2;
        let packed_count = np * (np + 1) / 2;
        if representation.starts_with("mo_") || fock {
            workspace_limit(
                packed_count + 16 * (n as u128).pow(2),
                request.workspace_budget_bytes,
            )?;
        }
        let reference = if representation.starts_with("mo_") || fock {
            Some(self.scf(settings)?)
        } else {
            None
        };
        let borrowed = reference.as_ref().map(|r| r.reference());
        let frames: Vec<&Matrix> = (0..arity)
            .filter_map(|axis| match representation {
                "cartesian_ao" => None,
                "physical_ao" => physical.as_ref(),
                "custom" => Some(
                    &request.coefficients[if request.coefficients.len() == 1 {
                        0
                    } else {
                        axis
                    }],
                ),
                _ => {
                    let spin = match representation {
                        "mo_beta" => 1,
                        "mo_alpha_beta" => usize::from(axis >= 2),
                        "mo_beta_alpha" => usize::from(axis < 2),
                        _ => 0,
                    };
                    borrowed.map(|r| r.coefficients(spin))
                }
            })
            .collect();
        let dimensions = if representation == "cartesian_ao" {
            vec![n; arity]
        } else {
            frames.iter().map(|c| c.cols).collect()
        };
        let indices = tensor_indices(&request.indices, &dimensions, request.max_elements)?;
        let shape: Vec<_> = indices.iter().map(Vec::len).collect();
        let coefficient_elements = if frames.is_empty() {
            0
        } else {
            n as u128 * shape.iter().map(|&d| d as u128).sum::<u128>()
        };
        workspace_limit(
            shape.iter().fold(1u128, |a, &b| a * b as u128) + coefficient_elements,
            request.workspace_budget_bytes,
        )?;
        let directions: Vec<_> = request
            .derivative_coordinates
            .iter()
            .map(|&i| {
                let mut v = vec![[0.; 3]; self.atoms.len()];
                v[i / 3][i % 3] = 1.;
                v
            })
            .collect();
        let coefficients: Vec<_> = frames
            .iter()
            .zip(&indices)
            .map(|(c, i)| selected_columns(c, i))
            .collect();
        let values = if arity == 4 {
            if representation == "cartesian_ao" {
                let mut out = Vec::with_capacity(shape.iter().product());
                for &i in &indices[0] {
                    for &j in &indices[1] {
                        for &k in &indices[2] {
                            for &l in &indices[3] {
                                out.push(if directions.is_empty() {
                                    crate::integrals::contracted_eri(
                                        &self.basis[i],
                                        &self.basis[j],
                                        &self.basis[k],
                                        &self.basis[l],
                                    )
                                } else {
                                    crate::hf_derivatives::eri_directional(
                                        &self.basis,
                                        [i, j, k, l],
                                        &directions,
                                    )
                                });
                            }
                        }
                    }
                }
                out
            } else {
                let dims: Vec<_> = shape.iter().map(|&d| d as u128).collect();
                let stages = [
                    dims[0] * (n as u128).pow(3),
                    dims[0] * dims[1] * (n as u128).pow(2),
                    dims[0] * dims[1] * dims[2] * n as u128,
                    dims.iter().product(),
                ];
                let live = stages.windows(2).map(|x| x[0] + x[1]).max().unwrap_or(0);
                workspace_limit(
                    packed_count + live + coefficient_elements + 16 * (n as u128).pow(2),
                    request.workspace_budget_bytes,
                )?;
                let owned = if directions.is_empty() {
                    if borrowed.is_none() {
                        Some(crate::scf::compute_eri_tensor(&self.basis))
                    } else {
                        None
                    }
                } else {
                    use rayon::prelude::*;
                    let pairs: Vec<_> = (0..n).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
                    let mut data = vec![0.; packed_count as usize];
                    data.par_iter_mut().enumerate().for_each(|(index, value)| {
                        let mut p = (((8.0 * index as f64 + 1.).sqrt() - 1.) / 2.) as usize;
                        while (p + 1) * (p + 2) / 2 <= index {
                            p += 1;
                        }
                        while p * (p + 1) / 2 > index {
                            p -= 1;
                        }
                        let q = index - p * (p + 1) / 2;
                        let (i, j) = pairs[p];
                        let (k, l) = pairs[q];
                        *value = crate::hf_derivatives::eri_directional(
                            &self.basis,
                            [i, j, k, l],
                            &directions,
                        );
                    });
                    Some(data)
                };
                let packed = owned
                    .as_deref()
                    .unwrap_or_else(|| borrowed.expect("SCF ERIs").eri());
                transform_eri_block(n, packed, &coefficients)
            }
        } else {
            // A selected AO block needs only the union of its two AO lists for
            // ECP quadrature. Keep a tiny diagnostic request independent of the
            // full AO matrix size.
            let ecp_indices: Vec<usize> = if representation == "cartesian_ao" {
                indices
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect()
            } else {
                (0..n).collect()
            };
            let matrix_elements = if representation == "cartesian_ao" {
                if matches!(operator, "ecp" | "core_hamiltonian") {
                    4 * (ecp_indices.len() as u128).pow(2)
                } else {
                    0
                }
            } else {
                16 * (n as u128).pow(2) + (shape[0] as u128) * (n as u128 + shape[1] as u128)
            };
            workspace_limit(
                matrix_elements
                    + coefficient_elements
                    + shape.iter().map(|&d| d as u128).product::<u128>(),
                request.workspace_budget_bytes,
            )?;
            let mut ecp_lookup = vec![0usize; n];
            for (local, &global) in ecp_indices.iter().enumerate() {
                ecp_lookup[global] = local;
            }
            let ecp = if matches!(operator, "ecp" | "core_hamiltonian") {
                let selected_basis: Vec<_> =
                    ecp_indices.iter().map(|&i| self.basis[i].clone()).collect();
                Some(
                    crate::ecp::ecp_nuclear_derivative_matrix(
                        &self.atoms,
                        &selected_basis,
                        &directions,
                    )
                    .map_err(HfError::Integral)?,
                )
            } else {
                None
            };
            let fock_matrix = if fock {
                Some(match &**reference.as_ref().expect("Fock reference") {
                    ScfReference::Rhf(hf) => &hf.fock_matrix,
                    ScfReference::Uhf(hf) => {
                        if operator == "fock_beta"
                            || (operator == "fock" && representation == "mo_beta")
                        {
                            &hf.fock_matrix_beta
                        } else {
                            &hf.fock_matrix_alpha
                        }
                    }
                })
            } else {
                None
            };
            let entry = |i: usize, j: usize| {
                if let Some(f) = fock_matrix {
                    return f[(i, j)];
                }
                let a = &self.basis[i];
                let b = &self.basis[j];
                match operator {
                    "ecp" => ecp.as_ref().expect("ECP matrix")[(ecp_lookup[i], ecp_lookup[j])],
                    "core_hamiltonian" => {
                        bare_pair_derivative(&self.atoms, a, b, &directions, "kinetic")
                            + bare_pair_derivative(
                                &self.atoms,
                                a,
                                b,
                                &directions,
                                "nuclear_attraction",
                            )
                            + ecp.as_ref().expect("ECP matrix")[(ecp_lookup[i], ecp_lookup[j])]
                    }
                    _ => bare_pair_derivative(&self.atoms, a, b, &directions, operator),
                }
            };
            let mut out = Vec::with_capacity(shape.iter().product());
            if representation == "cartesian_ao" {
                for &i in &indices[0] {
                    for &j in &indices[1] {
                        out.push(entry(i, j));
                    }
                }
            } else {
                let a = Matrix {
                    rows: n,
                    cols: n,
                    data: (0..n)
                        .flat_map(|i| (0..n).map(move |j| entry(i, j)))
                        .collect(),
                };
                out = coefficients[0]
                    .transpose()
                    .dot(&a)
                    .dot(&coefficients[1])
                    .data;
            }
            out
        };
        let order = directions.len();
        let energy_power = if operator == "overlap" { 0. } else { 1. };
        let unit = match (operator == "overlap", order) {
            (true, 0) => "1".into(),
            (true, _) => format!("bohr^-{order}"),
            (false, 0) => "hartree".into(),
            (false, _) => format!("hartree/bohr^{order}"),
        };
        let axes = if arity == 4 {
            vec!["p", "q", "r", "s"]
        } else {
            vec!["p", "q"]
        }
        .into_iter()
        .map(str::to_string)
        .collect();
        Ok(IntegralBlock {
            operator: request.operator.clone(),
            representation: request.representation.clone(),
            indices,
            derivative_coordinates: request.derivative_coordinates.clone(),
            orbital_response_included: false,
            tensor: TensorData::from_data(
                values,
                shape,
                axes,
                (&unit, energy_power, -(order as f64)),
                request.max_elements,
            )?,
        })
    }

    fn ri_integrals(&self, request: &IntegralRequest) -> Result<IntegralBlock, HfError> {
        use crate::output::TensorData;
        if request.representation != "cartesian_ao" || !request.coefficients.is_empty() {
            return Err(HfError::Unsupported {method:request.operator.clone(),detail:"raw auxiliary and whitened-factor inspection currently uses Cartesian AO/auxiliary rows; typed DfTensor::transform supplies other orbital frames".into()});
        }
        let factorized = request.operator == "df_factors";
        if !matches!(
            request.operator.as_str(),
            "df_metric" | "df_three_center" | "df_factors"
        ) {
            return Err(HfError::InvalidInput(
                "unknown df_* integral operator".into(),
            ));
        }
        if factorized && !request.derivative_coordinates.is_empty() {
            return Err(HfError::Unsupported {method:request.operator.clone(),detail:"whitened-factor derivatives include metric response; inspect raw metric/three-center derivatives instead".into()});
        }
        let name = request.aux_basis.as_deref().ok_or_else(|| {
            HfError::InvalidInput(
                "df_* inspection requires an explicit built-in aux_basis name".into(),
            )
        })?;
        let auxiliary = AuxBasis::from_builtin(&self.atoms, name)?;
        let na = auxiliary.functions.len();
        let n = self.basis.len();
        let dimensions = if request.operator == "df_metric" {
            vec![na, na]
        } else {
            vec![na, n, n]
        };
        let estimated_indices =
            tensor_indices(&request.indices, &dimensions, request.max_elements)?;
        let estimated_output: u128 = estimated_indices
            .iter()
            .map(|axis| axis.len() as u128)
            .product();
        let factors = if factorized {
            let np = (n as u128) * (n as u128 + 1) / 2;
            workspace_limit(
                2 * na as u128 * np
                    + 12 * (na as u128).pow(2)
                    + 4 * (n as u128).pow(2)
                    + estimated_output,
                request.workspace_budget_bytes,
            )?;
            Some(crate::df::DfTensor::build(
                &self.basis,
                &auxiliary.functions,
                request.metric_cutoff,
            )?)
        } else {
            None
        };
        let actual_dimensions = if let Some(b) = &factors {
            vec![b.b.rows, n, n]
        } else {
            dimensions
        };
        let indices = tensor_indices(&request.indices, &actual_dimensions, request.max_elements)?;
        let shape = indices.iter().map(Vec::len).collect::<Vec<_>>();
        workspace_limit(
            shape.iter().fold(1u128, |a, &b| a * b as u128)
                + factors.as_ref().map_or(0, |b| b.b.data.len() as u128),
            request.workspace_budget_bytes,
        )?;
        let directions: Vec<_> = request
            .derivative_coordinates
            .iter()
            .map(|&i| {
                let mut v = vec![[0.; 3]; self.atoms.len()];
                v[i / 3][i % 3] = 1.;
                v
            })
            .collect();
        let constant = |a: &ContractedGaussian| {
            let mut c = crate::df::constant(a.center);
            c.atom_index = a.atom_index;
            c
        };
        let integral = |functions: [ContractedGaussian; 4]| {
            if directions.is_empty() {
                crate::integrals::contracted_eri(
                    &functions[0],
                    &functions[1],
                    &functions[2],
                    &functions[3],
                )
            } else {
                crate::hf_derivatives::eri_directional(&functions, [0, 1, 2, 3], &directions)
            }
        };
        let mut values = Vec::with_capacity(shape.iter().product());
        for &p in &indices[0] {
            for &q in &indices[1] {
                if request.operator == "df_metric" {
                    let a = &auxiliary.functions[p];
                    let b = &auxiliary.functions[q];
                    values.push(integral([a.clone(), constant(a), b.clone(), constant(b)]));
                } else {
                    for &r in &indices[2] {
                        values.push(if let Some(b) = &factors {
                            b.b[(p, crate::scf::compound_index(q, r))]
                        } else {
                            let a = &auxiliary.functions[p];
                            integral([
                                a.clone(),
                                constant(a),
                                self.basis[q].clone(),
                                self.basis[r].clone(),
                            ])
                        });
                    }
                }
            }
        }
        let (energy_power, length_power) = if factorized {
            (0.5, 0.)
        } else if request.operator == "df_metric" {
            (1., 3. - directions.len() as f64)
        } else {
            (1., 1.5 - directions.len() as f64)
        };
        let unit = if factorized {
            "hartree^0.5".into()
        } else if length_power == 0. {
            "hartree".into()
        } else {
            format!("hartree*bohr^{length_power}")
        };
        let axes = if request.operator == "df_metric" {
            vec!["P", "Q"]
        } else if factorized {
            vec!["retained_auxiliary_mode", "p", "q"]
        } else {
            vec!["P", "p", "q"]
        }
        .into_iter()
        .map(str::to_string)
        .collect();
        Ok(IntegralBlock {
            operator: request.operator.clone(),
            representation: request.representation.clone(),
            indices,
            derivative_coordinates: request.derivative_coordinates.clone(),
            orbital_response_included: false,
            tensor: TensorData::from_data(
                values,
                shape,
                axes,
                (&unit, energy_power, length_power),
                request.max_elements,
            )?,
        })
    }
}
