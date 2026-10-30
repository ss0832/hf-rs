// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//! Energies and selected-state semantics shared by all post-HF methods.
use crate::error::HfError;
use crate::matrix::Matrix;
use crate::{
    api::ScfReference,
    basis::{Atom, ContractedGaussian},
};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// A bounded, row-major numerical tensor for integral and intermediate inspection.
#[derive(Clone, Debug, Serialize)]
pub struct TensorData {
    pub shape: Vec<usize>,
    pub axes: Vec<String>,
    pub values: Vec<f64>,
    pub unit: String,
    pub energy_power: f64,
    pub length_power: f64,
}
impl TensorData {
    pub fn from_data(
        values: Vec<f64>,
        shape: Vec<usize>,
        axes: Vec<String>,
        unit: (&str, f64, f64),
        limit: usize,
    ) -> Result<Self, HfError> {
        if Self::checked_len(&shape, limit)? != values.len()
            || axes.len() != shape.len()
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(HfError::InvalidInput(
                "invalid tensor dimensions, axes or nonfinite values".into(),
            ));
        }
        Ok(Self {
            shape,
            axes,
            values,
            unit: unit.0.into(),
            energy_power: unit.1,
            length_power: unit.2,
        })
    }
    pub fn checked_len(shape: &[usize], limit: usize) -> Result<usize, HfError> {
        let size = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
        match size {
            Some(size) if limit > 0 && size <= limit => Ok(size),
            _ => Err(HfError::InvalidInput(
                "tensor element count exceeds max_elements or overflows; request a smaller block"
                    .into(),
            )),
        }
    }
    pub fn from_slice(
        values: &[f64],
        shape: &[usize],
        axes: &[&str],
        unit: ObservableUnit,
        limit: usize,
    ) -> Result<Self, HfError> {
        if Self::checked_len(shape, limit)? != values.len()
            || axes.len() != shape.len()
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(HfError::InvalidInput(
                "invalid tensor dimensions, axes or nonfinite values".into(),
            ));
        }
        let (label, energy_power, length_power) = unit.descriptor();
        Ok(Self {
            shape: shape.to_vec(),
            axes: axes.iter().map(|x| x.to_string()).collect(),
            values: values.to_vec(),
            unit: label.into(),
            energy_power,
            length_power,
        })
    }
    pub fn from_matrix(
        matrix: &Matrix,
        unit: ObservableUnit,
        limit: usize,
    ) -> Result<Self, HfError> {
        Self::from_slice(
            &matrix.data,
            &[matrix.rows, matrix.cols],
            &["row", "column"],
            unit,
            limit,
        )
    }
}
pub type TensorMap = BTreeMap<String, TensorData>;

/// An aggregate element budget for opt-in copies of retained solver tensors.
pub struct TensorExport {
    values: TensorMap,
    remaining: usize,
}
impl TensorExport {
    pub fn new(max_elements: usize) -> Result<Self, HfError> {
        if max_elements == 0 {
            return Err(HfError::InvalidInput(
                "max_elements must be positive".into(),
            ));
        }
        Ok(Self {
            values: TensorMap::new(),
            remaining: max_elements,
        })
    }
    pub fn tensor(
        &mut self,
        name: &str,
        data: &[f64],
        shape: &[usize],
        axes: &[&str],
        unit: ObservableUnit,
    ) -> Result<(), HfError> {
        if self.values.contains_key(name) {
            return Err(HfError::InvalidInput(
                "duplicate intermediate tensor name".into(),
            ));
        }
        let tensor = TensorData::from_slice(data, shape, axes, unit, self.remaining)?;
        self.remaining -= tensor.values.len();
        self.values.insert(name.into(), tensor);
        Ok(())
    }
    pub fn matrix(
        &mut self,
        name: &str,
        matrix: &Matrix,
        unit: ObservableUnit,
    ) -> Result<(), HfError> {
        self.tensor(
            name,
            &matrix.data,
            &[matrix.rows, matrix.cols],
            &["row", "column"],
            unit,
        )
    }
    pub fn finish(self) -> TensorMap {
        self.values
    }
}

#[derive(Clone, Debug)]
pub struct State {
    pub energy: f64,
    pub s_squared: Option<f64>,
    pub weight: f64,
    pub converged: bool,
}

#[derive(Clone, Debug)]
pub struct PostHFOutput {
    pub method: String,
    pub e_scf: f64,
    /// Correlation/relaxation energy relative to SCF, including for MR methods.
    pub e_corr: f64,
    pub e_total: f64,
    pub components: BTreeMap<String, f64>,
    pub states: Vec<State>,
    pub root: usize,
    pub one_rdm: Option<Matrix>,
}
impl PostHFOutput {
    pub fn new(method: impl Into<String>, e_scf: f64, e_total: f64) -> Result<Self, HfError> {
        if !e_scf.is_finite() || !e_total.is_finite() {
            return Err(HfError::InvalidInput("non-finite post-HF energy".into()));
        }
        Ok(Self {
            method: method.into(),
            e_scf,
            e_corr: e_total - e_scf,
            e_total,
            components: BTreeMap::new(),
            states: vec![State {
                energy: e_total,
                s_squared: None,
                weight: 1.0,
                converged: true,
            }],
            root: 0,
            one_rdm: None,
        })
    }
    pub fn select_root(&mut self, root: usize) -> Result<(), HfError> {
        let state = self.states.get(root).ok_or_else(|| {
            HfError::InvalidInput(format!(
                "root {root} exceeds {} computed states",
                self.states.len()
            ))
        })?;
        if !state.converged || !state.energy.is_finite() {
            return Err(HfError::Response(format!(
                "selected root {root} did not converge"
            )));
        }
        self.root = root;
        self.e_total = state.energy;
        self.e_corr = self.e_total - self.e_scf;
        Ok(())
    }
    pub fn validate(&self) -> Result<(), HfError> {
        let state = self
            .states
            .get(self.root)
            .ok_or_else(|| HfError::InvalidInput("selected state is missing".into()))?;
        if !self.e_scf.is_finite()
            || !self.e_total.is_finite()
            || !self.e_corr.is_finite()
            || (self.e_scf + self.e_corr - self.e_total).abs() > 1e-11
            || (state.energy - self.e_total).abs() > 1e-11
            || self.components.values().any(|v| !v.is_finite())
            || !state.converged
        {
            return Err(HfError::InvalidInput(
                "inconsistent post-HF energies or states".into(),
            ));
        }
        Ok(())
    }
}

fn rows(matrix: &Matrix) -> Vec<&[f64]> {
    matrix.data.chunks(matrix.cols.max(1)).collect()
}

fn channel(
    spin: &str,
    coefficients: &Matrix,
    energies: &[f64],
    density: &Matrix,
    occupied: usize,
    occupation: f64,
    n_ao: usize,
) -> Result<Value, HfError> {
    if coefficients.rows != n_ao
        || coefficients.cols != energies.len()
        || density.rows != n_ao
        || density.cols != n_ao
        || occupied > energies.len()
        || coefficients
            .data
            .iter()
            .chain(&density.data)
            .chain(energies)
            .any(|x| !x.is_finite())
    {
        return Err(HfError::InvalidInput(
            "invalid SCF wavefunction dimensions or values".into(),
        ));
    }
    Ok(
        json!({"spin":spin, "coefficients":rows(coefficients), "orbital_energies":energies,
        "occupations":(0..energies.len()).map(|i| if i < occupied {occupation} else {0.}).collect::<Vec<_>>(),
        "density_matrix":rows(density)}),
    )
}

pub(crate) fn scf_snapshot(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    reference: &ScfReference,
) -> Result<Value, HfError> {
    let radial = |terms: &[crate::ecp::EcpTerm]| {
        terms
            .iter()
            .map(|t| json!({"power":t.power,"exponent":t.exponent,"coefficient":t.coefficient}))
            .collect::<Vec<_>>()
    };
    let geometry: Vec<_> = atoms.iter().map(|a| json!({"symbol":a.symbol,"atomic_number":a.atomic_number,
        "effective_nuclear_charge":a.charge,"position":a.position,
        "ecp":a.ecp.as_ref().map(|e| json!({"n_core":e.n_core,"l_max":e.l_max,
            "local":radial(&e.local),"semilocal":e.semilocal.iter().map(|c| radial(c)).collect::<Vec<_>>()}))})).collect();
    let functions: Vec<_> = basis.iter().map(|b| json!({"atom_index":b.atom_index,"center":b.center,"powers":b.l,
        "primitives":b.primitives.iter().map(|p| json!({"exponent":p.exponent,"coefficient":p.coefficient})).collect::<Vec<_>>()})).collect();
    let n = basis.len();
    let (name, energy, charge, multiplicity, alpha, beta, overlap, space, channels) =
        match reference {
            ScfReference::Rhf(hf) => (
                "rhf",
                hf.total_energy,
                hf.charge,
                hf.multiplicity,
                hf.n_occ,
                hf.n_occ,
                &hf.overlap_matrix,
                &hf.ao_space,
                vec![channel(
                    "restricted",
                    &hf.mo_coefficients,
                    &hf.orbital_energies,
                    &hf.density_matrix,
                    hf.n_occ,
                    2.,
                    n,
                )?],
            ),
            ScfReference::Uhf(hf) => (
                "uhf",
                hf.total_energy,
                hf.charge,
                hf.multiplicity,
                hf.n_alpha,
                hf.n_beta,
                &hf.overlap_matrix,
                &hf.ao_space,
                vec![
                    channel(
                        "alpha",
                        &hf.mo_coefficients_alpha,
                        &hf.orbital_energies_alpha,
                        &hf.density_matrix_alpha,
                        hf.n_alpha,
                        1.,
                        n,
                    )?,
                    channel(
                        "beta",
                        &hf.mo_coefficients_beta,
                        &hf.orbital_energies_beta,
                        &hf.density_matrix_beta,
                        hf.n_beta,
                        1.,
                        n,
                    )?,
                ],
            ),
        };
    if overlap.data.iter().any(|x| !x.is_finite()) {
        return Err(HfError::InvalidInput("nonfinite SCF overlap matrix".into()));
    }
    Ok(
        json!({"schema":"hf-rs-scf-wavefunction-1","version":env!("CARGO_PKG_VERSION"),
        "scope":"configured RHF/UHF SCF reference", "reference":name,"e_scf":energy,
        "charge":charge,"multiplicity":multiplicity,"n_alpha":alpha,"n_beta":beta,
        "ao_basis_kind":if space.as_ref().is_some_and(|s| s.kind == crate::ao::BasisKind::Spherical) {"spherical"} else {"cartesian"},
        "coefficient_basis":"normalized_cartesian_gaussians", "atoms":geometry,"basis_functions":functions,
        "overlap_matrix":rows(overlap),"channels":channels,
        "units":{"positions":"bohr","orbital_energies":"hartree","energy":"hartree"}}),
    )
}

/// Atomic units of a reported observable; exponents support ASE conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservableUnit {
    One,
    Hartree,
    HartreeSquared,
    InverseHartreeSquared,
    Bohr,
    InverseBohr,
    Dipole,
    Quadrupole,
    Charge,
    Coupling,
    RadialAmplitude,
}
impl ObservableUnit {
    pub fn descriptor(self) -> (&'static str, f64, f64) {
        match self {
            Self::One => ("1", 0., 0.),
            Self::Hartree => ("hartree", 1., 0.),
            Self::HartreeSquared => ("hartree^2", 2., 0.),
            Self::InverseHartreeSquared => ("hartree^-2", -2., 0.),
            Self::Bohr => ("bohr", 0., 1.),
            Self::InverseBohr => ("bohr^-1", 0., -1.),
            Self::Dipole => ("e*bohr", 0., 1.),
            Self::Quadrupole => ("e*bohr^2", 0., 2.),
            Self::Charge => ("e", 0., 0.),
            Self::Coupling => ("hartree^0.5/bohr", 0.5, -1.),
            Self::RadialAmplitude => ("bohr^-0.5", 0., -0.5),
        }
    }
    pub fn to_json(self) -> Value {
        let (unit, energy_power, length_power) = self.descriptor();
        json!({"unit":unit,"energy_power":energy_power,"length_power":length_power})
    }
}

/// A numerical observable in atomic units, or a dimensionless classification.
#[derive(Clone, Debug, Serialize)]
pub struct Observable {
    pub value: Value,
    pub unit: &'static str,
    pub energy_power: f64,
    pub length_power: f64,
}
pub type ObservableMap = BTreeMap<String, Observable>;

/// Method-specific, explicitly requested physical observables from a solved state.
pub trait HfObservables {
    fn observables(&self) -> Result<ObservableMap, HfError>;
    /// Stored model tensors only; requesting these never creates an ERI copy.
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        Ok(TensorExport::new(max_elements)?.finish())
    }
}
trait FiniteObservation {
    fn finite(&self) -> bool;
}
impl FiniteObservation for f64 {
    fn finite(&self) -> bool {
        self.is_finite()
    }
}
macro_rules! finite_scalar {
    ($($t:ty),*) => {$(impl FiniteObservation for $t { fn finite(&self) -> bool {true} })*};
}
finite_scalar!(usize, u64, u32, i32, i64, bool, String, str);
impl<T: FiniteObservation + ?Sized> FiniteObservation for &T {
    fn finite(&self) -> bool {
        (**self).finite()
    }
}
impl<T: FiniteObservation> FiniteObservation for Option<T> {
    fn finite(&self) -> bool {
        self.as_ref().is_none_or(FiniteObservation::finite)
    }
}
impl<T: FiniteObservation> FiniteObservation for [T] {
    fn finite(&self) -> bool {
        self.iter().all(FiniteObservation::finite)
    }
}
impl<T: FiniteObservation> FiniteObservation for Vec<T> {
    fn finite(&self) -> bool {
        self.as_slice().finite()
    }
}
impl<T: FiniteObservation, const N: usize> FiniteObservation for [T; N] {
    fn finite(&self) -> bool {
        self.as_slice().finite()
    }
}
fn observe<T: Serialize + FiniteObservation + ?Sized>(
    value: &T,
    kind: ObservableUnit,
) -> Result<Observable, HfError> {
    if !value.finite() {
        return Err(HfError::Response("nonfinite HF observable".into()));
    }
    let (unit, energy_power, length_power) = kind.descriptor();
    Ok(Observable {
        value: serde_json::to_value(value).map_err(|e| HfError::InvalidInput(e.to_string()))?,
        unit,
        energy_power,
        length_power,
    })
}
macro_rules! observations {
    ($($key:literal => $value:expr, $unit:ident);* $(;)?) => {{
        let mut values=ObservableMap::new();
        $(values.insert($key.into(),observe(&$value,ObservableUnit::$unit)?);)*
        Ok(values)
    }};
}
/// Unit metadata for the common post-HF component namespace. Unknown keys stay explicit.
pub fn component_unit(key: &str) -> Option<ObservableUnit> {
    if key.starts_with("E_")
        || key.starts_with("Delta_E_")
        || matches!(
            key,
            "excitation_energy_hartree"
                | "reference_energy_change"
                | "reference_ci_residual"
                | "source_residual_max"
                | "cholesky_max_residual"
                | "caspt2_residual"
                | "orbital_gradient_max"
                | "residual_max"
                | "triples_residual_max"
        )
    {
        Some(ObservableUnit::Hartree)
    } else if key == "dynamic_weight_parameter" {
        Some(ObservableUnit::InverseHartreeSquared)
    } else if key.starts_with("F12_")
        || key.starts_with("dmrg_")
        || matches!(
            key,
            "S_squared"
                | "oscillator_strength"
                | "contracted_dimension"
                | "cholesky_rank"
                | "factorized_sectors"
                | "peak_sector_terms"
                | "source_transformations"
                | "contracted_model_space"
                | "reference_iterations"
                | "reference_ci_change_max"
                | "lambda2_norm"
                | "lambda3_norm"
                | "amplitude_iterations"
                | "amplitude_residual_max"
                | "bch_order"
                | "factor_rank"
                | "model_dimension"
                | "active_ci_dimension"
                | "reference_weight"
                | "caspt2_iterations"
                | "orbital_iterations"
        )
    {
        Some(ObservableUnit::One)
    } else {
        None
    }
}

impl HfObservables for crate::properties::MolecularProperties {
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "dipole" => self.dipole_moment_au, Dipole;
            "dipole_magnitude" => self.dipole_magnitude_au, Dipole;
            "quadrupole" => self.quadrupole_moment_au, Quadrupole;
            "mulliken_charges" => self.mulliken_charges, Charge;
            "lowdin_charges" => self.lowdin_charges, Charge;
        }
    }
}

impl HfObservables for crate::scf::HFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.coefficients_cartesian_ao",
            &self.mo_coefficients,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_cartesian_ao",
            &self.density_matrix,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.overlap_cartesian_ao",
            &self.overlap_matrix,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.core_cartesian_ao",
            &self.core_hamiltonian,
            ObservableUnit::Hartree,
        )?;
        tensors.matrix(
            "model.fock_cartesian_ao",
            &self.fock_matrix,
            ObservableUnit::Hartree,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        let mut result: ObservableMap = observations! {
            "electronic_energy" => self.electronic_energy, Hartree;
            "nuclear_repulsion_energy" => self.nuclear_repulsion_energy, Hartree;
            "orbital_energies" => self.orbital_energies, Hartree;
            "homo_energy" => self.homo_energy(), Hartree;
            "lumo_energy" => self.lumo_energy(), Hartree;
            "homo_lumo_gap" => self.homo_lumo_gap(), Hartree;
            "n_electrons" => 2*self.n_occ, One;
        }?;
        result.extend(self.properties.observables()?);
        Ok(result)
    }
}

impl HfObservables for crate::rohf::FermiSmearingResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        self.hf.intermediates(max_elements)
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        let mut values = self.hf.observables()?;
        for key in ["homo_energy", "lumo_energy", "homo_lumo_gap", "n_electrons"] {
            values.remove(key);
        }
        let thermal_energy = self.temperature_kelvin / 315775.02480407;
        let entropy_over_kb = self.electronic_entropy * 315775.02480407;
        let extra: ObservableMap = observations! {
            "chemical_potential" => self.chemical_potential, Hartree;
            "free_energy" => self.free_energy, Hartree;
            "thermal_energy" => thermal_energy, Hartree;
            "entropy_over_kb" => entropy_over_kb, One;
            "occupations" => self.occupations, One;
            "n_electrons" => self.occupations.iter().sum::<f64>(), One;
        }?;
        values.extend(extra);
        Ok(values)
    }
}

impl HfObservables for crate::scf::UHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.coefficients_alpha_cartesian_ao",
            &self.mo_coefficients_alpha,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.coefficients_beta_cartesian_ao",
            &self.mo_coefficients_beta,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_alpha_cartesian_ao",
            &self.density_matrix_alpha,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_beta_cartesian_ao",
            &self.density_matrix_beta,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.overlap_cartesian_ao",
            &self.overlap_matrix,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.core_cartesian_ao",
            &self.core_hamiltonian,
            ObservableUnit::Hartree,
        )?;
        tensors.matrix(
            "model.fock_alpha_cartesian_ao",
            &self.fock_matrix_alpha,
            ObservableUnit::Hartree,
        )?;
        tensors.matrix(
            "model.fock_beta_cartesian_ao",
            &self.fock_matrix_beta,
            ObservableUnit::Hartree,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        let mut result: ObservableMap = observations! {
            "electronic_energy" => self.electronic_energy, Hartree;
            "nuclear_repulsion_energy" => self.nuclear_repulsion_energy, Hartree;
            "orbital_energies_alpha" => self.orbital_energies_alpha, Hartree;
            "orbital_energies_beta" => self.orbital_energies_beta, Hartree;
            "homo_lumo_gap_alpha" => self.gap_alpha(), Hartree;
            "homo_lumo_gap_beta" => self.gap_beta(), Hartree;
            "s_squared" => self.s_squared, One;
            "spin_contamination" => self.s_squared - ((self.multiplicity-1) as f64/2.) * ((self.multiplicity-1) as f64/2.+1.), One;
            "n_alpha" => self.n_alpha, One;
            "n_beta" => self.n_beta, One;
        }?;
        result.extend(self.properties.observables()?);
        Ok(result)
    }
}

impl HfObservables for crate::rohf::ROHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.coefficients_cartesian_ao",
            &self.mo_coefficients,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_alpha_cartesian_ao",
            &self.density_matrix_alpha,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_beta_cartesian_ao",
            &self.density_matrix_beta,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.overlap_cartesian_ao",
            &self.overlap_matrix,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.core_cartesian_ao",
            &self.core_hamiltonian,
            ObservableUnit::Hartree,
        )?;
        tensors.matrix(
            "model.fock_alpha_cartesian_ao",
            &self.fock_matrix_alpha,
            ObservableUnit::Hartree,
        )?;
        tensors.matrix(
            "model.fock_beta_cartesian_ao",
            &self.fock_matrix_beta,
            ObservableUnit::Hartree,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        let mut result: ObservableMap = observations! {
            "orbital_energies" => self.orbital_energies, Hartree;
            "s_squared" => self.s_squared, One;
            "n_closed" => self.n_closed, One;
            "n_open" => self.n_open, One;
        }?;
        result.extend(self.properties.observables()?);
        Ok(result)
    }
}

impl HfObservables for crate::hf_variants::GHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.coefficients_spinor",
            &self.spinor_coefficients,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_spinor",
            &self.density_matrix_2n,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.fock_spinor",
            &self.fock_matrix_2n,
            ObservableUnit::Hartree,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        let mut result: ObservableMap = observations! {
            "spin_expectations" => [self.s_x,self.s_y,self.s_z], One;
            "s_squared" => self.s_squared, One;
            "fukutome_class" => format!("{:?}",self.fukutome_class), One;
            "spinor_energies" => self.spinor_energies, Hartree;
            "spinor_density" => rows(&self.density_matrix_2n), One;
            "residual_norm" => self.residual_norm, Hartree;
        }?;
        result.extend(self.properties.observables()?);
        Ok(result)
    }
}

impl HfObservables for crate::hf_variants::PUHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        Ok(self
            .raw_uhf
            .intermediates(max_elements)?
            .into_iter()
            .map(|(name, tensor)| (name.replacen("model.", "model.raw_uhf.", 1), tensor))
            .collect())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "projected_energy" => self.projected_energy, Hartree;
            "unprojected_energy" => self.raw_uhf.total_energy, Hartree;
            "unprojected_s_squared" => self.raw_uhf.s_squared, One;
            "target_spin" => self.target_spin, One;
            "target_spin_weight" => self.target_weight, One;
            "nontarget_spin_weight" => self.spin_contamination_weight, One;
            "high_spin_reference_energy" => self.high_spin_reference_energy, Hartree;
            "yamaguchi_energy_estimate" => self.yamaguchi_energy, Hartree;
        }
    }
}

impl HfObservables for crate::hf_variants::HFBResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.normal_density",
            &self.normal_density,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.pairing_tensor",
            &self.pairing_tensor,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.coefficients_cartesian_ao",
            &self.orbital_coefficients,
            ObservableUnit::One,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "normal_reference_energy" => self.normal_energy, Hartree;
            "pairing_energy" => self.pairing_energy, Hartree;
            "normal_component_energy" => self.normal_component_energy, Hartree;
            "coulomb_pairing_energy" => self.coulomb_pairing_energy, Hartree;
            "pairing_gap" => self.pairing_gap, Hartree;
            "chemical_potential" => self.chemical_potential, Hartree;
            "quasiparticle_energies" => self.quasiparticle_energies, Hartree;
            "normal_density" => rows(&self.normal_density), One;
            "pairing_tensor" => rows(&self.pairing_tensor), One;
            "particle_number_variance" => self.particle_number_variance, One;
            "number_error" => self.number_error, One;
        }
    }
}

impl HfObservables for crate::hf_variants::TCSCFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.coefficients_cartesian_ao",
            &self.mo_coefficients,
            ObservableUnit::One,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "reference_energy" => self.hf_reference_energy, Hartree;
            "correlation_energy" => self.correlation_energy, Hartree;
            "configuration_coefficients" => [self.c1,self.c2], One;
            "natural_occupations" => self.natural_occupations, One;
            "active_orbitals" => [self.active_bonding_orbital,self.active_antibonding_orbital], One;
            "s_squared" => self.s_squared, One;
            "orbital_gradient_max" => self.orbital_gradient_max, Hartree;
        }
    }
}

impl HfObservables for crate::hf_variants::QEDHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.coefficients_cartesian_ao",
            &self.mo_coefficients,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_cartesian_ao",
            &self.density_matrix,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.overlap_cartesian_ao",
            &self.overlap_matrix,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.fock_cartesian_ao",
            &self.fock_matrix,
            ObservableUnit::Hartree,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "dipole_self_energy" => self.dipole_self_energy, Hartree;
            "cavity_energy" => self.cavity_energy, Hartree;
            "coherent_displacement" => self.coherent_displacement, One;
            "mean_photon_number" => self.mean_photon_number, One;
            "cavity_frequency" => self.cavity_frequency_hartree, Hartree;
            "coupling" => self.coupling_vector, Coupling;
            "dipole" => self.expected_dipole_bohr, Dipole;
            "orbital_energies" => self.orbital_energies, Hartree;
            "ao_density" => rows(&self.density_matrix), One;
        }
    }
}

impl HfObservables for crate::hf_variants::DFHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.coefficients_cartesian_ao",
            &self.mo_coefficients,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_cartesian_ao",
            &self.density_matrix,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.fock_cartesian_ao",
            &self.fock_matrix,
            ObservableUnit::Hartree,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "n_auxiliary" => self.n_aux, One;
            "fitting_error" => self.fitting_error, Hartree;
            "residual_norm" => self.residual_norm, Hartree;
            "orbital_energies" => self.orbital_energies, Hartree;
        }
    }
}

impl HfObservables for crate::hf_variants::CDHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.coefficients_cartesian_ao",
            &self.mo_coefficients,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_cartesian_ao",
            &self.density_matrix,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.fock_cartesian_ao",
            &self.fock_matrix,
            ObservableUnit::Hartree,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "cholesky_rank" => self.n_cholesky_vectors, One;
            "cholesky_max_error" => self.cholesky_max_error, Hartree;
            "orbital_energies" => self.orbital_energies, Hartree;
            "residual_norm" => self.residual_norm, Hartree;
        }
    }
}

impl HfObservables for crate::hf_variants::RelativisticHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.coefficients_cartesian_ao",
            &self.mo_coefficients,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_cartesian_ao",
            &self.density_matrix,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.overlap_cartesian_ao",
            &self.overlap_matrix,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.core_cartesian_ao",
            &self.core_hamiltonian,
            ObservableUnit::Hartree,
        )?;
        tensors.matrix(
            "model.fock_cartesian_ao",
            &self.fock_matrix,
            ObservableUnit::Hartree,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "nonrelativistic_reference_energy" => self.non_relativistic_energy, Hartree;
            "relativistic_shift" => self.relativistic_shift_hartree, Hartree;
            "model" => format!("{:?}",self.model), One;
            "orbital_energies" => self.orbital_energies, Hartree;
            "residual_norm" => self.residual_norm, Hartree;
        }
    }
}

impl HfObservables for crate::hf_variants::ComplexHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.density_real",
            &self.density_real,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_imaginary",
            &self.density_imag,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.spinor_density_real",
            &self.spinor_density_real,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.spinor_density_imaginary",
            &self.spinor_density_imag,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.coefficients_real",
            &self.coefficients_real,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.coefficients_imaginary",
            &self.coefficients_imag,
            ObservableUnit::One,
        )?;
        tensors.matrix("model.fock_real", &self.fock_real, ObservableUnit::Hartree)?;
        tensors.matrix(
            "model.fock_imaginary",
            &self.fock_imag,
            ObservableUnit::Hartree,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "s_squared" => self.s_squared, One;
            "spin_expectations" => self.spin_expectations, One;
            "fukutome_class" => format!("{:?}",self.fukutome_class), One;
            "imaginary_density_norm" => self.current_magnitude, One;
            "gauge_origin" => self.gauge_origin, Bohr;
            "orbital_energies" => self.orbital_energies, Hartree;
            "density_real" => rows(&self.density_real), One;
            "density_imag" => rows(&self.density_imag), One;
            "residual_norm" => self.residual_norm, Hartree;
        }
    }
}

impl HfObservables for crate::hf_variants::LHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.coefficients_cartesian_ao",
            &self.mo_coefficients,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_cartesian_ao",
            &self.density_matrix,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.ks_fock_cartesian_ao",
            &self.ks_fock_matrix,
            ObservableUnit::Hartree,
        )?;
        tensors.matrix(
            "model.local_exchange_cartesian_ao",
            &self.local_exchange.potential,
            ObservableUnit::Hartree,
        )?;
        tensors.matrix(
            "model.slater_cartesian_ao",
            &self.local_exchange.slater,
            ObservableUnit::Hartree,
        )?;
        tensors.matrix(
            "model.correction_constants_occupied",
            &self.local_exchange.correction_constants,
            ObservableUnit::Hartree,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "exchange_energy" => self.exchange_energy, Hartree;
            "reference_hf_energy" => self.reference_hf_energy, Hartree;
            "homo_lumo_gap" => self.homo_lumo_gap, Hartree;
            "orbital_energies" => self.orbital_energies, Hartree;
            "local_exchange_potential" => rows(&self.local_exchange.potential), Hartree;
            "slater_potential" => rows(&self.local_exchange.slater), Hartree;
            "grid_points" => self.local_exchange.grid_points, One;
            "grid_overlap_error" => self.local_exchange.grid_overlap_error, One;
            "residual_norm" => self.residual_norm, Hartree;
        }
    }
}

impl HfObservables for crate::hf_variants::OepResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.coefficients_cartesian_ao",
            &self.mo_coefficients,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_cartesian_ao",
            &self.density_matrix,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.reference_density_cartesian_ao",
            &self.reference_density_matrix,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.ks_hamiltonian_cartesian_ao",
            &self.ks_hamiltonian,
            ObservableUnit::Hartree,
        )?;
        tensors.matrix(
            "model.hf_fock_cartesian_ao",
            &self.hf_fock_matrix,
            ObservableUnit::Hartree,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "reference_hf_energy" => self.reference_hf_energy, Hartree;
            "minimum_gap" => self.minimum_gap, Hartree;
            "orbital_energies" => self.orbital_energies, Hartree;
            "ks_hamiltonian" => rows(&self.ks_hamiltonian), Hartree;
            "ks_residual_norm" => self.ks_residual_norm, Hartree;
            "fermi_amaldi_factor" => self.fermi_amaldi_factor, One;
        }
    }
}

impl HfObservables for crate::hf_variants::NEOHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.electronic_density",
            &self.electronic_density,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.protonic_density",
            &self.protonic_density,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.electronic_coefficients",
            &self.electronic_mo_coefficients,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.protonic_coefficients",
            &self.protonic_mo_coefficients,
            ObservableUnit::One,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "electronic_energy" => self.electronic_energy, Hartree;
            "protonic_energy" => self.protonic_energy, Hartree;
            "electron_proton_coulomb" => self.electron_proton_coulomb, Hartree;
            "protonic_kinetic_energy" => self.protonic_kinetic_energy, Hartree;
            "electronic_orbital_energies" => self.electronic_orbital_energies, Hartree;
            "protonic_orbital_energies" => self.protonic_orbital_energies, Hartree;
            "electronic_density" => rows(&self.electronic_density), One;
            "protonic_density" => rows(&self.protonic_density), One;
            "n_quantum_protons" => self.n_quantum_protons, One;
            "residual_norm" => self.residual_norm, Hartree;
        }
    }
}

impl HfObservables for crate::hf_variants::PositronicHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.electron_density",
            &self.electron_density,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.positron_density",
            &self.positron_density,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.electron_coefficients",
            &self.electron_mo_coefficients,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.positron_coefficients",
            &self.positron_mo_coefficients,
            ObservableUnit::One,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "electronic_energy" => self.electronic_energy, Hartree;
            "positronic_energy" => self.positronic_energy, Hartree;
            "electron_positron_attraction" => self.ep_attraction_energy, Hartree;
            "positron_kinetic_energy" => self.positron_kinetic_energy, Hartree;
            "positron_orbital_energy" => self.positron_orbital_energy, Hartree;
            "electron_orbital_energies" => self.electron_orbital_energies, Hartree;
            "electron_density" => rows(&self.electron_density), One;
            "positron_density" => rows(&self.positron_density), One;
        }
    }
}

impl HfObservables for crate::hf_variants::DCHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.density_cartesian_ao",
            &self.density_matrix,
            ObservableUnit::One,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "chemical_potential" => self.chemical_potential, Hartree;
            "electron_count" => self.electron_count, One;
            "density_residual" => self.density_residual, One;
            "n_subsystems" => self.n_subsystems, One;
            "ao_density" => rows(&self.density_matrix), One;
        }
    }
}

impl HfObservables for crate::hf_variants::FMOResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        let mut cluster =
            |name: String, state: &crate::hf_variants::FMOClusterResult| -> Result<(), HfError> {
                for (key, matrix, unit) in [
                    ("density", &state.density_matrix, ObservableUnit::One),
                    ("coefficients", &state.mo_coefficients, ObservableUnit::One),
                    ("overlap", &state.overlap_matrix, ObservableUnit::One),
                    ("fock", &state.fock_matrix, ObservableUnit::Hartree),
                    (
                        "embedding_potential",
                        &state.embedding_potential,
                        ObservableUnit::Hartree,
                    ),
                ] {
                    tensors.matrix(&format!("model.{name}.{key}"), matrix, unit)?;
                }
                Ok(())
            };
        for (i, state) in self.monomer_results.iter().enumerate() {
            cluster(format!("monomer.{i}"), state)?;
        }
        for ((i, j), state) in &self.dimer_results {
            cluster(format!("dimer.{i}.{j}"), state)?;
        }
        if let Some(states) = &self.trimer_results {
            for ((i, j, k), state) in states {
                cluster(format!("trimer.{i}.{j}.{k}"), state)?;
            }
        }
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "order" => self.order, One;
            "monomer_energies" => self.monomer_energies, Hartree;
            "fragment_atom_indices" => self.monomer_results.iter().map(|x| x.atom_indices.clone()).collect::<Vec<_>>(), One;
            "monomer_ao_indices" => self.monomer_results.iter().map(|x| x.ao_indices.clone()).collect::<Vec<_>>(), One;
            "dimer_ao_indices" => self.dimer_results.iter().map(|x| x.1.ao_indices.clone()).collect::<Vec<_>>(), One;
            "trimer_ao_indices" => self.trimer_results.as_ref().map(|v| v.iter().map(|x| x.1.ao_indices.clone()).collect::<Vec<_>>()), One;
            "dimer_indices" => self.dimer_energies.iter().map(|x| [x.0.0,x.0.1]).collect::<Vec<_>>(), One;
            "dimer_energies" => self.dimer_energies.iter().map(|x| x.1).collect::<Vec<_>>(), Hartree;
            "pair_indices" => self.ifie.iter().map(|x| [x.frag_i,x.frag_j]).collect::<Vec<_>>(), One;
            "pair_interaction_energies" => self.ifie.iter().map(|x| x.total_interaction_energy).collect::<Vec<_>>(), Hartree;
            "pair_electrostatic" => self.ifie.iter().map(|x| x.electrostatic).collect::<Vec<_>>(), Hartree;
            "pair_exchange_repulsion" => self.ifie.iter().map(|x| x.exchange_repulsion).collect::<Vec<_>>(), Hartree;
            "pair_charge_transfer_and_mixing" => self.ifie.iter().map(|x| x.charge_transfer).collect::<Vec<_>>(), Hartree;
            "pair_embedding_correction" => self.ifie.iter().map(|x| x.embedding_correction).collect::<Vec<_>>(), Hartree;
            "trimer_indices" => self.trimer_energies.as_ref().map(|v| v.iter().map(|x|[x.0.0,x.0.1,x.0.2]).collect::<Vec<_>>()), One;
            "trimer_corrections" => self.trimer_energies.as_ref().map(|v| v.iter().map(|x|x.1).collect::<Vec<_>>()), Hartree;
            "trimer_internal_energies" => self.trimer_results.as_ref().map(|v| v.iter().map(|x|x.1.internal_energy).collect::<Vec<_>>()), Hartree;
            "embedding_iterations" => self.embedding_iterations, One;
            "embedding_density_change" => self.embedding_density_change, One;
        }
    }
}

impl HfObservables for crate::hf_variants::TDHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        for (i, state) in self.excited_states.iter().enumerate() {
            for (key, data) in [("x", &state.x), ("y", &state.y)] {
                tensors.tensor(
                    &format!("model.state.{i}.{key}"),
                    data,
                    &[data.len()],
                    &["occupied_virtual_pair"],
                    ObservableUnit::One,
                )?;
            }
        }
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "reference_energy" => self.reference_energy, Hartree;
            "excitation_energies" => self.excited_states.iter().map(|x|x.excitation_energy_hartree).collect::<Vec<_>>(), Hartree;
            "oscillator_strengths" => self.excited_states.iter().map(|x|x.oscillator_strength).collect::<Vec<_>>(), One;
            "transition_dipoles" => self.excited_states.iter().map(|x|x.transition_dipole).collect::<Vec<_>>(), Dipole;
            "triplet_flags" => self.excited_states.iter().map(|x|x.is_triplet).collect::<Vec<_>>(), One;
            "x_amplitudes" => self.excited_states.iter().map(|x|x.x.clone()).collect::<Vec<_>>(), One;
            "y_amplitudes" => self.excited_states.iter().map(|x|x.y.clone()).collect::<Vec<_>>(), One;
        }
    }
}

impl HfObservables for crate::hf_variants::COHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        for (i, density) in self.density_blocks.iter().enumerate() {
            tensors.matrix(
                &format!(
                    "model.real_space_density_block.{}",
                    i as isize - self.density_range as isize
                ),
                density,
                ObservableUnit::One,
            )?;
        }
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        let cbm = self
            .conduction_band_minimum
            .is_finite()
            .then_some(self.conduction_band_minimum);
        let gap = cbm.map(|c| c - self.valence_band_maximum);
        observations! {
            "energy_per_cell" => self.total_energy_per_cell, Hartree;
            "kinetic_energy_per_cell" => self.kinetic_energy_per_cell, Hartree;
            "electrostatic_energy_per_cell" => self.electrostatic_energy_per_cell, Hartree;
            "exchange_energy_per_cell" => self.exchange_energy_per_cell, Hartree;
            "band_gap" => gap, Hartree;
            "valence_band_maximum" => self.valence_band_maximum, Hartree;
            "conduction_band_minimum" => cbm, Hartree;
            "lattice_constant" => self.lattice_constant_bohr, Bohr;
            "k_fractional" => self.band_structure.iter().map(|x|x.k_fractional).collect::<Vec<_>>(), One;
            "k_cartesian" => self.band_structure.iter().map(|x|x.k_cartesian).collect::<Vec<_>>(), InverseBohr;
            "band_energies" => self.band_structure.iter().map(|x|x.band_energies.clone()).collect::<Vec<_>>(), Hartree;
            "electrons_per_cell" => self.electrons_per_cell, One;
        }
    }
}

impl HfObservables for crate::hf_variants::NumericalHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.tensor(
            "model.radial_grid",
            &self.radial_grid,
            &[self.radial_grid.len()],
            &["grid_point"],
            ObservableUnit::Bohr,
        )?;
        tensors.tensor(
            "model.radial_wavefunction",
            &self.radial_wavefunction,
            &[self.radial_wavefunction.len()],
            &["grid_point"],
            ObservableUnit::RadialAmplitude,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "orbital_energy_1s" => self.orbital_energy_1s, Hartree;
            "hfl_total_energy" => self.hfl_total_energy, Hartree;
            "kinetic_energy" => self.kinetic_energy, Hartree;
            "nuclear_attraction_energy" => self.nuclear_attraction_energy, Hartree;
            "electron_repulsion_energy" => self.electron_repulsion_energy, Hartree;
            "radial_grid" => self.radial_grid, Bohr;
            "radial_wavefunction" => self.radial_wavefunction, RadialAmplitude;
            "r_max" => self.r_max_bohr, Bohr;
            "grid_spacing" => self.grid_spacing_bohr, Bohr;
            "virial_error" => self.virial_error, Hartree;
        }
    }
}

impl HfObservables for crate::hf_variants::StateAveragedHFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.coefficients_cartesian_ao",
            &self.mo_coefficients,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_cartesian_ao",
            &self.density_matrix,
            ObservableUnit::One,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "ensemble_energy" => self.sa_energy, Hartree;
            "state_energies" => self.state_energies, Hartree;
            "weights" => self.weights, One;
            "orbital_energies" => self.orbital_energies, Hartree;
        }
    }
}

impl HfObservables for crate::hf_variants::SigmaSCFResult {
    fn intermediates(&self, max_elements: usize) -> Result<TensorMap, HfError> {
        let mut tensors = TensorExport::new(max_elements)?;
        tensors.matrix(
            "model.coefficients_alpha_cartesian_ao",
            &self.mo_coefficients_alpha,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.coefficients_beta_cartesian_ao",
            &self.mo_coefficients_beta,
            ObservableUnit::One,
        )?;
        tensors.matrix(
            "model.density_cartesian_ao",
            &self.density_matrix,
            ObservableUnit::One,
        )?;
        Ok(tensors.finish())
    }
    fn observables(&self) -> Result<ObservableMap, HfError> {
        observations! {
            "target_energy" => self.target_energy, Hartree;
            "energy_variance" => self.energy_variance, HartreeSquared;
            "intrinsic_variance" => self.intrinsic_variance, HartreeSquared;
            "n_alpha" => self.n_alpha, One;
            "n_beta" => self.n_beta, One;
        }
    }
}
