// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Thin Python boundary; calculations run without holding the Python GIL.
use crate::api::{self, MoleculeOptions, OptionMap, ScfOptions};
use crate::error::HfError;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};

fn error(value: HfError) -> PyErr {
    match value {
        HfError::InvalidInput(_) | HfError::UnknownMethod(_) => {
            PyValueError::new_err(value.to_string())
        }
        _ => PyRuntimeError::new_err(value.to_string()),
    }
}
fn options_map(py: Python<'_>, options: Option<&Bound<'_, PyDict>>) -> PyResult<OptionMap> {
    match options {
        None => Ok(OptionMap::new()),
        Some(options) => {
            let text: String = py
                .import("json")?
                .call_method1("dumps", (options,))?
                .extract()?;
            serde_json::from_str(&text).map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }
}
fn python_json(py: Python<'_>, value: serde_json::Value) -> PyResult<PyObject> {
    Ok(py
        .import("json")?
        .call_method1("loads", (value.to_string(),))?
        .unbind())
}
fn python_value(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let text: String = py
        .import("json")?
        .call_method1("dumps", (value,))?
        .extract()?;
    serde_json::from_str(&text).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn configuration(
    py: Python<'_>,
    unit: &str,
    basis: Option<&Bound<'_, PyAny>>,
    ecp: Option<&Bound<'_, PyAny>>,
    cartesian: bool,
    charge: i32,
    multiplicity: usize,
    reference: &str,
) -> PyResult<MoleculeOptions> {
    let mut config = MoleculeOptions {
        unit: unit.parse().map_err(error)?,
        cartesian,
        charge,
        multiplicity,
        reference: reference.parse().map_err(error)?,
        ..Default::default()
    };
    if let Some(value) = basis {
        config.set_basis(&python_value(py, value)?).map_err(error)?;
    }
    if let Some(value) = ecp {
        config.set_ecp(&python_value(py, value)?).map_err(error)?;
    }
    Ok(config)
}

#[pyfunction]
#[pyo3(signature=(method, atomic_numbers, positions, *, unit="bohr", basis=None, ecp=None, cartesian=true, charge=0, multiplicity=1, reference="auto", options=None, properties=None))]
pub fn run(
    py: Python<'_>,
    method: &str,
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    unit: &str,
    basis: Option<&Bound<'_, PyAny>>,
    ecp: Option<&Bound<'_, PyAny>>,
    cartesian: bool,
    charge: i32,
    multiplicity: usize,
    reference: &str,
    options: Option<&Bound<'_, PyDict>>,
    properties: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let config = configuration(
        py,
        unit,
        basis,
        ecp,
        cartesian,
        charge,
        multiplicity,
        reference,
    )?;
    let options = options_map(py, options)?;
    let properties = properties.unwrap_or_else(|| vec!["energy".into()]);
    let result = py
        .allow_threads(|| {
            api::run(
                method,
                &atomic_numbers,
                &positions,
                config,
                &options,
                &properties,
            )
        })
        .map_err(error)?;
    python_json(py, result.to_json())
}

#[pyclass(name = "Molecule", frozen)]
pub struct PyMolecule {
    inner: api::Molecule,
}
#[pymethods]
impl PyMolecule {
    #[new]
    #[pyo3(signature=(atomic_numbers, positions, *, unit="bohr", basis=None, ecp=None, cartesian=true, charge=0, multiplicity=1, reference="auto"))]
    fn new(
        py: Python<'_>,
        atomic_numbers: Vec<usize>,
        positions: Vec<[f64; 3]>,
        unit: &str,
        basis: Option<&Bound<'_, PyAny>>,
        ecp: Option<&Bound<'_, PyAny>>,
        cartesian: bool,
        charge: i32,
        multiplicity: usize,
        reference: &str,
    ) -> PyResult<Self> {
        let options = configuration(
            py,
            unit,
            basis,
            ecp,
            cartesian,
            charge,
            multiplicity,
            reference,
        )?;
        Ok(Self {
            inner: py
                .allow_threads(|| api::Molecule::new(&atomic_numbers, &positions, options))
                .map_err(error)?,
        })
    }
    #[pyo3(signature=(method="hf", *, options=None, properties=None))]
    fn run(
        &self,
        py: Python<'_>,
        method: &str,
        options: Option<&Bound<'_, PyDict>>,
        properties: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let values = options_map(py, options)?;
        let properties = properties.unwrap_or_else(|| vec!["energy".into()]);
        let result = py
            .allow_threads(|| self.inner.run(method, &values, &properties))
            .map_err(error)?;
        python_json(py, result.to_json())
    }
    #[pyo3(signature=(*, options=None))]
    fn scf(&self, py: Python<'_>, options: Option<&Bound<'_, PyDict>>) -> PyResult<PyObject> {
        self.run(py, "hf", options, None)
    }
    #[pyo3(signature=(method="hf", *, options=None))]
    fn hf_observables(
        &self,
        py: Python<'_>,
        method: &str,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let values = options_map(py, options)?;
        let result = py
            .allow_threads(|| self.inner.hf_observables(method, &values))
            .map_err(error)?;
        python_json(
            py,
            serde_json::to_value(result).map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )
    }
    /// Copy only tensors retained by the selected solver, within a total bound.
    #[pyo3(signature=(method="hf", *, options=None, max_elements=1_000_000))]
    fn intermediates(
        &self,
        py: Python<'_>,
        method: &str,
        options: Option<&Bound<'_, PyDict>>,
        max_elements: usize,
    ) -> PyResult<PyObject> {
        let values = options_map(py, options)?;
        let result = py
            .allow_threads(|| self.inner.intermediates(method, &values, max_elements))
            .map_err(error)?;
        python_json(
            py,
            serde_json::to_value(result).map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )
    }
    /// Return a bounded bare-integral block in an explicitly selected orbital frame.
    #[pyo3(signature=(operator="overlap", *, representation="cartesian_ao", indices=None, derivative_coordinates=None, coefficients=None, aux_basis=None, metric_cutoff=1e-10, max_elements=1_000_000, workspace_budget_bytes=134_217_728, options=None))]
    fn integrals(
        &self,
        py: Python<'_>,
        operator: &str,
        representation: &str,
        indices: Option<Vec<Vec<usize>>>,
        derivative_coordinates: Option<Vec<usize>>,
        coefficients: Option<Vec<Vec<Vec<f64>>>>,
        aux_basis: Option<String>,
        metric_cutoff: f64,
        max_elements: usize,
        workspace_budget_bytes: usize,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let matrices = coefficients
            .unwrap_or_default()
            .into_iter()
            .map(|rows| {
                let columns = rows.first().map_or(0, Vec::len);
                if rows.iter().any(|r| r.len() != columns) {
                    return Err(PyValueError::new_err(
                        "coefficient matrices require rectangular rows",
                    ));
                }
                Ok(crate::matrix::Matrix {
                    rows: rows.len(),
                    cols: columns,
                    data: rows.into_iter().flatten().collect(),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let request = api::IntegralRequest {
            operator: operator.into(),
            representation: representation.into(),
            indices: indices.unwrap_or_default(),
            derivative_coordinates: derivative_coordinates.unwrap_or_default(),
            coefficients: matrices,
            aux_basis,
            metric_cutoff,
            max_elements,
            workspace_budget_bytes,
        };
        let settings = only_scf(py, options)?;
        let block = py
            .allow_threads(|| self.inner.integrals(&request, settings))
            .map_err(error)?;
        python_json(
            py,
            serde_json::to_value(block).map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        )
    }
    /// Return the configured RHF/UHF SCF reference; not a correlated wavefunction.
    #[pyo3(signature=(*, options=None))]
    fn scf_wavefunction(
        &self,
        py: Python<'_>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyObject> {
        let settings = only_scf(py, options)?;
        let value = py
            .allow_threads(|| self.inner.scf_wavefunction(settings))
            .map_err(error)?;
        python_json(py, value)
    }
    /// Return Molden text for the configured RHF/UHF reference, through g shells.
    #[pyo3(signature=(*, options=None))]
    fn scf_molden(&self, py: Python<'_>, options: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        let settings = only_scf(py, options)?;
        py.allow_threads(|| self.inner.scf_molden(settings))
            .map_err(error)
    }
    #[pyo3(signature=(method="hf", *, options=None))]
    fn gradient(
        &self,
        py: Python<'_>,
        method: &str,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<[f64; 3]>> {
        let values = options_map(py, options)?;
        py.allow_threads(|| self.inner.run(method, &values, &["gradient".into()]))
            .map_err(error)?
            .gradient
            .ok_or_else(|| PyRuntimeError::new_err("missing gradient"))
    }
    #[pyo3(signature=(method="hf", *, options=None))]
    fn hessian(
        &self,
        py: Python<'_>,
        method: &str,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        let values = options_map(py, options)?;
        let matrix = py
            .allow_threads(|| self.inner.run(method, &values, &["hessian".into()]))
            .map_err(error)?
            .hessian
            .ok_or_else(|| PyRuntimeError::new_err("missing Hessian"))?;
        Ok(matrix
            .data
            .chunks(matrix.cols.max(1))
            .map(|r| r.to_vec())
            .collect())
    }
    #[pyo3(signature=(method, *, options=None, properties=None))]
    fn hf_variant(
        &self,
        py: Python<'_>,
        method: &str,
        options: Option<&Bound<'_, PyDict>>,
        properties: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let values = options_map(py, options)?;
        let properties = properties.unwrap_or_else(|| vec!["energy".into()]);
        let result = py
            .allow_threads(|| self.inner.hf_variant(method, &values, &properties))
            .map_err(error)?;
        python_json(py, result.to_json())
    }
    #[pyo3(signature=(coordinates, *, options=None))]
    fn nuclear_derivative(
        &self,
        py: Python<'_>,
        coordinates: Vec<usize>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<f64> {
        let settings = only_scf(py, options)?;
        py.allow_threads(|| self.inner.nuclear_derivative(&coordinates, settings))
            .map_err(error)
    }
    #[pyo3(signature=(directions, *, options=None))]
    fn directional_derivative(
        &self,
        py: Python<'_>,
        directions: Vec<Vec<f64>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<f64> {
        let settings = only_scf(py, options)?;
        py.allow_threads(|| self.inner.directional_derivative(&directions, settings))
            .map_err(error)
    }
}
fn only_scf(py: Python<'_>, options: Option<&Bound<'_, PyDict>>) -> PyResult<ScfOptions> {
    let mut values = options_map(py, options)?;
    let settings = ScfOptions::extract(&mut values).map_err(error)?;
    if !values.is_empty() {
        return Err(PyValueError::new_err(
            "only SCF options apply to this operation",
        ));
    }
    Ok(settings)
}

#[pyfunction]
#[pyo3(signature=(positions, unit="bohr"))]
pub fn convert_positions(positions: Vec<[f64; 3]>, unit: &str) -> PyResult<Vec<[f64; 3]>> {
    let scale = match unit.parse().map_err(error)? {
        api::LengthUnit::Bohr => 1.,
        api::LengthUnit::Angstrom => crate::masses::ANGSTROM_TO_BOHR,
    };
    let result: Vec<_> = positions.iter().map(|p| p.map(|x| x * scale)).collect();
    if result.iter().flatten().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("nonfinite coordinates"));
    }
    Ok(result)
}

#[pyfunction]
pub fn list_builtin_basis_sets() -> Vec<&'static str> {
    crate::basis::list_builtin_basis_sets()
}

#[pyfunction]
pub fn method_catalog(py: Python<'_>) -> PyResult<PyObject> {
    python_json(py, api::method_catalog())
}
#[pyfunction]
pub fn option_catalog(py: Python<'_>) -> PyResult<PyObject> {
    python_json(py, api::option_catalog())
}
#[pyfunction]
pub fn hf_variant_catalog(py: Python<'_>) -> PyResult<PyObject> {
    python_json(py, api::hf_variant_catalog())
}
#[pyfunction]
pub fn oracle_compare(py: Python<'_>) -> PyResult<PyObject> {
    let value = py
        .allow_threads(crate::validation::compare)
        .map_err(error)?;
    python_json(py, value)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run, module)?)?;
    module.add_function(wrap_pyfunction!(convert_positions, module)?)?;
    module.add_function(wrap_pyfunction!(list_builtin_basis_sets, module)?)?;
    module.add_function(wrap_pyfunction!(method_catalog, module)?)?;
    module.add_function(wrap_pyfunction!(option_catalog, module)?)?;
    module.add_function(wrap_pyfunction!(hf_variant_catalog, module)?)?;
    module.add_function(wrap_pyfunction!(oracle_compare, module)?)?;
    module.add_class::<PyMolecule>()?;
    Ok(())
}
