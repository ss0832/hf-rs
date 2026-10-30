// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! PyO3 Python bindings for hf-rs.
//! Operates in atomic units (Bohr for coordinates, Hartree for energy, Hartree/Bohr for gradients).

use crate::basis::{Atom, atomic_number_to_symbol, build_basis, load_basis_from_file};
use crate::gradient::{compute_rhf_gradient, compute_uhf_gradient};
use crate::hessian::compute_vibrational_frequencies;
use crate::molden::{write_molden_file_rhf, write_molden_file_uhf};
use crate::scf::{
    analyze_rhf_stability, run_rhf_with_stability_rotation, run_scf_general, run_scf_with_charge,
    run_uhf,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn get_atoms_and_basis(
    atomic_numbers: &[usize],
    positions_bohr: &[[f64; 3]],
    basis: &str,
) -> PyResult<(Vec<Atom>, crate::basis::BasisSetMap)> {
    if atomic_numbers.len() != positions_bohr.len() {
        return Err(PyRuntimeError::new_err(format!(
            "Mismatch between atomic_numbers count ({}) and positions count ({})",
            atomic_numbers.len(),
            positions_bohr.len()
        )));
    }

    let basis_map = if let Some(map) = crate::basis::get_builtin_basis(basis) {
        map
    } else {
        load_basis_from_file(basis).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to load basis set file '{}': {}", basis, e))
        })?
    };

    let mut atoms = Vec::with_capacity(atomic_numbers.len());
    for (&z, &pos) in atomic_numbers.iter().zip(positions_bohr.iter()) {
        let sym = atomic_number_to_symbol(z);
        atoms.push(Atom::new(sym, pos)); // Atomic units (Bohr)
    }
    crate::ecp::assign_builtin_ecps(&mut atoms, basis);

    Ok((atoms, basis_map))
}

fn parse_conv_algo(algo: &str) -> PyResult<crate::diis::ConvAlgo> {
    match algo.to_lowercase().replace('-', "_").as_str() {
        "adiis_soscf" | "adiissoscf" | "default" => Ok(crate::diis::ConvAlgo::Adiissoscf),
        "cdiis" => Ok(crate::diis::ConvAlgo::Cdiis),
        "adiis" => Ok(crate::diis::ConvAlgo::Adiis),
        "adiis_cdiis" | "adiiscdiis" => Ok(crate::diis::ConvAlgo::AdiisCdiis),
        "soscf" => Ok(crate::diis::ConvAlgo::Soscf),
        "gdm" => Ok(crate::diis::ConvAlgo::Gdm),
        "trah" | "trustregion" | "trust_region" => Ok(crate::diis::ConvAlgo::Trah),
        other => Err(PyRuntimeError::new_err(format!(
            "Unknown convergence algorithm '{}'. Supported: 'adiis_soscf', 'cdiis', 'adiis', 'adiis_cdiis', 'soscf', 'gdm', 'trah'",
            other
        ))),
    }
}

/// Computes Restricted Hartree-Fock (RHF) total energy and analytical nuclear gradients.
/// All inputs/outputs are in atomic units (Bohr, Hartree, Hartree/Bohr).
#[pyfunction]
#[pyo3(signature = (atomic_numbers, positions, charge=0, basis="sto-3g", max_iter=60, tol=1e-11, conv_algo="adiis_soscf"))]
pub fn calc_energy_and_gradient(
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    charge: i32,
    basis: &str,
    max_iter: usize,
    tol: f64,
    conv_algo: &str,
) -> PyResult<(f64, Vec<[f64; 3]>)> {
    let (atoms, basis_map) = get_atoms_and_basis(&atomic_numbers, &positions, basis)?;
    let basis_objs = build_basis(&atoms, &basis_map);
    let algo = parse_conv_algo(conv_algo)?;
    let hf_res = run_scf_general(&atoms, &basis_objs, charge, max_iter, tol, algo, None)
        .map_err(|e| PyRuntimeError::new_err(format!("RHF SCF calculation failed: {}", e)))?;

    let grad = compute_rhf_gradient(&atoms, &basis_objs, &hf_res);

    Ok((hf_res.total_energy, grad))
}

/// Computes Unrestricted Hartree-Fock (UHF) total energy, <S^2>, and analytical nuclear gradients.
/// All inputs/outputs are in atomic units (Bohr, Hartree, Hartree/Bohr).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (atomic_numbers, positions, charge=0, multiplicity=1, is_broken_symmetry=false, basis="sto-3g", max_iter=60, tol=1e-11))]
pub fn calc_uhf_energy_and_gradient(
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    basis: &str,
    max_iter: usize,
    tol: f64,
) -> PyResult<(f64, f64, Vec<[f64; 3]>)> {
    let (atoms, basis_map) = get_atoms_and_basis(&atomic_numbers, &positions, basis)?;
    let basis_objs = build_basis(&atoms, &basis_map);
    let uhf_res = run_uhf(
        &atoms,
        &basis_objs,
        charge,
        multiplicity,
        is_broken_symmetry,
        max_iter,
        tol,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("UHF calculation failed: {}", e)))?;

    let grad = compute_uhf_gradient(&atoms, &basis_objs, &uhf_res);

    Ok((uhf_res.total_energy, uhf_res.s_squared, grad))
}

/// Analyzes RHF wavefunction stability towards UHF / broken-symmetry solutions.
#[pyfunction]
#[pyo3(signature = (atomic_numbers, positions, charge=0, basis="sto-3g", max_iter=60, tol=1e-11))]
pub fn analyze_stability(
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    charge: i32,
    basis: &str,
    max_iter: usize,
    tol: f64,
) -> PyResult<(bool, f64, String)> {
    let (atoms, basis_map) = get_atoms_and_basis(&atomic_numbers, &positions, basis)?;
    let basis_objs = build_basis(&atoms, &basis_map);
    let hf_res = run_scf_with_charge(&atoms, &basis_objs, charge, max_iter, tol)
        .map_err(|e| PyRuntimeError::new_err(format!("RHF SCF calculation failed: {}", e)))?;

    let stab = analyze_rhf_stability(&hf_res, hf_res.n_occ);

    Ok((stab.is_stable, stab.lowest_eigenvalue, stab.description))
}

/// Rotates spatial orbitals in the unstable direction and reconverges RHF SCF.
/// Strictly remains within RHF without changing to open shell.
#[pyfunction]
#[allow(clippy::type_complexity)]
#[pyo3(signature = (atomic_numbers, positions, charge=0, max_rotations=3, rotation_angle=0.2, basis="sto-3g", max_iter=60, tol=1e-11))]
pub fn calc_with_stability_rotation(
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    charge: i32,
    max_rotations: usize,
    rotation_angle: f64,
    basis: &str,
    max_iter: usize,
    tol: f64,
) -> PyResult<(f64, Vec<[f64; 3]>, bool, f64, usize)> {
    let (atoms, basis_map) = get_atoms_and_basis(&atomic_numbers, &positions, basis)?;
    let basis_objs = build_basis(&atoms, &basis_map);

    let (hf_res, stab, rotations) = run_rhf_with_stability_rotation(
        &atoms,
        &basis_objs,
        charge,
        max_iter,
        tol,
        max_rotations,
        rotation_angle,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("RHF stability rotation failed: {}", e)))?;

    let grad = compute_rhf_gradient(&atoms, &basis_objs, &hf_res);

    Ok((
        hf_res.total_energy,
        grad,
        stab.is_stable,
        stab.lowest_eigenvalue,
        rotations,
    ))
}

/// Returns orbital energies (Hartree), HOMO index, LUMO index, HOMO energy, LUMO energy, and HOMO-LUMO gap.
#[pyfunction]
#[allow(clippy::type_complexity)]
#[pyo3(signature = (atomic_numbers, positions, charge=0, basis="sto-3g", max_iter=60, tol=1e-11, conv_algo="adiis_soscf"))]
pub fn get_orbital_info(
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    charge: i32,
    basis: &str,
    max_iter: usize,
    tol: f64,
    conv_algo: &str,
) -> PyResult<(
    Vec<f64>,
    Option<usize>,
    Option<usize>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
)> {
    let (atoms, basis_map) = get_atoms_and_basis(&atomic_numbers, &positions, basis)?;
    let basis_objs = build_basis(&atoms, &basis_map);
    let algo = parse_conv_algo(conv_algo)?;
    let hf_res = run_scf_general(&atoms, &basis_objs, charge, max_iter, tol, algo, None)
        .map_err(|e| PyRuntimeError::new_err(format!("RHF SCF calculation failed: {}", e)))?;

    Ok((
        hf_res.orbital_energies.clone(),
        hf_res.homo_index(),
        hf_res.lumo_index(),
        hf_res.homo_energy(),
        hf_res.lumo_energy(),
        hf_res.homo_lumo_gap(),
    ))
}

/// Computes Mulliken and Löwdin charges, electric dipole moment (Debye), and quadrupole moment tensor (Buckingham).
#[pyfunction]
#[allow(clippy::type_complexity)]
#[pyo3(signature = (atomic_numbers, positions, charge=0, basis="sto-3g", max_iter=60, tol=1e-11, conv_algo="adiis_soscf"))]
pub fn calc_molecular_properties(
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    charge: i32,
    basis: &str,
    max_iter: usize,
    tol: f64,
    conv_algo: &str,
) -> PyResult<(Vec<f64>, Vec<f64>, [f64; 3], f64, Vec<Vec<f64>>)> {
    let (atoms, basis_map) = get_atoms_and_basis(&atomic_numbers, &positions, basis)?;
    let basis_objs = build_basis(&atoms, &basis_map);
    let algo = parse_conv_algo(conv_algo)?;
    let hf_res = run_scf_general(&atoms, &basis_objs, charge, max_iter, tol, algo, None)
        .map_err(|e| PyRuntimeError::new_err(format!("RHF SCF calculation failed: {}", e)))?;

    let quad_2d = vec![
        vec![
            hf_res.properties.quadrupole_moment_buckingham[0][0],
            hf_res.properties.quadrupole_moment_buckingham[0][1],
            hf_res.properties.quadrupole_moment_buckingham[0][2],
        ],
        vec![
            hf_res.properties.quadrupole_moment_buckingham[1][0],
            hf_res.properties.quadrupole_moment_buckingham[1][1],
            hf_res.properties.quadrupole_moment_buckingham[1][2],
        ],
        vec![
            hf_res.properties.quadrupole_moment_buckingham[2][0],
            hf_res.properties.quadrupole_moment_buckingham[2][1],
            hf_res.properties.quadrupole_moment_buckingham[2][2],
        ],
    ];

    Ok((
        hf_res.properties.mulliken_charges,
        hf_res.properties.lowdin_charges,
        hf_res.properties.dipole_moment_debye,
        hf_res.properties.dipole_magnitude_debye,
        quad_2d,
    ))
}

/// Computes Cartesian Nuclear Hessian (3N x 3N) in Hartree/Bohr^2 and Harmonic Vibrational Frequencies (cm^-1).
#[pyfunction]
#[pyo3(signature = (atomic_numbers, positions, charge=0, basis="sto-3g", max_iter=60, tol=1e-11, *, multiplicity=1, is_broken_symmetry=false))]
pub fn calc_hessian(
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    charge: i32,
    basis: &str,
    max_iter: usize,
    tol: f64,
    multiplicity: usize,
    is_broken_symmetry: bool,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let (atoms, basis_map) = get_atoms_and_basis(&atomic_numbers, &positions, basis)?;

    let hessian = crate::hf_derivatives::with_reference(
        &atoms,
        &basis_map,
        charge,
        multiplicity,
        is_broken_symmetry,
        max_iter,
        tol,
        |ctx| ctx.hessian(),
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let freqs = compute_vibrational_frequencies(&atoms, &hessian);

    let dim = 3 * atoms.len();
    let mut hess_2d = vec![vec![0.0; dim]; dim];
    for r in 0..dim {
        for c in 0..dim {
            hess_2d[r][c] = hessian[(r, c)];
        }
    }

    Ok((hess_2d, freqs))
}

/// Computes 3rd nuclear derivatives d^3 E / dR_x dR_y dR_z in Hartree/Bohr^3 as a 3D tensor.
#[pyfunction]
#[pyo3(signature = (atomic_numbers, positions, charge=0, basis="sto-3g", max_iter=60, tol=1e-11, *, multiplicity=1, is_broken_symmetry=false))]
pub fn calc_third_derivatives(
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    charge: i32,
    basis: &str,
    max_iter: usize,
    tol: f64,
    multiplicity: usize,
    is_broken_symmetry: bool,
) -> PyResult<Vec<Vec<Vec<f64>>>> {
    let (atoms, basis_map) = get_atoms_and_basis(&atomic_numbers, &positions, basis)?;

    let tensor = crate::third_deriv::try_compute_third_derivatives_general(
        &atoms,
        &basis_map,
        charge,
        multiplicity,
        is_broken_symmetry,
        max_iter,
        tol,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let dim = tensor.dim;

    let out = (0..dim)
        .map(|x| {
            (0..dim)
                .map(|y| (0..dim).map(|z| tensor.get(x, y, z)).collect())
                .collect()
        })
        .collect();

    Ok(out)
}

/// Computes 4th nuclear derivatives d^4 E / dR_w dR_x dR_y dR_z in Hartree/Bohr^4 as a 4D tensor.
/// Supports both RHF and UHF with charge, multiplicity, and broken symmetry.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (atomic_numbers, positions, charge=0, multiplicity=1, is_broken_symmetry=false, basis="sto-3g", max_iter=60, tol=1e-11))]
pub fn calc_fourth_derivatives(
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    basis: &str,
    max_iter: usize,
    tol: f64,
) -> PyResult<Vec<Vec<Vec<Vec<f64>>>>> {
    let (atoms, basis_map) = get_atoms_and_basis(&atomic_numbers, &positions, basis)?;

    let tensor = crate::fourth_deriv::try_compute_fourth_derivatives_general(
        &atoms,
        &basis_map,
        charge,
        multiplicity,
        is_broken_symmetry,
        max_iter,
        tol,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let dim = tensor.dim;

    let out = (0..dim)
        .map(|w| {
            (0..dim)
                .map(|x| {
                    (0..dim)
                        .map(|y| (0..dim).map(|z| tensor.get(w, x, y, z)).collect())
                        .collect()
                })
                .collect()
        })
        .collect();

    Ok(out)
}

/// Mixed directional HF nuclear derivative of order 1..4, including scalar ECPs.
/// Directions contain 3N Cartesian displacement components in atom-major order.
#[pyfunction]
#[pyo3(signature = (atomic_numbers, positions, directions, charge=0, multiplicity=1, is_broken_symmetry=false, basis="sto-3g", max_iter=60, tol=1e-11))]
pub fn calc_hf_nuclear_derivative(
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    directions: Vec<Vec<f64>>,
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    basis: &str,
    max_iter: usize,
    tol: f64,
) -> PyResult<f64> {
    let (atoms, basis_map) = get_atoms_and_basis(&atomic_numbers, &positions, basis)?;
    if directions.is_empty() || directions.len() > 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected 1..4 nuclear directions",
        ));
    }
    let directions = directions
        .iter()
        .map(|v| crate::hf_derivatives::direction_from_flat(v, atoms.len()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    crate::hf_derivatives::with_reference(
        &atoms,
        &basis_map,
        charge,
        multiplicity,
        is_broken_symmetry,
        max_iter,
        tol,
        |ctx| ctx.derivative(&directions),
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Exports molecular orbitals and basis set to a Molden file on disk.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (atomic_numbers, positions, file_path, charge=0, multiplicity=1, is_broken_symmetry=false, basis="sto-3g", max_iter=60, tol=1e-11, conv_algo="adiis_soscf"))]
pub fn export_molden_file(
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    file_path: &str,
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    basis: &str,
    max_iter: usize,
    tol: f64,
    conv_algo: &str,
) -> PyResult<()> {
    let (atoms, basis_map) = get_atoms_and_basis(&atomic_numbers, &positions, basis)?;
    let basis_objs = build_basis(&atoms, &basis_map);

    if multiplicity > 1 || is_broken_symmetry {
        let uhf_res = run_uhf(
            &atoms,
            &basis_objs,
            charge,
            multiplicity,
            is_broken_symmetry,
            max_iter,
            tol,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("UHF calculation failed: {}", e)))?;
        write_molden_file_uhf(&atoms, &basis_map, &uhf_res, file_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write Molden file: {}", e)))?;
    } else {
        let algo = parse_conv_algo(conv_algo)?;
        let hf_res = run_scf_general(&atoms, &basis_objs, charge, max_iter, tol, algo, None)
            .map_err(|e| PyRuntimeError::new_err(format!("RHF calculation failed: {}", e)))?;
        write_molden_file_rhf(&atoms, &basis_map, &hf_res, file_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to write Molden file: {}", e)))?;
    }

    Ok(())
}

/// Returns relative atomic mass in amu (CODATA) for atomic number Z (1 to 118).
#[pyfunction]
pub fn get_atomic_mass(z: usize) -> f64 {
    crate::masses::atomic_mass(z)
}

/// Returns atomic mass in atomic units (electron masses, CODATA) for atomic number Z.
#[pyfunction]
pub fn get_atomic_mass_au(z: usize) -> f64 {
    crate::masses::atomic_mass_au(z)
}

/// Returns standard atomic weight (IUPAC CIAAW) for atomic number Z.
#[pyfunction]
pub fn get_standard_atomic_weight(z: usize) -> f64 {
    crate::masses::standard_atomic_weight(z)
}

/// Returns chemical symbol for atomic number Z.
#[pyfunction]
pub fn get_element_symbol(z: usize) -> &'static str {
    crate::masses::element_symbol(z)
}

/// Returns element name for atomic number Z.
#[pyfunction]
pub fn get_element_name(z: usize) -> &'static str {
    crate::masses::element_name(z)
}

/// Looks up atomic number Z from element symbol (e.g. "H" -> 1, "Fe" -> 26).
#[pyfunction]
pub fn find_atomic_number(symbol: &str) -> Option<usize> {
    crate::masses::z_from_symbol(symbol)
}

#[pymodule]
pub fn _hf_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::py_api::register(m)?;
    m.add_function(wrap_pyfunction!(calc_energy_and_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(calc_uhf_energy_and_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_stability, m)?)?;
    m.add_function(wrap_pyfunction!(calc_with_stability_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(get_orbital_info, m)?)?;
    m.add_function(wrap_pyfunction!(calc_molecular_properties, m)?)?;
    m.add_function(wrap_pyfunction!(calc_hessian, m)?)?;
    m.add_function(wrap_pyfunction!(calc_third_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(calc_fourth_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(calc_hf_nuclear_derivative, m)?)?;
    m.add_function(wrap_pyfunction!(export_molden_file, m)?)?;
    m.add_function(wrap_pyfunction!(get_atomic_mass, m)?)?;
    m.add_function(wrap_pyfunction!(get_atomic_mass_au, m)?)?;
    m.add_function(wrap_pyfunction!(get_standard_atomic_weight, m)?)?;
    m.add_function(wrap_pyfunction!(get_element_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(get_element_name, m)?)?;
    m.add_function(wrap_pyfunction!(find_atomic_number, m)?)?;
    m.add("AMU_TO_AU", crate::masses::AMU_TO_AU)?;
    m.add("BOHR_TO_ANGSTROM", crate::masses::BOHR_TO_ANGSTROM)?;
    m.add("ANGSTROM_TO_BOHR", crate::masses::ANGSTROM_TO_BOHR)?;
    m.add("HARTREE_TO_EV", crate::masses::HARTREE_TO_EV)?;
    m.add("EV_TO_HARTREE", crate::masses::EV_TO_HARTREE)?;
    m.add("HARTREE_TO_J", crate::masses::HARTREE_TO_J)?;
    m.add("SPEED_OF_LIGHT_AU", crate::masses::SPEED_OF_LIGHT_AU)?;
    m.add("__author__", "ss0832")?;
    m.add("__license__", "Apache-2.0")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
