// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Central energy differences with independently reconverged RHF references.
//! Geometry points run sequentially; integral/SCF kernels retain their parallelism.

use crate::{
    ao::BasisKind,
    basis::{Atom, ContractedGaussian},
    error::HfError,
    registry::GradientKind,
    scf::{HFResult, RhfConvergence, run_scf_in_space_with_convergence},
};

#[derive(Clone, Copy, Debug)]
pub struct NumericalGradientOptions {
    /// Symmetric coordinate displacement in bohr.
    pub step_bohr: f64,
    pub scf_max_iter: usize,
    pub scf_tolerance: f64,
    pub scf_residual_tolerance: f64,
    pub scf_density_tolerance: f64,
}

impl Default for NumericalGradientOptions {
    fn default() -> Self {
        Self {
            step_bohr: 1e-3,
            scf_max_iter: 200,
            scf_tolerance: 1e-13,
            scf_residual_tolerance: 1e-12,
            scf_density_tolerance: 1e-12,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NumericalGradientResult {
    /// dE/dR in Hartree/bohr, including the nuclear repulsion.
    pub gradient: Vec<[f64; 3]>,
    pub kind: GradientKind,
    pub step_bohr: f64,
    pub reference_energy: f64,
    pub energy_evaluations: usize,
    pub scf_tolerance: f64,
    pub scf_residual_tolerance: f64,
    pub scf_density_tolerance: f64,
}

/// Translate each Gaussian with its owning atom, preserving any fixed offset.
pub fn displaced_basis(
    basis: &[ContractedGaussian],
    original: &[Atom],
    displaced: &[Atom],
) -> Result<Vec<ContractedGaussian>, HfError> {
    if original.len() != displaced.len()
        || basis
            .iter()
            .any(|g| g.atom_index >= original.len() || g.center.iter().any(|v| !v.is_finite()))
    {
        return Err(HfError::InvalidInput(
            "invalid atom ownership for a moving basis".into(),
        ));
    }
    let mut result = basis.to_vec();
    for g in &mut result {
        for axis in 0..3 {
            g.center[axis] +=
                displaced[g.atom_index].position[axis] - original[g.atom_index].position[axis];
        }
    }
    Ok(result)
}

/// Evaluate total energies, never a CPHF-perturbed surrogate or a frozen CABS.
/// The callback must converge its own post-HF equations and propagate failures.
pub fn compute_rhf_numerical_gradient<F>(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    basis_kind: BasisKind,
    options: &NumericalGradientOptions,
    mut total_energy: F,
) -> Result<NumericalGradientResult, HfError>
where
    F: FnMut(&HFResult, &[Atom], &[ContractedGaussian]) -> Result<f64, HfError>,
{
    let step = options.step_bohr;
    if atoms.is_empty()
        || basis.is_empty()
        || !step.is_finite()
        || step <= 0.0
        || options.scf_max_iter == 0
        || !options.scf_tolerance.is_finite()
        || options.scf_tolerance <= 0.0
        || options.scf_tolerance > 1e-10
        || [
            options.scf_residual_tolerance,
            options.scf_density_tolerance,
        ]
        .iter()
        .any(|&value| !value.is_finite() || value <= 0.0 || value > 1e-10)
        || atoms.iter().any(|a| {
            !a.charge.is_finite()
                || a.position.iter().any(|&v| {
                    !v.is_finite()
                        || !((v + step).is_finite() && (v - step).is_finite())
                        || v + step == v
                        || v - step == v
                })
        })
        || basis.iter().any(|g| {
            g.primitives.is_empty()
                || g.l.iter().any(|&l| l > 7)
                || g.l.iter().sum::<usize>() > 7
                || g.primitives.iter().any(|p| {
                    !p.exponent.is_finite() || p.exponent <= 0.0 || !p.coefficient.is_finite()
                })
        })
    {
        return Err(HfError::InvalidInput(
            "numerical RHF gradient needs a finite positive step, valid geometry/basis and SCF tolerance <=1e-10".into(),
        ));
    }
    let mut shape = None;
    let mut evaluations = 0;
    let mut evaluate = |point: &[Atom]| -> Result<f64, HfError> {
        let moved = displaced_basis(basis, atoms, point)?;
        let hf = run_scf_in_space_with_convergence(
            point,
            &moved,
            charge,
            options.scf_max_iter,
            RhfConvergence {
                energy: options.scf_tolerance,
                residual: options.scf_residual_tolerance,
                density: options.scf_density_tolerance,
            },
            crate::diis::ConvAlgo::Adiissoscf,
            None,
            basis_kind,
        )
        .map_err(HfError::Scf)?;
        if !hf.converged || !hf.total_energy.is_finite() {
            return Err(HfError::Scf(
                "numerical-gradient geometry did not converge".into(),
            ));
        }
        let current = (hf.n_occ, hf.mo_coefficients.cols);
        if shape.is_some_and(|expected| expected != current) {
            return Err(HfError::Response(
                "occupied count or retained OBS rank changes across numerical-gradient geometries"
                    .into(),
            ));
        }
        shape = Some(current);
        let energy = total_energy(&hf, point, &moved)?;
        if !energy.is_finite() {
            return Err(HfError::Response(
                "nonfinite numerical-gradient total energy".into(),
            ));
        }
        evaluations += 1;
        Ok(energy)
    };
    let reference_energy = evaluate(atoms)?;
    let mut gradient = vec![[0.0; 3]; atoms.len()];
    for atom in 0..atoms.len() {
        for axis in 0..3 {
            let mut point = atoms.to_vec();
            point[atom].position[axis] += step;
            let plus = evaluate(&point)?;
            point[atom].position[axis] = atoms[atom].position[axis] - step;
            let minus = evaluate(&point)?;
            let value = (plus - minus) / (2.0 * step);
            if !value.is_finite() {
                return Err(HfError::Response(
                    "nonfinite numerical-gradient component".into(),
                ));
            }
            gradient[atom][axis] = value;
        }
    }
    Ok(NumericalGradientResult {
        gradient,
        kind: GradientKind::Numerical,
        step_bohr: step,
        reference_energy,
        energy_evaluations: evaluations,
        scf_tolerance: options.scf_tolerance,
        scf_residual_tolerance: options.scf_residual_tolerance,
        scf_density_tolerance: options.scf_density_tolerance,
    })
}
