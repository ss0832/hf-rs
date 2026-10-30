// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Scalar one-electron relativistic HF with supplied-space decoupling and
//! untransformed Coulomb/exchange integrals. Only verified backends are enabled.

use super::{
    dkh2_core::SpinFreeDkh2Core,
    stationary_rhf,
    x2c_core::SpinFreeX2cCore,
    x2c_integrals::{spin_free_integrals, spin_free_nuclear_tangent},
};
use crate::{
    Matrix,
    basis::{Atom, ContractedGaussian},
    scf::{RhfConvergence, compute_eri_tensor, compute_nuclear_repulsion},
};

pub const SPEED_OF_LIGHT_AU: f64 = 137.035999084;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelativisticModel {
    X2C,
    DKH2,
    ZORA,
}

#[derive(Clone, Debug)]
pub struct RelativisticHfOptions {
    pub max_iterations: usize,
    pub convergence: RhfConvergence,
    pub speed_of_light: f64,
    pub max_work_bytes: usize,
}
impl Default for RelativisticHfOptions {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            convergence: RhfConvergence {
                energy: 1e-12,
                residual: 1e-10,
                density: 1e-10,
            },
            speed_of_light: SPEED_OF_LIGHT_AU,
            max_work_bytes: 256 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RelativisticHFResult {
    pub total_energy: f64,
    pub non_relativistic_energy: f64,
    pub relativistic_shift_hartree: f64,
    pub nuclear_repulsion_energy: f64,
    pub model: RelativisticModel,
    pub mo_coefficients: Matrix,
    pub orbital_energies: Vec<f64>,
    pub density_matrix: Matrix,
    pub fock_matrix: Matrix,
    pub overlap_matrix: Matrix,
    pub core_hamiltonian: Matrix,
    pub iterations: usize,
    pub converged: bool,
    pub residual_norm: f64,
    pub density_change: f64,
    pub charge: i32,
    pub n_occ: usize,
    pub options: RelativisticHfOptions,
}

fn supported(model: RelativisticModel) -> Result<(), String> {
    if model == RelativisticModel::ZORA {
        return Err(
            "ZORA is unsupported by this scalar-relativistic backend (it requires a numerical integration grid); use X2C or DKH2"
                .into(),
        );
    }
    Ok(())
}

/// One-electron decoupled core: spin-free X2C (exact in the basis) or second-order DKH.
enum Core {
    X2c(SpinFreeX2cCore),
    Dkh2(SpinFreeDkh2Core),
}

impl Core {
    fn build(model: RelativisticModel, s: &Matrix, t: &Matrix, v: &Matrix, w: &Matrix, c: f64, budget: usize) -> Result<Self, String> {
        supported(model)?;
        Ok(match model {
            RelativisticModel::X2C => Core::X2c(SpinFreeX2cCore::build(s, t, v, w, c, budget)?),
            RelativisticModel::DKH2 => Core::Dkh2(SpinFreeDkh2Core::build(s, t, v, w, c, budget)?),
            RelativisticModel::ZORA => unreachable!(),
        })
    }

    fn hamiltonian(&self) -> &Matrix {
        match self {
            Core::X2c(c) => &c.hamiltonian,
            Core::Dkh2(c) => &c.hamiltonian,
        }
    }

    fn into_hamiltonian(self) -> Matrix {
        match self {
            Core::X2c(c) => c.hamiltonian,
            Core::Dkh2(c) => c.hamiltonian,
        }
    }

    fn response(&self, ds: &Matrix, dt: &Matrix, dv: &Matrix, dw: &Matrix) -> Result<Matrix, String> {
        match self {
            Core::X2c(c) => c.response(ds, dt, dv, dw),
            Core::Dkh2(c) => c.response(ds, dt, dv, dw),
        }
    }
}

/// W=p.V.p is required. T, V and S alone cannot define a molecular X2C core.
pub fn compute_relativistic_core_hamiltonian(
    t: &Matrix,
    v: &Matrix,
    s: &Matrix,
    w: &Matrix,
    model: RelativisticModel,
) -> Result<Matrix, String> {
    Ok(Core::build(model, s, t, v, w, SPEED_OF_LIGHT_AU, 256 * 1024 * 1024)?.into_hamiltonian())
}

pub fn run_relativistic_hf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    model: RelativisticModel,
    max_iter: usize,
    tol: f64,
) -> Result<RelativisticHFResult, String> {
    let options = RelativisticHfOptions {
        max_iterations: max_iter,
        convergence: RhfConvergence {
            energy: tol,
            residual: tol.sqrt().min(1e-10),
            density: tol.sqrt().min(1e-10),
        },
        ..Default::default()
    };
    run_relativistic_hf_with_options(atoms, basis, charge, model, &options)
}

/// Closed-shell point-nucleus SFX2C-1e or DKH2 in the exact supplied AO space.
pub fn run_relativistic_hf_with_options(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    model: RelativisticModel,
    options: &RelativisticHfOptions,
) -> Result<RelativisticHFResult, String> {
    supported(model)?;
    let n = basis.len();
    let ne = atoms.iter().map(|a| a.charge).sum::<f64>().round() as i64 - i64::from(charge);
    if ne <= 0 || !(ne as usize).is_multiple_of(2) || ne > 2 * n as i64 {
        return Err(
            "scalar-relativistic RHF requires an even electron count within the AO space".into(),
        );
    }
    if options.max_iterations == 0
        || [
            options.convergence.energy,
            options.convergence.residual,
            options.convergence.density,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err("invalid scalar-relativistic convergence controls".into());
    }
    let pairs = (n as u128) * (n as u128 + 1) / 2;
    if 4 * pairs * (pairs + 1) + 4096 * (n as u128).pow(2) > options.max_work_bytes as u128 {
        return Err("relativistic HF integral/matrix workspace exceeds max_work_bytes".into());
    }
    let ints = spin_free_integrals(atoms, basis, options.max_work_bytes)?;
    let retained = ints
        .overlap
        .canonical_orthogonalizer(1e-8)
        .map_err(|e| e.to_string())?;
    if ne as usize / 2 > retained.cols {
        return Err("scalar-relativistic occupations exceed the retained overlap rank".into());
    }
    let hamiltonian = if retained.cols < n {
        let r = retained.cols;
        let t = retained.transpose().dot(&ints.kinetic).dot(&retained);
        let v = retained.transpose().dot(&ints.potential).dot(&retained);
        let w = retained.transpose().dot(&ints.pvp).dot(&retained);
        let core = Core::build(
            model,
            &Matrix::identity(r),
            &t,
            &v,
            &w,
            options.speed_of_light,
            options.max_work_bytes,
        )?;
        let dual = ints.overlap.dot(&retained);
        dual.dot(core.hamiltonian()).dot(&dual.transpose())
    } else {
        Core::build(
            model,
            &ints.overlap,
            &ints.kinetic,
            &ints.potential,
            &ints.pvp,
            options.speed_of_light,
            options.max_work_bytes,
        )?
        .into_hamiltonian()
    };
    let n_occ = ne as usize / 2;
    let nuclear = compute_nuclear_repulsion(atoms);
    let eri = compute_eri_tensor(basis);
    let reference = stationary_rhf::solve(
        &ints.kinetic.add(&ints.potential),
        &ints.overlap,
        &eri,
        nuclear,
        n_occ,
        options.max_iterations,
        options.convergence,
    )?
    .total_energy;
    let result = stationary_rhf::solve(
        &hamiltonian,
        &ints.overlap,
        &eri,
        nuclear,
        n_occ,
        options.max_iterations,
        options.convergence,
    )?;
    Ok(RelativisticHFResult {
        total_energy: result.total_energy,
        non_relativistic_energy: reference,
        relativistic_shift_hartree: result.total_energy - reference,
        nuclear_repulsion_energy: nuclear,
        model,
        mo_coefficients: result.coefficients,
        orbital_energies: result.orbital_energies,
        density_matrix: result.density,
        fock_matrix: result.fock,
        overlap_matrix: ints.overlap,
        core_hamiltonian: hamiltonian,
        iterations: result.iterations,
        converged: true,
        residual_norm: result.residual_norm,
        density_change: result.density_change,
        charge,
        n_occ,
        options: options.clone(),
    })
}

/// Complete first nuclear derivative of the one-electron X2C/DKH2 HF energy,
/// including the decoupling/normalization, integral and orbital Pulay response.
pub fn compute_relativistic_hf_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    result: &RelativisticHFResult,
) -> Result<Vec<[f64; 3]>, String> {
    supported(result.model)?;
    let n = basis.len();
    if !result.converged || result.mo_coefficients.rows != n || result.mo_coefficients.cols != n {
        return Err(
            "relativistic analytic gradient requires a converged full-rank RHF reference".into(),
        );
    }
    let ints = spin_free_integrals(atoms, basis, result.options.max_work_bytes)?;
    let core = Core::build(
        result.model,
        &ints.overlap,
        &ints.kinetic,
        &ints.potential,
        &ints.pvp,
        result.options.speed_of_light,
        result.options.max_work_bytes,
    )?;
    if core
        .hamiltonian()
        .data
        .iter()
        .zip(&result.core_hamiltonian.data)
        .any(|(a, b)| (a - b).abs() > 1e-8)
    {
        return Err("relativistic gradient geometry/basis does not match its reference".into());
    }
    let p = &result.density_matrix;
    let weighted = p.dot(&result.fock_matrix).dot(p).scale(0.5);
    let mut gradient = crate::gradient::compute_gradient_from_density(atoms, basis, p, &weighted);
    for (nucleus, row) in gradient.iter_mut().enumerate() {
        for (axis, g) in row.iter_mut().enumerate() {
            let d = spin_free_nuclear_tangent(
                atoms,
                basis,
                nucleus,
                axis,
                result.options.max_work_bytes,
            )?;
            let dh = core
                .response(&d.overlap, &d.kinetic, &d.potential, &d.pvp)?
                .sub(&d.kinetic)
                .sub(&d.potential);
            *g += p.data.iter().zip(dh.data).map(|(p, h)| p * h).sum::<f64>();
        }
    }
    Ok(gradient)
}
