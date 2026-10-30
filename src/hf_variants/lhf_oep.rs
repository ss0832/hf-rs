// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Self-consistent local multiplicative LHF/CEDA exchange, distinct from OEP.
//! The physical energy uses nonlocal exact exchange of the local-potential orbitals.
//! Quadrature and the full occupied correction equation are certified separately.
//! No analytic nuclear response is currently provided.

use super::local_exchange::{ExchangePoint, build_exchange, exchange_workspace};
use super::local_grid::LocalGrid;
pub use super::local_grid::LocalGridOptions;
use crate::scf::{contract_jk, run_scf_general};
use crate::{Atom, ContractedGaussian, ConvAlgo, HFResult, Matrix};

#[derive(Clone, Debug)]
pub struct LhfOptions {
    pub max_iterations: usize,
    pub reference_max_iterations: usize,
    pub energy_tolerance: f64,
    pub residual_tolerance: f64,
    pub density_tolerance: f64,
    pub linear_tolerance: f64,
    pub minimum_linear_eigenvalue: f64,
    pub grid: LocalGridOptions,
    pub max_grid_refinements: usize,
}
impl Default for LhfOptions {
    fn default() -> Self {
        Self {
            max_iterations: 150,
            reference_max_iterations: 300,
            energy_tolerance: 1e-11,
            residual_tolerance: 1e-8,
            density_tolerance: 1e-9,
            linear_tolerance: 1e-6,
            minimum_linear_eigenvalue: 1e-10,
            grid: LocalGridOptions::default(),
            max_grid_refinements: 2,
        }
    }
}
#[derive(Clone, Debug)]
pub struct LhfExchangeResult {
    pub potential: Matrix,
    pub slater: Matrix,
    pub correction_constants: Matrix,
    pub linear_residual: f64,
    pub minimum_linear_eigenvalue: f64,
    pub grid_overlap_error: f64,
    pub grid_points: usize,
    pub cached_kernel_points: usize,
    pub estimated_additional_workspace_bytes: usize,
    pub grid_refinements: usize,
}
#[derive(Clone, Debug)]
pub struct LHFResult {
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_energy: f64,
    /// Nonlocal HF exchange energy evaluated with the LHF orbitals.
    pub exchange_energy: f64,
    pub reference_hf_energy: f64,
    pub homo_lumo_gap: Option<f64>,
    pub density_matrix: Matrix,
    pub mo_coefficients: Matrix,
    pub orbital_energies: Vec<f64>,
    pub ks_fock_matrix: Matrix,
    pub local_exchange: LhfExchangeResult,
    pub residual_norm: f64,
    pub density_change: f64,
    pub iterations: usize,
    pub converged: bool,
}
fn trace(a: &Matrix, b: &Matrix) -> f64 {
    a.data.iter().zip(&b.data).map(|(a, b)| a * b).sum()
}
fn validate(hf: &HFResult, basis: &[ContractedGaussian], o: &LhfOptions) -> Result<(), String> {
    let n = basis.len();
    if !hf.converged
        || hf.multiplicity != 1
        || hf.n_occ == 0
        || hf.n_occ > hf.mo_coefficients.cols
        || hf.mo_coefficients.rows != n
        || hf.overlap_matrix.rows != n
        || hf.overlap_matrix.cols != n
        || hf.density_matrix.rows != n
        || hf.core_hamiltonian.rows != n
        || o.max_iterations == 0
        || o.reference_max_iterations == 0
        || o.max_grid_refinements > 8
        || [
            o.energy_tolerance,
            o.residual_tolerance,
            o.density_tolerance,
            o.linear_tolerance,
            o.minimum_linear_eigenvalue,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err("LHF requires a converged closed-shell reference and positive finite convergence controls".into());
    }
    Ok(())
}
fn prepare_grid(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    o: &LhfOptions,
    refinement: usize,
) -> Result<Option<LocalGrid>, String> {
    if hf.n_occ == 1 {
        return Ok(None);
    }
    let extra =
        exchange_workspace(basis.len(), hf.n_occ).ok_or("LHF equation workspace overflow")?;
    let mut grid = o.grid.clone();
    grid.workspace_limit_bytes = grid
        .workspace_limit_bytes
        .checked_sub(extra)
        .ok_or("LHF equation workspace exceeds budget")?;
    grid.radial_points = grid
        .radial_points
        .checked_add(16 * refinement)
        .ok_or("LHF radial grid overflow")?;
    grid.polar_points = grid
        .polar_points
        .checked_add(4 * refinement)
        .ok_or("LHF polar grid overflow")?;
    grid.azimuthal_points = grid
        .azimuthal_points
        .checked_add(8 * refinement)
        .ok_or("LHF azimuthal grid overflow")?;
    Ok(Some(LocalGrid::new(
        atoms,
        basis,
        &hf.mo_coefficients,
        &hf.overlap_matrix,
        &grid,
    )?))
}
fn exchange(
    grid: Option<&LocalGrid>,
    c: &Matrix,
    no: usize,
    j: &Matrix,
    k: &Matrix,
    o: &LhfOptions,
    refinement: usize,
) -> Result<LhfExchangeResult, String> {
    let extra = exchange_workspace(c.rows, no).ok_or("LHF equation workspace overflow")?;
    let p = if let Some(grid) = grid {
        build_exchange(
            grid,
            c,
            no,
            &k.scale(0.5),
            o.linear_tolerance,
            o.minimum_linear_eigenvalue,
            o.grid.workspace_limit_bytes,
        )?
    } else {
        // One occupied spatial orbital: exact v_x(r)=-J_one(r), including virtual matrix elements.
        ExchangePoint {
            potential: j.scale(-0.5),
            slater: j.scale(-0.5),
            constants: Matrix::zeros(1, 1),
            linear_residual: 0.,
            smallest_linear_eigenvalue: 1.,
        }
    };
    Ok(LhfExchangeResult {
        potential: p.potential,
        slater: p.slater,
        correction_constants: p.constants,
        linear_residual: p.linear_residual,
        minimum_linear_eigenvalue: p.smallest_linear_eigenvalue,
        grid_overlap_error: grid.map_or(0., |g| g.overlap_error),
        grid_points: grid.map_or(0, |g| g.points.len()),
        cached_kernel_points: grid.map_or(0, |g| g.cached_points),
        estimated_additional_workspace_bytes: grid
            .map_or(0, |g| g.estimated_workspace_bytes + extra),
        grid_refinements: refinement,
    })
}
fn canonicalize(c: &Matrix, f: &Matrix, no: usize) -> (Matrix, Vec<f64>) {
    let fm = c.transpose().dot(f).dot(c);
    let mut rotation = Matrix::identity(c.cols);
    let mut energies = vec![0.; c.cols];
    for (start, end) in [(0, no), (no, c.cols)] {
        if start == end {
            continue;
        }
        let block = Matrix::from_vec(
            end - start,
            end - start,
            (start..end)
                .flat_map(|i| (start..end).map(move |j| (i, j)))
                .map(|(i, j)| fm[(i, j)])
                .collect(),
        );
        let (values, vectors) = block.eigh();
        for i in start..end {
            energies[i] = values[i - start];
            for j in start..end {
                rotation[(i, j)] = vectors[(i - start, j - start)];
            }
        }
    }
    (c.dot(&rotation), energies)
}
fn solve(
    hf: &HFResult,
    grid: Option<&LocalGrid>,
    o: &LhfOptions,
    refinement: usize,
) -> Result<LHFResult, String> {
    let x = &hf.mo_coefficients;
    let s = &hf.overlap_matrix;
    let h = &hf.core_hamiltonian;
    let no = hf.n_occ;
    let mut c = x.clone();
    let mut previous = None;
    let mut previous_density: Option<Matrix> = None;
    let mut diis = crate::diis::CdiisEngine::new(8);
    let (mut residual_norm, mut density_change) = (f64::INFINITY, f64::INFINITY);
    for iteration in 1..=o.max_iterations {
        let p = c.occupied_density(no, 2.);
        let (j, k) = contract_jk(&p, &[&p], &hf.eri, c.rows);
        let hf_fock = h.add(&j).sub(&k[0].scale(0.5));
        let energy = hf.nuclear_repulsion_energy + 0.5 * trace(&p, &h.add(&hf_fock));
        let mut local = exchange(grid, &c, no, &j, &k[0], o, refinement)?;
        let fock = h.add(&j).add(&local.potential);
        let residual = x
            .transpose()
            .dot(&fock.dot(&p).dot(s).sub(&s.dot(&p).dot(&fock)))
            .dot(x);
        residual_norm = residual.data.iter().map(|v| v * v).sum::<f64>().sqrt();
        density_change = previous_density.as_ref().map_or(f64::INFINITY, |old| {
            p.data
                .iter()
                .zip(&old.data)
                .fold(0f64, |m, (a, b)| m.max((a - b).abs()))
        });
        if previous.is_some_and(|old: f64| (energy - old).abs() <= o.energy_tolerance)
            && residual_norm <= o.residual_tolerance
            && density_change <= o.density_tolerance
        {
            let (coefficients, energies) = canonicalize(&c, &fock, no);
            let transform = c
                .leading_columns(no)
                .transpose()
                .dot(s)
                .dot(&coefficients.leading_columns(no));
            local.correction_constants = transform
                .transpose()
                .dot(&local.correction_constants)
                .dot(&transform);
            let co = coefficients.leading_columns(no);
            let expected = co
                .transpose()
                .dot(&local.potential.add(&k[0].scale(0.5)))
                .dot(&co);
            local.linear_residual = local
                .correction_constants
                .sub(&expected)
                .data
                .iter()
                .fold(0f64, |m, v| m.max(v.abs()));
            if local.linear_residual > o.linear_tolerance {
                return Err(format!(
                    "local-grid final LHF equation residual {:e} exceeds {}",
                    local.linear_residual, o.linear_tolerance
                ));
            }
            let gap = (no < energies.len()).then(|| energies[no] - energies[no - 1]);
            if gap.is_some_and(|v| v < 0.) {
                return Err("LHF reached a non-Aufbau stationary determinant".into());
            }
            return Ok(LHFResult {
                total_energy: energy,
                electronic_energy: energy - hf.nuclear_repulsion_energy,
                nuclear_energy: hf.nuclear_repulsion_energy,
                exchange_energy: -0.25 * trace(&p, &k[0]),
                reference_hf_energy: hf.total_energy,
                homo_lumo_gap: gap,
                density_matrix: p,
                mo_coefficients: coefficients,
                orbital_energies: energies,
                ks_fock_matrix: fock,
                local_exchange: local,
                residual_norm,
                density_change,
                iterations: iteration,
                converged: true,
            });
        }
        previous = Some(energy);
        previous_density = Some(p);
        let orthofock = x.transpose().dot(&fock).dot(x);
        diis.push(orthofock.clone(), residual);
        let next = diis.extrapolate().unwrap_or(orthofock);
        c = x.dot(&next.eigh().1);
    }
    Err(format!(
        "LHF did not converge in {} iterations: residual {residual_norm:e}, density change {density_change:e}",
        o.max_iterations
    ))
}
fn refine<T>(
    o: &LhfOptions,
    mut attempt: impl FnMut(usize) -> Result<T, String>,
) -> Result<T, String> {
    let mut last = String::new();
    for refinement in 0..=o.max_grid_refinements {
        match attempt(refinement) {
            Ok(value) => return Ok(value),
            Err(error) if error.starts_with("local-grid") => last = error,
            Err(error) => return Err(error),
        }
    }
    Err(last)
}
pub fn run_lhf_from_reference(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    o: &LhfOptions,
) -> Result<LHFResult, String> {
    validate(hf, basis, o)?;
    refine(o, |refinement| {
        let grid = prepare_grid(atoms, basis, hf, o, refinement)?;
        solve(hf, grid.as_ref(), o, refinement)
    })
}
/// Local potential at the reference orbitals, before self-consistency.
pub fn evaluate_lhf_exchange(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    o: &LhfOptions,
) -> Result<LhfExchangeResult, String> {
    validate(hf, basis, o)?;
    refine(o, |refinement| {
        let grid = prepare_grid(atoms, basis, hf, o, refinement)?;
        let (j, k) = contract_jk(
            &hf.density_matrix,
            &[&hf.density_matrix],
            &hf.eri,
            basis.len(),
        );
        exchange(
            grid.as_ref(),
            &hf.mo_coefficients,
            hf.n_occ,
            &j,
            &k[0],
            o,
            refinement,
        )
    })
}
pub fn run_lhf_with_options(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    o: &LhfOptions,
) -> Result<LHFResult, String> {
    let ne = atoms.iter().map(|a| a.charge).sum::<f64>().round() as i64 - i64::from(charge);
    if ne <= 0 || ne % 2 != 0 {
        return Err("LHF currently requires a positive even electron count".into());
    }
    let hf = run_scf_general(
        atoms,
        basis,
        charge,
        o.reference_max_iterations,
        o.energy_tolerance.min(1e-12),
        ConvAlgo::Adiissoscf,
        None,
    )?;
    run_lhf_from_reference(atoms, basis, &hf, o)
}
pub fn run_lhf_scf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Result<LHFResult, String> {
    run_lhf_with_options(
        atoms,
        basis,
        charge,
        &LhfOptions {
            max_iterations: max_iter,
            energy_tolerance: tol,
            residual_tolerance: tol.max(1e-8),
            density_tolerance: tol.max(1e-9),
            ..Default::default()
        },
    )
}
