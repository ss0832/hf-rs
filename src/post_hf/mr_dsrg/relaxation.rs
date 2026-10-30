// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Effective-CAS reference relaxation with exact CI orbital transport.

use super::{
    density_cumulants::{CiEnsembleMember, exact_density_cumulants},
    gno::{GnoAlgebra, GnoReference},
    imsrg2::No2bOperator,
    mr_driven::MrDsrgEquations,
    sr_dsrg::LdsrgOptions,
};
use crate::{
    Matrix,
    context::spin_antisymmetrized,
    error::HfError,
    fci::{
        engine::{FciHamiltonian, FciSolverOptions},
        general_operator::SpinOrbitalHamiltonian,
        orbital_rotation::rotate_ci,
    },
    mo_integrals::{MoEriTensor, transform_1e_ao_to_mo, transform_eri_ao_to_mo},
    options::{PostHFOptions, ReferenceRelaxation},
    post_hf::casscf_engine::CasResult,
    scf::HFResult,
};

fn failure(detail: impl Into<String>) -> HfError {
    HfError::PostHf {
        method: "MR-DSRG reference relaxation".into(),
        detail: detail.into(),
    }
}

/// Final reference coordinates and diagnostics. The saved CI is a reference
/// wavefunction, not the correlated MR density or a post-HF response density.
pub struct MrReferenceSolution {
    pub total_energy: f64,
    pub initial_reference_energy: f64,
    pub source_reference_energy: f64,
    pub unrelaxed_energy: f64,
    pub second_order: f64,
    pub third_order: f64,
    pub lambda2_norm: f64,
    pub lambda3_norm: f64,
    pub reference_iterations: usize,
    pub reference_ci_change: f64,
    pub reference_energy_change: f64,
    pub reference_ci_residual: f64,
    pub source_residual: f64,
    pub amplitude_residual: f64,
    pub amplitude_iterations: usize,
    pub bch_order: usize,
    pub mo_coefficients: Matrix,
    pub ci_coefficients: Vec<f64>,
}

/// Build the physical generalized Fock from the current CI. Only rotations
/// within core, active and external blocks are allowed; they preserve the CAS.
fn semicanonicalize(
    hf: &HFResult,
    c: &mut Matrix,
    ci: &mut Vec<f64>,
    spin: &FciHamiltonian<'_>,
    nc: usize,
    budget: usize,
) -> Result<(), HfError> {
    let n = c.cols;
    let na = spin.space.n_orbitals;
    let mut d = Matrix::zeros(n, n);
    for i in 0..nc {
        d[(i, i)] = 2.;
    }
    let active_density = spin.rdm(ci, 1)?;
    for p in 0..na {
        for q in 0..na {
            d[(nc + p, nc + q)] = active_density[p * na + q];
        }
    }
    let mut f = transform_1e_ao_to_mo(c, &hf.core_hamiltonian);
    {
        let g = transform_eri_ao_to_mo(c, &hf.eri);
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        f[(p, q)] += d[(r, s)] * (g.get(p, q, r, s) - 0.5 * g.get(p, r, s, q));
                    }
                }
            }
        }
    }
    let mut u = Matrix::identity(n);
    for (start, end) in [(0, nc), (nc, nc + na), (nc + na, n)] {
        if start == end {
            continue;
        }
        let mut block = Matrix::zeros(end - start, end - start);
        for p in start..end {
            for q in start..end {
                block[(p - start, q - start)] = f[(p, q)];
            }
        }
        let (_, v) = block.eigh();
        for p in start..end {
            for q in start..end {
                u[(p, q)] = v[(p - start, q - start)];
            }
        }
    }
    let mut ua = Matrix::zeros(na, na);
    for p in 0..na {
        for q in 0..na {
            ua[(p, q)] = u[(nc + p, nc + q)];
        }
    }
    *ci = rotate_ci(&spin.space, ci, &ua, budget)?;
    *c = c.dot(&u);
    Ok(())
}

/// ORDER=2/3/4/5 selects MRPT2/MRPT3/MR-LDSRG(2)/its quadratic-BCH variant. A macroiteration rebuilds
/// the reference densities, semicanonical denominators and driven equations.
/// The effective operator retains all spin-orbital two-body symmetries; it is
/// never folded into a Coulomb ERI container. Full-space temporaries are released
/// before CI diagonalization and before the next macroiteration.
pub fn solve_reference(
    hf: &HFResult,
    cas: CasResult,
    order: u8,
    options: &PostHFOptions,
    budget: usize,
) -> Result<MrReferenceSolution, HfError> {
    options.validate()?;
    if options.density_fitting
        || options.frozen_core != 0
        || options.frozen_virtual != 0
        || options.root != 0
        || options.n_roots != 1
        || options
            .cas
            .target_s_squared
            .is_some_and(|s| s.abs() > 1e-10)
    {
        return Err(HfError::Unsupported {
            method: "MR-DSRG".into(),
            detail: "requires one full-integral singlet without frozen orbitals".into(),
        });
    }
    let n = cas.mo_coefficients.cols;
    let ns = 2 * n;
    let nc = cas.active_space.inactive.len();
    let na = cas.active_space.active.len();
    if !cas.output.e_total.is_finite()
        || !matches!(order, 2..=5)
        || cas.states.len() != 1
        || n == 0
        || cas.active_space.n_alpha != cas.active_space.n_beta
        || cas.active_space.inactive != (0..nc).collect::<Vec<_>>()
        || cas.active_space.active != (nc..nc + na).collect::<Vec<_>>()
        || cas.active_space.external != (nc + na..n).collect::<Vec<_>>()
        || cas.mo_coefficients.rows != hf.core_hamiltonian.rows
        || cas.mo_coefficients.data.iter().any(|v| !v.is_finite())
    {
        return Err(HfError::InvalidInput(
            "MR relaxation requires one ordered singlet CAS reference".into(),
        ));
    }
    if GnoAlgebra::estimated_workspace_bytes(ns)? > budget {
        return Err(HfError::Unsupported {
            method: "MR-DSRG".into(),
            detail: "operator/density workspace exceeds memory budget".into(),
        });
    }
    let initial_reference_energy = cas.output.e_total;
    let hspin = Matrix::zeros(na, na);
    let np = na * (na + 1) / 2;
    let gspin = MoEriTensor {
        n_basis: na,
        n_pairs: np,
        data: vec![0.; np * np],
    };
    let spin = FciHamiltonian::new(
        &hspin,
        &gspin,
        cas.active_space.n_alpha,
        cas.active_space.n_beta,
        0.,
    )?;
    let mut c = cas.mo_coefficients;
    let mut ci = cas
        .states
        .into_iter()
        .next()
        .ok_or_else(|| failure("missing CAS state"))?
        .vector;
    // Drop the remaining CAS RDM/gradient arrays before allocating GNO tensors.
    drop(cas.output);
    drop(cas.averaged_one_rdm);
    drop(cas.averaged_two_rdm);
    drop(cas.state_one_rdms);
    drop(cas.state_two_rdms);
    drop(cas.generalized_fock);
    let mode = options.dsrg.relaxation();
    let limit = match mode {
        ReferenceRelaxation::None | ReferenceRelaxation::Once => 1,
        ReferenceRelaxation::Twice => 2,
        ReferenceRelaxation::Iterate => options.dsrg.max_reference_iterations,
    };
    let amplitude_options = LdsrgOptions {
        max_iter: options.max_iter,
        residual_tolerance: options.residual_tol,
        energy_tolerance: options.energy_tol,
        bch_tolerance: options.dsrg.bch_tolerance,
        bch_max_order: options.dsrg.bch_max_order,
    };
    let ci_options = FciSolverOptions {
        max_iterations: options.max_iter,
        residual_tol: options.residual_tol,
        target_s_squared: Some(0.),
        memory_budget: budget,
        ..FciSolverOptions::default()
    };
    let core: Vec<_> = (0..2 * nc).collect();
    let active: Vec<_> = (2 * nc..2 * (nc + na)).collect();
    let mut first_energy = None;
    let mut previous_energy: Option<f64> = None;
    for iteration in 0..limit {
        semicanonicalize(hf, &mut c, &mut ci, &spin, nc, budget)?;
        // This scope ensures the full operator/density workspaces never overlap
        // successive macroiterations or the active eigensolver workspace.
        let (effective, mut result) = {
            let density = exact_density_cumulants(
                &[CiEnsembleMember {
                    space: &spin.space,
                    coefficients: &ci,
                    weight: 1.,
                }],
                budget,
            )?;
            let reference = GnoReference::new(&density, ns, &core, &active)?;
            let equations = MrDsrgEquations::new(&reference, budget)?;
            let h = {
                let one = transform_1e_ao_to_mo(&c, &hf.core_hamiltonian);
                let g = transform_eri_ao_to_mo(&c, &hf.eri);
                let mut bare = No2bOperator {
                    zero: hf.nuclear_repulsion_energy,
                    one: Matrix::zeros(ns, ns),
                    two: vec![0.; ns.pow(4)],
                };
                for p in 0..ns {
                    for q in 0..ns {
                        if p % 2 == q % 2 {
                            bare.one[(p, q)] = one[(p / 2, q / 2)];
                        }
                        for r in 0..ns {
                            for s in 0..ns {
                                bare.two[((p * ns + q) * ns + r) * ns + s] =
                                    spin_antisymmetrized(&g, &g, None, p, q, r, s);
                            }
                        }
                    }
                }
                equations.normal_order(&bare)?
            };
            if iteration == 0 && (h.zero - initial_reference_energy).abs() > 1e-9 {
                return Err(failure("CAS energy and normal-ordered reference disagree"));
            }
            let flow = options.dsrg.flow_parameter;
            let mut source_residual = 0.;
            let mut amplitude_residual = 0.;
            let mut amplitude_iterations = 0;
            let mut bch_order = 0;
            let (total, e2, e3, effective) = if order >= 4 {
                let e2 = equations.second_order_energy(&h, flow, amplitude_options)?;
                let solution = if order == 5 {
                    equations.solve_ldsrg2_qc(&h, flow, amplitude_options)?
                } else {
                    equations.solve_ldsrg2(&h, flow, amplitude_options)?
                };
                source_residual = solution.residual_norm;
                amplitude_residual = solution.amplitude_residual_norm;
                amplitude_iterations = solution.iterations;
                bch_order = solution.bch_order;
                let effective = if mode == ReferenceRelaxation::None {
                    None
                } else {
                    Some(equations.active_effective_hamiltonian(&solution.transformed)?)
                };
                (solution.transformed.zero, e2, 0., effective)
            } else if mode == ReferenceRelaxation::None {
                if order == 2 {
                    let e2 = equations.second_order_energy(&h, flow, amplitude_options)?;
                    (h.zero + e2, e2, 0., None)
                } else {
                    let pt =
                        equations.perturbation_through_third_order(&h, flow, amplitude_options)?;
                    (
                        h.zero + pt.second_order + pt.third_order,
                        pt.second_order,
                        pt.third_order,
                        None,
                    )
                }
            } else {
                let effective = equations.perturbative_active_hamiltonian(
                    &h,
                    flow,
                    order as usize,
                    amplitude_options,
                )?;
                let applied =
                    SpinOrbitalHamiltonian::new(&effective, &spin.space, budget)?.apply(&ci)?;
                let total: f64 = ci.iter().zip(applied).map(|(a, b)| a * b).sum();
                let e2 = if order == 2 {
                    total - h.zero
                } else {
                    equations.second_order_energy(&h, flow, amplitude_options)?
                };
                (
                    total,
                    e2,
                    if order == 3 { total - h.zero - e2 } else { 0. },
                    Some(effective),
                )
            };
            let unrelaxed = *first_energy.get_or_insert(total);
            (
                effective,
                MrReferenceSolution {
                    total_energy: total,
                    initial_reference_energy,
                    source_reference_energy: h.zero,
                    unrelaxed_energy: unrelaxed,
                    second_order: e2,
                    third_order: e3,
                    lambda2_norm: density.lambda2.full_norm(),
                    lambda3_norm: density.lambda3.full_norm(),
                    reference_iterations: 0,
                    reference_ci_change: 0.,
                    reference_energy_change: 0.,
                    reference_ci_residual: 0.,
                    source_residual,
                    amplitude_residual,
                    amplitude_iterations,
                    bch_order,
                    mo_coefficients: Matrix::zeros(0, 0),
                    ci_coefficients: Vec::new(),
                },
            )
        };
        if let Some(effective) = effective {
            let solution =
                SpinOrbitalHamiltonian::new(&effective, &spin.space, budget)?.solve(&ci_options)?;
            let mut state = solution
                .states
                .into_iter()
                .next()
                .ok_or_else(|| failure("missing effective CAS root"))?;
            let overlap: f64 = ci.iter().zip(&state.vector).map(|(a, b)| a * b).sum();
            if overlap < 0. {
                for x in &mut state.vector {
                    *x = -*x;
                }
            }
            result.reference_ci_change = ci
                .iter()
                .zip(&state.vector)
                .map(|(a, b)| (a - b).abs())
                .fold(0f64, f64::max);
            result.reference_ci_residual = state.residual_norm;
            result.reference_iterations = iteration + 1;
            result.total_energy = state.energy;
            result.reference_energy_change = previous_energy
                .map_or((state.energy - result.unrelaxed_energy).abs(), |old| {
                    (state.energy - old).abs()
                });
            ci = state.vector;
        }
        let converged = result.reference_ci_change <= options.dsrg.reference_ci_tol
            && (previous_energy.is_none()
                || result.reference_energy_change <= options.dsrg.reference_energy_tol);
        if (mode != ReferenceRelaxation::Iterate || converged)
            && (mode != ReferenceRelaxation::Twice || iteration + 1 == limit) {
                result.mo_coefficients = c;
                result.ci_coefficients = ci;
                return Ok(result);
            }
        previous_energy = Some(result.total_energy);
    }
    Err(failure(format!(
        "reference macroiterations did not converge in {limit} updates"
    )))
}
