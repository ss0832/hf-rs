// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Factorized CASCI reference and effective-CAS relaxation for THREE-MRPT2.

use super::{
    density_cumulants::{CiEnsembleMember, exact_density_cumulants},
    gno::{GnoAlgebra, GnoReference},
    imsrg2::No2bOperator,
    relaxation::MrReferenceSolution,
    three_index::factorized_mrpt2,
};
use crate::{
    Matrix,
    context::active_integrals_from_mo,
    df::DfTensor,
    error::HfError,
    fci::{
        engine::{FciHamiltonian, FciSolverOptions},
        general_operator::SpinOrbitalHamiltonian,
        orbital_rotation::rotate_ci,
    },
    mo_integrals::{MoEriTensor, transform_1e_ao_to_mo},
    options::{ActiveSpace, PostHFOptions, ReferenceRelaxation, ResolvedActiveSpace},
    scf::{HFResult, eri_index},
};

fn failure(detail: impl Into<String>) -> HfError {
    HfError::PostHf {
        method: "THREE-DSRG-MRPT2".into(),
        detail: detail.into(),
    }
}

/// Pivoted Cholesky accesses columns of the borrowed packed AO ERIs. The budget
/// is checked before each factor allocation; an AO pair supermatrix is absent.
pub fn cholesky_factors(
    hf: &HFResult,
    threshold: f64,
    budget: usize,
) -> Result<(DfTensor, f64), HfError> {
    let n = hf.core_hamiltonian.rows;
    if n == 0
        || n > i16::MAX as usize / 2
        || !threshold.is_finite()
        || threshold < 0.
        || hf.core_hamiltonian.cols != n
        || hf.mo_coefficients.rows != n
        || hf.eri.len() != eri_index(n - 1, n - 1, n - 1, n - 1) + 1
        || hf.eri.iter().any(|v| !v.is_finite())
    {
        return Err(HfError::InvalidInput(
            "invalid Cholesky threshold or AO reference dimensions/integrals".into(),
        ));
    }
    let np = n * (n + 1) / 2;
    if (np as u128) * 64 > budget as u128 {
        return Err(failure("Cholesky workspace exceeds memory budget"));
    }
    let pairs: Vec<_> = (0..n).flat_map(|p| (0..=p).map(move |q| (p, q))).collect();
    let mut diagonal: Vec<_> = pairs
        .iter()
        .map(|&(p, q)| hf.eri[eri_index(p, q, p, q)])
        .collect();
    let scale = diagonal.iter().copied().fold(0f64, f64::max).max(1.);
    if diagonal.iter().any(|v| *v < -1e-12 * scale) {
        return Err(failure("AO Coulomb metric is not positive semidefinite"));
    }
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let residual = loop {
        let (pivot, &maximum) = diagonal
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        if maximum <= threshold || rows.len() == np {
            break maximum.max(0.);
        }
        // Includes flattening and the AO/MO transformation overlap. The active
        // reference/CI budget is checked separately before its allocation.
        if (np as u128) * (32 * (rows.len() as u128 + 1) + 64) > budget as u128 {
            return Err(failure("Cholesky factors exceed memory budget"));
        }
        let (r, s) = pairs[pivot];
        let root = maximum.sqrt();
        let mut row = vec![0.; np];
        for (k, &(p, q)) in pairs.iter().enumerate() {
            row[k] = (hf.eri[eri_index(p, q, r, s)]
                - rows.iter().map(|l| l[k] * l[pivot]).sum::<f64>())
                / root;
        }
        row[pivot] = root;
        for (d, l) in diagonal.iter_mut().zip(&row) {
            *d -= l * l;
            if *d < -1e-10 * scale {
                return Err(failure("nonpositive residual Coulomb metric"));
            }
            *d = d.max(0.);
        }
        rows.push(row);
    };
    let rank = rows.len();
    Ok((
        DfTensor {
            n_ao: n,
            n_aux_original: rank,
            b: Matrix::from_vec(rank, np, rows.into_iter().flatten().collect()),
        },
        residual,
    ))
}

/// Spatial generalized Fock from exact reference one-density and three-index
/// factors. Auxiliary matrix products avoid a full four-index Coulomb tensor.
pub(crate) fn generalized_fock(h: &Matrix, factors: &DfTensor, density: &Matrix) -> Matrix {
    let n = h.rows;
    let mut f = h.clone();
    for k in 0..factors.rank() {
        let mut b = Matrix::zeros(n, n);
        for p in 0..n {
            for q in 0..n {
                b[(p, q)] = factors.b[(k, crate::scf::compound_index(p, q))];
            }
        }
        let charge: f64 = b.data.iter().zip(&density.data).map(|(v, d)| v * d).sum();
        let exchange = b.dot(density).dot(&b);
        for p in 0..n {
            for q in 0..n {
                f[(p, q)] += charge * b[(p, q)] - 0.5 * exchange[(p, q)];
            }
        }
    }
    f
}

pub(crate) fn physical_active(
    h: &Matrix,
    factors: &DfTensor,
    space: &ResolvedActiveSpace,
    nuclear: f64,
) -> Result<No2bOperator, HfError> {
    let a = active_integrals_from_mo(
        h,
        |p, q, r, s| factors.eri(p, q, r, s),
        space.clone(),
        nuclear,
    )?;
    let ns = 2 * space.active.len();
    let mut out = No2bOperator {
        zero: a.core_energy,
        one: Matrix::zeros(ns, ns),
        two: vec![0.; ns.pow(4)],
    };
    for p in 0..ns {
        for q in 0..ns {
            if p % 2 == q % 2 {
                out.one[(p, q)] = a.h[(p / 2, q / 2)];
            }
            for r in 0..ns {
                for s in 0..ns {
                    out.two[((p * ns + q) * ns + r) * ns + s] =
                        crate::context::spin_antisymmetrized(&a.eri, &a.eri, None, p, q, r, s);
                }
            }
        }
    }
    Ok(out)
}

/// The optional coefficients are ordered inactive/active/external orbitals from
/// a conventional CASSCF reference. CASCI and every subsequent MRPT2/reference
/// update use the supplied factors, including their approximation to the metric.
pub fn solve_three_reference(
    hf: &HFResult,
    ao_factors: &DfTensor,
    options: &PostHFOptions,
    initial_coefficients: Option<&Matrix>,
    budget: usize,
) -> Result<(MrReferenceSolution, usize, usize), HfError> {
    options.validate()?;
    if !hf.converged
        || options.frozen_core != 0
        || options.frozen_virtual != 0
        || options.n_roots != 1
        || options.root != 0
        || options
            .cas
            .target_s_squared
            .is_some_and(|s| s.abs() > 1e-10)
    {
        return Err(failure(
            "requires one converged RHF singlet without frozen orbitals",
        ));
    }
    let nao = hf.mo_coefficients.rows;
    if nao == 0
        || hf.mo_coefficients.cols == 0
        || hf.mo_coefficients.data.len() != nao * hf.mo_coefficients.cols
        || hf.mo_coefficients.data.iter().any(|v| !v.is_finite())
        || hf.core_hamiltonian.rows != nao
        || hf.core_hamiltonian.cols != nao
        || hf.core_hamiltonian.data.len() != nao * nao
        || hf.core_hamiltonian.data.iter().any(|v| !v.is_finite())
        || hf.overlap_matrix.rows != nao
        || hf.overlap_matrix.cols != nao
        || hf.overlap_matrix.data.len() != nao * nao
        || hf.overlap_matrix.data.iter().any(|v| !v.is_finite())
    {
        return Err(failure("invalid AO reference matrices"));
    }
    let requested = options
        .active
        .clone()
        .unwrap_or_else(|| ActiveSpace::new(0, 0, 0));
    let resolved = requested.resolve(hf.mo_coefficients.cols, hf.n_occ, hf.n_occ)?;
    if resolved.n_alpha != resolved.n_beta {
        return Err(failure("requires a spin-balanced singlet CAS"));
    }
    let nc = resolved.inactive.len();
    let na = resolved.active.len();
    let n = hf.mo_coefficients.cols;
    let ns = 2 * n;
    let active_bytes = if na == 0 {
        0
    } else {
        GnoAlgebra::estimated_workspace_bytes(2 * na)?
    };
    if 8 * active_bytes as u128
        + 64 * (n as u128).pow(2)
        + 32 * ao_factors.b.data.len() as u128
        + 320 * (2 * na as u128 + 1).pow(4)
        > budget as u128
    {
        return Err(failure(
            "factorized reference/active workspace exceeds memory budget",
        ));
    }
    if ao_factors.n_ao != hf.mo_coefficients.rows
        || ao_factors.b.cols != ao_factors.n_ao * (ao_factors.n_ao + 1) / 2
        || ao_factors.b.data.len() != ao_factors.b.rows * ao_factors.b.cols
        || ao_factors.b.data.iter().any(|v| !v.is_finite())
    {
        return Err(failure("invalid AO three-index factors"));
    }
    let mut c = if let Some(c) = initial_coefficients {
        if c.rows != hf.mo_coefficients.rows || c.cols != n || c.data.iter().any(|v| !v.is_finite())
        {
            return Err(failure("invalid reference coefficients"));
        }
        c.clone()
    } else {
        let permutation = resolved.permutation();
        let mut c = Matrix::zeros(hf.mo_coefficients.rows, n);
        for p in 0..c.rows {
            for q in 0..n {
                c[(p, q)] = hf.mo_coefficients[(p, permutation[q])];
            }
        }
        c
    };
    let space = ResolvedActiveSpace {
        n_alpha: resolved.n_alpha,
        n_beta: resolved.n_beta,
        inactive: (0..nc).collect(),
        active: (nc..nc + na).collect(),
        external: (nc + na..n).collect(),
    };
    let metric = c.transpose().dot(&hf.overlap_matrix).dot(&c);
    for p in 0..n {
        for q in 0..n {
            if (metric[(p, q)] - if p == q { 1. } else { 0. }).abs() > 1e-8 {
                return Err(failure(
                    "reference orbitals must be orthonormal in the AO metric",
                ));
            }
        }
    }
    let mut factors = ao_factors.transform(&c)?;
    let mut h = transform_1e_ao_to_mo(&c, &hf.core_hamiltonian);
    let spin_h = Matrix::zeros(na, na);
    let np = na * (na + 1) / 2;
    let spin_g = MoEriTensor {
        n_basis: na,
        n_pairs: np,
        data: vec![0.; np * np],
    };
    let spin = FciHamiltonian::new(&spin_h, &spin_g, resolved.n_alpha, resolved.n_beta, 0.)?;
    let ci_options = FciSolverOptions {
        max_iterations: options.max_iter,
        residual_tol: options.residual_tol,
        target_s_squared: Some(0.),
        memory_budget: budget,
        ..Default::default()
    };
    let initial = physical_active(&h, &factors, &space, hf.nuclear_repulsion_energy)?;
    let initial_state = SpinOrbitalHamiltonian::new(&initial, &spin.space, budget)?
        .solve(&ci_options)?
        .states
        .remove(0);
    let initial_reference_energy = initial_state.energy;
    let mut ci = initial_state.vector;
    drop(initial);
    let mode = options.dsrg.relaxation();
    let limit = match mode {
        ReferenceRelaxation::None | ReferenceRelaxation::Once => 1,
        ReferenceRelaxation::Twice => 2,
        ReferenceRelaxation::Iterate => options.dsrg.max_reference_iterations,
    };
    let core: Vec<_> = (0..2 * nc).collect();
    let active: Vec<_> = (2 * nc..2 * (nc + na)).collect();
    let mut first_energy = None;
    let mut previous_energy: Option<f64> = None;
    let mut peak_sector = 0;
    let mut sectors = 0;
    for iteration in 0..limit {
        let mut d = Matrix::zeros(n, n);
        for p in 0..nc {
            d[(p, p)] = 2.;
        }
        let da = spin.rdm(&ci, 1)?;
        for p in 0..na {
            for q in 0..na {
                d[(nc + p, nc + q)] = da[p * na + q];
            }
        }
        let f = generalized_fock(&h, &factors, &d);
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
        ci = rotate_ci(&spin.space, &ci, &ua, budget)?;
        c = c.dot(&u);
        factors = factors.transform(&u)?;
        h = u.transpose().dot(&h).dot(&u);
        let f = u.transpose().dot(&f).dot(&u);
        let mut fs = Matrix::zeros(ns, ns);
        for p in 0..ns {
            for q in 0..ns {
                if p % 2 == q % 2 {
                    fs[(p, q)] = f[(p / 2, q / 2)];
                }
            }
        }
        let density = exact_density_cumulants(
            &[CiEnsembleMember {
                space: &spin.space,
                coefficients: &ci,
                weight: 1.,
            }],
            budget,
        )?;
        let reference = GnoReference::new(&density, ns, &core, &active)?;
        let correction = factorized_mrpt2(
            &factors,
            &fs,
            &reference,
            options.dsrg.flow_parameter,
            budget,
        )?;
        peak_sector = peak_sector.max(correction.peak_sector_terms);
        sectors = sectors.max(correction.sectors);
        let physical = physical_active(&h, &factors, &space, hf.nuclear_repulsion_energy)?;
        let applied = SpinOrbitalHamiltonian::new(&physical, &spin.space, budget)?.apply(&ci)?;
        let source_reference_energy: f64 = ci.iter().zip(applied).map(|(x, y)| x * y).sum();
        let total = source_reference_energy + correction.energy;
        let unrelaxed = *first_energy.get_or_insert(total);
        let mut result = MrReferenceSolution {
            total_energy: total,
            initial_reference_energy,
            source_reference_energy,
            unrelaxed_energy: unrelaxed,
            second_order: correction.energy,
            third_order: 0.,
            lambda2_norm: density.lambda2.full_norm(),
            lambda3_norm: density.lambda3.full_norm(),
            reference_iterations: 0,
            reference_ci_change: 0.,
            reference_energy_change: 0.,
            reference_ci_residual: 0.,
            source_residual: 0.,
            amplitude_residual: 0.,
            amplitude_iterations: 0,
            bch_order: 0,
            mo_coefficients: Matrix::zeros(0, 0),
            ci_coefficients: Vec::new(),
        };
        if mode != ReferenceRelaxation::None {
            let effective = physical.add_scaled(&correction.active_operator, 1.);
            let mut state = SpinOrbitalHamiltonian::new(&effective, &spin.space, budget)?
                .solve(&ci_options)?
                .states
                .remove(0);
            if ci
                .iter()
                .zip(&state.vector)
                .map(|(x, y)| x * y)
                .sum::<f64>()
                < 0.
            {
                state.vector.iter_mut().for_each(|v| *v = -*v);
            }
            result.reference_ci_change = ci
                .iter()
                .zip(&state.vector)
                .map(|(x, y)| (x - y).abs())
                .fold(0f64, f64::max);
            result.reference_ci_residual = state.residual_norm;
            result.reference_iterations = iteration + 1;
            result.total_energy = state.energy;
            result.reference_energy_change = previous_energy
                .map_or((state.energy - unrelaxed).abs(), |old| {
                    (state.energy - old).abs()
                });
            ci = state.vector;
        }
        let converged = result.reference_ci_change <= options.dsrg.reference_ci_tol
            && (previous_energy.is_none()
                || result.reference_energy_change <= options.dsrg.reference_energy_tol);
        if (mode != ReferenceRelaxation::Iterate || converged)
            && (mode != ReferenceRelaxation::Twice || iteration + 1 == limit)
        {
            result.mo_coefficients = c;
            result.ci_coefficients = ci;
            return Ok((result, sectors, peak_sector));
        }
        previous_energy = Some(result.total_energy);
    }
    Err(failure(format!(
        "reference macroiterations did not converge in {limit} updates"
    )))
}
