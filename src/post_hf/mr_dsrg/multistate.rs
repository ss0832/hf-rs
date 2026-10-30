// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Rank-two SA/MS/XMS/DWMS perturbative effective Hamiltonians. Only active CI
//! strings and exact ensemble cumulants are used. Three-index integrals remain
//! factorized through every source transformation; model states are transported
//! exactly under each semicanonical change of frame.

use super::{
    density_cumulants::{
        CiEnsembleMember, DEFAULT_CUMULANT_MEMORY_BUDGET as BUDGET, exact_density_cumulants,
    },
    gno::{GnoAlgebra, GnoReference},
    imsrg2::No2bOperator,
    mrdsrg::{MultiStateDSRGResult, MultiStateDSRGVariant as Variant},
    three_index::factorized_mrpt2,
    three_reference::{cholesky_factors, generalized_fock, physical_active},
};
use crate::{
    Matrix,
    context::PostHFContext,
    df::DfTensor,
    error::HfError,
    fci::{
        engine::FciSolverOptions, general_operator::SpinOrbitalHamiltonian,
        orbital_rotation::rotate_ci, string::DeterminantSpace,
    },
    mo_integrals::transform_1e_ao_to_mo,
    options::{ActiveSpace, PostHFOptions, ReferenceRelaxation, ResolvedActiveSpace},
    output::{PostHFOutput, State},
    post_hf::{PostHFMethod, casscf_engine::solve_cas},
    registry::{ContextMethod, MethodId, MethodMeta},
    scf::HFResult,
};

fn failure(detail: impl Into<String>) -> HfError {
    HfError::PostHf {
        method: "Multistate DSRG-PT2".into(),
        detail: detail.into(),
    }
}
pub fn method_name(variant: Variant) -> &'static str {
    match variant {
        Variant::StateAverage => "SA-DSRG-PT2",
        Variant::StateAverageContracted => "SA-DSRG-PT2c",
        Variant::MS => "MS-DSRG-PT2",
        Variant::XMS => "XMS-DSRG-PT2",
        Variant::DWMS { .. } => "DWMS-DSRG-PT2",
    }
}

/// One source operator in C_source=C_anchor*rotation. Its coefficients are bare
/// active-space coefficients in that source frame, including the core scalar.
pub struct SourceOperator {
    pub operator: No2bOperator,
    pub rotation: Matrix,
    pub active_rotation: Matrix,
    pub sector_count: usize,
    pub peak_sector_terms: usize,
}

fn validate_mo(
    h: &Matrix,
    factors: &DfTensor,
    nuclear: f64,
    active: usize,
    flow: f64,
    budget: usize,
    ci_values: usize,
) -> Result<(), HfError> {
    let n = h.rows;
    if n == 0
        || n > i16::MAX as usize / 2
        || n != h.cols
        || h.data.len() != n * n
        || factors.n_ao != n
        || factors.b.cols != n * (n + 1) / 2
        || factors.b.rows.checked_mul(factors.b.cols) != Some(factors.b.data.len())
        || h.data.iter().chain(&factors.b.data).any(|v| !v.is_finite())
        || !nuclear.is_finite()
        || !flow.is_finite()
        || flow < 0.
        || active > n
    {
        return Err(failure(
            "invalid MO Hamiltonian/factors, flow or active dimension",
        ));
    }
    for p in 0..n {
        for q in 0..p {
            if (h[(p, q)] - h[(q, p)]).abs() > 1e-10 {
                return Err(failure("MO Hamiltonian must be Hermitian"));
            }
        }
    }
    let ns = 2 * active;
    let gno = if ns == 0 {
        0
    } else {
        GnoAlgebra::estimated_workspace_bytes(ns)?
    };
    if 8 * gno as u128
        + 32 * factors.b.data.len() as u128
        + 64 * (n as u128).pow(2)
        + 320 * (ns as u128 + 1).pow(4)
        + 64 * ci_values as u128
        > budget as u128
    {
        return Err(failure(
            "ensemble/active source workspace exceeds memory budget",
        ));
    }
    Ok(())
}

fn check_space(
    n: usize,
    space: &ResolvedActiveSpace,
    vectors: &[Vec<f64>],
    weights: &[f64],
) -> Result<DeterminantSpace, HfError> {
    let nc = space.inactive.len();
    let na = space.active.len();
    if space.inactive != (0..nc).collect::<Vec<_>>()
        || space.active != (nc..nc + na).collect::<Vec<_>>()
        || space.external != (nc + na..n).collect::<Vec<_>>()
        || nc + na > n
        || space.n_alpha != space.n_beta
        || space.n_alpha > na
        || vectors.is_empty()
        || weights.len() != vectors.len()
        || weights.iter().any(|v| !v.is_finite() || *v < 0.)
        || (weights.iter().sum::<f64>() - 1.).abs() > 1e-10
    {
        return Err(failure(
            "requires ordered spin-balanced CAS states and normalized nonnegative ensemble weights",
        ));
    }
    // The exact CI constructor checks the spin-string dimension before storage.
    let one = Matrix::zeros(na, na);
    let np = na * (na + 1) / 2;
    let two = crate::mo_integrals::MoEriTensor {
        n_basis: na,
        n_pairs: np,
        data: vec![0.; np * np],
    };
    let fci = crate::fci::engine::FciHamiltonian::new(&one, &two, space.n_alpha, space.n_beta, 0.)?;
    let dim = fci.space.n_determinants();
    if vectors
        .iter()
        .any(|v| v.len() != dim || v.iter().any(|x| !x.is_finite()))
    {
        return Err(failure("invalid CAS state vectors"));
    }
    for (i, a) in vectors.iter().enumerate() {
        if fci.spin_squared_action(a)?.iter().any(|v| v.abs() > 1e-9) {
            return Err(failure(
                "the spatial ensemble Fock requires singlet CAS states",
            ));
        }
        for (j, b) in vectors[..=i].iter().enumerate() {
            let overlap: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            if (overlap - if i == j { 1. } else { 0. }).abs() > 1e-9 {
                return Err(failure("CAS model states must be orthonormal"));
            }
        }
    }
    Ok(fci.space)
}

/// Exact ensemble Fdiag transformation. No state-specific cumulant averaging
/// is used: the averaged RDMs are formed before constructing their cumulants.
pub fn source_operator(
    h: &Matrix,
    factors: &DfTensor,
    nuclear: f64,
    space: &ResolvedActiveSpace,
    vectors: &[Vec<f64>],
    weights: &[f64],
    flow: f64,
    budget: usize,
) -> Result<SourceOperator, HfError> {
    validate_mo(
        h,
        factors,
        nuclear,
        space.active.len(),
        flow,
        budget,
        vectors.iter().map(Vec::len).sum(),
    )?;
    let n = h.rows;
    if n == 0
        || n != h.cols
        || factors.n_ao != n
        || h.data.len() != n * n
        || h.data.iter().any(|v| !v.is_finite())
    {
        return Err(failure("invalid MO Hamiltonian/factor dimensions"));
    }
    let ns = 2 * space.active.len();
    let estimate = if ns == 0 {
        0
    } else {
        GnoAlgebra::estimated_workspace_bytes(ns)?
    };
    if 8 * estimate as u128
        + 32 * factors.b.data.len() as u128
        + 64 * (n as u128).pow(2)
        + 320 * (ns as u128 + 1).pow(4)
        > budget as u128
    {
        return Err(failure(
            "ensemble/active source workspace exceeds memory budget",
        ));
    }
    let dets = check_space(n, space, vectors, weights)?;
    let members: Vec<_> = vectors
        .iter()
        .zip(weights)
        .map(|(v, &weight)| CiEnsembleMember {
            space: &dets,
            coefficients: v,
            weight,
        })
        .collect();
    let density = exact_density_cumulants(&members, budget)?;
    let nc = space.inactive.len();
    let na = space.active.len();
    let mut d = Matrix::zeros(n, n);
    for p in 0..nc {
        d[(p, p)] = 2.;
    }
    for p in 0..na {
        for q in 0..na {
            d[(nc + p, nc + q)] =
                density.gamma1[(2 * p, 2 * q)] + density.gamma1[(2 * p + 1, 2 * q + 1)];
        }
    }
    let f = generalized_fock(h, factors, &d);
    drop(density);
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
    let rotated: Vec<_> = vectors
        .iter()
        .map(|c| rotate_ci(&dets, c, &ua, budget))
        .collect::<Result<_, _>>()?;
    let members: Vec<_> = rotated
        .iter()
        .zip(weights)
        .map(|(v, &weight)| CiEnsembleMember {
            space: &dets,
            coefficients: v,
            weight,
        })
        .collect();
    let density = exact_density_cumulants(&members, budget)?;
    let reference = GnoReference::new(
        &density,
        2 * n,
        &(0..2 * nc).collect::<Vec<_>>(),
        &(2 * nc..2 * (nc + na)).collect::<Vec<_>>(),
    )?;
    let rotated_factors = factors.transform(&u)?;
    let rotated_h = u.transpose().dot(h).dot(&u);
    let f = u.transpose().dot(&f).dot(&u);
    let mut fs = Matrix::zeros(2 * n, 2 * n);
    for p in 0..2 * n {
        for q in 0..2 * n {
            if p % 2 == q % 2 {
                fs[(p, q)] = f[(p / 2, q / 2)];
            }
        }
    }
    let correction = factorized_mrpt2(&rotated_factors, &fs, &reference, flow, budget)?;
    let physical = physical_active(&rotated_h, &rotated_factors, space, nuclear)?;
    Ok(SourceOperator {
        operator: physical.add_scaled(&correction.active_operator, 1.),
        rotation: u,
        active_rotation: ua,
        sector_count: correction.sectors,
        peak_sector_terms: correction.peak_sector_terms,
    })
}

pub struct MultistateSolution {
    pub energies: Vec<f64>,
    pub ci_coefficients: Vec<Vec<f64>>,
    pub orbital_rotation: Matrix,
    /// Projection into the input/rotated model states. For uncontracted SA the
    /// reported energies instead come from the full active-space operator.
    pub model_hamiltonian: Matrix,
    pub model_rotation: Matrix,
    pub weights: Matrix,
    pub source_transformations: usize,
    pub peak_sector_terms: usize,
}

fn model_matrix(
    op: &No2bOperator,
    dets: &DeterminantSpace,
    vectors: &[Vec<f64>],
    budget: usize,
) -> Result<Matrix, HfError> {
    let engine = SpinOrbitalHamiltonian::new(op, dets, budget)?;
    let mut matrix = Matrix::zeros(vectors.len(), vectors.len());
    for (j, c) in vectors.iter().enumerate() {
        let applied = engine.apply(c)?;
        for (i, b) in vectors.iter().enumerate() {
            matrix[(i, j)] = b.iter().zip(&applied).map(|(x, y)| x * y).sum();
        }
    }
    Ok(matrix)
}

/// Rank-two multistate effective Hamiltonian, Eq. (15) of JCP 150, 144107.
/// Full-CAS SA and contracted SA have distinct variants. MS/XMS/DWMS use the
/// selected-state model space; no empirical change to the reference spectrum
/// or physical denominator is introduced.
pub fn multistate_effective(
    h: &Matrix,
    factors: &DfTensor,
    nuclear: f64,
    space: &ResolvedActiveSpace,
    input_vectors: &[Vec<f64>],
    state_weights: &[f64],
    variant: Variant,
    flow: f64,
    ci_options: &FciSolverOptions,
    budget: usize,
) -> Result<MultistateSolution, HfError> {
    validate_mo(
        h,
        factors,
        nuclear,
        space.active.len(),
        flow,
        budget,
        input_vectors.iter().map(Vec::len).sum(),
    )?;
    let dets = check_space(h.rows, space, input_vectors, state_weights)?;
    let roots = input_vectors.len();
    if ci_options.n_roots != roots {
        return Err(failure("CI and model root counts disagree"));
    }
    let mut vectors = input_vectors.to_vec();
    let mut model_rotation = Matrix::identity(roots);
    if matches!(variant, Variant::XMS) {
        if state_weights
            .iter()
            .any(|w| (*w - 1. / roots as f64).abs() > 1e-10)
        {
            return Err(failure("XMS requires equal model-state weights"));
        }
        let members: Vec<_> = vectors
            .iter()
            .zip(state_weights)
            .map(|(c, &weight)| CiEnsembleMember {
                space: &dets,
                coefficients: c,
                weight,
            })
            .collect();
        let density = exact_density_cumulants(&members, budget)?;
        let nc = space.inactive.len();
        let na = space.active.len();
        let mut d = Matrix::zeros(h.rows, h.rows);
        for p in 0..nc {
            d[(p, p)] = 2.;
        }
        for p in 0..na {
            for q in 0..na {
                d[(nc + p, nc + q)] =
                    density.gamma1[(2 * p, 2 * q)] + density.gamma1[(2 * p + 1, 2 * q + 1)];
            }
        }
        let f = generalized_fock(h, factors, &d);
        let mut op = No2bOperator {
            zero: 0.,
            one: Matrix::zeros(2 * na, 2 * na),
            two: vec![0.; (2 * na).pow(4)],
        };
        for p in 0..2 * na {
            for q in 0..2 * na {
                if p % 2 == q % 2 {
                    op.one[(p, q)] = f[(nc + p / 2, nc + q / 2)];
                }
            }
        }
        let (_, rotation) = model_matrix(&op, &dets, &vectors, budget)?.eigh();
        vectors = (0..roots)
            .map(|j| {
                (0..dets.n_determinants())
                    .map(|k| {
                        (0..roots)
                            .map(|i| input_vectors[i][k] * rotation[(i, j)])
                            .sum()
                    })
                    .collect()
            })
            .collect();
        model_rotation = rotation;
    }
    let physical = physical_active(h, factors, space, nuclear)?;
    let model_physical = model_matrix(&physical, &dets, &vectors, budget)?;
    let mut weights = Matrix::zeros(roots, roots);
    match variant {
        Variant::StateAverage | Variant::StateAverageContracted => {
            for i in 0..roots {
                for j in 0..roots {
                    weights[(i, j)] = state_weights[j];
                }
            }
        }
        Variant::DWMS { zeta } => {
            if !zeta.is_finite() || zeta < 0. {
                return Err(failure("DWMS zeta must be finite and nonnegative"));
            }
            for i in 0..roots {
                for j in 0..roots {
                    let gap = model_physical[(i, i)] - model_physical[(j, j)];
                    weights[(i, j)] = if zeta == 0. {
                        1.
                    } else {
                        (-zeta * gap * gap).exp()
                    };
                }
                let sum: f64 = (0..roots).map(|j| weights[(i, j)]).sum();
                for j in 0..roots {
                    weights[(i, j)] /= sum;
                }
            }
        }
        _ => {
            for i in 0..roots {
                weights[(i, i)] = 1.;
            }
        }
    }
    let common = matches!(
        variant,
        Variant::StateAverage | Variant::StateAverageContracted
    ) || matches!(variant, Variant::DWMS { zeta } if zeta == 0.);
    let mut matrix = Matrix::zeros(roots, roots);
    let mut peak_sector_terms = 0;
    for j in 0..if common { 1 } else { roots } {
        let weight: Vec<_> = (0..roots).map(|i| weights[(j, i)]).collect();
        let source = source_operator(h, factors, nuclear, space, &vectors, &weight, flow, budget)?;
        peak_sector_terms = peak_sector_terms.max(source.peak_sector_terms);
        let rotated = vectors
            .iter()
            .map(|c| rotate_ci(&dets, c, &source.active_rotation, budget))
            .collect::<Result<Vec<_>, _>>()?;
        let projected = model_matrix(&source.operator, &dets, &rotated, budget)?;
        if common {
            matrix = projected;
            if variant == Variant::StateAverage {
                let solution = SpinOrbitalHamiltonian::new(&source.operator, &dets, budget)?
                    .solve(ci_options)?;
                return Ok(MultistateSolution {
                    energies: solution.states.iter().map(|s| s.energy).collect(),
                    ci_coefficients: solution.states.into_iter().map(|s| s.vector).collect(),
                    orbital_rotation: source.rotation,
                    model_hamiltonian: matrix,
                    model_rotation,
                    weights,
                    source_transformations: 1,
                    peak_sector_terms,
                });
            }
        } else {
            for i in 0..roots {
                matrix[(i, j)] = projected[(i, j)];
            }
        }
    }
    for i in 0..roots {
        for j in 0..i {
            let value = 0.5 * (matrix[(i, j)] + matrix[(j, i)]);
            matrix[(i, j)] = value;
            matrix[(j, i)] = value;
        }
    }
    let (energies, mixing) = matrix.eigh();
    let coefficients = (0..roots)
        .map(|j| {
            (0..dets.n_determinants())
                .map(|k| (0..roots).map(|i| vectors[i][k] * mixing[(i, j)]).sum())
                .collect()
        })
        .collect();
    Ok(MultistateSolution {
        energies,
        ci_coefficients: coefficients,
        orbital_rotation: Matrix::identity(h.rows),
        model_hamiltonian: matrix,
        model_rotation,
        weights,
        source_transformations: if common { 1 } else { roots },
        peak_sector_terms,
    })
}

pub(crate) fn validate_options(
    hf: &HFResult,
    options: &PostHFOptions,
) -> Result<ResolvedActiveSpace, HfError> {
    options.validate()?;
    if !hf.converged
        || options.frozen_core != 0
        || options.frozen_virtual != 0
        || options.dsrg.relaxation() != ReferenceRelaxation::None
        || options
            .cas
            .target_s_squared
            .is_some_and(|s| s.abs() > 1e-10)
        || (options.density_fitting && options.cas.optimize_orbitals)
    {
        return Err(failure(
            "requires RHF singlets without frozen orbitals; iterative reference relaxation and DF orbital optimization are unsupported",
        ));
    }
    let active = options
        .active
        .clone()
        .unwrap_or_else(|| ActiveSpace::new(0, 0, 0))
        .resolve(hf.mo_coefficients.cols, hf.n_occ, hf.n_occ)?;
    let ns = 2 * active.active.len();
    if active.n_alpha != active.n_beta || (ns == 0 && options.n_roots != 1) {
        return Err(failure(
            "requires a spin-balanced active space; an empty CAS has one state",
        ));
    }
    let n = hf.mo_coefficients.cols as u128;
    let active_bytes = if ns == 0 {
        0
    } else {
        GnoAlgebra::estimated_workspace_bytes(ns)?
    };
    if 8 * active_bytes as u128 + 64 * n * n + 320 * (ns as u128 + 1).pow(4) > BUDGET as u128 {
        return Err(failure("active ensemble workspace exceeds memory budget"));
    }
    if options.cas.optimize_orbitals
        && 64 * n.max(hf.mo_coefficients.rows as u128).pow(4) > BUDGET as u128
    {
        return Err(failure(
            "conventional CASSCF starting-orbital workspace exceeds memory budget",
        ));
    }
    Ok(active)
}

/// Shared molecular adapter. Only the conventional starting-orbital optimizer
/// uses full integrals; all subsequent CASCI and DSRG stages use factors.
pub fn solve_multistate(
    hf: &HFResult,
    factors: &DfTensor,
    options: &PostHFOptions,
    variant: Variant,
    initial_coefficients: Option<&Matrix>,
) -> Result<MultiStateDSRGResult, HfError> {
    let resolved = validate_options(hf, options)?;
    let n = hf.mo_coefficients.cols;
    let nc = resolved.inactive.len();
    let na = resolved.active.len();
    let mut c = if let Some(c) = initial_coefficients {
        if c.rows != hf.mo_coefficients.rows
            || c.cols != n
            || c.data.len() != c.rows * n
            || c.data.iter().any(|v| !v.is_finite())
        {
            return Err(failure("invalid starting-orbital dimensions/values"));
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
    let metric = c.transpose().dot(&hf.overlap_matrix).dot(&c);
    for p in 0..n {
        for q in 0..n {
            if (metric[(p, q)] - if p == q { 1. } else { 0. }).abs() > 1e-8 {
                return Err(failure("reference orbitals must be orthonormal"));
            }
        }
    }
    let space = ResolvedActiveSpace {
        n_alpha: resolved.n_alpha,
        n_beta: resolved.n_beta,
        inactive: (0..nc).collect(),
        active: (nc..nc + na).collect(),
        external: (nc + na..n).collect(),
    };
    let factors = factors.transform(&c)?;
    let h = transform_1e_ao_to_mo(&c, &hf.core_hamiltonian);
    let physical = physical_active(&h, &factors, &space, hf.nuclear_repulsion_energy)?;
    let dets = DeterminantSpace::new(na, space.n_alpha, space.n_beta);
    let ci_options = FciSolverOptions {
        n_roots: options.n_roots,
        max_iterations: options.max_iter,
        residual_tol: options.residual_tol,
        target_s_squared: Some(0.),
        memory_budget: BUDGET,
        ..Default::default()
    };
    let roots = SpinOrbitalHamiltonian::new(&physical, &dets, BUDGET)?.solve(&ci_options)?;
    let reference_energies: Vec<_> = roots.states.iter().map(|s| s.energy).collect();
    let vectors: Vec<_> = roots.states.into_iter().map(|s| s.vector).collect();
    let mut weights = if options.cas.state_weights.is_empty() {
        vec![1. / options.n_roots as f64; options.n_roots]
    } else {
        options.cas.state_weights.clone()
    };
    let total: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= total);
    let solution = multistate_effective(
        &h,
        &factors,
        hf.nuclear_repulsion_energy,
        &space,
        &vectors,
        &weights,
        variant,
        options.dsrg.flow_parameter,
        &ci_options,
        BUDGET,
    )?;
    c = c.dot(&solution.orbital_rotation);
    let mut output = PostHFOutput::new(
        method_name(variant),
        hf.total_energy,
        solution.energies[options.root],
    )?;
    output.states = solution
        .energies
        .iter()
        .enumerate()
        .map(|(i, &energy)| State {
            energy,
            s_squared: Some(0.),
            weight: weights[i],
            converged: true,
        })
        .collect();
    output.select_root(options.root)?;
    output
        .components
        .insert("E_CASCI".into(), reference_energies[options.root]);
    output
        .components
        .insert("factor_rank".into(), factors.rank() as f64);
    output.components.insert(
        "source_transformations".into(),
        solution.source_transformations as f64,
    );
    output
        .components
        .insert("model_dimension".into(), options.n_roots as f64);
    output
        .components
        .insert("active_ci_dimension".into(), dets.n_determinants() as f64);
    output.components.insert(
        "contracted_model_space".into(),
        if variant == Variant::StateAverage {
            0.
        } else {
            1.
        },
    );
    output.components.insert(
        "peak_sector_terms".into(),
        solution.peak_sector_terms as f64,
    );
    if let Variant::DWMS { zeta } = variant {
        output
            .components
            .insert("dynamic_weight_parameter".into(), zeta);
    }
    output.validate()?;
    Ok(MultiStateDSRGResult {
        variant,
        state_energies: solution.energies,
        casci_energies: reference_energies,
        effective_hamiltonian: solution.model_hamiltonian,
        flow_param_s: options.dsrg.flow_parameter,
        mo_coefficients: c,
        ci_coefficients: solution.ci_coefficients,
        model_rotation: solution.model_rotation,
        ensemble_weights: solution.weights,
        output,
    })
}

pub struct MultistateMethod<'a, const KIND: u8> {
    context: &'a PostHFContext<'a>,
    options: PostHFOptions,
}
impl<const KIND: u8> MethodMeta for MultistateMethod<'_, KIND> {
    const METHOD: MethodId = match KIND {
        0 => MethodId::SaDsrgPt2,
        1 => MethodId::SaDsrgPt2C,
        2 => MethodId::MsDsrgPt2,
        3 => MethodId::XmsDsrgPt2,
        _ => MethodId::DwmsDsrgPt2,
    };
}
impl<'a, const KIND: u8> ContextMethod<'a> for MultistateMethod<'a, KIND> {
    fn from_context(
        context: &'a PostHFContext<'a>,
        options: &PostHFOptions,
    ) -> Result<Self, HfError> {
        let mut options = options.clone();
        if options.active.is_none() {
            options.active = Some(ActiveSpace::new(0, 0, 0));
            options.cas.optimize_orbitals = false;
        }
        if options.cas.state_weights.is_empty() && options.n_roots > 0 {
            options.cas.state_weights = vec![1. / options.n_roots as f64; options.n_roots];
        }
        options.cas.target_s_squared = Some(options.cas.target_s_squared.unwrap_or(0.));
        validate_options(context.rhf()?, &options)?;
        if KIND == 3 {
            let sum: f64 = options.cas.state_weights.iter().sum();
            if options
                .cas
                .state_weights
                .iter()
                .any(|w| (w / sum - 1. / options.n_roots as f64).abs() > 1e-10)
            {
                return Err(failure("XMS requires equal model-state weights"));
            }
        }
        if KIND > 4 {
            return Err(failure("unknown multistate formalism"));
        }
        Ok(Self { context, options })
    }
}
impl<const KIND: u8> PostHFMethod for MultistateMethod<'_, KIND> {
    type Output = MultiStateDSRGResult;
    fn compute(&self) -> Result<Self::Output, String> {
        self.evaluate().map_err(|e| e.to_string())
    }
}
impl<const KIND: u8> MultistateMethod<'_, KIND> {
    fn evaluate(&self) -> Result<MultiStateDSRGResult, HfError> {
        let hf = self.context.rhf()?;
        let fitted = if self.options.density_fitting {
            Some(self.context.ints.df_ao()?)
        } else {
            None
        };
        let cd = if fitted.is_none() {
            Some(cholesky_factors(
                hf,
                self.options.dsrg.cholesky_tolerance,
                BUDGET,
            )?)
        } else {
            None
        };
        let factors = if let Some(df) = fitted {
            df.as_ref()
        } else {
            &cd.as_ref().ok_or_else(|| failure("missing factors"))?.0
        };
        let initial = if self.options.cas.optimize_orbitals {
            Some(solve_cas(
                self.context.reference,
                &self.options,
                None,
                true,
            )?)
        } else {
            None
        };
        let variant = match KIND {
            0 => Variant::StateAverage,
            1 => Variant::StateAverageContracted,
            2 => Variant::MS,
            3 => Variant::XMS,
            _ => Variant::DWMS {
                zeta: self.options.dsrg.dynamic_weight_parameter,
            },
        };
        let mut result = solve_multistate(
            hf,
            factors,
            &self.options,
            variant,
            initial.as_ref().map(|cas| &cas.mo_coefficients),
        )?;
        if let Some(cas) = initial {
            result
                .output
                .components
                .insert("E_conventional_CASSCF_orbitals".into(), cas.output.e_total);
        }
        if let Some((_, residual)) = cd {
            result
                .output
                .components
                .insert("cholesky_max_residual".into(), residual);
        }
        Ok(result)
    }
}
