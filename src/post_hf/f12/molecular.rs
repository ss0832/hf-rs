// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Mixed-basis F12 tensors with two occupied legs. No union-AO N^4 tensor.
use super::{
    cabs::{CABSBasis, ExtendedFock, extended_fock},
    density_fit::{F12DensityFitOptions, F12PairFitter},
    intermediates::{F12FockSpace, F12NonlocalPairView, F12PairIntegralView, F12PairIntermediates},
    mp2_f12::{F12Approximation, F12Hybrid},
    operators::{GaussianGeminal, GeminalOperator, PreparedGeminalOperator},
};
use crate::{
    basis::{Atom, ContractedGaussian},
    context::AuxBasis,
    error::HfError,
    matrix::Matrix,
    scf::HFResult,
};
use rayon::prelude::*;

/// Equal electron spaces permit exact (a,b,ij) = (b,a,ji) storage. This also
/// applies to [f,T1+T2], whose electron-exchange symmetry is independent of
/// its bra/ket antisymmetry. No integral screening or pair truncation is used.
fn symmetric_pair_tensor(
    target: &Matrix,
    occupied: &Matrix,
    integral: impl Fn(usize, usize, usize, usize) -> Result<f64, HfError> + Sync,
) -> Result<Vec<Matrix>, HfError> {
    let (n, no) = (target.rows, occupied.cols);
    let np = no * no;
    if np == 0 {
        return Ok(Vec::new());
    }
    let count = n as u128 * (n as u128 + 1) / 2 * np as u128;
    if count * 8 > 128 * 1024 * 1024 {
        return Err(HfError::Integral(
            "packed mixed F12 occupied-leg workspace exceeds 128 MiB".into(),
        ));
    }
    let mut rows: Vec<_> = (0..n).map(|a| vec![0.0; (a + 1) * np]).collect();
    let co_t = occupied.transpose();
    rows.par_iter_mut()
        .enumerate()
        .try_for_each(|(a, row)| -> Result<(), HfError> {
            let mut block = Matrix::zeros(occupied.rows, occupied.rows);
            for (b, output) in row.chunks_mut(np).enumerate() {
                for mu in 0..occupied.rows {
                    for nu in 0..occupied.rows {
                        block[(mu, nu)] = integral(a, mu, b, nu)?;
                    }
                }
                output.copy_from_slice(&co_t.dot(&block).dot(occupied).data);
            }
            Ok(())
        })?;
    let transpose = target.transpose();
    let mut result = Vec::with_capacity(np);
    for pair in 0..np {
        let reversed = (pair % no) * no + pair / no;
        let mut block = Matrix::zeros(n, n);
        for a in 0..n {
            for b in 0..=a {
                block[(a, b)] = rows[a][b * np + pair];
                if a != b {
                    block[(b, a)] = rows[a][b * np + reversed];
                }
            }
        }
        result.push(transpose.dot(&block).dot(target));
    }
    Ok(result)
}

fn pair_tensor(
    left: &Matrix,
    right: &Matrix,
    occupied_left: &Matrix,
    occupied_right: &Matrix,
    integral: impl Fn(usize, usize, usize, usize) -> Result<f64, HfError> + Sync,
) -> Result<Vec<Matrix>, HfError> {
    let np = occupied_left.cols * occupied_right.cols;
    if np == 0 {
        return Ok(Vec::new());
    }
    let count = left.rows as u128 * right.rows as u128 * np as u128;
    if count * 8 > 128 * 1024 * 1024 {
        return Err(HfError::Integral(
            "mixed F12 occupied-leg workspace exceeds 128 MiB".into(),
        ));
    }
    let mut ao = vec![0.0; count as usize];
    let co_t = occupied_left.transpose();
    ao.par_chunks_mut(right.rows * np)
        .enumerate()
        .try_for_each(|(a, row)| -> Result<(), HfError> {
            let mut block = Matrix::zeros(occupied_left.rows, occupied_right.rows);
            for (b, output) in row.chunks_mut(np).enumerate() {
                for mu in 0..occupied_left.rows {
                    for nu in 0..occupied_right.rows {
                        block[(mu, nu)] = integral(a, mu, b, nu)?;
                    }
                }
                let values = co_t.dot(&block).dot(occupied_right);
                output.copy_from_slice(&values.data);
            }
            Ok(())
        })?;
    let left_t = left.transpose();
    let mut result = Vec::with_capacity(np);
    for pair in 0..np {
        let mut block = Matrix::zeros(left.rows, right.rows);
        for (index, value) in block.data.iter_mut().enumerate() {
            *value = ao[index * np + pair];
        }
        result.push(left_t.dot(&block).dot(right));
    }
    Ok(result)
}

pub(crate) fn ri_coefficients(hf: &HFResult, cabs: &CABSBasis) -> Matrix {
    let nm = hf.mo_coefficients.cols;
    let mut ri = Matrix::zeros(cabs.n_obs + cabs.n_raw_aux, nm + cabs.n_cabs);
    for mu in 0..cabs.n_obs {
        for p in 0..nm {
            ri[(mu, p)] = hf.mo_coefficients[(mu, p)];
        }
    }
    let complement = cabs.union_coefficients();
    for mu in 0..ri.rows {
        for p in 0..cabs.n_cabs {
            ri[(mu, nm + p)] = complement[(mu, p)];
        }
    }
    ri
}

/// Ordered occupied pairs of <pq|r12^-1 Q f|ij>, with both bra indices in
/// the full OBS. These include the mixed occupied/virtual blocks needed for
/// CCSD-F12 singles. An exchange-like AO contraction avoids RI four-index ERIs.
pub fn molecular_ccsd_projected_coulomb(
    hf: &HFResult,
    obs: &[ContractedGaussian],
    cabs: &CABSBasis,
    factor: &GaussianGeminal,
) -> Result<Vec<Matrix>, HfError> {
    cabs.validate_reference(hf, obs.len())
        .map_err(HfError::InvalidInput)?;
    let (nao, no) = (obs.len(), hf.n_occ);
    let ri = ri_coefficients(hf, cabs);
    let mut union = obs.to_vec();
    union.extend_from_slice(&cabs.aux_functions);
    let mut co = Matrix::zeros(nao, no);
    for mu in 0..nao {
        for i in 0..no {
            co[(mu, i)] = hf.mo_coefficients[(mu, i)];
        }
    }
    let f_operator = PreparedGeminalOperator::new(factor, GeminalOperator::Factor)?;
    let f = symmetric_pair_tensor(&ri, &co, |a, mu, b, nu| {
        f_operator.contracted(&union[a], &obs[mu], &union[b], &obs[nu])
    })?;
    projected_coulomb_from_factor(hf, obs, &union, &ri, &co, factor, &f)
}

fn projected_coulomb_from_factor(
    hf: &HFResult,
    obs: &[ContractedGaussian],
    union: &[ContractedGaussian],
    ri: &Matrix,
    co: &Matrix,
    factor: &GaussianGeminal,
    f: &[Matrix],
) -> Result<Vec<Matrix>, HfError> {
    let (nao, no, nm) = (obs.len(), hf.n_occ, hf.mo_coefficients.cols);
    let rt = ri.transpose();
    let gf_operator = PreparedGeminalOperator::new(factor, GeminalOperator::FactorOverR)?;
    let mut projected = symmetric_pair_tensor(&hf.mo_coefficients, co, |a, mu, b, nu| {
        gf_operator.contracted(&obs[a], &obs[mu], &obs[b], &obs[nu])
    })?;
    for (pair, result) in projected.iter_mut().enumerate() {
        let mut removed = f[pair].clone();
        for p in 0..ri.cols {
            for q in 0..ri.cols {
                if !((p < nm && q < nm) || (p >= nm && q < no) || (p < no && q >= nm)) {
                    removed[(p, q)] = 0.;
                }
            }
        }
        let density = ri.dot(&removed).dot(&rt);
        let mut action = Matrix::zeros(nao, nao);
        action
            .data
            .par_chunks_mut(nao)
            .enumerate()
            .for_each(|(a, row)| {
                for (b, value) in row.iter_mut().enumerate() {
                    for mu in 0..union.len() {
                        for nu in 0..union.len() {
                            let weight = density[(mu, nu)];
                            if weight == 0. {
                                continue;
                            }
                            let integral = if mu < nao && nu < nao {
                                hf.eri[crate::scf::eri_index(a, mu, b, nu)]
                            } else {
                                crate::integrals::contracted_eri(
                                    &obs[a], &union[mu], &obs[b], &union[nu],
                                )
                            };
                            *value += weight * integral;
                        }
                    }
                }
            });
        *result = result.sub(
            &hf.mo_coefficients
                .transpose()
                .dot(&action)
                .dot(&hf.mo_coefficients),
        );
        if result.data.iter().any(|v| !v.is_finite()) {
            return Err(HfError::Integral(
                "nonfinite CCSD-F12 projected Coulomb matrix".into(),
            ));
        }
    }
    Ok(projected)
}

/// Robustly fitted full-OBS CC coupling, with exact orbital MP2/CCSD and Fock.
/// The fitting basis is distinct from the CABS raw auxiliary basis.
pub fn molecular_ccsd_projected_coulomb_fitted(
    hf: &HFResult,
    obs: &[ContractedGaussian],
    cabs: &CABSBasis,
    factor: &GaussianGeminal,
    fitting: &AuxBasis,
    frozen_core: usize,
    budget: usize,
) -> Result<Vec<Matrix>, HfError> {
    molecular_ccsd_projected_coulomb_fitted_in_space(
        hf,
        obs,
        cabs,
        factor,
        fitting,
        crate::ao::BasisKind::Cartesian,
        frozen_core,
        budget,
    )
}

/// Fitted projected Coulomb with an explicit physical auxiliary space.
pub fn molecular_ccsd_projected_coulomb_fitted_in_space(
    hf: &HFResult,
    obs: &[ContractedGaussian],
    cabs: &CABSBasis,
    factor: &GaussianGeminal,
    fitting: &AuxBasis,
    kind: crate::ao::BasisKind,
    frozen_core: usize,
    budget: usize,
) -> Result<Vec<Matrix>, HfError> {
    projected_coulomb_fitted(
        hf,
        obs,
        cabs,
        factor,
        fitting,
        kind,
        frozen_core,
        budget,
        false,
    )
}

/// Fitted projected Coulomb with the driver's physical fitting and orthogonal
/// OBS mixing controls. Conventional MP2/Schur G is independent of this operator.
pub fn molecular_ccsd_projected_coulomb_fitted_with_options(
    hf: &HFResult,
    obs: &[ContractedGaussian],
    cabs: &CABSBasis,
    factor: &GaussianGeminal,
    options: &super::MP2F12Options,
) -> Result<Vec<Matrix>, HfError> {
    let fitting = options.fitting_basis.as_ref().ok_or_else(|| {
        HfError::InvalidInput("fitted projected Coulomb requires an F12 fitting basis".into())
    })?;
    projected_coulomb_fitted(
        hf,
        obs,
        cabs,
        factor,
        fitting,
        options.fitting_basis_kind,
        options.frozen_core,
        options.workspace_budget_bytes,
        options.exact_obs_coulomb,
    )
}

fn projected_coulomb_fitted(
    hf: &HFResult,
    obs: &[ContractedGaussian],
    cabs: &CABSBasis,
    factor: &GaussianGeminal,
    fitting: &AuxBasis,
    kind: crate::ao::BasisKind,
    frozen_core: usize,
    budget: usize,
    exact_obs_coulomb: bool,
) -> Result<Vec<Matrix>, HfError> {
    cabs.validate_reference(hf, obs.len())
        .map_err(HfError::InvalidInput)?;
    check_workspace(
        hf,
        obs,
        cabs,
        frozen_core,
        budget,
        Some(fitting),
        true,
        exact_obs_coulomb,
    )?;
    let nc = hf.n_occ - frozen_core;
    if nc == 0 {
        return Ok(Vec::new());
    }
    let mut union = obs.to_vec();
    union.extend_from_slice(&cabs.aux_functions);
    let ri = ri_coefficients(hf, cabs);
    let mut occupied = Matrix::zeros(union.len(), nc);
    for mu in 0..obs.len() {
        for i in 0..nc {
            occupied[(mu, i)] = hf.mo_coefficients[(mu, frozen_core + i)];
        }
    }
    let fitter = F12PairFitter::new(
        &union,
        &fitting.functions,
        &ri,
        &occupied,
        F12DensityFitOptions {
            auxiliary_basis_kind: kind,
            workspace_budget_bytes: budget,
            ..Default::default()
        },
    )?;
    let f = fitter
        .kernel(factor, GeminalOperator::Factor)?
        .pairs(0..ri.cols, 0..ri.cols)?;
    let bare = fitter
        .kernel(factor, GeminalOperator::FactorOverR)?
        .pairs(0..hf.mo_coefficients.cols, 0..hf.mo_coefficients.cols)?;
    drop(fitter);
    projected_coulomb_from_fit(
        hf,
        &union,
        &ri,
        fitting,
        kind,
        budget,
        bare,
        &f,
        exact_obs_coulomb,
    )
}

fn projected_coulomb_from_fit(
    hf: &HFResult,
    union: &[ContractedGaussian],
    ri: &Matrix,
    fitting: &AuxBasis,
    kind: crate::ao::BasisKind,
    budget: usize,
    mut projected: Vec<Matrix>,
    factor: &[Matrix],
    exact_obs_coulomb: bool,
) -> Result<Vec<Matrix>, HfError> {
    let (nm, nr, no) = (hf.mo_coefficients.cols, ri.cols, hf.n_occ);
    let mut orbital = Matrix::zeros(union.len(), nm);
    for mu in 0..hf.mo_coefficients.rows {
        for p in 0..nm {
            orbital[(mu, p)] = hf.mo_coefficients[(mu, p)];
        }
    }
    // B[A,p*r] contracts an OBS bra and an RI resolution index before storage.
    // Sum B_A P(f_ij) B_A^T instead of constructing an OBS^2-by-RI^2 tensor.
    let fitter = F12PairFitter::new(
        union,
        &fitting.functions,
        &orbital,
        ri,
        F12DensityFitOptions {
            auxiliary_basis_kind: kind,
            workspace_budget_bytes: budget,
            ..Default::default()
        },
    )?;
    let modes = fitter.coulomb_modes();
    projected
        .par_iter_mut()
        .zip(factor)
        .try_for_each(|(output, f)| -> Result<(), HfError> {
            let mut removed = f.clone();
            for p in 0..nr {
                for q in 0..nr {
                    let obs_block = p < nm && q < nm;
                    if !((obs_block && !exact_obs_coulomb)
                        || (p >= nm && q < no)
                        || (p < no && q >= nm))
                    {
                        removed[(p, q)] = 0.;
                    }
                }
            }
            let mut block = Matrix::zeros(nm, nr);
            for a in 0..modes.rows {
                block
                    .data
                    .copy_from_slice(&modes.data[a * nm * nr..(a + 1) * nm * nr]);
                let action = block.dot(&removed).dot(&block.transpose());
                for (value, delta) in output.data.iter_mut().zip(action.data) {
                    *value -= delta;
                }
            }
            if exact_obs_coulomb {
                // Only after the consistent raw DF CABS transformation may the
                // orthogonal OBS block use exact integrals (Knizia 2010, 4.5.1).
                // Apply its two-electron exchange-like action to f's OBS block;
                // ordered occupied pairs can have nonsymmetric densities.
                let mut obs_factor = Matrix::zeros(nm, nm);
                for p in 0..nm {
                    for q in 0..nm {
                        obs_factor[(p, q)] = f[(p, q)];
                    }
                }
                let coefficients = &hf.mo_coefficients;
                let density = coefficients.dot(&obs_factor).dot(&coefficients.transpose());
                let nao = coefficients.rows;
                let mut action = Matrix::zeros(nao, nao);
                for mu in 0..nao {
                    for nu in 0..nao {
                        for rho in 0..nao {
                            for sigma in 0..nao {
                                action[(mu, nu)] += hf.eri
                                    [crate::scf::eri_index(mu, rho, nu, sigma)]
                                    * density[(rho, sigma)];
                            }
                        }
                    }
                }
                let exact = coefficients.transpose().dot(&action).dot(coefficients);
                for (value, delta) in output.data.iter_mut().zip(exact.data) {
                    *value -= delta;
                }
            }
            if output.data.iter().any(|x| !x.is_finite()) {
                return Err(HfError::Integral(
                    "nonfinite fitted CCSD-F12 projected Coulomb matrix".into(),
                ));
            }
            Ok(())
        })?;
    Ok(projected)
}

pub fn molecular_intermediates(
    hf: &HFResult,
    atoms: &[Atom],
    obs: &[ContractedGaussian],
    cabs: &CABSBasis,
    factor: &GaussianGeminal,
) -> Result<Vec<F12PairIntermediates>, HfError> {
    let extended = extended_fock(hf, atoms, obs, &cabs.aux_functions).map_err(HfError::Integral)?;
    molecular_intermediates_with_fock(
        hf,
        atoms,
        obs,
        cabs,
        factor,
        &extended,
        MolecularF12Options {
            frozen_core: 0,
            approximation: F12Approximation::ApproximationC,
            hybrid: F12Hybrid::Full,
            workspace_budget_bytes: DEFAULT_F12_WORKSPACE_BYTES,
            fitting_basis: None,
            fitting_basis_kind: crate::ao::BasisKind::Cartesian,
            fitted_fock: false,
            fit_orbital_mp2: false,
            exact_obs_coulomb: false,
        },
    )
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MolecularF12Options<'a> {
    pub frozen_core: usize,
    pub approximation: F12Approximation,
    pub hybrid: F12Hybrid,
    /// Limit for the live mixed-tensor workspace estimate.
    pub workspace_budget_bytes: usize,
    pub fitting_basis: Option<&'a AuxBasis>,
    pub fitting_basis_kind: crate::ao::BasisKind,
    pub fitted_fock: bool,
    pub fit_orbital_mp2: bool,
    pub exact_obs_coulomb: bool,
}

/// Default limit for the live F12 mixed-tensor workspace (256 MiB).
pub const DEFAULT_F12_WORKSPACE_BYTES: usize = 256 * 1024 * 1024;

/// Check the mixed-tensor bound before constructing the extended Fock matrix.
pub(crate) fn check_workspace(
    hf: &HFResult,
    obs: &[ContractedGaussian],
    cabs: &CABSBasis,
    frozen_core: usize,
    budget: usize,
    fitting: Option<&AuxBasis>,
    ccsd_coupling: bool,
    exact_obs_coulomb: bool,
) -> Result<(), HfError> {
    if budget == 0 {
        return Err(HfError::InvalidInput(
            "F12 workspace budget must be positive".into(),
        ));
    }
    if exact_obs_coulomb && fitting.is_none() {
        return Err(HfError::InvalidInput(
            "exact OBS Coulomb mixing requires an explicit F12 fitting basis".into(),
        ));
    }
    let correlated = hf
        .n_occ
        .checked_sub(frozen_core)
        .ok_or_else(|| HfError::InvalidInput("F12 frozen core exceeds occupied space".into()))?;
    let n = obs.len() as u128 + cabs.aux_functions.len() as u128;
    let nr = hf.mo_coefficients.cols as u128 + cabs.n_cabs as u128;
    let no = hf.n_occ as u128;
    let mut upper = if let Some(fitting) = fitting {
        if fitting.functions.is_empty() {
            return Err(HfError::InvalidInput("empty F12 fitting basis".into()));
        }
        let na = fitting.functions.len() as u128;
        let nm = hf.mo_coefficients.cols as u128;
        let lmax = obs
            .iter()
            .chain(&cabs.aux_functions)
            .map(|g| g.l.iter().sum::<usize>())
            .max()
            .unwrap_or(0);
        let amax = fitting
            .functions
            .iter()
            .map(|g| g.l.iter().sum::<usize>())
            .max()
            .unwrap_or(0);
        if lmax > 7 || amax > 7 {
            return Err(HfError::InvalidInput(
                "F12 fitting angular momentum exceeds 7".into(),
            ));
        }
        let workers = fitting.functions.len().min(rayon::current_num_threads()) as u128;
        let legs = (nr * correlated as u128).max(if ccsd_coupling { nm * nr } else { 0 });
        8 * ((correlated as u128).pow(2) * (4 * nr.pow(2) + 4 * nr * no + 4 * no.pow(2))
            + 10 * na.pow(2)
            + 6 * na * legs
            + 32 * n.pow(2)
            + workers
                * (16 * (2 * lmax as u128 + amax as u128 + 1).pow(4)
                    + 2 * n.pow(2)
                    + 2 * nr.pow(2)))
    } else {
        8 * (correlated as u128).pow(2)
            * (4 * n.pow(2) + 4 * nr.pow(2) + 4 * nr * no + 4 * no.pow(2))
    };
    if exact_obs_coulomb {
        let nao = obs.len() as u128;
        let nm = hf.mo_coefficients.cols as u128;
        let nc = correlated as u128;
        let workers = (correlated * correlated).min(rayon::current_num_threads()) as u128;
        upper += 8
            * (nc.pow(2) * (4 * nao.pow(2) + nm.pow(2))
                + nr.pow(2)
                + workers * (3 * nao.pow(2) + 3 * nm.pow(2)));
    }
    if upper > budget as u128 {
        return Err(HfError::Integral(format!(
            "F12 live mixed-tensor workspace estimate {upper} bytes exceeds the budget {budget} (MP2F12Options::workspace_budget_bytes; registry option f12_workspace_bytes)"
        )));
    }
    Ok(())
}

pub(crate) fn molecular_intermediates_with_fock(
    hf: &HFResult,
    atoms: &[Atom],
    obs: &[ContractedGaussian],
    cabs: &CABSBasis,
    factor: &GaussianGeminal,
    extended: &ExtendedFock,
    options: MolecularF12Options<'_>,
) -> Result<Vec<F12PairIntermediates>, HfError> {
    Ok(molecular_intermediates_and_coupling(
        hf, atoms, obs, cabs, factor, extended, options, false,
    )?
    .0)
}

/// Build the pair functional and optional full-OBS CCSD coupling from the same
/// factor family, RI coefficients and union Fock operator.
pub(crate) fn molecular_intermediates_and_coupling(
    hf: &HFResult,
    atoms: &[Atom],
    obs: &[ContractedGaussian],
    cabs: &CABSBasis,
    factor: &GaussianGeminal,
    extended: &ExtendedFock,
    options: MolecularF12Options<'_>,
    ccsd_coupling: bool,
) -> Result<(Vec<F12PairIntermediates>, Option<Vec<Matrix>>), HfError> {
    let frozen_core = options.frozen_core;
    let nao = obs.len();
    let no = hf.n_occ;
    let nm = hf.mo_coefficients.cols;
    cabs.validate_reference(hf, nao)
        .map_err(HfError::InvalidInput)?;
    if !hf.converged
        || nao == 0
        || hf.mo_coefficients.rows != nao
        || no > nm
        || frozen_core > no
        || hf.orbital_energies.len() != nm
        || cabs.n_obs != nao
        || cabs.n_raw_aux != cabs.aux_functions.len()
        || cabs.c_obs_to_cabs.rows != nao
        || cabs.c_obs_to_cabs.cols != cabs.n_cabs
        || cabs.c_aux_to_cabs.rows != cabs.n_raw_aux
        || cabs.c_aux_to_cabs.cols != cabs.n_cabs
        || hf.eri.len() <= crate::scf::eri_index(nao - 1, nao - 1, nao - 1, nao - 1)
    {
        return Err(HfError::InvalidInput(
            "F12 requires a converged compatible RHF/OBS/CABS reference".into(),
        ));
    }
    GaussianGeminal::new(factor.terms.clone())?;
    let correlated = no - frozen_core;
    check_workspace(
        hf,
        obs,
        cabs,
        frozen_core,
        options.workspace_budget_bytes,
        options.fitting_basis,
        ccsd_coupling,
        options.exact_obs_coulomb,
    )?;
    if correlated == 0 {
        return Ok((Vec::new(), None));
    }
    let mut union = obs.to_vec();
    union.extend_from_slice(&cabs.aux_functions);
    let nr = nm + cabs.n_cabs;
    if union.iter().any(|cg| {
        cg.primitives.is_empty()
            || cg.l.iter().sum::<usize>() > 7
            || cg.center.iter().any(|x| !x.is_finite())
            || cg
                .primitives
                .iter()
                .any(|p| !p.exponent.is_finite() || p.exponent <= 0.0 || !p.coefficient.is_finite())
    }) || hf.mo_coefficients.data.iter().any(|x| !x.is_finite())
    {
        return Err(HfError::InvalidInput(
            "invalid F12 basis or MO coefficients".into(),
        ));
    }
    let mut co = Matrix::zeros(nao, no);
    let mut correlated_occ = Matrix::zeros(nao, correlated);
    let ri = ri_coefficients(hf, cabs);
    for mu in 0..nao {
        for i in 0..no {
            co[(mu, i)] = hf.mo_coefficients[(mu, i)];
        }
        for i in 0..correlated {
            correlated_occ[(mu, i)] = hf.mo_coefficients[(mu, frozen_core + i)];
        }
    }
    let mut nonlocal = extended.exchange.clone();
    let nonlocal_ecp = if atoms.iter().any(|atom| {
        atom.ecp
            .as_ref()
            .is_some_and(|e| e.semilocal.iter().any(|v| !v.is_empty()))
    }) {
        let mut nonlocal_atoms = atoms.to_vec();
        for atom in &mut nonlocal_atoms {
            atom.ecp = atom.ecp.as_ref().and_then(|e| {
                if e.semilocal.iter().all(|v| v.is_empty()) {
                    None
                } else {
                    let mut e = (**e).clone();
                    e.local.clear();
                    Some(std::sync::Arc::new(e))
                }
            });
        }
        let potential = crate::ecp::ecp_one_electron_matrix(&nonlocal_atoms, &union);
        nonlocal = nonlocal.sub(&potential);
        Some(potential)
    } else {
        None
    };
    let rt = ri.transpose();
    let mut space = F12FockSpace {
        n_occ: no,
        n_obs: nm,
        fock: super::cabs::symmetric_congruence(&extended.fock, &ri)?,
        nonlocal: super::cabs::symmetric_congruence(&nonlocal, &ri)?,
        orbital_energies: hf.orbital_energies.clone(),
    };
    if options.fitted_fock {
        // Only orthogonal MO/CABS blocks may be mixed (Knizia 2010, 4.5.1).
        // Delta is a finite-rank nonlocal operator. Subtract it from n so
        // F+n remains the original local DF operator T+J+V_local.
        let delta = super::cabs::restore_canonical_obs(hf, &mut space.fock)?;
        space.nonlocal = space.nonlocal.sub(&delta);
    }
    space.validate()?;
    let factor_operator = PreparedGeminalOperator::new(factor, GeminalOperator::Factor)?;
    let fitted = if let Some(auxiliary) = options.fitting_basis {
        let mut padded_occ = Matrix::zeros(union.len(), correlated);
        for mu in 0..nao {
            for i in 0..correlated {
                padded_occ[(mu, i)] = correlated_occ[(mu, i)];
            }
        }
        Some(F12PairFitter::new(
            &union,
            &auxiliary.functions,
            &ri,
            &padded_occ,
            F12DensityFitOptions {
                auxiliary_basis_kind: options.fitting_basis_kind,
                workspace_budget_bytes: options.workspace_budget_bytes,
                ..Default::default()
            },
        )?)
    } else {
        None
    };
    let f = if let Some(fitted) = &fitted {
        fitted
            .kernel(factor, GeminalOperator::Factor)?
            .pairs(0..nr, 0..nr)?
    } else {
        symmetric_pair_tensor(&ri, &correlated_occ, |a, mu, b, nu| {
            factor_operator.contracted(&union[a], &obs[mu], &union[b], &obs[nu])
        })?
    };
    let g = if let Some(fitted) = &fitted {
        fitted.coulomb_kernel().pairs(0..nr, 0..nr)?
    } else {
        symmetric_pair_tensor(&ri, &correlated_occ, |a, mu, b, nu| {
            Ok(if a < nao && b < nao {
                hf.eri[crate::scf::eri_index(a, mu, b, nu)]
            } else {
                crate::integrals::contracted_eri(&union[a], &obs[mu], &union[b], &obs[nu])
            })
        })?
    };
    let square_operator = PreparedGeminalOperator::new(factor, GeminalOperator::FactorSquared)?;
    let square = if let Some(fitted) = &fitted {
        fitted
            .kernel(factor, GeminalOperator::FactorSquared)?
            .pairs(0..nr, 0..no)?
    } else {
        pair_tensor(
            &ri,
            &co,
            &correlated_occ,
            &correlated_occ,
            |a, mu, b, nu| square_operator.contracted(&union[a], &obs[mu], &obs[b], &obs[nu]),
        )?
    };
    let coulomb_operator = PreparedGeminalOperator::new(factor, GeminalOperator::FactorOverR)?;
    let (gf, fitted_bare_cc) = if let Some(fitted) = &fitted {
        let kernel = fitted.kernel(factor, GeminalOperator::FactorOverR)?;
        (
            kernel.pairs(frozen_core..no, frozen_core..no)?,
            if ccsd_coupling {
                Some(kernel.pairs(0..nm, 0..nm)?)
            } else {
                None
            },
        )
    } else {
        (
            symmetric_pair_tensor(&correlated_occ, &correlated_occ, |a, mu, b, nu| {
                coulomb_operator.contracted(&obs[a], &obs[mu], &obs[b], &obs[nu])
            })?,
            None,
        )
    };
    let double_commutator =
        PreparedGeminalOperator::new(factor, GeminalOperator::PositiveDoubleCommutator)?;
    let comm = if let Some(fitted) = &fitted {
        fitted
            .kernel(factor, GeminalOperator::PositiveDoubleCommutator)?
            .pairs(frozen_core..no, frozen_core..no)?
    } else {
        symmetric_pair_tensor(&correlated_occ, &correlated_occ, |a, mu, b, nu| {
            double_commutator.contracted(&obs[a], &obs[mu], &obs[b], &obs[nu])
        })?
    };
    // Raw fitted CABS quantities are transformed consistently before replacing
    // the orthogonal OBS Coulomb block. This choice is independent of the
    // conventional MP2/Schur G model; orbital CCSD/(T) remains exact.
    let exact_obs_g = if options.exact_obs_coulomb {
        Some(symmetric_pair_tensor(
            &hf.mo_coefficients,
            &correlated_occ,
            |a, mu, b, nu| Ok(hf.eri[crate::scf::eri_index(a, mu, b, nu)]),
        )?)
    } else {
        None
    };
    let exact_orbital_g = if fitted.is_some() && !options.fit_orbital_mp2 && exact_obs_g.is_none() {
        let mut virtuals = Matrix::zeros(nao, nm - no);
        for mu in 0..nao {
            for a in no..nm {
                virtuals[(mu, a - no)] = hf.mo_coefficients[(mu, a)];
            }
        }
        Some(symmetric_pair_tensor(
            &virtuals,
            &correlated_occ,
            |a, mu, b, nu| Ok(hf.eri[crate::scf::eri_index(a, mu, b, nu)]),
        )?)
    } else {
        None
    };
    drop(fitted);
    let single = if options.approximation == F12Approximation::ApproximationA {
        let single_commutator =
            PreparedGeminalOperator::new(factor, GeminalOperator::KineticCommutator)?;
        Some(symmetric_pair_tensor(
            &ri,
            &correlated_occ,
            |a, mu, b, nu| single_commutator.contracted(&union[a], &obs[mu], &union[b], &obs[nu]),
        )?)
    } else {
        None
    };
    let nonlocal_action = if options.approximation == F12Approximation::ApproximationA {
        if let Some(potential) = &nonlocal_ecp {
            let potential_ri = rt.dot(potential).dot(&ri);
            let mut active_columns = Matrix::zeros(nr, correlated);
            for p in 0..nr {
                for i in 0..correlated {
                    active_columns[(p, i)] = potential_ri[(p, frozen_core + i)];
                }
            }
            let acted_occ = ri.dot(&active_columns);
            // Resolve V|i> first, so the additional family still has two
            // occupied legs. Electron exchange supplies the second V action.
            let first = pair_tensor(&ri, &ri, &acted_occ, &correlated_occ, |a, mu, b, nu| {
                factor_operator.contracted(&union[a], &union[mu], &union[b], &obs[nu])
            })?;
            Some((potential_ri, first))
        } else {
            None
        }
    } else {
        None
    };
    let mut pairs = Vec::with_capacity(correlated * (correlated + 1) / 2);
    for i in frozen_core..no {
        for j in i..no {
            let a = i - frozen_core;
            let b = j - frozen_core;
            let ij = a * correlated + b;
            let ji = b * correlated + a;
            let mixed_g = exact_obs_g.as_ref().map(|exact| {
                let mut mixed = g[ij].clone();
                for p in 0..nm {
                    for q in 0..nm {
                        mixed[(p, q)] = exact[ij][(p, q)];
                    }
                }
                mixed
            });
            let values = F12PairIntegralView {
                pair: [i, j],
                factor: &f[ij],
                coulomb: mixed_g.as_ref().unwrap_or(&g[ij]),
                squared: &square[ij],
                squared_exchange: &square[ji],
                coulomb_factor: [gf[ij][(a, b)], gf[ji][(a, b)]],
                positive_commutator: [comm[ij][(a, b)], comm[ji][(a, b)]],
            };
            let mut result = if let Some(single) = &single {
                if let Some((potential, first)) = &nonlocal_action {
                    let right_action = first[ij].add(&first[ji].transpose());
                    space.pair_intermediates_3a_with_nonlocal(
                        values,
                        &single[ij],
                        Some(F12NonlocalPairView {
                            potential,
                            factor_right_action: &right_action,
                        }),
                    )?
                } else {
                    space.pair_intermediates_3a(values, &single[ij])?
                }
            } else {
                space.pair_intermediates_with_hybrid(values, options.hybrid)?
            };
            if exact_obs_g.is_some() || exact_orbital_g.is_some() {
                for p in 0..result.g.rows {
                    for q in 0..result.g.cols {
                        let desired = if options.fit_orbital_mp2 {
                            g[ij][(no + p, no + q)]
                        } else if let Some(exact) = &exact_obs_g {
                            exact[ij][(no + p, no + q)]
                        } else {
                            // The branch is entered only when an exact family exists.
                            exact_orbital_g.as_ref().ok_or_else(|| {
                                HfError::Integral("missing F12 orbital Coulomb family".into())
                            })?[ij][(p, q)]
                        };
                        let gap = hf.orbital_energies[no + p] + hf.orbital_energies[no + q]
                            - hf.orbital_energies[i]
                            - hf.orbital_energies[j];
                        let delta = (desired - result.g[(p, q)]) / gap;
                        result.v_bar[0] -= delta * result.c[(p, q)];
                        result.v_bar[1] -= delta * result.c[(q, p)];
                        result.g[(p, q)] = desired;
                    }
                }
            }
            pairs.push(result);
        }
    }
    // These integral families are no longer needed. Release them before the
    // additional CCSD Coulomb action and, later, the CC amplitude workspace.
    drop(g);
    drop(square);
    drop(gf);
    drop(comm);
    drop(single);
    drop(nonlocal_action);
    drop(exact_orbital_g);
    drop(exact_obs_g);
    let projected = if ccsd_coupling {
        Some(
            if let (Some(auxiliary), Some(bare)) = (options.fitting_basis, fitted_bare_cc) {
                projected_coulomb_from_fit(
                    hf,
                    &union,
                    &ri,
                    auxiliary,
                    options.fitting_basis_kind,
                    options.workspace_budget_bytes,
                    bare,
                    &f,
                    options.exact_obs_coulomb,
                )?
            } else {
                projected_coulomb_from_factor(hf, obs, &union, &ri, &correlated_occ, factor, &f)?
            },
        )
    } else {
        None
    };
    Ok((pairs, projected))
}
