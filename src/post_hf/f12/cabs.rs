// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! CABS+ as the metric complement of the retained OBS, and singles in the full
//! OBS-virtual/CABS Fock space. Extended Fock blocks include Coulomb, exchange
//! and ECP terms; no virtual-energy floor is applied.
use crate::{
    basis::{Atom, ContractedGaussian},
    matrix::Matrix,
    scf::{HFResult, compute_one_electron_integrals},
};
use rayon::prelude::*;

fn valid_matrix(matrix: &Matrix, rows: usize, cols: usize) -> bool {
    matrix.rows == rows
        && matrix.cols == cols
        && rows.checked_mul(cols) == Some(matrix.data.len())
        && matrix.data.iter().all(|value| value.is_finite())
}

/// Transform a real Hermitian AO operator without amplifying independent GEMM
/// roundoff in the two triangles of an ill-conditioned CABS transformation.
/// Product residuals from FMA and compensated sums retain cancellation digits.
/// Only one transformed triangle is evaluated, as in a symmetric eigensolver.
pub(crate) fn symmetric_congruence(
    operator: &Matrix,
    coefficients: &Matrix,
) -> Result<Matrix, crate::error::HfError> {
    use crate::error::HfError;
    let (n, k) = (coefficients.rows, coefficients.cols);
    if !valid_matrix(operator, n, n) || !valid_matrix(coefficients, n, k) {
        return Err(HfError::InvalidInput(
            "invalid symmetric F12 operator transformation".into(),
        ));
    }
    for i in 0..n {
        for j in 0..i {
            if (operator[(i, j)] - operator[(j, i)]).abs() > 1e-12 {
                return Err(HfError::InvalidInput(
                    "F12 transformation requires a Hermitian real AO operator".into(),
                ));
            }
        }
    }
    if k == 0 {
        return Ok(Matrix::zeros(0, 0));
    }
    fn dot(terms: impl Iterator<Item = (f64, f64)>) -> f64 {
        let mut sum = 0.0_f64;
        let mut correction = 0.0;
        for (a, b) in terms {
            let product = a * b;
            let next = sum + product;
            correction += if sum.abs() >= product.abs() {
                (sum - next) + product
            } else {
                (product - next) + sum
            };
            correction += a.mul_add(b, -product);
            sum = next;
        }
        sum + correction
    }
    let mut action = Matrix::zeros(n, k);
    action
        .data
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(i, row)| {
            for (q, value) in row.iter_mut().enumerate() {
                *value =
                    dot((0..n).map(|j| (operator[(i.max(j), i.min(j))], coefficients[(j, q)])));
            }
        });
    let mut transformed = Matrix::zeros(k, k);
    transformed
        .data
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(p, row)| {
            for (q, value) in row.iter_mut().enumerate().take(p + 1) {
                *value = dot((0..n).map(|i| (coefficients[(i, p)], action[(i, q)])));
            }
        });
    for p in 0..k {
        for q in 0..p {
            transformed[(q, p)] = transformed[(p, q)];
        }
    }
    if transformed.data.iter().any(|x| !x.is_finite()) {
        return Err(HfError::Integral(
            "nonfinite symmetric F12 transformation".into(),
        ));
    }
    Ok(transformed)
}

#[derive(Clone, Debug)]
pub struct CABSBasis {
    pub n_obs: usize,
    pub n_raw_aux: usize,
    pub n_cabs: usize,
    /// Applied projected-metric cutoff, retained when rebuilding displaced CABS.
    pub metric_threshold: f64,
    pub c_aux_to_cabs: Matrix,
    pub c_obs_to_cabs: Matrix,
    pub aux_functions: Vec<ContractedGaussian>,
}
fn overlap(a: &[ContractedGaussian], b: &[ContractedGaussian]) -> Matrix {
    let mut out = Matrix::zeros(a.len(), b.len());
    out.data
        .par_chunks_mut(b.len().max(1))
        .enumerate()
        .for_each(|(i, row)| {
            for (j, v) in row.iter_mut().enumerate() {
                *v = crate::integrals::contracted_overlap(&a[i], &b[j]);
            }
        });
    out
}
impl CABSBasis {
    pub(crate) fn validate_reference(&self, hf: &HFResult, nao: usize) -> Result<(), String> {
        if !hf.converged
            || nao == 0
            || !hf.total_energy.is_finite()
            || hf.n_occ > hf.mo_coefficients.cols
            || !valid_matrix(&hf.mo_coefficients, nao, hf.mo_coefficients.cols)
            || !valid_matrix(&hf.density_matrix, nao, nao)
            || hf.eri.len() <= crate::scf::eri_index(nao - 1, nao - 1, nao - 1, nao - 1)
            || hf.orbital_energies.len() != hf.mo_coefficients.cols
            || hf.orbital_energies.iter().any(|value| !value.is_finite())
            || self.n_obs != nao
            || self.n_raw_aux != self.aux_functions.len()
            || !self.metric_threshold.is_finite()
            || self.metric_threshold < 1e-10
            || !valid_matrix(&self.c_obs_to_cabs, nao, self.n_cabs)
            || !valid_matrix(&self.c_aux_to_cabs, self.n_raw_aux, self.n_cabs)
        {
            return Err("invalid CABS/reference dimensions or nonfinite coefficients".into());
        }
        Ok(())
    }

    fn build(
        obs: &[ContractedGaussian],
        aux: &[ContractedGaussian],
        physical_obs: &Matrix,
        physical_aux: &Matrix,
        threshold: f64,
    ) -> Result<Self, String> {
        if obs.is_empty()
            || aux.is_empty()
            || physical_obs.rows != obs.len()
            || physical_aux.rows != aux.len()
            || !valid_matrix(physical_obs, obs.len(), physical_obs.cols)
            || !valid_matrix(physical_aux, aux.len(), physical_aux.cols)
            || !threshold.is_finite()
            || threshold < 0.
        {
            return Err("invalid CABS bases, subspaces or threshold".into());
        }
        let s = overlap(obs, obs);
        let gram = physical_obs.transpose().dot(&s).dot(physical_obs);
        if (0..gram.rows)
            .any(|i| (0..gram.cols).any(|j| (gram[(i, j)] - f64::from(i == j)).abs() > 1e-8))
        {
            return Err("CABS OBS coefficients are not metric orthonormal".into());
        }
        let cross = overlap(obs, aux);
        let auxiliary = overlap(aux, aux);
        let projection = physical_obs.dot(&physical_obs.transpose()).dot(&cross);
        let projected = physical_aux
            .transpose()
            .dot(&auxiliary.sub(&cross.transpose().dot(&projection)))
            .dot(physical_aux);
        let (values, u) = projected.eigh();
        if values.iter().any(|v| !v.is_finite() || *v < -1e-8) {
            return Err("CABS projected metric has a negative/nonfinite eigenvalue".into());
        }
        let retained = (0..values.len())
            .filter(|&i| values[i] > threshold.max(1e-10))
            .collect::<Vec<_>>();
        let mut scaled = Matrix::zeros(u.rows, retained.len());
        for (a, &column) in retained.iter().enumerate() {
            for b in 0..u.rows {
                scaled[(b, a)] = u[(b, column)] / values[column].sqrt();
            }
        }
        let c_aux = physical_aux.dot(&scaled);
        let c_obs = projection.dot(&c_aux).scale(-1.);
        Ok(Self {
            n_obs: obs.len(),
            n_raw_aux: aux.len(),
            n_cabs: retained.len(),
            metric_threshold: threshold.max(1e-10),
            c_aux_to_cabs: c_aux,
            c_obs_to_cabs: c_obs,
            aux_functions: aux.to_vec(),
        })
    }
    /// Metric complement of a Cartesian OBS, with canonical overlap-rank removal.
    pub fn from_auxiliary_basis(
        obs: &[ContractedGaussian],
        aux: &[ContractedGaussian],
        atoms: &[Atom],
        threshold: f64,
    ) -> Result<Self, String> {
        if atoms.is_empty() {
            return Err("CABS requires molecular geometry".into());
        }
        let physical = overlap(obs, obs)
            .canonical_orthogonalizer(1e-8)
            .map_err(|e| e.to_string())?;
        Self::build(obs, aux, &physical, &Matrix::identity(aux.len()), threshold)
    }
    /// Use the retained physical OBS, including spherical MOs and dropped overlap
    /// modes. Auxiliary spherical functions follow the reference representation.
    pub fn from_reference(
        hf: &HFResult,
        obs: &[ContractedGaussian],
        aux: &[ContractedGaussian],
        threshold: f64,
    ) -> Result<Self, String> {
        if !hf.converged {
            return Err("CABS requires a converged HF reference".into());
        }
        let physical_aux = if hf
            .ao_space
            .as_ref()
            .is_some_and(|space| space.kind == crate::ao::BasisKind::Spherical)
        {
            crate::ao::spherical_transform(aux).map_err(|e| e.to_string())?
        } else {
            Matrix::identity(aux.len())
        };
        Self::build(obs, aux, &hf.mo_coefficients, &physical_aux, threshold)
    }
    pub fn from_builtin_reference(
        hf: &HFResult,
        obs: &[ContractedGaussian],
        atoms: &[Atom],
        name: &str,
        threshold: f64,
    ) -> Result<Self, String> {
        let auxiliary =
            crate::context::AuxBasis::from_builtin(atoms, name).map_err(|e| e.to_string())?;
        Self::from_reference(hf, obs, &auxiliary.functions, threshold)
    }
    /// Compatibility default: an actual embedded OptRI basis replaces the old
    /// heuristic unnormalized polarization/diffuse functions.
    pub fn auto_generate(
        obs: &[ContractedGaussian],
        atoms: &[Atom],
        threshold: f64,
    ) -> Result<Self, String> {
        let auxiliary = crate::context::AuxBasis::from_builtin(atoms, "cc-pvdz-f12-optri")
            .map_err(|e| e.to_string())?;
        Self::from_auxiliary_basis(obs, &auxiliary.functions, atoms, threshold)
    }
    pub fn union_coefficients(&self) -> Matrix {
        let mut c = Matrix::zeros(self.n_obs + self.n_raw_aux, self.n_cabs);
        for i in 0..self.n_obs {
            for a in 0..self.n_cabs {
                c[(i, a)] = self.c_obs_to_cabs[(i, a)];
            }
        }
        for i in 0..self.n_raw_aux {
            for a in 0..self.n_cabs {
                c[(self.n_obs + i, a)] = self.c_aux_to_cabs[(i, a)];
            }
        }
        c
    }
}
#[derive(Clone, Debug)]
pub struct ExtendedFock {
    pub fock: Matrix,
    /// Half of exchange formed from the spin-summed RHF density.
    pub exchange: Matrix,
}

/// Restore the independently checked exact OBS Fock only after the complete
/// raw operator has been transformed to orthogonal MO/CABS coordinates.
/// Return the finite-rank change, needed in the nonlocal operator as -Delta.
pub(crate) fn restore_canonical_obs(
    hf: &HFResult,
    fock: &mut Matrix,
) -> Result<Matrix, crate::error::HfError> {
    use crate::error::HfError;
    let nm = hf.mo_coefficients.cols;
    let nao = hf.mo_coefficients.rows;
    if !valid_matrix(&hf.fock_matrix, nao, nao)
        || !valid_matrix(&hf.mo_coefficients, nao, nm)
        || !valid_matrix(fock, fock.rows, fock.rows)
        || fock.rows < nm
        || hf.orbital_energies.len() != nm
    {
        return Err(HfError::InvalidInput(
            "invalid canonical F12 Fock dimensions".into(),
        ));
    }
    let exact = symmetric_congruence(&hf.fock_matrix, &hf.mo_coefficients)?;
    let mut delta = Matrix::zeros(fock.rows, fock.cols);
    for p in 0..nm {
        for q in 0..nm {
            let expected = if p == q { hf.orbital_energies[p] } else { 0.0 };
            if !expected.is_finite()
                || !exact[(p, q)].is_finite()
                || (exact[(p, q)] - expected).abs() > 1e-7
            {
                return Err(HfError::InvalidInput(
                    "F12 requires an independently canonical exact reference Fock".into(),
                ));
            }
            delta[(p, q)] = exact[(p, q)] - fock[(p, q)];
            fock[(p, q)] = exact[(p, q)];
        }
    }
    Ok(delta)
}

/// Mixed exact-OBS/DF-CABS singles. Raw DF quantities are transformed together
/// before restoring the exact OBS block; the raw supplied operator is unchanged.
pub(crate) fn compute_cabs_singles_with_mixed_fock(
    hf: &HFResult,
    cabs: &CABSBasis,
    fock: &ExtendedFock,
    frozen_core: usize,
) -> Result<CabsSinglesResult, String> {
    cabs.validate_reference(hf, cabs.n_obs)?;
    if frozen_core > hf.n_occ {
        return Err("CABS singles frozen core exceeds occupied space".into());
    }
    let ri = super::molecular::ri_coefficients(hf, cabs);
    if !valid_matrix(&fock.fock, ri.rows, ri.rows) {
        return Err("invalid fitted CABS Fock matrix".into());
    }
    let mut mixed = symmetric_congruence(&fock.fock, &ri).map_err(|e| e.to_string())?;
    restore_canonical_obs(hf, &mut mixed).map_err(|e| e.to_string())?;
    let no = hf.n_occ;
    let nv = ri.cols - no;
    let mut vv = Matrix::zeros(nv, nv);
    let mut ov = Matrix::zeros(no, nv);
    let mut virtuals = Matrix::zeros(ri.rows, nv);
    for a in 0..nv {
        for b in 0..nv {
            vv[(a, b)] = mixed[(no + a, no + b)];
        }
        for i in 0..no {
            ov[(i, a)] = mixed[(i, no + a)];
        }
        for mu in 0..ri.rows {
            virtuals[(mu, a)] = ri[(mu, no + a)];
        }
    }
    let (energies, rotation) = vv.eigh();
    singles_from_blocks(
        hf,
        ov.dot(&rotation),
        energies,
        virtuals.dot(&rotation),
        frozen_core,
    )
}

/// Density-fitted Fock and half-exchange in OBS+raw CABS at a fixed RHF
/// density. This is the unmodified fitted operator: its OBS block need not
/// equal the exact reference Fock. It must not be passed to a canonical F12
/// doubles calculation unless that reference/operator consistency is satisfied.
/// CABS singles can use this operator through compute_cabs_singles_with_fock.
pub fn extended_fock_fitted(
    hf: &HFResult,
    atoms: &[Atom],
    obs: &[ContractedGaussian],
    aux: &[ContractedGaussian],
    fitting: &[ContractedGaussian],
    options: super::density_fit::F12DensityFitOptions,
) -> Result<ExtendedFock, crate::error::HfError> {
    use crate::error::HfError;
    let no = obs.len();
    let n = no + aux.len();
    if no == 0
        || !hf.converged
        || hf.n_occ == 0
        || hf.n_occ > hf.mo_coefficients.cols
        || !valid_matrix(&hf.mo_coefficients, no, hf.mo_coefficients.cols)
    {
        return Err(HfError::InvalidInput(
            "invalid fitted extended Fock reference/basis dimensions".into(),
        ));
    }
    // One-electron construction has four simultaneous quadratic matrices.
    let reserved = 8 * 6 * (n as u128).pow(2);
    let remaining = (options.workspace_budget_bytes as u128)
        .checked_sub(reserved)
        .filter(|&bytes| bytes > 0)
        .ok_or_else(|| HfError::Integral("fitted extended Fock workspace exceeds budget".into()))?;
    let mut union = obs.to_vec();
    union.extend_from_slice(aux);
    let mut occupied = Matrix::zeros(n, hf.n_occ);
    for mu in 0..no {
        for i in 0..hf.n_occ {
            occupied[(mu, i)] = hf.mo_coefficients[(mu, i)];
        }
    }
    let (coulomb, spin_exchange) = super::density_fit::fitted_coulomb_exchange_rhf(
        &union,
        fitting,
        &occupied,
        super::density_fit::F12DensityFitOptions {
            workspace_budget_bytes: remaining as usize,
            ..options
        },
    )?;
    let (_, _, _, core) = compute_one_electron_integrals(atoms, &union);
    let exchange = spin_exchange.scale(0.5);
    let fock = core.add(&coulomb).sub(&exchange);
    if fock.data.iter().any(|x| !x.is_finite()) {
        return Err(HfError::Integral("nonfinite fitted extended Fock".into()));
    }
    Ok(ExtendedFock { fock, exchange })
}
/// Direct mixed-basis Fock construction. Only a quadratic AO matrix is stored;
/// the union-basis fourth-power ERI tensor is never allocated.
pub fn extended_fock(
    hf: &HFResult,
    atoms: &[Atom],
    obs: &[ContractedGaussian],
    aux: &[ContractedGaussian],
) -> Result<ExtendedFock, String> {
    let no = obs.len();
    let mut union = obs.to_vec();
    union.extend_from_slice(aux);
    let n = union.len();
    if no == 0
        || !hf.converged
        || hf.n_occ > hf.mo_coefficients.cols
        || !valid_matrix(&hf.mo_coefficients, no, hf.mo_coefficients.cols)
        || !valid_matrix(&hf.density_matrix, no, no)
        || hf.eri.len() <= crate::scf::eri_index(no - 1, no - 1, no - 1, no - 1)
    {
        return Err("invalid extended Fock reference/basis dimensions".into());
    }
    if n as u128 * n as u128 * 8 * 6 > 256 * 1024 * 1024 {
        return Err("extended Fock matrix workspace exceeds 256 MiB".into());
    }
    let (_, _, _, core) = compute_one_electron_integrals(atoms, &union);
    let eri = |p, q, r, s| {
        if p < no && q < no && r < no && s < no {
            hf.eri[crate::scf::eri_index(p, q, r, s)]
        } else {
            crate::integrals::contracted_eri(&union[p], &union[q], &union[r], &union[s])
        }
    };
    let rows = (0..n)
        .into_par_iter()
        .map(|mu| {
            (0..=mu)
                .map(|nu| {
                    let mut coulomb = 0.;
                    let mut exchange = 0.;
                    for rho in 0..no {
                        for sigma in 0..=rho {
                            let density = hf.density_matrix[(rho, sigma)];
                            coulomb += density
                                * if rho == sigma { 1. } else { 2. }
                                * eri(mu, nu, rho, sigma);
                            exchange += density
                                * if rho == sigma {
                                    eri(mu, rho, nu, sigma)
                                } else {
                                    eri(mu, rho, nu, sigma) + eri(mu, sigma, nu, rho)
                                };
                        }
                    }
                    [core[(mu, nu)] + coulomb - 0.5 * exchange, 0.5 * exchange]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut fock = Matrix::zeros(n, n);
    let mut exchange = Matrix::zeros(n, n);
    for (mu, row) in rows.into_iter().enumerate() {
        for (nu, [f, k]) in row.into_iter().enumerate() {
            fock[(mu, nu)] = f;
            fock[(nu, mu)] = f;
            exchange[(mu, nu)] = k;
            exchange[(nu, mu)] = k;
        }
    }
    Ok(ExtendedFock { fock, exchange })
}
#[derive(Clone, Debug)]
pub struct CabsSinglesResult {
    pub energy: f64,
    pub virtual_energies: Vec<f64>,
    pub couplings: Matrix,
    pub amplitudes: Matrix,
    pub virtual_coefficients: Matrix,
}
pub fn compute_cabs_singles_full(
    hf: &HFResult,
    atoms: &[Atom],
    obs: &[ContractedGaussian],
    cabs: &CABSBasis,
) -> Result<CabsSinglesResult, String> {
    compute_cabs_singles_in_space(hf, atoms, obs, cabs, 0)
}

/// Full virtual resolvent with the first `frozen_core` occupied contributions omitted.
pub fn compute_cabs_singles_in_space(
    hf: &HFResult,
    atoms: &[Atom],
    obs: &[ContractedGaussian],
    cabs: &CABSBasis,
    frozen_core: usize,
) -> Result<CabsSinglesResult, String> {
    let nao = obs.len();
    cabs.validate_reference(hf, nao)?;
    if frozen_core > hf.n_occ {
        return Err("CABS singles frozen core exceeds occupied space".into());
    }
    let fock = extended_fock(hf, atoms, obs, &cabs.aux_functions)?;
    compute_cabs_singles_with_fock(hf, cabs, &fock, frozen_core)
}

/// Reuse the same union Fock matrix for singles and the F12 pair contractions.
/// Singles in the complete OBS-virtual/CABS space using a supplied fixed-density
/// Fock operator and reference occupied energies. The returned response uses
/// the unmodified supplied operator, including a possible fitted OBS block.
pub fn compute_cabs_singles_with_fock(
    hf: &HFResult,
    cabs: &CABSBasis,
    fock: &ExtendedFock,
    frozen_core: usize,
) -> Result<CabsSinglesResult, String> {
    let no = hf.n_occ;
    let nao = cabs.n_obs;
    let nc = cabs.n_cabs;
    cabs.validate_reference(hf, nao)?;
    if frozen_core > no {
        return Err("CABS singles frozen core exceeds occupied space".into());
    }
    let n = nao + cabs.n_raw_aux;
    let nv = hf.mo_coefficients.cols - no;
    if !valid_matrix(&fock.fock, n, n) {
        return Err("invalid reused CABS Fock matrix".into());
    }
    let mut occupied = Matrix::zeros(n, no);
    let mut virtuals = Matrix::zeros(n, nv + nc);
    for mu in 0..nao {
        for i in 0..no {
            occupied[(mu, i)] = hf.mo_coefficients[(mu, i)];
        }
        for a in 0..nv {
            virtuals[(mu, a)] = hf.mo_coefficients[(mu, no + a)];
        }
    }
    let cabs_coeff = cabs.union_coefficients();
    for mu in 0..n {
        for a in 0..nc {
            virtuals[(mu, nv + a)] = cabs_coeff[(mu, a)];
        }
    }
    let (energies, u) = symmetric_congruence(&fock.fock, &virtuals)
        .map_err(|e| e.to_string())?
        .eigh();
    let virtuals = virtuals.dot(&u);
    let couplings = occupied.transpose().dot(&fock.fock).dot(&virtuals);
    singles_from_blocks(hf, couplings, energies, virtuals, frozen_core)
}

fn singles_from_blocks(
    hf: &HFResult,
    couplings: Matrix,
    energies: Vec<f64>,
    virtuals: Matrix,
    frozen_core: usize,
) -> Result<CabsSinglesResult, String> {
    let no = hf.n_occ;
    let mut amplitudes = Matrix::zeros(no, energies.len());
    let mut energy = 0.;
    for i in frozen_core..no {
        for a in 0..energies.len() {
            let gap = energies[a] - hf.orbital_energies[i];
            if !gap.is_finite() || gap <= 1e-12 {
                return Err("CABS singles requires positive full-Fock excitation gaps".into());
            }
            amplitudes[(i, a)] = -couplings[(i, a)] / gap;
            energy += 2. * couplings[(i, a)] * amplitudes[(i, a)];
        }
    }
    Ok(CabsSinglesResult {
        energy,
        virtual_energies: energies,
        couplings,
        amplitudes,
        virtual_coefficients: virtuals,
    })
}
/// Compatibility tuple now contains the complete semicanonical virtual space,
/// including OBS virtuals mixed with CABS, rather than diagonal CABS estimates.
pub fn compute_cabs_singles_details(
    hf: &HFResult,
    atoms: &[Atom],
    obs: &[ContractedGaussian],
    cabs: &CABSBasis,
) -> Result<(f64, Matrix, Vec<f64>), String> {
    let result = compute_cabs_singles_full(hf, atoms, obs, cabs)?;
    Ok((result.energy, result.couplings, result.virtual_energies))
}
pub fn compute_cabs_singles(
    hf: &HFResult,
    atoms: &[Atom],
    obs: &[ContractedGaussian],
    cabs: &CABSBasis,
) -> Result<f64, String> {
    compute_cabs_singles_full(hf, atoms, obs, cabs).map(|result| result.energy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ill_conditioned_congruence_matches_every_high_precision_component() {
        let data: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/data/f12_symmetric_congruence_reference.json"
        ))
        .unwrap();
        let matrix = |value: &serde_json::Value| {
            let rows = value.as_array().unwrap();
            Matrix::from_vec(
                rows.len(),
                rows[0].as_array().unwrap().len(),
                rows.iter()
                    .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
                    .collect(),
            )
        };
        let mut maximum = 0.0_f64;
        let mut count = 0;
        for case in data["cases"].as_array().unwrap() {
            let from_bits = |value: &serde_json::Value| {
                let rows = value.as_array().unwrap();
                Matrix::from_vec(
                    rows.len(),
                    rows[0].as_array().unwrap().len(),
                    rows.iter()
                        .flat_map(|row| {
                            row.as_array()
                                .unwrap()
                                .iter()
                                .map(|x| f64::from_bits(x.as_u64().unwrap()))
                        })
                        .collect(),
                )
            };
            let mut operator = from_bits(&case["operator_bits"]);
            let coefficients = from_bits(&case["coefficient_bits"]);
            let reference = matrix(&case["reference"]);
            let result = symmetric_congruence(&operator, &coefficients).unwrap();
            for p in 0..result.rows {
                for q in 0..result.cols {
                    let error = (result[(p, q)] - reference[(p, q)]).abs();
                    maximum = maximum.max(error);
                    assert!(error < 1e-10, "({p},{q}) error {error:e}");
                    assert_eq!(result[(p, q)], result[(q, p)]);
                    count += 1;
                }
            }
            operator[(0, 1)] += 1e-4;
            assert!(symmetric_congruence(&operator, &coefficients).is_err());
            assert!(symmetric_congruence(&operator, &Matrix::zeros(2, 1)).is_err());
        }
        eprintln!("Compensated F12 congruence: {count} components; maximum error {maximum:e}");
    }
}
