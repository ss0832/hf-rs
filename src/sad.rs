// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! Cached superposition of self-consistent fractional-occupation atomic HF densities.

use crate::basis::{Atom, ContractedGaussian};
use crate::diis::{CdiisEngine, compute_diis_error, max_abs_element};
use crate::matrix::Matrix;
use crate::scf::{build_fock_matrix, compute_eri_tensor, compute_one_electron_integrals};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Debug)]
pub struct AtomicHFResult {
    pub density: Matrix,
    pub orbital_energies: Vec<f64>,
    pub occupations: Vec<f64>,
    pub energy: f64,
    pub iterations: usize,
}

fn occupations(energies: &[f64], n_electrons: f64) -> Result<Vec<f64>, String> {
    if n_electrons < 0.0 || n_electrons > 2.0 * energies.len() as f64 + 1e-9 {
        return Err("Atomic electron count exceeds retained basis capacity".into());
    }
    let mut occ = vec![0.0; energies.len()];
    let mut left = n_electrons;
    let mut i = 0;
    while i < energies.len() && left > 1e-12 {
        let mut end = i + 1;
        while end < energies.len() && (energies[end] - energies[i]).abs() < 1e-6 {
            end += 1;
        }
        let population = left.min(2.0 * (end - i) as f64);
        occ[i..end].fill(population / (end - i) as f64);
        left -= population;
        i = end;
    }
    Ok(occ)
}

fn fractional_density(c: &Matrix, occ: &[f64]) -> Matrix {
    let mut weighted = c.clone();
    for row in 0..c.rows {
        for col in 0..c.cols {
            weighted[(row, col)] *= occ[col].sqrt();
        }
    }
    weighted.dot(&weighted.transpose())
}

/// Spin-averaged atomic RHF ensemble. Degenerate subshells have equal fractional
/// occupations. The two-electron Fock operator and scalar ECP are included.
pub fn atomic_fractional_hf(
    atom: &Atom,
    basis: &[ContractedGaussian],
) -> Result<AtomicHFResult, String> {
    let mut atom = atom.clone();
    atom.position = [0.0; 3];
    let mut basis = basis.to_vec();
    for cg in &mut basis {
        cg.center = [0.0; 3];
        cg.atom_index = 0;
    }
    let (s, _, _, h) = compute_one_electron_integrals(std::slice::from_ref(&atom), &basis);
    let x = s
        .canonical_orthogonalizer(1e-8)
        .map_err(|e| e.to_string())?;
    let xt = x.transpose();
    let eri = compute_eri_tensor(&basis);
    let atomic_fock = |p: &Matrix| {
        if (atom.charge - 1.0).abs() < 1e-12 {
            h.clone()
        } else {
            build_fock_matrix(&h, p, &eri, basis.len())
        }
    };
    let (eps, cp) = xt.dot(&h).dot(&x).eigh();
    let occ = occupations(&eps, atom.charge)?;
    let mut density = fractional_density(&x.dot(&cp), &occ);
    let mut previous = f64::INFINITY;
    let mut diis = CdiisEngine::new(8);
    for iteration in 1..=200 {
        let f = atomic_fock(&density);
        let energy = 0.5
            * density
                .data
                .iter()
                .zip(h.add(&f).data)
                .map(|(p, hf)| p * hf)
                .sum::<f64>();
        let (eps, cp) = xt.dot(&f).dot(&x).eigh();
        let occ = occupations(&eps, atom.charge)?;
        let next = fractional_density(&x.dot(&cp), &occ);
        let error = compute_diis_error(&f, &density, &s);
        if (energy - previous).abs() < 1e-11
            && density.rmsd(&next) < 1e-9
            && max_abs_element(&error) < 1e-8
        {
            let fnext = atomic_fock(&next);
            let energy = 0.5
                * next
                    .data
                    .iter()
                    .zip(h.add(&fnext).data)
                    .map(|(p, hf)| p * hf)
                    .sum::<f64>();
            return Ok(AtomicHFResult {
                density: next,
                orbital_energies: eps,
                occupations: occ,
                energy,
                iterations: iteration,
            });
        }
        previous = energy;
        diis.push(f.clone(), error);
        let f_use = diis.extrapolate().unwrap_or(f);
        let (eps, cp) = xt.dot(&f_use).dot(&x).eigh();
        let occ = occupations(&eps, atom.charge)?;
        let next = fractional_density(&x.dot(&cp), &occ);
        density = if iteration <= 5 {
            density.scale(0.5).add(&next.scale(0.5))
        } else {
            next
        };
    }
    Err(format!(
        "Fractional atomic HF did not converge for {}",
        atom.symbol
    ))
}

type CacheEntry = (Vec<u64>, Arc<AtomicHFResult>);
fn cache() -> &'static Mutex<VecDeque<CacheEntry>> {
    static CACHE: OnceLock<Mutex<VecDeque<CacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(VecDeque::new()))
}
fn cache_key(atom: &Atom, basis: &[ContractedGaussian]) -> Vec<u64> {
    let mut key = vec![
        atom.atomic_number as u64,
        atom.charge.to_bits(),
        basis.len() as u64,
    ];
    for cg in basis {
        key.extend(cg.l.iter().map(|&x| x as u64));
        key.push(cg.primitives.len() as u64);
        for p in &cg.primitives {
            key.extend([p.exponent.to_bits(), p.coefficient.to_bits()]);
        }
    }
    if let Some(ecp) = &atom.ecp {
        key.extend([1, ecp.n_core as u64, ecp.l_max as u64]);
        for channel in std::iter::once(&ecp.local).chain(ecp.semilocal.iter()) {
            key.push(channel.len() as u64);
            for term in channel {
                key.extend([
                    term.power as u64,
                    term.exponent.to_bits(),
                    term.coefficient.to_bits(),
                ]);
            }
        }
    } else {
        key.push(0);
    }
    key
}
fn entry_bytes((key, result): &CacheEntry) -> usize {
    8 * (key.len()
        + result.density.data.len()
        + result.orbital_energies.len()
        + result.occupations.len())
        + 256
}
fn cached_atomic_density(
    atom: &Atom,
    basis: &[ContractedGaussian],
) -> Result<Arc<AtomicHFResult>, String> {
    let key = cache_key(atom, basis);
    {
        let mut entries = cache()
            .lock()
            .map_err(|_| "Atomic HF cache lock poisoned")?;
        if let Some(i) = entries.iter().position(|(k, _)| *k == key) {
            let entry = entries.remove(i).ok_or("Atomic HF cache index invalid")?;
            let result = entry.1.clone();
            entries.push_back(entry);
            return Ok(result);
        }
    }
    // Independent atoms/threads do not hold the cache lock during SCF.
    let result = Arc::new(atomic_fractional_hf(atom, basis)?);
    let entry = (key, result.clone());
    const BUDGET: usize = 32 * 1024 * 1024;
    let bytes = entry_bytes(&entry);
    if bytes <= BUDGET {
        let mut entries = cache()
            .lock()
            .map_err(|_| "Atomic HF cache lock poisoned")?;
        let mut used = entries.iter().map(entry_bytes).sum::<usize>();
        while used + bytes > BUDGET || entries.len() >= 128 {
            if let Some(old) = entries.pop_front() {
                used -= entry_bytes(&old);
            } else {
                break;
            }
        }
        entries.push_back(entry);
    }
    Ok(result)
}

/// Superpose neutral atomic ensembles, then rescale Tr(P S) to the molecular charge.
pub fn compute_sad_density(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    overlap: &Matrix,
) -> Result<Matrix, String> {
    let n = basis.len();
    let target = atoms.iter().map(|a| a.charge).sum::<f64>() - charge as f64;
    if target <= 0.0 || overlap.rows != n || overlap.cols != n {
        return Err("Invalid SAD electron count or overlap".into());
    }
    let mut density = Matrix::zeros(n, n);
    for (a, atom) in atoms.iter().enumerate() {
        let indices = basis
            .iter()
            .enumerate()
            .filter_map(|(i, cg)| (cg.atom_index == a).then_some(i))
            .collect::<Vec<_>>();
        if indices.is_empty() {
            return Err(format!("No basis functions for {}", atom.symbol));
        }
        let local = indices
            .iter()
            .map(|&i| basis[i].clone())
            .collect::<Vec<_>>();
        let atomic = cached_atomic_density(atom, &local)?;
        for (i, &ii) in indices.iter().enumerate() {
            for (j, &jj) in indices.iter().enumerate() {
                density[(ii, jj)] = atomic.density[(i, j)];
            }
        }
    }
    let electrons = density
        .data
        .iter()
        .zip(&overlap.data)
        .map(|(p, s)| p * s)
        .sum::<f64>();
    if electrons <= 1e-12 {
        return Err("Zero SAD population".into());
    }
    Ok(density.scale(target / electrons))
}
