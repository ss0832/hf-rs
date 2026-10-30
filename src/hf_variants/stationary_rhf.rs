// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Residual-certified RHF for a supplied one-electron Hamiltonian. The caller
//! owns the packed ERIs; reference/model solves can borrow the same allocation.

use crate::{
    Matrix,
    scf::{RhfConvergence, build_fock_matrix},
};

pub(super) struct StationaryRhf {
    pub total_energy: f64,
    pub density: Matrix,
    pub fock: Matrix,
    pub coefficients: Matrix,
    pub orbital_energies: Vec<f64>,
    pub residual_norm: f64,
    pub density_change: f64,
    pub iterations: usize,
}
fn norm(a: &Matrix) -> f64 {
    a.data.iter().map(|x| x * x).sum::<f64>().sqrt()
}

pub(super) fn solve(
    h: &Matrix,
    s: &Matrix,
    eri: &[f64],
    nuclear: f64,
    n_occ: usize,
    max_iterations: usize,
    convergence: RhfConvergence,
) -> Result<StationaryRhf, String> {
    solve_with_fock(h, s, nuclear, n_occ, max_iterations, convergence, |p| {
        build_fock_matrix(h, p, eri, s.rows)
    })
}

/// Variational RHF for a linear model Fock operator. `h` is its one-electron term,
/// so `Tr(P(h+F))/2 + nuclear` is the model energy, including every field term.
pub(super) fn solve_with_fock(
    h: &Matrix,
    s: &Matrix,
    nuclear: f64,
    n_occ: usize,
    max_iterations: usize,
    convergence: RhfConvergence,
    build_fock: impl Fn(&Matrix) -> Matrix,
) -> Result<StationaryRhf, String> {
    let n = s.rows;
    if n == 0
        || max_iterations == 0
        || n_occ == 0
        || n_occ > n
        || !nuclear.is_finite()
        || [
            convergence.energy,
            convergence.residual,
            convergence.density,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err("invalid supplied-Hamiltonian RHF controls".into());
    }
    let x = s
        .canonical_orthogonalizer(1e-8)
        .map_err(|e| e.to_string())?;
    let rank = x.cols;
    if n_occ > rank {
        return Err("relativistic RHF occupations exceed the retained overlap rank".into());
    }
    if n_occ == rank {
        // A completely occupied retained space has the unique density 2 X X^T.
        // Rotating that whole space at each iteration only adds roundoff noise;
        // there is no occupied/virtual variational degree of freedom to solve.
        let density = x.occupied_density(rank, 2.);
        let fock = build_fock(&density);
        let residual = x
            .transpose()
            .dot(&fock.dot(&density).dot(s).sub(&s.dot(&density).dot(&fock)))
            .dot(&x);
        let residual_norm = norm(&residual);
        if residual_norm > convergence.residual {
            return Err(format!(
                "full-occupation RHF metric residual {residual_norm:e} exceeds tolerance"
            ));
        }
        let (orbital_energies, u) = x.transpose().dot(&fock).dot(&x).eigh();
        let total_energy = nuclear
            + 0.5
                * density
                    .data
                    .iter()
                    .zip(h.add(&fock).data)
                    .map(|(p, h)| p * h)
                    .sum::<f64>();
        return Ok(StationaryRhf {
            total_energy,
            density,
            fock,
            coefficients: x.dot(&u),
            orbital_energies,
            residual_norm,
            density_change: 0.,
            iterations: 1,
        });
    }
    let (_, u) = x.transpose().dot(h).dot(&x).eigh();
    let mut c = x.dot(&u);
    let mut p = c.occupied_density(n_occ, 2.);
    let mut previous = None;
    let mut change = f64::INFINITY;
    let mut residual_norm = f64::INFINITY;
    let mut diis = crate::diis::CdiisEngine::new(8);
    for iteration in 1..=max_iterations {
        let f = build_fock(&p);
        let energy = nuclear
            + 0.5
                * p.data
                    .iter()
                    .zip(h.add(&f).data)
                    .map(|(p, h)| p * h)
                    .sum::<f64>();
        let residual = x
            .transpose()
            .dot(&f.dot(&p).dot(s).sub(&s.dot(&p).dot(&f)))
            .dot(&x);
        residual_norm = norm(&residual);
        if previous.is_some_and(|e: f64| (energy - e).abs() <= convergence.energy)
            && residual_norm <= convergence.residual
            && change <= convergence.density
        {
            let fm = c.transpose().dot(&f).dot(&c);
            let mut rotation = Matrix::identity(rank);
            let mut energies = vec![0.; rank];
            for (start, end) in [(0, n_occ), (n_occ, rank)] {
                if start == end {
                    continue;
                }
                let block = Matrix::from_vec(
                    end - start,
                    end - start,
                    (start..end)
                        .flat_map(|p| (start..end).map(move |q| (p, q)))
                        .map(|(p, q)| fm[(p, q)])
                        .collect(),
                );
                let (values, vectors) = block.eigh();
                for p in start..end {
                    energies[p] = values[p - start];
                    for q in start..end {
                        rotation[(p, q)] = vectors[(p - start, q - start)];
                    }
                }
            }
            return Ok(StationaryRhf {
                total_energy: energy,
                density: p,
                fock: f,
                coefficients: c.dot(&rotation),
                orbital_energies: energies,
                residual_norm,
                density_change: change,
                iterations: iteration,
            });
        }
        previous = Some(energy);
        let orthofock = x.transpose().dot(&f).dot(&x);
        diis.push(orthofock.clone(), residual);
        let next = diis.extrapolate().unwrap_or(orthofock);
        let (_, u) = next.eigh();
        c = x.dot(&u);
        let next = c.occupied_density(n_occ, 2.);
        change = norm(&next.sub(&p)) / n as f64;
        p = next;
    }
    Err(format!(
        "relativistic RHF did not converge in {max_iterations} iterations: residual {residual_norm:e}, density change {change:e}"
    ))
}
