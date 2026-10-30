// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Point-nucleus scalar-relativistic S/T/V/p.V.p integrals and exact nuclear
//! tangents. Translation identities cancel same-center potential terms before
//! evaluation; no displaced geometry or empirical smoothing is used.

use crate::{
    Matrix,
    basis::{Atom, ContractedGaussian},
    derivatives::diff_cg_1st,
    integrals::{contracted_kinetic, contracted_nuclear_attraction, contracted_overlap},
    scf::compute_one_electron_integrals,
};

pub struct X2cIntegralMatrices {
    pub overlap: Matrix,
    pub kinetic: Matrix,
    pub potential: Matrix,
    pub pvp: Matrix,
}

fn check(atoms: &[Atom], basis: &[ContractedGaussian], budget: usize) -> Result<(), String> {
    let n = basis.len();
    if n == 0
        || atoms.is_empty()
        || atoms.iter().any(|a| {
            !a.charge.is_finite() || a.charge < 0. || a.position.iter().any(|v| !v.is_finite())
        })
        || basis.iter().any(|b| {
            b.atom_index >= atoms.len()
                || b.center.iter().any(|v| !v.is_finite())
                || b.l.iter().any(|&l| l > 7)
                || b.l.iter().sum::<usize>() > 7
                || b.primitives.is_empty()
                || b.primitives.iter().any(|p| {
                    !p.exponent.is_finite() || p.exponent <= 0. || !p.coefficient.is_finite()
                })
        })
    {
        return Err("invalid scalar-relativistic atoms or Cartesian Gaussian basis".into());
    }
    if atoms.iter().any(|a| a.ecp.is_some()) {
        return Err("point-nucleus X2C requires an all-electron Hamiltonian".into());
    }
    let primitives = basis
        .iter()
        .map(|b| b.primitives.len() as u128)
        .sum::<u128>();
    if 64 * (n as u128).pow(2) + 384 * primitives > budget as u128 {
        return Err("scalar-relativistic integral workspace exceeds max_work_bytes".into());
    }
    Ok(())
}

fn pvp(a: &ContractedGaussian, b: &ContractedGaussian, atoms: &[Atom]) -> f64 {
    let mut value = 0.;
    for axis in 0..3 {
        let da = diff_cg_1st(a, axis);
        let db = diff_cg_1st(b, axis);
        for left in &da.terms {
            for right in &db.terms {
                value += contracted_nuclear_attraction(left, right, atoms);
            }
        }
    }
    value
}

pub fn spin_free_integrals(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    max_work_bytes: usize,
) -> Result<X2cIntegralMatrices, String> {
    check(atoms, basis, max_work_bytes)?;
    let (overlap, kinetic, potential, _) = compute_one_electron_integrals(atoms, basis);
    let n = basis.len();
    let mut w = Matrix::zeros(n, n);
    for p in 0..n {
        for q in 0..=p {
            let value = pvp(&basis[p], &basis[q], atoms);
            w[(p, q)] = value;
            w[(q, p)] = value;
        }
    }
    Ok(X2cIntegralMatrices {
        overlap,
        kinetic,
        potential,
        pvp: w,
    })
}

/// Complete derivative of S/T/V/W with respect to one nuclear coordinate.
/// The explicit potential-center derivative is minus the two basis-center
/// derivatives of that nucleus's potential. Combining those coefficients first
/// removes exact same-center cancellations, including the singular core term.
pub fn spin_free_nuclear_tangent(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    nucleus: usize,
    axis: usize,
    max_work_bytes: usize,
) -> Result<X2cIntegralMatrices, String> {
    check(atoms, basis, max_work_bytes)?;
    if nucleus >= atoms.len() || axis >= 3 {
        return Err("invalid scalar-relativistic nuclear coordinate".into());
    }
    let n = basis.len();
    let mut result = X2cIntegralMatrices {
        overlap: Matrix::zeros(n, n),
        kinetic: Matrix::zeros(n, n),
        potential: Matrix::zeros(n, n),
        pvp: Matrix::zeros(n, n),
    };
    for p in 0..n {
        for q in 0..=p {
            let (a, b) = (&basis[p], &basis[q]);
            let mut ds = 0.;
            let mut dt = 0.;
            let mut dv = 0.;
            let mut dw = 0.;
            let left_moves = a.atom_index == nucleus;
            let right_moves = b.atom_index == nucleus;
            for term in diff_cg_1st(a, axis).terms {
                if left_moves && !right_moves {
                    ds += contracted_overlap(&term, b);
                    dt += contracted_kinetic(&term, b);
                }
                for (center, atom) in atoms.iter().enumerate() {
                    let coefficient = i32::from(left_moves) - i32::from(center == nucleus);
                    if coefficient != 0 {
                        dv += f64::from(coefficient)
                            * contracted_nuclear_attraction(&term, b, std::slice::from_ref(atom));
                        dw += f64::from(coefficient) * pvp(&term, b, std::slice::from_ref(atom));
                    }
                }
            }
            for term in diff_cg_1st(b, axis).terms {
                if right_moves && !left_moves {
                    ds += contracted_overlap(a, &term);
                    dt += contracted_kinetic(a, &term);
                }
                for (center, atom) in atoms.iter().enumerate() {
                    let coefficient = i32::from(right_moves) - i32::from(center == nucleus);
                    if coefficient != 0 {
                        dv += f64::from(coefficient)
                            * contracted_nuclear_attraction(a, &term, std::slice::from_ref(atom));
                        dw += f64::from(coefficient) * pvp(a, &term, std::slice::from_ref(atom));
                    }
                }
            }
            for (matrix, value) in [
                (&mut result.overlap, ds),
                (&mut result.kinetic, dt),
                (&mut result.potential, dv),
                (&mut result.pvp, dw),
            ] {
                matrix[(p, q)] = value;
                matrix[(q, p)] = value;
            }
        }
    }
    Ok(result)
}
