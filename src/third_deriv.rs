// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
// Copyright (c) 2026 ss0832

//! Analytical nuclear third derivatives for RHF and UHF, including scalar ECPs.
//! A single reference geometry and order-by-order SCF response are used throughout.

use crate::basis::{Atom, BasisSetMap};
use crate::error::HfError;
use crate::hf_derivatives::{direction_from_flat, permutations3, with_reference};

/// 3rd derivative tensor with shape [3N][3N][3N]
#[derive(Clone, Debug)]
pub struct ThirdDerivativeTensor {
    pub n_atoms: usize,
    pub dim: usize, // 3 * n_atoms
    pub data: Vec<f64>,
}

impl ThirdDerivativeTensor {
    pub fn zeros(n_atoms: usize) -> Self {
        let dim = 3 * n_atoms;
        ThirdDerivativeTensor {
            n_atoms,
            dim,
            data: vec![0.0; dim * dim * dim],
        }
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize, z: usize) -> f64 {
        self.data[(x * self.dim + y) * self.dim + z]
    }

    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, z: usize, val: f64) {
        self.data[(x * self.dim + y) * self.dim + z] = val;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThirdDerivativeBlock {
    pub active_atoms: Vec<usize>,
    pub n_active_atoms: usize,
    pub dim: usize,     // 3 * n_active_atoms
    pub data: Vec<f64>, // shape [dim][dim][dim]
}

impl ThirdDerivativeBlock {
    pub fn zeros(active_atoms: &[usize]) -> Self {
        let n_active = active_atoms.len();
        let dim = 3 * n_active;
        ThirdDerivativeBlock {
            active_atoms: active_atoms.to_vec(),
            n_active_atoms: n_active,
            dim,
            data: vec![0.0; dim * dim * dim],
        }
    }

    #[inline(always)]
    pub fn get(&self, local_x: usize, local_y: usize, local_z: usize) -> f64 {
        self.data[(local_x * self.dim + local_y) * self.dim + local_z]
    }

    #[inline(always)]
    pub fn set(&mut self, local_x: usize, local_y: usize, local_z: usize, val: f64) {
        self.data[(local_x * self.dim + local_y) * self.dim + local_z] = val;
    }

    /// Retrieve derivative using active atom index (0..n_active) and Cartesian component (0..3).
    pub fn get_by_active_atom(
        &self,
        a_idx: usize,
        comp_a: usize,
        b_idx: usize,
        comp_b: usize,
        c_idx: usize,
        comp_c: usize,
    ) -> f64 {
        let lx = a_idx * 3 + comp_a;
        let ly = b_idx * 3 + comp_b;
        let lz = c_idx * 3 + comp_c;
        self.get(lx, ly, lz)
    }
}

pub fn compute_third_derivatives_general(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    multiplicity: usize,
    broken_symmetry: bool,
    max_iter: usize,
    tol: f64,
) -> ThirdDerivativeTensor {
    try_compute_third_derivatives_general(
        atoms,
        basis_map,
        charge,
        multiplicity,
        broken_symmetry,
        max_iter,
        tol,
    )
    .expect("analytical third-order HF response failed")
}

/// Checked full-tensor API; nonconvergence and singular response are returned as errors.
pub fn try_compute_third_derivatives_general(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    multiplicity: usize,
    broken_symmetry: bool,
    max_iter: usize,
    tol: f64,
) -> Result<ThirdDerivativeTensor, HfError> {
    with_reference(
        atoms,
        basis_map,
        charge,
        multiplicity,
        broken_symmetry,
        max_iter,
        tol,
        |ctx| {
            let mut tensor = ThirdDerivativeTensor::zeros(atoms.len());
            for i in 0..tensor.dim {
                for j in 0..=i {
                    for k in 0..=j {
                        let value = ctx.coordinate_derivative(&[i, j, k])?;
                        for [a, b, c] in permutations3([i, j, k]) {
                            tensor.set(a, b, c, value);
                        }
                    }
                }
            }
            Ok(tensor)
        },
    )
}

pub fn compute_third_derivatives_with_charge(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> ThirdDerivativeTensor {
    compute_third_derivatives_general(atoms, basis_map, charge, 1, false, max_iter, tol)
}

pub fn compute_third_derivatives(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    max_iter: usize,
    tol: f64,
) -> ThirdDerivativeTensor {
    compute_third_derivatives_with_charge(atoms, basis_map, 0, max_iter, tol)
}

pub fn compute_uhf_third_derivatives(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    multiplicity: usize,
    broken_symmetry: bool,
    max_iter: usize,
    tol: f64,
) -> ThirdDerivativeTensor {
    compute_third_derivatives_general(
        atoms,
        basis_map,
        charge,
        multiplicity,
        broken_symmetry,
        max_iter,
        tol,
    )
}

pub fn compute_third_derivatives_block(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    active_atoms: &[usize],
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> ThirdDerivativeBlock {
    with_reference(atoms, basis_map, charge, 1, false, max_iter, tol, |ctx| {
        for (i, &a) in active_atoms.iter().enumerate() {
            if a >= atoms.len() || active_atoms[..i].contains(&a) {
                return Err(HfError::InvalidInput(
                    "invalid active atom selection".into(),
                ));
            }
        }
        let mut block = ThirdDerivativeBlock::zeros(active_atoms);
        for i in 0..block.dim {
            for j in 0..=i {
                for k in 0..=j {
                    let coordinate = |x: usize| 3 * active_atoms[x / 3] + x % 3;
                    let value =
                        ctx.coordinate_derivative(&[coordinate(i), coordinate(j), coordinate(k)])?;
                    for [a, b, c] in permutations3([i, j, k]) {
                        block.set(a, b, c, value);
                    }
                }
            }
        }
        Ok(block)
    })
    .expect("analytical third-order HF block response failed")
}

pub fn compute_third_directional_derivative(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    u: &[f64],
    v: &[f64],
    w: &[f64],
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> f64 {
    with_reference(atoms, basis_map, charge, 1, false, max_iter, tol, |ctx| {
        let directions = [u, v, w]
            .iter()
            .map(|x| direction_from_flat(x, atoms.len()))
            .collect::<Result<Vec<_>, _>>()?;
        ctx.derivative(&directions)
    })
    .expect("analytical third-order directional HF response failed")
}

pub fn compute_third_derivatives_subspace(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    modes: &[Vec<f64>],
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    with_reference(atoms, basis_map, charge, 1, false, max_iter, tol, |ctx| {
        let m = modes.len();
        let mut tensor = vec![0.0; m * m * m];
        let directions = modes
            .iter()
            .map(|v| direction_from_flat(v, atoms.len()))
            .collect::<Result<Vec<_>, _>>()?;
        for i in 0..m {
            for j in 0..=i {
                for k in 0..=j {
                    let value = ctx.derivative(&[
                        directions[i].clone(),
                        directions[j].clone(),
                        directions[k].clone(),
                    ])?;
                    for [a, b, c] in permutations3([i, j, k]) {
                        tensor[(a * m + b) * m + c] = value;
                    }
                }
            }
        }
        Ok(tensor)
    })
    .expect("analytical third-order normal-mode HF response failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::parse_gaussian94;

    #[test]
    fn test_h2_third_derivatives() {
        let basis_map = parse_gaussian94(crate::basis::STO_3G_GBS);
        let r = 1.4;
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -r / 2.0]),
            Atom::new("H", [0.0, 0.0, r / 2.0]),
        ];

        let tensor = compute_third_derivatives(&atoms, &basis_map, 50, 1e-12);
        let dim = 3 * atoms.len();

        // 2. Translational invariance: \sum_A T_{A\tau, B\eta, C\xi} = 0
        for tau in 0..3 {
            for eta in 0..3 {
                for xi in 0..3 {
                    let mut sum_a = 0.0;
                    for a in 0..atoms.len() {
                        sum_a += tensor.get(3 * a + tau, eta, xi);
                    }
                    assert!(
                        sum_a.abs() < 1e-5,
                        "3rd deriv translational invariance failed: sum = {}",
                        sum_a
                    );
                }
            }
        }

        // Significant non-zero 3rd derivative components along z (stretch axis)
        let t_z0_z0_z0 = tensor.get(2, 2, 2);
        let t_z0_z0_z1 = tensor.get(2, 2, 5);
        assert!((t_z0_z0_z0 + t_z0_z0_z1).abs() < 1e-5);
        assert!(t_z0_z0_z0.abs() > 0.1);

        // Directional derivative along z-axis unit vector
        let mut u = vec![0.0; dim];
        u[2] = 1.0;
        let d3_uuu =
            compute_third_directional_derivative(&atoms, &basis_map, &u, &u, &u, 0, 50, 1e-12);
        assert!((d3_uuu - t_z0_z0_z0).abs() < 1e-6);
    }
}
