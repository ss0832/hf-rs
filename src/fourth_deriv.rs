// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
// Copyright (c) 2026 ss0832

//! Analytical nuclear fourth derivatives for RHF and UHF, including scalar ECPs.
//! Explicit integral derivatives and implicit density response are evaluated at one
//! reference geometry. Directional, tensor, block and normal-mode APIs share this backend.

use crate::basis::{Atom, BasisSetMap};
use crate::error::HfError;
use crate::hf_derivatives::{direction_from_flat, permutations4, with_reference};

/// 4th derivative tensor with shape [3N][3N][3N][3N]
#[derive(Clone, Debug)]
pub struct FourthDerivativeTensor {
    pub n_atoms: usize,
    pub dim: usize, // 3 * n_atoms
    pub data: Vec<f64>,
}

impl FourthDerivativeTensor {
    pub fn zeros(n_atoms: usize) -> Self {
        let dim = 3 * n_atoms;
        FourthDerivativeTensor {
            n_atoms,
            dim,
            data: vec![0.0; dim * dim * dim * dim],
        }
    }

    #[inline(always)]
    pub fn get(&self, w: usize, x: usize, y: usize, z: usize) -> f64 {
        let d = self.dim;
        self.data[((w * d + x) * d + y) * d + z]
    }

    #[inline(always)]
    pub fn set(&mut self, w: usize, x: usize, y: usize, z: usize, val: f64) {
        let d = self.dim;
        self.data[((w * d + x) * d + y) * d + z] = val;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FourthDerivativeBlock {
    pub active_atoms: Vec<usize>,
    pub n_active_atoms: usize,
    pub dim: usize,     // 3 * n_active_atoms
    pub data: Vec<f64>, // shape [dim][dim][dim][dim]
}

impl FourthDerivativeBlock {
    pub fn zeros(active_atoms: &[usize]) -> Self {
        let n_active = active_atoms.len();
        let dim = 3 * n_active;
        FourthDerivativeBlock {
            active_atoms: active_atoms.to_vec(),
            n_active_atoms: n_active,
            dim,
            data: vec![0.0; dim * dim * dim * dim],
        }
    }

    #[inline(always)]
    pub fn get(&self, local_w: usize, local_x: usize, local_y: usize, local_z: usize) -> f64 {
        let d = self.dim;
        self.data[((local_w * d + local_x) * d + local_y) * d + local_z]
    }

    #[inline(always)]
    pub fn set(
        &mut self,
        local_w: usize,
        local_x: usize,
        local_y: usize,
        local_z: usize,
        val: f64,
    ) {
        let d = self.dim;
        self.data[((local_w * d + local_x) * d + local_y) * d + local_z] = val;
    }

    pub fn get_by_active_atom(
        &self,
        a_idx: usize,
        comp_a: usize,
        b_idx: usize,
        comp_b: usize,
        c_idx: usize,
        comp_c: usize,
        d_idx: usize,
        comp_d: usize,
    ) -> f64 {
        let lw = a_idx * 3 + comp_a;
        let lx = b_idx * 3 + comp_b;
        let ly = c_idx * 3 + comp_c;
        let lz = d_idx * 3 + comp_d;
        self.get(lw, lx, ly, lz)
    }
}

/// Compatibility helper for callers that explicitly need to average a numerical tensor.
/// The analytical HF drivers do not call this function.
pub fn symmetrize_fourth_tensor(tensor: &mut FourthDerivativeTensor, dim: usize) {
    let perms: [[usize; 4]; 24] = [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [0, 2, 1, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [0, 3, 2, 1],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 0, 2],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 0, 3, 1],
        [2, 1, 0, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 0, 1, 2],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 1, 2, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ];

    for w in 0..dim {
        for x in 0..=w {
            for y in 0..=x {
                for z in 0..=y {
                    let idx = [w, x, y, z];
                    let mut sum_val = 0.0;
                    for p in &perms {
                        sum_val += tensor.get(idx[p[0]], idx[p[1]], idx[p[2]], idx[p[3]]);
                    }
                    let avg = sum_val / 24.0;
                    for p in &perms {
                        tensor.set(idx[p[0]], idx[p[1]], idx[p[2]], idx[p[3]], avg);
                    }
                }
            }
        }
    }
}

pub fn compute_fourth_derivatives_general(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    max_iter: usize,
    tol: f64,
) -> FourthDerivativeTensor {
    try_compute_fourth_derivatives_general(
        atoms,
        basis_map,
        charge,
        multiplicity,
        is_broken_symmetry,
        max_iter,
        tol,
    )
    .expect("analytical fourth-order HF response failed")
}

/// Checked full-tensor API; nonconvergence and singular response are returned as errors.
pub fn try_compute_fourth_derivatives_general(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    max_iter: usize,
    tol: f64,
) -> Result<FourthDerivativeTensor, HfError> {
    with_reference(
        atoms,
        basis_map,
        charge,
        multiplicity,
        is_broken_symmetry,
        max_iter,
        tol,
        |ctx| {
            let mut tensor = FourthDerivativeTensor::zeros(atoms.len());
            for i in 0..tensor.dim {
                for j in 0..=i {
                    for k in 0..=j {
                        for l in 0..=k {
                            let value = ctx.coordinate_derivative(&[i, j, k, l])?;
                            for [a, b, c, d] in permutations4([i, j, k, l]) {
                                tensor.set(a, b, c, d, value);
                            }
                        }
                    }
                }
            }
            Ok(tensor)
        },
    )
}

pub fn compute_fourth_derivatives_with_charge(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> FourthDerivativeTensor {
    compute_fourth_derivatives_general(atoms, basis_map, charge, 1, false, max_iter, tol)
}

pub fn compute_fourth_derivatives(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    max_iter: usize,
    tol: f64,
) -> FourthDerivativeTensor {
    compute_fourth_derivatives_general(atoms, basis_map, 0, 1, false, max_iter, tol)
}

pub fn compute_uhf_fourth_derivatives(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    max_iter: usize,
    tol: f64,
) -> FourthDerivativeTensor {
    compute_fourth_derivatives_general(
        atoms,
        basis_map,
        charge,
        multiplicity,
        is_broken_symmetry,
        max_iter,
        tol,
    )
}

pub fn compute_fourth_derivatives_block(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    active_atoms: &[usize],
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    max_iter: usize,
    tol: f64,
) -> FourthDerivativeBlock {
    with_reference(
        atoms,
        basis_map,
        charge,
        multiplicity,
        is_broken_symmetry,
        max_iter,
        tol,
        |ctx| {
            for (i, &a) in active_atoms.iter().enumerate() {
                if a >= atoms.len() || active_atoms[..i].contains(&a) {
                    return Err(HfError::InvalidInput(
                        "invalid active atom selection".into(),
                    ));
                }
            }
            let mut block = FourthDerivativeBlock::zeros(active_atoms);
            for i in 0..block.dim {
                for j in 0..=i {
                    for k in 0..=j {
                        for l in 0..=k {
                            let coordinate = |x: usize| 3 * active_atoms[x / 3] + x % 3;
                            let value = ctx.coordinate_derivative(&[
                                coordinate(i),
                                coordinate(j),
                                coordinate(k),
                                coordinate(l),
                            ])?;
                            for [a, b, c, d] in permutations4([i, j, k, l]) {
                                block.set(a, b, c, d, value);
                            }
                        }
                    }
                }
            }
            Ok(block)
        },
    )
    .expect("analytical fourth-order HF block response failed")
}

pub fn compute_fourth_directional_derivative(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    u: &[f64],
    v: &[f64],
    w: &[f64],
    t: &[f64],
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    max_iter: usize,
    tol: f64,
) -> f64 {
    with_reference(
        atoms,
        basis_map,
        charge,
        multiplicity,
        is_broken_symmetry,
        max_iter,
        tol,
        |ctx| {
            let directions = [u, v, w, t]
                .iter()
                .map(|x| direction_from_flat(x, atoms.len()))
                .collect::<Result<Vec<_>, _>>()?;
            ctx.derivative(&directions)
        },
    )
    .expect("analytical fourth-order directional HF response failed")
}

pub fn compute_fourth_derivatives_subspace(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    modes: &[Vec<f64>],
    charge: i32,
    multiplicity: usize,
    is_broken_symmetry: bool,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    with_reference(
        atoms,
        basis_map,
        charge,
        multiplicity,
        is_broken_symmetry,
        max_iter,
        tol,
        |ctx| {
            let m = modes.len();
            let mut tensor = vec![0.0; m * m * m * m];
            let directions = modes
                .iter()
                .map(|v| direction_from_flat(v, atoms.len()))
                .collect::<Result<Vec<_>, _>>()?;
            for i in 0..m {
                for j in 0..=i {
                    for k in 0..=j {
                        for l in 0..=k {
                            let value = ctx.derivative(&[
                                directions[i].clone(),
                                directions[j].clone(),
                                directions[k].clone(),
                                directions[l].clone(),
                            ])?;
                            for [a, b, c, d] in permutations4([i, j, k, l]) {
                                tensor[((a * m + b) * m + c) * m + d] = value;
                            }
                        }
                    }
                }
            }
            Ok(tensor)
        },
    )
    .expect("analytical fourth-order normal-mode HF response failed")
}

/// Exact fourth directional derivative of point-charge nuclear repulsion.
pub fn compute_nuclear_repulsion_fourth_directional(atoms: &[Atom], v: &[f64]) -> f64 {
    let mut sum_v4 = 0.0;
    for a in 0..atoms.len() {
        for b in (a + 1)..atoms.len() {
            let za_zb = atoms[a].charge * atoms[b].charge;
            let dx = atoms[a].position[0] - atoms[b].position[0];
            let dy = atoms[a].position[1] - atoms[b].position[1];
            let dz = atoms[a].position[2] - atoms[b].position[2];

            let vx = v[3 * a] - v[3 * b];
            let vy = v[3 * a + 1] - v[3 * b + 1];
            let vz = v[3 * a + 2] - v[3 * b + 2];

            let r2 = dx * dx + dy * dy + dz * dz;
            let r = r2.sqrt();
            let d1 = 2.0 * (dx * vx + dy * vy + dz * vz);
            let d2 = vx * vx + vy * vy + vz * vz;

            let r5 = r2 * r2 * r;
            let r7 = r5 * r2;
            let r9 = r7 * r2;

            let term = (105.0 / 16.0) * (d1 * d1 * d1 * d1) / r9
                - (45.0 / 2.0) * (d1 * d1 * d2) / r7
                + 9.0 * (d2 * d2) / r5;

            sum_v4 += za_zb * term;
        }
    }
    sum_v4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::parse_gaussian94;

    #[test]
    fn test_h2_fourth_derivatives_rhf() {
        let basis_map = parse_gaussian94(crate::basis::STO_3G_GBS);
        let r = 1.4;
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -r / 2.0]),
            Atom::new("H", [0.0, 0.0, r / 2.0]),
        ];

        let tensor = compute_fourth_derivatives(&atoms, &basis_map, 50, 1e-12);
        let dim = tensor.dim;
        assert_eq!(dim, 6);

        // 2. Non-zero elements along molecular bond axis (z0 = 2, z1 = 5)
        let q_2222 = tensor.get(2, 2, 2, 2);
        let q_5555 = tensor.get(5, 5, 5, 5);

        assert!(
            q_2222.abs() > 0.01,
            "Bond-axis 4th derivative should be non-zero"
        );
        // Inversion symmetry between H(0) and H(1)
        assert!((q_2222 - q_5555).abs() < 1e-6);

        // 3. Directional derivative along z-axis unit vector:
        let mut u = vec![0.0; dim];
        u[2] = 1.0;
        let d4_uuuu = compute_fourth_directional_derivative(
            &atoms, &basis_map, &u, &u, &u, &u, 0, 1, false, 50, 1e-12,
        );
        assert!((d4_uuuu - q_2222).abs() < 1e-6);
    }

    #[test]
    fn test_h2_plus_fourth_derivatives_uhf() {
        let basis_map = parse_gaussian94(crate::basis::STO_3G_GBS);
        let r = 2.0;
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -r / 2.0]),
            Atom::new("H", [0.0, 0.0, r / 2.0]),
        ];

        let tensor = compute_uhf_fourth_derivatives(&atoms, &basis_map, 1, 2, false, 50, 1e-12);
        let dim = tensor.dim;
        assert_eq!(dim, 6);

        let q_2222 = tensor.get(2, 2, 2, 2);
        let q_5555 = tensor.get(5, 5, 5, 5);

        assert!(q_2222.abs() > 0.01);
        assert!((q_2222 - q_5555).abs() < 1e-6);
    }

    #[test]
    fn test_nuclear_repulsion_fourth_directional_exact() {
        let atoms = vec![
            Atom::new("O", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.757, 0.0, 0.586]),
            Atom::new("H", [-0.757, 0.0, 0.586]),
        ];

        let dim = 3 * atoms.len();
        // Arbitrary displacement mode vector (e.g. asymmetric stretch + bend)
        let mut v = vec![0.0; dim];
        v[0] = 0.12; // O_x
        v[2] = -0.05; // O_z
        v[3] = 0.58; // H1_x
        v[5] = 0.42; // H1_z
        v[6] = -0.35; // H2_x
        v[8] = 0.28; // H2_z

        let d4_anal = compute_nuclear_repulsion_fourth_directional(&atoms, &v);

        // 5-point finite difference formula of nuclear repulsion energy:
        // f^(4)(0) = (-f(2h) + 16 f(h) - 30 f(0) + 16 f(-h) - f(-2h)) / (12 h^4)
        let eval_vnuc = |step: f64| -> f64 {
            let mut a_copy = atoms.clone();
            for a in 0..atoms.len() {
                for comp in 0..3 {
                    a_copy[a].position[comp] += step * v[3 * a + comp];
                }
            }
            crate::scf::compute_nuclear_repulsion(&a_copy)
        };

        let h1 = 0.005;
        let v_p2 = eval_vnuc(2.0 * h1);
        let v_p1 = eval_vnuc(h1);
        let v_0 = eval_vnuc(0.0);
        let v_m1 = eval_vnuc(-h1);
        let v_m2 = eval_vnuc(-2.0 * h1);
        let d4_h1 = (v_p2 - 4.0 * v_p1 + 6.0 * v_0 - 4.0 * v_m1 + v_m2) / (h1 * h1 * h1 * h1);

        let h2 = 0.01;
        let v2_p2 = eval_vnuc(2.0 * h2);
        let v2_p1 = eval_vnuc(h2);
        let v2_m1 = eval_vnuc(-h2);
        let v2_m2 = eval_vnuc(-2.0 * h2);
        let d4_h2 = (v2_p2 - 4.0 * v2_p1 + 6.0 * v_0 - 4.0 * v2_m1 + v2_m2) / (h2 * h2 * h2 * h2);

        // Richardson extrapolation: D_extrap = (4 * D(h) - D(2h)) / 3 cancels O(h^2) error:
        let d4_richardson = (4.0 * d4_h1 - d4_h2) / 3.0;
        let diff = (d4_anal - d4_richardson).abs();

        eprintln!(
            "Nuclear 4th directional: Anal = {:16.8}, Num = {:16.8}, Diff = {:10.2e}",
            d4_anal, d4_richardson, diff
        );
        assert!(
            diff < 1e-4,
            "Exact algebraic 4th derivative of nuclear repulsion must match numerical limit to < 1e-4: diff = {}",
            diff
        );
    }
}
