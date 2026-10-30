// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Molecular property evaluations:
//! - Mulliken and Löwdin atomic population analyses
//! - Electric dipole moment vector (a.u. and Debye)
//! - Electric quadrupole moment tensor (traceless, a.u. and Buckingham)

use crate::basis::{Atom, ContractedGaussian};
use crate::integrals::primitive_overlap;
use crate::matrix::Matrix;

pub const AU_TO_DEBYE: f64 = 2.54174623;
pub const AU_TO_BUCKINGHAM: f64 = 1.345034;

/// Result of molecular population analysis and electric multipole moment calculations.
#[derive(Clone, Debug)]
pub struct MolecularProperties {
    pub mulliken_charges: Vec<f64>,
    pub lowdin_charges: Vec<f64>,
    /// Dipole moment vector [mu_x, mu_y, mu_z] in atomic units (e * a0)
    pub dipole_moment_au: [f64; 3],
    /// Dipole moment vector [mu_x, mu_y, mu_z] in Debye
    pub dipole_moment_debye: [f64; 3],
    /// Total dipole moment magnitude in atomic units
    pub dipole_magnitude_au: f64,
    /// Total dipole moment magnitude in Debye
    pub dipole_magnitude_debye: f64,
    /// Traceless electric quadrupole moment tensor [3 x 3] in atomic units
    pub quadrupole_moment_au: [[f64; 3]; 3],
    /// Traceless electric quadrupole moment tensor [3 x 3] in Buckingham
    pub quadrupole_moment_buckingham: [[f64; 3]; 3],
}

/// Evaluates 1-electron dipole operator matrix elements: <mu | r_comp | nu>
/// where r_comp is 0 for x, 1 for y, 2 for z.
pub fn compute_dipole_matrix(basis: &[ContractedGaussian], component: usize) -> Matrix {
    let n = basis.len();
    let mut mat = Matrix::zeros(n, n);

    for i in 0..n {
        let cg_i = &basis[i];
        let mut la_shift = cg_i.l;
        la_shift[component] += 1;
        let a_center = cg_i.center;

        for j in 0..=i {
            let cg_j = &basis[j];
            let b_center = cg_j.center;
            let lb = cg_j.l;

            let mut val = 0.0;
            for p1 in &cg_i.primitives {
                for p2 in &cg_j.primitives {
                    let c_prod = p1.coefficient * p2.coefficient;
                    let s_shifted = primitive_overlap(p1.exponent, la_shift, a_center, p2.exponent, lb, b_center);
                    let s_orig = primitive_overlap(p1.exponent, cg_i.l, a_center, p2.exponent, lb, b_center);
                    val += c_prod * (s_shifted + a_center[component] * s_orig);
                }
            }

            mat[(i, j)] = val;
            mat[(j, i)] = val;
        }
    }

    mat
}

/// Evaluates 2nd moment matrix elements: <mu | r_alpha * r_beta | nu>
pub fn compute_second_moment_matrix(basis: &[ContractedGaussian], comp_a: usize, comp_b: usize) -> Matrix {
    let n = basis.len();
    let mut mat = Matrix::zeros(n, n);

    for i in 0..n {
        let cg_i = &basis[i];
        let a_center = cg_i.center;
        let mut la_both = cg_i.l;
        la_both[comp_a] += 1;
        la_both[comp_b] += 1;

        let mut la_a = cg_i.l;
        la_a[comp_a] += 1;

        let mut la_b = cg_i.l;
        la_b[comp_b] += 1;

        for j in 0..=i {
            let cg_j = &basis[j];
            let b_center = cg_j.center;
            let lb = cg_j.l;

            let mut val = 0.0;
            for p1 in &cg_i.primitives {
                for p2 in &cg_j.primitives {
                    let c_prod = p1.coefficient * p2.coefficient;
                    let s_both = primitive_overlap(p1.exponent, la_both, a_center, p2.exponent, lb, b_center);
                    let s_a = primitive_overlap(p1.exponent, la_a, a_center, p2.exponent, lb, b_center);
                    let s_b = primitive_overlap(p1.exponent, la_b, a_center, p2.exponent, lb, b_center);
                    let s_orig = primitive_overlap(p1.exponent, cg_i.l, a_center, p2.exponent, lb, b_center);

                    let elem = s_both
                        + a_center[comp_b] * s_a
                        + a_center[comp_a] * s_b
                        + a_center[comp_a] * a_center[comp_b] * s_orig;
                    val += c_prod * elem;
                }
            }

            mat[(i, j)] = val;
            mat[(j, i)] = val;
        }
    }

    mat
}

/// Computes Mulliken atomic charges: Q_A = Z_A - \sum_{mu \in A} (P * S)_{mu, mu}
pub fn compute_mulliken_charges(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    density: &Matrix,
    overlap: &Matrix,
) -> Vec<f64> {
    let ps = density.dot(overlap);
    let mut charges = Vec::with_capacity(atoms.len());

    for (atom_idx, atom) in atoms.iter().enumerate() {
        let mut pop = 0.0;
        for (mu, cg) in basis.iter().enumerate() {
            if cg.atom_index == atom_idx {
                pop += ps[(mu, mu)];
            }
        }
        charges.push(atom.charge - pop);
    }

    charges
}

/// Computes Löwdin atomic charges: Q_A = Z_A - \sum_{mu \in A} (S^(1/2) * P * S^(1/2))_{mu, mu}
pub fn compute_lowdin_charges(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    density: &Matrix,
    overlap: &Matrix,
) -> Vec<f64> {
    // S^(1/2) = (S^(-1/2))^(-1) or from spectral decomposition S = U * Lambda * U^T => S^(1/2) = U * Lambda^(1/2) * U^T
    let (evals, evecs) = overlap.eigh();
    let n = overlap.rows;
    let mut s_half = Matrix::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                let lambda_sqrt = evals[k].max(1e-15).sqrt();
                sum += evecs[(i, k)] * lambda_sqrt * evecs[(j, k)];
            }
            s_half[(i, j)] = sum;
        }
    }

    let p_lowdin = s_half.dot(density).dot(&s_half);
    let mut charges = Vec::with_capacity(atoms.len());

    for (atom_idx, atom) in atoms.iter().enumerate() {
        let mut pop = 0.0;
        for (mu, cg) in basis.iter().enumerate() {
            if cg.atom_index == atom_idx {
                pop += p_lowdin[(mu, mu)];
            }
        }
        charges.push(atom.charge - pop);
    }

    charges
}

/// Computes molecular properties: Mulliken & Löwdin charges, dipole moment, and quadrupole moment tensor.
pub fn compute_molecular_properties(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    density: &Matrix,
    overlap: &Matrix,
) -> MolecularProperties {
    let mulliken = compute_mulliken_charges(atoms, basis, density, overlap);
    let lowdin = compute_lowdin_charges(atoms, basis, density, overlap);

    // Dipole moment: mu_alpha = \sum_A Z_A * R_{A, alpha} - Tr(P * D_alpha)
    let dx = compute_dipole_matrix(basis, 0);
    let dy = compute_dipole_matrix(basis, 1);
    let dz = compute_dipole_matrix(basis, 2);

    let mut mu_au = [0.0; 3];
    for comp in 0..3 {
        let nuc: f64 = atoms.iter().map(|a| a.charge * a.position[comp]).sum();
        let mat = match comp {
            0 => &dx,
            1 => &dy,
            _ => &dz,
        };
        let mut elec = 0.0;
        for i in 0..density.rows {
            for j in 0..density.cols {
                elec += density[(i, j)] * mat[(i, j)];
            }
        }
        mu_au[comp] = nuc - elec;
    }

    let mu_debye = [
        mu_au[0] * AU_TO_DEBYE,
        mu_au[1] * AU_TO_DEBYE,
        mu_au[2] * AU_TO_DEBYE,
    ];
    let mag_au = (mu_au[0].powi(2) + mu_au[1].powi(2) + mu_au[2].powi(2)).sqrt();
    let mag_debye = mag_au * AU_TO_DEBYE;

    // Second moment matrix M_{alpha, beta} = \sum_A Z_A * R_{A, alpha} * R_{A, beta} - Tr(P * Q_{alpha, beta})
    let mut m_2nd = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..=a {
            let q_mat = compute_second_moment_matrix(basis, a, b);
            let nuc: f64 = atoms.iter().map(|at| at.charge * at.position[a] * at.position[b]).sum();
            let mut elec = 0.0;
            for i in 0..density.rows {
                for j in 0..density.cols {
                    elec += density[(i, j)] * q_mat[(i, j)];
                }
            }
            let val = nuc - elec;
            m_2nd[a][b] = val;
            m_2nd[b][a] = val;
        }
    }

    // Traceless quadrupole moment: Theta_{alpha, beta} = 0.5 * (3 * M_{alpha, beta} - delta_{alpha, beta} * Tr(M))
    let tr_m = m_2nd[0][0] + m_2nd[1][1] + m_2nd[2][2];
    let mut quad_au = [[0.0; 3]; 3];
    let mut quad_buck = [[0.0; 3]; 3];

    for a in 0..3 {
        for b in 0..3 {
            let delta = if a == b { 1.0 } else { 0.0 };
            let val = 0.5 * (3.0 * m_2nd[a][b] - delta * tr_m);
            quad_au[a][b] = val;
            quad_buck[a][b] = val * AU_TO_BUCKINGHAM;
        }
    }

    MolecularProperties {
        mulliken_charges: mulliken,
        lowdin_charges: lowdin,
        dipole_moment_au: mu_au,
        dipole_moment_debye: mu_debye,
        dipole_magnitude_au: mag_au,
        dipole_magnitude_debye: mag_debye,
        quadrupole_moment_au: quad_au,
        quadrupole_moment_buckingham: quad_buck,
    }
}
