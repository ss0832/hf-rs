// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Analytical RHF/UHF nuclear derivatives through fourth order.
//!
//! Square-free Taylor coefficients of D S D = D and F D S = S D F are solved
//! order by order using one coupled-perturbed HF factorization. Coefficients are
//! actual mixed derivatives, including repeated physical directions. There are
//! no displaced geometries, difference stencils, or numerical step sizes.

use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::basis::{Atom, BasisSetMap, ContractedGaussian, build_basis};
use crate::derivatives::{
    GeneralBasisDerivative, diff_cg_directions, eval_eri_deriv, eval_kinetic_deriv,
    eval_nuclear_deriv_single_atom, eval_overlap_deriv,
};
use crate::error::HfError;
use crate::matrix::{LUFactorization, Matrix};
use crate::scf::{
    HFResult, UHFResult, build_fock_matrix, build_uhf_fock_matrices, run_scf_with_charge, run_uhf,
};

/// One nuclear displacement vector per atom, in Cartesian coordinates.
pub type NuclearDirection = Vec<[f64; 3]>;

fn add_scaled(a: &mut Matrix, b: &Matrix, scale: f64) {
    for (x, y) in a.data.iter_mut().zip(&b.data) {
        *x += scale * y;
    }
}

fn subsets(mask: usize) -> impl Iterator<Item = usize> {
    std::iter::successors(
        Some(mask),
        move |&s| if s == 0 { None } else { Some((s - 1) & mask) },
    )
}

fn relative(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| a[k] - b[k])
}

fn matrix_trace_product(a: &Matrix, b: &Matrix) -> f64 {
    (0..a.rows)
        .map(|i| (0..a.cols).map(|j| a[(i, j)] * b[(j, i)]).sum::<f64>())
        .sum()
}

fn selected_directions(directions: &[NuclearDirection], mask: usize) -> Vec<NuclearDirection> {
    directions
        .iter()
        .enumerate()
        .filter(|(i, _)| mask & (1 << i) != 0)
        .map(|(_, v)| v.clone())
        .collect()
}

/// Overlap and core Hamiltonian derivative, including all ECP-center terms.
fn one_electron_derivative(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    directions: &[NuclearDirection],
) -> Result<(Matrix, Matrix), HfError> {
    let n = basis.len();
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
    let values: Vec<(f64, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let a = &basis[i];
            let b = &basis[j];
            let b0 = GeneralBasisDerivative::from_single(b.clone());
            // Overlap and kinetic energy depend only on A-B.
            let ds: Vec<_> = directions
                .iter()
                .map(|v| relative(v[a.atom_index], v[b.atom_index]))
                .collect();
            let da = diff_cg_directions(a, &ds);
            let s = eval_overlap_deriv(&da, &b0);
            let mut h = eval_kinetic_deriv(&da, &b0);
            for (c, atom) in atoms.iter().enumerate() {
                for mask in 0..1usize << directions.len() {
                    let left: Vec<_> = directions
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| mask & (1 << k) != 0)
                        .map(|(_, v)| relative(v[a.atom_index], v[c]))
                        .collect();
                    let right: Vec<_> = directions
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| mask & (1 << k) == 0)
                        .map(|(_, v)| relative(v[b.atom_index], v[c]))
                        .collect();
                    let da = diff_cg_directions(a, &left);
                    let db = diff_cg_directions(b, &right);
                    h += eval_nuclear_deriv_single_atom(&da, &db, atom);
                }
            }
            (s, h)
        })
        .collect();
    let mut s = Matrix::zeros(n, n);
    let mut h = Matrix::zeros(n, n);
    for (&(i, j), &(sv, hv)) in pairs.iter().zip(&values) {
        s[(i, j)] = sv;
        s[(j, i)] = sv;
        h[(i, j)] = hv;
        h[(j, i)] = hv;
    }
    if crate::ecp::has_ecp(atoms) {
        let ecp = crate::ecp::ecp_nuclear_derivative_matrix(atoms, basis, directions)
            .map_err(HfError::Integral)?;
        add_scaled(&mut h, &ecp, 1.0);
    }
    Ok((s, h))
}

pub(crate) fn eri_directional(
    basis: &[ContractedGaussian],
    indices: [usize; 4],
    directions: &[NuclearDirection],
) -> f64 {
    let atoms: [usize; 4] = std::array::from_fn(|i| basis[indices[i]].atom_index);
    // Anchor the most frequent center to reduce the number of nonzero partitions.
    let anchor = (0..4)
        .max_by_key(|&i| atoms.iter().filter(|&&a| a == atoms[i]).count())
        .unwrap_or(3);
    let mut slots = indices;
    // ERIs are symmetric under exchanging the two pairs and members of a pair;
    // move the chosen anchor to slot 3 using only these symmetries.
    if anchor < 2 {
        slots.swap(0, 2);
        slots.swap(1, 3);
    }
    if basis[slots[3]].atom_index != atoms[anchor] {
        slots.swap(2, 3);
    }
    let anchor_atom = basis[slots[3]].atom_index;
    let mut functions: [Vec<[f64; 3]>; 3] = std::array::from_fn(|_| Vec::new());
    fn recurse(
        basis: &[ContractedGaussian],
        slots: [usize; 4],
        directions: &[NuclearDirection],
        anchor: usize,
        depth: usize,
        functions: &mut [Vec<[f64; 3]>; 3],
    ) -> f64 {
        if depth == directions.len() {
            let a = diff_cg_directions(&basis[slots[0]], &functions[0]);
            let b = diff_cg_directions(&basis[slots[1]], &functions[1]);
            let c = diff_cg_directions(&basis[slots[2]], &functions[2]);
            let d = GeneralBasisDerivative::from_single(basis[slots[3]].clone());
            return eval_eri_deriv(&a, &b, &c, &d);
        }
        let mut value = 0.0;
        for slot in 0..3 {
            let v = relative(
                directions[depth][basis[slots[slot]].atom_index],
                directions[depth][anchor],
            );
            if v == [0.0; 3] {
                continue;
            }
            functions[slot].push(v);
            value += recurse(basis, slots, directions, anchor, depth + 1, functions);
            functions[slot].pop();
        }
        value
    }
    recurse(basis, slots, directions, anchor_atom, 0, &mut functions)
}

fn eri_derivative(basis: &[ContractedGaussian], directions: &[NuclearDirection]) -> Vec<f64> {
    let n = basis.len();
    let pairs: Vec<_> = (0..n).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();
    let np = pairs.len();
    let mut out = vec![0.0; np * (np + 1) / 2];
    // Fill the final packed tensor directly: no duplicate row-results tensor.
    let mut tail = out.as_mut_slice();
    let mut rows = Vec::with_capacity(np);
    for i in 0..np {
        let (row, rest) = tail.split_at_mut(i + 1);
        rows.push(row);
        tail = rest;
    }
    rows.into_par_iter().enumerate().for_each(|(ij, row)| {
        let (i, j) = pairs[ij];
        for (kl, value) in row.iter_mut().enumerate() {
            let (k, l) = pairs[kl];
            *value = eri_directional(basis, [i, j, k, l], directions);
        }
    });
    out
}

fn scalar_product(a: &[f64], b: &[f64], mask: usize) -> f64 {
    subsets(mask).map(|s| a[s] * b[mask ^ s]).sum()
}

fn nuclear_derivative(atoms: &[Atom], directions: &[NuclearDirection]) -> f64 {
    let size = 1 << directions.len();
    let full = size - 1;
    let mut total = 0.0;
    for a in 0..atoms.len() {
        for b in 0..a {
            let mut r2 = vec![0.0; size];
            for axis in 0..3 {
                let mut r = vec![0.0; size];
                r[0] = atoms[a].position[axis] - atoms[b].position[axis];
                for (i, v) in directions.iter().enumerate() {
                    r[1 << i] = v[a][axis] - v[b][axis];
                }
                for (mask, value) in r2.iter_mut().enumerate() {
                    *value += scalar_product(&r, &r, mask);
                }
            }
            let base = r2[0];
            r2[0] = 0.0;
            let mut power = vec![0.0; size];
            power[0] = 1.0;
            let mut inverse = vec![0.0; size];
            let mut coefficient = base.sqrt().recip();
            for order in 0..=directions.len() {
                for (x, p) in inverse.iter_mut().zip(&power) {
                    *x += coefficient * p;
                }
                power = (0..size).map(|m| scalar_product(&power, &r2, m)).collect();
                coefficient *= (-0.5 - order as f64) / ((order + 1) as f64 * base);
            }
            total += atoms[a].charge * atoms[b].charge * inverse[full];
        }
    }
    total
}

struct Spin<'a> {
    c: &'a Matrix,
    ct: Matrix,
    eps: &'a [f64],
    n_occ: usize,
    d0_mo: Matrix,
    d0_ao: Matrix,
}

impl<'a> Spin<'a> {
    fn new(c: &'a Matrix, eps: &'a [f64], n_occ: usize) -> Self {
        let mut d0_mo = Matrix::zeros(c.cols, c.cols);
        for i in 0..n_occ {
            d0_mo[(i, i)] = 1.0;
        }
        Self {
            c,
            ct: c.transpose(),
            eps,
            n_occ,
            d0_mo,
            d0_ao: c.occupied_density(n_occ, 1.0),
        }
    }
    fn to_mo(&self, m: &Matrix) -> Matrix {
        self.ct.dot(m).dot(self.c)
    }
    fn to_ao(&self, d: &Matrix) -> Matrix {
        self.c.dot(d).dot(&self.ct)
    }
}

/// Reusable reference and SCF-response factorization for analytical derivatives.
/// Derivative ERIs use a 64 MiB LRU cache; the reference ERIs are borrowed.
/// At most one additional packed derivative ERI tensor is generated at a time.
pub struct HfNuclearDerivativeContext<'a> {
    atoms: &'a [Atom],
    basis: &'a [ContractedGaussian],
    s0: &'a Matrix,
    h0: &'a Matrix,
    eri0: &'a [f64],
    e0: f64,
    spins: Vec<Spin<'a>>,
    variables: Vec<(usize, usize, usize)>,
    response: Option<LUFactorization>,
    one_electron_cache: Mutex<OneElectronCache>,
}

impl<'a> HfNuclearDerivativeContext<'a> {
    pub fn from_rhf(
        atoms: &'a [Atom],
        basis: &'a [ContractedGaussian],
        hf: &'a HFResult,
    ) -> Result<Self, HfError> {
        if !hf.converged {
            return Err(HfError::Scf("reference did not converge".into()));
        }
        if let Some(space) = &hf.ao_space {
            space.ensure_differentiable()?;
        } else if hf.mo_coefficients.cols < basis.len() {
            return Err(HfError::LinearDependence {
                n_ao: basis.len(),
                n_mo: hf.mo_coefficients.cols,
            });
        }
        Self::new(
            atoms,
            basis,
            &hf.overlap_matrix,
            &hf.core_hamiltonian,
            &hf.eri,
            hf.total_energy,
            vec![Spin::new(
                &hf.mo_coefficients,
                &hf.orbital_energies,
                hf.n_occ,
            )],
        )
    }
    pub fn from_uhf(
        atoms: &'a [Atom],
        basis: &'a [ContractedGaussian],
        hf: &'a UHFResult,
    ) -> Result<Self, HfError> {
        if !hf.converged {
            return Err(HfError::Scf("reference did not converge".into()));
        }
        if let Some(space) = &hf.ao_space {
            space.ensure_differentiable()?;
        } else if hf.mo_coefficients_alpha.cols < basis.len() {
            return Err(HfError::LinearDependence {
                n_ao: basis.len(),
                n_mo: hf.mo_coefficients_alpha.cols,
            });
        }
        Self::new(
            atoms,
            basis,
            &hf.overlap_matrix,
            &hf.core_hamiltonian,
            &hf.eri,
            hf.total_energy,
            vec![
                Spin::new(
                    &hf.mo_coefficients_alpha,
                    &hf.orbital_energies_alpha,
                    hf.n_alpha,
                ),
                Spin::new(
                    &hf.mo_coefficients_beta,
                    &hf.orbital_energies_beta,
                    hf.n_beta,
                ),
            ],
        )
    }
    fn new(
        atoms: &'a [Atom],
        basis: &'a [ContractedGaussian],
        s0: &'a Matrix,
        h0: &'a Matrix,
        eri0: &'a [f64],
        e0: f64,
        spins: Vec<Spin<'a>>,
    ) -> Result<Self, HfError> {
        let n = basis.len();
        if n == 0 || atoms.is_empty() {
            return Err(HfError::InvalidInput(
                "a nonempty molecular basis is required".into(),
            ));
        }
        if basis.iter().any(|cg| cg.atom_index >= atoms.len()) {
            return Err(HfError::InvalidInput(
                "basis atom index out of range".into(),
            ));
        }
        for spin in &spins {
            if spin.c.rows != n || spin.c.cols != spin.eps.len() || spin.n_occ > spin.c.cols {
                return Err(HfError::InvalidInput(
                    "inconsistent reference orbital dimensions".into(),
                ));
            }
        }
        let variables = spins
            .iter()
            .enumerate()
            .flat_map(|(sigma, s)| {
                (s.n_occ..s.c.cols).flat_map(move |a| (0..s.n_occ).map(move |i| (sigma, a, i)))
            })
            .collect();
        let mut out = Self {
            atoms,
            basis,
            s0,
            h0,
            eri0,
            e0,
            spins,
            variables,
            response: None,
            one_electron_cache: Mutex::new(OneElectronCache::default()),
        };
        let nv = out.variables.len();
        if nv != 0 {
            let zero = Matrix::zeros(n, n);
            let mut a = Matrix::zeros(nv, nv);
            for (column, &(sigma, v, i)) in out.variables.iter().enumerate() {
                let mut dm = Matrix::zeros(out.spins[sigma].c.cols, out.spins[sigma].c.cols);
                dm[(v, i)] = 1.0;
                dm[(i, v)] = 1.0;
                let mut da = vec![zero.clone(); out.spins.len()];
                da[sigma] = out.spins[sigma].to_ao(&dm);
                let response = out.fock(&zero, &da, eri0);
                let response_mo: Vec<_> = out
                    .spins
                    .iter()
                    .zip(&response)
                    .map(|(s, f)| s.to_mo(f))
                    .collect();
                for (row, &(spin, b, j)) in out.variables.iter().enumerate() {
                    a[(row, column)] = response_mo[spin][(b, j)];
                    if row == column {
                        a[(row, column)] += out.spins[spin].eps[b] - out.spins[spin].eps[j];
                    }
                }
            }
            out.response =
                Some(a.lu_decompose().ok_or_else(|| {
                    HfError::Response("singular coupled-perturbed HF matrix".into())
                })?);
        }
        Ok(out)
    }
    fn fock(&self, h: &Matrix, densities: &[Matrix], eri: &[f64]) -> Vec<Matrix> {
        let n = self.basis.len();
        if self.spins.len() == 1 {
            vec![build_fock_matrix(h, &densities[0].scale(2.0), eri, n)]
        } else {
            let (a, b) = build_uhf_fock_matrices(h, &densities[0], &densities[1], eri, n);
            vec![a, b]
        }
    }

    fn one_electron(
        &self,
        directions: &[NuclearDirection],
    ) -> Result<Arc<(Matrix, Matrix)>, HfError> {
        let mut key: DerivativeKey = directions
            .iter()
            .map(|v| {
                v.iter()
                    .flatten()
                    .map(|&x| if x == 0.0 { 0 } else { x.to_bits() })
                    .collect()
            })
            .collect();
        key.sort();
        if let Some(value) = self
            .one_electron_cache
            .lock()
            .map_err(|_| HfError::Integral("one-electron derivative cache lock poisoned".into()))?
            .get(&key)
        {
            return Ok(value);
        }
        // Do not hold the cache lock across parallel quadrature/integral work.
        let value = Arc::new(one_electron_derivative(self.atoms, self.basis, directions)?);
        Ok(self
            .one_electron_cache
            .lock()
            .map_err(|_| HfError::Integral("one-electron derivative cache lock poisoned".into()))?
            .insert(key, value))
    }

    /// Actual mixed derivative, of order zero through four, at the reference geometry.
    pub fn derivative(&self, directions: &[NuclearDirection]) -> Result<f64, HfError> {
        if directions.len() > 4
            || directions
                .iter()
                .any(|v| v.len() != self.atoms.len() || v.iter().flatten().any(|x| !x.is_finite()))
        {
            return Err(HfError::InvalidInput(
                "expected at most four finite nuclear directions".into(),
            ));
        }
        if directions.is_empty() {
            return Ok(self.e0);
        }
        if directions.iter().any(|v| v.iter().all(|&x| x == v[0])) {
            // Any rigid translation is an exact zero derivative, for every mixed order.
            return Ok(0.0);
        }
        let n = self.basis.len();
        let size = 1usize << directions.len();
        let full = size - 1;
        let mut representatives = vec![0; size];
        let mut classes: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut groups = Vec::<&NuclearDirection>::new();
        let ids: Vec<_> = directions
            .iter()
            .map(|v| {
                if let Some(i) = groups.iter().position(|&p| p == v) {
                    i
                } else {
                    groups.push(v);
                    groups.len() - 1
                }
            })
            .collect();
        for (mask, rep) in representatives.iter_mut().enumerate() {
            let mut key = vec![0; groups.len()];
            for (i, &group) in ids.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    key[group] += 1;
                }
            }
            *rep = *classes.entry(key).or_insert(mask);
        }
        let mut s = vec![self.s0.clone()];
        let mut h = vec![self.h0.clone()];
        for mask in 1..size {
            if representatives[mask] != mask {
                s.push(s[representatives[mask]].clone());
                h.push(h[representatives[mask]].clone());
            } else {
                let dirs = selected_directions(directions, mask);
                let integrals = self.one_electron(&dirs)?;
                s.push(integrals.0.clone());
                h.push(integrals.1.clone());
            }
        }
        let s_mo: Vec<Vec<_>> = self
            .spins
            .iter()
            .map(|spin| s.iter().map(|sm| spin.to_mo(sm)).collect())
            .collect();
        let mut d_mo = self
            .spins
            .iter()
            .map(|s| vec![Matrix::zeros(s.c.cols, s.c.cols); size])
            .collect::<Vec<_>>();
        let mut d_ao = vec![vec![Matrix::zeros(n, n); size]; self.spins.len()];
        let mut f_ao = d_ao.clone();
        let mut f_mo = d_mo.clone();
        for (sigma, spin) in self.spins.iter().enumerate() {
            d_mo[sigma][0] = spin.d0_mo.clone();
            d_ao[sigma][0] = spin.d0_ao.clone();
        }
        let initial: Vec<_> = d_ao.iter().map(|d| d[0].clone()).collect();
        let f0 = self.fock(self.h0, &initial, self.eri0);
        for sigma in 0..self.spins.len() {
            f_ao[sigma][0] = f0[sigma].clone();
            f_mo[sigma][0] = self.spins[sigma].to_mo(&f0[sigma]);
        }
        let mut eri_cache = EriDerivativeCache::new(self.basis, directions, &representatives);
        let zero = Matrix::zeros(n, n);
        for mask in 1..size {
            let rep = representatives[mask];
            if rep != mask {
                for sigma in 0..self.spins.len() {
                    d_mo[sigma][mask] = d_mo[sigma][rep].clone();
                    d_ao[sigma][mask] = d_ao[sigma][rep].clone();
                    f_mo[sigma][mask] = f_mo[sigma][rep].clone();
                    f_ao[sigma][mask] = f_ao[sigma][rep].clone();
                }
                continue;
            }
            // Known normalization terms determine the oo and vv density blocks.
            for (sigma, spin) in self.spins.iter().enumerate() {
                let mut t = triple_coefficient(&d_mo[sigma], &s_mo[sigma], &d_mo[sigma], mask);
                for i in 0..spin.c.cols {
                    for j in 0..spin.c.cols {
                        t[(i, j)] = if i < spin.n_occ && j < spin.n_occ {
                            -t[(i, j)]
                        } else if i >= spin.n_occ && j >= spin.n_occ {
                            t[(i, j)]
                        } else {
                            0.0
                        };
                    }
                }
                d_mo[sigma][mask] = t;
                d_ao[sigma][mask] = spin.to_ao(&d_mo[sigma][mask]);
                f_ao[sigma][mask] = h[mask].clone();
            }
            // F_mask = h_mask + sum_s G(ERI_s,D_(mask\s)).
            for sub in subsets(mask) {
                let ds: Vec<_> = d_ao.iter().map(|d| d[mask ^ sub].clone()).collect();
                let contribution = if sub == 0 {
                    self.fock(&zero, &ds, self.eri0)
                } else {
                    let eri = eri_cache.get(sub);
                    self.fock(&zero, &ds, &eri)
                };
                for (sigma, f) in contribution.iter().enumerate() {
                    add_scaled(&mut f_ao[sigma][mask], f, 1.0);
                }
            }
            for (sigma, spin) in self.spins.iter().enumerate() {
                f_mo[sigma][mask] = spin.to_mo(&f_ao[sigma][mask]);
            }
            let mut rhs = vec![0.0; self.variables.len()];
            let residual: Vec<_> = (0..self.spins.len())
                .map(|sigma| {
                    let mut g = triple_coefficient(&f_mo[sigma], &d_mo[sigma], &s_mo[sigma], mask);
                    let other = triple_coefficient(&s_mo[sigma], &d_mo[sigma], &f_mo[sigma], mask);
                    add_scaled(&mut g, &other, -1.0);
                    g
                })
                .collect();
            for (row, &(sigma, a, i)) in self.variables.iter().enumerate() {
                rhs[row] = -residual[sigma][(a, i)];
            }
            if let Some(lu) = &self.response {
                let solution = lu.solve(&rhs);
                if solution.iter().any(|x| !x.is_finite()) {
                    return Err(HfError::Response("non-finite density response".into()));
                }
                let mut delta = self
                    .spins
                    .iter()
                    .map(|s| Matrix::zeros(s.c.cols, s.c.cols))
                    .collect::<Vec<_>>();
                for (&(sigma, a, i), &x) in self.variables.iter().zip(&solution) {
                    delta[sigma][(a, i)] = x;
                    delta[sigma][(i, a)] = x;
                }
                let delta_ao: Vec<_> = self
                    .spins
                    .iter()
                    .zip(&delta)
                    .map(|(s, d)| s.to_ao(d))
                    .collect();
                let df = self.fock(&zero, &delta_ao, self.eri0);
                for (sigma, spin) in self.spins.iter().enumerate() {
                    add_scaled(&mut d_mo[sigma][mask], &delta[sigma], 1.0);
                    add_scaled(&mut d_ao[sigma][mask], &delta_ao[sigma], 1.0);
                    add_scaled(&mut f_ao[sigma][mask], &df[sigma], 1.0);
                    add_scaled(&mut f_mo[sigma][mask], &spin.to_mo(&df[sigma]), 1.0);
                }
            }
        }
        let degeneracy = if self.spins.len() == 1 { 1.0 } else { 0.5 };
        let mut energy = nuclear_derivative(self.atoms, directions);
        for sigma in 0..self.spins.len() {
            for sub in subsets(full) {
                energy += degeneracy
                    * (matrix_trace_product(&d_ao[sigma][sub], &h[full ^ sub])
                        + matrix_trace_product(&d_ao[sigma][sub], &f_ao[sigma][full ^ sub]));
            }
        }
        if energy.is_finite() {
            Ok(energy)
        } else {
            Err(HfError::Response("non-finite energy derivative".into()))
        }
    }

    pub fn coordinate_derivative(&self, coordinates: &[usize]) -> Result<f64, HfError> {
        if coordinates.iter().any(|&i| i >= 3 * self.atoms.len()) {
            return Err(HfError::InvalidInput(
                "nuclear coordinate out of range".into(),
            ));
        }
        let directions: Vec<_> = coordinates
            .iter()
            .map(|&i| {
                let mut v = vec![[0.0; 3]; self.atoms.len()];
                v[i / 3][i % 3] = 1.0;
                v
            })
            .collect();
        self.derivative(&directions)
    }

    pub fn hessian(&self) -> Result<Matrix, HfError> {
        let dim = 3 * self.atoms.len();
        let mut h = Matrix::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..=i {
                let x = self.coordinate_derivative(&[i, j])?;
                h[(i, j)] = x;
                h[(j, i)] = x;
            }
        }
        Ok(h)
    }
}

fn triple_coefficient(a: &[Matrix], b: &[Matrix], c: &[Matrix], mask: usize) -> Matrix {
    let mut out = Matrix::zeros(a[0].rows, c[0].cols);
    for left in subsets(mask) {
        for middle in subsets(mask ^ left) {
            let right = mask ^ left ^ middle;
            add_scaled(&mut out, &a[left].dot(&b[middle]).dot(&c[right]), 1.0);
        }
    }
    out
}

type DerivativeKey = Vec<Vec<u64>>;

#[derive(Default)]
struct OneElectronCache {
    tensors: HashMap<DerivativeKey, Arc<(Matrix, Matrix)>>,
    order: VecDeque<DerivativeKey>,
    bytes: usize,
}

impl OneElectronCache {
    fn entry_bytes(key: &DerivativeKey, value: &(Matrix, Matrix)) -> usize {
        // Account for both key copies (map/LRU), their nested vectors, matrix data
        // and a conservative entry/Arc/table allowance. The cache has a hard
        // accounting budget rather than growing with the full nuclear tensor.
        8 * (value.0.data.len() + value.1.data.len())
            + 2 * key.iter().map(|v| 24 + 8 * v.len()).sum::<usize>()
            + 384
    }
    fn get(&mut self, key: &DerivativeKey) -> Option<Arc<(Matrix, Matrix)>> {
        let value = self.tensors.get(key)?.clone();
        self.order.retain(|x| x != key);
        self.order.push_back(key.clone());
        Some(value)
    }
    fn insert(
        &mut self,
        key: DerivativeKey,
        value: Arc<(Matrix, Matrix)>,
    ) -> Arc<(Matrix, Matrix)> {
        const BUDGET: usize = 8 * 1024 * 1024;
        if let Some(existing) = self.get(&key) {
            return existing;
        }
        let bytes = Self::entry_bytes(&key, &value);
        if bytes <= BUDGET {
            while self.bytes + bytes > BUDGET {
                let Some(old) = self.order.pop_front() else {
                    break;
                };
                if let Some(value) = self.tensors.remove(&old) {
                    self.bytes -= Self::entry_bytes(&old, &value);
                }
            }
            self.bytes += bytes;
            self.order.push_back(key.clone());
            self.tensors.insert(key, value.clone());
        }
        value
    }
}

struct EriDerivativeCache<'a> {
    basis: &'a [ContractedGaussian],
    directions: &'a [NuclearDirection],
    representatives: &'a [usize],
    tensors: HashMap<usize, Arc<Vec<f64>>>,
    order: VecDeque<usize>,
    bytes: usize,
}

impl<'a> EriDerivativeCache<'a> {
    fn new(
        basis: &'a [ContractedGaussian],
        directions: &'a [NuclearDirection],
        representatives: &'a [usize],
    ) -> Self {
        Self {
            basis,
            directions,
            representatives,
            tensors: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
        }
    }
    fn get(&mut self, mask: usize) -> Arc<Vec<f64>> {
        const BUDGET: usize = 64 * 1024 * 1024;
        let key = self.representatives[mask];
        if let Some(tensor) = self.tensors.get(&key) {
            self.order.retain(|&x| x != key);
            self.order.push_back(key);
            return tensor.clone();
        }
        let directions = selected_directions(self.directions, key);
        let tensor = Arc::new(eri_derivative(self.basis, &directions));
        let bytes = tensor.len() * std::mem::size_of::<f64>();
        if bytes <= BUDGET {
            while self.bytes + bytes > BUDGET {
                let Some(old) = self.order.pop_front() else {
                    break;
                };
                if let Some(t) = self.tensors.remove(&old) {
                    self.bytes -= t.len() * std::mem::size_of::<f64>();
                }
            }
            self.bytes += bytes;
            self.order.push_back(key);
            self.tensors.insert(key, tensor.clone());
        }
        tensor
    }
}

pub(crate) fn with_reference<T>(
    atoms: &[Atom],
    basis_map: &BasisSetMap,
    charge: i32,
    multiplicity: usize,
    broken: bool,
    max_iter: usize,
    tol: f64,
    f: impl FnOnce(&HfNuclearDerivativeContext<'_>) -> Result<T, HfError>,
) -> Result<T, HfError> {
    let basis = build_basis(atoms, basis_map);
    if multiplicity > 1 || broken {
        let hf = run_uhf(atoms, &basis, charge, multiplicity, broken, max_iter, tol)
            .map_err(HfError::Scf)?;
        f(&HfNuclearDerivativeContext::from_uhf(atoms, &basis, &hf)?)
    } else {
        let hf = run_scf_with_charge(atoms, &basis, charge, max_iter, tol).map_err(HfError::Scf)?;
        f(&HfNuclearDerivativeContext::from_rhf(atoms, &basis, &hf)?)
    }
}

pub(crate) fn direction_from_flat(
    values: &[f64],
    n_atoms: usize,
) -> Result<NuclearDirection, HfError> {
    if values.len() != 3 * n_atoms || values.iter().any(|x| !x.is_finite()) {
        return Err(HfError::InvalidInput(
            "displacement vectors must contain 3N finite values".into(),
        ));
    }
    Ok(values.chunks_exact(3).map(|x| [x[0], x[1], x[2]]).collect())
}

pub(crate) fn permutations3(i: [usize; 3]) -> [[usize; 3]; 6] {
    [
        [i[0], i[1], i[2]],
        [i[0], i[2], i[1]],
        [i[1], i[0], i[2]],
        [i[1], i[2], i[0]],
        [i[2], i[0], i[1]],
        [i[2], i[1], i[0]],
    ]
}

pub(crate) fn permutations4(i: [usize; 4]) -> [[usize; 4]; 24] {
    let mut out = [[0; 4]; 24];
    let mut k = 0;
    for a in 0..4 {
        for b in 0..4 {
            if b == a {
                continue;
            }
            for c in 0..4 {
                if c == a || c == b {
                    continue;
                }
                let d = 6 - a - b - c;
                out[k] = [i[a], i[b], i[c], i[d]];
                k += 1;
            }
        }
    }
    out
}
