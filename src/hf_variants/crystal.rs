// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Crystal-orbital restricted Hartree-Fock for chains periodic along z.
//!
//! Real-space blocks `X(n)_mn = <chi_m(0)|X|chi_n(n a z)>` define the Bloch matrices
//! `X(k) = sum_n exp(i k n a) X(n)`; the closed-shell density blocks are
//! `P(n) = (1/N_k) sum_k Re[exp(-i k n a) 2 C_occ(k) C_occ(k)^H]` on a uniform k mesh, so that
//! `rho(r) = sum_{c,n} P(n)_mn chi_m(r - c a z) chi_n(r - (c + n) a z)`.
//!
//! Electrostatics are grouped by neutral cells: the electron density assigned to cell `c`
//! (first AO in cell `c`) together with the nuclei of cell `c`. Cells with `|c| <= D` are
//! summed explicitly with exact Gaussian integrals; all farther cells are included through a
//! Cartesian multipole expansion of both the source cell and the target charge (total order
//! `multipole_order`) with exact one-dimensional lattice sums (Hurwitz zeta functions). For a
//! neutral cell the far-field series is absolutely convergent (dipole-dipole terms decay as
//! `R^-3`). Exchange is summed in real space over overlap-screened pairs and the density range
//! of the k mesh. The energy per cell is
//!
//! `E = sum_n P(n).T(n) + 1/2 sum_n P(n).W(n) + 1/2 sum_A Z_A phi(R_A) - 1/4 sum_n P(n).K(n)`,
//!
//! with the total electrostatic potential matrix `W` and the potential `phi` of all cells at the
//! nuclei (self term excluded), and the Fock blocks are `F(n) = T(n) + W(n) - K(n)/2`.
//! Fixed occupations require a band gap; metallic chains are rejected. Scope: neutral,
//! closed-shell unit cells, all-electron (no ECP), periodicity along z, no nuclear gradients.

use super::complex_matrix::ComplexMatrix;
use crate::basis::{Atom, ContractedGaussian};
use crate::integrals::{
    contracted_eri, contracted_kinetic, contracted_nuclear_attraction, contracted_overlap,
    primitive_overlap,
};
use crate::matrix::Matrix;
use rayon::prelude::*;

/// Band energies at one k point.
#[derive(Clone, Debug)]
pub struct BandPoint {
    /// k in units of the reciprocal lattice vector, in (-1/2, 1/2].
    pub k_fractional: f64,
    /// k in 1/bohr.
    pub k_cartesian: f64,
    pub band_energies: Vec<f64>,
}

/// Controls for [`run_crystal_orbital_hf_with_options`].
#[derive(Clone, Debug)]
pub struct CoHfOptions {
    /// Number of uniform k points (Gamma-centered); at least `2 L_ov + 1`.
    pub n_kpoints: usize,
    /// Explicit near-field cells `D`; `None` chooses `max(2 L_ov + 3, ceil(20 bohr / a))`.
    pub near_field_cells: Option<usize>,
    /// Total Cartesian multipole order of the far-field expansion.
    pub multipole_order: usize,
    /// Overlap threshold that defines the AO pair range `L_ov`.
    pub overlap_threshold: f64,
    /// Largest real-space density/exchange range in cells (also limited by the k mesh).
    pub max_exchange_cells: usize,
    pub max_iterations: usize,
    pub energy_tol: f64,
    /// Largest element of the density residual `|P_out - P_in|`.
    pub density_tol: f64,
    pub diis_size: usize,
    /// Memory limit for the stored periodic electron-repulsion integrals.
    pub integral_budget_bytes: usize,
}

impl Default for CoHfOptions {
    fn default() -> Self {
        Self {
            n_kpoints: 16,
            near_field_cells: None,
            multipole_order: 8,
            overlap_threshold: 1e-12,
            max_exchange_cells: 12,
            max_iterations: 200,
            energy_tol: 1e-10,
            density_tol: 1e-9,
            diis_size: 8,
            integral_budget_bytes: 2 * 1024 * 1024 * 1024,
        }
    }
}

/// Result of a crystal-orbital Hartree-Fock calculation.
#[derive(Clone, Debug)]
pub struct COHFResult {
    pub lattice_constant_bohr: f64,
    pub n_kpoints: usize,
    /// Total energy per unit cell (Hartree).
    pub total_energy_per_cell: f64,
    pub kinetic_energy_per_cell: f64,
    /// Electrostatic energy per cell (electron-electron, electron-nuclear, nuclear-nuclear).
    pub electrostatic_energy_per_cell: f64,
    pub exchange_energy_per_cell: f64,
    pub band_structure: Vec<BandPoint>,
    pub valence_band_maximum: f64,
    pub conduction_band_minimum: f64,
    pub band_gap_hartree: f64,
    pub band_gap_ev: f64,
    /// Density blocks `P(n)` for `n = -range..=range`.
    pub density_blocks: Vec<Matrix>,
    pub density_range: usize,
    pub overlap_range: usize,
    pub near_field_cells: usize,
    pub electrons_per_cell: f64,
    pub density_residual: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Crystal-orbital HF with default controls except the k mesh and convergence.
pub fn run_crystal_orbital_hf(
    unit_cell_atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    lattice_constant_bohr: f64,
    n_kpoints: usize,
    max_iter: usize,
    tol: f64,
) -> Result<COHFResult, String> {
    run_crystal_orbital_hf_with_options(
        unit_cell_atoms,
        basis,
        charge,
        lattice_constant_bohr,
        &CoHfOptions {
            n_kpoints,
            max_iterations: max_iter,
            energy_tol: tol,
            density_tol: (tol.sqrt() * 0.1).clamp(1e-11, 1e-5),
            ..Default::default()
        },
    )
}

fn translated(basis: &[ContractedGaussian], shift: f64) -> Vec<ContractedGaussian> {
    basis
        .iter()
        .map(|g| {
            let mut t = g.clone();
            t.center[2] += shift;
            t
        })
        .collect()
}

fn binomial(n: usize, k: usize) -> f64 {
    (1..=k).fold(1.0, |v, j| v * (n + 1 - j) as f64 / j as f64)
}

/// `<a| x^p y^q z^r |b>` about the global origin.
fn moment(a: &ContractedGaussian, b: &ContractedGaussian, power: [usize; 3]) -> f64 {
    let mut value = 0.0;
    for pa in &a.primitives {
        for pb in &b.primitives {
            // x^p = sum_t C(p,t) (x - B_x)^t B_x^(p-t): raise b's angular powers.
            for t in 0..=power[0] {
                for u in 0..=power[1] {
                    for v in 0..=power[2] {
                        let factor = binomial(power[0], t)
                            * binomial(power[1], u)
                            * binomial(power[2], v)
                            * b.center[0].powi((power[0] - t) as i32)
                            * b.center[1].powi((power[1] - u) as i32)
                            * b.center[2].powi((power[2] - v) as i32);
                        if factor == 0.0 {
                            continue;
                        }
                        value += pa.coefficient
                            * pb.coefficient
                            * factor
                            * primitive_overlap(
                                pa.exponent,
                                a.l,
                                a.center,
                                pb.exponent,
                                [b.l[0] + t, b.l[1] + u, b.l[2] + v],
                                b.center,
                            );
                    }
                }
            }
        }
    }
    value
}

/// Multi-indices with total order `<= order`.
fn multi_indices(order: usize) -> Vec<[usize; 3]> {
    let mut out = Vec::new();
    for total in 0..=order {
        for x in (0..=total).rev() {
            for y in (0..=total - x).rev() {
                out.push([x, y, total - x - y]);
            }
        }
    }
    out
}

fn factorial(n: usize) -> f64 {
    (1..=n).map(|x| x as f64).product()
}

/// Cartesian derivatives `d^g (1/r)` at `(0, 0, 1)` for all `|g| <= order`.
fn coulomb_derivatives(order: usize) -> std::collections::HashMap<[usize; 3], f64> {
    // McMurchie-Davidson point-charge recursion: R^n_000 = (-1)^n (2n-1)!! / r^(2n+1),
    // R^n_{t+1,u,v} = t R^{n+1}_{t-1,u,v} + X R^{n+1}_{t,u,v} (similarly for u, v).
    let (x, y, z) = (0.0, 0.0, 1.0);
    let mut table: Vec<std::collections::HashMap<[usize; 3], f64>> = vec![Default::default(); order + 1];
    for n in 0..=order {
        let double_factorial: f64 = (1..=n).map(|k| (2 * k - 1) as f64).product();
        table[n].insert([0, 0, 0], if n % 2 == 0 { 1.0 } else { -1.0 } * double_factorial);
    }
    for total in 1..=order {
        for n in 0..=order - total {
            for g in multi_indices(total).into_iter().filter(|g| g[0] + g[1] + g[2] == total) {
                let (axis, coordinate) = if g[0] > 0 {
                    (0, x)
                } else if g[1] > 0 {
                    (1, y)
                } else {
                    (2, z)
                };
                let mut lower = g;
                lower[axis] -= 1;
                let mut value = coordinate * table[n + 1][&lower];
                if lower[axis] > 0 {
                    let mut lower2 = lower;
                    lower2[axis] -= 1;
                    value += lower[axis] as f64 * table[n + 1][&lower2];
                }
                table[n].insert(g, value);
            }
        }
    }
    table.swap_remove(0)
}

/// Hurwitz zeta `sum_{m >= q} m^-s` for integer `s >= 2`, `q >= 1`.
fn hurwitz_zeta(s: usize, q: usize) -> f64 {
    let explicit = 40;
    let mut sum: f64 = (q..q + explicit).map(|m| (m as f64).powi(-(s as i32))).sum();
    let n = (q + explicit) as f64;
    let s_f = s as f64;
    // Euler-Maclaurin tail starting at n.
    sum += n.powf(1.0 - s_f) / (s_f - 1.0) + 0.5 * n.powf(-s_f);
    let bernoulli = [1.0 / 6.0, -1.0 / 30.0, 1.0 / 42.0, -1.0 / 30.0, 5.0 / 66.0];
    let mut rising = s_f; // s (s+1) ... (s + 2j - 2)
    for (j, b) in bernoulli.iter().enumerate() {
        let order = 2 * (j + 1);
        sum += b / factorial(order) * rising * n.powf(-s_f - order as f64 + 1.0);
        rising *= (s_f + order as f64 - 1.0) * (s_f + order as f64);
    }
    sum
}

/// Stored periodic integrals `(m0 n@p | l@d s@(d+q))` for `|p|, |q| <= lp` and `|d| <= ld`.
struct PeriodicEri {
    nb: usize,
    lp: usize,
    ld: usize,
    values: Vec<f64>,
}

impl PeriodicEri {
    fn index(&self, p: isize, d: isize, q: isize) -> usize {
        let w = 2 * self.lp + 1;
        let block = self.nb.pow(4);
        (((d + self.ld as isize) as usize * w + (p + self.lp as isize) as usize) * w
            + (q + self.lp as isize) as usize)
            * block
    }

    fn block(&self, p: isize, d: isize, q: isize) -> &[f64] {
        let start = self.index(p, d, q);
        &self.values[start..start + self.nb.pow(4)]
    }
}

struct Model<'a> {
    atoms: &'a [Atom],
    nb: usize,
    a: f64,
    lp: usize,
    lx: usize,
    near: usize,
    order: usize,
    kpoints: Vec<f64>,
    n_occ: usize,
    kinetic: Vec<Matrix>,
    overlap: Vec<Matrix>,
    nuclear: Vec<Matrix>,
    eri: PeriodicEri,
    /// `<l@d| 1/|r - R_A| |s@(d+q)>` for every atom, `|d| <= near`, `|q| <= lp`.
    nuclear_site: Vec<Vec<Matrix>>,
    nuclear_nuclear_near: Vec<f64>,
    moments: Vec<Vec<Matrix>>,
    indices: Vec<[usize; 3]>,
    lattice_sums: std::collections::HashMap<[usize; 3], f64>,
}

impl Model<'_> {
    fn range(l: usize) -> std::ops::RangeInclusive<isize> {
        -(l as isize)..=l as isize
    }

    fn get(blocks: &[Matrix], l: usize, n: isize) -> Option<&Matrix> {
        (n.unsigned_abs() <= l).then(|| &blocks[(n + l as isize) as usize])
    }

    /// Fock blocks `F(n)`, `|n| <= lx`, and the energy components for density blocks `P`.
    fn fock(&self, p: &[Matrix]) -> (Vec<Matrix>, f64, f64, f64, f64) {
        let nb = self.nb;
        let (lp, lx, near) = (self.lp, self.lx, self.near);
        let dot = |a: &Matrix, b: &Matrix| -> f64 { a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum() };
        // Cell multipoles of the neutral cell charge (nuclei minus assigned electrons).
        let mut cell_moments = vec![0.0; self.indices.len()];
        for (k, power) in self.indices.iter().enumerate() {
            let mut value: f64 = self
                .atoms
                .iter()
                .map(|a| a.charge * (0..3).map(|i| a.position[i].powi(power[i] as i32)).product::<f64>())
                .sum();
            for q in Self::range(lp) {
                if let Some(pq) = Self::get(p, lx, q) {
                    value -= dot(pq, &self.moments[k][(q + lp as isize) as usize]);
                }
            }
            cell_moments[k] = value;
        }
        // Far-field potential coefficients: phi(r) = sum_alpha r^alpha c_alpha.
        let mut far = vec![0.0; self.indices.len()];
        for (ia, alpha) in self.indices.iter().enumerate() {
            for (ib, beta) in self.indices.iter().enumerate() {
                let total: usize = alpha.iter().chain(beta).sum();
                let beta_order: usize = beta.iter().sum();
                if beta_order == 0 || total > self.order {
                    continue;
                }
                let gamma = [alpha[0] + beta[0], alpha[1] + beta[1], alpha[2] + beta[2]];
                let g = self.lattice_sums.get(&gamma).copied().unwrap_or(0.0);
                if g == 0.0 {
                    continue;
                }
                let sign = if beta_order.is_multiple_of(2) { 1.0 } else { -1.0 };
                let denominator: f64 = alpha.iter().chain(beta).map(|&i| factorial(i)).product();
                far[ia] += sign * cell_moments[ib] * g / denominator;
            }
        }
        // Electrostatic potential matrix W(n) for |n| <= lp, exchange K(n) for |n| <= lx.
        let blocks: Vec<(Matrix, Matrix)> = Self::range(lx)
            .collect::<Vec<_>>()
            .par_iter()
            .map(|&n| {
                let mut w = Matrix::zeros(nb, nb);
                if n.unsigned_abs() <= lp {
                    w = self.nuclear[(n + lp as isize) as usize].clone();
                    for d in Self::range(near) {
                        for q in Self::range(lp) {
                            let Some(pq) = Self::get(p, lx, q) else { continue };
                            let block = self.eri.block(n, d, q);
                            for mu in 0..nb {
                                for nu in 0..nb {
                                    let base = (mu * nb + nu) * nb * nb;
                                    w[(mu, nu)] += block[base..base + nb * nb]
                                        .iter()
                                        .zip(&pq.data)
                                        .map(|(v, p)| v * p)
                                        .sum::<f64>();
                                }
                            }
                        }
                    }
                    for (k, coefficient) in far.iter().enumerate() {
                        if *coefficient != 0.0 {
                            let m = &self.moments[k][(n + lp as isize) as usize];
                            for (x, y) in w.data.iter_mut().zip(&m.data) {
                                *x -= coefficient * y;
                            }
                        }
                    }
                }
                // K(n)_mu,nu = sum_{m,l} P(l)_ls (mu0 l@m | nu@n s@(m+l)), q = m + l - n.
                let mut k_block = Matrix::zeros(nb, nb);
                for m in Self::range(lp) {
                    for q in Self::range(lp) {
                        let l = q + n - m;
                        let Some(pl) = Self::get(p, lx, l) else { continue };
                        if n.unsigned_abs() > self.eri.ld {
                            continue;
                        }
                        let block = self.eri.block(m, n, q);
                        for mu in 0..nb {
                            for la in 0..nb {
                                for nu in 0..nb {
                                    let base = ((mu * nb + la) * nb + nu) * nb;
                                    let mut v = 0.0;
                                    for si in 0..nb {
                                        v += block[base + si] * pl[(la, si)];
                                    }
                                    k_block[(mu, nu)] += v;
                                }
                            }
                        }
                    }
                }
                (w, k_block)
            })
            .collect();
        let mut fock = Vec::with_capacity(blocks.len());
        let (mut kinetic, mut electronic_w, mut exchange) = (0.0, 0.0, 0.0);
        for (i, n) in Self::range(lx).enumerate() {
            let (w, k) = &blocks[i];
            let mut f = w.sub(&k.scale(0.5));
            if let Some(t) = Self::get(&self.kinetic, lp, n) {
                f = f.add(t);
                kinetic += dot(&p[i], t);
            }
            electronic_w += 0.5 * dot(&p[i], w);
            exchange -= 0.25 * dot(&p[i], k);
            fock.push(f);
        }
        // Potential of all cells at each nucleus of cell 0 (self term excluded).
        let mut nuclear_term = 0.0;
        for (ia, atom) in self.atoms.iter().enumerate() {
            let mut phi = self.nuclear_nuclear_near[ia];
            for d in Self::range(near) {
                for q in Self::range(lp) {
                    if let Some(pq) = Self::get(p, lx, q) {
                        let integrals = &self.nuclear_site[ia][((d + near as isize) as usize) * (2 * lp + 1)
                            + (q + lp as isize) as usize];
                        phi -= dot(pq, integrals);
                    }
                }
            }
            for (k, power) in self.indices.iter().enumerate() {
                phi += far[k] * (0..3).map(|i| atom.position[i].powi(power[i] as i32)).product::<f64>();
            }
            nuclear_term += 0.5 * atom.charge * phi;
        }
        let electrostatic = electronic_w + nuclear_term;
        (fock, kinetic + electrostatic + exchange, kinetic, electrostatic, exchange)
    }

    fn bloch(&self, blocks: &[Matrix], l: usize, k: f64) -> ComplexMatrix {
        let mut out = ComplexMatrix::zeros(self.nb, self.nb);
        for (i, n) in Self::range(l).enumerate() {
            let phase = k * n as f64 * self.a;
            out = out.add(
                &ComplexMatrix {
                    re: blocks[i].scale(phase.cos()),
                    im: blocks[i].scale(phase.sin()),
                },
                1.0,
            );
        }
        // Remove roundoff anti-Hermitian parts.
        let h = out.dagger();
        ComplexMatrix {
            re: out.re.add(&h.re).scale(0.5),
            im: out.im.add(&h.im).scale(0.5),
        }
    }

    /// Output density blocks and band energies for Fock blocks `F`.
    fn diagonalize(&self, fock: &[Matrix]) -> Result<(Vec<Matrix>, Vec<Vec<f64>>), String> {
        let nk = self.kpoints.len() as f64;
        let results: Vec<Result<(Vec<f64>, ComplexMatrix), String>> = self
            .kpoints
            .par_iter()
            .map(|&k| {
                let s = self.bloch(&self.overlap, self.lp, k);
                let (values, vectors) = s.eigh()?;
                let keep: Vec<usize> = (0..values.len()).filter(|&i| values[i] > 1e-8 * values[values.len() - 1]).collect();
                if self.n_occ > keep.len() {
                    return Err("crystal-orbital occupations exceed the Bloch overlap rank".into());
                }
                let mut x = ComplexMatrix::zeros(self.nb, keep.len());
                for (c, &i) in keep.iter().enumerate() {
                    let scale = values[i].powf(-0.5);
                    for r in 0..self.nb {
                        x.re[(r, c)] = vectors.re[(r, i)] * scale;
                        x.im[(r, c)] = vectors.im[(r, i)] * scale;
                    }
                }
                let f = self.bloch(fock, self.lx, k);
                let (energies, u) = x.dagger().dot(&f).dot(&x).eigh()?;
                Ok((energies, x.dot(&u)))
            })
            .collect();
        let mut density = vec![Matrix::zeros(self.nb, self.nb); 2 * self.lx + 1];
        let mut bands = Vec::with_capacity(self.kpoints.len());
        for (result, &k) in results.into_iter().zip(&self.kpoints) {
            let (energies, c) = result?;
            let d = c.density(self.n_occ);
            for (i, n) in Self::range(self.lx).enumerate() {
                let phase = -k * n as f64 * self.a;
                let (cos, sin) = (phase.cos(), phase.sin());
                // Re[exp(i phase) D] = cos Re D - sin Im D, times 2 / N_k.
                for (out, (re, im)) in density[i].data.iter_mut().zip(d.re.data.iter().zip(&d.im.data)) {
                    *out += 2.0 * (cos * re - sin * im) / nk;
                }
            }
            bands.push(energies);
        }
        Ok((density, bands))
    }
}

/// Crystal-orbital RHF of a neutral closed-shell cell periodic along z.
pub fn run_crystal_orbital_hf_with_options(
    unit_cell_atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    lattice_constant_bohr: f64,
    options: &CoHfOptions,
) -> Result<COHFResult, String> {
    let a = lattice_constant_bohr;
    let nb = basis.len();
    if unit_cell_atoms.is_empty() || nb == 0 || !(a.is_finite() && a > 0.0) {
        return Err("crystal-orbital HF requires atoms, a basis and a positive lattice constant".into());
    }
    if options.n_kpoints == 0
        || options.max_iterations == 0
        || options.diis_size == 0
        || options.multipole_order < 2
        || [options.energy_tol, options.density_tol, options.overlap_threshold]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err("invalid crystal-orbital HF controls".into());
    }
    if crate::ecp::has_ecp(unit_cell_atoms) {
        return Err("crystal-orbital HF does not support effective core potentials".into());
    }
    let z_total: f64 = unit_cell_atoms.iter().map(|a| a.charge).sum();
    let electrons = z_total.round() - f64::from(charge);
    if charge != 0 {
        return Err("crystal-orbital HF requires a neutral unit cell (charged chains diverge)".into());
    }
    if electrons <= 0.0 || electrons as i64 % 2 != 0 {
        return Err("crystal-orbital RHF requires a positive even electron count per cell".into());
    }
    let n_occ = electrons as usize / 2;
    let shifted = |n: isize| translated(basis, n as f64 * a);
    // Overlap range L_ov.
    let mut lp = 0usize;
    loop {
        let other = shifted(lp as isize + 1);
        let largest = basis
            .iter()
            .flat_map(|x| other.iter().map(move |y| contracted_overlap(x, y).abs()))
            .fold(0f64, f64::max);
        if largest < options.overlap_threshold {
            break;
        }
        lp += 1;
        if lp > 64 {
            return Err("basis functions overlap beyond 64 cells; increase the lattice constant".into());
        }
    }
    // P(n) from N_k points is periodic with N_k cells; every block the overlap range touches
    // must be distinct, otherwise densities alias.
    let window = (options.n_kpoints - 1) / 2;
    if window < lp {
        return Err(format!(
            "{} k points alias density blocks: AOs overlap over {lp} cells, so at least {} k points are required",
            options.n_kpoints,
            2 * lp + 1
        ));
    }
    let lx = window.min(options.max_exchange_cells).max(lp);
    let near = options
        .near_field_cells
        .unwrap_or_else(|| (2 * lp + 3).max((20.0 / a).ceil() as usize));
    let ld = near.max(lx);
    let block = nb.pow(4);
    let count = block
        .checked_mul((2 * lp + 1).pow(2))
        .and_then(|v| v.checked_mul(2 * ld + 1))
        .and_then(|v| v.checked_mul(8));
    if count.is_none_or(|bytes| bytes > options.integral_budget_bytes) {
        return Err(format!(
            "periodic ERIs need {count:?} bytes, above the budget {} (reduce near-field cells or k range)",
            options.integral_budget_bytes
        ));
    }
    let cells: Vec<isize> = (-(ld as isize)..=ld as isize).collect();
    let pair_range: Vec<isize> = (-(lp as isize)..=lp as isize).collect();
    let translated_cells: std::collections::HashMap<isize, Vec<ContractedGaussian>> = (-(ld as isize + lp as isize)
        ..=ld as isize + lp as isize)
        .map(|n| (n, shifted(n)))
        .collect();
    let mut values = vec![0.0; count.unwrap() / 8];
    let eri_shape = PeriodicEri { nb, lp, ld, values: Vec::new() };
    let jobs: Vec<(isize, isize, isize)> = cells
        .iter()
        .flat_map(|&d| {
            let range = &pair_range;
            range.iter().flat_map(move |&p| range.iter().map(move |&q| (d, p, q)))
        })
        .collect();
    let blocks: Vec<Vec<f64>> = jobs
        .par_iter()
        .map(|&(d, p, q)| {
            let (b0, bp, bd, bdq) = (
                &translated_cells[&0],
                &translated_cells[&p],
                &translated_cells[&d],
                &translated_cells[&(d + q)],
            );
            let mut out = vec![0.0; block];
            for mu in 0..nb {
                for nu in 0..nb {
                    for la in 0..nb {
                        for si in 0..nb {
                            out[((mu * nb + nu) * nb + la) * nb + si] =
                                contracted_eri(&b0[mu], &bp[nu], &bd[la], &bdq[si]);
                        }
                    }
                }
            }
            out
        })
        .collect();
    for (&(d, p, q), v) in jobs.iter().zip(blocks) {
        let start = eri_shape.index(p, d, q);
        values[start..start + block].copy_from_slice(&v);
    }
    let eri = PeriodicEri { nb, lp, ld, values };
    let pair_blocks = |f: &dyn Fn(&ContractedGaussian, &ContractedGaussian) -> f64| -> Vec<Matrix> {
        pair_range
            .iter()
            .map(|&n| {
                let other = &translated_cells[&n];
                let mut m = Matrix::zeros(nb, nb);
                for i in 0..nb {
                    for j in 0..nb {
                        m[(i, j)] = f(&basis[i], &other[j]);
                    }
                }
                m
            })
            .collect()
    };
    let overlap = pair_blocks(&contracted_overlap);
    let kinetic = pair_blocks(&contracted_kinetic);
    let near_atoms: Vec<Atom> = (-(near as isize)..=near as isize)
        .flat_map(|m| {
            unit_cell_atoms.iter().map(move |atom| {
                let mut t = atom.clone();
                t.position[2] += m as f64 * a;
                t
            })
        })
        .collect();
    let nuclear = pair_blocks(&|x, y| contracted_nuclear_attraction(x, y, &near_atoms));
    let nuclear_site: Vec<Vec<Matrix>> = unit_cell_atoms
        .par_iter()
        .map(|atom| {
            let mut probe = atom.clone();
            probe.charge = 1.0;
            let probe = [probe];
            (-(near as isize)..=near as isize)
                .flat_map(|d| pair_range.iter().map(move |&q| (d, q)))
                .map(|(d, q)| {
                    let (bd, bdq) = (&translated_cells[&d], &translated_cells[&(d + q)]);
                    let mut m = Matrix::zeros(nb, nb);
                    for i in 0..nb {
                        for j in 0..nb {
                            m[(i, j)] = -contracted_nuclear_attraction(&bd[i], &bdq[j], &probe);
                        }
                    }
                    m
                })
                .collect()
        })
        .collect();
    let nuclear_nuclear_near: Vec<f64> = unit_cell_atoms
        .iter()
        .enumerate()
        .map(|(ia, atom)| {
            let mut phi = 0.0;
            for m in -(near as isize)..=near as isize {
                for (ib, other) in unit_cell_atoms.iter().enumerate() {
                    if m == 0 && ia == ib {
                        continue;
                    }
                    let r: f64 = (0..3)
                        .map(|k| {
                            let shift = if k == 2 { m as f64 * a } else { 0.0 };
                            (atom.position[k] - other.position[k] - shift).powi(2)
                        })
                        .sum::<f64>()
                        .sqrt();
                    if r < 1e-8 {
                        return Err("coincident nuclei in the periodic chain".to_string());
                    }
                    phi += other.charge / r;
                }
            }
            Ok(phi)
        })
        .collect::<Result<_, String>>()?;
    let indices = multi_indices(options.multipole_order);
    let moments: Vec<Vec<Matrix>> = indices
        .par_iter()
        .map(|&power| pair_blocks(&|x, y| moment(x, y, power)))
        .collect();
    let derivatives = coulomb_derivatives(options.multipole_order);
    let lattice_sums = derivatives
        .iter()
        .filter(|(g, _)| (g[0] + g[1] + g[2]) % 2 == 0 && g.iter().sum::<usize>() >= 2)
        .map(|(g, c)| {
            let order = g.iter().sum::<usize>();
            (*g, 2.0 * c * hurwitz_zeta(order + 1, near + 1) / a.powi(order as i32 + 1))
        })
        .collect();
    let kpoints: Vec<f64> = (0..options.n_kpoints)
        .map(|j| {
            let mut f = j as f64 / options.n_kpoints as f64;
            if f > 0.5 {
                f -= 1.0;
            }
            2.0 * std::f64::consts::PI * f / a
        })
        .collect();
    let model = Model {
        atoms: unit_cell_atoms,
        nb,
        a,
        lp,
        lx,
        near,
        order: options.multipole_order,
        kpoints,
        n_occ,
        kinetic,
        overlap,
        nuclear,
        eri,
        nuclear_site,
        nuclear_nuclear_near,
        moments,
        indices,
        lattice_sums,
    };

    // Initial guess: the superposition-of-atoms density of one cell.
    let s0 = &model.overlap[lp];
    let mut density = vec![Matrix::zeros(nb, nb); 2 * lx + 1];
    density[lx] = crate::sad::compute_sad_density(unit_cell_atoms, basis, 0, s0)
        .map_err(|e| format!("crystal-orbital initial guess: {e}"))?;
    let (mut inputs, mut residuals): (Vec<Vec<Matrix>>, Vec<Vec<Matrix>>) = (Vec::new(), Vec::new());
    let mut previous: Option<f64> = None;
    let mut residual = f64::INFINITY;
    for iteration in 1..=options.max_iterations {
        let (fock, energy, kinetic, electrostatic, exchange) = model.fock(&density);
        let (out, bands) = model.diagonalize(&fock)?;
        residual = out
            .iter()
            .zip(&density)
            .flat_map(|(a, b)| a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()))
            .fold(0f64, f64::max);
        if residual <= options.density_tol && previous.is_some_and(|p: f64| (energy - p).abs() <= options.energy_tol) {
            let vbm = bands.iter().map(|b| b[n_occ - 1]).fold(f64::NEG_INFINITY, f64::max);
            let cbm = bands
                .iter()
                .filter(|b| b.len() > n_occ)
                .map(|b| b[n_occ])
                .fold(f64::INFINITY, f64::min);
            if cbm <= vbm {
                return Err(format!("no band gap (VBM {vbm}, CBM {cbm}); metallic chains are not supported"));
            }
            let electrons_per_cell: f64 = Model::range(lp)
                .map(|n| {
                    let p = &density[(n + lx as isize) as usize];
                    p.data.iter().zip(&model.overlap[(n + lp as isize) as usize].data).map(|(x, y)| x * y).sum::<f64>()
                })
                .sum();
            let gap = cbm - vbm;
            return Ok(COHFResult {
                lattice_constant_bohr: a,
                n_kpoints: options.n_kpoints,
                total_energy_per_cell: energy,
                kinetic_energy_per_cell: kinetic,
                electrostatic_energy_per_cell: electrostatic,
                exchange_energy_per_cell: exchange,
                band_structure: model
                    .kpoints
                    .iter()
                    .zip(bands)
                    .map(|(&k, band_energies)| BandPoint {
                        k_fractional: k * a / (2.0 * std::f64::consts::PI),
                        k_cartesian: k,
                        band_energies,
                    })
                    .collect(),
                valence_band_maximum: vbm,
                conduction_band_minimum: cbm,
                band_gap_hartree: gap,
                band_gap_ev: gap * 27.211386245988,
                density_blocks: density,
                density_range: lx,
                overlap_range: lp,
                near_field_cells: near,
                electrons_per_cell,
                density_residual: residual,
                iterations: iteration,
                converged: true,
            });
        }
        previous = Some(energy);
        let difference: Vec<Matrix> = out.iter().zip(&density).map(|(a, b)| a.sub(b)).collect();
        inputs.push(out);
        residuals.push(difference);
        if inputs.len() > options.diis_size {
            inputs.remove(0);
            residuals.remove(0);
        }
        let m = residuals.len();
        let mut b = Matrix::zeros(m + 1, m + 1);
        for i in 0..m {
            for j in 0..m {
                b[(i, j)] = residuals[i]
                    .iter()
                    .zip(&residuals[j])
                    .map(|(x, y)| x.data.iter().zip(&y.data).map(|(u, v)| u * v).sum::<f64>())
                    .sum();
            }
            b[(i, m)] = -1.0;
            b[(m, i)] = -1.0;
        }
        let mut rhs = vec![0.0; m + 1];
        rhs[m] = -1.0;
        density = match b.lu_decompose() {
            Some(lu) => {
                let c = lu.solve(&rhs);
                (0..2 * lx + 1)
                    .map(|i| inputs.iter().zip(&c).fold(Matrix::zeros(nb, nb), |acc, (x, c)| acc.add(&x[i].scale(*c))))
                    .collect()
            }
            None => {
                inputs.drain(..m - 1);
                residuals.drain(..m - 1);
                inputs[0].clone()
            }
        };
    }
    Err(format!(
        "crystal-orbital HF did not converge in {} iterations: density residual {residual:e}",
        options.max_iterations
    ))
}
