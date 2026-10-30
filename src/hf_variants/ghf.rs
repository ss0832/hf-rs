// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Real noncollinear GHF with metric/residual-certified stationary determinants.
//! The spinor density retains all four spin blocks. Variational nuclear
//! gradients include spin coherence, Pulay terms and local/nonlocal ECPs.

use crate::basis::{Atom, ContractedGaussian};
use crate::matrix::Matrix;
use crate::properties::{MolecularProperties, compute_molecular_properties};
use crate::scf::{
    compute_eri_tensor, compute_nuclear_repulsion, compute_one_electron_integrals, eri_index,
};
use rayon::prelude::*;

/// Fukutome 8-fold classification of Hartree-Fock wavefunctions based on
/// spin-rotation and time-reversal symmetry breaking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FukutomeClass {
    /// Time-reversal Invariant Closed-Shell (RHF)
    TICS,
    /// Charge Current Wave (Complex RHF)
    CCW,
    /// Axial Spin Current Wave (Complex collinear)
    ASCW,
    /// Axial Spin Density Wave (Real UHF, collinear)
    ASDW,
    /// Axial Spin Wave (Complex UHF)
    ASW,
    /// Torsional Spin Current Wave (Complex non-collinear)
    TSCW,
    /// Torsional Spin Density Wave (Real GHF, non-collinear)
    TSDW,
    /// Torsional Spin Wave (Most general complex GHF)
    TSW,
}

impl FukutomeClass {
    pub fn description(&self) -> &'static str {
        match self {
            Self::TICS => "TICS (Time-reversal Invariant Closed-Shell: standard RHF)",
            Self::CCW => "CCW (Charge Current Wave: complex RHF with ring currents)",
            Self::ASCW => "ASCW (Axial Spin Current Wave: complex spin-collinear)",
            Self::ASDW => "ASDW (Axial Spin Density Wave: standard real UHF)",
            Self::ASW => "ASW (Axial Spin Wave: complex UHF with spin currents)",
            Self::TSCW => "TSCW (Torsional Spin Current Wave: complex non-collinear)",
            Self::TSDW => "TSDW (Torsional Spin Density Wave: real GHF with non-collinear spin)",
            Self::TSW => "TSW (Torsional Spin Wave: fully general complex GHF)",
        }
    }
}

/// Results from a Generalized Hartree-Fock (GHF) calculation.
#[derive(Clone, Debug)]
pub struct GHFResult {
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_repulsion_energy: f64,
    pub spinor_energies: Vec<f64>,
    pub spinor_coefficients: Matrix, // Dimensions: 2N x 2N
    pub density_matrix_2n: Matrix,   // Dimensions: 2N x 2N
    pub fock_matrix_2n: Matrix,      // Dimensions: 2N x 2N
    pub s_squared: f64,              // <S^2> expectation value
    pub s_z: f64,                    // <S_z> expectation value
    pub s_x: f64,                    // <S_x> expectation value
    pub s_y: f64,                    // <S_y> expectation value
    pub fukutome_class: FukutomeClass,
    pub iterations: usize,
    pub converged: bool,
    /// Frobenius norm of the commutator in the retained orthonormal basis.
    pub residual_norm: f64,
    pub density_change: f64,
    pub charge: i32,
    pub n_electrons: usize,
    pub properties: MolecularProperties,
}

/// Fixed-spin-axis representation diagnostic for a real GHF density.
/// The historical TICS/ASDW/TSDW labels do not constitute a complete,
/// spin-rotation-invariant eight-class analysis of complex determinants.
pub fn classify_fukutome_symmetry(density_2n: &Matrix, n: usize) -> FukutomeClass {
    assert_eq!(density_2n.rows, 2 * n);
    assert_eq!(density_2n.cols, 2 * n);

    let mut max_diff_diag: f64 = 0.0;
    let mut max_offdiag_spin: f64 = 0.0;

    for i in 0..n {
        for j in 0..n {
            let p_aa = density_2n[(i, j)];
            let p_bb = density_2n[(n + i, n + j)];
            let p_ab = density_2n[(i, n + j)];

            max_diff_diag = max_diff_diag.max((p_aa - p_bb).abs());
            max_offdiag_spin = max_offdiag_spin.max(p_ab.abs());
        }
    }

    let is_closed = max_diff_diag < 1e-5;
    let is_collinear = max_offdiag_spin < 1e-5;

    if is_closed && is_collinear {
        FukutomeClass::TICS
    } else if !is_closed && is_collinear {
        FukutomeClass::ASDW
    } else {
        FukutomeClass::TSDW
    }
}

/// Controls for real, noncollinear GHF. Initial densities use alpha-then-beta
/// AO ordering and must represent an orthonormal Slater determinant.
#[derive(Clone, Debug)]
pub struct GhfOptions {
    pub max_iterations: usize,
    pub energy_tol: f64,
    pub residual_tol: f64,
    pub density_tol: f64,
    pub overlap_cutoff: f64,
    pub max_work_bytes: usize,
    pub initial_density: Option<Matrix>,
}
impl Default for GhfOptions {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            energy_tol: 1e-12,
            residual_tol: 1e-10,
            density_tol: 1e-10,
            overlap_cutoff: 1e-10,
            max_work_bytes: 256 * 1024 * 1024,
            initial_density: None,
        }
    }
}

/// Exchange acts on nonsymmetric spin-coherence blocks as well.
pub fn build_exchange_block(d: &Matrix, eri: &[f64], n: usize) -> Matrix {
    assert_eq!((d.rows, d.cols), (n, n));
    let rows = (0..n)
        .into_par_iter()
        .map(|p| {
            (0..n)
                .map(|q| {
                    (0..n)
                        .flat_map(|r| (0..n).map(move |s| d[(r, s)] * eri[eri_index(p, r, q, s)]))
                        .sum::<f64>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Matrix::from_vec(n, n, rows.into_iter().flatten().collect())
}

/// F_st = delta_st (h + J[Tr_spin P]) - K[P_st]. No density cutoffs.
pub fn build_ghf_fock_matrix(h: &Matrix, p: &Matrix, eri: &[f64], n: usize) -> Matrix {
    assert_eq!((h.rows, h.cols), (n, n));
    assert_eq!((p.rows, p.cols), (2 * n, 2 * n));
    let jrows = (0..n)
        .into_par_iter()
        .map(|a| {
            (0..n)
                .map(|b| {
                    (0..n)
                        .flat_map(|r| {
                            (0..n).map(move |s| {
                                (p[(r, s)] + p[(n + r, n + s)]) * eri[eri_index(a, b, r, s)]
                            })
                        })
                        .sum::<f64>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let rows = (0..2 * n)
        .into_par_iter()
        .map(|a| {
            (0..2 * n)
                .map(|b| {
                    let (spin_a, spin_b, mu, nu) = (a / n, b / n, a % n, b % n);
                    let exchange = (0..n)
                        .flat_map(|r| {
                            (0..n).map(move |s| {
                                p[(spin_a * n + r, spin_b * n + s)] * eri[eri_index(mu, r, nu, s)]
                            })
                        })
                        .sum::<f64>();
                    if spin_a == spin_b {
                        h[(mu, nu)] + jrows[mu][nu] - exchange
                    } else {
                        -exchange
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Matrix::from_vec(2 * n, 2 * n, rows.into_iter().flatten().collect())
}

fn block_diagonal(a: &Matrix) -> Matrix {
    let mut out = Matrix::zeros(2 * a.rows, 2 * a.cols);
    for p in 0..a.rows {
        for q in 0..a.cols {
            out[(p, q)] = a[(p, q)];
            out[(a.rows + p, a.cols + q)] = a[(p, q)];
        }
    }
    out
}
fn norm(a: &Matrix) -> f64 {
    a.data.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Spin vector and exact S2 for a real metric-idempotent Slater density.
/// Tiny signed roundoff in a singlet's S2 is retained without clipping.
pub fn ghf_spin_expectations(p: &Matrix, s: &Matrix) -> ([f64; 3], f64) {
    let n = s.rows;
    assert_eq!((s.rows, s.cols), (n, n));
    assert_eq!((p.rows, p.cols), (2 * n, 2 * n));
    let mut total = Matrix::zeros(n, n);
    let mut sx = 0.0;
    let mut sz = 0.0;
    for a in 0..n {
        for b in 0..n {
            total[(a, b)] = p[(a, b)] + p[(n + a, n + b)];
            sx += 0.5 * (p[(a, n + b)] + p[(n + a, b)]) * s[(b, a)];
            sz += 0.5 * (p[(a, b)] - p[(n + a, n + b)]) * s[(b, a)];
        }
    }
    let ps = total.dot(s);
    let s2 = sx * sx + sz * sz + ps.trace() - 0.5 * ps.dot(&ps).trace();
    ([sx, 0.0, sz], s2)
}
fn occupied_density(c: &Matrix, ne: usize) -> Matrix {
    c.occupied_density(ne, 1.0)
}
fn canonicalize_subspaces(c: &Matrix, f: &Matrix, ne: usize) -> (Vec<f64>, Matrix) {
    let fm = c.transpose().dot(f).dot(c);
    let mut u = Matrix::identity(c.cols);
    let mut energies = vec![0.; c.cols];
    for (first, last) in [(0, ne), (ne, c.cols)] {
        if first == last {
            continue;
        }
        let size = last - first;
        let mut block = Matrix::zeros(size, size);
        for p in 0..size {
            for q in 0..size {
                block[(p, q)] = fm[(first + p, first + q)];
            }
        }
        let (values, vectors) = block.eigh();
        for p in 0..size {
            energies[first + p] = values[p];
            for q in 0..size {
                u[(first + p, first + q)] = vectors[(p, q)];
            }
        }
    }
    (energies, c.dot(&u))
}

// A tilted core guess can converge to an unstable RHF stationary point.
// Follow its lowest real spin-breaking curvature using the same AO ERIs.
// No full MO tensor or second copy of the AO tensor is formed.
fn spin_breaking_seed(
    f: &Matrix,
    eri: &[f64],
    xs: &Matrix,
    ne: usize,
    tilt: f64,
    max_bytes: usize,
) -> Result<Option<Matrix>, String> {
    if !ne.is_multiple_of(2) {
        return Ok(None);
    }
    let n = xs.rows;
    let rank = xs.cols;
    let occupied = ne / 2;
    if occupied == rank {
        return Ok(None);
    }
    let size = occupied * (rank - occupied);
    if size
        .checked_mul(size)
        .and_then(|x| x.checked_mul(32))
        .is_none_or(|x| x > max_bytes)
    {
        return Err("GHF spin-breaking stability matrix exceeds max_work_bytes".into());
    }
    let mut spatial_fock = Matrix::zeros(n, n);
    for p in 0..n {
        for q in 0..n {
            spatial_fock[(p, q)] = f[(p, q)];
        }
    }
    let (eps, u) = xs.transpose().dot(&spatial_fock).dot(xs).eigh();
    let c = xs.dot(&u);
    let co = c.leading_columns(occupied);
    let mut cv = Matrix::zeros(n, rank - occupied);
    for p in 0..n {
        for a in occupied..rank {
            cv[(p, a - occupied)] = c[(p, a)];
        }
    }
    let mut hessian = Matrix::zeros(size, size);
    for b in occupied..rank {
        for j in 0..occupied {
            let column = (b - occupied) * occupied + j;
            let mut density = Matrix::zeros(n, n);
            for p in 0..n {
                for q in 0..n {
                    density[(p, q)] = c[(p, b)] * c[(q, j)] + c[(p, j)] * c[(q, b)];
                }
            }
            let exchange = build_exchange_block(&density, eri, n);
            let response = cv.transpose().dot(&exchange).dot(&co);
            for a in occupied..rank {
                for i in 0..occupied {
                    let row = (a - occupied) * occupied + i;
                    hessian[(row, column)] = -response[(a - occupied, i)];
                    if row == column {
                        hessian[(row, column)] += eps[a] - eps[i];
                    }
                }
            }
        }
    }
    let (values, vectors) = hessian.add(&hessian.transpose()).scale(0.5).eigh();
    if values[0] >= -1e-8 {
        return Ok(None);
    }
    let mut generator = Matrix::zeros(rank, rank);
    for a in occupied..rank {
        for i in 0..occupied {
            let value = 0.2 * vectors[((a - occupied) * occupied + i, 0)];
            generator[(a, i)] = value;
            generator[(i, a)] = -value;
        }
    }
    let alpha = c.dot(&generator.exp());
    let beta = c.dot(&generator.scale(-1.).exp());
    let mut seed = Matrix::zeros(2 * n, 2 * rank);
    for p in 0..n {
        for q in 0..rank {
            seed[(p, 2 * q)] = alpha[(p, q)];
            seed[(n + p, 2 * q + 1)] = beta[(p, q)];
        }
    }
    let (co, si) = ((tilt * 0.5).cos(), (tilt * 0.5).sin());
    for p in 0..n {
        for q in 0..2 * rank {
            let (a, b) = (seed[(p, q)], seed[(n + p, q)]);
            seed[(p, q)] = co * a - si * b;
            seed[(n + p, q)] = si * a + co * b;
        }
    }
    Ok(Some(seed))
}

/// Compatibility runner; convergence also requires the metric residual and density.
pub fn run_ghf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    tilt_angle_rad: f64,
    max_iter: usize,
    tol: f64,
) -> Result<GHFResult, String> {
    let options = GhfOptions {
        max_iterations: max_iter,
        energy_tol: tol,
        residual_tol: tol.sqrt().min(1e-8),
        density_tol: tol.sqrt().min(1e-8),
        ..Default::default()
    };
    run_ghf_with_options(atoms, basis, charge, tilt_angle_rad, &options)
}

/// Solve real GHF with a retained overlap space and consistent final P, F and C.
pub fn run_ghf_with_options(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    tilt: f64,
    options: &GhfOptions,
) -> Result<GHFResult, String> {
    let n = basis.len();
    if n == 0
        || atoms.is_empty()
        || !tilt.is_finite()
        || options.max_iterations == 0
        || [
            options.energy_tol,
            options.residual_tol,
            options.density_tol,
            options.overlap_cutoff,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
        || atoms
            .iter()
            .any(|a| !a.charge.is_finite() || a.position.iter().any(|x| !x.is_finite()))
        || basis.iter().any(|b| b.atom_index >= atoms.len())
    {
        return Err("invalid GHF geometry, basis or convergence controls".into());
    }
    let ne = atoms.iter().map(|a| a.charge).sum::<f64>().round() as i64 - i64::from(charge);
    if ne <= 0 || ne > 2 * n as i64 {
        return Err("GHF electron count exceeds the AO spinor space".into());
    }
    let ne = ne as usize;
    let pairs = n.checked_mul(n + 1).and_then(|x| x.checked_div(2));
    let bytes = pairs
        .and_then(|x| x.checked_mul(x + 1))
        .and_then(|x| x.checked_mul(4))
        .and_then(|x| n.checked_mul(n)?.checked_mul(1024)?.checked_add(x));
    if bytes.is_none_or(|x| x > options.max_work_bytes) {
        return Err("GHF integral/matrix working-storage estimate exceeds max_work_bytes".into());
    }
    let e_nuc = compute_nuclear_repulsion(atoms);
    if !e_nuc.is_finite() {
        return Err("GHF nuclei must have distinct finite positions".into());
    }
    let (s, _, _, h) = compute_one_electron_integrals(atoms, basis);
    let xs = s
        .canonical_orthogonalizer(options.overlap_cutoff)
        .map_err(|e| e.to_string())?;
    let rank = xs.cols;
    if ne > 2 * rank {
        return Err("GHF electrons exceed retained overlap rank".into());
    }
    let x = block_diagonal(&xs);
    let s2 = block_diagonal(&s);
    let h2 = block_diagonal(&h);
    let eri = compute_eri_tensor(basis);
    let hc = xs.transpose().dot(&h).dot(&xs);
    let (_, spatial) = hc.eigh();
    let mut u = Matrix::zeros(2 * rank, 2 * rank);
    for p in 0..rank {
        for q in 0..rank {
            u[(p, 2 * q)] = spatial[(p, q)];
            u[(rank + p, 2 * q + 1)] = spatial[(p, q)];
        }
    }
    if tilt != 0. {
        // A single antisymmetric generator rotates the COMPLETE orbital basis.
        // All occupied and virtual columns retain mutual orthogonality.
        let mut rotation = Matrix::zeros(2 * rank, 2 * rank);
        for p in 0..rank {
            let angle = tilt * (1. + 0.3 * p as f64) * 0.5;
            rotation[(rank + p, p)] = angle;
            rotation[(p, rank + p)] = -angle;
        }
        if ne >= 2 && ne + 1 < 2 * rank {
            for (i, a, angle) in [
                (ne - 2, ne, std::f64::consts::FRAC_PI_4),
                (ne - 1, ne + 1, -std::f64::consts::FRAC_PI_4),
            ] {
                let (co, si) = (angle.cos(), angle.sin());
                for p in 0..2 * rank {
                    let (ci, ca) = (u[(p, i)], u[(p, a)]);
                    u[(p, i)] = co * ci + si * ca;
                    u[(p, a)] = -si * ci + co * ca;
                }
            }
        }
        u = rotation.exp().dot(&u);
    }
    let mut c = x.dot(&u);
    let mut p = occupied_density(&c, ne);
    if let Some(initial) = &options.initial_density {
        if initial.rows != 2 * n
            || initial.cols != 2 * n
            || initial.data.len() != 4 * n * n
            || initial.data.iter().any(|x| !x.is_finite())
            || norm(&initial.sub(&initial.transpose())) > 1e-8
            || (initial.dot(&s2).trace() - ne as f64).abs() > 1e-8
            || norm(&initial.dot(&s2).dot(initial).sub(initial)) > 1e-8
        {
            return Err(
                "GHF initial density must be a normalized metric-idempotent determinant".into(),
            );
        }
        let projected = x.transpose().dot(&s2).dot(initial).dot(&s2).dot(&x);
        if norm(&x.dot(&projected).dot(&x.transpose()).sub(initial)) > 1e-8 {
            return Err(
                "GHF initial density has components outside the retained overlap space".into(),
            );
        }
        let (values, vectors) = projected.scale(-1.).eigh();
        if values[ne - 1] > -0.5 {
            return Err("GHF initial determinant rank is inconsistent".into());
        }
        c = x.dot(&vectors);
        p = occupied_density(&c, ne);
    }
    let mut previous = None;
    let mut density_change = f64::INFINITY;
    let mut diis = crate::diis::CdiisEngine::new(8);
    let mut last_residual = f64::INFINITY;
    let mut stability_checked = false;
    let mut diis_start = 1;
    for iteration in 1..=options.max_iterations {
        let f = build_ghf_fock_matrix(&h, &p, &eri, n);
        let energy = 0.5
            * p.data
                .iter()
                .zip(h2.add(&f).data)
                .map(|(p, f)| p * f)
                .sum::<f64>()
            + e_nuc;
        let residual = x
            .transpose()
            .dot(&f.dot(&p).dot(&s2).sub(&s2.dot(&p).dot(&f)))
            .dot(&x);
        last_residual = norm(&residual);
        if previous.is_some_and(|e: f64| (energy - e).abs() <= options.energy_tol)
            && last_residual <= options.residual_tol
            && density_change <= options.density_tol
        {
            if tilt != 0.
                && options.initial_density.is_none()
                && !stability_checked
                && classify_fukutome_symmetry(&p, n) == FukutomeClass::TICS
            {
                stability_checked = true;
                if let Some(seed) = spin_breaking_seed(
                    &f,
                    &eri,
                    &xs,
                    ne,
                    tilt,
                    options.max_work_bytes.saturating_sub(bytes.unwrap_or(0)),
                )? {
                    c = seed;
                    p = occupied_density(&c, ne);
                    previous = None;
                    density_change = f64::INFINITY;
                    diis = crate::diis::CdiisEngine::new(8);
                    // Initially follow the unstable fixed-point direction;
                    // DIIS can otherwise return to the RHF saddle immediately.
                    diis_start = iteration + 8;
                    continue;
                }
            }
            let (energies, c) = canonicalize_subspaces(&c, &f, ne);
            let mut total = Matrix::zeros(n, n);
            for a in 0..n {
                for b in 0..n {
                    total[(a, b)] = p[(a, b)] + p[(n + a, n + b)];
                }
            }
            let ([sx, _, sz], ss) = ghf_spin_expectations(&p, &s);
            return Ok(GHFResult {
                total_energy: energy,
                electronic_energy: energy - e_nuc,
                nuclear_repulsion_energy: e_nuc,
                spinor_energies: energies,
                spinor_coefficients: c,
                density_matrix_2n: p.clone(),
                fock_matrix_2n: f,
                s_squared: ss,
                s_z: sz,
                s_x: sx,
                s_y: 0.,
                fukutome_class: super::symmetry::classify_complex_hf_symmetry(
                    &p, &Matrix::zeros(2*n,2*n), &s, 1e-8,
                )?.class,
                iterations: iteration,
                converged: true,
                residual_norm: last_residual,
                density_change,
                charge,
                n_electrons: ne,
                properties: compute_molecular_properties(atoms, basis, &total, &s),
            });
        }
        previous = Some(energy);
        let orthofock = x.transpose().dot(&f).dot(&x);
        let next = if iteration >= diis_start {
            diis.push(orthofock.clone(), residual);
            diis.extrapolate().unwrap_or(orthofock)
        } else {
            orthofock
        };
        let (_, vectors) = next.eigh();
        c = x.dot(&vectors);
        let next = occupied_density(&c, ne);
        density_change = norm(&next.sub(&p)) / (2 * n) as f64;
        p = next;
    }
    Err(format!(
        "GHF did not converge in {} iterations; metric residual {last_residual:e}, density change {density_change:e}",
        options.max_iterations
    ))
}

/// Variational real-GHF gradient, including spin coherence and nonlocal ECP.
/// Canonically truncated overlap spaces require explicit projector response.
pub fn compute_ghf_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    result: &GHFResult,
) -> Result<Vec<[f64; 3]>, String> {
    let n = basis.len();
    if !result.converged
        || result.spinor_coefficients.rows != 2 * n
        || result.spinor_coefficients.cols != 2 * n
    {
        return Err("GHF analytic gradient requires a converged full-rank overlap space".into());
    }
    let p = &result.density_matrix_2n;
    let weighted = p.dot(&result.fock_matrix_2n).dot(p);
    let mut total = Matrix::zeros(n, n);
    let mut w = Matrix::zeros(n, n);
    for a in 0..n {
        for b in 0..n {
            total[(a, b)] = p[(a, b)] + p[(n + a, n + b)];
            w[(a, b)] = weighted[(a, b)] + weighted[(n + a, n + b)];
        }
    }
    let gradient = crate::gradient::compute_gradient_from_density_with_eri_contraction(
        atoms,
        basis,
        &total,
        &w,
        |mu, nu, lambda, sigma| {
            let direct = if lambda == sigma { 2.0 } else { 4.0 };
            let mut value = direct * total[(mu, nu)] * total[(lambda, sigma)];
            for a in 0..2 {
                for b in 0..2 {
                    value -= 2.0 * p[(a * n + mu, b * n + lambda)] * p[(a * n + nu, b * n + sigma)];
                    if lambda != sigma {
                        value -=
                            2.0 * p[(a * n + mu, b * n + sigma)] * p[(a * n + nu, b * n + lambda)];
                    }
                }
            }
            value
        },
    );
    Ok(gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{Atom, STO_3G_GBS, build_basis, parse_gaussian94};

    #[test]
    fn test_ghf_h2_collinear() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0, 0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);

        // Standard collinear GHF on H2
        let res = run_ghf(&atoms, &basis, 0, 0.0, 50, 1e-10).expect("GHF H2 failed");
        assert!(res.converged);
        assert!((res.total_energy - (-1.116712)).abs() < 1e-4);
        assert_eq!(res.fukutome_class, FukutomeClass::TICS);
    }

    #[test]
    fn test_ghf_h2_non_collinear_spin_tilt() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0, 0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);

        // GHF with non-collinear spin tilt of 90 degrees (pi/2)
        let tilt = std::f64::consts::FRAC_PI_2;
        let res = run_ghf(&atoms, &basis, 0, tilt, 50, 1e-10).expect("GHF tilt failed");
        assert!(res.converged);
        assert!(res.total_energy < -1.1);
        eprintln!(
            "GHF with 90 deg tilt: E = {:.6} Eh, class = {:?}",
            res.total_energy, res.fukutome_class
        );
    }
}
