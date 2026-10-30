// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Internally contracted CASPT2 in an excitation-generated first-order space.
//!
//! Spin-free A-H perturbers act on the exact active CI vector. Contracting the
//! generalized Fock action between these vectors includes the exact 3-RDM and
//! F.4-RDM terms without storing an eighth-rank tensor. No full-orbital FCI
//! space is enumerated. This bounded correctness backend retains dense metric
//! blocks and the orthonormal contracted Fock matrix.

use crate::error::HfError;
use crate::fci::excitations::{Determinant, cas_references};
use crate::fci::subspace::SubspaceHamiltonian;
use crate::matrix::Matrix;
use crate::mo_integrals::MoEriTensor;
use crate::options::ResolvedActiveSpace;
use std::collections::{BTreeMap, BTreeSet, HashMap};

const BUDGET: usize = 256 * 1024 * 1024;
type Sparse = BTreeMap<Determinant, f64>;

/// Andersson classes, specified by spin-free excitation products.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExcitationClass {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
}
impl ExcitationClass {
    pub fn index(self) -> usize {
        self as usize
    }
    pub fn name(self) -> &'static str {
        ["A", "B", "C", "D", "E", "F", "G", "H"][self.index()]
    }
}

#[derive(Clone, Debug)]
pub struct IcCaspt2Options {
    pub frozen_core: usize,
    pub frozen_virtual: usize,
    pub metric_tol: f64,
    pub residual_tol: f64,
    pub max_iterations: usize,
    pub ipea_shift: f64,
    pub imaginary_shift: f64,
    pub real_shift: f64,
}
impl Default for IcCaspt2Options {
    fn default() -> Self {
        Self {
            frozen_core: 0,
            frozen_virtual: 0,
            metric_tol: 1e-10,
            residual_tol: 1e-10,
            max_iterations: 500,
            ipea_shift: 0.0,
            imaginary_shift: 0.0,
            real_shift: 0.0,
        }
    }
}
#[derive(Clone, Debug)]
pub struct IcCaspt2Result {
    pub correlation_energy: f64,
    pub uncorrected_energy: f64,
    pub class_energies: [f64; 8],
    pub raw_dimensions: [usize; 8],
    pub retained_dimensions: [usize; 8],
    pub metric_eigenvalues: Vec<Vec<f64>>,
    pub fock_reference_energy: f64,
    pub reference_weight: f64,
    pub iterations: usize,
    pub residual_norm: f64,
    pub amplitudes: Vec<f64>,
    pub coupling: Vec<f64>,
    /// Orthonormal contracted H0-E0 before real/imaginary regularization.
    pub zeroth_order: Matrix,
    pub first_order_determinants: Vec<Determinant>,
    pub first_order: Vec<f64>,
}

fn budget(bytes: Option<usize>) -> Result<(), HfError> {
    if bytes.is_none_or(|x| x > BUDGET) {
        Err(HfError::Unsupported {
            method: "CASPT2".into(),
            detail: "internally contracted working storage exceeds 256 MiB".into(),
        })
    } else {
        Ok(())
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
fn matvec(a: &Matrix, x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![0.0; a.rows];
    }
    a.data.chunks_exact(a.cols).map(|row| dot(row, x)).collect()
}

fn excite(vector: &Sparse, p: usize, q: usize) -> Sparse {
    let mut out = Sparse::new();
    for (&(a, b), &value) in vector {
        if let Some((next, sign)) = a.apply_single_excitation(p, q) {
            *out.entry((next, b)).or_default() += sign * value;
        }
        if let Some((next, sign)) = b.apply_single_excitation(p, q) {
            *out.entry((a, next)).or_default() += sign * value;
        }
    }
    out.retain(|_, v| *v != 0.0);
    out
}

struct Perturber {
    class: ExcitationClass,
    vector: Sparse,
    orbitals: Vec<(usize, usize)>,
}

fn make_perturbers(
    reference: &Sparse,
    space: &ResolvedActiveSpace,
    options: &IcCaspt2Options,
) -> Result<Vec<Perturber>, HfError> {
    let mut out = Vec::new();
    let mut used = 0usize;
    let mut push =
        |class, first: (usize, usize), second: Option<(usize, usize)>| -> Result<(), HfError> {
            budget(
                reference
                    .len()
                    .checked_mul(8 * 96)
                    .and_then(|x| x.checked_add(used)),
            )?;
            let intermediate = second.map(|(p, q)| excite(reference, p, q));
            let vector = excite(intermediate.as_ref().unwrap_or(reference), first.0, first.1);
            if vector.is_empty() {
                return Ok(());
            }
            used = used
                .checked_add(vector.len().saturating_mul(96) + 128)
                .ok_or_else(|| HfError::InvalidInput("CASPT2 perturber storage overflow".into()))?;
            budget(Some(used))?;
            let mut orbitals = vec![first];
            if let Some(pair) = second {
                orbitals.push(pair);
            }
            out.push(Perturber {
                class,
                vector,
                orbitals,
            });
            Ok(())
        };
    let core = &space.inactive[options.frozen_core..];
    let active = &space.active;
    let virtuals = &space.external[..space.external.len() - options.frozen_virtual];
    // In the usual nonempty CAS the trace of active E_uu already spans all
    // singles. Retain singles explicitly only for the empty-CAS HF limit.
    if active.is_empty() {
        for &a in virtuals {
            for &i in core {
                push(ExcitationClass::D, (a, i), None)?;
            }
        }
    }
    for &i in core {
        for &t in active {
            for &u in active {
                for &v in active {
                    push(ExcitationClass::A, (t, i), Some((u, v)))?;
                }
            }
        }
    }
    for &i in core {
        for &j in core {
            for &t in active {
                for &u in active {
                    if (t, i) <= (u, j) {
                        push(ExcitationClass::B, (t, i), Some((u, j)))?;
                    }
                }
            }
        }
    }
    for &a in virtuals {
        for &t in active {
            for &u in active {
                for &v in active {
                    push(ExcitationClass::C, (a, t), Some((u, v)))?;
                }
            }
        }
    }
    for &a in virtuals {
        for &i in core {
            for &t in active {
                for &u in active {
                    push(ExcitationClass::D, (a, i), Some((t, u)))?;
                    push(ExcitationClass::D, (t, i), Some((a, u)))?;
                }
            }
        }
    }
    for &i in core {
        for &j in core {
            for &t in active {
                for &a in virtuals {
                    push(ExcitationClass::E, (t, i), Some((a, j)))?;
                }
            }
        }
    }
    for &a in virtuals {
        for &b in virtuals {
            for &t in active {
                for &u in active {
                    if (a, t) <= (b, u) {
                        push(ExcitationClass::F, (a, t), Some((b, u)))?;
                    }
                }
            }
        }
    }
    for &a in virtuals {
        for &b in virtuals {
            for &i in core {
                for &t in active {
                    push(ExcitationClass::G, (a, i), Some((b, t)))?;
                }
            }
        }
    }
    for &a in virtuals {
        for &b in virtuals {
            for &i in core {
                for &j in core {
                    if (a, i) <= (b, j) {
                        push(ExcitationClass::H, (a, i), Some((b, j)))?;
                    }
                }
            }
        }
    }
    out.sort_by_key(|v| v.class.index());
    Ok(out)
}

/// Symmetric spin-averaged generalized Fock matrix; this is not the
/// density-indexed, nonsymmetric orbital-gradient generalized Fock matrix.
pub fn generalized_fock(h: &Matrix, g: &MoEriTensor, d: &Matrix) -> Matrix {
    let n = h.rows;
    let mut f = h.clone();
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    f[(p, q)] += d[(r, s)] * (g.get(p, q, r, s) - 0.5 * g.get(p, r, q, s));
                }
            }
        }
    }
    f
}

fn one_body_action(
    dets: &[Determinant],
    addresses: &HashMap<Determinant, usize>,
    f: &Matrix,
    x: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for (source, &(a, b)) in dets.iter().enumerate() {
        if x[source] == 0.0 {
            continue;
        }
        for q in a.occupied_orbitals() {
            for p in 0..f.rows {
                if let Some((next, sign)) = a.apply_single_excitation(p, q)
                    && let Some(&target) = addresses.get(&(next, b)) {
                        out[target] += f[(p, q)] * sign * x[source];
                    }
            }
        }
        for q in b.occupied_orbitals() {
            for p in 0..f.rows {
                if let Some((next, sign)) = b.apply_single_excitation(p, q)
                    && let Some(&target) = addresses.get(&(a, next)) {
                        out[target] += f[(p, q)] * sign * x[source];
                    }
            }
        }
    }
    out
}

/// Jacobi-preconditioned CG on normal equations handles indefinite shifted
/// Fock matrices too. Convergence is certified with the ORIGINAL residual.
fn solve_normal_pcg(
    a: &Matrix,
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, usize, f64), HfError> {
    let n = b.len();
    if n == 0 {
        return Ok((vec![], 0, 0.0));
    }
    let rhs = matvec(a, b).iter().map(|x| -x).collect::<Vec<_>>();
    let diagonal = (0..n)
        .map(|i| (0..n).map(|j| a[(j, i)].powi(2)).sum::<f64>().max(1e-24))
        .collect::<Vec<_>>();
    let mut x = vec![0.0; n];
    let mut r = rhs.clone();
    let mut z = r
        .iter()
        .zip(&diagonal)
        .map(|(r, d)| r / d)
        .collect::<Vec<_>>();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    for iteration in 0..=max_iter {
        let residual = matvec(a, &x)
            .iter()
            .zip(b)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        let maximum = norm(&residual);
        if maximum <= tol {
            return Ok((x, iteration, maximum));
        }
        if iteration == max_iter {
            break;
        }
        // Restart from a freshly evaluated residual periodically to control
        // loss of conjugacy in ill-conditioned, internally contracted spaces.
        if iteration % 25 == 0 && iteration != 0 {
            r = matvec(a, &residual).iter().map(|v| -v).collect();
            z = r.iter().zip(&diagonal).map(|(r, d)| r / d).collect();
            p = z.clone();
            rz = dot(&r, &z);
        }
        let ap = matvec(a, &matvec(a, &p));
        let pap = dot(&p, &ap);
        if !pap.is_finite() || pap <= 1e-300 || !rz.is_finite() {
            break;
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
            z[i] = r[i] / diagonal[i];
        }
        let next = dot(&r, &z);
        let beta = next / rz;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = next;
    }
    Err(HfError::PostHf {
        method: "CASPT2".into(),
        detail: format!(
            "PCG failed or singular intruder: original residual {:.3e}",
            norm(
                &matvec(a, &x)
                    .iter()
                    .zip(b)
                    .map(|(a, b)| a + b)
                    .collect::<Vec<_>>()
            )
        ),
    })
}

/// Prefer the unsquared symmetric equations, retaining the normal-equation
/// fallback when a nonpositive curvature proves that PCG is inapplicable.
fn solve_pcg(
    a: &Matrix,
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, usize, f64), HfError> {
    let n = b.len();
    if n == 0 {
        return Ok((vec![], 0, 0.0));
    }
    if (0..n).any(|i| a[(i, i)] <= 0.0) {
        return solve_normal_pcg(a, b, max_iter, tol);
    }
    let mut x = vec![0.0; n];
    let mut r = b.iter().map(|x| -x).collect::<Vec<_>>();
    let mut z = (0..n).map(|i| r[i] / a[(i, i)]).collect::<Vec<_>>();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    for iteration in 0..=max_iter {
        let exact = matvec(a, &x)
            .iter()
            .zip(b)
            .map(|(a, b)| -a - b)
            .collect::<Vec<_>>();
        let residual = norm(&exact);
        if residual <= tol {
            return Ok((x, iteration, residual));
        }
        if iteration == max_iter {
            break;
        }
        if iteration != 0 && iteration % 25 == 0 {
            r = exact;
            z = (0..n).map(|i| r[i] / a[(i, i)]).collect();
            p = z.clone();
            rz = dot(&r, &z);
        }
        let ap = matvec(a, &p);
        let pap = dot(&p, &ap);
        if pap <= 0.0 || !pap.is_finite() || !rz.is_finite() {
            let (x, count, error) = solve_normal_pcg(a, b, max_iter - iteration, tol)?;
            return Ok((x, count + iteration, error));
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
            z[i] = r[i] / a[(i, i)];
        }
        let next = dot(&r, &z);
        for i in 0..n {
            p[i] = z[i] + next / rz * p[i];
        }
        rz = next;
    }
    Err(HfError::PostHf {
        method: "CASPT2".into(),
        detail: format!("PCG did not converge in {max_iter} iterations"),
    })
}

/// Exact IC A-H equations from an active CI reference in the supplied MOs.
pub fn solve_ic_caspt2(
    h: &Matrix,
    g: &MoEriTensor,
    space: &ResolvedActiveSpace,
    ci: &[f64],
    one: &Matrix,
    options: &IcCaspt2Options,
) -> Result<IcCaspt2Result, HfError> {
    let n = h.rows;
    let na = space.active.len();
    if n > 64
        || h.cols != n
        || g.n_basis != n
        || one.rows != na
        || one.cols != na
        || space.n_alpha > na
        || space.n_beta > na
        || h.data.len() != n * n
        || one.data.len() != na * na
        || g.n_pairs != n * (n + 1) / 2
        || g.data.len() != g.n_pairs * g.n_pairs
        || space.permutation() != (0..n).collect::<Vec<_>>()
        || h.data
            .iter()
            .chain(&g.data)
            .chain(&one.data)
            .chain(ci)
            .any(|x| !x.is_finite())
        || !(0.0..1.0).contains(&options.metric_tol)
        || options.metric_tol == 0.0
        || !options.residual_tol.is_finite()
        || options.residual_tol <= 0.0
        || options.max_iterations == 0
        || options.frozen_core > space.inactive.len()
        || options.frozen_virtual > space.external.len()
        || [
            options.ipea_shift,
            options.imaginary_shift,
            options.real_shift,
        ]
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0)
        || (options.imaginary_shift != 0.0 && options.real_shift != 0.0)
    {
        return Err(HfError::InvalidInput(
            "invalid internally contracted CASPT2 problem or controls".into(),
        ));
    }
    let refs = cas_references(space, n, 1_000_000)?;
    if ci.len() != refs.len() || (norm(ci) - 1.0).abs() > 1e-8 {
        return Err(HfError::InvalidInput(
            "CASPT2 requires a normalized active CI vector".into(),
        ));
    }
    let reference = refs
        .iter()
        .copied()
        .zip(ci.iter().copied())
        .filter(|(_, x)| *x != 0.0)
        .collect::<Sparse>();
    let perturbers = make_perturbers(&reference, space, options)?;
    let mut d = Matrix::zeros(n, n);
    for &i in &space.inactive {
        d[(i, i)] = 2.0;
    }
    for (t, &p) in space.active.iter().enumerate() {
        for (u, &q) in space.active.iter().enumerate() {
            d[(p, q)] = one[(t, u)];
        }
    }
    let fock = generalized_fock(h, g, &d);
    let e0 = dot(&fock.data, &d.data);
    let mut set = refs.iter().copied().collect::<BTreeSet<_>>();
    for p in &perturbers {
        set.extend(p.vector.keys().copied());
    }
    let dets = set.into_iter().collect::<Vec<_>>();
    let rows = dets.len();
    let columns = perturbers.len();
    let sparse_bytes = perturbers
        .iter()
        .fold(refs.len().checked_mul(128), |total, p| {
            total.and_then(|x| x.checked_add(p.vector.len().checked_mul(96)? + 128))
        });
    // Account for primitive/orthogonal vectors, Fock-action scratch and all
    // dense metric/Fock work before allocating rectangular tensor storage.
    budget(
        rows.checked_mul(columns)
            .and_then(|x| x.checked_mul(40))
            .and_then(|x| {
                columns
                    .checked_mul(columns)?
                    .checked_mul(80)?
                    .checked_add(x)
            })
            .and_then(|x| x.checked_add(sparse_bytes?))
            .and_then(|x| x.checked_add(rows.checked_mul(128)?)),
    )?;
    let addresses = dets
        .iter()
        .enumerate()
        .map(|(i, &d)| (d, i))
        .collect::<HashMap<_, _>>();
    let mut raw_dimensions = [0usize; 8];
    for p in &perturbers {
        raw_dimensions[p.class.index()] += 1;
    }
    let mut retained = [0usize; 8];
    let mut metric_eigenvalues = Vec::new();
    let mut ipea_blocks = Vec::new();
    let mut orthogonal = Vec::<Vec<f64>>::new();
    let mut labels = Vec::new();
    let mut first = 0;
    for class in 0..8 {
        let size = raw_dimensions[class];
        if size == 0 {
            metric_eigenvalues.push(vec![]);
            ipea_blocks.push(Matrix::zeros(0, 0));
            continue;
        }
        let mut b = Matrix::zeros(rows, size);
        for (j, p) in perturbers[first..first + size].iter().enumerate() {
            for (det, &value) in &p.vector {
                b[(addresses[det], j)] = value;
            }
        }
        let metric = b.transpose().dot(&b);
        let (values, vectors) = metric.eigh();
        let maximum = values.iter().copied().fold(0.0, f64::max);
        let keep = (0..size)
            .filter(|&k| values[k] > options.metric_tol * maximum)
            .collect::<Vec<_>>();
        retained[class] = keep.len();
        metric_eigenvalues.push(values.clone());
        let mut x = Matrix::zeros(size, keep.len());
        for (j, &k) in keep.iter().enumerate() {
            for i in 0..size {
                x[(i, j)] = vectors[(i, k)] / values[k].sqrt();
            }
        }
        let q = b.dot(&x);
        let mut fq = Matrix::zeros(rows, keep.len());
        for j in 0..keep.len() {
            let v = (0..rows).map(|i| q[(i, j)]).collect::<Vec<_>>();
            let action = one_body_action(&dets, &addresses, &fock, &v);
            for i in 0..rows {
                fq[(i, j)] = action[i];
            }
        }
        let mut within = q.transpose().dot(&fq);
        let mut ipea = Matrix::zeros(keep.len(), keep.len());
        if options.ipea_shift != 0.0 {
            // Published diagonal-IC IPEA convention (Wouters et al., Eq. 48).
            // It is formed BEFORE canonical metric orthogonalization. The
            // convention and its orbital/basis dependence are explicit.
            let mut correction = Matrix::zeros(size, size);
            for (j, p) in perturbers[first..first + size].iter().enumerate() {
                let weight = p
                    .orbitals
                    .iter()
                    .map(|&(p, q)| 2.0 + d[(p, p)] - d[(q, q)])
                    .sum::<f64>()
                    * 0.5;
                correction[(j, j)] = options.ipea_shift * weight * metric[(j, j)];
            }
            ipea = x.transpose().dot(&correction).dot(&x);
            within = within.add(&ipea);
        }
        let (_, rotate) = within.eigh();
        ipea_blocks.push(rotate.transpose().dot(&ipea).dot(&rotate));
        let q = q.dot(&rotate);
        for j in 0..keep.len() {
            orthogonal.push((0..rows).map(|i| q[(i, j)]).collect());
            labels.push(class);
        }
        first += size;
    }
    let k = orthogonal.len();
    let mut vectors = Matrix::zeros(rows, k);
    for (j, q) in orthogonal.iter().enumerate() {
        for i in 0..rows {
            vectors[(i, j)] = q[i];
        }
    }
    let mut action = Matrix::zeros(rows, k);
    for (j, q) in orthogonal.iter().enumerate() {
        let fq = one_body_action(&dets, &addresses, &fock, q);
        for i in 0..rows {
            action[(i, j)] = fq[i];
        }
    }
    let mut a = vectors.transpose().dot(&action);
    for i in 0..k {
        a[(i, i)] -= e0;
    }
    let mut offset = 0;
    for block in &ipea_blocks {
        for i in 0..block.rows {
            for j in 0..block.cols {
                a[(offset + i, offset + j)] += block[(i, j)];
            }
        }
        offset += block.rows;
    }
    // Explicit symmetrization removes only contraction roundoff.
    a = a.add(&a.transpose()).scale(0.5);
    let hamiltonian = SubspaceHamiltonian::new(h, g, 0.0, dets)?;
    let mut c0 = vec![0.0; rows];
    for (det, &value) in &reference {
        c0[addresses[det]] = value;
    }
    let h0 = hamiltonian.apply(&c0)?;
    let coupling = orthogonal.iter().map(|q| dot(q, &h0)).collect::<Vec<_>>();
    // The symmetric scaled imaginary-shift equation avoids division by a
    // small diagonal denominator. Its D=0 limit has exactly zero amplitude.
    let scale = (0..k)
        .map(|i| {
            if options.imaginary_shift == 0.0 {
                1.0
            } else {
                a[(i, i)].abs().sqrt()
            }
        })
        .collect::<Vec<_>>();
    let mut shifted = a.clone();
    for i in 0..k {
        for j in 0..k {
            shifted[(i, j)] *= scale[i] * scale[j];
        }
    }
    for i in 0..k {
        shifted[(i, i)] += options.real_shift;
        if options.imaginary_shift != 0.0 {
            shifted[(i, i)] +=
                options.imaginary_shift.powi(2) * if a[(i, i)] < 0.0 { -1.0 } else { 1.0 };
        }
    }
    let rhs = coupling
        .iter()
        .zip(&scale)
        .map(|(v, s)| v * s)
        .collect::<Vec<_>>();
    let minimum_scale = scale
        .iter()
        .copied()
        .filter(|s| *s > 0.0)
        .fold(1.0, f64::min);
    let (solution, iterations, _) = solve_pcg(
        &shifted,
        &rhs,
        options.max_iterations,
        options.residual_tol * minimum_scale,
    )?;
    let amplitudes = solution
        .iter()
        .zip(&scale)
        .map(|(v, s)| v * s)
        .collect::<Vec<_>>();
    let ax = matvec(&a, &amplitudes);
    let residual_norm = norm(
        &(0..k)
            .map(|i| {
                let regularizer = if options.imaginary_shift == 0.0 {
                    options.real_shift * amplitudes[i]
                } else if a[(i, i)] != 0.0 {
                    options.imaginary_shift.powi(2) * (amplitudes[i] / a[(i, i)])
                } else {
                    -coupling[i] - ax[i]
                };
                ax[i] + coupling[i] + regularizer
            })
            .collect::<Vec<_>>(),
    );
    if !residual_norm.is_finite() || residual_norm > options.residual_tol {
        return Err(HfError::PostHf {
            method: "CASPT2".into(),
            detail: format!("shifted-equation residual {residual_norm:e} exceeds tolerance"),
        });
    }
    let mut energies = [0.0; 8];
    for i in 0..k {
        energies[labels[i]] += amplitudes[i] * (2.0 * coupling[i] + ax[i]);
    }
    let correlation_energy = energies.iter().sum();
    let first_order = matvec(&vectors, &amplitudes);
    Ok(IcCaspt2Result {
        correlation_energy,
        uncorrected_energy: dot(&amplitudes, &coupling),
        class_energies: energies,
        raw_dimensions,
        retained_dimensions: retained,
        metric_eigenvalues,
        fock_reference_energy: e0,
        reference_weight: 1.0 / (1.0 + dot(&amplitudes, &amplitudes)),
        iterations,
        residual_norm,
        amplitudes,
        coupling,
        zeroth_order: a,
        first_order_determinants: hamiltonian.determinants,
        first_order,
    })
}
