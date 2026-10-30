// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Metric/residual-certified complex RHF, UHF and GHF in a common magnetic gauge.

use super::{
    complex_matrix::{ComplexMatrix, spin_diagonal},
    ghf::FukutomeClass,
    magnetic::{magnetic_one_electron_integrals, magnetic_pair, origin_pair},
};
use crate::{
    Matrix,
    basis::{Atom, ContractedGaussian},
    derivatives::diff_cg_1st,
    integrals::contracted_overlap,
    scf::{
        compute_eri_tensor, compute_nuclear_repulsion, compute_one_electron_integrals, eri_index,
    },
};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComplexHfReference {
    Restricted,
    Unrestricted,
    Generalized,
}

#[derive(Clone, Debug)]
pub struct ComplexHfOptions {
    pub reference: ComplexHfReference,
    /// Fixes N_alpha-N_beta for UHF; restricted HF requires one.
    pub multiplicity: usize,
    pub magnetic_field: [f64; 3],
    /// None selects the nuclear centroid, transported with the nuclei.
    pub gauge_origin: Option<[f64; 3]>,
    /// Include B.S with the free-electron g factor equal to two.
    pub spin_zeeman: bool,
    pub max_iterations: usize,
    pub energy_tol: f64,
    pub residual_tol: f64,
    pub density_tol: f64,
    pub overlap_cutoff: f64,
    pub max_work_bytes: usize,
    /// Hermitian metric-idempotent spinor density (real, imaginary), alpha first.
    pub initial_density: Option<(Matrix, Matrix)>,
}
impl Default for ComplexHfOptions {
    fn default() -> Self {
        Self {
            reference: ComplexHfReference::Restricted,
            multiplicity: 1,
            magnetic_field: [0.; 3],
            gauge_origin: None,
            spin_zeeman: false,
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

#[derive(Clone, Debug)]
pub struct ComplexHFResult {
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_energy: f64,
    /// Spatial spin trace retained for compatibility.
    pub density_real: Matrix,
    pub density_imag: Matrix,
    pub spinor_density_real: Matrix,
    pub spinor_density_imag: Matrix,
    pub fock_real: Matrix,
    pub fock_imag: Matrix,
    pub coefficients_real: Matrix,
    pub coefficients_imag: Matrix,
    /// Occupied columns precede virtual columns in the explicit spinor frame.
    pub orbital_energies: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    /// AO representation norm of the imaginary spatial spin trace.
    pub current_magnitude: f64,
    pub fukutome_class: FukutomeClass,
    pub symmetry: super::symmetry::HfSymmetryAnalysis,
    pub residual_norm: f64,
    pub density_change: f64,
    pub n_electrons: usize,
    pub charge: i32,
    pub options: ComplexHfOptions,
    pub gauge_origin: [f64; 3],
    pub spin_expectations: [f64; 3],
    pub s_squared: f64,
}

fn add_spin_zeeman(h: &mut ComplexMatrix, s: &Matrix, field: [f64; 3]) {
    let n = s.rows;
    for p in 0..n {
        for q in 0..n {
            h.re[(p, q)] += 0.5 * field[2] * s[(p, q)];
            h.re[(n + p, n + q)] -= 0.5 * field[2] * s[(p, q)];
            h.re[(p, n + q)] += 0.5 * field[0] * s[(p, q)];
            h.re[(n + p, q)] += 0.5 * field[0] * s[(p, q)];
            h.im[(p, n + q)] -= 0.5 * field[1] * s[(p, q)];
            h.im[(n + p, q)] += 0.5 * field[1] * s[(p, q)];
        }
    }
}

fn fock(h: &ComplexMatrix, p: &ComplexMatrix, eri: &[f64], n: usize) -> ComplexMatrix {
    let j = Matrix::from_vec(
        n,
        n,
        (0..n * n)
            .into_par_iter()
            .map(|pq| {
                let (a, b) = (pq / n, pq % n);
                (0..n)
                    .flat_map(|r| {
                        (0..n).map(move |s| {
                            (p.re[(r, s)] + p.re[(n + r, n + s)]) * eri[eri_index(a, b, r, s)]
                        })
                    })
                    .sum()
            })
            .collect(),
    );
    let values: Vec<_> = (0..4 * n * n)
        .into_par_iter()
        .map(|pq| {
            let (a, b) = (pq / (2 * n), pq % (2 * n));
            let (u, v) = (a % n, b % n);
            let mut re = h.re[(a, b)];
            let mut im = h.im[(a, b)];
            if a / n == b / n {
                re += j[(u, v)];
            }
            for r in 0..n {
                for s in 0..n {
                    let g = eri[eri_index(u, r, v, s)];
                    re -= p.re[(a / n * n + r, b / n * n + s)] * g;
                    im -= p.im[(a / n * n + r, b / n * n + s)] * g;
                }
            }
            (re, im)
        })
        .collect();
    ComplexMatrix {
        re: Matrix::from_vec(2 * n, 2 * n, values.iter().map(|v| v.0).collect()),
        im: Matrix::from_vec(2 * n, 2 * n, values.iter().map(|v| v.1).collect()),
    }
}

/// Checked integral-level contraction, also useful for independent spin-frame tests.
pub fn build_complex_hf_fock(
    h_real: &Matrix,
    h_imag: &Matrix,
    p_real: &Matrix,
    p_imag: &Matrix,
    eri: &[f64],
) -> Result<(Matrix, Matrix), String> {
    let dim = h_real.rows;
    let n = dim / 2;
    let pairs = n * (n + 1) / 2;
    if dim == 0
        || !dim.is_multiple_of(2)
        || [h_real, h_imag, p_real, p_imag].iter().any(|m| {
            m.rows != dim
                || m.cols != dim
                || m.data.len() != dim * dim
                || m.data.iter().any(|v| !v.is_finite())
        })
        || eri.len() != pairs * (pairs + 1) / 2
        || eri.iter().any(|v| !v.is_finite())
    {
        return Err("invalid complex HF Fock inputs".into());
    }
    for (r, i) in [(h_real, h_imag), (p_real, p_imag)] {
        for a in 0..dim {
            for b in 0..dim {
                if (r[(a, b)] - r[(b, a)]).abs() > 1e-10 || (i[(a, b)] + i[(b, a)]).abs() > 1e-10 {
                    return Err("complex HF matrices must be Hermitian".into());
                }
            }
        }
    }
    let f = fock(
        &ComplexMatrix {
            re: h_real.clone(),
            im: h_imag.clone(),
        },
        &ComplexMatrix {
            re: p_real.clone(),
            im: p_imag.clone(),
        },
        eri,
        n,
    );
    Ok((f.re, f.im))
}

fn orbitals(
    f: &ComplexMatrix,
    reference: ComplexHfReference,
    na: usize,
    nb: usize,
) -> Result<ComplexMatrix, String> {
    let n = f.re.rows / 2;
    if reference == ComplexHfReference::Generalized {
        return Ok(f.eigh()?.1);
    }
    let mut blocks = Vec::new();
    for spin in 0..2 {
        let mut block = ComplexMatrix::zeros(n, n);
        for p in 0..n {
            for q in 0..n {
                if reference == ComplexHfReference::Restricted {
                    block.re[(p, q)] = 0.5 * (f.re[(p, q)] + f.re[(n + p, n + q)]);
                    block.im[(p, q)] = 0.5 * (f.im[(p, q)] + f.im[(n + p, n + q)]);
                } else {
                    block.re[(p, q)] = f.re[(spin * n + p, spin * n + q)];
                    block.im[(p, q)] = f.im[(spin * n + p, spin * n + q)];
                }
            }
        }
        blocks.push(block.eigh()?.1);
    }
    let columns: Vec<_> = [(0, 0, na), (1, 0, nb), (0, na, n), (1, nb, n)]
        .into_iter()
        .flat_map(|(s, a, b)| (a..b).map(move |i| (s, i)))
        .collect();
    let mut c = ComplexMatrix::zeros(2 * n, 2 * n);
    for (q, &(spin, old)) in columns.iter().enumerate() {
        for p in 0..n {
            c.re[(spin * n + p, q)] = blocks[spin].re[(p, old)];
            c.im[(spin * n + p, q)] = blocks[spin].im[(p, old)];
        }
    }
    Ok(c)
}

fn canonicalize(
    c: &ComplexMatrix,
    f: &ComplexMatrix,
    ne: usize,
) -> Result<(Vec<f64>, ComplexMatrix), String> {
    let fm = c.dagger().dot(f).dot(c);
    let n = c.re.cols;
    let mut u = ComplexMatrix::real(Matrix::identity(n));
    let mut values = vec![0.; n];
    for (start, end) in [(0, ne), (ne, n)] {
        if start == end {
            continue;
        }
        let mut block = ComplexMatrix::zeros(end - start, end - start);
        for p in start..end {
            for q in start..end {
                block.re[(p - start, q - start)] = fm.re[(p, q)];
                block.im[(p - start, q - start)] = fm.im[(p, q)];
            }
        }
        let (e, v) = block.eigh()?;
        for p in start..end {
            values[p] = e[p - start];
            for q in start..end {
                u.re[(p, q)] = v.re[(p - start, q - start)];
                u.im[(p, q)] = v.im[(p - start, q - start)];
            }
        }
    }
    Ok((values, c.dot(&u)))
}

/// Compatibility route: closed-shell complex RHF in an orbital field B_z.
pub fn run_complex_hf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    magnetic_field_z: f64,
    max_iter: usize,
    tol: f64,
) -> Result<ComplexHFResult, String> {
    let options = ComplexHfOptions {
        magnetic_field: [0., 0., magnetic_field_z],
        max_iterations: max_iter,
        energy_tol: tol,
        residual_tol: tol.sqrt().min(1e-8),
        density_tol: tol.sqrt().min(1e-8),
        ..Default::default()
    };
    run_complex_hf_with_options(atoms, basis, charge, &options)
}

pub fn run_complex_hf_with_options(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    options: &ComplexHfOptions,
) -> Result<ComplexHFResult, String> {
    let n = basis.len();
    if n == 0
        || atoms.is_empty()
        || options.max_iterations == 0
        || options.multiplicity == 0
        || [
            options.energy_tol,
            options.residual_tol,
            options.density_tol,
            options.overlap_cutoff,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
        || options.magnetic_field.iter().any(|v| !v.is_finite())
        || options
            .gauge_origin
            .is_some_and(|o| o.iter().any(|v| !v.is_finite()))
        || atoms
            .iter()
            .any(|a| !a.charge.is_finite() || a.position.iter().any(|x| !x.is_finite()))
        || basis.iter().any(|b| b.atom_index >= atoms.len())
    {
        return Err("invalid complex HF geometry, basis or controls".into());
    }
    let ne = atoms.iter().map(|a| a.charge).sum::<f64>().round() as i64 - i64::from(charge);
    let spin = options.multiplicity - 1;
    if ne <= 0
        || ne > 2 * n as i64
        || (options.reference != ComplexHfReference::Generalized
            && (spin > ne as usize || !(ne as usize + spin).is_multiple_of(2)))
        || (options.reference == ComplexHfReference::Restricted && options.multiplicity != 1)
    {
        return Err("incompatible complex HF electron count or multiplicity".into());
    }
    if options.spin_zeeman
        && options.reference != ComplexHfReference::Generalized
        && (options.magnetic_field[0] != 0. || options.magnetic_field[1] != 0.)
    {
        return Err("transverse spin-Zeeman coupling requires generalized HF".into());
    }
    let ne = ne as usize;
    let pairs = (n as u128) * (n as u128 + 1) / 2;
    if 4 * pairs * (pairs + 1) + 4096 * (n as u128).pow(2) > options.max_work_bytes as u128 {
        return Err("complex HF workspace exceeds max_work_bytes".into());
    }
    let e_nuc = compute_nuclear_repulsion(atoms);
    if !e_nuc.is_finite() {
        return Err("complex HF requires distinct nuclei".into());
    }
    let origin = options.gauge_origin.unwrap_or_else(|| {
        std::array::from_fn(|k| {
            atoms.iter().map(|a| a.position[k]).sum::<f64>() / atoms.len() as f64
        })
    });
    let (s, _, _, h0) = compute_one_electron_integrals(atoms, basis);
    let xs = s
        .canonical_orthogonalizer(options.overlap_cutoff)
        .map_err(|e| e.to_string())?;
    let rank = xs.cols;
    let na = if options.reference == ComplexHfReference::Generalized {
        ne.div_ceil(2)
    } else {
        (ne + spin) / 2
    };
    let nb = ne - na;
    if na > rank || nb > rank {
        return Err("complex HF occupations exceed retained overlap rank".into());
    }
    let x = ComplexMatrix::real(spin_diagonal(&xs));
    let sm = ComplexMatrix::real(spin_diagonal(&s));
    let (dia, orbital) = magnetic_one_electron_integrals(basis, options.magnetic_field, origin)?;
    let mut h = ComplexMatrix {
        re: spin_diagonal(&h0.add(&dia)),
        im: spin_diagonal(&orbital),
    };
    if options.spin_zeeman {
        add_spin_zeeman(&mut h, &s, options.magnetic_field);
    }
    let eri = compute_eri_tensor(basis);
    let mut c = x.dot(&orbitals(
        &x.dagger().dot(&h).dot(&x),
        options.reference,
        na,
        nb,
    )?);
    let mut p = c.density(ne);
    if let Some((re, im)) = &options.initial_density {
        if re.rows != 2 * n
            || re.cols != 2 * n
            || im.rows != 2 * n
            || im.cols != 2 * n
            || re.data.len() != 4 * n * n
            || im.data.len() != 4 * n * n
            || re.data.iter().chain(&im.data).any(|v| !v.is_finite())
        {
            return Err("invalid complex HF initial density dimensions/values".into());
        }
        let trial = ComplexMatrix {
            re: re.clone(),
            im: im.clone(),
        };
        let projected = x.dagger().dot(&sm).dot(&trial).dot(&sm).dot(&x);
        if trial.add(&trial.dagger(), -1.).norm() > 1e-8
            || trial.dot(&sm).dot(&trial).add(&trial, -1.).norm() > 1e-8
            || (trial.dot(&sm).re.trace() - ne as f64).abs() > 1e-8
            || x.dot(&projected).dot(&x.dagger()).add(&trial, -1.).norm() > 1e-8
        {
            return Err(
                "initial density must be a retained metric-idempotent Hermitian determinant".into(),
            );
        }
        c = x.dot(&orbitals(
            &ComplexMatrix::zeros(2 * rank, 2 * rank).add(&projected, -1.),
            options.reference,
            na,
            nb,
        )?);
        p = c.density(ne);
        if p.add(&trial, -1.).norm() > 1e-8 {
            return Err("initial determinant violates the selected spin restriction".into());
        }
    }
    let mut diis = crate::diis::CdiisEngine::new(8);
    let mut previous = None;
    let mut density_change = f64::INFINITY;
    let mut residual_norm = f64::INFINITY;
    for iteration in 1..=options.max_iterations {
        let f = fock(&h, &p, &eri, n);
        let energy = e_nuc
            + 0.5
                * (p.re
                    .data
                    .iter()
                    .zip(&h.re.data)
                    .zip(&f.re.data)
                    .map(|((p, h), f)| p * (h + f))
                    .sum::<f64>()
                    + p.im
                        .data
                        .iter()
                        .zip(&h.im.data)
                        .zip(&f.im.data)
                        .map(|((p, h), f)| p * (h + f))
                        .sum::<f64>());
        let mut residual = x
            .dagger()
            .dot(&f.dot(&p).dot(&sm).add(&sm.dot(&p).dot(&f), -1.))
            .dot(&x);
        // Project onto the variational tangent space of the selected restriction.
        if options.reference != ComplexHfReference::Generalized {
            for a in 0..rank {
                for b in 0..rank {
                    residual.re[(a, rank + b)] = 0.;
                    residual.re[(rank + a, b)] = 0.;
                    residual.im[(a, rank + b)] = 0.;
                    residual.im[(rank + a, b)] = 0.;
                    if options.reference == ComplexHfReference::Restricted {
                        let re = 0.5 * (residual.re[(a, b)] + residual.re[(rank + a, rank + b)]);
                        let im = 0.5 * (residual.im[(a, b)] + residual.im[(rank + a, rank + b)]);
                        residual.re[(a, b)] = re;
                        residual.re[(rank + a, rank + b)] = re;
                        residual.im[(a, b)] = im;
                        residual.im[(rank + a, rank + b)] = im;
                    }
                }
            }
        }
        residual_norm = residual.norm();
        if previous.is_some_and(|e: f64| (energy - e).abs() <= options.energy_tol)
            && residual_norm <= options.residual_tol
            && density_change <= options.density_tol
        {
            let (energies, c) = canonicalize(&c, &f, ne)?;
            let mut real = Matrix::zeros(n, n);
            let mut imag = real.clone();
            for a in 0..n {
                for b in 0..n {
                    real[(a, b)] = p.re[(a, b)] + p.re[(n + a, n + b)];
                    imag[(a, b)] = p.im[(a, b)] + p.im[(n + a, n + b)];
                }
            }
            let current = imag.data.iter().map(|v| v * v).sum::<f64>().sqrt();
            let symmetry = super::symmetry::classify_complex_hf_symmetry(&p.re, &p.im, &s, 1e-8)?;
            let class = symmetry.class;
            let mut spin = [0.; 3];
            for a in 0..n {
                for b in 0..n {
                    spin[0] += 0.5 * (p.re[(a, n + b)] + p.re[(n + a, b)]) * s[(b, a)];
                    spin[1] += 0.5 * (p.im[(n + a, b)] - p.im[(a, n + b)]) * s[(b, a)];
                    spin[2] += 0.5 * (p.re[(a, b)] - p.re[(n + a, n + b)]) * s[(b, a)];
                }
            }
            let ds = ComplexMatrix {
                re: real.clone(),
                im: imag.clone(),
            }
            .dot(&ComplexMatrix::real(s.clone()));
            let ss =
                spin.iter().map(|x| x * x).sum::<f64>() + ne as f64 - 0.5 * ds.dot(&ds).re.trace();
            return Ok(ComplexHFResult {
                total_energy: energy,
                electronic_energy: energy - e_nuc,
                nuclear_energy: e_nuc,
                density_real: real,
                density_imag: imag,
                spinor_density_real: p.re,
                spinor_density_imag: p.im,
                fock_real: f.re,
                fock_imag: f.im,
                coefficients_real: c.re,
                coefficients_imag: c.im,
                orbital_energies: energies,
                iterations: iteration,
                converged: true,
                current_magnitude: current,
                fukutome_class: class,
                symmetry,
                residual_norm,
                density_change,
                n_electrons: ne,
                charge,
                options: options.clone(),
                gauge_origin: origin,
                spin_expectations: spin,
                s_squared: ss,
            });
        }
        previous = Some(energy);
        let orthofock = x.dagger().dot(&f).dot(&x);
        diis.push(orthofock.pack(), residual.pack());
        let next = ComplexMatrix::unpack(diis.extrapolate().unwrap_or_else(|| orthofock.pack()));
        c = x.dot(&orbitals(&next, options.reference, na, nb)?);
        let next = c.density(ne);
        density_change = next.add(&p, -1.).norm() / (2 * n) as f64;
        p = next;
    }
    Err(format!(
        "complex HF did not converge in {} iterations: residual {residual_norm:e}, density change {density_change:e}",
        options.max_iterations
    ))
}

/// Variational analytic first nuclear derivative, including complex exchange,
/// Pulay, local/nonlocal ECP and the chosen magnetic-origin transport.
pub fn compute_complex_hf_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    result: &ComplexHFResult,
) -> Result<Vec<[f64; 3]>, String> {
    let n = basis.len();
    if !result.converged
        || atoms.is_empty()
        || result.coefficients_real.rows != 2 * n
        || result.coefficients_real.cols != 2 * n
    {
        return Err("complex HF gradient requires a converged full-rank overlap space".into());
    }
    let p = ComplexMatrix {
        re: result.spinor_density_real.clone(),
        im: result.spinor_density_imag.clone(),
    };
    let f = ComplexMatrix {
        re: result.fock_real.clone(),
        im: result.fock_imag.clone(),
    };
    let weighted = p.dot(&f).dot(&p);
    let mut w = Matrix::zeros(n, n);
    for a in 0..n {
        for b in 0..n {
            w[(a, b)] = weighted.re[(a, b)] + weighted.re[(n + a, n + b)];
        }
    }
    let total = &result.density_real;
    let mut gradient = crate::gradient::compute_gradient_from_density_with_eri_contraction(
        atoms,
        basis,
        total,
        &w,
        |mu, nu, la, si| {
            let mut value = (if la == si { 2. } else { 4. }) * total[(mu, nu)] * total[(la, si)];
            for a in 0..2 {
                for b in 0..2 {
                    let (u, v) = (a * n + mu, b * n + la);
                    let (r, s) = (a * n + nu, b * n + si);
                    value -= 2. * (p.re[(u, v)] * p.re[(r, s)] + p.im[(u, v)] * p.im[(r, s)]);
                    if la != si {
                        let (u, v) = (a * n + mu, b * n + si);
                        let (r, s) = (a * n + nu, b * n + la);
                        value -= 2. * (p.re[(u, v)] * p.re[(r, s)] + p.im[(u, v)] * p.im[(r, s)]);
                    }
                }
            }
            value
        },
    );
    let field = result.options.magnetic_field;
    let origin = result.gauge_origin;
    for (atom, row) in gradient.iter_mut().enumerate() {
        for (axis, g) in row.iter_mut().enumerate() {
            let mut dh = ComplexMatrix::zeros(2 * n, 2 * n);
            let mut ds = Matrix::zeros(n, n);
            for mu in 0..n {
                for nu in 0..n {
                    let mut dr = 0.;
                    let mut di = 0.;
                    let mut overlap = 0.;
                    if basis[mu].atom_index == atom {
                        for term in diff_cg_1st(&basis[mu], axis).terms {
                            let (r, i) = magnetic_pair(&term, &basis[nu], field, origin);
                            dr += r;
                            di += i;
                            overlap += contracted_overlap(&term, &basis[nu]);
                        }
                    }
                    if basis[nu].atom_index == atom {
                        for term in diff_cg_1st(&basis[nu], axis).terms {
                            let (r, i) = magnetic_pair(&basis[mu], &term, field, origin);
                            dr += r;
                            di += i;
                            overlap += contracted_overlap(&basis[mu], &term);
                        }
                    }
                    if result.options.gauge_origin.is_none() {
                        let (r, i) = origin_pair(&basis[mu], &basis[nu], field, origin, axis);
                        dr += r / atoms.len() as f64;
                        di += i / atoms.len() as f64;
                    }
                    ds[(mu, nu)] = overlap;
                    for spin in 0..2 {
                        dh.re[(spin * n + mu, spin * n + nu)] = dr;
                        dh.im[(spin * n + mu, spin * n + nu)] = di;
                    }
                }
            }
            if result.options.spin_zeeman {
                add_spin_zeeman(&mut dh, &ds, field);
            }
            *g +=
                p.re.data
                    .iter()
                    .zip(&dh.re.data)
                    .map(|(p, h)| p * h)
                    .sum::<f64>()
                    + p.im
                        .data
                        .iter()
                        .zip(&dh.im.data)
                        .map(|(p, h)| p * h)
                        .sum::<f64>();
        }
    }
    Ok(gradient)
}
