// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Spin-free one-electron X2C in the supplied contracted Gaussian space.
//! Positive-energy spectral projectors and analytic Frechet matrix responses
//! avoid electron/electron eigenvalue-gap denominators at degeneracy.

use crate::Matrix;

#[derive(Clone)]
pub(super) struct Spectrum {
    pub(super) values: Vec<f64>,
    vectors: Matrix,
}
#[derive(Clone, Copy)]
pub(super) enum Power {
    Sqrt,
    InverseSqrt,
}
impl Spectrum {
    pub(super) fn positive(a: &Matrix) -> Result<Self, String> {
        let (values, vectors) = a.eigh();
        if values.iter().any(|v| !v.is_finite() || *v <= 0.) {
            return Err("X2C matrix power requires a positive definite metric".into());
        }
        Ok(Self { values, vectors })
    }
    pub(super) fn power(&self, power: Power) -> Matrix {
        let mut u = self.vectors.clone();
        for p in 0..u.rows {
            for q in 0..u.cols {
                let root = self.values[q].sqrt();
                u[(p, q)] *= match power {
                    Power::Sqrt => root,
                    Power::InverseSqrt => 1. / root,
                };
            }
        }
        u.dot(&self.vectors.transpose())
    }
    pub(super) fn response(&self, tangent: &Matrix, power: Power) -> Matrix {
        let mut d = self.vectors.transpose().dot(tangent).dot(&self.vectors);
        for p in 0..d.rows {
            for q in 0..d.cols {
                let (a, b) = (self.values[p].sqrt(), self.values[q].sqrt());
                // Exact divided differences, including coincident eigenvalues.
                d[(p, q)] *= match power {
                    Power::Sqrt => 1. / (a + b),
                    Power::InverseSqrt => -1. / (a * b * (a + b)),
                };
            }
        }
        self.vectors.dot(&d).dot(&self.vectors.transpose())
    }
}

pub(super) fn transformed(x: &Matrix, a: &Matrix) -> Matrix {
    x.dot(a).dot(x)
}
pub(super) fn transformed_response(x: &Matrix, dx: &Matrix, a: &Matrix, da: &Matrix) -> Matrix {
    dx.dot(a)
        .dot(x)
        .add(&x.dot(da).dot(x))
        .add(&x.dot(a).dot(dx))
}
fn leading(a: &Matrix, n: usize) -> Matrix {
    Matrix::from_vec(
        n,
        n,
        (0..n)
            .flat_map(|p| (0..n).map(move |q| a[(p, q)]))
            .collect(),
    )
}
fn four_block(v: &Matrix, t: &Matrix, w: &Matrix, b: &Matrix, c2: f64) -> Matrix {
    let n = t.rows;
    let mut a = Matrix::zeros(2 * n, 2 * n);
    let off = t.dot(b);
    let lower = b.dot(&w.scale(1. / (4. * c2)).sub(t)).dot(b);
    for p in 0..n {
        for q in 0..n {
            a[(p, q)] = v[(p, q)];
            a[(p, n + q)] = off[(p, q)];
            a[(n + p, q)] = off[(q, p)];
            a[(n + p, n + q)] = lower[(p, q)];
        }
    }
    a
}
pub(super) fn validate(n: usize, matrices: &[&Matrix]) -> Result<(), String> {
    let size = n.checked_mul(n).ok_or("X2C matrix dimensions overflow")?;
    if n == 0
        || matrices.iter().any(|m| {
            m.rows != n
                || m.cols != n
                || m.data.len() != size
                || m.data.iter().any(|v| !v.is_finite())
        })
    {
        return Err("invalid X2C matrix dimensions or values".into());
    }
    for m in matrices {
        for p in 0..n {
            for q in 0..p {
                if (m[(p, q)] - m[(q, p)]).abs()
                    > 1e-10 * (1. + m[(p, q)].abs().max(m[(q, p)].abs()))
                {
                    return Err("X2C matrix inputs must be Hermitian".into());
                }
            }
        }
    }
    Ok(())
}

/// Analytic one-electron decoupling and its reusable linear response state.
/// Two-electron picture-change and spin-orbit contributions are not included.
pub struct SpinFreeX2cCore {
    pub hamiltonian: Matrix,
    pub electronic_gap: f64,
    s: Spectrum,
    sinv: Matrix,
    sroot: Matrix,
    t_ao: Matrix,
    v_ao: Matrix,
    w_ao: Matrix,
    t: Matrix,
    w: Matrix,
    tspectrum: Spectrum,
    small: Matrix,
    dirac: Matrix,
    roots: Vec<f64>,
    vectors: Matrix,
    projector: Matrix,
    large: Spectrum,
    large_inverse: Matrix,
    projected: Matrix,
    effective: Matrix,
    c2: f64,
}
impl SpinFreeX2cCore {
    /// S, T, V and W=<p chi|V|p chi> refer to the SAME AO basis and potential.
    /// No primitive decontraction or atomic approximation is performed.
    pub fn build(
        s: &Matrix,
        t: &Matrix,
        v: &Matrix,
        w: &Matrix,
        speed_of_light: f64,
        max_work_bytes: usize,
    ) -> Result<Self, String> {
        let n = s.rows;
        validate(n, &[s, t, v, w])?;
        let c2 = speed_of_light * speed_of_light;
        if !speed_of_light.is_finite()
            || speed_of_light <= 0.
            || !c2.is_finite()
            || c2 == 0.
            || !c2.recip().is_finite()
        {
            return Err("invalid X2C speed of light".into());
        }
        if 2048 * (n as u128).pow(2) > max_work_bytes as u128 {
            return Err("X2C matrix workspace exceeds max_work_bytes".into());
        }
        let sspectrum = Spectrum::positive(s)?;
        if sspectrum.values[0] < 1e-10 {
            return Err("X2C requires a full-rank overlap metric".into());
        }
        let sinv = sspectrum.power(Power::InverseSqrt);
        let sroot = sspectrum.power(Power::Sqrt);
        let (ta, va, wa) = (t.clone(), v.clone(), w.clone());
        let t = transformed(&sinv, t);
        let v = transformed(&sinv, v);
        let w = transformed(&sinv, w);
        let tspectrum = Spectrum::positive(&t)?;
        let small = tspectrum.power(Power::InverseSqrt).scale((2. * c2).sqrt());
        let dirac = four_block(&v, &t, &w, &small, c2);
        let (roots, vectors) = dirac.eigh();
        let electronic_gap = roots[n] - roots[n - 1];
        let scale = roots.iter().map(|v| v.abs()).fold(1., f64::max);
        if !electronic_gap.is_finite() || electronic_gap <= 64. * f64::EPSILON * scale {
            return Err("X2C positive/negative spectral gap is not resolved".into());
        }
        let mut occupied = Matrix::zeros(2 * n, n);
        for p in 0..2 * n {
            for q in 0..n {
                occupied[(p, q)] = vectors[(p, n + q)];
            }
        }
        let projector = occupied.dot(&occupied.transpose());
        let large = Spectrum::positive(&leading(&projector, n))?;
        let large_inverse = large.power(Power::InverseSqrt);
        // P A P depends on the positive subspace, not its internal eigenbasis.
        let projected = leading(&projector.dot(&dirac).dot(&projector), n);
        let effective = transformed(&large_inverse, &projected);
        let hamiltonian = transformed(&sroot, &effective);
        Ok(Self {
            hamiltonian,
            electronic_gap,
            s: sspectrum,
            sinv,
            sroot,
            t_ao: ta,
            v_ao: va,
            w_ao: wa,
            t,
            w,
            tspectrum,
            small,
            dirac,
            roots,
            vectors,
            projector,
            large,
            large_inverse,
            projected,
            effective,
            c2,
        })
    }

    /// Directional analytic derivative for arbitrary S/T/V/W tangents. Matrix
    /// powers use divided differences; the positive projector only couples
    /// opposite-energy sectors, so electronic degeneracies remain continuous.
    pub fn response(
        &self,
        ds: &Matrix,
        dt: &Matrix,
        dv: &Matrix,
        dw: &Matrix,
    ) -> Result<Matrix, String> {
        let n = self.sinv.rows;
        validate(n, &[ds, dt, dv, dw])?;
        let dx = self.s.response(ds, Power::InverseSqrt);
        let dr = self.s.response(ds, Power::Sqrt);
        let dt = transformed_response(&self.sinv, &dx, &self.t_ao, dt);
        let dv = transformed_response(&self.sinv, &dx, &self.v_ao, dv);
        let dw = transformed_response(&self.sinv, &dx, &self.w_ao, dw);
        let db = self
            .tspectrum
            .response(&dt, Power::InverseSqrt)
            .scale((2. * self.c2).sqrt());
        let base = self.w.scale(1. / (4. * self.c2)).sub(&self.t);
        let delta = dw.scale(1. / (4. * self.c2)).sub(&dt);
        let off = dt.dot(&self.small).add(&self.t.dot(&db));
        let lower = db
            .dot(&base)
            .dot(&self.small)
            .add(&self.small.dot(&delta).dot(&self.small))
            .add(&self.small.dot(&base).dot(&db));
        let mut da = Matrix::zeros(2 * n, 2 * n);
        for p in 0..n {
            for q in 0..n {
                da[(p, q)] = dv[(p, q)];
                da[(p, n + q)] = off[(p, q)];
                da[(n + p, q)] = off[(q, p)];
                da[(n + p, n + q)] = lower[(p, q)];
            }
        }
        let rotated = self.vectors.transpose().dot(&da).dot(&self.vectors);
        let mut dp = Matrix::zeros(2 * n, 2 * n);
        for a in 0..n {
            for i in n..2 * n {
                let value = rotated[(a, i)] / (self.roots[i] - self.roots[a]);
                dp[(a, i)] = value;
                dp[(i, a)] = value;
            }
        }
        let dp = self.vectors.dot(&dp).dot(&self.vectors.transpose());
        let dd = leading(&dp, n);
        let dlarge = self.large.response(&dd, Power::InverseSqrt);
        let dk = leading(
            &dp.dot(&self.dirac)
                .dot(&self.projector)
                .add(&self.projector.dot(&da).dot(&self.projector))
                .add(&self.projector.dot(&self.dirac).dot(&dp)),
            n,
        );
        let deffective = transformed_response(&self.large_inverse, &dlarge, &self.projected, &dk);
        Ok(transformed_response(
            &self.sroot,
            &dr,
            &self.effective,
            &deffective,
        ))
    }
}
