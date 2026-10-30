// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Spin-free second-order Douglas-Kroll-Hess (DKH2) one-electron Hamiltonian in
//! the supplied contracted Gaussian space (matrix formulation).
//!
//! In the orthonormal eigenbasis of the kinetic matrix (`p_i^2 = 2 t_i`) the
//! free-particle Foldy-Wouthuysen transformation is diagonal with
//! `E_i = c sqrt(p_i^2 + c^2)`, `A_i = sqrt((E_i + c^2) / (2 E_i))`,
//! `K_i = c / (E_i + c^2)`. With `V` and `W = <p chi|V|p chi>`:
//!
//! * `E1 = A (V + K W K) A` (even, first order),
//! * `O1 = A (K W P^-1 - V K P) A` (odd, first order),
//! * `W1` solves `E W1 + W1 E = O1`,
//! * `E2 = (W1 O1^T + O1 W1^T) / 2`,
//!
//! and `h_DKH2 = (E - c^2) + E1 + E2`. All quantities are matrix functions of
//! the kinetic matrix, so their directional derivatives follow from exact
//! divided differences (Daleckii-Krein) and a second Sylvester solve; coincident
//! kinetic eigenvalues are therefore handled without gap denominators.
//! Two-electron picture-change and spin-orbit terms are not included.

use super::x2c_core::{transformed, transformed_response, validate, Power, Spectrum};
use crate::Matrix;

/// Eigen-decomposition of a positive matrix used to evaluate matrix functions
/// and their Frechet derivatives.
struct MatrixFunctions {
    values: Vec<f64>,
    vectors: Matrix,
}

#[derive(Clone, Copy)]
enum Scalar {
    /// `E - c^2`, evaluated without cancellation.
    Kinetic,
    /// `E = c sqrt(2t + c^2)`
    Energy,
    /// `A = sqrt((E + c^2) / (2E))`
    A,
    /// `K = c / (E + c^2)`
    K,
    /// `K p`, `p = sqrt(2t)`
    KP,
    /// `1 / p`
    InvP,
}

fn evaluate(f: Scalar, t: f64, c: f64) -> (f64, f64) {
    let c2 = c * c;
    let e = c * (2. * t + c2).sqrt();
    let de = c2 / e;
    let p = (2. * t).sqrt();
    match f {
        Scalar::Kinetic => (2. * t * c2 / (e + c2), de),
        Scalar::Energy => (e, de),
        Scalar::A => {
            let a = ((e + c2) / (2. * e)).sqrt();
            (a, -c2 / (4. * a * e * e) * de)
        }
        Scalar::K => (c / (e + c2), -c / ((e + c2) * (e + c2)) * de),
        Scalar::KP => {
            let k = c / (e + c2);
            let dk = -c / ((e + c2) * (e + c2)) * de;
            (k * p, dk * p + k / p)
        }
        Scalar::InvP => (1. / p, -1. / (p * p * p)),
    }
}

impl MatrixFunctions {
    fn new(a: &Matrix) -> Result<Self, String> {
        let (values, vectors) = a.eigh();
        if values.iter().any(|v| !v.is_finite() || *v <= 0.) {
            return Err("DKH2 requires a positive definite kinetic matrix".into());
        }
        Ok(Self { values, vectors })
    }

    fn rotate(&self, diag: impl Fn(usize, usize) -> f64, m: Option<&Matrix>) -> Matrix {
        let n = self.values.len();
        let mut d = match m {
            Some(m) => self.vectors.transpose().dot(m).dot(&self.vectors),
            None => Matrix::identity(n),
        };
        for p in 0..n {
            for q in 0..n {
                d[(p, q)] *= diag(p, q);
            }
        }
        self.vectors.dot(&d).dot(&self.vectors.transpose())
    }

    fn apply(&self, f: Scalar, c: f64) -> Matrix {
        let vals: Vec<f64> = self.values.iter().map(|&t| evaluate(f, t, c).0).collect();
        self.rotate(|p, q| if p == q { vals[p] } else { 0. }, None)
    }

    /// Frechet derivative of `f(T)` along `dt` (exact divided differences).
    fn response(&self, f: Scalar, c: f64, dt: &Matrix) -> Matrix {
        let fv: Vec<(f64, f64)> = self.values.iter().map(|&t| evaluate(f, t, c)).collect();
        let t = &self.values;
        self.rotate(
            |p, q| {
                let gap = t[p] - t[q];
                if gap.abs() > 1e-4 * (1. + t[p].abs().max(t[q].abs())) {
                    (fv[p].0 - fv[q].0) / gap
                } else {
                    evaluate(f, 0.5 * (t[p] + t[q]), c).1
                }
            },
            Some(dt),
        )
    }

    /// Solves `E X + X E = rhs` where `E = f_Energy(T)` shares the eigenvectors of `T`.
    fn sylvester(&self, c: f64, rhs: &Matrix) -> Matrix {
        let e: Vec<f64> = self.values.iter().map(|&t| evaluate(Scalar::Energy, t, c).0).collect();
        self.rotate(|p, q| 1. / (e[p] + e[q]), Some(rhs))
    }
}

/// DKH2 core Hamiltonian and the state needed for its analytic response.
pub struct SpinFreeDkh2Core {
    pub hamiltonian: Matrix,
    s: Spectrum,
    sinv: Matrix,
    sroot: Matrix,
    t_ao: Matrix,
    v_ao: Matrix,
    w_ao: Matrix,
    v: Matrix,
    w: Matrix,
    funcs: MatrixFunctions,
    a: Matrix,
    k: Matrix,
    kp: Matrix,
    invp: Matrix,
    m: Matrix,
    nmat: Matrix,
    odd: Matrix,
    w1: Matrix,
    effective: Matrix,
    c: f64,
}

impl SpinFreeDkh2Core {
    /// `S`, `T`, `V` and `W = <p chi|V|p chi>` refer to the same AO basis and potential.
    pub fn build(s: &Matrix, t: &Matrix, v: &Matrix, w: &Matrix, speed_of_light: f64, max_work_bytes: usize) -> Result<Self, String> {
        let n = s.rows;
        validate(n, &[s, t, v, w])?;
        let c = speed_of_light;
        if !c.is_finite() || c <= 0. || !(c * c).is_finite() {
            return Err("invalid DKH2 speed of light".into());
        }
        if 512 * (n as u128).pow(2) > max_work_bytes as u128 {
            return Err("DKH2 matrix workspace exceeds max_work_bytes".into());
        }
        let sspectrum = Spectrum::positive(s)?;
        if sspectrum.values[0] < 1e-10 {
            return Err("DKH2 requires a full-rank overlap metric".into());
        }
        let sinv = sspectrum.power(Power::InverseSqrt);
        let sroot = sspectrum.power(Power::Sqrt);
        let to = transformed(&sinv, t);
        let vo = transformed(&sinv, v);
        let wo = transformed(&sinv, w);
        let funcs = MatrixFunctions::new(&to)?;
        let kinetic = funcs.apply(Scalar::Kinetic, c);
        let a = funcs.apply(Scalar::A, c);
        let k = funcs.apply(Scalar::K, c);
        let kp = funcs.apply(Scalar::KP, c);
        let invp = funcs.apply(Scalar::InvP, c);
        let m = vo.add(&k.dot(&wo).dot(&k));
        let e1 = a.dot(&m).dot(&a);
        let nmat = k.dot(&wo).dot(&invp).sub(&vo.dot(&kp));
        let odd = a.dot(&nmat).dot(&a);
        let w1 = funcs.sylvester(c, &odd);
        let e2 = w1.dot(&odd.transpose()).add(&odd.dot(&w1.transpose())).scale(0.5);
        let effective = kinetic.add(&e1).add(&e2);
        let hamiltonian = transformed(&sroot, &effective);
        Ok(Self {
            hamiltonian,
            s: sspectrum,
            sinv,
            sroot,
            t_ao: t.clone(),
            v_ao: v.clone(),
            w_ao: w.clone(),
            v: vo,
            w: wo,
            funcs,
            a,
            k,
            kp,
            invp,
            m,
            nmat,
            odd,
            w1,
            effective,
            c,
        })
    }

    /// Directional analytic derivative of the AO DKH2 Hamiltonian for arbitrary S/T/V/W tangents.
    pub fn response(&self, ds: &Matrix, dt: &Matrix, dv: &Matrix, dw: &Matrix) -> Result<Matrix, String> {
        let n = self.sinv.rows;
        validate(n, &[ds, dt, dv, dw])?;
        let c = self.c;
        let dx = self.s.response(ds, Power::InverseSqrt);
        let dr = self.s.response(ds, Power::Sqrt);
        let dto = transformed_response(&self.sinv, &dx, &self.t_ao, dt);
        let dvo = transformed_response(&self.sinv, &dx, &self.v_ao, dv);
        let dwo = transformed_response(&self.sinv, &dx, &self.w_ao, dw);
        let f = &self.funcs;
        let dkin = f.response(Scalar::Kinetic, c, &dto);
        let de = f.response(Scalar::Energy, c, &dto);
        let da = f.response(Scalar::A, c, &dto);
        let dk = f.response(Scalar::K, c, &dto);
        let dkp = f.response(Scalar::KP, c, &dto);
        let dinvp = f.response(Scalar::InvP, c, &dto);
        let (a, k) = (&self.a, &self.k);
        let triple = |x: &Matrix, y: &Matrix, z: &Matrix| x.dot(y).dot(z);
        let dm = dvo
            .add(&triple(&dk, &self.w, k))
            .add(&triple(k, &dwo, k))
            .add(&triple(k, &self.w, &dk));
        let de1 = triple(&da, &self.m, a).add(&triple(a, &dm, a)).add(&triple(a, &self.m, &da));
        let dn = triple(&dk, &self.w, &self.invp)
            .add(&triple(k, &dwo, &self.invp))
            .add(&triple(k, &self.w, &dinvp))
            .sub(&dvo.dot(&self.kp))
            .sub(&self.v.dot(&dkp));
        let dodd = triple(&da, &self.nmat, a).add(&triple(a, &dn, a)).add(&triple(a, &self.nmat, &da));
        let rhs = dodd.sub(&de.dot(&self.w1)).sub(&self.w1.dot(&de));
        let dw1 = f.sylvester(c, &rhs);
        let de2 = dw1
            .dot(&self.odd.transpose())
            .add(&self.w1.dot(&dodd.transpose()))
            .add(&dodd.dot(&self.w1.transpose()))
            .add(&self.odd.dot(&dw1.transpose()))
            .scale(0.5);
        let deff = dkin.add(&de1).add(&de2);
        Ok(transformed_response(&self.sroot, &dr, &self.effective, &deff))
    }
}
