// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Molecular integral evaluation using the McMurchie-Davidson algorithm.
//! Evaluates Overlap (S), Kinetic (T), Nuclear Attraction (V), and Electron Repulsion Integrals (ERI).

use std::f64::consts::PI;
use crate::basis::{Atom, ContractedGaussian};

/// Boys function F_n(T) = \int_0^1 t^{2n} exp(-T t^2) dt.
///
/// For T <= `BOYS_T_SWITCH` the values are interpolated with a 7th-order Taylor expansion
/// around a pretabulated grid. The grid itself is generated from the convergent series
/// F_m(T) = e^{-T} \sum_i (2T)^i / [(2m+1)(2m+3)...(2m+2i+1)] at the highest tabulated order
/// followed by stable downward recursion F_{m-1} = (2T F_m + e^{-T}) / (2m-1).
/// For T > `BOYS_T_SWITCH` (> `BOYS_MAX_M`) the exact closed form of F_0 and the upward recursion
/// F_n = ((2n-1) F_{n-1} - e^{-T}) / (2T) are used, which is numerically stable for T > n.
pub const BOYS_MAX_M: usize = 40;
const BOYS_TAYLOR_ORDER: usize = 7;
const BOYS_TABLE_M: usize = BOYS_MAX_M + BOYS_TAYLOR_ORDER;
const BOYS_GRID_STEP: f64 = 0.05;
const BOYS_T_SWITCH: f64 = 50.0;
const BOYS_GRID_SIZE: usize = 1002; // covers T in [0, 50.05]

static BOYS_TABLE: std::sync::LazyLock<Vec<[f64; BOYS_TABLE_M + 1]>> = std::sync::LazyLock::new(|| {
    (0..BOYS_GRID_SIZE)
        .map(|idx| {
            let t = idx as f64 * BOYS_GRID_STEP;
            let mut row = [0.0; BOYS_TABLE_M + 1];
            boys_reference_downward(BOYS_TABLE_M, t, &mut row);
            row
        })
        .collect()
});

/// Accurate (but slow) Boys evaluation used to build the interpolation table.
fn boys_reference_downward(max_n: usize, t: f64, out: &mut [f64]) {
    let exp_t = (-t).exp();
    // Series for F_{max_n}(T); every term is positive so there is no cancellation.
    let mut term = 1.0 / (2 * max_n + 1) as f64;
    let mut sum = term;
    let mut i = 1usize;
    loop {
        term *= 2.0 * t / (2 * max_n + 2 * i + 1) as f64;
        sum += term;
        if term < sum * 1e-17 || i > 10_000 {
            break;
        }
        i += 1;
    }
    out[max_n] = exp_t * sum;
    for m in (0..max_n).rev() {
        out[m] = (2.0 * t * out[m + 1] + exp_t) / (2 * m + 1) as f64;
    }
}

pub const TWO_PI_POW_2_5: f64 = 34.986836655249726164477;
pub const PI_POW_1_5: f64 = 5.56832799683170784528;
pub const HALF_SQRT_PI: f64 = 0.88622692545275801365;

pub fn boys(n: usize, t: f64) -> f64 {
    let mut out = [0.0; BOYS_MAX_M + 1];
    boys_array(n, t, &mut out[..=n]);
    out[n]
}

/// Evaluates Boys functions F_0(T), F_1(T), ..., F_{max_n}(T) simultaneously (`max_n <= BOYS_MAX_M`).
pub fn boys_array(max_n: usize, t: f64, out: &mut [f64]) {
    assert!(max_n <= BOYS_MAX_M, "Boys order {} exceeds supported maximum {}", max_n, BOYS_MAX_M);
    assert!(out.len() > max_n);
    if t > BOYS_T_SWITCH {
        let exp_t = (-t).exp();
        let inv_2t = 0.5 / t;
        out[0] = HALF_SQRT_PI * libm::erf(t.sqrt()) / t.sqrt();
        for n in 1..=max_n {
            out[n] = ((2 * n - 1) as f64 * out[n - 1] - exp_t) * inv_2t;
        }
        return;
    }

    let idx = (t * (1.0 / BOYS_GRID_STEP)).round() as usize;
    let t0 = idx as f64 * BOYS_GRID_STEP;
    let dt = t - t0;
    let row = &BOYS_TABLE[idx];

    // d^k F_n / dT^k = (-1)^k F_{n+k}
    let mut coeff = [0.0; BOYS_TAYLOR_ORDER + 1];
    coeff[0] = 1.0;
    for k in 1..=BOYS_TAYLOR_ORDER {
        coeff[k] = coeff[k - 1] * (-dt) / k as f64;
    }
    for n in 0..=max_n {
        let mut acc = 0.0;
        for k in (0..=BOYS_TAYLOR_ORDER).rev() {
            acc += coeff[k] * row[n + k];
        }
        out[n] = acc;
    }
}

pub const MAX_HERMITE_L: usize = 6;
pub const MAX_HERMITE_T: usize = 12;
pub const MAX_4C_HERMITE_T: usize = BOYS_MAX_M;

#[derive(Clone)]
pub struct HermiteBuffer {
    pub data: [[[f64; MAX_HERMITE_T + 1]; MAX_HERMITE_L + 1]; MAX_HERMITE_L + 1],
    dynamic: Option<Vec<Vec<Vec<f64>>>>,
}

impl Default for HermiteBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl HermiteBuffer {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            data: [[[0.0; MAX_HERMITE_T + 1]; MAX_HERMITE_L + 1]; MAX_HERMITE_L + 1],
            dynamic: None,
        }
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize, t: usize) -> f64 {
        if let Some(data) = &self.dynamic { data[i][j][t] } else { self.data[i][j][t] }
    }
}

#[inline(always)]
pub fn compute_hermite_coeffs(
    la: usize,
    lb: usize,
    alpha: f64,
    beta: f64,
    a_coord: f64,
    b_coord: f64,
    out: &mut HermiteBuffer,
) {
    if la > MAX_HERMITE_L || lb > MAX_HERMITE_L {
        out.dynamic = Some(hermite_expansion_coefficients(la, lb, alpha, beta, a_coord, b_coord));
        return;
    }
    out.dynamic = None;
    let p = alpha + beta;
    let inv_p = 1.0 / p;
    let p_coord = (alpha * a_coord + beta * b_coord) * inv_p;
    let x_pa = p_coord - a_coord;
    let x_pb = p_coord - b_coord;
    let mu = (alpha * beta) * inv_p;

    let max_t = la + lb;
    let inv_2p = 0.5 * inv_p;

    for i in 0..=la {
        for j in 0..=lb {
            for t in 0..=(i + j) {
                out.data[i][j][t] = 0.0;
            }
        }
    }

    let diff = a_coord - b_coord;
    out.data[0][0][0] = (-mu * diff * diff).exp();

    for i in 0..=la {
        for j in 0..=lb {
            let max_ij = i + j;
            for t in 0..=max_ij {
                if i == 0 && j == 0 {
                    continue;
                }
                if i > 0 {
                    let term1 = if t > 0 { out.data[i - 1][j][t - 1] * inv_2p } else { 0.0 };
                    let term2 = x_pa * out.data[i - 1][j][t];
                    let term3 = if t < max_t { (t as f64 + 1.0) * out.data[i - 1][j][t + 1] } else { 0.0 };
                    out.data[i][j][t] = term1 + term2 + term3;
                } else {
                    let term1 = if t > 0 { out.data[i][j - 1][t - 1] * inv_2p } else { 0.0 };
                    let term2 = x_pb * out.data[i][j - 1][t];
                    let term3 = if t < max_t { (t as f64 + 1.0) * out.data[i][j - 1][t + 1] } else { 0.0 };
                    out.data[i][j][t] = term1 + term2 + term3;
                }
            }
        }
    }
}

/// 1D McMurchie-Davidson expansion coefficients E_t^{i, j}
/// E has shape [la + 1][lb + 1][la + lb + 1]
pub fn hermite_expansion_coefficients(
    la: usize,
    lb: usize,
    alpha: f64,
    beta: f64,
    a_coord: f64,
    b_coord: f64,
) -> Vec<Vec<Vec<f64>>> {
    if la <= MAX_HERMITE_L && lb <= MAX_HERMITE_L {
        let mut buf = HermiteBuffer::new();
        compute_hermite_coeffs(la, lb, alpha, beta, a_coord, b_coord, &mut buf);
        let max_t = la + lb;
        let mut e = vec![vec![vec![0.0; max_t + 1]; lb + 1]; la + 1];
        for i in 0..=la {
            for j in 0..=lb {
                for t in 0..=max_t {
                    e[i][j][t] = buf.data[i][j][t];
                }
            }
        }
        e
    } else {
        let p = alpha + beta;
        let p_coord = (alpha * a_coord + beta * b_coord) / p;
        let x_pa = p_coord - a_coord;
        let x_pb = p_coord - b_coord;
        let mu = (alpha * beta) / p;

        let max_t = la + lb;
        let mut e = vec![vec![vec![0.0; max_t + 1]; lb + 1]; la + 1];

        e[0][0][0] = (-mu * (a_coord - b_coord).powi(2)).exp();

        for i in 0..=la {
            for j in 0..=lb {
                for t in 0..=(i + j) {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    if i > 0 {
                        let term1 = if t > 0 { e[i - 1][j][t - 1] / (2.0 * p) } else { 0.0 };
                        let term2 = x_pa * e[i - 1][j][t];
                        let term3 = if t < max_t { (t as f64 + 1.0) * e[i - 1][j][t + 1] } else { 0.0 };
                        e[i][j][t] = term1 + term2 + term3;
                    } else {
                        let term1 = if t > 0 { e[i][j - 1][t - 1] / (2.0 * p) } else { 0.0 };
                        let term2 = x_pb * e[i][j - 1][t];
                        let term3 = if t < max_t { (t as f64 + 1.0) * e[i][j - 1][t + 1] } else { 0.0 };
                        e[i][j][t] = term1 + term2 + term3;
                    }
                }
            }
        }
        e
    }
}

#[inline(always)]
pub fn coulomb_auxiliary_integrals_flat(
    max_t:usize,max_u:usize,max_v:usize,p:f64,p_minus_c:[f64;3],r_out:&mut[f64],
) {
    coulomb_auxiliary_integrals_total(max_t,max_u,max_v,max_t+max_u+max_v,p,p_minus_c,r_out)
}

pub(crate) fn coulomb_auxiliary_integrals_total(
    max_t: usize,
    max_u: usize,
    max_v: usize,
    max_n: usize,
    p: f64,
    p_minus_c: [f64; 3],
    r_out: &mut [f64],
) {
    let x_pc = p_minus_c[0];
    let y_pc = p_minus_c[1];
    let z_pc = p_minus_c[2];
    let r2 = x_pc * x_pc + y_pc * y_pc + z_pc * z_pc;

    let stride_v = max_n + 1;
    let stride_u = (max_v + 1) * stride_v;
    let stride_t = (max_u + 1) * stride_u;

    let mut f_vals = [0.0; BOYS_MAX_M + 1];
    boys_array(max_n, p * r2, &mut f_vals[0..=max_n]);

    let neg_2p = -2.0 * p;
    let mut neg_2p_pow = 1.0;
    for n in 0..=max_n {
        r_out[n] = neg_2p_pow * f_vals[n];
        neg_2p_pow *= neg_2p;
    }

    for t in 0..max_t.min(max_n) {
        let t_curr = t * stride_t;
        let t_prev = if t > 0 { (t - 1) * stride_t } else { 0 };
        let t_next = (t + 1) * stride_t;
        let t_f64 = t as f64;
        let limit = max_n - t - 1;
        if t > 0 {
            for n in 0..=limit {
                r_out[t_next + n] = t_f64 * r_out[t_prev + n + 1] + x_pc * r_out[t_curr + n + 1];
            }
        } else {
            for n in 0..=limit {
                r_out[t_next + n] = x_pc * r_out[t_curr + n + 1];
            }
        }
    }

    for t in 0..=max_t.min(max_n) {
        let t_offset = t * stride_t;
        for u in 0..max_u.min(max_n-t) {
            let u_curr = t_offset + u * stride_u;
            let u_prev = if u > 0 { t_offset + (u - 1) * stride_u } else { 0 };
            let u_next = t_offset + (u + 1) * stride_u;
            let u_f64 = u as f64;
            let limit = max_n - t - u - 1;
            if u > 0 {
                for n in 0..=limit {
                    r_out[u_next + n] = u_f64 * r_out[u_prev + n + 1] + y_pc * r_out[u_curr + n + 1];
                }
            } else {
                for n in 0..=limit {
                    r_out[u_next + n] = y_pc * r_out[u_curr + n + 1];
                }
            }
        }
    }

    for t in 0..=max_t.min(max_n) {
        let t_offset = t * stride_t;
        for u in 0..=max_u.min(max_n-t) {
            let tu_offset = t_offset + u * stride_u;
            for v in 0..max_v.min(max_n-t-u) {
                let v_curr = tu_offset + v * stride_v;
                let v_prev = if v > 0 { tu_offset + (v - 1) * stride_v } else { 0 };
                let v_next = tu_offset + (v + 1) * stride_v;
                let v_f64 = v as f64;
                let limit = max_n - t - u - v - 1;
                if v > 0 {
                    for n in 0..=limit {
                        r_out[v_next + n] = v_f64 * r_out[v_prev + n + 1] + z_pc * r_out[v_curr + n + 1];
                    }
                } else {
                    for n in 0..=limit {
                        r_out[v_next + n] = z_pc * r_out[v_curr + n + 1];
                    }
                }
            }
        }
    }
}

/// Hermite Coulomb auxiliary integrals R_{t, u, v}^n(p, P - C)
/// Returns a 4D array indexed by [t][u][v][n]
pub fn coulomb_auxiliary_integrals(
    max_t: usize,
    max_u: usize,
    max_v: usize,
    p: f64,
    p_minus_c: [f64; 3],
) -> Vec<Vec<Vec<Vec<f64>>>> {
    let max_n = max_t + max_u + max_v;
    let x_pc = p_minus_c[0];
    let y_pc = p_minus_c[1];
    let z_pc = p_minus_c[2];
    let r2 = x_pc * x_pc + y_pc * y_pc + z_pc * z_pc;

    let mut r = vec![vec![vec![vec![0.0; max_n + 1]; max_v + 1]; max_u + 1]; max_t + 1];

    let mut f_vals = vec![0.0; max_n + 1];
    boys_array(max_n, p * r2, &mut f_vals);
    let mut neg_2p_pow = 1.0;
    let neg_2p = -2.0 * p;
    for n in 0..=max_n {
        r[0][0][0][n] = neg_2p_pow * f_vals[n];
        neg_2p_pow *= neg_2p;
    }

    for t in 0..max_t {
        for n in 0..=(max_n - t - 1) {
            let term1 = if t > 0 { t as f64 * r[t - 1][0][0][n + 1] } else { 0.0 };
            let term2 = x_pc * r[t][0][0][n + 1];
            r[t + 1][0][0][n] = term1 + term2;
        }
    }

    for t in 0..=max_t {
        for u in 0..max_u {
            for n in 0..=(max_n - t - u - 1) {
                let term1 = if u > 0 { u as f64 * r[t][u - 1][0][n + 1] } else { 0.0 };
                let term2 = y_pc * r[t][u][0][n + 1];
                r[t][u + 1][0][n] = term1 + term2;
            }
        }
    }

    for t in 0..=max_t {
        for u in 0..=max_u {
            for v in 0..max_v {
                for n in 0..=(max_n - t - u - v - 1) {
                    let term1 = if v > 0 { v as f64 * r[t][u][v - 1][n + 1] } else { 0.0 };
                    let term2 = z_pc * r[t][u][v][n + 1];
                    r[t][u][v + 1][n] = term1 + term2;
                }
            }
        }
    }

    r
}

/// Evaluates primitive overlap integral <g_a | g_b>
pub fn primitive_overlap(
    alpha: f64,
    la: [usize; 3],
    a: [f64; 3],
    beta: f64,
    lb: [usize; 3],
    b: [f64; 3],
) -> f64 {
    let p = alpha + beta;
    let mut ex = HermiteBuffer::new();
    let mut ey = HermiteBuffer::new();
    let mut ez = HermiteBuffer::new();
    compute_hermite_coeffs(la[0], lb[0], alpha, beta, a[0], b[0], &mut ex);
    compute_hermite_coeffs(la[1], lb[1], alpha, beta, a[1], b[1], &mut ey);
    compute_hermite_coeffs(la[2], lb[2], alpha, beta, a[2], b[2], &mut ez);

    let inv_p = 1.0 / p;
    ex.get(la[0], lb[0], 0) * ey.get(la[1], lb[1], 0) * ez.get(la[2], lb[2], 0) * (PI_POW_1_5 * inv_p * inv_p.sqrt())
}

/// Evaluates primitive kinetic energy integral <g_a | -1/2 \nabla^2 | g_b>
pub fn primitive_kinetic(
    alpha: f64,
    la: [usize; 3],
    a: [f64; 3],
    beta: f64,
    lb: [usize; 3],
    b: [f64; 3],
) -> f64 {
    let p = alpha + beta;
    let inv_p = 1.0 / p;
    let mut ex = HermiteBuffer::new();
    let mut ey = HermiteBuffer::new();
    let mut ez = HermiteBuffer::new();
    compute_hermite_coeffs(la[0], lb[0] + 2, alpha, beta, a[0], b[0], &mut ex);
    compute_hermite_coeffs(la[1], lb[1] + 2, alpha, beta, a[1], b[1], &mut ey);
    compute_hermite_coeffs(la[2], lb[2] + 2, alpha, beta, a[2], b[2], &mut ez);

    let sx = |i: usize, j: usize| ex.get(i, j, 0);
    let sy = |i: usize, j: usize| ey.get(i, j, 0);
    let sz = |i: usize, j: usize| ez.get(i, j, 0);

    let calc_t1d = |la_i: usize, lb_i: usize, s1d: &dyn Fn(usize, usize) -> f64| -> f64 {
        let term1 = beta * (2.0 * lb_i as f64 + 1.0) * s1d(la_i, lb_i);
        let term2 = -2.0 * beta * beta * s1d(la_i, lb_i + 2);
        let term3 = if lb_i >= 2 {
            -0.5 * (lb_i * (lb_i - 1)) as f64 * s1d(la_i, lb_i - 2)
        } else {
            0.0
        };
        term1 + term2 + term3
    };

    let tx = calc_t1d(la[0], lb[0], &sx);
    let ty = calc_t1d(la[1], lb[1], &sy);
    let tz = calc_t1d(la[2], lb[2], &sz);

    let prefactor = PI_POW_1_5 * inv_p * inv_p.sqrt();
    let total = tx * sy(la[1], lb[1]) * sz(la[2], lb[2])
        + sx(la[0], lb[0]) * ty * sz(la[2], lb[2])
        + sx(la[0], lb[0]) * sy(la[1], lb[1]) * tz;

    prefactor * total
}

/// Evaluates primitive nuclear attraction integral <g_a | -Z_c / |r - C| | g_b>
pub fn primitive_nuclear_attraction(
    alpha: f64,
    la: [usize; 3],
    a: [f64; 3],
    beta: f64,
    lb: [usize; 3],
    b: [f64; 3],
    c: [f64; 3],
    charge: f64,
) -> f64 {
    let p = alpha + beta;
    let inv_p = 1.0 / p;
    let p_coord = [
        (alpha * a[0] + beta * b[0]) * inv_p,
        (alpha * a[1] + beta * b[1]) * inv_p,
        (alpha * a[2] + beta * b[2]) * inv_p,
    ];
    let p_minus_c = [
        p_coord[0] - c[0],
        p_coord[1] - c[1],
        p_coord[2] - c[2],
    ];

    let mut ex = HermiteBuffer::new();
    let mut ey = HermiteBuffer::new();
    let mut ez = HermiteBuffer::new();
    compute_hermite_coeffs(la[0], lb[0], alpha, beta, a[0], b[0], &mut ex);
    compute_hermite_coeffs(la[1], lb[1], alpha, beta, a[1], b[1], &mut ey);
    compute_hermite_coeffs(la[2], lb[2], alpha, beta, a[2], b[2], &mut ez);

    let max_t = la[0] + lb[0];
    let max_u = la[1] + lb[1];
    let max_v = la[2] + lb[2];
    let max_n = max_t + max_u + max_v;

    let _stride_n = 1;
    let stride_v = max_n + 1;
    let stride_u = (max_v + 1) * stride_v;
    let stride_t = (max_u + 1) * stride_u;
    let total_r_size = (max_t + 1) * stride_t;

    let mut r_buf = [0.0; 8192];
    coulomb_auxiliary_integrals_flat(max_t, max_u, max_v, p, p_minus_c, &mut r_buf[..total_r_size]);

    let mut sum = 0.0;
    for t in 0..=max_t {
        let et = ex.get(la[0], lb[0], t);
        if et == 0.0 { continue; }
        let t_offset = t * stride_t;
        for u in 0..=max_u {
            let eu = ey.get(la[1], lb[1], u);
            if eu == 0.0 { continue; }
            let tu_offset = t_offset + u * stride_u;
            for v in 0..=max_v {
                let ev = ez.get(la[2], lb[2], v);
                if ev == 0.0 { continue; }
                let r_idx = tu_offset + v * stride_v;
                sum += et * eu * ev * r_buf[r_idx];
            }
        }
    }

    -charge * (2.0 * PI * inv_p) * sum
}

#[inline(always)]
pub fn primitive_eri_fast(
    alpha: f64, la: [usize; 3], a: [f64; 3],
    beta:  f64, lb: [usize; 3], b: [f64; 3],
    gamma: f64, lc: [usize; 3], c: [f64; 3],
    delta: f64, ld: [usize; 3], d: [f64; 3],
    ex_ab: &HermiteBuffer, ey_ab: &HermiteBuffer, ez_ab: &HermiteBuffer,
    ex_cd: &HermiteBuffer, ey_cd: &HermiteBuffer, ez_cd: &HermiteBuffer,
    r_buf: &mut [f64],
) -> f64 {
    let p = alpha + beta;
    let q = gamma + delta;
    let inv_p = 1.0 / p;
    let inv_q = 1.0 / q;
    let alpha_prime = (p * q) / (p + q);

    let p_coord = [
        (alpha * a[0] + beta * b[0]) * inv_p,
        (alpha * a[1] + beta * b[1]) * inv_p,
        (alpha * a[2] + beta * b[2]) * inv_p,
    ];
    let q_coord = [
        (gamma * c[0] + delta * d[0]) * inv_q,
        (gamma * c[1] + delta * d[1]) * inv_q,
        (gamma * c[2] + delta * d[2]) * inv_q,
    ];
    let p_minus_q = [
        p_coord[0] - q_coord[0],
        p_coord[1] - q_coord[1],
        p_coord[2] - q_coord[2],
    ];

    let t_ab_max = la[0] + lb[0];
    let u_ab_max = la[1] + lb[1];
    let v_ab_max = la[2] + lb[2];

    let t_cd_max = lc[0] + ld[0];
    let u_cd_max = lc[1] + ld[1];
    let v_cd_max = lc[2] + ld[2];

    let max_t = t_ab_max + t_cd_max;
    let max_u = u_ab_max + u_cd_max;
    let max_v = v_ab_max + v_cd_max;
    let max_n = max_t + max_u + max_v;

    let _stride_n = 1;
    let stride_v = max_n + 1;
    let stride_u = (max_v + 1) * stride_v;
    let stride_t = (max_u + 1) * stride_u;
    let total_r_size = (max_t + 1) * stride_t;

    if total_r_size <= r_buf.len() {
        coulomb_auxiliary_integrals_flat(max_t, max_u, max_v, alpha_prime, p_minus_q, &mut r_buf[..total_r_size]);
        evaluate_eri_contraction(
            la, lb, lc, ld,
            t_ab_max, u_ab_max, v_ab_max,
            t_cd_max, u_cd_max, v_cd_max,
            stride_t, stride_u, stride_v,
            ex_ab, ey_ab, ez_ab,
            ex_cd, ey_cd, ez_cd,
            &r_buf[..total_r_size],
            p, q,
        )
    } else {
        let mut dynamic_r = vec![0.0; total_r_size];
        coulomb_auxiliary_integrals_flat(max_t, max_u, max_v, alpha_prime, p_minus_q, &mut dynamic_r);
        evaluate_eri_contraction(
            la, lb, lc, ld,
            t_ab_max, u_ab_max, v_ab_max,
            t_cd_max, u_cd_max, v_cd_max,
            stride_t, stride_u, stride_v,
            ex_ab, ey_ab, ez_ab,
            ex_cd, ey_cd, ez_cd,
            &dynamic_r,
            p, q,
        )
    }
}

#[inline(always)]
fn evaluate_eri_contraction(
    la: [usize; 3], lb: [usize; 3], lc: [usize; 3], ld: [usize; 3],
    t_ab_max: usize, u_ab_max: usize, v_ab_max: usize,
    t_cd_max: usize, u_cd_max: usize, v_cd_max: usize,
    stride_t: usize, stride_u: usize, stride_v: usize,
    ex_ab: &HermiteBuffer, ey_ab: &HermiteBuffer, ez_ab: &HermiteBuffer,
    ex_cd: &HermiteBuffer, ey_cd: &HermiteBuffer, ez_cd: &HermiteBuffer,
    r_buf: &[f64],
    p: f64, q: f64,
) -> f64 {
    let mut sum = 0.0;
    for t in 0..=t_ab_max {
        let et = ex_ab.get(la[0], lb[0], t);
        if et == 0.0 { continue; }
        for u in 0..=u_ab_max {
            let eu = ey_ab.get(la[1], lb[1], u);
            if eu == 0.0 { continue; }
            for v in 0..=v_ab_max {
                let ev = ez_ab.get(la[2], lb[2], v);
                if ev == 0.0 { continue; }
                let et_eu_ev = et * eu * ev;

                for tp in 0..=t_cd_max {
                    let etp = ex_cd.get(lc[0], ld[0], tp);
                    if etp == 0.0 { continue; }
                    let t_tot = t + tp;
                    let t_offset = t_tot * stride_t;

                    for up in 0..=u_cd_max {
                        let eup = ey_cd.get(lc[1], ld[1], up);
                        if eup == 0.0 { continue; }
                        let u_tot = u + up;
                        let tu_offset = t_offset + u_tot * stride_u;

                        for vp in 0..=v_cd_max {
                            let evp = ez_cd.get(lc[2], ld[2], vp);
                            if evp == 0.0 { continue; }
                            let v_tot = v + vp;
                            let r_idx = tu_offset + v_tot * stride_v;

                            let sign = if (tp + up + vp) % 2 == 1 { -1.0 } else { 1.0 };
                            sum += et_eu_ev * (sign * etp * eup * evp) * r_buf[r_idx];
                        }
                    }
                }
            }
        }
    }

    let prefactor = TWO_PI_POW_2_5 / (p * q * (p + q).sqrt());
    prefactor * sum
}

/// Evaluates primitive 2-electron repulsion integral (g_a g_b | g_c g_d)
pub fn primitive_eri(
    alpha: f64, la: [usize; 3], a: [f64; 3],
    beta:  f64, lb: [usize; 3], b: [f64; 3],
    gamma: f64, lc: [usize; 3], c: [f64; 3],
    delta: f64, ld: [usize; 3], d: [f64; 3],
) -> f64 {
    let mut ex_ab = HermiteBuffer::new();
    let mut ey_ab = HermiteBuffer::new();
    let mut ez_ab = HermiteBuffer::new();
    compute_hermite_coeffs(la[0], lb[0], alpha, beta, a[0], b[0], &mut ex_ab);
    compute_hermite_coeffs(la[1], lb[1], alpha, beta, a[1], b[1], &mut ey_ab);
    compute_hermite_coeffs(la[2], lb[2], alpha, beta, a[2], b[2], &mut ez_ab);

    let mut ex_cd = HermiteBuffer::new();
    let mut ey_cd = HermiteBuffer::new();
    let mut ez_cd = HermiteBuffer::new();
    compute_hermite_coeffs(lc[0], ld[0], gamma, delta, c[0], d[0], &mut ex_cd);
    compute_hermite_coeffs(lc[1], ld[1], gamma, delta, c[1], d[1], &mut ey_cd);
    compute_hermite_coeffs(lc[2], ld[2], gamma, delta, c[2], d[2], &mut ez_cd);

    let mut r_buf = [0.0; 8192];
    primitive_eri_fast(
        alpha, la, a,
        beta, lb, b,
        gamma, lc, c,
        delta, ld, d,
        &ex_ab, &ey_ab, &ez_ab,
        &ex_cd, &ey_cd, &ez_cd,
        &mut r_buf,
    )
}

/// Evaluates contracted overlap integral <cg_a | cg_b>
pub fn contracted_overlap(a: &ContractedGaussian, b: &ContractedGaussian) -> f64 {
    let mut val = 0.0;
    for pa in &a.primitives {
        for pb in &b.primitives {
            val += pa.coefficient * pb.coefficient * primitive_overlap(pa.exponent, a.l, a.center, pb.exponent, b.l, b.center);
        }
    }
    val
}

/// Evaluates contracted kinetic energy integral <cg_a | -1/2 \nabla^2 | cg_b>
pub fn contracted_kinetic(a: &ContractedGaussian, b: &ContractedGaussian) -> f64 {
    let mut val = 0.0;
    for pa in &a.primitives {
        for pb in &b.primitives {
            val += pa.coefficient * pb.coefficient * primitive_kinetic(pa.exponent, a.l, a.center, pb.exponent, b.l, b.center);
        }
    }
    val
}

/// Evaluates contracted nuclear attraction integral for all atoms <cg_a | \sum_C -Z_C/|r-C| | cg_b>
pub fn contracted_nuclear_attraction(a: &ContractedGaussian, b: &ContractedGaussian, atoms: &[Atom]) -> f64 {
    let mut total_val = 0.0;
    let mut ex = HermiteBuffer::new();
    let mut ey = HermiteBuffer::new();
    let mut ez = HermiteBuffer::new();
    let mut r_buf = [0.0; 8192];

    for pa in &a.primitives {
        for pb in &b.primitives {
            let coeff = pa.coefficient * pb.coefficient;
            if coeff.abs() < 1e-15 { continue; }

            let p = pa.exponent + pb.exponent;
            let inv_p = 1.0 / p;
            let p_coord = [
                (pa.exponent * a.center[0] + pb.exponent * b.center[0]) * inv_p,
                (pa.exponent * a.center[1] + pb.exponent * b.center[1]) * inv_p,
                (pa.exponent * a.center[2] + pb.exponent * b.center[2]) * inv_p,
            ];

            compute_hermite_coeffs(a.l[0], b.l[0], pa.exponent, pb.exponent, a.center[0], b.center[0], &mut ex);
            compute_hermite_coeffs(a.l[1], b.l[1], pa.exponent, pb.exponent, a.center[1], b.center[1], &mut ey);
            compute_hermite_coeffs(a.l[2], b.l[2], pa.exponent, pb.exponent, a.center[2], b.center[2], &mut ez);

            let max_t = a.l[0] + b.l[0];
            let max_u = a.l[1] + b.l[1];
            let max_v = a.l[2] + b.l[2];
            let max_n = max_t + max_u + max_v;

            let stride_v = max_n + 1;
            let stride_u = (max_v + 1) * stride_v;
            let stride_t = (max_u + 1) * stride_u;
            let total_r_size = (max_t + 1) * stride_t;

            let prefactor = 2.0 * PI * inv_p;

            for atom in atoms {
                let p_minus_c = [
                    p_coord[0] - atom.position[0],
                    p_coord[1] - atom.position[1],
                    p_coord[2] - atom.position[2],
                ];

                coulomb_auxiliary_integrals_flat(max_t, max_u, max_v, p, p_minus_c, &mut r_buf[..total_r_size]);

                let mut sum = 0.0;
                for it in 0..=max_t {
                    let e_t = ex.get(a.l[0], b.l[0], it);
                    if e_t == 0.0 { continue; }
                    for iu in 0..=max_u {
                        let e_u = ey.get(a.l[1], b.l[1], iu);
                        if e_u == 0.0 { continue; }
                        for iv in 0..=max_v {
                            let e_v = ez.get(a.l[2], b.l[2], iv);
                            if e_v == 0.0 { continue; }
                            let idx = it * stride_t + iu * stride_u + iv * stride_v;
                            sum += e_t * e_u * e_v * r_buf[idx];
                        }
                    }
                }
                total_val += coeff * (-atom.charge * prefactor * sum);
            }
        }
    }
    total_val
}

/// Common low-angular-momentum pairs stay inline; higher angular momenta grow on demand.
#[derive(Clone, Debug)]
pub enum HermiteAxis {
    Inline([f64; MAX_HERMITE_T + 1]),
    Dynamic(Vec<f64>),
}

impl HermiteAxis {
    fn from_fn(max: usize, f: impl Fn(usize) -> f64) -> Self {
        if max <= MAX_HERMITE_T {
            let mut values = [0.0; MAX_HERMITE_T + 1];
            for (i, v) in values.iter_mut().enumerate().take(max + 1) { *v = f(i); }
            Self::Inline(values)
        } else { Self::Dynamic((0..=max).map(f).collect()) }
    }
}

impl std::ops::Deref for HermiteAxis {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        match self { Self::Inline(v) => v, Self::Dynamic(v) => v }
    }
}

#[derive(Clone, Debug)]
pub struct PrimitivePairData {
    pub alpha: f64,
    pub beta: f64,
    pub p: f64,
    pub inv_p: f64,
    pub p_coord: [f64; 3],
    pub coeff: f64,
    pub t_max: usize,
    pub u_max: usize,
    pub v_max: usize,
    pub ex: HermiteAxis,
    pub ey: HermiteAxis,
    pub ez: HermiteAxis,
}

#[derive(Clone, Debug, Default)]
pub struct BasisPairData {
    pub primitives: Vec<PrimitivePairData>,
}

impl BasisPairData {
    pub fn new(a: &ContractedGaussian, b: &ContractedGaussian) -> Self {
        let mut primitives = Vec::with_capacity(a.primitives.len() * b.primitives.len());
        let mut ex_buf = HermiteBuffer::new();
        let mut ey_buf = HermiteBuffer::new();
        let mut ez_buf = HermiteBuffer::new();

        let dx = a.center[0] - b.center[0];
        let dy = a.center[1] - b.center[1];
        let dz = a.center[2] - b.center[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;

        for pa in &a.primitives {
            for pb in &b.primitives {
                let coeff = pa.coefficient * pb.coefficient;
                if coeff.abs() < 1e-15 {
                    continue;
                }
                let p = pa.exponent + pb.exponent;
                let inv_p = 1.0 / p;
                let mu = (pa.exponent * pb.exponent) * inv_p;
                let exp_factor = (-mu * dist_sq).exp();
                if (coeff * exp_factor).abs() < 1e-13 {
                    continue;
                }

                let p_coord = [
                    (pa.exponent * a.center[0] + pb.exponent * b.center[0]) * inv_p,
                    (pa.exponent * a.center[1] + pb.exponent * b.center[1]) * inv_p,
                    (pa.exponent * a.center[2] + pb.exponent * b.center[2]) * inv_p,
                ];

                compute_hermite_coeffs(a.l[0], b.l[0], pa.exponent, pb.exponent, a.center[0], b.center[0], &mut ex_buf);
                compute_hermite_coeffs(a.l[1], b.l[1], pa.exponent, pb.exponent, a.center[1], b.center[1], &mut ey_buf);
                compute_hermite_coeffs(a.l[2], b.l[2], pa.exponent, pb.exponent, a.center[2], b.center[2], &mut ez_buf);

                let t_max = a.l[0] + b.l[0];
                let u_max = a.l[1] + b.l[1];
                let v_max = a.l[2] + b.l[2];

                let ex = HermiteAxis::from_fn(t_max, |t| ex_buf.get(a.l[0], b.l[0], t));
                let ey = HermiteAxis::from_fn(u_max, |u| ey_buf.get(a.l[1], b.l[1], u));
                let ez = HermiteAxis::from_fn(v_max, |v| ez_buf.get(a.l[2], b.l[2], v));

                primitives.push(PrimitivePairData {
                    alpha: pa.exponent,
                    beta: pb.exponent,
                    p,
                    inv_p,
                    p_coord,
                    coeff,
                    t_max,
                    u_max,
                    v_max,
                    ex,
                    ey,
                    ez,
                });
            }
        }
        Self { primitives }
    }
}

#[inline(always)]
pub(crate) fn evaluate_eri_contraction_pairs(
    t_ab_max: usize, u_ab_max: usize, v_ab_max: usize,
    t_cd_max: usize, u_cd_max: usize, v_cd_max: usize,
    stride_t: usize, stride_u: usize, stride_v: usize,
    ex_ab: &[f64], ey_ab: &[f64], ez_ab: &[f64],
    ex_cd: &[f64], ey_cd: &[f64], ez_cd: &[f64],
    r_buf: &[f64],
    prefactor: f64,
) -> f64 {
    let max_t = t_ab_max + t_cd_max;
    let max_u = u_ab_max + u_cd_max;
    let max_v = v_ab_max + v_cd_max;

    let mut cx = [0.0; MAX_4C_HERMITE_T + 1];
    let mut cy = [0.0; MAX_4C_HERMITE_T + 1];
    let mut cz = [0.0; MAX_4C_HERMITE_T + 1];

    for t in 0..=t_ab_max {
        let et = ex_ab[t];
        if et == 0.0 { continue; }
        for tp in 0..=t_cd_max {
            let etp = if tp % 2 == 1 { -ex_cd[tp] } else { ex_cd[tp] };
            cx[t + tp] += et * etp;
        }
    }

    for u in 0..=u_ab_max {
        let eu = ey_ab[u];
        if eu == 0.0 { continue; }
        for up in 0..=u_cd_max {
            let eup = if up % 2 == 1 { -ey_cd[up] } else { ey_cd[up] };
            cy[u + up] += eu * eup;
        }
    }

    for v in 0..=v_ab_max {
        let ev = ez_ab[v];
        if ev == 0.0 { continue; }
        for vp in 0..=v_cd_max {
            let evp = if vp % 2 == 1 { -ez_cd[vp] } else { ez_cd[vp] };
            cz[v + vp] += ev * evp;
        }
    }

    let mut sum = 0.0;
    for t in 0..=max_t {
        let cxt = cx[t];
        if cxt == 0.0 { continue; }
        let t_offset = t * stride_t;
        for u in 0..=max_u {
            let cyu = cy[u];
            if cyu == 0.0 { continue; }
            let tu_offset = t_offset + u * stride_u;
            let cxt_cyu = cxt * cyu;
            let mut r_idx = tu_offset;
            for v in 0..=max_v {
                let czv = cz[v];
                if czv != 0.0 {
                    sum += cxt_cyu * czv * r_buf[r_idx];
                }
                r_idx += stride_v;
            }
        }
    }

    prefactor * sum
}

#[inline(always)]
pub fn primitive_eri_from_pairs(
    prim_ab: &PrimitivePairData,
    prim_cd: &PrimitivePairData,
    r_buf: &mut [f64],
) -> f64 {
    let p = prim_ab.p;
    let q = prim_cd.p;
    let p_plus_q = p + q;
    let alpha_prime = (p * q) / p_plus_q;

    let p_minus_q = [
        prim_ab.p_coord[0] - prim_cd.p_coord[0],
        prim_ab.p_coord[1] - prim_cd.p_coord[1],
        prim_ab.p_coord[2] - prim_cd.p_coord[2],
    ];

    let t_ab_max = prim_ab.t_max;
    let u_ab_max = prim_ab.u_max;
    let v_ab_max = prim_ab.v_max;

    let t_cd_max = prim_cd.t_max;
    let u_cd_max = prim_cd.u_max;
    let v_cd_max = prim_cd.v_max;

    let max_t = t_ab_max + t_cd_max;
    let max_u = u_ab_max + u_cd_max;
    let max_v = v_ab_max + v_cd_max;
    let max_n = max_t + max_u + max_v;

    let r2 = p_minus_q[0] * p_minus_q[0] + p_minus_q[1] * p_minus_q[1] + p_minus_q[2] * p_minus_q[2];
    let prefactor = TWO_PI_POW_2_5 / (p * q * p_plus_q.sqrt());

    if max_n == 0 {
        let f0 = boys(0, alpha_prime * r2);
        return prefactor * (prim_ab.ex[0] * prim_ab.ey[0] * prim_ab.ez[0]) * (prim_cd.ex[0] * prim_cd.ey[0] * prim_cd.ez[0]) * f0;
    }

    let stride_v = max_n + 1;
    let stride_u = (max_v + 1) * stride_v;
    let stride_t = (max_u + 1) * stride_u;
    let total_r_size = (max_t + 1) * stride_t;

    if total_r_size <= r_buf.len() {
        coulomb_auxiliary_integrals_flat(max_t, max_u, max_v, alpha_prime, p_minus_q, &mut r_buf[..total_r_size]);
        evaluate_eri_contraction_pairs(
            t_ab_max, u_ab_max, v_ab_max,
            t_cd_max, u_cd_max, v_cd_max,
            stride_t, stride_u, stride_v,
            &prim_ab.ex, &prim_ab.ey, &prim_ab.ez,
            &prim_cd.ex, &prim_cd.ey, &prim_cd.ez,
            &r_buf[..total_r_size],
            prefactor,
        )
    } else {
        let mut dynamic_r = vec![0.0; total_r_size];
        coulomb_auxiliary_integrals_flat(max_t, max_u, max_v, alpha_prime, p_minus_q, &mut dynamic_r);
        evaluate_eri_contraction_pairs(
            t_ab_max, u_ab_max, v_ab_max,
            t_cd_max, u_cd_max, v_cd_max,
            stride_t, stride_u, stride_v,
            &prim_ab.ex, &prim_ab.ey, &prim_ab.ez,
            &prim_cd.ex, &prim_cd.ey, &prim_cd.ez,
            &dynamic_r,
            prefactor,
        )
    }
}

#[inline(always)]
pub fn evaluate_pair_eri(
    pair_ab: &BasisPairData,
    pair_cd: &BasisPairData,
    r_buf: &mut [f64],
) -> f64 {
    let mut val = 0.0;
    for prim_ab in &pair_ab.primitives {
        for prim_cd in &pair_cd.primitives {
            let total_coeff = prim_ab.coeff * prim_cd.coeff;
            if total_coeff.abs() < 1e-13 {
                continue;
            }
            val += total_coeff * primitive_eri_from_pairs(prim_ab, prim_cd, r_buf);
        }
    }
    val
}

/// Evaluates contracted 2-electron repulsion integral (cg_a cg_b | cg_c cg_d)
pub fn contracted_eri(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    c: &ContractedGaussian,
    d: &ContractedGaussian,
) -> f64 {
    let pair_ab = BasisPairData::new(a, b);
    let pair_cd = BasisPairData::new(c, d);
    let mut r_buf = [0.0; 8192];
    evaluate_pair_eri(&pair_ab, &pair_cd, &mut r_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boys_function() {
        assert!((boys(0, 0.0) - 1.0).abs() < 1e-12);
        assert!((boys(1, 0.0) - 1.0 / 3.0).abs() < 1e-12);
        assert!((boys(0, 1.0) - 0.746824).abs() < 1e-5);
        assert!((boys(1, 1.0) - 0.189472).abs() < 1e-5);
    }
}
