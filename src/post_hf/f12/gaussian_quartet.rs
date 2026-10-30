// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Primitive Hermite data shared by every Gaussian exponent in a geminal fit.
//! The r12² kernel is an analytic exponent derivative, without orbital raising.

use crate::integrals::{boys_array, hermite_expansion_coefficients};
use std::f64::consts::PI;

pub(super) struct GaussianQuartet {
    p: f64,
    q: f64,
    separation: [f64; 3],
    combined: [Vec<f64>; 3],
}

impl GaussianQuartet {
    pub fn new(e: [f64; 4], l: [[usize; 3]; 4], c: [[f64; 3]; 4]) -> Self {
        let p = e[0] + e[1];
        let q = e[2] + e[3];
        let separation = std::array::from_fn(|a| {
            (e[0] * c[0][a] + e[1] * c[1][a]) / p - (e[2] * c[2][a] + e[3] * c[3][a]) / q
        });
        let combined = std::array::from_fn(|a| {
            let ab = hermite_expansion_coefficients(l[0][a], l[1][a], e[0], e[1], c[0][a], c[1][a]);
            let cd = hermite_expansion_coefficients(l[2][a], l[3][a], e[2], e[3], c[2][a], c[3][a]);
            let mut values = vec![0.; l.iter().map(|x| x[a]).sum::<usize>() + 1];
            for (t, &x) in ab[l[0][a]][l[1][a]].iter().enumerate() {
                for (u, &y) in cd[l[2][a]][l[3][a]].iter().enumerate() {
                    values[t + u] += x * y * if u % 2 == 0 { 1. } else { -1. };
                }
            }
            values
        });
        Self {
            p,
            q,
            separation,
            combined,
        }
    }

    pub fn gaussian(&self, gamma: f64) -> f64 {
        self.gaussian_moments::<false>(gamma)[0]
    }

    pub fn r2(&self, gamma: f64) -> f64 {
        self.gaussian_moments::<true>(gamma)[1]
    }

    fn gaussian_moments<const DIFFERENTIATE: bool>(&self, gamma: f64) -> [f64; 2] {
        let pq = self.p * self.q;
        let determinant = pq + gamma * (self.p + self.q);
        let eta = pq * gamma / determinant;
        let mut moment = [0.; 3];
        let mut differentiated = [0.; 3];
        for axis in 0..3 {
            // Original angular degree <=28; kinetic actions raise it by at most
            // two. No additional powers are needed for the exponent derivative.
            let mut r = [0.; 33];
            let mut dr = [0.; 33];
            let delta = self.separation[axis];
            let order = self.combined[axis].len() - 1;
            r[0] = (-eta * delta * delta).exp();
            if DIFFERENTIATE {
                dr[0] = -delta * delta * r[0];
            }
            if order > 0 {
                r[1] = -2. * eta * delta * r[0];
                if DIFFERENTIATE {
                    dr[1] = -2. * delta * (r[0] + eta * dr[0]);
                }
            }
            for degree in 1..order {
                r[degree + 1] = -2. * eta * (delta * r[degree] + degree as f64 * r[degree - 1]);
                if DIFFERENTIATE {
                    dr[degree + 1] = -2. * delta * (r[degree] + eta * dr[degree])
                        - 2. * degree as f64 * (r[degree - 1] + eta * dr[degree - 1]);
                }
            }
            for (degree, &coefficient) in self.combined[axis].iter().enumerate() {
                moment[axis] += coefficient * r[degree];
                if DIFFERENTIATE {
                    differentiated[axis] += coefficient * dr[degree];
                }
            }
        }
        let prefactor = (PI * PI * PI) / determinant.powf(1.5);
        let product = moment.iter().product::<f64>();
        let r2 = if DIFFERENTIATE {
            let derivative = (0..3)
                .map(|a| differentiated[a] * moment[(a + 1) % 3] * moment[(a + 2) % 3])
                .sum::<f64>();
            prefactor
                * (1.5 * (self.p + self.q) / determinant * product
                    - (pq / determinant).powi(2) * derivative)
        } else {
            0.
        };
        [prefactor * product, r2]
    }

    pub fn coulomb(&self, gamma: f64) -> f64 {
        let pq = self.p * self.q;
        let determinant = pq + gamma * (self.p + self.q);
        let r2 = self.separation.iter().map(|x| x * x).sum::<f64>();
        let eta = pq * gamma / determinant;
        let rho = pq.powi(2) / ((self.p + self.q) * determinant);
        let maximum = self.combined.each_ref().map(|values| values.len() - 1);
        let order = maximum.iter().sum::<usize>();
        let mut boys = vec![0.; order + 1];
        boys_array(order, rho * r2, &mut boys);
        let sv = order + 1;
        let su = (maximum[2] + 1) * sv;
        let st = (maximum[1] + 1) * su;
        let at = |t, u, v, n| t * st + u * su + v * sv + n;
        let mut auxiliary = vec![0.; (maximum[0] + 1) * st];
        for n in 0..=order {
            let mut binomial = 1.;
            let mut moment = 0.;
            for (k, &fk) in boys.iter().enumerate().take(n + 1) {
                moment += binomial * eta.powi((n - k) as i32) * rho.powi(k as i32) * fk;
                binomial *= (n - k) as f64 / (k + 1) as f64;
            }
            auxiliary[n] = (-2f64).powi(n as i32) * (-eta * r2).exp() * moment;
        }
        for n in (0..order).rev() {
            for t in 0..=maximum[0] {
                for u in 0..=maximum[1] {
                    for v in 0..=maximum[2] {
                        if t + u + v == 0 || t + u + v > order - n {
                            continue;
                        }
                        let degree = [t, u, v];
                        let axis = degree.iter().position(|&l| l > 0).unwrap_or(0);
                        let mut lowered = degree;
                        lowered[axis] -= 1;
                        let mut value = self.separation[axis]
                            * auxiliary[at(lowered[0], lowered[1], lowered[2], n + 1)];
                        if degree[axis] > 1 {
                            lowered[axis] -= 1;
                            value += (degree[axis] - 1) as f64
                                * auxiliary[at(lowered[0], lowered[1], lowered[2], n + 1)];
                        }
                        auxiliary[at(t, u, v, n)] = value;
                    }
                }
            }
        }
        let mut sum = 0.;
        for t in 0..=maximum[0] {
            for u in 0..=maximum[1] {
                for v in 0..=maximum[2] {
                    sum += self.combined[0][t]
                        * self.combined[1][u]
                        * self.combined[2][v]
                        * auxiliary[at(t, u, v, 0)];
                }
            }
        }
        2. * PI.powf(2.5) / (determinant * (self.p + self.q).sqrt()) * sum
    }
}
