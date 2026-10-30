// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Explicitly correlated geminal correlation factors and analytical Gaussian geminal integrals.
//!
//! Evaluates the Slater-type geminal (STG) correlation factor:
//!   F_{12}(r_{12}) = -(1 / \gamma) * exp(-\gamma * r_{12})
//! expanded into a linear combination of 6 Gaussian geminals:
//!   F_{12}(r_{12}) \approx \sum_{k=1}^6 c_k exp(-\alpha_k * r_{12}^2)
//!
//! Provides analytical 2-electron geminal integrals for arbitrary angular momentum (s, p, d, f, g...)
//! using McMurchie-Davidson Hermite expansion recurrence.

use crate::basis::ContractedGaussian;
use crate::integrals::hermite_expansion_coefficients;
use std::f64::consts::PI;

/// High-precision six-Gaussian fit returned by Psi4 MintsHelper::f12_cgtg.
/// Exponents scale by γ²;
/// the coefficients below represent -exp(-r), before the 1/γ normalization.
/// Provenance and scale checks: tests/data/f12_slater_fit_reference.json.
pub const TEW_KLOPPER_EXPONENTS_GAMMA_1: [f64; 6] = [
    0.22085085450735284,
    1.0040191632019282,
    3.621217309837873,
    12.162483236221904,
    45.85533244802934,
    254.23460688554644,
];

pub const TEW_KLOPPER_COEFFICIENTS_GAMMA_1: [f64; 6] = [
    -0.31442480597241274,
    -0.303695753533872,
    -0.16806968430232927,
    -0.09811581215285761,
    -0.060246640234342785,
    -0.03726354196850484,
];

/// Slater-type geminal correlation factor F_{12}(r_{12}) = -(1/\gamma) * exp(-\gamma * r_{12}).
#[derive(Clone, Debug)]
pub struct GeminalCorrelationFactor {
    /// Length-scale parameter \gamma in a_0^{-1}.
    pub gamma: f64,
    /// Gaussian exponents \alpha_k.
    pub exponents: [f64; 6],
    /// Contraction coefficients c_k.
    pub coefficients: [f64; 6],
}

impl Default for GeminalCorrelationFactor {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            exponents: TEW_KLOPPER_EXPONENTS_GAMMA_1,
            coefficients: TEW_KLOPPER_COEFFICIENTS_GAMMA_1,
        }
    }
}

impl GeminalCorrelationFactor {
    /// Creates a new correlation factor with given exponent parameter \gamma.
    pub fn new(gamma: f64) -> Result<Self, crate::error::HfError> {
        if !gamma.is_finite() || gamma <= 0.0 {
            return Err(crate::error::HfError::InvalidInput(
                "geminal exponent must be finite and positive".into(),
            ));
        }
        let gamma_sq = gamma * gamma;
        let mut exponents = [0.0; 6];
        let mut coefficients = [0.0; 6];
        for k in 0..6 {
            exponents[k] = TEW_KLOPPER_EXPONENTS_GAMMA_1[k] * gamma_sq;
            coefficients[k] = TEW_KLOPPER_COEFFICIENTS_GAMMA_1[k] / gamma;
        }
        if exponents.iter().any(|x| !x.is_finite() || *x <= 0.0)
            || coefficients.iter().any(|x| !x.is_finite())
        {
            return Err(crate::error::HfError::InvalidInput(
                "geminal scale is outside the representable exponent range".into(),
            ));
        }
        Ok(Self {
            gamma,
            exponents,
            coefficients,
        })
    }

    /// Evaluates F_{12}(r) directly from the exact STG expression: -(1/\gamma) * exp(-\gamma * r).
    pub fn eval_exact(&self, r: f64) -> f64 {
        -(1.0 / self.gamma) * (-self.gamma * r).exp()
    }

    /// Evaluates F_{12}(r) using the 6-term Gaussian expansion: \sum_k c_k exp(-\alpha_k * r^2).
    pub fn eval_gaussian(&self, r: f64) -> f64 {
        let r2 = r * r;
        let mut sum = 0.0;
        for k in 0..6 {
            sum += self.coefficients[k] * (-self.exponents[k] * r2).exp();
        }
        sum
    }

    /// Actual Gaussian expansion, shared by every factor/product/commutator
    /// operator. Differentiating an unrelated STG fit would be inconsistent.
    pub fn gaussian(&self) -> Result<super::operators::GaussianGeminal, crate::error::HfError> {
        super::operators::GaussianGeminal::new(
            self.exponents
                .iter()
                .zip(&self.coefficients)
                .map(|(&a, &c)| (a, c))
                .collect(),
        )
    }
}

/// Evaluates 1D derivatives of exp(-\eta * \Delta^2):
///   R_n = (d / d\Delta)^n exp(-\eta * \Delta^2)
/// via 3-term recurrence:
///   R_0 = exp(-\eta * \Delta^2)
///   R_1 = -2\eta \Delta R_0
///   R_{n+1} = -2\eta \Delta R_n - 2\eta n R_{n-1}
fn evaluate_gaussian_derivatives(max_order: usize, eta: f64, delta: f64) -> Vec<f64> {
    let mut r = vec![0.0; max_order + 1];
    let r0 = (-eta * delta * delta).exp();
    r[0] = r0;
    if max_order == 0 {
        return r;
    }
    let two_eta = 2.0 * eta;
    let two_eta_delta = two_eta * delta;
    r[1] = -two_eta_delta * r0;
    for n in 1..max_order {
        r[n + 1] = -two_eta_delta * r[n] - two_eta * (n as f64) * r[n - 1];
    }
    r
}

/// Evaluates 1D Cartesian Gaussian geminal integral:
///   \int_{-\infty}^\infty dx_1 dx_2 (x_1 - A_x)^{la} (x_1 - B_x)^{lb} e^{-\alpha_a(x_1 - A_x)^2 - \alpha_b(x_1 - B_x)^2}
///     * e^{-\alpha_{gem}(x_1 - x_2)^2} * (x_2 - C_x)^{lc} (x_2 - D_x)^{ld} e^{-\alpha_c(x_2 - C_x)^2 - \alpha_d(x_2 - D_x)^2}
fn primitive_geminal_1d(
    la: usize,
    lb: usize,
    alpha_a: f64,
    alpha_b: f64,
    ax: f64,
    bx: f64,
    lc: usize,
    ld: usize,
    alpha_c: f64,
    alpha_d: f64,
    cx: f64,
    dx: f64,
    alpha_gem: f64,
) -> f64 {
    let p = alpha_a + alpha_b;
    let px = (alpha_a * ax + alpha_b * bx) / p;

    let q = alpha_c + alpha_d;
    let qx = (alpha_c * cx + alpha_d * dx) / q;

    let denom = p * q + alpha_gem * (p + q);
    let eta = (p * q * alpha_gem) / denom;
    let delta = px - qx;

    let base_integral = (PI * PI / denom).sqrt();
    let max_order = (la + lb) + (lc + ld);
    let r_derivs = evaluate_gaussian_derivatives(max_order, eta, delta);

    let ex_ab = hermite_expansion_coefficients(la, lb, alpha_a, alpha_b, ax, bx);
    let ex_cd = hermite_expansion_coefficients(lc, ld, alpha_c, alpha_d, cx, dx);

    let mut sum = 0.0;
    for t in 0..=(la + lb) {
        let et = ex_ab[la][lb][t];
        if et == 0.0 {
            continue;
        }
        for u in 0..=(lc + ld) {
            let eu = ex_cd[lc][ld][u];
            if eu == 0.0 {
                continue;
            }
            let sign = if u % 2 == 1 { -1.0 } else { 1.0 };
            sum += et * eu * sign * r_derivs[t + u];
        }
    }

    // Hermite E coefficients already contain both Gaussian product factors.
    base_integral * sum
}

/// Evaluates primitive 3D Gaussian geminal integral <g_a g_b | exp(-\alpha_{gem} r_{12}^2) | g_c g_d>.
pub fn primitive_geminal_eri(
    alpha_a: f64,
    la: [usize; 3],
    a: [f64; 3],
    alpha_b: f64,
    lb: [usize; 3],
    b: [f64; 3],
    alpha_c: f64,
    lc: [usize; 3],
    c: [f64; 3],
    alpha_d: f64,
    ld: [usize; 3],
    d: [f64; 3],
    alpha_gem: f64,
) -> f64 {
    let ix = primitive_geminal_1d(
        la[0], lb[0], alpha_a, alpha_b, a[0], b[0], lc[0], ld[0], alpha_c, alpha_d, c[0], d[0],
        alpha_gem,
    );
    if ix == 0.0 {
        return 0.0;
    }
    let iy = primitive_geminal_1d(
        la[1], lb[1], alpha_a, alpha_b, a[1], b[1], lc[1], ld[1], alpha_c, alpha_d, c[1], d[1],
        alpha_gem,
    );
    if iy == 0.0 {
        return 0.0;
    }
    let iz = primitive_geminal_1d(
        la[2], lb[2], alpha_a, alpha_b, a[2], b[2], lc[2], ld[2], alpha_c, alpha_d, c[2], d[2],
        alpha_gem,
    );
    ix * iy * iz
}

/// Evaluates contracted 2-electron Gaussian geminal integral:
///   <\chi_a \chi_b | exp(-\alpha_{gem} r_{12}^2) | \chi_c \chi_d>
pub fn contracted_geminal_integral(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    c: &ContractedGaussian,
    d: &ContractedGaussian,
    alpha_gem: f64,
) -> f64 {
    let mut val = 0.0;
    for pa in &a.primitives {
        for pb in &b.primitives {
            let ab_coeff = pa.coefficient * pb.coefficient;
            for pc in &c.primitives {
                for pd in &d.primitives {
                    let cd_coeff = pc.coefficient * pd.coefficient;
                    let prim = primitive_geminal_eri(
                        pa.exponent,
                        a.l,
                        a.center,
                        pb.exponent,
                        b.l,
                        b.center,
                        pc.exponent,
                        c.l,
                        c.center,
                        pd.exponent,
                        d.l,
                        d.center,
                        alpha_gem,
                    );
                    val += ab_coeff * cd_coeff * prim;
                }
            }
        }
    }
    val
}

/// Evaluates contracted 2-electron STG geminal integral:
///   <\chi_a \chi_b | F_{12} | \chi_c \chi_d> = \sum_{k=1}^6 c_k <\chi_a \chi_b | exp(-\alpha_k r_{12}^2) | \chi_c \chi_d>
pub fn contracted_stg_integral(
    a: &ContractedGaussian,
    b: &ContractedGaussian,
    c: &ContractedGaussian,
    d: &ContractedGaussian,
    factor: &GeminalCorrelationFactor,
) -> f64 {
    let mut sum = 0.0;
    for k in 0..6 {
        let integ = contracted_geminal_integral(a, b, c, d, factor.exponents[k]);
        sum += factor.coefficients[k] * integ;
    }
    sum
}

/// Computes the complete 4-index STG geminal tensor (ij|F_{12}|kl) in the AO basis.
/// Packed indexing: [eri_index(i, j, k, l)].
pub fn compute_geminal_tensor(
    basis: &[ContractedGaussian],
    factor: &GeminalCorrelationFactor,
) -> Vec<f64> {
    let n = basis.len();
    let num_pairs = n * (n + 1) / 2;
    let total_elements = num_pairs * (num_pairs + 1) / 2;
    let mut tensor = vec![0.0; total_elements];

    for i in 0..n {
        for j in 0..=i {
            let ij = i * (i + 1) / 2 + j;
            for k in 0..n {
                for l in 0..=k {
                    let kl = k * (k + 1) / 2 + l;
                    if ij >= kl {
                        let idx = ij * (ij + 1) / 2 + kl;
                        let val = contracted_stg_integral(
                            &basis[i], &basis[j], &basis[k], &basis[l], factor,
                        );
                        tensor[idx] = val;
                    }
                }
            }
        }
    }

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::Primitive;

    #[test]
    fn test_stg_geminal_expansion_accuracy() {
        let factor = GeminalCorrelationFactor::new(1.0).unwrap();
        // Compare 6-term Gaussian fit to exact STG -(1/\gamma)*exp(-\gamma*r)
        for &r in &[0.0, 0.5, 1.0, 2.0, 3.0] {
            let exact = factor.eval_exact(r);
            let fit = factor.eval_gaussian(r);
            assert!(
                (exact - fit).abs() < 0.08,
                "Geminal fit mismatch at r={}: exact={}, fit={}",
                r,
                exact,
                fit
            );
        }
    }

    #[test]
    fn test_primitive_geminal_eri_symmetry() {
        let alpha = 1.2;
        let pos_a = [0.0, 0.0, -0.7];
        let pos_b = [0.0, 0.0, 0.7];
        let la = [0, 0, 0];
        let lb = [0, 0, 0];

        let val_1 = primitive_geminal_eri(
            alpha, la, pos_a, alpha, lb, pos_b, alpha, la, pos_a, alpha, lb, pos_b, 1.5,
        );

        let val_2 = primitive_geminal_eri(
            alpha, lb, pos_b, alpha, la, pos_a, alpha, la, pos_a, alpha, lb, pos_b, 1.5,
        );

        let val_3 = primitive_geminal_eri(
            alpha, la, pos_a, alpha, lb, pos_b, alpha, lb, pos_b, alpha, la, pos_a, 1.5,
        );

        assert!((val_1 - val_2).abs() < 1e-14);
        assert!((val_1 - val_3).abs() < 1e-14);
        assert!(val_1 > 0.0, "Geminal overlap must be strictly positive");
    }

    #[test]
    fn test_contracted_geminal_integral_h2() {
        let cg1 = ContractedGaussian {
            atom_index: 0,
            center: [0.0, 0.0, -0.7],
            l: [0, 0, 0],
            primitives: vec![
                Primitive {
                    exponent: 3.42525,
                    coefficient: 0.154329,
                },
                Primitive {
                    exponent: 0.62391,
                    coefficient: 0.535328,
                },
            ],
        };
        let cg2 = ContractedGaussian {
            atom_index: 1,
            center: [0.0, 0.0, 0.7],
            l: [0, 0, 0],
            primitives: vec![
                Primitive {
                    exponent: 3.42525,
                    coefficient: 0.154329,
                },
                Primitive {
                    exponent: 0.62391,
                    coefficient: 0.535328,
                },
            ],
        };

        let factor = GeminalCorrelationFactor::default();
        let stg_val = contracted_stg_integral(&cg1, &cg1, &cg2, &cg2, &factor);
        // STG value with negative prefactor -(1/\gamma) must be negative
        assert!(stg_val < 0.0, "STG integral must be negative: {}", stg_val);
    }
}
