// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Fixed-cusp CCSD-F12a/b residual and energy coupling contractions.
//! Used in the modified CCSD iteration and the uneliminated energy functional.
//! Knizia, doctoral thesis (2010), Eqs. 4.123--4.127, gives the retained terms:
//! <https://elib.uni-stuttgart.de/server/api/core/bitstreams/587dac0c-7ea0-47d0-9a5f-92e4fa47e301/content>.

use crate::{error::HfError, matrix::Matrix};

#[derive(Clone, Debug)]
pub struct CcsdF12Coupling {
    n_occ: usize,
    n_virt: usize,
    /// Already contracted with fixed cusp amplitudes .375 I + .125 P.
    fixed_v: Vec<Matrix>,
    fixed_c: Vec<Matrix>,
}

#[derive(Clone, Debug)]
pub struct CcsdF12ResidualCorrection {
    pub singles: Vec<f64>,
    pub doubles: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct CcsdF12EnergyCorrection {
    /// C times the current CCSD tau, present in both F12a and F12b.
    pub fock_amplitude: f64,
    /// Additional projected-Coulomb/tau/singles contribution of F12b.
    pub additional_f12b: f64,
}

fn valid(m: &Matrix, rows: usize, cols: usize) -> bool {
    m.rows == rows
        && m.cols == cols
        && rows.checked_mul(cols) == Some(m.data.len())
        && m.data.iter().all(|x| x.is_finite())
}

impl CcsdF12Coupling {
    /// Ordered ij families: V_ij[p,q]=<pq|r12^-1 Q f|ij>, with p/q in the
    /// full OBS; C_ij[a,b] uses virtual OBS indices. These are projected
    /// operators, not bare geminal matrix elements or an MP2 energy increment.
    pub fn from_pair_integrals(
        n_occ: usize,
        n_virt: usize,
        v: &[Matrix],
        c: &[Matrix],
    ) -> Result<Self, HfError> {
        let invalid =
            || HfError::InvalidInput("invalid CCSD-F12 coupling dimensions/values".into());
        let n = n_occ.checked_add(n_virt).ok_or_else(invalid)?;
        let np = n_occ.checked_mul(n_occ).ok_or_else(invalid)?;
        if n_occ == 0
            || n_virt == 0
            || v.len() != np
            || c.len() != np
            || v.iter().any(|m| !valid(m, n, n))
            || c.iter().any(|m| !valid(m, n_virt, n_virt))
        {
            return Err(invalid());
        }
        if (np as u128) * ((n as u128).pow(2) + (n_virt as u128).pow(2)) * 8 > 128 * 1024 * 1024 {
            return Err(HfError::Integral(
                "CCSD-F12 coupling storage exceeds 128 MiB".into(),
            ));
        }
        for family in [v, c] {
            for i in 0..n_occ {
                for j in 0..n_occ {
                    let a = &family[i * n_occ + j];
                    let b = &family[j * n_occ + i];
                    if (0..a.rows)
                        .any(|p| (0..a.cols).any(|q| (a[(p, q)] - b[(q, p)]).abs() > 1e-10))
                    {
                        return Err(HfError::InvalidInput(
                            "CCSD-F12 pair exchange symmetry is violated".into(),
                        ));
                    }
                }
            }
        }
        let contract = |family: &[Matrix]| {
            (0..np)
                .map(|ij| {
                    let ji = (ij % n_occ) * n_occ + ij / n_occ;
                    family[ij].scale(0.375).add(&family[ji].scale(0.125))
                })
                .collect()
        };
        Ok(Self {
            n_occ,
            n_virt,
            fixed_v: contract(v),
            fixed_c: contract(c),
        })
    }

    fn validate_singles(&self, t1: &[f64]) -> Result<(), HfError> {
        if t1.len() != self.n_occ * self.n_virt || t1.iter().any(|t| !t.is_finite()) {
            return Err(HfError::InvalidInput(
                "invalid CCSD-F12 singles amplitudes".into(),
            ));
        }
        Ok(())
    }

    /// Add to the true conventional residuals before preconditioning/DIIS.
    /// Singles and doubles storage is i,a and i,j,a,b, respectively.
    pub fn residual_corrections(&self, t1: &[f64]) -> Result<CcsdF12ResidualCorrection, HfError> {
        self.validate_singles(t1)?;
        let (no, nv) = (self.n_occ, self.n_virt);
        let mut singles = vec![0.; no * nv];
        let mut doubles = vec![0.; no * no * nv * nv];
        for i in 0..no {
            for a in 0..nv {
                for k in 0..no {
                    singles[i * nv + a] += 2. * self.fixed_v[i * no + k][(no + a, k)]
                        - self.fixed_v[k * no + i][(no + a, k)];
                }
            }
            for j in 0..no {
                let v = &self.fixed_v[i * no + j];
                let c = &self.fixed_c[i * no + j];
                for a in 0..nv {
                    for b in 0..nv {
                        let mut value = v[(no + a, no + b)] + c[(a, b)];
                        for k in 0..no {
                            value -=
                                v[(no + a, k)] * t1[k * nv + b] + t1[k * nv + a] * v[(k, no + b)];
                        }
                        doubles[((i * no + j) * nv + a) * nv + b] = value;
                    }
                }
            }
        }
        if singles.iter().chain(&doubles).any(|v| !v.is_finite()) {
            return Err(HfError::Response(
                "nonfinite CCSD-F12 residual correction".into(),
            ));
        }
        Ok(CcsdF12ResidualCorrection { singles, doubles })
    }

    /// The uneliminated fixed-geminal V/B/X functional is added separately.
    /// This coupling uses the current CCSD tau=T2+T1*T1, not an MP2 estimate.
    pub fn energy_corrections(
        &self,
        t1: &[f64],
        t2: &[f64],
    ) -> Result<CcsdF12EnergyCorrection, HfError> {
        self.validate_singles(t1)?;
        let (no, nv) = (self.n_occ, self.n_virt);
        if t2.len() != no * no * nv * nv || t2.iter().any(|t| !t.is_finite()) {
            return Err(HfError::InvalidInput(
                "invalid CCSD-F12 doubles amplitudes".into(),
            ));
        }
        let mut fock_amplitude = 0.;
        let mut additional_f12b = 0.;
        for i in 0..no {
            for j in 0..no {
                let ij = i * no + j;
                let ji = j * no + i;
                let v = |p, q| 2. * self.fixed_v[ij][(p, q)] - self.fixed_v[ji][(p, q)];
                for a in 0..nv {
                    additional_f12b +=
                        t1[i * nv + a] * v(no + a, j) + v(i, no + a) * t1[j * nv + a];
                    for b in 0..nv {
                        let tau = t2[(ij * nv + a) * nv + b] + t1[i * nv + a] * t1[j * nv + b];
                        fock_amplitude +=
                            (2. * self.fixed_c[ij][(a, b)] - self.fixed_c[ji][(a, b)]) * tau;
                        additional_f12b += v(no + a, no + b) * tau;
                    }
                }
            }
        }
        if !fock_amplitude.is_finite() || !additional_f12b.is_finite() {
            return Err(HfError::Response(
                "nonfinite CCSD-F12 energy correction".into(),
            ));
        }
        Ok(CcsdF12EnergyCorrection {
            fock_amplitude,
            additional_f12b,
        })
    }
}
