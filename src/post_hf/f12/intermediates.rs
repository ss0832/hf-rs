// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Ansatz-3 projector and 3C pair intermediates in an orthonormal OBS+CABS.
//!
//! Spatial product indices are (electron 1, electron 2). The identity part of
//! f(F1+F2)f is evaluated with f² and one half of [f,[T1+T2,f]];
//! finite projector terms use the complete RI Fock matrix, including CABS.
use crate::options::F12Hybrid;
use crate::{error::HfError, matrix::Matrix};

#[derive(Clone, Debug)]
pub struct F12FockSpace {
    pub n_occ: usize,
    pub n_obs: usize,
    pub fock: Matrix,
    /// Subtracted nonlocal part: K for all-electron Hamiltonians. With ECPs,
    /// K - V_ECP,nonlocal, minus any finite-rank exact-OBS Fock restoration,
    /// so F + nonlocal contains only local potentials and T.
    pub nonlocal: Matrix,
    pub orbital_energies: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct F12PairIntermediates {
    pub pair: [usize; 2],
    /// Direct and exchanged occupied-pair matrix elements.
    pub v: [f64; 2],
    pub x: [f64; 2],
    pub b: [f64; 2],
    /// C_ab,ij and G_ab,ij, with a,b in the OBS virtual space.
    pub c: Matrix,
    pub g: Matrix,
    pub v_bar: [f64; 2],
    pub b_bar: [f64; 2],
}

#[derive(Clone, Copy)]
pub struct F12PairIntegralView<'a> {
    pub pair: [usize; 2],
    pub factor: &'a Matrix,
    pub coulomb: &'a Matrix,
    pub squared: &'a Matrix,
    pub squared_exchange: &'a Matrix,
    pub coulomb_factor: [f64; 2],
    pub positive_commutator: [f64; 2],
}

/// Nonlocal one-electron potential and its right action on the occupied pair.
/// Both use the same orthonormal OBS+CABS resolution as the Fock intermediates.
#[derive(Clone, Copy)]
pub struct F12NonlocalPairView<'a> {
    pub potential: &'a Matrix,
    /// W[p,q] = <pq|f(V1+V2)|ij>; this is not (V1+V2)<pq|f|ij>.
    pub factor_right_action: &'a Matrix,
}

impl F12PairIntermediates {
    /// Unordered spatial pair, with fixed singlet/triplet cusp amplitudes.
    pub fn fixed_energy(&self) -> (f64, f64) {
        self.fixed_energy_with(self.v_bar, self.b_bar)
    }

    /// Fixed geminal functional before eliminating conventional doubles.
    pub fn fixed_uneliminated_energy(&self, occupied_energy_sum: f64) -> (f64, f64) {
        self.fixed_energy_with(
            self.v,
            std::array::from_fn(|k| self.b[k] - occupied_energy_sum * self.x[k]),
        )
    }

    fn fixed_energy_with(&self, v: [f64; 2], b: [f64; 2]) -> (f64, f64) {
        let singlet = (if self.pair[0] == self.pair[1] {
            0.5
        } else {
            1.0
        }) * (v[0] + v[1] + 0.25 * (b[0] + b[1]));
        let triplet = if self.pair[0] == self.pair[1] {
            0.0
        } else {
            3.0 * (0.5 * (v[0] - v[1]) + 0.0625 * (b[0] - b[1]))
        };
        (singlet, triplet)
    }
}

fn finite_matrix(m: &Matrix, rows: usize, cols: usize) -> bool {
    m.rows == rows
        && m.cols == cols
        && rows.checked_mul(cols) == Some(m.data.len())
        && m.data.iter().all(|v| v.is_finite())
}
fn inner(a: &Matrix, b: &Matrix) -> f64 {
    a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum()
}
fn one_body(f: &Matrix, r: &Matrix) -> Matrix {
    f.dot(r).add(&r.dot(&f.transpose()))
}

impl F12FockSpace {
    /// 3C hybrid approximations of Knizia (2010), Eqs. 4.47--4.55.
    /// The minimally projected g has no occupied/CABS coupling. QgQ=QFQ,
    /// but n=h-g contains that removed coupling in addition to K-V_ECP,nl.
    /// HY1 omits <f|n12 Q|f>; HY2 omits only its CABS*CABS resolution.
    pub fn pair_intermediates_with_hybrid(
        &self,
        integrals: F12PairIntegralView<'_>,
        hybrid: F12Hybrid,
    ) -> Result<F12PairIntermediates, HfError> {
        let mut result = self.pair_intermediates(integrals)?;
        if hybrid == F12Hybrid::Full {
            return Ok(result);
        }
        let n = self.fock.rows;
        let mut nonlocal = self.nonlocal.clone();
        for i in 0..self.n_occ {
            for x in self.n_obs..n {
                nonlocal[(i, x)] += self.fock[(i, x)];
                nonlocal[(x, i)] += self.fock[(x, i)];
            }
        }
        let mut right = integrals.factor.clone();
        for p in 0..n {
            for q in 0..n {
                if self.removed(p, q)
                    || (hybrid == F12Hybrid::Hy2 && (p < self.n_obs || q < self.n_obs))
                {
                    right[(p, q)] = 0.0;
                }
            }
        }
        let action = one_body(&nonlocal, &right);
        let delta = [
            inner(integrals.factor, &action),
            inner(&integrals.factor.transpose(), &action),
        ];
        for (k, increment) in delta.into_iter().enumerate() {
            if !increment.is_finite() {
                return Err(HfError::Integral(
                    "nonfinite hybrid F12 intermediate".into(),
                ));
            }
            result.b[k] += increment;
            result.b_bar[k] += increment;
        }
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), HfError> {
        let n = self.fock.rows;
        if self.n_occ > self.n_obs
            || self.n_obs > n
            || self.orbital_energies.len() != self.n_obs
            || self.orbital_energies.iter().any(|e| !e.is_finite())
            || !finite_matrix(&self.fock, n, n)
            || !finite_matrix(&self.nonlocal, n, n)
        {
            return Err(HfError::InvalidInput(
                "invalid F12 Fock space dimensions or values".into(),
            ));
        }
        for matrix in [&self.fock, &self.nonlocal] {
            for i in 0..n {
                for j in 0..i {
                    if (matrix[(i, j)] - matrix[(j, i)]).abs() > 1e-9 {
                        return Err(HfError::InvalidInput(
                            "F12 requires Hermitian real Fock/nonlocal matrices".into(),
                        ));
                    }
                }
            }
        }
        for i in 0..self.n_obs {
            for j in 0..self.n_obs {
                let expected = if i == j {
                    self.orbital_energies[i]
                } else {
                    0.0
                };
                if (self.fock[(i, j)] - expected).abs() > 1e-7 {
                    return Err(HfError::InvalidInput(
                        "F12 conventional doubles require canonical OBS orbitals".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Complement of Q=(1-O1)(1-O2)-V1V2, expanded in OBS+CABS:
    /// OBS*OBS + CABS*occupied + occupied*CABS. These three sets are disjoint.
    pub fn removed(&self, p: usize, q: usize) -> bool {
        (p < self.n_obs && q < self.n_obs)
            || (p >= self.n_obs && q < self.n_occ)
            || (p < self.n_occ && q >= self.n_obs)
    }

    fn validate_integrals(&self, values: &F12PairIntegralView<'_>) -> Result<(), HfError> {
        self.validate()?;
        let n = self.fock.rows;
        if values.pair.iter().any(|&i| i >= self.n_occ)
            || !finite_matrix(values.factor, n, n)
            || !finite_matrix(values.coulomb, n, n)
            || !finite_matrix(values.squared, n, self.n_occ)
            || !finite_matrix(values.squared_exchange, n, self.n_occ)
            || values
                .coulomb_factor
                .iter()
                .chain(&values.positive_commutator)
                .any(|v| !v.is_finite())
        {
            return Err(HfError::InvalidInput(
                "invalid F12 pair integral dimensions or values".into(),
            ));
        }
        Ok(())
    }

    /// Canonical 3*A(FIX): GBC, neglect of the exchange commutator in A,
    /// EBC (C=0), and Z=0. Werner--Adler--Manby (2007), Eqs. 31,70,94--96.
    /// U[p,q]=<pq|[f,T1+T2]|ij>. Its real antisymmetry fixes the projector sign.
    pub fn pair_intermediates_3a(
        &self,
        values: F12PairIntegralView<'_>,
        kinetic_commutator: &Matrix,
    ) -> Result<F12PairIntermediates, HfError> {
        self.pair_intermediates_3a_with_nonlocal(values, kinetic_commutator, None)
    }

    /// 3*A neglects the exchange commutator, but retains [f,V_ECP,nonlocal].
    /// The additional nonlocal products are resolved in the complete OBS+CABS.
    pub fn pair_intermediates_3a_with_nonlocal(
        &self,
        values: F12PairIntegralView<'_>,
        kinetic_commutator: &Matrix,
        nonlocal: Option<F12NonlocalPairView<'_>>,
    ) -> Result<F12PairIntermediates, HfError> {
        self.validate_integrals(&values)?;
        let n = self.fock.rows;
        if !finite_matrix(kinetic_commutator, n, n) {
            return Err(HfError::InvalidInput(
                "invalid F12 kinetic commutator matrix".into(),
            ));
        }
        let [i, j] = values.pair;
        let mut v = values.coulomb_factor;
        let mut x = [values.squared[(i, j)], values.squared_exchange[(i, j)]];
        let mut b_bar = values.positive_commutator.map(|value| 0.5 * value);
        let nonlocal_commutator = if let Some(nonlocal) = nonlocal {
            let potential = nonlocal.potential;
            if !finite_matrix(potential, n, n)
                || !finite_matrix(nonlocal.factor_right_action, n, n)
                || (0..n)
                    .any(|p| (0..p).any(|q| (potential[(p, q)] - potential[(q, p)]).abs() > 1e-9))
            {
                return Err(HfError::InvalidInput(
                    "invalid F12 nonlocal potential/action matrix".into(),
                ));
            }
            let action = one_body(potential, values.factor);
            // The unit term is f V f - {V,f²}/2. The f² matrix elements retain
            // their unprojected integral values; f V f uses the RI resolution.
            b_bar[0] += inner(values.factor, &action);
            b_bar[1] += inner(&values.factor.transpose(), &action);
            for p in 0..n {
                b_bar[0] -= potential[(i, p)] * values.squared[(p, j)]
                    + potential[(j, p)] * values.squared_exchange[(p, i)];
                b_bar[1] -= potential[(i, p)] * values.squared_exchange[(p, j)]
                    + potential[(j, p)] * values.squared[(p, i)];
            }
            Some(nonlocal.factor_right_action.sub(&action))
        } else {
            None
        };
        for p in 0..n {
            for q in 0..n {
                if self.removed(p, q) {
                    for (exchange, index) in [(false, 0), (true, 1)] {
                        let projected = if exchange {
                            values.factor[(q, p)]
                        } else {
                            values.factor[(p, q)]
                        };
                        v[index] -= values.coulomb[(p, q)] * projected;
                        x[index] -= values.factor[(p, q)] * projected;
                        b_bar[index] += kinetic_commutator[(p, q)] * projected;
                        if let Some(commutator) = &nonlocal_commutator {
                            b_bar[index] += commutator[(p, q)] * projected;
                        }
                    }
                }
            }
        }
        let eps_sum = self.orbital_energies[i] + self.orbital_energies[j];
        let nv = self.n_obs - self.n_occ;
        let mut g = Matrix::zeros(nv, nv);
        for a in 0..nv {
            for d in 0..nv {
                let p = self.n_occ + a;
                let q = self.n_occ + d;
                let gap = self.orbital_energies[p] + self.orbital_energies[q] - eps_sum;
                if !gap.is_finite() || gap <= 1e-12 {
                    return Err(HfError::Response(
                        "F12 needs positive conventional-doubles gaps".into(),
                    ));
                }
                g[(a, d)] = values.coulomb[(p, q)];
            }
        }
        let b = std::array::from_fn(|k| b_bar[k] + eps_sum * x[k]);
        if v.iter()
            .chain(&x)
            .chain(&b)
            .chain(&b_bar)
            .any(|value| !value.is_finite())
        {
            return Err(HfError::Integral("nonfinite 3*A pair intermediate".into()));
        }
        Ok(F12PairIntermediates {
            pair: values.pair,
            v,
            x,
            b,
            c: Matrix::zeros(nv, nv),
            g,
            v_bar: v,
            b_bar,
        })
    }

    /// f[p,q]=<pq|f12|ij>; squared[p,k]=<p k|f12²|ij>, k occupied.
    /// squared_exchange has ket ji. The commutator is [f,[T1+T2,f]], not
    /// its negative and not the one-electron gradient square alone.
    pub fn pair_intermediates(
        &self,
        integrals: F12PairIntegralView<'_>,
    ) -> Result<F12PairIntermediates, HfError> {
        self.validate_integrals(&integrals)?;
        let F12PairIntegralView {
            pair,
            factor,
            coulomb,
            squared,
            squared_exchange,
            coulomb_factor,
            positive_commutator,
        } = integrals;
        let n = self.fock.rows;
        let [i, j] = pair;
        let mut removed = factor.clone();
        for p in 0..n {
            for q in 0..n {
                if !self.removed(p, q) {
                    removed[(p, q)] = 0.0;
                }
            }
        }
        let removed_exchange = removed.transpose();
        let full_action = one_body(&self.fock, factor);
        let exchange_action = full_action.transpose();
        let removed_action = one_body(&self.fock, &removed);
        let exchange_removed_action = removed_action.transpose();
        let local = self.fock.add(&self.nonlocal);
        let nonlocal_action = one_body(&self.nonlocal, factor);
        let nonlocal_exchange_action = nonlocal_action.transpose();
        let mut v = coulomb_factor;
        let mut x = [squared[(i, j)], squared_exchange[(i, j)]];
        let mut b = [0.5 * positive_commutator[0], 0.5 * positive_commutator[1]];
        for p in 0..n {
            b[0] += local[(i, p)] * squared[(p, j)] + local[(j, p)] * squared_exchange[(p, i)];
            b[1] += local[(i, p)] * squared_exchange[(p, j)] + local[(j, p)] * squared[(p, i)];
        }
        for (k, (pr, ar, apr, nr)) in [
            (&removed, &full_action, &removed_action, &nonlocal_action),
            (
                &removed_exchange,
                &exchange_action,
                &exchange_removed_action,
                &nonlocal_exchange_action,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            v[k] -= inner(coulomb, pr);
            x[k] -= inner(factor, pr);
            b[k] -= inner(factor, nr);
            b[k] -= inner(&removed, ar) + inner(&full_action, pr);
            b[k] += inner(&removed, apr);
            // Keep the unprojected and projected actions distinct: replacing
            // <f|P F|f> by <P f|F|P f> loses OBS/CABS cross terms.
        }
        let nv = self.n_obs - self.n_occ;
        let mut c = Matrix::zeros(nv, nv);
        let mut g = Matrix::zeros(nv, nv);
        for a in 0..nv {
            for d in 0..nv {
                let p = self.n_occ + a;
                let q = self.n_occ + d;
                c[(a, d)] = full_action[(p, q)] - removed_action[(p, q)];
                g[(a, d)] = coulomb[(p, q)];
            }
        }
        let eps_sum = self.orbital_energies[i] + self.orbital_energies[j];
        let mut v_bar = v;
        let mut b_bar = [b[0] - eps_sum * x[0], b[1] - eps_sum * x[1]];
        for a in 0..nv {
            for d in 0..nv {
                let gap = self.orbital_energies[self.n_occ + a]
                    + self.orbital_energies[self.n_occ + d]
                    - eps_sum;
                if !gap.is_finite() || gap <= 1e-12 {
                    return Err(HfError::Response(
                        "F12 needs positive conventional-doubles gaps".into(),
                    ));
                }
                v_bar[0] -= g[(a, d)] * c[(a, d)] / gap;
                v_bar[1] -= g[(a, d)] * c[(d, a)] / gap;
                b_bar[0] -= c[(a, d)] * c[(a, d)] / gap;
                b_bar[1] -= c[(a, d)] * c[(d, a)] / gap;
            }
        }
        if v.iter()
            .chain(&x)
            .chain(&b)
            .chain(&v_bar)
            .chain(&b_bar)
            .chain(&c.data)
            .chain(&g.data)
            .any(|v| !v.is_finite())
        {
            return Err(HfError::Integral("nonfinite F12 pair intermediate".into()));
        }
        Ok(F12PairIntermediates {
            pair,
            v,
            x,
            b,
            c,
            g,
            v_bar,
            b_bar,
        })
    }
}
