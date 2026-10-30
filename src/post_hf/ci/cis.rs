// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Configuration Interaction Singles (CIS) for electronic excited states.
//!
//! Provides:
//! - Spin-adapted singlet/triplet excitation energies in Hartree and eV.
//! - Transition dipole moments and oscillator strengths f_k.
//! - Dense symmetric diagonalization in the single-excitation space.

use crate::context::{MoIntegralCache, PostHFContext};
use crate::error::HfError;
use crate::matrix::Matrix;
use crate::mo_integrals::transform_eri_ao_to_mo;
use crate::options::PostHFOptions;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::{ContextMethod, MethodId, MethodMeta};
use crate::scf::HFResult;
use std::borrow::Cow;

/// Single excited state properties from CIS.
#[derive(Clone, Debug, PartialEq)]
pub struct CISEstate {
    pub state_index: usize,
    pub excitation_energy_hartree: f64,
    pub excitation_energy_ev: f64,
    pub oscillator_strength: f64,
    pub transition_dipole: [f64; 3],
    pub amplitudes: Vec<f64>, // [n_occ * n_virt]
}

/// Results of a CIS calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct CISResult {
    pub reference_energy: f64,
    pub excited_states: Vec<CISEstate>,
    pub n_occ: usize,
    pub n_virt: usize,
    /// Zero-based excited root; root zero is the lowest CIS excited state.
    pub root: usize,
    pub triplet: bool,
}

impl PostHFResult for CISResult {
    fn method_name(&self) -> &'static str {
        "CIS"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.excited_states[self.root].excitation_energy_hartree
    }

    fn total_energy(&self) -> f64 {
        self.reference_energy + self.correlation_energy()
    }

    fn to_output(&self) -> Result<crate::output::PostHFOutput, HfError> {
        let mut out =
            crate::output::PostHFOutput::new("CIS", self.reference_energy, self.total_energy())?;
        out.states = self
            .excited_states
            .iter()
            .enumerate()
            .map(|(root, state)| crate::output::State {
                energy: self.reference_energy + state.excitation_energy_hartree,
                s_squared: Some(if self.triplet { 2. } else { 0. }),
                weight: if root == self.root { 1. } else { 0. },
                converged: true,
            })
            .collect();
        out.select_root(self.root)?;
        out.components.insert(
            "excitation_energy_hartree".into(),
            self.correlation_energy(),
        );
        out.components.insert(
            "oscillator_strength".into(),
            self.excited_states[self.root].oscillator_strength,
        );
        Ok(out)
    }
}

/// Configuration Interaction Singles (CIS) solver.
pub struct CIS<'a> {
    pub integral_cache: Option<&'a MoIntegralCache<'a>>,
    pub hf: &'a HFResult,
    pub n_roots: usize,
    pub basis: Option<&'a [crate::basis::ContractedGaussian]>,
    pub root: usize,
    pub triplet: bool,
}

impl<'a> CIS<'a> {
    pub fn new(hf: &'a HFResult) -> Self {
        Self {
            hf,
            n_roots: 5,
            basis: None,
            integral_cache: None,
            root: 0,
            triplet: false,
        }
    }

    pub fn with_roots(mut self, n_roots: usize) -> Self {
        self.n_roots = n_roots;
        self
    }

    pub fn with_basis(mut self, basis: &'a [crate::basis::ContractedGaussian]) -> Self {
        self.basis = Some(basis);
        self
    }
    pub fn with_root(mut self, root: usize) -> Self {
        self.root = root;
        self
    }
    pub fn with_triplet(mut self, triplet: bool) -> Self {
        self.triplet = triplet;
        self
    }
}

impl<'a> PostHFMethod for CIS<'a> {
    type Output = CISResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let n_occ = self.hf.n_occ;
        let n_basis = self.hf.orbital_energies.len();
        if !self.hf.converged
            || n_basis <= n_occ
            || self.n_roots == 0
            || self.hf.mo_coefficients.cols != n_basis
            || self.hf.orbital_energies.iter().any(|v| !v.is_finite())
        {
            return Err("CIS needs a converged compatible RHF reference and nonzero roots".into());
        }
        let n_virt = n_basis - n_occ;

        if n_occ == 0 || n_virt == 0 {
            return Err("CIS requires at least one occupied and one virtual orbital".to_string());
        }

        let dim = n_occ * n_virt;
        let n_roots = self.n_roots.min(dim);
        if self.root >= n_roots {
            return Err("selected CIS root exceeds computed excited states".into());
        }
        if (dim as u128).pow(2) * 8 > 128 * 1024 * 1024 {
            return Err("dense CIS Hamiltonian exceeds 128 MiB".into());
        }
        if self
            .basis
            .is_some_and(|b| b.len() != self.hf.mo_coefficients.rows)
        {
            return Err("CIS dipole basis dimension mismatch".into());
        }

        let mo_eri = if let Some(cache) = self.integral_cache {
            Cow::Borrowed(cache.full().map_err(|e| e.to_string())?.as_ref())
        } else {
            Cow::Owned(transform_eri_ao_to_mo(
                &self.hf.mo_coefficients,
                &self.hf.eri,
            ))
        };
        let eps = &self.hf.orbital_energies;

        // Construct CIS Hamiltonian matrix A:
        // A_{ia, jb} = (eps_a - eps_i) * delta_{ij} delta_{ab} + 2*(ia|bj) - (ij|ba)
        let mut a_mat = Matrix::zeros(dim, dim);

        for i in 0..n_occ {
            for a in 0..n_virt {
                let a_mo = n_occ + a;
                let row = i * n_virt + a;
                for j in 0..n_occ {
                    for b in 0..n_virt {
                        let b_mo = n_occ + b;
                        let col = j * n_virt + b;

                        let mut val = (if self.triplet { 0. } else { 2. })
                            * mo_eri.get(i, a_mo, b_mo, j)
                            - mo_eri.get(i, j, b_mo, a_mo);
                        if i == j && a == b {
                            val += eps[a_mo] - eps[i];
                        }
                        a_mat[(row, col)] = val;
                    }
                }
            }
        }

        // Symmetrize A matrix
        for r in 0..dim {
            for c in (r + 1)..dim {
                let avg = 0.5 * (a_mat[(r, c)] + a_mat[(c, r)]);
                a_mat[(r, c)] = avg;
                a_mat[(c, r)] = avg;
            }
        }

        let (evals, evecs) = a_mat.eigh();

        let ev_per_hartree = 27.211_386_245_988;
        let mut states = Vec::with_capacity(n_roots);
        let dipoles = if self.triplet {
            None
        } else {
            self.basis.map(|basis| {
                std::array::from_fn::<_, 3, _>(|comp| {
                    self.hf
                        .mo_coefficients
                        .transpose()
                        .dot(&crate::properties::compute_dipole_matrix(basis, comp))
                        .dot(&self.hf.mo_coefficients)
                })
            })
        };

        for k in 0..n_roots {
            let omega = evals[k];
            let omega_ev = omega * ev_per_hartree;
            let amps: Vec<f64> = (0..dim).map(|r| evecs[(r, k)]).collect();

            // Exact transition dipole: \mu_{0k} = \sqrt{2} \sum_{ia} c_{ia}^{(k)} \langle i | \hat{\mu} | a \rangle
            let mut td = [0.0; 3];
            if let Some(dipoles) = &dipoles {
                for comp in 0..3 {
                    let d_mo = &dipoles[comp];
                    let mut comp_val = 0.0;
                    for i in 0..n_occ {
                        for a in 0..n_virt {
                            let a_mo = n_occ + a;
                            let c_ia = amps[i * n_virt + a];
                            comp_val += c_ia * d_mo[(i, a_mo)];
                        }
                    }
                    td[comp] = std::f64::consts::SQRT_2 * comp_val;
                }
            }

            let td_sq = td[0] * td[0] + td[1] * td[1] + td[2] * td[2];
            let osc_strength = (2.0 / 3.0) * omega.max(0.0) * td_sq;

            states.push(CISEstate {
                state_index: k + 1,
                excitation_energy_hartree: omega,
                excitation_energy_ev: omega_ev,
                oscillator_strength: osc_strength,
                transition_dipole: td,
                amplitudes: amps,
            });
        }

        Ok(CISResult {
            reference_energy: self.hf.total_energy,
            excited_states: states,
            n_occ,
            n_virt,
            root: self.root,
            triplet: self.triplet,
        })
    }
}

impl MethodMeta for CIS<'_> {
    const METHOD: MethodId = MethodId::Cis;
}
impl<'a> ContextMethod<'a> for CIS<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        if options.frozen_core != 0
            || options.frozen_virtual != 0
            || options.density_fitting
            || options.active.is_some()
        {
            return Err(HfError::Unsupported {
                method: "CIS".into(),
                detail: "CIS adapter requires a full RHF orbital space without DF/CAS".into(),
            });
        }
        let mut engine = Self::new(ctx.rhf()?)
            .with_basis(ctx.basis)
            .with_roots(options.n_roots)
            .with_root(options.root)
            .with_triplet(options.cis_triplet);
        engine.integral_cache = Some(&ctx.ints);
        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{Atom, STO_3G_GBS, build_basis, parse_gaussian94};
    use crate::diis::ConvAlgo;
    use crate::scf::run_scf_general;

    #[test]
    fn test_h2_cis() {
        let basis_map = parse_gaussian94(STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, -0.7]),
            Atom::new("H", [0.0, 0.0, 0.7]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let hf = run_scf_general(&atoms, &basis, 0, 50, 1e-12, ConvAlgo::Adiissoscf, None)
            .expect("HF failed");

        let cis = CIS::new(&hf).with_roots(1).compute().expect("CIS failed");
        assert_eq!(cis.excited_states.len(), 1);
        assert!(cis.excited_states[0].excitation_energy_hartree > 0.0);
        assert!(cis.excited_states[0].excitation_energy_ev > 5.0);
    }
}
