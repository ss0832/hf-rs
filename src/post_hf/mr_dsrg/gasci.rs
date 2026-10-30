// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Generalized Active Space (`GASCI`, `GASSCF`) and Seniority-Zero Doubly Occupied CI (`DOCI`).
//!
//! - `GASCI` / `GASSCF`: Partition active orbitals into up to 6 Generalized Active Spaces
//!   (`GAS1..GAS6`) with individual electron occupation bounds $[N_{\min}^{(g)}, N_{\max}^{(g)}]$,
//!   coupled with 1-step or macro-iteration orbital optimization (`GASSCF`) including inter-GAS
//!   orbital rotations.
//! - `DOCI`: Exact seniority-zero ($\Omega = 0$) Configuration Interaction where every spatial
//!   orbital is either doubly occupied ($|\uparrow\downarrow\rangle$) or empty ($|0\rangle$).

use crate::basis::{Atom, ContractedGaussian};
use crate::fci::slater_condon::slater_condon_element;
use crate::fci::string::{DeterminantSpace, SpinString};
use crate::mo_integrals::{transform_1e_ao_to_mo, transform_eri_ao_to_mo};
use crate::matrix::Matrix;
use crate::post_hf::grad_common::{compute_rhf_total_rdm_analytical_gradient, compute_subset_rdms};
use crate::post_hf::mr_dsrg::cumulants::exp_antisymmetric;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::scf::HFResult;

/// Definition of a single Generalized Active Space (`GAS1..GAS6`) subspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GASSubspace {
    /// Number of spatial orbitals in this GAS subspace.
    pub n_orbitals: usize,
    /// Minimum number of electrons (alpha + beta) allowed in this subspace.
    pub min_electrons: usize,
    /// Maximum number of electrons (alpha + beta) allowed in this subspace.
    pub max_electrons: usize,
}

impl GASSubspace {
    pub fn new(n_orbitals: usize, min_electrons: usize, max_electrons: usize) -> Self {
        Self {
            n_orbitals,
            min_electrons,
            max_electrons,
        }
    }
}

/// Options for Generalized Active Space CI (`GASCI`) and SCF (`GASSCF`).
#[derive(Clone, Debug)]
pub struct GASOptions {
    /// Number of doubly-occupied inactive core orbitals.
    pub n_core: usize,
    /// Partition of the active space into `GAS1..GAS6` subspaces.
    pub subspaces: Vec<GASSubspace>,
    /// Whether to optimize orbitals self-consistently (`GASSCF`).
    pub optimize_orbitals: bool,
    /// Maximum number of orbital optimization macro-iterations.
    pub max_macro_iter: usize,
    /// Energy and orbital gradient convergence threshold.
    pub tol: f64,
}

impl GASOptions {
    /// Creates a standard 2-subspace GAS partition (e.g. valence bonding GAS1 + antibonding GAS2).
    pub fn new(n_core: usize, subspaces: Vec<GASSubspace>, optimize_orbitals: bool) -> Self {
        Self {
            n_core,
            subspaces,
            optimize_orbitals,
            max_macro_iter: 40,
            tol: 1e-8,
        }
    }
}

/// Result of a `GASCI` or `GASSCF` calculation.
#[derive(Clone, Debug)]
pub struct GASResult {
    pub is_scf: bool,
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub total_energy: f64,
    pub n_determinants: usize,
    pub converged: bool,
    pub iterations: usize,
    pub one_rdm_mo: Matrix,
    pub two_rdm_mo: Vec<f64>,
    pub mo_coefficients: Matrix,
}

impl PostHFResult for GASResult {
    fn method_name(&self) -> &'static str {
        if self.is_scf {
            "GASSCF"
        } else {
            "GASCI"
        }
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }

    fn one_rdm(&self) -> Option<&Matrix> {
        Some(&self.one_rdm_mo)
    }
}

/// Generates all determinants satisfying the `GAS1..GAS6` subspace electron occupation bounds.
pub fn generate_gas_determinants(
    n_basis: usize,
    n_occ: usize,
    n_core: usize,
    subspaces: &[GASSubspace],
) -> Vec<(SpinString, SpinString)> {
    let full_space = DeterminantSpace::new(n_basis, n_occ, n_occ);
    let total_gas_orbs: usize = subspaces.iter().map(|s| s.n_orbitals).sum();
    let virt_start = (n_core + total_gas_orbs).min(n_basis);

    // Precompute orbital ranges for each GAS subspace
    let mut ranges = Vec::with_capacity(subspaces.len());
    let mut offset = n_core;
    for sub in subspaces {
        let end = (offset + sub.n_orbitals).min(n_basis);
        ranges.push((offset, end, sub.min_electrons, sub.max_electrons));
        offset = end;
    }

    let mut gas_dets = Vec::new();
    for &sa in &full_space.alpha_strings {
        let core_ok_a = (0..n_core).all(|i| sa.is_occupied(i));
        let virt_ok_a = (virt_start..n_basis).all(|a| !sa.is_occupied(a));
        if !core_ok_a || !virt_ok_a {
            continue;
        }
        for &sb in &full_space.beta_strings {
            let core_ok_b = (0..n_core).all(|i| sb.is_occupied(i));
            let virt_ok_b = (virt_start..n_basis).all(|a| !sb.is_occupied(a));
            if !core_ok_b || !virt_ok_b {
                continue;
            }

            let mut gas_ok = true;
            for &(start, end, min_e, max_e) in &ranges {
                let mut n_e = 0usize;
                for p in start..end {
                    if sa.is_occupied(p) {
                        n_e += 1;
                    }
                    if sb.is_occupied(p) {
                        n_e += 1;
                    }
                }
                if n_e < min_e || n_e > max_e {
                    gas_ok = false;
                    break;
                }
            }
            if gas_ok {
                gas_dets.push((sa, sb));
            }
        }
    }

    if gas_dets.is_empty() {
        let hf_mask = (1u64 << n_occ) - 1;
        gas_dets.push((SpinString(hf_mask), SpinString(hf_mask)));
    }
    gas_dets
}

/// Generalized Active Space CI (`GASCI`) and SCF (`GASSCF`) solver.
pub struct GASCI<'a> {
    hf: &'a HFResult,
    options: GASOptions,
}

impl<'a> GASCI<'a> {
    pub fn new(hf: &'a HFResult, options: GASOptions) -> Self {
        Self { hf, options }
    }

    /// Creates a `GASSCF` instance with orbital optimization enabled.
    pub fn new_scf(hf: &'a HFResult, mut options: GASOptions) -> Self {
        options.optimize_orbitals = true;
        Self { hf, options }
    }
}

impl<'a> PostHFMethod for GASCI<'a> {
    type Output = GASResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let n_occ = self.hf.n_occ;
        let n_basis = self.hf.orbital_energies.len();
        let e_nuc = self.hf.nuclear_repulsion_energy;

        let gas_dets = generate_gas_determinants(
            n_basis,
            n_occ,
            self.options.n_core,
            &self.options.subspaces,
        );
        let n_det = gas_dets.len();

        // Map each orbital to its subspace class:
        // 0 = inactive core, 1..=G = GAS_g, G+1 = virtual
        let mut orb_class = vec![self.options.subspaces.len() + 1; n_basis];
        for i in 0..self.options.n_core.min(n_basis) {
            orb_class[i] = 0;
        }
        let mut offset = self.options.n_core;
        for (g_idx, sub) in self.options.subspaces.iter().enumerate() {
            let end = (offset + sub.n_orbitals).min(n_basis);
            for p in offset..end {
                orb_class[p] = g_idx + 1;
            }
            offset = end;
        }

        let mut c_mo = self.hf.mo_coefficients.clone();
        let mut last_energy = self.hf.total_energy;
        let mut converged = !self.options.optimize_orbitals;
        let mut iterations = 1;
        let mut rdm1 = Matrix::zeros(n_basis, n_basis);
        let mut rdm2 = vec![0.0; n_basis * n_basis * n_basis * n_basis];

        let max_iter = if self.options.optimize_orbitals {
            self.options.max_macro_iter.max(1)
        } else {
            1
        };

        for iter in 0..max_iter {
            iterations = iter + 1;
            let h_mo = transform_1e_ao_to_mo(&c_mo, &self.hf.core_hamiltonian);
            let mo_eri = transform_eri_ao_to_mo(&c_mo, &self.hf.eri);

            let mut h_ci = Matrix::zeros(n_det, n_det);
            for i in 0..n_det {
                let (sa_i, sb_i) = &gas_dets[i];
                for j in 0..=i {
                    let (sa_j, sb_j) = &gas_dets[j];
                    let mut val = slater_condon_element(sa_i, sb_i, sa_j, sb_j, &h_mo, &mo_eri);
                    if i == j {
                        val += e_nuc;
                    }
                    h_ci[(i, j)] = val;
                    h_ci[(j, i)] = val;
                }
            }

            let (evals, evecs) = h_ci.eigh();
            let e_ci = evals[0];
            let coeffs: Vec<f64> = (0..n_det).map(|i| evecs[(i, 0)]).collect();
            let (cur_rdm1, cur_rdm2) = compute_subset_rdms(n_basis, &gas_dets, &coeffs);
            rdm1 = cur_rdm1;
            rdm2 = cur_rdm2;

            if !self.options.optimize_orbitals {
                last_energy = e_ci;
                break;
            }

            // Compute generalized Fock matrix F_{pq} = sum_r gamma_{pr} h_{qr} + sum_{rst} Gamma_{prst} (qr|st)
            let mut f_gen = Matrix::zeros(n_basis, n_basis);
            for p in 0..n_basis {
                for q in 0..n_basis {
                    let mut sum1 = 0.0;
                    for r in 0..n_basis {
                        sum1 += rdm1[(p, r)] * h_mo[(q, r)];
                    }
                    let mut sum2 = 0.0;
                    for r in 0..n_basis {
                        for s in 0..n_basis {
                            for t in 0..n_basis {
                                let g2 = rdm2[((p * n_basis + r) * n_basis + s) * n_basis + t];
                                if g2.abs() > 1e-15 {
                                    sum2 += g2 * mo_eri.get(q, r, s, t);
                                }
                            }
                        }
                    }
                    f_gen[(p, q)] = sum1 + sum2;
                }
            }

            // Non-redundant orbital gradient between distinct classes (including inter-GAS rotations!)
            let mut kappa = Matrix::zeros(n_basis, n_basis);
            let mut grad_norm_sq = 0.0;
            for p in 0..n_basis {
                for q in 0..p {
                    if orb_class[p] != orb_class[q] {
                        let g_pq = 2.0 * (f_gen[(q, p)] - f_gen[(p, q)]);
                        grad_norm_sq += g_pq * g_pq;
                        let hess_diag = (f_gen[(p, p)] + f_gen[(q, q)]).abs().max(0.5) * 4.0;
                        let step = (-g_pq / hess_diag).clamp(-0.2, 0.2);
                        kappa[(p, q)] = step;
                        kappa[(q, p)] = -step;
                    }
                }
            }

            let de = (e_ci - last_energy).abs();
            last_energy = e_ci;
            if iter > 0 && (de < self.options.tol && grad_norm_sq.sqrt() < self.options.tol * 10.0) {
                converged = true;
                break;
            }

            // Unitary orbital update U = exp(kappa) via Cayley / Taylor Padé
            let u_step = exp_antisymmetric(&kappa);
            c_mo = c_mo.dot(&u_step);
        }

        Ok(GASResult {
            is_scf: self.options.optimize_orbitals,
            reference_energy: self.hf.total_energy,
            correlation_energy: last_energy - self.hf.total_energy,
            total_energy: last_energy,
            n_determinants: n_det,
            converged,
            iterations,
            one_rdm_mo: rdm1,
            two_rdm_mo: rdm2,
            mo_coefficients: c_mo,
        })
    }
}

/// Computes the exact analytical nuclear gradient of a `GASCI` or `GASSCF` wavefunction.
pub fn compute_gas_analytical_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    gas: &GASResult,
) -> Vec<[f64; 3]> {
    let mut hf_rot = hf.clone();
    hf_rot.mo_coefficients = gas.mo_coefficients.clone();
    compute_rhf_total_rdm_analytical_gradient(atoms, basis, &hf_rot, &gas.one_rdm_mo, &gas.two_rdm_mo)
}

// ============================================================================
// 2. Doubly Occupied Configuration Interaction (DOCI, Seniority Omega = 0)
// ============================================================================

/// Result of a Doubly Occupied Configuration Interaction (`DOCI`) calculation.
#[derive(Clone, Debug)]
pub struct DOCIResult {
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub total_energy: f64,
    pub n_pair_states: usize,
    pub seniority_zero: bool,
    pub one_rdm_mo: Matrix,
    pub two_rdm_mo: Vec<f64>,
}

impl PostHFResult for DOCIResult {
    fn method_name(&self) -> &'static str {
        "DOCI"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }

    fn one_rdm(&self) -> Option<&Matrix> {
        Some(&self.one_rdm_mo)
    }
}

/// Doubly Occupied Configuration Interaction (`DOCI`) solver: exact seniority-zero ($\Omega = 0$) CI.
pub struct DOCI<'a> {
    hf: &'a HFResult,
}

impl<'a> DOCI<'a> {
    pub fn new(hf: &'a HFResult) -> Self {
        Self { hf }
    }
}

impl crate::registry::MethodMeta for DOCI<'_> {
    const METHOD: crate::registry::MethodId = crate::registry::MethodId::Doci;
}
impl<'a> crate::registry::ContextMethod<'a> for DOCI<'a> {
    fn from_context(ctx: &'a crate::context::PostHFContext<'a>, options: &crate::options::PostHFOptions) -> Result<Self, crate::error::HfError> {
        options.validate()?;
        if options.frozen_core != 0 || options.frozen_virtual != 0 || options.density_fitting {
            return Err(crate::error::HfError::Unsupported { method: "DOCI".into(),
                detail: "the DOCI adapter requires the complete exact RHF orbital space".into() });
        }
        Ok(Self::new(ctx.rhf()?))
    }
}

impl<'a> PostHFMethod for DOCI<'a> {
    type Output = DOCIResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let n_occ = self.hf.n_occ;
        let n_basis = self.hf.orbital_energies.len();
        let h_mo = transform_1e_ao_to_mo(&self.hf.mo_coefficients, &self.hf.core_hamiltonian);
        let mo_eri = transform_eri_ao_to_mo(&self.hf.mo_coefficients, &self.hf.eri);
        let e_nuc = self.hf.nuclear_repulsion_energy;

        // In DOCI, alpha and beta strings are identical: |I_alpha> = |I_beta>
        let full_space = DeterminantSpace::new(n_basis, n_occ, n_occ);
        let pair_dets: Vec<(SpinString, SpinString)> = full_space
            .alpha_strings
            .iter()
            .map(|&s| (s, s))
            .collect();
        let n_pairs = pair_dets.len();

        let mut h_doci = Matrix::zeros(n_pairs, n_pairs);
        for i in 0..n_pairs {
            let (sa_i, sb_i) = &pair_dets[i];
            for j in 0..=i {
                let (sa_j, sb_j) = &pair_dets[j];
                let mut val = slater_condon_element(sa_i, sb_i, sa_j, sb_j, &h_mo, &mo_eri);
                if i == j {
                    val += e_nuc;
                }
                h_doci[(i, j)] = val;
                h_doci[(j, i)] = val;
            }
        }

        let (evals, evecs) = h_doci.eigh();
        let e_doci = evals[0];
        let coeffs: Vec<f64> = (0..n_pairs).map(|i| evecs[(i, 0)]).collect();
        let (one_rdm_mo, two_rdm_mo) = compute_subset_rdms(n_basis, &pair_dets, &coeffs);

        Ok(DOCIResult {
            reference_energy: self.hf.total_energy,
            correlation_energy: e_doci - self.hf.total_energy,
            total_energy: e_doci,
            n_pair_states: n_pairs,
            seniority_zero: true,
            one_rdm_mo,
            two_rdm_mo,
        })
    }
}

/// Computes the exact analytical nuclear gradient of the `DOCI` wavefunction via Z-vector response.
pub fn compute_doci_analytical_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    hf: &HFResult,
    doci: &DOCIResult,
) -> Vec<[f64; 3]> {
    compute_rhf_total_rdm_analytical_gradient(atoms, basis, hf, &doci.one_rdm_mo, &doci.two_rdm_mo)
}
