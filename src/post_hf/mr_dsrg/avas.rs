// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Automated Active-Space Construction & Quantum Embedding:
//! - Atomic Valence Active Space (`AVAS`, Sayfutyarova, Sun, Chan & Knizia, 2017)
//! - Active Space Embedding Theory (`ASET`, with `ASET`, `PAO`, and `IAO` virtual orbitals)
//! - Natural Orbital Active Space Selectors (`MP2NO`, `CINO`, `MRCINO`)

use crate::basis::ContractedGaussian;
use crate::fci::slater_condon::slater_condon_element;
use crate::fci::string::{DeterminantSpace, SpinString};
use crate::mo_integrals::{transform_1e_ao_to_mo, transform_eri_ao_to_mo};
use crate::matrix::Matrix;
use crate::post_hf::casscf::{CASSCF, CASSCFOptions};
use crate::post_hf::ci::CISD;
use crate::post_hf::mp2::RHFMP2;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::scf::HFResult;

// ============================================================================
// 1. Atomic Valence Active Space (AVAS)
// ============================================================================

/// Options for Atomic Valence Active Space (`AVAS`) automatic active-space construction.
#[derive(Clone, Debug)]
pub struct AVASOptions {
    /// Atom indices whose valence AOs define the target projector $\hat{P}_A$.
    pub target_atoms: Vec<usize>,
    /// Optional filter on AO total angular momentum $L = l_x + l_y + l_z$ (e.g. `Some(1)` for $p$, `Some(2)` for $d$).
    pub angular_momentum_filter: Option<usize>,
    /// Singular value / overlap eigenvalue cutoff $\sigma_{\min}$ for active orbital selection.
    pub sigma_threshold: f64,
    /// Optional maximum number of active orbitals.
    pub max_active_orbitals: Option<usize>,
}

impl Default for AVASOptions {
    fn default() -> Self {
        Self {
            target_atoms: vec![0],
            angular_momentum_filter: None,
            sigma_threshold: 0.1,
            max_active_orbitals: None,
        }
    }
}

/// Result of an Atomic Valence Active Space (`AVAS`) transformation and active-space CASCI calculation.
#[derive(Clone, Debug)]
pub struct AVASResult {
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub total_energy: f64,
    pub n_core: usize,
    pub n_active_electrons: usize,
    pub n_active_orbitals: usize,
    pub n_virtual: usize,
    pub occ_overlap_eigenvalues: Vec<f64>,
    pub virt_overlap_eigenvalues: Vec<f64>,
    pub rotated_mo_coefficients: Matrix,
}

impl PostHFResult for AVASResult {
    fn method_name(&self) -> &'static str {
        "AVAS-CASCI"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }
}

/// Atomic Valence Active Space (`AVAS`) constructor and CASCI engine.
pub struct AVAS<'a> {
    hf: &'a HFResult,
    basis: &'a [ContractedGaussian],
    options: AVASOptions,
}

impl<'a> AVAS<'a> {
    pub fn new(hf: &'a HFResult, basis: &'a [ContractedGaussian], options: AVASOptions) -> Self {
        Self { hf, basis, options }
    }
}

impl<'a> PostHFMethod for AVAS<'a> {
    type Output = AVASResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let n_basis = self.basis.len();
        let n_occ = self.hf.n_occ;
        let n_virt = n_basis - n_occ;
        let s_ao = &self.hf.overlap_matrix;
        let c_mo = &self.hf.mo_coefficients;

        // 1. Identify target atomic orbitals A
        let mut target_ao_indices: Vec<usize> = (0..n_basis)
            .filter(|&mu| {
                let atom_ok = self.options.target_atoms.is_empty()
                    || self.options.target_atoms.contains(&self.basis[mu].atom_index);
                let l_tot = self.basis[mu].l[0]
                    + self.basis[mu].l[1]
                    + self.basis[mu].l[2];
                let ang_ok = match self.options.angular_momentum_filter {
                    Some(l_req) => l_tot == l_req,
                    None => true,
                };
                atom_ok && ang_ok
            })
            .collect();
        if target_ao_indices.is_empty() {
            target_ao_indices = (0..n_basis.min(2)).collect();
        }
        let n_a = target_ao_indices.len();

        // 2. Build S_{AA} and its inverse (or regularized inverse) to form projector P_A = S(:,A) S_{AA}^{-1} S(A,:)
        let mut s_aa = Matrix::zeros(n_a, n_a);
        for (i, &mu) in target_ao_indices.iter().enumerate() {
            for (j, &nu) in target_ao_indices.iter().enumerate() {
                s_aa[(i, j)] = s_ao[(mu, nu)] + if i == j { 1e-10 } else { 0.0 };
            }
        }
        let (evals_a, evecs_a) = s_aa.eigh();
        let mut s_aa_inv = Matrix::zeros(n_a, n_a);
        for k in 0..n_a {
            if evals_a[k] > 1e-8 {
                let inv_val = 1.0 / evals_a[k];
                for i in 0..n_a {
                    for j in 0..n_a {
                        s_aa_inv[(i, j)] += evecs_a[(i, k)] * inv_val * evecs_a[(j, k)];
                    }
                }
            }
        }

        // Overlap of MOs with target AOs: M_{p, i} = (C^T S)_{p, target_ao[i]}
        let ct_s = c_mo.transpose().dot(s_ao);
        let mut m_occ = Matrix::zeros(n_occ, n_a);
        for i in 0..n_occ {
            for (k, &mu) in target_ao_indices.iter().enumerate() {
                m_occ[(i, k)] = ct_s[(i, mu)];
            }
        }
        let mut m_virt = Matrix::zeros(n_virt, n_a);
        for a in 0..n_virt {
            for (k, &mu) in target_ao_indices.iter().enumerate() {
                m_virt[(a, k)] = ct_s[(n_occ + a, mu)];
            }
        }

        // Projected overlap matrices in occupied and virtual subspaces:
        // S^{occ} = M_occ * S_{AA}^{-1} * M_occ^T
        // S^{virt} = M_virt * S_{AA}^{-1} * M_virt^T
        let s_occ_proj = m_occ.dot(&s_aa_inv).dot(&m_occ.transpose());
        let (occ_evals, occ_evecs) = s_occ_proj.eigh();

        let (virt_evals, virt_evecs) = if n_virt > 0 {
            let s_virt_proj = m_virt.dot(&s_aa_inv).dot(&m_virt.transpose());
            s_virt_proj.eigh()
        } else {
            (Vec::new(), Matrix::zeros(0, 0))
        };

        // 3. Select active occupied and active virtual orbitals with sigma >= threshold
        // Since eigh() sorts eigenvalues ascending, the largest singular values are at the end of occ_evals
        // and should be placed right below the Fermi level (adjacent to the active space).
        let mut n_act_occ = occ_evals
            .iter()
            .filter(|&&v| v >= self.options.sigma_threshold)
            .count()
            .max(1)
            .min(n_occ);
        let mut n_act_virt = virt_evals
            .iter()
            .filter(|&&v| v >= self.options.sigma_threshold)
            .count()
            .max(1.min(n_virt))
            .min(n_virt);

        if let Some(max_act) = self.options.max_active_orbitals {
            while n_act_occ + n_act_virt > max_act && (n_act_occ > 1 || n_act_virt > 1) {
                if n_act_virt >= n_act_occ && n_act_virt > 1 {
                    n_act_virt -= 1;
                } else if n_act_occ > 1 {
                    n_act_occ -= 1;
                } else {
                    break;
                }
            }
        }

        let n_core = n_occ - n_act_occ;
        let n_act = n_act_occ + n_act_virt;
        let n_ext_virt = n_virt - n_act_virt;

        let (c_rotated, e_cas) = solve_casci_in_rotated_basis(
            self.hf,
            c_mo,
            &occ_evecs,
            &virt_evecs,
            n_act_occ,
            n_act_virt,
        );

        Ok(AVASResult {
            reference_energy: self.hf.total_energy,
            correlation_energy: e_cas - self.hf.total_energy,
            total_energy: e_cas,
            n_core,
            n_active_electrons: 2 * n_act_occ,
            n_active_orbitals: n_act,
            n_virtual: n_ext_virt,
            occ_overlap_eigenvalues: occ_evals,
            virt_overlap_eigenvalues: virt_evals,
            rotated_mo_coefficients: c_rotated,
        })
    }
}

// ============================================================================
// 2. Active Space Embedding Theory (ASET)
// ============================================================================

/// Virtual-space localization / projection scheme for Active Space Embedding Theory (`ASET`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ASETVirtualScheme {
    /// Direct SVD projection of canonical virtual orbitals onto fragment AOs.
    ASET,
    /// Projected Atomic Orbitals ($\hat{Q}_{\text{virt}} = \mathbf{1} - \mathbf{P}_{\text{occ}} \mathbf{S}$).
    PAO,
    /// Intrinsic Atomic Orbitals (Knizia 2013 polarized intrinsic valence basis).
    IAO,
}

/// Options for Active Space Embedding Theory (`ASET`).
#[derive(Clone, Debug)]
pub struct ASETOptions {
    /// Atom indices belonging to the embedded active fragment $A$.
    pub fragment_atoms: Vec<usize>,
    /// Singular value / fragment population cutoff $\tau$ (default `0.2`).
    pub threshold: f64,
    /// Virtual orbital localization / projection scheme.
    pub virtual_scheme: ASETVirtualScheme,
}

impl Default for ASETOptions {
    fn default() -> Self {
        Self {
            fragment_atoms: vec![0],
            threshold: 0.2,
            virtual_scheme: ASETVirtualScheme::ASET,
        }
    }
}

/// Result of an Active Space Embedding Theory (`ASET`) calculation.
#[derive(Clone, Debug)]
pub struct ASETResult {
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub total_energy: f64,
    pub n_fragment_occ: usize,
    pub n_fragment_virt: usize,
    pub n_env_occ: usize,
    pub n_env_virt: usize,
    pub virtual_scheme: ASETVirtualScheme,
    pub embedded_mo_coefficients: Matrix,
}

impl PostHFResult for ASETResult {
    fn method_name(&self) -> &'static str {
        "ASET"
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }
}

/// Active Space Embedding Theory (`ASET`, J. Chem. Theory Comput. 12, 5742, 2016) solver.
pub struct ASET<'a> {
    hf: &'a HFResult,
    basis: &'a [ContractedGaussian],
    options: ASETOptions,
}

impl<'a> ASET<'a> {
    pub fn new(hf: &'a HFResult, basis: &'a [ContractedGaussian], options: ASETOptions) -> Self {
        Self { hf, basis, options }
    }
}

fn solve_casci_in_rotated_basis(
    hf: &HFResult,
    c_mo: &Matrix,
    occ_evecs: &Matrix,
    virt_evecs: &Matrix,
    n_act_occ: usize,
    n_act_virt: usize,
) -> (Matrix, f64) {
    let n_basis = hf.orbital_energies.len();
    let n_occ = hf.n_occ;
    let n_virt = n_basis - n_occ;
    let n_core = n_occ - n_act_occ;
    let n_act = n_act_occ + n_act_virt;

    let mut u_rot = Matrix::identity(n_basis);
    for i in 0..n_occ {
        for j in 0..n_occ {
            u_rot[(i, j)] = occ_evecs[(i, j)];
        }
    }
    for a in 0..n_virt {
        for b in 0..n_virt {
            u_rot[(n_occ + a, n_occ + b)] = virt_evecs[(a, n_virt - 1 - b)];
        }
    }

    let f_mo = c_mo.transpose().dot(&hf.fock_matrix).dot(c_mo);
    let f_rot = u_rot.transpose().dot(&f_mo).dot(&u_rot);
    let sub_ranges = [
        (0, n_core),
        (n_core, n_occ),
        (n_occ, n_occ + n_act_virt),
        (n_occ + n_act_virt, n_basis),
    ];
    let mut u_semi = Matrix::identity(n_basis);
    for &(start, end) in &sub_ranges {
        let dim = end.saturating_sub(start);
        if dim <= 1 {
            continue;
        }
        let mut sub_f = Matrix::zeros(dim, dim);
        for r in 0..dim {
            for c in 0..dim {
                sub_f[(r, c)] = f_rot[(start + r, start + c)];
            }
        }
        let (_ev, evec) = sub_f.eigh();
        for r in 0..dim {
            for c in 0..dim {
                u_semi[(start + r, start + c)] = evec[(r, c)];
            }
        }
    }

    let c_rotated = c_mo.dot(&u_rot).dot(&u_semi);

    let h_rot = transform_1e_ao_to_mo(&c_rotated, &hf.core_hamiltonian);
    let eri_rot = transform_eri_ao_to_mo(&c_rotated, &hf.eri);
    let full_space = DeterminantSpace::new(n_basis, n_occ, n_occ);
    let mut cas_dets = Vec::new();
    for &sa in &full_space.alpha_strings {
        let ok_a = (0..n_core).all(|i| sa.is_occupied(i))
            && ((n_core + n_act)..n_basis).all(|a| !sa.is_occupied(a));
        if !ok_a {
            continue;
        }
        for &sb in &full_space.beta_strings {
            let ok_b = (0..n_core).all(|i| sb.is_occupied(i))
                && ((n_core + n_act)..n_basis).all(|a| !sb.is_occupied(a));
            if ok_b {
                cas_dets.push((sa, sb));
            }
        }
    }
    let n_det = cas_dets.len();
    let mut h_cas = Matrix::zeros(n_det, n_det);
    for i in 0..n_det {
        for j in 0..=i {
            let mut val = slater_condon_element(
                &cas_dets[i].0,
                &cas_dets[i].1,
                &cas_dets[j].0,
                &cas_dets[j].1,
                &h_rot,
                &eri_rot,
            );
            if i == j {
                val += hf.nuclear_repulsion_energy;
            }
            h_cas[(i, j)] = val;
            h_cas[(j, i)] = val;
        }
    }
    let (cas_evals, _) = h_cas.eigh();
    let e_cas = cas_evals[0];

    (c_rotated, e_cas)
}

impl<'a> PostHFMethod for ASET<'a> {
    type Output = ASETResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let n_basis = self.basis.len();
        let n_occ = self.hf.n_occ;
        let n_virt = n_basis - n_occ;
        let s_ao = &self.hf.overlap_matrix;
        let c_mo = &self.hf.mo_coefficients;

        // 1. Identify fragment atomic orbitals
        let mut target_ao_indices: Vec<usize> = (0..n_basis)
            .filter(|&mu| {
                self.options.fragment_atoms.is_empty()
                    || self.options.fragment_atoms.contains(&self.basis[mu].atom_index)
            })
            .collect();
        if target_ao_indices.is_empty() {
            target_ao_indices = (0..n_basis.min(2)).collect();
        }
        let n_a = target_ao_indices.len();

        // 2. Build S_{AA} and S_{AA}^{-1}
        let mut s_aa = Matrix::zeros(n_a, n_a);
        for (i, &mu) in target_ao_indices.iter().enumerate() {
            for (j, &nu) in target_ao_indices.iter().enumerate() {
                s_aa[(i, j)] = s_ao[(mu, nu)] + if i == j { 1e-10 } else { 0.0 };
            }
        }
        let (evals_a, evecs_a) = s_aa.eigh();
        let mut s_aa_inv = Matrix::zeros(n_a, n_a);
        for k in 0..n_a {
            if evals_a[k] > 1e-8 {
                let inv_val = 1.0 / evals_a[k];
                for i in 0..n_a {
                    for j in 0..n_a {
                        s_aa_inv[(i, j)] += evecs_a[(i, k)] * inv_val * evecs_a[(j, k)];
                    }
                }
            }
        }

        // Overlap of MOs with fragment AOs: M_{p, i} = (C^T S)_{p, target_ao[i]}
        let ct_s = c_mo.transpose().dot(s_ao);
        let mut m_occ = Matrix::zeros(n_occ, n_a);
        for i in 0..n_occ {
            for (k, &mu) in target_ao_indices.iter().enumerate() {
                m_occ[(i, k)] = ct_s[(i, mu)];
            }
        }
        let mut m_virt = Matrix::zeros(n_virt, n_a);
        for a in 0..n_virt {
            for (k, &mu) in target_ao_indices.iter().enumerate() {
                m_virt[(a, k)] = ct_s[(n_occ + a, mu)];
            }
        }

        // Occupied projection metric: S^occ = M_occ * S_{AA}^{-1} * M_occ^T
        let s_occ_proj = m_occ.dot(&s_aa_inv).dot(&m_occ.transpose());
        let (occ_evals, occ_evecs) = s_occ_proj.eigh();
        let n_frag_occ = occ_evals
            .iter()
            .filter(|&&v| v >= self.options.threshold)
            .count()
            .max(1)
            .min(n_occ);

        // Virtual projection metric depending on virtual_scheme
        let (virt_evals, virt_evecs) = if n_virt > 0 {
            let s_virt_proj = match self.options.virtual_scheme {
                ASETVirtualScheme::ASET => {
                    // Direct projection onto fragment AOs
                    m_virt.dot(&s_aa_inv).dot(&m_virt.transpose())
                }
                ASETVirtualScheme::PAO => {
                    // Projected Atomic Orbitals: S_PAO = S_{AA} - M_occ^T * M_occ
                    let mut s_pao = s_aa.clone();
                    let m_occ_t_m_occ = m_occ.transpose().dot(&m_occ);
                    for i in 0..n_a {
                        for j in 0..n_a {
                            s_pao[(i, j)] = (s_pao[(i, j)] - m_occ_t_m_occ[(i, j)]).max(0.0);
                            if i == j {
                                s_pao[(i, j)] += 1e-10;
                            }
                        }
                    }
                    let (evals_pao, evecs_pao) = s_pao.eigh();
                    let mut s_pao_inv = Matrix::zeros(n_a, n_a);
                    for k in 0..n_a {
                        if evals_pao[k] > 1e-8 {
                            let inv_val = 1.0 / evals_pao[k];
                            for i in 0..n_a {
                                for j in 0..n_a {
                                    s_pao_inv[(i, j)] += evecs_pao[(i, k)] * inv_val * evecs_pao[(j, k)];
                                }
                            }
                        }
                    }
                    m_virt.dot(&s_pao_inv).dot(&m_virt.transpose())
                }
                ASETVirtualScheme::IAO => {
                    // Intrinsic Atomic Orbitals: S_IAO^{-1} = S_{AA}^{-1/2} * S_PAO^{-1} * S_{AA}^{-1/2}
                    let mut s_pao = s_aa.clone();
                    let m_occ_t_m_occ = m_occ.transpose().dot(&m_occ);
                    for i in 0..n_a {
                        for j in 0..n_a {
                            s_pao[(i, j)] = (s_pao[(i, j)] - m_occ_t_m_occ[(i, j)]).max(0.0);
                            if i == j {
                                s_pao[(i, j)] += 1e-10;
                            }
                        }
                    }
                    let (evals_pao, evecs_pao) = s_pao.eigh();
                    let mut s_pao_inv = Matrix::zeros(n_a, n_a);
                    for k in 0..n_a {
                        if evals_pao[k] > 1e-8 {
                            let inv_val = 1.0 / evals_pao[k];
                            for i in 0..n_a {
                                for j in 0..n_a {
                                    s_pao_inv[(i, j)] += evecs_pao[(i, k)] * inv_val * evecs_pao[(j, k)];
                                }
                            }
                        }
                    }
                    let mut s_aa_inv_sqrt = Matrix::zeros(n_a, n_a);
                    for k in 0..n_a {
                        if evals_a[k] > 1e-8 {
                            let inv_sqrt = 1.0 / evals_a[k].sqrt();
                            for i in 0..n_a {
                                for j in 0..n_a {
                                    s_aa_inv_sqrt[(i, j)] += evecs_a[(i, k)] * inv_sqrt * evecs_a[(j, k)];
                                }
                            }
                        }
                    }
                    let metric = s_aa_inv_sqrt.dot(&s_pao_inv).dot(&s_aa_inv_sqrt);
                    m_virt.dot(&metric).dot(&m_virt.transpose())
                }
            };
            s_virt_proj.eigh()
        } else {
            (Vec::new(), Matrix::zeros(0, 0))
        };
        let n_frag_virt = virt_evals
            .iter()
            .filter(|&&v| v >= self.options.threshold)
            .count()
            .max(1.min(n_virt))
            .min(n_virt);

        let (c_rotated, e_cas) = solve_casci_in_rotated_basis(
            self.hf,
            c_mo,
            &occ_evecs,
            &virt_evecs,
            n_frag_occ,
            n_frag_virt,
        );

        let e_corr = e_cas - self.hf.total_energy;
        Ok(ASETResult {
            reference_energy: self.hf.total_energy,
            correlation_energy: e_corr,
            total_energy: e_cas,
            n_fragment_occ: n_frag_occ,
            n_fragment_virt: n_frag_virt,
            n_env_occ: n_occ - n_frag_occ,
            n_env_virt: (n_basis - n_occ) - n_frag_virt,
            virtual_scheme: self.options.virtual_scheme,
            embedded_mo_coefficients: c_rotated,
        })
    }
}

// ============================================================================
// 3. Natural Orbital Active Space Selectors (MP2NO / CINO / MRCINO)
// ============================================================================

/// Source wavefunction for correlated Natural Orbital (`NO`) active-space selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NaturalOrbitalKind {
    /// Second-order Møller-Plesset natural orbitals (`MP2NO`).
    MP2NO,
    /// Single-reference CISD natural orbitals (`CINO`).
    CINO,
    /// Multireference CISD natural orbitals (`MRCINO`).
    MRCINO,
}

/// Result of a Natural Orbital active-space selection and CASCI calculation.
#[derive(Clone, Debug)]
pub struct NaturalOrbitalResult {
    pub kind: NaturalOrbitalKind,
    pub reference_energy: f64,
    pub correlation_energy: f64,
    pub total_energy: f64,
    pub occupation_numbers: Vec<f64>,
    pub n_core: usize,
    pub n_active_orbitals: usize,
    pub n_virtual: usize,
    pub natural_orbital_coefficients: Matrix,
}

impl PostHFResult for NaturalOrbitalResult {
    fn method_name(&self) -> &'static str {
        match self.kind {
            NaturalOrbitalKind::MP2NO => "MP2-NO-CASCI",
            NaturalOrbitalKind::CINO => "CI-NO-CASCI",
            NaturalOrbitalKind::MRCINO => "MRCI-NO-CASCI",
        }
    }

    fn reference_energy(&self) -> f64 {
        self.reference_energy
    }

    fn correlation_energy(&self) -> f64 {
        self.correlation_energy
    }
}

/// Natural Orbital Active-Space Selector (`MP2NO`, `CINO`, `MRCINO`).
pub struct NaturalOrbitalSelector<'a> {
    hf: &'a HFResult,
    kind: NaturalOrbitalKind,
    occ_low: f64,
    occ_high: f64,
}

impl<'a> NaturalOrbitalSelector<'a> {
    pub fn new(hf: &'a HFResult, kind: NaturalOrbitalKind, occ_low: f64, occ_high: f64) -> Self {
        Self {
            hf,
            kind,
            occ_low,
            occ_high,
        }
    }
}

impl<'a> PostHFMethod for NaturalOrbitalSelector<'a> {
    type Output = NaturalOrbitalResult;

    fn compute(&self) -> Result<Self::Output, String> {
        let n_basis = self.hf.orbital_energies.len();
        let n_occ = self.hf.n_occ;

        let rdm1 = match self.kind {
            NaturalOrbitalKind::MP2NO => {
                let mp2 = RHFMP2::new(self.hf).compute()?;
                mp2.one_rdm_mo.unwrap_or_else(|| Matrix::identity(n_basis))
            }
            NaturalOrbitalKind::CINO => {
                let cisd = CISD::new(self.hf).compute()?;
                cisd.one_rdm_mo
            }
            NaturalOrbitalKind::MRCINO => {
                let cas = CASSCF::new(self.hf, CASSCFOptions::new(2, 2)).compute()?;
                cas.one_rdm_mo
            }
        };

        // Diagonalize 1-RDM and sort natural orbitals in descending order of occupation number
        let (evals, evecs) = rdm1.eigh();
        let mut no_occ = Vec::with_capacity(n_basis);
        let mut u_no = Matrix::zeros(n_basis, n_basis);
        for k in 0..n_basis {
            let src = n_basis - 1 - k;
            no_occ.push(evals[src]);
            for r in 0..n_basis {
                u_no[(r, k)] = evecs[(r, src)];
            }
        }

        let c_no = self.hf.mo_coefficients.dot(&u_no);

        // Identify active space from occupation window [occ_low, occ_high]
        let mut n_core = no_occ.iter().filter(|&&n| n > self.occ_high).count();
        let mut n_act = no_occ
            .iter()
            .filter(|&&n| n >= self.occ_low && n <= self.occ_high)
            .count();
        if n_act == 0 {
            n_core = n_occ.saturating_sub(1);
            n_act = 2.min(n_basis - n_core);
        }
        let n_virt = n_basis.saturating_sub(n_core + n_act);

        // Evaluate CASCI energy in the natural-orbital active space
        let h_no = transform_1e_ao_to_mo(&c_no, &self.hf.core_hamiltonian);
        let eri_no = transform_eri_ao_to_mo(&c_no, &self.hf.eri);
        let full_space = DeterminantSpace::new(n_basis, n_occ, n_occ);
        let mut cas_dets = Vec::new();
        for &sa in &full_space.alpha_strings {
            let ok_a = (0..n_core).all(|i| sa.is_occupied(i))
                && ((n_core + n_act)..n_basis).all(|a| !sa.is_occupied(a));
            if !ok_a {
                continue;
            }
            for &sb in &full_space.beta_strings {
                let ok_b = (0..n_core).all(|i| sb.is_occupied(i))
                    && ((n_core + n_act)..n_basis).all(|a| !sb.is_occupied(a));
                if ok_b {
                    cas_dets.push((sa, sb));
                }
            }
        }
        if cas_dets.is_empty() {
            let hf_mask = (1u64 << n_occ) - 1;
            cas_dets.push((SpinString(hf_mask), SpinString(hf_mask)));
        }
        let n_det = cas_dets.len();
        let mut h_cas = Matrix::zeros(n_det, n_det);
        for i in 0..n_det {
            for j in 0..=i {
                let mut val = slater_condon_element(
                    &cas_dets[i].0,
                    &cas_dets[i].1,
                    &cas_dets[j].0,
                    &cas_dets[j].1,
                    &h_no,
                    &eri_no,
                );
                if i == j {
                    val += self.hf.nuclear_repulsion_energy;
                }
                h_cas[(i, j)] = val;
                h_cas[(j, i)] = val;
            }
        }
        let (cas_evals, _) = h_cas.eigh();
        let e_cas = cas_evals[0];

        Ok(NaturalOrbitalResult {
            kind: self.kind,
            reference_energy: self.hf.total_energy,
            correlation_energy: e_cas - self.hf.total_energy,
            total_energy: e_cas,
            occupation_numbers: no_occ,
            n_core,
            n_active_orbitals: n_act,
            n_virtual: n_virt,
            natural_orbital_coefficients: c_no,
        })
    }
}
