// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Orbital space partitioning and indexing for Post-HF, CASSCF, and local correlation methods.

use std::ops::Range;

/// Classification of molecular orbitals in correlated and multi-reference calculations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OrbitalSpaceType {
    /// Deep core orbitals not correlated in Post-HF (e.g. 1s on C, N, O).
    FrozenCore,
    /// Inactive closed-shell occupied orbitals (in CASSCF) or correlated occupied orbitals.
    ActiveOccupied,
    /// Active virtual orbitals within a CASSCF active space.
    ActiveVirtual,
    /// Secondary / external virtual orbitals correlated in post-HF (MPn, CC, CI).
    ExternalVirtual,
    /// High-energy virtual orbitals frozen/truncated from correlation space.
    FrozenVirtual,
}

/// Defines the partitioning of molecular orbitals into core, active, external, and frozen subspaces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrbitalSpace {
    /// Total number of basis functions / molecular orbitals.
    pub n_basis: usize,
    /// Number of frozen core orbitals (lowest energy).
    pub n_frozen_core: usize,
    /// Number of active or correlated occupied orbitals.
    pub n_active_occ: usize,
    /// Number of active virtual orbitals (e.g. in CASSCF).
    pub n_active_virt: usize,
    /// Number of correlated external virtual orbitals.
    pub n_external_virt: usize,
    /// Number of frozen virtual orbitals (highest energy).
    pub n_frozen_virt: usize,
}

impl OrbitalSpace {
    /// Construct the legacy contiguous partition from an explicitly spin-resolved
    /// active space. MR solvers use ResolvedActiveSpace directly for selections.
    pub fn from_active_space(n_mo:usize,total_alpha:usize,total_beta:usize,
        active:&crate::options::ActiveSpace)->Result<Self,crate::error::HfError> {
        let resolved=active.resolve(n_mo,total_alpha,total_beta)?;
        if total_alpha!=total_beta || active.n_alpha!=active.n_beta || active.orbitals.is_some() {
            return Err(crate::error::HfError::InvalidInput("the contiguous closed-shell orbital partition cannot represent this active selection; use ResolvedActiveSpace".into()));
        }
        Ok(Self{n_basis:n_mo,n_frozen_core:resolved.inactive.len(),n_active_occ:active.n_alpha,
            n_active_virt:active.n_orb-active.n_alpha,n_external_virt:resolved.external.len(),n_frozen_virt:0})
    }
    /// Creates a standard closed-shell orbital space with no frozen or CAS partitioning.
    pub fn new_closed_shell(n_basis: usize, n_occ: usize) -> Self {
        assert!(n_occ <= n_basis, "n_occ ({}) cannot exceed n_basis ({})", n_occ, n_basis);
        Self {
            n_basis,
            n_frozen_core: 0,
            n_active_occ: n_occ,
            n_active_virt: 0,
            n_external_virt: n_basis - n_occ,
            n_frozen_virt: 0,
        }
    }

    /// Sets the number of frozen core orbitals.
    pub fn with_frozen_core(mut self, n_core: usize) -> Self {
        assert!(
            n_core <= self.n_occ(),
            "Frozen core count ({}) cannot exceed occupied orbitals ({})",
            n_core,
            self.n_occ()
        );
        let total_occ = self.n_occ();
        self.n_frozen_core = n_core;
        self.n_active_occ = total_occ - n_core;
        self
    }

    /// Sets the number of frozen virtual orbitals (truncated from the top).
    pub fn with_frozen_virtual(mut self, n_frozen_virt: usize) -> Self {
        assert!(
            n_frozen_virt <= self.n_virt(),
            "Frozen virtual count ({}) cannot exceed virtual orbitals ({})",
            n_frozen_virt,
            self.n_virt()
        );
        let total_virt = self.n_virt();
        self.n_frozen_virt = n_frozen_virt;
        self.n_external_virt = total_virt - self.n_active_virt - n_frozen_virt;
        self
    }

    /// Configures a Complete Active Space (CAS) with `n_active_orbitals` and `n_active_electrons`.
    ///
    /// For a closed-shell reference, `n_cas_occ = n_active_electrons / 2`.
    pub fn with_cas(mut self, n_active_orbitals: usize, n_active_electrons: usize) -> Self {
        assert!(
            n_active_electrons.is_multiple_of(2),
            "Active electrons must be even for closed-shell CAS (got {})",
            n_active_electrons
        );
        let n_cas_occ = n_active_electrons / 2;
        assert!(
            n_cas_occ <= self.n_occ(),
            "Active occupied count ({}) exceeds total occupied orbitals ({})",
            n_cas_occ,
            self.n_occ()
        );
        assert!(
            n_active_orbitals >= n_cas_occ,
            "Total active orbitals ({}) must be at least active occupied ({})",
            n_active_orbitals,
            n_cas_occ
        );
        let n_cas_virt = n_active_orbitals - n_cas_occ;
        assert!(
            n_cas_virt <= self.n_virt(),
            "Active virtual count ({}) exceeds total virtual orbitals ({})",
            n_cas_virt,
            self.n_virt()
        );

        let total_occ = self.n_occ();
        let total_virt = self.n_virt();

        self.n_frozen_core = total_occ - n_cas_occ;
        self.n_active_occ = n_cas_occ;
        self.n_active_virt = n_cas_virt;
        self.n_external_virt = total_virt - n_cas_virt - self.n_frozen_virt;

        self
    }

    /// Total number of occupied orbitals (frozen core + active occupied).
    #[inline]
    pub fn n_occ(&self) -> usize {
        self.n_frozen_core + self.n_active_occ
    }

    /// Total number of virtual orbitals (active + external + frozen virtual).
    #[inline]
    pub fn n_virt(&self) -> usize {
        self.n_active_virt + self.n_external_virt + self.n_frozen_virt
    }

    /// Total number of correlated occupied orbitals.
    #[inline]
    pub fn n_correlated_occ(&self) -> usize {
        self.n_active_occ
    }

    /// Total number of correlated virtual orbitals.
    #[inline]
    pub fn n_correlated_virt(&self) -> usize {
        self.n_active_virt + self.n_external_virt
    }

    /// Total number of active orbitals in CAS space ($n_{\text{act,occ}} + n_{\text{act,virt}}$).
    #[inline]
    pub fn n_cas_orbitals(&self) -> usize {
        self.n_active_occ + self.n_active_virt
    }

    /// Range of frozen core orbital indices: `0..n_frozen_core`.
    #[inline]
    pub fn frozen_core_range(&self) -> Range<usize> {
        0..self.n_frozen_core
    }

    /// Range of correlated occupied orbital indices: `n_frozen_core..n_occ`.
    #[inline]
    pub fn correlated_occ_range(&self) -> Range<usize> {
        self.n_frozen_core..self.n_occ()
    }

    /// Range of all occupied orbital indices: `0..n_occ`.
    #[inline]
    pub fn occupied_range(&self) -> Range<usize> {
        0..self.n_occ()
    }

    /// Range of CAS active virtual orbital indices.
    #[inline]
    pub fn active_virt_range(&self) -> Range<usize> {
        let start = self.n_occ();
        start..(start + self.n_active_virt)
    }

    /// Range of external correlated virtual orbital indices.
    #[inline]
    pub fn external_virt_range(&self) -> Range<usize> {
        let start = self.n_occ() + self.n_active_virt;
        start..(start + self.n_external_virt)
    }

    /// Range of all correlated virtual orbitals (active virtual + external virtual).
    #[inline]
    pub fn correlated_virt_range(&self) -> Range<usize> {
        self.n_occ()..(self.n_occ() + self.n_active_virt + self.n_external_virt)
    }

    /// Range of all virtual orbital indices: `n_occ..n_basis`.
    #[inline]
    pub fn virtual_range(&self) -> Range<usize> {
        self.n_occ()..self.n_basis
    }

    /// Range of complete active space (CAS) orbital indices (both occupied and virtual).
    #[inline]
    pub fn cas_range(&self) -> Range<usize> {
        self.n_frozen_core..(self.n_occ() + self.n_active_virt)
    }

    /// Range of frozen virtual orbital indices (top virtuals).
    #[inline]
    pub fn frozen_virt_range(&self) -> Range<usize> {
        (self.n_basis - self.n_frozen_virt)..self.n_basis
    }

    /// Returns the space classification for an arbitrary MO index.
    pub fn orbital_type(&self, mo_idx: usize) -> OrbitalSpaceType {
        assert!(mo_idx < self.n_basis, "MO index {} out of bounds ({})", mo_idx, self.n_basis);
        if mo_idx < self.n_frozen_core {
            OrbitalSpaceType::FrozenCore
        } else if mo_idx < self.n_occ() {
            OrbitalSpaceType::ActiveOccupied
        } else if mo_idx < self.n_occ() + self.n_active_virt {
            OrbitalSpaceType::ActiveVirtual
        } else if mo_idx < self.n_basis - self.n_frozen_virt {
            OrbitalSpaceType::ExternalVirtual
        } else {
            OrbitalSpaceType::FrozenVirtual
        }
    }
}

/// Paired orbital spaces for open-shell (UHF / ROHF) calculations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpinOrbitalSpace {
    pub alpha: OrbitalSpace,
    pub beta: OrbitalSpace,
}

impl SpinOrbitalSpace {
    /// Creates a spin-orbital space from alpha and beta electron counts.
    pub fn new_open_shell(n_basis: usize, n_alpha: usize, n_beta: usize) -> Self {
        Self {
            alpha: OrbitalSpace::new_closed_shell(n_basis, n_alpha),
            beta: OrbitalSpace::new_closed_shell(n_basis, n_beta),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orbital_space_closed_shell() {
        let space = OrbitalSpace::new_closed_shell(10, 5);
        assert_eq!(space.n_occ(), 5);
        assert_eq!(space.n_virt(), 5);
        assert_eq!(space.n_correlated_occ(), 5);
        assert_eq!(space.n_correlated_virt(), 5);
        assert_eq!(space.frozen_core_range(), 0..0);
        assert_eq!(space.correlated_occ_range(), 0..5);
        assert_eq!(space.correlated_virt_range(), 5..10);
    }

    #[test]
    fn test_orbital_space_frozen_core() {
        let space = OrbitalSpace::new_closed_shell(10, 5).with_frozen_core(1);
        assert_eq!(space.n_frozen_core, 1);
        assert_eq!(space.n_correlated_occ(), 4);
        assert_eq!(space.frozen_core_range(), 0..1);
        assert_eq!(space.correlated_occ_range(), 1..5);
        assert_eq!(space.orbital_type(0), OrbitalSpaceType::FrozenCore);
        assert_eq!(space.orbital_type(1), OrbitalSpaceType::ActiveOccupied);
    }

    #[test]
    fn test_orbital_space_cas() {
        // 10 basis functions, 5 occupied electrons pairs (10 electrons).
        // CAS(4, 4): 4 active electrons (2 occ pairs), 4 active orbitals (2 occ + 2 virt).
        let space = OrbitalSpace::from_active_space(10,5,5,&crate::options::ActiveSpace::new(2,2,4)).unwrap();
        assert_eq!(space.n_frozen_core, 3);
        assert_eq!(space.n_active_occ, 2);
        assert_eq!(space.n_active_virt, 2);
        assert_eq!(space.n_external_virt, 3);
        assert_eq!(space.cas_range(), 3..7);
        assert_eq!(space.n_cas_orbitals(), 4);
    }
}
