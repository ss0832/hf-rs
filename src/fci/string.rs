// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Bitstring determinant representation and lexical addressing (Knowles & Handy).

/// Represents an alpha or beta spin-string as a 64-bit integer bitmask.
/// Bit `p` is 1 if orbital `p` is occupied, 0 otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SpinString(pub u64);

impl SpinString {
    #[inline(always)]
    pub fn is_occupied(&self, orbital: usize) -> bool {
        (self.0 & (1u64 << orbital)) != 0
    }

    #[inline(always)]
    pub fn set_occupied(&mut self, orbital: usize) {
        self.0 |= 1u64 << orbital;
    }

    #[inline(always)]
    pub fn set_unoccupied(&mut self, orbital: usize) {
        self.0 &= !(1u64 << orbital);
    }

    #[inline(always)]
    pub fn n_electrons(&self) -> usize {
        self.0.count_ones() as usize
    }

    /// Computes single excitation: a_p^\dagger a_q |string>.
    /// Returns Some((new_string, phase)) if valid (q is occupied and p is unoccupied),
    /// or None if the excitation vanishes.
    #[inline(always)]
    pub fn apply_single_excitation(&self, p: usize, q: usize) -> Option<(Self, f64)> {
        if p == q {
            if self.is_occupied(p) {
                return Some((*self, 1.0));
            } else {
                return None;
            }
        }

        // q must be occupied, p must be empty
        if !self.is_occupied(q) || self.is_occupied(p) {
            return None;
        }

        let min_idx = p.min(q);
        let max_idx = p.max(q);

        // Count electrons strictly between min_idx and max_idx
        let mask = ((1u64 << max_idx) - 1) & !((1u64 << (min_idx + 1)) - 1);
        let n_intervening = (self.0 & mask).count_ones();
        let phase = if n_intervening % 2 == 1 { -1.0 } else { 1.0 };

        let new_mask = (self.0 ^ (1u64 << q)) | (1u64 << p);
        Some((SpinString(new_mask), phase))
    }

    /// Returns the indices of all occupied orbitals in ascending order.
    pub fn occupied_orbitals(&self) -> Vec<usize> {
        let mut occ = Vec::with_capacity(self.n_electrons());
        let mut mask = self.0;
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            occ.push(idx);
            mask &= mask - 1; // Clear lowest set bit
        }
        occ
    }
}

/// Computes binomial coefficient n choose k.
pub fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut c = 1;
    for i in 0..k {
        c = c * (n - i) / (i + 1);
    }
    c
}

/// Generates all spin strings with `n_electrons` in `n_orbitals` in canonical lexical order.
pub fn generate_strings(n_electrons: usize, n_orbitals: usize) -> Vec<SpinString> {
    assert!(n_electrons <= n_orbitals);
    assert!(n_orbitals <= 64, "SpinString supports up to 64 orbitals");

    let total = binomial(n_orbitals, n_electrons);
    let mut strings = Vec::with_capacity(total);

    fn recurse(
        current_idx: usize,
        n_elec_left: usize,
        n_orb_left: usize,
        current_mask: u64,
        out: &mut Vec<SpinString>,
    ) {
        if n_elec_left == 0 {
            out.push(SpinString(current_mask));
            return;
        }
        if n_elec_left > n_orb_left {
            return;
        }

        // Branch 1: occupy orbital current_idx
        recurse(
            current_idx + 1,
            n_elec_left - 1,
            n_orb_left - 1,
            current_mask | (1u64 << current_idx),
            out,
        );

        // Branch 2: leave orbital current_idx empty
        recurse(
            current_idx + 1,
            n_elec_left,
            n_orb_left - 1,
            current_mask,
            out,
        );
    }

    recurse(0, n_electrons, n_orbitals, 0, &mut strings);
    // Sort into canonical ascending order
    strings.sort_by_key(|s| s.0);
    strings
}

/// Determinant space spanned by alpha and beta strings.
#[derive(Clone, Debug)]
pub struct DeterminantSpace {
    pub n_orbitals: usize,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub alpha_strings: Vec<SpinString>,
    pub beta_strings: Vec<SpinString>,
}

impl DeterminantSpace {
    pub fn new(n_orbitals: usize, n_alpha: usize, n_beta: usize) -> Self {
        let alpha_strings = generate_strings(n_alpha, n_orbitals);
        let beta_strings = generate_strings(n_beta, n_orbitals);
        Self {
            n_orbitals,
            n_alpha,
            n_beta,
            alpha_strings,
            beta_strings,
        }
    }

    /// Total number of Slater determinants N_det = N_alpha * N_beta.
    pub fn n_determinants(&self) -> usize {
        self.alpha_strings.len() * self.beta_strings.len()
    }

    /// Returns the (alpha_idx, beta_idx) corresponding to a linear determinant index.
    #[inline(always)]
    pub fn linear_to_pair(&self, det_idx: usize) -> (usize, usize) {
        let n_b = self.beta_strings.len();
        (det_idx / n_b, det_idx % n_b)
    }

    /// Returns the linear determinant index from (alpha_idx, beta_idx).
    #[inline(always)]
    pub fn pair_to_linear(&self, a_idx: usize, b_idx: usize) -> usize {
        a_idx * self.beta_strings.len() + b_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spin_string_excitations() {
        let mut s = SpinString(0);
        s.set_occupied(0);
        s.set_occupied(2);
        assert_eq!(s.n_electrons(), 2);
        assert!(s.is_occupied(0));
        assert!(!s.is_occupied(1));
        assert!(s.is_occupied(2));

        // Excitation 0 -> 1 (intervening: none) -> phase +1
        let (s_new, phase) = s.apply_single_excitation(1, 0).expect("Valid excitation");
        assert_eq!(phase, 1.0);
        assert_eq!(s_new.occupied_orbitals(), vec![1, 2]);

        // Excitation 3 -> 0 (intervening: orbital 2) -> phase -1
        let (s_new2, phase2) = s.apply_single_excitation(3, 0).expect("Valid excitation");
        assert_eq!(phase2, -1.0);
        assert_eq!(s_new2.occupied_orbitals(), vec![2, 3]);
    }

    #[test]
    fn test_generate_strings() {
        let strings = generate_strings(2, 4);
        assert_eq!(strings.len(), 6); // 4 choose 2 = 6
        for s in strings {
            assert_eq!(s.n_electrons(), 2);
        }
    }
}
