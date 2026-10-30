// SPDX-License-Identifier: Apache-2.0
//! Determinant-space construction by excitation generation, without enumerating FCI.
use super::string::{SpinString, generate_strings};
use crate::error::HfError;
use crate::options::ResolvedActiveSpace;
use std::collections::BTreeSet;
pub type Determinant = (SpinString, SpinString);

/// Complete the spin couplings of each spatial occupation pattern. In an open
/// shell, truncation by spin-orbital excitation rank alone need not commute with
/// S²; spin completion can require additional complementary determinants.
pub fn spin_complete(
    n_mo: usize,
    n_alpha: usize,
    n_beta: usize,
    dets: &[Determinant],
    limit: usize,
) -> Result<Vec<Determinant>, HfError> {
    if n_mo > 64 || n_alpha > n_mo || n_beta > n_mo || dets.is_empty() || limit == 0 {
        return Err(HfError::InvalidInput(
            "invalid spin-completion space".into(),
        ));
    }
    let allowed = mask(n_mo);
    let mut patterns = BTreeSet::new();
    for &(a, b) in dets {
        if a.n_electrons() != n_alpha || b.n_electrons() != n_beta || (a.0 | b.0) & !allowed != 0 {
            return Err(HfError::InvalidInput(
                "invalid determinant in spin-completion space".into(),
            ));
        }
        patterns.insert((a.0 & b.0, a.0 ^ b.0));
    }
    let mut result = BTreeSet::new();
    for (doubly, singly) in patterns {
        let positions = bits(singly);
        let n_up = n_alpha - doubly.count_ones() as usize;
        let count = choose(positions.len(), n_up)
            .ok_or_else(|| HfError::InvalidInput("spin-completion count overflow".into()))?;
        if result.len().checked_add(count).is_none_or(|n| n > limit) {
            return Err(HfError::Unsupported {
                method: "spin-complete CI".into(),
                detail: "determinant count exceeds storage limit".into(),
            });
        }
        for up in generate_strings(n_up, positions.len()) {
            let alpha_single = bits(up.0)
                .iter()
                .fold(0u64, |m, &i| m | (1u64 << positions[i]));
            result.insert((
                SpinString(doubly | alpha_single),
                SpinString(doubly | (singly ^ alpha_single)),
            ));
        }
    }
    Ok(result.into_iter().collect())
}

fn mask(n: usize) -> u64 {
    if n == 64 { u64::MAX } else { (1u64 << n) - 1 }
}
fn bits(mut x: u64) -> Vec<usize> {
    let mut indices = Vec::with_capacity(x.count_ones() as usize);
    while x != 0 {
        indices.push(x.trailing_zeros() as usize);
        x &= x - 1;
    }
    indices
}
fn choose(n: usize, k: usize) -> Option<usize> {
    if k > n {
        return None;
    }
    let mut value = 1u128;
    for i in 0..k.min(n - k) {
        value = value.checked_mul((n - i) as u128)? / (i + 1) as u128;
    }
    usize::try_from(value).ok()
}

pub fn cas_references(
    space: &ResolvedActiveSpace,
    n_mo: usize,
    limit: usize,
) -> Result<Vec<Determinant>, HfError> {
    let na = space.active.len();
    let indices = space.permutation();
    if n_mo > 64
        || indices.len() != n_mo
        || indices.iter().any(|&i| i >= n_mo)
        || indices.iter().copied().collect::<BTreeSet<_>>().len() != n_mo
    {
        return Err(HfError::InvalidInput(
            "invalid CAS orbital partition".into(),
        ));
    }
    let count = choose(na, space.n_alpha)
        .and_then(|a| choose(na, space.n_beta).and_then(|b| a.checked_mul(b)));
    if count.is_none_or(|n| n > limit) {
        return Err(HfError::Unsupported {
            method: "CAS reference space".into(),
            detail: "determinant count exceeds storage limit".into(),
        });
    }
    let core = space.inactive.iter().fold(0u64, |m, &i| m | (1u64 << i));
    let embed = |s: SpinString| {
        SpinString(
            bits(s.0)
                .iter()
                .fold(core, |m, &i| m | (1u64 << space.active[i])),
        )
    };
    let alpha = generate_strings(space.n_alpha, na);
    let beta = generate_strings(space.n_beta, na);
    let mut determinants = Vec::with_capacity(
        count.ok_or_else(|| HfError::InvalidInput("CAS determinant count overflow".into()))?,
    );
    for &a in &alpha {
        for &b in &beta {
            determinants.push((embed(a), embed(b)));
        }
    }
    determinants.sort_unstable();
    Ok(determinants)
}

/// All determinants reachable by zero, one or two spin-conserving substitutions.
/// Storage is bounded by `limit`; growth is checked before each insertion.
pub fn singles_doubles(
    n_mo: usize,
    n_alpha: usize,
    n_beta: usize,
    references: &[Determinant],
    limit: usize,
) -> Result<Vec<Determinant>, HfError> {
    if n_mo > 64 || n_alpha > n_mo || n_beta > n_mo || limit == 0 || references.is_empty() {
        return Err(HfError::InvalidInput("invalid SD determinant space".into()));
    }
    let allowed = mask(n_mo);
    let mut set = BTreeSet::new();
    let mut insert = |a: u64, b: u64| -> Result<(), HfError> {
        let d = (SpinString(a), SpinString(b));
        if set.len() == limit && !set.contains(&d) {
            return Err(HfError::Unsupported {
                method: "SD determinant space".into(),
                detail: format!(
                    "more than {limit} determinants; tighten selection or reduce the orbital space"
                ),
            });
        }
        set.insert(d);
        Ok(())
    };
    for &(a, b) in references {
        if (a.0 | b.0) & !allowed != 0 || a.n_electrons() != n_alpha || b.n_electrons() != n_beta {
            return Err(HfError::InvalidInput(
                "reference determinant has invalid orbital/electron count".into(),
            ));
        }
        insert(a.0, b.0)?;
        let oa = bits(a.0);
        let ob = bits(b.0);
        let va = bits(allowed & !a.0);
        let vb = bits(allowed & !b.0);
        for &i in &oa {
            for &p in &va {
                let aa = a.0 ^ (1u64 << i) ^ (1u64 << p);
                insert(aa, b.0)?;
                for &j in &ob {
                    for &q in &vb {
                        insert(aa, b.0 ^ (1u64 << j) ^ (1u64 << q))?;
                    }
                }
            }
        }
        for &j in &ob {
            for &q in &vb {
                insert(a.0, b.0 ^ (1u64 << j) ^ (1u64 << q))?;
            }
        }
        for (spin, occupied, vacant) in [(0, &oa, &va), (1, &ob, &vb)] {
            for (ii, &i) in occupied.iter().enumerate() {
                for &j in &occupied[..ii] {
                    for (pp, &p) in vacant.iter().enumerate() {
                        for &q in &vacant[..pp] {
                            let change = (1u64 << i) ^ (1u64 << j) ^ (1u64 << p) ^ (1u64 << q);
                            if spin == 0 {
                                insert(a.0 ^ change, b.0)?;
                            } else {
                                insert(a.0, b.0 ^ change)?;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(set.into_iter().collect())
}
