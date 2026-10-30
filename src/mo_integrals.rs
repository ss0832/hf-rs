// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Molecular Orbital (MO) 1-electron and 2-electron integral transformation engines.
//!
//! Provides:
//! - Full 4-index AO -> MO integral transformation in O(N^5).
//! - Block-selective (OOVV) transformation for MP2 and Coupled Cluster doubles in O(N_occ^2 N^3).
//! - Active space (CAS) selective transformation in O(N_act^2 N^3).
//! - 1-electron core Hamiltonian and Fock matrix MO transformations: H_mo = C^T H_ao C.
//! - Anti-symmetrized Dirac integrals: <pq||rs> = (pr|qs) - (ps|qr).

use rayon::prelude::*;
use crate::matrix::Matrix;
use crate::orbitals::OrbitalSpace;
use crate::scf::eri_index;

/// Full 4-index transformed 2-electron repulsion integrals in MO basis.
/// Storage layout is `n_pairs x n_pairs` packed floats where `n_pairs = n_basis * (n_basis + 1) / 2`.
/// Permutation symmetries: (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq).
#[derive(Clone, Debug)]
pub struct MoEriTensor {
    pub n_basis: usize,
    pub n_pairs: usize,
    pub data: Vec<f64>,
}

impl MoEriTensor {
    /// Returns the 2-electron integral in chemist's notation: `(pq|rs)`.
    #[inline(always)]
    pub fn get(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        let pq = if p >= q { p * (p + 1) / 2 + q } else { q * (q + 1) / 2 + p };
        let rs = if r >= s { r * (r + 1) / 2 + s } else { s * (s + 1) / 2 + r };
        self.data[rs * self.n_pairs + pq]
    }

    /// Returns the anti-symmetrized 2-electron integral in Dirac notation:
    /// `<pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr)`.
    #[inline(always)]
    pub fn anti_symmetrized(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        self.get(p, r, q, s) - self.get(p, s, q, r)
    }

    /// Extracts the OOVV block `(ia|jb)` corresponding to a given `OrbitalSpace`.
    pub fn to_oovv(&self, space: &OrbitalSpace) -> OovvIntegrals {
        let occ_range = space.correlated_occ_range();
        let virt_range = space.correlated_virt_range();
        let n_occ = occ_range.len();
        let n_virt = virt_range.len();

        let mut data = vec![0.0; (n_occ * n_virt) * (n_occ * n_virt)];
        let ov = n_occ * n_virt;

        for (i_idx, i_mo) in occ_range.clone().enumerate() {
            for (a_idx, a_mo) in virt_range.clone().enumerate() {
                let ia = i_idx * n_virt + a_idx;
                for (j_idx, j_mo) in occ_range.clone().enumerate() {
                    for (b_idx, b_mo) in virt_range.clone().enumerate() {
                        let jb = j_idx * n_virt + b_idx;
                        data[ia * ov + jb] = self.get(i_mo, a_mo, j_mo, b_mo);
                    }
                }
            }
        }

        OovvIntegrals { n_occ, n_virt, data }
    }

    /// Extracts the active space CAS integrals `(tu|vw)` corresponding to `OrbitalSpace::cas_range()`.
    pub fn to_active(&self, space: &OrbitalSpace) -> ActiveMoIntegrals {
        let cas_range = space.cas_range();
        let n_act = cas_range.len();
        let n_act_pairs = n_act * (n_act + 1) / 2;
        let mut act_data = vec![0.0; n_act_pairs * n_act_pairs];

        for (t_idx, t_mo) in cas_range.clone().enumerate() {
            for (u_idx, u_mo) in cas_range.clone().enumerate().take(t_idx + 1) {
                let tu = t_idx * (t_idx + 1) / 2 + u_idx;
                for (v_idx, v_mo) in cas_range.clone().enumerate() {
                    for (w_idx, w_mo) in cas_range.clone().enumerate().take(v_idx + 1) {
                        let vw = v_idx * (v_idx + 1) / 2 + w_idx;
                        act_data[vw * n_act_pairs + tu] = self.get(t_mo, u_mo, v_mo, w_mo);
                    }
                }
            }
        }

        ActiveMoIntegrals {
            n_active: n_act,
            n_pairs: n_act_pairs,
            data: act_data,
        }
    }
}

/// Transformed 2-electron integrals in the OOVV space: `(ia|jb)`.
/// Dirac anti-symmetrized integrals are: `<ij||ab> = (ia|jb) - (ib|ja)`.
#[derive(Clone, Debug)]
pub struct OovvIntegrals {
    pub n_occ: usize,
    pub n_virt: usize,
    pub data: Vec<f64>,
}

impl OovvIntegrals {
    /// Chemist's integral: `(ia|jb)`.
    #[inline(always)]
    pub fn get(&self, i: usize, a: usize, j: usize, b: usize) -> f64 {
        let ia = i * self.n_virt + a;
        let jb = j * self.n_virt + b;
        self.data[ia * (self.n_occ * self.n_virt) + jb]
    }

    /// Anti-symmetrized Dirac integral: `<ij||ab> = <ij|ab> - <ij|ba> = (ia|jb) - (ib|ja)`.
    #[inline(always)]
    pub fn anti_symmetrized(&self, i: usize, j: usize, a: usize, b: usize) -> f64 {
        self.get(i, a, j, b) - self.get(i, b, j, a)
    }
}

/// Transformed 2-electron integrals restricted to an active orbital space: `(tu|vw)`.
#[derive(Clone, Debug)]
pub struct ActiveMoIntegrals {
    pub n_active: usize,
    pub n_pairs: usize,
    pub data: Vec<f64>,
}

impl ActiveMoIntegrals {
    /// Returns the active integral `(tu|vw)` in chemist's notation.
    #[inline(always)]
    pub fn get(&self, t: usize, u: usize, v: usize, w: usize) -> f64 {
        let tu = if t >= u { t * (t + 1) / 2 + u } else { u * (u + 1) / 2 + t };
        let vw = if v >= w { v * (v + 1) / 2 + w } else { w * (w + 1) / 2 + v };
        self.data[vw * self.n_pairs + tu]
    }

    /// Returns the anti-symmetrized Dirac active integral: `<tu||vw> = (tv|uw) - (tw|uv)`.
    #[inline(always)]
    pub fn anti_symmetrized(&self, t: usize, u: usize, v: usize, w: usize) -> f64 {
        self.get(t, v, u, w) - self.get(t, w, u, v)
    }
}

/// Transforms 1-electron AO matrix (such as H_core or Fock matrix) into MO basis:
/// `H_mo = C^T H_ao C`.
pub fn transform_1e_ao_to_mo(c: &Matrix, h_ao: &Matrix) -> Matrix {
    let hc = h_ao.dot(c);
    c.transpose().dot(&hc)
}

/// Transforms 2-electron integrals from AO basis to MO basis using 2-step quarter transformations in O(N^5)
/// accelerated with faer BLAS GEMM and zero-allocation chunk processing.
pub fn transform_eri_ao_to_mo(c: &Matrix, eri_ao: &[f64]) -> MoEriTensor {
    let n = c.rows;
    let n_mo=c.cols;
    let n_mo_pairs=n_mo*(n_mo+1)/2;
    let n_pairs = n * (n + 1) / 2;

    let mut pair_to_indices = Vec::with_capacity(n_pairs);
    for mu in 0..n {
        for nu in 0..=mu {
            pair_to_indices.push((mu, nu));
        }
    }

    let c_faer = faer::MatRef::from_row_major_slice(&c.data, n, n_mo);

    // Step 1: Half-transformation (mu nu | r s) = sum_{lambda, sigma} C_{lambda r} C_{sigma s} (mu nu | lambda sigma)
    let mut half_eri = vec![0.0; n_pairs * n_mo_pairs];

    half_eri
        .par_chunks_mut(n_mo_pairs)
        .enumerate()
        .for_each(|(ij_ao, chunk)| {
            let (mu, nu) = pair_to_indices[ij_ao];
            if n <= 64 {
                let mut a_buf = [0.0; 4096];
                let a_slice = &mut a_buf[..n * n];
                for lambda in 0..n {
                    for sigma in 0..=lambda {
                        let val = eri_ao[eri_index(mu, nu, lambda, sigma)];
                        a_slice[lambda * n + sigma] = val;
                        a_slice[sigma * n + lambda] = val;
                    }
                }
                let a_mat = faer::MatRef::from_row_major_slice(a_slice, n, n);
                let w = a_mat * c_faer;
                let b = c_faer.transpose() * &w;
                for r in 0..n_mo {
                    for s in 0..=r {
                        chunk[r * (r + 1) / 2 + s] = b[(r, s)];
                    }
                }
            } else {
                let mut a = vec![0.0; n * n];
                for lambda in 0..n {
                    for sigma in 0..=lambda {
                        let val = eri_ao[eri_index(mu, nu, lambda, sigma)];
                        a[lambda * n + sigma] = val;
                        a[sigma * n + lambda] = val;
                    }
                }
                let a_mat = faer::MatRef::from_row_major_slice(&a, n, n);
                let w = a_mat * c_faer;
                let b = c_faer.transpose() * &w;
                for r in 0..n_mo {
                    for s in 0..=r {
                        chunk[r * (r + 1) / 2 + s] = b[(r, s)];
                    }
                }
            }
        });

    // Step 2: Second half-transformation (p q | r s) = sum_{mu, nu} C_{mu p} C_{nu q} (mu nu | r s)
    let mut mo_data = vec![0.0; n_mo_pairs * n_mo_pairs];

    mo_data
        .par_chunks_mut(n_mo_pairs)
        .enumerate()
        .for_each(|(rs_mo, chunk)| {
            if n <= 64 {
                let mut d_buf = [0.0; 4096];
                let d_slice = &mut d_buf[..n * n];
                for (ij_ao, &(mu, nu)) in pair_to_indices.iter().enumerate() {
                    let val = half_eri[ij_ao * n_mo_pairs + rs_mo];
                    d_slice[mu * n + nu] = val;
                    d_slice[nu * n + mu] = val;
                }
                let d_mat = faer::MatRef::from_row_major_slice(d_slice, n, n);
                let w = d_mat * c_faer;
                let e = c_faer.transpose() * &w;
                for p in 0..n_mo {
                    for q in 0..=p {
                        chunk[p * (p + 1) / 2 + q] = e[(p, q)];
                    }
                }
            } else {
                let mut d = vec![0.0; n * n];
                for (ij_ao, &(mu, nu)) in pair_to_indices.iter().enumerate() {
                    let val = half_eri[ij_ao * n_mo_pairs + rs_mo];
                    d[mu * n + nu] = val;
                    d[nu * n + mu] = val;
                }
                let d_mat = faer::MatRef::from_row_major_slice(&d, n, n);
                let w = d_mat * c_faer;
                let e = c_faer.transpose() * &w;
                for p in 0..n_mo {
                    for q in 0..=p {
                        chunk[p * (p + 1) / 2 + q] = e[(p, q)];
                    }
                }
            }
        });

    MoEriTensor {
        n_basis: n_mo,
        n_pairs: n_mo_pairs,
        data: mo_data,
    }
}

/// Block-selective transformation directly into the OOVV space `(ia|jb)`.
/// Operates in O(N_occ^2 N^3) time and O(N_occ^2 N_virt^2) memory, accelerated with faer BLAS GEMM.
pub fn transform_oovv(c_occ: &Matrix, c_virt: &Matrix, eri_ao: &[f64]) -> OovvIntegrals {
    let n = c_occ.rows;
    let n_occ = c_occ.cols;
    let n_virt = c_virt.cols;
    let n_pairs = n * (n + 1) / 2;

    let mut pair_to_indices = Vec::with_capacity(n_pairs);
    for mu in 0..n {
        for nu in 0..=mu {
            pair_to_indices.push((mu, nu));
        }
    }

    let ov = n_occ * n_virt;
    let c_occ_mat = faer::MatRef::from_row_major_slice(&c_occ.data, n, n_occ);
    let c_virt_mat = faer::MatRef::from_row_major_slice(&c_virt.data, n, n_virt);

    // Step 1: (mu nu | j b) = sum_{lambda, sigma} C_{lambda j} C_{sigma b} (mu nu | lambda sigma)
    let mut half_eri = vec![0.0; n_pairs * ov];

    half_eri
        .par_chunks_mut(ov)
        .enumerate()
        .for_each(|(ij_ao, chunk)| {
            let (mu, nu) = pair_to_indices[ij_ao];
            if n <= 64 {
                let mut a_buf = [0.0; 4096];
                let a_slice = &mut a_buf[..n * n];
                for lambda in 0..n {
                    for sigma in 0..=lambda {
                        let val = eri_ao[eri_index(mu, nu, lambda, sigma)];
                        a_slice[lambda * n + sigma] = val;
                        a_slice[sigma * n + lambda] = val;
                    }
                }
                let a_mat = faer::MatRef::from_row_major_slice(a_slice, n, n);
                let w = a_mat * c_virt_mat;
                let y = c_occ_mat.transpose() * &w;
                for j in 0..n_occ {
                    for b in 0..n_virt {
                        chunk[j * n_virt + b] = y[(j, b)];
                    }
                }
            } else {
                let mut a = vec![0.0; n * n];
                for lambda in 0..n {
                    for sigma in 0..=lambda {
                        let val = eri_ao[eri_index(mu, nu, lambda, sigma)];
                        a[lambda * n + sigma] = val;
                        a[sigma * n + lambda] = val;
                    }
                }
                let a_mat = faer::MatRef::from_row_major_slice(&a, n, n);
                let w = a_mat * c_virt_mat;
                let y = c_occ_mat.transpose() * &w;
                for j in 0..n_occ {
                    for b in 0..n_virt {
                        chunk[j * n_virt + b] = y[(j, b)];
                    }
                }
            }
        });

    // Step 2: (i a | j b) = sum_{mu, nu} C_{mu i} C_{nu a} (mu nu | j b)
    let mut oovv_data = vec![0.0; ov * ov];

    oovv_data
        .par_chunks_mut(ov)
        .enumerate()
        .for_each(|(jb, chunk)| {
            if n <= 64 {
                let mut d_buf = [0.0; 4096];
                let d_slice = &mut d_buf[..n * n];
                for (ij_ao, &(mu, nu)) in pair_to_indices.iter().enumerate() {
                    let val = half_eri[ij_ao * ov + jb];
                    d_slice[mu * n + nu] = val;
                    d_slice[nu * n + mu] = val;
                }
                let d_mat = faer::MatRef::from_row_major_slice(d_slice, n, n);
                let w_prime = d_mat * c_virt_mat;
                let z = c_occ_mat.transpose() * &w_prime;
                for i in 0..n_occ {
                    for a in 0..n_virt {
                        chunk[i * n_virt + a] = z[(i, a)];
                    }
                }
            } else {
                let mut d = vec![0.0; n * n];
                for (ij_ao, &(mu, nu)) in pair_to_indices.iter().enumerate() {
                    let val = half_eri[ij_ao * ov + jb];
                    d[mu * n + nu] = val;
                    d[nu * n + mu] = val;
                }
                let d_mat = faer::MatRef::from_row_major_slice(&d, n, n);
                let w_prime = d_mat * c_virt_mat;
                let z = c_occ_mat.transpose() * &w_prime;
                for i in 0..n_occ {
                    for a in 0..n_virt {
                        chunk[i * n_virt + a] = z[(i, a)];
                    }
                }
            }
        });

    OovvIntegrals {
        n_occ,
        n_virt,
        data: oovv_data,
    }
}

/// Active space selective transformation: (tu|vw) for active orbitals C_act.
pub fn transform_active(c_act: &Matrix, eri_ao: &[f64]) -> ActiveMoIntegrals {
    let n = c_act.rows;
    let n_act = c_act.cols;
    let n_act_pairs = n_act * (n_act + 1) / 2;
    let n_pairs = n * (n + 1) / 2;

    let mut pair_to_indices = Vec::with_capacity(n_pairs);
    for mu in 0..n {
        for nu in 0..=mu {
            pair_to_indices.push((mu, nu));
        }
    }

    let c_act_mat = faer::MatRef::from_row_major_slice(&c_act.data, n, n_act);

    // Step 1: (mu nu | v w)
    let mut half_eri = vec![0.0; n_pairs * n_act_pairs];

    half_eri
        .par_chunks_mut(n_act_pairs)
        .enumerate()
        .for_each(|(ij_ao, chunk)| {
            let (mu, nu) = pair_to_indices[ij_ao];
            if n <= 64 {
                let mut a_buf = [0.0; 4096];
                let a_slice = &mut a_buf[..n * n];
                for lambda in 0..n {
                    for sigma in 0..=lambda {
                        let val = eri_ao[eri_index(mu, nu, lambda, sigma)];
                        a_slice[lambda * n + sigma] = val;
                        a_slice[sigma * n + lambda] = val;
                    }
                }
                let a_mat = faer::MatRef::from_row_major_slice(a_slice, n, n);
                let w = a_mat * c_act_mat;
                let b = c_act_mat.transpose() * &w;
                for v in 0..n_act {
                    for w_idx in 0..=v {
                        chunk[v * (v + 1) / 2 + w_idx] = b[(v, w_idx)];
                    }
                }
            } else {
                let mut a = vec![0.0; n * n];
                for lambda in 0..n {
                    for sigma in 0..=lambda {
                        let val = eri_ao[eri_index(mu, nu, lambda, sigma)];
                        a[lambda * n + sigma] = val;
                        a[sigma * n + lambda] = val;
                    }
                }
                let a_mat = faer::MatRef::from_row_major_slice(&a, n, n);
                let w = a_mat * c_act_mat;
                let b = c_act_mat.transpose() * &w;
                for v in 0..n_act {
                    for w_idx in 0..=v {
                        chunk[v * (v + 1) / 2 + w_idx] = b[(v, w_idx)];
                    }
                }
            }
        });

    // Step 2: (t u | v w)
    let mut act_data = vec![0.0; n_act_pairs * n_act_pairs];

    act_data
        .par_chunks_mut(n_act_pairs)
        .enumerate()
        .for_each(|(vw, chunk)| {
            if n <= 64 {
                let mut d_buf = [0.0; 4096];
                let d_slice = &mut d_buf[..n * n];
                for (ij_ao, &(mu, nu)) in pair_to_indices.iter().enumerate() {
                    let val = half_eri[ij_ao * n_act_pairs + vw];
                    d_slice[mu * n + nu] = val;
                    d_slice[nu * n + mu] = val;
                }
                let d_mat = faer::MatRef::from_row_major_slice(d_slice, n, n);
                let w_prime = d_mat * c_act_mat;
                let e = c_act_mat.transpose() * &w_prime;
                for t in 0..n_act {
                    for u in 0..=t {
                        chunk[t * (t + 1) / 2 + u] = e[(t, u)];
                    }
                }
            } else {
                let mut d = vec![0.0; n * n];
                for (ij_ao, &(mu, nu)) in pair_to_indices.iter().enumerate() {
                    let val = half_eri[ij_ao * n_act_pairs + vw];
                    d[mu * n + nu] = val;
                    d[nu * n + mu] = val;
                }
                let d_mat = faer::MatRef::from_row_major_slice(&d, n, n);
                let w_prime = d_mat * c_act_mat;
                let e = c_act_mat.transpose() * &w_prime;
                for t in 0..n_act {
                    for u in 0..=t {
                        chunk[t * (t + 1) / 2 + u] = e[(t, u)];
                    }
                }
            }
        });

    ActiveMoIntegrals {
        n_active: n_act,
        n_pairs: n_act_pairs,
        data: act_data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{Atom, build_basis};
    use crate::scf::run_scf;

    #[test]
    fn test_mo_integral_transformations() {
        let basis_map = crate::basis::parse_gaussian94(crate::basis::STO_3G_GBS);
        let atoms = vec![
            Atom::new("H", [0.0, 0.0, 0.0]),
            Atom::new("H", [0.0, 0.0, 1.4]),
        ];
        let basis = build_basis(&atoms, &basis_map);
        let hf_res = run_scf(&atoms, &basis, 50, 1e-12).expect("SCF failed");

        let c = &hf_res.mo_coefficients;
        let eri_ao = &hf_res.eri;
        let full_mo = transform_eri_ao_to_mo(c, eri_ao);
        let space = hf_res.orbital_space();

        // 1. Check OOVV extraction vs direct transform_oovv
        let oovv_extracted = full_mo.to_oovv(&space);

        let n_occ = space.n_occ();
        let n_virt = space.n_virt();
        let mut c_occ = Matrix::zeros(c.rows, n_occ);
        let mut c_virt = Matrix::zeros(c.rows, n_virt);

        for mu in 0..c.rows {
            for i in 0..n_occ {
                c_occ[(mu, i)] = c[(mu, i)];
            }
            for a in 0..n_virt {
                c_virt[(mu, a)] = c[(mu, n_occ + a)];
            }
        }

        let oovv_direct = transform_oovv(&c_occ, &c_virt, eri_ao);

        for i in 0..n_occ {
            for a in 0..n_virt {
                for j in 0..n_occ {
                    for b in 0..n_virt {
                        let v1 = oovv_extracted.get(i, a, j, b);
                        let v2 = oovv_direct.get(i, a, j, b);
                        assert!((v1 - v2).abs() < 1e-12, "OOVV mismatch at ({},{},{},{}): {} vs {}", i, a, j, b, v1, v2);

                        let asym1 = oovv_extracted.anti_symmetrized(i, j, a, b);
                        let asym2 = oovv_direct.anti_symmetrized(i, j, a, b);
                        assert!((asym1 - asym2).abs() < 1e-12);
                    }
                }
            }
        }

        // 2. Check active space transformation consistency
        let cas_space = OrbitalSpace::from_active_space(space.n_basis,space.n_occ(),space.n_occ(),&crate::options::ActiveSpace::new(1,1,2)).unwrap();
        let act_extracted = full_mo.to_active(&cas_space);
        let act_direct = transform_active(c, eri_ao);

        for t in 0..c.cols {
            for u in 0..c.cols {
                for v in 0..c.cols {
                    for w in 0..c.cols {
                        let v1 = act_extracted.get(t, u, v, w);
                        let v2 = act_direct.get(t, u, v, w);
                        assert!((v1 - v2).abs() < 1e-12, "Active mismatch at ({},{},{},{}): {} vs {}", t, u, v, w, v1, v2);
                    }
                }
            }
        }
    }
}

