// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Davidson-Liu subspace eigensolver for large symmetric linear operators.
//!
//! Widely used in Configuration Interaction (CIS, CISD, FCI) and CASSCF
//! to find the lowest eigenvalues and eigenvectors without explicit construction
//! of the full Hamiltonian matrix.

use crate::matrix::Matrix;

/// Configuration options for the Davidson-Liu eigensolver.
#[derive(Clone, Debug)]
pub struct DavidsonOptions {
    /// Number of lowest eigenvalues/eigenvectors to find.
    pub n_roots: usize,
    /// Maximum dimension of the Krylov subspace before collapsing.
    pub max_subspace: usize,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on the norm of residual vectors.
    pub tol: f64,
}

impl Default for DavidsonOptions {
    fn default() -> Self {
        Self {
            n_roots: 1,
            max_subspace: 30,
            max_iterations: 100,
            tol: 1e-8,
        }
    }
}

/// Output of the Davidson eigensolver.
#[derive(Clone, Debug)]
pub struct DavidsonResult {
    /// Converged eigenvalues (lowest to highest).
    pub eigenvalues: Vec<f64>,
    /// Converged eigenvectors corresponding to each eigenvalue.
    pub eigenvectors: Vec<Vec<f64>>,
    /// Number of matrix-vector products (iterations) performed.
    pub iterations: usize,
    /// Whether all requested roots converged within tolerance.
    pub converged: bool,
}

/// Solves for the lowest eigenvalues and eigenvectors using the Davidson-Liu algorithm.
///
/// - `dim`: Dimension of the full vector space.
/// - `diag`: Diagonal elements of the operator H_{ii}.
/// - `matvec`: Closure computing the matrix-vector product `sigma = H * v`.
/// - `options`: Configuration options.
pub fn davidson_solver<F>(
    dim: usize,
    diag: &[f64],
    matvec: F,
    options: DavidsonOptions,
) -> Result<DavidsonResult, String>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    try_davidson_solver(dim, diag, |v| Ok(matvec(v)), options).map_err(|e| e.to_string())
}

/// Checked matrix-free Davidson solver, including operator errors and output validation.
pub fn try_davidson_solver<F>(
    dim: usize,
    diag: &[f64],
    matvec: F,
    options: DavidsonOptions,
) -> Result<DavidsonResult, crate::error::HfError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, crate::error::HfError>,
{
    try_davidson_solver_projected(dim, diag, matvec, |v| Ok(v.to_vec()), options)
}

/// Davidson restricted to an invariant subspace by an orthogonal projector.
pub fn try_davidson_solver_projected<F, P>(
    dim: usize,
    diag: &[f64],
    matvec: F,
    project: P,
    options: DavidsonOptions,
) -> Result<DavidsonResult, crate::error::HfError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, crate::error::HfError>,
    P: Fn(&[f64]) -> Result<Vec<f64>, crate::error::HfError>,
{
    use crate::error::HfError;
    if dim == 0
        || diag.len() != dim
        || options.n_roots == 0
        || options.n_roots > dim
        || options.max_iterations == 0
        || !options.tol.is_finite()
        || options.tol <= 0.0
        || diag.iter().any(|v| !v.is_finite())
    {
        return Err(HfError::InvalidInput(
            "invalid Davidson dimensions, controls or diagonal".into(),
        ));
    }
    let apply = |vector: &[f64]| {
        let out = matvec(vector)?;
        if out.len() != dim || out.iter().any(|v| !v.is_finite()) {
            return Err(HfError::Response("invalid Davidson operator output".into()));
        }
        Ok(out)
    };

    let n_roots = options.n_roots;
    let max_sub = options.max_subspace.max(n_roots * 4).min(dim);

    // A pure determinant guess can lie entirely in one spin/spatial symmetry
    // sector. Its residual may converge while lower roots in another sector
    // remain undiscovered. Add a small reproducible dense component, then apply
    // the caller's exact projector, so every allowed sector has initial overlap.
    let mut sorted_diag_indices: Vec<usize> = (0..dim).collect();
    sorted_diag_indices.sort_by(|&a, &b| {
        diag[a]
            .partial_cmp(&diag[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut v_subspace: Vec<Vec<f64>> = Vec::with_capacity(max_sub);
    let mut sigma_subspace: Vec<Vec<f64>> = Vec::with_capacity(max_sub);
    let initial_count = n_roots.max((2 * n_roots + 2).min(max_sub / 2));
    let noise_scale = 1e-3 / (dim as f64).sqrt();

    for &idx in &sorted_diag_indices {
        let mut unit = (0..dim)
            .map(|i| {
                let mut bits = (i as u64)
                    .wrapping_add((v_subspace.len() as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15));
                bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
                bits ^= bits >> 31;
                noise_scale * (2.0 * ((bits >> 11) as f64 / (1u64 << 53) as f64) - 1.0)
            })
            .collect::<Vec<_>>();
        unit[idx] += 1.0;
        let mut v = project(&unit)?;
        if v.len() != dim || v.iter().any(|x| !x.is_finite()) {
            return Err(HfError::Response(
                "invalid Davidson projector output".into(),
            ));
        }
        for _ in 0..2 {
            for old in &v_subspace {
                let dot = v.iter().zip(old).map(|(a, b)| a * b).sum::<f64>();
                for (x, y) in v.iter_mut().zip(old) {
                    *x -= dot * y;
                }
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            continue;
        }
        for x in &mut v {
            *x /= norm;
        }
        sigma_subspace.push(apply(&v)?);
        v_subspace.push(v);
        if v_subspace.len() == initial_count {
            break;
        }
    }
    if v_subspace.len() < n_roots {
        return Err(HfError::InvalidInput(
            "projected Davidson space has fewer independent roots than requested".into(),
        ));
    }

    let mut final_evals = vec![0.0; n_roots];
    let mut final_evecs = vec![vec![0.0; dim]; n_roots];
    let mut converged = false;
    let mut iters = 0;

    for iter in 0..options.max_iterations {
        iters = iter + 1;
        let k = v_subspace.len();

        // 1. Build small subspace Hamiltonian: H_proj_{ij} = v_i^T * sigma_j
        let mut h_proj = Matrix::zeros(k, k);
        for i in 0..k {
            for j in 0..=i {
                let dot_ij: f64 = v_subspace[i]
                    .iter()
                    .zip(&sigma_subspace[j])
                    .map(|(a, b)| a * b)
                    .sum();
                h_proj[(i, j)] = dot_ij;
                h_proj[(j, i)] = dot_ij;
            }
        }

        // 2. Diagonalize subspace Hamiltonian
        let (sub_evals, sub_evecs) = h_proj.eigh();

        // 3. For each requested root, form Ritz vector and residual vector
        let mut all_roots_converged = true;
        let mut new_directions = Vec::new();

        for root in 0..n_roots {
            let theta = sub_evals[root];
            final_evals[root] = theta;

            // Ritz vector: u = sum_j c_j v_j
            let mut u = vec![0.0; dim];
            // sigma_u = sum_j c_j sigma_j
            let mut sigma_u = vec![0.0; dim];

            for j in 0..k {
                let c_j = sub_evecs[(j, root)];
                for d in 0..dim {
                    u[d] += c_j * v_subspace[j][d];
                    sigma_u[d] += c_j * sigma_subspace[j][d];
                }
            }
            final_evecs[root] = u.clone();

            // Residual: r = sigma_u - theta * u
            let mut r = vec![0.0; dim];
            let mut res_norm_sq = 0.0;
            for d in 0..dim {
                let val = sigma_u[d] - theta * u[d];
                r[d] = val;
                res_norm_sq += val * val;
            }
            let res_norm = res_norm_sq.sqrt();

            if res_norm > options.tol {
                all_roots_converged = false;

                // Preconditioner: q_d = r_d / (theta - H_{dd})
                let mut q = vec![0.0; dim];
                for d in 0..dim {
                    let mut denom = theta - diag[d];
                    if denom.abs() < 1e-12 {
                        denom = if denom >= 0.0 { 1e-12 } else { -1e-12 };
                    }
                    q[d] = r[d] / denom;
                }
                let q = project(&q)?;
                if q.len() != dim || q.iter().any(|x| !x.is_finite()) {
                    return Err(HfError::Response(
                        "invalid Davidson projector output".into(),
                    ));
                }
                new_directions.push((q, r));
            }
        }

        if all_roots_converged {
            converged = true;
            break;
        }

        // Restart only after diagonalizing the complete current subspace. Restarting
        // immediately after expansion discarded the new directions before using them
        // and could cycle forever when max_subspace equaled the full dimension.
        if v_subspace.len() >= max_sub {
            v_subspace = final_evecs.clone();
            sigma_subspace = v_subspace
                .iter()
                .map(|u| apply(u))
                .collect::<Result<Vec<_>, _>>()?;
        }

        // Subspace expansion: normalize before testing linear independence, so
        // a small but unconverged residual is not discarded by an absolute cutoff.
        for (preconditioned, residual) in new_directions {
            if v_subspace.len() == max_sub {
                break;
            }
            for attempt in 0..2 {
                // For an exactly diagonal block, the diagonal preconditioner
                // can return -u, entirely inside the current space. The raw
                // residual still supplies a valid independent Krylov direction.
                let mut q = if attempt == 0 {
                    preconditioned.clone()
                } else {
                    project(&residual)?
                };
                if q.len() != dim || q.iter().any(|x| !x.is_finite()) {
                    return Err(HfError::Response("invalid Davidson correction".into()));
                }
                let initial_norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
                if initial_norm == 0.0 || !initial_norm.is_finite() {
                    continue;
                }
                for x in &mut q {
                    *x /= initial_norm;
                }
                for _ in 0..2 {
                    for v in &v_subspace {
                        let proj: f64 = q.iter().zip(v).map(|(a, b)| a * b).sum();
                        for d in 0..dim {
                            q[d] -= proj * v[d];
                        }
                    }
                }
                let norm = q.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    for x in &mut q {
                        *x /= norm;
                    }
                    sigma_subspace.push(apply(&q)?);
                    v_subspace.push(q);
                    break;
                }
            }
        }
    }

    Ok(DavidsonResult {
        eigenvalues: final_evals,
        eigenvectors: final_evecs,
        iterations: iters,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_davidson_diagonal_matrix() {
        let dim = 20;
        let diag: Vec<f64> = (0..dim).map(|i| (i + 1) as f64 * 1.5).collect();
        let diag_clone = diag.clone();

        let matvec = move |v: &[f64]| -> Vec<f64> {
            v.iter().zip(&diag_clone).map(|(x, d)| x * d).collect()
        };

        let options = DavidsonOptions {
            n_roots: 3,
            max_subspace: 10,
            max_iterations: 50,
            tol: 1e-9,
        };

        let res = davidson_solver(dim, &diag, matvec, options).expect("Davidson failed");
        assert!(res.converged);
        assert_eq!(res.eigenvalues.len(), 3);
        assert!((res.eigenvalues[0] - 1.5).abs() < 1e-8);
        assert!((res.eigenvalues[1] - 3.0).abs() < 1e-8);
        assert!((res.eigenvalues[2] - 4.5).abs() < 1e-8);
    }
}
