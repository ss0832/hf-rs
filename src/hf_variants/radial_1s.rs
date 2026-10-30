// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Numerical radial 1s and restricted 1s^2 atoms/ions. The uniform Dirichlet grid is a
//! second-order spatial discretization, not an assertion of an exact complete-basis limit.
//! Tridiagonal Sturm counts/inverse iteration and prefix Coulomb sums use linear memory.

#[derive(Clone, Debug)]
pub struct NumericalAtomicOptions {
    pub n_grid_points: usize,
    pub max_iterations: usize,
    pub energy_tol: f64,
    pub residual_tol: f64,
    pub density_tol: f64,
    /// Radial boundary in Bohr. None chooses 20/(Z-(N-1)/2).
    pub r_max_bohr: Option<f64>,
    /// Conservative workspace budget, including the returned radial arrays.
    pub memory_limit_bytes: usize,
}

impl Default for NumericalAtomicOptions {
    fn default() -> Self {
        Self {
            n_grid_points: 4096,
            max_iterations: 100,
            energy_tol: 1e-12,
            residual_tol: 1e-8,
            density_tol: 1e-10,
            r_max_bohr: None,
            memory_limit_bytes: 256 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NumericalHFResult {
    pub atomic_number: usize,
    pub electron_count: usize,
    pub charge: i32,
    /// Finite-grid electronic energy in Hartree.
    pub total_energy: f64,
    /// Compatibility alias for total_energy; this is not an exact infinite-grid HF limit.
    pub hfl_total_energy: f64,
    pub kinetic_energy: f64,
    pub nuclear_attraction_energy: f64,
    pub electron_repulsion_energy: f64,
    pub n_grid_points: usize,
    pub r_max_bohr: f64,
    pub grid_spacing_bohr: f64,
    pub orbital_energy_1s: f64,
    /// Reduced radial orbital P(r), normalized by sum_i P_i^2 dr = 1.
    pub radial_wavefunction: Vec<f64>,
    pub radial_grid: Vec<f64>,
    /// Norm of the final self-consistent discrete orbital residual.
    pub residual_norm: f64,
    pub density_change: f64,
    /// Finite-grid deviation from the continuum Coulomb virial relation 2T+V=0.
    pub virial_error: f64,
    pub iterations: usize,
    pub converged: bool,
}

fn norm(vector: &[f64]) -> f64 {
    vector.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn normalize(vector: &mut [f64]) {
    let scale = norm(vector);
    vector.iter_mut().for_each(|x| *x /= scale);
}

/// One orbital's spherical Coulomb potential. Each radial point is counted once.
fn coulomb(orbital: &[f64], spacing: f64) -> Vec<f64> {
    // orbital_i = sqrt(dr) P(r_i), so its squared components are quadrature charges.
    let mut outside: f64 = orbital
        .iter()
        .enumerate()
        .map(|(i, p)| p * p / ((i + 1) as f64 * spacing))
        .sum();
    let mut inside = 0.;
    orbital
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let r = (i + 1) as f64 * spacing;
            inside += p * p;
            outside -= p * p / r;
            inside / r + outside
        })
        .collect()
}

fn sturm_count(diagonal: &[f64], off_squared: f64, value: f64) -> usize {
    let mut negative = 0;
    let mut pivot = diagonal[0] - value;
    for i in 0..diagonal.len() {
        if i != 0 {
            pivot = diagonal[i] - value - off_squared / pivot;
        }
        // The negative limiting pivot defines the Sturm count at an exact zero.
        // This is factorization bookkeeping, not a modification of the Hamiltonian.
        if pivot.abs() < 1e-300 {
            pivot = -1e-300;
        }
        if pivot < 0. {
            negative += 1;
        }
    }
    negative
}

fn ground_state(diagonal: &[f64], off: f64, initial: &[f64]) -> Result<Vec<f64>, String> {
    let n = diagonal.len();
    let mut lower = diagonal
        .iter()
        .enumerate()
        .map(|(i, d)| d - off.abs() * ((i > 0) as u8 + (i + 1 < n) as u8) as f64)
        .fold(f64::INFINITY, f64::min);
    let mut upper = diagonal.iter().copied().fold(f64::INFINITY, f64::min);
    lower -= 1e-10 * (1. + lower.abs());
    for _ in 0..100 {
        let middle = 0.5 * (lower + upper);
        if middle == lower || middle == upper {
            break;
        }
        if sturm_count(diagonal, off * off, middle) == 0 {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    let eigenvalue = 0.5 * (lower + upper);
    let mut separation = 1e-7 * (1. + eigenvalue.abs());
    let mut factors = vec![0.; n];
    let mut factored = false;
    for _ in 0..16 {
        let shift = eigenvalue - separation;
        factors[0] = diagonal[0] - shift;
        let mut positive = factors[0] > 0.;
        for i in 1..n {
            if !positive {
                break;
            }
            factors[i] = diagonal[i] - shift - off * off / factors[i - 1];
            positive = factors[i] > 0. && factors[i].is_finite();
        }
        if positive {
            factored = true;
            break;
        }
        separation *= 10.;
    }
    if !factored {
        return Err("radial inverse iteration could not factor a positive shifted operator".into());
    }
    let mut orbital = initial.to_vec();
    normalize(&mut orbital);
    for _ in 0..8 {
        for i in 1..n {
            orbital[i] -= off * orbital[i - 1] / factors[i - 1];
        }
        orbital[n - 1] /= factors[n - 1];
        for i in (0..n - 1).rev() {
            orbital[i] = (orbital[i] - off * orbital[i + 1]) / factors[i];
        }
        if orbital.iter().any(|x| !x.is_finite()) || norm(&orbital) == 0. {
            return Err("nonfinite radial inverse-iteration orbital".into());
        }
        normalize(&mut orbital);
    }
    if orbital.iter().sum::<f64>() < 0. {
        orbital.iter_mut().for_each(|x| *x = -*x);
    }
    Ok(orbital)
}

fn kinetic(orbital: &[f64], spacing: f64) -> f64 {
    // Positive quadratic form avoids subtracting large kinetic matrix entries.
    let boundary = orbital[0].powi(2) + orbital[orbital.len() - 1].powi(2);
    (boundary
        + orbital
            .windows(2)
            .map(|p| (p[1] - p[0]).powi(2))
            .sum::<f64>())
        / (2. * spacing * spacing)
}

/// Solves explicitly specified 1s or restricted 1s^2 positive ions/neutral H and He.
/// Multishell neutral atoms and negative ions are rejected, never relabeled as 1s^2 ions.
pub fn run_numerical_atomic_1s_hf(
    atomic_number: usize,
    electron_count: usize,
    options: &NumericalAtomicOptions,
) -> Result<NumericalHFResult, String> {
    if atomic_number == 0
        || atomic_number > 118
        || !matches!(electron_count, 1 | 2)
        || electron_count > atomic_number
    {
        return Err("radial 1s HF supports Z=1..118 with one or two electrons and nonnegative charge; multishell atoms and negative ions are not implemented".into());
    }
    let n = options.n_grid_points;
    if n < 16
        || options.max_iterations == 0
        || [
            options.energy_tol,
            options.residual_tol,
            options.density_tol,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err("radial HF requires at least 16 points, a positive iteration budget and finite positive tolerances".into());
    }
    let bytes = n
        .checked_mul(16 * std::mem::size_of::<f64>())
        .ok_or("radial HF workspace size overflow")?;
    if bytes > options.memory_limit_bytes {
        return Err(format!(
            "radial HF workspace estimate {bytes} bytes exceeds its memory budget"
        ));
    }
    let z = atomic_number as f64;
    let r_max = options
        .r_max_bohr
        .unwrap_or(20. / (z - 0.5 * (electron_count - 1) as f64));
    if !r_max.is_finite() || r_max <= 0. {
        return Err("radial HF boundary must be finite and positive".into());
    }
    let dr = r_max / (n + 1) as f64;
    let inverse_spacing_squared = 1. / (dr * dr);
    if !inverse_spacing_squared.is_finite() {
        return Err("radial grid spacing is outside the representable range".into());
    }
    let nuclear: Vec<f64> = (1..=n).map(|i| -z / (i as f64 * dr)).collect();
    let effective_charge = z - 5. / 16. * (electron_count - 1) as f64;
    let mut orbital: Vec<f64> = (1..=n)
        .map(|i| {
            let r = i as f64 * dr;
            r * (-effective_charge * r).exp()
        })
        .collect();
    if norm(&orbital) == 0. {
        return Err("radial grid does not resolve the initial atomic orbital".into());
    }
    normalize(&mut orbital);
    let mut previous_energy = f64::INFINITY;
    let mut density_change = f64::INFINITY;
    let mut residual_norm = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;
    for iteration in 1..=options.max_iterations {
        iterations = iteration;
        let j = coulomb(&orbital, dr);
        let diagonal: Vec<_> = nuclear
            .iter()
            .zip(&j)
            .map(|(v, j)| inverse_spacing_squared + v + (electron_count - 1) as f64 * j)
            .collect();
        let next = ground_state(&diagonal, -0.5 * inverse_spacing_squared, &orbital)?;
        density_change = next
            .iter()
            .zip(&orbital)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        orbital = next;
        let j = coulomb(&orbital, dr);
        let t = kinetic(&orbital, dr);
        let v: f64 = orbital.iter().zip(&nuclear).map(|(p, v)| p * p * v).sum();
        let interaction: f64 = orbital.iter().zip(&j).map(|(p, j)| p * p * j).sum();
        let epsilon = t + v + (electron_count - 1) as f64 * interaction;
        let energy = electron_count as f64 * (t + v)
            + 0.5 * (electron_count * (electron_count - 1)) as f64 * interaction;
        let mut square = 0.;
        for i in 0..n {
            let left = if i > 0 { orbital[i - 1] } else { 0. };
            let right = if i + 1 < n { orbital[i + 1] } else { 0. };
            let residual = inverse_spacing_squared * (orbital[i] - 0.5 * (left + right))
                + (nuclear[i] + (electron_count - 1) as f64 * j[i] - epsilon) * orbital[i];
            square += residual * residual;
        }
        residual_norm = square.sqrt();
        if (energy - previous_energy).abs() <= options.energy_tol
            && density_change <= options.density_tol
            && residual_norm <= options.residual_tol
        {
            converged = true;
            break;
        }
        previous_energy = energy;
    }
    let j = coulomb(&orbital, dr);
    let t = electron_count as f64 * kinetic(&orbital, dr);
    let v = electron_count as f64
        * orbital
            .iter()
            .zip(&nuclear)
            .map(|(p, v)| p * p * v)
            .sum::<f64>();
    let interaction: f64 = orbital.iter().zip(&j).map(|(p, j)| p * p * j).sum();
    let repulsion = 0.5 * (electron_count * (electron_count - 1)) as f64 * interaction;
    let energy = t + v + repulsion;
    Ok(NumericalHFResult {
        atomic_number,
        electron_count,
        charge: (atomic_number - electron_count) as i32,
        total_energy: energy,
        hfl_total_energy: energy,
        kinetic_energy: t,
        nuclear_attraction_energy: v,
        electron_repulsion_energy: repulsion,
        n_grid_points: n,
        r_max_bohr: r_max,
        grid_spacing_bohr: dr,
        orbital_energy_1s: (t + v) / electron_count as f64
            + (electron_count - 1) as f64 * interaction,
        radial_wavefunction: orbital.into_iter().map(|p| p / dr.sqrt()).collect(),
        radial_grid: (1..=n).map(|i| i as f64 * dr).collect(),
        residual_norm,
        density_change,
        virial_error: 2. * t + v + repulsion,
        iterations,
        converged,
    })
}
