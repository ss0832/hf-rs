// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Hartree-Fock Self-Consistent Field (SCF) engine supporting RHF, UHF,
//! formal charges, open-shell systems, broken-symmetry singlets, and stability analysis.

use rayon::prelude::*;
use crate::basis::{Atom, ContractedGaussian};
use crate::integrals::{contracted_kinetic, contracted_nuclear_attraction, contracted_overlap, BasisPairData, evaluate_pair_eri};
use crate::matrix::Matrix;

#[derive(Clone, Debug)]
pub struct HFResult {
    pub ao_space: Option<std::sync::Arc<crate::ao::AoSpace>>,
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_repulsion_energy: f64,
    pub orbital_energies: Vec<f64>,
    pub mo_coefficients: Matrix, // C matrix: rows = AO, cols = MO
    pub density_matrix: Matrix,  // P matrix
    pub fock_matrix: Matrix,     // F matrix
    pub overlap_matrix: Matrix,  // S matrix
    pub core_hamiltonian: Matrix,// H matrix = T + V
    pub eri: Vec<f64>,           // Compressed 8-fold symmetric ERI tensor
    pub iterations: usize,
    pub converged: bool,
    pub charge: i32,
    pub multiplicity: usize,
    pub n_occ: usize,
    pub properties: crate::properties::MolecularProperties,
}

impl HFResult {
    pub fn homo_index(&self) -> Option<usize> {
        if self.n_occ > 0 {
            Some(self.n_occ - 1)
        } else {
            None
        }
    }

    pub fn lumo_index(&self) -> Option<usize> {
        if self.n_occ < self.orbital_energies.len() {
            Some(self.n_occ)
        } else {
            None
        }
    }

    pub fn homo_energy(&self) -> Option<f64> {
        self.homo_index().map(|idx| self.orbital_energies[idx])
    }

    pub fn lumo_energy(&self) -> Option<f64> {
        self.lumo_index().map(|idx| self.orbital_energies[idx])
    }

    pub fn homo_lumo_gap(&self) -> Option<f64> {
        match (self.homo_energy(), self.lumo_energy()) {
            (Some(h), Some(l)) => Some(l - h),
            _ => None,
        }
    }

    pub fn mulliken_charges(&self) -> &[f64] {
        &self.properties.mulliken_charges
    }

    pub fn lowdin_charges(&self) -> &[f64] {
        &self.properties.lowdin_charges
    }

    pub fn dipole_moment_debye(&self) -> [f64; 3] {
        self.properties.dipole_moment_debye
    }

    pub fn dipole_magnitude_debye(&self) -> f64 {
        self.properties.dipole_magnitude_debye
    }

    pub fn quadrupole_moment_buckingham(&self) -> [[f64; 3]; 3] {
        self.properties.quadrupole_moment_buckingham
    }

    /// Returns the default closed-shell orbital space partition.
    pub fn orbital_space(&self) -> crate::orbitals::OrbitalSpace {
        crate::orbitals::OrbitalSpace::new_closed_shell(self.mo_coefficients.cols, self.n_occ)
    }
}

#[derive(Clone, Debug)]
pub struct UHFResult {
    pub ao_space: Option<std::sync::Arc<crate::ao::AoSpace>>,
    pub total_energy: f64,
    pub electronic_energy: f64,
    pub nuclear_repulsion_energy: f64,
    pub orbital_energies_alpha: Vec<f64>,
    pub orbital_energies_beta: Vec<f64>,
    pub mo_coefficients_alpha: Matrix,
    pub mo_coefficients_beta: Matrix,
    pub density_matrix_alpha: Matrix,
    pub density_matrix_beta: Matrix,
    pub fock_matrix_alpha: Matrix,
    pub fock_matrix_beta: Matrix,
    pub overlap_matrix: Matrix,
    pub core_hamiltonian: Matrix,
    pub eri: Vec<f64>,
    pub s_squared: f64,          // <S^2> expectation value
    pub iterations: usize,
    pub converged: bool,
    pub charge: i32,
    pub multiplicity: usize,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub properties: crate::properties::MolecularProperties,
}

impl UHFResult {
    pub fn homo_alpha_index(&self) -> Option<usize> {
        if self.n_alpha > 0 { Some(self.n_alpha - 1) } else { None }
    }

    pub fn lumo_alpha_index(&self) -> Option<usize> {
        if self.n_alpha < self.orbital_energies_alpha.len() { Some(self.n_alpha) } else { None }
    }

    pub fn homo_alpha_energy(&self) -> Option<f64> {
        self.homo_alpha_index().map(|idx| self.orbital_energies_alpha[idx])
    }

    pub fn lumo_alpha_energy(&self) -> Option<f64> {
        self.lumo_alpha_index().map(|idx| self.orbital_energies_alpha[idx])
    }

    pub fn gap_alpha(&self) -> Option<f64> {
        match (self.homo_alpha_energy(), self.lumo_alpha_energy()) {
            (Some(h), Some(l)) => Some(l - h),
            _ => None,
        }
    }

    pub fn homo_beta_index(&self) -> Option<usize> {
        if self.n_beta > 0 { Some(self.n_beta - 1) } else { None }
    }

    pub fn lumo_beta_index(&self) -> Option<usize> {
        if self.n_beta < self.orbital_energies_beta.len() { Some(self.n_beta) } else { None }
    }

    pub fn homo_beta_energy(&self) -> Option<f64> {
        self.homo_beta_index().map(|idx| self.orbital_energies_beta[idx])
    }

    pub fn lumo_beta_energy(&self) -> Option<f64> {
        self.lumo_beta_index().map(|idx| self.orbital_energies_beta[idx])
    }

    pub fn gap_beta(&self) -> Option<f64> {
        match (self.homo_beta_energy(), self.lumo_beta_energy()) {
            (Some(h), Some(l)) => Some(l - h),
            _ => None,
        }
    }

    pub fn mulliken_charges(&self) -> &[f64] {
        &self.properties.mulliken_charges
    }

    pub fn lowdin_charges(&self) -> &[f64] {
        &self.properties.lowdin_charges
    }

    pub fn dipole_moment_debye(&self) -> [f64; 3] {
        self.properties.dipole_moment_debye
    }

    pub fn dipole_magnitude_debye(&self) -> f64 {
        self.properties.dipole_magnitude_debye
    }

    pub fn quadrupole_moment_buckingham(&self) -> [[f64; 3]; 3] {
        self.properties.quadrupole_moment_buckingham
    }

    /// Returns the default open-shell spin-orbital space partition.
    pub fn spin_orbital_space(&self) -> crate::orbitals::SpinOrbitalSpace {
        crate::orbitals::SpinOrbitalSpace::new_open_shell(
            self.mo_coefficients_alpha.cols,
            self.n_alpha,
            self.n_beta,
        )
    }
}

/// Result of RHF wavefunction stability analysis
#[derive(Clone, Debug)]
pub struct StabilityAnalysisResult {
    pub is_stable: bool,
    pub lowest_eigenvalue: f64,
    pub lowest_eigenvector: Vec<f64>,
    pub description: String,
}

/// Computes nuclear repulsion energy: \sum_{A < B} Z_A * Z_B / R_{AB}
pub fn compute_nuclear_repulsion(atoms: &[Atom]) -> f64 {
    let mut e_nuc = 0.0;
    for i in 0..atoms.len() {
        for j in (i + 1)..atoms.len() {
            let dx = atoms[i].position[0] - atoms[j].position[0];
            let dy = atoms[i].position[1] - atoms[j].position[1];
            let dz = atoms[i].position[2] - atoms[j].position[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            e_nuc += (atoms[i].charge * atoms[j].charge) / dist;
        }
    }
    e_nuc
}

#[inline(always)]
pub fn compound_index(i: usize, j: usize) -> usize {
    if i >= j {
        i * (i + 1) / 2 + j
    } else {
        j * (j + 1) / 2 + i
    }
}

#[inline(always)]
pub fn eri_index(i: usize, j: usize, k: usize, l: usize) -> usize {
    let ij = compound_index(i, j);
    let kl = compound_index(k, l);
    compound_index(ij, kl)
}

/// Diagonal MO energies `eps_i = c_i^T F c_i` (Rayleigh quotients) for orbital-rotation SCF algorithms
/// that do not diagonalize the Fock matrix.
fn rayleigh_orbital_energies(c: &Matrix, f: &Matrix) -> Vec<f64> {
    let fc = f.dot(c);
    (0..c.cols)
        .map(|i| (0..c.rows).map(|mu| c[(mu, i)] * fc[(mu, i)]).sum())
        .collect()
}

/// Computes the 1-electron integrals: S, T, V, and H = T + V
pub fn compute_one_electron_integrals(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
) -> (Matrix, Matrix, Matrix, Matrix) {
    let n = basis.len();
    let mut s = Matrix::zeros(n, n);
    let mut t = Matrix::zeros(n, n);
    let mut v = Matrix::zeros(n, n);

    let rows=(0..n).into_par_iter().map(|i| {
        (0..=i).map(|j| [contracted_overlap(&basis[i],&basis[j]),
            contracted_kinetic(&basis[i],&basis[j]),
            contracted_nuclear_attraction(&basis[i],&basis[j],atoms)]).collect::<Vec<_>>()
    }).collect::<Vec<_>>();
    for (i,row) in rows.iter().enumerate() {for (j,&[sv,tv,vv]) in row.iter().enumerate() {
        s[(i,j)]=sv;s[(j,i)]=sv;t[(i,j)]=tv;t[(j,i)]=tv;v[(i,j)]=vv;v[(j,i)]=vv;
    }}

    // Effective core potentials act as additional one-electron potentials.
    if crate::ecp::has_ecp(atoms) {
        v = v.add(&crate::ecp::ecp_one_electron_matrix(atoms, basis));
    }
    let h = t.add(&v);
    (s, t, v, h)
}

/// Computes 8-fold symmetric 2-electron repulsion integral (ERI) tensor using Rayon
/// with Cauchy-Schwarz integral screening: |(ij|kl)| <= sqrt((ij|ij)) * sqrt((kl|kl)).
pub fn compute_eri_tensor(basis: &[ContractedGaussian]) -> Vec<f64> {
    if let Ok(eri)=crate::shell_integrals::compute_shell_eri_tensor(basis) {return eri;}
    compute_eri_tensor_cartesian_pairs(basis)
}

/// Fallback for arbitrary collections of individual Cartesian functions, including
/// differentiated functions that do not form complete molecular shells.
pub fn compute_eri_tensor_cartesian_pairs(basis: &[ContractedGaussian]) -> Vec<f64> {
    let n = basis.len();
    let n_pairs = n * (n + 1) / 2;
    let total_elements = n_pairs * (n_pairs + 1) / 2;

    let mut pairs = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in 0..=i {
            pairs.push((i, j));
        }
    }

    // Precompute BasisPairData once for all unique pairs
    let basis_pairs: Vec<BasisPairData> = (0..n_pairs)
        .into_par_iter()
        .map(|ij| {
            let (i, j) = pairs[ij];
            BasisPairData::new(&basis[i], &basis[j])
        })
        .collect();

    // Precompute Cauchy-Schwarz diagonal bounds Q_{ij} = sqrt((ij|ij))
    let schwarz_q: Vec<f64> = (0..n_pairs)
        .into_par_iter()
        .map(|ij| {
            let pair = &basis_pairs[ij];
            if pair.primitives.is_empty() {
                return 0.0;
            }
            let mut r_buf = [0.0; 8192];
            let diag_val = evaluate_pair_eri(pair, pair, &mut r_buf);
            diag_val.max(0.0).sqrt()
        })
        .collect();

    let mut eri=vec![0.0;total_elements];
    let mut tail=eri.as_mut_slice();
    let mut rows=Vec::with_capacity(n_pairs);
    for ij in 0..n_pairs {
        let (row,rest)=tail.split_at_mut(ij+1);rows.push(row);tail=rest;
    }
    rows.into_par_iter().enumerate().for_each(|(ij,row)| {
        let q=schwarz_q[ij];
        let mut workspace=[0.0;8192];
        for (kl,value) in row.iter_mut().enumerate() {
            if q*schwarz_q[kl]>1e-12 {
                *value=evaluate_pair_eri(&basis_pairs[ij],&basis_pairs[kl],&mut workspace);
            }
        }
    });
    eri
}

/// Contract every unique eightfold quartet once. Each worker owns its J/K
/// matrices; reduction has no shared writes and O(threads * n_AO^2) memory.
pub(crate) fn contract_jk(total:&Matrix,spins:&[&Matrix],eri:&[f64],n:usize)->(Matrix,Vec<Matrix>) {
    let pairs=(0..n).flat_map(|i|(0..=i).map(move|j|(i,j))).collect::<Vec<_>>();
    let zero=|| (Matrix::zeros(n,n),vec![Matrix::zeros(n,n);spins.len()]);
    (0..pairs.len()).into_par_iter().fold(zero,|(mut j,mut k),ij| {
        let (mu,nu)=pairs[ij];
        let row=&eri[ij*(ij+1)/2..ij*(ij+1)/2+ij+1];
        for (kl,&value) in row.iter().enumerate() {
            if value==0.0 {continue;}
            let (lambda,sigma)=pairs[kl];
            for pair_flip in 0..if ij==kl {1} else {2} {
                let (aa,bb,cc,dd)=if pair_flip==0 {(mu,nu,lambda,sigma)} else {(lambda,sigma,mu,nu)};
                for ab_flip in 0..if aa==bb {1} else {2} {
                    let (a,b)=if ab_flip==0 {(aa,bb)} else {(bb,aa)};
                    for cd_flip in 0..if cc==dd {1} else {2} {
                        let (c,d)=if cd_flip==0 {(cc,dd)} else {(dd,cc)};
                        j[(a,b)]+=total[(c,d)]*value;
                        for (ks,p) in k.iter_mut().zip(spins) {ks[(a,c)]+=p[(b,d)]*value;}
                    }
                }
            }
        }
        (j,k)
    }).reduce(zero,|(mut j,mut k),(other_j,other_k)| {
        for (a,b) in j.data.iter_mut().zip(other_j.data) {*a+=b;}
        for (a,b) in k.iter_mut().zip(other_k) {for (x,y) in a.data.iter_mut().zip(b.data) {*x+=y;}}
        (j,k)
    })
}

pub fn build_fock_matrix(h:&Matrix,p:&Matrix,eri:&[f64],n:usize)->Matrix {
    let (j,k)=contract_jk(p,&[p],eri,n);
    h.add(&j).sub(&k[0].scale(0.5))
}

pub fn build_uhf_fock_matrices(h:&Matrix,pa:&Matrix,pb:&Matrix,eri:&[f64],n:usize)->(Matrix,Matrix) {
    let (j,k)=contract_jk(&pa.add(pb),&[pa,pb],eri,n);
    let hj=h.add(&j);
    (hj.sub(&k[0]),hj.sub(&k[1]))
}

pub fn run_scf_general(
    atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, max_iter: usize, tol: f64,
    conv_algo: crate::diis::ConvAlgo, initial_p: Option<&Matrix>,
) -> Result<HFResult,String> {
    run_scf_in_space(atoms,basis,charge,max_iter,tol,conv_algo,initial_p,crate::ao::BasisKind::Cartesian)
}

pub fn run_scf_in_space(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    max_iter: usize,
    tol: f64,
    conv_algo: crate::diis::ConvAlgo,
    initial_p: Option<&Matrix>,
    basis_kind: crate::ao::BasisKind,
) -> Result<HFResult, String> {
    run_scf_in_space_with_convergence(atoms, basis, charge, max_iter,
        RhfConvergence {
            energy: tol,
            residual: (tol.sqrt() * 1e-2).max(1e-11),
            density: (tol.sqrt() * 0.1).clamp(1e-12, 1e-10),
        }, conv_algo, initial_p, basis_kind)
}

/// Independent convergence conditions, useful for nonstationary post-HF energies.
#[derive(Clone, Copy, Debug)]
pub struct RhfConvergence {
    pub energy: f64,
    pub residual: f64,
    pub density: f64,
}

pub fn run_scf_in_space_with_convergence(
    atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, max_iter: usize,
    convergence: RhfConvergence, conv_algo: crate::diis::ConvAlgo,
    initial_p: Option<&Matrix>, basis_kind: crate::ao::BasisKind,
) -> Result<HFResult, String> {
    let tol = convergence.energy;
    let n = basis.len();
    if max_iter == 0 || n == 0 || [tol, convergence.residual, convergence.density]
        .iter().any(|&value| !value.is_finite() || value <= 0.0) {
        return Err("SCF requires a nonempty basis, positive tolerance and iteration limit".into());
    }
    if initial_p.is_some_and(|p| p.rows != n || p.cols != n || p.data.iter().any(|v| !v.is_finite())) {
        return Err("Invalid initial density matrix".into());
    }
    let total_nuclear_charge: f64 = atoms.iter().map(|a| a.charge).sum();
    let total_electrons = (total_nuclear_charge.round() as i32) - charge;

    if total_electrons <= 0 {
        return Err(format!("Invalid electron count {} for charge {}", total_electrons, charge));
    }
    let n_electrons = total_electrons as usize;

    if !n_electrons.is_multiple_of(2) {
        return Err(format!(
            "Molecule has {} electrons (odd number). RHF requires an even number of electrons. Use UHF for open-shell systems.",
            n_electrons
        ));
    }
    let n_occ = n_electrons / 2;
    if n_occ > n {
        return Err(format!("More occupied orbitals ({}) than basis functions ({})", n_occ, n));
    }

    let e_nuc = compute_nuclear_repulsion(atoms);
    let (s, _t, _v, h) = compute_one_electron_integrals(atoms, basis);
    let mut ao_space = crate::ao::AoSpace::new(basis,basis_kind).map_err(|e| e.to_string())?;
    let x_ortho = ao_space.orthogonalizer(&s).map_err(|e| e.to_string())?;
    if n_occ > x_ortho.cols { return Err("More occupied orbitals than retained overlap rank".into()); }
    let eri = compute_eri_tensor(basis);

    // Initial density guess: User initial_p if given, else SAD default, falling back to core Hamiltonian
    let mut c;
    let mut p = if let Some(init_p) = initial_p {
        let f_init = build_fock_matrix(&h, init_p, &eri, n);
        let f_prime = x_ortho.transpose().dot(&f_init).dot(&x_ortho);
        let (_eigvals, c_prime) = f_prime.eigh();
        c = x_ortho.dot(&c_prime);
        init_p.clone()
    } else if let Ok(sad_p) = crate::sad::compute_sad_density(atoms, basis, charge, &s) {
        let f_init = build_fock_matrix(&h, &sad_p, &eri, n);
        let f_prime = x_ortho.transpose().dot(&f_init).dot(&x_ortho);
        let (_eigvals, c_prime) = f_prime.eigh();
        c = x_ortho.dot(&c_prime);
        sad_p
    } else {
        let f_prime = x_ortho.transpose().dot(&h).dot(&x_ortho);
        let (_eigvals, c_prime) = f_prime.eigh();
        c = x_ortho.dot(&c_prime);

        c.occupied_density(n_occ, 2.0)
    };

    let mut prev_energy = 0.0;
    let mut converged = false;
    let mut fock = h.clone();
    let mo_energies;
    let final_e_elec;
    let mut iter_count = 0;

    let mut cdiis = crate::diis::CdiisEngine::new(8);
    let mut adiis = crate::diis::AdiisEngine::new(8);
    let mut ediis = crate::diis::EdiisEngine::new(8);
    let mut trah = crate::diis::TrahEngine::default();

    let conv_tol_grad = convergence.residual;

    eprintln!("\n  Iter        Total Energy (Eh)             ΔE (Eh)             |Error|");
    eprintln!("  {}", "-".repeat(66));

    let mut previous_density=p.clone();
    for iteration in 1..=max_iter {
        iter_count = iteration;

        fock = build_fock_matrix(&h, &p, &eri, n);

        let h_plus_f = h.add(&fock);
        let mut e_elec = 0.0;
        for i in 0..n {
            for j in 0..n {
                e_elec += 0.5 * p[(i, j)] * h_plus_f[(i, j)];
            }
        }
        let total_e = e_elec + e_nuc;

        let delta_e = (total_e - prev_energy).abs();
        prev_energy = total_e;

        // Commutator error vector e = F * P * S - S * P * F
        let error = crate::diis::compute_diis_error(&fock, &p, &s);
        let error = if x_ortho.cols < n { x_ortho.transpose().dot(&error).dot(&x_ortho) } else { error };
        let err_norm = crate::diis::max_abs_element(&error);

        if iteration == 1 {
            eprintln!("  {:>4}   {:>24.12}   {:>17}   {:>17.8e}", iteration, total_e, "---", err_norm);
        } else {
            eprintln!("  {:>4}   {:>24.12}   {:>17.8e}   {:>17.8e}", iteration, total_e, delta_e, err_norm);
        }

        // PySCF dual convergence test
        if iteration > 1 && delta_e < tol && err_norm < conv_tol_grad
            && p.rmsd(&previous_density)<convergence.density {
            converged = true;
            break;
        }

        previous_density=p.clone();
        use crate::diis::ConvAlgo;
        match conv_algo {
            ConvAlgo::Soscf => {
                let current_energies = rayleigh_orbital_energies(&c, &fock);
                let (c_new, p_new, _g) = crate::diis::soscf_step(&c, &fock, &s, &current_energies, n_occ);
                c = c_new;
                p = p_new;
            }
            ConvAlgo::Gdm => {
                let (c_new, p_new, _g) = crate::diis::gdm_step(&c, &fock, &s, n_occ, 0.15);
                c = c_new;
                p = p_new;
            }
            ConvAlgo::Trah => {
                let current_energies = rayleigh_orbital_energies(&c, &fock);
                let (c_new, p_new, _energies_new, _g) = trah.step(&c, &fock, &s, &current_energies, n_occ, total_e);
                c = c_new;
                p = p_new;
            }
            _ => {
                // Fock-extrapolation family. Every accelerator receives matched (F_k, P_k, E_k, e_k) tuples.
                let extrapolated = match conv_algo {
                    ConvAlgo::Cdiis => {
                        cdiis.push(fock.clone(), error);
                        cdiis.extrapolate()
                    }
                    ConvAlgo::Adiis => {
                        adiis.push(fock.clone(), p.clone(), total_e);
                        adiis.extrapolate()
                    }
                    ConvAlgo::Ediis => {
                        ediis.push(fock.clone(), p.clone(), total_e);
                        ediis.extrapolate()
                    }
                    _ => {
                        // Adiissoscf / AdiisCdiis: ADIIS far from convergence, CDIIS close to it.
                        let switch = if conv_algo == ConvAlgo::AdiisCdiis { 0.08 } else { 0.05 };
                        if err_norm > switch || iteration <= 2 {
                            adiis.push(fock.clone(), p.clone(), total_e);
                            adiis.extrapolate()
                        } else {
                            cdiis.push(fock.clone(), error);
                            cdiis.extrapolate()
                        }
                    }
                };
                let f_use = extrapolated.unwrap_or_else(|| fock.clone());
                let f_prime = x_ortho.transpose().dot(&f_use).dot(&x_ortho);
                let (_evals, c_prime) = f_prime.eigh();
                c = x_ortho.dot(&c_prime);
                p = c.occupied_density(n_occ, 2.0);
            }
        }
    }

    // Final orbitals: diagonalize the *true* Fock matrix F(P) (not a DIIS-extrapolated one) so that
    // orbital energies, MO coefficients, density and Fock matrix are mutually consistent. The
    // energy-weighted density used in gradients depends on these orbital energies.
    if !converged {
        fock = build_fock_matrix(&h, &p, &eri, n);
    }
    {
        let f_prime = x_ortho.transpose().dot(&fock).dot(&x_ortho);
        let (evals, c_prime) = f_prime.eigh();
        mo_energies = evals;
        c = x_ortho.dot(&c_prime);
        p = c.occupied_density(n_occ, 2.0);
        fock = build_fock_matrix(&h, &p, &eri, n);
        let h_plus_f = h.add(&fock);
        final_e_elec = 0.5 * p.data.iter().zip(&h_plus_f.data).map(|(x, y)| x * y).sum::<f64>();
        prev_energy = final_e_elec + e_nuc;
    }

    eprintln!("  {}", "-".repeat(66));
    if converged {
        eprintln!("  SCF converged in {} iterations. Final Energy = {:.12} Hartree\n", iter_count, prev_energy);
    } else {
        eprintln!("  SCF did NOT converge within {} iterations. Final Energy = {:.12} Hartree\n", iter_count, prev_energy);
    }

    let properties = crate::properties::compute_molecular_properties(atoms, basis, &p, &s);

    Ok(HFResult {
        ao_space: Some(std::sync::Arc::new(ao_space)),
        total_energy: final_e_elec + e_nuc,
        electronic_energy: final_e_elec,
        nuclear_repulsion_energy: e_nuc,
        orbital_energies: mo_energies,
        mo_coefficients: c,
        density_matrix: p,
        fock_matrix: fock,
        overlap_matrix: s,
        core_hamiltonian: h,
        eri,
        iterations: iter_count,
        converged,
        charge,
        multiplicity: 1,
        n_occ,
        properties,
    })
}

/// Runs Restricted Hartree-Fock (RHF) SCF calculation with optional initial density matrix
pub fn run_scf_with_initial_density(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    max_iter: usize,
    tol: f64,
    initial_p: Option<&Matrix>,
) -> Result<HFResult, String> {
    run_scf_general(atoms, basis, charge, max_iter, tol, crate::diis::ConvAlgo::Adiissoscf, initial_p)
}

/// Runs Restricted Hartree-Fock (RHF) SCF calculation with optional formal charge
pub fn run_scf_with_charge(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    max_iter: usize,
    tol: f64,
) -> Result<HFResult, String> {
    run_scf_general(atoms, basis, charge, max_iter, tol, crate::diis::ConvAlgo::Adiissoscf, None)
}

/// Backward compatibility wrapper: charge = 0
pub fn run_scf(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    max_iter: usize,
    tol: f64,
) -> Result<HFResult, String> {
    run_scf_with_charge(atoms, basis, 0, max_iter, tol)
}

/// Runs Unrestricted Hartree-Fock (UHF) SCF calculation.
/// Supports open-shell molecules, radicals, formal charge, triplets, and broken-symmetry singlets.
pub fn run_uhf(
    atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, multiplicity: usize,
    is_broken_symmetry: bool, max_iter: usize, tol: f64,
) -> Result<UHFResult,String> {
    run_uhf_in_space(atoms,basis,charge,multiplicity,is_broken_symmetry,max_iter,tol,crate::ao::BasisKind::Cartesian)
}

pub fn run_uhf_in_space(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    multiplicity: usize, // 2S + 1 (1 for singlet, 2 for doublet, 3 for triplet, etc.)
    is_broken_symmetry: bool, // Break alpha/beta initial symmetry to find open-shell singlet diradicals
    max_iter: usize,
    tol: f64,
    basis_kind: crate::ao::BasisKind,
) -> Result<UHFResult, String> {
    run_uhf_in_space_with_convergence(atoms, basis, charge, multiplicity,
        is_broken_symmetry, max_iter, RhfConvergence {
            energy: tol,
            residual: (tol.sqrt() * 1e-2).max(1e-11),
            density: (tol.sqrt() * 0.1).clamp(1e-12, 1e-10),
        }, basis_kind)
}

/// UHF with independently specified energy, commutator and density tolerances.
/// The legacy entry points retain their original derived tolerances.
pub fn run_uhf_in_space_with_convergence(
    atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, multiplicity: usize,
    is_broken_symmetry: bool, max_iter: usize, convergence: RhfConvergence,
    basis_kind: crate::ao::BasisKind,
) -> Result<UHFResult, String> {
    let tol = convergence.energy;
    let n = basis.len();
    if max_iter == 0 || n == 0 || [tol, convergence.residual, convergence.density]
        .iter().any(|v| !v.is_finite() || *v <= 0.) {
        return Err("SCF requires a nonempty basis, positive tolerance and iteration limit".into());
    }
    let total_nuclear_charge: f64 = atoms.iter().map(|a| a.charge).sum();
    let total_electrons = (total_nuclear_charge.round() as i32) - charge;

    if total_electrons <= 0 {
        return Err(format!("Invalid electron count {} for charge {}", total_electrons, charge));
    }
    let n_electrons = total_electrons as usize;

    if multiplicity == 0 {
        return Err("Multiplicity must be >= 1".to_string());
    }
    if multiplicity - 1 > n_electrons { return Err("Multiplicity exceeds electron count".into()); }
    let two_s = multiplicity - 1; // 2S = N_alpha - N_beta
    if !(n_electrons + two_s).is_multiple_of(2) {
        return Err(format!(
            "Incompatible electron count ({}) and multiplicity ({})",
            n_electrons, multiplicity
        ));
    }

    let n_alpha = (n_electrons + two_s) / 2;
    let n_beta = (n_electrons - two_s) / 2;

    if n_alpha > n || n_beta > n {
        return Err(format!(
            "Number of alpha ({}) or beta ({}) electrons exceeds basis functions ({})",
            n_alpha, n_beta, n
        ));
    }

    let e_nuc = compute_nuclear_repulsion(atoms);
    let (s, _t, _v, h) = compute_one_electron_integrals(atoms, basis);
    let mut ao_space = crate::ao::AoSpace::new(basis,basis_kind).map_err(|e| e.to_string())?;
    let x_ortho = ao_space.orthogonalizer(&s).map_err(|e| e.to_string())?;
    let n_mo = x_ortho.cols;
    if n_alpha > n_mo || n_beta > n_mo { return Err("Electron count exceeds retained overlap rank".into()); }
    let eri = compute_eri_tensor(basis);

    // Initial orbitals from the Fock matrix of the SAD density (core Hamiltonian fallback);
    // broken-symmetry mixing below acts on these physically meaningful frontier orbitals.
    let guess_fock = match crate::sad::compute_sad_density(atoms, basis, charge, &s) {
        Ok(sad) => build_fock_matrix(&h, &sad, &eri, n),
        Err(_) => h.clone(),
    };
    let f_prime = x_ortho.transpose().dot(&guess_fock).dot(&x_ortho);
    let (_evals, c_prime) = f_prime.eigh();
    let c_init = x_ortho.dot(&c_prime);

    let mut c_alpha = c_init.clone();
    let mut c_beta = c_init.clone();

    // Symmetry breaking for open-shell singlet: mix HOMO and LUMO in beta orbitals
    if is_broken_symmetry && n_alpha == n_beta && n_alpha > 0 && n_alpha < n_mo {
        let homo = n_alpha - 1;
        let lumo = n_alpha;
        let theta = std::f64::consts::FRAC_PI_4; // 45 degree rotation
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        for mu in 0..n {
            let c_homo = c_init[(mu, homo)];
            let c_lumo = c_init[(mu, lumo)];
            c_alpha[(mu, homo)] = cos_t * c_homo + sin_t * c_lumo;
            c_beta[(mu, homo)]  = cos_t * c_homo - sin_t * c_lumo;
        }
    }

    let problem = UhfProblem {
        atoms, basis, charge, multiplicity, n_alpha, n_beta, two_s, e_nuc, s: &s, h: &h,
        x_ortho: &x_ortho, ao_space: std::sync::Arc::new(ao_space), max_iter, tol,
        residual_tol: convergence.residual, density_tol: convergence.density,
    };
    let mut result = uhf_iterations(&problem, eri, c_alpha, c_beta);
    // A broken-symmetry start can converge to a saddle point; follow real internal
    // instabilities (lowest eigenvector of the UHF orbital Hessian) down to a stable solution.
    if is_broken_symmetry {
        for _ in 0..8 {
            if !result.converged {
                break;
            }
            let Some((ca, cb)) = crate::uhf_stability::follow_internal_instability(&result)? else {
                break;
            };
            result = uhf_iterations(&problem, std::mem::take(&mut result.eri), ca, cb);
        }
    }
    Ok(result)
}

struct UhfProblem<'a> {
    atoms: &'a [Atom],
    basis: &'a [ContractedGaussian],
    charge: i32,
    multiplicity: usize,
    n_alpha: usize,
    n_beta: usize,
    two_s: usize,
    e_nuc: f64,
    s: &'a Matrix,
    h: &'a Matrix,
    x_ortho: &'a Matrix,
    ao_space: std::sync::Arc<crate::ao::AoSpace>,
    max_iter: usize,
    tol: f64,
    residual_tol: f64,
    density_tol: f64,
}

/// UHF SCF iterations from the given initial orbitals.
fn uhf_iterations(problem: &UhfProblem, eri: Vec<f64>, c_alpha: Matrix, c_beta: Matrix) -> UHFResult {
    let UhfProblem { atoms, basis, charge, multiplicity, n_alpha, n_beta, two_s, e_nuc, s, h, x_ortho, max_iter, tol, .. } = *problem;
    let (s, h, x_ortho) = (s.clone(), h.clone(), x_ortho.clone());
    let n = basis.len();
    let n_mo = x_ortho.cols;
    let (mut c_alpha, mut c_beta) = (c_alpha, c_beta);
    // Initial density matrices
    let mut p_alpha = c_alpha.occupied_density(n_alpha, 1.0);
    let mut p_beta = c_beta.occupied_density(n_beta, 1.0);

    let mut prev_energy = 0.0;
    let mut converged = false;
    let mut fock_alpha = h.clone();
    let mut fock_beta = h.clone();
    let mo_energies_alpha;
    let mo_energies_beta;
    let final_e_elec;
    let mut iter_count = 0;

    let mut cdiis = crate::diis::UhfCdiisEngine::new(8);

    eprintln!("\n  Iter        Total Energy (Eh)             ΔE (Eh)             |Error|");
    eprintln!("  {}", "-".repeat(66));

    let mut previous_a=p_alpha.clone();
    let mut previous_b=p_beta.clone();
    for iteration in 1..=max_iter {
        iter_count = iteration;

        let (fa, fb) = build_uhf_fock_matrices(&h, &p_alpha, &p_beta, &eri, n);
        fock_alpha = fa;
        fock_beta = fb;

        // Electronic energy: 0.5 * Tr(P_tot * H + P_alpha * F_alpha + P_beta * F_beta)
        let mut e_elec = 0.0;
        let p_tot = p_alpha.add(&p_beta);
        for i in 0..n {
            for j in 0..n {
                e_elec += 0.5 * (p_tot[(i, j)] * h[(i, j)]
                    + p_alpha[(i, j)] * fock_alpha[(i, j)]
                    + p_beta[(i, j)] * fock_beta[(i, j)]);
            }
        }
        let total_e = e_elec + e_nuc;

        let delta_e = (total_e - prev_energy).abs();
        prev_energy = total_e;

        let error_a = crate::diis::compute_diis_error(&fock_alpha, &p_alpha, &s);
        let error_b = crate::diis::compute_diis_error(&fock_beta, &p_beta, &s);
        let error_a = if n_mo < n { x_ortho.transpose().dot(&error_a).dot(&x_ortho) } else { error_a };
        let error_b = if n_mo < n { x_ortho.transpose().dot(&error_b).dot(&x_ortho) } else { error_b };
        let err_norm = crate::diis::max_abs_element(&error_a).max(crate::diis::max_abs_element(&error_b));

        if iteration == 1 {
            eprintln!("  {:>4}   {:>24.12}   {:>17}   {:>17.8e}", iteration, total_e, "---", err_norm);
        } else {
            eprintln!("  {:>4}   {:>24.12}   {:>17.8e}   {:>17.8e}", iteration, total_e, delta_e, err_norm);
        }

        if iteration > 1 && delta_e < tol && err_norm < problem.residual_tol
            && p_alpha.rmsd(&previous_a).max(p_beta.rmsd(&previous_b)) < problem.density_tol {
            converged = true;
            break;
        }

        previous_a=p_alpha.clone();previous_b=p_beta.clone();
        cdiis.push(fock_alpha.clone(), fock_beta.clone(), error_a, error_b);

        let (fa_use, fb_use) = cdiis.extrapolate().unwrap_or_else(|| (fock_alpha.clone(), fock_beta.clone()));

        // Diagonalize alpha Fock matrix
        let fa_prime = x_ortho.transpose().dot(&fa_use).dot(&x_ortho);
        let (evals_a, ca_prime) = fa_prime.eigh();
        let _ = evals_a;
        c_alpha = x_ortho.dot(&ca_prime);

        // Diagonalize beta Fock matrix
        let fb_prime = x_ortho.transpose().dot(&fb_use).dot(&x_ortho);
        let (evals_b, cb_prime) = fb_prime.eigh();
        let _ = evals_b;
        c_beta = x_ortho.dot(&cb_prime);

        // Update densities
        let p_a_new = c_alpha.occupied_density(n_alpha, 1.0);
        let p_b_new = c_beta.occupied_density(n_beta, 1.0);

        if iteration <= 2 {
            p_alpha = p_a_new.scale(0.7).add(&p_alpha.scale(0.3));
            p_beta = p_b_new.scale(0.7).add(&p_beta.scale(0.3));
        } else {
            p_alpha = p_a_new;
            p_beta = p_b_new;
        }
    }

    // Final orbitals from the true Fock matrices F_sigma(P) (see the RHF driver for the rationale).
    if !converged {
        let (fa, fb) = build_uhf_fock_matrices(&h, &p_alpha, &p_beta, &eri, n);
        fock_alpha = fa;
        fock_beta = fb;
    }
    {
        let (evals_a, ca_prime) = x_ortho.transpose().dot(&fock_alpha).dot(&x_ortho).eigh();
        mo_energies_alpha = evals_a;
        c_alpha = x_ortho.dot(&ca_prime);
        let (evals_b, cb_prime) = x_ortho.transpose().dot(&fock_beta).dot(&x_ortho).eigh();
        mo_energies_beta = evals_b;
        c_beta = x_ortho.dot(&cb_prime);
        p_alpha = c_alpha.occupied_density(n_alpha, 1.0);
        p_beta = c_beta.occupied_density(n_beta, 1.0);
        let (fa, fb) = build_uhf_fock_matrices(&h, &p_alpha, &p_beta, &eri, n);
        fock_alpha = fa;
        fock_beta = fb;
        let p_tot = p_alpha.add(&p_beta);
        let dot = |a: &Matrix, b: &Matrix| -> f64 { a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum() };
        final_e_elec = 0.5 * (dot(&p_tot, &h) + dot(&p_alpha, &fock_alpha) + dot(&p_beta, &fock_beta));
        prev_energy = final_e_elec + e_nuc;
    }

    // Compute <S^2> = S_exact + N_beta - \sum_{i \in occ^a, j \in occ^b} |<psi_i^a | psi_j^b>|^2
    // <psi_i^a | psi_j^b> = (C_alpha^T * S * C_beta)_{ij}
    let mo_overlap_ab = c_alpha.transpose().dot(&s).dot(&c_beta);
    let mut overlap_sum = 0.0;
    for i in 0..n_alpha {
        for j in 0..n_beta {
            let s_ij = mo_overlap_ab[(i, j)];
            overlap_sum += s_ij * s_ij;
        }
    }
    let s_exact = (two_s as f64 / 2.0) * (two_s as f64 / 2.0 + 1.0);
    let s_squared = s_exact + (n_beta as f64) - overlap_sum;

    eprintln!("  {}", "-".repeat(66));
    if converged {
        eprintln!("  UHF converged in {} iterations. Final Energy = {:.12} Hartree (<S^2> = {:.6})\n", iter_count, prev_energy, s_squared);
    } else {
        eprintln!("  UHF did NOT converge within {} iterations. Final Energy = {:.12} Hartree (<S^2> = {:.6})\n", iter_count, prev_energy, s_squared);
    }

    let p_tot = p_alpha.add(&p_beta);
    let properties = crate::properties::compute_molecular_properties(atoms, basis, &p_tot, &s);

    UHFResult {
        ao_space: Some(problem.ao_space.clone()),
        total_energy: final_e_elec + e_nuc,
        electronic_energy: final_e_elec,
        nuclear_repulsion_energy: e_nuc,
        orbital_energies_alpha: mo_energies_alpha,
        orbital_energies_beta: mo_energies_beta,
        mo_coefficients_alpha: c_alpha,
        mo_coefficients_beta: c_beta,
        density_matrix_alpha: p_alpha,
        density_matrix_beta: p_beta,
        fock_matrix_alpha: fock_alpha,
        fock_matrix_beta: fock_beta,
        overlap_matrix: s,
        core_hamiltonian: h,
        eri,
        s_squared,
        iterations: iter_count,
        converged,
        charge,
        multiplicity,
        n_alpha,
        n_beta,
        properties,
    }
}

/// Real orbital-rotation Hessian divided by four, in (virtual, occupied) order.
/// `triplet=false` is the internal singlet A+B; `true` breaks spin symmetry.
pub fn rhf_stability_matrix(hf: &HFResult, n_occ: usize, triplet: bool) -> Matrix {
    let n_mo = hf.mo_coefficients.cols;
    let actual_n_occ = if n_occ > 0 && n_occ <= n_mo { n_occ } else { hf.n_occ };
    let n_virt = n_mo.saturating_sub(actual_n_occ);
    let dim = n_virt * actual_n_occ;
    let mo = crate::cphf::transform_eri_ao_to_mo(&hf.mo_coefficients, &hf.eri);
    let mut matrix = Matrix::zeros(dim, dim);
    for a in 0..n_virt {
        let aa = actual_n_occ + a;
        for i in 0..actual_n_occ {
            let row = a * actual_n_occ + i;
            matrix[(row, row)] = hf.orbital_energies[aa] - hf.orbital_energies[i];
            for b in 0..n_virt {
                let bb = actual_n_occ + b;
                for j in 0..actual_n_occ {
                    matrix[(row, b * actual_n_occ + j)] +=
                        -mo.get(aa, bb, j, i) - mo.get(aa, j, bb, i)
                        + if triplet { 0.0 } else { 4.0 * mo.get(aa, i, bb, j) };
                }
            }
        }
    }
    matrix
}

fn stability_result(matrix: &Matrix, channel: &str) -> StabilityAnalysisResult {
    if matrix.rows == 0 {
        return StabilityAnalysisResult { is_stable: true, lowest_eigenvalue: 0.0,
            lowest_eigenvector: Vec::new(), description: "No orbital rotations available".into() };
    }
    let (values, vectors) = matrix.eigh();
    let lowest = values[0];
    StabilityAnalysisResult { is_stable: lowest >= -1e-6, lowest_eigenvalue: lowest,
        lowest_eigenvector: (0..matrix.rows).map(|i| vectors[(i,0)]).collect(),
        description: format!("{channel} stability: lowest A+B eigenvalue = {lowest:.10} Hartree") }
}

/// RHF -> UHF stability against real, opposite alpha/beta orbital rotations.
pub fn analyze_rhf_stability(hf: &HFResult, n_occ: usize) -> StabilityAnalysisResult {
    stability_result(&rhf_stability_matrix(hf, n_occ, true), "RHF -> UHF")
}

/// RHF internal stability against a common spatial orbital rotation (singlet A+B).
pub fn analyze_rhf_internal_stability(hf: &HFResult, n_occ: usize) -> StabilityAnalysisResult {
    stability_result(&rhf_stability_matrix(hf, n_occ, false), "RHF internal")
}

/// Applies the complete antisymmetric occupied/virtual generator as C exp(kappa).
/// Columns shared by several rotation components are transformed simultaneously.
pub fn rotate_rhf_orbitals(hf: &HFResult, direction: &[f64], angle: f64) -> (Matrix, Matrix) {
    let n_mo = hf.mo_coefficients.cols;
    let n_occ = hf.n_occ;
    assert_eq!(direction.len(), (n_mo - n_occ) * n_occ, "invalid orbital rotation dimension");
    let mut generator = Matrix::zeros(n_mo, n_mo);
    for a in n_occ..n_mo {
        for i in 0..n_occ {
            let value = angle * direction[(a - n_occ) * n_occ + i];
            generator[(a,i)] = value;
            generator[(i,a)] = -value;
        }
    }
    let coefficients = hf.mo_coefficients.dot(&generator.exp());
    let density = coefficients.occupied_density(n_occ, 2.0);
    (coefficients, density)
}

/// Runs RHF SCF and if the resulting wavefunction is unstable,
/// rotates the spatial orbitals in the direction of the stability matrix's lowest eigenvector
/// and re-runs RHF SCF, for up to `max_rotations` cycles.
/// Strictly remains within RHF (does NOT convert to open-shell).
pub fn run_rhf_with_stability_rotation(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    charge: i32,
    max_iter: usize,
    tol: f64,
    max_rotations: usize,
    rotation_angle: f64,
) -> Result<(HFResult, StabilityAnalysisResult, usize), String> {
    let mut current_hf = run_scf_with_charge(atoms, basis, charge, max_iter, tol)?;
    let mut current_stab = analyze_rhf_internal_stability(&current_hf, current_hf.n_occ);
    let mut rotations_done = 0;

    for _ in 0..max_rotations {
        if current_stab.is_stable {
            break;
        }
        rotations_done += 1;
        let (_c_rot, p_rot) = rotate_rhf_orbitals(&current_hf, &current_stab.lowest_eigenvector, rotation_angle);

        // Re-run RHF SCF with the rotated density
        let next_hf = run_scf_with_initial_density(atoms, basis, charge, max_iter, tol, Some(&p_rot))?;
        let next_stab = analyze_rhf_internal_stability(&next_hf, next_hf.n_occ);

        current_hf = next_hf;
        current_stab = next_stab;
    }

    Ok((current_hf, current_stab, rotations_done))
}
