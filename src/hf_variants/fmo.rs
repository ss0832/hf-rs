// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Closed-shell FMO with self-consistent monomer electrostatic embedding.
//!
//! E'_X excludes Tr(D_X V_X). Pair corrections are
//! E'_IJ-E'_I-E'_J+Tr[(D_IJ-D_I-D_J)V_IJ]; triple corrections are
//! E'_IJK-sum_I E'_I-sum_pairs Delta E_IJ+Tr[(D_IJK-sum_I D_I)V_IJK].
//! Fields use converged monomer densities, without interfragment exchange.
//! Clusters use their own AOs: no bond-detachment projector or counterpoise.
//! Order 1 is the internal monomer-energy truncation.
//!
//! Pair decomposition uses a normalized antisymmetrized frozen determinant,
//! followed by RHF orbital relaxation (CT + MIX, not directed CT).
//! No distance switching changes the energy. Cluster ERIs are released after
//! use; a bounded optional monomer Coulomb cache otherwise streams.

use super::stationary_rhf;
use crate::integrals::contracted_eri;
use crate::scf::{
    RhfConvergence, build_fock_matrix, compute_eri_tensor, compute_nuclear_repulsion,
    compute_one_electron_integrals,
};
use crate::{Atom, ContractedGaussian, Matrix};
use rayon::prelude::*;

pub type DimerResultItem = ((usize, usize), FMOClusterResult);
pub type DimerEnergyItem = ((usize, usize), f64);
pub type TrimerEnergyItem = ((usize, usize, usize), f64);
pub type TrimerResultItem = ((usize, usize, usize), FMOClusterResult);

#[derive(Clone, Debug)]
pub struct FMOFragment {
    pub name: String,
    pub atom_indices: Vec<usize>,
    pub charge: i32,
    /// Only closed-shell singlets are implemented.
    pub multiplicity: usize,
}
impl FMOFragment {
    pub fn new(name: &str, atom_indices: Vec<usize>, charge: i32) -> Self {
        Self {
            name: name.into(),
            atom_indices,
            charge,
            multiplicity: 1,
        }
    }
}
#[derive(Clone, Debug)]
pub struct FMOOptions {
    pub order: usize,
    pub electrostatic_embedding: bool,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub residual_tolerance: f64,
    pub density_tolerance: f64,
    pub max_embedding_iterations: usize,
    pub embedding_density_tolerance: f64,
    /// Conservative integral/matrix estimate, not a process-wide RSS cap.
    pub workspace_limit_bytes: usize,
    /// Zero forces streamed monomer Coulomb contractions.
    pub esp_cache_limit_bytes: usize,
}
impl Default for FMOOptions {
    fn default() -> Self {
        Self {
            order: 2,
            electrostatic_embedding: true,
            max_iterations: 300,
            tolerance: 1e-11,
            residual_tolerance: 1e-9,
            density_tolerance: 1e-10,
            max_embedding_iterations: 100,
            embedding_density_tolerance: 1e-9,
            workspace_limit_bytes: 256 * 1024 * 1024,
            esp_cache_limit_bytes: 64 * 1024 * 1024,
        }
    }
}
/// No AO ERIs are retained in public cluster results.
#[derive(Clone, Debug)]
pub struct FMOClusterResult {
    pub atom_indices: Vec<usize>,
    pub ao_indices: Vec<usize>,
    pub internal_energy: f64,
    pub embedded_energy: f64,
    pub nuclear_energy: f64,
    pub density_matrix: Matrix,
    pub mo_coefficients: Matrix,
    pub orbital_energies: Vec<f64>,
    pub fock_matrix: Matrix,
    pub overlap_matrix: Matrix,
    pub embedding_potential: Matrix,
    pub n_occ: usize,
    pub residual_norm: f64,
    pub iterations: usize,
}
#[derive(Clone, Debug)]
pub struct FMOPairInteraction {
    pub frag_i: usize,
    pub frag_j: usize,
    pub total_interaction_energy: f64,
    /// Frozen charge interaction; includes external-atom ECP operators if present.
    pub electrostatic: f64,
    /// Antisymmetrized frozen determinant minus the unantisymmetrized product.
    pub exchange_repulsion: f64,
    /// RHF relaxation (CT + MIX), not directed charge transfer.
    pub charge_transfer: f64,
    pub embedding_correction: f64,
}
#[derive(Clone, Debug)]
pub struct FMOResult {
    pub total_energy: f64,
    pub monomer_energies: Vec<f64>,
    /// Internal cluster energies, without the embedding expectation.
    pub dimer_energies: Vec<DimerEnergyItem>,
    pub ifie: Vec<FMOPairInteraction>,
    /// Connected trimer corrections, including embedding density transfer.
    pub trimer_energies: Option<Vec<TrimerEnergyItem>>,
    pub monomer_results: Vec<FMOClusterResult>,
    pub dimer_results: Vec<DimerResultItem>,
    pub trimer_results: Option<Vec<TrimerResultItem>>,
    pub order: usize,
    pub electrostatic_embedding: bool,
    pub embedding_iterations: usize,
    pub embedding_density_change: f64,
    pub esp_cache_bytes: usize,
    pub estimated_workspace_bytes: usize,
    pub converged: bool,
}
/// Consecutive groups only; callers must assign appropriate fragment charges and spins.
pub fn auto_partition_fragments(atoms: &[Atom], fragment_size: usize) -> Vec<FMOFragment> {
    (0..atoms.len())
        .step_by(fragment_size.max(1))
        .enumerate()
        .map(|(i, start)| {
            FMOFragment::new(
                &format!("Frag_{}", i + 1),
                (start..start.saturating_add(fragment_size.max(1)).min(atoms.len())).collect(),
                0,
            )
        })
        .collect()
}
fn trace(a: &Matrix, b: &Matrix) -> f64 {
    a.data.iter().zip(&b.data).map(|(a, b)| a * b).sum()
}
fn norm(m: &Matrix) -> f64 {
    m.data.iter().map(|v| v * v).sum::<f64>().sqrt()
}
fn max_diff(a: &Matrix, b: &Matrix) -> f64 {
    a.data
        .iter()
        .zip(&b.data)
        .fold(0f64, |m, (a, b)| m.max((a - b).abs()))
}
fn submatrix(m: &Matrix, indices: &[usize]) -> Matrix {
    Matrix::from_vec(
        indices.len(),
        indices.len(),
        indices
            .iter()
            .flat_map(|&i| indices.iter().map(move |&j| m[(i, j)]))
            .collect(),
    )
}
fn pairs(n: usize) -> Vec<(usize, usize)> {
    (0..n).flat_map(|i| (0..=i).map(move |j| (i, j))).collect()
}
fn packed_len(n: usize) -> Option<usize> {
    let p = n.checked_mul(n.checked_add(1)?)?.checked_div(2)?;
    p.checked_mul(p.checked_add(1)?)?.checked_div(2)
}
struct Cluster {
    atom_indices: Vec<usize>,
    ao_indices: Vec<usize>,
    basis: Vec<ContractedGaussian>,
    h: Matrix,
    s: Matrix,
    nuclear_field: Matrix,
    nuclear: f64,
    eri: Vec<f64>,
    n_occ: usize,
}
impl Cluster {
    fn new(
        atoms: &[Atom],
        basis: &[ContractedGaussian],
        fragments: &[FMOFragment],
        members: &[usize],
        global_h: &Matrix,
    ) -> Self {
        let mut atom_indices: Vec<usize> = members
            .iter()
            .flat_map(|&i| fragments[i].atom_indices.iter().copied())
            .collect();
        atom_indices.sort_unstable();
        let sub_atoms: Vec<_> = atom_indices.iter().map(|&i| atoms[i].clone()).collect();
        let ao_indices: Vec<_> = basis
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                atom_indices
                    .binary_search(&b.atom_index)
                    .is_ok()
                    .then_some(i)
            })
            .collect();
        let sub_basis: Vec<_> = ao_indices
            .iter()
            .map(|&i| {
                let mut b = basis[i].clone();
                b.atom_index = atom_indices.binary_search(&b.atom_index).unwrap();
                b
            })
            .collect();
        let charge: i64 = members
            .iter()
            .map(|&i| i64::from(fragments[i].charge))
            .sum();
        let ne = sub_atoms.iter().map(|a| a.charge).sum::<f64>().round() as i64 - charge;
        let (s, _, _, h) = compute_one_electron_integrals(&sub_atoms, &sub_basis);
        let nuclear_field = submatrix(global_h, &ao_indices).sub(&h);
        let eri = compute_eri_tensor(&sub_basis);
        Self {
            atom_indices,
            ao_indices,
            basis: sub_basis,
            h,
            s,
            nuclear_field,
            nuclear: compute_nuclear_repulsion(&sub_atoms),
            eri,
            n_occ: ne as usize / 2,
        }
    }
    fn solve(&self, field: Matrix, o: &FMOOptions) -> Result<FMOClusterResult, String> {
        let p = stationary_rhf::solve(
            &self.h.add(&field),
            &self.s,
            &self.eri,
            self.nuclear,
            self.n_occ,
            o.max_iterations,
            RhfConvergence {
                energy: o.tolerance,
                residual: o.residual_tolerance,
                density: o.density_tolerance,
            },
        )?;
        let mut r = FMOClusterResult {
            atom_indices: self.atom_indices.clone(),
            ao_indices: self.ao_indices.clone(),
            internal_energy: 0.,
            embedded_energy: 0.,
            nuclear_energy: self.nuclear,
            density_matrix: p.density,
            mo_coefficients: p.coefficients,
            orbital_energies: p.orbital_energies,
            fock_matrix: p.fock,
            overlap_matrix: self.s.clone(),
            embedding_potential: field,
            n_occ: self.n_occ,
            residual_norm: p.residual_norm,
            iterations: p.iterations,
        };
        self.refresh(&mut r);
        Ok(r)
    }
    fn internal(&self, p: &Matrix) -> f64 {
        self.nuclear
            + 0.5
                * trace(
                    p,
                    &self
                        .h
                        .add(&build_fock_matrix(&self.h, p, &self.eri, self.basis.len())),
                )
    }
    fn refresh(&self, r: &mut FMOClusterResult) {
        r.internal_energy = self.internal(&r.density_matrix);
        r.embedded_energy = r.internal_energy + trace(&r.density_matrix, &r.embedding_potential);
        r.fock_matrix = build_fock_matrix(
            &self.h.add(&r.embedding_potential),
            &r.density_matrix,
            &self.eri,
            self.basis.len(),
        );
        let residual = r
            .mo_coefficients
            .transpose()
            .dot(
                &r.fock_matrix
                    .dot(&r.density_matrix)
                    .dot(&self.s)
                    .sub(&self.s.dot(&r.density_matrix).dot(&r.fock_matrix)),
            )
            .dot(&r.mo_coefficients);
        r.residual_norm = norm(&residual);
        // Canonicalize within the occupied and virtual spaces of the final field,
        // preserving the density while reporting consistent orbital energies.
        let fm = r
            .mo_coefficients
            .transpose()
            .dot(&r.fock_matrix)
            .dot(&r.mo_coefficients);
        let rank = r.mo_coefficients.cols;
        let mut rotation = Matrix::identity(rank);
        for (start, end) in [(0, r.n_occ), (r.n_occ, rank)] {
            if start == end {
                continue;
            }
            let indices: Vec<_> = (start..end).collect();
            let (values, vectors) = submatrix(&fm, &indices).eigh();
            for i in start..end {
                r.orbital_energies[i] = values[i - start];
                for j in start..end {
                    rotation[(i, j)] = vectors[(i - start, j - start)];
                }
            }
        }
        r.mo_coefficients = r.mo_coefficients.dot(&rotation);
    }
}
struct Coupling {
    i: usize,
    j: usize,
    values: Vec<f64>,
}
fn coupling(left: &[ContractedGaussian], right: &[ContractedGaussian]) -> Vec<f64> {
    let lp = pairs(left.len());
    let rp = pairs(right.len());
    lp.par_iter()
        .flat_map_iter(|&(a, b)| {
            rp.iter()
                .map(move |&(c, d)| contracted_eri(&left[a], &left[b], &right[c], &right[d]))
        })
        .collect()
}
fn coulomb(
    target: &[ContractedGaussian],
    source: &[ContractedGaussian],
    density: &Matrix,
) -> Matrix {
    let tp = pairs(target.len());
    let sp = pairs(source.len());
    let values: Vec<f64> = tp
        .par_iter()
        .map(|&(a, b)| {
            sp.iter()
                .map(|&(c, d)| {
                    density[(c, d)]
                        * (if c == d { 1. } else { 2. })
                        * contracted_eri(&target[a], &target[b], &source[c], &source[d])
                })
                .sum()
        })
        .collect();
    let mut j = Matrix::zeros(target.len(), target.len());
    for (&(a, b), v) in tp.iter().zip(values) {
        j[(a, b)] = v;
        j[(b, a)] = v;
    }
    j
}
fn add_cached(field: &mut Matrix, density: &Matrix, block: &Coupling, transpose: bool) {
    let tp = pairs(field.rows);
    let sp = pairs(density.rows);
    for (a, &(i, j)) in tp.iter().enumerate() {
        let value: f64 = sp
            .iter()
            .enumerate()
            .map(|(b, &(k, l))| {
                let index = if transpose {
                    b * tp.len() + a
                } else {
                    a * sp.len() + b
                };
                block.values[index] * density[(k, l)] * if k == l { 1. } else { 2. }
            })
            .sum();
        field[(i, j)] += value;
        if i != j {
            field[(j, i)] += value;
        }
    }
}
fn monomer_field(
    i: usize,
    clusters: &[Cluster],
    states: &[FMOClusterResult],
    cache: &[Coupling],
) -> Matrix {
    let mut field = clusters[i].nuclear_field.clone();
    for j in 0..clusters.len() {
        if i == j {
            continue;
        }
        if let Some(block) = cache
            .iter()
            .find(|b| (b.i == i && b.j == j) || (b.i == j && b.j == i))
        {
            add_cached(&mut field, &states[j].density_matrix, block, block.j == i);
        } else {
            field = field.add(&coulomb(
                &clusters[i].basis,
                &clusters[j].basis,
                &states[j].density_matrix,
            ));
        }
    }
    field
}
fn cluster_field(
    cluster: &Cluster,
    members: &[usize],
    monomers: &[Cluster],
    states: &[FMOClusterResult],
) -> Matrix {
    let mut field = cluster.nuclear_field.clone();
    for i in 0..monomers.len() {
        if !members.contains(&i) {
            field = field.add(&coulomb(
                &cluster.basis,
                &monomers[i].basis,
                &states[i].density_matrix,
            ));
        }
    }
    field
}
fn monomer_sum(
    cluster: &Cluster,
    members: &[usize],
    monomers: &[FMOClusterResult],
) -> (Matrix, Matrix) {
    let n = cluster.basis.len();
    let no: usize = members.iter().map(|&i| monomers[i].n_occ).sum();
    let mut density = Matrix::zeros(n, n);
    let mut c = Matrix::zeros(n, no);
    let mut offset = 0;
    for &i in members {
        let r = &monomers[i];
        let indices: Vec<_> = r
            .ao_indices
            .iter()
            .map(|ao| cluster.ao_indices.binary_search(ao).unwrap())
            .collect();
        for (a, &mu) in indices.iter().enumerate() {
            for (b, &nu) in indices.iter().enumerate() {
                density[(mu, nu)] += r.density_matrix[(a, b)];
            }
            for k in 0..r.n_occ {
                c[(mu, offset + k)] = r.mo_coefficients[(a, k)];
            }
        }
        offset += r.n_occ;
    }
    (density, c)
}
fn decompose(
    c: &Cluster,
    state: &FMOClusterResult,
    i: usize,
    j: usize,
    monomers: &[Cluster],
    states: &[FMOClusterResult],
) -> Result<FMOPairInteraction, String> {
    let (product, occupied) = monomer_sum(c, &[i, j], states);
    let overlap = occupied.transpose().dot(&c.s).dot(&occupied);
    let (values, _) = overlap.eigh();
    if values.iter().any(|v| !v.is_finite() || *v < 1e-8) {
        return Err("FMO frozen occupied orbitals are linearly dependent; exchange decomposition is undefined".into());
    }
    let frozen_c = occupied.dot(&overlap.inv_sqrt());
    let frozen = frozen_c.occupied_density(occupied.cols, 2.);
    let hi = submatrix(
        &c.h,
        &states[i]
            .ao_indices
            .iter()
            .map(|ao| c.ao_indices.binary_search(ao).unwrap())
            .collect::<Vec<_>>(),
    )
    .sub(&monomers[i].h);
    let hj = submatrix(
        &c.h,
        &states[j]
            .ao_indices
            .iter()
            .map(|ao| c.ao_indices.binary_search(ao).unwrap())
            .collect::<Vec<_>>(),
    )
    .sub(&monomers[j].h);
    let es = c.nuclear - monomers[i].nuclear - monomers[j].nuclear
        + trace(&states[i].density_matrix, &hi)
        + trace(&states[j].density_matrix, &hj)
        + trace(
            &states[i].density_matrix,
            &coulomb(
                &monomers[i].basis,
                &monomers[j].basis,
                &states[j].density_matrix,
            ),
        );
    let sum = states[i].internal_energy + states[j].internal_energy;
    let embedding_correction = trace(
        &state.density_matrix.sub(&product),
        &state.embedding_potential,
    );
    let delta = state.internal_energy - sum + embedding_correction;
    let exchange =
        c.internal(&frozen) - sum - es + trace(&frozen.sub(&product), &state.embedding_potential);
    Ok(FMOPairInteraction {
        frag_i: i,
        frag_j: j,
        total_interaction_energy: delta,
        electrostatic: es,
        exchange_repulsion: exchange,
        charge_transfer: delta - es - exchange,
        embedding_correction,
    })
}
fn validate(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    fragments: &[FMOFragment],
    o: &FMOOptions,
) -> Result<usize, String> {
    if atoms.is_empty()
        || basis.is_empty()
        || fragments.is_empty()
        || !(1..=3).contains(&o.order)
        || o.max_iterations == 0
        || o.max_embedding_iterations == 0
        || [
            o.tolerance,
            o.residual_tolerance,
            o.density_tolerance,
            o.embedding_density_tolerance,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err("FMO requires atoms, AOs, fragments, order 1..3 and positive finite convergence controls".into());
    }
    if atoms
        .iter()
        .any(|a| !a.charge.is_finite() || a.position.iter().any(|v| !v.is_finite()))
        || basis.iter().any(|b| b.atom_index >= atoms.len())
    {
        return Err("invalid FMO geometry or AO atom index".into());
    }
    for i in 0..atoms.len() {
        for j in 0..i {
            let r2: f64 = (0..3)
                .map(|k| (atoms[i].position[k] - atoms[j].position[k]).powi(2))
                .sum();
            if r2 < 1e-20 {
                return Err("FMO nuclei must have distinct coordinates".into());
            }
        }
    }
    let mut owner = vec![false; atoms.len()];
    let mut sizes = Vec::new();
    for frag in fragments {
        if frag.atom_indices.is_empty() || frag.multiplicity != 1 {
            return Err("FMO fragments must be nonempty closed-shell singlets".into());
        }
        let mut z = 0.;
        for &a in &frag.atom_indices {
            if a >= atoms.len() || owner[a] {
                return Err("FMO fragments must cover each atom exactly once".into());
            }
            owner[a] = true;
            z += atoms[a].charge;
        }
        let n = basis
            .iter()
            .filter(|b| frag.atom_indices.contains(&b.atom_index))
            .count();
        let ne = z.round() as i64 - i64::from(frag.charge);
        if (z - z.round()).abs() > 1e-10 || ne <= 0 || ne % 2 != 0 || ne as usize > 2 * n {
            return Err(
                "FMO requires a positive even fragment electron count within the AO capacity"
                    .into(),
            );
        }
        sizes.push(n);
    }
    if owner.contains(&false) {
        return Err("FMO fragments must cover each atom exactly once".into());
    }
    let estimate = || -> Option<usize> {
        let mut retained = basis.len().checked_mul(basis.len())?.checked_mul(32)?;
        let mut largest = 0;
        let mut add = |n: usize| -> Option<()> {
            let square = n.checked_mul(n)?;
            retained = retained.checked_add(square.checked_mul(8)?)?;
            largest = largest.max(
                packed_len(n)?
                    .checked_mul(2)?
                    .checked_add(square.checked_mul(32)?)?,
            );
            Some(())
        };
        for (i, &ni) in sizes.iter().enumerate() {
            add(ni)?;
            if o.order > 1 {
                for (j, &nj) in sizes.iter().enumerate().skip(i + 1) {
                    add(ni.checked_add(nj)?)?;
                    if o.order > 2 {
                        for &nk in sizes.iter().skip(j + 1) {
                            add(ni.checked_add(nj)?.checked_add(nk)?)?;
                        }
                    }
                }
            }
        }
        let monomer_eri = sizes
            .iter()
            .try_fold(0usize, |sum, &n| sum.checked_add(packed_len(n)?))?;
        retained
            .checked_add(largest)?
            .checked_add(monomer_eri)?
            .checked_mul(8)
    }()
    .ok_or("FMO workspace size overflow")?;
    if estimate > o.workspace_limit_bytes {
        return Err(format!(
            "FMO workspace estimate {estimate} exceeds budget {} bytes",
            o.workspace_limit_bytes
        ));
    }
    Ok(estimate)
}
pub fn run_fmo(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    fragments: &[FMOFragment],
    o: &FMOOptions,
) -> Result<FMOResult, String> {
    let estimate = validate(atoms, basis, fragments, o)?;
    let (_, _, _, global_h) = compute_one_electron_integrals(atoms, basis);
    let monomers: Vec<_> = (0..fragments.len())
        .map(|i| Cluster::new(atoms, basis, fragments, &[i], &global_h))
        .collect();
    let mut states: Vec<_> = monomers
        .iter()
        .map(|m| m.solve(Matrix::zeros(m.basis.len(), m.basis.len()), o))
        .collect::<Result<_, _>>()?;
    let mut cache = Vec::new();
    let mut cache_bytes = 0;
    if o.electrostatic_embedding {
        let budget = o
            .esp_cache_limit_bytes
            .min((o.workspace_limit_bytes - estimate) / 2);
        for i in 0..monomers.len() {
            for j in i + 1..monomers.len() {
                let len = pairs(monomers[i].basis.len())
                    .len()
                    .checked_mul(pairs(monomers[j].basis.len()).len())
                    .and_then(|v| v.checked_mul(8));
                if let Some(bytes) = len.filter(|b| *b <= budget - cache_bytes) {
                    cache.push(Coupling {
                        i,
                        j,
                        values: coupling(&monomers[i].basis, &monomers[j].basis),
                    });
                    cache_bytes += bytes;
                }
            }
        }
    }
    let mut embedding_iterations = 0;
    let mut change = 0f64;
    if o.electrostatic_embedding && fragments.len() > 1 {
        let mut previous: f64 = states.iter().map(|s| s.internal_energy).sum();
        let mut converged = false;
        for iteration in 1..=o.max_embedding_iterations {
            embedding_iterations = iteration;
            change = 0.;
            for i in 0..monomers.len() {
                let field = monomer_field(i, &monomers, &states, &cache);
                let next = monomers[i].solve(field, o)?;
                change = change.max(max_diff(&next.density_matrix, &states[i].density_matrix));
                states[i] = next;
            }
            let fields: Vec<_> = (0..monomers.len())
                .map(|i| monomer_field(i, &monomers, &states, &cache))
                .collect();
            for (i, field) in fields.into_iter().enumerate() {
                states[i].embedding_potential = field;
                monomers[i].refresh(&mut states[i]);
            }
            let energy: f64 = states.iter().map(|s| s.internal_energy).sum();
            if change <= o.embedding_density_tolerance
                && (energy - previous).abs() <= o.tolerance
                && states
                    .iter()
                    .all(|s| s.residual_norm <= o.residual_tolerance)
            {
                converged = true;
                break;
            }
            previous = energy;
        }
        if !converged {
            return Err(format!(
                "FMO monomer embedding did not converge in {embedding_iterations} cycles: density change {change:e}"
            ));
        }
    }
    let monomer_energies: Vec<_> = states.iter().map(|s| s.internal_energy).collect();
    let mut total_energy = monomer_energies.iter().sum::<f64>();
    let mut dimer_results = Vec::new();
    let mut dimer_energies = Vec::new();
    let mut ifie = Vec::new();
    if o.order > 1 {
        for i in 0..fragments.len() {
            for j in i + 1..fragments.len() {
                let c = Cluster::new(atoms, basis, fragments, &[i, j], &global_h);
                let field = if o.electrostatic_embedding {
                    cluster_field(&c, &[i, j], &monomers, &states)
                } else {
                    Matrix::zeros(c.basis.len(), c.basis.len())
                };
                let state = c.solve(field, o)?;
                let interaction = decompose(&c, &state, i, j, &monomers, &states)?;
                total_energy += interaction.total_interaction_energy;
                ifie.push(interaction);
                dimer_energies.push(((i, j), state.internal_energy));
                dimer_results.push(((i, j), state));
            }
        }
    }
    let mut trimer_energies = None;
    let mut trimer_results = None;
    if o.order > 2 {
        let mut energies = Vec::new();
        let mut results = Vec::new();
        for i in 0..fragments.len() {
            for j in i + 1..fragments.len() {
                for k in j + 1..fragments.len() {
                    let members = [i, j, k];
                    let c = Cluster::new(atoms, basis, fragments, &members, &global_h);
                    let field = if o.electrostatic_embedding {
                        cluster_field(&c, &members, &monomers, &states)
                    } else {
                        Matrix::zeros(c.basis.len(), c.basis.len())
                    };
                    let state = c.solve(field, o)?;
                    let (product, _) = monomer_sum(&c, &members, &states);
                    let pair_sum: f64 = ifie
                        .iter()
                        .filter(|p| members.contains(&p.frag_i) && members.contains(&p.frag_j))
                        .map(|p| p.total_interaction_energy)
                        .sum();
                    let delta = state.internal_energy
                        - members.iter().map(|&a| monomer_energies[a]).sum::<f64>()
                        - pair_sum
                        + trace(
                            &state.density_matrix.sub(&product),
                            &state.embedding_potential,
                        );
                    total_energy += delta;
                    energies.push(((i, j, k), delta));
                    results.push(((i, j, k), state));
                }
            }
        }
        trimer_energies = Some(energies);
        trimer_results = Some(results);
    }
    Ok(FMOResult {
        total_energy,
        monomer_energies,
        dimer_energies,
        ifie,
        trimer_energies,
        monomer_results: states,
        dimer_results,
        trimer_results,
        order: o.order,
        electrostatic_embedding: o.electrostatic_embedding,
        embedding_iterations,
        embedding_density_change: change,
        esp_cache_bytes: cache_bytes,
        estimated_workspace_bytes: estimate + 2 * cache_bytes,
        converged: true,
    })
}
/// Analytic gradients for isolated expansions and full-system truncations.
/// Other embedded expansions require coupled monomer/environment response.
pub fn compute_fmo_gradient(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    fragments: &[FMOFragment],
    r: &FMOResult,
) -> Result<Vec<[f64; 3]>, String> {
    if r.electrostatic_embedding && fragments.len() > r.order {
        return Err("analytic embedded FMO nuclear gradients require coupled monomer/environment response, which is not implemented".into());
    }
    if !r.converged || fragments.len() != r.monomer_results.len() || !(1..=3).contains(&r.order) {
        return Err("FMO gradient requires a matching converged result".into());
    }
    let mut gradient = vec![[0.; 3]; atoms.len()];
    let nf = fragments.len();
    let order = r.order.min(nf);
    let weight = |size: usize| -> f64 {
        if size > order {
            return 0.;
        }
        if nf == order {
            return if size == nf { 1. } else { 0. };
        }
        let mut choose = 1.;
        for k in 0..order - size {
            choose *= (nf - size - 1 - k) as f64 / (k + 1) as f64;
        }
        if (order - size).is_multiple_of(2) {
            choose
        } else {
            -choose
        }
    };
    let mut add = |s: &FMOClusterResult, factor: f64| -> Result<(), String> {
        if factor == 0. {
            return Ok(());
        }
        if s.atom_indices.iter().any(|&a| a >= atoms.len())
            || s.ao_indices.iter().any(|&a| a >= basis.len())
            || s.mo_coefficients.cols != s.ao_indices.len()
        {
            return Err("FMO analytic gradient requires a matching full-rank AO frame".into());
        }
        let sub_atoms: Vec<_> = s.atom_indices.iter().map(|&a| atoms[a].clone()).collect();
        let sub_basis: Vec<_> = s
            .ao_indices
            .iter()
            .map(|&a| {
                let mut b = basis[a].clone();
                b.atom_index = s.atom_indices.binary_search(&b.atom_index).unwrap();
                b
            })
            .collect();
        let mut w = Matrix::zeros(sub_basis.len(), sub_basis.len());
        for i in 0..sub_basis.len() {
            for j in 0..sub_basis.len() {
                for k in 0..s.n_occ {
                    w[(i, j)] += 2.
                        * s.orbital_energies[k]
                        * s.mo_coefficients[(i, k)]
                        * s.mo_coefficients[(j, k)];
                }
            }
        }
        let g = crate::gradient::compute_gradient_from_density(
            &sub_atoms,
            &sub_basis,
            &s.density_matrix,
            &w,
        );
        for (local, &global) in s.atom_indices.iter().enumerate() {
            for axis in 0..3 {
                gradient[global][axis] += factor * g[local][axis];
            }
        }
        Ok(())
    };
    for s in &r.monomer_results {
        add(s, weight(1))?;
    }
    for (_, s) in &r.dimer_results {
        add(s, weight(2))?;
    }
    if let Some(trimers) = &r.trimer_results {
        for (_, s) in trimers {
            add(s, weight(3))?;
        }
    }
    Ok(gradient)
}
