// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! Constant-dimensional local spaces for continuous nuclear motion. Atomic
//! domains, strong/weak classifications and virtual subspaces are selected at
//! one geometry, then overlap-transported without new threshold decisions.
//! Analytic derivatives require differentiating this transport and the HF orbitals.
use super::local_ccsd::{LocalCcsdResult, compute_with_pair_problem};
use super::local_mp2::{
    BuiltPairProblem, DomainFits, LocalMp2, LocalMp2Result, build_pair_problem,
};
use super::local_triples::{LocalCcsdTriplesResult, TriplesSpaceMap};
use crate::local::{
    pair_space::{LocalPairProblem, PairBlock, natural_pair_space, pair_energy},
    pao::orthonormalize_paos,
};
use crate::registry::ContextMethod;
use crate::{
    Atom, ContractedGaussian, Matrix, context::PostHFContext, error::HfError,
    options::PostHFOptions,
};
use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Debug)]
struct AnchoredPair {
    i: usize,
    j: usize,
    pno: Matrix,
    domain: Matrix,
    strong: bool,
    atoms: Vec<usize>,
    fit_atoms: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct LocalSpaceAnchor {
    atoms: Vec<Atom>,
    basis: Vec<ContractedGaussian>,
    occupied: Matrix,
    virtuals: Matrix,
    pairs: Vec<AnchoredPair>,
    options: PostHFOptions,
}
/// Fixed pair and triples spaces, selected only at the anchor geometry.
/// Geometry evaluation transports both spaces and reconverges CCSD and (T1).
#[derive(Clone, Debug)]
pub struct LocalTriplesAnchor {
    pairs: LocalSpaceAnchor,
    spaces: TriplesSpaceMap,
}

impl LocalTriplesAnchor {
    pub fn from_context<'a>(
        ctx: &'a PostHFContext<'a>,
        options: &PostHFOptions,
    ) -> Result<Self, HfError> {
        let pairs = LocalSpaceAnchor::from_context(ctx, options)?;
        let cc = pairs.compute_ccsd(ctx)?;
        let problem = super::local_triples::build_problem(ctx, options, &cc, None)?;
        let mut spaces = HashMap::new();
        for block in problem.blocks {
            let mut key = [block.i, block.j, block.k];
            key.sort_unstable();
            spaces.entry(key).or_insert(block.space);
        }
        Ok(Self { pairs, spaces })
    }

    pub fn compute_t0<'a>(
        &self,
        ctx: &'a PostHFContext<'a>,
    ) -> Result<LocalCcsdTriplesResult, HfError> {
        if self.pairs.options.local.iterative_triples {
            return Err(HfError::InvalidInput(
                "iterative_triples=true requires the (T1) method".into(),
            ));
        }
        self.compute(ctx, false)
    }

    pub fn compute_t1<'a>(
        &self,
        ctx: &'a PostHFContext<'a>,
    ) -> Result<LocalCcsdTriplesResult, HfError> {
        self.compute(ctx, true)
    }

    fn compute<'a>(
        &self,
        ctx: &'a PostHFContext<'a>,
        iterative: bool,
    ) -> Result<LocalCcsdTriplesResult, HfError> {
        // Pair transport checks the ordered basis and physical orbital partition.
        let ccsd = self.pairs.compute_ccsd(ctx)?;
        let hf = ctx.rhf()?;
        let cross =
            aligned_cross_overlap(ctx.atoms, ctx.basis, &self.pairs.atoms, &self.pairs.basis);
        let transport = ccsd
            .virtuals
            .transpose()
            .dot(&cross)
            .dot(&self.pairs.virtuals);
        let mut fv = Matrix::zeros(ccsd.virtuals.cols, ccsd.virtuals.cols);
        for a in 0..fv.rows {
            fv[(a, a)] = hf.orbital_energies[hf.n_occ + a];
        }
        let mut spaces = HashMap::new();
        for (key, original) in &self.spaces {
            let basis = polar_columns(&transport.dot(&original.basis))?;
            let (energies, rotation) = basis.transpose().dot(&fv).dot(&basis).eigh();
            spaces.insert(
                *key,
                Arc::new(crate::local::triples_space::TriplesVirtualSpace {
                    basis: basis.dot(&rotation),
                    energies,
                }),
            );
        }
        let options = &self.pairs.options;
        let problem = super::local_triples::build_problem(ctx, options, &ccsd, Some(&spaces))?;
        let tno_ranks = problem.blocks.iter().map(|b| b.space.basis.cols).collect();
        let triples = problem.solve(iterative, options.max_iter, options.residual_tol)?;
        Ok(LocalCcsdTriplesResult {
            ccsd,
            triples,
            iterative,
            tno_ranks,
        })
    }
}

use crate::local::transport::orthonormal_frame as polar_columns;
// Horn quaternion alignment removes rigid translation/rotation before overlap
// transport. Rotating the Cartesian AO polynomial keeps finite spaces covariant.
fn aligned_cross_overlap(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    reference_atoms: &[Atom],
    reference_basis: &[ContractedGaussian],
) -> Matrix {
    let center = |a: &[Atom]| {
        std::array::from_fn::<_, 3, _>(|axis| {
            a.iter().map(|a| a.position[axis]).sum::<f64>() / a.len() as f64
        })
    };
    let c = center(atoms);
    let r = center(reference_atoms);
    let mut covariance = [[0.; 3]; 3];
    for (a, b) in atoms.iter().zip(reference_atoms) {
        for i in 0..3 {
            for j in 0..3 {
                covariance[i][j] += (a.position[i] - c[i]) * (b.position[j] - r[j]);
            }
        }
    }
    let v = covariance;
    let mut horn = Matrix::zeros(4, 4);
    horn[(0, 0)] = v[0][0] + v[1][1] + v[2][2];
    for (axis, value) in [v[1][2] - v[2][1], v[2][0] - v[0][2], v[0][1] - v[1][0]]
        .into_iter()
        .enumerate()
    {
        horn[(0, axis + 1)] = value;
        horn[(axis + 1, 0)] = value;
    }
    for i in 0..3 {
        for j in 0..3 {
            horn[(i + 1, j + 1)] = v[i][j] + v[j][i] - if i == j { horn[(0, 0)] } else { 0. };
        }
    }
    let (_, eigenvectors) = horn.eigh();
    let w = eigenvectors[(0, 3)];
    let x = eigenvectors[(1, 3)];
    let y = eigenvectors[(2, 3)];
    let z = eigenvectors[(3, 3)];
    let rotation = [
        [
            1. - 2. * (y * y + z * z),
            2. * (x * y - z * w),
            2. * (x * z + y * w),
        ],
        [
            2. * (x * y + z * w),
            1. - 2. * (x * x + z * z),
            2. * (y * z - x * w),
        ],
        [
            2. * (x * z - y * w),
            2. * (y * z + x * w),
            1. - 2. * (x * x + y * y),
        ],
    ];
    let mut cross = Matrix::zeros(basis.len(), reference_basis.len());
    for (mu, g) in basis.iter().enumerate() {
        let center = std::array::from_fn(|i| {
            r[i] + (0..3)
                .map(|j| rotation[i][j] * (g.center[j] - c[j]))
                .sum::<f64>()
        });
        let mut polynomial = std::collections::BTreeMap::new();
        polynomial.insert([0usize; 3], 1.);
        for original in 0..3 {
            for _ in 0..g.l[original] {
                let mut next = std::collections::BTreeMap::new();
                for (powers, coefficient) in polynomial {
                    for aligned in 0..3 {
                        let mut raised = powers;
                        raised[aligned] += 1;
                        *next.entry(raised).or_insert(0.) +=
                            coefficient * rotation[aligned][original];
                    }
                }
                polynomial = next;
            }
        }
        for (powers, coefficient) in polynomial {
            if coefficient == 0. {
                continue;
            }
            let component = ContractedGaussian {
                atom_index: g.atom_index,
                center,
                l: powers,
                primitives: g.primitives.clone(),
            };
            for (nu, reference) in reference_basis.iter().enumerate() {
                cross[(mu, nu)] +=
                    coefficient * crate::integrals::contracted_overlap(&component, reference);
            }
        }
    }
    cross
}

impl LocalSpaceAnchor {
    pub fn from_context<'a>(
        ctx: &'a PostHFContext<'a>,
        options: &PostHFOptions,
    ) -> Result<Self, HfError> {
        LocalMp2::from_context(ctx, options)?;
        let built = build_pair_problem(ctx, options)?;
        let hf = ctx.rhf()?;
        let q = built
            .virtuals
            .dot(&built.virtuals.transpose())
            .dot(&hf.overlap_matrix);
        let mut pairs = Vec::with_capacity(built.problem.pairs.len());
        for (index, p) in built.problem.pairs.iter().enumerate() {
            let selected = ctx
                .basis
                .iter()
                .enumerate()
                .filter(|(_, g)| built.domain_atoms[index].contains(&g.atom_index))
                .map(|(mu, _)| mu)
                .collect::<Vec<_>>();
            let paos = orthonormalize_paos(&q, &hf.overlap_matrix, &selected, 1e-10)?;
            pairs.push(AnchoredPair {
                i: p.i,
                j: p.j,
                pno: p.basis.clone(),
                domain: built
                    .virtuals
                    .transpose()
                    .dot(&hf.overlap_matrix)
                    .dot(&paos),
                strong: p.strong,
                atoms: built.domain_atoms[index].clone(),
                fit_atoms: built.fit_atoms[index].clone(),
            });
        }
        Ok(Self {
            atoms: ctx.atoms.to_vec(),
            basis: ctx.basis.to_vec(),
            occupied: built.occupied,
            virtuals: built.virtuals,
            pairs,
            options: options.clone(),
        })
    }
    pub fn pair_ranks(&self) -> Vec<usize> {
        self.pairs.iter().map(|p| p.pno.cols).collect()
    }
    /// Re-evaluate the converged Galerkin/Hylleraas energy in transported spaces.
    /// The anchor is immutable: no geometry-dependent domain/pair/rank screening.
    pub fn compute_mp2<'a>(&self, ctx: &'a PostHFContext<'a>) -> Result<LocalMp2Result, HfError> {
        let built = self.transported_problem(ctx)?;
        let solution = built
            .problem
            .solve(self.options.max_iter, self.options.residual_tol)?;
        Ok(LocalMp2Result {
            e_scf: ctx.reference.energy(),
            solution,
            n_strong_pairs: built.problem.pairs.iter().filter(|p| p.strong).count(),
            n_weak_pairs: built.problem.pairs.iter().filter(|p| !p.strong).count(),
            pair_ranks: built.problem.pairs.iter().map(|p| p.basis.cols).collect(),
            pair_domain_atoms: built.domain_atoms,
            pair_fit_atoms: built.fit_atoms,
        })
    }

    /// Converge the true CCSD residual in the fixed transported pair spaces.
    /// PNO/domain/weak-pair choices remain frozen at the anchor geometry.
    pub fn compute_ccsd<'a>(&self, ctx: &'a PostHFContext<'a>) -> Result<LocalCcsdResult, HfError> {
        compute_with_pair_problem(ctx, &self.options, self.transported_problem(ctx)?)
    }

    fn transported_problem<'a>(
        &self,
        ctx: &'a PostHFContext<'a>,
    ) -> Result<BuiltPairProblem, HfError> {
        LocalMp2::from_context(ctx, &self.options)?;
        let hf = ctx.rhf()?;
        let no = self.occupied.cols;
        let nv = self.virtuals.cols;
        if ctx.atoms.len() != self.atoms.len()
            || ctx.basis.len() != self.basis.len()
            || ctx
                .atoms
                .iter()
                .zip(&self.atoms)
                .any(|(a, b)| a.atomic_number != b.atomic_number || a.charge != b.charge)
            || ctx.basis.iter().zip(&self.basis).any(|(a, b)| {
                a.atom_index != b.atom_index
                    || a.l != b.l
                    || a.primitives.len() != b.primitives.len()
                    || a.primitives
                        .iter()
                        .zip(&b.primitives)
                        .any(|(a, b)| a.exponent != b.exponent || a.coefficient != b.coefficient)
            })
            || hf.n_occ - self.options.frozen_core != no
            || hf.mo_coefficients.cols - hf.n_occ - self.options.frozen_virtual != nv
        {
            return Err(HfError::InvalidInput("local-space transport requires the same atoms, ordered basis and retained orbital partition".into()));
        }
        let cross = aligned_cross_overlap(ctx.atoms, ctx.basis, &self.atoms, &self.basis);
        let mut occ = Matrix::zeros(ctx.basis.len(), no);
        let mut virt = Matrix::zeros(ctx.basis.len(), nv);
        for mu in 0..ctx.basis.len() {
            for i in 0..no {
                occ[(mu, i)] = hf.mo_coefficients[(mu, self.options.frozen_core + i)];
            }
            for a in 0..nv {
                virt[(mu, a)] = hf.mo_coefficients[(mu, hf.n_occ + a)];
            }
        }
        let rotation = polar_columns(&occ.transpose().dot(&cross).dot(&self.occupied))?;
        let occupied = occ.dot(&rotation);
        let virtual_overlap = virt.transpose().dot(&cross).dot(&self.virtuals);
        let f_occ = occupied.transpose().dot(&hf.fock_matrix).dot(&occupied);
        let mut fv = Matrix::zeros(nv, nv);
        for a in 0..nv {
            fv[(a, a)] = hf.orbital_energies[hf.n_occ + a];
        }
        let mut fits = DomainFits::new(ctx);
        let mut pairs = Vec::with_capacity(self.pairs.len());
        for anchor in &self.pairs {
            let domain = polar_columns(&virtual_overlap.dot(&anchor.domain))?;
            let pno = polar_columns(&virtual_overlap.dot(&anchor.pno))?;
            let (energies, u) = pno.transpose().dot(&fv).dot(&pno).eigh();
            let pno = pno.dot(&u);
            let domain_ao = virt.dot(&domain);
            let m = domain.cols;
            let mut coeff = Matrix::zeros(ctx.basis.len(), m + 2);
            for mu in 0..coeff.rows {
                coeff[(mu, 0)] = occupied[(mu, anchor.i)];
                coeff[(mu, 1)] = occupied[(mu, anchor.j)];
                for a in 0..m {
                    coeff[(mu, a + 2)] = domain_ao[(mu, a)];
                }
            }
            let fit = fits.get(&anchor.fit_atoms)?.transform(&coeff)?;
            let mut k = Matrix::zeros(m, m);
            for a in 0..m {
                for b in 0..m {
                    k[(a, b)] = fit.eri(0, a + 2, 1, b + 2);
                }
            }
            let f = domain.transpose().dot(&fv).dot(&domain);
            let full = natural_pair_space(
                &k,
                &f,
                f_occ[(anchor.i, anchor.i)] + f_occ[(anchor.j, anchor.j)],
                anchor.i == anchor.j,
                0.,
            )?;
            let t = full
                .coefficients
                .dot(&full.initial_amplitudes)
                .dot(&full.coefficients.transpose());
            let overlap = domain.transpose().dot(&pno);
            let exchange = overlap.transpose().dot(&k).dot(&overlap);
            let initial = overlap.transpose().dot(&t).dot(&overlap);
            let delta_pno = if anchor.strong {
                full.untruncated_estimate - pair_energy(&exchange, &initial)
            } else {
                0.
            };
            pairs.push(PairBlock {
                i: anchor.i,
                j: anchor.j,
                basis: pno,
                energies,
                exchange,
                initial,
                strong: anchor.strong,
                delta_pno,
            });
        }
        let problem = LocalPairProblem { f_occ, pairs };
        Ok(BuiltPairProblem {
            problem,
            occupied,
            virtuals: virt,
            domain_atoms: self.pairs.iter().map(|p| p.atoms.clone()).collect(),
            fit_atoms: self.pairs.iter().map(|p| p.fit_atoms.clone()).collect(),
        })
    }
}
