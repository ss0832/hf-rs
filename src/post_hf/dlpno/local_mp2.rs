// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! RI local-MP2 with checked localization, atomic PAO/fit domains, pair-natural
//! virtuals and occupied-Fock coupling in the amplitude equations.
use crate::context::PostHFContext;
use crate::df::DfTensor;
use crate::error::HfError;
use crate::local::domains::{differential_overlaps, modified_mulliken_populations};
use crate::local::localization::foster_boys;
use crate::local::pair_space::{LocalMp2Solution, LocalPairProblem, PairBlock, natural_pair_space};
use crate::local::pao::orthonormalize_paos;
use crate::matrix::Matrix;
use crate::options::PostHFOptions;
use crate::output::PostHFOutput;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::{ContextMethod, MethodId, MethodMeta};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

fn columns(c: &Matrix, start: usize, end: usize) -> Matrix {
    let mut out = Matrix::zeros(c.rows, end - start);
    for mu in 0..c.rows {
        for p in start..end {
            out[(mu, p - start)] = c[(mu, p)];
        }
    }
    out
}
#[derive(Clone, Debug)]
pub struct LocalMp2Result {
    pub e_scf: f64,
    pub solution: LocalMp2Solution,
    pub n_strong_pairs: usize,
    pub n_weak_pairs: usize,
    pub pair_ranks: Vec<usize>,
    pub pair_domain_atoms: Vec<Vec<usize>>,
    pub pair_fit_atoms: Vec<Vec<usize>>,
}
impl PostHFResult for LocalMp2Result {
    fn method_name(&self) -> &'static str {
        "DLPNO-MP2"
    }
    fn reference_energy(&self) -> f64 {
        self.e_scf
    }
    fn correlation_energy(&self) -> f64 {
        self.solution.correlation_energy()
    }
    fn to_output(&self) -> Result<PostHFOutput, HfError> {
        let mut out = PostHFOutput::new(self.method_name(), self.e_scf, self.total_energy())?;
        out.components
            .insert("E_strong_pairs".into(), self.solution.e_strong);
        out.components
            .insert("E_weak_pairs".into(), self.solution.e_weak);
        out.components
            .insert("E_delta_PNO".into(), self.solution.delta_pno);
        out.components
            .insert("residual_max".into(), self.solution.residual_max);
        Ok(out)
    }
}
pub struct LocalMp2<'a> {
    ctx: &'a PostHFContext<'a>,
    options: PostHFOptions,
}
impl MethodMeta for LocalMp2<'_> {
    const METHOD: MethodId = MethodId::DlpnoMp2;
}
impl<'a> ContextMethod<'a> for LocalMp2<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        options.validate()?;
        let hf = ctx.rhf()?;
        if ctx.aux.is_none() {
            return Err(HfError::InvalidInput(
                "DLPNO-MP2 requires an auxiliary basis".into(),
            ));
        }
        if options.active.is_some() || options.root != 0 || options.n_roots != 1 {
            return Err(HfError::InvalidInput(
                "DLPNO-MP2 uses the RHF occupied/virtual partition and one ground state".into(),
            ));
        }
        if options.frozen_core > hf.n_occ
            || options.frozen_virtual > hf.mo_coefficients.cols - hf.n_occ
        {
            return Err(HfError::InvalidInput(
                "frozen local-MP2 partition exceeds the reference space".into(),
            ));
        }
        Ok(Self {
            ctx,
            options: options.clone(),
        })
    }
}
impl PostHFMethod for LocalMp2<'_> {
    type Output = LocalMp2Result;
    fn compute(&self) -> Result<Self::Output, String> {
        self.compute_checked().map_err(|e| e.to_string())
    }
}
impl LocalMp2<'_> {
    pub fn compute_checked(&self) -> Result<LocalMp2Result, HfError> {
        let built = build_pair_problem(self.ctx, &self.options)?;
        let solution = built
            .problem
            .solve(self.options.max_iter, self.options.residual_tol)?;
        Ok(LocalMp2Result {
            e_scf: self.ctx.reference.energy(),
            solution,
            n_strong_pairs: built.problem.pairs.iter().filter(|p| p.strong).count(),
            n_weak_pairs: built.problem.pairs.iter().filter(|p| !p.strong).count(),
            pair_ranks: built.problem.pairs.iter().map(|p| p.basis.cols).collect(),
            pair_domain_atoms: built.domain_atoms,
            pair_fit_atoms: built.fit_atoms,
        })
    }
}

/// Bounded, shared AO DF fits for distinct auxiliary domains. The global fit is
/// the context cache; local fits are evicted at 64 MiB rather than accumulated
/// for every occupied pair.
pub(crate) struct DomainFits<'a> {
    ctx: &'a PostHFContext<'a>,
    cache: HashMap<Vec<usize>, Arc<DfTensor>>,
    order: VecDeque<Vec<usize>>,
    bytes: usize,
}
impl<'a> DomainFits<'a> {
    pub(crate) fn new(ctx: &'a PostHFContext<'a>) -> Self {
        Self {
            ctx,
            cache: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
        }
    }
    pub(crate) fn get(&mut self, atoms: &[usize]) -> Result<Arc<DfTensor>, HfError> {
        if atoms.len() == self.ctx.atoms.len() {
            return Ok(self.ctx.ints.df_ao()?.clone());
        }
        if let Some(fit) = self.cache.get(atoms) {
            return Ok(fit.clone());
        }
        let aux = self
            .ctx
            .aux
            .ok_or_else(|| HfError::InvalidInput("local RI requires auxiliary functions".into()))?;
        let functions = aux
            .functions
            .iter()
            .filter(|g| atoms.contains(&g.atom_index))
            .cloned()
            .collect::<Vec<_>>();
        let fit = Arc::new(DfTensor::build(self.ctx.basis, &functions, 1e-10)?);
        let size = fit.b.data.len() * 8;
        if size <= 64 * 1024 * 1024 {
            while self.bytes + size > 64 * 1024 * 1024 {
                if let Some(old) = self.order.pop_front() {
                    if let Some(fit) = self.cache.remove(&old) {
                        self.bytes -= fit.b.data.len() * 8;
                    }
                } else {
                    break;
                }
            }
            self.bytes += size;
            self.cache.insert(atoms.to_vec(), fit.clone());
            self.order.push_back(atoms.to_vec());
        }
        Ok(fit)
    }
}
pub(crate) struct BuiltPairProblem {
    pub problem: LocalPairProblem,
    pub occupied: Matrix,
    pub virtuals: Matrix,
    pub domain_atoms: Vec<Vec<usize>>,
    pub fit_atoms: Vec<Vec<usize>>,
}
fn union(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = a.to_vec();
    out.extend_from_slice(b);
    out.sort_unstable();
    out.dedup();
    out
}
pub(crate) fn build_pair_problem<'a>(
    ctx: &'a PostHFContext<'a>,
    options: &PostHFOptions,
) -> Result<BuiltPairProblem, HfError> {
    let hf = ctx.rhf()?;
    let no = hf.n_occ - options.frozen_core;
    let nv = hf.mo_coefficients.cols - hf.n_occ - options.frozen_virtual;
    let canonical_occ = columns(&hf.mo_coefficients, options.frozen_core, hf.n_occ);
    let virtuals = columns(&hf.mo_coefficients, hf.n_occ, hf.n_occ + nv);
    let dipoles = [
        crate::properties::compute_dipole_matrix(ctx.basis, 0),
        crate::properties::compute_dipole_matrix(ctx.basis, 1),
        crate::properties::compute_dipole_matrix(ctx.basis, 2),
    ];
    let occupied = foster_boys(&canonical_occ, &dipoles, 1000, 1e-10)?.coefficients;
    let f_occ = occupied.transpose().dot(&hf.fock_matrix).dot(&occupied);
    let mut f_virtual = Matrix::zeros(nv, nv);
    for a in 0..nv {
        f_virtual[(a, a)] = hf.orbital_energies[hf.n_occ + a];
    }
    // Only the retained physical virtual space is permitted: Q=V V^T S also
    // handles spherical AO spaces, overlap rank removal and frozen virtuals.
    let q = virtuals.dot(&virtuals.transpose()).dot(&hf.overlap_matrix);
    let nat = ctx.atoms.len();
    let all_atoms = (0..nat).collect::<Vec<_>>();
    let all_domains = options.local.full_domains;
    let mut virtual_domains = vec![all_atoms.clone(); no];
    let mut fit_domains = vec![all_atoms.clone(); no];
    if !all_domains && no > 0 && nv > 0 {
        let doi = if options.local.t_cut_do > 0. {
            Some(differential_overlaps(
                ctx.atoms,
                ctx.basis,
                &occupied,
                &q,
                &hf.overlap_matrix,
                96,
                40,
            )?)
        } else {
            None
        };
        let mkn = modified_mulliken_populations(&occupied, &hf.overlap_matrix, ctx.basis, nat)?;
        for i in 0..no {
            let largest = (0..nat)
                .max_by(|&a, &b| mkn[(i, a)].abs().total_cmp(&mkn[(i, b)].abs()))
                .unwrap_or(0);
            if let Some(doi) = &doi {
                let mut selected = vec![false; nat];
                for (mu, g) in ctx.basis.iter().enumerate() {
                    if doi.occupied_pao[(i, mu)] > options.local.t_cut_do {
                        selected[g.atom_index] = true;
                    }
                }
                // A nonempty atomic domain is required even above all DOI values.
                if !selected.contains(&true) {
                    selected[largest] = true;
                }
                virtual_domains[i] = (0..nat).filter(|&a| selected[a]).collect();
            }
            if options.local.t_cut_mkn > 0. {
                fit_domains[i] = (0..nat)
                    .filter(|&a| mkn[(i, a)].abs() > options.local.t_cut_mkn)
                    .collect();
                if fit_domains[i].is_empty() {
                    fit_domains[i].push(largest);
                }
            }
        }
    }
    let mut fits = DomainFits {
        ctx,
        cache: HashMap::new(),
        order: VecDeque::new(),
        bytes: 0,
    };
    let mut pairs = Vec::new();
    let mut domain_atoms = Vec::new();
    let mut fit_atoms = Vec::new();
    let mut workspace = 0usize;
    for i in 0..no {
        for j in 0..=i {
            let domain = union(&virtual_domains[i], &virtual_domains[j]);
            let fit_domain = union(&fit_domains[i], &fit_domains[j]);
            let selected = ctx
                .basis
                .iter()
                .enumerate()
                .filter(|(_, g)| domain.contains(&g.atom_index))
                .map(|(mu, _)| mu)
                .collect::<Vec<_>>();
            let paos = orthonormalize_paos(&q, &hf.overlap_matrix, &selected, 1e-10)?;
            let v = virtuals.transpose().dot(&hf.overlap_matrix).dot(&paos);
            let m = paos.cols;
            workspace = workspace
                .checked_add(8 * (nv * m + 16 * m * m))
                .ok_or_else(|| HfError::InvalidInput("pair workspace size overflow".into()))?;
            if workspace > 256 * 1024 * 1024 {
                return Err(HfError::Unsupported {
                    method: "DLPNO-MP2".into(),
                    detail: "pair workspace exceeds 256 MiB".into(),
                });
            }
            let fit = fits.get(&fit_domain)?;
            let mut coeff = Matrix::zeros(ctx.basis.len(), m + 2);
            for mu in 0..coeff.rows {
                coeff[(mu, 0)] = occupied[(mu, i)];
                coeff[(mu, 1)] = occupied[(mu, j)];
                for a in 0..m {
                    coeff[(mu, a + 2)] = paos[(mu, a)];
                }
            }
            let transformed = fit.transform(&coeff)?;
            let mut k = Matrix::zeros(m, m);
            for a in 0..m {
                for b in 0..m {
                    k[(a, b)] = transformed.eri(0, a + 2, 1, b + 2);
                }
            }
            let f = v.transpose().dot(&f_virtual).dot(&v);
            let esum = f_occ[(i, i)] + f_occ[(j, j)];
            let initial = natural_pair_space(&k, &f, esum, i == j, 0.)?;
            let strong = !options.local.weak_pairs
                || initial.untruncated_estimate.abs() >= options.local.t_cut_pairs;
            let pno = if strong && options.local.t_cut_pno > 0. {
                natural_pair_space(&k, &f, esum, i == j, options.local.t_cut_pno)?
            } else {
                initial
            };
            let exchange = pno.coefficients.transpose().dot(&k).dot(&pno.coefficients);
            pairs.push(PairBlock {
                i,
                j,
                basis: v.dot(&pno.coefficients),
                energies: pno.energies,
                exchange,
                initial: pno.initial_amplitudes,
                strong,
                delta_pno: pno.delta_pno,
            });
            domain_atoms.push(domain);
            fit_atoms.push(fit_domain);
        }
    }
    let problem = LocalPairProblem { f_occ, pairs };
    problem.validate()?;
    Ok(BuiltPairProblem {
        problem,
        occupied,
        virtuals,
        domain_atoms,
        fit_atoms,
    })
}
