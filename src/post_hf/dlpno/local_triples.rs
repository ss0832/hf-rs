// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! Semicanonical (T0) and occupied-Fock-coupled iterative (T1) in atomic-domain
//! triples natural orbitals. All ordered occupied triples are retained.
use super::local_ccsd::{LocalCcsd, LocalCcsdResult};
use crate::context::PostHFContext;
use crate::error::HfError;
use crate::local::pao::orthonormalize_paos;
use crate::local::triples_space::*;
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::{ContextMethod, MethodId, MethodMeta};
use crate::{Matrix, options::PostHFOptions, output::PostHFOutput};
use std::{collections::HashMap, sync::Arc};

pub(crate) type TriplesSpaceMap = HashMap<[usize; 3], Arc<TriplesVirtualSpace>>;

#[derive(Clone, Debug)]
pub struct LocalCcsdTriplesResult {
    pub ccsd: LocalCcsdResult,
    pub triples: LocalTriplesSolution,
    pub iterative: bool,
    pub tno_ranks: Vec<usize>,
}
impl PostHFResult for LocalCcsdTriplesResult {
    fn method_name(&self) -> &'static str {
        if self.iterative {
            "DLPNO-CCSD(T1)"
        } else {
            "DLPNO-CCSD(T0)"
        }
    }
    fn reference_energy(&self) -> f64 {
        self.ccsd.e_scf
    }
    fn correlation_energy(&self) -> f64 {
        self.ccsd.correlation_energy() + self.triples.energy
    }
    fn to_output(&self) -> Result<PostHFOutput, HfError> {
        let mut out = PostHFOutput::new(
            self.method_name(),
            self.reference_energy(),
            self.total_energy(),
        )?;
        out.components
            .insert("E_CCSD_pair_space".into(), self.ccsd.e_ccsd);
        out.components
            .insert("E_delta_PNO".into(), self.ccsd.delta_pno);
        out.components
            .insert("E_triples".into(), self.triples.energy);
        out.components
            .insert("triples_residual_max".into(), self.triples.residual_max);
        Ok(out)
    }
}
pub struct LocalCcsdT0<'a> {
    ctx: &'a PostHFContext<'a>,
    options: PostHFOptions,
}
pub struct LocalCcsdT1<'a> {
    ctx: &'a PostHFContext<'a>,
    options: PostHFOptions,
}
macro_rules! engine {
    ($name:ident,$method:ident,$iterative:literal) => {
        impl MethodMeta for $name<'_> {
            const METHOD: MethodId = MethodId::$method;
        }
        impl<'a> ContextMethod<'a> for $name<'a> {
            fn from_context(
                ctx: &'a PostHFContext<'a>,
                options: &PostHFOptions,
            ) -> Result<Self, HfError> {
                LocalCcsd::from_context(ctx, options)?;
                if !$iterative && options.local.iterative_triples {
                    return Err(HfError::InvalidInput(
                        "iterative_triples=true requires the (T1) method".into(),
                    ));
                }
                Ok(Self {
                    ctx,
                    options: options.clone(),
                })
            }
        }
        impl PostHFMethod for $name<'_> {
            type Output = LocalCcsdTriplesResult;
            fn compute(&self) -> Result<Self::Output, String> {
                compute(self.ctx, &self.options, $iterative).map_err(|e| e.to_string())
            }
        }
    };
}
engine!(LocalCcsdT0, DlpnoCcsdT0, false);
engine!(LocalCcsdT1, DlpnoCcsdT1, true);
fn pair(i: usize, j: usize) -> usize {
    let hi = i.max(j);
    hi * (hi + 1) / 2 + i.min(j)
}
fn compute<'a>(
    ctx: &'a PostHFContext<'a>,
    options: &PostHFOptions,
    iterative: bool,
) -> Result<LocalCcsdTriplesResult, HfError> {
    let ccsd = LocalCcsd::from_context(ctx, options)?.compute_checked()?;
    let problem = build_problem(ctx, options, &ccsd, None)?;
    let tno_ranks = problem.blocks.iter().map(|b| b.space.basis.cols).collect();
    let triples = problem.solve(iterative, options.max_iter, options.residual_tol)?;
    Ok(LocalCcsdTriplesResult {
        ccsd,
        triples,
        iterative,
        tno_ranks,
    })
}
pub(crate) fn build_problem(
    ctx: &PostHFContext<'_>,
    options: &PostHFOptions,
    cc: &LocalCcsdResult,
    fixed_spaces: Option<&TriplesSpaceMap>,
) -> Result<LocalTriplesProblem, HfError> {
    let hf = ctx.rhf()?;
    let no = cc.occupied.cols;
    let nv = cc.virtuals.cols;
    let fo = cc
        .occupied
        .transpose()
        .dot(&hf.fock_matrix)
        .dot(&cc.occupied);
    let mut fv = Matrix::zeros(nv, nv);
    for a in 0..nv {
        fv[(a, a)] = hf.orbital_energies[hf.n_occ + a];
    }
    let q = cc
        .virtuals
        .dot(&cc.virtuals.transpose())
        .dot(&hf.overlap_matrix);
    let mut densities = Vec::new();
    for i in 0..no {
        for j in 0..=i {
            let t = Matrix::from_vec(
                nv,
                nv,
                cc.t2[((i * no + j) * nv * nv)..((i * no + j + 1) * nv * nv)].to_vec(),
            );
            let tilde = t.scale(2.).sub(&t.transpose());
            densities.push(
                tilde
                    .transpose()
                    .dot(&t)
                    .add(&tilde.dot(&t.transpose()))
                    .scale(if i == j { 0.5 } else { 1. }),
            );
        }
    }
    let mut coeff = Matrix::zeros(cc.occupied.rows, no + nv);
    for mu in 0..coeff.rows {
        for i in 0..no {
            coeff[(mu, i)] = cc.occupied[(mu, i)];
        }
        for a in 0..nv {
            coeff[(mu, no + a)] = cc.virtuals[(mu, a)];
        }
    }
    let fitted = ctx.ints.df_ao()?.transform(&coeff)?;
    let np = (no + nv) * (no + nv + 1) / 2;
    if np as u128 * np as u128 * 8 > 128 * 1024 * 1024 {
        return Err(HfError::Unsupported {
            method: "local triples".into(),
            detail: "global RI triples integrals exceed 128 MiB".into(),
        });
    }
    let mo = crate::mo_integrals::MoEriTensor {
        n_basis: no + nv,
        n_pairs: np,
        data: fitted.b.transpose().dot(&fitted.b).data,
    };
    let cutoff = options.local.t_cut_tno.unwrap_or(options.local.t_cut_pno);
    let mut spaces = fixed_spaces.cloned().unwrap_or_default();
    let mut blocks = Vec::new();
    let mut stored = 0u128;
    for i in 0..no {
        for j in 0..no {
            for k in 0..no {
                let mut key = [i, j, k];
                key.sort_unstable();
                let space = if let Some(space) = spaces.get(&key) {
                    space.clone()
                } else if fixed_spaces.is_some() {
                    return Err(HfError::InvalidInput(
                        "fixed triples anchor is missing an occupied triple".into(),
                    ));
                } else {
                    let mut domain = cc.pair_domain_atoms[pair(i, j)].clone();
                    domain.extend_from_slice(&cc.pair_domain_atoms[pair(j, k)]);
                    domain.extend_from_slice(&cc.pair_domain_atoms[pair(i, k)]);
                    domain.sort_unstable();
                    domain.dedup();
                    let selected = ctx
                        .basis
                        .iter()
                        .enumerate()
                        .filter(|(_, g)| domain.contains(&g.atom_index))
                        .map(|(mu, _)| mu)
                        .collect::<Vec<_>>();
                    let paos = orthonormalize_paos(&q, &hf.overlap_matrix, &selected, 1e-10)?;
                    let common = cc.virtuals.transpose().dot(&hf.overlap_matrix).dot(&paos);
                    // Average the three converged CCSD pair densities in the union domain.
                    let density = densities[pair(i, j)]
                        .add(&densities[pair(j, k)])
                        .add(&densities[pair(i, k)])
                        .scale(1. / 3.);
                    let (occupation, u) = common.transpose().dot(&density).dot(&common).eigh();
                    if occupation.iter().any(|v| *v < -1e-10) {
                        return Err(HfError::Integral("negative TNO density eigenvalue".into()));
                    }
                    let retained = (0..occupation.len())
                        .filter(|&a| cutoff == 0. || occupation[a] > cutoff)
                        .collect::<Vec<_>>();
                    let mut selected = Matrix::zeros(u.rows, retained.len());
                    for (a, &column) in retained.iter().enumerate() {
                        for b in 0..u.rows {
                            selected[(b, a)] = u[(b, column)];
                        }
                    }
                    let basis = common.dot(&selected);
                    let (energies, rotation) = basis.transpose().dot(&fv).dot(&basis).eigh();
                    let space = Arc::new(TriplesVirtualSpace {
                        basis: basis.dot(&rotation),
                        energies,
                    });
                    spaces.insert(key, space.clone());
                    space
                };
                stored += (space.basis.cols as u128).pow(3);
                if stored * 16 * 8 > 256 * 1024 * 1024 {
                    return Err(HfError::Unsupported {
                        method: "local triples".into(),
                        detail: "ordered-triples workspace exceeds 256 MiB".into(),
                    });
                }
                let (connected, singles) = crate::post_hf::cc::ccsd_t::triples_sources(
                    no, nv, &cc.t1, &cc.t2, &mo, i, j, k,
                );
                let overlap = space.basis.transpose();
                let connected = transform_cube(&connected, nv, &overlap)?;
                let singles = transform_cube(&singles, nv, &overlap)?;
                blocks.push(TriplesBlock {
                    i,
                    j,
                    k,
                    space,
                    connected,
                    singles,
                });
            }
        }
    }
    Ok(LocalTriplesProblem { f_occ: fo, blocks })
}
