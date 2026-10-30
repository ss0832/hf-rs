// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! Pair-projected RI-CCSD. True CC residuals are projected during every iteration;
//! occupied Fock coupling is retained and weak-pair amplitudes remain at LMP2.
//! This implementation retains global RI integrals in the CC contractions.
use super::local_mp2::{BuiltPairProblem, build_pair_problem};
use crate::context::PostHFContext;
use crate::error::HfError;
use crate::matrix::Matrix;
use crate::options::PostHFOptions;
use crate::output::PostHFOutput;
use crate::post_hf::cc::ccsd::{CCSDOptions, CcsdProjection, solve_general_spin_orbital_ccsd};
use crate::post_hf::{PostHFMethod, PostHFResult};
use crate::registry::{ContextMethod, MethodId, MethodMeta};

#[derive(Clone, Debug)]
pub struct LocalCcsdResult {
    pub e_scf: f64,
    pub e_ccsd: f64,
    pub e_lmp2: f64,
    pub delta_pno: f64,
    pub iterations: usize,
    pub occupied: Matrix,
    pub virtuals: Matrix,
    /// Spatial amplitudes in the localized occupied/common virtual bases.
    pub t1: Vec<f64>,
    pub t2: Vec<f64>,
    pub pair_ranks: Vec<usize>,
    pub pair_domain_atoms: Vec<Vec<usize>>,
}
impl PostHFResult for LocalCcsdResult {
    fn method_name(&self) -> &'static str {
        "DLPNO-CCSD"
    }
    fn reference_energy(&self) -> f64 {
        self.e_scf
    }
    fn correlation_energy(&self) -> f64 {
        self.e_ccsd + self.delta_pno
    }
    fn to_output(&self) -> Result<PostHFOutput, HfError> {
        let mut out = PostHFOutput::new(self.method_name(), self.e_scf, self.total_energy())?;
        out.components
            .insert("E_CCSD_pair_space".into(), self.e_ccsd);
        out.components.insert("E_delta_PNO".into(), self.delta_pno);
        Ok(out)
    }
}
pub struct LocalCcsd<'a> {
    ctx: &'a PostHFContext<'a>,
    options: PostHFOptions,
}
impl MethodMeta for LocalCcsd<'_> {
    const METHOD: MethodId = MethodId::DlpnoCcsd;
}
impl<'a> ContextMethod<'a> for LocalCcsd<'a> {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError> {
        // The same RHF, frozen-partition, auxiliary and root checks apply.
        super::local_mp2::LocalMp2::from_context(ctx, options)?;
        Ok(Self {
            ctx,
            options: options.clone(),
        })
    }
}
impl PostHFMethod for LocalCcsd<'_> {
    type Output = LocalCcsdResult;
    fn compute(&self) -> Result<Self::Output, String> {
        self.compute_checked().map_err(|e| e.to_string())
    }
}
fn index(i: usize, j: usize) -> usize {
    let hi = i.max(j);
    hi * (hi + 1) / 2 + i.min(j)
}
impl LocalCcsd<'_> {
    pub fn compute_checked(&self) -> Result<LocalCcsdResult, HfError> {
        let built = build_pair_problem(self.ctx, &self.options)?;
        compute_with_pair_problem(self.ctx, &self.options, built)
    }
}

/// Shared residual solver for adaptive and fixed transported pair spaces.
pub(crate) fn compute_with_pair_problem(
    ctx: &PostHFContext<'_>,
    options: &PostHFOptions,
    built: BuiltPairProblem,
) -> Result<LocalCcsdResult, HfError> {
    let lmp2 = built
        .problem
        .solve(options.max_iter, options.residual_tol)?;
    let no = built.occupied.cols;
    let nv = built.virtuals.cols;
    let nso = 2 * no;
    let nsv = 2 * nv;
    if no == 0 || nv == 0 {
        return Ok(LocalCcsdResult {
            e_scf: ctx.reference.energy(),
            e_ccsd: 0.,
            e_lmp2: 0.,
            delta_pno: 0.,
            iterations: 0,
            occupied: built.occupied,
            virtuals: built.virtuals,
            t1: vec![0.; no * nv],
            t2: vec![0.; no * no * nv * nv],
            pair_ranks: built.problem.pairs.iter().map(|p| p.basis.cols).collect(),
            pair_domain_atoms: built.domain_atoms,
        });
    }
    let mut coeff = Matrix::zeros(built.occupied.rows, no + nv);
    for mu in 0..coeff.rows {
        for i in 0..no {
            coeff[(mu, i)] = built.occupied[(mu, i)];
        }
        for a in 0..nv {
            coeff[(mu, no + a)] = built.virtuals[(mu, a)];
        }
    }
    let fitted = ctx.ints.df_ao()?.transform(&coeff)?;
    let np = (no + nv) * (no + nv + 1) / 2;
    if (np as u128) * (np as u128) * 8 > 128 * 1024 * 1024 {
        return Err(HfError::Unsupported {
            method: "DLPNO-CCSD".into(),
            detail: "global RI CC integral workspace exceeds 128 MiB".into(),
        });
    }
    let mo = crate::mo_integrals::MoEriTensor {
        n_basis: no + nv,
        n_pairs: np,
        data: fitted.b.transpose().dot(&fitted.b).data,
    };
    let anti = |p, q, r, s| crate::context::spin_antisymmetrized(&mo, &mo, None, p, q, r, s);
    let mut fo = Matrix::zeros(nso, nso);
    let mut fv = Matrix::zeros(nsv, nsv);
    let hf = ctx.rhf()?;
    for i in 0..nso {
        for j in 0..nso {
            if i % 2 == j % 2 {
                fo[(i, j)] = built.problem.f_occ[(i / 2, j / 2)];
            }
        }
    }
    for a in 0..nsv {
        fv[(a, a)] = hf.orbital_energies[hf.n_occ + a / 2];
    }
    let eo = (0..nso).map(|i| fo[(i, i)]).collect::<Vec<_>>();
    let ev = (0..nsv).map(|a| fv[(a, a)]).collect::<Vec<_>>();
    let projectors = built
        .problem
        .pairs
        .iter()
        .map(|p| {
            if p.strong {
                p.basis.dot(&p.basis.transpose())
            } else {
                Matrix::zeros(nv, nv)
            }
        })
        .collect::<Vec<_>>();
    let project = |t1: &mut [f64], t2: &mut [f64]| {
        for i in 0..nso {
            let p = &projectors[index(i / 2, i / 2)];
            for spin in 0..2 {
                let v = (0..nv)
                    .map(|a| t1[i * nsv + 2 * a + spin])
                    .collect::<Vec<_>>();
                for a in 0..nv {
                    t1[i * nsv + 2 * a + spin] = (0..nv).map(|b| p[(a, b)] * v[b]).sum();
                }
            }
        }
        for i in 0..nso {
            for j in 0..nso {
                let p = &projectors[index(i / 2, j / 2)];
                for sa in 0..2 {
                    for sb in 0..2 {
                        let mut m = Matrix::zeros(nv, nv);
                        for a in 0..nv {
                            for b in 0..nv {
                                m[(a, b)] =
                                    t2[((i * nso + j) * nsv + 2 * a + sa) * nsv + 2 * b + sb];
                            }
                        }
                        let m = p.dot(&m).dot(p);
                        for a in 0..nv {
                            for b in 0..nv {
                                t2[((i * nso + j) * nsv + 2 * a + sa) * nsv + 2 * b + sb] =
                                    m[(a, b)];
                            }
                        }
                    }
                }
            }
        }
    };
    let initial1 = vec![0.; nso * nsv];
    let mut initial2 = vec![0.; nso * nso * nsv * nsv];
    for i in 0..no {
        for j in 0..no {
            let pi = index(i, j);
            let p = &built.problem.pairs[pi];
            let pair = p.basis.dot(&lmp2.amplitudes[pi]).dot(&p.basis.transpose());
            let pair = if i >= j { pair } else { pair.transpose() };
            for si in 0..2 {
                for sj in 0..2 {
                    for a in 0..nv {
                        for b in 0..nv {
                            for sa in 0..2 {
                                for sb in 0..2 {
                                    initial2[((((2 * i + si) * nso + 2 * j + sj) * nsv
                                        + 2 * a
                                        + sa)
                                        * nsv)
                                        + 2 * b
                                        + sb] = if si == sa && sj == sb {
                                        pair[(a, b)]
                                    } else {
                                        0.
                                    } - if si == sb && sj == sa {
                                        pair[(b, a)]
                                    } else {
                                        0.
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let cc_options = CCSDOptions {
        max_iterations: options.max_iter,
        energy_tol: options.energy_tol,
        residual_tol: options.residual_tol,
        diis_subspace: 8,
    };
    let (energy, iterations, converged, t1, t2) = solve_general_spin_orbital_ccsd(
        nso,
        nsv,
        &eo,
        &ev,
        anti,
        &cc_options,
        Some((&fo, &fv)),
        Some(CcsdProjection {
            project: &project,
            initial: Some((&initial1, &initial2)),
        }),
    )
    .map_err(|detail| HfError::PostHf {
        method: "DLPNO-CCSD".into(),
        detail,
    })?;
    if !converged {
        return Err(HfError::PostHf {
            method: "DLPNO-CCSD".into(),
            detail: format!("projected residual did not converge in {iterations} iterations"),
        });
    }
    let mut spatial1 = vec![0.; no * nv];
    let mut spatial2 = vec![0.; no * no * nv * nv];
    for i in 0..no {
        for a in 0..nv {
            spatial1[i * nv + a] = t1[2 * i * nsv + 2 * a];
        }
    }
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    spatial2[((i * no + j) * nv + a) * nv + b] =
                        t2[((2 * i * nso + 2 * j + 1) * nsv + 2 * a) * nsv + 2 * b + 1];
                }
            }
        }
    }
    Ok(LocalCcsdResult {
        e_scf: hf.total_energy,
        e_ccsd: energy,
        e_lmp2: lmp2.correlation_energy(),
        delta_pno: lmp2.delta_pno,
        iterations,
        occupied: built.occupied,
        virtuals: built.virtuals,
        t1: spatial1,
        t2: spatial2,
        pair_ranks: built.problem.pairs.iter().map(|p| p.basis.cols).collect(),
        pair_domain_atoms: built.domain_atoms,
    })
}
