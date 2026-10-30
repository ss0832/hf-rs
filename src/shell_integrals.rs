// SPDX-License-Identifier: Apache-2.0
//! Shell-quartet ERIs. Primitive Hermite expansions are shared by all Cartesian
//! components of a shell pair; each primitive quartet builds Boys/R only once.
use crate::basis::ContractedGaussian;
use crate::error::HfError;
use crate::integrals::{
    HermiteBuffer, TWO_PI_POW_2_5, compute_hermite_coeffs, coulomb_auxiliary_integrals_total,
    evaluate_eri_contraction_pairs,
};
use rayon::prelude::*;

struct PairPrimitive {
    exponent: f64,
    center: [f64; 3],
    hermite: [Vec<f64>; 3],
    coefficients: Vec<f64>,
}
struct ShellPair<'a> {
    a: &'a [ContractedGaussian],
    b: &'a [ContractedGaussian],
    la: usize,
    lb: usize,
    primitives: Vec<PairPrimitive>,
}
impl<'a> ShellPair<'a> {
    fn new(a: &'a [ContractedGaussian], b: &'a [ContractedGaussian]) -> Self {
        let la = a[0].total_angular_momentum();
        let lb = b[0].total_angular_momentum();
        let stride = (lb + 1) * (la + lb + 1);
        let mut primitives = Vec::new();
        for (ia, pa) in a[0].primitives.iter().enumerate() {
            for (ib, pb) in b[0].primitives.iter().enumerate() {
                let p = pa.exponent + pb.exponent;
                let center = std::array::from_fn(|k| {
                    (pa.exponent * a[0].center[k] + pb.exponent * b[0].center[k]) / p
                });
                let hermite = std::array::from_fn(|axis| {
                    let mut buffer = HermiteBuffer::new();
                    compute_hermite_coeffs(
                        la,
                        lb,
                        pa.exponent,
                        pb.exponent,
                        a[0].center[axis],
                        b[0].center[axis],
                        &mut buffer,
                    );
                    let mut values = vec![0.0; (la + 1) * stride];
                    for i in 0..=la {
                        for j in 0..=lb {
                            for t in 0..=i + j {
                                values[i * stride + j * (la + lb + 1) + t] = buffer.get(i, j, t);
                            }
                        }
                    }
                    values
                });
                let coefficients = a
                    .iter()
                    .flat_map(|ga| {
                        b.iter().map(move |gb| {
                            ga.primitives[ia].coefficient * gb.primitives[ib].coefficient
                        })
                    })
                    .collect();
                primitives.push(PairPrimitive {
                    exponent: p,
                    center,
                    hermite,
                    coefficients,
                });
            }
        }
        Self {
            a,
            b,
            la,
            lb,
            primitives,
        }
    }
    fn expansion<'b>(&self, p: &'b PairPrimitive, axis: usize, a: usize, b: usize) -> &'b [f64] {
        let stride = self.la + self.lb + 1;
        let offset = (self.a[a].l[axis] * (self.lb + 1) + self.b[b].l[axis]) * stride;
        &p.hermite[axis][offset..offset + self.a[a].l[axis] + self.b[b].l[axis] + 1]
    }
    fn len(&self) -> usize {
        self.a.len() * self.b.len()
    }
}

fn quartet(ab: &ShellPair<'_>, cd: &ShellPair<'_>, diagonal_only: bool) -> Vec<f64> {
    let nab = ab.len();
    let ncd = cd.len();
    let mut result = vec![0.0; if diagonal_only { nab } else { nab * ncd }];
    let order = ab.la + ab.lb + cd.la + cd.lb;
    let stride_v = order + 1;
    let stride_u = stride_v * stride_v;
    let stride_t = stride_u * stride_v;
    let mut r = vec![0.0; stride_t * stride_v];
    for p in &ab.primitives {
        for q in &cd.primitives {
            let rho = p.exponent * q.exponent / (p.exponent + q.exponent);
            let delta = std::array::from_fn(|k| p.center[k] - q.center[k]);
            coulomb_auxiliary_integrals_total(order, order, order, order, rho, delta, &mut r);
            let prefactor =
                TWO_PI_POW_2_5 / (p.exponent * q.exponent * (p.exponent + q.exponent).sqrt());
            for ij in 0..nab {
                let i = ij / ab.b.len();
                let j = ij % ab.b.len();
                let eab = std::array::from_fn::<_, 3, _>(|k| ab.expansion(p, k, i, j));
                let lab = std::array::from_fn::<_, 3, _>(|k| ab.a[i].l[k] + ab.b[j].l[k]);
                let (start, end) = if diagonal_only {
                    (ij, ij + 1)
                } else {
                    (0, ncd)
                };
                for kl in start..end {
                    let k = kl / cd.b.len();
                    let l = kl % cd.b.len();
                    let ecd = std::array::from_fn::<_, 3, _>(|axis| cd.expansion(q, axis, k, l));
                    let lcd =
                        std::array::from_fn::<_, 3, _>(|axis| cd.a[k].l[axis] + cd.b[l].l[axis]);
                    let value = evaluate_eri_contraction_pairs(
                        lab[0], lab[1], lab[2], lcd[0], lcd[1], lcd[2], stride_t, stride_u,
                        stride_v, eab[0], eab[1], eab[2], ecd[0], ecd[1], ecd[2], &r, prefactor,
                    );
                    let index = if diagonal_only { ij } else { ij * ncd + kl };
                    result[index] += p.coefficients[ij] * q.coefficients[kl] * value;
                }
            }
        }
    }
    result
}

/// A complete Cartesian shell block in a,b,c,d row-major order.
pub fn compute_shell_quartet(shells: [&[ContractedGaussian]; 4]) -> Result<Vec<f64>, HfError> {
    for shell in shells {
        if crate::ao::shell_ranges(shell)?.len() != 1 {
            return Err(HfError::InvalidInput(
                "one shell per quartet slot is required".into(),
            ));
        }
    }
    Ok(quartet(
        &ShellPair::new(shells[0], shells[1]),
        &ShellPair::new(shells[2], shells[3]),
        false,
    ))
}

/// Packed eightfold tensor with shell Schwarz screening. Completed shell blocks
/// are scattered in batches with an 8 MiB output budget (or a single larger
/// quartet), keeping memory bounded without a Rayon barrier for each tiny block.
pub fn compute_shell_eri_tensor(basis: &[ContractedGaussian]) -> Result<Vec<f64>, HfError> {
    let shells = crate::ao::shell_ranges(basis)?;
    let shell_pairs = (0..shells.len())
        .flat_map(|a| (0..=a).map(move |b| (a, b)))
        .collect::<Vec<_>>();
    let pairs = shell_pairs
        .par_iter()
        .map(|&(a, b)| ShellPair::new(&basis[shells[a].clone()], &basis[shells[b].clone()]))
        .collect::<Vec<_>>();
    let bounds = pairs
        .par_iter()
        .map(|ab| {
            quartet(ab, ab, true)
                .into_iter()
                .fold(0.0_f64, f64::max)
                .sqrt()
        })
        .collect::<Vec<_>>();
    let nao_pairs = basis.len() * (basis.len() + 1) / 2;
    let mut result = vec![0.0; nao_pairs * (nao_pairs + 1) / 2];
    const BATCH_BYTES:usize=8*1024*1024;
    const BATCH_QUARTETS:usize=4096;
    let mut batch = Vec::with_capacity(BATCH_QUARTETS);
    let mut batch_bytes=0;
    let scatter = |batch: &mut Vec<(usize, usize)>, result: &mut [f64]| {
        let blocks = batch
            .par_iter()
            .map(|&(ij, kl)| {
                let block = if bounds[ij] * bounds[kl] > 1e-12 {
                    quartet(&pairs[ij], &pairs[kl], false)
                } else {
                    Vec::new()
                };
                (ij, kl, block)
            })
            .collect::<Vec<_>>();
        for (ij, kl, block) in blocks {
            if block.is_empty() {
                continue;
            }
            let (a, b) = shell_pairs[ij];
            let (c, d) = shell_pairs[kl];
            let ab = &pairs[ij];
            let cd = &pairs[kl];
            for i in 0..ab.a.len() {
                for j in 0..ab.b.len() {
                    let mu = shells[a].start + i;
                    let nu = shells[b].start + j;
                    if mu < nu {
                        continue;
                    }
                    for k in 0..cd.a.len() {
                        for l in 0..cd.b.len() {
                            let lambda = shells[c].start + k;
                            let sigma = shells[d].start + l;
                            if lambda < sigma {
                                continue;
                            }
                            let index = crate::scf::eri_index(mu, nu, lambda, sigma);
                            result[index] =
                                block[(i * ab.b.len() + j) * cd.len() + k * cd.b.len() + l];
                        }
                    }
                }
            }
        }
        batch.clear();
    };
    for ij in 0..pairs.len() {
        for kl in 0..=ij {
            if bounds[ij]*bounds[kl]<=1e-12 {continue;}
            let bytes=8*pairs[ij].len()*pairs[kl].len()+64;
            if !batch.is_empty() && (batch_bytes+bytes>BATCH_BYTES || batch.len()==BATCH_QUARTETS) {
                scatter(&mut batch,&mut result);
                batch_bytes=0;
            }
            batch.push((ij, kl));
            batch_bytes+=bytes;
        }
    }
    if !batch.is_empty() {
        scatter(&mut batch, &mut result);
    }
    Ok(result)
}
