// SPDX-License-Identifier: Apache-2.0
//! Shared references and lazily transformed post-HF integrals.
use crate::basis::{Atom, ContractedGaussian};
use crate::df::DfTensor;
use crate::error::HfError;
use crate::matrix::Matrix;
use crate::mo_integrals::{MoEriTensor, OovvIntegrals, transform_eri_ao_to_mo, transform_oovv};
use crate::options::{ActiveSpace, ResolvedActiveSpace};
use crate::scf::{HFResult, UHFResult, compound_index, eri_index};
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefKind {
    Rhf,
    Uhf,
}

pub fn orbital_denominator(ei: f64, ej: f64, ea: f64, eb: f64) -> Result<f64, HfError> {
    let value = ei + ej - ea - eb;
    if !value.is_finite() || value.abs() < 1e-14 {
        Err(HfError::Response(format!(
            "singular orbital denominator: {ei}+{ej}-{ea}-{eb}"
        )))
    } else {
        Ok(value)
    }
}

/// Antisymmetrized Dirac integral in interleaved alpha/beta spin-orbital order.
#[inline]
pub fn spin_antisymmetrized(
    alpha: &MoEriTensor,
    beta: &MoEriTensor,
    cross: Option<&MixedMoEri>,
    p: usize,
    q: usize,
    r: usize,
    s: usize,
) -> f64 {
    let coulomb = |p: usize, r: usize, q: usize, s: usize| match (p % 2, q % 2) {
        (0, 0) => alpha.get(p / 2, r / 2, q / 2, s / 2),
        (1, 1) => beta.get(p / 2, r / 2, q / 2, s / 2),
        (0, 1) => cross.map_or_else(
            || alpha.get(p / 2, r / 2, q / 2, s / 2),
            |x| x.get(p / 2, r / 2, q / 2, s / 2),
        ),
        (1, 0) => cross.map_or_else(
            || alpha.get(p / 2, r / 2, q / 2, s / 2),
            |x| x.get(q / 2, s / 2, p / 2, r / 2),
        ),
        _ => unreachable!(),
    };
    let direct = if p % 2 == r % 2 && q % 2 == s % 2 {
        coulomb(p, r, q, s)
    } else {
        0.0
    };
    let exchange = if p % 2 == s % 2 && q % 2 == r % 2 {
        coulomb(p, s, q, r)
    } else {
        0.0
    };
    direct - exchange
}

pub struct Mp2Guess {
    pub amplitudes: Vec<f64>,
    pub denominators: Vec<f64>,
}
pub fn spin_mp2_guess(
    occupied: &[f64],
    virtuals: &[f64],
    eri: impl Fn(usize, usize, usize, usize) -> f64,
) -> Result<Mp2Guess, HfError> {
    let no = occupied.len();
    let nv = virtuals.len();
    let mut amplitudes = vec![0.0; no * no * nv * nv];
    let mut denominators = amplitudes.clone();
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let index = ((i * no + j) * nv + a) * nv + b;
                    let gap =
                        orbital_denominator(occupied[i], occupied[j], virtuals[a], virtuals[b])?;
                    let value = eri(i, j, no + a, no + b);
                    if !value.is_finite() {
                        return Err(HfError::Integral("non-finite MP2 guess integral".into()));
                    }
                    denominators[index] = gap;
                    amplitudes[index] = value / gap;
                }
            }
        }
    }
    Ok(Mp2Guess {
        amplitudes,
        denominators,
    })
}
#[derive(Clone, Copy, Debug)]
pub enum Reference<'a> {
    Rhf(&'a HFResult),
    Uhf(&'a UHFResult),
}
impl<'a> Reference<'a> {
    pub fn kind(self) -> RefKind {
        match self {
            Self::Rhf(_) => RefKind::Rhf,
            Self::Uhf(_) => RefKind::Uhf,
        }
    }
    pub fn energy(self) -> f64 {
        match self {
            Self::Rhf(hf) => hf.total_energy,
            Self::Uhf(hf) => hf.total_energy,
        }
    }
    pub fn coefficients(self, spin: usize) -> &'a Matrix {
        match self {
            Self::Rhf(hf) => &hf.mo_coefficients,
            Self::Uhf(hf) => {
                if spin == 0 {
                    &hf.mo_coefficients_alpha
                } else {
                    &hf.mo_coefficients_beta
                }
            }
        }
    }
    pub fn energies(self, spin: usize) -> &'a [f64] {
        match self {
            Self::Rhf(hf) => &hf.orbital_energies,
            Self::Uhf(hf) => {
                if spin == 0 {
                    &hf.orbital_energies_alpha
                } else {
                    &hf.orbital_energies_beta
                }
            }
        }
    }
    pub fn occupations(self) -> (usize, usize) {
        match self {
            Self::Rhf(hf) => (hf.n_occ, hf.n_occ),
            Self::Uhf(hf) => (hf.n_alpha, hf.n_beta),
        }
    }
    pub fn eri(self) -> &'a [f64] {
        match self {
            Self::Rhf(hf) => &hf.eri,
            Self::Uhf(hf) => &hf.eri,
        }
    }
    pub fn core(self) -> &'a Matrix {
        match self {
            Self::Rhf(hf) => &hf.core_hamiltonian,
            Self::Uhf(hf) => &hf.core_hamiltonian,
        }
    }
    pub fn nuclear_energy(self) -> f64 {
        match self {
            Self::Rhf(hf) => hf.nuclear_repulsion_energy,
            Self::Uhf(hf) => hf.nuclear_repulsion_energy,
        }
    }
    pub fn converged(self) -> bool {
        match self {
            Self::Rhf(hf) => hf.converged,
            Self::Uhf(hf) => hf.converged,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AuxBasis {
    pub name: Option<String>,
    pub functions: Vec<ContractedGaussian>,
}

impl AuxBasis {
    pub fn from_builtin(atoms:&[Atom],name:&str)->Result<Self,HfError> {
        let definitions=crate::basis::get_builtin_aux_basis(name).ok_or_else(||HfError::InvalidInput(format!("unknown auxiliary basis {name}")))?;
        if atoms.is_empty() || atoms.iter().any(|atom|!definitions.contains_key(&atom.symbol.to_uppercase())) {
            return Err(HfError::InvalidInput(format!("auxiliary basis {name} does not cover the molecular elements")));
        }
        Ok(Self {name:Some(name.into()),functions:crate::basis::build_basis(atoms,&definitions)})
    }
}

#[derive(Clone, Debug)]
pub struct MixedMoEri {
    pub n_alpha: usize,
    pub n_beta: usize,
    pub data: Matrix,
}
impl MixedMoEri {
    #[inline]
    pub fn get(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        self.data[(compound_index(r, s), compound_index(p, q))]
    }
}

/// GEMM mixed-spin transform; rectangular alpha/beta coefficient matrices are supported.
pub fn transform_mixed_eri(ca: &Matrix, cb: &Matrix, eri: &[f64]) -> Result<MixedMoEri, HfError> {
    if ca.rows != cb.rows {
        return Err(HfError::InvalidInput(
            "mixed MO transform AO dimension mismatch".into(),
        ));
    }
    let nao = ca.rows;
    let np = nao * (nao + 1) / 2;
    if eri.len() != np * (np + 1) / 2 {
        return Err(HfError::InvalidInput(
            "packed AO ERI dimension mismatch".into(),
        ));
    }
    let pairs = (0..nao)
        .flat_map(|p| (0..=p).map(move |q| (p, q)))
        .collect::<Vec<_>>();
    let na = ca.cols * (ca.cols + 1) / 2;
    let nb = cb.cols * (cb.cols + 1) / 2;
    let mut half = Matrix::zeros(np, nb);
    half.data
        .par_chunks_mut(nb.max(1))
        .enumerate()
        .for_each(|(pq, row)| {
            let (p, q) = pairs[pq];
            let mut ao = Matrix::zeros(nao, nao);
            for r in 0..nao {
                for s in 0..nao {
                    ao[(r, s)] = eri[eri_index(p, q, r, s)];
                }
            }
            let transformed = cb.transpose().dot(&ao).dot(cb);
            for r in 0..cb.cols {
                for s in 0..=r {
                    row[compound_index(r, s)] = transformed[(r, s)];
                }
            }
        });
    let mut data = Matrix::zeros(nb, na);
    data.data
        .par_chunks_mut(na.max(1))
        .enumerate()
        .for_each(|(rs, row)| {
            let mut ao = Matrix::zeros(nao, nao);
            for (pq, &(p, q)) in pairs.iter().enumerate() {
                ao[(p, q)] = half[(pq, rs)];
                ao[(q, p)] = half[(pq, rs)];
            }
            let transformed = ca.transpose().dot(&ao).dot(ca);
            for p in 0..ca.cols {
                for q in 0..=p {
                    row[compound_index(p, q)] = transformed[(p, q)];
                }
            }
        });
    Ok(MixedMoEri {
        n_alpha: ca.cols,
        n_beta: cb.cols,
        data,
    })
}

/// Spin-orbital antisymmetrized integrals represented by spatial spin blocks,
/// avoiding the sixteenfold allocation of a dense spin-orbital fourth-rank tensor.
#[derive(Clone, Debug)]
pub struct SpinEriBlocks {
    pub alpha: Arc<MoEriTensor>,
    pub beta: Arc<MoEriTensor>,
    pub cross: Option<Arc<MixedMoEri>>,
}
impl SpinEriBlocks {
    #[inline]
    pub fn get(&self, p: usize, q: usize, r: usize, s: usize) -> f64 {
        spin_antisymmetrized(&self.alpha, &self.beta, self.cross.as_deref(), p, q, r, s)
    }
}

#[derive(Clone, Debug)]
pub struct ActiveIntegrals {
    pub space: ResolvedActiveSpace,
    pub h: Matrix,
    pub eri: MoEriTensor,
    pub core_energy: f64,
}

/// Shared inactive-core contraction for CASCI, CASSCF and correlated active-space methods.
pub fn active_integrals_from_mo(
    hm: &Matrix,
    get: impl Fn(usize, usize, usize, usize) -> f64,
    space: ResolvedActiveSpace,
    nuclear_energy: f64,
) -> Result<ActiveIntegrals, HfError> {
    let n = space.active.len();
    let np = n * (n + 1) / 2;
    if hm.rows != hm.cols || space.permutation().iter().any(|&i| i >= hm.rows) {
        return Err(HfError::InvalidInput(
            "active integrals: inconsistent MO dimensions".into(),
        ));
    }
    let mut core_energy = nuclear_energy;
    for &i in &space.inactive {
        core_energy += 2.0 * hm[(i, i)];
        for &j in &space.inactive {
            core_energy += 2.0 * get(i, i, j, j) - get(i, j, j, i);
        }
    }
    let mut h = Matrix::zeros(n, n);
    let mut eri = MoEriTensor {
        n_basis: n,
        n_pairs: np,
        data: vec![0.0; np * np],
    };
    for (p, &pp) in space.active.iter().enumerate() {
        for (q, &qq) in space.active.iter().enumerate() {
            h[(p, q)] = hm[(pp, qq)]
                + space
                    .inactive
                    .iter()
                    .map(|&i| 2.0 * get(pp, qq, i, i) - get(pp, i, i, qq))
                    .sum::<f64>();
            if q > p {
                continue;
            }
            for (r, &rr) in space.active.iter().enumerate() {
                for (ss, &s) in space.active.iter().enumerate().take(r + 1) {
                    eri.data[compound_index(r, ss) * np + compound_index(p, q)] =
                        get(pp, qq, rr, s);
                }
            }
        }
    }
    if !core_energy.is_finite() || h.data.iter().chain(&eri.data).any(|x| !x.is_finite()) {
        return Err(HfError::Integral("non-finite active integrals".into()));
    }
    Ok(ActiveIntegrals {
        space,
        h,
        eri,
        core_energy,
    })
}

type Lazy<T> = OnceLock<Result<Arc<T>, HfError>>;
fn cached<T>(
    cell: &Lazy<T>,
    build: impl FnOnce() -> Result<T, HfError>,
) -> Result<&Arc<T>, HfError> {
    match cell.get_or_init(|| build().map(Arc::new)) {
        Ok(value) => Ok(value),
        Err(error) => Err(error.clone()),
    }
}
#[derive(Default, Debug)]
struct ActiveCache {
    cells: HashMap<ActiveSpace, Arc<Lazy<ActiveIntegrals>>>,
    order: VecDeque<ActiveSpace>,
    bytes: usize,
}
impl ActiveCache {
    fn bytes_for(space: &ActiveSpace) -> usize {
        let n = space.n_orb;
        let np = n * (n + 1) / 2;
        8 * (np * np + n * n) + 256 + 16 * n
    }
    fn cell(&mut self, space: &ActiveSpace) -> Arc<Lazy<ActiveIntegrals>> {
        if let Some(cell) = self.cells.get(space) {
            self.order.retain(|x| x != space);
            self.order.push_back(space.clone());
            return cell.clone();
        }
        let cell = Arc::new(OnceLock::new());
        let bytes = Self::bytes_for(space);
        const BUDGET: usize = 64 * 1024 * 1024;
        if bytes > BUDGET {
            return cell;
        }
        while self.bytes + bytes > BUDGET || self.cells.len() >= 128 {
            if let Some(old) = self.order.pop_front() {
                self.cells.remove(&old);
                self.bytes -= Self::bytes_for(&old);
            } else {
                break;
            }
        }
        self.cells.insert(space.clone(), cell.clone());
        self.order.push_back(space.clone());
        self.bytes += bytes;
        cell
    }
}
type ActiveCells = Mutex<ActiveCache>;

#[derive(Debug)]
pub struct MoIntegralCache<'a> {
    reference: Reference<'a>,
    aux: Option<&'a AuxBasis>,
    basis: &'a [ContractedGaussian],
    alpha: Lazy<MoEriTensor>,
    beta: Lazy<MoEriTensor>,
    cross: Lazy<MixedMoEri>,
    spin: Lazy<SpinEriBlocks>,
    oovv: Lazy<OovvIntegrals>,
    df_ao: Lazy<DfTensor>,
    df_alpha: Lazy<DfTensor>,
    df_beta: Lazy<DfTensor>,
    active: ActiveCells,
}
impl<'a> MoIntegralCache<'a> {
    fn new(
        reference: Reference<'a>,
        basis: &'a [ContractedGaussian],
        aux: Option<&'a AuxBasis>,
    ) -> Self {
        Self {
            reference,
            aux,
            basis,
            alpha: OnceLock::new(),
            beta: OnceLock::new(),
            cross: OnceLock::new(),
            spin: OnceLock::new(),
            oovv: OnceLock::new(),
            df_ao: OnceLock::new(),
            df_alpha: OnceLock::new(),
            df_beta: OnceLock::new(),
            active: Mutex::new(ActiveCache::default()),
        }
    }
    pub fn full(&self) -> Result<&Arc<MoEriTensor>, HfError> {
        cached(&self.alpha, || {
            Ok(transform_eri_ao_to_mo(
                self.reference.coefficients(0),
                self.reference.eri(),
            ))
        })
    }
    pub fn full_beta(&self) -> Result<&Arc<MoEriTensor>, HfError> {
        if self.reference.kind() == RefKind::Rhf {
            return self.full();
        }
        cached(&self.beta, || {
            Ok(transform_eri_ao_to_mo(
                self.reference.coefficients(1),
                self.reference.eri(),
            ))
        })
    }
    pub fn cross(&self) -> Result<&Arc<MixedMoEri>, HfError> {
        cached(&self.cross, || {
            transform_mixed_eri(
                self.reference.coefficients(0),
                self.reference.coefficients(1),
                self.reference.eri(),
            )
        })
    }
    pub fn spin_blocks(&self) -> Result<&Arc<SpinEriBlocks>, HfError> {
        cached(&self.spin, || {
            Ok(SpinEriBlocks {
                alpha: self.full()?.clone(),
                beta: self.full_beta()?.clone(),
                cross: if self.reference.kind() == RefKind::Uhf {
                    Some(self.cross()?.clone())
                } else {
                    None
                },
            })
        })
    }
    pub fn oovv(&self) -> Result<&Arc<OovvIntegrals>, HfError> {
        if self.reference.kind() != RefKind::Rhf {
            return Err(HfError::InvalidInput(
                "restricted OOVV requested for UHF".into(),
            ));
        }
        cached(&self.oovv, || {
            let c = self.reference.coefficients(0);
            let no = self.reference.occupations().0;
            let occupied = c.leading_columns(no);
            let mut virtuals = Matrix::zeros(c.rows, c.cols - no);
            for i in 0..c.rows {
                for a in no..c.cols {
                    virtuals[(i, a - no)] = c[(i, a)];
                }
            }
            if no == 0 || no == c.cols {
                return Ok(OovvIntegrals {
                    n_occ: no,
                    n_virt: c.cols - no,
                    data: Vec::new(),
                });
            }
            Ok(transform_oovv(&occupied, &virtuals, self.reference.eri()))
        })
    }
    pub fn df_ao(&self) -> Result<&Arc<DfTensor>, HfError> {
        cached(&self.df_ao, || {
            DfTensor::build(
                self.basis,
                &self
                    .aux
                    .ok_or_else(|| HfError::InvalidInput("an auxiliary basis is required".into()))?
                    .functions,
                1e-10,
            )
        })
    }
    pub fn df(&self, spin: usize) -> Result<&Arc<DfTensor>, HfError> {
        let cell = if spin == 0 || self.reference.kind() == RefKind::Rhf {
            &self.df_alpha
        } else {
            &self.df_beta
        };
        cached(cell, || {
            self.df_ao()?.transform(self.reference.coefficients(spin))
        })
    }
    pub fn active(&self, space: &ActiveSpace) -> Result<Arc<ActiveIntegrals>, HfError> {
        let c = self.reference.coefficients(0);
        let (na, nb) = self.reference.occupations();
        let resolved = space.resolve(c.cols, na, nb)?;
        let cell = {
            let mut cells = self
                .active
                .lock()
                .map_err(|_| HfError::Integral("active integral cache lock poisoned".into()))?;
            cells.cell(space)
        };
        Ok(cached(&cell, || {
            let c = self.reference.coefficients(0);
            let hm = c.transpose().dot(self.reference.core()).dot(c);
            let mo = self.full()?;
            active_integrals_from_mo(
                &hm,
                |p, q, r, s| mo.get(p, q, r, s),
                resolved,
                self.reference.nuclear_energy(),
            )
        })?
        .clone())
    }
}

pub struct PostHFContext<'a> {
    pub reference: Reference<'a>,
    pub atoms: &'a [Atom],
    pub basis: &'a [ContractedGaussian],
    pub aux: Option<&'a AuxBasis>,
    pub ints: MoIntegralCache<'a>,
}
impl<'a> PostHFContext<'a> {
    pub fn new(
        reference: Reference<'a>,
        atoms: &'a [Atom],
        basis: &'a [ContractedGaussian],
        aux: Option<&'a AuxBasis>,
    ) -> Result<Self, HfError> {
        if !reference.converged() {
            return Err(HfError::Scf("post-HF reference did not converge".into()));
        }
        let c = reference.coefficients(0);
        let (na, nb) = reference.occupations();
        let np = basis.len() * (basis.len() + 1) / 2;
        if atoms.is_empty()
            || basis.is_empty()
            || basis.iter().any(|g| g.atom_index >= atoms.len())
            || c.rows != basis.len()
            || c.cols == 0
            || reference.coefficients(1).rows != c.rows
            || reference.coefficients(1).cols != c.cols
            || reference.energies(0).len() != c.cols
            || reference.energies(1).len() != c.cols
            || na > c.cols
            || nb > c.cols
            || reference.eri().len() != np * (np + 1) / 2
        {
            return Err(HfError::InvalidInput(
                "inconsistent post-HF reference dimensions".into(),
            ));
        }
        Ok(Self {
            reference,
            atoms,
            basis,
            aux,
            ints: MoIntegralCache::new(reference, basis, aux),
        })
    }
    pub fn from_rhf(
        hf: &'a HFResult,
        atoms: &'a [Atom],
        basis: &'a [ContractedGaussian],
        aux: Option<&'a AuxBasis>,
    ) -> Result<Self, HfError> {
        Self::new(Reference::Rhf(hf), atoms, basis, aux)
    }
    pub fn from_uhf(
        hf: &'a UHFResult,
        atoms: &'a [Atom],
        basis: &'a [ContractedGaussian],
        aux: Option<&'a AuxBasis>,
    ) -> Result<Self, HfError> {
        Self::new(Reference::Uhf(hf), atoms, basis, aux)
    }
    pub fn rhf(&self) -> Result<&'a HFResult, HfError> {
        match self.reference {
            Reference::Rhf(hf) => Ok(hf),
            _ => Err(HfError::InvalidInput("RHF reference required".into())),
        }
    }
    pub fn uhf(&self) -> Result<&'a UHFResult, HfError> {
        match self.reference {
            Reference::Uhf(hf) => Ok(hf),
            _ => Err(HfError::InvalidInput("UHF reference required".into())),
        }
    }
}
