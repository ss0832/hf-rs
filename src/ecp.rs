// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Semi-local effective core potentials (ECPs).
//!
//! `V_ECP = U_L(r) + sum_{l<L} DeltaU_l(r) sum_m |lm><lm|` with every radial function of the form
//! `sum_k d_k r^(n_k - 2) exp(-zeta_k r^2)` (Stuttgart-Koeln / Gaussian 94 convention). The potential is centered
//! on the ECP atom `C`.
//!
//! The matrix elements are evaluated by quadrature that is exact for the angular part up to a controllable
//! tolerance and converged for the radial part:
//! * radial: composite Gauss-Legendre panels with geometrically growing width,
//! * angular: Gauss-Legendre in `cos(theta)` times an equidistant `phi` grid; the degree is chosen per radial
//!   shell from the largest `kappa = 2 alpha r |A - C|` of the significant primitives.
//!
//! The semi-local projections `A_a^{lm}(r) = \int dOmega Y_lm(Omega) a(C + r Omega)` and the local part
//! `\int dOmega a(C + r Omega) b(C + r Omega)` are obtained on the same grids, so any contracted Cartesian
//! function (and any derivative function, a finite linear combination of Cartesian Gaussians) is supported.
//! Gradients follow from derivative functions and translational invariance.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;

use crate::basis::{Atom, ContractedGaussian};
use crate::derivatives::{GeneralBasisDerivative, diff_cg_1st};
use crate::matrix::Matrix;

// ---------------------------------------------------------------------------------------------
// Data model and parser
// ---------------------------------------------------------------------------------------------

/// One radial term `d * r^(power - 2) * exp(-exponent * r^2)` of an ECP channel.
#[derive(Clone, Debug, PartialEq)]
pub struct EcpTerm {
    pub power: i32,
    pub exponent: f64,
    pub coefficient: f64,
}

impl EcpTerm {
    #[inline]
    fn value(&self, r: f64) -> f64 {
        self.coefficient * r.powi(self.power - 2) * (-self.exponent * r * r).exp()
    }
}

/// Effective core potential of one element.
#[derive(Clone, Debug, PartialEq)]
pub struct EcpDef {
    pub symbol: String,
    /// Number of electrons replaced by the potential.
    pub n_core: usize,
    /// Highest angular momentum `L` of the local channel.
    pub l_max: usize,
    /// Local channel `U_L(r)`.
    pub local: Vec<EcpTerm>,
    /// Semi-local channels `DeltaU_l(r) = U_l(r) - U_L(r)`, indexed by `l = 0..l_max` (may contain empty channels).
    pub semilocal: Vec<Vec<EcpTerm>>,
}

impl EcpDef {
    fn radial_exponents(&self) -> impl Iterator<Item = f64> + '_ {
        self.local
            .iter()
            .chain(self.semilocal.iter().flatten())
            .map(|t| t.exponent)
    }

    fn local_value(&self, r: f64) -> f64 {
        self.local.iter().map(|t| t.value(r)).sum()
    }

    fn semilocal_value(&self, l: usize, r: f64) -> f64 {
        self.semilocal
            .get(l)
            .map_or(0.0, |ch| ch.iter().map(|t| t.value(r)).sum())
    }
}

/// Map of upper-case element symbol to its ECP.
pub type EcpMap = HashMap<String, Arc<EcpDef>>;

fn channel_letter_to_l(c: &str) -> Option<usize> {
    match c.to_lowercase().as_str() {
        "s" => Some(0),
        "p" => Some(1),
        "d" => Some(2),
        "f" => Some(3),
        "g" => Some(4),
        "h" => Some(5),
        "i" => Some(6),
        "k" => Some(7),
        _ => None,
    }
}

/// Parses ECP definitions in the Gaussian 94 format written by the Basis Set Exchange:
/// ```text
/// I-ECP     3     28
/// f potential
///   4
/// 2     19.45860900           -21.84204000
/// ...
/// s-f potential
/// ```
/// The blocks `X potential` define the local channel (`X` = `L`), `l-X potential` the semi-local `U_l - U_L`.
pub fn parse_ecp_gaussian94(content: &str) -> Result<EcpMap, String> {
    let lines: Vec<&str> = content
        .lines()
        .map(|line| line.split(['!', '#']).next().unwrap_or("").trim())
        .filter(|line| !line.is_empty() && *line != "****")
        .collect();
    let mut map = EcpMap::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.is_empty() || line.starts_with('!') || line.starts_with('#') || line == "****" {
            i += 1;
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() == 3 && toks[0].to_uppercase().ends_with("-ECP") {
            let symbol = toks[0][..toks[0].len() - 4].to_uppercase();
            let l_max: usize = toks[1]
                .parse()
                .map_err(|_| format!("bad ECP l_max in '{}'", line))?;
            let n_core: usize = toks[2]
                .parse()
                .map_err(|_| format!("bad ECP core count in '{}'", line))?;
            let z = crate::masses::z_from_symbol(&symbol)
                .ok_or_else(|| format!("invalid ECP element {symbol}"))?;
            if l_max > 7 || n_core >= z {
                return Err(format!(
                    "invalid ECP core count or angular momentum for {symbol}"
                ));
            }
            let mut seen_channels = vec![false; l_max + 1];
            let mut def = EcpDef {
                symbol: symbol.clone(),
                n_core,
                l_max,
                local: Vec::new(),
                semilocal: vec![Vec::new(); l_max],
            };
            i += 1;
            // channel blocks until the next ECP header or the end
            while i < lines.len() {
                let head = lines[i];
                let ht: Vec<&str> = head.split_whitespace().collect();
                if head.is_empty() {
                    i += 1;
                    continue;
                }
                if ht.len() == 3 && ht[0].to_uppercase().ends_with("-ECP") {
                    break;
                }
                if ht.len() == 2 && ht[1] == "0" && crate::masses::z_from_symbol(ht[0]).is_some() {
                    break;
                }
                if ht.len() != 2 || !ht[1].eq_ignore_ascii_case("potential") {
                    return Err(format!(
                        "unexpected line in ECP block of {}: '{}'",
                        symbol, head
                    ));
                }
                let (lower, local) = match ht[0].split_once('-') {
                    Some((a, _)) => (channel_letter_to_l(a), false),
                    None => (channel_letter_to_l(ht[0]), true),
                };
                let l = lower.ok_or_else(|| format!("bad ECP channel '{}'", ht[0]))?;
                if l > l_max || seen_channels[l] {
                    return Err(format!("duplicate or invalid ECP channel for {symbol}"));
                }
                seen_channels[l] = true;
                i += 1;
                let count_line = lines
                    .get(i)
                    .ok_or_else(|| format!("truncated ECP channel for {symbol}"))?;
                let count: usize = count_line
                    .parse()
                    .map_err(|_| format!("bad ECP term count '{count_line}'"))?;
                if count > lines.len().saturating_sub(i + 1) {
                    return Err(format!("truncated ECP terms for {symbol}"));
                }
                let mut terms = Vec::with_capacity(count);
                for _ in 0..count {
                    i += 1;
                    let t: Vec<&str> = lines[i].split_whitespace().collect();
                    if t.len() < 3 {
                        return Err(format!("bad ECP term '{}'", lines[i]));
                    }
                    let fl = |s: &str| {
                        let value = s
                            .replace(['D', 'd'], "E")
                            .parse::<f64>()
                            .map_err(|_| format!("bad number '{}'", s))?;
                        if !value.is_finite() {
                            return Err(format!("non-finite ECP number {s}"));
                        }
                        Ok(value)
                    };
                    let term = EcpTerm {
                        power: t[0].parse().map_err(|_| format!("bad power '{}'", t[0]))?,
                        exponent: fl(t[1])?,
                        coefficient: fl(t[2])?,
                    };
                    if term.power < 0 || term.exponent < 0.0 {
                        return Err(format!("invalid ECP radial term for {symbol}"));
                    }
                    terms.push(term);
                }
                i += 1;
                if local {
                    if l != l_max {
                        return Err(format!(
                            "{}: local channel must be l = {}, found {}",
                            symbol, l_max, l
                        ));
                    }
                    def.local = terms;
                } else {
                    if l >= l_max {
                        return Err(format!("{}: semi-local channel l = {} >= l_max", symbol, l));
                    }
                    def.semilocal[l] = terms;
                }
            }
            if !seen_channels[l_max] {
                return Err(format!("missing local ECP channel for {symbol}"));
            }
            if map.insert(symbol.clone(), Arc::new(def)).is_some() {
                return Err(format!("duplicate ECP element {symbol}"));
            }
        } else {
            i += 1;
        }
    }
    Ok(map)
}

/// Read scalar ECP potentials from an official BSE JSON export, with or without orbitals.
pub fn parse_ecp_bse_json(content: &str) -> Result<EcpMap, String> {
    let data: serde_json::Value = serde_json::from_str(content).map_err(|e| e.to_string())?;
    let elements = data["elements"].as_object().ok_or("missing BSE elements")?;
    let mut result = EcpMap::new();
    for (key, element) in elements {
        let Some(potentials) = element.get("ecp_potentials") else {
            continue;
        };
        let channels = potentials
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or("empty BSE ECP channels")?;
        let z = key
            .parse::<usize>()
            .ok()
            .filter(|z| (1..=118).contains(z))
            .or_else(|| crate::masses::z_from_symbol(key))
            .ok_or("invalid BSE ECP element")?;
        let symbol = crate::basis::atomic_number_to_symbol(z).to_uppercase();
        let core = element["ecp_electrons"]
            .as_u64()
            .ok_or("missing BSE ECP electron count")?;
        if core >= z as u64 {
            return Err(format!("invalid ECP core count for {symbol}"));
        }
        let mut parsed = std::collections::BTreeMap::new();
        for channel in channels {
            if channel
                .get("ecp_type")
                .and_then(|v| v.as_str())
                .is_some_and(|v| v != "scalar_ecp")
            {
                return Err("only scalar BSE ECPs are supported".into());
            }
            let momenta = channel["angular_momentum"]
                .as_array()
                .filter(|v| v.len() == 1)
                .ok_or("invalid scalar ECP angular momentum")?;
            let l = momenta[0]
                .as_u64()
                .filter(|l| *l <= 7)
                .ok_or("unsupported ECP angular momentum")? as usize;
            let exponents = channel["gaussian_exponents"]
                .as_array()
                .ok_or("missing ECP exponents")?;
            let powers = channel["r_exponents"]
                .as_array()
                .ok_or("missing ECP powers")?;
            let columns = channel["coefficients"]
                .as_array()
                .filter(|v| v.len() == 1)
                .ok_or("invalid scalar ECP coefficients")?;
            let coefficients = columns[0].as_array().ok_or("invalid ECP coefficients")?;
            if exponents.len() != powers.len() || exponents.len() != coefficients.len() {
                return Err("ECP term dimensions differ".into());
            }
            let number = |value: &serde_json::Value| -> Result<f64, String> {
                let n = if let Some(n) = value.as_f64() {
                    n
                } else {
                    value
                        .as_str()
                        .ok_or("invalid ECP number")?
                        .replace(['D', 'd'], "E")
                        .parse::<f64>()
                        .map_err(|e| e.to_string())?
                };
                if !n.is_finite() {
                    return Err("non-finite ECP number".into());
                }
                Ok(n)
            };
            let mut terms = Vec::with_capacity(exponents.len());
            for i in 0..exponents.len() {
                let exponent = number(&exponents[i])?;
                let power = powers[i]
                    .as_i64()
                    .filter(|p| (0..=i32::MAX as i64).contains(p))
                    .ok_or("invalid ECP power")?;
                if exponent < 0.0 {
                    return Err("negative ECP exponent".into());
                }
                terms.push(EcpTerm {
                    power: power as i32,
                    exponent,
                    coefficient: number(&coefficients[i])?,
                });
            }
            if parsed.insert(l, terms).is_some() {
                return Err("duplicate scalar ECP channel".into());
            }
        }
        let l_max = *parsed
            .keys()
            .next_back()
            .ok_or("missing local ECP channel")?;
        let local = parsed.remove(&l_max).unwrap();
        let mut semilocal = vec![Vec::new(); l_max];
        for (l, terms) in parsed {
            semilocal[l] = terms;
        }
        result.insert(
            symbol.clone(),
            Arc::new(EcpDef {
                symbol,
                n_core: core as usize,
                l_max,
                local,
                semilocal,
            }),
        );
    }
    Ok(result)
}

/// Returns the built-in ECPs that belong to the basis set (`def2-*` for Z = 37-86), or `None` if the basis set
/// is all-electron.
pub fn get_builtin_ecp(basis_name: &str) -> Option<EcpMap> {
    match crate::basis::normalize_basis_name(basis_name).as_str() {
        "lanl2dzecp" => get_builtin_ecp("lanl2dz"),
        "def2ecp" | "def2" | "def2svp" | "def2tzvp" | "def2qzvp" | "svp" | "tzvp" | "qzvp" => Some(
            parse_ecp_gaussian94(crate::basis_data::ECP_DEF2_GBS)
                .expect("embedded def2 ECP data must parse"),
        ),
        _ => crate::builtin_catalog::json(basis_name).and_then(|data| {
            let map = parse_ecp_bse_json(data).expect("embedded BSE ECP data must parse");
            (!map.is_empty()).then_some(map)
        }),
    }
}

/// Attaches ECPs to the matching atoms and reduces their effective nuclear charge by the number of core electrons
/// (`charge = Z - n_core`). Atoms without an entry are left untouched.
pub fn assign_ecps(atoms: &mut [Atom], ecps: &EcpMap) {
    for atom in atoms.iter_mut() {
        if let Some(def) = ecps.get(&atom.symbol.to_uppercase()) {
            atom.ecp = Some(def.clone());
            atom.charge = atom.atomic_number as f64 - def.n_core as f64;
        }
    }
}

/// Attaches the built-in ECPs of the named basis set (no-op for all-electron basis sets).
pub fn assign_builtin_ecps(atoms: &mut [Atom], basis_name: &str) {
    if let Some(map) = get_builtin_ecp(basis_name) {
        assign_ecps(atoms, &map);
    }
}

/// True if any atom carries an ECP.
pub fn has_ecp(atoms: &[Atom]) -> bool {
    atoms.iter().any(|a| a.ecp.is_some())
}

// ---------------------------------------------------------------------------------------------
// Quadrature control
// ---------------------------------------------------------------------------------------------

/// Quadrature parameters of the ECP integrals.
#[derive(Clone, Copy, Debug)]
pub struct EcpQuadrature {
    /// Gauss-Legendre nodes per radial panel.
    pub panel_nodes: usize,
    /// Ratio of consecutive radial panel edges.
    pub panel_ratio: f64,
    /// Outer edge of the first panel (bohr).
    pub first_edge: f64,
    /// Multiplies the estimated angular degree (>= 1.0; larger is more accurate and slower).
    pub angular_scale: f64,
    /// Additive margin of the angular degree.
    pub angular_margin: usize,
}

impl Default for EcpQuadrature {
    fn default() -> Self {
        EcpQuadrature {
            panel_nodes: 16,
            panel_ratio: 1.4,
            first_edge: 0.003,
            angular_scale: 1.6,
            angular_margin: 20,
        }
    }
}

type GlRule = Arc<(Vec<f64>, Vec<f64>)>;

fn gauss_legendre_cached(n: usize) -> GlRule {
    static CACHE: OnceLock<Mutex<HashMap<usize, GlRule>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(r) = cache.lock().unwrap().get(&n) {
        return r.clone();
    }
    let rule = Arc::new(gauss_legendre(n));
    cache.lock().unwrap().insert(n, rule.clone());
    rule
}

/// Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration on P_n).
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    assert!(n >= 1, "Gauss-Legendre rule needs at least one node");
    for i in 0..n.div_ceil(2) {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp;
        loop {
            // Upward recurrence: after the loop p1 = P_n(z), p0 = P_{n-1}(z).
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        let wi = 2.0 / ((1.0 - z * z) * dp * dp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    (x, w)
}

fn radial_grid(ecp: &EcpDef, q: &EcpQuadrature) -> Vec<(f64, f64)> {
    let zeta_min = ecp.radial_exponents().fold(f64::INFINITY, f64::min);
    if !zeta_min.is_finite() {
        return Vec::new();
    }
    let r_max = (48.0 / zeta_min).sqrt();
    let mut edges = vec![0.0, q.first_edge.min(r_max)];
    while *edges.last().unwrap() < r_max {
        let next = (edges.last().unwrap() * q.panel_ratio).min(r_max);
        edges.push(next);
    }
    let rule = gauss_legendre_cached(q.panel_nodes);
    let mut out = Vec::with_capacity((edges.len() - 1) * q.panel_nodes);
    for pair in edges.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        for (t, wt) in rule.0.iter().zip(&rule.1) {
            let r = a + (b - a) * (t + 1.0) * 0.5;
            out.push((r, 0.5 * (b - a) * wt * r * r));
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Real spherical harmonics on the product grid
// ---------------------------------------------------------------------------------------------

/// Weighted real harmonics for a contiguous angular block, without a full-grid copy.
fn real_ylm_on_block(
    l: usize,
    cos_theta: &[f64],
    cos_phi: &[f64],
    sin_phi: &[f64],
    start: usize,
    weights: &[f64],
) -> Matrix {
    let n_phi = cos_phi.len();
    let end = start + weights.len();
    let mut out = Matrix::zeros(weights.len(), 2 * l + 1);
    let fact = |n: usize| -> f64 { (1..=n).fold(1.0, |a, b| a * b as f64) };
    let norms: Vec<f64> = (0..=l)
        .map(|m| {
            ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI) * fact(l - m) / fact(l + m)).sqrt()
        })
        .collect();
    let mut p = vec![0.0; l + 1];
    for it in start / n_phi..end.div_ceil(n_phi) {
        let x = cos_theta[it];
        let s = (1.0 - x * x).sqrt();
        let mut pmm = 1.0;
        // Associated Legendre polynomials without the Condon-Shortley phase.
        for m in 0..=l {
            if m > 0 {
                pmm *= (2 * m - 1) as f64 * s;
            }
            let (mut prev, mut value) = (pmm, pmm);
            if m < l {
                value = x * (2 * m + 1) as f64 * pmm;
                for ll in m + 2..=l {
                    let next = ((2 * ll - 1) as f64 * x * value - (ll + m - 1) as f64 * prev)
                        / (ll - m) as f64;
                    prev = value;
                    value = next;
                }
            }
            p[m] = value * norms[m];
        }
        for idx in start.max(it * n_phi)..end.min((it + 1) * n_phi) {
            let row = idx - start;
            let k = idx % n_phi;
            let w = weights[row];
            out[(row, l)] = p[0] * w;
            let (mut cm, mut sm) = (1.0, 0.0);
            for m in 1..=l {
                (cm, sm) = (
                    cm * cos_phi[k] - sm * sin_phi[k],
                    sm * cos_phi[k] + cm * sin_phi[k],
                );
                let f = std::f64::consts::SQRT_2 * p[m] * w;
                out[(row, l + m)] = f * cm;
                out[(row, l - m)] = f * sm;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Basis functions as flat primitive lists
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct FPrim {
    alpha: f64,
    coef: f64,
    center: [f64; 3],
    l: [usize; 3],
    /// Distance between the primitive center and the ECP center.
    dist: f64,
}

impl FPrim {
    /// Bound the full Cartesian primitive, including the polynomial generated by
    /// high-order center derivatives. Screening only its Gaussian can discard
    /// significant high-angular-momentum derivative terms.
    fn significant_at(&self, r: f64) -> bool {
        if self.coef == 0.0 {
            return false;
        }
        let degree = self.l.iter().sum::<usize>();
        let log_polynomial = if degree == 0 {
            0.0
        } else {
            degree as f64 * (r + self.dist).max(f64::MIN_POSITIVE).ln()
        };
        let log_bound =
            self.coef.abs().ln() + log_polynomial - self.alpha * (r - self.dist).powi(2);
        log_bound >= 1e-17_f64.ln()
    }
}

#[derive(Clone, Debug)]
pub struct EcpFunction {
    prims: Vec<FPrim>,
}

impl EcpFunction {
    /// Flattens a (derivative) basis function into primitives and records the distance to the ECP center `c`.
    pub fn new(f: &GeneralBasisDerivative, c: [f64; 3]) -> Self {
        let mut prims = Vec::new();
        for term in &f.terms {
            let d = ((term.center[0] - c[0]).powi(2)
                + (term.center[1] - c[1]).powi(2)
                + (term.center[2] - c[2]).powi(2))
            .sqrt();
            for p in &term.primitives {
                prims.push(FPrim {
                    alpha: p.exponent,
                    coef: p.coefficient,
                    center: term.center,
                    l: term.l,
                    dist: d,
                });
            }
        }
        EcpFunction { prims }
    }

    pub fn from_contracted(cg: &ContractedGaussian, c: [f64; 3]) -> Self {
        Self::new(&GeneralBasisDerivative::from_single(cg.clone()), c)
    }
}

/// Largest `kappa = 2 alpha r R` and angular momentum among the primitives that matter at radius `r`.
fn shell_extent(funcs: &[EcpFunction], r: f64) -> (f64, usize) {
    let mut kappa: f64 = 0.0;
    let mut lmax = 0;
    for f in funcs {
        for p in &f.prims {
            if !p.significant_at(r) {
                continue;
            }
            kappa = kappa.max(2.0 * p.alpha * r * p.dist);
            lmax = lmax.max(p.l[0] + p.l[1] + p.l[2]);
        }
    }
    (kappa, lmax)
}

/// Function values on the shell: `nf x np`.
fn eval_on_points(funcs: &[EcpFunction], r: f64, pts: &[[f64; 3]]) -> Matrix {
    let np = pts.len();
    let mut out = Matrix::zeros(funcs.len(), np);
    for (fi, f) in funcs.iter().enumerate() {
        let row = &mut out.data[fi * np..(fi + 1) * np];
        for p in &f.prims {
            if !p.significant_at(r) {
                continue;
            }
            for (k, pt) in pts.iter().enumerate() {
                let dx = pt[0] - p.center[0];
                let dy = pt[1] - p.center[1];
                let dz = pt[2] - p.center[2];
                let e = (-p.alpha * (dx * dx + dy * dy + dz * dz)).exp();
                row[k] += p.coef
                    * e
                    * dx.powi(p.l[0] as i32)
                    * dy.powi(p.l[1] as i32)
                    * dz.powi(p.l[2] as i32);
            }
        }
    }
    out
}

fn round_up(n: usize, m: usize) -> usize {
    n.div_ceil(m) * m
}

// Bound the point-dependent workspace per radial worker, independent of angular degree.
// The AO output and the small accumulated projector matrices are separate, O(n_AO^2 + n_AO L^2).
const ANGULAR_WORKSPACE_BYTES: usize = 1024 * 1024;
const MAX_ANGULAR_BLOCK_POINTS: usize = 1024;

fn add_scaled(target: &mut Matrix, source: &Matrix, scale: f64) {
    for (a, b) in target.data.iter_mut().zip(&source.data) {
        *a += scale * b;
    }
}

/// Accumulate tiny SIMD tiles directly into the result. Large faer GEMMs retain
/// an L3-cache-sized packing buffer on every worker; these tiles use its
/// allocation-free small-matrix kernel instead. Views also avoid transpose copies.
fn multiply_add(
    target: &mut Matrix,
    left: &Matrix,
    right: &Matrix,
    transpose_right: bool,
    scale: f64,
) {
    let a = faer::MatRef::from_row_major_slice(&left.data, left.rows, left.cols).as_dyn_stride();
    let b = faer::MatRef::from_row_major_slice(&right.data, right.rows, right.cols).as_dyn_stride();
    let b = if transpose_right { b.transpose() } else { b };
    assert_eq!(a.ncols(), b.nrows());
    assert_eq!((target.rows, target.cols), (a.nrows(), b.ncols()));
    let mut dst =
        faer::MatMut::from_row_major_slice_mut(&mut target.data, target.rows, target.cols);
    // 8 * 8 * 64 = 4096: the small-matrix dispatch limit in faer 0.24.
    for i in (0..a.nrows()).step_by(8) {
        let ni = (a.nrows() - i).min(8);
        for j in (0..b.ncols()).step_by(8) {
            let nj = (b.ncols() - j).min(8);
            for k in (0..a.ncols()).step_by(64) {
                let nk = (a.ncols() - k).min(64);
                faer::linalg::matmul::matmul(
                    dst.as_mut().submatrix_mut(i, j, ni, nj),
                    faer::Accum::Add,
                    a.submatrix(i, k, ni, nk),
                    b.submatrix(k, j, nk, nj),
                    scale,
                    faer::Par::Seq,
                );
            }
        }
    }
}

struct Projection {
    l: usize,
    coefficient: f64,
    left: Matrix,
    right: Option<Matrix>,
}

/// Contribution of one radial shell to the `nl x nr` ECP matrix (radial weight included).
fn shell_matrix(
    left: &[EcpFunction],
    right: &[EcpFunction],
    same: bool,
    c: [f64; 3],
    ecp: &EcpDef,
    r: f64,
    w_r: f64,
    q: &EcpQuadrature,
) -> Matrix {
    let u_loc = ecp.local_value(r);
    let du: Vec<f64> = (0..ecp.l_max).map(|l| ecp.semilocal_value(l, r)).collect();
    let nl = left.len();
    let nr = right.len();
    let mut m = Matrix::zeros(nl, nr);
    if u_loc.abs() < 1e-300 && du.iter().all(|v| v.abs() < 1e-300) {
        return m;
    }
    let (kl, ll) = shell_extent(left, r);
    let (kr, lr) = if same {
        (kl, ll)
    } else {
        shell_extent(right, r)
    };
    let lecp = ecp.l_max;
    let deg_prod = (q.angular_scale * (kl + kr)).ceil() as usize + ll + lr;
    let deg_l = (q.angular_scale * kl).ceil() as usize + ll + lecp;
    let deg_r = (q.angular_scale * kr).ceil() as usize + lr + lecp;
    let degree = deg_prod.max(deg_l).max(deg_r) + q.angular_margin;
    let n_theta = round_up(degree / 2 + 1, 4).min(1024);
    let n_phi = round_up(degree + 1, 4).min(2048);

    let rule = gauss_legendre_cached(n_theta);
    let np = n_theta * n_phi;
    let (sin_phi, cos_phi): (Vec<f64>, Vec<f64>) = (0..n_phi)
        .map(|k| (2.0 * std::f64::consts::PI * k as f64 / n_phi as f64).sin_cos())
        .unzip();
    let mut projections: Vec<Projection> = du
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0.0)
        .map(|(l, &coefficient)| Projection {
            l,
            coefficient,
            left: Matrix::zeros(nl, 2 * l + 1),
            right: if same {
                None
            } else {
                Some(Matrix::zeros(nr, 2 * l + 1))
            },
        })
        .collect();
    // Points/weights, AO values, and one harmonic block. Matrix products use views.
    let values_per_point = 4 + nl + (if same { 0 } else { nr }) + 2 * lecp + 1;
    let block_points =
        (ANGULAR_WORKSPACE_BYTES / (8 * values_per_point)).clamp(1, MAX_ANGULAR_BLOCK_POINTS);
    let mut pts = Vec::with_capacity(block_points);
    let mut wp = Vec::with_capacity(block_points);
    for start in (0..np).step_by(block_points) {
        pts.clear();
        wp.clear();
        for idx in start..(start + block_points).min(np) {
            let it = idx / n_phi;
            let k = idx % n_phi;
            let x = rule.0[it];
            let s = (1.0 - x * x).sqrt();
            pts.push([
                c[0] + r * s * cos_phi[k],
                c[1] + r * s * sin_phi[k],
                c[2] + r * x,
            ]);
            wp.push(rule.1[it] * 2.0 * std::f64::consts::PI / n_phi as f64);
        }
        let mut fl = eval_on_points(left, r, &pts);
        let mut fr = if same {
            None
        } else {
            Some(eval_on_points(right, r, &pts))
        };
        for projection in &mut projections {
            let ymat = real_ylm_on_block(projection.l, &rule.0, &cos_phi, &sin_phi, start, &wp);
            multiply_add(&mut projection.left, &fl, &ymat, false, 1.0);
            if let (Some(ar), Some(fr)) = (&mut projection.right, &fr) {
                multiply_add(ar, fr, &ymat, false, 1.0);
            }
        }
        if u_loc != 0.0 {
            if let Some(fr) = &mut fr {
                for row in fr.data.chunks_exact_mut(wp.len()) {
                    for (value, weight) in row.iter_mut().zip(&wp) {
                        *value *= weight;
                    }
                }
                multiply_add(&mut m, &fl, fr, true, u_loc);
            } else {
                for row in fl.data.chunks_exact_mut(wp.len()) {
                    for (value, weight) in row.iter_mut().zip(&wp) {
                        *value *= weight.sqrt();
                    }
                }
                multiply_add(&mut m, &fl, &fl, true, u_loc);
            }
        }
    }
    // Nonlocal projectors couple different angular blocks. Sum each projection over
    // the entire sphere BEFORE forming A_left A_right^T, retaining all cross terms.
    for projection in &projections {
        let ar = projection.right.as_ref().unwrap_or(&projection.left);
        multiply_add(&mut m, &projection.left, ar, true, projection.coefficient);
    }
    for a in m.data.iter_mut() {
        *a *= w_r;
    }
    m
}

/// ECP matrix `<left_i| V_ECP(C) |right_j>` for an ECP centered at `c`.
pub fn ecp_matrix_with(
    left: &[EcpFunction],
    right: &[EcpFunction],
    same: bool,
    c: [f64; 3],
    ecp: &EcpDef,
    q: &EcpQuadrature,
) -> Matrix {
    let grid = radial_grid(ecp, q);
    let (nl, nr) = (left.len(), right.len());
    grid.par_iter()
        .fold(
            || Matrix::zeros(nl, nr),
            |mut total, &(r, w)| {
                add_scaled(
                    &mut total,
                    &shell_matrix(left, right, same, c, ecp, r, w, q),
                    1.0,
                );
                total
            },
        )
        .reduce(
            || Matrix::zeros(nl, nr),
            |mut a, b| {
                add_scaled(&mut a, &b, 1.0);
                a
            },
        )
}

// ---------------------------------------------------------------------------------------------
// Molecular drivers
// ---------------------------------------------------------------------------------------------

/// Sum of all ECP one-electron matrices of the molecule in the AO basis.
pub fn ecp_one_electron_matrix(atoms: &[Atom], basis: &[ContractedGaussian]) -> Matrix {
    ecp_one_electron_matrix_with(atoms, basis, &EcpQuadrature::default())
}

pub fn ecp_one_electron_matrix_with(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    q: &EcpQuadrature,
) -> Matrix {
    let n = basis.len();
    let mut total = Matrix::zeros(n, n);
    for atom in atoms {
        if let Some(def) = &atom.ecp {
            let funcs: Vec<EcpFunction> = basis
                .iter()
                .map(|cg| EcpFunction::from_contracted(cg, atom.position))
                .collect();
            let m = ecp_matrix_with(&funcs, &funcs, true, atom.position, def, q);
            // symmetrize against round-off
            for i in 0..n {
                for j in 0..n {
                    total[(i, j)] += 0.5 * (m[(i, j)] + m[(j, i)]);
                }
            }
        }
    }
    total
}

/// ECP contribution to the nuclear gradient for the total one-particle density `p` (AO basis):
/// `dE/dR = sum_{mu nu} P_{mu nu} d(V_ECP)_{mu nu}/dR`.
///
/// Derivatives with respect to basis-function centers use derivative functions; the derivative with respect to the
/// ECP center follows from translational invariance.
pub fn ecp_gradient(atoms: &[Atom], basis: &[ContractedGaussian], p: &Matrix) -> Vec<[f64; 3]> {
    ecp_gradient_with(atoms, basis, p, &EcpQuadrature::default())
}

pub fn ecp_gradient_with(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    p: &Matrix,
    q: &EcpQuadrature,
) -> Vec<[f64; 3]> {
    let n = basis.len();
    let mut grad = vec![[0.0; 3]; atoms.len()];
    for (c_idx, atom) in atoms.iter().enumerate() {
        let def = match &atom.ecp {
            Some(d) => d,
            None => continue,
        };
        let c = atom.position;
        let funcs: Vec<EcpFunction> = basis
            .iter()
            .map(|cg| EcpFunction::from_contracted(cg, c))
            .collect();
        for tau in 0..3 {
            let dfuncs: Vec<EcpFunction> = basis
                .iter()
                .map(|cg| EcpFunction::new(&diff_cg_1st(cg, tau), c))
                .collect();
            let m = ecp_matrix_with(&dfuncs, &funcs, false, c, def, q);
            let mut g_atom = vec![0.0; atoms.len()];
            for mu in 0..n {
                let x = basis[mu].atom_index;
                let mut s = 0.0;
                for nu in 0..n {
                    s += 2.0 * p[(mu, nu)] * m[(mu, nu)];
                }
                g_atom[x] += s;
            }
            let mut sum_other = 0.0;
            for (x, g) in g_atom.iter().enumerate() {
                if x != c_idx {
                    grad[x][tau] += g;
                    sum_other += g;
                }
            }
            grad[c_idx][tau] -= sum_other;
        }
    }
    grad
}

/// Analytical mixed nuclear directional derivative of the scalar ECP matrix.
/// Each direction contains one displacement vector per atom. Up to four directions
/// are supported, including repeated directions. No displaced integrals are evaluated.
/// Translational invariance includes motion of the nonlocal projector center:
/// D_v = (v_A-v_C).d_A + (v_B-v_C).d_B for an ECP centered at C.
pub fn ecp_nuclear_derivative_matrix(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    directions: &[Vec<[f64; 3]>],
) -> Result<Matrix, String> {
    ecp_nuclear_derivative_matrix_with(atoms, basis, directions, &EcpQuadrature::default())
}

pub fn ecp_nuclear_derivative_matrix_with(
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    directions: &[Vec<[f64; 3]>],
    q: &EcpQuadrature,
) -> Result<Matrix, String> {
    if directions.len() > 4
        || directions
            .iter()
            .any(|v| v.len() != atoms.len() || v.iter().flatten().any(|x| !x.is_finite()))
    {
        return Err(
            "ECP derivatives require up to four finite directions, each with one vector per atom"
                .into(),
        );
    }
    let n = basis.len();
    let mut total = Matrix::zeros(n, n);
    for (center, atom) in atoms.iter().enumerate() {
        let Some(def) = &atom.ecp else { continue };
        for mask in 0..1usize << directions.len() {
            let functions = |left: bool| -> Vec<EcpFunction> {
                basis
                    .iter()
                    .map(|cg| {
                        let relative: Vec<[f64; 3]> = directions
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| ((mask >> i) & 1 != 0) == left)
                            .map(|(_, v)| {
                                std::array::from_fn(|axis| v[cg.atom_index][axis] - v[center][axis])
                            })
                            .collect();
                        let deriv = crate::derivatives::diff_cg_directions(cg, &relative);
                        EcpFunction::new(&deriv, atom.position)
                    })
                    .collect()
            };
            let left = functions(true);
            let right = functions(false);
            if left.iter().all(|f| f.prims.is_empty()) || right.iter().all(|f| f.prims.is_empty()) {
                continue;
            }
            add_scaled(
                &mut total,
                &ecp_matrix_with(&left, &right, false, atom.position, def, q),
                1.0,
            );
        }
    }
    Ok(total)
}
