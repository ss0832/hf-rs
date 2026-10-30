// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Basis set data structures and parsers.
//! Supports BasisSetExchange (BSE) JSON format, Gaussian 94 format (.gbs), custom definitions, and built-in basis sets.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde::Deserialize;

#[derive(Clone, Debug, PartialEq)]
pub struct Atom {
    pub symbol: String,
    pub atomic_number: usize,
    /// Nuclear charge (usually equal to atomic_number for all-electron calculations)
    pub charge: f64,
    /// Coordinates in atomic units (Bohr)
    pub position: [f64; 3],
    /// Effective core potential replacing `atomic_number - charge` core electrons (see `crate::ecp`)
    pub ecp: Option<std::sync::Arc<crate::ecp::EcpDef>>,
}

impl Atom {
    pub fn new(symbol: &str, pos_bohr: [f64; 3]) -> Self {
        let z = symbol_to_atomic_number(symbol);
        Atom {
            symbol: symbol.to_string(),
            atomic_number: z,
            charge: z as f64,
            position: pos_bohr,
            ecp: None,
        }
    }

    pub fn from_angstrom(symbol: &str, pos_ang: [f64; 3]) -> Self {
        const ANG_TO_BOHR: f64 = 1.8897261246257702;
        let pos_bohr = [
            pos_ang[0] * ANG_TO_BOHR,
            pos_ang[1] * ANG_TO_BOHR,
            pos_ang[2] * ANG_TO_BOHR,
        ];
        Self::new(symbol, pos_bohr)
    }
}

pub const PERIODIC_TABLE: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
];

pub fn symbol_to_atomic_number(sym: &str) -> usize {
    let sym_upper = sym.to_uppercase();
    for (idx, &elem) in PERIODIC_TABLE.iter().enumerate() {
        if elem.to_uppercase() == sym_upper {
            return idx + 1;
        }
    }
    panic!("Unknown atomic symbol: '{}'. Supported elements are 1 (H) to 118 (Og).", sym);
}

pub fn atomic_number_to_symbol(z: usize) -> &'static str {
    if (1..=118).contains(&z) {
        PERIODIC_TABLE[z - 1]
    } else {
        panic!("Unknown atomic number: {}. Supported range is 1 to 118.", z);
    }
}

/// Primitive Gaussian function
#[derive(Clone, Debug)]
pub struct Primitive {
    pub exponent: f64,
    pub coefficient: f64,
}

/// Contracted Gaussian orbital on an atom
#[derive(Clone, Debug)]
pub struct ContractedGaussian {
    pub atom_index: usize,
    pub center: [f64; 3],
    /// Angular momentum powers (lx, ly, lz)
    pub l: [usize; 3],
    /// Primitives: (exponent, normalized_contraction_coeff)
    pub primitives: Vec<Primitive>,
}

impl ContractedGaussian {
    pub fn total_angular_momentum(&self) -> usize {
        self.l[0] + self.l[1] + self.l[2]
    }
}

/// Double factorial (2n - 1)!! with (-1)!! = 1
pub fn double_factorial(n: i64) -> f64 {
    if n <= 0 {
        1.0
    } else {
        let mut res = 1.0;
        let mut k = n;
        while k > 0 {
            res *= k as f64;
            k -= 2;
        }
        res
    }
}

/// Computes primitive Cartesian GTO normalization constant:
/// N(alpha, lx, ly, lz) = (2*alpha/pi)^(3/4) * sqrt( (4*alpha)^(lx+ly+lz) / [ (2lx-1)!! (2ly-1)!! (2lz-1)!! ] )
pub fn primitive_norm(alpha: f64, l: [usize; 3]) -> f64 {
    use std::f64::consts::PI;
    let l_tot = l[0] + l[1] + l[2];
    let prefactor = (2.0 * alpha / PI).powf(0.75);
    let num = (4.0 * alpha).powi(l_tot as i32);
    let denom = double_factorial(2 * l[0] as i64 - 1)
        * double_factorial(2 * l[1] as i64 - 1)
        * double_factorial(2 * l[2] as i64 - 1);
    prefactor * (num / denom).sqrt()
}

/// Maps shell type string to angular momentum L (S=0, P=1, D=2, F=3, G=4, H=5, I=6, K=7, etc.)
pub fn shell_type_to_l(shell_type: &str) -> Option<usize> {
    let s = shell_type.trim().to_uppercase();
    match s.as_str() {
        "S" => Some(0),
        "P" => Some(1),
        "D" => Some(2),
        "F" => Some(3),
        "G" => Some(4),
        "H" => Some(5),
        "I" => Some(6),
        "K" => Some(7),
        "L" => Some(8),
        "M" => Some(9),
        "N" => Some(10),
        _ => {
            if let Some(rest) = s.strip_prefix('L') {
                rest.parse::<usize>().ok()
            } else {
                s.parse::<usize>().ok()
            }
        }
    }
}

/// Maps angular momentum L to standard shell type string (0 -> S, 1 -> P, 2 -> D, 3 -> F, 4 -> G, etc.)
pub fn l_to_shell_type(l: usize) -> String {
    match l {
        0 => "S".to_string(),
        1 => "P".to_string(),
        2 => "D".to_string(),
        3 => "F".to_string(),
        4 => "G".to_string(),
        5 => "H".to_string(),
        6 => "I".to_string(),
        7 => "K".to_string(),
        8 => "L".to_string(),
        9 => "M".to_string(),
        10 => "N".to_string(),
        other => format!("L{}", other),
    }
}

/// Generates all Cartesian exponent triplets (lx, ly, lz) such that lx + ly + lz = L.
/// Total count is (L + 1) * (L + 2) / 2.
pub fn cartesian_components_for_l(l: usize) -> Vec<[usize; 3]> {
    let mut comps = Vec::with_capacity((l + 1) * (l + 2) / 2);
    for lx in (0..=l).rev() {
        for ly in (0..=(l - lx)).rev() {
            let lz = l - lx - ly;
            comps.push([lx, ly, lz]);
        }
    }
    comps
}

/// Definition of a shell in a basis set (supports arbitrary shells: S, P, D, F, G, H, I, SP, etc.)
#[derive(Clone, Debug)]
pub struct ShellDef {
    pub shell_type: String, // "S", "P", "D", "F", "G", "H", "I", "SP", etc.
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
    /// Secondary coefficients for combined SP shells
    pub p_coefficients: Option<Vec<f64>>,
}

/// Map of element symbol -> list of ShellDefs
pub type BasisSetMap = HashMap<String, Vec<ShellDef>>;

/// BSE JSON Schema representation
#[derive(Deserialize)]
struct BseBasisJson {
    elements: HashMap<String, BseElement>,
}

#[derive(Deserialize)]
struct BseElement {
    electron_shells: Vec<BseShell>,
}

#[derive(Deserialize)]
struct BseShell {
    angular_momentum: Vec<usize>,
    exponents: Vec<serde_json::Value>,
    coefficients: Vec<Vec<serde_json::Value>>,
}

fn parse_basis_number(text: &str) -> Result<f64, String> {
    let value = text.replace(['D', 'd'], "E").parse::<f64>()
        .map_err(|_| format!("Invalid basis number: {text}"))?;
    if !value.is_finite() {
        return Err(format!("Non-finite basis number: {text}"));
    }
    Ok(value)
}

fn parse_json_f64(val: &serde_json::Value) -> Result<f64, String> {
    let value = if let Some(num) = val.as_f64() {
        num
    } else if let Some(s) = val.as_str() {
        return parse_basis_number(s);
    } else {
        return Err(format!("Invalid BSE number: {val}"));
    };
    if !value.is_finite() { return Err("Non-finite BSE number".into()); }
    Ok(value)
}

/// Parses basis set from BasisSetExchange (BSE) JSON format string
pub fn parse_bse_json(content: &str) -> Result<BasisSetMap, String> {
    let bse: BseBasisJson = serde_json::from_str(content).map_err(|e| format!("BSE JSON parse error: {}", e))?;
    let mut map = BasisSetMap::new();

    for (elem_key, elem_data) in bse.elements {
        // elem_key can be atomic number ("1", "8") or element symbol ("H", "O")
        let sym = if let Ok(z) = elem_key.parse::<usize>() {
            if !(1..=118).contains(&z) { return Err(format!("Invalid atomic number: {z}")); }
            atomic_number_to_symbol(z).to_uppercase()
        } else {
            elem_key.to_uppercase()
        };

        let mut shells = Vec::new();
        for s in elem_data.electron_shells {
            let exps = s.exponents.iter().map(parse_json_f64).collect::<Result<Vec<_>, _>>()?;
            if exps.is_empty() || exps.iter().any(|&x| x <= 0.0)
                || s.angular_momentum.is_empty() || s.angular_momentum.iter().any(|&l| l > 7)
                || s.coefficients.is_empty() || s.coefficients.len() % s.angular_momentum.len() != 0 {
                return Err(format!("Invalid BSE shell for {sym}"));
            }
            let coefficients = s.coefficients.iter().map(|column| {
                if column.len() != exps.len() { return Err(format!("BSE primitive/coefficients length mismatch for {sym}")); }
                column.iter().map(parse_json_f64).collect::<Result<Vec<_>, _>>()
            }).collect::<Result<Vec<_>, _>>()?;
            if s.angular_momentum == [0, 1] {
                for pair in coefficients.chunks_exact(2) {
                    shells.push(ShellDef {
                        shell_type: "SP".to_string(), exponents: exps.clone(),
                        coefficients: pair[0].clone(), p_coefficients: Some(pair[1].clone()),
                    });
                }
            } else {
                // A single angular momentum may have several general contractions.
                for (idx, coeffs) in coefficients.into_iter().enumerate() {
                    let l = s.angular_momentum[idx % s.angular_momentum.len()];
                    shells.push(ShellDef {
                        shell_type: l_to_shell_type(l),
                        exponents: exps.clone(),
                        coefficients: coeffs,
                        p_coefficients: None,
                    });
                }
            }
        }

        map.insert(sym, shells);
    }

    Ok(map)
}

/// Parses Gaussian 94 shells, including exponent scale factors and general contractions.
pub fn try_parse_gaussian94(content: &str) -> Result<BasisSetMap, String> {
    let lines = content.lines().map(|line| line.split(['!', '#']).next().unwrap_or("").trim())
        .filter(|line| !line.is_empty()).collect::<Vec<_>>();
    let mut map = BasisSetMap::new();
    let mut element: Option<String> = None;
    let mut i = 0;
    while i < lines.len() {
        if lines[i] == "****" { element = None; i += 1; continue; }
        let tokens = lines[i].split_whitespace().collect::<Vec<_>>();
        if element.is_none() {
            let sym = tokens[0].to_uppercase();
            if !PERIODIC_TABLE.iter().any(|s| s.to_uppercase() == sym)
                || tokens.len() != 2 || tokens[1] != "0" {
                return Err(format!("Invalid Gaussian94 element header: {}", lines[i]));
            }
            map.entry(sym.clone()).or_default();
            element = Some(sym);
            i += 1;
            continue;
        }
        let shell_type = tokens[0].to_uppercase();
        if tokens.len() != 3 || (shell_type != "SP" && shell_type_to_l(&shell_type).is_none_or(|l| l > 7)) {
            return Err(format!("Invalid Gaussian94 shell header: {}", lines[i]));
        }
        let n = tokens[1].parse::<usize>().map_err(|_| "Invalid primitive count")?;
        let scale = parse_basis_number(tokens[2])?;
        if n == 0 || scale <= 0.0 || i + n >= lines.len() {
            return Err("Invalid or truncated Gaussian94 shell".into());
        }
        let mut exponents = Vec::with_capacity(n);
        let mut columns: Vec<Vec<f64>> = Vec::new();
        for row in 0..n {
            i += 1;
            let values = lines[i].split_whitespace().map(parse_basis_number)
                .collect::<Result<Vec<_>, _>>()?;
            if values.len() < 2 || values[0] <= 0.0 {
                return Err(format!("Invalid Gaussian94 primitive: {}", lines[i]));
            }
            if row == 0 { columns = vec![Vec::with_capacity(n); values.len() - 1]; }
            if values.len() != columns.len() + 1 { return Err("Inconsistent contraction column count".into()); }
            // Gaussian's shell scale multiplies orbital exponents by scale squared.
            exponents.push(values[0] * scale * scale);
            for (column, &value) in columns.iter_mut().zip(&values[1..]) { column.push(value); }
        }
        let shells = map.get_mut(element.as_ref().ok_or("Missing element")?).ok_or("Missing element")?;
        if shell_type == "SP" {
            if !columns.len().is_multiple_of(2) { return Err("SP shells require pairs of coefficient columns".into()); }
            for pair in columns.chunks_exact(2) {
                shells.push(ShellDef { shell_type: shell_type.clone(), exponents: exponents.clone(),
                    coefficients: pair[0].clone(), p_coefficients: Some(pair[1].clone()) });
            }
        } else {
            for coefficients in columns {
                shells.push(ShellDef { shell_type: shell_type.clone(), exponents: exponents.clone(),
                    coefficients, p_coefficients: None });
            }
        }
        i += 1;
    }
    if map.is_empty() || map.values().any(Vec::is_empty) { return Err("Empty Gaussian94 basis".into()); }
    Ok(map)
}

/// Legacy convenience parser. Use `try_parse_gaussian94` for external input.
pub fn parse_gaussian94(content: &str) -> BasisSetMap {
    try_parse_gaussian94(content).expect("Invalid Gaussian94 basis")
}

/// Reads a basis set file (supports BSE .json and Gaussian 94 .gbs formats automatically)
pub fn load_basis_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<BasisSetMap> {
    let content = fs::read_to_string(path)?;
    let trimmed = content.trim();
    if trimmed.starts_with('{') {
        parse_bse_json(trimmed).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    } else {
        try_parse_gaussian94(trimmed).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Generates Cartesian Contracted Gaussians for a molecule using a BasisSetMap
pub fn try_build_basis(atoms:&[Atom],basis_map:&BasisSetMap)->Result<Vec<ContractedGaussian>,crate::error::HfError> {
    use crate::error::HfError;
    if atoms.is_empty() {return Err(HfError::InvalidInput("empty molecule".into()));}
    for atom in atoms {
        if atom.position.iter().any(|v|!v.is_finite()) || !atom.charge.is_finite() || atom.charge<0.0 {
            return Err(HfError::InvalidInput("non-finite geometry or invalid effective nuclear charge".into()));
        }
        let shells=basis_map.get(&atom.symbol.to_uppercase()).ok_or_else(||HfError::InvalidInput(format!("missing basis for {}",atom.symbol)))?;
        if shells.is_empty() {return Err(HfError::InvalidInput(format!("empty basis for {}",atom.symbol)));}
        for shell in shells {
            let n=shell.exponents.len();
            let coefficients_valid=|c:&[f64]|c.len()==n && c.iter().all(|v|v.is_finite()) && c.iter().any(|v|*v!=0.0);
            if n==0 || shell.exponents.iter().any(|v|!v.is_finite() || *v<=0.0) || !coefficients_valid(&shell.coefficients)
                || (shell.shell_type=="SP" && shell.p_coefficients.as_ref().is_none_or(|c|!coefficients_valid(c)))
                || (shell.shell_type!="SP" && shell_type_to_l(&shell.shell_type).is_none_or(|l|l>7)) {
                return Err(HfError::InvalidInput(format!("invalid {} shell for {}",shell.shell_type,atom.symbol)));
            }
        }
    }
    let basis=build_basis(atoms,basis_map);
    if basis.iter().any(|cg|cg.primitives.iter().any(|p|!p.coefficient.is_finite())) {
        return Err(HfError::Integral("basis normalization overflow or singular contraction".into()));
    }
    Ok(basis)
}

pub fn build_basis(atoms: &[Atom], basis_map: &BasisSetMap) -> Vec<ContractedGaussian> {
    let mut basis = Vec::new();

    for (atom_idx, atom) in atoms.iter().enumerate() {
        let shells = basis_map
            .get(&atom.symbol.to_uppercase())
            .unwrap_or_else(|| panic!("Basis set not defined for element '{}'", atom.symbol));

        for shell in shells {
            if shell.shell_type == "SP" {
                // S shell
                basis.push(make_contracted_gaussian(atom_idx, atom.position, [0, 0, 0], &shell.exponents, &shell.coefficients));
                // P shells using p_coefficients
                let p_coeffs = shell.p_coefficients.as_ref().unwrap();
                basis.push(make_contracted_gaussian(atom_idx, atom.position, [1, 0, 0], &shell.exponents, p_coeffs));
                basis.push(make_contracted_gaussian(atom_idx, atom.position, [0, 1, 0], &shell.exponents, p_coeffs));
                basis.push(make_contracted_gaussian(atom_idx, atom.position, [0, 0, 1], &shell.exponents, p_coeffs));
            } else if let Some(l) = shell_type_to_l(&shell.shell_type) {
                match l {
                    0 => {
                        basis.push(make_contracted_gaussian(atom_idx, atom.position, [0, 0, 0], &shell.exponents, &shell.coefficients));
                    }
                    1 => {
                        basis.push(make_contracted_gaussian(atom_idx, atom.position, [1, 0, 0], &shell.exponents, &shell.coefficients));
                        basis.push(make_contracted_gaussian(atom_idx, atom.position, [0, 1, 0], &shell.exponents, &shell.coefficients));
                        basis.push(make_contracted_gaussian(atom_idx, atom.position, [0, 0, 1], &shell.exponents, &shell.coefficients));
                    }
                    2 => {
                        // 6 Cartesian d-orbitals: xx, yy, zz, xy, xz, yz
                        basis.push(make_contracted_gaussian(atom_idx, atom.position, [2, 0, 0], &shell.exponents, &shell.coefficients));
                        basis.push(make_contracted_gaussian(atom_idx, atom.position, [0, 2, 0], &shell.exponents, &shell.coefficients));
                        basis.push(make_contracted_gaussian(atom_idx, atom.position, [0, 0, 2], &shell.exponents, &shell.coefficients));
                        basis.push(make_contracted_gaussian(atom_idx, atom.position, [1, 1, 0], &shell.exponents, &shell.coefficients));
                        basis.push(make_contracted_gaussian(atom_idx, atom.position, [1, 0, 1], &shell.exponents, &shell.coefficients));
                        basis.push(make_contracted_gaussian(atom_idx, atom.position, [0, 1, 1], &shell.exponents, &shell.coefficients));
                    }
                    _ => {
                        // Arbitrary angular momentum L >= 3: F (10), G (15), H (21), I (28), ...
                        for comp in cartesian_components_for_l(l) {
                            basis.push(make_contracted_gaussian(atom_idx, atom.position, comp, &shell.exponents, &shell.coefficients));
                        }
                    }
                }
            } else {
                panic!("Unsupported shell type: {}", shell.shell_type);
            }
        }
    }

    basis
}

/// Creates a normalized ContractedGaussian function
fn make_contracted_gaussian(
    atom_index: usize,
    center: [f64; 3],
    l: [usize; 3],
    exponents: &[f64],
    coefficients: &[f64],
) -> ContractedGaussian {
    let mut prims = Vec::with_capacity(exponents.len());
    for (&exp, &coeff) in exponents.iter().zip(coefficients.iter()) {
        let norm = primitive_norm(exp, l);
        prims.push(Primitive {
            exponent: exp,
            coefficient: coeff * norm,
        });
    }

    // Compute self-overlap to ensure exact normalization: <chi|chi> = 1
    use std::f64::consts::PI;
    let mut self_overlap = 0.0;
    for p1 in &prims {
        for p2 in &prims {
            let p = p1.exponent + p2.exponent;
            let s_xyz = (PI / p).powf(1.5)
                * (double_factorial(2 * l[0] as i64 - 1) / (2.0 * p).powi(l[0] as i32))
                * (double_factorial(2 * l[1] as i64 - 1) / (2.0 * p).powi(l[1] as i32))
                * (double_factorial(2 * l[2] as i64 - 1) / (2.0 * p).powi(l[2] as i32));
            self_overlap += p1.coefficient * p2.coefficient * s_xyz;
        }
    }

    let norm_factor = 1.0 / self_overlap.sqrt();
    for p in &mut prims {
        p.coefficient *= norm_factor;
    }

    ContractedGaussian {
        atom_index,
        center,
        l,
        primitives: prims,
    }
}

pub use crate::basis_data::*;

/// Normalizes a user-facing basis set name: lower case, no spaces / hyphens / underscores,
/// `(d)` / `(d,p)` polarization notation mapped to `*` / `**`.
pub(crate) fn normalize_basis_name(name: &str) -> String {
    name.trim()
        .to_lowercase()
        .replace([' ', '_', '-'], "")
        .replace("(d,p)", "**")
        .replace("(d)", "*")
        .replace("gdp", "g**")
        .replace("631gd", "631g*")
        .replace("6311gss", "6311g**")
}

/// Looks up and parses a built-in orbital basis set by name (case-insensitive, ignoring hyphens/spaces/underscores).
/// All sets come from the Basis Set Exchange with their full element coverage.
///
/// Use `list_builtin_basis_sets` for the full orbital catalog and
/// third_party/basis_set_exchange/manifest.json for additional export provenance.
pub fn get_builtin_basis(name: &str) -> Option<BasisSetMap> {
    let text = match normalize_basis_name(name).as_str() {
        "sto3g" => STO_3G_GBS,
        "321g" => BASIS_3_21G_GBS,
        "631g" => BASIS_6_31G_GBS,
        "631g*" => BASIS_6_31G_STAR_GBS,
        "631g**" => BASIS_6_31G_STAR_STAR_GBS,
        "6311g**" => BASIS_6_311G_SS_GBS,
        "def2svp" | "svp" => BASIS_DEF2_SVP_GBS,
        "def2tzvp" | "tzvp" => BASIS_DEF2_TZVP_GBS,
        "def2qzvp" | "qzvp" => BASIS_DEF2_QZVP_GBS,
        "ccpvdz" | "pvdz" => BASIS_CC_PVDZ_GBS,
        "ccpvtz" | "pvtz" => BASIS_CC_PVTZ_GBS,
        "ccpvqz" | "pvqz" => BASIS_CC_PVQZ_GBS,
        "augccpvdz" | "augpvdz" => BASIS_AUG_CC_PVDZ_GBS,
        "augccpvtz" | "augpvtz" => BASIS_AUG_CC_PVTZ_GBS,
        _ => return crate::builtin_catalog::json(name).map(|data| {
            parse_bse_json(data).expect("embedded BSE orbital data must parse")
        }),
    };
    Some(parse_gaussian94(text))
}

/// Looks up and parses a built-in auxiliary (density-fitting / RI / CABS) basis set by name.
///
/// Supported: def2-universal-JKFIT, def2-SVP-RIFIT, def2-TZVP-RIFIT, cc-pVDZ-RIFIT, cc-pVTZ-RIFIT,
/// aug-cc-pVDZ-RIFIT, cc-pVDZ-F12-OptRI, cc-pVTZ-F12-OptRI.
pub fn get_builtin_aux_basis(name: &str) -> Option<BasisSetMap> {
    let text = match normalize_basis_name(name).as_str() {
        "def2universaljkfit" | "def2jkfit" => AUX_DEF2_UNIVERSAL_JKFIT_GBS,
        "def2svprifit" | "def2svpri" => AUX_DEF2_SVP_RIFIT_GBS,
        "def2tzvprifit" | "def2tzvpri" => AUX_DEF2_TZVP_RIFIT_GBS,
        "ccpvdzrifit" | "ccpvdzri" => AUX_CC_PVDZ_RIFIT_GBS,
        "ccpvtzrifit" | "ccpvtzri" => AUX_CC_PVTZ_RIFIT_GBS,
        "augccpvdzrifit" | "augccpvdzri" => AUX_AUG_CC_PVDZ_RIFIT_GBS,
        "ccpvdzf12optri" => AUX_CC_PVDZ_F12_OPTRI_GBS,
        "ccpvtzf12optri" => AUX_CC_PVTZ_F12_OPTRI_GBS,
        _ => return None,
    };
    Some(parse_gaussian94(text))
}

/// Returns a list of all built-in orbital basis set canonical names.
pub fn list_builtin_basis_sets() -> Vec<&'static str> {
    let mut names = vec![
        "sto-3g", "3-21g", "6-31g", "6-31g*", "6-31g**", "6-311g**", "def2-svp", "def2-tzvp", "def2-qzvp",
        "cc-pvdz", "cc-pvtz", "cc-pvqz", "aug-cc-pvdz", "aug-cc-pvtz",
    ];
    names.extend(crate::builtin_catalog::ORBITAL.iter().map(|&(name, _)| name));
    names
}

/// Returns a list of all built-in auxiliary basis set canonical names.
pub fn list_builtin_aux_basis_sets() -> Vec<&'static str> {
    vec![
        "def2-universal-jkfit", "def2-svp-rifit", "def2-tzvp-rifit", "cc-pvdz-rifit", "cc-pvtz-rifit",
        "aug-cc-pvdz-rifit", "cc-pvdz-f12-optri", "cc-pvtz-f12-optri",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrals::{contracted_overlap, contracted_kinetic, contracted_nuclear_attraction, contracted_eri};

    #[test]
    fn test_arbitrary_orbitals_components() {
        assert_eq!(shell_type_to_l("S"), Some(0));
        assert_eq!(shell_type_to_l("P"), Some(1));
        assert_eq!(shell_type_to_l("D"), Some(2));
        assert_eq!(shell_type_to_l("F"), Some(3));
        assert_eq!(shell_type_to_l("G"), Some(4));
        assert_eq!(shell_type_to_l("H"), Some(5));
        assert_eq!(shell_type_to_l("I"), Some(6));

        // Cartesian count = (L+1)(L+2)/2
        assert_eq!(cartesian_components_for_l(0).len(), 1);   // S
        assert_eq!(cartesian_components_for_l(1).len(), 3);   // P
        assert_eq!(cartesian_components_for_l(2).len(), 6);   // D
        assert_eq!(cartesian_components_for_l(3).len(), 10);  // F
        assert_eq!(cartesian_components_for_l(4).len(), 15);  // G
        assert_eq!(cartesian_components_for_l(5).len(), 21);  // H
        assert_eq!(cartesian_components_for_l(6).len(), 28);  // I
    }

    #[test]
    fn test_f_and_g_orbitals_integrals() {
        let atom = Atom::new("C", [0.0, 0.0, 0.0]);
        let mut map = BasisSetMap::new();
        map.insert("C".to_string(), vec![
            ShellDef {
                shell_type: "F".to_string(),
                exponents: vec![0.8],
                coefficients: vec![1.0],
                p_coefficients: None,
            },
            ShellDef {
                shell_type: "G".to_string(),
                exponents: vec![0.5],
                coefficients: vec![1.0],
                p_coefficients: None,
            },
        ]);
        let basis = build_basis(std::slice::from_ref(&atom), &map);
        assert_eq!(basis.len(), 25);

        for cg in &basis {
            let s_ii = contracted_overlap(cg, cg);
            assert!((s_ii - 1.0).abs() < 1e-12, "Self overlap was {}", s_ii);
            let t_ii = contracted_kinetic(cg, cg);
            assert!(t_ii > 0.0, "Kinetic energy must be positive");
            let v_ii = contracted_nuclear_attraction(cg, cg, std::slice::from_ref(&atom));
            assert!(v_ii < 0.0, "Nuclear attraction must be negative");
            let eri_iiii = contracted_eri(cg, cg, cg, cg);
            assert!(eri_iiii > 0.0, "Self-repulsion ERI must be positive");
        }
    }

    #[test]
    fn test_all_builtin_basis_sets_from_bse() {
        let names = list_builtin_basis_sets();
        assert_eq!(names.len(), 80);
        let atom_h = Atom::new("H", [0.0, 0.0, 0.0]);
        let atom_c = Atom::new("C", [0.0, 0.0, 2.0]);
        let mol = [atom_h, atom_c];

        for name in &names {
            let map = get_builtin_basis(name).unwrap_or_else(|| panic!("Failed to get basis '{}'", name));
            if !map.contains_key("H") || !map.contains_key("C") {
                // ECP orbital families need not define first-row elements.
                continue;
            }
            let basis = build_basis(&mol, &map);
            assert!(!basis.is_empty(), "Basis '{}' generated 0 basis functions", name);
            for cg in &basis {
                let s_ii = contracted_overlap(cg, cg);
                assert!((s_ii - 1.0).abs() < 1e-10, "Basis '{}' orbital normalization error: s_ii = {}", name, s_ii);
            }
        }
    }
}
