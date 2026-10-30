// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Per-element orbital and ECP assignments shared by all molecular frontends.

use crate::{
    api::MoleculeOptions,
    basis::{Atom, BasisSetMap, atomic_number_to_symbol},
    ecp::EcpMap,
    error::HfError,
};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

fn invalid(message: impl Into<String>) -> HfError {
    HfError::InvalidInput(message.into())
}

fn element_key(key: &str) -> Result<String, HfError> {
    let text = key.trim();
    let z = text
        .parse::<usize>()
        .ok()
        .filter(|z| (1..=118).contains(z))
        .or_else(|| crate::masses::z_from_symbol(text));
    z.map(|z| atomic_number_to_symbol(z).to_uppercase())
        .ok_or_else(|| invalid(format!("invalid element assignment key {key}")))
}

pub(crate) fn assignment(
    value: &Value,
    ecp: bool,
) -> Result<(Option<String>, BTreeMap<String, String>), HfError> {
    match value {
        Value::String(name) if name.trim_start().starts_with('{') => assignment(
            &serde_json::from_str::<Value>(name).map_err(|e| invalid(e.to_string()))?,
            ecp,
        ),
        Value::String(name) if !name.trim().is_empty() => Ok((Some(name.clone()), BTreeMap::new())),
        Value::Null if ecp => Ok((None, BTreeMap::new())),
        Value::Object(entries) => {
            let mut default = None;
            let mut elements = BTreeMap::new();
            for (key, value) in entries {
                let source = match value {
                    Value::String(name) if !name.trim().is_empty() => name.clone(),
                    Value::Null if ecp => "none".into(),
                    _ => {
                        return Err(invalid(format!(
                            "assignment for {key} requires a name or filename"
                        )));
                    }
                };
                if key == "*" {
                    default = Some(source);
                } else if elements.insert(element_key(key)?, source).is_some() {
                    return Err(invalid(format!("duplicate element assignment {key}")));
                }
            }
            if entries.is_empty() {
                return Err(invalid("empty element assignment"));
            }
            Ok((default, elements))
        }
        _ => Err(invalid(
            "basis/ECP selection requires a name, filename or element map",
        )),
    }
}

fn read(source: &str) -> Result<String, HfError> {
    std::fs::read_to_string(source).map_err(|e| invalid(format!("basis/ECP {source}: {e}")))
}

fn orbital(source: &str) -> Result<(BasisSetMap, EcpMap), HfError> {
    if let Some(basis) = crate::basis::get_builtin_basis(source) {
        return Ok((
            basis,
            crate::ecp::get_builtin_ecp(source).unwrap_or_default(),
        ));
    }
    let text = read(source)?;
    let basis = if text.trim_start().starts_with('{') {
        crate::basis::parse_bse_json(&text)
    } else {
        let orbitals = text
            .lines()
            .take_while(|line| {
                !line
                    .split_whitespace()
                    .next()
                    .is_some_and(|token| token.to_uppercase().ends_with("-ECP"))
            })
            .collect::<Vec<_>>()
            .join("\n");
        // An element header may precede the first ECP header in combined exports.
        let orbitals = orbitals.trim_end();
        let has_ecp = text.lines().any(|line| {
            line.split_whitespace()
                .next()
                .is_some_and(|token| token.to_uppercase().ends_with("-ECP"))
        });
        let orbitals = match orbitals.rfind("****").filter(|_| has_ecp) {
            Some(last) => &orbitals[..last + 4],
            None => orbitals,
        };
        crate::basis::try_parse_gaussian94(orbitals)
    }
    .map_err(invalid)?;
    let potentials = potential_text(&text)?;
    Ok((basis, potentials))
}

fn potential_text(text: &str) -> Result<EcpMap, HfError> {
    if text.trim_start().starts_with('{') {
        crate::ecp::parse_ecp_bse_json(text).map_err(invalid)
    } else {
        crate::ecp::parse_ecp_gaussian94(text).map_err(invalid)
    }
}

fn potentials(source: &str) -> Result<EcpMap, HfError> {
    let map = match crate::ecp::get_builtin_ecp(source) {
        Some(map) => map,
        None => potential_text(&read(source)?)?,
    };
    if map.is_empty() {
        return Err(invalid(format!("no ECP definitions in {source}")));
    }
    Ok(map)
}

pub(crate) fn resolve(
    atoms: &mut [Atom],
    options: &MoleculeOptions,
) -> Result<BasisSetMap, HfError> {
    let normalize =
        |entries: &BTreeMap<String, String>| -> Result<BTreeMap<String, String>, HfError> {
            let mut normalized = BTreeMap::new();
            for (key, value) in entries {
                if normalized
                    .insert(element_key(key)?, value.clone())
                    .is_some()
                {
                    return Err(invalid(format!("duplicate element assignment {key}")));
                }
            }
            Ok(normalized)
        };
    let orbital_overrides = normalize(&options.basis_by_element)?;
    let ecp_overrides = normalize(&options.ecp_by_element)?;
    let mut orbital_cache = HashMap::new();
    let mut ecp_cache = HashMap::new();
    let mut result = BasisSetMap::new();
    for atom in atoms {
        let symbol = atom.symbol.to_uppercase();
        let source = orbital_overrides.get(&symbol).unwrap_or(&options.basis);
        if source.is_empty() {
            return Err(invalid(format!("missing orbital basis for {symbol}")));
        }
        if !orbital_cache.contains_key(source) {
            orbital_cache.insert(source.clone(), orbital(source)?);
        }
        let (basis, associated) = &orbital_cache[source];
        let shells = basis
            .get(&symbol)
            .ok_or_else(|| invalid(format!("basis {source} does not cover {symbol}")))?;
        result.insert(symbol.clone(), shells.clone());
        let explicit = ecp_overrides.get(&symbol).or(options.ecp.as_ref());
        let potential = match explicit {
            None => associated.get(&symbol).cloned(),
            Some(name)
                if name.eq_ignore_ascii_case("none")
                    || name.eq_ignore_ascii_case("all-electron") =>
            {
                None
            }
            Some(name) => {
                if !ecp_cache.contains_key(name) {
                    ecp_cache.insert(name.clone(), potentials(name)?);
                }
                let potential = ecp_cache[name].get(&symbol).cloned();
                if potential.is_none() && ecp_overrides.contains_key(&symbol) {
                    return Err(invalid(format!(
                        "ECP {name} does not cover explicitly assigned element {symbol}"
                    )));
                }
                potential
            }
        };
        atom.charge = atom.atomic_number as f64 - potential.as_ref().map_or(0, |p| p.n_core) as f64;
        atom.ecp = potential;
    }
    Ok(result)
}
