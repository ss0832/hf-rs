// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Small installed smoke oracle; the full scientific suites live under tests/.
use crate::{
    HfError,
    api::{self, MoleculeOptions, OptionMap},
};
use serde::Deserialize;
use serde_json::{Value, json};
#[derive(Deserialize)]
struct Oracle {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    method: String,
    atomic_numbers: Vec<usize>,
    positions: Vec<[f64; 3]>,
    basis: String,
    energy: f64,
    tolerance: f64,
}
pub fn compare() -> Result<Value, HfError> {
    let oracle: Oracle = serde_json::from_str(include_str!("oracle.json"))
        .map_err(|e| HfError::InvalidInput(e.to_string()))?;
    let mut records = Vec::new();
    let mut passed = true;
    for case in oracle.cases {
        let result = api::run(
            &case.method,
            &case.atomic_numbers,
            &case.positions,
            MoleculeOptions {
                basis: case.basis,
                ..Default::default()
            },
            &OptionMap::new(),
            &[],
        )?;
        let delta = (result.output.e_total - case.energy).abs();
        passed &= delta <= case.tolerance;
        records.push(json!({"name": case.name, "actual": result.output.e_total,
            "expected": case.energy, "absolute_error": delta, "tolerance": case.tolerance, "passed": delta <= case.tolerance}));
    }
    Ok(json!({"passed": passed, "cases": records}))
}
