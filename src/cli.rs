// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Command-line parsing with the same calculation boundary as Python and ASE.
use crate::{
    api::{self, MoleculeOptions, OptionMap},
    error::HfError,
};
use serde_json::Value;

pub const HELP: &str = "hf-rs --method METHOD [--geometry 'H 0 0 -0.7; H 0 0 0.7' | --xyz FILE]
  --unit bohr|angstrom       Default bohr; XYZ defaults to angstrom
  --basis NAME_OR_FILE       Default sto-3g
  --ecp NAME_OR_FILE         Optional independent ECP; both accept JSON element maps
  --cartesian | --spherical  Default Cartesian
  --charge INTEGER          Default 0
  --multiplicity INTEGER    Default 1
  --reference auto|rhf|uhf   Default auto
  --opt KEY=VALUE            Repeatable registry/SCF options; JSON values accepted
  --property energy|gradient|hessian|observables|intermediates  Repeatable; default energy
  --oracle-compare           Run the embedded independent reference checks
  --list-methods             Print registered methods and option schemas
  --version                 Print the package version
Results are JSON in atomic units. Progress is written to stderr.
";

pub fn execute(args: &[String]) -> Result<Value, HfError> {
    let mut options = MoleculeOptions::default();
    let mut method = "hf".to_string();
    let mut geometry = "H 0 0 -0.7; H 0 0 0.7".to_string();
    let mut xyz = None;
    let mut explicit_unit = false;
    let mut values = OptionMap::new();
    let mut properties = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        if flag == "--oracle-compare" {
            if args.len() != 1 {
                return Err(HfError::InvalidInput(
                    "--oracle-compare takes no calculation arguments".into(),
                ));
            }
            return crate::validation::compare();
        }
        if flag == "--list-methods" {
            if args.len() != 1 {
                return Err(HfError::InvalidInput(
                    "--list-methods takes no calculation arguments".into(),
                ));
            }
            return Ok(api::method_catalog());
        }
        if flag == "--spherical" {
            options.cartesian = false;
            i += 1;
            continue;
        }
        if flag == "--cartesian" {
            options.cartesian = true;
            i += 1;
            continue;
        }
        let value = args
            .get(i + 1)
            .ok_or_else(|| HfError::InvalidInput(format!("missing value for {flag}")))?;
        match flag.as_str() {
            "--method" => method = value.clone(),
            "--geometry" => geometry = value.clone(),
            "--xyz" => xyz = Some(value.clone()),
            "--basis" => options.set_basis(
                &serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.clone())),
            )?,
            "--ecp" => options.set_ecp(
                &serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.clone())),
            )?,
            "--unit" => {
                options.unit = value.parse()?;
                explicit_unit = true;
            }
            "--reference" => options.reference = value.parse()?,
            "--charge" => {
                options.charge = value
                    .parse()
                    .map_err(|_| HfError::InvalidInput("invalid charge".into()))?
            }
            "--multiplicity" => {
                options.multiplicity = value
                    .parse()
                    .map_err(|_| HfError::InvalidInput("invalid multiplicity".into()))?
            }
            "--property" => properties.push(value.clone()),
            "--opt" => {
                let (key, value) = value
                    .split_once('=')
                    .filter(|(key, _)| !key.is_empty())
                    .ok_or_else(|| HfError::InvalidInput("--opt requires KEY=VALUE".into()))?;
                if values
                    .insert(
                        key.into(),
                        serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.into())),
                    )
                    .is_some()
                {
                    return Err(HfError::InvalidInput(format!("duplicate option {key}")));
                }
            }
            _ => return Err(HfError::InvalidInput(format!("unknown argument {flag}"))),
        }
        i += 2;
    }
    let rows: Vec<String> = if let Some(path) = xyz {
        let text =
            std::fs::read_to_string(path).map_err(|e| HfError::InvalidInput(e.to_string()))?;
        let mut lines = text.lines();
        let count: usize = lines
            .next()
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| HfError::InvalidInput("invalid XYZ atom count".into()))?;
        lines
            .next()
            .ok_or_else(|| HfError::InvalidInput("missing XYZ comment line".into()))?;
        let rows: Vec<_> = lines.map(str::to_string).collect();
        if rows.len() != count {
            return Err(HfError::InvalidInput("XYZ atom count mismatch".into()));
        }
        if !explicit_unit {
            options.unit = api::LengthUnit::Angstrom;
        }
        rows
    } else {
        geometry.split(';').map(str::to_string).collect()
    };
    let mut numbers = Vec::new();
    let mut positions = Vec::new();
    for row in rows {
        let fields: Vec<_> = row.split_whitespace().collect();
        if fields.len() != 4 {
            return Err(HfError::InvalidInput(
                "geometry rows require SYMBOL X Y Z".into(),
            ));
        }
        numbers.push(
            crate::masses::z_from_symbol(fields[0])
                .ok_or_else(|| HfError::InvalidInput(format!("unknown element {}", fields[0])))?,
        );
        let mut position = [0.; 3];
        for k in 0..3 {
            position[k] = fields[k + 1]
                .parse()
                .map_err(|_| HfError::InvalidInput("invalid coordinate".into()))?;
        }
        positions.push(position);
    }
    Ok(api::run(&method, &numbers, &positions, options, &values, &properties)?.to_json())
}
