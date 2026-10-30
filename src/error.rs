// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

/// Recoverable errors from electronic structure calculations.
#[derive(Clone, Debug, PartialEq)]
pub enum HfError {
    InvalidInput(String),
    Scf(String),
    LinearDependence { n_ao: usize, n_mo: usize },
    Response(String),
    Integral(String),
    UnknownMethod(String),
    Unsupported { method: String, detail: String },
    PostHf { method: String, detail: String },
}

impl std::fmt::Display for HfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(s) => write!(f, "invalid input: {s}"),
            Self::Scf(s) => write!(f, "SCF reference: {s}"),
            Self::LinearDependence { n_ao, n_mo } => write!(
                f,
                "analytical derivatives with dropped overlap eigenvectors are unsupported ({n_ao} AOs, {n_mo} MOs)"
            ),
            Self::Response(s) => write!(f, "SCF response: {s}"),
            Self::Integral(s) => write!(f, "integral: {s}"),
            Self::UnknownMethod(s) => write!(f, "unknown method: {s}"),
            Self::Unsupported { method, detail } => write!(f, "{method} is unsupported: {detail}"),
            Self::PostHf { method, detail } => write!(f, "{method}: {detail}"),
        }
    }
}

impl std::error::Error for HfError {}
