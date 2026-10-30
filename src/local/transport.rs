// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.
//
//! Polar transport of full-rank orbital frames and its analytic Frechet response.
use crate::{Matrix, error::HfError};
fn inverse_sqrt_frame(m: &Matrix) -> Result<(Vec<f64>, Matrix, Matrix), HfError> {
    if m.rows < m.cols || m.data.iter().any(|v| !v.is_finite()) {
        return Err(HfError::InvalidInput(
            "invalid orbital transport matrix".into(),
        ));
    }
    let (values, u) = m.transpose().dot(m).eigh();
    if values.iter().any(|v| !v.is_finite() || *v <= 1e-8) {
        return Err(HfError::Response(
            "transported local space lost overlap/rank; rebuild the anchor at a suitable geometry"
                .into(),
        ));
    }
    let mut scaled = u.clone();
    for a in 0..u.rows {
        for b in 0..u.cols {
            scaled[(a, b)] /= values[b].sqrt();
        }
    }
    Ok((values, u.clone(), scaled.dot(&u.transpose())))
}
/// Q=M(M^T M)^(-1/2). Loss of rank is an explicit failure, never a clipped
/// eigenvalue or a geometry-dependent reduction of the space dimension.
pub fn orthonormal_frame(m: &Matrix) -> Result<Matrix, HfError> {
    if m.cols == 0 {
        return Ok(Matrix::zeros(m.rows, 0));
    }
    let (_, _, inverse) = inverse_sqrt_frame(m)?;
    Ok(m.dot(&inverse))
}
/// Exact first response of the polar frame. The divided difference of x^-1/2
/// is -1/(sqrt(x)*sqrt(y)*(sqrt(x)+sqrt(y))), including equal eigenvalues.
pub fn orthonormal_frame_response(m: &Matrix, dm: &Matrix) -> Result<Matrix, HfError> {
    if m.rows != dm.rows || m.cols != dm.cols || dm.data.iter().any(|v| !v.is_finite()) {
        return Err(HfError::InvalidInput(
            "invalid orbital transport response dimensions".into(),
        ));
    }
    if m.cols == 0 {
        return Ok(Matrix::zeros(m.rows, 0));
    }
    let (values, u, inverse) = inverse_sqrt_frame(m)?;
    let metric_response = dm.transpose().dot(m).add(&m.transpose().dot(dm));
    let mut response = u.transpose().dot(&metric_response).dot(&u);
    for a in 0..u.cols {
        for b in 0..u.cols {
            let x = values[a].sqrt();
            let y = values[b].sqrt();
            response[(a, b)] /= -x * y * (x + y);
        }
    }
    let derivative = u.dot(&response).dot(&u.transpose());
    Ok(dm.dot(&inverse).add(&m.dot(&derivative)))
}
