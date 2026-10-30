// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Higher-order spatial derivatives of Cartesian Contracted Gaussians.
//! Supports nuclear coordinate and directional derivatives through fourth order.

use crate::basis::{ContractedGaussian, Primitive};

/// Represents an arbitrary linear combination of Cartesian GTOs resulting from differentiation.
#[derive(Clone, Debug)]
pub struct GeneralBasisDerivative {
    pub terms: Vec<ContractedGaussian>,
}

impl GeneralBasisDerivative {
    pub fn empty() -> Self {
        GeneralBasisDerivative { terms: Vec::new() }
    }

    pub fn from_single(cg: ContractedGaussian) -> Self {
        GeneralBasisDerivative { terms: vec![cg] }
    }
}

/// Algebraic derivative along a displacement of the Gaussian center.
/// Equal angular terms are combined to keep repeated high-order derivatives small.
pub fn diff_deriv_directional(deriv: &GeneralBasisDerivative, direction: [f64; 3]) -> GeneralBasisDerivative {
    let mut out = GeneralBasisDerivative::empty();
    for cg in &deriv.terms {
        for (axis, &scale) in direction.iter().enumerate() {
            if scale == 0.0 { continue; }
            for mut term in diff_cg_1st(cg, axis).terms {
                for p in &mut term.primitives { p.coefficient *= scale; }
                if let Some(previous) = out.terms.iter_mut().find(|previous| {
                    previous.atom_index == term.atom_index && previous.center == term.center && previous.l == term.l
                    && previous.primitives.len() == term.primitives.len()
                    && previous.primitives.iter().zip(&term.primitives).all(|(a, b)| a.exponent == b.exponent)
                }) {
                    for (a, b) in previous.primitives.iter_mut().zip(&term.primitives) { a.coefficient += b.coefficient; }
                } else { out.terms.push(term); }
            }
        }
    }
    out.terms.retain(|t| t.primitives.iter().any(|p| p.coefficient != 0.0));
    out
}

pub fn diff_cg_directions(cg: &ContractedGaussian, directions: &[[f64; 3]]) -> GeneralBasisDerivative {
    directions.iter().fold(GeneralBasisDerivative::from_single(cg.clone()), |d, &v| diff_deriv_directional(&d, v))
}

/// Differentiates a ContractedGaussian along a Cartesian axis:
/// d/dA_tau [ (r - A)^l * exp(-alpha * |r - A|^2) ] = 2*alpha * (r - A)^(l + e_tau) - l_tau * (r - A)^(l - e_tau)
pub fn diff_cg_1st(cg: &ContractedGaussian, tau: usize) -> GeneralBasisDerivative {
    let mut terms = Vec::with_capacity(2);

    // +e_tau term
    let mut l_plus = cg.l;
    l_plus[tau] += 1;
    let mut prims_plus = Vec::with_capacity(cg.primitives.len());
    for p in &cg.primitives {
        prims_plus.push(Primitive {
            exponent: p.exponent,
            coefficient: p.coefficient * (2.0 * p.exponent),
        });
    }
    terms.push(ContractedGaussian {
        atom_index: cg.atom_index,
        center: cg.center,
        l: l_plus,
        primitives: prims_plus,
    });

    // -e_tau term
    if cg.l[tau] > 0 {
        let mut l_minus = cg.l;
        l_minus[tau] -= 1;
        let l_factor = cg.l[tau] as f64;
        let mut prims_minus = Vec::with_capacity(cg.primitives.len());
        for p in &cg.primitives {
            prims_minus.push(Primitive {
                exponent: p.exponent,
                coefficient: p.coefficient * (-l_factor),
            });
        }
        terms.push(ContractedGaussian {
            atom_index: cg.atom_index,
            center: cg.center,
            l: l_minus,
            primitives: prims_minus,
        });
    }

    GeneralBasisDerivative { terms }
}

/// Differentiates a GeneralBasisDerivative along a Cartesian axis.
pub fn diff_deriv_1st(deriv: &GeneralBasisDerivative, tau: usize) -> GeneralBasisDerivative {
    let mut all_terms = Vec::new();
    for cg in &deriv.terms {
        let next = diff_cg_1st(cg, tau);
        all_terms.extend(next.terms);
    }
    GeneralBasisDerivative { terms: all_terms }
}

/// 2nd spatial derivative of a ContractedGaussian: d^2 chi / dA_tau dA_eta
pub fn diff_cg_2nd(cg: &ContractedGaussian, tau: usize, eta: usize) -> GeneralBasisDerivative {
    let d1 = diff_cg_1st(cg, tau);
    diff_deriv_1st(&d1, eta)
}

/// 3rd spatial derivative of a ContractedGaussian: d^3 chi / dA_tau dA_eta dA_xi
pub fn diff_cg_3rd(cg: &ContractedGaussian, tau: usize, eta: usize, xi: usize) -> GeneralBasisDerivative {
    let d2 = diff_cg_2nd(cg, tau, eta);
    diff_deriv_1st(&d2, xi)
}

/// 4th spatial derivative of a ContractedGaussian: d^4 chi / dA_tau dA_eta dA_xi dA_zeta
pub fn diff_cg_4th(
    cg: &ContractedGaussian,
    tau: usize,
    eta: usize,
    xi: usize,
    zeta: usize,
) -> GeneralBasisDerivative {
    let d3 = diff_cg_3rd(cg, tau, eta, xi);
    diff_deriv_1st(&d3, zeta)
}

/// Evaluates overlap integral between two differentiated basis functions
pub fn eval_overlap_deriv(d1: &GeneralBasisDerivative, d2: &GeneralBasisDerivative) -> f64 {
    let mut sum = 0.0;
    for t1 in &d1.terms {
        for t2 in &d2.terms {
            sum += crate::integrals::contracted_overlap(t1, t2);
        }
    }
    sum
}

/// Evaluates kinetic integral between two differentiated basis functions
pub fn eval_kinetic_deriv(d1: &GeneralBasisDerivative, d2: &GeneralBasisDerivative) -> f64 {
    let mut sum = 0.0;
    for t1 in &d1.terms {
        for t2 in &d2.terms {
            sum += crate::integrals::contracted_kinetic(t1, t2);
        }
    }
    sum
}

/// Evaluates nuclear attraction integral for a single atom center between two differentiated basis functions
pub fn eval_nuclear_deriv_single_atom(
    d1: &GeneralBasisDerivative,
    d2: &GeneralBasisDerivative,
    atom: &crate::basis::Atom,
) -> f64 {
    let mut sum = 0.0;
    for t1 in &d1.terms {
        for t2 in &d2.terms {
            for pa in &t1.primitives {
                for pb in &t2.primitives {
                    sum += pa.coefficient
                        * pb.coefficient
                        * crate::integrals::primitive_nuclear_attraction(
                            pa.exponent,
                            t1.l,
                            t1.center,
                            pb.exponent,
                            t2.l,
                            t2.center,
                            atom.position,
                            atom.charge,
                        );
                }
            }
        }
    }
    sum
}

/// Evaluates 2-electron repulsion integral between four differentiated basis functions
pub fn eval_eri_deriv(
    d1: &GeneralBasisDerivative,
    d2: &GeneralBasisDerivative,
    d3: &GeneralBasisDerivative,
    d4: &GeneralBasisDerivative,
) -> f64 {
    let mut sum = 0.0;
    for t1 in &d1.terms {
        for t2 in &d2.terms {
            for t3 in &d3.terms {
                for t4 in &d4.terms {
                    sum += crate::integrals::contracted_eri(t1, t2, t3, t4);
                }
            }
        }
    }
    sum
}

