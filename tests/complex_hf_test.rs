// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

use hf_rs::scf::compute_eri_tensor;
use hf_rs::{hf_variants::magnetic::magnetic_one_electron_integrals, *};

fn oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("data/complex_hf_reference.json")).unwrap()
}
fn value(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap()
}
fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(value))
            .collect(),
    )
}

// One global SU(2) gauge aligns a spin-free noncollinear stationary branch.
// The same proper SO(3) rotation is applied to all density and Fock elements.
fn pauli(real: &Matrix, imag: &Matrix) -> (Vec<Matrix>, Vec<Matrix>) {
    let n = real.rows / 2;
    let mut r = vec![Matrix::zeros(n, n); 4];
    let mut i = r.clone();
    for a in 0..n {
        for b in 0..n {
            r[0][(a, b)] = 0.5 * (real[(a, b)] + real[(n + a, n + b)]);
            i[0][(a, b)] = 0.5 * (imag[(a, b)] + imag[(n + a, n + b)]);
            r[1][(a, b)] = 0.5 * (real[(a, n + b)] + real[(n + a, b)]);
            i[1][(a, b)] = 0.5 * (imag[(a, n + b)] + imag[(n + a, b)]);
            r[2][(a, b)] = 0.5 * (imag[(n + a, b)] - imag[(a, n + b)]);
            i[2][(a, b)] = 0.5 * (real[(a, n + b)] - real[(n + a, b)]);
            r[3][(a, b)] = 0.5 * (real[(a, b)] - real[(n + a, n + b)]);
            i[3][(a, b)] = 0.5 * (imag[(a, b)] - imag[(n + a, n + b)]);
        }
    }
    (r, i)
}
fn spin_frame(real: &Matrix, imag: &Matrix, target_r: &Matrix, target_i: &Matrix) -> Matrix {
    let (r, i) = pauli(real, imag);
    let (tr, ti) = pauli(target_r, target_i);
    let mut b = Matrix::zeros(3, 3);
    for a in 0..3 {
        for c in 0..3 {
            b[(a, c)] = r[a + 1]
                .data
                .iter()
                .zip(&tr[c + 1].data)
                .map(|(a, b)| a * b)
                .sum::<f64>()
                + i[a + 1]
                    .data
                    .iter()
                    .zip(&ti[c + 1].data)
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
        }
    }
    let sigma = b.trace();
    let z = [
        b[(1, 2)] - b[(2, 1)],
        b[(2, 0)] - b[(0, 2)],
        b[(0, 1)] - b[(1, 0)],
    ];
    let mut k = Matrix::zeros(4, 4);
    k[(0, 0)] = sigma;
    for a in 0..3 {
        k[(0, a + 1)] = z[a];
        k[(a + 1, 0)] = z[a];
        for c in 0..3 {
            k[(a + 1, c + 1)] = b[(a, c)] + b[(c, a)] - if a == c { sigma } else { 0. };
        }
    }
    let (_, v) = k.eigh();
    let w = v[(0, 3)];
    let q = [v[(1, 3)], v[(2, 3)], v[(3, 3)]];
    let mut rotation = Matrix::zeros(3, 3);
    for a in 0..3 {
        for c in 0..3 {
            rotation[(a, c)] = (if a == c {
                w * w - q.iter().map(|q| q * q).sum::<f64>()
            } else {
                0.
            }) + 2. * q[a] * q[c];
        }
    }
    for (a, c, d) in [(0, 1, 2), (1, 2, 0), (2, 0, 1)] {
        rotation[(a, c)] -= 2. * w * q[d];
        rotation[(c, a)] += 2. * w * q[d];
    }
    rotation
}
fn rotate_spin(real: &Matrix, imag: &Matrix, rotation: &Matrix) -> (Matrix, Matrix) {
    let (r, i) = pauli(real, imag);
    let n = r[0].rows;
    let mut rr = vec![r[0].clone()];
    let mut ii = vec![i[0].clone()];
    for a in 0..3 {
        let mut re = Matrix::zeros(n, n);
        let mut im = re.clone();
        for c in 0..3 {
            re = re.add(&r[c + 1].scale(rotation[(a, c)]));
            im = im.add(&i[c + 1].scale(rotation[(a, c)]));
        }
        rr.push(re);
        ii.push(im);
    }
    let mut re = Matrix::zeros(2 * n, 2 * n);
    let mut im = re.clone();
    for a in 0..n {
        for b in 0..n {
            re[(a, b)] = rr[0][(a, b)] + rr[3][(a, b)];
            im[(a, b)] = ii[0][(a, b)] + ii[3][(a, b)];
            re[(n + a, n + b)] = rr[0][(a, b)] - rr[3][(a, b)];
            im[(n + a, n + b)] = ii[0][(a, b)] - ii[3][(a, b)];
            re[(a, n + b)] = rr[1][(a, b)] + ii[2][(a, b)];
            im[(a, n + b)] = ii[1][(a, b)] - rr[2][(a, b)];
            re[(n + a, b)] = rr[1][(a, b)] - ii[2][(a, b)];
            im[(n + a, b)] = ii[1][(a, b)] + rr[2][(a, b)];
        }
    }
    (re, im)
}
fn setup(model: &serde_json::Value) -> (Vec<Atom>, Vec<ContractedGaussian>, ComplexHfOptions) {
    let mut atoms: Vec<_> = model["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            Atom::new(
                a["symbol"].as_str().unwrap(),
                std::array::from_fn(|k| value(&a["position"][k])),
            )
        })
        .collect();
    if !model["ecp"].is_null() {
        atoms[0].ecp = Some(std::sync::Arc::new(ecp::EcpDef {
            symbol: "He".into(),
            n_core: 0,
            l_max: 2,
            local: vec![ecp::EcpTerm {
                power: 2,
                exponent: 0.7,
                coefficient: 0.17,
            }],
            semilocal: vec![
                vec![ecp::EcpTerm {
                    power: 2,
                    exponent: 0.53,
                    coefficient: 0.11,
                }],
                vec![ecp::EcpTerm {
                    power: 2,
                    exponent: 0.61,
                    coefficient: -0.07,
                }],
            ],
        }));
    }
    let basis = if let Some(gbs) = model["gaussian94"].as_str() {
        build_basis(&atoms, &parse_gaussian94(gbs))
    } else {
        build_basis(
            &atoms,
            &get_builtin_basis(if model["basis_constant"] == "BASIS_6_31G_GBS" {
                "6-31g"
            } else {
                "sto-3g"
            })
            .unwrap(),
        )
    };
    let options = ComplexHfOptions {
        reference: match model["reference"].as_str().unwrap() {
            "rhf" => ComplexHfReference::Restricted,
            "uhf" => ComplexHfReference::Unrestricted,
            _ => ComplexHfReference::Generalized,
        },
        multiplicity: model["spin"].as_u64().unwrap() as usize + 1,
        gauge_origin: model["origin"]
            .as_array()
            .map(|_| std::array::from_fn(|k| value(&model["origin"][k]))),
        initial_density: if model["initial_real"].is_null() {
            None
        } else {
            Some((
                matrix(&model["initial_real"]),
                matrix(&model["initial_imag"]),
            ))
        },
        magnetic_field: std::array::from_fn(|k| value(&model["field"][k])),
        spin_zeeman: model["spin_zeeman"].as_bool().unwrap_or(false),
        energy_tol: 1e-14,
        residual_tol: 1e-12,
        density_tol: 1e-12,
        max_iterations: 300,
        ..Default::default()
    };
    (atoms, basis, options)
}

#[test]
fn complex_stationary_energies_all_magnetic_density_and_fock_components_match_pyscf() {
    let data = oracle();
    let mut maximum = [0f64; 4];
    let mut counts = [0; 3];
    for model in data["cases"].as_array().unwrap() {
        let (atoms, basis, options) = setup(model);
        let result = run_complex_hf_with_options(
            &atoms,
            &basis,
            model["charge"].as_i64().unwrap() as i32,
            &options,
        )
        .unwrap();
        let error = (result.total_energy - value(&model["e_total"])).abs();
        maximum[0] = maximum[0].max(error);
        assert!(error < 1e-8, "{}: {error:e}", model["name"]);
        assert!(
            (result.s_squared - value(&model["s_squared"])).abs() < 1e-8,
            "{} S2 {} vs {}",
            model["name"],
            result.s_squared,
            model["s_squared"]
        );
        let origin = options.gauge_origin.unwrap_or_else(|| {
            std::array::from_fn(|k| {
                atoms.iter().map(|a| a.position[k]).sum::<f64>() / atoms.len() as f64
            })
        });
        let (dia, orbital) =
            magnetic_one_electron_integrals(&basis, options.magnetic_field, origin).unwrap();
        for (actual, key) in [(&dia, "diamagnetic"), (&orbital, "orbital_imag")] {
            for (a, b) in actual.data.iter().zip(matrix(&model[key]).data) {
                let e = (a - b).abs();
                maximum[1] = maximum[1].max(e);
                assert!(e < 1e-10, "{} {key}: {e:e}", model["name"]);
                counts[0] += 1;
            }
        }
        let (pr, pi, fr, fi) = if model["seed"] == "triangle" {
            assert!(!options.spin_zeeman);
            let rotation = spin_frame(
                &result.spinor_density_real,
                &result.spinor_density_imag,
                &matrix(&model["density_real"]),
                &matrix(&model["density_imag"]),
            );
            let (pr, pi) = rotate_spin(
                &result.spinor_density_real,
                &result.spinor_density_imag,
                &rotation,
            );
            let (fr, fi) = rotate_spin(&result.fock_real, &result.fock_imag, &rotation);
            (pr, pi, fr, fi)
        } else {
            (
                result.spinor_density_real.clone(),
                result.spinor_density_imag.clone(),
                result.fock_real.clone(),
                result.fock_imag.clone(),
            )
        };
        for (actual, key) in [
            (&pr, "density_real"),
            (&pi, "density_imag"),
            (&fr, "fock_real"),
            (&fi, "fock_imag"),
        ] {
            for (a, b) in actual.data.iter().zip(matrix(&model[key]).data) {
                let e = (a - b).abs();
                maximum[2] = maximum[2].max(e);
                assert!(e < 1e-8, "{} {key}: {e:e}", model["name"]);
                counts[1] += 1;
            }
        }
        let eri = compute_eri_tensor(&basis);
        let (re, im) = build_complex_hf_fock(
            &matrix(&model["h_real"]),
            &matrix(&model["h_imag"]),
            &matrix(&model["arbitrary_density_real"]),
            &matrix(&model["arbitrary_density_imag"]),
            &eri,
        )
        .unwrap();
        for (actual, key) in [(&re, "arbitrary_fock_real"), (&im, "arbitrary_fock_imag")] {
            for (a, b) in actual.data.iter().zip(matrix(&model[key]).data) {
                let e = (a - b).abs();
                maximum[3] = maximum[3].max(e);
                assert!(e < 1e-10, "{} {key}: {e:e}", model["name"]);
                counts[2] += 1;
            }
        }
        assert!(
            result.residual_norm <= options.residual_tol
                && result.density_change <= options.density_tol
        );
        let spatial = hf_rs::scf::compute_one_electron_integrals(&atoms, &basis).0;
        let n = basis.len();
        let mut overlap = Matrix::zeros(2 * n, 2 * n);
        for spin in 0..2 {
            for p in 0..n {
                for q in 0..n {
                    overlap[(spin * n + p, spin * n + q)] = spatial[(p, q)];
                }
            }
        }
        let (cr, ci) = (&result.coefficients_real, &result.coefficients_imag);
        let real = cr
            .transpose()
            .dot(&overlap)
            .dot(cr)
            .add(&ci.transpose().dot(&overlap).dot(ci));
        let imag = cr
            .transpose()
            .dot(&overlap)
            .dot(ci)
            .sub(&ci.transpose().dot(&overlap).dot(cr));
        for p in 0..real.rows {
            for q in 0..real.cols {
                assert!((real[(p, q)] - if p == q { 1. } else { 0. }).abs() < 1e-10);
                assert!(imag[(p, q)].abs() < 1e-10);
            }
        }
        let frc = result.fock_real.dot(cr).sub(&result.fock_imag.dot(ci));
        let fic = result.fock_real.dot(ci).add(&result.fock_imag.dot(cr));
        let src = overlap.dot(cr);
        let sic = overlap.dot(ci);
        for p in 0..frc.rows {
            for q in 0..frc.cols {
                assert!(
                    (frc[(p, q)] - src[(p, q)] * result.orbital_energies[q]).abs() < 1e-9,
                    "{} orbital residual",
                    model["name"]
                );
                assert!((fic[(p, q)] - sic[(p, q)] * result.orbital_energies[q]).abs() < 1e-9);
            }
        }
    }
    println!(
        "Complex HF: nine energies; all magnetic/P-F/arbitrary-F components {counts:?}; maximum errors {maximum:?}"
    );
}

#[test]
fn every_complex_hf_nuclear_component_matches_independent_energies_including_ecp() {
    let data = oracle();
    let mut maximum = 0f64;
    let mut translation = 0f64;
    let mut count = 0;
    for model in data["cases"].as_array().unwrap() {
        let (atoms, basis, options) = setup(model);
        let result = run_complex_hf_with_options(
            &atoms,
            &basis,
            model["charge"].as_i64().unwrap() as i32,
            &options,
        )
        .unwrap();
        let actual = compute_complex_hf_gradient(&atoms, &basis, &result).unwrap();
        for (k, &g) in actual.iter().flatten().enumerate() {
            let e = (g - value(&model["gradient"][k])).abs();
            maximum = maximum.max(e);
            count += 1;
            assert!(
                e < 1e-6,
                "{} coordinate {k}: {e:e}; actual {g:e}, oracle {}",
                model["name"],
                model["gradient"][k]
            );
        }
        for axis in 0..3 {
            let sum = actual.iter().map(|g| g[axis]).sum::<f64>().abs();
            if options.gauge_origin.is_none() {
                translation = translation.max(sum);
                assert!(sum < 1e-9, "{} translation {sum:e}", model["name"]);
            }
        }
    }
    assert_eq!(count, 63);
    println!(
        "Complex HF nuclear: all {count} components, error {maximum:e}, translation {translation:e}"
    );
}

#[test]
fn complex_hf_rejects_invalid_controls_and_truncated_gradient_spaces() {
    let data = oracle();
    let model = &data["cases"][0];
    let (atoms, mut basis, mut options) = setup(model);
    options.max_work_bytes = 1;
    assert!(run_complex_hf_with_options(&atoms, &basis, 0, &options).is_err());
    options = setup(model).2;
    assert!(run_complex_hf_with_options(&atoms, &basis, 1, &options).is_err());
    options.max_iterations = 1;
    assert!(run_complex_hf_with_options(&atoms, &basis, 0, &options).is_err());
    options = setup(model).2;
    options.initial_density = Some((Matrix::zeros(4, 4), Matrix::zeros(4, 4)));
    assert!(run_complex_hf_with_options(&atoms, &basis, 0, &options).is_err());
    options = setup(model).2;
    basis.push(basis[0].clone());
    let result = run_complex_hf_with_options(&atoms, &basis, 0, &options).unwrap();
    assert!((result.total_energy - value(&model["e_total"])).abs() < 1e-10);
    assert!(compute_complex_hf_gradient(&atoms, &basis, &result).is_err());
    options.magnetic_field[0] = f64::NAN;
    assert!(run_complex_hf_with_options(&atoms, &basis, 0, &options).is_err());
}

#[test]
fn all_eight_symmetry_classes_are_invariant_under_su2_and_real_ao_changes() {
    use hf_rs::hf_variants::symmetry::classify_complex_hf_symmetry;
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/hf_symmetry_reference.json")).unwrap();
    let mut count = 0;
    for case in data["cases"].as_array().unwrap() {
        let result = classify_complex_hf_symmetry(
            &matrix(&case["real"]),
            &matrix(&case["imag"]),
            &matrix(&case["overlap"]),
            1e-8,
        )
        .unwrap();
        assert_eq!(
            format!("{:?}", result.class),
            case["expected"].as_str().unwrap(),
            "frame {}, metric {}",
            case["spin_frame"],
            case["nonorthogonal"]
        );
        assert_eq!(
            result.time_reversal_invariant,
            matches!(
                result.class,
                FukutomeClass::TICS | FukutomeClass::ASCW | FukutomeClass::TSCW
            )
        );
        count += 1;
    }
    assert_eq!(count, 64);
    assert!(
        classify_complex_hf_symmetry(
            &Matrix::identity(2).scale(0.5),
            &Matrix::zeros(2, 2),
            &Matrix::identity(1),
            1e-8
        )
        .is_err()
    );
    println!("All eight Fukutome determinant classes: {count} spin-frame / AO-metric cases");
}

#[test]
fn fixed_gauge_transports_with_translated_geometry_and_accepts_a_valid_restart() {
    let data = oracle();
    let model = &data["cases"][8];
    let (mut atoms, mut basis, mut options) = setup(model);
    let result = run_complex_hf_with_options(&atoms, &basis, 0, &options).unwrap();
    let gradient = compute_complex_hf_gradient(&atoms, &basis, &result).unwrap();
    let shift = [0.31, -0.27, 0.19];
    for a in &mut atoms {
        for (k, &delta) in shift.iter().enumerate() {
            a.position[k] += delta;
        }
    }
    for b in &mut basis {
        for (k, &delta) in shift.iter().enumerate() {
            b.center[k] += delta;
        }
    }
    for (k, &delta) in shift.iter().enumerate() {
        options.gauge_origin.as_mut().unwrap()[k] += delta;
    }
    options.initial_density = Some((
        result.spinor_density_real.clone(),
        result.spinor_density_imag.clone(),
    ));
    let moved = run_complex_hf_with_options(&atoms, &basis, 0, &options).unwrap();
    assert!((moved.total_energy - result.total_energy).abs() < 1e-10);
    let actual = compute_complex_hf_gradient(&atoms, &basis, &moved).unwrap();
    for (a, b) in actual.iter().flatten().zip(gradient.iter().flatten()) {
        assert!((a - b).abs() < 1e-10);
    }
    assert_eq!(HFVariantType::parse("cuhf").unwrap(), HFVariantType::ComplexUHF);
    assert_eq!(HFVariantType::parse("cghf").unwrap(), HFVariantType::ComplexGHF);
    assert_eq!(HFVariantType::parse("crhf").unwrap(), HFVariantType::ComplexHF);
}
