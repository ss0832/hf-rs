use hf_rs::ao::{BasisKind, spherical_transform};
use hf_rs::hf_derivatives::HfNuclearDerivativeContext;
use hf_rs::scf::run_scf_in_space;
use hf_rs::{Atom, Matrix, build_basis, parse_gaussian94, run_scf};

#[test]
fn real_solid_harmonics_through_k_match_pyscf_all_coefficients() {
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/spherical_reference.json")).unwrap();
    for entry in oracle["transforms"].as_array().unwrap() {
        let l = entry["l"].as_u64().unwrap() as usize;
        let map = parse_gaussian94(&format!(
            "He 0\n{} 1 1\n0.7 1\n****",
            hf_rs::basis::l_to_shell_type(l)
        ));
        let basis = build_basis(&[Atom::new("He", [0.0; 3])], &map);
        let t = spherical_transform(&basis).unwrap();
        for (i, cg) in basis.iter().enumerate() {
            let row = entry["powers"]
                .as_array()
                .unwrap()
                .iter()
                .position(|v| (0..3).all(|k| v[k].as_u64().unwrap() as usize == cg.l[k]))
                .unwrap();
            for j in 0..t.cols {
                let expected = entry["matrix"][row][j].as_f64().unwrap();
                assert!(
                    (t[(i, j)] - expected).abs() < 2e-12,
                    "l={l} ({i},{j}): {} vs {expected}",
                    t[(i, j)]
                );
            }
        }
        let mut s = Matrix::zeros(basis.len(), basis.len());
        for i in 0..basis.len() {
            for j in 0..basis.len() {
                s[(i, j)] = hf_rs::integrals::contracted_overlap(&basis[i], &basis[j]);
            }
        }
        assert!(
            t.transpose()
                .dot(&s)
                .dot(&t)
                .rmsd(&Matrix::identity(2 * l + 1))
                < 2e-13
        );
    }
}

#[test]
fn spherical_scf_gradient_and_hessian_match_pyscf() {
    let atoms = [
        Atom::new("He", [-0.1, 0.2, 0.3]),
        Atom::new("H", [0.4, -0.3, 1.8]),
    ];
    let map = parse_gaussian94("He 0\nS 1 1\n1.4 1\nD 1 1\n0.6 1\n****\nH 0\nS 1 1\n1.1 1\n****");
    let basis = build_basis(&atoms, &map);
    let hf = run_scf_in_space(
        &atoms,
        &basis,
        1,
        150,
        1e-14,
        hf_rs::diis::ConvAlgo::Cdiis,
        None,
        BasisKind::Spherical,
    )
    .unwrap();
    assert!(hf.converged);
    assert_eq!((hf.mo_coefficients.rows, hf.mo_coefficients.cols), (8, 7));
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/spherical_reference.json")).unwrap();
    let expected = &oracle["systems"]["heh_model"];
    assert!((hf.total_energy - expected["energy"].as_f64().unwrap()).abs() < 1e-10);
    let g = hf_rs::gradient::try_compute_rhf_gradient(&atoms, &basis, &hf).unwrap();
    let ctx = HfNuclearDerivativeContext::from_rhf(&atoms, &basis, &hf).unwrap();
    for i in 0..6 {
        let reference = expected["gradient"][i / 3][i % 3].as_f64().unwrap();
        assert!((g[i / 3][i % 3] - reference).abs() < 1e-8);
        assert!((ctx.coordinate_derivative(&[i]).unwrap() - reference).abs() < 1e-8);
        for j in 0..6 {
            let reference = expected["hessian"][i][j].as_f64().unwrap();
            assert!(
                (ctx.coordinate_derivative(&[i, j]).unwrap() - reference).abs() < 1e-7,
                "({i},{j})"
            );
        }
    }
    // Rectangular full MO integral transformation also preserves the occupied HF energy.
    let mo = hf_rs::mo_integrals::transform_eri_ao_to_mo(&hf.mo_coefficients, &hf.eri);
    assert_eq!(mo.n_basis, 7);
    let hm = hf
        .mo_coefficients
        .transpose()
        .dot(&hf.core_hamiltonian)
        .dot(&hf.mo_coefficients);
    let energy = 2.0 * hm[(0, 0)] + mo.get(0, 0, 0, 0) + hf.nuclear_repulsion_energy;
    assert!((energy - hf.total_energy).abs() < 1e-11);
}

#[test]
fn duplicate_ao_is_dropped_and_derivatives_report_linear_dependence() {
    let atoms = [Atom::new("H", [0.0; 3]), Atom::new("H", [0.2, 0.1, 1.4])];
    let basis = build_basis(&atoms, &parse_gaussian94(hf_rs::basis::STO_3G_GBS));
    let original = run_scf(&atoms, &basis, 100, 1e-13).unwrap();
    let doubled = basis
        .iter()
        .chain(basis.iter())
        .cloned()
        .collect::<Vec<_>>();
    let hf = run_scf(&atoms, &doubled, 100, 1e-13).unwrap();
    assert!(hf.converged);
    assert_eq!((hf.mo_coefficients.rows, hf.mo_coefficients.cols), (4, 2));
    assert!((original.total_energy - hf.total_energy).abs() < 1e-11);
    let s = hf
        .mo_coefficients
        .transpose()
        .dot(&hf.overlap_matrix)
        .dot(&hf.mo_coefficients);
    assert!(s.rmsd(&Matrix::identity(2)) < 2e-13);
    assert!(matches!(
        hf_rs::gradient::try_compute_rhf_gradient(&atoms, &doubled, &hf),
        Err(hf_rs::error::HfError::LinearDependence { n_ao: 4, n_mo: 2 })
    ));
    assert!(matches!(
        hf_rs::hessian::try_compute_rhf_analytical_hessian(&atoms, &doubled, &hf),
        Err(hf_rs::error::HfError::LinearDependence { .. })
    ));
    assert!(matches!(
        HfNuclearDerivativeContext::from_rhf(&atoms, &doubled, &hf),
        Err(hf_rs::error::HfError::LinearDependence { .. })
    ));
}

#[test]
fn water_ccpvdz_spherical_energy_and_all_gradient_components_match_pyscf() {
    let atoms = [
        Atom::new("O", [0.0; 3]),
        Atom::new("H", [1.43, 0.2, 1.1]),
        Atom::new("H", [-1.2, -0.3, 1.3]),
    ];
    let basis = build_basis(&atoms, &hf_rs::get_builtin_basis("cc-pvdz").unwrap());
    let hf = run_scf_in_space(
        &atoms,
        &basis,
        0,
        150,
        1e-13,
        hf_rs::diis::ConvAlgo::Cdiis,
        None,
        BasisKind::Spherical,
    )
    .unwrap();
    assert!(hf.converged);
    assert_eq!(hf.mo_coefficients.cols, 24);
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/spherical_reference.json")).unwrap();
    let expected = &oracle["systems"]["water_ccpvdz"];
    assert!((hf.total_energy - expected["energy"].as_f64().unwrap()).abs() < 1e-9);
    let g = hf_rs::gradient::try_compute_rhf_gradient(&atoms, &basis, &hf).unwrap();
    for i in 0..3 {
        for k in 0..3 {
            assert!((g[i][k] - expected["gradient"][i][k].as_f64().unwrap()).abs() < 1e-8);
        }
    }
}
