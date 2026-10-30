use hf_rs::*;

fn h2() -> (Vec<Atom>, Vec<ContractedGaussian>) {
    let atoms = vec![Atom::new("H", [0.0; 3]), Atom::new("H", [0.2, -0.1, 3.5])];
    let basis = build_basis(&atoms, &parse_gaussian94(basis::STO_3G_GBS));
    (atoms, basis)
}

#[test]
fn mom_reports_stationary_consistent_density_and_iteration_limit() {
    let (atoms, basis) = h2();
    let result = rohf::run_scf_mom(&atoms, &basis, 0, &[1], 200, 1e-13).unwrap();
    assert!(result.converged);
    assert!(
        result
            .density_matrix
            .rmsd(&result.mo_coefficients.occupied_density(1, 2.0))
            < 1e-14
    );
    let f = scf::build_fock_matrix(
        &result.core_hamiltonian,
        &result.density_matrix,
        &result.eri,
        basis.len(),
    );
    assert!(result.fock_matrix.rmsd(&f) < 1e-14);
    assert!(
        diis::max_abs_element(&diis::compute_diis_error(
            &f,
            &result.density_matrix,
            &result.overlap_matrix
        )) < 1e-9
    );
    let limited = rohf::run_scf_mom(&atoms, &basis, 0, &[1], 1, 1e-13).unwrap();
    assert!(!limited.converged);
    assert!(rohf::run_scf_mom(&atoms, &basis, 0, &[2], 20, 1e-10).is_err());
}

#[test]
fn fermi_odd_electron_population_and_fixed_point_are_consistent() {
    let (atoms, basis) = h2();
    for charge in [0, 1] {
        let result =
            rohf::run_scf_fermi_smearing(&atoms, &basis, charge, 40000.0, 200, 1e-13).unwrap();
        assert!(result.hf.converged);
        let hf = &result.hf;
        let electrons = 2.0 - charge as f64;
        assert!((result.occupations.iter().sum::<f64>() - electrons).abs() < 1e-12);
        assert!((hf.density_matrix.dot(&hf.overlap_matrix).trace() - electrons).abs() < 1e-12);
        let mut weighted = hf.mo_coefficients.clone();
        for i in 0..weighted.rows {
            for j in 0..weighted.cols {
                weighted[(i, j)] *= result.occupations[j].sqrt();
            }
        }
        assert!(hf.density_matrix.rmsd(&weighted.dot(&weighted.transpose())) < 1e-14);
        let f = scf::build_fock_matrix(
            &hf.core_hamiltonian,
            &hf.density_matrix,
            &hf.eri,
            basis.len(),
        );
        assert!(hf.fock_matrix.rmsd(&f) < 1e-14);
        let eigen = hf
            .mo_coefficients
            .transpose()
            .dot(&f)
            .dot(&hf.mo_coefficients);
        for i in 0..eigen.rows {
            assert!((eigen[(i, i)] - hf.orbital_energies[i]).abs() < 1e-8);
            for j in 0..i {
                assert!(eigen[(i, j)].abs() < 1e-8);
            }
            let arg = (eigen[(i, i)] - result.chemical_potential) / (40000.0 / 315775.02480407);
            let expected = 2.0 / (1.0 + arg.exp());
            assert!((expected - result.occupations[i]).abs() < 1e-8);
        }
        assert!(
            (result.free_energy - (hf.total_energy - 40000.0 * result.electronic_entropy)).abs()
                < 1e-14
        );
    }
    let limited = rohf::run_scf_fermi_smearing(&atoms, &basis, 0, 40000.0, 1, 1e-13).unwrap();
    assert!(!limited.hf.converged);
    assert!(rohf::run_scf_fermi_smearing(&atoms, &basis, 0, -1.0, 10, 1e-10).is_err());
    assert!(rohf::run_scf_fermi_smearing(&atoms, &basis, -3, 300.0, 10, 1e-10).is_err());
}

#[test]
fn qrhf_rejects_unconverged_reference() {
    let (atoms, basis) = h2();
    assert!(rohf::run_qrhf(&atoms, &basis, 1, 2, 1, 1e-14).is_err());
    assert!(
        rohf::run_qrhf(&atoms, &basis, 1, 2, 100, 1e-13)
            .unwrap()
            .converged
    );
}
