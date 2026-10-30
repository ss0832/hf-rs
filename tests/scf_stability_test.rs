use hf_rs::scf::{build_uhf_fock_matrices, rhf_stability_matrix, rotate_rhf_orbitals};
use hf_rs::{Atom, Matrix, build_basis, parse_gaussian94, run_scf};

#[test]
fn complete_singlet_and_triplet_hessians_match_independent_energy_curvature() {
    let atoms = [
        Atom::new("Li", [-0.2, 0.1, 0.0]),
        Atom::new("H", [0.4, -0.3, 2.9]),
    ];
    let basis = build_basis(&atoms, &parse_gaussian94(hf_rs::basis::STO_3G_GBS));
    let hf = run_scf(&atoms, &basis, 150, 1e-14).unwrap();
    assert!(hf.converged);
    let dim = hf.n_occ * (hf.mo_coefficients.cols - hf.n_occ);
    for triplet in [false, true] {
        // This evaluates the AO HF functional directly, without using the stability matrix.
        let energy = |v: &[f64]| {
            let (ca, _) = rotate_rhf_orbitals(&hf, v, 1.0);
            let (cb, _) = rotate_rhf_orbitals(&hf, v, if triplet { -1.0 } else { 1.0 });
            let pa = ca.occupied_density(hf.n_occ, 1.0);
            let pb = cb.occupied_density(hf.n_occ, 1.0);
            let (fa, fb) =
                build_uhf_fock_matrices(&hf.core_hamiltonian, &pa, &pb, &hf.eri, basis.len());
            0.5 * pa
                .data
                .iter()
                .zip(&fa.data)
                .zip(&hf.core_hamiltonian.data)
                .map(|((&p, &f), &h)| p * (h + f))
                .sum::<f64>()
                + 0.5
                    * pb.data
                        .iter()
                        .zip(&fb.data)
                        .zip(&hf.core_hamiltonian.data)
                        .map(|((&p, &f), &h)| p * (h + f))
                        .sum::<f64>()
        };
        let matrix = rhf_stability_matrix(&hf, 0, triplet); // actual_n_occ fallback
        assert_eq!(matrix.rows, dim);
        for i in 0..dim {
            for j in 0..dim {
                let curvature = |h: f64| {
                    let mut pp = vec![0.0; dim];
                    pp[i] += h;
                    pp[j] += h;
                    let mut pm = vec![0.0; dim];
                    pm[i] += h;
                    pm[j] -= h;
                    let mut mp = vec![0.0; dim];
                    mp[i] -= h;
                    mp[j] += h;
                    let mut mm = vec![0.0; dim];
                    mm[i] -= h;
                    mm[j] -= h;
                    (energy(&pp) - energy(&pm) - energy(&mp) + energy(&mm)) / (4.0 * h * h)
                };
                let fd = (4.0 * curvature(0.001) - curvature(0.002)) / 3.0;
                assert!(
                    (4.0 * matrix[(i, j)] - fd).abs() < 2e-7,
                    "triplet={triplet} ({i},{j}): analytical={} FD={fd}",
                    4.0 * matrix[(i, j)]
                );
            }
        }
    }
}

#[test]
fn coupled_orbital_rotation_preserves_metric_and_has_group_inverse() {
    let atoms = [Atom::new("Li", [0.0; 3]), Atom::new("H", [0.3, -0.2, 3.0])];
    let basis = build_basis(&atoms, &parse_gaussian94(hf_rs::basis::STO_3G_GBS));
    let mut hf = run_scf(&atoms, &basis, 100, 1e-13).unwrap();
    let initial = hf.mo_coefficients.clone();
    let v = (0..hf.n_occ * (initial.cols - hf.n_occ))
        .map(|i| (i as f64 + 0.4).sin())
        .collect::<Vec<_>>();
    let (c, p) = rotate_rhf_orbitals(&hf, &v, 0.7);
    let metric = c.transpose().dot(&hf.overlap_matrix).dot(&c);
    assert!(metric.rmsd(&Matrix::identity(c.cols)) < 2e-13);
    assert!((p.dot(&hf.overlap_matrix).trace() - 2.0 * hf.n_occ as f64).abs() < 1e-12);
    hf.mo_coefficients = c;
    let (back, _) = rotate_rhf_orbitals(&hf, &v, -0.7);
    assert!(initial.rmsd(&back) < 2e-13);
}
