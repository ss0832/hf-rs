use hf_rs::post_hf::cc::ccsd::CCSDOptions;
use hf_rs::post_hf::cc::ccsd_t::compute_ccsd_t_correction;
use hf_rs::*;
#[test]
fn nonzero_canonical_triples_match_pyscf_and_singular_gaps_fail() {
    let atoms = [
        Atom::new("O", [0.13, -0.21, 0.31]),
        Atom::new("H", [0.23, 1.22, -0.77]),
        Atom::new("H", [-0.08, -1.64, -0.83]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let mut hf = run_scf(&atoms, &basis, 150, 1e-14).unwrap();
    let cc = CCSD::new(&hf)
        .with_options(CCSDOptions {
            max_iterations: 300,
            energy_tol: 1e-13,
            residual_tol: 1e-11,
            diis_subspace: 8,
        })
        .compute()
        .unwrap();
    assert!(cc.converged);
    let result = compute_ccsd_t_correction(&hf, &cc).unwrap();
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/ccsd_triples_reference.json")).unwrap();
    assert!((cc.correlation_energy - oracle["e_ccsd"].as_f64().unwrap()).abs() < 1e-10);
    assert!(
        (result.triples_correlation_energy - oracle["e_triples"].as_f64().unwrap()).abs() < 1e-10
    );
    assert!(result.triples_correlation_energy.abs() > 1e-5);
    hf.orbital_energies[hf.n_occ] = hf.orbital_energies[hf.n_occ - 1];
    assert!(compute_ccsd_t_correction(&hf, &cc).is_err());
}
