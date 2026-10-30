use hf_rs::df::{DfTensor, coulomb_metric, three_center_integrals};
use hf_rs::{Atom, build_basis, get_builtin_basis, parse_gaussian94, run_scf};

#[test]
fn df_s_gaussian_metric_matches_closed_form_coulomb_integral() {
    let atoms = [Atom::new("H", [0.1, -0.2, 0.3])];
    let map = parse_gaussian94("H 0\nS 1 1\n0.7 1\nS 1 1\n1.2 1\n****");
    let aux = build_basis(&atoms, &map);
    let metric = coulomb_metric(&aux);
    for i in 0..2 {
        for j in 0..2 {
            let a = aux[i].primitives[0].exponent;
            let b = aux[j].primitives[0].exponent;
            let expected = 2.0 * std::f64::consts::PI.powf(2.5) / (a * b * (a + b).sqrt())
                * aux[i].primitives[0].coefficient
                * aux[j].primitives[0].coefficient;
            assert!((metric[(i, j)] - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn all_df_integrals_and_ri_mp2_match_pyscf() {
    let atoms = [
        Atom::new("O", [0.0; 3]),
        Atom::new("H", [1.43, 0.2, 1.1]),
        Atom::new("H", [-1.2, -0.3, 1.3]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let aux = build_basis(
        &atoms,
        &hf_rs::get_builtin_aux_basis("def2-svp-ri").unwrap(),
    );
    let oracle: serde_json::Value =
        serde_json::from_str(include_str!("data/df_reference.json")).unwrap();
    assert_eq!(aux.len(), oracle["n_aux"].as_u64().unwrap() as usize);
    let metric = coulomb_metric(&aux);
    let three = three_center_integrals(&basis, &aux);
    for p in 0..aux.len() {
        for q in 0..aux.len() {
            let expected = oracle["metric"][p][q].as_f64().unwrap();
            assert!(
                (metric[(p, q)] - expected).abs() < 2e-9,
                "metric ({p},{q}): {} vs {expected}",
                metric[(p, q)]
            );
        }
        for i in 0..basis.len() {
            for j in 0..=i {
                let expected = oracle["three"][i][j][p].as_f64().unwrap();
                assert!(
                    (three[(p, hf_rs::scf::compound_index(i, j))] - expected).abs() < 2e-10,
                    "three ({i},{j},{p})"
                );
            }
        }
    }
    let hf = run_scf(&atoms, &basis, 150, 1e-13).unwrap();
    assert!(hf.converged);
    let fitted = DfTensor::build(&basis, &aux, 1e-10)
        .unwrap()
        .transform(&hf.mo_coefficients)
        .unwrap();
    let mp2 = fitted
        .mp2_energy(&hf.orbital_energies, hf.n_occ, 0)
        .unwrap();
    assert!(
        (mp2 - oracle["ri_mp2"].as_f64().unwrap()).abs() < 1e-9,
        "RI-MP2 {mp2}"
    );
}
