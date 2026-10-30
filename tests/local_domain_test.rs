use hf_rs::local::domains::{differential_overlaps, modified_mulliken_populations};
use hf_rs::{Atom, Matrix, build_basis, get_builtin_basis};
fn matrix(v: &serde_json::Value) -> Matrix {
    let rows = v.as_array().unwrap();
    Matrix::from_vec(
        rows.len(),
        rows[0].as_array().unwrap().len(),
        rows.iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect(),
    )
}
fn maximum(a: &Matrix, b: &Matrix) -> f64 {
    assert_eq!((a.rows, a.cols), (b.rows, b.cols));
    a.data
        .iter()
        .zip(&b.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max)
}
#[test]
fn every_doi_and_modified_mulliken_component_matches_independent_pyscf_grid() {
    let atoms = [
        Atom::new("O", [0.13, -0.21, 0.31]),
        Atom::new("H", [0.23, 1.22, -0.77]),
        Atom::new("H", [-0.08, -1.64, -0.83]),
    ];
    let basis = build_basis(&atoms, &get_builtin_basis("sto-3g").unwrap());
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/local_domain_reference.json")).unwrap();
    let localization: serde_json::Value =
        serde_json::from_str(include_str!("data/localization_reference.json")).unwrap();
    let c = matrix(&data["coefficients"]);
    let s = matrix(&localization["overlap"]);
    let q = matrix(&data["pao_projector"]);
    let coarse = differential_overlaps(&atoms, &basis, &c, &q, &s, 64, 24).unwrap();
    let fine = differential_overlaps(&atoms, &basis, &c, &q, &s, 96, 40).unwrap();
    let eo = maximum(&fine.occupied, &matrix(&data["occupied"]));
    let ep = maximum(&fine.occupied_pao, &matrix(&data["occupied_pao"]));
    let convergence = maximum(&coarse.occupied, &fine.occupied)
        .max(maximum(&coarse.occupied_pao, &fine.occupied_pao));
    eprintln!("DOI reference errors occupied={eo:e}, PAO={ep:e}; grid refinement={convergence:e}");
    assert!(eo < 1e-6 && ep < 1e-6 && convergence < 1e-6);
    let pop = modified_mulliken_populations(&c, &s, &basis, atoms.len()).unwrap();
    assert!(maximum(&pop, &matrix(&data["mulliken"])) < 1e-12);
    for i in 0..c.cols {
        assert!(((0..atoms.len()).map(|a| pop[(i, a)]).sum::<f64>() - 1.).abs() < 1e-12);
    }
}
