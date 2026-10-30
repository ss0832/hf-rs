//! Generated files are independently read and evaluated by reference/check_molden.py.
use hf_rs::ao::{AoSpace, BasisKind};
use hf_rs::*;
use std::sync::Arc;

#[test]
fn export_cartesian_and_spherical_orbitals_through_g_for_independent_reader() {
    let atoms = [
        Atom::new("He", [-0.1, 0.2, 0.3]),
        Atom::new("H", [0.4, -0.3, 1.8]),
    ];
    let map = parse_gaussian94(
        "He 0\nS 2 1\n1.4 1.7\n0.8 0.2\nP 1 1\n0.9 1\nD 1 1\n0.6 1\nF 1 1\n0.5 1\nG 1 1\n0.4 1\n****\nH 0\nS 1 1\n1.1 1\n****",
    );
    let basis = build_basis(&atoms, &map);
    let small = build_basis(
        &atoms,
        &parse_gaussian94("He 0\nS 1 1\n1.4 1\n****\nH 0\nS 1 1\n1.1 1\n****"),
    );
    let scaffold = run_scf_with_charge(&atoms, &small, 1, 100, 1e-12).unwrap();
    let points = [
        [0.21, -0.47, 0.81],
        [-0.72, 0.13, 1.23],
        [1.14, 0.84, -0.26],
    ];
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/molden_cases");
    std::fs::create_dir_all(&directory).unwrap();
    let mut cases = serde_json::Map::new();
    for (name, kind) in [
        ("cartesian", BasisKind::Cartesian),
        ("spherical", BasisKind::Spherical),
    ] {
        let space = AoSpace::new(&basis, kind).unwrap();
        let mut physical = Matrix::zeros(space.cart_to_ao.cols, 3);
        for i in 0..physical.rows {
            for j in 0..3 {
                physical[(i, j)] = ((i + 2 * j + 1) as f64 * 0.31).sin();
            }
        }
        let c = space.cart_to_ao.dot(&physical);
        let mut hf = scaffold.clone();
        hf.ao_space = Some(Arc::new(space));
        hf.mo_coefficients = c.clone();
        hf.orbital_energies = vec![-0.8, 0.1, 0.5];
        let molden = molden::try_generate_molden_string_rhf(&atoms, &map, &hf).unwrap();
        assert_eq!(molden.contains("[5D7F]"), kind == BasisKind::Spherical);
        std::fs::write(directory.join(format!("{name}.molden")), molden).unwrap();
        let values = points
            .iter()
            .map(|point| {
                (0..3)
                    .map(|mo| {
                        basis
                            .iter()
                            .enumerate()
                            .map(|(i, cg)| {
                                let delta =
                                    std::array::from_fn::<_, 3, _>(|k| point[k] - cg.center[k]);
                                let r2 = delta.iter().map(|x| x * x).sum::<f64>();
                                let polynomial = (0..3)
                                    .map(|k| delta[k].powi(cg.l[k] as i32))
                                    .product::<f64>();
                                polynomial
                                    * cg.primitives
                                        .iter()
                                        .map(|p| p.coefficient * (-p.exponent * r2).exp())
                                        .sum::<f64>()
                                    * c[(i, mo)]
                            })
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        cases.insert(
            name.into(),
            serde_json::json!({"values":values,"points_bohr":points}),
        );
    }
    std::fs::write(
        directory.join("reference.json"),
        serde_json::to_string_pretty(&cases).unwrap(),
    )
    .unwrap();
}
