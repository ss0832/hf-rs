use hf_rs::basis::*;

#[test]
fn bse_general_contractions_preserve_every_column() {
    let json = r#"{"elements":{"1":{"electron_shells":[
      {"angular_momentum":[0],"exponents":["2.0","0.4"],
       "coefficients":[["0.7","0.3"],["-0.2","0.8"]]},
      {"angular_momentum":[0,1],"exponents":["0.2"],
       "coefficients":[["1"],["2"],["3"],["4"]]}
    ]}}}"#;
    let map = parse_bse_json(json).unwrap();
    assert_eq!(map["H"].len(), 4);
    assert_eq!(map["H"][1].coefficients, [-0.2, 0.8]);
    assert_eq!(map["H"][3].p_coefficients.as_ref().unwrap(), &[4.0]);
    assert_eq!(build_basis(&[Atom::new("H", [0.0; 3])], &map).len(), 10);
}

#[test]
fn gaussian94_scale_and_general_contractions_match_explicit_basis() {
    let scaled = try_parse_gaussian94("H 0\nS 2 2.0\n1D0 0.7 -0.2\n0.1 0.3 0.8\n****").unwrap();
    let explicit =
        try_parse_gaussian94("H 0\nS 2 1\n4 0.7\n0.4 0.3\nS 2 1\n4 -0.2\n0.4 0.8\n****").unwrap();
    let atoms = [Atom::new("H", [0.2, -0.1, 0.3])];
    let a = build_basis(&atoms, &scaled);
    let b = build_basis(&atoms, &explicit);
    assert_eq!(a.len(), 2);
    for (ga, gb) in a.iter().zip(&b) {
        assert_eq!(ga.l, gb.l);
        for (pa, pb) in ga.primitives.iter().zip(&gb.primitives) {
            assert_eq!(pa.exponent, pb.exponent);
            assert_eq!(pa.coefficient, pb.coefficient);
        }
    }
}

#[test]
fn malformed_basis_is_rejected_without_panicking_or_zero_substitution() {
    for text in [
        "H 0\nS 2 1\n1 1",
        "H 0\nSP 1 1\n1 1",
        "H 0\nS 1 1\nNaN 1",
        "H 0\nS 1 -1\n1 1",
        "H 0\nS 1 1\n0 1",
    ] {
        assert!(try_parse_gaussian94(text).is_err(), "{text}");
    }
    for value in ["null", "\"bad\"", "\"NaN\"", "-1"] {
        let json = format!(
            r#"{{"elements":{{"1":{{"electron_shells":[{{"angular_momentum":[0],"exponents":[{value}],"coefficients":[["1"]]}}]}}}}}}"#
        );
        assert!(parse_bse_json(&json).is_err(), "{value}");
    }
}

#[test]
fn checked_basis_construction_rejects_missing_shells_and_zero_contractions() {
    let atoms=[Atom::new("H",[0.0;3])];
    assert!(try_build_basis(&atoms,&BasisSetMap::new()).is_err());
    let map=parse_gaussian94("H 0\nS 1 1\n0.8 0\n****");
    assert!(try_build_basis(&atoms,&map).is_err());
}
