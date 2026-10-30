use hf_rs::Matrix;
use hf_rs::local::transport::{orthonormal_frame, orthonormal_frame_response};
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
#[test]
fn every_analytic_transport_response_component_matches_independent_sylvester_solution() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("data/local_transport_reference.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let m = matrix(&case["matrix"]);
        let frame = orthonormal_frame(&m).unwrap();
        let expected = matrix(&case["frame"]);
        for (a, b) in frame.data.iter().zip(expected.data) {
            assert!((a - b).abs() < 1e-12);
        }
        for coordinate in 0..m.data.len() {
            let mut direction = Matrix::zeros(m.rows, m.cols);
            direction.data[coordinate] = 1.;
            let actual = orthonormal_frame_response(&m, &direction).unwrap();
            let expected = matrix(&case["response"][coordinate]);
            for (index, (&a, b)) in actual.data.iter().zip(expected.data).enumerate() {
                assert!(
                    (a - b).abs() < 1e-11,
                    "input {coordinate}, output {index}: {a} vs {b}"
                );
            }
            let tangent = frame
                .transpose()
                .dot(&actual)
                .add(&actual.transpose().dot(&frame));
            assert!(tangent.data.iter().all(|v| v.abs() < 1e-11));
        }
    }
    let singular = Matrix::from_vec(3, 2, vec![1., 1., 2., 2., 3., 3.]);
    assert!(orthonormal_frame(&singular).is_err());
    assert!(orthonormal_frame_response(&singular, &singular).is_err());
}
