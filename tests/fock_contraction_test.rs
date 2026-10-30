use hf_rs::{Matrix, scf::*};

#[test]
fn unique_quartet_fock_matches_explicit_ordered_contractions_for_all_elements() {
    for n in 1..=7 {
        let np = n * (n + 1) / 2;
        let eri = (0..np * (np + 1) / 2)
            .map(|i| ((i + 1) as f64 * 0.37).sin())
            .collect::<Vec<_>>();
        let mut h = Matrix::zeros(n, n);
        let mut pa = h.clone();
        let mut pb = h.clone();
        for i in 0..n {
            for j in 0..=i {
                h[(i, j)] = ((i + j + 1) as f64).cos();
                h[(j, i)] = h[(i, j)];
                pa[(i, j)] = ((i * 3 + j + 1) as f64).sin();
                pa[(j, i)] = pa[(i, j)];
                pb[(i, j)] = ((i + j * 2 + 1) as f64).cos();
                pb[(j, i)] = pb[(i, j)];
            }
        }
        let restricted = build_fock_matrix(&h, &pa, &eri, n);
        let (alpha, beta) = build_uhf_fock_matrices(&h, &pa, &pb, &eri, n);
        for a in 0..n {
            for b in 0..n {
                let mut r = h[(a, b)];
                let mut u = [r, r];
                for c in 0..n {
                    for d in 0..n {
                        let coulomb = eri[eri_index(a, b, c, d)];
                        let exchange = eri[eri_index(a, c, b, d)];
                        r += pa[(c, d)] * (coulomb - 0.5 * exchange);
                        u[0] += (pa[(c, d)] + pb[(c, d)]) * coulomb - pa[(c, d)] * exchange;
                        u[1] += (pa[(c, d)] + pb[(c, d)]) * coulomb - pb[(c, d)] * exchange;
                    }
                }
                assert!((restricted[(a, b)] - r).abs() < 1e-12, "n={n} R({a},{b})");
                assert!((alpha[(a, b)] - u[0]).abs() < 1e-12, "n={n} A({a},{b})");
                assert!((beta[(a, b)] - u[1]).abs() < 1e-12, "n={n} B({a},{b})");
            }
        }
    }
}
