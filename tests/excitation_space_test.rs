use hf_rs::fci::excitations::{cas_references, singles_doubles};
use hf_rs::fci::{DeterminantSpace, SpinString};
use hf_rs::options::ActiveSpace;
use std::collections::BTreeSet;

#[test]
fn every_generated_determinant_matches_brute_force_excitation_rank() {
    for (n, na, nb) in [(5, 2, 2), (6, 3, 2), (7, 4, 0)] {
        let full = DeterminantSpace::new(n, na, nb);
        let refs = vec![
            (full.alpha_strings[0], full.beta_strings[0]),
            (
                *full.alpha_strings.last().unwrap(),
                *full.beta_strings.last().unwrap(),
            ),
        ];
        let generated = singles_doubles(n, na, nb, &refs, 100_000)
            .unwrap()
            .into_iter()
            .collect::<BTreeSet<_>>();
        let expected = full
            .alpha_strings
            .iter()
            .flat_map(|&a| full.beta_strings.iter().map(move |&b| (a, b)))
            .filter(|&(a, b)| {
                refs.iter()
                    .any(|&(ra, rb)| ((a.0 ^ ra.0).count_ones() + (b.0 ^ rb.0).count_ones()) <= 4)
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(generated, expected);
    }
}
#[test]
fn cas_references_respect_explicit_open_shell_partition_and_storage_limits() {
    let mut active = ActiveSpace::new(2, 1, 3);
    active.orbitals = Some(vec![4, 2, 0]);
    let space = active.resolve(7, 3, 2).unwrap();
    let refs = cas_references(&space, 7, 100).unwrap();
    assert_eq!(refs.len(), 9);
    for &(a, b) in &refs {
        assert_eq!(a.n_electrons(), 3);
        assert_eq!(b.n_electrons(), 2);
        for &i in &space.inactive {
            assert!(a.is_occupied(i) && b.is_occupied(i));
        }
        for &i in &space.external {
            assert!(!a.is_occupied(i) && !b.is_occupied(i));
        }
    }
    assert!(cas_references(&space, 7, 8).is_err());
    assert!(singles_doubles(7, 3, 2, &refs, 8).is_err());
    let one = [(SpinString(1), SpinString(0))];
    let complete = singles_doubles(64, 1, 0, &one, 64).unwrap();
    assert_eq!(complete.len(), 64);
    assert_eq!(complete.last().unwrap().0.0, 1u64 << 63);
}

#[test]
fn open_shell_spin_completion_matches_independent_closure_under_fci_spin_operator() {
    use hf_rs::fci::engine::FciHamiltonian;
    use hf_rs::fci::excitations::spin_complete;
    use hf_rs::{Matrix, mo_integrals::MoEriTensor};
    let n = 6;
    let np = n * (n + 1) / 2;
    let h = Matrix::zeros(n, n);
    let g = MoEriTensor {
        n_basis: n,
        n_pairs: np,
        data: vec![0.0; np * np],
    };
    let full = FciHamiltonian::new(&h, &g, 3, 2, 0.0).unwrap();
    let all = full
        .space
        .alpha_strings
        .iter()
        .flat_map(|&a| full.space.beta_strings.iter().map(move |&b| (a, b)))
        .collect::<Vec<_>>();
    let generated = singles_doubles(n, 3, 2, &[(SpinString(7), SpinString(3))], 100_000).unwrap();
    let completed = spin_complete(n, 3, 2, &generated, 100_000)
        .unwrap()
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut reached = generated.into_iter().collect::<BTreeSet<_>>();
    loop {
        let before = reached.len();
        let snapshot = reached.clone();
        for d in snapshot {
            let mut unit = vec![0.0; all.len()];
            unit[all.iter().position(|x| *x == d).unwrap()] = 1.0;
            for (i, value) in full.spin_squared_action(&unit).unwrap().iter().enumerate() {
                if value.abs() > 1e-12 {
                    reached.insert(all[i]);
                }
            }
        }
        if reached.len() == before {
            break;
        }
    }
    assert_eq!(completed, reached);
}
