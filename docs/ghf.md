# Real noncollinear generalized Hartree-Fock

This page describes the typed Rust model and its recorded validation scope.
For the common Python/CLI adapter and analytic-property availability, see
[the current method catalog](methods.md).

`run_ghf_with_options` solves a real two-component spinor determinant with
alpha-then-beta AO ordering. All four spin density blocks are retained.
The Hamiltonian is nonrelativistic with optional scalar local/nonlocal ECPs.
The number of electrons is fixed; spin polarization is free to change.
A stationary solution is not guaranteed to be the global HF minimum.

```rust
let options = hf_rs::GhfOptions {
    energy_tol: 1e-12,
    residual_tol: 1e-10,
    density_tol: 1e-10,
    ..Default::default()
};
let result = hf_rs::run_ghf_with_options(&atoms, &basis, charge, 0.0, &options)?;
let gradient = hf_rs::compute_ghf_gradient(&atoms, &basis, &result)?;
```

## Equations and convergence

For real spinor density P, the Fock blocks are
`F_st = delta_st (h + J[P_aa + P_bb]) - K[P_st]`.
Exchange is contracted for nonsymmetric off-diagonal spin blocks. Contractions
have no density-element cutoff. Integral storage is borrowed within the solve;
individual spin-density blocks and a full MO ERI tensor are not copied out.

The canonical overlap orthogonalizer retains eigenvectors with eigenvalue
at least `overlap_cutoff` (default 1e-10). Spinor coefficients are rectangular
when overlap modes are removed. Electron capacity is checked after removal.
The initial occupied projector and all subsequent densities satisfy the
AO metric, including `Tr(P S2) = N` and `P S2 P = P`.

Core spinors are interleaved deterministically. A nonzero tilt uses a complete
orthogonal orbital rotation. If it converges to a closed-shell stationary
point, the lowest real spin-breaking curvature is checked using AO exchange
actions. A negative curvature below -1e-8 supplies opposite-spin orbital
rotations. Eight initial fixed-point steps let that direction grow before
DIIS resumes. The requested iteration limit covers both stages. An explicit
`initial_density` overrides generated guesses and must be symmetric, finite,
metric-idempotent and contained in the retained AO space.

Convergence requires energy change, density change and the commutator
`X2^T (F P S2 - S2 P F) X2`. The residual Frobenius norm and density RMS
are returned. DIIS operates in the retained orthonormal space. Nonconvergence
is an error. The final occupied and virtual orbital blocks are canonicalized
with the physical Fock matrix, preserving the returned density and energy.
The compatibility `run_ghf` signature remains available; it uses the supplied
energy tolerance and caps residual/density tolerances at 1e-8.

Spin moments use the AO overlap. For the metric-idempotent real determinant,
`<S2> = |<S>|^2 + N - Tr[(P_total S)^2]/2`.
Signed singlet roundoff is retained. `ghf_spin_expectations` exposes the same
formula. The legacy `classify_fukutome_symmetry` is a fixed-spin-axis real
representation diagnostic with TICS/ASDW/TSDW labels. The separate
`hf_variants::symmetry::classify_complex_hf_symmetry` implementation provides
the complete invariant eight-class analysis for real and complex determinants;
see [complex HF](complex_hf.md).

## Analytic nuclear gradient

`compute_ghf_gradient` evaluates the stationary energy derivative without
geometry displacements or orbital finite differences. It uses the spin trace
of the density, energy-weighted density `W2 = P F P`, the shared Coulomb/Pulay
and nuclear/ECP derivatives, and the exchange contraction from all spin blocks.
One shared screened derivative pass contracts differentiated Gaussian basis
functions with the full spin-block density. There is no fourth-rank density
allocation or second unscreened exchange-derivative pass.

The analytic GHF branch is connected to the HF-variant gradient dispatcher.
Gradients for rank-truncated overlap spaces are rejected because projector
response is not implemented. `compute_hf_variant_hessian` remains a numerical
gradient-difference routine; it is not analytic second-order response.
Existing analytic RHF/local+nonlocal ECP derivatives through fourth order
are unchanged.

The default 256 MiB `max_work_bytes` limit checks a conservative packed-ERI
and matrix estimate before integral allocation, and separately bounds the
optional stability matrix. It is a working-storage estimate, not total process
RSS. Large-system performance is still P5 work.

## Independent verification

PySCF provides independent Fock, Slater spin and energy evaluations. Its
stationarity is refined with a SciPy occupied/virtual residual solve when
spin-rotation zero modes prevent normal SCF stopping. Density comparisons
align the physically equivalent global spin frame. The split-SP 6-31G
matrices are explicitly permuted from PySCF angular-momentum order to the
repository G94 shell order. The ethylene reference uses the independently
converged broken-symmetry Ms=0 UHF branch, which is a scalar GHF stationary
state under global spin rotation. Energies, Fock contractions
and nuclear derivatives are compared directly. Arbitrary spin-coherent
projectors test every Fock entry and rotational covariance.

The tests cover an H2+ doublet, closed-shell LiH, frustrated noncollinear H3,
nonlocal-ECP HeH+, and the spin instability of twisted ethylene/6-31G.
All nuclear coordinates are compared with five-point h=1e-3 bohr energies.
Run with native Linux/WSL:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_ghf.py
cargo test --offline --test ghf_reference_test
cargo test --offline --test hf_variants_suite_test test_ghf
```

The [PySCF SCF API](https://pyscf.org/pyscf_api_docs/pyscf.scf.html)
documents the GHF `spin_square`, density, Fock and energy oracle calls.

Five energies agree within 3.27e-13 Eh. All 2,936 Fock/density entries are
checked, with maximum Fock error 9.71e-14 and spin-frame density error 3.33e-9.
All 45 analytic nuclear components agree with independent energy stencils
within 6.97e-10 Eh/bohr; translation sums are below 8.63e-14. The four-test
native process peaks at 17,016 KiB RSS with RAYON_NUM_THREADS=4 and one test
thread. This excludes compilation and Python and is a small-model measurement.
