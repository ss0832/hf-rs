# Complex Hartree-Fock in a common magnetic gauge

This page describes the typed Rust model and its recorded validation scope.
For the common Python/CLI adapter and analytic-property availability, see
[the current method catalog](methods.md).

`run_complex_hf_with_options` solves complex restricted, unrestricted or
generalized HF. Orbitals, densities and Fock matrices retain both real and
imaginary parts. Convergence requires energy, the metric orbital residual and
density change. Failure to converge returns an error. Final occupied and
virtual orbitals are canonicalized separately without changing the determinant.

The ordinary Cartesian Gaussian basis uses the common vector potential
`A(r) = B x (r - O) / 2`. The spatial Hamiltonian contains `B.L/2` and
`|B x (r - O)|^2/8`, evaluated as Gaussian derivative and polynomial integrals.
`magnetic_field` is in atomic units. `spin_zeeman=true` additionally includes
`B.S` with g=2. Transverse spin coupling requires generalized HF. Restricted HF
is closed shell; unrestricted HF fixes the spin populations through
`multiplicity`. Generalized HF fixes total electron number and uses the initial
determinant to select a stationary branch; multiplicity does not constrain S_z.

```rust,ignore
use hf_rs::{ComplexHfOptions, ComplexHfReference,
    run_complex_hf_with_options, compute_complex_hf_gradient};

let options = ComplexHfOptions {
    reference: ComplexHfReference::Unrestricted,
    multiplicity: 2,
    magnetic_field: [0.03, 0.06, -0.08],
    energy_tol: 1e-14,
    residual_tol: 1e-12,
    density_tol: 1e-12,
    ..Default::default()
};
let result = run_complex_hf_with_options(&atoms, &basis, 1, &options)?;
let gradient = compute_complex_hf_gradient(&atoms, &basis, &result)?;
```

The default origin is the nuclear centroid and moves with the geometry. An
explicit `gauge_origin` remains fixed during differentiation. These are common
gauge calculations with ordinary Gaussian AOs, so finite-basis energies can
depend on that origin. London/GIAO phases are not implemented. For fixed-origin
calculations a translation sum at fixed origin need not vanish; transporting
both geometry and origin preserves the energy and gradient. Real scalar
local/nonlocal ECP matrices are retained in the specified gauge. Spin-orbit
ECP and relativistic magnetic Hamiltonians are unsupported.

The analytic first nuclear gradient contracts complex spin exchange, the real
charge density, the energy-weighted Pulay density, ECP and magnetic derivatives.
It includes origin response for the moving-centroid convention. It uses no
geometry differences. A canonically truncated overlap space returns an error
until projector response is available. Analytic complex-HF Hessians are not
implemented; the variant Hessian wrapper is explicitly numerical.

`spinor_density_real/imag`, `fock_real/imag` and `coefficients_real/imag` use
alpha-then-beta AO rows. Occupied coefficient columns precede virtual columns;
`orbital_energies` includes both spin sectors. The spatial `density_real/imag`
fields are spin traces. `current_magnitude` is an AO norm of their imaginary
part, rather than a physical current density. `spin_expectations` and
`s_squared` refer to the actual determinant.

The compatibility `run_complex_hf` selects closed-shell complex RHF and an
orbital field B_z. The legacy variant dispatcher distinguishes `crhf`, `cuhf`
and `cghf`, with its existing B_z=0.02 convention. The typed options above expose
the field, spin constraint, gauge, starting density and convergence controls.

`hf_variants::symmetry::classify_complex_hf_symmetry` classifies all eight
Fukutome determinant classes using metric Pauli density matrices. It reports
collinearity, real/imaginary spin Gram spectra, time-reversal invariance and
whether a global spin frame makes the determinant real. Both real and complex
HF results use this invariant classification. The historical
`classify_fukutome_symmetry(density, n)` remains a fixed-axis diagnostic.
The definitions follow Henderson, Jimenez-Hoyos and Scuseria,
[Magnetic Structure of Density Matrices](https://doi.org/10.1021/acs.jctc.7b01016).

Independent PySCF comparisons check every magnetic integral, spinor density and
Fock entry, arbitrary complex spin-coherence contractions, spin expectation S2,
all nuclear coordinates, explicit origin transport and eight-class covariance.
See [validation](validation.md).

```sh
cargo test --offline --test complex_hf_test --test ghf_reference_test
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_complex_hf.py
OPENBLAS_NUM_THREADS=4 python tests/reference/gen_hf_symmetry.py
```
