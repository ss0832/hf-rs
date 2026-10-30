# Multistate DSRG-PT2 energies

This page describes the typed Rust model and its recorded validation scope.
For the common Python/CLI adapter and analytic-property availability, see
[the current method catalog](methods.md).

The registry and legacy `MultiStateDSRG` share an effective-Hamiltonian backend
with exact ensemble CI cumulants, singlet spin targeting and factorized
integrals. The GNO transformed Hamiltonian is truncated to rank two. The H3
variant is not implemented.

| Registry name | Rust `MethodId` | Final diagonalization |
| --- | --- | --- |
| `SA-DSRG-PT2` | `SaDsrgPt2` | Full active CI space |
| `SA-DSRG-PT2c` | `SaDsrgPt2C` | Selected reference states |
| `MS-DSRG-PT2` | `MsDsrgPt2` | Selected reference states |
| `XMS-DSRG-PT2` | `XmsDsrgPt2` | Ensemble-Fock-rotated reference states |
| `DWMS-DSRG-PT2` | `DwmsDsrgPt2` | Selected reference states |

SA uses one ensemble transformation. MS combines state-specific transformed
Hamiltonians symmetrically. XMS first rotates the model states with the
state-averaged Fock matrix. DWMS builds a separate ensemble for each state,
with Gaussian weights based on reference energy separations. Its parameter
`dynamic_weight_parameter` is nonnegative and defaults to `50.0` in inverse
squared Hartree. At zero it gives contracted SA; at large values it approaches
MS for separated states. Exactly degenerate states retain equal weights.

The definitions follow Li and Evangelista,
[JCP 148, 124106 (2018)](https://doi.org/10.1063/1.5019793), and Li, Lindh and
Evangelista, [JCP 150, 144107 (2019)](https://doi.org/10.1063/1.5088120),
especially its rank-two effective Hamiltonian and Gaussian weighting.

## Usage

```rust,ignore
use hf_rs::{
    options::{ActiveSpace, PostHFOptions},
    registry::{MethodId, create_solver},
};

let mut options = PostHFOptions::default();
options.active = Some(ActiveSpace::new(1, 1, 2));
options.n_roots = 2;
options.root = 1;
options.cas.optimize_orbitals = false;
options.dsrg.flow_parameter = 0.5;
options.dsrg.cholesky_tolerance = 1e-12;
let output = create_solver(MethodId::SaDsrgPt2, &context, &options)?.compute()?;
```

`n_roots` selects the singlet reference manifold. `root` selects the returned
correlated energy and does not change the reference ensemble. Default reference
weights are equal, including optional SA-CASSCF starting-orbital optimization.
Explicit nonnegative `state_weights` are normalized for that optimization and
the SA transformation. XMS requires equal weights. MS and DWMS choose their
correlation-source weights according to their definitions.

The default route uses pivoted Cholesky. `density_fitting=true` requires an
explicit auxiliary basis and `optimize_orbitals=false`. It uses the common
Coulomb-metric DF factors. Fixed real rectangular MOs, arbitrary active-orbital
selection, local/nonlocal ECPs and multiple singlet roots are supported.
Conventional SA-CASSCF may supply starting orbitals on the Cholesky route;
its full-integral phase is released before factorized CASCI/DSRG. An empty CAS
is supported for one state and recovers the single-reference energy.

Frozen orbitals, spin-unbalanced or nonsinglet references, DF orbital
optimization, iterative state-reference updates and higher-body effective
Hamiltonians are explicitly unsupported. Workspace checks precede active/sector
allocation; only active determinant strings are generated. Production never
constructs a full Fock-space matrix or full-space rank-six density tensor.

`PostHFOutput.states` contains all computed energies. `e_total` and `e_corr`
refer to `root`, with correlation relative to SCF. `E_CASCI` is that root's
factorized reference energy. Other components identify the factor rank,
source count, model/active-CI dimensions and contracted/full-CAS selection.
`E_conventional_CASSCF_orbitals` records an optional initial orbital
optimization separately. A correlated or response density is not returned.

The legacy result exports final `ci_coefficients` in `mo_coefficients`,
`model_rotation`, and `ensemble_weights`. Its `effective_hamiltonian` is the
projection onto the selected model states. For full-CAS SA, the final energies
come from the full active operator and need not equal the eigenvalues of that
smaller diagnostic matrix. `StateAverageContracted` selects the restricted
model-space variant explicitly. Legacy none/once relaxation selectors both
select the method's one final diagonalization; twice/iterated selectors fail.

## Derivatives and validation

`typed_mr::numerical_gradient` and `numerical_gradient_with_aux` are numerical
validation/debug routes. The latter rebuilds a co-moving auxiliary basis and
metric. Every displaced point reconverges SCF, CAS and all source transformations.
Analytic post-HF response and Hessians remain next-version work. Existing
analytic HF/ECP derivatives through fourth order are unchanged.

Independent NumPy full-Fock matrices verify every source-operator coefficient,
model-matrix element, energy and CI coordinate. Tests also cover state rotations,
DW limits, exact degeneracy, zero flow, the single-reference/full-CAS limits,
selected roots and all nuclear components for both states, including nonlocal
ECP and DF. See [validation](validation.md).

```sh
cargo test --offline --test multistate_test
cargo test --offline --test typed_mr_test multistate
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_multistate.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_multistate.py --gradients
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_multistate.py --verify-fits
```
