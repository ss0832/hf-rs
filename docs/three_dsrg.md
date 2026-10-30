# THREE-DSRG-MRPT2

This page describes the typed Rust model and its recorded validation scope.
For the common Python/CLI adapter and analytic-property availability, see
[the current method catalog](methods.md).

`MethodId::ThreeDsrgMrPt2` evaluates Fdiag MRPT2 with exact active CI cumulants,
using three-index integrals throughout the correlation and reference-relaxation
calculation. The legacy `ThreeDSRGMRPT2` adapter uses the same backend.

The theory is described by Li and Evangelista,
[JCTC 11, 2097 (2015)](https://doi.org/10.1021/acs.jctc.5b00134), and the
integral-factorized formulation by Hannon, Li and Evangelista,
[JCP 144, 204111 (2016)](https://doi.org/10.1063/1.4951684).

## Inputs and outputs

```rust,ignore
use hf_rs::{
    options::{ActiveSpace, PostHFOptions, ReferenceRelaxation},
    registry::{MethodId, create_solver},
};

let mut options = PostHFOptions::default();
options.active = Some(ActiveSpace::new(1, 1, 2));
options.cas.optimize_orbitals = false;
options.dsrg.flow_parameter = 0.5;
options.dsrg.cholesky_tolerance = 1e-12;
options.dsrg.reference_relaxation = ReferenceRelaxation::Once;
let output = create_solver(MethodId::ThreeDsrgMrPt2, &context, &options)?.compute()?;
```

The default integral route is pivoted AO Cholesky, with a residual Coulomb
diagonal threshold `cholesky_tolerance` (default `1e-10`). `density_fitting=true`
instead requires an explicit `AuxBasis` in the context and uses its Gaussian
Coulomb metric with the shared absolute `1e-10` metric cutoff. The auxiliary
basis is never inferred from an orbital-basis label.

One spin-balanced singlet, rectangular real MOs, arbitrary active-orbital
selection, and none/once/twice/iterated reference relaxation are supported.
Purely active excitation amplitudes are absent. All eight core/active/external
double-excitation sectors and the semi-internal singles remain. Zero gaps use
the smooth limit of `(1-exp(-s*gap^2))/gap`; no physical denominator clipping
or empirical smoothing is applied.

CASCI is solved with the supplied factorized integrals. With Cholesky and
`optimize_orbitals=true`, conventional CASSCF supplies initial orbitals, followed
by factorized CASCI and MRPT2. `E_conventional_CASSCF_orbitals` records that
initial orbital optimization energy; `E_CASCI` records the actual factorized
reference. The conventional orbital-optimization phase still uses full MO
integrals. DF orbital optimization, frozen orbitals, multiple states, alternate
zeroth-order partitions, approximate cumulants, NIVO and sequential
transformations are explicitly unsupported on this path.

Standard `e_corr` is relative to SCF. Components include `E_CASCI`,
`E_unrelaxed`, `E_PT2`, `E_reference_source`, `E_reference_relaxation`, CI
convergence diagnostics, `factor_rank`, `factorized_sectors` and
`peak_sector_terms`. The Cholesky route also reports `cholesky_max_residual`.
The legacy result retains its CAS-relative `correlation_energy` field and
reports `cholesky_rank`. Neither route exposes a reference density as a
correlated or response density.

## Memory and derivatives

The correlation backend does not reconstruct fitted four-index ERIs, allocate
full-space rank-four amplitudes, or allocate full-space rank-six densities.
It stores three-index factors, one excitation sector, and active-space CI/GNO
operators. Active-space rank-three tensors remain and can limit the active
space. Workspace estimates and incremental Cholesky allocation guards reject
infeasible requests under the 256 MiB working budget. The borrowed HF AO ERIs,
the caller's caches and external processes are outside this estimate.

`typed_mr::numerical_gradient` provides an explicitly numerical validation/debug
route. For DF, use `numerical_gradient_with_aux` with the explicit auxiliary
basis; its centers and metric move with the atoms. All electronic steps are
reconverged at each point. Analytic post-HF response and Hessians remain deferred
work. Existing analytic HF/ECP nuclear derivatives through fourth order are
unchanged. See [validation](validation.md).

```sh
cargo test --offline --test three_index_test
cargo test --offline --test typed_mr_test three_index
cargo test --offline --test typed_mr_test factorized_cholesky
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_three_df.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_three_df.py --gradients
```
