# Single-reference IM-SRG(2)

This page describes the typed Rust model and its recorded validation scope.
For the common Python/CLI adapter and analytic-property availability, see
[the current method catalog](methods.md).

`SRGODE` now evolves the complete normal-ordered scalar, one-body matrix and
antisymmetric two-body tensor. Its previous occupied/virtual doubles-only flow
is replaced by the NO2B commutator. The retained equations and decoupling
definition follow [Hergert et al., Eqs. 9.104–9.106 and Sec. 9.3.4](https://arxiv.org/html/1612.08315).
Induced normal-ordered three-body operators are discarded at each commutator;
their contractions into the retained lower ranks are included.

The off-diagonal Hamiltonian contains the reference-to-single and
reference-to-double couplings. Wegner uses `[H_d,H_od]_NO2B`. White divides
these couplings by the current diagonal Fock excitation gaps, using
Møller–Plesset denominators. A nonzero coupling with a gap below 1e-12 Eh
returns an error. This is the unregularized White generator.

The implementation uses full spin-orbital tensors with O(N^4) storage and
O(N^6) commutators. An estimate for sixteen tensor workspaces rejects allocations
above 256 MiB before the main tensors are created. This is an allocation estimate,
not a process-wide memory cap. The implementation is intended for small systems.

## API and integration

```rust
use hf_rs::post_hf::{PostHFMethod, mr_dsrg::{SRGGenerator, SRGODE}};

let result = SRGODE::new(&hf, SRGGenerator::Wegner, 0.5, 50)
    .with_ode_tolerance(1e-12)
    .compute()?;
```

The fourth argument sets fixed RK4 steps. With `with_ode_tolerance`, it instead
sets the initial step size, followed by step-doubling adaptation. The local
error estimate is `max_abs(fine - coarse)/15`, scaled against
`tolerance * (1 + max_abs(fine))`. Rejected steps do not advance the flow.
Nonfinite trial states cause a smaller step in adaptive mode; fixed-step
nonfinite results return an error. Invalid controls are rejected rather than
clipped. The accepted-step/attempt limits are 100,000.

`flow_param_s` is the requested finite flow time. It is dimensionless for White
and has inverse-energy-squared units for Wegner. A finite-flow result is not
a claim of converged ground-state decoupling. Inspect
`residual_off_diagonal_norm` and refine the integration separately from increasing
the flow time. `n_rk4_steps` records accepted macro steps; each accepted adaptive
macro step uses two half-size RK4 steps. `flowed.one` and `flowed.two` are the
flowing Hamiltonian coefficients, not reduced density matrices. `flowed.zero`
and the reported correlation energy are relative to the original HF energy.

The typed registry accepts `MethodId::SrGode` (`imsrg2`, `imsrg(2)`, `srgode`),
with `flow_parameter`, `ode_step`, `ode_tolerance` and `ode_generator` (`wegner`
or `white`). It reuses the context's MO-integral cache. The molecular adapter
currently supports a full-space RHF determinant. UHF, active/frozen spaces,
density fitting and reference relaxation are rejected explicitly. Python and
CLI select this adapter with `run("imsrg2", ...)` or `--method imsrg2`;
`ode_generator` explicitly selects Wegner or White. The individual legacy Python
SRG function was replaced by the common API.

## Independent checks

The independent Python oracle constructs complete creation/annihilation matrices
in a 64-dimensional Fock space. It forms matrix commutators, extracts bare
zero/one/two/**three**-body coefficients, and normal orders them with the reference
before discarding the remaining rank-three operator. The induced bare three-body
terms are nonzero. This avoids copying the production occupation-index formulas.

All 2,666 scalar/one/two-body components agree within 4.27e-14, including
Hermitian/Hermitian and anti-Hermitian/Hermitian commutators. Both six-orbital
flows compare every retained component with an independent SciPy DOP853 solve.
RK4 step halving shows the expected convergence, and adaptive results agree
within 2.27e-11. Separate molecular H2 checks use PySCF RHF integrals with the
same embedded STO-3G data and compare both legacy and typed APIs. A stiff
two-level model checks adaptive recovery against its exact eigenvalue.

See [`imsrg2_validation.json`](validation.md#external-validation-records) for commands,
tolerances and memory. These checks concern single-reference IMSRG(2).
They do not validate the legacy SR/MR driven-SRG approximations or MR cumulant
reconstruction. Relaxed nuclear derivatives and general MR reference states
require separate implementations.
