# Scalar-relativistic Hartree-Fock (spin-free X2C and DKH2)

This page describes the typed Rust model and its recorded validation scope.
For the common Python/CLI adapter and analytic-property availability, see
[the current method catalog](methods.md).

`run_relativistic_hf_with_options` solves closed-shell RHF with a one-electron
scalar-relativistic core Hamiltonian built in the supplied contracted AO space.
Two-electron interactions are untransformed (no picture-change correction) and
spin-orbit coupling is not included. Point nuclei are used; atoms carrying an
effective core potential are rejected, because ECPs already contain relativity.

| `RelativisticModel` | Hamiltonian | Analytic gradient |
| --- | --- | --- |
| `X2C` | spin-free exact two-component (one-step, SFX2C-1e) | yes |
| `DKH2` | spin-free second-order Douglas-Kroll-Hess (matrix form) | yes |
| `ZORA` | not implemented (requires a numerical integration grid); returns an error | - |

Both Hamiltonians use the AO matrices `S`, `T`, `V` and `W = <p chi|V|p chi>`.

## DKH2 construction

In the orthonormal eigenbasis of the kinetic matrix (`p_i^2 = 2 t_i`) the
free-particle Foldy-Wouthuysen transformation is diagonal with
`E_i = c sqrt(p_i^2 + c^2)`, `A_i = sqrt((E_i + c^2)/(2 E_i))` and
`K_i = c/(E_i + c^2)`. With `R = K p` the first-order even and odd operators are

- `E1 = A (V + K W K) A`,
- `O1 = A (K W p^-1 - V K p) A`,

`W1` solves the Sylvester equation `E W1 + W1 E = O1`, and the second-order term
is `E2 = (W1 O1^T + O1 W1^T)/2 = [W1, O1]/2` (B. A. Hess, Phys. Rev. A 33, 3742
(1986); G. Jansen and B. A. Hess, Phys. Rev. A 39, 6016 (1989)). The core
Hamiltonian is `h = (E - c^2) + E1 + E2`, where `E - c^2 = p^2 c^2/(E + c^2)` is
evaluated without cancellation. All factors are matrix functions of the kinetic
matrix, so the analytic nuclear derivative uses exact divided differences
(Daleckii-Krein) and a second Sylvester solve; coincident kinetic eigenvalues
need no gap denominators. Overlap-rank reduction is supported for energies;
rank-reduced gradients are rejected until projector response is implemented.

```rust,ignore
use hf_rs::{RelativisticHfOptions, RelativisticModel,
    run_relativistic_hf_with_options, compute_relativistic_hf_gradient};

let options = RelativisticHfOptions { speed_of_light: 137.035999084, ..Default::default() };
let result = run_relativistic_hf_with_options(&atoms, &basis, 0, RelativisticModel::DKH2, &options)?;
let gradient = compute_relativistic_hf_gradient(&atoms, &basis, &result)?;
```

`speed_of_light` defaults to the CODATA 2018 value 137.035999084 a.u.

## Validation

`tests/dkh2_test.rs` and `tests/x2c_core_test.rs` check:

- every AO DKH2 matrix entry (500) and directional response entry (1,500)
  against an independent element-wise NumPy construction from PySCF integrals
  (`tests/reference/gen_dkh2.py`): maximum errors 8.2e-13 and 5.9e-11;
- exact limits: constant potential (`h = h_free + v0 S`, also equal to X2C),
  `c -> infinity` (`h -> T + V`), the `c^-4` scaling of the DKH2-X2C
  difference, and continuity of the response at degenerate kinetic eigenvalues;
- molecular energies against PySCF/NumPy (1.7e-12 Eh) and against the external
  Psi4 1.11 program for Ne, Ar, Kr, water and HCl (`gen_dkh2_psi4.py`):
  DKH2 within 6.5e-12 Eh and X2C within 7.2e-11 Eh. Black-box runs show that
  Psi4 uses c = 137.0359895 for DKH and the CODATA 2014 value 137.035999139
  for X2C; the comparison passes those values explicitly;
- all 21 analytic DKH2 nuclear-gradient components against five-point energy
  differences of the independent oracle (maximum 2.6e-10 Eh/bohr) and
  translational invariance (6.6e-12).
