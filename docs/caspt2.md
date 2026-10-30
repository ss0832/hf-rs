# Internally contracted CASPT2

This page describes the typed Rust model and its recorded validation scope.
For the common Python/CLI adapter and analytic-property availability, see
[the current method catalog](methods.md).

`post_hf::caspt2_typed::solve_caspt2` and the `MethodId::Caspt2` registry entry
use the shared CASCI/CASSCF reference solver and a spin-free, internally
contracted first-order space. The legacy `CASPT2::from_hf` and
`CASPT2::new(hf, casscf, options)` interfaces use the same equations.

## Equations and excitation classes

With inactive indices i,j, active indices t,u,v and external indices a,b,
E_pq is the spin-summed one-electron excitation operator. The classes are:

| Class | Operators acting on the CAS reference |
| --- | --- |
| A | E_ti E_uv |
| B | E_ti E_uj |
| C | E_at E_uv |
| D | E_ai E_tu and E_ti E_au |
| E | E_ti E_aj |
| F | E_at E_bu |
| G | E_ai E_bt |
| H | E_ai E_bj |

Excitation-generated sparse determinant vectors retain the exact active CI
coefficients. The active CAS alone is enumerated; a full-orbital FCI space
or Hamiltonian is never constructed. Projected generalized Fock actions
include the exact three-particle density and Fock-contracted four-particle
density contributions without allocating the eight-index four-particle RDM.

The symmetric spin-averaged Fock matrix is
`f_pq = h_pq + sum_rs D_rs [(pq|rs) - (pr|qs)/2]`.
The zeroth-order reference energy is `E0 = sum_pq f_pq D_pq`; it is distinct
from the CAS electronic energy. Each class metric S is diagonalized and
vectors with eigenvalue at most `caspt2_metric_tol * max(eigenvalue)` are
removed. Canonical metric whitening and intraclass Fock diagonalization
precede the coupled solution. Off-diagonal couplings between classes are
retained. PCG uses Jacobi preconditioning, with a normal-equation fallback
on nonpositive curvature, and verifies the original equation residual.

## Shifts and continuity

IPEA defaults to 0.25 Eh in the typed adapter and Rust compatibility constructor. The low-level
`IcCaspt2Options` default is unshifted. In pseudocanonical orbital blocks,
the diagonal primitive IC correction is
`Delta_F_JJ = IPEA * S_JJ * sum_(p,q in J)(2 + D_pp - D_qq)/2`.
It is formed before canonical metric whitening, using unnormalized
spin-free excitation products. IPEA is basis dependent: primitive rescaling,
spin coupling, linear-dependence removal and orbital rotations can change
its value. Use explicit `CORT` for the OpenMolcas comparisons below.
This specifies the implemented diagonal-IC convention; exact agreement
between different IPEA conventions is not promised. In BeH2 the measured
IPEA difference is 3.46e-7 Eh, within the declared 1e-6 tolerance.

The real shift solves `(A + delta I)t = -v`. The diagonal
Forsberg-Malmqvist imaginary shift solves
`(A + sigma^2 / diag(A))t = -v` in the intraclass eigenbasis.
A symmetric scaling by `sqrt(abs(diag(A)))` avoids division by a small
number, including the exact zero-denominator limit. No physical denominator
is clipped or discarded. Real and imaginary shifts are mutually exclusive.
The reported correction is the shift-corrected Hylleraas value
`2 v.t + t.A.t`; the regularizing shift is excluded from this energy
operator, while IPEA remains part of A. The uncorrected `v.t` is separate.
A real shift can still move an intruder singularity; convergence failure
is reported instead of silently removing a contribution.

## API, limits and verification

Use `PostHFOptions.active`, `cas.optimize_orbitals`, `frozen_core`,
`frozen_virtual`, `cas.ipea_shift`, `cas.real_shift`, `cas.imaginary_shift`,
`cas.caspt2_metric_tol`, `residual_tol` and `max_iter`. The metric default is
1e-10. Freezing leaves the reference density in the Fock matrix but removes
its corresponding excitation indices. Empty CAS with zero IPEA reduces to
MP2, including frozen-core/virtual limits; a complete CAS has zero correction.

These dotted names describe fields of the Rust options struct. In Python/CLI
dictionaries, use the flat registry keys `ipea_shift`, `real_shift`,
`imaginary_shift`, `caspt2_metric_tol` and `optimize_orbitals`, together with
`active_n_alpha`, `active_n_beta`, `active_n_orb` and optional `active_orbitals`.

Output includes the actual CAS energy, eight class contributions, PT2
correction, uncorrected correction, reference weight, rank and residual.
`e_corr` always means total minus SCF energy. A CAS reference density is not
reported as a relaxed CASPT2 density. The legacy partition uses H for
`e_external`, A-G for `e_semi_internal`, and zero for `e_singles`, since
singles have no independent invariant partition in this contracted space.

This is a bounded correctness backend: dense contracted metric/Fock work is
retained, with a conservative 256 MiB working-storage estimate checked before
large intermediate allocation and at most one million CAS determinants.
This estimate excludes input integrals, the CAS reference solver and process
runtime. It is not a large-system scaling or total-RSS guarantee.
Single-state references and conventional integrals are supported; multistate
CASPT2 and density fitting are rejected explicitly. Rectangular MO matrices
are accepted by the shared reference machinery.

`numerical_gradient` and the legacy gradient dispatcher reconverge the full
SCF/CAS/PT2 energy at every point. They are numerical verification APIs.
Analytic CASPT2 response and Hessians remain the next-version P4 work.
Existing analytic HF/ECP derivatives through fourth order are unaffected.

Run in native Linux/WSL:

```sh
cargo test --offline --test caspt2_engine_test --test caspt2_molecular_test
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_caspt2.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_caspt2_molecular.py
```

The tests check all eight classes, exact metrics, 4,374 first-order coefficients,
24 model/shift combinations, 18 OpenMolcas energies, MP2/full-CAS limits,
zero-denominator continuity, allocation guards and all 48 nuclear components
of CASCI/CASSCF and nonlocal-ECP cases. OpenMolcas 25.02-2 inputs, printed
energies and output hashes are in `tests/data/caspt2_openmolcas_reference.json`.

## References

- Andersson, Malmqvist and Roos, J. Chem. Phys. 96, 1218 (1992),
  [DOI: 10.1063/1.462209](https://doi.org/10.1063/1.462209).
- Wouters, Van Speybroeck and Van Neck, J. Chem. Phys. 145, 054120 (2016),
  [public manuscript, especially Eq. 48](https://arxiv.org/abs/1605.05526).
- Park, Al-Saadon, Strand and Shiozaki, J. Chem. Theory Comput. 15, 4088–4098 (2019),
  [imaginary-shift equations and Hylleraas energy](https://arxiv.org/abs/1904.06718).
- [OpenMolcas CASPT2 manual](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/caspt2.html)
  for executable oracle settings, CORT and convergence controls.
