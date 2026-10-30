# Exact spin-orbital density cumulants

This page describes the typed Rust model and its recorded validation scope.
For the common Python/CLI adapter and analytic-property availability, see
[the current method catalog](methods.md).

`post_hf::mr_dsrg::density_cumulants` computes exact one-, two- and three-body
density information from CI coefficients. It accepts a pure state or a weighted
ensemble. Members share the spatial orbitals and total electron count, while
their alpha/beta populations may differ. This permits explicit ensembles over
spin projections as well as ordinary state averages.

The normal-ordering conventions follow
[Kutzelnigg and Mukherjee, J. Chem. Phys. **107**, 432 (1997)](https://doi.org/10.1063/1.474405)
and the ensemble definitions in
[Li and Evangelista, J. Chem. Phys. **155**, 114111 (2021), Sec. II.1](https://arxiv.org/html/2106.07097).
RDMs are averaged first; nonlinear cumulants are computed from those averaged
RDMs. Averaging the pure-state cumulants would define a different object.

## Conventions and algebra

Spin-orbital indices are interleaved: `2*p` is alpha and `2*p+1` is beta.
For strictly increasing creation and annihilation tuples `I` and `J`,

```text
Gamma_k[I,J] = <a^+_i1 ... a^+_ik a_jk ... a_j1>.
lambda2[ab,ij] = Gamma2[ab,ij] - gamma[a,i]*gamma[b,j]
                                    + gamma[a,j]*gamma[b,i]
lambda3[abc,ijk] = Gamma3[abc,ijk] - det(gamma[abc,ijk])
                  - sum(r,s=0..2) (-1)^(r+s) gamma[a_r,i_s]
                      * lambda2[abc without a_r, ijk without i_s].
```

Every term uses the same spin-orbital ordering. There are nine signed
one-body/two-body products in the last line. No spin averaging is inferred
from a spin-summed density. In particular, the exact three-body cumulant
requires the true three-RDM.

The production RDM algorithm removes each occupied tuple from each determinant,
including its fermion sign. It groups amplitudes by the residual determinant
and accumulates their outer products. It does not construct creation matrices
or a determinant-pair Hamiltonian. CI vectors are normalized with a stable
Euclidean norm; rescaling by factors as small as 1e-200 or as large as 1e200
does not change the densities. Ensemble weights are normalized independently.

## Storage and API

`AntisymmetricDensity` stores only `binomial(N_spin,k)^2` coefficients.
`element` restores both permutation signs, returns zero for repeated indices,
and rejects wrong-rank or out-of-range indices. `full_norm` includes the
permutation multiplicity of the full tensor. Third-order storage is at most
`N_spin^6/36` entries, rather than a full sixth-rank spin tensor.

The raw packed three-RDM is converted to `lambda3` in place. `gamma3` reconstructs
an arbitrary exact three-RDM element when requested, so both large arrays are
never retained together. A caller-supplied memory budget checks packed tensors
and a conservative sparse-removal workspace estimate before allocation.
`DEFAULT_CUMULANT_MEMORY_BUDGET` is 256 MiB. This estimates new working storage;
it excludes the borrowed CI coefficients and is not a process RSS cap.

Use `exact_density_cumulants(&members, memory_budget)` with explicit
`CiEnsembleMember` values. Coefficients follow `space.linear_to_pair`: alpha
strings are the outer index and beta strings the inner index, with the usual
alpha-before-beta creation convention. The saved CASCI/CASSCF result also exposes:

```rust,ignore
let averaged = cas.exact_active_cumulants(256 * 1024 * 1024)?;
let selected = cas.exact_root_cumulants(root, 256 * 1024 * 1024)?;
```

The averaged call uses the weights saved in `cas.output.states`; the root call
selects one CI root regardless of those weights. These are **active-space**
spin-orbital densities in the returned CAS orbital order. The CAS adapter checks
CI/string dimensions and includes its newly generated spin strings in the budget.

The compatibility `ThreePDC::MK` path in `cumulants.rs` remains a reconstruction
from lower RDMs and is documented accordingly. It is not the exact API above.
This standalone reconstruction helper is distinct from the verified MR compatibility
constructors: their historical `ThreePDC::MK` selector now uses exact CI cumulants,
as described in [MR driven equations](mr_driven.md#compatibility-constructors).
The exact GNO commutator and typed MR driven solvers use this density
backend; see [MR driven equations](mr_driven.md) for their separate validation
and supported interfaces. Density checks alone do not certify MR energies or derivatives.

## Independent verification

The Python oracle explicitly builds full Fock-space creation/annihilation
matrices. Its cumulants are coefficients of the paired exterior-algebra
logarithm `log(1+Gamma1+Gamma2+Gamma3)`, evaluated through degree three. It does
not evaluate the production disconnected-density expressions.

Nine ensembles cover correlated states with two, three and four electrons,
an ordinary Slater determinant, an orbitally rotated Slater determinant,
coherent superpositions and mixtures, and a spin-projection average. A pair
of three-electron coherent states has exactly the same one- and two-RDMs but
opposite three-body coherence, proving that the three-body information cannot
be reconstructed from lower densities.

The native suite checks **all 4,792,896 spin-orbital components**, including
every permutation and Pauli zero, of `gamma1`, `Gamma2`, `lambda2`, `Gamma3`
and `lambda3`. It also checks every spatial spin trace against the independent
FCI RDM algorithm, all one- and two-particle contractions, Slater limits,
ensemble order, coefficient-scale invariance, invalid inputs and budget errors.
The CAS integration test compares all active one-/two-RDM spin traces for two
LiH roots and their weighted state average.

Reproduce in an activated scientific Python environment:

```sh
OPENBLAS_NUM_THREADS=1 \
  "${VIRTUAL_ENV}/bin/python" tests/reference/gen_density_cumulants.py
PYO3_PYTHON="${VIRTUAL_ENV}/bin/python" \
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/native}" \
CARGO_BUILD_JOBS=1 CARGO_PROFILE_TEST_OPT_LEVEL=3 CARGO_PROFILE_TEST_DEBUG=0 \
RAYON_NUM_THREADS=4 cargo test --offline --test density_cumulants_test -- --nocapture
```

At the density-cumulant checkpoint, the combined regression passed 72 tests and
its Python test/reference sources passed Ruff. Those counts are historical.
The five small-model tests peaked at 12,820 KiB native RSS, compiler excluded;
that measurement does not bound the current complete suite. See
[the validation record (`density_cumulants_validation.json`)](validation.md#external-validation-records).
