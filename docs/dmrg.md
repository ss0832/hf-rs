# DMRG-CI and DMRGSCF

This page describes the typed Rust model and its recorded validation scope.
For the common Python/CLI adapter and analytic-property availability, see
[the current method catalog](methods.md).

The typed registry names are `DMRG` (`dmrgci`) and `DMRGSCF`. Both use a real
matrix product state with four states per spatial orbital: empty, alpha, beta
and doubly occupied. Bond sectors store `(Nalpha, Nbeta)`, equivalent to
`U(1) x U(1)` particle number and spin projection. The solver targets the lowest
state in this fixed sector. It does not project a specified total spin.

## Hamiltonian and optimization

The input spatial integrals use chemist's notation. The Hamiltonian is

\[
H=E_\mathrm{core}+\sum_{pq\sigma}h_{pq}a^\dagger_{p\sigma}a_{q\sigma}
+\tfrac12\sum_{pqrs\sigma\tau}(pq|rs)
a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}.
\]

Interleaved alpha/beta Jordan-Wigner operators supply all fermion signs.
Each operator string is a product of local signed partial permutations.
Identical strings are combined without an integral screening threshold.
Their direct sum is an exact, uncompressed MPO: its first tensor selects a
string and coefficient, its intermediate tensors are diagonal in the string
index, and its last tensor closes that index. This correctness backend does
not use optimized quantum chemistry complementary operators.

Directional two-site sweeps solve charge-allowed effective Hamiltonians with
matrix-free Davidson iterations. Each local wavefunction is decomposed by a
separate thin SVD in every bond sector. The largest Schmidt weights across
sectors are retained up to `M`; discarded weights are reported. A relative
`1e-8` degeneracy grouping makes the selection of equal Schmidt weights
deterministic, preferring spin balance closest to the target spin density.
This prevents eigensolver roundoff from selecting a different part of an
exactly degenerate spin multiplet.

Final one-site passes optimize the retained charge-sector dimensions without
further truncation. They are variational and reduce residual changes in the
returned MPS after two-site truncation. The reported energy is always the
normalized expectation of the **returned state**, also reproduced by
contraction of its 1- and 2-RDMs. It is not the minimum pre-truncation two-site
eigenvalue encountered during a sweep.

No determinant enumeration, full CI vector or full CI Hamiltonian is used,
including initialization, densities and orbital optimization. The direct-sum
MPO, MPS tensors, local solve storage and materialized active 2-RDM have
explicit 256 MiB allocation guards. These estimates are not a claim that total
process RSS has a universal 256 MiB limit. Unsupported sizes return errors.

## Densities and orbital optimization

Every spin-traced density element is contracted directly from the MPS:

\[
D_{pq}=\sum_\sigma\langle a^\dagger_{p\sigma}a_{q\sigma}\rangle,
\quad\Gamma_{pqrs}=\sum_{\sigma\tau}
\langle a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}\rangle.
\]

The active 2-RDM uses flattened `(p,q,r,s)` order. Inactive-core contraction,
explicit active-orbital selection, rectangular AO-to-MO coefficients and
open-shell occupations use the shared CAS infrastructure. DMRGSCF reuses its
generalized Fock matrix, diagonal orbital curvature, accumulated-rotation
DIIS and line search. Active-active rotations are included: they are
nonredundant for a finite-M MPS, even though they are redundant for full CI.

`DmrgDensityResult` retains active densities, the MPS and orbital convergence
diagnostics. Its standard output contains the full spatial 1-RDM in the
returned MO basis. The compatibility `ActiveSpaceDMRG` adapter uses the same
solver and additionally materializes a budget-checked full-space 2-RDM.
Its historical `three_rdm_norm` field denotes the norm of the active diagonal
`Gamma3[u,u,v,v,w,w]`, not the full 3-RDM norm.

## Options and derivatives

Along with the common active-space, convergence and orbital options:

| Option | Meaning | Default |
| --- | --- | --- |
| `bond_dimensions` | M for successive two-site directional passes; repeat the last entry | `[64,128,256]` |
| `sweeps` | Maximum two-site directional passes | `12` |
| `one_site_sweeps` | Maximum final fixed-bond directional passes; zero disables polishing | `12` |
| `truncation_tol` | Relative discarded Schmidt weight target, subject to M | `1e-10` |
| `noise` | Only zero is currently supported | `0` |

Only a single root is supported. Nonzero noise, state averaging and explicit
S2 projection return errors. The reference must be converged; density fitting
is not implemented for this adapter. Nonconverged tensor or orbital solves
propagate errors rather than reporting a converged output.

Nuclear derivatives are currently **numerical verification/debugging routes**
with independently reconverged energies. Complete finite-M tensor, orbital
and reference response is not certified as an analytic derivative. The
historical `compute_dmrg_analytical_gradient` function is a frozen-density
contraction and is not selected by the registry or the legacy gradient
dispatcher as a complete response implementation.

## References and validation

- U. Schollwoeck, *The density-matrix renormalization group in the age of matrix
  product states*, Ann. Phys. **326**, 96 (2011),
  [arXiv:1008.3477](https://arxiv.org/abs/1008.3477).
- The independent executable oracle uses the public
  [block2 driver API](https://block2.readthedocs.io/en/latest/api/pyblock2.html)
  in `SZ` mode, with identical M, two-site/one-site pass limits, no noise and
  no orbital reordering. No block2 implementation source is used.
- Numerical results, component counts and reproduction commands are recorded
  in [validation](validation.md#dmrg-mpsmpo-and-orbital-optimization).
