# Selected and projector CI

This page describes the typed Rust model and its recorded validation scope.
For the common Python/CLI adapter and analytic-property availability, see
[the current method catalog](methods.md).

ACI, ASCI and HCI share the excitation-generated, matrix-free determinant-space
Hamiltonian. They do not enumerate FCI to discover connected determinants. The
iterative solver checks the physical residual; selection or propagation reaching
its iteration limit returns an error. Explicit determinant and workspace limits
bound allocation.
`max_iter` controls outer selection/propagation; `ci_max_iter` controls each
inner variational Davidson solve independently (default 100).

- **ACI:** the importance is the lower eigenvalue of the reference/external 2×2
  Hamiltonian relative to the reference energy, including negative diagonal gaps.
  The discarded absolute energy-importance sum is at most `selection_threshold`
  (σ). Coarse graining discards a cumulative squared-coefficient weight at most
  `coarse_graining * selection_threshold` (γσ). The optional correction sums the
  discarded two-state importance values, as in Eq. 9; it is not an EN denominator
  with an arbitrary floor. [Schriber–Evangelista](https://arxiv.org/abs/1603.08063)
- **ASCI:** rank the core by coefficient magnitude. Rank searched noncore
  determinants by the perturbative amplitude from the core; previous noncore
  coefficients do not override this ranking. Keep `n_target_determinants` and
  optionally compute the final variational state's EN correction.
  [Tubman et al.](https://arxiv.org/abs/1603.02686)
- **HCI:** select when an individual `abs(H_ij * c_j)` exceeds
  `selection_threshold`. The PT2 search screens individual contributions using
  `pt2_threshold` **before** summing them and squaring the sum. Parents are not
  independently discarded by coefficient magnitude. A singular nonzero EN
  contribution returns an error. [Holmes et al.](https://arxiv.org/abs/1606.07453)

The typed context supports inactive/active/external partitions, frozen orbitals,
explicit active-orbital indices, open-shell populations, selected roots and shared
integrals. `target_s_squared` requests exact spin projection and completes each
selected spatial occupation pattern's spin couplings. Without this option SCI
reports the spin expectation of an unrestricted selected determinant space;
CASCI's default minimum-spin root ordering therefore need not match it. Multiroot
searches initialize from the connected SD space. ACI uses normalized state weights
for selection/coarse graining; ASCI and HCI use the maximum importance over roots.

The returned 1/2-RDMs describe the normalized **variational** CI state. An optional
energy correction does not turn these into a relaxed density of the corrected
energy. The typed result computes the active 2-RDM only when requested.

## PCI

The default wall-Chebyshev propagation uses the shifts in Eq. 25 and intermediate
normalization/path filtering in Eqs. 45–48 of
[Zhang–Evangelista](https://arxiv.org/abs/1606.08475). Each factor retains diagonal
contributions and screens each off-diagonal spawning amplitude. It neither
preserves a special HF anchor nor prunes the resulting coefficients by η. The
final variational estimator is the expectation of the propagated wavefunction;
there is no final model-space rediagonalization. The projective estimator uses
the determinant with the largest coefficient. A rigorous operator-norm spectral
bound avoids enumerating all determinants.
The lowest-orbital determinant supplies the trial wavefunction. As with any
projector calculation, it must overlap the intended ground state; a state in an
orthogonal spin/spatial symmetry sector is not selected by that trial.

Linear/power, pure Chebyshev, exponential Chebyshev (modified Bessel coefficients)
and Taylor choices are also available. Their recurrence extensions apply the
threshold to normalized intermediate vectors and restore their norms. The
historical Davidson/Lanczos choices are selected-space accelerators: spawning
selects a model and the unfiltered Hermitian model Hamiltonian supplies a
Davidson or fully reorthogonalized Lanczos/Ritz step. They are distinct from the
paper's wall propagation; a nonlinear path-filtered action is not used as a
Hermitian Lanczos operator. The optional EN correction is an additional estimator.

## Verification

`tests/reference/gen_selected_ci.py` and `gen_projector_ci.py` construct the full
small-model Hamiltonian with PySCF, then use independent dense NumPy selection or
propagation. JSON records include thresholds, energies, corrections and every
1/2-RDM component. Tests also cover tight-threshold FCI limits, spin-selected and
open-shell roots, active/frozen orbital mapping and failed convergence. These
energy/density checks do not establish the post-HF gradient implementation (P4).
