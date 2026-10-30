# Local correlation implementation and validation

This page describes the typed Rust model and its recorded validation scope.
For the common Python/CLI adapter and analytic-property availability, see
[the current method catalog](methods.md).

The typed registry connects `DLPNO-MP2`, `DLPNO-CCSD`,
`DLPNO-CCSD(T0)` and `DLPNO-CCSD(T1)` to the checked local engines. All require a converged RHF reference and an explicit auxiliary basis.
The `DLPNOMP2`, `DLPNOMCCSD` and `DLPNOMCCSDT` compatibility classes delegate
to the checked engines. Use `with_context` or `with_system` with an explicit
auxiliary basis. The last compatibility name selects iterative (T1). The
common Python/CLI API requires `aux_basis`; it does not infer a fitting basis
from the orbital basis. For example, pass `options={"aux_basis": "def2-svp-ri"}`
to `run`. The removed empirical taper and neglected-pair options return errors.
Transported constant-dimensional spaces are available for MP2, CCSD and both
triples variants, as described below.

Boys and exponent-2 Mulliken Pipek–Mezey localization use exact real Jacobi pair
maximization, check every orbital-pair gradient and reject unconverged runs.
The pair gain check also escapes a zero-gradient pair minimum. PySCF's canonical
starting orbitals led to inferior stationary points for the validation systems;
independent multistart SciPy optimization and gradient refinement establish the
reference objective, with both results recorded in the JSON provenance.

Projected AOs are orthonormalized as `Q U lambda^(-1/2)`. The molecular engine
uses `Q = C_virtual C_virtual^T S`, retaining the physical spherical/overlap-rank
subspace and respecting frozen virtual orbitals. The numerical PAO nullspace
cutoff is 1e-10. Atomic virtual domains follow normalized-PAO differential overlap;
local auxiliary domains follow the absolute modified Mulliken atom populations.
If either selection is empty, the atom with largest absolute population is
retained. Pair domains are unions. The DOI grid streams 512 points at a time,
using 96 radial and 40 polar Gauss–Legendre points and 80 azimuthal points per
atom, with equal-radius Becke partition weights. Auxiliary domain fits use a
64 MiB bounded cache.

LMP2 retains off-diagonal occupied Fock coupling and overlap transformations
between every pair of virtual spaces. PCG solves the strong-pair Galerkin
residual; weak pairs retain their untruncated semicanonical PAO amplitudes.
Energies use the Hylleraas functional. Pair density selection, virtual
semicanonicalization and the discarded-PNO energy estimate are separate steps.
A zero PNO threshold retains all domain modes, including zero occupations;
a finite threshold may remove an entire pair. These equations follow the
[PSI4 local MP2 theory description](https://psi4.github.io/psi4docs/1.9.x/dlpnomp2.html).

The CC engine includes noncanonical occupied Fock terms in each true CCSD
residual. It projects the residual before preconditioning and DIIS, and projects
the update again. Singles use diagonal-pair PNO spaces. Weak-pair doubles stay
fixed at the converged LMP2 values. CC contractions currently retain the global
RI integral tensor; local auxiliary domains determine the MP2 seed and PNO
selection. This is an explicitly defined pair-Galerkin RI variant. The current
occupied-pair graph is complete, and no linear-scaling performance claim is made.
Pair and spin-orbital CC workspace estimates reject allocations over 256 MiB;
the spatial RI CC integral matrix has a separate 128 MiB bound.

With all selection thresholds zero, full domains and weak-pair approximations
disabled, the method recovers the same-auxiliary RI-MP2/RI-CCSD equations. Tests
include frozen occupied/virtual orbitals and rectangular spherical MO spaces.
A finite-PNO CCSD test compares all AO-transformed singles/doubles amplitudes
with an independent root of PySCF's spin-adapted residual equations. The
molecular finite-domain/weak-pair MP2 oracle instead forms a full ordered tensor
and solves a dense Galerkin system; it does not reproduce the Rust pair PCG.

Triples natural orbitals diagonalize the average of the three converged CCSD
pair densities in the union PAO domain, followed by virtual semicanonicalization.
`t_cut_tno` inherits `t_cut_pno` unless supplied separately. (T0) uses diagonal
occupied Fock elements; (T1) solves the occupied-Fock-coupled equation by PCG,
with three-leg overlap transforms between TNO spaces. This follows the
[noncanonical triples equations of Guo et al. (2018)](https://doi.org/10.1063/1.5011798).
All ordered occupied triples are currently retained. The 256 MiB workspace
estimate bounds their storage; there is no linear-scaling triples claim.
Full and finite-TNO molecular energies are validated independently from PySCF
FCI actions `H*T1` and `H*T2`, with dense Fock resolvents in determinant space.
The full (T1) limit also matches canonical RI-CCSD(T), and full (T0) matches
PySCF's noncanonical diagonal-denominator reference. Model tests check every
triple amplitude and three-leg overlap component against dense Kronecker matrices.

The canonical spatial (T) kernel also had an incorrect connected-term factor.
It now uses the ordered-triple factor 1/3 and three singles/integral terms. A
nonzero-triples PySCF regression replaces reliance on the vanishing H2 limit.
The shared source builder computes each virtual cube once before permutation
contractions, avoiding repeated reconstruction of the same connected numerator.

For continuous local-MP2 and local-CCSD energy surfaces, construct
`LocalSpaceAnchor` at a reference geometry and call
`anchor.compute_mp2(&next_context)` or `anchor.compute_ccsd(&next_context)` at
displaced geometries. Atomic domains, strong/weak pair classifications, PAO dimensions and
PNO subspaces are held fixed. Cross-overlap/polar transport follows the current
HF occupied and virtual spaces. Quaternion rigid alignment and explicit rotation
of Cartesian AO polynomials keep the finite-space model covariant under rigid
translation/rotation. A lost overlap/rank or changed orbital/basis partition is
an explicit error; no clipped eigenvalue or automatic space refresh is applied.
Rebuilding an anchor changes the approximation and requires an energy accuracy
check. This mode assumes the same smoothly followed SCF state and a valid rigid
alignment chart; it does not establish global smoothness across state changes or
ambiguous molecular orientation. Full analytic nuclear response remains unfinished.

For triples, construct `LocalTriplesAnchor::from_context(&reference_context,
&options)` and call `compute_t0` or `compute_t1` at the next context. This selects
PNO and TNO dimensions once, transports both subspaces, reconverges projected
CCSD, and solves the current triples resolvent. A missing fixed triple or lost
overlap rank returns an error. The evaluator does not refresh domain/pair/PNO/TNO
threshold decisions. Semicanonical rotations within retained spaces change only
their representation.

All nine nuclear-coordinate displacements and a combined rigid rotation/shift
match an independent PySCF/Kabsch/dense-Galerkin energy reference. The analytic
polar-frame response is checked in every Jacobian component (370 entries),
including repeated metric eigenvalues, against an independent square-root
Sylvester equation. This validates the transport response, not a complete
relaxed nuclear gradient. The latter still requires the HF, alignment and pair
space responses in P4.

The same 20 geometries independently validate transported CCSD against spatial
PySCF residuals and dense SciPy roots: every one of 49,000 AO amplitude components
passes the unchanged 1e-8 tolerance (maximum error 8.54e-12). CCSD energies agree
within 8.93e-14 Eh. Determinant CI `H*T1/H*T2` moments and dense Fock resolvents
verify both finite-TNO triples energies within 2.68e-15 Eh. See
[`local_continuity_validation.json`](validation.md#external-validation-records).
These checks establish the transported energy model; DLPNO nuclear derivatives
remain `Unsupported` pending the complete relaxed response, including transport.
