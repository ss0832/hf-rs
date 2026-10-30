# Generalized normal ordering and MR driven equations

This page describes the typed Rust model and its recorded validation scope.
For the common Python/CLI adapter and analytic-property availability, see
[the current method catalog](methods.md).

The typed registry connects `DSRG-MRPT2`, `DSRG-MRPT3`, `MR-LDSRG(2)` and its
quadratic-BCH `MR-LDSRG(2)+QC` variant to
exact real CI densities through rank three. The molecular adapter currently
supports one singlet CASCI/CASSCF reference with RHF spatial orbitals, with
unrelaxed, once, twice or iterated effective-CAS reference updates.
DF, frozen orbitals and multiple-state variants return
explicit errors in this adapter. The single-state compatibility constructors
`DSRGMRPT2`, `DSRGMRPT3`, `MRLDSRG2::new` and `MRLDSRG2::new_qc` use this same engine with a CASCI
reference. The common Python/CLI `run` interface reaches these typed adapters
and preserves their selected-state/component semantics. Advanced constructor
options outside this adapter's stated scope remain explicitly unsupported.

## Operator conventions and reference spaces

Spin indices are interleaved, `2*p + spin`. A `No2bOperator` stores

```text
O = E + sum_pq f[p,q] {p+ q}
      + (1/4) sum_pqrs G[p,q,r,s] {p+ q+ s r}.
```

Both pairs of `G` are antisymmetric. Scalars, one-body coefficients and every
two-body coefficient are retained without a Hermiticity projection. The
commutator supports arbitrary real operators; the driven solver requires a
Hermitian Hamiltonian.

`GnoReference` embeds an occupied core, an exact correlated active density and
empty external orbitals. Its explicit maps may permute individual spin orbitals.
`GnoReference::from_cas` uses the inactive/active/external ordering saved in the
CAS result. Products involving an external contraction vanish. Core contractions
are exact determinant minors, including the signs from crossing active indices.

The [exact density backend](density_cumulants.md) supplies the active densities.
The algebra uses the generalized-normal-ordering definition of
[Kutzelnigg and Mukherjee (1997)](https://doi.org/10.1063/1.474405).
Normal-ordering a bare three-body operator produces lower-rank coefficients
contracted with the full one-, two- and three-RDMs. Vacuum fermionic
anticommutation relations generate the commutator. Induced bare rank-three
terms contribute to all retained ranks before their GNO rank-three remainder
is discarded. Uncontracted rank-four products cancel in the commutator; no
rank-four tensor or full Fock-space matrix is allocated in production.

## Driven equations and perturbation orders

The generator contains core/active holes to active/external particles, excluding
purely active amplitudes. With `A = T - T^dagger`, the recursive transformation is

```text
C0 = H
Ck = [C(k-1), A]_GNO(0,1,2) / k
Hbar = sum_k Ck.
```

The source condition is `Hbar_ex = (Hbar_ex + Delta*t_ex)*exp(-s*Delta^2)`.
Pulay acceleration solves its regularized fixed-point equations; convergence
requires amplitude and physical-source residuals and energy change. See
[Li and Evangelista (2016)](https://doi.org/10.1063/1.4947218), Eqs. 16, 18,
24--27. Zero flow leaves the reference unchanged. Stable `expm1`/Taylor factors
give the continuous zero-gap limit.

For the Fdiag partition, `H0` is the diagonal semicanonical generalized Fock
operator and `H1 = H - H0`, with its reference scalar removed. Expanding the
same driven equations through third order gives the unrelaxed perturbative
energies. At each order the linear amplitude equations are triangular: doubles
are determined first, and `[H0,A2]` supplies the active-density contribution to
singles. Omitting this contribution is incorrect when the active density and
diagonal Fock do not commute. The third-order nested products use the same
rank-two truncation; this is the approximation discussed in
[Li and Evangelista (2017)](https://arxiv.org/pdf/1701.02011), Eqs. 8--24 and 27.

The molecular adapter diagonalizes the core, active and external blocks of the
singlet generalized Fock matrix. It transports the CI coefficients exactly by
Givens rotations of the spin strings. This preserves the physical state without
an overlap matrix of determinant pairs or cloning the AO reference/ERIs.
The initial normal-ordered scalar is checked against the CAS reference energy. The output includes `E_CASCI` or `E_CASSCF`, `E_PT2`, and
`E_PT3_increment` for MRPT3; `e_corr` remains relative to SCF.

`typed_mr::numerical_gradient` reconverges SCF, CAS and driven amplitudes at each
displacement. This route is numerical, including when CASSCF orbitals are used.
The reference CAS density is not returned as a relaxed correlated density.

## Quadratic-BCH variant (QC)

The project's existing QC API specifies a BCH depth of two. With `P2` retaining
GNO ranks zero, one and two, its transformed Hamiltonian is

```text
C1 = P2[H,A]
C2 = P2[C1,A]
Hbar_QC = H + C1 + C2/2.
```

This finite polynomial follows by retaining k=0,1,2 in the BCH series and the
rank-two projection of
[Li and Evangelista (2016)](https://arxiv.org/html/1602.05667), Eqs. 11 and 18.
The Gaussian source equations, single/double amplitudes and reference-CI updates
are still solved to convergence. Two nested commutators do not mean two
amplitude iterations. `bch_order` reports the fixed depth two; maximum BCH order
and BCH convergence threshold are not options for this method. Other common
flow, amplitude and reference-update controls retain their meanings.

This documents the project's quadratic-BCH convention. Recursive quadratic
commutator theories retaining additional induced higher-body terms require a
separate implementation. The present QC implementation uses the same exact GNO
rank-two algebra and density cumulants as the complete recursive-BCH method.

Independent full-Fock operator maps and nonlinear roots compare every generator
and transformed coefficient. Molecular checks cover CASCI, optimized CASSCF,
rectangular MOs and nonlocal ECPs in none/once/twice/iterated modes. The existing
numerical derivative route remains labeled numerical and is checked on every
Cartesian coordinate. Expanded analytic response is scheduled for the next
version. Detailed records
are in the external `mr_qc_validation.json`; see
[external validation records](validation.md#external-validation-records).

## Reference relaxation

`reference_relaxation` accepts `none` (default), `once`, `twice` and `iterate`.
The compatibility boolean `relax_reference=true` selects `once` when the explicit
mode is `none`. An explicit non-`none` mode takes precedence. The controls
`max_reference_iterations=30`, `reference_energy_tol=1e-10` and
`reference_ci_tol=1e-8` apply to convergence of the iterated reference.

Each update de-normal-orders the retained transformed Hamiltonian and contracts
its occupied core, then solves its general spin-orbital CAS Hamiltonian in the
singlet sector. Its two-body coefficients are Hermitian and pair-antisymmetric;
they need not obey the additional Coulomb eightfold symmetry. Repacking them as
spatial Coulomb ERIs would lose valid effective interactions. The CI solver uses
streamed operator actions and the existing dense/Davidson spin-targeted solver.
This implements the effective-CAS equation of
[Li and Evangelista (2016)](https://arxiv.org/html/1602.05667), Eqs. 7 and 29,
and the perturbative projection in
[Li and Evangelista (2017)](https://arxiv.org/pdf/1701.02011), Eq. 26.

The perturbative effective operator is expanded consistently through order two
or three. Define `C01=[H0,A1]`, `C02=[H0,A2]`, `C11=[H1,A1]` and
`C011=[C01,A1]`, with the same GNO rank-two truncation as the energy equations:

```text
K2 = C11 + (1/2) C011
K3 = [H1,A2] + (1/2)[C01,A2] + (1/2)[C02,A1]
     + (1/2)[C11,A1] + (1/6)[C011,A1]
Hbar(<=2) = H + C01 + K2
Hbar(<=3) = H + C01 + C02 + K2 + K3.
```

The unknown `[H0,A2]` at second order and `[H0,A3]` at third order have zero
CAS projection: their generators change core/external occupations, whereas the
block-diagonal `H0` preserves them. These APIs return only the active effective
operator, and do not claim to provide missing higher-order external blocks.

After a CI update, its exact densities rebuild the physical generalized Fock,
block semicanonical orbitals, denominators and driven equations. Active orbital
rotations transport the saved CI vector; solving the bare CAS again would erase
the reference update. `twice` performs two updates; `iterate` requires both the
maximum phase-aligned CI coefficient change and energy change to converge.
Failure to converge returns an error.

`E_CASCI`/`E_CASSCF` remains the initial CAS energy. `E_unrelaxed` is the first
macroiteration's unrelaxed driven energy, `E_reference_relaxation` is the change
from that energy, and `E_reference_source` is the physical reference expectation
used by the last driven solve. The final CI vector is a reference wavefunction;
it is not reported as a relaxed correlated density. Numerical gradients reconverge
the reference updates at every displacement. Energy stationarity with respect to
all reference/orbital variables and analytic response are not claimed.

## Memory and scope

The present algebra is a dense small-system correctness backend. Its streamed
connected CAR products have worst-case O(N_spin^7) work; it does not claim the
factorized O(N^6) scaling of the published production algorithms. Packed pair
and triple matrices use `C(N_spin,2)^2` and `C(N_spin,3)^2` coefficients.

`GnoAlgebra::estimated_workspace_bytes` includes cached contractions, temporary
rank-three storage, operator arrays and term/index workspaces. Construction
rejects requests above the caller's budget before allocating these arrays; the
typed adapter applies a 256 MiB estimate before CAS/integral work as well.
This estimates new workspace, excludes borrowed inputs and is not a process
RSS cap. CAS/transform temporaries are released before BCH evaluation; full
operator/density workspaces are dropped before effective-CAS diagonalization and
before the next reference update. General-CI term/address, eigensolver and spin
workspaces have separate checked estimates. Further
factorization and large-system scaling remain pending.

## Independent checks

The GNO oracle builds full Fock creation matrices and solves a dense linear
system in the complete rank-zero through rank-three normal-product basis.
BCH values use a SciPy Pade superoperator exponential. The driven oracle uses
an independent SciPy nonlinear root solver; refined finite-coupling polynomial
fits supply first/second-order generators and PT2/PT3 energies.

All components are checked for correlated states, ensembles, Slater/vacuum
limits and permuted core/active/external spaces. Coherent states with identical
one/two-RDMs and different three-RDMs produce different commutator scalars.
The molecular oracle obtains integrals and CAS matrices from PySCF. Small CAS
matrices are fully diagonalized and their CI residuals checked explicitly;
energy-only CAS convergence is insufficient for this comparison. The optimized
reference also uses an independent SciPy root refinement of the PySCF orbital
gradient with exact CI coefficients, checked below 1e-11.

See [the reproducible validation record (`mr_driven_validation.json`)](validation.md#external-validation-records) for
component counts, numerical errors, gradient checks, commands and measured RSS.

The [reference-relaxation validation record (`mr_reference_relaxation_validation.json`)](validation.md#external-validation-records)
adds arbitrary effective-operator CI matrices, every CI-rotation coefficient,
non-Coulomb spin-scalar interactions, effective-CAS coefficients and all
reference-relaxation modes. Its molecular checks include rectangular MOs,
nonlocal ECPs and optimized CASSCF orbitals.

## Compatibility constructors

The four single-state constructors map `MRDSRGOptions` into the verified typed
controls. `RelaxRef` maps directly to none/once/twice/iterate, and `tol` controls
CI, amplitude, energy and reference convergence. Their existing default remains
`Once`; the typed API defaults to `none`. The default `ThreePDC::MK` selector now
uses exact CI cumulants in these constructors. Other partitions, approximate
cumulant selectors, NIVO, sequential transformations and spin-unbalanced inputs
return errors before active-space allocation.

These paths build only the active spin strings and borrow the AO reference. They
use the actual rectangular MO coefficient dimensions, and avoid the former
full-orbital determinant enumeration in the single-state compatibility path.
`MRDSRGResult::to_output` reports `e_corr` relative to SCF and includes the CAS
energy and driven diagnostics. The compatibility `correlation_energy` field
retains its CAS-relative meaning. `one_rdm_mo` is the final *reference* density,
with its frame in `reference_mo_coefficients`; the result trait and standard
output return no correlated RDM or correlated natural occupations.

All 36 unrelaxed/relaxed energies of the original three compatibility methods
and all 486 AO reference-density components are checked against the independent molecular data. The selected
legacy DSRG smoke route is exercised separately. See
[the compatibility validation record (`mr_compatibility_validation.json`)](validation.md#external-validation-records).

The [THREE-DSRG-MRPT2 route](three_dsrg.md) now streams three-index Cholesky or
Coulomb-metric DF integrals through CASCI, amplitudes and effective-CAS reference
updates. It no longer reconstructs a full four-index integral tensor or invokes
the full-space GNO backend. Nine compatibility checks retain full/rectangular
MOs and nonlocal ECPs in none/once/iterate modes. The typed route additionally
supports an explicit auxiliary basis and conventional CASSCF starting orbitals;
its supported options and memory/derivative limits are documented separately.

The [multistate DSRG methods](multistate_dsrg.md) now share exact ensemble
cumulants and the factorized rank-two effective-operator backend. Full-CAS
`SA-DSRG-PT2` and contracted `SA-DSRG-PT2c` are separate registry entries.
MS/XMS/DWMS include both one- and two-body effective terms, exact CI orbital
transport, selected-root output and explicit singlet targeting. Their source
ensembles and final diagonalization spaces are documented in that method guide.
