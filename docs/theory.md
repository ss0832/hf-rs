# Theory and mathematical conventions

This overview describes the implemented conventions. Method-specific equations,
tests and limits are linked below; an energy implementation does not establish
an analytic response implementation.

## Orbitals, integrals and Hartree-Fock

Real molecular orbitals satisfy $C^TSC=I$. Cartesian Gaussian rows are retained
even when the physical orbital space is spherical. Removing near-dependent
overlap eigenvectors changes the retained subspace; analytic response rejects
unsupported rank/projector changes rather than differentiating an unrelated space.

For RHF, $P=2C_oC_o^T$ and $F=h+J(P)-K(P)/2$. The energy is
$E=E_{nuc}+\mathrm{Tr}[P(h+F)]/2$. UHF uses separate alpha/beta densities,
a shared Coulomb term and spin-specific exchange. Electronic and nuclear terms
must use the same effective charges after an ECP replaces core electrons.

Cartesian Gaussian integrals use Hermite recurrences. The Boys function is
$F_n(T)=\int_0^1t^{2n}e^{-Tt^2}\,dt$. Finite quadrature is not mathematically
exact for arbitrary integrands; numerical branches, recurrences and their
verified limits are tested independently. Gaussian basis-center derivatives
raise/lower polynomial powers algebraically.

## Local and nonlocal ECP response

A scalar semilocal ECP contains a local radial part and angular-momentum
projectors. The nonlocal channels cannot be represented by simply changing
nuclear charges. Basis-center and ECP-center motion both contribute.
The engine differentiates Gaussian integrands analytically and evaluates the
remaining radial/angular quadrature. ECP-center derivatives follow translational
identities for the complete operator, including its nonlocal projectors.

HF response through fourth order includes overlap, core, ERI, ECP and nuclear
repulsion. Finite quadrature error remains; validation checks all requested
tensor components with fixed criteria. Memory measurements have a specified
system/order/worker scope, not a universal process-memory guarantee.

## Stationarity and derivative orders

For a stationary variational energy, the $2n+1$ rule means wavefunction response
through order $n$ suffices to express energy derivatives through order $2n+1$.
Moving-basis overlap/Pulay terms remain necessary. It does not mean higher-order
wavefunction response vanishes. In a stationary constrained Lagrangian that is
linear in its multipliers, the corresponding multiplier response can obey the
$2n+2$ rule. For a fourth energy derivative, the minimum-order construction may
therefore use second-order state response and first-order multiplier response,
under those assumptions. Cancellation of higher response contributions in a
stationary energy expression is distinct from setting those responses to zero.
See [Helgaker's derivative lectures](https://trygvehelgaker.no/Presentations/MWM16_DerivativesProperties_2016.pdf).

The common HF engine deliberately differentiates the density equations order by
order, rather than relying on a minimum-order Lagrangian expression. It enforces
metric idempotency and SCF stationarity for each spin density. Square-free Taylor
variables represent mixed directions, including repeated physical coordinates.
The CPHF factorization is reused at successive orders. No displaced molecular
calculation or finite-difference stencil enters production HF response1–4.
See [analytic HF derivatives](analytic_hf_derivatives.md).

First derivatives of a correlated stationary Lagrangian require relaxed density,
energy-weighted density and two-particle contractions. Higher post-HF response
also needs amplitude/adjoint, orbital and reference-state response. The current
common API connects verified HF and restricted-scope MP2 response. Broader
post-HF analytic gradients/Hessians remain follow-up work; numerical debug
interfaces are labeled numerical.

## CI and multireference energies

The FCI core uses alpha/beta strings and matrix-free Hamiltonian action with
explicit particle numbers. RDMs and spin expectation values follow the same CI
state. CASSCF uses orbital rotations, orbital-gradient convergence and explicit
active spaces. Selected roots remain attached to their state energies and RDMs.

MRCISD spaces are generated from reference excitations. Its Davidson correction
uses the documented relaxed reference weight. Internally contracted CASPT2 uses
all eight excitation classes, removes metric dependencies and applies the
generalized-Fock zeroth-order Hamiltonian. Shifts are part of that explicit
model. See [CASPT2](caspt2.md) and [selected CI](selected_ci.md).

DMRG uses a genuine charge-conserving two-site MPS/MPO with separate alpha/beta
particle-number sectors. Its validation compares the same SZ sector, bond
dimensions and sweep settings, then checks complete-space limits and RDMs.
See [DMRG](dmrg.md). Large-space performance is a separate claim from these
energy/density limits.

## DF and local correlation

Coulomb-metric DF factors use retained physical auxiliary modes of the metric.
The same auxiliary basis and metric convention must be used in independent
RI comparisons. PAOs are formed in the occupied-orthogonal space and
orthonormalized by their overlap metric. PNO/TNO truncation defines a local
Galerkin model; complete thresholds/domains recover the corresponding canonical
RI model. Threshold changes can change the selected space discontinuously.
Fixed/transported spaces and analytic response must account for that dependence;
empirical clipping/smoothing does not supply the missing response.
See [local correlation](local_correlation.md).

## F12

The normalized factor is $f(r)=-e^{-\gamma r}/\gamma$, represented by an explicit
Gaussian expansion. Fixed-cusp singlet/triplet conventions belong to that factor
normalization. The molecular equations require factor, factor-over-distance,
squared-factor and kinetic-commutator operators, a consistent CABS projector and
complete Fock coupling in the chosen approximation.

Raw AO/auxiliary quantities must remain consistent during CABS construction.
After transformation to orthogonal MO/CABS spaces, explicitly selected exact/DF
blocks may be combined. A restored exact OBS Fock block requires the matching
finite-rank counterterm in the nonlocal operator. The optional exact OBS Coulomb
projector is independent of conventional MP2/Schur G fitting. Robust fitting
uses the complete symmetric lower-center expression, not a one-sided fitted
kernel. Occupied-leg contractions and streamed actions bound live storage.

CCSD-F12a/b share modified amplitude equations and differ in their energy
functionals. Their current-amplitude couplings are exposed separately. The (T*)
ratio uses correlation energies with CABS singles excluded. These explicit
models are independently verified, but the historical published/default He2
printed-digit criterion is still not met. See [F12 equations and evidence](f12_integrals.md).

## Driven and flow equations

Single-reference and multireference driven equations use explicit generators,
regularized denominators and converged residuals. MR generalized normal ordering
uses exact RDM cumulants within the documented rank truncation. Reference
relaxation, quadratic BCH and multistate effective-Hamiltonian constructions are
separate models with explicit unsupported sectors. THREE uses the common
factorized/DF infrastructure. See [SR](sr_dsrg.md), [MR](mr_driven.md),
[IM-SRG](imsrg2.md), [multistate](multistate_dsrg.md) and [THREE](three_dsrg.md).

## HF variants and physical properties

Spinor, complex/magnetic, scalar-relativistic, cavity, multicomponent, local
potential and fragment methods have different variational spaces. A familiar
method label does not establish equality with another program's defaults.
Their model-specific limits and validated components are described in the
[catalog](methods.md), specialized documents and [validation](validation.md).
Analytic forces used for geometry optimization must correspond to the same
energy functional and options. Rust and native Python use atomic units. ASE
converts physical input controls and numerical results, including observables,
Hessians and retained integral/intermediate tensors, to eV/angstrom-based units;
input geometry conversion remains in Rust.

## Independent numerical checks

Five-point energy differences validate gradients at step 1e-3 bohr; five-point
differences of independently verified analytic gradients validate Hessians at
step 5e-3 bohr. Molecular/response convergence, translational invariance and all
components are checked with the fixed criteria in [validation](validation.md).
Step refinement distinguishes truncation error from convergence/roundoff error.
Numerical reference differentiation is not a production analytic implementation.
