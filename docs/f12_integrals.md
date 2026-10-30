# F12 integrals, MP2-F12 and coupled CCSD-F12

Benchmark record filenames and measurement harnesses refer to the
[external validation directory](validation.md#external-validation-records).

The v0.3.0 F12 models are experimental. MP2-F12/3C(FIX) uses the Ansatz-3
projector, complete OBS/CABS Fock coupling, and fixed singlet/triplet cusp
amplitudes. Independent model and molecular references verify its energies and
every intermediate component. Public Psi4 1.11 executions with explicitly
matched factor/integral conventions agree within the recorded tolerances.
Historical printed values and external-program defaults are separate comparisons.
CCSD-F12a/b converge modified
CC amplitude equations and use their distinct uneliminated energy functionals;
optional (T*) uses those amplitudes. Independent molecular models verify every
amplitude and numerical validation-gradient component, including nonlocal ECPs and frozen
cores. Published/default CCSD-F12 printed-digit agreement is unresolved.
Those comparisons are a separate criterion from the passing explicitly specified
model-component checks. Current API controls are listed below; lower sections
also retain dated measurement results with their original model/system scope.

`post_hf::f12::operators` evaluates contracted Gaussian-geminal factors, factors
divided by the electron separation, squared factors and the positive double
commutator `[f,[T1+T2,f]]`. The latter is the negative of `[[T1+T2,f],f]`.
Contractions include cross terms between distinct geminal terms, including terms
with equal exponents. Inputs are checked for finite coefficients/exponents and
supported angular momenta.

Gaussian integrals share Hermite coefficients without multiplying the Gaussian
product prefactors twice. Coulomb-geminal integrals use analytic Boys moments;
the `r12²` factor uses an analytic Gaussian-exponent derivative. Production
integrals do not use radial quadrature or finite differences. The independent Python oracle integrates the
original bivariate polynomial with Gauss-Hermite quadrature and the Coulomb
Laplace integral with Gauss-Legendre quadrature. All Cartesian components of an
angular-momentum-0..7 shell are covered, giving 720 primitive comparisons plus
240 single-commutator antisymmetry/zero-exponent checks and five contracted
operator comparisons. Refining the radial quadrature from 128 to
256 points changes the reference by less than 8e-17.

`CABSBasis::from_reference` constructs the metric complement of **all retained
physical OBS MOs**, with spherical auxiliary transformations when appropriate.
`from_builtin_reference` accepts an embedded auxiliary basis name; the default
uses the actual `cc-pVDZ-F12-OptRI` data. The complement can be empty. Small metric
eigenvalues are removed explicitly, and indefinite metrics return errors.

`extended_fock` builds the union-OBS/auxiliary Fock and exchange matrices from the
HF density, including ECP one-electron terms. Direct mixed integrals are contracted
without storing a fourth-power union-AO ERI tensor. A quadratic workspace guard is
an allocation estimate, not a measured peak-memory claim. CABS singles diagonalize
the complete OBS-virtual/CABS Fock block and use its couplings to occupied MOs;
there is no artificial virtual-energy floor.

Real Hermitian one-electron operators use a symmetric congruence transformation
with compensated sums and FMA product residuals. Only one transformed triangle is
evaluated and mirrored; asymmetric AO inputs are rejected. This preserves the
operator's symmetry and cancellation digits when CABS coefficients are large.
It fixes a ghost-He calculation whose ordinary GEMM transformation crossed the
existing Hermiticity threshold. No metric cutoff or acceptance tolerance is
changed. Independent 70-digit matrix products check all 648 components of two
strongly cancelling positive/indefinite operator models at 1e-10 (maximum error
4.5e-16). The fixtures store input bit patterns to preserve the exact binary64
operands across JSON parser rounding modes.

Four independent PySCF union-AO references cover H2, He with a nonzero local ECP,
HeH+ with spherical d functions (8 Cartesian AOs, 7 physical MOs), and He with a
deliberately duplicated s function (3 AOs, 2 retained MOs). Tests compare every
CABS-projector, Fock, exchange, semicanonical-energy and AO singles-amplitude
component, plus singles energies and the empty-complement limit. The redundant
OBS uses canonical orthogonalization and physical overlap-metric projector and
amplitude components, since raw coefficients can contain an arbitrary null-space
part. The spherical test also checks raw Cartesian components, including the
normalization and d-component order transformation.

## Robust density-fitting integral building blocks

`PreparedGeminalOperator::two_center` and `three_center` evaluate `(P|W|Q)`
and `(mu nu|W|P)` for multiplicative Gaussian F12 kernels. The auxiliary
function has no second orbital factor: an internal unnormalized constant is
used, while ordinary public basis functions still require positive exponents.
The supported kernels are the factor, factor/r12, factor squared and positive
double commutator. The kinetic single commutator is not a multiplicative density
kernel and is rejected by these lower-center/fitting interfaces.

`density_fit::FittedF12Kernel` implements the robust Coulomb-metric formula
`D^T W3 + W3^T D - D^T W2 D`, following
[Knizia (2010), Eqs. 2.8--2.10](https://elib.uni-stuttgart.de/server/api/core/bitstreams/587dac0c-7ea0-47d0-9a5f-92e4fa47e301/content).
It reuses the common DF Coulomb metric and three-center integrals. Retained
metric modes replace an explicit inverse. Storing two rank-by-AO-pair matrices
allows fitted integrals and rectangular occupied-pair transformations without
materializing a fourth-power AO tensor. Construction and transformation have
pre-allocation workspace checks; the default budget is 256 MiB.

`density_fit::F12PairFitter` contracts both orbital legs before storing any
three-center family. Its storage scales with auxiliary rank times target
orbitals times occupied orbitals, rather than auxiliary functions times all
union-AO pairs. Exact zero coefficient rows are omitted without a screening
threshold. Kernels share the Coulomb modes; rectangular target ranges avoid
unused squared-factor blocks. The ordinary Coulomb kernel reuses these modes
without copying them. All four kernels and rectangular blocks are checked
against the independent transformed data below, with sparse coefficient and
empty occupied-space limits covered separately.

Independent public Psi4 MintsHelper calls use the auxiliary/constant bra order
and normalize Cartesian functions through S--G. NumPy evaluates the original
auxiliary-index robust formula and dense four-index transformations. All
52,856 lower-center, fitted-integral and transformed components agree within
5.4e-15. Redundant auxiliary functions, Coulomb fitting, complete-product-space
limits and invalid dimensions/budgets are also checked. Reproduce with
`tests/reference/gen_f12_density_fit.py` and `cargo test --test f12_density_fit_test`.

`F12DensityFitOptions::auxiliary_basis_kind` selects Cartesian or pure spherical
fitting functions (Cartesian by default). The Coulomb metric is restricted to
that physical subspace before rank truncation; the returned whitening map has
Cartesian rows. Both the complete-kernel and occupied-leg interfaces use this
map. Public Psi4 mixed-basis overlaps define an independent spherical subspace.
All 40,912 added kernel, mixed-pair and J/K components agree within 1.2e-14.

### Fixed-density fitted Fock operators

`density_fit::fitted_coulomb_exchange_rhf` stores `B[A,mu*i]` after contracting
the occupied right leg. For `P=2 C_occ C_occ^T`, it contracts the fitted charge
to auxiliary coefficients and streams each Coulomb matrix element; exchange
contracts the occupied-leg modes. Storage is bounded by auxiliary rank times
AO count times occupied count, plus quadratic matrices and integral scratch.
The budget includes metric transforms and parallel angular scratch. This avoids
an auxiliary-by-full-union-AO-pair tensor.

`cabs::extended_fock_fitted` adds the exact kinetic, nuclear and ECP one-electron
terms to these J/K matrices. Its result is the unmodified fixed-density DF Fock
operator, including the fitted OBS block. `compute_cabs_singles_with_fock`
accepts this result with reference occupied energies. The molecular drivers can
use a consistently transformed fitted Fock with an exact orthogonal OBS block,
as described below. The raw supplied operator retains its fitted OBS block.

Public PySCF lower-center arrays and independent original-auxiliary contractions
verify every Fock, half-exchange and metric-weighted CABS response element in six
He2 cases: all-electron/nonlocal S/P ECP, Cartesian/spherical fitting and duplicate
auxiliary shells. All 14,124 primary components pass 1e-10 (maximum 2.31e-12).
Raw AO response coefficients and CABS eigenvalues use the existing CABS 1e-9
criterion; their maxima are 2.41e-10 and 4.81e-10, respectively, owing to overlap
roundoff amplification in nearly dependent auxiliary modes. A complete density
product space recovers the exact Fock. Reproduce with
`tests/reference/gen_f12_fitted_fock.py` and `cargo test --test f12_fitted_fock_test`.

A practical spherical water/cc-pVDZ-F12 check uses 183 union Cartesian AOs,
110 CABS and 136 Cartesian aug-cc-pVDZ-RI fitting functions. All 66,980 Fock,
exchange and scalar components pass 1e-10; the maximum is 7.64e-12. One native
four-worker DF Fock construction took 1.52 s, with 23,536 KiB process peak RSS;
the separate exact-operator run took 30.82 s and 22,440 KiB. Compiler time is
excluded. The DF operator gives a different approximation: CABS singles is
-0.00323615568214 Eh versus -0.00324814923154 Eh with the exact operator.
These fixed-density operator measurements do not establish full-method speedups
or the memory requirements of high-order ECP derivatives. The external record
is `f12_fitted_fock_validation.json`.

### Molecular corrections and separate fitting controls

Set `MP2F12Options::fitting_basis` to an `AuxBasis`, or set the typed option
`f12_fitting_basis` to a built-in auxiliary basis name, to fit the 3C F12
correction kernels. This basis is independent of `cabs_basis`. The reference
SCF, extended Fock/exchange, CABS singles, conventional MP2/CCSD/(T) and the
orbital Coulomb matrix in the Schur complement retain exact integrals.
The projector contractions use fitted Coulomb and robustly fitted factor,
factor/r12, factor-squared and double-commutator kernels. This choice is
explicit: it does not select an external program's DF-SCF/DF-MP2 defaults.
The output component `F12_fitted_correction` records whether it was used.

The following controls also apply to the typed MP2-F12 and CCSD-F12 adapters:

| `MP2F12Options` field | Registry option | Default |
| --- | --- | --- |
| `fitting_basis` | `f12_fitting_basis` | Exact correction integrals |
| `fitting_basis_kind` | `f12_fitting_basis_kind` | `cartesian` |
| `fock_fitting_basis` | `f12_fock_fitting_basis` | Exact extended Fock |
| `fock_fitting_basis_kind` | `f12_fock_fitting_basis_kind` | `cartesian` |
| `fit_orbital_mp2` | `f12_fit_orbital_mp2` | `false` |
| `exact_obs_coulomb` | `f12_exact_obs_coulomb` | `false` |

Basis kinds accept `cartesian` or `spherical`. The two fitting bases and their
physical spaces are independent. `fit_orbital_mp2=true` requires an explicit
correction fitting basis; it uses that same metric for conventional MP2 and the
Schur-complement virtual Coulomb matrix. Conventional CCSD/(T) integrals stay
exact. `CCSDF12Result` exposes both `mp2_correlation_energy` and
`mp2_f12_correlation_energy` (the latter excludes CABS singles), also output as
`E_MP2` and `E_MP2_F12`, so the inputs to the (T*) ratio can be inspected.

`exact_obs_coulomb=true` requires an explicit correction fitting basis and
implements the orthogonal Coulomb-block choice described in Knizia (2010),
Sec. 4.5.1. The raw DF quantities first undergo the consistent CABS transform.
Only then is the projector's OBS/OBS Coulomb block replaced by exact cached
orbital integrals. Its CABS-containing blocks and robust geminal kernels retain
the selected fitting model. This choice is independent of `fit_orbital_mp2`:
the MP2 baseline and Schur G still follow that control. The full-OBS CC projected
operator applies the exact block as an AO exchange-like action, including
nonsymmetric ordered-pair densities, without storing union-AO fourth-power
integrals. `molecular_ccsd_projected_coulomb_fitted_with_options` exposes that
same operator. Additional occupied-leg and per-worker scratch is included in
the pre-allocation workspace estimate. `F12_exact_obs_coulomb` reports the choice.
The default remains `false`; this is an explicit theoretical convention, not
certification of a historical program default or a fit to molecular energies.

Four added independent models cover exact/DF orbital MP2, Cartesian/pure
correction fits, exact/mixed Fock, local/nonlocal S/P ECPs, frozen cores0/1 and
Full/HY1/HY2. Across the ten small models, all 3,680 pair/projected/hybrid and
4,800 CC amplitude/energy entries pass 1e-10 (maxima 7.63e-14 and8.10e-13).
The generator is `tests/reference/gen_f12_exact_obs_coulomb.py`.
Both full He2 dimer/ghost models also pass independent all-component checks:
31,771 entries each, including degenerate virtual-orbital rotations, at 1e-10;
the largest errors are below 4.21e-12. The exact-OBS diagnostic still differs
from the published printed-digit energies by0.0038–0.0052 K, above the fixed
0.00005K criterion. These component checks do not close that literature item.

With a Fock fitting basis, the complete raw AO/auxiliary operator is transformed
consistently to orthogonal MO/CABS coordinates before its OBS block is restored
from the independently checked canonical HF Fock. This follows the block-mixing
principle in [Knizia (2010), Sec. 4.5.1](https://elib.uni-stuttgart.de/server/api/core/bitstreams/587dac0c-7ea0-47d0-9a5f-92e4fa47e301/content).
Write the restoration as `Delta = F_OBS,exact - F_OBS,DF`. The mixed Fock is
`F_DF + Delta`, and the nonlocal term is `K_DF - V_ECP,nonlocal - Delta`.
Consequently their sum remains the local operator used by the independently
integrated squared-factor and kinetic double-commutator terms. Dropping this
finite-rank counterterm changes the B intermediate. CABS singles use the same
mixed Fock. An inconsistent reference is rejected before restoring its block.

Outputs record `F12_fitted_fock`, `F12_fitted_orbital_mp2`, `F12_spherical_fit`
and `F12_spherical_fock_fit` in addition to `F12_fitted_correction`. Fitted
correction/Fock response remains unsupported. The common `density_fitting=true`
flag is not an alias for these explicitly selected F12 approximations.

The same kernel family feeds CCSD-F12a/b. Its full-OBS projected Coulomb action
sums `B_A P(f_ij) B_A^T` over fitted modes; no OBS-squared by RI-squared Coulomb
array is stored. Workspace checks include live pair families, metric/mode
storage and parallel scratch before constructing the extended Fock matrix.
They exclude the reference SCF and conventional MO cache. Full, HY1 and HY2
are supported; fitted 3*A, frozen virtuals, and fitting-space nuclear response
return explicit errors. The common `density_fitting=true` option still returns
unsupported; the explicit F12 controls select their individual approximations.

`gen_f12_fitted_molecular.py` obtains public Psi4 two-/three-center arrays and
uses the original auxiliary-index robust formula. Independent dense product/
geminal Gram and Hamiltonian matrices evaluate the F12 functional; public
PySCF updates solve the modified CC equations. Off-axis He2, with/without
nonlocal S/P ECPs and one frozen occupied orbital, covers all pair and full-OBS
projected Coulomb entries, all CC amplitudes, both variants and Full/HY1/HY2.
All 368 checked molecular/projected/hybrid components agree within 7.6e-14,
and 432 CC amplitude/energy components within 5.1e-13 Hartree. These small
explicitly specified calculations do not certify published/default agreement,
which remains unfinished. Run `cargo test --test f12_fitted_molecular_test`.

`gen_f12_mixed_fock.py` adds five models with distinct correction and Fock
auxiliary bases, Cartesian/spherical spaces and exact/fitted orbital MP2,
including a fitted Fock with exact correction integrals. Public
Psi4 lower-center arrays define the fitted operators. An independent dense
product/geminal Hamiltonian includes the finite-rank restoration; public PySCF
converges the modified CC equations. Nonlocal S/P ECP, frozen cores 0/1,
Full/HY1/HY2, all pair/projected entries, all CC amplitudes and nonzero (T*) are
covered. Across the original and five new models, 2,208 pair/projected/hybrid
components agree within 7.7e-14 and 2,880 CC components within 8.1e-13. Typed
controls and output metadata, invalid fitting spaces, budgets, canonical
references and unsupported response paths are checked separately.

The practical water oracle also exercises a mixed exact-OBS/DF-CABS Fock and
DF-MP2 with 118 pure spherical fitting functions (136 Cartesian working rows).
It verifies all 103,713 pair/projected/CC amplitude/energy components at 1e-10;
the maximum is 8.14e-12. MP2-F12 total energy is -76.358124666663 Eh and
CCSD(T*)-F12b is -76.359089731803 Eh. Reproduce the independent reference using
`gen_f12_fitted_water.py --reuse-integrals --mixed-fock --spherical-fitting
--fit-orbital-mp2` after generating its public lower-center scratch. The existing
`f12_fitted_water_test` checks both the original and mixed models. The auxiliary
metric is solved directly in its original physical basis; this model has
condition number about 2.37e6. This verifies the selected model and does not
certify the historical defaults of the published He2 comparison, whose
remaining discrepancy is recorded below.

After the related regressions finished, separate four-worker native water runs
took 7.91 s for MP2-F12 and 11.11 s for CCSD(T*)-F12b, with peak process RSS
37,924 and 72,688 KiB, respectively. Compiler time is excluded. These are single
runs for the explicitly selected 118-function model, rather than comparisons of
equivalent approximations or measurements of high-order ECP derivatives. Detailed
external records and harnesses are in `f12_mixed_fock_validation.json`.

One native water/cc-pVDZ-F12 run (48 MOs, one frozen core, 110 CABS functions,
136 aug-cc-pVDZ-RI fitting functions, four workers) used 30,852 KiB peak RSS
for MP2-F12 and 69,944 KiB for CCSD(T*)-F12b. Method times were 44.34 s and
42.17 s respectively; the compiler was excluded. These individual measurements
are not a before/after speedup or a high-order ECP derivative memory test.
Water's exact SCF, MP2 and CABS components match the existing reference within
1e-10 Hartree. Its fitted molecular correction is now independently verified
as described below. Detailed logs and the measurement harness are stored externally.

### Molecular water regression

`gen_f12_fitted_water.py` generates a water/cc-pVDZ-F12 reference with 48
spherical MOs, one frozen occupied orbital, 110 CABS functions and 136 Cartesian
aug-cc-pVDZ-RI fitting functions. Public Psi4 shell metadata and independently
computed overlaps check the AO ordering and normalization. Lower-center arrays,
an original-auxiliary robust fit, and product/geminal Gram-Hamiltonian block
operators avoid union-AO fourth-power arrays and dense product Hamiltonians.
The virtual-pair diagonal is applied as a vector. Public PySCF updates solve
the modified CCSD equations with a nonzero (T*) correction.

The fitting metric has condition number about 1.54e9. The oracle solves its
linear equations directly without forming an explicit spectral inverse, and
reorthogonalizes within the retained CABS subspace using extended-precision
Gram products. These steps preserve both metric cutoffs and the retained spaces.
Psi4 integral screening is disabled and its integral tolerance is 1e-16.

`f12_fitted_water_test` checks every V/X/B/C/G, eliminated pair intermediate,
full-OBS projected Coulomb entry, CC single/double amplitude and energy component.
All 103,711 entries satisfy the unchanged absolute 1e-10 criterion. The observed
maximum errors are below 3.4e-12 for MP2/intermediate entries and 8.2e-12 for
CC/projected entries; total energies agree within 8e-13 Hartree. The fitted
MP2-F12 and CCSD(T*)-F12b totals are -76.358279540742 and -76.359227188425
Hartree, respectively. This verifies the explicit fitting choice above;
published/default-option agreement remains a separate unfinished requirement.

### Published He2 interaction-energy diagnostic

The CP-corrected He2/aug-cc-pVTZ comparison at 5.6 bohr uses the four
Table II targets in [Patkowski (2012)](https://doi.org/10.1063/1.4734597).
Both ghost monomers are computed explicitly and share the dimer's triples
scaling ratio. The explicit model uses Full 3C(FIX), beta=1.2, 376 CABS
functions, pure spherical aVTZ-RI correction fitting and aV6Z-RI Fock fitting,
an exact canonical OBS block and exact conventional CCSD/(T) integrals.
All values below are in K, using the paper's conversion factor.

| Variant | Published | Tew fit | Moment-weighted fit |
|---|---:|---:|---:|
| CCSD(T)-F12a | -9.5956 | -9.534141 | -9.575581 |
| CCSD(T*)-F12a | -9.7283 | -9.666286 | -9.707936 |
| CCSD(T)-F12b | -9.4578 | -9.400037 | -9.442305 |
| CCSD(T*)-F12b | -9.5905 | -9.532182 | -9.574660 |

Neither explicit factor choice passes the unchanged 0.00005 K criterion.
The moment-weighted factor comes from an independent function-space fit with
analytic moments and an analytic optimization Jacobian. All twelve beta=1
parameters match the printed values in
[public Molpro2010.1 output](https://www.molpro.net/pipermail/molpro-user/2011-April/004340.html).
Its scaled beta=1.2 use is a diagnostic; historical nonunit-beta and auxiliary
conventions remain unconfirmed. No molecular energies enter this fit, and the
verified Tew default remains in use. Approved BSE He RIFIT data versions from
2008 and 2019 both match the existing inputs numerically; this does not establish
identity with historical Molpro MP2FIT data. Detailed inputs, failed-run evidence,
logs, hashes and remaining questions are in the external
`f12_he2_literature_validation.json`. This comparison keeps P3c incomplete.

## MP2-F12/3C(FIX)

`F12FockSpace` applies `Q = (1-O1)(1-O2) - V1 V2`. Its removed product spaces
are the complete OBS product and the two disjoint occupied/CABS products.
The identity part of `f(F1+F2)f` uses the squared factor and half the positive
double commutator. All projector terms retain the complete RI Fock matrix.
The conventional doubles are eliminated with their canonical energy resolvent,
including both `G*C` and `C*C` contributions. Unordered occupied-pair energies
use singlet and triplet amplitudes 1/2 and 1/4, with the proper diagonal-pair
weight and triplet multiplicity.

For nonlocal ECPs, the local commutator identity subtracts `K-V_ECP,nonlocal`,
rather than only exchange, so the ECP's nonlocal action is explicitly retained
in the RI contractions. Local and nonlocal ECP model cases are independently
verified. This energy work does not add post-HF analytic ECP derivatives.

The Slater factor is `-exp(-beta*r12)/beta`, expanded using Psi4 MintsHelper's
high-precision six-Gaussian fit. Exponents scale by beta squared; coefficients
include the inverse-beta normalization. All six terms at beta=0.7, 1.0 and 1.4
are compared against an independent query. Gaussian products and kinetic
commutators use this same expansion, rather than an unrelated fitted derivative.
Psi4 1.11's complete energy routine uses the raw MintsHelper factor without the
inverse-beta normalization. Nonunit-beta comparisons therefore explicitly supply
that same factor through `custom_geminal`; the normalized default is independently
verified against the radial reference. These conventions coincide at beta=1.

The molecular builder contracts two occupied legs before storing mixed-basis
tensors. Its storage is quadratic in the union AO dimension for a fixed number
of occupied orbitals; no union-AO fourth-power tensor is allocated. Exact
electron-exchange symmetry stores equal-space AO blocks in `N(N+1)/2` rows
instead of `N^2`, and avoids the corresponding duplicate integral evaluations.
The occupied pair indices are transposed when reconstructing the upper triangle;
rectangular spaces and nonlocal acted-orbital families keep their full storage.
Checked
workspace estimates limit the mixed tensor families. The default limit is 256 MiB;
`MP2F12Options::workspace_budget_bytes` (registry option `f12_workspace_bytes`)
sets this bound for MP2-F12 and CCSD-F12. Over-budget inputs fail before the
extended Fock and mixed-integral construction. This bound is not a limit on the
entire process, SCF storage or the later CC amplitude workspace. These guards do not imply
that arbitrary molecular sizes have been benchmarked.

`f12_hybrid="full"` retains the complete 3C residual nonlocal term. The optional
`"HY1"` and `"HY2"` settings implement the approximations in
[Knizia (2010), Section 4.1.5, Eqs. 4.52--4.55](https://elib.uni-stuttgart.de/server/api/core/bitstreams/587dac0c-7ea0-47d0-9a5f-92e4fa47e301/content).
HY1 omits the complete residual `Y=<f|n12 Q|f>` term; HY2 omits its CABS-CABS
right resolution. The minimally projected Fock operator `g` removes occupied/CABS
couplings, which remain in `n=h-g=K-V_ECP,nonlocal+(F-g)`. This keeps the local
Hamiltonian and the physical occupied projector unchanged. HY1/HY2 apply to 3C
and propagate to the coupled CCSD-F12 driver; combining them with 3*A fails.
An independent full product-space oracle checks both spin energies and every
hybrid B component for all occupied pairs, including both signs of nonlocal
operators and off-diagonal occupied/CABS Fock elements. Published hybrid/default
calculations remain a separate, unfinished validation requirement.

Two independent reference constructions cover the complete method equations:

- A full two-particle product-space model applies explicit projectors and
  spin-adapted wavefunctions before solving the conventional amplitude problem.
  Both signs of the nonlocal one-body action and all three occupied pairs are
  covered. Every V/X/B/C/G component and both spin energies agree within 1e-12.
- Five He model calculations use independent PySCF HF/CABS/Coulomb data and
  direct radial integration of the geminal wavefunction's kinetic gradient.
  Extended Gram/Hamiltonian matrices apply the projector and evaluate the
  Hylleraas functional independently of the production intermediate expansion.
  Cases include local/nonlocal ECPs and two Slater-factor scales. Every
  intermediate component and total-energy contribution agrees within 1e-10.
  The largest radial quadrature refinement change is below 5.5e-12.

With exact kernels, a constant Gaussian factor projects to zero for every occupied pair of an
off-axis He2 reference. Fully frozen occupied spaces and malformed public
CABS/HF inputs are also checked. CABS singles contribute to correlation relative
to the original SCF energy, and are separately reported in the typed output.
Both `cabs_singles` and `cabs_core_singles` default to true, preserving the
all-occupied Psi4 convention. Set `cabs_core_singles=false` to omit frozen-core
occupied contributions, as in Molpro's default `CORE_SINGLES=0`; the occupied
projector, full virtual Fock diagonalization and doubles freezing are unaffected.
The typed registry distinguishes a CABS basis from the conventional DF fitting
basis. Frozen virtuals, the generic `density_fitting=true` switch and the
distinct B approximation return errors. Explicit correction/Fock fits and
orbital-MP2/orthogonal-OBS controls are supported as described above; they do
not select another program's complete DF defaults.

Five full Psi4 1.11 calculations independently verify every reported energy
contribution for He/He2, including nonzero triplet correlation, one frozen
occupied orbital and beta=0.7/1.4 raw factors. Maximum observed error across
SCF, conventional MP2, F12, CABS singles, singlet, triplet and total energies is
below 3e-14 Hartree. `tests/reference/gen_psi4_mp2_f12.py` generates the data in
an isolated environment; scratch and logs stay under `target`. Its optional
`--published` case reproduces the public Psi4 water cc-pVDZ-F12/OptRI geometry
using the installed package's basis data, with file hashes in the reference.

The union Fock matrix is constructed once for CABS singles and pair contractions.
Only correlated occupied pairs are stored; frozen occupied orbitals still enter
the reference, strong-orthogonality projector and the Fock/nonlocal action.

The optimized native Linux test binary for the six molecular/API tests, now
including 3*A, peaks at 9,196 KiB RSS with four Rayon workers
(`target/mp2-f12-3a-memory-validation.log`). The earlier unoptimized binary peaked
at 15,332 KiB RSS (`target/mp2-f12-memory-validation.log`). These builds differ;
their RSS ratio is not an isolated algorithmic speed/memory comparison.
This is a small-model measurement, not a large-system memory bound.

## Canonical 3*A(FIX)

`f12_approximation="3*A(FIX)"` selects the distinct A approximation in the common
typed options. The implementation applies the generalized and extended Brillouin
conditions and neglects the exchange commutator in the A intermediate. Its
projected kinetic terms use analytic `[f,T1+T2]` integrals, with explicit bra/ket
kinetic actions. The canonical diagonal energy cancels the occupied-energy X
terms, sets C to zero, and retains the projected single-commutator contribution.
It does not replace the A intermediate with the unprojected double commutator.
The equations follow [Werner, Adler and Manby (2007)](https://doi.org/10.1063/1.2712434),
Eqs. 31, 70 and 94–96, with fixed cusp amplitudes.

Every single-commutator Cartesian component through L=7 is checked by direct
quadrature of the original Gaussian Laplacians. Four additional quartets put
L=7 on all four orbitals and independently verify total polynomial degree 28
(30 after the kinetic action). Explicit full-space Q/commutator matrices
independently verify both spin energies for all twelve model pairs. Five
molecular references, including local/nonlocal ECPs and scaled Slater factors, use direct
radial gradient integration and closed bivariate-Gaussian second moments; energies
and diagonal B intermediates agree within 1e-10. Conventional doubles remain
decoupled, as required by EBC. The same zero/invalid-input checks still apply.

Nonlocal ECPs are supported in both 3C and 3*A. The A approximation neglects the
exchange commutator while retaining the ECP commutator. With the nonlocal
potential V, its extra unit term is `f V f - {V,f²}/2`, and the projector term
uses `[f,V] = f V - V f`. Both electron actions are included. Nonlocal products
are resolved in the finite orthonormal OBS+CABS, consistently with the nonlocal
Fock products in 3C; this is an RI approximation, not an exact complete-space
nonlocal-ECP identity. Unprojected f² integrals are retained. The ket action V|i>
is formed before the mixed integral contraction, without a union-AO N⁴ tensor.

Two noncommuting ECP matrix models check every pair intermediate against full
product-space operators. An additional off-axis He2 reference uses independent
PySCF ECP matrices and direct coupled-Gaussian polynomial quadrature, with both
nonlocal S/P channels, all three occupied pairs, a frozen occupied orbital, and
constant-factor limits for A and C. These pair B intermediates agree within
1e-10. Omitting the nonlocal commutator produces a resolved nonzero difference.
The seven small molecular/API tests peak at 9,612 KiB RSS with four workers;
see `target/f12-nonlocal-memory-validation.log`.

The integral engine now prepares Gaussian product coefficients once per
operator family and shares primitive Hermite polynomials across all fit terms.
The r12² kernel uses an analytic exponent derivative. A same-process kernel
comparison against commit `28570ae` gives median speedups of 3.9–162 on 24
off-axis S/P quartets, with maximum absolute value changes below 3.2e-16.
This microbenchmark does not establish a whole-method speedup or linear scaling.
See `f12_operator_performance.json` and `f12_operators.rs`.

The full public Psi4 water cc-pVDZ-F12/OptRI geometry now passes the comparison
with the tighter-converged Psi4 1.11 reference: all seven energy contributions
agree within 4.2e-13 Hartree. With 50 Cartesian OBS functions (48 physical MOs),
133 raw auxiliary functions, 110 CABS functions and one frozen core orbital,
the optimized native run takes 235.34 seconds and peaks at 31,000 KiB RSS with
four workers (`target/mp2-f12-water-prepared-validation.log`). The earlier run
was stopped without a result, so no completed whole-method baseline speedup is
claimed. The older published output and the tighter-converged Psi4 1.11 fixture
differ by up to 1.5e-10 Hartree; this does not certify every printed digit of that
older output. Model derivative and remaining method benchmarks are separate.

Variationally optimized A geminal amplitudes and linear-RI A are not
implemented. Published/default CCSD-F12 agreement and large-system performance
remain open. The common bindings expose the implemented energy options;
production analytic F12 response is unavailable, and numerical debug gradients
remain explicitly labeled numerical.

## Numerical nuclear gradients

`RHFMP2F12::numerical_gradient(&NumericalGradientOptions)` returns a checked
`NumericalGradientResult`, explicitly labeled `GradientKind::Numerical`, in
Hartree/bohr. It differences the complete total energy, including nuclear
repulsion and CABS singles, at ±h for every Cartesian coordinate. The default
step is 1e-3 bohr. Each point reconverges RHF and rebuilds the moving raw
auxiliary basis, CABS projector, full Fock matrix and pair intermediates. The
supplied CABS metric threshold and reference spherical representation are kept.
OBS or CABS rank changes and all failed/nonfinite energies return errors.
Geometry points run sequentially to bound live memory.

`NumericalGradientOptions` specifies separate SCF energy, residual and density
tolerances (defaults 1e-13, 1e-12 and 1e-12). The new
`scf::run_scf_in_space_with_convergence` entry point accepts these independent
conditions. Existing SCF entry points preserve their original thresholds.
The legacy F12 gradient dispatch now differences reconverged total energies
instead of evaluating a CPHF tangent with a frozen CABS. The CCSD-F12 branch
uses the checked coupled-CC gradient driver described below.

Independent full Psi4 calculations at all 72 displaced geometries provide
five-point references for every gradient component of off-axis He2 with and
without a frozen occupied orbital. At h=5e-4 bohr the largest error is below
9.5e-9 Hartree/bohr; halving h reduces the error by approximately four, as
expected for the central two-point production formula. A separate nonlocal S/P
ECP 3*A case checks every component's step convergence, zero total force and
zero torque. These ECP gradient checks are consistency checks; the independent
complete-program gradient comparison is for the all-electron 3C cases. HF/ECP
analytical derivatives through fourth order use their existing response engine.
The SCF compatibility regression rechecks all 15,150 components, including
spherical RHF/UHF and nonlocal ECPs, with unchanged tolerances. The three small
numerical-gradient tests peak at 8,400 KiB RSS with four workers and one test
thread (`target/f12-strict-gradient-memory.log`); this is not a large-system bound.

## CCSD-F12 coupling blocks

`CcsdF12Coupling` implements the fixed-cusp singles/doubles residual additions
and the current-amplitude energy couplings of F12a/b. Inputs are projected
`<pq|r12^-1 Q f|ij>` and C matrices; the conventional CCSD amplitudes enter as
`tau=T2+T1*T1`. The B variant has its distinct additional projected-Coulomb and
singles contributions. The definitions follow
[Knizia's thesis, Eqs. 4.123–4.127](https://elib.uni-stuttgart.de/server/api/core/bitstreams/587dac0c-7ea0-47d0-9a5f-92e4fa47e301/content).
Independent full occupied-pair tensors and polynomial orbital projections check
all 204 residual components and six energy terms in three models, within 1e-13.
Zero-operator and invalid-input limits are also checked.

`molecular_ccsd_projected_coulomb` builds the full-OBS projected V matrices,
including occupied/virtual mixed blocks. It contracts the removed projector
into an AO pair density and applies an exchange-like Coulomb contraction;
no union-AO or RI fourth-power ERI tensor is stored. Full Psi4 Gaussian
F12/Coulomb operators and independent PySCF orbitals/CABS verify all 128 matrix
components for all-electron and nonlocal S/P ECP He2 cases, with maximum observed
error below 1.2e-15. Constant factors vanish for every pair and bra component.
These blocks now enter true residuals before convergence, preconditioning and
DIIS in the shared CC engine. Same-spin blocks antisymmetrize the spatial source.
Frozen occupied indices map back to full HF MOs; the projector and Fock retain
all occupied orbitals. Conventional CC and local Galerkin paths keep their
previous equations.

The energy uses raw V/B/X before MP2's Schur elimination, with converged
`tau=T2+T1*T1` couplings. F12a/b share amplitudes and have distinct functionals.
`CCSDF12Result::coupling_components` exposes both the common C/tau term and the
additional b term from that same solution. The unified output reports these as
`E_F12_C_tau` and `E_F12_additional_b`; `E_F12_coupling` contains the sum selected
by the requested variant. This permits comparing a/b without rebuilding the
intermediates or reconverging the shared amplitude equations. Both components
and their output entries are checked against independent molecular references.
The old posthoc MP2 addition and bare-geminal b correction are removed. (T) uses
the changed amplitudes, and (T*) scales it with MP2-F12/MP2 correlation energies,
excluding CABS singles. Correlation is relative to the original HF energy,
including CABS as a separately named contribution.

Independent Psi4 AO integrals and a product/geminal Gram Hamiltonian provide
reference operators; PySCF converges the modified CC equations. Four off-axis
He2 cases (all-electron/nonlocal S/P ECP, frozen core 0/1) check all 104 T1/T2
components across a/b, every energy part and nonzero triples. Maximum amplitude
error is below 8.8e-13; acceptance limits remain 1e-9 for amplitudes and 1e-10
Hartree for energies. Zero and constant geminals recover ordinary CCSD. The
generator checks Psi4's single-electron double-commutator normalization against
a centered-S analytic integral.

Additional asymmetric He3 cases freeze one occupied MO while retaining two
correlated occupied MOs. All 84 active amplitude components and the nonzero
triples corrections pass independent references, including S/P nonlocal ECPs.
The (T*) energies are about -2.86e-5 and -2.92e-5 Hartree. Thus the frozen-core
triples mapping is checked beyond the two-active-electron limit.

`RHFCCSDF12::numerical_gradient` rebuilds moving OBS/CABS and reconverges HF and
modified CC equations at each geometry. CC energy/residual tolerances are at
most 1e-13/1e-12; invalid controls, rank changes and failed iterations return
errors. Independently rebuilt Psi4/PySCF five-point energies at 24 geometries
verify all 48 nuclear components for a/b/(T*), ECPs and frozen cores. At h=5e-4
bohr the maximum error is below 1.17e-8 Hartree/bohr. Halving the production
step reduces each case's maximum error by about four. Both atoms are displaced
independently; no component is inferred from symmetry. The unchanged acceptance
limit is 1e-7 Hartree/bohr. These F12 derivatives are numerical; HF/ECP retains
its separate analytic response through fourth order.

All four typed CCSD-F12 adapters preserve the shared MO integral cache.
MP2/CC reuse the F12 factor family and union Fock. Temporary families are
released before allocating CC workspaces; no union-AO fourth-power F12/ERI
tensor is stored. Closed-shell CCSD now uses spatial amplitudes and a spatial
workspace guard of 256 MiB; the full four-virtual block has one sixteenth of
the spin-orbital array's entries. This bound excludes the shared MO integral
cache and the SCF calculation. Published hybrid/default-option comparisons
and large-system performance remain pending. This CC driver accepts
3C. The generic `density_fitting=true` flag and frozen virtuals return
unsupported errors. Explicit correction/Fock, orbital-MP2 and orthogonal OBS
fitting controls are described above. Model equations do
not certify a complete off-the-shelf Molpro default calculation. See
`ccsd_f12_validation.json` for the verification record.

An earlier four-test He2 model-suite measurement, including numerical debug
gradients and the then-current typed APIs, peaked at 11,336 KiB RSS with four
workers and one test thread; the compiler was excluded. It is not a
large-system memory bound or a timing for the current full suite. The earlier
Python package, examples, tests and reference generators passed Ruff at that
checkpoint (106 files);
current interface and release checks are summarized in [validation](validation.md).

Run the native checks in an activated Python environment with an optional Cargo cache:

```sh
PYO3_PYTHON="${VIRTUAL_ENV}/bin/python" \
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/native}" CARGO_BUILD_JOBS=1 \
CARGO_PROFILE_TEST_OPT_LEVEL=3 CARGO_PROFILE_TEST_DEBUG=0 RAYON_NUM_THREADS=4 \
cargo test --offline --test f12_integral_test --test cabs_test \
  --test f12_projector_test --test f12_test --test f12_psi4_test \
  --test ccsd_f12_test --test ccsd_f12_coupling_test --test f12_gradient_test
```

The full water test is intentionally ignored in ordinary runs. Add
`--test f12_psi4_test -- --ignored published_water_cc_pvdz_f12_benchmark`
to run it explicitly. To reproduce the kernel comparison, extract
`git show 28570ae:src/post_hf/f12/operators.rs` to
`target/f12-baseline-operators.rs`, copy the external harness to `target/f12_operators.rs` and compile that copy with
`rustc -O --edition=2024 --extern hf_rs=<native-target>/debug/deps/libhf_rs.rlib
-L dependency=<native-target>/debug/deps`, and run the resulting native binary.

The implemented 3C conventions follow the
[MP2-F12/CABS formulation](https://psi4.github.io/psi4docs/master/mp2f12.html)
and the primary papers referenced above. The water comparison supplements
the independent model/component validation. Remaining technical limitations
are listed in the corresponding method sections.
