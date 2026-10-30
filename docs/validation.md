# Validation

This document records reproducible validation commands, acceptance criteria and measured results.
This document records completed checks; the presence of an implementation alone does not establish validation.

## v0.3.2 angular coverage and inspection APIs

Completed independent integral checks cover all 16 s/p/d/f pair combinations
and all 256 angular ERI quartets on four distinct noncoplanar centers.
The public PySCF Cartesian integral interface supplies reference values;
primitive/contraction normalization and the Gaussian/Molden d ordering are
matched explicitly. Every shell uses two primitives of mixed coefficient sign.

| Integral | Compared components | Maximum absolute difference |
| --- | --- | --- |
| Overlap | 400 | 7.22e-16 |
| Kinetic | 400 | 2.59e-15 Hartree |
| Point-nuclear attraction | 400 | 1.16e-13 Hartree |
| Scalar ECP, including nonlocal s/p/d/f channels | 400 | 4.52e-16 Hartree |
| ERI, scalar engine | 160,000 | 3.09e-15 Hartree |
| ERI, shell engine | The same 160,000 | 3.14e-15 Hartree |

Analytic bare-integral derivatives through order four are checked against
five-point differences of the next lower derivative at h=1e-3 bohr.
All 400 Cartesian components of each pair operator are checked for four
direction patterns: pure x/y/z plus a mixed pattern. ECP mixed derivatives
move the potential center. For each of the 256 ERI angular quartets, a
nontrivial representative Cartesian member is checked for the same orders
and pure/mixed patterns. This gives 29,696 derivative-component comparisons.
It is not a test of all 160,000 ERI values times every nuclear multi-index.
The fixed maximum-error criterion is 2e-7; every tested component passes.

### C1 complete-d-shell energy response

Six models use the noncoplanar four-nucleus HeH3 geometry from the s/p
regression, now with a full d shell on He, one He s function and one s
function on the first hydrogen. The other two hydrogen nuclei carry no
AOs, but enter every electronic/nuclear operator and all twelve coordinate
derivatives. Thus the Cartesian working space has eight AOs and the real-
spherical physical space seven. The model ECP has a genuine nonlocal d
channel and local l=3 channel. Nonzero occupied d-subspace projections
(0.24 through 0.62) agree with the independent PySCF projections.

For every model, independent PySCF energies, analytic gradients and Hessians
from five-point differences of those external gradients pass their fixed
criteria. Every native nuclear energy derivative is then checked through
order four: 12 gradient, 144 Hessian, 1,728 third-order and 20,736 fourth-
order entries per model. References are not symmetrized after construction.

| Physical space / model | Gradient | Hessian | Third order | Fourth order |
| --- | --- | --- | --- | --- |
| cartesian ae rhf | 6.042e-12 | 8.132e-12 | 1.683e-11 | 1.103e-10 |
| cartesian ecp rhf | 5.512e-12 | 6.889e-12 | 2.123e-11 | 1.520e-10 |
| cartesian ecp uhf | 5.415e-12 | 4.167e-12 | 7.942e-12 | 1.093e-10 |
| spherical ae rhf | 3.732e-12 | 1.429e-11 | 6.988e-11 | 2.099e-10 |
| spherical ecp rhf | 6.278e-12 | 1.334e-11 | 6.824e-11 | 2.589e-10 |
| spherical ecp uhf | 4.966e-12 | 2.765e-09 | 5.399e-11 | 1.130e-10 |

Errors use Hartree/bohr^n at order n. Fixed criteria remain
1e-6 / 1e-5 / 2e-7 / 3e-6; the reference step is 1.25e-3 bohr.
SCF energy/residual/density thresholds are 1e-15 / 1e-13 / 1e-13.
The seven new d-shell tests pass, including the independent anchor test.
The complete d run takes 1,705 seconds and peaks at 46,220 KiB RSS
with three test threads and twelve Rayon workers. This small model is
not a process-memory guarantee for arbitrary molecules.

```sh
cargo test --release --test shell_eri_test -- --nocapture
cargo test --release --test spherical_all_components_test c1_d_ -- --nocapture
python tests/reference/gen_spherical.py --spdf-only
python tests/reference/gen_spherical.py --c1-d-only
```

### C1 mixed-s/p/d/f energy response

Three additional four-nucleus C1 tests use one normalized primitive
Cartesian component per center: s on He, p-y on Li, d-yz on Be and f-xyz
on B, with exponents 0.8, 1.0, 1.2 and 1.4 bohr^-2. The positions are
the same noncoplanar coordinates as the complete-d tests. The charge is
+10 for the four-electron RHF model and +11 for the three-electron UHF
doublet. These intentionally small ionic models exercise all mixed angular
combinations in a four-AO response space. They do not assert a complete
spherical-f-shell or an 80-AO full-SPDF nuclear-response benchmark.

The zero-core model ECP is centered on Be, with local l=4 and nonlocal
s/p/d/f channels. All twelve nuclear coordinates are differentiated,
including basis-center and potential-center motion. At every order each
entry is compared with the unsymmetrized five-point next-lower reference:
12 gradient, 144 Hessian, 1,728 third-order and 20,736 fourth-order entries.

| C1 mixed-SPDF model | Gradient | Hessian | Third order | Fourth order |
| --- | --- | --- | --- | --- |
| All-electron RHF | 1.281e-11 | 1.394e-11 | 6.110e-11 | 5.572e-10 |
| ECP RHF | 1.916e-11 | 9.780e-12 | 6.148e-11 | 5.538e-10 |
| ECP UHF doublet | 1.882e-11 | 6.544e-12 | 3.455e-11 | 3.893e-10 |

Errors use Hartree/bohr^n. Fixed acceptance criteria remain
1e-6 / 1e-5 / 2e-7 / 3e-6. The step is 1.25e-3 bohr.
SCF energy/residual/density thresholds are 1e-14 / 1e-13 / 1e-13;
the energy threshold stays above the rounding floor of these positive-
energy ionic models and is tighter than the original 1e-12 criterion.
The three-case run completes in 3,610 seconds with 3 test threads and
12 Rayon workers; this timing is specific to the stated small models.

A separate independent PySCF public SCF driver uses projected Cartesian
integral callbacks for exactly these four functions. All three energies
agree within 9.24e-14 Hartree; all 36 gradient anchors agree within
2.19e-11 Hartree/bohr against five-point external energy differences at
h=1e-3 bohr. No external implementation source is used.

```sh
cargo test --release --test spherical_all_components_test c1_mixed_spdf_ -- --nocapture
python tests/reference/gen_spherical.py --c1-spdf-only
```

### Final interface and documentation checks

- The final native regression passes 104 tests: 63 library, 18 API,
  7 basis/ECP assignment, 8 context/options, 1 Molden, 5 shell-integral
  and 2 typed post-HF tests. These include spherical/custom/mixed-spin
  integral frames, fixed-MO derivatives, RI factors and raw RI response,
  aggregate copy/workspace bounds and reconstruction of CCSD energy.
- The separately run C1 d-shell tests pass 7 cases, the mixed-SPDF
  full nuclear tensors 3 cases, and its independent anchor test 1 case.
  They are distinct focused runs, not a claimed fresh full-suite total.
- Native Python executes 30 HF property examples and all 50 connected
  post-HF models. ASE executes the same 80 examples with physical unit
  conversions. The Rust documentation loop also executes all 50 connected
  post-HF methods. Full-active-space H2 cases are interface/limit checks,
  not substitutes for the independent scientific fixtures.
- A normal ABI3 wheel installs into a separate environment and passes
  all 21 Python/ASE tests outside the checkout. Ten installed CLI checks
  pass, including mixed basis/ECP gradients, the embedded oracle, input
  metadata, physical observables and retained CCSD T1/T2 output.
- All 30 Python documentation blocks and 10 Rust examples compile/run.
  Code-derived metadata describes 86 registered input keys and 29 HF
  model adapters. Existing Rust docs describe every public HF options
  struct and the additional positional model controls.

Workstation paths and identities from raw test/install logs are omitted
from these numerical summaries. The distribution retains 653 files and
24 existing documentation pages; no source/test/doc files are added.

Numerical values and test summaries above intentionally omit workstation
usernames, home directories, absolute paths and environment-specific log
headers. Reproduction commands are relative to the extracted source.

The earlier v0.3.0 full-suite total below remains a separately labeled
baseline; it is not a fresh v0.3.2 full-suite count. General fourth-order
response for every HF variant/post-HF model is not implied. The F12
published/default printed-digit discrepancy remains experimental.

## v0.3.1 interfaces and low-symmetry response

This update adds explicit HF observable maps, SCF wavefunction/Molden exports,
additional specialized HF adapters and the uniform `hf-rs` command name. Rust
and native Python observable values use atomic units. ASE converts physical
options and result/state/component energies, observables and analytic Hessians
to eV/angstrom-based units. See the [observable table](methods.md#hf-observables)
and [output representation](python_api.md#file-and-wavefunction-output).

Completed interface checks:

- 21 targeted native integration tests pass in the API, basis/ECP-assignment
  and Molden binaries (13 + 7 + 1). They verify opt-in observable maps against
  typed results and model energies, missing/nonfinite quantities, particle/FMO
  decomposition, specialized HF models, SCF-cache identity, complete orbital
  metric/density identities and mixed/spherical/ECP wavefunction exports.
- 19 native Python/CLI/ASE tests pass in the development environment and after
  an ordinary wheel installation into a separate environment outside the checkout.
- The installed wheel passes nine CLI checks, including the embedded oracle,
  method catalog, HF observables, mixed basis/ECP gradients and invalid methods.
  The final FMO3 schema is independently exercised through the installed native
  binding, including trimer indices, internal energies and connected corrections.
- The Cargo-installed `hf-rs 0.3.1` runs outside the checkout. All four embedded
  oracle cases pass; the largest energy error is 7.68e-13 Hartree. Mixed orbital
  bases and independent named/file ECP inputs agree with the independent fixture;
  the maximum gradient error is 2.05e-10 Hartree/bohr. The added pcseg-0 energy
  agrees with PySCF within 1.83e-14 Hartree.
- All 25 Python example blocks in README and the Python/ASE/method recipes run.
  All eight Rust examples in the Rust API and method recipes compile and run.
- All-target Clippy passes with `-D warnings`. Ruff passes on 111 files; all are already formatted. Python 3.8 syntax passes
  on 110 source files. All 213 local documentation links/anchors pass. The
  source distribution still contains 653 files and
  24 existing documentation pages; no new source/test/document files are added.

### Strict C1 all-component derivatives

The noncoplanar HeH3+/HeH3 s/p-AO cases in
`tests/hf_all_components_test.rs` have seven Cartesian AOs and no nonidentity
molecular symmetry. The model ECP includes local and nonlocal s/p terms on the
helium center. For each case the comparisons include every one of 12 gradient,
144 Hessian, 1,728 third-order and 20,736 fourth-order entries, without
symmetrizing the finite-difference reference after construction. Details are in
[analytic HF derivatives](analytic_hf_derivatives.md#low-symmetry-all-component-regression).

| C1 model | Gradient max error | Hessian max error | Third max error | Fourth max error |
| --- | --- | --- | --- | --- |
| All-electron RHF | 6.70e-12 | 1.34e-9 | 3.04e-9 | 1.44e-8 |
| Nonlocal-ECP RHF | 5.98e-12 | 6.25e-9 | 4.40e-10 | 6.64e-9 |
| Nonlocal-ECP UHF doublet | 2.71e-11 | 3.47e-8 | 2.38e-8 | 1.20e-6 |

All three C1 cases pass all fixed component-wise criteria (3 passed, 0 failed).
All errors use Hartree/bohr^n at derivative order n. Fixed criteria remain
1e-6, 1e-5, 2e-7 and 3e-6 for orders one through four. The five-point reference
step is 1.25e-3 bohr with SCF energy convergence 1e-15 Hartree. These results
establish the listed C1 s/p-AO models, not every possible basis, angular momentum,
geometry or overlap-rank change. General fourth-order HF-variant and post-HF
response is not implied. The F12 published/default printed-digit discrepancy
remains unresolved and experimental.

## v0.3.0 release baseline

The following records the completed v0.3.0 scientific and installation checks.
Its counts and timings retain their original models and measurement scope;
they are not new full-suite totals or current performance promises. The current
capabilities are listed in [methods](methods.md).

| Area | Verified scope / limitation at v0.3.0 |
| --- | --- |
| Common Rust/Python/ASE/CLI | Native `run` and cached `Molecule`; Rust-only input conversion; strict option validation; selected roots/states/components preserved |
| Interface verification | 30 targeted native tests in 5 binaries; 16 optimized Python/CLI/ASE tests, including mixed bases/ECPs, fourth response, BFGS and HI/ECP compatibility; installed Python CLI works outside checkout |
| Static checks | All-target clippy with `-D warnings`; Ruff 111 files including the assembly helper; Python 3.8 grammar: 110 sources; 176 local documentation links/anchors |
| Documentation and language | All 24 Markdown pages in `docs/` and README checked against current interfaces, limits and recorded evidence; 8 Python example blocks executed; all 653 distributable source files contain no Japanese text or filenames |
| Basis/ECP catalog | 80 orbital and 8 auxiliary families; 66 added official BSE JSON/Gaussian 94 exports with original element coverage, version and checksums; element-specific orbital and independent ECP inputs |
| HF response | Analytic RHF/UHF derivatives 1–4 including local/nonlocal ECP; retained-space/rank limits stay explicit |
| Common MP2 response | Existing analytic full-rank all-electron, unfrozen, non-DF gradient/Hessian; no additional post-HF response claim |
| Other post-HF production response | Unavailable in the common API; low-level numerical verification routes remain numerical |
| F12 components | Both He2 dimer/two-ghost models pass 31,771 components each at 1e-10; maxima below 4.21e-12; small exact/DF/frozen/ECP/hybrid models also pass |
| F12 published/default energies | Unresolved: exact-OBS diagnostic errors 0.0038–0.0052 K exceed the fixed 0.00005 K printed-digit criterion; model validation does not close this item |
| Memory/performance | Per-system/order/worker measurements only; no universal 256 MiB process cap, no global linear-scaling claim, and no equivalence between F12 harness RSS and the historical high-order ECP workload |
| Installed Rust CLI | Updated v0.3.0 runs outside the checkout; all 4 embedded cases pass (maximum error 7.68e-13 Eh); mixed basis/ECP maps, independent ECP files and added pcseg-0 pass independent checks |
| pip-installed Python CLI | Ordinary ABI3 wheel installed into a separate venv; 6 checks pass outside the checkout, including mixed basis/ECP gradients and failure on unknown methods |
| Full release suite | 541 passed, 0 failed, 2 intentionally ignored; all 102 integration binaries, library tests and documentation tests completed; no filtered tests |

```sh
CARGO_BUILD_JOBS=4 RAYON_NUM_THREADS=8 RUST_TEST_THREADS=1 cargo test --release --offline --no-fail-fast
cargo clippy --all-targets -- -D warnings
ruff check tests python scratch
ruff format --check tests python scratch
maturin develop --release
python -m unittest discover -s tests -p 'test_python_*.py' -v
python -m pip install .
hf-rs --version
hf-rs --method hf
hf-rs --oracle-compare
```

The oracle, Python/ASE and native commands require the documented build/runtime
dependencies. Scientific reference generators additionally require their stated
oracle packages; ordinary Rust tests use the committed independent fixtures.

The pip check built an ordinary wheel through the Maturin PEP 517 backend,
installed that wheel into a separate venv and invoked its `hf-rs` console script
from `/tmp`. The imported package was under that venv's `site-packages`, with
no editable source installation. The checks cover version/help, H2 RHF, the
added pcseg-0 basis, NaH with independent element-specific basis/ECP selections
and all six gradient components, and an unknown-method error. Energies and
gradients agree with the independently checked installed native CLI. Existing
NumPy/pip packages were reused without downloading dependencies. This was
verified on Linux x86-64 under WSL with Python 3.14. The wheel targets Linux
x86-64 with the CPython 3.8+ ABI3 interface. Both the project license and full
BSE BSD 3-Clause notice are present in the wheel and installed metadata. The
BSE notice matches the pinned license bytes and SHA256 exactly.

The 653 distributable source files were additionally scanned for personal
absolute paths, private workspace identifiers, private keys, common service
tokens, credential assignments, private IPv4 addresses and email addresses.
The final source scan reports no matches. Environment-specific oracle paths
were replaced with `--psi4-python` or `HF_RS_PSI4_PYTHON`; reproduction commands
use the active virtual environment and configurable Cargo cache. The three
oracle launchers pass help/missing-runtime checks, and the PowerShell assembly
helper passes syntax validation. The scan is pattern-based and covers the
distribution files. Public scholarly names and copyright notices are retained.
Document date annotations were removed while the original numerical exports
and their version/checksum provenance remain intact.

The reference list distinguishes mathematical papers, related development
philosophy and [software used for oracle validation](references.md#numerical-oracle-software).
Bibliographic checks use publisher/registry metadata and primary manuscripts;
software citations acknowledge numerical comparisons, not implementation reuse.

## Final release regression

The unfiltered release run completed all 102 integration binaries plus the
library and documentation tests: 541 passed, 0 failed and
2 intentionally ignored. This count is Rust test functions;
the all-component loops compare additional tensor and matrix entries within
each test. The ignored entries are: `profile_cartesian_core_costs`, `published_water_cc_pvdz_f12_benchmark`.
The manual timing test and full cc-pVDZ-F12 benchmark retain their explicit
opt-in status. The raw log and `p8-release-test-summary.json`, including the
log SHA256 and every test group's counts, are kept with the external records.

The run used Linux x86-64 under WSL, Rust/Cargo 1.97.1, Python 3.14.4, four
build workers, eight Rayon workers and one Rust test thread. All Cartesian
HF derivative components through fourth order, including scalar nonlocal ECP
RHF/UHF models, passed. Real-def2 ECP, molecular property workflows, the
80-basis catalog and HF-variant regressions also passed. No scientific
acceptance criterion was relaxed. The experimental F12 published/default
printed-digit discrepancy remains unresolved as stated above.

## Reading the earlier scientific records

The individual sections retain component-level evidence and reproducible inputs.
The current source has evolved after some measurements. Method settings, physical
basis spaces and numerical error must match before interpreting timings as a
before/after speedup. Windows execution restrictions described historically were
resolved for verification by using WSL, not by weakening signing policy.

## Expanded BSE catalog and independent ECP inputs

The 66 additional orbital exports bring the built-in orbital catalog to 80;
the eight existing auxiliary families remain separate. Official BSE JSON and
Gaussian 94 exports retain full element coverage and per-family versions.
Their SHA256 manifest and BSD notice are in `third_party/basis_set_exchange`.
The native tests compare every nonzero primitive exponent/coefficient pair
across both formats and every scalar ECP channel/radial term. The independent
shell manifest and normalization checks cover all 80 orbital families.

Element-specific orbital maps, associated ECP defaults, independent named/file
ECPs, explicit all-electron overrides and malformed inputs are checked at the
native boundary. This exposed and fixed case normalization for multiletter
element symbols in BSE JSON. Four independent PySCF models cover mixed-basis
water, NaH with associated LANL2DZ ECP, a different orbital basis with explicitly
selected ECP, and the all-electron override. All energies pass 1e-8 Eh and all
27 gradient components pass 1e-6 Eh/bohr; translation sums remain below 1e-9.
A separate local/nonlocal model ECP file preserves analytic fourth response.
Python tests cover the same selection through the native API, CLI and ASE.
The installed Rust CLI repeats the independent NaH model outside the checkout:
energy error 6.67e-16 Eh and maximum gradient error 2.05e-10 Eh/bohr. The same
potential loaded independently from a BSE JSON file gives energy error 2.00e-15
Eh. H2/pcseg-0 agrees with PySCF within 1.91e-14 Eh. ASE's existing positional
charge/multiplicity arguments retain their meanings after the ECP extension.

The original organometallic smoke tests contained two incorrect expectations.
Independent PySCF gives 79 Cartesian STO-3G AOs for the specified ferrocene and
approximately 1.881527 D for CpLi, below the old qualitative 3 D lower bound.
The replacement CpLi check compares energy at 1e-8 Eh and all dipole components
at 1e-8 atomic units. Atomic units avoid differences in the programs' Debye
conversion constants. The 7-test organometallic rerun passes. The ferrocene
fixture independently establishes the AO count; it does not claim a converged
external ferrocene energy. Reference generators and exact geometries are retained
in `tests/reference` and `tests/data`.

The combined targeted native scope is 30 tests, with 16 optimized Python tests
and eight executable documentation blocks. The full release suite is a separate
rerun, whose completion status is kept in the summary above.

## External validation records

Detailed benchmark JSON records and measurement-only harnesses are maintained
outside the source repository. Set `HF_RS_BENCH_DIR` to that external directory
when reproducing benchmark commands. Record filenames cited below refer to
files in that directory. Rust tests and their independent reference fixtures
remain in `tests/`; normal builds and tests do not depend on the external records.

The follow-up derivative target is analytic gradients and analytic post-HF Hessians.
Historical finite-difference results below remain verification evidence. A
numerical derivative route is not certified as analytic by those checks.

## CIS, SCS-MP3 and auxiliary-basis RI-HF

The native WSL regression passes 31 integration tests plus the existing default
RI-HF compatibility test and its unit test (33 total). All targets compile.
[The validation record (`cis_scs_mp3_ri_hf_validation.json`)](#external-validation-records) contains the
commands, reference sources and limits.

PySCF's independently assembled TDA operator verifies four singlet and four
triplet roots for each of water and LiH: all 16 energies and 192 amplitude/dipole
components, including oscillator strengths. Comparisons align MO rotations and
complete degenerate excited-state subspaces. The unchanged tolerance is 1e-9.
CIS selected-root outputs now report `E_HF + omega[root]` and the correct spin
sector. Analytic excited-state nuclear response remains uncertified.

Psi4's complete SCS-MP3 calculation verifies RHF water/LiH and UHF OH/CH2
doublet/triplet energies. Every unscaled SS/OS second-order contribution and pure
third-order increment agrees within 1e-9 Eh. The CH2 same-spin contribution is
nonzero. Setting unit spin scales and third-order scale zero/one recovers MP2/MP3.

Actual Coulomb-metric auxiliary-basis RI-HF replaces the AO-pair supermatrix
factorization. PySCF verifies all 504 P/F/J/K matrix entries and every orbital
energy, fitting error and retained rank for four systems, including nonlocal ECP,
spherical D functions and duplicate OBS/auxiliary functions. No four-center ERI
tensor is built; optional exact fitting-error comparisons stream individual
integrals. The four-model test peaks at 8,384 KiB RSS with four Rayon workers,
excluding compilation. This is a small-model measurement. The metric/three-center
allocation estimate is bounded separately; it is not a process-wide memory cap.

`ruff check tests bench` and `ruff format --check tests bench` pass (44 files).
Copyright notices and numerical acceptance tolerances are retained.

## Fixed local CCSD and triples spaces

`LocalSpaceAnchor::compute_ccsd` and `LocalTriplesAnchor::compute_t0/compute_t1`
transport the selected PNO/TNO subspaces through nuclear motion. All domain,
pair and rank choices are fixed at the reference geometry. Eight native WSL
tests cover CCSD at all 20 reference/displacement/rigid-motion geometries,
including all 49,000 AO amplitudes, plus finite-TNO T0/T1 energies against
independent determinant CI moments and dense resolvents. Maximum errors are
8.54e-12 in amplitudes, 8.93e-14 Eh for CCSD and 2.68e-15 Eh for triples.
The two 20-geometry tests peak at 11,620 KiB RSS with four workers, excluding
compilation. See [local_correlation.md](local_correlation.md) and the
[validation record (`local_continuity_validation.json`)](#external-validation-records). This validates
the transported energy approximation; full analytic nuclear response remains
uncertified and the derivative capability remains `Unsupported`.

## Complete single-reference IM-SRG(2) flow

The four native tests verify all 2,666 zero/one/two-body commutator components
against independent complete Fock-space matrix algebra, including nonzero
induced bare three-body contractions. Both flowing Hamiltonians compare every
retained component (1,333 per generator) with SciPy DOP853: maximum errors are
1.27e-12 for fine fixed RK4 and 2.27e-11 for adaptive integration, below the
unchanged 1e-9 criterion. H2 molecular/typed-API comparisons, fourth-order step
refinement, exact zero flow, stiff-step recovery and failure paths also pass.
Peak native test RSS is 9,180 KiB with four workers, compiler excluded. See
[imsrg2.md](imsrg2.md) and
[`imsrg2_validation.json`](#external-validation-records). This is the
single-determinant NO2B truncation; MR driven-SRG and nuclear response are not
certified by these checks.

## Single-reference driven SRG

The SR-DSRG-PT3 and SR-LDSRG(2) rewrite passes seven targeted native tests.
The combined P3 suite passes 57 tests, with one additional existing LiH SR/MR
call-path test (58 total); all targets compile and tests/bench pass Ruff.
Independent Fock-space projection and a Pade superoperator exponential verify
all 57,929 retained BCH, generator, transformed-Hamiltonian and perturbative
amplitude coefficients. Maximum molecular operator error is 1.67e-12;
finite-lambda polynomial refinement is below 5.45e-12. Models include nonzero
singles, noncontiguous occupied indices, semilocal ECPs and rectangular MOs.
Open-shell ECP PT2 also agrees with independent PySCF UMP2 contractions with a
nonzero same-spin contribution. Zero-flow and MP2/MP3 limits pass.

All 24 Cartesian numerical-gradient components of H2 and the ECP diatomic
agree with independent five-point energy differences within 2.63e-12 Eh/bohr.
The common numerical API compares all 24 components within 4.19e-8 Eh/bohr.
SCF residual/density tolerances of 1e-13 resolve finite-difference noise;
the acceptance criteria remain 1e-6 Eh/bohr and 1e-9 translational invariance.
These are numerical derivatives, not relaxed analytic response.

The seven small-model tests peak at 13,828 KiB native RSS with four workers,
compiler excluded. This is not a measurement of the historical large ECP
workload. See [sr_dsrg.md](sr_dsrg.md) and
[the validation record (`sr_dsrg_validation.json`)](#external-validation-records). Remaining MR-DSRG
theory and response are not certified by these checks.

## Exact three-body density cumulants

The exact spin-orbital CI/ensemble backend passes five targeted native tests.
Every one of 4,792,896 RDM/cumulant components agrees with an independent
complete Fock-space and exterior-logarithm oracle, with maximum error 2.23e-16.
All FCI spin traces and particle contractions agree within 8.89e-16. Saved
CASCI roots and a weighted LiH state average agree in every active one-/two-RDM
spin trace within 1.34e-15. A pair of coherent states with identical lower RDMs
and different lambda3 verifies that genuine three-body information is used.

Packed antisymmetric tuples and in-place Gamma3-to-lambda3 conversion avoid a
full sixth-rank spin tensor. Memory budgets cover packed tensors and conservative
sparse-removal/validation estimates; borrowed CI vectors are excluded. The five
small-model tests peak at 12,820 KiB RSS with four Rayon workers, compiler
excluded. This is not a large active-space scaling measurement.

At this cumulant checkpoint, the combined regression passed 72 tests; all targets
compiled and Ruff passed 48 Python files (49 in formatter scope). See
[density_cumulants.md](density_cumulants.md) and
[the validation record (`density_cumulants_validation.json`)](#external-validation-records).
This cumulant check preceded the subsequent MR commutator, driven-energy
and multistate checks documented in the later sections below.

## Basis, SCF and integral checks

The core numerical changes provide real spherical AO subspaces, canonical overlap rank removal
at `1e-8`, self-consistent fractional atomic HF densities, and shell-quartet ERIs.
The low-level `run_scf_general`/`run_uhf` APIs retain Cartesian defaults for existing
reference comparisons; `run_scf_in_space`/`run_uhf_in_space` accept `BasisKind` explicitly.
The low-level `BasisKind::default_for_name` helper selects spherical def2,
cc-pV and 6-311G families and Cartesian 6-31G families. The common native API
uses the caller's explicit `cartesian` flag, defaulting to true; it does not
apply that name-based helper implicitly.
Cartesian AO rows are retained in MO coefficients and densities, so spherical results
have rectangular `C`. Linear dependence is detected in the physical AO overlap, rather
than inferred merely from rectangular coefficients. Checked derivative APIs return
`HfError::LinearDependence` when physical overlap eigenvectors were discarded.

Independent tests added in this phase include:

- `basis_parser_test`: every general-contraction column, Gaussian94 exponent scale
  factors, and malformed input errors.
- `scf_stability_test`: every singlet/triplet orbital Hessian component against the
  AO HF energy functional; simultaneous `C exp(kappa)` rotations and their inverse.
  Restricted stability optimization uses singlet A+B; external RHF→UHF analysis
  uses triplet A+B.
- `spherical_basis_test`: every real-harmonic transform coefficient through L=7
  against PySCF, a spherical HF energy/all gradient and Hessian components, and
  duplicate-AO rank removal with explicit derivative errors.
- `sad_test`: H/C/N/O atomic energies and every density element against PySCF's
  spherical atomic HF solver using exactly the embedded STO-3G data. Its H solver
  uses the one-electron Hamiltonian. Molecular charge and translation are also checked.
  The geometry-independent SAD cache has a 32 MiB accounting budget and 128-entry limit.
- `scf_special_test`: MOM and thermal density/Fock/occupation consistency, odd thermal
  electron populations, invalid input and iteration-limit flags, and QRHF reference failure.
- `fock_contraction_test`: every RHF/UHF Fock element from unique quartets against
  independently ordered four-index contractions, including every index coincidence.
- `shell_eri_test`: every block component for off-center contracted shells including
  L=7 against individual primitive integrals, and every packed AO quartet in a mixed
  s/p/d molecule against the independent AO-pair path.
- `df_test`: closed-form s-function Coulomb metrics and every two-/three-center
  integral plus RI-MP2 against PySCF with the same embedded auxiliary basis.

Reference generation uses the existing WSL Ubuntu PySCF 2.13.1 environment, with
no package downloads. The scripts under `tests/reference/` record the embedded
basis source hash. PySCF Cartesian normalization and d-component order are converted
explicitly before comparison. `gen_spherical.py`, `gen_sad.py` and `gen_df.py` regenerate
the corresponding JSON files in `tests/data/`.

`molden_roundtrip_test` writes nonzero coefficients for every s/p/d/f/g component in
both Cartesian and spherical representations. `tests/reference/check_molden.py` reads
these files using PySCF's Molden reader and evaluates the orbital values with its AO
engine. Maximum differences are `3.33e-16` (Cartesian) and `2.22e-16` (spherical).
Contractions are normalized from the actual working basis before export. Spherical
output uses `[5D7F]` and `[9G]`; Cartesian f/g components follow the
[Molden format specification](https://www.theochem.ru.nl/molden/molden_format.html).
The checked exporter rejects L>4 because the format has no standard for those shells.
ECP core-electron counts are preserved in `[core]`; the visualization format does not
encode the potential itself.

The optimized P1 regression passed 63 tests across 18 binaries. This includes every
component through fourth order in three additional spherical model systems (4,662
comparisons), bringing the all-component total to 15,150. The additional order-4 maximum
errors are 6.74e-9 (spherical RHF), 7.27e-9 (spherical ECP RHF), and 5.91e-10
(spherical ECP UHF). The unchanged order-4 acceptance limit is 3e-6.
See [`core_validation.json`](#external-validation-records) for raw-log and binary
hashes. This P1 snapshot predates the subsequent post-HF changes.

In the earlier core-numerics benchmark, runtime was slower than v0.2.0: water Cartesian
cc-pVDZ SCF 0.098 s (baseline 0.026), CCSD 3.464 s (1.489), and benzene 6-31G* SCF
4.181 s (1.815). Water 6-31G* gradient/Hessian are 0.071/12.351 s (0.017/4.102).
The whole process peaked at 304.86 MiB private memory; MP2 was 0.002 s in both runs.
The corrected integrals change the energies, so the timing record is not a comparison
at identical numerical error. These are historical timings, not measurements
of the current source or a current equivalent-model speedup; see
[`core_benchmark_v0.3-p1.json`](#external-validation-records).

P2's `context_options_test` passes seven tests covering active-space spin/selection,
strict option schemas, lazy storage sharing, every mixed-spin integral against
direct AO contraction, method-name round trips and selected-state energies.
`typed_post_hf_test` contains independent PySCF frozen-core MP2 full-density checks
and shared-context CC tests. The initial Windows execution was blocked by
code-signing policy. These tests subsequently ran successfully in WSL; the
Windows restriction is historical and is not an outstanding numerical test.
Current release verification also uses WSL, without changing signing policy.

## ECP workspace

The scalar, semi-local ECP operator is integrated with composite radial Gauss-Legendre
quadrature and a product angular grid. The memory optimization preserves the quadrature
parameters. High-order derivatives additionally use polynomial-aware primitive screening
to retain significant high-angular-momentum terms. Angular points are streamed in blocks
of at most 1,024 points, with an approximately 1 MiB point-dependent workspace budget per
radial worker. AO matrix outputs and accumulated harmonic projections require an additional
`O(n_AO^2 + n_AO L^2)` storage per worker. For nonlocal channels the projections are summed
over the entire sphere before multiplying them, preserving contributions between blocks.
Matrix products accumulate directly through views in small SIMD tiles. This avoids both
transposed AO copies and the large persistent per-worker packing buffers used by general
matrix multiplication.

RHF and UHF two-electron gradients retain one screened quartet list per worker and reuse
it across the derivative terms of each AO pair. This replaces the `O(n_AO^4)` candidate cache
with `O(threads * n_AO^2)` candidate storage. The existing Schwarz bounds and density
screening thresholds are preserved. Primitive-pair caches and the compressed SCF ERI
tensor remain separate memory costs.

The low-level Rust API takes atoms with ECPs already attached; the Python/CLI named-basis
resolver attaches the matching def2 ECPs automatically. For the built-in
def2 potentials (Rb–Rn), attach them before constructing and running the calculation:

```rust
use hf_rs::{assign_builtin_ecps, build_basis, get_builtin_basis, run_scf_general, Atom, ConvAlgo};

let mut atoms = vec![Atom::new("I", [0.0; 3]), Atom::new("H", [0.0, 0.0, 3.0])];
assign_builtin_ecps(&mut atoms, "def2-svp");
let basis = build_basis(&atoms, &get_builtin_basis("def2-svp").unwrap());
let hf = run_scf_general(&atoms, &basis, 0, 300, 1e-12, ConvAlgo::Cdiis, None).unwrap();
// HI has 26 explicit electrons after replacing iodine's 28 core electrons.
assert_eq!(hf.n_occ, 13);
```

Coordinates are in bohr. External Gaussian 94 ECP definitions can be read with
`hf_rs::ecp::parse_ecp_gaussian94` and attached with `assign_ecps`; they require a matching
valence orbital basis. This implementation evaluates scalar semi-local ECPs.

`tests/ecp_memory_test.rs` measures live Rust heap allocations with four radial workers.
It uses a dense angular grid and checks all matrix elements against closed-form local
Gaussian and s-projector integrals. The peak allocation above the initial live allocation
must stay below 32 MiB. This measures heap workspace, not total process memory or thread
stacks.

```powershell
cargo test --release --test ecp_memory_test -- --nocapture
cargo test --release --test ecp_test -- --nocapture
cargo test --release --test scf_reference_test --test basis_data_test --test boys_test
```

`measure_memory.ps1` additionally measures process private memory and peak working
set on Windows. It starts only the specified executable, excluding the Cargo compiler
from the measurement. The optional memory and time limits terminate only that process
and mark the run as incomplete.

```powershell
$binary = (Get-ChildItem target/release/deps/ecp_test-*.exe | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
& (Join-Path $env:HF_RS_BENCH_DIR 'measure_memory.ps1') -Binary $binary `
    -BinaryArgs @('quadrature_is_converged_and_matrices_are_symmetric', '--exact', '--nocapture', '--test-threads=1') `
    -RecordPath target/ecp_memory.json -Threads 16
```

The ECP oracle is `tests/data/ecp_reference.json`, generated by
`tests/reference/gen_ecp.py`: PySCF 2.14.0, Cartesian AOs, BSE 0.12 orbital basis and ECP
data. It covers HI, AgH, I2, AuH, PbH4, open-shell AgH+, and HI/def2-TZVP. The tests compare
generalized ECP matrix eigenvalues (AO ordering independent), SCF energies and nuclear
gradients. Five-point energy differences and rigid-motion invariance provide additional
checks independent of the oracle.

### Measured memory

The before/after snapshots below precede the analytical high-order derivative additions.
Both process measurements use the release build, Cartesian HI/def2-SVP AOs, the default
and refined quadrature convergence test, and 16 Rayon workers. The thread count and
quadrature accuracy were not reduced.

| Implementation | Peak private memory (MiB) | Peak working set (MiB) | Result |
|---|---:|---:|---|
| Inherited ECP implementation | >2,082.70 | >1,070.48 | Stopped at the 2 GiB private-memory guard |
| Streamed quadrature and tiled products | 16.12 | 19.24 | Passed |

The incomplete baseline establishes a lower bound on its peak, not its final peak.
The private-memory reduction for this test is at least 99.2%. The allocation-counter
regression measured 1.379 MiB of additional live Rust heap with four workers and passed
all closed-form matrix-element checks. Total process memory and additional heap workspace
are different measurements. Process sampling uses 20 ms intervals.

The original run was terminated, so these measurements do not establish a wall-time speedup.
Machine-readable measurements and binary hashes are recorded in
[`ecp_memory_optimization.json`](#external-validation-records).

A subsequent optimized test-profile run, including polynomial-aware screening and
the corrected Jacobi solver, completed **all 13 ECP tests in one process** with 16 Rayon
workers: peak private memory **250.79 MiB**, peak working set **116.86 MiB**. The runtime
was 1,575.509 s while compilation and other tests overlapped; it is not a standalone
speed benchmark. The record includes the executable hash. All seven PySCF cases
compare every gradient component, as well as energy and the ECP matrix spectrum.

## Other verified core numerics

- Boys functions: worst relative error `2.267e-15` against the 50-digit mpmath oracle.
- RHF/UHF SCF and analytical gradients: all 10 Cartesian PySCF cases pass. The maximum
  gradient discrepancy is `5.96e-10 Eh/bohr` (acceptance limit `1e-6`).
- RHF analytical Hessian: all 8 Cartesian PySCF cases pass. The maximum discrepancy is
  `6.40e-8 Eh/bohr^2` (acceptance limit `1e-5`). This validates the additional overlap-response
  factor in the inherited Hessian change. ECP Hessians are outside this check.
- Embedded orbital/auxiliary basis data: all 5 basis-data tests pass, including the
  independent BSE JSON manifest and orbital normalization checks.

The SCF gradient/Hessian oracle checks were run with `--profile dev`; Boys, basis data
and ECP allocation checks were run with `--release`. These targeted checks do not replace
the full v0.3.0 release checks in the implementation plan.

## Analytical nuclear derivatives through fourth order

The implementation and checked API are described in
[analytic_hf_derivatives.md](analytic_hf_derivatives.md). Production HF third/fourth
derivatives no longer displace the geometry. ECP-center derivatives include the local
potential and nonlocal angular projectors.

`tests/analytic_hf_derivatives_test.rs` checks:

- First/second-order density response against the independently implemented RHF gradient/Hessian.
- RHF mixed/repeated third and fourth derivatives against five-point differences of
  independently verified analytical Hessians (`h=5e-3` bohr; limits `2e-7` and `2e-6`).
- ECP-center derivatives of orders 1..4 against the closed-form three-center Gaussian
  integral, with absolute error below `2e-10`.
- RHF and UHF fourth derivatives with local, s- and p-projector ECP channels against
  five-point second differences of separately verified Hessians (limit `3e-6`).
  Representative results: RHF `1.0565146384` vs `1.0565146351`; one-electron UHF
  `3.3204476185` vs `3.3204476171`; three-electron UHF with both spin channels
  `3.3446819988` vs `3.3446821381` (`Eh/bohr^4`).
- Directly recomputed mixed-variable permutations, without using tensor symmetry to
  establish agreement; built-in iodine/def2 ECP fourth-integral quadrature convergence;
  directional/block/mode API contractions; high-L Hermite overflow and tight eigenpair residuals.

The small-matrix Jacobi diagonalizer previously compared a squared residual to an
unsquared threshold. Its resulting approximately `1e-8` eigenvector residual floor
prevented tight open-shell SCF convergence. The corrected, scale-aware criterion is
covered by an eigenpair residual check at `5e-14`.

`tests/test_python_hf_derivatives.py` covers native package import, directional/full
tensor agreement, UHF Hessian/third/fourth derivatives, multilinear direction scaling,
angstrom conversion, and error propagation. Run with a built extension on `PYTHONPATH`:

```powershell
python tests/test_python_hf_derivatives.py -v
cargo test --test analytic_hf_derivatives_test --test fourth_deriv_test --test hessian_third_deriv_test
cargo test --test hf_all_components_test -- --nocapture --test-threads=1
```

These ECP fourth-order HF checks use small Gaussian model systems. Real def2 molecules
are covered separately by PySCF energy/gradient/matrix-spectrum comparisons; this is
not an external PySCF fourth-order molecular oracle.

`tests/hf_all_components_test.rs` additionally compares every Cartesian component of
orders 1..4 in off-axis all-electron RHF H2, model-ECP RHF H2, and model-ECP open-shell
UHF H3 (two alpha/one beta electrons). The respective tensor dimensions are six, six,
and nine: 1,554 + 1,554 + 7,380 = 10,488 component comparisons. The base third/fourth
tensors come from the public checked tensor APIs. Five-point first differences
differentiate energy → gradient → Hessian → third → fourth. The RHF reference step
is `5e-3` bohr; the small-gap open-shell reference uses `1.25e-3` bohr, with `1e-15`
SCF energy tolerance for these small models.
Gradient references use the independent RHF/UHF gradient code; all-electron RHF third
references use the independently verified specialized Hessian. ECP lower derivatives
are validated in sequence before serving as higher-order references. Difference
results are neither symmetrized nor corrected by translation sum rules. Acceptance
limits for orders 1..4 are `1e-6`, `1e-5`, `2e-7`, and `3e-6` respectively.

All 10,488 comparisons pass. Maximum absolute errors for each complete tensor:

| System | Components (orders 1/2/3/4) | Order 1 | Order 2 | Order 3 | Order 4 |
|---|---|---:|---:|---:|---:|
| All-electron RHF H2/STO-3G | 6/36/216/1296 | 1.75e-10 | 5.08e-10 | 2.44e-9 | 1.87e-8 |
| Model-ECP RHF H2 | 6/36/216/1296 | 3.02e-10 | 6.55e-10 | 2.74e-9 | 1.49e-8 |
| Model-ECP UHF H3 doublet | 9/81/729/6561 | 2.07e-11 | 3.59e-7 | 7.62e-9 | 1.59e-7 |

The optimized test-profile run passed all four tests, including the step-convergence
regression, with 16 Rayon workers: 464.888 s, peak private memory 26.24 MiB, peak working
set 30.23 MiB. These small-model memory results are distinct from the real-def2 ECP
suite above. Source/binary hashes, worst-component indices, tolerances and reproduction
settings are saved in [`hf_derivative_validation.json`](#external-validation-records).

All-component testing exposed a finite-difference truncation error in the initial
open-shell reference. Tightening SCF from `1e-13` to `1e-16` left the discrepancy
unchanged. Halving the five-point step reduced the third/fourth reference errors
by approximately 16, as expected for fourth-order truncation bias:

| Step (bohr) | Third error, component [5,5,5] | Fourth error, component [2,5,5,5] |
|---|---:|---:|
| 0.005 | 1.94618e-6 | 4.05454e-5 |
| 0.0025 | 1.21841e-7 | 2.54096e-6 |
| 0.00125 | 7.62015e-9 | 1.58922e-7 |

The step-convergence regression checks this reduction explicitly; acceptance limits
are unchanged. The finer reference resolves this open-shell response without
substituting finite differences for the analytical production derivative.

The H2/water Hessian integration tests use five-point differences of independently
reconverged analytical gradients at `h=5e-3`, with a `1e-5` maximum-component tolerance.
The old two-point `1e-4` reference amplified residual SCF noise; it is not used as the
acceptance oracle for these tests.

At this earlier checkpoint, Windows application-control policy prevented the Rust
CLI runtime smoke test. Later installed CLI checks passed under WSL, including
execution outside the checkout; see the current status above. The five Python
native integration tests at this checkpoint passed, including automatic def2 ECP
assignment for HI against the PySCF oracle.


## FCI and CASSCF core

The combined P3a targeted regression passes 18 tests: FCI 5, CASSCF 4, excitation
spaces 2 and shared context/options 7. The string-driven FCI engine compares every
sigma, S² action, 1/2/3-RDM and directly F-contracted 4-RDM component against PySCF
2.13.1 in closed-, open- and parallel-spin sectors. Dense and Davidson first-three-root
energies and lowest S²=0/2/6 states pass. Spin-target reference energies come from
exact PySCF S² subspaces. Davidson projects guesses and preconditioned corrections
onto that sector, avoiding the ill-conditioned large-penalty iterative solve.

CASSCF stores active RDMs and uses the shared inactive-core contraction. Every LiH
orbital-gradient and diagonal-curvature component agrees with independent frozen-CI
energy differences at 1e-8/2e-7. Full-active and HF limits pass at 1e-10; state-average
and selected-root energies, open-shell/explicit selections and the overlap metric
also pass. LiH CAS(2,2) agrees with PySCF in energy and every AO-density element at
the unchanged 1e-9/1e-7 limits. The hf-rs orbital maximum is 3.12e-11 in the tight run.

The initial PySCF density differed by 3e-7 because a small orbital residual persisted
in a soft mode. Newton continuation and fresh CASCI stabilize it across independent
continuations to better than 1e-8. `casscf_reference.json` records the measured final
orbital residual and the stricter refinement convergence flags; those flags can remain
false at requested 1e-11/1e-10 even when the measured residual is below 1e-7. The
original PySCF solve reports convergence. No hf-rs acceptance tolerance was relaxed.

Earlier Windows test snapshots were blocked by code integrity. Subsequent
WSL builds/runs, including `typed_post_hf_test`, completed successfully without
security or signing changes. The following is the original targeted CI/CASSCF
command; use the current full release command for the final regression:

```powershell
$env:CARGO_PROFILE_TEST_OPT_LEVEL='0'
$env:RAYON_NUM_THREADS='4'
cargo test --offline --test fci_engine_test --test casscf_engine_test --test excitation_space_test --test context_options_test
```

Source/binary/log hashes and snapshot-specific settings are recorded in
[`fci_casscf_validation.json`](#external-validation-records). These targeted
checks do not replace the final full release suite. Shared SD generation is independently
checked against complete small spaces, open-shell selections, memory limits and orbital
index 63. The subsequent MRCISD and selected/projector-CI checks are recorded
in their dedicated sections below.


## Native Linux verification and MRCISD(+Q)

The user requests WSL verification and no subagents. Rust 1.97.1 builds this source
as native Linux code under WSL Ubuntu; Windows binaries are not used. Linux artifacts
are isolated under `target/linux`. The locked Cargo dependencies missing from the
existing cache were fetched, with no oracle-package installation. Serial Cargo builds
avoid competing with other WSL calculations. No other processes or security settings
were changed.

`typed_post_hf_test` now executes on Linux and both tests pass: every frozen/nonfrozen
MP2 AO-density component at 1e-8, energies/SCS components, shared UHF/RHF integrals,
CCSD/FCI equivalence and amplitude nonconvergence. The initial density discrepancy
came from comparing PySCF's grouped s/p AO sequence to Gaussian94's interleaved SP
sequence. The oracle now records the explicit permutation and PySCF AO labels;
the production MP2 density expression and acceptance tolerances were unchanged.

The P3e combined Linux regression passes 10 tests. The common CI subspace operator
has independent all-component sigma, S² and 1/2-RDM checks and Slater–Condon row checks.
MRCISD generates SD substitutions and completes spin couplings within each spatial
occupation pattern, so its spin-targeted space commutes with S². Spin completion is
checked separately against closure under the previously verified full-FCI S² operator.
The iterative solver stores CI vectors rather than a determinant-pair matrix or a full
FCI vector. Two-RDM materialization is explicit and limited to 256 MiB.

Independent PySCF references build the entire small FCI Hamiltonian, select spatial
occupation classes by brute force and diagonalize its exact target-spin subspace.
Checks cover HF/CASSCF orbitals, frozen core and open-shell common spatial orbitals,
every AO 1/2-RDM component (1e-7), root selection, reference weights and +Q energies.
CAS(0,0)→CISD and full-active→FCI limits pass at 1e-10. Open-shell CI uses UHF alpha
orbitals as one common spatial basis; its SCF-relative correlation includes the
reference change. Gradients and an unrestricted spin-dependent CI representation
are separate from these energy checks.

The Davidson correction implemented here is
`Delta_E_Q = (1 - sum_reference |c_I|^2) (E_MRCISD - E_ref)`.
The `c_I` are the relaxed MRCISD eigenvector coefficients. `E_ref` is the corresponding
CASCI/CASSCF root, or the common HF energy when there is only one reference determinant.
The reported +Q RDM remains the variational MRCISD density because +Q specifies an energy
correction, not a corrected wavefunction. Correlation energies are always relative to
SCF, with the active-space energy in a separate component.

Commands, hashes and limits are in
[`mrci_validation.json`](#external-validation-records). They establish the stated
MRCISD/+Q energy and density scope, not complete relaxed nuclear response or
large-system performance.


## Selected/projector CI (P3h)

The native WSL energy/density regression passes 30 tests in six binaries, plus
one filtered legacy compatibility test and a pure-diagonal Davidson regression
(32 total). `selected_ci_validation.json`
records commands, hashes and fixed tolerances. Independent PySCF Hamiltonians
with dense NumPy SCI/PCI calculations provide finite-threshold energies,
corrections and all 1/2-RDM entries. Limits agree with FCI to 1e-10 Eh; every RDM
entry agrees within 1e-7. PCI's finite-threshold test checks the propagated
wavefunction, not a final rediagonalized model eigenstate. Ten historical PCI
generator choices pass the zero-threshold FCI limit.

Molecular checks use common AO densities when explicit orbital selection
permutes the CASCI MOs. CASCI/SCI spin sectors are explicitly matched when
comparing excited roots. Nonconvergence, determinant limits and singular EN
corrections return errors; no denominator floor or zero-result fallback is used.
The legacy API uses the same new engines. Inner CI solves have an independent
`ci_max_iter`, so short outer selection limits do not truncate their Davidson
iterations. See [selected_ci.md](selected_ci.md) for algorithm/estimator details.
Complete analytic post-HF response remains follow-up work. Current release
checks and the limits of the historical performance measurements are summarized
at the top of this document.

## F12 integral/CABS checkpoint and Python lint

The three native WSL F12-integral/CABS tests pass. Gaussian-geminal operators
are checked against independent polynomial/radial quadrature through L=7, and
all CABS/Fock/singles components are compared with PySCF, including a nonzero
local ECP, spherical 8-AO/7-MO and overlap-rank-drop 3-AO/2-MO references. In the
redundant OBS, projector/amplitude comparisons use all physical overlap-metric
components; raw AO coefficients have null-space freedom. Tolerances are unchanged.
See [f12_integrals.md](f12_integrals.md) and
[`f12_building_blocks_validation.json`](#external-validation-records).
These initial checks certify building blocks. Subsequent full MP2-F12 and
coupled CCSD-F12 model checks are described in [F12](f12_integrals.md); the
published/default printed-digit discrepancy remains open.

Ruff 0.17.1 checks all 30 Python test/reference files with E4/E7/E9/F and import
ordering enabled. The original 333 findings are resolved without suppressing
rules. Both `ruff check tests` and `ruff format --check tests` pass. All files
parse with Python 3.8 grammar and compile in the existing WSL Python environment.
Existing copyright notices are retained. An AST comparison of 28 tracked scripts
accounts for the reviewed import, angular-momentum name and lambda-to-function
cleanups; no numerical formula or acceptance tolerance was changed.

Nine independent generators were rerun in `target/python-lint-oracle-check`,
leaving the committed JSON data intact. FCI, selected CI, local MP2 model,
triples-space model, DF, spherical, molecular triples, CABS and F12-integral
outputs agree in every numeric component; the largest difference is 3.02e-12.
The lint-only environment needs no scientific packages. Reproducible commands
and the pinned dependency are in [tests/README.md](../tests/README.md) and
[`python_lint_validation.json`](#external-validation-records).


## Exact GNO and typed unrelaxed MR driven equations

The typed MRPT2/MRPT3/MR-LDSRG(2) adapters use exact CI densities, complete
rank-two GNO commutators and semicanonical CASCI/CASSCF references. Independent
full Fock-space basis solves, Pade superoperator exponentials, nonlinear roots
and refined finite-coupling fits check every retained operator component.
Molecular checks include rectangular MO spaces and semilocal ECPs. Every one
of 45 nuclear-gradient components is compared against independent five-point
energy differences; the numerical API is checked separately on all 45.
See [MR driven methods](mr_driven.md) and
[the validation record (`mr_driven_validation.json`)](#external-validation-records) for scope and evidence.
Reference relaxation, QC/THREE/state variants, compatibility binding migration
and large-system factorization remain outside this checkpoint.


## MR effective-CAS reference relaxation

The typed MRPT2/MRPT3/MR-LDSRG(2) adapters now support once, twice and iterated
singlet reference updates. Exact Givens CI transport preserves the saved state
under semicanonical block rotations. A general spin-orbital CI action retains
non-Coulomb effective two-body interactions and shares dense/Davidson spin
selection with the physical CAS solver.

Independent full-Fock projections check every one of 675 general-CI matrix
entries, 273 core-contracted coefficients, 84 CI-rotation coefficients and 648
spin-root projector entries. Four molecular models and three methods/modes give
36 relaxed energies (maximum error 1.64e-12 Eh), all 144 final CI coefficients
(maximum 1.63e-10) and all 2,184 PT effective-CAS coefficients (maximum 9.09e-10).
Models include optimized CASSCF, rectangular 6-AO/3-MO LiH and local/nonlocal ECPs.

All 135 nuclear-gradient components (nine H3+ coordinates and six ECP-diatomic
coordinates, for every method/mode) agree with independent five-point differences
at h=1e-3 within 1.20e-9 Eh/bohr. The numerical API at h=5e-4 agrees on all 135
within 4.27e-8; the largest translational sum is 8.72e-12. Acceptance remains
1e-6 for gradient differences and 1e-9 for translational invariance. The existing
45 unrelaxed components also pass. Reference/SCF/amplitude reconvergence is
included; these are numerical derivatives, without an analytic-response claim.

The combined native WSL regression passes 92 tests in 20 binaries, all targets
compile, and Ruff passes all 55 Python files (56 formatter files). The twelve
small-system CI/MR tests peak at 11,328 KiB native RSS, four Rayon workers and
one test thread, excluding compilation and the Python full-Fock oracle. This
is not a measurement of the historical 8 GB ECP workload. See
[MR driven methods](mr_driven.md) and
[the relaxation validation record (`mr_reference_relaxation_validation.json`)](#external-validation-records)
for algorithms, controls, limits, exact commands and memory logs.


## MR single-state compatibility migration

`DSRGMRPT2`, `DSRGMRPT3` and `MRLDSRG2::new` now share the verified active-CI,
exact-cumulant and reference-relaxation engine. This removes full-orbital
spin-string enumeration from these paths and handles rectangular MO coefficients.
All 36 unrelaxed/relaxed compatibility energies agree with the independent
molecular oracle within 1.64e-12 Eh. All 486 physical AO reference-density
components agree within 2.76e-11. Standard outputs are relative to SCF, preserve
the CAS component and do not present a reference density as a correlated RDM.
Unimplemented switches, invalid controls and unconverged output return errors.

The reconstructed-Cholesky wrapper uses the same MRPT2 base; nine checks with
rectangular MOs, nonlocal ECPs and different reference-update modes agree with
the same four-index Hamiltonian within 3.56e-15 Eh. AO factorization dimensions,
invalid-cutoff errors, pre-allocation workspace estimates and ERI-clone avoidance
are checked. This is a compatibility consistency check, not certification of a
factorized THREE-DSRG algorithm or its asymptotic scaling.

The combined native WSL regression, selected legacy DSRG route, all existing
135 relaxed/45 unrelaxed gradient and API components, all-target compilation
and small-model RSS are recorded in
[the compatibility validation record (`mr_compatibility_validation.json`)](#external-validation-records).
The [method documentation](mr_driven.md#compatibility-constructors) describes
option mapping, density-frame semantics and remaining scope limits.

This checkpoint passes 96 tests in 21 native binaries. Peak RSS is 15,188 KiB
across the ten-test typed MR suite and the selected legacy DSRG suite (including
six-MO LiH), with four Rayon workers, one test thread and compilation/Python
oracle excluded. This small-model measurement does not establish the memory
usage of the historical 8 GB ECP workload. At this checkpoint Ruff passed 55
Python sources/56 formatter files; current lint counts are in the summary above.


## Quadratic-BCH MR variant

The documented two-nested-commutator QC convention now uses the complete GNO
kernel, converged single/double Gaussian source equations and shared reference
relaxation. The typed registry and `MRLDSRG2::new_qc` use the same implementation.
The isolated approximate QC commutator and determinant enumeration are removed.

Independent full-Fock operator maps and nonlinear roots verify all 13,330
generator/transformed coefficients within 2.07e-14. The source requires up to
11 amplitude iterations, despite the fixed BCH depth two. Its energy differs
from complete recursive BCH by up to 1.36e-5 Eh in these models, confirming that
the QC calculation is a distinct transformation. Four molecular references and
four relaxation modes give 16 energies within 1.25e-13 Eh and all 64 final CI
coefficients within 6.91e-10. Twelve compatibility energies agree within
2.76e-14 Eh. Models include optimized CASSCF, rectangular MOs and nonlocal ECPs.

The existing numerical/debug route is checked on all 45 new QC Cartesian
components (nine H3+ and six ECP coordinates, each in once/twice/iterate modes).
Five-point h=1e-3 differences agree with the independent oracle within 4.67e-12
Eh/bohr, and the numerical API at h=5e-4 within 4.21e-8. Translation sums are
below 7.35e-12. Acceptance remains 1e-6 and 1e-9, respectively. The previous
180 MR components also pass. These checks validate the energy surface; analytic
post-HF response is a next-version P4 deliverable.

The combined native WSL regression passes 99 tests in 21 binaries, all targets
compile, and Ruff passes 57 Python files (58 formatter files). Peak RSS across
the twelve-test MR suite and selected legacy DSRG suite is 14,824 KiB, compiler
and Python full-Fock oracle excluded. No large-ECP scaling result is inferred.
The detailed record is external `mr_qc_validation.json`; see
[external validation records](#external-validation-records) and the
[QC convention](mr_driven.md#quadratic-bch-variant-qc).

## DMRG MPS/MPO and orbital optimization

The determinant-matrix compatibility solver is replaced by the shared
[charge-conserving MPS/MPO implementation](dmrg.md). Initialization, two-site
Davidson sweeps, sector SVD, final fixed-bond one-site optimization, densities
and DMRGSCF use no full CI wavefunction. DMRGSCF includes the finite-M
nonredundant active-active orbital rotations; a deliberately rotated M=1
two-orbital reference returns to the stationary RHF limit.

The executable block2 0.5.4 oracle uses SZ symmetry, the same M and limits of
40 two-site plus 500 one-site directional passes, no noise, no orbital
reordering, pure SVD and squared Davidson residual threshold 1e-24. Independent
PySCF FCI references cover open/closed synthetic models, a six-site Hubbard
model and molecular H4. Nine M/model combinations agree in returned-state
energy within 1.06e-10 Eh, against the fixed 1e-6 criterion. Both energies are
fresh MPS contractions, not the minimum pre-truncation two-site eigenvalue.

All 4,204 1/2-RDM components are compared. Finite-M density differences reach
1.89e-5 and are checked at 3e-5; the independent large-M FCI density criterion
remains 1e-8. Large-M energies retain the 1e-10 limit criterion. Every model
also satisfies particle number, all 2-to-1-RDM contractions and reconstruction
of its MPS energy within 1e-10. Degenerate Schmidt weights use the documented
deterministic spin-balance rule rather than roundoff-dependent sector choices.

Molecular CASCI/DMRGSCF energies agree with PySCF within 2.63e-14 Eh, including
rectangular orbitals and a nonlocal ECP. Open-shell context occupations and
every active density component are checked against the independent FCI engine.
All 30 nuclear method/coordinate components (H3+ and ECP HeH+, DMRG-CI and
DMRGSCF) agree with independently reconverged PySCF five-point h=1e-3 energy
stencils within 3.37e-12 Eh/bohr. The numerical API's central differences agree
within 1.71e-7 and translation sums are below 3.93e-12. The fixed criteria
remain 1e-6 and 1e-9. These are energy-surface verification checks, not
certification of complete analytic tensor/orbital response.

The combined native WSL regression passes 120 distinct tests in 26 binaries,
including the previous 225 MR derivative components and the independent F12
coupling/energy tests. All targets compile. Ruff checks all 60 Python test
files; its formatter checks 61 test files, including the notebook. Python
test sources also parse under the Python 3.8 grammar.

A 20-site, (10 alpha,10 beta), M=4 Hubbard test runs without enumerating its
34,134,779,536 determinants. The returned MPS tensors occupy 9,152 bytes, and
the native test process peaks at 10,536 KiB RSS, including oversized-M and
MPO-budget error checks. The complete five-test molecular DMRG suite peaks at
13,368 KiB RSS. These measurements exclude compilation and Python/block2;
they do not establish memory use for the historical large-ECP case.

To regenerate the executable references in the approved isolated environment:

```sh
"${VIRTUAL_ENV}/bin/python" tests/reference/gen_dmrg.py \
  --block2-python "$PWD/target/block2-oracle-py311/bin/python"
"${VIRTUAL_ENV}/bin/python" tests/reference/gen_dmrg_gradients.py
cargo test --offline --test dmrg_engine_test --test dmrg_oracle_test \
  --test dmrg_molecular_test
target/python-lint-venv/bin/ruff check tests
target/python-lint-venv/bin/ruff format --check tests
```

Exact native commands, per-binary results, diagnostic comparisons and memory
conditions are in external `dmrg_validation.json`; see
[external validation records](#external-validation-records).


## Internally contracted CASPT2 checkpoint

The shared single-state CASPT2 backend constructs spin-free A-H perturbers,
removes metric dependencies, applies the complete generalized Fock operator
and solves the coupled equations with residual-certified PCG. The exact
Fock-contracted four-particle density is implicit in the sparse wavefunction
action; an eight-index RDM and a full-orbital FCI Hamiltonian are not stored.
See [CASPT2 equations and explicit shift conventions](caspt2.md).

Independent PySCF full-sector matrices and NumPy direct solves verify all
eight classes over four active-electron models and 24 shift combinations.
The maximum energy error is 2.09e-15 Eh. All 4,374 first-order coefficients
agree within 5.15e-13; class energies, metric eigenvalues and shifted spectra
are also checked. The four-active-electron model exercises a nonzero 4-RDM.
MP2, frozen-core/virtual and full-CAS limits, exact-zero imaginary-shift
continuity and early allocation guards are tested.

Eighteen native OpenMolcas 25.02-2 executable energies cover H3+, LiH and
BeH2, IPEA 0/0.25, real 0.17 and imaginary 0.12 shifts, including combined
IPEA/regularization. All pass the fixed 1e-6 Eh acceptance criterion. The
maximum difference is 3.46e-7 Eh for BeH2 with IPEA; the unshifted and
real/imaginary-only results agree within 5.21e-11. IPEA depends on the IC
basis and metric convention. The oracle explicitly uses CORT for nonzero
IPEA; comparisons do not assert equality across different IPEA conventions.
Inputs, printed energies and output hashes are in the test fixture.

Every nuclear component is checked for H3+ CASCI, H3+ CASSCF and HeH+ with
a nonlocal ECP, both without shifts and with IPEA/imaginary shifts: 48
components total. Independently reconverged NumPy/PySCF five-point h=1e-3
energy stencils agree within 1.38e-10 Eh/bohr. The numerical API agrees
within 1.68e-7 and translation sums are below 1.61e-10, retaining the fixed
1e-6 and 1e-9 criteria. These are numerical energy-surface checks. Complete
analytic CASPT2 response and Hessians remain next-version P4 work.

The legacy and typed routes share the same solver. SCF-relative correlation
includes the CAS reference correction; the legacy gradient dispatcher uses
reconverged energies. A reference CAS density is not exposed as a relaxed
CASPT2 density. Existing analytic HF/ECP derivatives through fourth order
are unchanged.

Reproduce the bounded checks with native Linux/WSL:

```sh
cargo test --offline --test caspt2_engine_test --test caspt2_molecular_test
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_caspt2.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_caspt2_molecular.py
target/python-lint-venv/bin/ruff check tests
target/python-lint-venv/bin/ruff format --check tests
```

The rectangular three-MO case also matches independent NumPy energies. All
nine components of the legacy gradient dispatcher agree with the shared
numerical API within 6.70e-8 Eh/bohr. The four-test native engine suite peaks
at 10,236 KiB RSS and the four-test molecular suite at 20,372 KiB, with
RAYON_NUM_THREADS=4 and sequential test execution. Compilation and Python /
OpenMolcas processes are excluded. These bounded measurements do not
characterize the historical 8 GB ECP workload.

The combined native regression passes 128 distinct integration tests in 28
binaries. All targets compile. Ruff checks all 62 Python test sources, its
formatter checks 63 files, and the sources parse with the Python 3.8 grammar.
Detailed commands, output hashes, measurements and the continuation plan
remain outside the repository.


## Real noncollinear GHF and analytic gradient checkpoint

The real GHF solver now verifies energy, orthonormal-space commutator and
density convergence, retains the overlap rank explicitly and returns consistent
physical Fock, occupied projector and canonical orbital blocks. Tilted guesses
preserve orthogonality; an unstable closed-shell solution can seed a lower
spin-broken stationary branch using AO exchange actions. No full MO ERI tensor
is allocated. See [GHF equations and API limits](ghf.md).

Independent PySCF energies for H2+, LiH, noncollinear H3, nonlocal-ECP HeH+
and twisted ethylene agree within 3.27e-13 Eh. All 2,936 Fock/density entries
are compared, after a global spin-frame alignment for the density. Arbitrary
spin-coherent projectors test the Fock contractions directly. The maximum
Fock and density errors are 9.71e-14 and 3.33e-9, respectively. The split-SP
6-31G matrices use an explicit G94/PySCF AO permutation. The ethylene
comparison follows the same broken-symmetry Ms=0 stationary branch; other
HF stationary roots have different energies.

The GHF nuclear gradient is analytic. All 45 Cartesian components, including
nonlocal ECP and the spin-broken ethylene branch, match independently
reconverged five-point h=1e-3 energies within 6.97e-10 Eh/bohr. Translation
sums are below 8.63e-14. A shared screened ERI contraction evaluates Coulomb
and all four spin exchange blocks in one pass, without a fourth-rank density
or a second unscreened derivative pass. Rank-truncated gradients are rejected
pending projector response. The HF-variant Hessian remains explicitly numerical.

Final native verification passes 159 integration tests in 32 binaries plus
three filtered unit tests. This includes all 11 existing gradient tests, all
13 existing ECP tests and the three GHF compatibility cases. All targets
compile. Ruff checks 63 Python sources / 64 formatter files, and test sources
parse with the Python 3.8 grammar. Native peak RSS for the four new GHF tests
is 17,016 KiB at RAYON_NUM_THREADS=4 and one test thread; compiler and Python
processes are excluded. This small-model result does not characterize the
historical 8 GB ECP workload. Existing analytic RHF/ECP derivatives through
fourth order are unchanged.

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_ghf.py
cargo test --offline --test ghf_reference_test
cargo test --offline --test hf_variants_suite_test test_ghf
cargo test --offline --test gradient_test --test ecp_test
```

This real-GHF checkpoint preceded the complex-HF, complete invariant symmetry
and other HF-variant checks documented in the subsequent sections below.
Detailed measurement records, the continuation plan and working notes stay
outside the codebase.

## Factorized THREE-MRPT2 checkpoint

The Cholesky and explicit Coulomb-metric DF routes now use the same factorized
CASCI/amplitude/effective-CAS backend. They avoid reconstructed full-space
four-index ERIs/amplitudes and full-space rank-six GNO densities. Conventional
CASSCF starting-orbital optimization remains a separate full-integral phase,
with its own preallocation guard. See [supported THREE options](three_dsrg.md).

All 18,180 active effective-operator coefficients in 20 model/flow conditions
match the independently validated full-space GNO backend within 2.68e-15.
Four molecular Cholesky cases include rectangular MOs, CASSCF starting orbitals
and a nonlocal ECP. Their 16 unrelaxed/once/twice/iterated energies match the
independent NumPy full-Fock polynomial-fit oracle within 2.67e-13 Eh; all
48 final CI coefficients match within 1.63e-10 after exact orbital transport.
Nine legacy compatibility energies differ from the full-integral engine by
at most 3.11e-15 Eh.

Two explicit def2-SVP-RI cases match independent PySCF Coulomb-metric
integrals and full-Fock MRPT2 fits: all 162 AO integral entries agree within
1.95e-14, eight energies within 3.02e-14 Eh and all 24 relaxed CI coefficients
within 7.75e-13. Empty/full-CAS and zero-flow limits, invalid controls,
failed reference convergence and Cholesky residual/allocation guards are tested.

Every nuclear coordinate is checked in all four reference-relaxation modes:
60 Cholesky and 60 DF components. The DF stencils recompute the moving auxiliary
basis and Coulomb metric, including a nonlocal ECP. Five-point h=1e-3 energies
agree with the independent Fock oracle within 2.10e-11 Eh/bohr; the numerical
API agrees within 4.20e-8 and translation sums are below 5.00e-12. The fixed
1e-6 derivative and 1e-9 translation criteria are unchanged. These are numerical
validation/debug derivatives, not an analytic post-HF response implementation.

A 160-spin-orbital correlated-active-space test streams 47,743 sectors under
an 8 MiB workspace estimate, retaining at most 48 terms in any sector. A second
160-spin-orbital test agrees with an independent spatial regularized-MP2 sum
within 1.79e-18 Eh. No global GNO tensor is allocated in either test. The three
native kernel tests peak at 5,008 KiB RSS; the six THREE molecular/control tests
peak at 12,868 KiB, with four Rayon threads and sequential tests. Compiler and
Python processes are excluded. This does not measure the historical large-ECP
high-derivative workload.

```sh
cargo test --offline --test three_index_test
cargo test --offline --test typed_mr_test
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_three_df.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_three_df.py --gradients
```

The final regression passes 141 integration tests in 30 binaries and all targets
compile. Ruff checks 64 Python sources / 65 formatter files, and all Python
sources parse with the Python 3.8 grammar. The
implementation working record, source manifest, raw logs and measurements stay
outside the repository. Subsequent SA/MS/XMS/DWMS energy checks follow in
the next section. The counts above describe that earlier THREE checkpoint,
not the current full regression or a release-completion claim.

## SA/MS/XMS/DWMS energy checkpoint

The [multistate DSRG API](multistate_dsrg.md) now uses exact ensemble cumulants,
complete rank-two effective operators and active-only CI spaces. Full-CAS SA
and contracted SA are distinct methods. MS/XMS/DWMS combine source operators
in the selected model space; every semicanonical change transports the CI
coordinates. Optional starting orbitals use equal-weight SA-CASSCF by default,
independently of the selected output root. Spatial source Fock matrices require
verified singlet states; unsupported spin inputs fail explicitly.

Seven independent model/reference cases cover CASCI, two-root SA-CASSCF,
rectangular MOs, nonlocal ECPs, explicit DF and a complete three-singlet model
space. At zero and finite flow, the full-Fock NumPy oracle checks all 210
energies, 840 CI coefficients, 462 model-matrix elements and 44,226 source
operator coefficients. Their maximum errors are 1.43e-14 Eh, 1.47e-14,
1.51e-14 and 9.22e-15, respectively. Orbital/state phases are aligned explicitly.
An additional nonlinear finite-lambda oracle verifies all 1,365 coefficients
of five ensemble/XMS/DWMS sources within 3.76e-12.

The molecular registry checks all 210 state energies and selected-root outputs
within 9.30e-13 Eh, including independent SA-CASSCF starting-orbital optimization.
The 42 legacy energy comparisons agree within 2.67e-14 Eh. Standard correlation
energies are SCF-relative and every returned state has its singlet label.
The tests verify state-rotation covariance, the two DW limits and exact
degeneracy, zero flow, single-state/single-reference/full-CAS limits, invalid
weights/spin/dimensions/budgets, option schemas and explicit unsupported modes.

Every nuclear coordinate for both states and all five methods is checked for
Cholesky and DF, including nonlocal ECPs, the co-moving auxiliary metric and
SA-CASSCF starting orbitals: 390 components. Independent five-point h=1e-3
energies agree within 6.22e-12 Eh/bohr; the numerical API agrees within 2.58e-7
and translation sums are below 6.01e-10. Individual state energies are not
stationary under the ensemble orbital objective, so this derivative test uses
a 1e-13 native orbital-gradient tolerance and independently refines the PySCF
orbital residual before displacement stencils. The 1e-6 derivative / 1e-9
translation criteria are unchanged. These are numerical validation/debug
derivatives; analytic response remains P4.

The combined native WSL verification passes 149 distinct integration tests in
31 binaries, including the Forte comparison and the 160-spin-orbital test.
All targets compile. Ruff checks 65 Python sources / 66 formatter files;
all sources parse with the Python 3.8 grammar. The three native operator/control
tests peak at 10,140 KiB RSS and the five multistate molecular/control tests at
15,356 KiB, with four Rayon threads and sequential test execution. The large
test holds two four-component active CI vectors, streams two transformations
with 48 peak sector terms and uses an 8 MiB workspace estimate. Compilation and
Python are excluded. This does not measure the historical large-ECP run.

```sh
cargo test --offline --test multistate_test
cargo test --offline --test typed_mr_test multistate
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_multistate.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_multistate.py --gradients
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_multistate.py --verify-fits
```

The separately approved Forte binary oracle additionally verifies all 46
energies within 9.33e-14 Eh: single-state MRPT2, MRPT3, MR-LDSRG(2) and QC with
none/once/twice/iterated reference relaxation, and full/contracted SA, using H3+
and the LiH three-MO subspace with two/three-state ensembles. The unused three highest
LiH virtuals are frozen only in the oracle input to reproduce the rectangular
Hamiltonian. Public energy variables are read separately from the returned
ensemble-average energy. This Forte build explicitly rejects MS/XMS; their
independent full-Fock checks are kept distinct from executable agreement.
No executable DWMS agreement is claimed.

Implementation notes, pre-oracle source hashes, raw inputs/outputs and binary
environment records stay outside the repository. Only the public energy
fixture is shipped. Forte is invoked through its documented Psi4 input/output;
no external MR implementation source or internal MR routine is used. This
executable comparison has the stated energy/model scope. Existing analytic
HF/ECP derivatives through fourth order are unchanged; current release
verification is summarized at the top of this document.

## Complex HF and invariant spin classification

The [complex HF solver](complex_hf.md) implements ordinary-Gaussian common-gauge
orbital Zeeman and diamagnetic integrals, optional spin Zeeman coupling, complex
RHF/UHF/GHF and a variational analytic first nuclear gradient. Energy, metric
residual and density certify convergence; returned orbitals, Fock and density
are consistent. Global SU(2) Pauli-density analysis supplies the eight-class
Fukutome label for both real and complex HF results.

Nine independent Cartesian PySCF cases include zero/tilted fields, STO-3G and
6-31G with explicit split-SP AO ordering, open-shell LiH+, nonlocal ECP, a
transverse spin field, a frustrated noncollinear H3 determinant and a fixed
common-gauge origin. All nine energies agree within 1.28e-13 Eh. All 666
magnetic integral entries agree within 2.43e-17; 5,328 stationary density/Fock
entries within 7.75e-12; 2,664 arbitrary complex spin-coherence Fock entries
within 2.51e-13. Every returned orbital satisfies metric orthonormality and
the final generalized eigenproblem. S2 agrees with the independent PySCF
determinant expectation. One global proper spin rotation aligns the spin-free
noncollinear model; the same rotation acts on every density and Fock entry.

All 63 analytic nuclear components match independent five-point h=1e-3 energy
stencils within 1.12e-10 Eh/bohr. Moving-origin cases retain translation within
1.71e-13. The nine fixed-origin components are differentiated at that fixed
origin; simultaneous geometry/origin transport separately preserves energy
and every gradient component within 1e-10. Ordinary finite Gaussian bases
retain common-gauge origin dependence. No London/GIAO phases are claimed.
The 1e-6 derivative / 1e-9 moving-origin translation criteria are unchanged.

Eight independently constructed physical determinant classes are checked in
four global SU(2) frames and two real AO metrics: 64 cases. This distinguishes
spin-rotated collinear determinants from intrinsically noncollinear states and
includes paired UHF/GHF. Existing real-GHF energy, full-matrix and all 45
analytic nuclear component comparisons also pass.

The native WSL regression passes 154 tests in 32 binaries, plus six targeted
compatibility tests: 160 distinct tests in 33 binaries. All targets compile;
Ruff checks 67 Python sources / 68 formatter files and Python 3.8 grammar.
The NO compatibility calculation uses a 300-iteration budget and stricter
1e-12 energy / 1e-10 residual/density criteria after its earlier 70-iteration
budget stopped before convergence. Acceptance criteria were not relaxed.
Five sequential native complex-HF tests peak at 11,392 KiB RSS with four Rayon
threads, excluding compiler/Python. This small measurement does not describe
the historical approximately 8 GB ECP derivative workload.

```sh
cargo test --offline --test complex_hf_test --test ghf_reference_test
cargo test --offline --test hf_variants_suite_test complex_hf
cargo test --offline --test hf_variants_suite_test ghf
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python tests/reference/gen_complex_hf.py
OPENBLAS_NUM_THREADS=4 python tests/reference/gen_hf_symmetry.py
```

Detailed records and measurement harnesses remain external. Later sections
record the remaining HF-variant audits. Existing analytic RHF/UHF and
local/nonlocal ECP derivatives through fourth order are unchanged; analytic
complex-HF Hessians remain unavailable. Current API/release status is summarized
at the top of this document, and the F12 published-value discrepancy stays open.

## Scalar-relativistic HF: spin-free X2C and DKH2

Spin-free X2C uses the one-step positive-energy projector of the modified Dirac
matrix in the supplied contracted Cartesian space; DKH2 uses the matrix
Douglas-Kroll-Hess transformation in the kinetic eigenbasis (see
[scalar_relativistic.md](scalar_relativistic.md)). Both provide analytic first
nuclear gradients through matrix-function and Sylvester responses; ZORA is
explicitly unsupported.

X2C: all 424 core-Hamiltonian entries and 1,272 response entries match
independent PySCF/NumPy four-component oracles; 1,696 integral, 11,892 nuclear
tangent and 2,973 core-response entries match PySCF; five molecular energies and
all 27 nuclear components pass the 1e-8 Eh / 1e-6 Eh/bohr criteria.
DKH2: 500 matrix and 1,500 response entries agree with an independent
element-wise NumPy construction within 8.2e-13 / 5.9e-11; four molecular
energies agree within 1.7e-12 Eh; all 21 analytic nuclear components agree with
five-point energy differences within 2.6e-10 Eh/bohr (translation 6.6e-12).
External Psi4 1.11 energies (Ne, Ar, Kr, water, HCl) agree within 6.5e-12 Eh
(DKH2, c = 137.0359895) and 7.2e-11 Eh (X2C, c = 137.035999139); these speed
of light values were identified from Psi4's public input/output only.
Constant-potential, nonrelativistic and `c^-4` DKH2-X2C scaling limits pass.

```sh
cargo test --offline --test dkh2_test --test x2c_core_test
cargo test --offline --test hf_variants_suite_test relativ
python tests/reference/gen_dkh2.py        # PySCF + independent NumPy DKH2
python tests/reference/gen_dkh2_psi4.py   # Psi4 public interface
python tests/reference/gen_x2c_core.py
```

## TDHF / RPA

`run_tdhf_from_reference` solves the spin-adapted singlet and triplet RPA problems
of a closed-shell RHF reference through the (A-B)-based Hermitian reduction and
returns X/Y amplitudes normalized to (X+Y).(X-Y) = 1. Non-positive A-B or
imaginary frequencies are reported as reference instabilities instead of being
clamped. `run_tdhf` keeps the singlet-only legacy behaviour.

Complete singlet and triplet spectra of LiH, water, formaldehyde and HI (def2-SVP
with ECP) are compared with an independent (A+B)-based reduction of PySCF orbital
Hessians, cross-checked against PySCF Davidson TDHF where it converges: all 648
roots agree within 5.3e-11 Eh, degenerate-group oscillator strengths within
7.8e-10, gauge-invariant amplitude Gram spectra within 4.1e-10, and metric
orthonormality X_k.X_l - Y_k.Y_l = delta_kl holds. The formaldehyde/STO-3G triplet
instability (confirmed by PySCF's external stability analysis) is reported as an
error.

```sh
cargo test --offline --test tdhf_test
python tests/reference/gen_tdhf.py
```

## Cholesky-decomposition HF

`pivoted_cholesky_eri` factorizes the AO ERI supermatrix on the fly: only the diagonal and
the pivot columns are evaluated from complete shell quartets, so neither a four-index ERI
tensor nor the pair supermatrix is allocated. Pivots are the largest residual diagonals,
with ties within 1e-12 (relative) resolved by the smallest packed index. CD-HF reuses the
residual-certified factorized RHF driver shared with RI-HF (`run_factorized_rhf`).

For LiH, water and HCl at thresholds 1e-2, 1e-4 and 1e-6, the pivot sequences, factor
counts and residuals equal an independent dense NumPy factorization of PySCF integrals,
and CD-HF energies agree with PySCF RHF on the reconstructed integrals within 2.2e-12 Eh.
Every reconstructed water integral obeys the rigorous bound |error| <= max residual
diagonal, and delta = 1e-12 reproduces conventional RHF within 1e-9 Eh, including HI with
the def2 ECP. CD-HF has no analytic gradient (the pivot set changes with geometry).

```sh
cargo test --offline --test cd_hf_test
python tests/reference/gen_cd_hf.py
```

## TCSCF (one-pair GVB)

Fully optimized TCSCF `c1|core g^2> - c2|core u^2>` equals singlet CASSCF(2,2) with the
selected bonding/antibonding pair as the active space; `run_tcscf_from_reference` uses the
verified CAS engine (all orbital rotations) and reports active natural orbitals with
`c1 = sqrt(n_g/2)`, `c2 = sqrt(n_u/2)`. The previous implementation only rotated the two
active orbitals into each other with a bounded line search. Multi-pair GVB-PP is not
implemented.

Against PySCF CASSCF(2,2) for stretched H2, LiH, stretched HF (5 bohr) and twisted
ethylene, energies agree within 2.0e-12 Eh, active natural occupations within 7.0e-7 and
all 36 analytic nuclear-gradient components within 8.0e-8 Eh/bohr (translation < 1e-9).
Minimal-basis H2 TCSCF equals FCI within 1e-10 Eh. Weakly correlated pairs (e.g. water
HOMO/LUMO, n_u ~ 1e-3) have several CASSCF(2,2) stationary points in PySCF itself and are
therefore not used as oracle cases.

```sh
cargo test --offline --test tcscf_test
python tests/reference/gen_tcscf.py
```

## Coherent-state QED-HF

`run_qed_hf_with_options` minimizes the Pauli-Fierz RHF energy over the coherent photon
displacement, which cancels the mean-dipole terms and leaves half the dipole variance,
`E = E_HF + Tr(P q)/2 - Tr(P d P d)/4` with `d = lambda.r` and exact AO `q = (lambda.r)^2`
integrals (not a product of projected dipole matrices). The photon zero-point constant is
omitted. The previous implementation coupled the permanent dipole directly; its legacy test
assumed that a polarization without permanent dipole has a small cavity shift, which the
coherent-state theory does not imply (the out-of-plane formamide shift is 74% of the
in-plane one).

Against independent PySCF models (effective core plus `-d P d/2` exchange-like term) for
H2, He, LiH, water (STO-3G, 6-31G) and a semilocal-ECP model, all 7 energies agree within
6.4e-13 Eh, 1360 dipole/quadratic/density/Fock matrix entries within 1.7e-11, and all 45
analytic nuclear-gradient components within 8.8e-10 Eh/bohr of refined five-point energy
differences (translation 1.7e-13). Formamide/STO-3G in-plane and out-of-plane cavities
match energy-only oracle values within 1e-8 Eh and lie above the cavity-free RHF minimum.

```sh
cargo test --offline --test qed_hf_test
python tests/reference/gen_qed_hf.py
```

## Structure-to-property workflows

`tests/property_workflow_test.rs` runs complete RHF workflows with analytic forces and
Hessians: geometry optimization, electric multipoles and isotope-specific harmonic
frequencies. Water/6-31G* converges in four steps to r = 0.9473 Angstrom and
theta = 105.50 deg (dipole 2.1989 D; modes 1826.55, 4070.46 and 4188.70 cm^-1); all 81
Cartesian Hessian entries agree with the PySCF analytic Hessian within 2.7e-8 Eh/bohr^2.
85RbH/def2-SVP with the def2 ECP28 reaches r = 2.4822 Angstrom in five steps (dipole
9.6848 D, 921.50 cm^-1); all 36 Hessian entries agree within 3.8e-10 with two refined
five-point differences of PySCF analytic ECP gradients (PySCF's public RHF Hessian has no
ECP response). Raw rigid-body frequencies stay below 0.011 cm^-1 for water and 0.034 cm^-1
for RbH, and every dipole/quadrupole component transforms correctly under rigid motion.
These are same-theory comparisons, not experimental accuracy. In the earlier
four-thread measurement, the RbH case took about 45 minutes, dominated by the
ECP Hessian. This timing retains its original worker count and measurement scope.

```sh
cargo test --release --offline --test property_workflow_test
python tests/reference/gen_property_workflow.py
```

## NEO-HF

`run_neo_hf_with_options` treats selected protons as quantum particles: closed-shell RHF
electrons and a high-spin protonic determinant (`G_p = J - K`) with `h_p = T/M - V_cl`
and electron-proton attraction. Block-coordinate descent (electrons in the frozen protonic
field, protons in the frozen electronic field) is followed by one coupled DIIS over both
commutator residuals; the result has idempotent densities and residual-certified
convergence. The previous implementation returned damped, non-idempotent densities,
stopped on the energy change alone and doubly occupied the last orbital for odd electron
counts (now an error). Starting the coupled DIIS from a core guess converges HF with a
quantum proton to a stationary point 0.19 Eh above the minimum, which is why the descent
phase is required. ECP atoms are rejected; protonic basis centers are fixed.

The oracle relaxes the same model alternately with PySCF RHF and a separate exact-Hessian
trust-region orbital Newton solver for the particles, from PySCF integrals
(concatenated-molecule electron-proton ERIs, `1/|r-R|` nuclear potentials). For HF and
FHF- (one quantum proton) and water with two quantum protons (6-31G electrons with ghost
functions on the proton sites, even-tempered s/p protonic basis), all energy components
agree within 2.1e-10 Eh and both densities within 6.5e-10. Densities are idempotent with
exact particle numbers, a point-like heavy particle (exponent 1e10, mass 1e20) recovers
RHF within 1e-8 Eh, `dE/d(1/M) = <T_p>` holds to 1e-6 relative (Hellmann-Feynman), and
the energy is invariant under rotation plus translation within 1e-9 Eh.

Positronic HF is the same two-component model with one unit-mass positive particle
(`run_positronic_hf_with_options` accepts a separate positronic basis; `charge` is that of
the electronic molecule). The previous driver had the NEO defects listed above and silently
treated the 9-electron OH radical with ten electrons; odd electron counts are now errors.
PsH, positronic LiH and positronic F- (6-31G for both particles) agree with the same oracle
within 6.1e-12 Eh in every energy component and within 1e-6 in both densities.

```sh
cargo test --offline --test neo_hf_test
python tests/reference/gen_neo_hf.py
```

## UHF internal stability and PUHF

`uhf_internal_stability` returns the lowest eigenpair of the real UHF orbital Hessian
`A + B` by Davidson iteration on AO-direct products (one UHF Fock build of the symmetrized
transition densities per product; no dense `(ov)^2` matrix). Broken-symmetry `run_uhf`
follows negative modes (a 0.5 rad rotation along the eigenvector, then SCF) until the
solution is stable. Without this, twisted ethylene/6-31G (degenerate frontier orbitals in the
SAD guess) converged to -77.9306 or to the unstable -77.8305 Eh depending on parallel
rounding, and PySCF 2.13 and 2.14 stored different PUHF references for the same input.

Lowest eigenvalues agree with dense PySCF-integral Hessians within 6.3e-10 Eh for stretched
H2, LiH and H4 broken-symmetry singlets, OH, triplet CH2, the stable broken-symmetry twisted
ethylene and its closed-shell RHF point (triplet instability -0.2024 Eh). The energy
curvature along the eigenvector equals `2 lambda` to 1e-5, and at a closed-shell point the
lowest eigenvalue equals the lower of the RHF singlet and triplet stability eigenvalues. The
PUHF oracle follows PySCF stability analysis in the same way. Exact Lowdin projections then
agree with the full-CI-space projections within 1.7e-10 Eh (weights 8.4e-10), and the
twisted-ethylene projection with the independent rotation quadrature within 1e-8 Eh.

The default RHF guess for twisted ethylene/6-31G also ends at an internally unstable
closed-shell point (-77.7913 Eh, singlet instability -0.069 Eh); `run_rhf_with_stability_rotation`
reaches the stable RHF (-77.8260 Eh). Plain `run_scf` does not follow instabilities.

```sh
cargo test --offline --test uhf_stability_test
cargo test --offline --test puhf_test
python tests/reference/gen_uhf_stability.py
python tests/reference/gen_puhf.py
```

## SA-HF and sigma-SCF

Both methods now optimize explicit determinants with exact energies and closed-form orbital
gradients through a shared preconditioned L-BFGS (`src/hf_variants/orbital_optimizer.rs`).
The previous SA-HF evaluated open-shell configurations with the closed-shell spin-summed
energy expression (for H2 (1,1) this adds a spurious `(J11 + J22)/4` self-interaction) and
updated orbitals by damped diagonalization of the averaged Fock matrix, which is not the
stationary condition of the ensemble energy. The previous sigma-SCF diagonalized
`(F - omega)^2` and reported an orbital-energy spread, not `<(H - omega)^2>`.

SA-HF agrees with an independent SciPy BFGS minimization (exact expm-adjoint gradients,
checked against central differences) within 3.1e-10 Eh for the boron 2p ensemble, H2
ground plus triplet, a LiH three-determinant ensemble and a single water triplet; ensemble
densities agree within 1e-6. Single closed-shell and high-spin states reproduce RHF and
ROHF, the equally weighted boron ensemble keeps the three 2p determinants degenerate, and
the ensemble energy is stationary along random rotations.

sigma-SCF variances agree within 4.3e-14 with SciPy minimizations of the exact full-CI-space
variance `|H Phi|^2 - 2 omega <H> + omega^2` (PySCF direct_uhf sigma vectors) for doubly
excited and open-shell H2, excited HeH+ and a LiH triplet. One-electron eigenstates have zero
intrinsic variance and their exact eigenvalues, and for `omega -> -infinity` the minimizer
approaches the HF determinant with an error that falls faster than `1/omega`.

At three nonstationary orbital frames, all 18 SA-HF and all 81 spin-resolved sigma-SCF
orbital-gradient components agree with independent five-point objective differences within
8.9e-11 and 8.4e-10, respectively. These are orbital derivatives, not nuclear gradients.
The intrinsic variance is retained directly as the sum of squared single/double couplings;
a target offset of 1e9 verifies that it survives cancellation in the total target variance.
All 65 library unit tests and the eight method integration tests pass, all targets compile,
and Ruff plus Python 3.8 syntax checks pass for 81 Python sources.

```sh
cargo test --offline --test sa_hf_test
cargo test --offline --test sigma_scf_test
python tests/reference/gen_sa_hf.py
python tests/reference/gen_sigma_scf.py
```

## Numerical radial 1s HF

The supported configurations are 1s and restricted 1s-squared, with an explicit
electron count for ions. Neutral multishell atoms are rejected instead of silently
being evaluated as two-electron ions. These are nonrelativistic point-nucleus
calculations on a finite uniform second-order grid, with separate SCF, mesh and
boundary checks; the historical `hfl_total_energy` field is not an exact limit.

Seven independent SciPy tridiagonal calculations agree within 8.0e-13 Eh and
1.1e-11 for all 1,792 radial orbital values. A separate coupled orbital/Poisson
boundary-value solver gives continuum He/Li+/Be2+ energies. At 32,768 grid points
the finite-grid differences are 5.52e-7, 1.09e-6 and 1.81e-6 Eh, with the expected
second-order refinement. The helium continuum value also agrees with the
[Bunge atomic HF table](https://server.ccl.net/cca/data/atomic-RHF-wavefunctions/tables.html).

Tridiagonal Sturm/inverse iteration and Coulomb prefix/suffix sums require linear
workspace. Direct execution of the continuum-refinement test (compiler and Python
excluded) takes 0.97 seconds and peaks at 5,284 KiB RSS on the measured WSL host.
The former dense 32,768-square matrix alone would require 8 GiB. This measurement
does not characterize the separate historical approximately 8 GB ECP derivative
workload. Detailed timing, memory and source records remain external.

```sh
cargo test --offline --test numerical_atomic_test
python tests/reference/gen_numerical_atomic.py
```

## Finite-potential-basis OEP

`run_oep_hf` (exchange-only OEP by direct minimization over Gaussian potential
coefficients, fixed RHF Fermi-Amaldi baseline) is compared with an independent PySCF/SciPy
oracle built from public three-center overlap integrals and RHF J/K builds. All 16118
potential-matrix entries agree within 5.6e-16, including every Cartesian S/P/D/F/G pair
for potential exponents 0.001, 0.7 and 30 (unit-normalized AOs, native d order). At arbitrary
coefficients the energies agree within 1.2e-11 Eh and all analytic coefficient derivatives
within 2.0e-10 of the oracle's analytic values and within 3e-8 of refined five-point energy
differences; densities, KS and HF Fock matrices and KS orbital energies agree as well.
Optimized energies for He (STO-3G complete occupation, 6-31G), H2/6-31G, water/STO-3G,
water/6-31G and a semilocal-ECP HeH+ model agree within 3.8e-13 Eh and densities within
1e-6. Two-electron and complete-occupation systems reproduce RHF; water lies above RHF.

LiH/STO-3G with s potentials (0.5, 2.0) on both atoms has no open-gap minimizer within the
reached domain: BFGS from zero and from eight seeded random starts all keep lowering the
physical energy toward RHF (best 9.9e-6 Eh above) while the Kohn-Sham gap collapses (best
1.7e-8 Eh) and the coefficient gradient stays above 1e-4; a nearly flat valley leads there.
`run_oep_hf` therefore returns an explicit error for this case instead of a clipped or
regularized result.

```sh
cargo test --offline --test oep_test
python tests/reference/gen_oep.py
```

## Divide-and-conquer HF

`run_dc_hf` now implements the Yang-Lee partition (central/buffer partition matrix 1, 1/2, 0),
a common Fermi level from `Tr(PS) = N`, subsystem eigenproblems on submatrices of the global
Fock operator, and Pulay-mixed fixed-point iterations on the density residual. The previous
code weighted only the row index of each subsystem density by a 1/0 core flag and then
symmetrized, mixed densities linearly, stopped on the energy change alone, and labeled the
variational formula `Tr[P dh] + ... - Tr[W dS]` an exact analytic gradient for this
non-variational energy; that gradient now returns an error.

An independent NumPy fixed-point oracle from PySCF integrals (H8 chain/STO-3G, a water chain
/6-31G with partial buffers, hexatriene/STO-3G; beta = 200) agrees within 5.7e-12 Eh in the
energy, 1.8e-11 in every density element and 1e-7 in the Fermi level; these small buffers
deviate from RHF by +0.063, -0.030 and +0.058 Eh. With every buffer covering the chain and
beta = 2000, DC-HF equals RHF within 1e-9 Eh; buffer radii 3, 6 and 12 bohr approach RHF
monotonically; reordering the subsystems leaves the energy unchanged.

```sh
cargo test --offline --test dc_hf_test
python tests/reference/gen_dc_hf.py
```

## Fragment molecular orbital HF

FMO now solves the self-consistent monomer field and evaluates internal energies
plus pair/triple density-transfer corrections. The former embedding option had
no effect. Explicit AO maps remove the assumption that the two fragments occupy
consecutive blocks. Exchange repulsion now uses a normalized antisymmetrized
frozen determinant; the remaining relaxation is labeled CT + MIX. There is no
finite dimer cutoff, fitted screening or empirical energy smoothing.

The independent reference uses public PySCF RHF/gradient interfaces, dense NumPy
Coulomb contractions and simultaneous Jacobi polarization. Native monomers use
block Gauss-Seidel updates. Ten cases cover orders 1/2/3, embedded/isolated water
trimers, an interleaved water dimer, four H2 fragments with a nonzero trimer
environment, Li+/H-, and a synthetic semilocal-ECP HeH+ fragment plus H2.
Every internal cluster energy, density, Fock and embedding matrix and every pair
decomposition term is compared. All 144 nuclear components in the supported
analytic cases are checked against independent analytic PySCF gradients.

The maximum energy/decomposition difference is 1.22e-12 Eh. All 14,730 matrix
entries agree within 1.82e-11 and all 144 nuclear components within 1.03e-10
Eh/bohr. The native executable alone peaks at 16,016 KiB RSS in 7.32 seconds
for this small ten-case comparison (compiler and Python excluded). This does
not characterize the earlier high-order ECP derivative memory workload.

Cache/stream calculations, AO/fragment permutations and rigid coordinate
transport agree. Full-system expansion orders recover RHF. Invalid partitions,
odd/open-shell fragment counts, invalid controls and workspace limits fail
explicitly. No global four-index tensor or ERIs in public cluster results are
retained; the conservative matrix/integral estimate includes temporary copies.
This is not a process-wide RSS cap or a claim of asymptotic linear scaling.

Analytic nuclear response for an embedded expansion below full-system order
remains unsupported and returns an error. The isolated-cluster expansion and
full-system limit use analytic gradients with the correct inclusion-exclusion
weights, including order 1. There is no numerical production fallback. The
frozen-determinant decomposition is an explicit convention; no directed CT or
independent GAMESS/ABINIT-MP benchmark is claimed.

```sh
cargo test --offline --test fmo_test --test dc_fmo_test
python tests/reference/gen_fmo.py
```

## LHF / CEDA local exchange

`run_lhf_scf` now builds the Della Sala-Gorling LHF (CEDA) local exchange potential on a
native atom-centered product quadrature with exact point-Coulomb kernels for the Slater
potential, solves the full occupied correction equations, and iterates the local
Kohn-Sham orbitals to self-consistency; the energy is the nonlocal exact-exchange energy of
those orbitals. An independent oracle from public PySCF grids (levels 4 and 5, energy change
below 1e-9 Eh), point-charge integrals and RHF operators with its own CEDA equations agrees
within 3.8e-10 Eh in the energy and 3.9e-7 in every potential, Slater, Fock and density
element (grid-limited) for He (STO-3G, 6-31G), H2/6-31G, water (STO-3G, 6-31G) and a
semilocal-ECP HeH+ model. Two-electron cases equal RHF exactly without a grid; water lies
above RHF by 5.6e-4 (STO-3G) and 1.4e-3 Eh (6-31G). Cached and streamed Coulomb kernels
agree within 1e-12, an occupied-orbital rotation changes the potential only by a constant,
and the S-G point-charge operators are checked in the grid unit tests.

```sh
cargo test --offline --test lhf_test
python tests/reference/gen_lhf.py
```

## Crystal-orbital HF (1D chains)

The reference is the infinite-chain limit of finite PySCF molecular chains
(`tests/reference/gen_crystal_hf.py`): energy differences per added unit for the nonpolar
alternating H2 chain and H-capped trans-polyacetylene (STO-3G), and a least-squares fit
`E(N) = N e + b + c/N + d/N^2` for a water chain with dipoles perpendicular to the axis, whose
dipole-dipole tail produces the 1/N term. Every finite chain is checked for internal RHF
stability. PySCF 2.14 does not build `dimension=1` cells with uniform grids, and a
vacuum-padded `dimension=3` GDF/Ewald calculation differs by about 1e-3 Eh, so it is
recorded only as a coarse cross-check.

`run_crystal_orbital_hf` agrees with these limits within 8.9e-12 Eh (H2 chain, 24 k points),
1.2e-8 Eh (polyacetylene, 32 k points; the oracle's own convergence is 2.3e-8) and 2.1e-9 Eh
(water chain, 16 k points; fit change 1.0e-9). Electron counts per cell are exact. A doubled
cell with folded k points and the same real-space reach reproduces the energy within 1e-9
Eh and the band edges within 1e-8; widely spaced cells (40 bohr) reproduce the isolated RHF
molecule within 1e-7 Eh; moving the near/far boundary by five cells or raising the multipole
order from 8 to 10 changes the dipolar water-chain energy by less than 1e-9 Eh; band
energies satisfy eps(k) = eps(-k). Charged or odd-electron cells, aliasing k meshes and
nonpositive lattice constants are rejected.

```sh
cargo test --release --offline --test crystal_hf_test
python tests/reference/gen_crystal_hf.py
```
