# Single-reference driven SRG

This page describes the typed Rust model and its recorded validation scope.
For the common Python/CLI adapter and analytic-property availability, see
[the current method catalog](methods.md).

`SRDSRGPT3` and `SRLDSRG2` use the same real spin-orbital NO2B algebra as
[IM-SRG(2)](imsrg2.md). The reference is a converged RHF determinant; the scalar
Hamiltonian starts at zero so all reported increments are relative to HF.

The Gaussian source and recursively truncated BCH transformation follow
[Evangelista, J. Chem. Phys. **141**, 054109 (2014),
Eqs. 41, 43, 44 and 54](https://arxiv.org/html/1406.0114).
For each single or double excitation, with occupied-minus-virtual gap
`Delta`, the implemented condition is

```text
A = T - T^dagger
C0 = H
Ck = [C(k-1), A]_NO2B / k
Hbar = sum(k >= 0) Ck
Hbar_ex = (Hbar_ex + Delta*t_ex) exp(-s*Delta^2).
```

The projection retains ranks zero, one and two after every commutator.
Consequently this transformation differs from exponentiating the physical
Hamiltonian in a determinant space and projecting only the final answer.
Both single and double amplitudes are solved, including correlation-induced
singles from a canonical RHF reference.

## Numerical solution

The fixed-point target is `(Hbar_ex + Delta*t_ex) r_s(Delta)`, where
`r_s(Delta) = -expm1(-s*Delta^2)/Delta`. The implementation evaluates the
removable singularity by an analytic series near zero. It does not floor a
denominator, replace zero flow by positive flow, or clip amplitudes. At `s=0`
all amplitudes and correlation increments are exactly zero. At a degenerate
gap the continuous regularized solution selects zero amplitude.

Anderson/Pulay acceleration combines fixed-point targets using their input
residuals. Convergence requires the maximum amplitude residual, the maximum
physical source residual, and the energy change to meet their tolerances.
The physical residual uses `expm1` to avoid cancellation near zero gap.
Failure to converge amplitudes or the BCH series returns an error.

Legacy Rust callers can pass `LdsrgOptions` through
`SRLDSRG2::with_options`. Typed registry options are:

| Option | Default |
| --- | --- |
| `flow_parameter` | 0.5 Eh^-2 |
| `max_iter` | 100 |
| `residual_tol` | 1e-10 |
| `energy_tol` | 1e-12 Eh |
| `bch_tolerance` | 1e-13, maximum absolute coefficient of the last term |
| `bch_max_order` | 80 |

PT3 accepts the flow parameter; ODE controls are reserved for IM-SRG(2).
The PT2 adapters support RHF/UHF and borrow the shared alpha/beta/mixed spatial
MO integral cache. PT3/LDSRG(2) borrow the RHF cache. Frozen
spaces, DF, active spaces, excited roots, UHF PT3/LDSRG(2), and reference
relaxation return `Unsupported`. UHF PT2 checks
the retained MO column count rather than assuming AO and MO dimensions agree.

## Third-order expansion

The PT3 implementation expands the driven equations with
`H(lambda)=H0+lambda*H1`, `H0=diagonal Fock`, `H1=H-H0` and
`A=lambda*A1+lambda^2*A2+...`. Writing each bracket with the same NO2B
projection gives the following independent algebraic construction:

```text
(A1)_ex = (H1)_ex r_s(Delta)
C01 = [H0,A1]; C11 = [H1,A1]; C011 = [C01,A1]
K2 = C11 + C011/2
(A2)_ex = (K2)_ex r_s(Delta)
E2 = (K2)_0
C02 = [H0,A2]
E3 = [H1,A2]_0
     + ([C01,A2]_0 + [C02,A1]_0 + [C11,A1]_0)/2
     + [C011,A1]_0/6.
```

This retains the source and unitary BCH contributions that a damped MP3
ladder/ring formula misses. With the diagonal MP partition and an RHF
determinant, the infinite-flow limits of `E2` and `E3` are the ordinary MP2
energy and pure third-order increment, respectively. A normal-ordered
three-body term generated at second order cannot contribute a scalar when
commuted with the first-order single/double generator.

`SRDSRGResult.perturbative` exposes both generator orders.
`SRDSRGResult.ldsrg` exposes the final generator, every transformed coefficient,
residuals and BCH order. PT3/LDSRG(2) return no one-particle density until a
corresponding response density is implemented. Nuclear derivatives for these
methods remain numerical. `sr_dsrg::numerical_gradient` forwards the method,
flow parameter and amplitude controls through the typed registry at every
independently reconverged geometry. Its common numerical backend accepts
separate SCF energy, residual and density tolerances. These checks do not
certify relaxed analytic derivatives.

## Independent verification

`tests/reference/gen_sr_dsrg.py` builds explicit fermionic Fock-space matrices.
For each retained operator it forms the physical matrix commutator, extracts
bare rank-three coefficients, normal orders them with the reference, and
discards only the normal-ordered rank-three remainder. These projected matrix
actions form a finite-dimensional superoperator. SciPy's Padé matrix
exponential evaluates its action without a nested BCH implementation.
`scipy.optimize.root` independently solves the driven equations.

PT2/PT3 reference coefficients and both generator orders come from degree-ten
polynomial interpolation of the fully solved finite-lambda equations at
eleven points. Halving the lambda spacing checks interpolation error. These
references do not evaluate the Rust perturbation formulas above.

The fixtures include a six-spin-orbital Hamiltonian with noncontiguous occupied
indices and nonzero singles, plus H2/STO-3G, He with three uncontracted s
functions, a semilocal-ECP He model and an off-axis HeH+ model with the same
ECP at three finite flows. Molecular integrals come from PySCF using
the same embedded basis coefficients. Full orbital rotations align every
one-body and antisymmetric two-body coefficient. The infinite-flow checks use
PySCF MP2 and polynomial derivatives of exact finite-lambda diagonalization
for the MP3 increment. An open-shell HeH ECP doublet additionally compares
PT2 against PySCF UMP2 energy contractions of regularized amplitudes at four
flows, including a nonzero same-spin contribution.

The native suite compares all 57,929 retained operator coefficients. Nuclear
checks compare all 24 Cartesian components of the two methods on H2 and the
ECP diatomic: five-point central energy differences at 1e-3 bohr, plus the
common two-point numerical API at 5e-4 bohr. Each native displaced reference
uses SCF energy tolerance 1e-14 and residual/density tolerances 1e-13; amplitude
residual tolerance is 1e-12. The gradient acceptance remains 1e-6 Eh/bohr and
five-point translational invariance remains 1e-9. No external DSRG implementation
supplies these values.

Reproduce in an activated scientific Python environment, with the repository as working
directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  "${VIRTUAL_ENV}/bin/python" tests/reference/gen_sr_dsrg.py
PYO3_PYTHON="${VIRTUAL_ENV}/bin/python" \
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target/native}" \
CARGO_BUILD_JOBS=1 CARGO_PROFILE_TEST_OPT_LEVEL=3 CARGO_PROFILE_TEST_DEBUG=0 \
RAYON_NUM_THREADS=4 cargo test --offline --test sr_dsrg_test -- --nocapture
```

The dense production implementation stores O(N^4) spin-orbital tensors and
uses O(N^6) commutators. A conservative 16-tensor workspace estimate rejects
models above 256 MiB before allocating the spin tensors; this excludes SCF
storage and the spatial integral cache. Current small-model results do not
establish large-system performance. See
[the machine-readable validation record (`sr_dsrg_validation.json`)](validation.md#external-validation-records)
for component counts, tolerances and measured runtime memory.
