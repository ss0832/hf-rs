# Analytical HF nuclear derivatives through order four

RHF and UHF support analytical nuclear-coordinate derivatives through fourth
order, including scalar semi-local ECPs. Production evaluation uses one reference
geometry. Finite differences are used only in validation tests.

Implementation:

1. Differentiate Cartesian Gaussian functions algebraically through order four. For
   an ECP at C, translational invariance converts each directional nuclear derivative
   to `(v_A - v_C).d_A + (v_B - v_C).d_B`. This includes movement of the local and
   nonlocal ECP center and all mixed derivatives. Partition the directions between
   the two differentiated AO functions and reuse the streamed quadrature engine.
2. Differentiate overlap/kinetic integrals using relative AO-center directions.
   Nuclear attraction uses the same relative-center formula as the ECP. ERIs use
   three relative AO-center directions with the fourth center as translation anchor.
   Generate only one packed derivative ERI tensor at a time where possible; do not
   parallelize whole fourth-order components or retain a global quartet list.
3. Represent at most four perturbations by square-free Taylor variables, one per
   direction. A coefficient indexed by a bit mask is the actual mixed derivative.
   Product coefficients sum over all disjoint subset partitions; repeated physical
   directions still have separate variables, so factorials arise automatically.
4. Solve SCF response order by order from the density projector equations in each
   spin's fixed reference MO frame:
   `D S D = D`, `F D S - S D F = 0`.
   RHF uses the per-spin density D=P/2 and F=h+2J(D)-K(D); UHF uses separate
   projectors and F_sigma=h+J(D_alpha+D_beta)-K(D_sigma).
5. At each nonempty mask, set the unknown density coefficient to zero and form
   the known projector residual T=(DSD-D)_mask. The diagonal blocks are
   `D_mask[oo]=-T[oo]` and `D_mask[vv]=T[vv]`; the occupied-virtual block is
   determined by the same coupled-perturbed HF matrix at every order. Form the
   known stationarity residual after inserting these diagonal blocks and solve
   `A * D_mask[vo] = -(FDS-SDF)_mask[vo]`. In RHF,
   `A_ai,bj=(eps_a-eps_i) delta_ai,bj+4(ai|bj)-(ab|ij)-(aj|ib)`.
   UHF includes both spin channels and their Coulomb coupling.
6. Contract the HF energy Taylor coefficient after solving the required response
   orders: `E_elec=Tr[(Da+Db)h]+1/2 sum_sigma Tr[Dsigma (Fsigma-h)]`.
   Differentiate nuclear repulsion by the exact polynomial expansion of r^-1.
   This is analytical implicit differentiation at one geometry, with no displaced
   SCF calculations and no finite-difference step size.
7. Reuse one reference SCF and response factorization for directional, full-tensor,
   active-atom block, and normal-mode subspace APIs. Enumerate canonical tensor
   components and copy their index permutations; never impose translation identities
   as a substitute for computing the derivative.
8. Validate against closed-form ECP Gaussian integrals and independent finite
   differences of verified lower derivatives. Include nonlocal channels, mixed axes,
   repeated directions, rigid translation, RHF and open-shell UHF. Finite differences
   remain validation tools only.

The response construction intentionally uses more orders than the minimum 2n+1
Lagrangian formula to keep a common, auditable RHF/UHF implementation. Memory and
CPU costs of high-order ERIs must be stated separately from the bounded ECP angular
workspace. High-angular-momentum Hermite tables need dynamic overflow storage.

## Low-symmetry all-component regression

`tests/hf_all_components_test.rs` includes noncoplanar HeH3+/HeH3 geometries
with six distinct pair distances, which exclude nonidentity atom permutations;
the nonzero signed volume excludes a reflection plane. The working basis has
seven Cartesian AOs, including all px/py/pz components on helium. Tests cover
all-electron RHF and model nonlocal-ECP RHF/UHF. The model ECP contains local,
semilocal s and semilocal p terms, with its center moving with the helium atom.

For each case, every one of 12 gradient, 144 Hessian, 1,728 third-order and
20,736 fourth-order entries is compared against a five-point difference of
the next lower analytic quantity. The displaced third derivatives are computed
through the checked response context; the base third/fourth tensors use the
public checked full-tensor functions. The finite-difference reference is not
symmetrized after construction. SCF energy convergence is 1e-15 Hartree and the
C1 reference step is 1.25e-3 bohr. Fixed maximum-error criteria remain
1e-6, 1e-5, 2e-7 and 3e-6 for orders one through four.

The [validation record](validation.md) gives measured outcomes. This establishes
the listed C1 s/p-AO models; it does not assert exhaustive agreement for every
element, basis family, angular momentum or singular/rank-changing geometry.

## Rust API

The checked context borrows an already converged reference and reuses its response
factorization. Coordinates are in bohr. For example, after building `atoms`, `basis`
and `hf` with the ECPs attached:

```rust,ignore
let context = hf_rs::HfNuclearDerivativeContext::from_rhf(&atoms, &basis, &hf)?;
let fourth = context.coordinate_derivative(&[2, 2, 5, 5])?;
let third = context.coordinate_derivative(&[0, 2, 5])?;
let hessian = context.hessian()?;
```

For UHF use `from_uhf`. `derivative(&[NuclearDirection])` accepts zero through four
directions, each containing one `[f64; 3]` per atom. Directions are not normalized;
the result is multilinear in them. An empty list returns the reference energy.
Errors include unconverged references, invalid directions, singular response, and
discarded physical overlap eigenvectors (`HfError::LinearDependence`). A rectangular
Cartesian-row/spherical-column MO matrix alone is not an overlap-rank failure.

`try_compute_third_derivatives_general` and
`try_compute_fourth_derivatives_general` build a reference and return checked full
tensors. The older tensor, active-atom block, directional and normal-mode subspace
functions use the same analytical backend and retain their existing return types.
Legacy wrappers panic on calculation errors; prefer the checked context or the
`try_` full-tensor functions when handling errors programmatically.

## Python API

Named def2 bases automatically attach their built-in ECPs. The following evaluates
a fourth derivative without constructing the full output tensor:

```python
import hf_rs

z = [1, 1]
r = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]  # bohr
u = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
fourth = hf_rs.calc_hf_nuclear_derivative(z, r, directions=[u, u, u, u])
```

The current cached interface can evaluate the same fourth derivative by coordinate
indices or directions:

```python
molecule = hf_rs.Molecule(z, r, unit="bohr")
fourth_coordinate = molecule.nuclear_derivative([2, 2, 2, 2])
fourth_directional = molecule.directional_derivative([u, u, u, u])
```

For the full tensor, `calc_fourth_derivatives(z, r)` returns a
`3N x 3N x 3N x 3N` nested array. This fourth-order capability concerns the
RHF/UHF energy, including its scalar local/nonlocal ECP terms. It is not an
assertion of fourth-order response for every post-HF or HF-variant model; their
individual capabilities are listed in [methods](methods.md).

Supply one through four flattened 3N directions. Unit Cartesian directions give
`Eh/bohr**order`. In the compatibility functions, set `charge`, `multiplicity`,
and `is_broken_symmetry` for UHF; the common `Molecule` instead uses `reference`
and `options={"broken_symmetry": True}` as described in the [Python API](python_api.md).
`calc_hessian`, `calc_third_derivatives`, and `calc_fourth_derivatives` also support
these spin options. Python returns errors for SCF/response failure. Native coordinates
are in bohr; `positions_angstrom=` requests conversion by Rust. The common
`Molecule` interface instead accepts `unit="angstrom"` directly. Both retain
atomic-unit derivative outputs.

## Cost and limits

ECP potential values use numerical radial/angular quadrature; derivatives of the
integrands and the SCF equations are analytical. The quadrature accuracy is still
finite. Primitive screening includes polynomial factors from high-order Gaussian
derivatives, and Hermite tables allocate overflow storage for high angular momenta.

The derivative ERI cache holds at most 64 MiB. A per-reference LRU cache reuses overlap
and core-Hamiltonian derivatives across tensor components with an 8 MiB accounting
budget (matrix data, direction keys and an entry-overhead allowance). A calculation also needs the borrowed
reference ERIs, one additional packed derivative ERI tensor, response/density matrices,
and the requested output. Whole tensor components are evaluated sequentially, while
AO integral work is parallel. The dense CPHF factorization still requires
`O((n_occ*n_virt)^2)` memory per coupled spin space. A full fourth-order output requires
`8*(3N)^4` bytes. Use directional derivatives, blocks or mode subspaces for large systems;
the bounded ECP quadrature workspace does not bound total HF memory.

AO integrals and derivatives use Cartesian work functions. Spherical reference
subspaces are supported through L=7 by the transformed MO coefficients and
densities, including all-component derivative checks. Spin-orbit ECPs and
derivatives through rank-changing overlap truncation are not implemented.
See [validation.md](validation.md) for tested systems and numerical tolerances.
