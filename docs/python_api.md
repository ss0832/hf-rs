# Python API

Install from the extracted source directory with `python -m pip install .`,
using Python 3.8 or newer and a recent stable Rust toolchain. The Maturin build
backend compiles the native extension. The distribution `hf-rs-python` exports
`hf_rs` and installs the `hf-rs` command; the extension uses ABI3 for Python 3.8
and newer. See [CLI installation](cli.md) and [the README](../README.md) for
virtual-environment and editable-development instructions.

`hf_rs.run` and `hf_rs.Molecule` are direct aliases of the PyO3 native
bindings. `from hf_rs import _hf_rs as native` exposes those same functions
and class directly. The Python layer does not implement the scientific solvers.
Complete family inputs are in the [calculation recipes](methods.md#calculation-recipes);
all registered keys are in the [option index](methods.md#registered-method-options).

```python
import hf_rs

result = hf_rs.run(
    "mp2", [1, 1], [[0., 0., -0.7], [0., 0., 0.7]],
    unit="bohr", basis="sto-3g", cartesian=True,
    charge=0, multiplicity=1, reference="auto",
    options={"scf_energy_tol": 1e-13}, properties=["energy"],
)
print(result["e_total"])
```

`run(method, atomic_numbers, positions, *, unit, basis, ecp, cartesian, charge,
multiplicity, reference, options, properties)` performs native validation and
calculation. `unit` is `bohr` (default) or `angstrom`; Rust converts coordinates.
`options` is a dictionary of the selected registry's controls. Scalar, Boolean,
string and array values are supported. Misspelled/inapplicable controls are errors.
`properties` defaults to energy and accepts `observables` for HF models, retained `intermediates`, plus `gradient` and `hessian` where an
analytic route exists. Calculations release the Python GIL.

## Basis and ECP selection

`basis` accepts a built-in name, a Gaussian 94/BSE JSON filename, or an element
dictionary. The default is STO-3G. `hf_rs.list_builtin_basis_sets()` lists all
80 orbital families; element coverage and source versions are recorded in
the [BSE data notices](../third_party/basis_set_exchange/README.md).
Element keys can be symbols or atomic-number strings. A `"*"` entry supplies
a fallback; without it, every molecular element must have an orbital entry.

```python
mixed = hf_rs.Molecule(
    [11, 1], [[0.0, 0.0, 0.0], [0.2, 0.1, 3.1]],
    basis={"Na": "6-31g", "H": "sto-3g"},
    ecp={"Na": "lanl2dz"},
)
print(mixed.scf()["e_total"])
```

ECPs have independent selection through `ecp`: a name, scalar ECP filename or
element dictionary. If omitted, each element uses the ECP associated with its
chosen basis, including ECPs embedded in a combined Gaussian 94/BSE JSON file.
An explicit element entry overrides the global ECP choice and associated ECP.
Unspecified elements retain that global choice, or their basis-associated ECP
when no global choice was given. `ecp="none"` disables all ECPs;
`ecp={"Na": None}` explicitly disables only sodium's ECP. Global named sets
apply to their covered elements; an explicit element assignment without a
matching potential raises an error. `def2-ecp` is an independent built-in name.
ECP files can contain only potentials or combined orbital/potential data.

Effective charges and electron counts are computed after selection. The same
assigned operators enter energies and analytic HF derivatives through order
four. This permits explicit combinations; it does not establish their physical
suitability. Spin-orbit ECPs remain unsupported.

## Reference and results

`reference="auto"` selects RHF for multiplicity 1 and UHF otherwise. The `hf`
and `rhf` method names follow this reference choice; `reference="rhf"` forces
RHF. The `uhf` method forces UHF and rejects `reference="rhf"`. Correlation
aliases with RHF/UHF prefixes do not select the reference. For a broken-symmetry
singlet, set `reference="uhf"` and `options={"broken_symmetry": True}` explicitly.

The dictionary contains `method`, `reference`, `e_scf`, `e_corr`, `e_total`,
`components`, `component_units`, `observables`, `intermediates`, `states`, `root`, `one_rdm`, `gradient`, `hessian`, derivative-kind
labels and dimensions. Missing properties/RDMs are `None`. Energies are Hartree,
gradients Hartree/bohr and Hessians Hartree/bohr squared, independently of the
input unit. `e_total` follows the requested root. UHF reports `S_squared`.

```python
molecule = hf_rs.Molecule([1, 1], [[0., 0., -0.7], [0., 0., 0.7]])
print(molecule.scf())
print(molecule.run("ccsd"))
print(molecule.gradient("hf"))
print(molecule.hessian("hf"))
print(molecule.nuclear_derivative([2, 2, 2, 2]))
```

`Molecule` owns immutable input and caches SCF for identical reference/settings.
Its `.run()`, `.gradient()` and `.hessian()` accept `options=`. `.hf_variant()`
selects the molecular default-model adapter; advanced variant controls use typed
Rust APIs. `.directional_derivative()` accepts one through four flat 3N directions.
For moved atoms, construct another molecule. See [method limits](methods.md).

SCF controls use `scf_max_iter`, `scf_energy_tol`, `scf_residual_tol`,
`scf_density_tol`, `scf_conv_algo` and `broken_symmetry`. Post-HF controls use
their own `max_iter`, `energy_tol`, `residual_tol`, active-space and method keys.
Coordinate conversion and scientific method dispatch do not occur in `core.py`.

## HF physical observables

```python
import hf_rs

mol = hf_rs.Molecule([1, 1], [[0., 0., -0.7], [0., 0., 0.7]])
observables = mol.hf_observables("hf")
print(observables["dipole"])  # value, e*bohr, dimension metadata
qed = mol.run("qed-hf", properties=["observables"])
print(qed["observables"]["mean_photon_number"])
print(mol.hf_observables("tdhf")["excitation_energies"])
```

`hf_observables` is a native method. Energy-only calls do not copy observable
arrays. [HF observables](methods.md#hf-observables) lists all available quantities
and their physical scope, including specialized periodic, radial, ensemble and
excited-state controls. Rust and native Python use the same atomic-unit values;
the [ASE adapter](ase.md#units-and-observables) converts physical inputs/outputs
to eV/angstrom-based units. Ordinary post-HF calls retain method-specific scalar
`components` with explicit `component_units`, state energies and available RDMs;
the HF observable map is not a correlated response calculation.

## Migration from 0.2

Use `run("mp2", ...)`, `run("caspt2", ..., options={...})`, etc. instead of
individual post-HF `calc_*` entry points. This preserves method-specific controls,
all returned states and selected-root semantics through one boundary. Unknown
methods no longer fall back to another calculation. Analytic response capability
is checked before starting a production derivative.

HF compatibility functions (`calc_energy_and_gradient`, `calc_hessian`,
`calc_third_derivatives`, `calc_fourth_derivatives`, `calc_hf_nuclear_derivative`)
and `HartreeFock` remain for existing callers. Their positional coordinates are
bohr; `positions_angstrom=` requests Rust conversion. New applications should
use `Molecule`. The tensor compatibility functions can be expensive; mixed
directional derivatives avoid allocating full third/fourth-order output tensors.

## File and wavefunction output

```python
import json
from pathlib import Path
import hf_rs

molecule = hf_rs.Molecule([1, 1], [[0., 0., -0.7], [0., 0., 0.7]])
result = molecule.run("ccsd")
Path("result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
Path("reference.molden").write_text(molecule.scf_molden(), encoding="utf-8")
wavefunction = molecule.scf_wavefunction()
Path("reference.json").write_text(json.dumps(wavefunction, indent=2, allow_nan=False), encoding="utf-8")
print(wavefunction["channels"][0]["coefficients"])
print(wavefunction["channels"][0]["occupations"])
```

`options=` accepts only SCF controls. Pass the same controls used for the
reference calculation to reuse it. The return is ordinary text/dictionaries;
Python file I/O determines encoding, directory creation and overwrite behavior.
The native calculation releases the GIL.

### JSON representation

The schema identifier is `hf-rs-scf-wavefunction-1`. Fields include:

- `reference`, `e_scf`, `charge`, `multiplicity`, `n_alpha`, `n_beta` and units.
- `atoms`: atomic numbers, bohr positions, effective nuclear charges and the
  selected scalar ECP's local/semilocal radial terms, including `n_core`.
- `basis_functions`: zero-based `atom_index`, bohr `center`, Cartesian
  `powers=[lx,ly,lz]` and normalized primitive exponents/coefficients.
- `ao_basis_kind`: the calculation's physical Cartesian/spherical choice.
- `channels`: one `restricted` channel with occupations 2/0, or `alpha` and
  `beta` channels with occupations 1/0. Each contains MO coefficients,
  Hartree orbital energies, occupations and its AO density matrix.
- `overlap_matrix`: the Cartesian AO metric used with the coefficient rows.

Coefficient arrays have shape `(n_cartesian_ao, n_retained_mo)`, even for a
spherical calculation. Rows follow `basis_functions`; columns follow orbital
energies/occupations. Cartesian rows are lifted representations of the physical
MO space. `C.T @ S @ C` is the identity; summing `Tr(D_spin @ S)` gives the
valence electron count. The restricted channel's density is spin-summed.
Do not interpret its coefficient rows as a Molden ordering.

Primitive coefficients already include primitive and contraction normalization:
`phi_i(r) = sum_p c_ip * dx^lx * dy^ly * dz^lz * exp(-alpha_p * |r-center|^2)`.
This snapshot contains numerical definitions, rather than local source filenames.
It is not advertised as a complete restart or a correlated wavefunction.

### Molden limits and legacy calls

Molden output contains `[Atoms] AU`, `[GTO]`, `[MO]`, Cartesian/spherical flags
and ECP core counts when present. AO components are reordered to Molden's
convention; alpha/beta UHF channels are written separately. Shells above g
(`L > 4`) are rejected, even if an energy calculation supports that basis.
The scalar ECP potential itself is retained in the JSON snapshot; Molden's
core-count section does not encode the full ECP operator.

The old Python helper remains available:

```python
import hf_rs
hf_rs.export_molden_file([1, 1], [[0., 0., -0.7], [0., 0., 0.7]], file_path="legacy.molden")
```

It runs a separate legacy SCF with its own controls and has narrower basis/ECP
selection than native `Molecule`. Use the native snapshot/text methods for
mixed bases, independent ECP files and spherical spaces.

## Complete common input signatures

```text
Molecule(atomic_numbers, positions, *, unit="bohr", basis=None, ecp=None,
         cartesian=True, charge=0, multiplicity=1, reference="auto")
run(method, atomic_numbers, positions, *, unit="bohr", basis=None, ecp=None,
    cartesian=True, charge=0, multiplicity=1, reference="auto",
    options=None, properties=None)
molecule.run(method="hf", *, options=None, properties=None)
molecule.hf_variant(method, *, options=None, properties=None)
molecule.hf_observables(method="hf", *, options=None)
molecule.intermediates(method="hf", *, options=None, max_elements=1_000_000)
```

| Input | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `atomic_numbers` | integer sequence, length N | Required | Atomic numbers 1 through 118; nonempty geometry |
| `positions` | real N by 3 array | Required | Finite nuclear positions; distinct nuclei |
| `unit` | string | `"bohr"` | `"bohr"` or `"angstrom"`; controls input positions only |
| `basis` | name, file or element dictionary | `None`, selecting `"sto-3g"` | Every element needs a basis; `"*"` is an optional map fallback |
| `ecp` | name, file, element dictionary or `None` | `None` | Omission uses each orbital basis's associated scalar ECP; explicit `"none"` disables it |
| `cartesian` | Boolean | `True` | `False` selects the physical real-spherical AO space; common HF-variant adapters require `True` |
| `charge` | signed integer | `0` | Molecular charge after ECP core removal determines valence electron count |
| `multiplicity` | positive integer | `1` | 2S+1; must be compatible with the valence electron count |
| `reference` | string | `"auto"` | `"auto"`, `"rhf"`, `"uhf"`; auto follows multiplicity |
| `method` | string | Required for top-level `run`; `"hf"` for molecule `run` | Registered method or supported HF-model alias |
| `options` | dictionary or `None` | `None`, an empty dictionary | SCF and method controls; unknown/inapplicable keys are errors |
| `properties` | string sequence or `None` | `None`, requesting energy | `energy`, HF `observables`, `intermediates`, and supported analytic `gradient`/`hessian` |
| `max_elements` | positive integer | `1_000_000` | Aggregate numerical-value limit for retained intermediate copies |

`hf_rs.method_catalog()` returns every registry method's accepted keys and
`option_details`; `hf_rs.option_catalog()` describes each distinct registered
key with its type, native unit, actual typed default, description and enum
choices. `hf_rs.hf_variant_catalog()` describes the common HF-model controls
and default-model restrictions separately. A metadata default of `None` means
omission; it does not authorize passing `None` as a numerical option. Required
model inputs are identified in their descriptions. Method-specific guards
still apply to otherwise well-typed inputs.

## Integrals and retained intermediates

```text
molecule.integrals(operator="overlap", *, representation="cartesian_ao",
    indices=None, derivative_coordinates=None, coefficients=None,
    aux_basis=None, metric_cutoff=1e-10, max_elements=1_000_000,
    workspace_budget_bytes=134_217_728, options=None)
```

| Input | Type | Default | Meaning and constraints |
| --- | --- | --- | --- |
| `operator` | string | `"overlap"` | `overlap`, `kinetic`, point-charge `nuclear_attraction`, `ecp`, `core_hamiltonian`, `fock`, `fock_alpha`, `fock_beta`, `eri`, `df_metric`, `df_three_center`, `df_factors` |
| `representation` | string | `"cartesian_ao"` | Normalized Cartesian AOs, `physical_ao`, `mo_alpha`, `mo_beta`, `mo_alpha_beta`, `mo_beta_alpha` or `custom` |
| `indices` | list of integer lists or `None` | `None`, all axes | One nonempty zero-based index list per tensor axis; preserves supplied order, including repetitions |
| `derivative_coordinates` | integer list or `None` | `None`, order zero | Zero to four flattened 3N nuclear-coordinate indices; repetitions are allowed |
| `coefficients` | list of matrices or `None` | `None` | For `custom` only: one shared coefficient matrix or one per axis, with finite Cartesian-AO rows |
| `aux_basis` | string or `None` | `None` | Explicit built-in auxiliary basis required for every `df_*` operator |
| `metric_cutoff` | positive real | `1e-10` Hartree bohr cubed | Absolute Coulomb-metric eigenvalue cutoff for whitened RI factors |
| `max_elements` | positive integer | `1_000_000` | Maximum returned numerical values; select smaller index blocks when exceeded |
| `workspace_budget_bytes` | positive integer | `134_217_728` | Conservative integral/transform workspace bound; not a process-wide RSS cap |
| `options` | dictionary or `None` | `None` | Reference SCF controls only; used for MO/Fock requests |

Two-index operators use `(p,q)`; Coulomb ERIs use chemists' `(p q | r s)`
with four ordered axes. `mo_alpha_beta` means alpha coefficients on p/q and
beta on r/s; `mo_beta_alpha` reverses them. A restricted reference shares both
channels. `physical_ao` applies the fixed Cartesian-to-physical transformation,
and its indices have the physical dimension, not the Cartesian dimension.
AO and custom-coefficient ERIs do not require an SCF calculation. MO/Fock
requests use the configured cached RHF/UHF reference, irrespective of a
previously requested correlated or HF-variant energy.

The result includes the request's operator, representation and resolved index
lists, plus `tensor` with `shape`, `axes`, flat row-major `values`, `unit`,
`energy_power` and `length_power`. Overlap is dimensionless, ordinary
one-/two-electron integrals are Hartree, and each nuclear derivative contributes
bohr^-1. With individually normalized auxiliary Gaussians, the RI metric has
units Hartree bohr^3, raw three-center values Hartree bohr^1.5 and whitened
factors Hartree^0.5. Their axes are `(P,Q)`, `(P,p,q)` and `(Q,p,q)` respectively;
Q in the factors labels the retained metric eigenvectors. RI inspection currently
supports Cartesian AOs only. The factor builder also retains its intrinsic
256 MiB workspace guard; increasing the inspection budget does not bypass it.
Bare metric/three-center derivatives are analytic
through order four; factor derivatives require metric response and are rejected.

All integral derivatives hold the supplied/base-geometry MO coefficients fixed:
`orbital_response_included` is `False`. These are bare operator derivatives,
not relaxed total-energy derivatives. Use `nuclear_derivative`,
`directional_derivative`, `gradient` or `hessian` for HF energy response.
Fock derivatives are rejected because they require a density-response contract.
Invalid bounds, incompatible coefficient frames and unsupported combinations
raise errors before returning partial output.

```python
import numpy as np
import hf_rs

m = hf_rs.Molecule([1, 1], [[0., 0., -0.7], [0., 0., 0.7]])
debug = m.intermediates("hf")
def matrix(name):
    t = debug["model." + name + "_cartesian_ao"]
    return np.asarray(t["values"]).reshape(t["shape"])
d, h, f = matrix("density"), matrix("core"), matrix("fock")
energy = 0.5 * np.einsum("ij,ji", d, h + f) + 1. / 1.4
assert abs(energy - m.run()["e_total"]) < 1e-11

ovov = m.integrals("eri", representation="mo_alpha",
                  indices=[[0], [1], [0], [1]])["tensor"]["values"][0]
eps = m.scf_wavefunction()["channels"][0]["orbital_energies"]
mp2_correlation = ovov**2 / (2. * (eps[0] - eps[1]))
assert abs(mp2_correlation - m.run("mp2")["e_corr"]) < 1e-11
amplitudes = m.intermediates("ccsd")
t1 = amplitudes["method.t1"]["values"][0]
t2 = amplitudes["method.t2"]["values"][0]
assert abs(ovov * (t2 + t1*t1) - m.run("ccsd")["e_corr"]) < 1e-10
```

Intermediate output is opt-in: `run(properties=["intermediates"])` uses a total
limit of one million values; `.intermediates(..., max_elements=...)` changes
that limit. Tensors use the same row-major/unit-tagged format as integral
blocks. `model.*` denotes the actual solved HF model's retained matrices;
`method.*` denotes retained correlated amplitudes and the selected output
1-RDM. Ordinary energy calls return an empty `intermediates` map.
No four-index AO/MO ERI is copied into intermediate output.

RHF/UHF/ROHF export their actual coefficients, densities, core, overlap and
Fock matrices. GHF exports spinor tensors; complex HF exports real/imaginary
parts separately. HFB adds the pairing tensor; LHF/OEP export their local
potential/KS operators; NEO and positronic HF distinguish particle species.
FMO tensors are labeled by monomer/dimer/trimer fragment indices; their AO
rows follow each cluster's `ao_indices` in the typed Rust result and the
`monomer_ao_indices`/`dimer_ao_indices`/`trimer_ao_indices` observable maps.
CO-HF density-block keys contain the signed lattice translation. Radial HF
exports its grid and reduced radial orbital. TDHF exports X/Y response vectors.
PUHF's `model.raw_uhf.*` tensors belong to its raw UHF determinant. They
do not represent a projected one-particle density. HFB normal/pairing densities
use the real orthonormal frame spanned by its exported coefficient columns.

RHF-CCD/CCSD T2 has shape `(occupied_i,occupied_j,virtual_a,virtual_b)` and
RHF-CCSD T1 `(occupied,virtual)` in the spatial-MO convention. UHF-CCSD uses
spin orbitals, with alpha occupied then beta occupied and alpha virtual then
beta virtual. Consult the method's equations before contracting these tensors.
The method's selected available 1-RDM is exported as `method.one_rdm_mo`;
its MO frame is method-specific, particularly for optimized active references.
CCSD(T), MP2 and other energy-result types that do not retain amplitudes return
no synthesized amplitude tensors. Their reference integrals remain available,
and typed Rust results expose any additional method-specific retained state.
