# Methods and capability limits

Method names and aliases are parsed by `registry`; unknown names are errors.
The installed `hf-rs --list-methods` prints registered post-HF option schemas.
SCF controls use the `scf_` prefix. Common post-HF keys include `frozen_core`,
`frozen_virtual`, `root`, `n_roots`, `max_iter`, `energy_tol`, `residual_tol`,
`density_fitting` and `aux_basis`; each adapter rejects unsupported combinations.

## Common calculation boundary

| Method / aliases | Reference and options | Production analytic properties |
| --- | --- | --- |
| HF / RHF, UHF | Cartesian or spherical physical space; charge/multiplicity; named ECP bases | HF gradient/Hessian; mixed derivatives 1–4, including local/nonlocal ECP |
| MP2 | Conventional RHF/UHF; frozen core; DF branch is RHF only | Full-rank all-electron, unfrozen, non-DF gradient/Hessian only |
| SCS-MP2 | Conventional RHF/UHF; DF branch RHF only; `scs_same_spin`, `scs_opposite_spin` | Energy; response not connected |
| RI-MP2 / DF-MP2 | RHF, explicit `aux_basis`, same physical fitting space | Energy |
| MP3, SCS-MP3 | Canonical RHF/UHF where supported; spin scales and `scs_third_order` | Energy |
| MP4 / `mp4sdtq`, MP5–MP8 | Complete exact canonical RHF space | Energy |
| CCD, CCSD, CCSD(T) | CCD/(T) RHF; CCSD RHF/UHF with existing guards; amplitude residual convergence | Energy |
| CIS | RHF; singlet/triplet, selected root and multiple roots | Excitation energy/state output; no excited-state nuclear response |
| CISD | RHF single-reference adapter, selected roots | Energy/RDM/state output |
| FCI / CASCI, CASSCF | RHF/UHF source reference with one common spatial-orbital frame; active space, roots, spin targeting and orbital/state controls | Energy/RDM/state output; full analytic response deferred |
| IC-CASPT2 | Single-state CASCI/CASSCF reference; IPEA, real/imaginary shifts and overlap cutoff; validation scope in CASPT2 document | Energy; multistate CASPT2/DF explicitly unsupported |
| MRCISD / MRCI, MRCISD+Q | Common spatial-orbital active reference, including open-shell populations, and reference optimization | Energy/RDM; +Q uses relaxed reference weight |
| DOCI | Complete exact canonical RHF space, seniority zero | Energy |
| DLPNO-MP2/CCSD/(T0)/(T1) | Explicit RI auxiliary basis, PAO/PNO/TNO and pair/domain thresholds | Energy; adaptive and fixed-space production analytic response deferred |
| MP2-F12 | RHF, explicit CABS; exact 3C/3*A, fitted 3C; hybrid controls | Energy; numerical debug routes remain separate |
| CCSD-F12a/b, CCSD(T*)-F12a/b | RHF 3C, shared converged CC amplitudes, distinct a/b energy functionals | Energy; analytic F12 response deferred |
| ACI, ASCI, HBCI / HCI, PCI | Selected-space thresholds; root options where supported | Energy/RDM; analytic response deferred |
| DMRG / DMRGCI, DMRGSCF | Active space, bond dimensions, sweep schedule, truncation; two-site U(1)xU(1) MPS/MPO | Energy/RDM; multiple roots/noise/S2 projection not connected |
| SR-DSRG-PT2/PT3, SR-LDSRG(2) | PT2: RHF/UHF; PT3/LDSRG(2): RHF only; flow and iterative controls | Energy; production analytic response deferred |
| IM-SRG(2) / SRGODE | Generator, flow interval/step and integration tolerances | Energy; adaptive ODE checks against independent integration |
| DSRG-MRPT2/PT3, MR-LDSRG(2), +QC | Active reference, exact cumulants, reference relaxation and convergence | Energy; H3 and unsupported spin sectors return errors |
| THREE-DSRG-MRPT2 | Common explicit DF or factorized exact kernels | Energy; source/space limits documented separately |
| SA-/MS-/XMS-/DWMS-DSRG-PT2, SA-DSRG-PT2c | Multiple roots, state weights, effective-Hamiltonian controls | Selected-state energies and state output; documented spin/source limits |

An available low-level numerical debug driver does not make a production gradient
analytic. The common API rejects such requests. `MethodId::gradient_kind` also
describes legacy/debug routes; the common API exposes only verified analytic ones.
RDM availability is method-specific and is reported as `None` when absent.

Legacy `ccsdt` resolves to perturbative CCSD(T); there is no full iterative
CCSDT adapter. `dlpnoccsdt` resolves to the local iterative-(T1) correction,
with the precise model defined in [local correlation](local_correlation.md).
Method-name RHF/UHF prefixes select an alias of the correlation method; the
reference is specified separately by `reference` and multiplicity.

### Nuclear derivative orders

| Energy model | Analytic nuclear-coordinate orders | Interface and scope |
| --- | --- | --- |
| RHF/UHF, including scalar local/nonlocal ECPs | 1, 2, 3, 4 | Checked Rust HF response; common `Molecule` coordinate/directional derivatives; Python full-tensor compatibility functions |
| Canonical MP2 in the common API | 1, 2 | Full-rank all-electron, unfrozen, non-DF space only |
| Real/complex GHF, ROHF, QED-HF, X2C/DKH2, TCSCF and supported DC/FMO limits | 1 in their documented typed models | Common adapter exposes only the subset listed below; no general fourth-order HF-variant claim |
| Other post-HF models | No complete production analytic response exposed by the common API | Numerical verification/debug functions remain numerical; higher-order analytic response is follow-up work |

HF response is local to the converged branch and retained differentiable space.
Discarded overlap eigenvectors, singular response and unsupported spin-orbit ECPs
are reported as errors. See [analytic HF/ECP derivatives](analytic_hf_derivatives.md)
for all-component validation and full fourth-order tensor versus mixed directions.

## Active spaces and scientific options

The active space uses `active_n_alpha`, `active_n_beta`, `active_n_orb` and
optional `active_orbitals` (explicit MO indices). `root` selects `e_total` and
`n_roots` controls the number of states. Inactive occupations must be compatible
with the supplied reference and the selected adapter. Unsupported spin/reference
choices are rejected; an alias does not silently change the reference.

Important method groups and their detailed equations/options:

- [Local correlation](local_correlation.md): pair-Galerkin RI model, domain/PNO/TNO
  truncation, full limits and the precise (T0)/(T1) conventions. Large-system scaling
  is separate from these verified energy limits.
- [F12](f12_integrals.md): `cabs_basis`, `beta`, `f12_approximation`,
  `f12_hybrid`, correction/Fock fitting bases and physical spaces,
  `f12_fit_orbital_mp2`, `f12_exact_obs_coulomb`, CABS singles and workspace controls.
  Public/default printed-digit energy agreement is incomplete. Existing default
  Gaussian coefficients remain unchanged; these models are experimental.
- [CASPT2](caspt2.md), [DMRG](dmrg.md), [selected CI](selected_ci.md).
- [SR-DSRG](sr_dsrg.md), [IM-SRG](imsrg2.md), [MR driven](mr_driven.md),
  [THREE](three_dsrg.md), [multistate](multistate_dsrg.md).

## Molecular HF-variant adapter

`Molecule.hf_variant(name, options=..., properties=...)` and `run(name,...)`
share the registry. The current molecular adapter uses Cartesian working space.
Its default-model interface accepts `scf_max_iter` and `scf_energy_tol`. Additional
controls below are explicit; unknown or ignored controls are rejected.

| Name / aliases | Common adapter | Additional controls / limits |
| --- | --- | --- |
| ROHF | Energy and analytic gradient | Supplied multiplicity, not inferred from electron parity |
| GHF | Energy and analytic gradient | Unconstrained real determinant; common adapter accepts multiplicity 1 input, not a total-spin projection; typed Rust initial densities for other branches |
| QED-HF / QED / polaritonic | Energy and analytic gradient | `cavity_frequency`, `coupling` three-vector; same operator for energy and gradient |
| DF-HF / RI-HF / densityfitting | Energy | `aux_basis` (default `def2-svp-ri`), `metric_tol`; actual Coulomb-metric RI-JK |
| PUHF / PHF / SUHF, HFB, TCSCF / GVB | Default-model energy | Typed Rust APIs expose projection, pairing and active-pair controls |
| X2C / relativistichf, DKH2 | Default-model energy | Analytic gradient is available through the typed Rust module |
| ZORA | Recognized but unsupported | Its numerical integration-grid Hamiltonian is not implemented |
| ComplexHF / CHF / CRHF, CUHF, CGHF | Default-model energy | Typed Rust complex/magnetic operators and analytic gradients; complex Hessian response not implemented |
| CD-HF / Cholesky, LHF, OEP / OEP-EXX | Default-model energy | Typed options expose decomposition/grid/potential-basis controls |
| NEO-HF / NEO, positronicHF | Default-model energy | Read the typed model's particle/basis and one-particle-sector limits |
| DC-HF / DC, FMO / FMO2 / FMO3 | Default-model energy | Charged FMO needs typed fragment charges; `fmo3` selects body order 3; `fmo`/`fmo2` select order 2 |

These default adapters require multiplicity 1 except ROHF. Their typed Rust
modules retain the broader independently validated models and analytic properties.
All implemented adapters also accept `properties=["observables"]` to expose the
method-specific quantities listed below. QRHF, MOM, Fermi-HF, crystal CO-HF,
radial atomic HF, SA-HF, sigma-SCF and TDHF have the explicit common controls
listed under [specialized HF inputs](methods.md#specialized-hf-inputs).
GASCI/GASSCF and AVAS/ASET retain typed Rust interfaces; their registry names
return an explicit unconnected-adapter error at the common boundary.

See [GHF](ghf.md), [complex HF](complex_hf.md), [scalar relativity](scalar_relativistic.md),
[analytic HF/ECP derivatives](analytic_hf_derivatives.md) and [validation](validation.md)
for the exact verified scope. Scientific references are collected in
[references](references.md).

## HF observables

Use `molecule.hf_observables(name, options=...)` in native Python,
`molecule.hf_observables(name, &options)` in Rust, or
`atoms.calc.get_observables(atoms)` in ASE. Equivalently request
`properties=["observables"]` in `run`. Energy-only calls leave this map empty,
avoiding copies of densities and orbital arrays. Each map entry has `value`,
`unit`, `energy_power` and `length_power`. Powers describe its physical dimension
and support ASE conversion. Rust/native Python use atomic units; ASE uses eV and
angstrom. Missing frontier orbitals or empty crystal conduction bands are `null`.

| HF model | Quantities exposed by the observable API |
| --- | --- |
| RHF / MOM | Orbital energies, HOMO/LUMO and gap, electronic/nuclear energies, dipole and Cartesian quadrupole, Mulliken/Lowdin charges |
| UHF | Separate alpha/beta frontier levels and gaps, electron counts, S-squared and spin contamination, multipoles and charges |
| ROHF / QRHF | Orbital energies, closed/open orbital counts, S-squared, multipoles and charges |
| GHF | Spin vector, S-squared, Fukutome classification, spinor energies/density, multipoles and charges |
| PUHF | Projected/unprojected energies, target spin and projection weights, unprojected S-squared, available high-spin/Yamaguchi estimates |
| HFB | Pairing gap/tensor, normal density, chemical potential, quasiparticle energies, particle-number variance and energy decomposition |
| TCSCF | Configuration coefficients, natural occupations, active pair, correlation energy and S-squared |
| QED-HF | Dipole self-energy, coherent displacement, mean photon number, cavity frequency/coupling and molecular dipole |
| DF-HF / CD-HF | Auxiliary dimension or Cholesky rank, available fitting error, factorization error and orbital energies |
| X2C / DKH2 | Model identifier, scalar-relativistic shift, nonrelativistic reference and orbital energies |
| Complex HF/UHF/GHF | Spin vector/S-squared, symmetry classification, real/imaginary density, imaginary-density norm and gauge origin |
| LHF / OEP | Orbital energies/gap and available local-potential matrices; LHF exchange/Slater terms; OEP KS Hamiltonian/residual |
| NEO-HF | Electron/proton energies and densities, proton kinetic energy and electron-proton attraction |
| Positronic HF | Electronic and positronic energies/densities, positron kinetic/orbital energy and attraction |
| DC-HF | Chemical potential, electron count, subsystem count, density and density residual |
| FMO2 / FMO3 | Fragment atoms, monomer/dimer energies, trimer cluster energies and connected corrections with indices; pair interaction, electrostatic, exchange-repulsion, charge-transfer-plus-mixing and embedding corrections |
| TDHF / RPA | Excitation energies, oscillator strengths, transition dipoles, spin-manifold flags and X/Y amplitudes |
| Crystal HF | Energy per cell and its decomposition, band energies/gap/edges, fractional and Cartesian k vectors, lattice constant and electrons per cell |
| Radial HF | 1s orbital and total energies, radial grid/reduced orbital, kinetic/attraction/repulsion energies and virial error |
| SA-HF | Ensemble/state energies, weights and orbital energies |
| Sigma-SCF | Target energy, energy and intrinsic variances, alpha/beta counts |
| Fermi-HF | Chemical potential, fractional occupations, thermal energy kBT, entropy divided by kB and free energy |

Multipoles use the global coordinate origin. AO populations depend on the basis.
Gaps are orbital gaps, not optical/fundamental gaps. QED photon numbers describe
the implemented coherent-state ansatz. The complex imaginary-density norm is
not a physical current density. FMO charge transfer includes mixing and is not
a directed charge-transfer analysis. The pair identity is
`interaction = electrostatic + exchange_repulsion + charge_transfer_and_mixing`;
the separately reported embedding correction is already included in that
decomposition and must not be added again. FMO3 `trimer_corrections` are connected
three-body contributions; `trimer_internal_energies` are the cluster energies.
Positron binding requires a separately
computed parent energy: `E_parent - E_electron+positron`; the orbital energy alone
is not that binding energy. Annihilation rates and unsupported response functions
are not supplied by these APIs. ZORA remains unsupported.

### Specialized HF inputs

These common adapters accept Cartesian energy/observables requests and
`scf_max_iter`/`scf_energy_tol`; their complete typed Rust options remain available.
All following numerical inputs are in atomic units for Rust/native Python.

| Name | Required / additional options | Scope |
| --- | --- | --- |
| `qrhf` | Molecular multiplicity | Quasi-restricted natural orbitals |
| `mom` | `occupied_orbitals=[...]` | Closed-shell occupied guess-orbital indices |
| `fermi` | `thermal_energy` | Positive kBT in Hartree; closed-shell thermal ensemble |
| `tdhf` / `rpa` | `n_singlets=1`, `n_triplets=0`, `root=0` | Stable RHF; root indexes singlets then triplets; total energy is reference plus selected excitation |
| `sa-hf` | `configurations=[{"alpha":[...],"beta":[...]}]`, `weights=[...]` | Same electron count in every determinant; weights sum to one; returned total is the ensemble energy |
| `sigma-scf` | `configuration={"alpha":[...],"beta":[...]}`, `target_energy` | Variance minimization; target in Hartree, variance in Hartree squared |
| `crystal-hf` / `cohf` | `lattice_constant`, `n_kpoints=16` | Neutral all-electron closed-shell chain periodic along z; energies per cell |
| `radial-hf` / `atomic-hf` | `n_grid_points=4096`, optional `r_max` | One all-electron atom/ion with one or two electrons; finite radial grid; Gaussian orbital basis unused |

```python
import hf_rs

mol = hf_rs.Molecule([1, 1], [[0., 0., -0.7], [0., 0., 0.7]])
print(mol.hf_observables("hf")["dipole"])
print(mol.hf_observables("qed-hf", options={"cavity_frequency": 0.5, "coupling": [0., 0., 0.02]}))
print(mol.hf_observables("tdhf")["oscillator_strengths"])
print(mol.hf_observables("mom", options={"occupied_orbitals": [0]}))
print(mol.hf_observables("fermi", options={"thermal_energy": 0.01})["occupations"])
configuration = {"alpha": [0], "beta": [0]}
print(mol.run("sa-hf", options={"configurations": [configuration], "weights": [1.]}))
print(mol.run("sigma-scf", options={"configuration": configuration, "target_energy": -1.1167143251757503}))
helium = hf_rs.Molecule([2], [[0., 0., 0.]])
print(helium.hf_observables("radial-hf", options={"n_grid_points": 256})["virial_error"])
print(helium.hf_observables("crystal-hf", options={"lattice_constant": 8., "n_kpoints": 4})["band_energies"])
```

## Calculation recipes

The common Rust, native Python and ASE interfaces use the same method strings
and option keys. This page supplies starting inputs for every connected method
family. The [complete option index](methods.md#registered-method-options) lists every registered
method; [capability limits](methods.md) specify reference and response support.
These small examples demonstrate the interface, rather than basis-set accuracy.

### Translate a recipe between the three interfaces

In native Python, construct the molecule once and reuse its SCF reference:

```python
import hf_rs

h2 = hf_rs.Molecule([1, 1], [[0., 0., -0.7], [0., 0., 0.7]])
lih = hf_rs.Molecule([3, 1], [[0., 0., 0.], [0., 0., 3.]])
active = dict(active_n_alpha=1, active_n_beta=1, active_n_orb=2)
result = lih.run("caspt2", options={**active, "ipea_shift": 0.25})
print(result["e_scf"], result["e_corr"], result["e_total"])
```

The matching complete Rust call is:

```rust
use hf_rs::api::{Molecule, MoleculeOptions, OptionMap};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let lih = Molecule::new(
        &[3, 1], &[[0., 0., 0.], [0., 0., 3.]], MoleculeOptions::default(),
    )?;
    let options = OptionMap::from([
        ("active_n_alpha".into(), json!(1)),
        ("active_n_beta".into(), json!(1)),
        ("active_n_orb".into(), json!(2)),
        ("ipea_shift".into(), json!(0.25)),
    ]);
    let result = lih.run("caspt2", &options, &["energy".into()])?;
    println!("{}", result.to_json());
    Ok(())
}
```

The same calculation through ASE takes coordinates in angstrom and returns eV:

```python
from ase import Atoms
from ase.units import Bohr, Hartree
from hf_rs import HFCalculator

atoms = Atoms("LiH", positions=[[0., 0., 0.], [0., 0., 3. * Bohr]])
atoms.calc = HFCalculator(
    method="caspt2", basis="sto-3g",
    options=dict(active_n_alpha=1, active_n_beta=1, active_n_orb=2, ipea_shift=0.25 * Hartree),
)
print(atoms.get_potential_energy())
print(atoms.calc.calculation_result["e_total"])  # eV
```

Use each recipe's method and dimensionless options in any of these call patterns.
Convert physical inputs to ASE's eV/angstrom units as shown for `ipea_shift`;
native Python and Rust examples use atomic units. See [ASE units](ase.md#units-and-observables).
Energy-only methods work with `get_potential_energy()`; requesting forces still
requires an implemented analytic response. ASE does not infer such support from
the existence of an energy method.

### RHF, UHF and open-shell references

```python
import hf_rs

hydrogen = hf_rs.Molecule([1], [[0., 0., 0.]], multiplicity=2, reference="uhf")
print(hydrogen.run("uhf")["components"]["S_squared"])
print(hydrogen.gradient("uhf"))
rohf = hf_rs.run("rohf", [3], [[0., 0., 0.]], multiplicity=2)
print(rohf["e_total"])
```

For an open-shell correlated calculation, specify `reference="uhf"` and the
physical multiplicity in `MoleculeOptions`, native `Molecule`, or `HFCalculator`.
An alias such as `uhf-mp2` names the method, without changing the reference.
Broken-symmetry singlets require `reference="uhf"` and `broken_symmetry=True`.

### Molecular HF variants

The following native calls use `h2` from the first example. The corresponding
Rust call is `molecule.hf_variant(name, &options, &properties)`; ASE uses
`HFCalculator(method=name, options=options)`. `run(name, ...)` is also accepted.

```python
qed = h2.hf_variant("qed-hf", options={
    "cavity_frequency": 0.5, "coupling": [0., 0., 0.02],
}, properties=["energy", "gradient"])
ri = h2.hf_variant("ri-hf", options={"aux_basis": "def2-svp-ri", "metric_tol": 1e-10})
print(qed["e_total"], qed["gradient"], ri["e_total"])
```

| Method strings | Input / controls | Common output |
| --- | --- | --- |
| `rohf` | Explicit multiplicity; `scf_max_iter`, `scf_energy_tol` | Energy and analytic gradient |
| `ghf` | Real unconstrained default model; common multiplicity-1 input | Energy and analytic gradient |
| `qed-hf`, `qed`, `polaritonic` | `cavity_frequency`, `coupling=[x,y,z]` | Energy and analytic gradient |
| `ri-hf`, `df-hf`, `densityfitting` | `aux_basis`, `metric_tol` | Energy |
| `cd-hf`, `cholesky` | Molecular default Cholesky threshold | Energy |
| `puhf`, `phf`, `suhf` | Default spin-projected model | Energy |
| `hfb` | Default pairing model | Energy |
| `tcscf`, `gvb` | Default occupied/virtual pair | Energy |
| `x2c`, `relativistichf`, `dkh2` | Default scalar-relativistic model | Energy |
| `complexhf`, `crhf`, `chf`, `cuhf`, `cghf` | Default complex model | Energy |
| `lhf` | Default numerical integration grid / CEDA model | Energy |
| `oep`, `oep-exx` | Default local potential basis | Energy |
| `neo-hf`, `neo` | Default electronic/quantum-proton model | Energy |
| `positronichf`, `positron` | Default one-positron model | Energy |
| `dc-hf`, `dc`, `divideandconquer` | Default molecular partition | Energy |
| `fmo`, `fmo2`, `fmo3` | Default neutral fragments; body order 2/3 | Energy |

Except ROHF, these common default-model adapters require multiplicity 1 and
Cartesian working space. Their accepted convergence controls are
`scf_max_iter` and `scf_energy_tol`; other SCF controls are rejected here.
Advanced pairing, magnetic fields, initial spin density, fragment charges,
grids and periodic/atomic inputs use the [typed Rust routes](rust_api.md#specialized-hf-variants).
`zora` is recognized and unsupported. Additional common adapters use the
[specialized controls](methods.md#specialized-hf-inputs).

### Perturbation and coupled cluster

```python
for name in ("mp2", "scs-mp2", "mp3", "scs-mp3", "mp4", "mp5", "mp6", "mp7", "mp8",
             "ccd", "ccsd", "ccsd(t)"):
    value = h2.run(name)
    print(name, value["e_corr"], value["e_total"])
```

`scs_same_spin`, `scs_opposite_spin` and `scs_third_order` set the documented
spin scaling where supported. `max_iter`, `energy_tol` and `residual_tol`
control correlation iterations; `scf_*` keys control the reference separately.
`frozen_core` and `frozen_virtual` count spatial orbitals, subject to each
adapter's guards. MP4–MP8 and DOCI require the complete exact canonical RHF
space. `ccsdt` is an alias for perturbative CCSD(T), not iterative CCSDT.

```python
ri_mp2 = h2.run("ri-mp2", options={"aux_basis": "def2-svp-ri"})
print(ri_mp2["e_total"])
```

DF-MP2 uses the same `ri-mp2` alias. For an explicit DF branch of another
supported method, set `density_fitting=True` and supply `aux_basis`.
Analytic MP2 gradient/Hessian requests require the documented all-electron,
full-rank, unfrozen, non-DF space.

### CI, active spaces and selected roots

```python
ci = lih.run("fci", options={**active, "n_roots": 2, "root": 1})
print(ci["root"], ci["e_total"], ci["states"], ci["one_rdm"])
cas = lih.run("casscf", options=active)
print(cas["e_total"])
```

`active_n_alpha`, `active_n_beta`, `active_n_orb` describe active electrons
and orbitals; inactive orbitals are doubly occupied. Optional
`active_orbitals=[...]` supplies distinct zero-based original MO indices.
Without that list, active orbitals start after the inactive occupied space.
`state_weights` and `target_s_squared` apply only to adapters that implement
them. `root` selects `e_total`; retain `states` to keep all returned roots.

| Method | Example options / output |
| --- | --- |
| `cis` | `n_roots=2`, `root=0`, `cis_triplet=False`; returned energies include the selected excitation above SCF |
| `cisd` | Selected roots where available; correlated energy and available RDM |
| `fci`, `casci` | Active options, `n_roots`, `root`; exact selected-space states |
| `casscf` | Active options, `max_orbital_iterations`, `orbital_gradient_tol`, `state_weights` |
| `caspt2` | Active options, `ipea_shift`, `real_shift`, `imaginary_shift`, `caspt2_metric_tol` |
| `mrcisd`, `mrci`, `mrcisd+q` | Active options, `optimize_orbitals`; +Q uses reference weight |
| `doci` | Full canonical RHF space; seniority-zero model |

CASPT2 is single-state here. Read [CASPT2](caspt2.md) for shift conventions and
the unsupported multistate/DF combinations. The common API returns available
RDMs and states, rather than a complete correlated-wavefunction restart file.

### Selected CI and DMRG

```python
for name in ("aci", "asci", "hbci", "pci"):
    value = lih.run(name, options=active)
    print(name, value["e_total"], value["one_rdm"])
dmrg = lih.run("dmrg", options={**active, "bond_dimensions": [8, 16], "sweeps": 8})
print(dmrg["e_total"])
```

PCI uses the active-space schema but has its own projector options. ACI/ASCI/HCI
selection and PT2 controls are listed in the [option index](methods.md#registered-method-options).
`dmrgscf` adds orbital optimization using the same active-space definition.
`bond_dimensions` is a sweep schedule, not an accuracy guarantee. Nonzero
`noise`, multiple roots and total-spin projection remain unsupported by this
DMRG adapter. See [DMRG](dmrg.md) and [selected CI](selected_ci.md).

### Local correlation

```python
local = h2.run("dlpno-mp2", options={
    "aux_basis": "def2-svp-ri", "full_domains": True,
    "t_cut_pairs": 0., "t_cut_pno": 0.,
})
print(local["e_total"])
```

Use the same pattern for `dlpno-ccsd`, `dlpno-ccsd(t0)` and
`dlpno-ccsd(t1)`. T0/T1 and finite PNO/TNO spaces are distinct approximations;
read [local correlation](local_correlation.md) before selecting thresholds.
Fixed-domain and transported-space controls are typed Rust interfaces.
Production analytic local-correlation forces remain unavailable.

### Driven and flow methods

```python
sr = h2.run("sr-dsrg-pt2", options={"flow_parameter": 0.5})
mr = lih.run("dsrg-mrpt2", options={**active, "flow_parameter": 0.5,
                                    "reference_relaxation": "once"})
print(sr["e_total"], mr["e_total"])
```

| Family | Method strings | Important options |
| --- | --- | --- |
| Single reference | `sr-dsrg-pt2`, `sr-dsrg-pt3`, `sr-ldsrg(2)` | `flow_parameter`; LDSRG BCH/residual controls |
| Differential flow | `imsrg(2)`, `srgode` | `flow_parameter`, `ode_step`, `ode_tolerance`, `ode_generator` |
| Multireference | `dsrg-mrpt2`, `dsrg-mrpt3`, `mr-ldsrg(2)`, `mr-ldsrg(2)+qc` | Active options, `flow_parameter`, reference relaxation; method-specific BCH controls |
| Three-index MRPT2 | `three-dsrg-mrpt2` | Active options; explicit `density_fitting`/`aux_basis` or Cholesky tolerance |
| Multistate | `sa-dsrg-pt2`, `sa-dsrg-pt2c`, `ms-dsrg-pt2`, `xms-dsrg-pt2`, `dwms-dsrg-pt2` | Active options, `n_roots`, `root`, `state_weights`; DWMS `dynamic_weight_parameter` |

Reference-relaxation values are `none`, `once`, `twice`, `iterate`.
MR H3 sources and unsupported spin sectors fail explicitly. Multistate
`states`/`root` semantics and unavailable executable oracle comparisons are
described in [multistate DSRG](multistate_dsrg.md). These are energy interfaces;
numerical debug derivatives do not enable production analytic forces.

### Experimental F12

```python
f12 = h2.run("mp2-f12", options={
    "cabs_basis": "cc-pvdz-f12-optri", "beta": 1.0,
    "f12_approximation": "3C(FIX)", "cabs_singles": True,
})
print(f12["e_total"], f12["components"])
```

The related names are `ccsd-f12a`, `ccsd-f12b`, `ccsd(t*)-f12a` and
`ccsd(t*)-f12b`. Physical auxiliary/CABS spaces, fitting basis kinds, frozen
spaces and F12 hybrids must match the intended model. Read the
[F12 controls](f12_integrals.md) and [option index](methods.md#registered-method-options).
The published/default printed-digit discrepancy remains unresolved. A working
API and component tests do not close that scientific validation item.

### Errors and outputs

Use `result.to_json()` in Rust, the returned dictionary in native Python, or
`calculator.calculation_result` after an ASE calculation to preserve `components`,
`states`, selected `root` and available `one_rdm`. Native values are atomic units;
ASE's complete calculation dictionary uses eV/angstrom-based units.
Missing RDMs are `None`/`null`.

Unknown options, unsupported references, unconverged equations and unavailable
analytic properties are errors. There is no fallback method. GASCI/GASSCF,
AVAS/ASET require their low-level Rust
interfaces; a registry name alone does not imply a connected common adapter.
See [wavefunction and file output](python_api.md#file-and-wavefunction-output) for MO coefficients, Molden,
SCF snapshots and access to typed correlated amplitudes.

## Output capabilities

| Output | Rust | Native Python / ASE | Scope |
| --- | --- | --- | --- |
| Calculation JSON | `CalculationResult::to_json()` | `run()` dictionary; ASE `calculation_result` | Energies, components, states, available RDM/derivatives |
| SCF wavefunction JSON | `Molecule::scf_wavefunction(ScfOptions)` | `Molecule.scf_wavefunction(options=...)` | Configured RHF/UHF orbitals, occupations, density, overlap, geometry, basis and scalar ECP |
| SCF Molden | `Molecule::scf_molden(ScfOptions)` | `Molecule.scf_molden(options=...)` | Configured RHF/UHF reference, Cartesian/spherical through g |
| Already computed RHF/UHF Molden | `molden::try_generate_molden_string_rhf/uhf`, `write_molden_file_rhf/uhf` | Legacy `export_molden_file` performs a separate SCF | Real orbitals; ECP core count included |
| Retained solver tensors | Typed result fields and `Molecule::intermediates` | `Molecule.intermediates`; ASE `get_intermediates` | Available CCD/CCSD T2, CCSD T1, selected RDM and actual HF-model matrices; no synthesized absent wavefunction |

`scf_*` exports use the reference configured when constructing `Molecule`,
just like `.scf()`. If UHF is required, set `reference="uhf"` explicitly.
Calling `run("uhf")` on an auto-reference singlet does not change the immutable
molecule configuration. QED/GHF/complex/paired/fragment states and optimized
CASSCF orbitals are not exported by these SCF-reference methods.

Molden and this JSON snapshot are visualization/data outputs. There is no
general restart reader, WFN/WFX writer, cube writer or complete correlated
checkpoint API. A correlated 1-RDM is not a full many-electron wavefunction.

## Registered method options

This index is generated from `api::method_catalog()` / `hf_rs.method_catalog()`.
It covers every `MethodId`; molecular HF variants have the separate table above.
The registry schema lists accepted names, while physical/reference combinations
are checked by each adapter. A schema entry does not guarantee every option
combination or production analytic property. `numerical_gradient_step` belongs
to low-level numerical/debug routes, not analytic force selection.

All connected post-HF rows also accept these common keys:

`frozen_core`, `frozen_virtual`, `root`, `n_roots`, `max_iter`, `energy_tol`,
`residual_tol`, `density_fitting`, `aux_basis`, `numerical_gradient_step`.

HF/ROHF/GHF are exceptions: RHF/UHF accept only the documented SCF keys;
the ROHF/GHF default adapters accept only `scf_max_iter`, `scf_energy_tol`.
The native dictionaries from `method_catalog()` retain legacy gradient labels;
use the production capability tables above before requesting forces.

| Registered method | Family | Method-specific keys | Common adapter |
| --- | --- | --- | --- |
| `RHF` | Hf | SCF keys only | Connected; see guards above |
| `UHF` | Hf | SCF keys only | Connected; see guards above |
| `ROHF` | Hf | SCF keys only | Connected; see guards above |
| `QRHF` | Hf | SCF keys only | Energy / observables |
| `MOM` | Hf | SCF keys, `occupied_orbitals` | Energy / observables |
| `Fermi-HF` | Hf | SCF keys, `thermal_energy` | Energy / observables |
| `GHF` | Hf | SCF keys only | Connected; see guards above |
| `MP2` | Perturbation | `scs_opposite_spin`, `scs_same_spin` | Connected; see guards above |
| `SCS-MP2` | Perturbation | `scs_opposite_spin`, `scs_same_spin` | Connected; see guards above |
| `RI-MP2` | Perturbation | Common keys only | Connected; see guards above |
| `MP3` | Perturbation | `scs_opposite_spin`, `scs_same_spin` | Connected; see guards above |
| `SCS-MP3` | Perturbation | `scs_opposite_spin`, `scs_same_spin`, `scs_third_order` | Connected; see guards above |
| `MP4` | Perturbation | Common keys only | Connected; see guards above |
| `MP5` | Perturbation | Common keys only | Connected; see guards above |
| `MP6` | Perturbation | Common keys only | Connected; see guards above |
| `MP7` | Perturbation | Common keys only | Connected; see guards above |
| `MP8` | Perturbation | Common keys only | Connected; see guards above |
| `CCD` | CoupledCluster | Common keys only | Connected; see guards above |
| `CCSD` | CoupledCluster | Common keys only | Connected; see guards above |
| `CCSD(T)` | CoupledCluster | Common keys only | Connected; see guards above |
| `CIS` | Ci | `cis_triplet` | Connected; see guards above |
| `CISD` | Ci | Common keys only | Connected; see guards above |
| `FCI` | Ci | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `max_orbital_iterations`, `max_rotation`, `orbital_gradient_tol`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `CASSCF` | Multireference | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `max_orbital_iterations`, `max_rotation`, `orbital_gradient_tol`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `CASPT2` | Multireference | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `caspt2_metric_tol`, `imaginary_shift`, `ipea_shift`, `max_orbital_iterations`, `max_rotation`, `orbital_gradient_tol`, `real_shift`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `MRCISD` | Multireference | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `max_orbital_iterations`, `max_rotation`, `optimize_orbitals`, `orbital_gradient_tol`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `MRCISD+Q` | Multireference | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `max_orbital_iterations`, `max_rotation`, `optimize_orbitals`, `orbital_gradient_tol`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `DLPNO-MP2` | Local | `full_domains`, `iterative_triples`, `t_cut_do`, `t_cut_mkn`, `t_cut_pairs`, `t_cut_pno`, `t_cut_tno`, `weak_pairs` | Connected; see guards above |
| `DLPNO-CCSD` | Local | `full_domains`, `iterative_triples`, `t_cut_do`, `t_cut_mkn`, `t_cut_pairs`, `t_cut_pno`, `t_cut_tno`, `weak_pairs` | Connected; see guards above |
| `DLPNO-CCSD(T0)` | Local | `full_domains`, `iterative_triples`, `t_cut_do`, `t_cut_mkn`, `t_cut_pairs`, `t_cut_pno`, `t_cut_tno`, `weak_pairs` | Connected; see guards above |
| `DLPNO-CCSD(T1)` | Local | `full_domains`, `iterative_triples`, `t_cut_do`, `t_cut_mkn`, `t_cut_pairs`, `t_cut_pno`, `t_cut_tno`, `weak_pairs` | Connected; see guards above |
| `MP2-F12` | F12 | `beta`, `cabs_basis`, `cabs_core_singles`, `cabs_singles`, `f12_approximation`, `f12_exact_obs_coulomb`, `f12_fit_orbital_mp2`, `f12_fitting_basis`, `f12_fitting_basis_kind`, `f12_fock_fitting_basis`, `f12_fock_fitting_basis_kind`, `f12_hybrid`, `f12_workspace_bytes`, `triples_scaling` | Connected; see guards above |
| `CCSD-F12a` | F12 | `beta`, `cabs_basis`, `cabs_core_singles`, `cabs_singles`, `f12_approximation`, `f12_exact_obs_coulomb`, `f12_fit_orbital_mp2`, `f12_fitting_basis`, `f12_fitting_basis_kind`, `f12_fock_fitting_basis`, `f12_fock_fitting_basis_kind`, `f12_hybrid`, `f12_workspace_bytes`, `triples_scaling` | Connected; see guards above |
| `CCSD-F12b` | F12 | `beta`, `cabs_basis`, `cabs_core_singles`, `cabs_singles`, `f12_approximation`, `f12_exact_obs_coulomb`, `f12_fit_orbital_mp2`, `f12_fitting_basis`, `f12_fitting_basis_kind`, `f12_fock_fitting_basis`, `f12_fock_fitting_basis_kind`, `f12_hybrid`, `f12_workspace_bytes`, `triples_scaling` | Connected; see guards above |
| `CCSD(T*)-F12a` | F12 | `beta`, `cabs_basis`, `cabs_core_singles`, `cabs_singles`, `f12_approximation`, `f12_exact_obs_coulomb`, `f12_fit_orbital_mp2`, `f12_fitting_basis`, `f12_fitting_basis_kind`, `f12_fock_fitting_basis`, `f12_fock_fitting_basis_kind`, `f12_hybrid`, `f12_workspace_bytes`, `triples_scaling` | Connected; see guards above |
| `CCSD(T*)-F12b` | F12 | `beta`, `cabs_basis`, `cabs_core_singles`, `cabs_singles`, `f12_approximation`, `f12_exact_obs_coulomb`, `f12_fit_orbital_mp2`, `f12_fitting_basis`, `f12_fitting_basis_kind`, `f12_fock_fitting_basis`, `f12_fock_fitting_basis_kind`, `f12_hybrid`, `f12_workspace_bytes`, `triples_scaling` | Connected; see guards above |
| `ACI` | Sci | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `ci_max_iter`, `coarse_graining`, `compute_pt2`, `max_determinants`, `max_orbital_iterations`, `max_rotation`, `orbital_gradient_tol`, `pt2_threshold`, `selection_threshold`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `ASCI` | Sci | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `ci_max_iter`, `compute_pt2`, `max_determinants`, `max_orbital_iterations`, `max_rotation`, `n_core_determinants`, `n_target_determinants`, `orbital_gradient_tol`, `pt2_threshold`, `selection_threshold`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `HBCI` | Sci | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `ci_max_iter`, `compute_pt2`, `max_determinants`, `max_orbital_iterations`, `max_rotation`, `orbital_gradient_tol`, `pt2_threshold`, `selection_threshold`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `PCI` | Sci | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `ci_max_iter`, `compute_pt2`, `max_determinants`, `polynomial_order`, `projector_generator`, `selection_threshold` | Connected; see guards above |
| `GASCI` | Multireference | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `max_orbital_iterations`, `max_rotation`, `orbital_gradient_tol`, `state_weights`, `target_s_squared` | Unconnected; typed Rust only |
| `GASSCF` | Multireference | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `max_orbital_iterations`, `max_rotation`, `orbital_gradient_tol`, `state_weights`, `target_s_squared` | Unconnected; typed Rust only |
| `DOCI` | Ci | Common keys only | Connected; see guards above |
| `DMRG` | Dmrg | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `bond_dimensions`, `max_orbital_iterations`, `max_rotation`, `noise`, `one_site_sweeps`, `orbital_gradient_tol`, `state_weights`, `sweeps`, `target_s_squared`, `truncation_tol` | Connected; see guards above |
| `DMRGSCF` | Dmrg | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `bond_dimensions`, `max_orbital_iterations`, `max_rotation`, `noise`, `one_site_sweeps`, `orbital_gradient_tol`, `state_weights`, `sweeps`, `target_s_squared`, `truncation_tol` | Connected; see guards above |
| `AVAS` | Multireference | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `max_orbital_iterations`, `max_rotation`, `orbital_gradient_tol`, `state_weights`, `target_s_squared` | Unconnected; typed Rust only |
| `ASET` | Multireference | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `max_orbital_iterations`, `max_rotation`, `orbital_gradient_tol`, `state_weights`, `target_s_squared` | Unconnected; typed Rust only |
| `SR-DSRG-PT2` | Dsrg | `flow_parameter` | Connected; see guards above |
| `SR-DSRG-PT3` | Dsrg | `flow_parameter` | Connected; see guards above |
| `SR-LDSRG(2)` | Dsrg | `bch_max_order`, `bch_tolerance`, `flow_parameter` | Connected; see guards above |
| `IM-SRG(2)` | Dsrg | `flow_parameter`, `ode_generator`, `ode_step`, `ode_tolerance` | Connected; see guards above |
| `DSRG-MRPT2` | Dsrg | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `bch_max_order`, `bch_tolerance`, `flow_parameter`, `max_orbital_iterations`, `max_reference_iterations`, `max_rotation`, `optimize_orbitals`, `orbital_gradient_tol`, `reference_ci_tol`, `reference_energy_tol`, `reference_relaxation`, `relax_reference`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `THREE-DSRG-MRPT2` | Dsrg | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `cholesky_tolerance`, `flow_parameter`, `max_orbital_iterations`, `max_reference_iterations`, `max_rotation`, `optimize_orbitals`, `orbital_gradient_tol`, `reference_ci_tol`, `reference_energy_tol`, `reference_relaxation`, `relax_reference`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `DSRG-MRPT3` | Dsrg | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `bch_max_order`, `bch_tolerance`, `flow_parameter`, `max_orbital_iterations`, `max_reference_iterations`, `max_rotation`, `optimize_orbitals`, `orbital_gradient_tol`, `reference_ci_tol`, `reference_energy_tol`, `reference_relaxation`, `relax_reference`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `MR-LDSRG(2)` | Dsrg | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `bch_max_order`, `bch_tolerance`, `flow_parameter`, `max_orbital_iterations`, `max_reference_iterations`, `max_rotation`, `optimize_orbitals`, `orbital_gradient_tol`, `reference_ci_tol`, `reference_energy_tol`, `reference_relaxation`, `relax_reference`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `MR-LDSRG(2)+QC` | Dsrg | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `flow_parameter`, `max_orbital_iterations`, `max_reference_iterations`, `max_rotation`, `optimize_orbitals`, `orbital_gradient_tol`, `reference_ci_tol`, `reference_energy_tol`, `reference_relaxation`, `relax_reference`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `SA-DSRG-PT2` | Dsrg | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `cholesky_tolerance`, `flow_parameter`, `max_orbital_iterations`, `max_rotation`, `optimize_orbitals`, `orbital_gradient_tol`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `SA-DSRG-PT2c` | Dsrg | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `cholesky_tolerance`, `flow_parameter`, `max_orbital_iterations`, `max_rotation`, `optimize_orbitals`, `orbital_gradient_tol`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `MS-DSRG-PT2` | Dsrg | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `cholesky_tolerance`, `flow_parameter`, `max_orbital_iterations`, `max_rotation`, `optimize_orbitals`, `orbital_gradient_tol`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `XMS-DSRG-PT2` | Dsrg | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `cholesky_tolerance`, `flow_parameter`, `max_orbital_iterations`, `max_rotation`, `optimize_orbitals`, `orbital_gradient_tol`, `state_weights`, `target_s_squared` | Connected; see guards above |
| `DWMS-DSRG-PT2` | Dsrg | `active_n_alpha`, `active_n_beta`, `active_n_orb`, `active_orbitals`, `cholesky_tolerance`, `dynamic_weight_parameter`, `flow_parameter`, `max_orbital_iterations`, `max_rotation`, `optimize_orbitals`, `orbital_gradient_tol`, `state_weights`, `target_s_squared` | Connected; see guards above |

#### Option values and index conventions

- `root`, `active_orbitals` and frozen-space counts use zero-based MO indexing / counts as stated in the recipes.
- `active_n_alpha`, `active_n_beta`, `active_n_orb` are integer counts; specify all three together.
- `state_weights`, `bond_dimensions` and `coupling` are arrays.
- Boolean controls are Boolean values, not strings.
- `reference_relaxation`: `none`, `once`, `twice`, `iterate`.
- `f12_approximation`: `3C(FIX)` or `3*A(FIX)`; `f12_hybrid`: `full`, `HY1`, `HY2`.
- `f12_fitting_basis_kind` and `f12_fock_fitting_basis_kind`: `cartesian` or `spherical`.
- `noise=0` is the supported DMRG value.
- Complete common defaults/types/units are in the option tables below. Typed HF defaults and advanced controls are in the [Rust option reference](rust_api.md#typed-hf-option-reference).

## Executed HF property examples

The following is numerical output from the public native Python API, using
STO-3G and default common-model controls unless the runnable input below
sets an option. Coordinates are bohr. Energies are Hartree; observable units
are explicit. These are small API/property demonstrations, not basis-limit
predictions or independent accuracy benchmarks. In particular, the NEO example
uses the electronic hydrogen basis for its proton; the positronic example
does not establish a bound positron state. The radial result uses a finite
4096-point grid, rather than a Gaussian STO-3G calculation.

| Method | Geometry | Total energy / Hartree | Observable | Measured value | Unit |
| --- | --- | --- | --- | --- | --- |
| `hf` | `water` | -74.960337093218 | `homo_lumo_gap` | 1.01523999 | hartree |
| `uhf` | `hydrogen_atom` | -0.466581850378 | `s_squared` | 0.75 | 1 |
| `rohf` | `lithium_atom` | -7.315526005555 | `s_squared` | 0.75 | 1 |
| `qrhf` | `lithium_atom` | -7.315526005555 | `s_squared` | 0.75 | 1 |
| `mom` | `hydrogen_molecule` | 0.460576446524 | `homo_lumo_gap` | -0.328820034 | hartree |
| `fermi-hf` | `stretched_hydrogen` | -0.816324207030 | `entropy_over_kb` | 0.00109118603 | 1 |
| `ghf` | `stretched_hydrogen` | -0.816344159870 | `s_squared` | -4.4408921e-16 | 1 |
| `puhf` | `stretched_hydrogen` | -0.957448929407 | `target_spin_weight` | 0.545254507 | 1 |
| `hfb` | `stretched_hydrogen` | -0.866344159870 | `pairing_gap` | 0 | hartree |
| `tcscf` | `stretched_hydrogen` | -0.957675170838 | `natural_occupations` | [1.51859445, 0.48140555] | 1 |
| `qed-hf` | `water` | -74.955962547860 | `mean_photon_number` | 0.00596236843 | 1 |
| `df-hf` | `water` | -74.960479454564 | `n_auxiliary` | 85 | 1 |
| `cd-hf` | `water` | -74.960323592031 | `cholesky_rank` | 24 | 1 |
| `x2c` | `neon_atom` | -126.716941516115 | `relativistic_shift` | -0.112416427 | hartree |
| `dkh2` | `neon_atom` | -126.716941509940 | `relativistic_shift` | -0.112416421 | hartree |
| `complex-hf` | `water` | -74.959829516093 | `imaginary_density_norm` | 0.0372425819 | 1 |
| `cuhf` | `water` | -74.959829516093 | `imaginary_density_norm` | 0.0372425819 | 1 |
| `cghf` | `water` | -74.959829516093 | `s_squared` | 3.55271368e-15 | 1 |
| `lhf` | `water` | -74.959872019704 | `homo_lumo_gap` | 0.420987927 | hartree |
| `oep` | `water` | -74.960337093218 | `minimum_gap` | 0.152248764 | hartree |
| `neo-hf` | `hydrogen_molecule` | -0.764842068860 | `protonic_kinetic_energy` | 0.000413926299 | hartree |
| `positronic-hf` | `hydrogen_molecule` | -0.012025936835 | `positron_orbital_energy` | 1.10468839 | hartree |
| `dc-hf` | `water` | -74.960337093218 | `chemical_potential` | 0.451446662 | hartree |
| `fmo2` | `water_dimer` | -149.924728954988 | `pair_interaction_energies` | [-0.00411055698] | hartree |
| `fmo3` | `helium_trimer` | -8.423183517483 | `trimer_corrections` | [-5.20921413e-07] | hartree |
| `tdhf` | `water` | -74.459405664852 | `oscillator_strengths` | [0.00333485512, 2.24052268e-31, 0] | 1 |
| `sahf` | `hydrogen_molecule` | -0.643527093666 | `state_energies` | [-1.11671433, 0.460576447] | hartree |
| `sigma-scf` | `hydrogen_molecule` | -1.116714325176 | `intrinsic_variance` | 0.0328544314 | hartree^2 |
| `crystal-hf` | `hydrogen_molecule` | -1.101487464565 | `band_gap` | 0.820145163 | hartree |
| `radial-hf` | `helium_atom` | -2.861644725455 | `virial_error` | -7.05386324e-05 | hartree |

MOM follows the requested excited determinant; its occupied/unoccupied gap
can be negative. The default HFB model has zero anomalous gap in this
example; its additional normal pairing-interaction energy is still present.
The tiny negative GHF S-squared is floating-point roundoff about zero.
TDHF `e_total` is the selected vertical excited-state total energy.
FMO2 uses two water fragments; FMO3 uses three neutral helium fragments
and has a nonzero connected trimer correction. Each entry also successfully
exports its retained intermediate tensors.

```python
import hf_rs

geometries = {
    'water': ([8, 1, 1], [[0.0, 0.0, 0.0], [0.0, -1.4, 1.1], [0.0, 1.4, 1.1]], {}),
    'hydrogen_atom': ([1], [[0.0, 0.0, 0.0]], {'multiplicity': 2, 'reference': 'uhf'}),
    'lithium_atom': ([3], [[0.0, 0.0, 0.0]], {'multiplicity': 2}),
    'hydrogen_molecule': ([1, 1], [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], {}),
    'stretched_hydrogen': ([1, 1], [[0.0, 0.0, -1.75], [0.0, 0.0, 1.75]], {}),
    'neon_atom': ([10], [[0.0, 0.0, 0.0]], {}),
    'water_dimer': ([8, 1, 1, 8, 1, 1], [[0.0, 0.0, 0.0], [0.0, -1.4, 1.1], [0.0, 1.4, 1.1], [0.0, 0.0, 6.0], [0.0, -1.4, 7.1], [0.0, 1.4, 7.1]], {}),
    'helium_trimer': ([2, 2, 2], [[0.0, 0.0, 0.0], [0.3, 0.2, 5.0], [4.5, -0.4, 1.5]], {}),
    'helium_atom': ([2], [[0.0, 0.0, 0.0]], {}),
}
cases = [
    ('hf', 'water', {}, 'homo_lumo_gap'),
    ('uhf', 'hydrogen_atom', {}, 's_squared'),
    ('rohf', 'lithium_atom', {}, 's_squared'),
    ('qrhf', 'lithium_atom', {}, 's_squared'),
    ('mom', 'hydrogen_molecule', {'occupied_orbitals': [1]}, 'homo_lumo_gap'),
    ('fermi-hf', 'stretched_hydrogen', {'thermal_energy': 0.02}, 'entropy_over_kb'),
    ('ghf', 'stretched_hydrogen', {}, 's_squared'),
    ('puhf', 'stretched_hydrogen', {}, 'target_spin_weight'),
    ('hfb', 'stretched_hydrogen', {}, 'pairing_gap'),
    ('tcscf', 'stretched_hydrogen', {}, 'natural_occupations'),
    ('qed-hf', 'water', {}, 'mean_photon_number'),
    ('df-hf', 'water', {}, 'n_auxiliary'),
    ('cd-hf', 'water', {}, 'cholesky_rank'),
    ('x2c', 'neon_atom', {}, 'relativistic_shift'),
    ('dkh2', 'neon_atom', {}, 'relativistic_shift'),
    ('complex-hf', 'water', {}, 'imaginary_density_norm'),
    ('cuhf', 'water', {}, 'imaginary_density_norm'),
    ('cghf', 'water', {}, 's_squared'),
    ('lhf', 'water', {}, 'homo_lumo_gap'),
    ('oep', 'water', {}, 'minimum_gap'),
    ('neo-hf', 'hydrogen_molecule', {}, 'protonic_kinetic_energy'),
    ('positronic-hf', 'hydrogen_molecule', {}, 'positron_orbital_energy'),
    ('dc-hf', 'water', {}, 'chemical_potential'),
    ('fmo2', 'water_dimer', {}, 'pair_interaction_energies'),
    ('fmo3', 'helium_trimer', {}, 'trimer_corrections'),
    ('tdhf', 'water', {'n_singlets': 2, 'n_triplets': 1}, 'oscillator_strengths'),
    ('sahf', 'hydrogen_molecule', {'configurations': [{'alpha': [0], 'beta': [0]}, {'alpha': [1], 'beta': [1]}], 'weights': [0.7, 0.3]}, 'state_energies'),
    ('sigma-scf', 'hydrogen_molecule', {'configuration': {'alpha': [0], 'beta': [0]}, 'target_energy': -1.1167143251757694}, 'intrinsic_variance'),
    ('crystal-hf', 'hydrogen_molecule', {'lattice_constant': 4.0, 'n_kpoints': 16}, 'band_gap'),
    ('radial-hf', 'helium_atom', {}, 'virial_error'),
]
for method, geometry, options, observable in cases:
    numbers, positions, configuration = geometries[geometry]
    m = hf_rs.Molecule(numbers, positions, **configuration)
    result = m.run(method, options=options, properties=["observables", "intermediates"])
    print(method, result["e_total"], result["observables"][observable])
hf_cases = cases
```

RHF/UHF use the configured standard reference; the remaining names use
their actual model solvers. `zora` is a recognized unavailable Hamiltonian
and returns an error. Advanced model changes use the
[typed Rust option tables](rust_api.md#typed-hf-option-reference).

### Equivalent Rust and ASE model selection

Use the identical method string and option keys with
`Molecule::run(name, &options, &["observables".into()])` in Rust, or
`HFCalculator(method=name, **configuration, options=...)` and
`get_observables(atoms)` in ASE. Rust/native Python input options and
outputs use atomic units; ASE uses eV/angstrom-based physical values.
Convert the geometry from bohr to angstrom before creating ASE atoms.
The [ASE unit table](ase.md#units-and-observables) covers physical options.

## Executed post-HF API examples

All 50 connected post-HF names below execute through the public Python
API at H2/STO-3G, R=1.4 bohr. The complete two-orbital active space is a
small limit check: some multireference external-space corrections vanish.
This confirms API behavior, root/RDM output and supported input routes;
it does not replace the independent method-specific validation or close
the experimental F12 literature discrepancy.

| Method | Explicit options (all other controls default) | Total energy / Hartree | States | Retained intermediate keys |
| --- | --- | --- | --- | --- |
| `MP2` | `{}` | -1.129872195222 | 1 | None retained |
| `SCS-MP2` | `{}` | -1.132503769231 | 1 | None retained |
| `RI-MP2` | `{"aux_basis":"def2-svp-ri"}` | -1.129865932249 | 1 | None retained |
| `MP3` | `{}` | -1.134718381872 | 1 | None retained |
| `SCS-MP3` | `{}` | -1.133715315894 | 1 | None retained |
| `MP4` | `{}` | -1.136433948479 | 1 | None retained |
| `MP5` | `{}` | -1.137014735440 | 1 | None retained |
| `MP6` | `{}` | -1.137201159195 | 1 | None retained |
| `MP7` | `{}` | -1.137257040756 | 1 | None retained |
| `MP8` | `{}` | -1.137272224638 | 1 | None retained |
| `CCD` | `{}` | -1.137275943783 | 1 | `method.t2` |
| `CCSD` | `{}` | -1.137275943783 | 1 | `method.t1`, `method.t2` |
| `CCSD(T)` | `{}` | -1.137275943783 | 1 | None retained |
| `CIS` | `{}` | -0.169291749578 | 1 | None retained |
| `CISD` | `{}` | -1.137275943783 | 1 | `method.one_rdm_mo` |
| `FCI` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | `method.one_rdm_mo` |
| `CASSCF` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | `method.one_rdm_mo` |
| `CASPT2` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | None retained |
| `MRCISD` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | `method.one_rdm_mo` |
| `MRCISD+Q` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | `method.one_rdm_mo` |
| `DLPNO-MP2` | `{"aux_basis":"def2-svp-ri","full_domains":true,"t_cut_pairs":0.0,"t_cut_pno":0.0}` | -1.129865932249 | 1 | None retained |
| `DLPNO-CCSD` | `{"aux_basis":"def2-svp-ri","full_domains":true,"t_cut_pairs":0.0,"t_cut_pno":0.0}` | -1.137268765396 | 1 | None retained |
| `DLPNO-CCSD(T0)` | `{"aux_basis":"def2-svp-ri","full_domains":true,"t_cut_pairs":0.0,"t_cut_pno":0.0}` | -1.137268765396 | 1 | None retained |
| `DLPNO-CCSD(T1)` | `{"aux_basis":"def2-svp-ri","full_domains":true,"t_cut_pairs":0.0,"t_cut_pno":0.0}` | -1.137268765396 | 1 | None retained |
| `MP2-F12` | `{"cabs_basis":"cc-pvdz-f12-optri","beta":1.0}` | -1.158480459462 | 1 | None retained |
| `CCSD-F12a` | `{"cabs_basis":"cc-pvdz-f12-optri","beta":1.0}` | -1.163733903426 | 1 | None retained |
| `CCSD-F12b` | `{"cabs_basis":"cc-pvdz-f12-optri","beta":1.0}` | -1.162269862437 | 1 | None retained |
| `CCSD(T*)-F12a` | `{"cabs_basis":"cc-pvdz-f12-optri","beta":1.0}` | -1.163733903426 | 1 | None retained |
| `CCSD(T*)-F12b` | `{"cabs_basis":"cc-pvdz-f12-optri","beta":1.0}` | -1.162269862437 | 1 | None retained |
| `ACI` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | `method.one_rdm_mo` |
| `ASCI` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | `method.one_rdm_mo` |
| `HBCI` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | `method.one_rdm_mo` |
| `PCI` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | `method.one_rdm_mo` |
| `DOCI` | `{}` | -1.137275943783 | 1 | `method.one_rdm_mo` |
| `DMRG` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2,"bond_dimensions":[8],"sweeps":4}` | -1.137275943783 | 1 | `method.one_rdm_mo` |
| `DMRGSCF` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2,"bond_dimensions":[8],"sweeps":4}` | -1.137275943783 | 1 | `method.one_rdm_mo` |
| `SR-DSRG-PT2` | `{}` | -1.129846403370 | 1 | `method.one_rdm_mo` |
| `SR-DSRG-PT3` | `{}` | -1.134664932916 | 1 | None retained |
| `SR-LDSRG(2)` | `{}` | -1.137723390751 | 1 | None retained |
| `IM-SRG(2)` | `{}` | -1.135898397471 | 1 | None retained |
| `DSRG-MRPT2` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | None retained |
| `THREE-DSRG-MRPT2` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | None retained |
| `DSRG-MRPT3` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | None retained |
| `MR-LDSRG(2)` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | None retained |
| `MR-LDSRG(2)+QC` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2}` | -1.137275943783 | 1 | None retained |
| `SA-DSRG-PT2` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2,"n_roots":2}` | -1.137275943783 | 2 | None retained |
| `SA-DSRG-PT2c` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2,"n_roots":2}` | -1.137275943783 | 2 | None retained |
| `MS-DSRG-PT2` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2,"n_roots":2}` | -1.137275943783 | 2 | None retained |
| `XMS-DSRG-PT2` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2,"n_roots":2}` | -1.137275943783 | 2 | None retained |
| `DWMS-DSRG-PT2` | `{"active_n_alpha":1,"active_n_beta":1,"active_n_orb":2,"n_roots":2}` | -1.137275943783 | 2 | None retained |

The following reproduces every row. `GASCI`, `GASSCF`, `AVAS` and `ASET`
have registered schemas but require their typed Rust constructors; their
common API calls return an explicit unavailable-adapter error.

```python
import hf_rs

m = hf_rs.Molecule([1,1], [[0.,0.,-0.7],[0.,0.,0.7]])
cases = [
    ('MP2', {}),
    ('SCS-MP2', {}),
    ('RI-MP2', {'aux_basis': 'def2-svp-ri'}),
    ('MP3', {}),
    ('SCS-MP3', {}),
    ('MP4', {}),
    ('MP5', {}),
    ('MP6', {}),
    ('MP7', {}),
    ('MP8', {}),
    ('CCD', {}),
    ('CCSD', {}),
    ('CCSD(T)', {}),
    ('CIS', {}),
    ('CISD', {}),
    ('FCI', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('CASSCF', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('CASPT2', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('MRCISD', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('MRCISD+Q', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('DLPNO-MP2', {'aux_basis': 'def2-svp-ri', 'full_domains': True, 't_cut_pairs': 0.0, 't_cut_pno': 0.0}),
    ('DLPNO-CCSD', {'aux_basis': 'def2-svp-ri', 'full_domains': True, 't_cut_pairs': 0.0, 't_cut_pno': 0.0}),
    ('DLPNO-CCSD(T0)', {'aux_basis': 'def2-svp-ri', 'full_domains': True, 't_cut_pairs': 0.0, 't_cut_pno': 0.0}),
    ('DLPNO-CCSD(T1)', {'aux_basis': 'def2-svp-ri', 'full_domains': True, 't_cut_pairs': 0.0, 't_cut_pno': 0.0}),
    ('MP2-F12', {'cabs_basis': 'cc-pvdz-f12-optri', 'beta': 1.0}),
    ('CCSD-F12a', {'cabs_basis': 'cc-pvdz-f12-optri', 'beta': 1.0}),
    ('CCSD-F12b', {'cabs_basis': 'cc-pvdz-f12-optri', 'beta': 1.0}),
    ('CCSD(T*)-F12a', {'cabs_basis': 'cc-pvdz-f12-optri', 'beta': 1.0}),
    ('CCSD(T*)-F12b', {'cabs_basis': 'cc-pvdz-f12-optri', 'beta': 1.0}),
    ('ACI', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('ASCI', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('HBCI', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('PCI', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('DOCI', {}),
    ('DMRG', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2, 'bond_dimensions': [8], 'sweeps': 4}),
    ('DMRGSCF', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2, 'bond_dimensions': [8], 'sweeps': 4}),
    ('SR-DSRG-PT2', {}),
    ('SR-DSRG-PT3', {}),
    ('SR-LDSRG(2)', {}),
    ('IM-SRG(2)', {}),
    ('DSRG-MRPT2', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('THREE-DSRG-MRPT2', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('DSRG-MRPT3', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('MR-LDSRG(2)', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('MR-LDSRG(2)+QC', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2}),
    ('SA-DSRG-PT2', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2, 'n_roots': 2}),
    ('SA-DSRG-PT2c', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2, 'n_roots': 2}),
    ('MS-DSRG-PT2', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2, 'n_roots': 2}),
    ('XMS-DSRG-PT2', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2, 'n_roots': 2}),
    ('DWMS-DSRG-PT2', {'active_n_alpha': 1, 'active_n_beta': 1, 'active_n_orb': 2, 'n_roots': 2}),
]
for method, options in cases:
    result = m.run(method, options=options, properties=["intermediates"])
    print(method, result["e_total"], result["root"], len(result["states"]))
post_hf_cases = cases
```

### Run all model examples through ASE

Reuse the `geometries`, `hf_cases` and `post_hf_cases` input lists above.
Every one of these 80 ASE calculations has been executed. Energies agree
with the native examples within 1e-7 eV after conversion. Model observables
also agree after applying their energy/length powers at the identical
angstrom geometry. Physical options in these particular input lists are
converted explicitly below; omitted controls retain their native defaults.

```python
import numpy as np
from ase import Atoms, units
from hf_rs import HFCalculator

def ase_options(native):
    options = dict(native)
    for key, factor in {
        "thermal_energy": units.Hartree,
        "target_energy": units.Hartree,
        "lattice_constant": units.Bohr,
        "beta": 1. / units.Bohr,
    }.items():
        if key in options:
            options[key] *= factor
    return options

for method, geometry, options, observable in hf_cases:
    numbers, positions, configuration = geometries[geometry]
    atoms = Atoms(numbers=numbers, positions=np.asarray(positions)*units.Bohr)
    atoms.calc = HFCalculator(method=method, options=ase_options(options), **configuration)
    quantities = atoms.calc.get_observables(atoms)
    print(method, atoms.calc.calculation_result["e_total"], quantities[observable])

for method, options in post_hf_cases:
    atoms = Atoms("H2", positions=np.asarray([[0.,0.,-0.7],[0.,0.,0.7]])*units.Bohr)
    atoms.calc = HFCalculator(method=method, options=ase_options(options))
    print(method, atoms.get_potential_energy())  # eV
```

In an insulating DC-HF calculation the electron count can be numerically
constant over part of the gap. Tiny roundoff or geometry-unit round trips
can select different chemical potentials on that plateau without changing
the converged electron count, density or energy. Check these quantities as
well as the reported chemical potential when comparing executions.

### Run every connected post-HF model through Rust

This executes the same H2 limit cases using typed registry names and an
`OptionMap`. The separately listed GAS/AVAS/ASET typed constructors require
additional model inputs and are deliberately excluded from this common-API loop.

```rust
use hf_rs::api::{Molecule, MoleculeOptions, OptionMap};
use hf_rs::registry::{MethodFamily, MethodId};
use serde_json::json;

let m = Molecule::new(&[1,1], &[[0.,0.,-0.7],[0.,0.,0.7]], MoleculeOptions::default())?;
let mut count = 0;
for &method in MethodId::ALL {
    if method.family() == MethodFamily::Hf || matches!(method,
        MethodId::Gasci | MethodId::Gasscf | MethodId::Avas | MethodId::Aset) {
        continue;
    }
    let mut options = OptionMap::new();
    if method.option_schemas().iter().any(|keys| keys.contains(&"active_n_orb")) {
        for key in ["active_n_alpha", "active_n_beta"] { options.insert(key.into(), json!(1)); }
        options.insert("active_n_orb".into(), json!(2));
    }
    if method.family() == MethodFamily::Local || method == MethodId::RiMp2 {
        options.insert("aux_basis".into(), json!("def2-svp-ri"));
    }
    if method.family() == MethodFamily::Local {
        options.insert("full_domains".into(), json!(true));
        for key in ["t_cut_pairs", "t_cut_pno"] { options.insert(key.into(), json!(0.)); }
    }
    if method.family() == MethodFamily::F12 {
        options.insert("cabs_basis".into(), json!("cc-pvdz-f12-optri"));
        options.insert("beta".into(), json!(1.));
    }
    if method.family() == MethodFamily::Dmrg {
        options.insert("bond_dimensions".into(), json!([8]));
        options.insert("sweeps".into(), json!(4));
    }
    if ["SA-DSRG", "MS-DSRG", "XMS-DSRG", "DWMS-DSRG"].iter()
        .any(|prefix| method.name().starts_with(prefix)) {
        options.insert("n_roots".into(), json!(2));
    }
    let result = m.run(method.name(), &options, &["intermediates".into()])?;
    println!("{} {}", method, result.output.e_total); // Hartree
    count += 1;
}
assert_eq!(count, 50);
# Ok::<(), hf_rs::HfError>(())
```

The same names/options work with Rust `Molecule::run` and ASE
`HFCalculator`. Use `get_potential_energy()` for energy-only post-HF
methods; asking ASE for forces does not enable unavailable analytic
response. Active-space/optimized RDMs retain their method-specific MO
frame. [Detailed family recipes](#calculation-recipes) explain scientifically
useful active spaces, frozen spaces, multiple roots, UHF references and
the limitations hidden by this very small complete-space example.

## Complete option defaults and input types

This table is generated from `hf_rs.option_catalog()`. Defaults read the
actual typed `Default` implementations; each registered method lists its
accepted subset in the [schema index](#registered-method-options).
`omit` means leave out the key; it is not a valid explicit null value.
An empty list is distinct from omission. A typed/default option is still
subject to each adapter's reference, spin, fitting and state guards.
Solver-residual/selection thresholds use the stated method-native
definition; they are not universal energy observables.

| Key | Type | Default | Native unit | Meaning / constraints |
| --- | --- | --- | --- | --- |
| `active_n_alpha` | integer | `omit` | 1 | Active alpha-electron count; supply together with active_n_beta and active_n_orb. |
| `active_n_beta` | integer | `omit` | 1 | Active beta-electron count; inactive orbitals must be doubly occupied. |
| `active_n_orb` | integer | `omit` | 1 | Active spatial-orbital count; positive and large enough for both spin populations. |
| `active_orbitals` | integer list | `omit` | 1 | Distinct zero-based original MO indices; omission selects consecutive active orbitals after inactive occupied orbitals. |
| `aux_basis` | string | `omit` | 1 | Name of the physical auxiliary Coulomb-fitting basis; required by fitted adapters. |
| `bch_max_order` | integer | `80` | 1 | Positive maximum recursive BCH order; exhaustion is an error. |
| `bch_tolerance` | real | `1e-13` | hartree | Positive operator-norm cutoff for the converged recursive BCH series. |
| `beta` | real | `1.0` | bohr^-1 | Positive Slater-geminal exponent; physical CABS/fitting spaces must match the chosen model. |
| `bond_dimensions` | integer list | `[64,128,256]` | 1 | Positive MPS bond-dimension schedule, extended using its last value. |
| `broken_symmetry` | boolean | `false` | 1 | Request an unrestricted broken-symmetry initial determinant; requires reference=uhf. |
| `cabs_basis` | string | `omit` | 1 | Physical complementary auxiliary basis name; required by the F12 molecular adapter. |
| `cabs_core_singles` | boolean | `true` | 1 | Include frozen occupied orbitals in CABS singles. |
| `cabs_singles` | boolean | `true` | 1 | Include the implemented CABS-singles correction. |
| `caspt2_metric_tol` | real | `1e-10` | 1 | Positive relative overlap cutoff in the internally contracted CASPT2 space. |
| `cholesky_tolerance` | real | `1e-10` | hartree | Positive maximum residual Coulomb diagonal for factorized THREE kernels. |
| `ci_max_iter` | integer | `100` | 1 | Positive inner variational Davidson iteration limit, separate from outer selection/propagation max_iter. |
| `cis_triplet` | boolean | `false` | 1 | Select triplet rather than singlet spin-adapted CIS excitations. |
| `coarse_graining` | real | `0.5` | 1 | Nonnegative ACI pruning factor applied to the numerical Hartree selection threshold; the cumulative squared-weight bound must remain below one. |
| `compute_pt2` | boolean | `true` | 1 | Compute the adapter's deterministic selected-space correction; returned RDMs remain variational. |
| `density_fitting` | boolean | `false` | 1 | Request a density-fitted branch where implemented; fitted method names select their own branch. |
| `dynamic_weight_parameter` | real | `50.0` | hartree^-2 | Nonnegative DWMS Gaussian energy-separation weight parameter. |
| `energy_tol` | real | `1e-12` | hartree | Positive post-HF energy-change convergence threshold. |
| `f12_approximation` | string | `"3C(FIX)"` | 1 | Implemented fixed-amplitude F12 approximation; the distinct 3B theory is unsupported. Choices: `3C(FIX)`, `3*A(FIX)`. |
| `f12_exact_obs_coulomb` | boolean | `false` | 1 | Restore exact Coulomb integrals inside the OBS projector; fitted remaining spaces stay fitted. |
| `f12_fit_orbital_mp2` | boolean | `false` | 1 | Use the selected robust fitting space for the orbital MP2 contribution where supported. |
| `f12_fitting_basis` | string | `omit` | 1 | Optional robust-fitting basis for F12 correction kernels. |
| `f12_fitting_basis_kind` | string | `"cartesian"` | 1 | Cartesian or real spherical physical correction-fitting space. Choices: `cartesian`, `spherical`. |
| `f12_fock_fitting_basis` | string | `omit` | 1 | Optional independent Fock fitting basis; omission uses the implemented unfitted/F12 selection rules. |
| `f12_fock_fitting_basis_kind` | string | `"cartesian"` | 1 | Cartesian or real spherical physical Fock-fitting space. Choices: `cartesian`, `spherical`. |
| `f12_hybrid` | string | `"full"` | 1 | Full residual nonlocal resolution, HY1 omission or HY2 CABS-CABS omission. Choices: `full`, `hy1`, `hy2`. |
| `f12_workspace_bytes` | integer | `268435456` | byte | Positive limit on the live mixed-tensor workspace estimate; excludes reference storage and process overhead. |
| `flow_parameter` | real | `0.5` | hartree^-2 | Nonnegative DSRG/IM-SRG flow endpoint. |
| `frozen_core` | integer | `0` | 1 | Number of lowest doubly occupied spatial orbitals excluded from correlation. |
| `frozen_virtual` | integer | `0` | 1 | Number of highest virtual spatial orbitals excluded from correlation. |
| `full_domains` | boolean | `false` | 1 | Use the full molecular domain instead of adaptive local domains. |
| `imaginary_shift` | real | `0.0` | hartree | Nonnegative CASPT2 imaginary level shift. |
| `ipea_shift` | real | `0.25` | hartree | CASPT2 IPEA shift in the documented contracted zeroth-order Hamiltonian. |
| `iterative_triples` | boolean | `false` | 1 | Select the supported iterative local triples correction rather than the T0 approximation. |
| `max_determinants` | integer | `100000` | 1 | Positive determinant-space limit; reaching the limit does not certify convergence. |
| `max_iter` | integer | `100` | 1 | Positive post-HF amplitude, CI selection or flow iteration limit; separate from scf_max_iter. |
| `max_orbital_iterations` | integer | `80` | 1 | Positive CASSCF/DMRGSCF orbital-optimization iteration limit. |
| `max_reference_iterations` | integer | `30` | 1 | Positive iteration limit for iterative reference relaxation. |
| `max_rotation` | real | `0.1` | radian | Positive orbital-rotation trust bound. |
| `n_core_determinants` | integer | `50` | 1 | Positive ASCI core-search determinant count. |
| `n_roots` | integer | `1` | 1 | Positive number of requested states; support depends on the selected method. |
| `n_target_determinants` | integer | `200` | 1 | Positive ASCI target variational determinant count. |
| `noise` | real | `0.0` | method-native noise | Only zero noise is implemented by the current DMRG adapter. |
| `numerical_gradient_step` | real | `0.001` | bohr | Positive displacement for explicitly numerical low-level debug routes; does not enable analytic forces. |
| `occupied_orbitals` | integer list | `omit` | 1 | Required distinct zero-based MOM occupied guess-orbital indices, with the molecular closed-shell electron count. |
| `ode_generator` | string | `"wegner"` | 1 | IM-SRG generator: Wegner commutator or regularized White form. Choices: `wegner`, `white`. |
| `ode_step` | real | `0.01` | hartree^-2 | Positive initial flow-integration step. |
| `ode_tolerance` | real | `1e-09` | method-native residual | Positive adaptive ODE local-error tolerance. |
| `one_site_sweeps` | integer | `12` | 1 | Number of final one-site DMRG sweeps. |
| `optimize_orbitals` | boolean | `true` | 1 | Optimize the multireference orbitals where the adapter implements that option. |
| `orbital_gradient_tol` | real | `1e-07` | hartree | Positive maximum orbital-gradient convergence threshold. |
| `polynomial_order` | integer | `4` | 1 | Positive polynomial order for supported PCI generators. |
| `projector_generator` | string | `"wallchebyshev"` | 1 | PCI polynomial projector or selected-space Davidson/Lanczos accelerator. Choices: `wallchebyshev`, `chebyshev`, `expchebyshev`, `linear`, `quadratic`, `cubic`, `quartic`, `power`, `davidson`, `lanczos`. |
| `pt2_threshold` | real | `1e-08` | hartree | Nonnegative HCI individual coupling screen before summation in the PT2 search. |
| `real_shift` | real | `0.0` | hartree | CASPT2 real level shift; its energy correction is reported separately. |
| `reference_ci_tol` | real | `1e-08` | 1 | Positive relaxed-reference CI-vector change threshold. |
| `reference_energy_tol` | real | `1e-10` | hartree | Positive relaxed-reference energy-change threshold. |
| `reference_relaxation` | string | `"none"` | 1 | Number or convergence policy of effective-CAS reference rediagonalizations. Choices: `none`, `once`, `twice`, `iterate`. |
| `relax_reference` | boolean | `false` | 1 | Compatibility flag selecting one reference rediagonalization unless an explicit relaxation mode is supplied. |
| `residual_tol` | real | `1e-10` | method-native residual | Positive convergence threshold for the selected solver's documented residual. |
| `root` | integer | `0` | 1 | Zero-based selected state; must be smaller than the computed state count. |
| `scf_conv_algo` | string | `omit` | 1 | Omission selects RHF ADIIS/SOSCF or UHF CDIIS; explicit unsupported accelerators are errors. Choices: `adiis_soscf`, `cdiis`, `adiis`, `adiis_cdiis`, `soscf`, `gdm`, `trah`. |
| `scf_density_tol` | real | `1e-10` | 1 | Positive SCF density-change threshold. |
| `scf_energy_tol` | real | `1e-12` | hartree | Positive SCF energy-change threshold. |
| `scf_max_iter` | integer | `100` | 1 | Positive SCF iteration limit. |
| `scf_residual_tol` | real | `1e-10` | hartree | Positive SCF FDS-SDF residual threshold. |
| `scs_opposite_spin` | real | `1.2` | 1 | Finite opposite-spin second-order energy scale. |
| `scs_same_spin` | real | `0.3333333333333333` | 1 | Finite same-spin second-order energy scale. |
| `scs_third_order` | real | `0.25` | 1 | Finite third-order increment scale in SCS-MP3. |
| `selection_threshold` | real | `1e-05` | method-native selection | Nonnegative ACI energy importance, HCI coupling threshold or PCI spawning threshold; definitions differ by method. |
| `state_weights` | real list | `[]` | 1 | Nonnegative state-average weights; an empty list requests the adapter's default equal weighting. |
| `sweeps` | integer | `12` | 1 | Positive number of two-site DMRG sweeps. |
| `t_cut_do` | real | `0.01` | native basis metric | Nonnegative differential-overlap domain threshold in the documented local model. |
| `t_cut_mkn` | real | `0.001` | native basis metric | Nonnegative local fitting-domain threshold. |
| `t_cut_pairs` | real | `1e-05` | hartree | Nonnegative pair-energy threshold separating strong and weak pairs. |
| `t_cut_pno` | real | `1e-07` | 1 | Nonnegative pair-natural-orbital occupation cutoff. |
| `t_cut_tno` | real | `omit` | 1 | Triples-natural-orbital occupation cutoff; omission inherits t_cut_pno. |
| `target_s_squared` | real | `omit` | 1 | Requested S(S+1) for supported exact spin projection; omission uses the adapter's spin convention. |
| `thermal_energy` | real | `omit` | hartree | Required positive Fermi-HF kBT; the legacy typed Rust entry point instead labels its temperature argument in kelvin. |
| `triples_scaling` | boolean | `false` | 1 | Use the selected F12 triples-scaling convention where implemented. |
| `truncation_tol` | real | `1e-10` | 1 | Positive discarded Schmidt-weight threshold. |
| `weak_pairs` | boolean | `true` | 1 | Retain the documented weak-pair correction. |

Ordinary MP2/MP3 names select conventional total energies. SCS controls
describe the scaled diagnostic/typed result; use the SCS-MP2 or SCS-MP3
method name to select a scaled total energy.

### Common HF-model option defaults

Generated from `hf_rs.hf_variant_catalog()`. Each row accepts only its
listed controls, plus `scf_max_iter=100` and `scf_energy_tol=1e-12`.
These common defaults can differ from the typed Rust defaults.
For sigma-SCF only, the legacy `scf_energy_tol` key controls the variance
change in Hartree squared; ASE converts it as eV squared. Its typed
`SigmaScfOptions` separates variance and variance-gradient thresholds.

| HF model | Additional control/type/default/unit | Meaning / restrictions |
| --- | --- | --- |
| `rohf` | None | The supplied multiplicity defines the open-shell populations. |
| `ghf` | None | Real spinor determinant; the adapter supplies multiplicity=1 and zero spin bias. |
| `qrhf` | None | Quasi-restricted natural orbitals with the supplied multiplicity. |
| `mom` | `occupied_orbitals`: integer list, `omit`, 1; Required distinct zero-based MOM occupied guess-orbital indices, with the molecular closed-shell electron count. | Closed-shell maximum-overlap determinant with explicit occupied guess indices. |
| `fermi-hf` | `thermal_energy`: real, `omit`, hartree; Required positive Fermi-HF kBT; the legacy typed Rust entry point instead labels its temperature argument in kelvin. | Closed-shell finite-temperature ensemble; thermal_energy is kBT in Hartree. |
| `puhf` | None | Default singlet projection; exported raw-UHF tensors are not a projected density. |
| `hfb` | None | Default pairing strength 0.05 Hartree; typed HfbOptions exposes the pairing model. |
| `tcscf` | None | Active pair is the highest occupied and lowest unoccupied guess orbital. |
| `qed-hf` | `cavity_frequency`: real, `0.1`, hartree; Strictly positive photon energy.<br>`coupling`: real[3], `[0.0,0.0,0.05]`, hartree^0.5/bohr; Finite Cartesian length-gauge coupling vector. | Single coherent photon product state; no photon zero-point constant or electron-photon correlation. |
| `df-hf` | `aux_basis`: string, `"def2-svp-ri"`, 1; Built-in physical Coulomb-fitting basis name.<br>`metric_tol`: real, `1e-10`, hartree*bohr^3; Positive absolute auxiliary Coulomb-metric eigenvalue cutoff. | Coulomb-metric RI-JK with an explicit physical auxiliary basis. |
| `cd-hf` | None | Default Cholesky tolerance 1e-4 Hartree; typed API changes the threshold. |
| `x2c` | None | Spin-free one-electron X2C with untransformed two-electron integrals. |
| `dkh2` | None | Spin-free one-electron DKH2 with untransformed two-electron integrals. |
| `zora` | None | Recognized name; the required grid Hamiltonian is not implemented and returns an error. |
| `complex-hf` | None | Restricted complex determinant in magnetic field [0,0,0.02] atomic units. |
| `cuhf` | None | Unrestricted complex determinant in magnetic field [0,0,0.02] atomic units. |
| `cghf` | None | Generalized complex determinant in magnetic field [0,0,0.02] atomic units. |
| `lhf` | None | Self-consistent grid LHF/CEDA; typed LhfOptions exposes grid/convergence controls. |
| `oep` | None | Finite Gaussian-potential OEP; typed OepOptions controls the potential space. |
| `neo-hf` | None | First hydrogen is quantized in the selected electronic hydrogen basis; use typed API for a proton basis. |
| `positronic-hf` | None | Default one-positron product mean field in the supplied Gaussian working space. |
| `dc-hf` | None | Default consecutive subsystems; typed API supplies central/buffer partitions. |
| `fmo2` | None | Neutral consecutive fragments of max(N_atoms/2,1) atoms; typed FMOFragment specifies chemical partitions. |
| `fmo3` | None | Same default fragments as FMO2; connected trimers require at least three fragments. |
| `tdhf` | `n_singlets`: integer, `1`, 1; Nonnegative number of singlet response roots; total requested roots must be positive.<br>`n_triplets`: integer, `0`, 1; Nonnegative number of triplet response roots.<br>`root`: integer, `0`, 1; Zero-based selected state; must be smaller than the computed state count. | RHF linear-response RPA; e_total selects a vertical excited state. |
| `sahf` | `configurations`: configuration list, `omit`, 1; Required determinants, each with distinct zero-based alpha and beta occupied indices and the molecular electron count.<br>`weights`: real list, `omit`, 1; Required finite nonnegative weights, one per configuration, summing to one within 1e-12. | Common-orbital determinant ensemble with explicit populations and nonnegative weights summing to one. |
| `sigma-scf` | `configuration`: configuration, `omit`, 1; Required determinant with alpha/beta occupied indices and the molecular electron count.<br>`target_energy`: real, `omit`, hartree; Required finite target energy for variance optimization. | Determinant variance optimization about an explicit target energy. |
| `crystal-hf` | `lattice_constant`: real, `omit`, bohr; Required positive z lattice period; the adapter validates its real-space/k-point model.<br>`n_kpoints`: integer, `16`, 1; Positive nonaliasing one-dimensional k mesh. | Neutral closed-shell one-dimensional chain along z; energy is per cell. |
| `radial-hf` | `n_grid_points`: integer, `4096`, 1; Radial grid size, at least 16 points.<br>`r_max`: real, `omit`, bohr; Omission uses the solver's charge-scaled radial boundary; a supplied boundary must be positive. | Single all-electron atom/ion with one or two 1s electrons; Gaussian basis is unused by the radial solver. |

Configuration objects use `{"alpha":[...],"beta":[...]}` with distinct
zero-based occupied indices. SA-HF weights must be nonnegative and sum
to one; they are not silently renormalized. MOM occupation indices refer
to the guess orbitals, not an Aufbau reordering after targeting.
For all common geometry/function arguments and integral defaults, see
[native inputs](python_api.md#complete-common-input-signatures) and
[integral inspection](python_api.md#integrals-and-retained-intermediates).
For independent typed HF model controls use the
[complete Rust tables](rust_api.md#typed-hf-option-reference).
