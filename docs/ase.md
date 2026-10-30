# ASE calculator

Install ASE and build the native extension. `HFCalculator` and its alias
`HFRSCalculator` expose energy and implemented analytic forces.

The [family recipes](methods.md#calculation-recipes) translate directly to
`HFCalculator(method=..., options=...)`, including correlated methods and
connected molecular HF variants. The [option index](methods.md#registered-method-options)
and [capability table](methods.md) identify unsupported combinations.

```python
from ase import Atoms
from ase.optimize import BFGS
from hf_rs import HFCalculator

atoms = Atoms("H2", positions=[[0., 0., -0.4], [0., 0., 0.4]])
atoms.calc = HFCalculator(method="hf", basis="sto-3g", scf_energy_tol=1e-13)
BFGS(atoms).run(fmax=0.01, steps=15)
print(atoms.get_potential_energy(), atoms.get_forces())
```

The constructor accepts `method`, `basis`, `ecp`, `charge`, `multiplicity`, `reference`,
`cartesian`, `options={...}`, and additional method keyword options. Duplicate
keys between `options` and extra keywords are rejected. Unknown methods/options
propagate native errors. Use `scf_` keys for SCF and the selected method's schema
for post-HF controls; see the [catalog](methods.md).
`basis` and `ecp` accept the native element maps and file selections described
in [the Python API](python_api.md#basis-and-ecp-selection).

ASE positions pass unchanged to native `Molecule(..., unit="angstrom")`.
Only Rust converts atomic coordinates. Energy is converted from Hartree to eV;
forces are minus the native analytic gradient, converted to eV/angstrom.
Production forces do not silently use finite differences. Post-HF/variant force
requests without an available analytic route raise an error. The calculator
constructs a new native molecule when ASE requests a changed geometry.

## Post-HF energies and active spaces

```python
from ase import Atoms
from ase.units import Bohr
from hf_rs import HFCalculator

atoms = Atoms("LiH", positions=[[0., 0., 0.], [0., 0., 3. * Bohr]])
active = dict(active_n_alpha=1, active_n_beta=1, active_n_orb=2)
for method in ("mp2", "ccsd", "ccsd(t)", "cis", "cisd", "fci", "casscf", "caspt2"):
    controls = active if method in {"fci", "casscf", "caspt2"} else {}
    atoms.calc = HFCalculator(method=method, basis="sto-3g", options=controls)
    print(method, atoms.get_potential_energy())
    print(atoms.calc.calculation_result["states"], atoms.calc.calculation_result["one_rdm"])
```

Use the same pattern for MP3–MP8, DOCI, SCS/RI/local correlation, selected CI,
DMRG, F12 and SR/MR-DSRG; select options from the shared recipes. Requesting
`get_forces()` for energy-only methods raises an error. ASE finite-difference
vibration tools are numerical analyses; using them does not create an analytic
Hessian implementation for a method.

## HF variants and open shells

```python
from ase import Atoms
from hf_rs import HFCalculator

hydrogen = Atoms("H", positions=[[0., 0., 0.]])
hydrogen.calc = HFCalculator(method="uhf", reference="uhf", multiplicity=2)
print(hydrogen.get_potential_energy(), hydrogen.get_forces())
print(hydrogen.calc.calculation_result["components"]["S_squared"])

h2 = Atoms("H2", positions=[[0., 0., -0.37], [0., 0., 0.37]])
from ase.units import Hartree, Bohr
h2.calc = HFCalculator(method="qed-hf", options={
    "cavity_frequency": 0.5 * Hartree, "coupling": [0., 0., 0.02 * Hartree**0.5 / Bohr],
})
print(h2.get_potential_energy(), h2.get_forces())
```

For RI-HF set `method="ri-hf"` and `aux_basis="def2-svp-ri"`; for the other
connected variants use the table in [HF recipes](methods.md#molecular-hf-variants).
The [specialized common controls](methods.md#specialized-hf-inputs) also expose
TDHF, QRHF/MOM/Fermi, periodic/radial, SA-HF and sigma-SCF energies/observables.
Custom typed Rust controls, such as explicit charged fragments, remain outside
this ASE adapter.

## Selected roots, options and stored results

`root`/`n_roots` remain discrete method options. `calculation_result` holds the
last successful dictionary, including the selected state's energy,
all returned states, components, available RDM and derivative labels. All physical
values use eV/angstrom-based units, including component/state energies.
`reset()` clears both result stores. After changing geometry, trigger another
ASE calculation before reading or saving the dictionary.

Set `basis`/`ecp` to per-element dictionaries for mixed systems, for example
`basis={"Na":"6-31g","H":"sto-3g"}, ecp={"Na":"lanl2dz"}`.
Unknown keys are rejected by the native solver; duplicate keys in `options`
and additional constructor keyword options are rejected by the calculator.

## Units and observables

`get_observables(atoms)` requests method-specific HF quantities with unit
metadata. Dipoles are `e*angstrom`, quadrupoles `e*angstrom^2`, energies eV,
reciprocal k vectors `angstrom^-1` and variances `eV^2`. Spin, occupations,
photon numbers, amplitudes and classifications are dimensionless.
The full model-specific list is in [HF observables](methods.md#hf-observables).
`get_hessian(atoms)` returns the analytic Hessian in eV/angstrom squared where
the common native API supports it.

Physical option values also use ASE units: `cavity_frequency`, `thermal_energy`,
`target_energy`, CASPT2 `ipea_shift`/`real_shift`/`imaginary_shift`,
`scf_energy_tol`, `scf_residual_tol`, `energy_tol` and `reference_energy_tol`
are eV; `lattice_constant`, `r_max` and `numerical_gradient_step` are angstrom;
F12 `beta` is inverse angstrom; cavity `coupling` is eV^0.5/angstrom;
DSRG `flow_parameter`, `ode_step` and `dynamic_weight_parameter` are eV^-2.
Multiply native atomic-unit inputs by `ase.units.Hartree` and `ase.units.Bohr`
using these powers. Omitted options retain native defaults. Discrete settings,
occupations and dimensionless overlap/CABS thresholds retain their native
definition. In sigma-SCF the legacy `scf_energy_tol` name denotes a variance-
change threshold and uses eV squared, as its HF-model metadata specifies. The DF-HF `metric_tol` uses eV angstrom cubed when supplied. Solver residual tolerances have method-specific definitions;
they are algorithmic controls rather than exported observables.

```python
from ase import Atoms
from hf_rs import HFCalculator

atoms = Atoms("H2", positions=[[0., 0., -0.37], [0., 0., 0.37]])
atoms.calc = HFCalculator(method="tdhf")
spectra = atoms.calc.get_observables(atoms)
print(spectra["excitation_energies"], spectra["oscillator_strengths"])
atoms.calc = HFCalculator(method="hf")
print(atoms.calc.get_observables(atoms)["dipole"])
print(atoms.calc.get_hessian(atoms))
```

## Final-geometry output

```python
import json
from pathlib import Path
from ase import Atoms
from ase.optimize import BFGS
from hf_rs import HFCalculator, Molecule

atoms = Atoms("H2", positions=[[0., 0., -0.4], [0., 0., 0.4]])
atoms.calc = HFCalculator(method="mp2", basis="sto-3g", scf_energy_tol=1e-13)
BFGS(atoms, logfile=None).run(fmax=0.01, steps=15)
atoms.get_forces()
result = atoms.calc.calculation_result
Path("ase-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
configuration = dict(atoms.calc.configuration)
configuration["reference"] = result["reference"]
reference = Molecule(atoms.numbers.tolist(), atoms.positions.tolist(), unit="angstrom", **configuration)
Path("ase-reference.molden").write_text(reference.scf_molden(), encoding="utf-8")
Path("ase-reference.json").write_text(json.dumps(reference.scf_wavefunction(), indent=2), encoding="utf-8")
```

Export after optimization so the coordinates match the final structure.
`calculation_result` is populated by a successful calculation and cleared by
`reset()` or a failed recalculation; its physical values use eV/angstrom units.
The ASE bridge constructs another reference for the final geometry, without
retaining an ERI-bearing molecule in the calculator. The SCF export is the
configured reference, not the correlated MP2/CASSCF/other state.
The explicitly created native `Molecule` uses atomic units in its JSON snapshot;
Molden uses its standard atomic-unit file convention. These native file exports
are separate from the ASE calculator's numerical results.

## Integral and intermediate inspection

`HFCalculator.get_integrals(atoms, operator="overlap", **request)` accepts the
[native integral controls](python_api.md#integrals-and-retained-intermediates).
`get_intermediates(atoms=None, max_elements=1_000_000)` copies actual retained
method tensors. If atoms are omitted, the calculator must already have atoms.
The normal `intermediates` property uses the default one-million-value bound.
These diagnostic quantities use eV/angstrom-based units, including derivatives,
RI factors, radial grids and radial orbital amplitudes; shape and index lists
are unchanged. Explicit `metric_cutoff` uses eV angstrom cubed in this getter.
Omission retains the native metric cutoff. Memory limits are bytes and counts.

```python
import numpy as np
from ase import Atoms, units
from hf_rs import HFCalculator

atoms = Atoms("H2", positions=[[0.,0.,-0.7*units.Bohr],[0.,0.,0.7*units.Bohr]])
atoms.calc = HFCalculator(method="hf")
block = atoms.calc.get_integrals(atoms, "core_hamiltonian", indices=[[0],[1]])
assert block["tensor"]["unit"] == "eV"
tensors = atoms.calc.get_intermediates(atoms)
fock = tensors["model.fock_cartesian_ao"]
assert np.asarray(fock["values"]).reshape(fock["shape"]).shape == (2,2)
assert fock["unit"] == "eV"
```

Integral MO/Fock getters inspect the configured ordinary RHF/UHF reference.
For an HF variant's actual Hamiltonian, density or pairing tensor use its
retained `model.*` intermediates. Bare integral derivatives include no orbital
response. Request ASE forces/Hessians for the supported relaxed energy response.
