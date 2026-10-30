# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""ASE adapter forwarding coordinates and method options to the native API."""

from copy import deepcopy

from .core import Molecule

try:
    from ase import units
    from ase.calculators.calculator import Calculator, all_changes
except ImportError:
    Calculator = object
    all_changes = []
    units = None


class HFCalculator(Calculator):
    implemented_properties = [
        "energy",
        "forces",
        "hessian",
        "observables",
        "intermediates",
    ]

    def __init__(
        self,
        method="hf",
        basis="sto-3g",
        charge=0,
        multiplicity=1,
        reference="auto",
        cartesian=True,
        options=None,
        atoms=None,
        ecp=None,
        **method_options,
    ):
        if units is None:
            raise ImportError("ASE is required for HFCalculator")
        super().__init__(atoms=atoms)
        self.method = method
        self.configuration = dict(
            basis=basis,
            ecp=ecp,
            charge=charge,
            multiplicity=multiplicity,
            reference=reference,
            cartesian=cartesian,
        )
        self.method_options = dict(options or {})
        overlap = self.method_options.keys() & method_options.keys()
        if overlap:
            raise ValueError("duplicate method options: " + ", ".join(sorted(overlap)))
        self.method_options.update(method_options)
        self.calculation_result = None

    def reset(self):
        self.calculation_result = None
        super().reset()

    def get_observables(self, atoms=None):
        """Return method-specific quantities with eV/angstrom unit metadata."""
        return deepcopy(self.get_property("observables", atoms))

    def get_integrals(self, atoms, operator="overlap", **request):
        """Inspect standard-reference integrals in eV/angstrom units."""
        molecule = Molecule(
            atoms.numbers.tolist(),
            atoms.positions.tolist(),
            unit="angstrom",
            **self.configuration,
        )
        if "options" not in request:
            request["options"] = _native_options(
                {
                    key: value
                    for key, value in self.method_options.items()
                    if key.startswith("scf_") or key == "broken_symmetry"
                    if not (_is_sigma(self.method) and key == "scf_energy_tol")
                }
            )
        else:
            request["options"] = _native_options(request["options"])
        if "metric_cutoff" in request:
            request["metric_cutoff"] /= units.Hartree * units.Bohr**3
        result = molecule.integrals(operator, **request)
        tensor = dict(result["tensor"])
        factor = (
            units.Hartree ** tensor["energy_power"]
            * units.Bohr ** tensor["length_power"]
        )
        tensor["values"] = _scale(tensor["values"], factor)
        tensor["unit"] = (
            tensor["unit"].replace("hartree", "eV").replace("bohr", "angstrom")
        )
        result["tensor"] = tensor
        return result

    def get_intermediates(self, atoms=None, max_elements=1_000_000):
        """Return retained method tensors in eV/angstrom-based units."""
        atoms = atoms if atoms is not None else self.atoms
        if atoms is None:
            raise ValueError("atoms are required to inspect intermediates")
        molecule = Molecule(
            atoms.numbers.tolist(),
            atoms.positions.tolist(),
            unit="angstrom",
            **self.configuration,
        )
        result = molecule.intermediates(
            self.method,
            options=_native_options(self.method_options, self.method),
            max_elements=max_elements,
        )
        return {key: _converted_tensor(item) for key, item in result.items()}

    def get_hessian(self, atoms=None):
        """Return the analytic Hessian in eV/angstrom squared."""
        return self.get_property("hessian", atoms).copy()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calculation_result = None
        self.results = {}
        molecule = Molecule(
            self.atoms.get_atomic_numbers().tolist(),
            self.atoms.get_positions().tolist(),
            unit="angstrom",
            **self.configuration,
        )
        requested = ["energy"]
        for source, target in (
            ("forces", "gradient"),
            ("hessian", "hessian"),
            ("observables", "observables"),
            ("intermediates", "intermediates"),
        ):
            if source in properties:
                requested.append(target)
        result = molecule.run(
            self.method,
            options=_native_options(self.method_options, self.method),
            properties=requested,
        )
        converted = _ase_result(result)
        self.calculation_result = converted
        self.results["energy"] = converted["e_total"]
        if "forces" in properties:
            import numpy as np

            self.results["forces"] = -np.asarray(converted["gradient"])
        if "hessian" in properties:
            import numpy as np

            self.results["hessian"] = np.asarray(converted["hessian"])
        if "observables" in properties:
            self.results["observables"] = converted["observables"]
        if "intermediates" in properties:
            self.results["intermediates"] = converted["intermediates"]


def _scale(value, factor):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, list):
        return [_scale(item, factor) for item in value]
    return value if factor == 1 else value * factor


def _converted_observable(item):
    result = dict(item)
    factor = units.Hartree ** item["energy_power"] * units.Bohr ** item["length_power"]
    result["value"] = _scale(item["value"], factor)
    result["unit"] = item["unit"].replace("hartree", "eV").replace("bohr", "angstrom")
    return result


def _converted_tensor(item):
    result = dict(item)
    factor = units.Hartree ** item["energy_power"] * units.Bohr ** item["length_power"]
    result["values"] = _scale(item["values"], factor)
    result["unit"] = item["unit"].replace("hartree", "eV").replace("bohr", "angstrom")
    return result


def _ase_result(native):
    result = dict(native)
    for key in ("e_scf", "e_corr", "e_total"):
        result[key] *= units.Hartree
    result["states"] = [
        dict(state, energy=state["energy"] * units.Hartree)
        for state in native["states"]
    ]
    result["gradient"] = _scale(native["gradient"], units.Hartree / units.Bohr)
    result["hessian"] = _scale(native["hessian"], units.Hartree / units.Bohr**2)
    result["observables"] = {
        key: _converted_observable(item) for key, item in native["observables"].items()
    }
    result["intermediates"] = {
        key: _converted_tensor(item) for key, item in native["intermediates"].items()
    }
    result["components"] = {}
    result["component_units"] = {}
    for key, value in native["components"].items():
        descriptor = native["component_units"].get(key)
        if descriptor is None:
            raise ValueError("missing unit metadata for component: " + key)
        converted = _converted_observable(dict(descriptor, value=value))
        result["components"][key] = converted.pop("value")
        result["component_units"][key] = converted
    result["units"] = dict(
        energy="eV", gradient="eV/angstrom", hessian="eV/angstrom^2", one_rdm="MO"
    )
    return result


def _is_sigma(method):
    return (
        str(method).lower().replace("-", "").replace("_", "").replace(" ", "")
        == "sigmascf"
    )


def _native_options(options, method=None):
    """Convert physical controls; discrete and dimensionless controls stay unchanged."""
    dimensions = {
        "cavity_frequency": (1, 0),
        "coupling": (0.5, -1),
        "target_energy": (1, 0),
        "thermal_energy": (1, 0),
        "lattice_constant": (0, 1),
        "r_max": (0, 1),
        "metric_tol": (1, 3),
        "beta": (0, -1),
        "real_shift": (1, 0),
        "ipea_shift": (1, 0),
        "imaginary_shift": (1, 0),
        "flow_parameter": (-2, 0),
        "ode_step": (-2, 0),
        "dynamic_weight_parameter": (-2, 0),
        "scf_energy_tol": (1, 0),
        "scf_residual_tol": (1, 0),
        "energy_tol": (1, 0),
        "reference_energy_tol": (1, 0),
        "numerical_gradient_step": (0, 1),
    }
    result = dict(options)
    if _is_sigma(method):
        dimensions["scf_energy_tol"] = (2, 0)
    for key, (energy, length) in dimensions.items():
        if key in result:
            result[key] = _scale(
                result[key], units.Hartree**-energy * units.Bohr**-length
            )
    return result


HFRSCalculator = HFCalculator
