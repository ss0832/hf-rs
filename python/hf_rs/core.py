# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Thin native API. Geometry conversion and method dispatch happen in Rust."""

from functools import wraps

from . import _hf_rs as _rust

run = _rust.run
Molecule = _rust.Molecule
list_builtin_basis_sets = _rust.list_builtin_basis_sets
method_catalog = _rust.method_catalog
option_catalog = _rust.option_catalog
hf_variant_catalog = _rust.hf_variant_catalog
oracle_compare = _rust.oracle_compare


def _normalize_inputs(atomic_numbers, positions_bohr=None, positions_angstrom=None):
    if (positions_bohr is None) == (positions_angstrom is None):
        raise ValueError("supply exactly one coordinate array")
    positions = positions_bohr if positions_bohr is not None else positions_angstrom
    unit = "bohr" if positions_bohr is not None else "angstrom"
    numbers = list(atomic_numbers)
    converted = _rust.convert_positions(positions, unit)
    if len(numbers) != len(converted):
        raise ValueError("atomic number and coordinate counts differ")
    return numbers, converted


def _legacy_call(function):
    @wraps(function)
    def call(
        atomic_numbers, positions_bohr=None, *, positions_angstrom=None, **options
    ):
        numbers, positions = _normalize_inputs(
            atomic_numbers, positions_bohr, positions_angstrom
        )
        return function(numbers, positions, **options)

    return call


for _name in (
    "analyze_stability",
    "calc_with_stability_rotation",
    "get_orbital_info",
    "calc_molecular_properties",
    "calc_hessian",
    "calc_third_derivatives",
    "calc_fourth_derivatives",
    "calc_hf_nuclear_derivative",
    "export_molden_file",
):
    globals()[_name] = _legacy_call(getattr(_rust, _name))


def _configuration(options):
    values = dict(options)
    config = {
        key: values.pop(key)
        for key in ("basis", "ecp", "cartesian", "charge", "multiplicity", "reference")
        if key in values
    }
    for old, new in (
        ("max_iter", "scf_max_iter"),
        ("tol", "scf_energy_tol"),
        ("conv_algo", "scf_conv_algo"),
        ("is_broken_symmetry", "broken_symmetry"),
    ):
        if old in values:
            values[new] = values.pop(old)
    return config, values


def calc_energy_and_gradient(
    atomic_numbers, positions_bohr=None, *, positions_angstrom=None, **options
):
    numbers, positions = _normalize_inputs(
        atomic_numbers, positions_bohr, positions_angstrom
    )
    config, values = _configuration(options)
    result = run(
        "hf", numbers, positions, **config, options=values, properties=["gradient"]
    )
    return result["e_total"], result["gradient"]


def calc_uhf_energy_and_gradient(
    atomic_numbers, positions_bohr=None, *, positions_angstrom=None, **options
):
    numbers, positions = _normalize_inputs(
        atomic_numbers, positions_bohr, positions_angstrom
    )
    config, values = _configuration(options)
    config["reference"] = "uhf"
    result = run(
        "uhf", numbers, positions, **config, options=values, properties=["gradient"]
    )
    return result["e_total"], result["components"]["S_squared"], result["gradient"]


def calc_post_hf_energy_and_gradient(
    atomic_numbers,
    positions_bohr=None,
    *,
    method="mp2",
    positions_angstrom=None,
    **options,
):
    numbers, positions = _normalize_inputs(
        atomic_numbers, positions_bohr, positions_angstrom
    )
    config, values = _configuration(options)
    result = run(
        method, numbers, positions, **config, options=values, properties=["gradient"]
    )
    return result["e_total"], result["gradient"]


class HartreeFock:
    """Compatibility facade; new applications should use native Molecule."""

    def __init__(
        self, atomic_numbers, positions_bohr=None, *, positions_angstrom=None, **options
    ):
        self.numbers, self.positions = _normalize_inputs(
            atomic_numbers, positions_bohr, positions_angstrom
        )
        self.config, self.options = _configuration(options)
        self.molecule = Molecule(self.numbers, self.positions, **self.config)

    def energy(self):
        return self.molecule.scf(options=self.options)["e_total"]

    def gradient(self):
        return self.molecule.gradient(options=self.options)

    def hessian(self):
        return self.molecule.hessian(options=self.options)

    def _tensor(self, name):
        if self.config.get("ecp") is not None or not isinstance(
            self.config.get("basis", "sto-3g"), str
        ):
            raise NotImplementedError(
                "compatibility full tensors do not accept independent ECPs or element maps; "
                "use native Molecule mixed derivatives"
            )
        if not self.config.get("cartesian", True):
            raise NotImplementedError(
                "compatibility tensor functions require Cartesian orbitals; "
                "use native Molecule mixed derivatives for spherical spaces"
            )
        options = dict(self.config)
        options.pop("reference", None)
        options.pop("cartesian", None)
        for new, old in (
            ("scf_max_iter", "max_iter"),
            ("scf_energy_tol", "tol"),
            ("broken_symmetry", "is_broken_symmetry"),
        ):
            if new in self.options:
                options[old] = self.options[new]
        return globals()[name](self.numbers, self.positions, **options)

    def third_derivatives(self):
        return self._tensor("calc_third_derivatives")

    def fourth_derivatives(self):
        return self._tensor("calc_fourth_derivatives")


for _name in (
    "get_atomic_mass",
    "get_atomic_mass_au",
    "get_standard_atomic_weight",
    "get_element_symbol",
    "get_element_name",
    "find_atomic_number",
    "AMU_TO_AU",
    "BOHR_TO_ANGSTROM",
    "ANGSTROM_TO_BOHR",
    "HARTREE_TO_EV",
    "EV_TO_HARTREE",
    "HARTREE_TO_J",
    "SPEED_OF_LIGHT_AU",
):
    globals()[_name] = getattr(_rust, _name)
AU_TO_AMU = 1.0 / _rust.AMU_TO_AU
__all__ = [name for name in globals() if not name.startswith("_") and name != "wraps"]
