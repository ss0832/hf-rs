# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Installed native API, Python CLI and actual ASE geometry workflow."""

import json
import subprocess
import sys
import unittest

import hf_rs


class NativeApi(unittest.TestCase):
    numbers = [1, 1]
    positions = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]

    def test_integrals_intermediates_and_default_metadata(self):
        import numpy as np

        molecule = hf_rs.Molecule(self.numbers, self.positions)
        self.assertEqual(molecule.run()["intermediates"], {})
        overlap = molecule.integrals()
        self.assertEqual(overlap["tensor"]["shape"], [2, 2])
        self.assertFalse(overlap["orbital_response_included"])
        with self.assertRaises(ValueError):
            molecule.integrals(indices=[[], [0]])
        hf = molecule.intermediates(max_elements=20)
        self.assertEqual(sum(len(x["values"]) for x in hf.values()), 20)
        with self.assertRaises(ValueError):
            molecule.intermediates(max_elements=19)
        cc = molecule.intermediates("ccsd")
        self.assertEqual(cc["method.t2"]["shape"], [1, 1, 1, 1])
        eri = molecule.integrals(
            "eri", representation="mo_alpha", indices=[[0], [1], [0], [1]]
        )["tensor"]["values"][0]
        t1, t2 = cc["method.t1"]["values"][0], cc["method.t2"]["values"][0]
        self.assertAlmostEqual(
            eri * (t2 + t1 * t1), molecule.run("ccsd")["e_corr"], delta=1e-10
        )
        defaults = {x["name"]: x for x in hf_rs.option_catalog()}
        self.assertEqual(defaults["scf_max_iter"]["default"], 100)
        self.assertEqual(defaults["scf_energy_tol"]["default"], 1e-12)
        variants = {x["name"]: x for x in hf_rs.hf_variant_catalog()}
        qed = {x["name"]: x for x in variants["qed-hf"]["option_details"]}
        np.testing.assert_allclose(qed["coupling"]["default"], [0.0, 0.0, 0.05])

    def test_ase_integral_and_intermediate_unit_conversions(self):
        import numpy as np
        from ase import Atoms, units

        molecule = hf_rs.Molecule(self.numbers, self.positions)
        atoms = Atoms("H2", positions=np.asarray(self.positions) * units.Bohr)
        atoms.calc = hf_rs.HFCalculator()
        for operator, derivative in [
            ("overlap", [0]),
            ("kinetic", []),
            ("eri", [0, 1]),
        ]:
            native = molecule.integrals(operator, derivative_coordinates=derivative)
            actual = atoms.calc.get_integrals(
                atoms, operator, derivative_coordinates=derivative
            )
            tensor = native["tensor"]
            factor = (
                units.Hartree ** tensor["energy_power"]
                * units.Bohr ** tensor["length_power"]
            )
            np.testing.assert_allclose(
                actual["tensor"]["values"],
                np.asarray(tensor["values"]) * factor,
                atol=1e-10,
            )
        native = molecule.intermediates()
        actual = atoms.calc.get_intermediates(atoms)
        for name, tensor in native.items():
            factor = (
                units.Hartree ** tensor["energy_power"]
                * units.Bohr ** tensor["length_power"]
            )
            np.testing.assert_allclose(
                actual[name]["values"],
                np.asarray(tensor["values"]) * factor,
                atol=1e-10,
            )
        atoms.calc.calculate(atoms, properties=["intermediates"])
        self.assertEqual(actual, atoms.calc.calculation_result["intermediates"])
        from hf_rs.ase_calc import _native_options

        self.assertAlmostEqual(
            _native_options({"scf_energy_tol": 0.25 * units.Hartree**2}, "sigma-scf")[
                "scf_energy_tol"
            ],
            0.25,
        )

    def test_scf_exports_and_observables_use_native_units(self):
        import numpy as np

        from hf_rs import _hf_rs as native

        self.assertIs(hf_rs.Molecule, native.Molecule)
        molecule = hf_rs.Molecule(self.numbers, self.positions)
        data = molecule.scf_wavefunction()
        self.assertEqual(data["schema"], "hf-rs-scf-wavefunction-1")
        overlap = np.asarray(data["overlap_matrix"])
        channel = data["channels"][0]
        coefficients = np.asarray(channel["coefficients"])
        np.testing.assert_allclose(
            coefficients.T @ overlap @ coefficients, np.eye(2), atol=1e-12
        )
        self.assertIn("[Molden Format]", molecule.scf_molden())
        self.assertEqual(molecule.run()["observables"], {})
        observable = molecule.hf_observables()
        self.assertEqual(observable["dipole"]["unit"], "e*bohr")
        self.assertEqual(
            observable, molecule.run(properties=["observables"])["observables"]
        )
        with self.assertRaises(RuntimeError):
            molecule.hf_observables("mp2")

    def test_ase_observables_and_all_stored_results_use_ev_angstrom(self):
        import numpy as np
        from ase import Atoms, units

        atoms = Atoms("H2", positions=np.asarray(self.positions) * units.Bohr)
        options = dict(cavity_frequency=0.5, coupling=[0.01, 0.02, 0.03])
        direct = hf_rs.Molecule(
            self.numbers, atoms.positions.tolist(), unit="angstrom"
        ).run("qed-hf", options=options, properties=["observables"])
        atoms.calc = hf_rs.HFCalculator(
            method="qed-hf",
            options=dict(
                cavity_frequency=0.5 * units.Hartree,
                coupling=[
                    x * units.Hartree**0.5 / units.Bohr for x in options["coupling"]
                ],
            ),
        )
        observed = atoms.calc.get_observables(atoms)
        for key, item in direct["observables"].items():
            factor = (
                units.Hartree ** item["energy_power"]
                * units.Bohr ** item["length_power"]
            )
            np.testing.assert_allclose(
                observed[key]["value"], np.asarray(item["value"]) * factor, atol=1e-11
            )
        self.assertEqual(observed["dipole"]["unit"], "e*angstrom")
        self.assertEqual(observed["coupling"]["unit"], "eV^0.5/angstrom")
        self.assertAlmostEqual(
            atoms.calc.calculation_result["e_total"],
            direct["e_total"] * units.Hartree,
            delta=1e-10,
        )
        observed["coupling"]["value"][0] = 1000
        self.assertNotEqual(
            atoms.calc.results["observables"]["coupling"]["value"][0], 1000
        )
        atoms.calc.reset()
        self.assertIsNone(atoms.calc.calculation_result)
        atoms.calc = hf_rs.HFCalculator()
        hessian = atoms.calc.get_hessian(atoms)
        native_hessian = hf_rs.Molecule(
            self.numbers, atoms.positions.tolist(), unit="angstrom"
        ).hessian()
        np.testing.assert_allclose(
            hessian,
            np.asarray(native_hessian) * units.Hartree / units.Bohr**2,
            atol=1e-9,
        )
        atoms.calc = hf_rs.HFCalculator(
            method="fci",
            options=dict(
                active_n_alpha=1, active_n_beta=1, active_n_orb=2, n_roots=2, root=1
            ),
        )
        atoms.get_potential_energy()
        stored = atoms.calc.calculation_result
        self.assertEqual(stored["units"]["energy"], "eV")
        self.assertEqual(stored["e_total"], stored["states"][1]["energy"])
        self.assertEqual(stored["components"]["E_CASCI"], stored["e_total"])
        atoms.calc.method = "typo"
        atoms.positions[0, 0] += 0.01
        with self.assertRaises(ValueError):
            atoms.get_potential_energy()
        self.assertIsNone(atoms.calc.calculation_result)

    def test_both_cli_catalog_and_embedded_oracle_are_available(self):
        for flag in ("--list-methods", "--oracle-compare"):
            result = subprocess.run(
                [sys.executable, "-m", "hf_rs.cli", flag],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue(json.loads(result.stdout))
        result = subprocess.run(
            [sys.executable, "-m", "hf_rs.cli", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "hf-rs " + hf_rs.__version__)

    def test_units_molecule_and_method_options(self):
        expected = hf_rs.run("mp2", self.numbers, self.positions)
        molecule = hf_rs.Molecule(self.numbers, self.positions)
        self.assertAlmostEqual(
            molecule.run("mp2")["e_total"], expected["e_total"], delta=1e-12
        )
        self.assertAlmostEqual(
            molecule.scf()["e_total"], -1.1167143251757503, delta=1e-9
        )
        angstrom = [[x * hf_rs.BOHR_TO_ANGSTROM for x in p] for p in self.positions]
        actual = hf_rs.run("mp2", self.numbers, angstrom, unit="angstrom")
        self.assertAlmostEqual(actual["e_total"], expected["e_total"], delta=1e-12)
        self.assertEqual(actual["units"]["gradient"], "hartree/bohr")
        for method, options in [("misspelled", {}), ("mp2", {"typo": 1})]:
            with self.assertRaises(ValueError):
                molecule.run(method, options=options)
        with self.assertRaises(RuntimeError):
            molecule.gradient("ccsd")

    def test_selected_state_is_not_dropped_by_binding(self):
        options = dict(
            active_n_alpha=1, active_n_beta=1, active_n_orb=2, n_roots=2, root=1
        )
        result = hf_rs.run("fci", self.numbers, self.positions, options=options)
        self.assertEqual(result["root"], 1)
        self.assertEqual(len(result["states"]), 2)
        self.assertEqual(result["e_total"], result["states"][1]["energy"])
        self.assertGreater(result["e_total"], result["states"][0]["energy"])

    def test_all_derivative_entries_and_fourth_order_remain_available(self):
        molecule = hf_rs.Molecule(self.numbers, self.positions)
        result = molecule.run(properties=["gradient", "hessian"])
        self.assertEqual(result["gradient_kind"], "analytic")
        self.assertEqual(result["hessian_kind"], "analytic")
        self.assertEqual(molecule.gradient(), result["gradient"])
        self.assertEqual(molecule.hessian(), result["hessian"])
        value = molecule.nuclear_derivative([2, 2, 2, 2])
        direction = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(
            value, molecule.directional_derivative([direction] * 4), delta=1e-10
        )

    def test_python_cli_returns_json_and_errors_on_unknown_method(self):
        result = subprocess.run(
            [sys.executable, "-m", "hf_rs.cli", "--method", "mp2"],
            capture_output=True,
            text=True,
            check=True,
        )
        parsed = json.loads(result.stdout)
        expected = hf_rs.run("mp2", self.numbers, self.positions)
        self.assertAlmostEqual(parsed["e_total"], expected["e_total"], delta=1e-12)
        bad = subprocess.run(
            [sys.executable, "-m", "hf_rs.cli", "--method", "typo"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("unknown method", bad.stderr)

    def test_ase_forces_and_bfgs_use_native_angstrom_input(self):
        from ase import Atoms, units
        from ase.optimize import BFGS

        atoms = Atoms("H2", positions=[[0, 0, -0.4], [0, 0, 0.4]])
        atoms.calc = hf_rs.HFCalculator(scf_energy_tol=1e-13)
        direct = hf_rs.run(
            "hf",
            [1, 1],
            atoms.positions.tolist(),
            unit="angstrom",
            options={"scf_energy_tol": 1e-13},
            properties=["gradient"],
        )
        forces = atoms.get_forces()
        for a in range(2):
            for k in range(3):
                self.assertAlmostEqual(
                    forces[a, k],
                    -direct["gradient"][a][k] * units.Hartree / units.Bohr,
                    delta=1e-8,
                )
        optimizer = BFGS(atoms, logfile=None)
        self.assertTrue(optimizer.run(fmax=0.01, steps=15))
        self.assertLess(float(abs(atoms.get_forces()).max()), 0.01)
        atoms.calc = hf_rs.HFCalculator(method="typo")
        with self.assertRaises(ValueError):
            atoms.get_potential_energy()


if __name__ == "__main__":
    unittest.main()
