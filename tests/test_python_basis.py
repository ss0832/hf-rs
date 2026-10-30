# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Native element assignments, scalar ECP precedence, CLI/ASE and fourth response."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import hf_rs

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads(
    (ROOT / "tests/data/basis_assignment_reference.json").read_text(encoding="utf-8")
)["cases"]


class BasisAssignments(unittest.TestCase):
    def test_all_mixed_basis_energies_and_gradient_components(self):
        self.assertEqual(len(set(hf_rs.list_builtin_basis_sets())), 80)
        for case in CASES:
            result = hf_rs.run(
                "hf",
                case["atomic_numbers"],
                case["positions"],
                basis=case["basis"],
                ecp=case["ecp"],
                properties=["gradient"],
            )
            self.assertAlmostEqual(result["e_total"], case["energy"], delta=1e-8)
            for actual, expected in zip(result["gradient"], case["gradient"]):
                for value, reference in zip(actual, expected):
                    self.assertAlmostEqual(value, reference, delta=1e-6)

    def test_combined_and_independent_potential_files(self):
        case = CASES[1]
        source = ROOT / "third_party/basis_set_exchange/data"
        for extension in ("json", "gbs"):
            path = str(source / ("orbital_15." + extension))
            molecule = hf_rs.Molecule(
                case["atomic_numbers"],
                case["positions"],
                basis={"Na": path, "H": "sto-3g"},
            )
            self.assertAlmostEqual(
                molecule.scf()["e_total"], case["energy"], delta=1e-8
            )
            result = hf_rs.run(
                "hf",
                case["atomic_numbers"],
                case["positions"],
                basis=CASES[2]["basis"],
                ecp={"Na": path},
            )
            self.assertAlmostEqual(result["e_total"], CASES[2]["energy"], delta=1e-8)
        named = hf_rs.run(
            "hf",
            case["atomic_numbers"],
            case["positions"],
            basis=CASES[2]["basis"],
            ecp={"Na": "LANL2DZ ECP"},
        )
        self.assertAlmostEqual(named["e_total"], CASES[2]["energy"], delta=1e-8)

    def test_missing_or_invalid_element_assignments_raise(self):
        for basis, ecp in [
            ({"H": "sto-3g"}, None),
            ({"*": "sto-3g"}, {"Na": "def2-ecp"}),
            ({"Typo": "sto-3g"}, None),
        ]:
            with self.assertRaises(ValueError):
                hf_rs.Molecule([11, 1], [[0, 0, 0], [0, 0, 3]], basis=basis, ecp=ecp)

    def test_python_cli_passes_both_element_maps(self):
        case = CASES[2]
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "hf_rs.cli",
                "--geometry",
                "Na 0.2 -0.1 -1.4; H 0.5 0.25 1.65",
                "--basis",
                json.dumps(case["basis"]),
                "--ecp",
                json.dumps(case["ecp"]),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertAlmostEqual(
            json.loads(result.stdout)["e_total"], case["energy"], delta=1e-8
        )

    def test_ase_passes_native_basis_and_ecp_maps(self):
        import numpy as np
        from ase import Atoms, units

        case = CASES[2]
        atoms = Atoms(
            numbers=case["atomic_numbers"],
            positions=np.asarray(case["positions"]) * units.Bohr,
        )
        atoms.calc = hf_rs.HFCalculator(basis=case["basis"], ecp=case["ecp"])
        self.assertAlmostEqual(
            atoms.get_potential_energy(), case["energy"] * units.Hartree, delta=1e-7
        )
        np.testing.assert_allclose(
            atoms.get_forces(),
            -np.asarray(case["gradient"]) * units.Hartree / units.Bohr,
            atol=1e-5,
            rtol=0,
        )
        hydrogen = Atoms(
            "H2", positions=[[0, 0, -0.7 * units.Bohr], [0, 0, 0.7 * units.Bohr]]
        )
        hydrogen.calc = hf_rs.HFCalculator("hf", "sto-3g", 0, 1)
        self.assertAlmostEqual(
            hydrogen.get_potential_energy(),
            -1.1167143251757503 * units.Hartree,
            delta=1e-8,
        )

    def test_explicit_nonlocal_ecp_file_retains_fourth_order_response(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gbs"
            path.write_text(
                "H-ECP 1 0\np potential\n1\n2 0.7 -0.25\ns-p potential\n1\n2 0.4 0.15\n",
                encoding="utf-8",
            )
            molecule = hf_rs.Molecule(
                [1, 1],
                [[0.1, 0.2, -0.7], [-0.1, 0, 0.7]],
                basis={"H": "sto-3g"},
                ecp={"H": str(path)},
            )
            value = molecule.nuclear_derivative([0, 2, 2, 5])
            directions = [[float(j == i) for j in range(6)] for i in [0, 2, 2, 5]]
            self.assertAlmostEqual(
                value, molecule.directional_derivative(directions), delta=1e-10
            )
            self.assertGreater(abs(value), 1e-8)


if __name__ == "__main__":
    unittest.main()
