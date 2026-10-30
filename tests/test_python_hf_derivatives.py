# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832

"""Run with the built native extension and python/ on PYTHONPATH."""

import json
import unittest
from pathlib import Path

import hf_rs
from hf_rs import core


@unittest.skipIf(core._rust is None, "build the native extension before running")
class AnalyticalHfDerivatives(unittest.TestCase):
    numbers = [1, 1]
    positions = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
    direction = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    def test_fourth_direction_matches_tensor_and_scales_multilinearly(self):
        q = hf_rs.calc_fourth_derivatives(self.numbers, self.positions, tol=1e-13)
        directions = [self.direction] * 4
        value = hf_rs.calc_hf_nuclear_derivative(
            self.numbers, self.positions, directions=directions, tol=1e-13
        )
        self.assertAlmostEqual(value, q[2][2][2][2], delta=1e-9)
        scaled = [[-2.0 * x for x in self.direction]] + directions[1:]
        self.assertAlmostEqual(
            hf_rs.calc_hf_nuclear_derivative(
                self.numbers, self.positions, directions=scaled, tol=1e-13
            ),
            -2.0 * value,
            delta=1e-9,
        )

    def test_open_shell_third_and_fourth_match_hessian_differences(self):
        options = dict(charge=1, multiplicity=2, tol=1e-13)
        step = 5e-3

        def curvature(displacement):
            moved = [p[:] for p in self.positions]
            moved[0][2] += displacement
            hessian, _ = hf_rs.calc_hessian(self.numbers, moved, **options)
            return hessian[2][2]

        pp, p, z, m, mm = [curvature(d * step) for d in (2, 1, 0, -1, -2)]
        third = hf_rs.calc_third_derivatives(self.numbers, self.positions, **options)
        self.assertAlmostEqual(
            third[2][2][2], (-pp + 8 * p - 8 * m + mm) / (12 * step), delta=2e-7
        )
        calculation = hf_rs.HartreeFock(self.numbers, self.positions, **options)
        self.assertAlmostEqual(calculation.hessian()[2][2], z, delta=1e-10)
        self.assertAlmostEqual(
            calculation.third_derivatives()[2][2][2], third[2][2][2], delta=1e-10
        )
        fourth = hf_rs.calc_hf_nuclear_derivative(
            self.numbers, self.positions, directions=[self.direction] * 4, **options
        )
        self.assertAlmostEqual(
            fourth,
            (-pp + 16 * p - 30 * z + 16 * m - mm) / (12 * step * step),
            delta=3e-6,
        )

    def test_angstrom_coordinates_are_converted(self):
        bohr = core._normalize_inputs(self.numbers, self.positions)[1]
        angstrom = [[x * core.BOHR_TO_ANGSTROM for x in p] for p in self.positions]
        converted = core._normalize_inputs(self.numbers, positions_angstrom=angstrom)[1]
        for a, b in zip(bohr, converted):
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y, delta=1e-14)
        expected = hf_rs.calc_hf_nuclear_derivative(
            self.numbers, self.positions, directions=[self.direction] * 3, tol=1e-13
        )
        actual = hf_rs.calc_hf_nuclear_derivative(
            self.numbers,
            positions_angstrom=angstrom,
            directions=[self.direction] * 3,
            tol=1e-13,
        )
        self.assertAlmostEqual(actual, expected, delta=1e-10)
        with self.assertRaises(ValueError):
            core._normalize_inputs(self.numbers, self.positions, angstrom)

    def test_invalid_directions_and_nonconvergence_raise(self):
        for directions in ([], [self.direction] * 5, [[1.0]], [[float("nan")] * 6]):
            with self.assertRaises(ValueError):
                hf_rs.calc_hf_nuclear_derivative(
                    self.numbers, self.positions, directions=directions
                )
        for function in (
            hf_rs.calc_hessian,
            hf_rs.calc_third_derivatives,
            hf_rs.calc_fourth_derivatives,
        ):
            with self.assertRaises(RuntimeError):
                function(self.numbers, self.positions, max_iter=0)

    def test_named_def2_basis_attaches_iodine_ecp(self):
        reference = json.loads(
            (Path(__file__).parent / "data/ecp_reference.json").read_text()
        )["systems"]["hi"]
        energy, gradient = hf_rs.calc_energy_and_gradient(
            [1, 53],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
            basis="def2-svp",
            max_iter=150,
            tol=1e-12,
            conv_algo="cdiis",
        )
        self.assertAlmostEqual(energy, reference["energy"], delta=1e-8)
        for actual, expected in zip(gradient, reference["gradient"]):
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
