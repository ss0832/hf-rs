# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

from hf_rs import Molecule, run

positions = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
print(run("mp2", [1, 1], positions)["e_total"])
molecule = Molecule([1, 1], positions)
print(molecule.gradient())
