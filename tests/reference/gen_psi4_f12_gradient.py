# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent five-point MP2-F12 total-energy gradients from full Psi4 runs."""

import json
import os
from pathlib import Path

import numpy as np
import psi4

root = Path(__file__).resolve().parents[2]
work = root / "target/psi4-f12-gradient-reference"
work.mkdir(parents=True, exist_ok=True)
os.chdir(work)
psi4.set_num_threads(4)
psi4.set_memory("512 MiB")
obs = "He 0\nS 1 1.0\n1.4 1.0\nS 1 1.0\n0.45 1.0\n****\n"
aux = "He 0\nS 1 1.0\n0.2 1.0\nS 1 1.0\n0.9 1.0\nS 1 1.0\n3.2 1.0\nP 1 1.0\n0.8 1.0\n****\n"
for name, content in [("gradient_obs", obs), ("gradient_aux", aux)]:
    (work / f"{name}.gbs").write_text("cartesian\n****\n" + content, encoding="utf-8")
positions = np.array([[-0.1, 0.2, -1.3], [0.3, -0.2, 1.4]])


def energy(point, frozen, label):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.core.set_output_file(str(work / f"{label}.out"), False)
    geometry = "\n".join("He " + " ".join(map(str, p)) for p in point)
    mol = psi4.geometry(
        "0 1\n" + geometry + "\nunits bohr\nsymmetry c1\nno_com\nno_reorient"
    )
    psi4.set_options(
        {
            "basis": "gradient_obs",
            "cabs_basis": "gradient_aux",
            "puream": False,
            "reference": "rhf",
            "scf_type": "pk",
            "mp2_type": "conv",
            "freeze_core": False,
            "frozen_docc": [frozen],
            "e_convergence": 1e-13,
            "d_convergence": 1e-13,
            "f12_beta": 1.0,
            "cabs_singles": True,
        }
    )
    total = psi4.energy("mp2-f12", molecule=mol)
    return float(total)


step = 1e-3
cases = []
for frozen in [0, 1]:
    gradient = np.zeros_like(positions)
    coarse = np.zeros_like(positions)
    values = []
    for atom in range(2):
        for axis in range(3):
            samples = {}
            for multiple in [-4, -2, -1, 1, 2, 4]:
                point = positions.copy()
                point[atom, axis] += multiple * step
                samples[multiple] = energy(
                    point, frozen, f"fc{frozen}_{atom}_{axis}_{multiple}"
                )
            gradient[atom, axis] = (
                samples[-2] - 8 * samples[-1] + 8 * samples[1] - samples[2]
            ) / (12 * step)
            coarse[atom, axis] = (
                samples[-4] - 8 * samples[-2] + 8 * samples[2] - samples[4]
            ) / (24 * step)
            values.append(
                dict(
                    atom=atom,
                    axis=axis,
                    energies={str(k): v for k, v in samples.items()},
                )
            )
    refinement = float(np.max(np.abs(gradient - coarse)))
    assert refinement < 1e-8, refinement
    cases.append(
        dict(
            frozen_core=frozen,
            gradient=gradient.tolist(),
            refinement=refinement,
            samples=values,
        )
    )
    print(
        "frozen",
        frozen,
        "gradient",
        gradient.tolist(),
        "refinement",
        refinement,
        flush=True,
    )
out = dict(
    source=f"Psi4 {psi4.__version__}, fresh SCF/CABS/MP2-F12 at all 72 displaced geometries; five-point central differences",
    positions=positions.tolist(),
    obs_g94=obs,
    aux_g94=aux,
    step_bohr=step,
    cases=cases,
)
(root / "tests/data/psi4_f12_gradient_reference.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8"
)
