# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Every nuclear component from independent five-point CCSD-F12 energies.

Every displacement rebuilds Psi4 AO operators, PySCF RHF/CABS and the modified
CCSD amplitudes. The two atoms' derivatives are calculated separately.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
reference = json.loads((root / "tests/data/ccsd_f12_reference.json").read_text())
positions = np.array(reference["positions"])
work = root / "target/ccsd-f12-gradient-reference"
work.mkdir(parents=True, exist_ok=True)
h = 0.0005
cases = [
    dict(
        nonlocal_ecp=c["nonlocal_ecp"],
        frozen_core=c["frozen_core"],
        gradient_a=[],
        gradient_b=[],
    )
    for c in reference["cases"]
]
for atom in range(len(positions)):
    for axis in range(3):
        values = []
        for multiple in [-2, -1, 1, 2]:
            point = positions.copy()
            point[atom, axis] += multiple * h
            output = work / "energy.json"
            environment = dict(
                os.environ,
                HF_RS_F12_REFERENCE_POSITIONS=json.dumps(point.tolist()),
                HF_RS_F12_REFERENCE_OUTPUT=str(work / "projected.json"),
                HF_RS_CCSD_F12_REFERENCE_OUTPUT=str(output),
            )
            with (work / f"point-{atom}-{axis}-{multiple}.log").open("w") as log:
                subprocess.run(
                    [sys.executable, str(Path(__file__).with_name("gen_ccsd_f12.py"))],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            values.append(json.loads(output.read_text())["cases"])
        for index, case in enumerate(cases):
            for variant in ["a", "b"]:
                energy = [point[index][f"e_total_{variant}"] for point in values]
                derivative = (energy[0] - 8 * energy[1] + 8 * energy[2] - energy[3]) / (
                    12 * h
                )
                case[f"gradient_{variant}"].append(float(derivative))
        print("CCSD(T*)-F12 nuclear component", atom, axis, "complete", flush=True)
for case in cases:
    for variant in ["a", "b"]:
        case[f"gradient_{variant}"] = (
            np.array(case[f"gradient_{variant}"]).reshape(-1, 3).tolist()
        )
        assert np.max(np.abs(np.sum(case[f"gradient_{variant}"], axis=0))) < 1e-9
(root / "tests/data/ccsd_f12_gradient_reference.json").write_text(
    json.dumps(
        dict(
            source=reference["source"]
            + "; five-point total energies with independent atom displacements",
            includes_scaled_triples=True,
            h_bohr=h,
            positions=positions.tolist(),
            cases=cases,
        ),
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
