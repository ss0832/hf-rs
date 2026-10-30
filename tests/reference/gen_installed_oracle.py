# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Build the installed CLI oracle exclusively from committed independent data."""

import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data = root / "tests/data"
scf = json.loads((data / "scf_reference.json").read_text())
mp = json.loads((data / "scs_mp3_reference.json").read_text())
cc = json.loads((data / "ccsd_triples_reference.json").read_text())
cases = [
    dict(
        name="H2/STO-3G",
        method="hf",
        atomic_numbers=[1, 1],
        positions=[[0, 0, -0.7], [0, 0, 0.7]],
        basis="sto-3g",
        energy=scf["systems"]["h2"]["results"]["sto-3g"]["energy"],
        tolerance=1e-9,
    )
]
water = mp["cases"][0]
geometry = [position for _, position in water["atoms"]]
for method, energy in [
    ("mp2", water["e_scf"] + water["e_mp2"]),
    ("scs-mp3", water["e_scs_total"]),
    ("ccsd(t)", cc["e_scf"] + cc["e_ccsd"] + cc["e_triples"]),
]:
    cases.append(
        dict(
            name="Water/STO-3G " + method,
            method=method,
            atomic_numbers=[8, 1, 1],
            positions=geometry,
            basis="sto-3g",
            energy=energy,
            tolerance=1e-8,
        )
    )
destination = root / "src/validation/oracle.json"
destination.parent.mkdir(exist_ok=True)
destination.write_text(
    json.dumps(
        dict(
            source_files=[
                "scf_reference.json",
                "scs_mp3_reference.json",
                "ccsd_triples_reference.json",
            ],
            cases=cases,
        ),
        indent=2,
    )
    + "\n"
)
