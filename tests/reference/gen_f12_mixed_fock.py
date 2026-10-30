# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent mixed-Fock molecular F12 models with separate physical fits."""

import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
work = root / "target/f12-mixed-fock-reference"
work.mkdir(parents=True, exist_ok=True)
models = []
for index, flags in enumerate(
    [
        [],
        ["--spherical-fitting", "--spherical-fock-fitting", "--fit-orbital-mp2"],
        ["--spherical-fitting"],
        ["--spherical-fock-fitting", "--fit-orbital-mp2"],
        ["--spherical-fock-fitting", "--exact-correction"],
    ]
):
    path = work / f"model-{index}.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("gen_f12_fitted_molecular.py")),
            "--mixed-fock",
            "--output",
            str(path),
        ]
        + flags,
        check=True,
    )
    models.append(json.loads(path.read_text(encoding="utf-8")))
(root / "tests/data/f12_mixed_fock_reference.json").write_text(
    json.dumps(
        dict(
            source="Public Psi4 lower-center arrays and public PySCF CCSD iteration; independent product/geminal Gram-Hamiltonian with a finite-rank exact-OBS Fock restoration",
            scope="Exact CCSD/(T), independent correction/Fock fitting bases and physical spaces, exact/DF MP2; no published/default agreement claimed",
            models=models,
        ),
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
