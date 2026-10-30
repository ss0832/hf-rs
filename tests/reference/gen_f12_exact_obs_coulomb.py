# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent exact-OBS Coulomb projector models (Knizia 2010, Sec. 4.5.1).

Public Psi4 operators and an explicit product-space block replacement precede
the independent Gram-Hamiltonian projection and public PySCF CC iteration.
Conventional MP2/Schur G remains independently exact or fitted. These cases
do not establish any external program's historical default conventions.
"""

import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
work = root / "target/f12-exact-obs-coulomb-reference"
work.mkdir(parents=True, exist_ok=True)
models = []
for index, flags in enumerate(
    [
        [],
        ["--fit-orbital-mp2"],
        ["--mixed-fock", "--spherical-fitting", "--spherical-fock-fitting"],
        [
            "--mixed-fock",
            "--spherical-fitting",
            "--spherical-fock-fitting",
            "--fit-orbital-mp2",
        ],
    ]
):
    path = work / f"model-{index}.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("gen_f12_fitted_molecular.py")),
            "--exact-obs-coulomb",
            "--output",
            str(path),
        ]
        + flags,
        check=True,
    )
    models.append(json.loads(path.read_text(encoding="utf-8")))
(root / "tests/data/f12_exact_obs_coulomb_reference.json").write_text(
    json.dumps(
        dict(
            source="Public Psi4 lower-center/exact OBS integrals and public PySCF CCSD iteration; independent orthogonal product-space Coulomb block replacement",
            scope="Exact OBS Coulomb projector with independently exact/DF MP2, Cartesian/pure correction fits and exact/mixed Fock; not a published/default certification",
            models=models,
        ),
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
