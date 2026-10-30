# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent full Psi4 SCS-MP3 energies with exact embedded orbital data."""

import json
import os
import re
from pathlib import Path

import psi4

root = Path(__file__).resolve().parents[2]
work = root / "target/psi4-scs-mp3-reference"
work.mkdir(parents=True, exist_ok=True)
os.chdir(work)
g94 = re.search(
    r'pub const STO_3G_GBS: &str = r#"(.*?)"#;',
    (root / "src/basis_data.rs").read_text(encoding="utf-8"),
    re.S,
).group(1)
(work / "hf_rs_obs.gbs").write_text("cartesian\n****\n" + g94, encoding="utf-8")
psi4.set_num_threads(4)
psi4.set_memory("512 MiB")
cases = []
for name, atoms, multiplicity in [
    (
        "water",
        [
            ("O", [0.13, -0.21, 0.31]),
            ("H", [0.23, 1.22, -0.77]),
            ("H", [-0.08, -1.64, -0.83]),
        ],
        1,
    ),
    ("lih", [("Li", [0.1, -0.2, 0.3]), ("H", [0.4, 0.1, 3.2])], 1),
    ("oh", [("O", [0.13, -0.21, 0.31]), ("H", [0.23, 0.12, 2.11])], 2),
    (
        "ch2",
        [("C", [0.1, -0.2, 0.3]), ("H", [0.4, 0.1, 2.1]), ("H", [1.7, 0.3, -0.8])],
        3,
    ),
]:
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.core.set_output_file(str(work / f"{name}.out"), False)
    mol = psi4.geometry(
        f"0 {multiplicity}\n"
        + "\n".join(f"{symbol} {p[0]} {p[1]} {p[2]}" for symbol, p in atoms)
        + "\nunits bohr\nsymmetry c1\nno_com\nno_reorient"
    )
    psi4.set_options(
        dict(
            basis="hf_rs_obs",
            reference="uhf" if multiplicity > 1 else "rhf",
            scf_type="pk",
            mp_type="conv",
            freeze_core=False,
            puream=False,
            e_convergence=1e-13,
            d_convergence=1e-12,
        )
    )
    energy = psi4.energy("scs-mp3", molecule=mol)
    variables = psi4.core.variables()
    cases.append(
        dict(
            name=name,
            atoms=atoms,
            multiplicity=multiplicity,
            e_scf=variables["HF TOTAL ENERGY"],
            e_mp2=variables["MP2 CORRELATION ENERGY"],
            e_mp3=variables["MP3 CORRELATION ENERGY"]
            - variables["MP2 CORRELATION ENERGY"],
            e_same=variables["MP2 SAME-SPIN CORRELATION ENERGY"],
            e_opposite=variables["MP2 OPPOSITE-SPIN CORRELATION ENERGY"],
            e_scs_total=energy,
        )
    )
    print(name, energy, flush=True)
(root / "tests/data/scs_mp3_reference.json").write_text(
    json.dumps(
        dict(
            source=f"Psi4 {psi4.__version__} scs-mp3, conventional integrals, no frozen core, exact embedded STO-3G",
            cases=cases,
        ),
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
