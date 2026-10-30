# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""External-program oracle: Psi4 scalar-relativistic DKH (dkh_order 2) and X2C RHF energies.

The exact embedded Gaussian 94 basis data is supplied to Psi4 for both the orbital and the
relativistic basis (no decontraction), Cartesian functions, PK integrals and tight convergence.
Only the public input/output interface is used. Black-box experiments show that Psi4's DKH backend
uses the speed of light 137.0359895 a.u. and its X2C the CODATA 2014 value 137.035999139 a.u.; the
hf-rs comparisons pass these values explicitly. An independent NumPy DKH2 at c = 137.0359895
reproduces the DKH energies to 1e-11 Eh.
"""

import json
import re
from pathlib import Path

import psi4

PSI4_DKH_SPEED_OF_LIGHT = 137.0359895
PSI4_X2C_SPEED_OF_LIGHT = 137.035999139

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ("Ne", [("Ne", [0.1, -0.2, 0.3])], "STO_3G_GBS"),
    ("Ar", [("Ar", [0.1, -0.2, 0.3])], "STO_3G_GBS"),
    (
        "water",
        [("O", [0.1, -0.2, 0.3]), ("H", [1.53, 0.68, 0.3]), ("H", [-1.33, 0.68, 0.3])],
        "BASIS_6_31G_GBS",
    ),
    ("HCl", [("Cl", [0.1, -0.2, 0.3]), ("H", [0.1, -0.2, 2.7])], "BASIS_6_31G_GBS"),
    ("Kr", [("Kr", [0.0, 0.0, 0.0])], "STO_3G_GBS"),
]


def basis_block(constant, symbols):
    text = (ROOT / "src/basis_data.rs").read_text(encoding="utf-8")
    gbs = re.search(rf'pub const {constant}: &str = r#"(.*?)"#;', text, re.S).group(1)
    parts = ["cartesian", "****"]
    for symbol in symbols:
        segment = re.search(rf"(?m)^{symbol}\s+0\s*\n(.*?)\*\*\*\*", gbs, re.S).group(1)
        parts.append(f"{symbol} 0\n{segment.strip()}\n****")
    return "\n".join(parts)


def main():
    psi4.core.be_quiet()
    psi4.set_memory("2 GB")
    psi4.set_num_threads(4)
    out = []
    for name, atoms, constant in CASES:
        psi4.core.clean()
        psi4.geometry(
            "units bohr\nsymmetry c1\nno_reorient\nno_com\n"
            + "\n".join(f"{s} {x[0]} {x[1]} {x[2]}" for s, x in atoms)
        )
        block = basis_block(constant, sorted({a[0] for a in atoms}))
        psi4.basis_helper(block, name="hfrsorb", key="BASIS")
        psi4.basis_helper(block, name="hfrsrel", key="BASIS_RELATIVISTIC")
        energies = {}
        for rel in ("no", "dkh", "x2c"):
            psi4.set_options(
                {
                    "relativistic": rel,
                    "dkh_order": 2,
                    "scf_type": "pk",
                    "puream": False,
                    "reference": "rhf",
                    "e_convergence": 1e-12,
                    "d_convergence": 1e-10,
                    "ints_tolerance": 1e-14,
                    "guess": "sad",
                    "maxiter": 500,
                }
            )
            energies[rel] = float(psi4.energy("scf"))
        out.append(
            dict(
                name=name,
                e_nonrelativistic=energies["no"],
                e_dkh2=energies["dkh"],
                e_x2c=energies["x2c"],
            )
        )
        print(name, energies, flush=True)
    (ROOT / "tests/data/dkh2_psi4_reference.json").write_text(
        json.dumps(
            dict(
                source=__doc__,
                psi4_version=psi4.__version__,
                dkh_speed_of_light=PSI4_DKH_SPEED_OF_LIGHT,
                x2c_speed_of_light=PSI4_X2C_SPEED_OF_LIGHT,
                cases=out,
            ),
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
