# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent oracle for the embedded basis data: per element and shell, the angular momentum,
number of primitives and the exact exponent/coefficient sums, taken from the BSE JSON representation
(not from the Gaussian 94 text hf-rs embeds).

Usage: python gen_basis_manifest.py > tests/data/basis_manifest.json
"""

import json
from pathlib import Path

import basis_set_exchange as bse
from fetch_basis_catalog import ADDITIONAL_ORBITAL

NAMES = {
    "sto-3g": "sto-3g",
    "3-21g": "3-21g",
    "6-31g": "6-31g",
    "6-31g*": "6-31g_st_",
    "6-31g**": "6-31g_st__st_",
    "6-311g**": "6-311g_st__st_",
    "def2-svp": "def2-svp",
    "def2-tzvp": "def2-tzvp",
    "def2-qzvp": "def2-qzvp",
    "cc-pvdz": "cc-pvdz",
    "cc-pvtz": "cc-pvtz",
    "cc-pvqz": "cc-pvqz",
    "aug-cc-pvdz": "aug-cc-pvdz",
    "aug-cc-pvtz": "aug-cc-pvtz",
    "def2-universal-jkfit": "def2-universal-jkfit",
    "def2-svp-rifit": "def2-svp-rifit",
    "def2-tzvp-rifit": "def2-tzvp-rifit",
    "cc-pvdz-rifit": "cc-pvdz-rifit",
    "cc-pvtz-rifit": "cc-pvtz-rifit",
    "aug-cc-pvdz-rifit": "aug-cc-pvdz-rifit",
    "cc-pvdz-f12-optri": "cc-pvdz-f12-optri",
    "cc-pvtz-f12-optri": "cc-pvtz-f12-optri",
}

ecp_manifest = {}
NAMES.update({name: name for name in ADDITIONAL_ORBITAL})
catalog_root = Path(__file__).resolve().parents[2] / "third_party/basis_set_exchange"
catalog = json.loads((catalog_root / "manifest.json").read_text(encoding="utf-8"))
exports = {entry["name"]: entry["json_file"] for entry in catalog["basis_sets"]}
out = {
    "source": f"basis_set_exchange {bse.version()} (JSON representation)",
    "basis_sets": {},
}
for hfname, bsename in NAMES.items():
    data = (
        (catalog_root / "data" / exports[hfname]).read_text(encoding="utf-8")
        if hfname in exports
        else bse.get_basis(bsename, fmt="json")
    )
    data = json.loads(data) if isinstance(data, str) else data
    entry = {}
    for z, el in sorted(data["elements"].items(), key=lambda kv: int(kv[0])):
        if hfname.startswith("def2-") and "ecp_potentials" in el:
            ecp_manifest[z] = {
                "n_core": int(el["ecp_electrons"]),
                "l_max": max(max(p["angular_momentum"]) for p in el["ecp_potentials"]),
                "channels": {
                    str(p["angular_momentum"][0]): [
                        len(p["gaussian_exponents"]),
                        sum(float(x) for x in p["gaussian_exponents"]),
                        sum(float(x) for x in p["coefficients"][0]),
                        sum(int(x) for x in p["r_exponents"]),
                    ]
                    for p in el["ecp_potentials"]
                },
            }
        shells = []
        for sh in el["electron_shells"]:
            ams, rows = sh["angular_momentum"], sh["coefficients"]
            exps = [float(x) for x in sh["exponents"]]
            pairs = (
                list(zip(ams, rows)) if len(ams) > 1 else [(ams[0], r) for r in rows]
            )
            for angular, row in pairs:
                nonzero = [(e, float(c)) for e, c in zip(exps, row) if float(c) != 0.0]
                shells.append(
                    [
                        angular,
                        len(nonzero),
                        sum(e for e, _ in nonzero),
                        sum(abs(c) for _, c in nonzero),
                    ]
                )
        entry[z] = shells
    out["basis_sets"][hfname] = entry
out["ecp_def2"] = (
    ecp_manifest  # identical for def2-SVP/TZVP/QZVP (checked by gen_basis_data.py)
)
print(json.dumps(out))
