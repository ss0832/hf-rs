# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Query Psi4's actual high-precision MintsHelper Slater/Gaussian fit."""

import json
import os
from pathlib import Path

import psi4

root = Path(__file__).resolve().parents[2]
work = root / "target/psi4-fit-oracle"
work.mkdir(parents=True, exist_ok=True)
os.chdir(work)  # Psi4 writes timer.dat when the process exits.
psi4.core.be_quiet()
molecule = psi4.geometry("0 1\nHe 0 0 0\nunits bohr\nsymmetry c1")
basis = psi4.core.BasisSet.build(molecule, "BASIS", "STO-3G")
mints = psi4.core.MintsHelper(basis)
cases = []
for beta in [0.7, 1.0, 1.4]:
    terms = mints.f12_cgtg(beta)
    cases.append(dict(beta=beta, normalized_terms=[(a, c / beta) for a, c in terms]))
output = dict(
    source=f"Psi4 {psi4.__version__} MintsHelper.f12_cgtg; coefficients divided by beta for -exp(-beta*r)/beta",
    cases=cases,
)
(root / "tests/data/f12_slater_fit_reference.json").write_text(
    json.dumps(output, indent=1) + "\n", encoding="utf-8"
)
print(output)
