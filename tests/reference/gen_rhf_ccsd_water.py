# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Public-interface PySCF reference for the spatial CCSD molecular run."""

import json
import re
from pathlib import Path

import pyscf
from pyscf import cc, gto, lib, scf
from pyscf.gto.basis import parse_gaussian

lib.num_threads(4)
root = Path(__file__).resolve().parents[2]
case = json.loads(
    (root / "tests/data/psi4_mp2_f12_published_reference.json").read_text()
)["cases"][0]
defs = {
    s: parse_gaussian.parse(
        re.search(rf"(?m)^\s*{s}\s+0\s*\n(.*?)\*\*\*\*", case["obs_g94"], re.S).group(
            1
        ),
        optimize=False,
    )
    for s in ["H", "O"]
}
mol = gto.M(
    atom=case["atoms"], basis=defs, unit="Bohr", cart=False, max_memory=512, verbose=0
)
mf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-11, max_cycle=200)
if not mf.converged:
    mf = mf.newton().run(conv_tol=1e-13, conv_tol_grad=1e-11, max_cycle=100)
assert mf.converged
solver = cc.CCSD(mf)
solver.conv_tol = 1e-13
solver.conv_tol_normt = 1e-11
solver.max_cycle = 200
solver.kernel()
assert solver.converged
record = dict(
    input_fixture="psi4_mp2_f12_published_reference.json#cases/0",
    spherical=True,
    frozen_core=0,
    source=f"PySCF {pyscf.__version__} public RHF/CCSD execution",
    e_scf=float(mf.e_tot),
    e_corr=float(solver.e_corr),
    e_total=float(solver.e_tot),
)
(root / "tests/data/rhf_ccsd_water_reference.json").write_text(
    json.dumps(record, indent=2) + "\n"
)
print(record)
