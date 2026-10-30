"""Canonical RHF CCSD(T) regression with a nonzero triples contribution."""

import json
from pathlib import Path

import pyscf
from pyscf import cc, gto, lib, scf

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)
mol = gto.M(
    atom=[
        ("O", (0.13, -0.21, 0.31)),
        ("H", (0.23, 1.22, -0.77)),
        ("H", (-0.08, -1.64, -0.83)),
    ],
    basis=embedded_basis("STO_3G_GBS", ["O", "H"]),
    unit="Bohr",
    cart=True,
    verbose=0,
)
mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-10)
solver = cc.CCSD(mf)
solver.conv_tol = 1e-13
solver.conv_tol_normt = 1e-11
solver.max_cycle = 300
solver.kernel()
out = dict(
    source=f"PySCF {pyscf.__version__} RHF/CCSD(T), cart=True, converged nonzero triples",
    basis_source_sha256=basis_hash(),
    e_scf=mf.e_tot,
    e_ccsd=solver.e_corr,
    e_triples=solver.ccsd_t(),
)
Path(__file__).resolve().parents[1].joinpath(
    "data/ccsd_triples_reference.json"
).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(out)
