"""Fractional atomic HF reference from PySCF's spherical atomic SCF solver."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, lib
from pyscf.scf.atom_hf import AtomHF1e, AtomSphAverageRHF

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)
out = {
    "source": f"PySCF {pyscf.__version__}, AtomSphAverageRHF (AtomHF1e for H), embedded STO-3G, conv_tol=1e-13",
    "basis_source_sha256": basis_hash(),
    "atoms": {},
}
for symbol in ["H", "C", "N", "O"]:
    z = gto.charge(symbol)
    mol = gto.M(
        atom=f"{symbol} 0 0 0",
        basis=embedded_basis("STO_3G_GBS", [symbol]),
        spin=z % 2,
        cart=False,
        verbose=0,
    )
    mf = AtomHF1e(mol) if z == 1 else AtomSphAverageRHF(mol)
    mf.conv_tol = 1e-13
    mf.kernel()
    assert mf.converged
    density = mf.make_rdm1()
    if np.ndim(density) == 3:
        density = np.sum(density, axis=0)
    occupations = mf.mo_occ
    if np.ndim(occupations) == 2:
        occupations = np.sum(occupations, axis=0)
    out["atoms"][symbol] = {
        "energy": mf.e_tot,
        "density": density.tolist(),
        "occupations": occupations.tolist(),
    }
Path(__file__).resolve().parents[1].joinpath("data/sad_reference.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
