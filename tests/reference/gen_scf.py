# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""PySCF reference values for SCF energies, gradients and Hessians (Cartesian basis functions).

Usage: python gen_scf.py > tests/data/scf_reference.json
"""

import json
import os
import tempfile

import basis_set_exchange as bse
import pyscf
from pyscf import gto, scf

SYSTEMS = {
    "h2": ("H 0 0 -0.7; H 0 0 0.7", 0, 0),
    "lih": ("Li 0 0 0; H 0 0 3.014", 0, 0),
    "h2o": ("O 0 0 0.2217; H 0 1.4309 -0.8867; H 0 -1.4309 -0.8867", 0, 0),
    "oh": ("O 0 0 0; H 0 0 1.8324", 0, 1),
    "nh3": (
        "N 0 0 0.2128; H 0 1.7703 -0.4966; H 1.5331 -0.8851 -0.4966; H -1.5331 -0.8851 -0.4966",
        0,
        0,
    ),
}
BASES = {"sto-3g": "sto-3g", "6-31g*": "6-31g_st_"}  # hf-rs name -> BSE name


def bse_basis(bse_name, symbols):
    """Loads the Basis Set Exchange definition (the same data hf-rs embeds) into PySCF."""
    text = bse.get_basis(bse_name, fmt="nwchem", header=False)
    with tempfile.NamedTemporaryFile("w", suffix=".nw", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        return {sym: gto.basis.load(path, sym) for sym in symbols}
    finally:
        os.unlink(path)


def run(atom, charge, spin, basis_name):
    symbols = sorted({tok.split()[0] for tok in atom.split(";")})
    basis = bse_basis(BASES[basis_name], symbols)
    mol = gto.M(
        atom=atom,
        unit="bohr",
        basis=basis,
        cart=True,
        charge=charge,
        spin=spin,
        verbose=0,
    )
    mf = scf.RHF(mol) if spin == 0 else scf.UHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    assert mf.converged
    out = {
        "reference": "rhf" if spin == 0 else "uhf",
        "n_ao": mol.nao,
        "energy": mf.e_tot,
        "gradient": mf.nuc_grad_method().kernel().tolist(),
    }
    h = mf.Hessian().kernel()  # (natm, natm, 3, 3)
    n = mol.natm
    out["hessian"] = [
        [float(h[i // 3, j // 3, i % 3, j % 3]) for j in range(3 * n)]
        for i in range(3 * n)
    ]
    return out


data = {
    "source": f"PySCF {pyscf.__version__}, cart=True, conv_tol=1e-13, basis from basis_set_exchange {bse.version()}",
    "systems": {},
}
for name, (atom, charge, spin) in SYSTEMS.items():
    data["systems"][name] = {
        "geometry_bohr": atom,
        "charge": charge,
        "spin_2s": spin,
        "results": {},
    }
    for basis in BASES:
        data["systems"][name]["results"][basis] = run(atom, charge, spin, basis)
print(json.dumps(data, indent=1))
