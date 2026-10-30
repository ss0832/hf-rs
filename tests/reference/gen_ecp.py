# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""PySCF oracle data for effective core potentials (def2 basis sets with Stuttgart-Koeln ECPs).

Per system: RHF/UHF energy and analytical gradient (Cartesian basis functions, BSE basis + ECP data) and the
generalized eigenvalues of the ECP matrix with respect to the overlap matrix, which are invariant to the AO
ordering/phase conventions and therefore directly comparable with hf-rs.

Usage: python gen_ecp.py > tests/data/ecp_reference.json
"""

import json
import os
import tempfile

import basis_set_exchange as bse
import numpy as np
import pyscf
import scipy.linalg
from pyscf import gto, scf

SYSTEMS = {
    "hi": ("H 0 0 0; I 0 0 3.0", 0, 0, "def2-svp"),
    "agh": ("Ag 0 0 0; H 0 0 3.0", 0, 0, "def2-svp"),
    "i2": ("I 0 0 0; I 0 0 5.0", 0, 0, "def2-svp"),
    "auh": ("Au 0 0 0.1; H 0.2 0.1 3.3", 0, 0, "def2-svp"),
    "pbh4": (
        "Pb 0 0 0; H 2.2 2.2 2.2; H 2.2 -2.2 -2.2; H -2.2 2.2 -2.2; H -2.2 -2.2 2.2",
        0,
        0,
        "def2-svp",
    ),
    "agh_cation": ("Ag 0 0 0; H 0 0 3.1", 1, 1, "def2-svp"),
    "hi_tzvp": ("H 0 0 0; I 0 0 3.0", 0, 0, "def2-tzvp"),
}


def load(name, symbols):
    text = bse.get_basis(name, fmt="nwchem", header=False)
    with tempfile.NamedTemporaryFile("w", suffix=".nw", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        basis = {s: gto.basis.load(path, s) for s in symbols}
        ecp = {}
        for s in symbols:
            try:
                e = gto.basis.load_ecp(path, s)
            except Exception:
                e = None
            if e:
                ecp[s] = e
        return basis, ecp
    finally:
        os.unlink(path)


out = {
    "source": f"PySCF {pyscf.__version__}, cart=True, basis+ECP from basis_set_exchange {bse.version()}",
    "systems": {},
}
for key, (atom, charge, spin, basis_name) in SYSTEMS.items():
    symbols = sorted({t.split()[0] for t in atom.split(";")})
    basis, ecp = load(basis_name, symbols)
    mol = gto.M(
        atom=atom,
        unit="bohr",
        basis=basis,
        ecp=ecp,
        cart=True,
        charge=charge,
        spin=spin,
        verbose=0,
    )
    mf = scf.RHF(mol) if spin == 0 else scf.UHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 300
    mf.kernel()
    assert mf.converged, key
    grad = mf.nuc_grad_method().kernel()
    s = mol.intor("int1e_ovlp_cart")
    v = mol.intor("ECPscalar_cart")
    gen_eigs = scipy.linalg.eigh(v, s, eigvals_only=True)
    out["systems"][key] = {
        "geometry_bohr": atom,
        "charge": charge,
        "spin_2s": spin,
        "basis": basis_name,
        "n_ao": mol.nao,
        "n_electrons": int(mol.nelectron),
        "energy": float(mf.e_tot),
        "gradient": grad.tolist(),
        "ecp_generalized_eigenvalues": gen_eigs.tolist(),
        "ecp_frobenius_norm": float(np.linalg.norm(v)),
    }
print(json.dumps(out))
