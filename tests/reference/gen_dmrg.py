# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent PySCF full-CI and block2 SZ same-M/schedule DMRG oracles."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from pyscf import ao2mo, fci, gto, lib, scf

from embedded_basis import embedded_basis

lib.num_threads(4)


def generic(n, na, nb, dimensions):
    h = np.empty((n, n))
    factors = np.empty((3, n, n))
    for p in range(n):
        for q in range(n):
            h[p, q] = -1.2 + 0.31 * p if p == q else 0.17 * np.sin(p + q + 1)
            pair = max(p, q) * (max(p, q) + 1) // 2 + min(p, q)
            for k in range(3):
                factors[k, p, q] = np.cos(pair + 2 * k + 1)
    eri = 0.08 * np.einsum("kpq,krs->pqrs", factors, factors)
    return pack(f"generic-{n}-{na}-{nb}", h, eri, na, nb, 0.37, dimensions)


def pack(name, h, eri, na, nb, constant, dimensions):
    n = len(h)
    energy, ci = fci.direct_spin1.kernel(
        h, eri, n, (na, nb), ecore=constant, tol=1e-14, max_cycle=300
    )
    one, two = fci.direct_spin1.make_rdm12(ci, n, (na, nb))
    return dict(
        name=name,
        n=n,
        n_alpha=na,
        n_beta=nb,
        constant=float(constant),
        h=h.tolist(),
        eri=eri.tolist(),
        fci_energy=float(energy),
        fci_one_rdm=one.ravel().tolist(),
        fci_two_rdm=two.ravel().tolist(),
        bond_dimensions=dimensions,
        sweeps=40,
        one_site_sweeps=500,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block2-python", required=True)
    args = parser.parse_args()
    cases = [generic(3, 2, 1, [4, 16]), generic(4, 2, 2, [4, 8, 16])]
    n = 6
    h = np.diag(np.linspace(-0.07, 0.09, n))
    for p in range(n - 1):
        h[p, p + 1] = h[p + 1, p] = -1.0
    eri = np.zeros((n,) * 4)
    for p in range(n):
        eri[p, p, p, p] = 2.0
    cases.append(pack("Hubbard-6-U2", h, eri, 3, 3, 0.0, [8, 64]))
    mol = gto.M(
        atom=[("H", (0.03 * i, 0.02 * i * i, 1.6 * i)) for i in range(4)],
        basis=embedded_basis("STO_3G_GBS", ["H"]),
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-12)
    assert mf.converged
    c = mf.mo_coeff
    cases.append(
        pack(
            "H4-STO3G",
            c.T @ mf.get_hcore() @ c,
            ao2mo.kernel(mol, c, compact=False).reshape((4,) * 4),
            2,
            2,
            mol.energy_nuc(),
            [8, 16],
        )
    )
    root = Path(__file__).resolve().parents[2]
    inputs = root / "target/dmrg-oracle-input.json"
    inputs.write_text(json.dumps(cases) + "\n", encoding="utf-8")
    subprocess.run(
        [
            args.block2_python,
            str(Path(__file__).with_name("gen_dmrg_block2.py")),
            str(inputs),
            str(root / "tests/data/dmrg_reference.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
