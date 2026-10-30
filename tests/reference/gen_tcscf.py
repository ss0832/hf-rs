# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""TCSCF oracle: PySCF singlet CASSCF(2,2) with the selected RHF HOMO/LUMO pair as the active
space (TCSCF = one-pair GVB = CASSCF(2,2) in natural orbitals), tight orbital convergence,
active natural occupations and PySCF analytic CASSCF nuclear gradients.
Contracted Cartesian AO space, embedded basis data.
"""

import json
from pathlib import Path

import numpy as np
from pyscf import __version__ as pyscf_version
from pyscf import gto, lib, mcscf, scf

from embedded_basis import embedded_basis

lib.num_threads(4)

MODELS = [
    (
        "H2_stretched",
        [("H", [0.0, 0.0, -1.5]), ("H", [0.0, 0.0, 1.5])],
        "STO_3G_GBS",
        None,
    ),
    ("LiH", [("Li", [0.0, 0.0, 0.0]), ("H", [0.13, 0.21, 3.2])], "STO_3G_GBS", None),
    # stretched HF: at 5 bohr the HOMO/LUMO pair is sigma/sigma*
    (
        "HF_stretched",
        [("F", [0.1, -0.2, 0.3]), ("H", [0.17, -0.11, 5.0])],
        "BASIS_6_31G_GBS",
        None,
    ),
    (
        "twisted_ethylene",
        [
            ("C", [0.0, 0.0, 1.266]),
            ("C", [0.0, 0.0, -1.266]),
            ("H", [0.0, 1.744, 2.336]),
            ("H", [0.0, -1.744, 2.336]),
            ("H", [1.744, 0.0, -2.336]),
            ("H", [-1.744, 0.0, -2.336]),
        ],
        "STO_3G_GBS",
        None,
    ),
]


def main():
    out = []
    for name, atoms, constant, pair in MODELS:
        mol = gto.M(
            atom=atoms,
            basis=embedded_basis(constant, sorted({a[0] for a in atoms})),
            cart=True,
            unit="Bohr",
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-10
        mf.max_cycle = 300
        mf.kernel()
        if (
            not mf.converged
        ):  # near-degenerate pi/pi* (twisted ethylene): second-order SCF
            mf = scf.newton(mf)
            mf.conv_tol = 1e-12
            mf.kernel()
        assert mf.converged
        nocc = mol.nelectron // 2
        mc = mcscf.CASSCF(mf, 2, 2)
        mc.fix_spin_(ss=0)
        mc.conv_tol = 1e-12
        mc.conv_tol_grad = 1e-6
        mc.max_cycle_macro = 200
        bonding, antibonding = pair if pair else (nocc - 1, nocc)
        mo = mc.sort_mo([bonding + 1, antibonding + 1])  # 1-based indices
        mc.kernel(mo)
        assert mc.converged
        occupations = np.sort(np.linalg.eigvalsh(mc.fcisolver.make_rdm1(mc.ci, 2, 2)))[
            ::-1
        ]
        gradient = mc.nuc_grad_method().kernel()
        out.append(
            dict(
                name=name,
                basis_constant=constant,
                atoms=[dict(symbol=s, position=r) for s, r in atoms],
                bonding=bonding,
                antibonding=antibonding,
                e_rhf=float(mf.e_tot),
                e_tcscf=float(mc.e_tot),
                natural_occupations=occupations.tolist(),
                gradient=gradient.tolist(),
            )
        )
        print(name, mc.e_tot, occupations, flush=True)
    path = Path(__file__).resolve().parents[1] / "data/tcscf_reference.json"
    path.write_text(
        json.dumps(
            dict(source=__doc__, pyscf_version=pyscf_version, cases=out), indent=1
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
