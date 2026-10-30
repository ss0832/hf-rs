# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent RHF TDA (= spin-adapted CIS) roots and transition properties."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, lib, scf, tdscf

from embedded_basis import embedded_basis

lib.num_threads(4)
cases = []
for name, atoms in [
    (
        "water",
        [
            ("O", [0.13, -0.21, 0.31]),
            ("H", [0.23, 1.22, -0.77]),
            ("H", [-0.08, -1.64, -0.83]),
        ],
    ),
    ("lih", [("Li", [0.1, -0.2, 0.3]), ("H", [0.4, 0.1, 3.2])]),
]:
    mol = gto.M(
        atom=atoms,
        unit="Bohr",
        basis=embedded_basis("STO_3G_GBS", {s for s, _ in atoms}),
        cart=True,
        verbose=0,
    )
    hf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-10, max_cycle=200)
    continuation = hf.newton()
    continuation.conv_tol, continuation.conv_tol_grad = 1e-13, 1e-10
    continuation.kernel(hf.mo_coeff, hf.mo_occ)
    hf = continuation.undo_soscf()
    assert hf.converged
    for triplet in [False, True]:
        solver = tdscf.TDA(hf)
        solver.singlet = not triplet
        solver.nstates = 4
        no = mol.nelectron // 2
        nv = hf.mo_coeff.shape[1] - no
        vind, _ = solver.gen_vind()
        response = vind(np.eye(no * nv)).T
        energies, vectors = np.linalg.eigh(response)
        assert np.max(np.abs(response @ vectors - vectors * energies)) < 1e-12
        solver.e = energies[: solver.nstates]
        solver.xy = [
            (vectors[:, i].reshape(no, nv) / np.sqrt(2), 0)
            for i in range(solver.nstates)
        ]
        cases.append(
            dict(
                name=name,
                atoms=atoms,
                triplet=triplet,
                e_scf=hf.e_tot,
                mo_coefficients=hf.mo_coeff.tolist(),
                energies=solver.e.tolist(),
                amplitudes=[(np.sqrt(2) * xy[0]).ravel().tolist() for xy in solver.xy],
                transition_dipoles=solver.transition_dipole().tolist(),
                oscillator_strengths=solver.oscillator_strength().tolist(),
            )
        )
Path(__file__).resolve().parents[1].joinpath("data/cis_reference.json").write_text(
    json.dumps(
        dict(
            source=f"PySCF {pyscf.__version__} spin-adapted RHF TDA, exact embedded STO-3G",
            cases=cases,
        ),
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
