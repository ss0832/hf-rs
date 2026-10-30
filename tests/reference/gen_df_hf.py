# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent Coulomb-metric RI-JK SCF and every Cartesian J/K/F/P component."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import ao2mo, gto, lib, scf

from embedded_basis import embedded_basis

lib.num_threads(4)
cases = []
he_obs = [[0, [1.4, 1.0]], [0, [0.45, 1.0]]]
he_aux = [
    [0, [0.2, 1.0]],
    [0, [0.9, 1.0]],
    [0, [3.2, 1.0]],
    [1, [0.8, 1.0]],
    [2, [0.8, 1.0]],
]
for name in ["water", "he_ecp", "he_rank_drop", "he_spherical_d"]:
    spherical = name == "he_spherical_d"
    if name == "water":
        atoms = [
            ("O", [0.13, -0.21, 0.31]),
            ("H", [0.23, 1.22, -0.77]),
            ("H", [-0.08, -1.64, -0.83]),
        ]
        basis = embedded_basis("STO_3G_GBS", ["O", "H"])
        auxiliary = embedded_basis("AUX_DEF2_SVP_RIFIT_GBS", ["O", "H"])
        ecp = {}
    else:
        atoms = [("He", [0.13, -0.21, 0.31])]
        obs = (
            he_obs
            + ([he_obs[0]] if name == "he_rank_drop" else [])
            + ([[2, [0.65, 1.0]]] if spherical else [])
        )
        basis = {"He": obs}
        auxiliary = {"He": he_aux + ([he_aux[0]] if name == "he_rank_drop" else [])}
        ecp = (
            {
                "He": [
                    0,
                    [
                        [-1, [[], [], [[0.7, 0.17]]]],
                        [0, [[], [], [[0.53, 0.11]]]],
                        [1, [[], [], [[0.61, -0.07]]]],
                    ],
                ]
            }
            if name == "he_ecp"
            else {}
        )
    mol = gto.M(atom=atoms, basis=basis, ecp=ecp, unit="Bohr", cart=True, verbose=0)
    hf = scf.RHF(mol).density_fit(auxbasis=auxiliary)
    overlap = mol.intor("int1e_ovlp")
    transform = mol.cart2sph_coeff() if spherical else np.eye(mol.nao)
    values, vectors = np.linalg.eigh(transform.T @ overlap @ transform)
    ortho = transform @ (vectors[:, values > 1e-8] / np.sqrt(values[values > 1e-8]))

    def eig(h, s, **kwargs):
        values, vectors = np.linalg.eigh(ortho.T @ h @ ortho)
        return values, ortho @ vectors

    hf.eig = eig
    hf.diis = scf.diis.CDIIS(hf)
    hf.diis.Corth = ortho
    hf.init_guess = "1e"
    hf.conv_tol, hf.conv_tol_grad, hf.max_cycle = 1e-13, 1e-10, 300
    hf.kernel()
    print(
        name,
        "SCF",
        hf.converged,
        "energy",
        hf.e_tot,
        "orbital gradient",
        np.linalg.norm(hf.get_grad(hf.mo_coeff, hf.mo_occ)),
        flush=True,
    )
    assert hf.converged, name
    dm = hf.make_rdm1()
    j, k = hf.get_jk(dm=dm)
    f = hf.get_hcore() + j - 0.5 * k
    scale = np.sqrt(overlap.diagonal())
    order = []
    offset = 0
    for shell in range(mol.nbas):
        angular = mol.bas_angular(shell)
        ncart = (angular + 1) * (angular + 2) // 2
        for _ in range(mol.bas_nctr(shell)):
            order.extend(
                [
                    offset + i
                    for i in ([0, 3, 5, 1, 2, 4] if angular == 2 else range(ncart))
                ]
            )
            offset += ncart

    def normalized(operator):
        return (operator / scale[:, None] / scale[None, :])[
            np.ix_(order, order)
        ].tolist()

    exact = mol.intor("int2e")
    fitted = ao2mo.restore(1, hf.with_df.get_eri(), mol.nao)
    error = (
        (exact - fitted)
        / scale[:, None, None, None]
        / scale[None, :, None, None]
        / scale[None, None, :, None]
        / scale[None, None, None, :]
    )
    cases.append(
        dict(
            name=name,
            atoms=atoms,
            spherical=spherical,
            e_scf=hf.e_tot,
            n_mo=hf.mo_coeff.shape[1],
            n_aux=hf.with_df._cderi.shape[0],
            orbital_energies=hf.mo_energy.tolist(),
            density=(
                (dm * scale[:, None] * scale[None, :])[np.ix_(order, order)]
            ).tolist(),
            fock=normalized(f),
            j=normalized(j),
            k=normalized(k),
            maximum_fitting_error=float(np.max(np.abs(error))),
        )
    )
    print(name, hf.e_tot, hf.mo_coeff.shape, flush=True)
Path(__file__).resolve().parents[1].joinpath("data/df_hf_reference.json").write_text(
    json.dumps(
        dict(
            source=f"PySCF {pyscf.__version__} RHF density_fit(auxbasis), Cartesian normalized AO representation; explicit spherical/rank-truncated eigenspaces",
            cases=cases,
        ),
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
