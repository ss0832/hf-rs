# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Public PySCF integrals, independent fitted Fock and CABS singles response.

The reference contracts the original-auxiliary metric inverse with all ordered
AO pairs. Native code instead stores only occupied-leg modes and streams J.
No external implementation sources or private execution routines are used.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import df, gto, lib, scf
from pyscf.gto.basis import parse_gaussian

lib.num_threads(4)
root = Path(__file__).resolve().parents[2]
positions = [[-0.1, 0.2, -1.3], [0.3, -0.2, 1.4]]
obs_g94 = "He 0\nS 1 1.0\n1.4 1.0\nS 1 1.0\n0.45 1.0\n****\n"
aux_g94 = (
    "He 0\nS 1 1.0\n0.2 1.0\nS 1 1.0\n0.9 1.0\nS 1 1.0\n3.2 1.0\n"
    "P 1 1.0\n0.8 1.0\nD 1 1.0\n0.52 1.0\n****\n"
)
fit_g94 = (
    "He 0\nS 1 1.0\n0.18 1.0\nS 1 1.0\n0.65 1.0\nS 1 1.0\n2.0 1.0\n"
    "S 1 1.0\n5.6 1.0\nP 1 1.0\n0.25 1.0\nP 1 1.0\n1.1 1.0\n"
    "D 1 1.0\n0.7 1.0\n****\n"
)


def basis(text):
    return parse_gaussian.parse(text, optimize=False)


def native_order(mol):
    order = []
    offset = 0
    for shell in range(mol.nbas):
        angular = mol.bas_angular(shell)
        size = (angular + 1) * (angular + 2) // 2
        for _ in range(mol.bas_nctr(shell)):
            order.extend(
                offset + i
                for i in ([0, 3, 5, 1, 2, 4] if angular == 2 else range(size))
            )
            offset += size
    return order


cases = []
for nonlocal_ecp in [False, True]:
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
        if nonlocal_ecp
        else {}
    )
    mol = gto.M(
        atom=[("He", p) for p in positions],
        basis={"He": basis(obs_g94)},
        ecp=ecp,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-11)
    assert mf.converged
    aux = gto.M(
        atom=[("ghost-He", p) for p in positions],
        basis={"ghost-He": basis(aux_g94)},
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    union = gto.conc_mol(mol, aux)
    order = native_order(union)
    raw_s = union.intor("int1e_ovlp")
    norms = np.sqrt(np.diag(raw_s))[order]
    overlap = raw_s[np.ix_(order, order)] / norms[:, None] / norms[None, :]
    raw_core = union.intor("int1e_kin") + union.intor("int1e_nuc")
    if nonlocal_ecp:
        raw_core += union.intor("ECPscalar")
    core = raw_core[np.ix_(order, order)] / norms[:, None] / norms[None, :]
    nao, n = mol.nao, union.nao
    co = np.zeros((n, nao))
    co[:nao] = norms[:nao, None] * mf.mo_coeff
    no = mol.nelectron // 2
    density = 2 * co[:, :no] @ co[:, :no].T
    projection = co[:nao] @ co[:nao].T @ overlap[:nao, nao:]
    projected_metric = overlap[nao:, nao:] - overlap[nao:, :nao] @ projection
    eig, vectors = np.linalg.eigh(projected_metric)
    complement = vectors[:, eig > 1e-8] / np.sqrt(eig[eig > 1e-8])
    cabs = np.vstack((-projection @ complement, complement))
    virtuals = np.column_stack((co[:, no:], cabs))
    for spherical, duplicated in [(False, False), (True, False), (True, True)]:
        fitting = basis(fit_g94)
        if duplicated:
            fitting += [fitting[0]]
        fit = gto.M(
            atom=[("ghost-He", p) for p in positions],
            basis={"ghost-He": fitting},
            unit="Bohr",
            cart=True,
            verbose=0,
        )
        metric = fit.intor("int2c2e")
        fit_overlap = fit.intor("int1e_ovlp")
        three = df.incore.aux_e2(union, fit, intor="int3c2e", aosym="s1")
        # Public mixed Cartesian/spherical three-center evaluation is not
        # supported; transform its Cartesian result with the public AO map.
        if spherical:
            transform = fit.cart2sph_coeff(normalized="sp")
            metric = transform.T @ metric @ transform
            fit_overlap = transform.T @ fit_overlap @ transform
            three = three @ transform
        fit_norm = np.sqrt(np.diag(fit_overlap))
        metric = metric / fit_norm[:, None] / fit_norm[None, :]
        three = (
            three[np.ix_(order, order, np.arange(len(fit_norm)))]
            / norms[:, None, None]
            / norms[None, :, None]
            / fit_norm[None, None, :]
        )
        eig, vectors = np.linalg.eigh(metric)
        keep = eig > 1e-10
        inverse = (vectors[:, keep] / eig[keep]) @ vectors[:, keep].T
        # Independent original-auxiliary contraction; no native whitened modes.
        fitted_density = inverse @ np.einsum("pqP,pq->P", three, density)
        coulomb = np.einsum("pqP,P->pq", three, fitted_density)
        exchange = np.einsum(
            "prP,PQ,qsQ,rs->pq", three, inverse, three, density, optimize=True
        )
        fock = core + coulomb - 0.5 * exchange
        ev, u = np.linalg.eigh(virtuals.T @ fock @ virtuals)
        va = virtuals @ u
        coupling = co[:, :no].T @ fock @ va
        amplitudes = -coupling / (ev[None, :] - mf.mo_energy[:no, None])
        response = co[:, :no] @ amplitudes @ va.T
        cases.append(
            dict(
                nonlocal_ecp=nonlocal_ecp,
                spherical=spherical,
                duplicated=duplicated,
                rank=int(sum(keep)),
                e_scf=float(mf.e_tot),
                fock=fock.tolist(),
                half_exchange=(0.5 * exchange).tolist(),
                singles_energy=float(2 * np.sum(coupling * amplitudes)),
                virtual_energies=ev.tolist(),
                singles_response=response.tolist(),
                singles_metric_response=(overlap @ response @ overlap).tolist(),
            )
        )
        print("Fitted Fock", nonlocal_ecp, spherical, duplicated, sum(keep), flush=True)

destination = root / "tests/data/f12_fitted_fock_reference.json"
destination.write_text(
    json.dumps(
        dict(
            source=f"Public PySCF {pyscf.__version__} lower-center integrals; NumPy original-auxiliary J/K and full-virtual CABS response",
            positions=positions,
            obs_g94=obs_g94,
            aux_g94=aux_g94,
            fit_g94=fit_g94,
            cases=cases,
        ),
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
