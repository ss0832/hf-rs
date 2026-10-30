# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent PySCF CABS metric, full Fock and singles reference, including ECP."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, lib, scf

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)


def native_cartesian_order(molecule):
    order = []
    offset = 0
    for shell in range(molecule.nbas):
        angular = molecule.bas_angular(shell)
        size = (angular + 1) * (angular + 2) // 2
        components = [0, 3, 5, 1, 2, 4] if angular == 2 else range(size)
        for _ in range(molecule.bas_nctr(shell)):
            order.extend(offset + component for component in components)
            offset += size
    return order


cases = []
for name, symbols, positions, ecp in [
    ("h2", ["H", "H"], [[-0.1, 0.2, -0.7], [0.3, -0.25, 0.8]], {}),
    ("he2", ["He", "He"], [[-0.1, 0.2, -0.7], [0.3, -0.25, 2.8]], {}),
    (
        "he_local_ecp",
        ["He"],
        [[0.13, -0.21, 0.31]],
        {"He": [0, [[-1, [[], [], [[0.7, 0.17]]]]]]},
    ),
    ("heh_spherical", ["He", "H"], [[-0.1, 0.2, 0.3], [0.4, -0.3, 1.8]], {}),
    ("he_overlap_rank_drop", ["He"], [[0.13, -0.21, 0.31]], {}),
]:
    obsdefs = embedded_basis("STO_3G_GBS", list(set(symbols)))
    spherical = name == "heh_spherical"
    if spherical:
        obsdefs = {"He": [[0, [1.4, 1.0]], [2, [0.6, 1.0]]], "H": [[0, [1.1, 1.0]]]}
    if name == "he_overlap_rank_drop":
        obsdefs = {"He": [[0, [1.4, 1.0]], [0, [1.4, 1.0]], [0, [0.4, 1.0]]]}
    mol = gto.M(
        atom=list(zip(symbols, positions)),
        basis=obsdefs,
        ecp=ecp,
        unit="Bohr",
        charge=1 if spherical else 0,
        cart=not spherical,
        verbose=0,
    )
    mf = scf.RHF(mol)
    if name == "he_overlap_rank_drop":
        # Fix the null-space coefficient gauge with canonical orthogonalization.
        # Pivoted Cholesky gives the same physical orbitals but different raw-AO
        # coefficients for the deliberately duplicated basis function.
        orthogonalizer = scf.addons.canonical_orth_(mf.get_ovlp(), thr=1e-8)

        def canonical_eigh(fock, overlap, x=None):
            energies, rotation = np.linalg.eigh(
                orthogonalizer.T @ fock @ orthogonalizer
            )
            return energies, orthogonalizer @ rotation

        mf.eig = canonical_eigh
        mf.init_guess = "1e"
    mf.run(conv_tol=1e-14, conv_tol_grad=1e-10)
    assert mf.converged
    cartmol = mol.copy()
    cartmol.cart = True
    physical_obs = (
        mol.cart2sph_coeff(normalized="sp") @ mf.mo_coeff if spherical else mf.mo_coeff
    )
    auxdefs = {
        f"ghost-{symbol}": [[0, [0.41, 1.0]], [1, [0.65, 1.0]]]
        for symbol in set(symbols)
    }
    if spherical:
        for shells in auxdefs.values():
            shells.append([2, [0.52, 1.0]])
    aux = gto.M(
        atom=[
            (f"ghost-{symbol}", position)
            for symbol, position in zip(symbols, positions)
        ],
        basis=auxdefs,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    union = gto.conc_mol(cartmol, aux)
    nao = cartmol.nao
    nmo = mf.mo_coeff.shape[1]
    no = mol.nelectron // 2
    s = union.intor("int1e_ovlp")
    projection = physical_obs @ physical_obs.T @ s[:nao, nao:]
    metric = s[nao:, nao:] - s[nao:, :nao] @ projection
    physical_aux = aux.cart2sph_coeff(normalized="sp") if spherical else np.eye(aux.nao)
    values, u = np.linalg.eigh(physical_aux.T @ metric @ physical_aux)
    caux = physical_aux @ (u[:, values > 1e-8] / np.sqrt(values[values > 1e-8]))
    cobs = -projection @ caux
    cabs = np.vstack((cobs, caux))
    dm = np.zeros_like(s)
    dm[:nao, :nao] = 2 * physical_obs[:, :no] @ physical_obs[:, :no].T
    h = union.intor("int1e_kin") + union.intor("int1e_nuc")
    if ecp:
        h += union.intor("ECPscalar")
    j, k = scf.hf.get_jk(union, dm)
    fock = h + j - 0.5 * k
    occupied = np.zeros((union.nao, no))
    occupied[:nao] = physical_obs[:, :no]
    obsvirtual = np.zeros((union.nao, nmo - no))
    obsvirtual[:nao] = physical_obs[:, no:]
    virtual = np.column_stack((obsvirtual, cabs))
    eps, rotation = np.linalg.eigh(virtual.T @ fock @ virtual)
    virtual = virtual @ rotation
    coupling = occupied.T @ fock @ virtual
    t = -coupling / (eps[None, :] - mf.mo_energy[:no, None])
    energy = float(2 * np.sum(coupling * t))
    assert np.max(abs(cabs.T @ s @ cabs - np.eye(cabs.shape[1]))) < 1e-10
    order = native_cartesian_order(union)
    normalization = np.sqrt(np.diag(s))
    cabs_covectors = ((s @ cabs) / normalization[:, None])[order]
    occupied_covectors = ((s @ occupied) / normalization[:, None])[order]
    virtual_covectors = ((s @ virtual) / normalization[:, None])[order]
    cabs = (normalization[:, None] * cabs)[order]
    singles = (
        normalization[:, None] * (occupied @ t @ virtual.T) * normalization[None, :]
    )[np.ix_(order, order)]
    fock = (fock / normalization[:, None] / normalization[None, :])[
        np.ix_(order, order)
    ]
    exchange = (0.5 * k / normalization[:, None] / normalization[None, :])[
        np.ix_(order, order)
    ]
    cases.append(
        dict(
            name=name,
            symbols=symbols,
            positions=positions,
            n_cart=nao,
            n_mo=nmo,
            e_scf=float(mf.e_tot),
            cabs_rank=cabs.shape[1],
            cabs_projector=(cabs @ cabs.T).tolist(),
            cabs_metric_projector=(cabs_covectors @ cabs_covectors.T).tolist(),
            full_fock=fock.tolist(),
            exchange=exchange.tolist(),
            singles_energy=energy,
            singles_energy_by_frozen_core=[
                float(2 * np.sum(coupling[fc:] * t[fc:])) for fc in range(no + 1)
            ],
            singles_metric_by_frozen_core=[
                (occupied_covectors[:, fc:] @ t[fc:] @ virtual_covectors.T).tolist()
                for fc in range(no + 1)
            ],
            virtual_energies=eps.tolist(),
            singles_ao=singles.tolist(),
            singles_metric=(occupied_covectors @ t @ virtual_covectors.T).tolist(),
        )
    )
    if ecp:
        assert mol._ecpbas.shape[0] > 0
out = dict(
    source=f"PySCF {pyscf.__version__} direct union-AO Coulomb/exchange/ECP Fock, NumPy CABS metric complement and full virtual singles resolvent",
    basis_source_sha256=basis_hash(),
    cases=cases,
)
Path(__file__).resolve().parents[1].joinpath("data/cabs_reference.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8"
)
print([(c["name"], c["cabs_rank"], c["singles_energy"]) for c in cases])
