# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Public PySCF complex-HF oracle with explicit common-gauge magnetic integrals."""

import json
from pathlib import Path

import numpy as np
from gen_ghf import rust_ao_permutation
from gen_mr_molecular import models as mr_models
from pyscf import __version__ as pyscf_version
from pyscf import gto, lib, scf
from scipy.linalg import block_diag, eigh, expm
from scipy.optimize import least_squares

from embedded_basis import embedded_basis

lib.num_threads(4)


def cases():
    h2 = [
        dict(symbol="H", position=[0.2, -0.1, -0.7]),
        dict(symbol="H", position=[0.2, -0.1, 0.7]),
    ]
    water = [
        dict(symbol="O", position=[0.1, -0.2, 0.3]),
        dict(symbol="H", position=[1.53, 0.68, 0.3]),
        dict(symbol="H", position=[-1.33, 0.68, 0.3]),
    ]
    return [
        dict(
            name="H2 zero-field cRHF",
            atoms=h2,
            basis_constant="STO_3G_GBS",
            reference="rhf",
            charge=0,
            spin=0,
            field=[0.0] * 3,
        ),
        dict(
            name="H2 tilted-field cRHF",
            atoms=h2,
            basis_constant="STO_3G_GBS",
            reference="rhf",
            charge=0,
            spin=0,
            field=[0.1, -0.04, 0.08],
        ),
        dict(
            name="water cRHF STO-3G",
            atoms=water,
            basis_constant="STO_3G_GBS",
            reference="rhf",
            charge=0,
            spin=0,
            field=[0.0, 0.0, 0.08],
        ),
        dict(
            name="water cRHF 6-31G",
            atoms=water,
            basis_constant="BASIS_6_31G_GBS",
            reference="rhf",
            charge=0,
            spin=0,
            field=[0.07, -0.04, 0.09],
        ),
        dict(
            name="LiH+ cUHF",
            atoms=[
                dict(symbol="Li", position=[0.2, 0.3, -0.2]),
                dict(symbol="H", position=[0.4, -0.1, 2.7]),
            ],
            basis_constant="STO_3G_GBS",
            reference="uhf",
            charge=1,
            spin=1,
            field=[0.03, 0.06, -0.08],
        ),
        dict(
            mr_models()[2],
            name="HeH+ cRHF nonlocal ECP",
            basis_constant=None,
            reference="rhf",
            spin=0,
            field=[0.08, -0.03, 0.09],
        ),
        dict(
            name="H cGHF transverse spin Zeeman",
            atoms=[dict(symbol="H", position=[0.1, -0.2, 0.3])],
            basis_constant="BASIS_6_31G_GBS",
            reference="ghf",
            charge=0,
            spin=1,
            field=[0.04, -0.07, 0.08],
            spin_zeeman=True,
        ),
        dict(
            name="H3 frustrated cGHF with orbital field",
            atoms=[
                dict(symbol="H", position=r)
                for r in (
                    [0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [1.5, 2.598076211353316, 0.0],
                )
            ],
            basis_constant="STO_3G_GBS",
            reference="ghf",
            charge=0,
            spin=1,
            field=[0.01, 0.03, 0.04],
            spin_zeeman=False,
            seed="triangle",
        ),
        dict(
            name="water cRHF fixed common gauge origin",
            atoms=water,
            basis_constant="STO_3G_GBS",
            reference="rhf",
            charge=0,
            spin=0,
            field=[0.04, 0.02, 0.07],
            origin=[0.2, 0.7, -0.4],
        ),
    ]


def problem(model, coordinates=None):
    atoms = [(a["symbol"], a["position"]) for a in model["atoms"]]
    if coordinates is not None:
        atoms = [(a[0], r) for a, r in zip(atoms, coordinates)]
    if model["basis_constant"]:
        basis = embedded_basis(model["basis_constant"], sorted({a[0] for a in atoms}))
    else:
        basis = model["basis"]
    mol = gto.M(
        atom=atoms,
        basis=basis,
        ecp=model.get("ecp") or {},
        charge=model["charge"],
        spin=model["spin"],
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    origin = np.array(model.get("origin", np.mean(mol.atom_coords(), axis=0)))
    field = np.array(model["field"])
    with mol.with_common_orig(origin):
        # PySCF i*r cross p is a real antisymmetric integral family.
        orbital = -0.5j * np.einsum(
            "x,xij->ij", field, mol.intor("int1e_cg_irxp", comp=3)
        )
        moments = mol.intor("int1e_rr", comp=9).reshape(3, 3, mol.nao, mol.nao)
        diamagnetic = np.einsum(
            "xy,xyij->ij",
            (np.dot(field, field) * np.eye(3) - np.outer(field, field)) / 8,
            moments,
        )
    h = scf.RHF(mol).get_hcore() + diamagnetic + orbital
    h2 = block_diag(h, h)
    if model.get("spin_zeeman", False):
        b = 0.5 * np.array(
            [
                [field[2], field[0] - 1j * field[1]],
                [field[0] + 1j * field[1], -field[2]],
            ]
        )
        h2 += np.kron(b, mol.intor("int1e_ovlp"))
    return mol, h, h2, diamagnetic, orbital


def triangle_seed(mol):
    values, u = eigh(mol.intor("int1e_ovlp"))
    x = (u / np.sqrt(values)) @ u.T
    c = np.zeros((2 * mol.nao, mol.nelectron), dtype=complex)
    for k in range(3):
        angle = 2 * np.pi * k / 3
        c[: mol.nao, k] = x[:, k] * np.cos(angle / 2)
        c[mol.nao :, k] = x[:, k] * np.exp(0.43j) * np.sin(angle / 2)
    return c @ c.conj().T


def energy(model, coordinates=None):
    mol, h, h2, _, _ = problem(model, coordinates)
    kind = model["reference"]
    mf = {"rhf": scf.RHF, "uhf": scf.UHF, "ghf": scf.GHF}[kind](mol)
    core = h2 if kind == "ghf" else h
    mf.get_hcore = lambda *args: core
    mf.max_cycle = 500
    mf.conv_tol, mf.conv_tol_grad = 1e-14, 1e-12
    mf.chkfile = None
    overlap = mf.get_ovlp()
    _, c = eigh(core, overlap)
    if kind == "rhf":
        initial = 2 * c[:, : mol.nelectron // 2] @ c[:, : mol.nelectron // 2].conj().T
    elif kind == "uhf":
        initial = np.array([c[:, :ne] @ c[:, :ne].conj().T for ne in mol.nelec])
    else:
        initial = c[:, : mol.nelectron] @ c[:, : mol.nelectron].conj().T
    if model.get("seed") == "triangle":
        initial = triangle_seed(mol)
    mf.kernel(dm0=initial)
    residual = np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ))
    if kind == "ghf" and residual > 1e-11:
        # Independently certify stationarity near global spin-rotation zero
        # modes rather than accepting an energy-only stopping condition.
        anchor = mf.mo_coeff.copy()
        ne = mol.nelectron
        pairs = [(a, i) for a in range(ne, len(anchor)) for i in range(ne)]

        def trial(parameters):
            rotation = np.zeros_like(anchor, dtype=complex)
            for k, (a, i) in enumerate(pairs):
                rotation[a, i] = parameters[k] + 1j * parameters[len(pairs) + k]
                rotation[i, a] = -rotation[a, i].conjugate()
            return anchor @ expm(rotation)

        def stationarity(parameters):
            c = trial(parameters)
            p = c[:, :ne] @ c[:, :ne].conj().T
            f = mf.get_fock(dm=p, diis=None)
            values = (c[:, ne:].conj().T @ f @ c[:, :ne]).ravel()
            return np.concatenate((values.real, values.imag))

        solution = least_squares(
            stationarity,
            np.zeros(2 * len(pairs)),
            jac="3-point",
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
            max_nfev=1000,
        )
        assert np.linalg.norm(stationarity(solution.x)) < 1e-11
        mf.mo_coeff = trial(solution.x)
        mf.e_tot = mf.energy_tot(dm=mf.make_rdm1())
        residual = np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ))
        mf.converged = residual < 1e-11
    assert mf.converged and residual < 1e-9, (model["name"], residual)
    return mf


def main():
    records = []
    for model in cases():
        mol, _, h2, dia, orbital = problem(model)
        mf = energy(model)
        dm = mf.make_rdm1()
        if model["reference"] == "rhf":
            p = block_diag(dm / 2, dm / 2)
            occupied = mf.mo_coeff[:, mf.mo_occ > 0]
            occupied = block_diag(occupied, occupied)
        elif model["reference"] == "uhf":
            p = block_diag(*dm)
            occupied = block_diag(
                *[mf.mo_coeff[s][:, mf.mo_occ[s] > 0] for s in range(2)]
            )
        else:
            p = dm
            occupied = mf.mo_coeff[:, mf.mo_occ > 0]
        check = scf.GHF(mol)
        check.get_hcore = lambda *args, core=h2: core
        fock = check.get_fock(dm=p, diis=None)
        assert abs(check.energy_tot(dm=p) - mf.e_tot) < 1e-11
        s = mol.intor("int1e_ovlp")
        values, u = eigh(s)
        x = (u / np.sqrt(values)) @ u.T
        n = mol.nao
        a = np.fromfunction(
            lambda i, j: np.sin((i + 1) * (j + 2)) + 1j * np.cos((i + 2) * (j + 1)),
            (2 * n, 2 * n),
        )
        c = block_diag(x, x) @ expm(0.13 * (a - a.conj().T))
        arbitrary = c[:, : mol.nelectron] @ c[:, : mol.nelectron].conj().T
        arbitrary_fock = check.get_fock(dm=arbitrary, diis=None)
        order = (
            rust_ao_permutation(mol, model["basis_constant"])
            if model["basis_constant"]
            else list(range(n))
        )
        spin_order = order + [n + i for i in order]
        record = {k: v for k, v in model.items() if k != "basis"}
        record.update(e_total=float(mf.e_tot), pyscf_ao_indices_in_rust_order=order)
        record["s_squared"] = float(scf.ghf.spin_square(occupied, s)[0])
        if model.get("seed") == "triangle":
            seed = triangle_seed(mol)[np.ix_(spin_order, spin_order)]
            record["initial_real"], record["initial_imag"] = (
                seed.real.tolist(),
                seed.imag.tolist(),
            )
        for name, matrix in (
            ("h", h2),
            ("density", p),
            ("fock", fock),
            ("arbitrary_density", arbitrary),
            ("arbitrary_fock", arbitrary_fock),
        ):
            matrix = matrix[np.ix_(spin_order, spin_order)]
            record[name + "_real"], record[name + "_imag"] = (
                matrix.real.tolist(),
                matrix.imag.tolist(),
            )
        record["diamagnetic"] = dia[np.ix_(order, order)].tolist()
        record["orbital_imag"] = orbital.imag[np.ix_(order, order)].tolist()
        coordinates = np.array([a["position"] for a in model["atoms"]])
        gradient = []
        for coordinate in range(coordinates.size):
            energies = []
            for offset in (-2, -1, 1, 2):
                point = coordinates.copy()
                point.ravel()[coordinate] += offset * 1e-3
                energies.append(float(energy(model, point).e_tot))
            gradient.append(
                (energies[0] - 8 * energies[1] + 8 * energies[2] - energies[3]) / 0.012
            )
        record["gradient"] = gradient
        records.append(record)
        print(model["name"], mf.e_tot, "gradient", len(gradient), flush=True)
    path = Path(__file__).resolve().parents[1] / "data/complex_hf_reference.json"
    path.write_text(
        json.dumps(
            dict(
                source="Public PySCF complex RHF/UHF/GHF; common-gauge orbital/diamagnetic integrals; all nuclear components from independent five-point energies",
                pyscf_version=pyscf_version,
                cases=records,
            ),
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
