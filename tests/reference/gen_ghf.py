# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""PySCF real GHF references, spin-coherence contractions and all-coordinate energies."""

import json
import re
from pathlib import Path

import numpy as np
from gen_mr_molecular import models as mr_models
from pyscf import ao2mo, gto, lib, scf
from scipy.linalg import block_diag, eigh, expm
from scipy.optimize import least_squares

from embedded_basis import SOURCE, embedded_basis

lib.num_threads(4)


def invsqrt(s):
    values, vectors = eigh(s)
    return (vectors / np.sqrt(values)) @ vectors.T


def make_mol(model, coordinates=None):
    atoms = [(a["symbol"], a["position"]) for a in model["atoms"]]
    if coordinates is not None:
        atoms = [(a[0], r) for a, r in zip(atoms, coordinates)]
    return gto.M(
        atom=atoms,
        basis=model["basis"],
        ecp=model.get("ecp") or {},
        charge=model["charge"],
        spin=model["spin"],
        unit="Bohr",
        cart=True,
        verbose=0,
    )


def energy(model, initial, coordinates=None):
    mol = make_mol(model, coordinates)
    s = mol.intor("int1e_ovlp")
    s2 = block_diag(s, s)
    c = initial @ invsqrt(initial.T @ s2 @ initial)
    mf = scf.GHF(mol)
    mf.max_cycle = 500
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-11
    mf.kernel(dm0=c @ c.T)
    if not mf.converged and np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)) > 1e-11:
        mf = mf.newton()
        mf.max_cycle = 150
        mf.kernel()
    if np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)) > 1e-11:
        # Remove energy-only stopping ambiguity around spin-rotation zero
        # modes by independently solving the occupied/virtual stationarity.
        anchor = mf.mo_coeff.real.copy()
        ne = mol.nelectron
        pairs = [(a, i) for a in range(ne, 2 * mol.nao_nr()) for i in range(ne)]

        def trial(parameters):
            rotation = np.zeros((len(anchor), len(anchor)))
            for value, (a, i) in zip(parameters, pairs):
                rotation[a, i], rotation[i, a] = value, -value
            return anchor @ expm(rotation)

        def residual(parameters):
            orbitals = trial(parameters)
            density = orbitals[:, :ne] @ orbitals[:, :ne].T
            fock = mf.get_fock(dm=density, diis=None)
            return (orbitals[:, ne:].T @ fock @ orbitals[:, :ne]).ravel()

        solution = least_squares(
            residual,
            np.zeros(len(pairs)),
            jac="3-point",
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
            max_nfev=1000,
        )
        assert np.linalg.norm(residual(solution.x)) < 1e-11
        mf.mo_coeff = trial(solution.x)
        density = mf.make_rdm1()
        mf.e_tot = mf.energy_tot(dm=density)
    assert np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)) < 1e-10
    assert np.max(abs(mf.mo_coeff.imag)) < 1e-10
    return mf


def model_cases():
    basis = embedded_basis("STO_3G_GBS", ["H", "Li"])
    triangle = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, 1.5 * np.sqrt(3), 0.0]]
    )
    generator = np.array([[0.0, -0.31, 0.21], [0.31, 0.0, -0.27], [-0.21, 0.27, 0.0]])
    triangle = triangle @ expm(generator).T + [0.13, -0.21, 0.31]
    ecp = mr_models()[2]
    return [
        dict(
            name="H2+/STO-3G doublet",
            atoms=[
                dict(symbol="H", position=r)
                for r in [[0.1, -0.2, 0.3], [0.4, 0.1, 2.7]]
            ],
            basis=basis,
            charge=1,
            spin=1,
            seed="core",
        ),
        dict(
            name="LiH/STO-3G closed shell",
            atoms=[
                dict(symbol="Li", position=[0.1, -0.2, 0.3]),
                dict(symbol="H", position=[0.4, 0.1, 3.2]),
            ],
            basis=basis,
            charge=0,
            spin=0,
            seed="core",
        ),
        dict(
            name="H3/STO-3G frustrated real GHF",
            atoms=[dict(symbol="H", position=r.tolist()) for r in triangle],
            basis=basis,
            charge=0,
            spin=1,
            seed="triangle",
        ),
        dict(ecp, name="HeH+/nonlocal-ECP closed shell", spin=0, seed="core"),
    ]


def rust_ao_permutation(mol, constant):
    """Match G94 shell order; PySCF groups shells by angular momentum."""
    text = SOURCE.read_text(encoding="utf-8")
    gbs = re.search(rf'pub const {constant}: &str = r#"(.*?)"#;', text, re.S).group(1)
    locations = mol.ao_loc_nr()
    order = []
    for atom in range(mol.natm):
        symbol = mol.atom_symbol(atom)
        segment = re.search(rf"(?m)^{symbol}\s+0\s*\n(.*?)\*\*\*\*", gbs, re.S).group(1)
        lines = [line.strip() for line in segment.splitlines() if line.strip()]
        angular = []
        first = 0
        while first < len(lines):
            fields = lines[first].split()
            angular.extend({"S": [0], "P": [1], "SP": [0, 1]}[fields[0].upper()])
            first += int(fields[1]) + 1
        shells = {momentum: [] for momentum in set(angular)}
        for shell in range(mol.nbas):
            if mol.bas_atom(shell) == atom:
                assert mol.bas_nctr(shell) == 1
                shells[mol.bas_angular(shell)].append(
                    list(range(locations[shell], locations[shell + 1]))
                )
        for momentum in angular:
            order.extend(shells[momentum].pop(0))
    assert sorted(order) == list(range(mol.nao_nr()))
    return order


def ethylene_reference():
    atoms = [
        ("C", [0.0, 0.0, 1.266]),
        ("C", [0.0, 0.0, -1.266]),
        ("H", [0.0, 1.744, 2.336]),
        ("H", [0.0, -1.744, 2.336]),
        ("H", [1.744, 0.0, -2.336]),
        ("H", [-1.744, 0.0, -2.336]),
    ]
    model = dict(
        name="Twisted ethylene/6-31G spin instability",
        atoms=[dict(symbol=a, position=r) for a, r in atoms],
        basis=embedded_basis("BASIS_6_31G_GBS", ["C", "H"]),
        basis_name="6-31g",
        charge=0,
        spin=0,
    )
    mol = make_mol(model)
    rhf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-11)
    c = rhf.mo_coeff
    n, occ = len(c), mol.nelectron // 2
    g = ao2mo.kernel(mol, c, compact=False).reshape((n,) * 4)
    pairs = [(a, i) for a in range(occ, n) for i in range(occ)]
    curvature = np.array(
        [
            [
                (rhf.mo_energy[a] - rhf.mo_energy[i] if a == b and i == j else 0.0)
                - g[a, b, i, j]
                - g[a, j, i, b]
                for b, j in pairs
            ]
            for a, i in pairs
        ]
    )
    values, vectors = eigh(curvature)
    assert values[0] < 0.0
    # The target is the broken-symmetry Ms=0 stationary branch. Its global
    # spin rotation is also a GHF stationary solution of the scalar Hamiltonian.
    frontier = np.zeros((n, n))
    frontier[occ, occ - 1], frontier[occ - 1, occ] = np.pi / 4, -np.pi / 4
    ca, cb = c @ expm(frontier), c @ expm(-frontier)
    initial = np.zeros((2 * n, 2 * occ))
    initial[:n, :occ], initial[n:, occ:] = ca[:, :occ], cb[:, :occ]

    def branch(initial_coeff, coordinates=None):
        point = make_mol(model, coordinates)
        overlap = point.intor("int1e_ovlp")
        ca = initial_coeff[:n, :occ]
        cb = initial_coeff[n:, occ:]
        ca = ca @ invsqrt(ca.T @ overlap @ ca)
        cb = cb @ invsqrt(cb.T @ overlap @ cb)
        reference = scf.UHF(point)
        reference.max_cycle = 500
        reference.conv_tol, reference.conv_tol_grad = 1e-13, 3e-11
        reference.kernel(dm0=(ca @ ca.T, cb @ cb.T))
        if (
            np.linalg.norm(reference.get_grad(reference.mo_coeff, reference.mo_occ))
            > 1e-10
        ):
            reference = reference.newton().run()
        assert (
            np.linalg.norm(reference.get_grad(reference.mo_coeff, reference.mo_occ))
            < 1e-10
        )
        return reference

    mf = branch(initial)
    assert mf.e_tot < rhf.e_tot - 1e-3
    transported = np.zeros_like(initial)
    transported[:n, :occ] = mf.mo_coeff[0][:, :occ]
    transported[n:, occ:] = mf.mo_coeff[1][:, :occ]
    coordinates = np.array([r for _, r in atoms])
    gradient = []
    for k in range(coordinates.size):
        vals = []
        for offset in [-2.0, -1.0, 1.0, 2.0]:
            point = coordinates.copy()
            point.ravel()[k] += offset * 1e-3
            vals.append(float(branch(transported, point).e_tot))
        gradient.append((vals[0] - 8 * vals[1] + 8 * vals[2] - vals[3]) / 0.012)
        print("ethylene reference component", k + 1, "of", coordinates.size, flush=True)
    density = block_diag(*mf.make_rdm1())
    fock = scf.GHF(mol).get_fock(dm=density, diis=None)
    ao_order = rust_ao_permutation(mol, "BASIS_6_31G_GBS")
    spin_order = ao_order + [n + p for p in ao_order]
    density = density[np.ix_(spin_order, spin_order)]
    fock = fock[np.ix_(spin_order, spin_order)]
    record = {k: v for k, v in model.items() if k != "basis"}
    record.update(
        e_total=float(mf.e_tot),
        e_rhf=float(rhf.e_tot),
        spin_curvature=float(values[0]),
        s_squared=float(mf.spin_square()[0]),
        reference_kind="PySCF UHF Ms=0 branch; scalar GHF energy and spin-rotation equivalence",
        pyscf_ao_indices_in_rust_order=ao_order,
        density=density.tolist(),
        fock=fock.tolist(),
        gradient=gradient,
    )
    print(model["name"], mf.e_tot, mf.spin_square()[0], flush=True)
    return record


def main():
    records = []
    for model in model_cases():
        mol = make_mol(model)
        n = mol.nao_nr()
        s = mol.intor("int1e_ovlp")
        h = scf.RHF(mol).get_hcore()
        x = invsqrt(s)
        if model["seed"] == "triangle":
            c0 = np.zeros((2 * n, mol.nelectron))
            for k in range(3):
                angle = 2 * np.pi * k / 3
                c0[:n, k] = x[:, k] * np.cos(angle / 2)
                c0[n:, k] = x[:, k] * np.sin(angle / 2)
        else:
            _, u = eigh(x.T @ h @ x)
            spatial = x @ u
            c0 = np.zeros((2 * n, mol.nelectron))
            for k in range(mol.nelectron):
                c0[(k % 2) * n : (k % 2 + 1) * n, k] = spatial[:, k // 2]
        mf = energy(model, c0)
        density = mf.make_rdm1().real
        fock = mf.get_fock(dm=density, diis=None)
        occupied = mf.mo_coeff[:, mf.mo_occ > 0].real
        spin_square = scf.ghf.spin_square(occupied, s)[0]
        sz = 0.5 * np.trace((density[:n, :n] - density[n:, n:]) @ s)
        sx = 0.5 * np.trace((density[:n, n:] + density[n:, :n]) @ s)
        random_rotation = np.fromfunction(
            lambda i, j: np.sin((i + 1) * (j + 2)), (2 * n, 2 * n)
        )
        q = expm(0.19 * (random_rotation - random_rotation.T))
        orbitals = block_diag(x, x) @ q
        arbitrary = orbitals[:, : mol.nelectron] @ orbitals[:, : mol.nelectron].T
        random_fock = mf.get_fock(dm=arbitrary, diis=None)
        random_ss = scf.ghf.spin_square(orbitals[:, : mol.nelectron], s)[0]
        coordinates = np.array([a["position"] for a in model["atoms"]])
        step = 1e-3
        gradient = []
        for k in range(coordinates.size):
            values = []
            for offset in [-2.0, -1.0, 1.0, 2.0]:
                point = coordinates.copy()
                point.ravel()[k] += offset * step
                values.append(float(energy(model, c0, point).e_tot))
            gradient.append(
                (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)
            )
        record = {k: v for k, v in model.items() if k != "basis"}
        record.update(
            e_total=float(mf.e_tot),
            initial_coeff=c0.tolist(),
            density=density.tolist(),
            fock=fock.tolist(),
            s_squared=float(spin_square),
            s_x=float(sx),
            s_z=float(sz),
            gradient=gradient,
            random_density=arbitrary.tolist(),
            random_fock=random_fock.tolist(),
            random_s_squared=float(random_ss),
        )
        records.append(record)
        print(model["name"], mf.e_tot, spin_square, flush=True)
    path = Path(__file__).resolve().parents[1] / "data/ghf_reference.json"
    path.write_text(
        json.dumps(
            dict(
                source=__doc__,
                step=1e-3,
                cases=records,
                stress_cases=[ethylene_reference()],
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
