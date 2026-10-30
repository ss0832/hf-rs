# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""PySCF CASCI references and independent full-Fock MR driven energies."""

import json
import math
from pathlib import Path

import numpy as np
from gen_gno import GeneralizedOracle
from gen_mr_driven import driven
from pyscf import ao2mo, fci, gto, lib, mcscf, scf
from scipy.linalg import expm
from scipy.optimize import root

from embedded_basis import embedded_basis

lib.num_threads(4)


def dense_ci(h1, h2, norb, nelec, ci0=None, ecore=0.0, **kwargs):
    # The oracle has two active orbitals. Solve its complete CI matrix at
    # every CASSCF microiteration rather than accepting energy-only CI accuracy.
    na, nb = nelec
    shape = (math.comb(norb, na), math.comb(norb, nb))
    addresses, hamiltonian = fci.direct_spin1.pspace(
        h1, h2, norb, nelec, np=math.prod(shape)
    )
    energies, vectors = np.linalg.eigh(hamiltonian)
    vector = np.zeros(math.prod(shape))
    vector[addresses] = vectors[:, 0]
    return float(ecore + energies[0]), vector.reshape(shape)


def exact_cas(mc, coefficients):
    mc.kernel(coefficients)
    h1, core = mc.get_h1eff(coefficients)
    h2 = mc.get_h2eff(coefficients)
    addresses, hamiltonian = fci.direct_spin1.pspace(
        h1, h2, mc.ncas, mc.nelecas, np=mc.ci.size
    )
    energies, vectors = np.linalg.eigh(hamiltonian)
    vector = np.zeros(mc.ci.size)
    vector[addresses] = vectors[:, 0]
    mc.ci = vector.reshape(mc.ci.shape)
    mc.e_tot = float(core + energies[0])
    assert (
        np.linalg.norm(hamiltonian @ vectors[:, 0] - energies[0] * vectors[:, 0])
        < 1e-13
    )
    assert abs(fci.spin_op.spin_square(mc.ci, mc.ncas, mc.nelecas)[0]) < 1e-12


def molecule(model, coordinates=None, return_problem=False, reference_only=False):
    atoms = [(a["symbol"], a["position"]) for a in model["atoms"]]
    if coordinates is not None:
        atoms = [(a[0], r) for a, r in zip(atoms, coordinates, strict=True)]
    mol = gto.M(
        atom=atoms,
        basis=model["basis"],
        ecp=model.get("ecp") or {},
        charge=model["charge"],
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-12, max_cycle=300)
    assert mf.converged
    c = mf.mo_coeff[:, : model.get("n_mo", mf.mo_coeff.shape[1])]
    initial = c.copy()
    if model.get("optimize", False):
        optimizer = mcscf.CASSCF(mf, model["n_active"], model["active_electrons"])
        optimizer.conv_tol = 1e-13
        optimizer.conv_tol_grad = 1e-10
        optimizer.max_cycle_macro = 150
        optimizer.fcisolver.conv_tol = 1e-15
        optimizer.fcisolver.kernel = dense_ci
        optimizer.fcisolver.approx_kernel = dense_ci
        optimizer.kernel(c)
        # Refine the orbital stationarity independently of energy-based macro
        # stopping. At this scale orbital steps can change energy below 1 ulp.
        anchor = optimizer.mo_coeff.copy()
        classes = [0] * optimizer.ncore + [1] * optimizer.ncas
        classes += [2] * (anchor.shape[1] - len(classes))
        pairs = [
            (p, q)
            for p in range(len(classes))
            for q in range(p)
            if classes[p] != classes[q]
        ]
        ci_oracle = mcscf.CASCI(mf, model["n_active"], model["active_electrons"])
        ci_oracle.canonicalization = False

        def orbital_gradient(parameters):
            rotation = np.zeros((len(classes),) * 2)
            for value, (p, q) in zip(parameters, pairs, strict=True):
                rotation[p, q], rotation[q, p] = value, -value
            trial = anchor @ expm(rotation)
            exact_cas(ci_oracle, trial)
            optimizer.mo_coeff, optimizer.ci = trial, ci_oracle.ci
            return optimizer.get_grad(trial)

        solution = root(orbital_gradient, np.zeros(len(pairs)), tol=1e-10)
        assert np.max(abs(orbital_gradient(solution.x))) < 1e-11
        c = optimizer.mo_coeff
    mc = mcscf.CASCI(mf, model["n_active"], model["active_electrons"])
    mc.canonicalization = False
    mc.fcisolver.conv_tol = 1e-14
    exact_cas(mc, c)
    nc = mc.ncore
    na = mc.ncas
    f = c.T @ (mf.get_hcore() + mf.get_veff(dm=mc.make_rdm1())) @ c
    u = np.eye(c.shape[1])
    for a, b in ((0, nc), (nc, nc + na), (nc + na, c.shape[1])):
        if a < b:
            _, u[a:b, a:b] = np.linalg.eigh(f[a:b, a:b])
    c = c @ u
    exact_cas(mc, c)
    if reference_only:
        return mf, mc, c
    active = list(range(2 * nc, 2 * (nc + na)))
    core = list(range(2 * nc))
    members = [
        dict(
            na=mc.nelecas[0],
            nb=mc.nelecas[1],
            coefficients=mc.ci.ravel().tolist(),
            weight=1.0,
        )
    ]
    oracle = GeneralizedOracle(c.shape[1] * 2, members, active, core)
    n = oracle.n
    hh = c.T @ mf.get_hcore() @ c
    eri = ao2mo.kernel(mol, c, compact=False).reshape((c.shape[1],) * 4)
    one = np.zeros((n, n))
    g = np.zeros((len(oracle.groups[2]),) * 2)
    for p in range(n):
        for q in range(n):
            if p % 2 == q % 2:
                one[p, q] = hh[p // 2, q // 2]
    for i, (p, q) in enumerate(oracle.groups[2]):
        for j, (r, s) in enumerate(oracle.groups[2]):
            if p % 2 == r % 2 and q % 2 == s % 2:
                g[i, j] += eri[p // 2, r // 2, q // 2, s // 2]
            if p % 2 == s % 2 and q % 2 == r % 2:
                g[i, j] -= eri[p // 2, s // 2, q // 2, r // 2]
    physical = oracle.base.matrix(
        np.concatenate(([mol.energy_nuc()], one.ravel(), g.ravel()))
    )
    h, _ = oracle.project(physical)
    assert abs(h[0] - mc.e_tot) < 1e-10
    if return_problem:
        return (
            dict(
                e_scf=float(mf.e_tot),
                e_cas=float(mc.e_tot),
                n_core=nc,
                n_active=na,
                n_alpha=mc.nelecas[0],
                n_beta=mc.nelecas[1],
                mo_coeff=c.tolist(),
                coefficients=mc.ci.ravel().tolist(),
            ),
            oracle,
            h,
        )
    result = driven(oracle, h, active, core, 0.5)
    return dict(
        e_scf=float(mf.e_tot),
        e_cas=float(mc.e_tot),
        initial_mo_coeff=initial.tolist(),
        semicanonical_mo_coeff=c.tolist(),
        **result,
    )


def models():
    ecp = {
        "He": [
            0,
            [
                [-1, [[], [], [[0.7, 0.17]]]],
                [0, [[], [], [[0.53, 0.11]]]],
                [1, [[], [], [[0.61, -0.07]]]],
            ],
        ]
    }
    models = [
        dict(
            name="H3+/STO-3G CAS(2,2)",
            charge=1,
            n_active=2,
            active_electrons=2,
            atoms=[
                dict(symbol="H", position=x)
                for x in [[0.1, -0.2, 0.3], [0.4, 0.2, 1.7], [1.4, -0.3, -0.4]]
            ],
            basis=embedded_basis("STO_3G_GBS", ["H"]),
            gaussian94=None,
        ),
        dict(
            name="LiH/rectangular-3MO CAS(2,2)",
            charge=0,
            n_active=2,
            active_electrons=2,
            n_mo=3,
            atoms=[
                dict(symbol="Li", position=[0.1, -0.2, 0.3]),
                dict(symbol="H", position=[0.4, 0.1, 3.2]),
            ],
            basis=embedded_basis("STO_3G_GBS", ["Li", "H"]),
            gaussian94=None,
        ),
        dict(
            name="HeH+/semilocal-ECP CAS(2,2)",
            charge=1,
            n_active=2,
            active_electrons=2,
            ecp=ecp,
            atoms=[
                dict(symbol="He", position=[0.13, -0.21, 0.31]),
                dict(symbol="H", position=[-0.18, 0.17, 1.81]),
            ],
            basis={"He": [[0, [1.4, 1.0]], [0, [0.45, 1.0]]], "H": [[0, [0.9, 1.0]]]},
            gaussian94="He 0\nS 1 1.0\n1.4 1.0\nS 1 1.0\n0.45 1.0\n****\nH 0\nS 1 1.0\n0.9 1.0\n****\n",
        ),
    ]
    models.append(dict(models[0], name="H3+/STO-3G CASSCF(2,2)", optimize=True))
    return models


def main():
    root_path = Path(__file__).resolve().parents[1]
    records = []
    for model in models():
        result = molecule(model)
        record = {key: value for key, value in model.items() if key != "basis"}
        record.update(result)
        records.append(record)
        print(
            model["name"],
            result["e_cas"],
            result["pt2"],
            result["pt3"],
            result["transformed"]["zero"],
            flush=True,
        )
    (root_path / "data/mr_molecular_reference.json").write_text(
        json.dumps(dict(source=__doc__, cases=records), indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
