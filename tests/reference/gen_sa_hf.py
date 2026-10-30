# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""State-averaged HF over explicit determinants, minimized independently with SciPy.

Each state is a determinant of common restricted orbitals with occupied alpha/beta lists.
Its exact energy uses PySCF J/K builds, E = sum_s Tr(P_s h) + 1/2 sum_s Tr(P_s (J - K_s)).
The ensemble energy sum_I w_I E_I is minimized over all nonredundant real orbital rotations
by SciPy BFGS from PySCF RHF/ROHF canonical orbitals; the analytic orbital gradient is
checked against central differences first. Indices refer to energy-ordered canonical
orbitals. Contracted Cartesian AO space, embedded basis data. No SA-HF source is read.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
import scipy.linalg
import scipy.optimize
from gen_ghf import rust_ao_permutation
from pyscf import gto, lib, scf

from embedded_basis import embedded_basis

lib.num_threads(4)

MODELS = [
    dict(
        name="B_atom_p_ensemble",
        atoms=[("B", [0.1, -0.2, 0.3])],
        constant="BASIS_6_31G_GBS",
        spin=1,
        states=[([0, 1, 2], [0, 1]), ([0, 1, 3], [0, 1]), ([0, 1, 4], [0, 1])],
        weights=[1 / 3, 1 / 3, 1 / 3],
    ),
    dict(
        name="H2_ground_and_triplet",
        atoms=[("H", [0.0, 0.0, -0.7]), ("H", [0.1, 0.0, 0.7])],
        constant="BASIS_6_31G_GBS",
        spin=0,
        states=[([0], [0]), ([0, 1], [])],
        weights=[0.5, 0.5],
    ),
    dict(
        name="LiH_ground_and_open_shell_singlet_determinant",
        atoms=[("Li", [0.0, 0.0, 0.0]), ("H", [0.2, 0.1, 3.0])],
        constant="STO_3G_GBS",
        spin=0,
        states=[([0, 1], [0, 1]), ([0, 1], [0, 2]), ([0, 2], [0, 1])],
        weights=[0.6, 0.2, 0.2],
    ),
    dict(
        name="water_single_high_spin_triplet",
        atoms=[
            ("O", [0.1, -0.2, 0.3]),
            ("H", [1.5, 0.7, 0.4]),
            ("H", [-0.4, 1.3, -0.2]),
        ],
        constant="STO_3G_GBS",
        spin=0,
        states=[([0, 1, 2, 3, 4, 5], [0, 1, 2, 3])],
        weights=[1.0],
    ),
]


def canonical_orbitals(mol, spin):
    mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf


def ensemble(mf, c0, states, weights):
    h = mf.get_hcore()
    n_mo = c0.shape[1]
    occ = []
    for alpha, beta in states:
        na, nb = np.zeros(n_mo), np.zeros(n_mo)
        na[alpha], nb[beta] = 1, 1
        occ.append((na, nb))
    pairs = [
        (p, q)
        for p in range(n_mo)
        for q in range(p)
        if any(
            w > 0 and (o[0][p] != o[0][q] or o[1][p] != o[1][q])
            for o, w in zip(occ, weights)
        )
    ]

    def generator(kappa):
        x = np.zeros((n_mo, n_mo))
        for (p, q), k in zip(pairs, kappa):
            x[p, q], x[q, p] = k, -k
        return x

    def evaluate(kappa):
        """Orbitals, state energies and dE_SA/dC for C = C0 expm(X(kappa))."""
        c = c0 @ scipy.linalg.expm(generator(kappa))
        energies, d_c = [], np.zeros_like(c)
        for (na, nb), w in zip(occ, weights):
            pa, pb = (c * na) @ c.T, (c * nb) @ c.T
            j, k = mf.get_jk(dm=np.array([pa, pb]))
            fa, fb = h + j[0] + j[1] - k[0], h + j[0] + j[1] - k[1]
            energies.append(
                np.sum((pa + pb) * h) / 2
                + (np.sum(pa * fa) + np.sum(pb * fb)) / 2
                + mf.mol.energy_nuc()
            )
            d_c += 2 * w * (fa @ c * na + fb @ c * nb)
        return c, np.array(energies), d_c

    def energy_and_gradient(kappa):
        x = generator(kappa)
        _, energies, d_c = evaluate(kappa)
        # Adjoint of the expm Frechet derivative: dE/dX = L(X^T, C0^T dE/dC).
        d_x = scipy.linalg.expm_frechet(x.T, c0.T @ d_c, compute_expm=False)
        g = np.array([d_x[p, q] - d_x[q, p] for p, q in pairs])
        return float(np.dot(weights, energies)), g

    return pairs, evaluate, energy_and_gradient


def solve(model):
    mol = gto.M(
        atom=model["atoms"],
        basis=embedded_basis(model["constant"], sorted({a for a, _ in model["atoms"]})),
        cart=True,
        unit="Bohr",
        spin=model["spin"],
        verbose=0,
    )
    mf = canonical_orbitals(mol, model["spin"])
    c = mf.mo_coeff.copy()
    weights = np.array(model["weights"])
    for _ in range(20):
        pairs, evaluate, energy_and_gradient = ensemble(mf, c, model["states"], weights)
        zero = np.zeros(len(pairs))
        e0, g0 = energy_and_gradient(zero)
        # BFGS stalls near 1e-9 at the J/K roundoff floor; the energy error is then ~g^2.
        if np.max(np.abs(g0)) < 1e-8:
            break
        probe = np.random.default_rng(7).normal(size=len(pairs)) * 1e-3
        step = 1e-5
        fd = (
            energy_and_gradient(step * probe)[0] - energy_and_gradient(-step * probe)[0]
        ) / (2 * step)
        assert abs(fd - g0 @ probe) < 1e-9 * max(1, abs(fd)), (fd, g0 @ probe)
        result = scipy.optimize.minimize(
            energy_and_gradient,
            zero,
            jac=True,
            method="BFGS",
            options=dict(gtol=1e-12, maxiter=5000),
        )
        c = evaluate(result.x)[0]
    else:
        raise RuntimeError(model["name"])
    _, energies, _ = evaluate(zero)
    return mol, c, energies, float(np.dot(weights, energies)), float(np.max(np.abs(g0)))


def main():
    out = []
    for model in MODELS:
        mol, c, energies, average, residual = solve(model)
        order = rust_ao_permutation(mol, model["constant"])
        record = {k: v for k, v in model.items()}
        record["atoms"] = [dict(symbol=a, position=r) for a, r in model["atoms"]]
        record.update(
            sa_energy=average,
            state_energies=energies.tolist(),
            gradient_residual=residual,
            density=sum(
                w * ((c[:, a] @ c[:, a].T) + (c[:, b] @ c[:, b].T))
                for (a, b), w in zip(model["states"], model["weights"])
            )[np.ix_(order, order)].tolist(),
        )
        out.append(record)
        print(model["name"], average, energies, residual, flush=True)
    path = Path(__file__).resolve().parents[1] / "data/sa_hf_reference.json"
    path.write_text(
        json.dumps(
            dict(source=__doc__, pyscf_version=pyscf.__version__, cases=out), indent=1
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
