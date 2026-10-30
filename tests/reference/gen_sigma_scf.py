# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""sigma-SCF determinants from exact full-CI-space variances.

For unrestricted determinant orbitals (C_a, C_b) the determinant is a unit vector in the FCI
space of its own orbitals; the variance <(H - omega)^2> = |H Phi|^2 - 2 omega <H> + omega^2 is
evaluated with PySCF direct_uhf sigma vectors, independently of the closed-form
singles/doubles expression used by hf-rs. The variance is minimized by SciPy BFGS over
occupied-virtual rotations of each spin, with central-difference gradients, starting from
PySCF RHF canonical orbitals with the occupied orbitals moved to the front. Contracted
Cartesian AO space, embedded basis data. No sigma-SCF implementation source is read.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
import scipy.linalg
import scipy.optimize
from pyscf import ao2mo, fci, gto, lib, scf
from pyscf.fci import cistring

from embedded_basis import embedded_basis

lib.num_threads(4)

MODELS = [
    dict(
        name="H2_doubly_excited",
        atoms=[("H", [0.0, 0.0, -0.7]), ("H", [0.1, 0.0, 0.7])],
        constant="BASIS_6_31G_GBS",
        charge=0,
        alpha=[1],
        beta=[1],
        omega=-0.2,
    ),
    dict(
        name="H2_open_shell_determinant",
        atoms=[("H", [0.0, 0.0, -0.7]), ("H", [0.1, 0.0, 0.7])],
        constant="BASIS_6_31G_GBS",
        charge=0,
        alpha=[0],
        beta=[1],
        omega=-0.6,
    ),
    dict(
        name="HeH_cation_excited",
        atoms=[("He", [0.0, 0.0, 0.0]), ("H", [0.0, 0.1, 1.46])],
        constant="BASIS_6_31G_GBS",
        charge=1,
        alpha=[0],
        beta=[1],
        omega=-2.3,
    ),
    dict(
        name="LiH_HOMO_LUMO_triplet",
        atoms=[("Li", [0.0, 0.0, 0.0]), ("H", [0.2, 0.1, 3.0])],
        constant="STO_3G_GBS",
        charge=0,
        alpha=[0, 1, 2],
        beta=[0],
        omega=-7.75,
    ),
]


def variance_function(mol, c0a, c0b, na, nb, omega):
    h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    n = c0a.shape[1]
    address = (
        cistring.str2addr(n, na, (1 << na) - 1),
        cistring.str2addr(n, nb, (1 << nb) - 1),
    )
    shape = cistring.num_strings(n, na), cistring.num_strings(n, nb)
    pairs = [(a, i, 0) for a in range(na, n) for i in range(na)]
    pairs += [(a, i, 1) for a in range(nb, n) for i in range(nb)]

    def orbitals(kappa):
        xs = [np.zeros((n, n)), np.zeros((n, n))]
        for (a, i, s), k in zip(pairs, kappa):
            xs[s][a, i], xs[s][i, a] = k, -k
        return c0a @ scipy.linalg.expm(xs[0]), c0b @ scipy.linalg.expm(xs[1])

    def evaluate(kappa):
        ca, cb = orbitals(kappa)
        h1 = (ca.T @ h @ ca, cb.T @ h @ cb)
        eri = (
            ao2mo.restore(1, ao2mo.general(mol, (ca, ca, ca, ca)), n),
            ao2mo.general(mol, (ca, ca, cb, cb), compact=False).reshape(n, n, n, n),
            ao2mo.restore(1, ao2mo.general(mol, (cb, cb, cb, cb)), n),
        )
        h2 = fci.direct_uhf.absorb_h1e(h1, eri, n, (na, nb), 0.5)
        phi = np.zeros(shape)
        phi[address] = 1.0
        h_phi = (
            fci.direct_uhf.contract_2e(h2, phi, n, (na, nb)) + mol.energy_nuc() * phi
        )
        energy = float(h_phi[address])
        return float(np.sum(h_phi**2) - 2 * omega * energy + omega**2), energy

    def value_and_gradient(kappa, step=1e-5):
        value = evaluate(kappa)[0]
        g = np.zeros(len(kappa))
        for k in range(len(kappa)):
            e = np.zeros(len(kappa))
            e[k] = step
            g[k] = (evaluate(kappa + e)[0] - evaluate(kappa - e)[0]) / (2 * step)
        return value, g

    return pairs, orbitals, evaluate, value_and_gradient


def front(c, occupied):
    order = list(occupied) + [k for k in range(c.shape[1]) if k not in occupied]
    return c[:, order]


def solve(model):
    mol = gto.M(
        atom=model["atoms"],
        basis=embedded_basis(model["constant"], sorted({a for a, _ in model["atoms"]})),
        charge=model["charge"],
        cart=True,
        unit="Bohr",
        verbose=0,
    )
    rhf = scf.RHF(mol).run(conv_tol=1e-12)
    ca, cb = front(rhf.mo_coeff, model["alpha"]), front(rhf.mo_coeff, model["beta"])
    na, nb = len(model["alpha"]), len(model["beta"])
    for _ in range(60):
        pairs, orbitals, evaluate, value_and_gradient = variance_function(
            mol, ca, cb, na, nb, model["omega"]
        )
        zero = np.zeros(len(pairs))
        value, g = value_and_gradient(zero)
        if np.max(np.abs(g)) < 1e-8:
            break
        result = scipy.optimize.minimize(
            value_and_gradient,
            zero,
            jac=True,
            method="BFGS",
            options=dict(gtol=1e-10, maxiter=2000),
        )
        ca, cb = orbitals(result.x)
    else:
        raise RuntimeError(model["name"])
    variance, energy = evaluate(zero)
    return dict(
        variance=variance, energy=energy, gradient_residual=float(np.max(np.abs(g)))
    )


def main():
    out = []
    for model in MODELS:
        r = solve(model)
        record = dict(model)
        record["atoms"] = [dict(symbol=a, position=p) for a, p in model["atoms"]]
        record.update(r)
        out.append(record)
        print(model["name"], r, flush=True)
    path = Path(__file__).resolve().parents[1] / "data/sigma_scf_reference.json"
    path.write_text(
        json.dumps(
            dict(source=__doc__, pyscf_version=pyscf.__version__, cases=out), indent=1
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
