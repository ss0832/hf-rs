# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Divide-and-conquer HF fixed points from public PySCF integrals and NumPy.

Definition (Yang and Lee, J. Chem. Phys. 103, 5674 (1995); DC-HF as in Akama, Kobayashi
and Nakai, J. Comput. Chem. 28, 2003 (2007)): every subsystem has a central atom set S and a
buffer B (other atoms within the buffer radius of any central atom). Its localization region
L = S + B has the AO submatrices of the global Fock operator F[P] and overlap. With a common
Fermi level mu and occupations f = 2 / (1 + exp(beta (eps - mu))),

  P = sum_a sum_k f_k (p^a o C_k C_k^T),   p^a_mn = 1 (m, n in S), 1/2 (one in S, other in B),

where o is the elementwise product, and mu solves Tr(P S) = N. The fixed point P = P_out[P]
is found by Pulay mixing of the density residual; E = Tr(P h)/2 + Tr(P F[P])/2 + E_nuc.
Contracted Cartesian AO space, embedded basis data. No DC implementation source is read.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from gen_ghf import rust_ao_permutation
from pyscf import gto, lib, scf
from scipy.linalg import eigh
from scipy.special import expit

from embedded_basis import embedded_basis

lib.num_threads(4)


def h_chain():
    positions, z = [], 0.0
    for k in range(8):
        positions.append([0.05 * (k % 3), 0.0, z])
        z += 1.4 if k % 2 == 0 else 2.6
    return [("H", p) for p in positions], [[2 * k, 2 * k + 1] for k in range(4)]


def water_chain():
    atoms, central = [], []
    for k in range(3):
        x = 5.4 * k
        start = len(atoms)
        atoms += [
            ("O", [x, 0.1 * k, 0.0]),
            ("H", [x + 1.43, 0.0, 1.1]),
            ("H", [x - 1.43, 0.0, 1.1]),
        ]
        central.append([start, start + 1, start + 2])
    return atoms, central


def hexatriene():
    carbons = [[0.0, 0.0, 0.0], [2.33, 1.35, 0.0], [4.66, 0.0, 0.0]]
    carbons += [[7.0, 1.35, 0.0], [9.33, 0.0, 0.0], [11.66, 1.35, 0.0]]
    atoms = [("C", c) for c in carbons]
    hydrogens = [
        [-1.75, 1.0, 0.0],
        [0.0, -2.05, 0.0],
        [2.33, 3.4, 0.0],
        [4.66, -2.05, 0.0],
        [7.0, 3.4, 0.0],
        [9.33, -2.05, 0.0],
        [11.66, 3.4, 0.0],
        [13.4, 0.35, 0.0],
    ]
    atoms += [("H", h) for h in hydrogens]
    central = [[0, 6, 7], [1, 8], [2, 9], [3, 10], [4, 11], [5, 12, 13]]
    return atoms, central


MODELS = [
    dict(name="H8_chain", builder=h_chain, constant="STO_3G_GBS", buffer=3.0),
    dict(
        name="water_chain", builder=water_chain, constant="BASIS_6_31G_GBS", buffer=4.5
    ),
    dict(name="hexatriene", builder=hexatriene, constant="STO_3G_GBS", buffer=4.0),
]
BETA = 200.0


def subsystems(atoms, central, radius):
    out = []
    for s in central:
        buffer = [
            a
            for a in range(len(atoms))
            if a not in s
            and min(np.linalg.norm(np.subtract(atoms[a][1], atoms[c][1])) for c in s)
            <= radius
        ]
        out.append((sorted(s), sorted(buffer)))
    return out


def solve(model, radius=None):
    atoms, central = model["builder"]()
    radius = model["buffer"] if radius is None else radius
    mol = gto.M(
        atom=atoms,
        basis=embedded_basis(model["constant"], sorted({a for a, _ in atoms})),
        cart=True,
        unit="Bohr",
        verbose=0,
    )
    rhf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-10)
    h, s = rhf.get_hcore(), rhf.get_ovlp()
    slices = mol.aoslice_by_atom()
    ao = [list(range(slices[a, 2], slices[a, 3])) for a in range(mol.natm)]
    regions = []
    for core, buffer in subsystems(atoms, central, radius):
        local = sorted(sum((ao[a] for a in core + buffer), []))
        in_core = np.array([any(m in ao[a] for a in core) for m in local])
        p = 0.5 * (in_core[:, None].astype(float) + in_core[None, :].astype(float))
        regions.append((local, p))
    n = mol.nelectron

    def output(density):
        fock = h + rhf.get_veff(dm=density)
        blocks = []
        for local, p in regions:
            ix = np.ix_(local, local)
            eps, c = eigh(fock[ix], s[ix])
            q = np.einsum("mk,mn,nk,mn->k", c, s[ix], c, p)
            blocks.append((local, p, eps, c, q))
        energies = np.concatenate([b[2] for b in blocks])

        def count(mu):
            return sum(np.sum(2.0 * expit(-BETA * (b[2] - mu)) * b[4]) for b in blocks)

        lo, hi = energies.min() - 1.0, energies.max() + 1.0
        assert count(lo) < n < count(hi)
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if count(mid) < n else (lo, mid)
        mu = 0.5 * (lo + hi)
        result = np.zeros_like(h)
        for local, p, eps, c, _ in blocks:
            f = 2.0 * expit(-BETA * (eps - mu))
            result[np.ix_(local, local)] += p * ((c * f) @ c.T)
        return result, mu

    density = rhf.make_rdm1()
    inputs, residuals = [], []
    for iteration in range(500):
        out, mu = output(density)
        residual = out - density
        if np.max(np.abs(residual)) < 1e-12:
            break
        inputs.append(out)
        residuals.append(residual)
        inputs, residuals = inputs[-8:], residuals[-8:]
        m = len(residuals)
        b = -np.ones((m + 1, m + 1))
        b[m, m] = 0.0
        for i in range(m):
            for j in range(m):
                b[i, j] = np.sum(residuals[i] * residuals[j])
        rhs = np.zeros(m + 1)
        rhs[m] = -1.0
        coefficients = np.linalg.solve(b, rhs)[:m]
        density = sum(c * x for c, x in zip(coefficients, inputs))
    else:
        raise RuntimeError(model["name"])
    fock = h + rhf.get_veff(dm=density)
    energy = 0.5 * np.sum(density * (h + fock)) + mol.energy_nuc()
    assert abs(np.sum(density * s) - n) < 1e-10
    return mol, rhf, density, energy, mu, iteration


def main():
    cases = []
    for model in MODELS:
        mol, rhf, density, energy, mu, iterations = solve(model)
        order = rust_ao_permutation(mol, model["constant"])
        atoms, central = model["builder"]()
        record = dict(
            name=model["name"],
            constant=model["constant"],
            atoms=[dict(symbol=a, position=p) for a, p in atoms],
            central_regions=central,
            buffer_radius=model["buffer"],
            inverse_temperature=BETA,
            energy=float(energy),
            rhf_energy=float(rhf.e_tot),
            chemical_potential=float(mu),
            density=density[np.ix_(order, order)].tolist(),
        )
        cases.append(record)
        print(model["name"], energy, energy - rhf.e_tot, mu, iterations, flush=True)
    path = Path(__file__).resolve().parents[1] / "data/dc_hf_reference.json"
    path.write_text(
        json.dumps(
            dict(source=__doc__, pyscf_version=pyscf.__version__, cases=cases), indent=1
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
