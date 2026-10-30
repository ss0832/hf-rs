# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Off-axis, multiple-occupied-pair 3*A nonlocal-ECP oracle.

Direct coupled-Gaussian polynomial quadrature gives the full small AO operators.
Dense product-space operators apply both nonlocal ket actions and the projector;
the Rust implementation instead contracts occupied ket legs before storing them.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pyscf
from numpy.polynomial.hermite import hermgauss
from pyscf import gto, lib, scf

lib.num_threads(4)
positions = np.array([[-0.1, 0.2, -1.3], [0.3, -0.2, 1.4]])
obs_exp = [1.4, 0.45]
aux_exp = [0.2, 0.8, 0.8, 0.8]
exponents = np.array(obs_exp * 2 + aux_exp * 2)
powers = np.array([[0, 0, 0]] * 4 + ([[0, 0, 0]] + np.eye(3, dtype=int).tolist()) * 2)
centers = np.concatenate(
    (np.repeat(positions, 2, axis=0), np.repeat(positions, 4, axis=0))
)
norms = (2 * exponents / np.pi) ** 0.75 * np.sqrt((4 * exponents) ** powers.sum(axis=1))
terms = [(0.3, -0.4), (1.2, -0.2)]
gammas = np.array([0.3, 1.2, 0.6, 1.5, 2.4])
nodes, weights = hermgauss(6)
weights2 = weights[:, None] * weights[None, :]


def quartet(indices):
    alpha = exponents[list(indices)]
    c = centers[list(indices)]
    angular = powers[list(indices)]
    p, q = alpha[:2].sum(), alpha[2:].sum()
    det = p * q + gammas * (p + q)
    inv00, inv01, inv11 = (q + gammas) / det, gammas / det, (p + gammas) / det
    a00 = np.sqrt(inv00)
    a10 = inv01 / a00
    a11 = np.sqrt(inv11 - a10**2)
    base, distance, kinetic = [], [], []
    for axis in range(3):
        b0, b1 = alpha[:2] @ c[:2, axis], alpha[2:] @ c[2:, axis]
        mean0, mean1 = inv00 * b0 + inv01 * b1, inv01 * b0 + inv11 * b1
        x = mean0[:, None, None] + a00[:, None, None] * nodes[None, :, None]
        y = (
            mean1[:, None, None]
            + a10[:, None, None] * nodes[None, :, None]
            + a11[:, None, None] * nodes[None, None, :]
        )
        coordinates = [(x if i < 2 else y) - c[i, axis] for i in range(4)]
        polynomial = np.ones_like(y)
        for i in range(4):
            polynomial *= coordinates[i] ** angular[i, axis]
        differential = np.zeros_like(polynomial)
        for i in range(4):
            z, degree = coordinates[i], angular[i, axis]
            laplacian = (
                4 * alpha[i] ** 2 * z ** (degree + 2)
                - 2 * alpha[i] * (2 * degree + 1) * z**degree
            )
            for k in range(4):
                if k != i:
                    laplacian = laplacian * coordinates[k] ** angular[k, axis]
            differential += (0.5 if i % 2 == 0 else -0.5) * laplacian
        prefactor = np.exp(
            -alpha @ (c[:, axis] ** 2) + b0 * mean0 + b1 * mean1
        ) / np.sqrt(det)
        base.append(prefactor * np.sum(weights2 * polynomial, axis=(1, 2)))
        distance.append(
            prefactor * np.sum(weights2 * polynomial * (x - y) ** 2, axis=(1, 2))
        )
        kinetic.append(prefactor * np.sum(weights2 * differential, axis=(1, 2)))
    normal = np.prod(norms[list(indices)])
    gaussian = normal * np.prod(base, axis=0)
    r2 = normal * sum(
        distance[a] * base[(a + 1) % 3] * base[(a + 2) % 3] for a in range(3)
    )
    u = normal * sum(
        kinetic[a] * base[(a + 1) % 3] * base[(a + 2) % 3] for a in range(3)
    )
    return np.array(
        [
            -0.4 * gaussian[0] - 0.2 * gaussian[1],
            0.16 * gaussian[2] + 0.16 * gaussian[3] + 0.04 * gaussian[4],
            -0.4 * u[0] - 0.2 * u[1],
            4
            * (
                0.3**2 * 0.16 * r2[2]
                + 2 * 0.3 * 1.2 * 0.08 * r2[3]
                + 1.2**2 * 0.04 * r2[4]
            ),
        ]
    )


def local_element(a, b):
    # Independent three-center polynomial quadrature for the local ECP channel.
    total = 0.0
    for center in positions:
        alpha, beta, gamma = exponents[a], exponents[b], 0.7
        p = alpha + beta + gamma
        mean = (alpha * centers[a] + beta * centers[b] + gamma * center) / p
        value = 0.17 * norms[a] * norms[b]
        for axis in range(3):
            x = mean[axis] + nodes / np.sqrt(p)
            value *= (
                np.exp(
                    -alpha * centers[a, axis] ** 2
                    - beta * centers[b, axis] ** 2
                    - gamma * center[axis] ** 2
                    + p * mean[axis] ** 2
                )
                / np.sqrt(p)
                * np.sum(
                    weights
                    * (x - centers[a, axis]) ** powers[a, axis]
                    * (x - centers[b, axis]) ** powers[b, axis]
                )
            )
        total += value
    return total


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
mol = gto.M(
    atom=[("He", p) for p in positions],
    basis={"He": [[0, [a, 1.0]] for a in obs_exp]},
    ecp=ecp,
    unit="Bohr",
    cart=True,
    verbose=0,
)
aux = gto.M(
    atom=[("ghost-He", p) for p in positions],
    basis={"He": [[0, [0.2, 1.0]], [1, [0.8, 1.0]]]},
    unit="Bohr",
    cart=True,
    verbose=0,
)
union = gto.conc_mol(mol, aux)
hf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-12)
assert hf.converged
nao, nm, no, n = mol.nao, hf.mo_coeff.shape[1], mol.nelectron // 2, union.nao
assert n == len(exponents)
overlap = union.intor("int1e_ovlp")
projection = hf.mo_coeff @ hf.mo_coeff.T @ overlap[:nao, nao:]
eigenvalues, rotation = np.linalg.eigh(
    overlap[nao:, nao:] - overlap[nao:, :nao] @ projection
)
caux = rotation[:, eigenvalues > 1e-8] / np.sqrt(eigenvalues[eigenvalues > 1e-8])
ri = np.zeros((n, nm + caux.shape[1]))
ri[:nao, :nm] = hf.mo_coeff
ri[:, nm:] = np.vstack((-projection @ caux, caux))
nri = ri.shape[1]
local = np.array([[local_element(a, b) for b in range(n)] for a in range(n)])
potential = ri.T @ (union.intor("ECPscalar") - local) @ ri
potential_pair = np.kron(potential, np.eye(nri)) + np.kron(np.eye(nri), potential)
ao = np.empty((4, n, n, n, n))
for indices in itertools.product(range(n), repeat=4):
    ao[(slice(None), *indices)] = quartet(indices)
factor, square, u, half = [
    np.einsum("ap,br,cq,ds,abcd->pqrs", ri, ri, ri, ri, tensor, optimize=True).reshape(
        nri**2, nri**2
    )
    for tensor in ao
]
removed = np.array(
    [
        (p < nm and q < nm) or (p >= nm and q < no) or (p < no and q >= nm)
        for p in range(nri)
        for q in range(nri)
    ]
)
q = np.diag((~removed).astype(float))
comm = factor @ potential_pair - potential_pair @ factor
unit = factor @ potential_pair @ factor - 0.5 * (
    potential_pair @ square + square @ potential_pair
)
projected = -0.5 * (
    comm @ (np.eye(nri**2) - q) @ factor - factor @ (np.eye(nri**2) - q) @ comm
)
kinetic_projected = -0.5 * (
    u @ (np.eye(nri**2) - q) @ factor - factor @ (np.eye(nri**2) - q) @ u
)
bbar = half + kinetic_projected + unit + projected
pairs = []
for i in range(no):
    for j in range(i, no):
        ij, ji = i * nri + j, j * nri + i
        pairs.append(
            dict(
                pair=[i, j],
                b_bar=[float(bbar[ij, ij]), float(bbar[ij, ji])],
                nonlocal_correction=[
                    float((unit + projected)[ij, ij]),
                    float((unit + projected)[ij, ji]),
                ],
            )
        )
out = dict(
    source=f"PySCF {pyscf.__version__}, independent GH-6 polynomial quadrature and dense two-electron projector/nonlocal operators",
    positions=positions.tolist(),
    terms=terms,
    e_scf=float(hf.e_tot),
    pairs=pairs,
)
Path(__file__).resolve().parents[1].joinpath(
    "data/f12_nonlocal_reference.json"
).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print("Nonlocal S/P ECP, all off-axis pairs:", pairs, flush=True)
