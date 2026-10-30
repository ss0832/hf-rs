# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent PySCF spin-free IC-CASPT2 matrix and NumPy solve oracles.

Small full determinant spaces are used only in this oracle. PySCF supplies
Hamiltonian and excitation actions; NumPy handles overlap/Fock diagonalization
and direct linear solves, independently of the production sparse CAR/PCG path.
"""

import json
import math
from pathlib import Path

import numpy as np
from pyscf import fci, lib
from pyscf.fci import direct_nosym

lib.num_threads(4)


def operators(nc, na, nv):
    c, t, v = range(nc), range(nc, nc + na), range(nc + na, nc + na + nv)
    result = [[] for _ in range(8)]
    if not na:
        for a in v:
            for i in c:
                result[3].append(((a, i),))
    for i in c:
        for p in t:
            for q in t:
                for r in t:
                    result[0].append(((p, i), (q, r)))
    for i in c:
        for j in c:
            for p in t:
                for q in t:
                    if (p, i) <= (q, j):
                        result[1].append(((p, i), (q, j)))
    for a in v:
        for p in t:
            for q in t:
                for r in t:
                    result[2].append(((a, p), (q, r)))
    for a in v:
        for i in c:
            for p in t:
                for q in t:
                    result[3].extend([((a, i), (p, q)), ((p, i), (a, q))])
    for i in c:
        for j in c:
            for p in t:
                for a in v:
                    result[4].append(((p, i), (a, j)))
    for a in v:
        for b in v:
            for p in t:
                for q in t:
                    if (a, p) <= (b, q):
                        result[5].append(((a, p), (b, q)))
    for a in v:
        for b in v:
            for i in c:
                for p in t:
                    result[6].append(((a, i), (b, p)))
    for a in v:
        for b in v:
            for i in c:
                for j in c:
                    if (a, i) <= (b, j):
                        result[7].append(((a, i), (b, j)))
    return result


def problem(nc, na, nv, nelec):
    n = nc + na + nv
    h = np.zeros((n, n))
    for p in range(n):
        h[p, p] = (
            -3.0 + 0.4 * p
            if p < nc
            else (-0.9 + 0.13 * (p - nc) if p < nc + na else 1.1 + 0.3 * (p - nc - na))
        )
        for q in range(p):
            h[p, q] = h[q, p] = 0.027 * np.cos(p + 2 * q + 1)
    factors = np.array(
        [
            [[np.cos((p + 1) * (q + 1) + k) for q in range(n)] for p in range(n)]
            for k in range(3)
        ]
    )
    g = 0.018 * np.einsum("kpq,krs->pqrs", factors, factors)
    active = slice(nc, nc + na)
    ha = h[active, active].copy()
    for i in range(nc):
        ha += 2 * g[active, active, i, i] - g[active, i, i, active]
    ga = g[active, active, active, active].copy()
    energy, ci = fci.direct_spin1.kernel(ha, ga, na, nelec, tol=1e-14)
    one = fci.direct_spin1.make_rdm1(ci, na, nelec)
    record = contracted_problem(h, g, nc, na, nelec, ci, one)
    record["active_energy"] = float(energy)
    return record


def contracted_problem(
    h, g, nc, na, nelec, ci, one, ipea_values=(0.0, 0.25), shifts=None
):
    """Independent full-sector matrix oracle; restricted to small tests."""
    n = len(h)
    nv = n - nc - na
    active = slice(nc, nc + na)
    da = np.zeros((n, n))
    for i in range(nc):
        da[i, i] = 2.0
    da[active, active] = one
    fock = h + np.einsum("rs,pqrs->pq", da, g) - 0.5 * np.einsum("rs,prqs->pq", da, g)
    counts = (nc + nelec[0], nc + nelec[1])
    shape = tuple(math.comb(n, k) for k in counts)
    dim = math.prod(shape)
    alpha, beta = [fci.cistring.make_strings(range(n), k) for k in counts]
    addresses = {
        (int(a), int(b)): i * shape[1] + j
        for i, a in enumerate(alpha)
        for j, b in enumerate(beta)
    }
    reference = np.zeros(shape)
    strings = [fci.cistring.make_strings(range(na), k) for k in nelec]
    core_mask = (1 << nc) - 1
    for i, a in enumerate(strings[0]):
        for j, b in enumerate(strings[1]):
            address = addresses[
                (core_mask | (int(a) << nc), core_mask | (int(b) << nc))
            ]
            reference.ravel()[address] = ci[i, j]
    address, physical = fci.direct_spin1.pspace(h, g, n, counts, np=dim)
    full_h = np.zeros_like(physical)
    full_h[np.ix_(address, address)] = physical
    one_body = np.column_stack(
        [
            fci.direct_spin1.contract_1e(fock, column.reshape(shape), n, counts).ravel()
            for column in np.eye(dim)
        ]
    )

    def excitation(p, q, vector):
        matrix = np.zeros((n, n))
        matrix[p, q] = 1
        return direct_nosym.contract_1e(
            matrix, vector.reshape(shape), n, counts
        ).ravel()

    blocks, metrics, raw, corrections = [], [], [], []
    for group in operators(nc, na, nv):
        vectors, primitive_weights = [], []
        for product in group:
            vector = reference.ravel()
            for p, q in reversed(product):
                vector = excitation(p, q, vector)
            if np.any(vector):
                vectors.append(vector)
                primitive_weights.append(
                    sum(2 + da[p, p] - da[q, q] for p, q in product) / 2
                )
        raw.append(len(vectors))
        if not vectors:
            blocks.append(np.empty((dim, 0)))
            metrics.append([])
            corrections.append(np.empty((0, 0)))
            continue
        b = np.column_stack(vectors)
        values, x = np.linalg.eigh(b.T @ b)
        metrics.append(values.tolist())
        keep = values > 1e-10 * max(values)
        whitening = x[:, keep] / np.sqrt(values[keep])
        q = b @ whitening
        corrections.append(
            whitening.T @ np.diag(np.diag(b.T @ b) * primitive_weights) @ whitening
        )
        _, rotate = np.linalg.eigh(q.T @ one_body @ q)
        blocks.append(q @ rotate)
        corrections[-1] = rotate.T @ corrections[-1] @ rotate
    basis = np.column_stack(blocks)
    assert np.max(abs(basis.T @ basis - np.eye(basis.shape[1]))) < 1e-10
    e0 = np.einsum("pq,pq", fock, da)
    a = basis.T @ one_body @ basis - e0 * np.eye(basis.shape[1])
    rhs = basis.T @ full_h @ reference.ravel()
    results = []
    for ipea in ipea_values:
        shifted_ipea = a.copy()
        first = 0
        for block in corrections:
            size = len(block)
            shifted_ipea[first : first + size, first : first + size] += ipea * block
            first += size
        # Diagonalize each shifted intraclass block before defining the
        # diagonal Forsberg-Malmqvist imaginary-shift denominators.
        rotate = np.zeros_like(a)
        first = 0
        for block in blocks:
            size = block.shape[1]
            _, u = np.linalg.eigh(
                shifted_ipea[first : first + size, first : first + size]
            )
            rotate[first : first + size, first : first + size] = u
            first += size
        transformed = rotate.T @ shifted_ipea @ rotate
        coupling = rotate.T @ rhs
        for real, imag in shifts or [(0.0, 0.0), (0.17, 0.0), (0.0, 0.12)]:
            shifted = transformed + real * np.eye(len(a))
            if imag:
                shifted += np.diag(imag**2 / np.diag(transformed))
            t = np.linalg.solve(shifted, -coupling)
            first_order = basis @ rotate @ t
            energies = 2 * coupling * t + t * (transformed @ t)
            first, components = 0, []
            for block in blocks:
                components.append(float(sum(energies[first : first + block.shape[1]])))
                first += block.shape[1]
            results.append(
                dict(
                    ipea_shift=ipea,
                    real_shift=real,
                    imaginary_shift=imag,
                    energy=float(sum(energies)),
                    uncorrected=float(coupling @ t),
                    first_order=first_order.tolist(),
                    components=components,
                    zeroth_spectrum=np.linalg.eigvalsh(shifted_ipea).tolist(),
                )
            )
    return dict(
        name=f"core{nc}-active{na}-external{nv}-{nelec}",
        n=n,
        n_core=nc,
        n_active=na,
        n_alpha=nelec[0],
        n_beta=nelec[1],
        h=h.tolist(),
        eri=g.tolist(),
        active_ci=ci.ravel().tolist(),
        active_one=one.ravel().tolist(),
        reference_fock_energy=float(e0),
        raw_dimensions=raw,
        ranks=[b.shape[1] for b in blocks],
        metric_eigenvalues=metrics,
        zeroth_spectrum=np.linalg.eigvalsh(a).tolist(),
        determinants=[[int(a), int(b)] for a in alpha for b in beta],
        results=results,
    )


def main():
    cases = [
        problem(1, 4, 1, (2, 2)),
        problem(2, 2, 2, (1, 1)),
        problem(0, 2, 1, (1, 1)),
        problem(1, 3, 1, (2, 1)),
    ]
    Path(__file__).resolve().parents[1].joinpath(
        "data/caspt2_reference.json"
    ).write_text(
        json.dumps(dict(source=__doc__, cases=cases), indent=1) + "\n", encoding="utf-8"
    )
    print([(c["name"], c["ranks"], c["results"][0]["energy"]) for c in cases])


if __name__ == "__main__":
    main()
