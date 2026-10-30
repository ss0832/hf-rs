# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Exact Fock matrices and exterior-algebra logarithm for density cumulants.

The independent oracle uses full 2^N Fock vectors and explicit annihilation
matrices. Cumulants are coefficients of log(1+Gamma1+Gamma2+Gamma3) under
the exterior product of paired creation/annihilation variables. No Rust
disconnected-density formula or sparse determinant-removal map is used.
"""

import itertools
import json
from pathlib import Path

import numpy as np


def strings(n, electrons):
    return sorted(
        sum(1 << i for i in group)
        for group in itertools.combinations(range(n), electrons)
    )


def parity(left, right):
    if set(left) & set(right):
        return 0
    return (-1) ** sum(p > q for p in left for q in right)


def wedge(a, b):
    out = {}
    for (left_a, right_a), va in a.items():
        for (left_b, right_b), vb in b.items():
            sl, sr = parity(left_a, left_b), parity(right_a, right_b)
            if sl and sr:
                key = (tuple(sorted(left_a + left_b)), tuple(sorted(right_a + right_b)))
                out[key] = out.get(key, 0.0) + sl * sr * va * vb
    return out


def record(name, n, members):
    ns = 2 * n
    groups = {
        rank: list(itertools.combinations(range(ns), rank)) for rank in range(1, 4)
    }
    create = []
    for p in range(ns):
        c = np.zeros((1 << ns, 1 << ns))
        for bits in range(1 << ns):
            if not (bits >> p) & 1:
                c[bits | (1 << p), bits] = (-1) ** ((bits & ((1 << p) - 1)).bit_count())
        create.append(c)
    densities = {rank: np.zeros((len(g), len(g))) for rank, g in groups.items()}
    total_weight = sum(m["weight"] for m in members)
    for member in members:
        alpha, beta = strings(n, member["na"]), strings(n, member["nb"])
        psi = np.zeros(1 << ns)
        for index, (a, b) in enumerate(itertools.product(alpha, beta)):
            occupied = [2 * p for p in range(n) if (a >> p) & 1]
            occupied += [2 * p + 1 for p in range(n) if (b >> p) & 1]
            determinant = np.zeros(1 << ns)
            determinant[0] = 1.0
            for p in reversed(occupied):
                determinant = create[p] @ determinant
            psi += member["coefficients"][index] * determinant
        psi /= np.linalg.norm(psi)
        for rank, tuples in groups.items():
            rows = []
            for group in tuples:
                vector = psi.copy()
                for p in group:
                    vector = create[p].T @ vector
                rows.append(vector)
            rows = np.array(rows)
            densities[rank] += member["weight"] / total_weight * (rows @ rows.T)
    gamma = {
        rank: {
            (a, b): densities[rank][i, j]
            for i, a in enumerate(tuples)
            for j, b in enumerate(tuples)
        }
        for rank, tuples in groups.items()
    }
    # The truncated exterior logarithm: X-X^2/2+X^3/3, X=G1+G2+G3.
    squared2 = wedge(gamma[1], gamma[1])
    squared3a, squared3b = wedge(gamma[1], gamma[2]), wedge(gamma[2], gamma[1])
    cubed3 = wedge(squared2, gamma[1])
    cumulants = {
        2: {key: value - squared2.get(key, 0.0) / 2 for key, value in gamma[2].items()},
        3: {
            key: value
            - (squared3a.get(key, 0.0) + squared3b.get(key, 0.0)) / 2
            + cubed3.get(key, 0.0) / 3
            for key, value in gamma[3].items()
        },
    }
    out = dict(name=name, n_spatial=n, members=members)
    for rank in (1, 2, 3):
        out[f"gamma{rank}"] = densities[rank].tolist()
    for rank in (2, 3):
        out[f"lambda{rank}"] = [
            [float(cumulants[rank][a, b]) for b in groups[rank]] for a in groups[rank]
        ]
    return out


def member(n, na, nb, coefficients, weight=1.0):
    assert len(coefficients) == len(strings(n, na)) * len(strings(n, nb))
    return dict(
        na=na, nb=nb, coefficients=list(map(float, coefficients)), weight=weight
    )


def main():
    n = 4
    dimension = len(strings(n, 2)) * len(strings(n, 1))
    plus = np.zeros(dimension)
    minus = np.zeros(dimension)
    # Two determinants differ by three substitutions but share N_alpha/N_beta.
    left = strings(n, 2).index(3) * len(strings(n, 1)) + strings(n, 1).index(1)
    right = strings(n, 2).index(12) * len(strings(n, 1)) + strings(n, 1).index(8)
    plus[left] = plus[right] = minus[left] = 1 / np.sqrt(2)
    minus[right] = -1 / np.sqrt(2)
    rng = np.random.default_rng(20261023)
    random = rng.normal(size=dimension)
    random /= np.linalg.norm(random)
    slater = np.zeros(dimension)
    slater[left] = 1.0
    # A rotated Slater determinant has non-diagonal gamma1 and zero higher cumulants.
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    occupied = q[:, :2]
    alpha_coeff = [
        np.linalg.det(occupied[[i for i in range(n) if (a >> i) & 1]])
        for a in strings(n, 2)
    ]
    beta_coeff = q[:, 0]
    rotated = np.outer(alpha_coeff, beta_coeff).ravel()
    one_alpha = member(n, 1, 0, [1.0, 0.0, 0.0, 0.0], 3.0)
    one_beta = member(n, 0, 1, [1.0, 0.0, 0.0, 0.0], 3.0)
    cases = [
        record("coherent_plus", n, [member(n, 2, 1, plus)]),
        record("coherent_minus", n, [member(n, 2, 1, minus)]),
        record("random_doublet_sector", n, [member(n, 2, 1, random)]),
        record("slater", n, [member(n, 2, 1, slater)]),
        record("rotated_slater", n, [member(n, 2, 1, rotated)]),
        record(
            "mixed_correlated",
            n,
            [member(n, 2, 1, plus, 2.0), member(n, 2, 1, random, 1.0)],
        ),
        record("spin_multiplet_average", n, [one_alpha, one_beta]),
        record("two_electron_correlated", n, [member(n, 1, 1, rng.normal(size=16))]),
        record("four_electron_correlated", n, [member(n, 2, 2, rng.normal(size=36))]),
    ]
    assert np.max(abs(np.array(cases[0]["gamma1"]) - cases[1]["gamma1"])) < 1e-14
    assert np.max(abs(np.array(cases[0]["gamma2"]) - cases[1]["gamma2"])) < 1e-14
    assert np.max(abs(np.array(cases[0]["lambda3"]) - cases[1]["lambda3"])) > 0.9
    output = dict(
        source="Independent full Fock creation/annihilation matrices and paired exterior-algebra logarithm through degree three",
        cases=cases,
    )
    Path(__file__).resolve().parents[1].joinpath(
        "data/density_cumulants_reference.json"
    ).write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    print("Exact density cumulants:", len(cases), "ensembles", flush=True)


if __name__ == "__main__":
    main()
