# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""General effective CI operators and orbital changes from explicit Fock space."""

import itertools
import json
from pathlib import Path

import numpy as np
from gen_density_cumulants import strings
from gen_gno import creation
from gen_imsrg2 import FockOracle


def determinant_basis(create, n, na, nb):
    dimension = create[0].shape[0] if len(create) else 1
    columns = []
    for alpha, beta in itertools.product(strings(n, na), strings(n, nb)):
        indices = [2 * p for p in range(n) if (alpha >> p) & 1]
        indices += [2 * p + 1 for p in range(n) if (beta >> p) & 1]
        state = np.zeros(dimension)
        state[0] = 1.0
        for p in reversed(indices):
            state = create[p] @ state
        columns.append(state)
    return np.column_stack(columns)


def full(n, zero, one, pair):
    groups = list(itertools.combinations(range(n), 2))
    two = np.zeros((n,) * 4)
    for i, (p, q) in enumerate(groups):
        for j, (r, s) in enumerate(groups):
            two[p, q, r, s] = two[q, p, s, r] = pair[i, j]
            two[q, p, r, s] = two[p, q, s, r] = -pair[i, j]
    return dict(zero=zero, one=one.tolist(), two=two.ravel().tolist())


def physical(create, one, pair, constant):
    n = len(create)
    dimension = create[0].shape[0] if n else 1
    out = constant * np.eye(dimension)
    groups = list(itertools.combinations(range(n), 2))
    for p in range(n):
        for q in range(n):
            out += one[p, q] * (create[p] @ create[q].T)
    for i, (p, q) in enumerate(groups):
        left = create[p] @ create[q]
        for j, (r, s) in enumerate(groups):
            out += pair[i, j] * (left @ create[s].T @ create[r].T)
    return out


def generic_cases(rng):
    cases = []
    for n, na, nb in [(3, 2, 1), (4, 2, 1), (2, 1, 1), (0, 0, 0), (3, 3, 0)]:
        create = creation(2 * n)
        one = rng.normal(size=(2 * n,) * 2)
        one += one.T
        for p in range(2 * n):
            for q in range(2 * n):
                if p % 2 != q % 2:
                    one[p, q] = 0.0
        groups = list(itertools.combinations(range(2 * n), 2))
        pair = rng.normal(size=(len(groups),) * 2)
        pair += pair.T
        for i, (p, q) in enumerate(groups):
            for j, (r, s) in enumerate(groups):
                if p % 2 + q % 2 != r % 2 + s % 2:
                    pair[i, j] = 0.0
        h = physical(create, one, pair, 0.31)
        basis = determinant_basis(create, n, na, nb)
        matrix = basis.T @ h @ basis
        cases.append(
            dict(
                n_spatial=n,
                na=na,
                nb=nb,
                operator=full(2 * n, 0.31, one, pair),
                matrix=matrix.tolist(),
                energies=np.linalg.eigvalsh(matrix).tolist(),
            )
        )
    return cases


def spin_scalar(rng):
    n = 3
    create = creation(2 * n)
    e = np.array(
        [
            [
                sum(create[2 * p + spin] @ create[2 * q + spin].T for spin in range(2))
                for q in range(n)
            ]
            for p in range(n)
        ]
    )
    f = rng.normal(size=(n, n))
    f += f.T
    a = rng.normal(size=(n, n))
    a -= a.T
    generator = np.einsum("pq,pqij->ij", a, e)
    h = (
        0.11 * np.eye(1 << (2 * n))
        + np.einsum("pq,pqij->ij", f, e)
        + 0.13 * generator @ generator
    )
    oracle = FockOracle([False] * (2 * n))
    bare, _ = oracle.project(h)
    plus = sum(create[2 * p] @ create[2 * p + 1].T for p in range(n))
    sz = sum(
        0.5
        * (create[2 * p] @ create[2 * p].T - create[2 * p + 1] @ create[2 * p + 1].T)
        for p in range(n)
    )
    ss = 0.5 * (plus @ plus.T + plus.T @ plus) + sz @ sz
    basis = determinant_basis(create, n, 1, 1)
    matrix = basis.T @ h @ basis
    spin = basis.T @ ss @ basis
    values, vectors = np.linalg.eigh(spin)
    roots = []
    for target in (0.0, 2.0):
        selected = vectors[:, abs(values - target) < 1e-10]
        energies, ci = np.linalg.eigh(selected.T @ matrix @ selected)
        coefficients = selected @ ci
        roots.append(
            dict(
                s_squared=target,
                energies=energies[:2].tolist(),
                projectors=[np.outer(c, c).tolist() for c in coefficients.T[:2]],
            )
        )
    return dict(
        n_spatial=n,
        na=1,
        nb=1,
        operator=oracle.full(bare),
        matrix=matrix.tolist(),
        spin_roots=roots,
    )


def projection_case(rng):
    n = 8
    create = creation(n)
    one = rng.normal(scale=0.2, size=(n, n))
    one += one.T
    groups = list(itertools.combinations(range(n), 2))
    pair = rng.normal(scale=0.2, size=(len(groups),) * 2)
    pair += pair.T
    h = physical(create, one, pair, -0.19)
    core, active = [0, 1], [4, 5, 2, 3]
    columns = []
    for bits in range(1 << len(active)):
        indices = core + [p for i, p in enumerate(active) if (bits >> i) & 1]
        state = np.zeros(1 << n)
        state[0] = 1.0
        for p in reversed(indices):
            state = create[p] @ state
        columns.append(state)
    basis = np.column_stack(columns)
    projected = basis.T @ h @ basis
    oracle = FockOracle([False] * len(active))
    coeff, _ = oracle.project(projected)
    return dict(
        operator=full(n, -0.19, one, pair),
        core=core,
        active=active,
        projected=oracle.full(coeff),
    )


def rotations(rng):
    n = 4
    create = creation(2 * n)
    cases = []
    for na, nb in ((2, 1), (1, 1), (0, 0), (4, 4)):
        for reflection in (False, True):
            u, _ = np.linalg.qr(rng.normal(size=(n, n)))
            if (np.linalg.det(u) < 0) != reflection:
                u[:, 0] *= -1
            rotated = []
            for p in range(n):
                for spin in range(2):
                    rotated.append(
                        sum(u[q, p] * create[2 * q + spin] for q in range(n))
                    )
            old = determinant_basis(create, n, na, nb)
            new = determinant_basis(rotated, n, na, nb)
            ci = rng.normal(size=old.shape[1])
            ci /= np.linalg.norm(ci)
            transformed = new.T @ old @ ci
            assert np.linalg.norm(new @ transformed - old @ ci) < 1e-12
            cases.append(
                dict(
                    n_spatial=n,
                    na=na,
                    nb=nb,
                    u=u.tolist(),
                    coefficients=ci.tolist(),
                    rotated=transformed.tolist(),
                    reflection=reflection,
                )
            )
    return cases


def main():
    rng = np.random.default_rng(20261024)
    data = dict(
        source=__doc__,
        operators=generic_cases(rng),
        spin_scalar=spin_scalar(rng),
        core_projection=projection_case(rng),
        rotations=rotations(rng),
    )
    Path(__file__).resolve().parents[1].joinpath(
        "data/mr_relaxation_reference.json"
    ).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(
        "General operators, spin targeting, full core projection and eight CI rotations generated",
        flush=True,
    )


if __name__ == "__main__":
    main()
