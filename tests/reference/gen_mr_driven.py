# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""MR driven equations: full Fock GNO projection, Pade exponential and roots.

Perturbative coefficients come from refined finite-lambda polynomial fits to
the converged nonlinear solution, independently of the production expansion.
No external MR implementation is imported or read.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from gen_gno import GeneralizedOracle, random_operator
from scipy.linalg import expm
from scipy.optimize import root
from scipy.sparse.linalg import expm_multiply


def driven(
    oracle,
    h,
    active,
    core,
    flow,
    modelspace=False,
    compute_perturbation=True,
    exponential_action=False,
    conserve_spin=False,
    bch_order=None,
):
    if bch_order not in (None, 2):
        raise ValueError("Only the converged BCH or fixed quadratic BCH is defined")
    n = oracle.n
    active_order, core_order = list(active), list(core)
    active, core = set(active), set(core)
    holes = sorted(active | core)
    particles = sorted(set(range(n)) - core)
    singles = [
        (a, i)
        for a in particles
        for i in holes
        if {a, i} - active and (not conserve_spin or a % 2 == i % 2)
    ]
    pairs = oracle.groups[2]
    pair_index = {p: i for i, p in enumerate(pairs)}
    doubles = [
        (a, b, i, j)
        for a, b in itertools.combinations(particles, 2)
        for i, j in itertools.combinations(holes, 2)
        if {a, b, i, j} - active
        and (not conserve_spin or a % 2 + b % 2 == i % 2 + j % 2)
    ]
    excitation = np.array(
        [1 + a * n + i for a, i in singles]
        + [
            1 + n * n + pair_index[a, b] * len(pairs) + pair_index[i, j]
            for a, b, i, j in doubles
        ],
        dtype=int,
    )
    reverse = np.array(
        [1 + i * n + a for a, i in singles]
        + [
            1 + n * n + pair_index[i, j] * len(pairs) + pair_index[a, b]
            for a, b, i, j in doubles
        ],
        dtype=int,
    )
    eps = h[1 : 1 + n * n].reshape(n, n).diagonal()
    gaps = np.array(
        [eps[i] - eps[a] for a, i in singles]
        + [eps[i] + eps[j] - eps[a] - eps[b] for a, b, i, j in doubles]
    )
    regularized = np.divide(
        -np.expm1(-flow * gaps**2), gaps, out=np.zeros_like(gaps), where=gaps != 0
    )
    # Each matrix is an independently projected full Fock commutator map.
    maps = []
    sectors = [
        oracle.normal2[:, indices, :][:, :, indices] for indices in oracle.base.sectors
    ]
    for forward, backward in zip(excitation, reverse):
        am = oracle.normal2[forward] - oracle.normal2[backward]
        # Particle-number sectors 0..3 uniquely determine every rank <=3
        # coefficient. These are exact full Fock blocks, including signs.
        samples = []
        for indices, basis in zip(oracle.base.sectors, sectors, strict=True):
            block = am[np.ix_(indices, indices)]
            samples.append(
                (basis @ block - block @ basis).reshape(oracle.retained, -1).T
            )
        from scipy.linalg import lu_solve

        maps.append(
            lu_solve(oracle.lu, np.concatenate(samples, axis=0))[: oracle.retained]
        )
    maps = np.array(maps)

    def solve(h, guess=None, verify=False):
        if not len(excitation):
            return np.zeros(oracle.retained), h.copy(), 0.0

        def transform(t):
            matrix = np.einsum("a,aij->ij", t, maps)
            if bch_order == 2:
                linear = matrix @ h
                return h + linear + 0.5 * matrix @ linear
            if exponential_action:
                return expm_multiply(matrix, h, traceA=float(np.trace(matrix)))
            return expm(matrix) @ h

        def residual(t):
            bar = transform(t)
            return (bar[excitation] + gaps * t) * regularized - t

        initial = h[excitation] * regularized if guess is None else guess
        solution = root(
            residual,
            initial,
            method="broyden1",
            options={
                "fatol": 1e-14,
                "xatol": 1e-14,
                "maxiter": 300,
                "jac_options": {"alpha": 1.0},
            },
        )
        error = float(np.max(abs(residual(solution.x))))
        assert error < 1e-12, (solution.message, error)
        generator = np.zeros(oracle.retained)
        generator[excitation] = solution.x
        generator[reverse] = -solution.x
        bar = transform(solution.x)
        if verify and exponential_action and bch_order is None:
            pade = expm(np.einsum("a,aij->ij", solution.x, maps)) @ h
            assert np.max(abs(pade - bar)) < 1e-12
        return generator, bar, error

    a, bar, residual = solve(h, verify=True)
    base_result = dict(
        flow=flow,
        generator=oracle.base.full(a),
        transformed=oracle.base.full(bar),
        residual=residual,
        excitation_count=len(excitation),
    )
    if not compute_perturbation:
        return base_result
    h0 = np.zeros_like(h)
    h0[0] = h[0]
    h0[1 : 1 + n * n] = np.diag(eps).ravel()
    h1 = h - h0
    h0[0] = 0.0

    def fit(radius):
        nodes = np.linspace(-radius, radius, 13)
        generators, energies, changes = [], [], []
        for lam in nodes:
            gen, transformed, _ = solve(h0 + lam * h1)
            generators.append(gen)
            energies.append(transformed[0])
            if modelspace:
                changes.append(transformed - h0 - lam * h1)
        energy = np.polynomial.polynomial.polyfit(nodes / radius, energies, 8)
        amplitudes = np.polynomial.polynomial.polyfit(nodes / radius, generators, 8)
        result = (
            np.array([energy[2] / radius**2, energy[3] / radius**3]),
            amplitudes[1] / radius,
            amplitudes[2] / radius**2,
        )
        if modelspace:
            coefficients = np.polynomial.polynomial.polyfit(nodes / radius, changes, 8)
            second = h0 + h1 + coefficients[1] / radius + coefficients[2] / radius**2
            third = second + coefficients[3] / radius**3
            second[0] += h[0]
            third[0] += h[0]
            return result + (second, third)
        return result

    coarse = fit(0.04)
    fine = fit(0.025)
    pt, a1, a2 = coarse[:3]
    refined, b1, b2 = fine[:3]
    refinement = max(
        np.max(abs(pt - refined)), np.max(abs(a1 - b1)), np.max(abs(a2 - b2))
    )
    assert refinement < 1e-8, refinement
    result = dict(
        **base_result,
        pt2=float(refined[0]),
        pt3=float(refined[1]),
        a1=oracle.base.full(b1),
        a2=oracle.base.full(b2),
        polynomial_refinement=float(refinement),
    )
    if modelspace:
        dimension = 1 << oracle.n
        columns = []
        for bits in range(1 << len(active)):
            state = np.zeros(dimension)
            state[0] = 1.0
            indices = core_order + [
                p for i, p in enumerate(active_order) if (bits >> i) & 1
            ]
            for p in reversed(indices):
                state = oracle.create[p] @ state
            columns.append(state)
        embedding = np.column_stack(columns)
        from gen_imsrg2 import FockOracle

        def projected(coefficients):
            matrix = embedding.T @ oracle.matrix(coefficients) @ embedding
            if not active:
                return dict(zero=float(matrix[0, 0]), one=[], two=[])
            vacuum = FockOracle([False] * len(active))
            bare, _ = vacuum.project(matrix)
            return vacuum.full(bare)

        result.update(
            pt2_active=projected(fine[3]),
            pt3_active=projected(fine[4]),
            ldsrg_active=projected(bar),
        )
    return result


def main():
    root_path = Path(__file__).resolve().parents[1]
    gno = json.loads(
        (root_path / "data/gno_reference.json").read_text(encoding="utf-8")
    )
    rng = np.random.default_rng(20261023)
    cases = []
    for index in (0, 2, 3, 4, 5):
        model = gno["cases"][index]
        if index == 2:
            model = dict(
                model,
                name="slater_reference",
                n_spatial=0,
                active=[],
                core=[0, 1],
                members=[dict(na=0, nb=0, coefficients=[1.0], weight=1.0)],
            )
        oracle = GeneralizedOracle(6, model["members"], model["active"], model["core"])
        h = random_operator(rng, 6, 0.02)
        f = h[1:37].reshape(6, 6)
        classes = [
            0 if p in model["core"] else 1 if p in model["active"] else 2
            for p in range(6)
        ]
        for p in range(6):
            for q in range(6):
                if classes[p] == classes[q]:
                    f[p, q] = -1.3 + 0.5 * p if p == q else 0.0
        record = {
            key: model[key]
            for key in ("name", "n_spin", "n_spatial", "members", "active", "core")
        }
        record["h"] = oracle.base.full(h)
        record.update(driven(oracle, h, model["active"], model["core"], 0.5))
        cases.append(record)
        print(
            record["name"],
            "PT2",
            record["pt2"],
            "PT3",
            record["pt3"],
            "E",
            record["transformed"]["zero"],
            "refinement",
            record["polynomial_refinement"],
            flush=True,
        )
    (root_path / "data/mr_driven_reference.json").write_text(
        json.dumps(dict(source=__doc__, cases=cases), indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
