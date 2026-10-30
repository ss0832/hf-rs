# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""MR reference relaxation from full Fock matrices and finite-lambda fits.

The reference state lives in one fixed Fock basis. Each semicanonical change
uses an explicitly built exterior-power orbital matrix. Effective CAS matrices
come from a Pade exponential of independently projected commutator maps; PT2/3
are polynomial coefficients of the nonlinear solution. No external MR source
or internal routine is used.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from gen_gno import GeneralizedOracle, creation
from gen_imsrg2 import FockOracle
from gen_mr_driven import driven
from gen_mr_molecular import models, molecule
from gen_mr_relaxation import determinant_basis


def orbital_matrix(create, u):
    n = len(create)
    rotated = [
        sum(u[q // 2, p // 2] * create[q] for q in range(p % 2, n, 2)) for p in range(n)
    ]
    columns = []
    for bits in range(1 << n):
        state = np.zeros(1 << n)
        state[0] = 1.0
        for p in reversed([p for p in range(n) if (bits >> p) & 1]):
            state = rotated[p] @ state
        columns.append(state)
    return np.column_stack(columns)


def embedding(create, core, na, nalpha, nbeta):
    from gen_density_cumulants import strings

    columns = []
    for a, b in itertools.product(strings(na, nalpha), strings(na, nbeta)):
        indices = list(range(2 * core))
        indices += [2 * (core + p) for p in range(na) if (a >> p) & 1]
        indices += [2 * (core + p) + 1 for p in range(na) if (b >> p) & 1]
        state = np.zeros(1 << len(create))
        state[0] = 1.0
        for p in reversed(indices):
            state = create[p] @ state
        columns.append(state)
    return np.column_stack(columns)


def singlet_basis(na, nalpha, nbeta):
    create = creation(2 * na)
    b = determinant_basis(create, na, nalpha, nbeta)
    plus = sum(create[2 * p] @ create[2 * p + 1].T for p in range(na))
    sz = sum(
        0.5
        * (create[2 * p] @ create[2 * p].T - create[2 * p + 1] @ create[2 * p + 1].T)
        for p in range(na)
    )
    spin = b.T @ (0.5 * (plus @ plus.T + plus.T @ plus) + sz @ sz) @ b
    values, vectors = np.linalg.eigh(spin)
    return vectors[:, abs(values) < 1e-10]


def relax(model, order, mode, coordinates=None, problem=None, driver=driven):
    info, original, h = (
        molecule(model, coordinates, return_problem=True)
        if problem is None
        else problem
    )
    physical = original.matrix(h)
    n, nc, na = original.n, info["n_core"], info["n_active"]
    core, active = list(range(2 * nc)), list(range(2 * nc, 2 * (nc + na)))
    create = original.create
    b = embedding(create, nc, na, info["n_alpha"], info["n_beta"])
    psi = b @ np.array(info["coefficients"])
    anchor_c = np.array(info["mo_coeff"])
    vacuum = FockOracle([False] * n)
    bare, _ = vacuum.project(physical)
    bare = vacuum.full(bare)
    one = np.array(bare["one"])
    two = np.array(bare["two"]).reshape((n,) * 4)
    spin_basis = singlet_basis(na, info["n_alpha"], info["n_beta"])
    history = []
    initial_effective = None
    previous = None
    limit = dict(once=1, twice=2, iterate=30)[mode]
    for iteration in range(limit):
        gamma = np.array(
            [[psi @ create[p] @ create[q].T @ psi for q in range(n)] for p in range(n)]
        )
        f = one + np.einsum("prqs,rs->pq", two, gamma)
        assert np.max(abs(f[::2, ::2] - f[1::2, 1::2])) < 1e-11
        u = np.eye(n // 2)
        for start, end in ((0, nc), (nc, nc + na), (nc + na, n // 2)):
            if start < end:
                _, u[start:end, start:end] = np.linalg.eigh(
                    f[2 * start : 2 * end : 2, 2 * start : 2 * end : 2]
                )
        rotation = orbital_matrix(create, u)
        ci = b.T @ rotation.T @ psi
        members = [
            dict(
                na=info["n_alpha"],
                nb=info["n_beta"],
                coefficients=ci.tolist(),
                weight=1.0,
            )
        ]
        oracle = GeneralizedOracle(n, members, active, core)
        current_h, discarded = oracle.project(rotation.T @ physical @ rotation)
        assert discarded < 1e-12
        solution = driver(
            oracle,
            current_h,
            active,
            core,
            0.5,
            modelspace=order < 4,
            compute_perturbation=order < 4,
            exponential_action=True,
            conserve_spin=True,
            bch_order=2 if order == 5 else None,
        )
        if order >= 4:
            full_bar = oracle.matrix(
                np.concatenate(
                    (
                        [solution["transformed"]["zero"]],
                        np.array(solution["transformed"]["one"]).ravel(),
                        [
                            np.array(solution["transformed"]["two"]).reshape((n,) * 4)[
                                p, q, r, s
                            ]
                            for p, q in oracle.groups[2]
                            for r, s in oracle.groups[2]
                        ],
                    )
                )
            )
            matrix = b.T @ full_bar @ b
        else:
            effective = solution[f"pt{order}_active"]
            if initial_effective is None:
                initial_effective = effective
            active_vacuum = FockOracle([False] * (2 * na))
            groups = list(itertools.combinations(range(2 * na), 2))
            pair = np.array(effective["two"]).reshape((2 * na,) * 4)
            packed = np.concatenate(
                (
                    [effective["zero"]],
                    np.array(effective["one"]).ravel(),
                    [pair[p, q, r, s] for p, q in groups for r, s in groups],
                )
            )
            ba = determinant_basis(
                creation(2 * na),
                na,
                info["n_alpha"],
                info["n_beta"],
            )
            matrix = ba.T @ active_vacuum.matrix(packed) @ ba
        energies, vectors = np.linalg.eigh(spin_basis.T @ matrix @ spin_basis)
        updated = spin_basis @ vectors[:, 0]
        if np.dot(ci, updated) < 0:
            updated *= -1
        ci_change = float(np.max(abs(updated - ci)))
        energy = float(energies[0])
        energy_change = abs(energy - previous) if previous is not None else 0.0
        history.append(
            dict(
                energy=energy,
                ci_change=ci_change,
                energy_change=energy_change,
                mo_coeff=(anchor_c @ u).tolist(),
                coefficients=updated.tolist(),
            )
        )
        if order == 5:
            history[-1].update(
                source_energy=float(solution["transformed"]["zero"]),
                source_coefficients=ci.tolist(),
            )
        psi = rotation @ b @ updated
        if (
            mode == "iterate"
            and ci_change < 1e-10
            and (previous is None or energy_change < 1e-12)
        ):
            break
        previous = energy
    else:
        assert mode != "iterate", history
    return dict(
        info,
        order=order,
        mode=mode,
        energy=energy,
        mo_coeff=(anchor_c @ u).tolist(),
        coefficients=updated.tolist(),
        history=history,
        initial_effective=initial_effective,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--order", type=int, nargs="*", default=[2, 3, 4])
    parser.add_argument("--mode", nargs="*", default=["once", "twice", "iterate"])
    parser.add_argument("--output", default="mr_relaxed_molecular_reference.json")
    args = parser.parse_args()
    records = []
    for index in args.model:
        model = models()[index]
        results = []
        for order in args.order:
            converged = relax(model, order, "iterate")
            for mode in args.mode:
                size = {"once": 1, "twice": 2, "iterate": len(converged["history"])}[
                    mode
                ]
                step = converged["history"][min(size, len(converged["history"])) - 1]
                result = dict(
                    converged, **step, mode=mode, history=converged["history"][:size]
                )
                results.append(result)
                print(
                    model["name"],
                    order,
                    mode,
                    result["energy"],
                    len(result["history"]),
                    flush=True,
                )
        records.append(dict(model_index=index, name=model["name"], results=results))
    Path(__file__).resolve().parents[1].joinpath("data", args.output).write_text(
        json.dumps(dict(source=__doc__, cases=records), indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
