# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent SA/MS/XMS/DWMS: full-Fock GNO matrices and CI projections.

Public theory: JCP 148, 124106 (2018), DOI 10.1063/1.5019793;
JCP 150, 144107 (2019), Eqs. 15, 16 and 19, DOI 10.1063/1.5088120.
No external MR implementation or generated MR contraction code is used.
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
from gen_mr_relaxed_molecular import embedding, orbital_matrix, singlet_basis
from gen_three_df import direct_pt2
from pyscf import ao2mo, df, fci, mcscf
from scipy.linalg import expm
from scipy.optimize import root as solve_root

from embedded_basis import embedded_basis

VARIANTS = ["sa", "sac", "ms", "xms", "dw0", "dw50", "dwlarge"]


def build_problem(index, fitted=False, roots=2, coordinates=None):
    model = models()[index]
    if coordinates is not None:
        model = dict(
            model,
            atoms=[
                dict(a, position=r.tolist())
                for a, r in zip(model["atoms"], coordinates)
            ],
        )
    mf, _, _ = molecule(dict(model, optimize=False), reference_only=True)
    c = mf.mo_coeff[:, : model.get("n_mo", mf.mo_coeff.shape[1])]
    initial = c.copy()
    na = model["n_active"]
    pop = (model["active_electrons"] // 2,) * 2
    nc = mf.mol.nelectron // 2 - pop[0]
    if model.get("optimize", False):
        optimizer = mcscf.CASSCF(mf, na, pop)
        optimizer.fcisolver = fci.direct_spin0.FCI(mf.mol)
        optimizer = optimizer.state_average_([1.0 / roots] * roots)
        optimizer.conv_tol = 1e-13
        optimizer.conv_tol_grad = 1e-10
        optimizer.max_cycle_macro = 200
        optimizer.fcisolver.conv_tol = 1e-14
        optimizer.kernel(c)
        assert optimizer.converged and np.max(abs(optimizer.get_grad())) < 1e-7
        # Individual excited-state energies are not stationary under the
        # ensemble objective. Refine the orbital residual independently of
        # energy-only stopping before constructing nuclear energy stencils.
        anchor = optimizer.mo_coeff.copy()
        groups = [0] * nc + [1] * na + [2] * (anchor.shape[1] - nc - na)
        pairs = [
            (p, q)
            for p in range(len(groups))
            for q in range(p)
            if groups[p] != groups[q]
        ]
        ci_oracle = mcscf.CASCI(mf, na, pop)
        spin = singlet_basis(na, *pop)

        def orbital_residual(parameters):
            kappa = np.zeros((len(groups), len(groups)))
            for value, (p, q) in zip(parameters, pairs):
                kappa[p, q], kappa[q, p] = value, -value
            trial = anchor @ expm(kappa)
            h1, _ = ci_oracle.get_h1eff(trial)
            h2 = ci_oracle.get_h2eff(trial)
            addresses, matrix = fci.direct_spin1.pspace(h1, h2, na, pop, np=na**2)
            reordered = np.zeros_like(matrix)
            reordered[np.ix_(addresses, addresses)] = matrix
            _, vectors = np.linalg.eigh(spin.T @ reordered @ spin)
            optimizer.ci = [v.reshape(na, na) for v in (spin @ vectors[:, :roots]).T]
            optimizer.mo_coeff = trial
            return optimizer.get_grad(trial)

        solution = solve_root(orbital_residual, np.zeros(len(pairs)), tol=1e-11)
        assert np.max(abs(orbital_residual(solution.x))) < 1e-11
        c = optimizer.mo_coeff
    if fitted:
        auxiliary = embedded_basis(
            "AUX_DEF2_SVP_RIFIT_GBS", list({a["symbol"] for a in model["atoms"]})
        )
        auxmol = df.addons.make_auxmol(mf.mol, auxiliary)
        values, vectors = np.linalg.eigh(auxmol.intor("int2c2e"))
        retained = values >= 1e-10
        three = df.incore.aux_e2(mf.mol, auxmol, intor="int3c2e", aosym="s1")
        b = np.einsum(
            "pqQ,QK->Kpq", three, vectors[:, retained] / np.sqrt(values[retained])
        )
        b = np.einsum("Kpq,pa,qb->Kab", b, c, c)
        eri = np.einsum("Kpq,Krs->pqrs", b, b)
    else:
        eri = ao2mo.kernel(mf.mol, c, compact=False).reshape((c.shape[1],) * 4)
        pairs = [(p, q) for p in range(c.shape[1]) for q in range(p + 1)]
        metric = np.array([[eri[p, q, r, s] for r, s in pairs] for p, q in pairs])
        values, vectors = np.linalg.eigh(metric)
        assert values[0] > -1e-12
        packed = np.sqrt(np.maximum(values, 0))[:, None] * vectors.T
        b = np.zeros((len(values), c.shape[1], c.shape[1]))
        for k, (p, q) in enumerate(pairs):
            b[:, p, q] = b[:, q, p] = packed[:, k]
        eri = np.einsum("Kpq,Krs->pqrs", b, b)
    h = c.T @ mf.get_hcore() @ c
    active = slice(nc, nc + na)
    ha = h[active, active].copy()
    core = mf.mol.energy_nuc() + 2 * np.trace(h[:nc, :nc])
    for i, j in itertools.product(range(nc), repeat=2):
        core += 2 * eri[i, i, j, j] - eri[i, j, j, i]
    for i in range(nc):
        ha += 2 * eri[active, active, i, i] - eri[active, i, i, active]
    addresses, matrix = fci.direct_spin1.pspace(
        ha, eri[active, active, active, active], na, pop, np=na**2
    )
    reordered = np.zeros_like(matrix)
    reordered[np.ix_(addresses, addresses)] = matrix
    spin = singlet_basis(na, *pop)
    energies, vectors = np.linalg.eigh(spin.T @ reordered @ spin)
    ci = (spin @ vectors[:, :roots]).T
    ns = 2 * c.shape[1]
    vacuum = FockOracle([False] * ns)
    one = np.zeros((ns, ns))
    for p, q in itertools.product(range(ns), repeat=2):
        if p % 2 == q % 2:
            one[p, q] = h[p // 2, q // 2]
    pair = np.zeros((vacuum.np, vacuum.np))
    for i, (p, q) in enumerate(vacuum.pairs):
        for j, (r, s) in enumerate(vacuum.pairs):
            if p % 2 == r % 2 and q % 2 == s % 2:
                pair[i, j] += eri[p // 2, r // 2, q // 2, s // 2]
            if p % 2 == s % 2 and q % 2 == r % 2:
                pair[i, j] -= eri[p // 2, s // 2, q // 2, r // 2]
    physical = vacuum.matrix(
        np.concatenate(([mf.mol.energy_nuc()], one.ravel(), pair.ravel()))
    )
    create = creation(ns)
    embed = embedding(create, nc, na, *pop)
    assert (
        np.max(
            abs(
                np.diag(ci @ embed.T @ physical @ embed @ ci.T)
                - (core + energies[:roots])
            )
        )
        < 1e-10
    )
    packed = b[:, np.tril_indices(c.shape[1])[0], np.tril_indices(c.shape[1])[1]]
    info = dict(
        model_index=index,
        fitted=fitted,
        roots=roots,
        n_core=nc,
        n_active=na,
        n_alpha=pop[0],
        n_beta=pop[1],
        e_scf=float(mf.e_tot),
        nuclear=float(mf.mol.energy_nuc()),
        cas_energies=(core + energies[:roots]).tolist(),
        mo_coeff=c.tolist(),
        initial_mo_coeff=initial.tolist(),
        h=h.tolist(),
        factors=packed.tolist(),
        ci=ci.tolist(),
    )
    return info, physical, create, embed


def ensemble(info, ci, weights):
    nc, na = info["n_core"], info["n_active"]
    members = [
        dict(
            na=info["n_alpha"],
            nb=info["n_beta"],
            coefficients=c.tolist(),
            weight=float(w),
        )
        for c, w in zip(ci, weights)
    ]
    return GeneralizedOracle(
        2 * len(info["h"]),
        members,
        list(range(2 * nc, 2 * (nc + na))),
        list(range(2 * nc)),
    )


def evaluate(problem, variant, flow=0.5, details=True, verify_fit=False):
    info, physical, create, embed = problem
    ci = np.array(info["ci"])
    count = len(ci)
    nc, na = info["n_core"], info["n_active"]
    active, core = list(range(2 * nc, 2 * (nc + na))), list(range(2 * nc))
    model_rotation = np.eye(count)
    if variant == "xms":
        oracle = ensemble(info, ci, np.ones(count) / count)
        normal, _ = oracle.project(physical)
        f = normal[1 : 1 + oracle.n**2]
        fock = oracle.base.matrix(
            np.concatenate(([0.0], f, np.zeros(oracle.base.np**2)))
        )
        _, model_rotation = np.linalg.eigh(ci @ embed.T @ fock @ embed @ ci.T)
        ci = model_rotation.T @ ci
    diagonal = np.diag(ci @ embed.T @ physical @ embed @ ci.T)
    if variant in ("sa", "sac", "dw0"):
        weights = np.ones((count, count)) / count
    elif variant.startswith("dw"):
        zeta = 50.0 if variant == "dw50" else 1e14
        weights = np.exp(-zeta * (diagonal[:, None] - diagonal[None, :]) ** 2)
        weights /= weights.sum(axis=1)[:, None]
    else:
        weights = np.eye(count)
    common = variant in ("sa", "sac", "dw0")
    result = np.zeros((count, count))
    sources = []
    for target in range(1 if common else count):
        oracle = ensemble(info, ci, weights[target])
        normal, _ = oracle.project(physical)
        f = normal[1 : 1 + oracle.n**2].reshape(oracle.n, oracle.n)[::2, ::2]
        u = np.eye(len(info["h"]))
        for start, end in ((0, nc), (nc, nc + na), (nc + na, len(u))):
            if start < end:
                _, u[start:end, start:end] = np.linalg.eigh(f[start:end, start:end])
        rotation = orbital_matrix(create, u)
        rotated_ci = (embed.T @ rotation.T @ embed @ ci.T).T
        oracle = ensemble(info, rotated_ci, weights[target])
        normal, _ = oracle.project(rotation.T @ physical @ rotation)
        pt = direct_pt2(oracle, normal, active, core, flow)
        if verify_fit:
            fitted = driven(
                oracle,
                normal,
                active,
                core,
                flow,
                modelspace=True,
                exponential_action=True,
                conserve_spin=True,
            )["pt2_active"]
            differences = [abs(pt["pt2_active"]["zero"] - fitted["zero"])]
            for key in ("one", "two"):
                differences.extend(
                    abs(
                        np.array(pt["pt2_active"][key]).ravel()
                        - np.array(fitted[key]).ravel()
                    )
                )
            assert max(differences) < 1e-8
            print(
                "ensemble nonlinear-fit source",
                variant,
                target,
                "all",
                len(differences),
                "coefficients",
                max(differences),
                flush=True,
            )
        vacuum = FockOracle([False] * (2 * na))
        op = pt["pt2_active"]
        two = np.array(op["two"]).reshape((2 * na,) * 4)
        packed = np.concatenate(
            (
                [op["zero"]],
                np.array(op["one"]).ravel(),
                [two[p, q, r, s] for p, q in vacuum.pairs for r, s in vacuum.pairs],
            )
        )
        effective = vacuum.matrix(packed)
        basis = determinant_basis(creation(2 * na), na, info["n_alpha"], info["n_beta"])
        matrix = basis.T @ effective @ basis
        projected = rotated_ci @ matrix @ rotated_ci.T
        if common:
            result = projected
        else:
            result[:, target] = projected[:, target]
        act_rotation = orbital_matrix(creation(2 * na), u[nc : nc + na, nc : nc + na])
        anchor, _ = vacuum.project(act_rotation @ effective @ act_rotation.T)
        if details:
            sources.append(vacuum.full(anchor))
        if variant == "sa":
            spin = singlet_basis(na, info["n_alpha"], info["n_beta"])
            values, vectors = np.linalg.eigh(spin.T @ matrix @ spin)
            coefficients = (embed.T @ rotation @ embed @ (spin @ vectors[:, :count])).T
    result = 0.5 * (result + result.T)
    if variant != "sa":
        values, mixing = np.linalg.eigh(result)
        coefficients = mixing.T @ ci
    return dict(
        variant=variant,
        flow=flow,
        energies=values[:count].tolist(),
        ci=coefficients.tolist(),
        model_hamiltonian=result.tolist(),
        model_rotation=model_rotation.tolist(),
        weights=weights.tolist(),
        sources=sources,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradients", action="store_true")
    parser.add_argument("--verify-fits", action="store_true")
    parser.add_argument("--append-casscf-gradients", action="store_true")
    args = parser.parse_args()
    if args.verify_fits:
        current = build_problem(0)
        for variant in ("sa", "xms", "dw50"):
            evaluate(current, variant, verify_fit=True)
        return
    if args.gradients or args.append_casscf_gradients:
        gradients(args.append_casscf_gradients)
        return
    cases = []
    for index, fitted, roots in [
        (0, False, 2),
        (1, False, 2),
        (2, False, 2),
        (3, False, 2),
        (0, True, 2),
        (2, True, 2),
        (0, False, 3),
    ]:
        problem = build_problem(index, fitted, roots)
        records = [
            evaluate(problem, variant, flow)
            for flow in (0.0, 0.5)
            for variant in VARIANTS
        ]
        cases.append(dict(**problem[0], results=records))
        print(
            index,
            fitted,
            roots,
            [(r["variant"], r["energies"]) for r in records if r["flow"]],
            flush=True,
        )
        Path(__file__).resolve().parents[1].joinpath(
            "data/multistate_reference.json"
        ).write_text(
            json.dumps(dict(source=__doc__, cases=cases), indent=1) + "\n",
            encoding="utf-8",
        )


def gradients(append_casscf=False):
    output = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("data/multistate_gradient_reference.json")
    )
    records = (
        [
            r
            for r in json.loads(output.read_text(encoding="utf-8"))["cases"]
            if r["model_index"] != 3
        ]
        if append_casscf
        else []
    )
    step = 1e-3
    selections = (
        [(3, False)]
        if append_casscf
        else [(0, False), (2, False), (0, True), (2, True), (3, False)]
    )
    for index, fitted in selections:
        model = models()[index]
        coordinates = np.array([a["position"] for a in model["atoms"]])
        gradient = []
        for component in range(coordinates.size):
            energies = []
            for offset in (-2, -1, 1, 2):
                shifted = coordinates.copy()
                shifted.flat[component] += offset * step
                problem = build_problem(index, fitted, 2, shifted)
                energies.append(
                    [
                        evaluate(problem, variant, details=False)["energies"]
                        for variant in VARIANTS[:4] + ["dw50"]
                    ]
                )
            row = np.einsum("i,ijk->jk", [1, -8, 8, -1], energies) / (12 * step)
            gradient.append(row.tolist())
            print(index, fitted, component, row.ravel(), flush=True)
        records.append(
            dict(
                model_index=index,
                fitted=fitted,
                step=step,
                gradient=np.transpose(gradient, (1, 2, 0)).tolist(),
            )
        )
        Path(__file__).resolve().parents[1].joinpath(
            "data/multistate_gradient_reference.json"
        ).write_text(
            json.dumps(
                dict(
                    source=__doc__,
                    variants=["sa", "sac", "ms", "xms", "dw50"],
                    cases=records,
                ),
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
