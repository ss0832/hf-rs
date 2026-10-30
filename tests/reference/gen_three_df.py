# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""THREE Coulomb-metric DF from PySCF integrals and full-Fock MRPT2 fits.

Only public theory and this repository's independent NumPy/Fock oracles are
used. No external MR implementation or generated MR expression is imported.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from gen_gno import GeneralizedOracle
from gen_imsrg2 import FockOracle
from gen_mr_driven import driven
from gen_mr_molecular import models, molecule
from gen_mr_relaxed_molecular import relax
from pyscf import df, fci

from embedded_basis import embedded_basis


def problem(model):
    mf, _, _ = molecule(model, reference_only=True)
    mol = mf.mol
    symbols = list({a["symbol"] for a in model["atoms"]})
    auxiliary = embedded_basis("AUX_DEF2_SVP_RIFIT_GBS", symbols)
    auxmol = df.addons.make_auxmol(mol, auxiliary)
    metric = auxmol.intor("int2c2e")
    values, vectors = np.linalg.eigh(metric)
    retained = values >= 1e-10
    three = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    b = np.einsum(
        "pqQ,QK->Kpq", three, vectors[:, retained] / np.sqrt(values[retained])
    )
    eri = np.einsum("Kpq,Krs->pqrs", b, b)
    c = mf.mo_coeff[:, : model.get("n_mo", mf.mo_coeff.shape[1])]
    na = model["n_active"]
    nelec = (model["active_electrons"] // 2,) * 2
    nc = mol.nelectron // 2 - nelec[0]

    def integrals(c):
        h = c.T @ mf.get_hcore() @ c
        g = np.einsum("up,vq,wr,xs,uvwx->pqrs", c, c, c, c, eri, optimize=True)
        a = slice(nc, nc + na)
        core = mol.energy_nuc() + 2 * np.trace(h[:nc, :nc])
        for i, j in itertools.product(range(nc), repeat=2):
            core += 2 * g[i, i, j, j] - g[i, j, j, i]
        ha = h[a, a].copy()
        for i in range(nc):
            ha += 2 * g[a, a, i, i] - g[a, i, i, a]
        addresses, matrix = fci.direct_spin1.pspace(
            ha, g[a, a, a, a], na, nelec, np=na**2
        )
        energies, vectors = np.linalg.eigh(matrix)
        ci = np.zeros(na**2)
        ci[addresses] = vectors[:, 0]
        return h, g, float(core + energies[0]), ci.reshape(na, na)

    h, g, _, ci = integrals(c)
    density = np.zeros_like(h)
    density[:nc, :nc] = 2 * np.eye(nc)
    density[nc : nc + na, nc : nc + na] = fci.direct_spin1.make_rdm1(ci, na, nelec)
    fock = (
        h
        + np.einsum("pqrs,rs->pq", g, density)
        - 0.5 * np.einsum("prsq,rs->pq", g, density)
    )
    u = np.eye(c.shape[1])
    for start, end in ((0, nc), (nc, nc + na), (nc + na, c.shape[1])):
        if start < end:
            _, u[start:end, start:end] = np.linalg.eigh(fock[start:end, start:end])
    c = c @ u
    h, g, energy, ci = integrals(c)
    active, core = list(range(2 * nc, 2 * (nc + na))), list(range(2 * nc))
    member = dict(
        na=nelec[0], nb=nelec[1], coefficients=ci.ravel().tolist(), weight=1.0
    )
    oracle = GeneralizedOracle(2 * c.shape[1], [member], active, core)
    n = oracle.n
    one = np.zeros((n, n))
    pair = np.zeros((len(oracle.groups[2]),) * 2)
    for p, q in itertools.product(range(n), repeat=2):
        if p % 2 == q % 2:
            one[p, q] = h[p // 2, q // 2]
    for k, (p, q) in enumerate(oracle.groups[2]):
        for column, (r, s) in enumerate(oracle.groups[2]):
            if p % 2 == r % 2 and q % 2 == s % 2:
                pair[k, column] += g[p // 2, r // 2, q // 2, s // 2]
            if p % 2 == s % 2 and q % 2 == r % 2:
                pair[k, column] -= g[p // 2, s // 2, q // 2, r // 2]
    physical = oracle.base.matrix(
        np.concatenate(([mol.energy_nuc()], one.ravel(), pair.ravel()))
    )
    normal, discarded = oracle.project(physical)
    assert discarded < 1e-12 and abs(normal[0] - energy) < 1e-11
    info = dict(
        e_scf=float(mf.e_tot),
        e_cas=energy,
        n_core=nc,
        n_active=na,
        n_alpha=nelec[0],
        n_beta=nelec[1],
        mo_coeff=c.tolist(),
        coefficients=ci.ravel().tolist(),
    )
    return (info, oracle, normal), dict(
        rank=int(np.count_nonzero(retained)), eri=eri.ravel().tolist()
    )


def direct_pt2(oracle, h, active, core, flow, **unused):
    """First-order generator from explicit full-Fock commutator matrices.

    The initial energies are cross-checked against the independent nonlinear
    finite-lambda fits before using this faster oracle for nuclear displacements.
    """
    n = oracle.n
    holes, particles = (
        sorted(set(active) | set(core)),
        sorted(set(range(n)) - set(core)),
    )
    pairs = {p: k for k, p in enumerate(oracle.groups[2])}
    singles = [(a, i) for a in particles for i in holes if {a, i} - set(active)]
    doubles = [
        (a, b, i, j)
        for a, b in itertools.combinations(particles, 2)
        for i, j in itertools.combinations(holes, 2)
        if {a, b, i, j} - set(active)
    ]
    eps = h[1 : 1 + n * n].reshape(n, n).diagonal()
    h0 = np.zeros_like(h)
    h0[1 : 1 + n * n] = np.diag(eps).ravel()
    amplitude = np.zeros_like(h)

    def regularized(gap):
        return -np.expm1(-flow * gap * gap) / gap if gap else 0.0

    for a, b, i, j in doubles:
        forward = 1 + n * n + pairs[a, b] * len(pairs) + pairs[i, j]
        backward = 1 + n * n + pairs[i, j] * len(pairs) + pairs[a, b]
        value = h[forward] * regularized(eps[i] + eps[j] - eps[a] - eps[b])
        amplitude[forward], amplitude[backward] = value, -value
    coupling, _ = oracle.commutator(h0, amplitude)
    for a, i in singles:
        forward, backward = 1 + a * n + i, 1 + i * n + a
        value = (h[forward] + coupling[forward]) * regularized(eps[i] - eps[a])
        amplitude[forward], amplitude[backward] = value, -value
    c01, _ = oracle.commutator(h0, amplitude)
    correction, _ = oracle.commutator(h - h0 + 0.5 * c01, amplitude)
    bar = h + c01 + correction
    columns = []
    for bits in range(1 << len(active)):
        state = np.zeros(1 << n)
        state[0] = 1.0
        for p in reversed(
            list(core) + [p for k, p in enumerate(active) if (bits >> k) & 1]
        ):
            state = oracle.create[p] @ state
        columns.append(state)
    embedding = np.column_stack(columns)
    vacuum = FockOracle([False] * len(active))
    bare, _ = vacuum.project(embedding.T @ oracle.matrix(bar) @ embedding)
    return dict(pt2=float(correction[0]), pt2_active=vacuum.full(bare))


def values(model):
    current, _ = problem(model)
    info, oracle, h = current
    nc, na = info["n_core"], info["n_active"]
    pt = direct_pt2(
        oracle, h, list(range(2 * nc, 2 * (nc + na))), list(range(2 * nc)), 0.5
    )
    history = relax(model, 2, "iterate", problem=current, driver=direct_pt2)["history"]
    return [
        info["e_cas"] + pt["pt2"],
        history[0]["energy"],
        history[min(1, len(history) - 1)]["energy"],
        history[-1]["energy"],
    ]


def gradients():
    energy = json.loads(
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("data/three_df_reference.json")
        .read_text(encoding="utf-8")
    )
    records = []
    step = 1e-3
    for record in energy["cases"]:
        index = record["model_index"]
        model = models()[index]
        actual = values(model)
        expected = [record["e_cas"] + record["pt2"]] + [
            r["energy"] for r in record["results"]
        ]
        assert np.max(abs(np.array(actual) - expected)) < 1e-10
        coordinates = np.array([a["position"] for a in model["atoms"]])
        gradient = []
        for component in range(coordinates.size):
            displaced = []
            for offset in (-2, -1, 1, 2):
                shifted = coordinates.copy()
                shifted.flat[component] += offset * step
                moved = dict(
                    model,
                    atoms=[
                        dict(a, position=r.tolist())
                        for a, r in zip(model["atoms"], shifted)
                    ],
                )
                displaced.append(values(moved))
            row = np.array([1, -8, 8, -1]) @ displaced / (12 * step)
            gradient.append(row.tolist())
            print(model["name"], component, row, flush=True)
        records.append(
            dict(model_index=index, step=step, gradient=np.array(gradient).T.tolist())
        )
        Path("target/three-df-gradient-partial.json").write_text(
            json.dumps(records, indent=1) + "\n", encoding="utf-8"
        )
    Path(__file__).resolve().parents[1].joinpath(
        "data/three_df_gradient_reference.json"
    ).write_text(
        json.dumps(
            dict(
                source=__doc__,
                modes=["none", "once", "twice", "iterate"],
                cases=records,
            ),
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradients", action="store_true")
    args = parser.parse_args()
    if args.gradients:
        gradients()
        return
    records = []
    for index in (0, 2):
        model = models()[index]
        current, fitted = problem(model)
        info, oracle, h = current
        nc, na = info["n_core"], info["n_active"]
        unrelaxed = driven(
            oracle, h, list(range(2 * nc, 2 * (nc + na))), list(range(2 * nc)), 0.5
        )
        relaxed = relax(model, 2, "iterate", problem=current)
        history = relaxed["history"]
        result = dict(
            model_index=index,
            **fitted,
            e_cas=info["e_cas"],
            pt2=unrelaxed["pt2"],
            results=[history[0], history[min(1, len(history) - 1)], history[-1]],
        )
        records.append(result)
        print(
            model["name"],
            result["rank"],
            info["e_cas"],
            unrelaxed["pt2"],
            [r["energy"] for r in result["results"]],
            flush=True,
        )
    Path(__file__).resolve().parents[1].joinpath(
        "data/three_df_reference.json"
    ).write_text(
        json.dumps(dict(source=__doc__, cases=records), indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
