"""Finite-threshold SCI references using full PySCF matrices and dense numpy solves.

ACI: arxiv:1603.08063, Eqs 5/6/9/10; ASCI: arxiv:1603.02686, Eq 4;
HCI: arxiv:1606.07453, Eqs 2/4. No Rust selection or sigma code is used.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import fci, lib

lib.num_threads(4)
n = 4
h = np.array(
    [
        [-1.4 + 0.4 * p if p == q else 0.05 * np.sin(p + q + 1) for q in range(n)]
        for p in range(n)
    ]
)


def pair(p, q):
    return max(p, q) * (max(p, q) + 1) // 2 + min(p, q)


g = np.empty((n, n, n, n))
for p in range(n):
    for q in range(n):
        for r in range(n):
            for s in range(n):
                g[p, q, r, s] = sum(
                    0.1
                    * np.sin((k + 1) * (pair(p, q) + 1) * 0.37)
                    * np.sin((k + 1) * (pair(r, s) + 1) * 0.37)
                    for k in range(3)
                )
strings = fci.cistring.make_strings(range(n), 2)
dets = [(int(a), int(b)) for a in strings for b in strings]
absorbed = fci.direct_spin1.absorb_h1e(h, g, n, (2, 2), fac=0.5)
eye = np.eye(len(dets))
ham = np.column_stack(
    [
        fci.direct_spin1.contract_2e(absorbed, v, n, (2, 2)).ravel() + 0.321 * v
        for v in eye
    ]
)
reference = dets.index((3, 3))


def neighbors(indices):
    return [
        i
        for i, d in enumerate(dets)
        if any(
            ((d[0] ^ dets[j][0]).bit_count() + (d[1] ^ dets[j][1]).bit_count()) // 2
            <= 2
            for j in indices
        )
    ]


def diagonalize(model):
    e, c = np.linalg.eigh(ham[np.ix_(model, model)])
    return float(e[0]), c[:, 0]


def importance(delta, v):
    return 0.5 * delta - np.hypot(0.5 * delta, v)


def run(kind, threshold, eps2=0.0, core=5, target=13, gamma=0.5):
    model = [reference]
    old_model = []
    old_energy = None
    for iteration in range(100):
        energy, coeff = diagonalize(model)
        correction = 0.0
        if kind == "ASCI":
            source = sorted(
                range(len(model)), key=lambda i: (-abs(coeff[i]), dets[model[i]])
            )[:core]
            core_model = [model[i] for i in source]
            ranked = [(model[i], abs(coeff[i])) for i in source]
            for j in neighbors(core_model):
                if j in core_model:
                    continue
                v = float(ham[j, core_model] @ coeff[source])
                gap = abs(ham[j, j] - energy)
                score = abs(v) / gap if gap else (np.inf if v else 0.0)
                ranked.append((j, score))
            next_model = sorted(
                i for i, _ in sorted(ranked, key=lambda x: (-x[1], dets[x[0]]))[:target]
            )
        else:
            external = [i for i in neighbors(model) if i not in model]
            coupling = ham[np.ix_(external, model)] @ coeff
            if kind == "ACI":
                eps = importance(np.diag(ham)[external] - energy, coupling)
                ranked = sorted(
                    zip(external, eps), key=lambda x: (abs(x[1]), dets[x[0]])
                )
                discarded = 0.0
                next_model = model.copy()
                for j, value in ranked:
                    if discarded + abs(value) <= threshold:
                        discarded += abs(value)
                        correction += value
                    else:
                        next_model.append(j)
                next_model = sorted(set(next_model))
            else:
                products = np.abs(ham[np.ix_(external, model)] * coeff)
                next_model = sorted(
                    model
                    + [
                        j
                        for k, j in enumerate(external)
                        if np.max(products[k]) > threshold
                    ]
                )
        new_energy, new_coeff = diagonalize(next_model)
        if kind == "ACI":
            rank = sorted(
                range(len(next_model)),
                key=lambda i: (new_coeff[i] ** 2, dets[next_model[i]]),
            )
            tail = 0.0
            pruned = next_model.copy()
            for i in rank:
                if len(pruned) <= 1:
                    break
                weight = new_coeff[i] ** 2
                if tail + weight <= gamma * threshold:
                    tail += weight
                    pruned.remove(next_model[i])
                else:
                    break
            stable = (
                next_model == old_model and abs(new_energy - old_energy) < 1e-12
            ) or (next_model == model and pruned == model)
        elif kind == "ASCI":
            stable = next_model == model or (
                old_energy is not None and abs(new_energy - old_energy) < 1e-12
            )
        else:
            stable = next_model == model
        if stable:
            if kind != "ACI":
                external = [i for i in neighbors(next_model) if i not in next_model]
                for j in external:
                    products = ham[j, next_model] * new_coeff
                    if kind == "HBCI":
                        products = products[abs(products) > eps2]
                    numerator = float(np.sum(products))
                    if numerator:
                        correction += numerator**2 / (new_energy - ham[j, j])
            vector = np.zeros(len(dets))
            vector[next_model] = new_coeff
            one, two = fci.direct_spin1.make_rdm12(vector, n, (2, 2))
            return {
                "kind": kind,
                "threshold": threshold,
                "eps2": eps2,
                "core": core,
                "target": target,
                "gamma": gamma,
                "iterations": iteration + 1,
                "energy_variational": new_energy,
                "correction": float(correction),
                "total": new_energy + float(correction),
                "n_determinants": len(next_model),
                "one_rdm": one.ravel().tolist(),
                "two_rdm": two.ravel().tolist(),
            }
        old_model = next_model.copy()
        old_energy = new_energy
        model = pruned if kind == "ACI" else next_model
    raise RuntimeError((kind, threshold, "selection did not converge"))


if __name__ == "__main__":
    cases = []
    for threshold in [0.003, 0.01]:
        cases.append(run("ACI", threshold))
    cases.append(run("ASCI", 0.0, core=5, target=13))
    cases.append(run("ASCI", 0.0, core=10, target=25))
    cases.append(run("HBCI", 0.03, eps2=0.001))
    cases.append(run("HBCI", 1.0, eps2=0.0))
    cases.append(run("HBCI", 1.0, eps2=1.0))
    for kind in ["ACI", "ASCI", "HBCI"]:
        cases.append(run(kind, 0.0, core=36, target=36))
    out = {
        "source": f"PySCF {pyscf.__version__}, direct_spin1 full Hamiltonian; independent dense numpy SCI implementation of the original equations",
        "n_orb": n,
        "n_alpha": 2,
        "n_beta": 2,
        "cases": cases,
    }
    Path(__file__).resolve().parents[1].joinpath(
        "data/selected_ci_reference.json"
    ).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
