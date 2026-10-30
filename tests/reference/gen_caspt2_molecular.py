# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Small molecular CASPT2 energy and all-coordinate numerical reference stencils.

PySCF supplies independently converged CAS references and full determinant
Hamiltonian/excitation matrices. NumPy solves the contracted equations directly.
The production solver uses excitation-generated sparse states and iterative PCG.
"""

import json
from pathlib import Path

import numpy as np
from gen_caspt2 import contracted_problem
from gen_mr_molecular import models, molecule
from pyscf import ao2mo, fci


def evaluate(model, coordinates=None, points=None):
    mf, mc, c = molecule(model, coordinates, reference_only=True)
    nc, na, nelec = mc.ncore, mc.ncas, mc.nelecas
    one = fci.direct_spin1.make_rdm1(mc.ci, na, nelec)
    h = c.T @ mf.get_hcore() @ c
    g = ao2mo.kernel(mf.mol, c, compact=False).reshape((c.shape[1],) * 4)
    results = []
    for ipea, real, imag in points or [(0.0, 0.0, 0.0), (0.25, 0.0, 0.12)]:
        oracle = contracted_problem(
            h, g, nc, na, nelec, mc.ci, one, ipea_values=[ipea], shifts=[(real, imag)]
        )
        result = oracle["results"][0]
        results.append(
            dict(
                ipea_shift=ipea,
                real_shift=real,
                imaginary_shift=imag,
                e_total=float(mc.e_tot + result["energy"]),
            )
        )
    return results


def main():
    step = 1e-3
    cases = []
    for index in [0, 2, 3]:
        model = models()[index]
        coordinates = np.array([a["position"] for a in model["atoms"]])
        energies = evaluate(model)
        gradients = [[], []]
        for k in range(coordinates.size):
            values = []
            for offset in [-2, -1, 1, 2]:
                point = coordinates.copy()
                point.ravel()[k] += offset * step
                values.append(evaluate(model, point))
            for mode in range(2):
                a, b, c, d = [v[mode]["e_total"] for v in values]
                gradients[mode].append((a - 8 * b + 8 * c - d) / (12 * step))
        cases.append(dict(model_index=index, results=energies, gradient=gradients))
        print(model["name"], energies, flush=True)
    path = Path(__file__).resolve().parents[1] / "data/caspt2_molecular_reference.json"
    path.write_text(
        json.dumps(
            dict(
                source=__doc__,
                step=step,
                cases=cases,
                extra_cases=[dict(model_index=1, results=evaluate(models()[1]))],
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
