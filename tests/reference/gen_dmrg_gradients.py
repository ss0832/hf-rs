# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""All-coordinate PySCF CASCI/CASSCF energy stencils for large-M DMRG checks."""

import json
from pathlib import Path

import numpy as np
from gen_mr_molecular import dense_ci, exact_cas, models
from pyscf import gto, lib, mcscf, scf
from scipy.linalg import expm
from scipy.optimize import root

lib.num_threads(4)


def energies(model, coordinates):
    mol = gto.M(
        atom=[
            (a["symbol"], r) for a, r in zip(model["atoms"], coordinates, strict=True)
        ],
        basis=model["basis"],
        ecp=model.get("ecp") or {},
        charge=model["charge"],
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-12, max_cycle=300)
    assert mf.converged
    values = []
    for method in (mcscf.CASCI, mcscf.CASSCF):
        solver = method(mf, 2, 2)
        solver.fcisolver.kernel = dense_ci
        solver.fcisolver.approx_kernel = dense_ci
        solver.conv_tol = 1e-13
        if method is mcscf.CASSCF:
            solver.conv_tol_grad = 1e-10
            solver.max_cycle_macro = 150
        solver.kernel()
        if method is mcscf.CASSCF:
            anchor = solver.mo_coeff.copy()
            classes = [0] * solver.ncore + [1] * solver.ncas
            classes += [2] * (anchor.shape[1] - len(classes))
            pairs = [
                (p, q)
                for p in range(len(classes))
                for q in range(p)
                if classes[p] != classes[q]
            ]
            ci_solver = mcscf.CASCI(mf, 2, 2)
            ci_solver.canonicalization = False

            def gradient(parameters):
                kappa = np.zeros((len(classes),) * 2)
                for value, (p, q) in zip(parameters, pairs, strict=True):
                    kappa[p, q], kappa[q, p] = value, -value
                trial = anchor @ expm(kappa)
                exact_cas(ci_solver, trial)
                solver.mo_coeff, solver.ci = trial, ci_solver.ci
                return solver.get_grad(trial)

            solution = root(gradient, np.zeros(len(pairs)), tol=1e-10)
            assert np.max(abs(gradient(solution.x))) < 1e-10
            solver.e_tot = ci_solver.e_tot
        values.append(float(solver.e_tot))
    return values


def main():
    cases = []
    step = 1e-3
    for index in (0, 2):
        model = models()[index]
        coordinates = np.array([a["position"] for a in model["atoms"]])
        stencils = []
        for component in range(coordinates.size):
            values = []
            for offset in (-2, -1, 1, 2):
                shifted = coordinates.copy()
                shifted.flat[component] += offset * step
                values.append(energies(model, shifted))
            stencils.append(values)
            print(model["name"], component, flush=True)
        stencils = np.array(stencils)
        gradient = np.einsum("s,csm->mc", [1, -8, 8, -1], stencils) / (12 * step)
        cases.append(
            dict(
                model_index=index,
                energies=energies(model, coordinates),
                gradient=gradient.tolist(),
                stencils=stencils.tolist(),
            )
        )
    Path(__file__).resolve().parents[1].joinpath(
        "data/dmrg_gradient_reference.json"
    ).write_text(
        json.dumps(
            dict(source=__doc__, step=step, methods=["CASCI", "CASSCF"], cases=cases),
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
