# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent LHF/CEDA grid equations using public PySCF operators and SciPy.

No external implementation source is read. Public quadrature output is used
only as an oracle; the native code constructs its own product quadrature.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from gen_ghf import rust_ao_permutation
from gen_oep import high_angular_momentum, models, native_cartesian_order
from pyscf import df, dft, gto, lib, scf
from scipy.linalg import eigh

lib.num_threads(4)


def point_operators(mol, coords):
    auxiliary = gto.fakemol_for_charges(coords)
    auxiliary.cart = True
    return df.incore.aux_e2(mol, auxiliary, intor="int3c2e").transpose(2, 0, 1)


def calculate(model, level):
    mol = gto.M(
        atom=[(a["symbol"], a["position"]) for a in model["atoms"]],
        basis=model["basis"],
        ecp=model.get("ecp") or {},
        charge=model["charge"],
        cart=True,
        unit="Bohr",
        verbose=0,
    )
    hf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-12, max_cycle=300)
    assert hf.converged
    x = hf.mo_coeff
    h, s = hf.get_hcore(), hf.get_ovlp()
    no = mol.nelectron // 2
    pairs = [(i, j) for i in range(no) for j in range(i + 1)]
    factors = np.array([1.0 if i == j else np.sqrt(2.0) for i, j in pairs])
    grid_error = 0.0
    if no > 1:
        grid = dft.gen_grid.Grids(mol)
        grid.level = level
        grid.build()
        nonzero = grid.weights != 0
        coords, weights = grid.coords[nonzero], grid.weights[nonzero]
        ao = mol.eval_gto("GTOval_cart", coords)
        point = point_operators(mol, coords)
        for index in [0, len(coords) // 3, len(coords) - 1]:
            with mol.with_rinv_origin(coords[index]):
                assert np.max(abs(point[index] - mol.intor("int1e_rinv"))) < 1e-10
        grid_error = float(np.max(abs(x.T @ ((ao.T * weights) @ ao - s) @ x)))

    def exchange(c, j, k):
        if no == 1:
            return -0.5 * j, -0.5 * j, 0.0
        co = c[:, :no]
        phi = ao @ co
        rho = np.einsum("pi,pi->p", phi, phi)
        assert rho.min() > 0
        u = phi / np.sqrt(rho[:, None])
        w = u @ co.T
        slater = -np.einsum("pm,pmn,pn->p", w, point, w, optimize=True)
        features = np.array(
            [factor * u[:, i] * u[:, j] for (i, j), factor in zip(pairs, factors)]
        ).T
        operators = np.einsum(
            "p,pm,pn,pa->amn", weights, ao, ao, features, optimize=True
        )
        vs = (ao.T * (weights * slater)) @ ao
        bmat = co.T @ (vs + 0.5 * k) @ co
        b = np.array([factor * bmat[i, j] for (i, j), factor in zip(pairs, factors)])
        system = np.eye(len(pairs)) - np.einsum(
            "p,pa,pb->ab", weights * rho, features, features, optimize=True
        )
        coefficients = np.zeros(len(pairs))
        coefficients[:-1] = np.linalg.solve(system[:-1, :-1], b[:-1])
        error = float(np.max(abs(system @ coefficients - b)))
        return vs + np.einsum("a,amn->mn", coefficients, operators), vs, error

    c = hf.mo_coeff.copy()
    history = []
    previous, oldp, reference = None, None, None
    for iteration in range(100):
        p = 2 * c[:, :no] @ c[:, :no].T
        j, k = hf.get_jk(dm=p)
        vx, vs, linear_error = exchange(c, j, k)
        if reference is None:
            reference = dict(potential=vx.copy(), slater=vs.copy())
        fock = h + j + vx
        energy = hf.energy_tot(dm=p)
        residual = x.T @ (fock @ p @ s - s @ p @ fock) @ x
        dp = np.max(abs(p - oldp)) if oldp is not None else np.inf
        de = abs(energy - previous) if previous is not None else np.inf
        if np.linalg.norm(residual) < 1e-10 and dp < 1e-11 and de < 1e-12:
            break
        previous, oldp = energy, p
        history.append((x.T @ fock @ x, residual.ravel()))
        history = history[-8:]
        count = len(history)
        system = np.empty((count + 1, count + 1))
        system[-1, :] = system[:, -1] = -1
        system[-1, -1] = 0
        system[:count, :count] = [
            [np.dot(a[1], b[1]) for b in history] for a in history
        ]
        rhs = np.zeros(count + 1)
        rhs[-1] = -1
        try:
            coefficients = np.linalg.solve(system, rhs)[:count]
            extrapolated = sum(
                value * state[0] for value, state in zip(coefficients, history)
            )
        except np.linalg.LinAlgError:
            extrapolated = history[-1][0]
        c = x @ eigh(extrapolated)[1]
    else:
        raise AssertionError("independent LHF SCF did not converge")
    order = (
        rust_ao_permutation(mol, model["constant"])
        if "constant" in model
        else list(range(mol.nao_nr()))
    )

    def matrix(value):
        return value[np.ix_(order, order)].tolist()

    result = {key: value for key, value in model.items() if key != "basis"}
    result.update(
        reference_energy=float(hf.e_tot),
        energy=float(energy),
        density=matrix(p),
        potential=matrix(vx),
        slater=matrix(vs),
        fock=matrix(fock),
        reference_potential=matrix(reference["potential"]),
        reference_slater=matrix(reference["slater"]),
        grid_level=level,
        grid_overlap_error=grid_error,
        linear_residual=linear_error,
        orbital_energies=eigh(fock, s)[0].tolist(),
    )
    print(model["name"], level, energy, grid_error, linear_error, flush=True)
    return result


def angular_operators():
    model = high_angular_momentum()
    mol = gto.M(
        atom=[(a["symbol"], a["position"]) for a in model["atoms"]],
        basis={"H": [[momentum, [1.1, 0.7], [0.35, 0.3]] for momentum in range(5)]},
        cart=True,
        unit="Bohr",
        verbose=0,
    )
    centers = np.array([[0.1, 0.3, -0.2], [0.3, -0.2, 0.5], [-0.4, 0.2, 0.3]])
    point = point_operators(mol, centers)
    norm = np.sqrt(np.diag(mol.intor("int1e_ovlp")))
    order = native_cartesian_order(mol)
    return dict(
        atoms=model["atoms"],
        gaussian94=model["gaussian94"],
        operators=[
            dict(
                center=center.tolist(),
                matrix=(value / norm[:, None] / norm[None, :])[
                    np.ix_(order, order)
                ].tolist(),
            )
            for center, value in zip(centers, point)
        ],
    )


if __name__ == "__main__":
    cases = []
    for model in models():
        coarse, fine = calculate(model, 4), calculate(model, 5)
        assert abs(coarse["energy"] - fine["energy"]) < 1e-7
        fine["energy_grid_change"] = fine["energy"] - coarse["energy"]
        cases.append(fine)
    output = dict(
        interface=f"PySCF {pyscf.__version__} public grid, point integrals and RHF interfaces; independent CEDA equations",
        cases=cases,
        angular_operators=angular_operators(),
    )
    path = Path(__file__).resolve().parents[1] / "data/lhf_reference.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
