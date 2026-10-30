# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Finite-potential-basis OEP from public integrals and independent SciPy BFGS.

Three-center overlap integrals provide the Gaussian multiplicative potentials.
Every coefficient derivative is checked with refined five-point energy differences.
Only public PySCF execution interfaces are used; no implementation source is read.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from gen_ghf import rust_ao_permutation
from gen_mr_molecular import models as mr_models
from pyscf import df, gto, lib, scf
from scipy.linalg import eigh
from scipy.optimize import least_squares, minimize

from embedded_basis import embedded_basis

lib.num_threads(4)


def models():
    water = [
        dict(symbol=s, position=r)
        for s, r in [
            ("O", [0.1, -0.2, 0.3]),
            ("H", [1.5, 0.7, 0.4]),
            ("H", [-0.4, 1.3, -0.2]),
        ]
    ]
    h2 = [dict(symbol="H", position=r) for r in [[0.1, -0.2, -0.7], [0.3, 0.1, 0.7]]]
    result = [
        dict(
            name="He_complete_occupation",
            atoms=[dict(symbol="He", position=[0.2, -0.1, 0.4])],
            constant="STO_3G_GBS",
            basis_name="sto-3g",
        ),
        dict(
            name="He_631g_two_electron",
            atoms=[dict(symbol="He", position=[0.2, -0.1, 0.4])],
            constant="BASIS_6_31G_GBS",
            basis_name="6-31g",
        ),
        dict(
            name="H2_631g_two_electron",
            atoms=h2,
            constant="BASIS_6_31G_GBS",
            basis_name="6-31g",
        ),
        dict(
            name="water_sto3g",
            atoms=water,
            constant="STO_3G_GBS",
            basis_name="sto-3g",
        ),
        dict(
            name="water_631g",
            atoms=water,
            constant="BASIS_6_31G_GBS",
            basis_name="6-31g",
        ),
    ]
    for model in result:
        model["charge"] = 0
        model["basis"] = embedded_basis(
            model["constant"], sorted({a["symbol"] for a in model["atoms"]})
        )
    result.append(dict(mr_models()[2], name="HeH_semilocal_ECP"))
    for model in result:
        model["exponents"] = [0.5, 2.0]
    return result


def ill_posed_models():
    """Cases whose gap-constrained minimization has no attained minimizer."""
    model = dict(
        name="LiH_sto3g",
        atoms=[
            dict(symbol=s, position=r)
            for s, r in [("Li", [0.1, -0.2, 0.3]), ("H", [0.4, 0.2, 3.2])]
        ],
        constant="STO_3G_GBS",
        basis_name="sto-3g",
        charge=0,
        exponents=[0.5, 2.0],
    )
    model["basis"] = embedded_basis("STO_3G_GBS", ["H", "Li"])
    return [model]


def potential(mol, center, exponent):
    auxiliary = gto.M(
        atom=[("H", center)],
        basis={"H": [[0, [exponent, 1.0]]]},
        spin=1,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    normalization = auxiliary.eval_gto("GTOval_cart", np.array([center]))[0, 0]
    return df.incore.aux_e2(mol, auxiliary, intor="int3c1e")[:, :, 0] / normalization


def problem(model):
    mol = gto.M(
        atom=[(a["symbol"], a["position"]) for a in model["atoms"]],
        basis=model["basis"],
        ecp=model.get("ecp") or {},
        charge=model["charge"],
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-12, max_cycle=300)
    assert mf.converged
    h = mf.get_hcore()
    s = mf.get_ovlp()
    reference_density = mf.make_rdm1()
    factor = 1.0 - 1.0 / mol.nelectron
    baseline = h + factor * mf.get_j(dm=reference_density)
    operators = np.array(
        [
            potential(mol, atom["position"], exponent)
            for atom in model["atoms"]
            for exponent in model["exponents"]
        ]
    )
    no = mol.nelectron // 2

    def evaluate(parameters):
        ks = baseline + np.einsum("b,bmn->mn", parameters, operators)
        energies, coefficients = eigh(ks, s)
        occupied = coefficients[:, :no]
        density = 2.0 * occupied @ occupied.T
        j, k = mf.get_jk(dm=density)
        fock = h + j - 0.5 * k
        energy = mol.energy_nuc() + 0.5 * np.einsum("mn,mn", density, h + fock)
        fm = coefficients[:, no:].T @ fock @ occupied
        gap = energies[no:, None] - energies[None, :no]
        if gap.size and gap.min() <= 1e-8:
            raise FloatingPointError(f"occupied/virtual gap {gap.min()}")
        gradient = np.array(
            [
                -4.0 * np.sum(fm * (coefficients[:, no:].T @ g @ occupied) / gap)
                for g in operators
            ]
        )
        return dict(
            energy=float(energy),
            gradient=gradient,
            density=density,
            fock=fock,
            ks=ks,
            orbital_energies=energies,
            gap=float(gap.min()) if gap.size else None,
        )

    return mol, mf, operators, evaluate


def calculate(model):
    mol, mf, operators, evaluate = problem(model)
    reference_density = mf.make_rdm1()
    initial = np.zeros(len(operators))
    solution = minimize(
        lambda p: (evaluate(p)["energy"], evaluate(p)["gradient"]),
        initial,
        jac=True,
        method="BFGS",
        options=dict(gtol=1e-11, maxiter=3000),
    )
    parameters = solution.x
    print(
        "BFGS",
        model["name"],
        solution.fun,
        np.max(abs(solution.jac)),
        parameters,
        flush=True,
    )
    if np.max(abs(evaluate(parameters)["gradient"])) > 3e-10:
        # Polish potential stationarity after energy differences reach double precision.
        polished = least_squares(
            lambda p: evaluate(p)["gradient"],
            parameters,
            jac="3-point",
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
            max_nfev=1000,
        )
        parameters = polished.x
    result = evaluate(parameters)
    assert np.max(abs(result["gradient"])) < 3e-9, (
        model["name"],
        solution.message,
        parameters,
        result["gradient"],
    )
    assert result["energy"] >= mf.e_tot - 1e-9
    trial_parameters = 0.11 * np.sin(np.arange(len(operators)) + 1.0)
    trial = evaluate(trial_parameters)

    def difference(step):
        result = []
        for index in range(len(operators)):
            direction = np.eye(len(operators))[index]
            values = [
                evaluate(trial_parameters + k * step * direction)["energy"]
                for k in [-2, -1, 1, 2]
            ]
            result.append(
                (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)
            )
        return np.array(result)

    coarse, fine = difference(1e-3), difference(5e-4)
    assert np.max(abs(coarse - fine)) < 3e-8
    assert np.max(abs(trial["gradient"] - fine)) < 3e-8
    if "gaussian94" in model:
        order = list(range(mol.nao_nr()))
    else:
        order = rust_ao_permutation(mol, model["constant"])

    def matrix(m):
        return m[np.ix_(order, order)].tolist()

    record = dict(model)
    record.update(
        reference_energy=float(mf.e_tot),
        reference_density=matrix(reference_density),
        operators=[matrix(g) for g in operators],
        energy=result["energy"],
        density=matrix(result["density"]),
        parameters=parameters.tolist(),
        gradient=result["gradient"].tolist(),
        trial_parameters=trial_parameters.tolist(),
        trial_energy=trial["energy"],
        trial_gradient=trial["gradient"].tolist(),
        trial_gradient_fd=fine.tolist(),
        trial_gradient_fd_change=(fine - coarse).tolist(),
        trial_density=matrix(trial["density"]),
        trial_ks=matrix(trial["ks"]),
        trial_fock=matrix(trial["fock"]),
        trial_orbital_energies=trial["orbital_energies"].tolist(),
    )
    print(model["name"], result["energy"], np.max(abs(result["gradient"])), flush=True)
    return record


def ill_posed(model):
    """Every descent closes the HOMO-LUMO gap; no positive-gap minimizer is reached.

    BFGS from zero and from eight seeded random starts lowers the physical HF energy toward
    the RHF value while the Kohn-Sham gap collapses and the parameter gradient grows. The
    native solver must therefore stop with an explicit gap or convergence error.
    """
    mol, mf, operators, evaluate = problem(model)

    def safe(p):
        try:
            r = evaluate(p)
        except FloatingPointError:
            return np.inf, np.zeros(len(p))
        return r["energy"], r["gradient"]

    rng = np.random.default_rng(3)
    starts = [np.zeros(len(operators))] + [
        rng.normal(size=len(operators)) * 0.5 for _ in range(8)
    ]
    runs = []
    for start in starts:
        path = [start]
        minimize(
            safe,
            start,
            jac=True,
            method="BFGS",
            options=dict(gtol=1e-11, maxiter=3000),
            callback=lambda p: path.append(p.copy()),
        )
        final = min((evaluate(p) for p in path), key=lambda r: r["energy"])
        runs.append(
            dict(
                energy_above_rhf=final["energy"] - mf.e_tot,
                gap=final["gap"],
                gradient=float(np.max(abs(final["gradient"]))),
            )
        )
        assert final["energy"] >= mf.e_tot - 1e-9
        # No descent stops at a stationary point with an open gap.
        assert np.max(abs(final["gradient"])) > 1e-4, runs[-1]
    best = min(runs, key=lambda r: r["energy_above_rhf"])
    assert best["gap"] < 1e-5, best
    print(model["name"], "ill-posed", runs, flush=True)
    record = {k: v for k, v in model.items() if k != "basis"}
    record.update(reference_energy=float(mf.e_tot), descents=runs)
    return record


def native_cartesian_order(mol):
    """hf-rs orders Cartesian d shells xx, yy, zz, xy, xz, yz; other shells as PySCF."""
    order, offset = [], 0
    for shell in range(mol.nbas):
        size = (mol.bas_angular(shell) + 1) * (mol.bas_angular(shell) + 2) // 2
        components = [0, 3, 5, 1, 2, 4] if mol.bas_angular(shell) == 2 else range(size)
        order.extend(offset + c for c in components)
        offset += size
    return order


def high_angular_momentum():
    """All Cartesian S/P/D/F/G components, with independent AO normalization."""
    atoms = [dict(symbol="H", position=r) for r in [[0.2, -0.3, 0.1], [1.1, 0.7, -0.8]]]
    basis = {"H": [[momentum, [1.1, 0.7], [0.35, 0.3]] for momentum in range(5)]}
    gbs = (
        "H 0\n"
        + "".join(f"{shell} 2 1.0\n1.1 0.7\n0.35 0.3\n" for shell in "SPDFG")
        + "****\n"
    )
    mol = gto.M(
        atom=[(a["symbol"], a["position"]) for a in atoms],
        basis=basis,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    norm = np.sqrt(np.diag(mol.intor("int1e_ovlp")))
    cases = []
    for center, exponent in [
        ([0.1, 0.3, -0.2], 0.001),
        ([0.3, -0.2, 0.5], 0.7),
        ([-0.4, 0.2, 0.3], 30.0),
    ]:
        matrix = potential(mol, center, exponent) / norm[:, None] / norm[None, :]
        order = native_cartesian_order(mol)
        cases.append(
            dict(
                center=center,
                exponent=exponent,
                matrix=matrix[np.ix_(order, order)].tolist(),
            )
        )
    return dict(atoms=atoms, gaussian94=gbs, cases=cases)


if __name__ == "__main__":
    output = dict(
        interface=f"PySCF {pyscf.__version__}: public three-center overlaps and RHF J/K; independent SciPy BFGS",
        cases=[calculate(model) for model in models()],
        ill_posed_cases=[ill_posed(model) for model in ill_posed_models()],
        high_angular_momentum=high_angular_momentum(),
    )
    path = Path(__file__).resolve().parents[1] / "data/oep_reference.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
