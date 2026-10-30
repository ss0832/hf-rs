# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Coherent-state Pauli-Fierz RHF via public PySCF interfaces and exact moments.

The photon zero-point constant is omitted. Minimizing the coherent photon
displacement leaves one half of the molecular dipole variance. Exact AO r*r
integrals retain the complement outside the finite orbital space. A product of
two projected dipole matrices is deliberately not used for these integrals.
All nuclear components use independently refined five-point energy differences.
No external implementation source is read.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from gen_ghf import rust_ao_permutation
from gen_mr_molecular import models as mr_models
from pyscf import gto, lib, scf

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
    models = [
        dict(
            name="H2_zero",
            atoms=h2,
            constant="STO_3G_GBS",
            basis_name="sto-3g",
            charge=0,
            coupling=[0.0, 0.0, 0.0],
        ),
        dict(
            name="H2_tilted",
            atoms=h2,
            constant="STO_3G_GBS",
            basis_name="sto-3g",
            charge=0,
            coupling=[0.08, -0.03, 0.12],
        ),
        dict(
            name="He_complete_occupation",
            atoms=[dict(symbol="He", position=[0.2, -0.1, 0.4])],
            constant="STO_3G_GBS",
            basis_name="sto-3g",
            charge=0,
            coupling=[0.1, -0.04, 0.07],
        ),
        dict(
            name="LiH",
            atoms=[
                dict(symbol=s, position=r)
                for s, r in [("Li", [0.1, -0.2, 0.3]), ("H", [0.4, 0.2, 3.2])]
            ],
            constant="STO_3G_GBS",
            basis_name="sto-3g",
            charge=0,
            coupling=[0.07, 0.04, 0.08],
        ),
        dict(
            name="water_sto3g",
            atoms=water,
            constant="STO_3G_GBS",
            basis_name="sto-3g",
            charge=0,
            coupling=[0.08, -0.03, 0.05],
        ),
        dict(
            name="water_631g",
            atoms=water,
            constant="BASIS_6_31G_GBS",
            basis_name="6-31g",
            charge=0,
            coupling=[0.08, -0.03, 0.05],
        ),
    ]
    ecp = dict(mr_models()[2])
    ecp.update(name="HeH_semilocal_ECP", coupling=[0.06, -0.04, 0.09])
    models.append(ecp)
    for model in models:
        model["frequency"] = 0.13
        if "basis" not in model:
            model["basis"] = embedded_basis(
                model["constant"], sorted({a["symbol"] for a in model["atoms"]})
            )
    return models


def legacy_models():
    """Energy-only cases replacing qualitative legacy assertions (no gradient)."""
    formamide = [
        dict(symbol=s, position=r)
        for s, r in [
            ("C", [0.000, 0.730, 0.0]),
            ("O", [2.054, 1.542, 0.0]),
            ("N", [-2.054, -0.669, 0.0]),
            ("H", [0.198, -1.332, 0.0]),
            ("H", [-3.719, 0.170, 0.0]),
            ("H", [-1.980, -2.553, 0.0]),
        ]
    ]
    models = [
        dict(name=f"formamide_{label}", coupling=coupling)
        for label, coupling in [
            ("zero", [0.0, 0.0, 0.0]),
            ("in_plane", [0.0, 0.05, 0.0]),
            ("out_of_plane", [0.0, 0.0, 0.05]),
        ]
    ]
    for model in models:
        model.update(
            atoms=formamide,
            constant="STO_3G_GBS",
            basis_name="sto-3g",
            charge=0,
            frequency=0.1,
            basis=embedded_basis("STO_3G_GBS", ["C", "H", "N", "O"]),
        )
    return models


def calculate(model, positions=None):
    if positions is None:
        positions = [a["position"] for a in model["atoms"]]
    mol = gto.M(
        atom=[(a["symbol"], r) for a, r in zip(model["atoms"], positions)],
        basis=model["basis"],
        ecp=model.get("ecp") or {},
        charge=model["charge"],
        cart=True,
        unit="Bohr",
        verbose=0,
    )
    coupling = np.array(model["coupling"])
    dip = np.einsum("a,apq->pq", coupling, mol.intor("int1e_r"))
    quadratic = np.einsum(
        "a,b,abpq->pq",
        coupling,
        coupling,
        mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao),
    )
    mf = scf.RHF(mol)
    physical_core = mf.get_hcore()
    effective = physical_core + quadratic / 2
    mf.get_hcore = lambda *args: effective

    def veff(mol=None, dm=None, *args, **kwargs):
        j, k = mf.get_jk(mol, dm)
        return j - k / 2 - dip @ dm @ dip / 2

    mf.get_veff = veff
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-11, 300
    mf.kernel()
    residual = (
        np.max(np.abs(mf.get_grad(mf.mo_coeff, mf.mo_occ)))
        if mol.nelectron // 2 < mol.nao
        else 0.0
    )
    assert mf.converged and residual < 1e-9, (model["name"], residual)
    density = mf.make_rdm1()
    variance_energy = (
        np.sum(density * quadratic) / 2 - np.trace(density @ dip @ density @ dip) / 4
    )
    physical_j, physical_k = mf.get_jk(dm=density)
    physical_electronic = np.sum(
        density * (physical_core + physical_j / 2 - physical_k / 4)
    )
    assert (
        abs(mf.e_tot - physical_electronic - mol.energy_nuc() - variance_energy) < 1e-10
    )
    return mol, mf, dip, quadratic, variance_energy


def main():
    result = []
    rng = np.random.default_rng(631709)
    for model in models():
        mol, mf, dip, quadratic, variance = calculate(model)
        n = mol.nao
        order = (
            rust_ao_permutation(mol, model["constant"])
            if "constant" in model
            else list(range(n))
        )
        density = mf.make_rdm1()
        fock = mf.get_fock(dm=density, diis=None)
        arbitrary = rng.normal(size=(n, n))
        arbitrary = (arbitrary + arbitrary.T) / 5
        arbitrary_fock = mf.get_fock(dm=arbitrary, diis=None)
        dipole = mf.dip_moment(unit="AU", verbose=0)
        weighted_dipole = np.dot(model["coupling"], dipole)
        positions = np.array([a["position"] for a in model["atoms"]])

        def difference(step):
            gradient = np.zeros_like(positions)
            for i in range(positions.size):
                for factor, coefficient in [(2, -1), (1, 8), (-1, -8), (-2, 1)]:
                    moved = positions.copy()
                    moved.flat[i] += factor * step
                    gradient.flat[i] += (
                        coefficient * calculate(model, moved)[1].e_tot / (12 * step)
                    )
            return gradient

        coarse, fine = difference(1e-3), difference(5e-4)
        refinement = np.max(np.abs(coarse - fine))
        assert refinement < 3e-8, (model["name"], refinement)
        record = {k: v for k, v in model.items() if k != "basis"}
        record.update(
            energy=float(mf.e_tot),
            cavity_energy=float(variance),
            bare_dipole_self_energy=float(variance + weighted_dipole**2 / 2),
            coherent_displacement=float(
                weighted_dipole / np.sqrt(2 * model["frequency"])
            ),
            dipole_au=dipole.tolist(),
            gradient=fine.tolist(),
            gradient_refinement_error=float(refinement),
            matrices={
                name: value[np.ix_(order, order)].tolist()
                for name, value in dict(
                    dipole=dip,
                    quadratic=quadratic,
                    density=density,
                    fock=fock,
                    arbitrary_density=arbitrary,
                    arbitrary_fock=arbitrary_fock,
                ).items()
            },
        )
        result.append(record)
        print(
            model["name"],
            mf.e_tot,
            "cavity",
            variance,
            "gradient_refinement",
            refinement,
            flush=True,
        )
    legacy = []
    for model in legacy_models():
        mol, mf, dip, quadratic, variance = calculate(model)
        record = {k: v for k, v in model.items() if k != "basis"}
        record.update(
            energy=float(mf.e_tot),
            cavity_energy=float(variance),
            dipole_au=mf.dip_moment(unit="AU", verbose=0).tolist(),
        )
        legacy.append(record)
        print(model["name"], mf.e_tot, "cavity", variance, flush=True)
    path = Path(__file__).resolve().parents[1] / "data/qed_hf_reference.json"
    path.write_text(
        json.dumps(
            dict(
                source=__doc__,
                pyscf_version=pyscf.__version__,
                cases=result,
                legacy_cases=legacy,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
