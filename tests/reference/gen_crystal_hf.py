# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Infinite-chain RHF energies per unit cell from finite PySCF molecular chains.

For a gapped, closed-shell chain the RHF energy of an N-unit finite chain is
E(N) = N e + b + c/N + ..., where e is the energy per cell of the infinite periodic RHF
solution. Nonpolar chains (alternating H2, capped trans-polyacetylene) converge
exponentially, so e is the difference E(N+1) - E(N) at the largest N. The water chain has
dipoles perpendicular to the axis; its dipole-dipole lattice tail gives the 1/N term, and e
is obtained by least squares on E(N) = N e + b + c/N + d/N^2. A vacuum-padded PySCF pbc
calculation (dimension=3, GDF, Ewald exchange correction) is recorded as a coarse
cross-check only; PySCF 2.14 does not build dimension=1 cells with uniform grids.
Contracted Cartesian AO space, embedded basis data. No periodic-HF source is read.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, lib, scf
from pyscf.pbc import df as pdf
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf

from embedded_basis import embedded_basis

lib.num_threads(8)


def h2_chain():
    unit = [("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 1.4])]
    return dict(
        name="H2_chain", unit=unit, lattice=4.0, constant="STO_3G_GBS", caps=None
    )


def polyacetylene():
    a = 4.65
    unit = [
        ("C", [0.0, 0.68, 0.0]),
        ("C", [0.0, -0.68, 2.20]),
        ("H", [0.0, 2.70, -0.40]),
        ("H", [0.0, -2.70, 2.60]),
    ]

    def caps(n):
        return [("H", [0.0, -0.31, -1.78]), ("H", [0.0, 0.31, 3.98 + (n - 1) * a])]

    return dict(
        name="trans_polyacetylene",
        unit=unit,
        lattice=a,
        constant="STO_3G_GBS",
        caps=caps,
    )


def water_chain():
    unit = [("O", [0.0, 0.0, 0.0]), ("H", [1.43, 1.1, 0.0]), ("H", [-1.43, 1.1, 0.0])]
    return dict(
        name="water_chain", unit=unit, lattice=5.5, constant="STO_3G_GBS", caps=None
    )


def chain_energy(model, n):
    atoms = []
    for k in range(n):
        atoms += [
            (s, [p[0], p[1], p[2] + k * model["lattice"]]) for s, p in model["unit"]
        ]
    if model["caps"]:
        atoms += model["caps"](n)
    symbols = sorted({s for s, _ in atoms})
    mol = gto.M(
        atom=atoms,
        basis=embedded_basis(model["constant"], symbols),
        cart=True,
        unit="Bohr",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-9, 300
    mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf).run(mf.mo_coeff, mf.mo_occ, conv_tol=1e-12)
    assert mf.converged
    stable = mf.stability(return_status=True)[2]
    assert stable, (model["name"], n)
    return float(mf.e_tot)


def per_cell(model):
    if model["name"] == "water_chain":
        sizes = [10, 14, 18, 22, 26, 30]
        energies = [chain_energy(model, n) for n in sizes]
        n = np.array(sizes, float)
        design = np.array([n, np.ones_like(n), 1 / n, 1 / n**2]).T
        fit = np.linalg.lstsq(design, energies, rcond=None)[0]
        residual = float(np.max(abs(design @ fit - energies)))
        # Stability of the extrapolation: drop the smallest chain.
        refit = np.linalg.lstsq(design[1:], energies[1:], rcond=None)[0]
        return dict(
            energy_per_cell=float(fit[0]),
            extrapolation_change=float(fit[0] - refit[0]),
            fit_residual=residual,
            chain_sizes=sizes,
            chain_energies=energies,
        )
    sizes = [12, 16, 20, 24]
    energies = [chain_energy(model, n) for n in sizes]
    differences = np.diff(energies) / np.diff(sizes)
    return dict(
        energy_per_cell=float(differences[-1]),
        extrapolation_change=float(differences[-1] - differences[-2]),
        chain_sizes=sizes,
        chain_energies=energies,
    )


def padded_pbc(model, nk=16, vacuum=30.0):
    cell = pgto.Cell()
    cell.atom = model["unit"]
    cell.a = np.diag([vacuum, vacuum, model["lattice"]])
    cell.basis = embedded_basis(
        model["constant"], sorted({s for s, _ in model["unit"]})
    )
    cell.cart, cell.unit, cell.verbose = True, "Bohr", 0
    cell.build()
    kpts = cell.make_kpts([1, 1, nk])
    mf = pscf.KRHF(cell, kpts, exxdiv="ewald")
    mf.with_df = pdf.GDF(cell, kpts)
    return float(mf.run(conv_tol=1e-10).e_tot)


def main():
    cases = []
    for model in [h2_chain(), polyacetylene(), water_chain()]:
        record = dict(
            name=model["name"],
            constant=model["constant"],
            lattice_constant=model["lattice"],
            unit=[dict(symbol=s, position=p) for s, p in model["unit"]],
        )
        record.update(per_cell(model))
        record["padded_pbc_energy"] = padded_pbc(model)
        cases.append(record)
        print(
            model["name"],
            record["energy_per_cell"],
            record["extrapolation_change"],
            record["padded_pbc_energy"],
            flush=True,
        )
    path = Path(__file__).resolve().parents[1] / "data/crystal_hf_reference.json"
    path.write_text(
        json.dumps(
            dict(source=__doc__, pyscf_version=pyscf.__version__, cases=cases), indent=1
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
