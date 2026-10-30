# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""RHF structure-to-property workflows: H2O/6-31G* and 85RbH/def2-SVP/ECP28.

Only public PySCF execution interfaces are used. SciPy BFGS optimizes internal
coordinates with analytic RHF gradients. Water has an analytic Hessian oracle;
RbH uses two independently refined five-point differences of analytic ECP
gradients because the public PySCF RHF Hessian does not include ECP response.
All Cartesian Hessian entries are retained, along with electric multipoles and
independently projected isotope-specific harmonic frequencies. No external
implementation source is read.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, lib, scf
from scipy import constants
from scipy.optimize import brentq, minimize, root

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)
MASSES = {"H": 1.00782503223, "O": 15.99491461957, "Rb": 84.91178973790}
CM1 = np.sqrt(
    constants.physical_constants["Hartree energy"][0]
    / (constants.physical_constants["Bohr radius"][0] ** 2 * constants.atomic_mass)
) / (2 * np.pi * constants.c * 100)


def geometry(name, x):
    if name == "water":
        r, theta = x
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [r * np.sin(theta / 2), 0.0, r * np.cos(theta / 2)],
                [-r * np.sin(theta / 2), 0.0, r * np.cos(theta / 2)],
            ]
        )
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, x[0]]])


def jacobian(name, x):
    if name == "water":
        r, theta = x
        s, c = np.sin(theta / 2), np.cos(theta / 2)
        return np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [s, r * c / 2],
                [0, 0],
                [c, -r * s / 2],
                [-s, -r * c / 2],
                [0, 0],
                [c, -r * s / 2],
            ]
        )
    return np.array([[0], [0], [0], [0], [0], [1]])


def frequencies(symbols, xyz, hessian):
    mass = np.array([MASSES[s] for s in symbols])
    root = np.repeat(np.sqrt(mass), 3)
    centered = xyz - np.sum(xyz * mass[:, None], axis=0) / np.sum(mass)
    modes = []
    for axis in np.eye(3):
        modes.append((np.ones_like(xyz) * axis * np.sqrt(mass[:, None])).ravel())
        modes.append((np.cross(centered, axis) * np.sqrt(mass[:, None])).ravel())
    u, sigma, _ = np.linalg.svd(np.array(modes).T, full_matrices=True)
    q = u[:, np.count_nonzero(sigma > 1e-9) :]
    values = np.linalg.eigvalsh(q.T @ (hessian / np.outer(root, root)) @ q)
    return np.sign(values) * np.sqrt(np.abs(values)) * CM1


def run(name, symbols, basis, ecp, initial):
    density = None

    def calculate(xyz):
        nonlocal density
        mol = gto.M(
            atom=list(zip(symbols, xyz)),
            basis=basis,
            ecp=ecp,
            unit="Bohr",
            cart=True,
            verbose=0,
        )
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-11
        mf.max_cycle = 300
        mf.kernel(dm0=density)
        assert mf.converged, name
        density = mf.make_rdm1()
        return mol, mf

    def objective(x):
        _, mf = calculate(geometry(name, x))
        return mf.e_tot, jacobian(name, x).T @ mf.nuc_grad_method().kernel().ravel()

    opt = minimize(
        objective, initial, jac=True, method="BFGS", options=dict(gtol=1e-9, maxiter=60)
    )
    # Energy line searches lose resolution before the force vanishes. Finish with
    # a force root, using the analytic internal-coordinate Hessian for water.
    if name == "water":

        def force_and_response(x):
            positions = geometry(name, x)
            _, mf = calculate(positions)
            g = mf.nuc_grad_method().kernel().ravel()
            h = mf.Hessian().kernel().transpose(0, 2, 1, 3).reshape(9, 9)
            j = jacobian(name, x)
            response = j.T @ h @ j
            cross = np.dot(g, j[:, 1]) / x[0]
            response[0, 1] += cross
            response[1, 0] += cross
            response[1, 1] -= np.dot(g, positions.ravel()) / 4
            return j.T @ g, response

        stationary = root(force_and_response, opt.x, jac=True, tol=1e-10)
        optimized = stationary.x
    else:
        center = opt.x[0]
        optimized = np.array(
            [
                brentq(
                    lambda r: objective([r])[1][0],
                    center - 0.05,
                    center + 0.05,
                    xtol=1e-11,
                )
            ]
        )
    assert np.max(np.abs(objective(optimized)[1])) < 2e-9, name
    xyz = geometry(name, optimized)
    mol, mf = calculate(xyz)
    gradient = mf.nuc_grad_method().kernel()
    energy = float(mf.e_tot)
    dipole = mf.dip_moment(unit="AU", verbose=0)
    rr = mol.intor("int1e_rr").reshape(3, 3, mol.nao, mol.nao)
    second = np.einsum("a,ai,aj->ij", mol.atom_charges(), xyz, xyz) - np.einsum(
        "ijpq,qp->ij", rr, mf.make_rdm1()
    )
    quadrupole = (3 * second - np.eye(3) * np.trace(second)) / 2
    if ecp:

        def difference(step):
            h = np.zeros((xyz.size, xyz.size))
            for k in range(xyz.size):
                for factor, coefficient in [(2, -1), (1, 8), (-1, -8), (-2, 1)]:
                    moved = xyz.copy()
                    moved.flat[k] += factor * step
                    _, displaced = calculate(moved)
                    h[:, k] += (
                        coefficient
                        * displaced.nuc_grad_method().kernel().ravel()
                        / (12 * step)
                    )
            return h

        hessian = difference(4e-4)
        refined = difference(2e-4)
        refinement_error = float(np.max(np.abs(refined - hessian)))
        assert refinement_error < 2e-7, refinement_error
        hessian = refined
    else:
        hessian = (
            mf.Hessian().kernel().transpose(0, 2, 1, 3).reshape(xyz.size, xyz.size)
        )
        refinement_error = None
    result = dict(
        name=name,
        symbols=symbols,
        initial=initial,
        optimized=optimized.tolist(),
        positions_bohr=xyz.tolist(),
        energy=energy,
        gradient=gradient.tolist(),
        dipole_au=dipole.tolist(),
        quadrupole_au=quadrupole.tolist(),
        hessian=hessian.tolist(),
        frequencies_cm1=frequencies(symbols, xyz, hessian).tolist(),
        masses_amu=[MASSES[s] for s in symbols],
        ecp_hessian_refinement_error=refinement_error,
    )
    print(
        name,
        "optimized",
        optimized,
        "energy",
        energy,
        "dipole_au",
        dipole,
        "frequencies_cm1",
        result["frequencies_cm1"],
        "hessian_refinement",
        refinement_error,
        flush=True,
    )
    return result


def main():
    water = run(
        "water",
        ["O", "H", "H"],
        embedded_basis("BASIS_6_31G_STAR_GBS", ["H", "O"]),
        {},
        [1.84, np.deg2rad(104.5)],
    )
    # Public built-in def2 ECP data; the orbital basis uses exact repo data.
    ecp = {"Rb": gto.basis.load_ecp("def2-svp", "Rb")}
    rbh = run(
        "rbh",
        ["Rb", "H"],
        embedded_basis("BASIS_DEF2_SVP_GBS", ["Rb", "H"]),
        ecp,
        [4.4],
    )
    out = dict(
        source=__doc__,
        pyscf_version=pyscf.__version__,
        basis_sha256=basis_hash(),
        cm1_conversion=CM1,
        cases=[water, rbh],
    )
    path = Path(__file__).resolve().parents[1] / "data/property_workflow_reference.json"
    path.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
