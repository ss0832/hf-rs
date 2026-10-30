# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Public PySCF X2C matrices and independent nonlinear matrix-response oracle.

Contracted Cartesian AO space, point nuclei, scalar one-electron X2C and
untransformed two-electron interactions. No geometry differences in production.
"""

import json
from pathlib import Path

import numpy as np
from gen_ghf import rust_ao_permutation
from pyscf import __version__ as pyscf_version
from pyscf import gto, lib, scf
from scipy.linalg import block_diag, eigh

from embedded_basis import embedded_basis

lib.num_threads(4)
lib.param.LIGHT_SPEED = 137.035999084
C = lib.param.LIGHT_SPEED


def power(s, exponent):
    values, u = eigh(s)
    assert np.min(values) > 0
    return (u * values**exponent) @ u.T


def nonlinear_core(s, t, v, w):
    """Generalized 4c eigenvectors and polar large-component normalization."""
    n = len(s)
    h4 = np.block([[v, t], [t, w / (4 * C * C) - t]])
    values, vectors = eigh(h4, block_diag(s, t / (2 * C * C)))
    large = power(s, 0.5) @ vectors[:n, n:]
    unitary = large @ power(large.T @ large, -0.5)
    return power(s, 0.5) @ (unitary * values[n:]) @ unitary.T @ power(s, 0.5)


def models():
    return [
        ("H2", [("H", [0.0, 0.0, -0.7]), ("H", [0.0, 0.0, 0.7])], "STO_3G_GBS"),
        ("He", [("He", [0.1, -0.2, 0.3])], "STO_3G_GBS"),
        ("Ne", [("Ne", [0.1, -0.2, 0.3])], "STO_3G_GBS"),
        (
            "water",
            [
                ("O", [0.1, -0.2, 0.3]),
                ("H", [1.53, 0.68, 0.3]),
                ("H", [-1.33, 0.68, 0.3]),
            ],
            "BASIS_6_31G_GBS",
        ),
        ("HCl", [("Cl", [0.1, -0.2, 0.3]), ("H", [0.1, -0.2, 2.7])], "BASIS_6_31G_GBS"),
    ]


def reference(mol):
    mf = scf.RHF(mol).sfx2c1e()
    mf.with_x2c.xuncontract = False
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 300
    mf.chkfile = None
    mf.kernel()
    assert mf.converged
    return mf


def main():
    records = []
    for name, atoms, constant in models():
        mol = gto.M(
            atom=atoms,
            basis=embedded_basis(constant, sorted({a[0] for a in atoms})),
            cart=True,
            unit="Bohr",
            verbose=0,
        )
        matrices = [
            mol.intor(key)
            for key in ("int1e_ovlp", "int1e_kin", "int1e_nuc", "int1e_pnucp")
        ]
        mf = reference(mol)
        target = mf.get_hcore()
        assert np.max(abs(nonlinear_core(*matrices) - target)) < 1e-8
        order = rust_ao_permutation(mol, constant)
        n = len(target)
        record = dict(
            name=name,
            atoms=[dict(symbol=s, position=r) for s, r in atoms],
            basis_constant=constant,
            e_total=float(mf.e_tot),
            matrices=[m[np.ix_(order, order)].tolist() for m in matrices],
            hamiltonian=target[np.ix_(order, order)].tolist(),
            directions=[],
            nuclear=[],
            gradient=mf.nuc_grad_method().kernel().tolist(),
        )
        generator = mf.nuc_grad_method().hcore_generator(mol)
        coordinates = mol.atom_coords()
        maximum = 0.0
        for nucleus in range(mol.natm):
            core_derivatives = generator(nucleus)
            for axis in range(3):
                samples = []
                energies = []
                step = 1e-3
                for offset in (-2, -1, 1, 2):
                    point = coordinates.copy()
                    point[nucleus, axis] += offset * step
                    moved = mol.copy()
                    moved.set_geom_(point, unit="Bohr")
                    samples.append(
                        [
                            moved.intor(key)
                            for key in (
                                "int1e_ovlp",
                                "int1e_kin",
                                "int1e_nuc",
                                "int1e_pnucp",
                            )
                        ]
                    )
                    energies.append(float(reference(moved).e_tot))
                derivatives = [
                    (
                        (samples[0][k] - samples[3][k])
                        + 8 * (samples[2][k] - samples[1][k])
                    )
                    / (12 * step)
                    for k in range(4)
                ]
                gradient = (
                    (energies[0] - energies[3]) + 8 * (energies[2] - energies[1])
                ) / (12 * step)
                maximum = max(
                    maximum, abs(gradient - record["gradient"][nucleus][axis])
                )
                record["nuclear"].append(
                    dict(
                        nucleus=nucleus,
                        axis=axis,
                        tangents=[
                            d[np.ix_(order, order)].tolist() for d in derivatives
                        ],
                        core_response=core_derivatives[axis][
                            np.ix_(order, order)
                        ].tolist(),
                        five_point_gradient=gradient,
                    )
                )
        assert maximum < 1e-6, (name, maximum)
        for direction in range(3):
            tangents = []
            for k in range(4):
                a = np.fromfunction(
                    lambda p, q: np.sin(
                        (p + 1) * (q + 2) + 0.37 * (k + 1) * (direction + 1)
                    ),
                    (n, n),
                )
                tangents.append((a + a.T) * 0.003)
            step = 0.01
            samples = [
                nonlinear_core(
                    *[m + offset * step * d for m, d in zip(matrices, tangents)]
                )
                for offset in (-2, -1, 1, 2)
            ]
            derivative = (samples[0] - 8 * samples[1] + 8 * samples[2] - samples[3]) / (
                12 * step
            )
            record["directions"].append(
                dict(
                    tangents=[m[np.ix_(order, order)].tolist() for m in tangents],
                    response=derivative[np.ix_(order, order)].tolist(),
                )
            )
        records.append(record)
        print(
            name,
            "AO",
            n,
            "X2C",
            mf.e_tot,
            "all nuclear",
            len(record["nuclear"]),
            "PySCF analytic/FD",
            maximum,
            flush=True,
        )
    path = Path(__file__).resolve().parents[1] / "data/x2c_core_reference.json"
    path.write_text(
        json.dumps(
            dict(
                source=__doc__,
                pyscf_version=pyscf_version,
                speed_of_light=C,
                cases=records,
            ),
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
