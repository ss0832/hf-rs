# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent spin-free DKH2 oracle: PySCF S/T/V/pVp integrals and an element-wise
NumPy construction in the generalized kinetic eigenbasis.

h_DKH2 = (E - c^2) + E1 + E2 with E1 = A (V + K W K) A, O1 = A (K W / p - V K p) A,
W1_ij = O1_ij / (E_i + E_j) and E2 = (W1 O1^T + O1 W1^T) / 2, written with explicit
indices (no matrix-function machinery shared with the Rust implementation).
Energies use PySCF RHF with the replaced core Hamiltonian; nuclear derivatives are
five-point differences of those energies. Contracted Cartesian AO space, point nuclei.
"""

import json
from pathlib import Path

import numpy as np
from gen_ghf import rust_ao_permutation
from pyscf import __version__ as pyscf_version
from pyscf import gto, lib, scf
from scipy.linalg import eigh

from embedded_basis import embedded_basis

lib.num_threads(4)
C = 137.035999084
KEYS = ("int1e_ovlp", "int1e_kin", "int1e_nuc", "int1e_pnucp")


def dkh2(s, t, v, w, c=C):
    tv, u = eigh(t, s)
    vp = u.T @ v @ u
    wp = u.T @ w @ u
    p = np.sqrt(2.0 * tv)
    e = c * np.sqrt(p * p + c * c)
    a = np.sqrt((e + c * c) / (2.0 * e))
    k = c / (e + c * c)
    n = len(tv)
    e1 = np.empty((n, n))
    odd = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            e1[i, j] = a[i] * a[j] * (vp[i, j] + k[i] * k[j] * wp[i, j])
            odd[i, j] = a[i] * a[j] * (k[i] * wp[i, j] / p[j] - k[j] * p[j] * vp[i, j])
    w1 = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            w1[i, j] = odd[i, j] / (e[i] + e[j])
    e2 = 0.5 * (w1 @ odd.T + odd @ w1.T)
    hp = np.diag(p * p * c * c / (e + c * c)) + e1 + e2
    return s @ u @ hp @ u.T @ s


def models():
    return [
        ("Ne", [("Ne", [0.1, -0.2, 0.3])], "STO_3G_GBS", True),
        ("Ar", [("Ar", [0.1, -0.2, 0.3])], "STO_3G_GBS", True),
        (
            "water",
            [
                ("O", [0.1, -0.2, 0.3]),
                ("H", [1.53, 0.68, 0.3]),
                ("H", [-1.33, 0.68, 0.3]),
            ],
            "BASIS_6_31G_GBS",
            True,
        ),
        (
            "HCl",
            [("Cl", [0.1, -0.2, 0.3]), ("H", [0.1, -0.2, 2.7])],
            "BASIS_6_31G_GBS",
            True,
        ),
    ]


def energy(mol):
    mats = [mol.intor(k) for k in KEYS]
    h = dkh2(*mats)
    nonrel = scf.RHF(mol)
    nonrel.conv_tol = 1e-13
    nonrel.conv_tol_grad = 1e-11
    nonrel.max_cycle = 300
    nonrel.chkfile = None
    nonrel.kernel()
    assert nonrel.converged
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h
    # 1e-13 Eh is below the double-precision resolution of a 2,750 Eh total energy (Kr)
    mf.conv_tol = max(1e-13, 4e-15 * abs(nonrel.e_tot))
    mf.conv_tol_grad = 1e-11
    mf.max_cycle = 300
    mf.chkfile = None
    mf.kernel(dm0=nonrel.make_rdm1())
    if not mf.converged:  # second-order polishing of the same stationary point
        newton = scf.newton(mf)
        newton.conv_tol = mf.conv_tol
        newton.kernel(mf.mo_coeff, mf.mo_occ)
        mf = newton
    assert mf.converged
    return float(mf.e_tot), float(nonrel.e_tot), mats, h


def main():
    records = []
    for name, atoms, constant, matrices_ok in models():
        mol = gto.M(
            atom=atoms,
            basis=embedded_basis(constant, sorted({a[0] for a in atoms})),
            cart=True,
            unit="Bohr",
            verbose=0,
        )
        e_tot, e_nr, mats, h = energy(mol)
        record = dict(
            name=name,
            atoms=[dict(symbol=s, position=r) for s, r in atoms],
            basis_constant=constant,
            e_total=e_tot,
            e_nonrelativistic=e_nr,
        )
        # c -> infinity limit and c-scaling of the one-electron Hamiltonian
        record["hcore_norm_large_c_minus_t_plus_v"] = float(
            np.max(abs(dkh2(*mats, c=1e6) - (mats[1] + mats[2])))
        )
        if matrices_ok:
            order = rust_ao_permutation(mol, constant)
            sel = np.ix_(order, order)
            record["matrices"] = [m[sel].tolist() for m in mats]
            record["hamiltonian"] = h[sel].tolist()
            n = len(h)
            record["directions"] = []
            for direction in range(3):
                tangents = []
                for k in range(4):
                    x = np.fromfunction(
                        lambda p, q: np.sin(
                            (p + 1) * (q + 2) + 0.37 * (k + 1) * (direction + 1)
                        ),
                        (n, n),
                    )
                    tangents.append((x + x.T) * 0.003)
                step = 0.01
                samples = [
                    dkh2(*[m + off * step * d for m, d in zip(mats, tangents)])
                    for off in (-2, -1, 1, 2)
                ]
                deriv = (samples[0] - 8 * samples[1] + 8 * samples[2] - samples[3]) / (
                    12 * step
                )
                record["directions"].append(
                    dict(
                        tangents=[m[sel].tolist() for m in tangents],
                        response=deriv[sel].tolist(),
                    )
                )
            gradient = []
            coords = mol.atom_coords()
            step = 1e-3
            for nucleus in range(mol.natm):
                row = []
                for axis in range(3):
                    es = []
                    for off in (-2, -1, 1, 2):
                        point = coords.copy()
                        point[nucleus, axis] += off * step
                        moved = mol.copy()
                        moved.set_geom_(point, unit="Bohr")
                        es.append(energy(moved)[0])
                    row.append(((es[0] - es[3]) + 8 * (es[2] - es[1])) / (12 * step))
                gradient.append(row)
            record["five_point_gradient"] = gradient
        records.append(record)
        print(name, "DKH2", e_tot, "shift", e_tot - e_nr, flush=True)
    path = Path(__file__).resolve().parents[1] / "data/dkh2_reference.json"
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
