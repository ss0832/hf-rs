# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Cholesky-decomposition HF oracle: an independent dense NumPy pivoted Cholesky of the full
PySCF AO ERI supermatrix (hf-rs AO order, packed pair index p(p+1)/2+q for p >= q, pivot =
largest residual diagonal; values within 1e-12 relative of it are ties resolved by the
smallest index), and PySCF RHF on the reconstructed integrals.
Low-symmetry geometries avoid exact diagonal ties. Contracted Cartesian s/p bases.
"""

import json
from pathlib import Path

import numpy as np
from gen_ghf import rust_ao_permutation
from pyscf import __version__ as pyscf_version
from pyscf import ao2mo, gto, lib, scf

from embedded_basis import embedded_basis

lib.num_threads(4)
THRESHOLDS = [1e-2, 1e-4, 1e-6]
MODELS = [
    ("LiH", [("Li", [0.0, 0.0, 0.0]), ("H", [0.13, 0.21, 3.0])], "STO_3G_GBS"),
    (
        "water",
        [("O", [0.1, -0.2, 0.3]), ("H", [1.53, 0.68, 0.41]), ("H", [-1.33, 0.71, 0.3])],
        "BASIS_6_31G_GBS",
    ),
    ("HCl", [("Cl", [0.1, -0.2, 0.3]), ("H", [0.17, -0.11, 2.71])], "BASIS_6_31G_GBS"),
]


def pivoted_cholesky(m, threshold):
    d = np.diag(m).copy()
    vectors, pivots = [], []
    while True:
        largest = float(np.max(d))
        k = int(np.flatnonzero(d >= largest - 1e-12 * abs(largest))[0])
        if d[k] <= threshold or len(vectors) == len(d):
            return np.array(vectors), pivots, float(max(d[k], 0.0))
        col = m[:, k] - sum(v * v[k] for v in vectors) if vectors else m[:, k].copy()
        v = col / np.sqrt(d[k])
        v[k] = np.sqrt(d[k])
        d = np.maximum(d - v * v, 0.0)
        vectors.append(v)
        pivots.append(k)


def main():
    out = []
    for name, atoms, constant in MODELS:
        mol = gto.M(
            atom=atoms,
            basis=embedded_basis(constant, sorted({a[0] for a in atoms})),
            cart=True,
            unit="Bohr",
            verbose=0,
        )
        n = mol.nao
        order = rust_ao_permutation(mol, constant)
        eri = mol.intor("int2e")[np.ix_(order, order, order, order)]
        pairs = [(p, q) for p in range(n) for q in range(p + 1)]
        idx_p = np.array([p for p, _ in pairs])
        idx_q = np.array([q for _, q in pairs])
        supermatrix = eri[
            idx_p[:, None], idx_q[:, None], idx_p[None, :], idx_q[None, :]
        ]
        mf0 = scf.RHF(mol)
        mf0.conv_tol = 1e-13
        mf0.kernel()
        record = dict(
            name=name,
            basis_constant=constant,
            atoms=[dict(symbol=s, position=r) for s, r in atoms],
            e_rhf=float(mf0.e_tot),
            thresholds=[],
        )
        inverse = np.argsort(order)
        for threshold in THRESHOLDS:
            vectors, pivots, residual = pivoted_cholesky(supermatrix, threshold)
            approx = np.zeros((n, n, n, n))
            for v in vectors:
                full = np.zeros((n, n))
                full[idx_p, idx_q] = v
                full[idx_q, idx_p] = v
                approx += np.einsum("pq,rs->pqrs", full, full)
            approx_pyscf = approx[np.ix_(inverse, inverse, inverse, inverse)]
            mf = scf.RHF(mol)
            mf._eri = ao2mo.restore(8, approx_pyscf, n)
            mf.conv_tol = 1e-12
            mf.max_cycle = 500
            mf.kernel(dm0=mf0.make_rdm1())
            if not mf.converged:  # second-order polishing of the same stationary point
                newton = scf.newton(mf)
                newton.conv_tol = 1e-12
                newton.kernel(mf.mo_coeff, mf.mo_occ)
                mf = newton
            assert mf.converged, (name, threshold)
            record["thresholds"].append(
                dict(
                    threshold=threshold,
                    n_vectors=len(vectors),
                    pivots=[int(k) for k in pivots],
                    max_residual=residual,
                    max_integral_error=float(np.max(abs(approx - eri))),
                    e_cd_hf=float(mf.e_tot),
                )
            )
        out.append(record)
        print(
            name,
            [
                (t["threshold"], t["n_vectors"], t["e_cd_hf"] - record["e_rhf"])
                for t in record["thresholds"]
            ],
            flush=True,
        )
    path = Path(__file__).resolve().parents[1] / "data/cd_hf_reference.json"
    path.write_text(
        json.dumps(
            dict(source=__doc__, pyscf_version=pyscf_version, cases=out), indent=1
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
