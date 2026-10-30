# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent full pair-tensor and orbital-projector CCSD-F12x contractions.

The doubles source is the constant plus linear coefficient of L(lambda) V
L(lambda)^T; symmetric polynomial interpolation removes the quadratic term.
Spin/cusp contractions retain four occupied indices until the final einsum.
"""

import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(503122)
cases = []
for no, nv in [(1, 2), (2, 3), (3, 4)]:
    nm = no + nv
    v = rng.normal(size=(no, no, nm, nm)) * 0.08
    v = (v + v.transpose(1, 0, 3, 2)) / 2
    c = rng.normal(size=(no, no, nv, nv)) * 0.03
    c = (c + c.transpose(1, 0, 3, 2)) / 2
    t1 = rng.normal(size=(no, nv)) * 0.07
    t2 = rng.normal(size=(no, no, nv, nv)) * 0.05
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    cusp = np.zeros((no, no, no, no))
    for i in range(no):
        for j in range(no):
            cusp[i, j, i, j] += 0.375
            cusp[i, j, j, i] += 0.125
    spin_cusp = 2 * cusp - cusp.transpose(0, 1, 3, 2)
    singles = np.einsum(
        "mnak,ikmn->ia", v[:, :, no:, :no], 2 * cusp - cusp.transpose(1, 0, 2, 3)
    )

    def projected(scale):
        left = np.hstack((-scale * t1.T, np.eye(nv)))
        return np.einsum("ap,mnpq,bq,ijmn->ijab", left, v, left, cusp, optimize=True)

    doubles = (
        projected(0)
        + (projected(1) - projected(-1)) / 2
        + np.einsum("mnab,ijmn->ijab", c, cusp)
    )
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
    fock_energy = np.einsum("ijmn,mnab,ijab->", spin_cusp, c, tau)
    b_energy = (
        np.einsum("ijmn,mnab,ijab->", spin_cusp, v[:, :, no:, no:], tau)
        + np.einsum("ijmn,mnib,jb->", spin_cusp, v[:, :, :no, no:], t1)
        + np.einsum("ijmn,mnaj,ia->", spin_cusp, v[:, :, no:, :no], t1)
    )
    cases.append(
        dict(
            n_occ=no,
            n_virt=nv,
            v=v.reshape(no**2, nm, nm).tolist(),
            c=c.reshape(no**2, nv, nv).tolist(),
            t1=t1.ravel().tolist(),
            t2=t2.ravel().tolist(),
            singles=singles.ravel().tolist(),
            doubles=doubles.ravel().tolist(),
            fock_energy=float(fock_energy),
            additional_f12b=float(b_energy),
        )
    )
Path(__file__).resolve().parents[1].joinpath(
    "data/ccsd_f12_coupling_reference.json"
).write_text(
    json.dumps(
        dict(
            source=f"NumPy {np.__version__}, full four-index occupied cusp tensors and explicit orbital projectors",
            cases=cases,
        ),
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
print(
    "CCSD-F12 coupling references:",
    [(x["n_occ"], x["fock_energy"], x["additional_f12b"]) for x in cases],
)
