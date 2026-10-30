# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Fixed transported TNOs: determinant CI moments and dense Fock resolvents."""

import itertools
import json

import numpy as np
import pyscf
from gen_local_ccsd_continuity import folder, frame, no, nv, polar
from pyscf import ao2mo, fci

reference = json.loads((folder / "local_ccsd_continuity_reference.json").read_text())
nm = no + nv
tno_cutoff = 1e-5
strings = list(map(int, fci.cistring.gen_strings4orblist(range(nm), no)))
address = {bits: i for i, bits in enumerate(strings)}
nci = len(strings)
ref = (1 << no) - 1
refdet = ref | (ref << nm)
triple, holes, particles = [], [], []
for ia, a in enumerate(strings):
    for ib, b in enumerate(strings):
        ha = tuple(i for i in range(no) if not (a >> i & 1))
        hb = tuple(i for i in range(no) if not (b >> i & 1))
        if len(ha) + len(hb) == 3:
            triple.append(ia * nci + ib)
            holes.append((ha, hb))
            particles.append(
                (
                    [p - no for p in range(no, nm) if a >> p & 1],
                    [p - no for p in range(no, nm) if b >> p & 1],
                )
            )
triple = np.array(triple)


def apply(det, annihilate, create):
    sign = 1
    for p in annihilate:
        assert det >> p & 1
        sign *= (-1) ** ((det & ((1 << p) - 1)).bit_count())
        det ^= 1 << p
    for p in create:
        assert not det >> p & 1
        sign *= (-1) ** ((det & ((1 << p) - 1)).bit_count())
        det |= 1 << p
    return det, sign


def address_of(det):
    return address[det & ((1 << nm) - 1)] * nci + address[det >> nm]


def states(t1, t2):
    ci1 = np.zeros(nci * nci)
    ci2 = np.zeros_like(ci1)
    for i in range(no):
        for a in range(nv):
            for spin in range(2):
                det, sign = apply(refdet, [i + spin * nm], [no + a + spin * nm])
                ci1[address_of(det)] += sign * t1[i, a]
    for i, j in itertools.combinations(range(2 * no), 2):
        for a, b in itertools.combinations(range(2 * nv), 2):
            val = (
                t2[i // 2, j // 2, a // 2, b // 2]
                if i % 2 == a % 2 and j % 2 == b % 2
                else 0.0
            ) - (
                t2[i // 2, j // 2, b // 2, a // 2]
                if i % 2 == b % 2 and j % 2 == a % 2
                else 0.0
            )
            if val == 0:
                continue
            det, sign = apply(
                refdet,
                [i // 2 + (i % 2) * nm, j // 2 + (j % 2) * nm],
                [no + b // 2 + (b % 2) * nm, no + a // 2 + (a % 2) * nm],
            )
            ci2[address_of(det)] += sign * val
    return ci1.reshape(nci, nci), ci2.reshape(nci, nci)


def operator(fock):
    columns = []
    e0 = 2 * np.trace(fock[:no, :no])
    for col in triple:
        unit = np.zeros(nci * nci)
        unit[col] = 1
        action = (
            fci.direct_spin1.contract_1e(
                fock, unit.reshape(nci, nci), nm, (no, no)
            ).ravel()
            - e0 * unit
        )
        columns.append(action[triple])
    return np.array(columns).T


original_spaces = None
cv0 = None
snapshots = []
for case in reference["snapshots"]:
    coords = np.array(case["positions"])
    mol, mf, co, cv, transport, solver, eris = frame(coords)
    s = mf.get_ovlp()
    t1 = co.T @ s @ np.array(case["t1_ao"]).reshape(nm, nm) @ s @ cv
    t2 = np.einsum(
        "mi,nj,ra,sb,mnrs->ijab",
        s @ co,
        s @ co,
        s @ cv,
        s @ cv,
        np.array(case["t2_ao"]).reshape((nm,) * 4),
        optimize=True,
    )
    if original_spaces is None:
        densities = {}
        for i in range(no):
            for j in range(i + 1):
                t = t2[i, j]
                tilde = 2 * t - t.T
                densities[i, j] = densities[j, i] = (tilde.T @ t + tilde @ t.T) / (
                    2 if i == j else 1
                )
        original_spaces = {}
        for key in itertools.combinations_with_replacement(range(no), 3):
            i, j, k = key
            density = (densities[i, j] + densities[j, k] + densities[i, k]) / 3
            values, vectors = np.linalg.eigh(density)
            original_spaces[key] = vectors[:, values > tno_cutoff]
        ranks = [
            original_spaces[tuple(sorted(key))].shape[1]
            for key in itertools.product(range(no), repeat=3)
        ]
    spaces = {key: polar(transport @ basis) for key, basis in original_spaces.items()}
    g = ao2mo.restore(1, solver.with_df.ao2mo(np.column_stack((co, cv))), nm)
    f = np.array(eris.fock)
    h = f.copy()
    for i in range(no):
        h -= 2 * g[:, :, i, i] - g[:, i, i, :]
    heff = fci.direct_spin1.absorb_h1e(h, g, nm, (no, no), 0.5)
    ci1, ci2 = states(t1, t2)
    w = fci.direct_spin1.contract_2e(heff, ci2, nm, (no, no)).ravel()[triple]
    v = fci.direct_spin1.contract_2e(heff, ci1, nm, (no, no)).ravel()[triple]
    columns = []
    for hole in sorted(set(holes)):
        ha, hb = hole
        u = spaces[tuple(sorted(ha + hb))]
        rank = u.shape[1]
        indices = [i for i, other in enumerate(holes) if other == hole]
        for a in itertools.combinations(range(rank), len(ha)):
            for b in itertools.combinations(range(rank), len(hb)):
                col = np.zeros(len(triple))
                for index in indices:
                    pa, pb = particles[index]
                    col[index] = np.linalg.det(u[np.ix_(pa, a)]) * np.linalg.det(
                        u[np.ix_(pb, b)]
                    )
                columns.append(col)
    p = np.array(columns).reshape(-1, len(triple)).T
    assert p.shape[1] > 0
    assert np.max(abs(p.T @ p - np.eye(p.shape[1]))) < 1e-12
    wp, vp = p.T @ w, p.T @ v
    f0 = f.copy()
    f0[:no, :no] = np.diag(np.diag(f)[:no])
    data = {key: value for key, value in case.items() if key not in ("t1_ao", "t2_ao")}
    for label, fock in [("t0", f0), ("t1", f)]:
        op = p.T @ operator(fock) @ p
        amplitude = np.linalg.solve(op, -wp)
        maximum = float(np.max(abs(op @ amplitude + wp)))
        assert maximum < 1e-12
        data["e_" + label] = float(np.dot(amplitude, wp + vp))
        data["residual_" + label] = maximum
    snapshots.append(data)
    print(len(snapshots), data["e_t0"], data["e_t1"], flush=True)
for key in ("e_t0", "e_t1"):
    assert abs(snapshots[-1][key] - snapshots[0][key]) < 1e-11
out = dict(
    source=f"PySCF {pyscf.__version__} determinant-space H*T1/H*T2 moments and dense Fock resolvents; fixed TNO subspaces with independent Kabsch/polar transport",
    cutoff=reference["cutoff"],
    tno_cutoff=tno_cutoff,
    tno_ranks=ranks,
    snapshots=snapshots,
)
(folder / "local_triples_continuity_reference.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8"
)
