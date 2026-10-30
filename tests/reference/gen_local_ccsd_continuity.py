# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent PySCF spatial CCSD roots in fixed transported pair subspaces."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import cc, df, gto, lib, scf
from scipy.optimize import root

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)
folder = Path(__file__).resolve().parents[1] / "data"
localization = json.loads((folder / "localization_reference.json").read_text())
geometry = json.loads((folder / "local_continuity_reference.json").read_text())
definitions = embedded_basis("STO_3G_GBS", ["O", "H"])
auxdefs = embedded_basis("BASIS_6_31G_GBS", ["O", "H"])
positions = np.array(geometry["snapshots"][0]["positions"])
co0 = np.array(localization["objectives"]["boys"]["coefficients"])
no, nv = 5, 2
cutoff = geometry["cutoff"]
pairs = [(i, j) for i in range(no) for j in range(i + 1)]


def system(coords):
    mol = gto.M(
        atom=list(zip(["O", "H", "H"], coords.tolist())),
        basis=definitions,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-10)
    assert mf.converged
    assert np.max(abs(mf.get_grad(mf.mo_coeff, mf.mo_occ))) < 1e-9
    return mol, mf


def polar(m):
    values, vectors = np.linalg.eigh(m.T @ m)
    assert np.min(values, initial=1.0) > 1e-8
    return m @ ((vectors / np.sqrt(values)) @ vectors.T)


mol0, mf0 = system(positions)
cv0 = mf0.mo_coeff[:, no:]
auxmol = df.addons.make_auxmol(mol0, auxbasis=auxdefs)
metric = auxmol.intor("int2c2e")
raw = df.incore.aux_e2(mol0, auxmol, aosym="s1")
values, vectors = np.linalg.eigh(metric)
b = np.einsum(
    "mnP,PQ->Qmn", raw, vectors[:, values > 1e-10] / np.sqrt(values[values > 1e-10])
)
bia = np.einsum("Qmn,mi,na->Qia", b, co0, cv0)
k0 = np.einsum("Qia,Qjb->ijab", bia, bia)
fo0 = co0.T @ mf0.get_fock() @ co0
ev0 = mf0.mo_energy[no:]
bases = {}
for i, j in pairs:
    t = k0[i, j] / (fo0[i, i] + fo0[j, j] - ev0[:, None] - ev0[None, :])
    tilde = 2 * t - t.T
    density = (tilde.T @ t + tilde @ t.T) / (2 if i == j else 1)
    occupation, vectors = np.linalg.eigh(density)
    bases[i, j] = vectors[:, occupation > cutoff]


def frame(coords):
    mol, mf = system(coords)
    current = coords - coords.mean(axis=0)
    target = positions - positions.mean(axis=0)
    left, _, right = np.linalg.svd(current.T @ target)
    proper = np.eye(3)
    proper[2, 2] = np.linalg.det(right.T @ left.T)
    rigid = right.T @ proper @ left.T
    aligned = current @ rigid.T + positions.mean(axis=0)
    aligned_mol, _ = system(aligned)
    transform = np.eye(mol.nao)
    transform[2:5, 2:5] = rigid
    cross = transform.T @ gto.intor_cross("int1e_ovlp", aligned_mol, mol0)
    co = mf.mo_coeff[:, :no] @ polar(mf.mo_coeff[:, :no].T @ cross @ co0)
    cv = mf.mo_coeff[:, no:]
    transport = cv.T @ cross @ cv0
    solver = cc.CCSD(mf).density_fit(auxbasis=auxdefs)
    eris = solver.ao2mo(np.column_stack((co, cv)))
    return mol, mf, co, cv, transport, solver, eris


def solve(coords):
    mol, mf, co, cv, transport, solver, eris = frame(coords)
    g = np.asarray(eris.ovov).reshape(no, nv, no, nv).transpose(0, 2, 1, 3)
    e = np.diag(eris.fock)
    d1 = e[:no, None] - e[no:]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    projected = {key: polar(transport @ basis) for key, basis in bases.items()}
    delta = 0.0
    singles, doubles = [], []
    for i in range(no):
        for column in projected[i, i].T:
            v = np.zeros((no, nv))
            v[i] = column
            singles.append(v.ravel())
    for i, j in pairs:
        p = projected[i, j]
        t = g[i, j] / d2[i, j]
        tp = p @ (p.T @ t @ p) @ p.T
        delta += (1 if i == j else 2) * np.sum(g[i, j] * (2 * (t - tp) - (t - tp).T))
        rank = p.shape[1]
        for a in range(rank):
            for b in range(rank):
                if i == j and b > a:
                    continue
                local = np.zeros((rank, rank))
                local[a, b] = 1 / (np.sqrt(2.0) if i != j or a != b else 1.0)
                if i == j:
                    local[b, a] = local[a, b]
                v = np.zeros((no, no, nv, nv))
                v[i, j] = p @ local @ p.T
                if i != j:
                    v[j, i] = v[i, j].T
                doubles.append(v.ravel())
    p1 = np.array(singles).reshape(-1, no * nv).T
    p2 = np.array(doubles).reshape(-1, no * no * nv * nv).T
    n1, n2 = p1.shape[1], p2.shape[1]

    def unpack(x):
        return (p1 @ x[:n1]).reshape(no, nv), (p2 @ x[n1:]).reshape(no, no, nv, nv)

    def residual(x):
        t1, t2 = unpack(x)
        new1, new2 = solver.update_amps(t1, t2, eris)
        return np.concatenate(
            (p1.T @ ((new1 - t1) * d1).ravel(), p2.T @ ((new2 - t2) * d2).ravel())
        )

    fit = root(residual, np.zeros(n1 + n2), method="hybr", options={"xtol": 1e-10})
    maximum = float(np.max(abs(residual(fit.x))))
    assert maximum < 1e-11, (fit.message, maximum)
    t1, t2 = unpack(fit.x)
    return dict(
        e_ccsd=float(solver.energy(t1, t2, eris)),
        delta_pno=float(delta),
        residual_max=maximum,
        t1_ao=(co @ t1 @ cv.T).ravel().tolist(),
        t2_ao=np.einsum("mi,nj,ra,sb,ijab->mnrs", co, co, cv, cv, t2, optimize=True)
        .ravel()
        .tolist(),
    )


def main():
    snapshots = []
    for case in geometry["snapshots"]:
        data = {
            key: value
            for key, value in case.items()
            if key in ("coordinate", "displacement", "positions", "rigid")
        }
        data.update(solve(np.array(case["positions"])))
        snapshots.append(data)
        print(len(snapshots), data["e_ccsd"], data["residual_max"], flush=True)
    assert abs(snapshots[-1]["e_ccsd"] - snapshots[0]["e_ccsd"]) < 1e-11
    out = dict(
        source=f"PySCF {pyscf.__version__} spin-adapted RI-CCSD residuals; independent SciPy dense Galerkin roots, SVD/Kabsch and polar transport",
        basis_source_sha256=basis_hash(),
        cutoff=cutoff,
        pair_ranks=[basis.shape[1] for basis in bases.values()],
        snapshots=snapshots,
    )
    (folder / "local_ccsd_continuity_reference.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
