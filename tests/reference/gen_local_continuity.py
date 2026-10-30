"""Independent PySCF/dense-Galerkin reference for overlap-transported MP2 spaces."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import df, gto, lib, scf

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)
folder = Path(__file__).resolve().parents[1] / "data"
reference = json.loads(
    (folder / "localization_reference.json").read_text(encoding="utf-8")
)
positions = np.array([[0.13, -0.21, 0.31], [0.23, 1.22, -0.77], [-0.08, -1.64, -0.83]])
symbols = ["O", "H", "H"]
definitions = embedded_basis("STO_3G_GBS", ["O", "H"])
auxdefs = embedded_basis("BASIS_6_31G_GBS", ["O", "H"])
no, nv = 5, 2
cutoff = 1e-4
co0 = np.array(reference["objectives"]["boys"]["coefficients"])


def system(coords):
    mol = gto.M(
        atom=list(zip(symbols, coords.tolist())),
        basis=definitions,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-10)
    return mol, mf


mol0, mf0 = system(positions)
cv0 = mf0.mo_coeff[:, no:]


def polar(m):
    val, u = np.linalg.eigh(m.T @ m)
    assert min(val) > 1e-8
    return m @ ((u / np.sqrt(val)) @ u.T)


def fitted(mol, co, cv):
    aux = df.addons.make_auxmol(mol, auxbasis=auxdefs)
    metric = aux.intor("int2c2e")
    raw = df.incore.aux_e2(mol, aux, aosym="s1")
    val, u = np.linalg.eigh(metric)
    b = np.einsum("mnP,PQ->Qmn", raw, u[:, val > 1e-10] / np.sqrt(val[val > 1e-10]))
    bia = np.einsum("Qmn,mi,na->Qia", b, co, cv)
    return np.einsum("Qia,Qjb->ijab", bia, bia)


k0 = fitted(mol0, co0, cv0)
fo0 = co0.T @ mf0.get_fock() @ co0
ev0 = mf0.mo_energy[no:]
bases = {}
for i in range(no):
    for j in range(i + 1):
        t = k0[i, j] / (fo0[i, i] + fo0[j, j] - ev0[:, None] - ev0[None, :])
        tilde = 2 * t - t.T
        density = (tilde.T @ t + tilde @ t.T) / (2 if i == j else 1)
        val, u = np.linalg.eigh(density)
        bases[i, j] = u[:, val > cutoff]


def energy(coords):
    mol, mf = system(coords)
    current = coords - coords.mean(axis=0)
    target = positions - positions.mean(axis=0)
    u, _, vt = np.linalg.svd(current.T @ target)
    proper = np.eye(3)
    proper[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ proper @ u.T
    aligned = current @ rotation.T + positions.mean(axis=0)
    aligned_mol, _ = system(aligned)
    # Cartesian p_x/p_y/p_z functions rotate as ordinary vectors. This model
    # contains only s and p functions; the Rust implementation handles all L.
    transform = np.eye(mol.nao)
    transform[2:5, 2:5] = rotation
    cross = transform.T @ gto.intor_cross("int1e_ovlp", aligned_mol, mol0)
    canonical_occ = mf.mo_coeff[:, :no]
    cv = mf.mo_coeff[:, no:]
    co = canonical_occ @ polar(canonical_occ.T @ cross @ co0)
    transport = cv.T @ cross @ cv0
    fo = co.T @ mf.get_fock() @ co
    ev = mf.mo_energy[no:]
    fv = np.diag(ev)
    kk = fitted(mol, co, cv)
    columns = []
    delta = 0.0
    for i in range(no):
        for j in range(i + 1):
            u = (
                polar(transport @ bases[i, j])
                if bases[i, j].shape[1]
                else np.zeros((nv, 0))
            )
            eps, rotation = np.linalg.eigh(u.T @ fv @ u)
            u = u @ rotation
            initial = kk[i, j] / (fo[i, i] + fo[j, j] - ev[:, None] - ev[None, :])
            projected = u @ (u.T @ initial @ u) @ u.T
            delta += (1 if i == j else 2) * float(
                np.sum(kk[i, j] * (2 * (initial - projected) - (initial - projected).T))
            )
            rank = u.shape[1]
            for a in range(rank):
                for b in range(rank):
                    if i == j and b > a:
                        continue
                    local = np.zeros((rank, rank))
                    scale = np.sqrt(2.0) if i != j or a != b else 1.0
                    local[a, b] = 1 / scale
                    if i == j:
                        local[b, a] = local[a, b]
                    col = np.zeros_like(kk)
                    col[i, j] = u @ local @ u.T
                    if i != j:
                        col[j, i] = col[i, j].T
                    columns.append(col.ravel())

    def action(t):
        return (
            np.einsum("ac,ijcb->ijab", fv, t)
            + np.einsum("bc,ijac->ijab", fv, t)
            - np.einsum("ik,kjab->ijab", fo, t)
            - np.einsum("kj,ikab->ijab", fo, t)
        )

    p = np.array(columns).T
    ap = np.array([action(col.reshape(kk.shape)).ravel() for col in p.T]).T
    x = np.linalg.solve(p.T @ ap, -p.T @ kk.ravel())
    t = (p @ x).reshape(kk.shape)
    return float(np.sum((2 * kk + action(t)) * (2 * t - t.swapaxes(2, 3))) + delta)


snapshots = [
    dict(
        coordinate=None,
        displacement=0.0,
        positions=positions.tolist(),
        e_corr=energy(positions),
    )
]
for coordinate in range(9):
    for displacement in [-2e-3, 2e-3]:
        moved = positions.copy()
        moved.flat[coordinate] += displacement
        snapshots.append(
            dict(
                coordinate=coordinate,
                displacement=displacement,
                positions=moved.tolist(),
                e_corr=energy(moved),
            )
        )
angle = 0.37
rotation = np.array(
    [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
)
rigid = positions @ rotation.T + np.array([0.2, -0.4, 0.1])
snapshots.append(
    dict(
        coordinate=None,
        displacement=0.0,
        rigid=True,
        positions=rigid.tolist(),
        e_corr=energy(rigid),
    )
)
assert abs(snapshots[-1]["e_corr"] - snapshots[0]["e_corr"]) < 1e-11
out = dict(
    source=f"PySCF {pyscf.__version__} independent SCFs/Kabsch rigid alignment and Cartesian p rotation; NumPy polar transport and dense Galerkin amplitudes",
    basis_source_sha256=basis_hash(),
    cutoff=cutoff,
    pair_ranks=[u.shape[1] for u in bases.values()],
    snapshots=snapshots,
)
(folder / "local_continuity_reference.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8"
)
print(
    "snapshots",
    len(snapshots),
    "ranks",
    out["pair_ranks"],
    "base",
    snapshots[0]["e_corr"],
)
