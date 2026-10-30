"""Independent dense Kronecker/Galerkin triples Fock-resolvent reference."""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(90218)
no, nv = 2, 3
u = expm(np.array([[0.0, 0.31], [-0.31, 0.0]]))
fo = u.T @ np.diag([-0.9, -0.37]) @ u
fv = np.diag([0.21, 0.53, 0.82])


def symmetric_source():
    raw = rng.normal(size=(no, no, no, nv, nv, nv)) * 0.02
    out = np.zeros_like(raw)
    for p in itertools.permutations(range(3)):
        out += raw.transpose(p + tuple(i + 3 for i in p)) / 6
    return out


w = symmetric_source()
v = symmetric_source()


def action(t):
    return (
        np.einsum("ad,ijkdbc->ijkabc", fv, t)
        + np.einsum("bd,ijkadc->ijkabc", fv, t)
        + np.einsum("cd,ijkabd->ijkabc", fv, t)
        - np.einsum("il,ljkabc->ijkabc", fo, t)
        - np.einsum("lj,ilkabc->ijkabc", fo, t)
        - np.einsum("lk,ijlabc->ijkabc", fo, t)
    )


def energy(blocks, ts):
    total = 0.0
    for b, t in zip(blocks, ts):
        n = len(b["energies"])
        t = t.reshape((n,) * 3)
        tilde = (
            4 * t
            - 2 * t.transpose(0, 2, 1)
            - 2 * t.transpose(1, 0, 2)
            - 2 * t.transpose(2, 1, 0)
            + t.transpose(1, 2, 0)
            + t.transpose(2, 0, 1)
        )
        total += (
            np.sum(tilde * (np.array(b["w"]) + np.array(b["v"])).reshape((n,) * 3)) / 3
        )
    return float(total)


def case(name, ranks):
    spaces = {}
    for ijk in itertools.combinations_with_replacement(range(no), 3):
        q, _ = np.linalg.qr(rng.normal(size=(nv, nv)))
        q = q[:, : ranks[len(spaces)]]
        eps, u = np.linalg.eigh(q.T @ fv @ q)
        spaces[ijk] = (q @ u, eps)
    blocks = []
    columns = []
    t0 = []
    for ijk in itertools.product(range(no), repeat=3):
        q, eps = spaces[tuple(sorted(ijk))]
        ww = np.einsum("pa,qb,rc,pqr->abc", q, q, q, w[ijk]).ravel()
        vv = np.einsum("pa,qb,rc,pqr->abc", q, q, q, v[ijk]).ravel()
        diagonal = (
            eps[:, None, None]
            + eps[None, :, None]
            + eps[None, None, :]
            - sum(fo[i, i] for i in ijk)
        ).ravel()
        t0.append(-ww / diagonal)
        blocks.append(
            dict(
                i=ijk[0],
                j=ijk[1],
                k=ijk[2],
                basis=q.tolist(),
                energies=eps.tolist(),
                w=ww.tolist(),
                v=vv.tolist(),
            )
        )
        embed = np.kron(np.kron(q, q), q)
        for column in embed.T:
            full = np.zeros_like(w)
            full[ijk] = column.reshape((nv,) * 3)
            columns.append(full.ravel())
    p = np.array(columns).T
    projected = p.T @ np.array([action(col.reshape(w.shape)).ravel() for col in p.T]).T
    source = np.concatenate([np.array(b["w"]) for b in blocks])
    x = np.linalg.solve(projected, -source)
    boundaries = np.cumsum([0] + [len(b["w"]) for b in blocks])
    t1 = [x[start:end] for start, end in zip(boundaries[:-1], boundaries[1:])]
    return dict(
        name=name,
        f_occ=fo.tolist(),
        blocks=blocks,
        t0=[t.tolist() for t in t0],
        t1=[t.tolist() for t in t1],
        e_t0=energy(blocks, t0),
        e_t1=energy(blocks, t1),
    )


source = rng.normal(size=(3, 3, 3))
overlap = rng.normal(size=(2, 3))
out = dict(
    source=f"NumPy {np.__version__} dense Kronecker three-leg embedding and full Fock resolvent",
    cases=[case("full", [3] * 4), case("finite", [3, 2, 2, 2])],
    transform=dict(
        cube=source.ravel().tolist(),
        overlap=overlap.tolist(),
        expected=np.einsum("ap,bq,cr,pqr->abc", overlap, overlap, overlap, source)
        .ravel()
        .tolist(),
    ),
)
Path(__file__).resolve().parents[1].joinpath(
    "data/local_triples_space_reference.json"
).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print([(c["name"], c["e_t0"], c["e_t1"]) for c in out["cases"]])
