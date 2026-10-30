"""Independent dense NumPy Galerkin local-MP2/PNO references.

The oracle embeds pair coordinates into a full ordered occupied/virtual tensor
and forms P^T A P. It does not implement a pair-overlap iterative solver.
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

rng = np.random.default_rng(61712)
no, nv = 3, 4
x = rng.normal(size=(no, no)) * 0.35
u = expm(x - x.T)
focc = u.T @ np.diag([-1.12, -0.67, -0.41]) @ u
fvirt = np.diag([0.18, 0.39, 0.72, 1.03])
b = rng.normal(size=(7, no, nv)) * 0.11
k = np.einsum("Qia,Qjb->ijab", b, b)


def action(t):
    return (
        np.einsum("ac,ijcb->ijab", fvirt, t)
        + np.einsum("bc,ijac->ijab", fvirt, t)
        - np.einsum("ik,kjab->ijab", focc, t)
        - np.einsum("kj,ikab->ijab", focc, t)
    )


def energy(t):
    return float(np.sum((2 * k + action(t)) * (2 * t - t.swapaxes(2, 3))))


def solve_case(name, ranks, weak):
    blocks = []
    columns = []
    fixed = np.zeros_like(k)
    index = 0
    for i in range(no):
        for j in range(i + 1):
            rank = ranks[index]
            index += 1
            q, _ = np.linalg.qr(rng.normal(size=(nv, nv)))
            q = q[:, :rank]
            eps, v = np.linalg.eigh(q.T @ fvirt @ q)
            q = q @ v
            kp = q.T @ k[i, j] @ q
            initial = -kp / (eps[:, None] + eps[None, :] - focc[i, i] - focc[j, j])
            strong = (i, j) not in weak
            blocks.append(
                dict(
                    i=i,
                    j=j,
                    basis=q.tolist(),
                    energies=eps.tolist(),
                    exchange=kp.tolist(),
                    initial=initial.tolist(),
                    strong=strong,
                    delta_pno=0.0,
                )
            )
            if not strong:
                fixed[i, j] = q @ initial @ q.T
                fixed[j, i] = fixed[i, j].T
                continue
            for a in range(rank):
                for bb in range(rank):
                    if i == j and bb > a:
                        continue
                    local = np.zeros((rank, rank))
                    scale = np.sqrt(2.0) if i != j or a != bb else 1.0
                    local[a, bb] = 1 / scale
                    if i == j:
                        local[bb, a] = local[a, bb]
                    col = np.zeros_like(k)
                    col[i, j] = q @ local @ q.T
                    if i != j:
                        col[j, i] = col[i, j].T
                    columns.append(col.ravel())
    p = np.array(columns).T
    assert np.max(abs(p.T @ p - np.eye(p.shape[1]))) < 1e-12
    a = np.array([action(col.reshape(k.shape)).ravel() for col in p.T]).T
    operator = p.T @ a
    assert np.max(abs(operator - operator.T)) < 1e-12
    x = np.linalg.solve(operator, -p.T @ (k + action(fixed)).ravel())
    t = (p @ x).reshape(k.shape) + fixed
    residual = p.T @ (k + action(t)).ravel()
    amplitudes = [
        (
            np.array(block["basis"]).T
            @ t[block["i"], block["j"]]
            @ np.array(block["basis"])
        ).tolist()
        for block in blocks
    ]
    split = [0.0, 0.0]
    for block in blocks:
        i, j = block["i"], block["j"]
        split[not block["strong"]] += (1 if i == j else 2) * np.sum(
            (2 * k + action(t))[i, j] * (2 * t[i, j] - t[i, j].T)
        )
    return dict(
        name=name,
        f_occ=focc.tolist(),
        pairs=blocks,
        amplitudes=amplitudes,
        energy=energy(t),
        e_strong=float(split[0]),
        e_weak=float(split[1]),
        residual_max=float(max(abs(residual))),
    )


cases = [
    solve_case("full_spaces", [4] * 6, []),
    solve_case("finite_domains", [4, 3, 2, 4, 1, 3], []),
    solve_case("fixed_weak", [4, 3, 2, 4, 1, 3], [(2, 0)]),
]
# The full-space energy must equal canonical MP2 after an occupied rotation.
bcanonical = np.einsum("ij,Qja->Qia", u, b)
kcanonical = np.einsum("Qia,Qjb->ijab", bcanonical, bcanonical)
occ = np.array([-1.12, -0.67, -0.41])
v = np.diag(fvirt)
tcanonical = kcanonical / (
    occ[:, None, None, None]
    + occ[None, :, None, None]
    - v[None, None, :, None]
    - v[None, None, None, :]
)
canonical = float(np.sum(kcanonical * (2 * tcanonical - tcanonical.swapaxes(2, 3))))
assert abs(cases[0]["energy"] - canonical) < 1e-13

pnos = []
for diagonal in [False, True]:
    kk = k[1, 1] if diagonal else k[2, 1]
    z = rng.normal(size=(nv, nv)) * 0.2
    rotation = expm(z - z.T)
    f = rotation.T @ fvirt @ rotation
    eps, w = np.linalg.eigh(f)
    kc = w.T @ kk @ w
    t = kc / (-1.1 - eps[:, None] - eps[None, :])
    tilde = 2 * t - t.T
    d = (tilde.T @ t + tilde @ t.T) / (2 if diagonal else 1)
    occupations, vec = np.linalg.eigh(d)
    for cutoff in [0.0, float((occupations[1] + occupations[2]) / 2), 1.0]:
        retain = np.ones(nv, dtype=bool) if cutoff == 0 else occupations > cutoff
        p = vec[:, retain]
        projector = w @ p @ p.T @ w.T
        truncated = p @ (p.T @ t @ p) @ p.T

        def pair_energy(amplitudes):
            return float(np.sum(kc * (2 * amplitudes - amplitudes.T)))

        pnos.append(
            dict(
                k=kk.tolist(),
                f=f.tolist(),
                occupied_sum=-1.1,
                diagonal=diagonal,
                cutoff=cutoff,
                density=(w @ d @ w.T).tolist(),
                projector=projector.tolist(),
                occupations=occupations[retain][::-1].tolist(),
                initial_ao=(w @ truncated @ w.T).tolist(),
                delta_pno=pair_energy(t) - pair_energy(truncated),
                estimate=pair_energy(t),
            )
        )
out = dict(
    source=f"NumPy {np.__version__} dense full-tensor Kronecker Fock operator, Galerkin solve and independent eigensystem PNO reference",
    canonical_mp2=canonical,
    cases=cases,
    pnos=pnos,
)
Path(__file__).resolve().parents[1].joinpath(
    "data/local_pair_reference.json"
).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print({case["name"]: case["energy"] for case in cases})
