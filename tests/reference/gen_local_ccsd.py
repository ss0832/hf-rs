"""Independent spin-adapted PySCF RI-CCSD and finite-PNO Galerkin roots.

Root finding projects PySCF's true spatial CC residuals into explicit dense
amplitude coordinates. This is independent of the Rust spin-orbital iteration.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import cc, gto, lib, scf
from scipy.optimize import root

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)
ref = json.loads(
    Path(__file__)
    .resolve()
    .parents[1]
    .joinpath("data/localization_reference.json")
    .read_text(encoding="utf-8")
)
mol = gto.M(
    atom=[
        ("O", (0.13, -0.21, 0.31)),
        ("H", (0.23, 1.22, -0.77)),
        ("H", (-0.08, -1.64, -0.83)),
    ],
    basis=embedded_basis("STO_3G_GBS", ["O", "H"]),
    unit="Bohr",
    cart=True,
    verbose=0,
)
mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-10)
aux = embedded_basis("BASIS_6_31G_GBS", ["O", "H"])
local = np.array(ref["objectives"]["boys"]["coefficients"])
no = mol.nelectron // 2
nv = mol.nao - no
co = local
cv = mf.mo_coeff[:, no:]
coeff = np.column_stack((co, cv))
solver = cc.CCSD(mf).density_fit(auxbasis=aux)
solver.conv_tol = 1e-13
solver.conv_tol_normt = 1e-11
solver.max_cycle = 300
solver.kernel()
canonical_energy = float(solver.e_corr)
eris = solver.ao2mo(coeff)
g = np.asarray(eris.ovov).reshape(no, nv, no, nv).transpose(0, 2, 1, 3)
e = np.diag(eris.fock)
ev = e[no:]
d1 = e[:no, None] - ev
d2 = d1[:, None, :, None] + d1[None, :, None, :]
pairs = [(i, j) for i in range(no) for j in range(i + 1)]


def solve(cutoff):
    bases = {}
    delta = 0.0
    ranks = []
    for i, j in pairs:
        t = g[i, j] / (e[i] + e[j] - ev[:, None] - ev[None, :])
        tilde = 2 * t - t.T
        density = (tilde.T @ t + tilde @ t.T) / (2 if i == j else 1)
        occupation, p = np.linalg.eigh(density)
        p = p[:, np.ones(nv, dtype=bool) if cutoff == 0 else occupation > cutoff]
        bases[i, j] = bases[j, i] = p
        ranks.append(p.shape[1])
        tp = p @ (p.T @ t @ p) @ p.T
        delta += (1 if i == j else 2) * np.sum(g[i, j] * (2 * (t - tp) - (t - tp).T))
    singles = []
    doubles = []
    for i in range(no):
        p = bases[i, i]
        for a in range(p.shape[1]):
            v = np.zeros((no, nv))
            v[i] = p[:, a]
            singles.append(v.ravel())
    for i, j in pairs:
        p = bases[i, j]
        rank = p.shape[1]
        for a in range(rank):
            for b in range(rank):
                if i == j and b > a:
                    continue
                local = np.zeros((rank, rank))
                scale = np.sqrt(2.0) if i != j or a != b else 1.0
                local[a, b] = 1 / scale
                if i == j:
                    local[b, a] = local[a, b]
                v = np.zeros((no, no, nv, nv))
                v[i, j] = p @ local @ p.T
                if i != j:
                    v[j, i] = v[i, j].T
                doubles.append(v.ravel())
    p1 = np.array(singles).reshape(-1, no * nv).T
    p2 = np.array(doubles).reshape(-1, no * no * nv * nv).T
    n1 = p1.shape[1]
    n2 = p2.shape[1]

    def unpack(x):
        return (p1 @ x[:n1]).reshape(no, nv), (p2 @ x[n1:]).reshape(no, no, nv, nv)

    def residual(x):
        t1, t2 = unpack(x)
        new1, new2 = solver.update_amps(t1, t2, eris)
        return np.concatenate(
            (p1.T @ ((new1 - t1) * d1).ravel(), p2.T @ ((new2 - t2) * d2).ravel())
        )

    fit = root(residual, np.zeros(n1 + n2), method="hybr", options={"xtol": 1e-10})
    max_res = float(np.max(abs(residual(fit.x))))
    assert max_res < 1e-10, (fit.message, max_res)
    t1, t2 = unpack(fit.x)
    energy = float(solver.energy(t1, t2, eris))
    return dict(
        name="full_space" if cutoff == 0 else "finite_pno",
        cutoff=cutoff,
        e_ccsd=energy,
        delta_pno=float(delta),
        residual_max=max_res,
        pair_ranks=ranks,
        t1_ao=(co @ t1 @ cv.T).tolist(),
        t2_ao=np.einsum(
            "mi,nj,ra,sb,ijab->mnrs", co, co, cv, cv, t2, optimize=True
        ).tolist(),
    )


cases = [solve(0.0), solve(1e-4)]
assert abs(cases[0]["e_ccsd"] - canonical_energy) < 1e-10
out = dict(
    source=f"PySCF {pyscf.__version__} DF spin-adapted CCSD residuals and independent SciPy dense Galerkin root",
    basis_source_sha256=basis_hash(),
    cart=True,
    canonical_ri_ccsd=canonical_energy,
    cases=cases,
)
Path(__file__).resolve().parents[1].joinpath(
    "data/local_ccsd_reference.json"
).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print([(c["name"], c["e_ccsd"], c["residual_max"], c["pair_ranks"]) for c in cases])
