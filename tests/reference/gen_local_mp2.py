"""Independent molecular local-fit/domain LMP2 dense Galerkin reference."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import df, dft, gto, lib, lo, scf
from scipy.linalg import expm
from scipy.optimize import least_squares, minimize

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)
atoms = [("He", (0.0, 0.0, 0.0)), ("He", (0.2, 0.3, 5.0))]
basis = embedded_basis("BASIS_6_31G_GBS", ["He"])
mol = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=True, verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-10)
no = mol.nelectron // 2
nv = mol.nao - no
s = mf.get_ovlp()
c = mf.mo_coeff[:, :no]
v = mf.mo_coeff[:, no:]
r = mol.intor("int1e_r", comp=3)
loc = lo.Boys(mol, c)
loc.init_guess = None
loc.conv_tol = 1e-13
loc.conv_tol_grad = 1e-10
localized = loc.kernel()


def rotate(x):
    return localized @ expm(np.array([[0.0, x[0]], [-x[0], 0.0]]))


def gradient(x):
    d = np.asarray([rotate(x).T @ op @ rotate(x) for op in r])
    return [4 * np.sum((d[:, 0, 0] - d[:, 1, 1]) * d[:, 0, 1])]


def objective(x):
    return sum(np.sum(np.diag(rotate(x).T @ op @ rotate(x)) ** 2) for op in r)


candidates = []
for start in [0.3, 0.7, -0.4]:
    fit = minimize(lambda x: -objective(x), [start], method="BFGS")
    refined = least_squares(gradient, fit.x, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    candidates.append(refined)
fit = max(candidates, key=lambda fit: objective(fit.x))
c = rotate(fit.x)
fo = c.T @ mf.get_fock() @ c
ev = mf.mo_energy[no:]
fv = np.diag(ev)
q = v @ v.T @ s
norm = np.sqrt(np.diag(q.T @ s @ q))
pao = q / norm
# Independent Lebedev DOI evaluation, streamed in bounded AO blocks.
grid = dft.gen_grid.Grids(mol)
grid.atom_grid = (150, 974)
grid.prune = None
grid.radii_adjust = None
grid.build()
doi = np.zeros((no, mol.nao))
for start in range(0, len(grid.weights), 2048):
    ao = mol.eval_gto("GTOval_cart", grid.coords[start : start + 2048])
    doi += ((ao @ c) ** 2).T @ (
        grid.weights[start : start + 2048, None] * (ao @ pao) ** 2
    )
doi = np.sqrt(doi)
owner = np.array([0, 0, 1, 1])
mkn = np.zeros((no, 2))
for i in range(no):
    pop = c[:, i, None] * c[None, :, i] * s
    diag = np.diag(pop)
    for mu in range(mol.nao):
        for nu in range(mol.nao):
            if diag[mu] + diag[nu] > 0:
                mkn[i, owner[mu]] += 2 * pop[mu, nu] * diag[mu] / (diag[mu] + diag[nu])
virtual_domains = [sorted(set(owner[np.flatnonzero(doi[i] > 0.1)])) for i in range(no)]
fit_domains = [np.flatnonzero(abs(mkn[i]) > 0.1).tolist() for i in range(no)]
fullk = np.zeros((no, no, nv, nv))
blocks = []
delta = 0.0
fixed = np.zeros_like(fullk)
columns = []
for i in range(no):
    for j in range(i + 1):
        domain = sorted(set(virtual_domains[i] + virtual_domains[j]))
        fitdomain = sorted(set(fit_domains[i] + fit_domains[j]))
        selected = np.flatnonzero(np.isin(owner, domain))
        qa = q[:, selected]
        val, w = np.linalg.eigh(qa.T @ s @ qa)
        p = qa @ w[:, val > 1e-10] / np.sqrt(val[val > 1e-10])
        vp = v.T @ s @ p
        aux = gto.M(
            atom=[atoms[a] for a in fitdomain],
            basis=basis,
            unit="Bohr",
            cart=True,
            verbose=0,
        )
        metric = aux.intor("int2c2e")
        three = df.incore.aux_e2(mol, aux, aosym="s1")
        val, w = np.linalg.eigh(metric)
        b = np.einsum(
            "mnP,PQ->Qmn", three, w[:, val > 1e-10] / np.sqrt(val[val > 1e-10])
        )
        bl = np.einsum("Qmn,mi,na->Qia", b, c, v)
        kk = bl[:, i, :].T @ bl[:, j, :]
        fullk[i, j] = kk
        fullk[j, i] = kk.T
        kp = vp.T @ kk @ vp
        fp = vp.T @ fv @ vp
        eps, u = np.linalg.eigh(fp)
        kc = u.T @ kp @ u
        t = kc / (fo[i, i] + fo[j, j] - eps[:, None] - eps[None, :])
        tilde = 2 * t - t.T
        den = (tilde.T @ t + tilde @ t.T) / (2 if i == j else 1)
        occ, pno = np.linalg.eigh(den)
        estimate = float(np.sum(kc * (2 * t - t.T)))
        strong = abs(estimate) >= 1e-3
        keep = np.ones(len(occ), dtype=bool) if not strong else occ > 1e-5
        pn = pno[:, keep]
        projected = pn @ (pn.T @ t @ pn) @ pn.T
        delta += (1 if i == j else 2) * float(
            np.sum(kc * (2 * (t - projected) - (t - projected).T))
        )
        uu = vp @ u @ pn
        initial = pn.T @ t @ pn
        rank = uu.shape[1]
        blocks.append(
            dict(
                i=i,
                j=j,
                domain=[int(a) for a in domain],
                fit_domain=fitdomain,
                rank=rank,
                strong=strong,
            )
        )
        if not strong:
            fixed[i, j] = uu @ initial @ uu.T
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
                col = np.zeros_like(fullk)
                col[i, j] = uu @ local @ uu.T
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
ap = np.array([action(col.reshape(fullk.shape)).ravel() for col in p.T]).T
x = np.linalg.solve(p.T @ ap, -p.T @ (fullk + action(fixed)).ravel())
t = (p @ x).reshape(fullk.shape) + fixed
strong = weak = 0.0
for block in blocks:
    i, j = block["i"], block["j"]
    energy = (1 if i == j else 2) * float(
        np.sum((2 * fullk + action(t))[i, j] * (2 * t[i, j] - t[i, j].T))
    )
    if block["strong"]:
        strong += energy
    else:
        weak += energy
out = dict(
    source=f"PySCF {pyscf.__version__} AO integrals, NumPy independent local Coulomb fits and dense full-tensor Galerkin solve",
    basis_source_sha256=basis_hash(),
    cart=True,
    e_strong=strong,
    e_weak=weak,
    delta_pno=delta,
    e_corr=strong + weak + delta,
    blocks=blocks,
)
Path(__file__).resolve().parents[1].joinpath(
    "data/local_mp2_reference.json"
).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(out)
