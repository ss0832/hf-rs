"""Independent determinant-space (T0)/(T1) resolvent from PySCF FCI actions.

Triples moments come from H*T2 and H*T1 in the full alpha/beta CI space.
No spatial triples numerator or pair-overlap recurrence is implemented here.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import ao2mo, cc, fci, gto, lib, scf
from pyscf.cc import ccsd_t_slow

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)
folder = Path(__file__).resolve().parents[1] / "data"
local = json.loads((folder / "localization_reference.json").read_text(encoding="utf-8"))
ccref = json.loads((folder / "local_ccsd_reference.json").read_text(encoding="utf-8"))
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
no = mol.nelectron // 2
nm = mol.nao
nv = nm - no
s = mf.get_ovlp()
co = np.array(local["objectives"]["boys"]["coefficients"])
cv = mf.mo_coeff[:, no:]
coeff = np.column_stack((co, cv))
solver = cc.CCSD(mf).density_fit(auxbasis=embedded_basis("BASIS_6_31G_GBS", ["O", "H"]))
solver.conv_tol = 1e-13
solver.conv_tol_normt = 1e-11
solver.max_cycle = 300
solver.kernel()
canonical = float(solver.ccsd_t())
g = ao2mo.restore(1, solver.with_df.ao2mo(coeff), nm)
f = coeff.T @ mf.get_fock() @ coeff
h = f.copy()
for i in range(no):
    h -= 2 * g[:, :, i, i] - g[:, i, i, :]
heff = fci.direct_spin1.absorb_h1e(h, g, nm, (no, no), 0.5)
strings = list(map(int, fci.cistring.gen_strings4orblist(range(nm), no)))
address = {bits: i for i, bits in enumerate(strings)}
nci = len(strings)
ref = (1 << no) - 1
refdet = ref | (ref << nm)
triple = []
holes = []
particles = []
for ia, a in enumerate(strings):
    for ib, b in enumerate(strings):
        ha = [i for i in range(no) if not (a >> i & 1)]
        hb = [i for i in range(no) if not (b >> i & 1)]
        if len(ha) + len(hb) == 3:
            triple.append(ia * nci + ib)
            holes.append((tuple(ha), tuple(hb)))
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
        if not det >> p & 1:
            return None
        sign *= (-1) ** ((det & ((1 << p) - 1)).bit_count())
        det ^= 1 << p
    for p in create:
        if det >> p & 1:
            return None
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


operator1 = operator(f)
diagonal_occ = f.copy()
diagonal_occ[:no, :no] = np.diag(np.diag(f)[:no])
operator0 = operator(diagonal_occ)
assert np.max(abs(operator1 - operator1.T)) < 1e-12


def solve(case, tno_cutoff):
    t1 = co.T @ s @ np.array(case["t1_ao"]) @ s @ cv
    t2 = np.einsum(
        "mi,nj,ra,sb,mnrs->ijab",
        s @ co,
        s @ co,
        s @ cv,
        s @ cv,
        np.array(case["t2_ao"]),
        optimize=True,
    )
    ci1, ci2 = states(t1, t2)
    w = fci.direct_spin1.contract_2e(heff, ci2, nm, (no, no)).ravel()[triple]
    v = fci.direct_spin1.contract_2e(heff, ci1, nm, (no, no)).ravel()[triple]
    densities = {}
    for i in range(no):
        for j in range(i + 1):
            t = t2[i, j]
            tilde = 2 * t - t.T
            densities[i, j] = densities[j, i] = (tilde.T @ t + tilde @ t.T) / (
                2 if i == j else 1
            )
    spaces = {}
    ranks = []
    for ijk in itertools.product(range(no), repeat=3):
        key = tuple(sorted(ijk))
        if key not in spaces:
            i, j, k = key
            den = (densities[i, j] + densities[j, k] + densities[i, k]) / 3
            occ, u = np.linalg.eigh(den)
            u = u[:, np.ones(nv, dtype=bool) if tno_cutoff == 0 else occ > tno_cutoff]
            eps, rotation = np.linalg.eigh(u.T @ f[no:, no:] @ u)
            spaces[key] = u @ rotation
        ranks.append(spaces[key].shape[1])
    columns = []
    for hole in sorted(set(holes)):
        ha, hb = hole
        key = tuple(sorted(ha + hb))
        u = spaces[key]
        rank = u.shape[1]
        indices = [i for i, hole2 in enumerate(holes) if hole2 == hole]
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
    assert np.max(abs(p.T @ p - np.eye(p.shape[1]))) < 1e-12
    wp = p.T @ w
    vp = p.T @ v
    result = dict(
        name=case["name"], cutoff=case["cutoff"], tno_cutoff=tno_cutoff, tno_ranks=ranks
    )
    for label, op in [("t0", operator0), ("t1", operator1)]:
        projected = p.T @ op @ p
        t = np.linalg.solve(projected, -wp)
        result["e_" + label] = float(np.dot(t, wp + vp))
        result["residual_" + label] = float(np.max(abs(projected @ t + wp)))
    return result


cases = [solve(ccref["cases"][0], 0.0), solve(ccref["cases"][1], 1e-5)]
assert abs(cases[0]["e_t1"] - canonical) < 1e-10, (cases[0], canonical)
fulleris = solver.ao2mo(coeff)
t1 = co.T @ s @ np.array(ccref["cases"][0]["t1_ao"]) @ s @ cv
t2 = np.einsum(
    "mi,nj,ra,sb,mnrs->ijab",
    s @ co,
    s @ co,
    s @ cv,
    s @ cv,
    np.array(ccref["cases"][0]["t2_ao"]),
    optimize=True,
)
check = float(ccsd_t_slow.kernel(solver, fulleris, t1, t2))
assert abs(check - cases[0]["e_t0"]) < 1e-10, (check, cases[0])
out = dict(
    source=f"PySCF {pyscf.__version__} determinant CI H*T1/H*T2 moments and dense Fock resolvent; full limits cross-checked with canonical CCSD(T) and noncanonical PySCF slow (T0)",
    basis_source_sha256=basis_hash(),
    canonical_ri_triples=canonical,
    cases=cases,
)
(folder / "local_triples_reference.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8"
)
print(cases)
