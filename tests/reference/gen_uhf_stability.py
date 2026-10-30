# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Dense real UHF -> UHF orbital Hessians from PySCF MO integrals.

(A + B)_{ai s, bj t} = d_st (d_ij F_ab - d_ab F_ij) + 2 (ai|bj) - d_st [(ab|ij) + (aj|bi)]
is assembled explicitly with ao2mo for the stationary UHF determinants of the PUHF oracle
(broken-symmetry singlets after instability following, OH doublet, CH2 triplet), the stable
broken-symmetry twisted-ethylene solution, and the closed-shell (RHF) twisted-ethylene point,
whose lowest eigenvalue is its triplet instability. hf-rs forms Hessian-vector products
AO-direct and solves for the lowest eigenpair iteratively; no implementation source is read.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from gen_puhf import MODELS, uhf
from pyscf import ao2mo, gto, lib, scf

from embedded_basis import embedded_basis

lib.num_threads(4)

ETHYLENE = [
    ("C", [0.0, 0.0, 1.266]),
    ("C", [0.0, 0.0, -1.266]),
    ("H", [0.0, 1.744, 2.336]),
    ("H", [0.0, -1.744, 2.336]),
    ("H", [1.744, 0.0, -2.336]),
    ("H", [-1.744, 0.0, -2.336]),
]


def hessian(mf):
    mol = mf.mol
    fock = mf.get_fock()
    spins = []
    for c, n_occ, f in zip(mf.mo_coeff, mol.nelec, fock):
        o, v = c[:, :n_occ], c[:, n_occ:]
        spins.append((o, v, o.T @ f @ o, v.T @ f @ v))
    sizes = [v.shape[1] * o.shape[1] for o, v, _, _ in spins]
    offsets = [0, sizes[0]]
    h = np.zeros((sum(sizes), sum(sizes)))
    for s, (o_s, v_s, foo, fvv) in enumerate(spins):
        nv, no = v_s.shape[1], o_s.shape[1]
        rows = slice(offsets[s], offsets[s] + sizes[s])
        h[rows, rows] += np.kron(fvv, np.eye(no)) - np.kron(np.eye(nv), foo)
        for t, (o_t, v_t, _, _) in enumerate(spins):
            cols = slice(offsets[t], offsets[t] + sizes[t])
            mv, mo = v_t.shape[1], o_t.shape[1]
            vovo = ao2mo.general(mol, (v_s, o_s, v_t, o_t), compact=False)
            vovo = vovo.reshape(nv, no, mv, mo)
            block = 2 * vovo
            if s == t:
                vvoo = ao2mo.general(mol, (v_s, v_s, o_s, o_s), compact=False)
                block = block - vvoo.reshape(nv, nv, no, no).transpose(0, 2, 1, 3)
                block = block - vovo.transpose(0, 3, 2, 1)
            h[rows, cols] += block.reshape(nv * no, mv * mo)
    assert np.max(np.abs(h - h.T)) < 1e-10
    return np.linalg.eigvalsh(h)


def record(name, mf, constant, atoms, charge, spin, broken):
    values = hessian(mf)
    print(name, mf.e_tot, values[:3], flush=True)
    return dict(
        name=name,
        basis_constant=constant,
        atoms=[dict(symbol=a, position=r) for a, r in atoms],
        charge=charge,
        spin_2s=spin,
        broken_symmetry=broken,
        e_uhf=float(mf.e_tot),
        s2_uhf=float(mf.spin_square()[0]),
        lowest_eigenvalues=values[:4].tolist(),
    )


def main():
    out = []
    for name, atoms, constant, charge, spin in MODELS:
        mol = gto.M(
            atom=atoms,
            basis=embedded_basis(constant, sorted({a[0] for a in atoms})),
            cart=True,
            unit="Bohr",
            verbose=0,
            charge=charge,
            spin=spin,
        )
        mf = uhf(mol, broken=spin == 0)
        out.append(record(name, mf, constant, atoms, charge, spin, spin == 0))
    mol = gto.M(
        atom=ETHYLENE,
        basis=embedded_basis("BASIS_6_31G_GBS", ["C", "H"]),
        cart=True,
        unit="Bohr",
        verbose=0,
    )
    out.append(
        record(
            "C2H4_twisted_broken_symmetry",
            uhf(mol, broken=True),
            "BASIS_6_31G_GBS",
            ETHYLENE,
            0,
            0,
            True,
        )
    )
    rhf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-10)
    closed = scf.UHF(mol)
    closed.conv_tol, closed.conv_tol_grad = 1e-13, 1e-10
    closed.kernel(dm0=(rhf.make_rdm1() / 2, rhf.make_rdm1() / 2))
    assert (
        abs(closed.e_tot - rhf.e_tot) < 1e-10 and abs(closed.spin_square()[0]) < 1e-10
    )
    out.append(
        record(
            "C2H4_twisted_closed_shell",
            closed,
            "BASIS_6_31G_GBS",
            ETHYLENE,
            0,
            0,
            False,
        )
    )
    path = Path(__file__).resolve().parents[1] / "data/uhf_stability_reference.json"
    path.write_text(
        json.dumps(
            dict(source=__doc__, pyscf_version=pyscf.__version__, cases=out), indent=1
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
