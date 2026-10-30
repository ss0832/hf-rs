# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Exact spin-projected UHF oracle in the full CI space.

The (broken-symmetry) PySCF UHF determinant is expanded in the determinant basis of its alpha
molecular orbitals (beta-string coefficients are minors of C_a^T S C_b), projected onto
S = |S_z| with the Lowdin product projector built from the FCI S^2 operator, and its projected
energy <P Phi|H|P Phi>/<P Phi|P Phi> is evaluated with the FCI Hamiltonian. This is independent of
the spin-rotation quadrature used by hf-rs. Contracted Cartesian AO space, embedded basis data.
"""

import json
from pathlib import Path

import numpy as np
from pyscf import __version__ as pyscf_version
from pyscf import ao2mo, fci, gto, lib, scf
from pyscf.fci import cistring

from embedded_basis import embedded_basis

lib.num_threads(4)

MODELS = [
    (
        "H2_stretched",
        [("H", [0.0, 0.0, -2.0]), ("H", [0.0, 0.0, 2.0])],
        "STO_3G_GBS",
        0,
        0,
    ),
    (
        "LiH_stretched",
        [("Li", [0.0, 0.0, 0.0]), ("H", [0.1, 0.2, 5.5])],
        "STO_3G_GBS",
        0,
        0,
    ),
    (
        "H4_rectangle",
        [
            ("H", [0.0, 0.0, 0.0]),
            ("H", [3.2, 0.0, 0.0]),
            ("H", [0.0, 2.1, 0.0]),
            ("H", [3.2, 2.1, 0.1]),
        ],
        "STO_3G_GBS",
        0,
        0,
    ),
    ("OH", [("O", [0.0, 0.0, 0.0]), ("H", [0.1, -0.2, 1.84])], "STO_3G_GBS", 0, 1),
    (
        "CH2_triplet",
        [("C", [0.0, 0.0, 0.0]), ("H", [1.6, 0.0, 1.0]), ("H", [-1.6, 0.0, 1.0])],
        "STO_3G_GBS",
        0,
        2,
    ),
]


def uhf(mol, broken):
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 500
    if broken:
        rhf = scf.RHF(mol).run(conv_tol=1e-12)
        nocc = mol.nelectron // 2
        ca, cb = rhf.mo_coeff.copy(), rhf.mo_coeff.copy()
        t = np.pi / 4
        h, lu = rhf.mo_coeff[:, nocc - 1].copy(), rhf.mo_coeff[:, nocc].copy()
        ca[:, nocc - 1] = np.cos(t) * h + np.sin(t) * lu
        cb[:, nocc - 1] = np.cos(t) * h - np.sin(t) * lu
        occ = np.zeros(mol.nao)
        occ[:nocc] = 1
        dm = (
            np.einsum("pi,qi->pq", ca * occ, ca),
            np.einsum("pi,qi->pq", cb * occ, cb),
        )
        mf.kernel(dm0=dm)
    else:
        mf.kernel()
    if not mf.converged:
        mf = scf.newton(mf).run(mf.mo_coeff, mf.mo_occ, conv_tol=1e-12)
    assert mf.converged
    if broken:
        # A broken-symmetry start may converge to a saddle point (twisted ethylene does,
        # depending on rounding in degenerate frontier orbitals): follow internal instabilities.
        for _ in range(8):
            mo, _, stable, _ = mf.stability(return_status=True)
            if stable:
                break
            mf.kernel(dm0=mf.make_rdm1(mo, mf.mo_occ))
            assert mf.converged
        assert stable
    return mf


def projected(mf, mol):
    na, nb = mol.nelec
    ca, cb = mf.mo_coeff
    n = ca.shape[1]
    u = ca.T @ mol.intor("int1e_ovlp") @ cb
    strings_b = cistring.make_strings(range(n), nb)
    ia = cistring.str2addr(n, na, (1 << na) - 1)
    ci = np.zeros((cistring.num_strings(n, na), len(strings_b)))
    for ib, sb in enumerate(strings_b):
        occ = [k for k in range(n) if (int(sb) >> k) & 1]
        ci[ia, ib] = np.linalg.det(u[occ, :nb])
    assert abs(np.linalg.norm(ci) - 1) < 1e-10
    h1 = ca.T @ mf.get_hcore() @ ca
    eri = ao2mo.restore(1, ao2mo.full(mol, ca), n)
    h2 = fci.direct_spin1.absorb_h1e(h1, eri, n, (na, nb), 0.5)

    def energy(vec):
        return float(
            vec.ravel() @ fci.direct_spin1.contract_2e(h2, vec, n, (na, nb)).ravel()
        )

    s = 0.5 * (na - nb)
    smax = 0.5 * (na + nb)
    pc = ci.copy()
    k = s + 1
    while k <= smax + 1e-9:
        ss = fci.spin_op.contract_ss(pc, n, (na, nb))
        pc = (ss - k * (k + 1) * pc) / (s * (s + 1) - k * (k + 1))
        k += 1
    weight = float(ci.ravel() @ pc.ravel())
    e_proj = (
        float(ci.ravel() @ fci.direct_spin1.contract_2e(h2, pc, n, (na, nb)).ravel())
        / weight
    )
    return energy(ci) + mol.energy_nuc(), e_proj + mol.energy_nuc(), weight, s


def rotation_projected(mf, mol, nodes):
    """Independent dense NumPy kernel for the larger legacy ethylene case.

    A full CI space is impractical here. Unlike the five small FCI-space oracles,
    this additional check uses a separate implementation of spin rotation.
    """
    ca, cb = mf.mo_coeff
    na, nb = mol.nelec
    n = ca.shape[1]
    beta = ca.T @ mol.intor("int1e_ovlp") @ cb[:, :nb]
    h = ca.T @ mf.get_hcore() @ ca
    eri = ao2mo.restore(1, ao2mo.full(mol, ca), n)
    ref = np.zeros((2 * n, na + nb))
    ref[:na, :na] = np.eye(na)
    ref[n:, na:] = beta
    numerator, denominator = 0.0, 0.0
    for x, w in zip(*np.polynomial.legendre.leggauss(nodes)):
        c, s = np.sqrt((1 + x) / 2), np.sqrt((1 - x) / 2)
        rot = np.vstack((c * ref[:n] - s * ref[n:], s * ref[:n] + c * ref[n:]))
        overlap = ref.T @ rot
        density = (rot @ np.linalg.solve(overlap, ref.T)).reshape(2, n, 2, n)
        total = density[0, :, 0, :] + density[1, :, 1, :]
        e = np.einsum("pq,qp", h, total)
        e += 0.5 * np.einsum("prqs,rp,sq", eri, total, total, optimize=True)
        e -= 0.5 * np.einsum("prqs,asbp,braq", eri, density, density, optimize=True)
        weight = w * c ** (na - nb) * np.linalg.det(overlap)
        numerator += weight * (e + mol.energy_nuc())
        denominator += weight
    return float(numerator / denominator), float((na - nb + 1) * denominator / 2)


def main():
    out = []
    for name, atoms, constant, charge, spin in MODELS:
        kwargs = dict(
            atom=atoms,
            basis=embedded_basis(constant, sorted({a[0] for a in atoms})),
            cart=True,
            unit="Bohr",
            verbose=0,
            charge=charge,
        )
        mol = gto.M(spin=spin, **kwargs)
        mf = uhf(mol, broken=spin == 0)
        e_det, e_proj, weight, s = projected(mf, mol)
        assert abs(e_det - mf.e_tot) < 1e-9, (name, e_det, mf.e_tot)
        high = uhf(gto.M(spin=spin + 2, **kwargs), broken=False)
        out.append(
            dict(
                name=name,
                basis_constant=constant,
                atoms=[dict(symbol=a, position=r) for a, r in atoms],
                charge=charge,
                spin_2s=spin,
                e_uhf=float(mf.e_tot),
                s2_uhf=float(mf.spin_square()[0]),
                e_projected=e_proj,
                target_weight=weight,
                e_high_spin=float(high.e_tot),
                s2_high_spin=float(high.spin_square()[0]),
            )
        )
        print(name, mf.e_tot, mf.spin_square()[0], e_proj, weight, flush=True)
    ethylene = [
        ("C", [0.0, 0.0, 1.266]),
        ("C", [0.0, 0.0, -1.266]),
        ("H", [0.0, 1.744, 2.336]),
        ("H", [0.0, -1.744, 2.336]),
        ("H", [1.744, 0.0, -2.336]),
        ("H", [-1.744, 0.0, -2.336]),
    ]
    mol = gto.M(
        atom=ethylene,
        basis=embedded_basis("BASIS_6_31G_GBS", ["C", "H"]),
        cart=True,
        unit="Bohr",
        verbose=0,
    )
    mf = uhf(mol, broken=True)
    e, weight = rotation_projected(mf, mol, 16)
    refined = rotation_projected(mf, mol, 24)
    assert abs(e - refined[0]) < 1e-10 and abs(weight - refined[1]) < 1e-12
    large = dict(
        name="C2H4_twisted",
        method="Independent NumPy spin-rotation quadrature, 16/24 nodes",
        e_uhf=float(mf.e_tot),
        s2_uhf=float(mf.spin_square()[0]),
        e_projected=e,
        target_weight=weight,
    )
    print(large, flush=True)
    path = Path(__file__).resolve().parents[1] / "data/puhf_reference.json"
    path.write_text(
        json.dumps(
            dict(
                source=__doc__, pyscf_version=pyscf_version, cases=out, large_case=large
            ),
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
