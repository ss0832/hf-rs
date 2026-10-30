# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""TDHF/RPA oracle: complete singlet and triplet spectra from the full non-Hermitian
RPA eigenproblem [[A, B], [-B, -A]] built from PySCF RHF orbitals and MO integrals,
cross-checked against PySCF's Davidson TDHF for the lowest roots (energies,
length-gauge oscillator strengths). Contracted Cartesian AO space, embedded basis data.
Degenerate roots are compared through gauge-invariant group sums and subspace projectors.
"""

import json
import os
import tempfile
from pathlib import Path

import basis_set_exchange as bse
import numpy as np
from pyscf import __version__ as pyscf_version
from pyscf import ao2mo, gto, lib, scf, tdscf

from embedded_basis import embedded_basis

lib.num_threads(4)

MODELS = [
    ("LiH", [("Li", [0.0, 0.0, 0.0]), ("H", [0.1, 0.2, 3.0])], "STO_3G_GBS", None),
    (
        "water",
        [("O", [0.1, -0.2, 0.3]), ("H", [1.53, 0.68, 0.3]), ("H", [-1.33, 0.68, 0.3])],
        "BASIS_6_31G_GBS",
        None,
    ),
    (
        "formaldehyde",
        [
            ("C", [0.0, 0.0, 0.0]),
            ("O", [0.0, 0.0, 2.279]),
            ("H", [0.0, 1.776, -1.111]),
            ("H", [0.0, -1.776, -1.111]),
        ],
        "STO_3G_GBS",
        None,
    ),
    (
        "HI",
        [("H", [0.0, 0.0, 0.0]), ("I", [0.2, -0.1, 3.05])],
        "BASIS_DEF2_SVP_GBS",
        "def2-svp",
    ),
]


def rpa_spectrum(a, b):
    """Positive RPA roots from the (A+B)-based Hermitian reduction
    (A+B)^{1/2} (A-B) (A+B)^{1/2} Z = w^2 Z, X+Y = w^{1/2} (A+B)^{-1/2} Z, so that
    (X+Y).(X-Y) = 1. (hf-rs uses the complementary (A-B)-based reduction.)"""
    vals, vecs = np.linalg.eigh(a + b)
    if np.min(vals) <= 0 or np.min(np.linalg.eigvalsh(a - b)) <= 0:
        return None
    root = (vecs * np.sqrt(vals)) @ vecs.T
    inv_root = (vecs / np.sqrt(vals)) @ vecs.T
    w2, z = np.linalg.eigh(root @ (a - b) @ root)
    if np.min(w2) <= 0:
        return None
    w = np.sqrt(w2)
    plus = (inv_root @ z * np.sqrt(w)).T
    return w, plus


def main():
    out = []
    for name, atoms, constant, ecp in MODELS:
        symbols = sorted({a[0] for a in atoms})
        kwargs = {}
        if ecp:
            text = bse.get_basis(ecp, fmt="nwchem", header=False)
            with tempfile.NamedTemporaryFile("w", suffix=".nw", delete=False) as handle:
                handle.write(text)
                path = handle.name
            kwargs["ecp"] = {
                s: gto.basis.load_ecp(path, s)
                for s in symbols
                if gto.basis.load_ecp(path, s)
            }
            os.unlink(path)
        mol = gto.M(
            atom=atoms,
            basis=embedded_basis(constant, symbols),
            cart=True,
            unit="Bohr",
            verbose=0,
            **kwargs,
        )
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-10
        mf.max_cycle = 300
        mf.kernel()
        assert mf.converged
        nocc = mol.nelectron // 2
        c = mf.mo_coeff
        e = mf.mo_energy
        nmo = c.shape[1]
        nvir = nmo - nocc
        eri = ao2mo.restore(1, ao2mo.full(mol, c), nmo)
        o, v = slice(0, nocc), slice(nocc, nmo)
        d = (e[v][None, :] - e[o][:, None]).ravel()
        iajb = eri[o, v, o, v].reshape(nocc * nvir, nocc * nvir)
        ijab = eri[o, o, v, v].transpose(0, 2, 1, 3).reshape(nocc * nvir, nocc * nvir)
        ibja = eri[o, v, o, v].transpose(0, 3, 2, 1).reshape(nocc * nvir, nocc * nvir)
        dip_ao = mol.intor("int1e_r")
        dip_ov = np.einsum("xpq,pi,qa->xia", dip_ao, c[:, o], c[:, v]).reshape(3, -1)
        record = dict(
            name=name,
            basis_constant=constant,
            ecp=ecp,
            e_rhf=float(mf.e_tot),
            atoms=[dict(symbol=s, position=r) for s, r in atoms],
        )
        for label, coulomb in (("singlet", 2.0), ("triplet", 0.0)):
            a = np.diag(d) + coulomb * iajb - ijab
            b = coulomb * iajb - ibja
            spectrum = rpa_spectrum(a, b)
            if spectrum is None:
                # imaginary RPA frequency: confirm with PySCF's independent stability analysis
                internal, external = mf.stability(external=True, return_status=True)[2:]
                assert (not internal) if coulomb else (not external), (name, label)
                record[label] = dict(
                    unstable=True,
                    min_a_plus_b=float(np.min(np.linalg.eigvalsh(a + b))),
                    min_a_minus_b=float(np.min(np.linalg.eigvalsh(a - b))),
                )
                continue
            roots, plus = spectrum
            f = []
            for w, p in zip(roots, plus):
                mu = np.sqrt(2.0) * dip_ov @ p if coulomb else np.zeros(3)
                f.append(float(2.0 / 3.0 * w * mu @ mu))
            record[label] = dict(
                unstable=False,
                energies=roots.tolist(),
                oscillator_strengths=f,
                x_plus_y=plus.tolist(),
            )
            td = tdscf.TDHF(mf)
            td.singlet = coulomb != 0.0
            # Davidson may miss individual roots in a small subspace; request extra roots and
            # cross-check the lowest five against the complete spectrum.
            td.nstates = min(12, len(roots))
            td.conv_tol = 1e-11
            td.max_cycle = 300
            td.kernel()
            k = min(5, len(roots))
            checked = bool(all(td.converged[:k]))
            if checked:
                assert np.max(abs(td.e[:k] - roots[:k])) < 1e-8, (
                    name,
                    label,
                    td.e,
                    roots[:k],
                )
                if coulomb:
                    fs = td.oscillator_strength(gauge="length")[:k]
                    assert np.max(abs(np.array(fs) - np.array(f[:k]))) < 1e-8, (
                        name,
                        fs,
                        f,
                    )
            record[label]["davidson_crosscheck"] = checked
        out.append(record)
        print(
            name,
            "n_ov",
            nocc * nvir,
            {
                k: record[k].get("energies", ["unstable"])[0]
                for k in ("singlet", "triplet")
            },
            flush=True,
        )
    path = Path(__file__).resolve().parents[1] / "data/tdhf_reference.json"
    path.write_text(
        json.dumps(dict(source=__doc__, pyscf_version=pyscf_version, cases=out)) + "\n"
    )


if __name__ == "__main__":
    main()
