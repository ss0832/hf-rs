# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Dense independent projected Coulomb/F12 matrices: Psi4 integrals + PySCF MOs.

Run in the existing PySCF Python; --psi4-python points to an isolated Psi4 Python.
The two installed runtimes exchange a small npz file, without new dependencies.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--integrals-only", action="store_true")
parser.add_argument(
    "--psi4-python",
    default=os.environ.get("HF_RS_PSI4_PYTHON"),
    help="Python executable with Psi4 (or set HF_RS_PSI4_PYTHON)",
)
args = parser.parse_args()
if not args.integrals_only and not args.psi4_python:
    parser.error("Provide --psi4-python or set HF_RS_PSI4_PYTHON")
root = Path(__file__).resolve().parents[2]
work = root / "target/ccsd-f12-projected-reference"
work.mkdir(parents=True, exist_ok=True)
os.chdir(work)
positions = [[-0.1, 0.2, -1.3], [0.3, -0.2, 1.4]]
if "HF_RS_F12_REFERENCE_POSITIONS" in os.environ:
    positions = json.loads(os.environ["HF_RS_F12_REFERENCE_POSITIONS"])
reference_output = Path(
    os.environ.get(
        "HF_RS_F12_REFERENCE_OUTPUT",
        root / "tests/data/ccsd_f12_projected_reference.json",
    )
)
terms = [(0.3, -0.4), (1.2, -0.2)]
obs = "He 0\nS 1 1.0\n1.4 1.0\nS 1 1.0\n0.45 1.0\n****\n"
aux = "He 0\nS 1 1.0\n0.2 1.0\nS 1 1.0\n0.9 1.0\nS 1 1.0\n3.2 1.0\nP 1 1.0\n0.8 1.0\n****\n"

if args.integrals_only:
    import psi4

    psi4.set_num_threads(4)
    psi4.set_memory("512 MiB")
    psi4.core.set_output_file(str(work / "mints.out"), False)
    (work / "coupling_union.gbs").write_text(
        "cartesian\n****\n" + obs.replace("****\n", "") + aux.replace("He 0\n", ""),
        encoding="utf-8",
    )
    mol = psi4.geometry(
        "0 1\n"
        + "\n".join("He " + " ".join(map(str, p)) for p in positions)
        + "\nunits bohr\nsymmetry c1\nno_com\nno_reorient"
    )
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", "coupling_union", puream=0)
    mints = psi4.core.MintsHelper(basis)
    n = basis.nbf()
    assert n == 8 * len(positions)
    # Psi4 groups shells by atom; production/PySCF concatenate OBS then raw aux.
    permutation = [
        8 * atom + shell for atom in range(len(positions)) for shell in [0, 1]
    ]
    permutation += [
        8 * atom + shell for atom in range(len(positions)) for shell in range(2, 8)
    ]
    arrays = {}
    for name, matrix in [
        ("g", mints.ao_eri()),
        ("f", mints.ao_f12(terms)),
        ("gf", mints.ao_f12g12(terms)),
        ("squared", mints.ao_f12_squared(terms)),
        ("commutator", mints.ao_f12_double_commutator(terms)),
    ]:
        arrays[name] = np.asarray(matrix).reshape((n,) * 4)[
            np.ix_(*([permutation] * 4))
        ]
    arrays["overlap"] = np.asarray(mints.ao_overlap())[np.ix_(permutation, permutation)]
    arrays["version"] = np.array(psi4.__version__)
    np.savez(work / "operators.npz", **arrays)
    sys.exit(0)

subprocess.run(
    [args.psi4_python, str(Path(__file__).resolve()), "--integrals-only"], check=True
)
# The integral-only process deliberately does not need PySCF in the Psi4 env.
import pyscf  # noqa: E402
from pyscf import gto, lib, scf  # noqa: E402

lib.num_threads(4)
arrays = np.load(work / "operators.npz")
cases = []
for ecp in [False, True]:
    channels = [
        [-1, [[], [], [[0.7, 0.17]]]],
        [0, [[], [], [[0.53, 0.11]]]],
        [1, [[], [], [[0.61, -0.07]]]],
    ]
    mol = gto.M(
        atom=[("He", p) for p in positions],
        basis={"He": [[0, [1.4, 1.0]], [0, [0.45, 1.0]]]},
        ecp={"He": [0, channels]} if ecp else {},
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    auxmol = gto.M(
        atom=[("ghost-He", p) for p in positions],
        basis={
            "He": [[0, [0.2, 1.0]], [0, [0.9, 1.0]], [0, [3.2, 1.0]], [1, [0.8, 1.0]]]
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    union = gto.conc_mol(mol, auxmol)
    hf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-12)
    assert hf.converged
    nao, nm, no = mol.nao, hf.mo_coeff.shape[1], mol.nelectron // 2
    overlap = union.intor("int1e_ovlp")
    assert np.max(np.abs(overlap - arrays["overlap"])) < 1e-13
    assert np.max(np.abs(union.intor("int2e") - arrays["g"])) < 1e-13
    projection = hf.mo_coeff @ hf.mo_coeff.T @ overlap[:nao, nao:]
    eigenvalues, rotation = np.linalg.eigh(
        overlap[nao:, nao:] - overlap[nao:, :nao] @ projection
    )
    caux = rotation[:, eigenvalues > 1e-8] / np.sqrt(eigenvalues[eigenvalues > 1e-8])
    ri = np.zeros((union.nao, nm + caux.shape[1]))
    ri[:nao, :nm] = hf.mo_coeff
    ri[:, nm:] = np.vstack((-projection @ caux, caux))
    nri = ri.shape[1]
    obsmos = ri[:, :nm]
    occupied = ri[:, :no]
    factor = np.einsum(
        "ap,bi,cq,dj,abcd->pqij", ri, occupied, ri, occupied, arrays["f"], optimize=True
    ).reshape(nri**2, no**2)
    gf = np.einsum(
        "ap,bi,cq,dj,abcd->pqij",
        obsmos,
        occupied,
        obsmos,
        occupied,
        arrays["gf"],
        optimize=True,
    ).reshape(nm**2, no**2)
    coulomb = np.einsum(
        "ap,br,cq,ds,abcd->pqrs", obsmos, ri, obsmos, ri, arrays["g"], optimize=True
    ).reshape(nm**2, nri**2)
    removed = np.array(
        [
            (p < nm and q < nm) or (p >= nm and q < no) or (p < no and q >= nm)
            for p in range(nri)
            for q in range(nri)
        ]
    )
    v = (gf - coulomb @ (removed[:, None] * factor)).T.reshape(no**2, nm, nm)
    cases.append(
        dict(nonlocal_ecp=ecp, mo_coefficients=hf.mo_coeff.tolist(), v=v.tolist())
    )
out = dict(
    source=f"Psi4 {arrays['version']} full Gaussian Coulomb/F12 AO operators; PySCF {pyscf.__version__} RHF/CABS; explicit dense product-space projector",
    positions=positions,
    obs_g94=obs,
    aux_g94=aux,
    terms=terms,
    cases=cases,
)
reference_output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(
    "Projected full-OBS Coulomb/F12 matrices:",
    len(cases),
    "cases,",
    sum(len(c["v"]) * (2 * len(positions)) ** 2 for c in cases),
    "components",
    flush=True,
)
