# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Public-interface PySCF CCSD trial residuals; no implementation source is read."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import ao2mo, cc, gto, lib, scf

from embedded_basis import embedded_basis

lib.num_threads(4)
rng = np.random.default_rng(613098)
cases = []
for name, atoms, constant, frozen in [
    ("h2", [("H", [0, 0, -0.7]), ("H", [0, 0, 0.7])], "STO_3G_GBS", 0),
    ("lih", [("Li", [0.1, -0.2, 0.3]), ("H", [0.4, 0.1, 3.2])], "BASIS_6_31G_GBS", 0),
    (
        "lih_frozen",
        [("Li", [0.1, -0.2, 0.3]), ("H", [0.4, 0.1, 3.2])],
        "BASIS_6_31G_GBS",
        1,
    ),
    (
        "water",
        [("O", [0, 0, 0]), ("H", [0.2, 1.4, 1.1]), ("H", [-0.1, -1.5, 1.0])],
        "STO_3G_GBS",
        0,
    ),
]:
    mol = gto.M(
        atom=atoms,
        basis=embedded_basis(constant, {a[0] for a in atoms}),
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-10, max_cycle=200)
    if not mf.converged:
        mf = mf.newton().run(conv_tol=1e-13, conv_tol_grad=1e-10, max_cycle=100)
    assert mf.converged, name
    engine = cc.CCSD(mf, frozen=frozen)
    eris = engine.ao2mo()
    no = mol.nelectron // 2 - frozen
    eps = mf.mo_energy[frozen:]
    # The trial problem has an exactly canonical Fock operator, so its algebra
    # does not inherit the last SCF occupied/virtual residual.
    eris.fock = np.diag(eps)
    nm = len(eps)
    nv = nm - no
    dense = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff[:, frozen:]), nm)
    pairs = [(p, q) for p in range(nm) for q in range(p + 1)]
    packed = [float(dense[p, q, r, s]) for r, s in pairs for p, q in pairs]
    trials = []
    for singles in [False, True]:
        t1 = rng.normal(size=(no, nv)) * 0.04 if singles else np.zeros((no, nv))
        raw = rng.normal(size=(no, no, nv, nv)) * 0.02
        t2 = (raw + raw.transpose(1, 0, 3, 2)) / 2
        next1, next2 = engine.update_amps(t1, t2, eris)
        gap = eps[:no, None] - eps[None, no:]
        trials.append(
            dict(
                t1=t1.ravel().tolist(),
                t2=t2.ravel().tolist(),
                energy=float(engine.energy(t1, t2, eris)),
                numerator1=(next1 * gap).ravel().tolist(),
                numerator2=(next2 * (gap[:, None, :, None] + gap[None, :, None, :]))
                .ravel()
                .tolist(),
            )
        )
    engine.conv_tol = 1e-13
    engine.conv_tol_normt = 1e-11
    engine.max_cycle = 200
    correlation, converged_t1, converged_t2 = engine.kernel(eris=eris)
    assert engine.converged, name
    doubles = cc.CCSD(mf, frozen=frozen)
    doubles.conv_tol = 1e-13
    doubles.conv_tol_normt = 1e-11
    doubles.max_cycle = 200
    public_update = doubles.update_amps

    def doubles_update(t1, t2, integrals, update=public_update):
        _, new_t2 = update(np.zeros_like(t1), t2, integrals)
        return np.zeros_like(t1), new_t2

    doubles.update_amps = doubles_update
    ccd_energy, _, ccd_t2 = doubles.kernel(eris=eris)
    assert doubles.converged, name
    cases.append(
        dict(
            name=name,
            n_occ=no,
            n_virt=nv,
            integrals=packed,
            orbital_energies=eps.tolist(),
            trials=trials,
            converged_energy=float(correlation),
            converged_t1=converged_t1.ravel().tolist(),
            converged_t2=converged_t2.ravel().tolist(),
            ccd_energy=float(ccd_energy),
            ccd_t2=ccd_t2.ravel().tolist(),
        )
    )
path = Path(__file__).resolve().parents[1] / "data/rhf_ccsd_residual_reference.json"
path.write_text(
    json.dumps(
        dict(
            source=f"PySCF {pyscf.__version__} public CCSD.update_amps and energy, exactly canonical Fock, seeded trial amplitudes",
            cases=cases,
        ),
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
print([(c["name"], c["n_occ"], c["n_virt"]) for c in cases])
