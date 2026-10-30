"""Complete spin-string sigma, S², 1/2/3-RDM and F.4-RDM oracles."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import fci, lib

lib.num_threads(4)
n = 4
h = np.array(
    [
        [-1.4 + 0.4 * p if p == q else 0.05 * np.sin(p + q + 1) for q in range(n)]
        for p in range(n)
    ]
)


def pair(p, q):
    return max(p, q) * (max(p, q) + 1) // 2 + min(p, q)


g = np.empty((n, n, n, n))
for p in range(n):
    for q in range(n):
        for r in range(n):
            for s in range(n):
                g[p, q, r, s] = sum(
                    0.1
                    * np.sin((k + 1) * (pair(p, q) + 1) * 0.37)
                    * np.sin((k + 1) * (pair(r, s) + 1) * 0.37)
                    for k in range(3)
                )
f = np.array(
    [
        [0.3 * (p + 1) if p == q else 0.07 * np.cos(p + q + 1) for q in range(n)]
        for p in range(n)
    ]
)
out = {
    "source": f"PySCF {pyscf.__version__}, direct_spin1 normal-ordered spin-traced RDMs and spin_op.contract_ss",
    "n_orb": n,
    "f": f.tolist(),
    "cases": [],
}
for na, nb in [(2, 2), (2, 1), (3, 0)]:
    size = fci.cistring.num_strings(n, na) * fci.cistring.num_strings(n, nb)
    c = np.sin(np.arange(size) + 0.37)
    c /= np.linalg.norm(c)
    absorbed = fci.direct_spin1.absorb_h1e(h, g, n, (na, nb), fac=0.5)
    sigma = fci.direct_spin1.contract_2e(absorbed, c, n, (na, nb)).ravel() + 0.321 * c
    ss = fci.spin_op.contract_ss(c, n, (na, nb)).ravel()
    dm1, dm2, dm3, dm4 = fci.direct_spin1.make_rdm1234(c, n, (na, nb))
    contracted = np.einsum("pqrsvwtu,tu->pqrsvw", dm4, f)
    energy, ci = fci.direct_spin1.kernel(
        h, g, n, (na, nb), ecore=0.321, nroots=3, verbose=0, tol=1e-13
    )
    targets = {}
    if (na, nb) == (2, 2):
        # Diagonalize the exact S² sectors. A spin-penalty Davidson run can
        # converge to an excited state of the requested spin from its default guess.
        unit = np.eye(size)
        hfull = np.column_stack(
            [
                fci.direct_spin1.contract_2e(absorbed, v, n, (na, nb)).ravel()
                + 0.321 * v
                for v in unit
            ]
        )
        ssfull = np.column_stack(
            [fci.spin_op.contract_ss(v, n, (na, nb)).ravel() for v in unit]
        )
        ss_e, ss_c = np.linalg.eigh(ssfull)
        for target in [0.0, 2.0, 6.0]:
            sector = ss_c[:, abs(ss_e - target) < 1e-9]
            target_e = np.linalg.eigvalsh(sector.T @ hfull @ sector)
            targets[str(target)] = float(target_e[0])
    out["cases"].append(
        {
            "n_alpha": na,
            "n_beta": nb,
            "vector": c.tolist(),
            "sigma": sigma.tolist(),
            "s_squared_action": ss.tolist(),
            "spin_targets": targets,
            "one_rdm": dm1.ravel().tolist(),
            "two_rdm": dm2.ravel().tolist(),
            "three_rdm": dm3.ravel().tolist(),
            "contracted_four_rdm": contracted.ravel().tolist(),
            "energies": np.asarray(energy).tolist(),
        }
    )
Path(__file__).resolve().parents[1].joinpath(
    "data/fci_engine_reference.json"
).write_text(json.dumps(out, indent=1) + "\n")
