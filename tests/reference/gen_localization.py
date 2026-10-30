"""Independent PySCF Boys/PM and NumPy PAO reference matrices."""

import json
from pathlib import Path

import numpy as np
import pyscf
import scipy
from pyscf import gto, lib, lo, scf
from scipy.linalg import expm
from scipy.optimize import least_squares, minimize

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)
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
c = mf.mo_coeff[:, mf.mo_occ > 0]
s = mf.get_ovlp()
r = mol.intor("int1e_r", comp=3)


def objective(coeff, operators):
    return float(sum(np.sum(np.diag(coeff.T @ op @ coeff) ** 2) for op in operators))


pop = []
ranges = []
for _, _, start, end in mol.aoslice_by_atom():
    ranges.append([int(start), int(end)])
    p = np.zeros_like(s)
    p[start:end, :] = s[start:end, :]
    pop.append((p + p.T) / 2)
out = {
    "source": f"PySCF {pyscf.__version__} Boys/PM and SciPy {scipy.__version__} multistart BFGS plus stationary-gradient refinement; NumPy metric PAOs",
    "basis_source_sha256": basis_hash(),
    "cart": True,
    "coefficients": c.tolist(),
    "overlap": s.tolist(),
    "dipoles": r.tolist(),
    "atom_ao_ranges": ranges,
    "objectives": {},
}
for name, localizer, ops in [
    ("boys", lo.Boys(mol, c), r),
    ("pipek_mezey", lo.PM(mol, c, pop_method="mulliken"), pop),
]:
    localizer.init_guess = None
    localizer.conv_tol = 1e-13
    localizer.conv_tol_grad = 1e-10
    localizer.max_cycle = 1000
    if name == "pipek_mezey":
        localizer.exponent = 2
    localized = localizer.kernel(c)
    pyscf_objective = objective(localized, ops)
    pairs = [(i, j) for i in range(c.shape[1]) for j in range(i)]

    def rotation(x):
        k = np.zeros((c.shape[1], c.shape[1]))
        for value, (i, j) in zip(x, pairs):
            k[i, j] = value
            k[j, i] = -value
        return expm(k)

    def gradient(x):
        coeff = c @ rotation(x)
        dipoles = np.asarray([coeff.T @ op @ coeff for op in ops])
        return np.asarray(
            [
                4 * np.sum((dipoles[:, i, i] - dipoles[:, j, j]) * dipoles[:, i, j])
                for i, j in pairs
            ]
        )

    candidates = []
    for seed in range(3):
        start = np.random.default_rng(seed).normal(size=len(pairs)) * 0.3
        fit = minimize(
            lambda x: -objective(c @ rotation(x), ops),
            start,
            method="BFGS",
            options={"gtol": 1e-9, "maxiter": 1000},
        )
        stationary = least_squares(
            gradient, fit.x, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=1000
        )
        candidates.append((objective(c @ rotation(stationary.x), ops), stationary.x))
    best, x = max(candidates, key=lambda entry: entry[0])
    if np.max(np.abs(gradient(x))) > 1e-10:
        raise RuntimeError("unconverged localization reference")
    out["objectives"][name] = {
        "initial": objective(c, ops),
        "optimized": best,
        "pyscf_canonical_start": pyscf_objective,
        "multistart_objectives": [entry[0] for entry in candidates],
        "gradient_max": float(np.max(np.abs(gradient(x)))),
        "coefficients": (c @ rotation(x)).tolist(),
    }
q = np.eye(s.shape[0]) - c @ c.T @ s
values, vectors = np.linalg.eigh(q.T @ s @ q)
pao = q @ vectors[:, values > 1e-10] / np.sqrt(values[values > 1e-10])
out["pao_projector"] = (pao @ pao.T).tolist()
out["n_pao"] = pao.shape[1]
Path(__file__).resolve().parents[1].joinpath(
    "data/localization_reference.json"
).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print({name: case["optimized"] for name, case in out["objectives"].items()})
