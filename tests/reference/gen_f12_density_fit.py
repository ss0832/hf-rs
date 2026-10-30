# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Public Psi4 zero-basis integrals and independent robust fitting (Knizia Eq. 2.9).

Only public execution interfaces and their returned arrays are used; no external
implementation sources are read. All scratch stays below target.
"""

import json
import os
from pathlib import Path

import numpy as np
import psi4

root = Path(__file__).resolve().parents[2]
work = root / "target/f12-density-fit-reference"
work.mkdir(parents=True, exist_ok=True)
os.chdir(work)
psi4.set_num_threads(4)
psi4.set_memory("512 MiB")
psi4.core.set_output_file(str(work / "oracle.out"), False)
obs_g94 = "He 0\nS 1 1\n1.14 1\nP 1 1\n0.62 1\nD 1 1\n0.8 1\n****\n"
aux_g94 = "He 0\nS 1 1\n0.71 1\nP 1 1\n0.43 1\nD 1 1\n0.38 1\nF 1 1\n0.19 1\nG 1 1\n0.23 1\n****\n"
positions = [[-0.41, 0.23, 0.19], [0.54, -0.13, 0.05]]
obs_g94 = obs_g94.replace(" 1\n", " 1.0\n")
aux_g94 = aux_g94.replace(" 1\n", " 1.0\n")
bases = []
for name, gbs, position in zip(
    ["hf_rs_obs", "hf_rs_aux"], [obs_g94, aux_g94], positions
):
    (work / f"{name}.gbs").write_text("cartesian\n****\n" + gbs)
    molecule = psi4.geometry(
        f"0 1\nHe {position[0]} {position[1]} {position[2]}\nunits bohr\nno_com\nno_reorient\nsymmetry c1"
    )
    bases.append(psi4.core.BasisSet.build(molecule, "BASIS", name, puream=0))
obs, aux = bases
mints = psi4.core.MintsHelper(obs)
zero = psi4.core.BasisSet.zero_ao_basis_set()


def order(basis):
    result = []
    offset = 0
    for shell_index in range(basis.nshell()):
        shell = basis.shell(shell_index)
        angular = shell.am
        count = (angular + 1) * (angular + 2) // 2
        result.extend(
            offset + k for k in ([0, 3, 5, 1, 2, 4] if angular == 2 else range(count))
        )
        offset += count
    return result


io, ia = order(obs), order(aux)
no, na = obs.nbf(), aux.nbf()
print("overlap", no, na, flush=True)
normo = np.sqrt(np.diag(mints.ao_overlap(obs, obs).np))[io]
norma = np.sqrt(np.diag(mints.ao_overlap(aux, aux).np))[ia]


def two(function):
    raw = np.asarray(function(aux, zero, aux, zero)).reshape(na, na)
    return raw[np.ix_(ia, ia)] / norma[:, None] / norma[None, :]


def three(function):
    # The public engine's three-center braket has the unit in the bra.
    # Electron exchange maps this to the requested (mu nu|W|P) order.
    raw = (
        np.asarray(function(aux, zero, obs, obs)).reshape(na, no, no).transpose(1, 2, 0)
    )
    return (
        raw[np.ix_(io, io, ia)]
        / normo[:, None, None]
        / normo[None, :, None]
        / norma[None, None, :]
    )


print("Coulomb metric", flush=True)
metric = two(mints.ao_eri)
print("Coulomb three center", flush=True)
coulomb = three(mints.ao_eri)
pairs = [(p, q) for p in range(no) for q in range(p + 1)]
tc = np.array([coulomb[p, q, :] for p, q in pairs]).T
values, vectors = np.linalg.eigh(metric)
x = vectors[:, values > 1e-10] / np.sqrt(values[values > 1e-10])
d = x @ x.T @ tc
# Obtain the pure auxiliary subspace from public mixed-basis overlaps, rather
# than sharing the native solid-harmonic implementation.
pure_aux = psi4.core.BasisSet.build(aux.molecule(), "BASIS", "hf_rs_aux", puream=1)
overlap_cart = np.asarray(mints.ao_overlap(aux, aux))
cross_pure = np.asarray(mints.ao_overlap(aux, pure_aux))
transform = np.linalg.solve(overlap_cart, cross_pure)[ia] * norma[:, None]
pure_metric = transform.T @ metric @ transform
pure_values, pure_vectors = np.linalg.eigh(pure_metric)
pure_x = transform @ (
    pure_vectors[:, pure_values > 1e-10] / np.sqrt(pure_values[pure_values > 1e-10])
)
pure_d = pure_x @ pure_x.T @ tc
terms = [(0.31, -0.4), (0.92, 0.16), (0.31, 0.09)]
rng = np.random.default_rng(781401)
left = rng.normal(size=(no, 4)) * 0.2
occupied = rng.normal(size=(no, 2)) * 0.3
pair_index = np.array(
    [
        [max(p, q) * (max(p, q) + 1) // 2 + min(p, q) for q in range(no)]
        for p in range(no)
    ]
)
cases = []
for name, method, scale in [
    ("factor", mints.ao_f12, 1),
    ("factor_over_r", mints.ao_f12g12, 1),
    ("squared", mints.ao_f12_squared, 1),
    ("commutator", mints.ao_f12_double_commutator, 2),
]:
    print("kernel", name, flush=True)

    def function(*bases, method=method):
        return method(terms, *bases)

    w2 = scale * two(function)
    w3 = scale * three(function)
    t3 = np.array([w3[p, q, :] for p, q in pairs]).T
    fitted = d.T @ t3 + t3.T @ d - d.T @ w2 @ d
    pure_fitted = pure_d.T @ t3 + t3.T @ pure_d - pure_d.T @ w2 @ pure_d
    full = fitted[np.ix_(pair_index.ravel(), pair_index.ravel())].reshape(
        no, no, no, no
    )
    mixed = np.einsum(
        "ap,bi,cq,dj,abcd->ijpq", left, occupied, left, occupied, full, optimize=True
    )
    pure_full = pure_fitted[np.ix_(pair_index.ravel(), pair_index.ravel())].reshape(
        no, no, no, no
    )
    pure_mixed = np.einsum(
        "ap,bi,cq,dj,abcd->ijpq",
        left,
        occupied,
        left,
        occupied,
        pure_full,
        optimize=True,
    )
    cases.append(
        dict(
            name=name,
            two=w2.tolist(),
            three=t3.tolist(),
            fitted=fitted.tolist(),
            mixed=mixed.tolist(),
            spherical_fitted=pure_fitted.tolist(),
            spherical_mixed=pure_mixed.tolist(),
        )
    )
fock_components = []
for kind, coefficients in [("cartesian", d), ("spherical", pure_d)]:
    pair_metric = coefficients.T @ tc
    full = pair_metric[np.ix_(pair_index.ravel(), pair_index.ravel())].reshape(
        no, no, no, no
    )
    density = 2 * occupied @ occupied.T
    fock_components.append(
        dict(
            kind=kind,
            coulomb=np.einsum("pqrs,rs->pq", full, density).tolist(),
            exchange=np.einsum("prqs,rs->pq", full, density).tolist(),
        )
    )
out = dict(
    source=f"Psi4 {psi4.__version__} public MintsHelper zero-basis F12 arrays; NumPy {np.__version__} original-auxiliary robust fitting Eq. 2.9",
    obs_g94=obs_g94,
    aux_g94=aux_g94,
    positions=positions,
    terms=terms,
    metric=metric.tolist(),
    coulomb_three=tc.tolist(),
    metric_cutoff=1e-10,
    spherical_rank=int(sum(pure_values > 1e-10)),
    fock_components=fock_components,
    left=left.tolist(),
    occupied=occupied.tolist(),
    cases=cases,
)
(root / "tests/data/f12_density_fit_reference.json").write_text(
    json.dumps(out, indent=1) + "\n"
)
print([(c["name"], np.max(np.abs(c["fitted"]))) for c in cases])
