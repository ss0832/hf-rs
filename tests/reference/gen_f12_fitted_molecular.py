# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Fitted F12 corrections with separate orbital-MP2 and mixed-Fock controls.

Public Psi4 lower-center integrals feed the original-auxiliary robust formula.
The existing independent product/geminal Gram-Hamiltonian oracle then evaluates
the functional and public PySCF modified CCSD iteration. No external sources or
internal routines are inspected. This is not a published default benchmark.
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
parser.add_argument("--mixed-fock", action="store_true")
parser.add_argument("--spherical-fitting", action="store_true")
parser.add_argument("--spherical-fock-fitting", action="store_true")
parser.add_argument("--fit-orbital-mp2", action="store_true")
parser.add_argument("--exact-obs-coulomb", action="store_true")
parser.add_argument("--exact-correction", action="store_true")
parser.add_argument("--output")
parser.add_argument(
    "--psi4-python",
    default=os.environ.get("HF_RS_PSI4_PYTHON"),
    help="Python executable with Psi4 (or set HF_RS_PSI4_PYTHON)",
)
args = parser.parse_args()
if not args.integrals_only and not args.psi4_python:
    parser.error("Provide --psi4-python or set HF_RS_PSI4_PYTHON")
root = Path(__file__).resolve().parents[2]
work = root / (
    "target/f12-mixed-fock-reference"
    if args.mixed_fock
    else "target/f12-fitted-molecular-reference"
)
work.mkdir(parents=True, exist_ok=True)
positions = [[-0.1, 0.2, -1.3], [0.3, -0.2, 1.4]]
terms = [(0.3, -0.4), (1.2, -0.2)]
obs = "He 0\nS 1 1.0\n1.4 1.0\nS 1 1.0\n0.45 1.0\n****\n"
aux = "He 0\nS 1 1.0\n0.2 1.0\nS 1 1.0\n0.9 1.0\nS 1 1.0\n3.2 1.0\nP 1 1.0\n0.8 1.0\n****\n"
fit = (
    "He 0\n"
    + "".join(
        f"{angular} 1 1.0\n{exponent} 1.0\n"
        for angular, exponent in [
            ("S", 0.18),
            ("S", 0.65),
            ("S", 2.0),
            ("S", 5.6),
            ("P", 0.25),
            ("P", 1.1),
            ("D", 0.7),
        ]
    )
    + "****\n"
)
fock_fit = (
    fit.replace("0.18 ", "0.22 ")
    .replace("0.65 ", "0.72 ")
    .replace("2.0 ", "2.3 ")
    .replace("5.6 ", "6.2 ")
    .replace("0.25 ", "0.32 ")
    .replace("1.1 ", "1.4 ")
    .replace("0.7 ", "0.83 ")
)
operators_path = work / "operators.npz"
if args.integrals_only:
    import psi4

    os.chdir(work)
    psi4.set_num_threads(4)
    psi4.set_memory("512 MiB")
    psi4.core.set_output_file(str(work / "mints.out"), False)
    molecule = psi4.geometry(
        "0 1\n"
        + "\n".join("He " + " ".join(map(str, p)) for p in positions)
        + "\nunits bohr\nsymmetry c1\nno_com\nno_reorient"
    )
    union_g94 = obs.replace("****\n", "") + aux.replace("He 0\n", "")
    bases = []
    for name, g94 in [
        ("hf_rs_union", union_g94),
        ("hf_rs_fit", fit),
        ("hf_rs_fock_fit", fock_fit),
    ]:
        (work / f"{name}.gbs").write_text("cartesian\n****\n" + g94, encoding="utf-8")
        bases.append(psi4.core.BasisSet.build(molecule, "BASIS", name, puream=0))
    union, fitting, fock_fitting = bases
    mints = psi4.core.MintsHelper(union)
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    n, na = union.nbf(), fitting.nbf()
    io = [8 * atom + shell for atom in range(2) for shell in [0, 1]]
    io += [8 * atom + shell for atom in range(2) for shell in range(2, 8)]
    ia = [
        16 * atom + shell
        for atom in range(2)
        for shell in list(range(10)) + [10, 13, 15, 11, 12, 14]
    ]
    norm = np.sqrt(np.diag(mints.ao_overlap(union, union).np))[io]
    norm_aux = np.sqrt(np.diag(mints.ao_overlap(fitting, fitting).np))[ia]

    def physical_transform(basis, name, spherical, normalization):
        if not spherical:
            return np.eye(basis.nbf())
        pure = psi4.core.BasisSet.build(molecule, "BASIS", name, puream=1)
        raw_s = np.asarray(mints.ao_overlap(basis, basis))
        cross = np.asarray(mints.ao_overlap(basis, pure))
        return np.linalg.solve(raw_s, cross)[ia] * normalization[:, None]

    physical = physical_transform(
        fitting, "hf_rs_fit", args.spherical_fitting, norm_aux
    )

    def two(function):
        raw = np.asarray(function(fitting, zero, fitting, zero)).reshape(na, na)
        normalized = raw[np.ix_(ia, ia)] / norm_aux[:, None] / norm_aux[None, :]
        return physical.T @ normalized @ physical

    def three(function):
        raw = np.asarray(function(fitting, zero, union, union)).reshape(na, n, n)
        normalized = (
            raw[np.ix_(ia, io, io)]
            / norm_aux[:, None, None]
            / norm[None, :, None]
            / norm[None, None, :]
        )
        return np.einsum("PA,Pmn->Amn", physical, normalized)

    metric = two(mints.ao_eri)
    values, vectors = np.linalg.eigh(metric)
    inverse = (vectors[:, values > 1e-10] / values[values > 1e-10]) @ vectors[
        :, values > 1e-10
    ].T
    coulomb = three(mints.ao_eri).reshape(physical.shape[1], n * n)
    d = inverse @ coulomb
    arrays = {"g": (d.T @ coulomb).reshape((n,) * 4)}
    for name, method in [
        ("f", mints.ao_f12),
        ("gf", mints.ao_f12g12),
        ("squared", mints.ao_f12_squared),
        ("commutator", mints.ao_f12_double_commutator),
    ]:
        print("Fitted kernel", name, flush=True)

        def function(*bases, method=method):
            return method(terms, *bases)

        w2, w3 = two(function), three(function).reshape(physical.shape[1], n * n)
        arrays[name] = (d.T @ w3 + w3.T @ d - d.T @ w2 @ d).reshape((n,) * 4)
        if args.exact_correction:
            raw_exact = np.asarray(function(union, union, union, union)).reshape(
                (n,) * 4
            )
            exact = raw_exact[np.ix_(*([io] * 4))]
            for axis in range(4):
                shape = [1] * 4
                shape[axis] = n
                exact /= norm.reshape(shape)
            arrays[name] = exact
    raw = np.asarray(mints.ao_eri(union, union, union, union)).reshape((n,) * 4)
    arrays["g_exact"] = raw[np.ix_(*([io] * 4))]
    for axis in range(4):
        shape = [1] * 4
        shape[axis] = n
        arrays["g_exact"] /= norm.reshape(shape)
    if args.exact_correction:
        arrays["g"] = arrays["g_exact"]
    if args.mixed_fock:
        fit_norm = np.sqrt(np.diag(mints.ao_overlap(fock_fitting, fock_fitting).np))[ia]
        transform = physical_transform(
            fock_fitting, "hf_rs_fock_fit", args.spherical_fock_fitting, fit_norm
        )
        metric = np.asarray(
            mints.ao_eri(fock_fitting, zero, fock_fitting, zero)
        ).reshape(na, na)
        metric = metric[np.ix_(ia, ia)] / fit_norm[:, None] / fit_norm[None, :]
        metric = transform.T @ metric @ transform
        three_fock = np.asarray(mints.ao_eri(fock_fitting, zero, union, union)).reshape(
            na, n, n
        )
        three_fock = (
            three_fock[np.ix_(ia, io, io)]
            / fit_norm[:, None, None]
            / norm[None, :, None]
            / norm[None, None, :]
        )
        three_fock = np.einsum("PA,Pmn->Amn", transform, three_fock).reshape(
            transform.shape[1], n * n
        )
        arrays["g_fock"] = (three_fock.T @ np.linalg.solve(metric, three_fock)).reshape(
            (n,) * 4
        )
    arrays.update(
        version=np.array(psi4.__version__),
        positions=np.array(json.dumps(positions)),
        terms=np.array(json.dumps(terms)),
        obs_g94=np.array(obs),
        aux_g94=np.array(aux),
        mixed_fock=np.array(args.mixed_fock),
        fit_orbital_mp2=np.array(args.fit_orbital_mp2),
        exact_obs_coulomb=np.array(args.exact_obs_coulomb),
    )
    np.savez(operators_path, **arrays)
    print("Fitting metric rank", sum(values > 1e-10), "of", na, flush=True)
    sys.exit(0)

flags = [
    flag
    for enabled, flag in [
        (args.mixed_fock, "--mixed-fock"),
        (args.spherical_fitting, "--spherical-fitting"),
        (args.spherical_fock_fitting, "--spherical-fock-fitting"),
        (args.fit_orbital_mp2, "--fit-orbital-mp2"),
        (args.exact_obs_coulomb, "--exact-obs-coulomb"),
        (args.exact_correction, "--exact-correction"),
    ]
    if enabled
]
subprocess.run(
    [args.psi4_python, str(Path(__file__).resolve()), "--integrals-only"] + flags,
    check=True,
)
destination = (
    Path(args.output)
    if args.output
    else root / "tests/data/f12_fitted_molecular_reference.json"
)
subprocess.run(
    [sys.executable, str(Path(__file__).with_name("gen_ccsd_f12.py"))],
    env=dict(
        os.environ,
        HF_RS_F12_FITTED_OPERATORS=str(operators_path),
        HF_RS_CCSD_F12_REFERENCE_OUTPUT=str(destination),
    ),
    check=True,
)
data = json.loads(destination.read_text(encoding="utf-8"))
data["fit_g94"] = fit
data["fock_fit_g94"] = fock_fit
data["mixed_fock"] = args.mixed_fock
data["spherical_fitting"] = args.spherical_fitting
data["spherical_fock_fitting"] = args.spherical_fock_fitting
data["fit_orbital_mp2"] = args.fit_orbital_mp2
data["exact_obs_coulomb"] = args.exact_obs_coulomb
data["exact_correction"] = args.exact_correction
data["fitted_correction"] = not args.exact_correction
data["source"] = (
    "Public Psi4 lower-center integrals; original-auxiliary robust fitting; "
    + data["source"]
)
data["scope"] = (
    "Explicit fitted correction, Fock and orbital-MP2 controls; exact orbital CCSD/(T). Not a published/default calculation."
)
destination.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
