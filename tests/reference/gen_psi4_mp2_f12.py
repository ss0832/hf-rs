# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent full Psi4 MP2-F12/3C(FIX) energies, without density fitting.

Run with an isolated Psi4 >=1.10 Python. All scratch/output stays under target.
The optional --published calculation reproduces Psi4's public mp2f12-1 water
benchmark with basis data from that installed package, without basis downloads.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import psi4

parser = argparse.ArgumentParser()
parser.add_argument("--published", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
work = root / "target/psi4-mp2-f12-reference"
work.mkdir(parents=True, exist_ok=True)
os.chdir(work)  # Psi4 also writes timer.dat when exiting.
psi4.set_num_threads(4)
psi4.set_memory("512 MiB")

obs_he = "He 0\nS 1 1.0\n1.4 1.0\nS 1 1.0\n0.45 1.0\n****\n"
aux_he = "He 0\nS 1 1.0\n0.2 1.0\nS 1 1.0\n0.9 1.0\nS 1 1.0\n3.2 1.0\n****\n"
aux_he2 = aux_he.replace("****", "P 1 1.0\n0.8 1.0\n****")
he = [("He", [0.13, -0.21, 0.31])]
he2 = [("He", [-0.1, 0.2, -1.3]), ("He", [0.3, -0.2, 1.4])]


def calculate(name, atoms, obs, aux, beta=1.0, frozen=0, spherical=False):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    psi4.core.set_output_file(str(work / f"{name}.out"), False)
    representation = "spherical" if spherical else "cartesian"
    (work / "hf_rs_obs.gbs").write_text(
        representation + "\n****\n" + obs, encoding="utf-8"
    )
    (work / "hf_rs_aux.gbs").write_text(
        representation + "\n****\n" + aux, encoding="utf-8"
    )
    coordinates = "\n".join(f"{s} {p[0]} {p[1]} {p[2]}" for s, p in atoms)
    molecule = psi4.geometry(
        "0 1\n" + coordinates + "\nunits bohr\nsymmetry c1\nno_com\nno_reorient"
    )
    psi4.set_options(
        {
            "basis": "hf_rs_obs",
            "cabs_basis": "hf_rs_aux",
            "puream": spherical,
            "reference": "rhf",
            "scf_type": "pk",
            "mp2_type": "conv",
            "freeze_core": False,
            "frozen_docc": [frozen],
            "e_convergence": 1e-12,
            "d_convergence": 1e-12,
            "f12_beta": beta,
            "cabs_singles": True,
        }
    )
    energy, wavefunction = psi4.energy("mp2-f12", molecule=molecule, return_wfn=True)
    variables = psi4.core.variables()
    assert wavefunction.frzcpi()[0] == frozen
    output = dict(
        name=name,
        atoms=atoms,
        obs_g94=obs,
        aux_g94=aux,
        beta=beta,
        frozen_core=frozen,
        spherical=spherical,
        n_mo=wavefunction.nmo(),
        n_occ=wavefunction.nalpha(),
        # Psi4 1.11 uses the raw -exp(-beta*r) fit with its fixed amplitudes.
        # Store that actual factor explicitly: hf-rs defaults to the physically
        # cusp-normalized -exp(-beta*r)/beta, which differs when beta != 1.
        terms=wavefunction.mintshelper().f12_cgtg(beta),
        factor_normalization="raw Psi4 MintsHelper fit, without division by beta",
        e_scf=variables["HF TOTAL ENERGY"],
        e_mp2=variables["MP2 CORRELATION ENERGY"],
        e_f12=variables["MP2-F12 CORRECTION ENERGY"],
        e_cabs=variables["F12 CABS CORRECTION ENERGY"],
        e_total=energy,
        e_f12_singlet=variables["MP2-F12 OPPOSITE-SPIN CORRELATION ENERGY"]
        - variables["MP2 OPPOSITE-SPIN CORRELATION ENERGY"],
        e_f12_triplet=variables["MP2-F12 SAME-SPIN CORRELATION ENERGY"]
        - variables["MP2 SAME-SPIN CORRELATION ENERGY"],
    )
    print(
        {k: v for k, v in output.items() if k.startswith("e_") or k == "name"},
        flush=True,
    )
    return output


cases = [
    calculate("he_beta_1", he, obs_he, aux_he),
    calculate("he_beta_0p7", he, obs_he, aux_he, beta=0.7),
    calculate("he2", he2, obs_he, aux_he2),
    calculate("he2_frozen_core", he2, obs_he, aux_he2, frozen=1),
    calculate("he2_beta_1p4", he2, obs_he, aux_he2, beta=1.4),
]
output = dict(
    source=f"Psi4 {psi4.__version__}, CONV MP2-F12/3C(FIX), exact PK SCF, full CABS singles",
    cases=cases,
)
(root / "tests/data/psi4_mp2_f12_reference.json").write_text(
    json.dumps(output, indent=1) + "\n", encoding="utf-8"
)


def subset(path):
    text = path.read_text(encoding="utf-8")
    blocks = []
    for block in text.split("****"):
        match = re.search(r"(?m)^\s*(H|O)\s+0\s*$", block)
        if match:
            blocks.append(block[match.start() :].strip() + "\n****\n")
    assert len(blocks) == 2
    return "\n".join(blocks), hashlib.sha256(text.encode()).hexdigest()


if args.published:
    bases = Path(psi4.core.get_datadir()) / "basis"
    obs, obs_hash = subset(bases / "cc-pvdz-f12.gbs")
    aux, aux_hash = subset(bases / "cc-pvdz-f12-optri.gbs")
    water = [
        ("O", [0.0, 0.0, 0.221664874]),
        ("H", [0.0, 1.430900622, -0.886659498]),
        ("H", [0.0, -1.430900622, -0.886659498]),
    ]
    case = calculate("water_cc_pvdz_f12", water, obs, aux, frozen=1, spherical=True)
    case["installed_basis_sha256"] = {
        "cc-pvdz-f12": obs_hash,
        "cc-pvdz-f12-optri": aux_hash,
    }
    case["public_reference"] = (
        "https://github.com/psi4/psi4/blob/master/tests/mp2f12-1/input.dat"
    )
    published = dict(source=output["source"], cases=[case])
    (root / "tests/data/psi4_mp2_f12_published_reference.json").write_text(
        json.dumps(published, indent=1) + "\n", encoding="utf-8"
    )
