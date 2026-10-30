# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent PySCF AO counts, RHF energies and all dipole components."""

import json
from math import cos, pi, sin
from pathlib import Path

import pyscf
from pyscf import gto, scf

from embedded_basis import basis_hash, embedded_basis


def ring(z, offset=0.0):
    atoms = []
    for k in range(5):
        theta = 2.0 * pi * k / 5 + offset
        atoms.append(("C", (2.268 * cos(theta), 2.268 * sin(theta), z)))
        atoms.append(("H", (4.309 * cos(theta), 4.309 * sin(theta), z)))
    return atoms


systems = {
    "cpli": ring(0.0) + [("Li", (0.0, 0.0, 3.496))],
    "ferrocene": [("Fe", (0.0, 0.0, 0.0))] + ring(3.137) + ring(-3.137, pi / 5),
}
record = dict(
    source="PySCF " + pyscf.__version__,
    cartesian=True,
    basis_sha256=basis_hash(),
    systems={},
)
for name, atoms in systems.items():
    symbols = {symbol for symbol, _ in atoms}
    basis = embedded_basis("STO_3G_GBS", symbols)
    molecule = gto.M(atom=atoms, unit="bohr", basis=basis, cart=True, verbose=0)
    if name == "ferrocene":
        record["systems"][name] = dict(
            atoms=atoms,
            n_ao=molecule.nao,
            scope="Independent Cartesian AO count; no energy or dipole oracle claimed",
        )
        print(name, molecule.nao, flush=True)
        continue
    reference = scf.RHF(molecule)
    reference.conv_tol = 1e-13
    reference.conv_tol_grad = 1e-10
    reference.max_cycle = 300
    reference.kernel()
    if not reference.converged:
        initial = reference.make_rdm1()
        reference = reference.newton()
        reference.conv_tol = 1e-13
        reference.conv_tol_grad = 1e-10
        reference.max_cycle = 300
        reference.kernel(dm0=initial)
    assert reference.converged, name
    dipole = reference.dip_moment(unit="Debye", verbose=0).tolist()
    dipole_au = reference.dip_moment(unit="AU", verbose=0).tolist()
    record["systems"][name] = dict(
        atoms=atoms,
        n_ao=molecule.nao,
        energy=reference.e_tot,
        dipole_debye=dipole,
        dipole_au=dipole_au,
        comparison="Use atomic units for cross-program dipoles; Debye conversion constants differ.",
    )
    print(name, molecule.nao, reference.e_tot, dipole, flush=True)
path = Path(__file__).resolve().parents[1] / "data/organometallic_reference.json"
path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(
    json.dumps(
        {
            name: {key: value for key, value in item.items() if key != "atoms"}
            for name, item in record["systems"].items()
        },
        indent=2,
    )
)
