# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent mixed-element basis and independent/associated scalar-ECP oracle."""

import json
import re
from pathlib import Path

import pyscf
from pyscf import gto, scf
from pyscf.gto.basis import parse_gaussian

from embedded_basis import embedded_basis

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "third_party/basis_set_exchange"
manifest = json.loads((CATALOG / "manifest.json").read_text(encoding="utf-8"))
exports = {entry["name"]: entry for entry in manifest["basis_sets"]}


def orbital(name, symbol):
    constants = {"sto-3g": "STO_3G_GBS", "6-31g": "BASIS_6_31G_GBS"}
    if name in constants:
        return embedded_basis(constants[name], [symbol])[symbol]
    text = (CATALOG / "data" / exports[name]["gaussian94_file"]).read_text(
        encoding="utf-8"
    )
    block = re.search(
        r"(?m)^{}\s+0\s*\n(.*?)\*\*\*\*".format(symbol), text, re.S
    ).group(1)
    return parse_gaussian.parse(block, optimize=False)


nah = [("Na", [0.2, -0.1, -1.4]), ("H", [0.5, 0.25, 1.65])]
cases = [
    (
        "water_mixed",
        [("O", [0.0, 0.0, 0.0]), ("H", [1.43, 0.0, 1.10]), ("H", [-1.43, 0.0, 1.10])],
        {"O": "6-31g", "H": "sto-3g"},
        None,
    ),
    ("nah_associated", nah, {"Na": "lanl2dz", "H": "sto-3g"}, None),
    ("nah_independent", nah, {"Na": "6-31g", "H": "sto-3g"}, {"Na": "lanl2dz"}),
    ("nah_all_electron", nah, {"Na": "6-31g", "H": "sto-3g"}, {"Na": "none"}),
]
record = dict(
    source="PySCF " + pyscf.__version__,
    cartesian=True,
    orbital_source="Exact embedded/BSE Gaussian 94 exports",
    ecp_source="Public PySCF LANL2DZ scalar ECP input data; cross-format BSE checks are separate",
    cases=[],
)
for name, atoms, basis_spec, ecp_spec in cases:
    basis = {symbol: orbital(family, symbol) for symbol, family in basis_spec.items()}
    ecps = {}
    if (ecp_spec is None and basis_spec.get("Na") == "lanl2dz") or (ecp_spec or {}).get(
        "Na"
    ) == "lanl2dz":
        ecps["Na"] = gto.basis.load_ecp("lanl2dz", "Na")
    molecule = gto.M(
        atom=atoms, unit="bohr", cart=True, basis=basis, ecp=ecps, verbose=0
    )
    reference = scf.RHF(molecule)
    reference.conv_tol = 1e-13
    reference.conv_tol_grad = 1e-10
    reference.max_cycle = 300
    reference.kernel()
    assert reference.converged, name
    gradient = reference.nuc_grad_method().kernel().tolist()
    record["cases"].append(
        dict(
            name=name,
            atomic_numbers=[gto.charge(symbol) for symbol, _ in atoms],
            positions=[position for _, position in atoms],
            basis=basis_spec,
            ecp=ecp_spec,
            n_ao=molecule.nao,
            electrons=molecule.nelectron,
            energy=reference.e_tot,
            gradient=gradient,
        )
    )
    print(name, reference.e_tot, flush=True)
(ROOT / "tests/data/basis_assignment_reference.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
