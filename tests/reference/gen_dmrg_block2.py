# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""External block2 executable-I/O oracle; public driver API only.

Run in the isolated block2 Python environment, taking JSON integral inputs and
returning SZ two-site energies and spin-traced density tensors. No block2
implementation source or internal expressions are read by this harness.
"""

import json
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes


def main():
    records = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    for case in records:
        case["block2"] = []
        for m in case["bond_dimensions"]:
            with TemporaryDirectory(prefix="hf-rs-block2-") as scratch:
                driver = DMRGDriver(
                    scratch=scratch,
                    symm_type=SymmetryTypes.SZ,
                    n_threads=4,
                    stack_mem=256 << 20,
                )
                driver.initialize_system(
                    n_sites=case["n"],
                    n_elec=case["n_alpha"] + case["n_beta"],
                    spin=case["n_alpha"] - case["n_beta"],
                )
                mpo = driver.get_qc_mpo(
                    h1e=np.array(case["h"]),
                    g2e=np.array(case["eri"]),
                    ecore=case["constant"],
                    reorder=None,
                    cutoff=0.0,
                    iprint=0,
                )
                ket = driver.get_random_mps(tag="KET", bond_dim=m, nroots=1, dot=2)
                reported = driver.dmrg(
                    mpo,
                    ket,
                    n_sweeps=case["sweeps"] + case["one_site_sweeps"],
                    bond_dims=[m] * (case["sweeps"] + case["one_site_sweeps"]),
                    noises=[0.0],
                    thrds=[1e-24],
                    tol=0.0,
                    cutoff=1e-28,
                    twosite_to_onesite=case["sweeps"],
                    decomp_type="PureSVD",
                    iprint=0,
                )
                # A fresh contraction records the energy of the returned MPS.
                energy = driver.expectation(ket, mpo, ket)
                one = np.array(driver.get_1pdm(ket)).sum(axis=0)
                aa, ab, bb = np.array(driver.get_2pdm(ket))
                # block2: i^+ j^+ k l; hf-rs/PySCF: p^+ r^+ s q.
                two = (aa + ab + ab.transpose(1, 0, 3, 2) + bb).transpose(0, 3, 1, 2)
                reconstructed = (
                    case["constant"]
                    + np.einsum("pq,pq", case["h"], one)
                    + 0.5 * np.einsum("pqrs,pqrs", case["eri"], two)
                )
                assert abs(energy - reconstructed) < 1e-10
                case["block2"].append(
                    dict(
                        m=m,
                        reported_energy=float(reported),
                        energy=float(energy),
                        one_rdm=one.ravel().tolist(),
                        two_rdm=two.ravel().tolist(),
                    )
                )
                print(case["name"], m, reported, energy, flush=True)
                driver.finalize()
    Path(sys.argv[2]).write_text(
        json.dumps(
            dict(
                source=__doc__,
                block2_version=version("block2"),
                symmetry="SZ",
                sweep_definition="one directional two-site pass",
                noise=0.0,
                davidson_residual_squared=1e-24,
                cases=records,
            ),
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
