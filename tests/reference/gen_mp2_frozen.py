"""Frozen-core MP2 energies and full AO 1-RDM, using the embedded basis."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, lib, mp, scf

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)
mol = gto.M(
    atom=[("Li", [0.1, -0.2, 0.3]), ("H", [0.4, 0.1, 3.2])],
    unit="Bohr",
    basis=embedded_basis("BASIS_6_31G_GBS", ["Li", "H"]),
    cart=True,
    verbose=0,
)
mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-10)
# PySCF groups Li's S shells before P shells. The G94 builder preserves
# the interleaved S,P sequence of its two SP contractions.
hf_rs_to_pyscf = [0, 1, 3, 4, 5, 2, 6, 7, 8, 9, 10]
assert mol.nao_nr() == len(hf_rs_to_pyscf)
out = {
    "hf_rs_to_pyscf": hf_rs_to_pyscf,
    "pyscf_ao_labels": mol.ao_labels(),
    "source": f"PySCF {pyscf.__version__}, Cartesian, embedded 6-31G",
    "basis_source_sha256": basis_hash(),
    "results": {},
}
for frozen in [0, 1]:
    pt = mp.MP2(mf, frozen=frozen).run()
    out["results"][str(frozen)] = {
        "e_scf": mf.e_tot,
        "e_corr": pt.e_corr,
        "e_same_spin": pt.e_corr_ss,
        "e_opposite_spin": pt.e_corr_os,
        "one_rdm_ao": pt.make_rdm1(ao_repr=True)[
            np.ix_(hf_rs_to_pyscf, hf_rs_to_pyscf)
        ].tolist(),
    }
Path(__file__).resolve().parents[1].joinpath(
    "data/mp2_frozen_reference.json"
).write_text(json.dumps(out, indent=2) + "\n")
