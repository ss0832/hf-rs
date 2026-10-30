"""Evaluate hf-rs Molden orbitals using PySCF's independent file reader/AO engine."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import lib
from pyscf.tools import molden

lib.num_threads(4)
directory = Path(__file__).resolve().parents[2] / "target/molden_cases"
cases = json.loads((directory / "reference.json").read_text())
results = {}
for name, case in cases.items():
    mol, energy, coefficients, occupations, _, _ = molden.load(
        str(directory / f"{name}.molden")
    )
    values = (
        mol.eval_gto(
            "GTOval_cart" if mol.cart else "GTOval_sph", np.array(case["points_bohr"])
        )
        @ coefficients
    )
    error = float(np.max(np.abs(values - np.array(case["values"]))))
    assert error < 1e-12, (name, error)
    assert np.allclose(energy, [-0.8, 0.1, 0.5], atol=1e-14, rtol=0)
    assert np.allclose(occupations, [2, 0, 0], atol=1e-14, rtol=0)
    results[name] = {"n_ao": mol.nao, "max_orbital_value_error": error}
out = {
    "source": f"PySCF {pyscf.__version__}, tools.molden.load + eval_gto, all s/p/d/f/g coefficients nonzero",
    "results": results,
}
(directory / "verified.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
