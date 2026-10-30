"""Independent CASSCF energies and AO densities from embedded basis data."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import fci, gto, lib, mcscf, scf

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)


class ExactSmallFCI(fci.direct_spin1.FCI):
    # These reference spaces contain at most 225 determinants. Re-diagonalize the
    # full Hamiltonian: warm-start Davidson CI residuals otherwise limit the
    # orbital gradient long before the requested density accuracy is reached.
    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
        return super().kernel(h1e, eri, norb, nelec, ci0=None, **kwargs)


geometry = [("Li", [0.1, -0.2, 0.3]), ("H", [0.4, 0.1, 3.2])]
mol = gto.M(
    atom=geometry,
    unit="Bohr",
    basis=embedded_basis("STO_3G_GBS", ["Li", "H"]),
    cart=True,
    verbose=0,
)
mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-10)
results = {}
for name, ncas, nelec in [
    ("lih_cas22", 2, (1, 1)),
    ("lih_full", 6, (2, 2)),
    ("lih_closed", 1, (1, 1)),
]:
    mc = mcscf.CASSCF(mf, ncas, nelec)
    mc.conv_tol = 1e-13
    mc.conv_tol_grad = 1e-8
    mc.fcisolver.conv_tol = 1e-14
    mc.fcisolver.conv_tol_residual = 1e-12
    mc.max_cycle_macro = 150
    mc.kernel()
    assert mc.converged
    density_continuation_delta = None
    gradient = max(
        abs(mc.get_grad(mc.mo_coeff, mc.fcisolver.make_rdm12(mc.ci, ncas, nelec))),
        default=0.0,
    )
    if gradient > 1e-11:
        mc = mc.newton()
        mc.conv_tol_grad = 1e-11
        mc.kernel(mc.mo_coeff, mc.ci)
        gradient = max(
            abs(mc.get_grad(mc.mo_coeff, mc.fcisolver.make_rdm12(mc.ci, ncas, nelec))),
            default=0.0,
        )
    if gradient > 1e-10:
        previous = mc
        mc = mcscf.CASSCF(mf, ncas, nelec)
        mc.fcisolver = ExactSmallFCI(mol)
        mc.conv_tol = 1e-13
        mc.conv_tol_grad = 1e-10
        mc.max_cycle_macro = 200
        mc.kernel(previous.mo_coeff, previous.ci)
        density_continuation_delta = float(
            np.max(abs(mc.make_rdm1() - previous.make_rdm1()))
        )
        assert density_continuation_delta < 1e-8
    mc.e_tot, mc.e_cas, mc.ci = mc.casci(mc.mo_coeff, None)
    gradient = max(
        abs(mc.get_grad(mc.mo_coeff, mc.fcisolver.make_rdm12(mc.ci, ncas, nelec))),
        default=0.0,
    )
    print(name, mc.converged, mc.e_tot, mc.make_rdm1()[0, 0], gradient, flush=True)
    assert gradient < 1e-7
    results[name] = {
        "energy": float(mc.e_tot),
        "mo_coefficients": mc.mo_coeff.tolist(),
        "one_rdm_ao": mc.make_rdm1().tolist(),
        "s_squared": float(mc.fcisolver.spin_square(mc.ci, ncas, nelec)[0]),
        "orbital_gradient_max": float(gradient),
        "reported_converged_after_refinement": bool(mc.converged),
        "density_continuation_delta": density_continuation_delta,
    }
mc = mcscf.CASSCF(mf, 2, (1, 1))
mc.fcisolver = fci.direct_spin0.FCI(mol)
mc = mc.state_average_([0.4, 0.6])
mc.fcisolver.conv_tol = 1e-14
mc.fcisolver.conv_tol_residual = 1e-12
mc.conv_tol = 1e-13
mc.conv_tol_grad = 1e-8
mc.max_cycle_macro = 150
mc.kernel()
assert mc.converged
results["lih_sa22"] = {
    "energy_average": float(mc.e_tot),
    "energies": [float(x) for x in mc.e_states],
    "one_rdm_ao": mc.make_rdm1().tolist(),
}
Path(__file__).resolve().parents[1].joinpath("data/casscf_reference.json").write_text(
    json.dumps(
        {
            "source": f"PySCF {pyscf.__version__}, Cartesian embedded STO-3G",
            "basis_source_sha256": basis_hash(),
            "refinement_note": "Converged one-step CASSCF followed by Newton continuation and fresh CASCI. Stricter refinement flags are recorded; final max orbital gradient <1e-7 and density changes <1e-8 between refinements. Density acceptance in Rust remains 1e-7.",
            "results": results,
        },
        indent=2,
    )
    + "\n"
)
