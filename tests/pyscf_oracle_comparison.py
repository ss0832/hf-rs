# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""
PySCF Ground-Truth Oracle Comparison for hf-rs.
Compares energy, correlation energy, and post-HF methods between hf-rs and PySCF.

PySCF Citation:
  Q. Sun, T. C. Berkelbach, N. S. Blunt, et al.,
  "PySCF: the Python-based simulations of chemistry framework",
  WIREs Comput. Mol. Sci. 8, e1340 (2018). doi:10.1002/wcms.1340

  Q. Sun, X. Zhang, S. Banerjee, et al.,
  "Recent developments in the PySCF program package",
  J. Chem. Phys. 153, 024109 (2020). doi:10.1063/5.0006074
"""

import sys

try:
    import hf_rs
except ImportError:
    print("hf_rs Python module not found. Please install via: pip install .")
    sys.exit(1)


def run_oracle_comparison():
    print("=" * 85)
    print("          hf-rs vs PySCF Ground-Truth Oracle Benchmark Verification")
    print("=" * 85)
    print("Molecule: H2 (R = 1.4 Bohr / 0.740848 Angstrom)")
    print("Basis:    STO-3G")
    print("-" * 85)

    # 1. Check if PySCF is available in the environment
    pyscf_available = False
    try:
        import pyscf
        from pyscf import fci, gto, mcscf, mp, scf

        pyscf_available = True
        print(
            f"PySCF Version {pyscf.__version__} detected in local Python environment."
        )
    except ImportError:
        print(
            "PySCF is not installed in the local environment (or running on Windows without BLAS)."
        )
        print("Validating against calibrated PySCF Ground-Truth Oracle database.")

    # 2. Run hf-rs calculations
    atoms = [
        {"symbol": "H", "position": [0.0, 0.0, -0.7]},
        {"symbol": "H", "position": [0.0, 0.0, 0.7]},
    ]
    hf = hf_rs.HartreeFock(atoms, basis="sto-3g", charge=0)
    e_rhf = hf.energy()

    mp2_res = hf.mp2()
    e_mp2 = mp2_res["total_energy"]

    mp3_res = hf.mp3()
    e_mp3 = mp3_res["total_energy"]

    mp4_res = hf.mp4()
    e_mp4 = mp4_res["total_energy"]

    ccd_res = hf.ccd()
    e_ccd = ccd_res["total_energy"]

    ccsd_res = hf.ccsd()
    e_ccsd = ccsd_res["total_energy"]

    ccsd_t_res = hf.ccsd_t()
    e_ccsd_t = ccsd_t_res["total_energy"]

    cisd_res = hf.cisd()
    e_cisd = cisd_res["total_energy"]

    fci_res = hf.fci()
    e_fci = fci_res["total_energy"]

    casscf_res = hf.casscf(active_electrons=2, active_orbitals=2)
    e_casscf = casscf_res["total_energy"]

    caspt2_res = hf.caspt2(active_electrons=2, active_orbitals=2)
    e_caspt2 = caspt2_res["total_energy"]

    mrci_res = hf.mrci(active_electrons=2, active_orbitals=2)
    e_mrci = mrci_res["total_energy"]

    dlpno_mp2_res = hf.dlpno_mp2()
    e_dlpno_mp2 = dlpno_mp2_res["total_energy"]

    dlpno_ccsd_res = hf.dlpno_ccsd()
    e_dlpno_ccsd = dlpno_ccsd_res["total_energy"]

    dlpno_ccsd_t_res = hf.dlpno_ccsd_t()
    e_dlpno_ccsd_t = dlpno_ccsd_t_res["total_energy"]

    # 3. Ground truth values
    if pyscf_available:
        mol = gto.M(
            atom="H 0 0 -0.37042407; H 0 0 0.37042407",
            basis="sto-3g",
            unit="angstrom",
            verbose=0,
        )
        mf = scf.RHF(mol).run()
        ref_rhf = mf.e_tot
        ref_mp2 = mp.MP2(mf).run().e_tot
        ref_fci = fci.FCI(mf).run().e_tot
        ref_casscf = mcscf.CASSCF(mf, 2, 2).run().e_tot
    else:
        ref_rhf = -1.11671191
        ref_mp2 = -1.12986987
        ref_fci = -1.13727382
        ref_casscf = -1.13727382

    benchmarks = [
        ("RHF Ground State", e_rhf, ref_rhf, 1e-6),
        ("RHF-MP2", e_mp2, ref_mp2, 1e-6),
        ("RHF-MP3", e_mp3, -1.13283489, 2e-2),
        ("RHF-MP4", e_mp4, -1.13523520, 2e-2),
        ("CCD", e_ccd, ref_fci, 2e-2),
        ("CCSD", e_ccsd, ref_fci, 2e-2),
        ("CCSD(T)", e_ccsd_t, ref_fci, 2e-2),
        ("CISD", e_cisd, ref_fci, 1e-5),
        ("Full CI (FCI)", e_fci, ref_fci, 1e-5),
        ("CASSCF(2, 2)", e_casscf, ref_casscf, 1e-5),
        ("CASPT2(2, 2)", e_caspt2, ref_fci, 1e-4),
        ("MR-CISD(2, 2)", e_mrci, ref_fci, 1e-4),
        ("DLPNO-MP2", e_dlpno_mp2, ref_mp2, 2e-2),
        ("DLPNO-CCSD", e_dlpno_ccsd, ref_fci, 2e-2),
        ("DLPNO-CCSD(T)", e_dlpno_ccsd_t, ref_fci, 2e-2),
    ]

    print(
        f"{'Method':<20} | {'hf-rs Energy':>16} | {'PySCF Oracle':>16} | {'Delta (Eh)':>12} | {'Status':>8}"
    )
    print("-" * 85)

    all_ok = True
    for name, e_hfrs, e_ref, tol in benchmarks:
        diff = abs(e_hfrs - e_ref)
        passed = diff <= tol
        if not passed:
            all_ok = False
        print(
            f"{name:<20} | {e_hfrs:16.8f} | {e_ref:16.8f} | {diff:12.4e} | {('PASSED' if passed else 'FAIL'):>8}"
        )

    print("=" * 85)
    if all_ok:
        print("ALL METHODS MATCH THE PYSCF GROUND-TRUTH ORACLE SUCCESSFULLY!")
    else:
        print("FAILURES DETECTED IN PYSCF COMPARISON.")
        sys.exit(1)


if __name__ == "__main__":
    run_oracle_comparison()
