# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Published Molpro F12 targets and an explicitly assumed comparison setup.

Source: H.-J. Werner, molpro-user mailing list, 19 October 2008,
https://www.molpro.net/pipermail/molpro-user/2008-October/002688.html (water dimer, row
"vdz-f12"). The geometry (Angstrom, the dummy "lp" center omitted) and the published
energies are transcribed below. The post explicitly sets df_basis_exch=vtz, but does
not identify the RI/CABS basis, every numerical threshold, geminal fit coefficients,
or all details of the 2008 implementation. Current documented defaults do not establish
those historical settings. The comparison setup below assumes OptRI, beta=1, two frozen
core orbitals and omitted core singles. These assumptions are not verified attributes
of the publication. A first native comparison fails the printed-digit criterion;
these numbers remain diagnostic targets. Basis data come from the local BSE package
or from its existing embedded export (no download).

Molpro's release notes explicitly date the OptRI default to 2010.1 and a CABS
construction/threshold change to 2009.1:
https://www.molpro.net/manual/doku.php?id=recent_changes
"""

import argparse
import json
from pathlib import Path

ANGSTROM_TO_BOHR = 1.0 / 0.52917721092

GEOMETRY_ANGSTROM = [
    ("O", [0.0640488861, -1.5542001572, 0.0000000000]),
    ("H", [-0.0275603110, -0.6109761929, 0.0000000000]),
    ("H", [-0.8291328873, -1.8994223998, 0.0000000000]),
    ("O", [-0.0539423286, 1.4751039841, 0.0000000000]),
    ("H", [0.3402097464, 1.8881478835, 0.7547208224]),
    ("H", [0.3402097464, 1.8881478835, -0.7547208224]),
]

PUBLISHED_VDZ_F12 = dict(
    rhf=-152.123795554166,
    cabs_singles=-0.013293840277,
    reference=-152.137089394443,
    mp2_correlation=-0.482269701797,
    mp2_f12_correlation=-0.593338690191,
    ccsd_t_f12a_correlation=-0.589941424263,
    ccsd_t_f12b_correlation=-0.577029504611,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-basis",
        action="store_true",
        help="Reuse the BSE export already embedded in the fixture; no BSE install is needed.",
    )
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "data/f12_literature_reference.json"
    if args.reuse_basis:
        previous = json.loads(path.read_text(encoding="utf-8"))
        bse_version = previous["bse_version"]
        obs_g94, cabs_g94 = previous["obs_g94"], previous["cabs_g94"]
    else:
        import basis_set_exchange as bse

        bse_version = bse.version()
        obs_g94, cabs_g94 = [
            bse.get_basis(name, elements=["H", "O"], fmt="gaussian94", header=False)
            for name in ["cc-pvdz-f12", "cc-pvdz-f12-optri"]
        ]
    record = dict(
        source=__doc__,
        bse_version=bse_version,
        name="water_dimer_cc_pvdz_f12",
        atoms=[
            dict(symbol=s, position=[x * ANGSTROM_TO_BOHR for x in p])
            for s, p in GEOMETRY_ANGSTROM
        ],
        obs_g94=obs_g94,
        cabs_g94=cabs_g94,
        spherical=True,
        geminal_exponent=1.0,
        frozen_core=2,
        cabs_core_singles=False,
        published_settings={"df_basis_exch": "vtz"},
        comparison_assumptions={
            "ri_basis": "cc-pvdz-f12-optri",
            "geminal_exponent": 1.0,
            "frozen_core": 2,
            "cabs_core_singles": False,
            "published_ri_basis_confirmed": False,
            "printed_digit_agreement_verified": False,
        },
        historical_default_changes={
            "source": "https://www.molpro.net/manual/doku.php?id=recent_changes",
            "optri_default_since": "2010.1",
            "cabs_construction_changed": "2009.1",
            "exact_2008_ri_settings_confirmed": False,
        },
        published=PUBLISHED_VDZ_F12,
    )
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print("written", path)


if __name__ == "__main__":
    main()
