# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent quadratic-BCH MR source roots and reference-CI updates.

Every commutator map comes from explicit full Fock matrices and a complete
GNO basis solve. The transformation is the degree-two polynomial of that map;
SciPy roots solve the nonlinear source without production CAR/fixed-point code.
Only the project's explicit quadratic-BCH convention is evaluated here.
"""

import json
from pathlib import Path

import numpy as np
from gen_gno import GeneralizedOracle
from gen_mr_driven import driven
from gen_mr_molecular import models
from gen_mr_relaxed_molecular import relax


def pack(operator):
    n = len(operator["one"])
    from itertools import combinations

    two = np.array(operator["two"]).reshape((n,) * 4)
    pairs = list(combinations(range(n), 2))
    return np.concatenate(
        (
            [operator["zero"]],
            np.array(operator["one"]).ravel(),
            [two[p, q, r, s] for p, q in pairs for r, s in pairs],
        )
    )


def main():
    root = Path(__file__).resolve().parents[1]
    source = json.loads(
        (root / "data/mr_driven_reference.json").read_text(encoding="utf-8")
    )
    generic = []
    for case in source["cases"]:
        oracle = GeneralizedOracle(
            case["n_spin"], case["members"], case["active"], case["core"]
        )
        solved = driven(
            oracle,
            pack(case["h"]),
            case["active"],
            case["core"],
            case["flow"],
            compute_perturbation=False,
            bch_order=2,
        )
        record = {
            k: case[k]
            for k in ("name", "n_spin", "n_spatial", "members", "active", "core", "h")
        }
        record.update(solved)
        generic.append(record)
        print(
            case["name"], solved["transformed"]["zero"], solved["residual"], flush=True
        )
    molecular = []
    for index, model in enumerate(models()):
        solution = relax(model, 5, "iterate")
        history = solution["history"]
        first = history[0]
        records = [
            dict(
                solution,
                mode="none",
                energy=first["source_energy"],
                mo_coeff=first["mo_coeff"],
                coefficients=first["source_coefficients"],
                history=[],
            )
        ]
        for mode, size in (("once", 1), ("twice", 2), ("iterate", len(history))):
            record = dict(
                solution, **history[size - 1], mode=mode, history=history[:size]
            )
            records.append(record)
        molecular.append(dict(model_index=index, name=model["name"], results=records))
        print(
            model["name"],
            [(r["mode"], r["energy"], len(r["history"])) for r in records],
            flush=True,
        )
    (root / "data/mr_qc_reference.json").write_text(
        json.dumps(dict(source=__doc__, generic=generic, molecules=molecular), indent=1)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
