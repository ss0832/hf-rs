# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""High-precision congruence references with CABS-like cancellation."""

import json
from pathlib import Path

import mpmath as mp
import numpy as np

mp.mp.dps = 70
n, k = 32, 18
coefficients = np.array(
    [
        [(((i + 3) * (p + 5)) % 19 - 9) * 125.123456789 for p in range(k)]
        for i in range(n)
    ]
)
coefficients[-1] -= coefficients.sum(axis=0)
cm = mp.matrix([[mp.mpf(float(x)) for x in row] for row in coefficients])
cases = []
for rank_weight in [1.0, -0.3]:
    operator = np.full((n, n), rank_weight) + np.diag(np.arange(1, n + 1) * 2.0**-30)
    am = mp.matrix([[mp.mpf(float(x)) for x in row] for row in operator])
    reference = cm.T * am * cm
    reference = np.array(reference.tolist(), dtype=float)
    ordinary = coefficients.T @ operator @ coefficients
    cases.append(
        dict(
            operator=operator.tolist(),
            coefficients=coefficients.tolist(),
            operator_bits=operator.view(np.uint64).tolist(),
            coefficient_bits=coefficients.view(np.uint64).tolist(),
            reference=reference.tolist(),
            ordinary_maximum_error=float(np.max(abs(ordinary - reference))),
            ordinary_maximum_asymmetry=float(np.max(abs(ordinary - ordinary.T))),
        )
    )
record = dict(
    source="Independent mpmath 70-digit matrix products; input bit patterns preserve the exact binary64 operands across JSON parser rounding modes",
    criterion=1e-10,
    cases=cases,
)
path = (
    Path(__file__).resolve().parents[1] / "data/f12_symmetric_congruence_reference.json"
)
path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
for case in cases:
    print(case["ordinary_maximum_error"], case["ordinary_maximum_asymmetry"])
