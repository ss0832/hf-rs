# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Generates high-precision Boys function reference values with mpmath.

F_n(T) = gamma_lower(n + 1/2, T) / (2 T^(n + 1/2)),  F_n(0) = 1 / (2n + 1)
Usage: python gen_boys.py > tests/data/boys_reference.json
"""

import json

import mpmath

mpmath.mp.dps = 50

T_VALUES = [
    0.0,
    1e-14,
    1e-8,
    1e-4,
    0.013,
    0.05,
    0.1234,
    0.5,
    0.999,
    1.0,
    2.5,
    3.7777,
    7.0,
    10.0,
    12.345,
    17.5,
    23.9,
    29.99,
    34.99,
    35.01,
    40.0,
    44.4,
    49.97,
    50.0,
    50.03,
    55.5,
    64.0,
    80.0,
    100.0,
    137.0,
    200.0,
]


def boys(n, t):
    if t == 0:
        return mpmath.mpf(1) / (2 * n + 1)
    a = mpmath.mpf(n) + mpmath.mpf(1) / 2
    return mpmath.gammainc(a, 0, t) / (2 * mpmath.power(t, a))


rows = []
for t in T_VALUES:
    tt = mpmath.mpf(repr(t))
    rows.append(
        {
            "t": t,
            "f": [
                mpmath.nstr(boys(n, tt), 25, min_fixed=-1, max_fixed=-1)
                for n in range(41)
            ],
        }
    )

print(
    json.dumps(
        {"source": f"mpmath {mpmath.__version__}, dps=50", "max_n": 40, "rows": rows},
        indent=1,
    )
)
