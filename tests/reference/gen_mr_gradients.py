# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Every MR nuclear coordinate: independent five-point total-energy stencils."""

import json
from pathlib import Path

import numpy as np
from gen_mr_molecular import models, molecule


def main():
    records = []
    step = 1e-3
    for index in (0, 2):
        model = models()[index]
        coordinates = np.array([a["position"] for a in model["atoms"]])
        gradients = []
        for component in range(coordinates.size):
            values = []
            for offset in (-2, -1, 1, 2):
                shifted = coordinates.copy()
                shifted.flat[component] += offset * step
                r = molecule(model, shifted)
                values.append(
                    [
                        r["e_cas"] + r["pt2"],
                        r["e_cas"] + r["pt2"] + r["pt3"],
                        r["transformed"]["zero"],
                    ]
                )
            gradient = np.array([1, -8, 8, -1]) @ values / (12 * step)
            gradients.append(gradient.tolist())
            print(model["name"], component, gradient, flush=True)
        records.append(
            dict(model_index=index, step=step, gradient=np.array(gradients).T.tolist())
        )
    Path(__file__).resolve().parents[1].joinpath(
        "data/mr_gradient_reference.json"
    ).write_text(
        json.dumps(dict(source=__doc__, cases=records), indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
