# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""All nuclear coordinates for three MR theories and three relaxation modes."""

import json
from pathlib import Path

import numpy as np
from gen_mr_molecular import models, molecule
from gen_mr_relaxed_molecular import relax


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
                problem = molecule(model, shifted, return_problem=True)
                energies = []
                for order in (2, 3, 4):
                    solution = relax(model, order, "iterate", shifted, problem)
                    history = solution["history"]
                    energies.extend(
                        [
                            history[0]["energy"],
                            history[min(1, len(history) - 1)]["energy"],
                            history[-1]["energy"],
                        ]
                    )
                values.append(energies)
            gradient = np.array([1, -8, 8, -1]) @ values / (12 * step)
            gradients.append(gradient.tolist())
            print(model["name"], component, gradient, flush=True)
            # Save completed coordinates as a recoverable checkpoint.
            partial = dict(
                model_index=index,
                step=step,
                completed_coordinates=component + 1,
                gradient=np.array(gradients).T.tolist(),
            )
            Path("target/mr-relaxed-gradient-partial.json").write_text(
                json.dumps(partial, indent=1) + "\n", encoding="utf-8"
            )
        records.append(
            dict(model_index=index, step=step, gradient=np.array(gradients).T.tolist())
        )
    Path(__file__).resolve().parents[1].joinpath(
        "data/mr_relaxed_gradient_reference.json"
    ).write_text(
        json.dumps(
            dict(
                source=__doc__,
                order=[2, 3, 4],
                modes=["once", "twice", "iterate"],
                cases=records,
            ),
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
