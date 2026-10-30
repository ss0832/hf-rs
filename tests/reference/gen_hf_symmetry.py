# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Physical determinants for all eight Fukutome classes and global spin frames.

Occupied coefficient structures follow Table I of arXiv:1710.00932,
DOI 10.1021/acs.jctc.7b01016. No HF implementation is called.
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, expm

rng = np.random.default_rng(19546)


def random(shape, complex_values=False):
    values = rng.normal(size=shape)
    return values + 1j * rng.normal(size=shape) if complex_values else values


def orthogonal(shape, complex_values=False):
    return np.linalg.qr(random(shape, complex_values))[0]


def main():
    a, b = orthogonal((4, 2)), orthogonal((4, 2))
    z, w = orthogonal((4, 2), True), orthogonal((4, 2), True)
    generators = {
        "TICS": block_diag(a, a),
        "CCW": block_diag(z, z),
        "ASCW": block_diag(z, z.conj()),
        "ASDW": block_diag(a, b),
        "ASW": block_diag(z, w),
        "TSDW": orthogonal((8, 4)),
        "TSW": orthogonal((8, 4), True),
    }
    anti = random((4, 4), True)
    anti = anti - anti.conj().T
    symmetric = random((4, 4), True)
    symmetric = symmetric + symmetric.T
    symplectic = np.block([[anti, symmetric], [-symmetric.conj(), anti.conj()]])
    generators["TSCW"] = expm(0.23 * symplectic) @ block_diag(a, a)
    pauli = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    rotations = [
        np.eye(2),
        *[
            expm(-0.5j * np.einsum("k,kij->ij", vector, pauli))
            for vector in ([0.3, -0.7, 0.2], [-0.9, 0.2, 1.1], [1.4, 0.5, -0.8])
        ],
    ]
    ao = expm(0.13 * random((4, 4)))
    records = []
    for name, occupied in generators.items():
        for frame, rotation in enumerate(rotations):
            c = np.kron(rotation, np.eye(4)) @ occupied
            for metric in (False, True):
                transform = ao if metric else np.eye(4)
                overlap = transform.T @ transform
                orbitals = np.linalg.solve(block_diag(transform, transform), c)
                density = orbitals @ orbitals.conj().T
                s2 = block_diag(overlap, overlap)
                assert np.max(abs(density @ s2 @ density - density)) < 1e-12
                records.append(
                    dict(
                        expected=name,
                        spin_frame=frame,
                        nonorthogonal=metric,
                        overlap=overlap.tolist(),
                        real=density.real.tolist(),
                        imag=density.imag.tolist(),
                    )
                )
    path = Path(__file__).resolve().parents[1] / "data/hf_symmetry_reference.json"
    path.write_text(json.dumps(dict(source=__doc__, cases=records), indent=1) + "\n")
    print(
        "Eight determinant classes x four global SU(2) frames x two AO metrics:",
        len(records),
    )


if __name__ == "__main__":
    main()
