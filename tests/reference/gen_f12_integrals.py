# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent bivariate Gauss-Hermite + radial Gauss-Legendre F12 integrals.

Each Cartesian axis integrates the original coupled Gaussian polynomial directly;
no Hermite-derivative or Boys recurrence is used by this oracle.
"""

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

quadratures = {n: leggauss(n) for n in [128, 256]}
xh, wh = hermgauss(12)
weights = wh[:, None] * wh[None, :] / np.pi
exponents = np.array([0.73, 1.14, 0.62, 1.41])
centers = np.array(
    [
        [0.12, -0.24, 0.37],
        [-0.41, 0.23, 0.19],
        [0.54, -0.13, 0.05],
        [-0.22, -0.32, 0.48],
    ]
)


def moments(powers, gamma):
    gamma = np.atleast_1d(gamma)
    p = exponents[0] + exponents[1]
    q = exponents[2] + exponents[3]
    det = p * q + gamma * (p + q)
    inverse00 = (q + gamma) / det
    inverse01 = gamma / det
    inverse11 = (p + gamma) / det
    l00 = np.sqrt(inverse00)
    l10 = inverse01 / l00
    l11 = np.sqrt(inverse11 - l10 * l10)
    base = []
    r2 = []
    kinetic_commutator = []
    for axis in range(3):
        b0 = exponents[:2] @ centers[:2, axis]
        b1 = exponents[2:] @ centers[2:, axis]
        m0 = inverse00 * b0 + inverse01 * b1
        m1 = inverse01 * b0 + inverse11 * b1
        x = (
            m0[:, None, None]
            + l00[:, None, None] * xh[None, :, None]
            + np.zeros((len(gamma), 1, len(xh)))
        )
        y = (
            m1[:, None, None]
            + l10[:, None, None] * xh[None, :, None]
            + l11[:, None, None] * xh[None, None, :]
        )
        polynomial = np.ones_like(x)
        coordinates = []
        for index in range(4):
            coordinates.append((x if index < 2 else y) - centers[index, axis])
            polynomial *= coordinates[-1] ** powers[index][axis]
        differential = np.zeros_like(x)
        for index, z in enumerate(coordinates):
            degree = powers[index][axis]
            alpha = exponents[index]
            laplacian = (
                4 * alpha**2 * z ** (degree + 2)
                - 2 * alpha * (2 * degree + 1) * z**degree
            )
            if degree >= 2:
                laplacian += degree * (degree - 1) * z ** (degree - 2)
            for other in range(4):
                if other != index:
                    laplacian = laplacian * coordinates[other] ** powers[other][axis]
            differential += (0.5 if index in [0, 2] else -0.5) * laplacian
        prefactor = (
            np.pi
            / np.sqrt(det)
            * np.exp(-np.sum(exponents * centers[:, axis] ** 2) + b0 * m0 + b1 * m1)
        )
        base.append(prefactor * np.sum(polynomial * weights, axis=(1, 2)))
        r2.append(prefactor * np.sum(polynomial * (x - y) ** 2 * weights, axis=(1, 2)))
        kinetic_commutator.append(
            prefactor * np.sum(differential * weights, axis=(1, 2))
        )
    return (
        np.prod(base, axis=0),
        sum(
            r2[axis] * base[(axis + 1) % 3] * base[(axis + 2) % 3] for axis in range(3)
        ),
        sum(
            kinetic_commutator[axis] * base[(axis + 1) % 3] * base[(axis + 2) % 3]
            for axis in range(3)
        ),
    )


def coulomb(powers, gamma, n):
    x, w = quadratures[n]
    u = (x + 1) / 2
    p = exponents[0] + exponents[1]
    q = exponents[2] + exponents[3]
    det = p * q + gamma * (p + q)
    effective = gamma + det / (p + q) * u * u / (1 - u * u)
    jacobian = 2 / np.sqrt(np.pi) * np.sqrt(det / (p + q)) / (1 - u * u) ** 1.5
    return float(np.dot(w / 2 * jacobian, moments(powers, effective)[0]))


cases = []
error = 0.0
for angular in range(8):
    for lx in range(angular, -1, -1):
        for ly in range(angular - lx, -1, -1):
            powers = [[lx, ly, angular - lx - ly], [1, 0, 0], [0, 1, 0], [0, 0, 0]]
            gamma = 0.17 + 0.083 * angular
            gaussian, r2, commutator = moments(powers, gamma)
            coarse = coulomb(powers, gamma, 128)
            fine = coulomb(powers, gamma, 256)
            error = max(error, abs(coarse - fine))
            assert abs(coarse - fine) < 2e-11 * max(1.0, abs(fine))
            cases.append(
                dict(
                    exponents=exponents.tolist(),
                    centers=centers.tolist(),
                    powers=powers,
                    gamma=gamma,
                    gaussian=float(gaussian[0]),
                    r2=float(r2[0]),
                    kinetic_commutator=float(commutator[0]),
                    coulomb=fine,
                    zero_gaussian=float(moments(powers, 0.0)[0][0]),
                    zero_coulomb=coulomb(powers, 0.0, 256),
                )
            )
# Contracted single-primitive test, including distinct terms with equal exponents.
terms = [(0.31, -0.4), (0.92, 0.16), (0.31, 0.09)]
powers = [[2, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
coefficient = 0.7 * (-0.4) * 1.2 * 0.8
values = {}
values["factor"] = coefficient * sum(
    c * float(moments(powers, a)[0][0]) for a, c in terms
)
values["factor_over_r"] = coefficient * sum(
    c * coulomb(powers, a, 256) for a, c in terms
)
values["kinetic_commutator"] = coefficient * sum(
    c * float(moments(powers, a)[2][0]) for a, c in terms
)
values["squared"] = coefficient * sum(
    c * d * float(moments(powers, a + b)[0][0]) for a, c in terms for b, d in terms
)
values["commutator"] = coefficient * sum(
    8 * a * b * c * d * float(moments(powers, a + b)[1][0])
    for a, c in terms
    for b, d in terms
)
out = dict(
    source=f"NumPy {np.__version__} independent direct bivariate GH-12 polynomial quadrature and GL-256 Coulomb radial integration",
    coulomb_quadrature_convergence=error,
    cases=cases,
    contracted=dict(
        exponents=exponents.tolist(),
        centers=centers.tolist(),
        powers=powers,
        coefficients=[0.7, -0.4, 1.2, 0.8],
        terms=terms,
        expected=values,
    ),
)
Path(__file__).resolve().parents[1].joinpath(
    "data/f12_integral_reference.json"
).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print("Cartesian cases", len(cases), "radial convergence", error)

# All four orbitals can have L=7. GH-20 integrates their degree-28 polynomial
# and the kinetic degree-30 polynomial exactly, independently of the recurrence.
xh, wh = hermgauss(20)
weights = wh[:, None] * wh[None, :] / np.pi
high_degree = []
for powers in [
    [[7, 0, 0]] * 4,
    [[3, 2, 2]] * 4,
    [[7, 0, 0], [0, 7, 0], [0, 0, 7], [3, 2, 2]],
    [[1, 3, 3], [3, 1, 3], [3, 3, 1], [2, 2, 3]],
]:
    gamma = 0.43
    gaussian, r2, commutator = moments(powers, gamma)
    coarse = coulomb(powers, gamma, 128)
    fine = coulomb(powers, gamma, 256)
    assert abs(coarse - fine) < 2e-11 * max(1.0, abs(fine))
    high_degree.append(
        dict(
            exponents=exponents.tolist(),
            centers=centers.tolist(),
            powers=powers,
            gamma=gamma,
            gaussian=float(gaussian[0]),
            r2=float(r2[0]),
            kinetic_commutator=float(commutator[0]),
            coulomb=fine,
            zero_gaussian=float(moments(powers, 0.0)[0][0]),
            zero_coulomb=coulomb(powers, 0.0, 256),
        )
    )
Path(__file__).resolve().parents[1].joinpath(
    "data/f12_high_degree_reference.json"
).write_text(
    json.dumps(
        dict(
            source="Independent GH-20 / GL-256 polynomial quadrature", cases=high_degree
        ),
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
print("Four-L=7 quartet cases", len(high_degree))
