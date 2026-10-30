# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent radial 1s/1s^2 references: SciPy tridiagonal grids and continuum BVP.

Finite-grid Coulomb fields use a dense max(r_i,r_j) kernel, independent of hf-rs
prefix sums. The continuum two-electron oracle solves the coupled radial orbital,
Poisson and normalization boundary-value equations with an unknown eigenvalue;
it does not perform the grid SCF used in production. Only mathematical library
interfaces are used, with no external HF implementation source.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import simpson, solve_bvp
from scipy.linalg import eigh_tridiagonal


def discrete(z, ne, n):
    boundary = 20 / (z - (ne - 1) / 2)
    dr = boundary / (n + 1)
    r = np.arange(1, n + 1) * dr
    nuclear = -z / r
    off = np.full(n - 1, -0.5 / dr**2)
    kernel = 1 / np.maximum.outer(r, r)
    orbital = r * np.exp(-z * r)
    orbital /= np.linalg.norm(orbital)
    previous = np.inf
    for iteration in range(200):
        j = kernel @ (orbital**2)
        diagonal = 1 / dr**2 + nuclear + (ne - 1) * j
        _, vectors = eigh_tridiagonal(
            diagonal,
            off,
            select="i",
            select_range=(0, 0),
            tol=1e-13,
            lapack_driver="stebz",
        )
        new = vectors[:, 0]
        if np.sum(new) < 0:
            new = -new
        change = np.linalg.norm(new - orbital)
        orbital = new
        j = kernel @ (orbital**2)
        t = (2 * np.dot(orbital, orbital) - 2 * np.dot(orbital[1:], orbital[:-1])) / (
            2 * dr**2
        )
        v = np.dot(orbital**2, nuclear)
        pair = np.dot(orbital**2, j)
        energy = ne * (t + v) + ne * (ne - 1) * pair / 2
        if abs(energy - previous) < 2e-12 and change < 1e-11:
            break
        previous = energy
    else:
        raise RuntimeError((z, ne, n, change))
    return dict(
        z=z,
        electrons=ne,
        n_grid=n,
        r_max=boundary,
        energy=float(energy),
        orbital_energy=float(t + v + (ne - 1) * pair),
        radial_wavefunction=(orbital / np.sqrt(dr)).tolist(),
    )


def continuum(z):
    boundary = 20 / (z - 0.5)
    r = boundary * np.linspace(0, 1, 501) ** 2
    charge = z - 5 / 16
    p = 2 * charge**1.5 * r * np.exp(-charge * r)
    dp = 2 * charge**1.5 * (1 - charge * r) * np.exp(-charge * r)
    u = 1 - (1 + charge * r) * np.exp(-2 * charge * r)
    du = charge * (1 + 2 * charge * r) * np.exp(-2 * charge * r)
    probability = 1 - np.exp(-2 * charge * r) * (
        1 + 2 * charge * r + 2 * charge**2 * r**2
    )
    initial = np.array([p, dp, u, du, probability])

    def differential(r, y, energy):
        pair = np.divide(y[0] * y[2], r, out=np.zeros_like(r), where=r != 0)
        density = np.divide(y[0] ** 2, r, out=np.zeros_like(r), where=r != 0)
        return np.array(
            [y[1], 2 * pair - 2 * energy[0] * y[0], y[3], -density, y[0] ** 2]
        )

    def boundary_conditions(left, right, energy):
        return np.array(
            [left[0], right[0], left[2], right[2] - 1, left[4], right[4] - 1]
        )

    singular = np.zeros((5, 5))
    singular[1, 0] = -2 * z
    epsilon = charge**2 / 2 - z * charge + 5 * charge / 8
    solution = solve_bvp(
        differential,
        boundary_conditions,
        r,
        initial,
        p=[epsilon],
        S=singular,
        tol=2e-10,
        max_nodes=30000,
    )
    assert solution.success, (z, solution.message)
    fine = np.linspace(0, boundary, 20001)
    p, dp, u, du, probability = solution.sol(fine)
    j_integrand = np.divide(p**2 * u, fine, out=np.zeros_like(fine), where=fine != 0)
    pair = simpson(j_integrand, x=fine)
    norm = simpson(p**2, x=fine)
    assert abs(norm - 1) < 1e-9
    energy = float(2 * solution.p[0] - pair)
    result = dict(
        z=z,
        electrons=2,
        r_max=boundary,
        energy=energy,
        orbital_energy=float(solution.p[0]),
        normalization=float(norm),
        bvp_max_residual=float(np.max(solution.rms_residuals)),
    )
    print("Continuum BVP", result, flush=True)
    return result


def main():
    cases = [
        discrete(*model)
        for model in [
            (1, 1, 128),
            (1, 1, 512),
            (2, 2, 128),
            (2, 2, 512),
            (3, 2, 192),
            (4, 2, 192),
            (10, 1, 128),
        ]
    ]
    continuous = [continuum(z) for z in [2, 3, 4]]
    out = dict(source=__doc__, grid_cases=cases, continuum_cases=continuous)
    path = Path(__file__).resolve().parents[1] / "data/numerical_atomic_reference.json"
    path.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
