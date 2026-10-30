# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent variational BCS and full-Fock-space oracle for singlet HFB.

The Hamiltonian is the physical electronic Hamiltonian plus -G B^dagger B,
B^dagger = sum_p a^dagger_{p alpha} a^dagger_{p beta} in a real orthonormal frame.
SciPy optimizes canonical orbital rotations and BCS angles at fixed mean number;
it does not iterate the Bogoliubov equations used by hf-rs. An explicit Fock-space
expansion and fermion creation/annihilation algebra independently check the final
energy, every normal/pair-density entry and particle-number variance.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import ao2mo, gto, lib, scf
from scipy.linalg import expm, expm_frechet
from scipy.optimize import least_squares, minimize

from embedded_basis import embedded_basis

lib.num_threads(4)
MODELS = [
    ("H2_normal", [("H", [0.0, 0.0, -0.7]), ("H", [0.0, 0.0, 0.7])], 0, 0.0),
    ("H2_weak", [("H", [0.0, 0.0, -0.7]), ("H", [0.0, 0.0, 0.7])], 0, 0.1),
    ("H2_paired", [("H", [0.0, 0.0, -0.7]), ("H", [0.0, 0.0, 0.7])], 0, 4.0),
    ("H2_stretched", [("H", [0.0, 0.0, -2.0]), ("H", [0.0, 0.0, 2.0])], 0, 2.0),
    (
        "H3_cation",
        [("H", [0.0, 0.0, 0.0]), ("H", [2.1, 0.2, 0.0]), ("H", [0.4, 1.7, 0.3])],
        1,
        1.5,
    ),
    (
        "H4_paired",
        [
            ("H", [0.0, 0.0, 0.0]),
            ("H", [2.3, 0.0, 0.0]),
            ("H", [0.0, 1.8, 0.0]),
            ("H", [2.3, 1.8, 0.2]),
        ],
        0,
        2.0,
    ),
    ("He_full", [("He", [0.0, 0.0, 0.0])], 0, 0.4),
]


def fock_oracle(q, theta, h, eri, g):
    """Expand product_i (u_i + v_i pair_i^dagger) and apply the Hamiltonian."""
    n = len(theta)
    u, v = np.cos(theta), np.sin(theta)
    state = np.zeros(1 << (2 * n))
    for count in range(n + 1):
        subsets = list(itertools.combinations(range(n), count))
        sign = (-1) ** (count * (count - 1) // 2)
        for aa in subsets:
            for bb in subsets:
                amplitude = 0.0
                for occupied in subsets:
                    amplitude += (
                        np.prod([v[i] if i in occupied else u[i] for i in range(n)])
                        * np.linalg.det(q[np.ix_(aa, occupied)])
                        * np.linalg.det(q[np.ix_(bb, occupied)])
                    )
                address = sum(1 << i for i in aa) + sum(1 << (n + i) for i in bb)
                state[address] = sign * amplitude
    assert abs(state @ state - 1) < 1e-12
    active = np.flatnonzero(np.abs(state) > 1e-16)

    def expectation(operators):
        result = 0.0
        for address in active:
            determinant, parity = int(address), 1
            for index, create in operators:
                occupied = (determinant >> index) & 1
                if occupied == create:
                    break
                parity *= (-1) ** bin(determinant & ((1 << index) - 1)).count("1")
                determinant ^= 1 << index
            else:
                result += state[determinant] * parity * state[address]
        return result

    d, k = np.zeros((n, n)), np.zeros((n, n))
    energy, pair_energy = 0.0, 0.0
    for p, r in itertools.product(range(n), repeat=2):
        d[p, r] = expectation([(p, False), (r, True)])
        k[p, r] = expectation([(p, False), (n + r, False)])
        for spin in range(2):
            energy += h[p, r] * expectation(
                [(r + spin * n, False), (p + spin * n, True)]
            )
        pair_energy -= g * expectation(
            [(r, False), (n + r, False), (n + p, True), (p, True)]
        )
    for p, r, q0, s in itertools.product(range(n), repeat=4):
        for a, b in itertools.product(range(2), repeat=2):
            energy += (
                0.5
                * eri[p, r, q0, s]
                * expectation(
                    [
                        (r + a * n, False),
                        (s + b * n, False),
                        (q0 + b * n, True),
                        (p + a * n, True),
                    ]
                )
            )
    numbers = np.array([bin(i).count("1") for i in range(len(state))])
    mean = np.dot(state * state, numbers)
    variance = np.dot(state * state, numbers**2) - mean**2
    return energy + pair_energy, pair_energy, d, k, variance


def run(name, atoms, charge, g):
    mol = gto.M(
        atom=atoms,
        basis=embedded_basis("STO_3G_GBS", sorted({a for a, _ in atoms})),
        charge=charge,
        cart=True,
        unit="Bohr",
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-11, max_cycle=300)
    assert mf.converged
    c = mf.mo_coeff
    h = c.T @ mf.get_hcore() @ c
    n = len(h)
    eri = ao2mo.restore(1, ao2mo.full(mol, c), n)
    pairs = list(itertools.combinations(range(n), 2))
    generators = []
    for i, j in pairs:
        a = np.zeros((n, n))
        a[i, j], a[j, i] = 1.0, -1.0
        generators.append(a)

    def matrices(x):
        a = sum(
            (value * generator for value, generator in zip(x[n:], generators)),
            start=np.zeros((n, n)),
        )
        q = expm(a)
        occ, kap = np.sin(x[:n]) ** 2, np.sin(2 * x[:n]) / 2
        return a, q, q @ np.diag(occ) @ q.T, q @ np.diag(kap) @ q.T, occ, kap

    def evaluate(x):
        a, q, d, k, occ, kap = matrices(x)
        j = np.einsum("pqrs,rs->pq", eri, d)
        kd = np.einsum("prqs,rs->pq", eri, d)
        kk = np.einsum("prqs,rs->pq", eri, k)
        f = h + 2 * j - kd - g * d
        gap = kk - g * np.trace(k) * np.eye(n)
        energy = (
            np.sum(d * (2 * h + 2 * j - kd))
            + np.sum(k * kk)
            - g * (np.sum(d * d) + np.trace(k) ** 2)
        )
        gradient = np.empty(len(x))
        gradient[:n] = 2 * np.diag(q.T @ f @ q) * np.sin(2 * x[:n]) + 2 * np.diag(
            q.T @ gap @ q
        ) * np.cos(2 * x[:n])
        q_gradient = 4 * (f @ q * occ + gap @ q * kap)
        for i, generator in enumerate(generators):
            gradient[n + i] = np.sum(
                q_gradient * expm_frechet(a, generator, compute_expm=False)
            )
        return energy, gradient, f, gap

    def constraint(x):
        return 2 * np.sum(np.sin(x[:n]) ** 2) - mol.nelectron

    def constraint_gradient(x):
        return np.r_[2 * np.sin(2 * x[:n]), np.zeros(len(pairs))]

    start = np.r_[
        np.full(n, np.arcsin(np.sqrt(mol.nelectron / (2 * n)))), np.zeros(len(pairs))
    ]
    if g <= 0.1:
        start[:n] = 0.0
        start[: mol.nelectron // 2] = np.pi / 2
    opt = minimize(
        lambda x: evaluate(x)[:2],
        start,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, np.pi / 2)] * n + [(None, None)] * len(pairs),
        constraints=[dict(type="eq", fun=constraint, jac=constraint_gradient)],
        options=dict(ftol=1e-13, maxiter=500),
    )
    grad_n = constraint_gradient(opt.x)
    f = evaluate(opt.x)[2]
    occ = mol.nelectron // 2
    mu = (
        float(np.diag(f)[:occ].max() + 1)
        if occ == n
        else float((np.linalg.eigvalsh(f)[occ - 1] + np.linalg.eigvalsh(f)[occ]) / 2)
    )
    if grad_n @ grad_n > 1e-10:
        mu = float(evaluate(opt.x)[1] @ grad_n / (grad_n @ grad_n))

    def stationarity(y):
        x, chemical = y[:-1], y[-1]
        return np.r_[evaluate(x)[1] - chemical * constraint_gradient(x), constraint(x)]

    polished = least_squares(
        stationarity, np.r_[opt.x, mu], ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=300
    )
    x, mu = polished.x[:-1], polished.x[-1]
    energy, _, f, gap = evaluate(x)
    assert energy <= evaluate(start)[0] + 1e-10, (name, energy, evaluate(start)[0])
    _, q, d, k, _, _ = matrices(x)
    assert np.max(np.abs(stationarity(polished.x))) < 1e-9
    exact_e, pair_e, exact_d, exact_k, variance = fock_oracle(q, x[:n], h, eri, g)
    assert abs(energy - exact_e) < 1e-10
    assert np.max(np.abs(d - exact_d)) < 1e-12 and np.max(np.abs(k - exact_k)) < 1e-12
    assert abs(variance - 4 * np.sum(k * k)) < 1e-10
    if np.trace(k) < 0:
        k, gap = -k, -gap
    r = np.block([[d, k], [k, np.eye(n) - d]])
    bdg = np.block([[f - mu * np.eye(n), gap], [gap, -f + mu * np.eye(n)]])
    residual = np.max(np.abs(bdg @ r - r @ bdg))
    assert residual < 1e-9
    eigenvalues, vectors = np.linalg.eigh(bdg)
    assert np.max(np.abs(r - vectors[:, :n] @ vectors[:, :n].T)) < 1e-8
    out = dict(
        name=name,
        atoms=[dict(symbol=s, position=r) for s, r in atoms],
        charge=charge,
        pairing_strength=g,
        energy=float(energy + mol.energy_nuc()),
        rhf_energy=float(mf.e_tot),
        pairing_energy=float(pair_e),
        normal_density_ao=(2 * c @ d @ c.T).tolist(),
        pairing_tensor_ao=(c @ k @ c.T).tolist(),
        particle_number_variance=float(variance),
        chemical_potential=float(mu),
        quasiparticle_energies=eigenvalues[n:].tolist(),
        pairing_gap=float(np.max(np.abs(np.linalg.eigvalsh(gap)))),
        full_fock_energy_error=float(abs(exact_e - energy)),
        stationarity_error=float(residual),
    )
    print(
        name,
        "energy",
        out["energy"],
        "variance",
        variance,
        "residual",
        residual,
        flush=True,
    )
    return out


def main():
    out = dict(
        source=__doc__,
        pyscf_version=pyscf.__version__,
        cases=[run(*model) for model in MODELS],
    )
    path = Path(__file__).resolve().parents[1] / "data/hfb_reference.json"
    path.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
