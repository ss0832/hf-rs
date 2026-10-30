# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Fock-space projection + Padé matrix exponential oracle for SR-DSRG.

This uses the explicit fermion matrices from gen_imsrg2. The retained-operator
superoperator is exponentiated with SciPy (no nested BCH implementation), and
the driven equations are solved with scipy.optimize.root. Perturbative energy
and amplitude coefficients are extracted by fitting the lambda dependence of
the fully solved equations, rather than evaluating the production PT3 formula.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from gen_imsrg2 import FockOracle
from pyscf import ao2mo, gto, lib, mp, scf
from pyscf.mp.ump2 import energy as ump2_energy
from scipy.linalg import expm
from scipy.optimize import root

from embedded_basis import embedded_basis

lib.num_threads(4)


class DrivenOracle:
    def __init__(self, occupied):
        self.fock = o = FockOracle(occupied)
        n, npair = o.n, o.np
        self.size = 1 + n * n + npair * npair
        # Explicit physical Fock-space matrices of every retained unit operator.
        self.basis = np.array([o.matrix(v) for v in np.eye(self.size)])
        holes, particles = np.flatnonzero(occupied), np.flatnonzero(~o.occupied)
        indices = [1 + a * n + i for a in particles for i in holes]
        gaps = [(i, a) for a in particles for i in holes]
        pairs = {p: k for k, p in enumerate(o.pairs)}
        for a, b in itertools.combinations(particles, 2):
            for i, j in itertools.combinations(holes, 2):
                indices.append(1 + n * n + pairs[a, b] * npair + pairs[i, j])
                gaps.append((i, j, a, b))
        self.indices = np.array(indices)
        self.gaps = gaps
        self.generators = []
        self.superoperators = []
        for index in indices:
            a = np.zeros(self.size)
            a[index] = 1.0
            f = a[1 : 1 + n * n].reshape(n, n)
            f -= f.T.copy()
            g = a[1 + n * n :].reshape(npair, npair)
            g -= g.T.copy()
            self.generators.append(a)
            am = o.matrix(a)
            self.superoperators.append(
                self.project_batch(self.basis @ am - am @ self.basis).T
            )
        self.generators = np.array(self.generators)
        self.superoperators = np.array(self.superoperators)

    def project_batch(self, matrices):
        o = self.fock
        e = matrices[:, 0, 0]
        one = matrices[:, o.sectors[1], :][:, :, o.sectors[1]]
        one = one - e[:, None, None] * np.eye(o.n)
        t = one.reshape(len(e), -1)
        two = matrices[:, o.sectors[2], :][:, :, o.sectors[2]]
        two = two - e[:, None, None] * np.eye(o.np)
        two -= (t @ o.one_sectors[2].reshape(o.n**2, -1)).reshape(two.shape)
        g = two.reshape(len(e), -1)
        three = matrices[:, o.sectors[3], :][:, :, o.sectors[3]]
        three = three - e[:, None, None] * np.eye(o.nt)
        three -= (t @ o.one_sectors[3].reshape(o.n**2, -1)).reshape(three.shape)
        three -= (g @ o.two_sector3.reshape(o.np**2, -1)).reshape(three.shape)
        w = three.reshape(len(e), -1)
        return np.column_stack(
            (
                e + t @ o.e1 + g @ o.e2 + w @ o.e3,
                t + g @ o.f2.T + w @ o.f3.T,
                g + w @ o.g3.T,
            )
        )

    def transform(self, h, amplitudes):
        linear_map = np.einsum("a,aij->ij", amplitudes, self.superoperators)
        return expm(linear_map) @ h

    def solve(self, h, flow):
        f = h[1 : 1 + self.fock.n**2].reshape(self.fock.n, self.fock.n)
        eps = np.diag(f)
        gaps = np.array(
            [
                eps[g[0]] - eps[g[1]]
                if len(g) == 2
                else eps[g[0]] + eps[g[1]] - eps[g[2]] - eps[g[3]]
                for g in self.gaps
            ]
        )
        regulator = np.divide(
            -np.expm1(-flow * gaps**2), gaps, out=np.zeros_like(gaps), where=gaps != 0
        )

        def residual(t):
            bar = self.transform(h, t)
            return (bar[self.indices] + gaps * t) * regulator - t

        answer = root(residual, h[self.indices] * regulator, tol=2e-11)
        error = float(np.max(abs(residual(answer.x))))
        assert error < 2e-12, (answer.message, error)
        return answer.x, self.transform(h, answer.x), error

    def perturbation_coefficients(self, h, flow, step):
        h0 = h * 0.0
        n = self.fock.n
        diag = np.diag(h[1 : 1 + n * n].reshape(n, n))
        h0[1 : 1 + n * n] = np.diag(diag).ravel()
        values, generators = [], []
        # Ninth-degree polynomial cancellation of higher-order contamination.
        points = np.arange(-5, 6)
        for point in points:
            t, bar, _ = self.solve(h0 + point * step * (h - h0), flow)
            values.append(bar[0])
            generators.append(t @ self.generators)
        vander = np.polynomial.polynomial.polyvander(points, 10)
        ecoeff = np.linalg.solve(vander, values)
        acoeff = np.linalg.solve(vander, generators)
        return np.array([ecoeff[2] / step**2, ecoeff[3] / step**3]), (
            acoeff[1] / step,
            acoeff[2] / step**2,
        )


def record(oracle, h, flow):
    t, bar, error = oracle.solve(h, flow)
    energy, (a1, a2) = oracle.perturbation_coefficients(h, flow, 0.05)
    refined, (fine1, fine2) = oracle.perturbation_coefficients(h, flow, 0.025)
    refinement = max(
        np.max(abs(energy - refined)), np.max(abs(a1 - fine1)), np.max(abs(a2 - fine2))
    )
    assert refinement < 2e-10, refinement
    print(
        "flow",
        flow,
        "energy",
        bar[0],
        "PT2/PT3",
        refined,
        "refinement",
        refinement,
        flush=True,
    )
    return dict(
        flow=flow,
        initial=oracle.fock.full(h),
        generator=oracle.fock.full(t @ oracle.generators),
        transformed=oracle.fock.full(bar),
        second_order=float(refined[0]),
        third_order=float(refined[1]),
        first_order_generator=oracle.fock.full(fine1),
        second_order_generator=oracle.fock.full(fine2),
        residual=error,
        polynomial_refinement=float(refinement),
    )


def molecular_operator(atoms, basis, ecp=None, charge=0):
    mol = gto.M(
        atom=atoms,
        unit="Bohr",
        basis=basis,
        ecp=ecp or {},
        charge=charge,
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-11)
    assert mf.converged
    nm = mf.mo_coeff.shape[1]
    spatial = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), nm)
    occupied = np.arange(2 * nm) < mol.nelectron
    oracle = DrivenOracle(occupied)
    fock = mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff
    one, two = np.zeros((2 * nm,) * 2), np.zeros((2 * nm,) * 4)
    for p, q in itertools.product(range(2 * nm), repeat=2):
        if p % 2 == q % 2:
            one[p, q] = fock[p // 2, q // 2]
    for p, q, r, s in itertools.product(range(2 * nm), repeat=4):
        two[p, q, r, s] = (
            spatial[p // 2, r // 2, q // 2, s // 2]
            if p % 2 == r % 2 and q % 2 == s % 2
            else 0.0
        ) - (
            spatial[p // 2, s // 2, q // 2, r // 2]
            if p % 2 == s % 2 and q % 2 == r % 2
            else 0.0
        )
    h = np.concatenate(
        (
            [0.0],
            one.ravel(),
            [
                two[p, q, r, s]
                for p, q in oracle.fock.pairs
                for r, s in oracle.fock.pairs
            ],
        )
    )
    return mf, oracle, h


def molecular(name, atoms, basis, gaussian94=None, ecp=None, charge=0):
    mf, oracle, h = molecular_operator(atoms, basis, ecp, charge)
    mol = mf.mol
    nm = mf.mo_coeff.shape[1]
    occupied = oracle.fock.occupied
    one = h[1 : 1 + (2 * nm) ** 2].reshape(2 * nm, 2 * nm)
    cases = [record(oracle, h, flow) for flow in (0.2, 0.5, 2.0)]
    # A true finite-lambda Fock-space diagonalization supplies the MP3 limit.
    h0 = np.zeros_like(h)
    h0[1 : 1 + (2 * nm) ** 2] = np.diag(np.diag(one)).ravel()
    physical0, physical1 = oracle.fock.matrix(h0), oracle.fock.matrix(h - h0)
    sector = [
        bits for bits in range(1 << (2 * nm)) if bits.bit_count() == mol.nelectron
    ]
    m0, m1 = physical0[np.ix_(sector, sector)], physical1[np.ix_(sector, sector)]
    step = 0.025
    energies = [np.linalg.eigvalsh(m0 + x * step * m1)[0] for x in range(-5, 6)]
    coeff = np.linalg.solve(
        np.polynomial.polynomial.polyvander(np.arange(-5, 6), 10), energies
    )
    mp2 = mp.MP2(mf).run().e_corr
    assert abs(coeff[2] / step**2 - mp2) < 2e-10
    return dict(
        name=name,
        atoms=[
            dict(symbol=mol.atom_symbol(i), position=mol.atom_coord(i).tolist())
            for i in range(mol.natm)
        ],
        gaussian94=gaussian94,
        charge=charge,
        ecp=ecp,
        e_scf=float(mf.e_tot),
        mo_coeff=mf.mo_coeff.tolist(),
        occupied=occupied.tolist(),
        cases=cases,
        mp2=float(mp2),
        mp3=float(coeff[3] / step**3),
    )


def molecular_gradient(model_index, atoms, basis, ecp=None, charge=0):
    coordinates = np.array([a[1] for a in atoms])
    gradients = {"ldsrg": [], "pt3": []}
    step = 1e-3
    for component in range(6):
        values = []
        for displacement in (-2, -1, 1, 2):
            shifted = coordinates.copy()
            shifted.flat[component] += displacement * step
            mf, oracle, h = molecular_operator(
                [(a[0], x) for a, x in zip(atoms, shifted, strict=True)],
                basis,
                ecp,
                charge,
            )
            _, bar, _ = oracle.solve(h, 0.5)
            coeff, _ = oracle.perturbation_coefficients(h, 0.5, 0.025)
            values.append([mf.e_tot + bar[0], mf.e_tot + sum(coeff)])
        derivative = np.array([1, -8, 8, -1]) @ values / (12 * step)
        gradients["ldsrg"].append(float(derivative[0]))
        gradients["pt3"].append(float(derivative[1]))
    return dict(model_index=model_index, step=step, **gradients)


def unrestricted_pt2():
    atoms = [("He", [0.13, -0.21, 0.31]), ("H", [-0.18, 0.17, 1.81])]
    ecp = {
        "He": [
            0,
            [
                [-1, [[], [], [[0.7, 0.17]]]],
                [0, [[], [], [[0.53, 0.11]]]],
                [1, [[], [], [[0.61, -0.07]]]],
            ],
        ]
    }
    basis = {
        "He": [[0, [0.2, 1.0]], [0, [0.9, 1.0]], [0, [3.2, 1.0]]],
        "H": [[0, [0.9, 1.0]]],
    }
    mol = gto.M(
        atom=atoms, basis=basis, ecp=ecp, unit="Bohr", spin=1, cart=True, verbose=0
    )
    mf = scf.UHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-11, max_cycle=300)
    assert mf.converged
    perturbation = mp.UMP2(mf)
    eris = perturbation.ao2mo()
    _, t2 = perturbation.kernel(eris=eris)
    gaps = [
        eps[:no, None] - eps[None, no:]
        for eps, no in zip(mf.mo_energy, perturbation.nocc, strict=True)
    ]
    denominators = [
        gaps[a][:, None, :, None] + gaps[b][None, :, None, :]
        for a, b in [(0, 0), (0, 1), (1, 1)]
    ]
    cases = []
    for flow in (0.2, 0.5, 2.0, 1e6):
        scaled = tuple(
            t * (-np.expm1(-2 * flow * d * d))
            for t, d in zip(t2, denominators, strict=True)
        )
        energy = ump2_energy(perturbation, scaled, eris)
        cases.append(
            dict(
                flow=flow,
                energy=float(energy),
                same_spin=float(energy.e_corr_ss),
                opposite_spin=float(energy.e_corr_os),
            )
        )
    assert abs(cases[1]["same_spin"]) > 1e-8
    return dict(
        name="HeH doublet/ECP",
        atoms=[dict(symbol=a, position=x) for a, x in atoms],
        charge=0,
        ecp=ecp,
        gaussian94="He 0\nS 1 1.0\n0.2 1.0\nS 1 1.0\n0.9 1.0\nS 1 1.0\n3.2 1.0\n****\nH 0\nS 1 1.0\n0.9 1.0\n****\n",
        e_scf=float(mf.e_tot),
        cases=cases,
    )


def main():
    rng = np.random.default_rng(20261023)
    occupied = [True, False, True, False, False, True]
    oracle = DrivenOracle(occupied)
    f = rng.normal(scale=0.02, size=(6, 6))
    f = f + f.T + np.diag([-1.2, 0.7, -0.8, 0.9, 1.2, -0.4])
    g = rng.normal(scale=0.02, size=(15, 15))
    h = np.concatenate(([0.0], f.ravel(), (g + g.T).ravel()))
    batch = oracle.project_batch(
        oracle.basis @ oracle.fock.matrix(h) - oracle.fock.matrix(h) @ oracle.basis
    )
    for index in [1, 17, 150]:
        direct, _ = oracle.fock.commutator(np.eye(oracle.size)[index], h)
        assert np.max(abs(batch[index] - direct)) < 1e-12
    generic = record(oracle, h, 0.7)
    generic["occupied"] = occupied
    # A non-excitation, non-small anti-Hermitian generator probes every BCH term.
    f = rng.normal(scale=0.06, size=(6, 6))
    g = rng.normal(scale=0.04, size=(15, 15))
    a = np.concatenate(([0.0], (f - f.T).ravel(), (g - g.T).ravel()))
    am = oracle.fock.matrix(a)
    linear = oracle.project_batch(oracle.basis @ am - am @ oracle.basis).T
    bch = dict(
        occupied=occupied,
        h=oracle.fock.full(h),
        a=oracle.fock.full(a),
        bar=oracle.fock.full(expm(linear) @ h),
    )
    helium = "He 0\nS 1 1.0\n0.2 1.0\nS 1 1.0\n0.9 1.0\nS 1 1.0\n3.2 1.0\n****\n"
    ecp = {
        "He": [
            0,
            [
                [-1, [[], [], [[0.7, 0.17]]]],
                [0, [[], [], [[0.53, 0.11]]]],
                [1, [[], [], [[0.61, -0.07]]]],
            ],
        ]
    }
    molecules = [
        molecular(
            "H2/STO-3G",
            "H 0.1 -0.2 -0.7; H -0.1 0.2 0.7",
            embedded_basis("STO_3G_GBS", ["H"]),
        ),
        molecular(
            "He/3s",
            "He 0.0 0.0 0.0",
            {"He": [[0, [0.2, 1.0]], [0, [0.9, 1.0]], [0, [3.2, 1.0]]]},
            helium,
        ),
    ]
    molecules.append(
        molecular(
            "He/3s ECP",
            "He 0.13 -0.21 0.31",
            {"He": [[0, [0.2, 1.0]], [0, [0.9, 1.0]], [0, [3.2, 1.0]]]},
            helium,
            ecp,
        )
    )
    diatomic_atoms = [("He", [0.13, -0.21, 0.31]), ("H", [-0.18, 0.17, 1.81])]
    diatomic_basis = {"He": [[0, [1.4, 1.0]], [0, [0.45, 1.0]]], "H": [[0, [0.9, 1.0]]]}
    diatomic_g94 = (
        "He 0\nS 1 1.0\n1.4 1.0\nS 1 1.0\n0.45 1.0\n****\nH 0\nS 1 1.0\n0.9 1.0\n****\n"
    )
    molecules.append(
        molecular("HeH+/ECP", diatomic_atoms, diatomic_basis, diatomic_g94, ecp, 1)
    )
    gradients = [
        molecular_gradient(
            0,
            [("H", [0.1, -0.2, -0.7]), ("H", [-0.1, 0.2, 0.7])],
            embedded_basis("STO_3G_GBS", ["H"]),
        ),
        molecular_gradient(3, diatomic_atoms, diatomic_basis, ecp, 1),
    ]
    output = dict(
        source="Explicit full Fock-space projection; SciPy Padé superoperator exponential and root solver; polynomial lambda derivatives; PySCF molecular integrals and MP2; finite-lambda FCI MP3",
        generic=generic,
        uhf_pt2=unrestricted_pt2(),
        gradients=gradients,
        bch=bch,
        molecules=molecules,
    )
    Path(__file__).resolve().parents[1].joinpath(
        "data/sr_dsrg_reference.json"
    ).write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
