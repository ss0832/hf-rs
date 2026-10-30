# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent Fock-matrix commutators, rank-three extraction and NO2B flow.

The oracle constructs full creation/annihilation matrices, extracts the bare
zero/one/two/three-body coefficients, contracts the induced three-body operator
with the reference, and only then discards its normal-ordered rank-three part.
No production occupation-index commutator formula is implemented here.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from pyscf import ao2mo, gto, lib, scf
from scipy.integrate import solve_ivp

from embedded_basis import embedded_basis

lib.num_threads(4)


class FockOracle:
    def __init__(self, occupied):
        self.n = n = len(occupied)
        self.occupied = np.array(occupied)
        self.pairs = list(itertools.combinations(range(n), 2))
        self.triples = list(itertools.combinations(range(n), 3))
        self.np, self.nt = len(self.pairs), len(self.triples)
        dim = 1 << n
        self.identity = np.eye(dim)
        create = []
        for p in range(n):
            op = np.zeros((dim, dim))
            for bits in range(dim):
                if not bits >> p & 1:
                    op[bits | (1 << p), bits] = (-1) ** (
                        (bits & ((1 << p) - 1)).bit_count()
                    )
            create.append(op)
        self.one = np.array(
            [create[p] @ create[q].T for p in range(n) for q in range(n)]
        )
        self.two = np.array(
            [
                create[p] @ create[q] @ create[s].T @ create[r].T
                for p, q in self.pairs
                for r, s in self.pairs
            ]
        )
        self.sectors = [
            [
                sum(1 << i for i in group)
                for group in itertools.combinations(range(n), rank)
            ]
            for rank in range(4)
        ]
        self.one_sectors = {
            rank: self.one[:, indices, :][:, :, indices]
            for rank, indices in enumerate(self.sectors)
            if rank > 0
        }
        indices = self.sectors[3]
        self.two_sector3 = self.two[:, indices, :][:, :, indices]
        pindex = {p: i for i, p in enumerate(self.pairs)}
        tindex = {p: i for i, p in enumerate(self.triples)}

        def element(group1, group2, index):
            if len(set(group1)) < len(group1) or len(set(group2)) < len(group2):
                return 0, 0
            sign = (-1) ** sum(
                group[i] > group[j]
                for group in (group1, group2)
                for i in range(len(group))
                for j in range(i + 1, len(group))
            )
            length = len(index)
            return index[tuple(sorted(group1))] * length + index[
                tuple(sorted(group2))
            ], sign

        self.e1 = np.zeros(n * n)
        self.e2 = np.zeros(self.np**2)
        self.e3 = np.zeros(self.nt**2)
        self.f2 = np.zeros((n * n, self.np**2))
        self.f3 = np.zeros((n * n, self.nt**2))
        self.g3 = np.zeros((self.np**2, self.nt**2))
        holes = np.flatnonzero(occupied)
        for i in holes:
            self.e1[i * n + i] += 1
            for j in holes:
                address, sign = element((i, j), (i, j), pindex)
                self.e2[address] += sign / 2
                for k in holes:
                    address, sign = element((i, j, k), (i, j, k), tindex)
                    self.e3[address] += sign / 6
        for p in range(n):
            for q in range(n):
                for i in holes:
                    address, sign = element((p, i), (q, i), pindex)
                    self.f2[p * n + q, address] += sign
                    for j in holes:
                        address, sign = element((p, i, j), (q, i, j), tindex)
                        self.f3[p * n + q, address] += sign / 2
        for left, (p, q) in enumerate(self.pairs):
            for right, (r, s) in enumerate(self.pairs):
                for i in holes:
                    address, sign = element((p, q, i), (r, s, i), tindex)
                    self.g3[left * self.np + right, address] += sign

    def unpack(self, coefficients):
        return (
            coefficients[0],
            coefficients[1 : 1 + self.n**2],
            coefficients[1 + self.n**2 :],
        )

    def matrix(self, coefficients):
        e, f, g = self.unpack(coefficients)
        t = f - self.f2 @ g
        bare_e = e - self.e1 @ t - self.e2 @ g
        return (
            bare_e * self.identity
            + np.einsum("a,aij->ij", t, self.one)
            + np.einsum("a,aij->ij", g, self.two)
        )

    def project(self, matrix):
        e = matrix[0, 0]
        one = matrix[np.ix_(self.sectors[1], self.sectors[1])] - e * np.eye(self.n)
        two = matrix[np.ix_(self.sectors[2], self.sectors[2])] - e * np.eye(self.np)
        two -= np.einsum("a,aij->ij", one.ravel(), self.one_sectors[2])
        three = matrix[np.ix_(self.sectors[3], self.sectors[3])] - e * np.eye(self.nt)
        three -= np.einsum("a,aij->ij", one.ravel(), self.one_sectors[3])
        three -= np.einsum("a,aij->ij", two.ravel(), self.two_sector3)
        t, g, w = one.ravel(), two.ravel(), three.ravel()
        return np.concatenate(
            (
                [e + self.e1 @ t + self.e2 @ g + self.e3 @ w],
                t + self.f2 @ g + self.f3 @ w,
                g + self.g3 @ w,
            )
        ), float(np.max(abs(w)))

    def commutator(self, a, b):
        am, bm = self.matrix(a), self.matrix(b)
        return self.project(am @ bm - bm @ am)

    def full(self, coefficients):
        zero, one, pairs = self.unpack(coefficients)
        two = np.zeros((self.n,) * 4)
        for left, (p, q) in enumerate(self.pairs):
            for right, (r, s) in enumerate(self.pairs):
                v = pairs[left * self.np + right]
                two[p, q, r, s] = two[q, p, s, r] = v
                two[q, p, r, s] = two[p, q, s, r] = -v
        return dict(
            zero=float(zero),
            one=one.reshape(self.n, self.n).tolist(),
            two=two.ravel().tolist(),
        )

    def flow(self, coefficients, kind, end):
        mask1 = self.occupied[:, None] != self.occupied[None, :]
        # NumPy boolean addition is logical; count occupied pair members explicitly.
        pair_occ = np.array(
            [int(self.occupied[p]) + int(self.occupied[q]) for p, q in self.pairs]
        )
        mask2 = (pair_occ[:, None] == 0) & (pair_occ[None, :] == 2)
        mask2 |= mask2.T

        def derivative(_, y):
            _, f, g = self.unpack(y)
            od = np.concatenate(([0.0], f * mask1.ravel(), g * mask2.ravel()))
            if kind == "Wegner":
                eta, _ = self.commutator(y - od, od)
            else:
                diag = np.diag(f.reshape(self.n, self.n))
                gap1 = diag[:, None] - diag[None, :]
                pair_e = np.array([diag[p] + diag[q] for p, q in self.pairs])
                gap2 = pair_e[:, None] - pair_e[None, :]
                eta1 = np.divide(
                    f.reshape(self.n, self.n) * mask1,
                    gap1,
                    out=np.zeros_like(gap1),
                    where=mask1,
                )
                eta2 = np.divide(
                    g.reshape(self.np, self.np) * mask2,
                    gap2,
                    out=np.zeros_like(gap2),
                    where=mask2,
                )
                eta = np.concatenate(([0.0], eta1.ravel(), eta2.ravel()))
            return self.commutator(eta, y)[0]

        solution = solve_ivp(
            derivative,
            (0.0, end),
            coefficients,
            method="DOP853",
            rtol=1e-12,
            atol=1e-13,
        )
        assert solution.success
        return solution.y[:, -1]


def main():
    rng = np.random.default_rng(20261023)
    oracle = FockOracle([True, False, True, False, False, True])
    commutators = []
    for anti in (True, False):
        one = rng.normal(size=(6, 6))
        two = rng.normal(size=(15, 15))
        a = np.concatenate(
            (
                [0.1],
                (one - one.T if anti else one + one.T).ravel(),
                (two - two.T if anti else two + two.T).ravel(),
            )
        )
        one, two = rng.normal(size=(6, 6)), rng.normal(size=(15, 15))
        b = np.concatenate(([0.3], (one + one.T).ravel(), (two + two.T).ravel()))
        c, three = oracle.commutator(a, b)
        assert three > 1.0
        commutators.append(
            dict(
                occupied=oracle.occupied.tolist(),
                a=oracle.full(a),
                b=oracle.full(b),
                c=oracle.full(c),
                induced_bare_three_max=three,
            )
        )

    flow_oracle = FockOracle([True, True, True, False, False, False])
    one, two = (
        rng.normal(scale=0.01, size=(6, 6)),
        rng.normal(scale=0.02, size=(15, 15)),
    )
    one = one + one.T + np.diag([-1.1, -0.7, -0.3, 0.4, 0.8, 1.3])
    initial = np.concatenate(([0.0], one.ravel(), (two + two.T).ravel()))
    flows = []
    for kind, end in [("Wegner", 0.2), ("White", 0.5)]:
        final = flow_oracle.flow(initial, kind, end)
        assert final[0] < 0.0
        flows.append(
            dict(
                kind=kind,
                end=end,
                occupied=flow_oracle.occupied.tolist(),
                initial=flow_oracle.full(initial),
                final=flow_oracle.full(final),
            )
        )
        print(kind, final[0], flush=True)

    mol = gto.M(
        atom="H 0.1 -0.2 -0.7; H -0.1 0.2 0.7",
        unit="Bohr",
        basis=embedded_basis("STO_3G_GBS", ["H"]),
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-11)
    assert mf.converged
    spatial = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), 2)
    fock = mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff
    one, two = np.zeros((4, 4)), np.zeros((4,) * 4)
    for p, q, r, s in itertools.product(range(4), repeat=4):
        one[p, q] = fock[p // 2, q // 2] if p % 2 == q % 2 else 0.0
        two[p, q, r, s] = (
            spatial[p // 2, r // 2, q // 2, s // 2]
            if p % 2 == r % 2 and q % 2 == s % 2
            else 0.0
        ) - (
            spatial[p // 2, s // 2, q // 2, r // 2]
            if p % 2 == s % 2 and q % 2 == r % 2
            else 0.0
        )
    small = FockOracle([True, True, False, False])
    initial = np.concatenate(
        (
            [0.0],
            one.ravel(),
            [two[p, q, r, s] for p, q in small.pairs for r, s in small.pairs],
        )
    )
    energies = {
        kind: float(small.flow(initial, kind, 0.5)[0]) for kind in ("Wegner", "White")
    }
    out = dict(
        source="Independent complete Fock-space matrix commutators, explicit induced bare rank-three extraction and normal ordering; SciPy DOP853 flows; embedded-basis PySCF RHF H2",
        commutators=commutators,
        flows=flows,
        h2=dict(e_scf=mf.e_tot, correlation=energies),
    )
    Path(__file__).resolve().parents[1].joinpath(
        "data/imsrg2_reference.json"
    ).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("H2", energies, flush=True)


if __name__ == "__main__":
    main()
