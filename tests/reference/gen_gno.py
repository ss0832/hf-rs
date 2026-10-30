# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent full Fock matrices and a solved GNO operator-basis transformation.

All rank-zero through rank-three normal products are explicitly constructed.
The commutator is an ordinary full Fock-matrix product; its coefficients are
obtained by a dense linear solve in that operator basis, not CAR contractions.
The independent BCH oracle uses a Pade superoperator exponential.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from gen_density_cumulants import parity, strings
from gen_imsrg2 import FockOracle
from scipy.linalg import expm, lu_factor, lu_solve


def creation(n):
    out = []
    for p in range(n):
        c = np.zeros((1 << n, 1 << n))
        for bits in range(1 << n):
            if not (bits >> p) & 1:
                c[bits | (1 << p), bits] = (-1) ** ((bits & ((1 << p) - 1)).bit_count())
        out.append(c)
    return out


def state_vector(n, create, member, active, core):
    na, nb = member["na"], member["nb"]
    n_spatial = len(active) // 2
    psi = np.zeros(1 << n)
    for index, (a, b) in enumerate(
        itertools.product(strings(n_spatial, na), strings(n_spatial, nb))
    ):
        orbitals = list(core)
        orbitals += [active[2 * p] for p in range(n_spatial) if (a >> p) & 1]
        orbitals += [active[2 * p + 1] for p in range(n_spatial) if (b >> p) & 1]
        determinant = np.zeros(1 << n)
        determinant[0] = 1.0
        for p in reversed(orbitals):
            determinant = create[p] @ determinant
        psi += member["coefficients"][index] * determinant
    return psi / np.linalg.norm(psi)


class GeneralizedOracle:
    def __init__(self, n, members, active, core):
        self.base = FockOracle([False] * n)
        self.n = n
        self.create = creation(n)
        self.groups = {
            rank: list(itertools.combinations(range(n), rank)) for rank in range(4)
        }
        self.rho = np.zeros((1 << n, 1 << n))
        total = sum(m["weight"] for m in members)
        for member in members:
            psi = state_vector(n, self.create, member, active, core)
            self.rho += member["weight"] / total * np.outer(psi, psi)
        removals = {}
        for rank, groups in self.groups.items():
            for group in groups:
                op = np.eye(1 << n)
                for p in group:
                    op = self.create[p].T @ op
                removals[group] = op
        densities = {}
        normal = {}
        basis = []
        for rank, groups in self.groups.items():
            for a in groups:
                for b in groups:
                    bare = removals[a].T @ removals[b]
                    density = float(np.einsum("ij,ji->", self.rho, bare))
                    densities[a, b] = density
                    if rank == 0:
                        op = bare
                    else:
                        op = bare - density * np.eye(1 << n)
                        for size in range(1, rank):
                            for ia in itertools.combinations(range(rank), size):
                                ac = tuple(a[k] for k in ia)
                                ar = tuple(a[k] for k in range(rank) if k not in ia)
                                for ib in itertools.combinations(range(rank), size):
                                    bc = tuple(b[k] for k in ib)
                                    br = tuple(b[k] for k in range(rank) if k not in ib)
                                    op -= (
                                        parity(ac, ar)
                                        * parity(bc, br)
                                        * densities[ac, bc]
                                        * normal[ar, br]
                                    )
                    normal[a, b] = op
                    basis.append(op)
        self.basis = np.array(basis)
        # The particle-number sectors zero through three uniquely identify all
        # rank <=3 coefficients. LU is independent of production projection sums.
        sample = np.column_stack([self.sample(op) for op in basis])
        self.lu = lu_factor(sample)
        self.retained = 1 + n * n + len(self.groups[2]) ** 2
        self.normal2 = self.basis[: self.retained]
        self.densities = {
            str(rank): [
                [densities[a, b] for b in self.groups[rank]] for a in self.groups[rank]
            ]
            for rank in (1, 2, 3)
        }

    def sample(self, op):
        return np.concatenate(
            [op[np.ix_(indices, indices)].ravel() for indices in self.base.sectors]
        )

    def matrix(self, coeff):
        return np.einsum("a,aij->ij", coeff, self.normal2)

    def project(self, op):
        full = lu_solve(self.lu, self.sample(op))
        return full[: self.retained], float(np.max(abs(full[self.retained :])))

    def commutator(self, a, b):
        am, bm = self.matrix(a), self.matrix(b)
        return self.project(am @ bm - bm @ am)

    def bch(self, h, a):
        am = self.matrix(a)
        comms = self.normal2 @ am - am @ self.normal2
        projected = lu_solve(
            self.lu, np.column_stack([self.sample(op) for op in comms])
        )
        return expm(projected[: self.retained]) @ h


def random_operator(rng, n, scale, anti=False):
    f = rng.normal(scale=scale, size=(n, n))
    g = rng.normal(scale=scale, size=(n * (n - 1) // 2,) * 2)
    f = f - f.T if anti else f + f.T
    g = g - g.T if anti else g + g.T
    return np.concatenate(([0.0 if anti else 0.13], f.ravel(), g.ravel()))


def case(name, n_spatial, members, active, core, rng):
    n = 6
    oracle = GeneralizedOracle(n, members, active, core)
    a = random_operator(rng, n, 0.025, True)
    h = random_operator(rng, n, 0.06)
    h[1 : 1 + n * n] += np.diag(np.linspace(-1.1, 0.9, n)).ravel()
    c, induced = oracle.commutator(h, a)
    assert induced > 1e-4
    bar = oracle.bch(h, a)
    physical = oracle.matrix(h)
    bare, _ = oracle.base.project(physical)
    print(name, "commutator scalar", c[0], "induced rank 3", induced, flush=True)
    return dict(
        name=name,
        n_spin=n,
        n_spatial=n_spatial,
        members=members,
        active=active,
        core=core,
        a=oracle.base.full(a),
        h=oracle.base.full(h),
        commutator=oracle.base.full(c),
        bch=oracle.base.full(bar),
        bare_h=oracle.base.full(bare),
        densities=oracle.densities,
        induced_three_max=induced,
    )


def coherent_scalar_cases(rng):
    # Only vector actions are needed at eight spin orbitals: no large operator
    # basis or 3985-by-3985 GNO transformation is allocated.
    data = json.loads(
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("data/density_cumulants_reference.json")
        .read_text(encoding="utf-8")
    )
    models = data["cases"][:2]
    n = 8
    create = creation(n)
    groups = list(itertools.combinations(range(n), 2))
    f = rng.normal(scale=0.1, size=(n, n))
    v = rng.normal(scale=0.08, size=(len(groups),) * 2)
    ah = f - f.T
    ag = v - v.T
    f = rng.normal(scale=0.2, size=(n, n))
    v = rng.normal(scale=0.12, size=(len(groups),) * 2)
    hh = f + f.T
    hg = v + v.T

    def full(zero, one, g):
        two = np.zeros((n,) * 4)
        for i, (p, q) in enumerate(groups):
            for j, (r, s) in enumerate(groups):
                two[p, q, r, s] = two[q, p, s, r] = g[i, j]
                two[q, p, r, s] = two[p, q, s, r] = -g[i, j]
        return dict(zero=zero, one=one.tolist(), two=two.ravel().tolist())

    cases = []
    for model in models:
        psi = state_vector(n, create, model["members"][0], list(range(n)), [])
        gamma1 = np.array(model["gamma1"])
        gamma2 = np.array(model["gamma2"])
        one = np.array(
            [
                create[p] @ (create[q].T @ psi) - gamma1[p, q] * psi
                for p in range(n)
                for q in range(n)
            ]
        )
        pair = np.zeros((len(groups) ** 2, 1 << n))
        for i, a in enumerate(groups):
            for j, b in enumerate(groups):
                row = create[a[0]] @ (
                    create[a[1]] @ (create[b[1]].T @ (create[b[0]].T @ psi))
                )
                row -= gamma2[i, j] * psi
                for r in range(2):
                    for s in range(2):
                        row -= (
                            (-1) ** (r + s)
                            * gamma1[a[r], b[s]]
                            * one[a[1 - r] * n + b[1 - s]]
                        )
                pair[i * len(groups) + j] = row
        h = hh.ravel() @ one + hg.ravel() @ pair
        a = ah.ravel() @ one + ag.ravel() @ pair
        scalar = 2 * float(h @ a)
        cases.append(
            dict(
                density_model=model["name"],
                n_spatial=4,
                members=model["members"],
                a=full(0.0, ah, ag),
                h=full(0.0, hh, hg),
                scalar=scalar,
            )
        )
    assert abs(cases[0]["scalar"] - cases[1]["scalar"]) > 1e-3
    return cases


def main():
    rng = np.random.default_rng(20261024)

    def member(na, nb, c, w=1.0):
        return dict(na=na, nb=nb, coefficients=list(map(float, c)), weight=w)

    cases = [
        case(
            "correlated_three_electron",
            3,
            [member(2, 1, rng.normal(size=9))],
            list(range(6)),
            [],
            rng,
        ),
        case(
            "correlated_ensemble",
            3,
            [
                member(1, 1, rng.normal(size=9), 0.3),
                member(1, 1, rng.normal(size=9), 0.7),
            ],
            list(range(6)),
            [],
            rng,
        ),
        case(
            "slater_limit",
            3,
            [member(1, 1, [1.0] + [0.0] * 8)],
            list(range(6)),
            [],
            rng,
        ),
        case(
            "core_active_external",
            1,
            [member(1, 0, [1.0], 0.4), member(0, 1, [1.0], 0.6)],
            [3, 1],
            [0, 5],
            rng,
        ),
        case(
            "correlated_embedded_pair",
            2,
            [member(1, 1, [0.3, 0.5, -0.2, 0.7])],
            [4, 1, 5, 2],
            [0],
            rng,
        ),
        case("vacuum_limit", 0, [member(0, 0, [1.0])], [], [], rng),
    ]
    out = dict(
        source="Full Fock-space products; independent dense LU solve in the complete rank-zero/one/two/three GNO operator basis; Pade superoperator BCH; coherent-state vector-action scalar check",
        cases=cases,
        coherent_scalars=coherent_scalar_cases(rng),
    )
    Path(__file__).resolve().parents[1].joinpath("data/gno_reference.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
