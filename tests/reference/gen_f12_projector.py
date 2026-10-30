# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent full product-space projector/Schur-complement F12 oracle.

The oracle applies explicit Q and F1+F2 matrices to wavefunctions. It does not
implement the production commutator/projector contraction expansion. A state
beyond the RI space checks the unit-operator terms; its nonlocal action is zero.
"""

import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(91273)
no, nobs, nri, nfull = 2, 4, 6, 7
eps = np.array([-1.3, -0.72, 0.31, 0.74, 1.5, 2.1, 3.6])
fock = np.diag(eps)
fock[:nobs, nobs:nri] = rng.normal(size=(nobs, nri - nobs)) * 0.08
fock[nobs:nri, :nobs] = fock[:nobs, nobs:nri].T
identity = np.eye(nfull)
fock_pair = np.kron(fock, identity) + np.kron(identity, fock)
swap = (
    np.eye(nfull**2)
    .reshape(nfull, nfull, nfull, nfull)
    .transpose(1, 0, 2, 3)
    .reshape(nfull**2, nfull**2)
)
ri_pairs = [p * nfull + q for p in range(nri) for q in range(nri)]
supported = ri_pairs + [nfull**2 - 1]
removed = [
    p * nfull + q
    for p in range(nri)
    for q in range(nri)
    if (p < nobs and q < nobs) or (p >= nobs and q < no) or (p < no and q >= nobs)
]
projector = np.eye(nfull**2)
projector[removed, removed] = 0
vv = [a * nfull + b for a in range(no, nobs) for b in range(no, nobs)]

cases = []
for sign, ecp_strength in [(1.0, 0.0), (-0.4, 0.0), (1.0, 0.8), (-0.4, -0.55)]:
    raw = rng.normal(size=(nri, nri)) * 0.13
    nonlocal_part = np.zeros((nfull, nfull))
    nonlocal_part[:nri, :nri] = sign * raw @ raw.T
    potential = np.zeros((nfull, nfull))
    if ecp_strength:
        raw = rng.normal(size=(nri, nri)) * 0.27
        potential[:nri, :nri] = ecp_strength * (raw + raw.T) / 2
    exchange = nonlocal_part.copy()
    nonlocal_part = exchange - potential
    projected_nonlocal = nonlocal_part.copy()
    projected_nonlocal[:no, nobs:nri] += fock[:no, nobs:nri]
    projected_nonlocal[nobs:nri, :no] += fock[nobs:nri, :no]
    nonlocal_pair = np.kron(projected_nonlocal, identity) + np.kron(
        identity, projected_nonlocal
    )
    cabs_pair_projector = np.diag(
        [
            float(nobs <= p < nri and nobs <= q < nri)
            for p in range(nfull)
            for q in range(nfull)
        ]
    )
    hybrid_omissions = {}
    potential_pair = np.kron(potential, identity) + np.kron(identity, potential)
    kinetic = fock + nonlocal_part
    kinetic_pair = np.kron(kinetic, identity) + np.kron(identity, kinetic)
    raw = rng.normal(size=(len(supported), len(supported))) * 0.12
    factor = np.zeros((nfull**2, nfull**2))
    factor[np.ix_(supported, supported)] = (raw + raw.T) / 2
    factor = (factor + swap @ factor @ swap) / 2
    square = factor @ factor
    for label, resolution in [("hy1", projector), ("hy2", cabs_pair_projector)]:
        omission = factor @ nonlocal_pair @ resolution @ factor
        hybrid_omissions[label] = (omission + omission.T) / 2
    coulomb = 0.7 * np.eye(nfull**2) - 0.3 * factor + 0.2 * square
    gf = coulomb @ factor
    commutator = (
        2 * factor @ kinetic_pair @ factor
        - square @ kinetic_pair
        - kinetic_pair @ square
    )
    single_commutator = factor @ kinetic_pair - kinetic_pair @ factor
    # Explicit full-space commutator/Q operator after applying the 3*A GBC/EBC
    # and exchange-commutator approximations. No production RI expansion.
    total_commutator = (
        factor @ (kinetic_pair + potential_pair)
        - (kinetic_pair + potential_pair) @ factor
    )
    a_operator = 0.5 * (
        total_commutator @ projector @ factor - factor @ projector @ total_commutator
    )
    pairs = []
    for i in range(no):
        for j in range(i, no):
            ij, ji = i * nfull + j, j * nfull + i
            left = projector @ factor[:, ij]
            right = projector @ factor[:, ji]
            v = [float(coulomb[ij] @ left), float(coulomb[ij] @ right)]
            x = [float(left @ left), float(left @ right)]
            b = [float(left @ fock_pair @ left), float(left @ fock_pair @ right)]
            c = (fock_pair @ left)[vv].reshape(nobs - no, nobs - no)
            g = coulomb[vv, ij].reshape(nobs - no, nobs - no)
            energy = []
            a_energy = []
            operator = fock_pair - (eps[i] + eps[j]) * np.eye(nfull**2)
            conventional = operator[np.ix_(vv, vv)]
            for parity, cusp, multiplicity in [(1, 0.5, 1), (-1, 0.25, 3)]:
                if i == j and parity == -1:
                    energy.append(0.0)
                    a_energy.append(0.0)
                    continue
                ket = np.zeros(nfull**2)
                ket[ij] += 1
                ket[ji] += parity
                ket /= np.linalg.norm(ket)
                source = coulomb @ ket
                a_energy.append(
                    float(
                        multiplicity
                        * (
                            2 * cusp * (ket @ coulomb @ projector @ factor @ ket)
                            + cusp**2 * (ket @ a_operator @ ket)
                        )
                    )
                )
                fixed = cusp * projector @ factor @ ket
                conventional_amplitude = np.linalg.solve(
                    conventional, -(source + operator @ fixed)[vv]
                )
                wavefunction = fixed.copy()
                wavefunction[vv] += conventional_amplitude
                baseline = -source[vv] @ np.linalg.solve(conventional, source[vv])
                energy.append(
                    float(
                        multiplicity
                        * (
                            wavefunction @ operator @ wavefunction
                            + 2 * wavefunction @ source
                            - baseline
                        )
                    )
                )
            pairs.append(
                dict(
                    pair=[i, j],
                    factor=factor[np.ix_(ri_pairs, [ij])].reshape(nri, nri).tolist(),
                    coulomb=coulomb[np.ix_(ri_pairs, [ij])].reshape(nri, nri).tolist(),
                    squared=[
                        [float(square[p * nfull + k, ij]) for k in range(no)]
                        for p in range(nri)
                    ],
                    squared_exchange=[
                        [float(square[p * nfull + k, ji]) for k in range(no)]
                        for p in range(nri)
                    ],
                    coulomb_factor=[float(gf[ij, ij]), float(gf[ij, ji])],
                    commutator=[float(commutator[ij, ij]), float(commutator[ij, ji])],
                    kinetic_commutator=single_commutator[np.ix_(ri_pairs, [ij])]
                    .reshape(nri, nri)
                    .tolist(),
                    nonlocal_right_action=(factor @ potential_pair)[
                        np.ix_(ri_pairs, [ij])
                    ]
                    .reshape(nri, nri)
                    .tolist(),
                    a_b_bar=[float(a_operator[ij, ij]), float(a_operator[ij, ji])],
                    a_singlet=a_energy[0],
                    a_triplet=a_energy[1],
                    v=v,
                    x=x,
                    b=b,
                    c=c.tolist(),
                    g=g.tolist(),
                    singlet=energy[0],
                    triplet=energy[1],
                )
            )
            pairs[-1]["hybrids"] = {}
            for label, omission in hybrid_omissions.items():
                delta = np.array([omission[ij, ij], omission[ij, ji]])
                hybrid_energies = []
                for parity, cusp, multiplicity, baseline in [
                    (1, 0.5, 1, energy[0]),
                    (-1, 0.25, 3, energy[1]),
                ]:
                    if i == j and parity == -1:
                        hybrid_energies.append(0.0)
                        continue
                    ket = np.zeros(nfull**2)
                    ket[ij] += 1
                    ket[ji] += parity
                    ket /= np.linalg.norm(ket)
                    hybrid_energies.append(
                        float(
                            baseline + multiplicity * cusp**2 * (ket @ omission @ ket)
                        )
                    )
                pairs[-1]["hybrids"][label] = dict(
                    b=(np.array(b) + delta).tolist(),
                    singlet=hybrid_energies[0],
                    triplet=hybrid_energies[1],
                )
    cases.append(
        dict(
            n_occ=no,
            n_obs=nobs,
            fock=fock[:nri, :nri].tolist(),
            nonlocal_part=nonlocal_part[:nri, :nri].tolist(),
            nonlocal_potential=potential[:nri, :nri].tolist(),
            orbital_energies=eps[:nobs].tolist(),
            pairs=pairs,
        )
    )
out = dict(
    source=f"NumPy {np.__version__} explicit full product-space Q, F1+F2 and spin-adapted wavefunction Hylleraas functional with dense conventional solve",
    cases=cases,
)
Path(__file__).resolve().parents[1].joinpath(
    "data/f12_projector_reference.json"
).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print([(case["pairs"][0]["singlet"], case["pairs"][1]["triplet"]) for case in cases])
