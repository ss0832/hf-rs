# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent small-system FMO from public PySCF RHF and dense NumPy integrals.

Monomer polarization uses simultaneous Jacobi updates, unlike native block descent.
Only public scientific interfaces are executed; no external implementation source is read.
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pyscf
from gen_ghf import rust_ao_permutation
from gen_mr_molecular import models as mr_models
from pyscf import gto, lib, scf
from scipy.linalg import eigh

from embedded_basis import embedded_basis

lib.num_threads(4)


def models():
    waters = [
        ("O", [0.0, 0.0, 0.0]),
        ("H", [1.432, 0.0, 1.1]),
        ("H", [-1.432, 0.0, 1.1]),
        ("O", [0.6, 5.2, 0.2]),
        ("H", [1.7, 6.3, 0.8]),
        ("H", [-0.7, 5.9, -0.3]),
        ("O", [5.0, 0.7, -0.3]),
        ("H", [6.2, 1.4, 0.8]),
        ("H", [4.3, 2.2, 0.2]),
    ]
    result = []
    for embedding in [False, True]:
        for order in [1, 2, 3]:
            result.append(
                dict(
                    name=f"water_trimer_{order}_{embedding}",
                    atoms=waters,
                    fragments=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                    charges=[0, 0, 0],
                    order=order,
                    embedding=embedding,
                    constant="STO_3G_GBS",
                    basis_name="sto-3g",
                )
            )
    interleaved = [waters[i] for i in [0, 3, 1, 4, 2, 5]]
    result.append(
        dict(
            name="water_dimer_interleaved",
            atoms=interleaved,
            fragments=[[4, 0, 2], [3, 5, 1]],
            charges=[0, 0],
            order=2,
            embedding=True,
            constant="STO_3G_GBS",
            basis_name="sto-3g",
        )
    )
    h8 = [
        ("H", [x, y, z])
        for x, y in [(0.0, 0.0), (3.8, 0.3), (1.4, 4.1), (5.2, 4.3)]
        for z in [-0.7, 0.7]
    ]
    result.append(
        dict(
            name="H8_four_fragments_FMO3",
            atoms=h8,
            fragments=[[0, 1], [2, 3], [4, 5], [6, 7]],
            charges=[0] * 4,
            order=3,
            embedding=True,
            constant="STO_3G_GBS",
            basis_name="sto-3g",
        )
    )
    result.append(
        dict(
            name="Li_plus_H_minus",
            atoms=[("Li", [0.0, 0.0, 0.0]), ("H", [0.3, 0.2, 6.0])],
            fragments=[[0], [1]],
            charges=[1, -1],
            order=2,
            embedding=True,
            constant="BASIS_6_31G_GBS",
            basis_name="6-31g",
        )
    )
    for m in result:
        m["atoms"] = [dict(symbol=s, position=r) for s, r in m["atoms"]]
        m["basis"] = embedded_basis(
            m["constant"], sorted({a["symbol"] for a in m["atoms"]})
        )
    ecp = dict(mr_models()[2])
    ecp.update(
        name="HeH_ECP_plus_H2",
        fragments=[[0, 1], [2, 3]],
        charges=[1, 0],
        order=2,
        embedding=True,
    )
    ecp["atoms"] = list(ecp["atoms"]) + [
        dict(symbol="H", position=[4.5, 0.2, z]) for z in [-0.7, 0.7]
    ]
    result.append(ecp)
    return result


def calculate(model):
    fragments = [sorted(f) for f in model["fragments"]]

    def molecule(members):
        indices = sorted(a for i in members for a in fragments[i])
        atoms = [
            (model["atoms"][a]["symbol"], model["atoms"][a]["position"])
            for a in indices
        ]
        mol = gto.M(
            atom=atoms,
            basis=model["basis"],
            ecp=model.get("ecp") or {},
            charge=sum(model["charges"][i] for i in members),
            unit="Bohr",
            cart=True,
            verbose=0,
        )
        return mol, indices

    nf = len(fragments)
    global_mol, _ = molecule(range(nf))
    global_h = scf.RHF(global_mol).get_hcore()
    eri = global_mol.intor("int2e", aosym="s1")
    locations = global_mol.aoslice_by_atom()[:, 2:]
    clusters = {}
    for size in range(1, min(model["order"], nf) + 1):
        for members in itertools.combinations(range(nf), size):
            mol, atoms = molecule(members)
            aos = np.array(
                [ao for a in atoms for ao in range(*locations[a])], dtype=int
            )
            hf = scf.RHF(mol)
            clusters[members] = dict(
                mol=mol, atoms=atoms, aos=aos, h=hf.get_hcore(), s=hf.get_ovlp()
            )

    def coulomb(cluster, other, density):
        return np.einsum(
            "mnls,ls->mn",
            eri[np.ix_(cluster["aos"], cluster["aos"], other["aos"], other["aos"])],
            density,
        )

    def field(members, densities):
        cluster = clusters[members]
        v = global_h[np.ix_(cluster["aos"], cluster["aos"])] - cluster["h"]
        for j in range(nf):
            if j not in members:
                v = v + coulomb(cluster, clusters[(j,)], densities[j])
        return v

    def solve(members, v):
        c = clusters[members]
        mf = scf.RHF(c["mol"])
        mf.get_hcore = lambda *args: c["h"] + v
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-14, 1e-12, 300
        mf.kernel()
        if np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ)) > 1e-10:
            mf = mf.newton().run(conv_tol=1e-14, conv_tol_grad=1e-12)
        p = mf.make_rdm1()
        f = mf.get_fock(dm=p, diis=None)
        residual = mf.mo_coeff.T @ (f @ p @ c["s"] - c["s"] @ p @ f) @ mf.mo_coeff
        assert np.linalg.norm(residual) < 1e-9
        return dict(
            density=p,
            coefficients=mf.mo_coeff,
            fock=f,
            field=v,
            mf=mf,
            internal=float(mf.energy_tot(dm=p) - np.einsum("mn,mn", p, v)),
        )

    states = [solve((i,), np.zeros_like(clusters[(i,)]["h"])) for i in range(nf)]
    if model["embedding"]:
        for cycle in range(200):
            density = [r["density"] for r in states]
            next_states = [solve((i,), field((i,), density)) for i in range(nf)]
            change = max(
                np.max(abs(a["density"] - b["density"]))
                for a, b in zip(states, next_states)
            )
            states = next_states
            if change < 1e-11:
                break
        else:
            raise AssertionError("monomer Jacobi embedding did not converge")

    def product(members):
        c = clusters[members]
        p = np.zeros_like(c["h"])
        occ = []
        for i in members:
            local = np.array(
                [np.where(c["aos"] == ao)[0][0] for ao in clusters[(i,)]["aos"]]
            )
            p[np.ix_(local, local)] = states[i]["density"]
            occupied = np.zeros((len(p), states[i]["mf"].mol.nelectron // 2))
            occupied[local, :] = states[i]["coefficients"][:, : occupied.shape[1]]
            occ.append(occupied)
        return p, np.concatenate(occ, axis=1)

    def internal(members, p):
        c = clusters[members]
        j, k = scf.RHF(c["mol"]).get_jk(dm=p)
        return float(
            c["mol"].energy_nuc() + np.einsum("mn,mn", p, c["h"] + 0.5 * j - 0.25 * k)
        )

    def record(members, r):
        c = clusters[members]
        if "constant" in model:
            order = rust_ao_permutation(c["mol"], model["constant"])
        else:
            order = list(range(len(c["h"])))
        return dict(
            members=list(members),
            internal_energy=r["internal"],
            density=r["density"][np.ix_(order, order)].tolist(),
            fock=r["fock"][np.ix_(order, order)].tolist(),
            field=r["field"][np.ix_(order, order)].tolist(),
        )

    records = [record((i,), r) for i, r in enumerate(states)]
    energies = [r["internal"] for r in states]
    total = sum(energies)
    pairs = []
    all_states = {(i,): r for i, r in enumerate(states)}
    if model["order"] >= 2:
        for i, j in itertools.combinations(range(nf), 2):
            members = (i, j)
            c = clusters[members]
            v = (
                field(members, [r["density"] for r in states])
                if model["embedding"]
                else np.zeros_like(c["h"])
            )
            r = solve(members, v)
            all_states[members] = r
            p, occupied = product(members)
            g = occupied.T @ c["s"] @ occupied
            values, vectors = eigh(g)
            assert values.min() > 1e-8
            frozen_c = occupied @ (vectors / np.sqrt(values)) @ vectors.T
            frozen = 2 * frozen_c @ frozen_c.T
            direct = (
                c["mol"].energy_nuc()
                - clusters[(i,)]["mol"].energy_nuc()
                - clusters[(j,)]["mol"].energy_nuc()
            )
            for left in [i, j]:
                local = [
                    np.where(c["aos"] == ao)[0][0] for ao in clusters[(left,)]["aos"]
                ]
                direct += np.einsum(
                    "mn,mn",
                    states[left]["density"],
                    c["h"][np.ix_(local, local)] - clusters[(left,)]["h"],
                )
            direct += np.einsum(
                "mn,mn",
                states[i]["density"],
                coulomb(clusters[(i,)], clusters[(j,)], states[j]["density"]),
            )
            embedding = np.einsum("mn,mn", r["density"] - p, v)
            delta = r["internal"] - energies[i] - energies[j] + embedding
            exchange = (
                internal(members, frozen)
                - energies[i]
                - energies[j]
                - direct
                + np.einsum("mn,mn", frozen - p, v)
            )
            pairs.append(
                dict(
                    members=list(members),
                    total=float(delta),
                    electrostatic=float(direct),
                    exchange=float(exchange),
                    relaxation=float(delta - direct - exchange),
                    embedding=float(embedding),
                )
            )
            total += delta
            records.append(record(members, r))
    triples = []
    if model["order"] >= 3:
        for members in itertools.combinations(range(nf), 3):
            c = clusters[members]
            v = (
                field(members, [r["density"] for r in states])
                if model["embedding"]
                else np.zeros_like(c["h"])
            )
            r = solve(members, v)
            all_states[members] = r
            p, _ = product(members)
            pair_sum = sum(
                pair["total"]
                for pair in pairs
                if set(pair["members"]).issubset(members)
            )
            delta = (
                r["internal"]
                - sum(energies[i] for i in members)
                - pair_sum
                + np.einsum("mn,mn", r["density"] - p, v)
            )
            total += delta
            triples.append(dict(members=list(members), correction=float(delta)))
            records.append(record(members, r))
    gradient = None
    if not model["embedding"] or nf <= model["order"]:
        gradient = np.zeros((global_mol.natm, 3))
        order = min(model["order"], nf)
        for members, state in all_states.items():
            size = len(members)
            if nf == order:
                weight = int(size == nf)
            else:
                weight = (-1) ** (order - size) * math.comb(nf - size - 1, order - size)
            if weight:
                g = state["mf"].nuc_grad_method().kernel()
                gradient[clusters[members]["atoms"]] += weight * g
    result = {k: v for k, v in model.items() if k != "basis"}
    result.update(
        energy=float(total),
        clusters=records,
        pairs=pairs,
        triples=triples,
        gradient=gradient.tolist() if gradient is not None else None,
    )
    print(model["name"], total, flush=True)
    return result


if __name__ == "__main__":
    results = dict(
        interface=f"PySCF {pyscf.__version__}: public RHF/gradient interfaces, dense NumPy Coulomb contractions",
        cases=[calculate(m) for m in models()],
    )
    path = Path(__file__).resolve().parents[1] / "data/fmo_reference.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
