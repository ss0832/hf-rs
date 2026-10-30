# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent NEO-HF and positronic-HF oracle from public PySCF integrals.

Closed-shell electrons and high-spin positive quantum particles (protons or one positron)
are relaxed alternately to self-consistency: the electrons with PySCF RHF in the field of
the frozen particle density, the particles with a separate exact-Hessian trust-region
orbital Newton solver (G = J - K) in the field of the frozen electronic density. Each half
step lowers the energy. Nuclear potentials come from the classical nuclei only, through
1/|r-R| integrals; the particles feel them repulsively and carry kinetic energy T/M.
Electron-particle integrals are slices of the concatenated-molecule ERIs. Contracted
Cartesian AO space, embedded electronic basis data (ghost functions on the quantum-proton
sites), an even-tempered protonic basis, and the electronic basis for positrons. No NEO or
positronic implementation source is read.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from gen_ghf import rust_ao_permutation
from pyscf import gto, lib, scf
from pyscf.gto.basis import parse_gaussian

from embedded_basis import embedded_basis

lib.num_threads(4)

PROTON_MASS = 1836.15267343
PROTONIC_BASIS = """H     0
S   1   1.00
      4.0000000              1.0000000
S   1   1.00
      8.0000000              1.0000000
S   1   1.00
     16.0000000              1.0000000
S   1   1.00
     32.0000000              1.0000000
P   1   1.00
      8.0000000              1.0000000
P   1   1.00
     16.0000000              1.0000000
****
"""

MODELS = [
    dict(
        name="HF_quantum_H",
        classical=[("F", [0.0, 0.0, 0.0])],
        quantum=[[0.1, -0.05, 1.733]],
        charge=0,
    ),
    dict(
        name="FHF_anion_central_proton",
        classical=[("F", [0.0, 0.0, -2.14]), ("F", [0.0, 0.0, 2.14])],
        quantum=[[0.02, 0.0, 0.08]],
        charge=-1,
    ),
    dict(
        name="water_two_quantum_protons",
        classical=[("O", [0.1, -0.2, 0.3])],
        quantum=[[1.5, 0.7, 0.4], [-0.4, 1.3, -0.2]],
        charge=0,
    ),
]

# A positron (mass 1) in the electronic basis; `charge` is that of the electronic molecule.
POSITRONIC_MODELS = [
    dict(name="PsH", classical=[("H", [0.1, -0.2, 0.3])], charge=-1),
    dict(
        name="positron_LiH",
        classical=[("Li", [0.1, -0.2, 0.3]), ("H", [0.4, 0.2, 3.3])],
        charge=0,
    ),
    dict(name="positron_F_anion", classical=[("F", [0.0, 0.1, -0.2])], charge=-1),
]
for _model in POSITRONIC_MODELS:
    _model.update(particle="positron", mass=1.0, quantum=[])


def molecules(model):
    if model.get("particle") == "positron":
        atoms = model["classical"]
        basis = embedded_basis("BASIS_6_31G_GBS", sorted({a for a, _ in atoms}))
        z = sum(gto.charge(a) for a, _ in atoms)
        kwargs = dict(atom=atoms, basis=basis, cart=True, unit="Bohr", verbose=0)
        # Integrals do not depend on the charge; one "electron" stands for the positron.
        return gto.M(charge=model["charge"], **kwargs), gto.M(
            charge=z - 1, spin=1, **kwargs
        )
    symbols = [s for s, _ in model["classical"]] + ["H"] * len(model["quantum"])
    positions = [r for _, r in model["classical"]] + model["quantum"]
    mol_e = gto.M(
        atom=list(zip(symbols, positions)),
        basis=embedded_basis("BASIS_6_31G_GBS", sorted(set(symbols))),
        cart=True,
        unit="Bohr",
        # Ghost-site H atoms each add one electron, as each quantum proton does.
        charge=model["charge"],
        verbose=0,
    )
    mol_p = gto.M(
        atom=[("H", r) for r in model["quantum"]],
        basis={
            "H": parse_gaussian.parse(
                PROTONIC_BASIS.split("\n", 1)[1].split("****")[0], optimize=False
            )
        },
        cart=True,
        unit="Bohr",
        spin=len(model["quantum"]),
        verbose=0,
    )
    return mol_e, mol_p


def classical_potential(mol, model):
    v = np.zeros((mol.nao, mol.nao))
    for symbol, position in model["classical"]:
        with mol.with_rinv_origin(position):
            v -= gto.charge(symbol) * mol.intor("int1e_rinv")
    return v


def nuclear_repulsion(model):
    e = 0.0
    for i, (a, ra) in enumerate(model["classical"]):
        for b, rb in model["classical"][:i]:
            e += gto.charge(a) * gto.charge(b) / np.linalg.norm(np.subtract(ra, rb))
    return e


def rotate(c, kappa, o, v):
    """C exp(X) for the antisymmetric X with X_vo = kappa."""
    x = np.zeros((c.shape[1], c.shape[1]))
    x[v, o], x[o, v] = kappa, -kappa.T
    # exp(X) = cos(sqrt(-X^2)) + sinc(sqrt(-X^2)) X
    w, u = np.linalg.eigh(-x @ x)
    theta = np.sqrt(np.maximum(w, 0.0))
    return c @ (
        u @ np.diag(np.cos(theta)) @ u.T + u @ np.diag(np.sinc(theta / np.pi)) @ u.T @ x
    )


def same_spin_newton(h, s, eri, n_occ, c=None):
    """High-spin determinant (G = J - K) in a fixed field by trust-region orbital Newton.

    E(kappa) = E0 + 2 F_ai kappa_ai + kappa (A + B) kappa with
    (A + B)_ai,bj = d_ij F_ab - d_ab F_ij + 2 (ai|bj) - (ab|ij) - (aj|bi).
    Steps solve the level-shifted Newton equations and are accepted only if the energy drops,
    so the iterations end at a minimum (positive-definite Hessian), not at a saddle.
    """

    def energy(c):
        dm = c[:, :n_occ] @ c[:, :n_occ].T
        j = np.einsum("pqrs,rs->pq", eri, dm)
        k = np.einsum("prqs,rs->pq", eri, dm)
        return np.sum(dm * (h + 0.5 * (j - k))), h + j - k

    fock = h
    if c is not None:
        fock = energy(c)[1]
    inverse = np.linalg.inv(np.linalg.cholesky(s))
    c = inverse.T @ np.linalg.eigh(inverse @ fock @ inverse.T)[1]
    o, v = slice(0, n_occ), slice(n_occ, c.shape[1])
    e0, fock = energy(c)
    for _ in range(200):
        f = c.T @ fock @ c
        g = f[v, o]
        mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, c, c, c, c, optimize=True)
        nv, no = g.shape
        hess = (
            np.einsum("ij,ab->aibj", np.eye(no), f[v, v])
            - np.einsum("ab,ij->aibj", np.eye(nv), f[o, o])
            + 2 * mo[v, o, v, o]
            - mo[v, v, o, o].transpose(0, 2, 1, 3)
            - mo[v, o, v, o].transpose(0, 3, 2, 1)
        ).reshape(nv * no, nv * no)
        lowest = np.linalg.eigvalsh(hess).min()
        if np.max(np.abs(g)) < 1e-13 and lowest > 1e-6:
            for block in (o, v):
                c[:, block] = c[:, block] @ np.linalg.eigh(f[block, block])[1]
            return c
        shift = max(0.0, 0.05 - lowest)
        while True:
            kappa = np.linalg.solve(hess + shift * np.eye(nv * no), -g.ravel())
            if np.linalg.norm(kappa) > 0.3:
                kappa *= 0.3 / np.linalg.norm(kappa)
            trial = rotate(c, kappa.reshape(nv, no), o, v)
            e1, fock1 = energy(trial)
            if e1 < e0 + 1e-14 or shift > 1e6:
                break
            shift = 2 * shift + 0.1
        if lowest < 0 and np.max(np.abs(g)) < 1e-10:
            # At a saddle point: step along the lowest Hessian mode.
            mode = np.linalg.eigh(hess)[1][:, 0]
            trial = rotate(c, 0.1 * mode.reshape(nv, no), o, v)
            e1, fock1 = energy(trial)
        c, e0, fock = trial, e1, fock1
    raise RuntimeError("protonic Newton iterations did not converge")


def commutator(f, p, s):
    return np.max(np.abs(f @ p @ s - s @ p @ f))


def solve(model):
    mol_e, mol_p = molecules(model)
    positron = model.get("particle") == "positron"
    n_p = 1 if positron else len(model["quantum"])
    n_e = sum(gto.charge(s) for s, _ in model["classical"]) - model["charge"]
    if not positron:
        n_e += n_p
    assert n_e % 2 == 0
    h_e = mol_e.intor("int1e_kin") + classical_potential(mol_e, model)
    t_p = mol_p.intor("int1e_kin") / model.get("mass", PROTON_MASS)
    h_p = t_p - classical_potential(mol_p, model)
    joint = gto.conc_mol(mol_e, mol_p)
    ne_sh, tot = mol_e.nbas, joint.nbas
    eri_ep = joint.intor(
        "int2e", shls_slice=(0, ne_sh, 0, ne_sh, ne_sh, tot, ne_sh, tot)
    ).reshape(mol_e.nao, mol_e.nao, mol_p.nao, mol_p.nao)

    assert mol_e.nelectron == n_e
    rhf = scf.RHF(mol_e)
    rhf.conv_tol, rhf.conv_tol_grad, rhf.max_cycle = 1e-14, 1e-11, 300
    s_p = mol_p.intor("int1e_ovlp")
    eri_p = mol_p.intor("int2e")
    p_p = np.zeros((mol_p.nao, mol_p.nao))
    previous, dm_e, c_p = None, None, None
    for _ in range(200):
        field_e = h_e - np.einsum("pqrs,rs->pq", eri_ep, p_p)
        rhf.get_hcore = lambda *args, f=field_e: f
        rhf.kernel(dm0=dm_e)
        assert rhf.converged
        dm_e = rhf.make_rdm1()
        field_p = h_p - np.einsum("pqrs,pq->rs", eri_ep, dm_e)
        c_p = same_spin_newton(field_p, s_p, eri_p, n_p, c_p)
        p_p = c_p[:, :n_p] @ c_p[:, :n_p].T
        j_e, k_e = rhf.get_jk(dm=dm_e)
        j_p, k_p = scf.hf.get_jk(mol_p, p_p)
        coupling = -np.einsum("pqrs,pq,rs", eri_ep, dm_e, p_p)
        e_e = np.sum(dm_e * (h_e + 0.5 * (j_e - 0.5 * k_e)))
        e_p = np.sum(p_p * (h_p + 0.5 * (j_p - k_p)))
        energy = e_e + e_p + coupling + nuclear_repulsion(model)
        f_e = h_e + j_e - 0.5 * k_e - np.einsum("pqrs,rs->pq", eri_ep, p_p)
        f_p = h_p + j_p - k_p - np.einsum("pqrs,pq->rs", eri_ep, dm_e)
        residual = max(
            commutator(f_e, dm_e, mol_e.intor("int1e_ovlp")),
            commutator(f_p, p_p, mol_p.intor("int1e_ovlp")),
        )
        if previous is not None and abs(energy - previous) < 1e-13 and residual < 1e-10:
            break
        previous = energy
    else:
        raise RuntimeError(model["name"])
    return dict(
        mol_e=mol_e,
        mol_p=mol_p,
        energy=energy,
        electronic_energy=e_e,
        protonic_energy=e_p,
        electron_proton_coulomb=coupling,
        protonic_kinetic_energy=float(np.sum(p_p * t_p)),
        nuclear_repulsion=nuclear_repulsion(model),
        dm_e=dm_e,
        dm_p=p_p,
        residual=residual,
        n_electrons=n_e,
    )


def main():
    out, positronic = [], []
    for model in MODELS + POSITRONIC_MODELS:
        r = solve(model)
        order_e = rust_ao_permutation(r["mol_e"], "BASIS_6_31G_GBS")
        if model.get("particle") == "positron":
            order_p = rust_ao_permutation(r["mol_p"], "BASIS_6_31G_GBS")
        else:
            # The protonic basis lists all s shells before all p shells, as PySCF groups them.
            assert all(
                r["mol_p"].bas_angular(i) <= r["mol_p"].bas_angular(i + 1)
                for i in range(r["mol_p"].nbas - 1)
                if r["mol_p"].bas_atom(i) == r["mol_p"].bas_atom(i + 1)
            )
            order_p = list(range(r["mol_p"].nao))
        record = dict(
            name=model["name"],
            classical=[dict(symbol=s, position=p) for s, p in model["classical"]],
            quantum=model["quantum"],
            charge=model["charge"],
            electronic_basis="6-31g",
            mass=model.get("mass", PROTON_MASS),
            n_electrons=r["n_electrons"],
            energy=float(r["energy"]),
            electronic_energy=float(r["electronic_energy"]),
            protonic_energy=float(r["protonic_energy"]),
            electron_proton_coulomb=float(r["electron_proton_coulomb"]),
            protonic_kinetic_energy=r["protonic_kinetic_energy"],
            nuclear_repulsion=float(r["nuclear_repulsion"]),
            oracle_residual=float(r["residual"]),
            electronic_density=r["dm_e"][np.ix_(order_e, order_e)].tolist(),
            protonic_density=r["dm_p"][np.ix_(order_p, order_p)].tolist(),
        )
        (positronic if model.get("particle") == "positron" else out).append(record)
        print(
            model["name"],
            r["energy"],
            r["protonic_kinetic_energy"],
            r["residual"],
            flush=True,
        )
    path = Path(__file__).resolve().parents[1] / "data/neo_hf_reference.json"
    path.write_text(
        json.dumps(
            dict(
                source=__doc__,
                pyscf_version=pyscf.__version__,
                protonic_basis=PROTONIC_BASIS,
                cases=out,
                positronic_cases=positronic,
            ),
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
