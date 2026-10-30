# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent spherical transform and SCF oracles; no downloads required.

Run with an existing PySCF installation. Coordinates are in bohr.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, lib, scf

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)


def c1_d_reference():
    """Four C1 nuclear centers; full d shell and two basis-bearing centers.

    H2/H3 intentionally carry no AOs. They remain positive nuclear centers in
    every electronic/nuclear integral and all twelve nuclear derivatives.
    Only public PySCF calculation interfaces are executed.
    """
    positions = np.asarray(
        [
            [-0.4, 0.1, -0.6],
            [0.7, -0.5, 1.0],
            [-1.1, 1.0, 0.4],
            [0.9, 1.3, -0.8],
        ]
    )
    labels = ["He", "H1", "H2", "H3"]
    basis = {"He": [[0, [1.4, 1.0]], [2, [0.6, 1.0]]], "H1": [[0, [1.1, 1.0]]]}
    potential = [
        0,
        [
            [-1, [[], [], [[0.6, -0.05]], [], [], []]],
            [0, [[], [], [[0.9, 0.10]], [], [], []]],
            [1, [[], [], [[0.7, -0.04]], [], [], []]],
            [2, [[], [], [[0.8, 0.08]], [], [], []]],
        ],
    ]

    def solve(coords, cartesian, with_ecp, open_shell):
        mol = gto.M(
            atom=list(zip(labels, coords.tolist())),
            basis=basis,
            ecp={"He": potential} if with_ecp else {},
            unit="bohr",
            charge=2 if open_shell else 1,
            spin=1 if open_shell else 0,
            cart=cartesian,
            symmetry=False,
            verbose=0,
        )
        hf = scf.UHF(mol) if open_shell else scf.RHF(mol)
        hf.conv_tol = 1e-14
        hf.conv_tol_grad = 1e-11
        hf.max_cycle = 300
        hf.kernel()
        assert hf.converged
        gradient = hf.nuc_grad_method().kernel()
        density = hf.make_rdm1()
        if open_shell:
            density = density.sum(axis=0)
        overlap = hf.get_ovlp()
        d_orbitals = (
            mol.cart2sph_coeff(normalized="sp")[:, 1:6]
            if cartesian
            else np.eye(mol.nao)[:, 1:6]
        )
        d_weight = np.trace(d_orbitals.T @ overlap @ density @ overlap @ d_orbitals)
        return hf.e_tot, gradient, float(d_weight), mol.nao

    cases = {}
    for cartesian in [True, False]:
        for with_ecp, open_shell in [(False, False), (True, False), (True, True)]:
            name = (
                ("cartesian" if cartesian else "spherical")
                + ("_ecp" if with_ecp else "_ae")
                + ("_uhf" if open_shell else "_rhf")
            )
            energy, gradient, d_weight, n_ao = solve(
                positions, cartesian, with_ecp, open_shell
            )
            hessian = np.zeros((12, 12))
            step = 5e-3
            for coordinate in range(12):
                for offset, weight in [(-2, 1), (-1, -8), (1, 8), (2, -1)]:
                    moved = positions.copy()
                    moved.flat[coordinate] += offset * step
                    value = solve(moved, cartesian, with_ecp, open_shell)[1]
                    hessian[:, coordinate] += weight * value.ravel() / (12 * step)
            cases[name] = {
                "source": f"PySCF {pyscf.__version__}; public RHF/UHF and analytic gradients",
                "geometry_bohr": positions.tolist(),
                "basis_atom_indices": [0, 1],
                "basis_gaussian94": "He 0\nS 1 1\n1.4 1\nD 1 1\n0.6 1\n****\nH 0\nS 1 1\n1.1 1\n****\n",
                "cartesian": cartesian,
                "ecp": with_ecp,
                "charge": 2 if open_shell else 1,
                "multiplicity": 2 if open_shell else 1,
                "n_ao": n_ao,
                "energy": energy,
                "gradient": gradient.tolist(),
                "hessian": hessian.tolist(),
                "hessian_reference_step_bohr": step,
                "hessian_reference": "five-point differences of PySCF analytic gradients; not symmetrized",
                "d_projection_weight": d_weight,
            }
            print(
                f"C1 d-shell oracle {name}: {n_ao} AOs, E={energy:.14f}, d weight={d_weight:.8f}",
                flush=True,
            )
    return cases


def spdf_integral_reference():
    """All 16 angular pairs and 256 angular quartets at four distinct centers."""
    positions = [[0.1, -0.2, 0.3], [0.4, 0.6, -0.5], [-0.2, 0.7, 0.8], [0.9, -0.3, 0.2]]
    symbols = ["He", "Li", "Be", "B"]
    basis = {}
    blocks = []
    for center, symbol in enumerate(symbols):
        exponents = [0.8 + 0.2 * center, 0.3 + 0.07 * center]
        basis[symbol] = [
            [angular, [exponents[0], 0.7], [exponents[1], -0.2]] for angular in range(4)
        ]
        text = symbol + " 0\n"
        for shell in "SPDF":
            text += f"{shell} 2 1\n{exponents[0]:.17g} 0.7\n{exponents[1]:.17g} -0.2\n"
        blocks.append(text + "****\n")
    potential = [
        0,
        [
            [-1, [[], [], [[0.6, -0.05]], [], [], []]],
            [0, [[], [], [[0.9, 0.10]], [], [], []]],
            [1, [[], [], [[0.7, -0.04]], [], [], []]],
            [2, [[], [], [[0.8, 0.08]], [], [], []]],
            [3, [[], [], [[1.0, -0.03]], [], [], []]],
        ],
    ]
    mol = gto.M(
        atom=list(zip(symbols, positions)),
        basis=basis,
        ecp={"Be": potential},
        unit="bohr",
        cart=True,
        symmetry=False,
        verbose=0,
    )
    # hf-rs retains Gaussian/Molden d order (xx,yy,zz,xy,xz,yz),
    # while libcint uses xx,xy,xz,yy,yz,zz. Other shells share the order.
    order = np.asarray([0, 1, 2, 3, 4, 7, 9, 5, 6, 8, *range(10, 20)])
    global_order = np.concatenate([order + 20 * center for center in range(4)])
    overlap = mol.intor("int1e_ovlp_cart")[np.ix_(global_order, global_order)]
    norms = np.sqrt(np.diag(overlap)).reshape(4, 20)
    pair_norm = norms[0, :, None] * norms[1, None, :]
    pair_blocks = {}
    for name, intor in [
        ("overlap", "int1e_ovlp_cart"),
        ("kinetic", "int1e_kin_cart"),
        ("nuclear_attraction", "int1e_nuc_cart"),
        ("ecp", "ECPscalar_cart"),
    ]:
        values = mol.intor(intor)[np.ix_(order, order + 20)]
        pair_blocks[name] = (values / pair_norm).ravel().tolist()
    eri = mol.intor("int2e_cart", shls_slice=(0, 4, 4, 8, 8, 12, 12, 16))
    for axis in range(4):
        eri = np.take(eri, order, axis=axis)
        shape = [1] * 4
        shape[axis] = 20
        eri /= norms[axis].reshape(shape)
    return {
        "source": f"PySCF {pyscf.__version__}; public normalized Cartesian integral blocks",
        "symbols": symbols,
        "geometry_bohr": positions,
        "basis_gaussian94": "".join(blocks),
        "ecp_center": 2,
        "ecp_l_max": 4,
        "ecp_terms": {
            "local": [2, 0.6, -0.05],
            "s": [2, 0.9, 0.10],
            "p": [2, 0.7, -0.04],
            "d": [2, 0.8, 0.08],
            "f": [2, 1.0, -0.03],
        },
        "shell_ranges": [[0, 1], [1, 4], [4, 10], [10, 20]],
        "cartesian_order": "s; px,py,pz; dxx,dyy,dzz,dxy,dxz,dyz; fxxx,fxxy,fxxz,fxyy,fxyz,fxzz,fyyy,fyyz,fyzz,fzzz",
        "pair_shape": [20, 20],
        "quartet_shape": [20, 20, 20, 20],
        "pairs": pair_blocks,
        "eri": eri.ravel().tolist(),
        "pair_angular_combinations": 16,
        "quartet_angular_combinations": 256,
    }


def c1_mixed_spdf_reference():
    """Public SCF driver with explicitly projected Cartesian integral callbacks."""
    from scipy.linalg import eigh

    positions = np.asarray(
        [[-0.4, 0.1, -0.6], [0.7, -0.5, 1.0], [-1.1, 1.0, 0.4], [0.9, 1.3, -0.8]]
    )
    symbols = ["He", "Li", "Be", "B"]
    basis = {
        symbol: [[angular, [0.8 + 0.2 * angular, 1.0]]]
        for angular, symbol in enumerate(symbols)
    }
    selected = [0, 2, 8, 14]  # s, py, dyz, fxyz in public libcint order.
    potential = [
        0,
        [
            [-1, [[], [], [[0.6, -0.05]], [], [], []]],
            [0, [[], [], [[0.9, 0.10]], [], [], []]],
            [1, [[], [], [[0.7, -0.04]], [], [], []]],
            [2, [[], [], [[0.8, 0.08]], [], [], []]],
            [3, [[], [], [[1.0, -0.03]], [], [], []]],
        ],
    ]

    def energy(geometry, use_ecp, unrestricted):
        mol = gto.M(
            atom=list(zip(symbols, geometry)),
            basis=basis,
            ecp={"Be": potential} if use_ecp else {},
            unit="bohr",
            charge=11 if unrestricted else 10,
            spin=1 if unrestricted else 0,
            cart=True,
            symmetry=False,
            verbose=0,
        )
        indices = np.ix_(selected, selected)
        full_overlap = mol.intor("int1e_ovlp_cart")
        norms = np.sqrt(full_overlap.diagonal()[selected])
        overlap = full_overlap[indices] / np.outer(norms, norms)
        core = (mol.intor("int1e_kin_cart") + mol.intor("int1e_nuc_cart"))[indices]
        if use_ecp:
            core += mol.intor("ECPscalar_cart")[indices]
        core /= np.outer(norms, norms)
        eri = mol.intor("int2e_cart")[np.ix_(selected, selected, selected, selected)]
        eri /= np.einsum("p,q,r,s->pqrs", norms, norms, norms, norms)

        def get_jk(mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
            density = np.asarray(dm)
            coulomb = np.einsum("pqrs,...rs->...pq", eri, density) if with_j else None
            exchange = np.einsum("prqs,...rs->...pq", eri, density) if with_k else None
            return coulomb, exchange

        solver = scf.UHF(mol) if unrestricted else scf.RHF(mol)
        solver.get_hcore = lambda mol=None: core
        solver.get_ovlp = lambda mol=None: overlap
        solver.get_jk = get_jk
        _, coefficients = eigh(core, overlap)
        alpha = coefficients[:, :2] @ coefficients[:, :2].T
        initial = (
            np.asarray([alpha, coefficients[:, :1] @ coefficients[:, :1].T])
            if unrestricted
            else 2.0 * alpha
        )
        solver.conv_tol = 1e-13
        solver.conv_tol_grad = 1e-11
        solver.max_cycle = 200
        solver.kernel(dm0=initial)
        assert solver.converged
        return float(solver.e_tot)

    records = {}
    step = 1e-3
    for name, use_ecp, unrestricted in [
        ("ae_rhf", False, False),
        ("ecp_rhf", True, False),
        ("ecp_uhf", True, True),
    ]:
        gradient = np.zeros((4, 3))
        for coordinate in range(12):
            for shift, weight in [(-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)]:
                moved = positions.copy()
                moved.flat[coordinate] += shift * step
                gradient.flat[coordinate] += (
                    weight * energy(moved, use_ecp, unrestricted) / (12.0 * step)
                )
        records[name] = {
            "source": f"PySCF {pyscf.__version__}; public projected-integral SCF callbacks",
            "geometry_bohr": positions.tolist(),
            "symbols": symbols,
            "cartesian_powers": [[0, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 1]],
            "exponents": [0.8, 1.0, 1.2, 1.4],
            "charge": 11 if unrestricted else 10,
            "multiplicity": 2 if unrestricted else 1,
            "ecp": use_ecp,
            "energy": energy(positions, use_ecp, unrestricted),
            "gradient": gradient.tolist(),
            "gradient_reference": "five-point energy difference, h=1e-3 bohr",
        }
    return records


parser = argparse.ArgumentParser(description=__doc__)
mode = parser.add_mutually_exclusive_group()
mode.add_argument(
    "--c1-d-only",
    action="store_true",
    help="update only the C1 d-shell section of the existing fixture",
)
mode.add_argument(
    "--spdf-only",
    action="store_true",
    help="update only the complete mixed-SPDF integral fixture section",
)
mode.add_argument(
    "--c1-spdf-only",
    action="store_true",
    help="update the C1 projected-SPDF SCF anchors only",
)
arguments = parser.parse_args()
reference_path = Path(__file__).resolve().parents[1] / "data/spherical_reference.json"
if arguments.c1_spdf_only:
    existing = json.loads(reference_path.read_text(encoding="utf-8"))
    existing["c1_mixed_spdf"] = c1_mixed_spdf_reference()
    reference_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(
        "C1 mixed SPDF: three independent energies and all 36 gradient anchors",
        flush=True,
    )
    raise SystemExit(0)
if arguments.c1_d_only:
    existing = json.loads(reference_path.read_text(encoding="utf-8"))
    existing["c1_d_systems"] = c1_d_reference()
    reference_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0)
if arguments.spdf_only:
    existing = json.loads(reference_path.read_text(encoding="utf-8"))
    existing["spdf_integrals"] = spdf_integral_reference()
    reference_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(
        "SPDF reference: 16 pair combinations, 256 quartet combinations, 161600 integral components",
        flush=True,
    )
    raise SystemExit(0)

data = {
    "source": f"PySCF {pyscf.__version__}; normalized Cartesian rows; real spherical m=-l..l (p=x,y,z)",
    "transforms": [],
}
data["basis_source_sha256"] = basis_hash()
for angular in range(8):
    mol = gto.M(
        atom="He 0 0 0", basis={"He": [[angular, [0.7, 1.0]]]}, cart=True, verbose=0
    )
    s = mol.intor("int1e_ovlp")
    # libcint normalizes only Cartesian s/p; convert to individually normalized rows.
    t = mol.cart2sph_coeff(normalized="sp") * np.sqrt(np.diag(s))[:, None]
    powers = [
        [x, y, angular - x - y]
        for x in range(angular, -1, -1)
        for y in range(angular - x, -1, -1)
    ]
    data["transforms"].append({"l": angular, "powers": powers, "matrix": t.tolist()})
systems = {}
for name, basis, atoms in [
    (
        "water_ccpvdz",
        embedded_basis("BASIS_CC_PVDZ_GBS", ["O", "H"]),
        [("O", [0.0, 0.0, 0.0]), ("H", [1.43, 0.2, 1.1]), ("H", [-1.2, -0.3, 1.3])],
    ),
    (
        "heh_model",
        {"He": [[0, [1.4, 1.0]], [2, [0.6, 1.0]]], "H": [[0, [1.1, 1.0]]]},
        [("He", [-0.1, 0.2, 0.3]), ("H", [0.4, -0.3, 1.8])],
    ),
]:
    mol = gto.M(
        atom=atoms,
        basis=basis,
        charge=1 if name == "heh_model" else 0,
        unit="Bohr",
        cart=False,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-10, max_cycle=200)
    assert mf.converged
    systems[name] = {
        "n_ao": mol.nao,
        "energy": mf.e_tot,
        "gradient": mf.nuc_grad_method().kernel().tolist(),
        "hessian": mf.Hessian()
        .kernel()
        .transpose(0, 2, 1, 3)
        .reshape(3 * mol.natm, 3 * mol.natm)
        .tolist(),
    }
data["systems"] = systems
data["c1_d_systems"] = c1_d_reference()
data["spdf_integrals"] = spdf_integral_reference()
data["c1_mixed_spdf"] = c1_mixed_spdf_reference()
Path(__file__).resolve().parents[1].joinpath(
    "data/spherical_reference.json"
).write_text(json.dumps(data, indent=2) + "\n")
