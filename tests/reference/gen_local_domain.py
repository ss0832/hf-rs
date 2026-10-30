"""Independent PySCF high-order molecular quadrature DOI and Mulliken reference."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import dft, gto, lib

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)
path = Path(__file__).resolve().parents[1] / "data/localization_reference.json"
data = json.loads(path.read_text(encoding="utf-8"))
mol = gto.M(
    atom=[
        ("O", (0.13, -0.21, 0.31)),
        ("H", (0.23, 1.22, -0.77)),
        ("H", (-0.08, -1.64, -0.83)),
    ],
    basis=embedded_basis("STO_3G_GBS", ["O", "H"]),
    unit="Bohr",
    cart=True,
    verbose=0,
)
c = np.array(data["objectives"]["boys"]["coefficients"])
s = mol.intor("int1e_ovlp")
q = np.eye(len(s)) - c @ c.T @ s
norm = np.sqrt(np.diag(q.T @ s @ q))
paos = q / np.where(norm > 1e-10, norm, np.inf)


def integrate(radial, angular):
    grid = dft.gen_grid.Grids(mol)
    grid.atom_grid = (radial, angular)
    grid.prune = None
    grid.radii_adjust = None
    grid.build()
    ii = np.zeros((c.shape[1], c.shape[1]))
    iu = np.zeros((c.shape[1], len(s)))
    for start in range(0, len(grid.weights), 2048):
        ao = mol.eval_gto("GTOval_cart", grid.coords[start : start + 2048])
        occ = (ao @ c) ** 2
        pao = (ao @ paos) ** 2
        w = grid.weights[start : start + 2048, None]
        ii += occ.T @ (w * occ)
        iu += occ.T @ (w * pao)
    return np.sqrt(ii), np.sqrt(iu)


coarse = integrate(150, 974)
fine = integrate(200, 1202)
error = max(np.max(abs(a - b)) for a, b in zip(coarse, fine))
assert error < 1e-7, error
atom_indices = np.concatenate(
    [
        np.full(end - start, atom)
        for atom, (_, _, start, end) in enumerate(mol.aoslice_by_atom())
    ]
)
populations = np.zeros((c.shape[1], mol.natm))
for orbital in range(c.shape[1]):
    overlap = c[:, orbital, None] * c[None, :, orbital] * s
    diagonal = np.diag(overlap)
    for mu in range(len(s)):
        for nu in range(len(s)):
            denominator = diagonal[mu] + diagonal[nu]
            if denominator > 0:
                populations[orbital, atom_indices[mu]] += (
                    2 * overlap[mu, nu] * diagonal[mu] / denominator
                )
out = dict(
    source=f"PySCF {pyscf.__version__} Cartesian AO evaluation, unpruned Lebedev (200,1202) grid with equal-radius Becke partition",
    basis_source_sha256=basis_hash(),
    coefficients=c.tolist(),
    pao_projector=q.tolist(),
    occupied=fine[0].tolist(),
    occupied_pao=fine[1].tolist(),
    mulliken=populations.tolist(),
    grid_convergence_max=float(error),
)
Path(__file__).resolve().parents[1].joinpath(
    "data/local_domain_reference.json"
).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print("grid convergence", error)
