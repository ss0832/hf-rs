"""Brute-force small-space MRCISD oracle, independently projected from PySCF FCI."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import ao2mo, fci, gto, lib, scf

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)
root = Path(__file__).resolve().parents[1]
mol = gto.M(
    atom=[("Li", [0.1, -0.2, 0.3]), ("H", [0.4, 0.1, 3.2])],
    unit="Bohr",
    basis=embedded_basis("STO_3G_GBS", ["Li", "H"]),
    cart=True,
    verbose=0,
)
mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-10)
n = mol.nao_nr()
strings = fci.cistring.make_strings(range(n), 2)
dets = [(int(a), int(b)) for a in strings for b in strings]
unit = np.eye(len(dets))
ss = np.column_stack([fci.spin_op.contract_ss(v, n, (2, 2)).ravel() for v in unit])
out = {
    "source": f"PySCF {pyscf.__version__}; full FCI actions restricted by brute-force excitation-rank selection and spin completion of spatial occupations; singlet S2 subspace diagonalization",
    "basis_source_sha256": basis_hash(),
    "e_scf": float(mf.e_tot),
    "cases": {},
}
refined = json.loads((root / "data/casscf_reference.json").read_text())
for name, ncas, nelec, orbitals, frozen in [
    ("hf22", 2, (1, 1), mf.mo_coeff, 0),
    (
        "casscf22",
        2,
        (1, 1),
        np.array(refined["results"]["lih_cas22"]["mo_coefficients"]),
        0,
    ),
    ("cisd", 0, (0, 0), mf.mo_coeff, 0),
    ("full", 6, (2, 2), mf.mo_coeff, 0),
    ("frozen22", 2, (1, 1), mf.mo_coeff, 1),
]:
    ncore = 2 - nelec[0]
    core = (1 << ncore) - 1
    active_mask = ((1 << ncas) - 1) << ncore
    reference = [
        d
        for d in dets
        if all((s & core) == core and s & ~(core | active_mask) == 0 for s in d)
    ]
    selected = [
        i
        for i, d in enumerate(dets)
        if min(
            ((d[0] ^ r[0]).bit_count() + (d[1] ^ r[1]).bit_count()) // 2
            for r in reference
        )
        <= 2
        and all(s & ((1 << frozen) - 1) == ((1 << frozen) - 1) for s in d)
    ]
    patterns = {(dets[i][0] & dets[i][1], dets[i][0] ^ dets[i][1]) for i in selected}
    selected = [i for i, (a, b) in enumerate(dets) if (a & b, a ^ b) in patterns]
    h = orbitals.T @ mf.get_hcore() @ orbitals
    g = ao2mo.restore(1, ao2mo.kernel(mol, orbitals), n)
    absorbed = fci.direct_spin1.absorb_h1e(h, g, n, (2, 2), fac=0.5)
    full_h = np.column_stack(
        [
            fci.direct_spin1.contract_2e(absorbed, v, n, (2, 2)).ravel()
            + mol.energy_nuc() * v
            for v in unit
        ]
    )
    hsub = full_h[np.ix_(selected, selected)]
    spin_values, spin_vectors = np.linalg.eigh(ss[np.ix_(selected, selected)])
    singlet = spin_vectors[:, abs(spin_values) < 1e-9]
    energies, states = np.linalg.eigh(singlet.T @ hsub @ singlet)
    states = singlet @ states
    ref_inds = [i for i, d in enumerate(dets) if d in reference]
    ref_energy = float(np.linalg.eigvalsh(full_h[np.ix_(ref_inds, ref_inds)])[0])
    results = []
    for k in range(2):
        c = np.zeros(len(dets))
        c[selected] = states[:, k]
        weight = float(np.sum(c[ref_inds] ** 2))
        d1, d2 = fci.direct_spin1.make_rdm12(c, n, (2, 2))
        results.append(
            {
                "energy": float(energies[k]),
                "reference_weight": weight,
                "delta_q": (1 - weight) * (float(energies[k]) - ref_energy),
                "one_rdm_ao": (orbitals @ d1 @ orbitals.T).tolist(),
                "two_rdm_ao": np.einsum(
                    "pi,qj,rk,sl,ijkl->pqrs",
                    orbitals,
                    orbitals,
                    orbitals,
                    orbitals,
                    d2,
                    optimize=True,
                )
                .ravel()
                .tolist()
                if k == 0
                else None,
            }
        )
    out["cases"][name] = {
        "n_reference": len(reference),
        "n_determinants": len(selected),
        "e_reference": ref_energy,
        "states": results,
    }
# A common spatial orbital basis taken from an open-shell UHF reference. Complete
# its SD spatial occupations independently in the entire (2 alpha,1 beta) space.
open_mol = gto.M(
    atom=mol.atom, unit="Bohr", basis=mol.basis, charge=1, spin=1, cart=True, verbose=0
)
open_mf = scf.UHF(open_mol).run(conv_tol=1e-14, conv_tol_grad=1e-10)
c = open_mf.mo_coeff[0]
aa = fci.cistring.make_strings(range(n), 2)
bb = fci.cistring.make_strings(range(n), 1)
dets_open = [(int(a), int(b)) for a in aa for b in bb]
reference_open = [(3, 1)]
chosen = [
    i
    for i, (a, b) in enumerate(dets_open)
    if ((a ^ 3).bit_count() + (b ^ 1).bit_count()) // 2 <= 2
]
patterns = {
    (dets_open[i][0] & dets_open[i][1], dets_open[i][0] ^ dets_open[i][1])
    for i in chosen
}
chosen = [i for i, (a, b) in enumerate(dets_open) if (a & b, a ^ b) in patterns]
h = c.T @ open_mf.get_hcore() @ c
g = ao2mo.restore(1, ao2mo.kernel(open_mol, c), n)
absorbed = fci.direct_spin1.absorb_h1e(h, g, n, (2, 1), fac=0.5)
eye = np.eye(len(dets_open))
hh = np.column_stack(
    [
        fci.direct_spin1.contract_2e(absorbed, v, n, (2, 1)).ravel()
        + open_mol.energy_nuc() * v
        for v in eye
    ]
)
ss = np.column_stack([fci.spin_op.contract_ss(v, n, (2, 1)).ravel() for v in eye])
se, sv = np.linalg.eigh(ss[np.ix_(chosen, chosen)])
sector = sv[:, abs(se - 0.75) < 1e-9]
ee, cc = np.linalg.eigh(sector.T @ hh[np.ix_(chosen, chosen)] @ sector)
cc = sector @ cc
ref_index = dets_open.index((3, 1))
reference_energy = float(hh[ref_index, ref_index])
states = []
for k in range(2):
    vector = np.zeros(len(dets_open))
    vector[chosen] = cc[:, k]
    weight = float(vector[ref_index] ** 2)
    d1, d2 = fci.direct_spin1.make_rdm12(vector, n, (2, 1))
    states.append(
        {
            "energy": float(ee[k]),
            "reference_weight": weight,
            "delta_q": (1 - weight) * (float(ee[k]) - reference_energy),
            "one_rdm_ao": (c @ d1 @ c.T).tolist(),
            "two_rdm_ao": np.einsum(
                "pi,qj,rk,sl,ijkl->pqrs", c, c, c, c, d2, optimize=True
            )
            .ravel()
            .tolist()
            if k == 0
            else None,
        }
    )
out["cases"]["open_hf11"] = {
    "e_scf": float(open_mf.e_tot),
    "n_reference": 1,
    "n_determinants": len(chosen),
    "e_reference": reference_energy,
    "states": states,
}
Path(root / "data/mrci_reference.json").write_text(json.dumps(out, indent=1) + "\n")
