"""Cartesian-normalized DF integral and RI-MP2 references, existing PySCF only."""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import df, gto, lib, mp, scf

from embedded_basis import basis_hash, embedded_basis

lib.num_threads(4)
atoms = [("O", [0.0, 0.0, 0.0]), ("H", [1.43, 0.2, 1.1]), ("H", [-1.2, -0.3, 1.3])]
basis = embedded_basis("STO_3G_GBS", ["O", "H"])
aux = embedded_basis("AUX_DEF2_SVP_RIFIT_GBS", ["O", "H"])
mol = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=True, verbose=0)
auxmol = df.addons.make_auxmol(mol, auxbasis=aux)
s = np.sqrt(mol.intor("int1e_ovlp").diagonal())
sa = np.sqrt(auxmol.intor("int1e_ovlp").diagonal())
metric = auxmol.intor("int2c2e") / sa[:, None] / sa[None, :]
three = (
    df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    / s[:, None, None]
    / s[None, :, None]
    / sa[None, None, :]
)
# hf-rs uses xx,yy,zz,xy,xz,yz for Cartesian d (lexicographic for l>=3).
order = []
offset = 0
for shell in range(auxmol.nbas):
    angular = auxmol.bas_angular(shell)
    ncart = (angular + 1) * (angular + 2) // 2
    for contraction in range(auxmol.bas_nctr(shell)):
        order.extend(
            [offset + i for i in ([0, 3, 5, 1, 2, 4] if angular == 2 else range(ncart))]
        )
        offset += ncart
metric = metric[np.ix_(order, order)]
three = three[:, :, order]
mf = scf.RHF(mol).run(conv_tol=1e-14)
reference = mp.MP2(mf).density_fit(auxbasis=aux)
# Reuse PySCF's standard prebuilt DF tensor (also avoids its direct-build
# Fortran-contiguity assertion with the installed NumPy/Python combination).
reference.with_df.build()
reference.kernel()
out = {
    "source": f"PySCF {pyscf.__version__}, cart=True, individually normalized Cartesian functions; embedded basis",
    "basis_source_sha256": basis_hash(),
    "n_ao": mol.nao,
    "n_aux": auxmol.nao,
    "metric": metric.tolist(),
    "three": three.tolist(),
    "ri_mp2": reference.e_corr,
}
Path(__file__).resolve().parents[1].joinpath("data/df_reference.json").write_text(
    json.dumps(out, indent=1) + "\n"
)
