# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Small full CCSD-F12 references, not a published default-hybrid benchmark.

Psi4 supplies independent AO operators. Explicit product/geminal Gram and
Hamiltonian matrices apply Q, and PySCF solves the modified CCSD equations.
The imported generator also refreshes the projected-integral reference file.
Run using the existing PySCF Python and its --psi4-python argument.
"""

import json
import os
from pathlib import Path

import numpy as np
import pyscf
from pyscf import cc, gto, lib, mp, scf

fitted_reference = os.environ.get("HF_RS_F12_FITTED_OPERATORS")
if fitted_reference:
    arrays = np.load(fitted_reference)
    root = Path(__file__).resolve().parents[2]
    positions = json.loads(str(arrays["positions"]))
    obs, aux = str(arrays["obs_g94"]), str(arrays["aux_g94"])
    terms = json.loads(str(arrays["terms"]))
else:
    from gen_ccsd_f12_projected import arrays, aux, obs, positions, root, terms

lib.num_threads(4)
mixed_fock = bool(arrays["mixed_fock"]) if "mixed_fock" in arrays else False
fit_orbital_mp2 = (
    bool(arrays["fit_orbital_mp2"]) if "fit_orbital_mp2" in arrays else False
)
exact_obs_coulomb = (
    bool(arrays["exact_obs_coulomb"]) if "exact_obs_coulomb" in arrays else False
)
cases = []
# Psi4 exposes the single-electron double commutator. Fix its normalization
# independently using the closed form of a centered S Gaussian quartet.
normalization = (2 * 1.4 / np.pi) ** 3
analytic_commutator = sum(
    8
    * a
    * b
    * c
    * d
    * normalization
    * np.pi**3
    / (2.8**2 + (a + b) * 5.6) ** 1.5
    * 1.5
    * 5.6
    / (2.8**2 + (a + b) * 5.6)
    for a, c in terms
    for b, d in terms
)
if not fitted_reference:
    assert abs(2 * arrays["commutator"][0, 0, 0, 0] - analytic_commutator) < 1e-13
for ecp in [False, True]:
    channels = [
        [-1, [[], [], [[0.7, 0.17]]]],
        [0, [[], [], [[0.53, 0.11]]]],
        [1, [[], [], [[0.61, -0.07]]]],
    ]
    mol = gto.M(
        atom=[("He", p) for p in positions],
        basis={"He": [[0, [1.4, 1.0]], [0, [0.45, 1.0]]]},
        ecp={"He": [0, channels]} if ecp else {},
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    auxmol = gto.M(
        atom=[("ghost-He", p) for p in positions],
        basis={
            "He": [[0, [0.2, 1.0]], [0, [0.9, 1.0]], [0, [3.2, 1.0]], [1, [0.8, 1.0]]]
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    union = gto.conc_mol(mol, auxmol)
    hf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-12)
    assert hf.converged
    nao, nm, no = mol.nao, hf.mo_coeff.shape[1], mol.nelectron // 2
    nv = nm - no
    overlap = union.intor("int1e_ovlp")
    projection = hf.mo_coeff @ hf.mo_coeff.T @ overlap[:nao, nao:]
    eigenvalues, rotation = np.linalg.eigh(
        overlap[nao:, nao:] - overlap[nao:, :nao] @ projection
    )
    caux = rotation[:, eigenvalues > 1e-8] / np.sqrt(eigenvalues[eigenvalues > 1e-8])
    ri = np.zeros((union.nao, nm + caux.shape[1]))
    ri[:nao, :nm] = hf.mo_coeff
    ri[:, nm:] = np.vstack((-projection @ caux, caux))
    nr = ri.shape[1]
    identity = np.eye(nr)
    occupied_products = [i * nr + j for i in range(no) for j in range(no)]
    virtual_products = [a * nr + b for a in range(no, nm) for b in range(no, nm)]
    removed = np.diag(
        [
            (p < nm and q < nm) or (p >= nm and q < no) or (p < no and q >= nm)
            for p in range(nr)
            for q in range(nr)
        ]
    ).astype(float)

    def transform(operator):
        return np.einsum(
            "ap,br,cq,ds,abcd->pqrs", ri, ri, ri, ri, operator, optimize=True
        ).reshape(nr**2, nr**2)

    rfull = transform(arrays["f"])
    r = rfull[:, occupied_products]
    squared = transform(arrays["squared"])[:, occupied_products]
    gfull = transform(arrays["g"])
    orbital_coulomb = (
        transform(arrays["g_exact"])
        if fitted_reference and not fit_orbital_mp2
        else gfull
    )
    projector_coulomb = gfull
    if exact_obs_coulomb:
        # Mathematical orthogonal-block choice from Knizia 2010, Sec. 4.5.1.
        # Transform the full consistent raw DF operator first, then replace
        # only its OBS/OBS product block. Orbital MP2/Schur G stays independent.
        projector_coulomb = gfull.copy()
        obs_products = [p * nr + q for p in range(nm) for q in range(nm)]
        exact = transform(arrays["g_exact"])
        projector_coulomb[np.ix_(obs_products, obs_products)] = exact[
            np.ix_(obs_products, obs_products)
        ]
    gf = transform(arrays["gf"])
    comm = transform(arrays["commutator"])[np.ix_(occupied_products, occupied_products)]
    dm = np.zeros((union.nao, union.nao))
    dm[:nao, :nao] = hf.make_rdm1()
    jao, kao = scf.hf.get_jk(union, dm)
    if mixed_fock:
        # Independent dense contraction of the original-auxiliary fitted ERIs.
        jao = np.einsum("pqrs,rs->pq", arrays["g_fock"], dm)
        kao = np.einsum("prqs,rs->pq", arrays["g_fock"], dm)
    kinetic = union.intor("int1e_kin")
    core = kinetic + union.intor("int1e_nuc")
    nonlocal_part = 0.5 * kao
    if ecp:
        core += union.intor("ECPscalar")
        # Build a zero-local ECP through public molecular input, retaining all
        # semilocal channels. No implementation code or internal tables are read.
        nonlocal_mol = gto.conc_mol(
            gto.M(
                atom=[("He", p) for p in positions],
                basis={"He": [[0, [1.4, 1.0]], [0, [0.45, 1.0]]]},
                ecp={"He": [0, [[-1, [[], [], []]]] + channels[1:]]},
                unit="Bohr",
                cart=True,
                verbose=0,
            ),
            auxmol,
        )
        nonlocal_part -= nonlocal_mol.intor("ECPscalar")
    fock = ri.T @ (core + jao - 0.5 * kao) @ ri
    nonlocal_part = ri.T @ nonlocal_part @ ri
    if mixed_fock:
        # A finite-rank Delta acts on the product Hamiltonian. Keeping -Delta
        # in n leaves its local part unchanged, so the unit geminal block uses
        # the same independently integrated f^2 and kinetic commutator.
        delta = np.zeros_like(fock)
        delta[:nm, :nm] = np.diag(hf.mo_energy) - fock[:nm, :nm]
        fock += delta
        nonlocal_part -= delta
    fpair = np.kron(fock, identity) + np.kron(identity, fock)
    local_pair = np.kron(fock + nonlocal_part, identity) + np.kron(
        identity, fock + nonlocal_part
    )
    npair = local_pair - fpair
    # Unit geminal Hamiltonian uses independent f² and kinetic-commutator
    # operators. Projection is a single block-matrix congruence, with no
    # production pair expansion or Schur-complement formulas.
    # Psi4's Mints operator is [f,[T1,f]], half the full two-electron
    # [f,[T1+T2,f]]. A centered-S analytic limit independently fixes this factor.
    unit = (
        comm
        + 0.5
        * (
            local_pair[:, occupied_products].T @ squared
            + squared.T @ local_pair[:, occupied_products]
        )
        - r.T @ npair @ r
    )
    gram = np.block([[np.eye(nr**2), r], [r.T, squared[occupied_products]]])
    hamiltonian = np.block([[fpair, fpair @ r], [r.T @ fpair, unit]])
    q = np.vstack((-removed @ r, np.eye(no**2)))
    x = q.T @ gram @ q
    b = q.T @ hamiltonian @ q
    v = gf[:, occupied_products] - projector_coulomb @ removed @ r
    c = (fpair @ (np.eye(nr**2) - removed) @ r)[virtual_products]
    ev, rot = np.linalg.eigh(fock[no:, no:])
    singles = fock[:no, no:] @ rot
    e_cabs = np.sum(2 * singles**2 / (hf.mo_energy[:no, None] - ev))

    for fc in [0, 1]:
        nc = no - fc
        active_pairs = [i * no + j for i in range(fc, no) for j in range(fc, no)]
        active_bras = [i * nr + j for i in range(fc, nm) for j in range(fc, nm)]
        vv = v[np.ix_(active_bras, active_pairs)].T.reshape(nc, nc, nm - fc, nm - fc)
        cv = c[:, active_pairs].T.reshape(nc, nc, nv, nv)
        cusp = np.zeros((nc, nc, nc, nc))
        for i in range(nc):
            for j in range(nc):
                cusp[i, j, i, j] += 0.375
                cusp[i, j, j, i] += 0.125
        spin_cusp = 2 * cusp - cusp.transpose(0, 1, 3, 2)
        source1 = np.einsum(
            "mnak,ikmn->ia", vv[:, :, nc:, :nc], 2 * cusp - cusp.transpose(1, 0, 2, 3)
        )

        def sources(t1):
            def projected(scale):
                left = np.hstack((-scale * t1.T, np.eye(nv)))
                return np.einsum(
                    "ap,mnpq,bq,ijmn->ijab", left, vv, left, cusp, optimize=True
                )

            return source1, projected(0) + (
                projected(1) - projected(-1)
            ) / 2 + np.einsum("mnab,ijmn->ijab", cv, cusp)

        solver = cc.CCSD(hf, frozen=fc)
        solver.conv_tol, solver.conv_tol_normt, solver.max_cycle = 1e-14, 1e-12, 300
        eris = solver.ao2mo()
        d1 = eris.mo_energy[:nc, None] - eris.mo_energy[None, nc:]
        d2 = d1[:, None, :, None] + d1[None, :, None, :]
        ordinary_update = solver.update_amps

        def update(t1, t2, eris):
            u1, u2 = ordinary_update(t1, t2, eris)
            s1, s2 = sources(t1)
            return u1 + s1 / d1, u2 + s2 / d2

        solver.update_amps = update
        solver.kernel(eris=eris)
        assert solver.converged
        t1, t2 = solver.t1, solver.t2
        u1, u2 = update(t1, t2, eris)
        residual1, residual2 = (u1 - t1) * d1, (u2 - t2) * d2
        assert max(np.max(np.abs(residual1)), np.max(np.abs(residual2))) < 1e-11
        tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
        coupling_a = np.einsum("ijmn,mnab,ijab->", spin_cusp, cv, tau)
        additional_b = (
            np.einsum("ijmn,mnab,ijab->", spin_cusp, vv[:, :, nc:, nc:], tau)
            + np.einsum("ijmn,mnib,jb->", spin_cusp, vv[:, :, :nc, nc:], t1)
            + np.einsum("ijmn,mnaj,ia->", spin_cusp, vv[:, :, nc:, :nc], t1)
        )
        active_v = v[
            np.ix_(
                [i * nr + j for i in range(fc, no) for j in range(fc, no)], active_pairs
            )
        ]
        active_b, active_x = (
            b[np.ix_(active_pairs, active_pairs)],
            x[np.ix_(active_pairs, active_pairs)],
        )
        fixed = 0.0
        mp2f12 = 0.0
        zf = cusp.reshape(nc**2, nc**2)
        spin_zf = spin_cusp.reshape(nc**2, nc**2)
        gvv = orbital_coulomb[
            np.ix_(
                virtual_products,
                [i * nr + j for i in range(fc, no) for j in range(fc, no)],
            )
        ]
        active_c = c[:, active_pairs]
        for ij, (i, j) in enumerate((i, j) for i in range(nc) for j in range(nc)):
            eps_sum = hf.mo_energy[fc + i] + hf.mo_energy[fc + j]
            op = active_b - eps_sum * active_x
            fixed += spin_zf[ij] @ (2 * active_v[ij] + op @ zf[ij])
            gaps = -d2[i, j].ravel()
            amp = -(gvv[:, ij] + active_c @ zf[ij]) / gaps
            exchange_amp = -(gvv[:, j * nc + i] + active_c @ zf[j * nc + i]) / gaps
            coefficients = np.concatenate((amp, zf[ij]))
            spin_coefficients = np.concatenate((2 * amp - exchange_amp, spin_zf[ij]))
            op_full = np.block([[np.diag(gaps), active_c], [active_c.T, op]])
            source = np.concatenate((gvv[:, ij], active_v[ij]))
            mp2f12 += spin_coefficients @ (2 * source + op_full @ coefficients)
        ordinary_mp2 = mp.MP2(hf, frozen=fc).run().e_corr
        if fit_orbital_mp2:
            gij = gvv.T.reshape(nc, nc, nv, nv)
            ordinary_mp2 = np.sum(gij * (2 * gij - gij.transpose(0, 1, 3, 2)) / d2)
        t = solver.ccsd_t(eris=eris)
        scaled_t = t * mp2f12 / ordinary_mp2
        cases.append(
            dict(
                nonlocal_ecp=ecp,
                frozen_core=fc,
                mo_coefficients=hf.mo_coeff.tolist(),
                e_hf=hf.e_tot,
                e_cabs=float(e_cabs),
                e_cc=float(solver.e_corr),
                e_fixed=float(fixed),
                e_coupling_a=float(coupling_a),
                e_additional_b=float(additional_b),
                e_mp2=float(ordinary_mp2),
                e_mp2_f12=float(mp2f12),
                e_triples=float(t),
                e_scaled_triples=float(scaled_t),
                e_total_a=float(
                    hf.e_tot + e_cabs + solver.e_corr + fixed + coupling_a + scaled_t
                ),
                e_total_b=float(
                    hf.e_tot
                    + e_cabs
                    + solver.e_corr
                    + fixed
                    + coupling_a
                    + additional_b
                    + scaled_t
                ),
                t1=t1.ravel().tolist(),
                t2=t2.ravel().tolist(),
                residual1=residual1.ravel().tolist(),
                residual2=residual2.ravel().tolist(),
            )
        )
        if fitted_reference:
            pairs = []
            for i in range(fc, no):
                for j in range(i, no):
                    ij, ji = i * no + j, j * no + i
                    bra = i * nr + j
                    gaps = (
                        hf.mo_energy[no:, None]
                        + hf.mo_energy[None, no:]
                        - hf.mo_energy[i]
                        - hf.mo_energy[j]
                    )
                    ci, cj = c[:, ij].reshape(nv, nv), c[:, ji].reshape(nv, nv)
                    gi = orbital_coulomb[virtual_products, bra].reshape(nv, nv)
                    eps_sum = hf.mo_energy[i] + hf.mo_energy[j]
                    raw_v = np.array([v[bra, ij], v[bra, ji]])
                    raw_x = np.array([x[ij, ij], x[ij, ji]])
                    raw_b = np.array([b[ij, ij], b[ij, ji]])
                    vbar = raw_v - [np.sum(gi * ci / gaps), np.sum(gi * cj / gaps)]
                    bbar = (
                        raw_b
                        - eps_sum * raw_x
                        - [np.sum(ci * ci / gaps), np.sum(ci * cj / gaps)]
                    )
                    pairs.append(
                        dict(
                            pair=[i, j],
                            v=raw_v.tolist(),
                            x=raw_x.tolist(),
                            b=raw_b.tolist(),
                            v_bar=vbar.tolist(),
                            b_bar=bbar.tolist(),
                            c=ci.tolist(),
                            g=gi.tolist(),
                        )
                    )
            full_obs_bras = [i * nr + j for i in range(nm) for j in range(nm)]
            cases[-1]["pairs"] = pairs
            cases[-1]["projected_coulomb"] = (
                v[np.ix_(full_obs_bras, active_pairs)].T.reshape(nc**2, nm, nm).tolist()
            )
            hybrid_nonlocal = nonlocal_part.copy()
            hybrid_nonlocal[:no, nm:] += fock[:no, nm:]
            hybrid_nonlocal[nm:, :no] += fock[nm:, :no]
            hybrid_pair = np.kron(hybrid_nonlocal, identity) + np.kron(
                identity, hybrid_nonlocal
            )
            resolutions = {
                "hy1": np.eye(nr**2) - removed,
                "hy2": np.diag(
                    [p >= nm and q >= nm for p in range(nr) for q in range(nr)]
                ),
            }
            hybrids = {}
            for label, resolution in resolutions.items():
                delta = r.T @ hybrid_pair @ resolution @ r
                delta = 0.5 * (delta + delta.T)
                active_delta = delta[np.ix_(active_pairs, active_pairs)]
                shift = sum(spin_zf[ij] @ active_delta @ zf[ij] for ij in range(nc**2))
                hybrids[label] = dict(
                    energy_shift=float(shift),
                    b_shift=[
                        [
                            float(delta[i * no + j, i * no + j]),
                            float(delta[i * no + j, j * no + i]),
                        ]
                        for i in range(fc, no)
                        for j in range(i, no)
                    ],
                )
            cases[-1]["hybrids"] = hybrids
        print(
            "CCSD-F12 reference",
            ecp,
            fc,
            cases[-1]["e_total_a"],
            cases[-1]["e_total_b"],
            flush=True,
        )

output = Path(
    os.environ.get(
        "HF_RS_CCSD_F12_REFERENCE_OUTPUT", root / "tests/data/ccsd_f12_reference.json"
    )
)
output.write_text(
    json.dumps(
        dict(
            source=f"Psi4 {arrays['version']} Gaussian AO operators and PySCF {pyscf.__version__} modified CCSD iteration; explicit product/geminal Gram Hamiltonian",
            approximation="3C(FIX), full RI exchange/nonlocal ECP products, no published hybrid defaults",
            fitted_correction=bool(fitted_reference),
            positions=positions,
            obs_g94=obs,
            aux_g94=aux,
            terms=terms,
            cases=cases,
        ),
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
