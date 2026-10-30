# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent water F12 correction: lower-center integrals and block operators.

Only public Psi4/PySCF execution APIs and their returned arrays are used.
The original-auxiliary robust formula and product/geminal Gram-Hamiltonian
congruence avoid union-AO N^4 integrals and dense RI-product Hamiltonians.
The default orbital MP2/CCSD/(T) and extended Fock are exact; explicit flags
select physical spherical fits, fitted MP2 and a consistently mixed DF Fock.
These approximations are not a published external-program default. Scratch
stays in target.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--integrals-only", action="store_true")
parser.add_argument("--reuse-integrals", action="store_true")
parser.add_argument("--mixed-fock", action="store_true")
parser.add_argument("--spherical-fitting", action="store_true")
parser.add_argument("--fit-orbital-mp2", action="store_true")
parser.add_argument(
    "--psi4-python",
    default=os.environ.get("HF_RS_PSI4_PYTHON"),
    help="Python executable with Psi4 (or set HF_RS_PSI4_PYTHON)",
)
args = parser.parse_args()
if not args.integrals_only and not args.reuse_integrals and not args.psi4_python:
    parser.error("Provide --psi4-python or set HF_RS_PSI4_PYTHON")
root = Path(__file__).resolve().parents[2]
work = root / "target/f12-fitted-water-reference"
work.mkdir(parents=True, exist_ok=True)
case = json.loads(
    (root / "tests/data/psi4_mp2_f12_published_reference.json").read_text()
)["cases"][0]
atoms = case["atoms"]
symbols = sorted({atom[0] for atom in atoms})
basis_source = (root / "src/basis_data.rs").read_text(encoding="utf-8")
fit_g94 = re.search(
    r'pub const AUX_AUG_CC_PVDZ_RIFIT_GBS: &str = r#"(.*?)"#;', basis_source, re.S
).group(1)


def segment(text, symbol):
    return re.search(rf"(?m)^\s*{symbol}\s+0\s*\n(.*?)\*\*\*\*", text, re.S).group(1)


if args.integrals_only:
    import psi4

    os.chdir(work)
    psi4.set_num_threads(4)
    psi4.set_memory("512 MiB")
    psi4.set_options({"ints_tolerance": 1e-16, "screening": "none"})
    psi4.core.set_output_file(str(work / "mints.out"), False)
    molecule = psi4.geometry(
        "0 1\n"
        + "\n".join(s + " " + " ".join(map(str, p)) for s, p in atoms)
        + "\nunits bohr\nsymmetry c1\nno_com\nno_reorient"
    )
    union_g94 = "\n".join(
        s
        + " 0\n"
        + segment(case["obs_g94"], s)
        + segment(case["aux_g94"], s)
        + "****\n"
        for s in symbols
    )
    bases = []
    for name, g94 in [("hf_rs_water_union", union_g94), ("hf_rs_water_fit", fit_g94)]:
        (work / f"{name}.gbs").write_text("cartesian\n****\n" + g94, encoding="utf-8")
        bases.append(psi4.core.BasisSet.build(molecule, "BASIS", name, puream=0))
    union, fitting = bases
    mints = psi4.core.MintsHelper(union)
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    n, na = union.nbf(), fitting.nbf()
    overlap = np.asarray(mints.ao_overlap(union, union))
    fit_overlap = np.asarray(mints.ao_overlap(fitting, fitting))
    norm, norm_fit = np.sqrt(np.diag(overlap)), np.sqrt(np.diag(fit_overlap))

    def metadata(basis):
        result = []
        for i in range(basis.nshell()):
            shell = basis.shell(i)
            result.append(
                dict(
                    center=atoms[shell.ncenter][1],
                    angular=shell.am,
                    exponents=[shell.exp(k) for k in range(shell.nprimitive)],
                    coefficients=[
                        shell.original_coef(k) for k in range(shell.nprimitive)
                    ],
                    offset=basis.shell_to_basis_function(i),
                )
            )
        return json.dumps(result)

    def two(function):
        return (
            np.asarray(function(fitting, zero, fitting, zero)).reshape(na, na)
            / norm_fit[:, None]
            / norm_fit[None, :]
        )

    def three(function):
        return (
            np.asarray(function(fitting, zero, union, union)).reshape(na, n, n)
            / norm_fit[:, None, None]
            / norm[None, :, None]
            / norm[None, None, :]
        )

    np.savez(
        work / "coulomb.npz",
        two=two(mints.ao_eri),
        three=three(mints.ao_eri),
        overlap=overlap / norm[:, None] / norm[None, :],
        union_metadata=np.array(metadata(union)),
        fit_metadata=np.array(metadata(fitting)),
        version=np.array(psi4.__version__),
    )
    for name, method in [
        ("factor", mints.ao_f12),
        ("square", mints.ao_f12_squared),
        ("gf", mints.ao_f12g12),
        ("half_comm", mints.ao_f12_double_commutator),
    ]:
        print("Psi4 water lower-center kernel", name, flush=True)

        def function(*bases, method=method):
            return method(case["terms"], *bases)

        np.savez(work / f"{name}.npz", two=two(function), three=three(function))
    print("Public lower-center arrays complete", n, na, flush=True)
    sys.exit(0)

if not args.reuse_integrals:
    subprocess.run(
        [args.psi4_python, str(Path(__file__).resolve()), "--integrals-only"],
        check=True,
    )

import pyscf  # noqa: E402
from pyscf import cc, gto, lib, mp, scf  # noqa: E402
from pyscf.gto.basis import parse_gaussian  # noqa: E402

lib.num_threads(4)


def definitions(text):
    return {s: parse_gaussian.parse(segment(text, s), optimize=False) for s in symbols}


mol = gto.M(
    atom=atoms,
    basis=definitions(case["obs_g94"]),
    unit="Bohr",
    cart=False,
    verbose=0,
    max_memory=512,
)
hf = scf.RHF(mol).run(conv_tol=1e-13, conv_tol_grad=1e-11, max_cycle=200)
if not hf.converged:
    hf = hf.newton().run(conv_tol=1e-13, conv_tol_grad=1e-11)
assert hf.converged
cart = mol.copy()
cart.cart = True
aux = gto.M(
    atom=[("ghost-" + s, p) for s, p in atoms],
    basis=definitions(case["aux_g94"]),
    unit="Bohr",
    cart=True,
    verbose=0,
)
fitting = gto.M(
    atom=[("ghost-" + s, p) for s, p in atoms],
    basis=definitions(fit_g94),
    unit="Bohr",
    cart=True,
    verbose=0,
)
union = gto.conc_mol(cart, aux)
overlap = union.intor("int1e_ovlp")
raw = np.load(work / "coulomb.npz")


def match_shells(molecule, metadata):
    """Match public shell data; overlap independently verifies the AO mapping."""
    metadata = json.loads(str(metadata))
    offsets = molecule.ao_loc_nr(cart=True)
    mapping, phases, used = [], [], set()
    for item in metadata:
        candidates = []
        for shell in range(molecule.nbas):
            if shell in used or molecule.bas_angular(shell) != item["angular"]:
                continue
            center = molecule.atom_coord(molecule.bas_atom(shell))
            if np.max(np.abs(center - item["center"])) > 1e-12:
                continue
            exponents = molecule.bas_exp(shell)
            if len(exponents) != len(item["exponents"]) or not np.allclose(
                exponents, item["exponents"], atol=1e-12, rtol=1e-12
            ):
                continue
            assert molecule.bas_nctr(shell) == 1
            c, d = molecule.bas_ctr_coeff(shell)[:, 0], np.array(item["coefficients"])
            alignment = c @ d / np.linalg.norm(c) / np.linalg.norm(d)
            if abs(abs(alignment) - 1) < 1e-10:
                candidates.append((shell, np.sign(alignment)))
        assert candidates, item
        shell, sign = candidates[0]
        used.add(shell)
        mapping.extend(range(offsets[shell], offsets[shell + 1]))
        phases.extend([sign] * (offsets[shell + 1] - offsets[shell]))
    assert len(set(mapping)) == molecule.nao
    return np.array(mapping), np.array(phases)


iu, su = match_shells(union, raw["union_metadata"])
ia, sa = match_shells(fitting, raw["fit_metadata"])
norm = np.sqrt(np.diag(overlap))
independent_overlap = (
    overlap[np.ix_(iu, iu)]
    / norm[iu, None]
    / norm[None, iu]
    * su[:, None]
    * su[None, :]
)
assert np.max(np.abs(independent_overlap - raw["overlap"])) < 1e-11
nao, nm, no, fc = cart.nao, hf.mo_coeff.shape[1], mol.nelectron // 2, 1
nc, nv = no - fc, nm - no
co = mol.cart2sph_coeff(normalized="sp") @ hf.mo_coeff
projection = co @ co.T @ overlap[:nao, nao:]
metric_cabs = overlap[nao:, nao:] - overlap[nao:, :nao] @ projection
spherical = aux.cart2sph_coeff(normalized="sp")
values, vectors = np.linalg.eigh(spherical.T @ metric_cabs @ spherical)
retained = values > 1e-8
complement = spherical @ (vectors[:, retained] / np.sqrt(values[retained]))
ri = np.zeros((union.nao, nm + complement.shape[1]))
ri[:nao, :nm] = co
ri[:, nm:] = np.vstack((-projection @ complement, complement))
nr, npair = ri.shape[1], no**2
# Reorthogonalize within the retained CABS subspace. Extended-precision Gram
# products prevent cancellation in almost dependent raw auxiliary functions.
overlap_extended = overlap.astype(np.longdouble)
for _ in range(2):
    complement = ri[:, nm:]
    residual = (
        ri[:, :nm].astype(np.longdouble).T
        @ overlap_extended
        @ complement.astype(np.longdouble)
    )
    complement = complement - ri[:, :nm] @ np.asarray(residual, dtype=float)
    gram = (
        complement.astype(np.longdouble).T
        @ overlap_extended
        @ complement.astype(np.longdouble)
    )
    lower = np.linalg.cholesky(np.asarray(gram, dtype=float))
    ri[:, nm:] = np.linalg.solve(lower, complement.T).T
gram = ri.astype(np.longdouble).T @ overlap_extended @ ri.astype(np.longdouble)
orthonormality_error = float(np.max(np.abs(gram - np.eye(nr))))
assert orthonormality_error < 1e-9
coefficients = su[:, None] * norm[iu, None] * ri[iu]
metric = raw["two"]
physical_fit = np.eye(len(metric))
if args.spherical_fitting:
    fit_norm = np.sqrt(np.diag(fitting.intor("int1e_ovlp")))
    physical_fit = (
        sa[:, None] * fit_norm[ia, None] * fitting.cart2sph_coeff(normalized="sp")[ia]
    )
    metric = physical_fit.T @ metric @ physical_fit
values, vectors = np.linalg.eigh(metric)


def solve_metric(rhs):
    # This water metric has condition number ~1.5e9. Forming its explicit
    # spectral inverse loses digits in V although the retained rank is correct.
    if np.all(values > 1e-10):
        return np.linalg.solve(metric, rhs)
    retained_vectors = vectors[:, values > 1e-10]
    return retained_vectors @ (
        (retained_vectors.T @ rhs) / values[values > 1e-10, None]
    )


three = np.einsum("PA,Pab->Aab", physical_fit, raw["three"], optimize=True)
tc = np.einsum(
    "Aab,ap,bi->Api", three, coefficients, coefficients[:, :no], optimize=True
)
fitting_coefficients = solve_metric(tc.reshape(len(metric), -1))
d = fitting_coefficients


def fit_operator(name):
    arrays = np.load(work / f"{name}.npz")
    w3 = np.einsum(
        "Aab,ap,bi->Api",
        np.einsum("PA,Pab->Aab", physical_fit, arrays["three"], optimize=True),
        coefficients,
        coefficients[:, :no],
        optimize=True,
    ).reshape(len(metric), -1)
    two = physical_fit.T @ arrays["two"] @ physical_fit
    operator = d.T @ w3 + w3.T @ d - d.T @ two @ d
    return operator.reshape(nr, no, nr, no).transpose(0, 2, 1, 3).reshape(nr**2, no**2)


r = fit_operator("factor")
squared = fit_operator("square")
gf = fit_operator("gf")
occupied_products = np.array([i * nr + j for i in range(no) for j in range(no)])
half_comm = fit_operator("half_comm")[occupied_products]
obs_products = np.array([p * nr + q for p in range(nm) for q in range(nm)])
virtual_products = np.array([p * nr + q for p in range(no, nm) for q in range(no, nm)])
mask = np.array(
    [
        (p < nm and q < nm) or (p >= nm and q < no) or (p < no and q >= nm)
        for p in range(nr)
        for q in range(nr)
    ]
)
pr = mask[:, None] * r
dm = np.zeros((union.nao, union.nao))
dm[:nao, :nao] = 2 * co[:, :no] @ co[:, :no].T
jao, kao = scf.hf.get_jk(union, dm)
fock = (
    ri.T @ (union.intor("int1e_kin") + union.intor("int1e_nuc") + jao - 0.5 * kao) @ ri
)
nonlocal_part = ri.T @ (0.5 * kao) @ ri
if args.mixed_fock:
    # Independent original-auxiliary Fock in the orthogonal resolution space.
    # All raw contributions share the same metric before the OBS restoration.
    ri_three = np.einsum(
        "Aab,ap,bq->Apq", three, coefficients, coefficients, optimize=True
    )
    charge = 2 * np.einsum("Aii->A", ri_three[:, :no, :no])
    j_ri = np.einsum("Apq,A->pq", ri_three, solve_metric(charge[:, None])[:, 0])
    occupied_half = ri_three[:, :, :no]
    solved_half = solve_metric(occupied_half.reshape(len(metric), -1)).reshape(
        occupied_half.shape
    )
    k_ri = np.einsum("Api,Aqi->pq", occupied_half, solved_half, optimize=True)
    fock = (
        ri.T @ (union.intor("int1e_kin") + union.intor("int1e_nuc")) @ ri + j_ri - k_ri
    )
    delta = np.zeros_like(fock)
    delta[:nm, :nm] = np.diag(hf.mo_energy) - fock[:nm, :nm]
    fock += delta
    nonlocal_part = k_ri - delta
    del ri_three, occupied_half, solved_half


def pair_action(one_body, vectors):
    tensor = vectors.reshape(nr, nr, -1)
    return (
        np.einsum("pr,rqt->pqt", one_body, tensor, optimize=True)
        + np.einsum("qs,pst->pqt", one_body, tensor, optimize=True)
    ).reshape(nr**2, -1)


unit_vectors = np.zeros((nr**2, npair))
unit_vectors[occupied_products, np.arange(npair)] = 1
local_action = pair_action(fock + nonlocal_part, unit_vectors)
unit = (
    half_comm
    + 0.5 * (local_action.T @ squared + squared.T @ local_action)
    - r.T @ pair_action(nonlocal_part, r)
)
# Evaluate Q^T G Q and Q^T H Q as block-operator congruences. The product-space
# identity and F1+F2 are applied without allocating their squared dimensions.
x = squared[occupied_products] - pr.T @ r - r.T @ pr + pr.T @ pr
b = (
    unit
    - pr.T @ pair_action(fock, r)
    - r.T @ pair_action(fock, pr)
    + pr.T @ pair_action(fock, pr)
)
c = pair_action(fock, r - pr)[virtual_products]
raw_obs_ri = np.einsum(
    "Aab,ap,br->Apr", three, coefficients[:, :nm], coefficients, optimize=True
)
fit_obs_ri = solve_metric(raw_obs_ri.reshape(len(metric), -1)).reshape(raw_obs_ri.shape)
v_obs = gf[obs_products].copy()
for ij in range(npair):
    v_obs[:, ij] -= np.einsum(
        "Apr,rs,Aqs->pq",
        fit_obs_ri,
        pr[:, ij].reshape(nr, nr),
        raw_obs_ri,
        optimize=True,
    ).ravel()
ev, rot = np.linalg.eigh(fock[no:, no:])
singles = fock[:no, no:] @ rot
e_cabs = np.sum(2 * singles**2 / (hf.mo_energy[:no, None] - ev))
pt = mp.MP2(hf, frozen=fc)
e_mp2 = pt.kernel()[0]
gvv = np.asarray(pt.ao2mo().ovov).reshape(nc, nv, nc, nv).transpose(0, 2, 1, 3)
if args.fit_orbital_mp2:
    virtual_half = tc[:, no:nm, fc:no]
    solved_half = solve_metric(virtual_half.reshape(len(metric), -1)).reshape(
        virtual_half.shape
    )
    gvv = np.einsum("Aai,Abj->ijab", virtual_half, solved_half, optimize=True)
    e1 = hf.mo_energy[fc:no, None] - hf.mo_energy[None, no:]
    denominators = e1[:, None, :, None] + e1[None, :, None, :]
    e_mp2 = np.sum(gvv * (2 * gvv - gvv.transpose(0, 1, 3, 2)) / denominators)
active_pairs = np.array([i * no + j for i in range(fc, no) for j in range(fc, no)])
active_obs = np.array([p * nm + q for p in range(fc, nm) for q in range(fc, nm)])
v = v_obs[np.ix_(active_obs, active_pairs)].T.reshape(nc, nc, nm - fc, nm - fc)
cv = c[:, active_pairs].T.reshape(nc, nc, nv, nv)
cusp = np.zeros((nc, nc, nc, nc))
for i in range(nc):
    for j in range(nc):
        cusp[i, j, i, j] += 0.375
        cusp[i, j, j, i] += 0.125
spin_cusp = 2 * cusp - cusp.transpose(0, 1, 3, 2)
zf, spin_zf = cusp.reshape(nc**2, nc**2), spin_cusp.reshape(nc**2, nc**2)
active_v = np.array(
    [
        [v_obs[i * nm + j, ket] for ket in active_pairs]
        for i in range(fc, no)
        for j in range(fc, no)
    ]
)
active_b, active_x = (
    b[np.ix_(active_pairs, active_pairs)],
    x[np.ix_(active_pairs, active_pairs)],
)
active_c = c[:, active_pairs]
fixed, mp2f12 = 0.0, 0.0
for ij, (i, j) in enumerate((i, j) for i in range(nc) for j in range(nc)):
    eps_sum = hf.mo_energy[fc + i] + hf.mo_energy[fc + j]
    op = active_b - eps_sum * active_x
    gaps = (hf.mo_energy[no:, None] + hf.mo_energy[None, no:] - eps_sum).ravel()
    fixed += spin_zf[ij] @ (2 * active_v[ij] + op @ zf[ij])
    amp = -(gvv[i, j].ravel() + active_c @ zf[ij]) / gaps
    exchanged = -(gvv[j, i].ravel() + active_c @ zf[j * nc + i]) / gaps
    source = np.concatenate((gvv[i, j].ravel(), active_v[ij]))
    spin_coefficients = np.concatenate((2 * amp - exchanged, spin_zf[ij]))
    action = np.concatenate(
        (gaps * amp + active_c @ zf[ij], active_c.T @ amp + op @ zf[ij])
    )
    mp2f12 += spin_coefficients @ (2 * source + action)
print("Water MP2/F12 reference", e_mp2, mp2f12 - e_mp2, e_cabs, flush=True)

source1 = np.einsum(
    "mnak,ikmn->ia", v[:, :, nc:, :nc], 2 * cusp - cusp.transpose(1, 0, 2, 3)
)


def sources(t1):
    def projected(scale):
        left = np.hstack((-scale * t1.T, np.eye(nv)))
        return np.einsum("ap,mnpq,bq,ijmn->ijab", left, v, left, cusp, optimize=True)

    return source1, projected(0) + 0.5 * (projected(1) - projected(-1)) + np.einsum(
        "mnab,ijmn->ijab", cv, cusp
    )


solver = cc.CCSD(hf, frozen=fc)
solver.conv_tol, solver.conv_tol_normt, solver.max_cycle = 1e-13, 1e-11, 200
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
residual = max(np.max(abs((u1 - t1) * d1)), np.max(abs((u2 - t2) * d2)))
assert residual < 1e-10
tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
coupling_a = np.einsum("ijmn,mnab,ijab->", spin_cusp, cv, tau)
additional_b = (
    np.einsum("ijmn,mnab,ijab->", spin_cusp, v[:, :, nc:, nc:], tau)
    + np.einsum("ijmn,mnib,jb->", spin_cusp, v[:, :, :nc, nc:], t1)
    + np.einsum("ijmn,mnaj,ia->", spin_cusp, v[:, :, nc:, :nc], t1)
)
triples = solver.ccsd_t(eris=eris)
scaled = triples * mp2f12 / e_mp2
pairs = []
for i in range(fc, no):
    for j in range(i, no):
        ij, ji = i * no + j, j * no + i
        ci, cj = c[:, ij].reshape(nv, nv), c[:, ji].reshape(nv, nv)
        gi = gvv[i - fc, j - fc]
        gaps = (
            hf.mo_energy[no:, None]
            + hf.mo_energy[None, no:]
            - hf.mo_energy[i]
            - hf.mo_energy[j]
        )
        raw_v, raw_x, raw_b = (
            np.array([v_obs[i * nm + j, ij], v_obs[i * nm + j, ji]]),
            np.array([x[ij, ij], x[ij, ji]]),
            np.array([b[ij, ij], b[ij, ji]]),
        )
        pairs.append(
            dict(
                pair=[i, j],
                v=raw_v.tolist(),
                x=raw_x.tolist(),
                b=raw_b.tolist(),
                v_bar=(
                    raw_v - [np.sum(gi * ci / gaps), np.sum(gi * cj / gaps)]
                ).tolist(),
                b_bar=(
                    raw_b
                    - (hf.mo_energy[i] + hf.mo_energy[j]) * raw_x
                    - [np.sum(ci**2 / gaps), np.sum(ci * cj / gaps)]
                ).tolist(),
                c=ci.tolist(),
                g=gi.tolist(),
            )
        )
native_order = []
offsets = cart.ao_loc_nr(cart=True)
for shell in range(cart.nbas):
    count = offsets[shell + 1] - offsets[shell]
    native_order.extend(
        offsets[shell] + k
        for k in ([0, 3, 5, 1, 2, 4] if cart.bas_angular(shell) == 2 else range(count))
    )
native_coefficients = (
    np.sqrt(np.diag(overlap)[:nao])[native_order, None] * co[native_order]
)
output = dict(
    source=f"Public Psi4 {str(raw['version'])} lower-center integrals, NumPy robust fit and block Gram-Hamiltonian operators; public PySCF {pyscf.__version__} modified CCSD iteration",
    input_fixture="psi4_mp2_f12_published_reference.json#cases/0",
    fitting_basis="aug-cc-pvdz-rifit",
    mixed_fock=args.mixed_fock,
    spherical_fitting=args.spherical_fitting,
    fit_orbital_mp2=args.fit_orbital_mp2,
    fit_sha256=hashlib.sha256(fit_g94.encode()).hexdigest(),
    spherical=True,
    frozen_core=fc,
    n_occ=no,
    n_mo=nm,
    n_cabs=nr - nm,
    n_fit=len(metric),
    fit_rank=int(sum(values > 1e-10)),
    metric_solver="NumPy direct linear solve; spectral retained-space solve only if rank deficient",
    metric_condition_number=float(values[-1] / values[0]),
    maximum_ri_orthonormality_error=orthonormality_error,
    psi4_options=dict(ints_tolerance=1e-16, screening="none"),
    mo_coefficients=native_coefficients.tolist(),
    pairs=pairs,
    projected_coulomb=v_obs[:, active_pairs].T.reshape(nc**2, nm, nm).tolist(),
    t1=t1.ravel().tolist(),
    t2=t2.ravel().tolist(),
    maximum_cc_residual=float(residual),
    e_hf=float(hf.e_tot),
    e_cabs=float(e_cabs),
    e_mp2=float(e_mp2),
    e_f12=float(mp2f12 - e_mp2),
    e_cc=float(solver.e_corr),
    e_fixed=float(fixed),
    e_coupling_a=float(coupling_a),
    e_additional_b=float(additional_b),
    e_triples=float(triples),
    e_scaled_triples=float(scaled),
    e_total_mp2=float(hf.e_tot + e_cabs + mp2f12),
    e_total_a=float(hf.e_tot + e_cabs + solver.e_corr + fixed + coupling_a + scaled),
    e_total_b=float(
        hf.e_tot + e_cabs + solver.e_corr + fixed + coupling_a + additional_b + scaled
    ),
    scope="Explicit correction/Fock/orbital-MP2 fitting controls with exact orbital CCSD/(T); not a published/default benchmark.",
)
destination = (
    "f12_mixed_fock_water_reference.json"
    if args.mixed_fock
    else "f12_fitted_water_reference.json"
)
(root / "tests/data" / destination).write_text(
    json.dumps(output, indent=1) + "\n", encoding="utf-8"
)
print(
    "Water CCSD(T*)-F12 reference",
    output["e_total_a"],
    output["e_total_b"],
    "residual",
    residual,
    flush=True,
)
