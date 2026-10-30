# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
#
# This code is part of the hf-rs library.

"""Independent radial-wavefunction/extended-Gram-space MP2-F12 reference.

The unprojected local-Fock quadratic form is integrated from the actual gradient
of f(r12)*phi(r1)*phi(r2), not a double-commutator contraction. The RI local
potential has the required action on the single occupied orbital. Explicit
product-space/extra-geminal Gram and Hamiltonian matrices apply the projector and
solve conventional amplitudes independently of the Rust intermediate expansion.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pyscf
from numpy.polynomial.legendre import leggauss
from pyscf import gto, lib, mp, scf

lib.num_threads(4)
obs_exponents = np.array([1.4, 0.45])
aux_exponents = np.array([0.2, 0.9, 3.2])
exponents = np.concatenate((obs_exponents, aux_exponents))
normalization = (2 * exponents / np.pi) ** 0.75
terms_short = [(0.3, -0.4), (1.2, -0.2)]
fit = json.loads(
    Path(__file__)
    .resolve()
    .parents[1]
    .joinpath("data/f12_slater_fit_reference.json")
    .read_text(encoding="utf-8")
)
terms_slater = fit["cases"][1]["normalized_terms"]
terms_scaled = fit["cases"][0]["normalized_terms"]


def primitive(indices, exponent, over_r=False):
    a, b, c, d = exponents[list(indices)]
    p, q = a + b, c + d
    determinant = p * q + exponent * (p + q)
    prefactor = np.prod(normalization[list(indices)])
    return prefactor * (
        2 * np.pi**2.5 / (determinant * np.sqrt(p + q))
        if over_r
        else np.pi**3 / determinant**1.5
    )


def primitive_kinetic_commutator(indices, gamma):
    a, b, c, d = exponents[list(indices)]
    p, q = a + b, c + d
    determinant = p * q + gamma * (p + q)
    # Direct centered bivariate-Gaussian second moments, without orbital raising.
    return primitive(indices, gamma) * (
        3 * (b + d - a - c)
        - 3
        * ((b * b - a * a) * (q + gamma) + (d * d - c * c) * (p + gamma))
        / determinant
    )


def local_quadratic_form(nrad, nang, terms, co, ri, local, commutator_only=False):
    x, weights = leggauss(nrad)
    radial = 7 * (x + 1)
    weights = 7 * weights
    angular, angular_weights = leggauss(nang)
    values = np.exp(-(radial[:, None] ** 2) * exponents) * normalization
    phi = values @ co
    derivative = (values * (-2 * radial[:, None] * exponents)) @ co
    laplacian = (
        values * (4 * radial[:, None] ** 2 * exponents**2 - 6 * exponents)
    ) @ co
    action = values @ ri @ local[:, 0]
    r2 = radial[:, None]
    phi2 = phi[:, None]
    dphi2 = derivative[:, None]
    uphi2 = (action + 0.5 * laplacian)[:, None]
    measure = 8 * np.pi**2 * r2**2 * weights[:, None] * angular_weights
    total = 0.0
    for index, r1 in enumerate(radial):
        s2 = r1 * r1 + r2 * r2 - 2 * r1 * r2 * angular
        f = sum(c * np.exp(-a * s2) for a, c in terms)
        df_over_s = sum(-2 * a * c * np.exp(-a * s2) for a, c in terms)
        p1, dp1 = phi[index], derivative[index]
        grad1 = (
            f * f * dp1 * dp1 * phi2**2
            + df_over_s**2 * s2 * p1 * p1 * phi2**2
            + 2 * f * df_over_s * dp1 * p1 * phi2**2 * (r1 - r2 * angular)
        )
        grad2 = (
            f * f * dphi2**2 * p1 * p1
            + df_over_s**2 * s2 * p1 * p1 * phi2**2
            + 2 * f * df_over_s * dphi2 * phi2 * p1 * p1 * (r2 - r1 * angular)
        )
        potential = (
            f
            * f
            * (
                (action[index] + 0.5 * laplacian[index]) * p1 * phi2**2
                + uphi2 * phi2 * p1 * p1
            )
        )
        integrand = (
            (df_over_s**2 * s2 * p1 * p1 * phi2**2)
            if commutator_only
            else (0.5 * (grad1 + grad2) + potential)
        )
        total += r1 * r1 * weights[index] * np.sum(measure * integrand)
    return float(total)


cases = []
for name, terms in [
    ("he_gaussian", terms_short),
    ("he_local_ecp", terms_short),
    ("he_nonlocal_ecp", terms_short),
    ("he_slater_fit", terms_slater),
    ("he_slater_fit_scaled", terms_scaled),
]:
    channels = []
    if "ecp" in name:
        channels.append([-1, [[], [], [[0.7, 0.17]]]])
    if name == "he_nonlocal_ecp":
        channels.append([0, [[], [], [[0.53, 0.11]]]])
    ecp = {"He": [0, channels]} if channels else {}
    mol = gto.M(
        atom="He 0.13 -0.21 0.31",
        basis={"He": [[0, [a, 1.0]] for a in obs_exponents]},
        ecp=ecp,
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol).run(conv_tol=1e-14, conv_tol_grad=1e-12)
    assert mf.converged
    aux = gto.M(
        atom="ghost-He 0.13 -0.21 0.31",
        basis={"ghost-He": [[0, [a, 1.0]] for a in aux_exponents]},
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    union = gto.conc_mol(mol, aux)
    nao, nm, no = mol.nao, mf.mo_coeff.shape[1], mol.nelectron // 2
    overlap = union.intor("int1e_ovlp")
    projection = mf.mo_coeff @ mf.mo_coeff.T @ overlap[:nao, nao:]
    eigenvalues, rotation = np.linalg.eigh(
        overlap[nao:, nao:] - overlap[nao:, :nao] @ projection
    )
    caux = rotation[:, eigenvalues > 1e-8] / np.sqrt(eigenvalues[eigenvalues > 1e-8])
    cabs = np.vstack((-projection @ caux, caux))
    ri = np.zeros((union.nao, nm + cabs.shape[1]))
    ri[:nao, :nm] = mf.mo_coeff
    ri[:, nm:] = cabs
    co = ri[:, 0]
    dm = np.zeros_like(overlap)
    dm[:nao, :nao] = mf.make_rdm1()
    core = union.intor("int1e_kin") + union.intor("int1e_nuc")
    if ecp:
        core += union.intor("ECPscalar")
    jao, kao = scf.hf.get_jk(union, dm)
    fock = ri.T @ (core + jao - 0.5 * kao) @ ri
    nonlocal_part = 0.5 * kao
    nonlocal_ecp = np.zeros_like(nonlocal_part)
    if name == "he_nonlocal_ecp":
        nonlocal_ecp = (
            0.11
            * normalization[:, None]
            * normalization[None, :]
            * np.pi**1.5
            / (exponents[:, None] + exponents[None, :] + 0.53) ** 1.5
        )
        nonlocal_part -= nonlocal_ecp
    nonlocal_ecp = ri.T @ nonlocal_ecp @ ri
    nonlocal_part = ri.T @ nonlocal_part @ ri
    nri = ri.shape[1]
    rao = np.zeros((union.nao,) * 4)
    sqao = np.zeros_like(rao)
    gfao = np.zeros_like(rao)
    for indices in itertools.product(range(union.nao), repeat=4):
        rao[indices] = sum(c * primitive(indices, a) for a, c in terms)
        sqao[indices] = sum(
            c * d * primitive(indices, a + b) for a, c in terms for b, d in terms
        )
        gfao[indices] = sum(c * primitive(indices, a, True) for a, c in terms)
    r = np.einsum("ap,b,cq,d,abcd->pq", ri, co, ri, co, rao).ravel()
    f2 = float(np.einsum("a,b,c,d,abcd->", co, co, co, co, sqao))
    gf = float(np.einsum("a,b,c,d,abcd->", co, co, co, co, gfao))
    g = np.einsum("ap,b,cq,d,abcd->pq", ri, co, ri, co, union.intor("int2e")).ravel()
    short = len(terms) == 2
    coarse = local_quadratic_form(
        96 if short else 384, 80 if short else 240, terms, co, ri, fock + nonlocal_part
    )
    fine = local_quadratic_form(
        128 if short else 512,
        100 if short else 320,
        terms,
        co,
        ri,
        fock + nonlocal_part,
    )
    error = abs(coarse - fine)
    assert error < 1e-9, (name, error)
    full = np.kron(fock, np.eye(nri)) + np.kron(np.eye(nri), fock)
    nonlocal_pair = np.kron(nonlocal_part, np.eye(nri)) + np.kron(
        np.eye(nri), nonlocal_part
    )
    # Nonorthogonal product states plus the unprojected geminal wavefunction.
    gram = np.eye(nri * nri + 1)
    gram[:-1, -1] = r
    gram[-1, :-1] = r
    gram[-1, -1] = f2
    hamiltonian = np.zeros_like(gram)
    hamiltonian[:-1, :-1] = full
    hamiltonian[:-1, -1] = full @ r
    hamiltonian[-1, :-1] = r @ full
    hamiltonian[-1, -1] = fine - r @ nonlocal_pair @ r
    q = np.zeros(nri * nri + 1)
    q[-1] = 1
    for p, qindex in itertools.product(range(nri), repeat=2):
        if (
            (p < nm and qindex < nm)
            or (p >= nm and qindex < no)
            or (p < no and qindex >= nm)
        ):
            q[p * nri + qindex] = -r[p * nri + qindex]
    source = np.append(g, gf)
    operator = hamiltonian - 2 * mf.mo_energy[0] * gram
    vv = [p * nri + qindex for p, qindex in itertools.product(range(no, nm), repeat=2)]
    conventional = operator[np.ix_(vv, vv)]
    fixed = 0.5 * q
    t = np.linalg.solve(conventional, -(source + operator @ fixed)[vv])
    wavefunction = fixed.copy()
    wavefunction[vv] += t
    mp2 = float(-source[vv] @ np.linalg.solve(conventional, source[vv]))
    assert abs(mp2 - mp.MP2(mf).run(verbose=0).e_corr) < 1e-12
    correction = float(
        wavefunction @ operator @ wavefunction + 2 * source @ wavefunction - mp2
    )
    ev, u = np.linalg.eigh(fock[no:, no:])
    coupling = fock[:no, no:] @ u
    singles = float(-2 * np.sum(coupling**2 / (ev[None, :] - mf.mo_energy[:no, None])))
    uao = np.zeros_like(rao)
    for indices in itertools.product(range(union.nao), repeat=4):
        uao[indices] = sum(
            c * primitive_kinetic_commutator(indices, a) for a, c in terms
        )
    u = np.einsum("ap,b,cq,d,abcd->pq", ri, co, ri, co, uao).ravel()
    half_commutator = local_quadratic_form(
        128 if short else 512,
        100 if short else 320,
        terms,
        co,
        ri,
        fock + nonlocal_part,
        commutator_only=True,
    )
    coarse_commutator = local_quadratic_form(
        96 if short else 384,
        80 if short else 240,
        terms,
        co,
        ri,
        fock + nonlocal_part,
        commutator_only=True,
    )
    refinement = abs(half_commutator - coarse_commutator)
    assert refinement < 1e-9
    a_b_bar = float(half_commutator - u @ q[:-1])
    # Independent dense pair operators: explicitly multiply the full f operator
    # by V1+V2, including off-diagonal RI actions on either occupied ket leg.
    factor_full = np.einsum("ap,br,cq,ds,abcd->pqrs", ri, ri, ri, ri, rao).reshape(
        nri**2, nri**2
    )
    squared_full = np.einsum("ap,br,cq,ds,abcd->pqrs", ri, ri, ri, ri, sqao).reshape(
        nri**2, nri**2
    )
    potential_pair = np.kron(nonlocal_ecp, np.eye(nri)) + np.kron(
        np.eye(nri), nonlocal_ecp
    )
    nonlocal_commutator = factor_full @ potential_pair - potential_pair @ factor_full
    unit_correction = float(
        r @ potential_pair @ r - potential_pair[0] @ squared_full[:, 0]
    )
    ecp_correction = float(unit_correction - nonlocal_commutator[:, 0] @ q[:-1])
    a_b_bar += ecp_correction
    a_reference = dict(
        b_bar=a_b_bar,
        e_f12=float(source @ q + 0.25 * a_b_bar),
        radial_refinement=refinement,
        nonlocal_b_bar_correction=ecp_correction,
    )
    cases.append(
        dict(
            name=name,
            gamma=0.7 if name == "he_slater_fit_scaled" else 1.0,
            terms=terms,
            e_scf=float(mf.e_tot),
            e_mp2=mp2,
            e_f12=correction,
            e_cabs=singles,
            e_total=float(mf.e_tot + mp2 + correction + singles),
            v=float(source @ q),
            x=float(q @ gram @ q),
            b=float(q @ hamiltonian @ q),
            c=(hamiltonian[vv] @ q).reshape(nm - no, nm - no).tolist(),
            g=g[vv].reshape(nm - no, nm - no).tolist(),
            radial_quadrature_refinement=error,
            three_star_a=a_reference,
        )
    )
    print(name, correction, "quadrature refinement", error, flush=True)
out = dict(
    source=f"PySCF {pyscf.__version__} RHF/Fock/ERI; independent radial-gradient local-Fock quadrature and explicit nonorthogonal product/geminal Gram-space Hylleraas solve",
    cases=cases,
)
Path(__file__).resolve().parents[1].joinpath("data/mp2_f12_reference.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8"
)
