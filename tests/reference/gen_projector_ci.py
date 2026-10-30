"""PCI oracle from PySCF's full matrix; no Rust propagator/selection is called.

Wall generator: Zhang–Evangelista 2016 (arxiv:1606.08475), Eqs 25,45–48.
Other polynomial choices are filtered recurrence extensions. Historical Krylov
choices select determinants by spawning then use the unfiltered Hermitian model.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
import scipy.special
from gen_selected_ci import dets, g, h, ham, n, neighbors, reference
from pyscf import fci

estimate = ham[reference, reference]
upper = (
    0.321 + 4 * np.max(np.sum(abs(h), axis=1)) + 6 * np.max(np.sum(abs(g), axis=(1, 3)))
)
radius = (upper - estimate) / 2
center = estimate + radius
eye = np.eye(len(dets))
order = 4
tol = 1e-9


def norm(c):
    length = np.linalg.norm(c)
    if not np.isfinite(length) or length == 0:
        raise RuntimeError("invalid PCI vector")
    return c / length


def action(c, shift, eta):
    amplitudes = ham * c[None, :]
    amplitudes[abs(amplitudes) <= eta] = 0.0
    amplitudes[np.diag_indices(len(dets))] = (np.diag(ham) - shift) * c
    return amplitudes.sum(axis=1)


def homogeneous(c, shift, eta):
    length = np.linalg.norm(c)
    return length * action(c / length, shift, eta) if length else c.copy()


def run(kind, eta):
    state = eye[reference].copy()
    old_energy = estimate
    for iteration in range(5000):
        if kind == "WallChebyshev":
            new = state.copy()
            for k in range(1, order + 1):
                shift = estimate + radius * (1 - np.cos(np.pi * k / (order + 0.5)))
                new = norm(action(new, shift, eta))
        elif kind in ["Linear", "Power"]:
            new = action(state, upper, eta)
        elif kind in ["Chebyshev", "ExpChebyshev"]:
            weights = np.zeros(order + 1)
            if kind == "Chebyshev":
                weights[-1] = 1
            else:
                weights = scipy.special.iv(np.arange(order + 1), 0.5) * np.array(
                    [1] + [2] * order
                )
            previous = state.copy()
            current = -homogeneous(state, center, eta) / radius
            new = weights[0] * previous + weights[1] * current
            for k in range(2, order + 1):
                following = -2 * homogeneous(current, center, eta) / radius - previous
                new += weights[k] * following
                previous, current = current, following
        elif kind in ["Quadratic", "Cubic", "Quartic"]:
            new = state.copy()
            term = state.copy()
            for k in range(1, {"Quadratic": 2, "Cubic": 3, "Quartic": 4}[kind] + 1):
                term = -0.5 * homogeneous(term, estimate, eta) / (radius * k)
                new += term
        else:
            selected = sorted(
                set(np.flatnonzero(action(state, 0, eta))) | set(np.flatnonzero(state))
            )
            sub = ham[np.ix_(selected, selected)]
            if kind == "Davidson":
                _, vectors = np.linalg.eigh(sub)
                coeff = vectors[:, 0]
            else:
                v = state[selected]
                basis = []
                actions = []
                for k in range(min(order, len(selected))):
                    hv = sub @ v
                    basis.append(v.copy())
                    actions.append(hv.copy())
                    w = hv.copy()
                    for _ in range(2):
                        for b in basis:
                            w -= np.dot(b, w) * b
                    length = np.linalg.norm(w)
                    if length < 1e-13:
                        break
                    v = w / length
                basis = np.array(basis).T
                _, vectors = np.linalg.eigh(basis.T @ np.array(actions).T)
                coeff = basis @ vectors[:, 0]
            new = np.zeros(len(dets))
            new[selected] = coeff
        new = norm(new)
        if np.dot(new, state) < 0:
            new = -new
        sigma = ham @ new
        energy = float(np.dot(new, sigma))
        residual = float(np.linalg.norm(sigma - energy * new))
        if (
            abs(energy - old_energy) < 1e-12
            and np.linalg.norm(new - state) < tol
            and (eta > 0 or residual < tol)
        ):
            anchor = np.argmax(abs(new))
            projective = float(sigma[anchor] / new[anchor])
            selected = np.flatnonzero(new)
            external = [i for i in neighbors(selected) if i not in selected]
            correction = sum(
                float(sigma[i] ** 2 / (energy - ham[i, i]))
                for i in external
                if sigma[i]
            )
            one, two = fci.direct_spin1.make_rdm12(new, n, (2, 2))
            return {
                "generator": kind,
                "eta": eta,
                "order": order,
                "iterations": iteration + 1,
                "energy_variational": energy,
                "projective": projective,
                "correction": correction,
                "residual": residual,
                "n_determinants": len(selected),
                "rediagonalized_energy": float(
                    np.linalg.eigvalsh(ham[np.ix_(selected, selected)])[0]
                ),
                "one_rdm": one.ravel().tolist(),
                "two_rdm": two.ravel().tolist(),
            }
        state = new
        old_energy = energy
    raise RuntimeError((kind, eta, "PCI did not converge"))


cases = []
for eta in [0.01, 0.001]:
    cases.append(run("WallChebyshev", eta))
for kind in [
    "WallChebyshev",
    "Linear",
    "Power",
    "Chebyshev",
    "ExpChebyshev",
    "Quadratic",
    "Cubic",
    "Quartic",
    "Davidson",
    "Lanczos",
]:
    cases.append(run(kind, 0.0))
out = {
    "source": f"PySCF {pyscf.__version__} full Hamiltonian; independent dense NumPy propagation, SciPy modified Bessel coefficients; PCI arxiv:1606.08475 Eqs 25/45-48",
    "energy_estimate": float(estimate),
    "upper_bound": float(upper),
    "cases": cases,
}
Path(__file__).resolve().parents[1].joinpath(
    "data/projector_ci_reference.json"
).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(
    [
        (c["generator"], c["eta"], c["iterations"], c["energy_variational"])
        for c in cases
    ]
)
