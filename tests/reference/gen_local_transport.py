"""Independent polar-frame response from a square-root Sylvester equation."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_sylvester, sqrtm

rng = np.random.default_rng(51112)
q, _ = np.linalg.qr(rng.normal(size=(4, 2)))
u, _ = np.linalg.qr(rng.normal(size=(3, 3)))
cases = []
for m in [rng.normal(size=(5, 3)), q, u @ np.diag([1.0, 1.0, 0.4])]:
    root = sqrtm(m.T @ m)
    inverse = np.linalg.inv(root)
    response = []
    for coordinate in range(m.size):
        dm = np.zeros_like(m)
        dm.flat[coordinate] = 1
        dg = dm.T @ m + m.T @ dm
        ds = solve_sylvester(root, root, dg)
        response.append((dm @ inverse - m @ inverse @ ds @ inverse).tolist())
    cases.append(
        dict(matrix=m.tolist(), frame=(m @ inverse).tolist(), response=response)
    )
out = dict(
    source="NumPy/SciPy principal matrix square root; independent Sylvester solve for every input coordinate, including repeated metric eigenvalues",
    cases=cases,
)
Path(__file__).resolve().parents[1].joinpath(
    "data/local_transport_reference.json"
).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print("Jacobian components", sum(np.array(c["matrix"]).size ** 2 for c in cases))
