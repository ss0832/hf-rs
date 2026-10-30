# hf-rs

hf-rs is a Rust library for Hartree-Fock methods, post-Hartree-Fock electron
correlation methods and Hartree-Fock variants, with native Python bindings
and an ASE calculator.

It includes 80 orbital basis families, per-element basis selection and
analytic RHF/UHF nuclear derivatives through fourth order.
The supported methods, reference types and response limits are listed in the
[method catalog](docs/methods.md).

The crate forbids unsafe Rust in its own code and uses Rayon and faer for numerical work.
Independent scientific reference programs are used for validation, not at runtime.
Published/default F12 agreement remains an explicit open validation item; see
[F12 status](docs/f12_integrals.md) before using those model options.

## Install

Rust 2024 edition and a recent stable toolchain are required.

```sh
cargo build --release
cargo install --path .
```

For Python 3.8 or newer, install from the extracted source directory. Building
the native extension requires the Rust toolchain above.

```sh
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install .
hf-rs --version
```

The distribution is `hf-rs-python`; its import name is `hf_rs`.
This source installation also installs the `hf-rs` command. For ASE, use
`python -m pip install '.[ase]'`. For editable development, install `maturin`
and run `maturin develop --release` from the repository root.

## Command line

```sh
hf-rs --method mp2 --geometry 'H 0 0 -0.7; H 0 0 0.7' --basis sto-3g
hf-rs --oracle-compare
```

Geometry defaults to bohr; XYZ input defaults to angstrom. `--unit` overrides
the input unit. Results are JSON in atomic units; progress goes to stderr.
The Python entry point `hf-rs` uses the same native calculation boundary.

## Python

```python
import hf_rs

positions = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
result = hf_rs.run("mp2", [1, 1], positions, basis="sto-3g", unit="bohr")
print(result["e_total"])
molecule = hf_rs.Molecule([1, 1], positions, basis="sto-3g")
print(molecule.gradient())
```

`Molecule` reuses SCF for unchanged geometry, reference and SCF controls.
Options are validated against the selected method; unknown names are errors.

## ASE

```python
from ase import Atoms
from ase.optimize import BFGS
from hf_rs import HFCalculator

atoms = Atoms("H2", positions=[[0, 0, -0.4], [0, 0, 0.4]])
atoms.calc = HFCalculator(method="hf", basis="sto-3g", scf_energy_tol=1e-13)
BFGS(atoms).run(fmax=0.01)
```

ASE uses eV/angstrom-based physical inputs and results, including HF observables
and available analytic Hessians. Rust/native Python use atomic units.
See the [method recipes and observables](docs/methods.md#hf-observables) and
[Molden/wavefunction exports](docs/python_api.md#file-and-wavefunction-output).

## Documentation

- [Documentation index](docs/index.md)
- [Rust API](docs/rust_api.md), [Python API](docs/python_api.md), [ASE](docs/ase.md)
- [CLI](docs/cli.md), [methods and limits](docs/methods.md)
- [Theory](docs/theory.md), [validation](docs/validation.md), [references](docs/references.md)
- [Changelog](CHANGELOG.md)

## License

Copyright (c) 2026 ss0832. Licensed under [Apache-2.0](LICENSE).
Bundled BSE numerical data retain their [third-party notice](third_party/basis_set_exchange/README.md).
