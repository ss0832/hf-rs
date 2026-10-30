# Command line

Both the Rust binary and the pip-installed command are named `hf-rs`.
Both pass the same inputs/options to the native calculation boundary.

`--property intermediates` requests bounded retained solver tensors, in the same
atomic-unit format as the native API. For example:

```sh
hf-rs --method ccsd --property intermediates
```

The default H2/STO-3G calculation returns `method.t1` with shape `[1,1]`
and `method.t2` with shape `[1,1,1,1]` inside `intermediates`. Energy-only
calls leave this map empty. The CLI uses the default aggregate bound of one
million numerical values; use the [native inspection API](python_api.md#integrals-and-retained-intermediates)
to select integral blocks, representations or another explicit output bound.
`--list-methods` includes each registry method's `option_details`: input types,
native units, defaults and descriptions. The separate complete HF-model
catalog is available as `hf_rs.hf_variant_catalog()`.

Install the Python command from the extracted source directory with Python
3.8 or newer and a recent stable Rust toolchain:

```sh
python -m pip install .
hf-rs --version
hf-rs --help
```

Activate the target virtual environment before installing. The distribution
name is `hf-rs-python`; no separate CLI package is required. A locally built
wheel can also be installed with `python -m pip install path/to/package.whl`.
The native command is installed separately with `cargo install --path .`.

```sh
hf-rs --method mp2 --geometry 'H 0 0 -0.7; H 0 0 0.7'
hf-rs --method ccsd --xyz water.xyz --basis sto-3g --opt residual_tol=1e-11
hf-rs --method hf --xyz water.xyz --property gradient
```

`--geometry` contains semicolon-separated `SYMBOL X Y Z` rows in bohr by default.
`--xyz` reads standard XYZ (count, comment, atom rows), defaulting to angstrom.
`--unit bohr|angstrom` overrides either default. Rust performs conversion.

Common flags are `--method`, `--basis`, `--charge`, `--multiplicity`,
`--reference auto|rhf|uhf`, `--spherical` and repeatable `--property` selections
(`energy`, `gradient`, `hessian`, `observables`). Both accept explicit `--cartesian`.
The default is Cartesian STO-3G RHF and energy. With no geometry, H2 at 1.4 bohr
is the small demonstration system.

`--opt KEY=VALUE` is repeatable. Values are JSON where valid (numbers, Booleans,
arrays); other values are strings. Duplicate keys, unknown options and unknown
methods are errors. Examples: `--opt frozen_core=1`, `--opt aux_basis=def2-svp-ri`,
`--opt 'active_orbitals=[1,2]'`. SCF convergence uses `scf_` keys.

`--basis` and `--ecp` also accept JSON element maps in both CLIs, for example
`--basis '{"H":"sto-3g","Na":"6-31g"}' --ecp '{"Na":"lanl2dz"}'`.
Names and filenames can be selected independently. Omitting `--ecp` uses the
ECP associated with each element's basis; `--ecp none` disables that default.
See [the full precedence rules](python_api.md#basis-and-ecp-selection).

Results are a single JSON object on stdout in atomic units. Progress goes to
stderr. Invalid/unsupported calculations exit nonzero. `--help` and `--version`
report the interface and version.

## Installed independent oracle

```sh
hf-rs --oracle-compare
hf-rs --list-methods
```

The native oracle embeds `src/validation/oracle.json` with `include_str!`, so it
works after installation and outside a checkout. Generate it with
`python tests/reference/gen_installed_oracle.py` from committed independent
SCF/SCS-MP3/CCSD(T) fixtures. Four small checks cover H2 RHF and water MP2,
SCS-MP3 and CCSD(T); tolerances remain1e-9/1e-8Eh. A failure exits1. This is an
installation smoke test, separate from the full scientific regression suite.
Both command distributions expose these flags through the same native code.

The previous CLI's individual method switches and self-test workflow are replaced
by `--method`, registry `--opt`, the embedded oracle and `cargo test --release`.
