# Python test maintenance

Python tests and independent reference generators are checked with Ruff. Install
the pinned linter in a development virtual environment:

```sh
python -m pip install -r tests/requirements-lint.txt
python -m ruff check tests
python -m ruff format --check tests
```

To apply formatting, run `python -m ruff format tests`. The configuration in
`pyproject.toml` checks syntax, undefined names, unused imports/variables, compound
statements and import ordering, with Python 3.8 as the minimum syntax target.
Keep copyright notices, numerical formulas and validation tolerances intact.

Reference generators need their existing scientific dependencies, such as PySCF,
NumPy, SciPy or mpmath; linting does not import them or run calculations. Generate
references in a separate copy when checking formatting changes, so the committed
oracle data is not overwritten. Rust/Python numerical validation uses native
Linux builds in WSL for this checkout.
