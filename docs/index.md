# Documentation

Start with the common calculation interface, then the method-specific limits.

| Topic | Document |
| --- | --- |
| Owned molecules, typed contexts, units and errors | [Rust API](rust_api.md) |
| Native `run`, cached `Molecule`, observables and wavefunction exports | [Python API](python_api.md) |
| Forces, Hessians, observables, eV/angstrom units and optimization | [ASE](ase.md) |
| Rust/Python command line and embedded oracle | [CLI](cli.md) |
| All method families, recipes, HF observables, options and analytic response | [Catalog](methods.md) |
| Mathematical conventions and derivative rules | [Theory](theory.md) |
| Fixed criteria, commands and measured results | [Validation](validation.md) |
| Scientific primary references | [References](references.md) |
| Breaking changes | [Changelog](../CHANGELOG.md) |

Detailed implementations: [analytic HF/ECP derivatives](analytic_hf_derivatives.md),
[GHF](ghf.md), [complex HF](complex_hf.md), [scalar relativity](scalar_relativistic.md),
[local correlation](local_correlation.md), [F12](f12_integrals.md),
[CASPT2](caspt2.md), [DMRG](dmrg.md), [selected CI](selected_ci.md),
[SR-DSRG](sr_dsrg.md), [IM-SRG](imsrg2.md), [MR driven equations](mr_driven.md),
[multistate DSRG](multistate_dsrg.md), [THREE](three_dsrg.md), and
[density cumulants](density_cumulants.md).

Detailed measurement harnesses and work plans are stored outside the repository.
Committed tests and independent reference fixtures remain self-contained.

## Choose a task

1. Define geometry, units, element-specific orbital bases and independent ECPs
   in the [native input reference](python_api.md#complete-common-input-signatures).
2. Choose an [HF model or post-HF family](methods.md), check its reference and
   response limits, and inspect its accepted input types and defaults.
3. Request energy, physical observables, analytic derivatives or retained
   [solver tensors](python_api.md#integrals-and-retained-intermediates).
4. Export the configured reference as JSON/Molden, or use
   [ASE quantities](ase.md#integral-and-intermediate-inspection) in eV/angstrom units.
5. Check the [measured validation scope](validation.md), particularly angular
   coverage and the experimental F12 status.

The task/input/parameter/output/example organization is informed by the
[xTB user documentation](https://xtb-docs.readthedocs.io/en/latest/md.html) and
[xTB Python API documentation](https://xtb-python.readthedocs.io/en/latest/general-api.html).
The examples and parameter definitions here describe hf-rs itself.
