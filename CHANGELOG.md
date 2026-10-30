# Changelog

## 0.3.2

- Describe hf-rs as a Hartree-Fock, post-HF and HF-variant library in the README.
- Add bounded AO/physical-AO/spin-resolved-MO/custom integral blocks and analytic
  bare integral derivatives through fourth order. Expose RI metrics, three-center
  values and whitened factors with explicit axes and unit metadata.
- Add opt-in retained HF-model matrices, CCD/CCSD amplitudes and selected output
  RDM inspection in Rust/native Python/ASE. Energy-only results avoid these copies.
- Export code-derived input types, actual defaults and HF-model restrictions.
  Expand existing docs with parameter tables and executed physical/API examples
  for all implemented common HF models and connected post-HF methods.
- Verify all Cartesian s/p/d/f pair and quartet integral components independently,
  and complete C1 d-shell nuclear tensors through order four in Cartesian/spherical
  spaces, with all-electron and scalar nonlocal-ECP RHF/UHF models.
- Verify every nuclear tensor entry through order four in three additional
  mixed s/p/d/f C1 models, including nonlocal-f-channel ECP RHF/UHF, with
  independent public-interface energy/gradient anchors.
- Exercise every connected common post-HF method through Rust/native Python/ASE,
  and 30 HF property models through native Python/ASE. Fix pip CLI acceptance
  of retained intermediate output and retain strict atomic/ASE unit semantics.
- Keep numerical execution summaries free of workstation identities and paths.
  Experimental F12 agreement and unavailable post-HF response remain explicit.

## 0.3.1

- Named the Cargo binary `hf-rs`, matching the pip-installed command. Both
  commands expose method catalogs and the embedded independent oracle.
- Added explicit HF observable maps with physical units for the implemented
  molecular, excitation, ensemble, periodic and radial models. Common adapters
  now connect TDHF, QRHF/MOM/Fermi, SA-HF, sigma-SCF and crystal/radial HF.
- Added cached native SCF wavefunction JSON and Molden exports without copying
  four-index ERIs. Exported SCF references remain distinct from correlated states.
- Unified Rust/native Python observable units to atomic units. ASE converts
  result/state/component energies, observables, analytic Hessians and explicitly
  supplied physical controls to eV/angstrom-based units.
- Expanded existing Rust, native Python and ASE documentation with HF-variant,
  post-HF-family, option and wavefunction-output examples.
- Added strict noncoplanar C1, s/p-AO all-component derivative regressions through
  fourth order, including nonlocal ECP RHF and UHF cases. Measured pass/fail
  results and the exact tested scope are recorded in [validation](docs/validation.md).

The unresolved published/default F12 printed-digit comparison and the deferred
post-HF response work retain their documented status.

## 0.3.0

### Breaking API changes

- Added the common native `run` and owned, SCF-caching `Molecule` interfaces.
  Python, ASE and both CLIs share native method/option validation and output.
- Geometry conversion occurs in Rust. Input units are explicit; result energies
  and derivatives use atomic units. ASE converts returned energy/forces only.
- Individual post-HF Python entry points are replaced by `run(method, ...)`.
  State lists, selected-root energies, components and available RDMs are preserved.
  HF tensor/directional compatibility interfaces remain available.
- CLI method switches are replaced by `--method` and repeatable `--opt KEY=VALUE`.
  Rust embeds an independent installed oracle; structured stdout is JSON and
  calculation progress goes to stderr.
- Installing the Python source package with pip also installs the `hf-rs`
  console command. Maturin remains available for editable development.
- `HFVariantType::parse` returns an error for unknown names. FMO3 selects the
  existing third-body model instead of falling back to FMO2. QED default energy
  and gradient use the same coupling vector.
- Cargo is the single version source. Python metadata uses a dynamic version.

### Scientific and numerical changes

- Expanded built-in orbital bases to 80 BSE families, with pinned numerical
  exports and third-party notices. Common Rust/Python/CLI/ASE inputs support
  per-element bases and independent named/file ECPs, with associated ECP defaults.
- Replaced an incorrect qualitative CpLi dipole bound with all-component PySCF
  references, and corrected ferrocene's Cartesian STO-3G AO-count expectation.

- Reworked and independently checked the CI/CASSCF, local-correlation, F12,
  IC-CASPT2, MRCISD(+Q), driven/flow, DMRG and selected-CI energy cores and the
  documented HF-variant models. Their precise limits and references are in
  [the method catalog](docs/methods.md) and [validation](docs/validation.md).
- Preserved analytic HF nuclear response through fourth order, including local
  and nonlocal ECPs. The common API reuses converged references for MP2 Hessians.
  Broader analytic post-HF response remains follow-up work; numerical debug
  routes are not advertised as analytic production properties.
- Added robust F12 lower-center fitting, physical auxiliary spaces, streamed
  Fock/occupied-leg contractions and explicit exact orthogonal OBS Coulomb controls.
  Compensated symmetric transforms stabilize cancellation in ghost CABS spaces.
- Added independent all-component molecular, ECP, amplitude and property checks.
  The F12 published/default printed-digit comparison remains unresolved; current
  F12 model options are experimental, with the discrepancy explicitly recorded.
- Removed duplicated Python dispatch and unit-conversion implementations and
  reorganized documentation around the common interfaces and actual capabilities.
- Audited every documentation page and README against current APIs and recorded
  evidence, correcting obsolete defaults, response claims and bibliography errors.
  Historical measurements retain their original settings and scope.

Detailed benchmark harnesses and work plans remain outside the source archive.
The license remains Apache-2.0; copyright notices are retained.
