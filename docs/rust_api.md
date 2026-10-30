# Rust API

## Common calculations

The [calculation recipes](methods.md#calculation-recipes) give method names and
starting inputs for every connected post-HF family and molecular HF variant.
The [registered option index](methods.md#registered-method-options) is the
complete registry schema. This page also shows how to obtain typed solver
results and use specialized models outside the common adapter.

`api::Molecule` owns geometry, basis functions and an SCF cache. Input coordinates
use `LengthUnit::Bohr` or `LengthUnit::Angstrom`; conversion occurs only in Rust.
All returned energies and nuclear derivatives use atomic units.

```rust
use hf_rs::api::{Molecule, MoleculeOptions, OptionMap};

let molecule = Molecule::new(
    &[1, 1], &[[0., 0., -0.7], [0., 0., 0.7]], MoleculeOptions::default(),
)?;
let result = molecule.run("mp2", &OptionMap::new(), &[])?;
println!("{}", result.output.e_total);
# Ok::<(), hf_rs::HfError>(())
```

`api::run(method, numbers, positions, configuration, options, properties)` is the
stateless convenience entry point. `CalculationResult` contains `PostHFOutput`,
the reference, optional gradient/Hessian and basis/SCF dimensions. `to_json()` is
the common CLI/Python result representation.

`MoleculeOptions` controls basis, Cartesian/spherical space, charge, multiplicity,
input unit and `ReferenceChoice::{Auto,Rhf,Uhf}`. `Auto` uses RHF for a singlet
and UHF otherwise. The explicit UHF method selects UHF and rejects `reference=rhf`.
The `hf` and `rhf` method names follow the chosen reference; force RHF with
`ReferenceChoice::Rhf`. Correlation aliases with RHF/UHF prefixes do not select
the reference. A broken-symmetry singlet requires `ReferenceChoice::Uhf` and
`ScfOptions::broken_symmetry = true`. Atomic numbers, finite
coordinates, electron parity, multiplicity and element basis coverage are checked.
Named ECP bases attach their built-in local/nonlocal ECPs before counting electrons.

`MoleculeOptions::basis_by_element` overrides `basis` for selected element
symbols or atomic-number strings. `ecp` and `ecp_by_element` independently
select named/file potentials; omission uses the selected basis's associated
ECP. `set_basis(&serde_json::Value)` and `set_ecp(...)` also accept element maps
with an optional `*` fallback. `none` disables an explicitly selected ECP.
Combined Gaussian 94 and BSE JSON files retain their embedded potentials.
See [selection rules](python_api.md#basis-and-ecp-selection) and the
[80-family BSE catalog](../third_party/basis_set_exchange/README.md).

`Molecule::scf(ScfOptions)` returns an `Arc<ScfReference>`. Repeated identical
settings reuse that allocation. A changed reference or SCF setting replaces the
cached entry. Geometry/configuration is immutable; construct another molecule
after moving atoms. `PostHFContext` supplies the per-calculation lazy integral cache.

## Options and output

`OptionMap` is a `BTreeMap<String, serde_json::Value>`. SCF keys are
`scf_max_iter`, `scf_energy_tol`, `scf_residual_tol`, `scf_density_tol`,
`scf_conv_algo` and `broken_symmetry`. Defaults are 100 iterations and
energy/residual/density tolerances 1e-12/1e-10/1e-10. RHF defaults to ADIIS/SOSCF;
UHF uses CDIIS. An explicitly unsupported UHF accelerator is rejected.

Remaining options pass to `PostHFOptions::from_map` with the selected registry
schemas. `max_iter`, `energy_tol` and `residual_tol` apply to post-HF equations;
the `scf_` prefix distinguishes reference convergence. Unknown/inapplicable keys
raise `HfError::InvalidInput`. Scientific options are listed in [methods](methods.md).

`PostHFOutput` uses `e_corr = e_total - e_scf`, also for multireference methods.
It retains all computed states, selected `root`, method components and the
available MO 1-RDM. `e_total` is the selected state's energy, not always root zero.
MR components identify their CASCI/CASSCF reference. Errors and failed convergence
propagate; the common API does not substitute zero correlation or another state.

## Analytic properties

Properties are `energy`, HF-model `observables`, retained `intermediates`, `gradient` and `hessian`.
HF uses its checked analytic
gradient contraction and common high-order response context, including ECPs.
Existing analytic MP2 response is connected for
the full-rank all-electron canonical space, without frozen spaces or DF. The MP2
Hessian reuses the supplied reference. Other post-HF production response remains
unsupported here until its analytic implementation is verified. Explicit numerical
debug interfaces in low-level modules retain their names and status.

`nuclear_derivative(coordinates, settings)` evaluates an HF mixed coordinate
derivative of order 1 through 4; indices are `3*atom + axis` and may repeat.
`directional_derivative(directions, settings)` uses flat 3N directions in bohr.
The same backend handles RHF/UHF and local/nonlocal ECP response. See
[analytic HF derivatives](analytic_hf_derivatives.md) for costs and rank limits.

`hf_variant(name, options, properties)` selects a documented molecular default
model. It rejects unsupported spin, basis-space or option combinations rather
than discarding controls. QED exposes `cavity_frequency` and `coupling`; RI-HF
exposes `aux_basis` and `metric_tol`. Advanced model controls remain available
through their typed Rust modules. See the [adapter limits](methods.md).

## Typed post-HF API

For direct control, construct `PostHFContext::from_rhf` or `from_uhf`, a validated
`PostHFOptions`, and `registry::create_solver(MethodId, &context, &options)`.
`MethodId::from_str` is the only post-HF method-name parser; HF variant names
are parsed by the same registry. `ContextMethod` and `MethodMeta` connect typed
engines through `Entry<T>`. `PostHFResult::to_output()` provides common semantics.
Use the catalog and explicit errors to distinguish unconnected adapters from
the broader low-level scientific API.

```rust
use hf_rs::api::{Molecule, MoleculeOptions, ScfOptions, ScfReference};
use hf_rs::context::PostHFContext;
use hf_rs::options::{ActiveSpace, PostHFOptions};
use hf_rs::registry::{create_solver, MethodId};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let molecule = Molecule::new(
        &[3, 1], &[[0., 0., 0.], [0., 0., 3.]], MoleculeOptions::default(),
    )?;
    let reference = molecule.scf(ScfOptions::default())?;
    let ScfReference::Rhf(hf) = &*reference else { return Err("RHF required".into()); };
    let context = PostHFContext::from_rhf(hf, molecule.atoms(), molecule.basis(), None)?;
    let options = PostHFOptions {
        active: Some(ActiveSpace::new(1, 1, 2)),
        ..Default::default()
    };
    let output = create_solver(MethodId::Fci, &context, &options)?.compute()?;
    println!("{} {:?}", output.e_total, output.one_rdm);
    Ok(())
}
```

For an explicit auxiliary basis, create `context::AuxBasis::from_builtin`,
keep it alive and pass `Some(&auxiliary)` to the context. For UHF use
`PostHFContext::from_uhf`; retain its borrowed `Arc<ScfReference>` until the
context/solver is dropped. `create_solver` returns the common selected-state
output. Request bounded retained amplitudes with `Molecule::intermediates`;
for additional typed wavefunction fields, retain the engine's `compute()`
result. See [correlated output](#file-and-wavefunction-output).

## Specialized HF variants

Common molecular examples use the same option map as Python:

```rust
use hf_rs::api::{Molecule, MoleculeOptions, OptionMap};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let molecule = Molecule::new(
        &[1, 1], &[[0., 0., -0.7], [0., 0., 0.7]], MoleculeOptions::default(),
    )?;
    let options = OptionMap::from([
        ("cavity_frequency".into(), json!(0.5)),
        ("coupling".into(), json!([0., 0., 0.02])),
    ]);
    let result = molecule.hf_variant("qed-hf", &options, &["gradient".into()])?;
    println!("{} {:?}", result.output.e_total, result.gradient);
    Ok(())
}
```

Typed options expose the complete model controls rather than the restricted
common default-model adapter:

```rust
use hf_rs::api::{Molecule, MoleculeOptions};
use hf_rs::hf_variants::{run_qed_hf_with_options, QedHfOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let molecule = Molecule::new(
        &[1, 1], &[[0., 0., -0.7], [0., 0., 0.7]], MoleculeOptions::default(),
    )?;
    let options = QedHfOptions { coupling: [0., 0., 0.02], ..Default::default() };
    let result = run_qed_hf_with_options(molecule.atoms(), molecule.basis(), 0, &options)?;
    println!("{} {}", result.total_energy, result.mean_photon_number);
    Ok(())
}
```

The typed entry-point table below lists exact source-level argument names.
Construct atoms/bases in bohr with `Molecule` or checked basis builders; provide
the typed options object named by the function. All functions in the table
are re-exported from `hf_rs::hf_variants`.

| Typed model | Entry point and arguments | Explicit controls |
| --- | --- | --- |
| GHF | `run_ghf_with_options(atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, tilt: f64, options: &GhfOptions)` | Initial spin tilt/density and convergence |
| Spin projection | `run_puhf_with_multiplicity(atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, multiplicity: usize, max_iter: usize, tol: f64)` | Target multiplicity and exact projection result |
| HFB | `run_hfb_with_options(atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, options: &HfbOptions)` | Pair field, number constraint and pairing controls |
| TCSCF/GVB | `run_tcscf_from_reference(hf: &HFResult, bonding_orbital: usize, antibonding_orbital: usize, max_iter: usize, tol: f64)` | Occupied bonding / virtual antibonding MO indices |
| TDHF/RPA | `run_tdhf_from_reference(hf: &HFResult, basis: &[ContractedGaussian], n_singlets: usize, n_triplets: usize)` | Excited roots and transition properties |
| QED-HF | `run_qed_hf_with_options(atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, options: &QedHfOptions)` | Cavity frequency/coupling and photon observables |
| RI-HF | `run_df_hf_with_aux(atoms: &[Atom], basis: &[ContractedGaussian], aux: &[ContractedGaussian], charge: i32, options: &DFHFOptions)` | Explicit auxiliary functions and metric cutoff |
| CD-HF | `run_cd_hf_with_options(atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, cholesky_delta: f64, options: &DFHFOptions)` | Explicit Cholesky residual threshold |
| X2C/DKH2 | `run_relativistic_hf_with_options(atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, model: RelativisticModel, options: &RelativisticHfOptions)` | Model, speed of light and convergence |
| Complex HF | `run_complex_hf_with_options(atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, options: &ComplexHfOptions)` | Reference type, magnetic/spin fields and gauge origin |
| LHF | `run_lhf_with_options(atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, o: &LhfOptions)` | Numerical grid, local exchange and CEDA controls |
| OEP | `run_oep_hf(atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, options: &OepOptions)` | Potential basis and optimization controls |
| NEO-HF | `run_neo_hf_with_options(classical_atoms: &[Atom], electronic_basis: &[ContractedGaussian], protonic_basis: &[ContractedGaussian], charge: i32, n_quantum_protons: usize, options: &NeoHfOptions)` | Classical nuclei, electronic/protonic bases and quantum-proton count |
| Positronic HF | `run_positronic_hf_with_options(atoms: &[Atom], electronic_basis: &[ContractedGaussian], positronic_basis: &[ContractedGaussian], charge: i32, options: &PositronicHfOptions)` | Separate electronic/positronic bases and particle controls |
| DC-HF | `run_dc_hf(atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, options: &DCOptions)` | Explicit central/buffer subsystems |
| FMO2/FMO3 | `run_fmo(atoms: &[Atom], basis: &[ContractedGaussian], fragments: &[FMOFragment], o: &FMOOptions)` | Fragment atoms/charges/multiplicity and FMOOptions::order |
| Crystal HF | `run_crystal_orbital_hf_with_options(unit_cell_atoms: &[Atom], basis: &[ContractedGaussian], charge: i32, lattice_constant_bohr: f64, options: &CoHfOptions)` | Unit cell, lattice constant, k points and finite-range approximation |
| Radial atomic HF | `run_numerical_atomic_hf(atomic_number: usize, n_grid: usize, max_iter: usize, tol: f64)` | Neutral H/He radial grid; explicit 1s ion route for other Z |
| State-averaged HF | `run_state_averaged_hf_with_options(atoms: &[Atom], basis: &[ContractedGaussian], configurations: &[StateConfiguration], weights: &[f64], options: &SaHfOptions)` | Explicit determinants and ensemble weights |
| Sigma-SCF | `run_sigma_scf_with_options(atoms: &[Atom], basis: &[ContractedGaussian], target_energy: f64, configuration: &StateConfiguration, options: &SigmaScfOptions)` | Target energy, determinant and variance controls |

## Physical observables

Common calls return an explicitly requested `ObservableMap`:

```rust
use hf_rs::api::{Molecule, MoleculeOptions, OptionMap, ScfOptions, ScfReference};
use hf_rs::output::HfObservables;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mol = Molecule::new(&[1, 1], &[[0.,0.,-0.7],[0.,0.,0.7]], MoleculeOptions::default())?;
    let values = mol.hf_observables("hf", &OptionMap::new())?;
    println!("{:?} {}", values["dipole"].value, values["dipole"].unit);
    let reference = mol.scf(ScfOptions::default())?;
    let ScfReference::Rhf(hf) = &*reference else { return Err("RHF required".into()); };
    let typed_values = hf.observables()?;
    assert_eq!(values["dipole"].value, typed_values["dipole"].value);
    Ok(())
}
```

The `output::HfObservables` trait is also implemented by the typed ROHF, GHF,
PUHF, HFB, TCSCF, QED-HF, DF/CD-HF, scalar-relativistic, complex, LHF/OEP,
NEO/positronic, DC/FMO, TDHF, crystal, radial, SA-HF, sigma-SCF and
`FermiSmearingResult` types. Use `result.observables()?` after an explicit typed
calculation. Values and their unit metadata use atomic units, including dipoles
in e*bohr and variances in Hartree squared; spin and amplitudes are dimensionless.
Unavailable frontier quantities are JSON null, and nonfinite numerical data
return an error. The [observable table](methods.md#hf-observables) specifies
each model's quantities and interpretation limits. Older convenience accessors
with explicit `_debye`, `_ev` or `_kelvin` names retain those labeled units;
the new uniform observable interface uses atomic units.

Additional legacy typed entry points at the crate root are
`run_qrhf(atoms,basis,charge,multiplicity,max_iter,tol)`,
`run_scf_mom(atoms,basis,charge,occupied_orbital_indices,max_iter,tol)` and
`run_scf_fermi_smearing(atoms,basis,charge,temperature_kelvin,max_iter,tol)`.
The common Fermi adapter instead takes `thermal_energy=kBT` in Hartree and
reports entropy divided by kB, avoiding a temperature-unit exception.

## Integrals and bases

`basis::{get_builtin_basis, load_basis_from_file, try_parse_gaussian94,
parse_bse_json, try_build_basis}` supplies checked basis construction. Gaussian94
and BSE JSON use the same Cartesian working functions. `ao::AoSpace` maps that
space to the physical Cartesian or real spherical subspace. MO coefficient rows
remain Cartesian and columns index the retained physical orbitals.

`ecp` supplies definitions, assignment and local/nonlocal operators. `integrals`,
`shell_integrals`, `derivatives` and `hf_derivatives` supply value and analytic
nuclear-response operations. Quadrature of analytically differentiated ECP
integrands is distinct from finite differences of molecular energies.

`MoIntegralCache` lazily shares full/OOVV/active, mixed-spin and DF transforms.
`df::DfTensor` is the common Coulomb-metric factorization. F12 lower-center
operators, robust fitting and occupied-leg transforms are documented in
[F12 integrals](f12_integrals.md). No full union-AO F12 fourth-power tensor is
required by the molecular fitted path. Workspace guards exclude SCF/reference
storage where stated; process RSS is not a universal allocation bound.

## Linear algebra

`matrix::Matrix` provides contiguous row-major matrices and checked dimensions
at scientific API boundaries; dense products/eigensolvers use the Rust numerical
backend. `solvers::{davidson_solver, DavidsonOptions, TensorDIIS}` provides
iterative eigenproblems and tensor extrapolation. `cholesky` provides screened
shell-column ERI factorization. Dense CI/response spaces retain documented
small-system limits; these APIs do not promise linear scaling for arbitrary inputs.

## File and wavefunction output

```rust
use hf_rs::api::{Molecule, MoleculeOptions, OptionMap, ScfOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let molecule = Molecule::new(
        &[1, 1], &[[0., 0., -0.7], [0., 0., 0.7]], MoleculeOptions::default(),
    )?;
    let result = molecule.run("mp2", &OptionMap::new(), &["energy".into()])?;
    std::fs::write("result.json", serde_json::to_string_pretty(&result.to_json())?)?;
    std::fs::write("reference.molden", molecule.scf_molden(ScfOptions::default())?)?;
    let wavefunction = molecule.scf_wavefunction(ScfOptions::default())?;
    std::fs::write("reference.json", serde_json::to_string_pretty(&wavefunction)?)?;
    Ok(())
}
```

These calls reuse the cached SCF when the reference and controls match. They
serialize orbital matrices, not the four-index ERI tensor. `basis_map()` returns
the actual per-element definitions if a typed writer needs them; loading one
global basis name again could lose mixed-element/file selections.

### Typed correlated wavefunction data

The typed Rust engine also gives direct access to amplitudes and additional
wavefunction fields; the bounded common alternative is `Molecule::intermediates`:

```rust
use hf_rs::api::{Molecule, MoleculeOptions, ScfOptions, ScfReference};
use hf_rs::post_hf::{CCSD, PostHFMethod};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let molecule = Molecule::new(
        &[1, 1], &[[0., 0., -0.7], [0., 0., 0.7]], MoleculeOptions::default(),
    )?;
    let reference = molecule.scf(ScfOptions::default())?;
    let ScfReference::Rhf(hf) = &*reference else { return Err("RHF required".into()); };
    let result = CCSD::new(hf).compute()?;
    if !result.converged { return Err("CCSD did not converge".into()); }
    let amplitudes = json!({
        "representation": "RHF spatial occupied/virtual MOs",
        "n_occ": result.n_occ, "n_virt": result.n_virt,
        "t1": result.t1_amplitudes, "t2": result.t2_amplitudes,
        "reference": molecule.scf_wavefunction(ScfOptions::default())?,
    });
    std::fs::write("ccsd-amplitudes.json", serde_json::to_string_pretty(&amplitudes)?)?;
    Ok(())
}
```

Here RHF `t1` is flattened `(i,a)` and `t2` is `(i,j,a,b)`, with the last
index fastest. UHF CCSD uses its documented spin-orbital dimensions and ordering;
do not reuse the RHF spatial label. `FCIResult` exposes `ground_state_vector`,
`mo_coefficients`, `one_rdm_mo`, `two_rdm_mo` and `n_determinants`; a raw CI vector
also needs its determinant ordering to be interpreted outside that solver.
Typed CASSCF/MRCI/DMRG/DSRG results have method-specific fields; preserve their
orbital frame, active-space indices and state ordering together with the data.
See the [Rust API](rust_api.md#typed-post-hf-api) and method-specific documents.

## Bounded integral and intermediate inspection

`api::IntegralRequest::default()` selects `operator="overlap"`,
`representation="cartesian_ao"`, complete axes, derivative order zero,
no explicit coefficient matrices/auxiliary basis, `metric_cutoff=1e-10`,
`max_elements=1_000_000` and `workspace_budget_bytes=128*1024*1024`.
Its fields correspond directly to the [complete integral input table](python_api.md#integrals-and-retained-intermediates).
Native units, axis ordering, fixed-orbital derivative semantics and unsupported
RI/Fock response combinations are identical in Rust and native Python.

```rust
use hf_rs::api::{IntegralRequest, Molecule, MoleculeOptions, OptionMap, ScfOptions};
let molecule = Molecule::new(&[1,1], &[[0.,0.,-0.7],[0.,0.,0.7]], MoleculeOptions::default())?;
let block = molecule.integrals(&IntegralRequest {
    operator: "eri".into(), representation: "mo_alpha".into(),
    indices: vec![vec![0],vec![1],vec![0],vec![1]],
    ..Default::default()
}, ScfOptions::default())?;
assert_eq!(block.tensor.shape, vec![1,1,1,1]);
let tensors = molecule.intermediates("ccsd", &OptionMap::new(), 1_000_000)?;
assert_eq!(tensors["method.t2"].shape, vec![1,1,1,1]);
# Ok::<(), hf_rs::HfError>(())
```

`output::TensorData` contains shape, named axes, flat row-major numerical data
and energy/length unit powers. `output::TensorMap` is an ordered name-to-tensor
map. `HfObservables::intermediates(limit)` and
`PostHFResult::intermediates(limit)` also operate on retained typed results.
The common solver adds the selected available 1-RDM after root selection.
`TensorExport` enforces a cumulative numerical-element budget before each copy.
An empty map means that the result type does not retain exportable tensors;
it does not fabricate a wavefunction from scalar energies.

`api::method_catalog()`, `option_catalog()` and `hf_variant_catalog()` provide
accepted controls, types, units, defaults and model restrictions for bindings
and input tooling. Metadata does not bypass each adapter's scientific guards.

## Typed HF option reference

These tables cover every public HF-model `*Options` struct in
`src/hf_variants`. Defaults are transcribed from each actual Rust `Default`
implementation. They are the typed entry-point defaults: common adapters
override their iteration/energy controls and may fix additional model inputs.
All physical values below use atomic units. Nested/default enum expressions
are Rust expressions, not accepted Python option strings. Invalid dimensions,
nonfinite values and unsupported spin/reference combinations return errors.
The linked module defines model-specific stationarity and validation rules.

### `ComplexHfOptions`

Source and input validation: [complex_engine.rs](../src/hf_variants/complex_engine.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `reference` | `ComplexHfReference` | `ComplexHfReference::Restricted` | ComplexHfReference: Restricted, Unrestricted or Generalized. |
| `multiplicity` | `usize` | `1` | Fixes N_alpha-N_beta for UHF; restricted HF requires one. |
| `magnetic_field` | `[f64; 3]` | `[0.; 3]` | Finite Cartesian magnetic field in atomic units. |
| `gauge_origin` | `Option<[f64; 3]>` | `None` | None selects the nuclear centroid, transported with the nuclei. |
| `spin_zeeman` | `bool` | `false` | Include B.S with the free-electron g factor equal to two. |
| `max_iterations` | `usize` | `200` | Positive solver iteration limit. |
| `energy_tol` | `f64` | `1e-12` | Positive energy-change threshold in Hartree. |
| `residual_tol` | `f64` | `1e-10` | Positive solver stationarity/residual threshold; definition follows the model equations. |
| `density_tol` | `f64` | `1e-10` | Positive density-change threshold in the solver working frame. |
| `overlap_cutoff` | `f64` | `1e-10` | Positive dimensionless AO-overlap eigenvalue cutoff; retained rank must remain compatible with the electron count. |
| `max_work_bytes` | `usize` | `256 * 1024 * 1024` | Conservative matrix/integral workspace bound in bytes. |
| `initial_density` | `Option<(Matrix, Matrix)>` | `None` | Hermitian metric-idempotent spinor density (real, imaginary), alpha first. |

### `CoHfOptions`

Source and input validation: [crystal.rs](../src/hf_variants/crystal.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `n_kpoints` | `usize` | `16` | Number of uniform k points (Gamma-centered); at least `2 L_ov + 1`. |
| `near_field_cells` | `Option<usize>` | `None` | Explicit near-field cells `D`; `None` chooses `max(2 L_ov + 3, ceil(20 bohr / a))`. |
| `multipole_order` | `usize` | `8` | Total Cartesian multipole order of the far-field expansion. |
| `overlap_threshold` | `f64` | `1e-12` | Overlap threshold that defines the AO pair range `L_ov`. |
| `max_exchange_cells` | `usize` | `12` | Largest real-space density/exchange range in cells (also limited by the k mesh). |
| `max_iterations` | `usize` | `200` | Positive solver iteration limit. |
| `energy_tol` | `f64` | `1e-10` | Positive energy-change threshold in Hartree. |
| `density_tol` | `f64` | `1e-9` | Largest element of the density residual `/P_out - P_in/`. |
| `diis_size` | `usize` | `8` | Maximum number of stored DIIS vectors. |
| `integral_budget_bytes` | `usize` | `2 * 1024 * 1024 * 1024` | Memory limit for the stored periodic electron-repulsion integrals. |

### `DCOptions`

Source and input validation: [dc_hf.rs](../src/hf_variants/dc_hf.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `central_regions` | `Option<Vec<Vec<usize>>>` | `None` | Central atom sets; they must cover every atom exactly once. `None` uses consecutive groups of `atoms_per_subsystem` atoms. |
| `atoms_per_subsystem` | `usize` | `1` | Positive number of consecutive atoms per default central subsystem; explicit partitions use the typed constructor. |
| `buffer_radius_bohr` | `f64` | `6.0` | Buffer radius in bohr (distance from any central atom). |
| `inverse_temperature` | `f64` | `200.0` | Fermi-function inverse temperature `beta` in 1/Hartree. |
| `max_iterations` | `usize` | `300` | Positive solver iteration limit. |
| `energy_threshold` | `f64` | `1e-10` | Energy change between iterations. |
| `density_threshold` | `f64` | `1e-8` | Largest density-residual element `/P_out - P_in/`. |
| `diis_size` | `usize` | `8` | Number of Pulay mixing vectors. |

### `FMOOptions`

Source and input validation: [fmo.rs](../src/hf_variants/fmo.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `order` | `usize` | `2` | Fragment expansion order 1, 2 or 3. |
| `electrostatic_embedding` | `bool` | `true` | Include self-consistent monomer electrostatic fields when true. |
| `max_iterations` | `usize` | `300` | Positive solver iteration limit. |
| `tolerance` | `f64` | `1e-11` | Positive SCF energy-change threshold in Hartree. |
| `residual_tolerance` | `f64` | `1e-9` | Positive solver stationarity/residual threshold; definition follows the model equations. |
| `density_tolerance` | `f64` | `1e-10` | Positive density-change threshold in the solver working frame. |
| `max_embedding_iterations` | `usize` | `100` | Positive monomer-polarization iteration limit. |
| `embedding_density_tolerance` | `f64` | `1e-9` | Positive maximum change in the fragment density field. |
| `workspace_limit_bytes` | `usize` | `256 * 1024 * 1024` | Conservative integral/matrix estimate, not a process-wide RSS cap. |
| `esp_cache_limit_bytes` | `usize` | `64 * 1024 * 1024` | Zero forces streamed monomer Coulomb contractions. |

### `GhfOptions`

Source and input validation: [ghf.rs](../src/hf_variants/ghf.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `max_iterations` | `usize` | `200` | Positive solver iteration limit. |
| `energy_tol` | `f64` | `1e-12` | Positive energy-change threshold in Hartree. |
| `residual_tol` | `f64` | `1e-10` | Positive solver stationarity/residual threshold; definition follows the model equations. |
| `density_tol` | `f64` | `1e-10` | Positive density-change threshold in the solver working frame. |
| `overlap_cutoff` | `f64` | `1e-10` | Positive dimensionless AO-overlap eigenvalue cutoff; retained rank must remain compatible with the electron count. |
| `max_work_bytes` | `usize` | `256 * 1024 * 1024` | Conservative matrix/integral workspace bound in bytes. |
| `initial_density` | `Option<Matrix>` | `None` | Optional finite initial density in the documented model frame. |

### `HfbOptions`

Source and input validation: [hfb.rs](../src/hf_variants/hfb.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `pairing_strength` | `f64` | `0.05` | Nonnegative additional reduced-pairing interaction strength in Hartree. |
| `max_iterations` | `usize` | `200` | Positive solver iteration limit. |
| `energy_tol` | `f64` | `1e-11` | Positive energy-change threshold in Hartree. |
| `residual_tol` | `f64` | `1e-9` | Positive solver stationarity/residual threshold; definition follows the model equations. |
| `density_tol` | `f64` | `1e-9` | Positive density-change threshold in the solver working frame. |
| `number_tol` | `f64` | `1e-12` | Positive tolerance on the HFB expected particle count. |

### `LhfOptions`

Source and input validation: [lhf_oep.rs](../src/hf_variants/lhf_oep.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `max_iterations` | `usize` | `150` | Positive solver iteration limit. |
| `reference_max_iterations` | `usize` | `300` | Positive iteration limit for the reference HF calculation. |
| `energy_tolerance` | `f64` | `1e-11` | Positive energy-change threshold in Hartree. |
| `residual_tolerance` | `f64` | `1e-8` | Positive solver stationarity/residual threshold; definition follows the model equations. |
| `density_tolerance` | `f64` | `1e-9` | Positive density-change threshold in the solver working frame. |
| `linear_tolerance` | `f64` | `1e-6` | Tolerance for the occupied LHF correction equations. |
| `minimum_linear_eigenvalue` | `f64` | `1e-10` | Positive cutoff for the LHF correction-system eigenvalues. |
| `grid` | `LocalGridOptions` | `LocalGridOptions::default()` | Nested LocalGridOptions; see the table below. |
| `max_grid_refinements` | `usize` | `2` | Maximum additional quadrature refinements, from 0 through 8. |

### `LocalGridOptions`

Source and input validation: [local_grid.rs](../src/hf_variants/local_grid.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `radial_points` | `usize` | `64` | Positive radial quadrature point count per center. |
| `polar_points` | `usize` | `20` | Positive polar quadrature point count. |
| `azimuthal_points` | `usize` | `40` | Positive azimuthal quadrature point count. |
| `radial_scale_bohr` | `f64` | `1.` | Positive radial map scale in bohr. |
| `overlap_tolerance` | `f64` | `1e-6` | Positive maximum quadrature error in the dimensionless AO overlap. |
| `workspace_limit_bytes` | `usize` | `256 * 1024 * 1024` | Conservative workspace limit in bytes; not a process-wide RSS cap. |
| `kernel_cache_limit_bytes` | `usize` | `256 * 1024 * 1024` | Maximum Coulomb-kernel cache size in bytes; a bounded streamed route is available. |

### `NeoHfOptions`

Source and input validation: [neo.rs](../src/hf_variants/neo.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `proton_mass` | `f64` | `PROTON_MASS_AU` | Quantum-particle mass in electron masses. |
| `max_iterations` | `usize` | `200` | Positive solver iteration limit. |
| `energy_tol` | `f64` | `1e-10` | Change of the total energy between iterations. |
| `residual_tol` | `f64` | `1e-8` | Frobenius norm of each orthonormal-basis commutator `X^T(FPS - SPF)X`. |
| `density_tol` | `f64` | `1e-10` | Frobenius norm of the density change divided by the basis dimension. |

### `OepOptions`

Source and input validation: [oep.rs](../src/hf_variants/oep.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `exponents` | `Vec<f64>` | `vec![0.5, 2.]` | One atom-centered s-type potential function per atom and exponent (Bohr^-2). |
| `max_iterations` | `usize` | `1000` | Positive solver iteration limit. |
| `reference_max_iterations` | `usize` | `300` | Positive iteration limit for the reference HF calculation. |
| `energy_tol` | `f64` | `1e-12` | Positive energy-change threshold in Hartree. |
| `gradient_tol` | `f64` | `1e-8` | Positive orbital-gradient convergence tolerance. |
| `minimum_gap` | `f64` | `1e-8` | Smallest supported occupied/virtual gap in Hartree; no replacement denominator is used. |
| `max_parameter_step` | `f64` | `0.25` | Positive maximum norm of an OEP potential-parameter step. |
| `potential_budget_bytes` | `usize` | `256 * 1024 * 1024` | Additional AO potential matrices only; excludes the borrowed RHF reference. |

### `PositronicHfOptions`

Source and input validation: [positronic_hf.rs](../src/hf_variants/positronic_hf.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `max_iterations` | `usize` | `neo.max_iterations` | Positive solver iteration limit. |
| `energy_tol` | `f64` | `neo.energy_tol` | Positive energy-change threshold in Hartree. |
| `residual_tol` | `f64` | `neo.residual_tol` | Positive solver stationarity/residual threshold; definition follows the model equations. |
| `density_tol` | `f64` | `neo.density_tol` | Positive density-change threshold in the solver working frame. |

### `QedHfOptions`

Source and input validation: [qed_hf.rs](../src/hf_variants/qed_hf.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `cavity_frequency` | `f64` | `0.1` | Photon energy, strictly positive, in Hartree. |
| `coupling` | `[f64; 3]` | `[0., 0., 0.05]` | Length-gauge coupling in sqrt(Hartree)/Bohr. |
| `max_iterations` | `usize` | `200` | Positive solver iteration limit. |
| `energy_tol` | `f64` | `1e-12` | Positive energy-change threshold in Hartree. |
| `residual_tol` | `f64` | `1e-10` | Positive solver stationarity/residual threshold; definition follows the model equations. |
| `density_tol` | `f64` | `1e-10` | Positive density-change threshold in the solver working frame. |

### `NumericalAtomicOptions`

Source and input validation: [radial_1s.rs](../src/hf_variants/radial_1s.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `n_grid_points` | `usize` | `4096` | Radial grid size, at least 16 points. |
| `max_iterations` | `usize` | `100` | Positive solver iteration limit. |
| `energy_tol` | `f64` | `1e-12` | Positive energy-change threshold in Hartree. |
| `residual_tol` | `f64` | `1e-8` | Positive solver stationarity/residual threshold; definition follows the model equations. |
| `density_tol` | `f64` | `1e-10` | Positive density-change threshold in the solver working frame. |
| `r_max_bohr` | `Option<f64>` | `None` | Radial boundary in Bohr. None chooses 20/(Z-(N-1)/2). |
| `memory_limit_bytes` | `usize` | `256 * 1024 * 1024` | Conservative workspace budget, including the returned radial arrays. |

### `RelativisticHfOptions`

Source and input validation: [relativistic_engine.rs](../src/hf_variants/relativistic_engine.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `max_iterations` | `usize` | `200` | Positive solver iteration limit. |
| `convergence` | `RhfConvergence` | `RhfConvergence { energy: 1e-12, residual: 1e-10, density: 1e-10, }` | Nested RhfConvergence: positive energy-change (Hartree), commutator-residual (Hartree) and dimensionless density-change thresholds. |
| `speed_of_light` | `f64` | `137.035999084` (`SPEED_OF_LIGHT_AU`) | Positive speed of light in atomic units. |
| `max_work_bytes` | `usize` | `256 * 1024 * 1024` | Conservative matrix/integral workspace bound in bytes. |

### `DFHFOptions`

Source and input validation: [ri_df_hf.rs](../src/hf_variants/ri_df_hf.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `basis_kind` | `BasisKind` | `BasisKind::Cartesian` | Physical AO space, BasisKind::Cartesian or BasisKind::Spherical. |
| `max_iterations` | `usize` | `100` | Positive solver iteration limit. |
| `energy_tolerance` | `f64` | `1e-12` | Positive energy-change threshold in Hartree. |
| `residual_tolerance` | `f64` | `1e-10` | Positive solver stationarity/residual threshold; definition follows the model equations. |
| `density_tolerance` | `f64` | `1e-10` | Positive density-change threshold in the solver working frame. |
| `metric_cutoff` | `f64` | `1e-10` | Positive absolute auxiliary Coulomb-metric eigenvalue cutoff in Hartree bohr^3. |
| `estimate_fitting_error` | `bool` | `false` | Optional streamed exact-ERI comparison. Off by default: computing the full fitting error has fourth-power cost even without storing ERIs. |

### `SigmaScfOptions`

Source and input validation: [sigma_scf.rs](../src/hf_variants/sigma_scf.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `max_iterations` | `usize` | `1000` | Positive solver iteration limit. |
| `gradient_tol` | `f64` | `1e-7` | Positive variance orbital-gradient threshold in Hartree squared. |
| `variance_tol` | `f64` | `1e-12` | Positive change threshold for the energy variance in Hartree squared. |

### `SaHfOptions`

Source and input validation: [state_averaged.rs](../src/hf_variants/state_averaged.rs).

| Field | Rust type | Typed default | Meaning / unit |
| --- | --- | --- | --- |
| `max_iterations` | `usize` | `500` | Positive solver iteration limit. |
| `gradient_tol` | `f64` | `1e-7` | Largest absolute orbital-gradient component. |
| `energy_tol` | `f64` | `1e-11` | Energy change of the last accepted step. |

## Positional controls in HF entry points

Some specialized entry points use positional model inputs instead of a dedicated
options struct. Atoms/basis and charge have the same meanings as the common
molecule constructor. Required positional arguments have no hidden defaults.
All orbital indices are zero-based.

| Rust entry point | Required additional inputs | Meaning / native unit |
| --- | --- | --- |
| `run_puhf` | `max_iter`, `tol` | Positive UHF iteration limit and energy tolerance (Hartree); electron parity chooses singlet/doublet |
| `run_puhf_with_multiplicity` | `multiplicity`, `max_iter`, `tol` | Project to S=(multiplicity-1)/2; singlets start from a broken-symmetry determinant |
| `run_tcscf` | `bonding_orbital`, `antibonding_orbital`, `max_iter`, `tol` | The first index must be occupied, the second virtual; positive iteration limit and energy tolerance (Hartree) |
| `run_tcscf_from_reference` | Existing RHF result and the same pair/iteration/tolerance inputs | Reuses that exact orbital frame and converged reference |
| `run_cd_hf` | `cholesky_delta`, `max_iter`, `tol` | Positive residual-diagonal cutoff (Hartree), iteration count and SCF energy tolerance (Hartree); residual/density thresholds are derived from tol |
| `run_cd_hf_with_options` | `cholesky_delta`, `&DFHFOptions` | Same Cholesky cutoff; independent explicit SCF/basis controls from the table above |
| `run_rohf`, `run_qrhf` | `multiplicity`, `max_iter`, `tol` | Positive 2S+1 compatible with electron parity, iteration limit and energy tolerance (Hartree) |
| `run_scf_mom` | `target_occ_indices`, `max_iter`, `tol` | Distinct occupied guess-orbital indices with the closed-shell electron count; positive iteration/energy controls |
| Legacy `run_scf_fermi_smearing` | `temperature_kelvin`, `max_iter`, `tol` | Explicitly named legacy Kelvin input; use common `fermi-hf` with `thermal_energy=kBT` for an entirely atomic-unit interface |

The common HF-model adapter supplies these inputs using its documented model
defaults; it does not expose every positional input as a Python/ASE keyword.
The [model catalog](methods.md#common-hf-model-option-defaults) distinguishes
common input keys from advanced typed Rust controls.
