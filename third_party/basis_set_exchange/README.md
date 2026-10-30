# Basis Set Exchange numerical data

hf-rs embeds orbital, auxiliary and scalar ECP data from the
[Basis Set Exchange](https://www.basissetexchange.org/), maintained by MolSSI
and PNNL/EMSL. The BSE project's [BSD 3-Clause license](LICENSE) and copyright
notice are retained here. This directory contains numerical exports and their
metadata; it does not contain a third-party electronic-structure implementation.

The 66 additional orbital families in `data/` extend the existing 14 to 80.
Each is stored in official BSE JSON and Gaussian 94 forms, with full element
coverage, original references and version metadata. Coverage depends on the
family: an ECP orbital basis need not define hydrogen or carbon.
`manifest.json` records export URLs, versions, element/ECP coverage and SHA256
checksums. Regenerate with `python tests/reference/fetch_basis_catalog.py`.
Git attributes preserve the export and license bytes without checkout newline
conversion, so the recorded checksums also remain valid on Windows.
The complete BSE license is also included in Python wheels under
`hf_rs_python-<version>.dist-info/licenses/third_party/basis_set_exchange/LICENSE`.

The existing 14 orbital and eight auxiliary families, plus def2 ECPs, remain
in `src/basis_data.rs`, generated with BSE 0.12. Their independent numerical
manifest is `tests/data/basis_manifest.json`; the additional families are
included in that manifest as well. Retaining the original exports preserves
the basis precision used by existing scientific oracle fixtures.

Built-in orbital names can be listed with `hf_rs.list_builtin_basis_sets()` or
the Rust `basis::list_builtin_basis_sets()` function. A named basis carries
its associated ECPs automatically. The independent name `def2-ecp`, other
ECP-bearing family names and scalar ECP files can override that choice.
See [basis and ECP selection](../../docs/python_api.md#basis-and-ecp-selection)
for element maps, fallback rules and explicit all-electron selection.

The complete BSE publications and original basis references are retained in
the exported data. The project reference is Pritchard et al.,
[J. Chem. Inf. Model. 59, 4814 (2019)](https://doi.org/10.1021/acs.jcim.9b00725).
