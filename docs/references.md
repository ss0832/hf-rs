# Scientific references

The mathematical references define models and validation protocols. Software
citations below acknowledge external numerical oracle validation. Detailed
equations and independently tested scope are linked from [methods](methods.md).

## Related development philosophy

Related development philosophy: M. F. Peintinger,
[*vibe-qc: An Open-Source, General-Purpose Electronic-Structure Code for Molecules
and Solids Developed With Generative AI*](https://onlinelibrary.wiley.com/doi/10.1002/jcc.70525),
J. Comput. Chem. 47, e70525 (2026), DOI: 10.1002/jcc.70525. This citation concerns only the
similar development philosophy. It is not a source for hf-rs equations or
implementation; the accompanying code is not used as an implementation reference.

## Mathematical models

- L. E. McMurchie and E. R. Davidson, *J. Comput. Phys.* 26, 218–231 (1978),
  [Gaussian integral recurrences](https://doi.org/10.1016/0021-9991(78)90092-X).
- P. Jørgensen and T. Helgaker, *J. Chem. Phys.* 89, 1560–1570 (1988),
  [Møller–Plesset energy derivatives](https://doi.org/10.1063/1.455152).
  T. Helgaker, [derivative/Lagrangian lectures](https://trygvehelgaker.no/Presentations/MWM16_DerivativesProperties_2016.pdf).
- J. A. Pople, M. Head-Gordon and K. Raghavachari, *J. Chem. Phys.* 87, 5968 (1987),
  [quadratic CI and perturbative correlation](https://doi.org/10.1063/1.453520).
- J. F. Stanton, J. Gauss, J. D. Watts and R. J. Bartlett, *J. Chem. Phys.* 94, 4334 (1991),
  [CCSD equations](https://doi.org/10.1063/1.460620).
- P. J. Knowles and N. C. Handy, *Chem. Phys. Lett.* 111, 315 (1984),
  [determinant CI algorithm](https://doi.org/10.1016/0009-2614(84)85513-X).
- K. Andersson, P.-A. Malmqvist and B. O. Roos, *J. Chem. Phys.* 96, 1218 (1992),
  [CASPT2](https://doi.org/10.1063/1.462209); detailed shifts/IC references in [CASPT2](caspt2.md).
- S. R. White, *Phys. Rev. Lett.* 69, 2863 (1992),
  [DMRG](https://doi.org/10.1103/PhysRevLett.69.2863); implementation model and SZ validation in [DMRG](dmrg.md).
- H.-J. Werner, T. B. Adler and F. R. Manby, *J. Chem. Phys.* 126, 164102 (2007),
  [orbital-invariant MP2-F12](https://doi.org/10.1063/1.2712434).
- T. B. Adler, G. Knizia and H.-J. Werner, *J. Chem. Phys.* 127, 221106 (2007),
  [CCSD-F12a/b](https://doi.org/10.1063/1.2817618).
- G. Knizia, T. B. Adler and H.-J. Werner, *J. Chem. Phys.* 130, 054104 (2009),
  [CCSD(T)-F12 theory and benchmarks](https://doi.org/10.1063/1.3054300).
- G. Knizia, [*Explicitly correlated quantum chemistry methods for high-spin open-shell molecules*](https://elib.uni-stuttgart.de/server/api/core/bitstreams/587dac0c-7ea0-47d0-9a5f-92e4fa47e301/content),
  doctoral thesis, University of Stuttgart (2010):
  robust fitting, hybrid equations, orthogonal exact/DF blocks and auxiliary-space continuity.
- K. Patkowski, *J. Chem. Phys.* 137, 034103 (2012),
  [He2 F12 comparison](https://doi.org/10.1063/1.4734597). The current F12 printed-digit discrepancy remains open.
- F. A. Evangelista and C. Li: primary driven-SRG, MR perturbative and LDSRG
  references are collected with their equations in [SR-DSRG](sr_dsrg.md),
  [MR](mr_driven.md), [multistate](multistate_dsrg.md) and [THREE](three_dsrg.md).

HF-variant primary references and model-specific conventions are retained in
[GHF](ghf.md), [complex HF](complex_hf.md), [scalar relativity](scalar_relativistic.md),
[analytic HF/ECP response](analytic_hf_derivatives.md) and the validation record.

## Numerical oracle software

The following programs were used as external numerical oracles for the
comparisons recorded in [validation](validation.md). The reference generators
execute installed programs and compare their returned energies, integrals,
amplitudes, density matrices and available response quantities at the documented
settings. These citations acknowledge oracle use; the programs are not runtime
dependencies of native hf-rs calculations. The recorded comparisons use public
execution interfaces and outputs, without consulting external implementation
source as a reference for the corresponding hf-rs implementation. In particular,
Forte was used only as an executable oracle, without using its implementation
source or internal implementation routines.

- **Psi4:** D. G. A. Smith et al.,
  [*Psi4 1.4: Open-source software for high-throughput quantum chemistry*](https://doi.org/10.1063/5.0006002),
  J. Chem. Phys. 152, 184108 (2020). Used for independent energy and public
  integral-output comparisons, including the recorded F12 component oracles.
  The actual oracle versions and enabled features are stated with each check;
  the software-paper version is not a claim about the version used for F12.
- **Forte:** F. A. Evangelista et al.,
  [*Forte: A Suite of Advanced Multireference Quantum Chemistry Methods*](https://arxiv.org/abs/2405.10197),
  arXiv:2405.10197 (2024), preprint. Used for the recorded executable MR-DSRG
  energy/reference-relaxation and available state-averaged comparisons.
  Unavailable MS/XMS/DWMS executable comparisons remain explicitly unclaimed.
- **PySCF:** Q. Sun et al.,
  [*The Python simulations of chemistry framework: 10 years of an open-source quantum chemistry project*](https://doi.org/10.1063/5.0337441),
  J. Chem. Phys. 165, 102502 (2026);
  [*Recent developments in the PySCF program package*](https://doi.org/10.1063/5.0006074),
  J. Chem. Phys. 153, 024109 (2020); and
  [*PySCF: the Python-based simulations of chemistry framework*](https://doi.org/10.1002/wcms.1340),
  WIREs Comput. Mol. Sci. 8, e1340 (2018). Used for independent conventional
  molecular energies, integrals, RDMs, analytic forces/Hessians where available,
  and documented public-API numerical oracles.

OpenMolcas and block2 SZ additionally supply the CASPT2 and DMRG executable
comparisons identified in the validation record. Numerical settings, actual
versions and unavailable comparisons remain explicit for every oracle.
