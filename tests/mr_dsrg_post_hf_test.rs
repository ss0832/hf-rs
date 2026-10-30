// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

use hf_rs::basis::{Atom, ContractedGaussian, STO_3G_GBS, build_basis, parse_gaussian94};
use hf_rs::post_hf::caspt2::{CASPT2, CASPT2Options};
use hf_rs::post_hf::casscf::{CASSCF, CASSCFOptions};
use hf_rs::post_hf::cc::ccd::CCD;
use hf_rs::post_hf::cc::ccsd::{CCSD, UHFCCSD};
use hf_rs::post_hf::ci::cis::CIS;
use hf_rs::post_hf::ci::fci::FCI;
use hf_rs::post_hf::ci::mrci::MRCISD;
use hf_rs::post_hf::f12::ccsd_f12::{CCSDF12Variant, RHFCCSDF12};
use hf_rs::post_hf::grad_common::{
    compute_cis_analytical_gradient, compute_mrci_analytical_gradient,
};
use hf_rs::post_hf::hessian::{
    PostHFType, compute_post_hf_gradient, compute_post_hf_hessian, compute_uhf_post_hf_gradient,
};
use hf_rs::post_hf::mp2::RHFMP2;
use hf_rs::post_hf::mp2_hessian::{
    compute_rhf_mp2_analytical_hessian, compute_uhf_mp2_analytical_hessian,
};
use hf_rs::post_hf::mpn::{RHFMP3, RHFMP4, UHFMP3};
use hf_rs::post_hf::mr_dsrg::*;
use hf_rs::post_hf::{PostHFMethod, PostHFResult};
use hf_rs::rohf::run_rohf;
use hf_rs::scf::{run_scf, run_scf_with_charge, run_uhf};

fn setup_h2(r_bohr: f64) -> (Vec<Atom>, Vec<ContractedGaussian>) {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = vec![
        Atom::new("H", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, r_bohr]),
    ];
    let basis = build_basis(&atoms, &basis_map);
    (atoms, basis)
}

fn setup_heh_plus(r_bohr: f64) -> (Vec<Atom>, Vec<ContractedGaussian>) {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = vec![
        Atom::new("He", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, r_bohr]),
    ];
    let basis = build_basis(&atoms, &basis_map);
    (atoms, basis)
}

fn setup_lih(r_bohr: f64) -> (Vec<Atom>, Vec<ContractedGaussian>) {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms = vec![
        Atom::new("Li", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, r_bohr]),
    ];
    let basis = build_basis(&atoms, &basis_map);
    (atoms, basis)
}

#[test]
fn test_bugfixes_1_to_6_verification() {
    // Bug 1: CCD and CCSD full quadratic contractions match exact FCI for 2-electron systems
    let (atoms_h2, basis_h2) = setup_h2(1.4);
    let hf_h2 = run_scf(&atoms_h2, &basis_h2, 60, 1e-12).unwrap();
    let fci_h2 = FCI::new(&hf_h2).compute().unwrap();
    let ccd_h2 = CCD::new(&hf_h2).compute().unwrap();
    let ccsd_h2 = CCSD::new(&hf_h2).compute().unwrap();
    assert!(
        (ccd_h2.total_energy - fci_h2.total_energy).abs() < 1e-7,
        "H2 CCD ({}) must match FCI ({})",
        ccd_h2.total_energy,
        fci_h2.total_energy
    );
    assert!(
        (ccsd_h2.total_energy - fci_h2.total_energy).abs() < 1e-7,
        "H2 CCSD ({}) must match FCI ({})",
        ccsd_h2.total_energy,
        fci_h2.total_energy
    );

    let (atoms_heh, basis_heh) = setup_heh_plus(1.46);
    let hf_heh = run_scf_with_charge(&atoms_heh, &basis_heh, 1, 60, 1e-12).unwrap();
    let fci_heh = FCI::new(&hf_heh).compute().unwrap();
    let ccsd_heh = CCSD::new(&hf_heh).compute().unwrap();
    assert!(
        (ccsd_heh.total_energy - fci_heh.total_energy).abs() < 1e-6,
        "HeH+ CCSD ({}) must match FCI ({})",
        ccsd_heh.total_energy,
        fci_heh.total_energy
    );

    // Bug 2: UHF-CCSD and UHF-MP3 spin-orbital consistency with RHF on closed-shell LiH and open-shell Li
    let (atoms_lih, basis_lih) = setup_lih(3.015);
    let rhf_lih = run_scf(&atoms_lih, &basis_lih, 60, 1e-12).unwrap();
    let uhf_lih = run_uhf(&atoms_lih, &basis_lih, 0, 1, false, 60, 1e-12).unwrap();

    let rhf_mp3 = RHFMP3::new(&rhf_lih).compute().unwrap();
    let uhf_mp3 = UHFMP3::new(&uhf_lih).compute().unwrap();
    assert!(
        (rhf_mp3.total_energy - uhf_mp3.total_energy).abs() < 1e-8,
        "Closed-shell UHF-MP3 ({}) must match RHF-MP3 ({})",
        uhf_mp3.total_energy,
        rhf_mp3.total_energy
    );

    let rhf_ccsd = CCSD::new(&rhf_lih).compute().unwrap();
    let uhf_ccsd = UHFCCSD::new(&uhf_lih).compute().unwrap();
    assert!(
        (rhf_ccsd.total_energy - uhf_ccsd.total_energy).abs() < 1e-7,
        "Closed-shell UHF-CCSD ({}) must match RHF-CCSD ({})",
        uhf_ccsd.total_energy,
        rhf_ccsd.total_energy
    );

    // Open-shell doublet OH radical UHF-MP3 and UHF-CCSD
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let atoms_oh = vec![
        Atom::new("O", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, 1.83]),
    ];
    let basis_oh = build_basis(&atoms_oh, &basis_map);
    let uhf_oh = run_uhf(&atoms_oh, &basis_oh, 0, 2, false, 80, 1e-11).unwrap();
    let uhf_mp3_oh = UHFMP3::new(&uhf_oh).compute().unwrap();
    let uhf_ccsd_oh = UHFCCSD::new(&uhf_oh).compute().unwrap();
    assert!(uhf_mp3_oh.correlation_energy() < 0.0);
    assert!(uhf_ccsd_oh.correlation_energy < 0.0);
    assert!(uhf_ccsd_oh.converged);

    // Bug 3: RHFMP4 SDTQ decomposition
    let mp4_lih = RHFMP4::new(&rhf_lih).compute().unwrap();
    let sum_sdq = mp4_lih.mp2_correlation_energy
        + mp4_lih.mp3_correlation_energy
        + mp4_lih.e_singles
        + mp4_lih.e_doubles
        + mp4_lih.e_quadruples;
    let sum_sdtq = sum_sdq + mp4_lih.e_triples;
    assert!((mp4_lih.mp4_sdq_correlation_energy - sum_sdq).abs() < 1e-12);
    assert!((mp4_lih.mp4_total_correlation_energy - sum_sdtq).abs() < 1e-12);
    assert!(mp4_lih.e_doubles.abs() > 1e-6);

    // Both F12 variants solve their modified amplitude equations. Independent
    // tensor/molecular energy oracles live in the dedicated F12 test binaries.
    let f12a = RHFCCSDF12::new(&rhf_lih, &atoms_lih, &basis_lih, CCSDF12Variant::F12a)
        .compute()
        .unwrap();
    let f12b = RHFCCSDF12::new(&rhf_lih, &atoms_lih, &basis_lih, CCSDF12Variant::F12b)
        .compute()
        .unwrap();
    assert!(
        f12b.coupling_energy.abs() > 1e-10,
        "CCSD-F12b coupling term must be non-zero"
    );
    assert!(f12a.ccsd_result.converged && f12b.ccsd_result.converged);
    // The common C/tau term is already present in F12a. Only the difference
    // of the coupling functionals is added when comparing these variants.
    assert!((f12b.total_energy-f12a.total_energy
        -(f12b.coupling_energy-f12a.coupling_energy)).abs()<1e-12);

    // DLPNO full/finite-space and legacy adapters are verified separately
    // against same-auxiliary RI references in the local correlation tests.

    // Bug 6: ROHF CDIIS convergence to < 1e-11
    let rohf_oh = run_rohf(&atoms_oh, &basis_oh, 0, 2, 80, 1e-11).unwrap();
    assert!(rohf_oh.total_energy < -74.0);
    assert!(rohf_oh.converged);
}

#[test]
fn test_new_analytical_gradients_and_hessians() {
    let basis_map = parse_gaussian94(STO_3G_GBS);
    let (atoms, basis) = setup_h2(1.45);
    let hf = run_scf(&atoms, &basis, 60, 1e-12).unwrap();

    // CIS analytical gradient vs finite difference
    let cis = CIS::new(&hf).compute().unwrap();
    let cis_grad = compute_cis_analytical_gradient(&atoms, &basis, &hf, &cis, 0);
    let h = 1e-4;
    let (_, basis_p) = setup_h2(1.45 + h);
    let atoms_p = vec![
        Atom::new("H", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, 1.45 + h]),
    ];
    let hf_p = run_scf(&atoms_p, &basis_p, 60, 1e-12).unwrap();
    let cis_p = CIS::new(&hf_p).compute().unwrap();
    let e_cis_p = hf_p.total_energy + cis_p.excited_states[0].excitation_energy_hartree;

    let (_, basis_m) = setup_h2(1.45 - h);
    let atoms_m = vec![
        Atom::new("H", [0.0, 0.0, 0.0]),
        Atom::new("H", [0.0, 0.0, 1.45 - h]),
    ];
    let hf_m = run_scf(&atoms_m, &basis_m, 60, 1e-12).unwrap();
    let cis_m = CIS::new(&hf_m).compute().unwrap();
    let e_cis_m = hf_m.total_energy + cis_m.excited_states[0].excitation_energy_hartree;
    let fd_cis_gz = (e_cis_p - e_cis_m) / (2.0 * h);
    assert!(
        (cis_grad[1][2] - fd_cis_gz).abs() < 1e-5,
        "CIS analytical gradient ({}) must match FD ({})",
        cis_grad[1][2],
        fd_cis_gz
    );

    // MRCI analytical gradient vs finite difference
    let mrci_grad = compute_mrci_analytical_gradient(&atoms, &basis, &hf, 2, 2);
    let mrci_p = MRCISD::new(&hf_p, 2, 2).compute().unwrap().total_energy;
    let mrci_m = MRCISD::new(&hf_m, 2, 2).compute().unwrap().total_energy;
    let fd_mrci_gz = (mrci_p - mrci_m) / (2.0 * h);
    assert!(
        (mrci_grad[1][2] - fd_mrci_gz).abs() < 1e-6,
        "MRCI analytical gradient ({}) must match FD ({})",
        mrci_grad[1][2],
        fd_mrci_gz
    );

    // MP3, MP4, CCD, CCSD, CCSD(T), CASPT2 analytical gradients
    for method in [
        PostHFType::MP3,
        PostHFType::MP4,
        PostHFType::CCD,
        PostHFType::CCSD,
        PostHFType::CCSDT,
        PostHFType::CASPT2,
    ] {
        let g = compute_post_hf_gradient(&atoms, &basis_map, 0, method, 60, 1e-12);
        // Translational invariance: g[0][2] + g[1][2] == 0
        assert!(
            (g[0][2] + g[1][2]).abs() < 1e-7,
            "Translational invariance violated for {:?}",
            method
        );
    }

    // Open-shell UHF-MP3 and UHF-CCSD analytical gradients
    let g_uhf_mp3 =
        compute_uhf_post_hf_gradient(&atoms, &basis_map, 0, 1, false, PostHFType::MP3, 60, 1e-12);
    let g_rhf_mp3 = compute_post_hf_gradient(&atoms, &basis_map, 0, PostHFType::MP3, 60, 1e-12);
    assert!((g_uhf_mp3[1][2] - g_rhf_mp3[1][2]).abs() < 1e-6);

    // Pure analytical RHF-MP2 and UHF-MP2 Hessians
    let hess_rhf = compute_rhf_mp2_analytical_hessian(&atoms, &basis_map, 0, 60, 1e-12);
    let hess_uhf = compute_uhf_mp2_analytical_hessian(&atoms, &basis_map, 0, 1, false, 60, 1e-12);
    assert!((hess_rhf[(2, 2)] - hess_uhf[(2, 2)]).abs() < 1e-6);
    assert!((hess_rhf[(2, 5)] + hess_rhf[(2, 2)]).abs() < 1e-6);

    // Post-HF analytical Hessian wrapper
    let hess_ccsd = compute_post_hf_hessian(&atoms, &basis_map, 0, PostHFType::CCSD, 60, 1e-11);
    assert!(hess_ccsd[(2, 2)] > 0.0);
}

#[test]
fn test_gno_cumulants_and_selected_ci_suite() {
    let (atoms, basis) = setup_lih(3.015);
    let hf = run_scf(&atoms, &basis, 60, 1e-12).unwrap();
    let fci = FCI::new(&hf).compute().unwrap();

    // 1. Cumulants & Generalized Normal Ordering (GNO)
    let casscf = CASSCF::new(&hf, CASSCFOptions::new(2, 2))
        .compute()
        .unwrap();
    for approx in [ThreePDC::MK, ThreePDC::MKDecomp, ThreePDC::Zero] {
        let cum = compute_cumulants(
            &hf,
            &casscf.one_rdm_mo,
            &casscf.two_rdm_mo,
            casscf.n_core,
            casscf.n_active_orbitals,
            approx,
        );
        let n_mo = cum.gamma1.rows;
        let tr_gamma: f64 = (0..n_mo).map(|u| cum.gamma1[(u, u)]).sum();
        assert!(
            (tr_gamma - 4.0).abs() < 1e-10,
            "Tr(gamma1) must equal N_elec (4)"
        );
        for u in 0..n_mo {
            for v in 0..n_mo {
                let two_id = if u == v { 2.0 } else { 0.0 };
                assert!((cum.gamma1[(u, v)] + cum.eta1[(u, v)] - two_id).abs() < 1e-12);
            }
        }
    }

    // 2. ACI and SA-ACI
    let aci_res = ACI::new(
        &hf,
        ACIOptions {
            sigma: 0.0,
            gamma: 1.0,
            n_roots: 1,
            max_cycles: 100,
            compute_pt2: true,
        },
    )
    .compute()
    .unwrap();
    assert!(
        (aci_res.total_energy - fci.total_energy).abs() < 1e-10,
        "ACI+PT2 ({}) must approach FCI ({})",
        aci_res.total_energy,
        fci.total_energy
    );

    let sa_aci = ACI::new(
        &hf,
        ACIOptions {
            sigma: 1e-4,
            gamma: 1.0,
            n_roots: 2,
            max_cycles: 8,
            compute_pt2: true,
        },
    )
    .compute()
    .unwrap();
    assert_eq!(sa_aci.root_energies.len(), 2);
    assert!(sa_aci.root_energies[1] > sa_aci.root_energies[0]);

    // 3. ASCI
    let asci_res = ASCI::new(
        &hf,
        ASCIOptions {
            n_core_dets: 225,
            n_target_dets: 225,
            compute_pt2: true,
            max_iter: 100,
        },
    )
    .compute()
    .unwrap();
    assert!(
        (asci_res.total_energy - fci.total_energy).abs() < 1e-10,
        "ASCI+PT2 ({}) must match FCI ({})",
        asci_res.total_energy,
        fci.total_energy
    );

    // 4. HBCI
    let hbci_res = HBCI::new(
        &hf,
        HBCIOptions {
            eps1: 0.0,
            eps2: 0.0,
            compute_pt2: true,
            max_cycles: 100,
        },
    )
    .compute()
    .unwrap();
    assert!(
        (hbci_res.total_energy - fci.total_energy).abs() < 1e-10,
        "HBCI+PT2 ({}) must match FCI ({})",
        hbci_res.total_energy,
        fci.total_energy
    );

    // 5. Molecular compatibility for every generator, without the former final
    // rediagonalization that hid incomplete propagation. The finite-threshold
    // wavefunctions are independently validated in selected_ci_test.
    let (pci_atoms, pci_basis) = setup_h2(1.4);
    let pci_hf = run_scf(&pci_atoms, &pci_basis, 100, 1e-14).unwrap();
    let pci_fci = FCI::new(&pci_hf).compute().unwrap();
    for pci_gen in [
        PCIGenerator::WallChebyshev,
        PCIGenerator::Chebyshev,
        PCIGenerator::ExpChebyshev,
        PCIGenerator::Linear,
        PCIGenerator::Quadratic,
        PCIGenerator::Cubic,
        PCIGenerator::Quartic,
        PCIGenerator::Power,
        PCIGenerator::Davidson,
        PCIGenerator::Lanczos,
    ] {
        let pci_res = PCI::new(
            &pci_hf,
            PCIOptions {
                generator: pci_gen,
                spawning_threshold: 0.0,
                polynomial_order: 5,
                max_steps: 5000,
                compute_pt2: false,
            },
        )
        .compute()
        .unwrap();
        assert!(
            (pci_res.total_energy - pci_fci.total_energy).abs() < 1e-10,
            "PCI ({:?}) energy {} must match FCI {}",
            pci_gen,
            pci_res.total_energy,
            pci_fci.total_energy
        );
    }
}

#[test]
fn test_gasci_gasscf_doci_dmrg_and_avas() {
    let (atoms, basis) = setup_lih(3.015);
    let hf = run_scf(&atoms, &basis, 60, 1e-12).unwrap();

    // 1. GASCI and GASSCF
    let gas_opts_ci = GASOptions::new(
        1,
        vec![GASSubspace::new(1, 0, 2), GASSubspace::new(1, 0, 2)],
        false,
    );
    let gasci = GASCI::new(&hf, gas_opts_ci).compute().unwrap();
    let gas_opts_scf = GASOptions::new(
        1,
        vec![GASSubspace::new(1, 0, 2), GASSubspace::new(1, 0, 2)],
        true,
    );
    let gasscf = GASCI::new(&hf, gas_opts_scf).compute().unwrap();
    assert!(gasci.total_energy <= hf.total_energy + 1e-10);
    assert!(gasscf.total_energy <= gasci.total_energy + 1e-10);

    // 2. Seniority-Zero DOCI
    let doci = DOCI::new(&hf).compute().unwrap();
    let fci = FCI::new(&hf).compute().unwrap();
    assert!(doci.total_energy <= hf.total_energy);
    assert!(doci.total_energy >= fci.total_energy - 1e-10);

    // 3. Active-Space DMRG and DMRG-SCF vs GASCI/CASSCF
    let dmrg = ActiveSpaceDMRG::new(&hf, DMRGOptions::new(2, 2, 16))
        .compute()
        .unwrap();
    assert!(
        (dmrg.total_energy - gasci.total_energy).abs() < 1e-8,
        "DMRG ({}) must match CAS(2,2)-CI ({})",
        dmrg.total_energy,
        gasci.total_energy
    );
    let mut dmrg_scf_opts = DMRGOptions::new(2, 2, 16);
    dmrg_scf_opts.optimize_orbitals = true;
    let dmrg_scf = ActiveSpaceDMRG::new(&hf, dmrg_scf_opts).compute().unwrap();
    assert!(dmrg_scf.total_energy <= dmrg.total_energy + 1e-10);

    // 4. AVAS, ASET, and Natural Orbital Selection
    let avas = AVAS::new(&hf, &basis, AVASOptions::default())
        .compute()
        .unwrap();
    assert!(avas.n_active_orbitals >= 2);
    assert!(avas.total_energy <= hf.total_energy + 1e-10);

    for scheme in [
        ASETVirtualScheme::ASET,
        ASETVirtualScheme::PAO,
        ASETVirtualScheme::IAO,
    ] {
        let aset_opts = ASETOptions {
            fragment_atoms: vec![0],
            threshold: 0.2,
            virtual_scheme: scheme,
        };
        let aset = ASET::new(&hf, &basis, aset_opts).compute().unwrap();
        assert!(aset.total_energy <= hf.total_energy + 1e-10);
    }

    for src in [
        NaturalOrbitalKind::MP2NO,
        NaturalOrbitalKind::CINO,
        NaturalOrbitalKind::MRCINO,
    ] {
        let no_sel = NaturalOrbitalSelector::new(&hf, src, 0.001, 1.999)
            .compute()
            .unwrap();
        assert!(no_sel.n_active_orbitals >= 2);
    }
}

#[test]
fn test_single_and_multireference_dsrg_suite() {
    let (atoms, basis) = setup_lih(3.015);
    let hf = run_scf(&atoms, &basis, 60, 1e-12).unwrap();

    // 1. Single-Reference DSRG: s -> infinity reproduces canonical MP2 / MP3
    let mp2 = RHFMP2::new(&hf).compute().unwrap();
    let sr_dsrg2_inf = SRDSRGPT2::new(&hf, 1e6).compute().unwrap();
    assert!(
        (sr_dsrg2_inf.correlation_energy - mp2.correlation_energy).abs() < 1e-8,
        "SR-DSRG-PT2(s->inf) ({}) must match MP2 ({})",
        sr_dsrg2_inf.correlation_energy,
        mp2.correlation_energy
    );

    let sr_dsrg2_half = SRDSRGPT2::new(&hf, 0.5).compute().unwrap();
    assert!(sr_dsrg2_half.correlation_energy < 0.0);
    assert!(sr_dsrg2_half.correlation_energy.abs() < sr_dsrg2_inf.correlation_energy.abs());

    // UHF-SR-DSRG-PT2 matches RHF-SR-DSRG-PT2 on closed-shell singlet
    let uhf = run_uhf(&atoms, &basis, 0, 1, false, 60, 1e-12).unwrap();
    let uhf_sr_dsrg2 = UHFSRDSRGPT2::new(&uhf, 0.5).compute().unwrap();
    assert!(
        (uhf_sr_dsrg2.correlation_energy - sr_dsrg2_half.correlation_energy).abs() < 1e-8,
        "UHF-SR-DSRG-PT2 ({}) must match RHF-SR-DSRG-PT2 ({})",
        uhf_sr_dsrg2.correlation_energy,
        sr_dsrg2_half.correlation_energy
    );

    // SR-DSRG-PT3(s -> inf) matches canonical MP3
    let mp3 = RHFMP3::new(&hf).compute().unwrap();
    let sr_dsrg3_inf = SRDSRGPT3::new(&hf, 1e6).compute().unwrap();
    assert!(
        (sr_dsrg3_inf.correlation_energy - mp3.correlation_energy()).abs() < 1e-7,
        "SR-DSRG-PT3(s->inf) ({}) must match MP3 ({})",
        sr_dsrg3_inf.correlation_energy,
        mp3.correlation_energy()
    );

    // SR-LDSRG(2) and continuous-flow SRG-ODE (Wegner & White generators)
    let sr_ldsrg2 = SRLDSRG2::new(&hf, 1.0).compute().unwrap();
    assert!(sr_ldsrg2.converged);
    assert!(sr_ldsrg2.correlation_energy < 0.0);

    for srg_gen in [SRGGenerator::Wegner, SRGGenerator::White] {
        let srg_ode = SRGODE::new(&hf, srg_gen, 0.5, 25)
            .with_ode_tolerance(1e-11)
            .compute()
            .unwrap();
        assert!(srg_ode.correlation_energy < 0.0);
        assert!(srg_ode.residual_off_diagonal_norm.is_finite());
    }

    // 2. Multireference DSRG-MRPT2, 3-DSRG-MRPT2, DSRG-MRPT3, MR-LDSRG(2), MR-LDSRG(2)-QC
    let mr_opts = MRDSRGOptions::new(2, 2);
    let dsrg_mrpt2 = DSRGMRPT2::new(&hf, mr_opts.clone()).compute().unwrap();
    let three_dsrg = ThreeDSRGMRPT2::new(&hf, mr_opts.clone(), 1e-12)
        .compute()
        .unwrap();
    assert!(
        (dsrg_mrpt2.relaxed_total_energy - three_dsrg.relaxed_total_energy).abs() < 1e-7,
        "THREE-DSRG-MRPT2 ({}) must match exact 4-index DSRG-MRPT2 ({})",
        three_dsrg.relaxed_total_energy,
        dsrg_mrpt2.relaxed_total_energy
    );

    let dsrg_mrpt3 = DSRGMRPT3::new(&hf, mr_opts.clone()).compute().unwrap();
    let mr_ldsrg2 = MRLDSRG2::new(&hf, mr_opts.clone()).compute().unwrap();
    let mr_ldsrg2_qc = MRLDSRG2::new_qc(&hf, mr_opts.clone()).compute().unwrap();
    assert!(dsrg_mrpt3.relaxed_total_energy < hf.total_energy);
    assert!(mr_ldsrg2.converged);
    assert!(mr_ldsrg2_qc.converged);

    // Unimplemented compatibility approximations return explicit errors.
    for h0 in [
        ZerothOrderH::Fdiag,
        ZerothOrderH::Ffull,
        ZerothOrderH::FdiagVactv,
        ZerothOrderH::FdiagVdiag,
    ] {
        let mut opts = MRDSRGOptions::new(2, 2);
        opts.zeroth_order_h = h0;
        opts.relax_ref = RelaxRef::Iterate;
        opts.nivo = true;
        opts.sequential_transformation = true;
        assert!(DSRGMRPT2::new(&hf, opts).compute().is_err());
    }

    // Stretched H2 bond (R = 3.8 Bohr): DSRG-MRPT2 stays smooth and intruder-state-free
    let (atoms_str, basis_str) = setup_h2(3.8);
    let hf_str = run_scf(&atoms_str, &basis_str, 80, 1e-12).unwrap();
    let fci_str = FCI::new(&hf_str).compute().unwrap();
    let dsrg_str = DSRGMRPT2::new(&hf_str, MRDSRGOptions::new(2, 2))
        .compute()
        .unwrap();
    let caspt2_str = CASPT2::from_hf(&hf_str, CASPT2Options::new(2, 2)).unwrap();
    assert!((dsrg_str.relaxed_total_energy - fci_str.total_energy).abs() < 1e-3);
    assert!(caspt2_str.total_energy.is_finite());

    // 3. Multi-State DSRG (SA-DSRG, MS-DSRG-PT2, XMS-DSRG-PT2, DWMS-DSRG-PT2)
    for variant in [
        MultiStateDSRGVariant::StateAverage,
        MultiStateDSRGVariant::MS,
        MultiStateDSRGVariant::XMS,
        MultiStateDSRGVariant::DWMS { zeta: 0.5 },
    ] {
        let ms_res = MultiStateDSRG::new(&hf, MRDSRGOptions::new(2, 2), 2, variant)
            .compute()
            .unwrap();
        assert_eq!(ms_res.state_energies.len(), 2);
        assert!(ms_res.state_energies[1] > ms_res.state_energies[0]);
        assert!(
            (ms_res.effective_hamiltonian[(0, 1)] - ms_res.effective_hamiltonian[(1, 0)]).abs()
                < 1e-12
        );
    }
}
