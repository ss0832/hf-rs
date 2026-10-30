// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! The sole method-name parser and typed post-HF adapter registry.
use crate::context::{PostHFContext, RefKind};
use crate::error::HfError;
use crate::options::*;
use crate::output::PostHFOutput;
use crate::post_hf::{PostHFMethod, PostHFResult};
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientKind {
    Analytic,
    Numerical,
    Unsupported,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MethodFamily {
    Hf,
    Perturbation,
    CoupledCluster,
    Ci,
    Multireference,
    Local,
    F12,
    Dsrg,
    Dmrg,
    Sci,
}

/// The only string dispatcher for molecular HF-variant names.
pub fn parse_hf_variant(input: &str) -> Result<crate::hf_variants::HFVariantType, HfError> {
    use crate::hf_variants::HFVariantType::*;
    match input
        .to_ascii_lowercase()
        .replace(['-', '_', ' '], "")
        .as_str()
    {
        "rohf" => Ok(ROHF),
        "ghf" => Ok(GHF),
        "puhf" | "phf" | "suhf" => Ok(PUHF),
        "hfb" => Ok(HFB),
        "tcscf" | "gvb" => Ok(TCSCF),
        "neohf" | "neo" => Ok(NEOHF),
        "qedhf" | "qed" | "polaritonic" => Ok(QEDHF),
        "relativistichf" | "x2c" | "x2chf" => Ok(RelativisticHF),
        "dkh" | "dkh2" | "dkh2hf" => Ok(Dkh2HF),
        "zora" | "zorahf" => Ok(ZoraHF),
        "cdhf" | "cholesky" => Ok(CDHF),
        "dfhf" | "rihf" | "densityfitting" => Ok(DFHF),
        "complexhf" | "chf" | "crhf" => Ok(ComplexHF),
        "cuhf" => Ok(ComplexUHF),
        "cghf" => Ok(ComplexGHF),
        "lhf" => Ok(LHF),
        "oep" | "oepexx" => Ok(OEP),
        "positronichf" | "positron" => Ok(PositronicHF),
        "dchf" | "dc" | "divideandconquer" => Ok(DCHF),
        "fmo" | "fmo2" => Ok(FMO),
        "fmo3" => Ok(FMO3),
        _ => Err(HfError::UnknownMethod(input.into())),
    }
}

macro_rules! methods {
    ($($id:ident => ($name:literal,[$($alias:literal),*],$family:ident)),* $(,)?)=> {
        #[derive(Clone,Copy,Debug,PartialEq,Eq,Hash)]
        pub enum MethodId {$($id),*}
        impl MethodId {
            pub const ALL:&'static[MethodId]=&[$(Self::$id),*];
            pub fn name(self)->&'static str {match self {$(Self::$id=>$name),*}}
            pub fn family(self)->MethodFamily {match self {$(Self::$id=>MethodFamily::$family),*}}
        }
        impl FromStr for MethodId {
            type Err=HfError;
            fn from_str(input:&str)->Result<Self,HfError> {
                let normalized=input.to_ascii_lowercase().replace(['-','_',' '],"");
                let name=if normalized.len()>3 && (normalized.starts_with("rhf")||normalized.starts_with("uhf")) {&normalized[3..]} else {&normalized};
                match name {$( $($alias)|* => Ok(Self::$id), )* _=>Err(HfError::UnknownMethod(input.into()))}
            }
        }
    }
}
methods! {
    Rhf=>("RHF",["rhf","hf"],Hf),Uhf=>("UHF",["uhf"],Hf),Rohf=>("ROHF",["rohf"],Hf),Qrhf=>("QRHF",["qrhf"],Hf),
    Mom=>("MOM",["mom","momscf"],Hf),Fermi=>("Fermi-HF",["fermi","fermihf","fermismearing"],Hf),
    Ghf=>("GHF",["ghf"],Hf),Mp2=>("MP2",["mp2"],Perturbation),ScsMp2=>("SCS-MP2",["scsmp2"],Perturbation),
    RiMp2=>("RI-MP2",["rimp2","dfmp2"],Perturbation),Mp3=>("MP3",["mp3"],Perturbation),
    ScsMp3=>("SCS-MP3",["scsmp3"],Perturbation),Mp4=>("MP4",["mp4","mp4sdtq"],Perturbation),
    Mp5=>("MP5",["mp5"],Perturbation),Mp6=>("MP6",["mp6"],Perturbation),Mp7=>("MP7",["mp7"],Perturbation),Mp8=>("MP8",["mp8"],Perturbation),
    Ccd=>("CCD",["ccd"],CoupledCluster),Ccsd=>("CCSD",["ccsd"],CoupledCluster),CcsdT=>("CCSD(T)",["ccsd(t)","ccsdt"],CoupledCluster),
    Cis=>("CIS",["cis"],Ci),Cisd=>("CISD",["cisd"],Ci),Fci=>("FCI",["fci","casci"],Ci),
    Casscf=>("CASSCF",["casscf"],Multireference),Caspt2=>("CASPT2",["caspt2"],Multireference),
    Mrci=>("MRCISD",["mrci","mrcisd"],Multireference),MrciQ=>("MRCISD+Q",["mrci+q","mrcisd+q"],Multireference),
    DlpnoMp2=>("DLPNO-MP2",["dlpnomp2"],Local),DlpnoCcsd=>("DLPNO-CCSD",["dlpnoccsd"],Local),
    DlpnoCcsdT0=>("DLPNO-CCSD(T0)",["dlpnoccsd(t0)","dlpnoccsdt0"],Local),DlpnoCcsdT1=>("DLPNO-CCSD(T1)",["dlpnoccsd(t1)","dlpnoccsdt1","dlpnoccsdt"],Local),
    Mp2F12=>("MP2-F12",["mp2f12"],F12),CcsdF12a=>("CCSD-F12a",["ccsdf12a"],F12),CcsdF12b=>("CCSD-F12b",["ccsdf12b"],F12),
    CcsdTF12a=>("CCSD(T*)-F12a",["ccsd(t*)f12a","ccsdtf12a"],F12),CcsdTF12b=>("CCSD(T*)-F12b",["ccsd(t*)f12b","ccsdtf12b"],F12),
    Aci=>("ACI",["aci","acipt2"],Sci),Asci=>("ASCI",["asci","ascipt2"],Sci),Hbci=>("HBCI",["hbci","hci","hbcipt2"],Sci),Pci=>("PCI",["pci"],Sci),
    Gasci=>("GASCI",["gasci"],Multireference),Gasscf=>("GASSCF",["gasscf"],Multireference),Doci=>("DOCI",["doci"],Ci),
    Dmrg=>("DMRG",["dmrg","dmrgci"],Dmrg),DmrgScf=>("DMRGSCF",["dmrgscf"],Dmrg),
    Avas=>("AVAS",["avas","avascasci"],Multireference),Aset=>("ASET",["aset"],Multireference),
    SrDsrgPt2=>("SR-DSRG-PT2",["srdsrgpt2"],Dsrg),SrDsrgPt3=>("SR-DSRG-PT3",["srdsrgpt3"],Dsrg),SrLdsrg2=>("SR-LDSRG(2)",["srldsrg2","srldsrg(2)"],Dsrg),
    SrGode=>("IM-SRG(2)",["srgode","imsrg2","imsrg(2)","srgpt2","lsrg2"],Dsrg),
    DsrgMrPt2=>("DSRG-MRPT2",["dsrgmrpt2","mrdsrgpt2"],Dsrg),ThreeDsrgMrPt2=>("THREE-DSRG-MRPT2",["threedsrgmrpt2","3dsrgmrpt2"],Dsrg),
    DsrgMrPt3=>("DSRG-MRPT3",["dsrgmrpt3","mrdsrgpt3"],Dsrg),MrLdsrg2=>("MR-LDSRG(2)",["mrldsrg2","mrldsrg(2)"],Dsrg),
    MrLdsrg2Qc=>("MR-LDSRG(2)+QC",["mrldsrg2qc","mrldsrg(2)+qc"],Dsrg),
    SaDsrgPt2=>("SA-DSRG-PT2",["sadsrgpt2","sadsrg"],Dsrg),SaDsrgPt2C=>("SA-DSRG-PT2c",["sadsrgpt2c"],Dsrg),MsDsrgPt2=>("MS-DSRG-PT2",["msdsrgpt2"],Dsrg),
    XmsDsrgPt2=>("XMS-DSRG-PT2",["xmsdsrgpt2"],Dsrg),DwmsDsrgPt2=>("DWMS-DSRG-PT2",["dwmsdsrgpt2"],Dsrg)
}
impl std::fmt::Display for MethodId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
impl MethodId {
    pub fn option_schemas(self) -> Vec<&'static [&'static str]> {
        use MethodId::*;
        match self {
            Mp2 | ScsMp2 | Mp3 => vec![SCS_OPTIONS],
            ScsMp3 => vec![SCS_MP3_OPTIONS],
            Cis => vec![CIS_OPTIONS],
            Fci => vec![ACTIVE_OPTIONS, CAS_OPTIONS],
            Mrci | MrciQ => vec![ACTIVE_OPTIONS, CAS_OPTIONS, MRCI_OPTIONS],
            Casscf | Gasci | Gasscf | Avas | Aset => {
                vec![ACTIVE_OPTIONS, CAS_OPTIONS]
            }
            Caspt2 => vec![ACTIVE_OPTIONS, CAS_OPTIONS, CASPT2_OPTIONS],
            DlpnoMp2 | DlpnoCcsd | DlpnoCcsdT0 | DlpnoCcsdT1 => vec![LOCAL_OPTIONS],
            Mp2F12 | CcsdF12a | CcsdF12b | CcsdTF12a | CcsdTF12b => vec![F12_OPTIONS],
            Aci => vec![ACTIVE_OPTIONS, CAS_OPTIONS, SCI_OPTIONS, ACI_OPTIONS],
            Asci => vec![ACTIVE_OPTIONS, CAS_OPTIONS, SCI_OPTIONS, ASCI_OPTIONS],
            Hbci => vec![ACTIVE_OPTIONS, CAS_OPTIONS, SCI_OPTIONS, HCI_OPTIONS],
            Pci => vec![ACTIVE_OPTIONS, PCI_OPTIONS],
            Dmrg | DmrgScf => vec![ACTIVE_OPTIONS, CAS_OPTIONS, DMRG_OPTIONS],
            SrDsrgPt2 | SrDsrgPt3 => vec![SR_DSRG_OPTIONS],
            SrLdsrg2 => vec![LDSRG_OPTIONS],
            SrGode => vec![IMSRG_OPTIONS],
            DsrgMrPt2 | DsrgMrPt3 | MrLdsrg2 => vec![ACTIVE_OPTIONS, CAS_OPTIONS, MR_DSRG_OPTIONS],
            MrLdsrg2Qc => vec![ACTIVE_OPTIONS, CAS_OPTIONS, MR_QC_OPTIONS],
            ThreeDsrgMrPt2 => vec![ACTIVE_OPTIONS, CAS_OPTIONS, THREE_DSRG_OPTIONS],
            SaDsrgPt2 | SaDsrgPt2C | MsDsrgPt2 | XmsDsrgPt2 => {
                vec![ACTIVE_OPTIONS, CAS_OPTIONS, MULTISTATE_DSRG_OPTIONS]
            }
            DwmsDsrgPt2 => vec![
                ACTIVE_OPTIONS,
                CAS_OPTIONS,
                MULTISTATE_DSRG_OPTIONS,
                DWMS_DSRG_OPTIONS,
            ],
            _ => Vec::new(),
        }
    }
    /// This reports the implemented derivative route. P4 promotes additional
    /// entries only when their relaxed-density/response implementations are verified.
    pub fn gradient_kind(self, reference: RefKind, options: &PostHFOptions) -> GradientKind {
        use MethodId::*;
        match self {
            Rhf | Uhf => GradientKind::Analytic,
            Mp2 if options.frozen_core == 0
                && options.frozen_virtual == 0
                && !options.density_fitting =>
            {
                GradientKind::Analytic
            }
            DlpnoMp2 | DlpnoCcsd | DlpnoCcsdT0 | DlpnoCcsdT1 => GradientKind::Unsupported,
            Mp2F12 | CcsdF12a | CcsdF12b | CcsdTF12a | CcsdTF12b
                if options.f12.fitting_basis.is_some()
                    || options.f12.fock_fitting_basis.is_some() =>
            {
                GradientKind::Unsupported
            }
            Rohf | Qrhf | Mom | Fermi | Ghf => GradientKind::Unsupported,
            DsrgMrPt2 | DsrgMrPt3 | MrLdsrg2 | MrLdsrg2Qc
                if reference == RefKind::Uhf
                    || options.density_fitting
                    || options.frozen_core != 0
                    || options.frozen_virtual != 0
                    || options.n_roots != 1
                    || options.root != 0
                    || options
                        .cas
                        .target_s_squared
                        .is_some_and(|x| x.abs() > 1e-10) =>
            {
                GradientKind::Unsupported
            }
            ThreeDsrgMrPt2
                if reference == RefKind::Uhf
                    || options.frozen_core != 0
                    || options.frozen_virtual != 0
                    || options.n_roots != 1
                    || options.root != 0
                    || (options.density_fitting
                        && options.cas.optimize_orbitals
                        && options.active.is_some())
                    || options
                        .cas
                        .target_s_squared
                        .is_some_and(|x| x.abs() > 1e-10) =>
            {
                GradientKind::Unsupported
            }
            SaDsrgPt2 | SaDsrgPt2C | MsDsrgPt2 | XmsDsrgPt2 | DwmsDsrgPt2
                if reference == RefKind::Uhf
                    || options.frozen_core != 0
                    || options.frozen_virtual != 0
                    || (options.density_fitting
                        && options.cas.optimize_orbitals
                        && options.active.is_some())
                    || options.dsrg.relaxation() != crate::options::ReferenceRelaxation::None
                    || options
                        .cas
                        .target_s_squared
                        .is_some_and(|x| x.abs() > 1e-10) =>
            {
                GradientKind::Unsupported
            }
            XmsDsrgPt2
                if !options.cas.state_weights.is_empty()
                    && options.cas.state_weights.iter().any(|w| {
                        (*w / options.cas.state_weights.iter().sum::<f64>()
                            - 1. / options.n_roots as f64)
                            .abs()
                            > 1e-10
                    }) =>
            {
                GradientKind::Unsupported
            }
            _ if reference == RefKind::Uhf
                && matches!(self, Fci | Casscf | Caspt2 | Mrci | MrciQ | Dmrg | DmrgScf) =>
            {
                GradientKind::Unsupported
            }
            _ => GradientKind::Numerical,
        }
    }
}

pub trait MethodMeta {
    const METHOD: MethodId;
}
pub trait ContextMethod<'a>: PostHFMethod + MethodMeta + Sized {
    fn from_context(ctx: &'a PostHFContext<'a>, options: &PostHFOptions) -> Result<Self, HfError>;
}
pub trait PostHFSolver {
    fn compute(&self) -> Result<PostHFOutput, HfError>;
    fn compute_with_intermediates(
        &self,
        max_elements: usize,
    ) -> Result<(PostHFOutput, crate::output::TensorMap), HfError> {
        let tensors = crate::output::TensorExport::new(max_elements)?.finish();
        Ok((self.compute()?, tensors))
    }
}

/// Construct a solver from a typed method. Additional method adapters are wired
/// here as their context migrations are completed; unsupported combinations fail.
pub fn create_solver<'a>(
    method: MethodId,
    ctx: &'a PostHFContext<'a>,
    options: &PostHFOptions,
) -> Result<Box<dyn PostHFSolver + 'a>, HfError> {
    options.validate()?;
    if !matches!(
        method,
        MethodId::Fci
            | MethodId::Cis
            | MethodId::Casscf
            | MethodId::Mrci
            | MethodId::MrciQ
            | MethodId::Cisd
            | MethodId::Aci
            | MethodId::Asci
            | MethodId::Hbci
            | MethodId::SaDsrgPt2
            | MethodId::SaDsrgPt2C
            | MethodId::MsDsrgPt2
            | MethodId::XmsDsrgPt2
            | MethodId::DwmsDsrgPt2
    ) && (options.root != 0 || options.n_roots != 1)
    {
        return Err(HfError::Unsupported {
            method: method.name().into(),
            detail: "this adapter computes one ground state".into(),
        });
    }
    use crate::post_hf::mp2::{RHFMP2, UHFMP2};
    use crate::post_hf::{CCD, CCSD, UHFCCSD};
    let e_scf = ctx.reference.energy();
    macro_rules! entry {
        ($solver:expr) => {
            Ok(Box::new(Entry {
                engine: $solver,
                e_scf,
                root: options.root,
            }) as Box<dyn PostHFSolver + 'a>)
        };
    }
    match method {
        MethodId::Cis => entry!(crate::post_hf::ci::CIS::from_context(ctx, options)?),
        MethodId::ScsMp3 => entry!(crate::post_hf::scs_mp3::ScsMp3::from_context(ctx, options)?),
        MethodId::SrDsrgPt2 => entry!(crate::post_hf::mr_dsrg::dsrg::TypedDsrgPt2::from_context(
            ctx, options
        )?),
        MethodId::SrDsrgPt3 => entry!(crate::post_hf::mr_dsrg::dsrg::SRDSRGPT3::from_context(
            ctx, options
        )?),
        MethodId::SrLdsrg2 => entry!(crate::post_hf::mr_dsrg::dsrg::SRLDSRG2::from_context(
            ctx, options
        )?),
        MethodId::DsrgMrPt2 => entry!(
            crate::post_hf::mr_dsrg::typed_mr::MrDrivenMethod::<2>::from_context(ctx, options)?
        ),
        MethodId::ThreeDsrgMrPt2 => {
            entry!(crate::post_hf::mr_dsrg::typed_mr::ThreeIndexMrpt2::from_context(ctx, options)?)
        }
        MethodId::SaDsrgPt2 => entry!(
            crate::post_hf::mr_dsrg::multistate::MultistateMethod::<0>::from_context(ctx, options)?
        ),
        MethodId::SaDsrgPt2C => entry!(
            crate::post_hf::mr_dsrg::multistate::MultistateMethod::<1>::from_context(ctx, options)?
        ),
        MethodId::MsDsrgPt2 => entry!(
            crate::post_hf::mr_dsrg::multistate::MultistateMethod::<2>::from_context(ctx, options)?
        ),
        MethodId::XmsDsrgPt2 => entry!(
            crate::post_hf::mr_dsrg::multistate::MultistateMethod::<3>::from_context(ctx, options)?
        ),
        MethodId::DwmsDsrgPt2 => entry!(
            crate::post_hf::mr_dsrg::multistate::MultistateMethod::<4>::from_context(ctx, options)?
        ),
        MethodId::DsrgMrPt3 => entry!(
            crate::post_hf::mr_dsrg::typed_mr::MrDrivenMethod::<3>::from_context(ctx, options)?
        ),
        MethodId::MrLdsrg2 => entry!(
            crate::post_hf::mr_dsrg::typed_mr::MrDrivenMethod::<4>::from_context(ctx, options)?
        ),
        MethodId::MrLdsrg2Qc => entry!(
            crate::post_hf::mr_dsrg::typed_mr::MrDrivenMethod::<5>::from_context(ctx, options)?
        ),
        MethodId::SrGode => entry!(crate::post_hf::mr_dsrg::dsrg::SRGODE::from_context(
            ctx, options
        )?),
        MethodId::DlpnoCcsdT0 => entry!(crate::post_hf::dlpno::LocalCcsdT0::from_context(
            ctx, options
        )?),
        MethodId::DlpnoCcsdT1 => entry!(crate::post_hf::dlpno::LocalCcsdT1::from_context(
            ctx, options
        )?),
        MethodId::DlpnoCcsd => entry!(crate::post_hf::dlpno::LocalCcsd::from_context(
            ctx, options
        )?),
        MethodId::DlpnoMp2 => entry!(crate::post_hf::dlpno::LocalMp2::from_context(ctx, options)?),
        MethodId::Pci => {
            entry!(crate::post_hf::mr_dsrg::projector_ci::ProjectorCi::from_context(ctx, options)?)
        }
        MethodId::Aci => {
            entry!(crate::post_hf::mr_dsrg::selected_ci::AdaptiveCi::from_context(ctx, options)?)
        }
        MethodId::Asci => entry!(
            crate::post_hf::mr_dsrg::selected_ci::AdaptiveSamplingCi::from_context(ctx, options)?
        ),
        MethodId::Hbci => {
            entry!(crate::post_hf::mr_dsrg::selected_ci::HeatBathCi::from_context(ctx, options)?)
        }
        MethodId::Mrci => entry!(crate::post_hf::ci::mrci_engine::Mrci::from_context(
            ctx, options
        )?),
        MethodId::MrciQ => entry!(crate::post_hf::ci::mrci_engine::MrciQ::from_context(
            ctx, options
        )?),
        MethodId::Cisd => {
            entry!(crate::post_hf::ci::mrci_engine::SingleReferenceCi::from_context(ctx, options)?)
        }
        MethodId::Fci => entry!(crate::post_hf::casscf_engine::Casci::from_context(
            ctx, options
        )?),
        MethodId::Casscf => {
            entry!(crate::post_hf::casscf_engine::OrbitalOptimizedCas::from_context(ctx, options)?)
        }
        MethodId::Dmrg => {
            entry!(crate::post_hf::dmrg_orbitals::DmrgMethod::<false>::from_context(ctx, options)?)
        }
        MethodId::Caspt2 => entry!(crate::post_hf::caspt2_typed::Caspt2::from_context(
            ctx, options
        )?),
        MethodId::DmrgScf => {
            entry!(crate::post_hf::dmrg_orbitals::DmrgMethod::<true>::from_context(ctx, options)?)
        }
        MethodId::Mp2 if options.density_fitting => {
            entry!(crate::post_hf::ri_mp2::RiMp2::from_context(ctx, options)?)
        }
        MethodId::Mp2 => match ctx.reference.kind() {
            RefKind::Rhf => entry!(RHFMP2::from_context(ctx, options)?),
            RefKind::Uhf => entry!(UHFMP2::from_context(ctx, options)?),
        },
        MethodId::RiMp2 => entry!(crate::post_hf::ri_mp2::RiMp2::from_context(ctx, options)?),
        MethodId::Mp2F12 => entry!(crate::post_hf::f12::RHFMP2F12::from_context(ctx, options)?),
        MethodId::CcsdF12a => entry!(crate::post_hf::f12::typed_ccsd::CcsdF12Method::<
            false,
            false,
        >::from_context(ctx, options)?),
        MethodId::CcsdF12b => entry!(
            crate::post_hf::f12::typed_ccsd::CcsdF12Method::<true, false>::from_context(
                ctx, options
            )?
        ),
        MethodId::CcsdTF12a => entry!(
            crate::post_hf::f12::typed_ccsd::CcsdF12Method::<false, true>::from_context(
                ctx, options
            )?
        ),
        MethodId::CcsdTF12b => entry!(
            crate::post_hf::f12::typed_ccsd::CcsdF12Method::<true, true>::from_context(
                ctx, options
            )?
        ),
        MethodId::ScsMp2 => entry!(crate::post_hf::scs_mp2::ScsMp2::from_context(ctx, options)?),
        MethodId::Mp3 => match ctx.reference.kind() {
            RefKind::Rhf => entry!(crate::post_hf::RHFMP3::from_context(ctx, options)?),
            RefKind::Uhf => entry!(crate::post_hf::UHFMP3::from_context(ctx, options)?),
        },
        MethodId::Mp4 => entry!(crate::post_hf::RHFMP4::from_context(ctx, options)?),
        MethodId::Mp5 => entry!(crate::post_hf::RHFMP5::from_context(ctx, options)?),
        MethodId::Mp6 => entry!(crate::post_hf::RHFMP6::from_context(ctx, options)?),
        MethodId::Mp7 => entry!(crate::post_hf::RHFMP7::from_context(ctx, options)?),
        MethodId::Mp8 => entry!(crate::post_hf::RHFMP8::from_context(ctx, options)?),
        MethodId::Doci => entry!(crate::post_hf::mr_dsrg::DOCI::from_context(ctx, options)?),
        MethodId::Ccd => entry!(CCD::from_context(ctx, options)?),
        MethodId::CcsdT => {
            entry!(crate::post_hf::cc::typed_triples::CcsdTriples::from_context(ctx, options)?)
        }
        MethodId::Ccsd => match ctx.reference.kind() {
            RefKind::Rhf => entry!(CCSD::from_context(ctx, options)?),
            RefKind::Uhf => entry!(UHFCCSD::from_context(ctx, options)?),
        },
        _ => Err(HfError::Unsupported {
            method: method.name().into(),
            detail: "typed context adapter is not yet connected".into(),
        }),
    }
}

pub struct Entry<T> {
    pub engine: T,
    pub e_scf: f64,
    pub root: usize,
}
impl<T: PostHFMethod + MethodMeta> Entry<T> {
    fn compute_inner(
        &self,
        limit: Option<usize>,
    ) -> Result<(PostHFOutput, crate::output::TensorMap), HfError> {
        if let Some(limit) = limit {
            crate::output::TensorExport::new(limit)?;
        }
        let result = self.engine.compute().map_err(|detail| HfError::PostHf {
            method: T::METHOD.name().into(),
            detail,
        })?;
        if !result.converged() {
            return Err(HfError::PostHf {
                method: T::METHOD.name().into(),
                detail: "amplitudes or state did not converge".into(),
            });
        }
        let mut output = result.to_output()?;
        output.method = T::METHOD.name().into();
        output.e_scf = self.e_scf;
        output.e_corr = output.e_total - self.e_scf;
        if matches!(T::METHOD, MethodId::Casscf) {
            output
                .components
                .insert("E_CASSCF".into(), result.total_energy());
        }
        if matches!(T::METHOD, MethodId::Fci) {
            output
                .components
                .insert("E_CASCI".into(), result.total_energy());
        }
        output.select_root(self.root)?;
        output.validate()?;
        let mut tensors = if let Some(limit) = limit {
            result.intermediates(limit)?
        } else {
            crate::output::TensorMap::new()
        };
        if let (Some(limit), Some(rdm)) = (limit, output.one_rdm.as_ref()) {
            let used: usize = tensors.values().map(|x| x.values.len()).sum();
            let remaining = limit.checked_sub(used).ok_or_else(|| {
                HfError::InvalidInput("intermediate element count exceeds max_elements".into())
            })?;
            tensors.insert(
                "method.one_rdm_mo".into(),
                crate::output::TensorData::from_matrix(
                    rdm,
                    crate::output::ObservableUnit::One,
                    remaining,
                )?,
            );
        }
        Ok((output, tensors))
    }
}
impl<T: PostHFMethod + MethodMeta> PostHFSolver for Entry<T> {
    fn compute(&self) -> Result<PostHFOutput, HfError> {
        Ok(self.compute_inner(None)?.0)
    }
    fn compute_with_intermediates(
        &self,
        max_elements: usize,
    ) -> Result<(PostHFOutput, crate::output::TensorMap), HfError> {
        self.compute_inner(Some(max_elements))
    }
}
