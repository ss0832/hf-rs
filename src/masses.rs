// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
//
// This code is part of the hf-rs library.

//! Recommended CODATA and IUPAC/CIAAW atomic masses, standard weights, and physical constants.
//! Covers all 118 known chemical elements from Hydrogen (Z=1) to Oganesson (Z=118).
//!
//! Numerical values are taken directly from:
//! - NIST Physical Measurement Laboratory: Atomic Weights and Isotopic Compositions
//! - CODATA Recommended Values of the Fundamental Physical Constants
//! - IUPAC Commission on Isotopic Abundances and Atomic Weights (CIAAW)

/// Unified atomic mass unit in kilograms (CODATA): m_u = 1.66053906892e-27 kg
pub const AMU_TO_KG: f64 = 1.660_539_068_92e-27;

/// Electron mass in kilograms (CODATA): m_e = 9.1093837139e-31 kg
pub const ELECTRON_MASS_KG: f64 = 9.109_383_713_9e-31;

/// Conversion factor from unified atomic mass unit (amu or Da) to atomic unit of mass (m_e, CODATA):
/// m_u / m_e = 1822.888486209
pub const AMU_TO_AU: f64 = 1_822.888_486_209;

/// Conversion factor from atomic unit of mass (m_e) to unified atomic mass unit (amu)
pub const AU_TO_AMU: f64 = 1.0 / AMU_TO_AU;

/// Bohr radius in Angstroms (CODATA): a_0 = 0.529177210903 Angstrom
pub const BOHR_TO_ANGSTROM: f64 = 0.529_177_210_903;

/// Conversion factor from Angstroms to Bohr
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / BOHR_TO_ANGSTROM;

/// Hartree energy in electron-volts (CODATA): E_h = 27.211386245988 eV
pub const HARTREE_TO_EV: f64 = 27.211_386_245_988;

/// Conversion factor from eV to Hartree
pub const EV_TO_HARTREE: f64 = 1.0 / HARTREE_TO_EV;

/// Hartree energy in Joules (CODATA): E_h = 4.3597447222071e-18 J
pub const HARTREE_TO_J: f64 = 4.359_744_722_207_1e-18;

/// Speed of light in atomic units: c / (alpha * c) = alpha^-1 = 137.035999084
pub const SPEED_OF_LIGHT_AU: f64 = 137.035_999_084;

/// Data structure holding physical properties of a chemical element.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementData {
    /// Atomic number Z (1 to 118)
    pub atomic_number: usize,
    /// Standard IUPAC 1- or 2-letter symbol
    pub symbol: &'static str,
    /// English name of the element
    pub name: &'static str,
    /// Mass number A of the primary (most abundant or longest-lived) isotope
    pub primary_mass_number: usize,
    /// Relative atomic mass of the primary isotope in unified atomic mass units (amu / Da)
    pub mass_amu: f64,
    /// IUPAC standard atomic weight (conventional average atomic weight)
    pub standard_atomic_weight: f64,
}

/// Complete table of all 118 chemical elements from Z=1 (Hydrogen) to Z=118 (Oganesson).
pub static ELEMENTS: [ElementData; 118] = [
    ElementData {
        atomic_number: 1,
        symbol: "H",
        name: "Hydrogen",
        primary_mass_number: 1,
        mass_amu: 1.00782503223,
        standard_atomic_weight: 1.00797500,
    },
    ElementData {
        atomic_number: 2,
        symbol: "He",
        name: "Helium",
        primary_mass_number: 4,
        mass_amu: 4.00260325413,
        standard_atomic_weight: 4.00260200,
    },
    ElementData {
        atomic_number: 3,
        symbol: "Li",
        name: "Lithium",
        primary_mass_number: 7,
        mass_amu: 7.01600343660,
        standard_atomic_weight: 6.96750000,
    },
    ElementData {
        atomic_number: 4,
        symbol: "Be",
        name: "Beryllium",
        primary_mass_number: 9,
        mass_amu: 9.01218306500,
        standard_atomic_weight: 9.01218310,
    },
    ElementData {
        atomic_number: 5,
        symbol: "B",
        name: "Boron",
        primary_mass_number: 11,
        mass_amu: 11.00930536000,
        standard_atomic_weight: 10.81350000,
    },
    ElementData {
        atomic_number: 6,
        symbol: "C",
        name: "Carbon",
        primary_mass_number: 12,
        mass_amu: 12.00000000000,
        standard_atomic_weight: 12.01060000,
    },
    ElementData {
        atomic_number: 7,
        symbol: "N",
        name: "Nitrogen",
        primary_mass_number: 14,
        mass_amu: 14.00307400443,
        standard_atomic_weight: 14.00685500,
    },
    ElementData {
        atomic_number: 8,
        symbol: "O",
        name: "Oxygen",
        primary_mass_number: 16,
        mass_amu: 15.99491461957,
        standard_atomic_weight: 15.99940000,
    },
    ElementData {
        atomic_number: 9,
        symbol: "F",
        name: "Fluorine",
        primary_mass_number: 19,
        mass_amu: 18.99840316273,
        standard_atomic_weight: 18.99840316,
    },
    ElementData {
        atomic_number: 10,
        symbol: "Ne",
        name: "Neon",
        primary_mass_number: 20,
        mass_amu: 19.99244017620,
        standard_atomic_weight: 20.17970000,
    },
    ElementData {
        atomic_number: 11,
        symbol: "Na",
        name: "Sodium",
        primary_mass_number: 23,
        mass_amu: 22.98976928200,
        standard_atomic_weight: 22.98976928,
    },
    ElementData {
        atomic_number: 12,
        symbol: "Mg",
        name: "Magnesium",
        primary_mass_number: 24,
        mass_amu: 23.98504169700,
        standard_atomic_weight: 24.30550000,
    },
    ElementData {
        atomic_number: 13,
        symbol: "Al",
        name: "Aluminium",
        primary_mass_number: 27,
        mass_amu: 26.98153853000,
        standard_atomic_weight: 26.98153850,
    },
    ElementData {
        atomic_number: 14,
        symbol: "Si",
        name: "Silicon",
        primary_mass_number: 28,
        mass_amu: 27.97692653465,
        standard_atomic_weight: 28.08500000,
    },
    ElementData {
        atomic_number: 15,
        symbol: "P",
        name: "Phosphorus",
        primary_mass_number: 31,
        mass_amu: 30.97376199842,
        standard_atomic_weight: 30.97376200,
    },
    ElementData {
        atomic_number: 16,
        symbol: "S",
        name: "Sulfur",
        primary_mass_number: 32,
        mass_amu: 31.97207117440,
        standard_atomic_weight: 32.06750000,
    },
    ElementData {
        atomic_number: 17,
        symbol: "Cl",
        name: "Chlorine",
        primary_mass_number: 35,
        mass_amu: 34.96885268200,
        standard_atomic_weight: 35.45150000,
    },
    ElementData {
        atomic_number: 18,
        symbol: "Ar",
        name: "Argon",
        primary_mass_number: 40,
        mass_amu: 39.96238312370,
        standard_atomic_weight: 39.94800000,
    },
    ElementData {
        atomic_number: 19,
        symbol: "K",
        name: "Potassium",
        primary_mass_number: 39,
        mass_amu: 38.96370648640,
        standard_atomic_weight: 39.09830000,
    },
    ElementData {
        atomic_number: 20,
        symbol: "Ca",
        name: "Calcium",
        primary_mass_number: 40,
        mass_amu: 39.96259086300,
        standard_atomic_weight: 40.07800000,
    },
    ElementData {
        atomic_number: 21,
        symbol: "Sc",
        name: "Scandium",
        primary_mass_number: 45,
        mass_amu: 44.95590828000,
        standard_atomic_weight: 44.95590800,
    },
    ElementData {
        atomic_number: 22,
        symbol: "Ti",
        name: "Titanium",
        primary_mass_number: 48,
        mass_amu: 47.94794198000,
        standard_atomic_weight: 47.86700000,
    },
    ElementData {
        atomic_number: 23,
        symbol: "V",
        name: "Vanadium",
        primary_mass_number: 51,
        mass_amu: 50.94395704000,
        standard_atomic_weight: 50.94150000,
    },
    ElementData {
        atomic_number: 24,
        symbol: "Cr",
        name: "Chromium",
        primary_mass_number: 52,
        mass_amu: 51.94050623000,
        standard_atomic_weight: 51.99610000,
    },
    ElementData {
        atomic_number: 25,
        symbol: "Mn",
        name: "Manganese",
        primary_mass_number: 55,
        mass_amu: 54.93804391000,
        standard_atomic_weight: 54.93804400,
    },
    ElementData {
        atomic_number: 26,
        symbol: "Fe",
        name: "Iron",
        primary_mass_number: 56,
        mass_amu: 55.93493633000,
        standard_atomic_weight: 55.84500000,
    },
    ElementData {
        atomic_number: 27,
        symbol: "Co",
        name: "Cobalt",
        primary_mass_number: 59,
        mass_amu: 58.93319429000,
        standard_atomic_weight: 58.93319400,
    },
    ElementData {
        atomic_number: 28,
        symbol: "Ni",
        name: "Nickel",
        primary_mass_number: 58,
        mass_amu: 57.93534241000,
        standard_atomic_weight: 58.69340000,
    },
    ElementData {
        atomic_number: 29,
        symbol: "Cu",
        name: "Copper",
        primary_mass_number: 63,
        mass_amu: 62.92959772000,
        standard_atomic_weight: 63.54600000,
    },
    ElementData {
        atomic_number: 30,
        symbol: "Zn",
        name: "Zinc",
        primary_mass_number: 64,
        mass_amu: 63.92914201000,
        standard_atomic_weight: 65.38000000,
    },
    ElementData {
        atomic_number: 31,
        symbol: "Ga",
        name: "Gallium",
        primary_mass_number: 69,
        mass_amu: 68.92557350000,
        standard_atomic_weight: 69.72300000,
    },
    ElementData {
        atomic_number: 32,
        symbol: "Ge",
        name: "Germanium",
        primary_mass_number: 74,
        mass_amu: 73.92117776100,
        standard_atomic_weight: 72.63000000,
    },
    ElementData {
        atomic_number: 33,
        symbol: "As",
        name: "Arsenic",
        primary_mass_number: 75,
        mass_amu: 74.92159457000,
        standard_atomic_weight: 74.92159500,
    },
    ElementData {
        atomic_number: 34,
        symbol: "Se",
        name: "Selenium",
        primary_mass_number: 80,
        mass_amu: 79.91652180000,
        standard_atomic_weight: 78.97100000,
    },
    ElementData {
        atomic_number: 35,
        symbol: "Br",
        name: "Bromine",
        primary_mass_number: 79,
        mass_amu: 78.91833760000,
        standard_atomic_weight: 79.90400000,
    },
    ElementData {
        atomic_number: 36,
        symbol: "Kr",
        name: "Krypton",
        primary_mass_number: 84,
        mass_amu: 83.91149772820,
        standard_atomic_weight: 83.79800000,
    },
    ElementData {
        atomic_number: 37,
        symbol: "Rb",
        name: "Rubidium",
        primary_mass_number: 85,
        mass_amu: 84.91178973790,
        standard_atomic_weight: 85.46780000,
    },
    ElementData {
        atomic_number: 38,
        symbol: "Sr",
        name: "Strontium",
        primary_mass_number: 88,
        mass_amu: 87.90561250000,
        standard_atomic_weight: 87.62000000,
    },
    ElementData {
        atomic_number: 39,
        symbol: "Y",
        name: "Yttrium",
        primary_mass_number: 89,
        mass_amu: 88.90584030000,
        standard_atomic_weight: 88.90584000,
    },
    ElementData {
        atomic_number: 40,
        symbol: "Zr",
        name: "Zirconium",
        primary_mass_number: 90,
        mass_amu: 89.90469770000,
        standard_atomic_weight: 91.22400000,
    },
    ElementData {
        atomic_number: 41,
        symbol: "Nb",
        name: "Niobium",
        primary_mass_number: 93,
        mass_amu: 92.90637300000,
        standard_atomic_weight: 92.90637000,
    },
    ElementData {
        atomic_number: 42,
        symbol: "Mo",
        name: "Molybdenum",
        primary_mass_number: 98,
        mass_amu: 97.90540482000,
        standard_atomic_weight: 95.95000000,
    },
    ElementData {
        atomic_number: 43,
        symbol: "Tc",
        name: "Technetium",
        primary_mass_number: 98,
        mass_amu: 97.90721240000,
        standard_atomic_weight: 98.00000000,
    },
    ElementData {
        atomic_number: 44,
        symbol: "Ru",
        name: "Ruthenium",
        primary_mass_number: 102,
        mass_amu: 101.90434410000,
        standard_atomic_weight: 101.07000000,
    },
    ElementData {
        atomic_number: 45,
        symbol: "Rh",
        name: "Rhodium",
        primary_mass_number: 103,
        mass_amu: 102.90549800000,
        standard_atomic_weight: 102.90550000,
    },
    ElementData {
        atomic_number: 46,
        symbol: "Pd",
        name: "Palladium",
        primary_mass_number: 106,
        mass_amu: 105.90348040000,
        standard_atomic_weight: 106.42000000,
    },
    ElementData {
        atomic_number: 47,
        symbol: "Ag",
        name: "Silver",
        primary_mass_number: 107,
        mass_amu: 106.90509160000,
        standard_atomic_weight: 107.86820000,
    },
    ElementData {
        atomic_number: 48,
        symbol: "Cd",
        name: "Cadmium",
        primary_mass_number: 114,
        mass_amu: 113.90336509000,
        standard_atomic_weight: 112.41400000,
    },
    ElementData {
        atomic_number: 49,
        symbol: "In",
        name: "Indium",
        primary_mass_number: 115,
        mass_amu: 114.90387877600,
        standard_atomic_weight: 114.81800000,
    },
    ElementData {
        atomic_number: 50,
        symbol: "Sn",
        name: "Tin",
        primary_mass_number: 120,
        mass_amu: 119.90220163000,
        standard_atomic_weight: 118.71000000,
    },
    ElementData {
        atomic_number: 51,
        symbol: "Sb",
        name: "Antimony",
        primary_mass_number: 121,
        mass_amu: 120.90381200000,
        standard_atomic_weight: 121.76000000,
    },
    ElementData {
        atomic_number: 52,
        symbol: "Te",
        name: "Tellurium",
        primary_mass_number: 130,
        mass_amu: 129.90622274800,
        standard_atomic_weight: 127.60000000,
    },
    ElementData {
        atomic_number: 53,
        symbol: "I",
        name: "Iodine",
        primary_mass_number: 127,
        mass_amu: 126.90447190000,
        standard_atomic_weight: 126.90447000,
    },
    ElementData {
        atomic_number: 54,
        symbol: "Xe",
        name: "Xenon",
        primary_mass_number: 132,
        mass_amu: 131.90415508560,
        standard_atomic_weight: 131.29300000,
    },
    ElementData {
        atomic_number: 55,
        symbol: "Cs",
        name: "Caesium",
        primary_mass_number: 133,
        mass_amu: 132.90545196100,
        standard_atomic_weight: 132.90545196,
    },
    ElementData {
        atomic_number: 56,
        symbol: "Ba",
        name: "Barium",
        primary_mass_number: 138,
        mass_amu: 137.90524700000,
        standard_atomic_weight: 137.32700000,
    },
    ElementData {
        atomic_number: 57,
        symbol: "La",
        name: "Lanthanum",
        primary_mass_number: 139,
        mass_amu: 138.90635630000,
        standard_atomic_weight: 138.90547000,
    },
    ElementData {
        atomic_number: 58,
        symbol: "Ce",
        name: "Cerium",
        primary_mass_number: 140,
        mass_amu: 139.90544310000,
        standard_atomic_weight: 140.11600000,
    },
    ElementData {
        atomic_number: 59,
        symbol: "Pr",
        name: "Praseodymium",
        primary_mass_number: 141,
        mass_amu: 140.90765760000,
        standard_atomic_weight: 140.90766000,
    },
    ElementData {
        atomic_number: 60,
        symbol: "Nd",
        name: "Neodymium",
        primary_mass_number: 142,
        mass_amu: 141.90772900000,
        standard_atomic_weight: 144.24200000,
    },
    ElementData {
        atomic_number: 61,
        symbol: "Pm",
        name: "Promethium",
        primary_mass_number: 145,
        mass_amu: 144.91275590000,
        standard_atomic_weight: 145.00000000,
    },
    ElementData {
        atomic_number: 62,
        symbol: "Sm",
        name: "Samarium",
        primary_mass_number: 152,
        mass_amu: 151.91973970000,
        standard_atomic_weight: 150.36000000,
    },
    ElementData {
        atomic_number: 63,
        symbol: "Eu",
        name: "Europium",
        primary_mass_number: 153,
        mass_amu: 152.92123800000,
        standard_atomic_weight: 151.96400000,
    },
    ElementData {
        atomic_number: 64,
        symbol: "Gd",
        name: "Gadolinium",
        primary_mass_number: 158,
        mass_amu: 157.92411230000,
        standard_atomic_weight: 157.25000000,
    },
    ElementData {
        atomic_number: 65,
        symbol: "Tb",
        name: "Terbium",
        primary_mass_number: 159,
        mass_amu: 158.92535470000,
        standard_atomic_weight: 158.92535000,
    },
    ElementData {
        atomic_number: 66,
        symbol: "Dy",
        name: "Dysprosium",
        primary_mass_number: 164,
        mass_amu: 163.92918190000,
        standard_atomic_weight: 162.50000000,
    },
    ElementData {
        atomic_number: 67,
        symbol: "Ho",
        name: "Holmium",
        primary_mass_number: 165,
        mass_amu: 164.93032880000,
        standard_atomic_weight: 164.93033000,
    },
    ElementData {
        atomic_number: 68,
        symbol: "Er",
        name: "Erbium",
        primary_mass_number: 166,
        mass_amu: 165.93029950000,
        standard_atomic_weight: 167.25900000,
    },
    ElementData {
        atomic_number: 69,
        symbol: "Tm",
        name: "Thulium",
        primary_mass_number: 169,
        mass_amu: 168.93421790000,
        standard_atomic_weight: 168.93422000,
    },
    ElementData {
        atomic_number: 70,
        symbol: "Yb",
        name: "Ytterbium",
        primary_mass_number: 174,
        mass_amu: 173.93886640000,
        standard_atomic_weight: 173.05400000,
    },
    ElementData {
        atomic_number: 71,
        symbol: "Lu",
        name: "Lutetium",
        primary_mass_number: 175,
        mass_amu: 174.94077520000,
        standard_atomic_weight: 174.96680000,
    },
    ElementData {
        atomic_number: 72,
        symbol: "Hf",
        name: "Hafnium",
        primary_mass_number: 180,
        mass_amu: 179.94655700000,
        standard_atomic_weight: 178.49000000,
    },
    ElementData {
        atomic_number: 73,
        symbol: "Ta",
        name: "Tantalum",
        primary_mass_number: 181,
        mass_amu: 180.94799580000,
        standard_atomic_weight: 180.94788000,
    },
    ElementData {
        atomic_number: 74,
        symbol: "W",
        name: "Tungsten",
        primary_mass_number: 184,
        mass_amu: 183.95093092000,
        standard_atomic_weight: 183.84000000,
    },
    ElementData {
        atomic_number: 75,
        symbol: "Re",
        name: "Rhenium",
        primary_mass_number: 187,
        mass_amu: 186.95575010000,
        standard_atomic_weight: 186.20700000,
    },
    ElementData {
        atomic_number: 76,
        symbol: "Os",
        name: "Osmium",
        primary_mass_number: 192,
        mass_amu: 191.96147700000,
        standard_atomic_weight: 190.23000000,
    },
    ElementData {
        atomic_number: 77,
        symbol: "Ir",
        name: "Iridium",
        primary_mass_number: 193,
        mass_amu: 192.96292160000,
        standard_atomic_weight: 192.21700000,
    },
    ElementData {
        atomic_number: 78,
        symbol: "Pt",
        name: "Platinum",
        primary_mass_number: 195,
        mass_amu: 194.96479170000,
        standard_atomic_weight: 195.08400000,
    },
    ElementData {
        atomic_number: 79,
        symbol: "Au",
        name: "Gold",
        primary_mass_number: 197,
        mass_amu: 196.96656879000,
        standard_atomic_weight: 196.96656900,
    },
    ElementData {
        atomic_number: 80,
        symbol: "Hg",
        name: "Mercury",
        primary_mass_number: 202,
        mass_amu: 201.97064340000,
        standard_atomic_weight: 200.59200000,
    },
    ElementData {
        atomic_number: 81,
        symbol: "Tl",
        name: "Thallium",
        primary_mass_number: 205,
        mass_amu: 204.97442780000,
        standard_atomic_weight: 204.38350000,
    },
    ElementData {
        atomic_number: 82,
        symbol: "Pb",
        name: "Lead",
        primary_mass_number: 208,
        mass_amu: 207.97665250000,
        standard_atomic_weight: 207.20000000,
    },
    ElementData {
        atomic_number: 83,
        symbol: "Bi",
        name: "Bismuth",
        primary_mass_number: 209,
        mass_amu: 208.98039910000,
        standard_atomic_weight: 208.98040000,
    },
    ElementData {
        atomic_number: 84,
        symbol: "Po",
        name: "Polonium",
        primary_mass_number: 209,
        mass_amu: 208.98243080000,
        standard_atomic_weight: 209.00000000,
    },
    ElementData {
        atomic_number: 85,
        symbol: "At",
        name: "Astatine",
        primary_mass_number: 210,
        mass_amu: 209.98714790000,
        standard_atomic_weight: 210.00000000,
    },
    ElementData {
        atomic_number: 86,
        symbol: "Rn",
        name: "Radon",
        primary_mass_number: 222,
        mass_amu: 222.01757820000,
        standard_atomic_weight: 222.00000000,
    },
    ElementData {
        atomic_number: 87,
        symbol: "Fr",
        name: "Francium",
        primary_mass_number: 223,
        mass_amu: 223.01973600000,
        standard_atomic_weight: 223.00000000,
    },
    ElementData {
        atomic_number: 88,
        symbol: "Ra",
        name: "Radium",
        primary_mass_number: 226,
        mass_amu: 226.02541030000,
        standard_atomic_weight: 226.00000000,
    },
    ElementData {
        atomic_number: 89,
        symbol: "Ac",
        name: "Actinium",
        primary_mass_number: 227,
        mass_amu: 227.02775230000,
        standard_atomic_weight: 227.00000000,
    },
    ElementData {
        atomic_number: 90,
        symbol: "Th",
        name: "Thorium",
        primary_mass_number: 232,
        mass_amu: 232.03805580000,
        standard_atomic_weight: 232.03770000,
    },
    ElementData {
        atomic_number: 91,
        symbol: "Pa",
        name: "Protactinium",
        primary_mass_number: 231,
        mass_amu: 231.03588420000,
        standard_atomic_weight: 231.03588000,
    },
    ElementData {
        atomic_number: 92,
        symbol: "U",
        name: "Uranium",
        primary_mass_number: 238,
        mass_amu: 238.05078840000,
        standard_atomic_weight: 238.02891000,
    },
    ElementData {
        atomic_number: 93,
        symbol: "Np",
        name: "Neptunium",
        primary_mass_number: 237,
        mass_amu: 237.04817360000,
        standard_atomic_weight: 237.00000000,
    },
    ElementData {
        atomic_number: 94,
        symbol: "Pu",
        name: "Plutonium",
        primary_mass_number: 244,
        mass_amu: 244.06420530000,
        standard_atomic_weight: 244.00000000,
    },
    ElementData {
        atomic_number: 95,
        symbol: "Am",
        name: "Americium",
        primary_mass_number: 243,
        mass_amu: 243.06138130000,
        standard_atomic_weight: 243.00000000,
    },
    ElementData {
        atomic_number: 96,
        symbol: "Cm",
        name: "Curium",
        primary_mass_number: 247,
        mass_amu: 247.07035410000,
        standard_atomic_weight: 247.00000000,
    },
    ElementData {
        atomic_number: 97,
        symbol: "Bk",
        name: "Berkelium",
        primary_mass_number: 247,
        mass_amu: 247.07030730000,
        standard_atomic_weight: 247.00000000,
    },
    ElementData {
        atomic_number: 98,
        symbol: "Cf",
        name: "Californium",
        primary_mass_number: 251,
        mass_amu: 251.07958860000,
        standard_atomic_weight: 237.06219800,
    },
    ElementData {
        atomic_number: 99,
        symbol: "Es",
        name: "Einsteinium",
        primary_mass_number: 252,
        mass_amu: 252.08298000000,
        standard_atomic_weight: 252.00000000,
    },
    ElementData {
        atomic_number: 100,
        symbol: "Fm",
        name: "Fermium",
        primary_mass_number: 257,
        mass_amu: 257.09510610000,
        standard_atomic_weight: 257.00000000,
    },
    ElementData {
        atomic_number: 101,
        symbol: "Md",
        name: "Mendelevium",
        primary_mass_number: 258,
        mass_amu: 258.09843150000,
        standard_atomic_weight: 258.00000000,
    },
    ElementData {
        atomic_number: 102,
        symbol: "No",
        name: "Nobelium",
        primary_mass_number: 259,
        mass_amu: 259.10103000000,
        standard_atomic_weight: 259.00000000,
    },
    ElementData {
        atomic_number: 103,
        symbol: "Lr",
        name: "Lawrencium",
        primary_mass_number: 266,
        mass_amu: 266.11983000000,
        standard_atomic_weight: 266.00000000,
    },
    ElementData {
        atomic_number: 104,
        symbol: "Rf",
        name: "Rutherfordium",
        primary_mass_number: 267,
        mass_amu: 267.12179000000,
        standard_atomic_weight: 267.00000000,
    },
    ElementData {
        atomic_number: 105,
        symbol: "Db",
        name: "Dubnium",
        primary_mass_number: 268,
        mass_amu: 268.12567000000,
        standard_atomic_weight: 268.00000000,
    },
    ElementData {
        atomic_number: 106,
        symbol: "Sg",
        name: "Seaborgium",
        primary_mass_number: 269,
        mass_amu: 269.12863000000,
        standard_atomic_weight: 269.00000000,
    },
    ElementData {
        atomic_number: 107,
        symbol: "Bh",
        name: "Bohrium",
        primary_mass_number: 270,
        mass_amu: 270.13336000000,
        standard_atomic_weight: 270.00000000,
    },
    ElementData {
        atomic_number: 108,
        symbol: "Hs",
        name: "Hassium",
        primary_mass_number: 277,
        mass_amu: 277.15190000000,
        standard_atomic_weight: 277.00000000,
    },
    ElementData {
        atomic_number: 109,
        symbol: "Mt",
        name: "Meitnerium",
        primary_mass_number: 278,
        mass_amu: 278.15631000000,
        standard_atomic_weight: 278.00000000,
    },
    ElementData {
        atomic_number: 110,
        symbol: "Ds",
        name: "Darmstadtium",
        primary_mass_number: 281,
        mass_amu: 281.16451000000,
        standard_atomic_weight: 281.00000000,
    },
    ElementData {
        atomic_number: 111,
        symbol: "Rg",
        name: "Roentgenium",
        primary_mass_number: 282,
        mass_amu: 282.16912000000,
        standard_atomic_weight: 282.00000000,
    },
    ElementData {
        atomic_number: 112,
        symbol: "Cn",
        name: "Copernicium",
        primary_mass_number: 285,
        mass_amu: 285.17712000000,
        standard_atomic_weight: 285.00000000,
    },
    ElementData {
        atomic_number: 113,
        symbol: "Nh",
        name: "Nihonium",
        primary_mass_number: 286,
        mass_amu: 286.18221000000,
        standard_atomic_weight: 286.00000000,
    },
    ElementData {
        atomic_number: 114,
        symbol: "Fl",
        name: "Flerovium",
        primary_mass_number: 289,
        mass_amu: 289.19042000000,
        standard_atomic_weight: 289.00000000,
    },
    ElementData {
        atomic_number: 115,
        symbol: "Mc",
        name: "Moscovium",
        primary_mass_number: 290,
        mass_amu: 290.19598000000,
        standard_atomic_weight: 290.00000000,
    },
    ElementData {
        atomic_number: 116,
        symbol: "Lv",
        name: "Livermorium",
        primary_mass_number: 293,
        mass_amu: 293.20449000000,
        standard_atomic_weight: 293.00000000,
    },
    ElementData {
        atomic_number: 117,
        symbol: "Ts",
        name: "Tennessine",
        primary_mass_number: 294,
        mass_amu: 294.21046000000,
        standard_atomic_weight: 294.00000000,
    },
    ElementData {
        atomic_number: 118,
        symbol: "Og",
        name: "Oganesson",
        primary_mass_number: 294,
        mass_amu: 294.21392000000,
        standard_atomic_weight: 294.00000000,
    },
];

/// Returns relative atomic mass in atomic mass units (amu / Da) for atomic number Z.
/// Uses the primary isotope mass suitable for vibrational frequency and mass-weighted Hessian calculations.
/// If Z is out of range, falls back to approximate 2.0 * Z.
#[inline]
pub fn atomic_mass(z: usize) -> f64 {
    if (1..=118).contains(&z) {
        ELEMENTS[z - 1].mass_amu
    } else {
        z as f64 * 2.0
    }
}

/// Returns atomic mass in Hartree atomic units (m_e, electron masses) for atomic number Z.
#[inline]
pub fn atomic_mass_au(z: usize) -> f64 {
    atomic_mass(z) * AMU_TO_AU
}

/// Returns IUPAC standard atomic weight (conventional chemical atomic weight) for atomic number Z.
#[inline]
pub fn standard_atomic_weight(z: usize) -> f64 {
    if (1..=118).contains(&z) {
        ELEMENTS[z - 1].standard_atomic_weight
    } else {
        z as f64 * 2.0
    }
}

/// Returns element symbol for atomic number Z (e.g. 1 -> "H", 6 -> "C", 118 -> "Og").
#[inline]
pub fn element_symbol(z: usize) -> &'static str {
    if (1..=118).contains(&z) {
        ELEMENTS[z - 1].symbol
    } else {
        "X"
    }
}

/// Returns element name for atomic number Z (e.g. 1 -> "Hydrogen", 118 -> "Oganesson").
#[inline]
pub fn element_name(z: usize) -> &'static str {
    if (1..=118).contains(&z) {
        ELEMENTS[z - 1].name
    } else {
        "Unknown"
    }
}

/// Retrieves complete element data for atomic number Z (1 to 118).
#[inline]
pub fn element_info(z: usize) -> Option<&'static ElementData> {
    if (1..=118).contains(&z) {
        Some(&ELEMENTS[z - 1])
    } else {
        None
    }
}

/// Finds atomic number Z from chemical symbol (case-insensitive, e.g. "H" -> 1, "fe" -> 26, "OG" -> 118).
pub fn z_from_symbol(sym: &str) -> Option<usize> {
    let clean = sym.trim();
    for el in ELEMENTS.iter() {
        if el.symbol.eq_ignore_ascii_case(clean) {
            return Some(el.atomic_number);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_118_elements_present() {
        assert_eq!(ELEMENTS.len(), 118);
        for (i, el) in ELEMENTS.iter().enumerate() {
            assert_eq!(el.atomic_number, i + 1);
            assert!(!el.symbol.is_empty());
            assert!(!el.name.is_empty());
            assert!(el.mass_amu > 0.0);
            assert!(el.standard_atomic_weight > 0.0);
        }
    }

    #[test]
    fn test_codata_mass_values() {
        // H: 1.00782503223
        assert!((atomic_mass(1) - 1.00782503223).abs() < 1e-8);
        // He: 4.00260325413
        assert!((atomic_mass(2) - 4.00260325413).abs() < 1e-8);
        // C: 12.0
        assert!((atomic_mass(6) - 12.0).abs() < 1e-8);
        // N: 14.00307400443
        assert!((atomic_mass(7) - 14.00307400443).abs() < 1e-8);
        // O: 15.99491461957
        assert!((atomic_mass(8) - 15.99491461957).abs() < 1e-8);
        // Og: 294.21392
        assert!((atomic_mass(118) - 294.21392).abs() < 1e-4);
    }

    #[test]
    fn test_codata_conversions() {
        assert!((AMU_TO_AU - 1822.888486209).abs() < 1e-6);
        assert!((BOHR_TO_ANGSTROM - 0.529177210903).abs() < 1e-8);
        assert!((HARTREE_TO_EV - 27.211386245988).abs() < 1e-8);
    }

    #[test]
    fn test_symbol_lookups() {
        assert_eq!(z_from_symbol("H"), Some(1));
        assert_eq!(z_from_symbol("h"), Some(1));
        assert_eq!(z_from_symbol("C"), Some(6));
        assert_eq!(z_from_symbol("Fe"), Some(26));
        assert_eq!(z_from_symbol("FE"), Some(26));
        assert_eq!(z_from_symbol("Og"), Some(118));
        assert_eq!(z_from_symbol("NonExistent"), None);

        assert_eq!(element_symbol(1), "H");
        assert_eq!(element_symbol(6), "C");
        assert_eq!(element_symbol(118), "Og");
    }
}
