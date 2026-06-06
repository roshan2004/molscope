"""Per-element reference data: CPK colours and covalent radii.

Values cover the elements common in the sample structures; anything missing
falls back to a neutral default so unknown atoms still render and bond.
"""

# CPK colours (normalised RGB), the convention most molecular viewers use.
CPK_COLORS = {
    "H": (1.00, 1.00, 1.00),
    "C": (0.30, 0.30, 0.30),
    "N": (0.10, 0.10, 0.85),
    "O": (0.85, 0.10, 0.10),
    "S": (0.90, 0.80, 0.20),
    "P": (1.00, 0.50, 0.00),
    "F": (0.30, 0.80, 0.30),
    "CL": (0.20, 0.80, 0.20),
    "BR": (0.60, 0.20, 0.10),
    "I": (0.50, 0.10, 0.60),
    "FE": (0.80, 0.40, 0.10),
    "CA": (0.30, 0.70, 0.70),
    "NA": (0.50, 0.20, 0.80),
    "MG": (0.20, 0.60, 0.20),
    "ZN": (0.50, 0.50, 0.60),
}
DEFAULT_COLOR = (0.50, 0.50, 0.50)

# Covalent radii in angstrom (Cordero et al. 2008, rounded). Used to infer bonds.
COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05,
    "P": 1.07, "F": 0.57, "CL": 1.02, "BR": 1.20, "I": 1.39,
    "FE": 1.32, "CA": 1.76, "NA": 1.66, "MG": 1.41, "ZN": 1.22,
}
DEFAULT_RADIUS = 0.75

# Van der Waals radii in angstrom (Bondi 1964, with common protein/ion values).
# Used for solvent-accessible surface area; unknown atoms fall back to carbon.
VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "FE": 2.00, "CA": 2.31, "NA": 2.27, "MG": 1.73, "ZN": 1.39,
}
DEFAULT_VDW_RADIUS = 1.70


# Maximum solvent-accessible surface area per residue in Å² (Tien et al. 2013,
# "theoretical" values), used to normalise absolute SASA to relative SASA (RSA).
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLU": 223.0, "GLN": 225.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}


# Standard atomic weights (g/mol). Unknown atoms fall back to 1.0 so that a
# mass-weighted centre over all-unknown elements reduces to the geometric mean.
ATOMIC_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06,
    "P": 30.974, "F": 18.998, "CL": 35.45, "BR": 79.904, "I": 126.904,
    "FE": 55.845, "CA": 40.078, "NA": 22.990, "MG": 24.305, "ZN": 65.38,
}
DEFAULT_MASS = 1.0


def color(element: str):
    """CPK colour for an element symbol (case-insensitive)."""
    return CPK_COLORS.get(element.upper(), DEFAULT_COLOR)


def covalent_radius(element: str) -> float:
    """Covalent radius in angstrom for an element symbol (case-insensitive)."""
    return COVALENT_RADII.get(element.upper(), DEFAULT_RADIUS)


def vdw_radius(element: str) -> float:
    """Van der Waals radius in angstrom for an element symbol (case-insensitive)."""
    return VDW_RADII.get(element.upper(), DEFAULT_VDW_RADIUS)


def max_asa(resname: str):
    """Reference maximum SASA (Å², Tien et al. 2013) for a standard amino acid.

    Returns ``None`` for non-standard residues, ligands or waters, which have no
    reference and so cannot be normalised to a relative SASA.
    """
    return MAX_ASA.get((resname or "").upper())


def mass(element: str) -> float:
    """Atomic weight in g/mol for an element symbol (case-insensitive)."""
    return ATOMIC_MASSES.get(element.upper(), DEFAULT_MASS)


# Atomic numbers for the first four periods (enough for biomolecules and most
# small molecules); unknown symbols map to 0 so graph code never crashes.
ATOMIC_NUMBERS = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "NE": 10, "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15,
    "S": 16, "CL": 17, "AR": 18, "K": 19, "CA": 20, "SC": 21, "TI": 22,
    "V": 23, "CR": 24, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29,
    "ZN": 30, "GA": 31, "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36,
    "I": 53,
}


def atomic_number(element: str) -> int:
    """Atomic number (Z) for an element symbol (case-insensitive); 0 if unknown."""
    return ATOMIC_NUMBERS.get(element.upper(), 0)


# Every IUPAC element symbol (Z 1-118), upper-cased. Unlike ``ATOMIC_NUMBERS``
# (which only carries reference data for the light elements MolScope models),
# this is the full periodic table, used purely to tell a real element symbol
# apart from parse junk in quality checks. ``"D"``/``"T"`` (deuterium/tritium)
# are accepted as hydrogen isotopes that show up in real structure files.
ELEMENT_SYMBOLS = frozenset(
    {
        "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
        "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", "K", "CA",
        "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
        "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR",
        "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
        "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
        "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
        "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
        "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
        "PA", "U", "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM",
        "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS",
        "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG",
    }
    | {"D", "T"}
)


def is_element(symbol: str) -> bool:
    """True if ``symbol`` is a real IUPAC element (case-insensitive).

    Deuterium (``D``) and tritium (``T``) count as elements. A blank or
    ``None`` symbol is not an element. Used to flag parse junk in QC.
    """
    return bool(symbol) and symbol.strip().upper() in ELEMENT_SYMBOLS
