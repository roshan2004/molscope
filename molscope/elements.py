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
