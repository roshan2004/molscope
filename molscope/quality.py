"""Lightweight structure quality control: a fast parse-fidelity inventory.

``quality_report`` reads a structure once and reports *what is in the file and
whether it parsed cleanly* — atom and chain counts, the ligand / water / ion
inventory, which per-atom metadata the format carried, blank or non-element
atom symbols, whether bonds are explicit (from the file) or inferred from
geometry, alternate-location / occupancy flags, and CIF/PDB validity warnings.

This is the format-agnostic complement to :func:`molscope.prepare_structure`.
Where ``prepare_structure`` answers "is this protein *ML-ready*?" (backbone
gaps, chain breaks, net charge), ``quality_report`` answers the cheaper,
upstream question "did this file parse into something sensible?" and works on
``.xyz`` and ``.sdf`` small molecules just as well as on proteins. Everything
here runs on the bare NumPy install.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import elements
from .contacts import ION_RESNAMES, WATER_RESNAMES
from .molecule import Molecule

# Per-atom annotation arrays a structure file may or may not carry, paired with
# the human-readable label used in the report. Order is the report order.
_METADATA_FIELDS = (
    ("elements", "element symbols"),
    ("atom_names", "atom names"),
    ("resnames", "residue names"),
    ("resids", "residue numbers"),
    ("chains", "chain identifiers"),
    ("hetero", "ATOM/HETATM record types"),
)


@dataclass(frozen=True)
class QualityReport:
    """The result of :func:`quality_report`: a structure parse-quality summary."""

    path: str
    fmt: str  # data extension, e.g. ".pdb"; "" when unknown
    n_atoms: int
    n_models: int = 1
    chains: list[str] = field(default_factory=list)
    n_residues: int = 0
    ligands: dict = field(default_factory=dict)  # resname -> count (non-water/ion hetero)
    n_waters: int = 0
    n_ions: int = 0
    n_hetero_atoms: int = 0
    missing_metadata: list[str] = field(default_factory=list)  # absent per-atom fields
    unknown_elements: dict = field(default_factory=dict)  # symbol -> count (not an element)
    blank_elements: int = 0
    bond_source: str = "none"  # "explicit" | "inferred" | "none"
    n_bonds: int = 0
    altloc_atoms: int = 0
    low_occupancy_atoms: int = 0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def issues(self) -> list[str]:
        """Problems that point at a bad parse or a malformed file."""
        out = []
        if self.n_atoms == 0:
            out.append("no atoms parsed")
        if self.blank_elements:
            out.append(f"{self.blank_elements} atom(s) with no element symbol")
        if self.unknown_elements:
            n = sum(self.unknown_elements.values())
            syms = ", ".join(sorted(self.unknown_elements))
            out.append(f"{n} atom(s) with unrecognised element symbol(s): {syms}")
        out.extend(self.warnings)
        return out

    @property
    def clean(self) -> bool:
        """Heuristic verdict: true when nothing in :attr:`issues` fired."""
        return not self.issues

    def to_dict(self) -> dict:
        """JSON-serialisable view."""
        return {
            "path": self.path,
            "fmt": self.fmt,
            "n_atoms": self.n_atoms,
            "n_models": self.n_models,
            "chains": list(self.chains),
            "n_residues": self.n_residues,
            "ligands": dict(self.ligands),
            "n_waters": self.n_waters,
            "n_ions": self.n_ions,
            "n_hetero_atoms": self.n_hetero_atoms,
            "missing_metadata": list(self.missing_metadata),
            "unknown_elements": dict(self.unknown_elements),
            "blank_elements": self.blank_elements,
            "bond_source": self.bond_source,
            "n_bonds": self.n_bonds,
            "altloc_atoms": self.altloc_atoms,
            "low_occupancy_atoms": self.low_occupancy_atoms,
            "clean": self.clean,
            "issues": self.issues,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """One-line inventory for logs and CLI output."""
        verdict = "clean" if self.clean else "issues"
        bits = [f"{self.path}: {verdict}", f"{self.n_atoms} atoms"]
        if self.n_models > 1:
            bits.append(f"{self.n_models} models")
        if self.chains:
            bits.append(f"chains {','.join(self.chains)}")
        n_lig = sum(self.ligands.values())
        if n_lig:
            bits.append(f"{n_lig} ligand(s)")
        if self.bond_source != "none":
            bits.append(f"bonds {self.bond_source} ({self.n_bonds})")
        if not self.clean:
            bits.append("issues: " + "; ".join(self.issues))
        return " | ".join(bits)

    def report_markdown(self) -> str:
        """A human-readable Markdown report."""
        lines = [
            "# Structure quality report",
            "",
            f"- File: `{self.path}`" + (f" ({self.fmt})" if self.fmt else ""),
            f"- Verdict: **{'clean' if self.clean else 'issues found'}**",
            f"- Atoms: **{self.n_atoms}**"
            + (f" ({self.n_models} models)" if self.n_models > 1 else ""),
            f"- Chains: {', '.join(self.chains) or '(none)'}",
            f"- Residues: {self.n_residues}",
            f"- Bonds: {self.n_bonds} ({self.bond_source})",
        ]
        if self.issues:
            lines.extend(["", f"## Issues ({len(self.issues)})", ""])
            lines.extend(f"- {i}" for i in self.issues)
        if self.ligands:
            lines.extend(["", f"## Ligands ({sum(self.ligands.values())})", ""])
            lines.extend(
                f"- {name}: {count}" for name, count in sorted(self.ligands.items())
            )
        if self.n_waters or self.n_ions:
            lines.extend(
                ["", f"- Waters: {self.n_waters}", f"- Ions: {self.n_ions}"]
            )
        if self.altloc_atoms or self.low_occupancy_atoms:
            lines.extend([
                "", "## Alternate locations", "",
                f"- Atoms in alternate conformations: {self.altloc_atoms}",
                f"- Atoms with occupancy < 1: {self.low_occupancy_atoms}",
            ])
        if self.missing_metadata:
            lines.extend(["", "## Metadata not carried by this file", ""])
            lines.extend(f"- {m}" for m in self.missing_metadata)
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {n}" for n in self.notes)
        return "\n".join(lines) + "\n"


def quality_report(source: str | Molecule) -> QualityReport:
    """Read a structure once and return a parse-quality :class:`QualityReport`.

    ``source`` is a path to a ``.pdb`` / ``.cif`` / ``.xyz`` / ``.sdf`` file
    (optionally gzipped), or an already-read :class:`~molscope.Molecule`. The
    inventory (atoms, chains, residues, ligand / water / ion counts, metadata
    completeness, element-symbol sanity, bond provenance) runs on the bare
    NumPy install.

    Alternate-location / occupancy counts (PDB) and mmCIF validity warnings are
    read from the *file*, so they are only available when ``source`` is a path.
    Passing a :class:`~molscope.Molecule` skips those file-level checks with a
    note. Optional backends degrade gracefully: a missing gemmi makes mmCIF
    validation a note rather than an error.
    """
    notes: list[str] = []
    warnings: list[str] = []

    if isinstance(source, Molecule):
        mol = source
        path = mol.name or "<molecule>"
        fmt = ""
        notes.append("file-level checks (altLoc, CIF/PDB validity) skipped: in-memory molecule")
    else:
        from .io import _data_extension, read

        path = str(source)
        fmt = _data_extension(path)
        mol = read(path)

    n_atoms = len(mol)
    chains = _ordered_unique(mol.chains) if mol.chains else []

    (n_residues, ligands, n_waters, n_ions) = _residue_inventory(mol)
    n_hetero_atoms = int(sum(bool(h) for h in mol.hetero)) if mol.hetero else 0

    missing_metadata = _missing_metadata(mol)
    unknown_elements, blank_elements = _element_sanity(mol)
    bond_source, n_bonds = _bond_provenance(mol)

    altloc_atoms = low_occupancy_atoms = 0
    n_models = 1
    if fmt == ".pdb":
        from .structure_prep import _pdb_occupancy_scan

        altloc_atoms, low_occupancy_atoms, n_models = _pdb_occupancy_scan(path)
    elif fmt in (".cif", ".mmcif"):
        _append_cif_warnings(path, warnings, notes)

    return QualityReport(
        path=path, fmt=fmt, n_atoms=n_atoms, n_models=n_models, chains=chains,
        n_residues=n_residues, ligands=ligands, n_waters=n_waters, n_ions=n_ions,
        n_hetero_atoms=n_hetero_atoms, missing_metadata=missing_metadata,
        unknown_elements=unknown_elements, blank_elements=blank_elements,
        bond_source=bond_source, n_bonds=n_bonds, altloc_atoms=altloc_atoms,
        low_occupancy_atoms=low_occupancy_atoms, warnings=warnings, notes=notes,
    )


# -- internals --------------------------------------------------------------

def _ordered_unique(values) -> list:
    seen, out = set(), []
    for v in values:
        key = str(v)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _residue_inventory(mol: Molecule):
    """Walk residues once: total count and the water / ion / ligand split."""
    n_residues = n_waters = n_ions = 0
    ligands: dict = {}
    hetero = mol.hetero
    for group in mol.residue_groups():
        n_residues += 1
        resname = str(group.resname).strip().upper()
        if resname in WATER_RESNAMES:
            n_waters += 1
        elif resname in ION_RESNAMES:
            n_ions += 1
        elif hetero and any(hetero[i] for i in group.atom_indices):
            # A non-water, non-ion HETATM group: a ligand or cofactor.
            ligands[resname] = ligands.get(resname, 0) + 1
    return n_residues, ligands, n_waters, n_ions


def _missing_metadata(mol: Molecule) -> list[str]:
    """Per-atom annotation arrays this molecule does not carry."""
    missing = []
    for attr, label in _METADATA_FIELDS:
        value = getattr(mol, attr)
        present = len(value) > 0 if attr == "resids" else bool(value)
        # ``elements`` is filled with blanks rather than left empty, so treat an
        # all-blank element list as "carried but empty" — the blank-element
        # check below reports the substance.
        if attr == "elements":
            present = any(str(e).strip() for e in value)
        if not present:
            missing.append(label)
    return missing


def _element_sanity(mol: Molecule):
    """Count blank symbols and tally non-element symbols (likely parse junk)."""
    unknown: dict = {}
    blank = 0
    for sym in mol.elements:
        text = str(sym).strip()
        if not text:
            blank += 1
        elif not elements.is_element(text):
            key = text.upper()
            unknown[key] = unknown.get(key, 0) + 1
    return unknown, blank


def _bond_provenance(mol: Molecule):
    """Whether bonds came from the file (explicit) or geometry (inferred)."""
    if mol.bond_index is not None:
        return "explicit", int(len(mol.bond_index))
    if len(mol) >= 2:
        return "inferred", int(len(mol.bonds()))
    return "none", 0


def _append_cif_warnings(path: str, warnings: list, notes: list) -> None:
    """Fold mmCIF validity errors/warnings into the report when gemmi is present."""
    try:
        from .cif import validate_cif

        report = validate_cif(path)
    except ImportError as exc:
        notes.append(f"mmCIF not validated: {exc}")
        return
    warnings.extend(report.warnings)
    if not report.valid:
        warnings.append("mmCIF invalid: " + ("; ".join(report.errors) or "validation failed"))
    else:
        notes.append("mmCIF validation: passed")
