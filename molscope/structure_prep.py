"""One-shot structure quality control: is this structure ML-ready?

MolScope already has the *pieces* needed to judge whether a coordinate file is
fit to feed a descriptor / contact-map / graph-ML pipeline — residue iteration,
template-bond perception, pKa-aware protonation, mmCIF validation. This module is
the glue: :func:`prepare_structure` reads a file once and returns a single
:class:`StructureReport` answering the questions that decide "can I trust this as
an ML input?":

- non-standard residues and a ligand / water / ion inventory,
- residue-numbering gaps and backbone chain breaks (missing density),
- residues missing backbone atoms or with truncated side chains,
- whether hydrogens are present,
- the net formal charge at a chosen pH (via the existing protonation backends),
- alternate conformations and partial occupancies (PDB only).

It composes existing MolScope functionality and adds no new dependency: the
topology checks run on the bare NumPy install, and only the net-charge step (and
the optional ``"pka"`` mode) need the ``chem`` / ``propka`` extras, degrading to
a clearly-labelled ``None`` when they are absent.

The ``ml_ready`` verdict is a *heuristic*: it flags missing backbone atoms and
internal chain breaks as blockers (they corrupt distance- and graph-based
features) and surfaces everything else as warnings. It is a triage aid, not a
substitute for judgement about a specific modelling task.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .contacts import AMINO_ACID_RESNAMES

#: Standard protein backbone heavy atoms; their absence is a hard blocker.
BACKBONE_ATOMS = ("N", "CA", "C", "O")

#: Expected heavy-atom (non-hydrogen) count per standard residue, including the
#: four backbone atoms. Used to flag truncated side chains: a residue with fewer
#: heavy atoms than this is missing density. Counting only *shortfalls* means the
#: terminal ``OXT`` and alternate locations never cause a false positive.
EXPECTED_HEAVY_ATOMS = {
    "GLY": 4, "ALA": 5, "SER": 6, "CYS": 6, "THR": 7, "VAL": 7, "PRO": 7,
    "LEU": 8, "ILE": 8, "ASN": 8, "ASP": 8, "MET": 8, "GLN": 9, "GLU": 9,
    "LYS": 9, "HIS": 10, "PHE": 11, "ARG": 11, "TYR": 12, "TRP": 14,
}

#: Standard nucleotide residue names (DNA + RNA), counted as polymer residues.
NUCLEOTIDE_RESNAMES = (
    "DA", "DC", "DG", "DT", "DU", "A", "C", "G", "U", "I",
)

#: Common crystallographic water residue names.
WATER_RESNAMES = ("HOH", "WAT", "H2O", "DOD", "TIP", "SOL")

#: CA-CA distance (Å) above which consecutive residues are treated as a chain
#: break. Bonded CA atoms sit at ~3.8 Å; 4.5 leaves margin for strained models.
CHAIN_BREAK_CA = 4.5

_STANDARD_AA = frozenset(AMINO_ACID_RESNAMES)
_STANDARD_NA = frozenset(NUCLEOTIDE_RESNAMES)
_WATER = frozenset(WATER_RESNAMES)


@dataclass(frozen=True)
class StructureReport:
    """The result of :func:`prepare_structure`: a structure-readiness summary."""

    path: str
    n_atoms: int
    n_models: int = 1
    chains: list[str] = field(default_factory=list)
    n_polymer_residues: int = 0
    has_hydrogens: bool = False
    net_charge: Optional[int] = None
    charge_method: str = "none"  # "standard" | "pka" | "as-read" | "unavailable"
    ph: Optional[float] = None
    nonstandard_residues: list[tuple] = field(default_factory=list)  # (chain, resid, resname)
    ligands: dict = field(default_factory=dict)  # resname -> count (non-water hetero)
    n_waters: int = 0
    residue_gaps: list[tuple] = field(default_factory=list)  # (chain, before, after, n_missing)
    chain_breaks: list[tuple] = field(default_factory=list)  # (chain, before, after, ca_dist)
    missing_backbone: list[tuple] = field(default_factory=list)  # (chain, resid, resname, [atoms])
    truncated_sidechains: list[tuple] = field(default_factory=list)  # (chain, resid, name, n, exp)
    altloc_atoms: int = 0
    low_occupancy_atoms: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def blockers(self) -> list[str]:
        """Issues that corrupt distance/graph features: missing backbone, breaks."""
        out = []
        if self.missing_backbone:
            out.append(
                f"{len(self.missing_backbone)} residue(s) missing backbone atoms"
            )
        if self.chain_breaks:
            out.append(f"{len(self.chain_breaks)} backbone chain break(s)")
        return out

    @property
    def warnings(self) -> list[str]:
        """Issues worth knowing about that don't, by themselves, block ML use."""
        out = []
        if self.residue_gaps:
            out.append(f"{len(self.residue_gaps)} residue-numbering gap(s)")
        if self.truncated_sidechains:
            out.append(f"{len(self.truncated_sidechains)} truncated side chain(s)")
        if self.nonstandard_residues:
            out.append(f"{len(self.nonstandard_residues)} non-standard residue(s)")
        if self.altloc_atoms:
            out.append(f"{self.altloc_atoms} atom(s) in alternate conformations")
        if self.low_occupancy_atoms:
            out.append(f"{self.low_occupancy_atoms} atom(s) with occupancy < 1")
        if not self.has_hydrogens:
            out.append("no hydrogens present")
        return out

    @property
    def ml_ready(self) -> bool:
        """Heuristic verdict: true when there are no blocker-level issues."""
        return not self.blockers

    def to_dict(self) -> dict:
        """JSON-serialisable view (tuples become lists)."""
        return {
            "path": self.path,
            "n_atoms": self.n_atoms,
            "n_models": self.n_models,
            "chains": list(self.chains),
            "n_polymer_residues": self.n_polymer_residues,
            "has_hydrogens": self.has_hydrogens,
            "net_charge": self.net_charge,
            "charge_method": self.charge_method,
            "ph": self.ph,
            "ml_ready": self.ml_ready,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "nonstandard_residues": [list(r) for r in self.nonstandard_residues],
            "ligands": dict(self.ligands),
            "n_waters": self.n_waters,
            "residue_gaps": [list(g) for g in self.residue_gaps],
            "chain_breaks": [list(b) for b in self.chain_breaks],
            "missing_backbone": [list(m) for m in self.missing_backbone],
            "truncated_sidechains": [list(t) for t in self.truncated_sidechains],
            "altloc_atoms": self.altloc_atoms,
            "low_occupancy_atoms": self.low_occupancy_atoms,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """One-line verdict for logs and CLI output."""
        verdict = "ML-ready" if self.ml_ready else "NOT ML-ready"
        bits = [f"{self.path}: {verdict}", f"{self.n_atoms} atoms"]
        if self.chains:
            bits.append(f"chains {','.join(self.chains)}")
        if self.net_charge is not None:
            at = f" @pH{self.ph:g}" if self.ph is not None and self.charge_method == "pka" else ""
            bits.append(f"net charge {self.net_charge:+d}{at}")
        if self.blockers:
            bits.append("blockers: " + "; ".join(self.blockers))
        elif self.warnings:
            bits.append("warnings: " + "; ".join(self.warnings))
        return " | ".join(bits)

    def report_markdown(self) -> str:
        """A human-readable Markdown report."""
        lines = [
            "# Structure preparation report",
            "",
            f"- File: `{self.path}`",
            f"- Verdict: **{'ML-ready' if self.ml_ready else 'NOT ML-ready'}**",
            f"- Atoms: **{self.n_atoms}**" + (f" ({self.n_models} models)"
                                              if self.n_models > 1 else ""),
            f"- Chains: {', '.join(self.chains) or '(none)'}",
            f"- Polymer residues: {self.n_polymer_residues}",
        ]
        if self.net_charge is not None:
            method = self.charge_method
            at = f" at pH {self.ph:g}" if method == "pka" and self.ph is not None else ""
            lines.append(f"- Net formal charge: **{self.net_charge:+d}** ({method}{at})")
        else:
            lines.append(f"- Net formal charge: not computed ({self.charge_method})")
        lines.append(f"- Hydrogens present: {'yes' if self.has_hydrogens else 'no'}")

        def block(title, items, fmt):
            if not items:
                return
            lines.extend(["", f"## {title} ({len(items)})", ""])
            lines.extend(fmt(it) for it in items)

        if self.blockers:
            lines.extend(["", "## Blockers", ""])
            lines.extend(f"- {b}" for b in self.blockers)
        block("Missing backbone atoms", self.missing_backbone,
              lambda m: f"- {m[0]}{m[1]} {m[2]}: missing {', '.join(m[3])}")
        block("Chain breaks", self.chain_breaks,
              lambda b: f"- {b[0]}: {b[1]}→{b[2]} (CA-CA {b[3]:.1f} Å)")
        block("Residue-numbering gaps", self.residue_gaps,
              lambda g: f"- {g[0]}: {g[1]}→{g[2]} ({g[3]} missing)")
        block("Truncated side chains", self.truncated_sidechains,
              lambda t: f"- {t[0]}{t[1]} {t[2]}: {t[3]}/{t[4]} heavy atoms")
        block("Non-standard residues", self.nonstandard_residues,
              lambda r: f"- {r[0]}{r[1]} {r[2]}")
        if self.ligands:
            lines.extend(["", f"## Ligands / hetero ({sum(self.ligands.values())})", ""])
            lines.extend(f"- {name}: {count}" for name, count in sorted(self.ligands.items()))
        if self.n_waters:
            lines.extend(["", f"- Waters: {self.n_waters}"])
        if self.altloc_atoms or self.low_occupancy_atoms:
            lines.extend([
                "", "## Occupancy", "",
                f"- Atoms in alternate conformations: {self.altloc_atoms}",
                f"- Atoms with occupancy < 1: {self.low_occupancy_atoms}",
            ])
        if self.notes:
            lines.extend(["", "## Notes", ""])
            lines.extend(f"- {n}" for n in self.notes)
        return "\n".join(lines) + "\n"


def prepare_structure(
    source: str, *, protonation: str = "standard", ph: float = 7.4,
) -> StructureReport:
    """Read a structure once and return an ML-readiness :class:`StructureReport`.

    ``source`` is a path to a ``.pdb``/``.cif``/``.xyz``/``.sdf`` file (optionally
    gzipped). Topology checks (non-standard residues, ligand inventory, numbering
    gaps, backbone breaks, missing/truncated residues, hydrogens) run on the bare
    NumPy install.

    ``protonation`` controls the net-charge calculation for protein PDBs:
    ``"standard"`` (default) uses the idealised pH-7 table, ``"pka"`` predicts it
    with PROPKA at ``ph`` (needs the ``propka`` extra), and ``"none"`` sums the
    as-modelled charges. The net charge needs RDKit (``chem`` extra) for template
    bonds; without it the charge is reported as ``None`` with an explanatory note
    rather than raising.
    """
    from .io import _data_extension, read

    if protonation not in ("none", "standard", "pka"):
        raise ValueError("protonation must be 'none', 'standard', or 'pka'")

    path = str(source)
    ext = _data_extension(path)
    mol = read(path)  # geometric bonds, primary altloc — enough for topology
    notes: list[str] = []

    chains = _ordered_unique(mol.chains) if mol.chains else []
    has_h = any(str(e).strip().upper() in ("H", "D") for e in mol.elements)

    (n_polymer, nonstandard, ligands, n_waters, missing_bb, truncated,
     gaps, breaks) = _topology_checks(mol)

    net_charge, charge_method = _net_charge(path, ext, mol, protonation, ph, notes)

    altloc_atoms, low_occ, n_models = 0, 0, 1
    if ext == ".pdb":
        altloc_atoms, low_occ, n_models = _pdb_occupancy_scan(path)
    elif ext in (".cif", ".mmcif"):
        _append_cif_validation(path, notes)

    return StructureReport(
        path=path, n_atoms=len(mol), n_models=n_models, chains=chains,
        n_polymer_residues=n_polymer, has_hydrogens=has_h,
        net_charge=net_charge, charge_method=charge_method,
        ph=ph if charge_method == "pka" else None,
        nonstandard_residues=nonstandard, ligands=ligands, n_waters=n_waters,
        residue_gaps=gaps, chain_breaks=breaks, missing_backbone=missing_bb,
        truncated_sidechains=truncated, altloc_atoms=altloc_atoms,
        low_occupancy_atoms=low_occ, notes=notes,
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


def _topology_checks(mol):
    """Walk residues once, returning every per-residue finding."""
    n_polymer = 0
    nonstandard: list[tuple] = []
    ligands: dict = {}
    n_waters = 0
    missing_bb: list[tuple] = []
    truncated: list[tuple] = []
    # Per chain: ordered list of (resid, ca_coord_or_None) for polymer residues.
    chain_polymers: dict = {}

    elements = mol.elements or [""] * len(mol)
    for group in mol.residue_groups():
        idx, resname, resid, chain = group.atom_indices, group.resname, group.resid, group.chain
        resname = str(resname).strip().upper()

        if resname in _WATER:
            n_waters += 1
            continue

        is_aa = resname in _STANDARD_AA
        is_na = resname in _STANDARD_NA
        if not (is_aa or is_na):
            # Heavy (non-water) hetero: ligand, ion, modified residue, or cofactor.
            ligands[resname] = ligands.get(resname, 0) + 1
            nonstandard.append((chain, int(resid), resname))
            continue

        n_polymer += 1
        names = {str(mol.atom_names[i]).strip().upper() for i in idx}

        if is_aa:
            absent = [a for a in BACKBONE_ATOMS if a not in names]
            if absent:
                missing_bb.append((chain, int(resid), resname, absent))
            expected = EXPECTED_HEAVY_ATOMS.get(resname)
            if expected is not None:
                n_heavy = sum(
                    1 for i in idx if str(elements[i]).strip().upper() not in ("H", "D")
                )
                if n_heavy < expected:
                    truncated.append((chain, int(resid), resname, n_heavy, expected))

        ca = None
        if is_aa:
            for i in idx:
                if str(mol.atom_names[i]).strip().upper() == "CA":
                    ca = mol.coords[i]
                    break
        chain_polymers.setdefault(chain, []).append((int(resid), ca))

    gaps, breaks = _sequence_gaps_and_breaks(chain_polymers)
    return n_polymer, nonstandard, ligands, n_waters, missing_bb, truncated, gaps, breaks


def _sequence_gaps_and_breaks(chain_polymers):
    """Numbering gaps (resid jumps) and spatial breaks (CA-CA distance)."""
    gaps: list[tuple] = []
    breaks: list[tuple] = []
    for chain, residues in chain_polymers.items():
        for (r0, ca0), (r1, ca1) in zip(residues, residues[1:]):
            if r1 - r0 > 1:
                gaps.append((chain, r0, r1, r1 - r0 - 1))
            if ca0 is not None and ca1 is not None:
                dist = float(np.linalg.norm(np.asarray(ca1) - np.asarray(ca0)))
                # Only call it a break between sequence-adjacent residues; a jump
                # in numbering is already reported as a gap.
                if r1 - r0 == 1 and dist > CHAIN_BREAK_CA:
                    breaks.append((chain, r0, r1, dist))
    return gaps, breaks


def _net_charge(path, ext, mol, protonation, ph, notes):
    """Net formal charge, degrading to (None, reason) when a backend is absent."""
    if ext == ".pdb" and protonation in ("standard", "pka"):
        try:
            from .io import read as _read

            # RDKit's PDB reader (and PROPKA) need an uncompressed path; transparently
            # decompress a .pdb.gz to a temp file so net charge works for RCSB downloads.
            with _maybe_decompressed(path) as template_path:
                charged = _read(template_path, bond_perception="template",
                                protonation=protonation, ph=ph)
            return int(sum(int(c) for c in charged.formal_charges)), protonation
        except ImportError as exc:
            notes.append(f"net charge not computed (missing backend): {exc}")
            return None, "unavailable"
        except ValueError as exc:
            # RDKit can't read gzipped or otherwise non-template-parseable PDBs;
            # the report should note this, not abort.
            notes.append(f"net charge not computed (template bonds failed): {exc}")
            return None, "unavailable"
    if len(mol.formal_charges):
        return int(sum(int(c) for c in mol.formal_charges)), "as-read"
    if protonation == "none":
        return 0, "none"
    notes.append(
        "net charge not computed: protonation charges need a .pdb file; "
        "got " + (ext or "no extension")
    )
    return None, "unavailable"


@contextlib.contextmanager
def _maybe_decompressed(path):
    """Yield ``path`` unchanged, or a temp uncompressed copy for a ``.pdb.gz``.

    RDKit's ``MolFromPDBFile`` and PROPKA both read a raw path and do not handle
    gzip, so the template-bond net-charge step would otherwise silently fail on
    the gzipped files RCSB serves. The temp file is removed on exit.
    """
    if not str(path).endswith(".gz"):
        yield path
        return
    from .io import _open

    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False, encoding="utf-8"
    )
    try:
        with _open(path) as src:
            handle.write(src.read())
        handle.close()
        yield handle.name
    finally:
        with contextlib.suppress(OSError):
            os.unlink(handle.name)


def _pdb_occupancy_scan(path):
    """Count alternate-location atoms, partial-occupancy atoms, and MODEL records.

    A raw pass over the file because alternate locations and occupancies are
    dropped once a :class:`Molecule` is built (they only steer altloc selection).
    """
    from .io import _open, _pdb_float

    altloc_atoms = low_occ = 0
    n_models = 0
    with _open(path) as handle:
        for line in handle:
            if line.startswith("MODEL"):
                n_models += 1
            elif line.startswith(("ATOM", "HETATM")):
                altloc = line[16] if len(line) > 16 else " "
                if altloc not in (" ", ""):
                    altloc_atoms += 1
                occ = _pdb_float(line[54:60], default=1.0) if len(line) >= 60 else 1.0
                if occ < 1.0:
                    low_occ += 1
    return altloc_atoms, low_occ, max(1, n_models)


def _append_cif_validation(path, notes):
    """Fold an mmCIF validity check into the notes when gemmi is available."""
    try:
        from .cif import validate_cif

        report = validate_cif(path)
    except ImportError as exc:
        notes.append(f"mmCIF not validated: {exc}")
        return
    if report.valid:
        notes.append("mmCIF validation: passed")
    else:
        notes.append("mmCIF validation: FAILED — " + ("; ".join(report.errors) or "invalid"))
