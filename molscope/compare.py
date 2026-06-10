"""Static-structure comparison: align two structures and quantify the difference.

:func:`compare_structures` reads two structures, matches their atoms by residue
identity (``chain``/``resid``/insertion-code/atom-name) so it works on two
*different* files — different atom counts, different ordering, point mutations —
not just two frames of one trajectory, and reports:

- the aligned (Kabsch) RMSD over the matched atom set,
- per-residue deviations after that superposition,
- a residue contact-map delta restricted to the residues both share
  (contacts gained vs lost), and
- a descriptor delta (per-feature change between the two whole structures).

This is deliberately a *static* comparison — one structure against another — built
from MolScope's existing superpose / contact-map / descriptor primitives. It is
not a trajectory analyser: there is no time axis, no per-frame averaging, and no
topology building. When the two structures carry no residue metadata (e.g. two
``.xyz`` files) the atoms are matched by index instead, which requires equal atom
counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .molecule import Molecule

#: Protein backbone heavy atoms, the basis for ``atoms="backbone"`` matching.
_BACKBONE = ("N", "CA", "C", "O")


@dataclass
class ResidueDeviation:
    """Per-residue RMSD between two superposed structures."""

    label: str
    n_atoms: int
    rmsd: float


@dataclass
class ContactDelta:
    """Residue contact-map difference over the residues both structures share."""

    cutoff: float
    method: str
    n_common_residues: int
    n_contacts_a: int
    n_contacts_b: int
    gained: int  # contacts present in B but not A (over common residues)
    lost: int    # contacts present in A but not B


@dataclass
class DescriptorDelta:
    """One descriptor's value in each structure and the B-minus-A change."""

    name: str
    a: float
    b: float
    delta: float


@dataclass
class ComparisonResult:
    """The result of :func:`compare_structures`: A (reference) vs B."""

    name_a: str
    name_b: str
    n_atoms_a: int
    n_atoms_b: int
    atom_set: str  # "all" | "ca" | "backbone"
    match_method: str  # "residue" | "index"
    n_matched_atoms: int
    n_common_residues: int
    superposed: bool
    rmsd: float  # aligned when superposed, else as-is
    rmsd_unaligned: float
    per_residue: list = field(default_factory=list)  # list[ResidueDeviation]
    contact: Optional[ContactDelta] = None
    descriptors: list = field(default_factory=list)  # list[DescriptorDelta]
    descriptor_preset: str = ""
    notes: list = field(default_factory=list)

    @property
    def n_changed_descriptors(self) -> int:
        return sum(1 for d in self.descriptors if d.delta != 0.0)

    def largest_deviations(self, n: int = 5) -> list:
        """The ``n`` residues that moved most, largest first."""
        return sorted(self.per_residue, key=lambda d: d.rmsd, reverse=True)[:n]

    def to_dict(self) -> dict:
        """JSON-serialisable view."""
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "n_atoms_a": self.n_atoms_a,
            "n_atoms_b": self.n_atoms_b,
            "atom_set": self.atom_set,
            "match_method": self.match_method,
            "n_matched_atoms": self.n_matched_atoms,
            "n_common_residues": self.n_common_residues,
            "superposed": self.superposed,
            "rmsd": self.rmsd,
            "rmsd_unaligned": self.rmsd_unaligned,
            "per_residue": [
                {"label": d.label, "n_atoms": d.n_atoms, "rmsd": d.rmsd}
                for d in self.per_residue
            ],
            "contact": (
                None if self.contact is None else {
                    "cutoff": self.contact.cutoff,
                    "method": self.contact.method,
                    "n_common_residues": self.contact.n_common_residues,
                    "n_contacts_a": self.contact.n_contacts_a,
                    "n_contacts_b": self.contact.n_contacts_b,
                    "gained": self.contact.gained,
                    "lost": self.contact.lost,
                }
            ),
            "descriptor_preset": self.descriptor_preset,
            "descriptors": [
                {"name": d.name, "a": d.a, "b": d.b, "delta": d.delta}
                for d in self.descriptors
            ],
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """Multi-line human-readable comparison for CLI output."""
        kind = "aligned" if self.superposed else "as-is"
        lines = [
            f"{self.name_a} vs {self.name_b}",
            f"matched {self.n_matched_atoms} atoms on '{self.atom_set}' "
            f"(A {self.n_atoms_a}, B {self.n_atoms_b}; by {self.match_method}); "
            f"{self.n_common_residues} common residue(s)",
            f"{kind} RMSD {self.rmsd:.3f} Å"
            + ("" if self.superposed else " (no superposition)")
            + (f" | unaligned {self.rmsd_unaligned:.3f} Å" if self.superposed else ""),
        ]
        top = self.largest_deviations()
        if top:
            moved = ", ".join(f"{d.label} {d.rmsd:.2f}" for d in top)
            lines.append(f"largest per-residue deviation(s): {moved}")
        if self.contact is not None:
            c = self.contact
            lines.append(
                f"contact map ({c.cutoff:g} Å, {c.method}): {c.n_contacts_a} → "
                f"{c.n_contacts_b} contact(s) over {c.n_common_residues} common "
                f"residue(s) — {c.gained} gained, {c.lost} lost"
            )
        if self.descriptors:
            lines.append(
                f"descriptors ({self.descriptor_preset}): "
                f"{self.n_changed_descriptors} of {len(self.descriptors)} changed"
            )
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)

    def report_markdown(self) -> str:
        """A human-readable Markdown report (full per-residue and descriptor tables)."""
        kind = "Aligned" if self.superposed else "As-is"
        lines = [
            f"# Structure comparison: {self.name_a} vs {self.name_b}",
            "",
            f"- Reference (A): `{self.name_a}` ({self.n_atoms_a} atoms)",
            f"- Compared (B): `{self.name_b}` ({self.n_atoms_b} atoms)",
            f"- Atom set: **{self.atom_set}** (matched by {self.match_method})",
            f"- Matched atoms: **{self.n_matched_atoms}** "
            f"over {self.n_common_residues} common residue(s)",
            f"- {kind} RMSD: **{self.rmsd:.3f} Å**",
        ]
        if self.superposed:
            lines.append(f"- Unaligned RMSD: {self.rmsd_unaligned:.3f} Å")

        if self.contact is not None:
            c = self.contact
            lines += [
                "", "## Contact-map delta", "",
                f"- Cutoff: {c.cutoff:g} Å ({c.method})",
                f"- Common residues: {c.n_common_residues}",
                f"- Contacts: {c.n_contacts_a} (A) → {c.n_contacts_b} (B)",
                f"- Gained in B: **{c.gained}** / Lost from A: **{c.lost}**",
            ]

        if self.per_residue:
            lines += ["", "## Per-residue deviations", "",
                      "| residue | atoms | RMSD (Å) |", "| --- | --- | --- |"]
            lines += [f"| {d.label} | {d.n_atoms} | {d.rmsd:.3f} |"
                      for d in self.per_residue]

        if self.descriptors:
            lines += ["", f"## Descriptor delta ({self.descriptor_preset})", "",
                      "| feature | A | B | Δ (B−A) |", "| --- | --- | --- | --- |"]
            lines += [f"| {d.name} | {_fmt(d.a)} | {_fmt(d.b)} | {_fmt(d.delta)} |"
                      for d in self.descriptors]

        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


def compare_structures(
    source_a: str | Molecule,
    source_b: str | Molecule,
    *,
    atoms: str = "all",
    superpose: bool = True,
    include_contact_map: bool = True,
    contact_cutoff: float = 8.0,
    contact_method: str = "ca",
    descriptor_preset: str = "native-basic",
) -> ComparisonResult:
    """Compare structure ``source_b`` against reference ``source_a``.

    Each source is a path to a structure file (``.pdb`` / ``.cif`` / ``.xyz`` /
    ``.sdf``, optionally gzipped) or an already-read :class:`~molscope.Molecule`.
    ``atoms`` chooses which atoms are matched and superposed: ``"all"`` common
    atoms (default), ``"ca"`` (alpha carbons), or ``"backbone"`` (N/CA/C/O).

    Atoms are matched by ``(chain, resid, insertion code, atom name)`` so two
    distinct files line up even with different atom counts or ordering; structures
    without residue metadata fall back to index matching and must then have equal
    atom counts. Raises :class:`ValueError` when no atoms can be matched.
    """
    if atoms not in ("all", "ca", "backbone"):
        raise ValueError(f"atoms must be 'all', 'ca' or 'backbone', got {atoms!r}")

    mol_a = source_a if isinstance(source_a, Molecule) else _read(source_a)
    mol_b = source_b if isinstance(source_b, Molecule) else _read(source_b)

    notes: list[str] = []
    idx_a, idx_b, match_method = _match_atoms(mol_a, mol_b, atoms, notes)
    if len(idx_a) == 0:
        raise ValueError(
            "no atoms could be matched between the two structures "
            f"(atom set {atoms!r}, matched by {match_method}); "
            "are they the same molecule?"
        )

    sub_a = mol_a.take(idx_a)
    sub_b = mol_b.take(idx_b)
    coords_a = sub_a.coords
    unaligned = _rmsd(sub_b.coords, coords_a)
    if superpose:
        coords_b = sub_b.superpose(sub_a).coords
        rmsd = _rmsd(coords_b, coords_a)
    else:
        coords_b = sub_b.coords
        rmsd = unaligned

    per_residue = _per_residue(mol_a, idx_a, coords_a, coords_b, match_method)
    n_common_residues = len(per_residue) if match_method == "residue" else 0

    contact = None
    if include_contact_map:
        contact = _contact_delta(mol_a, mol_b, contact_cutoff, contact_method, notes)

    descriptors = _descriptor_delta(mol_a, mol_b, descriptor_preset, notes)

    return ComparisonResult(
        name_a=mol_a.name or "A",
        name_b=mol_b.name or "B",
        n_atoms_a=len(mol_a),
        n_atoms_b=len(mol_b),
        atom_set=atoms,
        match_method=match_method,
        n_matched_atoms=len(idx_a),
        n_common_residues=n_common_residues,
        superposed=superpose,
        rmsd=rmsd,
        rmsd_unaligned=unaligned,
        per_residue=per_residue,
        contact=contact,
        descriptors=descriptors,
        descriptor_preset=descriptor_preset,
        notes=notes,
    )


# -- internals --------------------------------------------------------------

def _read(source: str) -> Molecule:
    from .io import read

    return read(source)


def _rmsd(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(((p - q) ** 2).sum() / len(q))) if len(q) else 0.0


def _atom_set_mask(mol: Molecule, atoms: str) -> np.ndarray:
    """Boolean mask selecting the atoms eligible for matching."""
    if atoms == "all":
        return np.ones(len(mol), dtype=bool)
    names = mol.atom_names
    if not names:
        return np.zeros(len(mol), dtype=bool)
    wanted = {"CA"} if atoms == "ca" else set(_BACKBONE)
    return np.array([str(n).strip().upper() in wanted for n in names], dtype=bool)


def _has_residue_metadata(mol: Molecule) -> bool:
    return len(mol.resids) > 0 and bool(mol.atom_names)


def _atom_key(mol: Molecule, i: int):
    chains = mol.chains or None
    icodes = mol.icodes or None
    chain = str(chains[i]).strip() if chains else ""
    icode = str(icodes[i]).strip() if icodes else ""
    name = str(mol.atom_names[i]).strip().upper()
    return (chain, int(mol.resids[i]), icode, name)


def _match_atoms(mol_a: Molecule, mol_b: Molecule, atoms: str, notes: list):
    """Return ``(idx_a, idx_b, method)`` aligning the two structures' atoms.

    Prefers residue-identity matching; falls back to index matching only when
    neither structure carries residue metadata and the atom counts agree.
    """
    if _has_residue_metadata(mol_a) and _has_residue_metadata(mol_b):
        mask_a = _atom_set_mask(mol_a, atoms)
        # Map each B atom key to its index (first occurrence wins, ignoring altlocs).
        b_lookup: dict = {}
        for j in range(len(mol_b)):
            key = _atom_key(mol_b, j)
            if key not in b_lookup:
                b_lookup[key] = j
        idx_a, idx_b, seen = [], [], set()
        for i in range(len(mol_a)):
            if not mask_a[i]:
                continue
            key = _atom_key(mol_a, i)
            if key in seen:  # duplicate atom key in A (altloc); keep the first
                continue
            j = b_lookup.get(key)
            if j is not None:
                seen.add(key)
                idx_a.append(i)
                idx_b.append(j)
        return np.array(idx_a, dtype=int), np.array(idx_b, dtype=int), "residue"

    # Fall back to index matching for metadata-free structures (e.g. .xyz).
    if len(mol_a) != len(mol_b):
        raise ValueError(
            "structures lack residue metadata and have different atom counts "
            f"({len(mol_a)} vs {len(mol_b)}); cannot match atoms by index"
        )
    if atoms != "all":
        raise ValueError(
            f"atom set {atoms!r} needs atom names, which these structures lack"
        )
    notes.append("matched by atom index (no residue metadata to match on)")
    idx = np.arange(len(mol_a))
    return idx, idx.copy(), "index"


def _per_residue(mol_a, idx_a, coords_a, coords_b, match_method):
    """Per-residue RMSD over the matched atoms, grouped by A's residues."""
    if match_method != "residue":
        return []
    sq = ((coords_b - coords_a) ** 2).sum(axis=1)  # squared deviation per matched atom
    out, start = [], 0
    rids = [mol_a.residue_id(int(i)) for i in idx_a]
    n = len(rids)
    for k in range(1, n + 1):
        if k == n or rids[k] != rids[start]:
            group = sq[start:k]
            out.append(ResidueDeviation(
                label=rids[start].label(),
                n_atoms=len(group),
                rmsd=float(np.sqrt(group.mean())),
            ))
            start = k
    return out


def _contact_delta(mol_a, mol_b, cutoff, method, notes):
    """Residue contact-map difference over the residues both structures share."""
    try:
        cm_a = mol_a.contact_map(cutoff=cutoff, level="residue", method=method)
        cm_b = mol_b.contact_map(cutoff=cutoff, level="residue", method=method)
    except (ValueError, ImportError) as exc:
        notes.append(f"contact-map delta skipped: {exc}")
        return None

    def keymap(cm):
        return {
            (rid.chain, int(rid.resid), rid.insertion_code): i
            for i, rid in enumerate(cm.residue_ids)
        }

    ka, kb = keymap(cm_a), keymap(cm_b)
    common = [k for k in ka if k in kb]  # preserve A's residue order
    if len(common) < 2:
        notes.append("contact-map delta skipped: fewer than two common residues")
        return None
    ia = [ka[k] for k in common]
    ib = [kb[k] for k in common]
    a = cm_a.matrix[np.ix_(ia, ia)] > 0
    b = cm_b.matrix[np.ix_(ib, ib)] > 0
    triu = np.triu_indices(len(common), k=1)
    a_up, b_up = a[triu], b[triu]
    return ContactDelta(
        cutoff=cutoff,
        method=method,
        n_common_residues=len(common),
        n_contacts_a=int(a_up.sum()),
        n_contacts_b=int(b_up.sum()),
        gained=int((b_up & ~a_up).sum()),
        lost=int((a_up & ~b_up).sum()),
    )


def _descriptor_delta(mol_a, mol_b, preset, notes):
    """Per-feature B-minus-A delta over the descriptors both structures expose."""
    try:
        from .descriptors import flatten_descriptors

        da = flatten_descriptors(mol_a.descriptors(preset=preset))
        db = flatten_descriptors(mol_b.descriptors(preset=preset))
    except (ValueError, ImportError) as exc:
        notes.append(f"descriptor delta skipped ({preset}): {exc}")
        return []
    return [
        DescriptorDelta(name=k, a=da[k], b=db[k], delta=db[k] - da[k])
        for k in da if k in db
    ]


def _fmt(value) -> str:
    fval = float(value)
    if fval == int(fval) and abs(fval) < 1e15:
        return str(int(fval))
    return f"{fval:.3f}"
