"""Preflight guardrails for the descriptor / graph / coarse-grain workflows.

The featurisation paths happily run on imperfect input and quietly hand back
degraded results: a structure with no explicit bonds gets a graph whose edges
were *guessed* from interatomic distances, one with no residue numbers silently
loses every residue-level feature, and an 80k-atom structure asked for an
atom-level contact map tries to allocate a 50 GB matrix. :func:`preflight` reads
the cheap signals MolScope already computes (:func:`molscope.quality_report`,
and optionally :func:`molscope.prepare_structure`) and turns them into explicit,
workflow-scoped warnings *before* the expensive step runs.

It changes no results and is opt-in everywhere: call :func:`preflight` directly,
``mol.preflight(workflow=...)``, ``molscope preflight FILE`` on the command line,
or pass ``preflight=True`` to :meth:`Molecule.to_graph` / :meth:`descriptors` /
:meth:`coarse_grain` to have the warnings emitted just before that call does its
work. Everything runs on the bare NumPy install.
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
from typing import Optional

from .molecule import Molecule

#: Workflow names a warning can be scoped to. ``None`` keeps every warning.
WORKFLOWS = ("graph", "descriptors", "coarse_grain", "contact_map")

#: Atom count above which atom-level dense distance work is flagged. A dense
#: ``(N, N)`` float64 matrix at N = 10000 is already 0.8 GB.
DENSE_ATOM_WARN = 10_000


@dataclass(frozen=True)
class PreflightWarning:
    """One preflight finding: a stable ``code``, a human ``message``, and the
    workflows it is relevant to (empty = relevant to all)."""

    code: str
    message: str
    workflows: tuple = ()


@dataclass
class PreflightReport:
    """The result of :func:`preflight`: workflow-scoped warnings for a structure."""

    source: str
    n_atoms: int
    workflow: Optional[str] = None
    warnings: list = field(default_factory=list)  # list[PreflightWarning]
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing was flagged."""
        return not self.warnings

    def codes(self) -> list:
        """The stable warning codes, in report order."""
        return [w.code for w in self.warnings]

    def messages(self) -> list:
        """The human-readable warning messages, in report order."""
        return [w.message for w in self.warnings]

    def to_dict(self) -> dict:
        """JSON-serialisable view."""
        return {
            "source": self.source,
            "n_atoms": self.n_atoms,
            "workflow": self.workflow,
            "ok": self.ok,
            "warnings": [
                {"code": w.code, "message": w.message, "workflows": list(w.workflows)}
                for w in self.warnings
            ],
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        """One-or-more line human-readable summary for CLI and logs."""
        head = f"{self.source}: {self.n_atoms} atoms"
        if self.workflow:
            head += f" (workflow: {self.workflow})"
        if not self.warnings:
            return head + " — no preflight warnings"
        lines = [head + f" — {len(self.warnings)} preflight warning(s):"]
        lines.extend(f"  - {w.message}" for w in self.warnings)
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)

    def emit(self, stacklevel: int = 2) -> PreflightReport:
        """Emit each warning through :mod:`warnings`; returns ``self`` for chaining."""
        for w in self.warnings:
            _warnings.warn(f"preflight: {w.message}", stacklevel=stacklevel)
        return self


def preflight(
    source: str | Molecule,
    *,
    workflow: Optional[str] = None,
    deep: bool = False,
) -> PreflightReport:
    """Inspect ``source`` and return workflow-scoped :class:`PreflightReport`.

    ``source`` is a path to a structure file or an already-read
    :class:`~molscope.Molecule`. ``workflow`` (one of ``"graph"``,
    ``"descriptors"``, ``"coarse_grain"``, ``"contact_map"``) keeps only the
    warnings relevant to that step; ``None`` keeps them all. ``deep`` additionally
    runs :func:`molscope.prepare_structure` for topology checks (chain breaks,
    missing backbone atoms, residue gaps), which needs a file path — when
    ``source`` is an in-memory molecule those checks are skipped with a note.

    Computes nothing expensive and changes no results: it reads the cheap signals
    already produced by :func:`molscope.quality_report` and flags the ones that
    silently degrade descriptor / graph / coarse-grain output.
    """
    if workflow is not None and workflow not in WORKFLOWS:
        raise ValueError(
            f"workflow must be one of {WORKFLOWS} or None, got {workflow!r}"
        )

    from .quality import quality_report

    if isinstance(source, Molecule):
        mol = source
        name = mol.name or "<molecule>"
        path = None
    else:
        from .io import read

        path = str(source)
        name = path
        mol = read(path)

    report_source = quality_report(path) if path is not None else quality_report(mol)

    found: list = []
    notes: list = []
    _quality_warnings(report_source, mol, found)
    if deep and path is not None:
        _deep_warnings(path, found, notes)
    elif deep:
        notes.append(
            "deep topology checks (chain breaks, missing backbone, net charge) "
            "need a file path; pass the file to preflight() for them"
        )

    if workflow is not None:
        found = [w for w in found if not w.workflows or workflow in w.workflows]

    return PreflightReport(
        source=name, n_atoms=report_source.n_atoms,
        workflow=workflow, warnings=found, notes=notes,
    )


# -- checks -----------------------------------------------------------------

def _quality_warnings(q, mol: Molecule, out: list) -> None:
    """Append parse-level warnings derived from a :class:`QualityReport`."""
    if q.bond_source == "inferred":
        out.append(PreflightWarning(
            "inferred_bonds",
            "bonds were inferred from interatomic distances (the file carries no "
            "explicit connectivity); graph edges and CG bonds therefore depend on "
            "the covalent-radius threshold, not the deposited topology",
            ("graph", "coarse_grain"),
        ))

    missing = [m for m in q.missing_metadata
               if m in ("atom names", "residue numbers", "chain identifiers")]
    if missing:
        out.append(PreflightWarning(
            "missing_metadata",
            f"missing per-atom metadata ({', '.join(missing)}); residue-level "
            "contact maps, CA/backbone selections, residue CG mappings and residue "
            "graph features that rely on it are unavailable",
            ("graph", "descriptors", "coarse_grain", "contact_map"),
        ))

    if q.n_atoms and not _has_hydrogens(mol):
        out.append(PreflightWarning(
            "missing_hydrogens",
            "no hydrogen atoms present; hydrogen-dependent descriptors and graph "
            "node features are computed on the heavy-atom skeleton only — add "
            "explicit H (e.g. a protonation step) if your model expects them",
            ("graph", "descriptors"),
        ))

    if q.blank_elements or q.unknown_elements:
        n_unknown = sum(q.unknown_elements.values())
        bits = []
        if q.blank_elements:
            bits.append(f"{q.blank_elements} blank")
        if n_unknown:
            bits.append(f"{n_unknown} unrecognised ({', '.join(sorted(q.unknown_elements))})")
        out.append(PreflightWarning(
            "bad_elements",
            f"{' and '.join(bits)} element symbol(s); element-based features, "
            "masses and geometric bond inference may be wrong for those atoms",
        ))

    if q.altloc_atoms:
        out.append(PreflightWarning(
            "altlocs",
            f"{q.altloc_atoms} atom(s) in alternate conformations; only the primary "
            "altLoc is used, so a model sees a single rotamer",
        ))

    if q.low_occupancy_atoms:
        out.append(PreflightWarning(
            "low_occupancy",
            f"{q.low_occupancy_atoms} atom(s) with occupancy < 1; their coordinates "
            "are less certain than the rest of the model",
        ))

    if q.n_models > 1:
        out.append(PreflightWarning(
            "multiple_models",
            f"the file holds {q.n_models} models; only the first is read — use "
            "read_pdb_models() or the ensemble tools for the rest",
        ))

    if q.n_atoms > DENSE_ATOM_WARN:
        est_gb = (q.n_atoms ** 2) * 8 / 1e9
        out.append(PreflightWarning(
            "large_dense",
            f"{q.n_atoms} atoms: atom-level dense distance work (atom contact maps, "
            f"k-NN / radius / Delaunay graphs, full distance matrices) allocates "
            f"about N² = {q.n_atoms ** 2:.1e} entries (~{est_gb:.1f} GB as "
            "float64); prefer residue-level maps, a distance cutoff with the KD-tree "
            "path, or chunking",
            ("graph", "contact_map"),
        ))


def _deep_warnings(path: str, out: list, notes: list) -> None:
    """Append topology warnings from :func:`prepare_structure` (proteins)."""
    from .structure_prep import prepare_structure

    try:
        prep = prepare_structure(path)
    except (OSError, ValueError, ImportError) as exc:
        notes.append(f"deep topology checks skipped: {exc}")
        return

    if prep.missing_backbone:
        out.append(PreflightWarning(
            "missing_backbone",
            f"{len(prep.missing_backbone)} residue(s) missing backbone atoms; "
            "distance- and graph-based features are distorted at those positions",
            ("graph", "descriptors", "contact_map"),
        ))
    if prep.chain_breaks:
        out.append(PreflightWarning(
            "chain_breaks",
            f"{len(prep.chain_breaks)} backbone chain break(s); through-space edges "
            "may bridge missing density rather than a real contact",
            ("graph", "contact_map"),
        ))
    if prep.residue_gaps:
        out.append(PreflightWarning(
            "residue_gaps",
            f"{len(prep.residue_gaps)} residue-numbering gap(s); sequence-separation "
            "filters and residue indexing assume a continuous chain",
            ("graph", "contact_map"),
        ))
    if prep.truncated_sidechains:
        out.append(PreflightWarning(
            "truncated_sidechains",
            f"{len(prep.truncated_sidechains)} truncated side chain(s); per-residue "
            "shape and atom-count features are understated there",
            ("graph", "descriptors"),
        ))
    if prep.nonstandard_residues:
        out.append(PreflightWarning(
            "nonstandard_residues",
            f"{len(prep.nonstandard_residues)} non-standard residue(s); template "
            "bonds and standard residue features may not apply",
            ("graph", "descriptors", "coarse_grain"),
        ))
    if prep.net_charge is not None:
        notes.append(f"net formal charge {prep.net_charge:+d} ({prep.charge_method})")


def _has_hydrogens(mol: Molecule) -> bool:
    return any(str(e).strip().upper() in ("H", "D") for e in mol.elements)
