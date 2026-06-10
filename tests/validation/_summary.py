"""Format a validation summary from collected pytest outcomes.

These are pure, dependency-free helpers so the formatting can be unit-tested
without running the reference-tool-dependent validation suite. The live
pass / skip / fail status is supplied by the ``conftest`` plugin from the actual
run, so the published summary is produced by the tests themselves and cannot
drift from them.

Only the *labels* in :data:`AREAS` are authored here; every count in the output
comes from the run.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Validation module stem -> (area label, reference it is checked against). New
#: ``*_ref`` modules are picked up automatically with a fallback label; listing
#: them here just gives a nicer area name and orders the table.
AREAS = {
    "test_invariants": ("Physical invariants (alignment, geometry, contacts)",
                        "none (math / conservation)"),
    "test_geometry_ref": ("Geometry, RMSD, distances, inertia", "MDAnalysis"),
    "test_ensembles_ref": ("NMR ensembles (RMSF / RMSD)", "MDAnalysis"),
    "test_bonds_ref": ("Bond perception from geometry", "RDKit"),
    "test_chem_ref": ("Chemical features and descriptors", "RDKit"),
    "test_esol_ref": ("Descriptors and dataset prep at scale", "RDKit (ESOL, 1128 mols)"),
    "test_dssp_ref": ("Secondary structure (simplified DSSP)", "mkdssp"),
    "test_template_bonds_ref": ("Protein template bonds and protonation",
                                "known residue chemistry"),
    "test_docking_ref": ("Docking triage (SDF read, ranking)", "RDKit"),
    "test_binding_sites_ref": ("Binding-site residues and pocket descriptors",
                               "real PDB structures"),
    "test_pocket_interactions_ref": ("Pocket interaction calling", "PLIP"),
    "test_graph_invariants": ("Molecular graphs and dataset assembly",
                              "none (graph invariants)"),
}

_ORDER = list(AREAS)
_OUTCOMES = ("passed", "skipped", "failed")


@dataclass(frozen=True)
class Outcome:
    """One recorded test result: ``module`` is the file stem, ``outcome`` is one
    of ``passed`` / ``skipped`` / ``failed``."""

    module: str
    name: str
    outcome: str
    reason: str = ""


def area_for(module: str):
    """Return ``(area, reference)`` for a module stem, with a sensible fallback."""
    if module in AREAS:
        return AREAS[module]
    label = module.removeprefix("test_").replace("_", " ")
    return (label or module, "—")


def summarize(outcomes) -> list:
    """Group outcomes by module into ordered per-area rows with counts."""
    groups: dict = {}
    for o in outcomes:
        area, ref = area_for(o.module)
        g = groups.setdefault(o.module, {
            "module": o.module, "area": area, "reference": ref,
            "passed": 0, "skipped": 0, "failed": 0, "checks": [],
        })
        if o.outcome in _OUTCOMES:
            g[o.outcome] += 1
        g["checks"].append({"name": o.name, "outcome": o.outcome, "reason": o.reason})

    def sort_key(g):
        return (_ORDER.index(g["module"]) if g["module"] in _ORDER else len(_ORDER), g["module"])

    return sorted(groups.values(), key=sort_key)


def totals(outcomes) -> dict:
    """Overall pass / skip / fail counts."""
    out = {"passed": 0, "skipped": 0, "failed": 0}
    for o in outcomes:
        if o.outcome in out:
            out[o.outcome] += 1
    out["total"] = len(outcomes)
    return out


def to_json(outcomes, *, version: str, generated_at=None) -> dict:
    """A machine-readable summary (stable shape for tooling / CI artifacts)."""
    return {
        "tool": "molscope",
        "version": version,
        "generated_at": generated_at,
        "totals": totals(outcomes),
        "areas": summarize(outcomes),
    }


def to_markdown(outcomes, *, version: str, generated_at=None) -> str:
    """A human-readable Markdown summary for docs / the CI run page."""
    t = totals(outcomes)
    areas = summarize(outcomes)
    stamp = f" on {generated_at}" if generated_at else ""
    lines = [
        "# MolScope validation summary",
        "",
        f"_Generated from the validation suite{stamp}, molscope {version}._",
        "",
        f"**{t['total']} checks: {t['passed']} passed, {t['skipped']} skipped, "
        f"{t['failed']} failed.**",
        "",
        "Tier-1 invariants need no external tools; Tier-2 checks compare MolScope "
        "against a reference implementation and are skipped when that tool is not "
        "installed. Every count below is produced by the run itself, so it stays "
        "honest as the tests evolve.",
        "",
        "| Area | Reference | Passed | Skipped | Failed |",
        "| --- | --- | --: | --: | --: |",
    ]
    for g in areas:
        lines.append(
            f"| {g['area']} | {g['reference']} | {g['passed']} | "
            f"{g['skipped']} | {g['failed']} |"
        )

    failures = [(g, c) for g in areas for c in g["checks"] if c["outcome"] == "failed"]
    if failures:
        lines += ["", "## Failures", ""]
        for g, c in failures:
            tail = f": {c['reason']}" if c["reason"] else ""
            lines.append(f"- `{g['module']}::{c['name']}` (vs {g['reference']}){tail}")

    skips = [(g, c) for g in areas for c in g["checks"] if c["outcome"] == "skipped"]
    if skips:
        lines += ["", "## Skipped checks", "",
                  "_Typically a reference tool is not installed, or an opt-in "
                  "remote-download panel is disabled._", ""]
        for g, c in skips:
            tail = f": {c['reason']}" if c["reason"] else ""
            lines.append(f"- `{g['module']}::{c['name']}`{tail}")

    lines.append("")
    return "\n".join(lines)
