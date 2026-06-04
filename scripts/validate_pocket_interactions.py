"""Validate MolScope's pocket-interaction heuristics against PLIP.

MolScope's :meth:`describe_environment` detects ligand-pocket interactions from
*heavy-atom geometry only* (no donor/acceptor typing, bond-angle criterion, or
protonation model). This harness quantifies how that compares to a reference
profiler -- PLIP (Adasme et al., *Nucleic Acids Res.* 2021), which protonates
the complex and applies full geometric criteria -- by reporting per-class
precision / recall / F1 across a panel of protein-ligand complexes.

The comparison is at **residue granularity** (does each tool flag residue X for
interaction class C around the ligand?). Atom-level matching is not meaningful
across two different protonation and atom-naming schemes. Because the two tools
draw the hydrogen-bond / salt-bridge boundary differently (PLIP reports the
benzamidine-Asp189 contact as hydrogen bonds, MolScope as a salt bridge), the
**primary metric is the polar-contact union** (H-bond OR salt bridge); the
finer per-class numbers are reported alongside for transparency.

Expected shape of the result, and why it is the honest story: MolScope's
permissive heavy-atom rules should give **high recall** (it rarely misses a
PLIP-detected contact) with **lower precision** (it over-calls, especially
hydrophobic and aromatic/pi contacts). That trade-off is exactly what the
"treat these as candidates" framing in the feature promises.

PLIP is conda-only in practice; create the reference environment once with::

    conda create -n plip-ref -c conda-forge plip openbabel -y

Then run, e.g.::

    .venv/bin/python scripts/validate_pocket_interactions.py            # local 3ptb only
    .venv/bin/python scripts/validate_pocket_interactions.py --remote   # full panel (downloads)
    .venv/bin/python scripts/validate_pocket_interactions.py --remote --csv pli.csv

Override how PLIP is launched with ``MOLSCOPE_PLIP_CMD`` (default
``"conda run -n plip-ref python"``).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import molscope as ms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
PLIP_REFERENCE = Path(__file__).with_name("_plip_reference.py")

# Reuse the validation binding-site panel: fold- and ligand-diverse complexes.
LOCAL_PANEL = ("3ptb",)
REMOTE_PANEL = ("1stp", "1iep", "3ert", "1hsg", "4hvp", "2br1")

# Pocket selection radius for the comparison. Wider than the 4.5 A default so a
# residue is never excluded merely by the pocket boundary before its per-
# interaction distance threshold gets a chance to judge it (PLIP salt bridges
# reach 5.5 A, pi contacts further).
POCKET_CUTOFF = 6.0

# Report order. "polar" (H-bond union salt bridge) is the headline class.
CLASSES = ("hydrophobic", "hbond", "salt_bridge", "polar", "pi")

ResidueKey = tuple[str, int]


def _default_plip_cmd() -> list[str]:
    raw = os.environ.get("MOLSCOPE_PLIP_CMD", "conda run -n plip-ref python")
    return shlex.split(raw)


def plip_available(plip_cmd: list[str] | None = None) -> bool:
    """True if the reference PLIP environment can be invoked."""
    plip_cmd = plip_cmd or _default_plip_cmd()
    try:
        out = subprocess.run(
            [*plip_cmd, "-c", "import plip; from openbabel import openbabel"],
            capture_output=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0


def _structure_path(pdb_id: str, cache_dir: Path | None) -> str:
    """Local file path for a structure (bundled when available, else fetched).

    Fetches go through MolScope's own download cache (the system temp dir by
    default), so the panel is not re-downloaded across runs.
    """
    bundled = DATA / f"{pdb_id}.pdb"
    if bundled.exists():
        return str(bundled)
    cache = str(cache_dir) if cache_dir else None
    return ms.io.fetch_file(pdb_id, cache_dir=cache)


def molscope_sets(mol, ligand, pocket_cutoff: float = POCKET_CUTOFF) -> dict[str, set[ResidueKey]]:
    """Per-class sets of ``(chain, resid)`` residues MolScope flags for a ligand."""
    env = mol.select_pocket(ligand=ligand, cutoff=pocket_cutoff).environment()
    hydrophobic = {(r.chain, int(r.resid)) for r in env.hydrophobic_residues}
    aromatic = {(r.chain, int(r.resid)) for r in env.aromatic_residues}
    hbond = {(hb.residue.chain, int(hb.residue.resid)) for hb in env.hydrogen_bonds}
    salt = {(sb.residue.chain, int(sb.residue.resid)) for sb in env.salt_bridges}
    return {
        "hydrophobic": hydrophobic,
        "hbond": hbond,
        "salt_bridge": salt,
        "polar": hbond | salt,
        "pi": aromatic,
    }


def plip_sites(
    pdb_path: str, plip_cmd: list[str] | None = None
) -> dict[str, dict[str, set[ResidueKey]]]:
    """Run PLIP and return ``{site_key: {class: {(chain, resid), ...}}}``."""
    plip_cmd = plip_cmd or _default_plip_cmd()
    out = subprocess.run(
        [*plip_cmd, str(PLIP_REFERENCE), pdb_path],
        capture_output=True, text=True, timeout=600,
    )
    if out.returncode != 0:
        raise RuntimeError(f"PLIP reference failed for {pdb_path}:\n{out.stderr[-2000:]}")
    raw = json.loads(out.stdout)

    sites: dict[str, dict[str, set[ResidueKey]]] = {}
    for key, classes in raw["sites"].items():
        hydrophobic = {(c, int(n)) for c, n, _ in classes["hydrophobic"]}
        hbond = {(c, int(n)) for c, n, _ in classes["hbond"]}
        salt = {(c, int(n)) for c, n, _ in classes["salt_bridge"]}
        pi = {(c, int(n)) for c, n, _ in classes["pi"]}
        sites[key] = {
            "hydrophobic": hydrophobic,
            "hbond": hbond,
            "salt_bridge": salt,
            "polar": hbond | salt,
            "pi": pi,
        }
    return sites


def _match_site(sites: dict, ligand) -> dict[str, set[ResidueKey]] | None:
    """Pick the PLIP site matching a MolScope ligand by resname (then chain/resid)."""
    resname = ligand.resname.upper()
    candidates = {k: v for k, v in sites.items() if k.split(":")[0].upper() == resname}
    if not candidates:
        return None
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    want = f"{resname}:{ligand.chain}:{int(ligand.resid)}"
    return candidates.get(want, next(iter(candidates.values())))


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, ms_set: set, ref_set: set) -> None:
        self.tp += len(ms_set & ref_set)
        self.fp += len(ms_set - ref_set)
        self.fn += len(ref_set - ms_set)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p == p and r == r and (p + r) else float("nan")


def evaluate(panel, *, plip_cmd=None, pocket_cutoff=POCKET_CUTOFF, cache_dir=None):
    """Aggregate per-class confusion across a panel; return (totals, per_structure)."""
    plip_cmd = plip_cmd or _default_plip_cmd()

    totals = {c: Confusion() for c in CLASSES}
    per_structure = []
    for pdb_id in panel:
        path = _structure_path(pdb_id, cache_dir)
        mol = ms.read(path)
        ligands = mol.ligands()
        if not ligands:
            per_structure.append((pdb_id, None, "no non-solvent ligand"))
            continue
        ligand = max(ligands, key=len)
        ref = _match_site(plip_sites(path, plip_cmd=plip_cmd), ligand)
        if ref is None:
            per_structure.append((pdb_id, ligand.resname, "no matching PLIP site"))
            continue
        ms_sets = molscope_sets(mol, ligand, pocket_cutoff=pocket_cutoff)
        row = {c: Confusion() for c in CLASSES}
        for c in CLASSES:
            totals[c].add(ms_sets[c], ref[c])
            row[c].add(ms_sets[c], ref[c])
        per_structure.append((pdb_id, ligand.resname, row))
    return totals, per_structure


def _fmt(value: float) -> str:
    return "  -  " if value != value else f"{value:.2f}"


def format_report(totals, per_structure, pocket_cutoff: float) -> str:
    lines = [
        "# MolScope vs PLIP: pocket-interaction agreement",
        "",
        f"Residue-level agreement over {len(per_structure)} complex(es); "
        f"pocket cutoff {pocket_cutoff:.1f} A. PLIP is the reference.",
        "",
        "| class | precision | recall | F1 | TP | FP | FN |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in CLASSES:
        cf = totals[c]
        lines.append(
            f"| {c} | {_fmt(cf.precision)} | {_fmt(cf.recall)} | {_fmt(cf.f1)} "
            f"| {cf.tp} | {cf.fp} | {cf.fn} |"
        )
    lines += ["", "## Per structure (polar-union recall / precision)", ""]
    lines.append("| pdb | ligand | polar P | polar R | hydrophobic P | hydrophobic R |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for pdb_id, ligand, row in per_structure:
        if not isinstance(row, dict):
            lines.append(f"| {pdb_id} | {ligand or '-'} | _{row}_ |  |  |  |")
            continue
        p, h = row["polar"], row["hydrophobic"]
        lines.append(
            f"| {pdb_id} | {ligand} | {_fmt(p.precision)} | {_fmt(p.recall)} "
            f"| {_fmt(h.precision)} | {_fmt(h.recall)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true",
                        help="include the remote panel (downloads from RCSB)")
    parser.add_argument("--pocket-cutoff", type=float, default=POCKET_CUTOFF)
    parser.add_argument("--csv", type=Path, default=None,
                        help="also write the per-class totals as CSV")
    args = parser.parse_args()

    plip_cmd = _default_plip_cmd()
    if not plip_available(plip_cmd):
        raise SystemExit(
            "PLIP reference environment not available. Create it with:\n"
            "  conda create -n plip-ref -c conda-forge plip openbabel -y\n"
            "or set MOLSCOPE_PLIP_CMD to a python that can 'import plip'."
        )

    panel = LOCAL_PANEL + (REMOTE_PANEL if args.remote else ())
    totals, per_structure = evaluate(panel, plip_cmd=plip_cmd, pocket_cutoff=args.pocket_cutoff)
    print(format_report(totals, per_structure, args.pocket_cutoff))

    if args.csv:
        import csv

        with args.csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["class", "precision", "recall", "f1", "tp", "fp", "fn"])
            for c in CLASSES:
                cf = totals[c]
                writer.writerow([c, cf.precision, cf.recall, cf.f1, cf.tp, cf.fp, cf.fn])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
