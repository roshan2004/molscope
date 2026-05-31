"""Post-docking triage: summarise, diversify, and rank docking hits.

Docking tools (AutoDock Vina, Gnina, Smina) emit a multi-record SDF with one
record per pose and the score(s) carried as ``> <tag>`` data fields. This module
turns that file into the things a chemist actually needs after a virtual screen:
a ranked summary table, a diverse subset of representatives (so a shortlist is
not 50 near-identical analogues), and a transparent consensus ranking when more
than one scoring function was run.

The core readers and the summary plot need only the base install (NumPy +
Matplotlib). SMILES perception, fingerprints, clustering, and MW/logP need RDKit
(``pip install "molscope[chem]"``); those entry points raise a clear error when
it is missing rather than failing deep inside RDKit.

Nothing here invents a "true" score. Consensus ranking is rank aggregation over
the score fields you point it at, and it reports exactly which fields and
directions it used. Treat it as a triage heuristic, not ground truth.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .io import _open, _parse_sdf_record, _stem
from .molecule import Molecule

# Known docking score fields and whether a larger value means a better hit.
# Affinities/energies (kcal/mol) are negative and lower-is-better; Gnina's CNN
# outputs are predicted-pK / pose-quality and higher-is-better. Matching is
# case-insensitive on the tag name; unknown fields default to lower-is-better
# (the docking convention) and that assumption is reported to the user.
_HIGHER_IS_BETTER = {
    "cnnscore", "cnnaffinity", "cnn_vs", "cnn", "cnnpose", "rfscore", "nnscore",
    "probability", "confidence", "gnina",
}
_LOWER_IS_BETTER = {
    "minimizedaffinity", "affinity", "docking_score", "score", "vina", "smina",
    "r_i_docking_score", "energy", "deltag", "dg", "glide_gscore", "chemplp", "plp",
}


@dataclass
class Pose:
    """One docking pose: its parsed molecule plus the raw SDF record text.

    ``block`` is the verbatim record (without the trailing ``$$$$``), so a subset
    of poses can be re-emitted as a faithful SDF without an RDKit round-trip.
    """

    index: int                       # 1-based record number in the source file
    name: str
    molecule: Molecule
    block: str
    source: str = ""                 # stem of the file this pose came from

    @property
    def properties(self) -> dict[str, str]:
        return self.molecule.properties

    def score(self, field_name: str) -> float:
        """Return the named data field as a float, or NaN if absent/non-numeric."""
        return _to_float(self.molecule.properties.get(field_name))

    def n_heavy_atoms(self) -> int:
        return sum(1 for e in self.molecule.elements if e and e.upper() != "H")


def stream_poses(path: str) -> Generator[Pose, None, None]:
    """Yield every record of an SDF into :class:`Pose` objects (raw block kept).

    Malformed records are skipped rather than aborting the whole file.
    """
    stem = _stem(path)
    record = [0]

    def _emit(lines):
        record[0] += 1
        try:
            mol, _ = _parse_sdf_record(
                lines, 0, path, f"{stem}#{record[0]}", record_no=record[0]
            )
        except ValueError:
            return None
        return Pose(
            index=record[0], name=mol.name, molecule=mol,
            block="".join(lines).rstrip("\n"), source=stem,
        )

    with _open(path) as f:
        current_lines = []
        for line in f:
            if line.rstrip() == "$$$$":
                if current_lines:
                    pose = _emit(current_lines)
                    if pose is not None:
                        yield pose
                    current_lines = []
            elif line.strip() or current_lines:
                # Skip blank padding before a record's title; keep blanks once a
                # block has started (a V2000 comment line is often blank).
                current_lines.append(line)
        if any(ln.strip() for ln in current_lines):
            pose = _emit(current_lines)
            if pose is not None:
                yield pose


class PoseStream:
    """A reusable stream of docking poses that reads from disk on iteration.

    This avoids loading all poses into memory, keeping the memory footprint O(1)
    with respect to the number of poses in the file.
    """

    def __init__(self, path: str):
        self.path = os.fspath(path)

    def __iter__(self) -> Generator[Pose, None, None]:
        return stream_poses(self.path)


def read_poses(path: str) -> list[Pose]:
    """Read every record of an SDF into :class:`Pose` objects (raw block kept).

    Malformed records are skipped rather than aborting the whole file. Raises
    :class:`ValueError` if no record could be read.
    """
    poses = list(stream_poses(path))
    if not poses:
        raise ValueError(f"{path}: no readable records in SDF")
    return poses


# -- score-field discovery --------------------------------------------------

def available_fields(poses: Iterable[Pose]) -> list[str]:
    """Union of all data-field names present across the poses, first-seen order."""
    seen: dict[str, None] = {}
    for pose in poses:
        for key in pose.molecule.properties:
            seen.setdefault(key, None)
    return list(seen)


def resolve_score_field(poses: Iterable[Pose], field_name: Optional[str]) -> str:
    """Return the score field to use, auto-detecting a known one if not given.

    Raises a helpful :class:`ValueError` listing the available fields when the
    requested field is absent or none of the known docking fields are present.
    """
    fields = available_fields(poses)
    if field_name is not None:
        if field_name in fields:
            return field_name
        listed = ", ".join(fields) if fields else "(none)"
        raise ValueError(
            f"score field {field_name!r} not found in SDF; available fields: {listed}"
        )
    lower = {f.lower(): f for f in fields}
    for known in (*_LOWER_IS_BETTER, *_HIGHER_IS_BETTER):
        if known in lower:
            return lower[known]
    listed = ", ".join(fields) if fields else "(none)"
    raise ValueError(
        "could not auto-detect a docking score field; pass --score-field. "
        f"available fields: {listed}"
    )


def higher_is_better(
    field_name: str,
    *,
    higher: Optional[set] = None,
    lower: Optional[set] = None,
) -> tuple[bool, bool]:
    """Decide a field's direction. Returns ``(higher_is_better, was_assumed)``.

    Explicit ``higher``/``lower`` overrides win, then the known-field tables,
    then a reported assumption of lower-is-better (the docking convention).
    """
    key = field_name.lower()
    if higher and (field_name in higher or key in {h.lower() for h in higher}):
        return True, False
    if lower and (field_name in lower or key in {ll.lower() for ll in lower}):
        return False, False
    if key in _HIGHER_IS_BETTER:
        return True, False
    if key in _LOWER_IS_BETTER:
        return False, False
    return False, True


# -- feature 1: summary -----------------------------------------------------

@dataclass
class SummaryResult:
    rows: list[dict]                 # one dict per pose, sorted best-first
    score_field: str
    higher_is_better: bool
    direction_assumed: bool
    scores: np.ndarray               # finite scores in ranked order, for plotting
    n_missing: int                   # poses dropped for a missing/non-numeric score
    with_smiles: bool
    n_poses: int = 0                 # total poses read (before any best-pose collapse)


def summarize(
    poses: Iterable[Pose],
    score_field: str,
    *,
    higher_is_better_flag: bool,
    direction_assumed: bool = False,
    with_smiles: bool = True,
    best_pose_per_ligand: bool = False,
) -> SummaryResult:
    """Rank poses by ``score_field`` and build summary rows.

    SMILES, when requested and RDKit is available, are perceived from each pose's
    3D coordinates and explicit bond orders. If RDKit is missing the SMILES
    column is left blank and ``with_smiles`` is reported as ``False``.
    """
    smiles_fn = _smiles_perceiver() if with_smiles else None
    rows, missing = [], 0
    for pose in poses:
        value = pose.score(score_field)
        if math.isnan(value):
            missing += 1
            continue
        heavy = pose.n_heavy_atoms()
        rows.append({
            "pose_id": pose.index,
            "name": pose.name,
            "smiles": smiles_fn(pose.molecule) if smiles_fn else "",
            "score": value,
            "n_heavy_atoms": heavy,
            "ligand_efficiency": _ligand_efficiency(value, heavy, higher_is_better_flag),
        })
    rows.sort(key=lambda r: r["score"], reverse=higher_is_better_flag)
    n_poses = len(rows) + missing                # total read, before any collapse

    if best_pose_per_ligand:
        rows = _keep_best_pose_per_ligand(rows)

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    ordered = ["rank", "pose_id", "name", "smiles", "score",
               "ligand_efficiency", "n_heavy_atoms"]
    rows = [{k: row[k] for k in ordered} for row in rows]
    return SummaryResult(
        rows=rows, score_field=score_field, higher_is_better=higher_is_better_flag,
        direction_assumed=direction_assumed,
        scores=np.array([r["score"] for r in rows], dtype=float),
        n_missing=missing, with_smiles=smiles_fn is not None, n_poses=n_poses,
    )


# Explicit pose-suffix conventions used by docking tools (e.g. "lig_pose1").
# Deliberately conservative: only strip clear pose markers, never a generic
# trailing number, so a real name like "NAD-2" is not mangled. Many engines
# (Gnina, Smina) instead repeat the exact ligand name across poses, which this
# collapses naturally without any suffix stripping.
_POSE_SUFFIX = re.compile(r"(_pose\d+|_conf\d+|_model\d+|_raw)$", re.IGNORECASE)


def _keep_best_pose_per_ligand(rows: list[dict]) -> list[dict]:
    """Keep one row per compound (the first, i.e. best after sorting).

    Compounds are keyed by SMILES when available (robust), else by name with any
    explicit pose suffix removed.
    """
    kept, seen = [], set()
    for row in rows:
        key = row["smiles"] or _POSE_SUFFIX.sub("", row["name"] or "")
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept


def _draw_histogram(scores: np.ndarray, score_field: str):
    """Build a score-distribution figure, or ``None`` when matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a core dependency
        return None, None
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(scores):
        bins = min(40, max(10, len(scores) // 5))
        ax.hist(scores, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_xlabel(score_field)
    ax.set_ylabel("poses")
    ax.set_title(f"Score distribution ({len(scores)} poses)")
    fig.tight_layout()
    return fig, plt


def plot_score_distribution(scores: np.ndarray, score_field: str, path: str) -> bool:
    """Write a histogram of the docking scores to ``path``. False if no matplotlib."""
    fig, plt = _draw_histogram(scores, score_field)
    if fig is None:
        return False
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def histogram_data_uri(scores: np.ndarray, score_field: str) -> Optional[str]:
    """Render the score histogram as a base64 ``data:`` PNG URI for inline HTML."""
    import base64
    import io

    fig, plt = _draw_histogram(scores, score_field)
    if fig is None:  # pragma: no cover - matplotlib is a core dependency
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# -- feature 2: diversity-aware selection -----------------------------------

@dataclass
class DiverseResult:
    selected: list[dict]             # chosen representatives, best-score-first
    n_pool: int                      # poses considered (after --top)
    n_clusters: int
    requested: int                   # --select value
    threshold: float                 # Tanimoto similarity cutoff used
    score_field: str
    capped_below_request: bool       # True when fewer clusters than requested
    n_failed_fp: int = 0             # Number of poses where fingerprint perception failed
def select_diverse_hits(
    poses: Iterable[Pose],
    score_field: str,
    *,
    higher_is_better_flag: bool,
    top: int = 500,
    select: int = 50,
    threshold: float = 0.7,
    radius: int = 2,
    n_bits: int = 2048,
) -> DiverseResult:
    """Cluster the top hits by Tanimoto similarity and pick diverse representatives.

    Ranks poses by score, keeps the best ``top``, fingerprints them (Morgan),
    clusters with Butina at a Tanimoto similarity ``threshold``, and returns the
    best-scoring member of each cluster, ordered by score and limited to
    ``select``. Picking one representative per cluster is what stops a shortlist
    from collapsing into many near-identical analogues. Needs RDKit.

    If the chemistry yields fewer clusters than ``select``, every cluster
    representative is returned and ``capped_below_request`` is set so the caller
    can say so rather than silently returning a short list.
    """
    Chem = _require_chem()
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina

    # First pass: gather scores, indices, and names of valid poses
    candidates = []
    for pose in poses:
        val = pose.score(score_field)
        if not math.isnan(val):
            candidates.append((val, pose.index, pose.name))

    if not candidates:
        raise ValueError(f"no poses with a numeric {score_field!r} score to select from")

    candidates.sort(key=lambda x: x[0], reverse=higher_is_better_flag)
    top_candidates = candidates[: top if top and top > 0 else len(candidates)]
    top_indices = {c[1] for c in top_candidates}

    # Second pass: gather full Pose objects for the top candidates
    pool_unordered = []
    for pose in poses:
        if pose.index in top_indices:
            pool_unordered.append(pose)

    # Sort the gathered Pose objects to match the top_candidates ranking order
    pool_by_id = {p.index: p for p in pool_unordered}
    pool = [pool_by_id[idx] for _, idx, _ in top_candidates if idx in pool_by_id]

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, valid = [], []
    for pose in pool:
        fp = _pose_fingerprint(pose.molecule, Chem, gen)
        if fp is not None:
            fps.append(fp)
            valid.append(pose)
    if not fps:
        raise ValueError(
            "could not build fingerprints for any pose (RDKit failed to perceive "
            "the molecules); check the bond orders in the SDF"
        )

    # Butina needs the condensed lower-triangle of the Tanimoto *distance* matrix.
    dists: list[float] = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(
        dists, len(fps), 1.0 - threshold, isDistData=True
    )

    smiles_fn = _smiles_perceiver()
    reps = []
    for cluster_id, members in enumerate(clusters, start=1):
        best = min(
            members,
            key=lambda idx: -valid[idx].score(score_field)
            if higher_is_better_flag else valid[idx].score(score_field),
        )
        pose = valid[best]
        reps.append({
            "pose": pose,
            "row": {
                "pose_id": pose.index,
                "name": pose.name,
                "smiles": smiles_fn(pose.molecule) if smiles_fn else "",
                "score": pose.score(score_field),
                "cluster_id": cluster_id,
                "cluster_size": len(members),
            },
        })
    reps.sort(key=lambda r: r["row"]["score"], reverse=higher_is_better_flag)
    n_clusters = len(reps)
    chosen = reps[:select] if select and select > 0 else reps
    for rank, rep in enumerate(chosen, start=1):
        rep["row"] = {"rank": rank, **rep["row"]}

    n_failed_fp = len(pool) - len(valid)

    return DiverseResult(
        selected=[{"pose": r["pose"], **r["row"]} for r in chosen],
        n_pool=len(valid), n_clusters=n_clusters, requested=select,
        threshold=threshold, score_field=score_field,
        capped_below_request=n_clusters < select,
        n_failed_fp=n_failed_fp,
    )


def write_poses_sdf(poses, path: str) -> None:
    """Write poses back out as a multi-record SDF by re-emitting their raw blocks."""
    with _open(path, "w") as f:
        for pose in poses:
            block = pose.block.rstrip("\n")
            f.write(block + "\n$$$$\n")


def collect_poses(poses: Iterable[Pose], ranked_ids: list[int]) -> list[Pose]:
    """Gather the poses for ``ranked_ids`` (pose indices), preserving that order.

    Streams ``poses`` once, keeping only the requested ids in memory -- so this
    stays O(len(ranked_ids)) even when ``poses`` is a whole-file :class:`PoseStream`.
    """
    wanted = set(ranked_ids)
    by_id = {p.index: p for p in poses if p.index in wanted}
    return [by_id[i] for i in ranked_ids if i in by_id]


def write_rows_csv(path: str, columns: list[str], rows: list[dict]) -> None:
    """Write ``rows`` (list of dicts) to ``path`` as CSV with ``columns`` order."""
    import csv

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


# -- feature 3: consensus ranking -------------------------------------------

@dataclass
class ConsensusResult:
    rows: list[dict]                 # joined + ranked, best consensus first
    columns: list[str]
    score_columns: list[str]         # the per-file score columns aggregated
    directions: dict[str, bool]      # column -> higher_is_better
    assumed: list[str]               # columns whose direction was assumed
    key: str
    n_dropped_filter: int


def consensus_rank(
    pose_sets: list[tuple[str, Iterable[Pose]]],
    *,
    score_fields: Optional[list[str]] = None,
    key: str = "name",
    higher: Optional[set] = None,
    lower: Optional[set] = None,
    mw_max: Optional[float] = None,
    logp_max: Optional[float] = None,
) -> ConsensusResult:
    """Join poses from several files and rank by mean rank across score columns.

    ``pose_sets`` is a list of ``(label, poses)`` (label is usually the file
    stem). Molecules are joined across files by ``key`` (their ``name`` title, or
    perceived ``smiles``). Each score column is ranked by its own direction and
    the consensus is the mean of the available ranks: scale-free and explainable,
    not a blended "true" score. Optional MW/logP ceilings drop rows (RDKit).
    """
    keyer = _row_keyer(key)
    columns_seen: dict[str, None] = {}
    joined: dict[str, dict] = {}
    representative: dict[str, Pose] = {}

    for label, poses in pose_sets:
        fields = score_fields or _detect_fields(poses)
        for pose in poses:
            row_key = keyer(pose)
            if row_key is None:
                continue
            entry = joined.setdefault(row_key, {"key": row_key})
            if mw_max is not None or logp_max is not None:
                representative.setdefault(row_key, pose)
            for fname in fields:
                value = pose.score(fname)
                if math.isnan(value):
                    continue
                col = f"{label}:{fname}" if len(pose_sets) > 1 else fname
                columns_seen.setdefault(col, None)
                # Best (per direction) value wins if a molecule appears twice.
                hib, _ = higher_is_better(fname, higher=higher, lower=lower)
                if col not in entry or _better(value, entry[col], hib):
                    entry[col] = value

    score_columns = list(columns_seen)
    if not score_columns:
        raise ValueError(
            "no numeric score fields found across the inputs; pass --score-fields"
        )

    directions, assumed = {}, []
    for col in score_columns:
        base = col.split(":", 1)[1] if ":" in col else col
        hib, was_assumed = higher_is_better(base, higher=higher, lower=lower)
        directions[col] = hib
        if was_assumed:
            assumed.append(col)

    rows = list(joined.values())
    rows, n_dropped = _apply_property_filters(rows, representative, mw_max, logp_max)

    # Per-column ranks (1 = best), averaged over the columns a row actually has.
    for col in score_columns:
        present = [r for r in rows if col in r]
        present.sort(key=lambda r: r[col], reverse=directions[col])
        for rank, r in enumerate(present, start=1):
            r[f"{col}__rank"] = rank
    for r in rows:
        ranks = [r[f"{col}__rank"] for col in score_columns if f"{col}__rank" in r]
        r["n_scores"] = len(ranks)
        r["consensus_rank"] = float(np.mean(ranks)) if ranks else float("nan")

    rows.sort(key=lambda r: (math.isnan(r["consensus_rank"]), r["consensus_rank"]))
    for final, r in enumerate(rows, start=1):
        r["final_rank"] = final

    columns = (
        ["final_rank", "key", "consensus_rank", "n_scores"]
        + score_columns
        + [f"{c}__rank" for c in score_columns]
    )
    rows = [{c: r.get(c, "") for c in columns} for r in rows]
    return ConsensusResult(
        rows=rows, columns=columns, score_columns=score_columns,
        directions=directions, assumed=assumed, key=key,
        n_dropped_filter=n_dropped,
    )


def _detect_fields(poses: Iterable[Pose]) -> list[str]:
    """Known docking score fields present across the poses, in first-seen order."""
    fields = available_fields(poses)
    lower = {f.lower(): f for f in fields}
    known = [lower[k] for k in (*_LOWER_IS_BETTER, *_HIGHER_IS_BETTER) if k in lower]
    # Preserve file order for ties / unknowns already surfaced by the known set.
    return list(dict.fromkeys(known))


def _apply_property_filters(rows, representative, mw_max, logp_max):
    if mw_max is None and logp_max is None:
        return rows, 0
    descr = _mw_logp_perceiver()
    kept, dropped = [], 0
    for r in rows:
        pose = representative.get(r["key"])
        mw, logp = descr(pose.molecule) if pose is not None else (float("nan"), float("nan"))
        r["mw"], r["logp"] = mw, logp
        if mw_max is not None and not math.isnan(mw) and mw > mw_max:
            dropped += 1
            continue
        if logp_max is not None and not math.isnan(logp) and logp > logp_max:
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


# -- feature 4: static HTML report ------------------------------------------

def molecule_svg(molecule, *, width: int = 220, height: int = 170) -> str:
    """Return an inline 2D depiction of a molecule as SVG markup (``""`` on failure).

    Hydrogens are stripped and 2D coordinates generated for a clean, scannable
    picture. Needs RDKit; returns ``""`` when it is missing or perception fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D

        from .chem import to_rdkit
    except ImportError:
        return ""
    try:
        rdmol, _ = to_rdkit(molecule, sanitize=True)
        rdmol = Chem.RemoveHs(rdmol)
        AllChem.Compute2DCoords(rdmol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(rdmol)
        drawer.FinishDrawing()
    except Exception:
        return ""
    svg = drawer.GetDrawingText()
    # Drop the XML prolog/doctype so the <svg> embeds directly in the page.
    start = svg.find("<svg")
    return svg[start:] if start != -1 else svg


def render_html_report(
    summary: SummaryResult,
    *,
    source_name: str,
    n_poses: int,
    diverse: Optional[DiverseResult] = None,
    table_rows: int = 50,
    poses_file: Optional[str] = None,
) -> str:
    """Assemble a self-contained HTML triage report from already-computed results.

    Pure string assembly over a :class:`SummaryResult` (and optional
    :class:`DiverseResult`): no file I/O, no RDKit requirement. When ``diverse``
    is given and RDKit can draw the molecules, each cluster representative is
    shown as a 2D depiction; otherwise that section is simply omitted.
    """
    from html import escape

    direction = "higher is better" if summary.higher_is_better else "lower is better"
    if summary.direction_assumed:
        direction += " (assumed)"
    scores = summary.scores
    stats = ""
    if len(scores):
        best = scores.max() if summary.higher_is_better else scores.min()
        worst = scores.min() if summary.higher_is_better else scores.max()
        stats = (
            f"<li>best: <b>{best:.3f}</b></li>"
            f"<li>median: {float(np.median(scores)):.3f}</li>"
            f"<li>worst: {worst:.3f}</li>"
        )

    hist = histogram_data_uri(scores, summary.score_field)
    hist_html = (
        f'<img class="hist" alt="score distribution" src="{hist}">' if hist else ""
    )

    head_cols = ["rank", "pose_id", "name", "score", "ligand_efficiency",
                 "n_heavy_atoms", "smiles"]
    header = "".join(f"<th>{escape(c)}</th>" for c in head_cols)
    body_rows = []
    for row in summary.rows[: max(0, table_rows)]:
        cells = [
            str(row["rank"]), str(row["pose_id"]), escape(str(row["name"])),
            f"{row['score']:.3f}", f"{row['ligand_efficiency']:.3f}",
            str(row["n_heavy_atoms"]), escape(str(row["smiles"])),
        ]
        body_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    table_html = (
        f"<table><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )

    clusters_html = ""
    if diverse is not None:
        cards = []
        for rep in diverse.selected:
            svg = molecule_svg(rep["pose"].molecule)
            depiction = svg or '<div class="nosvg">no depiction<br>(needs RDKit)</div>'
            cards.append(
                '<div class="card">'
                f'<div class="svg">{depiction}</div>'
                f'<div class="meta"><b>{escape(str(rep["name"]))}</b><br>'
                f'score {rep["score"]:.3f}<br>'
                f'cluster {rep["cluster_id"]} '
                f'(<span title="molecules this representative stands in for">'
                f'{rep["cluster_size"]} member(s)</span>)</div></div>'
            )
        capped = ""
        if diverse.capped_below_request:
            capped = (
                f'<p class="note">Only {diverse.n_clusters} diverse cluster(s) exist '
                f'at Tanimoto similarity {diverse.threshold:g}, fewer than requested; '
                f'all are shown.</p>'
            )
        clusters_html = (
            f'<h2>Diverse representatives</h2>'
            f'<p>{len(diverse.selected)} representative(s) from {diverse.n_pool} '
            f'top hit(s), one per Tanimoto cluster (similarity '
            f'{diverse.threshold:g}).</p>{capped}'
            f'<div class="grid">{"".join(cards)}</div>'
        )

    poses_note = ""
    if poses_file:
        poses_note = (
            f'<p class="note">Top poses written to <code>{escape(poses_file)}</code> '
            f'— load directly in PyMOL, ChimeraX, or Mol* for 3D inspection.</p>'
        )

    missing_note = ""
    if summary.n_missing:
        missing_note = (
            f'<p class="note">{summary.n_missing} pose(s) had no numeric '
            f'<code>{escape(summary.score_field)}</code> and were skipped.</p>'
        )
    smiles_note = ""
    if not summary.with_smiles:
        smiles_note = (
            '<p class="note">SMILES and 2D depictions need RDKit '
            '(<code>pip install "molscope[chem]"</code>).</p>'
        )

    return _REPORT_TEMPLATE.format(
        title=escape(source_name),
        score_field=escape(summary.score_field),
        direction=escape(direction),
        n_poses=n_poses,
        n_ranked=len(summary.rows),
        stats=stats,
        hist=hist_html,
        n_shown=min(max(0, table_rows), len(summary.rows)),
        table=table_html,
        clusters=clusters_html,
        poses_note=poses_note,
        notes=missing_note + smiles_note,
    )


_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Docking report: {title}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
         max-width: 1000px; color: #222; line-height: 1.45; padding: 0 1rem; }}
 h1 {{ margin-bottom: 0.2rem; }} h2 {{ margin-top: 2rem; }}
 .sub {{ color: #666; margin-top: 0; }}
 ul.stats {{ list-style: none; padding: 0; display: flex; gap: 1.5rem; }}
 img.hist {{ max-width: 520px; width: 100%; border: 1px solid #eee; border-radius: 6px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
 th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid #eee; }}
 th {{ background: #f7f7f9; position: sticky; top: 0; }}
 td:last-child {{ font-family: ui-monospace, monospace; color: #555; font-size: 0.8rem; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
          gap: 1rem; }}
 .card {{ border: 1px solid #eee; border-radius: 8px; padding: 0.6rem; text-align: center; }}
 .card .svg {{ min-height: 170px; display: flex; align-items: center; justify-content: center; }}
 .card .meta {{ font-size: 0.85rem; margin-top: 0.4rem; }}
 .nosvg {{ color: #999; font-size: 0.8rem; }}
 .note {{ color: #777; font-size: 0.85rem; }}
 footer {{ margin-top: 2.5rem; color: #888; font-size: 0.8rem; border-top: 1px solid #eee;
           padding-top: 1rem; }}
 code {{ background: #f3f3f5; padding: 0 3px; border-radius: 3px; }}
</style></head><body>
<h1>Docking report</h1>
<p class="sub">{title} &middot; {n_poses} pose(s) &middot; ranked by
 <code>{score_field}</code> ({direction})</p>
<ul class="stats">{stats}</ul>
{hist}
{notes}
<h2>Top hits <span class="sub">(showing {n_shown} of {n_ranked})</span></h2>
{table}
{clusters}
{poses_note}
<footer>
 Generated by MolScope. Ranking and consensus are transparent triage heuristics,
 not calibrated affinities; validate hits per target before acting on them.
</footer>
</body></html>
"""


# -- shared RDKit-backed helpers (best-effort, degrade without RDKit) --------

def _require_chem():
    from .chem import _require_rdkit
    Chem, _ = _require_rdkit()
    return Chem


def _smiles_perceiver():
    """Return a ``molecule -> SMILES`` function, or ``None`` when RDKit is absent."""
    try:
        from .chem import to_rdkit
    except ImportError:  # pragma: no cover
        return None
    try:
        from rdkit import Chem, RDLogger
    except ImportError:
        return None
    RDLogger.DisableLog("rdApp.*")

    def perceive(molecule) -> str:
        try:
            rdmol, _ = to_rdkit(molecule, sanitize=True)
            return Chem.MolToSmiles(rdmol)
        except Exception:
            return ""

    return perceive


def _mw_logp_perceiver():
    from rdkit.Chem import Descriptors

    from .chem import to_rdkit

    def perceive(molecule) -> tuple[float, float]:
        try:
            rdmol, _ = to_rdkit(molecule, sanitize=True)
            return float(Descriptors.MolWt(rdmol)), float(Descriptors.MolLogP(rdmol))
        except Exception:
            return float("nan"), float("nan")

    return perceive


def _pose_fingerprint(molecule, Chem, gen):
    from .chem import to_rdkit
    try:
        rdmol, _ = to_rdkit(molecule, sanitize=True)
        return gen.GetFingerprint(rdmol)
    except Exception:
        return None


def _row_keyer(key: str):
    if key == "name":
        return lambda pose: pose.name or None
    if key == "smiles":
        smiles_fn = _smiles_perceiver()
        if smiles_fn is None:
            raise ValueError("joining on SMILES needs RDKit; install molscope[chem]")
        return lambda pose: smiles_fn(pose.molecule) or None
    raise ValueError(f"unknown join key {key!r}; use 'name' or 'smiles'")


def _ligand_efficiency(score: float, heavy: int, higher_is_better_flag: bool) -> float:
    """Score per heavy atom. For affinities (lower-is-better) report -score/HAC so
    a more favourable (more negative) binding gives a larger positive efficiency."""
    if not heavy:
        return float("nan")
    return (score / heavy) if higher_is_better_flag else (-score / heavy)


def _better(value: float, current: float, higher_is_better_flag: bool) -> bool:
    return value > current if higher_is_better_flag else value < current


def _to_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return float("nan")
