"""One-command structure report: QC, chains/ligands, descriptors, contacts, graph.

:func:`build_report` reads a structure and gathers the headline outputs from
MolScope's existing analyses into a single :class:`StructureReportData`;
:func:`render_html` and :func:`render_markdown` turn that into a self-contained
report. This module is glue over :func:`molscope.quality_report`,
:func:`molscope.prepare_structure`, :meth:`Molecule.descriptors`,
:meth:`Molecule.contact_map`, :meth:`Molecule.to_graph` and
:meth:`Molecule.coarse_grain` — it adds no new analysis, just a cohesive summary
that makes the individual tools feel like one product.

Embedded figures (the contact-map heatmap and the optional coarse-grained
preview) are base64 ``data:`` URIs, so the HTML report is a single portable file
with no sidecar images. Everything runs on the bare NumPy + Matplotlib install;
sections whose inputs are missing (e.g. a residue contact map for a small
molecule with no residues) are skipped with a short note rather than failing.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from html import escape
from typing import Optional


@dataclass
class ContactMapInfo:
    """Headline contact-map statistics plus an embedded heatmap image."""

    level: str
    cutoff: float
    n_labels: int
    n_contacts: int
    contact_order: float
    image_uri: Optional[str] = None


@dataclass
class GraphInfo:
    """Headline molecular-graph statistics."""

    n_nodes: int
    n_edges: int
    avg_degree: float


@dataclass
class CoarseGrainInfo:
    """Coarse-grained preview: mapping, coverage and an embedded 3D image."""

    mapping: str
    coverage: str
    n_beads: int
    n_dropped: int
    image_uri: Optional[str] = None


@dataclass
class StructureReportData:
    """Everything :func:`build_report` gathered, ready for a renderer.

    Each optional section is ``None`` (with a matching entry in :attr:`notes`)
    when its inputs were missing or the analysis did not apply, so a renderer can
    show what it has without special-casing every structure type.
    """

    name: str
    source: str
    version: str
    summary_line: str
    quality: object  # QualityReport
    prep: object = None  # StructureReport | None
    ligands: list = field(default_factory=list)  # list[LigandResidue]
    descriptor_preset: str = ""
    descriptors: dict = field(default_factory=dict)
    contact: Optional[ContactMapInfo] = None
    graph: Optional[GraphInfo] = None
    coarse_grain: Optional[CoarseGrainInfo] = None
    notes: list = field(default_factory=list)


# -- gathering --------------------------------------------------------------

def _figure_data_uri(fig, *, dpi: int = 110) -> Optional[str]:
    """Render a Matplotlib figure to a base64 ``data:`` PNG URI, then close it."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_report(
    source: str,
    *,
    name: Optional[str] = None,
    descriptor_preset: str = "native-basic",
    include_contact_map: bool = True,
    contact_cutoff: float = 8.0,
    contact_level: str = "residue",
    contact_method: str = "ca",
    coarse_grain: Optional[str] = None,
) -> StructureReportData:
    """Read ``source`` once and gather a :class:`StructureReportData`.

    ``source`` is a path to a structure file (``.pdb`` / ``.cif`` / ``.xyz`` /
    ``.sdf``, optionally gzipped). ``descriptor_preset`` is one of the
    descriptor presets (``"native-basic"``, ``"native-3d"``, ``"rdkit-basic"``).
    A residue-level contact map is embedded when ``include_contact_map`` is true
    and the structure carries residues; pass ``coarse_grain`` (a mapping name) to
    add a coarse-grained preview. Sections that do not apply are left ``None``
    and explained in :attr:`StructureReportData.notes`.
    """
    from . import __version__
    from .io import read
    from .quality import quality_report
    from .structure_prep import prepare_structure

    notes: list[str] = []

    quality = quality_report(source)
    mol = read(source)

    prep = None
    try:
        prep = prepare_structure(source)
    except (OSError, ValueError, ImportError) as exc:
        notes.append(f"ML-readiness check skipped: {exc}")

    try:
        ligands = mol.ligands()
    except (ValueError, AttributeError):
        ligands = []

    descriptors: dict = {}
    try:
        from .descriptors import flatten_descriptors

        descriptors = flatten_descriptors(mol.descriptors(preset=descriptor_preset))
    except (ValueError, ImportError) as exc:
        notes.append(f"descriptors skipped ({descriptor_preset}): {exc}")

    contact = _build_contact(
        mol, quality, include_contact_map,
        contact_cutoff, contact_level, contact_method, notes,
    )
    graph = _build_graph(mol, notes)
    cg = _build_coarse_grain(mol, coarse_grain, notes) if coarse_grain else None

    title = name or mol.name or os.path.basename(str(source)) or "structure"
    return StructureReportData(
        name=title,
        source=str(source),
        version=__version__,
        summary_line=mol.summary(),
        quality=quality,
        prep=prep,
        ligands=ligands,
        descriptor_preset=descriptor_preset,
        descriptors=descriptors,
        contact=contact,
        graph=graph,
        coarse_grain=cg,
        notes=notes,
    )


def _build_contact(mol, quality, include, cutoff, level, method, notes):
    if not include:
        return None
    if level == "residue" and quality.n_residues < 2:
        notes.append("contact map skipped: fewer than two residues")
        return None
    try:
        cmap = mol.contact_map(cutoff=cutoff, level=level, method=method)
    except (ValueError, ImportError) as exc:
        notes.append(f"contact map skipped: {exc}")
        return None
    image_uri = None
    try:
        ax = cmap.plot(show=False)
        image_uri = _figure_data_uri(ax.figure)
    except Exception as exc:  # pragma: no cover - plotting backend variance
        notes.append(f"contact-map image skipped: {exc}")
    return ContactMapInfo(
        level=cmap.level,
        cutoff=cmap.cutoff,
        n_labels=len(cmap.labels),
        n_contacts=cmap.n_contacts,
        contact_order=cmap.contact_order(),
        image_uri=image_uri,
    )


def _build_graph(mol, notes):
    try:
        graph = mol.to_graph()
    except (ValueError, ImportError) as exc:
        notes.append(f"graph stats skipped: {exc}")
        return None
    n_nodes = graph.n_atoms
    n_edges = graph.n_bonds
    avg_degree = (2.0 * n_edges / n_nodes) if n_nodes else 0.0
    return GraphInfo(n_nodes=n_nodes, n_edges=n_edges, avg_degree=avg_degree)


def _build_coarse_grain(mol, mapping, notes):
    try:
        beads, report = mol.coarse_grain(mapping=mapping, return_report=True)
    except (ValueError, ImportError) as exc:
        notes.append(f"coarse-grain preview skipped ({mapping}): {exc}")
        return None
    image_uri = None
    try:
        ax = beads.plot(color_by="chain", show=False)
        image_uri = _figure_data_uri(ax.figure)
    except Exception as exc:  # pragma: no cover - plotting backend variance
        notes.append(f"coarse-grain image skipped: {exc}")
    return CoarseGrainInfo(
        mapping=mapping,
        coverage=report.coverage(),
        n_beads=report.n_beads,
        n_dropped=report.n_dropped,
        image_uri=image_uri,
    )


# -- HTML rendering ---------------------------------------------------------

_CSS = """
 body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
        max-width: 960px; color: #222; line-height: 1.45; padding: 0 1rem; }
 h1 { margin-bottom: 0.2rem; } h2 { margin-top: 2rem; border-bottom: 1px solid #eee;
        padding-bottom: 0.2rem; }
 .sub { color: #666; margin-top: 0; }
 .verdict { display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.85rem; font-weight: 600; }
 .ok { background: #e6f4ea; color: #137333; }
 .warn { background: #fef7e0; color: #946c00; }
 .bad { background: #fce8e6; color: #c5221f; }
 table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
 th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #eee; }
 th { background: #f7f7f9; }
 ul.stats { list-style: none; padding: 0; display: flex; gap: 1.5rem; flex-wrap: wrap; }
 ul.stats li { background: #f7f7f9; border-radius: 6px; padding: 6px 12px; }
 img.fig { max-width: 560px; width: 100%; border: 1px solid #eee; border-radius: 6px; }
 .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 2px 1.5rem; font-size: 0.88rem; }
 .grid div { display: flex; justify-content: space-between; border-bottom: 1px solid #f3f3f5;
        padding: 2px 0; }
 .grid .k { color: #555; } .grid .v { font-family: ui-monospace, monospace; }
 .note { color: #777; font-size: 0.85rem; }
 code { background: #f3f3f5; padding: 0 3px; border-radius: 3px; }
 footer { margin-top: 2.5rem; color: #888; font-size: 0.8rem; border-top: 1px solid #eee;
        padding-top: 1rem; }
"""


def _verdict_badge(label: str, css_class: str) -> str:
    return f'<span class="verdict {css_class}">{escape(label)}</span>'


def _qc_section_html(data: StructureReportData) -> str:
    q = data.quality
    badge = _verdict_badge("clean", "ok") if q.clean else _verdict_badge("issues", "bad")
    stats = [
        f"<li>atoms <b>{q.n_atoms}</b></li>",
        f"<li>chains {', '.join(q.chains) or '(none)'}</li>",
        f"<li>residues {q.n_residues}</li>",
        f"<li>bonds {q.n_bonds} ({q.bond_source})</li>",
    ]
    if q.n_waters or q.n_ions:
        stats.append(f"<li>waters {q.n_waters} &middot; ions {q.n_ions}</li>")
    parts = [f"<h2>Quality control</h2><p>{badge}</p>",
             f'<ul class="stats">{"".join(stats)}</ul>']
    if q.issues:
        items = "".join(f"<li>{escape(i)}</li>" for i in q.issues)
        parts.append(f"<p class=\"note\">Issues:</p><ul>{items}</ul>")

    prep = data.prep
    if prep is not None:
        if prep.ml_ready and not prep.warnings:
            pbadge = _verdict_badge("ML-ready", "ok")
        elif prep.ml_ready:
            pbadge = _verdict_badge("ML-ready (with warnings)", "warn")
        else:
            pbadge = _verdict_badge("not ML-ready", "bad")
        parts.append(f"<h3>ML readiness {pbadge}</h3>")
        if prep.net_charge is not None:
            parts.append(
                f'<p class="note">Net formal charge '
                f'<b>{prep.net_charge:+d}</b> ({escape(prep.charge_method)})</p>'
            )
        for label, items in (("Blockers", prep.blockers), ("Warnings", prep.warnings)):
            if items:
                lis = "".join(f"<li>{escape(i)}</li>" for i in items)
                parts.append(f"<p class=\"note\">{label}:</p><ul>{lis}</ul>")
    return "".join(parts)


def _ligands_section_html(data: StructureReportData) -> str:
    parts = ["<h2>Chains &amp; ligands</h2>"]
    chains = data.quality.chains
    parts.append(
        f'<p>{len(chains)} chain(s): {escape(", ".join(chains)) or "(none)"}</p>'
    )
    if not data.ligands:
        parts.append('<p class="note">No ligands detected (water and ions excluded).</p>')
        return "".join(parts)
    rows = []
    for lig in data.ligands:
        rid = lig.residue_id
        rows.append(
            "<tr>"
            f"<td>{escape(rid.resname)}</td><td>{escape(str(rid.chain))}</td>"
            f"<td>{rid.resid}{escape(rid.insertion_code)}</td><td>{len(lig)}</td>"
            "</tr>"
        )
    parts.append(
        "<table><thead><tr><th>resname</th><th>chain</th><th>resid</th>"
        f"<th>atoms</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )
    return "".join(parts)


def _descriptors_section_html(data: StructureReportData) -> str:
    if not data.descriptors:
        return ""
    cells = "".join(
        f'<div><span class="k">{escape(k)}</span>'
        f'<span class="v">{_fmt_num(v)}</span></div>'
        for k, v in data.descriptors.items()
    )
    return (
        f"<h2>Descriptors <span class=\"sub\">({escape(data.descriptor_preset)}, "
        f"{len(data.descriptors)} features)</span></h2>"
        f'<div class="grid">{cells}</div>'
    )


def _contact_section_html(data: StructureReportData) -> str:
    c = data.contact
    if c is None:
        return ""
    img = f'<img class="fig" alt="contact map" src="{c.image_uri}">' if c.image_uri else ""
    return (
        "<h2>Contact map</h2>"
        f'<ul class="stats"><li>level {escape(c.level)}</li>'
        f"<li>cutoff {c.cutoff:g} &#8491;</li><li>nodes {c.n_labels}</li>"
        f"<li>contacts <b>{c.n_contacts}</b></li>"
        f"<li>relative contact order {c.contact_order:.3f}</li></ul>"
        f"{img}"
    )


def _graph_section_html(data: StructureReportData) -> str:
    g = data.graph
    if g is None:
        return ""
    return (
        "<h2>Molecular graph</h2>"
        f'<ul class="stats"><li>nodes <b>{g.n_nodes}</b></li>'
        f"<li>edges <b>{g.n_edges}</b></li>"
        f"<li>mean degree {g.avg_degree:.2f}</li></ul>"
    )


def _coarse_grain_section_html(data: StructureReportData) -> str:
    cg = data.coarse_grain
    if cg is None:
        return ""
    img = f'<img class="fig" alt="coarse-grained beads" src="{cg.image_uri}">' if cg.image_uri else ""
    return (
        f"<h2>Coarse-grained preview <span class=\"sub\">({escape(cg.mapping)})</span></h2>"
        f"<p>{escape(cg.coverage)}</p>{img}"
    )


def render_html(data: StructureReportData) -> str:
    """Assemble a self-contained HTML report from a :class:`StructureReportData`.

    Pure string assembly: no file I/O, no extra dependencies. Embedded figures
    are inline ``data:`` URIs, so the returned string is a single portable file.
    """
    title = escape(data.name)
    notes_html = ""
    if data.notes:
        items = "".join(f"<li>{escape(n)}</li>" for n in data.notes)
        notes_html = f'<h2>Notes</h2><ul class="note">{items}</ul>'
    body = "".join([
        f"<h1>Structure report</h1>",
        f'<p class="sub">{title} &middot; {escape(data.summary_line)}</p>',
        _qc_section_html(data),
        _ligands_section_html(data),
        _descriptors_section_html(data),
        _contact_section_html(data),
        _graph_section_html(data),
        _coarse_grain_section_html(data),
        notes_html,
        f'<footer>Generated by MolScope {escape(data.version)} from '
        f'<code>{escape(os.path.basename(data.source))}</code>. '
        f'Verdicts are triage heuristics, not a substitute for inspection.</footer>',
    ])
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>MolScope report: {title}</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>\n"
    )


# -- Markdown rendering -----------------------------------------------------

def render_markdown(data: StructureReportData) -> str:
    """Assemble a Markdown report from a :class:`StructureReportData`.

    Embedded figures are dropped (Markdown has no portable inline-image story);
    the contact-map and coarse-grain sections keep their statistics.
    """
    q = data.quality
    lines = [
        f"# Structure report: {data.name}",
        "",
        f"_{data.summary_line}_",
        "",
        "## Quality control",
        "",
        f"- Verdict: **{'clean' if q.clean else 'issues found'}**",
        f"- Atoms: **{q.n_atoms}**",
        f"- Chains: {', '.join(q.chains) or '(none)'}",
        f"- Residues: {q.n_residues}",
        f"- Bonds: {q.n_bonds} ({q.bond_source})",
    ]
    if q.n_waters or q.n_ions:
        lines.append(f"- Waters: {q.n_waters} / Ions: {q.n_ions}")
    if q.issues:
        lines += ["", "### Issues", ""] + [f"- {i}" for i in q.issues]

    prep = data.prep
    if prep is not None:
        lines += ["", "### ML readiness", "",
                  f"- Verdict: **{'ML-ready' if prep.ml_ready else 'NOT ML-ready'}**"]
        if prep.net_charge is not None:
            lines.append(f"- Net formal charge: **{prep.net_charge:+d}** ({prep.charge_method})")
        if prep.blockers:
            lines += ["", "**Blockers**", ""] + [f"- {b}" for b in prep.blockers]
        if prep.warnings:
            lines += ["", "**Warnings**", ""] + [f"- {w}" for w in prep.warnings]

    lines += ["", "## Chains & ligands", "",
              f"- Chains ({len(q.chains)}): {', '.join(q.chains) or '(none)'}"]
    if data.ligands:
        lines += ["", f"### Ligands ({len(data.ligands)})", "",
                  "| resname | chain | resid | atoms |", "| --- | --- | --- | --- |"]
        for lig in data.ligands:
            rid = lig.residue_id
            lines.append(
                f"| {rid.resname} | {rid.chain} | {rid.resid}{rid.insertion_code} | {len(lig)} |"
            )
    else:
        lines.append("- No ligands detected (water and ions excluded).")

    if data.descriptors:
        lines += ["", f"## Descriptors ({data.descriptor_preset}, "
                  f"{len(data.descriptors)} features)", "",
                  "| feature | value |", "| --- | --- |"]
        lines += [f"| {k} | {_fmt_num(v)} |" for k, v in data.descriptors.items()]

    c = data.contact
    if c is not None:
        lines += ["", "## Contact map", "",
                  f"- Level: {c.level} (cutoff {c.cutoff:g} Å)",
                  f"- Nodes: {c.n_labels}",
                  f"- Contacts: **{c.n_contacts}**",
                  f"- Relative contact order: {c.contact_order:.3f}"]

    g = data.graph
    if g is not None:
        lines += ["", "## Molecular graph", "",
                  f"- Nodes: **{g.n_nodes}**",
                  f"- Edges: **{g.n_edges}**",
                  f"- Mean degree: {g.avg_degree:.2f}"]

    cg = data.coarse_grain
    if cg is not None:
        lines += ["", f"## Coarse-grained preview ({cg.mapping})", "",
                  f"- {cg.coverage}"]

    if data.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in data.notes]

    lines += ["", f"_Generated by MolScope {data.version}. Verdicts are triage "
              "heuristics, not a substitute for inspection._", ""]
    return "\n".join(lines)


def _fmt_num(value) -> str:
    """Compact numeric formatting for tables: ints stay ints, floats get 3 dp."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return escape(str(value))
    if fval == int(fval) and abs(fval) < 1e15:
        return str(int(fval))
    return f"{fval:.3f}"
