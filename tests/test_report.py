"""Tests for the one-command structure report (`molscope report`).

The report is glue over existing analyses, so these tests check that each
section is gathered and rendered, that embedded figures make the HTML
self-contained, and that sections whose inputs are missing (a residue contact
map for a residue-less ``.xyz``) are skipped with a note rather than failing.
"""

import os

import pytest

from molscope import build_report
from molscope.cli import main
from molscope.report import render_html, render_markdown

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")
PROTEIN = os.path.join(DATA, "3ptb.pdb")  # trypsin + BEN ligand
XYZ = os.path.join(DATA, "helix_201.xyz")  # no residue metadata


@pytest.fixture(autouse=True)
def _agg_backend():
    import matplotlib

    matplotlib.use("Agg")


def test_build_report_gathers_all_sections():
    data = build_report(PROTEIN, coarse_grain="residue_com")

    assert data.name == "3ptb"
    assert data.quality.n_atoms == 1701
    assert data.prep is not None and data.prep.ml_ready
    # The BEN ligand is detected, waters and ions excluded.
    assert [lig.residue_id.resname for lig in data.ligands] == ["BEN"]
    assert data.descriptors  # native-basic preset is non-empty
    assert data.descriptor_preset == "native-basic"
    assert data.contact is not None and data.contact.n_contacts > 0
    assert data.graph is not None and data.graph.n_nodes == 1701
    assert data.coarse_grain is not None and data.coarse_grain.n_beads > 0


def test_render_html_is_self_contained():
    data = build_report(PROTEIN)
    html = render_html(data)

    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    # The contact-map heatmap is embedded inline, not a sidecar file.
    assert "data:image/png;base64," in html
    for heading in ("Quality control", "Chains &amp; ligands", "Descriptors",
                    "Contact map", "Molecular graph"):
        assert heading in html
    assert "BEN" in html  # ligand row


def test_render_markdown_has_sections():
    data = build_report(PROTEIN)
    md = render_markdown(data)

    assert md.startswith("# Structure report: 3ptb")
    for heading in ("## Quality control", "## Chains & ligands", "## Descriptors",
                    "## Contact map", "## Molecular graph"):
        assert heading in md
    assert "| BEN |" in md


def test_residueless_structure_skips_contact_map():
    data = build_report(XYZ)

    assert data.contact is None
    assert any("contact map skipped" in n for n in data.notes)
    # The graph and descriptors still work without residues.
    assert data.graph is not None and data.graph.n_nodes == 201
    assert data.descriptors


def test_coarse_grain_only_when_requested():
    assert build_report(PROTEIN).coarse_grain is None
    assert build_report(PROTEIN, coarse_grain="residue_com").coarse_grain is not None


def test_cli_report_writes_html(tmp_path):
    rc = main(["report", PROTEIN, "--out-dir", str(tmp_path)])
    assert rc == 0
    out = tmp_path / "report.html"
    assert out.exists()
    assert "<!DOCTYPE html>" in out.read_text()


def test_cli_report_both_formats_and_coarse_grain(tmp_path):
    rc = main([
        "report", PROTEIN, "--out-dir", str(tmp_path), "--format", "both",
        "--coarse-grain", "--cg-mapping", "residue_com",
    ])
    assert rc == 0
    assert (tmp_path / "report.html").exists()
    md = (tmp_path / "report.md").read_text()
    assert "## Coarse-grained preview" in md


def test_cli_report_no_contact_map(tmp_path):
    rc = main([
        "report", PROTEIN, "--out-dir", str(tmp_path), "--format", "md",
        "--no-contact-map",
    ])
    assert rc == 0
    assert "## Contact map" not in (tmp_path / "report.md").read_text()


def test_cli_report_missing_file_returns_2(tmp_path, capsys):
    rc = main(["report", str(tmp_path / "nope.pdb"), "--out-dir", str(tmp_path)])
    assert rc == 2
    assert "report failed" in capsys.readouterr().err
