"""Unit tests for the generated validation summary (`_summary`).

These run in the normal suite (no reference tools needed): they exercise the
pure aggregation/formatting that turns recorded pytest outcomes into the JSON
and Markdown summaries the CI validation job publishes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _summary  # noqa: E402  (sibling module, made importable above)

Outcome = _summary.Outcome

_SAMPLE = [
    Outcome("test_geometry_ref", "test_rg", "passed"),
    Outcome("test_geometry_ref", "test_com", "passed"),
    Outcome("test_dssp_ref", "test_helix", "passed"),
    Outcome("test_pocket_interactions_ref", "(module skipped)", "skipped", "PLIP not installed"),
    Outcome("test_bonds_ref", "test_stretched", "failed", "recall 0.5 < 0.98"),
]


def test_totals_counts_each_outcome():
    t = _summary.totals(_SAMPLE)
    assert t == {"passed": 3, "skipped": 1, "failed": 1, "total": 5}


def test_summarize_groups_by_module_with_labels_and_order():
    rows = _summary.summarize(_SAMPLE)
    by_module = {r["module"]: r for r in rows}
    geo = by_module["test_geometry_ref"]
    assert geo["reference"] == "MDAnalysis"
    assert geo["passed"] == 2 and geo["failed"] == 0
    # Curated areas sort by their declared order (geometry before dssp).
    modules = [r["module"] for r in rows]
    assert modules.index("test_geometry_ref") < modules.index("test_dssp_ref")


def test_area_for_falls_back_for_unregistered_modules():
    assert _summary.area_for("test_geometry_ref")[1] == "MDAnalysis"
    label, ref = _summary.area_for("test_newthing_ref")
    assert label == "newthing ref" and ref == "—"


def test_to_json_shape():
    payload = _summary.to_json(_SAMPLE, version="9.9.9", generated_at="2026-01-01")
    assert payload["tool"] == "molscope"
    assert payload["version"] == "9.9.9"
    assert payload["totals"]["total"] == 5
    assert all({"area", "reference", "passed", "skipped", "failed", "checks"} <= set(a)
               for a in payload["areas"])


def test_to_markdown_lists_counts_failures_and_skips():
    md = _summary.to_markdown(_SAMPLE, version="9.9.9", generated_at="2026-01-01")
    assert "# MolScope validation summary" in md
    assert "5 checks: 3 passed, 1 skipped, 1 failed" in md
    assert "| Area | Reference | Passed | Skipped | Failed |" in md
    # Failures and skips get their own detail sections with reasons.
    assert "## Failures" in md
    assert "recall 0.5 < 0.98" in md
    assert "## Skipped checks" in md
    assert "PLIP not installed" in md


def test_to_markdown_clean_run_has_no_failure_section():
    clean = [Outcome("test_invariants", "test_kabsch", "passed")]
    md = _summary.to_markdown(clean, version="1.0", generated_at=None)
    assert "## Failures" not in md
    assert "1 checks: 1 passed, 0 skipped, 0 failed" in md
