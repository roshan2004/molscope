"""Tests for the shared, pipeline-friendly CLI output shapes.

Every `--json` command wraps its payload in one envelope (tool/version/command/
input/parser/backends/warnings/result), and the batch commands (`analyze`,
`export`) write a run manifest with feature names and skipped inputs. These tests
cover the envelope helpers directly and assert the shapes end-to-end through the
CLI.
"""

import json
import os

from molscope.cli import main
from molscope.cli_output import (
    backends_since,
    envelope,
    parser_for_inputs,
    parser_name,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def test_parser_name_maps_extensions():
    assert parser_name("x.pdb") == "pdb"
    assert parser_name("x.pdb.gz") == "pdb"
    assert parser_name("x.cif") == "cif"
    assert parser_name("x.sdf") == "sdf"
    assert parser_name("x.txt") is None


def test_parser_for_inputs_uniform_vs_mixed():
    assert parser_for_inputs(["a.pdb", "b.pdb"]) == "pdb"
    assert parser_for_inputs(["a.pdb", "b.xyz"]) == "mixed"
    assert parser_for_inputs(["a.txt"]) is None


def test_envelope_shape_and_extras():
    env = envelope(
        "demo", source="a.pdb", parser="pdb", backends=["scipy"],
        warnings=["w"], result={"k": 1}, feature_names=["f"],
    )
    assert env["tool"] == "molscope"
    assert env["command"] == "demo"
    assert env["input"] == "a.pdb"
    assert env["parser"] == "pdb"
    assert env["backends"] == ["scipy"]
    assert env["warnings"] == ["w"]
    assert env["result"] == {"k": 1}
    assert env["feature_names"] == ["f"]  # extra merged at top level
    assert "version" in env


def test_envelope_omits_result_when_none():
    assert "result" not in envelope("demo", source="x")


def test_backends_since_reports_only_new_imports():
    import sys

    before = frozenset(sys.modules)
    # Nothing imported since the snapshot.
    assert backends_since(before) == []
    # A backend present before the snapshot is not reported as "new".
    fake = before | {"scipy"}
    assert "scipy" not in backends_since(fake)


# -- end-to-end envelope through the CLI ------------------------------------

def _json_out(capsys, *argv):
    rc = main(list(argv))
    assert rc == 0
    return json.loads(capsys.readouterr().out)


def test_qc_json_is_enveloped(capsys):
    payload = _json_out(capsys, "qc", os.path.join(DATA, "3ptb.pdb"), "--json")
    assert payload["command"] == "qc"
    assert payload["parser"] == "pdb"
    assert isinstance(payload["backends"], list)
    assert payload["result"]["n_atoms"] == 1701


def test_preflight_json_surfaces_warnings_at_envelope_level(capsys):
    payload = _json_out(capsys, "preflight", os.path.join(DATA, "1ubq.pdb"),
                        "--workflow", "graph", "--json")
    assert payload["command"] == "preflight"
    # The standardized envelope warnings mirror the preflight messages.
    assert any("inferred" in w for w in payload["warnings"])


# -- batch run manifests ----------------------------------------------------

def test_analyze_manifest_records_features_and_skips(tmp_path, capsys):
    out = tmp_path / "f.csv"
    manifest = tmp_path / "run.json"
    rc = main([
        "analyze",
        os.path.join(DATA, "3ptb.pdb"),
        str(tmp_path / "missing.pdb"),
        "--out", str(out), "--manifest", str(manifest),
    ])
    assert rc == 0  # at least one structure succeeded
    m = json.loads(manifest.read_text())
    assert m["command"] == "analyze"
    assert m["n_inputs"] == 2 and m["n_written"] == 1
    assert "n_atoms" in m["feature_names"]
    assert "file" not in m["feature_names"]
    assert [s["input"] for s in m["skipped"]] == [str(tmp_path / "missing.pdb")]
    assert m["output"] == str(out)


def test_analyze_manifest_written_even_when_all_skipped(tmp_path):
    manifest = tmp_path / "run.json"
    rc = main([
        "analyze", str(tmp_path / "nope.pdb"),
        "--out", str(tmp_path / "f.csv"), "--manifest", str(manifest),
    ])
    assert rc == 1  # nothing produced
    m = json.loads(manifest.read_text())
    assert m["n_written"] == 0
    assert m["feature_names"] == []
    assert len(m["skipped"]) == 1


def test_export_manifest_records_feature_names(tmp_path):
    manifest = tmp_path / "run.json"
    rc = main([
        "export", os.path.join(DATA, "1ubq.pdb"),
        "--to", "nx", "--out-dir", str(tmp_path / "g"), "--manifest", str(manifest),
    ])
    assert rc == 0
    m = json.loads(manifest.read_text())
    assert m["command"] == "export"
    assert m["to"] == "nx"
    assert m["n_written"] == 1
    assert "node" in m["feature_names"] and "edge" in m["feature_names"]
    assert "atomic_number" in m["feature_names"]["node"]


def test_export_manifest_records_skips(tmp_path):
    manifest = tmp_path / "run.json"
    rc = main([
        "export", os.path.join(DATA, "1ubq.pdb"), str(tmp_path / "missing.pdb"),
        "--to", "nx", "--out-dir", str(tmp_path / "g"), "--manifest", str(manifest),
    ])
    assert rc == 0
    m = json.loads(manifest.read_text())
    assert m["n_inputs"] == 2 and m["n_written"] == 1
    assert [s["input"] for s in m["skipped"]] == [str(tmp_path / "missing.pdb")]
