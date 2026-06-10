"""Tests for the preflight guardrails (`molscope preflight`, `mol.preflight`).

Preflight reads the cheap signals MolScope already computes and turns the ones
that silently degrade descriptor / graph / coarse-grain output into explicit,
workflow-scoped warnings. These tests check that the right findings fire for
representative inputs (a crystal PDB with no hydrogens / inferred bonds, a
metadata-free `.xyz`, a deliberately huge structure), that workflow scoping and
the opt-in method integration work, and that a clean structure stays quiet.
"""

import os

import numpy as np
import pytest

import molscope as ms
from molscope.cli import main
from molscope.molecule import Molecule
from molscope.preflight import DENSE_ATOM_WARN, preflight

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")
CRYSTAL = os.path.join(DATA, "1ubq.pdb")  # no H, no CONECT, residue metadata
PROTEIN = os.path.join(DATA, "3ptb.pdb")  # trypsin: residue gaps + non-standard residues
XYZ = os.path.join(DATA, "helix_201.xyz")  # bare coordinates, no metadata


def _clean_molecule():
    """A small molecule with explicit bonds, hydrogens and full metadata."""
    return Molecule(
        coords=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        elements=["C", "H", "H"],
        atom_names=["C1", "H1", "H2"],
        resnames=["LIG", "LIG", "LIG"],
        resids=np.array([1, 1, 1]),
        chains=["A", "A", "A"],
        bond_index=np.array([[0, 1], [0, 2]]),
    )


def test_flags_inferred_bonds_and_missing_hydrogens():
    codes = preflight(CRYSTAL).codes()
    assert "inferred_bonds" in codes  # no CONECT records
    assert "missing_hydrogens" in codes


def test_metadata_free_xyz_flags_metadata_and_elements():
    codes = preflight(XYZ).codes()
    assert "missing_metadata" in codes
    assert "bad_elements" in codes  # the .xyz carries no element column
    assert "inferred_bonds" in codes


def test_unknown_element_symbol_flagged():
    mol = Molecule(coords=np.zeros((2, 3)), elements=["C", "Xx"])
    report = preflight(mol)
    assert "bad_elements" in report.codes()
    assert "unrecognised" in dict(zip(report.codes(), report.messages()))["bad_elements"]


def test_multiple_models_flagged():
    ensemble = os.path.join(os.path.dirname(__file__), "fixtures", "1d3z.pdb.gz")
    codes = preflight(ensemble).codes()
    assert "multiple_models" in codes
    # An NMR ensemble carries hydrogens, so that warning should not fire.
    assert "missing_hydrogens" not in codes


def test_clean_structure_is_ok():
    report = preflight(_clean_molecule())
    assert report.ok
    assert report.codes() == []
    assert "no preflight warnings" in report.summary()


def test_workflow_scoping():
    # inferred_bonds is a graph/CG concern; missing_hydrogens touches descriptors.
    graph = preflight(CRYSTAL, workflow="graph").codes()
    descriptors = preflight(CRYSTAL, workflow="descriptors").codes()
    assert "inferred_bonds" in graph
    assert "inferred_bonds" not in descriptors
    assert "missing_hydrogens" in descriptors


def test_large_dense_matrix_warning_is_graph_and_contact_scoped():
    n = DENSE_ATOM_WARN + 1
    big = Molecule(coords=np.zeros((n, 3)), elements=["C"] * n)
    assert "large_dense" in preflight(big, workflow="contact_map").codes()
    assert "large_dense" in preflight(big, workflow="graph").codes()
    assert "large_dense" not in preflight(big, workflow="descriptors").codes()


def test_deep_adds_topology_warnings_and_charge_note():
    report = preflight(PROTEIN, deep=True)
    # 3ptb (trypsin) has residue-numbering gaps and non-standard residues.
    assert any(c in report.codes()
               for c in ("residue_gaps", "nonstandard_residues",
                         "missing_backbone", "chain_breaks"))
    assert any("net formal charge" in n for n in report.notes)
    # Without --deep those topology checks do not run.
    assert "residue_gaps" not in preflight(PROTEIN, deep=False).codes()


def test_deep_on_molecule_notes_path_requirement():
    report = preflight(_clean_molecule(), deep=True)
    assert any("need a file path" in n for n in report.notes)


def test_invalid_workflow_raises():
    with pytest.raises(ValueError, match="workflow must be"):
        preflight(CRYSTAL, workflow="nonsense")


def test_to_dict_shape():
    d = preflight(CRYSTAL, workflow="graph").to_dict()
    assert d["workflow"] == "graph"
    assert d["ok"] is False
    assert all({"code", "message", "workflows"} <= set(w) for w in d["warnings"])


# -- method integration -----------------------------------------------------

def test_molecule_preflight_method():
    mol = ms.read(CRYSTAL)
    assert "inferred_bonds" in mol.preflight(workflow="graph").codes()


def test_to_graph_preflight_emits_warning():
    mol = ms.read(CRYSTAL)
    with pytest.warns(UserWarning, match="preflight: bonds were inferred"):
        mol.to_graph(preflight=True)


def test_descriptors_preflight_emits_but_returns_same_values():
    mol = ms.read(CRYSTAL)
    baseline = mol.descriptors()
    with pytest.warns(UserWarning, match="preflight:"):
        checked = mol.descriptors(preflight=True)
    assert checked == baseline


def test_coarse_grain_preflight_emits_warning():
    mol = ms.read(CRYSTAL)
    with pytest.warns(UserWarning, match="preflight:"):
        mol.coarse_grain(preflight=True)


# -- CLI --------------------------------------------------------------------

def test_cli_preflight_prints_summary(capsys):
    rc = main(["preflight", CRYSTAL])
    assert rc == 0
    out = capsys.readouterr().out
    assert "preflight warning(s)" in out
    assert "inferred from interatomic distances" in out


def test_cli_preflight_json_and_workflow(capsys):
    import json

    rc = main(["preflight", CRYSTAL, "--workflow", "graph", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workflow"] == "graph"
    assert "inferred_bonds" in [w["code"] for w in payload["warnings"]]


def test_cli_preflight_shallow_skips_topology(capsys):
    rc = main(["preflight", CRYSTAL, "--shallow"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "net formal charge" not in out


def test_cli_analyze_preflight_warns_on_stderr(tmp_path, capsys):
    rc = main(["analyze", XYZ, "--out", str(tmp_path / "a.csv"), "--preflight"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "preflight:" in err and "missing per-atom metadata" in err


def test_cli_coarse_grain_preflight_warns(tmp_path, capsys):
    rc = main(["coarse-grain", CRYSTAL, "--preflight", "--out", str(tmp_path / "cg.pdb")])
    assert rc == 0
    assert "preflight:" in capsys.readouterr().err


def test_cli_preflight_missing_file_returns_2(tmp_path, capsys):
    rc = main(["preflight", str(tmp_path / "nope.pdb")])
    assert rc == 2
    assert "preflight failed" in capsys.readouterr().err
