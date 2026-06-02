"""Tests for CLI argument helpers."""

import os

import pytest

import molscope as ms
from molscope.cli import (
    _default_to_view,
    _parse_ligand,
    _parse_selection,
    _parse_selection_value,
    _write_structure,
    main,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def test_default_to_view_keeps_subcommands_and_top_level_help():
    subcommands = {"view", "analyze", "binding-site", "export"}

    assert _default_to_view(["analyze", "a.pdb"], subcommands) == ["analyze", "a.pdb"]
    assert _default_to_view(["binding-site", "a.pdb"], subcommands) == [
        "binding-site",
        "a.pdb",
    ]
    assert _default_to_view(["--help"], subcommands) == ["--help"]


def test_default_to_view_accepts_leading_view_options():
    subcommands = {"view", "analyze", "export"}

    assert _default_to_view(["--fetch", "1aml"], subcommands) == ["view", "--fetch", "1aml"]


def test_parse_selection_accepts_single_key_value():
    assert _parse_selection("atom_name=CA") == {"atom_name": "CA"}


def test_parse_selection_accepts_and_expression():
    assert _parse_selection("chain=A and atom_name=CA") == {
        "chain": "A",
        "atom_name": "CA",
    }


def test_parse_selection_accepts_repeated_flags():
    assert _parse_selection(["chain=A", "atom_name=CA"]) == {
        "chain": "A",
        "atom_name": "CA",
    }


def test_parse_selection_coerces_resid_and_hetero_values():
    assert _parse_selection("resid=10-20 and hetero=false") == {
        "resid": (10, 20),
        "hetero": False,
    }


def test_parse_selection_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unsupported field"):
        _parse_selection("name=CA")


def test_parse_ligand_accepts_resname_and_location():
    assert _parse_ligand(None) is None
    assert _parse_ligand("BEN") == "BEN"
    assert _parse_ligand("A:1") == ("A", 1)


def test_parse_ligand_rejects_bad_location():
    with pytest.raises(ValueError, match="integer resid"):
        _parse_ligand("A:BEN")


def test_parse_selection_residue_id_variants():
    assert _parse_selection("residue_id=A:100") == {"residue_id": ("A", 100)}
    assert _parse_selection("residue_id=A:100:B") == {"residue_id": ("A", 100, "B")}
    assert _parse_selection("residue_id=A:100:B:THR") == {
        "residue_id": ("A", 100, "B", "THR"),
    }


def test_parse_selection_residue_id_rejects_bad_values():
    with pytest.raises(ValueError, match="chain:resid"):
        _parse_selection("residue_id=A")
    with pytest.raises(ValueError, match="integer resid"):
        _parse_selection("residue_id=A:BEN")


def test_parse_ligand_accepts_icode_and_resname():
    assert _parse_ligand("A:10:B") == ("A", 10, "B")
    assert _parse_ligand("A:10:B:LIG") == ("A", 10, "B", "LIG")


def test_parse_ligand_rejects_too_many_parts():
    with pytest.raises(ValueError, match="chain:resid"):
        _parse_ligand("A:10:B:LIG:extra")


def test_default_to_view_defaults_when_empty():
    assert _default_to_view([], {"view", "analyze"}) == ["view"]


def test_parse_selection_skips_empty_clauses():
    # An empty --select value (append yields a list) contributes no clause.
    assert _parse_selection(["chain=A", "   "]) == {"chain": "A"}


def test_parse_selection_requires_key_equals_value():
    with pytest.raises(ValueError, match="not key=value"):
        _parse_selection("chain")
    with pytest.raises(ValueError, match="not key=value"):
        _parse_selection("chain=")
    with pytest.raises(ValueError, match="not key=value"):
        _parse_selection("=A")


def test_parse_selection_rejects_duplicate_field():
    with pytest.raises(ValueError, match="more than once"):
        _parse_selection("chain=A and chain=B")


def test_parse_selection_rejects_empty_selection():
    with pytest.raises(ValueError, match="selection is empty"):
        _parse_selection(" ")


def test_parse_selection_value_resid_forms():
    assert _parse_selection_value("resid", "10") == 10
    assert _parse_selection_value("resid", "10:20") == (10, 20)
    assert _parse_selection_value("resid", "10-20") == (10, 20)
    with pytest.raises(ValueError, match="integer or inclusive range"):
        _parse_selection_value("resid", "abc")


def test_parse_selection_value_hetero_forms():
    assert _parse_selection_value("hetero", "true") is True
    assert _parse_selection_value("hetero", "false") is False
    with pytest.raises(ValueError, match="true/false"):
        _parse_selection_value("hetero", "maybe")


def test_parse_selection_value_strips_matching_quotes():
    assert _parse_selection_value("resname", "'HOH'") == "HOH"


# -- coarse-grain subcommand ------------------------------------------------


def test_write_structure_dispatches_by_extension(tmp_path):
    mol = ms.read(os.path.join(DATA, "1fqy.pdb"))
    cg = mol.coarse_grain("residue_com")
    for ext in (".pdb", ".cif", ".xyz"):
        out = tmp_path / f"cg{ext}"
        _write_structure(cg, str(out))
        assert out.exists() and out.stat().st_size > 0


def test_write_structure_rejects_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="unsupported output extension"):
        _write_structure(ms.read(os.path.join(DATA, "1fqy.pdb")), str(tmp_path / "x.foo"))


def test_coarse_grain_cli_writes_pdb_with_conect(tmp_path):
    out = tmp_path / "cg.pdb"
    rc = main(["coarse-grain", os.path.join(DATA, "1fqy.pdb"),
               "--mapping", "martini", "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    # martini beads are named BB/SC and the bond network is written as CONECT
    assert "BB " in text and "SC " in text
    assert "CONECT" in text


def test_coarse_grain_cli_summary_only_without_out():
    assert main(["coarse-grain", os.path.join(DATA, "1fqy.pdb"),
                 "--mapping", "residue_centroid"]) == 0


def test_coarse_grain_cli_bad_extension_returns_error(tmp_path):
    assert main(["coarse-grain", os.path.join(DATA, "1fqy.pdb"),
                 "--out", str(tmp_path / "cg.foo")]) == 2
