"""Tests for static-structure comparison (`molscope compare`).

The comparison matches atoms by residue identity so it works on two *different*
files, not just two frames of one trajectory. These tests drive it with two
models of an NMR ensemble (same topology, a real conformational difference) and
with deliberately mismatched inputs to exercise the common-residue restriction,
the index-matching fallback, and the error paths.
"""

import os

import pytest

import molscope as ms
from molscope import compare_structures
from molscope.cli import main
from molscope.io import read_pdb_models, write_pdb

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
ENSEMBLE = os.path.join(FIXTURES, "1d3z.pdb.gz")  # NMR ubiquitin, 10 models
XYZ = os.path.join(DATA, "helix_201.xyz")


@pytest.fixture(scope="module")
def models():
    return read_pdb_models(ENSEMBLE)


def test_two_models_align_with_real_difference(models):
    result = compare_structures(models[0], models[1])

    assert result.match_method == "residue"
    assert result.n_matched_atoms == len(models[0])
    assert result.n_common_residues > 0
    # Distinct conformers: a non-trivial RMSD that alignment reduces.
    assert result.rmsd > 0.1
    assert result.rmsd < result.rmsd_unaligned
    assert result.per_residue and all(d.rmsd >= 0 for d in result.per_residue)
    assert result.contact is not None
    assert any(d.delta != 0 for d in result.descriptors)


def test_identical_structure_is_zero(models):
    result = compare_structures(models[0], models[0])

    assert result.rmsd == pytest.approx(0.0, abs=1e-6)
    assert all(d.rmsd == pytest.approx(0.0, abs=1e-6) for d in result.per_residue)
    assert result.contact.gained == 0 and result.contact.lost == 0
    assert result.n_changed_descriptors == 0


def test_ca_matching_is_one_atom_per_residue(models):
    result = compare_structures(models[0], models[1], atoms="ca")

    assert result.atom_set == "ca"
    assert result.n_matched_atoms == result.n_common_residues
    assert all(d.n_atoms == 1 for d in result.per_residue)


def test_differing_atom_counts_restrict_to_common_residues(models):
    # B keeps only residues 1-40, so only those residues can be matched.
    partial = models[1].select(resid=(1, 40))
    result = compare_structures(models[0], partial, atoms="ca")

    assert result.n_atoms_a == len(models[0])
    assert result.n_atoms_b == len(partial)
    assert result.n_common_residues == 40
    assert result.n_matched_atoms == 40


def test_backbone_matching_uses_backbone_atoms(models):
    result = compare_structures(models[0], models[1], atoms="backbone")

    assert result.atom_set == "backbone"
    # Up to four backbone atoms (N, CA, C, O) per residue, more than one.
    assert result.n_matched_atoms > result.n_common_residues
    assert all(d.n_atoms <= 4 for d in result.per_residue)


def test_no_superpose_reports_raw_rmsd(models):
    result = compare_structures(models[0], models[1], superpose=False)

    assert result.superposed is False
    assert result.rmsd == result.rmsd_unaligned
    assert "no superposition" in result.summary()


def test_no_common_atoms_raises(models):
    # Disjoint residue ranges share no (chain, resid, atom name) keys.
    a = models[0].select(resid=(1, 30))
    b = models[1].select(resid=(40, 76))
    with pytest.raises(ValueError, match="no atoms could be matched"):
        compare_structures(a, b)


def test_contact_delta_skipped_with_too_few_common_residues(models):
    result = compare_structures(models[0], models[1].select(resid=1))

    assert result.contact is None
    assert any("common residue" in n for n in result.notes)


def test_index_fallback_for_metadata_free_structures():
    result = compare_structures(XYZ, XYZ)

    assert result.match_method == "index"
    assert result.rmsd == pytest.approx(0.0, abs=1e-6)
    assert any("atom index" in n for n in result.notes)


def test_index_fallback_requires_equal_atom_counts():
    a = ms.read(XYZ)
    b = a.take(range(len(a) - 1))  # one fewer atom, no residue metadata
    with pytest.raises(ValueError, match="different atom counts"):
        compare_structures(a, b)


def test_ca_without_atom_names_raises():
    with pytest.raises(ValueError, match="atom names"):
        compare_structures(XYZ, XYZ, atoms="ca")


def test_bad_atom_set_raises(models):
    with pytest.raises(ValueError, match="atoms must be"):
        compare_structures(models[0], models[1], atoms="sidechain")


def test_to_dict_and_summary_shapes(models):
    result = compare_structures(models[0], models[1])
    d = result.to_dict()

    assert d["name_a"] == result.name_a
    assert len(d["per_residue"]) == len(result.per_residue)
    assert d["contact"]["n_common_residues"] == result.contact.n_common_residues
    assert "vs" in result.summary()
    assert "# Structure comparison" in result.report_markdown()


# -- CLI --------------------------------------------------------------------

@pytest.fixture
def model_files(tmp_path, models):
    a = tmp_path / "a.pdb"
    b = tmp_path / "b.pdb"
    write_pdb(models[0], str(a))
    write_pdb(models[1], str(b))
    return str(a), str(b)


def test_cli_compare_prints_summary(model_files, capsys):
    a, b = model_files
    rc = main(["compare", a, b])
    assert rc == 0
    out = capsys.readouterr().out
    assert "RMSD" in out and "contact map" in out


def test_cli_compare_writes_markdown(model_files, tmp_path):
    a, b = model_files
    out = tmp_path / "cmp.md"
    rc = main(["compare", a, b, "--atoms", "ca", "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "# Structure comparison" in text
    assert "## Per-residue deviations" in text


def test_cli_compare_json(model_files, capsys):
    import json

    a, b = model_files
    rc = main(["compare", a, b, "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "compare"
    assert payload["input"] == [a, b]
    assert payload["result"]["match_method"] == "residue"
    assert payload["result"]["rmsd"] >= 0


def test_cli_compare_no_superpose_and_no_contact_map(model_files, capsys):
    a, b = model_files
    rc = main(["compare", a, b, "--no-superpose", "--no-contact-map"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no superposition" in out
    assert "contact map" not in out


def test_cli_compare_missing_file_returns_2(tmp_path, capsys):
    rc = main(["compare", str(tmp_path / "nope.pdb"), str(tmp_path / "x.pdb")])
    assert rc == 2
    assert "compare failed" in capsys.readouterr().err
