"""Tests for the lightweight structure-quality report (`molscope qc`)."""

import json
import os

import pytest

import molscope as ms
from molscope import elements
from molscope.cli import main
from molscope.quality import quality_report

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# A PDB with two models' worth of structure trimmed to one, carrying an
# alternate location, a partial occupancy, a HETATM ligand, and a bogus
# element symbol — every parse-quality signal in one tiny file.
WEIRD_PDB = """\
MODEL        1
ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N
ATOM      2  CA AALA A   1      11.639   6.071  -5.147  0.50  0.00           C
ATOM      3  CA BALA A   1      11.700   6.000  -5.100  0.50  0.00           C
HETATM    4  XX  LIG A 100       8.000   3.000  -2.000  1.00  0.00          Xx
ENDMDL
END
"""


@pytest.fixture
def weird_pdb(tmp_path):
    path = tmp_path / "weird.pdb"
    path.write_text(WEIRD_PDB)
    return str(path)


@pytest.fixture
def water_xyz(tmp_path):
    path = tmp_path / "water.xyz"
    path.write_text(
        "3\nwater\nO 0.000 0.000 0.000\nH 0.757 0.586 0.000\nH -0.757 0.586 0.000\n"
    )
    return str(path)


def test_element_symbols_table():
    assert elements.is_element("C")
    assert elements.is_element("cl")  # case-insensitive
    assert elements.is_element("Pt")  # heavy element beyond ATOMIC_NUMBERS
    assert elements.is_element("D")   # deuterium isotope
    assert not elements.is_element("Xx")
    assert not elements.is_element("")
    assert not elements.is_element(None)


def test_quality_report_clean_protein():
    report = quality_report(os.path.join(DATA, "1ubq.pdb"))
    assert report.n_atoms == 660
    assert report.chains == ["A"]
    assert report.fmt == ".pdb"
    assert report.bond_source == "inferred"
    assert report.n_bonds > 0
    assert report.clean
    assert report.issues == []


def test_quality_report_ligand_and_water_inventory():
    report = quality_report(os.path.join(DATA, "3ptb.pdb"))
    assert report.ligands == {"BEN": 1}
    assert report.n_waters > 0
    assert report.n_ions >= 1
    assert report.n_hetero_atoms > 0


def test_quality_report_flags_unknown_element_and_altloc(weird_pdb):
    report = quality_report(weird_pdb)
    # Primary-altloc policy keeps N, CA(A), and the HETATM -> 3 atoms.
    assert report.n_atoms == 3
    assert report.unknown_elements == {"XX": 1}
    assert report.altloc_atoms == 2          # both CA conformers in the file
    assert report.low_occupancy_atoms == 2
    assert report.ligands == {"LIG": 1}
    assert not report.clean
    assert any("unrecognised element" in i for i in report.issues)


def test_quality_report_explicit_bonds_from_sdf():
    report = quality_report(os.path.join(FIXTURES, "docking_poses.sdf"))
    assert report.bond_source == "explicit"
    assert report.n_bonds >= 1


def test_quality_report_inferred_bonds_from_xyz(water_xyz):
    report = quality_report(water_xyz)
    assert report.bond_source == "inferred"
    assert report.n_bonds == 2  # two O-H bonds


def test_quality_report_accepts_in_memory_molecule():
    mol = ms.read(os.path.join(DATA, "1ubq.pdb"))
    report = quality_report(mol)
    assert report.n_atoms == 660
    assert report.fmt == ""
    # File-level checks are skipped and reported as a note.
    assert any("skipped" in n for n in report.notes)
    assert report.altloc_atoms == 0


def test_quality_report_reports_missing_metadata_for_xyz(water_xyz):
    # An XYZ file carries only elements and coordinates.
    report = quality_report(water_xyz)
    assert "atom names" in report.missing_metadata
    assert "chain identifiers" in report.missing_metadata
    # An element list IS present, so it is not flagged as missing.
    assert "element symbols" not in report.missing_metadata


def test_quality_report_to_dict_is_json_serialisable(weird_pdb):
    report = quality_report(weird_pdb)
    blob = json.dumps(report.to_dict())
    restored = json.loads(blob)
    assert restored["unknown_elements"] == {"XX": 1}
    assert restored["clean"] is False
    assert "issues" in restored


def test_quality_report_markdown_contains_sections(weird_pdb):
    md = quality_report(weird_pdb).report_markdown()
    assert md.startswith("# Structure quality report")
    assert "## Issues" in md
    assert "## Ligands" in md


def test_cli_qc_text(capsys):
    rc = main(["qc", os.path.join(DATA, "1ubq.pdb")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "660 atoms" in out
    assert "clean" in out


def test_cli_qc_json_and_out(tmp_path, capsys):
    out_path = tmp_path / "qc.md"
    rc = main([
        "qc", os.path.join(DATA, "3ptb.pdb"),
        "--json", "--out", str(out_path),
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured.split("wrote")[0])
    assert payload["ligands"] == {"BEN": 1}
    assert out_path.exists()
    assert "Structure quality report" in out_path.read_text()


def test_cli_qc_missing_source():
    with pytest.raises(SystemExit) as exc:
        main(["qc"])
    assert exc.value.code == 2


def test_quality_report_on_cif_runs_validation(capsys):
    # Exercises the mmCIF branch: with gemmi present the report validates and
    # notes/warns; without it, a graceful "not validated" note is recorded.
    report = quality_report(os.path.join(FIXTURES, "insertion_codes.cif"))
    assert report.fmt == ".cif"
    assert report.chains  # the CIF carries chain ids
    blob = report.notes + report.warnings
    assert any("mmCIF" in line for line in blob)


def test_quality_report_flags_blank_element_symbols():
    # A molecule whose second atom has an empty element symbol.
    mol = ms.Molecule(coords=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], elements=["C", ""])
    report = quality_report(mol)
    assert report.blank_elements == 1
    assert not report.clean
    assert any("no element symbol" in i for i in report.issues)


def test_quality_report_single_atom_has_no_bonds():
    mol = ms.Molecule(coords=[[0.0, 0.0, 0.0]], elements=["C"])
    report = quality_report(mol)
    assert report.bond_source == "none"
    assert report.n_bonds == 0


def test_quality_report_empty_molecule_is_flagged():
    from molscope.quality import QualityReport

    empty = QualityReport(path="x", fmt="", n_atoms=0)
    assert not empty.clean
    assert "no atoms parsed" in empty.issues


def test_report_rendering_covers_all_sections():
    # Build a fully-populated report and render every optional section once.
    from molscope.quality import QualityReport

    report = QualityReport(
        path="busy.pdb", fmt=".pdb", n_atoms=10, n_models=3, chains=["A", "B"],
        n_residues=4, ligands={"BEN": 1}, n_waters=2, n_ions=1,
        n_hetero_atoms=3, missing_metadata=["atom names"],
        unknown_elements={"XX": 1}, blank_elements=1, bond_source="inferred",
        n_bonds=9, altloc_atoms=2, low_occupancy_atoms=2,
        warnings=["mmCIF invalid: bad"], notes=["some note"],
    )
    summary = report.summary()
    assert "3 models" in summary
    assert "chains A,B" in summary
    assert "1 ligand(s)" in summary
    assert "issues:" in summary

    md = report.report_markdown()
    for section in ("## Issues", "## Ligands", "Waters:", "## Alternate locations",
                    "## Metadata not carried", "## Notes"):
        assert section in md

    payload = report.to_dict()
    assert payload["n_models"] == 3
    assert payload["clean"] is False
