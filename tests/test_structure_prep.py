"""Tests for the structure-prep / QC report.

Topology checks run on synthetic molecules with no optional backend. The
net-charge integration paths skip without RDKit (and PROPKA for the pKa mode).
"""

import os

import numpy as np
import pytest

from molscope.molecule import Molecule
from molscope.structure_prep import (
    StructureReport,
    _topology_checks,
    prepare_structure,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _residue(resname, resid, atoms, chain="A", base=(0.0, 0.0, 0.0)):
    """Build per-atom (coord, element, name, resname, resid, chain) rows."""
    rows = []
    for k, (name, elem) in enumerate(atoms):
        rows.append((
            [base[0] + k, base[1], base[2]], elem, name, resname, resid, chain
        ))
    return rows


def _molecule(rows):
    coords = np.array([r[0] for r in rows], dtype=float)
    return Molecule(
        coords,
        elements=[r[1] for r in rows],
        atom_names=[r[2] for r in rows],
        resnames=[r[3] for r in rows],
        resids=np.array([r[4] for r in rows]),
        chains=[r[5] for r in rows],
    )


# Standard residue atom sets used by the synthetic fixtures.
_ALA_FULL = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")]
_ASP_FULL = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"),
             ("CG", "C"), ("OD1", "O"), ("OD2", "O")]


# -- topology checks (no optional backend) ----------------------------------

def test_truncated_sidechain_flagged():
    asp_missing = _ASP_FULL[:-1]  # drop OD2 -> 7 of 8 heavy atoms
    mol = _molecule(_residue("ASP", 1, asp_missing))
    *_, truncated, gaps, breaks = _topology_checks(mol)
    assert truncated == [("A", 1, "ASP", 7, 8)]
    assert gaps == [] and breaks == []


def test_missing_backbone_flagged():
    no_ca = [a for a in _ALA_FULL if a[0] != "CA"]
    mol = _molecule(_residue("ALA", 1, no_ca))
    _, _, _, _, missing_bb, *_ = _topology_checks(mol)
    assert missing_bb == [("A", 1, "ALA", ["CA"])]


def test_numbering_gap_flagged():
    rows = _residue("ALA", 1, _ALA_FULL) + _residue("ALA", 5, _ALA_FULL, base=(3, 0, 0))
    mol = _molecule(rows)
    *_, gaps, breaks = _topology_checks(mol)
    assert gaps == [("A", 1, 5, 3)]  # residues 2,3,4 missing
    # 5 - 1 != 1, so it is reported as a gap, not a spatial break.
    assert breaks == []


def test_chain_break_flagged():
    # Sequence-adjacent residues whose CA atoms are far apart.
    rows = _residue("ALA", 1, _ALA_FULL)
    rows += _residue("ALA", 2, _ALA_FULL, base=(20, 0, 0))
    mol = _molecule(rows)
    *_, gaps, breaks = _topology_checks(mol)
    assert gaps == []
    assert len(breaks) == 1
    chain, before, after, dist = breaks[0]
    assert (chain, before, after) == ("A", 1, 2)
    assert dist > 4.5


def test_ligand_water_and_nonstandard_inventory():
    rows = _residue("ALA", 1, _ALA_FULL)
    rows += _residue("HOH", 2, [("O", "O")])
    rows += _residue("HOH", 3, [("O", "O")])
    rows += _residue("ATP", 4, [("PA", "P"), ("PB", "P")])  # ligand / non-standard
    mol = _molecule(rows)
    n_polymer, nonstd, ligands, n_waters, *_ = _topology_checks(mol)
    assert n_polymer == 1
    assert n_waters == 2
    assert ligands == {"ATP": 1}
    assert nonstd == [("A", 4, "ATP")]


# -- StructureReport behaviour ----------------------------------------------

def test_report_verdict_and_serialisation():
    report = StructureReport(
        path="x.pdb", n_atoms=4,
        missing_backbone=[("A", 1, "ALA", ["CA"])],
    )
    assert report.ml_ready is False
    assert any("backbone" in b for b in report.blockers)
    d = report.to_dict()
    assert d["ml_ready"] is False
    assert d["missing_backbone"] == [["A", 1, "ALA", ["CA"]]]  # tuples -> lists
    assert "NOT ML-ready" in report.summary()
    assert "# Structure preparation report" in report.report_markdown()


def test_report_clean_is_ml_ready():
    report = StructureReport(path="x.pdb", n_atoms=5, has_hydrogens=True)
    assert report.ml_ready is True
    assert report.blockers == []
    assert "ML-ready" in report.summary()


# -- integration on a real PDB ----------------------------------------------

def test_prepare_structure_on_ubiquitin():
    report = prepare_structure(os.path.join(DATA, "1ubq.pdb"), protonation="none")
    assert report.n_polymer_residues == 76
    assert report.chains == ["A"]
    assert report.n_waters > 0
    assert report.ml_ready is True  # no missing backbone, no breaks
    assert report.low_occupancy_atoms > 0  # 1ubq has partial occupancies
    assert report.has_hydrogens is False


def test_prepare_structure_rejects_bad_protonation():
    with pytest.raises(ValueError, match="protonation"):
        prepare_structure(os.path.join(DATA, "1ubq.pdb"), protonation="acidic")


def test_net_charge_standard_needs_rdkit():
    pytest.importorskip("rdkit")
    report = prepare_structure(os.path.join(DATA, "1ubq.pdb"), protonation="standard")
    assert report.charge_method == "standard"
    assert report.net_charge == 0  # ubiquitin ~neutral under the textbook table


def test_net_charge_pka_tracks_ph():
    pytest.importorskip("rdkit")
    pytest.importorskip("propka")
    low = prepare_structure(os.path.join(DATA, "1ubq.pdb"), protonation="pka", ph=2.0)
    high = prepare_structure(os.path.join(DATA, "1ubq.pdb"), protonation="pka", ph=12.0)
    assert low.charge_method == "pka" and low.ph == 2.0
    assert low.net_charge > high.net_charge


def test_net_charge_works_for_gzipped_pdb():
    pytest.importorskip("rdkit")
    # 1d3z is a gzipped NMR ensemble; the temp-decompress path makes net charge work.
    report = prepare_structure(os.path.join(FIXTURES, "1d3z.pdb.gz"), protonation="standard")
    assert report.n_models == 10
    assert report.has_hydrogens is True
    assert report.net_charge is not None and report.charge_method == "standard"


def test_net_charge_unavailable_for_xyz_is_noted():
    # No protein template for .xyz -> net charge degrades with a note, no crash.
    report = prepare_structure(os.path.join(DATA, "helix_201.xyz"))
    assert report.net_charge is None
    assert report.charge_method == "unavailable"
    assert any("net charge" in n for n in report.notes)


# -- CLI --------------------------------------------------------------------

def test_cli_structure_report_text(capsys):
    from molscope.cli import main

    rc = main(["structure-report", os.path.join(DATA, "1ubq.pdb"), "--protonation", "none"])
    assert rc == 0
    assert "1ubq" in capsys.readouterr().out


def test_cli_structure_report_json_and_out(tmp_path, capsys):
    import json

    from molscope.cli import main

    out = tmp_path / "report.md"
    rc = main([
        "structure-report", os.path.join(DATA, "1ubq.pdb"),
        "--protonation", "none", "--json", "--out", str(out),
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.split("wrote")[0])
    assert payload["n_polymer_residues"] == 76
    assert out.exists()
    assert "# Structure preparation report" in out.read_text()


def test_cli_structure_report_missing_source():
    from molscope.cli import main

    # Mutually exclusive group is required: neither file nor --fetch -> argparse exit.
    with pytest.raises(SystemExit):
        main(["structure-report"])
