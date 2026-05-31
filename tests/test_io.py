import os
from pathlib import Path

import numpy as np
import pytest

import molscope as ms
from molscope import Molecule
from molscope.io import read_pdb, read_pdb_models, read_xyz, write_pdb, write_xyz

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def _atom_line(serial, altloc, x, y, z, element="N", occupancy=1.0,
               record_type="ATOM", resname="ASP"):
    """Build a fixed-column ATOM/HETATM record with an explicit altLoc at column 17."""
    line = list(" " * 80)

    def put(value, start):
        line[start - 1:start - 1 + len(value)] = value

    put(record_type, 1)
    put(f"{serial:>5}", 7)
    put(" N  ", 13)
    put(altloc, 17)
    put(f"{resname:>3}", 18)
    put("A", 22)
    put(f"{1:>4}", 23)
    put(f"{x:8.3f}", 31)
    put(f"{y:8.3f}", 39)
    put(f"{z:8.3f}", 47)
    put(f"{occupancy:6.2f}", 55)
    put(f"{element:>2}", 77)
    return "".join(line).rstrip() + "\n"

PDB_ATOM = (
    "ATOM      1  N   ASP A   1       8.950   5.340  -7.914  1.00  0.00           N  \n"
)


def test_read_xyz_bare_coordinates():
    mol = read_xyz(os.path.join(DATA, "helix_201.xyz"))
    assert len(mol) == 201
    assert mol.coords.shape == (201, 3)
    np.testing.assert_allclose(mol.coords[0], [0.3517846, -0.7869986, -2.873479])


def test_read_accepts_pathlike():
    mol = ms.read(Path(DATA) / "helix_201.xyz")
    assert len(mol) == 201


def test_read_standard_xyz_with_elements(tmp_path):
    f = tmp_path / "water.xyz"
    f.write_text("3\nwater\nO 0.0 0.0 0.0\nH 0.76 0.59 0.0\nH -0.76 0.59 0.0\n")
    mol = read_xyz(str(f))
    assert len(mol) == 3
    assert mol.elements == ["O", "H", "H"]


def test_read_pdb_fixed_columns(tmp_path):
    f = tmp_path / "one.pdb"
    f.write_text(PDB_ATOM)
    mol = read_pdb(str(f))
    assert len(mol) == 1
    np.testing.assert_allclose(mol.coords[0], [8.950, 5.340, -7.914])
    assert mol.elements == ["N"]


def test_pdb_columns_beat_whitespace_split(tmp_path):
    """Touching coordinate fields fool str.split but not column slicing."""
    line = (
        "ATOM      1  CA  ARG A 100    -123.456-789.012  12.345  1.00  0.00           C  \n"
    )
    f = tmp_path / "tight.pdb"
    f.write_text(line)
    mol = read_pdb(str(f))
    # Whitespace splitting would merge "-123.456-789.012" into one bad token.
    np.testing.assert_allclose(mol.coords[0], [-123.456, -789.012, 12.345])


def test_read_pdb_single_model_from_multimodel():
    mol = read_pdb(os.path.join(DATA, "1aml.pdb"), model=1)
    # 1aml has 20 NMR models; model 1 must be a slice of the 11960 total atoms.
    assert 0 < len(mol) < 11960


def test_read_dispatches_on_extension():
    assert len(ms.read(os.path.join(DATA, "1fqy.pdb"))) == 1661
    with pytest.raises(ValueError):
        ms.read("structure.mol2")


def test_pdb_skips_alternate_conformations(tmp_path):
    f = tmp_path / "alt.pdb"
    f.write_text(
        _atom_line(1, "A", 1.0, 2.0, 3.0) + _atom_line(2, "B", 9.0, 9.0, 9.0)
    )
    mol = read_pdb(str(f))
    assert len(mol) == 1  # only the primary "A" conformation is kept
    np.testing.assert_allclose(mol.coords[0], [1.0, 2.0, 3.0])


def test_pdb_altloc_all_keeps_every_location(tmp_path):
    f = tmp_path / "alt_all.pdb"
    f.write_text(
        _atom_line(1, "A", 1.0, 2.0, 3.0)
        + _atom_line(2, "B", 9.0, 9.0, 9.0)
    )
    mol = read_pdb(str(f), altloc="all")
    assert len(mol) == 2
    np.testing.assert_allclose(mol.coords[1], [9.0, 9.0, 9.0])


def test_pdb_altloc_first_keeps_first_location_per_atom(tmp_path):
    f = tmp_path / "alt_first.pdb"
    f.write_text(
        _atom_line(1, "B", 9.0, 9.0, 9.0)
        + _atom_line(2, "A", 1.0, 2.0, 3.0)
    )
    mol = read_pdb(str(f), altloc="first")
    assert len(mol) == 1
    np.testing.assert_allclose(mol.coords[0], [9.0, 9.0, 9.0])


def test_pdb_altloc_highest_occupancy(tmp_path):
    f = tmp_path / "alt_occ.pdb"
    f.write_text(
        _atom_line(1, "A", 1.0, 2.0, 3.0, occupancy=0.25)
        + _atom_line(2, "B", 9.0, 9.0, 9.0, occupancy=0.75)
    )
    mol = read_pdb(str(f), altloc="highest_occupancy")
    assert len(mol) == 1
    np.testing.assert_allclose(mol.coords[0], [9.0, 9.0, 9.0])


def test_pdb_altloc_rejects_unknown_policy(tmp_path):
    f = tmp_path / "alt_bad.pdb"
    f.write_text(_atom_line(1, "A", 1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="altloc"):
        read_pdb(str(f), altloc="bad")


def test_read_all_nmr_models():
    models = read_pdb_models(os.path.join(DATA, "1aml.pdb"))
    assert len(models) == 20
    assert {len(m) for m in models} == {len(models[0])}  # consistent atom count
    assert models[1].name.endswith("#2")


def test_rmsd_between_nmr_models():
    models = read_pdb_models(os.path.join(DATA, "1aml.pdb"))
    assert models[0].rmsd(models[1], align=True) >= 0.0


def test_read_pdb_model_out_of_range(tmp_path):
    f = tmp_path / "one.pdb"
    f.write_text(PDB_ATOM)
    with pytest.raises(ValueError):
        read_pdb(str(f), model=2)


def test_xyz_round_trip(tmp_path):
    mol = Molecule(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), ["O", "H"], name="t")
    path = str(tmp_path / "rt.xyz")
    write_xyz(mol, path)
    back = read_xyz(path)
    np.testing.assert_allclose(back.coords, mol.coords)
    assert back.elements == ["O", "H"]


def test_pdb_round_trip(tmp_path):
    mol = Molecule(np.array([[1.234, -5.678, 9.012], [0.0, 0.0, 0.0]]), ["C", "O"])
    path = str(tmp_path / "rt.pdb")
    write_pdb(mol, path)
    back = read_pdb(path)
    np.testing.assert_allclose(back.coords, mol.coords, atol=1e-3)
    assert back.elements == ["C", "O"]


def test_pdb_reads_hetatm_flag(tmp_path):
    f = tmp_path / "hetero.pdb"
    f.write_text(
        _atom_line(1, " ", 0.0, 0.0, 0.0, element="C", record_type="ATOM")
        + _atom_line(2, " ", 1.0, 1.0, 1.0, element="O", record_type="HETATM",
                     resname="LIG")
    )
    mol = read_pdb(str(f))
    assert mol.hetero == [False, True]
    assert len(mol.protein()) == 1
    assert len(mol.hetero_atoms()) == 1
    assert mol.hetero_atoms().resnames == ["LIG"]


def test_pdb_round_trip_preserves_hetatm(tmp_path):
    f = tmp_path / "hetero.pdb"
    f.write_text(
        _atom_line(1, " ", 0.0, 0.0, 0.0, element="C", record_type="ATOM")
        + _atom_line(2, " ", 1.0, 1.0, 1.0, element="O", record_type="HETATM")
    )
    mol = read_pdb(str(f))
    out = str(tmp_path / "rt.pdb")
    write_pdb(mol, out)
    back = read_pdb(out)
    assert back.hetero == [False, True]


def test_pdb_conect_records_create_explicit_bonds(tmp_path):
    f = tmp_path / "conect.pdb"
    f.write_text(
        _atom_line(1, " ", 0.0, 0.0, 0.0, element="C")
        + _atom_line(2, " ", 10.0, 0.0, 0.0, element="C")
        + "CONECT    1    2\n"
    )
    mol = read_pdb(str(f))
    np.testing.assert_array_equal(mol.bonds(), [[0, 1]])


def test_pdb_round_trip_preserves_explicit_bonds(tmp_path):
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        ["C", "C"],
        bond_index=[[0, 1]],
    )
    path = str(tmp_path / "bonded.pdb")
    write_pdb(mol, path)
    back = read_pdb(path)
    np.testing.assert_array_equal(back.bonds(), [[0, 1]])


# -- malformed / edge-case fixtures -----------------------------------------

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixture(name):
    return os.path.join(FIXTURES, name)


def test_xyz_truncated_frame_reports_count_mismatch():
    with pytest.raises(ValueError, match="declares 3 atoms but 2 were found"):
        ms.read(_fixture("truncated.xyz"))


def test_xyz_non_numeric_coordinate_names_the_line():
    with pytest.raises(ValueError, match=r"XYZ file \(line 4\).*coordinates"):
        ms.read(_fixture("bad_coord.xyz"))


def test_pdb_short_atom_record_names_columns():
    with pytest.raises(ValueError, match=r"PDB file \(line 2\).*coordinate columns 31-54"):
        ms.read(_fixture("short_atom.pdb"))


def test_pdb_non_numeric_coordinate_errors_clearly():
    with pytest.raises(ValueError, match="could not read coordinate columns"):
        ms.read(_fixture("bad_coord.pdb"))


def test_pdb_header_only_returns_empty_molecule():
    # A file with no ATOM/HETATM records is a valid (empty) read, not an error.
    mol = ms.read(_fixture("no_atoms.pdb"))
    assert len(mol) == 0


def test_cif_without_atom_site_loop_errors():
    with pytest.raises(ValueError, match="no _atom_site coordinate loop"):
        ms.read(_fixture("no_atom_site.cif"))


def test_cif_missing_coordinate_columns_lists_what_was_found():
    with pytest.raises(ValueError, match="missing coordinate column"):
        ms.read(_fixture("missing_coord_col.cif"))


def test_sdf_v3000_is_rejected_clearly():
    with pytest.raises(ValueError, match="V3000.*not supported"):
        ms.read(_fixture("v3000.sdf"))


def test_sdf_malformed_counts_line_errors():
    with pytest.raises(ValueError, match="malformed counts line"):
        ms.read(_fixture("bad_counts.sdf"))


def test_sdf_truncated_atom_block_errors():
    with pytest.raises(ValueError, match="declares 3 atoms .* only 2 block lines"):
        ms.read(_fixture("truncated.sdf"))


def test_valid_v2000_sdf_still_parses():
    mol = ms.read(_fixture("water.sdf"))
    assert mol.elements == ["O", "H", "H"]
    np.testing.assert_array_equal(np.sort(mol.bonds(), axis=0), [[0, 1], [0, 2]])


def test_sdf_data_tags_captured_into_properties():
    # read_sdf reads the first record and now carries its > <tag> data fields.
    mol = ms.read(_fixture("docking_poses.sdf"))
    assert mol.name == "ligA_pose1"
    assert mol.properties == {"minimizedAffinity": "-8.4", "CNNscore": "0.91"}


def test_read_sdf_frames_returns_every_pose_with_scores():
    poses = ms.read_sdf_frames(_fixture("docking_poses.sdf"))
    assert [m.name for m in poses] == ["ligA_pose1", "ligA_pose2", "ligB_pose1"]
    # 3D coordinates are preserved per pose (no SMILES round-trip).
    assert [len(m) for m in poses] == [3, 3, 2]
    assert poses[2].elements == ["N", "H"]
    # Docking scores ride along as string properties, ready for ranking.
    ranked = sorted(poses, key=lambda m: float(m.properties["minimizedAffinity"]))
    assert [m.name for m in ranked] == ["ligA_pose1", "ligB_pose1", "ligA_pose2"]


def test_read_sdf_frames_skips_unterminated_and_broken_records(tmp_path):
    # Last record omits the $$$$ terminator; a middle record is malformed.
    sdf = tmp_path / "messy.sdf"
    sdf.write_text(
        "good1\n  prog\n\n"
        "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n>  <score>\n-5.0\n\n$$$$\n"
        "broken\n  prog\n\n"
        "   X  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n"
        "good2\n  prog\n\n"
        "  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    2.0000    2.0000    2.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n>  <score>\n-9.0\n"
    )
    poses = ms.read_sdf_frames(str(sdf))
    assert [m.name for m in poses] == ["good1", "good2"]
    assert poses[0].properties["score"] == "-5.0"
    assert poses[1].properties["score"] == "-9.0"


def test_read_sdf_frames_empty_file_errors(tmp_path):
    empty = tmp_path / "empty.sdf"
    empty.write_text("\n\n")
    with pytest.raises(ValueError, match="no readable records"):
        ms.read_sdf_frames(str(empty))


def test_read_sdf_frames_reads_single_record_mol():
    # A plain single-molecule SDF is just a one-element list.
    poses = ms.read_sdf_frames(_fixture("water.sdf"))
    assert len(poses) == 1
    assert poses[0].elements == ["O", "H", "H"]


def test_pdb_preserves_rich_residue_ids_and_insertion_codes(tmp_path):
    mol = ms.read(_fixture("ugly_residue_ids.pdb"))
    assert len(mol) == 9                         # altLoc B ligand atom is skipped
    assert mol.icodes == ["", "", "A", "A", "B", "B", "", "", ""]
    assert mol.resids.tolist()[:6] == [-1, 0, 100, 100, 100, 100]

    groups = list(mol.residue_groups())
    assert [group.residue_id.label() for group in groups] == [
        "A:GLY-1",
        "A:ALA0",
        "A:SER100A",
        "A:THR100B",
        "A:LIG10",
        "B:LIG10",
    ]

    atom_idx, resname, resid, chain = groups[2]   # legacy unpacking still works
    assert atom_idx == [2, 3]
    assert (resname, resid, chain) == ("SER", 100, "A")

    assert len(mol.select(resid=100)) == 4
    assert mol.select(resid=100, icode="A").resnames == ["SER", "SER"]
    assert len(mol.select(residue_id=ms.ResidueId("A", 100, "B"))) == 2

    out = str(tmp_path / "roundtrip.pdb")
    write_pdb(mol, out)
    back = read_pdb(out)
    assert back.icodes == mol.icodes


def test_cif_preserves_atom_site_insertion_codes():
    mol = ms.read(_fixture("insertion_codes.cif"))
    assert mol.icodes == ["A", "B", ""]
    assert [rid.label() for rid in mol.residue_ids] == [
        "A:SER100A",
        "A:THR100B",
        "B:ALA0",
    ]
    assert [group.residue_id.label() for group in mol.residue_groups()] == [
        "A:SER100A",
        "A:THR100B",
        "B:ALA0",
    ]


def test_fetch_unknown_pdb_id_raises_value_error(monkeypatch, tmp_path):
    from urllib.error import HTTPError

    import molscope.io as io

    def boom(url):
        raise HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(io, "urlopen", boom)
    with pytest.raises(ValueError, match="not found at RCSB"):
        io.fetch("zzzz9", cache_dir=str(tmp_path))


def test_read_template_bond_perception_only_for_pdb():
    with pytest.raises(ValueError, match="only supported for .pdb"):
        ms.read(os.path.join(DATA, "helix_201.xyz"), bond_perception="template")


def test_read_pdb_rejects_unknown_bond_perception():
    with pytest.raises(ValueError, match="bond_perception"):
        read_pdb(os.path.join(DATA, "1ubq.pdb"), bond_perception="bogus")


def test_protonation_requires_template_bonds():
    with pytest.raises(ValueError, match="protonation requires"):
        read_pdb(os.path.join(DATA, "1ubq.pdb"), protonation="standard")
