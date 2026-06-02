"""Tests for metadata/selections, geometry, ensemble analysis, I/O and viz."""

import gzip
import os

import numpy as np
import pytest

import molscope as ms
from molscope import Molecule, ensemble

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


# -- selections / metadata --------------------------------------------------


def test_pdb_carries_metadata():
    mol = ms.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    assert mol.has_topology
    assert mol.atom_names[0] == "N"
    assert mol.resnames[0] == "LYS"
    assert mol.chains[0] == "A"
    assert mol.resids[0] == 8


def test_select_by_element_and_atom_name():
    mol = ms.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    carbons = mol.select(element="C")
    assert all(e == "C" for e in carbons.elements)
    assert 0 < len(carbons) < len(mol)

    ca = mol.alpha_carbons()
    assert all(n == "CA" for n in ca.atom_names)
    assert 0 < len(ca) <= len(mol.backbone())


def test_select_without_metadata_raises():
    xyz = ms.read(os.path.join(DATA, "helix_201.xyz"))
    with pytest.raises(ValueError):
        xyz.select(chain="A")


def test_getitem_subsets():
    mol = ms.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    first10 = mol[np.arange(10)]
    assert len(first10) == 10
    assert first10.atom_names == mol.atom_names[:10]


def test_ca_rmsd_smaller_atom_count_than_all_atom():
    models = ms.read_pdb_models(os.path.join(DATA, "1aml.pdb"))
    ca0, ca1 = models[0].alpha_carbons(), models[1].alpha_carbons()
    assert len(ca0) < len(models[0])
    assert ca0.rmsd(ca1, align=True) >= 0.0


# -- geometry ---------------------------------------------------------------


def water():
    return Molecule(
        np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        ["O", "H", "H"],
    )


def test_distance_angle_dihedral():
    mol = water()
    assert mol.distance(0, 1) == pytest.approx(0.96)
    assert mol.angle(1, 0, 2) == pytest.approx(104.0, abs=1.0)
    square = Molecule(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=float),
        ["C"] * 4,
    )
    assert abs(square.dihedral(0, 1, 2, 3)) == pytest.approx(90.0, abs=1e-6)


def test_distance_matrix_and_contacts():
    mol = water()
    dm = mol.distance_matrix()
    assert dm.shape == (3, 3)
    assert np.allclose(np.diag(dm), 0)
    assert len(mol.contacts(cutoff=2.0)) == 3  # all three pairs are within 2 A


def test_distance_matrix_numpy_backend_matches_default():
    mol = water()
    default = mol.distance_matrix()
    dense = mol.distance_matrix(backend="numpy")
    np.testing.assert_allclose(default, dense)
    assert mol.contact_count(cutoff=2.0, backend="numpy") == 3
    np.testing.assert_array_equal(
        np.sort(mol.contacts(cutoff=2.0, backend="numpy"), axis=0),
        np.sort(mol.contacts(cutoff=2.0), axis=0),
    )


def test_distance_matrix_invalid_backend():
    with pytest.raises(ValueError):
        water().distance_matrix(backend="not-a-backend")


def test_distance_matrix_torch_backend_if_installed():
    pytest.importorskip("torch")
    mol = water()
    np.testing.assert_allclose(
        mol.distance_matrix(backend="torch", device="cpu"),
        mol.distance_matrix(),
        atol=1e-6,
    )


def test_dimensions_and_formula():
    mol = Molecule(np.array([[0, 0, 0], [3, 0, 0]], dtype=float), ["O", "H"])
    np.testing.assert_allclose(mol.dimensions, [3, 0, 0])
    assert water().formula == "H2 O"


def test_summary_runs():
    assert "atoms" in ms.read_pdb(os.path.join(DATA, "1fqy.pdb")).summary()


# -- ensemble ---------------------------------------------------------------


def test_ensemble_average_and_rmsf():
    models = ms.read_pdb_models(os.path.join(DATA, "1aml.pdb"))[:5]
    avg = ensemble.average(models)
    assert len(avg) == len(models[0])
    fluct = ensemble.rmsf(models)
    assert fluct.shape == (len(models[0]),)
    assert (fluct >= 0).all()


def test_ensemble_rmsd_matrix():
    models = ms.read_pdb_models(os.path.join(DATA, "1aml.pdb"))[:4]
    mat = ensemble.rmsd_matrix(models)
    assert mat.shape == (4, 4)
    assert np.allclose(np.diag(mat), 0)
    assert np.allclose(mat, mat.T)


# -- I/O: gzip, cif, sdf, multi-frame xyz, fetch ----------------------------


def test_gzip_transparent_read(tmp_path):
    src = os.path.join(DATA, "1fqy.pdb")
    gz = tmp_path / "1fqy.pdb.gz"
    with open(src, "rb") as fin, gzip.open(gz, "wb") as fout:
        fout.write(fin.read())
    assert len(ms.read(str(gz))) == 1661


CIF = """\
data_TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 N N ALA A 1 1.000 2.000 3.000
ATOM 2 C CA ALA A 1 4.000 5.000 6.000
#
"""


def test_read_cif(tmp_path):
    f = tmp_path / "t.cif"
    f.write_text(CIF)
    mol = ms.read(str(f))
    assert len(mol) == 2
    assert mol.elements == ["N", "C"]
    assert mol.atom_names == ["N", "CA"]
    np.testing.assert_allclose(mol.coords[1], [4, 5, 6])


def test_read_cif_handles_quoted_atom_site_values(tmp_path):
    f = tmp_path / "quoted.cif"
    f.write_text("""\
data_TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 C 'CA alt' 'GLY A' A 1 1.000 2.000 3.000
#
""")
    mol = ms.read(str(f))
    assert mol.atom_names == ["CA alt"]
    assert mol.resnames == ["GLY A"]
    np.testing.assert_allclose(mol.coords[0], [1, 2, 3])


def test_read_cif_handles_semicolon_text_fields(tmp_path):
    f = tmp_path / "multiline.cif"
    f.write_text("""\
data_TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 C
;CA alt
;
GLY A 1 1.000 2.000 3.000
#
""")
    mol = ms.read(str(f))
    assert mol.atom_names == ["CA alt"]
    np.testing.assert_allclose(mol.coords[0], [1, 2, 3])


SDF = """\
water
  example

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0
    0.9600    0.0000    0.0000 H   0  0
   -0.2400    0.9300    0.0000 H   0  0
  1  2  1  0
  1  3  1  0
M  END
$$$$
"""


def test_read_sdf(tmp_path):
    f = tmp_path / "w.sdf"
    f.write_text(SDF)
    mol = ms.read(str(f))
    assert mol.elements == ["O", "H", "H"]
    np.testing.assert_allclose(mol.coords[1], [0.96, 0.0, 0.0])
    np.testing.assert_array_equal(mol.bonds(), [[0, 1], [0, 2]])
    np.testing.assert_array_equal(mol.bond_orders, [1.0, 1.0])
    np.testing.assert_array_equal(mol.formal_charges, [0, 0, 0])
    np.testing.assert_array_equal(mol.to_graph().edge_types, [1.0, 1.0])


def test_read_sdf_preserves_charges_and_aromatic_bond_code(tmp_path):
    f = tmp_path / "charged.sdf"
    f.write_text("""\
charged
  example

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 N   0  3
    1.2000    0.0000    0.0000 C   0  0
  1  2  4  0
M  CHG  1   2  -1
M  END
$$$$
""")
    mol = ms.read(str(f))
    np.testing.assert_array_equal(mol.bond_orders, [1.5])
    np.testing.assert_array_equal(mol.formal_charges, [1, -1])


def test_multi_frame_xyz(tmp_path):
    frame = "2\nframe\nH 0 0 0\nH 0 0 1\n"
    f = tmp_path / "traj.xyz"
    f.write_text(frame * 3)
    frames = ms.read_xyz_frames(str(f))
    assert len(frames) == 3
    assert all(len(fr) == 2 for fr in frames)


def test_fetch_uses_downloader(tmp_path, monkeypatch):
    import molscope.io as mio

    with open(os.path.join(DATA, "1fqy.pdb"), "rb") as fh:
        sample = fh.read()

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return sample

    monkeypatch.setattr(mio, "urlopen", lambda url: FakeResp())
    mol = ms.fetch("1fqy", cache_dir=str(tmp_path))
    assert len(mol) == 1661


# -- visualization ----------------------------------------------------------


def test_py3dmol_view_builds():
    pytest.importorskip("py3Dmol")
    mol = water()
    viewer = mol.view()
    assert viewer is not None


def _cg_pair():
    """A two-residue molecule and a CG model with a bond and a virtual site.

    The bond and virtual site make ``view_mapping`` exercise every render branch
    (bead spheres, the virtual-site sphere, and the bond cylinders).
    """
    coords = np.array([[0.0, 0, 0], [1, 0, 0], [5, 0, 0], [6, 0, 0]])
    mol = Molecule(
        coords, ["C", "O", "N", "C"],
        resids=[1, 1, 2, 2], resnames=["ALA", "ALA", "GLY", "GLY"],
        chains=["A"] * 4, atom_names=["CA", "CB", "CA", "CB"],
    )
    cg = mol.coarse_grain(
        "residue_com", bonds=[(0, 1)],
        virtual_sites=[{"name": "MID", "parents": [0, 1]}],
    )
    return mol, cg


def test_view_mapping_builds():
    pytest.importorskip("py3Dmol")
    mol, cg = _cg_pair()
    viewer = cg.view_mapping(mol)
    assert viewer is not None


def test_view_mapping_requires_a_cg_report():
    # A plain molecule carries no coarse-graining report; this is caught before
    # py3Dmol is even imported, so it holds with or without the [viz] extra.
    from molscope.plotting import view_mapping

    with pytest.raises(ValueError, match="coarse-grain"):
        view_mapping(water(), water())


def test_spin_gif(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from molscope.plotting import spin_gif

    out = str(tmp_path / "spin.gif")
    spin_gif(water(), out, frames=4, fps=5)
    assert os.path.getsize(out) > 0
