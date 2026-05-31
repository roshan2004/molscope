import gzip
import os

import numpy as np
import pytest

import molscope as ms
from molscope.io import read_pdb_models, read_xyz_frames

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def test_stream_xyz_frames():
    path = os.path.join(DATA, "helix_201.xyz")
    
    # 1. Compare eager read vs lazy stream
    eager_frames = read_xyz_frames(path)
    lazy_frames = list(ms.stream_xyz_frames(path))
    
    assert len(lazy_frames) == len(eager_frames)
    assert len(lazy_frames) == 1
    
    np.testing.assert_allclose(lazy_frames[0].coords, eager_frames[0].coords)
    assert lazy_frames[0].elements == eager_frames[0].elements
    assert lazy_frames[0].name == eager_frames[0].name


def test_stream_xyz_multi_frame(tmp_path):
    # Create a dummy multi-frame xyz file
    f = tmp_path / "multi.xyz"
    f.write_text(
        "3\nframe 1\nO 0.0 0.0 0.0\nH 0.7 0.5 0.0\nH -0.7 0.5 0.0\n"
        "2\nframe 2\nC 0.0 0.0 0.0\nO 0.0 0.0 1.2\n"
    )
    
    eager = read_xyz_frames(str(f))
    lazy = list(ms.stream_xyz_frames(str(f)))
    
    assert len(lazy) == 2
    assert len(eager) == 2
    assert lazy[0].elements == ["O", "H", "H"]
    assert lazy[1].elements == ["C", "O"]
    np.testing.assert_allclose(lazy[0].coords, eager[0].coords)
    np.testing.assert_allclose(lazy[1].coords, eager[1].coords)
    assert lazy[0].name.endswith("#1")
    assert lazy[1].name.endswith("#2")


def test_stream_pdb_models():
    path = os.path.join(DATA, "1aml.pdb")
    
    # 1aml has 20 NMR models
    eager_models = read_pdb_models(path)
    lazy_models = list(ms.stream_pdb_models(path))
    
    assert len(lazy_models) == 20
    assert len(eager_models) == 20
    
    # Check coords and properties for all models
    for m_eager, m_lazy in zip(eager_models, lazy_models):
        np.testing.assert_allclose(m_lazy.coords, m_eager.coords)
        assert m_lazy.elements == m_eager.elements
        assert m_lazy.atom_names == m_eager.atom_names
        assert m_lazy.resnames == m_eager.resnames
        assert m_lazy.chains == m_eager.chains
        assert m_lazy.resids.tolist() == m_eager.resids.tolist()
        assert m_lazy.icodes == m_eager.icodes
        assert m_lazy.hetero == m_eager.hetero
        assert m_lazy.name == m_eager.name
        # Check bonds parsed from CONECT
        if m_eager.bond_index is not None:
            np.testing.assert_array_equal(m_lazy.bond_index, m_eager.bond_index)
        else:
            assert m_lazy.bond_index is None


def test_stream_pdb_single_model(tmp_path):
    # Test PDB with no MODEL/ENDMDL records
    PDB_ATOM = "ATOM      1  N   ASP A   1       8.950   5.340  -7.914  1.00  0.00           N  \n"
    f = tmp_path / "single.pdb"
    f.write_text(PDB_ATOM)
    
    lazy = list(ms.stream_pdb_models(str(f)))
    assert len(lazy) == 1
    assert len(lazy[0]) == 1
    assert lazy[0].elements == ["N"]
    np.testing.assert_allclose(lazy[0].coords[0], [8.950, 5.340, -7.914])


def test_stream_dispatcher():
    # Test .xyz file
    xyz_path = os.path.join(DATA, "helix_201.xyz")
    lazy_xyz = list(ms.stream(xyz_path))
    assert len(lazy_xyz) == 1
    assert len(lazy_xyz[0]) == 201
    
    # Test .pdb file
    pdb_path = os.path.join(DATA, "1aml.pdb")
    lazy_pdb = list(ms.stream(pdb_path))
    assert len(lazy_pdb) == 20

    # Test unsupported extension
    with pytest.raises(ValueError, match="Streaming is not supported"):
        list(ms.stream("test.txt"))


def test_stream_gzip(tmp_path):
    # Create gzipped xyz file
    f = tmp_path / "water.xyz.gz"
    with gzip.open(f, "wt") as fh:
        fh.write("3\nwater\nO 0.0 0.0 0.0\nH 0.76 0.59 0.0\nH -0.76 0.59 0.0\n")
        
    lazy = list(ms.stream(str(f)))
    assert len(lazy) == 1
    assert len(lazy[0]) == 3
    assert lazy[0].elements == ["O", "H", "H"]


def test_stream_sdf_frames():
    from molscope.io import read_sdf_frames, stream_sdf_frames

    poses_sdf = os.path.join(os.path.dirname(__file__), "fixtures", "docking_poses.sdf")
    eager = read_sdf_frames(poses_sdf)
    lazy = list(stream_sdf_frames(poses_sdf))

    assert len(lazy) == 3
    assert len(eager) == 3
    assert [m.name for m in lazy] == [m.name for m in eager]
    np.testing.assert_allclose(lazy[0].coords, eager[0].coords)
    assert lazy[0].properties == eager[0].properties


def test_stream_dispatcher_sdf():
    poses_sdf = os.path.join(os.path.dirname(__file__), "fixtures", "docking_poses.sdf")
    lazy = list(ms.stream(poses_sdf))
    assert len(lazy) == 3
    assert lazy[0].name == "ligA_pose1"


def test_stream_sdf_handles_blank_line_padding(tmp_path):
    """Many SDF writers put a blank line between $$$$ and the next record's
    title; the streaming parser must not silently drop the padded record."""
    from molscope.io import stream_sdf_frames

    sdf = tmp_path / "padded.sdf"
    sdf.write_text(
        "molA\n p\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n$$$$\n"
        "\n"                                  # blank padding before molB
        "molB\n p\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    1.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n$$$$\n"
    )
    frames = list(stream_sdf_frames(str(sdf)))
    assert [m.name for m in frames] == ["molA", "molB"]
