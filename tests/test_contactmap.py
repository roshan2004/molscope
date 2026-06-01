"""Tests for contact maps and ensemble contact frequency."""

import os

import numpy as np
import pytest

import molscope as ms
from molscope import ContactMap, Molecule

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def ca_chain():
    """Three residues, one CA each, at x = 0, 5, 12."""
    return Molecule(
        np.array([[0.0, 0, 0], [5.0, 0, 0], [12.0, 0, 0]]), ["C", "C", "C"],
        atom_names=["CA", "CA", "CA"], resnames=["ALA", "ALA", "ALA"],
        resids=np.array([1, 2, 3]), chains=["A", "A", "A"],
    )


def test_atom_contact_map():
    m = Molecule(np.array([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0]]), ["C", "C", "C"])
    cm = m.contact_map(cutoff=2.0, level="atom")
    assert isinstance(cm, ContactMap)
    assert cm.matrix.shape == (3, 3)
    assert cm.matrix[0, 1] == 1 and cm.matrix[0, 2] == 0
    assert np.diag(cm.matrix).sum() == 0          # no self-contacts


def test_atom_contact_map_numpy_backend_matches_default():
    m = Molecule(np.array([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0]]), ["C", "C", "C"])
    np.testing.assert_array_equal(
        m.contact_map(cutoff=2.0, level="atom").matrix,
        m.contact_map(cutoff=2.0, level="atom", backend="numpy").matrix,
    )


def test_residue_ca_contact_map():
    cm = ca_chain().contact_map(cutoff=8.0, level="residue", method="ca")
    assert cm.matrix.shape == (3, 3)
    # 0-1 (5 A) and 1-2 (7 A) in contact; 0-2 (12 A) not
    assert cm.matrix[0, 1] == 1 and cm.matrix[1, 2] == 1 and cm.matrix[0, 2] == 0
    assert np.array_equal(cm.matrix, cm.matrix.T)
    assert cm.labels == ["A:ALA1", "A:ALA2", "A:ALA3"]
    assert not cm.is_frequency


def test_residue_contact_map_keeps_insertion_code_residues_separate():
    mol = ms.read(os.path.join(FIXTURES, "ugly_residue_ids.pdb"))
    cm = mol.contact_map(cutoff=2.0, level="residue", method="ca")
    assert cm.labels == [
        "A:GLY-1",
        "A:ALA0",
        "A:SER100A",
        "A:THR100B",
        "A:LIG10",
        "B:LIG10",
    ]
    assert cm.resids.tolist() == [-1, 0, 100, 100, 10, 10]
    assert cm.icodes == ["", "", "A", "B", "", ""]
    assert [rid.insertion_code for rid in cm.residue_ids[2:4]] == ["A", "B"]


def test_residue_methods_all_run():
    mol = ms.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    for method in ("ca", "com", "min"):
        cutoff = 4.5 if method == "min" else 8.0
        cm = mol.contact_map(cutoff=cutoff, level="residue", method=method)
        assert cm.matrix.shape == (226, 226)
        assert np.array_equal(cm.matrix, cm.matrix.T)
        assert np.diag(cm.matrix).sum() == 0


def test_residue_contact_map_torch_backend_if_installed():
    pytest.importorskip("torch")
    mol = ca_chain()
    np.testing.assert_array_equal(
        mol.contact_map(cutoff=8.0, level="residue", backend="torch", device="cpu").matrix,
        mol.contact_map(cutoff=8.0, level="residue").matrix,
    )


def test_residue_map_requires_residues():
    xyz = ms.read(os.path.join(DATA, "helix_201.xyz"))
    with pytest.raises(ValueError):
        xyz.contact_map(level="residue")


def test_invalid_level_and_method():
    with pytest.raises(ValueError):
        ca_chain().contact_map(level="bogus")
    with pytest.raises(ValueError):
        ca_chain().contact_map(level="residue", method="bogus")


def two_chain():
    """Chains A (resid 1,2) and B (resid 1): A2 and B1 are 4 A apart."""
    return Molecule(
        np.array([[0.0, 0, 0], [5.0, 0, 0], [9.0, 0, 0]]), ["C", "C", "C"],
        atom_names=["CA", "CA", "CA"], resnames=["ALA", "ALA", "ALA"],
        resids=np.array([1, 2, 1]), chains=["A", "A", "B"],
    )


def test_contact_metrics():
    cm = ca_chain().contact_map(cutoff=8.0, level="residue", method="ca")
    assert cm.n_contacts == 2                       # 0-1 and 1-2
    # mean |i-j| over contacts = 1; normalised by R = 3.
    assert cm.contact_order() == pytest.approx(1.0 / 3.0)
    empty = ca_chain().contact_map(cutoff=1.0, level="residue", method="ca")
    assert empty.n_contacts == 0 and empty.contact_order() == 0.0


def test_min_seq_sep_drops_local_contacts():
    full = ca_chain().contact_map(cutoff=8.0, level="residue", method="ca")
    sep = ca_chain().contact_map(cutoff=8.0, level="residue", method="ca", min_seq_sep=2)
    assert full.n_contacts == 2
    assert sep.n_contacts == 0                       # both contacts are i,i+1


def test_chain_mode_splits_intra_and_inter():
    mol = two_chain()
    allc = mol.contact_map(cutoff=8.0, level="residue", method="ca", chain_mode="all")
    intra = mol.contact_map(cutoff=8.0, level="residue", method="ca", chain_mode="intra")
    inter = mol.contact_map(cutoff=8.0, level="residue", method="ca", chain_mode="inter")
    assert allc.n_contacts == intra.n_contacts + inter.n_contacts
    assert intra.n_contacts == 1                     # A1-A2 (5 A)
    assert inter.n_contacts == 1                     # A2-B1 (4 A)
    with pytest.raises(ValueError):
        mol.contact_map(level="residue", chain_mode="bogus")


def test_min_method_matches_brute_force():
    mol = ms.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    groups = list(mol.residue_groups())[:30]         # a contiguous slice for speed
    sub = mol.take(np.concatenate([np.array(idx) for idx, *_ in groups]))
    new = sub.contact_map(cutoff=4.5, level="residue", method="min").matrix
    blocks = [sub.coords[idx] for idx, *_ in sub.residue_groups()]
    ref = np.zeros_like(new)
    for a in range(len(blocks)):
        for b in range(a + 1, len(blocks)):
            d = np.linalg.norm(blocks[a][:, None] - blocks[b][None, :], axis=-1).min()
            ref[a, b] = ref[b, a] = 1.0 if d < 4.5 else 0.0
    np.testing.assert_array_equal(new, ref)


def test_ensemble_contact_frequency():
    models = ms.read_pdb_models(os.path.join(DATA, "1aml.pdb"))
    freq = ms.ensemble_contact_frequency(models, cutoff=8.0)
    n = len(list(models[0].residue_groups()))
    assert freq.matrix.shape == (n, n)
    assert freq.is_frequency
    assert (freq.matrix >= 0).all() and (freq.matrix <= 1).all()
    assert np.array_equal(freq.matrix, freq.matrix.T)
    assert np.diag(freq.matrix).sum() == 0
    # the whole point: some pairs are in contact in only a fraction of models
    assert ((freq.matrix > 0) & (freq.matrix < 1)).any()


def test_plot_contact_map(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    ax = ca_chain().contact_map(level="residue").plot(show=False)
    assert ax is not None
    freq = ms.ensemble_contact_frequency(
        ms.read_pdb_models(os.path.join(DATA, "1aml.pdb"))[:5]
    )
    assert freq.plot(show=False) is not None


def test_plot_distance_matrix():
    import matplotlib

    matplotlib.use("Agg")
    ax = ca_chain().plot_distance_matrix(show=False)
    assert ax is not None


def test_contact_map_difference():
    mol = ca_chain()
    map1 = mol.contact_map(cutoff=6.0, level="residue")
    map2 = mol.contact_map(cutoff=8.0, level="residue")

    mol2 = Molecule(
        np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]]), ["C", "C", "C"],
        atom_names=["CA", "CA", "CA"], resnames=["ALA", "ALA", "ALA"],
        resids=np.array([1, 2, 3]), chains=["A", "A", "A"],
    )
    map3 = mol2.contact_map(cutoff=6.0, level="residue")

    #ALA3 shifted in mol2 so residue 2-3 is in contact in map3 but not map1
    diff = map1 - map3
    assert isinstance(diff, ContactMap)
    assert diff.cutoff == 6.0
    assert diff.level == "residue"
    assert diff.labels == ["A:ALA1", "A:ALA2", "A:ALA3"]
    assert diff.matrix[1, 2] == -1.0
    assert diff.matrix[0, 1] == 0.0
    assert diff.is_difference
    assert not diff.is_frequency

    # Mismatch checks
    with pytest.raises(ValueError, match="different cutoffs"):
        _ = map1 - map2

    atom_map = mol.contact_map(cutoff=6.0, level="atom")
    with pytest.raises(ValueError, match="different levels"):
        _ = map1 - atom_map

    two_res = Molecule(
        np.array([[0.0, 0, 0], [5.0, 0, 0]]), ["C", "C"],
        atom_names=["CA", "CA"], resnames=["ALA", "ALA"],
        resids=np.array([1, 2]), chains=["A", "A"],
    )
    map_two = two_res.contact_map(cutoff=6.0, level="residue")
    with pytest.raises(ValueError, match="different shapes"):
        _ = map1 - map_two

    mol_diff_labels = Molecule(
        np.array([[0.0, 0, 0], [5.0, 0, 0], [12.0, 0, 0]]), ["C", "C", "C"],
        atom_names=["CA", "CA", "CA"], resnames=["ALA", "GLY", "ALA"],
        resids=np.array([1, 2, 3]), chains=["A", "A", "A"],
    )
    map_diff_labels = mol_diff_labels.contact_map(cutoff=6.0, level="residue")
    with pytest.raises(ValueError, match="mismatched residue/atom labels"):
        _ = map1 - map_diff_labels

    with pytest.raises(TypeError):
        _ = map1 - 5


def test_difference_flag_holds_without_negative_values():
    # A "gained-only" difference (state B's contacts are a subset of A's) has no
    # negative entries, and an identical-state difference is all zeros. Neither
    # could be inferred from values, so is_difference is tracked explicitly.
    bigger = ContactMap(
        matrix=np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        level="residue", cutoff=8.0, labels=["A:ALA1", "A:ALA2", "A:ALA3"],
    )
    smaller = ContactMap(
        matrix=np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        level="residue", cutoff=8.0, labels=["A:ALA1", "A:ALA2", "A:ALA3"],
    )

    gained_only = bigger - smaller
    assert not np.any(gained_only.matrix < 0)  # no lost contacts -> no negatives
    assert gained_only.is_difference           # ...but still a difference map
    assert not gained_only.is_frequency        # boolean inputs -> discrete diff

    identical = bigger - bigger
    assert not np.any(identical.matrix != 0)   # all zeros
    assert identical.is_difference

    # An ordinary map is never a difference.
    assert bigger.is_difference is False


def test_frequency_difference_map():
    map_a = ContactMap(
        matrix=np.array([[0.0, 0.8, 0.1], [0.8, 0.0, 0.4], [0.1, 0.4, 0.0]]),
        level="residue",
        cutoff=8.0,
        labels=["A:ALA1", "A:ALA2", "A:ALA3"],
    )
    map_b = ContactMap(
        matrix=np.array([[0.0, 0.2, 0.5], [0.2, 0.0, 0.4], [0.5, 0.4, 0.0]]),
        level="residue",
        cutoff=8.0,
        labels=["A:ALA1", "A:ALA2", "A:ALA3"],
    )

    assert map_a.is_frequency
    assert map_b.is_frequency

    diff = map_a - map_b
    assert diff.is_difference
    assert diff.is_frequency
    np.testing.assert_array_almost_equal(
        diff.matrix,
        np.array([[0.0, 0.6, -0.4], [0.6, 0.0, 0.0], [-0.4, 0.0, 0.0]]),
    )


def test_plot_contact_map_difference():
    import matplotlib

    matplotlib.use("Agg")

    map_a = ContactMap(
        matrix=np.array([[0.0, 0.8, 0.1], [0.8, 0.0, 0.4], [0.1, 0.4, 0.0]]),
        level="residue",
        cutoff=8.0,
        labels=["A:ALA1", "A:ALA2", "A:ALA3"],
    )
    map_b = ContactMap(
        matrix=np.array([[0.0, 0.2, 0.5], [0.2, 0.0, 0.4], [0.5, 0.4, 0.0]]),
        level="residue",
        cutoff=8.0,
        labels=["A:ALA1", "A:ALA2", "A:ALA3"],
    )

    diff = map_a - map_b
    ax = diff.plot(show=False)
    assert ax is not None
