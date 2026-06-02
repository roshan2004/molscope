"""Tests for ensemble RMSD clustering and the RMSD heatmap."""

import os

import numpy as np
import pytest

import molscope as ms
from molscope import Molecule

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def two_conformer_set():
    """Four models: two near-copies of a straight chain, two of a bent chain.

    Differences are conformational (not rigid), so they survive Kabsch alignment
    and form two clean clusters.
    """
    straight = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    bent = np.array([[0, 0, 0], [1, 0, 0], [2, 1.5, 0], [3, 3, 0]], dtype=float)
    rng = np.random.default_rng(0)

    def jitter(coords):
        return Molecule(coords + rng.normal(scale=0.01, size=coords.shape), ["C"] * 4)

    return [jitter(straight), jitter(straight), jitter(bent), jitter(bent)]


def test_rmsd_matrix_top_level_matches_ensemble():
    models = ms.read_pdb_models(os.path.join(DATA, "1aml.pdb"))[:4]
    a = ms.rmsd_matrix(models, align=True)
    b = ms.ensemble.rmsd_matrix(models, align=True)
    np.testing.assert_array_equal(a, b)


def test_cluster_separates_two_conformers():
    cl = ms.cluster(two_conformer_set(), n_clusters=2)
    assert cl.n_clusters == 2
    # models 0,1 together; 2,3 together; the two groups differ
    assert cl.labels[0] == cl.labels[1]
    assert cl.labels[2] == cl.labels[3]
    assert cl.labels[0] != cl.labels[2]


def test_groups_partition_all_models():
    cl = ms.cluster(two_conformer_set(), n_clusters=2)
    members = sorted(i for ids in cl.groups().values() for i in ids)
    assert members == [0, 1, 2, 3]


def test_representatives_are_valid_members():
    cl = ms.cluster(two_conformer_set(), n_clusters=2)
    for cid, idx in cl.representatives().items():
        assert idx in cl.groups()[cid]


def test_order_is_a_permutation():
    cl = ms.cluster(two_conformer_set(), n_clusters=2)
    assert sorted(cl.order.tolist()) == [0, 1, 2, 3]


def test_cutoff_controls_granularity():
    models = ms.read_pdb_models(os.path.join(DATA, "1aml.pdb"))
    coarse = ms.cluster(models, cutoff=100.0)   # everything in one cluster
    fine = ms.cluster(models, cutoff=0.1)        # almost every model separate
    assert coarse.n_clusters == 1
    assert fine.n_clusters > coarse.n_clusters


def test_reuses_supplied_matrix():
    models = two_conformer_set()
    mat = ms.rmsd_matrix(models)
    cl = ms.cluster(models, matrix=mat, n_clusters=2)
    np.testing.assert_array_equal(cl.matrix, mat)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        ms.cluster(two_conformer_set(), method="kmeans")


def test_single_model_is_one_cluster():
    cl = ms.cluster([Molecule(np.zeros((3, 3)), ["C", "C", "C"])])
    assert cl.n_clusters == 1


def test_plot_rmsd_heatmap(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    cl = ms.cluster(two_conformer_set(), n_clusters=2)
    ax = ms.plot_rmsd_heatmap(cl.matrix, order=cl.order, show=False)
    assert ax is not None


# -- dynamical cross-correlation (DCCM) -------------------------------------


def coupled_motion_set():
    """Four models where atoms 0,1 move together and atom 2 moves opposite.

    Displacements are along x; building them directly (no alignment) gives an
    exact, hand-checkable correlation structure.
    """
    base = np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])
    models = []
    for d in (1.0, -1.0, 2.0, -2.0):  # zero-mean displacements
        coords = base + np.array([[d, 0, 0], [d, 0, 0], [-d, 0, 0]])
        models.append(Molecule(coords, ["C", "C", "C"]))
    return models


def test_cross_correlation_recovers_known_coupling():
    c = ms.cross_correlation(coupled_motion_set(), align=False)
    expected = np.array([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    np.testing.assert_allclose(c, expected, atol=1e-12)


def test_cross_correlation_is_symmetric_bounded_and_unit_diagonal():
    c = ms.cross_correlation(ms.read_pdb_models(os.path.join(DATA, "1aml.pdb")))
    n = len(ms.read_pdb_models(os.path.join(DATA, "1aml.pdb"))[0])
    assert c.shape == (n, n)
    np.testing.assert_allclose(c, c.T, atol=1e-9)
    np.testing.assert_allclose(np.diag(c), 1.0, atol=1e-9)
    assert c.min() >= -1.0 - 1e-9 and c.max() <= 1.0 + 1e-9


def test_cross_correlation_handles_static_atoms():
    # atom 2 never moves; its coupling to the others must be 0, diagonal 1
    base = np.array([[0.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])
    models = [
        Molecule(base + np.array([[d, 0, 0], [d, 0, 0], [0, 0, 0]]), ["C", "C", "C"])
        for d in (1.0, -1.0)
    ]
    c = ms.cross_correlation(models, align=False)
    assert c[2, 0] == 0.0 and c[2, 1] == 0.0
    assert c[2, 2] == 1.0


def test_cross_correlation_residue_level_via_alpha_carbons():
    models = ms.read_pdb_models(os.path.join(DATA, "1aml.pdb"))
    ca = [m.alpha_carbons() for m in models]
    c = ms.cross_correlation(ca)
    assert c.shape == (40, 40)  # one node per residue


def test_plot_cross_correlation():
    import matplotlib

    matplotlib.use("Agg")
    c = ms.cross_correlation(coupled_motion_set(), align=False)
    ax = ms.plot_cross_correlation(c, show=False)
    assert ax is not None
