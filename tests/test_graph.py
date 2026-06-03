"""Tests for the molecular graph layer and its exporters."""

import os
import sys

import numpy as np
import pytest

import molscope as ms
from molscope import MolecularGraph, Molecule, ResidueContactGraph
from molscope.graph import (
    edge_feature_names,
    node_feature_names,
    residue_edge_feature_names,
    residue_node_feature_names,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def water():
    coords = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    return Molecule(coords, ["O", "H", "H"], name="water")


def residue_toy():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.5, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ])
    return Molecule(
        coords,
        ["C", "O", "N", "C", "S"],
        name="residue-toy",
        atom_names=["CA", "CB", "CA", "CB", "CA"],
        resnames=["ALA", "ALA", "GLY", "GLY", "SER"],
        resids=[1, 1, 2, 2, 3],
        chains=["A", "A", "A", "A", "A"],
    )


# -- core graph (no optional deps) ------------------------------------------


def test_to_graph_nodes_and_edges():
    g = water().to_graph()
    assert isinstance(g, MolecularGraph)
    assert g.n_atoms == 3
    assert g.n_bonds == 2  # the two O-H bonds
    np.testing.assert_array_equal(g.atomic_numbers, [8, 1, 1])
    assert g.masses[0] == pytest.approx(15.999)


def test_graph_edge_distances_match_geometry():
    g = water().to_graph()
    # every edge distance equals the coordinate distance of its endpoints
    for (i, j), d in zip(g.edges, g.edge_distances):
        assert d == pytest.approx(np.linalg.norm(g.coords[i] - g.coords[j]))


def test_node_features_shape():
    feats = water().to_graph().node_features()
    assert feats.shape == (3, 2)  # [atomic_number, mass]


def test_graph_feature_presets_have_stable_names_and_shapes():
    g = water().to_graph()
    x, e, node_names, edge_names = g.feature_matrices(return_names=True)
    assert node_names == node_feature_names("ml")
    assert edge_names == edge_feature_names("ml")
    assert x.shape == (3, len(node_names))
    assert e.shape == (2, len(edge_names))
    assert "element_O" in node_names
    assert "formal_charge" in node_names
    assert "bond_order" in edge_names


def test_graph_basic_feature_presets_include_charge_and_bond_order():
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]]),
        ["N", "O"],
        bond_index=[[0, 1]],
        bond_orders=[2],
        formal_charges=[1, -1],
    )
    g = mol.to_graph()
    x, node_names = g.node_features("basic", return_names=True)
    e, edge_names = g.edge_features("basic", return_names=True)
    assert node_names == ["atomic_number", "mass", "formal_charge"]
    assert edge_names == ["distance", "bond_order"]
    np.testing.assert_array_equal(x[:, node_names.index("formal_charge")], [1.0, -1.0])
    np.testing.assert_array_equal(e[:, edge_names.index("bond_order")], [2.0])


def test_graph_ml_preset_marks_aromatic_bond_order():
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        ["C", "C"],
        bond_index=[[0, 1]],
        bond_orders=[1.5],
    )
    e, names = mol.to_graph().edge_features("ml", return_names=True)
    assert e[0, names.index("aromatic")] == 1.0


def test_to_graph_accepts_explicit_bonds():
    g = water().to_graph(bonds=[[0, 1]])
    assert g.n_bonds == 1


def test_to_graph_preserves_explicit_bond_orders():
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]]),
        ["C", "C"],
        bond_index=[[0, 1]],
        bond_orders=[2],
    )
    g = mol.to_graph()
    np.testing.assert_array_equal(g.edge_types, [2.0])


def line_molecule(n=6):
    coords = np.array([[float(i), 0.0, 0.0] for i in range(n)])
    return Molecule(coords, ["C"] * n, name="line")


def two_chain():
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.5, 0.0, 0.0]]
    )
    return Molecule(
        coords, ["C", "C", "C", "C"], name="two-chain",
        resids=[1, 2, 1, 2], chains=["A", "A", "B", "B"],
    )


# -- configurable edge construction (k-NN, sequence separation) -------------


def test_knn_edges_helper_is_undirected_and_loop_free():
    from molscope.graph import knn_edges

    edges = knn_edges(line_molecule(6).coords, k=2)
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert (edges[:, 0] < edges[:, 1]).all()  # i < j, no self-loops
    # union-symmetrised k-NN: every node keeps at least its own k neighbours
    degree = np.bincount(edges.reshape(-1), minlength=6)
    assert (degree >= 2).all()


def test_knn_edges_helper_is_importable_from_package():
    assert ms.knn_edges(line_molecule(3).coords, k=1).shape[1] == 2


def test_knn_edges_numpy_fallback_matches_scipy(monkeypatch):
    coords = line_molecule(6).coords
    expected = ms.knn_edges(coords, 2)
    # Block the scipy.spatial import so knn_edges takes the dense NumPy path.
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    np.testing.assert_array_equal(ms.knn_edges(coords, 2), expected)


def test_to_graph_knn_accepts_explicit_bond_orders():
    coords = line_molecule(3).coords
    n_edges = len(ms.knn_edges(coords, 2))
    g = line_molecule(3).to_graph(knn=2, bond_orders=[2.0] * n_edges)
    np.testing.assert_array_equal(g.edge_types, [2.0] * n_edges)


def test_to_graph_knn_infer_orders_runs():
    g = water().to_graph(knn=2, infer_orders=True)
    assert g.n_bonds == 3  # complete graph on the three atoms
    assert len(g.edge_types) == g.n_bonds


def test_to_graph_knn_builds_geometric_edges():
    g = line_molecule(6).to_graph(knn=2)
    degree = np.bincount(g.edges.reshape(-1), minlength=6)
    assert (degree >= 2).all()
    # k-NN edges have no chemical order, so they default to 1.0
    np.testing.assert_array_equal(g.edge_types, np.ones(g.n_bonds))


def test_to_graph_knn_caps_at_n_minus_one_and_completes_graph():
    g = line_molecule(5).to_graph(knn=100)
    assert g.n_bonds == 5 * 4 // 2  # complete graph


def test_to_graph_knn_rejects_zero_k():
    with pytest.raises(ValueError):
        line_molecule(4).to_graph(knn=0)


def test_to_graph_knn_and_explicit_bonds_are_mutually_exclusive():
    with pytest.raises(ValueError, match="at most one"):
        line_molecule(4).to_graph(knn=2, bonds=[[0, 1]])


def test_to_graph_radius_builds_proximity_edges():
    # six atoms on a line at spacing 1.0; radius 1.5 links only adjacent atoms
    g = line_molecule(6).to_graph(radius=1.5)
    pairs = {tuple(e) for e in g.edges}
    assert pairs == {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)}
    np.testing.assert_array_equal(g.edge_types, np.ones(g.n_bonds))
    # every edge is within the cutoff
    assert (g.edge_distances <= 1.5).all()


def test_to_graph_radius_wider_cutoff_adds_more_edges():
    near = line_molecule(6).to_graph(radius=1.5).n_bonds
    far = line_molecule(6).to_graph(radius=2.5).n_bonds
    assert far > near


def test_to_graph_radius_combines_with_min_seq_sep():
    filtered = residue_toy().to_graph(radius=4.0, min_seq_sep=1)
    seps = np.abs(
        np.array(residue_toy().resids)[filtered.edges[:, 0]]
        - np.array(residue_toy().resids)[filtered.edges[:, 1]]
    )
    assert (seps >= 1).all()


def test_to_graph_radius_and_knn_are_mutually_exclusive():
    with pytest.raises(ValueError, match="at most one"):
        line_molecule(4).to_graph(knn=2, radius=2.0)


def tetrahedron():
    coords = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return Molecule(coords, ["C", "C", "C", "C"], name="tetra")


def test_delaunay_edges_helper_undirected_and_loop_free():
    pytest.importorskip("scipy")
    from molscope.graph import delaunay_edges

    rng = np.random.RandomState(0)
    edges = delaunay_edges(rng.rand(12, 3))
    assert edges.shape[1] == 2
    assert (edges[:, 0] < edges[:, 1]).all()  # i < j, no self-loops
    assert len(edges) == len({tuple(e) for e in edges})  # unique


def test_delaunay_small_sets_are_complete_graphs():
    from molscope.graph import delaunay_edges

    # < 4 atoms: complete graph without invoking Qhull
    assert len(delaunay_edges(np.zeros((1, 3)))) == 0
    assert len(delaunay_edges(np.array([[0.0, 0, 0], [1, 0, 0], [2, 1, 0]]))) == 3


def test_delaunay_coplanar_input_uses_joggle_fallback():
    pytest.importorskip("scipy")
    from molscope.graph import delaunay_edges

    # All atoms in the z=0 plane: precise Delaunay fails, the QJ joggle recovers.
    grid = np.array([[x, y, 0.0] for x in range(3) for y in range(3)], dtype=float)
    edges = delaunay_edges(grid)
    assert len(edges) > 0
    assert (edges[:, 0] < edges[:, 1]).all()


def test_to_graph_delaunay_builds_edges():
    pytest.importorskip("scipy")
    # a single tetrahedron is one simplex -> complete graph on 4 atoms
    g = tetrahedron().to_graph(delaunay=True)
    assert g.n_bonds == 6
    np.testing.assert_array_equal(g.edge_types, np.ones(g.n_bonds))


def test_to_graph_delaunay_is_mutually_exclusive_with_other_modes():
    with pytest.raises(ValueError, match="at most one"):
        tetrahedron().to_graph(delaunay=True, knn=2)


def test_to_graph_delaunay_respects_min_seq_sep():
    pytest.importorskip("scipy")
    mol = Molecule(
        tetrahedron().coords, ["C"] * 4, resids=[1, 1, 2, 2], chains=["A"] * 4
    )
    g = mol.to_graph(delaunay=True, min_seq_sep=1)
    resids = np.array([1, 1, 2, 2])
    seps = np.abs(resids[g.edges[:, 0]] - resids[g.edges[:, 1]])
    assert (seps >= 1).all()  # intra-residue Delaunay edges dropped


def test_delaunay_requires_scipy(monkeypatch):
    import sys

    from molscope.graph import delaunay_edges

    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    with pytest.raises(ImportError, match="SciPy"):
        delaunay_edges(np.random.RandomState(1).rand(8, 3))


def test_min_seq_sep_drops_local_same_chain_edges():
    g = residue_toy().to_graph(knn=2)
    before = g.n_bonds
    filtered = residue_toy().to_graph(knn=2, min_seq_sep=1)
    # intra-residue edges (separation 0) are removed
    seps = np.abs(
        np.array(residue_toy().resids)[filtered.edges[:, 0]]
        - np.array(residue_toy().resids)[filtered.edges[:, 1]]
    )
    assert (seps >= 1).all()
    assert filtered.n_bonds < before


def test_min_seq_sep_without_chains_treats_structure_as_one_chain():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    mol = Molecule(coords, ["C", "C", "C"], resids=[1, 1, 5])
    g = mol.to_graph(knn=2, min_seq_sep=2)
    resids = np.array([1, 1, 5])
    i, j = g.edges[:, 0], g.edges[:, 1]
    # the two resid-1 atoms (separation 0) are no longer linked
    assert (np.abs(resids[i] - resids[j]) >= 2).all()


def test_min_seq_sep_keeps_cross_chain_edges():
    g = two_chain().to_graph(knn=3, min_seq_sep=2)
    resids = np.array(two_chain().resids)
    chains = np.array(two_chain().chains)
    i, j = g.edges[:, 0], g.edges[:, 1]
    # every surviving same-chain edge respects the separation threshold
    same = chains[i] == chains[j]
    assert (np.abs(resids[i] - resids[j])[same] >= 2).all()
    # the inter-chain edges are all retained
    assert (~same).sum() == 4


def test_min_seq_sep_filters_covalent_bonds_too():
    # the two covalent bonds in residue_toy are both intra-residue (sep 0)
    assert residue_toy().to_graph().n_bonds == 2
    assert residue_toy().to_graph(min_seq_sep=1).n_bonds == 0


def test_min_seq_sep_requires_residue_ids():
    with pytest.raises(ValueError, match="residue ids"):
        water().to_graph(min_seq_sep=1)


def test_graph_carries_metadata():
    mol = ms.read_pdb(os.path.join(DATA, "1fqy.pdb"))
    g = mol.to_graph()
    assert g.n_atoms == 1661
    assert len(g.chains) == 1661 and g.chains[0] == "A"


def test_graph_carries_formal_charges():
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        ["O", "C"],
        bond_index=[[0, 1]],
        formal_charges=[-1, 1],
    )
    g = mol.to_graph()
    np.testing.assert_array_equal(g.formal_charges, [-1, 1])


def test_graph_carries_virtual_site_flags():
    cg = residue_toy().coarse_grain(
        "residue_com",
        virtual_sites=[{"name": "MID", "parents": [0, 1]}],
    )
    g = cg.to_graph()
    np.testing.assert_array_equal(g.virtual_sites, [False, False, False, True])


def test_graph_can_attach_rdkit_aromatic_features():
    pytest.importorskip("rdkit")
    mol = Molecule(
        np.array([
            [1.396, 0.000, 0.000],
            [0.698, 1.209, 0.000],
            [-0.698, 1.209, 0.000],
            [-1.396, 0.000, 0.000],
            [-0.698, -1.209, 0.000],
            [0.698, -1.209, 0.000],
        ]),
        ["C"] * 6,
        bond_index=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]],
        bond_orders=[1.5] * 6,
    )
    g = mol.to_graph(include_chemical_features=True)
    assert g.aromatic_atoms.all()
    assert g.aromatic_bonds.all()


# -- networkx (in dev deps, tested for real) --------------------------------


def test_to_networkx():
    nx = pytest.importorskip("networkx")
    G = water().to_networkx()
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G.nodes[0]["element"] == "O"
    assert G.nodes[0]["atomic_number"] == 8
    # edge attributes present
    i, j = next(iter(G.edges))
    assert "distance" in G.edges[i, j]
    assert G.edges[i, j]["covalent"] is True   # bond-derived graph


def test_networkx_covalent_flag_on_knn_graph():
    pytest.importorskip("networkx")
    G = line_molecule(6).to_graph(knn=3).to_networkx()
    flags = [d["covalent"] for *_, d in G.edges(data=True)]
    # a spatial graph has both real bonds (short edges) and pure contacts
    assert any(flags) and not all(flags)


def test_networkx_includes_residue_metadata():
    pytest.importorskip("networkx")
    G = ms.read_pdb(os.path.join(DATA, "1fqy.pdb")).to_networkx()
    assert G.nodes[0]["chain"] == "A"
    assert G.nodes[0]["resname"] == "LYS"


def test_networkx_includes_formal_charge():
    pytest.importorskip("networkx")
    mol = Molecule(np.zeros((1, 3)), ["N"], formal_charges=[1])
    G = mol.to_networkx()
    assert G.nodes[0]["formal_charge"] == 1


def test_networkx_includes_virtual_site_flag():
    pytest.importorskip("networkx")
    cg = residue_toy().coarse_grain(
        "residue_com",
        virtual_sites=[{"name": "MID", "parents": [0, 1]}],
    )
    G = cg.to_networkx()
    assert G.nodes[3]["virtual_site"] is True
    assert G.nodes[0]["virtual_site"] is False


# -- residue contact graph --------------------------------------------------


def test_residue_contact_graph_nodes_edges_and_features():
    g = residue_toy().to_residue_contact_graph(cutoff=4.0, method="ca")
    assert isinstance(g, ResidueContactGraph)
    assert g.n_residues == 3
    assert g.n_contacts == 1
    assert g.labels == ["A:ALA1", "A:GLY2", "A:SER3"]
    np.testing.assert_array_equal(g.edges, [[0, 1]])
    np.testing.assert_array_equal(g.residue_sizes, [2, 2, 1])
    assert g.edge_distances[0] == pytest.approx(3.0)

    x, e, node_names, edge_names = g.feature_matrices(return_names=True)
    assert node_names == residue_node_feature_names("ml")
    assert edge_names == residue_edge_feature_names("ml")
    assert x.shape == (3, len(node_names))
    assert e.shape == (1, len(edge_names))
    assert x[0, node_names.index("residue_ALA")] == 1.0
    assert e[0, edge_names.index("contact_ca")] == 1.0


def test_residue_contact_graph_carries_rich_residue_ids():
    g = ms.read(os.path.join(FIXTURES, "ugly_residue_ids.pdb")).to_residue_contact_graph(
        cutoff=2.0,
        method="ca",
    )
    assert g.labels[2:4] == ["A:SER100A", "A:THR100B"]
    assert g.icodes[2:4] == ["A", "B"]
    assert [rid.label() for rid in g.residue_ids[2:4]] == ["A:SER100A", "A:THR100B"]


def test_residue_contact_graph_filters_sequence_local_contacts():
    g = residue_toy().to_residue_contact_graph(cutoff=4.0, method="ca", min_seq_sep=2)
    assert g.n_contacts == 0


def test_residue_contact_graph_min_method_uses_closest_atom_distance():
    g = residue_toy().to_residue_contact_graph(cutoff=2.6, method="min")
    assert g.n_contacts == 1
    assert g.edge_distances[0] == pytest.approx(2.5)
    assert g.edge_types == ["min"]


def test_residue_contact_graph_requires_residue_metadata():
    with pytest.raises(ValueError, match="residue contact graph needs residue information"):
        water().to_residue_contact_graph()


def test_residue_contact_graph_to_networkx():
    nx = pytest.importorskip("networkx")
    G = residue_toy().to_residue_contact_graph(cutoff=4.0).to_networkx()
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 1
    assert G.nodes[0]["resname"] == "ALA"
    assert G.nodes[0]["label"] == "A:ALA1"
    edge = G.edges[0, 1]
    assert edge["contact_type"] == "ca"
    assert edge["distance"] == pytest.approx(3.0)


def test_molecule_ml_shortcuts_forward_feature_presets(monkeypatch):
    seen = {}

    class FakeGraph:
        def to_pyg_data(self, node_preset="default", edge_preset="default"):
            seen["pyg"] = (node_preset, edge_preset)
            return "pyg-data"

        def to_dgl_graph(self, node_preset="default", edge_preset="default"):
            seen["dgl"] = (node_preset, edge_preset)
            return "dgl-graph"

    def fake_to_graph(self, **kwargs):
        seen["graph_kwargs"] = kwargs
        return FakeGraph()

    monkeypatch.setattr(Molecule, "to_graph", fake_to_graph)
    mol = water()

    assert (
        mol.to_pyg_data(node_preset="ml", edge_preset="basic", tolerance=1.1)
        == "pyg-data"
    )
    assert seen["graph_kwargs"] == {"tolerance": 1.1}
    assert seen["pyg"] == ("ml", "basic")

    assert mol.to_dgl_graph(node_preset="basic", edge_preset="ml") == "dgl-graph"
    assert seen["dgl"] == ("basic", "ml")


# -- PyTorch Geometric / DGL (skipped unless installed) ---------------------


def test_to_pyg_data():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    data = water().to_pyg_data()
    assert data.num_nodes == 3
    assert data.x.shape == (3, 2)
    assert data.pos.shape == (3, 3)
    # 2 undirected bonds -> 4 directed edges
    assert data.edge_index.shape == (2, 4)
    assert data.edge_attr.shape == (4, 1)
    assert data.bond_order.shape == (4,)
    assert data.formal_charge.shape == (3,)
    assert data.virtual_site.shape == (3,)


def test_to_pyg_data_forwards_knn():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    g = line_molecule(6).to_graph(knn=2)
    data = line_molecule(6).to_pyg_data(knn=2)
    # directed edges are double the undirected k-NN edge count
    assert data.edge_index.shape == (2, 2 * g.n_bonds)


def test_to_pyg_data_displacement_is_antisymmetric():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    data = water().to_graph().to_pyg_data(include_displacement=True)
    vec = data.edge_vec.numpy()
    n = water().to_graph().n_bonds
    assert vec.shape == (2 * n, 3)
    # the j->i half is the negation of the i->j half (r_ij = -r_ji)
    np.testing.assert_allclose(vec[:n], -vec[n:], atol=1e-6)
    # |edge_vec| equals the stored edge distance
    np.testing.assert_allclose(
        np.linalg.norm(vec[:n], axis=1), water().to_graph().edge_distances, atol=1e-5
    )


def test_to_pyg_data_is_covalent_flag():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    # covalent graph: every directed edge is a bond
    cov = water().to_pyg_data()
    assert bool(cov.is_covalent.all())
    # k-NN graph: only the edges coinciding with bonds are flagged
    g = line_molecule(6).to_graph(knn=3)
    data = line_molecule(6).to_pyg_data(knn=3)
    assert int(data.is_covalent.sum()) == 2 * int(g.covalent_edge_flags().sum())
    assert not bool(data.is_covalent.all())


def test_to_pyg_data_edge_attrs_track_global_node_and_self_loops():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    data = water().to_graph().to_pyg_data(
        include_global_node=True, include_self_loops=True, include_displacement=True
    )
    n_edges = data.edge_index.shape[1]
    assert data.is_covalent.shape == (n_edges,)
    assert data.edge_vec.shape == (n_edges, 3)
    # the 4 real directed bond edges are covalent; global/self-loop edges are not
    assert int(data.is_covalent.sum()) == 4


def test_to_pyg_data_displacement_handles_edgeless_graph():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    lone = Molecule(np.zeros((1, 3)), ["C"]).to_graph().to_pyg_data(include_displacement=True)
    assert lone.edge_vec.shape == (0, 3)
    assert lone.is_covalent.shape == (0,)


def test_to_dgl_graph():
    pytest.importorskip("dgl")
    pytest.importorskip("torch")
    g = water().to_graph().to_dgl_graph(include_displacement=True)
    assert g.num_nodes() == 3
    assert g.num_edges() == 4
    assert g.ndata["feat"].shape == (3, 2)
    assert g.ndata["formal_charge"].shape == (3,)
    assert g.edata["bond_order"].shape == (4,)
    assert g.edata["is_covalent"].shape == (4,)
    assert g.edata["edge_vec"].shape == (4, 3)


def test_to_dgl_graph_edge_attrs_track_global_node_and_self_loops():
    pytest.importorskip("dgl")
    pytest.importorskip("torch")
    g = water().to_graph().to_dgl_graph(
        include_global_node=True, include_self_loops=True, include_displacement=True
    )
    n_edges = g.num_edges()
    assert g.edata["is_covalent"].shape == (n_edges,)
    assert g.edata["edge_vec"].shape == (n_edges, 3)
    assert int(g.edata["is_covalent"].sum()) == 4


def test_residue_contact_graph_to_pyg_data():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    data = residue_toy().to_residue_contact_graph(cutoff=4.0).to_pyg_data(
        node_preset="ml",
        edge_preset="ml",
    )
    assert data.num_nodes == 3
    assert data.x.shape == (3, len(residue_node_feature_names("ml")))
    assert data.edge_index.shape == (2, 2)
    assert data.edge_attr.shape == (2, len(residue_edge_feature_names("ml")))
    assert data.residue_size.tolist() == [2, 2, 1]


def test_residue_contact_graph_to_dgl_graph():
    pytest.importorskip("dgl")
    pytest.importorskip("torch")
    g = residue_toy().to_residue_contact_graph(cutoff=4.0).to_dgl_graph(
        node_preset="ml",
        edge_preset="ml",
    )
    assert g.num_nodes() == 3
    assert g.num_edges() == 2
    assert g.ndata["feat"].shape == (3, len(residue_node_feature_names("ml")))
    assert g.edata["feat"].shape == (2, len(residue_edge_feature_names("ml")))


# -- residue interaction labels ---------------------------------------------


def _edge_label(coords, elements, atom_names, resnames, resids, chains=None):
    """Label the single edge (0,1) of a hand-built two-residue molecule."""
    from molscope.graph import _residue_interaction_labels

    n = len(elements)
    chains = chains if chains is not None else ["A"] * n
    mol = Molecule(np.array(coords, dtype=float), elements, atom_names=atom_names,
                   resnames=resnames, resids=resids, chains=chains)
    groups = list(mol.residue_groups())
    atom_groups = [g.atom_indices for g in groups]
    rns = [g.residue_id.resname for g in groups]
    rids = np.array([g.residue_id.resid for g in groups])
    chs = [g.residue_id.chain for g in groups]
    return _residue_interaction_labels(mol, atom_groups, rns, rids, chs, np.array([[0, 1]]))[0]


def test_interaction_label_disulfide():
    assert _edge_label(
        [[0, 0, 0], [1, 0, 0], [5, 0, 0], [2.0, 0, 0]],
        ["C", "S", "C", "S"], ["CA", "SG", "CA", "SG"],
        ["CYS", "CYS", "CYS", "CYS"], [1, 1, 2, 2],
    ) == "disulfide"


def test_interaction_label_salt_bridge_beats_adjacency():
    # Arg NH1 ~2.5 A from Asp OD1; salt_bridge has precedence over covalent.
    assert _edge_label(
        [[0, 0, 0], [1, 0, 0], [6, 0, 0], [3.5, 0, 0]],
        ["C", "N", "C", "O"], ["CA", "NH1", "CA", "OD1"],
        ["ARG", "ARG", "ASP", "ASP"], [1, 1, 2, 2],
    ) == "salt_bridge"


def test_interaction_label_ligand_but_not_water():
    assert _edge_label([[0, 0, 0], [5, 0, 0]], ["C", "C"], ["C1", "CA"],
                       ["BEN", "ALA"], [1, 10]) == "ligand"
    # water is solvent, not a ligand -> proximity
    assert _edge_label([[0, 0, 0], [5, 0, 0]], ["O", "C"], ["O", "CA"],
                       ["HOH", "ALA"], [1, 10]) == "proximity"


def test_interaction_label_covalent_for_sequence_neighbours():
    assert _edge_label([[0, 0, 0], [5, 0, 0]], ["C", "C"], ["CA", "CA"],
                       ["ALA", "ALA"], [1, 2]) == "covalent"


def test_interaction_label_hydrophobic_and_polar_and_proximity():
    # two LEU side chains in contact, not sequence-adjacent
    assert _edge_label([[0, 0, 0], [1.0, 0, 0], [9, 0, 0], [3.0, 0, 0]],
                       ["C", "C", "C", "C"], ["CA", "CD1", "CA", "CD1"],
                       ["LEU", "LEU", "LEU", "LEU"], [1, 1, 8, 8]) == "hydrophobic"
    # two SER side chains in contact
    assert _edge_label([[0, 0, 0], [1.0, 0, 0], [9, 0, 0], [3.0, 0, 0]],
                       ["C", "O", "C", "O"], ["CA", "OG", "CA", "OG"],
                       ["SER", "SER", "SER", "SER"], [1, 1, 8, 8]) == "polar"
    # GLY (no side chain) + ALA, not adjacent -> proximity fallback
    assert _edge_label([[0, 0, 0], [4, 0, 0]], ["C", "C"], ["CA", "CA"],
                       ["GLY", "ALA"], [1, 8]) == "proximity"


def test_residue_contact_graph_annotation_integration_and_one_hot():
    from molscope.graph import RESIDUE_INTERACTION_LABELS

    mol = ms.read(os.path.join(DATA, "3ptb.pdb"))
    g = mol.to_residue_contact_graph(annotate_interactions=True)
    assert len(g.edge_interactions) == g.n_contacts
    assert set(g.edge_interactions) <= set(RESIDUE_INTERACTION_LABELS)
    # trypsin has six disulfide bridges
    assert g.edge_interactions.count("disulfide") == 6
    oh = g.interaction_one_hot()
    assert oh.shape == (g.n_contacts, len(RESIDUE_INTERACTION_LABELS))
    assert bool((oh.sum(axis=1) == 1).all())


def test_residue_contact_graph_unannotated_has_no_labels():
    g = ms.read(os.path.join(DATA, "3ptb.pdb")).to_residue_contact_graph()
    assert g.edge_interactions == []
    assert bool((g.interaction_one_hot() == 0).all())


def test_residue_contact_graph_networkx_carries_interaction():
    pytest.importorskip("networkx")
    from molscope.graph import RESIDUE_INTERACTION_LABELS

    g = ms.read(os.path.join(DATA, "3ptb.pdb")).to_residue_contact_graph(
        annotate_interactions=True
    )
    interactions = [d["interaction"] for *_, d in g.to_networkx().edges(data=True)]
    assert len(interactions) == g.n_contacts
    assert set(interactions) <= set(RESIDUE_INTERACTION_LABELS)
    # the plain (unannotated) graph omits the attribute
    plain = ms.read(os.path.join(DATA, "3ptb.pdb")).to_residue_contact_graph().to_networkx()
    assert all("interaction" not in d for *_, d in plain.edges(data=True))
