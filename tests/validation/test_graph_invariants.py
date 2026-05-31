"""Validation for ML graph export and dataset assembly.

A graph handed to a GNN is only trustworthy if it faithfully encodes the
molecule it came from. These invariants assert exactly that: node count equals
atom count, the edge set equals the perceived bond set, the dense adjacency is
symmetric, and ``build_dataset`` keeps ids, labels and graphs aligned and
survives a save/load round trip. The raw-graph and dataset checks need no extra;
the NetworkX and PyG faithfulness checks skip when those backends are absent.
"""

from pathlib import Path

import numpy as np
import pytest

import molscope as ms

pytestmark = pytest.mark.validation

DATA = Path(__file__).resolve().parents[2] / "examples" / "data"
PROTEIN = str(DATA / "1ubq.pdb")
SECOND = str(DATA / "1fqy.pdb")

TOL = 1.2


def _edge_set(edges):
    return {frozenset(map(int, e)) for e in np.asarray(edges).reshape(-1, 2)}


# -- raw graph faithfully encodes the molecule ------------------------------

def test_raw_graph_nodes_and_edges_match_the_molecule():
    mol = ms.read(PROTEIN)
    graph = mol.to_graph(tolerance=TOL)

    assert graph.n_atoms == len(mol)                         # one node per atom
    assert list(graph.elements) == list(mol.elements)
    np.testing.assert_array_equal(graph.coords, mol.coords)
    # The edge set is exactly the perceived bond set (same i<j convention).
    assert _edge_set(graph.edges) == _edge_set(mol.bonds(tolerance=TOL))
    assert graph.n_bonds == len(mol.bonds(tolerance=TOL))


def test_raw_graph_adjacency_is_symmetric_and_bond_consistent():
    mol = ms.read(PROTEIN)
    graph = mol.to_graph(tolerance=TOL)
    adj = graph.adjacency_matrix()

    assert np.array_equal(adj, adj.T)                        # undirected
    assert np.all(np.diag(adj) == 0)                         # no self-loops
    assert int(adj.sum()) == 2 * graph.n_bonds              # each edge counted twice
    for i, j in graph.edges:
        assert adj[i, j] == 1 and adj[j, i] == 1


def test_explicit_bond_orders_survive_into_the_graph():
    # An SDF carries real bond orders; the graph must preserve them, not flatten
    # everything to 1.0 the way inferred bonds do.
    sdf = Path(__file__).resolve().parents[1] / "fixtures" / "docking_poses.sdf"
    mol = ms.read_sdf(str(sdf))
    graph = mol.to_graph()
    assert _edge_set(graph.edges) == _edge_set(mol.bonds())
    assert len(graph.edge_types) == graph.n_bonds


# -- build_dataset keeps ids, labels and graphs aligned ---------------------

def test_build_dataset_aligns_ids_labels_and_graphs():
    labels = {"1ubq": 1.5, "1fqy": -2.0}
    ds = ms.build_dataset([PROTEIN, SECOND], fmt="raw", labels=labels)

    assert ds.ids == ["1ubq", "1fqy"]                        # file stems, in order
    assert len(ds.graphs) == 2
    assert ds.labels == [1.5, -2.0]                          # joined by id, aligned
    # Each graph still has one node per atom of its source structure.
    for path, graph in zip([PROTEIN, SECOND], ds.graphs):
        assert graph.n_atoms == len(ms.read(path))


def test_build_dataset_raw_round_trips(tmp_path):
    ds = ms.build_dataset([PROTEIN, SECOND], fmt="raw", labels={"1ubq": 1.0})
    out = ds.save(str(tmp_path / "ds"))
    loaded = ms.dataset.GraphDataset.load(out)

    assert loaded.fmt == "raw"
    assert loaded.ids == ds.ids
    assert loaded.labels == ds.labels
    for before, after in zip(ds.graphs, loaded.graphs):
        assert after.n_atoms == before.n_atoms
        assert _edge_set(after.edges) == _edge_set(before.edges)


# -- backend exporters preserve the same node/edge structure ----------------

def test_networkx_export_preserves_nodes_and_edges():
    pytest.importorskip("networkx")
    mol = ms.read(PROTEIN)
    graph = mol.to_graph(tolerance=TOL)
    g = graph.to_networkx()

    assert g.number_of_nodes() == graph.n_atoms
    assert g.number_of_edges() == graph.n_bonds
    assert _edge_set(np.array(list(g.edges()))) == _edge_set(graph.edges)


def test_pyg_export_preserves_nodes_and_undirected_edges():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    mol = ms.read(PROTEIN)
    graph = mol.to_graph(tolerance=TOL)
    data = mol.to_pyg_data(tolerance=TOL)

    assert int(data.num_nodes) == graph.n_atoms
    ei = data.edge_index.numpy()
    # PyG stores directed edges; the unique undirected set must match the bonds.
    assert _edge_set(ei.T) == _edge_set(graph.edges)
    np.testing.assert_allclose(data.pos.numpy(), mol.coords, atol=1e-5)
