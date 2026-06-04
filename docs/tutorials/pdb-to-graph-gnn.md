# PDB to Graph/GNN

This tutorial starts with a PDB file, builds graph representations, and shows
how the same structures become a tiny PyTorch Geometric dataset for graph neural
network experiments.

You will build:

- an atom/bond graph from `1fqy.pdb`,
- a residue contact graph for protein-level topology,
- a small graph-level ML dataset from the `1aml.pdb` NMR ensemble.

## Build an atom graph

```python
import molscope as ms

mol = ms.read("examples/data/1fqy.pdb")
graph = mol.to_graph()

print(graph.n_atoms, "nodes")
print(graph.n_bonds, "edges")
print(graph.node_features("ml").shape)
print(graph.edge_features("ml").shape)
```

Expected output for the bundled Aquaporin-1 structure:

```text
1661 nodes
1693 edges
(1661, 19)
(1693, 3)
```

The atom graph is dependency-free. Nodes carry element, mass, charge, optional
metadata, and coordinates. Edges are explicit bonds when the file provides them
or distance-inferred bonds otherwise.

## Build a residue contact graph

For protein-scale workflows, residue graphs are often easier to learn from than
full atom graphs:

```python
residue_graph = mol.to_residue_contact_graph(
    cutoff=8.0,
    method="ca",
    min_seq_sep=4,
)

print(residue_graph.n_residues, "residue nodes")
print(residue_graph.n_contacts, "long-range contacts")
print(residue_graph.node_features("ml").shape)
print(residue_graph.edge_features("ml").shape)
```

`method="ca"` uses alpha-carbon distances, with a centroid fallback for residues
without a CA atom. `min_seq_sep=4` removes local backbone-neighbor contacts so
the edges emphasize folded tertiary structure.

## Export to graph libraries

Install only the backend you need:

```bash
pip install "molscope[graph]"  # NetworkX
pip install "molscope[pyg]"    # PyTorch Geometric
pip install "molscope[dgl]"    # DGL
```

Then export from the same `Molecule` or graph object:

```python
G = mol.to_networkx()
data = mol.to_pyg_data(node_preset="ml", edge_preset="ml")
residue_data = residue_graph.to_pyg_data(node_preset="ml", edge_preset="ml")

print(data.x.shape, data.edge_index.shape, data.edge_attr.shape)
print(data.pos.shape)
```

PyTorch Geometric receives `x`, `edge_index`, `edge_attr`, `pos`, and chemistry
metadata where available. The residue graph exporter uses residue node features
and contact edge features instead of atom-level bond features.

## Make a tiny graph-level dataset

The bundled `1aml.pdb` file is a 20-model NMR ensemble. Each model can become
one graph. Here the toy regression target is radius of gyration, and the toy
classification target is whether the conformer is more expanded than the median
model.

```python
import torch

models = ms.read_pdb_models("examples/data/1aml.pdb")
radii = torch.tensor([m.radius_of_gyration for m in models], dtype=torch.float)
median = radii.median()

graphs = []
for idx, model in enumerate(models):
    data = model.to_pyg_data(node_preset="ml", edge_preset="ml")

    centered_pos = data.pos - data.pos.mean(dim=0, keepdim=True)
    data.x = torch.cat([data.x, centered_pos], dim=1)

    data.y_reg = radii[idx].view(1)
    data.y_cls = (radii[idx] > median).float().view(1)
    data.model_id = idx + 1
    graphs.append(data)
```

Appending centered coordinates makes this geometry target visible to a compact
message-passing model. For real projects, replace these toy labels with
experimental values, simulation outputs, docking scores, annotations, or other
graph-level targets.

## Train a minimal GNN

```python
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


class GraphRegressorClassifier(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 64)
        self.reg_head = torch.nn.Linear(64, 1)
        self.cls_head = torch.nn.Linear(64, 1)

    def forward(self, batch):
        x = self.conv1(batch.x, batch.edge_index).relu()
        x = self.conv2(x, batch.edge_index).relu()
        pooled = global_mean_pool(x, batch.batch)
        return self.reg_head(pooled).squeeze(-1), self.cls_head(pooled).squeeze(-1)


loader = DataLoader(graphs, batch_size=4, shuffle=True)
model = GraphRegressorClassifier(graphs[0].x.size(1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for batch in loader:
    pred_reg, pred_cls = model(batch)
    loss_reg = torch.nn.functional.mse_loss(pred_reg, batch.y_reg.view(-1))
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_cls,
        batch.y_cls.view(-1),
    )
    loss = loss_reg + loss_cls

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

This is intentionally a tiny teaching model, not a scientific benchmark. The
loop above builds the graphs, attaches labels, and constructs the `DataLoader`
by hand so the moving parts are visible. Once you know them,
[`build_dataset`](../user-guide/molecular-graphs.md#building-a-dataset-in-one-call)
plus [`ds.loader()`](../user-guide/molecular-graphs.md#mini-batches-for-a-training-loop)
collapse all of that into two calls — the
[PDB to a trained GNN](../examples/pdb-to-pyg-ml.md) example walks through the
same task that way, and a runnable train/test version lives at
`examples/pdb_to_pyg_ml.py`:

```bash
uv pip install torch torch_geometric
.venv/bin/python examples/pdb_to_pyg_ml.py
```

Use `.venv/bin/python` directly in this repository because `uv run` may re-sync
the locked environment and remove optional packages that are not core MolScope
dependencies.

## Pick the graph type deliberately

| Graph | Nodes | Edges | Best for |
| --- | --- | --- | --- |
| Atom/bond graph | Atoms | Covalent bonds | Local chemistry, small molecules, atomistic message passing. |
| Residue contact graph | Residues | Spatial contacts | Fold topology, interfaces, protein-level tasks. |
| Coarse-grained graph | Beads | Bead bonds or inferred contacts | Reduced protein representations and fast prototypes. |

The construction choice is part of the model design. Keep it explicit in
experiments so results are comparable.
