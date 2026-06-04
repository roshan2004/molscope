# PDB to a trained GNN (the dataset on-ramp)

This tutorial shows the full path from a set of structures to a trained graph
neural network, leaning on [`build_dataset`](../user-guide/molecular-graphs.md#building-a-dataset-in-one-call)
and [`GraphDataset.loader`](../user-guide/molecular-graphs.md#mini-batches-for-a-training-loop)
so the only code you write is the model and the training loop:

1. `build_dataset` reads every structure, featurises it to a PyG graph, joins a
   per-graph label, and splits train/val/test — in one call;
2. `ds.loader("train", ...)` hands back a batching `DataLoader` for the loop;
3. a compact GCN trains on it and reports a held-out metric.

The dataset is intentionally tiny: the bundled `examples/data/1aml.pdb` NMR
ensemble has 20 conformers, each one graph. The regression target is radius of
gyration. The goal is to show the workflow clearly, not to claim a
scientifically meaningful predictor.

## Install the optional ML stack

Install PyTorch and PyTorch Geometric for your platform first:

```bash
uv pip install torch torch_geometric
.venv/bin/python examples/pdb_to_pyg_ml.py
```

Use `.venv/bin/python` directly in this repo because `uv run` may re-sync the
locked environment and remove optional packages that are not core MolScope
dependencies.

## Build the dataset in one call

Each NMR conformer is read with a unique name (`1aml#1`, `1aml#2`, ...), so a
plain `{name: value}` dict joins as the per-graph label. `build_dataset` does the
reading, featurising, label join, and split together:

```python
import molscope as ms

models = ms.read_pdb_models("examples/data/1aml.pdb")    # 20 conformers
labels = {m.name: m.radius_of_gyration for m in models}  # graph-level target

ds = ms.build_dataset(
    models,                  # a list of Molecules — or a glob like "data/*.pdb"
    fmt="pyg",
    node_features="ml",      # element one-hots, atomic number, mass, ...
    labels=labels,           # joined to each graph, attached as data.y
    split=(0.70, 0.15, 0.15),
    seed=7,
)
print(ds.summary())
```

In real work, swap the in-memory `models`/`labels` for a folder of files and a
CSV — `build_dataset("data/*.pdb", labels="labels.csv", ...)` — and nothing else
changes.

### Fold coordinates in for a geometric target

Radius of gyration is a *geometric* property, but the node-feature presets are
composition-only: every conformer has the same atoms, so without geometry the
model cannot tell them apart. `build_dataset` attaches each atom's coordinates as
`data.pos`, so fold the centred coordinates into the node features:

```python
import torch

for data in ds.graphs:                                   # views share these objects,
    centred = data.pos - data.pos.mean(dim=0, keepdim=True)
    data.x = torch.cat([data.x, centred], dim=1)         # so train/val/test update too
```

### Standardise the target on the train split only

Fit the mean/std on **train** and apply them everywhere, so val/test never leak
into the normalisation:

```python
train_y = torch.cat([g.y for g in ds.train]).float()
mean, std = train_y.mean(), train_y.std().clamp_min(1e-6)
for data in ds.graphs:
    data.y = (data.y.float() - mean) / std               # map predictions back with * std + mean
```

## Train a GCN on the loader

`ds.loader(split)` is the only bridge you need to a standard PyG training loop —
the train split shuffles each epoch, val/test do not:

```python
from torch_geometric.nn import GCNConv, global_mean_pool

train_loader = ds.loader("train", batch_size=4)
val_loader   = ds.loader("val", batch_size=8)
test_loader  = ds.loader("test", batch_size=8)


class GCNRegressor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.head = torch.nn.Linear(64, 1)

    def forward(self, batch):
        x = self.conv1(batch.x, batch.edge_index).relu()
        x = self.conv2(x, batch.edge_index).relu()
        x = self.conv3(x, batch.edge_index).relu()
        return self.head(global_mean_pool(x, batch.batch)).squeeze(-1)


model = GCNRegressor(ds.graphs[0].x.size(1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

for epoch in range(1, 121):
    model.train()
    for batch in train_loader:
        loss = torch.nn.functional.mse_loss(model(batch), batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Evaluate by mapping predictions back to angstroms with `* std + mean`:

```python
model.eval()
with torch.no_grad():
    batch = next(iter(test_loader))
    pred = model(batch) * std + mean
    true = batch.y.view(-1) * std + mean
    print(f"test MAE: {(pred - true).abs().mean():.3f} A")
```

## Run the complete script

The runnable version lives at `examples/pdb_to_pyg_ml.py`:

```bash
.venv/bin/python examples/pdb_to_pyg_ml.py
```

It prints per-epoch training loss with a validation MAE and a final held-out
radius-of-gyration MAE. The exact numbers are not the point; the pipeline is:

```text
structures -> build_dataset(...) -> ds.loader("train") -> trained GCN -> test MAE
```

For real work, replace the toy label with experimental values, simulation
outputs, docking scores, functional classes, or other graph-level targets, and
point `build_dataset` at your own folder of structures plus a labels CSV.
