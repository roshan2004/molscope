"""End-to-end GNN on-ramp: structures -> build_dataset -> loader -> trained GCN.

This is the canonical "folder of structures to a trained graph neural network"
walkthrough. It leans on :func:`molscope.build_dataset` and
:meth:`GraphDataset.loader` to do the heavy lifting, so the only code left is the
model and the training loop:

* ``build_dataset`` reads every structure, featurises it to a PyG graph, joins a
  per-graph label, and splits into train/val/test in one call;
* ``ds.loader("train", ...)`` hands back a batching ``DataLoader`` ready for the
  loop.

The dataset is intentionally tiny: the bundled ``examples/data/1aml.pdb`` NMR
ensemble has 20 conformers, each one graph. The regression target is radius of
gyration. The goal is to show the workflow clearly, not to claim a
scientifically meaningful predictor.

Because radius of gyration is a *geometric* property and MolScope's node-feature
presets are composition-only (they cannot see how a conformer is folded), we
fold each atom's centred coordinates into its feature vector before training --
otherwise every conformer would look identical to the model. ``build_dataset``
attaches those coordinates as ``data.pos`` for us.

Install the optional ML stack first:

    uv pip install torch torch_geometric
    .venv/bin/python examples/pdb_to_pyg_ml.py

Use ``.venv/bin/python`` directly because ``uv run`` may re-sync the locked
environment and remove optional packages that are not core MolScope deps.
"""

from __future__ import annotations

from pathlib import Path

import molscope as ms

DATA = Path(__file__).resolve().parent / "data"
ENSEMBLE = DATA / "1aml.pdb"


def _require_pyg():
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, global_mean_pool
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit(
            "Install the optional ML stack first:\n"
            "  uv pip install torch torch_geometric\n"
            "  .venv/bin/python examples/pdb_to_pyg_ml.py"
        ) from exc
    return torch, F, GCNConv, global_mean_pool


def make_dataset(seed: int = 7):
    """The whole data pipeline, in one ``build_dataset`` call.

    Each NMR conformer is read with a unique name (``1aml#1`` ...), so a plain
    ``{name: radius_of_gyration}`` dict joins as the per-graph label. The split
    is a deterministic 70/15/15.
    """
    models = ms.read_pdb_models(ENSEMBLE)
    labels = {m.name: m.radius_of_gyration for m in models}
    return ms.build_dataset(
        models,                       # a list of Molecules (or "data/*.pdb")
        fmt="pyg",
        node_features="ml",           # element one-hots, atomic number, mass, ...
        labels=labels,                # joined to each graph by name, attached as data.y
        split=(0.70, 0.15, 0.15),
        seed=seed,
    )


def fold_coordinates_into_features(torch, ds):
    """Append each atom's centred xyz to its node features (geometric target).

    Mutates the graphs in place; ``ds.train``/``val``/``test`` are views over the
    same objects, so they pick the change up too.
    """
    for data in ds.graphs:
        centred = data.pos - data.pos.mean(dim=0, keepdim=True)
        data.x = torch.cat([data.x, centred], dim=1)


def standardise_targets(torch, ds):
    """Standardise ``data.y`` using train-split statistics only.

    Fitting the mean/std on train and applying them everywhere keeps val/test out
    of the normalisation -- the small correctness detail that is easy to skip.
    Returns ``(mean, std)`` so predictions can be mapped back to angstroms.
    """
    train_y = torch.cat([g.y for g in ds.train]).float()
    mean = train_y.mean()
    std = train_y.std().clamp_min(1e-6)
    for data in ds.graphs:
        data.y = (data.y.float() - mean) / std
    return mean, std


def make_model(torch, GCNConv, global_mean_pool, in_channels: int):
    class GCNRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_channels, 64)
            self.conv2 = GCNConv(64, 64)
            self.conv3 = GCNConv(64, 64)
            self.head = torch.nn.Linear(64, 1)

        def forward(self, batch):
            x = self.conv1(batch.x, batch.edge_index).relu()
            x = self.conv2(x, batch.edge_index).relu()
            x = self.conv3(x, batch.edge_index).relu()
            pooled = global_mean_pool(x, batch.batch)
            return self.head(pooled).squeeze(-1)

    return GCNRegressor()


def _mae_angstroms(torch, model, loader, mean, std):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch) * std + mean
            true = batch.y.view(-1) * std + mean
            errors.append((pred - true).abs())
    return float(torch.cat(errors).mean())


def run(epochs: int = 120, seed: int = 7) -> dict:
    """Train the GCN and return a small results dict (used by the test too)."""
    torch, F, GCNConv, global_mean_pool = _require_pyg()
    torch.manual_seed(seed)

    ds = make_dataset(seed=seed)
    fold_coordinates_into_features(torch, ds)
    mean, std = standardise_targets(torch, ds)

    train_loader = ds.loader("train", batch_size=4)   # shuffles each epoch
    val_loader = ds.loader("val", batch_size=8)
    test_loader = ds.loader("test", batch_size=8)

    model = make_model(torch, GCNConv, global_mean_pool, ds.graphs[0].x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            loss = F.mse_loss(model(batch), batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * batch.num_graphs
        train_loss = total / len(ds.train)
        history.append(train_loss)
        if epoch == 1 or epoch % 40 == 0:
            val_mae = _mae_angstroms(torch, model, val_loader, mean, std)
            print(f"epoch {epoch:3d}  train_loss={train_loss:.3f}  val_MAE={val_mae:.3f} A")

    test_mae = _mae_angstroms(torch, model, test_loader, mean, std)
    return {
        "test_mae": test_mae,
        "first_loss": history[0],
        "last_loss": history[-1],
        "n_train": len(ds.train),
        "n_val": len(ds.val),
        "n_test": len(ds.test),
        "in_channels": ds.graphs[0].x.size(1),
    }


def main():
    result = run()
    print("\nHoldout metric")
    print(f"  radius-of-gyration test MAE: {result['test_mae']:.3f} A")
    print(
        f"  (split: {result['n_train']} train / {result['n_val']} val / "
        f"{result['n_test']} test graphs, {result['in_channels']} node features)"
    )


if __name__ == "__main__":
    main()
