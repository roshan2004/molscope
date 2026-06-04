"""One-call assembly of an ML graph dataset from structure files.

:func:`build_dataset` collapses the usual boilerplate (discover files, read each,
featurise to a graph, attach positional encodings, join labels, split) into a
single call. It is a thin orchestration layer over the existing public API
(:func:`molscope.read`, :meth:`Molecule.to_graph` and the graph exporters), so it
adds no new core dependency: ``fmt="pyg"``/``"dgl"`` need the matching opt-in
extra, and ``fmt="raw"``/``"networkx"`` run on the core install.

    import molscope as ms

    ds = ms.build_dataset("data/*.pdb", fmt="pyg", pe="laplacian",
                          labels="labels.csv", split=(0.8, 0.1, 0.1))
    ds.train, ds.val, ds.test       # framework graph objects, split
    ds.loader("train", batch_size=32)   # batching DataLoader for a train loop
    print(ds.summary())

It deliberately stops at the framework's ``DataLoader``: :meth:`GraphDataset.loader`
hands back a ready-to-iterate PyG/DGL loader, but there is no training loop, no
model code, and no new file format.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional

from .io import fetch_file, read
from .molecule import Molecule
from .prepare import random_split

DATASET_FORMATS = ("pyg", "dgl", "networkx", "raw")


@dataclass
class TargetScaler:
    """An affine target standardiser fit on a dataset's train split.

    ``transform`` maps physical-unit targets into standardised space and
    ``inverse_transform`` maps model outputs back. It uses only arithmetic
    operators, so it works unchanged on Python floats, NumPy arrays, and
    framework tensors. Produced by :meth:`GraphDataset.standardize_targets`.
    """

    mean: float
    std: float

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse_transform(self, values):
        return values * self.std + self.mean


@dataclass
class GraphDataset:
    """The result of :func:`build_dataset`.

    ``graphs`` holds one featurised object per successfully read structure (a
    ``torch_geometric.data.Data``, ``dgl.DGLGraph``, ``networkx.Graph``, or
    :class:`~molscope.graph.MolecularGraph` depending on ``fmt``), aligned with
    ``ids`` (the file stems) and, when provided, ``labels``.
    """

    graphs: list
    ids: list[str]
    fmt: str
    labels: Optional[list] = None
    split: Optional[object] = None  # prepare.SplitResult, or None
    skipped: list[tuple[str, str]] = field(default_factory=list)
    feature_names: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.graphs)

    def _subset(self, indices):
        return [self.graphs[i] for i in indices]

    @property
    def train(self) -> Optional[list]:
        return None if self.split is None else self._subset(self.split.train)

    @property
    def val(self) -> Optional[list]:
        return None if self.split is None else self._subset(self.split.val)

    @property
    def test(self) -> Optional[list]:
        return None if self.split is None else self._subset(self.split.test)

    def loader(
        self,
        split: Optional[str] = None,
        *,
        batch_size: int = 1,
        shuffle: Optional[bool] = None,
        **kwargs,
    ):
        """Return a framework ``DataLoader`` that batches the graphs.

        The last mile between a built dataset and a training loop: PyG graphs
        are wrapped in a :class:`torch_geometric.loader.DataLoader` and DGL
        graphs in a :class:`dgl.dataloading.GraphDataLoader`, both of which
        collate the per-graph objects into mini-batches the framework's models
        consume directly.

        Args:
            split: which subset to load, one of ``"train"``, ``"val"``,
                ``"test"``, or ``None`` for the whole dataset. A named split
                requires the dataset to have been built with ``split=``.
            batch_size: graphs per mini-batch.
            shuffle: whether to reshuffle each epoch. Defaults to ``True`` for
                the train split and ``False`` otherwise.
            **kwargs: forwarded to the underlying loader (e.g. ``num_workers``,
                ``drop_last``).

        Returns:
            A ``torch_geometric.loader.DataLoader`` (``fmt="pyg"``) or
            ``dgl.dataloading.GraphDataLoader`` (``fmt="dgl"``).

        Raises:
            ValueError: for ``fmt="networkx"``/``"raw"`` (no batching loader),
                an unknown ``split`` name, or a named split when none was built.
        """
        if self.fmt == "pyg":
            from torch_geometric.loader import DataLoader as _Loader
        elif self.fmt == "dgl":
            from dgl.dataloading import GraphDataLoader as _Loader
        else:
            raise ValueError(
                f"loader() needs fmt='pyg' or 'dgl', got {self.fmt!r}; "
                "networkx/raw graphs have no batching DataLoader"
            )

        graphs = self._loader_subset(split)
        if shuffle is None:
            shuffle = split == "train"
        return _Loader(graphs, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def _loader_subset(self, split):
        """The graph list for ``split``, validating the request."""
        if split is None:
            return self.graphs
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"unknown split {split!r}; use 'train', 'val', 'test', or None"
            )
        if self.split is None:
            raise ValueError(
                f"no split available; build_dataset(..., split=(...)) is required "
                f"to request the {split!r} loader"
            )
        return getattr(self, split)

    def standardize_targets(self) -> TargetScaler:
        """Standardise graph targets in place using train-split statistics only.

        Fits a mean and standard deviation on the **train** split's labels and
        rewrites every labelled graph's ``data.y`` into standardised space, so
        validation and test targets never leak into the normalisation -- the
        small correctness detail that is easy to get wrong by fitting on the
        whole set. ``self.labels`` keeps the original physical-unit values.

        Returns a :class:`TargetScaler`; map model outputs back with
        ``scaler.inverse_transform(pred)``.

        Requires ``fmt="pyg"`` (where labels are attached as ``data.y``), a built
        split, and at least one labelled train graph.
        """
        import numpy as np

        if self.fmt != "pyg":
            raise ValueError(
                "standardize_targets supports fmt='pyg' (labels attached as "
                f"data.y), got {self.fmt!r}"
            )
        if self.split is None:
            raise ValueError(
                "standardize_targets needs a split; build with "
                "build_dataset(..., split=(...))"
            )
        if self.labels is None:
            raise ValueError("standardize_targets needs labels; none were provided")

        train_values = [
            float(self.labels[i])
            for i in self.split.train
            if self.labels[i] is not None
        ]
        if not train_values:
            raise ValueError("no labelled graphs in the train split to fit on")

        mean = float(np.mean(train_values))
        std = max(float(np.std(train_values)), 1e-8)
        scaler = TargetScaler(mean=mean, std=std)

        for graph in self.graphs:
            y = getattr(graph, "y", None)
            if y is not None:
                graph.y = scaler.transform(y)
        return scaler

    def summary(self) -> str:
        """One-block human-readable description of the dataset."""
        lines = [
            f"GraphDataset: {len(self.graphs)} graph(s), fmt={self.fmt!r}",
        ]
        if self.feature_names:
            for key, names in self.feature_names.items():
                lines.append(f"  {key}: {len(names)} ({', '.join(names)})")
        if self.labels is not None:
            covered = sum(1 for v in self.labels if v is not None)
            lines.append(f"  labels: {covered}/{len(self.labels)} graphs labelled")
        if self.split is not None:
            sizes = ", ".join(f"{k}={v}" for k, v in self.split.sizes.items())
            lines.append(f"  split: {sizes}")
        if self.skipped:
            lines.append(f"  skipped: {len(self.skipped)} source(s)")
            for src, err in self.skipped:
                lines.append(f"    {src}: {err}")
        return "\n".join(lines)

    def save(self, out_dir: str) -> str:
        """Write each graph plus a ``manifest.json`` to ``out_dir``.

        File types: ``.pt`` (pyg, via ``torch.save``), ``.bin`` (dgl),
        ``.json`` (networkx node-link), ``.pkl`` (raw). Returns ``out_dir``.
        """
        import json

        os.makedirs(out_dir, exist_ok=True)
        files = []
        for gid, graph in zip(self.ids, self.graphs):
            files.append(_save_one(graph, self.fmt, out_dir, gid))

        manifest = {
            "fmt": self.fmt,
            "ids": self.ids,
            "files": files,
            "feature_names": self.feature_names,
            "labels": self.labels,
            "skipped": self.skipped,
        }
        if self.split is not None:
            manifest["split"] = {
                "method": self.split.method,
                "train": self.split.train,
                "val": self.split.val,
                "test": self.split.test,
            }
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        return out_dir

    @classmethod
    def load(cls, out_dir: str) -> GraphDataset:
        """Load a saved dataset from ``out_dir``.

        Loads the manifest.json and all referenced graph files, reconstructing
        the original :class:`GraphDataset` object.
        """
        import json

        manifest_path = os.path.join(out_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"manifest.json not found in {out_dir}")

        with open(manifest_path) as fh:
            manifest = json.load(fh)

        fmt = manifest["fmt"]
        ids = manifest["ids"]
        files = manifest["files"]
        labels = manifest.get("labels")
        skipped = [tuple(item) for item in manifest.get("skipped", [])]
        feature_names = manifest.get("feature_names", {})

        graphs = [_load_one(fmt, os.path.join(out_dir, fname)) for fname in files]

        split_result = None
        if "split" in manifest:
            from .prepare import SplitResult

            split_data = manifest["split"]
            split_result = SplitResult(
                method=split_data["method"],
                train=split_data["train"],
                val=split_data["val"],
                test=split_data["test"],
            )

        return cls(
            graphs=graphs,
            ids=ids,
            fmt=fmt,
            labels=labels,
            split=split_result,
            skipped=skipped,
            feature_names=feature_names,
        )


def build_dataset(
    source,
    *,
    fmt: str = "pyg",
    node_features: str = "default",
    edge_features: str = "default",
    pe: Optional[str] = None,
    pe_k: int = 8,
    self_loops: bool = False,
    global_node: bool = False,
    infer_orders: bool = False,
    labels=None,
    id_col: Optional[str] = None,
    label_col: Optional[str] = None,
    split=None,
    seed: int = 0,
    n_jobs: int = 1,
    on_error: str = "skip",
    cache_dir: Optional[str] = None,
) -> GraphDataset:
    """Build an ML graph dataset from structure files in one call.

    Args:
        source: a glob string, a list of paths, or a list of
            :class:`~molscope.molecule.Molecule` objects.
        fmt: output graph type, one of ``"pyg"``, ``"dgl"``, ``"networkx"``,
            ``"raw"`` (a :class:`~molscope.graph.MolecularGraph`).
        node_features, edge_features: feature presets passed to the exporter.
        pe: optional positional encoding (``"laplacian"`` or ``"random_walk"``)
            with dimension ``pe_k`` (pyg/dgl only).
        self_loops, global_node, infer_orders: passed through to the exporter.
        labels: optional ``{id: value}`` dict or a CSV path. CSV is keyed by
            ``id_col`` (default: first column) against the file stem, with the
            target in ``label_col`` (default: second column). For ``fmt="pyg"``
            the value is also attached as ``data.y``.
        split: optional ``(train, val, test)`` fractions for a random split.
        seed: random seed for the split.
        n_jobs: worker processes for featurisation (``fmt="dgl"`` runs serially).
        on_error: ``"skip"`` (record and continue) or ``"raise"``.
        cache_dir: optional directory for an on-disk featurisation cache. Each
            file-based structure is cached under a key derived from its *content*
            and the featurisation options (``fmt``, the feature presets, ``pe``,
            ``self_loops``, ...), so a second call reuses the stored graphs and
            re-featurises only inputs that are new or whose content or options
            changed. ``labels`` and ``split`` are applied after loading and are
            not part of the key, so re-labelling or re-splitting is free.
            In-memory ``Molecule`` sources are not cached (they have no stable
            on-disk identity). The directory is created if missing.

    Returns:
        A :class:`GraphDataset` with ``.graphs``/``.ids``/``.labels``, the split
        (if requested), and ``.summary()`` / ``.save()``.
    """
    if fmt not in DATASET_FORMATS:
        raise ValueError(f"unknown fmt {fmt!r}; use one of: {', '.join(DATASET_FORMATS)}")
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")
    if pe is not None and fmt in ("networkx", "raw"):
        raise ValueError(f"positional encodings are not supported for fmt={fmt!r}")

    items = _normalise_source(source)
    opts = {
        "fmt": fmt,
        "node_features": node_features,
        "edge_features": edge_features,
        "pe": pe,
        "pe_k": pe_k,
        "self_loops": self_loops,
        "global_node": global_node,
        "infer_orders": infer_orders,
    }

    # dgl graphs do not reliably pickle across processes; keep them serial.
    use_jobs = 1 if fmt == "dgl" else max(1, int(n_jobs))

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    results = _featurise_all(items, opts, use_jobs, cache_dir)

    graphs, ids, skipped = [], [], []
    for label, outcome in results:
        if isinstance(outcome, Exception):
            if on_error == "raise":
                raise outcome
            skipped.append((label, f"{type(outcome).__name__}: {outcome}"))
        else:
            ids.append(label)
            graphs.append(outcome)

    label_list = _resolve_labels(labels, ids, id_col, label_col)
    if label_list is not None and fmt == "pyg":
        _attach_pyg_labels(graphs, label_list)

    split_result = None
    if split is not None:
        train, val, test = _validate_split(split)
        split_result = random_split(len(graphs), test=test, val=val, seed=seed)

    return GraphDataset(
        graphs=graphs,
        ids=ids,
        fmt=fmt,
        labels=label_list,
        split=split_result,
        skipped=skipped,
        feature_names=_feature_names(node_features, edge_features),
    )


def fetch_dataset(
    ids,
    *,
    labels=None,
    fmt: str = "pyg",
    structure_fmt: str = "pdb",
    root: Optional[str] = None,
    on_error: str = "skip",
    **build_kwargs,
) -> GraphDataset:
    """Build a dataset from RCSB accessions, downloading each structure first.

    The adapter for turning a published *accession + label* table (an enzyme
    set, a fold-classification list, a stability benchmark, ...) into a trainable
    dataset: it downloads each PDB id from the RCSB — cached under ``root`` so a
    rerun does not re-download — and hands the files to :func:`build_dataset`,
    which featurises, joins labels, and splits them.

    Args:
        ids: an iterable of RCSB PDB ids (case-insensitive).
        labels: optional ``{id: value}`` dict (case-insensitive) or a CSV path
            understood by :func:`build_dataset`. CSV ids must match the
            *lowercased* accession (the downloaded file stem).
        fmt: output graph format (see :func:`build_dataset`).
        structure_fmt: ``"pdb"`` or ``"cif"``, the download format.
        root: directory for the downloaded structures (default: the shared
            ``molscope_cache`` temp dir). Reused across runs.
        on_error: ``"skip"`` (record a failed download/featurisation and
            continue) or ``"raise"``.
        **build_kwargs: forwarded to :func:`build_dataset` — ``node_features``,
            ``edge_features``, ``split``, ``seed``, ``pe``, ``n_jobs``, and
            ``cache_dir`` (the *featurisation* cache, separate from ``root``).

    Returns:
        A :class:`GraphDataset`; accessions whose download failed appear in
        ``ds.skipped`` alongside any that later failed to featurise.
    """
    if on_error not in ("skip", "raise"):
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")

    paths, fetch_skipped = [], []
    for pdb_id in ids:
        try:
            paths.append(fetch_file(str(pdb_id), fmt=structure_fmt, cache_dir=root))
        except Exception as exc:
            if on_error == "raise":
                raise
            fetch_skipped.append((str(pdb_id), f"{type(exc).__name__}: {exc}"))

    if isinstance(labels, dict):
        labels = {str(k).lower(): v for k, v in labels.items()}

    if not paths:
        node_preset = build_kwargs.get("node_features", "default")
        edge_preset = build_kwargs.get("edge_features", "default")
        return GraphDataset(
            graphs=[],
            ids=[],
            fmt=fmt,
            skipped=fetch_skipped,
            feature_names=_feature_names(node_preset, edge_preset),
        )

    ds = build_dataset(paths, fmt=fmt, labels=labels, on_error=on_error, **build_kwargs)
    # Surface download failures next to any featurisation failures.
    ds.skipped = fetch_skipped + ds.skipped
    return ds


# --- internals ---------------------------------------------------------------


def _expand_dirs(paths: list[str]) -> list[str]:
    expanded = []
    supported_exts = {".pdb", ".cif", ".xyz", ".sdf", ".pdb.gz", ".cif.gz", ".xyz.gz", ".sdf.gz"}
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    ext = ""
                    if f.lower().endswith(".gz"):
                        parts = f.lower().split(".")
                        if len(parts) >= 3:
                            ext = "." + parts[-2] + ".gz"
                    else:
                        ext = os.path.splitext(f)[1].lower()

                    if ext in supported_exts:
                        expanded.append(os.path.join(root, f))
        else:
            expanded.append(p)
    return sorted(set(expanded))


def _normalise_source(source):
    """Return a list of (label, path-or-Molecule) work items."""
    if isinstance(source, Molecule):
        source = [source]

    if isinstance(source, (str, os.PathLike)):
        paths = glob.glob(os.fspath(source), recursive=True)
        paths = _expand_dirs(paths)
        if not paths:
            raise ValueError(f"no files matched {source!r}")
        return [(_stem(p), p) for p in paths]

    items = []
    for i, item in enumerate(source):
        if isinstance(item, Molecule):
            items.append((item.name or f"mol_{i}", item))
        else:
            path_str = os.fspath(item)
            expanded = _expand_dirs([path_str])
            for p in expanded:
                items.append((_stem(p), p))
    return items


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _featurise_all(items, opts, n_jobs, cache_dir=None):
    if n_jobs <= 1:
        return [(label, _featurise(item, opts, cache_dir)) for label, item in items]
    from functools import partial
    from multiprocessing import Pool

    worker = partial(_featurise, opts=opts, cache_dir=cache_dir)
    with Pool(n_jobs) as pool:
        outcomes = pool.map(worker, [item for _, item in items])
    return [(label, outcome) for (label, _), outcome in zip(items, outcomes)]


def _featurise(item, opts, cache_dir=None):
    """Featurise one item, going through the on-disk cache when enabled.

    Returns the graph object, or the Exception on failure (so it is safe as a
    multiprocessing worker, and the caller decides whether to skip or raise).
    In-memory ``Molecule`` items bypass the cache (no stable on-disk identity).
    """
    try:
        if cache_dir is None or isinstance(item, Molecule):
            return _featurise_one(item, opts)

        key = _cache_key(item, opts)
        cached = _cache_load(cache_dir, key, opts["fmt"])
        if cached is not None:
            return cached
        graph = _featurise_one(item, opts)
        _cache_save(cache_dir, key, opts["fmt"], graph)
        return graph
    except Exception as exc:
        return exc


def _featurise_one(item, opts):
    """Read (if needed) and convert one item to the requested graph format."""
    mol = item if isinstance(item, Molecule) else read(item)
    graph = mol.to_graph(infer_orders=opts["infer_orders"])
    fmt = opts["fmt"]
    if fmt == "raw":
        return graph
    if fmt == "networkx":
        return graph.to_networkx()
    exporter = graph.to_pyg_data if fmt == "pyg" else graph.to_dgl_graph
    return exporter(
        node_preset=opts["node_features"],
        edge_preset=opts["edge_features"],
        include_self_loops=opts["self_loops"],
        include_global_node=opts["global_node"],
        include_pe=opts["pe"],
        pe_k=opts["pe_k"],
    )


# --- on-disk featurisation cache --------------------------------------------

_CACHE_EXT = {"pyg": ".pt", "dgl": ".bin", "networkx": ".json", "raw": ".pkl"}


def _cache_key(path, opts) -> str:
    """A content-and-options digest identifying one cached featurisation."""
    h = hashlib.sha1()
    h.update(repr(sorted(opts.items())).encode())
    h.update(b"\0")
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_load(cache_dir, key, fmt):
    """Return the cached graph for ``key``, or ``None`` on a miss.

    A corrupt or partially-written cache entry is treated as a miss so the
    caller simply re-featurises rather than failing.
    """
    path = os.path.join(cache_dir, f"{key}{_CACHE_EXT[fmt]}")
    if not os.path.exists(path):
        return None
    try:
        return _load_one(fmt, path)
    except Exception:
        return None


def _cache_save(cache_dir, key, fmt, graph):
    _save_one(graph, fmt, cache_dir, key)


def _resolve_labels(labels, ids, id_col, label_col):
    if labels is None:
        return None
    if isinstance(labels, dict):
        mapping = labels
    else:
        mapping = _read_label_csv(os.fspath(labels), id_col, label_col)
    return [mapping.get(gid) for gid in ids]


def _read_label_csv(path, id_col, label_col):
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return {}
        try:
            id_idx = header.index(id_col) if id_col else 0
        except ValueError as exc:
            raise ValueError(f"id_col={id_col!r} not found in CSV header: {header}") from exc
        try:
            label_idx = header.index(label_col) if label_col else 1
        except ValueError as exc:
            raise ValueError(f"label_col={label_col!r} not found in CSV header: {header}") from exc
        mapping = {}
        for row in reader:
            if len(row) <= max(id_idx, label_idx):
                continue
            mapping[row[id_idx]] = _coerce(row[label_idx])
    return mapping


def _coerce(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def _attach_pyg_labels(graphs, label_list):
    import torch

    for graph, value in zip(graphs, label_list):
        if value is not None:
            graph.y = torch.tensor([value], dtype=torch.float)


def _validate_split(split):
    try:
        train, val, test = (float(x) for x in split)
    except (TypeError, ValueError) as exc:
        raise ValueError("split must be a (train, val, test) tuple of fractions") from exc
    if abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {train + val + test}")
    return train, val, test


def _feature_names(node_preset, edge_preset):
    from .graph import edge_feature_names, node_feature_names

    return {
        "node_features": list(node_feature_names(node_preset)),
        "edge_features": list(edge_feature_names(edge_preset)),
    }


def _save_one(graph, fmt, out_dir, gid):
    if fmt == "pyg":
        import torch

        out = os.path.join(out_dir, f"{gid}.pt")
        torch.save(graph, out)
    elif fmt == "dgl":
        from dgl.data.utils import save_graphs

        out = os.path.join(out_dir, f"{gid}.bin")
        save_graphs(out, [graph])
    elif fmt == "networkx":
        import json

        import networkx as nx

        out = os.path.join(out_dir, f"{gid}.json")
        with open(out, "w") as fh:
            json.dump(nx.node_link_data(graph), fh)
    else:  # raw
        import pickle

        out = os.path.join(out_dir, f"{gid}.pkl")
        with open(out, "wb") as fh:
            pickle.dump(graph, fh)
    return os.path.basename(out)


def _load_one(fmt, fpath):
    """Load a single graph written by :func:`_save_one`."""
    if fmt == "pyg":
        import torch

        # PyG ``Data`` objects are not plain tensors, so the safe unpickler that
        # ``torch.load`` defaults to since PyTorch 2.6 (weights_only=True)
        # refuses them. These files are written by our own ``_save_one``, so
        # loading them fully is safe.
        return torch.load(fpath, weights_only=False)
    if fmt == "dgl":
        from dgl.data.utils import load_graphs

        return load_graphs(fpath)[0][0]
    if fmt == "networkx":
        import json

        import networkx as nx

        with open(fpath) as fh:
            return nx.node_link_graph(json.load(fh))
    # raw
    import pickle

    with open(fpath, "rb") as fh:
        return pickle.load(fh)
