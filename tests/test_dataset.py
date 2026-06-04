import os

import pytest

import molscope as ms
from molscope import build_dataset

DATA = "examples/data"
PDBS = [f"{DATA}/1fqy.pdb", f"{DATA}/3ptb.pdb", f"{DATA}/1aml.pdb"]


def test_build_raw_from_glob():
    ds = build_dataset(f"{DATA}/*.pdb", fmt="raw")
    assert len(ds) >= 3
    assert all(hasattr(g, "n_atoms") for g in ds.graphs)  # MolecularGraph
    assert ds.split is None and ds.train is None
    assert ds.feature_names["node_features"] == ["atomic_number", "mass"]


def test_build_from_molecule_list():
    mols = [ms.read(p) for p in PDBS]
    ds = build_dataset(mols, fmt="raw")
    assert len(ds) == 3
    # ids fall back to the molecule name
    assert ds.ids[0] == mols[0].name


def test_random_split_sizes():
    ds = build_dataset(PDBS, fmt="raw", split=(0.34, 0.33, 0.33), seed=0)
    assert len(ds.train) + len(ds.val) + len(ds.test) == len(ds)
    assert ds.split.sizes == {
        "train": len(ds.train),
        "validation": len(ds.val),
        "test": len(ds.test),
    }


def test_split_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        build_dataset(PDBS, fmt="raw", split=(0.5, 0.3, 0.3))


def test_labels_from_dict_align_with_ids():
    labels = {"1fqy": 1.0, "3ptb": 0.0}  # 1aml deliberately unlabelled
    ds = build_dataset(PDBS, fmt="raw", labels=labels)
    by_id = dict(zip(ds.ids, ds.labels))
    assert by_id["1fqy"] == 1.0
    assert by_id["3ptb"] == 0.0
    assert by_id["1aml"] is None


def test_labels_from_csv(tmp_path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("id,target\n1fqy,2.5\n3ptb,7.0\n")
    ds = build_dataset(PDBS, fmt="raw", labels=str(csv_path))
    by_id = dict(zip(ds.ids, ds.labels))
    assert by_id["1fqy"] == 2.5
    assert by_id["3ptb"] == 7.0
    assert by_id["1aml"] is None


def test_skip_unreadable_source():
    ds = build_dataset(
        [f"{DATA}/1fqy.pdb", "does_not_exist.pdb"], fmt="raw", on_error="skip"
    )
    assert ds.ids == ["1fqy"]
    assert len(ds.skipped) == 1
    assert "does_not_exist" in ds.skipped[0][0]


def test_on_error_raise():
    with pytest.raises((OSError, ValueError)):
        build_dataset(["nope.pdb"], fmt="raw", on_error="raise")


def test_njobs_parallel_matches_serial():
    serial = build_dataset(PDBS, fmt="raw", n_jobs=1)
    parallel = build_dataset(PDBS, fmt="raw", n_jobs=2)
    assert serial.ids == parallel.ids
    assert [g.n_atoms for g in serial.graphs] == [g.n_atoms for g in parallel.graphs]


def test_unknown_fmt_rejected():
    with pytest.raises(ValueError, match="unknown fmt"):
        build_dataset(PDBS, fmt="bogus")


def test_pe_rejected_for_non_tensor_formats():
    with pytest.raises(ValueError, match="positional encodings"):
        build_dataset(PDBS, fmt="raw", pe="laplacian")


def test_empty_glob_raises():
    with pytest.raises(ValueError, match="no files matched"):
        build_dataset(f"{DATA}/*.nope", fmt="raw")


def test_save_writes_manifest_and_files(tmp_path):
    ds = build_dataset(PDBS, fmt="raw", split=(0.34, 0.33, 0.33))
    out = ds.save(str(tmp_path))
    files = set(os.listdir(out))
    assert "manifest.json" in files
    for gid in ds.ids:
        assert f"{gid}.pkl" in files

    import json

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["fmt"] == "raw"
    assert manifest["ids"] == ds.ids
    assert manifest["split"]["method"] == ds.split.method


def test_networkx_format():
    pytest.importorskip("networkx")
    ds = build_dataset([f"{DATA}/1fqy.pdb"], fmt="networkx")
    import networkx as nx

    assert isinstance(ds.graphs[0], nx.Graph)


def test_pyg_format_with_pe_and_labels():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    ds = build_dataset(
        [f"{DATA}/1fqy.pdb", f"{DATA}/3ptb.pdb"],
        fmt="pyg",
        node_features="ml",
        edge_features="geom",
        pe="laplacian",
        pe_k=4,
        labels={"1fqy": 1.0, "3ptb": 0.0},
    )
    import torch

    data = ds.graphs[0]
    assert data.edge_index.shape[0] == 2
    assert hasattr(data, "y") and float(data.y[0]) == 1.0
    assert isinstance(data.x, torch.Tensor)


def test_dataset_summary():
    from molscope.dataset import GraphDataset
    from molscope.prepare import SplitResult

    split = SplitResult(
        method="random",
        train=[0],
        val=[],
        test=[1],
    )
    ds = GraphDataset(
        graphs=[None, None],
        ids=["mol1", "mol2"],
        fmt="raw",
        labels=[1.5, None],
        split=split,
        skipped=[("failed_mol", "ValueError: bad format")],
        feature_names={"node_features": ["feat1"], "edge_features": ["feat2"]},
    )
    summary = ds.summary()
    assert "GraphDataset: 2 graph(s)" in summary
    assert "fmt='raw'" in summary
    assert "node_features: 1 (feat1)" in summary
    assert "edge_features: 1 (feat2)" in summary
    assert "labels: 1/2 graphs labelled" in summary
    assert "split: train=1, validation=0, test=1" in summary
    assert "skipped: 1 source(s)" in summary
    assert "failed_mol: ValueError: bad format" in summary

    # Empty summary to cover the False branches of summary()
    ds_empty = GraphDataset(graphs=[], ids=[], fmt="raw")
    summary_empty = ds_empty.summary()
    assert "GraphDataset: 0 graph(s)" in summary_empty
    assert "labels" not in summary_empty
    assert "split" not in summary_empty
    assert "skipped" not in summary_empty


def test_invalid_on_error():
    with pytest.raises(ValueError, match="on_error must be"):
        build_dataset(PDBS, fmt="raw", on_error="invalid")


def test_invalid_splits():
    with pytest.raises(ValueError, match="split must be a"):
        build_dataset(PDBS, fmt="raw", split="not-a-tuple")
    with pytest.raises(ValueError, match="split must be a"):
        build_dataset(PDBS, fmt="raw", split=(0.5, "foo", 0.5))


def test_coerce_string_label():
    labels = {"1fqy": "active", "3ptb": "inactive"}
    ds = build_dataset(PDBS, fmt="raw", labels=labels)
    by_id = dict(zip(ds.ids, ds.labels))
    assert by_id["1fqy"] == "active"
    assert by_id["3ptb"] == "inactive"


def test_read_label_csv_empty_header(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")
    ds = build_dataset(PDBS, fmt="raw", labels=str(csv_path))
    assert ds.labels == [None, None, None]


def test_read_label_csv_short_row(tmp_path):
    csv_path = tmp_path / "short_row.csv"
    csv_path.write_text("id,target\n1fqy,2.5\n3ptb\n")
    ds = build_dataset(PDBS, fmt="raw", labels=str(csv_path))
    by_id = dict(zip(ds.ids, ds.labels))
    assert by_id["1fqy"] == 2.5
    assert by_id["3ptb"] is None


def test_read_label_csv_with_string_label(tmp_path):
    csv_path = tmp_path / "string_labels.csv"
    csv_path.write_text("id,target\n1fqy,active\n3ptb,inactive\n")
    ds = build_dataset(PDBS, fmt="raw", labels=str(csv_path))
    by_id = dict(zip(ds.ids, ds.labels))
    assert by_id["1fqy"] == "active"
    assert by_id["3ptb"] == "inactive"


def test_save_networkx(tmp_path):
    pytest.importorskip("networkx")
    ds = build_dataset([f"{DATA}/1fqy.pdb"], fmt="networkx")
    out = ds.save(str(tmp_path))
    files = os.listdir(out)
    assert "1fqy.json" in files
    assert "manifest.json" in files


def test_save_pyg(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    ds = build_dataset([f"{DATA}/1fqy.pdb"], fmt="pyg")
    out = ds.save(str(tmp_path))
    files = os.listdir(out)
    assert "1fqy.pt" in files
    assert "manifest.json" in files


def test_save_dgl_mocked(tmp_path):
    import sys
    from unittest.mock import MagicMock

    mock_save_graphs = MagicMock()

    original_modules = {}
    for mod in ["dgl", "dgl.data.utils"]:
        if mod in sys.modules:
            original_modules[mod] = sys.modules[mod]
        sys.modules[mod] = MagicMock()

    try:
        sys.modules["dgl.data.utils"].save_graphs = mock_save_graphs

        from molscope.dataset import GraphDataset

        mock_graph = MagicMock()
        ds = GraphDataset(graphs=[mock_graph], ids=["dummy_dgl"], fmt="dgl")

        out = ds.save(str(tmp_path))
        assert "manifest.json" in os.listdir(out)
        mock_save_graphs.assert_called_once()
        call_args = mock_save_graphs.call_args[0]
        assert "dummy_dgl.bin" in call_args[0]
        assert call_args[1] == [mock_graph]

    finally:
        for mod in ["dgl", "dgl.data.utils"]:
            if mod in original_modules:
                sys.modules[mod] = original_modules[mod]
            else:
                sys.modules.pop(mod, None)


def test_build_dgl_mocked():
    import sys
    from unittest.mock import MagicMock

    mock_dgl = MagicMock()
    mock_torch = MagicMock()

    original_modules = {}
    for mod, mock_obj in [("dgl", mock_dgl), ("torch", mock_torch)]:
        if mod in sys.modules:
            original_modules[mod] = sys.modules[mod]
        sys.modules[mod] = mock_obj

    try:
        ds = build_dataset(PDBS[:1], fmt="dgl")
        assert ds.fmt == "dgl"
        assert len(ds.graphs) == 1
        mock_dgl.graph.assert_called_once()
    finally:
        for mod in ["dgl", "torch"]:
            if mod in original_modules:
                sys.modules[mod] = original_modules[mod]
            else:
                sys.modules.pop(mod, None)


def test_build_dataset_single_molecule():
    mol = ms.read(PDBS[0])
    ds = build_dataset(mol, fmt="raw")
    assert len(ds) == 1
    assert ds.ids[0] == mol.name


def test_build_dataset_directory_source(tmp_path):
    # Pass a single directory path
    ds = build_dataset(DATA, fmt="raw")
    assert len(ds) >= 3
    assert "1fqy" in ds.ids

    # Pass a list containing a directory path
    ds2 = build_dataset([DATA], fmt="raw")
    assert len(ds2) >= 3
    assert "1fqy" in ds2.ids

    # Test .gz file matching and extension extraction in directory walking
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    gz_file = subdir / "dummy.pdb.gz"
    import gzip
    with open(PDBS[0], "rb") as f_in:
        with gzip.open(gz_file, "wb") as f_out:
            f_out.write(f_in.read())

    # Also create a non-structure .gz file (which should be ignored)
    ignored_gz = subdir / "ignore.nope.gz"
    ignored_gz.write_text("ignore")

    # Also create a short-named .gz file (which should be ignored)
    ignored_short_gz = subdir / "ignore.gz"
    ignored_short_gz.write_text("ignore")

    # Also create a non-structure non-gz file (which should be ignored)
    ignored_txt = subdir / "ignore.txt"
    ignored_txt.write_text("ignore")

    ds_gz = build_dataset(str(subdir), fmt="raw")
    assert len(ds_gz) == 1
    assert ds_gz.ids[0] == "dummy.pdb"


def test_csv_column_lookup_errors(tmp_path):
    csv_path = tmp_path / "columns.csv"
    csv_path.write_text("id,target\n1fqy,2.5\n")

    with pytest.raises(ValueError, match="id_col='missing_id' not found in CSV header"):
        build_dataset(PDBS, fmt="raw", labels=str(csv_path), id_col="missing_id")

    with pytest.raises(ValueError, match="label_col='missing_target' not found in CSV header"):
        build_dataset(PDBS, fmt="raw", labels=str(csv_path), label_col="missing_target")


def test_round_trip_loading_raw(tmp_path):
    from molscope.dataset import GraphDataset

    ds = build_dataset(PDBS, fmt="raw", split=(0.34, 0.33, 0.33))
    out_dir = ds.save(str(tmp_path / "raw_ds"))

    loaded = GraphDataset.load(out_dir)
    assert loaded.fmt == "raw"
    assert loaded.ids == ds.ids
    assert len(loaded.graphs) == len(ds.graphs)
    assert loaded.split is not None
    assert loaded.split.method == ds.split.method
    assert loaded.split.train == ds.split.train


def test_round_trip_loading_networkx(tmp_path):
    pytest.importorskip("networkx")
    from molscope.dataset import GraphDataset

    ds = build_dataset(PDBS[:2], fmt="networkx")
    out_dir = ds.save(str(tmp_path / "nx_ds"))

    loaded = GraphDataset.load(out_dir)
    assert loaded.fmt == "networkx"
    assert loaded.ids == ds.ids
    import networkx as nx
    assert isinstance(loaded.graphs[0], nx.Graph)


def test_round_trip_loading_pyg(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from molscope.dataset import GraphDataset

    ds = build_dataset(PDBS[:2], fmt="pyg")
    out_dir = ds.save(str(tmp_path / "pyg_ds"))

    loaded = GraphDataset.load(out_dir)
    assert loaded.fmt == "pyg"
    assert loaded.ids == ds.ids
    import torch
    assert isinstance(loaded.graphs[0], torch.nn.Module) or hasattr(loaded.graphs[0], "edge_index")


def test_round_trip_loading_dgl_mocked(tmp_path):
    import sys
    from unittest.mock import MagicMock

    mock_load_graphs = MagicMock()
    mock_load_graphs.return_value = [["mock_loaded_graph"], "dummy_labels"]

    original_modules = {}
    for mod in ["dgl", "dgl.data.utils"]:
        if mod in sys.modules:
            original_modules[mod] = sys.modules[mod]
        sys.modules[mod] = MagicMock()

    try:
        sys.modules["dgl.data.utils"].load_graphs = mock_load_graphs

        from molscope.dataset import GraphDataset

        manifest_path = tmp_path / "manifest.json"
        import json
        manifest_data = {
            "fmt": "dgl",
            "ids": ["mol_dgl"],
            "files": ["mol_dgl.bin"],
            "labels": None,
            "skipped": [],
        }
        manifest_path.write_text(json.dumps(manifest_data))

        (tmp_path / "mol_dgl.bin").write_text("")

        loaded = GraphDataset.load(str(tmp_path))
        assert loaded.fmt == "dgl"
        assert loaded.graphs == ["mock_loaded_graph"]
        mock_load_graphs.assert_called_once()
    finally:
        for mod in ["dgl", "dgl.data.utils"]:
            if mod in original_modules:
                sys.modules[mod] = original_modules[mod]
            else:
                sys.modules.pop(mod, None)


def test_load_dataset_missing_manifest():
    from molscope.dataset import GraphDataset

    with pytest.raises(FileNotFoundError, match="manifest.json not found"):
        GraphDataset.load("does_not_exist_dir")


# --- loader() bridge ------------------------------------------------------


def _raw_dataset(split=None):
    """A minimal GraphDataset for loader() validation (no framework needed)."""
    from molscope.dataset import GraphDataset

    return GraphDataset(graphs=[None, None], ids=["a", "b"], fmt="raw", split=split)


def test_loader_rejects_non_framework_fmt():
    # fmt is checked before any graphs are touched, so this runs everywhere.
    with pytest.raises(ValueError, match="fmt='pyg' or 'dgl'"):
        _raw_dataset().loader()


def test_loader_unknown_split_name():
    from molscope.dataset import GraphDataset

    ds = GraphDataset(graphs=[None], ids=["a"], fmt="pyg")
    with pytest.raises(ValueError, match="unknown split 'trian'"):
        ds._loader_subset("trian")


def test_loader_named_split_without_split():
    from molscope.dataset import GraphDataset

    ds = GraphDataset(graphs=[None], ids=["a"], fmt="pyg")
    with pytest.raises(ValueError, match="no split available"):
        ds._loader_subset("train")


def test_loader_subset_selects_split():
    from molscope.prepare import SplitResult

    split = SplitResult(method="random", train=[0], val=[], test=[1])
    ds = _raw_dataset(split=split)
    ds.graphs = ["g0", "g1"]
    assert ds._loader_subset("train") == ["g0"]
    assert ds._loader_subset("test") == ["g1"]
    assert ds._loader_subset(None) == ["g0", "g1"]


def test_loader_pyg_batches_graphs():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.loader import DataLoader

    ds = build_dataset(
        [f"{DATA}/1fqy.pdb", f"{DATA}/3ptb.pdb"],
        fmt="pyg",
        split=(0.5, 0.0, 0.5),
        seed=0,
    )
    loader = ds.loader(batch_size=2)
    assert isinstance(loader, DataLoader)
    batches = list(loader)
    assert len(batches) == 1  # both graphs in one batch
    assert batches[0].num_graphs == 2

    # a named split draws only from that subset
    train_loader = ds.loader("train", batch_size=1)
    assert sum(b.num_graphs for b in train_loader) == len(ds.train)


def test_loader_shuffle_default_follows_split():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch.utils.data import RandomSampler, SequentialSampler

    ds = build_dataset(PDBS, fmt="pyg", split=(0.34, 0.33, 0.33), seed=0)
    # shuffle is reflected in the sampler torch picks, not a public attribute.
    # train defaults to shuffling; val/test/whole default to not shuffling.
    assert isinstance(ds.loader("train").sampler, RandomSampler)
    assert isinstance(ds.loader("test").sampler, SequentialSampler)
    assert isinstance(ds.loader().sampler, SequentialSampler)
    # explicit override wins
    assert isinstance(ds.loader("train", shuffle=False).sampler, SequentialSampler)

