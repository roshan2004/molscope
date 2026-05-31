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
