"""Tests for dataset preparation: dedup, splits, descriptors, and the report.

The NumPy-only paths (random/diversity splits on existing numeric columns, exact
dedup, report writing) run everywhere. Scaffold splits, canonical dedup, RDKit
descriptors, fingerprints, and SDF input skip without RDKit.
"""

import csv
import json

import numpy as np
import pytest

from molscope.cli import main
from molscope.library import read_table
from molscope.prepare import (
    dedup_keys,
    diversity_split,
    prepare_dataset,
    random_split,
    scaffold_split,
)


def _write_csv(path, rows, columns):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


# A small library with numeric descriptor columns (no RDKit needed).
LIB_ROWS = [
    {"ID": f"m{i}", "MW": str(100 + 40 * i), "ALogP": str(0.3 * i)}
    for i in range(20)
]
LIB_COLS = ["ID", "MW", "ALogP"]

# SMILES library for the RDKit-gated paths.
SMI_ROWS = [
    {"ID": "a", "SMILES": "c1ccccc1"},        # benzene
    {"ID": "b", "SMILES": "c1ccccc1C"},       # toluene  (same ring scaffold)
    {"ID": "c", "SMILES": "c1ccccc1CC"},      # ethylbenzene
    {"ID": "d", "SMILES": "C1CCNCC1"},        # piperidine (different scaffold)
    {"ID": "e", "SMILES": "C1CCNCC1C"},       # methylpiperidine
    {"ID": "f", "SMILES": "CCO"},             # ethanol (no ring scaffold)
]
SMI_COLS = ["ID", "SMILES"]


# -- splitters (NumPy only) -------------------------------------------------

def test_random_split_partitions_all_rows():
    split = random_split(20, test=0.2, val=0.1, seed=0)
    allidx = sorted(split.train + split.val + split.test)
    assert allidx == list(range(20))
    assert split.sizes == {"train": 14, "validation": 2, "test": 4}


def test_random_split_is_deterministic_under_seed():
    a = random_split(50, test=0.2, val=0.2, seed=7)
    b = random_split(50, test=0.2, val=0.2, seed=7)
    c = random_split(50, test=0.2, val=0.2, seed=8)
    assert a.test == b.test
    assert a.test != c.test  # different seed -> different draw


def test_diversity_split_partitions_and_spreads():
    matrix = np.array([[float(i), float(i) ** 2] for i in range(20)])
    split = diversity_split(matrix, test=0.25, val=0.25)
    allidx = sorted(split.train + split.val + split.test)
    assert allidx == list(range(20))
    assert len(split.test) == 5 and len(split.val) == 5


def test_diversity_split_sends_incomplete_rows_to_train():
    matrix = np.array([[0.0], [1.0], [np.nan], [3.0], [4.0]])
    split = diversity_split(matrix, test=0.2, val=0.0)
    assert 2 in split.train  # NaN-descriptor row cannot be placed by diversity


def test_fractions_must_leave_training_rows():
    with pytest.raises(ValueError, match="must be < 1"):
        random_split(10, test=0.6, val=0.5)


# -- dedup ------------------------------------------------------------------

def test_exact_dedup_keeps_first_occurrence():
    kept, removed = dedup_keys(["x", "y", "x", "z", "y"], "exact")
    assert kept == [0, 1, 3]
    assert removed == 2


def test_exact_dedup_keeps_blank_keys():
    kept, removed = dedup_keys(["", "", "a"], "exact")
    assert kept == [0, 1, 2]  # blanks are not duplicates of each other
    assert removed == 0


# -- orchestrator (NumPy only) ----------------------------------------------

def test_prepare_random_on_table(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    ds = prepare_dataset(path, split="random", test=0.2, val=0.1, seed=1)
    assert ds.n_input == 20
    assert ds.n_prepared == 20
    assert ds.split.method == "random"


def test_prepare_diversity_uses_existing_columns(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    ds = prepare_dataset(
        path, split="diversity", descriptor_cols=["MW", "ALogP"], test=0.2, val=0.2
    )
    assert ds.descriptor_cols == ["MW", "ALogP"]
    assert sorted(ds.split.train + ds.split.val + ds.split.test) == list(range(20))


def test_prepare_diversity_without_descriptors_errors(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    with pytest.raises(ValueError, match="diversity split needs descriptors"):
        prepare_dataset(path, split="diversity")


def test_prepare_exact_dedup_on_table(tmp_path):
    rows = LIB_ROWS + [{"ID": "dup", "MW": "100", "ALogP": "0.0"}]  # MW/ALogP dup of m0
    path = _write_csv(tmp_path / "lib.csv", rows, LIB_COLS)
    ds = prepare_dataset(path, dedup="exact", smiles_col="MW")
    assert ds.n_duplicates == 1
    assert ds.n_prepared == 20


def test_write_emits_expected_files(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    ds = prepare_dataset(
        path, split="diversity", descriptor_cols=["MW", "ALogP"], test=0.2, val=0.2
    )
    out = tmp_path / "out"
    written = ds.write(str(out), make_figure=True)

    for name in ("train.csv", "validation.csv", "test.csv", "descriptors.csv",
                 "manifest.json", "report.md", "report.png"):
        assert (out / name).exists(), f"missing {name}"

    # Splits reload and together reconstruct the prepared table.
    total = sum(len(read_table(str(out / f"{n}.csv")))
                for n in ("train", "validation", "test"))
    assert total == ds.n_prepared

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["split_method"] == "diversity"
    assert manifest["split_sizes"]["test"] == 4
    assert "Dataset preparation report" in (out / "report.md").read_text()
    assert written  # non-empty list of paths


def test_write_skips_figure_when_disabled(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    ds = prepare_dataset(path, split="random")
    out = tmp_path / "out"
    ds.write(str(out), make_figure=False)
    assert not (out / "report.png").exists()
    assert not (out / "descriptors.csv").exists()  # no descriptors computed


def test_unknown_split_and_dedup_raise(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    with pytest.raises(ValueError, match="unknown split"):
        prepare_dataset(path, split="bogus")
    with pytest.raises(ValueError, match="unknown dedup"):
        prepare_dataset(path, dedup="bogus")


# -- CLI (NumPy only) -------------------------------------------------------

def test_cli_prepare_random(tmp_path, capsys):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    out = tmp_path / "prepared"
    rc = main(["prepare", path, "--out-dir", str(out), "--split", "random",
               "--test", "0.2", "--val", "0.1", "--seed", "3"])
    assert rc == 0
    assert (out / "train.csv").exists()
    captured = capsys.readouterr().out
    assert "prepared 20 of 20 molecules" in captured


def test_cli_prepare_rejects_bad_fractions(tmp_path, capsys):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    rc = main(["prepare", path, "--test", "0.7", "--val", "0.5"])
    assert rc == 2
    assert "sum to < 1" in capsys.readouterr().err


# -- RDKit-gated paths ------------------------------------------------------

def test_scaffold_split_keeps_scaffolds_intact():
    pytest.importorskip("rdkit")
    from molscope.prepare import murcko_scaffolds

    smiles = [r["SMILES"] for r in SMI_ROWS]
    split = scaffold_split(smiles, test=0.34, val=0.0)
    scaffolds = murcko_scaffolds(smiles)

    train_scaf = {scaffolds[i] for i in split.train}
    test_scaf = {scaffolds[i] for i in split.test}
    assert train_scaf.isdisjoint(test_scaf)  # no scaffold straddles the split
    assert split.test, "test split should not be empty for a 0.34 fraction"
    assert sorted(split.train + split.val + split.test) == list(range(len(smiles)))


def test_canonical_dedup_collapses_equivalent_smiles():
    pytest.importorskip("rdkit")
    # Two spellings of ethanol plus one distinct molecule.
    kept, removed = dedup_keys(["CCO", "OCC", "c1ccccc1"], "canonical")
    assert removed == 1
    assert kept == [0, 2]


def test_prepare_compute_descriptors_and_fingerprints(tmp_path):
    pytest.importorskip("rdkit")
    path = _write_csv(tmp_path / "smi.csv", SMI_ROWS, SMI_COLS)
    ds = prepare_dataset(
        path, smiles_col="SMILES", compute_descriptors=True,
        split="diversity", test=0.2, val=0.2, fingerprints=True,
    )
    assert "MolWt" in ds.descriptor_cols
    assert ds.fingerprint_col == "morgan_onbits"
    assert ds.fingerprint_params["radius"] == 2
    assert "morgan_onbits" in ds.table.columns


def test_prepare_scaffold_split_via_orchestrator(tmp_path):
    pytest.importorskip("rdkit")
    path = _write_csv(tmp_path / "smi.csv", SMI_ROWS, SMI_COLS)
    ds = prepare_dataset(path, smiles_col="SMILES", split="scaffold", test=0.34, val=0.0)
    assert ds.split.method == "scaffold"
    assert ds.n_prepared == 6


def test_prepare_reads_sdf(tmp_path):
    pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    sdf = tmp_path / "mols.sdf"
    writer = Chem.SDWriter(str(sdf))
    for smi in ["c1ccccc1", "CCO", "C1CCNCC1"]:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=1)
        writer.write(mol)
    writer.close()

    ds = prepare_dataset(str(sdf), split="random", test=0.34, val=0.0)
    assert ds.smiles_col == "smiles"
    assert ds.n_input == 3
    assert "smiles" in ds.table.columns


# -- pKa-aware protonation before featurising (Dimorphite-DL) ---------------

def test_prepare_pka_protonation_changes_descriptors_and_manifest(tmp_path):
    pytest.importorskip("rdkit")
    pytest.importorskip("dimorphite_dl")
    rows = [
        {"ID": "acetic", "SMILES": "CC(=O)O"},
        {"ID": "ethylamine", "SMILES": "CCN"},
    ]
    path = _write_csv(tmp_path / "smi.csv", rows, ["ID", "SMILES"])

    plain = prepare_dataset(
        path, smiles_col="SMILES", compute_descriptors=True,
        rdkit_descriptors=["MolWt"], split="random", test=0.0, val=0.0,
    )
    pka = prepare_dataset(
        path, smiles_col="SMILES", compute_descriptors=True,
        rdkit_descriptors=["MolWt"], split="random", test=0.0, val=0.0,
        protonation="pka", ph=7.4,
    )

    # The deprotonated acid weighs ~1 Da less than its neutral form.
    plain_mw = plain.table.numeric_matrix(["MolWt"])[0, 0]
    pka_mw = pka.table.numeric_matrix(["MolWt"])[0, 0]
    assert plain_mw - pka_mw == pytest.approx(1.008, abs=0.01)

    # The stored SMILES column is left on the original input SMILES.
    assert pka.table.column("SMILES")[0] == "CC(=O)O"

    manifest = pka.manifest()
    assert manifest["protonation"] == "pka"
    assert manifest["ph"] == 7.4
    assert plain.manifest()["ph"] is None


def test_prepare_pka_without_smiles_col_errors(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    with pytest.raises(ValueError, match="smiles_col"):
        prepare_dataset(
            path, compute_descriptors=False, descriptor_cols=["MW"],
            split="diversity", protonation="pka", fingerprints=True,
        )


def test_prepare_rejects_bad_protonation(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    with pytest.raises(ValueError, match="protonation"):
        prepare_dataset(path, split="random", protonation="acidic")
