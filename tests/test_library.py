"""Tests for molecule-table I/O, SMILES descriptors, and diverse selection.

The CSV and MaxMin-selection paths need no optional backend and run everywhere.
The XLSX cases skip without ``openpyxl``; the SMILES cases skip without RDKit.
"""

import csv

import numpy as np
import pytest

from molscope.cli import main
from molscope.library import (
    MoleculeTable,
    protonate_smiles,
    read_table,
    select_diverse,
    smiles_descriptors,
)


def _write_csv(path, rows, columns):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


# A small library: m1/m2/m5 cluster together; m3 is extreme; m4 is in between.
LIB_ROWS = [
    {"ID": "m1", "MW": "100", "ALogP": "0.5"},
    {"ID": "m2", "MW": "101", "ALogP": "0.6"},
    {"ID": "m3", "MW": "500", "ALogP": "5.0"},
    {"ID": "m4", "MW": "300", "ALogP": "2.5"},
    {"ID": "m5", "MW": "99", "ALogP": "0.4"},
]
LIB_COLS = ["ID", "MW", "ALogP"]


def test_read_csv_round_trip(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    table = read_table(path)
    assert table.columns == LIB_COLS
    assert len(table) == 5
    assert table.column("ID") == ["m1", "m2", "m3", "m4", "m5"]


def test_numeric_matrix_coerces_and_nans(tmp_path):
    rows = [{"ID": "a", "MW": "100", "ALogP": ""}, {"ID": "b", "MW": "x", "ALogP": "2"}]
    path = _write_csv(tmp_path / "m.csv", rows, LIB_COLS)
    table = read_table(path)
    matrix = table.numeric_matrix(["MW", "ALogP"])
    assert matrix.shape == (2, 2)
    assert np.isnan(matrix[0, 1])  # empty cell
    assert np.isnan(matrix[1, 0])  # non-numeric cell
    assert matrix[0, 0] == 100.0


def test_numeric_matrix_missing_column_raises(tmp_path):
    path = _write_csv(tmp_path / "m.csv", LIB_ROWS, LIB_COLS)
    with pytest.raises(KeyError):
        read_table(path).numeric_matrix(["nope"])


def test_select_diverse_prefers_spread_points():
    matrix = np.array(_mw_alogp())
    picks = set(select_diverse(matrix, 3))
    # The extreme point (m3, index 2) and the low cluster's edges must appear;
    # the near-duplicate of an already-picked point should not crowd the set.
    assert 2 in picks  # extreme point is always chosen
    assert len(picks) == 3


def test_select_diverse_excludes_nan_rows():
    matrix = np.array([[0.0, 0.0], [10.0, 10.0], [np.nan, 1.0], [5.0, 5.0]])
    picks = select_diverse(matrix, 4)
    assert 2 not in picks  # NaN row excluded
    assert len(picks) == 3  # only 3 complete rows exist


def test_select_diverse_is_deterministic():
    matrix = np.array([[float(i), float(i % 3)] for i in range(20)])
    assert select_diverse(matrix, 5) == select_diverse(matrix, 5)


def test_select_diverse_rejects_bad_args():
    with pytest.raises(ValueError):
        select_diverse(np.zeros((3, 2)), 0)
    with pytest.raises(ValueError):
        select_diverse(np.full((3, 2), np.nan), 2)


def test_cli_select_on_csv(tmp_path, capsys):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    out = tmp_path / "picked.csv"
    rc = main(["select", path, "--descriptor-cols", "MW", "ALogP", "-n", "3", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    picked = read_table(str(out))
    assert len(picked) == 3
    assert "MW" in picked.columns
    captured = capsys.readouterr()
    assert "selected 3 of 5" in captured.out


def test_cli_select_requires_a_descriptor_source(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    assert main(["select", path, "-n", "2"]) == 2


def test_cli_select_compute_without_smiles_col_errors(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    assert main(["select", path, "-n", "2", "--compute-descriptors"]) == 2


# -- optional backends ------------------------------------------------------

def _mw_alogp():
    return [[100.0, 0.5], [101.0, 0.6], [500.0, 5.0], [300.0, 2.5], [99.0, 0.4]]


def test_xlsx_round_trip(tmp_path):
    pytest.importorskip("openpyxl")
    table = MoleculeTable(columns=LIB_COLS, rows=LIB_ROWS)
    path = tmp_path / "lib.xlsx"
    table.write(str(path))
    back = read_table(str(path))
    assert back.columns == LIB_COLS
    assert len(back) == 5
    assert back.column("ID") == ["m1", "m2", "m3", "m4", "m5"]


def test_smiles_descriptors_and_invalid_rows():
    pytest.importorskip("rdkit")
    smis = ["CCO", "c1ccccc1", "not-a-smiles", ""]
    matrix, names = smiles_descriptors(smis)
    assert matrix.shape == (4, len(names))
    assert "MolWt" in names and "MolLogP" in names
    assert matrix[0, names.index("MolWt")] > 0
    assert np.isnan(matrix[2]).all()  # unparseable
    assert np.isnan(matrix[3]).all()  # empty


def test_cli_select_compute_descriptors_from_smiles(tmp_path, capsys):
    pytest.importorskip("rdkit")
    rows = [
        {"ID": "ethanol", "SMILES": "CCO"},
        {"ID": "benzene", "SMILES": "c1ccccc1"},
        {"ID": "caffeine", "SMILES": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
        {"ID": "acetic", "SMILES": "CC(=O)O"},
    ]
    path = _write_csv(tmp_path / "smi.csv", rows, ["ID", "SMILES"])
    out = tmp_path / "picked.csv"
    rc = main([
        "select", path, "--smiles-col", "SMILES", "--compute-descriptors",
        "-n", "2", "--out", str(out),
    ])
    assert rc == 0
    picked = read_table(str(out))
    assert len(picked) == 2
    # Computed descriptor columns are carried into the output.
    assert "MolWt" in picked.columns and "MolLogP" in picked.columns


# -- error paths and edge cases --------------------------------------------

def test_numeric_matrix_empty_names_raises():
    table = MoleculeTable(columns=["a"], rows=[{"a": "1"}])
    with pytest.raises(ValueError):
        table.numeric_matrix([])


def test_numeric_matrix_native_numeric_and_none():
    table = MoleculeTable(columns=["x"], rows=[{"x": 5}, {"x": 5.5}, {"x": None}])
    matrix = table.numeric_matrix(["x"])
    assert matrix[0, 0] == 5.0
    assert matrix[1, 0] == 5.5
    assert np.isnan(matrix[2, 0])


def test_read_table_unsupported_extension():
    with pytest.raises(ValueError):
        read_table("molecules.json")


def test_read_csv_without_header_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValueError):
        read_table(str(path))


def test_read_xlsx_skips_blank_rows(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "gappy.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["ID", "MW"])
    sheet.append(["m1", 100])
    sheet.append([None, None])  # blank row, as real spreadsheets often have
    sheet.append(["m2", 200])
    workbook.save(str(path))
    table = read_table(str(path))
    assert len(table) == 2
    assert table.column("ID") == ["m1", "m2"]


def test_write_unsupported_extension(tmp_path):
    table = MoleculeTable(columns=["a"], rows=[{"a": "1"}])
    with pytest.raises(ValueError):
        table.write(str(tmp_path / "out.json"))


def test_select_diverse_requires_2d():
    with pytest.raises(ValueError):
        select_diverse(np.array([1.0, 2.0, 3.0]), 2)


def test_select_diverse_without_standardize_still_picks_extreme():
    picks = select_diverse(np.array(_mw_alogp()), 3, standardize=False)
    assert 2 in picks  # the extreme point is chosen regardless of scaling
    assert len(picks) == 3


def test_smiles_descriptors_unknown_name_raises():
    pytest.importorskip("rdkit")
    with pytest.raises(ValueError):
        smiles_descriptors(["CCO"], names=["NotARealDescriptor"])


def test_read_xlsx_empty_worksheet_raises(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "empty.xlsx"
    openpyxl.Workbook().save(str(path))  # one sheet, no rows
    with pytest.raises(ValueError):
        read_table(str(path))


def test_cli_select_num_must_be_positive(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    assert main(["select", path, "--descriptor-cols", "MW", "-n", "0"]) == 2


def test_cli_select_read_error(tmp_path):
    assert main(["select", str(tmp_path / "nope.json"), "--descriptor-cols", "MW", "-n", "2"]) == 2


def test_cli_select_missing_column(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    assert main(["select", path, "--descriptor-cols", "NOPE", "-n", "2"]) == 2


def test_cli_select_all_nan_descriptors(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    # ID is all strings -> all-NaN selection space -> selection error.
    assert main(["select", path, "--descriptor-cols", "ID", "-n", "2"]) == 2


def test_cli_select_fewer_rows_than_requested(tmp_path, capsys):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    rc = main(["select", path, "--descriptor-cols", "MW", "ALogP", "-n", "10"])
    assert rc == 0
    assert "fewer than the requested 10" in capsys.readouterr().err


def test_cli_select_write_error(tmp_path):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    out = tmp_path / "out.json"  # unsupported output extension
    assert main(
        ["select", path, "--descriptor-cols", "MW", "ALogP", "-n", "2", "--out", str(out)]
    ) == 2


def test_cli_select_summary_without_out(tmp_path, capsys):
    path = _write_csv(tmp_path / "lib.csv", LIB_ROWS, LIB_COLS)
    rc = main(["select", path, "--descriptor-cols", "MW", "ALogP", "-n", "2"])
    assert rc == 0
    assert "selected 2 of 5" in capsys.readouterr().out


# -- pKa-aware SMILES protonation (Dimorphite-DL) ---------------------------

def test_protonate_smiles_tracks_ph():
    """Dimorphite-DL returns the dominant ionisation state at the target pH."""
    pytest.importorskip("rdkit")
    pytest.importorskip("dimorphite_dl")

    acid, amine, arene = "CC(=O)O", "CCN", "c1ccccc1"
    at74 = protonate_smiles([acid, amine, arene], ph=7.4)
    assert at74[0] == "CC(=O)[O-]"   # carboxylic acid is deprotonated
    assert at74[1] == "CC[NH3+]"     # amine is protonated
    assert at74[2] == arene          # neutral arene is unchanged

    # The acid stays neutral well below its pKa; the amine neutralises above its.
    assert protonate_smiles([acid], ph=2.0)[0] == "CC(=O)O"
    assert protonate_smiles([amine], ph=12.0)[0] == "CCN"


def test_protonate_smiles_passes_through_blanks():
    pytest.importorskip("rdkit")
    pytest.importorskip("dimorphite_dl")
    assert protonate_smiles(["", None, "   "], ph=7.0) == ["", "", "   "]


def test_smiles_descriptors_pka_changes_charge_sensitive_values():
    """Protonating before featurising shifts charge-sensitive descriptors."""
    pytest.importorskip("rdkit")
    pytest.importorskip("dimorphite_dl")

    plain, _ = smiles_descriptors(["CC(=O)O"], names=["MolWt"])
    pka, _ = smiles_descriptors(["CC(=O)O"], names=["MolWt"], protonation="pka", ph=7.4)
    # Deprotonation removes one hydrogen (~1.008 Da).
    assert plain[0, 0] - pka[0, 0] == pytest.approx(1.008, abs=0.01)


def test_smiles_descriptors_rejects_bad_protonation():
    pytest.importorskip("rdkit")
    with pytest.raises(ValueError, match="protonation"):
        smiles_descriptors(["CCO"], protonation="basic")


def test_require_dimorphite_install_hint(monkeypatch):
    import builtins

    from molscope.library import _require_dimorphite

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dimorphite_dl":
            raise ImportError("no dimorphite")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"molscope\[dimorphite\]"):
        _require_dimorphite()
