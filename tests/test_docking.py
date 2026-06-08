"""Tests for the docking post-processing module and its CLI subcommands."""

import os
import sys

import numpy as np
import pytest

from molscope import docking

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
POSES_SDF = os.path.join(FIXTURES, "docking_poses.sdf")


# -- a richer multi-ligand SDF for clustering/ranking, built with RDKit -----

_LIGANDS = [
    ("ethanol", "CCO", -5.1, 0.30),
    ("propanol", "CCCO", -5.0, 0.28),          # ethanol analogue
    ("benzene", "c1ccccc1", -7.2, 0.65),
    ("toluene", "Cc1ccccc1", -7.4, 0.70),      # benzene analogue
    ("xylene", "Cc1ccccc1C", -7.1, 0.66),      # benzene analogue
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O", -8.9, 0.88),
    ("pyridine", "c1ccncc1", -6.3, 0.50),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", -8.1, 0.80),
]


def _write_ligand_sdf(path, score_field="minimizedAffinity", offset=0.0):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    writer = Chem.SDWriter(str(path))
    for name, smi, vina, cnn in _LIGANDS:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", name)
        mol.SetProp(score_field, f"{vina + offset:.2f}")
        mol.SetProp("CNNscore", f"{cnn:.2f}")
        writer.write(mol)
    writer.close()
    return str(path)


# -- reading & field discovery ----------------------------------------------

def test_read_poses_keeps_properties_and_raw_block():
    poses = docking.read_poses(POSES_SDF)
    assert [p.name for p in poses] == ["ligA_pose1", "ligA_pose2", "ligB_pose1"]
    assert poses[0].index == 1
    assert poses[0].properties["minimizedAffinity"] == "-8.4"
    assert poses[0].score("minimizedAffinity") == pytest.approx(-8.4)
    # The raw block is preserved (minus the $$$$ terminator) for faithful re-export.
    assert "ligA_pose1" in poses[0].block
    assert "$$$$" not in poses[0].block


def test_score_returns_nan_for_missing_or_nonnumeric_field():
    poses = docking.read_poses(POSES_SDF)
    assert np.isnan(poses[0].score("does_not_exist"))


def test_resolve_score_field_explicit_and_autodetect():
    poses = docking.read_poses(POSES_SDF)
    assert docking.resolve_score_field(poses, "CNNscore") == "CNNscore"
    # minimizedAffinity is a known field, so it auto-detects.
    assert docking.resolve_score_field(poses, None) == "minimizedAffinity"


def test_resolve_score_field_unknown_lists_available():
    poses = docking.read_poses(POSES_SDF)
    with pytest.raises(ValueError, match="available fields:.*minimizedAffinity"):
        docking.resolve_score_field(poses, "nope")


def test_higher_is_better_table_and_assumption():
    assert docking.higher_is_better("minimizedAffinity") == (False, False)
    assert docking.higher_is_better("CNNscore") == (True, False)
    # Unknown field -> assumed lower-is-better, flagged as assumed.
    assert docking.higher_is_better("weird_field") == (False, True)
    # Explicit override wins.
    assert docking.higher_is_better("weird_field", higher={"weird_field"}) == (True, False)


# -- feature 1: summary -----------------------------------------------------

def test_summarize_ranks_and_computes_ligand_efficiency():
    poses = docking.read_poses(POSES_SDF)
    result = docking.summarize(
        poses, "minimizedAffinity", higher_is_better_flag=False, with_smiles=False,
    )
    assert [r["name"] for r in result.rows] == ["ligA_pose1", "ligB_pose1", "ligA_pose2"]
    assert [r["rank"] for r in result.rows] == [1, 2, 3]
    # Lower-is-better affinity -> efficiency is -score / heavy atoms (positive).
    top = result.rows[0]
    assert top["ligand_efficiency"] == pytest.approx(8.4 / top["n_heavy_atoms"])
    assert result.n_missing == 0


def test_summarize_higher_is_better_reverses_order():
    poses = docking.read_poses(POSES_SDF)
    result = docking.summarize(poses, "CNNscore", higher_is_better_flag=True, with_smiles=False)
    assert [r["name"] for r in result.rows] == ["ligA_pose1", "ligB_pose1", "ligA_pose2"]
    assert result.rows[0]["score"] == pytest.approx(0.91)


def test_summarize_counts_missing_scores(tmp_path):
    sdf = tmp_path / "partial.sdf"
    sdf.write_text(
        "a\n p\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n>  <score>\n-5.0\n\n$$$$\n"
        "b\n p\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n$$$$\n"
    )
    poses = docking.read_poses(str(sdf))
    result = docking.summarize(poses, "score", higher_is_better_flag=False, with_smiles=False)
    assert len(result.rows) == 1
    assert result.n_missing == 1


def test_plot_score_distribution_writes_png(tmp_path):
    out = tmp_path / "dist.png"
    assert docking.plot_score_distribution(np.array([-8.4, -7.2, -6.9]), "score", str(out))
    assert out.exists() and out.stat().st_size > 0


def test_write_poses_sdf_round_trips(tmp_path):
    from molscope import read_sdf_frames

    poses = docking.read_poses(POSES_SDF)
    out = tmp_path / "subset.sdf"
    docking.write_poses_sdf(poses[:2], str(out))
    frames = read_sdf_frames(str(out))
    assert [m.name for m in frames] == ["ligA_pose1", "ligA_pose2"]
    assert frames[0].properties["minimizedAffinity"] == "-8.4"


def test_summarize_smiles_needs_rdkit():
    pytest.importorskip("rdkit")
    poses = docking.read_poses(POSES_SDF)
    result = docking.summarize(poses, "minimizedAffinity", higher_is_better_flag=False)
    assert result.with_smiles
    assert all(r["smiles"] for r in result.rows)


# -- feature 2: diversity-aware selection -----------------------------------

def test_select_diverse_picks_representatives_not_analogues(tmp_path):
    pytest.importorskip("rdkit")
    sdf = _write_ligand_sdf(tmp_path / "vina.sdf")
    poses = docking.read_poses(sdf)
    result = docking.select_diverse_hits(
        poses, "minimizedAffinity", higher_is_better_flag=False,
        top=8, select=3, threshold=0.5,
    )
    assert len(result.selected) == 3
    # Best-scoring representative leads, and rows carry cluster provenance.
    assert result.selected[0]["name"] == "aspirin"
    assert all("cluster_id" in s and "cluster_size" in s for s in result.selected)
    # Selected names are distinct molecules (no duplicate analogues).
    names = [s["name"] for s in result.selected]
    assert len(set(names)) == len(names)


def test_select_diverse_caps_when_fewer_clusters_than_requested(tmp_path):
    pytest.importorskip("rdkit")
    sdf = _write_ligand_sdf(tmp_path / "vina.sdf")
    poses = docking.read_poses(sdf)
    result = docking.select_diverse_hits(
        poses, "minimizedAffinity", higher_is_better_flag=False,
        top=8, select=100, threshold=0.4,
    )
    assert result.capped_below_request
    assert len(result.selected) == result.n_clusters < 100


def test_select_diverse_writes_sdf_and_csv_via_cli(tmp_path):
    pytest.importorskip("rdkit")
    from molscope.cli import main

    sdf = _write_ligand_sdf(tmp_path / "vina.sdf")
    out = tmp_path / "out"
    rc = main([
        "dock-diverse", sdf, "--score-field", "minimizedAffinity",
        "--top", "8", "--select", "3", "--threshold", "0.5", "--out-dir", str(out),
    ])
    assert rc == 0
    assert (out / "diverse_hits.csv").exists()
    assert (out / "diverse_hits.sdf").exists()


def test_select_diverse_carries_direction_assumed(tmp_path):
    pytest.importorskip("rdkit")
    sdf = _write_ligand_sdf(tmp_path / "vina.sdf")
    kwargs = dict(higher_is_better_flag=False, top=8, select=3, threshold=0.5)
    assumed = docking.select_diverse_hits(
        docking.read_poses(sdf), "minimizedAffinity", direction_assumed=True, **kwargs
    )
    assert assumed.direction_assumed is True
    # Defaults to False when the caller knows the direction.
    known = docking.select_diverse_hits(
        docking.read_poses(sdf), "minimizedAffinity", **kwargs
    )
    assert known.direction_assumed is False


def test_dock_diverse_cli_warns_on_assumed_direction(tmp_path, capsys):
    pytest.importorskip("rdkit")
    from molscope.cli import main

    # An unrecognised score field forces the lower-is-better assumption.
    sdf = _write_ligand_sdf(tmp_path / "vina.sdf", score_field="customScore")
    rc = main([
        "dock-diverse", sdf, "--score-field", "customScore",
        "--top", "8", "--select", "3", "--threshold", "0.5",
        "--out-dir", str(tmp_path / "out"),
    ])
    assert rc == 0
    assert "assumed" in capsys.readouterr().out


# -- feature 3: consensus ranking -------------------------------------------

def test_consensus_rank_joins_by_name_and_averages_ranks():
    # Two "files" scoring the same molecules; core path needs no RDKit.
    poses = docking.read_poses(POSES_SDF)
    result = docking.consensus_rank(
        [("vina", poses), ("gnina", poses)], score_fields=["minimizedAffinity"], key="name",
    )
    assert result.rows[0]["key"] == "ligA_pose1"      # best affinity in both
    assert result.rows[0]["final_rank"] == 1
    assert result.rows[0]["consensus_rank"] == pytest.approx(1.0)
    # Direction of the (duplicated) score column is reported.
    assert all(d is False for d in result.directions.values())


def test_consensus_rank_reports_assumed_direction():
    poses = docking.read_poses(POSES_SDF)
    result = docking.consensus_rank([("f", poses)], score_fields=["CNNscore"])
    assert "CNNscore" in result.directions
    assert result.directions["CNNscore"] is True   # known higher-is-better
    assert result.assumed == []


def test_consensus_rank_no_numeric_fields_errors():
    poses = docking.read_poses(POSES_SDF)
    with pytest.raises(ValueError, match="no numeric score fields"):
        docking.consensus_rank([("f", poses)], score_fields=["not_a_field"])


def test_consensus_rank_mw_filter_drops_heavy(tmp_path):
    pytest.importorskip("rdkit")
    sdf = _write_ligand_sdf(tmp_path / "vina.sdf")
    poses = docking.read_poses(sdf)
    result = docking.consensus_rank(
        [("vina", poses)], score_fields=["minimizedAffinity"], mw_max=100.0,
    )
    assert result.n_dropped_filter > 0
    kept = {r["key"] for r in result.rows}
    assert "aspirin" not in kept          # ~180 Da, filtered out
    assert "ethanol" in kept


def test_consensus_rank_cli_writes_csv(tmp_path):
    from molscope.cli import main

    out = tmp_path / "ranking.csv"
    rc = main([
        "dock-rank", POSES_SDF, "--method", "consensus",
        "--score-fields", "minimizedAffinity", "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()
    header = out.read_text().splitlines()[0]
    assert "final_rank" in header and "consensus_rank" in header


# -- feature 1 CLI ----------------------------------------------------------

def test_dock_summary_cli_writes_all_outputs(tmp_path):
    from molscope.cli import main

    rc = main([
        "dock-summary", POSES_SDF, "--score-field", "minimizedAffinity",
        "--out-dir", str(tmp_path), "--top", "2", "--no-smiles",
    ])
    assert rc == 0
    assert (tmp_path / "dock_summary.csv").exists()
    assert (tmp_path / "top_hits.csv").exists()
    assert (tmp_path / "score_distribution.png").exists()
    top = (tmp_path / "top_hits.csv").read_text().splitlines()
    assert len(top) == 1 + 2                          # header + 2 rows


def test_dock_summary_cli_bad_field_errors(tmp_path, capsys):
    from molscope.cli import main

    rc = main(["dock-summary", POSES_SDF, "--score-field", "nope", "--out-dir", str(tmp_path)])
    assert rc == 2
    assert "available fields" in capsys.readouterr().err


# -- feature 4: HTML report -------------------------------------------------

def test_histogram_data_uri_is_inline_png():
    uri = docking.histogram_data_uri(np.array([-8.4, -7.2, -6.9]), "score")
    assert uri.startswith("data:image/png;base64,")


def test_render_html_report_is_self_contained_without_rdkit():
    # Pure assembly: no clustering, no RDKit needed.
    poses = docking.read_poses(POSES_SDF)
    summary = docking.summarize(
        poses, "minimizedAffinity", higher_is_better_flag=False, with_smiles=False,
    )
    html = docking.render_html_report(
        summary, source_name="docking_poses.sdf", n_poses=len(poses), table_rows=10,
    )
    assert "<!DOCTYPE html>" in html
    assert "Docking report" in html
    assert "minimizedAffinity" in html
    assert "ligA_pose1" in html                    # best hit appears in the table
    # No unfilled template placeholders leaked through.
    assert "{title}" not in html and "{table}" not in html


def test_molecule_svg_round_trips_or_degrades():
    poses = docking.read_poses(POSES_SDF)
    svg = docking.molecule_svg(poses[0].molecule)
    # With RDKit we get SVG markup; without it, an empty string (never raises).
    pytest.importorskip("rdkit")
    assert svg.startswith("<svg")


def test_render_html_report_includes_cluster_depictions(tmp_path):
    pytest.importorskip("rdkit")
    sdf = _write_ligand_sdf(tmp_path / "vina.sdf")
    poses = docking.read_poses(sdf)
    summary = docking.summarize(poses, "minimizedAffinity", higher_is_better_flag=False)
    diverse = docking.select_diverse_hits(
        poses, "minimizedAffinity", higher_is_better_flag=False,
        top=8, select=3, threshold=0.5,
    )
    html = docking.render_html_report(
        summary, source_name="vina.sdf", n_poses=len(poses), diverse=diverse,
    )
    assert "Diverse representatives" in html
    assert "<svg" in html                          # 2D depictions embedded
    assert "member(s)" in html                      # cluster size annotation


def test_dock_report_cli_writes_html_and_poses(tmp_path):
    from molscope.cli import main

    rc = main([
        "dock-report", POSES_SDF, "--score-field", "minimizedAffinity",
        "--out-dir", str(tmp_path), "--no-clusters", "--export-poses", "2",
    ])
    assert rc == 0
    report = tmp_path / "dock_report.html"
    assert report.exists()
    assert "<!DOCTYPE html>" in report.read_text()
    assert (tmp_path / "top_poses.sdf").exists()


def test_dock_report_cli_with_clusters(tmp_path):
    pytest.importorskip("rdkit")
    from molscope.cli import main

    sdf = _write_ligand_sdf(tmp_path / "vina.sdf")
    out = tmp_path / "report"
    rc = main([
        "dock-report", sdf, "--score-field", "minimizedAffinity",
        "--out-dir", str(out), "--select", "3", "--threshold", "0.5",
    ])
    assert rc == 0
    assert "Diverse representatives" in (out / "dock_report.html").read_text()


# -- reader edge cases and small helpers ------------------------------------

def test_read_poses_empty_file_errors(tmp_path):
    empty = tmp_path / "empty.sdf"
    empty.write_text("\n\n$$$$\n")
    with pytest.raises(ValueError, match="no readable records"):
        docking.read_poses(str(empty))


def test_read_poses_skips_malformed_and_keeps_unterminated(tmp_path):
    sdf = tmp_path / "messy.sdf"
    sdf.write_text(
        "broken\n p\n\n   X  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n"
        "good\n p\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n>  <score>\n-3.0\n"          # no trailing $$$$
    )
    poses = docking.read_poses(str(sdf))
    assert [p.name for p in poses] == ["good"]
    assert poses[0].score("score") == pytest.approx(-3.0)


def test_resolve_score_field_autodetect_failure(tmp_path):
    sdf = tmp_path / "x.sdf"
    sdf.write_text(
        "m\n p\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n>  <comment>\nhello\n\n$$$$\n"
    )
    poses = docking.read_poses(str(sdf))
    with pytest.raises(ValueError, match="could not auto-detect"):
        docking.resolve_score_field(poses, None)


def test_to_float_and_ligand_efficiency_and_better():
    assert np.isnan(docking._to_float(None))
    assert np.isnan(docking._to_float("not a number"))
    assert docking._to_float(" -7.5 ") == pytest.approx(-7.5)
    assert np.isnan(docking._ligand_efficiency(-8.0, 0, False))   # no heavy atoms
    assert docking._ligand_efficiency(-8.0, 4, False) == pytest.approx(2.0)
    assert docking._ligand_efficiency(0.9, 4, True) == pytest.approx(0.225)
    assert docking._better(0.9, 0.5, True) and not docking._better(0.9, 0.5, False)


# -- graceful degradation when RDKit is unavailable -------------------------

def test_smiles_perceiver_is_none_without_rdkit(monkeypatch):
    monkeypatch.setitem(sys.modules, "rdkit", None)
    assert docking._smiles_perceiver() is None


def test_molecule_svg_empty_without_rdkit(monkeypatch):
    monkeypatch.setitem(sys.modules, "rdkit", None)
    poses = docking.read_poses(POSES_SDF)
    assert docking.molecule_svg(poses[0].molecule) == ""


def test_summarize_leaves_smiles_blank_without_rdkit(monkeypatch):
    monkeypatch.setitem(sys.modules, "rdkit", None)
    poses = docking.read_poses(POSES_SDF)
    result = docking.summarize(poses, "minimizedAffinity", higher_is_better_flag=False)
    assert not result.with_smiles
    assert all(r["smiles"] == "" for r in result.rows)


def test_select_diverse_requires_rdkit(monkeypatch):
    monkeypatch.setitem(sys.modules, "rdkit", None)
    poses = docking.read_poses(POSES_SDF)
    with pytest.raises(ImportError):
        docking.select_diverse_hits(poses, "minimizedAffinity", higher_is_better_flag=False)


def test_consensus_smiles_key_requires_rdkit(monkeypatch):
    monkeypatch.setitem(sys.modules, "rdkit", None)
    poses = docking.read_poses(POSES_SDF)
    with pytest.raises(ValueError, match="needs RDKit"):
        docking.consensus_rank(
            [("f", poses)], score_fields=["minimizedAffinity"], key="smiles",
        )


def test_render_report_shows_no_depiction_placeholder_without_rdkit(monkeypatch, tmp_path):
    pytest.importorskip("rdkit")
    sdf = _write_ligand_sdf(tmp_path / "vina.sdf")
    poses = docking.read_poses(sdf)
    summary = docking.summarize(poses, "minimizedAffinity", higher_is_better_flag=False)
    diverse = docking.select_diverse_hits(
        poses, "minimizedAffinity", higher_is_better_flag=False,
        top=8, select=2, threshold=0.5,
    )
    # Depictions need RDKit at render time; force it absent and check the fallback.
    monkeypatch.setitem(sys.modules, "rdkit", None)
    html = docking.render_html_report(
        summary, source_name="vina.sdf", n_poses=len(poses), diverse=diverse,
    )
    assert "no depiction" in html


def test_pose_stream_is_reusable():
    poses_stream = docking.PoseStream(POSES_SDF)
    first_pass = list(poses_stream)
    second_pass = list(poses_stream)

    assert len(first_pass) == 3
    assert len(second_pass) == 3
    assert first_pass[0].name == second_pass[0].name


def test_summarize_best_pose_per_ligand():
    poses = list(docking.PoseStream(POSES_SDF))
    result = docking.summarize(
        poses, "minimizedAffinity", higher_is_better_flag=False,
        with_smiles=False, best_pose_per_ligand=True,
    )
    assert len(result.rows) == 2
    assert [r["name"] for r in result.rows] == ["ligA_pose1", "ligB_pose1"]
    assert result.rows[0]["rank"] == 1
    assert result.rows[1]["rank"] == 2


def test_select_diverse_hits_surfaces_failed_fingerprints(tmp_path):
    pytest.importorskip("rdkit")
    messy_sdf = tmp_path / "messy.sdf"
    # One good molecule, one bad molecule (overvalent Carbon with 5 bonds will fail valency check)
    messy_sdf.write_text(
        "good\n p\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n>  <score>\n-5.0\n\n$$$$\n"
        "bad\n p\n\n  2  5  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0\n"
        "  1  2  1  0\n"
        "  1  2  1  0\n"
        "  1  2  1  0\n"
        "  1  2  1  0\n"
        "M  END\n>  <score>\n-6.0\n\n$$$$\n"
    )
    poses = list(docking.PoseStream(str(messy_sdf)))
    # We want to select 1 diverse hit from the top 2
    result = docking.select_diverse_hits(
        poses, "score", higher_is_better_flag=False,
        top=2, select=1, threshold=0.5,
    )
    assert result.n_failed_fp == 1
    assert result.n_pool == 1  # 1 valid pose after fingerprinting

