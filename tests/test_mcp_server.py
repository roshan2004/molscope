"""Tests for the optional MCP server (``molscope.mcp_server``).

The whole module is skipped when the ``mcp`` extra is not installed (for example
on Python 3.9, where the reference SDK is unavailable). Tools are exercised
through the real FastMCP ``call_tool`` surface so the registered schemas and the
adapter logic are both covered. Every case uses bundled local structures, so no
network access is needed.
"""

import asyncio
import json
import os

import pytest

pytest.importorskip("mcp")

from molscope.mcp_server import _load, build_server  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")
UBQ = os.path.join(DATA, "1ubq.pdb")
TRYPSIN = os.path.join(DATA, "3ptb.pdb")


@pytest.fixture(scope="module")
def server():
    return build_server()


def _content(server, name, args):
    """Return the content blocks from a tool call.

    Text/JSON tools return ``(content, structured)``; image tools (no structured
    output) return the bare content list. Handle both shapes.
    """
    result = asyncio.run(server.call_tool(name, args))
    return result[0] if isinstance(result, tuple) else result


def _text(server, name, **args):
    return _content(server, name, args)[0].text


def _json(server, name, **args):
    return json.loads(_text(server, name, **args))


def _image(server, name, **args):
    return _content(server, name, args)[0]


EXPECTED_TOOLS = {
    "summarize_structure",
    "compute_descriptors",
    "secondary_structure",
    "contact_map",
    "binding_site",
    "molecular_graph",
    "coarse_grain",
    "geometry",
    "measure",
    "rmsd",
    "list_ligands",
    "chain_interfaces",
    "backbone_torsions",
    "ensemble_summary",
    "chemical_features",
    "validate_cif",
    "prepare_structure",
    "select_diverse",
    "prepare_dataset",
    "find_duplicates",
    "dock_summary",
    "dock_diverse",
    "dock_rank",
    "dock_report",
    "render_structure",
    "render_contact_map",
    "render_distance_matrix",
    "render_rmsd_heatmap",
    "render_cross_correlation",
}

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
ENSEMBLE = os.path.join(DATA, "1aml.pdb")  # 20-model NMR ensemble
TWO_CHAIN = os.path.join(FIXTURES, "ugly_residue_ids.pdb")  # chains A and B
DOCK_SDF = os.path.join(FIXTURES, "docking_poses.sdf")  # 3 poses, minimizedAffinity


def _write_ligand_sdf(path, score_field="minimizedAffinity"):
    """A few distinct scaffolds with docking scores, for clustering/ranking tests."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    ligands = [
        ("ethanol", "CCO", -5.1), ("propanol", "CCCO", -5.0),
        ("benzene", "c1ccccc1", -7.2), ("toluene", "Cc1ccccc1", -7.4),
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O", -8.9),
        ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", -8.1),
    ]
    writer = Chem.SDWriter(str(path))
    for name, smi, vina in ligands:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        mol.SetProp("_Name", name)
        mol.SetProp(score_field, f"{vina:.2f}")
        writer.write(mol)
    writer.close()
    return str(path)


def test_server_registers_expected_tools(server):
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == EXPECTED_TOOLS


def test_summarize_structure(server):
    out = _text(server, "summarize_structure", source=UBQ)
    assert "atoms" in out and "chains A" in out


def test_compute_descriptors_batch(server):
    out = _json(server, "compute_descriptors", sources=[UBQ, TRYPSIN])
    assert out["n_features"] == len(out["feature_names"]) > 0
    assert [row["source"] for row in out["rows"]] == [UBQ, TRYPSIN]
    # Each row carries one value per named feature.
    for row in out["rows"]:
        assert len(row["values"]) == out["n_features"]


def test_secondary_structure(server):
    out = _json(server, "secondary_structure", source=UBQ)
    comp = out["composition"]
    assert out["n_residues"] == 76
    assert comp["helix"] + comp["strand"] + comp["coil"] == out["n_residues"]
    assert {"chain", "resid", "resname", "code"} <= set(out["residues"][0])


def test_contact_map(server):
    out = _json(server, "contact_map", source=UBQ, cutoff=8.0)
    assert out["level"] == "residue"
    assert out["n_contacts"] > 0
    assert out["n_contacts"] == len(out["pairs"]) or out["pairs_truncated"]


def test_binding_site_auto_ligand(server):
    out = _json(server, "binding_site", source=TRYPSIN)
    assert out["n_residues"] > 0
    first = out["residues"][0]
    assert first["min_distance"] <= out["cutoff"]


def test_molecular_graph(server):
    out = _json(server, "molecular_graph", source=UBQ)
    assert out["n_nodes"] == 660
    assert out["n_edges"] > 0
    assert out["node_feature_matrix_shape"][0] == out["n_nodes"]
    assert out["node_feature_names"] and out["edge_feature_names"]


def test_coarse_grain(server):
    out = _json(server, "coarse_grain", source=UBQ, mapping="residue_com")
    assert out["mapping"] == "residue_com"
    # One bead per residue: 76 protein residues plus crystallographic waters.
    assert out["n_beads"] >= 76
    assert out["n_dropped_atoms"] == 0
    assert out["n_bonds"] == out["n_beads"] - 1


def test_render_structure_returns_png(server):
    block = _image(server, "render_structure", source=UBQ, color_by="chain")
    assert getattr(block, "mimeType", None) == "image/png"
    assert block.data  # base64 payload present


def test_render_contact_map_returns_png(server):
    block = _image(server, "render_contact_map", source=UBQ)
    assert getattr(block, "mimeType", None) == "image/png"
    assert block.data


def test_geometry(server):
    out = _json(server, "geometry", source=UBQ)
    assert out["n_atoms"] == 660
    assert out["radius_of_gyration"] > 0
    assert len(out["center_of_mass"]) == 3
    assert len(out["principal_moments"]) == 3


def test_measure_distance_angle_dihedral(server):
    assert _json(server, "measure", source=UBQ, atoms=[0, 1])["kind"] == "distance"
    assert _json(server, "measure", source=UBQ, atoms=[0, 1, 2])["kind"] == "angle"
    dih = _json(server, "measure", source=UBQ, atoms=[0, 1, 2, 3])
    assert dih["kind"] == "dihedral" and dih["value"] is not None


def test_measure_rejects_bad_atom_count(server):
    with pytest.raises(Exception):  # noqa: B017 - any error surfaces as a tool failure
        _text(server, "measure", source=UBQ, atoms=[0])


def test_rmsd_self_is_zero(server):
    out = _json(server, "rmsd", source_a=UBQ, source_b=UBQ)
    assert out["rmsd"] is not None and out["rmsd"] < 1e-6


def test_list_ligands(server):
    out = _json(server, "list_ligands", source=TRYPSIN)
    assert out["n_ligands"] == 1
    assert out["ligands"][0]["resname"] == "BEN"


def test_chain_interfaces_matrix_path(server):
    out = _json(server, "chain_interfaces", source=UBQ)  # single chain -> matrix
    assert out["chains"] == ["A"]
    assert out["contact_matrix"] == [[0]]


def test_chain_interfaces_pair_path(server):
    out = _json(server, "chain_interfaces", source=TWO_CHAIN,
                chain_a="A", chain_b="B", cutoff=50.0)
    assert out["chain_a"] == "A" and out["chain_b"] == "B"
    assert "residues_a" in out and "residues_b" in out


def test_backbone_torsions(server):
    out = _json(server, "backbone_torsions", source=UBQ)
    assert out["n_residues"] == 76
    first = out["residues"][0]
    assert first["phi"] is None  # undefined at a chain start -> JSON null, not NaN
    assert first["psi"] is not None


def test_ensemble_summary(server):
    out = _json(server, "ensemble_summary", source=ENSEMBLE)
    assert out["n_models"] == 20
    assert out["mean_pairwise_rmsd"] > 0
    assert out["n_clusters"] >= 1


def test_ensemble_summary_rejects_single_model(server):
    with pytest.raises(Exception):  # noqa: B017
        _text(server, "ensemble_summary", source=UBQ)


def test_chemical_features(server):
    pytest.importorskip("rdkit")
    # Default bond_perception="template" recovers aromatic rings on a protein PDB.
    out = _json(server, "chemical_features", source=UBQ)
    assert out["n_atoms"] == 660
    assert out["n_bonds"] > 0
    assert out["n_aromatic_atoms"] >= 20  # Phe/Tyr/His rings via residue templates


def test_chemical_features_geometric_has_no_aromaticity(server):
    pytest.importorskip("rdkit")
    out = _json(server, "chemical_features", source=UBQ, bond_perception="geometric")
    assert out["n_aromatic_atoms"] == 0  # geometric single bonds, no perception


def test_chemical_features_standard_protonation_is_meaningful(server):
    pytest.importorskip("rdkit")
    # Trypsin (3ptb) is basic: standard protonation should give a positive net charge.
    out = _json(server, "chemical_features", source=TRYPSIN)
    assert out["total_formal_charge"] > 0
    assert out["protonation"].startswith("standard")


def test_chemical_features_protonation_none_is_neutral_and_labelled(server):
    pytest.importorskip("rdkit")
    out = _json(server, "chemical_features", source=TRYPSIN, protonation="none")
    assert out["total_formal_charge"] == 0
    assert out["protonation"].startswith("as-modelled")


def test_chemical_features_non_pdb_falls_back_from_template(server):
    pytest.importorskip("rdkit")
    # An SDF carries explicit bonds; template perception is PDB-only, so the
    # default "template" must fall back gracefully rather than erroring.
    water = os.path.join(FIXTURES, "water.sdf")
    out = _json(server, "chemical_features", source=water)
    assert out["n_atoms"] == 3 and out["n_bonds"] == 2


def test_validate_cif(server):
    pytest.importorskip("gemmi")
    out = _json(server, "validate_cif", source=os.path.join(FIXTURES, "insertion_codes.cif"))
    assert "valid" in out and "n_atom_site_rows" in out


def test_select_diverse_on_descriptor_cols(server, tmp_path):
    import csv

    path = tmp_path / "lib.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "MW", "ALogP"])
        writer.writeheader()
        for i in range(6):
            writer.writerow({"ID": f"m{i}", "MW": 100 + i * 60, "ALogP": i * 0.9})
    out = _json(server, "select_diverse", table=str(path), n=3, descriptor_cols=["MW", "ALogP"])
    assert out["selected"] == 3 and out["of"] == 6


def test_select_diverse_requires_a_descriptor_source(server, tmp_path):
    import csv

    path = tmp_path / "lib.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "MW"])
        writer.writeheader()
        writer.writerow({"ID": "m0", "MW": 100})
    with pytest.raises(Exception):  # noqa: B017 - neither descriptor_cols nor compute given
        _text(server, "select_diverse", table=str(path), n=1)


def test_select_diverse_from_smiles(server, tmp_path):
    pytest.importorskip("rdkit")
    import csv

    path = tmp_path / "smi.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "SMILES"])
        writer.writeheader()
        for name, smi in [("ethanol", "CCO"), ("benzene", "c1ccccc1"),
                          ("acetic", "CC(=O)O"), ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")]:
            writer.writerow({"ID": name, "SMILES": smi})
    out = _json(server, "select_diverse", table=str(path), n=2,
                smiles_col="SMILES", compute_descriptors=True)
    assert out["selected"] == 2
    assert "MolWt" in out["descriptors"]


def test_select_diverse_compute_without_smiles_col_errors(server, tmp_path):
    import csv

    path = tmp_path / "lib.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "MW"])
        writer.writeheader()
        writer.writerow({"ID": "m0", "MW": 100})
    with pytest.raises(Exception):  # noqa: B017 - compute_descriptors set but no smiles_col
        _text(server, "select_diverse", table=str(path), n=1, compute_descriptors=True)


def _write_lib(path, n=20):
    import csv

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "MW", "ALogP"])
        writer.writeheader()
        for i in range(n):
            writer.writerow({"ID": f"m{i}", "MW": 100 + i * 40, "ALogP": i * 0.3})
    return str(path)


def test_select_diverse_by_fraction(server, tmp_path):
    path = _write_lib(tmp_path / "lib.csv", n=20)
    out = _json(server, "select_diverse", table=path, fraction=0.05,
                descriptor_cols=["MW", "ALogP"])
    assert out["selected"] == 1 and out["requested"] == 1  # ceil(0.05 * 20)


def test_select_diverse_rejects_both_n_and_fraction(server, tmp_path):
    path = _write_lib(tmp_path / "lib.csv", n=5)
    with pytest.raises(Exception):  # noqa: B017 - exactly one of n/fraction
        _text(server, "select_diverse", table=path, n=2, fraction=0.5,
              descriptor_cols=["MW", "ALogP"])


def test_prepare_dataset_returns_assignments(server, tmp_path):
    path = _write_lib(tmp_path / "lib.csv", n=20)
    out = _json(server, "prepare_dataset", table=path, split="diversity",
                descriptor_cols=["MW", "ALogP"], test=0.2, val=0.2)
    assert out["split_method"] == "diversity"
    assert out["split_sizes"]["test"] == 4
    assert len(out["assignments"]) == 20
    assert {a["split"] for a in out["assignments"]} == {"train", "validation", "test"}
    assert "written" not in out  # no save_dir given


def test_prepare_dataset_writes_files_with_save_dir(server, tmp_path):
    path = _write_lib(tmp_path / "lib.csv", n=20)
    save = tmp_path / "out"
    out = _json(server, "prepare_dataset", table=path, split="random",
                save_dir=str(save))
    assert any(p.endswith("train.csv") for p in out["written"])
    assert (save / "manifest.json").exists()


def test_find_duplicates_canonical(server, tmp_path):
    pytest.importorskip("rdkit")
    import csv

    path = tmp_path / "smi.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "SMILES"])
        writer.writeheader()
        for name, smi in [("a", "CCO"), ("b", "OCC"), ("c", "c1ccccc1")]:
            writer.writerow({"ID": name, "SMILES": smi})
    out = _json(server, "find_duplicates", table=str(path), smiles_col="SMILES")
    assert out["n_redundant_rows"] == 1
    assert out["n_duplicate_groups"] == 1
    assert sorted(out["groups"][0]["ids"]) == ["a", "b"]  # CCO and OCC are one molecule


def test_render_rmsd_heatmap_rejects_single_model(server):
    with pytest.raises(Exception):  # noqa: B017
        _image(server, "render_rmsd_heatmap", source=UBQ)


def test_render_distance_matrix_returns_png(server):
    block = _image(server, "render_distance_matrix", source=UBQ)
    assert getattr(block, "mimeType", None) == "image/png" and block.data


def test_render_rmsd_heatmap_returns_png(server):
    block = _image(server, "render_rmsd_heatmap", source=ENSEMBLE)
    assert getattr(block, "mimeType", None) == "image/png" and block.data


def test_render_cross_correlation_returns_png(server):
    block = _image(server, "render_cross_correlation", source=ENSEMBLE)
    assert getattr(block, "mimeType", None) == "image/png" and block.data


def test_render_cross_correlation_rejects_single_model(server):
    with pytest.raises(Exception):  # noqa: B017
        _image(server, "render_cross_correlation", source=UBQ)


def test_render_contact_map_saves_file(server, tmp_path):
    out = tmp_path / "cmap.png"
    msg = _text(server, "render_contact_map", source=TRYPSIN, save_path=str(out))
    assert str(out) in msg and out.exists() and out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_structure_saves_file(server, tmp_path):
    out = tmp_path / "view.png"
    msg = _text(server, "render_structure", source=UBQ, save_path=str(out))
    assert "Saved figure to" in msg and out.exists()


def test_render_save_path_format_follows_extension(server, tmp_path):
    out = tmp_path / "cmap.pdf"
    _text(server, "render_contact_map", source=TRYPSIN, save_path=str(out))
    assert out.exists() and out.read_bytes()[:5] == b"%PDF-"


def test_render_save_path_creates_parent_dirs(server, tmp_path):
    out = tmp_path / "nested" / "dir" / "dm.png"
    _text(server, "render_distance_matrix", source=UBQ, save_path=str(out))
    assert out.exists()


def test_load_accepts_paths_and_rejects_garbage():
    assert _load(UBQ).summary()
    with pytest.raises(FileNotFoundError):
        _load("not-a-file-or-id.zzz")


def test_load_accepts_smiles_prefix():
    pytest.importorskip("rdkit")
    mol = _load("smiles:CCO")  # ethanol
    assert len(mol) == 9  # C2H6O with explicit hydrogens


def test_load_rejects_empty_smiles():
    with pytest.raises(ValueError, match="empty SMILES"):
        _load("smiles:")


def test_smiles_source_works_through_a_tool(server):
    pytest.importorskip("rdkit")
    out = _text(server, "summarize_structure", source="smiles:CCO")
    assert "C2 H6 O" in out


def test_compute_descriptors_accepts_smiles(server):
    pytest.importorskip("rdkit")
    out = _json(server, "compute_descriptors", sources=["smiles:CCO", UBQ])
    assert [row["source"] for row in out["rows"]] == ["smiles:CCO", UBQ]


def test_load_rejection_message_mentions_smiles():
    with pytest.raises(FileNotFoundError, match="smiles:"):
        _load("definitely not a structure")


def test_dependency_error_maps_bare_module_to_extra():
    from molscope.mcp_server import _dependency_error

    err = _dependency_error(ModuleNotFoundError("No module named 'rdkit'", name="rdkit"))
    assert 'molscope[chem]' in str(err)
    err = _dependency_error(ModuleNotFoundError("No module named 'gemmi'", name="gemmi"))
    assert 'molscope[cif]' in str(err)


def test_dependency_error_passes_through_existing_guidance():
    from molscope.mcp_server import _dependency_error

    original = ImportError('needs RDKit; pip install "molscope[chem]"')
    # Library messages already name the extra, so they must be returned unchanged.
    assert _dependency_error(original) is original


def test_tools_carry_titles_and_annotations(server):
    tools = asyncio.run(server.list_tools())
    for tool in tools:
        assert tool.title, f"{tool.name} has no title"
        assert tool.annotations is not None, f"{tool.name} has no annotations"
    by_name = {t.name: t for t in tools}
    # Pure analysis tools are read-only; render/prepare tools may write files.
    assert by_name["summarize_structure"].annotations.readOnlyHint is True
    assert by_name["render_structure"].annotations.readOnlyHint is False
    assert by_name["prepare_dataset"].annotations.readOnlyHint is False
    # Tools that take a structure source may fetch from RCSB (open world);
    # table-only tools do not.
    assert by_name["summarize_structure"].annotations.openWorldHint is True
    assert by_name["select_diverse"].annotations.openWorldHint is False


def test_load_dispatches_pdb_id_to_fetch(monkeypatch):
    import molscope.io as mio

    sentinel = object()
    monkeypatch.setattr(mio, "fetch", lambda pdb_id, **kwargs: sentinel)
    assert _load("1abc") is sentinel


def test_jsonable_coerces_numpy():
    import numpy as np

    from molscope.mcp_server import _jsonable

    assert _jsonable(np.float64(3.5)) == 3.5
    assert _jsonable(np.array([1, 2])) == [1, 2]
    assert _jsonable("x") == "x"


def test_main_without_mcp_raises_systemexit(monkeypatch):
    import molscope.mcp_server as srv

    def boom():
        raise ImportError("mcp not installed")

    monkeypatch.setattr(srv, "build_server", boom)
    with pytest.raises(SystemExit):
        srv.main()


def test_main_runs_server(monkeypatch):
    import molscope.mcp_server as srv

    calls = {}

    class _FakeServer:
        def run(self):
            calls["ran"] = True

    monkeypatch.setattr(srv, "build_server", lambda: _FakeServer())
    srv.main()
    assert calls.get("ran") is True


# -- docking tools ----------------------------------------------------------

def test_dock_summary_ranks_and_autodetects(server):
    out = _json(
        server, "dock_summary", source=DOCK_SDF, with_smiles=False, best_pose_per_ligand=False,
    )
    assert out["score_field"] == "minimizedAffinity"
    assert out["direction"] == "lower_is_better"
    assert out["n_poses"] == 3
    assert [r["name"] for r in out["rows"]] == ["ligA_pose1", "ligB_pose1", "ligA_pose2"]
    assert out["rows"][0]["rank"] == 1


def test_dock_summary_best_pose_per_ligand(server):
    # Default is best_pose_per_ligand=True: the two ligA poses collapse to one.
    out = _json(server, "dock_summary", source=DOCK_SDF, with_smiles=False)
    assert out["n_poses"] == 3                       # total poses read (unchanged)
    assert out["n_ranked"] == 2                       # one row per compound after collapse
    assert [r["name"] for r in out["rows"]] == ["ligA_pose1", "ligB_pose1"]


def test_dock_summary_writes_files(server, tmp_path):
    out = _json(server, "dock_summary", source=DOCK_SDF, top=2, save_dir=str(tmp_path))
    written = {os.path.basename(p) for p in out["written"]}
    assert {"dock_summary.csv", "top_hits.csv", "score_distribution.png"} <= written
    assert (tmp_path / "dock_summary.csv").exists()


def test_dock_summary_unknown_field_errors(server):
    with pytest.raises(Exception, match="available fields"):
        _json(server, "dock_summary", source=DOCK_SDF, score_field="nope")


def test_dock_rank_consensus_across_files(server):
    out = _json(
        server, "dock_rank", sources=[DOCK_SDF, DOCK_SDF],
        score_fields=["minimizedAffinity"],
    )
    assert out["key"] == "name"
    assert out["rows"][0]["key"] == "ligA_pose1"
    assert out["rows"][0]["final_rank"] == 1
    assert "triage heuristic" in out["note"]


def test_dock_diverse_selects_representatives(server, tmp_path):
    pytest.importorskip("rdkit")
    sdf = _write_ligand_sdf(tmp_path / "vina.sdf")
    out = _json(
        server, "dock_diverse", source=sdf, score_field="minimizedAffinity",
        top=6, select=3, threshold=0.5, save_dir=str(tmp_path / "out"),
    )
    assert len(out["selected"]) == 3
    assert out["selected"][0]["name"] == "aspirin"        # best score leads
    assert {"cluster_id", "cluster_size"} <= set(out["selected"][0])
    assert (tmp_path / "out" / "diverse_hits.sdf").exists()


def test_dock_report_writes_html_and_poses(server, tmp_path):
    out = _json(
        server, "dock_report", source=DOCK_SDF, save_dir=str(tmp_path),
        clusters=False, export_poses=2,
    )
    written = {os.path.basename(p) for p in out["written"]}
    assert "dock_report.html" in written and "top_poses.sdf" in written
    assert "<!DOCTYPE html>" in (tmp_path / "dock_report.html").read_text()


def test_dock_summary_higher_is_better_explicit(server):
    out = _json(
        server, "dock_summary", source=DOCK_SDF, score_field="CNNscore",
        higher_is_better=True, with_smiles=False,
    )
    assert out["direction"] == "higher_is_better"
    assert out["rows"][0]["score"] >= out["rows"][-1]["score"]


def test_dock_diverse_without_save_dir(server, tmp_path):
    pytest.importorskip("rdkit")
    sdf = _write_ligand_sdf(tmp_path / "vina.sdf")
    out = _json(
        server, "dock_diverse", source=sdf, score_field="minimizedAffinity",
        top=6, select=2, threshold=0.5,
    )
    assert len(out["selected"]) == 2
    assert "written" not in out                       # no files when save_dir omitted


def test_dock_rank_writes_csv_with_overrides(server, tmp_path):
    out_csv = tmp_path / "ranking.csv"
    out = _json(
        server, "dock_rank", sources=[DOCK_SDF], score_fields=["minimizedAffinity"],
        lower_is_better=["minimizedAffinity"], save_path=str(out_csv),
    )
    assert out["directions"]["minimizedAffinity"] == "lower_is_better"
    assert out["assumed_direction"] == []
    assert out_csv.exists()
    assert "final_rank" in out_csv.read_text().splitlines()[0]


def test_dock_report_with_clusters(server, tmp_path):
    pytest.importorskip("rdkit")
    sdf = _write_ligand_sdf(tmp_path / "vina.sdf")
    out = _json(
        server, "dock_report", source=sdf, save_dir=str(tmp_path / "rep"),
        score_field="minimizedAffinity", top=6, select=2, threshold=0.5,
    )
    assert out["n_clusters"] is not None and out["n_clusters"] >= 1
    html = (tmp_path / "rep" / "dock_report.html").read_text()
    assert "Diverse representatives" in html


def test_dock_tools_reject_missing_file(server):
    with pytest.raises(Exception, match="no such SDF file"):
        _json(server, "dock_summary", source="/no/such/file.sdf")
