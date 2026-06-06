"""Tests for the preset catalogue (`ms.list_presets` / `molscope presets`)."""

import json

import pytest

import molscope as ms
from molscope.cli import main
from molscope.presets import CATEGORIES, format_presets, list_presets


def test_list_presets_returns_every_category():
    presets = list_presets()
    categories = {p.category for p in presets}
    assert categories == set(CATEGORIES)
    # A familiar preset from each top-level category is present.
    names = {p.name for p in presets}
    assert {"native-3d", "ml", "martini"} <= names


def test_list_presets_filters_by_category():
    graph = list_presets("graph")
    assert graph  # non-empty
    assert all(p.category == "graph" for p in graph)
    assert "martini" not in {p.name for p in graph}


def test_list_presets_rejects_unknown_category():
    with pytest.raises(ValueError, match="unknown preset category"):
        list_presets("bogus")


def test_feature_names_match_canonical_functions():
    # The catalogue must echo the canonical *_feature_names functions exactly.
    by_kind = {(p.kind, p.name): p for p in list_presets()}
    assert by_kind[("molecule descriptors", "native-3d")].feature_names == \
        ms.descriptor_feature_names("native-3d")
    assert by_kind[("graph node features", "ml")].feature_names == \
        ms.node_feature_names("ml")
    assert by_kind[("graph edge features", "geom")].feature_names == \
        ms.edge_feature_names("geom")


def test_rdkit_preset_names_listed_without_rdkit():
    # Enumerating rdkit-basic feature names must not require the chem extra.
    rdkit = next(p for p in list_presets("descriptors") if p.name == "rdkit-basic")
    assert rdkit.n_features == len(rdkit.feature_names)
    assert any(name.startswith("rdkit_") for name in rdkit.feature_names)


def test_coarse_grain_presets_are_mappings_without_features():
    from molscope.coarsegrain import COARSE_GRAIN_MAPPINGS

    cg = list_presets("coarse-grain")
    assert {p.name for p in cg} == set(COARSE_GRAIN_MAPPINGS)
    for p in cg:
        assert p.feature_names is None
        assert p.n_features is None


def test_preset_info_to_dict_is_json_serialisable():
    p = list_presets("graph")[0]
    blob = json.dumps(p.to_dict())
    restored = json.loads(blob)
    assert restored["name"] == p.name
    assert restored["category"] == "graph"


def test_format_presets_groups_and_counts():
    text = format_presets(list_presets("graph"))
    assert "graph node features" in text
    assert "[2 features]" in text     # node "default"
    assert "[1 feature]" in text      # edge "default", singular


def test_format_presets_features_flag_lists_names():
    text = format_presets(list_presets("coarse-grain"), show_features=True)
    # Mappings carry no feature names, so nothing is expanded.
    assert "residue_com" in text


def test_format_presets_features_flag_expands_feature_names():
    text = format_presets(list_presets("graph"), show_features=True)
    # A feature preset's column names are printed under it.
    assert "atomic_number" in text


def test_format_presets_handles_empty_list():
    assert format_presets([]) == "(no presets)"


def test_preset_info_repr_distinguishes_features_and_mappings():
    by_name = {p.name: p for p in list_presets()}
    assert "features" in repr(by_name["ml"])       # a feature preset
    assert "mapping" in repr(by_name["martini"])    # a coarse-grain mapping


def test_cli_presets_text(capsys):
    rc = main(["presets"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "native-3d" in out
    assert "bead mapping" in out


def test_cli_presets_one_category(capsys):
    rc = main(["presets", "descriptors"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "native-basic" in out
    assert "martini" not in out  # coarse-grain not shown


def test_cli_presets_json(capsys):
    rc = main(["presets", "graph", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert all(p["category"] == "graph" for p in payload)
    assert any(p["name"] == "ml" for p in payload)


def test_cli_presets_rejects_unknown_category():
    with pytest.raises(SystemExit) as exc:
        main(["presets", "nope"])
    assert exc.value.code == 2
