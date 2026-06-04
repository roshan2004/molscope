"""Smoke-test the end-to-end GNN on-ramp example so the tutorial cannot bit-rot.

Runs the bundled ``examples/pdb_to_pyg_ml.py`` for a handful of epochs and checks
that the build_dataset -> loader -> trained GCN path actually executes and
learns. Skipped unless the optional pyg stack is installed (CI installs it).
"""

import importlib.util
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "pdb_to_pyg_ml.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("pdb_to_pyg_ml", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_on_ramp_example_trains():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    example = _load_example()
    result = example.run(epochs=30, seed=7)

    # 20 NMR conformers, deterministic 70/15/15 split.
    assert result["n_train"] + result["n_val"] + result["n_test"] == 20
    assert result["n_train"] == 14
    # 19 composition features ("ml" preset) + 3 folded coordinates.
    assert result["in_channels"] == 22
    # The loop reduced the (standardised) training loss: it is learning, not idling.
    assert result["last_loss"] < result["first_loss"]
    # A sane radius-of-gyration error in angstroms.
    import math

    assert math.isfinite(result["test_mae"])
    assert 0.0 < result["test_mae"] < 25.0
