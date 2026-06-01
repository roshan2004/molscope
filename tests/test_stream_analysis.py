"""Tests for the streaming trajectory-lite analyzer (ensemble.analyze_stream)."""

import os

import numpy as np
import pytest

import molscope as ms
from molscope.ensemble import StreamAnalysis, analyze_stream
from molscope.molecule import Molecule

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
NMR = os.path.join(FIXTURES, "1d3z.pdb.gz")  # ubiquitin, 10-model NMR ensemble


def _linear_frame(n, shift, atom_names=None):
    coords = np.array([[float(i) + shift, 0.0, 0.0] for i in range(n)])
    kwargs = {}
    if atom_names:
        kwargs = dict(
            atom_names=atom_names, resnames=["ALA"] * n,
            resids=np.arange(1, n + 1), chains=["A"] * n,
        )
    return Molecule(coords, ["C"] * n, **kwargs)


def test_analyze_stream_from_path():
    a = analyze_stream(NMR)
    assert isinstance(a, StreamAnalysis)
    assert a.n_frames == 10
    assert a.selection == "ca"  # auto picks C-alphas for a protein
    assert a.radius_of_gyration.shape == (10,)
    assert a.rmsd.shape == (10,)
    assert a.rmsd[0] == pytest.approx(0.0, abs=1e-6)  # first frame vs itself
    assert a.rmsd[1:].max() > 0.0
    assert not a.has_secondary_structure  # off by default


def test_analyze_stream_tracks_secondary_structure():
    a = analyze_stream(NMR, secondary_structure=True)
    assert a.has_secondary_structure
    for arr in (a.helix_fraction, a.strand_fraction, a.coil_fraction):
        assert arr.shape == (10,)
        assert np.all((arr >= 0) & (arr <= 1))
    # Fractions partition the residues.
    total = a.helix_fraction + a.strand_fraction + a.coil_fraction
    np.testing.assert_allclose(total, 1.0, atol=1e-6)


def test_analyze_stream_accepts_iterable_of_frames():
    frames = list(ms.stream(NMR))
    a = analyze_stream(iter(frames), selection="all")
    assert a.n_frames == 10
    assert a.selection == "all"
    assert a.n_atoms == len(frames[0])


def test_analyze_stream_is_single_pass_over_a_generator():
    # A one-shot generator can only be consumed once; analyze_stream must not
    # require random access or a second pass.
    gen = ms.stream(NMR)
    a = analyze_stream(gen, selection="ca")
    assert a.n_frames == 10


def test_auto_selection_falls_back_to_all_without_atom_names():
    # Bare coordinate frames (no atom-name metadata, like an XYZ trajectory).
    frames = [_linear_frame(4, 0.0), _linear_frame(4, 0.5)]
    a = analyze_stream(frames)  # selection defaults to "auto"
    assert a.selection == "all"
    assert a.n_frames == 2


def test_secondary_structure_nan_for_non_protein():
    frames = [_linear_frame(4, 0.0), _linear_frame(4, 0.5)]
    a = analyze_stream(frames, secondary_structure=True)
    assert np.all(np.isnan(a.helix_fraction))  # no backbone -> NaN, not a crash


def test_topology_mismatch_raises():
    frames = [_linear_frame(3, 0.0), _linear_frame(2, 0.0)]
    with pytest.raises(ValueError, match="consistent topology"):
        analyze_stream(frames, selection="all")


def test_empty_and_bad_selection_raise():
    with pytest.raises(ValueError, match="no frames"):
        analyze_stream([], selection="all")
    with pytest.raises(ValueError, match="selection must be"):
        analyze_stream([_linear_frame(3, 0.0)], selection="backbone")


def test_summary_dict():
    a = analyze_stream(NMR, secondary_structure=True)
    s = a.summary()
    assert s["n_frames"] == 10
    assert s["selection"] == "ca"
    assert s["rmsd_final"] == pytest.approx(a.rmsd[-1])
    assert s["rg_min"] <= s["rg_mean"] <= s["rg_max"]
    assert "helix_fraction_mean" in s


def test_plot_returns_axes():
    import matplotlib

    matplotlib.use("Agg")
    a = analyze_stream(NMR, secondary_structure=True)
    axes = a.plot(show=False)
    assert len(axes) == 3  # Rg, RMSD, and SS panels
    plain = analyze_stream(NMR).plot(show=False)
    assert len(plain) == 2  # no SS panel
