"""Tests for cross-sectional area profiles (molscope.crosssection)."""

import numpy as np
import pytest

import molscope as ms
from molscope import Molecule
from molscope.crosssection import (
    CrossSectionProfile,
    _convex_hull_area,
    cross_section_profile,
)


def _ring(radius, z, n=400):
    """n points on a circle of given radius at height z (xy-plane ring)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z)])


def cylinder(radius=10.0, height=40.0, n_rings=21, n_per=120):
    """A hollow cylinder of point rings stacked along z.

    ``n_per=120`` keeps each ring's convex hull within ~0.1% of the true circle
    while staying small enough that the descriptor-integration tests (which run an
    O(n^2) distance histogram and SASA) finish quickly.
    """
    zs = np.linspace(0.0, height, n_rings)
    coords = np.vstack([_ring(radius, z, n_per) for z in zs])
    return Molecule(coords, ["C"] * len(coords), name="cylinder")


# --- convex-hull area primitive -----------------------------------------------

def test_convex_hull_area_of_unit_square():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=float)
    assert _convex_hull_area(square) == pytest.approx(1.0)


def test_convex_hull_area_degenerate_is_zero():
    assert _convex_hull_area(np.array([[0.0, 0.0], [1.0, 1.0]])) == 0.0  # 2 points
    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert _convex_hull_area(collinear) == 0.0


def test_convex_hull_area_matches_circle():
    theta = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    pts = np.column_stack([5 * np.cos(theta), 5 * np.sin(theta)])
    assert _convex_hull_area(pts) == pytest.approx(np.pi * 25, rel=1e-3)


# --- profile geometry ----------------------------------------------------------

def test_cylinder_profile_is_flat_and_correct_area():
    prof = cross_section_profile(cylinder(radius=10.0), axis="z", thickness=2.0)
    # Each slice is a ring of radius 10 -> area ~ pi r^2 = 314.16.
    assert prof.max == pytest.approx(np.pi * 100, rel=1e-2)
    assert prof.mean == pytest.approx(np.pi * 100, rel=1e-2)
    # A cylinder has a near-constant profile.
    assert prof.std < 0.05 * prof.mean
    assert prof.length == pytest.approx(40.0, abs=2.0)


def test_cone_profile_ramps_along_axis():
    # Radius grows linearly with z -> area grows quadratically; the first slice is
    # narrow and the last is wide.
    zs = np.linspace(0.0, 40.0, 41)
    coords = np.vstack([_ring(0.25 * z + 0.5, z) for z in zs])
    prof = cross_section_profile(
        Molecule(coords, ["C"] * len(coords)), axis="z", thickness=2.0
    )
    assert prof.areas[0] < prof.areas[-1]
    # monotone non-decreasing apart from discretisation noise
    assert np.all(np.diff(prof.areas) > -1.0)
    assert prof.max > prof.min


def test_default_axis_is_long_principal_axis():
    # A box elongated along x: the long principal axis should be x, so slicing the
    # default way spans ~the x-extent (60), not the y/z extent (10).
    rng = np.random.default_rng(0)
    coords = rng.uniform([-30, -5, -5], [30, 5, 5], size=(3000, 3))
    prof = Molecule(coords, ["C"] * len(coords)).cross_section_profile(thickness=2.0)
    assert prof.length == pytest.approx(60.0, abs=4.0)
    # the chosen axis is (anti)parallel to x
    assert abs(prof.axis[0]) > 0.98


def test_explicit_vector_axis_is_normalised():
    prof = cross_section_profile(cylinder(), axis=[0.0, 0.0, 5.0], thickness=2.0)
    np.testing.assert_allclose(prof.axis, [0.0, 0.0, 1.0])


# --- reductions / dataclass ----------------------------------------------------

def test_summary_ordering_and_keys():
    prof = cross_section_profile(cylinder(), axis="z", thickness=2.0)
    s = prof.summary()
    assert set(s) == {
        "cross_section_max",
        "cross_section_mean",
        "cross_section_min",
        "cross_section_std",
    }
    assert s["cross_section_max"] >= s["cross_section_mean"] >= s["cross_section_min"]


def test_min_ignores_empty_end_slices():
    # Reductions are over occupied slices, so a tapering structure does not report
    # a trivial min of zero.
    prof = cross_section_profile(cylinder(), axis="z", thickness=2.0)
    assert prof.min > 0.0


def test_to_dict_roundtrip_shapes():
    prof = cross_section_profile(cylinder(), axis="z", thickness=2.0)
    d = prof.to_dict()
    assert len(d["positions"]) == len(d["areas"]) == len(prof.areas)
    assert d["method"] == "hull"
    assert len(d["axis"]) == 3


# --- guards --------------------------------------------------------------------

def test_too_few_atoms_gives_empty_profile():
    prof = cross_section_profile(Molecule(np.zeros((2, 3)), ["O", "H"]))
    assert prof.areas.size == 0
    assert prof.summary() == {
        "cross_section_max": 0.0,
        "cross_section_mean": 0.0,
        "cross_section_min": 0.0,
        "cross_section_std": 0.0,
    }


def test_invalid_thickness_and_method_and_axis():
    mol = cylinder()
    with pytest.raises(ValueError, match="thickness"):
        cross_section_profile(mol, thickness=0.0)
    with pytest.raises(ValueError, match="method"):
        cross_section_profile(mol, method="alpha")
    with pytest.raises(ValueError, match="axis"):
        cross_section_profile(mol, axis="diagonal")
    with pytest.raises(ValueError, match="length 3"):
        cross_section_profile(mol, axis=[1.0, 0.0])


# --- voronoi method ------------------------------------------------------------

def test_voronoi_requires_environment_atoms():
    pytest.importorskip("scipy")
    # No hetero atoms -> voronoi would equal hull, so it refuses and points to hull.
    with pytest.raises(ValueError, match="environment"):
        cross_section_profile(cylinder(), axis="z", method="voronoi")


def _filled_disk(radius, z, n, rng):
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), np.full(n, z)])


def test_voronoi_runs_with_environment_and_stays_finite():
    pytest.importorskip("scipy")
    # A solid protein disk (r=8) hugged by an environment annulus (r in [9, 14])
    # per slice -- the lipid analogue that bounds the protein's outer cells.
    rng = np.random.default_rng(0)
    zs = np.linspace(0.0, 20.0, 11)
    prot = np.vstack([_filled_disk(8.0, z, 600, rng) for z in zs])
    env_pts = []
    for z in zs:
        ann = _filled_disk(14.0, z, 1500, rng)
        ann = ann[np.hypot(ann[:, 0], ann[:, 1]) >= 9.0]
        env_pts.append(ann)
    env = np.vstack(env_pts)
    coords = np.vstack([prot, env])
    hetero = [False] * len(prot) + [True] * len(env)
    mol = Molecule(coords, ["C"] * len(coords), hetero=hetero)

    prof = mol.cross_section_profile(axis="z", thickness=2.0, method="voronoi")
    assert prof.method == "voronoi"
    assert np.all(np.isfinite(prof.areas))

    # Correctness bracket: the protein cell sum sits between the protein's own
    # convex-hull area (~pi*64=201, cells reach slightly past the points) and the
    # full combined hull (~pi*196=616, the env outer edge).
    assert prof.max == pytest.approx(np.pi * 64, rel=0.4)
    assert np.pi * 64 * 0.9 < prof.max < np.pi * 196


# --- descriptor integration ----------------------------------------------------

def test_descriptors_expose_cross_section_scalars():
    mol = cylinder()
    desc = ms.descriptors(mol)
    for key in (
        "cross_section_max",
        "cross_section_mean",
        "cross_section_min",
        "cross_section_std",
    ):
        assert key in desc
    assert desc["cross_section_max"] >= desc["cross_section_min"] > 0.0


def test_cross_section_only_in_native_3d_preset():
    mol = cylinder()
    assert "cross_section_max" in ms.descriptors(mol, preset="native-3d")
    assert "cross_section_max" not in ms.descriptors(mol, preset="native-basic")


def test_empty_molecule_descriptors_have_zero_cross_section():
    desc = ms.descriptors(Molecule(np.empty((0, 3)), []))
    assert desc["cross_section_max"] == 0.0
    assert desc["cross_section_std"] == 0.0


def test_profile_is_a_dataclass_instance():
    assert isinstance(cross_section_profile(cylinder(), axis="z"), CrossSectionProfile)
