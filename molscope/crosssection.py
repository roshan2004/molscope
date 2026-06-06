"""Cross-sectional area profiles along a molecular axis.

Slice a structure into thin bands perpendicular to an axis and measure the area
of each band — the "how wide is it at each point along its length" profile. This
is the static-structure analogue of the membrane-protein cross-section profile in
``Becksteinlab/Protein_Area`` (Voronoi-cell areas per z-slice of an MD
trajectory). Two area methods are offered:

* ``"hull"`` (default, pure NumPy): the convex-hull area of the atoms projected
  into each slice. The *outer* cross-section; available for any structure with no
  extra dependency.
* ``"voronoi"`` (needs SciPy): the Beckstein-style sum of per-atom Voronoi cells,
  where surrounding non-protein (``hetero``) atoms bound the protein's outer
  cells. This measures the area the protein *occupies* against its environment
  and only differs from ``"hull"`` when such environment atoms are present.

The reduced scalars (``cross_section_max/mean/min/std``) feed the ML descriptor
table; the full profile is available for plotting and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

# Coarser default slice for the descriptor table than a user might pick
# interactively: a few-angstrom band keeps every interior slice well populated so
# the reduced scalars are stable, and keeps the per-structure cost negligible in
# batch featurisation.
DESCRIPTOR_SLICE_THICKNESS = 2.0

AxisSpec = Union[str, np.ndarray, "list[float]", "tuple[float, float, float]"]


@dataclass
class CrossSectionProfile:
    """A cross-sectional area profile sampled along an axis.

    Attributes
    ----------
    positions:
        Slice-centre coordinates along ``axis`` (angstrom), ascending.
    areas:
        Cross-sectional area of each slice (angstrom^2), aligned with
        ``positions``. Slices with fewer than three atoms have area ``0``.
    axis:
        The unit vector the structure was sliced along.
    thickness:
        Slice thickness (angstrom).
    method:
        ``"hull"`` or ``"voronoi"``.
    """

    positions: np.ndarray
    areas: np.ndarray
    axis: np.ndarray
    thickness: float
    method: str

    @property
    def _occupied(self) -> np.ndarray:
        """Areas of the slices that actually carry a cross-section (area > 0)."""
        return self.areas[self.areas > 0.0]

    @property
    def max(self) -> float:
        """Widest cross-section (angstrom^2); ``0`` for an empty profile."""
        occ = self._occupied
        return float(occ.max()) if occ.size else 0.0

    @property
    def min(self) -> float:
        """Narrowest occupied cross-section (angstrom^2) — e.g. a constriction.

        Taken over slices that contain a cross-section so the tapering ends (which
        fall to zero) do not make this trivially ``0``.
        """
        occ = self._occupied
        return float(occ.min()) if occ.size else 0.0

    @property
    def mean(self) -> float:
        """Mean cross-section over occupied slices (angstrom^2)."""
        occ = self._occupied
        return float(occ.mean()) if occ.size else 0.0

    @property
    def std(self) -> float:
        """Standard deviation of the cross-section over occupied slices."""
        occ = self._occupied
        return float(occ.std()) if occ.size else 0.0

    @property
    def length(self) -> float:
        """Axial extent spanned by the profile (angstrom)."""
        if self.positions.size < 2:
            return 0.0
        return float(self.positions[-1] - self.positions[0] + self.thickness)

    def summary(self) -> dict[str, float]:
        """The reduced scalars used as ML features."""
        return {
            "cross_section_max": self.max,
            "cross_section_mean": self.mean,
            "cross_section_min": self.min,
            "cross_section_std": self.std,
        }

    def to_dict(self) -> dict:
        """A JSON-friendly view of the whole profile."""
        return {
            "axis": self.axis.astype(float).tolist(),
            "thickness": float(self.thickness),
            "method": self.method,
            "positions": self.positions.astype(float).tolist(),
            "areas": self.areas.astype(float).tolist(),
            **self.summary(),
        }


def cross_section_profile(
    molecule,
    *,
    axis: AxisSpec = "principal",
    thickness: float = 1.0,
    method: str = "hull",
    environment=None,
) -> CrossSectionProfile:
    """Cross-sectional area of ``molecule`` along ``axis``, slice by slice.

    Parameters
    ----------
    axis:
        Slicing axis. ``"principal"`` (default) uses the long principal axis (the
        axis of smallest inertia), giving cross-sections perpendicular to the
        structure's length; ``"x"``/``"y"``/``"z"`` use a Cartesian axis (pass
        ``"z"`` for a membrane protein pre-oriented with its normal along z, as in
        the Beckstein convention); a length-3 vector is used directly (normalised).
    thickness:
        Slice thickness along ``axis`` in angstrom (must be > 0).
    method:
        ``"hull"`` (convex-hull area, pure NumPy) or ``"voronoi"`` (Beckstein-style
        Voronoi-cell sum bounded by ``environment`` atoms; needs SciPy).
    environment:
        Only for ``method="voronoi"``: a ``Molecule`` of bounding atoms (e.g. the
        membrane / solvent) whose Voronoi cells clip the target's outer cells. If
        ``None``, the molecule's own ``hetero`` atoms are used when present.

    Returns
    -------
    CrossSectionProfile
    """
    if thickness <= 0:
        raise ValueError(f"thickness must be > 0, got {thickness}")
    if method not in ("hull", "voronoi"):
        raise ValueError(f"unknown method {method!r}; expected 'hull' or 'voronoi'")

    unit = _resolve_axis(molecule, axis)
    coords = np.asarray(molecule.coords, dtype=float)
    if len(coords) < 3:
        return CrossSectionProfile(
            positions=np.empty(0),
            areas=np.empty(0),
            axis=unit,
            thickness=float(thickness),
            method=method,
        )

    e1, e2 = _plane_basis(unit)
    t = coords @ unit
    edges = _slice_edges(t, thickness)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if method == "hull":
        target_xy = np.column_stack([coords @ e1, coords @ e2])
        areas = _hull_profile(target_xy, t, edges)
    else:
        areas = _voronoi_profile(molecule, environment, unit, e1, e2, edges)

    return CrossSectionProfile(
        positions=centers,
        areas=areas,
        axis=unit,
        thickness=float(thickness),
        method=method,
    )


# --- axis / plane helpers ------------------------------------------------------

_CARTESIAN = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _resolve_axis(molecule, axis: AxisSpec) -> np.ndarray:
    if isinstance(axis, str):
        key = axis.lower()
        if key == "principal":
            # principal_axes() columns are ordered by ascending moment; the
            # smallest-moment axis is the axis of elongation (the long axis).
            return _unit(molecule.principal_axes()[:, 0])
        if key in _CARTESIAN:
            return _CARTESIAN[key].copy()
        raise ValueError(
            f"unknown axis {axis!r}; expected 'principal', 'x', 'y', 'z', or a 3-vector"
        )
    vec = np.asarray(axis, dtype=float).reshape(-1)
    if vec.shape != (3,):
        raise ValueError(f"axis vector must have length 3, got shape {vec.shape}")
    return _unit(vec)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError("axis vector must be non-zero")
    return vec / norm


def _plane_basis(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to ``unit``."""
    # Start from whichever Cartesian axis is least aligned with ``unit`` to keep
    # the cross product well-conditioned.
    ref = _CARTESIAN["x"] if abs(unit[0]) < 0.9 else _CARTESIAN["y"]
    e1 = _unit(np.cross(unit, ref))
    e2 = np.cross(unit, e1)
    return e1, e2


def _slice_edges(t: np.ndarray, thickness: float) -> np.ndarray:
    lo, hi = float(t.min()), float(t.max())
    span = hi - lo
    if span <= 0:
        # All atoms in one plane perpendicular to the axis: a single slice.
        return np.array([lo, lo + thickness])
    n = int(np.ceil(span / thickness))
    return lo + thickness * np.arange(n + 1)


# --- convex-hull area (pure NumPy) ---------------------------------------------

def _hull_profile(xy: np.ndarray, t: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # Bin each atom by its axial coordinate; the final edge is inclusive so the
    # topmost atoms are not dropped.
    bin_index = np.clip(np.digitize(t, edges[1:-1]), 0, len(edges) - 2)
    areas = np.zeros(len(edges) - 1, dtype=float)
    for b in range(len(areas)):
        pts = xy[bin_index == b]
        if len(pts) >= 3:
            areas[b] = _convex_hull_area(pts)
    return areas


def _convex_hull_area(points: np.ndarray) -> float:
    """Area of the convex hull of 2-D ``points`` via Andrew's monotone chain."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return 0.0
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def _half(chain_pts: np.ndarray) -> list:
        hull: list = []
        for p in chain_pts:
            while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        return hull

    lower = _half(pts)
    upper = _half(pts[::-1])
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    return _shoelace(hull)


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _shoelace(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# --- Voronoi-cell-sum area (SciPy, Beckstein-style) ----------------------------

def _voronoi_profile(molecule, environment, unit, e1, e2, edges) -> np.ndarray:
    try:
        from scipy.spatial import Voronoi  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised when SciPy absent
        raise ImportError(
            "method='voronoi' needs SciPy. Install it (`pip install scipy`) or use "
            "method='hull'."
        ) from exc

    # Two ways to obtain the (target, environment) split: an explicit environment
    # molecule means all of ``molecule`` is the target; otherwise fall back to the
    # protein / hetero split carried by the structure (the Beckstein convention).
    if environment is not None:
        target, env = molecule, environment
    elif getattr(molecule, "hetero", None):
        target, env = molecule.protein(), molecule.hetero_atoms()
    else:
        target, env = molecule, None
    if env is None or len(env) == 0 or len(target) == 0:
        raise ValueError(
            "method='voronoi' needs environment (hetero) atoms to bound the "
            "protein's outer cells; with none present it reduces exactly to "
            "method='hull', so use that instead (or pass environment=...)."
        )

    target_coords = np.asarray(target.coords, dtype=float)
    env_coords = np.asarray(env.coords, dtype=float)
    target_t = target_coords @ unit
    env_t = env_coords @ unit
    target_xy = np.column_stack([target_coords @ e1, target_coords @ e2])
    env_xy = np.column_stack([env_coords @ e1, env_coords @ e2])

    target_bin = np.clip(np.digitize(target_t, edges[1:-1]), 0, len(edges) - 2)
    env_bin = np.clip(np.digitize(env_t, edges[1:-1]), 0, len(edges) - 2)

    areas = np.zeros(len(edges) - 1, dtype=float)
    for b in range(len(areas)):
        tgt = target_xy[target_bin == b]
        bnd = env_xy[env_bin == b]
        if len(tgt) >= 3:
            areas[b] = _voronoi_cell_sum(tgt, bnd)
    return areas


def _voronoi_cell_sum(target_xy: np.ndarray, env_xy: np.ndarray) -> float:
    """Sum of the target atoms' Voronoi cells, clipped to the data hull.

    The environment points bound the target's outer cells (the role lipids play in
    the Beckstein method). To keep every real cell finite without fragile
    unbounded-cell reconstruction, a ring of far "bounding box" points is added so
    all real cells become closed (the same idea as Beckstein's periodic images);
    each target cell is then clipped to the convex hull of the *real* points so its
    area stays local and finite.
    """
    from scipy.spatial import Voronoi

    n_target = len(target_xy)
    real = np.vstack([target_xy, env_xy]) if len(env_xy) else target_xy
    if len(np.unique(real, axis=0)) < 4:
        return _convex_hull_area(target_xy)

    hull = _hull_polygon(real)
    # Far bounding points: pushing them well outside the data makes every real
    # point's Voronoi cell bounded, so no region carries the -1 (infinity) vertex.
    center = real.mean(axis=0)
    reach = float(np.ptp(real, axis=0).max()) * 5.0 + 1.0
    box = center + reach * np.array(
        [[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]],
        dtype=float,
    )
    vor = Voronoi(np.vstack([real, box]))

    total = 0.0
    for i in range(n_target):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            continue
        cell = _clip_polygon(vor.vertices[region], hull)
        if len(cell) >= 3:
            total += _shoelace(cell)
    return total


def _hull_polygon(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex-hull vertices of 2-D ``points``."""
    pts = np.unique(points, axis=0)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def _half(chain_pts: np.ndarray) -> list:
        hull: list = []
        for p in chain_pts:
            while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        return hull

    lower = _half(pts)
    upper = _half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of ``subject`` against convex polygon ``clip``."""
    if len(subject) < 3 or len(clip) < 3:
        return np.empty((0, 2))
    output = list(subject)
    for k in range(len(clip)):
        a, b = clip[k], clip[(k + 1) % len(clip)]
        edge = b - a
        if not output:
            break
        inputs = output
        output = []
        for j in range(len(inputs)):
            cur, prev = inputs[j], inputs[j - 1]
            cur_in = (edge[0] * (cur[1] - a[1]) - edge[1] * (cur[0] - a[0])) >= 0
            prev_in = (edge[0] * (prev[1] - a[1]) - edge[1] * (prev[0] - a[0])) >= 0
            if cur_in:
                if not prev_in:
                    output.append(_line_intersect(prev, cur, a, b))
                output.append(cur)
            elif prev_in:
                output.append(_line_intersect(prev, cur, a, b))
    return np.array(output) if output else np.empty((0, 2))


def _line_intersect(p1, p2, p3, p4) -> np.ndarray:
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if denom == 0:
        return p1
    s = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
    return p1 + s * d1
