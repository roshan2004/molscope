"""Solvent-accessible surface area (SASA) via a vectorised Shrake-Rupley sphere.

A pure-NumPy approximation: each atom's expanded sphere (its van der Waals
radius plus a water probe) is sampled with a fixed set of quasi-uniform points,
and a point counts as accessible when it falls outside every neighbouring atom's
expanded sphere. The accessible fraction times the sphere area is the atom's
SASA. Accuracy improves with ``n_points`` (the default is a good speed/accuracy
trade-off and lands within a few percent of an exact analytical surface). No C
extensions or external SASA libraries are required; neighbour search reuses the
optional SciPy KD-tree with a NumPy fallback, like the rest of MolScope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import elements

DEFAULT_PROBE_RADIUS = 1.4   # water probe, angstrom
DEFAULT_N_POINTS = 192
SASA_LEVELS = ("atom", "residue")
DEFAULT_RSA_THRESHOLD = 0.20   # RSA >= this -> "exposed", else "buried"


def _fibonacci_sphere(n: int) -> np.ndarray:
    """``(n, 3)`` quasi-uniform unit vectors via the Fibonacci spiral."""
    if n < 1:
        raise ValueError(f"n_points must be >= 1, got {n}")
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)              # polar angle, uniform in cos
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i          # golden-angle azimuth
    sin_phi = np.sin(phi)
    return np.stack(
        [sin_phi * np.cos(theta), sin_phi * np.sin(theta), np.cos(phi)], axis=1
    )


def _neighbor_lists(coords: np.ndarray, reach: np.ndarray) -> list[np.ndarray]:
    """For each atom, indices of other atoms whose expanded sphere can occlude it.

    ``reach[i]`` is atom ``i``'s expanded radius plus the largest expanded radius
    present, so the candidate set is a safe superset of true occluders; the
    caller refines with exact point-in-sphere tests. Uses SciPy's KD-tree when
    available, else a per-atom NumPy distance scan (no full N×N matrix).
    """
    n = len(coords)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None

    if cKDTree is not None:
        tree = cKDTree(coords)
        out = []
        for i in range(n):
            idx = np.asarray(tree.query_ball_point(coords[i], reach[i]), dtype=int)
            out.append(idx[idx != i])
        return out

    out = []
    for i in range(n):
        mask = np.linalg.norm(coords - coords[i], axis=1) < reach[i]
        mask[i] = False
        out.append(np.nonzero(mask)[0])
    return out


def atom_sasa(
    molecule,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    n_points: int = DEFAULT_N_POINTS,
) -> np.ndarray:
    """Per-atom solvent-accessible surface area in Å² (Shrake-Rupley).

    See :func:`sasa` for the parameters; this is the atom-level core that the
    residue level aggregates.
    """
    coords = np.asarray(molecule.coords, dtype=float)
    n = len(coords)
    if n == 0:
        return np.empty(0, dtype=float)
    radii = np.array(
        [elements.vdw_radius(e) for e in molecule.elements], dtype=float
    ) + probe_radius
    sphere = _fibonacci_sphere(n_points)
    full_area = 4.0 * np.pi * radii ** 2            # area of each atom's sphere
    if n == 1:
        return full_area.copy()

    neighbors = _neighbor_lists(coords, radii + radii.max())
    sasa_values = np.empty(n, dtype=float)
    for i in range(n):
        nbr = neighbors[i]
        if len(nbr) == 0:
            sasa_values[i] = full_area[i]
            continue
        points = coords[i] + radii[i] * sphere      # (P, 3) sample points
        d2 = ((points[:, None, :] - coords[nbr][None, :, :]) ** 2).sum(axis=2)
        buried = (d2 < (radii[nbr] ** 2)[None, :]).any(axis=1)
        sasa_values[i] = full_area[i] * (~buried).sum() / n_points
    return sasa_values


def sasa(
    molecule,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    n_points: int = DEFAULT_N_POINTS,
    level: str = "atom",
) -> np.ndarray:
    """Solvent-accessible surface area in Å² (vectorised Shrake-Rupley).

    ``level="atom"`` returns a per-atom ``(N,)`` array; ``level="residue"``
    returns a per-residue ``(R,)`` array (each residue's atoms summed, in
    ``molecule.residue_groups()`` order). ``probe_radius`` is the solvent probe
    (``1.4`` Å approximates water); a larger ``n_points`` is more accurate but
    slower. The whole-structure total is ``sasa(mol).sum()``.

    This is an approximation, not a substitute for an exact analytical surface:
    it is intended as a fast, dependency-free descriptor of solvent exposure for
    the "PDB to descriptors" workflow.
    """
    if level not in SASA_LEVELS:
        raise ValueError(f"level must be 'atom' or 'residue', got {level!r}")
    per_atom = atom_sasa(molecule, probe_radius=probe_radius, n_points=n_points)
    if level == "atom":
        return per_atom
    groups = list(molecule.residue_groups())
    if not groups:
        raise ValueError("residue-level SASA needs residue information")
    return np.array(
        [per_atom[group.atom_indices].sum() for group in groups], dtype=float
    )


@dataclass
class ResidueExposure:
    """Per-residue relative solvent accessibility (RSA), in residue order.

    ``sasa`` is the absolute residue SASA (Å²); ``rsa`` is that divided by the
    residue's reference maximum (Tien et al. 2013), ``NaN`` where no reference
    exists (ligands, waters, non-standard residues). ``exposed`` is
    ``rsa >= threshold`` (``False`` wherever ``rsa`` is ``NaN``). RSA can slightly
    exceed 1 because the reference is an extended Gly-X-Gly tripeptide.
    """

    resids: np.ndarray
    chains: list
    resnames: list
    sasa: np.ndarray         # absolute, Å²
    rsa: np.ndarray          # relative, NaN where no reference
    exposed: np.ndarray      # bool, rsa >= threshold
    threshold: float

    def __len__(self) -> int:
        return len(self.resids)


def relative_sasa(
    molecule,
    probe_radius: float = DEFAULT_PROBE_RADIUS,
    n_points: int = DEFAULT_N_POINTS,
    threshold: float = DEFAULT_RSA_THRESHOLD,
) -> ResidueExposure:
    """Relative solvent accessibility (RSA) per residue and an exposed/buried call.

    Computes absolute residue SASA on the whole structure (so burial reflects
    neighbours) and divides by each residue's reference maximum SASA (Tien et al.
    2013) to give RSA, then classifies ``rsa >= threshold`` (default ``0.20``) as
    exposed. Residues without a reference (ligands, waters, non-standard names)
    get ``NaN`` RSA and are not exposed. Returns a :class:`ResidueExposure` in
    ``molecule.residue_groups()`` order.
    """
    groups = list(molecule.residue_groups())
    if not groups:
        raise ValueError("relative SASA needs residue information")
    absolute = sasa(molecule, probe_radius=probe_radius, n_points=n_points, level="residue")
    resnames = [group.residue_id.resname for group in groups]
    resids = np.array([group.residue_id.resid for group in groups], dtype=int)
    chains = [group.residue_id.chain for group in groups]
    ref = np.array([elements.max_asa(name) or np.nan for name in resnames], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsa = absolute / ref
    exposed = np.nan_to_num(rsa, nan=-1.0) >= threshold
    return ResidueExposure(
        resids=resids, chains=chains, resnames=resnames,
        sasa=absolute, rsa=rsa, exposed=exposed, threshold=float(threshold),
    )
