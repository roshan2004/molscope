"""Analysis across a set of structures, e.g. the models of an NMR ensemble."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from .molecule import Molecule

#: Accepted RMSD/superposition selections for :func:`analyze_stream`.
STREAM_SELECTIONS = ("auto", "ca", "all")


def align_all(models: list[Molecule], reference: Optional[Molecule] = None) -> list[Molecule]:
    """Kabsch-superpose every model onto ``reference`` (default: the first model)."""
    ref = reference if reference is not None else models[0]
    return [m.superpose(ref) for m in models]


def average(models: list[Molecule], align: bool = True) -> Molecule:
    """Average structure over the ensemble (atoms matched by index)."""
    _check_consistent(models)
    aligned = align_all(models) if align else models
    coords = np.mean([m.coords for m in aligned], axis=0)
    return replace(models[0], coords=coords, name=f"{models[0].name} (average)")


def rmsf(models: list[Molecule], align: bool = True) -> np.ndarray:
    """Per-atom root-mean-square fluctuation about the mean position."""
    _check_consistent(models)
    aligned = align_all(models) if align else models
    stack = np.array([m.coords for m in aligned])      # (n_models, n_atoms, 3)
    mean = stack.mean(axis=0)
    return np.sqrt(((stack - mean) ** 2).sum(axis=2).mean(axis=0))


def contact_frequency(
    models: list[Molecule],
    cutoff: float = 8.0,
    level: str = "residue",
    method: str = "ca",
    backend: str = "numpy",
    device: str | None = None,
):
    """Fraction of models in which each pair is in contact (an ensemble map).

    Returns a :class:`~molscope.contactmap.ContactMap` whose matrix holds
    values in ``[0, 1]`` — the contact probability for each residue (or atom)
    pair across the ensemble. Useful for NMR variability and folding analysis.
    """
    from .contactmap import ContactMap, contact_map

    _check_consistent(models)
    maps = [
        contact_map(
            m, cutoff=cutoff, level=level, method=method,
            backend=backend, device=device,
        )
        for m in models
    ]
    freq = np.mean([cm.matrix for cm in maps], axis=0)
    first = maps[0]
    return ContactMap(
        freq,
        level=level,
        cutoff=cutoff,
        labels=first.labels,
        resids=first.resids,
        icodes=first.icodes,
        residue_ids=first.residue_ids,
    )


def rmsd_matrix(models: list[Molecule], align: bool = True) -> np.ndarray:
    """Symmetric ``(M, M)`` matrix of pairwise RMSDs between models."""
    _check_consistent(models)
    n = len(models)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = mat[j, i] = models[i].rmsd(models[j], align=align)
    return mat


def _check_consistent(models: list[Molecule]) -> None:
    if not models:
        raise ValueError("no models given")
    sizes = {len(m) for m in models}
    if len(sizes) != 1:
        raise ValueError(f"models have differing atom counts: {sorted(sizes)}")


@dataclass
class Clustering:
    """Result of clustering structures by RMSD.

    ``labels`` gives the 1-based cluster id of each model (same order as the
    input). ``matrix`` is the RMSD matrix used and ``linkage`` the scipy linkage.
    """

    labels: np.ndarray
    matrix: np.ndarray
    linkage: Optional[np.ndarray] = field(default=None)

    @property
    def n_clusters(self) -> int:
        return int(len(np.unique(self.labels)))

    @property
    def order(self) -> np.ndarray:
        """Model indices sorted by cluster (for a block-diagonal heatmap)."""
        return np.argsort(self.labels, kind="stable")

    def groups(self) -> dict[int, list[int]]:
        """Map each cluster id to the list of model indices it contains."""
        return {int(c): np.where(self.labels == c)[0].tolist()
                for c in np.unique(self.labels)}

    def medoid(self, cluster_id: int) -> int:
        """Index of the most central model of a cluster (min total RMSD)."""
        members = np.where(self.labels == cluster_id)[0]
        sub = self.matrix[np.ix_(members, members)]
        return int(members[sub.sum(axis=1).argmin()])

    def representatives(self) -> dict[int, int]:
        """Map each cluster id to its medoid model index."""
        return {int(c): self.medoid(int(c)) for c in np.unique(self.labels)}


def cluster(models, method: str = "hierarchical", cutoff: Optional[float] = None,
            n_clusters: Optional[int] = None, linkage: str = "average",
            align: bool = True, matrix=None) -> Clustering:
    """Cluster structures by pairwise RMSD.

    Pass ``n_clusters`` to cut the tree into a fixed number of clusters, or
    ``cutoff`` (an RMSD threshold in angstrom). With neither, a data-driven
    cutoff (the mean pairwise RMSD) is used. Reuses ``matrix`` if given, else
    computes :func:`rmsd_matrix`. Requires scipy.
    """
    if method != "hierarchical":
        raise ValueError(f"unknown method {method!r}; only 'hierarchical' is supported")

    dm = np.asarray(matrix, dtype=float) if matrix is not None else rmsd_matrix(models, align=align)
    if len(dm) < 2:
        return Clustering(labels=np.ones(len(dm), dtype=int), matrix=dm)

    try:
        from scipy.cluster.hierarchy import fcluster
        from scipy.cluster.hierarchy import linkage as _linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "clustering needs scipy; install it with: pip install 'molscope[fast]'"
        ) from exc

    z = _linkage(squareform(dm, checks=False), method=linkage)
    if n_clusters is not None:
        labels = fcluster(z, t=n_clusters, criterion="maxclust")
    else:
        if cutoff is None:
            cutoff = float(dm[np.triu_indices_from(dm, k=1)].mean())
        labels = fcluster(z, t=cutoff, criterion="distance")
    return Clustering(labels=labels, matrix=dm, linkage=z)


@dataclass
class StreamAnalysis:
    """Per-frame timelines from a single streaming pass over a trajectory.

    Holds one scalar per frame for each tracked property, so its memory is
    O(n_frames) in small floats regardless of system size. ``rmsd`` is measured
    against the first frame (so ``rmsd[0]`` is 0); the secondary-structure
    fraction arrays are present only when ``secondary_structure=True`` was
    requested and the frames are proteins.
    """

    n_frames: int
    n_atoms: int
    selection: str
    radius_of_gyration: np.ndarray
    rmsd: np.ndarray
    helix_fraction: Optional[np.ndarray] = None
    strand_fraction: Optional[np.ndarray] = None
    coil_fraction: Optional[np.ndarray] = None

    @property
    def has_secondary_structure(self) -> bool:
        return self.helix_fraction is not None

    def summary(self) -> dict:
        """A compact dict of the headline numbers (means, spread, drift)."""
        rg, rmsd = self.radius_of_gyration, self.rmsd
        out = {
            "n_frames": self.n_frames,
            "n_atoms": self.n_atoms,
            "selection": self.selection,
            "rg_mean": float(np.mean(rg)) if len(rg) else 0.0,
            "rg_std": float(np.std(rg)) if len(rg) else 0.0,
            "rg_min": float(np.min(rg)) if len(rg) else 0.0,
            "rg_max": float(np.max(rg)) if len(rg) else 0.0,
            "rmsd_mean": float(np.mean(rmsd)) if len(rmsd) else 0.0,
            "rmsd_max": float(np.max(rmsd)) if len(rmsd) else 0.0,
            "rmsd_final": float(rmsd[-1]) if len(rmsd) else 0.0,
        }
        if self.has_secondary_structure:
            out["helix_fraction_mean"] = float(np.nanmean(self.helix_fraction))
            out["strand_fraction_mean"] = float(np.nanmean(self.strand_fraction))
            out["coil_fraction_mean"] = float(np.nanmean(self.coil_fraction))
        return out

    def plot(self, show: bool = True):
        """Plot the timelines (Rg, RMSD, and SS fractions if tracked).

        See :func:`molscope.plotting.plot_stream_analysis`.
        """
        from .plotting import plot_stream_analysis

        return plot_stream_analysis(self, show=show)


def analyze_stream(
    source: str | os.PathLike | Iterable[Molecule],
    *,
    selection: str = "auto",
    align: bool = True,
    secondary_structure: bool = False,
) -> StreamAnalysis:
    """Track basic properties frame by frame over a trajectory, in one pass.

    This is a lightweight timeline helper, **not** a trajectory engine: it reads
    the multi-frame formats MolScope already reads (multi-model PDB, multi-frame
    XYZ, multi-record SDF) and computes a few scalars per frame without ever
    holding more than the reference frame in memory. It does not read binary MD
    formats (DCD/XTC/TRR), unwrap periodic boundaries, or track time/topology
    across frames; for that use a dedicated trajectory library.

    ``source`` is a path to a multi-frame file (streamed via
    :func:`molscope.stream`) or any iterable of :class:`Molecule` frames.
    Tracks, per frame:

    - the radius of gyration (whole frame);
    - the RMSD to the first frame, Kabsch-superposed when ``align`` is true,
      over ``selection`` (``"auto"`` uses C-alphas when present, else all atoms;
      ``"ca"`` forces C-alphas; ``"all"`` uses every atom);
    - with ``secondary_structure=True``, the helix/strand/coil fractions from the
      simplified DSSP assignment (proteins only; a frame whose assignment fails
      contributes ``NaN`` rather than aborting the run).

    Frames must all share the first frame's atom count; a mismatch raises
    :class:`ValueError`. Returns a :class:`StreamAnalysis`.
    """
    if selection not in STREAM_SELECTIONS:
        raise ValueError(
            f"selection must be one of {STREAM_SELECTIONS}, got {selection!r}"
        )

    frames = _iter_frames(source)

    n_atoms = 0
    sel = selection
    ref_sel: Optional[Molecule] = None
    rg: list[float] = []
    rmsd: list[float] = []
    helix: list[float] = []
    strand: list[float] = []
    coil: list[float] = []

    for i, frame in enumerate(frames):
        if i == 0:
            n_atoms = len(frame)
            sel = _resolve_selection(frame, selection)
            ref_sel = _select(frame, sel)
            if len(ref_sel) == 0:
                raise ValueError(
                    f"selection {selection!r} matched no atoms in the first frame"
                )
        elif len(frame) != n_atoms:
            raise ValueError(
                f"frame {i} has {len(frame)} atoms but the first frame has {n_atoms}; "
                "analyze_stream needs a consistent topology across frames"
            )

        rg.append(float(frame.radius_of_gyration))
        rmsd.append(float(_select(frame, sel).rmsd(ref_sel, align=align)))
        if secondary_structure:
            h, s, c = _ss_fractions(frame)
            helix.append(h)
            strand.append(s)
            coil.append(c)

    if not rg:
        raise ValueError("no frames in the stream")

    return StreamAnalysis(
        n_frames=len(rg),
        n_atoms=n_atoms,
        selection=sel,
        radius_of_gyration=np.asarray(rg, dtype=float),
        rmsd=np.asarray(rmsd, dtype=float),
        helix_fraction=np.asarray(helix, dtype=float) if secondary_structure else None,
        strand_fraction=np.asarray(strand, dtype=float) if secondary_structure else None,
        coil_fraction=np.asarray(coil, dtype=float) if secondary_structure else None,
    )


def _iter_frames(source):
    """Yield frames from a path (streamed) or pass an iterable of molecules through."""
    if isinstance(source, (str, os.PathLike)):
        from .io import stream

        return stream(os.fspath(source))
    return iter(source)


def _resolve_selection(frame: Molecule, selection: str) -> str:
    """Turn ``"auto"`` into ``"ca"`` or ``"all"`` based on the first frame.

    Falls back to all-atoms when there is no atom-name metadata (e.g. a bare XYZ
    trajectory), so ``"auto"`` never fails on a non-protein frame.
    """
    if selection != "auto":
        return selection
    if frame.atom_names and len(frame.alpha_carbons()):
        return "ca"
    return "all"


def _select(frame: Molecule, selection: str) -> Molecule:
    return frame.alpha_carbons() if selection == "ca" else frame


def _ss_fractions(frame: Molecule):
    """Helix/strand/coil fractions for one frame, or NaNs if assignment fails."""
    try:
        summary = frame.secondary_structure().summary()
    except Exception:  # pragma: no cover - non-protein or backbone-less frame
        return float("nan"), float("nan"), float("nan")
    return (
        float(summary["helix_fraction"]),
        float(summary["strand_fraction"]),
        float(summary["coil_fraction"]),
    )
