"""Contact maps and residue-level contact analysis.

A contact map records which pairs of atoms (or residues) are within a distance
cutoff. Residue contact maps are a staple of protein-folding intuition, peptide
and NMR-ensemble comparison, and coarse-graining validation.

    cmap = mol.contact_map(cutoff=8.0, level="residue")   # -> ContactMap
    cmap.matrix                                           # (R, R) array
    cmap.plot()                                           # heatmap
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import elements
from .molecule import Molecule, ResidueId


@dataclass
class ContactMap:
    """A square contact (or contact-frequency) matrix with axis labels.

    ``matrix`` holds booleans-as-floats for a single structure, or values in
    ``[0, 1]`` for an ensemble frequency map. ``labels`` name the rows/columns
    (e.g. ``"A:LYS8"``); ``resids`` are the residue numbers for a residue map.
    """

    matrix: np.ndarray
    level: str
    cutoff: float
    labels: list[str] = field(default_factory=list)
    resids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    icodes: list[str] = field(default_factory=list)
    residue_ids: list[ResidueId] = field(default_factory=list)
    #: A difference map (``map_a - map_b``), set by :meth:`__sub__`. Tracked
    #: explicitly rather than inferred from negative values, since a difference
    #: where one state's contacts are a subset of the other has no negatives.
    is_difference: bool = False

    @property
    def is_frequency(self) -> bool:
        """True if the matrix holds fractional values (an ensemble map).

        For a difference map the discrete values are ``{-1, 0, 1}`` (boolean
        inputs) and anything beyond that marks an ensemble-frequency difference;
        for an ordinary map the discrete values are ``{0, 1}``.
        """
        allowed = (-1.0, 0.0, 1.0) if self.is_difference else (0.0, 1.0)
        return not np.all(np.isin(np.unique(self.matrix), allowed))

    @property
    def n_contacts(self) -> int:
        """Number of contacting pairs (non-zero entries above the diagonal)."""
        return int((np.triu(self.matrix, 1) > 0).sum())

    def contact_order(self) -> float:
        """Relative contact order: ``mean(|i-j|) / N`` over contacts.

        A standard protein-folding descriptor of how local vs. long-range the
        contacts are (low = mostly local/helical, high = many long-range
        contacts). Computed from row/column index separation, so it is most
        meaningful for a single chain. Returns ``0.0`` if there are no contacts.
        """
        i, j = np.nonzero(np.triu(self.matrix, 1) > 0)
        if len(i) == 0:
            return 0.0
        return float(np.abs(i - j).sum() / (len(i) * self.matrix.shape[0]))

    def plot(self, **kwargs):
        """Draw the contact map as a heatmap. See :func:`molscope.plotting.plot_contact_map`."""
        from .plotting import plot_contact_map

        return plot_contact_map(self, **kwargs)

    def __sub__(self, other: ContactMap) -> ContactMap:
        """Return ``self - other`` as a difference map (requires matching maps).

        Both maps must share the same level, shape, labels, and cutoff so that
        cell ``(i, j)`` denotes the same residue/atom pair in each; otherwise a
        :class:`ValueError` is raised. The result has ``is_difference=True`` with
        entries in ``[-1, 1]`` (``+1`` gained in ``self``, ``-1`` lost).
        """
        if not isinstance(other, ContactMap):
            return NotImplemented
        if self.level != other.level:
            raise ValueError(
                "cannot subtract contact maps with different levels: "
                f"{self.level} vs {other.level}"
            )
        if self.matrix.shape != other.matrix.shape:
            raise ValueError(
                "cannot subtract contact maps with different shapes: "
                f"{self.matrix.shape} vs {other.matrix.shape}"
            )
        if self.labels != other.labels:
            raise ValueError("cannot subtract contact maps with mismatched residue/atom labels")
        if self.cutoff != other.cutoff:
            raise ValueError(
                "cannot subtract contact maps with different cutoffs: "
                f"{self.cutoff} vs {other.cutoff}"
            )

        return ContactMap(
            matrix=self.matrix - other.matrix,
            level=self.level,
            cutoff=self.cutoff,
            labels=self.labels,
            resids=self.resids,
            icodes=self.icodes,
            residue_ids=self.residue_ids,
            is_difference=True,
        )


def contact_map(
    molecule: Molecule,
    cutoff: float = 8.0,
    level: str = "residue",
    method: str = "ca",
    backend: str = "numpy",
    device: str | None = None,
    min_seq_sep: int = 0,
    chain_mode: str = "all",
) -> ContactMap:
    """Compute a contact map for one structure (see :class:`ContactMap`).

    ``min_seq_sep`` drops same-chain contacts closer than this many positions
    apart (e.g. ``min_seq_sep=4`` removes trivial helical i,i+1..i+3 contacts).
    ``chain_mode`` keeps ``"all"`` contacts, only ``"intra"``-chain, or only
    ``"inter"``-chain pairs.
    """
    if level == "atom":
        if backend == "scipy":
            mat = np.zeros((len(molecule), len(molecule)), dtype=float)
            pairs = molecule.contacts(cutoff=cutoff)
            if len(pairs):
                mat[pairs[:, 0], pairs[:, 1]] = 1.0
                mat[pairs[:, 1], pairs[:, 0]] = 1.0
        else:
            from .distance import contact_matrix

            mat = contact_matrix(
                molecule.coords, cutoff=cutoff, backend=backend, device=device
            )
        mat = _apply_contact_filters(mat, molecule.chains, min_seq_sep, chain_mode)
        return ContactMap(mat, level="atom", cutoff=cutoff)

    if level != "residue":
        raise ValueError(f"level must be 'atom' or 'residue', got {level!r}")

    groups = list(molecule.residue_groups())
    if not groups:
        raise ValueError("residue contact map needs residue information")
    residue_ids = [group.residue_id for group in groups]
    labels = [residue_id.label() for residue_id in residue_ids]
    resids = np.array([residue_id.resid for residue_id in residue_ids], dtype=int)
    icodes = [residue_id.insertion_code for residue_id in residue_ids]
    chains = [residue_id.chain for residue_id in residue_ids]
    mat = _residue_contacts(molecule, groups, cutoff, method, backend, device)
    mat = _apply_contact_filters(mat, chains, min_seq_sep, chain_mode)
    return ContactMap(
        mat,
        level="residue",
        cutoff=cutoff,
        labels=labels,
        resids=resids,
        icodes=icodes,
        residue_ids=residue_ids,
    )


def _apply_contact_filters(mat: np.ndarray, chains, min_seq_sep: int, chain_mode: str):
    """Mask a contact matrix by chain relationship and sequence separation."""
    n = mat.shape[0]
    # No chain labels: treat the structure as a single chain.
    same = (
        (np.asarray(chains)[:, None] == np.asarray(chains)[None, :])
        if chains else np.ones((n, n), dtype=bool)
    )
    if chain_mode == "intra":
        mat = mat * same
    elif chain_mode == "inter":
        mat = mat * ~same
    elif chain_mode != "all":
        raise ValueError(f"chain_mode must be 'all', 'intra' or 'inter', got {chain_mode!r}")
    if min_seq_sep > 0:
        sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        mat = mat * ~(same & (sep < min_seq_sep))
    return mat


def _residue_contacts(molecule, groups, cutoff, method, backend, device) -> np.ndarray:
    if backend == "scipy":
        backend = "numpy"
    if method in ("ca", "com"):
        reps = np.array([_representative(molecule, idx, method) for idx, *_ in groups])
        from .distance import contact_matrix

        mat = contact_matrix(reps, cutoff=cutoff, backend=backend, device=device)
    elif method == "min":
        mat = _min_distance_contacts(molecule, groups, cutoff, backend, device)
    else:
        raise ValueError(f"method must be 'ca', 'com' or 'min', got {method!r}")
    np.fill_diagonal(mat, 0.0)
    return mat


def _representative(molecule, idx, method) -> np.ndarray:
    if method == "ca" and molecule.atom_names:
        ca = [i for i in idx if molecule.atom_names[i] == "CA"]
        if ca:
            return molecule.coords[ca[0]]
    if method == "com":
        w = np.array([elements.mass(molecule.elements[i]) for i in idx])
        return (w[:, None] * molecule.coords[idx]).sum(axis=0) / w.sum()
    return molecule.coords[idx].mean(axis=0)  # CA fallback: residue centroid


def _min_distance_contacts(molecule, groups, cutoff, backend, device) -> np.ndarray:
    """Residue contacts by closest inter-residue atom distance.

    Residues are contiguous atom runs (see ``Molecule.residue_groups``), so one
    full atom-atom distance matrix collapses to a residue-residue minimum with
    two ``np.minimum.reduceat`` calls over the residue start offsets.
    """
    from .distance import distance_matrix

    dist = distance_matrix(molecule.coords, backend=backend, device=device)
    starts = np.array([idx[0] for idx, *_ in groups])
    row_min = np.minimum.reduceat(dist, starts, axis=0)
    block_min = np.minimum.reduceat(row_min, starts, axis=1)
    mat = (block_min < cutoff).astype(float)
    np.fill_diagonal(mat, 0.0)
    return mat


def _label(chain, resname, resid, icode="") -> str:
    return ResidueId(chain, int(resid), icode, resname).label()
