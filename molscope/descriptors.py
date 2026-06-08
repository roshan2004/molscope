"""Fixed-size molecular descriptors for quick ML feature tables."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .io import read

DEFAULT_ELEMENTS = (
    "H", "C", "N", "O", "S", "P", "F", "CL", "BR", "I", "NA", "MG", "CA", "FE", "ZN",
)

DESCRIPTOR_PRESETS = ("native-basic", "native-3d", "rdkit-basic")
RDKIT_BASIC_DESCRIPTORS = (
    "MolWt",
    "HeavyAtomCount",
    "TPSA",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
)

# SASA point count for the descriptor summary: coarser than the Molecule.sasa
# default (192) since a summary statistic tolerates a few percent error and the
# descriptor path is used for batch ML tables where speed matters.
SASA_DESCRIPTOR_N_POINTS = 96

# Polar-contact proxy: count N/O atom pairs whose separation falls in this window
# (angstrom). The lower bound excludes directly bonded / 1-3 heavy-atom contacts;
# the upper bound is the usual hydrogen-bond heavy-atom distance. This is a coarse
# geometric proxy (no angle or hydrogen-position check), not a validated H-bond.
POLAR_CONTACT_ELEMENTS = ("N", "O")
POLAR_CONTACT_MIN = 2.5
POLAR_CONTACT_MAX = 3.5

# Salt bridge: a basic side-chain nitrogen within this distance of an acidic
# side-chain oxygen (Barlow-Thornton style). Charged side-chain atom names:
SALT_BRIDGE_CUTOFF = 4.0
BASIC_SIDECHAIN_ATOMS = {"ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "HIS": {"ND1", "NE2"}}
ACIDIC_SIDECHAIN_ATOMS = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}


def descriptors(
    molecule,
    *,
    preset: Optional[str] = None,
    elements_to_count=DEFAULT_ELEMENTS,
    distance_bins: int = 10,
    distance_range: tuple[float, float] = (0.0, 20.0),
    distance_chunk_size: int = 1024,
    contact_cutoff: float = 5.0,
    residue_contact_cutoff: float = 8.0,
    sasa_n_points: int = SASA_DESCRIPTOR_N_POINTS,
    include_rdkit: bool = False,
    rdkit_descriptor_names: Optional[list[str]] = None,
    rdkit_prefix: str = "rdkit_",
) -> dict:
    """Return a flat descriptor dictionary for a molecule.

    The defaults are fixed-size and suitable for small ML tables. Matrix-valued
    features such as contact maps remain available through ``mol.contact_map()``;
    this function records table-friendly summaries of them. Pairwise distance
    histograms and contact counts are computed in coordinate blocks, controlled
    by ``distance_chunk_size``.
    """
    preset = _validate_preset(preset)
    if preset == "rdkit-basic":
        include_rdkit = True
        if rdkit_descriptor_names is None:
            rdkit_descriptor_names = list(RDKIT_BASIC_DESCRIPTORS)

    coords = np.asarray(molecule.coords, dtype=float)
    n_atoms = len(molecule)
    masses = molecule.masses if n_atoms else np.empty(0, dtype=float)
    desc = {
        "n_atoms": float(n_atoms),
        "n_residues": float(_n_residues(molecule)),
        "molecular_mass": float(masses.sum()) if n_atoms else 0.0,
    }

    counts = Counter(e.upper() for e in molecule.elements if e)
    for symbol in elements_to_count:
        desc[f"count_{symbol.upper()}"] = float(counts.get(symbol.upper(), 0))

    if n_atoms == 0:
        desc = _empty_descriptors(desc, distance_bins)
        if include_rdkit:
            desc.update(_rdkit_descriptors(molecule, rdkit_descriptor_names, rdkit_prefix))
        return _apply_preset(desc, preset, elements_to_count, distance_bins, rdkit_prefix)

    dims = molecule.dimensions
    desc.update({
        "radius_of_gyration": molecule.radius_of_gyration,
        "dim_x": float(dims[0]),
        "dim_y": float(dims[1]),
        "dim_z": float(dims[2]),
        "bbox_volume": float(np.prod(dims)),
        "compactness": _compactness(n_atoms, dims),
    })

    inertia = inertia_tensor(molecule)
    principal_moments, principal_axes = np.linalg.eigh(inertia)
    order = np.argsort(principal_moments)
    principal_moments = principal_moments[order]
    principal_axes = principal_axes[:, order]
    desc["inertia_tensor"] = inertia.reshape(-1).astype(float).tolist()
    desc["principal_moments"] = principal_moments.astype(float).tolist()
    desc["principal_axes"] = principal_axes.reshape(-1).astype(float).tolist()
    desc["shape_anisotropy"] = shape_anisotropy(principal_moments)
    desc.update(shape_descriptors(principal_moments, float(np.sum(molecule.masses))))

    hist = _pairwise_distance_histogram(
        coords,
        bins=distance_bins,
        distance_range=distance_range,
        chunk_size=distance_chunk_size,
    )
    desc["distance_histogram"] = hist.astype(float).tolist()
    desc.update(_bond_length_summary(molecule))
    desc.update(_contact_summary(molecule, contact_cutoff))
    desc.update(_residue_contact_summary(molecule, residue_contact_cutoff))
    # Surface and interaction features (native-3d only): the SASA scan is the
    # one costly step here, so skip it for the lighter presets that omit them.
    if preset in (None, "native-3d"):
        desc.update(_sasa_summary(molecule, sasa_n_points))
        desc["polar_contact_count"] = float(_polar_contact_count(molecule))
        desc["salt_bridge_count"] = float(_salt_bridge_count(molecule))
    if include_rdkit:
        desc.update(_rdkit_descriptors(molecule, rdkit_descriptor_names, rdkit_prefix))
    return _apply_preset(desc, preset, elements_to_count, distance_bins, rdkit_prefix)


def inertia_tensor(molecule) -> np.ndarray:
    """Mass-weighted inertia tensor around the centre of mass."""
    coords = np.asarray(molecule.coords, dtype=float)
    if len(molecule) == 0:
        return np.zeros((3, 3), dtype=float)
    centered = coords - molecule.center_of_mass
    masses = molecule.masses
    r2 = (centered ** 2).sum(axis=1)
    tensor = np.eye(3) * np.sum(masses * r2)
    tensor -= centered.T @ (centered * masses[:, None])
    return tensor


def shape_anisotropy(principal_moments) -> float:
    """Dimensionless anisotropy from principal moments of inertia."""
    moments = np.asarray(principal_moments, dtype=float)
    denom = float(np.sum(moments ** 2))
    if denom == 0.0:
        return 0.0
    mean = float(moments.mean())
    return float(1.5 * np.sum((moments - mean) ** 2) / denom)


def shape_descriptors(principal_moments, total_mass: float) -> dict:
    """Gyration-tensor shape descriptors from mass-weighted inertia moments.

    Returns ``asphericity`` (b), ``acylindricity`` (c) and
    ``relative_shape_anisotropy`` (κ²) — the standard polymer-physics shape
    parameters. They are defined on the eigenvalues of the gyration tensor
    ``λ₁ ≤ λ₂ ≤ λ₃``, which are recovered from the already-computed mass-weighted
    principal moments of inertia ``Iᵢ`` by ``λᵢ = (T − Iᵢ)/M`` with ``T = ΣIᵢ/2``
    and ``M`` the total mass::

        b  = λ₃ − (λ₁ + λ₂)/2          # 0 for a sphere, grows as it elongates
        c  = λ₂ − λ₁                    # 0 for any axially symmetric shape
        κ² = (b² + ¾c²) / R_g⁴          # in [0, 1]

    ``κ²`` is 0 for arrangements with tetrahedral or higher symmetry (a sphere)
    and 1 for a perfectly linear one, so it is the rigorous "how non-spherical"
    scalar. Degenerate inputs (no mass or a single point) return zeros.
    """
    moments = np.asarray(principal_moments, dtype=float)
    if total_mass <= 0.0 or moments.size != 3:
        return {
            "asphericity": 0.0,
            "acylindricity": 0.0,
            "relative_shape_anisotropy": 0.0,
        }
    half_trace = moments.sum() / 2.0
    # Gyration eigenvalues (mass-weighted), ascending; clip rounding noise.
    lam = np.sort(np.clip((half_trace - moments) / total_mass, 0.0, None))
    l1, l2, l3 = lam
    rg2 = float(lam.sum())
    b = l3 - 0.5 * (l1 + l2)
    c = l2 - l1
    kappa2 = (b * b + 0.75 * c * c) / (rg2 * rg2) if rg2 > 0.0 else 0.0
    return {
        "asphericity": float(b),
        "acylindricity": float(c),
        "relative_shape_anisotropy": float(min(max(kappa2, 0.0), 1.0)),
    }


def _sasa_summary(molecule, n_points: int) -> dict[str, float]:
    """Total/mean/std/max of the per-atom solvent-accessible surface area (Å²)."""
    from .sasa import sasa

    per_atom = sasa(molecule, n_points=n_points)
    if len(per_atom) == 0:
        return {"sasa_total": 0.0, "sasa_mean": 0.0, "sasa_std": 0.0, "sasa_max": 0.0}
    return {
        "sasa_total": float(per_atom.sum()),
        "sasa_mean": float(per_atom.mean()),
        "sasa_std": float(per_atom.std()),
        "sasa_max": float(per_atom.max()),
    }


def _polar_contact_count(molecule) -> int:
    """Count N/O atom pairs separated by 2.5-3.5 Å (a coarse polar-contact proxy).

    Same-residue pairs are excluded (so trivial intra-residue geometry does not
    dominate); the distance window keeps directly bonded heavy atoms out. This is
    a geometric proxy, not a validated hydrogen-bond count: it ignores bond angles
    and hydrogen positions.
    """
    elements = [e.upper() for e in molecule.elements]
    idx = np.array(
        [i for i, e in enumerate(elements) if e in POLAR_CONTACT_ELEMENTS], dtype=int
    )
    if len(idx) < 2:
        return 0

    from .distance import find_contacts

    pairs = find_contacts(molecule.coords[idx], POLAR_CONTACT_MAX)
    if len(pairs) == 0:
        return 0
    a, b = idx[pairs[:, 0]], idx[pairs[:, 1]]
    dist = np.linalg.norm(molecule.coords[a] - molecule.coords[b], axis=1)
    keep = dist >= POLAR_CONTACT_MIN
    if len(molecule.resids):
        resids = np.asarray(molecule.resids)
        same = resids[a] == resids[b]
        if molecule.chains:
            same &= np.asarray(molecule.chains)[a] == np.asarray(molecule.chains)[b]
        keep &= ~same
    return int(keep.sum())


def _salt_bridge_count(molecule) -> int:
    """Count basic/acidic residue pairs bridged within 4 Å (Barlow-Thornton style).

    A salt bridge is a basic side-chain nitrogen (Arg/Lys/His) within
    ``SALT_BRIDGE_CUTOFF`` of an acidic side-chain oxygen (Asp/Glu). Unique
    basic↔acidic residue pairs are counted, so a residue pair with several close
    atom pairs still counts once. Needs residue and atom-name metadata; returns 0
    otherwise (e.g. small molecules).
    """
    names, resnames = molecule.atom_names, molecule.resnames
    if not names or not resnames or not len(molecule.resids):
        return 0
    upper = [r.upper() for r in resnames]
    basic = [
        i for i in range(len(molecule))
        if upper[i] in BASIC_SIDECHAIN_ATOMS and names[i] in BASIC_SIDECHAIN_ATOMS[upper[i]]
    ]
    acidic = [
        i for i in range(len(molecule))
        if upper[i] in ACIDIC_SIDECHAIN_ATOMS and names[i] in ACIDIC_SIDECHAIN_ATOMS[upper[i]]
    ]
    if not basic or not acidic:
        return 0
    dist = np.linalg.norm(
        molecule.coords[basic][:, None, :] - molecule.coords[acidic][None, :, :], axis=2
    )
    bi, ai = np.nonzero(dist <= SALT_BRIDGE_CUTOFF)
    chains = molecule.chains or [""] * len(molecule)
    icodes = molecule.icodes or [""] * len(molecule)
    resids = molecule.resids
    pairs = {
        (
            (chains[basic[x]], int(resids[basic[x]]), icodes[basic[x]]),
            (chains[acidic[y]], int(resids[acidic[y]]), icodes[acidic[y]]),
        )
        for x, y in zip(bi, ai)
    }
    return len(pairs)


def featurize_many(
    paths,
    *,
    feature_names: Optional[list[str]] = None,
    return_names: bool = False,
    **descriptor_kwargs,
):
    """Read structures and return a numeric descriptor matrix.

    By default columns are the union of descriptor keys found across the input
    molecules. Pass ``feature_names`` to force a stable column order, or
    ``return_names=True`` to receive ``(X, names)``.
    """
    rows = [flatten_descriptors(descriptors(read(path), **descriptor_kwargs)) for path in paths]
    preset = descriptor_kwargs.get("preset")
    if feature_names is not None:
        names = feature_names
    elif preset is not None:
        names = descriptor_feature_names(
            preset,
            elements_to_count=descriptor_kwargs.get("elements_to_count", DEFAULT_ELEMENTS),
            distance_bins=descriptor_kwargs.get("distance_bins", 10),
            rdkit_prefix=descriptor_kwargs.get("rdkit_prefix", "rdkit_"),
        )
    else:
        names = sorted({key for row in rows for key in row})
    matrix = np.array([[row.get(name, 0.0) for name in names] for row in rows], dtype=float)
    return (matrix, names) if return_names else matrix


@dataclass
class FeatureScaler:
    """Per-column affine standardiser for a descriptor matrix.

    The feature-side companion to :class:`~molscope.TargetScaler`: where that
    standardises a graph dataset's *labels*, this standardises a
    :func:`featurize_many` *feature matrix*. ``transform`` maps a matrix into
    zero-mean, unit-variance-per-column space and ``inverse_transform`` maps it
    back. Fit it on the train rows only (see :func:`standardize_features`) so
    validation/test statistics never leak into training.

    ``mean`` and ``std`` are per-column arrays. Near-constant columns (training
    std below ``1e-8``) are given ``std = 1`` so they pass through as a plain
    mean shift instead of exploding a test row that happens to differ.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X) -> FeatureScaler:
        """Fit column means and standard deviations on the rows of ``X``."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D feature matrix, got shape {X.shape}")
        if len(X) == 0:
            raise ValueError("cannot fit a scaler on an empty matrix")
        std = X.std(axis=0)
        constant = std < 1e-8
        if constant.any():
            cols = np.flatnonzero(constant).tolist()
            shown = cols[:10]
            more = "" if len(cols) <= 10 else f" (+{len(cols) - 10} more)"
            warnings.warn(
                f"{len(cols)} near-constant column(s) {shown}{more} have "
                "~zero variance on the fitted rows; their std is set to 1 so "
                "they pass through as a plain mean shift. Consider dropping "
                "them -- a constant feature carries no signal.",
                stacklevel=2,
            )
        std = np.where(constant, 1.0, std)
        return cls(mean=X.mean(axis=0), std=std)

    def transform(self, X):
        """Standardise ``X`` with the fitted statistics."""
        X = np.asarray(X, dtype=float)
        self._check_width(X)
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        """Map standardised values back to original units."""
        X = np.asarray(X, dtype=float)
        self._check_width(X)
        return X * self.std + self.mean

    def _check_width(self, X):
        """Reject a matrix whose column count does not match the fit.

        NumPy broadcasting masks the mismatch in the single-column case --
        a (n, 1) matrix against a wider scaler (or vice versa) would silently
        broadcast to a wrong-shaped result rather than erroring -- so guard it.
        """
        width = X.shape[-1] if X.ndim else 0
        if width != len(self.mean):
            raise ValueError(
                f"feature matrix has {width} columns but the scaler was fit on "
                f"{len(self.mean)}"
            )


def standardize_features(X, train_index):
    """Standardise a feature matrix using train-split statistics only.

    Fits a :class:`FeatureScaler` on ``X[train_index]`` and returns
    ``(X_standardised, scaler)`` with **every** row transformed. The train-only
    fit is the point: computing column means/standard deviations over the whole
    matrix leaks the validation/test feature distribution into training, the
    same mistake :meth:`~molscope.GraphDataset.standardize_targets` avoids on the
    label side.

    ``train_index`` is any iterable of integer row indices -- a
    :class:`~molscope.prepare.SplitResult`'s ``.train``, scikit-learn split
    indices, or a hand-built list. Map a model's predictions or a fitted scaler
    back to physical units with ``scaler.inverse_transform(...)``.

    >>> X, names = ms.featurize_many(paths, preset="native-3d", return_names=True)
    >>> X_std, scaler = ms.standardize_features(X, split.train)
    """
    X = np.asarray(X, dtype=float)
    train_index = list(train_index)
    if not train_index:
        raise ValueError("train_index is empty; need at least one train row to fit")
    n = len(X)
    bad = [i for i in train_index if i < 0 or i >= n]
    if bad:
        raise ValueError(
            f"train_index has out-of-range rows {bad[:5]} for a matrix with {n} "
            "rows; indices must be in [0, n_rows). Negative indices are rejected "
            "because they would silently pull held-out rows into the fit."
        )
    scaler = FeatureScaler.fit(X[train_index])
    return scaler.transform(X), scaler


def descriptor_feature_names(
    preset: str,
    *,
    elements_to_count=DEFAULT_ELEMENTS,
    distance_bins: int = 10,
    rdkit_prefix: str = "rdkit_",
) -> list[str]:
    """Return stable flattened feature names for a descriptor preset."""
    preset = _validate_preset(preset, required=True)
    names = _preset_scalar_names(preset, elements_to_count, rdkit_prefix)
    if preset == "native-3d":
        names += [f"inertia_tensor_{i}" for i in range(9)]
        names += [f"principal_moments_{i}" for i in range(3)]
        names += [f"principal_axes_{i}" for i in range(9)]
        names += [f"distance_histogram_{i}" for i in range(distance_bins)]
    return names


def flatten_descriptors(desc: dict) -> dict[str, float]:
    """Expand list-valued descriptors into scalar columns."""
    flat = {}
    for key, value in desc.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            for i, item in enumerate(value):
                flat[f"{key}_{i}"] = float(item)
        else:
            flat[key] = float(value)
    return flat


def _empty_descriptors(desc: dict, distance_bins: int) -> dict:
    desc.update({
        "radius_of_gyration": 0.0,
        "dim_x": 0.0,
        "dim_y": 0.0,
        "dim_z": 0.0,
        "bbox_volume": 0.0,
        "compactness": 0.0,
        "inertia_tensor": [0.0] * 9,
        "principal_moments": [0.0] * 3,
        "principal_axes": [0.0] * 9,
        "shape_anisotropy": 0.0,
        "asphericity": 0.0,
        "acylindricity": 0.0,
        "relative_shape_anisotropy": 0.0,
        "sasa_total": 0.0,
        "sasa_mean": 0.0,
        "sasa_std": 0.0,
        "sasa_max": 0.0,
        "polar_contact_count": 0.0,
        "salt_bridge_count": 0.0,
        "distance_histogram": [0.0] * distance_bins,
        "bond_count": 0.0,
        "bond_length_mean": 0.0,
        "bond_length_std": 0.0,
        "bond_length_min": 0.0,
        "bond_length_max": 0.0,
        "atom_contact_count": 0.0,
        "atom_contact_density": 0.0,
        "residue_contact_count": 0.0,
        "residue_contact_density": 0.0,
    })
    return desc


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    n = len(coords)
    if n < 2:
        return np.empty(0, dtype=float)
    i, j = np.triu_indices(n, k=1)
    return np.linalg.norm(coords[i] - coords[j], axis=1)


def _pairwise_distance_histogram(
    coords: np.ndarray,
    *,
    bins: int,
    distance_range: tuple[float, float],
    chunk_size: int,
) -> np.ndarray:
    if distance_range is None:
        distances = _pairwise_distances(coords)
        hist, _ = np.histogram(distances, bins=bins, range=distance_range)
        return hist.astype(float)

    chunk_size = _validate_chunk_size(chunk_size, "distance_chunk_size")
    hist, edges = np.histogram(np.empty(0, dtype=float), bins=bins, range=distance_range)
    hist = hist.astype(float)
    for _, end, block in _iter_blocks(coords, chunk_size):
        if len(block) > 1:
            d2 = _squared_distances(block, block)
            i, j = np.triu_indices(len(block), k=1)
            hist += np.histogram(np.sqrt(d2[i, j]), bins=edges)[0]

        for _, _, other in _iter_blocks(coords, chunk_size, start=end):
            d2 = _squared_distances(block, other)
            hist += np.histogram(np.sqrt(d2.ravel()), bins=edges)[0]
    return hist


def _bond_length_summary(molecule) -> dict[str, float]:
    bonds = molecule.bonds()
    if len(bonds) == 0:
        return {
            "bond_count": 0.0,
            "bond_length_mean": 0.0,
            "bond_length_std": 0.0,
            "bond_length_min": 0.0,
            "bond_length_max": 0.0,
        }
    lengths = np.linalg.norm(molecule.coords[bonds[:, 0]] - molecule.coords[bonds[:, 1]], axis=1)
    return {
        "bond_count": float(len(lengths)),
        "bond_length_mean": float(lengths.mean()),
        "bond_length_std": float(lengths.std()),
        "bond_length_min": float(lengths.min()),
        "bond_length_max": float(lengths.max()),
    }


def _contact_summary(molecule, cutoff: float) -> dict[str, float]:
    contact_count = molecule.contact_count(cutoff=cutoff)
    possible = len(molecule) * (len(molecule) - 1) / 2
    return {
        "atom_contact_count": float(contact_count),
        "atom_contact_density": float(contact_count / possible) if possible else 0.0,
    }



def _residue_contact_summary(molecule, cutoff: float) -> dict[str, float]:
    if len(molecule.resids) == 0:
        return {"residue_contact_count": 0.0, "residue_contact_density": 0.0}
    try:
        matrix = molecule.contact_map(cutoff=cutoff, level="residue").matrix
    except ValueError:
        return {"residue_contact_count": 0.0, "residue_contact_density": 0.0}
    n = len(matrix)
    possible = n * (n - 1) / 2
    count = float(np.triu(matrix.astype(bool), k=1).sum())
    return {
        "residue_contact_count": count,
        "residue_contact_density": float(count / possible) if possible else 0.0,
    }


def _n_residues(molecule) -> int:
    if len(molecule.resids) == 0:
        return 0
    return sum(1 for _ in molecule.residue_groups())


def _compactness(n_atoms: int, dims: np.ndarray) -> float:
    volume = float(np.prod(dims))
    return float(n_atoms / volume) if volume > 0.0 else 0.0


def _validate_chunk_size(chunk_size: int, name: str) -> int:
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError(f"{name} must be >= 1")
    return chunk_size


def _iter_blocks(coords: np.ndarray, chunk_size: int, start: int = 0):
    n = len(coords)
    for block_start in range(start, n, chunk_size):
        block_end = min(block_start + chunk_size, n)
        yield block_start, block_end, coords[block_start:block_end]


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2 = np.einsum("ij,ij->i", a, a)[:, None]
    b2 = np.einsum("ij,ij->i", b, b)[None, :]
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    return np.maximum(d2, 0.0)


def _validate_preset(preset: Optional[str], required: bool = False) -> Optional[str]:
    if preset is None:
        if required:
            raise ValueError(f"preset must be one of {', '.join(DESCRIPTOR_PRESETS)}")
        return None
    if preset not in DESCRIPTOR_PRESETS:
        choices = "', '".join(DESCRIPTOR_PRESETS)
        raise ValueError(f"unknown descriptor preset {preset!r}; expected '{choices}'")
    return preset


def _apply_preset(
    desc: dict,
    preset: Optional[str],
    elements_to_count,
    distance_bins: int,
    rdkit_prefix: str,
) -> dict:
    if preset is None:
        return desc
    names = _preset_top_level_names(preset, elements_to_count, distance_bins, rdkit_prefix)
    return {name: desc[name] for name in names if name in desc}


def _preset_top_level_names(
    preset: str,
    elements_to_count,
    distance_bins: int,
    rdkit_prefix: str,
) -> list[str]:
    names = _preset_scalar_names(preset, elements_to_count, rdkit_prefix)
    if preset == "native-3d":
        names += ["inertia_tensor", "principal_moments", "principal_axes", "distance_histogram"]
    return names


def _preset_scalar_names(preset: str, elements_to_count, rdkit_prefix: str) -> list[str]:
    names = [
        "n_atoms",
        "n_residues",
        "molecular_mass",
        *[f"count_{symbol.upper()}" for symbol in elements_to_count],
        "radius_of_gyration",
        "dim_x",
        "dim_y",
        "dim_z",
        "bbox_volume",
        "compactness",
        "bond_count",
        "bond_length_mean",
        "bond_length_std",
        "bond_length_min",
        "bond_length_max",
        "atom_contact_count",
        "atom_contact_density",
        "residue_contact_count",
        "residue_contact_density",
    ]
    if preset == "native-3d":
        names += [
            "shape_anisotropy",
            "asphericity",
            "acylindricity",
            "relative_shape_anisotropy",
            "sasa_total",
            "sasa_mean",
            "sasa_std",
            "sasa_max",
            "polar_contact_count",
            "salt_bridge_count",
        ]
    if preset == "rdkit-basic":
        names += [f"{rdkit_prefix}{name}" for name in RDKIT_BASIC_DESCRIPTORS]
    return names


def _rdkit_descriptors(molecule, names, prefix: str) -> dict[str, float]:
    from .chem import rdkit_descriptors

    return rdkit_descriptors(molecule, names=names, prefix=prefix)
