"""Fixed-size molecular descriptors for quick ML feature tables."""

from __future__ import annotations

from collections import Counter
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
    centroid = molecule.centroid
    center_of_mass = molecule.center_of_mass
    desc.update({
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "center_of_mass_x": float(center_of_mass[0]),
        "center_of_mass_y": float(center_of_mass[1]),
        "center_of_mass_z": float(center_of_mass[2]),
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
        "centroid_x": 0.0,
        "centroid_y": 0.0,
        "centroid_z": 0.0,
        "center_of_mass_x": 0.0,
        "center_of_mass_y": 0.0,
        "center_of_mass_z": 0.0,
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
            "centroid_x",
            "centroid_y",
            "centroid_z",
            "center_of_mass_x",
            "center_of_mass_y",
            "center_of_mass_z",
            "shape_anisotropy",
            "asphericity",
            "acylindricity",
            "relative_shape_anisotropy",
        ]
    if preset == "rdkit-basic":
        names += [f"{rdkit_prefix}{name}" for name in RDKIT_BASIC_DESCRIPTORS]
    return names


def _rdkit_descriptors(molecule, names, prefix: str) -> dict[str, float]:
    from .chem import rdkit_descriptors

    return rdkit_descriptors(molecule, names=names, prefix=prefix)
