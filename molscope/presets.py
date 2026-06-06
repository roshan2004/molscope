"""A discoverable catalogue of MolScope's feature and mapping presets.

Presets are the short names you pass to descriptor, graph, and coarse-graining
APIs (``preset="native-3d"``, ``node_features="ml"``, ``mapping="martini"``).
They are convenient but easy to forget; :func:`list_presets` (and the
``molscope presets`` command) list every one with a description, the exact
feature names it expands to, and where to use it — so a newcomer can discover
the options without reading the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

CATEGORIES = ("descriptors", "graph", "coarse-grain")


@dataclass(frozen=True)
class PresetInfo:
    """One preset: its name, what it produces, and where it is used."""

    category: str  # one of CATEGORIES, the top-level group you filter by
    kind: str  # finer sub-group label, e.g. "graph node features"
    name: str  # the string you pass as the preset
    description: str
    used_by: str  # APIs / CLI flags that accept this preset
    # Flattened feature names the preset expands to, or ``None`` for presets
    # (such as coarse-grain mappings) that do not produce a feature vector.
    feature_names: Optional[list[str]] = None

    @property
    def n_features(self) -> Optional[int]:
        """Number of feature columns, or ``None`` for non-feature presets."""
        return None if self.feature_names is None else len(self.feature_names)

    def to_dict(self) -> dict:
        """JSON-serialisable view."""
        return {
            "category": self.category,
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "used_by": self.used_by,
            "n_features": self.n_features,
            "feature_names": list(self.feature_names)
            if self.feature_names is not None
            else None,
        }

    def __repr__(self) -> str:  # concise REPL view, not the verbose dataclass dump
        tail = f"{self.n_features} features" if self.feature_names is not None else "mapping"
        return f"PresetInfo({self.category}/{self.name}: {tail})"


def list_presets(category: Optional[str] = None) -> list[PresetInfo]:
    """Return MolScope's presets, optionally filtered to one ``category``.

    ``category`` is one of ``"descriptors"``, ``"graph"``, ``"coarse-grain"`` (or
    ``None`` for all). Each :class:`PresetInfo` carries the preset name, a short
    description, the APIs that accept it, and the feature names it expands to
    (``None`` for coarse-grain mappings). Enumerating names needs no optional
    backends, so this works on the bare NumPy install even for ``rdkit-basic``.

    >>> import molscope as ms
    >>> [p.name for p in ms.list_presets("descriptors")]
    ['native-basic', 'native-3d', 'rdkit-basic', 'pocket-basic']
    """
    if category is not None and category not in CATEGORIES:
        choices = "', '".join(CATEGORIES)
        raise ValueError(f"unknown preset category {category!r}; expected '{choices}'")
    presets = _build_registry()
    if category is not None:
        presets = [p for p in presets if p.category == category]
    return presets


def format_presets(presets: list[PresetInfo], *, show_features: bool = False) -> str:
    """Render presets as human-readable text, grouped by kind."""
    if not presets:
        return "(no presets)"
    lines: list[str] = []
    last_kind = None
    for p in presets:
        if p.kind != last_kind:
            if last_kind is not None:
                lines.append("")
            lines.append(f"{p.kind}  —  {p.used_by}")
            last_kind = p.kind
        count = ""
        if p.n_features is not None:
            count = f"  [{p.n_features} feature{'' if p.n_features == 1 else 's'}]"
        lines.append(f"  {p.name:<16} {p.description}{count}")
        if show_features and p.feature_names:
            lines.append(f"      {', '.join(p.feature_names)}")
    return "\n".join(lines)


# -- registry ---------------------------------------------------------------

# Short descriptions, keyed by (kind, name). The feature names themselves come
# from the canonical *_feature_names functions so this module never drifts from
# what the presets actually produce.
_DESCRIPTIONS = {
    ("molecule descriptors", "native-basic"):
        "Composition and geometry from coordinates alone: atom/residue counts, "
        "element counts, size, shape, bonds, contacts.",
    ("molecule descriptors", "native-3d"):
        "native-basic plus 3D shape: centre of mass, inertia tensor, principal "
        "moments/axes, shape anisotropy, SASA, and a pairwise-distance histogram.",
    ("molecule descriptors", "rdkit-basic"):
        "Cheminformatics descriptors via RDKit (MolWt, TPSA, LogP, H-bond "
        "donors/acceptors, rotatable bonds, rings, FractionCSP3). Needs the 'chem' extra.",
    ("binding-pocket descriptors", "pocket-basic"):
        "Binding-pocket geometry and composition: pocket/ligand atom counts, "
        "pocket size, ligand-distance stats, and per-residue-type counts.",
    ("graph node features", "default"):
        "Atomic number and mass.",
    ("graph node features", "basic"):
        "Atomic number, mass, and formal charge.",
    ("graph node features", "ml"):
        "One-hot element plus atomic number, mass, formal charge, and aromatic flag.",
    ("graph edge features", "default"):
        "Interatomic distance.",
    ("graph edge features", "basic"):
        "Distance and bond order.",
    ("graph edge features", "ml"):
        "Distance, bond order, and aromatic flag.",
    ("graph edge features", "geom"):
        "Distance, bond order, aromatic flag, bond angle, and dihedral.",
    ("residue-graph node features", "default"):
        "Residue size (atom count).",
    ("residue-graph node features", "ml"):
        "One-hot residue type plus residue size.",
    ("residue-graph edge features", "default"):
        "Inter-residue distance.",
    ("residue-graph edge features", "ml"):
        "Inter-residue distance plus per-method contact counts.",
    ("bead mapping", "residue_com"):
        "One bead per residue at its centre of mass.",
    ("bead mapping", "residue_centroid"):
        "One bead per residue at its (unweighted) geometric centroid.",
    ("bead mapping", "martini"):
        "Simplified backbone + side-chain (BB/SC) bead model.",
}


def _build_registry() -> list[PresetInfo]:
    from .coarsegrain import COARSE_GRAIN_MAPPINGS
    from .contacts import POCKET_DESCRIPTOR_PRESETS, pocket_descriptor_feature_names
    from .descriptors import DESCRIPTOR_PRESETS, descriptor_feature_names
    from .graph import (
        GRAPH_EDGE_FEATURE_PRESETS,
        GRAPH_NODE_FEATURE_PRESETS,
        RESIDUE_EDGE_FEATURE_PRESETS,
        RESIDUE_NODE_FEATURE_PRESETS,
        edge_feature_names,
        node_feature_names,
        residue_edge_feature_names,
        residue_node_feature_names,
    )

    out: list[PresetInfo] = []

    def add(category, kind, used_by, names, namer):
        for name in names:
            out.append(PresetInfo(
                category=category, kind=kind, name=name,
                description=_DESCRIPTIONS[(kind, name)], used_by=used_by,
                feature_names=None if namer is None else list(namer(name)),
            ))

    add("descriptors", "molecule descriptors",
        "ms.descriptors(mol, preset=…), ms.featurize_many(…); CLI: molscope analyze --preset",
        DESCRIPTOR_PRESETS, descriptor_feature_names)
    add("descriptors", "binding-pocket descriptors",
        "site.descriptors(mol, preset=…); ms.pocket_descriptor_feature_names(preset)",
        POCKET_DESCRIPTOR_PRESETS, pocket_descriptor_feature_names)

    add("graph", "graph node features",
        "ms.build_dataset(node_features=…), MolecularGraph.node_features(preset=…)",
        GRAPH_NODE_FEATURE_PRESETS, node_feature_names)
    add("graph", "graph edge features",
        "ms.build_dataset(edge_features=…), MolecularGraph.edge_features(preset=…)",
        GRAPH_EDGE_FEATURE_PRESETS, edge_feature_names)
    add("graph", "residue-graph node features",
        "ResidueContactGraph.node_features(preset=…)",
        RESIDUE_NODE_FEATURE_PRESETS, residue_node_feature_names)
    add("graph", "residue-graph edge features",
        "ResidueContactGraph.edge_features(preset=…)",
        RESIDUE_EDGE_FEATURE_PRESETS, residue_edge_feature_names)

    add("coarse-grain", "bead mapping",
        "mol.coarse_grain(mapping=…); CLI: molscope coarse-grain --mapping",
        COARSE_GRAIN_MAPPINGS, None)

    return out
