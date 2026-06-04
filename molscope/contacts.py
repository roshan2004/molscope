"""Inter-chain interfaces and ligand-binding-site contacts.

Two related protein-analysis tools, both built on the per-atom chain/residue
metadata and the ``hetero`` (ATOM vs HETATM) flag that ``Molecule`` carries:

* **Interfaces** -- which residues of one chain contact another chain
  (:func:`interface_residues`, :func:`chain_contact_matrix`).
* **Binding sites** -- which protein residues surround a ligand HETATM group
  (:func:`ligands`, :func:`binding_site`, :func:`select_pocket`), and a
  chemistry-aware natural-language description of the pocket for LLM / RAG
  prompts (:func:`analyze_pocket`, :meth:`Pocket.describe_environment`).

    mol.interface("A", "B")          # residues across the A/B interface
    mol.binding_site(cutoff=4.5)     # protein residues around the bound ligand
    mol.select_pocket(ligand="BEN").describe_environment()   # prompt-ready prose
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .molecule import Molecule, ResidueId

# Crystallographic solvent and common monatomic ions skipped when auto-detecting
# ligands (they are HETATM but rarely the "ligand" of interest).
WATER_RESNAMES = frozenset({"HOH", "WAT", "DOD", "H2O", "SOL", "TIP", "TIP3"})
ION_RESNAMES = frozenset({
    "NA", "K", "CL", "MG", "CA", "ZN", "FE", "MN", "CU", "NI", "CO", "CD",
    "HG", "BR", "IOD", "LI", "RB", "CS", "SR", "BA",
})
AMINO_ACID_RESNAMES = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
POCKET_DESCRIPTOR_PRESETS = ("pocket-basic",)

# -- semantic-description heuristics ----------------------------------------
# Pure-geometry element/residue tables used by describe_environment to translate
# a pocket into chemistry-aware prose. None of this is a force field: bonds are
# inferred from heavy-atom proximity, not from a topology or partial charges.

#: Heavy atoms that can donate or accept a hydrogen bond (hydrogens are usually
#: absent in crystal structures, so donor/acceptor is not distinguished).
HBOND_ELEMENTS = frozenset({"N", "O", "F"})

#: Apolar side chains that line a hydrophobic pocket wall.
HYDROPHOBIC_RESNAMES = frozenset({
    "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS",
})

#: Side chains with aromatic rings that can stack or cation-pi with a ligand.
AROMATIC_RESNAMES = frozenset({"PHE", "TYR", "TRP", "HIS"})

#: Side-chain atoms that carry a formal charge at neutral pH (textbook, not
#: pKa-aware): anchors the protein end of a salt bridge / electrostatic contact.
CATIONIC_ATOMS = frozenset({
    ("ARG", "NH1"), ("ARG", "NH2"), ("ARG", "NE"),
    ("LYS", "NZ"), ("HIS", "ND1"), ("HIS", "NE2"),
})
ANIONIC_ATOMS = frozenset({
    ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
})

#: Prose name for the charged group of each ionisable residue.
_CHARGED_GROUP_NAME = {
    "ASP": "carboxylate", "GLU": "carboxylate",
    "ARG": "guanidinium", "LYS": "ammonium", "HIS": "imidazole",
}


@dataclass(frozen=True)
class Residue:
    """A residue identity: chain id, residue number, insertion code, and name."""

    chain: str
    resid: int
    resname: str
    insertion_code: str = ""

    @property
    def icode(self) -> str:
        """Short alias for :attr:`insertion_code`."""
        return self.insertion_code

    @property
    def residue_id(self) -> ResidueId:
        """Return this residue as a :class:`molscope.molecule.ResidueId`."""
        return ResidueId(self.chain, self.resid, self.insertion_code, self.resname)

    def __repr__(self) -> str:
        return self.residue_id.label()


@dataclass(frozen=True)
class LigandResidue:
    """A HETATM group (ligand, cofactor, ion, or solvent) and its atom indices."""

    chain: str
    resid: int
    resname: str
    atom_indices: list[int]
    insertion_code: str = ""

    def __len__(self) -> int:
        return len(self.atom_indices)

    @property
    def icode(self) -> str:
        """Short alias for :attr:`insertion_code`."""
        return self.insertion_code

    @property
    def residue_id(self) -> ResidueId:
        """Return this ligand group as a :class:`molscope.molecule.ResidueId`."""
        return ResidueId(self.chain, self.resid, self.insertion_code, self.resname)

    def __repr__(self) -> str:
        return f"LigandResidue({self.residue_id.label()}, {len(self)} atoms)"


@dataclass
class Interface:
    """Residues and atom contacts across a two-chain interface."""

    chain_a: str
    chain_b: str
    cutoff: float
    residues_a: list[Residue]
    residues_b: list[Residue]
    contacts: list[tuple[int, int]]   # (atom index in chain_a, atom index in chain_b)

    @property
    def n_atom_contacts(self) -> int:
        return len(self.contacts)

    def __repr__(self) -> str:
        return (
            f"Interface({self.chain_a}-{self.chain_b}: "
            f"{len(self.residues_a)}+{len(self.residues_b)} residues, "
            f"{self.n_atom_contacts} atom contacts < {self.cutoff} A)"
        )


@dataclass
class ChainContactMatrix:
    """Symmetric counts of inter-chain atom contacts, labelled by chain id."""

    chains: list[str]
    matrix: np.ndarray                # (C, C) int counts; diagonal is 0

    def count(self, chain_a: str, chain_b: str) -> int:
        """Number of atom contacts between two chains."""
        return int(self.matrix[self.chains.index(chain_a), self.chains.index(chain_b)])


@dataclass
class BindingSite:
    """Protein residues surrounding a ligand, closest first.

    ``residues`` and ``min_distances`` are parallel lists ordered by increasing
    distance to the ligand; ``contacts`` are (protein atom, ligand atom) index
    pairs within ``cutoff``. ``residue_atom_indices`` is aligned with
    ``residues`` and contains all polymer atoms in each binding-site residue.
    """

    ligand: LigandResidue
    cutoff: float
    residues: list[Residue]
    min_distances: list[float]
    contacts: list[tuple[int, int]]
    residue_atom_indices: list[list[int]] = field(default_factory=list)

    @property
    def n_atom_contacts(self) -> int:
        """Number of protein-ligand atom pairs within ``cutoff``."""
        return len(self.contacts)

    @property
    def contact_atom_indices(self) -> list[int]:
        """Protein atoms that make at least one ligand contact."""
        return sorted({int(i) for i, _ in self.contacts})

    @property
    def protein_atom_indices(self) -> list[int]:
        """All polymer atoms in the binding-site residues.

        Sites created by older code may not carry ``residue_atom_indices``; in
        that case this falls back to the protein atoms that directly contact the
        ligand.
        """
        if not self.residue_atom_indices:
            return self.contact_atom_indices
        return sorted({int(i) for atoms in self.residue_atom_indices for i in atoms})

    @property
    def residue_contact_counts(self) -> list[int]:
        """Atom-contact counts aligned with ``residues``."""
        counts = [0] * len(self.residues)
        if not self.residue_atom_indices:
            return counts
        atom_to_residue = {
            int(atom): residue_i
            for residue_i, atoms in enumerate(self.residue_atom_indices)
            for atom in atoms
        }
        for protein_atom, _ in self.contacts:
            residue_i = atom_to_residue.get(int(protein_atom))
            if residue_i is not None:
                counts[residue_i] += 1
        return counts

    def to_records(self) -> list[dict[str, object]]:
        """Return table-friendly per-residue binding-site records."""
        contact_counts = self.residue_contact_counts
        return [
            {
                "residue_id": residue.residue_id.label(),
                "chain": residue.chain,
                "resid": residue.resid,
                "insertion_code": residue.insertion_code,
                "resname": residue.resname,
                "min_distance": float(distance),
                "n_atom_contacts": int(contact_counts[i]),
            }
            for i, (residue, distance) in enumerate(zip(self.residues, self.min_distances))
        ]

    def to_molecule(self, molecule: Molecule, include_ligand: bool = False) -> Molecule:
        """Return a subset molecule for the site residues, optionally with ligand atoms."""
        indices = self.protein_atom_indices
        if include_ligand:
            indices = sorted({*indices, *[int(i) for i in self.ligand.atom_indices]})
        if not indices:
            return molecule.take(np.array([], dtype=int))
        return molecule.take(indices)

    def descriptors(self, molecule: Molecule, preset: str = "pocket-basic") -> dict[str, float]:
        """Return fixed-size binding-pocket descriptors.

        The ``"pocket-basic"`` preset records pocket size, amino-acid
        composition, protein-ligand contact counts, binding-site residue
        dimensions, radius of gyration, and ligand-distance summaries.
        """
        preset = _validate_pocket_preset(preset)
        if preset == "pocket-basic":
            return _pocket_basic_descriptors(molecule, self)
        raise AssertionError("unreachable")

    def environment(self, molecule: Molecule, **thresholds) -> PocketEnvironment:
        """Analyse the pocket into chemistry-aware interaction features.

        Returns a :class:`PocketEnvironment` holding the hydrophobic wall,
        aromatic residues, hydrogen bonds, and salt-bridge / electrostatic
        contacts detected by pure-geometry heuristics. ``thresholds`` overrides
        the distance cut-offs (``hbond_cutoff``, ``salt_bridge_cutoff``,
        ``hydrophobic_cutoff``, ``aromatic_cutoff``). See
        :func:`analyze_pocket`.
        """
        return analyze_pocket(molecule, self, **thresholds)

    def describe_environment(self, molecule: Molecule, **thresholds) -> str:
        """Describe the pocket as a biochemist-style natural-language paragraph.

        Translates the binding pocket into prose suitable as LLM / RAG prompt
        context: the hydrophobic wall, aromatic residues, hydrogen bonds, and
        salt-bridge / electrostatic networks, with distances in angstrom.
        ``thresholds`` is forwarded to :meth:`environment`.

        The interactions are distance-only heuristics (see :func:`analyze_pocket`)
        and the prose is phrased accordingly ("likely", "possible"); confirm with
        a dedicated interaction profiler such as PLIP or ProLIF.
        """
        return self.environment(molecule, **thresholds).text()

    def plot(self, molecule: Molecule, include_ligand: bool = True, **kwargs):
        """Plot the binding-site residue subset.

        ``include_ligand`` defaults to ``True`` so ``site.plot(mol)`` shows the
        ligand context. Keyword arguments are forwarded to
        :meth:`molscope.Molecule.plot`.
        """
        kwargs.setdefault("color_by", "residue")
        return self.to_molecule(molecule, include_ligand=include_ligand).plot(**kwargs)

    def __repr__(self) -> str:
        return (
            f"BindingSite({self.ligand.residue_id.label()}: "
            f"{len(self.residues)} residues < {self.cutoff} A)"
        )


@dataclass(frozen=True)
class PocketInteraction:
    """A single detected ligand-pocket polar or charged interaction.

    ``kind`` is ``"hydrogen_bond"`` or ``"salt_bridge"``. ``ligand_atom`` and
    ``residue_atom`` are atom names (falling back to element symbols when the
    structure carries no atom names); ``distance`` is the heavy-atom separation
    in angstrom.
    """

    kind: str
    residue: Residue
    residue_atom: str
    ligand_atom: str
    distance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "residue": self.residue.residue_id.label(),
            "resname": self.residue.resname,
            "residue_atom": self.residue_atom,
            "ligand_atom": self.ligand_atom,
            "distance": round(float(self.distance), 2),
        }


@dataclass
class PocketEnvironment:
    """Chemistry-aware summary of a binding pocket, ready for LLM prompts.

    Produced by :func:`analyze_pocket`. :meth:`text` renders the biochemist-style
    paragraph; :meth:`to_dict` returns the structured findings for JSON / RAG use.
    """

    ligand: LigandResidue
    cutoff: float
    n_residues: int
    hydrophobic_residues: list[Residue]
    aromatic_residues: list[Residue]
    hydrogen_bonds: list[PocketInteraction]
    salt_bridges: list[PocketInteraction]

    def to_dict(self) -> dict[str, object]:
        return {
            "ligand": self.ligand.residue_id.label(),
            "cutoff": float(self.cutoff),
            "n_residues": int(self.n_residues),
            "hydrophobic_residues": [r.residue_id.label() for r in self.hydrophobic_residues],
            "aromatic_residues": [r.residue_id.label() for r in self.aromatic_residues],
            "hydrogen_bonds": [hb.to_dict() for hb in self.hydrogen_bonds],
            "salt_bridges": [sb.to_dict() for sb in self.salt_bridges],
        }

    def text(self) -> str:
        """Render the environment as a natural-language paragraph."""
        return _environment_prose(self)

    def __repr__(self) -> str:
        return (
            f"PocketEnvironment({self.ligand.residue_id.label()}: "
            f"{len(self.hydrophobic_residues)} hydrophobic, "
            f"{len(self.hydrogen_bonds)} H-bonds, "
            f"{len(self.salt_bridges)} salt bridges)"
        )


@dataclass(frozen=True)
class Pocket:
    """A binding pocket bound to its parent molecule.

    Returned by :meth:`molscope.Molecule.select_pocket`; pairs a
    :class:`BindingSite` with the :class:`~molscope.molecule.Molecule` it came
    from so the coordinate-dependent helpers can be called without re-passing the
    molecule, e.g. ``mol.select_pocket(ligand="BEN").describe_environment()``.
    """

    molecule: Molecule
    site: BindingSite

    @property
    def ligand(self) -> LigandResidue:
        return self.site.ligand

    @property
    def cutoff(self) -> float:
        return self.site.cutoff

    @property
    def residues(self) -> list[Residue]:
        return self.site.residues

    def environment(self, **thresholds) -> PocketEnvironment:
        """Analyse the pocket into :class:`PocketEnvironment` features."""
        return analyze_pocket(self.molecule, self.site, **thresholds)

    def describe_environment(self, **thresholds) -> str:
        """Biochemist-style natural-language description of the pocket."""
        return self.environment(**thresholds).text()

    def descriptors(self, preset: str = "pocket-basic") -> dict[str, float]:
        """Fixed-size pocket descriptors (see :meth:`BindingSite.descriptors`)."""
        return self.site.descriptors(self.molecule, preset=preset)

    def to_molecule(self, include_ligand: bool = False) -> Molecule:
        """Subset molecule for the pocket residues (see :meth:`BindingSite.to_molecule`)."""
        return self.site.to_molecule(self.molecule, include_ligand=include_ligand)

    def plot(self, include_ligand: bool = True, **kwargs):
        """Plot the pocket residue subset (see :meth:`BindingSite.plot`)."""
        return self.site.plot(self.molecule, include_ligand=include_ligand, **kwargs)

    def __repr__(self) -> str:
        return f"Pocket({self.site!r})"


# -- internals --------------------------------------------------------------


def _cross_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense ``(len(a), len(b))`` Euclidean distances between two atom sets."""
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def pocket_descriptor_feature_names(preset: str = "pocket-basic") -> list[str]:
    """Return stable feature names for a binding-pocket descriptor preset."""
    preset = _validate_pocket_preset(preset)
    if preset == "pocket-basic":
        return [
            "pocket_n_atoms",
            "pocket_n_residues",
            "ligand_n_atoms",
            "pocket_radius_of_gyration",
            "pocket_dim_x",
            "pocket_dim_y",
            "pocket_dim_z",
            "pocket_bbox_volume",
            "pocket_atom_contact_count",
            "pocket_contact_atom_count",
            "pocket_contact_residue_count",
            "pocket_contact_density",
            "ligand_distance_min",
            "ligand_distance_mean",
            "ligand_distance_std",
            "ligand_distance_max",
            "ligand_contact_distance_min",
            "ligand_contact_distance_mean",
            "ligand_contact_distance_std",
            "ligand_contact_distance_max",
            *[f"pocket_residue_count_{name}" for name in AMINO_ACID_RESNAMES],
            "pocket_residue_count_OTHER",
        ]
    raise AssertionError("unreachable")


def _validate_pocket_preset(preset: str) -> str:
    if preset not in POCKET_DESCRIPTOR_PRESETS:
        choices = "', '".join(POCKET_DESCRIPTOR_PRESETS)
        raise ValueError(f"unknown pocket descriptor preset {preset!r}; expected '{choices}'")
    return preset


def _pocket_basic_descriptors(molecule: Molecule, site: BindingSite) -> dict[str, float]:
    pocket = site.to_molecule(molecule)
    n_atoms = len(pocket)
    n_ligand_atoms = len(site.ligand)
    dims = pocket.dimensions if n_atoms else np.zeros(3, dtype=float)
    possible_contacts = n_atoms * n_ligand_atoms
    contact_distances = _contact_distances(molecule, site.contacts)
    residue_min_distances = np.asarray(site.min_distances, dtype=float)
    contact_counts = site.residue_contact_counts

    desc = {
        "pocket_n_atoms": float(n_atoms),
        "pocket_n_residues": float(len(site.residues)),
        "ligand_n_atoms": float(n_ligand_atoms),
        "pocket_radius_of_gyration": float(pocket.radius_of_gyration) if n_atoms else 0.0,
        "pocket_dim_x": float(dims[0]),
        "pocket_dim_y": float(dims[1]),
        "pocket_dim_z": float(dims[2]),
        "pocket_bbox_volume": float(np.prod(dims)) if n_atoms else 0.0,
        "pocket_atom_contact_count": float(site.n_atom_contacts),
        "pocket_contact_atom_count": float(len(site.contact_atom_indices)),
        "pocket_contact_residue_count": float(sum(count > 0 for count in contact_counts)),
        "pocket_contact_density": (
            float(site.n_atom_contacts / possible_contacts) if possible_contacts else 0.0
        ),
    }
    desc.update(_summary_stats("ligand_distance", residue_min_distances))
    desc.update(_summary_stats("ligand_contact_distance", contact_distances))

    counts = Counter(res.resname.upper() for res in site.residues)
    known = 0
    for name in AMINO_ACID_RESNAMES:
        value = counts.get(name, 0)
        desc[f"pocket_residue_count_{name}"] = float(value)
        known += value
    desc["pocket_residue_count_OTHER"] = float(max(0, len(site.residues) - known))
    return desc


def _summary_stats(prefix: str, values) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_min": float(values.min()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_max": float(values.max()),
    }


def _contact_distances(molecule: Molecule, contacts: list[tuple[int, int]]) -> np.ndarray:
    if not contacts:
        return np.empty(0, dtype=float)
    pairs = np.asarray(contacts, dtype=int).reshape(-1, 2)
    return np.linalg.norm(molecule.coords[pairs[:, 0]] - molecule.coords[pairs[:, 1]], axis=1)


def _require_residue_metadata(molecule: Molecule) -> None:
    if not molecule.chains:
        raise ValueError("this analysis needs chain information (read from PDB/mmCIF)")
    if len(molecule.resids) == 0:
        raise ValueError("this analysis needs residue information (read from PDB/mmCIF)")


def _unique_residues(molecule: Molecule, atom_indices) -> list[Residue]:
    """Ordered (by full residue id) unique residues covering the given atoms."""
    seen: dict[ResidueId, Residue] = {}
    for i in atom_indices:
        i = int(i)
        key = molecule.residue_id(i)
        if key not in seen:
            seen[key] = _residue_from_id(key)
    return [seen[k] for k in sorted(seen)]


def _hetero_groups(molecule: Molecule) -> list[LigandResidue]:
    """Every HETATM residue group, unfiltered."""
    if not molecule.hetero:
        return []
    hetero = molecule.hetero
    groups = []
    for group in molecule.residue_groups():
        het_idx = [int(i) for i in group.atom_indices if hetero[i]]
        if het_idx:
            rid = group.residue_id
            groups.append(
                LigandResidue(
                    rid.chain,
                    rid.resid,
                    rid.resname,
                    het_idx,
                    insertion_code=rid.insertion_code,
                )
            )
    return groups


def _residue_from_id(residue_id: ResidueId) -> Residue:
    return Residue(
        residue_id.chain,
        residue_id.resid,
        residue_id.resname,
        insertion_code=residue_id.insertion_code,
    )


def _ligand_selector_from_tuple(ligand: tuple) -> tuple[str, int, str | None, str | None]:
    chain, resid = ligand[0], ligand[1]
    icode = ligand[2] if len(ligand) >= 3 else None
    resname = ligand[3] if len(ligand) >= 4 else None
    return str(chain), int(resid), None if icode is None else str(icode), (
        None if not resname else str(resname)
    )


def _ligand_matches_residue_id(group: LigandResidue, residue_id: ResidueId) -> bool:
    return _ligand_matches_selector(
        group,
        (
            residue_id.chain,
            int(residue_id.resid),
            residue_id.insertion_code,
            residue_id.resname or None,
        ),
    )


def _ligand_matches_selector(
    group: LigandResidue,
    selector: tuple[str, int, str | None, str | None],
) -> bool:
    chain, resid, icode, resname = selector
    if group.chain != chain or int(group.resid) != int(resid):
        return False
    if icode is not None and group.insertion_code != icode:
        return False
    if resname is not None and group.resname.upper() != resname.upper():
        return False
    return True


def _selector_label(selector: tuple[str, int, str | None, str | None]) -> str:
    chain, resid, icode, resname = selector
    residue_number = f"{resid}{icode or ''}"
    name = f"{resname or 'HET'}{residue_number}"
    return f"{chain}:{name}" if chain else name


def _one_ligand_match(matches: list[LigandResidue], label: str) -> LigandResidue:
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"no HETATM group matching {label}")
    locs = ", ".join(g.residue_id.label() for g in matches)
    raise ValueError(
        f"ligand selector {label} matches multiple groups: {locs}; "
        "include insertion code or residue name"
    )


# -- interfaces -------------------------------------------------------------


def interface_residues(
    molecule: Molecule, chain_a: str, chain_b: str, cutoff: float = 5.0
) -> Interface:
    """Residues of ``chain_a`` and ``chain_b`` with atoms within ``cutoff`` (A)."""
    _require_residue_metadata(molecule)
    chains = np.asarray(molecule.chains)
    idx_a = np.nonzero(chains == chain_a)[0]
    idx_b = np.nonzero(chains == chain_b)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        raise ValueError(f"chains {chain_a!r} and/or {chain_b!r} not found")

    dist = _cross_distances(molecule.coords[idx_a], molecule.coords[idx_b])
    la, lb = np.nonzero(dist < cutoff)
    contacts = [(int(idx_a[i]), int(idx_b[j])) for i, j in zip(la, lb)]
    return Interface(
        chain_a, chain_b, cutoff,
        residues_a=_unique_residues(molecule, idx_a[np.unique(la)]),
        residues_b=_unique_residues(molecule, idx_b[np.unique(lb)]),
        contacts=contacts,
    )


def chain_contact_matrix(molecule: Molecule, cutoff: float = 5.0) -> ChainContactMatrix:
    """Symmetric matrix of inter-chain atom-contact counts (see :class:`ChainContactMatrix`)."""
    _require_residue_metadata(molecule)
    chain_list = molecule.chain_ids()
    chains = np.asarray(molecule.chains)
    coords = molecule.coords
    n = len(chain_list)
    mat = np.zeros((n, n), dtype=int)
    atom_idx = {c: np.nonzero(chains == c)[0] for c in chain_list}
    for a in range(n):
        for b in range(a + 1, n):
            dist = _cross_distances(coords[atom_idx[chain_list[a]]],
                                    coords[atom_idx[chain_list[b]]])
            mat[a, b] = mat[b, a] = int((dist < cutoff).sum())
    return ChainContactMatrix(list(chain_list), mat)


# -- binding sites ----------------------------------------------------------


def ligands(
    molecule: Molecule, exclude_water: bool = True, exclude_ions: bool = True
) -> list[LigandResidue]:
    """HETATM groups that look like ligands, skipping solvent/ions by default."""
    out = []
    for group in _hetero_groups(molecule):
        name = (group.resname or "").upper()
        if exclude_water and name in WATER_RESNAMES:
            continue
        if exclude_ions and name in ION_RESNAMES:
            continue
        out.append(group)
    return out


def _resolve_ligand(molecule: Molecule, ligand) -> LigandResidue:
    groups = _hetero_groups(molecule)
    if not groups:
        raise ValueError("no HETATM groups found; binding-site analysis needs a ligand")
    if ligand is None:
        candidates = ligands(molecule)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                "no non-solvent ligand detected; pass ligand=resname or (chain, resid)"
            )
        names = ", ".join(sorted({g.resname for g in candidates}))
        raise ValueError(
            f"multiple ligands present ({names}); specify ligand=resname or (chain, resid)"
        )
    if isinstance(ligand, LigandResidue):
        return ligand
    if isinstance(ligand, ResidueId) or isinstance(ligand, Residue):
        residue_id = ligand if isinstance(ligand, ResidueId) else ligand.residue_id
        matches = [g for g in groups if _ligand_matches_residue_id(g, residue_id)]
        return _one_ligand_match(matches, f"residue id {residue_id.label()}")
    if isinstance(ligand, tuple) and 2 <= len(ligand) <= 4:
        selector = _ligand_selector_from_tuple(ligand)
        matches = [g for g in groups if _ligand_matches_selector(g, selector)]
        label = _selector_label(selector)
        return _one_ligand_match(matches, label)
    matches = [g for g in groups if g.resname.upper() == str(ligand).upper()]
    if not matches:
        raise ValueError(f"no HETATM group with resname {ligand!r}")
    if len(matches) > 1:
        locs = ", ".join(g.residue_id.label() for g in matches)
        raise ValueError(
            f"resname {ligand!r} matches multiple groups: {locs}; "
            "pass (chain, resid[, insertion_code])"
        )
    return matches[0]


def binding_site(
    molecule: Molecule, ligand=None, cutoff: float = 4.5
) -> BindingSite:
    """Protein residues within ``cutoff`` (A) of a ligand (see :class:`BindingSite`).

    ``ligand`` selects the HETATM group: a resname (e.g. ``"BEN"``), a
    ``(chain, resid)`` pair, or a :class:`LigandResidue`. When omitted, the
    single non-solvent ligand is used (an error is raised if none or several).
    """
    _require_residue_metadata(molecule)
    target = _resolve_ligand(molecule, ligand)
    lig_idx = np.array(target.atom_indices, dtype=int)

    hetero = (
        np.array(molecule.hetero, dtype=bool)
        if molecule.hetero else np.zeros(len(molecule), bool)
    )
    prot_idx = np.nonzero(~hetero)[0]
    if len(prot_idx) == 0:
        raise ValueError("no polymer (ATOM) atoms to form a binding site")

    dist = _cross_distances(molecule.coords[prot_idx], molecule.coords[lig_idx])
    per_atom_min = dist.min(axis=1)
    close = dist < cutoff
    lp, ll = np.nonzero(close)
    contacts = [(int(prot_idx[i]), int(lig_idx[j])) for i, j in zip(lp, ll)]

    residue_atoms: dict[ResidueId, list[int]] = {}
    for group in molecule.residue_groups():
        atoms = [int(i) for i in group.atom_indices if not hetero[i]]
        if atoms:
            residue_atoms[group.residue_id] = atoms

    site_min: dict[ResidueId, float] = {}
    site_res: dict[ResidueId, Residue] = {}
    for local_i in np.unique(lp):
        gi = int(prot_idx[local_i])
        key = molecule.residue_id(gi)
        site_res[key] = _residue_from_id(key)
        site_min[key] = min(float(per_atom_min[local_i]), site_min.get(key, float("inf")))
    order = sorted(site_min, key=lambda k: site_min[k])
    return BindingSite(
        ligand=target, cutoff=cutoff,
        residues=[site_res[k] for k in order],
        min_distances=[site_min[k] for k in order],
        contacts=contacts,
        residue_atom_indices=[residue_atoms[k] for k in order],
    )


def select_pocket(molecule: Molecule, ligand=None, cutoff: float = 4.5) -> Pocket:
    """Select the binding pocket around a ligand as a :class:`Pocket`.

    A thin wrapper over :func:`binding_site` that keeps a reference to
    ``molecule`` so the coordinate-dependent helpers (notably
    :meth:`Pocket.describe_environment`) can be called without re-passing it.
    ``ligand`` and ``cutoff`` are forwarded to :func:`binding_site`.
    """
    return Pocket(molecule, binding_site(molecule, ligand=ligand, cutoff=cutoff))


# -- semantic pocket description --------------------------------------------


def _norm_element(value: str) -> str:
    return str(value).strip().upper()


def _atom_label(atom_names, element: str, index: int) -> str:
    """Atom name for prose, falling back to the element symbol when unnamed."""
    if atom_names:
        name = str(atom_names[index]).strip()
        if name:
            return name
    return element or "atom"


def _residue_tag(residue: Residue) -> str:
    """Compact ``RESNAME<resid><icode>`` tag, e.g. ``PHE70`` or ``ASP100A``."""
    return f"{residue.resname}{residue.resid}{residue.insertion_code}"


def _ligand_charge_sign(element: str, formal_charge) -> int:
    """Heuristic sign of a ligand atom's charge for electrostatic detection.

    Uses the formal charge when the structure carries one; otherwise treats a
    nitrogen as a potential cation (amine/ammonium) and an oxygen as a potential
    anion (carboxylate/hydroxyl), which is enough to pair against a formally
    charged protein side chain.
    """
    if formal_charge:
        return int(np.sign(formal_charge))
    if element == "N":
        return 1
    if element == "O":
        return -1
    return 0


def analyze_pocket(
    molecule: Molecule,
    site: BindingSite,
    *,
    hbond_cutoff: float = 3.5,
    salt_bridge_cutoff: float = 4.0,
    hydrophobic_cutoff: float = 4.5,
    aromatic_cutoff: float = 5.5,
) -> PocketEnvironment:
    """Translate a :class:`BindingSite` into :class:`PocketEnvironment` features.

    Pure-NumPy geometric heuristics over heavy-atom distances:

    * **Hydrogen bonds** -- N/O/F ligand atom within ``hbond_cutoff`` of an N/O/F
      protein atom (donor vs acceptor is not distinguished, as crystal hydrogens
      are usually absent).
    * **Salt bridges / electrostatic** -- a formally charged side-chain atom
      (Asp/Glu carboxylate, Arg/Lys/His cation) within ``salt_bridge_cutoff`` of a
      complementary-charge ligand atom.
    * **Hydrophobic wall** -- apolar side chains with a carbon within
      ``hydrophobic_cutoff`` of a ligand carbon.
    * **Aromatic residues** -- Phe/Tyr/Trp/His with a carbon within
      ``aromatic_cutoff`` of a ligand carbon (candidate pi-stacking).

    The closest contact per residue is kept for hydrogen bonds and salt bridges.

    These are distance-only heuristics: there is no donor/acceptor typing,
    hydrogen-bond angle criterion, or protonation-state model, so the results are
    candidates rather than verified interactions. For rigorous interaction
    profiling use a dedicated tool such as PLIP or ProLIF.
    """
    lig_idx = np.asarray(site.ligand.atom_indices, dtype=int)
    coords = molecule.coords
    elements = molecule.elements or [""] * len(molecule)
    atom_names = molecule.atom_names
    formal = molecule.formal_charges if len(molecule.formal_charges) else None

    lig_elems = [_norm_element(elements[i]) for i in lig_idx]
    lig_coords = coords[lig_idx]

    # Per-residue atom indices aligned with site.residues; fall back to the
    # contacting protein atoms grouped by residue for sites built without them.
    residue_atom_lists = site.residue_atom_indices
    if not residue_atom_lists or len(residue_atom_lists) != len(site.residues):
        residue_atom_lists = _residue_atoms_from_contacts(molecule, site)

    hydrophobic: list[Residue] = []
    aromatic: list[Residue] = []
    hydrogen_bonds: list[PocketInteraction] = []
    salt_bridges: list[PocketInteraction] = []

    for residue, atom_list in zip(site.residues, residue_atom_lists):
        resname = residue.resname.upper()
        atoms = np.asarray([int(i) for i in atom_list], dtype=int)
        if len(atoms) == 0:
            continue
        dist = _cross_distances(coords[atoms], lig_coords)   # (n_res_atoms, n_lig)
        res_elems = [_norm_element(elements[i]) for i in atoms]

        # Hydrophobic / aromatic walls: carbon-carbon proximity.
        carbon_res = np.array([e == "C" for e in res_elems], dtype=bool)
        carbon_lig = np.array([e == "C" for e in lig_elems], dtype=bool)
        if carbon_res.any() and carbon_lig.any():
            cc = dist[np.ix_(carbon_res, carbon_lig)]
            cc_min = float(cc.min())
            if resname in HYDROPHOBIC_RESNAMES and cc_min <= hydrophobic_cutoff:
                hydrophobic.append(residue)
            if resname in AROMATIC_RESNAMES and cc_min <= aromatic_cutoff:
                aromatic.append(residue)

        # Hydrogen bonds: closest polar-polar pair for this residue.
        hb = _closest_polar_pair(dist, res_elems, lig_elems, hbond_cutoff)
        if hb is not None:
            ai, lj, d = hb
            hydrogen_bonds.append(
                PocketInteraction(
                    "hydrogen_bond", residue,
                    _atom_label(atom_names, res_elems[ai], int(atoms[ai])),
                    _atom_label(atom_names, lig_elems[lj], int(lig_idx[lj])),
                    d,
                )
            )

        # Salt bridges: closest charged-complementary pair for this residue.
        sb = _closest_salt_bridge(
            dist, residue, res_elems, atoms, lig_elems, lig_idx,
            atom_names, formal, salt_bridge_cutoff,
        )
        if sb is not None:
            salt_bridges.append(sb)

    # A charged pair within H-bond range is geometrically also a hydrogen bond;
    # report it once, as the more specific salt bridge.
    bridged = {
        (sb.residue.residue_id, sb.residue_atom, sb.ligand_atom) for sb in salt_bridges
    }
    hydrogen_bonds = [
        hb for hb in hydrogen_bonds
        if (hb.residue.residue_id, hb.residue_atom, hb.ligand_atom) not in bridged
    ]

    hydrogen_bonds.sort(key=lambda x: x.distance)
    salt_bridges.sort(key=lambda x: x.distance)
    return PocketEnvironment(
        ligand=site.ligand,
        cutoff=site.cutoff,
        n_residues=len(site.residues),
        hydrophobic_residues=hydrophobic,
        aromatic_residues=aromatic,
        hydrogen_bonds=hydrogen_bonds,
        salt_bridges=salt_bridges,
    )


def _residue_atoms_from_contacts(molecule: Molecule, site: BindingSite) -> list[list[int]]:
    """Group the contacting protein atoms by residue, aligned with ``site.residues``."""
    by_residue: dict[ResidueId, list[int]] = {}
    for protein_atom, _ in site.contacts:
        by_residue.setdefault(molecule.residue_id(int(protein_atom)), []).append(int(protein_atom))
    return [by_residue.get(res.residue_id, []) for res in site.residues]


def _closest_polar_pair(dist, res_elems, lig_elems, cutoff):
    """Closest (protein atom, ligand atom) H-bond candidate within ``cutoff``."""
    res_polar = np.array([e in HBOND_ELEMENTS for e in res_elems], dtype=bool)
    lig_polar = np.array([e in HBOND_ELEMENTS for e in lig_elems], dtype=bool)
    if not res_polar.any() or not lig_polar.any():
        return None
    mask = res_polar[:, None] & lig_polar[None, :] & (dist <= cutoff)
    if not mask.any():
        return None
    masked = np.where(mask, dist, np.inf)
    ai, lj = np.unravel_index(int(np.argmin(masked)), masked.shape)
    return int(ai), int(lj), float(masked[ai, lj])


def _closest_salt_bridge(
    dist, residue, res_elems, atoms, lig_elems, lig_idx, atom_names, formal, cutoff,
):
    """Closest charged-complementary (protein, ligand) pair, as a PocketInteraction."""
    resname = residue.resname.upper()
    atom_names_seq = atom_names or []

    def protein_sign(local_i: int) -> int:
        name = (atom_names_seq[int(atoms[local_i])].strip() if atom_names_seq else "")
        if (resname, name) in CATIONIC_ATOMS:
            return 1
        if (resname, name) in ANIONIC_ATOMS:
            return -1
        return 0

    best = None
    for ai in range(len(res_elems)):
        p_sign = protein_sign(ai)
        if p_sign == 0:
            continue
        for lj in range(len(lig_elems)):
            d = float(dist[ai, lj])
            if d > cutoff:
                continue
            fc = int(formal[lig_idx[lj]]) if formal is not None else 0
            l_sign = _ligand_charge_sign(lig_elems[lj], fc)
            if l_sign != 0 and p_sign * l_sign < 0 and (best is None or d < best[2]):
                best = (ai, lj, d)
    if best is None:
        return None
    ai, lj, d = best
    return PocketInteraction(
        "salt_bridge", residue,
        _atom_label(atom_names, res_elems[ai], int(atoms[ai])),
        _atom_label(atom_names, lig_elems[lj], int(lig_idx[lj])),
        d,
    )


def _environment_prose(env: PocketEnvironment) -> str:
    """Render a :class:`PocketEnvironment` as a biochemist-style paragraph."""
    lig = env.ligand
    lig_name = lig.resname or "ligand"
    where = f" (chain {lig.chain})" if lig.chain else ""
    sentences = [
        f"The binding pocket around ligand {lig_name}{where} is lined by "
        f"{env.n_residues} residue{'s' if env.n_residues != 1 else ''} within "
        f"{env.cutoff:.1f} A of the ligand."
    ]

    if env.hydrophobic_residues:
        names = _join([_residue_tag(r) for r in env.hydrophobic_residues])
        sentences.append(f"A hydrophobic pocket wall appears to be formed by {names}.")

    if env.aromatic_residues:
        names = _join([_residue_tag(r) for r in env.aromatic_residues])
        verb = "rings" if len(env.aromatic_residues) > 1 else "ring"
        sentences.append(
            f"Aromatic {verb} from {names} may engage in pi-stacking or "
            f"cation-pi interactions with the ligand."
        )

    if env.hydrogen_bonds:
        parts = [
            f"the ligand {hb.ligand_atom} and {hb.residue_atom} of "
            f"{_residue_tag(hb.residue)} ({hb.distance:.1f} A)"
            for hb in env.hydrogen_bonds
        ]
        lead = "A likely hydrogen bond is suggested between" if len(parts) == 1 else (
            "Likely hydrogen bonds are suggested between"
        )
        sentences.append(f"{lead} {_join(parts)}.")

    if env.salt_bridges:
        parts = []
        for sb in env.salt_bridges:
            group = _CHARGED_GROUP_NAME.get(sb.residue.resname.upper(), "charged group")
            parts.append(
                f"the ligand {sb.ligand_atom} and the {group} of "
                f"{_residue_tag(sb.residue)} ({sb.distance:.1f} A)"
            )
        lead = (
            "A possible salt bridge / electrostatic contact is suggested between"
            if len(parts) == 1 else
            "Possible salt bridges / electrostatic contacts are suggested between"
        )
        sentences.append(f"{lead} {_join(parts)}.")

    if not (env.hydrogen_bonds or env.salt_bridges):
        sentences.append(
            "No close polar or charged contacts were detected; the pocket "
            "appears predominantly hydrophobic."
        )

    sentences.append(
        "(Contacts are inferred from heavy-atom distances only -- no "
        "donor/acceptor typing, bond-angle, or protonation-state analysis -- so "
        "treat them as candidates and confirm with a dedicated interaction "
        "profiler such as PLIP or ProLIF.)"
    )

    return " ".join(sentences)


def _join(items: list[str]) -> str:
    """Oxford-comma join: ``a``, ``a and b``, ``a, b and c``."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"
