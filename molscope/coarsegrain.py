"""Coarse-graining: map an atomistic structure onto a smaller set of beads.

The result is an ordinary :class:`~molscope.molecule.Molecule` whose "atoms"
are beads, so it plots, transforms and exports to a graph like any other. Bead
positions are the mass-weighted centre (or geometric centroid) of their member
atoms, and explicit CG bonds are attached.

Built-in modes:

- ``"residue_com"``      — one bead per residue at its centre of mass
- ``"residue_centroid"`` — one bead per residue at its geometric centroid
- ``"martini"``          — a simplified backbone + side-chain (BB/SC) model

Custom mappings come in two forms:

- **By residue/atom name** (needs PDB/mmCIF metadata)::

      {"ALA": {"BB": ["N", "CA", "C", "O"], "SC": ["CB"]}}

  Bonds are generated automatically (sequential within a residue, plus a
  backbone chain between consecutive residues) unless you pass ``bonds=``.

- **By atom index** (works on any structure, including ``.xyz``)::

      {"head": [0, 1, 2, 3], "tail": [4, 5, 6, 7]}

  No bonds are added unless you pass ``bonds=`` (see below).

``bonds`` lets you define the bead network explicitly as pairs of bead indices,
or by bead name when names are unique, e.g. ``bonds=[("head", "tail")]`` or
``bonds=[(0, 1)]``. Repeated names such as ``BB``/``SC`` in Martini-like residue
mappings are ambiguous; use bead indices for those.

``virtual_sites`` lets you append coordinate sites derived from existing beads
without treating them as ordinary atom-assignment beads, e.g.
``virtual_sites=[{"name": "MID", "parents": [0, 2]}]``. Supported construction
rules are ``"weighted_average"`` and ``"linear_combination"``.

Intended for teaching and prototyping CG mappings, not as a substitute for
production Martini parameters.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import elements
from .molecule import Molecule

_MAPPING_FORMAT = "molscope-cg-mapping"
_MAPPING_VERSION = 2

# Built-in named bead mappings accepted by :func:`coarse_grain` (a custom dict
# mapping may be passed instead). The single source of truth for the CLI
# ``--mapping`` choices and the preset catalogue.
COARSE_GRAIN_MAPPINGS = ("residue_com", "residue_centroid", "martini")

_BACKBONE = ("N", "CA", "C", "O", "OXT")


@dataclass(frozen=True)
class BeadMapping:
    """One coarse-grained bead and the atoms assigned to it.

    ``atom_indices`` are positions into the *source* (atomistic) molecule, in
    the same order as ``atom_names``. They drive mapping visualisation and
    export; ``atom_names`` stays the human-readable view.
    """

    name: str
    atom_names: list[str]
    reduction: str
    resname: str = ""
    resid: Optional[int] = None
    chain: str = ""
    atom_indices: list[int] = field(default_factory=list)
    insertion_code: str = ""


@dataclass(frozen=True)
class DroppedAtom:
    """An atom omitted by a custom coarse-graining mapping."""

    name: str
    element: str
    resname: str = ""
    resid: Optional[int] = None
    chain: str = ""
    insertion_code: str = ""


@dataclass(frozen=True)
class BondMapping:
    """One generated or user-defined CG bond."""

    a: str
    b: str
    reason: str


@dataclass(frozen=True)
class VirtualSiteMapping:
    """One coordinate site derived from parent CG beads.

    Virtual sites are present in the CG molecule's coordinates, but are not
    ordinary beads: they do not fold source atoms into a new assignment group.
    ``parents`` are CG bead indices from the non-virtual bead list.
    """

    index: int
    name: str
    parents: list[int]
    parent_names: list[str]
    rule: str
    weights: list[float]
    resname: str = ""
    resid: Optional[int] = None
    chain: str = ""
    insertion_code: str = ""


@dataclass(frozen=True)
class CoarseGrainReport:
    """Human-readable explanation of a coarse-graining operation."""

    mapping: str
    beads: list[BeadMapping] = field(default_factory=list)
    dropped_atoms: list[DroppedAtom] = field(default_factory=list)
    bonds: list[BondMapping] = field(default_factory=list)
    virtual_sites: list[VirtualSiteMapping] = field(default_factory=list)

    @property
    def n_beads(self) -> int:
        """Number of coarse-grained beads."""
        return len(self.beads)

    @property
    def n_virtual_sites(self) -> int:
        """Number of coordinate sites derived from CG beads."""
        return len(self.virtual_sites)

    @property
    def n_sites(self) -> int:
        """Total number of CG coordinate sites, including virtual sites."""
        return self.n_beads + self.n_virtual_sites

    @property
    def n_assigned(self) -> int:
        """Number of atoms folded into a bead."""
        return sum(len(bead.atom_indices) for bead in self.beads)

    @property
    def n_dropped(self) -> int:
        """Number of atoms left unassigned (and dropped)."""
        return len(self.dropped_atoms)

    def __str__(self) -> str:
        return self.format()

    def coverage(self) -> str:
        """One-line summary of how much of the structure the mapping covered."""
        total = self.n_assigned + self.n_dropped
        beads = f"{self.n_beads} bead{'' if self.n_beads == 1 else 's'}"
        atoms = f"{self.n_assigned}/{total} atom{'' if total == 1 else 's'}"
        line = f"{beads} from {atoms}"
        if self.n_virtual_sites:
            plural = "" if self.n_virtual_sites == 1 else "s"
            line += f" + {self.n_virtual_sites} virtual site{plural}"
        if self.n_dropped:
            line += f" ({self.n_dropped} dropped)"
        return line

    def format(self) -> str:
        lines = [f"Mapping: {self.mapping}", self.coverage(), "", "Beads:"]
        if self.beads:
            for bead in self.beads:
                prefix = _residue_label(
                    bead.resid, bead.resname, bead.chain, bead.insertion_code
                )
                atoms = ", ".join(bead.atom_names) if bead.atom_names else "(none)"
                lines.append(f"  {prefix}:")
                lines.append(f"    {bead.name} bead: {atoms} -> {bead.reduction}")
        else:
            lines.append("  (none)")

        if self.virtual_sites:
            lines.extend(["", "Virtual sites:"])
            for site in self.virtual_sites:
                parents = ", ".join(
                    f"{name}#{idx}" for idx, name in zip(site.parents, site.parent_names)
                )
                weights = ", ".join(f"{w:g}" for w in site.weights)
                prefix = _residue_label(
                    site.resid, site.resname, site.chain, site.insertion_code
                )
                lines.append(f"  {prefix}:")
                lines.append(
                    f"    {site.name} site: {site.rule}({parents}; weights={weights})"
                )

        lines.extend(["", "Dropped atoms:"])
        if self.dropped_atoms:
            for atom in self.dropped_atoms:
                label = _residue_label(
                    atom.resid, atom.resname, atom.chain, atom.insertion_code
                )
                name = atom.name or atom.element or "(unnamed)"
                lines.append(f"  {label}: {name}")
        else:
            lines.append("  (none)")

        lines.extend(["", "Generated bonds:"])
        if self.bonds:
            for bond in self.bonds:
                lines.append(f"  {bond.a}-{bond.b} {bond.reason}")
        else:
            lines.append("  (none)")
        return "\n".join(lines)


def coarse_grain(molecule: Molecule, mapping="residue_com", weighted: bool = True,
                 bonds=None, virtual_sites=None, return_report: bool = False):
    """Coarse-grain ``molecule``; see the module docstring for the options."""
    if _is_index_mapping(mapping):
        cg, report = _by_index(molecule, mapping, weighted, bonds, virtual_sites)
        return (cg, report) if return_report else cg

    if len(molecule.resids) == 0:
        raise ValueError(
            "coarse-graining by residue needs residue information; for a file "
            "without it (e.g. .xyz) use an index mapping {bead: [atom_indices]}"
        )
    cg, report = _by_residue(molecule, mapping, weighted, bonds, virtual_sites)
    return (cg, report) if return_report else cg


def _is_index_mapping(mapping) -> bool:
    """An index mapping is a dict whose values are index lists, not sub-dicts."""
    if not isinstance(mapping, dict) or not mapping:
        return False
    return not isinstance(next(iter(mapping.values())), dict)


def _by_index(molecule: Molecule, mapping: dict, weighted: bool, bonds, virtual_sites) -> Molecule:
    bead_names: list[str] = []
    bead_coords: list[np.ndarray] = []
    bead_report: list[BeadMapping] = []
    assigned: set[int] = set()
    for name, members in mapping.items():
        try:
            members = [int(i) for i in members]
        except (TypeError, ValueError):
            raise ValueError(
                f"bead {name!r}: an index mapping expects integer atom indices. "
                "For atom-name beads use a residue mapping {resname: {bead: [names]}}."
            ) from None
        if not members:
            continue
        assigned.update(members)
        bead_coords.append(_reduce(molecule, members, weighted))
        bead_names.append(name)
        bead_report.append(
            BeadMapping(
                name=name,
                atom_names=_atom_names(molecule, members),
                reduction=_reduction_name(weighted),
                atom_indices=[int(i) for i in members],
            )
        )

    if not bead_coords:
        raise ValueError("mapping produced no beads")
    dropped = _dropped_atoms(molecule, assigned)
    _warn_dropped(len(molecule), assigned)
    bond_index, bond_report = _resolve_bonds(bonds, bead_names)
    virtual_report = _append_virtual_sites(
        bead_coords,
        bead_names,
        virtual_sites,
    )
    report = CoarseGrainReport(
        mapping="index",
        beads=bead_report,
        dropped_atoms=dropped,
        bonds=bond_report,
        virtual_sites=virtual_report,
    )
    virtual_flags = _virtual_site_flags(len(bead_coords), virtual_report)
    cg = Molecule(
        np.array(bead_coords, dtype=float), elements=[""] * len(bead_coords),
        name=f"{molecule.name} (CG)", atom_names=bead_names,
        bond_index=bond_index, virtual_sites=virtual_flags, _mapping_report=report,
    )
    return cg, report


def _by_residue(molecule: Molecule, mapping, weighted: bool, bonds, virtual_sites) -> Molecule:
    use_mass = weighted and mapping != "residue_centroid"
    bead_coords, bead_names = [], []
    bead_resnames, bead_resids, bead_icodes, bead_chains = [], [], [], []
    residue_beads: list[list[int]] = []
    bead_report: list[BeadMapping] = []
    assigned: set[int] = set()

    for group in molecule.residue_groups():
        atom_idx = group.atom_indices
        resname = group.resname
        resid = group.resid
        chain = group.chain
        icode = group.insertion_code
        local = []
        for bead_name, members in _residue_beads(molecule, atom_idx, resname, mapping):
            if not members:
                continue
            assigned.update(members)
            bead_coords.append(_reduce(molecule, members, use_mass))
            bead_names.append(bead_name)
            bead_resnames.append(resname)
            bead_resids.append(resid)
            bead_icodes.append(icode)
            bead_chains.append(chain)
            local.append(len(bead_coords) - 1)
            bead_report.append(
                BeadMapping(
                    name=bead_name,
                    atom_names=_atom_names(molecule, members),
                    reduction=_reduction_name(use_mass),
                    resname=resname,
                    resid=resid,
                    chain=chain,
                    atom_indices=[int(i) for i in members],
                    insertion_code=icode,
                )
            )
        if local:
            residue_beads.append(local)

    if not bead_coords:
        raise ValueError("mapping produced no beads")
    dropped = _dropped_atoms(molecule, assigned)
    if isinstance(mapping, dict):  # only custom mappings can leave atoms unassigned
        _warn_dropped(len(molecule), assigned)

    if bonds is not None:
        bond_index, bond_report = _resolve_bonds(bonds, bead_names)
    else:
        bond_index, bond_report = _cg_bonds(residue_beads, bead_chains, bead_names)
    virtual_report = _append_virtual_sites(
        bead_coords,
        bead_names,
        virtual_sites,
        bead_resnames,
        bead_resids,
        bead_icodes,
        bead_chains,
    )
    report = CoarseGrainReport(
        mapping=_mapping_name(mapping),
        beads=bead_report,
        dropped_atoms=dropped if isinstance(mapping, dict) or mapping == "martini" else [],
        bonds=bond_report,
        virtual_sites=virtual_report,
    )
    virtual_flags = _virtual_site_flags(len(bead_coords), virtual_report)
    cg = Molecule(
        np.array(bead_coords, dtype=float), elements=[""] * len(bead_coords),
        name=f"{molecule.name} (CG)", atom_names=bead_names, resnames=bead_resnames,
        resids=np.array(bead_resids, dtype=int), icodes=bead_icodes, chains=bead_chains,
        bond_index=bond_index, virtual_sites=virtual_flags, _mapping_report=report,
    )
    return cg, report


def _residue_beads(molecule: Molecule, atom_idx, resname, mapping):
    """Return ``[(bead_name, [atom_index, ...]), ...]`` for one residue."""
    if mapping in ("residue_com", "residue_centroid"):
        return [(resname or "BEAD", atom_idx)]
    if mapping == "martini":
        return _backbone_sidechain(molecule, atom_idx)

    spec = mapping.get(resname)
    if spec is None:
        warnings.warn(
            f"no mapping for residue {resname!r}; collapsing it to one bead",
            stacklevel=4,
        )
        return [(resname or "BEAD", atom_idx)]
    names = {molecule.atom_names[i]: i for i in atom_idx}
    return [
        (bead, [names[a] for a in atoms if a in names])
        for bead, atoms in spec.items()
    ]


def _backbone_sidechain(molecule: Molecule, atom_idx):
    """Simplified Martini-like split: a backbone bead and a side-chain bead."""
    bb = [i for i in atom_idx if molecule.atom_names[i] in _BACKBONE]
    sc = [
        i for i in atom_idx
        if molecule.atom_names[i] not in _BACKBONE and molecule.elements[i] != "H"
    ]
    beads = []
    if bb:
        beads.append(("BB", bb))
    if sc:
        beads.append(("SC", sc))
    return beads


def _reduce(molecule: Molecule, members, use_mass: bool) -> np.ndarray:
    coords = molecule.coords[members]
    if use_mass:
        w = np.array([elements.mass(molecule.elements[i]) for i in members])
        return (w[:, None] * coords).sum(axis=0) / w.sum()
    return coords.mean(axis=0)


def _append_virtual_sites(
    bead_coords: list[np.ndarray],
    bead_names: list[str],
    virtual_sites,
    bead_resnames: Optional[list[str]] = None,
    bead_resids: Optional[list[int]] = None,
    bead_icodes: Optional[list[str]] = None,
    bead_chains: Optional[list[str]] = None,
) -> list[VirtualSiteMapping]:
    """Append virtual-site coordinates to the CG coordinate/name lists."""
    if not virtual_sites:
        return []

    base_count = len(bead_coords)
    if base_count == 0:
        raise ValueError("virtual sites require at least one parent bead")

    name_to_idx = _unique_name_index(bead_names)
    report: list[VirtualSiteMapping] = []
    for raw in virtual_sites:
        spec = _normalise_virtual_site_spec(raw)
        parents = [
            _resolve_virtual_parent(parent, bead_names, name_to_idx, base_count)
            for parent in spec["parents"]
        ]
        weights = _virtual_site_weights(spec.get("weights"), len(parents), spec["rule"])
        parent_coords = np.asarray([bead_coords[i] for i in parents], dtype=float)
        coord = weights @ parent_coords

        first = parents[0]
        resname = spec.get("resname")
        if resname is None and bead_resnames is not None:
            resname = bead_resnames[first]
        resid = spec.get("resid")
        if resid is None and bead_resids is not None:
            resid = bead_resids[first]
        icode = spec.get("insertion_code")
        if icode is None and bead_icodes is not None:
            icode = bead_icodes[first]
        chain = spec.get("chain")
        if chain is None and bead_chains is not None:
            chain = bead_chains[first]

        index = len(bead_coords)
        bead_coords.append(coord)
        bead_names.append(spec["name"])
        if bead_resnames is not None:
            bead_resnames.append(resname or "")
        if bead_resids is not None:
            bead_resids.append(0 if resid is None else int(resid))
        if bead_icodes is not None:
            bead_icodes.append(icode or "")
        if bead_chains is not None:
            bead_chains.append(chain or "")
        report.append(
            VirtualSiteMapping(
                index=index,
                name=spec["name"],
                parents=parents,
                parent_names=[bead_names[i] for i in parents],
                rule=spec["rule"],
                weights=[float(w) for w in weights],
                resname=resname or "",
                resid=None if resid is None else int(resid),
                chain=chain or "",
                insertion_code=icode or "",
            )
        )
    return report


def _normalise_virtual_site_spec(spec) -> dict:
    if isinstance(spec, dict):
        name = spec.get("name")
        parents = spec.get("parents")
        rule = spec.get("rule", "weighted_average")
        weights = spec.get("weights")
        out = {
            "name": name,
            "parents": parents,
            "rule": rule,
            "weights": weights,
            "resname": spec.get("resname"),
            "resid": spec.get("resid"),
            "insertion_code": spec.get("insertion_code", spec.get("icode")),
            "chain": spec.get("chain"),
        }
    else:
        try:
            name, parents = spec
        except (TypeError, ValueError):
            raise ValueError(
                "virtual sites must be dicts or (name, parents) pairs"
            ) from None
        out = {
            "name": name,
            "parents": parents,
            "rule": "weighted_average",
            "weights": None,
            "resname": None,
            "resid": None,
            "insertion_code": None,
            "chain": None,
        }

    if not out["name"]:
        raise ValueError("virtual site requires a name")
    if out["parents"] is None:
        raise ValueError(f"virtual site {out['name']!r} requires parents")
    out["parents"] = list(out["parents"])
    if not out["parents"]:
        raise ValueError(f"virtual site {out['name']!r} needs at least one parent")
    if out["rule"] not in {"weighted_average", "linear_combination"}:
        raise ValueError(
            f"virtual site {out['name']!r}: unsupported rule {out['rule']!r}"
        )
    return out


def _resolve_virtual_parent(parent, bead_names, name_to_idx, base_count: int) -> int:
    if isinstance(parent, str):
        if parent in name_to_idx:
            return name_to_idx[parent]
        if parent in bead_names:
            raise ValueError(
                f"virtual-site parent bead name {parent!r} is repeated; use bead indices"
            )
        raise ValueError(f"unknown virtual-site parent bead {parent!r}")
    idx = int(parent)
    if idx < 0 or idx >= base_count:
        raise ValueError(
            f"virtual-site parent index {idx} is out of range for {base_count} beads"
        )
    return idx


def _virtual_site_weights(weights, n_parents: int, rule: str) -> np.ndarray:
    if weights is None:
        weights = np.ones(n_parents, dtype=float) / n_parents
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
    if len(weights) != n_parents:
        raise ValueError(f"{len(weights)} virtual-site weights for {n_parents} parents")
    if rule == "weighted_average":
        total = float(weights.sum())
        if total == 0.0:
            raise ValueError("weighted_average virtual-site weights must not sum to zero")
        weights = weights / total
    return weights


def _virtual_site_flags(n_sites: int, virtual_sites: list[VirtualSiteMapping]) -> np.ndarray:
    if not virtual_sites:
        return np.empty(0, dtype=bool)
    flags = np.zeros(n_sites, dtype=bool)
    for site in virtual_sites:
        flags[site.index] = True
    return flags


def _resolve_bonds(bonds, bead_names):
    """Turn user bond pairs (bead names or indices) into an (E, 2) index array."""
    if bonds is None:
        return None, []
    name_to_idx = _unique_name_index(bead_names)
    pairs = []
    report = []
    for a, b in bonds:
        ai = _resolve_bond_endpoint(a, bead_names, name_to_idx)
        bi = _resolve_bond_endpoint(b, bead_names, name_to_idx)
        pairs.append((ai, bi))
        report.append(BondMapping(bead_names[ai], bead_names[bi], "(user-defined)"))
    return np.array(pairs, dtype=int).reshape(-1, 2), report


def _resolve_bond_endpoint(endpoint, bead_names, name_to_idx) -> int:
    if not isinstance(endpoint, str):
        return int(endpoint)
    if endpoint in name_to_idx:
        return name_to_idx[endpoint]
    if endpoint in bead_names:
        raise ValueError(
            f"bead name {endpoint!r} is repeated and cannot identify one bead; "
            "use bead indices for user-defined bonds"
        )
    raise ValueError(f"unknown bead name {endpoint!r}")


def _unique_name_index(bead_names):
    """Map unique bead names to indices; repeated names cannot identify one bead."""
    counts = {}
    for name in bead_names:
        counts[name] = counts.get(name, 0) + 1

    name_to_idx = {}
    for i, name in enumerate(bead_names):
        if counts[name] == 1:
            name_to_idx[name] = i
    return name_to_idx


def _cg_bonds(residue_beads, bead_chains, bead_names):
    """Bonds within each residue (sequential) plus a chain between residues."""
    bonds: list[tuple[int, int]] = []
    report: list[BondMapping] = []
    for beads in residue_beads:
        for a, b in zip(beads, beads[1:]):
            bonds.append((a, b))
            report.append(BondMapping(bead_names[a], bead_names[b], "within residue"))
    for prev, curr in zip(residue_beads, residue_beads[1:]):
        if bead_chains[prev[0]] == bead_chains[curr[0]]:
            bonds.append((prev[0], curr[0]))
            report.append(
                BondMapping(bead_names[prev[0]], bead_names[curr[0]], "between residues")
            )
    return np.array(bonds, dtype=int).reshape(-1, 2), report


def _warn_dropped(n_atoms: int, assigned: set) -> None:
    dropped = n_atoms - len(assigned)
    if dropped:
        warnings.warn(
            f"{dropped} atom(s) were not assigned to any bead and were dropped",
            stacklevel=3,
        )


def _atom_names(molecule: Molecule, atom_indices) -> list[str]:
    names = []
    for i in atom_indices:
        if molecule.atom_names:
            names.append(molecule.atom_names[i])
        elif molecule.elements:
            names.append(molecule.elements[i])
        else:
            names.append(str(i))
    return names


def _dropped_atoms(molecule: Molecule, assigned: set) -> list[DroppedAtom]:
    dropped = []
    chains = molecule.chains or [""] * len(molecule)
    resnames = molecule.resnames or [""] * len(molecule)
    resids = molecule.resids if len(molecule.resids) else [None] * len(molecule)
    icodes = molecule.icodes or [""] * len(molecule)
    atom_names = molecule.atom_names or [""] * len(molecule)
    for i in range(len(molecule)):
        if i in assigned:
            continue
        dropped.append(
            DroppedAtom(
                name=atom_names[i],
                element=molecule.elements[i] if molecule.elements else "",
                resname=resnames[i],
                resid=None if resids[i] is None else int(resids[i]),
                chain=chains[i],
                insertion_code=icodes[i],
            )
        )
    return dropped


def _reduction_name(weighted: bool) -> str:
    return "centre of mass" if weighted else "centroid"


def _mapping_name(mapping) -> str:
    return mapping if isinstance(mapping, str) else "custom residue"


def _residue_label(
    resid: Optional[int],
    resname: str,
    chain: str,
    insertion_code: str = "",
) -> str:
    parts = ["Residue"]
    if resid is not None:
        parts.append(f"{resid}{insertion_code}")
    if resname:
        parts.append(resname)
    if chain:
        parts.append(chain)
    return " ".join(parts) if len(parts) > 1 else "Molecule"


def bead_label(bead: BeadMapping, index: Optional[int] = None) -> str:
    """A short, unique-ish label for a bead, e.g. ``"BB (12 ALA A)"``.

    Falls back to ``name#index`` when no residue metadata is available, so beads
    from index mappings still get distinct labels.
    """
    residue = " ".join(
        str(p)
        for p in (
            f"{bead.resid}{bead.insertion_code}" if bead.resid is not None else None,
            bead.resname,
            bead.chain,
        )
        if p not in (None, "")
    )
    if residue:
        return f"{bead.name} ({residue})"
    if index is not None:
        return f"{bead.name}#{index}"
    return bead.name


# -- mapping export / round-trip -------------------------------------------


def mapping_to_dict(cg: Molecule) -> dict:
    """Serialise a coarse-grained molecule's mapping to a plain ``dict``.

    The record captures, per bead, the source atom indices and names, the
    reduction used, residue metadata and the bead position, plus the bead-index
    bond network, virtual-site construction rules and any dropped atoms. It is
    JSON-serialisable and can be re-applied with :func:`apply_mapping`.
    """
    report = cg.coarse_grain_report
    beads = []
    for index, bead in enumerate(report.beads):
        beads.append({
            "index": index,
            "name": bead.name,
            "resname": bead.resname,
            "resid": bead.resid,
            "insertion_code": bead.insertion_code,
            "chain": bead.chain,
            "reduction": bead.reduction,
            "atom_indices": list(bead.atom_indices),
            "atom_names": list(bead.atom_names),
            "position": [float(x) for x in cg.coords[index]],
        })
    # Only explicit CG bonds are part of the mapping; never geometry-infer beads.
    bonds = (
        [[int(i), int(j)] for i, j in cg.bond_index]
        if cg.bond_index is not None else []
    )
    dropped = [
        {
            "name": atom.name,
            "element": atom.element,
            "resname": atom.resname,
            "resid": atom.resid,
            "insertion_code": atom.insertion_code,
            "chain": atom.chain,
        }
        for atom in report.dropped_atoms
    ]
    virtual_sites = [
        {
            "index": site.index,
            "name": site.name,
            "resname": site.resname,
            "resid": site.resid,
            "insertion_code": site.insertion_code,
            "chain": site.chain,
            "parents": list(site.parents),
            "parent_names": list(site.parent_names),
            "rule": site.rule,
            "weights": [float(w) for w in site.weights],
            "position": [float(x) for x in cg.coords[site.index]],
        }
        for site in report.virtual_sites
    ]
    return {
        "format": _MAPPING_FORMAT,
        "version": _MAPPING_VERSION,
        "name": cg.name,
        "mapping": report.mapping,
        "n_beads": report.n_beads,
        "n_virtual_sites": report.n_virtual_sites,
        "n_sites": report.n_sites,
        "n_atoms_assigned": report.n_assigned,
        "n_dropped": report.n_dropped,
        "beads": beads,
        "virtual_sites": virtual_sites,
        "bonds": bonds,
        "dropped_atoms": dropped,
    }


def write_mapping(cg: Molecule, path: str) -> str:
    """Write a coarse-grained mapping to a JSON file (see :func:`mapping_to_dict`)."""
    path = os.fspath(path)
    with open(path, "w") as fh:
        json.dump(mapping_to_dict(cg), fh, indent=2)
        fh.write("\n")
    return path


def read_mapping(path: str) -> dict:
    """Read a mapping JSON written by :func:`write_mapping`.

    Returns the record ``dict``; apply it to a structure with
    :func:`apply_mapping`.
    """
    with open(os.fspath(path)) as fh:
        record = json.load(fh)
    fmt = record.get("format")
    if fmt != _MAPPING_FORMAT:
        raise ValueError(
            f"{path}: not a molscope CG mapping (format={fmt!r})"
        )
    return record


def apply_mapping(molecule: Molecule, record: dict) -> Molecule:
    """Re-apply a saved mapping ``record`` to ``molecule``.

    Beads are rebuilt from the stored atom indices (so repeated bead names such
    as ``BB``/``SC`` round-trip cleanly) and reduced exactly as recorded. The
    structure must expose every referenced atom index, so apply a mapping to the
    structure it was built from, or one with the same atom ordering.
    """
    beads = record.get("beads", [])
    if not beads:
        raise ValueError("mapping record has no beads")

    n = len(molecule)
    coords, names, resnames, resids, icodes, chains = [], [], [], [], [], []
    bead_report: list[BeadMapping] = []
    assigned: set[int] = set()
    has_residues = False
    for bead in beads:
        members = [int(i) for i in bead["atom_indices"]]
        out_of_range = [i for i in members if not 0 <= i < n]
        if out_of_range:
            raise ValueError(
                f"bead {bead.get('name')!r} references atom index "
                f"{out_of_range[0]} but the molecule has {n} atoms"
            )
        assigned.update(members)
        use_mass = bead.get("reduction") == "centre of mass"
        coords.append(_reduce(molecule, members, use_mass))
        names.append(bead["name"])
        resname, resid, chain = bead.get("resname", ""), bead.get("resid"), bead.get("chain", "")
        icode = bead.get("insertion_code", bead.get("icode", ""))
        resnames.append(resname)
        resids.append(0 if resid is None else int(resid))
        icodes.append(icode or "")
        chains.append(chain)
        has_residues = has_residues or resid is not None
        bead_report.append(
            BeadMapping(
                name=bead["name"],
                atom_names=list(bead.get("atom_names", _atom_names(molecule, members))),
                reduction=bead.get("reduction", _reduction_name(use_mass)),
                resname=resname,
                resid=resid,
                chain=chain,
                atom_indices=members,
                insertion_code=icode or "",
            )
        )

    virtual_report = _append_virtual_sites(
        coords,
        names,
        record.get("virtual_sites") or None,
        resnames if has_residues else None,
        resids if has_residues else None,
        icodes if has_residues else None,
        chains if has_residues else None,
    )
    bonds = record.get("bonds") or []
    bond_index = np.array(bonds, dtype=int).reshape(-1, 2) if bonds else None
    bond_report = [
        BondMapping(names[i], names[j], "(from mapping file)") for i, j in (bonds or [])
    ]
    dropped = _dropped_atoms(molecule, assigned)
    report = CoarseGrainReport(
        mapping=record.get("mapping", "custom"),
        beads=bead_report,
        dropped_atoms=dropped,
        bonds=bond_report,
        virtual_sites=virtual_report,
    )
    virtual_flags = _virtual_site_flags(len(coords), virtual_report)
    kwargs = dict(
        elements=[""] * len(coords),
        name=record.get("name") or f"{molecule.name} (CG)",
        atom_names=names,
        bond_index=bond_index,
        virtual_sites=virtual_flags,
        _mapping_report=report,
    )
    if has_residues:
        kwargs.update(
            resnames=resnames,
            resids=np.array(resids, dtype=int),
            icodes=icodes,
            chains=chains,
        )
    return Molecule(np.array(coords, dtype=float), **kwargs)


def write_index(cg: Molecule, path: str, per_line: int = 15) -> str:
    """Write a GROMACS-style ``.ndx`` index file, one group per bead.

    Each group lists the 1-based serial numbers of the source atoms folded into
    that bead (serial = atom index + 1). Handy for inspecting a mapping in tools
    that read index files. This is an educational convenience, not a validated
    GROMACS topology.
    """
    path = os.fspath(path)
    report = cg.coarse_grain_report
    used: dict[str, int] = {}
    lines = [
        f"; molscope coarse-grain index ({report.mapping})",
        f"; {report.coverage()}",
    ]
    for index, bead in enumerate(report.beads):
        group = _index_group_name(bead, index, used)
        lines.append(f"[ {group} ]")
        serials = [i + 1 for i in bead.atom_indices]
        for start in range(0, len(serials), per_line):
            lines.append(" ".join(str(s) for s in serials[start:start + per_line]))
    if report.virtual_sites:
        lines.append("; virtual sites are coordinate constructions, not source-atom groups")
        for site in report.virtual_sites:
            parents = ", ".join(
                f"{name}#{idx}" for idx, name in zip(site.parents, site.parent_names)
            )
            weights = ", ".join(f"{w:g}" for w in site.weights)
            lines.append(f"; {site.name} = {site.rule}({parents}; weights={weights})")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _index_group_name(bead: BeadMapping, index: int, used: dict[str, int]) -> str:
    """Build a unique, whitespace-free group name for one bead."""
    parts = [bead.name]
    if bead.resid is not None:
        parts.append(f"{bead.resid}{bead.insertion_code}")
    if bead.resname:
        parts.append(bead.resname)
    if bead.chain:
        parts.append(bead.chain)
    base = re.sub(r"\s+", "_", "_".join(parts)) or f"bead{index}"
    count = used.get(base, 0)
    used[base] = count + 1
    return base if count == 0 else f"{base}_{count + 1}"


def write_openmm_xml(cg: Molecule, path: str) -> str:
    """Write the CG molecule's residue mapping as an OpenMM residue-template XML.

    The output is a self-contained ``ForceField`` that OpenMM's
    :class:`openmm.app.ForceField` can load directly: an ``<AtomTypes>`` block
    defines one ``CG_<resname>_<bead>`` type per distinct bead (carrying the
    bead mass; coarse-grained beads have no element), and a ``<Residues>`` block
    holds one template per residue name. Internal bonds are written as
    ``<Bond atomName1="..." atomName2="..."/>`` and bonds crossing residues as
    ``<ExternalBond atomName="..."/>``.

    The file describes topology only (types, masses, connectivity); it carries
    no force definitions, so it is meant to be combined with bonded/nonbonded
    parameters before running dynamics. Returns the path to the written file.
    """
    path = os.fspath(path)

    # Fall back to synthetic bead names if the molecule carries none, so a
    # nameless CG molecule degrades gracefully instead of raising IndexError.
    bead_names = cg.atom_names if cg.atom_names else [f"B{i + 1}" for i in range(len(cg))]
    masses = cg.masses

    # 1. Map each bead to its residue key (resname, resid, icode, chain)
    bead_res_keys = []
    for i in range(len(cg)):
        resname = cg.resnames[i] if cg.resnames else "MOL"
        resid = int(cg.resids[i]) if len(cg.resids) else 1
        icode = cg.icodes[i] if cg.icodes else ""
        chain = cg.chains[i] if cg.chains else ""
        bead_res_keys.append((resname, resid, icode, chain))

    # 2. Group bead indices by residue instance
    res_instances = {}
    for i, rkey in enumerate(bead_res_keys):
        res_instances.setdefault(rkey, []).append(i)

    # 3. Identify beads with external bonds
    external_beads = [set() for _ in range(len(cg))]
    if cg.bond_index is not None:
        for u, v in cg.bond_index:
            if bead_res_keys[u] != bead_res_keys[v]:
                external_beads[u].add(bead_names[u])
                external_beads[v].add(bead_names[v])

    # 4. Consolidate templates by unique residue name
    templates = {}
    for rkey, bead_indices in res_instances.items():
        resname, resid, icode, chain = rkey
        inst_beads = [bead_names[idx] for idx in bead_indices]
        inst_masses = {bead_names[idx]: float(masses[idx]) for idx in bead_indices}

        inst_bonds = set()
        if cg.bond_index is not None:
            idx_set = set(bead_indices)
            for u, v in cg.bond_index:
                if u in idx_set and v in idx_set:
                    b_pair = tuple(sorted((bead_names[u], bead_names[v])))
                    inst_bonds.add(b_pair)

        inst_external = set()
        for idx in bead_indices:
            inst_external.update(external_beads[idx])

        if resname not in templates:
            templates[resname] = {
                "beads": inst_beads,
                "bonds": inst_bonds,
                "external": inst_external,
                "masses": inst_masses,
            }
        else:
            templates[resname]["bonds"].update(inst_bonds)
            templates[resname]["external"].update(inst_external)
            templates[resname]["masses"].update(inst_masses)
            for name in inst_beads:
                if name not in templates[resname]["beads"]:
                    templates[resname]["beads"].append(name)

    # 5. Build XML content. The <AtomTypes> block defines every type the
    # residue templates reference, so OpenMM can load the file standalone.
    lines = ["<ForceField>", "  <AtomTypes>"]
    for resname in sorted(templates.keys()):
        t = templates[resname]
        for bead_name in t["beads"]:
            type_name = f"CG_{resname}_{bead_name}"
            lines.append(
                f'    <Type name="{type_name}" class="{type_name}" '
                f'mass="{t["masses"][bead_name]:g}"/>'
            )
    lines.append("  </AtomTypes>")
    lines.append("  <Residues>")

    for resname in sorted(templates.keys()):
        t = templates[resname]
        lines.append(f'    <Residue name="{resname}">')
        for bead_name in t["beads"]:
            lines.append(
                f'      <Atom name="{bead_name}" '
                f'type="CG_{resname}_{bead_name}"/>'
            )
        for atom1, atom2 in sorted(t["bonds"]):
            lines.append(f'      <Bond atomName1="{atom1}" atomName2="{atom2}"/>')
        for atom in sorted(t["external"]):
            lines.append(f'      <ExternalBond atomName="{atom}"/>')
        lines.append("    </Residue>")

    lines.extend([
        "  </Residues>",
        "</ForceField>"
    ])

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_itp(cg: Molecule, path: str) -> str:
    """Write the CG molecule as a rough GROMACS ``.itp`` topology skeleton.

    Emits a ``[moleculetype]`` plus ``[atoms]`` (one row per bead: a
    ``CG_<resname>_<bead>`` type, residue assignment, zero charge and the bead
    mass), ``[bonds]`` from the bead bond network, and ``[angles]`` enumerated
    from that network (every ``i-j-k`` where ``i-j`` and ``j-k`` are bonds).

    This is a topology *skeleton* for prototyping, **not** a validated force
    field: bonds and angles are written with function type 1 but no force
    constants or reference values, and there are no non-bonded parameters. Fill
    those in (or map onto a Martini / elastic-network model) before running any
    dynamics. Returns the path to the written file.
    """
    from collections import defaultdict

    path = os.fspath(path)
    n = len(cg)
    bead_names = cg.atom_names if cg.atom_names else [f"B{i + 1}" for i in range(n)]
    masses = cg.masses
    resnames = cg.resnames if cg.resnames else ["MOL"] * n
    resids = cg.resids if len(cg.resids) else np.ones(n, dtype=int)
    mol_name = re.sub(r"\s+", "_", cg.name) or "MOL"

    bonds = (
        [(int(i), int(j)) for i, j in cg.bond_index]
        if cg.bond_index is not None else []
    )

    lines = [
        "; Coarse-grained topology skeleton generated by MolScope.",
        "; NOT a validated force field: bonds/angles carry no force constants and",
        "; there are no non-bonded parameters. For prototyping only.",
        "",
        "[ moleculetype ]",
        "; name   nrexcl",
        f"{mol_name}   1",
        "",
        "[ atoms ]",
        ";   nr           type  resnr residue   atom   cgnr     charge       mass",
    ]
    for i in range(n):
        atype = f"CG_{resnames[i]}_{bead_names[i]}"
        lines.append(
            f"{i + 1:>6} {atype:>14} {int(resids[i]):>6} {resnames[i]:>7} "
            f"{bead_names[i]:>6} {i + 1:>6} {0.0:>10.4f} {float(masses[i]):>10.4f}"
        )

    lines += ["", "[ bonds ]", ";  ai   aj  funct"]
    for i, j in bonds:
        lines.append(f"{i + 1:>5} {j + 1:>4}  1")

    # Angles: every i-j-k with i-j and j-k bonded (j the central bead).
    neighbors = defaultdict(set)
    for i, j in bonds:
        neighbors[i].add(j)
        neighbors[j].add(i)
    angles = []
    for j in range(n):
        nb = sorted(neighbors[j])
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                angles.append((nb[a], j, nb[b]))

    lines += ["", "[ angles ]", ";  ai   aj   ak  funct"]
    for i, j, k in angles:
        lines.append(f"{i + 1:>5} {j + 1:>4} {k + 1:>4}  1")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
