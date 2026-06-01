"""Optional RDKit-backed chemical perception.

MolScope keeps cheminformatics out of the core dependency set. When RDKit is
installed, this module can annotate a :class:`~molscope.molecule.Molecule` with
formal charges, valence, aromaticity and sanitized bond-order information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ChemicalFeatures:
    """Per-atom and per-bond features from RDKit sanitization.

    Bond arrays are aligned with the bond list used to build the RDKit molecule:
    explicit ``molecule.bond_index`` when present, otherwise geometrically
    inferred bonds when ``infer_bonds=True``.
    """

    formal_charges: np.ndarray
    total_valences: np.ndarray
    aromatic_atoms: np.ndarray
    bond_index: np.ndarray
    bond_orders: np.ndarray
    aromatic_bonds: np.ndarray


def chemical_features(molecule, *, sanitize: bool = True, infer_bonds: bool = True):
    """Return RDKit-backed chemical features for ``molecule``.

    Explicit SDF/MOL bond orders and formal charges are used when available. If
    the molecule has only coordinates, geometrically inferred bonds are passed
    to RDKit as single bonds. MolScope does not attempt general bond-order
    inference from raw coordinates.
    """
    rdmol, bond_index = to_rdkit(molecule, sanitize=sanitize, infer_bonds=infer_bonds)
    atoms = list(rdmol.GetAtoms())
    bond_orders, aromatic_bonds = [], []
    for i, j in bond_index:
        bond = rdmol.GetBondBetweenAtoms(int(i), int(j))
        if bond is None:
            bond_orders.append(0.0)
            aromatic_bonds.append(False)
        else:
            bond_orders.append(float(bond.GetBondTypeAsDouble()))
            aromatic_bonds.append(bool(bond.GetIsAromatic()))

    return ChemicalFeatures(
        formal_charges=np.array([a.GetFormalCharge() for a in atoms], dtype=int),
        total_valences=np.array([a.GetTotalValence() for a in atoms], dtype=int),
        aromatic_atoms=np.array([a.GetIsAromatic() for a in atoms], dtype=bool),
        bond_index=bond_index,
        bond_orders=np.array(bond_orders, dtype=float),
        aromatic_bonds=np.array(aromatic_bonds, dtype=bool),
    )


def rdkit_descriptors(
    molecule,
    *,
    names: Optional[list[str]] = None,
    prefix: str = "rdkit_",
    sanitize: bool = True,
    infer_bonds: bool = True,
    errors: str = "nan",
) -> dict[str, float]:
    """Return scalar RDKit molecular descriptors as a flat dictionary.

    ``names`` can restrict the descriptor set to RDKit descriptor names such as
    ``"MolWt"`` or ``"TPSA"``. Keys are prefixed by default to avoid collisions
    with MolScope-native descriptor names. ``errors="nan"`` records failing
    descriptor calculations as NaN; pass ``errors="raise"`` to surface them.
    """
    if errors not in ("nan", "raise"):
        raise ValueError("errors must be 'nan' or 'raise'")

    Chem, _ = _require_rdkit()
    from rdkit.Chem import Descriptors

    rdmol, _ = to_rdkit(molecule, sanitize=sanitize, infer_bonds=infer_bonds)
    if sanitize:
        # Keep descriptor preconditions explicit when to_rdkit was called with
        # an already-sanitized molecule; this is cheap and avoids stale caches.
        Chem.SanitizeMol(rdmol)
    else:
        rdmol.UpdatePropertyCache(strict=False)

    desc_map = dict(Descriptors._descList)
    selected = list(desc_map) if names is None else list(names)
    unknown = [name for name in selected if name not in desc_map]
    if unknown:
        raise ValueError(f"unknown RDKit descriptor(s): {', '.join(unknown)}")

    out: dict[str, float] = {}
    for name in selected:
        try:
            value = desc_map[name](rdmol)
            out[f"{prefix}{name}"] = _numeric_descriptor(value)
        except Exception:
            if errors == "raise":
                raise
            out[f"{prefix}{name}"] = float("nan")
    return out


def to_rdkit(molecule, *, sanitize: bool = True, infer_bonds: bool = True):
    """Build an RDKit molecule from a MolScope molecule.

    Returns ``(rdmol, bond_index)`` where ``bond_index`` is the edge list used to
    populate the RDKit molecule.
    """
    Chem, Point3D = _require_rdkit()
    rw = Chem.RWMol()
    charges = molecule.formal_charges if len(molecule.formal_charges) else None
    for i, symbol in enumerate(molecule.elements):
        atom = _rdkit_atom(Chem, symbol)
        if charges is not None:
            atom.SetFormalCharge(int(charges[i]))
        rw.AddAtom(atom)

    if molecule.bond_index is not None:
        bond_index = molecule.bond_index
        orders = molecule.bond_order_array()
    elif infer_bonds:
        bond_index = molecule.bonds()
        orders = np.ones(len(bond_index), dtype=float)
    else:
        bond_index = np.empty((0, 2), dtype=int)
        orders = np.empty(0, dtype=float)

    for (i, j), order in zip(bond_index, orders):
        bond_type = _rdkit_bond_type(Chem, order)
        rw.AddBond(int(i), int(j), bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            rw.GetAtomWithIdx(int(i)).SetIsAromatic(True)
            rw.GetAtomWithIdx(int(j)).SetIsAromatic(True)
            rw.GetBondBetweenAtoms(int(i), int(j)).SetIsAromatic(True)

    rdmol = rw.GetMol()
    conf = Chem.Conformer(len(molecule))
    for i, (x, y, z) in enumerate(molecule.coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    rdmol.AddConformer(conf)

    try:
        if sanitize:
            Chem.SanitizeMol(rdmol)
        else:
            rdmol.UpdatePropertyCache(strict=False)
    except Exception as exc:
        raise ValueError(
            "RDKit could not sanitize this molecule; check bond orders, valence "
            "and formal charges, or call chemical_features(sanitize=False)"
        ) from exc
    return rdmol, np.asarray(bond_index, dtype=int).reshape(-1, 2)


#: Idealised protonation of standard ionisable side chains near pH 7, as a
#: ``(resname, atom name) -> formal charge`` table. This is a fixed textbook
#: assignment, NOT a pKa- or environment-aware prediction: aspartate/glutamate
#: carboxylates are -1, lysine/arginine side chains +1, histidine is left
#: neutral, and chain termini are not charged. For an environment-aware
#: prediction that accounts for the local structure use ``protonation="pka"``
#: (PROPKA backend); see :func:`pdb_template_bonds`.
STANDARD_PROTONATION = {
    ("ASP", "OD2"): -1,
    ("GLU", "OE2"): -1,
    ("LYS", "NZ"): +1,
    ("ARG", "NH2"): +1,
}


#: Maps a PROPKA ionisable-group label to the MolScope atom that carries the
#: residue's formal charge and whether the group titrates as an ``"acid"``
#: (negative when deprotonated) or a ``"base"`` (positive when protonated). The
#: charge is placed on a single representative atom so the residue's *net* charge
#: matches the pKa prediction; PROPKA's own group labels (``N+``/``C-`` for the
#: termini) are translated to the corresponding backbone atom names.
RESIDUE_PKA_GROUPS = {
    "ASP": ("OD2", "acid"),
    "GLU": ("OE2", "acid"),
    "C-": ("OXT", "acid"),
    "CYS": ("SG", "acid"),
    "TYR": ("OH", "acid"),
    "LYS": ("NZ", "base"),
    "ARG": ("NH2", "base"),
    "HIS": ("NE2", "base"),
    "N+": ("N", "base"),
}


def _pka_formal_charge(kind: str, pka: float, ph: float) -> int:
    """Formal charge of an ionisable group from its pKa relative to ``ph``.

    An acid is deprotonated (charge ``-1``) when the solution is more basic than
    its pKa (``ph > pka``) and neutral otherwise; a base is protonated (charge
    ``+1``) when the solution is more acidic than its pKa (``ph < pka``) and
    neutral otherwise. At ``ph == pka`` the group is taken to be neutral (the
    half-protonated point is reported as the uncharged species).
    """
    if kind == "acid":
        return -1 if ph > pka else 0
    if kind == "base":
        return 1 if ph < pka else 0
    raise ValueError(f"unknown titration kind {kind!r}; expected 'acid' or 'base'")


def _require_propka():
    """Import PROPKA or raise a clear install hint (mirrors :func:`_require_rdkit`)."""
    try:
        import propka.run as propka_run
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "PROPKA is required for pKa-aware protonation; install it with "
            'pip install "molscope[propka]"'
        ) from exc
    return propka_run


def _pka_residue_charges(path: str, ph: float) -> list:
    """Predict per-residue formal charges for a PDB with PROPKA at ``ph``.

    Returns a list of ``(chain_id, res_num, icode, atom_name, charge)`` tuples,
    one per ionisable group PROPKA resolves to a known residue type. The caller
    maps each tuple onto a MolScope atom by ``(chain, resid, icode, atom name)``;
    groups whose target atom is absent are silently skipped. PROPKA's verbose
    logging is suppressed for the duration of the run.
    """
    propka_run = _require_propka()
    logger = logging.getLogger("propka")
    prev_level, prev_propagate = logger.level, logger.propagate
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    try:
        molecule = propka_run.single(path, optargs=["--pH", str(ph)], write_pka=False)
    finally:
        logger.setLevel(prev_level)
        logger.propagate = prev_propagate

    results = []
    conformation = molecule.conformations[molecule.conformation_names[0]]
    for group in conformation.groups:
        spec = RESIDUE_PKA_GROUPS.get(str(group.residue_type).strip())
        if spec is None or group.pka_value is None:
            continue
        atom_name, kind = spec
        charge = _pka_formal_charge(kind, float(group.pka_value), float(ph))
        if charge == 0:
            continue
        atom = group.atom
        icode = (getattr(atom, "icode", "") or "").strip()
        results.append(
            (str(atom.chain_id).strip(), int(atom.res_num), icode, atom_name, charge)
        )
    return results


def pdb_template_bonds(path: str, molecule, protonation: str = "none", ph: float = 7.0):
    """Perceive bonds for standard residues via RDKit's residue-aware PDB reader.

    RDKit's PDB parser assigns bonds *and bond orders* for standard amino acids
    and nucleotides from built-in residue templates (plus peptide bonds and
    disulfides), recovering aromatic rings and double bonds that geometric
    distance inference cannot. This returns ``(bond_index, bond_orders,
    formal_charges)`` in ``molecule`` atom order so they can be attached as
    explicit bonds. Aromatic rings are returned in Kekule form (alternating
    single/double bonds) so they round-trip cleanly and re-aromatise on
    sanitisation; per-atom formal charges (carboxylate, ammonium, ...) carry over.

    Atoms are matched to ``molecule`` by ``(chain, resid, insertion code, atom
    name)``; any RDKit bond whose endpoints are not both matched (e.g. an
    alternate location RDKit dropped, or a non-standard atom) is skipped. Needs
    RDKit (``pip install "molscope[chem]"``) and only helps for standard
    residues; modified residues and exotic ligands stay best-effort.

    ``protonation`` controls side-chain charges: ``"none"`` (default) keeps the
    as-modelled neutral state RDKit reads from the coordinates; ``"standard"``
    applies the idealised pH-7 assignment in :data:`STANDARD_PROTONATION`
    (aspartate/glutamate -1, lysine/arginine +1, histidine neutral, termini
    uncharged), a fixed textbook model; and ``"pka"`` runs PROPKA to predict
    per-residue pKa from the structure and assigns charges for the dominant state
    at ``ph`` (default 7.0). The ``"pka"`` mode needs PROPKA
    (``pip install "molscope[propka]"``) and is environment-aware: it accounts
    for the local electrostatics and burial of each ionisable group, unlike the
    fixed ``"standard"`` table.
    """
    if protonation not in ("none", "standard", "pka"):
        raise ValueError("protonation must be 'none', 'standard', or 'pka'")
    Chem, _ = _require_rdkit()
    rdmol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=True)
    if rdmol is None:
        raise ValueError(f"RDKit could not parse {path!r} for template bond perception")
    # Kekulise so aromatic rings become explicit single/double bonds: rebuilding a
    # molecule from aromatic-flagged bonds alone fails to re-kekulise, whereas an
    # explicit Kekule structure sanitises cleanly and re-aromatises downstream.
    Chem.Kekulize(rdmol, clearAromaticFlags=True)

    n = len(molecule)
    icodes = molecule.icodes if molecule.icodes is not None else [""] * n
    by_key: dict = {}
    for i in range(n):
        key = (str(molecule.chains[i]).strip(), int(molecule.resids[i]),
               (icodes[i] or "").strip(), str(molecule.atom_names[i]).strip())
        by_key.setdefault(key, i)  # first occurrence wins for duplicate keys (altlocs)

    rd_to_ms: dict = {}
    charges = np.zeros(n, dtype=int)
    for atom in rdmol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        key = (info.GetChainId().strip(), int(info.GetResidueNumber()),
               info.GetInsertionCode().strip(), info.GetName().strip())
        ms_index = by_key.get(key)
        if ms_index is not None:
            rd_to_ms[atom.GetIdx()] = ms_index
            charges[ms_index] = atom.GetFormalCharge()

    if protonation == "standard":
        for i in range(n):
            charge = STANDARD_PROTONATION.get(
                (str(molecule.resnames[i]).strip(), str(molecule.atom_names[i]).strip())
            )
            if charge is not None:
                charges[i] = charge
    elif protonation == "pka":
        for chain, resid, icode, atom_name, charge in _pka_residue_charges(path, ph):
            ms_index = by_key.get((chain, resid, icode, atom_name))
            if ms_index is not None:
                charges[ms_index] = charge

    pairs, orders = [], []
    for bond in rdmol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in rd_to_ms and j in rd_to_ms:
            pairs.append((rd_to_ms[i], rd_to_ms[j]))
            orders.append(float(bond.GetBondTypeAsDouble()))
    if not pairs:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=float), charges
    return np.array(pairs, dtype=int), np.array(orders, dtype=float), charges


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Geometry import Point3D
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "RDKit is required for chemical perception; install it with "
            'pip install "molscope[chem]"'
        ) from exc
    return Chem, Point3D


def _rdkit_atom(Chem, symbol: str):
    raw = (symbol or "").strip()
    if not raw:
        return Chem.Atom(0)
    normalized = raw[0].upper() + raw[1:].lower()
    try:
        return Chem.Atom(normalized)
    except RuntimeError:
        return Chem.Atom(0)


def _rdkit_bond_type(Chem, order: float):
    if np.isclose(order, 1.5) or np.isclose(order, 4.0):
        return Chem.BondType.AROMATIC
    rounded = int(round(float(order)))
    if rounded == 1:
        return Chem.BondType.SINGLE
    if rounded == 2:
        return Chem.BondType.DOUBLE
    if rounded == 3:
        return Chem.BondType.TRIPLE
    return Chem.BondType.UNSPECIFIED


def _numeric_descriptor(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise TypeError(f"RDKit descriptor returned non-scalar value {value!r}")
