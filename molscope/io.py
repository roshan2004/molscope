"""Readers and writers for molecular coordinate files.

``read`` dispatches on file extension; the individual readers/writers can also
be called directly. PDB parsing uses fixed columns (not whitespace splitting),
which is the only correct way to read the format. Per-atom metadata (atom name,
residue, chain) and explicit connectivity are captured where the format
provides them.
"""

from __future__ import annotations

import gzip
import os
import tempfile
from collections.abc import Generator
from dataclasses import replace
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np

from .molecule import Molecule

# altLoc codes we accept: blank (no alternates) or the primary "A" conformation.
_PRIMARY_ALTLOCS = (" ", "A", "")
_ALTLOC_POLICIES = ("primary", "first", "highest_occupancy", "all")


def _parse_error(path, fmt: str, detail: str, lineno: Optional[int] = None) -> ValueError:
    """Build a ValueError that names the file, format, line, and what went wrong."""
    where = f" (line {lineno})" if lineno else ""
    return ValueError(f"{path}: invalid {fmt} file{where}: {detail}")


def read(path: str, bond_perception: str = "geometric", protonation: str = "none") -> Molecule:
    """Read a molecule, picking the parser from the file extension.

    Transparently handles gzip-compressed files (``.pdb.gz``, ``.xyz.gz``).
    ``bond_perception="template"`` (with optional ``protonation="standard"``)
    requests RDKit residue-template bonds and is only supported for ``.pdb``
    files (see :func:`read_pdb`).
    """
    path = os.fspath(path)
    ext = _data_extension(path)
    if ext == ".pdb":
        return read_pdb(path, bond_perception=bond_perception, protonation=protonation)
    if bond_perception != "geometric" or protonation != "none":
        raise ValueError(
            "bond_perception='template'/protonation are only supported for .pdb files"
        )
    if ext == ".xyz":
        return read_xyz(path)
    if ext in (".cif", ".mmcif"):
        return read_cif(path)
    if ext in (".sdf", ".mol"):
        return read_sdf(path)
    raise ValueError(f"Unsupported file type {ext!r}; expected .pdb/.xyz/.cif/.sdf")


def fetch(
    pdb_id: str, fmt: str = "pdb", cache_dir: Optional[str] = None,
    bond_perception: str = "geometric", protonation: str = "none",
) -> Molecule:
    """Download a structure from RCSB by its PDB id and read it.

    ``fmt`` is ``"pdb"`` or ``"cif"``. Files are cached (default: the system temp
    directory) so repeat calls don't re-download. Example: ``ms.fetch("1fqy")``.
    ``bond_perception="template"`` attaches RDKit residue-template bonds and
    needs ``fmt="pdb"`` (see :func:`read_pdb`).
    """
    fmt = fmt.lower()
    if fmt not in ("pdb", "cif"):
        raise ValueError("fmt must be 'pdb' or 'cif'")
    pdb_id = pdb_id.lower()
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "molscope_cache")
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, f"{pdb_id}.{fmt}")
    if not os.path.exists(dest):
        url = f"https://files.rcsb.org/download/{pdb_id}.{fmt}"
        try:
            with urlopen(url) as resp:
                data = resp.read()
        except HTTPError as exc:
            if exc.code == 404:
                raise ValueError(
                    f"PDB id {pdb_id!r} not found at RCSB as {fmt} (HTTP 404)"
                ) from exc
            raise ValueError(f"failed to download {url} (HTTP {exc.code})") from exc
        except URLError as exc:
            raise ValueError(f"could not reach RCSB to download {url}: {exc.reason}") from exc
        with open(dest, "wb") as fh:
            fh.write(data)
    return read(dest, bond_perception=bond_perception, protonation=protonation)


def read_smiles(smiles: str, *, name: Optional[str] = None, add_hs: bool = True,
                seed: int = 0) -> Molecule:
    """Build a :class:`Molecule` from a SMILES string by generating one conformer.

    The bonds, Kekule bond orders, and formal charges come from RDKit; the
    coordinates are a single **generated** 3D conformer (RDKit ETKDG), *not* an
    experimental or energy-minimised structure. This makes it a good input for
    descriptors and graph-ML (where topology is what matters), but treat
    geometry-dependent results (contact maps, RMSD against experiment, precise
    distances) with care.

    Needs RDKit (``pip install "molscope[chem]"``). ``add_hs`` keeps explicit
    hydrogens (recommended for embedding quality); ``seed`` makes the conformer
    reproducible. Raises :class:`ValueError` for an invalid SMILES or if a 3D
    conformer cannot be embedded.
    """
    from .chem import _require_rdkit

    Chem, _ = _require_rdkit()
    from rdkit import RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")  # silence per-parse errors; we raise our own
    try:
        rdmol = Chem.MolFromSmiles(str(smiles))
    finally:
        RDLogger.EnableLog("rdApp.*")
    if rdmol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")

    rdmol = Chem.AddHs(rdmol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(rdmol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(rdmol, params) != 0:
            raise ValueError(f"RDKit could not embed a 3D conformer for {smiles!r}")
    if not add_hs:
        rdmol = Chem.RemoveHs(rdmol)
    # Kekulise so aromatic rings round-trip as explicit single/double bonds.
    Chem.Kekulize(rdmol, clearAromaticFlags=True)

    conf = rdmol.GetConformer()
    coords = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(rdmol.GetNumAtoms())],
        dtype=float,
    ).reshape(-1, 3)
    elements = [atom.GetSymbol() for atom in rdmol.GetAtoms()]
    charges = np.array([atom.GetFormalCharge() for atom in rdmol.GetAtoms()], dtype=int)
    bond_index = np.array(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in rdmol.GetBonds()], dtype=int
    ).reshape(-1, 2)
    bond_orders = np.array(
        [float(b.GetBondTypeAsDouble()) for b in rdmol.GetBonds()], dtype=float
    )
    return Molecule(
        coords, elements, name=name or f"smiles:{smiles}",
        bond_index=bond_index, bond_orders=bond_orders, formal_charges=charges,
    )


def read_xyz(path: str) -> Molecule:
    """Read a single-frame ``.xyz`` file.

    Handles both the standard ``element x y z`` layout and the bare
    ``x y z`` coordinate dumps (with ``#`` comment lines) used by some tools.
    """
    frames = read_xyz_frames(path)
    if not frames:
        return Molecule(np.empty((0, 3)), [], name=_stem(path))
    return frames[0]


def read_xyz_frames(path: str) -> list[Molecule]:
    """Read every frame of a (possibly multi-frame) ``.xyz`` trajectory.

    Standard xyz frames begin with an atom-count line; bare coordinate dumps
    with ``#`` comments are returned as a single frame.
    """
    with _open(path) as f:
        lines = f.readlines()

    stem = _stem(path)
    frames: list[Molecule] = []
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        tokens = lines[i].split()
        if tokens and tokens[0].isdigit():
            count = int(tokens[0])
            block = lines[i + 2:i + 2 + count]
            frames.append(_xyz_block(
                block, name=f"{stem}#{len(frames) + 1}", path=path,
                expected=count, first_line=i + 3,
            ))
            i += 2 + count
        else:
            # Bare coordinate dump (no header): consume the rest as one frame.
            frames.append(_xyz_block(lines[i:], name=stem, path=path, first_line=i + 1))
            break
    return frames


def read_pdb(
    path: str, model: int = 1, altloc: str = "primary", bond_perception: str = "geometric",
    protonation: str = "none",
) -> Molecule:
    """Read ``ATOM``/``HETATM`` records from a ``.pdb`` file.

    Coordinates, element, atom name, residue name/id and chain are sliced from
    their fixed columns per the PDB spec. ``altloc`` controls alternate
    conformations: ``"primary"`` keeps blank/``A`` locations, ``"first"`` keeps
    the first location per atom, ``"highest_occupancy"`` keeps the location with
    the largest occupancy, and ``"all"`` keeps every alternate. For multi-model
    (NMR) files the 1-based ``model`` is returned; files without ``MODEL``
    records are read in full.

    ``bond_perception`` chooses how bonds are assigned: ``"geometric"`` (the
    default) leaves connectivity to later distance-based inference, while
    ``"template"`` uses RDKit's residue-aware perception to attach explicit bonds
    and bond orders for standard residues (needs the ``chem`` extra). Template
    bonds make aromaticity, double bonds, and RDKit-backed features correct for
    proteins, where geometric inference yields single bonds only.

    ``protonation`` (only with ``bond_perception="template"``) assigns idealised
    pH-7 side-chain charges for standard residues when set to ``"standard"``; the
    default ``"none"`` keeps the as-modelled neutral state. See
    :func:`molscope.chem.pdb_template_bonds`.
    """
    models, unit_cell = _parse_pdb_models(path, altloc=altloc)
    if not models:
        return Molecule(np.empty((0, 3)), [], name=_stem(path))
    if not 1 <= model <= len(models):
        raise ValueError(f"model {model} out of range (1..{len(models)})")
    mol = _molecule_from_record(models[model - 1], _stem(path))
    mol = replace(mol, unit_cell=unit_cell)
    if bond_perception == "template":
        from .chem import pdb_template_bonds

        bond_index, bond_orders, formal_charges = pdb_template_bonds(
            path, mol, protonation=protonation
        )
        mol = replace(
            mol, bond_index=bond_index, bond_orders=bond_orders,
            formal_charges=formal_charges,
        )
    elif bond_perception != "geometric":
        raise ValueError("bond_perception must be 'geometric' or 'template'")
    elif protonation != "none":
        raise ValueError("protonation requires bond_perception='template'")
    return mol


def read_pdb_models(path: str, altloc: str = "primary") -> list[Molecule]:
    """Read every model from a ``.pdb`` file as a list of molecules."""
    stem = _stem(path)
    models, unit_cell = _parse_pdb_models(path, altloc=altloc)
    return [
        replace(_molecule_from_record(rec, f"{stem}#{i + 1}"), unit_cell=unit_cell)
        for i, rec in enumerate(models)
    ]



def read_cif(path: str, parser: str = "builtin") -> Molecule:
    """Basic mmCIF reader for standard ``_atom_site`` coordinate loops.

    ``parser="builtin"`` uses MolScope's lightweight atom-site reader.
    ``parser="gemmi"`` uses the optional Gemmi backend
    (``pip install "molscope[cif]"``). Neither parser performs full dictionary
    validation; use :func:`molscope.validate_cif` for validation checks.
    """
    path = os.fspath(path)
    if parser == "gemmi":
        return _read_cif_gemmi(path)
    if parser != "builtin":
        raise ValueError("parser must be 'builtin' or 'gemmi'")
    return _read_cif_builtin(path)


def _require_cif_coord_columns(idx: dict, path) -> None:
    """Ensure the atom-site loop declares the Cartesian coordinate columns."""
    missing = [c for c in ("Cartn_x", "Cartn_y", "Cartn_z") if c not in idx]
    if missing:
        raise _parse_error(
            path, "mmCIF",
            f"_atom_site loop is missing coordinate column(s) {missing}; "
            f"found columns {sorted(idx)}",
        )


def _cif_coords(row, idx: dict, path, rownum: int) -> tuple:
    """Read one atom's x/y/z from a CIF row, with a clear error on failure."""
    try:
        return (
            float(row[idx["Cartn_x"]]),
            float(row[idx["Cartn_y"]]),
            float(row[idx["Cartn_z"]]),
        )
    except (ValueError, IndexError):
        raise _parse_error(
            path, "mmCIF", f"could not read coordinates from _atom_site row {rownum}",
        ) from None


def _read_cif_builtin(path: str) -> Molecule:
    """Read a CIF/mmCIF atom-site loop with MolScope's lightweight tokenizer."""
    from .molecule import UnitCell

    with _open(path) as f:
        tokens = _cif_tokens(f.read())

    columns: list[str] = []
    rows: list[list[str]] = []
    unit_cell = None
    i = 0
    while i < len(tokens):
        token_lower = tokens[i].lower()
        if token_lower == "loop_":
            i += 1
            loop_columns = []
            while i < len(tokens) and tokens[i].startswith("_"):
                loop_columns.append(tokens[i])
                i += 1

            if any(col.lower().startswith("_atom_site.") for col in loop_columns):
                width = len(loop_columns)
                columns = [col.split(".", 1)[1] if "." in col else col for col in loop_columns]
                while i + width <= len(tokens) and not _cif_control_token(tokens[i]):
                    row = tokens[i:i + width]
                    if any(_cif_control_token(tok) for tok in row):
                        break
                    rows.append(row)
                    i += width
                continue

            # Skip other loops
            while i < len(tokens) and not _cif_control_token(tokens[i]):
                i += 1
            continue

        if tokens[i].startswith("_cell."):
            cell_data = {}
            while i < len(tokens) and tokens[i].startswith("_cell."):
                key = tokens[i].lower()
                i += 1
                if i < len(tokens) and not _cif_control_token(tokens[i]):
                    cell_data[key] = tokens[i]
                    i += 1

            try:
                unit_cell = UnitCell(
                    a=float(cell_data["_cell.length_a"]),
                    b=float(cell_data["_cell.length_b"]),
                    c=float(cell_data["_cell.length_c"]),
                    alpha=float(cell_data.get("_cell.angle_alpha", 90.0)),
                    beta=float(cell_data.get("_cell.angle_beta", 90.0)),
                    gamma=float(cell_data.get("_cell.angle_gamma", 90.0)),
                )
            except (KeyError, ValueError):
                pass
            continue

        i += 1

    if not columns or not rows:
        raise _parse_error(path, "mmCIF", "no _atom_site coordinate loop found")
    idx = {name: i for i, name in enumerate(columns)}

    def col(row, *names, default=""):
        for nm in names:
            if nm in idx and idx[nm] < len(row):
                value = row[idx[nm]]
                if value not in {".", "?"}:
                    return value
        return default

    _require_cif_coord_columns(idx, path)
    coords, els, anames, rnames, rids, icodes, chains, heteros = [], [], [], [], [], [], [], []
    for r, row in enumerate(rows, 1):
        coords.append(_cif_coords(row, idx, path, r))
        els.append(_clean_cif_value(col(row, "type_symbol")))
        anames.append(_clean_cif_value(col(row, "label_atom_id", "auth_atom_id")))
        rnames.append(_clean_cif_value(col(row, "label_comp_id", "auth_comp_id")))
        chains.append(_clean_cif_value(col(row, "auth_asym_id", "label_asym_id")))
        rid = _clean_cif_value(col(row, "auth_seq_id", "label_seq_id", default="0"))
        rids.append(int(rid) if rid.lstrip("-").isdigit() else 0)
        icodes.append(_clean_cif_value(col(row, "pdbx_PDB_ins_code")))
        heteros.append(_clean_cif_value(col(row, "group_PDB", default="ATOM")).upper() == "HETATM")

    return Molecule(
        np.array(coords, dtype=float), els, name=_stem(path),
        atom_names=anames, resnames=rnames, resids=np.array(rids, dtype=int),
        icodes=icodes, chains=chains, hetero=heteros, unit_cell=unit_cell,
    )



def _gemmi_unit_cell(block):
    """Return a UnitCell from a gemmi CIF block's _cell.* items, or None.

    gemmi's ``cif.Block`` exposes no ``.cell`` attribute, so the parameters are
    read tag by tag. Missing length items mean no cell was recorded.
    """
    import gemmi

    from .molecule import UnitCell

    def value(tag: str):
        raw = block.find_value(tag)
        return gemmi.cif.as_number(raw) if raw not in (None, ".", "?") else None

    a, b, c = (value(f"_cell.length_{axis}") for axis in ("a", "b", "c"))
    if a is None or b is None or c is None or a <= 0:
        return None
    return UnitCell(
        a=a, b=b, c=c,
        alpha=value("_cell.angle_alpha") or 90.0,
        beta=value("_cell.angle_beta") or 90.0,
        gamma=value("_cell.angle_gamma") or 90.0,
    )


def _read_cif_gemmi(path: str) -> Molecule:
    """Read a CIF/mmCIF atom-site loop with Gemmi when the optional extra exists."""
    try:
        import gemmi
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "Gemmi is required for parser='gemmi'; install it with "
            'pip install "molscope[cif]"'
        ) from exc

    doc = gemmi.cif.read_file(path)
    unit_cell = None
    for block in doc:
        # gemmi's cif.Block has no .cell attribute; read the _cell.* items
        # directly (absent items leave unit_cell unset).
        unit_cell = _gemmi_unit_cell(block) or unit_cell

        col = block.find_loop("_atom_site.Cartn_x")
        if not col:
            continue
        loop = col.get_loop()
        tags = [tag.split(".", 1)[1] if "." in tag else tag for tag in loop.tags]
        rows = _gemmi_loop_rows(loop)
        mol = _molecule_from_cif_rows(tags, rows, _stem(path))
        return replace(mol, unit_cell=unit_cell)
    raise ValueError(f"no _atom_site records found in {path}")



def _molecule_from_cif_rows(columns: list[str], rows: list[list[str]], name: str) -> Molecule:
    idx = {col: i for i, col in enumerate(columns)}

    def col(row, *names, default=""):
        for nm in names:
            if nm in idx and idx[nm] < len(row):
                value = row[idx[nm]]
                if value not in {".", "?"}:
                    return value
        return default

    _require_cif_coord_columns(idx, name)
    coords, els, anames, rnames, rids, icodes, chains, heteros = [], [], [], [], [], [], [], []
    for r, row in enumerate(rows, 1):
        coords.append(_cif_coords(row, idx, name, r))
        els.append(_clean_cif_value(col(row, "type_symbol")))
        anames.append(_clean_cif_value(col(row, "label_atom_id", "auth_atom_id")))
        rnames.append(_clean_cif_value(col(row, "label_comp_id", "auth_comp_id")))
        chains.append(_clean_cif_value(col(row, "auth_asym_id", "label_asym_id")))
        rid = _clean_cif_value(col(row, "auth_seq_id", "label_seq_id", default="0"))
        rids.append(int(rid) if rid.lstrip("-").isdigit() else 0)
        icodes.append(_clean_cif_value(col(row, "pdbx_PDB_ins_code")))
        heteros.append(_clean_cif_value(col(row, "group_PDB", default="ATOM")).upper() == "HETATM")

    return Molecule(
        np.array(coords, dtype=float), els, name=name,
        atom_names=anames, resnames=rnames, resids=np.array(rids, dtype=int),
        icodes=icodes, chains=chains, hetero=heteros,
    )


def read_sdf(path: str) -> Molecule:
    """Read the first molecule from an SDF / MDL MOL (V2000) file.

    Atom coordinates, element symbols, explicit bonds and V2000 bond orders are
    preserved. Any ``> <tag>`` data fields on the record are captured into
    :attr:`Molecule.properties`. For multi-record files (e.g. docking output
    with one record per pose) use :func:`read_sdf_frames` to read every pose.
    """
    with _open(path) as f:
        lines = f.readlines()
    mol, _ = _parse_sdf_record(lines, 0, path, _stem(path), record_no=1)
    return mol


def stream_sdf_frames(path: str) -> Generator[Molecule, None, None]:
    """Yield every record of a multi-record ``.sdf`` as a molecule, line by line.

    Keeps the memory footprint O(1) with respect to the number of records.
    """
    stem = _stem(path)
    record_no = 0
    with _open(path) as f:
        current_lines = []
        for line in f:
            if line.rstrip() == "$$$$":
                if current_lines:
                    record_no += 1
                    try:
                        mol, _ = _parse_sdf_record(
                            current_lines, 0, path, f"{stem}#{record_no}", record_no=record_no
                        )
                        yield mol
                    except ValueError:
                        pass
                    current_lines = []
            elif line.strip() or current_lines:
                # Skip blank padding *between* records (a blank line before a
                # record's title), but keep blanks once a block has started
                # (the comment line, line 3 of a V2000 record, is often blank).
                current_lines.append(line)
        if any(ln.strip() for ln in current_lines):
            record_no += 1
            try:
                mol, _ = _parse_sdf_record(
                    current_lines, 0, path, f"{stem}#{record_no}", record_no=record_no
                )
                yield mol
            except ValueError:
                pass


def read_sdf_frames(path: str) -> list[Molecule]:
    """Read every record of a multi-record ``.sdf`` as a list of molecules.

    SDF is the common output format for docking tools (AutoDock Vina, Gnina,
    Smina), with one record per pose and the score carried as a ``> <tag>`` data
    field (e.g. ``minimizedAffinity``, ``CNNaffinity``, ``CNNscore``). Each
    returned :class:`Molecule` keeps its 3D coordinates and exposes the record's
    data fields via :attr:`Molecule.properties`, so poses can be ranked, filtered
    or fed straight into descriptor and contact-map analysis without an RDKit
    round-trip through SMILES (which would discard the 3D pose).

    Records that fail to parse are skipped rather than aborting the whole file;
    an all-empty file raises :class:`ValueError`.
    """
    frames = list(stream_sdf_frames(path))
    if not frames:
        raise _parse_error(path, "SDF", "no readable records in file")
    return frames


def _parse_sdf_record(
    lines: list[str], start: int, path: str, name: str, record_no: int
) -> tuple[Molecule, int]:
    """Parse one V2000 record beginning at ``lines[start]``.

    Returns the molecule and the index of the line just past the record's
    ``$$$$`` terminator (or the end of file). ``start`` points at the record's
    title line; the counts line is therefore at ``start + 3``.
    """
    counts_idx = start + 3
    if counts_idx >= len(lines):
        raise _parse_error(path, "SDF", "file has fewer than the 4 required header lines")
    counts = lines[counts_idx]
    if "V3000" in counts.upper():
        raise _parse_error(
            path, "SDF",
            "V3000 connection tables are not supported (only V2000); "
            "convert the file with RDKit or OpenBabel first", counts_idx + 1,
        )
    try:
        n_atoms = int(counts[:3])
        n_bonds = int(counts[3:6])
    except ValueError:
        raise _parse_error(
            path, "SDF", f"malformed counts line: {counts.rstrip()!r}", counts_idx + 1,
        ) from None
    atom_start = counts_idx + 1
    bond_start = atom_start + n_atoms
    if len(lines) < bond_start + n_bonds:
        raise _parse_error(
            path, "SDF",
            f"counts line declares {n_atoms} atoms and {n_bonds} bonds, "
            f"but the file has only {len(lines) - atom_start} block lines after it",
            counts_idx + 1,
        )

    coords, els, formal_charges = [], [], []
    for offset, line in enumerate(lines[atom_start:atom_start + n_atoms]):
        try:
            coords.append((float(line[0:10]), float(line[10:20]), float(line[20:30])))
        except ValueError:
            raise _parse_error(
                path, "SDF", f"could not read atom coordinates from {line.rstrip()!r}",
                atom_start + offset + 1,
            ) from None
        els.append(line[31:34].strip())
        formal_charges.append(_sdf_charge_from_code(line[36:39].strip()))

    bonds, orders = [], []
    for offset, line in enumerate(lines[bond_start:bond_start + n_bonds]):
        try:
            a = int(line[0:3]) - 1
            b = int(line[3:6]) - 1
            order = int(line[6:9])
        except ValueError:
            raise _parse_error(
                path, "SDF", f"could not read bond record from {line.rstrip()!r}",
                bond_start + offset + 1,
            ) from None
        bonds.append((a, b))
        orders.append(_sdf_bond_order(order))

    i = bond_start + n_bonds
    while i < len(lines) and not lines[i].startswith("M  END"):
        if lines[i].startswith("M  CHG"):
            _apply_sdf_charge_line(lines[i], formal_charges)
        i += 1
    properties, end = _parse_sdf_data_block(lines, i)

    bond_index = np.array(bonds, dtype=int).reshape(-1, 2)
    bond_orders = np.array(orders, dtype=float) if bonds else None
    title = lines[start].strip()
    mol = Molecule(
        np.array(coords, dtype=float).reshape(-1, 3), els, name=title or name,
        bond_index=bond_index if bonds else None,
        bond_orders=bond_orders,
        formal_charges=np.array(formal_charges, dtype=int),
        properties=properties,
    )
    return mol, end


def _parse_sdf_data_block(lines: list[str], start: int) -> tuple[dict[str, str], int]:
    """Read the ``> <tag>`` data fields after ``M  END`` up to ``$$$$``.

    ``start`` points at (or before) the ``M  END`` line. Returns the parsed
    ``{tag: value}`` map and the index just past the ``$$$$`` terminator (or the
    end of file when the final record omits it).
    """
    properties: dict[str, str] = {}
    i = start
    while i < len(lines) and not lines[i].startswith("M  END"):
        i += 1
    i += 1  # step past "M  END"
    while i < len(lines):
        line = lines[i]
        if line.rstrip() == "$$$$":
            return properties, i + 1
        tag = _sdf_tag_name(line)
        if tag is not None:
            value_lines: list[str] = []
            i += 1
            while i < len(lines):
                v = lines[i]
                if not v.strip() or v.rstrip() == "$$$$" or _sdf_tag_name(v) is not None:
                    break
                value_lines.append(v.rstrip("\n"))
                i += 1
            if tag not in properties:  # first occurrence wins on duplicate tags
                properties[tag] = "\n".join(value_lines).strip()
        else:
            i += 1
    return properties, i


def _sdf_tag_name(line: str) -> Optional[str]:
    """Return the tag from an SDF data header line ``> <tag>``, else ``None``."""
    if not line.lstrip().startswith(">"):
        return None
    lo, hi = line.find("<"), line.rfind(">")
    if lo != -1 and hi > lo:
        return line[lo + 1:hi].strip() or None
    return None


def write_xyz(molecule: Molecule, path: str) -> None:
    """Write a molecule to an ``.xyz`` file (unknown elements written as ``X``)."""
    with _open(path, "w") as f:
        f.write(f"{len(molecule)}\n{molecule.name}\n")
        for element, (x, y, z) in zip(molecule.elements, molecule.coords):
            f.write(f"{element or 'X':<2} {x:15.8f} {y:15.8f} {z:15.8f}\n")


def write_pdb(molecule: Molecule, path: str) -> None:
    """Write a molecule to a ``.pdb`` file, preserving metadata when present."""
    with _open(path, "w") as f:
        f.write(_molecule_to_pdb_string(molecule))


def _molecule_to_pdb_string(molecule: Molecule) -> str:
    """Serialise a molecule to PDB text, preserving metadata when present."""
    n = len(molecule)
    names = molecule.atom_names or [e or "X" for e in molecule.elements]
    resnames = molecule.resnames or ["MOL"] * n
    chains = molecule.chains or ["A"] * n
    resids = molecule.resids if len(molecule.resids) else np.ones(n, dtype=int)
    icodes = molecule.icodes or [""] * n
    heteros = molecule.hetero if len(molecule.hetero) else [False] * n
    lines = [
        _pdb_atom_line(
            serial + 1, names[serial], resnames[serial], chains[serial] or "A",
            int(resids[serial]), molecule.elements[serial], *molecule.coords[serial],
            icode=icodes[serial], hetero=heteros[serial],
        )
        for serial in range(n)
    ]
    if molecule.bond_index is not None:
        lines.extend(
            _pdb_conect_line(int(i) + 1, int(j) + 1)
            for i, j in molecule.bond_index
        )
    lines.append("END\n")
    return "".join(lines)


# -- internals --------------------------------------------------------------


def _xyz_block(block: list[str], name: str, path=None, expected=None,
               first_line: int = 1) -> Molecule:
    path = path if path is not None else name
    coords, elements = [], []
    for offset, line in enumerate(block):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        try:
            if len(tokens) >= 4 and not _is_float(tokens[0]):
                elements.append(tokens[0])
                coords.append(tuple(float(t) for t in tokens[1:4]))
            else:
                elements.append("")
                coords.append(tuple(float(t) for t in tokens[:3]))
        except (ValueError, IndexError):
            raise _parse_error(
                path, "XYZ", f"could not read x/y/z coordinates from {stripped!r}",
                first_line + offset,
            ) from None
    if expected is not None and len(coords) != expected:
        raise _parse_error(
            path, "XYZ",
            f"header declares {expected} atoms but {len(coords)} were found",
            first_line - 2,
        )
    return Molecule(np.array(coords, dtype=float).reshape(-1, 3), elements, name=name)


def _cif_tokens(text: str) -> list[str]:
    """Tokenize enough CIF/mmCIF syntax for atom-site coordinate loops."""
    tokens: list[str] = []
    i, n = 0, len(text)
    at_line_start = True
    while i < n:
        char = text[i]
        if at_line_start and char == ";":
            i += 1
            start = i
            end = text.find("\n;", i)
            if end == -1:
                tokens.append(text[start:].rstrip("\n"))
                break
            tokens.append(text[start:end])
            i = end + 2
            while i < n and text[i] != "\n":
                i += 1
            at_line_start = True
            continue
        if char.isspace():
            at_line_start = char == "\n"
            i += 1
            continue
        if char == "#":
            while i < n and text[i] != "\n":
                i += 1
            at_line_start = True
            continue
        if char in ("'", '"'):
            quote = char
            i += 1
            start = i
            while i < n and text[i] != quote:
                i += 1
            tokens.append(text[start:i])
            i += 1 if i < n else 0
            at_line_start = False
            continue

        start = i
        while i < n and not text[i].isspace() and text[i] != "#":
            i += 1
        tokens.append(text[start:i])
        at_line_start = False
    return tokens


def _gemmi_loop_rows(loop) -> list[list[str]]:
    width = loop.width()
    values = list(loop.values)
    return [values[i:i + width] for i in range(0, len(values), width)]


def _cif_control_token(token: str) -> bool:
    lower = token.lower()
    return lower == "loop_" or lower.startswith(("data_", "save_")) or token.startswith("_")


def _clean_cif_value(value: str) -> str:
    """Normalize CIF null placeholders to MolScope's empty-string metadata."""
    return "" if value in {".", "?"} else value


def _sdf_bond_order(code: int) -> float:
    # V2000 uses 4 for aromatic bonds, not a numeric quadruple bond.
    return 1.5 if code == 4 else float(code)


def _sdf_charge_from_code(token: str) -> int:
    if not token:
        return 0
    try:
        code = int(token)
    except ValueError:
        return 0
    return {0: 0, 1: 3, 2: 2, 3: 1, 5: -1, 6: -2, 7: -3}.get(code, 0)


def _apply_sdf_charge_line(line: str, charges: list[int]) -> None:
    tokens = line.split()
    if len(tokens) < 4:
        return
    try:
        count = int(tokens[2])
    except ValueError:
        return
    pairs = tokens[3:3 + 2 * count]
    for i in range(0, len(pairs), 2):
        try:
            atom_idx = int(pairs[i]) - 1
            charge = int(pairs[i + 1])
        except (ValueError, IndexError):
            continue
        if 0 <= atom_idx < len(charges):
            charges[atom_idx] = charge


def _parse_pdb_models(
    path: str, altloc: str = "primary"
) -> tuple[list[dict], Optional[Molecule.UnitCell]]:
    """Return a tuple of (list of per-model records, unit_cell)."""
    from .molecule import UnitCell

    _validate_altloc_policy(altloc)
    models: list[dict] = []
    cur = _new_record()
    global_conect: list[tuple[int, int]] = []
    unit_cell = None

    def flush():
        nonlocal cur
        if cur["atoms"]:
            models.append(_record_from_atoms(cur["atoms"], cur["conect"], altloc))
            cur = _new_record()

    with _open(path) as f:
        for lineno, line in enumerate(f, 1):
            record = line[:6].strip()
            if record == "CRYST1":
                try:
                    unit_cell = UnitCell(
                        a=float(line[6:15]),
                        b=float(line[15:24]),
                        c=float(line[24:33]),
                        alpha=float(line[33:40]),
                        beta=float(line[40:47]),
                        gamma=float(line[47:54]),
                    )
                except ValueError:
                    pass
            elif record in ("MODEL", "ENDMDL"):
                flush()
            elif record in ("ATOM", "HETATM"):
                atom = _parse_pdb_atom(line, len(cur["atoms"]) + 1, path, lineno)
                atom["hetero"] = record == "HETATM"
                cur["atoms"].append(atom)
            elif record == "CONECT":
                pairs = _parse_conect(line)
                if cur["atoms"]:
                    cur["conect"].extend(pairs)
                else:
                    global_conect.extend(pairs)
    flush()
    if global_conect:
        for model in models:
            model["conect"].extend(global_conect)
    return models, unit_cell



def _new_record() -> dict:
    return {"atoms": [], "conect": []}


def _validate_altloc_policy(altloc: str) -> None:
    if altloc not in _ALTLOC_POLICIES:
        choices = "', '".join(_ALTLOC_POLICIES)
        raise ValueError(f"altloc must be one of '{choices}', got {altloc!r}")


def _parse_pdb_atom(line: str, fallback_serial: int, path=None, lineno=None) -> dict:
    serial = line[6:11].strip()
    serial_value = int(serial) if serial.lstrip("-").isdigit() else fallback_serial
    element = line[76:78].strip()
    atom_name = line[12:16].strip()
    if not element:
        element = atom_name.lstrip("0123456789")[:2]
    resid = line[22:26].strip()
    try:
        coords = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except ValueError:
        raise _parse_error(
            path, "PDB",
            f"could not read coordinate columns 31-54 from record: {line.rstrip()!r}",
            lineno,
        ) from None
    return {
        "serial": serial_value,
        "coords": coords,
        "element": element,
        "atom_name": atom_name,
        "resname": line[17:20].strip(),
        "chain": line[21].strip(),
        "resid": int(resid) if resid.lstrip("-").isdigit() else 0,
        "icode": line[26].strip() if len(line) > 26 else "",
        "altloc": line[16] if len(line) > 16 else " ",
        "occupancy": _pdb_float(line[54:60], default=0.0),
    }


def _record_from_atoms(atoms: list[dict], conect: list[tuple[int, int]], altloc: str) -> dict:
    selected = _select_altloc_atoms(atoms, altloc)
    return {
        "coords": [atom["coords"] for atom in selected],
        "elements": [atom["element"] for atom in selected],
        "atom_names": [atom["atom_name"] for atom in selected],
        "resnames": [atom["resname"] for atom in selected],
        "chains": [atom["chain"] for atom in selected],
        "resids": [atom["resid"] for atom in selected],
        "icodes": [atom["icode"] for atom in selected],
        "serials": [atom["serial"] for atom in selected],
        "heteros": [atom.get("hetero", False) for atom in selected],
        "conect": list(conect),
    }


def _select_altloc_atoms(atoms: list[dict], altloc: str) -> list[dict]:
    if altloc == "all":
        return atoms
    if altloc == "primary":
        return [atom for atom in atoms if atom["altloc"] in _PRIMARY_ALTLOCS]

    selected: dict[tuple, tuple[int, dict]] = {}
    for order, atom in enumerate(atoms):
        key = _altloc_group_key(atom)
        if key not in selected:
            selected[key] = (order, atom)
            continue
        if altloc == "highest_occupancy":
            selected[key] = _best_altloc(selected[key], (order, atom))
    return [atom for _, atom in sorted(selected.values(), key=lambda item: item[0])]


def _altloc_group_key(atom: dict) -> tuple:
    return (
        atom["chain"],
        atom["resid"],
        atom["icode"],
        atom["resname"],
        atom["atom_name"],
    )


def _best_altloc(left: tuple[int, dict], right: tuple[int, dict]) -> tuple[int, dict]:
    _, left_atom = left
    _, right_atom = right
    if right_atom["occupancy"] > left_atom["occupancy"]:
        return right
    if right_atom["occupancy"] < left_atom["occupancy"]:
        return left
    if _altloc_rank(right_atom["altloc"]) < _altloc_rank(left_atom["altloc"]):
        return right
    return left


def _altloc_rank(code: str) -> int:
    if code in (" ", ""):
        return 0
    if code == "A":
        return 1
    return 2


def _pdb_float(text: str, default: float = 0.0) -> float:
    try:
        return float(text)
    except ValueError:
        return default


def _molecule_from_record(rec: dict, name: str) -> Molecule:
    bond_index = _pdb_conect_bonds(rec)
    return Molecule(
        np.array(rec["coords"], dtype=float), rec["elements"], name=name,
        atom_names=rec["atom_names"], resnames=rec["resnames"],
        resids=np.array(rec["resids"], dtype=int), icodes=rec.get("icodes", []),
        chains=rec["chains"],
        hetero=rec.get("heteros", []),
        bond_index=bond_index if len(bond_index) else None,
    )


def _parse_conect(line: str) -> list[tuple[int, int]]:
    fields = [line[6:11], line[11:16], line[16:21], line[21:26], line[26:31]]
    if not fields[0].strip():
        fields = line.split()[1:]
    if len(fields) < 2:
        return []
    try:
        source = int(fields[0])
    except ValueError:
        return []
    pairs = []
    for token in fields[1:]:
        if not token.strip():
            continue
        try:
            target = int(token)
        except ValueError:
            continue
        if target != source:
            pairs.append((source, target))
    return pairs


def _pdb_conect_bonds(rec: dict) -> np.ndarray:
    serial_to_idx = {serial: i for i, serial in enumerate(rec["serials"])}
    seen = set()
    bonds = []
    for a_serial, b_serial in rec["conect"]:
        if a_serial not in serial_to_idx or b_serial not in serial_to_idx:
            continue
        i, j = serial_to_idx[a_serial], serial_to_idx[b_serial]
        if i == j:
            continue
        pair = tuple(sorted((i, j)))
        if pair not in seen:
            seen.add(pair)
            bonds.append(pair)
    return np.array(bonds, dtype=int).reshape(-1, 2)


def _pdb_atom_line(
    serial, atom_name, resname, chain, resid, element, x, y, z, icode="", hetero=False
):
    """Build a single fixed-column ``ATOM``/``HETATM`` record (80 columns)."""
    line = list(" " * 80)

    def put(value: str, start: int):  # start is 1-based
        line[start - 1:start - 1 + len(value)] = value

    put("HETATM" if hetero else "ATOM", 1)
    put(f"{serial:>5}", 7)
    put(f"{(atom_name or 'X')[:4]:<4}", 13)
    put(f"{(resname or 'MOL')[:3]:>3}", 18)
    put((chain or "A")[0], 22)
    put(f"{resid:>4}", 23)
    put((icode or " ")[0], 27)
    put(f"{x:8.3f}", 31)
    put(f"{y:8.3f}", 39)
    put(f"{z:8.3f}", 47)
    put(f"{1.0:6.2f}", 55)
    put(f"{0.0:6.2f}", 61)
    put(f"{(element or 'X'):>2}", 77)
    return "".join(line).rstrip() + "\n"


def stream(path: str, altloc: str = "primary") -> Generator[Molecule, None, None]:
    """Yield frames/models one by one, picking the stream parser from the file extension.

    Transparently handles gzip-compressed files (``.pdb.gz``, ``.xyz.gz``, ``.sdf.gz``).
    """
    path = os.fspath(path)
    ext = _data_extension(path)
    if ext == ".pdb":
        yield from stream_pdb_models(path, altloc=altloc)
    elif ext == ".xyz":
        yield from stream_xyz_frames(path)
    elif ext == ".sdf":
        yield from stream_sdf_frames(path)
    else:
        raise ValueError(
            f"Streaming is not supported for file type {ext!r}; expected .pdb/.xyz/.sdf"
        )


def stream_xyz_frames(path: str) -> Generator[Molecule, None, None]:
    """Yield every frame of a (possibly multi-frame) ``.xyz`` trajectory."""
    stem = _stem(path)
    frame_idx = 0
    with _open(path) as f:
        while True:
            first_line = f.readline()
            if not first_line:
                break
            if not first_line.strip():
                continue
            tokens = first_line.split()
            if tokens and tokens[0].isdigit():
                count = int(tokens[0])
                # Skip the comment line
                _ = f.readline()
                block = []
                for _ in range(count):
                    line = f.readline()
                    if not line:
                        break
                    block.append(line)

                frame_idx += 1
                yield _xyz_block(
                    block, name=f"{stem}#{frame_idx}", path=path,
                    expected=count, first_line=1
                )
            else:
                # Bare coordinate dump: consume the rest as one frame.
                remaining = [first_line]
                for line in f:
                    remaining.append(line)
                yield _xyz_block(remaining, name=stem, path=path, first_line=1)
                break


def stream_pdb_models(path: str, altloc: str = "primary") -> Generator[Molecule, None, None]:
    """Yield every model from a ``.pdb`` file as a molecule.

    Only reads and parses one model at a time into memory, keeping the memory
    footprint O(1) with respect to the number of frames.
    """
    _validate_altloc_policy(altloc)
    stem = _stem(path)

    # Fast pre-scan for CRYST1 and global/trailing CONECT records, so the main
    # pass does not have to look ahead for them.
    global_conect = []
    unit_cell = None
    in_model = False
    with _open(path) as f:
        for line in f:
            record = line[:6].strip()
            if record == "CRYST1":
                try:
                    from .molecule import UnitCell
                    unit_cell = UnitCell(
                        a=float(line[6:15]),
                        b=float(line[15:24]),
                        c=float(line[24:33]),
                        alpha=float(line[33:40]),
                        beta=float(line[40:47]),
                        gamma=float(line[47:54]),
                    )
                except ValueError:
                    pass
            elif record == "MODEL":
                in_model = True
            elif record == "ENDMDL":
                in_model = False
            elif record == "CONECT" and not in_model:
                global_conect.extend(_parse_conect(line))

    cur_atoms = []
    cur_conect = []
    model_idx = 0

    with _open(path) as f:
        for lineno, line in enumerate(f, 1):
            record = line[:6].strip()
            if record == "MODEL":
                if cur_atoms:
                    model_idx += 1
                    rec = _record_from_atoms(cur_atoms, cur_conect + global_conect, altloc)
                    mol = _molecule_from_record(rec, f"{stem}#{model_idx}")
                    yield replace(mol, unit_cell=unit_cell)
                    cur_atoms = []
                    cur_conect = []
            elif record == "ENDMDL":
                if cur_atoms:
                    model_idx += 1
                    rec = _record_from_atoms(cur_atoms, cur_conect + global_conect, altloc)
                    mol = _molecule_from_record(rec, f"{stem}#{model_idx}")
                    yield replace(mol, unit_cell=unit_cell)
                    cur_atoms = []
                    cur_conect = []
            elif record in ("ATOM", "HETATM"):
                atom = _parse_pdb_atom(line, len(cur_atoms) + 1, path, lineno)
                atom["hetero"] = record == "HETATM"
                cur_atoms.append(atom)
            elif record == "CONECT":
                cur_conect.extend(_parse_conect(line))

        if cur_atoms:
            model_idx += 1
            rec = _record_from_atoms(cur_atoms, cur_conect + global_conect, altloc)
            mol = _molecule_from_record(rec, f"{stem}#{model_idx}")
            yield replace(mol, unit_cell=unit_cell)


def _pdb_conect_line(a: int, b: int) -> str:
    return f"CONECT{a:5d}{b:5d}\n"


def _open(path: str, mode: str = "r"):
    """Open a file, transparently handling gzip by the ``.gz`` suffix."""
    path = os.fspath(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t")
    return open(path, mode)


def _data_extension(path: str) -> str:
    """Extension ignoring a trailing ``.gz`` (so ``a.pdb.gz`` -> ``.pdb``)."""
    path = os.fspath(path)
    base = path[:-3] if path.endswith(".gz") else path
    return os.path.splitext(base)[1].lower()


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _stem(path: str) -> str:
    path = os.fspath(path)
    base = os.path.basename(path)
    if base.endswith(".gz"):
        base = base[:-3]
    return os.path.splitext(base)[0]
