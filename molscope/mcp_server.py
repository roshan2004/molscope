"""A Model Context Protocol (MCP) server that exposes MolScope to AI assistants.

This wraps MolScope's existing analysis features as MCP *tools* so an assistant
such as Claude Code or Claude Desktop can drive them in natural language: load a
structure (local file, RCSB id, or ``"smiles:<SMILES>"`` string), compute
descriptor tables, assign secondary structure, build contact maps, find binding
sites, summarise a molecular graph, coarse-grain, and render PNG figures.

It adds no new science. Every tool is a thin, faithful adapter over the public
``molscope`` API documented in the user guide, returning JSON text (so results
are easy for a model to read) or a PNG image for the render tools.

Run it over stdio, which is how local MCP clients launch a server::

    molscope-mcp            # console script (needs the ``mcp`` extra)
    python -m molscope.mcp_server

Install the optional dependency with ``pip install "molscope[mcp]"``. Register it
with a client by pointing the client at the ``molscope-mcp`` command; for Claude
Code that is ``claude mcp add molscope -- molscope-mcp``.
"""

from __future__ import annotations

import functools
import io
import json
import math
import os
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .molecule import Molecule

# Cap on how many per-item rows a tool will inline before truncating, so a large
# structure cannot flood the model's context with thousands of residues/pairs.
_MAX_ROWS = 2000

# A ``source`` beginning with this (case-insensitive) is treated as a SMILES
# string rather than a path/PDB id, so every tool can analyse a molecule given
# only its SMILES. A bare token like "C" would be ambiguous with a filename, so
# the prefix makes the intent explicit.
_SMILES_PREFIX = "smiles:"

# Maps an importable top-level module to the MolScope optional extra that ships
# it, so a bare ``ModuleNotFoundError`` can be turned into install guidance.
_EXTRA_FOR_MODULE = {
    "rdkit": "chem",
    "gemmi": "cif",
    "networkx": "graph",
    "torch": "pyg",
    "torch_geometric": "pyg",
    "dgl": "dgl",
    "scipy": "fast",
    "openpyxl": "xlsx",
}


def _dependency_error(exc: ImportError) -> ImportError:
    """Rewrite a raw import failure into actionable, model-friendly guidance.

    MolScope's own code already raises messages naming the ``molscope[...]``
    extra; those are passed through unchanged. A bare ``No module named 'rdkit'``
    from a deeper import is mapped to the extra that provides it.
    """
    message = str(exc)
    if "molscope[" in message:
        return exc
    top = (getattr(exc, "name", None) or "").split(".")[0]
    extra = _EXTRA_FOR_MODULE.get(top)
    if extra:
        return ImportError(
            f"this tool needs the optional '{top}' backend; "
            f'install it with: pip install "molscope[{extra}]"'
        )
    if top:
        return ImportError(f"this tool needs the optional package {top!r}, which is not installed")
    return exc


def _friendly_errors(fn: Callable) -> Callable:
    """Wrap a tool so a missing optional dependency surfaces as install guidance."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ImportError as exc:
            raise _dependency_error(exc) from exc

    return wrapper


def _load(source: str, bond_perception: str = "geometric", protonation: str = "none",
          ph: float = 7.0) -> Molecule:
    """Resolve ``source`` to a :class:`~molscope.molecule.Molecule`.

    ``source`` is one of:

    - a path to a local coordinate file (``.pdb``, ``.cif``, ``.xyz``, ``.sdf``,
      optionally gzipped);
    - a 4-character RCSB PDB id, which is fetched and cached;
    - ``"smiles:<SMILES>"`` (e.g. ``"smiles:CCO"``), which builds a molecule from
      the SMILES with one RDKit-generated 3D conformer (needs the ``chem`` extra).
      The coordinates are *generated*, not experimental, so SMILES input suits
      topology-based work (descriptors, graphs) more than geometry-dependent
      results (precise distances, RMSD against experiment).

    ``bond_perception="template"`` attaches RDKit residue-template bonds (PDB
    only); ``protonation="standard"`` adds idealised pH-7 side-chain charges, and
    ``protonation="pka"`` predicts them from the structure with PROPKA at ``ph``
    (see :func:`molscope.read_pdb`). All are ignored for SMILES input, which
    already carries RDKit-perceived bonds and formal charges.
    """
    from .io import fetch, read

    token = source.strip()
    if token.lower().startswith(_SMILES_PREFIX):
        from .io import read_smiles

        smiles = token[len(_SMILES_PREFIX):].strip()
        if not smiles:
            raise ValueError('empty SMILES; pass e.g. source="smiles:CCO"')
        return read_smiles(smiles)
    if os.path.exists(source):
        return read(source, bond_perception=bond_perception, protonation=protonation, ph=ph)
    if len(token) == 4 and token.isalnum():
        return fetch(token, bond_perception=bond_perception, protonation=protonation, ph=ph)
    raise FileNotFoundError(
        f"{source!r} is neither an existing file nor a 4-character PDB id; "
        "pass a path like 'examples/data/1ubq.pdb', an id like '1ubq', "
        'or a SMILES like "smiles:CCO"'
    )


def _dock_path(source: str) -> str:
    """Resolve a docking-output SDF path. Unlike ``_load`` this keeps every pose.

    Docking tools read a multi-record ``.sdf`` (one record per pose), so they
    take a literal local file path, never a PDB id or SMILES.
    """
    path = os.path.abspath(os.path.expanduser(source.strip()))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{source!r}: no such SDF file. Docking tools take a path to a local "
            "docking-output .sdf (one record per pose)."
        )
    return path


def _three_state(code: str) -> str:
    return {"H": "H", "G": "H", "I": "H", "E": "E", "B": "E"}.get(code, "C")


def _num(value) -> Optional[float]:
    """A JSON-safe float: ``None`` for NaN/inf (invalid JSON), else ``float``."""
    f = float(value)
    return None if (math.isnan(f) or math.isinf(f)) else f


def _resolve_count(n: Optional[int], fraction: Optional[float], total: int) -> int:
    """Resolve an absolute ``n`` or a ``fraction`` of ``total`` to a positive count.

    Exactly one of ``n``/``fraction`` must be given. A fraction rounds up so that
    e.g. 5% of a small table still selects at least one molecule.
    """
    if (n is None) == (fraction is None):
        raise ValueError("provide exactly one of n or fraction")
    if fraction is not None:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        return max(1, math.ceil(fraction * total))
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return n


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars/arrays into plain, JSON-safe Python values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, float):
        return _num(value)
    return value


def build_server():  # noqa: C901 - a flat list of small tool adapters reads clearly
    """Construct and return the configured :class:`FastMCP` server.

    Imported lazily so ``import molscope.mcp_server`` works even when the ``mcp``
    extra is absent; only building/running the server needs it.
    """
    from mcp.server.fastmcp import FastMCP, Image
    from mcp.types import ToolAnnotations

    server = FastMCP("molscope")

    # Annotation presets describing each tool's effect for MCP clients.
    # "Net" tools accept a PDB id and so may reach out to RCSB (open world);
    # "local" tools only read a local table. Write tools can emit files.
    def _read(net: bool) -> ToolAnnotations:
        return ToolAnnotations(
            readOnlyHint=True, idempotentHint=True, destructiveHint=False, openWorldHint=net
        )

    def _write(net: bool) -> ToolAnnotations:
        return ToolAnnotations(
            readOnlyHint=False, idempotentHint=True, destructiveHint=False, openWorldHint=net
        )

    READ_NET = _read(True)  # analysis tools that take a structure source
    READ_LOCAL = _read(False)  # tools that only read a local table
    WRITE_NET = _write(True)  # render tools that may save a figure
    WRITE_LOCAL = ToolAnnotations(  # prepare_dataset: may write files, order matters
        readOnlyHint=False, idempotentHint=False, destructiveHint=False, openWorldHint=False
    )

    @server.tool(title="Summarise structure", annotations=READ_NET)
    @_friendly_errors
    def summarize_structure(source: str) -> str:
        """Load a structure and return a one-line summary.

        ``source`` is a local coordinate-file path, a 4-character PDB id, or a
        SMILES string written as ``"smiles:<SMILES>"`` (e.g. ``"smiles:CCO"``;
        builds one RDKit conformer, needs the ``chem`` extra). The same three
        forms are accepted by every tool that takes a ``source``. The summary
        reports atom count, formula, chains, and bounding-box size.
        """
        return _load(source).summary()

    @server.tool(title="Compute descriptors", annotations=READ_NET)
    @_friendly_errors
    def compute_descriptors(sources: list[str], preset: Optional[str] = None) -> str:
        """Compute MolScope's fixed-width structural descriptors for one or more structures.

        ``sources`` is a list of file paths, PDB ids, and/or ``"smiles:<SMILES>"``
        strings. ``preset`` selects a descriptor preset (omit for the default
        set). Returns JSON with the
        ordered ``feature_names`` and one ``rows`` entry per source. This is the
        batch tool: pass several structures to get a comparable descriptor table.
        """
        # Compute per-structure (rather than via featurize_many) so a mix of file
        # paths and fetched PDB ids works through the same _load resolution.
        kwargs = {} if preset is None else {"preset": preset}
        names: Optional[list[str]] = None
        rows = []
        for src in sources:
            values = _load(src).descriptors(**kwargs)
            if names is None:
                names = list(values.keys())
            rows.append({"source": src, "values": [_jsonable(values[n]) for n in names]})
        return json.dumps(
            {"feature_names": names or [], "n_features": len(names or []), "rows": rows},
            indent=2,
        )

    @server.tool(title="Assign secondary structure", annotations=READ_NET)
    @_friendly_errors
    def secondary_structure(source: str) -> str:
        """Assign protein secondary structure with MolScope's simplified DSSP.

        Returns JSON with per-residue codes (8-state DSSP letters) and a 3-state
        helix/strand/coil composition summary. Needs backbone N/CA/C/O atoms, so
        use a protein read from PDB/mmCIF.
        """
        ss = _load(source).secondary_structure()
        codes = ss.codes.tolist()
        resids = ss.resids.tolist()
        residues = [
            {"chain": c, "resid": int(r), "resname": rn, "code": code}
            for c, r, rn, code in zip(ss.chains, resids, ss.resnames, codes)
        ]
        total = len(codes) or 1
        helix = sum(1 for c in codes if _three_state(c) == "H")
        strand = sum(1 for c in codes if _three_state(c) == "E")
        coil = total - helix - strand
        truncated = len(residues) > _MAX_ROWS
        return json.dumps(
            {
                "n_residues": len(codes),
                "composition": {
                    "helix": helix,
                    "strand": strand,
                    "coil": coil,
                    "helix_fraction": helix / total,
                    "strand_fraction": strand / total,
                    "coil_fraction": coil / total,
                },
                "residues": residues[:_MAX_ROWS],
                "residues_truncated": truncated,
            },
            indent=2,
        )

    @server.tool(title="Contact map", annotations=READ_NET)
    @_friendly_errors
    def contact_map(
        source: str,
        cutoff: float = 8.0,
        level: str = "residue",
        method: str = "ca",
        min_seq_sep: int = 0,
    ) -> str:
        """Build a contact map and return its summary plus contacting pairs.

        ``level`` is ``"residue"`` or ``"atom"``; for residue level ``method`` is
        ``"ca"``, ``"com"`` or ``"min"``. ``min_seq_sep`` drops same-chain
        contacts closer than that many sequence positions. Returns JSON with the
        contact count, contact order, and labelled contacting pairs (truncated if
        very large). The full dense matrix is intentionally not inlined.
        """
        cmap = _load(source).contact_map(
            cutoff=cutoff, level=level, method=method, min_seq_sep=min_seq_sep
        )
        labels = list(cmap.labels)
        pairs_idx = np.argwhere(np.triu(cmap.matrix, 1) > 0)
        pairs = [
            [labels[i] if i < len(labels) else int(i), labels[j] if j < len(labels) else int(j)]
            for i, j in pairs_idx.tolist()
        ]
        truncated = len(pairs) > _MAX_ROWS
        return json.dumps(
            {
                "level": cmap.level,
                "cutoff": cmap.cutoff,
                "method": method,
                "n_labels": len(labels),
                "n_contacts": cmap.n_contacts,
                "contact_order": cmap.contact_order(),
                "pairs": pairs[:_MAX_ROWS],
                "pairs_truncated": truncated,
            },
            indent=2,
        )

    @server.tool(title="Binding site", annotations=READ_NET)
    @_friendly_errors
    def binding_site(source: str, ligand: Optional[str] = None, cutoff: float = 4.5) -> str:
        """Find protein residues around a bound ligand.

        ``ligand`` is a HETATM residue name (e.g. ``"BEN"``); omit it to use the
        single non-solvent ligand automatically. ``cutoff`` is the contact
        distance in angstrom. Returns JSON with the ligand and the binding-site
        residues ordered closest-first, each with its minimum distance.
        """
        from .contacts import binding_site as _binding_site

        site = _binding_site(_load(source), ligand=ligand, cutoff=cutoff)
        residues = [
            {
                "chain": res.chain,
                "resid": int(res.resid),
                "resname": res.resname,
                "min_distance": round(float(dist), 3),
            }
            for res, dist in zip(site.residues, site.min_distances)
        ]
        return json.dumps(
            {
                "ligand": str(site.ligand),
                "cutoff": site.cutoff,
                "n_residues": len(residues),
                "n_atom_contacts": site.n_atom_contacts,
                "residues": residues,
            },
            indent=2,
        )

    @server.tool(title="Molecular graph summary", annotations=READ_NET)
    @_friendly_errors
    def molecular_graph(
        source: str,
        preset: str = "default",
        include_chemical_features: bool = False,
        knn: Optional[int] = None,
        radius: Optional[float] = None,
        min_seq_sep: int = 0,
    ) -> str:
        """Summarise the atom/bond molecular graph MolScope would export for ML.

        Returns JSON with node and edge counts, the node-feature matrix shape,
        and the ordered node/edge feature names for ``preset``. Set
        ``include_chemical_features=True`` to attach RDKit-backed aromatic flags
        (needs the ``chem`` extra). For spatial-proximity graphs pass ``knn=k``
        (each atom's ``k`` nearest neighbours) or ``radius=r`` (all pairs within
        ``r`` angstrom) instead of covalent bonds, and ``min_seq_sep`` to drop
        same-chain edges whose residue-id separation is below the threshold
        (needs residue ids). This describes the graph; use the Python API or CLI
        to export the actual PyG/DGL/NetworkX object.
        """
        from .graph import edge_feature_names, node_feature_names

        graph = _load(source).to_graph(
            include_chemical_features=include_chemical_features,
            knn=knn,
            radius=radius,
            min_seq_sep=min_seq_sep,
        )
        node_matrix = graph.node_features(preset)
        return json.dumps(
            {
                "n_nodes": int(graph.n_atoms),
                "n_edges": int(graph.n_bonds),
                "preset": preset,
                "knn": knn,
                "radius": radius,
                "min_seq_sep": min_seq_sep,
                "node_feature_matrix_shape": list(node_matrix.shape),
                "node_feature_names": list(node_feature_names(preset)),
                "edge_feature_names": list(edge_feature_names(preset)),
            },
            indent=2,
        )

    @server.tool(title="Coarse-grain", annotations=READ_NET)
    @_friendly_errors
    def coarse_grain(source: str, mapping: str = "residue_com") -> str:
        """Coarse-grain a structure to beads and report the assignment.

        ``mapping`` is ``"residue_com"``, ``"residue_centroid"`` or ``"martini"``.
        Returns JSON with the bead count and the number of atoms assigned versus
        dropped. This is a mapping for inspection, not a force-field model.
        """
        beads, report = _load(source).coarse_grain(mapping=mapping, return_report=True)
        return json.dumps(
            {
                "mapping": report.mapping,
                "n_beads": report.n_beads,
                "n_bonds": len(report.bonds),
                "n_dropped_atoms": len(report.dropped_atoms),
                "n_virtual_sites": len(report.virtual_sites),
                "summary": beads.summary(),
            },
            indent=2,
        )

    @server.tool(title="Geometry", annotations=READ_NET)
    @_friendly_errors
    def geometry(source: str) -> str:
        """Report whole-structure geometry: size, mass distribution, shape.

        Returns JSON with atom count, formula, chains, centre of mass, radius of
        gyration, bounding-box dimensions, and the principal moments of inertia.
        """
        mol = _load(source)
        return json.dumps(
            {
                "n_atoms": len(mol),
                "formula": mol.formula,
                "chains": sorted(set(mol.chain_ids())),
                "center_of_mass": [_num(v) for v in mol.center_of_mass.tolist()],
                "radius_of_gyration": _num(mol.radius_of_gyration),
                "dimensions": [_num(v) for v in np.asarray(mol.dimensions).tolist()],
                "principal_moments": [_num(v) for v in mol.principal_moments().tolist()],
            },
            indent=2,
        )

    @server.tool(title="Measure distance/angle/dihedral", annotations=READ_NET)
    @_friendly_errors
    def measure(source: str, atoms: list[int]) -> str:
        """Measure a geometric quantity between atoms by 0-based index.

        Pass 2 atom indices for a distance (angstrom), 3 for an angle (degrees),
        or 4 for a dihedral (degrees).
        """
        mol = _load(source)
        if len(atoms) == 2:
            return json.dumps({"kind": "distance", "atoms": atoms,
                               "value": _num(mol.distance(*atoms)), "unit": "angstrom"})
        if len(atoms) == 3:
            return json.dumps({"kind": "angle", "atoms": atoms,
                               "value": _num(mol.angle(*atoms)), "unit": "degrees"})
        if len(atoms) == 4:
            return json.dumps({"kind": "dihedral", "atoms": atoms,
                               "value": _num(mol.dihedral(*atoms)), "unit": "degrees"})
        raise ValueError("atoms must hold 2 (distance), 3 (angle), or 4 (dihedral) indices")

    @server.tool(title="RMSD between structures", annotations=READ_NET)
    @_friendly_errors
    def rmsd(source_a: str, source_b: str, align: bool = True) -> str:
        """Root-mean-square deviation between two structures with the same atom count.

        With ``align`` (default), the structures are Kabsch-superposed first so the
        result is the minimal RMSD; set it false for the RMSD as-is. Returns the
        value in angstrom.
        """
        a, b = _load(source_a), _load(source_b)
        return json.dumps({"rmsd": _num(a.rmsd(b, align=align)), "aligned": align,
                           "n_atoms": len(a), "unit": "angstrom"})

    @server.tool(title="List ligands", annotations=READ_NET)
    @_friendly_errors
    def list_ligands(source: str, exclude_water: bool = True, exclude_ions: bool = True) -> str:
        """List the non-polymer (HETATM) groups in a structure.

        Useful before ``binding_site`` to see which ligand names are present.
        Returns JSON with each group's residue name, chain, residue id, and atom
        count. Waters and monatomic ions are excluded by default.
        """
        ligs = _load(source).ligands(exclude_water=exclude_water, exclude_ions=exclude_ions)
        return json.dumps(
            {
                "n_ligands": len(ligs),
                "ligands": [
                    {"resname": lig.resname, "chain": lig.chain, "resid": int(lig.resid),
                     "n_atoms": len(lig)}
                    for lig in ligs
                ],
            },
            indent=2,
        )

    @server.tool(title="Chain interfaces", annotations=READ_NET)
    @_friendly_errors
    def chain_interfaces(
        source: str, chain_a: Optional[str] = None, chain_b: Optional[str] = None,
        cutoff: float = 5.0,
    ) -> str:
        """Analyse inter-chain contacts.

        With both ``chain_a`` and ``chain_b``, return the residues on each side of
        that interface (within ``cutoff`` angstrom) and the atom-contact count.
        With neither, return the all-pairs chain contact matrix instead.
        """
        mol = _load(source)
        if chain_a and chain_b:
            iface = mol.interface(chain_a, chain_b, cutoff=cutoff)

            def fmt(residues):
                return [{"chain": r.chain, "resid": int(r.resid), "resname": r.resname}
                        for r in residues]

            return json.dumps(
                {"chain_a": iface.chain_a, "chain_b": iface.chain_b, "cutoff": cutoff,
                 "n_atom_contacts": iface.n_atom_contacts,
                 "residues_a": fmt(iface.residues_a), "residues_b": fmt(iface.residues_b)},
                indent=2,
            )
        ccm = mol.chain_contacts(cutoff=cutoff)
        return json.dumps(
            {"cutoff": cutoff, "chains": list(ccm.chains),
             "contact_matrix": [[int(v) for v in row] for row in ccm.matrix.tolist()]},
            indent=2,
        )

    @server.tool(title="Backbone torsions", annotations=READ_NET)
    @_friendly_errors
    def backbone_torsions(source: str) -> str:
        """Per-residue backbone dihedral angles (Ramachandran phi/psi/omega).

        Returns JSON with one entry per residue in chain/residue order. Angles are
        ``null`` where undefined (phi at a chain start, psi/omega at a chain end).
        """
        bt = _load(source).backbone_torsions()
        residues = [
            {"chain": c, "resid": int(r),
             "phi": _num(phi), "psi": _num(psi), "omega": _num(omega)}
            for c, r, phi, psi, omega in zip(
                bt.chains, bt.resids.tolist(), bt.phi.tolist(), bt.psi.tolist(), bt.omega.tolist()
            )
        ]
        return json.dumps(
            {"n_residues": len(residues), "residues": residues[:_MAX_ROWS],
             "residues_truncated": len(residues) > _MAX_ROWS},
            indent=2,
        )

    @server.tool(title="Ensemble summary", annotations=READ_NET)
    @_friendly_errors
    def ensemble_summary(source: str) -> str:
        """Summarise a multi-model (e.g. NMR) ensemble.

        Reads every model from a multi-model PDB and returns JSON with the model
        count, mean/max pairwise RMSD across models, a per-atom RMSF summary, and
        the number of conformational clusters. Errors on single-model inputs.
        """
        from .ensemble import cluster, rmsd_matrix, rmsf
        from .io import read_pdb_models

        models = read_pdb_models(source)
        if len(models) < 2:
            raise ValueError("ensemble_summary needs a multi-model file (e.g. an NMR PDB)")
        mat = rmsd_matrix(models)
        upper = mat[np.triu_indices_from(mat, k=1)]
        fluct = rmsf(models)
        return json.dumps(
            {
                "n_models": len(models),
                "mean_pairwise_rmsd": _num(upper.mean()),
                "max_pairwise_rmsd": _num(upper.max()),
                "rmsf_mean": _num(fluct.mean()),
                "rmsf_max": _num(fluct.max()),
                "n_clusters": cluster(models).n_clusters,
                "unit": "angstrom",
            },
            indent=2,
        )

    @server.tool(title="Chemical features", annotations=READ_NET)
    @_friendly_errors
    def chemical_features(
        source: str, bond_perception: str = "template", protonation: str = "standard",
        ph: float = 7.0,
    ) -> str:
        """RDKit-perceived per-atom chemistry (needs the ``chem`` extra).

        Returns JSON with the formal-charge sum, the number of aromatic atoms and
        bonds, and the atom/bond counts RDKit assigned after sanitisation.

        ``bond_perception`` defaults to ``"template"``, which uses RDKit's
        residue-aware PDB reader so standard-residue proteins get correct bond
        orders and aromatic rings. (Plain distance-based ``"geometric"`` bonds
        miss all of that on bare PDBs.) ``protonation`` defaults to ``"standard"``
        (idealised pH-7 side-chain charges: Asp/Glu -1, Lys/Arg +1, His neutral,
        termini uncharged) so ``total_formal_charge`` is meaningful; ``"pka"``
        instead predicts side-chain charges from the structure with PROPKA at
        ``ph`` (needs the ``propka`` extra), and ``"none"`` keeps the as-modelled
        neutral state. All apply to PDB inputs only; other formats fall back to
        their explicit/geometric bonds.
        """
        is_pdb = not (os.path.exists(source) and not source.lower().endswith(
            (".pdb", ".pdb.gz", ".ent")
        ))
        bp = bond_perception if is_pdb else "geometric"
        prot = protonation if (is_pdb and bp == "template") else "none"
        feats = _load(source, bond_perception=bp, protonation=prot, ph=ph).chemical_features()
        protonation_label = {
            "standard": "standard (idealised pH 7, standard side chains only)",
            "pka": f"pka (PROPKA prediction at pH {ph:g})",
            "none": "as-modelled (no protonation assigned)",
        }.get(prot, "as-modelled (no protonation assigned)")
        return json.dumps(
            {
                "n_atoms": int(len(feats.formal_charges)),
                "total_formal_charge": int(sum(int(c) for c in feats.formal_charges)),
                "protonation": protonation_label,
                "n_aromatic_atoms": int(sum(bool(a) for a in feats.aromatic_atoms)),
                "n_bonds": int(len(feats.bond_orders)),
                "n_aromatic_bonds": int(sum(bool(a) for a in feats.aromatic_bonds)),
            },
            indent=2,
        )

    @server.tool(title="Validate mmCIF", annotations=READ_LOCAL)
    @_friendly_errors
    def validate_cif(source: str) -> str:
        """Validate an mmCIF/CIF file (needs the ``cif`` extra / gemmi).

        Returns JSON with whether the file is valid, syntax/atom-site status, block
        and atom-row counts, and any errors or warnings.
        """
        from .cif import validate_cif as _validate

        report = _validate(source)
        return json.dumps(
            {
                "path": report.path, "valid": report.valid,
                "syntax_ok": report.syntax_ok, "atom_site_ok": report.atom_site_ok,
                "n_blocks": report.n_blocks, "n_atom_site_rows": report.n_atom_site_rows,
                "dictionary_checked": report.dictionary_checked,
                "errors": list(report.errors), "warnings": list(report.warnings),
            },
            indent=2,
        )

    @server.tool(title="Prepare / QC structure", annotations=READ_NET)
    @_friendly_errors
    def prepare_structure(
        source: str, protonation: str = "standard", ph: float = 7.4,
    ) -> str:
        """Check whether a structure is ML-ready and summarise what to fix.

        ``source`` is a path to a ``.pdb``/``.cif``/``.xyz``/``.sdf`` file or a
        4-character RCSB PDB id (downloaded and cached). Returns JSON with an
        ``ml_ready`` verdict plus blockers and warnings: non-standard residues, a
        ligand/water inventory, residue-numbering gaps, backbone chain breaks,
        residues missing backbone atoms or with truncated side chains, whether
        hydrogens are present, alternate conformations / partial occupancies, and
        the net formal charge. ``protonation`` controls the net charge
        (``"standard"`` pH-7 table, ``"pka"`` PROPKA at ``ph`` (``propka`` extra),
        or ``"none"``); it needs RDKit for protein PDBs and degrades to a null
        charge with a note otherwise. The ``ml_ready`` verdict is a heuristic
        (missing backbone atoms and chain breaks are blockers).
        """
        from .structure_prep import prepare_structure as _prepare

        token = source.strip()
        if not os.path.exists(source) and len(token) == 4 and token.isalnum():
            from .io import fetch_file

            source = fetch_file(token, fmt="pdb")
        report = _prepare(source, protonation=protonation, ph=ph)
        return json.dumps(report.to_dict(), indent=2, default=_jsonable)

    @server.tool(title="Select diverse subset", annotations=READ_LOCAL)
    @_friendly_errors
    def select_diverse(
        table: str, n: Optional[int] = None, fraction: Optional[float] = None,
        descriptor_cols: Optional[list[str]] = None,
        smiles_col: Optional[str] = None, compute_descriptors: bool = False,
    ) -> str:
        """Pick a diverse subset of molecules from a CSV/XLSX table.

        ``table`` is a path to a ``.csv`` or ``.xlsx`` file of molecules. Give
        either an absolute count ``n`` or a ``fraction`` of the table in ``(0, 1]``
        (e.g. ``fraction=0.05`` for "the most diverse 5%"); pass exactly one.
        Select on existing numeric columns via ``descriptor_cols`` (e.g.
        ``["MW", "ALogP"]``), or set ``compute_descriptors`` with ``smiles_col`` to
        compute RDKit descriptors (``MolLogP`` is the ALogP equivalent) and select
        on those. Returns the chosen rows by MaxMin (farthest-first) selection.
        """
        from .library import read_table, smiles_descriptors
        from .library import select_diverse as _pick

        tab = read_table(table)
        count = _resolve_count(n, fraction, len(tab))
        if compute_descriptors:
            if not smiles_col:
                raise ValueError("compute_descriptors needs smiles_col")
            matrix, names = smiles_descriptors(tab.column(smiles_col))
            tab = tab.with_columns(names, matrix)
        elif descriptor_cols:
            names, matrix = list(descriptor_cols), tab.numeric_matrix(descriptor_cols)
        else:
            raise ValueError("provide descriptor_cols, or compute_descriptors with smiles_col")
        chosen = tab.select_rows(_pick(matrix, count))
        return json.dumps(
            {"selected": len(chosen), "of": len(tab), "requested": count,
             "descriptors": names, "rows": [dict(r) for r in chosen.rows]},
            indent=2, default=_jsonable,
        )

    @server.tool(title="Prepare dataset splits", annotations=WRITE_LOCAL)
    @_friendly_errors
    def prepare_dataset(
        table: str, split: str = "random", test: float = 0.1, val: float = 0.1,
        seed: int = 0, smiles_col: Optional[str] = None,
        descriptor_cols: Optional[list[str]] = None, compute_descriptors: bool = False,
        dedup: str = "none", fingerprints: bool = False, save_dir: Optional[str] = None,
        protonation: str = "none", ph: float = 7.0,
    ) -> str:
        """Build train/validation/test splits from a molecule table or SDF.

        ``table`` is a path to a ``.csv``/``.xlsx`` table or a multi-record
        ``.sdf``. ``split`` is ``"random"``, ``"diversity"`` (needs descriptors,
        via ``descriptor_cols`` or ``compute_descriptors`` + ``smiles_col``), or
        ``"scaffold"`` (Bemis-Murcko, needs ``smiles_col``). ``dedup`` is
        ``"none"``/``"exact"``/``"canonical"``. ``protonation="pka"`` sets each
        SMILES to its dominant ionisation state at ``ph`` (Dimorphite-DL, needs the
        ``dimorphite`` extra) before descriptors/fingerprints are computed.
        Returns a JSON summary plus each molecule's split label inline; pass
        ``save_dir`` to also write ``train.csv``/``validation.csv``/``test.csv``,
        ``descriptors.csv``, ``report.md`` and ``manifest.json`` to that directory.
        """
        from .prepare import prepare_dataset as _prepare

        dataset = _prepare(
            table, smiles_col=smiles_col, descriptor_cols=descriptor_cols,
            compute_descriptors=compute_descriptors, split=split, test=test, val=val,
            seed=seed, dedup=dedup, fingerprints=fingerprints,
            protonation=protonation, ph=ph,
        )
        label_of = {}
        for name, indices in (("train", dataset.split.train),
                              ("validation", dataset.split.val),
                              ("test", dataset.split.test)):
            for i in indices:
                label_of[i] = name
        id_col = dataset.table.columns[0] if dataset.table.columns else None
        assignments = [
            {"id": (dataset.table.rows[i].get(id_col) if id_col else i),
             "split": label_of[i]}
            for i in range(dataset.n_prepared)
        ]
        payload = dict(dataset.manifest())
        payload["assignments"] = assignments[:_MAX_ROWS]
        if len(assignments) > _MAX_ROWS:
            payload["assignments_truncated"] = len(assignments) - _MAX_ROWS
        if save_dir:
            payload["written"] = dataset.write(
                os.path.abspath(os.path.expanduser(save_dir)), make_figure=False
            )
        return json.dumps(payload, indent=2, default=_jsonable)

    @server.tool(title="Find duplicate molecules", annotations=READ_LOCAL)
    @_friendly_errors
    def find_duplicates(
        table: str, smiles_col: str, method: str = "canonical",
    ) -> str:
        """Find redundant compounds in a table: rows that are the same molecule.

        Groups rows of ``table`` by the ``smiles_col`` using ``method``
        ``"canonical"`` (RDKit canonical SMILES, so different spellings of one
        molecule collapse) or ``"exact"`` (raw string match). Returns the
        duplicate groups (each as the list of row indices and ids that share a
        key) and how many rows would be removed by keeping the first of each.
        """
        from .library import read_table
        from .prepare import canonical_smiles, dedup_keys

        tab = read_table(table)
        raw = tab.column(smiles_col)
        keys = canonical_smiles(raw) if method == "canonical" else [
            "" if k is None else str(k).strip() for k in raw
        ]
        id_col = tab.columns[0] if tab.columns else None
        groups: dict[str, list[int]] = {}
        for i, key in enumerate(keys):
            if key:
                groups.setdefault(key, []).append(i)
        duplicate_groups = [
            {"key": key,
             "rows": members,
             "ids": [tab.rows[i].get(id_col) if id_col else i for i in members]}
            for key, members in groups.items() if len(members) > 1
        ]
        _, n_removed = dedup_keys(raw, method)
        return json.dumps(
            {"n_rows": len(tab), "method": method,
             "n_duplicate_groups": len(duplicate_groups),
             "n_redundant_rows": n_removed,
             "groups": duplicate_groups[:_MAX_ROWS]},
            indent=2, default=_jsonable,
        )

    @server.tool(title="Summarise docking hits", annotations=WRITE_LOCAL)
    @_friendly_errors
    def dock_summary(
        source: str, score_field: Optional[str] = None, top: int = 10,
        higher_is_better: Optional[bool] = None, with_smiles: bool = True,
        best_pose_per_ligand: bool = True,
        save_dir: Optional[str] = None,
    ) -> str:
        """Rank docking poses from an output SDF and summarise the hits.

        ``source`` is a path to a docking-output ``.sdf`` (one record per pose, as
        written by AutoDock Vina, Gnina or Smina). ``score_field`` is the
        ``> <tag>`` data field holding the score; omit it to auto-detect a known
        field (e.g. ``minimizedAffinity``, ``CNNscore``). Direction is inferred
        from the field name unless ``higher_is_better`` is set. SMILES need RDKit
        and are blank without it. Returns the ranked rows (best first) with score,
        ligand efficiency and heavy-atom count; pass ``save_dir`` to also write
        ``dock_summary.csv``, ``top_hits.csv`` and ``score_distribution.png``.
        """
        from . import docking

        poses = docking.PoseStream(_dock_path(source))
        field = docking.resolve_score_field(poses, score_field)
        higher, assumed = (
            docking.higher_is_better(field) if higher_is_better is None
            else (higher_is_better, False)
        )
        result = docking.summarize(
            poses, field, higher_is_better_flag=higher,
            direction_assumed=assumed, with_smiles=with_smiles,
            best_pose_per_ligand=best_pose_per_ligand,
        )
        columns = ["rank", "pose_id", "name", "smiles", "score",
                   "ligand_efficiency", "n_heavy_atoms"]
        payload = {
            "score_field": field,
            "direction": "higher_is_better" if higher else "lower_is_better",
            "direction_assumed": result.direction_assumed,
            "n_poses": result.n_poses,
            "n_ranked": len(result.rows),
            "n_missing": result.n_missing,
            "with_smiles": result.with_smiles,
            "rows": result.rows[:_MAX_ROWS],
        }
        if len(result.rows) > _MAX_ROWS:
            payload["rows_truncated"] = len(result.rows) - _MAX_ROWS
        if save_dir:
            out = os.path.abspath(os.path.expanduser(save_dir))
            os.makedirs(out, exist_ok=True)
            summary_csv = os.path.join(out, "dock_summary.csv")
            top_csv = os.path.join(out, "top_hits.csv")
            docking.write_rows_csv(summary_csv, columns, result.rows)
            docking.write_rows_csv(top_csv, columns, result.rows[: max(0, top)])
            written = [summary_csv, top_csv]
            fig = os.path.join(out, "score_distribution.png")
            if docking.plot_score_distribution(result.scores, field, fig):
                written.append(fig)
            payload["written"] = written
        return json.dumps(payload, indent=2, default=_jsonable)

    @server.tool(title="Select diverse docking hits", annotations=WRITE_LOCAL)
    @_friendly_errors
    def dock_diverse(
        source: str, score_field: Optional[str] = None, top: int = 500,
        select: int = 50, threshold: float = 0.7,
        higher_is_better: Optional[bool] = None, save_dir: Optional[str] = None,
    ) -> str:
        """Pick a chemically diverse subset of the top docking hits.

        Ranks the poses in ``source`` (a docking ``.sdf``), keeps the best
        ``top``, clusters them by Tanimoto similarity (Morgan fingerprints, Butina
        at similarity ``threshold``) and returns the best-scoring representative of
        each cluster, up to ``select`` — so a shortlist is not many near-identical
        analogues. Needs RDKit. Reports when fewer clusters exist than requested.
        Pass ``save_dir`` to also write ``diverse_hits.sdf`` and
        ``diverse_hits.csv``.
        """
        from . import docking

        poses = docking.PoseStream(_dock_path(source))
        field = docking.resolve_score_field(poses, score_field)
        higher = (
            docking.higher_is_better(field)[0]
            if higher_is_better is None else higher_is_better
        )
        result = docking.select_diverse_hits(
            poses, field, higher_is_better_flag=higher,
            top=top, select=select, threshold=threshold,
        )
        columns = ["rank", "pose_id", "name", "smiles", "score",
                   "cluster_id", "cluster_size"]
        rows = [{k: rep[k] for k in columns} for rep in result.selected]
        payload = {
            "score_field": field,
            "n_pool": result.n_pool,
            "n_clusters": result.n_clusters,
            "requested": result.requested,
            "threshold": result.threshold,
            "capped_below_request": result.capped_below_request,
            "selected": rows,
        }
        if result.n_failed_fp:
            payload["n_failed_fp"] = result.n_failed_fp
        if save_dir:
            out = os.path.abspath(os.path.expanduser(save_dir))
            os.makedirs(out, exist_ok=True)
            csv_path = os.path.join(out, "diverse_hits.csv")
            sdf_path = os.path.join(out, "diverse_hits.sdf")
            docking.write_rows_csv(csv_path, columns, rows)
            docking.write_poses_sdf([rep["pose"] for rep in result.selected], sdf_path)
            payload["written"] = [csv_path, sdf_path]
        return json.dumps(payload, indent=2, default=_jsonable)

    @server.tool(title="Consensus-rank docking hits", annotations=WRITE_LOCAL)
    @_friendly_errors
    def dock_rank(
        sources: list[str], score_fields: Optional[list[str]] = None,
        key: str = "name", higher_is_better: Optional[list[str]] = None,
        lower_is_better: Optional[list[str]] = None, mw_max: Optional[float] = None,
        logp_max: Optional[float] = None, save_path: Optional[str] = None,
    ) -> str:
        """Consensus-rank hits across one or more scored docking SDFs.

        ``sources`` is a list of docking ``.sdf`` paths (e.g. a Vina and a Gnina
        scoring of the same library). Molecules are joined across files by ``key``
        (``"name"`` or ``"smiles"``; smiles needs RDKit), each score field is
        ranked by its own direction, and the consensus is the mean rank across
        fields. ``score_fields`` selects fields (known docking fields are
        auto-detected otherwise). ``higher_is_better``/``lower_is_better`` are
        lists of field names overriding the inferred direction; ``mw_max`` /
        ``logp_max`` drop hits outside a property window (need RDKit). The result
        reports which fields and directions were used; the consensus rank is a
        transparent triage heuristic, not a calibrated affinity.
        """
        from . import docking

        pose_sets = [
            (docking._stem(s), docking.PoseStream(_dock_path(s))) for s in sources
        ]
        result = docking.consensus_rank(
            pose_sets, score_fields=score_fields, key=key,
            higher=set(higher_is_better) if higher_is_better else None,
            lower=set(lower_is_better) if lower_is_better else None,
            mw_max=mw_max, logp_max=logp_max,
        )
        payload = {
            "key": result.key,
            "method": "consensus (mean rank across score fields)",
            "score_columns": result.score_columns,
            "directions": {
                col: ("higher_is_better" if hib else "lower_is_better")
                for col, hib in result.directions.items()
            },
            "assumed_direction": result.assumed,
            "n_dropped_filter": result.n_dropped_filter,
            "n_molecules": len(result.rows),
            "note": ("consensus rank is the mean rank across the listed fields, a "
                     "transparent triage heuristic, not a calibrated affinity"),
            "rows": result.rows[:_MAX_ROWS],
        }
        if len(result.rows) > _MAX_ROWS:
            payload["rows_truncated"] = len(result.rows) - _MAX_ROWS
        if save_path:
            path = os.path.abspath(os.path.expanduser(save_path))
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            docking.write_rows_csv(path, result.columns, result.rows)
            payload["written"] = [path]
        return json.dumps(payload, indent=2, default=_jsonable)

    @server.tool(title="Build docking HTML report", annotations=WRITE_LOCAL)
    @_friendly_errors
    def dock_report(
        source: str, save_dir: str, score_field: Optional[str] = None,
        top: int = 50, select: int = 20, threshold: float = 0.7,
        export_poses: int = 20, clusters: bool = True,
        higher_is_better: Optional[bool] = None,
        best_pose_per_ligand: bool = True,
    ) -> str:
        """Write a self-contained HTML triage report for a docking run.

        Builds ``dock_report.html`` in ``save_dir`` (a ranked hit table, an
        embedded score histogram and, when RDKit is available and ``clusters`` is
        true, a grid of diverse cluster representatives drawn in 2D) plus
        ``top_poses.sdf`` holding the best ``export_poses`` poses for loading into
        PyMOL, ChimeraX or Mol*. ``source`` is a docking ``.sdf``. Unlike the other
        docking tools this always writes files, so ``save_dir`` is required.
        Returns the written paths and a short summary.
        """
        from . import docking

        poses = docking.PoseStream(_dock_path(source))
        field = docking.resolve_score_field(poses, score_field)
        higher, assumed = (
            docking.higher_is_better(field) if higher_is_better is None
            else (higher_is_better, False)
        )
        summary = docking.summarize(
            poses, field, higher_is_better_flag=higher, direction_assumed=assumed,
            best_pose_per_ligand=best_pose_per_ligand,
        )
        diverse = None
        cluster_note = None
        if clusters:
            try:
                diverse = docking.select_diverse_hits(
                    poses, field, higher_is_better_flag=higher,
                    top=max(500, top), select=select, threshold=threshold,
                )
            except (ImportError, ValueError) as exc:
                cluster_note = f"clustering skipped: {exc}"
        out = os.path.abspath(os.path.expanduser(save_dir))
        os.makedirs(out, exist_ok=True)

        ranked_ids = [r["pose_id"] for r in summary.rows[: max(0, export_poses)]]
        top_poses = docking.collect_poses(poses, ranked_ids)
        poses_name = "top_poses.sdf"
        if top_poses:
            docking.write_poses_sdf(top_poses, os.path.join(out, poses_name))

        html = docking.render_html_report(
            summary, source_name=os.path.basename(source), n_poses=summary.n_poses,
            diverse=diverse, table_rows=top,
            poses_file=poses_name if top_poses else None,
        )
        report_path = os.path.join(out, "dock_report.html")
        with open(report_path, "w") as handle:
            handle.write(html)
        written = [report_path]
        if top_poses:
            written.append(os.path.join(out, poses_name))
        payload = {
            "score_field": field,
            "n_poses": summary.n_poses,
            "n_ranked": len(summary.rows),
            "n_clusters": diverse.n_clusters if diverse is not None else None,
            "written": written,
        }
        if diverse is not None and diverse.n_failed_fp:
            payload["n_failed_fp"] = diverse.n_failed_fp
        if cluster_note:
            payload["cluster_note"] = cluster_note
        return json.dumps(payload, indent=2, default=_jsonable)

    @server.tool(title="Render structure", annotations=WRITE_NET)
    @_friendly_errors
    def render_structure(source: str, color_by: str = "element", save_path: Optional[str] = None):
        """Render the structure in 3D.

        ``color_by`` is ``"element"``, ``"chain"``, ``"residue"`` or ``"ss"``
        (secondary structure). Pass ``save_path`` (e.g. ``"~/Desktop/view.png"``)
        to write the figure to a file and return its path; omit it to return the
        image inline. The format follows the ``save_path`` extension
        (``.png``/``.pdf``/``.svg``), defaulting to PNG.
        """
        import matplotlib

        matplotlib.use("Agg")
        mol = _load(source)
        ax = mol.plot(color_by=color_by, show=False)
        return _figure_result(ax.figure, save_path)

    @server.tool(title="Render contact map", annotations=WRITE_NET)
    @_friendly_errors
    def render_contact_map(
        source: str, cutoff: float = 8.0, level: str = "residue", method: str = "ca",
        save_path: Optional[str] = None,
    ):
        """Render a contact map as a heatmap.

        Same ``cutoff``/``level``/``method`` options as the ``contact_map`` tool.
        Pass ``save_path`` to write the figure to a file and return its path;
        omit it to return the image inline.
        """
        import matplotlib

        matplotlib.use("Agg")
        cmap = _load(source).contact_map(cutoff=cutoff, level=level, method=method)
        ax = cmap.plot(show=False)
        return _figure_result(ax.figure, save_path)

    @server.tool(title="Render distance matrix", annotations=WRITE_NET)
    @_friendly_errors
    def render_distance_matrix(source: str, save_path: Optional[str] = None):
        """Render the dense pairwise atom-distance matrix as a heatmap.

        Pass ``save_path`` to write the figure to a file and return its path;
        omit it to return the image inline.
        """
        import matplotlib

        matplotlib.use("Agg")
        ax = _load(source).plot_distance_matrix(show=False)
        return _figure_result(ax.figure, save_path)

    @server.tool(title="Render RMSD heatmap", annotations=WRITE_NET)
    @_friendly_errors
    def render_rmsd_heatmap(source: str, save_path: Optional[str] = None):
        """Render a multi-model ensemble's pairwise-RMSD matrix as a heatmap.

        ``source`` must be a multi-model (e.g. NMR) PDB. Pass ``save_path`` to
        write the figure to a file and return its path; omit it for inline.
        """
        import matplotlib

        matplotlib.use("Agg")
        from .ensemble import rmsd_matrix
        from .io import read_pdb_models
        from .plotting import plot_rmsd_heatmap

        models = read_pdb_models(source)
        if len(models) < 2:
            raise ValueError("render_rmsd_heatmap needs a multi-model file (e.g. an NMR PDB)")
        ax = plot_rmsd_heatmap(rmsd_matrix(models), show=False)
        return _figure_result(ax.figure, save_path)

    @server.tool(title="Render cross-correlation", annotations=WRITE_NET)
    @_friendly_errors
    def render_cross_correlation(
        source: str, selection: str = "ca", save_path: Optional[str] = None
    ):
        """Render a multi-model ensemble's dynamical cross-correlation (DCCM).

        ``source`` must be a multi-model (e.g. NMR) PDB. ``selection`` is
        ``"ca"`` for a residue-level map over alpha-carbons (default) or
        ``"all"`` for an all-atom map. Pass ``save_path`` to write the figure to
        a file and return its path; omit it for inline. The heatmap runs from -1
        (anticorrelated) through 0 to +1 (correlated motion).
        """
        import matplotlib

        matplotlib.use("Agg")
        from .ensemble import cross_correlation
        from .io import read_pdb_models
        from .plotting import plot_cross_correlation

        models = read_pdb_models(source)
        if len(models) < 2:
            raise ValueError(
                "render_cross_correlation needs a multi-model file (e.g. an NMR PDB)"
            )
        if selection == "ca":
            models = [m.alpha_carbons() for m in models]
        elif selection != "all":
            raise ValueError(f"selection must be 'ca' or 'all', got {selection!r}")
        ax = plot_cross_correlation(cross_correlation(models), show=False)
        return _figure_result(ax.figure, save_path)

    def _figure_result(figure, save_path: Optional[str]):
        """Save the figure to ``save_path`` (returning the path) or return it inline.

        When ``save_path`` is given the figure is written to disk and the absolute
        path is returned as text, so the user gets a real file to open or share.
        The image format follows the path extension (png/pdf/svg/jpg/tiff),
        defaulting to PNG. Otherwise the PNG is returned inline.
        """
        import matplotlib.pyplot as plt

        if save_path:
            path = os.path.abspath(os.path.expanduser(save_path))
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            fmt = ext if ext in {"png", "pdf", "svg", "jpg", "jpeg", "tif", "tiff"} else "png"
            figure.savefig(path, format=fmt, dpi=150, bbox_inches="tight")
            plt.close(figure)
            return f"Saved figure to {path}"

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(figure)
        return Image(data=buffer.getvalue(), format="png")

    return server


def main() -> None:
    """Console-script entry point: build the server and serve over stdio."""
    try:
        server = build_server()
    except ImportError as exc:
        raise SystemExit(
            "The MolScope MCP server needs the 'mcp' package. "
            "Install it with: pip install 'molscope[mcp]'"
        ) from exc
    server.run()


if __name__ == "__main__":  # pragma: no cover
    main()
