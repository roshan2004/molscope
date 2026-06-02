# Use MolScope from an AI assistant (MCP)

MolScope ships an optional [Model Context Protocol](https://modelcontextprotocol.io)
(MCP) server. MCP is an open standard that lets an AI assistant call external
tools, so with this server an assistant such as Claude Code or Claude Desktop can
drive MolScope's analyses in natural language.

The server is a thin, faithful adapter over the public `molscope` API. It adds no
new science: every tool maps onto a function documented elsewhere in this user
guide. What it gives you is a conversational front end, for example:

> "Fetch trypsin (3ptb), find the benzamidine binding-site residues, and render
> a contact map."

The assistant turns that into a `binding_site` call followed by a
`render_contact_map` call and shows you the residues and the figure.

## Install

The reference MCP SDK needs Python 3.10 or newer, so the server is gated behind
an optional extra:

```bash
pip install "molscope[mcp]"
```

On Python 3.9 the extra installs nothing and `molscope-mcp` exits with a clear
hint, since the SDK is unavailable there.

## Register with a client

The server speaks MCP over stdio, which is how local clients launch it. The
console script is `molscope-mcp` (equivalently `python -m molscope.mcp_server`).

### Claude Code

```bash
claude mcp add molscope -- molscope-mcp
```

### Claude Desktop

Add an entry to the app's MCP server configuration:

```json
{
  "mcpServers": {
    "molscope": {
      "command": "molscope-mcp"
    }
  }
}
```

### Codex CLI

Either run:

```bash
codex mcp add molscope -- molscope-mcp
```

or add a `[mcp_servers.molscope]` table to `~/.codex/config.toml`:

```toml
[mcp_servers.molscope]
command = "molscope-mcp"
args = []
```

### Gemini CLI

Either run:

```bash
gemini mcp add molscope molscope-mcp
```

or add an `mcpServers` entry to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "molscope": {
      "command": "molscope-mcp"
    }
  }
}
```

Point `command` at the `molscope-mcp` executable from the environment where you
installed `molscope[mcp]` (use its absolute path if the client does not share
your shell's `PATH`). The same server works for any MCP client because it speaks
MCP over stdio; only the registration syntax differs between clients.

## Tools

Every tool takes a `source` that is one of: a path to a local coordinate file
(`.pdb`, `.cif`, `.xyz`, `.sdf`, optionally gzipped); a 4-character RCSB PDB id
that is fetched and cached; or a SMILES string written as `smiles:<SMILES>`
(for example `smiles:CCO`), which builds a molecule from one RDKit-generated 3D
conformer (needs the `chem` extra). Because the coordinates of a SMILES input
are *generated* rather than experimental, SMILES suits topology-based tools
(descriptors, graphs) better than geometry-dependent ones (precise distances,
RMSD against experiment).

### Structure and geometry

| Tool | Arguments | Returns |
| --- | --- | --- |
| `summarize_structure` | `source` | One-line summary: atoms, formula, chains, size. |
| `geometry` | `source` | Centre of mass, radius of gyration, bounding box, principal moments. |
| `sasa` | `source`, `probe_radius`, `n_points`, `level` | Approximate solvent-accessible surface area; total plus most-exposed residues. |
| `measure` | `source`, `atoms` | Distance (2 indices), angle (3), or dihedral (4). |

### Comparison

| Tool | Arguments | Returns |
| --- | --- | --- |
| `rmsd` | `source_a`, `source_b`, `align` | Kabsch RMSD between two structures. |
| `ensemble_summary` | `source` (multi-model) | Model count, mean/max pairwise RMSD, RMSF, cluster count. |

### Descriptors and chemistry

| Tool | Arguments | Returns |
| --- | --- | --- |
| `compute_descriptors` | `sources` (list), `preset` | Descriptor table, one row per structure. The batch tool. |
| `chemical_features` | `source`, `bond_perception`, `protonation`, `ph` | RDKit formal charges, aromatic atom/bond counts (`chem` extra). Defaults to `"template"` bonds + `"standard"` pH-7 protonation, so protein PDBs get correct bond orders, aromatic rings, and a meaningful net charge. `protonation="pka"` predicts charges from the structure with PROPKA at `ph` (`propka` extra). |
| `prepare_structure` | `source`, `protonation`, `ph` | One-shot "is this ML-ready?" QC: non-standard residues, ligand/water inventory, residue-numbering gaps, backbone chain breaks, missing/truncated residues, hydrogens, alternate conformations / occupancies, and net charge. Returns an `ml_ready` verdict with blockers and warnings. Topology needs no extra; net charge uses the `chem`/`propka` backends and degrades gracefully. |
| `molecular_graph` | `source`, `preset`, `include_chemical_features`, `knn`, `radius`, `min_seq_sep` | Node/edge counts and feature names for the ML graph. |

### Protein analysis

| Tool | Arguments | Returns |
| --- | --- | --- |
| `secondary_structure` | `source` | Per-residue DSSP codes and helix/strand/coil composition. |
| `backbone_torsions` | `source` | Per-residue phi/psi/omega (Ramachandran), `null` where undefined. |
| `contact_map` | `source`, `cutoff`, `level`, `method`, `min_seq_sep` | Contact count, contact order, labelled pairs. |
| `binding_site` | `source`, `ligand`, `cutoff` | Binding-site residues ordered closest-first. |
| `list_ligands` | `source`, `exclude_water`, `exclude_ions` | HETATM groups present (run before `binding_site`). |
| `chain_interfaces` | `source`, `chain_a`, `chain_b`, `cutoff` | Interface residues for a chain pair, or the all-pairs chain contact matrix. |

### Coarse-graining, library prep, files

| Tool | Arguments | Returns |
| --- | --- | --- |
| `coarse_grain` | `source`, `mapping` | Bead-assignment statistics. |
| `select_diverse` | `table`, `n`, `descriptor_cols` / `smiles_col` + `compute_descriptors` | Diverse subset of a CSV/XLSX molecule table (MaxMin). |
| `validate_cif` | `source` | mmCIF validation report (`cif` extra for full checks). |

### Docking-hit triage

These tools read a docking-output `.sdf` (one record per pose, as written by
AutoDock Vina, Gnina or Smina). Here `source` is a literal SDF path, not a PDB id
or SMILES. They return JSON; pass `save_dir` / `save_path` to also write files.
See [Docking-hit triage](docking-triage.md) for the workflow.

| Tool | Arguments | Returns |
| --- | --- | --- |
| `dock_summary` | `source`, `score_field`, `top`, `higher_is_better`, `save_dir` | Ranked hits with score, ligand efficiency, SMILES; optional CSVs + histogram. |
| `dock_diverse` | `source`, `score_field`, `top`, `select`, `threshold`, `save_dir` | Diverse cluster representatives (Tanimoto/Butina); optional SDF + CSV. Needs RDKit. |
| `dock_rank` | `sources` (list), `score_fields`, `key`, `mw_max`, `logp_max`, `save_path` | Mean-rank consensus across scored SDFs, with the fields and directions used. |
| `dock_report` | `source`, `save_dir`, `top`, `select`, `export_poses` | Writes a self-contained `dock_report.html` + `top_poses.sdf`. |

### Plots

| Tool | Arguments | Returns |
| --- | --- | --- |
| `render_structure` | `source`, `color_by`, `save_path` | 3D scatter view. |
| `render_contact_map` | `source`, `cutoff`, `level`, `method`, `save_path` | Contact-map heatmap. |
| `render_distance_matrix` | `source`, `save_path` | Dense pairwise distance heatmap. |
| `render_rmsd_heatmap` | `source` (multi-model), `save_path` | Ensemble pairwise-RMSD heatmap. |
| `render_cross_correlation` | `source` (multi-model), `selection`, `save_path` | Dynamical cross-correlation (DCCM) heatmap. |

Every plot tool takes an optional `save_path`. **Pass it to get a file you can
open or share** (e.g. *"render the contact map for 3ptb and save it to
~/Desktop/3ptb.png"*): the figure is written to disk and the tool returns the
absolute path. The format follows the extension (`.png`, `.pdf`, `.svg`, ...),
defaulting to PNG. Omit `save_path` to receive the image inline instead, which
suits clients that render MCP image content but leaves no file behind.

Most tools take a `source` that is a local coordinate-file path or a 4-character
RCSB PDB id; `select_diverse` instead takes a `table` (CSV/XLSX) path. Data tools
return JSON text so the model can read the values directly; the render tools
return PNG images. Large per-residue or per-pair lists are truncated with a
`*_truncated` flag so a big structure cannot flood the conversation. `NaN`/`inf`
values (e.g. undefined torsion angles) are emitted as JSON `null`.

## Scope

The server intentionally wraps only existing library functions. Most tools are
read-only; a few write files **only when you pass an output path** (`save_path`
on the plot tools, `save_dir` on `prepare_dataset` and the docking tools, which
`dock_report` always uses). It never mutates structures or adds capabilities
beyond the library, and it inherits every limitation documented in
[Limitations by workflow](../limitations.md). For scripted or batch use, prefer
the Python API or the `molscope` command-line interface; the MCP server is for
interactive, assistant-driven exploration.
