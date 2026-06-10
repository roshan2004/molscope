<p align="center">
  <img src="https://raw.githubusercontent.com/roshan2004/molscope/main/docs/assets/logo.svg" alt="MolScope logo" width="180">
</p>

<h1 align="center">MolScope</h1>

[![CI](https://github.com/roshan2004/molscope/actions/workflows/ci.yml/badge.svg)](https://github.com/roshan2004/molscope/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/roshan2004/molscope/branch/main/graph/badge.svg)](https://codecov.io/gh/roshan2004/molscope)
[![Docs](https://readthedocs.org/projects/molscope/badge/?version=latest)](https://molscope.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/molscope.svg)](https://pypi.org/project/molscope/)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11%20%7C%203.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20433850.svg)](https://doi.org/10.5281/zenodo.20433850)

**Turn a molecular structure file into descriptors, contact maps, ML graphs, and
coarse-grained bead models, with a small, readable Python API.**

Reads `.xyz`, `.pdb`, `.cif`, and `.sdf` (or fetches from the RCSB by ID). The
core depends only on NumPy and Matplotlib; heavier backends (RDKit, PyTorch
Geometric, DGL, Gemmi) are opt-in extras. Built for teaching, exploratory
analysis, and ML-for-molecules prototyping, not as a replacement for full
simulation or cheminformatics stacks.

📖 **Full documentation: <https://molscope.readthedocs.io>**

| 3D structure (element) | Secondary structure (DSSP) | Residue contact map | Coarse-grained beads |
| --- | --- | --- | --- |
| ![Aquaporin-1 rendered as a 3D element-coloured molecular structure](https://raw.githubusercontent.com/roshan2004/molscope/main/docs/assets/readme/aquaporin-structure-v2.png) | ![Aquaporin-1 coloured by DSSP secondary structure: helices red, turns cyan, coil grey](https://raw.githubusercontent.com/roshan2004/molscope/main/docs/assets/readme/secondary-structure.png) | ![Residue-level contact map heatmap for Aquaporin-1](https://raw.githubusercontent.com/roshan2004/molscope/main/docs/assets/readme/residue-contact-map.png) | ![Coarse-grained bead model of Aquaporin-1](https://raw.githubusercontent.com/roshan2004/molscope/main/docs/assets/readme/coarse-grained-beads-v2.png) |

## Install

```bash
pip install molscope            # core: NumPy + Matplotlib only
```

Optional extras, added only when a workflow needs them:

| Extra | Adds |
| --- | --- |
| `fast` | scipy KD-tree for faster bond/contact search on large structures |
| `chem` | RDKit chemical perception and descriptors |
| `cif` | Gemmi mmCIF parsing and validation |
| `pyg` / `dgl` / `graph` / `gnn` | PyTorch Geometric / DGL / NetworkX graph export |
| `viz` | py3Dmol interactive viewer |
| `xlsx` | read/write `.xlsx` molecule tables |
| `gpu` | Torch dense distance backend |
| `mcp` | MCP server for AI assistants (Python >= 3.10) |

```bash
pip install "molscope[chem,cif,pyg]"   # combine as needed
```

For local development with [uv](https://docs.astral.sh/uv/): `uv sync` (creates
`.venv` and installs deps + dev tools from the lockfile), then `uv run pytest`.

## Quickstart

Given a `.pdb` (or `.xyz` / `.cif` / `.sdf`), here is what you can pull out:

```python
import molscope as ms

mol = ms.read("protein.pdb")        # or ms.fetch("1fqy") from the RCSB
print(mol.summary())                # atoms, formula, chains, bounding box

ca   = mol.select(chain="A").alpha_carbons()   # metadata selections
cmap = mol.contact_map(cutoff=8.0)             # residue contact map (NumPy)
desc = mol.descriptors()                       # dict of structural descriptors
graph = mol.to_graph()                         # ML-ready graph, no extra deps
data  = mol.to_pyg_data()                      # PyTorch Geometric Data ([pyg])
cg    = mol.coarse_grain("residue_com")        # one bead per residue
```

`Molecule` is immutable: `translate`, `centered`, and `rotate` each return a new
molecule, so transformations chain cleanly.

### From a folder of structures to a trained GNN

`build_dataset` collapses read → featurise → label-join → split into one call,
and `GraphDataset` carries it the last mile to a training loop:

```python
ds = ms.build_dataset(
    "data/*.pdb",                 # glob, list of paths/Molecules, or fetch_dataset(ids) from RCSB
    fmt="pyg",                    # "pyg" | "dgl" | "networkx" | "raw"
    labels="labels.csv",          # joined to each graph by file stem
    split=(0.8, 0.1, 0.1),
    cache_dir=".graph_cache",     # on-disk featurisation cache; reruns reuse it
)
scaler = ds.standardize_targets() # fit on train only; no val/test leakage
for batch in ds.loader("train", batch_size=32):   # batching PyG/DGL DataLoader
    ...                           # scaler.inverse_transform(pred) -> physical units
```

This graph-ML on-ramp adds no core dependency (`fmt="pyg"`/`"dgl"` need their
extras). See [Molecular graphs](docs/user-guide/molecular-graphs.md) and the
runnable [PDB to a trained GNN](docs/examples/pdb-to-pyg-ml.md) example.

## What you can do

| Capability | Guide |
| --- | --- |
| Read/write XYZ, PDB, mmCIF, SDF; fetch from RCSB; build from SMILES | [Reading files](docs/user-guide/reading-files.md) |
| Stream large multi-model files frame by frame | [Reading files](docs/user-guide/reading-files.md) |
| Select atoms by metadata (chain, residue, name, ...) | [Selections](docs/user-guide/selections.md) |
| Geometry, RMSD, distances, angles, torsions | [Geometry and measurements](docs/user-guide/geometry.md) |
| Contact maps and distance matrices | [Contact maps](docs/user-guide/contact-maps.md) |
| DSSP secondary structure, torsions, interfaces, binding sites | [Protein analysis](docs/user-guide/protein-analysis.md) |
| Native and RDKit-backed descriptors | [Structural descriptors](docs/user-guide/descriptors.md) |
| Chemical perception, protein template bonds, bond-order inference | [Chemical perception](docs/user-guide/chemical-perception.md) |
| Atom/bond and residue-contact graphs for ML (with positional encodings) | [Molecular graphs](docs/user-guide/molecular-graphs.md) |
| Assemble a split, labelled graph dataset (cache, loaders, target scaling, RCSB fetch) | [Molecular graphs](docs/user-guide/molecular-graphs.md#building-a-dataset-in-one-call) |
| Coarse-grained bead mappings (residue, Martini-style, custom) | [Coarse-graining](docs/user-guide/coarse-graining.md) |
| NMR ensembles and clustering | [Ensemble analysis](docs/user-guide/ensembles.md) |
| Plotting and py3Dmol viewing | [Plotting and viewing](docs/user-guide/plotting.md) |
| Diverse subset selection from a CSV/XLSX table | [Diverse selection](docs/user-guide/library-selection.md) |

Task-oriented tutorials: [PDB to descriptors](docs/tutorials/pdb-to-descriptors.md),
[PDB to graph/GNN](docs/tutorials/pdb-to-graph-gnn.md), and
[PDB to coarse-grained beads](docs/tutorials/pdb-to-coarse-grained-beads.md). A
runnable tour over the bundled samples lives in [`examples/tour.py`](examples/tour.py).

## Command line

| Command | Does |
| --- | --- |
| `molscope <file>` (view) | visualise a structure, save a PNG or GIF |
| `molscope report` | one-file structure report (QC, chains/ligands, descriptors, contact map, graph stats, optional CG preview) as HTML/Markdown |
| `molscope compare` | compare two static structures: aligned RMSD, per-residue deviations, contact-map delta, descriptor delta |
| `molscope qc` | lightweight structure-quality report (atoms, chains, ligands, metadata, elements, bonds, altLoc, CIF/PDB warnings) |
| `molscope presets` | list the available descriptor / graph / coarse-grain presets and what each one produces |
| `molscope analyze` | batch descriptor table to CSV |
| `molscope binding-site` | ligand binding-site contacts and pocket descriptors |
| `molscope export` | batch graph export to PyG / DGL / NetworkX |
| `molscope coarse-grain` | map a structure to CG beads and write a coordinate file (PDB CONECT bonds) |
| `molscope select` | diverse subset from a CSV/XLSX table |
| `molscope dock-summary` | rank docking poses from an SDF; summary + top-hit tables + score plot |
| `molscope dock-diverse` | diverse shortlist of top hits by Tanimoto clustering |
| `molscope dock-rank` | transparent consensus ranking across scored SDFs |
| `molscope dock-report` | self-contained HTML report + top poses for PyMOL/ChimeraX/Mol* |

```bash
molscope examples/data/1fqy.pdb --select atom_name=CA --color-by residue --save ca.png
molscope report examples/data/3ptb.pdb --out-dir report/ --coarse-grain
molscope compare apo.pdb holo.pdb --atoms ca --out compare.md
molscope qc examples/data/3ptb.pdb
molscope presets descriptors          # discover the --preset / node_features options
molscope analyze examples/data/*.pdb --out results.csv --preset native-3d --jobs 4
molscope export "data/*.cif" --to pyg --out-dir pyg_graphs/ --pe laplacian --jobs 8
molscope coarse-grain examples/data/1fqy.pdb --mapping martini --out cg.pdb
molscope select molecules.csv --smiles-col SMILES --compute-descriptors -n 100 --out picked.csv
molscope dock-summary vina_out.sdf --score-field minimizedAffinity --top 20
molscope dock-diverse vina_out.sdf --top 500 --select 50
```

## Use from an AI assistant (MCP)

MolScope ships an optional [Model Context Protocol](https://modelcontextprotocol.io)
server, so an MCP-capable assistant (Claude Code/Desktop, Codex CLI, Gemini CLI)
can drive its analyses in natural language. It exposes the public API as 28 tools
(structure analysis, graphs, plots, dataset prep, docking-hit triage) and adds no
new science.

```bash
pip install "molscope[mcp]"              # needs Python >= 3.10
claude mcp add molscope -- molscope-mcp  # Claude Code
codex mcp add molscope -- molscope-mcp   # Codex CLI
gemini mcp add molscope molscope-mcp     # Gemini CLI
```

For example: *"fetch trypsin (3ptb), find the benzamidine binding-site residues,
and render a contact map."* See [`docs/user-guide/mcp-server.md`](docs/user-guide/mcp-server.md)
for the full tool reference.

## Scientific validation

MolScope is explicit about which results are cross-checked against reference
tools and which are intentionally lightweight:

| Feature | Status |
| --- | --- |
| Geometry, RMSD, contact maps | Cross-checked vs MDAnalysis (near machine precision) |
| Bond perception, chemical features | Cross-checked vs RDKit |
| Secondary structure (simplified DSSP) | Cross-checked vs `mkdssp`: ~98 to 99% 3-state agreement across helical, mixed, and all-beta folds |
| Protein template bonds | Cross-checked vs known per-residue chemistry |
| Native descriptors, molecular graphs | Deterministic; not benchmarked against a curated library |
| Coarse-graining | Mapping and visualisation only; **not** a validated force-field model |
| Standard protonation | Idealised pH-7 textbook model (`"standard"`) |
| pKa-aware protonation | Environment-aware via PROPKA (proteins) / Dimorphite-DL (SMILES) at a chosen pH |

Methods, tolerances, and failure modes are in [`docs/validation.md`](docs/validation.md).
The CI **validation** job runs physical invariants plus these cross-checks on every push.

## Scope and design philosophy

MolScope is a lightweight core (NumPy and Matplotlib only) with a broad optional
surface. Those optional extras are deliberately of two kinds, and the distinction
is what keeps "lightweight" honest:

- **Interop / output targets** (`networkx`, `pyg`, `dgl`, `viz`, `xlsx`, `gpu`,
  `mcp`). These let MolScope emit to your framework or run faster. Speaking many
  formats *is* the bridge mission, so this surface is expected to grow.
- **Method backends** (`chem`/RDKit, `cif`/gemmi, `propka`, `dimorphite`,
  `validation`/MDAnalysis). These delegate a scientific computation to an external
  tool, and each one inherits that tool's scope, versioning, and correctness
  surface. This is the surface we keep deliberately small.

A new method backend earns inclusion only when MolScope adds real integration
value beyond what you could write in a line or two against the tool directly (for
example, mapping RDKit perception back onto MolScope's atom model and residue
templates), and when the wrapped tool is maintained. The core does something
useful on its own; everything heavier degrades gracefully when its extra is
absent. MolScope is not, and is not trying to become, a re-implementation of a
cheminformatics or simulation stack: where a dedicated tool is the right answer,
it integrates that tool rather than reinventing it, and says so.

## FAQ

**Which formats can it read?** `.xyz`, `.pdb`, `.cif`, and `.sdf`; fetch from the
RCSB with `ms.fetch("1fqy")`; or build from SMILES with `ms.read_smiles(...)`
(needs `[chem]`).

**Does it handle MD trajectories?** It works on static structures and multi-model
files (NMR ensembles, and `ms.stream(...)` to iterate large multi-model PDB/XYZ
frame by frame). It has no trajectory engine; for DCD/XTC and friends use
MDAnalysis or MDTraj.

**Is the coarse-graining a real force field?** No. It produces CG mappings and
bead graphs for inspection and ML prototyping. The OpenMM XML export describes
topology only and is not a validated Martini parameter set.

**Do I need RDKit or PyTorch?** No. The core runs on NumPy and Matplotlib; those
are opt-in extras you install only for the matching workflow.

**Will odd PDB files parse?** ATOM/HETATM lines are read by fixed columns (not
whitespace), so touching, large, or negative coordinate fields read correctly.
Alternate conformations default to the primary altLoc.

## Development and citation

```bash
uv run pytest                  # full test suite
uv run pytest tests/validation # validation suite only
uv run ruff check .            # lint
```

CI runs the suite and linting across Python 3.9 / 3.11 / 3.13, smoke-imports the
extras, and runs a separate validation job on every push and PR. Notable changes
per release are in [`CHANGELOG.md`](CHANGELOG.md).

Each release is archived on Zenodo with a citable DOI. The concept DOI
[10.5281/zenodo.20433850](https://doi.org/10.5281/zenodo.20433850) always resolves
to the latest version; citation metadata is in [`CITATION.cff`](CITATION.cff), so
GitHub's "Cite this repository" button produces BibTeX and APA entries.

## License

[MIT](LICENSE)
