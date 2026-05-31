# MolScope documentation

**Turn a molecular structure file into descriptors, contact maps, ML graphs, and
coarse-grained bead models, with a small, readable Python API.**

The problem MolScope is built for: *you have structure files and you want
ML-ready graphs and descriptors without installing a heavy stack or writing glue
code.* The core depends only on NumPy and Matplotlib; heavier backends (RDKit,
PyTorch Geometric, DGL, Gemmi) are opt-in extras you add only when a workflow
needs them. It is built for teaching, exploratory analysis, and ML-for-molecules
prototyping, not as a replacement for full simulation or cheminformatics stacks.

```python
import molscope as ms

mol = ms.read("examples/data/1fqy.pdb")   # or ms.fetch("1fqy")
print(mol.summary())                        # atoms, formula, chains, bounding box

desc  = mol.descriptors()                   # dict of structural descriptors
graph = mol.to_graph()                       # ML-ready graph, no extra deps
data  = mol.to_pyg_data()                    # PyTorch Geometric Data ([pyg])
```

## Core workflows

Each has a task-oriented tutorial:

| Workflow | Output |
| --- | --- |
| [PDB to descriptors](tutorials/pdb-to-descriptors.md) | Fixed-width structural and optional RDKit-backed feature tables for screening, QC, and classical ML. |
| [PDB to graph/GNN](tutorials/pdb-to-graph-gnn.md) | Atom/bond, residue-contact, and PyTorch Geometric-ready graph data, with positional encodings. |
| [PDB to coarse-grained beads](tutorials/pdb-to-coarse-grained-beads.md) | Residue, simplified Martini-style, custom, and virtual-site bead models for inspection and graph prototyping. |

## What supports those workflows

- Read `.pdb`, `.xyz`, `.cif` atom-site loops, and `.sdf` files (and stream large
  multi-model files frame by frame), preserving explicit SDF/PDB bonds.
- Fetch structures from the RCSB by ID, or build from SMILES with RDKit.
- Select atoms by element, chain, residue name, atom name, and residue id.
- Compute geometry, RMSD, contacts, contact maps, ensembles, and descriptors.
- Analyse proteins through backbone/alpha-carbon selections, ligands, waters,
  binding sites, contact maps, and simplified DSSP-style secondary structure.
- Expose optional RDKit-backed chemical features, descriptors, and bond-order
  inference; preserve SDF formal charges.
- Export atom/bond and residue-contact graphs to NetworkX, PyTorch Geometric,
  or DGL, with Laplacian and random-walk positional encodings.
- Prototype interpretable coarse-grained mappings (and export OpenMM residue
  templates) for teaching, inspection, and graph representations.
- Visualise molecules with Matplotlib or py3Dmol, from Python or the CLI.
- Document scientific validation against MDAnalysis, RDKit, `mkdssp`, and
  invariant checks, with explicit assumptions and tolerances.

## Install

```bash
pip install molscope                 # core: NumPy + Matplotlib only
pip install "molscope[chem,cif,pyg]" # add extras for the workflows you need
```

For development from the repository:

```bash
uv sync
uv run pytest
```

See the [installation guide](installation.md) for the full list of extras, and
the [quickstart](quickstart.md) to get going.
