# Quickstart

Read a structure once, then choose one of the three main MolScope paths:
descriptors, graph ML, or coarse-grained beads.

```python
import molscope as ms

mol = ms.read("examples/data/1fqy.pdb")
print(mol.summary())
mol.plot()
```

## PDB to descriptors

```python
features = mol.descriptors()
X, names = ms.featurize_many(["a.pdb", "b.pdb", "c.xyz"], return_names=True)
```

Use this path for quick structure summaries, batch QC, and classical ML tables.

## PDB to graph/GNN

```python
g = mol.to_graph()
G = mol.to_networkx()
```

Use this path for atom/bond message passing, residue-contact graphs, or
framework exports such as PyTorch Geometric and DGL.

For a whole **dataset**, `build_dataset` reads, featurises, label-joins, and
splits a folder in one call, and `GraphDataset` carries it to a training loop:

```python
ds = ms.build_dataset("data/*.pdb", fmt="pyg", labels="labels.csv",
                      split=(0.8, 0.1, 0.1), cache_dir=".graph_cache")
scaler = ds.standardize_targets()                 # fit on train only
for batch in ds.loader("train", batch_size=32):   # batching PyG/DGL DataLoader
    ...
```

Start from RCSB accessions instead of files with `ms.fetch_dataset(ids, labels=...)`.

## PDB to coarse-grained beads

```python
cg = mol.coarse_grain("residue_com")
print(cg.mapping_report())
```

Use this path for reduced representations, mapping inspection, and bead-level
graph prototypes. MolScope does not generate production simulation topologies.

## Supporting moves

Transformations return new molecules:

```python
moved = mol.centered().rotate("z", 90).translate((1, 2, -1))
```

Read all models from an NMR PDB file:

```python
models = ms.read_pdb_models("examples/data/1aml.pdb")
matrix = ms.ensemble.rmsd_matrix(models[:5])
```
