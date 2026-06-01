# Reading Molecular Files

Use `ms.read()` to dispatch by file extension:

```python
import molscope as ms

mol = ms.read("structure.pdb")
mol = ms.read("trajectory.xyz")
mol = ms.read("small_molecule.sdf")
mol = ms.read("structure.cif")
```

Supported formats:

| Format | Notes |
| --- | --- |
| PDB | Fixed-column parser for `ATOM`/`HETATM`; preserves insertion codes and `CONECT` bonds. |
| XYZ | Single-frame and multi-frame XYZ files. |
| SDF/MOL | V2000 atom and bond block reader; preserves bond orders and formal charges. |
| CIF/mmCIF | Reader for standard `_atom_site` coordinate loops, including quoted values and `_atom_site.pdbx_PDB_ins_code`. |

For what each format stores, which metadata is reliable, and why PDB and mmCIF
differ, see [Coordinate formats compared](coordinate-formats.md).

PDB alternate conformations can be selected explicitly:

```python
mol = ms.read_pdb("structure.pdb", altloc="primary")
mol = ms.read_pdb("structure.pdb", altloc="highest_occupancy")
```

Supported policies are `primary`, `first`, `highest_occupancy`, and `all`.

Residue numbers remain available as the integer `mol.resids` array. PDB/mmCIF
insertion codes are available as `mol.icodes`, and full per-atom identities are
available as `mol.residue_ids`.

The built-in CIF reader handles standard atom-site coordinate loops with quoted
values and semicolon-delimited text fields. Install `molscope[cif]` to use the
optional Gemmi parser and validation helpers:

```python
mol = ms.read_cif("structure.cif", parser="gemmi")
report = ms.validate_cif("structure.cif")
report.raise_for_errors()
```

Dictionary-aware validation is available when you provide local dictionary
files:

```python
report = ms.validate_cif("structure.cif", dictionaries=["mmcif_pdbx_v50.dic"])
```

Download a structure from RCSB:

```python
mol = ms.fetch("1fqy")
```

## From a SMILES string

`ms.read_smiles()` builds a `Molecule` from a SMILES by generating one 3D
conformer with RDKit (needs the `chem` extra):

```python
mol = ms.read_smiles("CC(=O)O")                          # acetic acid
mol = ms.read_smiles("c1ccccc1", add_hs=False, seed=7)   # heavy atoms only, reproducible
```

Bonds, Kekule bond orders, and formal charges come from RDKit. The coordinates
are a **generated conformer, not an experimental or energy-minimised structure**:
ideal as input for descriptors and graph-ML (where topology matters), but treat
geometry-dependent results (contact maps, RMSD against experiment, precise
distances) with care. An invalid SMILES, or one RDKit cannot embed, raises
`ValueError`. For an experimental geometry, read a PDB/mmCIF/SDF instead.

## Writing

```python
ms.write_xyz(mol, "out.xyz")
ms.write_pdb(mol, "out.pdb")
ms.write_sdf(mol, "out.sdf")   # coords, bonds, Kekulé orders, formal charges, properties
ms.write_cif(mol, "out.cif")   # minimal _atom_site loop (coords + residue/chain metadata)
```

All four writers are dependency-free and round-trip with the matching readers.
Two scope limits worth knowing: SDF is an atom-and-bond format (no chain/residue
metadata) capped at 999 atoms/bonds by V2000, and `write_cif` emits a *coordinate*
mmCIF — an `_atom_site` loop only, with no symmetry, anisotropy, or bonds.
