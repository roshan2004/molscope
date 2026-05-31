# Coordinate formats compared: XYZ, PDB, mmCIF, SDF

MolScope reads four common molecular coordinate formats. They look
interchangeable (each one lists atoms and positions), but they differ sharply in
what metadata they carry and how reliably they carry it. Choosing the right
format, and knowing what you lose when you convert between them, avoids silent
data loss.

## What each format stores

| Capability | XYZ | PDB | mmCIF | SDF / MOL |
| --- | :---: | :---: | :---: | :---: |
| 3D coordinates | yes | yes | yes | yes |
| Element symbols | optional | yes | yes | yes |
| Atom names | no | yes | yes | no |
| Residues and residue ids | no | yes | yes | no |
| Chains | no | yes (1 char) | yes | no |
| Explicit bonds | no | `CONECT` only | optional | yes |
| Bond orders | no | no | optional | yes |
| Formal charges | no | rarely | optional | yes |
| Multiple models / frames | yes (frames) | yes (`MODEL`) | yes | yes (records) |
| ATOM vs HETATM distinction | no | yes | yes (`group_PDB`) | no |

"Optional" means the format can express it but many files omit it; MolScope reads
it when present.

## XYZ: coordinates and not much else

XYZ is the lowest common denominator: an atom count, a comment line, then
`element x y z` per atom. MolScope also accepts bare `x y z` dumps (with `#`
comments) and multi-frame trajectories via [`read_xyz_frames`](reading-files.md).

- **Reliable:** coordinates, and element symbols when the first column is present.
- **Missing:** atom names, residues, chains, bonds, charges. There is no concept
  of topology, so `mol.residue_groups()` or `secondary_structure()` will not work
  on an XYZ structure.

Reach for XYZ for quantum-chemistry inputs/outputs, small molecules, and quick
geometry dumps where connectivity does not matter.

## PDB: convenient but messy

The Protein Data Bank format is the workhorse of structural biology, and almost
everything reads it. Its problems come from its age: it is a **fixed-column**
format frozen around 80-character punched-card records.

- Fields live in **fixed columns**, not whitespace-separated tokens, so MolScope
  slices columns (e.g. coordinates from columns 31-54). Whitespace splitting
  silently mis-reads real files.
- Hard size limits leak into the science: **4 characters** for an atom name,
  **1 character** for a chain id, **4 digits** for a residue number, **5** for an
  atom serial. Large assemblies overflow these and need hacks or mmCIF.
- Connectivity is mostly **implicit**: bonds appear only in optional `CONECT`
  records (typically just for ligands/HETATM), so MolScope infers most bonds
  from geometry.
- **Alternate conformations** (altLoc) and insertion codes complicate "one
  residue, one atom" assumptions; `read_pdb(..., altloc=...)` selects a policy.

What MolScope reads from PDB: coordinates, element, atom name, residue
name/id, chain, the ATOM/HETATM flag, multiple `MODEL` records (NMR ensembles via
[`read_pdb_models`](reading-files.md)), and `CONECT` bonds.

## mmCIF: why it exists

mmCIF (macromolecular CIF) is the PDB's modern successor and the archival format
of the wwPDB. It exists specifically to fix PDB's structural limits.

- It is a **key/value and loop** format, not fixed columns, so there are no
  4-character or 5-digit ceilings. Large ribosomes and capsids that cannot fit in
  PDB are expressed cleanly.
- It is **self-describing and dictionary-based**: every column is a named data
  item (`_atom_site.Cartn_x`, `_atom_site.label_comp_id`, ...), validated against
  a published dictionary. New data items can be added without breaking parsers.
- It is **extensible**: experimental metadata, assemblies, and chemistry live in
  the same file under documented names.

MolScope ships a lightweight built-in `_atom_site` reader and an optional
[Gemmi](https://gemmi.readthedocs.io/) backend (`pip install "molscope[cif]"`)
for robust parsing and validation. Prefer mmCIF over PDB for anything large or
when you care about archival correctness.

## SDF / MOL: small molecules with real chemistry

SDF (and the single-molecule MOL it wraps) is the cheminformatics format. Its
V2000 connection table stores the chemistry that PDB and XYZ lack.

- **Reliable:** coordinates, elements, **explicit bonds with bond orders**, and
  **formal charges** (both the per-atom charge codes and `M  CHG` lines).
- **Missing:** residues, chains, and biological context. SDF describes a
  molecule, not a macromolecular assembly.
- MolScope reads the **V2000** format. The newer **V3000** connection table is a
  different layout and is rejected with a clear error rather than mis-read;
  convert V3000 to V2000 with RDKit or OpenBabel first.

Use SDF for ligands, drug-like molecules, and any workflow where bonds, bond
orders, and charges must be exact.

### Multi-pose SDF and docking output

Docking tools (AutoDock Vina, Gnina, Smina) write one SDF record per pose, with
the score in a `> <tag>` data field. [`read_sdf`](../api-reference.md) reads the
first record; [`read_sdf_frames`](../api-reference.md) reads every pose as a list
of molecules, keeping each pose's 3D coordinates and exposing its data fields via
`Molecule.properties`:

```python
import molscope as ms

poses = ms.read_sdf_frames("vina_out.sdf")        # core install, no extras
best = min(poses, key=lambda m: float(m.properties["minimizedAffinity"]))
print(best.name, best.properties["minimizedAffinity"])
```

Because the 3D pose is preserved (unlike an RDKit round-trip through SMILES),
each pose flows straight into MolScope's descriptor, contact-map, and diversity
tools, so a hit list can be ranked, de-duplicated, and triaged in place. Common
score tags are `minimizedAffinity` (Vina/Smina) and `CNNaffinity` / `CNNscore`
(Gnina); the values are kept as raw strings, so cast to `float` before sorting.

## Choosing a format

- Quantum chemistry, tiny molecules, geometry only: **XYZ**.
- Proteins and nucleic acids, broad tool compatibility: **PDB** (small/medium) or
  **mmCIF** (large or archival).
- Small molecules where chemistry matters: **SDF**.

Remember that converting *down* loses metadata permanently: PDB to XYZ drops
residues and chains; anything to XYZ drops bonds. MolScope's
[`Molecule`](../api-reference.md) keeps whatever the source provided, and
writers emit only what the target format supports.

## Handling malformed files

MolScope's readers report the file, format, and line when input is malformed,
instead of failing with a raw `ValueError` from deep inside a parser:

```text
structure.pdb: invalid PDB file (line 42): could not read coordinate columns 31-54 from record: 'ATOM ...'
frame.xyz: invalid XYZ file (line 1): header declares 30 atoms but 28 were found
model.cif: invalid mmCIF file: _atom_site loop is missing coordinate column(s) ['Cartn_x']; found columns [...]
ligand.sdf: invalid SDF file (line 4): V3000 connection tables are not supported (only V2000); convert the file with RDKit or OpenBabel first
```

See [Reading molecular files](reading-files.md) for the reader/writer API.
