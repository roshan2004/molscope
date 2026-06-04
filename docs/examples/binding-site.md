# Ligand binding site

Detect a bound ligand and report the protein residues that surround it, using
the bundled trypsin-benzamidine complex (3PTB).

```python
import molscope as ms

mol = ms.read("examples/data/3ptb.pdb")

# Water and ions are filtered out; only the real ligand remains.
print(mol.ligands())                 # [LigandResidue(A:BEN1, 9 atoms)]

site = mol.binding_site(cutoff=4.5)  # single ligand auto-detected
print(site)                          # BindingSite(A:BEN1: 13 residues < 4.5 A)

for res, dist in zip(site.residues, site.min_distances):
    print(f"{res!s:<10} {dist:.2f} A")
# A:GLY219   2.82
# A:ASP189   2.87   <- benzamidine specificity residue
# A:SER190   3.04
# A:GLY226   3.37
# A:SER195   3.65   <- catalytic serine
```

For quick figures or reports, convert the site to table-friendly residue
records and extract descriptors for only the site residues:

```python
site.to_records()[0]
# {'residue_id': 'A:GLY219', 'chain': 'A', 'resid': 219,
#  'insertion_code': '', 'resname': 'GLY',
#  'min_distance': 2.815..., 'n_atom_contacts': 5}

site.descriptors(mol, preset="pocket-basic")
site.plot(mol, show=False)          # pocket residues plus ligand
```

## Describe the pocket for an LLM

LLMs and RAG pipelines struggle with raw 3D coordinates. `describe_environment`
turns the pocket into a chemistry-aware paragraph you can drop straight into a
prompt: it reports the hydrophobic wall, aromatic residues, hydrogen bonds, and
salt-bridge / electrostatic contacts from pure-geometry heuristics (no force
field).

```python
pocket = mol.select_pocket(ligand="BEN", cutoff=4.5)
print(pocket.describe_environment())
# The binding pocket around ligand BEN (chain A) is lined by 13 residues within
# 4.5 A of the ligand. A hydrophobic pocket wall appears to be formed by VAL213,
# CYS191 and TRP215. Aromatic ring from TRP215 may engage in pi-stacking or
# cation-pi interactions with the ligand. Likely hydrogen bonds are suggested
# between the ligand N1 and O of GLY219 (2.8 A) and the ligand N2 and OG of
# SER190 (3.0 A). A possible salt bridge / electrostatic contact is suggested
# between the ligand N2 and the carboxylate of ASP189 (2.9 A). (Contacts are
# inferred from heavy-atom distances only ... confirm with a profiler such as
# PLIP or ProLIF.)
```

The interactions are **distance-only heuristics**: there is no donor/acceptor
typing, hydrogen-bond angle criterion, or protonation-state model, so the prose
is deliberately phrased as candidates ("likely", "possible"). Treat the output
as a first-pass scaffold and confirm with a dedicated interaction profiler such
as [PLIP](https://github.com/pharmai/plip) or
[ProLIF](https://github.com/chemosim-lab/ProLIF) for rigorous analysis.

For programmatic use, `pocket.environment()` returns a `PocketEnvironment` whose
`to_dict()` gives the structured findings (hydrophobic residues, aromatic
residues, hydrogen bonds, salt bridges) with atom names and distances. The same
output is available over the MCP server as the `describe_environment` tool, which
returns both the `prompt` text and the structured `features`.

The same residue table is available from the command line:

```bash
molscope binding-site examples/data/3ptb.pdb --out site.csv --cutoff 4.5
```

Add `--descriptors-out pocket.csv` to also write the one-row
`pocket-basic` descriptor table.

When a structure has several ligands, select one by residue name or location:

```python
mol.binding_site(ligand="BEN")
mol.binding_site(ligand=("A", 1))
mol.binding_site(ligand=("A", 100, "A"))  # with insertion code
```

The CLI accepts the same choices as `--ligand BEN`, `--ligand A:1`, or
`--ligand A:100:A`.

A runnable version lives in
[`examples/binding_site.py`](https://github.com/roshan2004/molscope/blob/main/examples/binding_site.py).
See the full guide:
[Protein analysis](../user-guide/protein-analysis.md).
