# Protein analysis: structured coordinate data

MolScope treats proteins read from PDB/mmCIF as structured coordinate data:
coordinates plus atom names, residue names, residue ids, chain ids and
ATOM/HETATM records. Those fields make protein-specific questions possible.
None of the tools on this page need extra dependencies.

## Backbone atoms, residues and chains

Protein backbone atoms use the standard atom names `N`, `CA`, `C` and `O`.
Alpha carbons (`CA`) are a common residue-level proxy for contact maps, RMSD and
coarse structural comparison.

```python
import molscope as ms

mol = ms.read("examples/data/1fqy.pdb")

len(list(mol.residue_groups()))     # 226 residue groups
mol.chain_ids()                     # ['A']
mol.backbone()                      # N, CA, C, O atoms
mol.alpha_carbons()                 # one CA per residue for this protein
```

Residues are identified internally by chain, integer residue number, insertion
code, and residue name. The old integer `resids` array remains available, while
`mol.icodes`, `mol.residue_ids`, contact-map `residue_ids`, and binding-site
records keep insertion-code distinctions such as `A:SER100A` vs `A:THR100B`.

For structures with HETATM records, MolScope separates polymer atoms from
ligands, waters and ions:

```python
trypsin = ms.read("examples/data/3ptb.pdb")

trypsin.protein()                   # ATOM atoms only
trypsin.hetero_atoms()              # HETATM atoms: ligand, waters, calcium
trypsin.select(resname="HOH")       # waters
trypsin.ligands()                   # non-solvent, non-ion HETATM groups
```

The tutorial notebook
[`notebooks/protein_analysis_from_scratch.ipynb`](https://github.com/roshan2004/molscope/blob/main/notebooks/protein_analysis_from_scratch.ipynb)
walks this from raw files through contacts, ligands and secondary structure
using the bundled `1fqy`, `1aml` and `3ptb` examples.

## Contact maps

Residue contact maps summarize which residue pairs are close in 3D:

```python
cmap = mol.contact_map(cutoff=8.0, level="residue", method="ca", min_seq_sep=4)
cmap.matrix.shape                  # (226, 226)
cmap.n_contacts                    # non-local contact count
cmap.contact_order()               # local vs long-range contact metric
```

For NMR ensembles, contact frequency reports how often each pair is in contact:

```python
models = ms.read_pdb_models("examples/data/1aml.pdb")
freq = ms.ensemble_contact_frequency(models, cutoff=8.0)
freq.matrix                         # values in [0, 1]
```

## Secondary structure elements

`secondary_structure()` runs MolScope's built-in simplified DSSP-style
assignment and returns a [`SecondaryStructure`][]. It is designed for teaching
and prototyping: it follows the Kabsch-Sander hydrogen-bond idea and is
cross-checked against `mkdssp`, but it is not bit-identical to canonical
reference DSSP on every edge case.

Use it for lightweight structure inspection. Use a reference `mkdssp`
installation when production-grade secondary-structure labels are required.
On top of the per-residue codes, you can extract elements, reduce to three
states, and break the assignment down by chain.

```python
ss = ms.read("examples/data/1fqy.pdb").secondary_structure()

ss.string            # 8-state codes, e.g. '--HHHHH--EEEE--'
ss.simplified()      # 3-state: 'CCHHHHHCCEEEEC...' (H helix, E strand, C coil)

for seg in ss.segments():           # contiguous helices/strands, coil omitted
    print(seg)                      # e.g. SSSegment(H A:9-36, len=28)

ss.summary()                        # helix/strand/coil counts and fractions
ss.per_chain()                      # the same breakdown per chain id
```

## Backbone torsions (Ramachandran)

`backbone_torsions()` returns the phi/psi/omega dihedrals per residue in degrees,
with `NaN` at chain termini and breaks:

```python
tor = ms.read("examples/data/1fqy.pdb").backbone_torsions()
tor.phi, tor.psi, tor.omega         # (R,) arrays aligned with tor.resids/chains
```

Helical residues cluster near phi -63, psi -42; trans peptide bonds give omega
near 180.

For the standard Ramachandran scatter, `plot_ramachandran` draws phi vs psi on a
`[-180, 180]` grid, coloured by secondary-structure class by default, with
schematic alpha/beta region guides:

```python
mol = ms.read("examples/data/1fqy.pdb")
mol.plot_ramachandran()                       # color_by="ss" (default)
mol.plot_ramachandran(color_by="navy", regions=False)
```

The shaded regions are approximate teaching guides, not statistically-derived
density contours. Residues whose phi or psi is undefined (chain ends, breaks)
are skipped.

## Chain interfaces

For multi-chain structures, find the residues that form an interface between two
chains, or summarise inter-chain contacts across the whole assembly:

```python
mol = ms.fetch("1brs")                       # barnase-barstar complex

iface = mol.interface("A", "D", cutoff=5.0)  # -> Interface
iface.residues_a                             # interface residues on chain A
iface.residues_b                             # ...and on chain D
iface.n_atom_contacts                        # atom pairs across the interface

ccm = mol.chain_contacts(cutoff=5.0)         # -> ChainContactMatrix
ccm.count("A", "D")                          # inter-chain atom contacts
```

## Ligand-binding sites

HETATM groups (ligands, cofactors, ions, water) are tracked separately from
polymer atoms. `ligands()` lists the non-solvent groups; `binding_site()` finds
the protein residues around one.

```python
mol = ms.read("examples/data/3ptb.pdb")      # trypsin + benzamidine

mol.ligands()                                # [LigandResidue(A:BEN1, 9 atoms)]

site = mol.binding_site(cutoff=4.5)          # the single ligand is auto-detected
for res, d in zip(site.residues, site.min_distances):
    print(res, round(d, 2))                  # closest residues first
# A:GLY219 2.82
# A:ASP189 2.87   <- the benzamidine specificity residue
# A:SER190 3.04
# ...

site.n_atom_contacts                         # protein-ligand atom contacts
site.to_records()                            # table-ready residue rows
site.descriptors(mol, preset="pocket-basic") # pocket descriptors
site.plot(mol, show=False)                   # residue subset plus ligand
```

When several ligands are present, name one explicitly:

```python
mol.binding_site(ligand="BEN")               # by residue name
mol.binding_site(ligand=("A", 1))            # by (chain, resid)
mol.binding_site(ligand=("A", 100, "A"))     # by (chain, resid, insertion code)
```

Water and common ions are excluded from `ligands()` by default
(`exclude_water`, `exclude_ions`). The polymer/hetero split is also available as
selections:

```python
mol.protein()        # ATOM atoms only
mol.hetero_atoms()   # HETATM atoms only
mol.select(hetero=True)
```

To export the binding-site residue table without writing Python:

```bash
molscope binding-site examples/data/3ptb.pdb --out site.csv --cutoff 4.5
molscope binding-site examples/data/3ptb.pdb --out site.csv --descriptors-out pocket.csv
```

[`SecondaryStructure`]: ../api-reference.md
