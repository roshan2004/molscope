# Chemical Perception

MolScope keeps cheminformatics out of the core install. For aromaticity,
valence and sanitized charge features, install the optional RDKit backend:

```bash
pip install "molscope[chem]"
```

Then call:

```python
import molscope as ms

mol = ms.read("ligand.sdf")
chem = mol.chemical_features()

chem.formal_charges
chem.total_valences
chem.aromatic_atoms
chem.bond_orders
chem.aromatic_bonds
```

RDKit descriptors are also available through the same optional backend:

```python
rdkit_features = mol.rdkit_descriptors(names=["MolWt", "TPSA"])
features = mol.descriptors(include_rdkit=True, rdkit_descriptor_names=["MolWt"])
```

SDF/MOL V2000 formal charges and bond orders are preserved by the built-in
reader before RDKit is involved. If a structure only has coordinates, MolScope
can pass geometrically inferred single bonds to RDKit, but it does not infer
general bond orders from raw coordinates.

## Proteins: residue-template bonds

A bare protein PDB has no bond records, so geometric inference gives single bonds
only: aromaticity, backbone carbonyls, and side-chain double bonds are all lost,
and `chemical_features` comes back essentially empty (zero aromatic atoms).

For **standard residues** the chemistry is known, so you don't need to guess it.
Read with `bond_perception="template"` to have RDKit's residue-aware PDB reader
assign the bonds, Kekule bond orders, and formal charges from its built-in
residue templates (plus peptide bonds and disulfides):

```python
mol = ms.read("protein.pdb", bond_perception="template")   # needs the chem extra
chem = mol.chemical_features()
chem.aromatic_atoms.sum()    # now counts the Phe/Tyr/His/Trp rings
```

`read`, `read_pdb`, and `fetch` all accept the option (PDB input only). It needs
RDKit, and it only helps for standard residues: modified residues, non-standard
ligands, and exotic chemistry still fall back to best-effort perception. This is
**not** a force field, just template-based connectivity and bond orders; for
energies and parameters use a dedicated force-field tool. The default stays
`"geometric"`, so nothing changes unless you opt in.

### Protonation and formal charges

A crystallographic PDB models heavy atoms only, so RDKit reads every residue as
neutral and `chemical_features().formal_charges` sums to zero — faithful to the
file, but not the ionisation state at physiological pH. Add
`protonation="standard"` (with template bonds) for an idealised pH-7 assignment
of the standard ionisable side chains:

```python
mol = ms.read("protein.pdb", bond_perception="template", protonation="standard")
mol.chemical_features().formal_charges.sum()   # e.g. +6 for trypsin
```

The fixed assignment is aspartate/glutamate `-1`, lysine/arginine `+1`,
histidine neutral, and termini uncharged (see `molscope.chem.STANDARD_PROTONATION`).
It is a textbook model: it ignores local pKa shifts, buried or
metal-coordinating residues, and termini.

For an **environment-aware** prediction, use `protonation="pka"`, which runs
[PROPKA](https://github.com/jensengroup/propka) to predict each ionisable group's
pKa from the structure and then assigns the charge of the dominant state at a
chosen pH:

```python
mol = ms.read(
    "protein.pdb", bond_perception="template", protonation="pka", ph=7.4
)
mol.chemical_features().formal_charges.sum()   # reflects local pKa shifts
```

This accounts for the side chains, the N- and C-termini, and cysteine/tyrosine,
and it tracks pH: lowering `ph` protonates (net charge rises) while raising it
deprotonates (net charge falls). It needs the `propka` extra
(`pip install "molscope[propka]"`). The default `"none"` keeps the as-modelled
neutral state.

For **small molecules** the analogous knob lives on the dataset/library path:
`prepare_dataset(..., protonation="pka", ph=7.4)` and
`smiles_descriptors(..., protonation="pka")` set each SMILES to its dominant
ionisation state at the target pH (via
[Dimorphite-DL](https://github.com/durrantlab/dimorphite_dl),
`pip install "molscope[dimorphite]"`) before descriptors and fingerprints are
computed — so charge-sensitive features match the species present at that pH.
