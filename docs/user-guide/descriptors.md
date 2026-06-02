# Structural Descriptors

`mol.descriptors()` returns a fixed-size descriptor dictionary for quick ML
feature tables:

```python
features = mol.descriptors()
features["radius_of_gyration"]
features["principal_moments"]
features["distance_histogram"]
```

Batch featurization:

```python
X, names = ms.featurize_many(
    ["a.pdb", "b.pdb", "c.xyz"],
    return_names=True,
)
```

Included features:

- atom and residue counts
- element counts
- molecular mass
- centroid and center of mass
- radius of gyration
- bounding-box dimensions and volume
- inertia tensor
- principal moments and axes
- shape anisotropy
- asphericity, acylindricity, and relative shape anisotropy (κ²)
- compactness
- distance histogram
- bond length summary statistics
- atom and residue contact summaries
- SASA summary statistics (total, mean, std, max)
- polar-contact count and salt-bridge count

Full contact matrices remain available through `mol.contact_map(...)`.
Distance histograms and atom contact counts are computed in coordinate blocks
instead of a full pairwise distance array:

```python
features = mol.descriptors(distance_chunk_size=2048)
```

Stable presets are available when you need reproducible feature columns:

```python
features = mol.descriptors(preset="native-basic")
X, names = ms.featurize_many(paths, preset="native-3d", return_names=True)
names = ms.descriptor_feature_names("native-3d")
```

Preset options:

- `native-basic`: counts, mass, size, compactness, bond summaries, and contact summaries.
- `native-3d`: `native-basic` plus centres, inertia, principal axes/moments, the
  gyration-tensor shape descriptors (asphericity, acylindricity, relative shape
  anisotropy κ²), SASA summary statistics, polar-contact and salt-bridge counts,
  and distance histograms.
- `rdkit-basic`: `native-basic` plus a stable subset of RDKit scalar descriptors.

The gyration-tensor shape descriptors in `native-3d` come from the eigenvalues
`λ₁ ≤ λ₂ ≤ λ₃` of the gyration tensor (recovered from the mass-weighted inertia
moments): asphericity `b = λ₃ − ½(λ₁+λ₂)` grows as a structure elongates,
acylindricity `c = λ₂ − λ₁` is zero for any axially symmetric shape, and the
relative shape anisotropy `κ² = (b² + ¾c²)/R_g⁴` runs from 0 (a sphere or
higher-symmetry arrangement) to 1 (a perfectly linear one). They are distinct
from the legacy `shape_anisotropy` column, which applies a similar formula to
the inertia moments directly.

`native-3d` also includes surface and interaction summaries: `sasa_total`,
`sasa_mean`, `sasa_std`, `sasa_max` from the Shrake-Rupley SASA (computed at a
coarser `sasa_n_points` than `mol.sasa()` for batch speed; tune it via the
`sasa_n_points` argument), a `polar_contact_count` (N/O atom pairs 2.5-3.5 Å
apart in different residues, a coarse geometric proxy for polar contacts rather
than a validated hydrogen-bond count), and a `salt_bridge_count` (basic
side-chain N within 4 Å of an acidic side-chain O, counting unique residue
pairs). These need coordinates and are computed only for `native-3d` (and the
unfiltered default), so `native-basic`/`rdkit-basic` stay fast.

Ligand binding sites have their own fixed-size preset because they need a
ligand context:

```python
mol = ms.read("examples/data/3ptb.pdb")
site = mol.binding_site(cutoff=4.5)
pocket = site.descriptors(mol, preset="pocket-basic")
names = ms.pocket_descriptor_feature_names("pocket-basic")
```

`pocket-basic` includes pocket atom and residue counts, amino-acid composition,
protein-ligand contact counts, radius of gyration, bounding-box dimensions, and
ligand-distance summaries.

## Solvent-accessible surface area (SASA)

`mol.sasa()` returns an approximate solvent-accessible surface area in Å² using
a vectorised Shrake-Rupley sphere — a fast, pure-NumPy descriptor of solvent
exposure with no C extensions or external SASA libraries:

```python
mol = ms.read("examples/data/1ubq.pdb")
per_atom = mol.sasa()                       # (n_atoms,) array, Å²
per_res = mol.sasa(level="residue")         # (n_residues,) summed per residue
total = mol.sasa().sum()                    # whole-structure total
```

Each atom's expanded sphere (its van der Waals radius plus a `probe_radius`
water probe, 1.4 Å by default) is sampled with `n_points` quasi-uniform points;
a point is accessible when it lies outside every neighbouring atom's expanded
sphere. Accuracy improves with `n_points` (default 192, within a few percent of
an exact analytical surface) at the cost of speed. Residue-level values follow
`mol.residue_groups()` order.

This is an approximation aimed at the descriptors workflow, not a replacement
for an exact analytical surface; it is not folded into the fixed `descriptors()`
presets, so those feature columns stay stable.

## RDKit descriptors

Install the optional chemical backend to access RDKit's scalar descriptor set:

```bash
pip install "molscope[chem]"
```

Use RDKit descriptors directly:

```python
rdkit_features = mol.rdkit_descriptors(names=["MolWt", "TPSA", "NumHDonors"])
```

Or merge selected RDKit descriptors into the standard MolScope descriptor
dictionary:

```python
features = mol.descriptors(
    include_rdkit=True,
    rdkit_descriptor_names=["MolWt", "TPSA", "NumHDonors"],
)
```

When `rdkit_descriptor_names` is omitted, all scalar RDKit descriptors available
in the installed RDKit version are included with an `rdkit_` prefix.
