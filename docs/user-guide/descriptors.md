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
- cross-sectional area summary (max, mean, min, std along the long axis)
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
  anisotropy κ²), the cross-sectional area summary, SASA summary statistics,
  polar-contact and salt-bridge counts, and distance histograms.
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

## Cross-sectional area profile

`native-3d` also records how wide the structure is at each point along its
length. `mol.cross_section_profile(...)` slices the structure into thin bands
perpendicular to an axis and measures each band's area, returning a
`CrossSectionProfile` (the full `positions`/`areas` curve plus `max`, `mean`,
`min`, `std`, and `length`). The descriptor table keeps the four reduced scalars
`cross_section_max`, `cross_section_mean`, `cross_section_min` (over occupied
slices, so a tapering terminus does not force it to zero), and
`cross_section_std`.

```python
profile = mol.cross_section_profile(axis="principal", thickness=1.0)
ms.plot_cross_section(profile)          # area vs. position along the axis
profile.summary()                       # the four reduced scalars
```

By default the slice axis is the **long principal axis** (the axis of smallest
inertia), so the cross-sections are taken perpendicular to the structure's
length and the profile is rotation-invariant. Pass `axis="x"|"y"|"z"` or a
3-vector to slice along a fixed direction — use `axis="z"` for a membrane protein
pre-oriented with its normal along z.

Two area methods are available:

- `method="hull"` (default, pure NumPy): the convex-hull area of the atoms in
  each slice — the *outer* cross-section, defined for any structure with no extra
  dependency.
- `method="voronoi"` (needs SciPy): the sum of the protein atoms' 2-D Voronoi
  cells, where surrounding `hetero` atoms (or an explicit `environment=` molecule)
  bound the protein's outer cells. This measures the area the protein *occupies*
  against its environment, and only differs from `hull` when such environment
  atoms are present; with none it raises and points you back to `hull`.

The method follows the per-slice cross-section idea of
[`Becksteinlab/Protein_Area`](https://github.com/Becksteinlab/Protein_Area)
(Voronoi-cell areas per membrane-normal slice of an MD trajectory), adapted to
single static structures: the slice axis defaults to the principal axis instead
of a hardcoded membrane normal, and the membrane-free `hull` method is the
dependency-light default.

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

### Relative solvent accessibility (RSA)

`mol.relative_sasa()` normalises each residue's absolute SASA by a reference
maximum (Tien et al. 2013) to give RSA, then classifies residues as exposed or
buried — a high-signal per-residue feature for interface/binding-site work and
residue-level graphs:

```python
exp = mol.relative_sasa(threshold=0.20)     # ResidueExposure, per residue
exp.rsa                                       # relative SASA (NaN where no reference)
exp.exposed                                   # bool: rsa >= threshold
exp.sasa                                      # absolute SASA (Å²)
zip(exp.resnames, exp.resids, exp.exposed)    # label each residue
```

RSA can slightly exceed 1 (the reference is an extended Gly-X-Gly tripeptide),
and residues with no reference (ligands, waters, non-standard names) get `NaN`
RSA and count as not exposed. SASA is computed on the whole structure, so burial
reflects neighbours. It pairs naturally as a custom node feature on a
[residue contact graph](molecular-graphs.md).

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
