# API Reference

## Top-level functions

- `molscope.read(path)`: read a molecule by extension.
- `molscope.fetch(pdb_id, fmt="pdb")`: download from RCSB and read.
- `molscope.read_pdb(path)`, `read_pdb_models(path)`, `read_xyz(path)`, `read_xyz_frames(path)`, `read_cif(path)`, `read_sdf(path)`, `read_sdf_frames(path)`.
- `molscope.read_sdf_frames(path)`: read every record of a multi-record SDF as a list of molecules (one per docking pose), keeping each pose's 3D coordinates and exposing its `> <tag>` data fields (e.g. Vina/Gnina scores) via `Molecule.properties`.
- `molscope.validate_cif(path)`: optional Gemmi-backed CIF/mmCIF validation.
- `molscope.write_pdb(molecule, path)`, `write_xyz(molecule, path)`, `write_sdf`, `write_cif`.
- `molscope.write_frames(frames, path)`: write a list/generator of molecules as a multi-frame `.pdb`/`.xyz`/`.sdf` file (streaming, O(1) memory).
- `molscope.featurize_many(paths, return_names=False)`: build an ML feature matrix.
- `molscope.descriptor_feature_names(preset)`: stable flattened descriptor columns.
- `molscope.pocket_descriptor_feature_names("pocket-basic")`: stable binding-pocket descriptor columns.
- `molscope.node_feature_names(preset)`, `edge_feature_names(preset)`: atom/bond graph preset columns.
- `molscope.residue_node_feature_names(preset)`, `residue_edge_feature_names(preset)`: residue contact graph preset columns.

Graph-dataset assembly (the ML on-ramp):

- `molscope.build_dataset(source, *, fmt="pyg", node_features=..., labels=..., split=..., cache_dir=...)`: read, featurise, label-join, and split a folder/list of structures into a `GraphDataset`. `cache_dir=` enables an on-disk featurisation cache.
- `molscope.fetch_dataset(ids, *, labels=..., **build_kwargs)`: same, starting from RCSB accessions (downloads each, cached, then `build_dataset`).
- `GraphDataset`: holds `.graphs`/`.ids`/`.labels`/`.skipped`, the `.train`/`.val`/`.test` split views, `.summary()`, and `.save()`/`.load()`. `.loader(split=None, *, batch_size=1, shuffle=None)` returns a PyG/DGL batching `DataLoader`; `.standardize_targets()` fits a train-only `TargetScaler` and standardises `data.y`.
- `molscope.interface_residues(mol, chain_a, chain_b, cutoff=5.0)`, `chain_contact_matrix(mol, cutoff=5.0)`: chain interfaces.
- `molscope.ligands(mol, ...)`, `binding_site(mol, ligand=None, cutoff=4.5)`: ligand detection and binding-site residues.
- `molscope.select_pocket(mol, ligand=None, cutoff=4.5)` (also `Molecule.select_pocket(...)`): returns a `Pocket` (a `BindingSite` bound to its molecule) whose `describe_environment()` renders a chemistry-aware natural-language paragraph for LLM / RAG prompts; `environment()`/`analyze_pocket(mol, site)` return the structured `PocketEnvironment`.
- `molscope.backbone_torsions(mol)`: per-residue phi/psi/omega.
- `molscope.sasa(mol, probe_radius=1.4, n_points=192, level="atom")`: approximate Shrake-Rupley solvent-accessible surface area (also `Molecule.sasa(...)`).
- `molscope.cross_section_profile(mol, axis="principal", thickness=1.0, method="hull")` (also `Molecule.cross_section_profile(...)`): cross-sectional area profile along an axis, returning a `CrossSectionProfile`; `ms.plot_cross_section(profile)` draws area vs. position.

Residue identity helpers:

- `molscope.ResidueId(chain, resid, insertion_code="", resname="")`: full residue identity used by PDB/mmCIF-aware APIs.
- `molscope.ResidueGroup`: yielded by `Molecule.residue_groups()`; has `.residue_id` and still unpacks as `(atom_indices, resname, resid, chain)`.

`BindingSite` results expose `to_records()`, `to_molecule(mol)`,
`descriptors(mol, preset="pocket-basic")`, `describe_environment(mol)`, and
`plot(mol)` for residue tables, pocket descriptor extraction, LLM-ready prose,
and quick figures.

## Molecule

Construction:

```python
mol = ms.Molecule(coords, elements, name="example")
```

Common methods:

- `select(...)`, `backbone()`, `alpha_carbons()`, `protein()`, `hetero_atoms()`, `chain_ids()`
- `translate(...)`, `centered(...)`, `rotate(...)`, `superpose(...)`
- `distance(...)`, `angle(...)`, `dihedral(...)`
- `centroid`, `center_of_mass`, `radius_of_gyration`, `dimensions`
- `inertia_tensor()`, `principal_moments()`, `principal_axes()`, `cross_section_profile(...)`
- `distance_matrix(backend="numpy")`, `contacts(...)`, `contact_count(...)`, `contact_map(...)`
- `secondary_structure()`, `backbone_torsions()`, `interface(...)`, `chain_contacts(...)`, `ligands(...)`, `binding_site(...)`
- `bonds(...)`, `bond_order_array(...)`
- `descriptors(...)`, `rdkit_descriptors(...)`
- `chemical_features(...)`
- `coarse_grain(..., virtual_sites=None)`, `mapping_report()`
- `to_graph()`, `to_networkx()`, `to_pyg_data()`, `to_dgl_graph()`
- `to_residue_contact_graph()`
- `plot(...)`, `view(...)`, `spin_gif(...)`

## Other modules

- `molscope.ensemble`: RMSD matrices, alignment, average structures, RMSF, dynamical cross-correlation (`cross_correlation`), clustering.
- `molscope.contactmap`: contact map construction, metrics, and plotting.
- `molscope.contacts`: chain interfaces and ligand-binding-site analysis.
- `molscope.dssp`: simplified DSSP-style secondary-structure assignment, segments, and backbone torsions.
- `molscope.distance`: optional NumPy, PyTorch, and CuPy dense distance backends.
- `molscope.coarsegrain`: coarse-graining, virtual-site metadata, and mapping report classes.
- `molscope.descriptors`: descriptor helpers and batch featurization.
- `molscope.graph`: graph container and backend exporters.
- `molscope.chem`: optional RDKit-backed chemical perception and descriptors.
- `molscope.docking`: post-docking triage — `read_poses`, `summarize`, `select_diverse_hits`, and `consensus_rank` behind the `dock-summary`, `dock-diverse`, and `dock-rank` CLI commands. See [Docking-hit triage](user-guide/docking-triage.md).
