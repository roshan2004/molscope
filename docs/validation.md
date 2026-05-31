# Scientific validation

MolScope is a lightweight teaching and prototyping toolkit, so validation is
not just "tests pass". For every scientific method, ask:

- What is the reference?
- What assumptions does the method make?
- Where does it fail?
- What tolerance is scientifically reasonable?

The validation suite is split into two tiers:

- **Tier 1 invariants** run everywhere and check mathematical or conservation
  truths that do not need an external tool.
- **Tier 2 reference comparisons** run when optional scientific tools are
  installed: MDAnalysis, RDKit, and `mkdssp`/`dssp`.

Run the full validation layer locally:

```bash
uv run pytest tests/validation -v -rs -s
```

The binding-site panel includes opt-in RCSB downloads. Run it explicitly with:

```bash
MOLSCOPE_RUN_REMOTE_PDB=1 uv run pytest tests/validation/test_binding_sites_ref.py
```

Install optional Python references with:

```bash
uv sync --extra validation
```

The secondary-structure reference additionally needs a system `mkdssp` or
`dssp` executable on `PATH`.

## Current panel scope

The reference checks are targeted scientific smoke tests rather than a full
benchmark. The DSSP panel spans three fold classes (`1fqy` helical, `1ubq`
mixed alpha/beta, `1shg` all-beta); the bond and chemistry panels cover 18 and
12 small molecules respectively across fused rings, O/N/S heteroaromatics,
halogens, sulfur, amides, a strained ring, and charged/zwitterionic species,
plus a stretched-bond negative case where distance-only perception is expected
to fail. a panel of real solution-NMR ensembles (`1aml`, plus the gzipped `1d3z`, `2lz3`,
`6qfp`, `1gab`, and the opt-in remote `6v5d`) cross-checks the alignment metrics
against MDAnalysis, complemented by deterministic synthetic-ensemble invariants.
`3ptb`
exercises the bundled binding-site path, and the opt-in remote panel adds
`1stp`, `1iep`, `3ert`, `1hsg`, `4hvp`, and `2br1` for ligand ambiguity,
multi-chain complexes, cofactors and larger inhibitors.

The public Delaney ESOL solubility set (1128 compounds) exercises the
dataset-prep pipeline end to end on real, messy SMILES -- scaffold/random splits,
canonical deduplication and fingerprinting -- and doubles as a large-scale
descriptor wrapper-transparency check against a fresh RDKit call. This is enough
to catch regressions across fold classes and chemistry families, but it is still
a curated mini-panel, not an exhaustive benchmark.

## Reference-tool checks

| Area | Reference | Validation file | Panel | Tolerance / threshold | Rationale |
| --- | --- | --- | --- | --- | --- |
| Mass geometry | MDAnalysis | `tests/validation/test_geometry_ref.py` | `1fqy.pdb` | `radius_of_gyration` relative `1e-6`; center of mass absolute `1e-5`; inertia relative `1e-5` | Same formulas and same PDB coordinates should agree to floating-point precision. |
| Geometry primitives | MDAnalysis | `tests/validation/test_geometry_ref.py` | `1fqy.pdb` | distances relative `1e-5`; angles/dihedrals absolute `1e-4` degrees | Coordinate precision and degree conversion dominate error. |
| CA distance/contact maps | MDAnalysis | `tests/validation/test_geometry_ref.py` | `1fqy.pdb` alpha carbons | distance matrix absolute `1e-5`; contact pairs exact at 8 A | Contact-map logic should match an independent distance-threshold implementation. |
| Ensemble RMSF/RMSD | MDAnalysis | `tests/validation/test_geometry_ref.py` | `1aml.pdb` NMR ensemble | RMSF absolute `1e-3`; Kabsch RMSD absolute `1e-4` | Alignment and trajectory APIs differ slightly, but biologically meaningful values should agree tightly. |
| Ensemble RMSF/RMSD (NMR panel) | MDAnalysis | `tests/validation/test_ensembles_ref.py` | bundled `1d3z`, `2lz3`, `6qfp`, `1gab`; opt-in remote `6v5d` | RMSF absolute `1e-3`; Kabsch RMSD absolute `1e-4` | Real solution-NMR ensembles across sizes and folds, gzipped to ~1 MB. `2hyn` (remote) additionally confirms that an ensemble whose models carry inconsistent atom counts is rejected, not silently misaligned. |
| Distance bond perception | RDKit topology | `tests/validation/test_bonds_ref.py` | 18 small molecules spanning fused rings, O/N/S heteroaromatics, halogens, sulfoxide, amide and a strained ring; plus a stretched-bond negative case | bond precision and recall each `>= 0.98` on clean geometries; the stretched bond is expected to be *missed* | Geometry-only perception should recover clean equilibrium topologies, and should provably fail on non-equilibrium geometry (the honest reason template bonds exist). |
| Chemical features | RDKit atom/bond APIs | `tests/validation/test_chem_ref.py` | 12 molecules: aromatics, O/N/S heteroaromatics, anion, cation, zwitterion, and histidine | formal charges and aromatic flags exact; bond orders exact within `1e-12` | MolScope delegates optional chemical perception to RDKit, so direct RDKit arrays are the reference. |
| RDKit descriptors | RDKit descriptor APIs | `tests/validation/test_chem_ref.py` | Same chemistry panel | selected scalar descriptors relative/absolute `1e-12` | Descriptor wrappers should not alter RDKit descriptor values. |
| Descriptors at scale | RDKit | `tests/validation/test_esol_ref.py` | Delaney ESOL solubility set (1128 compounds) | absolute `1e-9` vs a fresh RDKit call | Stretches the wrapper-transparency contract across a large, diverse, real chemistry set; version-proof since both sides use the installed RDKit. |
| Secondary structure | `mkdssp` / `dssp` | `tests/validation/test_dssp_ref.py` | `1fqy.pdb` (helical), `1ubq.pdb` (mixed alpha/beta), `1shg.pdb` (all-beta) | 3-state helix/strand/coil agreement per fold (`>= 0.95` helical, `>= 0.90` mixed and all-beta); helix fraction within `0.15` | MolScope's DSSP is simplified and educational, so reduced-state agreement is the honest target rather than byte-for-byte 8-state equality. The set spans three fold classes so agreement is reported as a range, not a single helical best case. |
| Binding sites | RCSB structures with HETATM ligands | `tests/validation/test_binding_sites_ref.py` | `3ptb`; opt-in remote panel `1stp`, `1iep`, `3ert`, `1hsg`, `4hvp`, `2br1` | residue records and `pocket-basic` descriptors finite and internally consistent | Real protein-ligand files expose ambiguity, multi-chain sites, cofactors, ions and larger inhibitors better than synthetic fixtures. |
| Multi-pose SDF parsing | RDKit `SDMolSupplier` | `tests/validation/test_docking_ref.py` | Hand-authored `docking_poses.sdf`; generated bonded ligands (benzene, aspirin, caffeine) | pose count and titles exact; score data fields exact; coordinates absolute `1e-4` | `read_poses` underlies every dock-* tool; an independent SDF parser is the natural reference. The hand-authored fixture is written by neither library, so this is a true two-parser cross-check. |

## Invariant checks

| Area | Validation file | Assertion | Tolerance |
| --- | --- | --- | --- |
| Rigid-body alignment | `tests/validation/test_invariants.py` | Kabsch alignment recovers a known rotation/translation | RMSD `< 1e-9` |
| Self RMSD | `tests/validation/test_invariants.py` | A structure aligned to itself has zero RMSD | RMSD `< 1e-12` |
| Geometry primitives | `tests/validation/test_invariants.py` | Euclidean distances, right angles, planar torsions | Exact or near machine precision |
| Radius of gyration | `tests/validation/test_invariants.py` | Uniform shell has radius of gyration equal to shell radius | Absolute `< 1e-3` |
| Coarse-graining | `tests/validation/test_invariants.py` | Residue COM and centroid beads equal direct reductions of source atoms | Absolute `< 1e-9` |
| Contact maps | `tests/validation/test_invariants.py` | Atom contact map equals brute-force all-pairs threshold | Exact matrix equality |
| Ensemble alignment | `tests/validation/test_invariants.py` | A rigidly-moving ensemble has zero RMSF and zero pairwise RMSD after superposition; a single displaced atom carries the largest RMSF | Absolute `< 1e-5`; argmax exact |
| Consensus ranking | `tests/validation/test_docking_ref.py` | A single score field reproduces that field's ranking; a pose best on every field takes rank 1 | Exact order |
| Ligand efficiency | `tests/validation/test_docking_ref.py` | Equals the signed score per heavy atom | Relative `1e-9` |
| Diversity selection | `tests/validation/test_docking_ref.py` | Identical molecules collapse to one (best-scoring) representative; representatives come from distinct clusters | Exact |
| Graph export | `tests/validation/test_graph_invariants.py` | Node count equals atom count, edge set equals the bond set, adjacency is symmetric; NetworkX and PyG exports preserve the same nodes/edges | Exact |
| Dataset assembly | `tests/validation/test_graph_invariants.py` | `build_dataset` keeps ids/labels/graphs aligned and survives a raw save/load round trip | Exact |
| Dataset-prep pipeline | `tests/validation/test_esol_ref.py` | On the real 1128-molecule ESOL table: random/scaffold splits are disjoint covers, canonical dedup collapses exactly the 11 duplicates, fingerprinting and diverse selection produce valid output | Exact |

## Assumptions and failure modes

| Method | Key assumptions | Expected failure modes |
| --- | --- | --- |
| Geometric bonds | Clean 3D coordinates, normal covalent distances, standard elements | Missing/extra bonds for strained structures, metals, unusual valence, bad coordinates, or raw PDB files without explicit chemistry. |
| RDKit chemical features | Explicit bond orders/formal charges or a geometry whose inferred single-bond graph RDKit can sanitize | Sanitization errors for inconsistent valence or missing bond-order chemistry; aromaticity depends on RDKit's model and version. |
| Contact maps | Static coordinates and a chosen distance cutoff/method (`ca`, `com`, or `min`) | Different cutoffs or representative atoms change the result; dense atom maps are `O(N^2)`. |
| Simplified DSSP | Complete protein backbone atoms (`N`, `CA`, `C`, `O`) and standard residue ordering | Not canonical `mkdssp`; boundary residues of helices and strands are where disagreements concentrate; bare XYZ input is insufficient. |
| Coarse-graining | Beads are coordinate reductions and simple bead graphs for inspection | No force-field parameters, charges, exclusions, elastic networks, or validation of simulation behavior. |
| Docking triage | A score data field is present in the SDF; V2000 records; fingerprint-based similarity for clustering | Reads docking output but does not dock, prepare, or re-score; the consensus rank is rank aggregation, not a calibrated affinity; diversity depends on the fingerprint and Tanimoto threshold. |

## Updating validation

When adding a scientific feature, add at least one of:

- an invariant test if the expected behavior follows from math or conservation,
- a reference-tool comparison if a credible external implementation exists,
- a limitations-table row if the method is intentionally approximate.

Prefer tight tolerances when two implementations should be numerically
equivalent. Use looser, justified thresholds only when the method is explicitly
approximate, as with simplified DSSP.
