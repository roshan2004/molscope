# Changelog

All notable changes to MolScope are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the package is pre-1.0, minor versions may include backwards-incompatible
API changes; these are called out under **Changed** where they occur.

## [Unreleased]

### Added

- ``build_dataset``: assemble an ML graph dataset from structure files in one
  call. Discovers files (glob, list of paths, or list of ``Molecule`` objects),
  featurises each to a graph (``fmt="pyg"``/``"dgl"``/``"networkx"``/``"raw"``)
  with optional positional encodings, joins labels from a dict or CSV, and
  applies an optional random train/val/test split. Returns a ``GraphDataset``
  with ``.train``/``.val``/``.test``, ``.summary()`` and ``.save()``. It is a
  thin layer over the existing exporters, so ``pyg``/``dgl`` need their extras
  and ``raw``/``networkx`` run on the core install.
- MCP server now accepts SMILES anywhere a structure ``source`` is expected:
  pass ``"smiles:<SMILES>"`` (e.g. ``"smiles:CCO"``) to any of the 23 tools and it
  builds the molecule from one RDKit conformer (``chem`` extra). This makes the
  descriptor and graph tools reachable from SMILES, not just files and PDB ids.
- Every MCP tool now carries a human-readable title and MCP tool annotations
  (``readOnlyHint``/``openWorldHint``/``idempotentHint``), so clients can tell the
  read-only analysis tools from the ones that fetch over the network or write
  files.
- ``read_smiles``: build a ``Molecule`` from a SMILES string by generating a
  single 3D conformer with RDKit (``chem`` extra), carrying RDKit's bonds, Kekule
  bond orders, and formal charges. This makes the descriptor and graph-ML
  workflows reachable directly from SMILES. The coordinates are a generated
  conformer, not an experimental or minimised structure (labelled as such), so it
  is best for topology-based work, not geometry-dependent analysis.

### Changed

- MCP tools turn a missing optional dependency into actionable install guidance:
  a bare ``No module named 'rdkit'`` (or ``gemmi``/``networkx``/etc.) is mapped to
  the ``pip install "molscope[...]"`` extra that provides it.
- Sharpened the project framing: a single identity line ("a lightweight bridge
  from molecular structures to descriptors, contact maps, graph-ML inputs, and
  educational coarse-grained representations") now leads the README, package
  description, and docs landing page. The README was restructured as a landing
  page (problem, install, three-line example, docs link) with the long API
  walkthrough moved to per-capability user-guide pages, and gained a per-feature
  "what is and isn't validated" table.

## [0.10.0] - 2026-05-29

### Added

- Residue-template bond perception for proteins: ``read``/``read_pdb``/``fetch``
  accept ``bond_perception="template"``, which uses RDKit's residue-aware PDB
  reader to attach explicit bonds, Kekule bond orders, and formal charges for
  standard residues (needs the ``chem`` extra). This fixes aromaticity and bond
  orders that geometric distance inference cannot recover, so RDKit-backed
  descriptors and ``chemical_features`` are correct on proteins. The MCP
  ``chemical_features`` tool now defaults to template perception for PDB inputs.
  Default behaviour of ``read`` elsewhere is unchanged (still ``"geometric"``).
  A Tier-2 validation test cross-checks perceived aromaticity against the known
  per-residue chemistry (Phe/Tyr 6, Trp 9, His 5 aromatic atoms).
- Idealised standard protonation: ``protonation="standard"`` (with template
  bonds) assigns pH-7 side-chain charges for standard residues (Asp/Glu -1,
  Lys/Arg +1, His neutral, termini uncharged; see
  ``molscope.chem.STANDARD_PROTONATION``), so ``formal_charges`` is meaningful
  (e.g. trypsin nets +6) instead of uniformly zero. The MCP ``chemical_features``
  tool defaults to it and labels the assignment. It is a textbook model, not a
  pKa-aware prediction; the default stays ``"none"`` (as-modelled neutral).
- The MCP render tools (`render_structure`, `render_contact_map`,
  `render_distance_matrix`, `render_rmsd_heatmap`) now take an optional
  `save_path`: when given, the figure is written to disk and the tool returns the
  file path (format follows the extension: png/pdf/svg/...), so a user gets a
  real file instead of only an inline image. Omitting it keeps the previous
  inline-image behaviour.
- Expanded the MCP server from 9 to 21 tools so it fronts most of the package:
  `geometry`, `measure` (distance/angle/dihedral), `rmsd`, `ensemble_summary`
  (multi-model RMSD/RMSF/clusters), `chemical_features`, `backbone_torsions`,
  `list_ligands`, `chain_interfaces`, `validate_cif`, `select_diverse`, and the
  `render_distance_matrix` / `render_rmsd_heatmap` plot tools. All NaN/inf values
  are now emitted as JSON `null` so tool output is always valid JSON.

## [0.9.0] - 2026-05-29

### Added

- Molecule-table workflow: `molscope select` and the `molscope.library` module
  read a CSV/XLSX of molecules and pick a diverse subset by MaxMin
  (farthest-first) selection over descriptors. Select on existing numeric columns
  (e.g. `MW`, `ALogP`) or compute RDKit descriptors from a SMILES column with
  `--compute-descriptors --smiles-col`. Adds an `xlsx` extra (`openpyxl`) for
  spreadsheet I/O; CSV input and selection need no optional backend.
- Optional MCP (Model Context Protocol) server, `molscope.mcp_server`, exposing
  MolScope's analyses as tools for AI assistants such as Claude Code and Claude
  Desktop. Adds a `molscope-mcp` console script, an `mcp` extra
  (`pip install "molscope[mcp]"`, Python >= 3.10), and nine read-only tools that
  wrap the existing API: summarise, descriptors, secondary structure, contact
  map, binding site, molecular graph, coarse-grain, and two PNG render tools.
- Broadened the DSSP reference cross-check to three fold classes instead of one:
  helix-dominated Aquaporin-1 (`1fqy`), mixed alpha/beta ubiquitin (`1ubq`), and
  the all-beta SH3 domain (`1shg`). The test is parametrised and prints
  per-fold 3-state agreement so results read as a range. Measured against
  `mkdssp` 4.5.8: 99.1% (`1fqy`), 100% (`1ubq`), 98.2% (`1shg`).
- Bundled `1ubq.pdb` and `1shg.pdb` in `examples/data` as the new validation
  structures.

### Changed

- Documentation and the JOSS paper now report DSSP agreement as a measured
  range across fold classes rather than a single helical figure, and no longer
  imply that strand-rich folds agree markedly less well.

## [0.8.3] - 2026-05-28

### Added

- JOSS paper draft under `paper/` and Zenodo deposition metadata
  (`.zenodo.json`) for an archival DOI.

### Changed

- Bumped GitHub Actions to Node 24-ready versions.

## [0.8.2] - 2026-05-28

### Added

- Read the Docs hosting for the documentation site.
- Coverage reporting via `pytest-cov` and Codecov, measured with the optional
  backends installed so the RDKit, gemmi, NetworkX, SciPy and Torch paths are
  exercised rather than skipped.
- Coarse-grained virtual-site support, preserved as derived coordinate metadata.
- PDB workflow tutorials and a protein-analysis-from-scratch walkthrough.
- CLI batch analysis and graph-export subcommands, with improved selection
  handling.
- `O(n)` cell-list neighbour search and opt-in periodic-boundary support for the
  distance and contact methods.
- Residue contact graphs for ML and an educational coarse-graining workflow.
- Scientific validation tables, references, and a reproducible benchmarks page.

### Changed

- Reorganised the limitations page by workflow and added a Graph ML section.
- Refocused the documentation around the three core workflows.
- Completed the geometry API and added a visual geometry guide validated against
  MDAnalysis.
- Polished the coarse-graining workflow, contact maps, and dense distance
  backends; improved coordinate-format parser errors and added edge-case
  fixtures.
- Stopped tracking generated graph exports and added `graphs/` to `.gitignore`.

### Fixed

- Batch CLI crash with `--jobs > 1` on spawn-based platforms.
- gemmi-backed mmCIF unit-cell read.
- Lint errors, and made periodic boundaries opt-in for distance and contact
  methods.

## [0.8.1] - 2026-05-26

### Added

- Tier-2 validation suite cross-checking against reference scientific tools,
  covering DSSP (`mkdssp`), geometry and RMSD (MDAnalysis), bond perception and
  chemical features (RDKit), and contact maps.
- PyTorch Geometric ML tutorial and `CITATION.cff` citation metadata.

### Changed

- DSSP validation now invokes `mkdssp` directly instead of going through
  Biopython, and the 3-state agreement floor was tightened to 0.95 after
  observing 99.1% on CI.
- CI runs the validation job with the required extras and fails loudly if the
  reference tools cannot be imported.
- Polished repository layout and documentation.

### Fixed

- README Mermaid diagram syntax.

## [0.8.0] - 2026-05-26

### Added

- Expanded molecular parsing and machine-learning feature support.

## [0.7.0] - 2026-05-25

### Added

- Simplified, dependency-free DSSP-style secondary-structure assignment based on
  the Kabsch-Sander hydrogen-bond model.

### Changed

- README: added a "Why MolScope" section, a tool comparison, and a CLI output
  example; the secondary-structure render replaced the earlier hero animation.

## [0.6.2] - 2026-05-25

### Fixed

- README images now render on PyPI, and the publish workflow was hardened.

## [0.6.1] - 2026-05-25

### Added

- PyPI trusted-publishing workflow.

### Changed

- Expanded the package docstring to cover the full feature scope.

## [0.6.0] - 2026-05-25

Initial public release under the **MolScope** name, renamed from the earlier
`molecule3d` prototype. This release consolidated the core toolkit:

### Added

- `Molecule` object on a NumPy core, with fixed-column PDB parsing and readers
  and writers for XYZ, PDB, mmCIF and SDF.
- Per-atom metadata, metadata-based selections, geometry measurements, and RMSD.
- Molecular graph construction with NetworkX, PyTorch Geometric and DGL
  exporters.
- Coarse-graining tools: residue and custom mappings, explicit-bond support,
  index-based mappings, and a dropped-atom warning.
- Contact maps and ensemble contact-frequency analysis, plus ensemble RMSD
  clustering and an RMSD heatmap.
- Native structural descriptors and a MkDocs documentation site, including a
  user-guide PDF builder.
- `uv` support (lockfile, dev dependency group, `.python-version`), continuous
  integration, and README visual examples.

[Unreleased]: https://github.com/roshan2004/molscope/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/roshan2004/molscope/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/roshan2004/molscope/compare/v0.8.3...v0.9.0
[0.8.3]: https://github.com/roshan2004/molscope/compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com/roshan2004/molscope/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/roshan2004/molscope/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/roshan2004/molscope/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/roshan2004/molscope/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/roshan2004/molscope/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/roshan2004/molscope/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/roshan2004/molscope/releases/tag/v0.6.0
