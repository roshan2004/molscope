# Changelog

All notable changes to MolScope are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the package is pre-1.0, minor versions may include backwards-incompatible
API changes; these are called out under **Changed** where they occur.

## [Unreleased]

### Added

- Surface and interaction descriptors in the ``native-3d`` preset, all pure-NumPy
  with no new dependencies: SASA summary statistics (``sasa_total``,
  ``sasa_mean``, ``sasa_std``, ``sasa_max``) from the existing Shrake-Rupley
  approximation (computed at a coarser ``sasa_n_points`` default of 96 for batch
  speed, configurable); a ``salt_bridge_count`` (basic side-chain N within 4 Å of
  an acidic side-chain O, counting unique Arg/Lys/His↔Asp/Glu residue pairs); and
  a ``polar_contact_count`` (N/O atom pairs 2.5-3.5 Å apart in different residues
  — a coarse geometric proxy for polar contacts, deliberately *not* named a
  hydrogen-bond count since it has no angle or hydrogen-position check, and
  MolScope already exposes rigorous H-bonds via DSSP and RDKit donor/acceptor
  counts). These are computed only for ``native-3d`` (and the unfiltered default),
  so ``native-basic``/``rdkit-basic`` keep their columns and their speed.

- ``molscope coarse-grain`` CLI subcommand: map a structure to coarse-grained
  beads and write a coordinate file from the command line, closing the
  pillar-parity gap (coarse-graining was the only core workflow without a CLI
  verb). ``molscope coarse-grain structure.pdb --mapping martini --out cg.pdb``
  reads the structure, applies the chosen mapping (``residue_com`` default,
  ``residue_centroid`` or ``martini``; or ``--fetch PDBID``), and writes the
  beads by output extension. Beads are pseudo-atoms with their bead names
  (``BB``/``SC`` for Martini), and the bead bond network is written as ``CONECT``
  records for ``.pdb`` output so it loads into PyMOL/ChimeraX/Mol\*; ``.cif`` and
  ``.xyz`` carry coordinates only (a note is printed). Wires the existing
  ``coarsegrain`` module and writers to the CLI; no new dependencies.

- ``view_mapping`` (and ``Molecule.view_mapping``): a notebook-friendly py3Dmol
  overlay of a coarse-grained model on its atomistic source. The atomistic
  structure is drawn as a semi-transparent model (``atom_style`` ``"stick"`` by
  default, or ``"cartoon"``/``"line"``/``"sphere"``) with each bead as a solid
  sphere at its position, coloured by the same palette as the matplotlib
  ``plot_mapping`` (virtual sites white, CG bonds as thin cylinders). It is the
  interactive, rotatable counterpart to ``plot_mapping`` for teaching and
  designing custom mappings, and builds only on the existing optional ``[viz]``
  py3Dmol extra. The coarse-grain report is validated before py3Dmol is imported,
  so a non-CG input fails the same way with or without the extra.

### Changed

- The ``native-3d`` descriptor preset gained six columns (``sasa_total``,
  ``sasa_mean``, ``sasa_std``, ``sasa_max``, ``polar_contact_count``,
  ``salt_bridge_count``), widening its feature vector. Consumers that hard-code
  the column count should regenerate it via ``descriptor_feature_names("native-3d")``.
  ``native-basic`` and ``rdkit-basic`` are unchanged.

## [0.12.0] - 2026-06-02

### Added

- Gyration-tensor shape descriptors in the ``native-3d`` descriptor preset:
  ``asphericity`` (b), ``acylindricity`` (c) and ``relative_shape_anisotropy``
  (κ²), the standard polymer-physics shape parameters. They are computed via
  ``descriptors.shape_descriptors`` from the eigenvalues of the gyration tensor
  (``λ₁ ≤ λ₂ ≤ λ₃``), which are recovered from the already-computed mass-weighted
  principal moments of inertia — no new dependencies and negligible runtime. κ²
  runs from 0 (sphere / higher-symmetry) to 1 (linear). These are distinct from
  the existing ``shape_anisotropy`` column, which applies a similar formula to the
  inertia moments directly (left unchanged).

- ``Molecule.sasa(...)`` (and ``molscope.sasa``): an approximate solvent-accessible
  surface area in Å² from a vectorised, pure-NumPy Shrake-Rupley sphere — a fast,
  dependency-free descriptor of solvent exposure with no C extensions or external
  SASA libraries. Each atom's expanded sphere (Bondi van der Waals radius plus a
  ``probe_radius`` water probe, 1.4 Å by default) is sampled with ``n_points``
  quasi-uniform Fibonacci points; ``level="atom"`` returns per-atom values and
  ``level="residue"`` sums them per residue. Accuracy scales with ``n_points``
  (default 192, within a few percent of an exact analytical surface). Neighbour
  search reuses the optional SciPy KD-tree with a NumPy fallback. A ``sasa`` MCP
  tool reports the total and the most solvent-exposed residues, and ``elements``
  gains a Bondi ``vdw_radius`` table. It is deliberately **not** folded into the
  fixed ``descriptors()`` presets, so those feature columns stay stable.

- ``ensemble.cross_correlation(models)`` (exported as ``molscope.cross_correlation``):
  the dynamical cross-correlation matrix (DCCM) of coordinate fluctuations across
  an ensemble or trajectory, a symmetric ``(N, N)`` matrix in ``[-1, 1]`` where
  ``+1`` is lockstep motion, ``-1`` anticorrelated and ``0`` uncorrelated — the
  standard primitive for spotting concerted motions and allosteric coupling.
  Structures are Kabsch-superposed first (like ``rmsf``) so rigid-body tumbling
  does not swamp the internal motion; static atoms are handled (off-diagonal 0,
  diagonal 1). Pass alpha-carbon-only models for the usual residue-level map. A
  diverging ``plot_cross_correlation`` heatmap and a ``render_cross_correlation``
  MCP tool accompany it. (A binary "contact fluctuation" std was considered and
  rejected: the standard deviation of a 0/1 contact is ``sqrt(p(1-p))``, fully
  determined by the existing ``contact_frequency``, so it would add no
  information.)

- Configurable spatial-proximity edge construction for the atom/bond graph
  builders. ``to_graph`` (and the ``to_pyg_data``/``to_dgl_graph``/``to_networkx``
  shortcuts) now accept ``knn=k`` to build edges from each atom's ``k`` nearest
  neighbours by Euclidean distance (union-symmetrised, capped at ``n - 1``) or
  ``radius=r`` to connect every atom pair within ``r`` angstrom (reusing the fast
  ``contacts`` KD-tree / cell-list search), or ``delaunay=True`` to build the
  Delaunay triangulation / Voronoi-adjacency graph — a threshold-free,
  density-adaptive option that avoids the dense-core/exposed-loop disparities of
  a fixed cutoff (requires SciPy; unlike k-NN/radius it has no pure-NumPy
  fallback, and a raw Delaunay graph adds some long convex-hull/surface edges
  best pruned with ``min_seq_sep`` or a distance filter). These are the standard
  ways to build graphs for 3D macromolecular GNNs. At most one of ``knn``,
  ``radius``, ``delaunay`` and an explicit ``bonds=`` array may be given.
  ``min_seq_sep`` drops same-chain edges whose residue-id separation is below the
  threshold (filtering trivial local backbone contacts) and applies to every edge
  mode; inter-chain edges are always kept. All are exposed on the ``export`` CLI
  subcommand (``--knn``/``--radius``/``--delaunay``/``--min-seq-sep``) and the
  ``molecular_graph`` MCP tool, and the ``knn_edges``/``delaunay_edges`` building
  blocks are published on the package. As part of this,
  ``include_chemical_features`` now aligns RDKit aromatic flags for explicit and
  geometric edge sets too, not only inferred covalent bonds.

- Registration instructions for the MCP server with **Codex CLI**
  (`~/.codex/config.toml` / `codex mcp add`) and **Gemini CLI**
  (`~/.gemini/settings.json` / `gemini mcp add`), alongside the existing Claude
  Code/Desktop docs. No code change: the same stdio `molscope-mcp` server works
  with any MCP client; only the registration syntax differs.

- ``write_sdf`` and ``write_cif`` writers, completing read/write symmetry with the
  existing ``write_xyz``/``write_pdb`` (MolScope already read all four formats).
  Both are lightweight and dependency-free, and round-trip with MolScope's own
  readers: ``write_sdf`` emits a V2000 record preserving coordinates, elements,
  bonds and Kekulé orders (aromatic written as code ``4``), formal charges (via an
  ``M  CHG`` block) and string ``properties``; ``write_cif`` emits a minimal
  ``_atom_site`` loop preserving coordinates, elements, atom/residue names,
  residue numbers, chain ids, and the ATOM/HETATM flag. SDF is capped at 999
  atoms/bonds (V2000) and does not carry chain/residue metadata; the CIF writer is
  a coordinate file (no symmetry/anisotropy/bonds).

- ``ensemble.analyze_stream`` (and ``ms.StreamAnalysis``): a single-pass,
  O(1)-memory trajectory-lite analyzer. Given a multi-frame file path (streamed
  via ``ms.stream``) or any iterable of frames, it tracks per-frame radius of
  gyration, RMSD to the first frame (Kabsch-superposed over a selection:
  ``"auto"`` uses C-alphas when present else all atoms, or force ``"ca"``/``"all"``),
  and optionally helix/strand/coil fractions (``secondary_structure=True``,
  proteins only; a frame whose assignment fails contributes ``NaN``). Frames must
  share the first frame's atom count or a ``ValueError`` is raised. Returns a
  ``StreamAnalysis`` with ``.summary()`` and ``.plot()`` (timeline panels via
  ``plotting.plot_stream_analysis``). It reads the multi-frame formats MolScope
  already reads (multi-model PDB, multi-frame XYZ, multi-record SDF) and is not a
  binary-trajectory (DCD/XTC/TRR) reader.

- Contact-map difference maps. ``ContactMap`` supports subtraction
  (``map_a - map_b``) to compare two structures, e.g. open vs closed states or
  before/after a point mutation. Subtraction guards that both maps share the same
  level, shape, residue/atom labels, and cutoff, raising ``ValueError`` otherwise,
  so mismatched structures can't be silently subtracted. The result carries
  ``is_difference=True`` (tracked explicitly, not inferred from values) with
  entries in ``[-1, 1]``; ``plot_contact_map`` renders it on a symmetric diverging
  colormap (gained contacts red, lost contacts blue) with difference-aware labels.

- ``prepare_structure`` (and ``ms.StructureReport``): a one-shot structure QC /
  readiness check that reads a file once and answers "is this ML-ready?". It
  reports non-standard residues, a ligand/water inventory, residue-numbering
  gaps, backbone chain breaks, residues missing backbone atoms or with truncated
  side chains, whether hydrogens are present, alternate conformations / partial
  occupancies (PDB), and the net formal charge at a chosen pH (reusing the
  ``"standard"``/``"pka"`` protonation backends). Topology checks need only the
  core NumPy install; the net-charge step degrades to a labelled ``None`` (with a
  note) when RDKit/PROPKA are absent or the file can't be template-parsed.
  Exposed as the ``molscope structure-report`` CLI command (text, ``--json``, or
  Markdown ``--out``) and the ``prepare_structure`` MCP tool. The ``ml_ready``
  verdict is a heuristic: missing backbone atoms and chain breaks are blockers,
  everything else is a warning. Adds ``io.fetch_file`` (download an RCSB entry
  and return its cached path).

- pKa-aware, environment-aware protonation. For proteins, ``protonation="pka"``
  (on ``read``/``read_pdb``/``fetch`` with ``bond_perception="template"``, and
  the ``chemical_features`` MCP tool) runs PROPKA to predict per-residue pKa from
  the structure and assigns side-chain and terminus charges for the dominant
  state at a chosen ``ph`` (default 7.0) — unlike the fixed ``"standard"`` table.
  For small molecules, ``prepare_dataset`` / ``smiles_descriptors`` /
  ``molscope prepare`` gain ``protonation="pka"`` + ``ph`` (and ``--protonation``
  / ``--ph`` on the CLI), which set each SMILES to its dominant ionisation state
  at the target pH (via Dimorphite-DL) before descriptors and fingerprints are
  computed; the stored SMILES and dedup keys are left untouched. New optional
  extras ``[propka]``, ``[dimorphite]``, and the umbrella ``[pka]``; each backend
  raises a clear install hint when absent.

- Streaming SDF readers for large docking files: ``stream_sdf_frames`` and
  ``molscope.docking.stream_poses`` / ``PoseStream`` yield records one at a time,
  keeping memory O(1) in the number of poses. ``ms.stream(...)`` now dispatches to
  SDF too. The ``dock-*`` commands and MCP tools read via ``PoseStream``, so a
  multi-hundred-thousand-pose file no longer has to be held in memory at once.
- ``dock-summary``/``dock-report`` (and the matching MCP tools) collapse multiple
  poses of the same compound to its single best pose by default
  (``--best-pose-per-ligand``, on by default; ``--no-best-pose-per-ligand`` to
  keep every pose). Compounds are keyed by SMILES when available, else by name
  with explicit pose suffixes (``_pose1``, ``_conf1`` …) removed. ``n_poses``
  still reports the total poses read; ``n_ranked`` reports the rows shown.
- ``dock-diverse`` now reports how many poses were dropped because RDKit could
  not build a fingerprint for them (``n_failed_fp``), instead of silently
  shrinking the pool.

### Changed

- The ``native-3d`` descriptor preset gained three columns (``asphericity``,
  ``acylindricity``, ``relative_shape_anisotropy``), so its feature-vector width
  increased. Consumers that hard-coded the column count should regenerate it via
  ``descriptor_feature_names("native-3d")``. ``native-basic`` and ``rdkit-basic``
  are unchanged.

### Fixed

- The streaming SDF reader no longer silently drops a record when a blank line
  pads the gap between ``$$$$`` and the next record's title (a common writer
  quirk).

## [0.11.0] - 2026-05-31

### Added

- MCP server now exposes the docking-hit triage workflow as four tools
  (``dock_summary``, ``dock_diverse``, ``dock_rank``, ``dock_report``), so an AI
  assistant can rank poses, pick a diverse shortlist, consensus-rank across scored
  SDFs, and build an HTML report from a docking-output SDF. They take a literal
  SDF path and write files only when given an output directory (27 tools total).
- Docking-hit triage commands for making sense of a virtual screen's output SDF,
  exposed both on the CLI and as functions in ``molscope.docking``:
  - ``molscope dock-summary results.sdf --score-field minimizedAffinity``: rank
    poses by a score field (auto-detected when omitted), extract name, SMILES,
    pose id, score, heavy-atom count and ligand efficiency, and write
    ``dock_summary.csv``, ``top_hits.csv`` and ``score_distribution.png``. Core
    install only; the SMILES column needs RDKit and is left blank without it.
  - ``molscope dock-diverse results.sdf --top 500 --select 50``: rank, keep the
    best N, Morgan-fingerprint them, Butina-cluster by Tanimoto similarity, and
    keep the best-scoring representative of each cluster so a shortlist is not 50
    near-identical analogues. Writes ``diverse_hits.sdf`` (faithful pose
    re-export) and ``diverse_hits.csv``; reports when fewer clusters exist than
    requested instead of silently returning a short list. Needs RDKit.
  - ``molscope dock-rank vina.sdf gnina.sdf --method consensus``: join hits
    across one or more scored SDFs by name (or SMILES), rank each score field by
    its own direction, and aggregate by mean rank. Optional ligand efficiency and
    MW/logP filters. The output table is transparent: it reports which fields and
    directions were used and states plainly that the consensus rank is a triage
    heuristic, not a calibrated affinity.
  - ``molscope dock-report results.sdf``: assemble a single self-contained
    ``dock_report.html`` (ranked hit table, embedded score histogram, and a grid
    of diverse cluster representatives drawn as 2D depictions) plus a
    ``top_poses.sdf`` for loading into PyMOL, ChimeraX, or Mol*. It is a static
    file you can email or archive, not a server; the depictions need RDKit and
    the section is omitted gracefully without it.
- ``read_sdf_frames``: read every record of a multi-record ``.sdf`` as a list of
  ``Molecule`` objects, keeping each pose's 3D coordinates. This is the common
  output format for docking tools (AutoDock Vina, Gnina, Smina), one record per
  pose with the score in a ``> <tag>`` data field. Each molecule's data fields
  (e.g. ``minimizedAffinity``, ``CNNaffinity``, ``CNNscore``) are captured into a
  new ``Molecule.properties`` dict, so poses can be ranked, filtered, and fed
  straight into the existing descriptor/contact-map/diversity tools without the
  RDKit SMILES round-trip (which discards the 3D pose). Core install only — no
  extras needed. ``read_sdf`` likewise now populates ``properties`` from the
  first record's data fields. Malformed records are skipped rather than aborting
  the whole file.
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

[Unreleased]: https://github.com/roshan2004/molscope/compare/v0.12.0...HEAD
[0.12.0]: https://github.com/roshan2004/molscope/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/roshan2004/molscope/compare/v0.10.0...v0.11.0
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
