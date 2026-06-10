# Changelog

All notable changes to MolScope are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the package is pre-1.0, minor versions may include backwards-incompatible
API changes; these are called out under **Changed** where they occur.

## [Unreleased]

### Added

- ``molscope report``: a one-command structure report that bundles the headline
  outputs of MolScope's existing analyses into a single self-contained file —
  the parse-quality and ML-readiness QC verdicts, the chain / ligand inventory,
  a descriptor table, contact-map statistics with an embedded heatmap, molecular
  graph stats, and an optional coarse-grained preview. Writes HTML (default),
  Markdown, or both (``--format``); embedded figures are inline ``data:`` URIs,
  so the HTML is a single portable file. Sections whose inputs are missing (e.g.
  a residue contact map for a residue-less ``.xyz``) are skipped with a note.
  Also exposed as the ``ms.build_report(...)`` API returning a
  ``StructureReportData``.
- ``molscope compare a.pdb b.pdb``: a static-structure comparison reporting the
  aligned (Kabsch) RMSD, per-residue deviations, a residue contact-map delta
  (contacts gained vs lost), and a per-feature descriptor delta. Atoms are
  matched by ``(chain, resid, insertion code, atom name)`` so it works on two
  *different* files — different atom counts, ordering, or point mutations — with
  ``--atoms all|ca|backbone`` choosing the matched/superposed set; structures
  without residue metadata fall back to index matching. Prints a summary, with
  ``--json`` for the full structured result and ``--out`` for a Markdown report.
  Also exposed as ``ms.compare_structures(...)`` returning a ``ComparisonResult``.
- ``molscope preflight`` and ``ms.preflight(...)``: opt-in guardrails that warn,
  *before* a descriptor / graph / coarse-grain run, about inputs that silently
  degrade the result — bonds inferred from geometry rather than the file, missing
  residue metadata, alternate locations / partial occupancy, absent hydrogens,
  unrecognised element symbols, and atom counts large enough to make atom-level
  dense distance work allocate gigabytes. Warnings are scoped per workflow and
  composed from the existing ``quality_report`` (and, with ``deep=True``,
  ``prepare_structure``) signals. Available as a standalone command, a
  ``--preflight`` flag on ``analyze`` / ``export`` / ``coarse-grain``, and a
  ``preflight=True`` argument to ``Molecule.to_graph`` / ``descriptors`` /
  ``coarse_grain``; returns a ``PreflightReport``.

## [0.16.0] - 2026-06-08

### Changed

- **Breaking:** the ``native-3d`` descriptor preset no longer emits the six
  absolute-coordinate columns ``centroid_x/y/z`` and ``center_of_mass_x/y/z``
  (preset width drops from 80 to 74). These recorded where a structure happened
  to sit in its input file's coordinate frame, so they were translation- and
  rotation-variant: two identical molecules placed differently produced different
  features, letting a model learn the arbitrary input frame. Size and shape are
  already captured frame-independently by ``radius_of_gyration``, ``dim_x/y/z``,
  ``principal_moments`` and the gyration-tensor shape scalars. The ``native-basic``
  and ``rdkit-basic`` presets are unaffected.

## [0.15.0] - 2026-06-07

### Added

- ``ms.standardize_features(X, train_index)`` / ``ms.FeatureScaler``: train-only
  standardisation for a :func:`featurize_many` descriptor matrix, the feature-side
  companion to ``TargetScaler`` (which only covered labels). Fits per-column mean
  and standard deviation on the train rows alone and transforms every row, so the
  validation/test feature distribution never leaks into training — the same
  correctness detail ``GraphDataset.standardize_targets`` enforces on the label
  side, previously left to the user on the feature side. ``train_index`` is any
  iterable of row indices (a ``SplitResult.train``, scikit-learn indices, or a
  hand-built list). Near-constant columns get ``std = 1`` so a differing test row
  cannot blow up; ``FeatureScaler.inverse_transform`` maps values back to units.

- ``molscope presets`` / ``ms.list_presets(category=None)``: a discoverable
  catalogue of every feature and mapping preset (descriptor, graph node/edge,
  residue-graph node/edge, and coarse-grain bead mappings). Each entry carries a
  short description, the APIs / CLI flags that accept it, and the exact feature
  names it expands to — sourced from the canonical ``*_feature_names`` functions
  so the catalogue can never drift from what the presets produce. The CLI groups
  them by kind and takes an optional category (``molscope presets graph``),
  ``--features`` to print the full name lists, and ``--json``. Enumerating names
  needs no optional backends (even ``rdkit-basic``). Also exposes
  ``coarsegrain.COARSE_GRAIN_MAPPINGS`` as the single source of truth for the
  built-in bead mappings.

- ``molscope qc`` / ``ms.quality_report(source)``: a lightweight, format-agnostic
  structure-quality report to run *before* analysis. Reads a ``.pdb`` / ``.cif`` /
  ``.xyz`` / ``.sdf`` file (or an in-memory ``Molecule``) once and inventories
  what is in it and whether it parsed cleanly — atom / chain / residue counts,
  the ligand / water / ion split, which per-atom metadata the format carried,
  blank or non-element atom symbols, whether bonds are explicit (from the file)
  or geometry-inferred, alternate-location / partial-occupancy atoms (PDB), and
  mmCIF validity warnings (when gemmi is available, degrading to a note when not).
  Returns a ``QualityReport`` with ``.summary()``, ``.to_dict()`` (JSON), and
  ``.report_markdown()``; the CLI mirrors ``structure-report`` with ``--json``
  and ``--out``. This is the upstream, parse-fidelity complement to
  ``prepare_structure`` (which answers the heavier "is this protein ML-ready?").
  Adds the full IUPAC element table as ``elements.ELEMENT_SYMBOLS`` /
  ``elements.is_element`` to tell real symbols from parse junk.

- ``mol.select_pocket(ligand=..., cutoff=...).describe_environment()``: translate
  a 3D binding pocket into a chemistry-aware, biochemist-style natural-language
  paragraph for LLM / RAG prompt context. Pure-NumPy geometric heuristics detect
  the hydrophobic wall, aromatic residues, hydrogen bonds, and salt-bridge /
  electrostatic contacts (with atom names and distances). ``select_pocket``
  returns a ``Pocket`` (a ``BindingSite`` bound to its molecule); the structured
  findings are also available via ``.environment()`` (a ``PocketEnvironment``
  with ``to_dict()``) and ``BindingSite.describe_environment(mol)``. Exposed over
  the MCP server as the ``describe_environment`` tool, which returns both the
  ``prompt`` text and the structured ``features``.

### Validation

- Downstream LLM utility eval for ``describe_environment``: a controlled,
  memorisation-guarded representation ablation on a pocket→ligand matching task
  (``scripts/eval_pocket_prose.py``, offline-tested in
  ``tests/test_eval_pocket_prose.py``, documented in ``docs/llm-eval.md``). With
  ``gpt-4.1`` over the full 96-complex panel (``--full``, chance 25%), the prose
  is the best pocket representation (accuracy 0.47) ahead of the bare residue
  list (0.34), the structured feature dict (0.46), and raw coordinates (0.32);
  prose beats the residue list 18-to-6 in the head-to-head, a statistically
  significant margin on the same-item McNemar test (p=0.023). Honest,
  reproducible evidence that the description significantly helps a downstream
  task. (An earlier 45-complex run pointed the same way at p=0.22; doubling the
  panel confirmed the effect.)

- Characterise the ``describe_environment`` interaction heuristics against PLIP
  (Adasme et al., *NAR* 2021) at residue granularity across the seven-complex
  panel. New Tier-2 reference test
  (``tests/validation/test_pocket_interactions_ref.py``) and a reproducible
  report harness (``scripts/validate_pocket_interactions.py``). Headline
  polar-contact union (H-bond ∪ salt bridge): precision 0.82, recall 0.97
  (F1 0.89); hydrophobic R 0.88; aromatic/pi is a permissive presence flag
  (P 0.07). The test skips cleanly when PLIP (conda-only) is absent. Results and
  method are documented in ``docs/validation.md``.

### Documentation

- Surface the graph-ML dataset on-ramp on the front pages: the README, docs
  landing page, and quickstart now show ``build_dataset`` → ``ds.loader()`` (with
  ``cache_dir``, ``standardize_targets``, and ``fetch_dataset``), not just
  single-graph export.

## [0.14.0] - 2026-06-04

### Added

- ``fetch_dataset(ids, labels=..., ...)``: build a dataset directly from RCSB
  accessions. The adapter for a published *accession + label* table — it
  downloads each PDB id (cached under ``root`` so reruns do not re-download) and
  hands the files to ``build_dataset`` (so the featurisation cache, splits, and
  feature presets all apply via ``**build_kwargs``). Accessions are
  case-insensitive; a failed download is recorded in ``ds.skipped`` and skipped
  unless ``on_error="raise"``. No curated label tables are bundled — callers
  bring the benchmark's own labels. ``TargetScaler`` is now also exported at the
  top level.

- ``GraphDataset.standardize_targets()`` and ``TargetScaler``: fit a target mean
  and standard deviation on the **train** split only, rewrite every labelled
  graph's ``data.y`` into standardised space, and return a ``TargetScaler`` whose
  ``inverse_transform`` maps model outputs back to physical units. Keeps
  validation and test out of the normalisation (the leakage mistake that inflates
  scores); ``ds.labels`` is left in original units. Requires ``fmt="pyg"`` with a
  built split and at least one labelled train graph. The
  ``examples/pdb_to_pyg_ml.py`` walkthrough now uses it in place of an inline
  normalisation loop.

- ``build_dataset(cache_dir=...)``: an on-disk featurisation cache. Each
  file-based structure's graph is stored under a key derived from the file's
  *content* and the featurisation options (``fmt``, feature presets, ``pe``,
  ``self_loops``, ...), so a second call reuses the saved graphs and
  re-featurises only inputs that are new or whose content or options changed.
  ``labels`` and ``split`` are applied after loading and are not part of the key,
  so re-labelling or re-splitting a cached set is free. In-memory ``Molecule``
  sources are not cached (no stable on-disk identity) and a corrupt/partial entry
  is transparently recomputed. The directory is created if missing.

- ``GraphDataset.loader()``: the batching ``DataLoader`` step between a built
  dataset and a training loop. For ``fmt="pyg"`` it returns a
  ``torch_geometric.loader.DataLoader`` and for ``fmt="dgl"`` a
  ``dgl.dataloading.GraphDataLoader``, collating the per-graph objects into
  mini-batches. ``loader("train"|"val"|"test")`` draws from a built split (or the
  whole dataset when called with no argument); the train split shuffles each
  epoch by default and the others do not, overridable via ``shuffle=``, with
  ``batch_size`` and any extra loader keywords (``num_workers``, ``drop_last``,
  ...) forwarded through. ``networkx``/``raw`` formats raise a clear error since
  they have no batching loader. No new core dependency. The
  ``examples/pdb_to_pyg_ml.py`` example and its docs page are rewritten around
  this on-ramp (``build_dataset`` → ``ds.loader()`` → a trained GCN, with
  train-only target standardisation), and a CI-gated smoke test runs it end to
  end so the tutorial cannot silently bit-rot.

- Optional coarse interaction labels on residue-contact-graph edges.
  ``to_residue_contact_graph(annotate_interactions=True)`` tags each contact edge
  with one mutually-exclusive label, computed from residue chemistry and
  atom-level geometry of the source structure (by precedence, most specific
  first): ``disulfide`` (CYS SG–SG < 2.5 Å), ``ligand`` (a non-standard, non-water
  residue), ``salt_bridge`` (basic↔acidic charged side-chain atoms < 4 Å),
  ``covalent`` (sequence-adjacent same-chain neighbours), ``hydrophobic`` and
  ``polar`` (both residues of that class with side chains in contact), and
  ``proximity`` (the fallback, including water contacts). Surfaced as
  ``ResidueContactGraph.edge_interactions``, a one-hot ``interaction_one_hot()``
  categorical edge feature, and a NetworkX ``interaction`` edge attribute; the
  vocabulary is ``molscope.RESIDUE_INTERACTION_LABELS``. These are
  geometric/contact heuristics, **not** binding-energy terms, and need per-atom
  names to resolve the chemistry (degrading to covalent/ligand/proximity
  otherwise). Pure NumPy, no new dependencies; off by default.

## [0.13.0] - 2026-06-03

### Added

- ``relative_sasa`` (and ``Molecule.relative_sasa``): per-residue relative solvent
  accessibility (RSA) with an exposed/buried call. Absolute residue SASA is
  divided by a reference maximum (Tien et al. 2013, added as ``elements.MAX_ASA``
  / ``elements.max_asa``) and classified ``rsa >= threshold`` (default 0.20) as
  exposed. Returns a ``ResidueExposure`` (resids, chains, resnames, absolute
  ``sasa``, ``rsa``, ``exposed``, threshold) in residue order. RSA may slightly
  exceed 1 (extended-tripeptide reference); residues with no reference (ligands,
  waters, non-standard names) get ``NaN`` RSA and are not exposed; SASA is
  computed on the whole structure so burial reflects neighbours. A high-signal
  per-residue feature for interface/binding-site work and residue graphs. Pure
  NumPy, no new dependencies; not folded into the fixed descriptor presets.

- ``plot_ramachandran`` (and ``Molecule.plot_ramachandran``): a Ramachandran plot
  of a protein's backbone phi/psi torsions on a ``[-180, 180]`` grid, coloured by
  simplified-DSSP secondary-structure class (``color_by="ss"`` default; pass a
  Matplotlib colour or ``None`` for a single colour). ``regions=True`` shades
  schematic right-handed-alpha / beta / left-handed-alpha guide boxes (approximate
  teaching aids, not statistically-derived density contours). Reuses the existing
  ``backbone_torsions`` and ``SS_COLORS``; residues with undefined phi/psi (chain
  ends, breaks) are skipped, and non-protein input raises a clear error. A
  ``render_ramachandran`` MCP tool accompanies it. Zero new dependencies.

- ``write_cg_itp`` (and ``coarsegrain.write_itp``): export a coarse-grained model
  as a rough GROMACS ``.itp`` topology skeleton — ``[moleculetype]``, ``[atoms]``
  (one ``CG_<resname>_<bead>`` row per bead with residue assignment, zero charge
  and bead mass), ``[bonds]`` from the bead network, and ``[angles]`` enumerated
  from it (every ``i-j-k`` where ``i-j`` and ``j-k`` are bonds). It is a
  prototyping skeleton, **not** a validated force field: no force constants,
  reference values or non-bonded parameters. Complements the existing
  ``write_cg_openmm_xml`` for GROMACS users; pure Python, no new dependencies.
  (The proposed JSON mapping export and OpenMM XML helper already shipped as
  ``write_cg_mapping`` and ``write_cg_openmm_xml``.)

- Edge attributes for 3D GNN workflows on the atom graph exporters, no new
  dependencies. ``MolecularGraph.to_pyg_data``/``to_dgl_graph`` now attach an
  ``is_covalent`` edge flag (and a ``covalent`` edge attribute on
  ``to_networkx``): all-``True`` for a bond-derived graph, but for a spatial
  graph (``knn``/``radius``/``delaunay``) it marks which contacts coincide with
  real covalent bonds, so a model can distinguish chemical bonds from proximity
  contacts. A new ``include_displacement=True`` option adds a directed
  relative-displacement vector ``edge_vec = pos[dst] - pos[src]`` per edge (for
  SchNet/EGNN-style models), computed direction-correctly so the reverse edge
  holds its negation. (``pos`` and ``edge_index`` are exported regardless, so
  equivariant models can derive ``r_ij`` themselves; the option just precomputes
  it.) ``MolecularGraph`` gains a ``covalent_edges`` field and a
  ``covalent_edge_flags()`` accessor.

- ``write_frames(frames, path)``: the write-side counterpart to the multi-frame
  readers and ``stream`` — write a list **or generator** of molecules to a
  multi-frame file, consuming frames one at a time so memory stays O(1). The
  format follows the extension: ``.pdb``/``.ent`` as ``MODEL``/``ENDMDL`` blocks,
  ``.xyz`` as concatenated frames, ``.sdf``/``.mol`` as ``$$$$``-delimited V2000
  records (each keeping its own bonds/charges/properties). Returns the frame
  count. Frames need not share an atom count (varied SDF records are fine);
  multi-frame PDB omits ``CONECT`` (per-model serials make a single global record
  ambiguous, so bonds are re-inferred on read — use ``.sdf`` for per-frame bonds);
  mmCIF is unsupported (no multi-frame form). Pure Python/NumPy, no new
  dependencies; lets you filter/slice/align an ensemble or trajectory-lite stream
  and save the subset.

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

[Unreleased]: https://github.com/roshan2004/molscope/compare/v0.16.0...HEAD
[0.16.0]: https://github.com/roshan2004/molscope/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/roshan2004/molscope/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/roshan2004/molscope/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/roshan2004/molscope/compare/v0.12.0...v0.13.0
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
