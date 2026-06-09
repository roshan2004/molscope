# Structure QC: is this structure ML-ready?

Before a PDB or mmCIF file becomes a descriptor table, contact map, or graph, it
is worth asking whether the *coordinates* are trustworthy: are residues missing,
is the chain broken, are there alternate conformations, what is the net charge?
`prepare_structure` reads the file once and answers all of that in a single
[`StructureReport`][report].

```python
import molscope as ms

report = ms.prepare_structure("1ubq.pdb")        # a path or a 4-char PDB id
print(report.summary())
# 1ubq.pdb: ML-ready | 660 atoms | chains A | net charge +0 | warnings: ...
```

## What it checks

| Check | Severity | Needs |
| --- | --- | --- |
| Missing backbone atoms (N, CA, C, O) | **blocker** | core |
| Backbone chain breaks (CA–CA > 4.5 Å between adjacent residues) | **blocker** | core |
| Residue-numbering gaps | warning | core |
| Truncated side chains (fewer heavy atoms than the residue should have) | warning | core |
| Non-standard residues + ligand / water inventory | warning | core |
| Hydrogens present | warning | core |
| Alternate conformations / partial occupancies | warning | core (PDB) |
| Net formal charge at a chosen pH | informational | `chem` (+ `propka` for `"pka"`) |

The topology checks run on the bare NumPy install. The net-charge step reuses the
protonation backends and **degrades gracefully**: if RDKit (or PROPKA) is missing,
or the file cannot be template-parsed, the charge is reported as `None` with an
explanatory note instead of raising.

```python
report = ms.prepare_structure("1ubq.pdb", protonation="pka", ph=7.4)
report.net_charge        # e.g. 0     (PROPKA prediction at pH 7.4)
report.ml_ready          # True
report.blockers          # []  -> nothing that corrupts distance/graph features
report.warnings          # ['57 atom(s) with occupancy < 1', 'no hydrogens present']
report.to_dict()         # JSON-serialisable, for pipelines
print(report.report_markdown())   # a full human-readable report
```

!!! note "`ml_ready` is a heuristic"
    A structure is called *not* ML-ready only for **blockers** — missing backbone
    atoms or internal chain breaks, which corrupt distance- and graph-based
    features. Everything else is surfaced as a warning. Treat the verdict as
    triage, not as a guarantee for a specific modelling task.

## From the command line

```bash
molscope structure-report 1ubq.pdb                  # one-line verdict
molscope structure-report --fetch 1ubq --json       # full JSON report
molscope structure-report 1ubq.pdb --out report.md  # write a Markdown report
molscope structure-report model.pdb --protonation pka --ph 7.4
```

`--fetch <PDBID>` downloads (and caches) the entry from RCSB first. Gzipped PDBs
(`.pdb.gz`, as RCSB serves them) are handled transparently, including for the
net-charge step.

## A faster, format-agnostic check: `quality_report`

`prepare_structure` is protein-shaped: it reasons about backbones, chain breaks,
and protonation. Sometimes you want a cheaper, upstream question answered first:
*did this file parse into something sensible at all?* — and you want it to work on
an `.xyz` or `.sdf` small molecule just as well as on a protein. That is
[`quality_report`][quality] (`molscope qc`).

```python
import molscope as ms

report = ms.quality_report("3ptb.pdb")     # a path, a PDB id, or a Molecule
print(report.summary())
# 3ptb.pdb: clean | 1701 atoms | chains A | 1 ligand(s) | bonds explicit (21)
```

It reads the structure once and reports an inventory plus a few parse-fidelity
signals:

| Field | What it tells you |
| --- | --- |
| `n_atoms`, `chains`, `n_residues` | basic size of what was parsed |
| `ligands`, `n_waters`, `n_ions`, `n_hetero_atoms` | the HETATM split |
| `missing_metadata` | per-atom fields the format did **not** carry (e.g. an XYZ has no chains or residue names) |
| `unknown_elements`, `blank_elements` | atom symbols that are not real elements, or empty — usually a sign of a misread column |
| `bond_source`, `n_bonds` | whether connectivity is **explicit** (from the file, e.g. SDF bonds or PDB `CONECT`) or **inferred** from geometry |
| `altloc_atoms`, `low_occupancy_atoms` | alternate conformations / partial occupancies (PDB) |
| `warnings` | mmCIF validity problems (needs the `cif` extra; degrades to a note without it) |

`report.clean` is `True` when nothing in `report.issues` fired (no atoms lost, no
junk element symbols, no CIF validity failure). Like the report above, it offers
`to_dict()` (JSON) and `report_markdown()`, and the CLI mirrors the same flags:

```bash
molscope qc 3ptb.pdb                 # one-line inventory
molscope qc --fetch 1ubq --json      # full JSON report
molscope qc ligand.sdf --out qc.md   # write a Markdown report
```

!!! tip "Which one do I want?"
    Reach for `molscope qc` as a quick gate on *any* structure file — it is fast,
    needs only NumPy, and surfaces parse problems (lost atoms, bad element
    columns, missing metadata, surprise inference of bonds). Reach for
    `molscope structure-report` when the structure is a protein headed for
    distance- or graph-based ML and you need the topology verdict (backbone
    integrity, chain breaks, net charge).

## One report for everything: `molscope report`

When you want the whole picture in a single shareable file rather than one
verdict at a time, `molscope report` bundles the QC verdicts above with the
chain / ligand inventory, a descriptor table, contact-map statistics and an
embedded heatmap, molecular-graph stats, and an optional coarse-grained preview:

```bash
molscope report 3ptb.pdb --out-dir report/            # writes report/report.html
molscope report --fetch 1ubq --format both            # HTML and Markdown
molscope report 3ptb.pdb --coarse-grain --cg-mapping martini
```

The HTML is self-contained (figures are inline `data:` URIs), so it travels as a
single file. The same data is available programmatically:

```python
import molscope as ms

data = ms.build_report("3ptb.pdb", coarse_grain="residue_com")
ms.report.render_html(data)        # a self-contained HTML string
ms.report.render_markdown(data)    # a Markdown string
```

Sections whose inputs are missing — a residue contact map for a residue-less
`.xyz`, say — are skipped with a note rather than failing.

[report]: ../api-reference.md
[quality]: ../api-reference.md
