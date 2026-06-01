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

[report]: ../api-reference.md
