# Docking-hit triage: summarise, diversify, rank

After a virtual screen, the bottleneck is rarely running the docking; it is
making sense of the thousands of poses that come back. MolScope reads the
multi-record SDF that AutoDock Vina, Gnina, and Smina write (one record per pose,
the score in a `> <tag>` data field) and turns it into the three things you
usually need next: a ranked summary, a diverse shortlist, and a transparent
consensus ranking across scoring functions.

These live both on the CLI and as functions in
[`molscope.docking`](../api-reference.md). The readers and the summary plot need
only the base install; SMILES perception, fingerprints, clustering, and MW/logP
need RDKit (`pip install "molscope[chem]"`) and fail with a clear message when it
is missing.

## `dock-summary`: rank and inspect

```bash
molscope dock-summary results.sdf --score-field minimizedAffinity
```

Writes three files to the output directory (`--out-dir`, default current):

- `dock_summary.csv`: every pose, ranked, with `pose_id`, `name`, `smiles`,
  `score`, `ligand_efficiency`, and `n_heavy_atoms`.
- `top_hits.csv`: the best `--top` rows (default 10).
- `score_distribution.png`: a histogram of the scores.

The score field is auto-detected when `--score-field` is omitted and a known
docking tag (`minimizedAffinity`, `CNNscore`, `CNNaffinity`, …) is present;
otherwise the error lists the fields it did find. Direction is inferred from the
field name (affinities are lower-is-better, Gnina CNN outputs higher-is-better);
override with `--higher-is-better` / `--lower-is-better`. Ligand efficiency is
the score per heavy atom, signed so a more favourable hit scores higher.

## `dock-diverse`: a shortlist that is not 50 analogues

```bash
molscope dock-diverse results.sdf --top 500 --select 50 --threshold 0.7
```

Ranks the poses, keeps the best `--top`, computes Morgan fingerprints, clusters
them with Butina at a Tanimoto similarity `--threshold`, and returns the
best-scoring member of each cluster, ordered by score and limited to `--select`.
Picking one representative per cluster is what stops a top-50 list from
collapsing into many near-identical analogues of the same scaffold.

Outputs `diverse_hits.sdf` (the selected poses, re-emitted faithfully from the
original records) and `diverse_hits.csv` (with `cluster_id` and `cluster_size`,
so you can see how big a family each representative stands in for). If the
chemistry yields fewer clusters than you asked for, every representative is
returned and the command says so rather than silently handing back a short list.

Because the selected `.sdf` keeps the 3D pose, it flows straight back into the
rest of MolScope: descriptors, contact maps, or graph export for an ML model.

## `dock-rank`: transparent consensus across scoring functions

```bash
molscope dock-rank vina.sdf gnina.sdf --method consensus
```

Joins the hits across the input files by molecule name (`--key smiles` to join on
perceived SMILES instead), ranks each score field by its own direction, and
aggregates by **mean rank**. Rank aggregation is scale-free and explainable: it
does not pretend the score fields share units or that the blended number is a
calibrated affinity. The CSV keeps every input score and its per-field rank
alongside the `consensus_rank` and `final_rank`, and the command prints exactly
which fields and directions it used:

```text
score fields used (direction):
  - vina:minimizedAffinity: lower=better
  - gnina:CNNscore: higher=better
note: consensus rank is mean rank across these fields, a transparent triage
heuristic, not a calibrated 'true' affinity
```

Pass `--score-fields` to choose the fields explicitly, `--higher-is-better` /
`--lower-is-better` to set directions for unknown fields, and `--mw-max` /
`--logp-max` to drop hits outside a property window (these need RDKit). Unknown
field directions are assumed lower-is-better and flagged `[assumed]` so the
choice is never hidden.

## `dock-report`: one HTML file to look at

```bash
molscope dock-report results.sdf --score-field minimizedAffinity
```

Writes a single self-contained `dock_report.html` that bundles the ranked hit
table, an embedded score histogram, and a grid of diverse cluster
representatives drawn as **2D depictions** — the view a CSV of SMILES cannot give
you. Everything is inlined (base64 image, inline SVG), so the file is one
artifact you can email, archive in a repo, or attach to a paper; it is not a
server you have to keep running.

Alongside it, `top_poses.sdf` holds the best `--export-poses` poses (default 20)
ready to load straight into PyMOL, ChimeraX, or Mol* for 3D inspection. Use
`--top` for the table length, `--select` and `--threshold` for the clustering,
and `--no-clusters` to skip the depiction grid. The 2D depictions need RDKit; the
report still builds without it, just without that section.

For live, interactive filtering, stay in a notebook and call the
[`molscope.docking`](../api-reference.md) functions directly: `summarize` and
`select_diverse_hits` return plain row dicts that drop straight into a pandas
DataFrame. That keeps inspection notebook-native rather than pulling in a
web-app framework.

## What this is not

MolScope does not dock, re-score, or minimise. It reads what your docking tool
produced and helps you triage it. Consensus ranking is a heuristic for ordering a
shortlist, not a replacement for careful per-target validation or a calibrated
scoring function.
