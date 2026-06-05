# Does the pocket description help an LLM?

`describe_environment` exists on the premise that a chemistry-aware paragraph is
better LLM/RAG context than raw coordinates. That premise should be *tested*,
not assumed. This page reports a controlled evaluation of whether the prose
actually improves a downstream task, and is reproducible with
`scripts/eval_pocket_prose.py`.

## Design

A **representation ablation** on an objective, automatically scored task. For
each protein-ligand complex the model is shown a description of the **binding
pocket only** (never the bound ligand) plus a multiple-choice list of candidate
ligands written as SMILES, and must pick the ligand the pocket binds. The bound
ligand is the ground truth, so labels are free and the panel scales cheaply.

Only the *pocket representation* changes between arms:

| arm | what the model sees |
| --- | --- |
| `coords` | raw pocket atom coordinates (element + x/y/z) -- the "LLMs cannot read coordinates" baseline |
| `residues` | the bare binding-site residue list with distances, i.e. what `binding_site` already returns |
| `prose` | `describe_environment` output (the feature under test) |
| `features` | the structured `environment().to_dict()` -- the prose's content without the natural language |

The headline question is **prose vs the residue list**. That is a deliberately
*strong* baseline: an LLM already knows "Asp is acidic" from the residue name,
so any gain from prose must come from the geometry-derived complementarity it
adds (which ligand group sits against which pocket group). A null result would
itself be an honest, useful finding.

## Controls

- **Memorisation.** Candidates are SMILES with no names or HET codes, and the
  pocket representations never include the ligand's resname, SMILES, or its own
  atoms (the prose's and feature dict's ligand identifiers are masked to `LIG`).
  The model must reason about chemical complementarity rather than recall which
  ligand a famous protein binds.
- **Trivial size cues.** Decoys are drawn from the nearest-in-size ligands of
  other panel complexes, so the answer cannot be read off heavy-atom count.
- **Determinism.** `temperature=0`, fixed seed, one fixed candidate set per
  complex shared across all arms, and on-disk response caching so the study is
  reproducible and reruns are free.
- **Statistics.** Per-arm top-1 accuracy with 95% bootstrap confidence
  intervals, and an exact McNemar test for the prose-vs-residues contrast on the
  same items.

## Results

Model `gpt-4.1`, 45 protein-ligand complexes, 4 candidates each (chance = 25%),
top-1 accuracy with 95% bootstrap confidence intervals:

| arm | accuracy | 95% CI | correct/total |
| --- | --- | --- | --- |
| `coords` | 0.20 | [0.09, 0.31] | 9/45 |
| `residues` | 0.22 | [0.11, 0.36] | 10/45 |
| **`prose`** | **0.31** | [0.18, 0.44] | 14/45 |
| `features` | 0.24 | [0.13, 0.38] | 11/45 |

Prose vs residue list (McNemar on the same items): prose correct where residues
wrong = **5**; residues correct where prose wrong = **1**; exact two-sided
p = **0.22**.

What this says, read honestly:

- **The prose is the best representation.** It beats the residue list by 9
  points and raw coordinates by 11, and wins the head-to-head 5-to-1. The
  direction is consistent with the feature's premise: the geometry-derived
  complementarity the prose adds (which ligand group sits against which pocket
  group) helps the model pick the matching ligand.
- **Raw coordinates are at chance** (0.20 vs 25%), confirming the "LLMs cannot
  read 3D coordinates" motivation directly.
- **Natural language beat the structured dict** (prose 0.31 vs features 0.24)
  even though they carry the same facts -- a tentative hint that the prose
  framing, not just the content, is doing useful work.
- **It is not yet statistically significant.** At n=45 the McNemar p-value is
  0.22 and the confidence intervals overlap. The honest conclusion is
  *suggestive but underpowered*: the effect points the right way and is
  consistent across the contrast, but a larger panel is needed to confirm it.
  Run `--full` (adds `PANEL_EXTRA`) for more power.

This is a real, not a manufactured, result: a controlled, memorisation-guarded
test shows the description helping, while being candid that one 45-complex task
does not yet settle the question.

## Reproduce

```bash
# Needs OPENAI_API_KEY.
.venv/bin/python scripts/eval_pocket_prose.py --n 8 --model gpt-4o-mini   # quick pilot
.venv/bin/python scripts/eval_pocket_prose.py --model gpt-4.1 --csv eval.csv
```

The harness internals (decoy sampling, representation builders, answer parsing,
statistics) are unit-tested offline with a fake backend in
`tests/test_eval_pocket_prose.py`, so they need neither a key nor network.

## Honest reading

This is a small, single-task study, not a benchmark suite. It asks one focused
question -- does the natural-language pocket description beat the representations
MolScope could already emit? -- on a hard, memorisation-controlled task. Treat
the numbers as evidence about *that* question, with the caveats above, not as a
general claim about LLM structural reasoning.
