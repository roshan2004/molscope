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

Model `gpt-4.1`, 96 protein-ligand complexes (the full panel, `--full`), 4
candidates each (chance = 25%), top-1 accuracy with 95% bootstrap confidence
intervals:

| arm | accuracy | 95% CI | correct/total |
| --- | --- | --- | --- |
| `coords` | 0.32 | [0.23, 0.42] | 31/96 |
| `residues` | 0.34 | [0.25, 0.44] | 33/96 |
| **`prose`** | **0.47** | [0.38, 0.57] | 45/96 |
| `features` | 0.46 | [0.35, 0.56] | 44/96 |

Prose vs residue list (McNemar on the same items): prose correct where residues
wrong = **18**; residues correct where prose wrong = **6**; exact two-sided
p = **0.023**.

What this says, read honestly:

- **The prose is the best representation, significantly so.** It beats the
  residue list by 13 points and raw coordinates by 15, and wins the head-to-head
  18-to-6. On the same-item McNemar test that contrast is significant
  (p = 0.023). The direction matches the feature's premise: the geometry-derived
  complementarity the prose adds (which ligand group sits against which pocket
  group) helps the model pick the matching ligand.
- **The residue list and raw coordinates land together near 0.33**, a third of
  the panel, well below prose. Coordinates are no longer at chance on the larger
  panel, but they carry no advantage over the residue names.
- **Natural language ties the structured dict** (prose 0.47 vs features 0.46):
  on the full panel the gain is in the *content* the description surfaces, not
  the prose framing per se. The two formats of the same geometry-derived facts
  perform equivalently, and both clearly beat the residue list.

This is a real, not a manufactured, result: a controlled, memorisation-guarded
test on a 96-complex panel shows the description helping by a statistically
significant margin over the representations MolScope could already emit.

An earlier underpowered run on the 45-complex base panel pointed the same way
(prose 0.31 vs residues 0.22) but did not reach significance (McNemar p = 0.22);
doubling the panel with `--full` confirmed the effect.

## Reproduce

```bash
# Needs OPENAI_API_KEY.
.venv/bin/python scripts/eval_pocket_prose.py --n 8 --model gpt-4o-mini          # quick pilot
.venv/bin/python scripts/eval_pocket_prose.py --model gpt-4.1 --full --csv eval.csv  # full 96-complex panel (the reported result)
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
