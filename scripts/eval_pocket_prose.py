"""Downstream LLM eval: does describe_environment prose help a real task?

This is a controlled *representation ablation*. For each protein-ligand complex
we hand an LLM a description of the **binding pocket only** (never the bound
ligand) plus a multiple-choice list of candidate ligands given as SMILES, and
ask it to pick the ligand the pocket binds. Ground truth is free (the actually
bound ligand). The only thing that varies between arms is how the pocket is
represented:

* ``coords``    -- raw pocket atom coordinates (element + x/y/z). The "LLMs
                   cannot read coordinates" baseline.
* ``residues``  -- the bare binding-site residue list with distances, i.e. what
                   ``binding_site`` already returns. A *strong* baseline: LLMs
                   know "Asp is acidic" from the residue name alone.
* ``prose``     -- ``describe_environment`` output (the feature under test).
* ``features``  -- the structured ``environment().to_dict()`` (prose's content
                   without the natural language), to separate "structure" from
                   "language".

The headline question is whether ``prose`` beats ``residues``. A null result is
itself an honest, publishable finding (it would mean the residue names already
carry the chemistry, and the geometry-derived complementarity adds little).

Memorisation control: candidate ligands are shown as SMILES with **no names or
HET codes**, so the model must reason about chemical complementarity (does this
ligand's charged / aromatic groups match the pocket?) rather than recall which
ligand a famous protein binds. Pocket representations likewise never include the
ligand's resname, SMILES, or its own atoms.

Usage::

    .venv/bin/python scripts/eval_pocket_prose.py --n 8 --model gpt-4o-mini   # quick pilot
    .venv/bin/python scripts/eval_pocket_prose.py --model gpt-4.1 --csv eval.csv

Needs ``OPENAI_API_KEY``. Responses are cached so reruns are free/deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import molscope as ms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"

ARMS = ("coords", "residues", "prose", "features")
POCKET_CUTOFF = 4.5
MAX_POCKET_ATOMS = 220   # token guard for the coords arm

# HETATM codes that are solvent / buffer / cryoprotectant, not the ligand of
# interest. molscope.ligands() already drops water and monatomic ions; this
# extends that to common crystallisation additives.
BUFFERS = frozenset({
    "GOL", "EDO", "PEG", "PGE", "PG4", "SO4", "PO4", "ACT", "DMS", "MES",
    "TRS", "FMT", "EPE", "BME", "IPA", "MPD", "ACY", "CIT", "TLA", "NO3",
    "BCT", "CO3", "NH4", "FLC", "1PE", "P6G", "OGA", "DTT", "GSH",
})

# A deliberately diverse panel of well-studied complexes (varied targets and
# ligand chemistries). The ligand is auto-detected (largest non-buffer HETATM
# group); entries without a usable, SMILES-resolvable ligand are skipped and
# reported, so the list can be generous.
PANEL = (
    "3ptb", "1stp", "1iep", "3ert", "1hsg", "4hvp", "2br1", "1m17", "1unl",
    "1fkf", "2wea", "1bcd", "4dfr", "1aq1", "1ke5", "1pph", "1oyt", "3ntb",
    "2vta", "1df8", "1e66", "1c5z", "1lpz", "1mq6", "1owe", "1s19", "1uwt",
    "1xkk", "1y6a", "2cbv", "2ctc", "2hyy", "2p2i", "2z5x", "3eml", "3g0e",
    "3kgp", "3pp0", "4ag8", "4gid", "5std", "1bzm", "1cbx", "1dwd", "1ezq",
)

# Extend the panel for more statistical power (and re-run) by adding these to
# PANEL. Entries without a usable, SMILES-resolvable ligand are skipped
# automatically. The reported results use the 45-complex PANEL above.
PANEL_EXTRA = (
    "1a4g", "1add", "1apv", "1atl", "1azm", "1bma", "1ckp", "1dhf", "1eve",
    "1f0r", "1f0s", "1fh8", "1gpk", "1h00", "1h1s", "1hnn", "1hvr", "1hwi",
    "1hwr", "1ig3", "1k3u", "1kv1", "1l2s", "1l7f", "1m48", "1mmv", "1n2v",
    "1nhz", "1of6", "1opk", "1oq5", "1p1q", "1pmn", "1q41", "1qbq", "1r55",
    "1r58", "1s3v", "1sj0", "1t40", "1t46", "1tow", "1u1c", "1uou", "1v0p",
    "1ydt", "1yqy", "1z95", "2bm2", "2gss", "3cpa",
)


# -- ligand SMILES (RCSB Chemical Component Dictionary) ----------------------


def _smiles_cache_path() -> Path:
    return Path(tempfile.gettempdir()) / "molscope_ccd_smiles.json"


def _load_smiles_cache() -> dict:
    path = _smiles_cache_path()
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_smiles_cache(cache: dict) -> None:
    _smiles_cache_path().write_text(json.dumps(cache))


def ligand_smiles(het: str, cache: dict) -> str | None:
    """Canonical SMILES for a HET code from the RCSB CCD (cached on disk)."""
    het = het.upper()
    if het in cache:
        return cache[het]
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{het}"
    smiles = None
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.load(resp)
        descriptors = data.get("pdbx_chem_comp_descriptor", []) or []
        for want in ("SMILES_CANONICAL", "SMILES"):
            for desc in descriptors:
                if desc.get("type") == want and desc.get("descriptor"):
                    smiles = desc["descriptor"]
                    break
            if smiles:
                break
    except Exception:
        smiles = None
    cache[het] = smiles
    return smiles


# -- pocket representations (pocket only; never the bound ligand) ------------


def _pocket_protein_atoms(mol, site) -> list[int]:
    atoms = site.protein_atom_indices
    return atoms[:MAX_POCKET_ATOMS]


def repr_coords(mol, site) -> str:
    """Raw pocket atom coordinates (element + x/y/z), ligand excluded."""
    lines = ["Binding-pocket atoms (element, x, y, z in angstrom):"]
    elements = mol.elements
    coords = mol.coords
    for i in _pocket_protein_atoms(mol, site):
        x, y, z = coords[i]
        lines.append(f"{elements[i]:>2s} {x:8.1f} {y:8.1f} {z:8.1f}")
    return "\n".join(lines)


def repr_residues(mol, site) -> str:
    """Bare binding-site residue list with min distances (what binding_site gives)."""
    lines = ["Binding-pocket residues (residue, min distance to ligand in angstrom):"]
    for rec in site.to_records():
        lines.append(f"{rec['residue_id']}  {rec['min_distance']:.1f}")
    return "\n".join(lines)


def _anonymize_ligand(text: str, het: str) -> str:
    """Replace the ligand's HET code with a generic token.

    The prose and feature dict name the ligand (``"ligand BEN"``, ``"A:BEN300"``);
    leaving that in would let the model recall which ligand a known protein binds
    instead of reasoning about complementarity. Pocket chemistry is kept; only
    the identity is masked.
    """
    # Leading boundary only: the code may be glued to a resid ("A:BEN300").
    return re.sub(rf"\b{re.escape(het)}", "LIG", text)


def repr_prose(mol, site) -> str:
    return _anonymize_ligand(site.describe_environment(mol), site.ligand.resname.upper())


def repr_features(mol, site) -> str:
    text = json.dumps(site.environment(mol).to_dict(), indent=1)
    return _anonymize_ligand(text, site.ligand.resname.upper())


REPR_BUILDERS = {
    "coords": repr_coords,
    "residues": repr_residues,
    "prose": repr_prose,
    "features": repr_features,
}


# -- task construction ------------------------------------------------------


@dataclass
class Complex:
    pdb_id: str
    ligand_het: str
    smiles: str
    n_heavy: int
    mol: object = field(repr=False)
    site: object = field(repr=False)


@dataclass
class Question:
    pdb_id: str
    options: list[tuple[str, str]]   # [(letter, smiles), ...]
    correct: str                     # correct letter


def build_question(complex_: Complex, pool: list[Complex], k: int, rng: random.Random) -> Question:
    """One MCQ: the true ligand + size-matched decoys from other complexes."""
    others = [c for c in pool if c.ligand_het != complex_.ligand_het]
    others.sort(key=lambda c: abs(c.n_heavy - complex_.n_heavy))
    # Draw decoys from the nearest-in-size half so the choice is not a size cue.
    window = others[: max(k * 3, 12)]
    decoys = rng.sample(window, min(k - 1, len(window)))
    smiles = [complex_.smiles] + [d.smiles for d in decoys]
    rng.shuffle(smiles)
    letters = [chr(ord("A") + i) for i in range(len(smiles))]
    options = list(zip(letters, smiles))
    correct = next(letter for letter, smi in options if smi == complex_.smiles)
    return Question(complex_.pdb_id, options, correct)


PROMPT_SYSTEM = (
    "You are a structural biology expert. You are shown a description of a "
    "protein binding pocket (the protein side only) and a list of candidate "
    "ligands written as SMILES. Decide which single candidate the pocket is "
    "most likely to bind, reasoning from chemical complementarity: charge "
    "pairing (e.g. an acidic pocket residue favours a basic/cationic ligand "
    "group), hydrogen-bonding, hydrophobic and aromatic contacts, and size. "
    "Answer with ONLY the letter of your choice."
)


def build_prompt(repr_text: str, question: Question) -> str:
    lines = [repr_text, "", "Candidate ligands (SMILES):"]
    for letter, smi in question.options:
        lines.append(f"{letter}. {smi}")
    lines.append("")
    lines.append("Which candidate does this pocket bind? Answer with one letter only.")
    return "\n".join(lines)


# -- LLM backend (OpenAI) with on-disk response cache -----------------------


class OpenAIBackend:
    def __init__(self, model: str, cache_path: Path):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI()
        self.cache_path = cache_path
        self.cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def _key(self, system: str, user: str) -> str:
        h = hashlib.sha256(f"{self.model}\x00{system}\x00{user}".encode()).hexdigest()
        return h

    def choose(self, system: str, user: str) -> str:
        key = self._key(system, user)
        if key in self.cache:
            return self.cache[key]
        text = self._call_with_backoff(system, user)
        self.cache[key] = text
        self.cache_path.write_text(json.dumps(self.cache))
        return text

    def _call_with_backoff(self, system: str, user: str, retries: int = 8) -> str:
        import time

        from openai import APIError, RateLimitError

        delay = 2.0
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    max_tokens=5, temperature=0, seed=0,
                )
                return resp.choices[0].message.content or ""
            except (RateLimitError, APIError):
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")


def parse_letter(text: str, n_options: int) -> str | None:
    valid = {chr(ord("A") + i) for i in range(n_options)}
    up = text.strip().upper()
    # Prefer a standalone letter token (handles "C", "C.", "The answer is C").
    for match in re.finditer(r"\b([A-Z])\b", up):
        if match.group(1) in valid:
            return match.group(1)
    # Fall back to the first valid character anywhere (e.g. glued like "BC").
    for ch in up:
        if ch in valid:
            return ch
    return None


# -- panel loading ----------------------------------------------------------


def _pick_ligand(mol):
    candidates = [lig for lig in mol.ligands() if lig.resname.upper() not in BUFFERS]
    if not candidates:
        return None
    return max(candidates, key=len)


def load_panel(pdb_ids, smiles_cache, *, min_heavy=6, max_heavy=70, log=print):
    complexes = []
    for pdb_id in pdb_ids:
        try:
            path = (DATA / f"{pdb_id}.pdb")
            path = str(path) if path.exists() else ms.io.fetch_file(pdb_id)
            mol = ms.read(path)
        except Exception as exc:
            log(f"  skip {pdb_id}: load failed ({exc})")
            continue
        ligand = _pick_ligand(mol)
        if ligand is None:
            log(f"  skip {pdb_id}: no non-buffer ligand")
            continue
        n_heavy = sum(1 for i in ligand.atom_indices if mol.elements[i].upper() != "H")
        if not (min_heavy <= n_heavy <= max_heavy):
            log(f"  skip {pdb_id}: ligand {ligand.resname} size {n_heavy} out of range")
            continue
        smiles = ligand_smiles(ligand.resname, smiles_cache)
        if not smiles:
            log(f"  skip {pdb_id}: no SMILES for {ligand.resname}")
            continue
        site = mol.select_pocket(ligand=ligand, cutoff=POCKET_CUTOFF).site
        complexes.append(Complex(pdb_id, ligand.resname.upper(), smiles, n_heavy, mol, site))
    _save_smiles_cache(smiles_cache)
    return complexes


# -- statistics -------------------------------------------------------------


def bootstrap_ci(correct_flags: list[int], rng: random.Random, n_boot=2000):
    n = len(correct_flags)
    if n == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n_boot):
        sample = [correct_flags[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def mcnemar(a_flags: list[int], b_flags: list[int]) -> dict:
    """McNemar discordance between two arms on the same items (a vs b)."""
    b_only = sum(1 for a, b in zip(a_flags, b_flags) if b and not a)   # b right, a wrong
    a_only = sum(1 for a, b in zip(a_flags, b_flags) if a and not b)   # a right, b wrong
    n = b_only + a_only
    # Exact two-sided binomial p-value under p=0.5 (no scipy dependency).
    from math import comb

    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {"a_only": a_only, "b_only": b_only, "p_value": p}


# -- run --------------------------------------------------------------------


def evaluate(complexes, *, arms, model, k, seed, cache_path, backend=None, log=print):
    if backend is None:
        backend = OpenAIBackend(model, cache_path)
    rng = random.Random(seed)
    # One fixed question (candidate set) per complex, shared across arms.
    questions = {c.pdb_id: build_question(c, complexes, k, rng) for c in complexes}

    results = {arm: [] for arm in arms}       # per-arm list of (pdb_id, correct_flag)
    for c in complexes:
        q = questions[c.pdb_id]
        for arm in arms:
            repr_text = REPR_BUILDERS[arm](c.mol, c.site)
            prompt = build_prompt(repr_text, q)
            answer = backend.choose(PROMPT_SYSTEM, prompt)
            letter = parse_letter(answer, len(q.options))
            results[arm].append((c.pdb_id, int(letter == q.correct)))
        log(f"  {c.pdb_id} ({c.ligand_het}): "
            + " ".join(f"{arm}={'Y' if results[arm][-1][1] else '.'}" for arm in arms))
    return results, questions


def format_report(results, *, model, k, n) -> str:
    rng = random.Random(12345)
    chance = 1.0 / k
    lines = [
        "# Does describe_environment prose help an LLM? (pocket -> ligand matching)",
        "",
        f"Model: `{model}` | complexes: {n} | choices: {k} (chance = {chance:.0%}) | "
        "top-1 accuracy, 95% bootstrap CI.",
        "",
        "| arm | accuracy | 95% CI | correct/total |",
        "| --- | --- | --- | --- |",
    ]
    flags = {arm: [f for _, f in rows] for arm, rows in results.items()}
    for arm in results:
        f = flags[arm]
        acc = sum(f) / len(f) if f else float("nan")
        lo, hi = bootstrap_ci(f, rng)
        lines.append(f"| {arm} | {acc:.2f} | [{lo:.2f}, {hi:.2f}] | {sum(f)}/{len(f)} |")

    if "prose" in flags and "residues" in flags:
        mc = mcnemar(flags["residues"], flags["prose"])
        lines += [
            "",
            "## Headline contrast: prose vs residue list (McNemar, same items)",
            "",
            f"- prose correct where residues wrong: **{mc['b_only']}**",
            f"- residues correct where prose wrong: **{mc['a_only']}**",
            f"- exact two-sided p-value: **{mc['p_value']:.3f}**",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--k", type=int, default=4, help="number of candidate ligands")
    parser.add_argument("--n", type=int, default=None, help="limit number of complexes")
    parser.add_argument("--full", action="store_true",
                        help="add PANEL_EXTRA for more statistical power (more API calls)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--cache", type=Path,
                        default=Path(tempfile.gettempdir()) / "molscope_llm_eval_cache.json")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    print("Loading panel...")
    panel = PANEL + (PANEL_EXTRA if args.full else ())
    complexes = load_panel(panel, _load_smiles_cache())
    if args.n:
        complexes = complexes[: args.n]
    print(f"Usable complexes: {len(complexes)}")

    print(f"Querying {args.model} ({len(complexes)} complexes x {len(arms)} arms)...")
    results, _ = evaluate(
        complexes, arms=arms, model=args.model, k=args.k,
        seed=args.seed, cache_path=args.cache,
    )

    report = format_report(results, model=args.model, k=args.k, n=len(complexes))
    print("\n" + report)

    if args.csv:
        import csv

        with args.csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["arm", "pdb_id", "correct"])
            for arm, rows in results.items():
                for pdb_id, flag in rows:
                    writer.writerow([arm, pdb_id, flag])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
