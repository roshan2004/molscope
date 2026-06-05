"""Deterministic unit tests for the LLM pocket-prose eval harness.

These exercise the harness internals -- decoy sampling, representation
builders, answer parsing, and the statistics -- with a *fake* backend, so they
run offline with no OpenAI key and no network. The actual LLM study lives in
``scripts/eval_pocket_prose.py`` and is run manually.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

import molscope as ms

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

ev = pytest.importorskip("eval_pocket_prose")


def _pocket_fixture():
    """A small synthetic PHE/ASP/SER pocket around a BEN-like ligand."""
    mol = ms.Molecule(
        np.array([
            [0.0, 0.0, 0.0], [3.5, 0.0, 0.0],       # PHE70 CA, CG
            [0.0, 5.0, 0.0], [3.1, 5.0, 0.0],       # ASP189 CA, OD2
            [0.0, -5.0, 0.0], [2.9, -5.0, 0.0],     # SER95 CA, OG
            [7.0, 0.0, 0.0], [6.2, 5.0, 0.0], [5.8, -5.0, 0.0],  # ligand C1, N1, O2
        ]),
        elements=["C", "C", "C", "O", "C", "O", "C", "N", "O"],
        atom_names=["CA", "CG", "CA", "OD2", "CA", "OG", "C1", "N1", "O2"],
        resnames=["PHE", "PHE", "ASP", "ASP", "SER", "SER", "BEN", "BEN", "BEN"],
        resids=np.array([70, 70, 189, 189, 95, 95, 300, 300, 300]),
        chains=["A"] * 9,
        hetero=[False] * 6 + [True] * 3,
    )
    site = mol.select_pocket(ligand="BEN", cutoff=4.5).site
    return mol, site


def _fake_complex(pdb_id, het, smiles, n_heavy):
    mol, site = _pocket_fixture()
    return ev.Complex(pdb_id, het, smiles, n_heavy, mol, site)


# -- decoy sampling ---------------------------------------------------------


def test_build_question_includes_truth_and_k_options():
    pool = [
        _fake_complex("aaaa", "AAA", "CCO", 3),
        _fake_complex("bbbb", "BBB", "c1ccccc1", 6),
        _fake_complex("cccc", "CCC", "CC(=O)O", 4),
        _fake_complex("dddd", "DDD", "CCN", 3),
        _fake_complex("eeee", "EEE", "CCCC", 4),
    ]
    rng = random.Random(0)
    q = ev.build_question(pool[0], pool, k=4, rng=rng)

    assert len(q.options) == 4
    smiles = [smi for _, smi in q.options]
    assert pool[0].smiles in smiles                      # truth is present
    assert len(set(smiles)) == 4                         # no duplicate candidates
    correct_smi = dict(q.options)[q.correct]
    assert correct_smi == pool[0].smiles                 # correct letter maps to truth


def test_build_question_decoys_exclude_same_het():
    pool = [
        _fake_complex("aaaa", "AAA", "CCO", 3),
        _fake_complex("aaab", "AAA", "CCO", 3),           # same HET as truth
        _fake_complex("bbbb", "BBB", "c1ccccc1", 6),
        _fake_complex("cccc", "CCC", "CC(=O)O", 4),
    ]
    q = ev.build_question(pool[0], pool, k=3, rng=random.Random(1))
    # The duplicate-HET entry must not appear as a separate decoy option.
    assert [smi for _, smi in q.options].count("CCO") == 1


# -- answer parsing ---------------------------------------------------------


@pytest.mark.parametrize("text,n,expected", [
    ("B", 4, "B"),
    ("The answer is C.", 4, "C"),
    ("a", 4, "A"),
    ("Z", 4, None),          # out of range
    ("", 4, None),
])
def test_parse_letter(text, n, expected):
    assert ev.parse_letter(text, n) == expected


# -- representations are pocket-only ----------------------------------------


def test_representations_nonempty_and_ligand_blind():
    mol, site = _pocket_fixture()
    for arm, builder in ev.REPR_BUILDERS.items():
        text = builder(mol, site)
        assert text.strip(), f"{arm} representation is empty"
        # The pocket representation must not leak the ligand resname.
        assert "BEN" not in text, f"{arm} leaks the ligand identity"


def test_coords_repr_excludes_ligand_atoms():
    mol, site = _pocket_fixture()
    # Ligand atoms sit at x ~ 5.8-7.0; pocket protein atoms are at x <= 3.5.
    text = ev.repr_coords(mol, site)
    assert " 7.0" not in text and " 6.2" not in text


# -- statistics -------------------------------------------------------------


def test_mcnemar_counts_and_symmetry():
    a = [1, 1, 0, 0, 1]   # arm A correct flags
    b = [1, 0, 1, 1, 1]   # arm B correct flags
    mc = ev.mcnemar(a, b)
    assert mc["b_only"] == 2     # B right & A wrong: items 2,3
    assert mc["a_only"] == 1     # A right & B wrong: item 1
    assert 0.0 <= mc["p_value"] <= 1.0


def test_mcnemar_no_discordance_is_p1():
    flags = [1, 0, 1, 1]
    mc = ev.mcnemar(flags, flags)
    assert mc["a_only"] == 0 and mc["b_only"] == 0
    assert mc["p_value"] == 1.0


def test_bootstrap_ci_brackets_accuracy():
    flags = [1] * 7 + [0] * 3        # accuracy 0.7
    lo, hi = ev.bootstrap_ci(flags, random.Random(0), n_boot=500)
    assert 0.0 <= lo <= 0.7 <= hi <= 1.0


# -- end-to-end with a fake backend -----------------------------------------


class _FakeBackend:
    """Always answers with a fixed letter -- deterministic, no network."""

    def __init__(self, letter="A"):
        self.letter = letter
        self.calls = 0

    def choose(self, system, user):
        self.calls += 1
        return self.letter


def test_evaluate_runs_offline_with_fake_backend():
    pool = [
        _fake_complex("aaaa", "AAA", "CCO", 3),
        _fake_complex("bbbb", "BBB", "c1ccccc1", 6),
        _fake_complex("cccc", "CCC", "CC(=O)O", 4),
        _fake_complex("dddd", "DDD", "CCN", 3),
    ]
    backend = _FakeBackend("A")
    results, questions = ev.evaluate(
        pool, arms=("residues", "prose"), model="fake", k=3, seed=0,
        cache_path=Path("/unused"), backend=backend,
    )
    assert set(results) == {"residues", "prose"}
    assert all(len(rows) == len(pool) for rows in results.values())
    # Every "A" answer is correct exactly when the question's correct letter is A.
    for arm in ("residues", "prose"):
        for (pdb_id, flag) in results[arm]:
            assert flag == int(questions[pdb_id].correct == "A")
    assert backend.calls == len(pool) * 2

    report = ev.format_report(results, model="fake", k=3, n=len(pool))
    assert "accuracy" in report and "McNemar" in report
