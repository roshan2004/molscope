"""Validation for the docking-triage suite.

Tier 2: MolScope's multi-pose SDF reader (:func:`molscope.docking.read_poses`,
which the dock-* tools all sit on) is cross-checked against RDKit's independent
``SDMolSupplier`` parser -- pose count, titles, score data fields and per-atom
coordinates should agree, since both read the same V2000 records.

Tier 1: invariants the ranking and diversity logic must satisfy by
construction -- consensus reduces to a single field's ranking, a pose that wins
on every field ranks first, identical molecules collapse to one diverse
representative, and ligand efficiency is exactly the signed score per heavy atom.
These need no reference tool (the diversity ones need RDKit only to fingerprint).
"""

from pathlib import Path

import numpy as np
import pytest

from molscope import docking

pytestmark = pytest.mark.validation

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
# Hand-authored (not written by RDKit), so comparing our parser to RDKit's is a
# genuine two-parser cross-check rather than a round-trip through one library.
POSES_SDF = str(FIXTURES / "docking_poses.sdf")


@pytest.fixture(scope="module")
def rdkit():
    Chem = pytest.importorskip("rdkit.Chem")
    pytest.importorskip("rdkit.Chem.AllChem")
    return Chem


def _write_ligand_sdf(path, items, score_field="minimizedAffinity"):
    """Embed ``(name, smiles, score)`` items into a multi-record SDF with RDKit."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    writer = Chem.SDWriter(str(path))
    for name, smiles, score in items:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=11)
        mol.SetProp("_Name", name)
        mol.SetProp(score_field, f"{score:.2f}")
        writer.write(mol)
    writer.close()
    return str(path)


# -- Tier 2: reader vs RDKit SDMolSupplier ----------------------------------

def _rdkit_records(Chem, path):
    """Parse every record with RDKit, keeping all atoms and the raw data fields."""
    supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
    return [m for m in supplier if m is not None]


def test_read_poses_matches_rdkit_on_handwritten_sdf(rdkit):
    poses = docking.read_poses(POSES_SDF)
    ref = _rdkit_records(rdkit, POSES_SDF)

    assert len(poses) == len(ref)
    # Titles agree.
    assert [p.name for p in poses] == [m.GetProp("_Name") for m in ref]
    # Score data fields agree (parsed independently from the > <tag> block).
    for pose, mol in zip(poses, ref):
        assert pose.score("minimizedAffinity") == pytest.approx(
            float(mol.GetProp("minimizedAffinity"))
        )
        assert pose.score("CNNscore") == pytest.approx(float(mol.GetProp("CNNscore")))
    # Per-atom elements and coordinates agree to file precision.
    for pose, mol in zip(poses, ref):
        conf = mol.GetConformer()
        assert pose.molecule.elements == [a.GetSymbol() for a in mol.GetAtoms()]
        ref_xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        np.testing.assert_allclose(pose.molecule.coords, ref_xyz, atol=1e-4)


def test_read_poses_matches_rdkit_on_bonded_ligands(rdkit, tmp_path):
    items = [
        ("benzene", "c1ccccc1", -7.2),
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O", -8.9),
        ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", -8.1),
    ]
    sdf = _write_ligand_sdf(tmp_path / "ligands.sdf", items)
    poses = docking.read_poses(sdf)
    ref = _rdkit_records(rdkit, sdf)

    assert len(poses) == len(ref) == len(items)
    assert [p.name for p in poses] == [name for name, _, _ in items]
    for pose, mol in zip(poses, ref):
        assert len(pose.molecule) == mol.GetNumAtoms()
        conf = mol.GetConformer()
        ref_xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        np.testing.assert_allclose(pose.molecule.coords, ref_xyz, atol=1e-4)
        assert pose.score("minimizedAffinity") == pytest.approx(
            float(mol.GetProp("minimizedAffinity"))
        )


# -- Tier 1: ranking invariants (no reference tool) -------------------------

def test_consensus_of_one_field_reproduces_its_ranking():
    """With a single score field, consensus order == that field's plain ranking."""
    poses = docking.read_poses(POSES_SDF)
    summary = docking.summarize(
        poses, "minimizedAffinity", higher_is_better_flag=False, with_smiles=False,
    )
    expected = [row["name"] for row in summary.rows]

    consensus = docking.consensus_rank(
        [("f", poses)], score_fields=["minimizedAffinity"], key="name",
    )
    ordered = sorted(consensus.rows, key=lambda r: r["final_rank"])
    assert [r["key"] for r in ordered] == expected


def test_pose_that_wins_every_field_ranks_first():
    """A molecule best on all score fields must take consensus rank 1 (domination)."""
    poses = docking.read_poses(POSES_SDF)
    # In the fixture ligA_pose1 has both the best (lowest) minimizedAffinity and
    # the best (highest) CNNscore, so it dominates on every field.
    consensus = docking.consensus_rank(
        [("f", poses)], score_fields=["minimizedAffinity", "CNNscore"], key="name",
    )
    best = min(consensus.rows, key=lambda r: r["final_rank"])
    assert best["key"] == "ligA_pose1"
    assert best["final_rank"] == 1


def test_ligand_efficiency_is_signed_score_per_heavy_atom():
    poses = docking.read_poses(POSES_SDF)
    summary = docking.summarize(
        poses, "minimizedAffinity", higher_is_better_flag=False, with_smiles=False,
    )
    for row in summary.rows:
        # lower-is-better affinity -> efficiency = -score / heavy atoms.
        assert row["ligand_efficiency"] == pytest.approx(
            -row["score"] / row["n_heavy_atoms"]
        )


# -- Tier 1: diversity invariants (need RDKit to fingerprint) ----------------

def test_identical_molecules_collapse_to_one_representative(rdkit, tmp_path):
    """The whole point of dock-diverse: near-identical analogues must not each be
    selected. Five copies of one molecule plus two distinct ones should yield
    three clusters and one representative for the duplicate group."""
    items = [
        ("dup1", "c1ccccc1", -7.0), ("dup2", "c1ccccc1", -7.5),
        ("dup3", "c1ccccc1", -6.8), ("dup4", "c1ccccc1", -7.9),
        ("dup5", "c1ccccc1", -7.1),
        ("distinct_a", "CCCCCCCCO", -8.2),
        ("distinct_b", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", -8.5),
    ]
    sdf = _write_ligand_sdf(tmp_path / "dupes.sdf", items)
    poses = docking.read_poses(sdf)
    result = docking.select_diverse_hits(
        poses, "minimizedAffinity", higher_is_better_flag=False,
        top=10, select=10, threshold=0.7,
    )
    # Three chemistries -> three clusters; never five benzene rows.
    assert result.n_clusters == 3
    assert len(result.selected) == 3
    names = [rep["name"] for rep in result.selected]
    dup_names = {"dup1", "dup2", "dup3", "dup4", "dup5"}
    chosen_dupes = dup_names & set(names)
    assert len(chosen_dupes) == 1                       # exactly one benzene survives
    assert chosen_dupes == {"dup4"}                     # and it is the best-scoring one


def test_diverse_representatives_come_from_distinct_clusters(rdkit, tmp_path):
    items = [
        ("ethanol", "CCO", -5.1), ("benzene", "c1ccccc1", -7.2),
        ("aspirin", "CC(=O)Oc1ccccc1C(=O)O", -8.9),
        ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", -8.1),
    ]
    sdf = _write_ligand_sdf(tmp_path / "mix.sdf", items)
    poses = docking.read_poses(sdf)
    result = docking.select_diverse_hits(
        poses, "minimizedAffinity", higher_is_better_flag=False,
        top=10, select=10, threshold=0.6,
    )
    cluster_ids = [rep["cluster_id"] for rep in result.selected]
    assert len(cluster_ids) == len(set(cluster_ids))   # one representative per cluster
    assert len(result.selected) <= result.n_clusters
    # Representatives are returned best-score-first (lower affinity is better).
    scores = [rep["score"] for rep in result.selected]
    assert scores == sorted(scores)
