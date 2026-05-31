"""Validation on a real public dataset: the Delaney ESOL solubility set.

The fixture is the ESOL aqueous-solubility benchmark (1128 drug-like compounds:
an id, a SMILES, and a measured log-solubility target).

  Delaney, J. S. "ESOL: Estimating Aqueous Solubility Directly from Molecular
  Structure." J. Chem. Inf. Comput. Sci. 44 (2004) 1000-1005. Redistributed via
  the MoleculeNet / DeepChem benchmark collection.

The main value of a large, messy, real table is exercising the dataset-prep
pipeline end to end -- scaffold/random splits, duplicate removal and
fingerprinting on real SMILES -- which toy fixtures cannot stress. The descriptor
check is a cheap bonus: it stretches the chemistry panel's
wrapper-transparency contract across 1128 real molecules. Skips without RDKit.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.validation

ESOL = Path(__file__).resolve().parents[1] / "fixtures" / "esol_solubility.csv"


@pytest.fixture(scope="module")
def esol_rows():
    pytest.importorskip("rdkit")
    with ESOL.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1128                              # guard against a truncated copy
    return rows


# -- dataset-prep pipeline on real data (the non-redundant value) ------------

def test_random_split_partitions_every_real_molecule(esol_rows):
    from molscope.prepare import prepare_dataset

    ds = prepare_dataset(
        str(ESOL), smiles_col="smiles", split="random",
        compute_descriptors=True, test=0.1, val=0.1, seed=0,
    )
    sizes = ds.split.sizes
    assert ds.n_prepared == len(esol_rows)               # nothing dropped without dedup
    assert sizes["train"] + sizes["validation"] + sizes["test"] == ds.n_prepared
    assert ds.descriptor_cols                            # descriptors computed from SMILES


def test_scaffold_split_is_a_disjoint_cover(esol_rows):
    """Scaffold splitting real drug-like molecules must still yield three disjoint
    index sets that cover every prepared row -- no leakage, nothing lost."""
    from molscope.prepare import prepare_dataset

    ds = prepare_dataset(
        str(ESOL), smiles_col="smiles", split="scaffold", test=0.1, val=0.1,
    )
    train, val, test = set(ds.split.train), set(ds.split.val), set(ds.split.test)
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert train | val | test == set(range(ds.n_prepared))


def test_canonical_dedup_finds_the_known_duplicates(esol_rows):
    """The ESOL set contains 11 molecules that are duplicates up to SMILES
    spelling; canonical dedup must collapse exactly those."""
    from molscope.prepare import prepare_dataset

    ds = prepare_dataset(
        str(ESOL), smiles_col="smiles", split="random", dedup="canonical",
        fingerprints=True, test=0.1, val=0.1,
    )
    assert ds.n_duplicates == 11
    assert ds.n_prepared == len(esol_rows) - 11
    assert ds.fingerprint_col == "morgan_onbits"         # fingerprints attached


def test_diverse_selection_returns_a_valid_subset(esol_rows):
    from molscope.library import read_table, select_diverse, smiles_descriptors

    table = read_table(str(ESOL))
    matrix, _ = smiles_descriptors(table.column("smiles"))
    chosen = select_diverse(matrix, 50)
    assert len(chosen) == 50
    assert len(set(chosen)) == 50                        # no row picked twice
    assert all(0 <= i < len(table) for i in chosen)


# -- descriptor wrapper transparency at scale (cheap bonus) ------------------

def test_smiles_descriptors_reproduce_rdkit_at_scale(esol_rows):
    """Over all 1128 real molecules, MolScope's batch descriptor wrapper must
    equal a direct RDKit computation -- the chemistry panel's transparency
    contract at scale, and RDKit-version-proof (both sides use installed RDKit)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    from molscope.library import smiles_descriptors

    names = ["MolWt", "RingCount", "NumHDonors", "NumRotatableBonds", "TPSA"]
    smiles = [r["smiles"].strip() for r in esol_rows]
    matrix, used = smiles_descriptors(smiles, names=names)
    assert used == names
    assert np.count_nonzero(~np.isnan(matrix[:, 0])) == len(esol_rows)   # all parse

    RDLogger.DisableLog("rdApp.*")
    funcs = dict(Descriptors._descList)
    try:
        direct = np.array([
            [funcs[name](Chem.MolFromSmiles(smi)) for name in names]
            for smi in smiles
        ])
    finally:
        RDLogger.EnableLog("rdApp.*")
    np.testing.assert_allclose(matrix, direct, rtol=0, atol=1e-9)
