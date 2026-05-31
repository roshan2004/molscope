"""Tier 2 validation: RDKit-backed chemistry features vs direct RDKit results.

MolScope delegates optional chemical perception to RDKit. These tests make that
contract explicit: for molecules with explicit RDKit-derived bond orders and
charges, MolScope's feature arrays and descriptor values should match RDKit's
own atom, bond and descriptor APIs. Skips when RDKit is not installed.
"""

import numpy as np
import pytest

from molscope import Molecule

pytestmark = pytest.mark.validation

PANEL = {
    "benzene": "c1ccccc1",
    "pyridine": "n1ccccc1",
    "nitrobenzene": "O=[N+]([O-])c1ccccc1",
    "glycine_zwitterion": "[NH3+]CC(=O)[O-]",
    "imidazole": "c1c[nH]cn1",            # multi-N heteroaromatic
    "furan": "c1ccoc1",                   # O heteroaromatic
    "thiophene": "c1ccsc1",               # S heteroaromatic
    "phenol": "Oc1ccccc1",                # aromatic + H-bond donor
    "aniline": "Nc1ccccc1",
    "acetate": "CC(=O)[O-]",              # anion
    "methylammonium": "C[NH3+]",          # cation
    "histidine": "N[C@@H](Cc1c[nH]cn1)C(=O)O",   # ring + amine + acid
}

DESCRIPTORS = ("MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "RingCount")


@pytest.fixture(scope="module")
def rdkit():
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")
    from rdkit.Chem import Descriptors

    return Chem, AllChem, Descriptors


def _rdkit_to_molscope(Chem, AllChem, smiles: str):
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(rdmol, randomSeed=17) != 0:
        pytest.skip(f"RDKit could not embed {smiles}")
    AllChem.MMFFOptimizeMolecule(rdmol)
    Chem.SanitizeMol(rdmol)

    coords = np.asarray(rdmol.GetConformer().GetPositions(), dtype=float)
    elements = [atom.GetSymbol() for atom in rdmol.GetAtoms()]
    charges = [atom.GetFormalCharge() for atom in rdmol.GetAtoms()]
    bonds = [
        [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        for bond in rdmol.GetBonds()
    ]
    orders = [bond.GetBondTypeAsDouble() for bond in rdmol.GetBonds()]
    return rdmol, Molecule(
        coords,
        elements,
        bond_index=bonds,
        bond_orders=orders,
        formal_charges=charges,
    )


@pytest.mark.parametrize("name", list(PANEL))
def test_chemical_features_match_direct_rdkit(rdkit, name):
    Chem, AllChem, _ = rdkit
    rdmol, mol = _rdkit_to_molscope(Chem, AllChem, PANEL[name])
    features = mol.chemical_features()

    np.testing.assert_array_equal(
        features.formal_charges,
        [atom.GetFormalCharge() for atom in rdmol.GetAtoms()],
    )
    np.testing.assert_array_equal(
        features.aromatic_atoms,
        [atom.GetIsAromatic() for atom in rdmol.GetAtoms()],
    )
    np.testing.assert_allclose(
        features.bond_orders,
        [bond.GetBondTypeAsDouble() for bond in rdmol.GetBonds()],
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        features.aromatic_bonds,
        [bond.GetIsAromatic() for bond in rdmol.GetBonds()],
    )


@pytest.mark.parametrize("name", list(PANEL))
def test_rdkit_descriptors_match_direct_rdkit(rdkit, name):
    Chem, AllChem, Descriptors = rdkit
    rdmol, mol = _rdkit_to_molscope(Chem, AllChem, PANEL[name])
    direct = dict(Descriptors._descList)
    mine = mol.rdkit_descriptors(names=list(DESCRIPTORS))

    for descriptor in DESCRIPTORS:
        expected = float(direct[descriptor](rdmol))
        observed = mine[f"rdkit_{descriptor}"]
        print(f"\n{name} {descriptor}: molscope={observed:.6g} rdkit={expected:.6g}")
        assert observed == pytest.approx(expected, rel=1e-12, abs=1e-12)
