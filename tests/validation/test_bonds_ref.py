"""Tier 2 validation: distance-based bond perception vs RDKit topology.

For a molecule with a clean 3D geometry, RDKit's bond graph is the ground truth.
We build small molecules from SMILES, embed and minimise a 3D conformer with
RDKit, then check that molscope's purely geometric ``bonds()`` recovers exactly
that connectivity. Scored as per-molecule recall and precision over the bond
set. Skips when RDKit is not installed.
"""

import numpy as np
import pytest

import molscope as ms

pytestmark = pytest.mark.validation

# name -> SMILES; a spread of hybridisations, fused rings, heteroatoms,
# halogens, sulfur and a strained small ring. Distance-only perception should
# recover the connectivity of all of these clean equilibrium geometries.
PANEL = {
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "glycine": "NCC(=O)O",
    "toluene": "Cc1ccccc1",
    "acetic_acid": "CC(=O)O",
    "furan": "c1ccoc1",                 # O heteroaromatic
    "thiophene": "c1ccsc1",             # S heteroaromatic
    "imidazole": "c1c[nH]cn1",          # two-nitrogen heteroaromatic
    "pyrrole": "c1cc[nH]c1",
    "naphthalene": "c1ccc2ccccc2c1",    # fused aromatic rings
    "indole": "c1ccc2[nH]ccc2c1",       # fused hetero/aromatic
    "chlorobenzene": "Clc1ccccc1",      # halogen
    "dimethyl_sulfoxide": "CS(=O)C",    # sulfoxide S=O
    "acetamide": "CC(=O)N",             # amide
    "cyclopropane": "C1CC1",            # strained small ring, acute angles
    "cyclohexane": "C1CCCCC1",
}


def _embed(smiles: str):
    Chem = pytest.importorskip("rdkit.Chem")
    AllChem = pytest.importorskip("rdkit.Chem.AllChem")
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(mol, randomSeed=7) != 0:
        pytest.skip(f"RDKit could not embed {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    coords = np.asarray(mol.GetConformer().GetPositions())
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    truth = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    return coords, elements, truth


@pytest.mark.parametrize("name", list(PANEL))
def test_distance_bonds_recover_rdkit_topology(name):
    coords, elements, truth = _embed(PANEL[name])
    perceived = {frozenset(map(int, p)) for p in ms.Molecule(coords, elements).bonds(tolerance=1.2)}

    shared = len(truth & perceived)
    recall = shared / len(truth)
    precision = shared / len(perceived)
    print(f"\n{name}: recall={recall:.3f} precision={precision:.3f} "
          f"(rdkit={len(truth)}, perceived={len(perceived)})")

    # Measured at 1.000/1.000 across the panel; keep a small margin for future
    # molecules without letting a real perception regression slip through.
    assert recall >= 0.98
    assert precision >= 0.98


def test_distance_bonds_miss_a_stretched_bond():
    """A documented failure mode: distance-only perception keys on equilibrium
    covalent distances, so a non-equilibrium (stretched) geometry drops the bond.

    This is the honest flip side of the panel above -- it is *not* a regression
    but the expected behaviour, and it is why MolScope offers RDKit template
    bonds for structures whose chemistry the geometry alone cannot carry.
    """
    coords, elements, truth = _embed("CCO")
    # Find the C-O bond and stretch it well past any covalent + tolerance range.
    o_idx = elements.index("O")
    c_idx = next(
        next(iter(bond - {o_idx})) for bond in truth
        if o_idx in bond and elements[next(iter(bond - {o_idx}))] == "C"
    )
    bond_key = frozenset((c_idx, o_idx))
    assert bond_key in truth                                  # RDKit keeps it

    direction = coords[o_idx] - coords[c_idx]
    stretched = coords.copy()
    stretched[o_idx] = coords[c_idx] + direction / np.linalg.norm(direction) * 2.6
    perceived = {
        frozenset(map(int, p))
        for p in ms.Molecule(stretched, elements).bonds(tolerance=1.2)
    }
    assert bond_key not in perceived                          # geometry drops it
