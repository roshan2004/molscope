"""Tests for optional RDKit-backed chemical perception."""

import os

import numpy as np
import pytest

import molscope as ms
from molscope import ChemicalFeatures, Molecule

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def test_pdb_template_bonds_perceive_aromatic_rings():
    """Residue templates recover aromaticity that geometric bonds miss."""
    pytest.importorskip("rdkit")
    path = os.path.join(DATA, "1ubq.pdb")

    geometric = ms.read(path)
    template = ms.read(path, bond_perception="template")
    assert template.bond_index is not None

    geo_arom = int(sum(bool(a) for a in geometric.chemical_features().aromatic_atoms))
    tpl_arom = int(sum(bool(a) for a in template.chemical_features().aromatic_atoms))
    assert geo_arom == 0  # geometric single bonds carry no aromatic perception
    assert tpl_arom >= 20  # Phe/Tyr/His rings of ubiquitin


def test_pdb_template_bonds_returns_aligned_arrays():
    pytest.importorskip("rdkit")
    from molscope.chem import pdb_template_bonds

    path = os.path.join(DATA, "1ubq.pdb")
    bond_index, bond_orders, charges = pdb_template_bonds(path, ms.read(path))
    assert bond_index.shape[1] == 2
    assert len(bond_orders) == len(bond_index)
    assert len(charges) == 660
    assert set(np.unique(bond_orders)).issubset({1.0, 2.0, 3.0})  # Kekule, no 1.5


def test_standard_protonation_assigns_sidechain_charges():
    pytest.importorskip("rdkit")
    path = os.path.join(DATA, "1ubq.pdb")
    neutral = ms.read(path, bond_perception="template")
    charged = ms.read(path, bond_perception="template", protonation="standard")
    assert int(neutral.formal_charges.sum()) == 0  # as-modelled, neutral
    # Ubiquitin is near-neutral at pH 7, but charges are actually assigned:
    assert int((charged.formal_charges != 0).sum()) > 0
    # ...and the molecule still sanitises with the assigned charges.
    assert charged.chemical_features().formal_charges.sum() == charged.formal_charges.sum()


def test_pdb_template_bonds_rejects_bad_protonation():
    pytest.importorskip("rdkit")
    from molscope.chem import pdb_template_bonds

    path = os.path.join(DATA, "1ubq.pdb")
    with pytest.raises(ValueError, match="protonation"):
        pdb_template_bonds(path, ms.read(path), protonation="ph9")


def test_pka_formal_charge_decision():
    """Pure pKa-vs-pH charge logic (no PROPKA needed)."""
    from molscope.chem import _pka_formal_charge

    # Acid: deprotonated (-1) only when the solution is more basic than its pKa.
    assert _pka_formal_charge("acid", 3.9, 7.0) == -1
    assert _pka_formal_charge("acid", 9.0, 7.0) == 0
    # Base: protonated (+1) only when the solution is more acidic than its pKa.
    assert _pka_formal_charge("base", 10.5, 7.0) == 1
    assert _pka_formal_charge("base", 6.0, 7.0) == 0
    # At pH == pKa the uncharged species is reported.
    assert _pka_formal_charge("acid", 7.0, 7.0) == 0
    assert _pka_formal_charge("base", 7.0, 7.0) == 0
    with pytest.raises(ValueError):
        _pka_formal_charge("amphoteric", 7.0, 7.0)


def test_pka_protonation_tracks_ph():
    """PROPKA-predicted charges are near-neutral at pH 7 and swing with pH."""
    pytest.importorskip("rdkit")
    pytest.importorskip("propka")
    path = os.path.join(DATA, "1ubq.pdb")

    at7 = ms.read(path, bond_perception="template", protonation="pka", ph=7.0)
    acidic = ms.read(path, bond_perception="template", protonation="pka", ph=2.0)
    basic = ms.read(path, bond_perception="template", protonation="pka", ph=12.0)

    # Charges are actually assigned, and ubiquitin is ~neutral at pH 7.
    assert int((at7.formal_charges != 0).sum()) > 0
    assert abs(int(at7.formal_charges.sum())) <= 1
    # Low pH protonates (more positive) than high pH (more negative).
    assert int(acidic.formal_charges.sum()) > int(basic.formal_charges.sum())
    assert int(acidic.formal_charges.sum()) > 0 > int(basic.formal_charges.sum())
    # The assigned charges still sanitise in RDKit.
    assert at7.chemical_features().formal_charges.sum() == at7.formal_charges.sum()


def test_pka_protonation_requires_template_bonds():
    pytest.importorskip("rdkit")
    path = os.path.join(DATA, "1ubq.pdb")
    with pytest.raises(ValueError, match="template"):
        ms.read(path, protonation="pka")


def test_require_propka_install_hint(monkeypatch):
    """A missing PROPKA yields the documented install hint, not an opaque error."""
    import builtins

    from molscope.chem import _require_propka

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "propka.run" or name.startswith("propka"):
            raise ImportError("no propka")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"molscope\[propka\]"):
        _require_propka()


def test_chemical_features_require_rdkit():
    pytest.importorskip("rdkit")
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        ["C", "O"],
        bond_index=[[0, 1]],
        bond_orders=[2],
    )
    features = mol.chemical_features()
    assert isinstance(features, ChemicalFeatures)
    np.testing.assert_array_equal(features.bond_orders, [2.0])
    np.testing.assert_array_equal(features.formal_charges, [0, 0])


def test_chemical_features_reports_aromaticity():
    pytest.importorskip("rdkit")
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    coords = np.stack([np.cos(angles), np.sin(angles), np.zeros(6)], axis=1)
    bonds = [[i, (i + 1) % 6] for i in range(6)]
    mol = Molecule(coords, ["C"] * 6, bond_index=bonds, bond_orders=[1.5] * 6)
    features = mol.chemical_features()
    assert features.aromatic_atoms.all()
    assert features.aromatic_bonds.all()


def test_rdkit_descriptors_by_name():
    pytest.importorskip("rdkit")
    mol = Molecule(
        np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        ["C", "O"],
        bond_index=[[0, 1]],
        bond_orders=[2],
    )
    desc = mol.rdkit_descriptors(names=["MolWt", "TPSA"])
    assert desc["rdkit_MolWt"] > 0.0
    assert desc["rdkit_TPSA"] >= 0.0


def test_rdkit_descriptors_reject_unknown_name():
    pytest.importorskip("rdkit")
    mol = Molecule(np.zeros((1, 3)), ["C"])
    with pytest.raises(ValueError, match="unknown RDKit descriptor"):
        mol.rdkit_descriptors(names=["not_a_descriptor"])
