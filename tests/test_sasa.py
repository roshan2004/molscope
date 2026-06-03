"""Tests for the pure-NumPy Shrake-Rupley SASA approximation."""

import os
import sys

import numpy as np
import pytest

import molscope as ms
from molscope import Molecule, elements
from molscope.sasa import atom_sasa, sasa

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def test_vdw_radius_table_and_default():
    assert elements.vdw_radius("C") == 1.70
    assert elements.vdw_radius("o") == 1.52  # case-insensitive
    assert elements.vdw_radius("Xx") == elements.DEFAULT_VDW_RADIUS


def test_isolated_atom_is_exact_sphere_area():
    # No neighbours, so every sample point is accessible regardless of n_points.
    mol = Molecule(np.zeros((1, 3)), ["C"])
    r = elements.vdw_radius("C") + 1.4
    np.testing.assert_allclose(mol.sasa(), [4 * np.pi * r**2])


def test_two_atoms_match_analytical_spherical_cap():
    # Two equal spheres distance d apart bury a cap of height R - d/2 on each;
    # accessible area per atom is 2*pi*R*(R + d/2).
    d = 4.0
    mol = Molecule(np.array([[0.0, 0, 0], [d, 0, 0]]), ["C", "C"])
    r = elements.vdw_radius("C") + 1.4
    expected = 2 * np.pi * r * (r + d / 2)
    values = mol.sasa(n_points=4000)
    np.testing.assert_allclose(values, [expected, expected], rtol=0.03)


def test_more_points_improves_accuracy():
    d = 4.0
    mol = Molecule(np.array([[0.0, 0, 0], [d, 0, 0]]), ["C", "C"])
    r = elements.vdw_radius("C") + 1.4
    expected = 2 * np.pi * r * (r + d / 2)
    coarse = abs(mol.sasa(n_points=24)[0] - expected)
    fine = abs(mol.sasa(n_points=2000)[0] - expected)
    assert fine < coarse


def test_buried_atom_has_much_less_area_than_isolated():
    # A central atom octahedrally surrounded by close neighbours is largely buried.
    coords = np.array([
        [0.0, 0, 0],
        [2.0, 0, 0], [-2.0, 0, 0],
        [0, 2.0, 0], [0, -2.0, 0],
        [0, 0, 2.0], [0, 0, -2.0],
    ])
    mol = Molecule(coords, ["C"] * 7)
    values = mol.sasa(n_points=2000)
    isolated = 4 * np.pi * (elements.vdw_radius("C") + 1.4) ** 2
    assert values[0] < 0.15 * isolated  # core atom barely exposed
    assert (values[1:] > values[0]).all()  # outer atoms more exposed


def test_residue_level_sums_atoms_and_matches_total():
    mol = ms.read(os.path.join(DATA, "1fqy.pdb"))
    per_atom = mol.sasa(level="atom", n_points=96)
    per_res = mol.sasa(level="residue", n_points=96)
    assert per_res.shape[0] == sum(1 for _ in mol.residue_groups())
    np.testing.assert_allclose(per_res.sum(), per_atom.sum(), rtol=1e-9)


def test_residue_level_requires_residue_info():
    mol = Molecule(np.zeros((2, 3)), ["C", "C"])
    with pytest.raises(ValueError, match="residue"):
        mol.sasa(level="residue")


def test_rejects_bad_level_and_zero_points():
    mol = Molecule(np.zeros((2, 3)), ["C", "C"])
    with pytest.raises(ValueError, match="level"):
        mol.sasa(level="molecule")
    with pytest.raises(ValueError, match="n_points"):
        mol.sasa(n_points=0)


def test_empty_molecule_returns_empty():
    assert sasa(Molecule(np.empty((0, 3)), [])).shape == (0,)


def test_numpy_fallback_matches_scipy(monkeypatch):
    coords = np.array([[0.0, 0, 0], [3.0, 0, 0], [1.5, 2.0, 0], [1.0, 1.0, 2.5]])
    mol = Molecule(coords, ["C", "N", "O", "C"])
    expected = atom_sasa(mol, n_points=256)
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    np.testing.assert_allclose(atom_sasa(mol, n_points=256), expected)


def test_top_level_export_matches_method():
    mol = Molecule(np.array([[0.0, 0, 0], [3.0, 0, 0]]), ["C", "C"])
    np.testing.assert_array_equal(ms.sasa(mol, n_points=128), mol.sasa(n_points=128))


# -- relative solvent accessibility (RSA) -----------------------------------


def test_max_asa_reference_table():
    assert elements.max_asa("ALA") == 129.0
    assert elements.max_asa("trp") == 285.0    # case-insensitive
    assert elements.max_asa("LIG") is None     # no reference for ligands


def _alanine():
    coords = np.array([[0.0, 0, 0], [1.5, 0, 0], [2.5, 1, 0], [2.5, 2, 0], [1.5, -1.4, 0]])
    return Molecule(
        coords, ["N", "C", "C", "O", "C"], name="ala",
        atom_names=["N", "CA", "C", "O", "CB"], resnames=["ALA"] * 5,
        resids=[1] * 5, chains=["A"] * 5,
    )


def test_relative_sasa_isolated_residue_is_exposed():
    exp = _alanine().relative_sasa(n_points=200)
    assert len(exp) == 1
    assert np.isfinite(exp.rsa[0]) and exp.rsa[0] > exp.threshold
    assert bool(exp.exposed[0])
    assert exp.resnames == ["ALA"]


def test_relative_sasa_nan_without_reference():
    lig = Molecule(
        np.array([[0.0, 0, 0], [1.5, 0, 0]]), ["C", "O"], name="lig",
        atom_names=["C1", "O1"], resnames=["LIG", "LIG"], resids=[1, 1], chains=["A", "A"],
    )
    exp = lig.relative_sasa(n_points=48)
    assert np.isnan(exp.rsa[0])
    assert not bool(exp.exposed[0])


def test_relative_sasa_threshold_classification():
    mol = ms.read(os.path.join(DATA, "1fqy.pdb"))
    exp = mol.relative_sasa(n_points=48)
    finite = np.isfinite(exp.rsa)
    # exposed flag is exactly rsa >= threshold where defined
    assert np.array_equal(exp.exposed[finite], exp.rsa[finite] >= exp.threshold)
    assert (exp.rsa[finite] >= 0).all()
    # a lower threshold cannot expose fewer residues
    loose = mol.relative_sasa(n_points=48, threshold=0.05)
    assert int(loose.exposed.sum()) >= int(exp.exposed.sum())


def test_relative_sasa_requires_residues():
    with pytest.raises(ValueError, match="residue"):
        Molecule(np.zeros((2, 3)), ["C", "C"]).relative_sasa()
