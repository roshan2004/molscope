"""Tests for the simplified DSSP secondary-structure assignment."""

import os

import numpy as np
import pytest

import molscope as ms
from molscope import SecondaryStructure, dssp
from molscope.molecule import Molecule

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")


def aquaporin():
    return ms.read(os.path.join(DATA, "1fqy.pdb"))


def test_assign_returns_one_code_per_residue():
    mol = aquaporin()
    ss = mol.secondary_structure()
    assert isinstance(ss, SecondaryStructure)
    n_residues = sum(1 for _ in mol.residue_groups())
    assert len(ss) == n_residues
    assert len(ss.string) == n_residues
    assert set(ss.string) <= set("HGIEBTS-")


def test_aquaporin_is_helix_rich():
    # Aquaporin-1 is an all-alpha membrane protein: helix-dominated, no sheets.
    summary = aquaporin().secondary_structure().summary()
    assert summary["helix_fraction"] > 0.4
    assert summary["strand"] == 0
    counts = summary["helix"] + summary["strand"] + summary["coil"]
    assert counts == summary["residues"]


def test_summary_fractions_sum_to_one():
    summary = aquaporin().secondary_structure().summary()
    total = (
        summary["helix_fraction"]
        + summary["strand_fraction"]
        + summary["coil_fraction"]
    )
    assert total == pytest.approx(1.0)


def test_per_atom_ss_aligns_with_atoms():
    mol = aquaporin()
    per_atom = dssp.per_atom_ss(mol)
    assert len(per_atom) == len(mol)
    assert set(per_atom) <= set("HGIEBTS-")


def test_plot_color_by_ss(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    mol = aquaporin()
    ax = mol.plot(color_by="ss", show=False)
    assert ax is not None


def test_requires_backbone_metadata():
    # A bare coordinate molecule (no atom names / resids) cannot be assigned.
    bare = Molecule(np.zeros((3, 3)), ["C", "C", "C"])
    with pytest.raises(ValueError):
        bare.secondary_structure()


def test_simplified_reduces_to_three_states():
    ss = aquaporin().secondary_structure()
    simple = ss.simplified()
    assert len(simple) == len(ss)
    assert set(simple) <= set("HEC")
    # Every 8-state code maps to the right 3-state bucket.
    for eight, three in zip(ss.string, simple):
        expected = {"H": "H", "G": "H", "I": "H", "E": "E", "B": "E"}.get(eight, "C")
        assert three == expected


def test_segments_are_contiguous_non_coil_runs():
    ss = aquaporin().secondary_structure()
    segs = ss.segments()
    assert segs, "aquaporin should have secondary-structure elements"
    assert all(seg.code != "-" for seg in segs)
    assert all(seg.length == seg.end - seg.start + 1 for seg in segs)
    # Residue spans within the dominant helix-rich protein are non-trivial.
    assert max(seg.length for seg in segs) > 5
    # include_coil exposes the gaps too; total residues then equals the assignment.
    assert sum(s.length for s in ss.segments(include_coil=True)) == len(ss)


def test_per_chain_totals_match_global():
    ss = aquaporin().secondary_structure()
    per_chain = ss.per_chain()
    assert set(per_chain) == set(ss.chains)
    assert sum(v["residues"] for v in per_chain.values()) == len(ss)
    assert sum(v["helix"] for v in per_chain.values()) == ss.summary()["helix"]


def test_backbone_torsions_match_ramachandran():
    mol = aquaporin()
    tor = mol.backbone_torsions()
    assert len(tor) == len(mol.secondary_structure())
    # Termini are undefined.
    assert np.isnan(tor.phi[0])
    assert np.isnan(tor.psi[-1])
    # Peptide bonds are overwhelmingly trans (omega ~ 180).
    assert np.nanmedian(np.abs(tor.omega)) > 170
    # Helix residues cluster in the canonical Ramachandran basin.
    codes = mol.secondary_structure().codes
    helix = np.isin(codes, ["H", "G", "I"])
    assert -90 < np.nanmedian(tor.phi[helix]) < -40
    assert -70 < np.nanmedian(tor.psi[helix]) < -20


def test_backbone_torsions_requires_backbone():
    bare = Molecule(np.zeros((3, 3)), ["C", "C", "C"])
    with pytest.raises(ValueError):
        bare.backbone_torsions()


# -- Ramachandran plot ------------------------------------------------------


def test_plot_ramachandran_scatters_defined_torsions():
    import matplotlib
    matplotlib.use("Agg")

    mol = aquaporin()
    ax = mol.plot_ramachandran(show=False)
    pts = ax.collections[-1].get_offsets()
    # one point per residue with both phi and psi defined (chain ends dropped)
    tor = mol.backbone_torsions()
    defined = int((~np.isnan(tor.phi) & ~np.isnan(tor.psi)).sum())
    assert len(pts) == defined
    assert bool((np.abs(np.asarray(pts)) <= 180).all())
    assert ax.get_legend() is not None              # SS legend by default
    assert len(ax.patches) >= 3                      # schematic region guides


def test_plot_ramachandran_single_colour_and_no_regions():
    import matplotlib
    matplotlib.use("Agg")

    ax = aquaporin().plot_ramachandran(color_by=None, regions=False, show=False)
    assert ax.get_legend() is None
    assert len(ax.patches) == 0


def test_plot_ramachandran_requires_backbone():
    import matplotlib
    matplotlib.use("Agg")

    bare = Molecule(np.zeros((3, 3)), ["O", "H", "H"], name="water")
    with pytest.raises(ValueError, match="protein backbone"):
        bare.plot_ramachandran(show=False)
