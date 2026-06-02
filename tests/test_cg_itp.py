"""Tests for the GROMACS .itp coarse-grained topology skeleton writer."""

import numpy as np

import molscope as ms
from molscope import Molecule


def _beads(bond_index, n=None):
    """A minimal bead Molecule (write_itp only reads Molecule attributes)."""
    n = n if n is not None else (int(np.max(bond_index)) + 1 if len(bond_index) else 1)
    coords = np.arange(3 * n).reshape(n, 3).astype(float)
    return Molecule(
        coords, ["C"] * n, name="cg model",
        atom_names=[f"B{i}" for i in range(n)],
        resnames=["MOL"] * n, resids=np.arange(1, n + 1),
        bond_index=np.asarray(bond_index, dtype=int) if len(bond_index) else None,
    )


def _sections(text):
    """Map each ``[ section ]`` to its list of non-comment data rows."""
    out, current = {}, None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped.strip("[] ").strip()
            out[current] = []
        elif current and stripped and not stripped.startswith(";"):
            out[current].append(stripped)
    return out


def test_write_cg_itp_has_all_sections(tmp_path):
    cg = _beads([[0, 1], [1, 2]])
    path = str(tmp_path / "m.itp")
    ms.write_cg_itp(cg, path)
    secs = _sections(open(path).read())
    assert set(secs) >= {"moleculetype", "atoms", "bonds", "angles"}
    assert len(secs["atoms"]) == 3
    assert len(secs["bonds"]) == 2


def test_write_cg_itp_atom_types_and_residues(tmp_path):
    cg = two_alanines().coarse_grain("martini")
    path = str(tmp_path / "ala.itp")
    ms.write_cg_itp(cg, path)
    text = open(path).read()
    assert "CG_ALA_BB" in text and "CG_ALA_SC" in text


def two_alanines():
    names = ["N", "CA", "C", "O", "CB"] * 2
    els = ["N", "C", "C", "O", "C"] * 2
    return Molecule(
        np.arange(30).reshape(10, 3).astype(float), els, name="dialanine",
        atom_names=names, resnames=["ALA"] * 10,
        resids=np.array([1] * 5 + [2] * 5), chains=["A"] * 10,
    )


def test_write_cg_itp_enumerates_angles_from_bond_graph(tmp_path):
    # linear 0-1-2 -> one angle (0-1-2)
    linear = str(tmp_path / "lin.itp")
    ms.write_cg_itp(_beads([[0, 1], [1, 2]]), linear)
    assert len(_sections(open(linear).read())["angles"]) == 1

    # star 0-{1,2,3} -> three angles around the central bead
    star = str(tmp_path / "star.itp")
    ms.write_cg_itp(_beads([[0, 1], [0, 2], [0, 3]]), star)
    assert len(_sections(open(star).read())["angles"]) == 3


def test_write_cg_itp_sanitises_molecule_name(tmp_path):
    path = str(tmp_path / "m.itp")
    ms.write_cg_itp(_beads([[0, 1]]), path)
    text = open(path).read()
    # "cg model" -> "cg_model" (no spaces in a GROMACS moleculetype name)
    assert "cg_model   1" in text


def test_write_cg_itp_without_bonds_has_empty_bond_and_angle_sections(tmp_path):
    path = str(tmp_path / "nobond.itp")
    ms.write_cg_itp(_beads([], n=3), path)
    secs = _sections(open(path).read())
    assert len(secs["atoms"]) == 3
    assert secs["bonds"] == [] and secs["angles"] == []
