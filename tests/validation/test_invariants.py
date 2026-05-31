"""Tier 1 validation: physical invariants that must hold without any reference tool.

These assert truths the implementation must satisfy by construction (mass-weighted
geometry, rigid-body algebra, mapping conservation), so they need no optional
dependency and run as part of the normal suite. They are cheap regression
insurance: if one breaks, the maths is wrong, independent of any external tool.
"""

from pathlib import Path

import numpy as np

import molscope as ms

DATA = Path(__file__).resolve().parents[2] / "examples" / "data"
PROTEIN = str(DATA / "1fqy.pdb")


def _fibonacci_sphere(n: int, radius: float) -> np.ndarray:
    i = np.arange(n)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * i
    z = 1.0 - 2.0 * i / (n - 1)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return radius * np.c_[r * np.cos(phi), r * np.sin(phi), z]


# -- rigid-body algebra -----------------------------------------------------


def test_kabsch_recovers_known_rigid_transform():
    """Superposing a known rotation+translation back must drive RMSD to ~0."""
    ca = ms.read(PROTEIN).alpha_carbons()
    moved = ca.rotate(axis="z", angle_deg=37.0).translate([5.0, -2.0, 1.0])
    assert ca.rmsd(moved, align=True) < 1e-9


def test_rmsd_is_zero_against_self():
    ca = ms.read(PROTEIN).alpha_carbons()
    assert ca.rmsd(ca, align=True) < 1e-12


# -- geometry primitives ----------------------------------------------------


def test_dihedral_of_planar_cis_is_zero():
    """Four coplanar points in a cis arrangement give a 0 deg torsion by definition."""
    mol = ms.Molecule(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float), ["C"] * 4)
    assert abs(mol.dihedral(0, 1, 2, 3)) < 1e-9


def test_right_angle_is_ninety_degrees():
    mol = ms.Molecule(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float), ["C"] * 4)
    assert mol.angle(0, 1, 2) == 90.0


def test_distance_matches_euclidean_norm():
    mol = ms.read(PROTEIN)
    expected = float(np.linalg.norm(mol.coords[0] - mol.coords[10]))
    assert mol.distance(0, 10) == np.float64(expected) or abs(mol.distance(0, 10) - expected) < 1e-9


def test_radius_of_gyration_of_uniform_shell_equals_radius():
    """For equal masses on a radius-R shell centred at the origin, Rg == R exactly."""
    pts = _fibonacci_sphere(500, radius=3.0)
    mol = ms.Molecule(pts, ["C"] * 500)
    assert abs(mol.radius_of_gyration - 3.0) < 1e-3


# -- coarse-graining conservation -------------------------------------------


def test_residue_com_bead_count_equals_residue_count():
    mol = ms.read(PROTEIN)
    n_residues = sum(1 for _ in mol.residue_groups())
    cg = mol.coarse_grain("residue_com", weighted=True)
    assert len(cg) == n_residues


def test_residue_com_beads_sit_at_residue_centres_of_mass():
    """Each residue_com bead must equal the mass-weighted COM of its own atoms."""
    mol = ms.read(PROTEIN)
    coords, masses = mol.coords, mol.masses
    expected = np.array(
        [(coords[idx] * masses[idx, None]).sum(0) / masses[idx].sum()
         for idx, *_ in mol.residue_groups()]
    )
    cg = mol.coarse_grain("residue_com", weighted=True)
    assert np.allclose(cg.coords, expected, atol=1e-9)


def test_residue_centroid_beads_sit_at_residue_centroids():
    """Each residue_centroid bead must equal the unweighted mean of its atoms."""
    mol = ms.read(PROTEIN)
    expected = np.array([mol.coords[idx].mean(0) for idx, *_ in mol.residue_groups()])
    cg = mol.coarse_grain("residue_centroid")
    assert np.allclose(cg.coords, expected, atol=1e-9)


# -- contact map correctness ------------------------------------------------


def test_atom_contact_map_equals_brute_force():
    """The atom-level contact map must equal a direct all-pairs distance threshold."""
    ca = ms.read(PROTEIN).alpha_carbons()
    cutoff = 8.0
    mat = ca.contact_map(cutoff=cutoff, level="atom").matrix
    d = np.linalg.norm(ca.coords[:, None, :] - ca.coords[None, :, :], axis=-1)
    brute = ((d <= cutoff) & ~np.eye(len(ca), dtype=bool)).astype(float)
    assert np.array_equal(mat, brute)


# -- ensemble alignment metrics ---------------------------------------------
# Deterministic, controlled ensembles (no new data files) that complement the
# MDAnalysis cross-check on the real 1aml ensemble: superposition must remove
# rigid-body motion entirely and localise real per-atom variance.


def _rigid_models(base, moves):
    """A base structure plus rigid (rotation+translation) copies of it."""
    models = [base]
    for axis, angle, shift in moves:
        models.append(base.rotate(axis=axis, angle_deg=angle).translate(shift))
    return models


def test_rmsf_of_a_rigidly_moving_ensemble_is_zero():
    """Models that differ only by rigid motion have no internal fluctuation, so
    after alignment every per-atom RMSF must collapse to zero."""
    base = ms.read(PROTEIN).alpha_carbons()
    models = _rigid_models(
        base, [("z", 15.0, [1, 0, 0]), ("y", 40.0, [0, 2, -1]), ("x", 75.0, [-3, 1, 2])]
    )
    rmsf = ms.ensemble.rmsf(models)
    assert np.allclose(rmsf, 0.0, atol=1e-5)


def test_rmsd_matrix_of_a_rigidly_moving_ensemble_is_zero():
    """Pairwise superposition of the same structure under rigid motion gives an
    all-zero RMSD matrix (and an exactly zero diagonal)."""
    base = ms.read(PROTEIN).alpha_carbons()
    models = _rigid_models(base, [("x", 30.0, [5, -2, 1]), ("y", 60.0, [0, 3, 0])])
    matrix = ms.ensemble.rmsd_matrix(models)
    assert np.allclose(matrix, 0.0, atol=1e-5)
    assert np.allclose(np.diag(matrix), 0.0)


def test_rmsf_localises_a_single_displaced_atom():
    """A single atom displaced by growing amounts across models must carry the
    largest fluctuation; on a large structure alignment barely compensates."""
    base = ms.read(PROTEIN).alpha_carbons()
    target = len(base) // 2
    models = [base]
    for d in (0.5, 1.0, 1.5):
        coords = base.coords.copy()
        coords[target] += np.array([d, 0.0, 0.0])
        models.append(ms.Molecule(coords, list(base.elements)))
    rmsf = ms.ensemble.rmsf(models)
    assert int(np.argmax(rmsf)) == target
