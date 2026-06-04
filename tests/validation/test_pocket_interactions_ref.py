"""Tier 2 validation: MolScope's pocket-interaction heuristics vs PLIP.

MolScope detects ligand-pocket interactions from *heavy-atom geometry only*, so
the honest test is not class-by-class equality with a full reference profiler
but a per-residue **agreement** measure: precision / recall against PLIP
(Adasme et al., *Nucleic Acids Res.* 2021) at residue granularity.

Because the two tools draw the hydrogen-bond / salt-bridge boundary differently
(PLIP reports the benzamidine-Asp189 contact as hydrogen bonds, MolScope as a
salt bridge), the asserted metric is the **polar-contact union** (H-bond OR
salt bridge); the finer per-class numbers are printed for visibility. The test
asserts defensible floors that hold both on the bundled ``3ptb`` alone and on
the full remote panel, and prints the table so any regression is obvious.

The comparison logic lives in ``scripts/validate_pocket_interactions.py`` (also
runnable as a standalone report). PLIP is conda-only in practice; create the
reference environment once with::

    conda create -n plip-ref -c conda-forge plip openbabel -y

The test skips cleanly when that environment (or any python that can
``import plip``) is not reachable. The remote panel downloads from RCSB and is
opt-in via ``MOLSCOPE_RUN_REMOTE_PDB=1``.
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.validation

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

harness = pytest.importorskip("validate_pocket_interactions")

if not harness.plip_available():
    pytest.skip(
        "PLIP reference environment not available; create it with "
        "'conda create -n plip-ref -c conda-forge plip openbabel -y' "
        "(or set MOLSCOPE_PLIP_CMD).",
        allow_module_level=True,
    )

# Floors with margin below the observed values (panel: polar P 0.82 / R 0.97,
# hydrophobic R 0.88; 3ptb-only: polar P 0.67 / R 1.00, hydrophobic R 1.00).
POLAR_RECALL_FLOOR = 0.80
POLAR_PRECISION_FLOOR = 0.55
HYDROPHOBIC_RECALL_FLOOR = 0.70


def _panel():
    panel = harness.LOCAL_PANEL
    if os.environ.get("MOLSCOPE_RUN_REMOTE_PDB") == "1":
        panel = panel + harness.REMOTE_PANEL
    return panel


def test_pocket_interactions_agree_with_plip(capsys):
    panel = _panel()
    totals, per_structure = harness.evaluate(panel)

    report = harness.format_report(totals, per_structure, harness.POCKET_CUTOFF)
    with capsys.disabled():
        print("\n" + report + "\n")

    polar = totals["polar"]
    hydrophobic = totals["hydrophobic"]

    # MolScope's permissive heavy-atom rules should rarely *miss* a PLIP polar
    # contact (high recall) while over-calling somewhat (lower precision).
    assert polar.recall >= POLAR_RECALL_FLOOR, (
        f"polar-contact recall {polar.recall:.2f} below floor {POLAR_RECALL_FLOOR}"
    )
    assert polar.precision >= POLAR_PRECISION_FLOOR, (
        f"polar-contact precision {polar.precision:.2f} below floor {POLAR_PRECISION_FLOOR}"
    )
    assert hydrophobic.recall >= HYDROPHOBIC_RECALL_FLOOR, (
        f"hydrophobic recall {hydrophobic.recall:.2f} below floor {HYDROPHOBIC_RECALL_FLOOR}"
    )


def test_benzamidine_polar_contacts_recovered():
    """The canonical 3ptb S1 site: MolScope must recover PLIP's Asp189 contact."""
    import molscope as ms

    path = str(harness.DATA / "3ptb.pdb")
    mol = ms.read(path)
    ligand = max(mol.ligands(), key=len)
    ms_sets = harness.molscope_sets(mol, ligand)
    ref = harness._match_site(harness.plip_sites(path), ligand)

    # Asp189 is the specificity residue; both tools see it as a polar contact
    # (PLIP as H-bonds, MolScope as a salt bridge -> both land in the union).
    assert ("A", 189) in ms_sets["polar"]
    assert ("A", 189) in ref["polar"]
    assert ref["polar"] <= ms_sets["polar"]   # MolScope recovers every PLIP polar residue
