"""Validation: real solution-NMR ensembles for the alignment metrics.

Complements the single bundled ``1aml`` ensemble (``test_geometry_ref.py``) with
a panel of real NMR structures, cross-checking MolScope's per-atom RMSF and
Kabsch RMSD against MDAnalysis on the same files.

* **Bundled (runs in CI):** four small ensembles, gzipped (~1 MB total) --
  ``1d3z`` (ubiquitin, 10 models), ``2lz3`` (21), ``6qfp`` (10), ``1gab`` (20).
* **Opt-in remote** (set ``MOLSCOPE_RUN_REMOTE_PDB=1``): ``6v5d`` (176 models),
  too large to bundle.
* ``2hyn`` is used only to confirm MolScope *rejects* an ensemble whose models
  carry inconsistent atom counts -- a real quirk of that deposition -- rather
  than silently misaligning them.

Skips without MDAnalysis (the ``validation`` extra).
"""

import gzip
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

import molscope as ms

pytestmark = pytest.mark.validation

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
BUNDLED = ("1d3z", "2lz3", "6qfp", "1gab")
# Bundled run in CI; 6v5d is opt-in remote (too large to commit).
ALIGNMENT_PANEL = BUNDLED + ("6v5d",)


@pytest.fixture(scope="module")
def mda():
    return pytest.importorskip("MDAnalysis")


@pytest.fixture(scope="module")
def workdir():
    return tempfile.mkdtemp()


def _require_remote():
    if os.environ.get("MOLSCOPE_RUN_REMOTE_PDB") != "1":
        pytest.skip("set MOLSCOPE_RUN_REMOTE_PDB=1 to fetch the large remote NMR ensembles")


def _plain_pdb(pid: str, workdir: str) -> str:
    """A plain (uncompressed) ``.pdb`` path, since MDAnalysis reads the same file.

    Bundled ids are decompressed from their gzipped fixture; remote ids are
    fetched into the MolScope cache.
    """
    if pid in BUNDLED:
        plain = os.path.join(workdir, f"{pid}.pdb")
        if not os.path.exists(plain):
            with gzip.open(FIXTURES / f"{pid}.pdb.gz", "rb") as src, open(plain, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return plain
    _require_remote()
    ms.fetch(pid)                                            # download + cache
    return os.path.join(tempfile.gettempdir(), "molscope_cache", f"{pid}.pdb")


@pytest.mark.parametrize("pid", ALIGNMENT_PANEL)
def test_rmsf_matches_mdanalysis(pid, mda, workdir):
    from MDAnalysis.analysis import align, rms

    path = _plain_pdb(pid, workdir)
    models = ms.read_pdb_models(path)
    assert len(models) > 1                                   # a genuine ensemble
    mine = ms.ensemble.rmsf(models)                          # aligns each model to model 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(path, in_memory=True)
        align.AlignTraj(u, u, select="all", ref_frame=0, in_memory=True).run()
        ref = rms.RMSF(u.atoms).run().results.rmsf
    assert mine.shape == ref.shape
    assert np.allclose(mine, ref, atol=1e-3)


@pytest.mark.parametrize("pid", ALIGNMENT_PANEL)
def test_kabsch_rmsd_matches_mdanalysis(pid, mda, workdir):
    from MDAnalysis.analysis import rms

    path = _plain_pdb(pid, workdir)
    m1 = ms.read_pdb(path, model=1)
    m2 = ms.read_pdb(path, model=2)
    mine = m1.rmsd(m2, align=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(path)
        u.trajectory[0]
        p0 = u.atoms.positions.copy()
        u.trajectory[1]
        p1 = u.atoms.positions.copy()
    assert len(m1) == len(u.atoms)                           # same atom selection/order
    assert mine == pytest.approx(rms.rmsd(p0, p1, superposition=True), abs=1e-4)


def test_inconsistent_ensemble_is_rejected(workdir):
    """2HYN's NMR models do not all carry the same atoms; the ensemble API must
    refuse to compute RMSF over them rather than silently misalign."""
    _require_remote()
    path = _plain_pdb("2hyn", workdir)
    models = ms.read_pdb_models(path)
    assert len({len(m) for m in models}) > 1                 # genuinely inconsistent
    with pytest.raises(ValueError, match="differing atom counts"):
        ms.ensemble.rmsf(models)
