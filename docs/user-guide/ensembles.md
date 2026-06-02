# Ensemble Analysis

Read all models from an NMR PDB file:

```python
models = ms.read_pdb_models("examples/data/1aml.pdb")
```

Compute ensemble descriptors:

```python
from molscope import ensemble

aligned = ensemble.align_all(models)
avg = ensemble.average(models)
rmsf = ensemble.rmsf(models)
matrix = ensemble.rmsd_matrix(models)
```

Cluster structures by RMSD:

```python
result = ensemble.cluster(models, n_clusters=3)
result.labels
result.representatives()
```

Contact frequency across models:

```python
freq = ms.ensemble_contact_frequency(models, cutoff=8.0)
freq.plot()
```

## Concerted motions (dynamical cross-correlation)

`contact_frequency` tells you *which* contacts form, but not whether parts of
the structure move in a coordinated way. The dynamical cross-correlation matrix
(DCCM) answers that: each entry is the correlation of two atoms' displacements
about their mean positions, from `+1` (moving together in lockstep) through `0`
(uncorrelated) to `-1` (moving in opposite directions). Coupled off-diagonal
blocks are the classic fingerprint of allosteric communication.

```python
import molscope as ms

models = ms.read_pdb_models("examples/data/1aml.pdb")
ca = [m.alpha_carbons() for m in models]    # residue-level DCCM
corr = ms.cross_correlation(ca)             # (n_residues, n_residues), in [-1, 1]
ms.plot_cross_correlation(corr)
```

Structures are Kabsch-superposed onto the first model first (`align=True`) so
that rigid-body tumbling does not swamp the internal motion, exactly as `rmsf`
does. Omit the alpha-carbon selection to get an all-atom map. It is a few NumPy
operations over the coordinate stack: lightweight, but `O(N²)` in memory, so
prefer the residue-level (alpha-carbon) map for large systems.

## Streaming trajectory-lite analysis

The functions above take a list of models held in memory. For a long trajectory,
`analyze_stream` walks the frames in a **single pass** and keeps only the
reference frame in memory, tracking a few scalars per frame:

```python
analysis = ms.analyze_stream("trajectory.pdb", secondary_structure=True)

analysis.radius_of_gyration   # (n_frames,) Rg per frame
analysis.rmsd                 # (n_frames,) RMSD to the first frame (rmsd[0] == 0)
analysis.helix_fraction       # helix/strand/coil fractions (proteins; else None)
analysis.summary()            # means, spread, and drift
analysis.plot()               # Rg / RMSD / SS timeline panels
```

`source` is a path to a multi-frame file (multi-model PDB, multi-frame XYZ, or
multi-record SDF, streamed via `ms.stream`) **or** any iterable of `Molecule`
frames. RMSD is Kabsch-superposed over `selection` — `"auto"` (C-alphas when
present, else all atoms), `"ca"`, or `"all"`. Frames must share the first frame's
atom count.

!!! note "Lite timeline, not a trajectory engine"
    `analyze_stream` reads the multi-frame formats MolScope already reads and
    computes a handful of scalars. It does **not** read binary MD trajectories
    (DCD/XTC/TRR), unwrap periodic boundaries, or track time/topology across
    frames. For those, use a dedicated trajectory library such as MDAnalysis or
    MDTraj.
