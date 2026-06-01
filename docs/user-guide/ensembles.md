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
