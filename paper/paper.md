---
title: 'MolScope: a lightweight bridge from molecular structures to descriptors, graph-ML inputs, and coarse-grained representations in Python'
tags:
  - Python
  - molecular structure
  - protein structure
  - graph neural networks
  - coarse-graining
  - cheminformatics
authors:
  - name: Roshan Shrestha
    orcid: 0000-0002-9356-5136
    corresponding: true
    affiliation: 1
affiliations:
  - name: Independent Researcher, Nepal
    index: 1
date: 29 May 2026
bibliography: paper.bib
---

# Summary

`MolScope` is a lightweight Python toolkit that bridges static molecular
structures to three research-ready outputs: fixed-width structural descriptor
tables, machine-learning graphs, and coarse-grained bead representations. It
reads common coordinate formats (XYZ, PDB, mmCIF, SDF), builds a molecule from a
SMILES string, fetches structures from the RCSB Protein Data Bank, and provides
atom- and residue-level selections, geometry, contacts and contact maps, a
dependency-free secondary-structure assignment, and protein-aware bond and
chemistry perception.

The package is built on `NumPy` [@harris2020array] and `matplotlib`
[@hunter2007matplotlib] only. All chemistry, file-format, and graph-backend
dependencies (`RDKit` [@rdkit], `Gemmi` [@wojdyr2022gemmi], `NetworkX`
[@hagberg2008networkx], `SciPy` [@virtanen2020scipy], `PyTorch`
[@paszke2019pytorch], `PyTorch Geometric` [@fey2019pytorch_geometric], `DGL`
[@wang2019dgl]) are optional extras installed only when the relevant feature is
used. This keeps the base install small and the dependency graph predictable,
while still allowing each workflow to hand off to the wider ecosystem.

# Statement of need

For teaching exercises, quick structural quality-control checks, and small
machine-learning-for-molecules prototypes, the available Python tooling fits
poorly. Students need readable code that exposes the structure of molecular data
rather than hiding it behind a framework. Researchers prototyping graph neural
networks need a deterministic way to turn a structure file into a
`torch_geometric.data.Data` object without first standing up a full
featurisation pipeline. Modellers want to inspect a coarse-grained mapping before
committing to a production Martini [@marrink2007martini] setup.

`MolScope` targets exactly these users. Every workflow ends in a concrete,
ready-to-use object (a descriptor table, a graph, or a bead model), and the
package documents what it deliberately is *not*, so users can recognise when
their work has outgrown it and reach for a heavier specialist tool.

# State of the field

Python tooling for molecular structure work clusters at two extremes.
Specialist packages are powerful and battle-tested but carry substantial
dependency footprints and production-oriented APIs: `MDAnalysis`
[@michaud2011mdanalysis] and `MDTraj` [@mcgibbon2015mdtraj] for trajectories,
`RDKit` for cheminformatics, `Biopython` [@cock2009biopython] for bioinformatics
parsing, and `PyMOL` [@pymol] for interactive visualisation. At the other
extreme, assembling parsers, selection logic, and graph builders directly on
`NumPy` quickly becomes a bespoke pile that is easy to get wrong and hard to
maintain.

`MolScope` occupies the middle ground rather than competing with those tools. It
is intentionally narrower than `MDAnalysis` or `RDKit` and intentionally more
opinionated than a `NumPy`-only approach, optimised for the shortest path from a
static structure to descriptors, an ML graph, or a coarse-grained prototype.
When a workflow needs the depth of a specialist package, `MolScope` exports to
it (`NetworkX`, `PyTorch Geometric`, `DGL`) or defers to it (`RDKit` chemistry,
`Gemmi` validation) rather than reimplementing it.

# Software design

`MolScope` is organised around an immutable `Molecule` object and a small core
(`NumPy` and `matplotlib`) with every heavier capability behind an optional
extra that degrades gracefully when absent. Three workflows are documented end
to end in the project's tutorials:

1. **Structure to descriptors.** Geometry, contacts, contact maps, residue-level
   analyses, a dependency-free DSSP-style secondary-structure assignment, and
   optional `RDKit`-backed scalar descriptors. For proteins, an opt-in
   residue-template path uses `RDKit` to assign correct bonds, bond orders, and
   aromaticity that geometric distance inference cannot recover.
2. **Structure to graph.** Atom-and-bond and residue-contact graphs with named
   node and edge feature presets, Laplacian and random-walk positional
   encodings, and exporters to `NetworkX`, `PyTorch Geometric`, and `DGL`.
3. **Structure to coarse-grained beads.** Residue centroids, simplified
   Martini-style backbone and side-chain mappings, custom mappings, and virtual
   sites. The output is a regular `Molecule`, so plotting and graph export remain
   available on the coarse-grained representation.

The package is reachable through a Python API, a command-line interface
(`molscope`) for common batch operations, and an optional Model Context Protocol
server that exposes the analyses to AI assistants. A design priority is honesty
about scope: a per-workflow limitations page and a per-feature validation table
state plainly which results are cross-checked against reference tools and which
are intentionally lightweight (for example, the coarse-graining tools are for
educational mapping and bead-graph prototyping, not validated force-field
generation).

That honesty extends to what `MolScope` itself contributes versus what it
delegates. The methods it implements are the structure-to-descriptor,
structure-to-graph, and structure-to-coarse-grained workflows above, built on a
NumPy and `matplotlib` core; capabilities such as chemical perception, robust
mmCIF parsing, environment-aware protonation, and reference cross-checks are
provided by integrating established tools (`RDKit`, `gemmi`, `PROPKA`,
`Dimorphite-DL`, `MDAnalysis`) behind optional extras, not by re-implementing
them. `MolScope`'s contribution in those cases is the lightweight, consistent,
gracefully-degrading bridge from a coordinate file to a usable artifact, and a
new external backend is added only when that integration adds value beyond a
direct call to the underlying tool. The package is therefore intentionally not a
replacement for a full cheminformatics or simulation stack.

That honesty is backed by a two-tier validation suite. Dependency-free
invariants (rigid-body algebra, geometry, coarse-grain conservation) run
everywhere, and reference cross-checks run where the reference tool is available:
geometry and contacts against `MDAnalysis`, bond and chemistry perception
against `RDKit`, and secondary structure against `mkdssp` [@kabsch1983dssp].
After collapsing to the helix/strand/coil alphabet, the DSSP-style assignment
reaches 98 to 99% per-residue agreement with `mkdssp` across three fold classes
(helix-dominated, mixed alpha/beta, and all-beta). Continuous integration runs
the suite on Python 3.9, 3.11, and 3.13.

# Research impact statement

`MolScope` is a young package, and its intended impact is in teaching,
exploratory analysis, and early-stage molecular machine learning rather than in
production pipelines. Its near-term significance rests on lowering the barrier
between a structure file and a usable research artifact: a reproducible
descriptor table for classical models, a deterministic graph for message-passing
experiments, or an inspectable coarse-grained mapping. The deliberately small
dependency core makes it suitable for classroom and continuous-integration
environments where a full simulation or cheminformatics stack is impractical,
and the reference-validated components make its descriptor and structural outputs
trustworthy enough to build on. The package is openly developed and released, so
adoption and downstream use can be tracked through its public repository and
archive.

# AI usage disclosure

Generative AI was used substantially in the development of `MolScope`. Anthropic
Claude models (via the Claude Code agent) assisted with implementation, test
authoring, documentation, and the drafting of this paper. The author directed
the work, reviewed and validated all contributions, and is solely responsible
for the correctness and content of the software and this manuscript. The two-tier
validation suite, which cross-checks results against the independent reference
tools `MDAnalysis`, `RDKit`, and `mkdssp` and against known per-residue
chemistry, provides verification of scientific correctness that is independent of
how the code was produced.

# Availability and archival

`MolScope` is distributed on PyPI as `molscope` and developed openly on GitHub at
<https://github.com/roshan2004/molscope> under the MIT licence, with
documentation at <https://molscope.readthedocs.io/>. Each tagged release is
archived on Zenodo; the concept DOI
[10.5281/zenodo.20433850](https://doi.org/10.5281/zenodo.20433850) resolves to
the latest archived version, and the version submitted here is archived under its
own version DOI.

# Acknowledgements

The author thanks the maintainers of `NumPy`, `matplotlib`, `RDKit`,
`MDAnalysis`, `PyTorch Geometric`, `DGL`, and `Gemmi`, on whose work `MolScope`'s
optional integrations depend. No external financial support was received for this
work.

# References
