"""Lightweight molecular coordinate workflows for descriptors, graphs, and CG beads.

molscope reads ``.xyz``, ``.pdb``, ``.cif`` and ``.sdf`` files (optionally
gzip-compressed) and turns static structures into three main outputs:
descriptor tables, molecular graphs for machine learning, and coarse-grained
bead representations.

Core workflows
--------------
- **PDB to descriptors**: fixed-width structural and optional RDKit-backed
  feature tables (:func:`descriptors`, :func:`featurize_many`).
- **PDB to graph/GNN**: atom/bond and residue-contact graphs for NetworkX,
  PyTorch Geometric and DGL (:class:`MolecularGraph`,
  :class:`ResidueContactGraph`).
- **PDB to coarse-grained beads**: residue, simplified Martini-style, custom,
  and virtual-site bead representations (:mod:`molscope.coarsegrain`).

Supporting capabilities include readers/writers, RCSB fetching, mmCIF
validation, selections, geometry, contact maps, simplified DSSP, ensemble
summaries, Matplotlib/py3Dmol visualization, and CLI automation.

Examples
--------
>>> import molscope as ms
>>> mol = ms.read("structure.pdb")     # parser chosen from the extension
>>> mol = ms.fetch("1fqy")             # ...or download straight from RCSB by id
>>> print(mol.summary())               # atoms, formula, chains, bounding box
>>> ca = mol.alpha_carbons()           # select the C-alpha atoms
>>> mol.plot(color_by="chain")         # render in 3D

>>> g = mol.to_graph()                 # molecular graph for ML, no extra deps
>>> cg = mol.coarse_grain("residue_com")   # one bead per residue

See https://github.com/roshan2004/molscope for the full documentation.
"""

from . import coarsegrain, contacts, docking, dssp, ensemble
from .chem import ChemicalFeatures, chemical_features, rdkit_descriptors, to_rdkit
from .cif import CifValidationReport, validate_cif
from .coarsegrain import (
    BeadMapping,
    BondMapping,
    CoarseGrainReport,
    DroppedAtom,
    VirtualSiteMapping,
)
from .coarsegrain import apply_mapping as apply_cg_mapping
from .coarsegrain import mapping_to_dict as cg_mapping_to_dict
from .coarsegrain import read_mapping as read_cg_mapping
from .coarsegrain import write_index as write_cg_index
from .coarsegrain import write_itp as write_cg_itp
from .coarsegrain import write_mapping as write_cg_mapping
from .coarsegrain import write_openmm_xml as write_cg_openmm_xml
from .contactmap import ContactMap
from .contacts import (
    BindingSite,
    ChainContactMatrix,
    Interface,
    LigandResidue,
    Pocket,
    PocketEnvironment,
    PocketInteraction,
    Residue,
    analyze_pocket,
    binding_site,
    chain_contact_matrix,
    interface_residues,
    ligands,
    pocket_descriptor_feature_names,
    select_pocket,
)
from .dataset import GraphDataset, TargetScaler, build_dataset, fetch_dataset
from .descriptors import descriptor_feature_names, descriptors, featurize_many, inertia_tensor
from .dssp import BackboneTorsions, SecondaryStructure, SSSegment, backbone_torsions
from .ensemble import (
    Clustering,
    StreamAnalysis,
    analyze_stream,
    cluster,
    cross_correlation,
    rmsd_matrix,
)
from .ensemble import contact_frequency as ensemble_contact_frequency
from .graph import (
    RESIDUE_INTERACTION_LABELS,
    MolecularGraph,
    ResidueContactGraph,
    delaunay_edges,
    edge_feature_names,
    knn_edges,
    node_feature_names,
    residue_contact_graph,
    residue_edge_feature_names,
    residue_node_feature_names,
)
from .io import (
    fetch,
    read,
    read_cif,
    read_pdb,
    read_pdb_models,
    read_sdf,
    read_sdf_frames,
    read_smiles,
    read_xyz,
    read_xyz_frames,
    stream,
    stream_pdb_models,
    stream_xyz_frames,
    write_cif,
    write_frames,
    write_pdb,
    write_sdf,
    write_xyz,
)
from .molecule import Molecule, ResidueGroup, ResidueId, UnitCell
from .plotting import (
    plot_cross_correlation,
    plot_distance_matrix,
    plot_mapping,
    plot_ramachandran,
    plot_rmsd_heatmap,
    view_mapping,
)
from .presets import PresetInfo, list_presets
from .quality import QualityReport, quality_report
from .sasa import ResidueExposure, relative_sasa, sasa
from .structure_prep import StructureReport, prepare_structure

__all__ = [
    "Clustering",
    "StreamAnalysis",
    "analyze_stream",
    "ChemicalFeatures",
    "CifValidationReport",
    "BackboneTorsions",
    "BeadMapping",
    "BindingSite",
    "BondMapping",
    "ChainContactMatrix",
    "CoarseGrainReport",
    "ContactMap",
    "DroppedAtom",
    "GraphDataset",
    "TargetScaler",
    "build_dataset",
    "fetch_dataset",
    "Interface",
    "LigandResidue",
    "Molecule",
    "MolecularGraph",
    "Pocket",
    "PocketEnvironment",
    "PocketInteraction",
    "Residue",
    "ResidueGroup",
    "ResidueId",
    "ResidueContactGraph",
    "SSSegment",
    "SecondaryStructure",
    "UnitCell",
    "VirtualSiteMapping",
    "analyze_pocket",
    "apply_cg_mapping",
    "backbone_torsions",
    "binding_site",
    "cg_mapping_to_dict",
    "chain_contact_matrix",
    "cluster",
    "chemical_features",
    "coarsegrain",
    "contacts",
    "descriptor_feature_names",
    "descriptors",
    "sasa",
    "relative_sasa",
    "ResidueExposure",
    "docking",
    "dssp",
    "ensemble",
    "ensemble_contact_frequency",
    "edge_feature_names",
    "featurize_many",
    "fetch",
    "inertia_tensor",
    "interface_residues",
    "ligands",
    "select_pocket",
    "plot_mapping",
    "view_mapping",
    "plot_rmsd_heatmap",
    "plot_distance_matrix",
    "plot_cross_correlation",
    "plot_ramachandran",
    "read",
    "read_cg_mapping",
    "read_cif",
    "read_pdb",
    "read_pdb_models",
    "read_sdf",
    "read_sdf_frames",
    "read_smiles",
    "read_xyz",
    "read_xyz_frames",
    "stream",
    "stream_pdb_models",
    "stream_xyz_frames",
    "rdkit_descriptors",
    "pocket_descriptor_feature_names",
    "residue_contact_graph",
    "RESIDUE_INTERACTION_LABELS",
    "residue_edge_feature_names",
    "residue_node_feature_names",
    "rmsd_matrix",
    "cross_correlation",
    "node_feature_names",
    "knn_edges",
    "delaunay_edges",
    "to_rdkit",
    "validate_cif",
    "prepare_structure",
    "StructureReport",
    "quality_report",
    "QualityReport",
    "list_presets",
    "PresetInfo",
    "write_cg_index",
    "write_cg_itp",
    "write_cg_mapping",
    "write_cg_openmm_xml",
    "write_cif",
    "write_frames",
    "write_pdb",
    "write_sdf",
    "write_xyz",
]
__version__ = "0.14.0"
