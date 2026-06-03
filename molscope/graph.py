"""Molecular graph layer.

``Molecule.to_graph()`` turns 3D coordinates plus explicit or inferred bonds
into a :class:`MolecularGraph` — a small, dependency-free container of node and edge
data. From there it exports to the common ML graph formats:

    G   = mol.to_graph()
    nxg = mol.to_networkx()     # networkx.Graph
    data = mol.to_pyg_data()    # torch_geometric.data.Data
    dglg = mol.to_dgl_graph()   # dgl.DGLGraph

The exporters import their backend lazily, so networkx / PyTorch Geometric / DGL
are only required if you actually call the matching method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import elements
from .molecule import ResidueId

DEFAULT_GRAPH_ELEMENTS = (
    "H", "C", "N", "O", "S", "P", "F", "CL", "BR", "I", "NA", "MG", "CA", "FE", "ZN",
)
DEFAULT_RESIDUE_TYPES = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
)

GRAPH_NODE_FEATURE_PRESETS = ("default", "basic", "ml")
GRAPH_EDGE_FEATURE_PRESETS = ("default", "basic", "ml", "geom")
RESIDUE_NODE_FEATURE_PRESETS = ("default", "ml")
RESIDUE_EDGE_FEATURE_PRESETS = ("default", "ml")
CONTACT_GRAPH_METHODS = ("ca", "com", "min")

# Coarse, mutually-exclusive interaction labels for residue-contact edges, in
# precedence order (most specific first). These are geometric/contact heuristics
# from residue chemistry and atom distances, NOT binding-energy terms.
RESIDUE_INTERACTION_LABELS = (
    "disulfide", "ligand", "salt_bridge", "covalent", "hydrophobic", "polar", "proximity",
)
_STANDARD_AA = set(DEFAULT_RESIDUE_TYPES) - {"UNK"}
# Solvent is not a "ligand"; its contacts fall through to "proximity".
_WATER_RESNAMES = {"HOH", "WAT", "H2O", "DOD", "TIP", "TIP3", "SOL"}
_HYDROPHOBIC_AA = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO"}
_POLAR_AA = {"SER", "THR", "ASN", "GLN", "TYR", "CYS", "HIS", "ASP", "GLU", "LYS", "ARG"}
_BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}
_BASIC_SIDECHAIN_ATOMS = {"ARG": {"NE", "NH1", "NH2"}, "LYS": {"NZ"}, "HIS": {"ND1", "NE2"}}
_ACIDIC_SIDECHAIN_ATOMS = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
_DISULFIDE_CUTOFF = 2.5      # SG-SG (angstrom)
_SALT_BRIDGE_CUTOFF = 4.0    # charged side-chain N...O
_SIDECHAIN_CONTACT_CUTOFF = 5.0   # side-chain heavy-atom contact for hydrophobic/polar


def _sign_stabilize(vecs: np.ndarray) -> np.ndarray:
    """Fix the (arbitrary) sign of each eigenvector column deterministically.

    Eigensolvers return eigenvectors with an arbitrary sign, and that sign can
    differ between the dense (``numpy.linalg.eigh``) and sparse (``scipy``
    ``eigsh``) paths. We flip each column so its largest-magnitude entry is
    positive, making Laplacian PEs reproducible regardless of which backend
    runs. Note this cannot resolve the rotational ambiguity *within* degenerate
    eigenspaces, where the two backends may still differ.
    """
    if vecs.size == 0:
        return vecs
    # Index of the largest-|.| entry in each column; flip so that entry is +ve.
    # Round the magnitudes first: in symmetric structures several entries share
    # the same magnitude, and raw float jitter between backends would otherwise
    # make argmax pick different (tied) pivots and so choose opposite signs.
    pivot = np.argmax(np.round(np.abs(vecs), 8), axis=0)
    signs = np.sign(vecs[pivot, np.arange(vecs.shape[1])])
    signs[signs == 0] = 1.0
    return vecs * signs


@dataclass
class MolecularGraph:
    """Atoms as nodes, bonds as edges, with attributes attached to both.

    Nodes carry element, atomic number, mass, coordinates and (when available)
    atom name, residue name/id, chain and formal charge. Edges carry the bonded
    atom pair, the interatomic distance, and a bond order (``1.0`` for inferred
    or unknown orders).
    """

    coords: np.ndarray              # (N, 3)
    elements: list[str]
    edges: np.ndarray               # (E, 2), i < j
    edge_distances: np.ndarray      # (E,)
    edge_types: np.ndarray          # (E,) bond order; 1.0 when inferred/unknown
    atom_names: list[str] = field(default_factory=list)
    resnames: list[str] = field(default_factory=list)
    resids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    icodes: list[str] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)
    name: str = ""
    formal_charges: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    aromatic_atoms: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    aromatic_bonds: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    virtual_sites: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    # True where an edge is a covalent bond; for spatial (k-NN/radius/Delaunay)
    # graphs this flags which contacts coincide with real bonds. Empty -> all True
    # (a covalent graph). See :meth:`covalent_edge_flags`.
    covalent_edges: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))

    @property
    def n_atoms(self) -> int:
        return len(self.coords)

    @property
    def n_bonds(self) -> int:
        return len(self.edges)

    def adjacency_matrix(self, weighted: bool = False) -> np.ndarray:
        """Return the dense adjacency matrix (N, N)."""
        adj = np.zeros((self.n_atoms, self.n_atoms), dtype=float)
        if self.n_bonds == 0:
            return adj
        i, j = self.edges[:, 0], self.edges[:, 1]
        val = self.edge_types if weighted else 1.0
        adj[i, j] = val
        adj[j, i] = val
        return adj

    def laplacian_pe(self, k: int = 8) -> np.ndarray:
        """Compute Laplacian Positional Encodings (N, k).

        This computes the eigenvectors of the normalized Laplacian matrix
        L = I - D^-1/2 A D^-1/2, returning the ``k`` belonging to the smallest
        eigenvalues (including the trivial constant eigenvector). scipy's sparse
        solver is used when available; column signs are stabilized so the result
        is independent of the backend.
        """
        if self.n_atoms <= 1:
            return np.zeros((self.n_atoms, k))

        try:
            import scipy.sparse as sp
            from scipy.sparse.linalg import eigsh
            has_scipy = True
        except ImportError:
            has_scipy = False

        if has_scipy and self.n_atoms > k + 1:
            try:
                row = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
                col = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
                data = np.ones(2 * self.n_bonds, dtype=float)
                adj_sparse = sp.coo_matrix(
                    (data, (row, col)), shape=(self.n_atoms, self.n_atoms)
                ).tocsr()
                deg = np.array(adj_sparse.sum(axis=1)).flatten()
                with np.errstate(divide="ignore", invalid="ignore"):
                    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
                d_inv_sqrt_mat = sp.diags(deg_inv_sqrt)
                scaled_adj = d_inv_sqrt_mat @ adj_sparse @ d_inv_sqrt_mat
                l_norm_sparse = sp.eye(self.n_atoms) - scaled_adj
                # Shift-invert (sigma=0) reliably finds the eigenvalues closest
                # to zero — i.e. the k smallest of the PSD Laplacian. ARPACK's
                # which='SM' mode is notoriously unreliable here and can skip the
                # trivial zero eigenvalue, returning a different set than the
                # dense path.
                eigvals, eigvecs = eigsh(l_norm_sparse, k=k, sigma=0, which='LM')
                idx = np.argsort(eigvals)
                return _sign_stabilize(eigvecs[:, idx])
            except Exception:
                pass

        adj = self.adjacency_matrix()
        deg = adj.sum(axis=1)
        # Avoid division by zero for isolated atoms
        with np.errstate(divide="ignore", invalid="ignore"):
            deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        l_norm = np.eye(self.n_atoms) - deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
        # eigh returns eigenvalues in ascending order; take the k smallest
        # (including the trivial constant eigenvector).
        eigvals, eigvecs = np.linalg.eigh(l_norm)
        return _sign_stabilize(eigvecs[:, :k])

    def random_walk_pe(self, k: int = 16) -> np.ndarray:
        """Compute Random Walk Structural Encodings (N, k).

        The features are the diagonal elements [P_ii, P^2_ii, ..., P^k_ii] of
        the transition matrix P = A D^-1.
        """
        if self.n_atoms == 0:
            return np.zeros((0, k))

        try:
            import scipy.sparse as sp
            has_scipy = True
        except ImportError:
            has_scipy = False

        if has_scipy:
            try:
                row = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
                col = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
                data = np.ones(2 * self.n_bonds, dtype=float)
                adj_sparse = sp.coo_matrix(
                    (data, (row, col)), shape=(self.n_atoms, self.n_atoms)
                ).tocsr()
                deg = np.array(adj_sparse.sum(axis=1)).flatten()
                with np.errstate(divide="ignore", invalid="ignore"):
                    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
                d_inv_mat = sp.diags(deg_inv)
                p_sparse = d_inv_mat @ adj_sparse

                pe = np.zeros((self.n_atoms, k))
                pk = p_sparse.copy()
                for i in range(k):
                    pe[:, i] = pk.diagonal()
                    if i < k - 1:
                        pk = pk @ p_sparse
                return pe
            except Exception:
                pass

        adj = self.adjacency_matrix()
        deg = adj.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(deg[:, None] > 0, adj / deg[:, None], 0.0)

        pe = np.zeros((self.n_atoms, k))
        pk = p.copy()
        for i in range(k):
            pe[:, i] = np.diag(pk)
            if i < k - 1:
                pk = pk @ p
        return pe

    @property
    def atomic_numbers(self) -> np.ndarray:
        return np.array([elements.atomic_number(e) for e in self.elements])

    @property
    def masses(self) -> np.ndarray:
        return np.array([elements.mass(e) for e in self.elements])

    def node_features(
        self,
        preset: str = "default",
        *,
        elements_to_encode=DEFAULT_GRAPH_ELEMENTS,
        return_names: bool = False,
    ):
        """Return a node feature matrix for a named preset.

        ``"default"`` preserves the historical ``[atomic_number, mass]`` matrix.
        ``"basic"`` adds formal charge. ``"ml"`` adds a fixed element one-hot
        basis and an aromatic-atom flag, using zeros when aromaticity was not
        supplied by optional chemical perception.
        """
        names = node_feature_names(preset, elements_to_encode=elements_to_encode)
        arrays = []
        for name in names:
            if name == "atomic_number":
                arrays.append(self.atomic_numbers.astype(float))
            elif name == "mass":
                arrays.append(self.masses.astype(float))
            elif name == "formal_charge":
                arrays.append(self._formal_charges_or_zero().astype(float))
            elif name == "aromatic":
                arrays.append(self._aromatic_atoms_or_false().astype(float))
            elif name.startswith("element_"):
                symbol = name.removeprefix("element_").upper()
                arrays.append(np.array(
                    [1.0 if (element or "").upper() == symbol else 0.0
                     for element in self.elements],
                    dtype=float,
                ))
            else:  # pragma: no cover - protected by node_feature_names
                raise ValueError(f"unknown node feature {name!r}")
        matrix = np.stack(arrays, axis=1).astype(float) if arrays else np.empty((self.n_atoms, 0))
        return (matrix, names) if return_names else matrix

    def _calculate_angle(self, coords, i, j, k) -> float:
        a = coords[i] - coords[j]
        b = coords[k] - coords[j]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    def _calculate_dihedral(self, coords, a, b, c, d) -> float:
        v0 = coords[a] - coords[b]
        v1 = coords[c] - coords[b]
        v2 = coords[d] - coords[c]
        v1 = v1 / np.linalg.norm(v1)
        v = v0 - np.dot(v0, v1) * v1
        w = v2 - np.dot(v2, v1) * v1
        x = np.dot(v, w)
        y = np.dot(np.cross(v1, v), w)
        return float(np.degrees(np.arctan2(y, x)))

    def _compute_edge_angles(self) -> np.ndarray:
        if self.n_bonds == 0:
            return np.empty(0, dtype=float)
        from collections import defaultdict
        neighbors = defaultdict(list)
        for u, v in self.edges:
            neighbors[u].append(v)
            neighbors[v].append(u)
        angles = []
        for u, v in self.edges:
            edge_angles = []
            for k in neighbors[u]:
                if k != v:
                    edge_angles.append(self._calculate_angle(self.coords, k, u, v))
            for k in neighbors[v]:
                if k != u:
                    edge_angles.append(self._calculate_angle(self.coords, u, v, k))
            angles.append(np.mean(edge_angles) if edge_angles else 0.0)
        return np.array(angles, dtype=float)

    def _compute_edge_dihedrals(self) -> np.ndarray:
        if self.n_bonds == 0:
            return np.empty(0, dtype=float)
        from collections import defaultdict
        neighbors = defaultdict(list)
        for u, v in self.edges:
            neighbors[u].append(v)
            neighbors[v].append(u)
        dihedrals = []
        for u, v in self.edges:
            edge_dihedrals = []
            for a in neighbors[u]:
                if a != v:
                    for b in neighbors[v]:
                        if b != u:
                            edge_dihedrals.append(self._calculate_dihedral(self.coords, a, u, v, b))
            dihedrals.append(np.mean(edge_dihedrals) if edge_dihedrals else 0.0)
        return np.array(dihedrals, dtype=float)

    def edge_features(self, preset: str = "default", *, return_names: bool = False):
        """Return an edge feature matrix for a named preset.

        ``"default"`` is distance-only for backwards compatibility. ``"basic"``
        adds bond order. ``"ml"`` adds an aromatic-bond flag, using explicit
        aromaticity when available and otherwise treating order ``1.5`` as
        aromatic.
        """
        names = edge_feature_names(preset)
        arrays = []
        for name in names:
            if name == "distance":
                arrays.append(np.asarray(self.edge_distances, dtype=float))
            elif name == "bond_order":
                arrays.append(np.asarray(self.edge_types, dtype=float))
            elif name == "aromatic":
                arrays.append(self._aromatic_bonds_or_order().astype(float))
            elif name == "bond_angle":
                arrays.append(self._compute_edge_angles())
            elif name == "dihedral":
                arrays.append(self._compute_edge_dihedrals())
            else:  # pragma: no cover - protected by edge_feature_names
                raise ValueError(f"unknown edge feature {name!r}")
        matrix = np.stack(arrays, axis=1).astype(float) if arrays else np.empty((self.n_bonds, 0))
        return (matrix, names) if return_names else matrix

    def feature_matrices(
        self,
        *,
        node_preset: str = "ml",
        edge_preset: str = "ml",
        elements_to_encode=DEFAULT_GRAPH_ELEMENTS,
        return_names: bool = False,
    ):
        """Return ``(node_features, edge_features)`` for graph ML workflows."""
        x, node_names = self.node_features(
            node_preset,
            elements_to_encode=elements_to_encode,
            return_names=True,
        )
        e, edge_names = self.edge_features(edge_preset, return_names=True)
        if return_names:
            return x, e, node_names, edge_names
        return x, e

    # -- exporters ----------------------------------------------------------

    def to_networkx(self):
        """Return a ``networkx.Graph`` with node and edge attributes."""
        nx = _require("networkx", "networkx", "pip install networkx")
        g = nx.Graph(name=self.name)
        z, m = self.atomic_numbers, self.masses
        for i in range(self.n_atoms):
            attrs = {
                "element": self.elements[i],
                "atomic_number": int(z[i]),
                "mass": float(m[i]),
                "pos": tuple(float(c) for c in self.coords[i]),
            }
            if self.atom_names:
                attrs["atom_name"] = self.atom_names[i]
            if self.resnames:
                attrs["resname"] = self.resnames[i]
            if len(self.resids):
                attrs["resid"] = int(self.resids[i])
            if self.icodes:
                attrs["insertion_code"] = self.icodes[i]
            if self.chains:
                attrs["chain"] = self.chains[i]
            if len(self.resids):
                chain = self.chains[i] if self.chains else ""
                resname = self.resnames[i] if self.resnames else ""
                icode = self.icodes[i] if self.icodes else ""
                attrs["residue_id"] = ResidueId(
                    chain, int(self.resids[i]), icode, resname
                ).label()
            if len(self.formal_charges):
                attrs["formal_charge"] = int(self.formal_charges[i])
            if len(self.aromatic_atoms):
                attrs["aromatic"] = bool(self.aromatic_atoms[i])
            if len(self.virtual_sites):
                attrs["virtual_site"] = bool(self.virtual_sites[i])
            g.add_node(i, **attrs)
        aromatic = self._aromatic_bonds_or_order()
        covalent = self.covalent_edge_flags()
        for edge_idx, ((i, j), dist, btype) in enumerate(
            zip(self.edges, self.edge_distances, self.edge_types)
        ):
            g.add_edge(
                int(i), int(j), distance=float(dist), bond_type=float(btype),
                aromatic=bool(aromatic[edge_idx]),
                covalent=bool(covalent[edge_idx]),
            )
        return g

    def to_pyg_data(
        self,
        node_preset: str = "default",
        edge_preset: str = "default",
        include_self_loops: bool = False,
        include_global_node: bool = False,
        include_pe: Optional[str] = None,
        pe_k: int = 8,
        include_displacement: bool = False,
    ):
        """Return a ``torch_geometric.data.Data`` object.

        Populates ``x`` (node features), ``z`` (atomic numbers), ``pos`` (3D
        coordinates), ``edge_index`` and ``edge_attr``, plus an ``is_covalent``
        edge flag.

        Args:
            node_preset: Node feature preset ("default", "basic", "ml").
            edge_preset: Edge feature preset ("default", "basic", "ml").
            include_self_loops: If True, add (i, i) edges to edge_index.
            include_global_node: If True, add a virtual node connected to all others.
            include_pe: Optional positional encoding to include ("laplacian" or "random_walk").
            pe_k: Dimension of the positional encoding.
            include_displacement: If True, add ``edge_vec`` (E, 3) holding the
                directed relative displacement ``pos[dst] - pos[src]`` for each
                edge (for SchNet/EGNN-style models). Note ``pos`` and
                ``edge_index`` already suffice to derive this on the fly.
        """
        torch = _require("torch", "PyTorch Geometric", "pip install torch torch_geometric")
        Data = _require(
            "torch_geometric.data", "PyTorch Geometric",
            "pip install torch torch_geometric", attr="Data",
        )
        x = self.node_features(node_preset)
        pos = self.coords
        z = self.atomic_numbers
        is_covalent = self._directed_covalent_flags()
        formal_charge = self._formal_charges_or_zero()
        virtual_site = self._virtual_sites_or_false()
        src, dst, dist, order = self._directed_edges()
        e = self._directed_edge_features(edge_preset)

        if include_global_node:
            # Add a virtual node at the end
            g_idx = len(x)
            x_g = x.mean(axis=0, keepdims=True)
            pos_g = pos.mean(axis=0, keepdims=True)
            x = np.concatenate([x, x_g], axis=0)
            pos = np.concatenate([pos, pos_g], axis=0)
            z = np.concatenate([z, [0]])  # 0 for global node
            formal_charge = np.concatenate([formal_charge, [0]])
            virtual_site = np.concatenate([virtual_site, [False]])

            # Connect global node to all other nodes (bidirectional)
            others = np.arange(g_idx)
            g_src = np.concatenate([others, np.full(g_idx, g_idx)])
            g_dst = np.concatenate([np.full(g_idx, g_idx), others])
            src = np.concatenate([src, g_src])
            dst = np.concatenate([dst, g_dst])

            # Pad edge features for global edges (zeros)
            g_e = np.zeros((len(g_src), e.shape[1]))
            e = np.concatenate([e, g_e], axis=0)
            dist = np.concatenate([dist, np.zeros(len(g_src))])
            order = np.concatenate([order, np.zeros(len(g_src))])
            is_covalent = np.concatenate([is_covalent, np.zeros(len(g_src), dtype=bool)])

        if include_self_loops:
            n = len(x)
            s_idx = np.arange(n)
            src = np.concatenate([src, s_idx])
            dst = np.concatenate([dst, s_idx])
            s_e = np.zeros((n, e.shape[1]))
            e = np.concatenate([e, s_e], axis=0)
            dist = np.concatenate([dist, np.zeros(n)])
            order = np.concatenate([order, np.ones(n)])
            is_covalent = np.concatenate([is_covalent, np.zeros(n, dtype=bool)])

        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            node_feature_names=node_feature_names(node_preset),
            z=torch.tensor(z, dtype=torch.long),
            formal_charge=torch.tensor(formal_charge, dtype=torch.long),
            virtual_site=torch.tensor(virtual_site, dtype=torch.bool),
            pos=torch.tensor(pos, dtype=torch.float),
            edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
            edge_attr=torch.tensor(e, dtype=torch.float),
            edge_feature_names=edge_feature_names(edge_preset),
            bond_order=torch.tensor(order, dtype=torch.float),
            is_covalent=torch.tensor(is_covalent, dtype=torch.bool),
            num_nodes=len(x),
        )

        if include_displacement:
            # Directed relative displacement r_dst - r_src for each edge.
            if len(src):
                edge_vec = pos[dst.astype(int)] - pos[src.astype(int)]
            else:
                edge_vec = np.empty((0, 3), dtype=float)
            data.edge_vec = torch.tensor(edge_vec, dtype=torch.float)


        if include_pe == "laplacian":
            pe = self.laplacian_pe(k=pe_k)
            if include_global_node:
                pe = np.concatenate([pe, np.zeros((1, pe_k))], axis=0)
            data.pe = torch.tensor(pe, dtype=torch.float)
        elif include_pe == "random_walk":
            pe = self.random_walk_pe(k=pe_k)
            if include_global_node:
                pe = np.concatenate([pe, np.zeros((1, pe_k))], axis=0)
            data.pe = torch.tensor(pe, dtype=torch.float)

        return data


    def to_dgl_graph(
        self,
        node_preset: str = "default",
        edge_preset: str = "default",
        include_self_loops: bool = False,
        include_global_node: bool = False,
        include_pe: Optional[str] = None,
        pe_k: int = 8,
        include_displacement: bool = False,
    ):
        """Return a ``dgl.DGLGraph`` with node/edge feature tensors.

        Args:
            node_preset: Node feature preset ("default", "basic", "ml").
            edge_preset: Edge feature preset ("default", "basic", "ml").
            include_self_loops: If True, add (i, i) edges.
            include_global_node: If True, add a virtual node connected to all others.
            include_pe: Optional positional encoding to include ("laplacian" or "random_walk").
            pe_k: Dimension of the positional encoding.
            include_displacement: If True, add ``edata["edge_vec"]`` (E, 3) holding
                the directed relative displacement ``pos[dst] - pos[src]``.
        """
        dgl = _require("dgl", "DGL", "pip install dgl")
        torch = _require("torch", "DGL", "pip install dgl torch")

        x = self.node_features(node_preset)
        pos = self.coords
        z = self.atomic_numbers
        formal_charge = self._formal_charges_or_zero()
        virtual_site = self._virtual_sites_or_false()
        src, dst, dist, order = self._directed_edges()
        e = self._directed_edge_features(edge_preset)
        is_covalent = self._directed_covalent_flags()

        if include_global_node:
            g_idx = len(x)
            x_g = x.mean(axis=0, keepdims=True)
            pos_g = pos.mean(axis=0, keepdims=True)
            x = np.concatenate([x, x_g], axis=0)
            pos = np.concatenate([pos, pos_g], axis=0)
            z = np.concatenate([z, [0]])
            formal_charge = np.concatenate([formal_charge, [0]])
            virtual_site = np.concatenate([virtual_site, [False]])

            others = np.arange(g_idx)
            g_src = np.concatenate([others, np.full(g_idx, g_idx)])
            g_dst = np.concatenate([np.full(g_idx, g_idx), others])
            src = np.concatenate([src, g_src])
            dst = np.concatenate([dst, g_dst])

            g_e = np.zeros((len(g_src), e.shape[1]))
            e = np.concatenate([e, g_e], axis=0)
            dist = np.concatenate([dist, np.zeros(len(g_src))])
            order = np.concatenate([order, np.zeros(len(g_src))])
            is_covalent = np.concatenate([is_covalent, np.zeros(len(g_src), dtype=bool)])

        if include_self_loops:
            n = len(x)
            s_idx = np.arange(n)
            src = np.concatenate([src, s_idx])
            dst = np.concatenate([dst, s_idx])
            s_e = np.zeros((n, e.shape[1]))
            e = np.concatenate([e, s_e], axis=0)
            dist = np.concatenate([dist, np.zeros(n)])
            order = np.concatenate([order, np.ones(n)])
            is_covalent = np.concatenate([is_covalent, np.zeros(n, dtype=bool)])

        g = dgl.graph(
            (torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)),
            num_nodes=len(x),
        )
        g.ndata["feat"] = torch.tensor(x, dtype=torch.float)
        g.ndata["z"] = torch.tensor(z, dtype=torch.long)
        g.ndata["formal_charge"] = torch.tensor(formal_charge, dtype=torch.long)
        g.ndata["virtual_site"] = torch.tensor(virtual_site, dtype=torch.bool)
        g.ndata["pos"] = torch.tensor(pos, dtype=torch.float)

        g.edata["feat"] = torch.tensor(e, dtype=torch.float)
        g.edata["distance"] = torch.tensor(dist, dtype=torch.float)
        g.edata["bond_order"] = torch.tensor(order, dtype=torch.float)
        if len(src):
            g.edata["is_covalent"] = torch.tensor(is_covalent, dtype=torch.bool)
        if include_displacement and len(src):
            edge_vec = pos[dst.astype(int)] - pos[src.astype(int)]
            g.edata["edge_vec"] = torch.tensor(edge_vec, dtype=torch.float)

        if include_pe == "laplacian":
            pe = self.laplacian_pe(k=pe_k)
            if include_global_node:
                pe = np.concatenate([pe, np.zeros((1, pe_k))], axis=0)
            g.ndata["pe"] = torch.tensor(pe, dtype=torch.float)
        elif include_pe == "random_walk":
            pe = self.random_walk_pe(k=pe_k)
            if include_global_node:
                pe = np.concatenate([pe, np.zeros((1, pe_k))], axis=0)
            g.ndata["pe"] = torch.tensor(pe, dtype=torch.float)

        return g


    def _directed_edges(self):
        """Edges in both directions: (src, dst, distance) for message passing."""
        if self.n_bonds == 0:
            empty_i = np.empty(0, dtype=int)
            empty_f = np.empty(0, dtype=float)
            return empty_i, empty_i, empty_f, empty_f
        i, j = self.edges[:, 0], self.edges[:, 1]
        src = np.concatenate([i, j])
        dst = np.concatenate([j, i])
        dist = np.concatenate([self.edge_distances, self.edge_distances])
        order = np.concatenate([self.edge_types, self.edge_types])
        return src, dst, dist, order

    def _formal_charges_or_zero(self) -> np.ndarray:
        if len(self.formal_charges):
            return np.asarray(self.formal_charges, dtype=int)
        return np.zeros(self.n_atoms, dtype=int)

    def _aromatic_atoms_or_false(self) -> np.ndarray:
        if len(self.aromatic_atoms):
            return np.asarray(self.aromatic_atoms, dtype=bool)
        return np.zeros(self.n_atoms, dtype=bool)

    def _aromatic_bonds_or_order(self) -> np.ndarray:
        if len(self.aromatic_bonds):
            return np.asarray(self.aromatic_bonds, dtype=bool)
        return np.isclose(np.asarray(self.edge_types, dtype=float), 1.5)

    def _virtual_sites_or_false(self) -> np.ndarray:
        if len(self.virtual_sites):
            return np.asarray(self.virtual_sites, dtype=bool)
        return np.zeros(self.n_atoms, dtype=bool)

    def covalent_edge_flags(self) -> np.ndarray:
        """Per-edge boolean: is this edge a covalent bond? (all True if unset)."""
        if len(self.covalent_edges):
            return np.asarray(self.covalent_edges, dtype=bool)
        return np.ones(self.n_bonds, dtype=bool)

    def _directed_covalent_flags(self) -> np.ndarray:
        """The covalent flag duplicated for both directed-edge halves."""
        if self.n_bonds == 0:
            return np.empty(0, dtype=bool)
        return np.concatenate([self.covalent_edge_flags()] * 2)

    def _directed_edge_features(self, preset: str) -> np.ndarray:
        undirected = self.edge_features(preset)
        if self.n_bonds == 0:
            return undirected
        return np.concatenate([undirected, undirected], axis=0)


@dataclass
class ResidueContactGraph:
    """Residues as nodes and spatial contacts as edges.

    This is a coarse structural graph for protein-scale ML and inspection. Nodes
    carry residue identity, chain, residue id, representative coordinates and
    atom count. Edges carry the residue pair, contact distance and the contact
    construction method (``"ca"``, ``"com"`` or ``"min"``).
    """

    coords: np.ndarray              # (R, 3), residue representative coordinates
    edges: np.ndarray               # (E, 2), i < j
    edge_distances: np.ndarray      # (E,)
    edge_types: list[str]           # contact method per edge
    resnames: list[str]
    resids: np.ndarray
    icodes: list[str]
    chains: list[str]
    residue_sizes: np.ndarray
    labels: list[str] = field(default_factory=list)
    residue_ids: list[ResidueId] = field(default_factory=list)
    cutoff: float = 8.0
    method: str = "ca"
    name: str = ""
    # Optional coarse interaction label per contact edge (see
    # RESIDUE_INTERACTION_LABELS); empty unless built with annotate_interactions.
    edge_interactions: list[str] = field(default_factory=list)

    @property
    def n_residues(self) -> int:
        return len(self.coords)

    @property
    def n_contacts(self) -> int:
        return len(self.edges)

    def interaction_one_hot(self) -> np.ndarray:
        """One-hot encode the per-edge interaction labels as ``(E, n_labels)``.

        Columns follow :data:`RESIDUE_INTERACTION_LABELS`. All-zero rows mean the
        graph was built without ``annotate_interactions``. Useful as a categorical
        edge feature for ML; pair it with :meth:`edge_features`.
        """
        out = np.zeros((self.n_contacts, len(RESIDUE_INTERACTION_LABELS)), dtype=float)
        if not self.edge_interactions:
            return out
        index = {label: k for k, label in enumerate(RESIDUE_INTERACTION_LABELS)}
        for edge, label in enumerate(self.edge_interactions):
            out[edge, index[label]] = 1.0
        return out

    def adjacency_matrix(self, weighted: bool = False) -> np.ndarray:
        """Return the dense adjacency matrix (R, R)."""
        adj = np.zeros((self.n_residues, self.n_residues), dtype=float)
        if self.n_contacts == 0:
            return adj
        i, j = self.edges[:, 0], self.edges[:, 1]
        val = self.edge_distances if weighted else 1.0
        adj[i, j] = val
        adj[j, i] = val
        return adj

    def laplacian_pe(self, k: int = 8) -> np.ndarray:
        """Compute Laplacian Positional Encodings (R, k)."""
        if self.n_residues <= 1:
            return np.zeros((self.n_residues, k))

        try:
            import scipy.sparse as sp
            from scipy.sparse.linalg import eigsh
            has_scipy = True
        except ImportError:
            has_scipy = False

        if has_scipy and self.n_residues > k + 1:
            try:
                row = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
                col = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
                data = np.ones(2 * self.n_contacts, dtype=float)
                adj_sparse = sp.coo_matrix(
                    (data, (row, col)), shape=(self.n_residues, self.n_residues)
                ).tocsr()
                deg = np.array(adj_sparse.sum(axis=1)).flatten()
                with np.errstate(divide="ignore", invalid="ignore"):
                    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
                d_inv_sqrt_mat = sp.diags(deg_inv_sqrt)
                scaled_adj = d_inv_sqrt_mat @ adj_sparse @ d_inv_sqrt_mat
                l_norm_sparse = sp.eye(self.n_residues) - scaled_adj
                # Shift-invert (sigma=0) reliably finds the eigenvalues closest
                # to zero — i.e. the k smallest of the PSD Laplacian. ARPACK's
                # which='SM' mode is notoriously unreliable here and can skip the
                # trivial zero eigenvalue, returning a different set than the
                # dense path.
                eigvals, eigvecs = eigsh(l_norm_sparse, k=k, sigma=0, which='LM')
                idx = np.argsort(eigvals)
                return _sign_stabilize(eigvecs[:, idx])
            except Exception:
                pass

        adj = self.adjacency_matrix()
        deg = adj.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        l_norm = np.eye(self.n_residues) - deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
        eigvals, eigvecs = np.linalg.eigh(l_norm)
        return _sign_stabilize(eigvecs[:, :k])

    def random_walk_pe(self, k: int = 16) -> np.ndarray:
        """Compute Random Walk Structural Encodings (R, k)."""
        if self.n_residues == 0:
            return np.zeros((0, k))

        try:
            import scipy.sparse as sp
            has_scipy = True
        except ImportError:
            has_scipy = False

        if has_scipy:
            try:
                row = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
                col = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
                data = np.ones(2 * self.n_contacts, dtype=float)
                adj_sparse = sp.coo_matrix(
                    (data, (row, col)), shape=(self.n_residues, self.n_residues)
                ).tocsr()
                deg = np.array(adj_sparse.sum(axis=1)).flatten()
                with np.errstate(divide="ignore", invalid="ignore"):
                    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
                d_inv_mat = sp.diags(deg_inv)
                p_sparse = d_inv_mat @ adj_sparse

                pe = np.zeros((self.n_residues, k))
                pk = p_sparse.copy()
                for i in range(k):
                    pe[:, i] = pk.diagonal()
                    if i < k - 1:
                        pk = pk @ p_sparse
                return pe
            except Exception:
                pass

        adj = self.adjacency_matrix()
        deg = adj.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(deg[:, None] > 0, adj / deg[:, None], 0.0)

        pe = np.zeros((self.n_residues, k))
        pk = p.copy()
        for i in range(k):
            pe[:, i] = np.diag(pk)
            if i < k - 1:
                pk = pk @ p
        return pe

    def node_features(
        self,
        preset: str = "default",
        *,
        residue_types_to_encode=DEFAULT_RESIDUE_TYPES,
        return_names: bool = False,
    ):
        """Return a residue-level node feature matrix for a named preset."""
        names = residue_node_feature_names(
            preset, residue_types_to_encode=residue_types_to_encode
        )
        known = {r.upper() for r in residue_types_to_encode if r.upper() != "UNK"}
        arrays = []
        for name in names:
            if name == "residue_size":
                arrays.append(np.asarray(self.residue_sizes, dtype=float))
            elif name.startswith("residue_"):
                code = name.removeprefix("residue_").upper()
                if code == "UNK":
                    arrays.append(np.array(
                        [1.0 if (res or "").upper() not in known else 0.0
                         for res in self.resnames],
                        dtype=float,
                    ))
                else:
                    arrays.append(np.array(
                        [1.0 if (res or "").upper() == code else 0.0
                         for res in self.resnames],
                        dtype=float,
                    ))
            else:  # pragma: no cover - protected by residue_node_feature_names
                raise ValueError(f"unknown residue node feature {name!r}")
        matrix = (
            np.stack(arrays, axis=1).astype(float)
            if arrays else np.empty((self.n_residues, 0))
        )
        return (matrix, names) if return_names else matrix

    def edge_features(self, preset: str = "default", *, return_names: bool = False):
        """Return a residue-contact edge feature matrix for a named preset."""
        names = residue_edge_feature_names(preset)
        arrays = []
        for name in names:
            if name == "distance":
                arrays.append(np.asarray(self.edge_distances, dtype=float))
            elif name.startswith("contact_"):
                method = name.removeprefix("contact_")
                arrays.append(np.array(
                    [1.0 if edge_type == method else 0.0 for edge_type in self.edge_types],
                    dtype=float,
                ))
            else:  # pragma: no cover - protected by residue_edge_feature_names
                raise ValueError(f"unknown residue edge feature {name!r}")
        matrix = (
            np.stack(arrays, axis=1).astype(float)
            if arrays else np.empty((self.n_contacts, 0))
        )
        return (matrix, names) if return_names else matrix

    def feature_matrices(
        self,
        *,
        node_preset: str = "ml",
        edge_preset: str = "ml",
        residue_types_to_encode=DEFAULT_RESIDUE_TYPES,
        return_names: bool = False,
    ):
        """Return ``(node_features, edge_features)`` for residue graph ML."""
        x, node_names = self.node_features(
            node_preset,
            residue_types_to_encode=residue_types_to_encode,
            return_names=True,
        )
        e, edge_names = self.edge_features(edge_preset, return_names=True)
        if return_names:
            return x, e, node_names, edge_names
        return x, e

    def to_networkx(self):
        """Return a ``networkx.Graph`` with residue and contact attributes."""
        nx = _require("networkx", "networkx", "pip install networkx")
        g = nx.Graph(name=self.name, cutoff=float(self.cutoff), method=self.method)
        chains = self.chains or [""] * self.n_residues
        labels = self.labels or [
            _residue_label(chain, resname, int(resid), icode)
            for chain, resname, resid, icode in zip(chains, self.resnames, self.resids, self.icodes)
        ]
        for i in range(self.n_residues):
            g.add_node(
                i,
                label=labels[i],
                resname=self.resnames[i],
                resid=int(self.resids[i]),
                insertion_code=self.icodes[i] if self.icodes else "",
                chain=chains[i],
                residue_id=labels[i],
                residue_size=int(self.residue_sizes[i]),
                pos=tuple(float(c) for c in self.coords[i]),
            )
        interactions = self.edge_interactions
        for edge_idx, ((i, j), dist, edge_type) in enumerate(
            zip(self.edges, self.edge_distances, self.edge_types)
        ):
            attrs = dict(
                distance=float(dist),
                contact_type=edge_type,
                cutoff=float(self.cutoff),
                edge_index=edge_idx,
            )
            if interactions:
                attrs["interaction"] = interactions[edge_idx]
            g.add_edge(int(i), int(j), **attrs)
        return g

    def to_pyg_data(
        self,
        node_preset: str = "default",
        edge_preset: str = "default",
        include_self_loops: bool = False,
        include_global_node: bool = False,
        include_pe: Optional[str] = None,
        pe_k: int = 8,
    ):
        """Return a ``torch_geometric.data.Data`` object for residue contacts.

        Args:
            node_preset: Node feature preset ("default", "ml").
            edge_preset: Edge feature preset ("default", "ml").
            include_self_loops: If True, add (i, i) edges.
            include_global_node: If True, add a virtual node connected to all others.
            include_pe: Optional positional encoding to include ("laplacian" or "random_walk").
            pe_k: Dimension of the positional encoding.
        """
        torch = _require("torch", "PyTorch Geometric", "pip install torch torch_geometric")
        Data = _require(
            "torch_geometric.data", "PyTorch Geometric",
            "pip install torch torch_geometric", attr="Data",
        )
        x = self.node_features(node_preset)
        pos = self.coords
        resids = self.resids
        residue_sizes = self.residue_sizes
        labels = list(self.labels)
        src, dst, dist = self._directed_edges()
        e = self._directed_edge_features(edge_preset)

        if include_global_node:
            g_idx = len(x)
            x_g = x.mean(axis=0, keepdims=True)
            pos_g = pos.mean(axis=0, keepdims=True)
            x = np.concatenate([x, x_g], axis=0)
            pos = np.concatenate([pos, pos_g], axis=0)
            resids = np.concatenate([resids, [0]])
            residue_sizes = np.concatenate([residue_sizes, [0]])
            labels.append("GLOBAL")

            others = np.arange(g_idx)
            g_src = np.concatenate([others, np.full(g_idx, g_idx)])
            g_dst = np.concatenate([np.full(g_idx, g_idx), others])
            src = np.concatenate([src, g_src])
            dst = np.concatenate([dst, g_dst])

            g_e = np.zeros((len(g_src), e.shape[1]))
            e = np.concatenate([e, g_e], axis=0)
            dist = np.concatenate([dist, np.zeros(len(g_src))])

        if include_self_loops:
            n = len(x)
            s_idx = np.arange(n)
            src = np.concatenate([src, s_idx])
            dst = np.concatenate([dst, s_idx])
            s_e = np.zeros((n, e.shape[1]))
            e = np.concatenate([e, s_e], axis=0)
            dist = np.concatenate([dist, np.zeros(n)])

        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            node_feature_names=residue_node_feature_names(node_preset),
            pos=torch.tensor(pos, dtype=torch.float),
            edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
            edge_attr=torch.tensor(e, dtype=torch.float),
            edge_feature_names=residue_edge_feature_names(edge_preset),
            distance=torch.tensor(dist, dtype=torch.float),
            resid=torch.tensor(resids, dtype=torch.long),
            residue_size=torch.tensor(residue_sizes, dtype=torch.long),
            residue_labels=labels,
            num_nodes=len(x),
        )


        if include_pe == "laplacian":
            pe = self.laplacian_pe(k=pe_k)
            if include_global_node:
                pe = np.concatenate([pe, np.zeros((1, pe_k))], axis=0)
            data.pe = torch.tensor(pe, dtype=torch.float)
        elif include_pe == "random_walk":
            pe = self.random_walk_pe(k=pe_k)
            if include_global_node:
                pe = np.concatenate([pe, np.zeros((1, pe_k))], axis=0)
            data.pe = torch.tensor(pe, dtype=torch.float)

        return data


    def to_dgl_graph(
        self,
        node_preset: str = "default",
        edge_preset: str = "default",
        include_self_loops: bool = False,
        include_global_node: bool = False,
        include_pe: Optional[str] = None,
        pe_k: int = 8,
    ):
        """Return a ``dgl.DGLGraph`` with residue/contact feature tensors.

        Args:
            node_preset: Node feature preset ("default", "ml").
            edge_preset: Edge feature preset ("default", "ml").
            include_self_loops: If True, add (i, i) edges.
            include_global_node: If True, add a virtual node connected to all others.
            include_pe: Optional positional encoding to include ("laplacian" or "random_walk").
            pe_k: Dimension of the positional encoding.
        """
        dgl = _require("dgl", "DGL", "pip install dgl")
        torch = _require("torch", "DGL", "pip install dgl torch")

        x = self.node_features(node_preset)
        pos = self.coords
        resids = self.resids
        residue_sizes = self.residue_sizes
        src, dst, dist = self._directed_edges()
        e = self._directed_edge_features(edge_preset)

        if include_global_node:
            g_idx = len(x)
            x_g = x.mean(axis=0, keepdims=True)
            pos_g = pos.mean(axis=0, keepdims=True)
            x = np.concatenate([x, x_g], axis=0)
            pos = np.concatenate([pos, pos_g], axis=0)
            resids = np.concatenate([resids, [0]])
            residue_sizes = np.concatenate([residue_sizes, [0]])

            others = np.arange(g_idx)
            g_src = np.concatenate([others, np.full(g_idx, g_idx)])
            g_dst = np.concatenate([np.full(g_idx, g_idx), others])
            src = np.concatenate([src, g_src])
            dst = np.concatenate([dst, g_dst])

            g_e = np.zeros((len(g_src), e.shape[1]))
            e = np.concatenate([e, g_e], axis=0)
            dist = np.concatenate([dist, np.zeros(len(g_src))])

        if include_self_loops:
            n = len(x)
            s_idx = np.arange(n)
            src = np.concatenate([src, s_idx])
            dst = np.concatenate([dst, s_idx])
            s_e = np.zeros((n, e.shape[1]))
            e = np.concatenate([e, s_e], axis=0)
            dist = np.concatenate([dist, np.zeros(n)])

        g = dgl.graph(
            (torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)),
            num_nodes=len(x),
        )
        g.ndata["feat"] = torch.tensor(x, dtype=torch.float)
        g.ndata["pos"] = torch.tensor(pos, dtype=torch.float)
        g.ndata["resid"] = torch.tensor(resids, dtype=torch.long)
        g.ndata["residue_size"] = torch.tensor(residue_sizes, dtype=torch.long)
        g.edata["feat"] = torch.tensor(e, dtype=torch.float)
        g.edata["distance"] = torch.tensor(dist, dtype=torch.float)


        if include_pe == "laplacian":
            pe = self.laplacian_pe(k=pe_k)
            if include_global_node:
                pe = np.concatenate([pe, np.zeros((1, pe_k))], axis=0)
            g.ndata["pe"] = torch.tensor(pe, dtype=torch.float)
        elif include_pe == "random_walk":
            pe = self.random_walk_pe(k=pe_k)
            if include_global_node:
                pe = np.concatenate([pe, np.zeros((1, pe_k))], axis=0)
            g.ndata["pe"] = torch.tensor(pe, dtype=torch.float)

        return g


    def _directed_edges(self):
        """Edges in both directions for message passing."""
        if self.n_contacts == 0:
            empty_i = np.empty(0, dtype=int)
            empty_f = np.empty(0, dtype=float)
            return empty_i, empty_i, empty_f
        i, j = self.edges[:, 0], self.edges[:, 1]
        src = np.concatenate([i, j])
        dst = np.concatenate([j, i])
        dist = np.concatenate([self.edge_distances, self.edge_distances])
        return src, dst, dist

    def _directed_edge_features(self, preset: str) -> np.ndarray:
        undirected = self.edge_features(preset)
        if self.n_contacts == 0:
            return undirected
        return np.concatenate([undirected, undirected], axis=0)


def node_feature_names(preset: str = "default", *, elements_to_encode=DEFAULT_GRAPH_ELEMENTS):
    """Feature names returned by :meth:`MolecularGraph.node_features`."""
    preset = _normalize_preset(preset, GRAPH_NODE_FEATURE_PRESETS, "node")
    if preset == "default":
        return ["atomic_number", "mass"]
    if preset == "basic":
        return ["atomic_number", "mass", "formal_charge"]
    return (
        [f"element_{symbol.upper()}" for symbol in elements_to_encode]
        + ["atomic_number", "mass", "formal_charge", "aromatic"]
    )


def edge_feature_names(preset: str = "default"):
    """Feature names returned by :meth:`MolecularGraph.edge_features`."""
    preset = _normalize_preset(preset, GRAPH_EDGE_FEATURE_PRESETS, "edge")
    if preset == "default":
        return ["distance"]
    if preset == "basic":
        return ["distance", "bond_order"]
    if preset == "ml":
        return ["distance", "bond_order", "aromatic"]
    return ["distance", "bond_order", "aromatic", "bond_angle", "dihedral"]


def residue_node_feature_names(
    preset: str = "default",
    *,
    residue_types_to_encode=DEFAULT_RESIDUE_TYPES,
):
    """Feature names returned by :meth:`ResidueContactGraph.node_features`."""
    preset = _normalize_preset(preset, RESIDUE_NODE_FEATURE_PRESETS, "residue node")
    if preset == "default":
        return ["residue_size"]
    return [f"residue_{res.upper()}" for res in residue_types_to_encode] + ["residue_size"]


def residue_edge_feature_names(preset: str = "default"):
    """Feature names returned by :meth:`ResidueContactGraph.edge_features`."""
    preset = _normalize_preset(preset, RESIDUE_EDGE_FEATURE_PRESETS, "residue edge")
    if preset == "default":
        return ["distance"]
    return ["distance"] + [f"contact_{method}" for method in CONTACT_GRAPH_METHODS]


def knn_edges(coords: np.ndarray, k: int) -> np.ndarray:
    """Undirected k-nearest-neighbour edges as unique ``(i, j)`` pairs, ``i < j``.

    Each atom is linked to its ``k`` closest neighbours by Euclidean distance.
    The (generally asymmetric) directed neighbour lists are symmetrised by
    union, so an edge survives when *either* endpoint ranks the other among its
    ``k`` nearest — the usual convention for undirected k-NN graphs. ``k`` is
    capped at ``n - 1``. Uses :class:`scipy.spatial.cKDTree` when available and
    falls back to a dense NumPy distance sort otherwise.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if k < 1:
        raise ValueError(f"knn must be >= 1, got {k}")
    if n < 2:
        return np.empty((0, 2), dtype=int)
    k = min(k, n - 1)

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None

    if cKDTree is not None:
        # Query k+1 neighbours: the nearest neighbour of each point is itself.
        _, idx = cKDTree(coords).query(coords, k=k + 1)
        idx = np.atleast_2d(idx)
        src = np.repeat(np.arange(n), idx.shape[1])
        dst = idx.reshape(-1)
    else:
        dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        order = np.argsort(dmat, axis=1)[:, : k + 1]
        src = np.repeat(np.arange(n), order.shape[1])
        dst = order.reshape(-1)

    keep = src != dst
    src, dst = src[keep], dst[keep]
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    return np.unique(np.stack([lo, hi], axis=1), axis=0).astype(int)


def seq_sep_mask(edges, resids, chains, min_seq_sep: int) -> np.ndarray:
    """Keep-mask dropping same-chain edges below a residue-index separation.

    An edge is dropped when both endpoints lie on the same chain and their
    residue ids differ by fewer than ``min_seq_sep`` positions — the standard
    way to filter out trivial local backbone contacts in protein GNNs. Edges
    that bridge two chains are always kept. Residue ids are required; chain
    labels are optional (a structure with no chains is treated as one chain).
    """
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if min_seq_sep <= 0 or len(edges) == 0:
        return np.ones(len(edges), dtype=bool)
    if resids is None or len(resids) == 0:
        raise ValueError(
            "min_seq_sep needs residue ids to measure sequence separation; "
            "none are available on this structure"
        )
    resids = np.asarray(resids)
    i, j = edges[:, 0], edges[:, 1]
    sep = np.abs(resids[i].astype(int) - resids[j].astype(int))
    if chains:
        chains = np.asarray(chains)
        same_chain = chains[i] == chains[j]
    else:
        same_chain = np.ones(len(edges), dtype=bool)
    return ~(same_chain & (sep < min_seq_sep))


def delaunay_edges(coords: np.ndarray) -> np.ndarray:
    """Undirected Delaunay-triangulation edges as unique ``(i, j)`` pairs, ``i < j``.

    Two atoms are linked when they share an edge of the 3D Delaunay
    tetrahedralization — equivalently, when their Voronoi cells share a facet.
    This adapts to local density without any distance threshold, the usual reason
    to prefer it over k-NN or a fixed radius. Like any raw Delaunay graph it does
    introduce some long edges across surface concavities and the convex hull, so
    combine it with ``min_seq_sep`` or a distance cap when those matter.

    Requires SciPy (``pip install "molscope[fast]"``): unlike the k-NN and radius
    modes there is no pure-NumPy fallback for a 3D triangulation. Fewer than four
    atoms yield the complete graph; Qhull's joggle option handles near-degenerate
    (coplanar/collinear) coordinates.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2:
        return np.empty((0, 2), dtype=int)
    if n < 4:
        # A tetrahedralization needs >= 4 non-coplanar points; below that the
        # Delaunay graph degenerates to the complete graph.
        i, j = np.triu_indices(n, k=1)
        return np.stack([i, j], axis=1).astype(int)

    try:
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise ImportError(
            "delaunay edges need SciPy; install it with: pip install 'molscope[fast]'"
        ) from exc

    try:
        simplices = Delaunay(coords).simplices  # (M, 4) tetra
    except Exception:
        # Retry with Qhull's joggle for near-degenerate (coplanar) inputs.
        try:
            simplices = Delaunay(coords, qhull_options="QJ").simplices
        except Exception as exc:  # pragma: no cover - QJ resolves all realistic cases
            raise ValueError(
                f"Delaunay triangulation failed ({exc}); the structure may be "
                "degenerate (collinear or coplanar atoms)"
            ) from exc

    # The six undirected edges of every tetrahedron, deduplicated.
    src = simplices[:, (0, 0, 0, 1, 1, 2)].reshape(-1)
    dst = simplices[:, (1, 2, 3, 2, 3, 3)].reshape(-1)
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    return np.unique(np.stack([lo, hi], axis=1), axis=0).astype(int)


def residue_contact_graph(
    molecule,
    cutoff: float = 8.0,
    method: str = "ca",
    backend: str = "numpy",
    device: Optional[str] = None,
    min_seq_sep: int = 0,
    chain_mode: str = "all",
    annotate_interactions: bool = False,
) -> ResidueContactGraph:
    """Build a residue-level spatial contact graph from a molecule.

    ``method="ca"`` connects residues whose alpha-carbon representatives are
    within ``cutoff``. ``"com"`` uses residue centres of mass. ``"min"`` uses
    closest inter-residue atom distance for edges while keeping centre-of-mass
    coordinates as node positions.

    With ``annotate_interactions=True``, each contact edge gets one coarse
    interaction label (see :data:`RESIDUE_INTERACTION_LABELS`) computed from
    residue chemistry and atom-level geometry of the source structure. These are
    geometric/contact heuristics (disulfide, ligand, salt bridge, covalent,
    hydrophobic, polar, proximity), **not** binding-energy terms; they need
    per-atom names to resolve the chemistry, degrading to covalent/proximity/
    ligand otherwise.
    """
    method = method.lower()
    if method not in CONTACT_GRAPH_METHODS:
        choices = "', '".join(CONTACT_GRAPH_METHODS)
        raise ValueError(f"method must be '{choices}', got {method!r}")
    if backend == "scipy":
        backend = "numpy"

    groups = list(molecule.residue_groups())
    if not groups:
        raise ValueError("residue contact graph needs residue information")

    atom_groups = [idx for idx, *_ in groups]
    residue_ids = [group.residue_id for group in groups]
    resnames = [residue_id.resname for residue_id in residue_ids]
    resids = np.array([residue_id.resid for residue_id in residue_ids], dtype=int)
    icodes = [residue_id.insertion_code for residue_id in residue_ids]
    chains = [residue_id.chain for residue_id in residue_ids]
    residue_sizes = np.array([len(idx) for idx in atom_groups], dtype=int)
    labels = [residue_id.label() for residue_id in residue_ids]

    if method in ("ca", "com"):
        coords = np.array([
            _residue_representative(molecule, idx, method) for idx in atom_groups
        ])
        from .distance import distance_matrix

        distances = distance_matrix(coords, backend=backend, device=device)
    else:
        coords = np.array([
            _residue_representative(molecule, idx, "com") for idx in atom_groups
        ])
        distances = _residue_min_distance_matrix(molecule, groups, backend, device)

    contact_matrix = (distances < cutoff).astype(float)
    np.fill_diagonal(contact_matrix, 0.0)
    from .contactmap import _apply_contact_filters

    contact_matrix = _apply_contact_filters(contact_matrix, chains, min_seq_sep, chain_mode)
    i, j = np.nonzero(np.triu(contact_matrix > 0, k=1))
    edges = (
        np.stack([i, j], axis=1).astype(int)
        if len(i) else np.empty((0, 2), dtype=int)
    )
    edge_distances = distances[i, j].astype(float) if len(i) else np.empty(0, dtype=float)
    edge_interactions = (
        _residue_interaction_labels(molecule, atom_groups, resnames, resids, chains, edges)
        if annotate_interactions else []
    )
    return ResidueContactGraph(
        coords=coords,
        edges=edges,
        edge_distances=edge_distances,
        edge_types=[method] * len(edges),
        resnames=resnames,
        resids=resids,
        icodes=icodes,
        chains=chains,
        residue_sizes=residue_sizes,
        labels=labels,
        residue_ids=residue_ids,
        cutoff=cutoff,
        method=method,
        name=molecule.name,
        edge_interactions=edge_interactions,
    )


def _residue_interaction_labels(molecule, atom_groups, resnames, resids, chains, edges):
    """One coarse interaction label per contact edge (see RESIDUE_INTERACTION_LABELS)."""
    coords = np.asarray(molecule.coords, dtype=float)
    names = molecule.atom_names or []

    def atom_class(idx_list, resname):
        rn = (resname or "").upper()
        anames = [(names[i] if names else "").upper() for i in idx_list]
        return {
            "resname": rn,
            "sg": [i for i, a in zip(idx_list, anames) if a == "SG"],
            "basic": [i for i, a in zip(idx_list, anames)
                      if rn in _BASIC_SIDECHAIN_ATOMS and a in _BASIC_SIDECHAIN_ATOMS[rn]],
            "acidic": [i for i, a in zip(idx_list, anames)
                       if rn in _ACIDIC_SIDECHAIN_ATOMS and a in _ACIDIC_SIDECHAIN_ATOMS[rn]],
            "sidechain": [i for i, a in zip(idx_list, anames) if a not in _BACKBONE_ATOMS],
        }

    info = [atom_class(atom_groups[r], resnames[r]) for r in range(len(atom_groups))]

    def min_dist(ia, ib):
        if not ia or not ib:
            return np.inf
        a, b = coords[ia], coords[ib]
        return float(np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min())

    labels = []
    for u, v in edges:
        pu, pv = info[u], info[v]
        rn_u, rn_v = pu["resname"], pv["resname"]
        adjacent = chains[u] == chains[v] and abs(int(resids[u]) - int(resids[v])) == 1
        ligand_u = rn_u not in _STANDARD_AA and rn_u not in _WATER_RESNAMES
        ligand_v = rn_v not in _STANDARD_AA and rn_v not in _WATER_RESNAMES
        if rn_u == "CYS" and rn_v == "CYS" and min_dist(pu["sg"], pv["sg"]) < _DISULFIDE_CUTOFF:
            label = "disulfide"
        elif ligand_u or ligand_v:
            label = "ligand"
        elif min(min_dist(pu["basic"], pv["acidic"]),
                 min_dist(pu["acidic"], pv["basic"])) < _SALT_BRIDGE_CUTOFF:
            label = "salt_bridge"
        elif adjacent:
            label = "covalent"
        elif (rn_u in _HYDROPHOBIC_AA and rn_v in _HYDROPHOBIC_AA
              and min_dist(pu["sidechain"], pv["sidechain"]) < _SIDECHAIN_CONTACT_CUTOFF):
            label = "hydrophobic"
        elif (rn_u in _POLAR_AA and rn_v in _POLAR_AA
              and min_dist(pu["sidechain"], pv["sidechain"]) < _SIDECHAIN_CONTACT_CUTOFF):
            label = "polar"
        else:
            label = "proximity"
        labels.append(label)
    return labels


def _residue_representative(molecule, idx, method: str) -> np.ndarray:
    if method == "ca" and molecule.atom_names:
        ca = [i for i in idx if molecule.atom_names[i] == "CA"]
        if ca:
            return molecule.coords[ca[0]]
    if method == "com":
        weights = np.array([elements.mass(molecule.elements[i]) for i in idx])
        return (weights[:, None] * molecule.coords[idx]).sum(axis=0) / weights.sum()
    return molecule.coords[idx].mean(axis=0)


def _residue_min_distance_matrix(molecule, groups, backend: str, device: Optional[str]):
    from .distance import distance_matrix

    distances = distance_matrix(molecule.coords, backend=backend, device=device)
    starts = np.array([idx[0] for idx, *_ in groups])
    row_min = np.minimum.reduceat(distances, starts, axis=0)
    return np.minimum.reduceat(row_min, starts, axis=1)


def _residue_label(chain: str, resname: str, resid: int, icode: str = "") -> str:
    return ResidueId(chain, int(resid), icode, resname).label()


def _normalize_preset(preset: str, allowed: tuple[str, ...], label: str) -> str:
    if preset not in allowed:
        choices = "', '".join(allowed)
        raise ValueError(f"unknown {label} feature preset {preset!r}; expected '{choices}'")
    return preset


def _require(module: str, feature: str, hint: str, attr: Optional[str] = None):
    """Import a backend module (or attribute), raising a friendly error if absent."""
    import importlib

    try:
        mod = importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(f"{feature} is required for this export; {hint}") from exc
    return getattr(mod, attr) if attr else mod
