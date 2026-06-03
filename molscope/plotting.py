"""3D visualization of molecules: matplotlib, py3Dmol, and GIF export."""

from __future__ import annotations

import itertools
import warnings
from typing import Optional

import numpy as np

from . import elements


def plot(
    molecule,
    show_bonds: Optional[bool] = None,
    bond_tolerance: float = 1.2,
    color_by: str = "element",
    scale: float = 60.0,
    ax=None,
    show: bool = True,
):
    """Scatter-plot atoms in 3D with an equal aspect ratio.

    ``color_by`` selects the colouring: ``"element"`` (CPK), ``"chain"``,
    ``"residue"`` (categorical palette), or ``"ss"`` (secondary structure, via
    a simplified DSSP). Atom sizes scale with covalent radius.
    Bonds are drawn when ``show_bonds`` is true, or, when ``None``, automatically
    for molecules small enough to infer bonds cheaply. Returns the ``Axes3D``;
    pass ``show=False`` to suppress ``plt.show()``.
    """
    import matplotlib.pyplot as plt  # imported lazily so the core has no GUI dep

    coords = molecule.coords
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")

    colors = _colors(molecule, color_by)
    sizes = np.array([elements.covalent_radius(e) for e in molecule.elements]) * scale
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=sizes, depthshade=True)

    if show_bonds is None:
        show_bonds = len(molecule) <= 2000
    if show_bonds and len(molecule) > 1:
        try:
            for i, j in molecule.bonds(tolerance=bond_tolerance):
                seg = coords[[i, j]]
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="0.5", linewidth=1.0)
        except ValueError as exc:
            warnings.warn(f"skipping bonds: {exc}", stacklevel=2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if molecule.name:
        ax.set_title(molecule.name)
    _set_equal_aspect(ax, coords)

    if show:
        plt.show()
    return ax


def plot_mapping(
    atomistic,
    cg,
    show_assignment: Optional[bool] = None,
    show_bonds: bool = True,
    scale: float = 60.0,
    bead_scale: float = 420.0,
    atom_alpha: float = 0.55,
    ax=None,
    show: bool = True,
    max_legend: int = 14,
):
    """Show how a coarse-grained model ``cg`` maps onto its ``atomistic`` source.

    Atoms are coloured by the bead they were folded into (dropped atoms grey),
    each bead is drawn as a large translucent sphere at its position, thin
    "assignment" lines join atoms to their bead, and the CG bond network is
    drawn between beads. ``cg`` must carry a coarse-graining report (i.e. come
    from :meth:`~molscope.molecule.Molecule.coarse_grain`) and ``atomistic``
    must be the structure it was built from, in the same atom order.

    ``show_assignment`` toggles the atom-to-bead lines (default: on for
    structures small enough to stay legible). Returns the ``Axes3D``; pass
    ``show=False`` to suppress ``plt.show()``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .coarsegrain import bead_label

    report = cg.coarse_grain_report
    n_atoms = len(atomistic)
    atom_bead = np.full(n_atoms, -1, dtype=int)
    for k, bead in enumerate(report.beads):
        for i in bead.atom_indices:
            if not 0 <= i < n_atoms:
                raise ValueError(
                    f"bead {bead.name!r} references atom index {i} but the "
                    f"atomistic molecule has {n_atoms} atoms; pass the structure "
                    "this CG model was built from"
                )
            atom_bead[i] = k

    palette = plt.get_cmap("tab20").colors
    bead_color = [palette[k % len(palette)] for k in range(report.n_beads)]
    virtual_site_flags = (
        np.asarray(cg.virtual_sites, dtype=bool)
        if len(cg.virtual_sites) else np.zeros(len(cg), dtype=bool)
    )
    grey = (0.6, 0.6, 0.6, 1.0)

    if ax is None:
        ax = plt.figure().add_subplot(1, 1, 1, projection="3d")

    a_coords = atomistic.coords
    radii = np.array([elements.covalent_radius(e) for e in atomistic.elements])
    assigned = atom_bead >= 0
    if np.any(assigned):
        ax.scatter(
            *a_coords[assigned].T, s=radii[assigned] * scale,
            c=[bead_color[k] for k in atom_bead[assigned]],
            alpha=atom_alpha, depthshade=True, zorder=1,
        )
    if np.any(~assigned):
        ax.scatter(
            *a_coords[~assigned].T, s=radii[~assigned] * scale,
            color=grey, alpha=0.25, marker="x", depthshade=True, zorder=1,
        )

    if show_assignment is None:
        show_assignment = n_atoms <= 1500
    if show_assignment and np.any(assigned):
        for i in np.flatnonzero(assigned):
            seg = np.vstack([a_coords[i], cg.coords[atom_bead[i]]])
            ax.plot(*seg.T, color=bead_color[atom_bead[i]], linewidth=0.5,
                    alpha=0.35, zorder=0)

    if show_bonds and cg.bond_index is not None:
        for i, j in cg.bond_index:
            seg = cg.coords[[i, j]]
            ax.plot(*seg.T, color="0.25", linewidth=2.0, zorder=2)
    if show_bonds and report.virtual_sites:
        for site in report.virtual_sites:
            for parent in site.parents:
                seg = cg.coords[[parent, site.index]]
                ax.plot(*seg.T, color="0.2", linewidth=1.2, linestyle="--",
                        alpha=0.75, zorder=2)

    real_site_flags = ~virtual_site_flags
    if np.any(real_site_flags):
        real_indices = np.flatnonzero(real_site_flags)
        ax.scatter(
            *cg.coords[real_indices].T,
            s=bead_scale,
            c=[bead_color[i] for i in real_indices],
            edgecolors="0.2",
            linewidths=1.2,
            alpha=0.85,
            depthshade=False,
            zorder=3,
        )
    if np.any(virtual_site_flags):
        v_indices = np.flatnonzero(virtual_site_flags)
        ax.scatter(
            *cg.coords[v_indices].T,
            s=bead_scale * 0.72,
            c="white",
            marker="D",
            edgecolors="0.05",
            linewidths=1.4,
            alpha=0.95,
            depthshade=False,
            zorder=4,
        )

    if report.n_sites <= max_legend:
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=bead_color[k],
                   markeredgecolor="0.2", markersize=9, label=bead_label(report.beads[k], k))
            for k in range(report.n_beads)
        ]
        handles.extend(
            Line2D([0], [0], marker="D", linestyle="", markerfacecolor="white",
                   markeredgecolor="0.05", markersize=7, label=f"{site.name} (virtual)")
            for site in report.virtual_sites
        )
        if report.n_dropped:
            handles.append(Line2D([0], [0], marker="x", linestyle="", color=grey,
                                  markersize=8, label=f"dropped ({report.n_dropped})"))
        ax.legend(handles=handles, loc="upper left", fontsize="small",
                  framealpha=0.9, borderaxespad=0.0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{cg.name or 'coarse-grained'} mapping\n{report.coverage()}")
    _set_equal_aspect(ax, np.vstack([a_coords, cg.coords]))

    if show:
        plt.show()
    return ax


def view(molecule, style: str = "stick", width: int = 480, height: int = 360):
    """Return an interactive py3Dmol viewer (for Jupyter notebooks).

    Requires py3Dmol (``pip install py3Dmol``). ``style`` is any py3Dmol style
    name such as ``"stick"``, ``"sphere"``, ``"line"`` or ``"cartoon"``.
    """
    try:
        import py3Dmol
    except ImportError as exc:  # pragma: no cover - exercised only without py3Dmol
        raise ImportError(
            "view() needs py3Dmol; install it with: pip install py3Dmol"
        ) from exc
    from .io import _molecule_to_pdb_string

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(_molecule_to_pdb_string(molecule), "pdb")
    viewer.setStyle({style: {"colorscheme": "default"}})
    viewer.zoomTo()
    return viewer


def view_mapping(
    atomistic,
    cg,
    *,
    atom_style: str = "stick",
    atom_opacity: float = 0.4,
    bead_radius: float = 1.6,
    bead_opacity: float = 0.9,
    show_bonds: bool = True,
    width: int = 480,
    height: int = 360,
):
    """Overlay an atomistic structure with its coarse-grained beads (py3Dmol).

    Renders ``atomistic`` as a semi-transparent model (``atom_style`` ``"stick"``
    by default, or ``"cartoon"``/``"line"``/``"sphere"``) and draws each bead of
    the coarse-grained model ``cg`` as a solid sphere at its position, so a
    mapping can be inspected directly in a Jupyter notebook. Real beads are
    coloured by the same categorical palette as :func:`plot_mapping`; virtual
    sites are drawn white, and the CG bond network as thin cylinders. ``cg`` must
    carry a coarse-graining report (i.e. come from
    :meth:`~molscope.molecule.Molecule.coarse_grain`) and ``atomistic`` must be
    the structure it was built from, in the same atom order.

    Requires py3Dmol (``pip install "molscope[viz]"``). Returns the py3Dmol
    viewer, which renders as the notebook cell output (or call ``.show()``).
    """
    report = cg.coarse_grain_report             # validates cg is a CG model
    n_atoms = len(atomistic)
    for bead in report.beads:
        for i in bead.atom_indices:
            if not 0 <= i < n_atoms:
                raise ValueError(
                    f"bead {bead.name!r} references atom index {i} but the "
                    f"atomistic molecule has {n_atoms} atoms; pass the structure "
                    "this CG model was built from"
                )

    try:
        import py3Dmol
    except ImportError as exc:  # pragma: no cover - exercised only without py3Dmol
        raise ImportError(
            'view_mapping() needs py3Dmol; install it with: pip install "molscope[viz]"'
        ) from exc
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    from .io import _molecule_to_pdb_string

    palette = [to_hex(c) for c in plt.get_cmap("tab20").colors]
    virtual_flags = (
        np.asarray(cg.virtual_sites, dtype=bool)
        if len(cg.virtual_sites) else np.zeros(len(cg), dtype=bool)
    )

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(_molecule_to_pdb_string(atomistic), "pdb")
    viewer.setStyle({}, {atom_style: {"opacity": float(atom_opacity)}})

    for site in range(len(cg)):
        x, y, z = (float(v) for v in cg.coords[site])
        if virtual_flags[site]:
            viewer.addSphere({
                "center": {"x": x, "y": y, "z": z},
                "radius": float(bead_radius) * 0.8,
                "color": "white", "opacity": float(bead_opacity),
            })
        else:
            viewer.addSphere({
                "center": {"x": x, "y": y, "z": z},
                "radius": float(bead_radius),
                "color": palette[site % len(palette)],
                "opacity": float(bead_opacity),
            })

    if show_bonds and cg.bond_index is not None:
        for i, j in cg.bond_index:
            a, b = cg.coords[int(i)], cg.coords[int(j)]
            viewer.addCylinder({
                "start": {"x": float(a[0]), "y": float(a[1]), "z": float(a[2])},
                "end": {"x": float(b[0]), "y": float(b[1]), "z": float(b[2])},
                "radius": 0.15, "color": "gray",
            })

    viewer.zoomTo()
    return viewer


def spin_gif(molecule, path: str, frames: int = 36, fps: int = 15, **plot_kwargs):
    """Render a spinning 3D view and save it as an animated GIF.

    Rotates a full turn about the vertical axis over ``frames`` steps. Requires
    Pillow (already a matplotlib dependency).
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot(molecule, ax=ax, show=False, **plot_kwargs)

    def update(i):
        ax.view_init(elev=20, azim=i * 360 / frames)
        return ()

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def plot_contact_map(contact_map, ax=None, cmap=None, show: bool = True, label_ticks=False):
    """Draw a :class:`~molscope.contactmap.ContactMap` as a heatmap.

    Booleans render as a binary map; ensemble frequencies render with a colour
    scale and a colourbar. ``label_ticks`` annotates the axes with the residue
    labels (e.g. ``A:LYS8``); it is auto-enabled for small maps and can be forced
    on or off. Returns the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    mat = contact_map.matrix
    freq = contact_map.is_frequency
    is_diff = contact_map.is_difference
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    if is_diff:
        v_max = float(np.abs(mat).max()) or 1.0
        vmin, vmax = -v_max, v_max
        plot_cmap = cmap or "bwr"
    else:
        vmin, vmax = 0.0, 1.0
        plot_cmap = cmap or ("viridis" if freq else "Greys")

    im = ax.imshow(
        mat, origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax,
        cmap=plot_cmap,
    )

    unit = "residue" if contact_map.level == "residue" else "atom"
    labels = contact_map.labels
    if labels and (label_ticks or (label_ticks is False and len(labels) <= 40)):
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_xlabel(f"{unit} index")
        ax.set_ylabel(f"{unit} index")

    if is_diff:
        label = "contact frequency difference" if freq else "contact difference"
        ax.set_title(f"{unit} contact difference map ({contact_map.cutoff} Å)")
    else:
        label = "contact frequency" if freq else f"contact (< {contact_map.cutoff} Å)"
        ax.set_title(f"{unit} contact map ({contact_map.cutoff} Å)")

    ax.figure.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)

    if show:
        plt.show()
    return ax


def plot_stream_analysis(analysis, axes=None, show: bool = True):
    """Plot per-frame timelines from a :class:`~molscope.ensemble.StreamAnalysis`.

    Draws radius of gyration and RMSD-to-first-frame against frame index, plus a
    stacked helix/strand/coil panel when secondary structure was tracked. Returns
    the array of matplotlib ``Axes``; pass ``show=False`` to suppress display.
    """
    import matplotlib.pyplot as plt

    frames = np.arange(analysis.n_frames)
    with_ss = analysis.has_secondary_structure
    n_panels = 3 if with_ss else 2

    if axes is None:
        _, axes = plt.subplots(n_panels, 1, figsize=(6, 2.0 * n_panels), sharex=True)
    axes = np.atleast_1d(axes)

    axes[0].plot(frames, analysis.radius_of_gyration, color="tab:blue")
    axes[0].set_ylabel("Rg (Å)")
    axes[1].plot(frames, analysis.rmsd, color="tab:red")
    axes[1].set_ylabel(f"RMSD to frame 0\n(Å, {analysis.selection})")

    if with_ss:
        axes[2].plot(frames, analysis.helix_fraction, label="helix", color="tab:purple")
        axes[2].plot(frames, analysis.strand_fraction, label="strand", color="tab:green")
        axes[2].plot(frames, analysis.coil_fraction, label="coil", color="tab:gray")
        axes[2].set_ylabel("SS fraction")
        axes[2].set_ylim(0, 1)
        axes[2].legend(loc="upper right", fontsize="small", ncol=3)

    axes[-1].set_xlabel("frame")
    axes[0].set_title(f"Trajectory timeline ({analysis.n_frames} frames)")

    if show:
        plt.show()
    return axes


def plot_distance_matrix(matrix, ax=None, cmap="magma_r", show: bool = True):
    """Draw a dense pairwise distance matrix heatmap."""
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, origin="lower", interpolation="nearest", cmap=cmap)
    ax.set_xlabel("atom index")
    ax.set_ylabel("atom index")
    ax.figure.colorbar(im, ax=ax, label="distance (Å)", fraction=0.046, pad=0.04)
    ax.set_title("pairwise distance matrix")
    if show:
        plt.show()
    return ax


def plot_rmsd_heatmap(matrix, order=None, ax=None, cmap="viridis", show: bool = True):
    """Draw a pairwise-RMSD matrix as a heatmap (angstrom).

    Pass ``order`` (e.g. ``clustering.order``) to reorder rows/columns so
    clusters appear as blocks along the diagonal. Returns the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=float)
    if order is not None:
        order = np.asarray(order)
        matrix = matrix[np.ix_(order, order)]
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, origin="lower", interpolation="nearest", cmap=cmap)
    ax.set_xlabel("model")
    ax.set_ylabel("model")
    ax.figure.colorbar(im, ax=ax, label="RMSD (Å)", fraction=0.046, pad=0.04)
    ax.set_title("pairwise RMSD")
    if show:
        plt.show()
    return ax


def plot_cross_correlation(matrix, ax=None, cmap="RdBu_r", show: bool = True):
    """Draw a dynamical cross-correlation matrix (DCCM) as a heatmap.

    Values run from -1 (anticorrelated) through 0 to +1 (correlated); the
    diverging colormap is centred at zero so coupled regions stand out. Returns
    the matplotlib ``Axes``.
    """
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        matrix, origin="lower", interpolation="nearest",
        cmap=cmap, vmin=-1.0, vmax=1.0,
    )
    ax.set_xlabel("atom index")
    ax.set_ylabel("atom index")
    ax.figure.colorbar(im, ax=ax, label="correlation", fraction=0.046, pad=0.04)
    ax.set_title("dynamical cross-correlation")
    if show:
        plt.show()
    return ax


_SS_LABELS = {
    "H": "alpha-helix", "G": "3-10 helix", "I": "pi-helix",
    "E": "beta-strand", "B": "beta-bridge", "T": "turn", "S": "bend", "-": "coil",
}

# Schematic favoured regions (phi_min, psi_min, width, height, label). These are
# rough teaching guides, NOT statistically-derived Ramachandran density contours.
_RAMA_REGIONS = [
    (-150, -77, 110, 94, "αR"),    # right-handed alpha / 3-10 helix basin
    (-180, 90, 130, 90, "β"),       # extended / beta-sheet
    (35, 20, 60, 75, "αL"),         # left-handed alpha
]


def plot_ramachandran(molecule, ax=None, color_by: str = "ss", show: bool = True,
                      regions: bool = True):
    """Ramachandran plot (phi vs psi backbone torsions) for a protein.

    Scatters each residue's ``(phi, psi)`` on a ``[-180, 180]`` grid from
    :meth:`~molscope.molecule.Molecule.backbone_torsions`. ``color_by="ss"``
    colours points by simplified-DSSP secondary-structure class (see
    :data:`molscope.dssp.SS_COLORS`); ``color_by=None`` or any Matplotlib colour
    draws a single colour. ``regions=True`` shades schematic guide boxes for the
    right-handed alpha, beta and left-handed alpha regions — approximate teaching
    aids, not statistically-derived density contours. Residues whose phi or psi
    is undefined (chain ends and breaks) are skipped. Returns the ``Axes``.
    """
    import matplotlib.pyplot as plt

    from . import dssp

    try:
        tor = molecule.backbone_torsions()
    except ValueError as exc:
        raise ValueError(
            "Ramachandran plot needs a protein backbone (per-atom names and "
            "residue ids); read the structure from PDB or mmCIF"
        ) from exc
    phi = np.asarray(tor.phi, dtype=float)
    psi = np.asarray(tor.psi, dtype=float)
    defined = ~np.isnan(phi) & ~np.isnan(psi)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    if regions:
        from matplotlib.patches import Rectangle
        for x, y, w, h, label in _RAMA_REGIONS:
            ax.add_patch(Rectangle((x, y), w, h, facecolor="#cfe8ff",
                                   edgecolor="none", alpha=0.5, zorder=1))
            ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                    color="0.45", fontsize="small", zorder=2)

    handles = []
    if color_by == "ss":
        codes = _residue_ss_codes(molecule, tor)[defined]
        colors = [dssp.SS_COLORS.get(c, dssp.SS_COLORS["-"]) for c in codes]
        from matplotlib.lines import Line2D
        present = [c for c in dssp.SS_COLORS if c in set(codes)]
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=dssp.SS_COLORS[c],
                   markeredgecolor="0.3", markersize=7, label=_SS_LABELS.get(c, c))
            for c in present
        ]
    else:
        colors = color_by if color_by is not None else "#1f77b4"

    ax.scatter(phi[defined], psi[defined], c=colors, s=14,
               edgecolors="0.3", linewidths=0.3, zorder=3)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(range(-180, 181, 90))
    ax.set_yticks(range(-180, 181, 90))
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel("φ (deg)")
    ax.set_ylabel("ψ (deg)")
    ax.set_aspect("equal")
    ax.set_title(f"Ramachandran: {molecule.name or 'structure'}")
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize="small", framealpha=0.9)
    if show:
        plt.show()
    return ax


def _residue_ss_codes(molecule, torsions) -> np.ndarray:
    """Per-residue SS code aligned to the torsion residues (coil where missing)."""
    try:
        ss = molecule.secondary_structure()
    except Exception:  # pragma: no cover - non-protein / backbone-less structure
        return np.full(len(torsions), "-")
    ss_icodes = ss.icodes if ss.icodes is not None else [""] * len(ss)
    lookup = {
        (chain, int(resid), icode or ""): code
        for chain, resid, icode, code in zip(ss.chains, ss.resids, ss_icodes, ss.codes)
    }
    tor_icodes = torsions.icodes if torsions.icodes is not None else [""] * len(torsions)
    return np.array([
        lookup.get((chain, int(resid), icode or ""), "-")
        for chain, resid, icode in zip(torsions.chains, torsions.resids, tor_icodes)
    ])


def _colors(molecule, color_by: str):
    if color_by == "element":
        return [elements.color(e) for e in molecule.elements]
    if color_by == "ss":
        from . import dssp

        return [dssp.SS_COLORS[c] for c in dssp.per_atom_ss(molecule)]
    if color_by == "chain":
        keys = molecule.chains
    elif color_by == "residue":
        keys = [rid.label() for rid in molecule.residue_ids] if len(molecule.resids) else []
    else:
        raise ValueError(f"unknown color_by {color_by!r}")
    if not keys:
        raise ValueError(f"no {color_by} information to colour by")
    return _categorical_colors(keys)


def _categorical_colors(keys):
    import matplotlib.pyplot as plt

    palette = plt.get_cmap("tab20").colors
    cycle = {}
    wheel = itertools.cycle(palette)
    return [cycle.setdefault(k, next(wheel)) for k in keys]


def _set_equal_aspect(ax, coords: np.ndarray) -> None:
    """Force equal scaling on all axes so the molecule isn't distorted."""
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    centers = (maxs + mins) / 2
    radius = (maxs - mins).max() / 2 or 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
