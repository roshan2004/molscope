"""Command-line entry point for MolScope.

Supports viewing single structures, batch analysis to CSV, and batch graph export.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

from .coarsegrain import COARSE_GRAIN_MAPPINGS
from .io import fetch, read

_SELECTION_KEYS = {
    "element", "chain", "resname", "atom_name", "resid", "icode", "residue_id", "hetero",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="molscope",
        description="Lightweight molecular structure analysis and ML tools.",
    )
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # -- VIEW subcommand ---------------------------------------------------
    view_parser = subparsers.add_parser(
        "view", help="visualise a single structure (default)"
    )
    src = view_parser.add_mutually_exclusive_group(required=True)
    src.add_argument("file", nargs="?", help="path to a structure file")
    src.add_argument("--fetch", metavar="PDBID", help="download from RCSB by id")

    view_parser.add_argument(
        "--select", metavar="SPEC", action="append",
        help=(
            "atom selection; repeat or combine with 'and', e.g. "
            "'chain=A and atom_name=CA'"
        ),
    )
    view_parser.add_argument(
        "--color-by", choices=["element", "chain", "residue"], default="element",
    )
    view_parser.add_argument("--center", action="store_true", help="center at origin")
    view_parser.add_argument(
        "--translate", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
        help="shift atoms by this vector",
    )
    view_parser.add_argument(
        "--rotate", nargs=2, metavar=("AXIS", "DEG"),
        help="rotate about AXIS (x/y/z) by DEG degrees",
    )
    bonds = view_parser.add_mutually_exclusive_group()
    bonds.add_argument("--bonds", dest="bonds", action="store_true", help="force bonds")
    bonds.add_argument("--no-bonds", dest="bonds", action="store_false", help="hide bonds")
    view_parser.set_defaults(bonds=None)

    view_parser.add_argument("--save", metavar="PATH", help="save figure to file")
    view_parser.add_argument("--gif", metavar="PATH", help="save spinning GIF")


    # -- ANALYZE subcommand ------------------------------------------------
    analyze_parser = subparsers.add_parser(
        "analyze", help="batch compute molecular descriptors"
    )
    analyze_parser.add_argument("files", nargs="+", help="files or glob patterns")
    analyze_parser.add_argument("--out", "-o", required=True, help="output CSV file")
    analyze_parser.add_argument(
        "--preset", choices=["native-basic", "native-3d", "rdkit-basic"],
        default="native-basic", help="descriptor preset"
    )
    analyze_parser.add_argument("--jobs", "-j", type=int, default=1, help="parallel jobs")

    # -- BINDING-SITE subcommand ------------------------------------------
    binding_parser = subparsers.add_parser(
        "binding-site", help="write ligand binding-site residue contacts to CSV"
    )
    src = binding_parser.add_mutually_exclusive_group(required=True)
    src.add_argument("file", nargs="?", help="path to a protein-ligand structure file")
    src.add_argument("--fetch", metavar="PDBID", help="download from RCSB by id")
    binding_parser.add_argument("--out", "-o", required=True, help="output residue CSV file")
    binding_parser.add_argument(
        "--cutoff", type=float, default=4.5,
        help="protein-ligand atom contact cutoff in angstrom"
    )
    binding_parser.add_argument(
        "--ligand",
        help="ligand residue name, or chain:resid[:icode] for a specific HETATM group",
    )
    binding_parser.add_argument(
        "--descriptors-out",
        help="optional one-row CSV of pocket-basic descriptors",
    )

    # -- EXPORT subcommand -------------------------------------------------
    export_parser = subparsers.add_parser(
        "export", help="batch export molecular graphs for ML"
    )
    export_parser.add_argument("files", nargs="+", help="files or glob patterns")
    export_parser.add_argument(
        "--to", choices=["pyg", "dgl", "nx"], required=True, help="target format"
    )
    export_parser.add_argument(
        "--knn", type=int, metavar="K",
        help="build edges from each atom's K nearest neighbours instead of bonds",
    )
    export_parser.add_argument(
        "--radius", type=float, metavar="R",
        help="build edges between all atom pairs within R angstrom instead of bonds",
    )
    export_parser.add_argument(
        "--delaunay", action="store_true",
        help="build edges from the Delaunay triangulation instead of bonds (needs SciPy)",
    )
    export_parser.add_argument(
        "--min-seq-sep", type=int, default=0, metavar="N",
        help="drop same-chain edges with residue-id separation below N (needs residue ids)",
    )
    export_parser.add_argument("--self-loops", action="store_true", help="add (i, i) edges")
    export_parser.add_argument("--global-node", action="store_true", help="add virtual master node")
    export_parser.add_argument(
        "--pe", choices=["laplacian", "random_walk"], help="add positional encodings"
    )
    export_parser.add_argument("--pe-k", type=int, default=8, help="PE dimension")
    export_parser.add_argument("--out-dir", "-o", required=True, help="output directory")

    export_parser.add_argument("--jobs", "-j", type=int, default=1, help="parallel jobs")

    # -- SELECT subcommand -------------------------------------------------
    select_parser = subparsers.add_parser(
        "select", help="pick a diverse subset from a molecule table (CSV/XLSX)"
    )
    select_parser.add_argument("file", help="input table (.csv, .tsv or .xlsx)")
    select_parser.add_argument(
        "--num", "-n", type=int, required=True, help="number of molecules to select"
    )
    select_parser.add_argument(
        "--out", "-o", help="write the selection to this .csv/.xlsx (default: print a summary)"
    )
    select_parser.add_argument(
        "--descriptor-cols", nargs="+", metavar="COL",
        help="existing numeric columns to select on, e.g. --descriptor-cols MW ALogP",
    )
    select_parser.add_argument(
        "--smiles-col", metavar="COL",
        help="column holding SMILES (use with --compute-descriptors)",
    )
    select_parser.add_argument(
        "--compute-descriptors", action="store_true",
        help="compute RDKit descriptors from --smiles-col and select on them",
    )
    select_parser.add_argument(
        "--rdkit-descriptors", nargs="+", metavar="NAME",
        help="which RDKit descriptors to compute (default: MolWt MolLogP TPSA ...)",
    )
    select_parser.add_argument(
        "--no-standardize", dest="standardize", action="store_false",
        help="select on raw descriptors instead of z-scored ones",
    )
    select_parser.set_defaults(standardize=True)

    # -- PREPARE subcommand ------------------------------------------------
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="build ML-ready train/validation/test splits from a table or SDF",
    )
    prepare_parser.add_argument("file", help="input dataset (.csv, .tsv, .xlsx or .sdf)")
    prepare_parser.add_argument(
        "--out-dir", "-o", default="prepared",
        help="output directory for splits, descriptors and report (default: prepared/)",
    )
    prepare_parser.add_argument(
        "--split", choices=["random", "diversity", "scaffold"], default="random",
        help="split strategy (scaffold needs RDKit)",
    )
    prepare_parser.add_argument(
        "--test", type=float, default=0.1, help="test fraction (default: 0.1)"
    )
    prepare_parser.add_argument(
        "--val", type=float, default=0.1, help="validation fraction (default: 0.1)"
    )
    prepare_parser.add_argument("--seed", type=int, default=0, help="random seed")
    prepare_parser.add_argument(
        "--smiles-col", metavar="COL",
        help="column holding SMILES (required for scaffold/dedup/descriptors on a table)",
    )
    prepare_parser.add_argument(
        "--descriptor-cols", nargs="+", metavar="COL",
        help="existing numeric columns to use for a diversity split",
    )
    prepare_parser.add_argument(
        "--compute-descriptors", action="store_true",
        help="compute RDKit descriptors from --smiles-col (needs RDKit)",
    )
    prepare_parser.add_argument(
        "--rdkit-descriptors", nargs="+", metavar="NAME",
        help="which RDKit descriptors to compute (default: MolWt MolLogP TPSA ...)",
    )
    prepare_parser.add_argument(
        "--dedup", choices=["none", "exact", "canonical"], default="none",
        help="drop duplicate molecules (canonical needs RDKit)",
    )
    prepare_parser.add_argument(
        "--fingerprints", action="store_true",
        help="add a Morgan fingerprint column (needs RDKit)",
    )
    prepare_parser.add_argument(
        "--protonation", choices=["none", "pka"], default="none",
        help="pKa-aware protonation of SMILES before featurising (pka needs Dimorphite-DL)",
    )
    prepare_parser.add_argument(
        "--ph", type=float, default=7.0,
        help="target pH for --protonation pka (default: 7.0)",
    )
    prepare_parser.add_argument(
        "--no-standardize", dest="standardize", action="store_false",
        help="diversity split on raw descriptors instead of z-scored ones",
    )
    prepare_parser.add_argument(
        "--no-figure", dest="figure", action="store_false",
        help="skip writing report.png",
    )
    prepare_parser.set_defaults(standardize=True, figure=True)

    # -- DOCK-SUMMARY subcommand -------------------------------------------
    dock_summary_parser = subparsers.add_parser(
        "dock-summary",
        help="rank docking poses from an SDF and write summary/top-hit tables",
    )
    dock_summary_parser.add_argument("file", help="docking output SDF (one record per pose)")
    dock_summary_parser.add_argument(
        "--score-field", metavar="TAG",
        help="SDF data field holding the score (auto-detected if omitted)",
    )
    dock_summary_parser.add_argument(
        "--out-dir", "-o", default=".", help="output directory (default: current)",
    )
    dock_summary_parser.add_argument(
        "--top", type=int, default=10, metavar="N", help="rows in top_hits.csv (default: 10)",
    )
    direction = dock_summary_parser.add_mutually_exclusive_group()
    direction.add_argument(
        "--higher-is-better", dest="higher", action="store_true",
        help="treat a larger score as a better hit (e.g. CNNscore)",
    )
    direction.add_argument(
        "--lower-is-better", dest="higher", action="store_false",
        help="treat a smaller score as a better hit (e.g. Vina affinity)",
    )
    dock_summary_parser.add_argument(
        "--no-smiles", dest="smiles", action="store_false",
        help="skip SMILES perception (otherwise needs RDKit; blank without it)",
    )
    dock_summary_parser.add_argument(
        "--no-figure", dest="figure", action="store_false", help="skip score_distribution.png",
    )
    dock_summary_parser.add_argument(
        "--best-pose-per-ligand", dest="best_pose", action="store_true",
        help="collapse multiple poses of the same compound to the single best pose (default)",
    )
    dock_summary_parser.add_argument(
        "--no-best-pose-per-ligand", dest="best_pose", action="store_false",
        help="keep all poses for each compound",
    )
    dock_summary_parser.set_defaults(higher=None, smiles=True, figure=True, best_pose=True)

    # -- DOCK-DIVERSE subcommand -------------------------------------------
    dock_diverse_parser = subparsers.add_parser(
        "dock-diverse",
        help="pick a diverse subset of top docking hits (Tanimoto clustering; needs RDKit)",
    )
    dock_diverse_parser.add_argument("file", help="docking output SDF (one record per pose)")
    dock_diverse_parser.add_argument("--score-field", metavar="TAG", help="score data field")
    dock_diverse_parser.add_argument(
        "--top", type=int, default=500, metavar="N",
        help="rank, then cluster the best N (default: 500)",
    )
    dock_diverse_parser.add_argument(
        "--select", type=int, default=50, metavar="N",
        help="diverse representatives to keep (default: 50)",
    )
    dock_diverse_parser.add_argument(
        "--threshold", type=float, default=0.7, metavar="T",
        help="Tanimoto similarity cutoff for clustering (default: 0.7)",
    )
    dock_diverse_parser.add_argument("--out-dir", "-o", default=".", help="output directory")
    ddir = dock_diverse_parser.add_mutually_exclusive_group()
    ddir.add_argument("--higher-is-better", dest="higher", action="store_true")
    ddir.add_argument("--lower-is-better", dest="higher", action="store_false")
    dock_diverse_parser.set_defaults(higher=None)

    # -- DOCK-RANK subcommand ----------------------------------------------
    dock_rank_parser = subparsers.add_parser(
        "dock-rank",
        help="consensus-rank hits across one or more scored SDFs (transparent rank aggregation)",
    )
    dock_rank_parser.add_argument("files", nargs="+", help="one or more scored SDF files")
    dock_rank_parser.add_argument(
        "--method", choices=["consensus"], default="consensus",
        help="ranking method (currently: consensus = mean rank across score fields)",
    )
    dock_rank_parser.add_argument(
        "--score-fields", nargs="+", metavar="TAG",
        help="score fields to aggregate (auto-detected from known fields if omitted)",
    )
    dock_rank_parser.add_argument(
        "--key", choices=["name", "smiles"], default="name",
        help="join molecules across files by this key (smiles needs RDKit)",
    )
    dock_rank_parser.add_argument(
        "--higher-is-better", nargs="+", metavar="TAG", default=None,
        help="score fields where a larger value is better (overrides the defaults)",
    )
    dock_rank_parser.add_argument(
        "--lower-is-better", nargs="+", metavar="TAG", default=None,
        help="score fields where a smaller value is better (overrides the defaults)",
    )
    dock_rank_parser.add_argument(
        "--mw-max", type=float, metavar="DA",
        help="drop hits above this molecular weight (needs RDKit)",
    )
    dock_rank_parser.add_argument(
        "--logp-max", type=float, metavar="X",
        help="drop hits above this cLogP (needs RDKit)",
    )
    dock_rank_parser.add_argument(
        "--out", "-o", default="dock_ranking.csv",
        help="output CSV path (default: dock_ranking.csv)",
    )

    # -- DOCK-REPORT subcommand --------------------------------------------
    dock_report_parser = subparsers.add_parser(
        "dock-report",
        help="build a self-contained HTML triage report (table, histogram, clusters)",
    )
    dock_report_parser.add_argument("file", help="docking output SDF (one record per pose)")
    dock_report_parser.add_argument("--score-field", metavar="TAG", help="score data field")
    dock_report_parser.add_argument("--out-dir", "-o", default=".", help="output directory")
    dock_report_parser.add_argument(
        "--top", type=int, default=50, metavar="N", help="hits shown in the table (default: 50)",
    )
    dock_report_parser.add_argument(
        "--select", type=int, default=20, metavar="N",
        help="diverse cluster representatives to show (default: 20; needs RDKit)",
    )
    dock_report_parser.add_argument(
        "--threshold", type=float, default=0.7, metavar="T",
        help="Tanimoto similarity cutoff for clustering (default: 0.7)",
    )
    dock_report_parser.add_argument(
        "--export-poses", type=int, default=20, metavar="N",
        help="top poses written to top_poses.sdf for PyMOL/ChimeraX/Mol* (default: 20)",
    )
    dock_report_parser.add_argument(
        "--no-clusters", dest="clusters", action="store_false",
        help="skip the diverse-representatives section",
    )
    rdir = dock_report_parser.add_mutually_exclusive_group()
    rdir.add_argument("--higher-is-better", dest="higher", action="store_true")
    rdir.add_argument("--lower-is-better", dest="higher", action="store_false")
    dock_report_parser.add_argument(
        "--best-pose-per-ligand", dest="best_pose", action="store_true",
        help="collapse multiple poses of the same compound to the single best pose (default)",
    )
    dock_report_parser.add_argument(
        "--no-best-pose-per-ligand", dest="best_pose", action="store_false",
        help="keep all poses for each compound",
    )
    dock_report_parser.set_defaults(higher=None, clusters=True, best_pose=True)

    # -- STRUCTURE-REPORT subcommand ---------------------------------------
    qc_parser = subparsers.add_parser(
        "structure-report",
        help="check whether a structure is ML-ready (gaps, missing atoms, charge, ...)",
    )
    src = qc_parser.add_mutually_exclusive_group(required=True)
    src.add_argument("file", nargs="?", help="path to a structure file")
    src.add_argument("--fetch", metavar="PDBID", help="download from RCSB by id")
    qc_parser.add_argument(
        "--protonation", choices=["none", "standard", "pka"], default="standard",
        help="net-charge model (pka needs PROPKA; default: standard)",
    )
    qc_parser.add_argument("--ph", type=float, default=7.4, help="pH for --protonation pka")
    qc_parser.add_argument("--json", action="store_true", help="print the full JSON report")
    qc_parser.add_argument(
        "--out", "-o", metavar="PATH", help="write a Markdown report to PATH",
    )

    # -- QC subcommand -----------------------------------------------------
    qc_quality_parser = subparsers.add_parser(
        "qc",
        help="lightweight structure-quality report (atoms, chains, ligands, "
        "metadata, elements, bonds, altLoc, CIF/PDB warnings)",
    )
    qc_src = qc_quality_parser.add_mutually_exclusive_group(required=True)
    qc_src.add_argument("file", nargs="?", help="path to a structure file")
    qc_src.add_argument("--fetch", metavar="PDBID", help="download from RCSB by id")
    qc_quality_parser.add_argument(
        "--json", action="store_true", help="print the full JSON report",
    )
    qc_quality_parser.add_argument(
        "--out", "-o", metavar="PATH", help="write a Markdown report to PATH",
    )

    # -- COARSE-GRAIN subcommand -------------------------------------------
    cg_parser = subparsers.add_parser(
        "coarse-grain",
        help="map a structure onto coarse-grained beads and write a coordinate file",
    )
    cg_src = cg_parser.add_mutually_exclusive_group(required=True)
    cg_src.add_argument("file", nargs="?", help="path to a structure file")
    cg_src.add_argument("--fetch", metavar="PDBID", help="download from RCSB by id")
    cg_parser.add_argument(
        "--mapping", choices=list(COARSE_GRAIN_MAPPINGS),
        default="residue_com", help="bead mapping (default: residue_com)",
    )
    cg_parser.add_argument(
        "--out", "-o", metavar="PATH",
        help="write the bead model to PATH (.pdb/.cif/.xyz; format by extension)",
    )

    # -- PRESETS subcommand ------------------------------------------------
    from .presets import CATEGORIES

    presets_parser = subparsers.add_parser(
        "presets",
        help="list the available feature/mapping presets (descriptors, graph, "
        "coarse-grain) and what each one produces",
    )
    presets_parser.add_argument(
        "category", nargs="?", choices=list(CATEGORIES),
        help="show only one category (default: all)",
    )
    presets_parser.add_argument(
        "--features", action="store_true",
        help="also print the full feature-name list each preset expands to",
    )
    presets_parser.add_argument("--json", action="store_true", help="print the full JSON catalogue")

    # Default to 'view' if no subcommand provided
    if argv is None:
        argv = sys.argv[1:]
    argv = _default_to_view(argv, subparsers.choices)

    args = parser.parse_args(argv)

    if args.command == "view":
        return _run_view(args)
    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "binding-site":
        return _run_binding_site(args)
    if args.command == "export":
        return _run_export(args)
    if args.command == "select":
        return _run_select(args)
    if args.command == "prepare":
        return _run_prepare(args)
    if args.command == "dock-summary":
        return _run_dock_summary(args)
    if args.command == "dock-diverse":
        return _run_dock_diverse(args)
    if args.command == "dock-rank":
        return _run_dock_rank(args)
    if args.command == "dock-report":
        return _run_dock_report(args)
    if args.command == "structure-report":
        return _run_structure_report(args)
    if args.command == "qc":
        return _run_qc(args)
    if args.command == "presets":
        return _run_presets(args)
    if args.command == "coarse-grain":
        return _run_coarse_grain(args)

    return 0


def _default_to_view(argv, subcommands) -> list[str]:
    if not argv:
        return ["view"]
    if argv[0] in subcommands or argv[0] in {"-h", "--help"}:
        return list(argv)
    return ["view"] + list(argv)


def _run_view(args: argparse.Namespace) -> int:
    mol = fetch(args.fetch) if args.fetch else read(args.file)
    if args.select:
        try:
            selection = _parse_selection(args.select)
        except ValueError as e:
            print(f"Invalid --select: {e}", file=sys.stderr)
            return 2
        try:
            mol = mol.select(**selection)
        except ValueError as e:
            print(f"Selection failed: {e}", file=sys.stderr)
            return 2
    if args.center:
        mol = mol.centered()
    if args.translate:
        mol = mol.translate(args.translate)
    if args.rotate:
        mol = mol.rotate(args.rotate[0], float(args.rotate[1]))

    print(mol.summary())

    if args.gif:
        from .plotting import spin_gif
        spin_gif(mol, args.gif, color_by=args.color_by, show_bonds=args.bonds)
        print(f"saved {args.gif}")
        return 0

    show = args.save is None
    ax = mol.plot(color_by=args.color_by, show_bonds=args.bonds, show=show)
    if args.save:
        ax.figure.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"saved {args.save}")
    return 0


def _run_select(args: argparse.Namespace) -> int:
    from .library import read_table, select_diverse, smiles_descriptors

    if args.num <= 0:
        print("--num must be a positive integer", file=sys.stderr)
        return 2
    if args.compute_descriptors and not args.smiles_col:
        print("--compute-descriptors requires --smiles-col", file=sys.stderr)
        return 2
    if not args.compute_descriptors and not args.descriptor_cols:
        print(
            "provide --descriptor-cols COL [COL ...], or "
            "--compute-descriptors --smiles-col COL",
            file=sys.stderr,
        )
        return 2

    try:
        table = read_table(args.file)
    except (OSError, ValueError, ImportError) as exc:
        print(f"could not read {args.file}: {exc}", file=sys.stderr)
        return 2

    try:
        if args.compute_descriptors:
            smiles = table.column(args.smiles_col)
            matrix, names = smiles_descriptors(smiles, names=args.rdkit_descriptors)
            table = table.with_columns(names, matrix)
        else:
            names = list(args.descriptor_cols)
            matrix = table.numeric_matrix(names)
    except (KeyError, ValueError, ImportError) as exc:
        print(f"descriptor error: {exc}", file=sys.stderr)
        return 2

    try:
        indices = select_diverse(matrix, args.num, standardize=args.standardize)
    except ValueError as exc:
        print(f"selection error: {exc}", file=sys.stderr)
        return 2

    selection = table.select_rows(indices)
    print(
        f"selected {len(selection)} of {len(table)} molecules "
        f"on {', '.join(names)} (diverse MaxMin)"
    )
    if len(selection) < args.num:
        print(
            f"note: only {len(selection)} rows had complete descriptors, "
            f"fewer than the requested {args.num}",
            file=sys.stderr,
        )

    if args.out:
        try:
            selection.write(args.out)
        except (OSError, ValueError, ImportError) as exc:
            print(f"could not write {args.out}: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {args.out}")
    return 0


def _run_prepare(args: argparse.Namespace) -> int:
    from .prepare import prepare_dataset

    if not 0 <= args.test < 1 or not 0 <= args.val < 1 or args.test + args.val >= 1:
        print("--test and --val must be in [0, 1) and sum to < 1", file=sys.stderr)
        return 2

    try:
        dataset = prepare_dataset(
            args.file,
            smiles_col=args.smiles_col,
            descriptor_cols=args.descriptor_cols,
            compute_descriptors=args.compute_descriptors,
            rdkit_descriptors=args.rdkit_descriptors,
            split=args.split,
            test=args.test,
            val=args.val,
            seed=args.seed,
            standardize=args.standardize,
            dedup=args.dedup,
            fingerprints=args.fingerprints,
            protonation=args.protonation,
            ph=args.ph,
        )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"prepare failed: {exc}", file=sys.stderr)
        return 2

    try:
        written = dataset.write(args.out_dir, make_figure=args.figure)
    except (OSError, ValueError, ImportError) as exc:
        print(f"could not write to {args.out_dir}: {exc}", file=sys.stderr)
        return 2

    sizes = dataset.split.sizes
    print(
        f"prepared {dataset.n_prepared} of {dataset.n_input} molecules "
        f"({args.split} split): "
        f"train={sizes['train']} validation={sizes['validation']} test={sizes['test']}"
    )
    if dataset.n_duplicates:
        print(f"removed {dataset.n_duplicates} duplicate(s) ({args.dedup})")
    print(f"wrote {len(written)} files to {args.out_dir}/")
    return 0


def _run_structure_report(args: argparse.Namespace) -> int:
    from .structure_prep import prepare_structure

    try:
        if args.fetch:
            from .io import fetch_file

            source = fetch_file(args.fetch)
        else:
            source = args.file
        report = prepare_structure(source, protonation=args.protonation, ph=args.ph)
    except (OSError, ValueError, ImportError) as exc:
        print(f"structure-report failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        import json

        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(report.report_markdown())
        except OSError as exc:
            print(f"could not write {args.out}: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {args.out}")
    return 0


def _run_qc(args: argparse.Namespace) -> int:
    from .quality import quality_report

    try:
        if args.fetch:
            from .io import fetch_file

            source = fetch_file(args.fetch)
        else:
            source = args.file
        report = quality_report(source)
    except (OSError, ValueError, ImportError) as exc:
        print(f"qc failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        import json

        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(report.report_markdown())
        except OSError as exc:
            print(f"could not write {args.out}: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {args.out}")
    return 0


def _run_presets(args: argparse.Namespace) -> int:
    from .presets import format_presets, list_presets

    presets = list_presets(args.category)
    if args.json:
        import json

        print(json.dumps([p.to_dict() for p in presets], indent=2))
    else:
        print(format_presets(presets, show_features=args.features))
    return 0


def _run_coarse_grain(args: argparse.Namespace) -> int:
    try:
        mol = fetch(args.fetch) if args.fetch else read(args.file)
        beads, report = mol.coarse_grain(mapping=args.mapping, return_report=True)
    except (OSError, ValueError, ImportError) as exc:
        print(f"coarse-grain failed: {exc}", file=sys.stderr)
        return 2

    print(f"{args.mapping}: {report.coverage()}")
    if report.n_dropped:
        print(f"{report.n_dropped} atom(s) left unassigned")

    if args.out:
        try:
            _write_structure(beads, args.out)
        except (OSError, ValueError) as exc:
            print(f"could not write {args.out}: {exc}", file=sys.stderr)
            return 2
        print(f"wrote {report.n_sites} beads to {args.out}")
        if os.path.splitext(args.out)[1].lower() != ".pdb" and beads.bond_index is not None:
            print(
                "note: the bead bond network is written only for .pdb output "
                "(CONECT records); other formats are coordinates only",
                file=sys.stderr,
            )
    return 0


def _write_structure(molecule, path: str) -> None:
    """Write a molecule to ``path``, choosing the writer from the extension."""
    from .io import write_cif, write_pdb, write_xyz

    ext = os.path.splitext(path)[1].lower()
    writers = {".pdb": write_pdb, ".ent": write_pdb, ".cif": write_cif, ".xyz": write_xyz}
    if ext not in writers:
        raise ValueError(
            f"unsupported output extension {ext!r}; use .pdb, .cif or .xyz"
        )
    writers[ext](molecule, path)


def _parse_selection(specs) -> dict:
    """Parse CLI selection specs into ``Molecule.select`` keyword arguments."""
    if isinstance(specs, str):
        specs = [specs]

    selection = {}
    for spec in specs:
        parts = [
            part.strip()
            for part in re.split(r"\s+and\s+", spec.strip(), flags=re.IGNORECASE)
        ]
        for part in parts:
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"{part!r} is not key=value")
            key, value = [piece.strip() for piece in part.split("=", 1)]
            if not key or not value:
                raise ValueError(f"{part!r} is not key=value")
            if key not in _SELECTION_KEYS:
                supported = ", ".join(sorted(_SELECTION_KEYS))
                raise ValueError(f"unsupported field {key!r}; use one of: {supported}")
            if key in selection:
                raise ValueError(f"field {key!r} was specified more than once")
            selection[key] = _parse_selection_value(key, value)

    if not selection:
        raise ValueError("selection is empty")
    return selection


def _parse_selection_value(key: str, value: str):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]

    if key == "resid":
        try:
            if ":" in value:
                low, high = value.split(":", 1)
                return (int(low), int(high))
            if "-" in value and not value.startswith("-"):
                low, high = value.split("-", 1)
                return (int(low), int(high))
            return int(value)
        except ValueError as exc:
            raise ValueError(
                "resid expects an integer or inclusive range like 10-20"
            ) from exc

    if key == "residue_id":
        parts = value.split(":")
        if len(parts) not in (2, 3, 4):
            raise ValueError("residue_id expects chain:resid[:icode[:resname]]")
        try:
            resid = int(parts[1])
        except ValueError as exc:
            raise ValueError("residue_id selectors need an integer resid") from exc
        return tuple([parts[0], resid, *parts[2:]])

    if key == "hetero":
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "hetatm", "hetero"}:
            return True
        if lowered in {"0", "false", "no", "atom", "protein"}:
            return False
        raise ValueError("hetero expects true/false")

    return value



def _parse_ligand(value: str | None):
    if value is None:
        return None
    if ":" in value:
        parts = value.split(":")
        if len(parts) not in (2, 3, 4):
            raise ValueError("ligand selectors expect chain:resid[:icode[:resname]]")
        try:
            return tuple([parts[0], int(parts[1]), *parts[2:]])
        except ValueError as exc:
            raise ValueError("chain:resid ligand selectors need an integer resid") from exc
    return value


def _analyze_one(path: str, preset: str):
    """Compute flattened descriptors for one structure (worker; must be top-level
    so it is picklable under the ``spawn`` start method on macOS/Windows)."""
    from .descriptors import descriptors, flatten_descriptors

    try:
        mol = read(path)
        desc = descriptors(mol, preset=preset)
        return {"file": path, **flatten_descriptors(desc)}
    except Exception as e:
        print(f"Error processing {path}: {e}", file=sys.stderr)
        return None


def _run_analyze(args: argparse.Namespace) -> int:
    import csv
    from functools import partial
    from multiprocessing import Pool

    paths = _expand_globs(args.files)
    print(f"Analyzing {len(paths)} structures using {args.jobs} jobs...")

    worker = partial(_analyze_one, preset=args.preset)
    if args.jobs > 1:
        with Pool(args.jobs) as p:
            results = p.map(worker, paths)
    else:
        results = [worker(p) for p in paths]

    results = [r for result in results if (r := result) is not None]

    if not results:
        print("No results to save.")
        return 1

    keys = results[0].keys()
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved descriptors for {len(results)} structures to {args.out}")
    return 0


def _run_binding_site(args: argparse.Namespace) -> int:
    import csv

    try:
        ligand = _parse_ligand(args.ligand)
        mol = fetch(args.fetch) if args.fetch else read(args.file)
        site = mol.binding_site(ligand=ligand, cutoff=args.cutoff)
    except ValueError as e:
        print(f"Binding-site analysis failed: {e}", file=sys.stderr)
        return 2

    source = args.fetch if args.fetch else args.file
    rows = [
        {
            "file": source,
            "ligand_chain": site.ligand.chain,
            "ligand_resid": site.ligand.resid,
            "ligand_insertion_code": site.ligand.insertion_code,
            "ligand_resname": site.ligand.resname,
            "cutoff": site.cutoff,
            **record,
        }
        for record in site.to_records()
    ]
    _write_binding_site_csv(args.out, rows)

    if args.descriptors_out:
        desc = {"file": source, **site.descriptors(mol, preset="pocket-basic")}
        with open(args.descriptors_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(desc))
            writer.writeheader()
            writer.writerow(desc)

    print(f"Saved {len(rows)} binding-site residue records to {args.out}")
    return 0


def _write_binding_site_csv(path: str, rows: list[dict]) -> None:
    import csv

    fieldnames = [
        "file",
        "ligand_chain",
        "ligand_resid",
        "ligand_insertion_code",
        "ligand_resname",
        "cutoff",
        "residue_id",
        "chain",
        "resid",
        "insertion_code",
        "resname",
        "min_distance",
        "n_atom_contacts",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _export_one(
    path: str, to_fmt: str, out_dir: str, kwargs: dict, graph_kwargs: dict | None = None
) -> bool:
    """Export one structure's graph (worker; must be top-level so it is picklable
    under the ``spawn`` start method on macOS/Windows)."""
    try:
        mol = read(path)
        g = mol.to_graph(**(graph_kwargs or {}))
        stem = os.path.splitext(os.path.basename(path))[0]

        if to_fmt == "pyg":
            import torch
            data = g.to_pyg_data(**kwargs)
            out_path = os.path.join(out_dir, f"{stem}.pt")
            torch.save(data, out_path)
        elif to_fmt == "dgl":
            from dgl.data.utils import save_graphs
            dg = g.to_dgl_graph(**kwargs)
            out_path = os.path.join(out_dir, f"{stem}.bin")
            save_graphs(out_path, [dg])
        elif to_fmt == "nx":
            import json

            import networkx as nx
            # NetworkX exporter doesn't support the new options yet
            ng = g.to_networkx()
            out_path = os.path.join(out_dir, f"{stem}.json")
            with open(out_path, "w") as f:
                json.dump(nx.node_link_data(ng), f)
        return True
    except Exception as e:
        print(f"Error exporting {path}: {e}", file=sys.stderr)
        return False


def _run_export(args: argparse.Namespace) -> int:
    from functools import partial
    from multiprocessing import Pool

    paths = _expand_globs(args.files)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Exporting {len(paths)} structures to {args.to} format...")

    kwargs = {
        "include_self_loops": args.self_loops,
        "include_global_node": args.global_node,
        "include_pe": args.pe,
        "pe_k": args.pe_k,
    }
    graph_kwargs = {"min_seq_sep": args.min_seq_sep}
    if args.knn is not None:
        graph_kwargs["knn"] = args.knn
    if args.radius is not None:
        graph_kwargs["radius"] = args.radius
    if args.delaunay:
        graph_kwargs["delaunay"] = True
    worker = partial(
        _export_one, to_fmt=args.to, out_dir=args.out_dir,
        kwargs=kwargs, graph_kwargs=graph_kwargs,
    )
    if args.jobs > 1:
        with Pool(args.jobs) as p:
            successes = p.map(worker, paths)
    else:
        successes = [worker(p) for p in paths]

    print(f"Successfully exported {sum(successes)} structures to {args.out_dir}")
    return 0



def _write_csv_rows(path: str, columns: list[str], rows: list[dict]) -> None:
    from .docking import write_rows_csv
    write_rows_csv(path, columns, rows)


def _run_dock_summary(args: argparse.Namespace) -> int:
    from . import docking

    try:
        poses = docking.PoseStream(args.file)
        score_field = docking.resolve_score_field(poses, args.score_field)
    except (OSError, ValueError) as exc:
        print(f"dock-summary failed: {exc}", file=sys.stderr)
        return 2

    if args.higher is None:
        higher, assumed_dir = docking.higher_is_better(score_field)
    else:
        higher, assumed_dir = args.higher, False

    result = docking.summarize(
        poses, score_field, higher_is_better_flag=higher,
        direction_assumed=assumed_dir, with_smiles=args.smiles,
        best_pose_per_ligand=args.best_pose,
    )
    if not result.rows:
        print(
            f"no poses had a numeric {score_field!r} value "
            f"({result.n_missing} skipped)", file=sys.stderr,
        )
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    columns = ["rank", "pose_id", "name", "smiles", "score",
               "ligand_efficiency", "n_heavy_atoms"]
    summary_path = os.path.join(args.out_dir, "dock_summary.csv")
    top_path = os.path.join(args.out_dir, "top_hits.csv")
    _write_csv_rows(summary_path, columns, result.rows)
    _write_csv_rows(top_path, columns, result.rows[: max(0, args.top)])
    written = [summary_path, top_path]

    if args.figure:
        fig_path = os.path.join(args.out_dir, "score_distribution.png")
        if docking.plot_score_distribution(result.scores, score_field, fig_path):
            written.append(fig_path)

    direction = "higher-is-better" if higher else "lower-is-better"
    note = " (assumed; pass --higher/--lower-is-better)" if result.direction_assumed else ""
    print(f"ranked {len(result.rows)} poses by {score_field!r} ({direction}{note})")
    if result.n_missing:
        print(f"skipped {result.n_missing} pose(s) with no numeric {score_field!r}")
    if args.smiles and not result.with_smiles:
        print("SMILES column left blank: RDKit not installed (pip install \"molscope[chem]\")")
    best = result.rows[0]
    print(f"top hit: {best['name']} (score={best['score']:.3f})")
    print(f"wrote {len(written)} file(s) to {args.out_dir}/")
    return 0


def _run_dock_diverse(args: argparse.Namespace) -> int:
    from . import docking

    try:
        poses = docking.PoseStream(args.file)
        score_field = docking.resolve_score_field(poses, args.score_field)
    except (OSError, ValueError) as exc:
        print(f"dock-diverse failed: {exc}", file=sys.stderr)
        return 2

    if args.higher is None:
        higher, assumed_dir = docking.higher_is_better(score_field)
    else:
        higher, assumed_dir = args.higher, False
    try:
        result = docking.select_diverse_hits(
            poses, score_field, higher_is_better_flag=higher,
            direction_assumed=assumed_dir,
            top=args.top, select=args.select, threshold=args.threshold,
        )
    except ImportError as exc:
        print(f"dock-diverse needs RDKit: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"dock-diverse failed: {exc}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "diverse_hits.csv")
    sdf_path = os.path.join(args.out_dir, "diverse_hits.sdf")
    columns = ["rank", "pose_id", "name", "smiles", "score", "cluster_id", "cluster_size"]
    _write_csv_rows(csv_path, columns, [s for s in result.selected])
    docking.write_poses_sdf([s["pose"] for s in result.selected], sdf_path)

    direction = "higher-is-better" if higher else "lower-is-better"
    if result.direction_assumed:
        print(
            f"ranked by {score_field!r} ({direction}, assumed; "
            "pass --higher/--lower-is-better)"
        )
    print(
        f"clustered {result.n_pool} hits into {result.n_clusters} cluster(s) "
        f"at Tanimoto similarity {result.threshold:g}"
    )
    if result.n_failed_fp:
        print(f"failed to generate fingerprints for {result.n_failed_fp} pose(s) (dropped)")
    if result.capped_below_request:
        print(
            f"only {result.n_clusters} diverse cluster(s) exist, fewer than the "
            f"{result.requested} requested: returning all of them"
        )
    print(f"selected {len(result.selected)} diverse representative(s)")
    print(f"wrote {csv_path} and {sdf_path}")
    return 0


def _run_dock_rank(args: argparse.Namespace) -> int:
    from . import docking

    pose_sets = []
    try:
        for path in args.files:
            pose_sets.append((docking._stem(path), docking.PoseStream(path)))
        result = docking.consensus_rank(
            pose_sets,
            score_fields=args.score_fields,
            key=args.key,
            higher=set(args.higher_is_better) if args.higher_is_better else None,
            lower=set(args.lower_is_better) if args.lower_is_better else None,
            mw_max=args.mw_max,
            logp_max=args.logp_max,
        )
    except ImportError as exc:
        print(f"dock-rank needs RDKit for that option: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"dock-rank failed: {exc}", file=sys.stderr)
        return 2

    _write_csv_rows(args.out, result.columns, result.rows)

    print(f"consensus ranking over {len(result.rows)} molecule(s), joined by {result.key}")
    print("score fields used (direction):")
    for col in result.score_columns:
        arrow = "higher=better" if result.directions[col] else "lower=better"
        tag = " [assumed]" if col in result.assumed else ""
        print(f"  - {col}: {arrow}{tag}")
    if result.n_dropped_filter:
        print(f"dropped {result.n_dropped_filter} hit(s) on MW/logP filters")
    print(
        "note: consensus rank is mean rank across these fields, a transparent "
        "triage heuristic, not a calibrated 'true' affinity"
    )
    print(f"wrote {args.out}")
    return 0


def _run_dock_report(args: argparse.Namespace) -> int:
    from . import docking

    try:
        poses = docking.PoseStream(args.file)
        score_field = docking.resolve_score_field(poses, args.score_field)
    except (OSError, ValueError) as exc:
        print(f"dock-report failed: {exc}", file=sys.stderr)
        return 2

    if args.higher is None:
        higher, assumed_dir = docking.higher_is_better(score_field)
    else:
        higher, assumed_dir = args.higher, False

    summary = docking.summarize(
        poses, score_field, higher_is_better_flag=higher, direction_assumed=assumed_dir,
        best_pose_per_ligand=args.best_pose,
    )
    if not summary.rows:
        print(f"no poses had a numeric {score_field!r} value", file=sys.stderr)
        return 2

    diverse = None
    if args.clusters:
        try:
            diverse = docking.select_diverse_hits(
                poses, score_field, higher_is_better_flag=higher,
                top=max(500, args.top), select=args.select, threshold=args.threshold,
            )
        except ImportError:
            print("clustering skipped: RDKit not installed (pip install \"molscope[chem]\")")
        except ValueError as exc:
            print(f"clustering skipped: {exc}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Top poses for external 3D viewers: re-emit the best records as one SDF.
    poses_name = "top_poses.sdf"
    ranked_ids = [r["pose_id"] for r in summary.rows[: max(0, args.export_poses)]]
    top_poses = docking.collect_poses(poses, ranked_ids)
    if top_poses:
        docking.write_poses_sdf(top_poses, os.path.join(args.out_dir, poses_name))

    html = docking.render_html_report(
        summary, source_name=os.path.basename(args.file), n_poses=summary.n_poses,
        diverse=diverse, table_rows=args.top,
        poses_file=poses_name if top_poses else None,
    )
    report_path = os.path.join(args.out_dir, "dock_report.html")
    with open(report_path, "w") as handle:
        handle.write(html)

    print(f"ranked {len(summary.rows)} poses by {score_field!r}")
    if diverse is not None:
        print(f"clustered into {diverse.n_clusters} group(s); showed {len(diverse.selected)}")
        if diverse.n_failed_fp:
            print(f"failed to generate fingerprints for {diverse.n_failed_fp} pose(s) (dropped)")
    if top_poses:
        print(f"wrote {len(top_poses)} top pose(s) to {poses_name} for PyMOL/ChimeraX/Mol*")
    print(f"wrote {report_path}")
    return 0


def _expand_globs(patterns: list[str]) -> list[str]:
    paths = []
    for p in patterns:
        if "*" in p or "?" in p:
            paths.extend(glob.glob(p, recursive=True))
        else:
            paths.append(p)
    return sorted(list(set(paths)))


if __name__ == "__main__":
    raise SystemExit(main())
