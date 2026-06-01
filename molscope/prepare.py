"""Dataset preparation: turn a molecule table or SDF into ML-ready splits.

This is the bridge from a curated library (a CSV/XLSX of SMILES + properties, or
a multi-record ``.sdf``) to the files a QSAR, property-prediction, or graph-neural
-network experiment expects: ``train.csv``, ``validation.csv``, ``test.csv``, an
optional ``descriptors.csv``, and a human-readable ``report.md`` (plus a summary
figure). It builds directly on :mod:`molscope.library` (table I/O, RDKit SMILES
descriptors, the MaxMin diverse picker) rather than reimplementing them.

Degradation follows the same rule as the rest of MolScope: the bare NumPy install
does something useful, and everything chemical is gated behind the ``chem`` extra.

================  ==========================================  ====================
Capability        Needs                                       Notes
================  ==========================================  ====================
random split      NumPy only                                  seeded, deterministic
diversity split   existing numeric columns (NumPy)            or RDKit descriptors
exact dedup       stdlib                                      string match on a key
basic report      NumPy + Matplotlib (core)                   ``report.md`` + ``.png``
SMILES descriptors  ``chem`` (RDKit)                          via library helper
canonical dedup     ``chem`` (RDKit)                          canonical-SMILES key
scaffold split      ``chem`` (RDKit)                          Bemis-Murcko groups
fingerprints        ``chem`` (RDKit)                          Morgan on-bit lists
``.sdf`` input      ``chem`` (RDKit)                          multi-record supplier
================  ==========================================  ====================

This is dataset *plumbing*, not a novel method: the splitters mirror the standard
random / MaxMin / scaffold strategies (cf. DeepChem). The value is the short,
readable, one-command path from a structure or SMILES file to balanced splits.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .library import (
    MoleculeTable,
    read_table,
    select_diverse,
    smiles_descriptors,
)

SPLIT_METHODS = ("random", "diversity", "scaffold")
DEDUP_METHODS = ("none", "exact", "canonical")

#: Default Morgan fingerprint settings, recorded in the manifest so a consumer can
#: reconstruct the bit vector from the stored on-bit indices.
DEFAULT_FP_RADIUS = 2
DEFAULT_FP_BITS = 2048


# -- result containers ------------------------------------------------------

@dataclass
class SplitResult:
    """Row indices (into a :class:`MoleculeTable`) for each split."""

    method: str
    train: list[int]
    val: list[int]
    test: list[int]

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "validation": len(self.val), "test": len(self.test)}


@dataclass
class PreparedDataset:
    """A deduplicated table plus its split and the metadata for a report."""

    table: MoleculeTable
    split: SplitResult
    descriptor_cols: list[str] = field(default_factory=list)
    smiles_col: Optional[str] = None
    n_input: int = 0
    dedup_method: str = "none"
    n_duplicates: int = 0
    fingerprint_col: Optional[str] = None
    fingerprint_params: Optional[dict] = None
    seed: int = 0
    protonation: str = "none"
    ph: float = 7.0

    @property
    def n_prepared(self) -> int:
        return len(self.table)

    def manifest(self) -> dict:
        """Return a JSON-serialisable summary for reproducibility / tooling."""
        return {
            "n_input": self.n_input,
            "n_prepared": self.n_prepared,
            "dedup_method": self.dedup_method,
            "n_duplicates_removed": self.n_duplicates,
            "split_method": self.split.method,
            "split_sizes": self.split.sizes,
            "seed": self.seed,
            "smiles_col": self.smiles_col,
            "descriptor_cols": list(self.descriptor_cols),
            "fingerprint_col": self.fingerprint_col,
            "fingerprint_params": self.fingerprint_params,
            "protonation": self.protonation,
            "ph": self.ph if self.protonation == "pka" else None,
        }

    def write(self, out_dir: str, *, make_figure: bool = True) -> list[str]:
        """Write the split CSVs, descriptors, report and manifest to ``out_dir``.

        Returns the list of paths written. ``train.csv``/``validation.csv``/
        ``test.csv`` always appear (an empty split still writes its header).
        ``descriptors.csv`` appears only when descriptor columns were computed.
        ``report.png`` appears only when ``make_figure`` is true.
        """
        os.makedirs(out_dir, exist_ok=True)
        written: list[str] = []

        for name, indices in (
            ("train", self.split.train),
            ("validation", self.split.val),
            ("test", self.split.test),
        ):
            path = os.path.join(out_dir, f"{name}.csv")
            self.table.select_rows(indices).write(path)
            written.append(path)

        if self.descriptor_cols:
            id_cols = [c for c in self.table.columns if c not in self.descriptor_cols][:1]
            desc_cols = id_cols + list(self.descriptor_cols)
            desc_table = MoleculeTable(
                columns=desc_cols,
                rows=[{c: row.get(c) for c in desc_cols} for row in self.table.rows],
            )
            path = os.path.join(out_dir, "descriptors.csv")
            desc_table.write(path)
            written.append(path)

        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(self.manifest(), handle, indent=2)
        written.append(manifest_path)

        report_path = os.path.join(out_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(self._report_markdown())
        written.append(report_path)

        if make_figure:
            fig_path = os.path.join(out_dir, "report.png")
            if self._write_figure(fig_path):
                written.append(fig_path)

        return written

    # -- report rendering ---------------------------------------------------

    def _report_markdown(self) -> str:
        m = self.manifest()
        lines = [
            "# Dataset preparation report",
            "",
            f"- Input molecules: **{m['n_input']}**",
            f"- After {m['dedup_method']} deduplication: **{m['n_prepared']}** "
            f"({m['n_duplicates_removed']} removed)",
            f"- Split method: **{m['split_method']}** (seed {m['seed']})",
            "",
            "## Split sizes",
            "",
            "| Split | Molecules | Fraction |",
            "| --- | --- | --- |",
        ]
        total = max(self.n_prepared, 1)
        for name, count in m["split_sizes"].items():
            lines.append(f"| {name} | {count} | {count / total:.1%} |")

        if self.descriptor_cols:
            lines += ["", "## Descriptor balance across splits", "",
                      "Per-split mean of each descriptor (a quick check that the "
                      "splits are not skewed).", "",
                      "| Descriptor | train | validation | test |",
                      "| --- | --- | --- | --- |"]
            for col in self.descriptor_cols:
                means = self._split_means(col)
                lines.append(
                    f"| {col} | {_fmt(means['train'])} | "
                    f"{_fmt(means['validation'])} | {_fmt(means['test'])} |"
                )

        if self.split.method == "scaffold":
            lines += ["", "## Scaffold integrity", "",
                      "Scaffold split assigns whole Bemis-Murcko scaffold groups to a "
                      "single split, so no scaffold appears in more than one split. "
                      "This makes the test set a stricter generalisation check than a "
                      "random split."]

        lines += ["", "## Files", "",
                  "- `train.csv`, `validation.csv`, `test.csv` - the split rows",
                  *( ["- `descriptors.csv` - id + computed descriptor columns"]
                     if self.descriptor_cols else [] ),
                  "- `manifest.json` - machine-readable summary",
                  "- `report.md` - this file", ""]
        return "\n".join(lines)

    def _split_values(self, col: str) -> dict[str, np.ndarray]:
        out = {}
        for name, indices in (
            ("train", self.split.train),
            ("validation", self.split.val),
            ("test", self.split.test),
        ):
            vals = np.array(
                [_as_float(self.table.rows[i].get(col)) for i in indices], dtype=float
            )
            out[name] = vals[~np.isnan(vals)]
        return out

    def _split_means(self, col: str) -> dict[str, float]:
        return {
            name: (float(vals.mean()) if len(vals) else float("nan"))
            for name, vals in self._split_values(col).items()
        }

    def _write_figure(self, path: str) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover - matplotlib is a core dependency
            return False

        has_desc = bool(self.descriptor_cols)
        fig, axes = plt.subplots(1, 2 if has_desc else 1, figsize=(10 if has_desc else 5, 4))
        axes = np.atleast_1d(axes)

        sizes = self.split.sizes
        colors = {"train": "#4C72B0", "validation": "#DD8452", "test": "#55A868"}
        axes[0].bar(list(sizes), list(sizes.values()),
                    color=[colors[k] for k in sizes])
        axes[0].set_ylabel("molecules")
        axes[0].set_title(f"{self.split.method} split sizes")

        if has_desc:
            col = self.descriptor_cols[0]
            values = self._split_values(col)
            allv = np.concatenate([v for v in values.values() if len(v)]) if any(
                len(v) for v in values.values()
            ) else np.array([0.0])
            bins = np.linspace(float(allv.min()), float(allv.max()), 20)
            for name, vals in values.items():
                if len(vals):
                    axes[1].hist(vals, bins=bins, alpha=0.6, label=name, color=colors[name])
            axes[1].set_xlabel(col)
            axes[1].set_ylabel("count")
            axes[1].set_title(f"{col} by split")
            axes[1].legend()

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return True


# -- splitters --------------------------------------------------------------

def random_split(n: int, *, test: float, val: float, seed: int = 0) -> SplitResult:
    """Shuffle ``n`` indices with a seeded RNG and slice off test/val fractions."""
    _check_fractions(test, val)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(round(n * test))
    n_val = int(round(n * val))
    test_idx = order[:n_test]
    val_idx = order[n_test:n_test + n_val]
    train_idx = order[n_test + n_val:]
    return SplitResult("random", sorted(map(int, train_idx)),
                       sorted(map(int, val_idx)), sorted(map(int, test_idx)))


def diversity_split(matrix, *, test: float, val: float, standardize: bool = True) -> SplitResult:
    """Deal molecules into splits in MaxMin (farthest-first) order.

    Walking the diversity order and assigning each next molecule to whichever
    split is currently most below its target spreads chemically diverse points
    evenly across train/validation/test, so no split is starved of coverage.
    Rows with incomplete descriptors cannot be placed by diversity and are added
    to the training set. Deterministic given the same ``matrix``.
    """
    _check_fractions(test, val)
    data = np.asarray(matrix, dtype=float)
    n = len(data)
    order = select_diverse(data, n, standardize=standardize)  # all valid rows, diverse order
    placed = set(order)
    leftover = [i for i in range(n) if i not in placed]  # incomplete-descriptor rows

    targets = {"train": 1.0 - test - val, "validation": val, "test": test}
    assigned = {"train": [], "validation": [], "test": []}
    counts = {"train": 0.0, "validation": 0.0, "test": 0.0}
    for idx in order:
        # Largest remaining deficit wins; ties resolve train > validation > test.
        pick = max(("train", "validation", "test"),
                   key=lambda k: targets[k] * n - counts[k])
        assigned[pick].append(int(idx))
        counts[pick] += 1
    assigned["train"].extend(int(i) for i in leftover)

    return SplitResult("diversity", sorted(assigned["train"]),
                       sorted(assigned["validation"]), sorted(assigned["test"]))


def scaffold_split(smiles, *, test: float, val: float) -> SplitResult:
    """Group by Bemis-Murcko scaffold and fill train, then validation, then test.

    Whole scaffold groups go to a single split (largest groups first), so a
    scaffold never straddles two splits and the test set measures generalisation
    to unseen cores. Needs RDKit (``pip install "molscope[chem]"``).
    """
    _check_fractions(test, val)
    scaffolds = murcko_scaffolds(smiles)
    n = len(scaffolds)

    groups: dict[str, list[int]] = {}
    for i, scaf in enumerate(scaffolds):
        groups.setdefault(scaf, []).append(i)
    # Largest groups first; scaffold string breaks ties for determinism.
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    n_test, n_val = int(round(n * test)), int(round(n * val))
    train_cap = n - n_test - n_val
    val_cap = n_val
    train, val_idx, test_idx = [], [], []
    for _scaf, members in ordered:
        # Fill train to its cap, then validation; a group that fits in neither
        # spills to test. Whole groups stay together, so no scaffold straddles
        # two splits.
        if len(train) + len(members) <= train_cap:
            train.extend(members)
        elif len(val_idx) + len(members) <= val_cap:
            val_idx.extend(members)
        else:
            test_idx.extend(members)

    return SplitResult("scaffold", sorted(train), sorted(val_idx), sorted(test_idx))


# -- dedup & fingerprints ---------------------------------------------------

def dedup_keys(keys, method: str) -> tuple[list[int], int]:
    """Return ``(kept_indices, n_removed)`` keeping the first of each duplicate key.

    ``method`` is ``"exact"`` (string match) or ``"canonical"`` (RDKit canonical
    SMILES, needs the ``chem`` extra). Empty/None keys are always kept (they are
    not treated as duplicates of one another).
    """
    if method == "canonical":
        keys = canonical_smiles(keys)
    seen: set[str] = set()
    kept: list[int] = []
    removed = 0
    for i, key in enumerate(keys):
        text = "" if key is None else str(key).strip()
        if text and text in seen:
            removed += 1
            continue
        if text:
            seen.add(text)
        kept.append(i)
    return kept, removed


def murcko_scaffolds(smiles) -> list[str]:
    """Bemis-Murcko scaffold SMILES for each input (empty string when unavailable)."""
    from rdkit import RDLogger

    from .chem import _require_rdkit

    Chem, _ = _require_rdkit()
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    try:
        out = []
        for smi in smiles:
            if not smi or not str(smi).strip():
                out.append("")
                continue
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                out.append("")
                continue
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                out.append(Chem.MolToSmiles(scaffold))
            except Exception:  # pragma: no cover - defensive
                out.append("")
        return out
    finally:
        RDLogger.EnableLog("rdApp.*")


def canonical_smiles(smiles) -> list[str]:
    """Canonical SMILES for each input (empty string when unparseable)."""
    from rdkit import RDLogger

    from .chem import _require_rdkit

    Chem, _ = _require_rdkit()
    RDLogger.DisableLog("rdApp.*")
    try:
        out = []
        for smi in smiles:
            if not smi or not str(smi).strip():
                out.append("")
                continue
            mol = Chem.MolFromSmiles(str(smi))
            out.append(Chem.MolToSmiles(mol) if mol is not None else "")
        return out
    finally:
        RDLogger.EnableLog("rdApp.*")


def morgan_fingerprints(
    smiles, *, radius: int = DEFAULT_FP_RADIUS, n_bits: int = DEFAULT_FP_BITS
) -> list[str]:
    """Morgan fingerprints as semicolon-joined on-bit indices (``""`` if unparseable).

    Storing only the set bits keeps the CSV compact; ``radius`` and ``n_bits`` are
    recorded in the manifest so the dense vector can be rebuilt. Needs RDKit.
    """
    from rdkit import RDLogger

    from .chem import _require_rdkit

    Chem, _ = _require_rdkit()
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    RDLogger.DisableLog("rdApp.*")
    try:
        out = []
        for smi in smiles:
            if not smi or not str(smi).strip():
                out.append("")
                continue
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                out.append("")
                continue
            fp = gen.GetFingerprint(mol)
            out.append(";".join(str(b) for b in fp.GetOnBits()))
        return out
    finally:
        RDLogger.EnableLog("rdApp.*")


def read_sdf_table(path: str) -> tuple[MoleculeTable, str]:
    """Read a multi-record ``.sdf`` into a table of canonical SMILES + properties.

    Returns ``(table, smiles_col)``. Each record contributes an ``id`` (its title
    line or ``mol{index}``), a canonical ``smiles``, and any SD property fields.
    Needs RDKit (``pip install "molscope[chem]"``).
    """
    from rdkit import RDLogger

    from .chem import _require_rdkit

    Chem, _ = _require_rdkit()
    RDLogger.DisableLog("rdApp.*")
    try:
        supplier = Chem.SDMolSupplier(os.fspath(path))
        rows: list[dict] = []
        columns: list[str] = ["id", "smiles"]
        for i, mol in enumerate(supplier):
            if mol is None:
                continue
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            row = {"id": name or f"mol{i}", "smiles": Chem.MolToSmiles(mol)}
            for prop in mol.GetPropNames():
                row[prop] = mol.GetProp(prop)
                if prop not in columns:
                    columns.append(prop)
            rows.append(row)
    finally:
        RDLogger.EnableLog("rdApp.*")
    if not rows:
        raise ValueError(f"{path}: no readable molecules in SDF")
    return MoleculeTable(columns=columns, rows=rows), "smiles"


# -- orchestrator -----------------------------------------------------------

def prepare_dataset(
    source,
    *,
    smiles_col: Optional[str] = None,
    descriptor_cols: Optional[list[str]] = None,
    compute_descriptors: bool = False,
    rdkit_descriptors: Optional[list[str]] = None,
    split: str = "random",
    test: float = 0.1,
    val: float = 0.1,
    seed: int = 0,
    standardize: bool = True,
    dedup: str = "none",
    fingerprints: bool = False,
    protonation: str = "none",
    ph: float = 7.0,
) -> PreparedDataset:
    """Read, deduplicate, optionally featurise, and split a molecule dataset.

    ``source`` is a path (``.csv``/``.tsv``/``.xlsx``/``.sdf``) or an existing
    :class:`MoleculeTable`. See the module docstring for which options need the
    ``chem`` extra. Returns a :class:`PreparedDataset`; call ``.write(out_dir)``
    to emit the split CSVs, descriptors, report and manifest.

    With ``protonation="pka"`` the SMILES are set to their dominant ionisation
    state at ``ph`` (Dimorphite-DL, ``pip install "molscope[dimorphite]"``) before
    descriptors and fingerprints are computed, so charge-sensitive features match
    the species present at that pH. Deduplication and the stored SMILES column are
    left on the original (input) SMILES.
    """
    if split not in SPLIT_METHODS:
        raise ValueError(f"unknown split {split!r}; use one of: {', '.join(SPLIT_METHODS)}")
    if dedup not in DEDUP_METHODS:
        raise ValueError(f"unknown dedup {dedup!r}; use one of: {', '.join(DEDUP_METHODS)}")
    if protonation not in ("none", "pka"):
        raise ValueError("protonation must be 'none' or 'pka'")

    table, smiles_col = _load_source(source, smiles_col)
    n_input = len(table)

    # 1. Deduplicate on the SMILES column (or fall back to the first column).
    n_duplicates = 0
    if dedup != "none":
        key_col = smiles_col or (table.columns[0] if table.columns else None)
        if key_col is None:
            raise ValueError("deduplication needs a column to key on")
        kept, n_duplicates = dedup_keys(table.column(key_col), dedup)
        table = table.select_rows(kept)

    # 2. Featurisation SMILES: optionally set to the dominant state at ``ph`` once,
    #    reused for both descriptors and fingerprints. The stored SMILES column and
    #    dedup keys stay on the original input SMILES.
    feat_smiles = None
    if protonation == "pka" and (compute_descriptors or fingerprints):
        if not smiles_col:
            raise ValueError("protonation='pka' needs a smiles_col")
        from .library import protonate_smiles

        feat_smiles = protonate_smiles(table.column(smiles_col), ph=ph)

    # 3. Descriptors: compute from SMILES, or use existing numeric columns.
    desc_cols: list[str] = []
    if compute_descriptors:
        if not smiles_col:
            raise ValueError("compute_descriptors needs a smiles_col")
        src = feat_smiles if feat_smiles is not None else table.column(smiles_col)
        matrix, names = smiles_descriptors(src, names=rdkit_descriptors)
        table = table.with_columns(names, matrix)
        desc_cols = names
    elif descriptor_cols:
        desc_cols = list(descriptor_cols)

    # 4. Optional Morgan fingerprints as an extra column.
    fp_col = None
    fp_params = None
    if fingerprints:
        if not smiles_col:
            raise ValueError("fingerprints need a smiles_col")
        bits = morgan_fingerprints(
            feat_smiles if feat_smiles is not None else table.column(smiles_col)
        )
        fp_col = "morgan_onbits"
        table = MoleculeTable(
            columns=table.columns + ([fp_col] if fp_col not in table.columns else []),
            rows=[{**row, fp_col: bits[i]} for i, row in enumerate(table.rows)],
        )
        fp_params = {"type": "morgan", "radius": DEFAULT_FP_RADIUS, "n_bits": DEFAULT_FP_BITS}

    # 5. Split.
    if split == "random":
        result = random_split(len(table), test=test, val=val, seed=seed)
    elif split == "diversity":
        if not desc_cols:
            raise ValueError(
                "diversity split needs descriptors: pass descriptor_cols, or "
                "compute_descriptors=True with a smiles_col"
            )
        result = diversity_split(
            table.numeric_matrix(desc_cols), test=test, val=val, standardize=standardize
        )
    else:  # scaffold
        if not smiles_col:
            raise ValueError("scaffold split needs a smiles_col")
        result = scaffold_split(table.column(smiles_col), test=test, val=val)

    return PreparedDataset(
        table=table,
        split=result,
        descriptor_cols=desc_cols,
        smiles_col=smiles_col,
        n_input=n_input,
        dedup_method=dedup,
        n_duplicates=n_duplicates,
        fingerprint_col=fp_col,
        fingerprint_params=fp_params,
        seed=seed,
        protonation=protonation,
        ph=ph,
    )


# -- internal helpers -------------------------------------------------------

def _load_source(source, smiles_col):
    if isinstance(source, MoleculeTable):
        return source, smiles_col
    ext = os.path.splitext(os.fspath(source))[1].lower()
    if ext in (".sdf", ".mol"):
        return read_sdf_table(source)
    return read_table(source), smiles_col


def _check_fractions(test: float, val: float) -> None:
    if test < 0 or val < 0:
        raise ValueError("test and val fractions must be non-negative")
    if test + val >= 1.0:
        raise ValueError("test + val fractions must be < 1 (leaving rows for training)")


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.3g}"  # value != value catches NaN
