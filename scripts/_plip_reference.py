"""PLIP reference extractor for the pocket-interaction validation study.

Runs *inside an environment that has PLIP + OpenBabel installed* (PLIP is
conda-only in practice) and emits, as JSON on stdout, the protein residues PLIP
detects for each interaction class around every non-solvent ligand in a PDB
file. It is the ground-truth side of
``scripts/validate_pocket_interactions.py`` and the
``tests/validation/test_pocket_interactions_ref.py`` cross-check; MolScope
itself is never imported here.

Usage (typically invoked for you by the harness)::

    conda run -n plip-ref python scripts/_plip_reference.py structure.pdb

Output schema::

    {"sites": {"BEN:A:1": {"hydrophobic": [["A", 213, "VAL"], ...],
                            "hbond": [...], "salt_bridge": [...], "pi": [...]}}}

``pi`` is the union of pi-stacking and pi-cation contacts. Residue identity is
``[chain, resnr, restype]``. PLIP's own logging is silenced so stdout is clean
JSON.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import sys

# PLIP is chatty on import and during analysis; keep stdout pure JSON.
logging.disable(logging.CRITICAL)

WATER = frozenset({"HOH", "WAT", "DOD", "H2O", "SOL"})
IONS = frozenset({
    "NA", "K", "CL", "MG", "CA", "ZN", "FE", "MN", "CU", "NI", "CO", "CD",
    "HG", "BR", "IOD", "LI", "RB", "CS", "SR", "BA",
})


def _residues(interactions) -> list[list]:
    """Unique ``[chain, resnr, restype]`` triples for a list of PLIP contacts."""
    seen = {(i.reschain, int(i.resnr), i.restype) for i in interactions}
    return sorted([list(t) for t in seen])


def extract(pdb_path: str) -> dict:
    from plip.structure.preparation import PDBComplex

    complex_ = PDBComplex()
    # PLIP prints preparation notes to stdout; redirect them away from our JSON.
    with contextlib.redirect_stdout(io.StringIO()):
        complex_.load_pdb(pdb_path)
        complex_.analyze()

    sites: dict[str, dict] = {}
    for key, site in complex_.interaction_sets.items():
        ligtype = key.split(":")[0]
        if ligtype in WATER or ligtype in IONS:
            continue
        sites[key] = {
            "hydrophobic": _residues(site.hydrophobic_contacts),
            "hbond": _residues(site.hbonds_ldon + site.hbonds_pdon),
            "salt_bridge": _residues(site.saltbridge_lneg + site.saltbridge_pneg),
            "pi": _residues(
                list(site.pistacking)
                + site.pication_laro
                + site.pication_paro
            ),
        }
    return {"sites": sites}


def main() -> None:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python _plip_reference.py STRUCTURE.pdb\n")
        raise SystemExit(2)
    json.dump(extract(sys.argv[1]), sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
