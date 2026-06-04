"""Tests for chain interfaces and ligand-binding-site contacts."""

import os

import numpy as np
import pytest

import molscope as ms
from molscope import Molecule

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "data")
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def two_chain():
    """Chain A (res 1 at x=0, res 2 at x=4) and chain B (res 1 at x=6).

    A-res2 and B-res1 sit 2 A apart; A-res1 is 6 A from B.
    """
    return Molecule(
        np.array([[0.0, 0, 0], [4.0, 0, 0], [6.0, 0, 0]]), ["C", "C", "C"],
        atom_names=["CA", "CA", "CA"], resnames=["ALA", "ALA", "GLU"],
        resids=np.array([1, 2, 1]), chains=["A", "A", "B"],
    )


def protein_with_ligand():
    """Two protein residues + a LIG ligand atom + a water, with hetero flags.

    Protein res 1 (x=0) is 2 A from the ligand (x=2); res 2 (x=10) is far.
    """
    return Molecule(
        np.array([[0.0, 0, 0], [10.0, 0, 0], [2.0, 0, 0], [20.0, 0, 0]]),
        ["C", "C", "N", "O"],
        atom_names=["CA", "CA", "N1", "O"],
        resnames=["ALA", "GLY", "LIG", "HOH"],
        resids=np.array([1, 2, 100, 200]),
        chains=["A", "A", "A", "A"],
        hetero=[False, False, True, True],
    )


# -- interfaces -------------------------------------------------------------


def test_interface_residues_finds_contacting_residues():
    iface = two_chain().interface("A", "B", cutoff=5.0)
    assert iface.n_atom_contacts == 1
    assert [r.resid for r in iface.residues_a] == [2]      # only A-res2 is close
    assert [r.resid for r in iface.residues_b] == [1]
    assert iface.contacts == [(1, 2)]                       # atom 1 (A) - atom 2 (B)


def test_interface_unknown_chain_raises():
    with pytest.raises(ValueError):
        two_chain().interface("A", "Z")


def test_interface_requires_chain_metadata():
    bare = Molecule(np.zeros((2, 3)), ["C", "C"])
    with pytest.raises(ValueError):
        bare.interface("A", "B")


def test_chain_contact_matrix_is_symmetric_counts():
    ccm = two_chain().chain_contacts(cutoff=5.0)
    assert ccm.chains == ["A", "B"]
    assert ccm.matrix.shape == (2, 2)
    assert np.array_equal(ccm.matrix, ccm.matrix.T)
    assert np.diag(ccm.matrix).sum() == 0
    assert ccm.count("A", "B") == 1


# -- ligands & binding sites ------------------------------------------------


def test_ligands_excludes_water_and_ions():
    mol = protein_with_ligand()
    ligs = mol.ligands()
    assert [g.resname for g in ligs] == ["LIG"]             # HOH filtered out
    assert mol.ligands(exclude_water=False)                 # water kept when asked
    assert any(g.resname == "HOH" for g in mol.ligands(exclude_water=False))


def test_binding_site_synthetic():
    bs = protein_with_ligand().binding_site(cutoff=4.5)
    assert bs.ligand.resname == "LIG"
    assert [r.resid for r in bs.residues] == [1]            # only res 1 is within 4.5 A
    assert bs.min_distances[0] == pytest.approx(2.0)
    assert all(p < q for p, q in zip(bs.min_distances, bs.min_distances[1:]))
    assert bs.n_atom_contacts == 1
    assert bs.residue_contact_counts == [1]
    assert bs.to_records() == [{
        "residue_id": "A:ALA1",
        "chain": "A",
        "resid": 1,
        "insertion_code": "",
        "resname": "ALA",
        "min_distance": 2.0,
        "n_atom_contacts": 1,
    }]


def test_binding_site_can_select_full_site_residues():
    mol = Molecule(
        np.array([[0.0, 0, 0], [8.0, 0, 0], [10.0, 0, 0], [2.0, 0, 0]]),
        ["C", "C", "C", "N"],
        atom_names=["CA", "CB", "CA", "N1"],
        resnames=["ALA", "ALA", "GLY", "LIG"],
        resids=np.array([1, 1, 2, 100]),
        chains=["A", "A", "A", "A"],
        hetero=[False, False, False, True],
    )
    bs = mol.binding_site(cutoff=4.5)
    assert bs.contact_atom_indices == [0]
    assert bs.protein_atom_indices == [0, 1]
    assert len(bs.to_molecule(mol)) == 2
    assert len(bs.to_molecule(mol, include_ligand=True)) == 3


def test_binding_site_pocket_basic_descriptors_are_fixed_size():
    mol = protein_with_ligand()
    bs = mol.binding_site(cutoff=4.5)
    desc = bs.descriptors(mol)
    assert set(desc) == set(ms.pocket_descriptor_feature_names("pocket-basic"))
    assert desc["pocket_n_atoms"] == 1.0
    assert desc["pocket_n_residues"] == 1.0
    assert desc["ligand_n_atoms"] == 1.0
    assert desc["pocket_atom_contact_count"] == 1.0
    assert desc["pocket_residue_count_ALA"] == 1.0
    assert desc["ligand_distance_min"] == pytest.approx(2.0)
    assert desc["ligand_contact_distance_mean"] == pytest.approx(2.0)


def test_binding_site_plot_shortcut_returns_axes():
    import matplotlib

    matplotlib.use("Agg")
    mol = protein_with_ligand()
    ax = mol.binding_site(cutoff=4.5).plot(mol, show=False)
    assert ax is not None


def test_pocket_descriptors_reject_unknown_preset():
    mol = protein_with_ligand()
    bs = mol.binding_site(cutoff=4.5)
    with pytest.raises(ValueError, match="unknown pocket descriptor preset"):
        bs.descriptors(mol, preset="bogus")
    with pytest.raises(ValueError, match="unknown pocket descriptor preset"):
        ms.pocket_descriptor_feature_names("bogus")


def test_pocket_descriptors_handle_empty_site():
    mol = protein_with_ligand()
    bs = mol.binding_site(cutoff=0.01)          # nothing this close to the ligand
    assert bs.residues == []
    assert bs.contacts == []

    desc = bs.descriptors(mol)
    # still a fixed-size vector, all zeros for the empty pocket
    assert set(desc) == set(ms.pocket_descriptor_feature_names("pocket-basic"))
    assert desc["pocket_n_atoms"] == 0.0
    assert desc["pocket_n_residues"] == 0.0
    assert desc["pocket_radius_of_gyration"] == 0.0
    assert desc["pocket_dim_x"] == 0.0
    assert desc["pocket_bbox_volume"] == 0.0
    assert desc["pocket_contact_density"] == 0.0
    assert desc["ligand_distance_mean"] == 0.0
    assert desc["ligand_contact_distance_max"] == 0.0
    assert desc["pocket_residue_count_OTHER"] == 0.0


def test_binding_site_ambiguous_requires_explicit_ligand():
    mol = Molecule(
        np.array([[0.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]),
        ["C", "N", "N"], atom_names=["CA", "N1", "N1"],
        resnames=["ALA", "LIG", "FAD"], resids=np.array([1, 100, 101]),
        chains=["A", "A", "A"], hetero=[False, True, True],
    )
    with pytest.raises(ValueError):
        mol.binding_site()                                  # two candidate ligands
    bs = mol.binding_site(ligand="LIG", cutoff=4.5)
    assert bs.ligand.resname == "LIG"


def test_binding_site_disambiguates_ligand_and_residue_insertion_codes():
    mol = ms.read(os.path.join(FIXTURES, "ugly_residue_ids.pdb"))
    ligands = mol.ligands()
    assert [lig.residue_id.label() for lig in ligands] == ["A:LIG10", "B:LIG10"]
    assert [len(lig) for lig in ligands] == [2, 1]

    with pytest.raises(ValueError, match="matches multiple groups"):
        mol.binding_site(ligand="LIG", cutoff=2.0)

    site = mol.binding_site(ligand=("A", 10), cutoff=2.0)
    assert site.ligand.residue_id.label() == "A:LIG10"
    assert [res.residue_id.label() for res in site.residues] == ["A:SER100A"]
    assert site.to_records()[0]["insertion_code"] == "A"
    assert site.to_records()[0]["residue_id"] == "A:SER100A"


def test_binding_site_no_hetatm_raises():
    bare = Molecule(
        np.zeros((2, 3)), ["C", "C"], atom_names=["CA", "CA"],
        resnames=["ALA", "ALA"], resids=np.array([1, 2]), chains=["A", "A"],
    )
    with pytest.raises(ValueError):
        bare.binding_site()


def test_binding_site_on_trypsin_benzamidine():
    mol = ms.read(os.path.join(DATA, "3ptb.pdb"))
    ligs = mol.ligands()                                    # CA ion + waters excluded
    assert [g.resname for g in ligs] == ["BEN"]
    bs = mol.binding_site(cutoff=4.5)
    resids = {r.resid for r in bs.residues}
    # The benzamidine S1 pocket: Asp189 (specificity), Ser190, Gly219, Ser195.
    assert {189, 190, 195, 219} <= resids
    assert bs.min_distances == sorted(bs.min_distances)


# -- semantic pocket description --------------------------------------------


def pocket_with_interactions():
    """A synthetic BEN pocket with one residue of each interaction class.

    PHE70 (apolar ring) sits ~3.5 A from a ligand carbon (hydrophobic wall +
    aromatic), ASP189 OD2 ~3.1 A from the ligand amine N (salt bridge), and
    SER95 OG ~2.9 A from a ligand carbonyl O (hydrogen bond). GLY10 is too far.
    """
    return Molecule(
        np.array([
            [0.0, 0.0, 0.0],     # 0 PHE70 CA
            [3.5, 0.0, 0.0],     # 1 PHE70 CG (apolar carbon, ~3.5 A from BEN C1)
            [0.0, 5.0, 0.0],     # 2 ASP189 CA
            [3.1, 5.0, 0.0],     # 3 ASP189 OD2 (anionic, ~3.1 A from BEN N1)
            [0.0, -5.0, 0.0],    # 4 SER95 CA
            [2.9, -5.0, 0.0],    # 5 SER95 OG (~2.9 A from BEN O2)
            [40.0, 0.0, 0.0],    # 6 GLY10 CA (far)
            # ligand BEN
            [7.0, 0.0, 0.0],     # 7 BEN C1
            [6.2, 5.0, 0.0],     # 8 BEN N1 (amine)
            [5.8, -5.0, 0.0],    # 9 BEN O2 (carbonyl)
        ]),
        elements=["C", "C", "C", "O", "C", "O", "C", "C", "N", "O"],
        atom_names=["CA", "CG", "CA", "OD2", "CA", "OG", "CA", "C1", "N1", "O2"],
        resnames=["PHE", "PHE", "ASP", "ASP", "SER", "SER", "GLY",
                  "BEN", "BEN", "BEN"],
        resids=np.array([70, 70, 189, 189, 95, 95, 10, 300, 300, 300]),
        chains=["A"] * 10,
        hetero=[False] * 7 + [True] * 3,
    )


def test_describe_environment_detects_each_interaction_class():
    mol = pocket_with_interactions()
    env = mol.select_pocket(ligand="BEN", cutoff=4.5).environment()

    assert [r.resname for r in env.hydrophobic_residues] == ["PHE"]
    assert [r.resname for r in env.aromatic_residues] == ["PHE"]

    assert len(env.salt_bridges) == 1
    sb = env.salt_bridges[0]
    assert (sb.residue.resname, sb.residue_atom, sb.ligand_atom) == ("ASP", "OD2", "N1")
    assert sb.distance == pytest.approx(3.1, abs=1e-3)

    # The ASP-N1 pair is reported once, as a salt bridge, not also as an H-bond.
    assert all(hb.residue.resname != "ASP" for hb in env.hydrogen_bonds)
    assert len(env.hydrogen_bonds) == 1
    hb = env.hydrogen_bonds[0]
    assert (hb.residue.resname, hb.residue_atom, hb.ligand_atom) == ("SER", "OG", "O2")
    assert hb.distance == pytest.approx(2.9, abs=1e-3)


def test_describe_environment_prose_mentions_key_features():
    mol = pocket_with_interactions()
    prompt = mol.select_pocket(ligand="BEN", cutoff=4.5).describe_environment()
    assert "ligand BEN" in prompt
    assert "hydrophobic" in prompt and "PHE70" in prompt
    assert "pi-stacking" in prompt
    assert "hydrogen bond" in prompt.lower() and "SER95" in prompt
    assert "electrostatic" in prompt and "carboxylate of ASP189" in prompt
    # Heuristic confidence: prose hedges and points to a rigorous profiler.
    assert "likely" in prompt and "possible" in prompt
    assert "PLIP or ProLIF" in prompt


def test_describe_environment_purely_hydrophobic_pocket():
    mol = Molecule(
        np.array([[0.0, 0, 0], [3.5, 0, 0], [40.0, 0, 0], [7.0, 0, 0]]),
        elements=["C", "C", "C", "C"],
        atom_names=["CA", "CB", "CA", "C1"],
        resnames=["LEU", "LEU", "GLY", "LIG"],
        resids=np.array([1, 1, 9, 100]),
        chains=["A", "A", "A", "A"],
        hetero=[False, False, False, True],
    )
    env = mol.select_pocket(cutoff=4.5).environment()
    assert [r.resname for r in env.hydrophobic_residues] == ["LEU"]
    assert env.hydrogen_bonds == [] and env.salt_bridges == []
    assert "predominantly hydrophobic" in env.text()


def test_select_pocket_wrapper_delegates_to_binding_site():
    mol = pocket_with_interactions()
    pocket = mol.select_pocket(ligand="BEN", cutoff=4.5)
    assert pocket.ligand.resname == "BEN"
    assert pocket.cutoff == 4.5
    assert {r.resname for r in pocket.residues} == {"PHE", "ASP", "SER"}
    # descriptors / to_molecule reachable without re-passing the molecule.
    assert pocket.descriptors()["pocket_n_residues"] == 3.0
    assert len(pocket.to_molecule(include_ligand=True)) == len(pocket.to_molecule()) + 3


def test_describe_environment_on_trypsin_benzamidine():
    mol = ms.read(os.path.join(DATA, "3ptb.pdb"))
    env = mol.select_pocket(ligand="BEN", cutoff=4.5).environment()
    # The hallmark of the benzamidine S1 site: a salt bridge to Asp189.
    assert any(sb.residue.resid == 189 and sb.residue.resname == "ASP"
               for sb in env.salt_bridges)
    prompt = env.text()
    assert "ligand BEN" in prompt
    assert "electrostatic" in prompt


def test_describe_environment_to_dict_is_json_serialisable():
    import json

    env = pocket_with_interactions().select_pocket(ligand="BEN").environment()
    payload = json.dumps(env.to_dict())
    assert "salt_bridges" in payload
    assert "hydrogen_bonds" in payload


# -- rich residue identity on contact records -------------------------------


def test_residue_repr_icode_and_residue_id():
    res = ms.Residue("A", 100, "SER", insertion_code="A")
    assert res.icode == "A"
    assert res.residue_id.label() == "A:SER100A"
    assert repr(res) == "A:SER100A"
    # Empty chain drops the chain prefix.
    assert repr(ms.Residue("", 5, "GLY")) == "GLY5"


def test_ligand_residue_repr_icode_and_residue_id():
    lig = ms.LigandResidue("A", 10, "LIG", [0, 1], insertion_code="C")
    assert lig.icode == "C"
    assert len(lig) == 2
    assert lig.residue_id.label() == "A:LIG10C"
    assert repr(lig) == "LigandResidue(A:LIG10C, 2 atoms)"


def test_resolve_ligand_by_residue_id_and_residue_objects():
    mol = ms.read(os.path.join(FIXTURES, "ugly_residue_ids.pdb"))
    by_rid = mol.binding_site(ligand=ms.ResidueId("A", 10), cutoff=2.0)
    assert by_rid.ligand.residue_id.label() == "A:LIG10"
    by_res = mol.binding_site(ligand=ms.Residue("B", 10, "LIG"), cutoff=2.0)
    assert by_res.ligand.chain == "B"


def test_resolve_ligand_by_tuple_with_icode_and_resname():
    mol = ms.read(os.path.join(FIXTURES, "ugly_residue_ids.pdb"))
    assert mol.binding_site(ligand=("A", 10, ""), cutoff=2.0).ligand.chain == "A"
    site = mol.binding_site(ligand=("A", 10, "", "LIG"), cutoff=2.0)
    assert site.ligand.resname == "LIG"


def test_resolve_ligand_passthrough_for_ligand_residue():
    mol = ms.read(os.path.join(FIXTURES, "ugly_residue_ids.pdb"))
    target = mol.ligands()[0]
    assert mol.binding_site(ligand=target, cutoff=2.0).ligand is target


def test_resolve_ligand_no_match_raises():
    mol = ms.read(os.path.join(FIXTURES, "ugly_residue_ids.pdb"))
    with pytest.raises(ValueError, match="no HETATM group matching"):
        mol.binding_site(ligand=("Z", 99), cutoff=2.0)
    # Right chain/resid but a non-matching insertion code or resname misses too.
    with pytest.raises(ValueError, match="no HETATM group matching"):
        mol.binding_site(ligand=("A", 10, "Z"), cutoff=2.0)
    with pytest.raises(ValueError, match="no HETATM group matching"):
        mol.binding_site(ligand=("A", 10, "", "WAT"), cutoff=2.0)
    with pytest.raises(ValueError, match="no HETATM group with resname"):
        mol.binding_site(ligand="XXX", cutoff=2.0)


def test_resolve_ligand_ambiguous_selector_reports_all_matches():
    # Two HETATM groups share chain+resid but differ in resname, so a bare
    # (chain, resid) selector cannot disambiguate them.
    mol = Molecule(
        np.array([[0.0, 0, 0], [2.0, 0, 0], [2.5, 0, 0]]),
        ["C", "C", "O"],
        atom_names=["CA", "C1", "O1"],
        resnames=["ALA", "LIG", "DRG"],
        resids=np.array([1, 50, 50]),
        chains=["A", "X", "X"],
        hetero=[False, True, True],
    )
    with pytest.raises(ValueError, match="matches multiple groups"):
        mol.binding_site(ligand=("X", 50), cutoff=3.0)
