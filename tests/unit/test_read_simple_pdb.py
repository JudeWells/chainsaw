"""read_simple_pdb must match Biopython's residues and C-alpha coordinates, or return None."""
from pathlib import Path

import numpy as np
import pytest

from src import featurisers

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"
FIXTURE_EXPECT_FAST = {
    "4wgvC.pdb": True,
    "5yclA.pdb": False,
    "AF-A0A0A9LVA1-F1-model_v4.pdb": True,
    "AF-A0A1W2PQ64-F1-model_v4.pdb": True,
}


def _atom(serial, name, resname, chain, resseq, xyz, altloc=" ", icode=" ", occ=1.0, record="ATOM  "):
    return (
        f"{record}{serial:5d} {name:^4s}{altloc}{resname:3s} {chain}{resseq:4d}{icode}   "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occ:6.2f} 20.00           C\n"
    )


def _residue(serial, resname, chain, resseq, base, ca_lines=None):
    lines = []
    for i, name in enumerate(["N", "CA", "C", "O"]):
        if name == "CA" and ca_lines is not None:
            lines += ca_lines
            continue
        lines.append(_atom(serial + i, name, resname, chain, resseq, base + 0.37 * i))
    return lines


def _chain(n=6, chain="A", start=5):
    names = ["ALA", "GLY", "LEU", "SER", "VAL", "LYS"]
    lines = []
    for i in range(n):
        lines += _residue(4 * i + 1, names[i % len(names)], chain, start + i, np.array([3.8 * i, 1.0 * (i % 3), 0.5]))
    return lines


def _write(tmp_path, lines, name="t.pdb"):
    p = tmp_path / name
    p.write_text("".join(lines) + "END   \n")
    return p


def _check(path, expect_fast):
    fast = featurisers.read_simple_pdb(str(path))
    assert (fast is not None) == expect_fast
    if fast is None:
        return
    model = featurisers.get_model_structure(str(path))
    chain = [c.id for c in model.get_chains()][0]
    residues = featurisers.get_model_structure_residues(model, chain=chain)
    dist = featurisers.get_distance(model, chain=chain)
    assert fast.chain_id == chain
    assert [(r.index, r.res_label, r.aa) for r in fast.residues] == [(r.index, r.res_label, r.aa) for r in residues]
    assert featurisers.distance_matrix(fast.ca_coords.astype(np.float16)).tobytes() == dist.tobytes()


@pytest.mark.parametrize("name,expect_fast", sorted(FIXTURE_EXPECT_FAST.items()))
def test_fixtures(name, expect_fast):
    _check(FIXTURES_DIR / name, expect_fast)


@pytest.mark.parametrize("occs", [(0.5, 0.5), (0.4, 0.6), (0.6, 0.4)])
def test_altloc_selection_matches_biopython(tmp_path, occs):
    ca = [
        _atom(900 + j, "CA", "LEU", "A", 50, (7.1 + 3 * j, -2.2 + j, 4.4), altloc="ABC"[j], occ=o)
        for j, o in enumerate(occs)
    ]
    lines = _chain() + _residue(800, "LEU", "A", 50, np.array([30.0, 0.0, 0.0]), ca_lines=ca) + _chain(3, start=60)
    _check(_write(tmp_path, lines), True)


_B = np.array([40.0, 0.0, 0.0])
DECLINED_CASES = {
    "blank_then_altloc": lambda: _chain() + _residue(800, "LEU", "A", 50, _B, ca_lines=[
        _atom(900, "CA", "LEU", "A", 50, (1, 2, 3)),
        _atom(901, "CA", "LEU", "A", 50, (2, 2, 3), altloc="A"),
    ]),
    "altloc_then_blank": lambda: _chain() + _residue(800, "LEU", "A", 50, _B, ca_lines=[
        _atom(900, "CA", "LEU", "A", 50, (1, 2, 3), altloc="A"),
        _atom(901, "CA", "LEU", "A", 50, (2, 2, 3)),
    ]),
    "repeated_altloc": lambda: _chain() + _residue(800, "LEU", "A", 50, _B, ca_lines=[
        _atom(900, "CA", "LEU", "A", 50, (1, 2, 3), altloc="A"),
        _atom(901, "CA", "LEU", "A", 50, (2, 2, 3), altloc="A"),
    ]),
    "nan_occupancy_altloc": lambda: _chain() + [
        "ATOM    800  CA AALA A  50       1.000   2.000   3.000   nan 20.00           C\n",
        "ATOM    801  CA BALA A  50       2.000   2.000   3.000  0.50 20.00           C\n",
    ],
    "icode": lambda: _chain() + [
        _atom(800 + i, n, "LEU", "A", 70, _B + i, icode="A") for i, n in enumerate(["N", "CA", "C"])
    ],
    "hetatm_std": lambda: _chain() + [_atom(800, "CA", "ALA", "A", 70, _B, record="HETATM")],
    "two_chains": lambda: _chain() + _chain(3, chain="B"),
    "split_residue": lambda: _chain() + [_atom(800, "CB", "ALA", "A", 5, _B)],
    "nonstd": lambda: _chain() + [_atom(800 + i, n, "UNK", "A", 70, _B + i) for i, n in enumerate(["N", "CA", "C"])],
    "missing_ca": lambda: _chain() + [_atom(800 + i, n, "ALA", "A", 70, _B + i) for i, n in enumerate(["N", "C", "O"])],
    "odd_ca_alignment": lambda: _chain() + [
        _atom(800, "N", "ALA", "A", 70, _B),
        "ATOM    801 CA   ALA A  70      40.000   1.000   0.000  1.00 20.00           C\n",
    ],
    "point_mutation": lambda: _chain() + [
        _atom(800, "CA", "ALA", "A", 70, _B, altloc="A", occ=0.5),
        _atom(801, "CA", "GLY", "A", 70, _B + 1, altloc="B", occ=0.5),
    ],
    "empty_model1_no_endmdl": lambda: ["MODEL        1\n", "MODEL        2\n"] + _chain(3) + ["ENDMDL\n"],
    "zero_padded_duplicate_ca": lambda: [
        "ATOM      1  CA  ALA A0001       0.000   0.000   0.000  1.00 20.00           C\n",
        "ATOM      2  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C\n",
    ],
}


@pytest.mark.parametrize("case", sorted(DECLINED_CASES))
def test_declines_unusual_files(tmp_path, case):
    _check(_write(tmp_path, DECLINED_CASES[case]()), False)


def test_record_boundaries_and_resseq_padding(tmp_path):
    for name, lines in (
        ("conect.pdb", _chain(3) + ["CONECT    1\n"] + _chain(2, start=20)),
        ("end.pdb", _chain(3) + ["END   \n"] + _chain(2, start=20)),
        ("models.pdb", ["MODEL        1\n"] + _chain(4) + ["ENDMDL\n", "MODEL        2\n"] + _chain(5, start=20)),
        ("pad.pdb", [
            "ATOM      1  N   ALA A0001       0.000   0.000   0.000  1.00 20.00           N\n",
            "ATOM      2  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C\n",
        ]),
    ):
        _check(_write(tmp_path, lines, name=name), True)


def test_cr_line_endings_fall_back_to_biopython(tmp_path):
    """CRLF, pure CR, and mixed LF+CR files fall back to Biopython so load_chain keeps all 194 residues."""
    raw = (FIXTURES_DIR / "AF-A0A1W2PQ64-F1-model_v4.pdb").read_bytes().replace(b"\r\n", b"\n")
    lines = raw.split(b"\n")
    idx165 = next(i for i, line in enumerate(lines) if line.startswith(b"ATOM  ") and line[22:26].strip() == b"165")
    variants = {
        "crlf.pdb": raw.replace(b"\n", b"\r\n"),
        "cr.pdb": raw.replace(b"\n", b"\r"),
        "mixed.pdb": b"\n".join(lines[:idx165]) + b"\r" + b"\r".join(lines[idx165:]),
    }
    for name, payload in variants.items():
        p = tmp_path / name
        p.write_bytes(payload)
        assert featurisers.read_simple_pdb(str(p)) is None
        chain, residues, dist = featurisers.load_chain(str(p))
        assert chain == "A" and len(residues) == 194 and dist.shape == (194, 194)
