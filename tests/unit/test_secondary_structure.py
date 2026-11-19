"""In-process renumbering must write the same bytes as the original pdb_reres.py subprocess."""
import subprocess
import sys
from pathlib import Path

import pytest

from src.constants import REPO_ROOT
from src.utils.secondary_structure import renum_pdb_file

FIXTURE = Path(__file__).parents[1] / "fixtures" / "4wgvC.pdb"


def _renum_subprocess(pdb_path: Path, output_path: Path) -> None:
    with open(output_path, "w") as out:
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "src/utils/pdb_reres.py"), str(pdb_path)],
            stdout=out,
            check=True,
            text=True,
        )


def test_renumber_identical_to_subprocess(tmp_path):
    # Real fixture + synthetic edge cases (MODEL reset, icode, short line, HETATM/TER/ANISOU, no trailing newline)
    edge = tmp_path / "edge.pdb"
    edge.write_text(
        "HEADER    TEST\n"
        "MODEL        1\n"
        "ATOM      1  N   ALA A  10      11.104   6.134  -6.504  1.00  0.00           N\n"
        "ANISOU    1  N   ALA A  10     2406   1892   1614    198    519   -328       N\n"
        "ATOM      2  CA  ALA A  10A     11.639   6.071  -5.147  1.00  0.00           C\n"
        "ATOM      3  CA  GLY A  11      12.639   6.071  -5.147\n"
        "TER       4      GLY A  11\n"
        "HETATM    5  O   HOH A 501       1.000   2.000   3.000  1.00  0.00           O\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "ATOM      1  N   ALA A  10      11.104   6.134  -6.504  1.00  0.00           N\n"
        "END"
    )
    for src in (FIXTURE, edge):
        a, b = tmp_path / f"{src.stem}_inproc.pdb", tmp_path / f"{src.stem}_subproc.pdb"
        renum_pdb_file(str(src), str(a))
        _renum_subprocess(src, b)
        assert a.read_bytes() == b.read_bytes()


def test_renumber_overflow_raises(tmp_path):
    src = tmp_path / "big.pdb"
    src.write_text(
        "".join(f"ATOM  {i % 100000:5d}  CA  ALA A{i % 10000:4d}       0.000   0.000   0.000\n" for i in range(10001))
    )
    with pytest.raises(ValueError):
        renum_pdb_file(str(src), str(tmp_path / "out.pdb"))
