"""--workers N must match single-process predictions (CPU mode and GPU two-stage pipeline)."""
import os
import shutil
import subprocess
import sys
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import get_predictions as gp  # noqa: E402

DATA_DIR = REPO_ROOT / "tests" / "fixtures"
MODEL_V3_DIR = REPO_ROOT / "saved_models" / "model_v3"


def _run_cli(structure_dir, out, extra=()):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "get_predictions.py"),
        "--structure_directory",
        str(structure_dir),
        "-o",
        str(out),
        "--save_dir",
        str(out.parent / "save"),
        "--renumber_pdbs",
        *extra,
    ]
    completed = subprocess.run(cmd, capture_output=True, cwd=str(out.parent))
    assert completed.returncode == 0, completed.stderr.decode()[-2000:]
    return [line.split("\t")[:-1] for line in out.read_text().splitlines()]


def test_workers_match_single_process(tmp_path):
    structures = tmp_path / "structures"
    structures.mkdir()
    for pdb in DATA_DIR.glob("*.pdb"):
        shutil.copy2(pdb, structures)
    serial = _run_cli(structures, tmp_path / "serial.tsv")
    parallel = _run_cli(structures, tmp_path / "parallel.tsv", ("--workers", "3"))
    assert len(serial) == 1 + len(list(DATA_DIR.glob("*.pdb")))
    assert parallel == serial


@pytest.mark.parametrize("post_process", [True, False], ids=["post_processing", "no_post_processing"])
def test_gpu_pipeline_matches_serial(tmp_path, monkeypatch, post_process):
    """Exercise prepare_features -> _run_network -> _decode_worker without requiring a GPU."""
    pdb_paths = []
    # 1 fast-reader fixture + 1 Biopython-fallback fixture
    for name in ("AF-A0A1W2PQ64-F1-model_v4.pdb", "5yclA.pdb"):
        dst = tmp_path / name
        shutil.copy2(DATA_DIR / name, dst)
        pdb_paths.append(str(dst))

    model_kwargs = dict(model_dir=str(MODEL_V3_DIR), post_process_domains=post_process)
    real_model = gp.load_model(**model_kwargs, force_cpu=True)
    serial = [gp.predict(real_model, p, renumber_pdbs=True, pdbchain=None) for p in pdb_paths]

    monkeypatch.setattr(gp.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gp, "load_model", lambda **kw: real_model)
    pipelined = list(gp._predict_all(pdb_paths, model_kwargs, None, True, workers=2))

    assert [(r.chain_id, r.sequence_md5, r.nres, r.ndom, r.chopping, r.confidence) for r in pipelined] == [
        (r.chain_id, r.sequence_md5, r.nres, r.ndom, r.chopping, r.confidence) for r in serial
    ]


def test_missing_model_dir_fails_fast(tmp_path):
    structures = tmp_path / "structures"
    structures.mkdir()
    for name in ("4wgvC.pdb", "5yclA.pdb"):
        shutil.copy2(DATA_DIR / name, structures / name)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "get_predictions.py"),
            "--structure_directory",
            str(structures),
            "--model_dir",
            str(tmp_path / "no_such_model"),
            "--workers",
            "2",
            "-o",
            str(tmp_path / "out.tsv"),
        ],
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert b"Model file not found" in completed.stderr


def _crash_worker(_):
    os._exit(1)


def test_abrupt_worker_death_does_not_hang():
    model_kwargs = dict(model_dir=str(MODEL_V3_DIR), post_process_domains=False)
    with pytest.raises(BrokenProcessPool):
        with gp._worker_pool(workers=2, model_kwargs=model_kwargs, threads=1) as pool:
            list(gp._ordered_map(pool, _crash_worker, [1, 2], window=2))
