import os
from pathlib import Path
import subprocess

from test_basic_usage import REPO_ROOT, DATA_DIR, normalise_output

def test_basic_usage(tmp_path):

    # setup test data
    af_id = "AF-A0A0A9LVA1-F1-model_v4"
    example_structure_path = DATA_DIR / f"{af_id}.pdb"
    expected_cols = [
        ['chain_id', 'sequence_md5', 'nres', 'ndom', 'chopping', 'confidence'],
        ['AF-A0A0A9LVA1-F1-model_v4', 'aed99aecb8126442b8129ea77da917af', '16', '0', 'NULL', '0.932'],
    ]
    expected_output = "\n".join(["\t".join(row) for row in expected_cols])
    orig_path = Path.cwd()
    script_path = REPO_ROOT / "get_predictions.py"
    model_path = REPO_ROOT / "saved_models" / "model_v1"

    results_file = tmp_path / "test_output.tsv"

    # make sure we can run this in an isolated directory
    results_output = None
    try:
        os.chdir(str(tmp_path))
        cmd_args = ["python", str(script_path), "--structure_file", str(example_structure_path), "-o", str(results_file),
                    "--model_dir", str(model_path), "--renumber_pdbs",
                    ]
        completed_process = subprocess.run(cmd_args, check=True, capture_output=True)
        results_output = results_file.read_text().strip()
    except subprocess.CalledProcessError as e:
        print(f"ERROR: CMD: {e.cmd}")
        print(f"ERROR: STDOUT: {e.stdout.decode()}")
        print(f"ERROR: STDERR: {e.stderr.decode()}")
        raise
    finally:
        os.chdir(str(orig_path))

    # check the process ran successfully
    assert completed_process.returncode == 0

    # check the output file has the expected content
    assert normalise_output(results_output) == normalise_output(expected_output)

    # check that the logs warned about stride failing
    assert "Stride failed" in completed_process.stderr.decode()
