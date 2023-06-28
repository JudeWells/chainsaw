"""
End to end test to check the basic usage 
"""
import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = REPO_ROOT / "example_files"

def test_basic_usage(tmp_path):

    # setup test data
    af_id = "AF-A0A1W2PQ64-F1-model_v4"
    example_structure_path = DATA_DIR / f"{af_id}.pdb"
    expected_cols = [
        ['chain_id', 'sequence_md5', 'nres', 'ndom', 'chopping', 'uncertainty'],
        ['AF-A0A1W2PQ64-F1-model_v4', 'a126e3d4d1a2dcadaa684287855d19f4', '194', '2', '6-39_94-192,41-90', '0.0444'],
    ]
    expected_output = "\r\n".join(["\t".join(row) for row in expected_cols])
    orig_path = Path.cwd()
    script_path = REPO_ROOT / "get_predictions.py"

    results_file = tmp_path / "results.tsv"

    def run_chainsaw(cmd_args):
        completed_process = None
        results_output = None
        try:
            os.chdir(str(tmp_path))
            shutil.copy2(str(example_structure_path), ".")
            completed_process = subprocess.run(cmd_args, check=False, capture_output=True)
            if results_file.exists():
                results_output = results_file.read_text().strip()
        except subprocess.CalledProcessError as e:
            print(f"ERROR: CMD: {e.cmd}")
            print(f"ERROR: STDOUT: {e.stdout.decode()}")
            print(f"ERROR: STDERR: {e.stderr.decode()}")
        finally:
            os.chdir(str(orig_path))

        return completed_process, results_output
 
    base_args =  ["python", str(script_path), "--structure_directory", ".", "-o", str(results_file)]

    # make sure we can run this in an isolated directory
    completed_process, results_output = run_chainsaw(base_args)
    assert completed_process.returncode == 0

    # check the output file has the expected content
    assert normalise_output(results_output) == normalise_output(expected_output)

    # run the same thing again and make sure that it complains about the result file existing
    completed_process, results_output = run_chainsaw(base_args)
    assert completed_process.returncode != 0

    # run the same thing again allowing for updates 
    # and make sure that we skip results that have already been computed
    completed_process, results_output = run_chainsaw(base_args + ["--append"])
    assert completed_process.returncode == 0
    assert normalise_output(results_output) == normalise_output(expected_output)

    # check that we have skipped the result that has already been computed
    chainsaw_logs = completed_process.stderr.decode()
    assert f"result for '{af_id}' already exists" in chainsaw_logs


def normalise_output(output_str):
    return output_str.replace("\r\n", "\n")