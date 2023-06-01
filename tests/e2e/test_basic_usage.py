"""
End to end test to check the basic usage 
"""
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = REPO_ROOT / "example_files"

def test_basic_usage(tmp_path):

    # setup test data
    af_id = "AF-A0A1W2PQ64-F1-model_v4"
    example_structure_path = DATA_DIR / f"{af_id}.pdb"
    expected_output = f"""
chain_id	domain_id	chopping	uncertainty
AF-A0A1W2PQ64-F1-model_v4	domain_1	6-39	0.0444
AF-A0A1W2PQ64-F1-model_v4	domain_1	94-192	0.0444
AF-A0A1W2PQ64-F1-model_v4	domain_2	41-90	0.0444
""".strip()
    orig_path = Path.cwd()
    script_path = REPO_ROOT / "get_predictions.py"

    # make sure we can run this in an isolated directory
    results_output = None
    try:
        os.chdir(str(tmp_path))
        # os.chdir(str(orig_path))
        cmd_args = ["python", str(script_path), "--structure_file", str(example_structure_path)]
        print("CWD: " + str(tmp_path))
        print("CMD: " + " ".join(cmd_args))
        completed_process = subprocess.run(cmd_args, check=True, capture_output=True)
        results_output = completed_process.stdout.decode().strip()
    except subprocess.CalledProcessError as e:
        print(f"ERROR: CMD: {e.cmd}")
        print(f"ERROR: STDOUT: {e.stdout.decode()}")
        print(f"ERROR: STDERR: {e.stderr.decode()}")
        raise
    finally:
        os.chdir(str(orig_path))

    # check the process ran successfully
    assert completed_process.returncode == 0
    # check the output file was created
    # check the output file has the expected content
    assert normalise_output(results_output) == normalise_output(expected_output)


def normalise_output(output_str):
    return output_str.replace("\r\n", "\n")