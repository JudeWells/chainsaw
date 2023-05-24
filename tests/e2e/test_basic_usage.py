"""
End to end test to check the basic usage 
"""
import os
import subprocess
from pathlib import Path
import time

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = REPO_ROOT / "example_files"

def test_basic_usage(tmp_path):

    # setup test data
    af_id = "AF-Q5T5X7-F1-model_v4"
    example_structure_path = DATA_DIR / f"{af_id}.pdb"
    expected_output = f"""
chain_id	domain_id	chopping	uncertainty
AF-Q5T5X7-F1-model_v4	domain_1	189-237	0.028
AF-Q5T5X7-F1-model_v4	domain_2	238-353	0.0
AF-Q5T5X7-F1-model_v4	domain_3	381-505	0.0
AF-Q5T5X7-F1-model_v4	domain_4	524-658	0.0
AF-Q5T5X7-F1-model_v4	domain_5	687-826	0.0
""".strip()
    orig_path = Path.cwd()
    script_path = REPO_ROOT / "get_predictions.py"

    # make sure we can run this in a temp directory
    stdout = None
    stderr = None
    try:
        os.chdir(str(tmp_path))
        # os.chdir(str(orig_path))
        cmd_args = ["python", str(script_path), "--structure_file", str(example_structure_path)]
        completed_process = subprocess.run(cmd_args, check=True, capture_output=True)
        stdout = completed_process.stdout.decode()
        stderr = completed_process.stderr.decode()
    finally:
        os.chdir(str(orig_path))

    # check the process ran successfully
    assert completed_process.returncode == 0
    # check the output file was created
    # check the output file has the expected content
    assert stdout.strip() == expected_output
