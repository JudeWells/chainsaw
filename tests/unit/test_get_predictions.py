import shutil
import sys
from pathlib import Path

# ruff: noqa: E402
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.append(f"{REPO_ROOT}")

from get_predictions import predict, load_model, PredictionResult

DEFAULT_DISORDERED_DOMAIN_THRESHOLD = 0.35

def get_length_from_pdb_file(pdb_file):
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('SEQRES'):
                return int(line.split()[3])

def test_predict_from_pdb_file(tmp_path, capsys):

    example_af_id = "AF-A0A1W2PQ64-F1-model_v4"
    expected_structure_file = Path(f'{REPO_ROOT}/example_files/{example_af_id}.pdb')
    expected_nres = get_length_from_pdb_file(expected_structure_file)
    expected_result = PredictionResult(
        pdb_path=expected_structure_file,
        chain_id=example_af_id, 
        sequence_md5='a126e3d4d1a2dcadaa684287855d19f4',
        nres=expected_nres, 
        ndom=3,
        chopping='6-39_94-192,41-90', 
        uncertainty=0.0123,
    )

    tmp_results_dir = tmp_path / "results"
    tmp_results_dir.mkdir()

    expected_model_dir = Path(f'{REPO_ROOT}/saved_models/secondary_structure_epoch17/version_2')
    tmp_model_dir = tmp_path / "models"
    tmp_model_dir.mkdir()
    for model_fname in ['weights.pt', 'config.json']:
        shutil.copyfile(str(expected_model_dir / model_fname), str(tmp_model_dir / model_fname))

    tmp_structure_file = tmp_path / f"{example_af_id}.pdb"
    shutil.copyfile(str(expected_structure_file), str(tmp_structure_file))

    model = load_model(
        model_dir=str(tmp_model_dir), 
        remove_disordered_domain_threshold=DEFAULT_DISORDERED_DOMAIN_THRESHOLD)

    result = predict(model, str(tmp_structure_file))

    assert normalise_result(result) == normalise_result(expected_result)

def normalise_result(res):
    res.pdb_path = '__PDB_PATH__'


