import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.append(f"{REPO_ROOT}")

from get_predictions import predict, load_model, PredictionResult

DEFAULT_DISORDERED_DOMAIN_THRESHOLD = 0.35

def test_predict_from_pdb_file(tmp_path, capsys):

    example_af_id = "AF-A0A1W2PQ64-F1-model_v4"
    expected_results = [
        ['domain_1', '189-237'],
        ['domain_2', '238-353'],
        ['domain_3', '381-505'],
        ['domain_4', '524-658'],
        ['domain_5', '687-826'],
    ]

    tmp_results_dir = tmp_path / "results"
    tmp_results_dir.mkdir()

    expected_model_dir = Path(f'{REPO_ROOT}/saved_models/secondary_structure_epoch17/version_2')
    tmp_model_dir = tmp_path / "models"
    tmp_model_dir.mkdir()
    for model_fname in ['weights.pt', 'config.json']:
        shutil.copyfile(str(expected_model_dir / model_fname), str(tmp_model_dir / model_fname))

    expected_structure_file = Path(f'{REPO_ROOT}/example_files/{example_af_id}.pdb')
    tmp_structure_file = tmp_path / f"{example_af_id}.pdb"
    shutil.copyfile(str(expected_structure_file), str(tmp_structure_file))

    model = load_model(
        model_dir=str(tmp_model_dir), 
        remove_disordered_domain_threshold=DEFAULT_DISORDERED_DOMAIN_THRESHOLD)

    results = predict(model, str(tmp_structure_file))

    assert [r.chain_id for r in results] == [example_af_id for r in expected_results]
    assert [r.domain_id for r in results] == [r[0] for r in expected_results]
    assert [r.chopping for r in results] == [r[1] for r in expected_results]


