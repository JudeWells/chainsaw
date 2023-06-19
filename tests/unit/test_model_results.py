from pathlib import Path
from src.models.results import PredictionResult

def test_results_without_chain_id():

    mock_pdb_path = Path("/path/to/test_chain_id.pdb")
    mock_chain_id = "test_chain_id"
    mock_md5 = "1234567890abcdef"
    mock_chopping = "12-34_56-78"
    mock_uncertainty = 0.0123

    res = PredictionResult(
        pdb_path=mock_pdb_path,
        sequence_md5=mock_md5,
        nres=1234,
        ndom=1,
        chopping=mock_chopping,
        uncertainty=mock_uncertainty,
    )

    assert res.chain_id == mock_chain_id

