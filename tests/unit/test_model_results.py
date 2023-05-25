from pathlib import Path
from src.models.results import PredictionResult

def test_results_without_chain_id():

    mock_pdb_path = Path("/path/to/test_chain_id.pdb")
    mock_chain_id = "test_chain_id"
    mock_domain_id = "test_domain_id"
    mock_chopping = "test_chopping"
    mock_uncertainty = 0.0123

    res = PredictionResult(
        pdb_path=mock_pdb_path,
        domain_id=mock_domain_id,
        chopping=mock_chopping,
        uncertainty=mock_uncertainty,
    )

    assert res.chain_id == mock_chain_id

