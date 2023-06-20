from pathlib import Path
import io
from src.models.results import PredictionResult
from get_predictions import get_csv_writer, write_csv_results


def test_write_csv_results():

    mock_pdb_path = Path("mock.pdb")
    chain_id = 'mock_id'
    expected_sequence_md5 = 'mock_md5'
    expected_chopping_str = '189-237,238-353,381-505,524-658,687-826'
    expected_ndom = len(expected_chopping_str.split(','))
    expected_nres = 1234
    expected_uncertainty = 0.0123

    result = PredictionResult(
        pdb_path=mock_pdb_path, 
        chain_id=chain_id, 
        sequence_md5=expected_sequence_md5,
        ndom=expected_ndom,
        nres=expected_nres,
        chopping=expected_chopping_str,
        uncertainty=expected_uncertainty,
    )

    fp = io.StringIO()
    csv_writer = get_csv_writer(fp)
    write_csv_results(csv_writer, [result])
    fp.flush()
    fp.seek(0)

    expected_cols = [chain_id, expected_sequence_md5, expected_nres, 
                     expected_ndom, expected_chopping_str, expected_uncertainty]
    actual_text = fp.read().replace('\r\n', '\n')
    assert actual_text == '\t'.join([str(col) for col in expected_cols]) + '\n'


