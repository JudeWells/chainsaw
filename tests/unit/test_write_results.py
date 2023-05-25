from pathlib import Path
import io
import csv
from src.models.results import PredictionResult
from get_predictions import get_csv_writer, write_csv_results, OUTPUT_COLNAMES


def test_write_csv_results():

    mock_pdb_path = Path("fake.pdb")
    expected_results_text = """
AF-Q5T5X7-F1-model_v4	domain_1	189-237	0.01
AF-Q5T5X7-F1-model_v4	domain_2	238-353	0.002
AF-Q5T5X7-F1-model_v4	domain_3	381-505	0.0234
AF-Q5T5X7-F1-model_v4	domain_4	524-658	0.0135
AF-Q5T5X7-F1-model_v4	domain_5	687-826	0.00342
    """.strip() + '\n'

    results = []
    for line in expected_results_text.split('\n'):
        if line == "":
            continue
        cols = line.split()
        res = PredictionResult(
            pdb_path=mock_pdb_path, 
            chain_id=cols[0], 
            domain_id=cols[1], 
            chopping=cols[2], 
            uncertainty=float(cols[3])
        )
        results.append(res)

    assert len(results) == 5
    assert isinstance(results[0], PredictionResult) 

    fp = io.StringIO()
    csv_writer = get_csv_writer(fp)
    write_csv_results(csv_writer, results)
    fp.flush()
    fp.seek(0)

    actual_text = fp.read().replace('\r\n', '\n')
    assert actual_text == expected_results_text


