import pytest
from pathlib import Path
from src.models.results import PredictionResult
from src.prediction_result_file import PredictionResultsFile
from src import errors

@pytest.fixture
def create_n_mock_results():
    """
    Returns a fixture that creates n mock results
    """
    def _create_n_mock_results(n):
        return [ 
            PredictionResult(
                pdb_path=Path(f"mock_path_{i}.pdb"),
                chain_id=f"mock_chain_id_{i}",
                sequence_md5=f"mock_md5_{i}",
                ndom=1,
                nres=1234,
                chopping="12-34_56-78,90-123",
                uncertainty=0.0123,
            ) for i in range(n)
        ]
    return _create_n_mock_results


def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)

def test_chunked_results(tmp_path, create_n_mock_results):

    expected_path = tmp_path / "expected_results.tsv"
    result_file = PredictionResultsFile(expected_path, chunk_size=20)
    results = create_n_mock_results(50)
    expected_headers = '\t'.join([
        'chain_id', 'sequence_md5', 'nres', 'ndom', 'chopping', 'uncertainty'
    ])

    assert expected_path.exists() == False

    results_before_chunk = results[0:19]
    assert len(results_before_chunk) == 19
    result_file.add_results(results_before_chunk)
    assert expected_path.exists() == False

    results_after_chunk = results[19:20]
    assert len(results_after_chunk) == 1
    result_file.add_results(results_after_chunk)
    assert expected_path.exists() == True

    lines = expected_path.read_text().split('\n')
    firstline = lines[0]
    assert firstline == expected_headers
    assert 'mock_chain_id_0' in lines[1]
    assert 'mock_chain_id_19' in lines[20]
    assert count_lines(expected_path) == 21
    

def test_check_add_repeated_result_raises_error(tmp_path, create_n_mock_results):

    expected_path = tmp_path / "expected_results.tsv"
    result_file = PredictionResultsFile(expected_path, chunk_size=20)
    results = create_n_mock_results(10)

    assert expected_path.exists() == False

    result_file.add_result(results[0])
    with pytest.raises(errors.PredictionResultExistsError) as err:
        result_file.add_result(results[0])



@pytest.mark.parametrize("allow_append,exception", [
    (True, None),
    (False, errors.FileExistsError),
])
def test_check_appending_to_existing_file(tmp_path, create_n_mock_results, allow_append, exception):

    expected_path = tmp_path / "expected_results.tsv"
    results = create_n_mock_results(60)

    def get_result_file():
        return PredictionResultsFile(expected_path, chunk_size=20, allow_append=allow_append)

    f1 = get_result_file()

    assert expected_path.exists() == False
    f1.add_results(results[0:25])
    assert expected_path.exists() == True
    assert count_lines(expected_path) == 21
    f1.flush()
    assert count_lines(expected_path) == 26

    def append_results():
        f2 = get_result_file()
        f2.add_results(results[25:50])
        f2.flush()

    if exception is None:
        append_results()
        count_lines(expected_path) == 51
        f3 = get_result_file()
        assert f3.has_result(results[0]) is True
        assert f3.has_result(results[59]) is False
    else:
        with pytest.raises(exception) as err:
            append_results()
