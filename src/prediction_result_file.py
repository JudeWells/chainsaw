"""
Class that represents a file with the prediction results

* allows prediction results to be appended to the file in chunks
* allows results to be skipped if they already exist in the file
"""

import csv
from typing import List, OrderedDict
from pathlib import Path
import logging

from .models.results import PredictionResult
from .errors import PredictionResultExistsError, FileExistsError

LOG = logging.getLogger(__name__)

class PredictionResultsFileIter:
    def __init__(self, results_file_class):
        self._results = results_file_class.get_results()
        self._results_size = len(self._results)
        self._current_index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._current_index < self._results_size:
            member = self._results[self._current_index] 
            self._current_index += 1
            return member
        raise StopIteration


class PredictionResultsFile:
    """
    Writes prediction results to a file in chunks
    """
    
    COLNAMES = ['chain_id', 'sequence_md5', 'nres', 'ndom', 'chopping', 'uncertainty']

    def __init__(self, 
                 csv_path: Path, *, 
                 chunk_size: int=20, 
                 write_header: bool=True, 
                 allow_append: bool=False, 
                 allow_skip: bool=False):

        self.csv_path = Path(str(csv_path)).absolute()
        self.chunk_size = chunk_size
        self.write_header = write_header
        self.allow_append = allow_append
        self.allow_skip = allow_skip
        self._results_by_id: OrderedDict[PredictionResult] = OrderedDict()
        self._unflushed_results_by_id: OrderedDict[PredictionResult] = OrderedDict()
        self._written_header = False

        self._init()

    def _init(self):
        if self.csv_path.exists():
            msg = f"file '{self.csv_path}' already exists (allow_append={self.allow_append}))"
            LOG.warning(msg)
            if not self.allow_append:
                raise FileExistsError(msg)
            self._read_results()

    def _read_results(self):
        """
        Read the results from the file
        """
        with self.csv_path.open('r') as fp:
            reader = self.get_csv_reader(fp)
            for row in reader:
                if row['chain_id'] == 'chain_id':
                    continue
                pdb_path = Path(row['chain_id'] + '.pdb')
                result = PredictionResult(pdb_path=pdb_path, **row)
                self._results_by_id[result.chain_id] = result

    def get_flushed_results(self):
        self.flush()
        return self._results_by_id.values()

    def get_csv_reader(self, fp):
        return csv.DictReader(fp, delimiter='\t', fieldnames=self.COLNAMES)

    def get_csv_writer(self, fp):
        return csv.DictWriter(fp, delimiter='\t', fieldnames=self.COLNAMES)

    def write_csv_result(self, csv_writer, res: PredictionResult):
        """
        Render PredictionResult result to file pointer
        """
        row = {
            'chain_id': res.chain_id,
            'sequence_md5': res.sequence_md5,
            'nres': res.nres,
            'ndom': res.ndom,
            'chopping': res.chopping if res.chopping is not None else 'NULL',
            'uncertainty': f'{res.uncertainty:.3g}' if res.uncertainty is not None else 'NULL',
        }
        csv_writer.writerow(row)


    def flush(self):
        """
        Write unflushed results to the output file
        """
        if len(self._unflushed_results_by_id) == 0:
            return

        with self.csv_path.open('a') as fp:

            csv_writer = self.get_csv_writer(fp)

            if self.write_header and not self._written_header and len(self._results_by_id) == 0:
                csv_writer.writeheader()
                self._written_header = True

            for result in self._unflushed_results_by_id.values():
                self.write_csv_result(csv_writer, result)

    def has_result(self, result: PredictionResult):
        """
        Check if the prediction result already exists in the file
        """
        return self.has_result_for_chain_id(result.chain_id)

    def has_result_for_chain_id(self, chain_id: str):
        """
        Check if the chain_id already exists in the file
        """
        # msg = (
        #     f"checking if result '{chain_id}' exists "
        #     f"({len(self._results_by_id)} results, {len(self._results_by_id)} unflushed results)"
        # )
        # LOG.debug(msg)
        if chain_id in self._results_by_id:
            return True
        if chain_id in self._unflushed_results_by_id:
            return True

        return False

    def add_result(self, result: PredictionResult):
        """
        Add a result to the buffer (and possibly flush the buffer)
        """
        if self.has_result(result):
            msg = f"result '{result.chain_id}' already exists in file '{self.csv_path}'"
            raise PredictionResultExistsError(msg)

        self._unflushed_results_by_id[result.chain_id] = result
        if len(self._unflushed_results_by_id) >= self.chunk_size:
            self.flush()
            self._results_by_id.update(self._unflushed_results_by_id)
            self._unflushed_results_by_id = OrderedDict()

    def add_results(self, results: List[PredictionResult]):
        for res in results:
            self.add_result(res)

    def __iter__(self):
        return PredictionResultsFileIter(self)
