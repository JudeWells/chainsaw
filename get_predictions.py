"""
Script for running Chainsaw

Created by: Jude Wells 2023-04-19

User can provide any of the following as an input to get predictions:
    - a single uniprot id (alphafold model will be downloaded and parsed)
    - a list of uniprot ids (alphafold model will be downloaded and parsed)
    - a list of pdb ids (alphafold model will be downloaded and parsed)
    - a path to a directory with PDBs or MMCIF files
"""

import argparse
import csv
import hashlib
import logging
import multiprocessing
import os
import signal
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import compile as torch_compile

from src import constants, featurisers
from src.domain_assignment.util import convert_domain_dict_strings
from src.factories import pairwise_predictor
from src.models.results import PredictionResult
from src.prediction_result_file import PredictionResultsFile
from src.utils import common as common_utils
from src.utils.pymol_3d_visuals import generate_pymol_image


LOG = logging.getLogger(__name__)
OUTPUT_COLNAMES = ['chain_id', 'sequence_md5', 'nres', 'ndom', 'chopping', 'confidence', 'time_sec']
ACCEPTED_STRUCTURE_FILE_SUFFIXES = ['.pdb', '.cif']


def setup_logging():
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    # log all messages to stderr so results can be sent to stdout
    logging.basicConfig(level=loglevel,
                    stream=sys.stderr,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


def get_input_method(args):
    number_of_input_methods = sum([ args.uniprot_id is not None,
                                    args.uniprot_id_list_file is not None,
                                    args.structure_directory is not None,
                                    args.structure_file is not None,
                                    args.pdb_id_list_file is not None,
                                    args.pdb_id is not None])
    if number_of_input_methods != 1:
        raise ValueError('Exactly one input method must be provided')
    if args.uniprot_id is not None:
        return 'uniprot_id'
    elif args.uniprot_id_list_file is not None:
        return 'uniprot_id_list_file'
    elif args.structure_directory is not None:
        return 'structure_directory'
    elif args.structure_file is not None:
        return 'structure_file'
    else:
        raise ValueError('No input method provided')

def load_model(*,
               model_dir: str,
               remove_disordered_domain_threshold: float = 0.35,
               min_ss_components: int = 2,
               min_domain_length: int = 30,
               post_process_domains: bool = True,
               force_cpu: bool = False):
    config = common_utils.load_json(os.path.join(model_dir, "config.json"))
    feature_config = common_utils.load_json(os.path.join(model_dir, "feature_config.json"))
    config["learner"]["remove_disordered_domain_threshold"] = remove_disordered_domain_threshold
    config["learner"]["post_process_domains"] = post_process_domains
    config["learner"]["min_ss_components"] = min_ss_components
    config["learner"]["min_domain_length"] = min_domain_length
    config["learner"]["dist_transform_type"] = config["data"].get("dist_transform", 'min_replace_inverse')
    config["learner"]["distance_denominator"] = config["data"].get("distance_denominator", None)
    learner = pairwise_predictor(config["learner"], force_cpu=force_cpu, output_dir=model_dir)
    learner.feature_config = feature_config
    learner.load_checkpoints()
    learner.eval()
    try:
        learner = torch_compile(learner)
    except Exception:
        pass
    return learner


@dataclass
class PreparedFeatures:
    """CPU-prepared inputs and residue metadata for one chain (picklable across worker processes)."""

    pdb_path: str
    res_label_by_index: Dict[int, str]
    nres: int
    sequence_md5: str
    x: Optional[np.ndarray]
    elapsed: float


def predict(model, pdb_path, renumber_pdbs=True, pdbchain=None) -> PredictionResult:
    """Make a domain prediction for one structure file."""
    prepared = prepare_features(pdb_path, model.feature_config, renumber_pdbs=renumber_pdbs, pdbchain=pdbchain)
    start = time.time()
    _, domain_dict, confidence = model.predict(torch.from_numpy(prepared.x))
    return _make_result(replace(prepared, elapsed=prepared.elapsed + time.time() - start), domain_dict, confidence)


def prepare_features(pdb_path, feature_config, renumber_pdbs=True, pdbchain=None) -> PreparedFeatures:
    """Run the CPU preprocessing stage of `predict`: parse structure, extract residues, and build input tensor."""
    start = time.time()
    pdbchain, model_residues, dist_matrix = featurisers.load_chain(pdb_path, chain=pdbchain)
    model_res_label_by_index = {int(r.index): str(r.res_label) for r in model_residues}
    model_structure_seq = "".join(r.aa for r in model_residues)
    model_structure_md5 = hashlib.md5(model_structure_seq.encode('utf-8')).hexdigest()

    x = featurisers.inference_time_create_features(
        pdb_path,
        feature_config=feature_config,
        chain=pdbchain,
        renumber_pdbs=renumber_pdbs,
        dist_matrix=dist_matrix,
    )
    return PreparedFeatures(
        pdb_path=pdb_path,
        res_label_by_index=model_res_label_by_index,
        nres=len(model_structure_seq),
        sequence_md5=model_structure_md5,
        x=x.numpy(),
        elapsed=time.time() - start,
    )


def _make_result(prepared: PreparedFeatures, domain_dict, confidence) -> PredictionResult:
    """Format the domain assignment of one chain (the end of the original `predict`)."""
    start = time.time()
    pdb_path = prepared.pdb_path
    model_res_label_by_index = prepared.res_label_by_index
    model_structure_md5 = prepared.sequence_md5
    # Convert 0-indexed to 1-indexed to match AlphaFold indexing:
    domain_dict = [{k: [r + 1 for r in v] for k, v in d.items()} for d in domain_dict]
    names_str, bounds_str = convert_domain_dict_strings(domain_dict[0])
    confidence = confidence[0]

    if names_str == "":
        names = bounds = ()
    else:
        names = names_str.split('|')
        bounds = bounds_str.split('|')

    assert len(names) == len(bounds)

    class Seg:
        def __init__(self, domain_id: str, start_index: int, end_index: int):
            self.domain_id = domain_id
            self.start_index = int(start_index)
            self.end_index = int(end_index)
        
        def res_label_of_index(self, index: int):
            if index not in model_res_label_by_index:
                raise ValueError(f"Index {index} not in model_res_label_by_index ({model_res_label_by_index})")
            return model_res_label_by_index[int(index)]

        @property
        def start_label(self):
            return self.res_label_of_index(self.start_index)
        
        @property
        def end_label(self):
            return self.res_label_of_index(self.end_index)

    class Dom:
        def __init__(self, domain_id, segs: List[Seg] = None):
            self.domain_id = domain_id
            if segs is None:
                segs = []
            self.segs = segs

        def add_seg(self, seg: Seg):
            self.segs.append(seg)

    # gather choppings into segments in domains
    domains_by_domain_id = {}
    for domain_id, chopping_by_index in zip(names, bounds):
        if domain_id not in domains_by_domain_id:
            domains_by_domain_id[domain_id] = Dom(domain_id)
        start_index, end_index = chopping_by_index.split('-')
        seg = Seg(domain_id, start_index, end_index)
        domains_by_domain_id[domain_id].add_seg(seg)

    # sort domain choppings by the start residue in first segment
    domains = sorted(domains_by_domain_id.values(), key=lambda dom: dom.segs[0].start_index)

    # collect domain choppings as strings
    domain_choppings = []
    for dom in domains:
        # convert segments to strings
        segs_str = [f"{seg.start_label}-{seg.end_label}" for seg in dom.segs]
        segs_index_str = [f"{seg.start_index}-{seg.end_index}" for seg in dom.segs]
        LOG.info(f"Segments (index to label): {segs_index_str} -> {segs_str}")
        # join discontinuous segs with '_' 
        domain_choppings.append('_'.join(segs_str))

    # join domains with ','
    chopping_str = ','.join(domain_choppings)

    num_domains = len(domain_choppings)
    if num_domains == 0:
        chopping_str = None
    runtime = round(prepared.elapsed + time.time() - start, 3)
    result = PredictionResult(
        pdb_path=pdb_path,
        sequence_md5=model_structure_md5,
        nres=prepared.nres,
        ndom=num_domains,
        chopping=chopping_str,
        confidence=confidence,
        time_sec=runtime,
    )

    LOG.info(f"Runtime: {round(runtime, 3)}s")
    return result


def write_csv_results(csv_writer, prediction_results: List[PredictionResult]):
    """
    Render list of PredictionResult results to file pointer
    """
    for res in prediction_results:
        row = {
            'chain_id': res.chain_id,
            'sequence_md5': res.sequence_md5,
            'nres': res.nres,
            'ndom': res.ndom,
            'chopping': res.chopping if res.chopping is not None else 'NULL',
            'confidence': f'{res.confidence:.3g}' if res.confidence is not None else 'NULL',
            'time_sec': f'{res.time_sec}' if res.time_sec is not None else 'NULL',
        }
        csv_writer.writerow(row)


def get_csv_writer(file_pointer):
    csv_writer = csv.DictWriter(file_pointer,
                                fieldnames=OUTPUT_COLNAMES,
                                delimiter='\t')
    return csv_writer


def main(args):
    outer_save_dir = args.save_dir
    pdb_chain_id = None if args.use_first_chain else 'A'

    input_method = get_input_method(args)
    for required in ("config.json", "feature_config.json"):
        required_path = os.path.join(args.model_dir, required)
        if not os.path.isfile(required_path):
            raise FileNotFoundError(f"Model file not found: '{required_path}'")
    model_kwargs = dict(
        model_dir=args.model_dir,
        remove_disordered_domain_threshold=args.remove_disordered_domain_threshold,
        min_ss_components=args.min_ss_components,
        min_domain_length=args.min_domain_length,
        post_process_domains=args.post_process_domains,
    )
    os.makedirs(outer_save_dir, exist_ok=True)
    output_path = Path(args.output).absolute()

    prediction_results_file = PredictionResultsFile(
        csv_path=output_path,
        # use args.allow_append to mean allow_skip and allow_append
        allow_append=args.allow_append,
        allow_skip=args.allow_append,
    )

    if input_method == 'structure_directory':
        structure_dir = args.structure_directory
        pdb_paths = []
        seen_chain_ids = set()  # match upstream: skip duplicate stems (e.g. foo.pdb + foo.cif) in the same run
        for fname in os.listdir(structure_dir):
            suffix = Path(fname).suffix
            LOG.debug(f"Checking file {fname} (suffix: {suffix}) ..")
            if suffix not in ACCEPTED_STRUCTURE_FILE_SUFFIXES:
                continue

            chain_id = Path(fname).stem
            if chain_id in seen_chain_ids or prediction_results_file.has_result_for_chain_id(chain_id):
                LOG.info(f"Skipping file {fname} (result for '{chain_id}' already exists)")
                continue
            seen_chain_ids.add(chain_id)
            pdb_paths.append(os.path.join(structure_dir, fname))

        if pdb_paths:
            for result in _predict_all(pdb_paths, model_kwargs, pdb_chain_id, args.renumber_pdbs, args.workers):
                prediction_results_file.add_result(result)
                if args.pymol_visual:
                    generate_pymol_image(
                        pdb_path=str(result.pdb_path),
                        chopping=result.chopping or '',
                        image_out_path=os.path.join(
                            str(outer_save_dir),
                            f'{result.pdb_path.name.replace(".pdb", "")}.png',
                        ),
                        path_to_script=os.path.join(str(outer_save_dir), 'image_gen.pml'),
                        pymol_executable=constants.PYMOL_EXE,
                    )
    elif input_method == 'structure_file':
        model = load_model(**model_kwargs)
        result = predict(model, args.structure_file, pdbchain=pdb_chain_id)
        prediction_results_file.add_result(result)
        if args.pymol_visual:
            generate_pymol_image(
                pdb_path=str(result.pdb_path),
                chopping=result.chopping or '',
                image_out_path=os.path.join(str(outer_save_dir), f'{result.pdb_path.name.replace(".pdb", "")}.png'),
                path_to_script=os.path.join(str(outer_save_dir), 'image_gen.pml'),
                pymol_executable=constants.PYMOL_EXE,
            )
    else:
        raise NotImplementedError('Not implemented yet')

    prediction_results_file.flush()
    LOG.info("DONE")


_WORKER_MODEL = None  # per-worker CPU model


@contextmanager
def _worker_pool(workers: int, model_kwargs: dict, threads: int):
    """Create a ProcessPoolExecutor that avoids GPU inheritance and cleans up workers on error or SIGTERM."""
    methods = multiprocessing.get_all_start_methods()
    # Use forkserver/spawn so workers never fork from a CUDA-initialized main process.
    mp_ctx = multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_ctx,
        initializer=_init_worker,
        initargs=(model_kwargs, threads),
    )
    # Convert SIGTERM into SystemExit so scheduler timeouts / `kill` run worker cleanup (otherwise
    # sibling workers keep the call-queue pipe open and survive parent SIGTERM).
    prev_sigterm = signal.signal(signal.SIGTERM, lambda signum, _f: sys.exit(128 + signum))
    try:
        yield executor
    except BaseException:
        for proc in list((executor._processes or {}).values()):
            proc.terminate()
        raise
    finally:
        executor.shutdown(wait=True)
        signal.signal(signal.SIGTERM, prev_sigterm)


def _init_worker(model_kwargs, threads):
    if not logging.getLogger().handlers:
        setup_logging()
    torch.set_num_threads(threads)
    global _WORKER_MODEL
    _WORKER_MODEL = load_model(**model_kwargs, force_cpu=True)


def _decode_worker(task):
    """Run domain assignment, post-processing, and formatting in a worker process."""
    start = time.time()
    prepared, y_pred, x_post = task
    x_tensor = torch.from_numpy(x_post) if x_post is not None else None
    domain_dicts, confidence = _WORKER_MODEL.assign_domains(torch.from_numpy(y_pred), x_tensor)
    return _make_result(replace(prepared, elapsed=prepared.elapsed + time.time() - start), domain_dicts, confidence)


def _predict_worker(task):
    """Run a full prediction in a worker process (CPU inference)."""
    pdb_path, renumber_pdbs, pdbchain = task
    _log_making_prediction(pdb_path)
    return predict(_WORKER_MODEL, pdb_path, renumber_pdbs=renumber_pdbs, pdbchain=pdbchain)


def _run_network(model, prepared: PreparedFeatures):
    """Run the network stage of `PairwiseDomainPredictor.predict` in the GPU process."""
    _log_making_prediction(prepared.pdb_path)
    start = time.time()
    y_pred, x = model.predict_network(torch.from_numpy(prepared.x))
    x_post = x.cpu().numpy() if model.post_process_domains else None
    meta = replace(prepared, x=None, elapsed=prepared.elapsed + time.time() - start)
    return meta, y_pred.cpu().numpy(), x_post


def _predict_all(pdb_paths, model_kwargs, pdbchain, renumber_pdbs, workers=1):
    """Predict each file in input order using serial, CPU-worker, or GPU-pipeline execution."""
    n_workers = min(workers, len(pdb_paths))
    if n_workers <= 1:
        model = load_model(**model_kwargs)
        for pdb_path in pdb_paths:
            _log_making_prediction(pdb_path)
            yield predict(model, pdb_path, pdbchain=pdbchain, renumber_pdbs=renumber_pdbs)
        return
    use_gpu = torch.cuda.is_available()
    window = 2 * n_workers  # per-stage bound (<= 4 * n_workers across prepare + decode)
    threads = 1 if use_gpu else max(1, torch.get_num_threads() // n_workers)
    with _worker_pool(n_workers, model_kwargs, threads) as pool:
        if not use_gpu:
            tasks = [(p, renumber_pdbs, pdbchain) for p in pdb_paths]
            yield from _ordered_map(pool, _predict_worker, tasks, window)
            return
        model = load_model(**model_kwargs)
        prep = partial(
            prepare_features, feature_config=model.feature_config, renumber_pdbs=renumber_pdbs, pdbchain=pdbchain
        )
        prepared = _ordered_map(pool, prep, pdb_paths, window)
        yield from _ordered_map(pool, _decode_worker, (_run_network(model, p) for p in prepared), window)


def _ordered_map(pool, fn, tasks, window):
    """Submit tasks to `pool` in order, keeping at most `window` tasks in flight at once."""
    pending = deque()
    for task in tasks:
        pending.append(pool.submit(fn, task))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def _log_making_prediction(pdb_path):
    fname = os.path.basename(pdb_path)
    LOG.info(f"Making prediction for file {fname} (chain '{Path(fname).stem}')")


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        default=f'{constants.REPO_ROOT}/saved_models/model_v3',
                        help='path to model directory must contain model.pt and config.json')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='write results to this file')
    parser.add_argument('--uniprot_id', type=str, default=None, help='single uniprot id')
    parser.add_argument('--uniprot_id_list_file', type=str, default=None,
                        help='path to file containing uniprot ids')
    parser.add_argument('--structure_directory', type=str, default=None,
                        help='path to directory containing PDB or MMCIF files')
    parser.add_argument('--structure_file', type=str, default=None,
                        help='path to PDB or MMCIF files')
    parser.add_argument('--append', '-a', dest='allow_append', action='store_true', default=False, 
                        help='allow results to be appended to an existing file')
    parser.add_argument('--pdb_id', type=str, default=None, help='single pdb id')
    parser.add_argument('--pdb_id_list_file', type=str, default=None, help='path to file containing uniprot ids')
    parser.add_argument('--save_dir', type=str, default='results', help='path where results and images will be saved')
    parser.add_argument('--no_post_processing', dest='post_process_domains', action='store_false')
    parser.add_argument('--remove_disordered_domain_threshold', type=float, default=0.35,
                        help='if the domain is less than this proportion secondary structure, it will be removed')
    parser.add_argument('--min_domain_length', type=int, default=30,
                        help='if the domain has fewer residues than this it will be removed')
    parser.add_argument('--min_ss_components', type=int, default=2,
                        help='if the domain has fewer than this number of distinct secondary structure components,'
                             'it will be removed')
    parser.add_argument('--pymol_visual', dest='pymol_visual', action='store_true',
                        help='whether to generate pymol images')
    parser.add_argument('--use_first_chain', default=True, action="store_true", help='use the first chain in the structure (rather than "A")')
    parser.add_argument('--renumber_pdbs', default=False, action="store_true", help='renumber pdb files')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of worker processes for --structure_directory (1 = single process); '
                             'produces identical results in the same order')

    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be >= 1')
    return args


if __name__ == "__main__":
    setup_logging()
    main(parse_args())
