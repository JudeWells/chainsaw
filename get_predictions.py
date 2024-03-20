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
import os
import sys
import time
from pathlib import Path
from typing import List
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
               post_process_domains: bool = True,):
    config = common_utils.load_json(os.path.join(model_dir, "config.json"))
    feature_config = common_utils.load_json(os.path.join(model_dir, "feature_config.json"))
    config["learner"]["remove_disordered_domain_threshold"] = remove_disordered_domain_threshold
    config["learner"]["post_process_domains"] = post_process_domains
    config["learner"]["min_ss_components"] = min_ss_components
    config["learner"]["min_domain_length"] = min_domain_length
    config["learner"]["dist_transform_type"] = config["data"].get("dist_transform", 'min_replace_inverse')
    config["learner"]["distance_denominator"] = config["data"].get("distance_denominator", None)
    learner = pairwise_predictor(config["learner"], output_dir=model_dir)
    learner.feature_config = feature_config
    learner.load_checkpoints()
    learner.eval()
    try:
        learner = torch_compile(learner)
    except:
        pass
    return learner


def predict(model, pdb_path, renumber_pdbs=True, pdbchain=None) -> List[PredictionResult]:
    """
    Makes the prediction and returns a list of PredictionResult objects
    """
    start = time.time()

    # get model structure metadata
    model_structure = featurisers.get_model_structure(pdb_path)

    if pdbchain is None:
        LOG.warning(f"No chain specified for {pdb_path}, using first chain")
        # get all the chain ids from the model structure
        all_chain_ids = [c.id for c in model_structure.get_chains()]
        # take the first chain id
        pdbchain = all_chain_ids[0]

    model_residues = featurisers.get_model_structure_residues(model_structure, chain=pdbchain)
    model_res_label_by_index = { int(r.index): str(r.res_label) for r in model_residues}
    model_structure_seq = "".join([r.aa for r in model_residues])
    model_structure_md5 = hashlib.md5(model_structure_seq.encode('utf-8')).hexdigest()

    x = featurisers.inference_time_create_features(pdb_path,
                                                    feature_config=model.feature_config,
                                                    chain=pdbchain,
                                                    renumber_pdbs=renumber_pdbs,
                                                    model_structure=model_structure,
                                                   )

    A_hat, domain_dict, confidence = model.predict(x)
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
    runtime = round(time.time() - start, 3)
    result = PredictionResult(
        pdb_path=pdb_path,
        sequence_md5=model_structure_md5,
        nres=len(model_structure_seq),
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
    pdb_chain_id = 'A'
    if args.use_first_chain:
        pdb_chain_id = None

    input_method = get_input_method(args)
    model = load_model(
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
        for idx, fname in enumerate(os.listdir(structure_dir)):
            suffix = Path(fname).suffix
            LOG.debug(f"Checking file {fname} (suffix: {suffix}) ..")
            if suffix not in ACCEPTED_STRUCTURE_FILE_SUFFIXES:
                continue

            chain_id = Path(fname).stem
            result_exists = prediction_results_file.has_result_for_chain_id(chain_id)
            if result_exists:
                LOG.info(f"Skipping file {fname} (result for '{chain_id}' already exists)")
                continue

            pdb_path = os.path.join(structure_dir, fname)
            LOG.info(f"Making prediction for file {fname} (chain '{chain_id}')")
            result = predict(model, pdb_path, pdbchain=pdb_chain_id, renumber_pdbs=args.renumber_pdbs)
            prediction_results_file.add_result(result)
            if args.pymol_visual:
                generate_pymol_image(
                    pdb_path=str(result.pdb_path),
                    chopping=result.chopping or '',
                    image_out_path=os.path.join(str(outer_save_dir), f'{result.pdb_path.name.replace(".pdb", "")}.png'),
                    path_to_script=os.path.join(str(outer_save_dir), 'image_gen.pml'),
                    pymol_executable=constants.PYMOL_EXE,
                )
    elif input_method == 'structure_file':
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
    parser.add_argument('--use_first_chain', default=False, action="store_true", help='use the first chain in the structure (rather than "A")')
    parser.add_argument('--renumber_pdbs', default=False, action="store_true", help='renumber pdb files')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    setup_logging()
    main(parse_args())
