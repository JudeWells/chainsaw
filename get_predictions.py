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
import logging
import hashlib
import os
from pathlib import Path
import sys
import time
import torch
from typing import List

from src import featurisers
from src.utils import common as common_utils
from src.factories import pairwise_predictor

from src.utils.pymol_3d_visuals import generate_pymol_image
from src.models.results import PredictionResult
from src.prediction_result_file import PredictionResultsFile
from src.domain_assignment.utils import convert_domain_dict_strings


LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).parent.resolve()
STRIDE_EXE = os.environ.get('STRIDE_EXE', str(REPO_ROOT / "stride" / "stride"))
PYMOL_EXE = "/Applications/PyMOL.app/Contents/MacOS/PyMOL" # only required if you want to generate 3D images
OUTPUT_COLNAMES = ['chain_id', 'sequence_md5', 'nres', 'ndom', 'chopping', 'uncertainty']

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


def get_model_structure(structure_path, chain='A') -> Bio.PDB.Structure:
    """
    Returns the Bio.PDB.Structure object for a given PDB or MMCIF file
    """
    chain_id = os.path.split(structure_path)[-1].split('.')[0]
    if structure_path.endswith('.pdb'):
        structure = Bio.PDB.PDBParser().get_structure(chain_id, structure_path)
    elif structure_path.endswith('.cif'):
        structure = Bio.PDB.MMCIFParser().get_structure(chain_id, structure_path)
    else:
        raise ValueError(f'Unrecognized file extension: {structure_path}')
    model = structure[0]
    return model


def get_model_structure_sequence(structure_model: Bio.PDB.Structure, chain='A') -> str:
    """
    Returns the MD5 hash of a given PDB or MMCIF structure
    """
    residues = [c for c in structure_model[chain].child_list]
    _3to1 = Bio.PDB.Polypeptide.protein_letters_3to1
    sequence = ''.join([_3to1[r.get_resname()] for r in residues])
    return sequence


def load_model(*, model_dir: str, remove_disordered_domain_threshold: float = 0.35,
                    min_ss_components: int = 2, min_domain_length: int = 30):
    config = common_utils.load_json(os.path.join(model_dir, "config.json"))
    config["learner"]["remove_disordered_domain_threshold"] = remove_disordered_domain_threshold
    config["learner"]["post_process_domains"] = True
    config["learner"]["min_ss_components"] = min_ss_components
    config["learner"]["min_domain_length"] = min_domain_length
    learner = pairwise_predictor(config["learner"], output_dir=model_dir)
    learner.eval()
    learner.load_checkpoints()
    return learner


def predict(model, pdb_path, renumber_pdbs=True) -> List[PredictionResult]:
    """
    Makes the prediction and returns a list of PredictionResult objects
    """
    start = time.time()

    # get model structure metadata
    pdbchain = "A"
    model_structure = get_model_structure(pdb_path, chain=pdbchain)
    model_structure_seq = get_model_structure_sequence(model_structure)
    model_structure_md5 = hashlib.md5(model_structure_seq.encode('utf-8')).hexdigest()

    x = featurisers.inference_time_create_features(
        pdb_path,
        chain=pdbchain,
        secondary_structure=True,
        renumber_pdbs=renumber_pdbs,
        model_structure=model_structure,
    )

    A_hat, domain_dict, uncertainty_array = model.predict(x)
    names_str, bounds_str = convert_domain_dict_strings(domain_dict[0])
    uncertainty = uncertainty_array[0]

    if names_str == "":
        names = bounds = ()
    else:
        names = names_str.split('|')
        bounds = bounds_str.split('|')

    assert len(names) == len(bounds)

    # gather choppings into segments in domains 
    chopping_segs_by_domain = {}
    for domain_id, chopping in zip(names, bounds):
        if domain_id not in chopping_segs_by_domain:
            chopping_segs_by_domain[domain_id] = []
        chopping_segs_by_domain[domain_id].append(chopping)

    # convert list of segments "start-end" into chopping string for the domain 
    # (join distontiguous segs with "_")
    chopping_str_by_domain = { domid: '_'.join(segs) for domid, segs in chopping_segs_by_domain.items() }

    # sort domain choppings by the start residue in first segment
    sorted_domain_chopping_strs = sorted(chopping_str_by_domain.values(), key=lambda x: int(x.split('-')[0]))

    # convert to string (join domains with ",")
    chopping_str = ','.join(sorted_domain_chopping_strs)

    num_domains = len(chopping_str_by_domain)
    if num_domains == 0:
        chopping_str = None

    result = PredictionResult(pdb_path=pdb_path,
                                sequence_md5=model_structure_md5,
                                nres=len(model_structure_seq),
                                ndom=num_domains,
                                chopping=chopping_str,
                                uncertainty=uncertainty)

    runtime = time.time() - start
    LOG.info(f"Runtime: {round(runtime, 3)}s")
    return result


def write_pymol_script(results: List[PredictionResult],
                       save_dir: Path,
                       default_chain_id="A"):

    # group the results by pdb_path
    results_by_pdb_path = {}
    for result in results:
        if result.pdb_path.name not in results_by_pdb_path:
            results_by_pdb_path[result.pdb_path.name] = []
        results_by_pdb_path[result.pdb_path.name].append(result)

    for pdb_filename, results in results_by_pdb_path.items():

        names = "|".join([result.domain_id for result in results])
        bounds = "|".join([result.chopping for result in results])
        fname = Path(pdb_filename).stem

        LOG.info(f"Generating pymol script for {fname} ({len(results)} domains: {bounds})")
        generate_pymol_image(
            pdb_path=str(results[0].pdb_path),
            chain=default_chain_id,
            names=names,
            bounds=bounds,
            image_out_path=os.path.join(str(save_dir), f'{fname}.png'),
            path_to_script=os.path.join(str(save_dir), 'image_gen.pml'),
            pymol_executable=PYMOL_EXE,
        )


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
            'uncertainty': f'{res.uncertainty:.3g}' if res.uncertainty is not None else 'NULL',
        }
        csv_writer.writerow(row)


def get_csv_writer(file_pointer):
    csv_writer = csv.DictWriter(file_pointer,
                                fieldnames=OUTPUT_COLNAMES,
                                delimiter='\t')
    return csv_writer


def main(args):
    outer_save_dir = args.save_dir
    input_method = get_input_method(args)
    model = load_model(
        model_dir=args.model_dir,
        remove_disordered_domain_threshold=args.remove_disordered_domain_threshold,
        min_ss_components=args.min_ss_components,
        min_domain_length=args.min_domain_length,
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
            result = predict(model, pdb_path)
            prediction_results_file.add_result(result)
            if args.pymol_visual:
                write_pymol_script(results=[result], save_dir=outer_save_dir)
    elif input_method == 'structure_file':
        result = predict(model, args.structure_file)
        prediction_results_file.add_result(result)
        if args.pymol_visual:
            write_pymol_script(results=[result], save_dir=outer_save_dir)
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
                        default=f'{REPO_ROOT}/saved_models/secondary_structure_epoch17/version_2',
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
    parser.add_argument('--remove_disordered_domain_threshold', type=float, default=0.35,
                        help='if the domain is less than this proportion secondary structure, it will be removed')
    parser.add_argument('--min_domain_length', type=int, default=30,
                        help='if the domain has fewer residues than this it will be removed')
    parser.add_argument('--min_ss_components', type=int, default=2,
                        help='if the domain has fewer than this number of distinct secondary structure components,'
                             'it will be removed')
    parser.add_argument('--pymol_visual', dest='pymol_visual', action='store_true',
                        help='whether to generate pymol images')
    args = parser.parse_args()
    return args


if __name__=="__main__":
    setup_logging()
    main(parse_args())
