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
import numpy as np
import os
from pathlib import Path
import sys
import time
import torch
from typing import List

import Bio.PDB

from src.create_features.make_2d_features import calc_dist_matrix
from src.utils import common as common_utils
from src.factories import pairwise_predictor
from src.utils.cif2pdb import cif2pdb
from src.create_features.secondary_structure.secondary_structure_features import renum_pdb_file, calculate_ss, make_ss_matrix
from src.utils.pymol_3d_visuals import generate_pymol_image
from src.models.results import PredictionResult

LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).parent.resolve()
STRIDE_EXE = os.environ.get('STRIDE_EXE', str(REPO_ROOT / "stride" / "stride"))
PYMOL_EXE = "/Applications/PyMOL.app/Contents/MacOS/PyMOL" # only required if you want to generate 3D images
OUTPUT_COLNAMES = ['chain_id', 'domain_id', 'chopping', 'uncertainty']

def setup_logging():
    # log all messages to stderr so results can be sent to stdout
    logging.basicConfig(level=logging.INFO,
                    stream=sys.stderr,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

def inference_time_create_features(pdb_path, chain="A", secondary_structure=True,
                                   renumber_pdbs=True, add_recycling=True, add_mask=False,
                    stride_path=STRIDE_EXE):
    if pdb_path.endswith(".cif"):
        pdb_path = cif2pdb(pdb_path)
    dist_matrix = get_distance(pdb_path, chain=chain)
    n_res = dist_matrix.shape[-1]
    if not secondary_structure:
        if add_recycling:
            recycle_dimensions = np.zeros([2, n_res, n_res]).astype(np.float32)
            dist_matrix = np.concatenate((dist_matrix, recycle_dimensions), axis=0)
        if add_mask:
            dist_matrix = np.concatenate((dist_matrix, np.zeros((1, n_res, n_res)).astype(np.float32)), axis=0)
        return dist_matrix
    else:
        if renumber_pdbs:
            output_pdb_path = pdb_path.replace('.pdb', '_renum.pdb').replace('.cif', '_renum.cif')
            renum_pdb_file(pdb_path, output_pdb_path)
        else:
            output_pdb_path = pdb_path
        ss_filepath = pdb_path + '_ss.txt'
        calculate_ss(output_pdb_path, chain, stride_path, ssfile=ss_filepath)
        helix, strand = make_ss_matrix(ss_filepath, nres=dist_matrix.shape[-1])
        if renumber_pdbs:
            os.remove(output_pdb_path)
        os.remove(ss_filepath)
    LOG.info(f"Distance matrix shape: {dist_matrix.shape}, SS matrix shape: {helix.shape}")
    stacked_features = np.stack((dist_matrix[0], helix, strand), axis=0)
    if add_recycling:
        recycle_dimensions = np.zeros([2, n_res, n_res]).astype(np.float32)
        stacked_features = np.concatenate((stacked_features, recycle_dimensions), axis=0)
    if add_mask:
        stacked_features = np.concatenate((stacked_features, np.zeros((1, n_res, n_res)).astype(np.float32)), axis=0)
    stacked_features = stacked_features[None] # add batch dimension
    return torch.Tensor(stacked_features)


def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    try:
        diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
        dist = np.sqrt(np.sum(diff_vector * diff_vector))
    except:
        dist = 20.0
    return dist


def calc_dist_matrix(chain) :
    """Returns a matrix of C-alpha distances between two chains"""
    distances = np.zeros((len(chain), len(chain)), 'float')
    for row, residue_one in enumerate(chain):
        for col, residue_two in enumerate(chain):
            distances[row, col] = calc_residue_dist(residue_one, residue_two)
    return distances

def get_distance(structure_path, chain='A'):
    chain_id = os.path.split(structure_path)[-1].split('.')[0]
    if structure_path.endswith('.pdb'):
        structure = Bio.PDB.PDBParser().get_structure(chain_id, structure_path)
    elif structure_path.endswith('.cif'):
        structure = Bio.PDB.MMCIFParser().get_structure(chain_id, structure_path)
    else:
        raise ValueError(f'Unrecognized file extension: {structure_path}')
    model = structure[0]
    if chain is not None:
        residues = [c for c in model[chain].child_list]
    dist_matrix = calc_dist_matrix(residues) # recycling dimensions are added later
    x = np.expand_dims(dist_matrix, axis=0)
    # replace zero values and then invert.
    x[0][x[0] == 0] = x[0][x[0] > 0].min()  # replace zero values in pae / distance
    x[0] = x[0] ** (-1)
    return x

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


def convert_domain_dict_strings(domain_dict):
    """
    Converts the domain dictionary into domain_name string and domain_bounds string
    eg. domain names D1|D2|D1
    eg. domain bounds 0-100|100-200|200-300
    """
    domain_names = []
    domain_bounds = []
    for k,v in domain_dict.items():
        if k=='linker':
            continue
        residues = sorted(v)
        for i, res in enumerate(residues):
            if i==0:
                start = res
            elif residues[i-1] != res - 1:
                domain_bounds.append(f'{start}-{residues[i-1]}')
                domain_names.append(k)
                start = res
            if i == len(residues)-1:
                domain_bounds.append(f'{start}-{res}')
                domain_names.append(k)

    return '|'.join(domain_names), '|'.join(domain_bounds)

def get_predictions_from_pdb(model, pdb_path, secondary_structure=False):
    x = inference_time_create_features(pdb_path, chain="A", secondary_structure=secondary_structure)
    A_hat, domain_dict, uncertainty = model.predict(x)
    names, bounds = convert_domain_dict_strings(domain_dict[0])
    return names, bounds


def predict(model, pdb_path, renumber_pdbs=True) -> List[PredictionResult]:
    """
    Makes the prediction and returns a list of PredictionResult objects
    """
    start = time.time()
    x = inference_time_create_features(pdb_path, chain="A", secondary_structure=True, renumber_pdbs=renumber_pdbs)
    A_hat, domain_dict, uncertainty_array = model.predict(x)
    names_str, bounds_str = convert_domain_dict_strings(domain_dict[0])
    uncertainty = uncertainty_array[0]

    names = names_str.split('|')
    bounds = bounds_str.split('|')

    assert len(names) == len(bounds)

    # return a list of PredictionResult objects
    prediction_results = []
    for domain_id, chopping in zip(names, bounds):
        result = PredictionResult(pdb_path=pdb_path,
                                  domain_id=domain_id,
                                  chopping=chopping,
                                  uncertainty=uncertainty)
        prediction_results.append(result)

    runtime = time.time() - start
    LOG.info(f"Runtime: {round(runtime, 3)}s")
    return prediction_results


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
            'domain_id': res.domain_id,
            'chopping': res.chopping,
            'uncertainty': f'{res.uncertainty:.3g}',
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

    prediction_results = []
    csv_writer = get_csv_writer(args.output)
    csv_writer.writeheader()
    if input_method == 'structure_directory':
        structure_dir = args.structure_directory
        for idx, fname in enumerate(os.listdir(structure_dir)):
            pdb_path = os.path.join(structure_dir, fname)
            _results = predict(model, pdb_path)
            prediction_results.extend(_results)
            write_csv_results(csv_writer, _results)
            if args.pymol_visual:
                write_pymol_script(results=_results, save_dir=outer_save_dir) # todo refactor this so that it naturally calls after each result generated
    elif input_method == 'structure_file':
        prediction_results = predict(model, args.structure_file)
        write_csv_results(csv_writer, prediction_results)
    else:
        raise NotImplementedError('Not implemented yet')




def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=f'{REPO_ROOT}/saved_models/secondary_structure_epoch17/version_2',
                        help='path to model directory must contain model.pt and config.json')
    parser.add_argument('--output', '-o', type=argparse.FileType('w'), default='-',
                        help='write results to this file (default: stdout)')
    parser.add_argument('--uniprot_id', type=str, default=None, help='single uniprot id')
    parser.add_argument('--uniprot_id_list_file', type=str, default=None,
                        help='path to file containing uniprot ids')
    parser.add_argument('--structure_directory', type=str, default=None,
                        help='path to directory containing PDB or MMCIF files')
    parser.add_argument('--structure_file', type=str, default=None,
                        help='path to PDB or MMCIF files')
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
    parser.add_argument('--pymol_visual', dest='pymol_visual', action='store_true', help='whether to generate pymol images')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    # note: any variables created here will be global (bad)
    setup_logging()
    main(parse_args())

