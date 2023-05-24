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
                    stride_path=stride_executable):
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
        ss_filepath = pdb_path.replace('.pdb', '_ss.txt').replace('.cif', '_ss.txt')
        calculate_ss(output_pdb_path, chain, stride_path, ssfile=ss_filepath)
        helix, strand = make_ss_matrix(ss_filepath, nres=dist_matrix.shape[-1])
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

def load_model(args):
    config = common_utils.load_json(os.path.join(args.model_dir, "config.json"))
    config["learner"]["remove_disordered_domain_threshold"] = args.remove_disordered_domain_threshold
    config["learner"]["trim_disordered"] = True
    learner = pairwise_predictor(config["learner"], output_dir=args.model_dir)
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

def predict(model, pdb_path, outer_save_dir):
    start = time.time()
    fname = os.path.split(pdb_path)[-1].split('.')[0]
    save_dir = os.path.join(outer_save_dir, fname)
    os.makedirs(save_dir, exist_ok=True)
    x = inference_time_create_features(pdb_path, chain="A", secondary_structure=True)
    A_hat, domain_dict, uncertainty = model.predict(x)
    names, bounds = convert_domain_dict_strings(domain_dict[0])
    with open(os.path.join(save_dir, f'{fname}.txt'), 'w') as f:
        f.write(f'{names}\n{bounds}')
    if args.pymol_visual:
        generate_pymol_image(
            pdb_path=pdb_path,
            chain='A',
            names=names,
            bounds=bounds,
            image_out_path=os.path.join(save_dir, f'{fname}.png'),
            path_to_script=os.path.join(outer_save_dir, 'image_gen.pml'),
            pymol_executable=pymol_executable
        )
    runtime = time.time() - start
    print(f"Runtime: {round(runtime, 3)}s")

def main(args):
    outer_save_dir = args.save_dir
    input_method = get_input_method(args)
    model = load_model(args)
    os.makedirs(outer_save_dir, exist_ok=True)
    if input_method == 'structure_directory':
        structure_dir = args.structure_directory
        for i, fname in enumerate(os.listdir(structure_dir)):
            pdb_path = os.path.join(structure_dir, fname)
            predict(model, pdb_path, outer_save_dir)
    elif input_method == 'structure_file':
        predict(model, args.structure_file, outer_save_dir)
    else:
        raise NotImplementedError('Not implemented yet')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='saved_models/secondary_structure_epoch17/version_2',
                        help='path to model directory must contain model.pt and config.json')
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
                        help='if the domain is less than this fraction secondary structure, it will be removed')
    parser.add_argument('--pymol_visual', dest='pymol_visual', action='store_true', help='whether to generate pymol images')
    args = parser.parse_args()
    main(args)

