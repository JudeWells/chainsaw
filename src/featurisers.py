import os
import numpy as np
import Bio.PDB
import torch
from src import constants
from src.utils.cif2pdb import cif2pdb
from src.utils.secondary_structure import renum_pdb_file,\
    calculate_ss, make_ss_matrix

import logging


LOG = logging.getLogger(__name__)


def inference_time_create_features(pdb_path, chain="A", secondary_structure=True,
                                   renumber_pdbs=True, add_recycling=True, add_mask=False,
                                   stride_path=constants.STRIDE_EXE,
                                   *,
                                   model_structure: Bio.PDB.Structure=None,
                                   ):
    if pdb_path.endswith(".cif"):
        pdb_path = cif2pdb(pdb_path)
    
    # HACK: allow `model_structure` to be created elsewhere (to avoid reparsing)
    # Ideally this would always happen further upstream and we wouldn't
    # need to pass in `pdb_path`, however `pdb_path` is used to generate
    # additional files and I don't want to mess around with that logic.
    # -- Ian
    if not model_structure:
        model_structure = get_model_structure(pdb_path, chain=chain)
    
    dist_matrix = get_distance(model_structure)
    
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
    # TODO: vectorise the entire thing
    distances = np.zeros((len(chain), len(chain)), 'float')
    for row, residue_one in enumerate(chain):
        for col, residue_two in enumerate(chain):
            distances[row, col] = calc_residue_dist(residue_one, residue_two)
    return distances


def get_distance(structure_model: Bio.PDB.Structure, chain='A'):
    if chain is not None:
        residues = [c for c in structure_model[chain].child_list]
    dist_matrix = calc_dist_matrix(residues) # recycling dimensions are added later
    x = np.expand_dims(dist_matrix, axis=0)
    # replace zero values and then invert.
    x[0][x[0] == 0] = x[0][x[0] > 0].min()  # replace zero values in pae / distance
    x[0] = x[0] ** (-1)
    return x
