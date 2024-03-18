import logging
import os
from typing import List

import Bio.PDB
import numpy as np
import torch
from scipy.spatial import distance_matrix

from src import constants
from src.constants import _3to1
from src.utils.cif2pdb import cif2pdb
from src.utils.secondary_structure import renum_pdb_file, \
    calculate_ss, make_ss_matrix

LOG = logging.getLogger(__name__)


def get_model_structure(structure_path) -> Bio.PDB.Structure:
    """
    Returns the Bio.PDB.Structure object for a given PDB or MMCIF file
    """
    structure_id = os.path.split(structure_path)[-1].split('.')[0]
    if structure_path.endswith('.pdb'):
        structure = Bio.PDB.PDBParser().get_structure(structure_id, structure_path)
    elif structure_path.endswith('.cif'):
        structure = Bio.PDB.MMCIFParser().get_structure(structure_id, structure_path)
    else:
        raise ValueError(f'Unrecognized file extension: {structure_path}')
    model = structure[0]
    return model


class Residue:
    def __init__(self, index: int, res_label: str, aa: str):
        self.index = int(index)
        self.res_label = str(res_label)
        self.aa = str(aa)

def get_model_structure_residues(structure_model: Bio.PDB.Structure, chain='A') -> List[Residue]:
    """
    Returns a list of residues from a given PDB or MMCIF structure
    """
    residues = []
    res_index = 1
    for biores in structure_model[chain].child_list:
        res_num = biores.id[1]
        res_ins = biores.id[2]
        res_label = str(res_num)
        if res_ins != ' ':
            res_label += str(res_ins)
        
        aa3 = biores.get_resname()
        if aa3 not in _3to1:
            continue

        aa = _3to1[aa3]
        res = Residue(res_index, res_label, aa)
        residues.append(res)
        
        # increment the residue index after we have filtered out non-standard amino acids
        res_index += 1
    
    return residues


def inference_time_create_features(pdb_path, feature_config, chain="A", *,
                                   model_structure: Bio.PDB.Structure=None,
                                   renumber_pdbs=True, stride_path=constants.STRIDE_EXE,
                                   ):
    if pdb_path.endswith(".cif"):
        pdb_path = cif2pdb(pdb_path)

    if not model_structure:
        model_structure = get_model_structure(pdb_path)

    dist_matrix = get_distance(model_structure, chain=chain)

    n_res = dist_matrix.shape[-1]

    if renumber_pdbs:
        output_pdb_path = pdb_path.replace('.pdb', '_renum.pdb').replace('.cif', '_renum.cif')
        renum_pdb_file(pdb_path, output_pdb_path)
    else:
        output_pdb_path = pdb_path
    ss_filepath = pdb_path + '_ss.txt'
    calculate_ss(output_pdb_path, chain, stride_path, ssfile=ss_filepath)
    helix, strand = make_ss_matrix(ss_filepath, nres=dist_matrix.shape[-1])
    if feature_config['ss_bounds']:
        end_res_val = -1 if feature_config['negative_ss_end'] else 1
        helix_boundaries = make_boundary_matrix(helix, end_res_val=end_res_val)
        strand_boundaries = make_boundary_matrix(strand, end_res_val=end_res_val)
    if renumber_pdbs:
        os.remove(output_pdb_path)
    os.remove(ss_filepath)
    LOG.info(f"Distance matrix shape: {dist_matrix.shape}, SS matrix shape: {helix.shape}")
    if feature_config['ss_bounds']:
        if feature_config['same_channel_boundaries_and_ss']:
            helix_boundaries[helix == 1] = 1
            strand_boundaries[strand == 1] = 1
            stacked_features = np.stack((dist_matrix, helix_boundaries, strand_boundaries), axis=0)
        else:
            stacked_features = np.stack((dist_matrix, helix, strand, helix_boundaries, strand_boundaries), axis=0)
    else:
        stacked_features = np.stack((dist_matrix, helix, strand), axis=0)
    stacked_features = stacked_features[None] # add batch dimension
    return torch.Tensor(stacked_features)



def get_distance(structure_model: Bio.PDB.Structure, chain='A'):
    alpha_coords = np.array([residue['CA'].get_coord() for residue in \
                             structure_model[chain].get_residues() if Bio.PDB.is_aa(residue) and \
                             'CA' in residue and residue.get_resname() in _3to1])
    x = distance_matrix(alpha_coords, alpha_coords)
    return x


def make_boundary_matrix(ss, end_res_val=1):
    """
    makes a matrix where  the boundary residues
    of the sec struct component are 1
    """
    ss_lines = np.zeros_like(ss)
    diag = np.diag(ss)
    if max(diag) == 0:
        return ss_lines
    padded_diag = np.zeros(len(diag) + 2)
    padded_diag[1:-1] = diag
    diff_before = diag - padded_diag[:-2]
    diff_after = diag - padded_diag[2:]
    start_res = np.where(diff_before == 1)[0]
    end_res = np.where(diff_after == 1)[0]
    ss_lines[start_res, :] = 1
    ss_lines[:, start_res] = 1
    ss_lines[end_res, :] = end_res_val
    ss_lines[:, end_res] = end_res_val
    return ss_lines
