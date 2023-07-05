import logging
import os

import Bio.PDB
import numpy as np
import torch
from scipy.spatial import distance_matrix

from src import constants
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


def get_model_structure_sequence(structure_model: Bio.PDB.Structure, chain='A') -> str:
    """Get sequence of specified chain from parsed PDB/CIF file."""
    residues = [c for c in structure_model[chain].child_list]
    _3to1 = Bio.PDB.Polypeptide.protein_letters_3to1
    sequence = ''.join([_3to1[r.get_resname()] for r in residues])
    return sequence


def inference_time_create_features(pdb_path, chain="A", secondary_structure=True,
                                   renumber_pdbs=True, add_recycling=True, add_mask=False,
                                   stride_path=constants.STRIDE_EXE, ss_mod=False,
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
        if ss_mod:
            helix_boundaries = make_boundary_matrix(helix)
            strand_boundaries = make_boundary_matrix(strand)
        if renumber_pdbs:
            os.remove(output_pdb_path)
        os.remove(ss_filepath)
    LOG.info(f"Distance matrix shape: {dist_matrix.shape}, SS matrix shape: {helix.shape}")
    if ss_mod:
        stacked_features = np.stack((dist_matrix[0], helix, strand, helix_boundaries, strand_boundaries), axis=0)
    else:
        stacked_features = np.stack((dist_matrix[0], helix, strand), axis=0)
    if add_recycling:
        recycle_dimensions = np.zeros([2, n_res, n_res]).astype(np.float32)
        stacked_features = np.concatenate((stacked_features, recycle_dimensions), axis=0)
    if add_mask:
        stacked_features = np.concatenate((stacked_features, np.zeros((1, n_res, n_res)).astype(np.float32)), axis=0)
    stacked_features = stacked_features[None] # add batch dimension
    return torch.Tensor(stacked_features)

def get_distance(structure_model: Bio.PDB.Structure, chain='A'):
    alpha_coords = np.array([residue['CA'].get_coord() for residue in \
                             structure_model[chain].get_residues()])
    x = distance_matrix(alpha_coords, alpha_coords)
    x[x == 0] = x[x > 0].min()  # replace zero values in pae / distance
    x = x ** (-1)
    x = np.expand_dims(x, axis=0) # todo is batch dimension needed?
    return x


def make_boundary_matrix(ss):
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
    ss_lines[end_res, :] = 1
    ss_lines[:, end_res] = 1
    return ss_lines
