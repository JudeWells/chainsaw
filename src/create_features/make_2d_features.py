import logging
import os
from itertools import product

import numpy as np

import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)

LOG = logging.getLogger(__name__)

"""
This script creates the 2d features: PAE, Distance matrix and alignment flags and also the 2d pairwise domain labels

Alphafold PAEs are extracted from the API
Given that alphafold PAEs are for uniprot sequences
it is necessary to find the corresponding uniprot sequence for a given PDB structure
The sequences of the PDB is then aligned with the uniprot sequence
and the trimmed and aligned PAE is then saved as a matrix which is (n_residues x n_residues) 

Same applies to the pairwise distance matrices which are calculated from AlphaFold predicted structure pdb files
  
"""

valid_residues = ['LEU', 'GLU', 'ARG', 'VAL', 'LYS', 'ILE', 'ASP', 'PHE', 'ALA', 'TYR', 'THR', 'SER', 'GLN', 'PRO',
                  'ASN', 'GLY', 'HIS', 'MET', 'TRP', 'CYS']

name2letter = dict(zip(valid_residues, 'LERVKIDFAYTSQPNGHMWC'))

MISSING_UP_IX = -1

def check_non_assigned_positions_are_non_domain(mapping, labels):
    unaccounted_positions = set(range(len(labels)))
    for dom, res_pos in mapping.items():
        unaccounted_positions = unaccounted_positions - set(res_pos)
    assert "D" not in labels[list(unaccounted_positions)]


def make_pair_labels(n_res, domain_dict, id_string=None, save_dir=None, non_aligned_residues=[]):
    """n_res: number of residues in the non-trimmed sequence

        non_aligned_residues: these will be used to trim down from n_res

        domain_dict: eg. {'D1': [0,1,2,3], 'D2': [4,5,6]}
    """
    pair_labels = np.zeros([n_res, n_res])
    for domain, res_ix in domain_dict.items():
        if domain == 'linker':
            continue
        coords_tuples = list(product(res_ix, res_ix))
        x_ix = [i[0] for i in coords_tuples]
        y_ix = [i[1] for i in coords_tuples]
        pair_labels[x_ix, y_ix] = 1
    if len(non_aligned_residues):
        aligned_residues = [i for i in range(n_res) if i not in non_aligned_residues]
        pair_labels = pair_labels[aligned_residues,:][:,aligned_residues]
    if save_dir is not None:
        save_path = os.path.join(save_dir, id_string)
        np.savez_compressed(save_path, pair_labels)

    return pair_labels

def sort_domain_limits(limits, dom_names):
    start_positions = [x[0] for x in limits]
    end_positions = [x[1] for x in limits]
    sorted_index = np.argsort(start_positions)
    assert (sorted_index == np.argsort(end_positions)).all()
    return np.array(limits)[sorted_index], list(np.array(dom_names)[sorted_index])

def resolve_residue_in_multiple_domain(mapping, shared_res):
    """
    This is a stupid slow recursive solution: but I think it only applies to one
    case so going to leave it for now
    """
    for one_shared in shared_res:
        for domain, res in mapping.items():
            if one_shared in res:
                mapping[domain].remove(one_shared)
                return check_no_residue_in_multiple_domains(mapping)


def check_no_residue_in_multiple_domains(mapping, resolve_conflics=True):
    # ensures no residue index is associated with more than one domain
    for dom, res in mapping.items():
        for dom2, res2 in mapping.items():
            if dom == dom2:
                continue
            shared_res = set(res).intersection(set(res2))
            if len(shared_res):
                print(f'Found {len(shared_res)} shared residues')
                if resolve_conflics:
                    mapping = resolve_residue_in_multiple_domain(mapping, shared_res)
                else:
                    raise ValueError("SAME RESIDUE NUMBER FOUND IN MULTIPLE DOMAINS")
    return mapping

def make_domain_mapping_dict(row):
    dom_limit_list = row.dom_bounds_pdb_ix.split('|')
    dom_names = row.dom_names.split('|')
    dom_limit_list = convert_limits_to_numbers(dom_limit_list)
    dom_limit_array, dom_names = sort_domain_limits(dom_limit_list, dom_names)
    mapping = {}

    for i, d_lims in enumerate(dom_limit_array):
        dom_name = dom_names[i]
        pdb_start, pdb_end = d_lims
        if dom_name not in mapping:
            mapping[dom_name] = []
        mapping[dom_name] += list(range(pdb_start, pdb_end))
    check_no_residue_in_multiple_domains(mapping)
    return mapping

def convert_limits_to_numbers(dom_limit_list):
    processed_dom_limit_list = []
    for lim in dom_limit_list:
        dash_idx = [i for i, char in enumerate(lim) if char == '-']
        if len(dash_idx) == 1:
            start_index = int(lim.split('-')[0]) -1
            end_index = int(lim.split('-')[1])
        else:
            raise ValueError('Invalid format for domain limits', str(dom_limit_list))
        processed_dom_limit_list.append((start_index, end_index))
    return processed_dom_limit_list






