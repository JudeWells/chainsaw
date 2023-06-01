import gzip
import os
import subprocess
import re
from itertools import product
import urllib
import numpy as np
import pandas as pd
import requests
import logging

LOG = logging.getLogger(__name__)


import Bio.PDB
# from Bio.PDB.DSSP import DSSP
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)


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
                LOG.info(f'Found {len(shared_res)} shared residues')
                if resolve_conflics:
                    mapping = resolve_residue_in_multiple_domain(mapping, shared_res)
                else:
                    raise ValueError("SAME RESIDUE NUMBER FOUND IN MULTIPLE DOMAINS")
    return mapping


def adjust_domain_dict_for_non_aligned(dom_dict, non_aligned):
    non_aligned = set(non_aligned)
    # remove the non aligned from the domains
    dom_dict = {k: np.array(list(set(v) - non_aligned)) for k,v in dom_dict.items()}
    # adjust the remaining indexes
    for dname, dres in dom_dict.items():
        updated_residues = np.copy(dres)
        if len(dres) == 0:
            continue
        if max(dres) > min(non_aligned):
            for r in non_aligned:
                updated_residues[dres>r] = updated_residues[dres>r] - 1
            dom_dict[dname] = updated_residues
    return {k:sorted(v.tolist()) for k,v in dom_dict.items()}


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


def check_consistency(res_col, idx_col):
    """
    This function determins if the idx_col is consistent with being a list of indexes for each residue
    """
    tuple_list = list(zip(idx_col, res_col))
    idx_set = set(idx_col)
    if len(idx_set) > len(idx_col) * 0.5: # the true idx set should be significantly shorter than the entire index col because many repeated idx (res) across atoms
        return False
    if len(idx_set) < 9:
        return False
    for idx in idx_set:
        # if there is an index associated with more than one amino acid type
        if len(set([t[1] for t in tuple_list if t[0]==idx]))>1:
            return False
    return True


def remove_alphabet_chars(dom_limit_list):
    return [re.sub('[A-z]', '', lim) for lim in dom_limit_list]


def remove_inconsistent(res_seq, res_name):
    idx_set = set(res_seq)
    tuple_list = list(zip(res_seq, res_name))
    for idx in idx_set:
        # if there is an index associated with more than one amino acid type
        if len(set([t[1] for t in tuple_list if t[0]==idx]))>1:
            res_name = [n for i, n in enumerate(res_name) if res_seq[i] != idx]
            res_seq = [i for i in res_seq if i != idx]
    return res_seq, res_name


def make_labels_one_row(row, output_dir, non_aligned_residues=[]):
    if 'chain_id' in row:
        chain_id = row.chain_id
    else:
        chain_id = row.pdb_id + row.pdb_chain
    try:
        one_map = make_domain_mapping_dict(row)
        make_pair_labels(len(row.pdb_seq), one_map, chain_id, output_dir, non_aligned_residues=non_aligned_residues)

    except Exception as e:
        LOG.error('Failed to map domains for', chain_id)
        LOG.error(e)


def sort_domain_limits(limits, dom_names):
    start_positions = [x[0] for x in limits]
    end_positions = [x[1] for x in limits]
    sorted_index = np.argsort(start_positions)
    assert (sorted_index == np.argsort(end_positions)).all()
    return np.array(limits)[sorted_index], list(np.array(dom_names)[sorted_index])


def atom_identifier(lines, chain=None):
    lines = [l.strip('b\'') for l in lines]
    lines = [l for l in lines if l[0:4] == 'ATOM']
    if chain is not None:
        lines = [l for l in lines if l[21]==chain]
    res_seq = [l[22:26].strip() for l in lines]
    res_name = [l[17:20].strip() for l in lines]
    res_seq = remove_alphabet_chars(res_seq)
    res_seq = [int(i) for i in res_seq]
    if not check_consistency(res_name, res_seq):
        res_seq, res_name = remove_inconsistent(res_seq, res_name)
        assert check_consistency(res_name, res_seq)
    return dict(zip(res_seq, res_name))


def fetch_pdb_file(pdbid, pdb_dir):
    url = 'http://files.rcsb.org/download/' + pdbid.upper() + '.pdb'
    savepath = os.path.join(pdb_dir, pdbid.lower() + '.pdb')
    urllib.request.urlretrieve(url, savepath)


def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    try:
        diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
        dist = np.sqrt(np.sum(diff_vector * diff_vector))
    except:
        dist = 20.0
    return dist


def download_pdb_file(pdb_id, pdb_dir):
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    result = requests.get(pdb_url).text
    with open(os.path.join(pdb_dir, f"{pdb_id}.pdb"), 'w') as filehandle:
        filehandle.writelines(result)


def fetch_alphafold_pdb(up_id, pdb_dir, fragment=1):
    urls = [f"https://alphafold.ebi.ac.uk/files/AF-{up_id}-F{fragment}-model_v4.pdb",
            f"https://alphafold.ebi.ac.uk/files/AF-{up_id}-F{fragment}-model_v3.pdb"
            ]
    for url in urls:
        try:
            result = requests.get(url).text
            write_path = os.path.join(pdb_dir, f"{up_id}.pdb")
            with open(write_path, 'w') as filehandle:
                filehandle.writelines(result)
            return 'success'
        except:
            return 'no af struct'

def make_distance_matrix_from_pdb_file(pdb_path, chain=None, chain_id="", return_plddt=False):
    structure = Bio.PDB.PDBParser().get_structure(chain_id, pdb_path)
    model = structure[0]
    if chain is None:
        chain = model.child_list[0].id
    one_chain_res = [c for c in model[chain].child_list if c.resname in valid_residues]
    dist_matrix = calc_dist_matrix(one_chain_res, return_plddt=return_plddt)
    return dist_matrix

def get_up_distance_and_plddt(row, id_col='uniprot_id', downloaded_pdbs=[],
                              pdb_dir='../../data/af_pdbs', chain=None, return_plddt=False):
    if 'uniprot_id' in id_col:
        chain = 'A'
    elif chain is None:
        chain = row.pdb_chain
    up_id = row[id_col]
    if up_id not in downloaded_pdbs:
        fetch_status = fetch_alphafold_pdb(up_id, pdb_dir)
    pdb_path = os.path.join(pdb_dir, up_id + '.pdb')
    return make_distance_matrix_from_pdb_file(pdb_path, chain, chain_id="up_id", return_plddt=return_plddt)



def bounds_string_remove_gaps(bounds, remove_gaps):
    new_bounds = []
    bounds = bounds.split("|")
    for i, b in enumerate(bounds):
        if i!=0 and i-1 in remove_gaps:
            continue
        if i in remove_gaps:
            end_seg_ix = i + 1
            while end_seg_ix in remove_gaps:
                end_seg_ix += 1
            new_start = b.split('-')[0]
            new_end = bounds[end_seg_ix].split('-')[1]
            new_bounds.append(f"{new_start}-{new_end}")
        else:
            new_bounds.append(b)
    return '|'.join(new_bounds)


def name_string_remove_gaps(names, remove_gaps):
    return "|".join([d for j, d in enumerate(names.split("|")) if j not in remove_gaps])


def fetch_missing_pdb_res_ix(pdbid, chain, pdb_dir = '../../pdbs'):
    """
    JW recently changed this function to return the missing positional indices
    rather than the missing auth indices.
    """
    if pdbid.lower() + '.pdb' not in os.listdir(pdb_dir):
        fetch_pdb_file(pdbid, pdb_dir)
    if pdbid.lower() + '.pdb' not in os.listdir(pdb_dir):
        LOG.warning('Unable to fetch file for', pdbid)
        return None
    with open(os.path.join(pdb_dir, pdbid + '.pdb'), 'r') as filehandle:
        lines = filehandle.readlines()
    atom_line_dict = atom_identifier(lines, chain=chain)
    missing_auth = set(range(min(atom_line_dict), max(atom_line_dict) + 1)) - set(atom_line_dict.keys())
    return np.array(list(missing_auth)) - min(atom_line_dict)


def fill_domain_gaps(df, min_linker=20, pdb_dir='../../pdbs'):
    """
    This version of the function gets the missing residues from the pdb file if not available
    """
    bounds_columns = ['dom_bounds_pdb_auth', 'dom_bounds_pdb_ix']
    bounds_columns = [c for c in bounds_columns if c in df.columns]
    names_columns = ['dom_names', 'dom_chain', 'cath_code']
    names_columns = [c for c in names_columns if c in df.columns]
    for i, row in df.iterrows():
        try:
            names = row.dom_names.split('|')
            if 'missing' in row and isinstance(row.missing, str) and len(row.missing) > 0:
                missing = set([int(m) for m in row.missing.split('-')])
            elif isinstance(row.pdb_id, str) and isinstance(row.pdb_chain, str):
                try:
                    missing = fetch_missing_pdb_res_ix(row.pdb_id, row.pdb_chain, pdb_dir=pdb_dir)
                    df.loc[i, 'pdb_missing'] = '-'.join([str(m) for m in missing])
                except:
                    pass
            else:
                missing = set()
            if len(names) > len(set(names)):
                remove_gaps = []
                bounds = row.dom_bounds_pdb_ix.split('|')
                name_bound = [(b,n) for b,n in zip(bounds, names)]
                name_bound = sorted(name_bound, key=lambda x: int(x[0].split('-')[0]))
                bounds = [tup[0] for tup in name_bound]
                names = [tup[1] for tup in name_bound]
                for j, dname, b in zip(range(len(names)), names, bounds):
                    if j+1 < len(names):
                        if names[j+1] == dname: # if the same domain has a break without another domain in between
                            gap_start = int(b.split('-')[1])
                            gap_end = int(bounds[j+1].split('-')[0])
                            if set(range(gap_start+1, gap_end)).issubset(missing):
                                remove_gaps.append(j)
                            elif gap_end - gap_start < min_linker+1:
                                LOG.info(f'Removing gap not caused by missing {row[df.columns[0]]}')
                                remove_gaps.append(j)
                            elif len(set(range(gap_start+1, gap_end)) - missing) < min_linker:
                                LOG.info(f'Removing gap which is partially missing {row[df.columns[0]]}')
                                remove_gaps.append(j)
                if len(remove_gaps):
                    for n_col in names_columns:
                        if n_col not in row or row.isnull()[n_col]:
                            continue
                        new_names = name_string_remove_gaps(df.loc[i, n_col], remove_gaps)
                        LOG.info(f'Names change: {df.loc[i, n_col]} -> {new_names}')
                        df.loc[i, n_col] = new_names
                    for b_col in bounds_columns:
                        if b_col not in row or row.isnull()[b_col]:
                            continue
                        new_bounds = bounds_string_remove_gaps(df.loc[i, b_col], remove_gaps)
                        LOG.info(f'Bounds change: {df.loc[i, b_col]} -> {new_bounds}')
                        df.loc[i, b_col] = new_bounds
                LOG.info()
        except Exception as e:
            LOG.error(f'Exception {e}')
            pass
    return df


def add_match_status_column(pae_matrix, match_status):
    new_matrix = np.zeros([len(pae_matrix), len(pae_matrix) + 1])
    new_matrix[:,1:] = pae_matrix
    new_matrix[:,0] = match_status
    return new_matrix


def get_pae(uniprot_id):
    urls =[
        f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-predicted_aligned_error_v4.json",
        f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-predicted_aligned_error_v3.json",
        f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-predicted_aligned_error_v2.json",
        # f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-predicted_aligned_error_v1.json",
    ]
    for url in urls:
        try:
            r = requests.get(url)
            pae_graph = r.json()[0]
            return pae_graph
        except:
            pass


def execute_shell_cmd(cmd_str):
    process = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    return stdout, stderr


def run_pdb_renum(pdbid, save_dir):
    working_directory = os.getcwd()
    os.chdir(save_dir)
    command = f"python3 /Users/judewells/Documents/dataScienceProgramming/PDBrenum/PDBrenum.py -rfla {pdbid} -PDB --set_to_off_mode_gzip"
    stdout, stderr = execute_shell_cmd(command)
    os.chdir(working_directory)
    return stderr


def load_lines_pdb(pdb_path, chain):
    with gzip.open(pdb_path, 'rt') as filehandle:
        pdb_lines = filehandle.readlines()
    atom_lines = [l for l in pdb_lines if l[:4] == 'ATOM' and l[21]==chain]
    return atom_lines


def merge_lines_create_dict(old_lines, new_lines):
    pdb_resnums = [l[22:26].strip() for l in old_lines]
    unp_resnums = [int(l[22:26].strip()) for l in new_lines]
    return dict(zip(pdb_resnums, unp_resnums))


def create_mapping(row, sequence_col='pdb_seq', pdb_seg='pdb_ix_covered_by_up', up_seg='unp_covered_by_pdb'):
    """
    A small percentage (around 0.5%) of SIFTs mappings have segments of uniprot sequences which are
    longer than the corresponding PDB segment. This is indicative of insertions in the aligned
    Uniprot sequence (or equivalently deletions in the aligned PDB sequence). This function
    simply takes the first part of the uniprot alignment and discards the end.
    """
    mapping = {}
    up_segs = row[up_seg].split('|')
    for i, segment in enumerate(row[pdb_seg].split('|')):
        pdb_start, pdb_end = [int(b) for b in segment.split('-')]
        pdb_start -= 1
        up_start, up_end = [int(b) for b in up_segs[i].split('-')]
        up_start -= 1
        mapping.update(dict(zip(range(pdb_start, pdb_end), range(up_start, up_end))))
    missing_pdb_residues = set(range(len(row[sequence_col]))) - set(mapping.keys())
    n_missing = len(missing_pdb_residues)
    if n_missing:
        missing_map = dict(zip(missing_pdb_residues, [MISSING_UP_IX]*n_missing))
        mapping.update(missing_map)
    mapping = dict(sorted(mapping.items()))
    return mapping


def identify_misaligned_pdb(mapping_dict, pdb_seq=None):
    """
    misalignment scenarios:
    1) residues are aligned but they are different amino acids (this is not really considered a misalignment)
    2) query to gap (this shows up in mapping dict as {i:-1} where -1 is the indicator for missing
    3) gap to target (this shows up as non-consecutive values in the mapping dict eg. {1:1, 2:4} (residues 2&3 in target are aligned to a gap in the seq)
    4) misalignment caused because residues are down-chain from a misalignment of type 2 or 3
    """
    included_unp = [v for v in mapping_dict.values() if v != MISSING_UP_IX]
    # dealing with misalignment scenario 3 (uniprot has insertion compared to pdb):
    unp_gaps = set(range(min(included_unp), max(included_unp))) - set(included_unp)
    if len(unp_gaps):
        non_continuous_mappings = set([i for i,v in enumerate(mapping_dict.values()) if v >= min(unp_gaps)])
    else:
        non_continuous_mappings = set()
    # dealing with misalignment scenario 2 (pdb has insertion compared to uniprot):
    non_aligned_pdb_indicies = [i for i, v in enumerate(mapping_dict.values()) if v == MISSING_UP_IX]
    first_aligned_pdb_index = min([i for i,v in enumerate(mapping_dict.values()) if v != MISSING_UP_IX])
    non_aligned_after_start = set([i for i in non_aligned_pdb_indicies if i > first_aligned_pdb_index])
    # dealing with misalignment scenario 4 (all residues downchain from a misalignment):
    non_aligned_combined = non_aligned_after_start.union(non_continuous_mappings)
    if len(non_aligned_combined):
        first_break_in_alignment = min(non_aligned_combined)
        secondary_misaligned = set(range(first_break_in_alignment, len(mapping_dict)))
    else:
        secondary_misaligned = set()
    return set(non_aligned_pdb_indicies), secondary_misaligned

def reshape_pae_matrix(pae, mapping, trim_non_aligned=False):
    n_residues = len(mapping.keys())
    pae_index = list(mapping.values())
    if 'residue1' in pae: #V2 API data
        pae['residue1'][-1] = MISSING_UP_IX
        pae['residue2'][-1] = MISSING_UP_IX
        res1_select = [pae['residue1'][i] in pae_index for i in range(len(pae['residue1']))]
        res2_select = [pae['residue2'][i] in pae_index for i in range(len(pae['residue2']))]
        selection_index = np.where(np.logical_and(res1_select, res2_select))
        distances = np.array(pae['distance'])[selection_index]
        pae_array = distances.reshape([n_residues, n_residues])
    elif 'predicted_aligned_error' in pae: # V3 API data
        pae_array = np.array(pae['predicted_aligned_error'])
        # add a column and row of default_missing_val values which are selected wherever there is a missing index
        pae_array = np.vstack([pae_array, np.full(pae_array.shape[1], MISSING_UP_IX)])
        pae_array = np.hstack([pae_array, np.full((pae_array.shape[0], 1), MISSING_UP_IX)])
        index_tuples = list(product(pae_index, pae_index))
        select_x = [e[0] for e in index_tuples]
        select_y = [e[1] for e in index_tuples]
        pae_array = pae_array[select_x, select_y].reshape([n_residues, n_residues])

    else:
        raise Exception('Unknown PAE format ')
    if trim_non_aligned:
        pae_array = trim_non_aligned_residues(pae_array, mapping)
    return pae_array

def reshape_distance_matrix(distance, mapping, trim_non_aligned=False):
    n_residues = len(mapping.keys())
    pae_index = list(mapping.values())
    distance = np.vstack([distance, np.full(distance.shape[1], MISSING_UP_IX)])
    distance = np.hstack([distance, np.full((distance.shape[0], 1), MISSING_UP_IX)])
    index_tuples = list(product(pae_index, pae_index))
    select_x = [e[0] for e in index_tuples]
    select_y = [e[1] for e in index_tuples]
    distance = distance[select_x, select_y].reshape([n_residues, n_residues])
    if trim_non_aligned:
        distance = trim_non_aligned_residues(distance, mapping)
    return distance

def trim_non_aligned_residues(array, mapping):
    aligned_residues = [k for k, v in mapping.items() if v != MISSING_UP_IX]
    if len(array.shape) == 2:
        return array[aligned_residues, :][:, aligned_residues]
    else:
        return array[aligned_residues]

def create_misalignment_flag_matrices(mapping):
    misaligned, secondary_misaligned = identify_misaligned_pdb(mapping)
    n_residues = len(mapping.keys())
    direct_miss = np.zeros([n_residues, n_residues])
    secondary_miss = np.zeros([n_residues, n_residues])
    for res_index in misaligned:
        direct_miss[res_index, :] = 1
        direct_miss[:, res_index] = 1
    for res_index in secondary_misaligned:
        secondary_miss[res_index, :] = 1
        secondary_miss[:, res_index] = 1
    return direct_miss, secondary_miss

def save_matrices(matrix_list, pdbid, save_dir):
    combined = np.squeeze(np.stack([matrix_list]))
    save_path = os.path.join(save_dir, pdbid)
    np.savez_compressed(save_path, combined)
    LOG.info('Success', pdbid)


def reshape_plddt(plddt, mapping, trim_non_aligned=False):
    plddt = np.append(plddt, MISSING_UP_IX)
    plddt = plddt[list(mapping.values())]
    if trim_non_aligned:
        plddt = trim_non_aligned_residues(plddt, mapping)
    return plddt

def reformat_test(df):
    for i, row in df.iterrows():
        if not isinstance(row.casp_sequence, str):
            if isinstance(row.pdb_fasta_seq, str):
                df.loc[i,'casp_sequence'] = row.pdb_fasta_seq
                df.loc[i, 'missing_casp_seq_replaced_w_pdb_fasta'] = 1
    if 'chain_id' not in df.columns and 'casp_id' in df.columns:
        df['chain_id'] = df.casp_id
    drop_cols = [
        'casp_n_res',
        'n_domains',
        'pdb_casp_seq_match',
        'aligned_casp_sequence',
        'aligned_uniprot_sequence',
        'blast_uniprot_id',
        'blast_identity',
        'blast_coverage',
        'sifts_pdb_cover_on_up_auth',
        'sifts_dom_names',
        'sifts_dom_bounds_pdb_auth',
        'sifts_dom_bounds_pdb_ix',
        'sifts_dom_chain',
        'sifts_cath_code',
    ]
    drop_cols = [c for c in df.columns if c in drop_cols]
    df = df.drop(drop_cols, axis=1)

    df = df.rename(columns={
            'casp_sequence': 'pdb_seq',
            'sifts_up_id': 'uniprot_id'
    })
    rename_dict = {c: c.replace('sifts_', '') for c in df.columns}
    df = df.rename(columns=rename_dict)
    return df

def get_unique_completed_up(df, completed):
    completed_df = df[df.chain_id.isin(completed)]
    unique_up = set(completed_df.uniprot_id + completed_df.pdb_ix_covered_by_up)
    return unique_up

def fill_missing_plddts(df_orig, plddt_dir):
    df = df_orig.copy()
    for i, row in df.iterrows():
        if row.notnull().plddt:
            continue
        try:
            plddt = np.load(os.path.join(plddt_dir, f"{row.chain_id}.npz"))['arr_0']
        except FileNotFoundError:
            continue
        df.loc[i, 'plddt'] = plddt.mean()
    return df


if __name__=='__main__':
    # df_path = '../../facebook_domains_filtered_fill_gaps_count_missing.csv'
    # df_path = '../../casp_test_sifts_remove_gaps_score_alignments_v3.csv'

    df_path = '../../all_multidomain.csv'
    df = pd.read_csv(df_path)
    if 'test' in df_path:
        df = reformat_test(df)
    save_dir = '../../features/2d_features_casp14/'
    plddt_dir = '../../features/plddt_casp14'
    pairwise_labels_dir = '../../features/pairwise_casp14'
    pdb_dir = '../../pdbs'
    for directory_path in [save_dir, plddt_dir, pdb_dir, pairwise_labels_dir]:
        os.makedirs(directory_path, exist_ok=True)
    df = fill_domain_gaps(df, min_linker=20, pdb_dir=pdb_dir)
    df.to_csv(df_path, index=False)
    error_file = 'make_feature_errors.txt'
    if 'pdb_chain' not in df.columns and 'up_to_pdb_chain' in df.columns:
        df['pdb_chain'] = df.up_to_pdb_chain.apply(lambda x: x.split('|')[0])
    if 'uniprot_id' not in df.columns and 'up_id' in df.columns:
        df = df.rename(columns={'up_id': 'uniprot_id'})
    if os.path.exists(error_file):
        with open(error_file, 'r') as f:
            error_ids = f.readlines()
        error_ids = [id.split()[0] for id in error_ids]
    else:
        error_ids = []

    downloaded_pdbs = set(f.split('.')[0] for f in os.listdir(pdb_dir))
    completed = [f.split('.')[0] for f in os.listdir(save_dir)]
    completed_up = get_unique_completed_up(df, completed)
    for i, row in df.iterrows():
        pdbid = row.pdb_id
        pdb_seq = row.pdb_seq
        unique_up = row.uniprot_id + row.pdb_ix_covered_by_up
        try:
            if 'test' in df_path:
                chain_id = row.casp_id
            else:
                chain_id = pdbid.lower() + row.pdb_chain.upper()
            # if id in completed or id in error_ids:
            if chain_id in completed or chain_id in error_ids or unique_up in completed_up:
                continue
            up_id = row.uniprot_id
            pae_dict = get_pae(uniprot_id=up_id)
            if not isinstance(pae_dict, dict):
                raise Exception(f"No PAE dictionary returned {up_id}")
            # LOG.info(row.unp_covered_by_pdb, row.pdb_ix_covered_by_up)
            mapping_dict = create_mapping(row)
            non_aligned = [k for k,v in mapping_dict.items() if v==MISSING_UP_IX]
            make_labels_one_row(row, pairwise_labels_dir, non_aligned_residues=non_aligned)
            pae_matrix = reshape_pae_matrix(pae_dict, mapping_dict, trim_non_aligned=True)
            # direct_miss, secondary_miss = create_misalignment_flag_matrices(mapping_dict)
            dist_matrix, plddt = get_up_distance_and_plddt(row, downloaded_pdbs=downloaded_pdbs)

            dist_matrix = reshape_distance_matrix(dist_matrix, mapping_dict, trim_non_aligned=True)
            plddt = reshape_plddt(plddt, mapping_dict, trim_non_aligned=True)
            if not (len(row.pdb_seq) - len(non_aligned) == len(dist_matrix)==len(pae_matrix)==len(plddt)):
                LOG.warning(
                    f"inconsistent residue lengths {row.uniprot_id}, matrix: {len(dist_matrix)}, df: {len(row.pdb_seq)}")
                continue
            df.loc[i, 'plddt'] = plddt.mean()
            np.savez_compressed(os.path.join(plddt_dir, chain_id), plddt)
            save_matrices([pae_matrix, dist_matrix], chain_id, save_dir)

        except Exception as e:
            LOG.error(f'Failed: {chain_id}, exception: {e}')
            try:
                with open(error_file, 'a') as filehandle:
                    filehandle.write(f"{chain_id} {e}\n")
            except:
                pass
    df.to_csv(df_path, index=False)
