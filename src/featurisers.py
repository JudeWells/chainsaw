import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import Bio.PDB
import numpy as np
import torch

from src import constants
from src.constants import _3to1
from src.utils.cif2pdb import cif2pdb
from src.utils.secondary_structure import renum_pdb_file, \
    calculate_ss, make_ss_matrix

LOG = logging.getLogger(__name__)
_3TO1_BYTES = {k.encode(): v for k, v in _3to1.items()}


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


@dataclass
class SimpleChain:
    """Residues and C-alpha coordinates extracted by `read_simple_pdb`."""

    chain_id: str
    residues: List[Residue]
    ca_coords: np.ndarray  # float32, shape (n_res, 3)


def read_simple_pdb(pdb_path: str) -> Optional[SimpleChain]:
    """Read a simple single-chain PDB with fixed-column slicing, matching Biopython's PDBParser.

    Accepts Unix-LF files with ATOM records only (in the first model), one chain, standard amino
    acids, no insertion codes, no revisited residues, and one C-alpha per residue (or C-alphas with
    distinct alternate-location labels resolved by Biopython's strictly-highest-occupancy rule).
    Returns None for anything else so the caller falls back to Biopython.
    """
    if not pdb_path.endswith('.pdb'):
        return None
    try:
        with open(pdb_path, 'rb') as fh:
            data = fh.read()
        if b'\r' in data:
            return None
        chain = None
        seen_model = False
        order, resname, ca = [], {}, {}
        ca_altlocs = {}
        best_occupancy = {}
        prev_resseq = None
        for line in data.split(b'\n'):
            record = line[:6]
            if record == b'ATOM  ':
                ch = line[21:22]
                if chain is None:
                    chain = ch
                elif ch != chain:
                    return None
                if line[26:27] != b' ':  # insertion code (or truncated line)
                    return None
                resseq = int(line[22:26])
                name = line[17:20]
                if resseq != prev_resseq:
                    if resseq in resname or name not in _3TO1_BYTES:
                        return None
                    resname[resseq] = name
                    order.append(resseq)
                    prev_resseq = resseq
                elif resname[resseq] != name:
                    return None
                atom = line[12:16]
                if atom.strip() == b'CA':
                    if atom != b' CA ':
                        return None
                    coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    altloc = line[16:17]
                    if altloc == b' ':
                        if resseq in ca:  # duplicate C-alpha or mixed with altlocs
                            return None
                        ca[resseq] = coord
                    else:
                        # Match Biopython's DisorderedAtom.disordered_add: select the first altloc
                        # with strictly highest occupancy (> -sys.maxsize).
                        seen = ca_altlocs.setdefault(resseq, set())
                        if altloc in seen or (resseq in ca and resseq not in best_occupancy):
                            return None
                        occupancy = float(line[54:60])
                        if not (occupancy > -sys.maxsize):
                            return None
                        seen.add(altloc)
                        if resseq not in best_occupancy or occupancy > best_occupancy[resseq]:
                            best_occupancy[resseq] = occupancy
                            ca[resseq] = coord
            elif record == b'HETATM':
                return None
            elif record == b'MODEL ':
                if seen_model or order:  # second model: only the first is read
                    break
                seen_model = True
            elif record in (b'ENDMDL', b'END   ', b'CONECT'):
                break
        if not order or len(ca) != len(order):
            return None
        residues = [Residue(i + 1, str(seq), _3TO1_BYTES[resname[seq]]) for i, seq in enumerate(order)]
        coords = np.array([ca[seq] for seq in order], dtype=np.float32)
        return SimpleChain(chain.decode(), residues, coords)
    except (ValueError, UnicodeDecodeError, OSError):
        return None


def load_chain(pdb_path: str, chain: Optional[str] = None):
    """Load `(chain_id, residues, dist_matrix)` using `read_simple_pdb` when possible, else Biopython."""
    simple = read_simple_pdb(pdb_path)
    if chain is None:
        LOG.warning(f"No chain specified for {pdb_path}, using first chain")
    if simple is not None and chain in (None, simple.chain_id):
        return simple.chain_id, simple.residues, distance_matrix(simple.ca_coords.astype(np.float16))
    model_structure = get_model_structure(pdb_path)
    if chain is None:
        chain = [c.id for c in model_structure.get_chains()][0]
    residues = get_model_structure_residues(model_structure, chain=chain)
    return chain, residues, get_distance(model_structure, chain=chain)


def inference_time_create_features(pdb_path, feature_config, chain="A", *,
                                   model_structure: Bio.PDB.Structure=None,
                                   renumber_pdbs=True, stride_path=constants.STRIDE_EXE,
                                   dist_matrix: np.ndarray = None,
                                   ):
    """dist_matrix: the C-alpha distance matrix, if already computed (then the structure is not parsed here)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        if pdb_path.endswith(".cif"):
            pdb_path = cif2pdb(pdb_path, os.path.join(tmpdir, "input.pdb"))
        if dist_matrix is None:
            if not model_structure:
                model_structure = get_model_structure(pdb_path)
            dist_matrix = get_distance(model_structure, chain=chain)
        if renumber_pdbs:
            output_pdb_path = os.path.join(tmpdir, "renum.pdb")
            renum_pdb_file(pdb_path, output_pdb_path)
        else:
            output_pdb_path = pdb_path
        ss_filepath = os.path.join(tmpdir, "ss.txt")
        calculate_ss(output_pdb_path, chain, stride_path, ssfile=ss_filepath)
        helix, strand = make_ss_matrix(ss_filepath, nres=dist_matrix.shape[-1])
    if feature_config['ss_bounds']:
        end_res_val = -1 if feature_config['negative_ss_end'] else 1
        helix_boundaries = make_boundary_matrix(helix, end_res_val=end_res_val)
        strand_boundaries = make_boundary_matrix(strand, end_res_val=end_res_val)
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


def distance_matrix(x):
    """Compute the distance matrix.

    Returns the matrix of all pair-wise distances.

    Parameters
    ----------
    x : (M, K) array_like
        Matrix of M vectors in K dimensions.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    Returns
    -------
    result : (M, M) ndarray
        Matrix containing the distance from every vector in `x` to every vector
        in `x`.
    """
    x1 = x[:, np.newaxis, :]  # Expand x to 3D for broadcasting, shape (M, 1, K)
    x2 = x[np.newaxis, :, :]  # Expand x to 3D for broadcasting, shape (1, M, K)

    distance_matrix = np.sum(np.abs(x2 - x1) ** 2, axis=-1) ** 0.5

    return distance_matrix.astype(np.float16)


def get_distance(structure_model: Bio.PDB.Structure, chain='A'):
    alpha_coords = np.array([residue['CA'].get_coord() for residue in \
                             structure_model[chain].get_residues() if Bio.PDB.is_aa(residue) and \
                             'CA' in residue and residue.get_resname() in _3to1], dtype=np.float16)
    x = distance_matrix(alpha_coords)
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
